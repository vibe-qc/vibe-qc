"""The reaction-field step is shared by every method (issue #554).

Before this, the five lines that turn a solute density into screened surface
charges, a Hamiltonian contribution and an energy split existed twice: inside
``driver.run_cpcm_scf``'s macro-iteration for Gaussian HF/DFT, and inside
``msindo_cosmo._cosmo_reaction_field`` for MSINDO, against the same provider
protocol. Two copies of one equation is the same failure shape as two copies
of one constant (#546, #548), one level up.

What stays separate is the outer convergence strategy: Gaussian macro-iterates
an outer ``q`` loop around a full inner SCF; MSINDO folds the field into a
single SCF at every Fock build. Those are different, legitimate strategies
over the same step.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc.solvation.engine import (
    ReactionField,
    lu_cavity_solve,
    reaction_field_step,
    solve_screened_charges,
)
from vibeqc.solvation.screening import ScreeningModel


class _StubProvider:
    """Minimal SolutePotentialProvider: the step must not need more."""

    def __init__(self, esp, fock_scale=2.0):
        self._esp = np.asarray(esp, dtype=np.float64)
        self._fock_scale = fock_scale
        self.esp_calls = 0
        self.fock_calls = 0

    def esp_at_cavity(self, density):
        self.esp_calls += 1
        return self._esp * float(np.trace(np.atleast_2d(density)))

    def fock_contribution(self, charges):
        self.fock_calls += 1
        return self._fock_scale * np.outer(charges[:2], charges[:2])


def _cavity(n=5, seed=7):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    A = 0.5 * (A + A.T) + np.eye(n) * (n + 5.0)
    return A, rng.normal(size=n), rng.normal(size=n)


def test_step_matches_the_closed_form_it_replaces():
    """q = -f A^-1 (V_elec + V_core), e_pol = 1/2 q.V_total, exactly."""
    A, esp, V_core = _cavity()
    provider = _StubProvider(esp)
    screening = ScreeningModel.from_variant(2.27, "cosmo")
    density = np.eye(3)

    field = reaction_field_step(
        provider, lu_cavity_solve(A), V_core, density, screening
    )

    V_elec_ref = esp * float(np.trace(density))
    V_tot_ref = V_elec_ref + V_core
    q_ref = np.linalg.solve(A, -screening.f * V_tot_ref)

    np.testing.assert_allclose(field.V_elec, V_elec_ref, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(field.V_total, V_tot_ref, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(field.q, q_ref, rtol=1e-12, atol=1e-14)
    assert field.e_pol == pytest.approx(0.5 * float(q_ref @ V_tot_ref), rel=1e-12)
    assert field.e_core_share == pytest.approx(
        0.5 * float(q_ref @ V_core), rel=1e-12
    )


def test_energy_split_is_exhaustive():
    """The core and electronic shares must sum to the polarisation energy.

    The split is not decoration: MSINDO's single-SCF bookkeeping adds only the
    core share, because the electronic half is already counted through the
    density's trace against the modified Hamiltonian. If these stopped summing,
    that route would silently double- or half-count.
    """
    A, esp, V_core = _cavity(seed=11)
    field = reaction_field_step(
        _StubProvider(esp), lu_cavity_solve(A), V_core, np.eye(2) * 1.5,
        ScreeningModel.from_variant(78.39, "cpcm"),
    )
    assert field.e_core_share + field.e_elec_share == pytest.approx(
        field.e_pol, rel=1e-14
    )
    assert field.e_elec_share == pytest.approx(
        0.5 * float(field.q @ field.V_elec), rel=1e-12
    )


def test_charges_scale_linearly_with_the_screening_factor():
    """q = f q0, so the variant only rescales the conductor solution."""
    A, esp, V_core = _cavity(seed=3)
    density = np.eye(2)
    solve = lu_cavity_solve(A)
    fields = {
        v: reaction_field_step(
            _StubProvider(esp), solve, V_core, density,
            ScreeningModel.from_variant(2.27, v),
        )
        for v in ("cpcm", "cosmo")
    }
    ratio = fields["cosmo"].screening.f / fields["cpcm"].screening.f
    np.testing.assert_allclose(
        fields["cosmo"].q, ratio * fields["cpcm"].q, rtol=1e-12, atol=1e-15
    )


def test_with_charges_reuses_the_potentials_and_skips_the_esp():
    """The accelerator seam must not pay for a second ESP pass.

    The ESP is one provider pass over every segment and dominates the step;
    re-solving it to apply an extrapolated ``q`` would make q-DIIS cost more
    than it saves, which is why the Gaussian driver used to rebuild the
    downstream quantities by hand instead. Doing that by hand is where a
    convention drifts, so the seam has to be both shared and cheap.
    """
    A, esp, V_core = _cavity(seed=5)
    provider = _StubProvider(esp)
    screening = ScreeningModel.from_variant(35.94, "cosmo")
    field = reaction_field_step(
        provider, lu_cavity_solve(A), V_core, np.eye(2), screening
    )
    assert provider.esp_calls == 1 and provider.fock_calls == 1

    q_new = field.q * 1.37
    accelerated = field.with_charges(q_new, provider)

    assert provider.esp_calls == 1, "with_charges recomputed the ESP"
    assert provider.fock_calls == 2
    np.testing.assert_array_equal(accelerated.V_total, field.V_total)
    np.testing.assert_array_equal(accelerated.V_elec, field.V_elec)
    np.testing.assert_array_equal(accelerated.q, q_new)
    assert accelerated.e_pol == pytest.approx(
        0.5 * float(q_new @ field.V_total), rel=1e-14
    )
    assert accelerated.screening is field.screening


def test_lu_solve_matches_a_dense_solve():
    """The injected solve must be exact: an approximate A here would make the
    energy and its derivative disagree."""
    A, _, _ = _cavity(seed=13)
    rhs = np.arange(1.0, A.shape[0] + 1.0)
    np.testing.assert_allclose(
        lu_cavity_solve(A)(rhs), np.linalg.solve(A, rhs), rtol=1e-12, atol=1e-14
    )


def test_solve_screened_charges_applies_the_carried_factor():
    A, _, _ = _cavity(seed=17)
    V = np.linspace(-1.0, 1.0, A.shape[0])
    screening = ScreeningModel.from_variant(4.71, "cosmo")
    np.testing.assert_allclose(
        solve_screened_charges(lu_cavity_solve(A), V, screening),
        np.linalg.solve(A, -screening.f * V),
        rtol=1e-12, atol=1e-14,
    )


def test_step_rejects_a_mismatched_core_potential():
    A, esp, _ = _cavity(seed=19)
    with pytest.raises(ValueError, match="segments"):
        reaction_field_step(
            _StubProvider(esp), lu_cavity_solve(A), np.zeros(3), np.eye(2),
            ScreeningModel.from_variant(78.39, "cpcm"),
        )


def test_with_charges_rejects_a_mismatched_charge_vector():
    A, esp, V_core = _cavity(seed=23)
    provider = _StubProvider(esp)
    field = reaction_field_step(
        provider, lu_cavity_solve(A), V_core, np.eye(2),
        ScreeningModel.from_variant(78.39, "cpcm"),
    )
    with pytest.raises(ValueError, match="segments"):
        field.with_charges(np.zeros(2), provider)


def test_step_fails_loudly_on_a_singular_cavity():
    """Silence here would propagate NaN charges into the Fock matrix."""
    n = 4
    A = np.zeros((n, n))
    provider = _StubProvider(np.ones(n))
    with pytest.raises((RuntimeError, np.linalg.LinAlgError)):
        reaction_field_step(
            provider, lambda rhs: np.linalg.solve(A, rhs), np.ones(n),
            np.eye(2), ScreeningModel.from_variant(78.39, "cpcm"),
        )


def test_both_shipped_routes_use_the_shared_step():
    """Guard against a third copy reappearing.

    The point of #554 is that there is one implementation. If a route stops
    importing it, this fails rather than the duplication being rediscovered by
    a divergence years later.
    """
    import inspect

    from vibeqc.semiempirical.methods import msindo_cosmo as msindo
    from vibeqc.solvation import driver

    assert "reaction_field_step" in inspect.getsource(driver.run_cpcm_scf)
    assert "reaction_field_step" in inspect.getsource(
        msindo._cosmo_reaction_field
    )
    assert isinstance(
        msindo._cosmo_reaction_field.__doc__, str
    ) and "ReactionField" in msindo._cosmo_reaction_field.__doc__
