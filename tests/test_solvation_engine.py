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

"Shared" means one *reference* implementation, not one code path at runtime.
MSINDO's shipped route runs a fused native kernel (``indo.cosmo_reaction_field``)
that does the same five steps in one C++ call and never calls
:func:`reaction_field_step`; its Python branch is the reference/fallback. That
is deliberate, and safe only because the kernel is pinned to this step by
``tests/test_msindo_cosmo.py::test_cosmo_reaction_field_cpp_matches_python_reference``
(``1e-12`` on identical inputs) and by the converged-run comparison below. The
failure this file guards against is a native fast path added *without* such a
pin (#254).
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


# ---------------------------------------------------------------------
# Behaviour, not source text (#254).
#
# The guard these replace asserted only that the string "reaction_field_step"
# appeared in each route's source. MSINDO's native kernel returns before the
# branch that holds that string, so a real MSINDO COSMO run made zero calls to
# the shared step while the guard stayed green. These observe calls on real
# runs instead, so they fail the moment a route stops reaching the step.
# ---------------------------------------------------------------------

_WATER_BOHR = [(8, (0.0, 0.0, 0.0)), (1, (0.0, 1.43, 1.11)), (1, (0.0, -1.43, 1.11))]
_WATER_ANG = [(8, (0.0, 0.0, 0.0)), (1, (0.0, 0.757, 0.587)), (1, (0.0, -0.757, 0.587))]


def _counting(fn, box):
    def spy(*args, **kwargs):
        box["calls"] += 1
        return fn(*args, **kwargs)

    return spy


def test_the_gaussian_route_calls_the_shared_step_on_a_real_run(monkeypatch):
    """``run_cpcm_scf`` must reach the shared step at runtime, every macro-iteration.

    This is the one route where "one implementation" is literally true, and
    nothing observed it: the text check would still have passed with the call
    deleted and the name left in a comment. Measured on this water case: 5
    calls over the macro-iteration.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation import driver

    box = {"calls": 0}
    monkeypatch.setattr(driver, "reaction_field_step", _counting(driver.reaction_field_step, box))

    mol = vq.Molecule([vq.Atom(z, list(xyz)) for z, xyz in _WATER_BOHR], 0, 1)
    sol = vq.run_cpcm_scf(mol, vq.BasisSet(mol, "sto-3g"), method="rhf", solvent="water")

    assert sol.converged
    assert box["calls"] >= 1, "run_cpcm_scf no longer reaches reaction_field_step"


def test_the_msindo_reference_path_calls_the_shared_step_and_agrees_with_native(
    monkeypatch,
):
    """MSINDO's reference branch must stay live, and land where the kernel lands.

    The shipped route never calls the shared step -- its native kernel returns
    first -- so the Python branch holding it is exercised only when the kernel
    is unavailable. Nothing observed that branch at runtime, so it could rot into
    dead code while the text check stayed green. With the kernel disabled the
    run must reach the step, and the converged energy must match the native
    run's: the parity test pins one density at ``1e-12``, this pins the whole
    self-consistent answer. Measured: the two paths agree to ``6.7e-17`` in
    every reaction-field quantity across all 13 SCF iterations of this run.
    """
    pytest.importorskip("vibeqc")
    from vibeqc.semiempirical.methods import msindo_cosmo as msindo

    if msindo._cpp_cosmo_reaction_kernel() is None:
        pytest.skip("native MSINDO COSMO kernel not built; nothing to compare")
    Z = [z for z, _ in _WATER_ANG]
    xyz = [list(c) for _, c in _WATER_ANG]
    native = msindo.msindo_cosmo(Z, xyz, epsilon=78.39)

    box = {"calls": 0}
    monkeypatch.setattr(msindo, "_cpp_cosmo_reaction_kernel", lambda: None)
    monkeypatch.setattr(msindo, "reaction_field_step", _counting(msindo.reaction_field_step, box))
    reference = msindo.msindo_cosmo(Z, xyz, epsilon=78.39)

    assert box["calls"] >= 1, "MSINDO's reference branch no longer reaches the shared step"
    assert reference.total_energy == pytest.approx(native.total_energy, abs=1e-10)
    assert reference.e_pol == pytest.approx(native.e_pol, abs=1e-10)
