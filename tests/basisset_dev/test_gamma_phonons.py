"""Gamma-point BIPOLE phonons + ZPE (M1).

The numerics that matter here -- the finite-difference stencil, the mass
weighting, the acoustic identification, the ZPE sum and its sign -- are
all separable from the SCF. So they are tested against a **scripted
energy surface** and against **analytic Hessians** whose frequencies are
known in closed form, at zero SCF cost. The one thing that cannot be
faked, agreement with the GAPW route on a real solid, is the milestone
gate and lives in ``examples/basisset_dev/`` (see
``handovers/HANDOVER_BASISOPT.md`` § M1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.basis_optimization.phonons import (
    DEFAULT_STEP_BOHR,
    HARTREE_TO_KJ_PER_MOL,
    GammaPhonons,
    apply_acoustic_sum_rule,
    compute_hessian_fd_energy,
    estimate_scf_count,
    phonons_from_hessian,
)
from vibeqc.periodic_gapw_phonon import (
    _AMU_TO_ELECTRON_MASS,
    _HARTREE_TO_CM_INV,
)

_CM_INV_TO_HARTREE = 1.0 / _HARTREE_TO_CM_INV


# ---------------------------------------------------------------------------
# Fixtures: a two-atom cubic cell, deliberately not a reference geometry.
# ---------------------------------------------------------------------------


def _two_atom_cell(Z1: int = 12, Z2: int = 8, a: float = 8.0):
    return vq.PeriodicSystem(
        3,
        np.eye(3) * a,
        [vq.Atom(Z1, [0.0, 0.0, 0.0]), vq.Atom(Z2, [a / 2, a / 2, a / 2])],
    )


def _spring_hessian(k: float) -> np.ndarray:
    """Force constants of two atoms joined by an isotropic spring.

    ``Phi = k * [[I, -I], [-I, I]]``. Translationally invariant by
    construction, so the acoustic sum rule holds exactly and the three
    acoustic modes are exactly zero. The other three are degenerate at
    ``omega = sqrt(k / mu)`` with ``mu`` the reduced mass -- a closed-form
    answer to check the mass weighting and the unit conversion against.
    """
    eye = np.eye(3)
    return k * np.block([[eye, -eye], [-eye, eye]])


def _expected_optical_cm1(k: float, m1_amu: float, m2_amu: float) -> float:
    m1 = m1_amu * _AMU_TO_ELECTRON_MASS
    m2 = m2_amu * _AMU_TO_ELECTRON_MASS
    mu = m1 * m2 / (m1 + m2)
    return math.sqrt(k / mu) * _HARTREE_TO_CM_INV


# ---------------------------------------------------------------------------
# Cost estimate -- quoted before a run starts, so it has to be right.
# ---------------------------------------------------------------------------


def test_energy_route_costs_18n2_plus_one_scf():
    # 1 central + 6N singles + 4 per unordered DOF pair.
    for n in (1, 2, 3, 8):
        n_dof = 3 * n
        expected = 1 + 2 * n_dof + 2 * n_dof * (n_dof - 1)
        assert estimate_scf_count(n, "energy") == expected
    # The number the handover quotes for a 2-atom primitive cell.
    assert estimate_scf_count(2, "energy") == 73


def test_analytic_force_route_costs_6n_scf():
    assert estimate_scf_count(2, "analytic_force") == 12
    assert estimate_scf_count(8, "analytic_force") == 48


def test_scf_count_rejects_unknown_mode_and_empty_cell():
    with pytest.raises(ValueError, match="hessian_mode"):
        estimate_scf_count(2, "magic")
    with pytest.raises(ValueError, match="n_atoms"):
        estimate_scf_count(0)


# ---------------------------------------------------------------------------
# The finite-difference stencil, against a scripted quadratic energy.
# ---------------------------------------------------------------------------


class _ScriptedResult:
    def __init__(self, energy):
        self.energy = energy
        self.converged = True
        self.n_iter = 3
        self.smearing_temperature = 0.0


def _install_quadratic_surface(monkeypatch, H_true, *, e0=-1.25, counter=None):
    """Make ``_run_bipole`` return ``E = e0 + 1/2 dx^T H dx``.

    Exact for the stencil, so any deviation the test sees is the
    implementation's, not the surface's.
    """
    import vibeqc.basis_optimization.phonons as ph

    reference = None

    def fake_run(system, basis, kmesh, options, *, method, functional, bipole_kwargs):
        nonlocal reference
        xyz = np.array(
            [list(atom.xyz) for atom in system.unit_cell], dtype=float
        ).reshape(-1)
        if reference is None:
            reference = xyz.copy()
        dx = xyz - reference
        if counter is not None:
            counter.append(1)
        return _ScriptedResult(e0 + 0.5 * float(dx @ H_true @ dx))

    monkeypatch.setattr(ph, "_run_bipole", fake_run)
    monkeypatch.setattr(ph, "BasisSet", lambda *a, **kw: object())


def test_fd_stencil_reproduces_a_known_quadratic_hessian(monkeypatch):
    rng = np.random.default_rng(20260803)
    A = rng.normal(size=(6, 6))
    H_true = 0.5 * (A + A.T) * 1e-2

    calls: list[int] = []
    _install_quadratic_surface(monkeypatch, H_true, counter=calls)

    system = _two_atom_cell()
    H, n_scf = compute_hessian_fd_energy(
        system, "sto-3g", None, None, step_bohr=0.02
    )

    # A quadratic surface makes the central difference exact to round-off.
    assert np.allclose(H, H_true, atol=1e-8)
    # And the sweep must actually cost what `estimate_scf_count` promises:
    # the cache is load-bearing, not an optimisation.
    assert n_scf == estimate_scf_count(2, "energy") == len(calls) == 73


def test_fd_refuses_to_difference_a_non_converged_scf(monkeypatch):
    import vibeqc.basis_optimization.phonons as ph

    class _Unconverged(_ScriptedResult):
        def __init__(self):
            super().__init__(-1.0)
            self.converged = False

    monkeypatch.setattr(
        ph, "_run_bipole", lambda *a, **kw: _Unconverged()
    )
    monkeypatch.setattr(ph, "BasisSet", lambda *a, **kw: object())

    with pytest.raises(RuntimeError, match="did not converge"):
        compute_hessian_fd_energy(_two_atom_cell(), "sto-3g", None, None)


def test_fd_uses_the_mermin_free_energy_when_smearing_is_on(monkeypatch):
    """A smeared KS run must be differenced on A = E - TS, not on E.

    The production FD force (``compute_bipole_gradient_fd``) differences
    the free energy because that is the variationally consistent
    quantity; a Hessian that differenced ``E`` instead would not describe
    the same surface as the forces the geometry was relaxed on.
    """
    import vibeqc.basis_optimization.phonons as ph

    class _Smeared(_ScriptedResult):
        def __init__(self, energy):
            super().__init__(energy)
            self.smearing_temperature = 0.01
            self.free_energy = 0.0  # deliberately flat

    monkeypatch.setattr(
        ph, "_run_bipole", lambda *a, **kw: _Smeared(np.random.random())
    )
    monkeypatch.setattr(ph, "BasisSet", lambda *a, **kw: object())

    H, _ = compute_hessian_fd_energy(_two_atom_cell(), "sto-3g", None, None)
    # Every point returns free_energy = 0, so a free-energy difference is
    # exactly zero while an E difference would be random noise.
    assert np.allclose(H, 0.0)


def test_fd_rejects_a_non_positive_step():
    with pytest.raises(ValueError, match="step_bohr"):
        compute_hessian_fd_energy(_two_atom_cell(), "sto-3g", None, None, step_bohr=0.0)


def test_fd_rejects_a_non_3d_cell():
    system = vq.PeriodicSystem(1, np.eye(3) * 8.0, [vq.Atom(1, [0.0, 0.0, 0.0])])
    with pytest.raises(NotImplementedError, match="3D"):
        compute_hessian_fd_energy(system, "sto-3g", None, None)


# ---------------------------------------------------------------------------
# Mass weighting, frequencies and the ZPE, against closed form.
# ---------------------------------------------------------------------------


def test_spring_model_reproduces_the_closed_form_optical_frequency():
    system = _two_atom_cell(12, 8)
    k = 0.05
    ph = phonons_from_hessian(_spring_hessian(k), system)

    masses = ph.masses_amu
    expected = _expected_optical_cm1(k, masses[0], masses[1])

    optical = np.sort(ph.optical_frequencies_cm1)
    assert optical.shape == (3,)
    assert np.allclose(optical, expected, rtol=1e-10)

    # Translational invariance is exact for this model, so the acoustic
    # residual must be at round-off, not merely "small". 1e-3 cm^-1 is the
    # round-off floor: eigh resolves omega^2 to ~eps*||D||, which for these
    # force constants is a few 1e-6 cm^-1 in omega. Four orders below the
    # 15 cm^-1 gate either way.
    assert max(abs(w) for w in ph.acoustic_residual_cm1) < 1e-3
    assert ph.n_acoustic_modes_excluded == 3
    assert ph.n_imaginary_modes_excluded == 0


def test_zero_point_energy_is_half_sum_hbar_omega_and_positive():
    system = _two_atom_cell(12, 8)
    k = 0.05
    ph = phonons_from_hessian(_spring_hessian(k), system)

    expected_cm1 = _expected_optical_cm1(k, ph.masses_amu[0], ph.masses_amu[1])
    # Three degenerate optical modes; the acoustic three carry no ZPE.
    expected_ha = 0.5 * 3.0 * expected_cm1 * _CM_INV_TO_HARTREE

    assert ph.zero_point_hartree == pytest.approx(expected_ha, rel=1e-10)
    assert ph.zero_point_hartree > 0.0
    assert ph.zero_point_kj_per_mol == pytest.approx(
        expected_ha * HARTREE_TO_KJ_PER_MOL, rel=1e-12
    )


def test_zero_point_divides_by_formula_units():
    system = _two_atom_cell(12, 8)
    ph = phonons_from_hessian(_spring_hessian(0.05), system)

    assert ph.zero_point_per_formula_unit(1) == ph.zero_point_hartree
    assert ph.zero_point_per_formula_unit(4) == pytest.approx(
        ph.zero_point_hartree / 4.0
    )
    assert ph.zero_point_kj_per_mol_per_formula_unit(4) == pytest.approx(
        ph.zero_point_hartree / 4.0 * HARTREE_TO_KJ_PER_MOL
    )
    with pytest.raises(ValueError, match="n_formula_units"):
        ph.zero_point_per_formula_unit(0)


def test_imaginary_modes_are_counted_and_excluded_not_discarded():
    """An imaginary mode is information: the geometry is not a minimum."""
    system = _two_atom_cell(12, 8)
    H = _spring_hessian(0.05)
    # Break one optical direction: make the x-x stretch a maximum.
    H[0, 0] -= 0.2
    H[3, 3] -= 0.2
    H[0, 3] += 0.2
    H[3, 0] += 0.2

    ph = phonons_from_hessian(H, system)
    assert ph.n_imaginary_modes_excluded == 1
    # Count against the 1 cm^-1 zero threshold, not against 0: an acoustic
    # mode lands at round-off and its sign there is a coin flip.
    assert (ph.frequencies_cm1 < -1.0).sum() == 1
    # ZPE counts only the two remaining real optical modes.
    real_optical = [
        w
        for i, w in enumerate(ph.frequencies_cm1)
        if i not in ph.acoustic_indices and w > 1.0
    ]
    assert len(real_optical) == 2
    assert ph.zero_point_hartree == pytest.approx(
        0.5 * sum(real_optical) * _CM_INV_TO_HARTREE, rel=1e-10
    )


def test_acoustic_modes_are_found_by_projection_not_by_lowest_frequency():
    """The lowest three frequencies are the wrong rule, and this proves it.

    Here an optical mode is imaginary and sorts *below* the acoustic
    residual. Picking "the three smallest" would drop the physical
    imaginary mode and keep a spurious acoustic one in the ZPE sum.
    """
    system = _two_atom_cell(12, 8)
    H = _spring_hessian(0.05)
    H[0, 0] -= 0.2
    H[3, 3] -= 0.2
    H[0, 3] += 0.2
    H[3, 0] += 0.2
    # A small translational-invariance violation, so the acoustic modes
    # come out at a nonzero (positive) residual.
    H[1, 1] += 2.0e-5

    ph = phonons_from_hessian(H, system)
    lowest_three = set(np.argsort(ph.frequencies_cm1)[:3].tolist())
    assert set(ph.acoustic_indices) != lowest_three
    # Each identified acoustic mode really is (almost) a pure translation.
    assert min(ph.acoustic_projection) > 0.9
    assert ph.n_imaginary_modes_excluded == 1


# ---------------------------------------------------------------------------
# The acoustic sum rule -- and the promise that we say which we did.
# ---------------------------------------------------------------------------


def test_asr_defaults_to_none_so_the_residual_is_reported_not_hidden():
    system = _two_atom_cell(12, 8)
    H = _spring_hessian(0.05)
    H[0, 0] += 5.0e-4  # break translational invariance

    ph = phonons_from_hessian(H, system)
    assert ph.asr == "none"
    assert max(abs(w) for w in ph.acoustic_residual_cm1) > 1.0
    assert np.allclose(ph.hessian, 0.5 * (H + H.T))


@pytest.mark.parametrize("mode", ["rowsum", "project"])
def test_asr_drives_the_acoustic_residual_to_zero(mode):
    system = _two_atom_cell(12, 8)
    H = _spring_hessian(0.05)
    H[0, 0] += 5.0e-4

    ph = phonons_from_hessian(H, system, asr=mode)
    assert ph.asr == mode
    assert max(abs(w) for w in ph.acoustic_residual_cm1) < 1e-3


def test_rowsum_asr_zeroes_the_row_sums_and_leaves_inter_atomic_constants():
    H = _spring_hessian(0.05)
    H[0, 0] += 5.0e-4
    H[4, 1] += 1.0e-4
    corrected = apply_acoustic_sum_rule(H, "rowsum")

    n_atoms = 2
    for i in range(n_atoms):
        for a in range(3):
            for b in range(3):
                total = sum(
                    corrected[3 * i + a, 3 * j + b] for j in range(n_atoms)
                )
                assert abs(total) < 1e-12
    # The off-diagonal blocks (the inter-atomic constants) are untouched
    # apart from the symmetrisation; only the self terms absorbed the
    # residual.
    assert corrected[0, 3] == pytest.approx(0.5 * (H[0, 3] + H[3, 0]))


def test_asr_none_is_the_identity_and_unknown_modes_raise():
    H = _spring_hessian(0.05)
    assert np.array_equal(apply_acoustic_sum_rule(H, "none"), H)
    with pytest.raises(ValueError, match="rowsum"):
        apply_acoustic_sum_rule(H, "average")
    with pytest.raises(ValueError, match=r"\(3N, 3N\)"):
        apply_acoustic_sum_rule(np.zeros((4, 4)), "rowsum")


def test_asr_does_not_mutate_the_caller_matrix():
    H = _spring_hessian(0.05)
    H[0, 0] += 1.0e-3
    before = H.copy()
    apply_acoustic_sum_rule(H, "rowsum")
    assert np.array_equal(H, before)


# ---------------------------------------------------------------------------
# Result plumbing
# ---------------------------------------------------------------------------


def test_result_records_the_settings_that_produced_it():
    system = _two_atom_cell(12, 8)
    ph = phonons_from_hessian(
        _spring_hessian(0.05),
        system,
        fd_step_bohr=0.02,
        hessian_mode="energy",
        asr="none",
        n_scf=73,
        detail={"method": "RKS", "functional": "pbe"},
    )
    assert isinstance(ph, GammaPhonons)
    assert ph.fd_step_bohr == 0.02
    assert ph.hessian_mode == "energy"
    assert ph.n_scf == 73
    assert ph.detail["functional"] == "pbe"
    text = ph.summary(n_formula_units=1)
    assert "ZPE" in text and "acoustic" in text
    assert "kJ/mol per formula unit" in text


def test_default_step_is_larger_than_the_gapw_force_step():
    """Documented, not incidental: an energy Hessian divides by h^2."""
    from vibeqc.periodic_gapw_phonon import compute_dynamical_matrix_fd  # noqa: F401

    assert DEFAULT_STEP_BOHR == 0.02


def test_phonons_from_hessian_validates_the_shape():
    system = _two_atom_cell(12, 8)
    with pytest.raises(ValueError, match="expected a "):
        phonons_from_hessian(np.zeros((3, 3)), system)
