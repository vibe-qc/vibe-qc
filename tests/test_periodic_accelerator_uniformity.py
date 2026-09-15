"""Periodic SCF — accelerator + dynamic-damping option-surface uniformity.

Verifies that the same SCF-accelerator + dynamic-damping selection
surface available on the molecular options classes (RHFOptions /
UHFOptions / RKSOptions / UKSOptions) is also available on the three
periodic options classes (PeriodicRHFOptions / PeriodicSCFOptions /
PeriodicKSOptions), and that the Γ-point Python Ewald backends
(RHF / RKS / UHF / UKS) wire the full accelerator family + dynamic
damping end-to-end.

Coverage status (M1 — Γ-Ewald):
  * periodic_rhf.cpp (Γ-only RHF) — KDIIS / EDIIS / EDIIS+DIIS /
    dynamic_damping all dispatch correctly (C++ kernel).
  * periodic_scf.cpp (multi-k RHF + RKS) — DIIS / EDIIS / EDIIS+DIIS /
    ADIIS only; KDIIS multi-k deferred (per-k MO-basis design).
  * Python Γ-Ewald (this module's end-to-end tests) — full family
    + dynamic_damping wired via :mod:`vibeqc.periodic_scf_accelerators`.
  * Python multi-k Ewald / BIPOLE / GDF — DIIS-only today (M2/M3/M4
    in the accelerator-rollout plan; the
    ``_reject_unsupported_python_accelerator`` helper still gates them).
"""

from __future__ import annotations

import pytest

from vibeqc import (
    PeriodicRHFOptions,
    PeriodicSCFOptions,
    PeriodicKSOptions,
    SCFAccelerator,
)


_PERIODIC_OPTION_CLASSES = (
    PeriodicRHFOptions,
    PeriodicSCFOptions,
    PeriodicKSOptions,
)


# ---------------------------------------------------------------------------
# scf_accelerator — selectable on every periodic options class.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("OptionsCls", _PERIODIC_OPTION_CLASSES)
def test_periodic_scf_accelerator_default_is_ediis_diis(OptionsCls):
    """All periodic options default to the EDIIS_DIIS production hybrid
    (v0.15.x; was DIIS)."""
    assert OptionsCls().scf_accelerator == SCFAccelerator.EDIIS_DIIS


@pytest.mark.parametrize("OptionsCls", _PERIODIC_OPTION_CLASSES)
@pytest.mark.parametrize("accel", [
    SCFAccelerator.DIIS,
    SCFAccelerator.KDIIS,
    SCFAccelerator.EDIIS,
    SCFAccelerator.EDIIS_DIIS,
])
def test_periodic_scf_accelerator_field_is_settable(OptionsCls, accel):
    """Every accelerator-family choice round-trips on every periodic options."""
    o = OptionsCls()
    o.scf_accelerator = accel
    assert o.scf_accelerator == accel


@pytest.mark.parametrize("OptionsCls", _PERIODIC_OPTION_CLASSES)
def test_periodic_ediis_diis_switch_threshold_default(OptionsCls):
    """Default threshold matches the molecular default (PySCF convention)."""
    assert OptionsCls().ediis_diis_switch_threshold == pytest.approx(1e-1)


# ---------------------------------------------------------------------------
# Dynamic damping — selectable on every periodic options class.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("OptionsCls", _PERIODIC_OPTION_CLASSES)
def test_periodic_dynamic_damping_default_is_true(OptionsCls):
    # ORCA/CRYSTAL-style adaptive damping on by default for periodic SCF
    # (v0.15.x); charge-sloshing-prone systems benefit.
    assert OptionsCls().dynamic_damping is True


@pytest.mark.parametrize("OptionsCls", _PERIODIC_OPTION_CLASSES)
def test_periodic_dynamic_damping_bounds_defaults(OptionsCls):
    o = OptionsCls()
    assert o.dynamic_damping_min == pytest.approx(0.0)
    assert o.dynamic_damping_max == pytest.approx(0.95)


@pytest.mark.parametrize("OptionsCls", _PERIODIC_OPTION_CLASSES)
def test_periodic_dynamic_damping_fields_are_settable(OptionsCls):
    o = OptionsCls()
    o.dynamic_damping = True
    o.dynamic_damping_min = 0.1
    o.dynamic_damping_max = 0.9
    assert o.dynamic_damping is True
    assert o.dynamic_damping_min == pytest.approx(0.1)
    assert o.dynamic_damping_max == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Python periodic backends — full {DIIS, KDIIS, EDIIS, EDIIS_DIIS, ADIIS}
# family + dynamic damping are now wired on every Python periodic backend
# (Ewald M1/M2, BIPOLE M3, GDF M4). The legacy
# ``_reject_unsupported_python_accelerator`` guard + its rejection tests
# were retired in M4 — the end-to-end convergence tests below (which
# actually run each accelerator through an SCF) are the live coverage.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# End-to-end SCF runs — Γ-Ewald RHF / RKS / UHF / UKS converge to the
# same energy regardless of which accelerator from the
# {DIIS, KDIIS, EDIIS, EDIIS_DIIS, ADIIS} family is selected, and the
# dynamic_damping option is honoured (no NotImplementedError).
#
# An accelerator is a convergence aid, not a solver change — different
# choices must reach the same fixed point to within the convergence
# threshold. The reference value comes from the default DIIS run; the
# other accelerators are checked against it.
# ---------------------------------------------------------------------------

import numpy as np

import vibeqc as vq


_ENERGY_TOL = 5e-6  # Ha; loose enough for KDIIS's slightly different
                     # asymptotic convergence path on a small system.

_ACCELERATORS = [
    SCFAccelerator.DIIS,
    SCFAccelerator.KDIIS,
    SCFAccelerator.EDIIS,
    SCFAccelerator.EDIIS_DIIS,
    SCFAccelerator.ADIIS,
    SCFAccelerator.ADIIS_DIIS,
]


def _h2_in_box(box: float = 20.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h_atom_in_box(box: float = 20.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c])],
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _rhf_opts(accel: SCFAccelerator) -> "vq.PeriodicRHFOptions":
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 60
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    opts.scf_accelerator = accel
    return opts


def _ks_opts(accel: SCFAccelerator) -> "vq.PeriodicKSOptions":
    opts = vq.PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 60
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    opts.scf_accelerator = accel
    return opts


@pytest.fixture(scope="module")
def _rhf_reference():
    """Reference DIIS energy + system for the closed-shell Γ-Ewald tests."""
    sysp, basis = _h2_in_box()
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, _rhf_opts(SCFAccelerator.DIIS),
        omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged, "reference DIIS RHF must converge"
    return sysp, basis, r.energy


@pytest.mark.parametrize("accel", _ACCELERATORS)
def test_rhf_ewald_accelerator_converges_to_same_energy(_rhf_reference, accel):
    """Every accelerator must reach the reference DIIS energy on H₂/STO-3G."""
    sysp, basis, e_ref = _rhf_reference
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, _rhf_opts(accel),
        omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged, f"{accel.name} did not converge"
    assert r.energy == pytest.approx(e_ref, abs=_ENERGY_TOL), (
        f"{accel.name} energy {r.energy} disagrees with DIIS reference "
        f"{e_ref} beyond {_ENERGY_TOL} Ha"
    )


@pytest.mark.parametrize("accel", _ACCELERATORS)
def test_rks_ewald_accelerator_converges_to_same_energy(accel):
    """RKS/LDA Γ-Ewald: same fixed point across the accelerator family."""
    sysp, basis = _h2_in_box()
    opts = _ks_opts(SCFAccelerator.DIIS)
    opts.functional = "svwn"
    r_ref = vq.run_rks_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.4,
    )
    assert r_ref.converged
    opts_a = _ks_opts(accel)
    opts_a.functional = "svwn"
    r = vq.run_rks_periodic_gamma_ewald3d(
        sysp, basis, opts_a, omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged, f"{accel.name} did not converge"
    assert r.energy == pytest.approx(r_ref.energy, abs=_ENERGY_TOL)


@pytest.mark.parametrize("accel", _ACCELERATORS)
def test_uhf_ewald_accelerator_converges_to_same_energy(accel):
    """UHF Γ-Ewald: H-atom doublet, same fixed point across accelerators."""
    sysp, basis = _h_atom_in_box()
    r_ref = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, _rhf_opts(SCFAccelerator.DIIS),
        omega=0.5, spacing_bohr=0.4,
    )
    assert r_ref.converged
    r = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, _rhf_opts(accel),
        omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged, f"{accel.name} did not converge"
    assert r.energy == pytest.approx(r_ref.energy, abs=_ENERGY_TOL)


@pytest.mark.parametrize("accel", _ACCELERATORS)
def test_uks_ewald_accelerator_converges_to_same_energy(accel):
    """UKS/LDA Γ-Ewald: H-atom doublet, same fixed point across accelerators."""
    sysp, basis = _h_atom_in_box()
    opts_ref = _ks_opts(SCFAccelerator.DIIS)
    opts_ref.functional = "svwn"
    r_ref = vq.run_uks_periodic_gamma_ewald3d(
        sysp, basis, opts_ref, omega=0.5, spacing_bohr=0.4,
    )
    assert r_ref.converged
    opts_a = _ks_opts(accel)
    opts_a.functional = "svwn"
    r = vq.run_uks_periodic_gamma_ewald3d(
        sysp, basis, opts_a, omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged, f"{accel.name} did not converge"
    assert r.energy == pytest.approx(r_ref.energy, abs=_ENERGY_TOL)


# ---------------------------------------------------------------------------
# Dynamic damping — selecting it must not raise and must still converge
# to the same energy. Smoke-test on the closed-shell RHF driver only;
# the helper class is shared across all 4 backends.
# ---------------------------------------------------------------------------

def test_rhf_ewald_dynamic_damping_smoke(_rhf_reference):
    sysp, basis, e_ref = _rhf_reference
    opts = _rhf_opts(SCFAccelerator.DIIS)
    opts.damping = 0.3
    opts.dynamic_damping = True
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged, "dynamic_damping run did not converge"
    assert r.energy == pytest.approx(e_ref, abs=_ENERGY_TOL)


# ---------------------------------------------------------------------------
# End-to-end multi-k SCF runs — Python multi-k Ewald drivers (RHF / RKS /
# UHF) honour the full accelerator family on a non-trivial k-mesh.
#
# The reference value comes from the DIIS run with the same k-mesh; other
# accelerators must converge to the same energy. KDIIS multi-k uses the
# per-k orbital-rotation-gradient design introduced in M2c; EDIIS / ADIIS
# / EDIIS_DIIS bridge through the per-cell representation (M2b) and call
# into the C++ block-vector kernels (M2a).
#
# Unlike the Γ-Ewald driver (where commit ``49f8ae91`` silently
# overrides the ``omega`` kwarg via a getattr-with-auto-default), the
# multi-k drivers honour ``omega`` directly — so these tests pin the
# accelerator convergence at a fixed ``omega = 0.5`` and the
# reference / extrapolated runs see the same Coulomb split.
# ---------------------------------------------------------------------------


# Multi-k accelerator family wired end-to-end on the Python multi-k
# Ewald drivers: DIIS + KDIIS run natively per-k (Pulay /
# orbital-rotation-gradient, landed M2c); EDIIS / ADIIS / EDIIS_DIIS
# bridge per-k Hermitian to stacked-real blocks via the √w_k
# `per_k_to_stacked_real_blocks` helper and call into the C++
# block-vector kernel (landed M2e). The stacked bridge makes the C++
# kernel's sum-of-Frobenius bilinear form match the periodic per-k
# energy bilinear form ``Σ_k w_k Re Tr[F(k) D(k)]`` — see the
# "per-k ↔ stacked-real-block bridge" comment in
# :mod:`vibeqc.periodic_scf_accelerators` for the derivation.
_MULTIK_ACCELERATORS = [
    SCFAccelerator.DIIS,
    SCFAccelerator.KDIIS,
    SCFAccelerator.EDIIS,
    SCFAccelerator.EDIIS_DIIS,
    SCFAccelerator.ADIIS,
    SCFAccelerator.ADIIS_DIIS,
    # Adaptive-depth commutator-DIIS (Chupin et al. 2021); per-k weighted
    # depth policy on _MultiKPulayDIIS.
    SCFAccelerator.R_CDIIS,
    SCFAccelerator.AD_CDIIS,
]


def _h2_chain():
    sysp = vq.PeriodicSystem(
        1, np.diag([6.0, 30.0, 30.0]),
        [vq.Atom(1, [0.0, 15.0, 15.0]),
         vq.Atom(1, [1.4, 15.0, 15.0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h_atom_1d():
    sysp = vq.PeriodicSystem(
        1, np.diag([10.0, 30.0, 30.0]),
        [vq.Atom(1, [5.0, 15.0, 15.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _multi_k_rhf_opts(accel) -> "vq.PeriodicRHFOptions":
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    # EDIIS / ADIIS are asymptotically slower than DIIS for tight
    # gradient thresholds (no quadratic-information correction past
    # the convex-hull guess); allow extra iterations rather than
    # loosen ``conv_tol_grad``. EDIIS_DIIS hybrid converges in the
    # DIIS-arm budget once the commutator-error norm drops below
    # ``ediis_diis_switch_threshold``.
    opts.max_iter = 200
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    opts.scf_accelerator = accel
    return opts


def _multi_k_ks_opts(accel) -> "vq.PeriodicKSOptions":
    opts = vq.PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    # EDIIS / ADIIS are asymptotically slower than DIIS for tight
    # gradient thresholds (no quadratic-information correction past
    # the convex-hull guess); allow extra iterations rather than
    # loosen ``conv_tol_grad``. EDIIS_DIIS hybrid converges in the
    # DIIS-arm budget once the commutator-error norm drops below
    # ``ediis_diis_switch_threshold``.
    opts.max_iter = 200
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    opts.scf_accelerator = accel
    return opts


@pytest.fixture(scope="module")
def _multi_k_rhf_reference():
    sysp, basis = _h2_chain()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    r = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_rhf_opts(SCFAccelerator.DIIS),
        omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged, "reference multi-k DIIS RHF must converge"
    return sysp, basis, km, r.energy


@pytest.mark.parametrize("accel", _MULTIK_ACCELERATORS)
def test_multi_k_rhf_accelerator_converges_to_same_energy(
    _multi_k_rhf_reference, accel,
):
    sysp, basis, km, e_ref = _multi_k_rhf_reference
    r = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_rhf_opts(accel),
        omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged, f"multi-k RHF {accel.name} did not converge"
    assert r.energy == pytest.approx(e_ref, abs=_ENERGY_TOL), (
        f"multi-k RHF {accel.name} energy {r.energy} disagrees with "
        f"DIIS reference {e_ref} beyond {_ENERGY_TOL} Ha"
    )


@pytest.mark.parametrize("accel", _MULTIK_ACCELERATORS)
def test_multi_k_rks_accelerator_converges_to_same_energy(accel):
    sysp, basis = _h2_chain()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    opts_ref = _multi_k_ks_opts(SCFAccelerator.DIIS)
    opts_ref.functional = "svwn"
    r_ref = vq.run_rks_periodic_multi_k_ewald3d(
        sysp, basis, km, opts_ref, omega=0.5, spacing_bohr=0.4,
    )
    assert r_ref.converged
    opts_a = _multi_k_ks_opts(accel)
    opts_a.functional = "svwn"
    r = vq.run_rks_periodic_multi_k_ewald3d(
        sysp, basis, km, opts_a, omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged, f"multi-k RKS {accel.name} did not converge"
    assert r.energy == pytest.approx(r_ref.energy, abs=_ENERGY_TOL)


@pytest.mark.parametrize("accel", _MULTIK_ACCELERATORS)
def test_multi_k_uhf_accelerator_converges_to_same_energy(accel):
    sysp, basis = _h_atom_1d()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    r_ref = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_rhf_opts(SCFAccelerator.DIIS),
        omega=0.5, spacing_bohr=0.4,
    )
    assert r_ref.converged
    r = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_rhf_opts(accel),
        omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged, f"multi-k UHF {accel.name} did not converge"
    assert r.energy == pytest.approx(r_ref.energy, abs=_ENERGY_TOL)


def test_multi_k_rhf_dynamic_damping_smoke(_multi_k_rhf_reference):
    """Dynamic damping must not raise on the multi-k Ewald RHF path
    and must still converge to the same energy."""
    sysp, basis, km, e_ref = _multi_k_rhf_reference
    opts = _multi_k_rhf_opts(SCFAccelerator.DIIS)
    opts.damping = 0.3
    opts.dynamic_damping = True
    r = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged, "multi-k dynamic_damping run did not converge"
    assert r.energy == pytest.approx(e_ref, abs=_ENERGY_TOL)


# ---------------------------------------------------------------------------
# End-to-end BIPOLE SCF runs — RHF / RKS / UHF / UKS honour the full
# accelerator family + dynamic_damping end to end through the same
# :class:`MultiKPeriodicSCFAccelerator` / :class:`MultiKPeriodicUHFAccelerator`
# helper that the multi-k Ewald drivers use (so EDIIS / ADIIS /
# EDIIS_DIIS go through the stacked-real-block bridge landed in M2e,
# while DIIS / KDIIS run natively per-k).
# ---------------------------------------------------------------------------


def _h2_3d_box():
    """3D-periodic H₂ in a vacuum box. Even electron count; suitable
    for closed-shell parity tests."""
    sysp = vq.PeriodicSystem(
        3, np.diag([20.0, 20.0, 20.0]),
        [vq.Atom(1, [10.0, 10.0, 10.0 - 0.7]),
         vq.Atom(1, [10.0, 10.0, 10.0 + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h_atom_3d_box():
    """3D-periodic H atom in a vacuum box. Open-shell doublet."""
    sysp = vq.PeriodicSystem(
        3, np.diag([20.0, 20.0, 20.0]),
        [vq.Atom(1, [10.0, 10.0, 10.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _bipole_rhf_opts(accel) -> "vq.PeriodicRHFOptions":
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    # EDIIS / ADIIS are asymptotically slower than DIIS for tight
    # gradient thresholds (no quadratic-information correction past
    # the convex-hull guess); allow extra iterations rather than
    # loosen ``conv_tol_grad``. EDIIS_DIIS hybrid converges in the
    # DIIS-arm budget once the commutator-error norm drops below
    # ``ediis_diis_switch_threshold``.
    opts.max_iter = 200
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    opts.scf_accelerator = accel
    return opts


def _bipole_ks_opts(accel) -> "vq.PeriodicKSOptions":
    opts = vq.PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    # EDIIS / ADIIS are asymptotically slower than DIIS for tight
    # gradient thresholds (no quadratic-information correction past
    # the convex-hull guess); allow extra iterations rather than
    # loosen ``conv_tol_grad``. EDIIS_DIIS hybrid converges in the
    # DIIS-arm budget once the commutator-error norm drops below
    # ``ediis_diis_switch_threshold``.
    opts.max_iter = 200
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    opts.scf_accelerator = accel
    return opts


@pytest.fixture(scope="module")
def _bipole_rhf_reference():
    sysp, basis = _h2_3d_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_pbc_bipole_rhf(
        sysp, basis, km, _bipole_rhf_opts(SCFAccelerator.DIIS),
    )
    assert r.converged, "reference BIPOLE DIIS RHF must converge"
    return sysp, basis, km, r.energy


@pytest.mark.parametrize("accel", _MULTIK_ACCELERATORS)
def test_bipole_rhf_accelerator_converges_to_same_energy(
    _bipole_rhf_reference, accel,
):
    sysp, basis, km, e_ref = _bipole_rhf_reference
    r = vq.run_pbc_bipole_rhf(
        sysp, basis, km, _bipole_rhf_opts(accel),
    )
    assert r.converged, f"BIPOLE RHF {accel.name} did not converge"
    assert r.energy == pytest.approx(e_ref, abs=_ENERGY_TOL)


@pytest.mark.parametrize("accel", _MULTIK_ACCELERATORS)
def test_bipole_rks_accelerator_converges_to_same_energy(accel):
    sysp, basis = _h2_3d_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts_ref = _bipole_ks_opts(SCFAccelerator.DIIS)
    opts_ref.functional = "svwn"
    r_ref = vq.run_pbc_bipole_rks(sysp, basis, km, opts_ref)
    assert r_ref.converged
    opts_a = _bipole_ks_opts(accel)
    opts_a.functional = "svwn"
    r = vq.run_pbc_bipole_rks(sysp, basis, km, opts_a)
    assert r.converged, f"BIPOLE RKS {accel.name} did not converge"
    assert r.energy == pytest.approx(r_ref.energy, abs=_ENERGY_TOL)


@pytest.mark.parametrize("accel", _MULTIK_ACCELERATORS)
def test_bipole_uhf_accelerator_converges_to_same_energy(accel):
    sysp, basis = _h_atom_3d_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_ref = vq.run_pbc_bipole_uhf(
        sysp, basis, km, _bipole_rhf_opts(SCFAccelerator.DIIS),
    )
    assert r_ref.converged
    r = vq.run_pbc_bipole_uhf(
        sysp, basis, km, _bipole_rhf_opts(accel),
    )
    assert r.converged, f"BIPOLE UHF {accel.name} did not converge"
    assert r.energy == pytest.approx(r_ref.energy, abs=_ENERGY_TOL)


@pytest.mark.parametrize("accel", _MULTIK_ACCELERATORS)
def test_bipole_uks_accelerator_converges_to_same_energy(accel):
    sysp, basis = _h_atom_3d_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts_ref = _bipole_ks_opts(SCFAccelerator.DIIS)
    opts_ref.functional = "svwn"
    r_ref = vq.run_pbc_bipole_uks(sysp, basis, km, opts_ref)
    assert r_ref.converged
    opts_a = _bipole_ks_opts(accel)
    opts_a.functional = "svwn"
    r = vq.run_pbc_bipole_uks(sysp, basis, km, opts_a)
    assert r.converged, f"BIPOLE UKS {accel.name} did not converge"
    assert r.energy == pytest.approx(r_ref.energy, abs=_ENERGY_TOL)


def test_bipole_rhf_dynamic_damping_smoke(_bipole_rhf_reference):
    """dynamic_damping = True must not raise on BIPOLE and must reach
    the same energy as the static-damping reference."""
    sysp, basis, km, e_ref = _bipole_rhf_reference
    opts = _bipole_rhf_opts(SCFAccelerator.DIIS)
    opts.damping = 0.3
    opts.dynamic_damping = True
    r = vq.run_pbc_bipole_rhf(sysp, basis, km, opts)
    assert r.converged, "BIPOLE dynamic_damping run did not converge"
    assert r.energy == pytest.approx(e_ref, abs=_ENERGY_TOL)


# ---------------------------------------------------------------------------
# Native GDF (M4) — the {DIIS, KDIIS, EDIIS, EDIIS_DIIS, ADIIS} family +
# dynamic damping are wired into the three GDF drivers:
#   * run_rhf_periodic_gamma_gdf  (Γ-only; PeriodicSCFAccelerator)
#   * run_pbc_gdf_rhf             (Γ-only compensated-cell; PeriodicSCFAccelerator)
#   * run_krhf_periodic_gdf       (multi-k loop; MultiKPeriodicSCFAccelerator)
# An accelerator is a convergence aid, not a solver change — every choice
# must reach the same fixed point as the DIIS reference.
# ---------------------------------------------------------------------------

_GDF_AUX = "def2-svp-jk"


def _gdf_gamma_opts(accel) -> "vq.PeriodicRHFOptions":
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 200
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    opts.scf_accelerator = accel
    return opts


@pytest.fixture(scope="module")
def _gdf_gamma_rhf_reference():
    sysp, basis = _h2_3d_box()
    r = vq.run_rhf_periodic_gamma_gdf(
        sysp, basis, _gdf_gamma_opts(SCFAccelerator.DIIS),
        aux_basis=_GDF_AUX, progress=False,
    )
    assert r.converged, "reference Γ-GDF DIIS RHF must converge"
    return sysp, basis, r.energy


@pytest.mark.parametrize("accel", _ACCELERATORS)
def test_gamma_gdf_rhf_accelerator_converges_to_same_energy(
    _gdf_gamma_rhf_reference, accel,
):
    sysp, basis, e_ref = _gdf_gamma_rhf_reference
    r = vq.run_rhf_periodic_gamma_gdf(
        sysp, basis, _gdf_gamma_opts(accel),
        aux_basis=_GDF_AUX, progress=False,
    )
    assert r.converged, f"Γ-GDF RHF {accel.name} did not converge"
    assert r.energy == pytest.approx(e_ref, abs=_ENERGY_TOL), (
        f"Γ-GDF RHF {accel.name} energy {r.energy} disagrees with DIIS "
        f"reference {e_ref} beyond {_ENERGY_TOL} Ha"
    )


def test_gamma_gdf_rhf_dynamic_damping_smoke(_gdf_gamma_rhf_reference):
    """dynamic_damping = True must not raise on Γ-GDF and must reach the
    same energy as the static-damping reference."""
    sysp, basis, e_ref = _gdf_gamma_rhf_reference
    opts = _gdf_gamma_opts(SCFAccelerator.DIIS)
    opts.damping = 0.3
    opts.dynamic_damping = True
    r = vq.run_rhf_periodic_gamma_gdf(
        sysp, basis, opts, aux_basis=_GDF_AUX, progress=False,
    )
    assert r.converged, "Γ-GDF dynamic_damping run did not converge"
    assert r.energy == pytest.approx(e_ref, abs=_ENERGY_TOL)


@pytest.fixture(scope="module")
def _pbc_gdf_rhf_reference():
    sysp, basis = _h2_3d_box()
    opts = _gdf_gamma_opts(SCFAccelerator.DIIS)
    opts.lattice_opts.cutoff_bohr = 15.0
    r = vq.run_pbc_gdf_rhf(
        sysp, basis, opts, aux_basis=_GDF_AUX,
        exxdiv="ewald", compcell_eta=0.25, progress=False,
    )
    assert r.converged, "reference compcell-GDF DIIS RHF must converge"
    return sysp, basis, r.energy


@pytest.mark.parametrize("accel", _ACCELERATORS)
def test_pbc_gdf_rhf_accelerator_converges_to_same_energy(
    _pbc_gdf_rhf_reference, accel,
):
    sysp, basis, e_ref = _pbc_gdf_rhf_reference
    opts = _gdf_gamma_opts(accel)
    opts.lattice_opts.cutoff_bohr = 15.0
    r = vq.run_pbc_gdf_rhf(
        sysp, basis, opts, aux_basis=_GDF_AUX,
        exxdiv="ewald", compcell_eta=0.25, progress=False,
    )
    assert r.converged, f"compcell-GDF RHF {accel.name} did not converge"
    assert r.energy == pytest.approx(e_ref, abs=_ENERGY_TOL)


def _h2_cubic_cell(box: float = 12.0):
    """H₂ in a compact dim=3 cubic cell — Bravais carrier for the
    multi-k GDF accelerator tests (same shape as
    ``test_periodic_k_gdf._h2_cubic_box``)."""
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


@pytest.fixture(scope="module")
def _multik_gdf_rhf_reference():
    # This fixture used the dim=1 H₂ chain until 2026-07-27. The GDF gauge
    # dispatch now fails closed for dim<3 (ad7bdcd3: bare V_ne/E_nn lattice
    # sums are conditionally convergent and the neutral cderi collapses to a
    # transverse-uniform sheet term — no consistent Coulomb gauge exists), so
    # the multi-k GDF loop only accepts dim=3 cells. The accelerator-uniformity
    # property under test is dimension-independent; a compact 3-D cell with a
    # real (2,1,1) mesh exercises the same multi-k extrapolation paths.
    sysp, basis = _h2_cubic_cell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    opts = _gdf_gamma_opts(SCFAccelerator.DIIS)
    opts.lattice_opts.cutoff_bohr = 14.0
    opts.lattice_opts.nuclear_cutoff_bohr = 16.0
    r = vq.run_krhf_periodic_gdf(
        sysp, basis, km, opts, aux_basis=_GDF_AUX, progress=False,
    )
    assert r.converged, "reference multi-k GDF DIIS RHF must converge"
    return sysp, basis, km, r.energy


@pytest.mark.slow
@pytest.mark.parametrize("accel", _MULTIK_ACCELERATORS)
def test_multik_gdf_rhf_accelerator_converges_to_same_energy(
    _multik_gdf_rhf_reference, accel,
):
    """Multi-k GDF loop (``run_krhf_periodic_gdf``, real [2,1,1] mesh) —
    every accelerator reaches the DIIS fixed point. Exercises the
    multi-k extrapolation paths (per-k Pulay DIIS, KDIIS
    orbital-rotation gradient, and the stacked-real-block EDIIS /
    ADIIS bridge). Kept in the slow lane with the other end-to-end
    multi-k SCF sweeps (~2 s per SCF on the compact 12-bohr cell;
    the retired dim=1 chain variant was ~30-45 s per SCF)."""
    sysp, basis, km, e_ref = _multik_gdf_rhf_reference
    opts = _gdf_gamma_opts(accel)
    opts.lattice_opts.cutoff_bohr = 14.0
    opts.lattice_opts.nuclear_cutoff_bohr = 16.0
    r = vq.run_krhf_periodic_gdf(
        sysp, basis, km, opts, aux_basis=_GDF_AUX, progress=False,
    )
    assert r.converged, f"multi-k GDF RHF {accel.name} did not converge"
    assert r.energy == pytest.approx(e_ref, abs=_ENERGY_TOL)
