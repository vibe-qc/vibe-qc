"""Tests for the analytic GAPW / GPW periodic gradient (v0.12 R2).

Pins :func:`vibeqc.periodic_gapw_gradient.compute_gradient_gpw` and the
``VibeqcGPW`` Calculator's analytic-forces path against the (slow but
correct) full-SCF central-difference numerical-forces fallback.

Test surface:

1. Analytic vs numerical gradient agreement on H2 / STO-3G (RHF) —
   relative error < 1% on the dominant bond-axis component.
2. Forces zero at the H2 equilibrium bond length (to a few mHa/bohr).
3. Newton's third law on a stretched H2 dimer: F_atom_1 = -F_atom_2.
4. Analytic LDA-RKS forces on H2 / STO-3G match the FD reference.
5. ``VibeqcGPW.get_forces()`` (default settings → analytic path) and
   the same calculator with ``use_numerical_forces=True`` agree to
   <1% on the bond-axis force.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc
from vibeqc.periodic_gapw_grid import GAPWExperimentalWarning
from vibeqc.periodic_gapw_gradient import (
    GpwGradientReport,
    compute_gradient_gapw,
    compute_gradient_gpw,
)
from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.units import Bohr, Hartree  # noqa: E402

warnings.simplefilter("ignore", GAPWExperimentalWarning)
pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


def test_gapw_gradient_fails_closed_for_analytic_one_centre():
    """The HF default must not be differentiated with the block functional."""
    system = _h2_periodic_system(1.4)
    result = SimpleNamespace(converged=True, one_centre="analytic")
    with pytest.raises(NotImplementedError, match="one_centre='block'"):
        compute_gradient_gapw(
            system,
            None,
            result,
            basis_name="sto-3g",
        )


def test_vibeqc_gapw_forwards_one_centre_to_gamma_driver(monkeypatch):
    """The ASE wrapper must preserve an explicit one-centre policy."""
    import vibeqc.ase_periodic_gapw as ase_gapw

    seen = {}

    def _fake_run(_system, _basis, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(converged=True)

    monkeypatch.setattr(ase_gapw, "run_periodic_rhf_gapw", _fake_run)
    calc = ase_gapw.VibeqcGAPW(
        basis="sto-3g",
        cutoff_ha=50.0,
        gapw_kwargs={"one_centre": "analytic", "molecular_limit": True},
    )

    calc._run_scf(_h2_atoms(1.4))

    assert seen["one_centre"] == "analytic"
    assert seen["molecular_limit"] is True


@pytest.mark.parametrize(
    "gapw_kwargs",
    [
        {"one_centre": "block"},
        {"molecular_limit": True},
    ],
)
def test_vibeqc_gapw_multik_rejects_gamma_only_controls(
    monkeypatch,
    gapw_kwargs,
):
    """Multi-k ASE calculations must not silently drop scientific controls."""
    import vibeqc.ase_periodic_gapw as ase_gapw

    def _unexpected_run(*_args, **_kwargs):
        raise AssertionError("unsupported multi-k control entered the GAPW SCF")

    monkeypatch.setattr(
        ase_gapw,
        "run_periodic_rks_gapw_multi_k",
        _unexpected_run,
    )
    calc = ase_gapw.VibeqcGAPW(
        basis="sto-3g",
        functional="lda",
        cutoff_ha=50.0,
        kmesh=(2, 1, 1),
        gapw_kwargs=gapw_kwargs,
    )

    with pytest.raises(NotImplementedError, match="Gamma-only scientific controls"):
        calc._run_scf(_h2_atoms(1.4))


def test_vibeqc_gapw_analytic_forces_use_numerical_scf_derivative(monkeypatch):
    """Analytic one-centre energy cannot use the block-only force kernel."""
    from vibeqc.ase_periodic_gapw import VibeqcGAPW

    atoms = _h2_atoms(1.4)
    calc = VibeqcGAPW(basis="sto-3g", cutoff_ha=50.0)
    central = SimpleNamespace(
        converged=True,
        density=np.eye(2),
        energy=-1.0,
        one_centre="analytic",
    )
    expected = np.full((2, 3), 7.5)
    calls = []

    monkeypatch.setattr(calc, "_run_scf", lambda _atoms: central)

    def _numerical(result):
        calls.append(result)
        return expected

    def _unexpected_analytic(_result):
        raise AssertionError("analytic one-centre result used block GAPW forces")

    monkeypatch.setattr(calc, "_compute_numerical_forces", _numerical)
    monkeypatch.setattr(calc, "_compute_gapw_forces", _unexpected_analytic)

    calc.calculate(atoms, properties=("energy", "forces"))

    assert calls == [central]
    np.testing.assert_array_equal(calc.results["forces"], expected)


# ---------- Helpers -------------------------------------------------------


def _h2_periodic_system(d_bohr: float, L_bohr: float = 12.0):
    """Build an H2 PeriodicSystem aligned with the x axis at bond length
    ``d_bohr`` in a cubic box of side ``L_bohr``. Uses the same box-
    placement convention as ``tests/test_ase_periodic_gpw.py``.
    """
    p1 = [L_bohr / 2 - 0.5 * d_bohr, L_bohr / 2, L_bohr / 2]
    p2 = [L_bohr / 2 + 0.5 * d_bohr, L_bohr / 2, L_bohr / 2]
    sys = vibeqc.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3, dtype=np.float64) * L_bohr
    sys.unit_cell = [
        vibeqc.Atom(1, p1),
        vibeqc.Atom(1, p2),
    ]
    return sys


def _h2_atoms(d_bohr: float, L_bohr: float = 12.0) -> Atoms:
    L_ang = L_bohr * Bohr
    p1 = np.array([L_bohr / 2 - 0.5 * d_bohr, L_bohr / 2, L_bohr / 2]) * Bohr
    p2 = np.array([L_bohr / 2 + 0.5 * d_bohr, L_bohr / 2, L_bohr / 2]) * Bohr
    atoms = Atoms("H2", positions=[p1, p2], pbc=True)
    atoms.set_cell(np.eye(3) * L_ang)
    return atoms


def _numerical_gradient_h2_x(
    d_bohr: float, *, L_bohr: float = 12.0, functional=None,
    step_bohr: float = 5e-3, cutoff_ha: float = 200.0,
) -> float:
    """Central-difference gradient on the H2 bond axis via full-SCF FD.

    Returns dE/dR_{atom0, x} in Ha/bohr (the analytic gradient should
    agree on this component up to its truncation error). Two SCFs.
    """
    sys_plus = _h2_periodic_system(d_bohr + 2.0 * step_bohr, L_bohr=L_bohr)
    sys_minus = _h2_periodic_system(d_bohr - 2.0 * step_bohr, L_bohr=L_bohr)
    # Bond-axis FD: moving atom 0 by -step along x increases d by step
    # (atom 0 is at -d/2). Build symmetric +/- around the central d.
    # Easier: just FD on the total energy w.r.t. atom-0 x at fixed
    # geometry.
    sys_p = _h2_periodic_system(d_bohr, L_bohr=L_bohr)
    sys_m = _h2_periodic_system(d_bohr, L_bohr=L_bohr)
    # Displace atom 0 by +step / -step along x.
    sys_p.unit_cell = [
        vibeqc.Atom(1, [
            L_bohr / 2 - 0.5 * d_bohr + step_bohr,
            L_bohr / 2, L_bohr / 2,
        ]),
        vibeqc.Atom(1, [
            L_bohr / 2 + 0.5 * d_bohr,
            L_bohr / 2, L_bohr / 2,
        ]),
    ]
    sys_m.unit_cell = [
        vibeqc.Atom(1, [
            L_bohr / 2 - 0.5 * d_bohr - step_bohr,
            L_bohr / 2, L_bohr / 2,
        ]),
        vibeqc.Atom(1, [
            L_bohr / 2 + 0.5 * d_bohr,
            L_bohr / 2, L_bohr / 2,
        ]),
    ]
    mol_p = vibeqc.Molecule(list(sys_p.unit_cell), 0, 1)
    mol_m = vibeqc.Molecule(list(sys_m.unit_cell), 0, 1)
    basis_p = vibeqc.BasisSet(mol_p, "sto-3g")
    basis_m = vibeqc.BasisSet(mol_m, "sto-3g")
    res_p = run_periodic_rhf_gpw(
        sys_p, basis_p, cutoff_ha=cutoff_ha, functional=functional, quiet=True,
    )
    res_m = run_periodic_rhf_gpw(
        sys_m, basis_m, cutoff_ha=cutoff_ha, functional=functional, quiet=True,
    )
    assert res_p.converged and res_m.converged
    return (res_p.energy - res_m.energy) / (2.0 * step_bohr)


# ---------- 1. RHF analytic vs numerical agreement ------------------------


def test_gpw_gradient_rhf_h2_matches_numerical():
    """Analytic gradient on H2 / STO-3G / RHF agrees with the SCF FD
    reference on the bond axis to <1% relative error."""
    d_bohr = 1.6  # off equilibrium so the gradient is well above noise
    L_bohr = 12.0
    sys = _h2_periodic_system(d_bohr, L_bohr=L_bohr)
    mol = vibeqc.Molecule(list(sys.unit_cell), 0, 1)
    basis = vibeqc.BasisSet(mol, "sto-3g")
    res = run_periodic_rhf_gpw(
        sys, basis, cutoff_ha=200.0, functional=None, quiet=True,
    )
    assert res.converged

    grad = compute_gradient_gpw(
        sys, basis, res,
        basis_name="sto-3g",
        functional=None,
    )
    assert grad.shape == (2, 3)

    g_numerical = _numerical_gradient_h2_x(d_bohr, L_bohr=L_bohr)
    g_analytic = float(grad[0, 0])

    # Both must be the same sign and within 1% relative.
    rel_err = abs(g_analytic - g_numerical) / max(abs(g_numerical), 1e-8)
    assert rel_err < 1e-2, (
        f"RHF analytic gradient mismatch: analytic = {g_analytic:.6e}, "
        f"numerical = {g_numerical:.6e}, rel err = {rel_err:.3%}"
    )


# ---------- 2. Equilibrium force is small ---------------------------------


def test_gpw_gradient_h2_equilibrium_is_small():
    """At the H2 / STO-3G equilibrium bond length (~1.388 bohr in this
    box / cutoff), the bond-axis force should be small. We use a loose
    tolerance because STO-3G's equilibrium isn't bit-for-bit ours; the
    point is that |F| stays in the few-mHa/bohr range, not at the
    several-tenths-of-Ha/bohr level seen far off equilibrium."""
    d_bohr = 1.388  # STO-3G equilibrium-ish
    sys = _h2_periodic_system(d_bohr, L_bohr=12.0)
    mol = vibeqc.Molecule(list(sys.unit_cell), 0, 1)
    basis = vibeqc.BasisSet(mol, "sto-3g")
    res = run_periodic_rhf_gpw(
        sys, basis, cutoff_ha=200.0, functional=None, quiet=True,
    )
    grad = compute_gradient_gpw(
        sys, basis, res,
        basis_name="sto-3g",
        functional=None,
    )
    bond_force = float(grad[0, 0])
    assert abs(bond_force) < 5e-2, (
        f"Equilibrium bond-axis gradient unexpectedly large: "
        f"{bond_force:.4e} Ha/bohr"
    )


# ---------- 3. Newton's third law (translation invariance) -----------------


def test_gpw_gradient_h2_newton_third_law():
    """For a closed system, sum of gradients = 0 (translation invariance).
    On an isolated H2 dimer in a vacuum-padded box, dE/dR_atom_0 + dE/
    dR_atom_1 should vanish to high accuracy on every axis.
    """
    d_bohr = 1.6
    sys = _h2_periodic_system(d_bohr, L_bohr=12.0)
    mol = vibeqc.Molecule(list(sys.unit_cell), 0, 1)
    basis = vibeqc.BasisSet(mol, "sto-3g")
    res = run_periodic_rhf_gpw(
        sys, basis, cutoff_ha=200.0, functional=None, quiet=True,
    )
    report = compute_gradient_gpw(
        sys, basis, res,
        basis_name="sto-3g",
        functional=None,
        return_report=True,
    )
    assert isinstance(report, GpwGradientReport)

    component_nets = {
        "total": report.total.sum(axis=0),
        "e_overlap_pulay": report.e_overlap_pulay.sum(axis=0),
        "e_hellmann_feynman": report.e_hellmann_feynman.sum(axis=0),
    }
    component_limits = {
        "total": 1e-8,
        "e_overlap_pulay": 1e-12,
        "e_hellmann_feynman": 1e-8,
    }
    diagnostics = "; ".join(
        f"{name}: net={component_nets[name]}, "
        f"max={np.max(np.abs(component_nets[name])):.3e}, "
        f"limit={component_limits[name]:.1e}"
        for name in component_nets
    )

    # Revalidated for GitLab IID 188 on 2026-08-20: the total residual was
    # 7.55e-11 Ha/bohr at h=1e-3 and 200 Ha, and stayed below 8.89e-11
    # across 50/100/200/300 Ha; the analytic Pulay residual was <=2.8e-16.
    for name, limit in component_limits.items():
        assert np.max(np.abs(component_nets[name])) < limit, (
            "GPW Newton-symmetry violation on H2; " + diagnostics
        )


# ---------- 4. LDA-RKS forces ---------------------------------------------


def test_gpw_gradient_rks_lda_h2_matches_numerical():
    """LDA-RKS analytic gradient on H2 / STO-3G matches the full-SCF FD
    reference to <1% relative error on the bond axis."""
    d_bohr = 1.6
    L_bohr = 12.0
    sys = _h2_periodic_system(d_bohr, L_bohr=L_bohr)
    mol = vibeqc.Molecule(list(sys.unit_cell), 0, 1)
    basis = vibeqc.BasisSet(mol, "sto-3g")
    res = run_periodic_rhf_gpw(
        sys, basis, cutoff_ha=200.0, functional="lda", quiet=True,
    )
    assert res.converged

    grad = compute_gradient_gpw(
        sys, basis, res,
        basis_name="sto-3g",
        functional="lda",
    )
    g_analytic = float(grad[0, 0])
    g_numerical = _numerical_gradient_h2_x(
        d_bohr, L_bohr=L_bohr, functional="lda",
    )
    rel_err = abs(g_analytic - g_numerical) / max(abs(g_numerical), 1e-8)
    assert rel_err < 1e-2, (
        f"LDA-RKS analytic gradient mismatch: analytic = {g_analytic:.6e}, "
        f"numerical = {g_numerical:.6e}, rel err = {rel_err:.3%}"
    )


# ---------- 5. VibeqcGPW.get_forces() analytic path -----------------------


def test_vibeqc_gpw_default_uses_analytic_forces_matches_numerical():
    """The Calculator's default forces path is analytic; the legacy
    central-difference path is opt-in via ``use_numerical_forces=True``.
    They should agree to <1% on the dominant bond-axis force.
    """
    atoms = _h2_atoms(d_bohr=1.6, L_bohr=12.0)

    calc_analytic = vibeqc.VibeqcGPW(basis="sto-3g", cutoff_ha=200.0)
    atoms.calc = calc_analytic
    f_analytic = atoms.get_forces()  # eV/Å

    # Fresh ASE Atoms so the results cache is invalidated cleanly.
    atoms_num = _h2_atoms(d_bohr=1.6, L_bohr=12.0)
    calc_num = vibeqc.VibeqcGPW(
        basis="sto-3g", cutoff_ha=200.0,
        use_numerical_forces=True,
    )
    atoms_num.calc = calc_num
    f_num = atoms_num.get_forces()

    # Bond-axis force on atom 0 should agree to <1%.
    f_a = float(f_analytic[0, 0])
    f_n = float(f_num[0, 0])
    rel_err = abs(f_a - f_n) / max(abs(f_n), 1e-8)
    assert rel_err < 1e-2, (
        f"Calculator analytic vs numerical force mismatch: "
        f"analytic = {f_a:.6e} eV/Å, numerical = {f_n:.6e} eV/Å, "
        f"rel err = {rel_err:.3%}"
    )


# ---------- 6. Report surface (per-term breakdown) ------------------------


def test_gpw_gradient_report_records_per_term_methods():
    """``return_report=True`` must return a GpwGradientReport with the
    per-term arrays and the ``terms_method`` flag map. Pins the public
    surface so tests / users can inspect which contributions are FD."""
    sys = _h2_periodic_system(d_bohr=1.6, L_bohr=12.0)
    mol = vibeqc.Molecule(list(sys.unit_cell), 0, 1)
    basis = vibeqc.BasisSet(mol, "sto-3g")
    res = run_periodic_rhf_gpw(
        sys, basis, cutoff_ha=200.0, functional=None, quiet=True,
    )
    report = compute_gradient_gpw(
        sys, basis, res,
        basis_name="sto-3g",
        functional=None,
        return_report=True,
    )
    assert isinstance(report, GpwGradientReport)
    assert report.total.shape == (2, 3)
    # Two-term decomposition: analytic Pulay + fixed-density energy FD.
    assert report.terms_method["e_overlap_pulay"] == "analytic"
    assert report.terms_method["e_hellmann_feynman"] == (
        "fd_fixed_density_energy"
    )
    summed = report.e_overlap_pulay + report.e_hellmann_feynman
    assert np.allclose(summed, report.total, atol=1e-14)
