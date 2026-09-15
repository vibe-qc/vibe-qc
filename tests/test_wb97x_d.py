"""ωB97X-D — range-separated hybrid + Chai-Head-Gordon dispersion.

ωB97X-D (Chai & Head-Gordon, PCCP 10, 6615 (2008)) is a
range-separated hybrid GGA (libxc XC_HYB_GGA_XC_WB97X_D, the
milestone-A RSH path) carrying an intrinsic empirical "-D" dispersion
of the older DFT-D2 family. The two pieces are independent — the
dispersion is geometry-only — so the complete ωB97X-D energy is
``E_SCF + E_disp``, assembled by :func:`vibeqc.run_wb97x_d`.

The CHG dispersion algorithm and the Grimme-2006 D2 C6 / vdW-radius
tables are transcribed verbatim from Psi4's ``libdisp`` "CHG" scheme;
the unit tests below hand-compute the pairwise sum from the published
formula to confirm the vibe-qc kernel reproduces it exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Functional,
    Molecule,
    UKSOptions,
    XCKind,
    compute_chg_dispersion,
    run_wb97x_d,
)


# ---------------------------------------------------------------------------
# CHG dispersion kernel — compute_chg_dispersion
# ---------------------------------------------------------------------------

# Grimme-2006 "DFT-D2" atomic parameters (atomic units) for He, from
# Psi4's libdisp table — used to hand-compute the reference below.
_C6_HE = 1.3876202159098600    # Eh·bohr^6
_RVDW_HE = 1.91240270086800    # bohr
_CHG_S6 = 1.0
_CHG_D = 6.0


def _chg_pair_energy(C6_i, C6_j, Rv_i, Rv_j, R):
    """Hand evaluation of one CHG dispersion pair term:
    -s6 · sqrt(C6_i·C6_j) / R^6 · 1/(1 + d·(R/(Rv_i+Rv_j))^-12)."""
    C6 = math.sqrt(C6_i * C6_j)
    R0 = Rv_i + Rv_j
    f = 1.0 / (1.0 + _CHG_D * (R / R0) ** (-12.0))
    return -_CHG_S6 * C6 / R**6 * f


def test_chg_dispersion_matches_hand_computed_formula():
    """compute_chg_dispersion on a He dimer must reproduce the
    hand-evaluated CHG pairwise formula bit-for-bit."""
    R = 5.0
    mol = Molecule([Atom(2, [0.0, 0.0, 0.0]), Atom(2, [0.0, 0.0, R])])
    e_ref = _chg_pair_energy(_C6_HE, _C6_HE, _RVDW_HE, _RVDW_HE, R)
    res = compute_chg_dispersion(mol)
    assert res.energy == pytest.approx(e_ref, abs=1e-14)


def test_chg_dispersion_is_attractive_and_decays():
    """The dispersion energy is negative (attractive) at every
    separation and decays toward zero as the atoms move apart."""
    energies = []
    for R in (4.0, 6.0, 10.0, 30.0):
        mol = Molecule([Atom(10, [0.0, 0.0, 0.0]),
                        Atom(10, [0.0, 0.0, R])])
        e = compute_chg_dispersion(mol).energy
        assert e < 0.0
        energies.append(abs(e))
    # Monotone decay once past the damping turn-on (R ≥ 6 bohr here).
    assert energies[1] > energies[2] > energies[3]


def test_chg_dispersion_gradient_matches_finite_difference():
    """The analytic CHG dispersion gradient agrees with a central
    finite difference of the energy."""
    geom = [(8, [0.0, 0.0, 0.0]), (8, [0.0, 0.0, 4.0])]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in geom])
    g = np.asarray(compute_chg_dispersion(mol, with_gradient=True).gradient)
    h = 1e-5
    fd = np.zeros((2, 3))
    for a in range(2):
        for c in range(3):
            gp = [(Z, list(xyz)) for Z, xyz in geom]
            gm = [(Z, list(xyz)) for Z, xyz in geom]
            gp[a][1][c] += h
            gm[a][1][c] -= h
            ep = compute_chg_dispersion(
                Molecule([Atom(Z, xyz) for Z, xyz in gp])).energy
            em = compute_chg_dispersion(
                Molecule([Atom(Z, xyz) for Z, xyz in gm])).energy
            fd[a, c] = (ep - em) / (2 * h)
    np.testing.assert_allclose(g, fd, atol=1e-10)


def test_chg_dispersion_rejects_element_outside_table():
    """The DFT-D2 C6 table covers Z ≤ 54 (H–Xe). A heavier element
    must raise a clear error, not silently return a wrong number."""
    mol = Molecule([Atom(79, [0.0, 0.0, 0.0]),     # Au
                    Atom(79, [0.0, 0.0, 5.0])])
    with pytest.raises((RuntimeError, ValueError), match="54|Xe|table"):
        compute_chg_dispersion(mol)


# ---------------------------------------------------------------------------
# ωB97X-D functional alias + run_wb97x_d dispatcher
# ---------------------------------------------------------------------------

def test_wb97x_d_alias_is_range_separated():
    """The ``wb97x-d`` alias resolves to libxc XC_HYB_GGA_XC_WB97X_D —
    a range-separated hybrid GGA (ω = 0.2, 22.2 % HF short-range →
    100 % long-range). It is the *XC* part only; the dispersion is
    added by run_wb97x_d."""
    f = Functional("wb97x-d")
    assert f.kind == XCKind.GGA
    assert f.is_range_separated
    assert f.rsh_omega == pytest.approx(0.2, abs=1e-9)
    assert f.cam_alpha == pytest.approx(0.222036, abs=1e-5)
    assert (f.cam_alpha + f.cam_beta) == pytest.approx(1.0, abs=1e-9)
    # Dashless alias resolves identically.
    assert Functional("wb97xd").rsh_omega == pytest.approx(0.2, abs=1e-9)


def test_run_wb97x_d_closed_shell_h2o():
    """End-to-end run_wb97x_d on H2O — the RSH SCF (libxc 471) plus the
    CHG dispersion. e_scf cross-checked vs PySCF.dft RKS 'wb97xd'
    (the libxc XC part) ≈ -76.33200; e_total = e_scf + e_disp."""
    mol = Molecule([
        Atom(8, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])
    res = run_wb97x_d(mol, BasisSet(mol, "def2-svp"))
    assert res.scf.converged
    assert res.functional == "wb97x-d"
    # XC/SCF part — RSH path on libxc 471.
    assert res.scf.energy == pytest.approx(-76.332002, abs=5e-4)
    # Dispersion is small but non-zero and attractive for H2O.
    assert res.dispersion.energy < 0.0
    assert res.dispersion.energy == pytest.approx(-2.648e-05, abs=2e-6)
    # Total is the exact sum.
    assert res.e_total == pytest.approx(
        res.scf.energy + res.dispersion.energy, abs=1e-12)


def test_run_wb97x_d_open_shell_o2_triplet():
    """run_wb97x_d on the O2 triplet — exercises the open-shell (UKS)
    range-separated path with the dispersion correction."""
    mol = Molecule([Atom(8, [0, 0, -1.1]), Atom(8, [0, 0, 1.1])],
                   charge=0, multiplicity=3)
    opts = UKSOptions()
    opts.max_iter = 200
    res = run_wb97x_d(mol, BasisSet(mol, "def2-svp"), opts)
    assert res.scf.converged
    assert res.scf.s_squared == pytest.approx(2.0, abs=0.05)
    assert res.e_total == pytest.approx(
        res.scf.energy + res.dispersion.energy, abs=1e-12)
    assert res.e_total == pytest.approx(-150.1542, abs=5e-4)
