"""ωB97M-V (Mardirossian & Head-Gordon, *J. Chem. Phys.* **144**, 214110
(2016)) — the range-separated meta-GGA hybrid with VV10 nonlocal
correlation; the most accurate member of the ωB97-V line.

ωB97M-V composes all three of vibe-qc's advanced XC machineries at once:
the range-separated erf-attenuated exchange (libxc reports its CAM
coefficients), the τ-dependent meta-GGA Kohn-Sham path, and the VV10
nonlocal correlation (``compute_vv10``). This pins that they compose
correctly and that the complete functional matches PySCF.

Also pins the honest gate on ωB97M(2): libxc 7.0.0 has no ωB97M(2)
semilocal base (only ωB97M-V), so the name raises a specific, helpful
error rather than the generic "unknown name".
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, Functional, Molecule, RKSOptions, run_rks

from .conftest import GEOMETRIES


def test_wb97m_v_functional_resolves():
    """ωB97M-V is a range-separated (long-range-corrected) meta-GGA hybrid
    that flags VV10 with b = 6.0, C = 0.01."""
    f = Functional("wb97m-v")
    assert f.kind == "XCKind.MGGA" or "MGGA" in str(f.kind)
    assert f.is_range_separated is True
    assert f.rsh_omega == pytest.approx(0.3, abs=1e-9)
    # Long-range-corrected: 100 % HF at long range.
    assert f.cam_alpha + f.cam_beta == pytest.approx(1.0, abs=1e-9)
    assert f.needs_vv10 is True
    assert f.vv10_b == pytest.approx(6.0, abs=1e-10)
    assert f.vv10_C == pytest.approx(0.01, abs=1e-10)


def test_wb97m2_is_gated_with_a_helpful_message():
    """ωB97M(2) is not implementable as an alias (libxc lacks the
    re-optimised B97M(2) base); the resolver says so specifically."""
    for name in ("wb97m(2)", "wb97m2", "WB97M(2)"):
        with pytest.raises(Exception, match="ωB97M.2.|not yet implemented"):
            Functional(name)


def _atoms_to_mol_basis(atoms_bohr, basis_name):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    return mol, BasisSet(mol, basis_name)


def test_wb97m_v_total_vs_pyscf():
    """End-to-end ωB97M-V SCF — RSH + meta-GGA + VV10 composed — matches
    PySCF's self-consistent ωB97M-V to grid accuracy on H2O / cc-pVDZ."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft

    atoms = GEOMETRIES["H2O"]
    mol, basis = _atoms_to_mol_basis(atoms, "cc-pvdz")
    opts = RKSOptions()
    opts.functional = "wb97m-v"
    opts.density_fit = False  # RSH needs the direct path
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-7
    res = run_rks(mol, basis, opts)
    assert res.converged

    pm = gto.Mole()
    pm.unit = "Bohr"
    pm.atom = [[Z, tuple(xyz)] for Z, xyz in atoms]
    pm.basis = "cc-pvdz"
    pm.verbose = 0
    pm.build()
    mf = dft.RKS(pm, xc="wb97m-v")
    mf.nlc = "VV10"
    mf.grids.level = 3
    mf.nlcgrids.level = 3
    mf.conv_tol = 1e-10
    e_pyscf = mf.kernel()
    assert mf.converged

    # Grid-limited cross-code agreement (the kernel itself is pinned
    # tightly by test_vv10.py's shared-grid comparison). 5e-5 leaves
    # headroom for grid-default / BLAS-summation drift across machines.
    assert res.energy == pytest.approx(e_pyscf, abs=5e-5), (
        f"vibeqc ωB97M-V = {res.energy:.8f}, PySCF = {e_pyscf:.8f}, "
        f"gap = {(res.energy - e_pyscf) * 1e6:+.2f} µHa")
