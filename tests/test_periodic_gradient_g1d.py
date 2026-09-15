"""Phase G1d — open-shell (UKS) periodic atomic gradient.

Pinned contracts:

1. **Public API** — ``vq.compute_gradient_periodic_uks_multi_k`` is
   exposed at the top level. Returns ``(n_atoms, 3)`` Ha/bohr.

2. **Closed-shell limit (multiplicity=1) ≡ RKS multi-k**. Running
   UKS on a closed-shell system should reproduce the RKS result to
   machine precision.

3. **Hybrid functionals refused** — α_HF ≠ 0 raises
   ``NotImplementedError`` (per-spin periodic K piece deferred to
   v0.6.x).

4. **Refusal on un-converged SCF**.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


def _h2_closed_shell_box(box_ang: float = 20.0):
    R = 1.0 * ANGSTROM_TO_BOHR
    big_box = box_ang * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(1, [0, 0, 0]),
        vq.Atom(1, [0, 0, R]),
    ]
    sys = vq.PeriodicSystem(3, np.diag([big_box]*3), atoms,
                              charge=0, multiplicity=1)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicKSOptions()
    opts.functional = "lda"
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 100
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    return sys, basis, opts, kmesh


# ---------------------------------------------------------------------------
# 1. Public API
# ---------------------------------------------------------------------------

def test_compute_gradient_periodic_uks_multi_k_exposed():
    assert hasattr(vq, "compute_gradient_periodic_uks_multi_k")


# ---------------------------------------------------------------------------
# 2. Closed-shell limit: UKS LDA matches RKS LDA bit-for-bit
# ---------------------------------------------------------------------------

def test_uks_closed_shell_matches_rks():
    sys, basis, opts, kmesh = _h2_closed_shell_box()
    r_uks = vq.run_uks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    r_rks = vq.run_rks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    assert r_uks.converged
    assert r_rks.converged

    g_uks = vq.compute_gradient_periodic_uks_multi_k(
        sys, basis, r_uks, kmesh, lattice_opts=opts.lattice_opts)
    g_rks = vq.compute_gradient_periodic_rks_multi_k(
        sys, basis, r_rks, kmesh, lattice_opts=opts.lattice_opts)
    np.testing.assert_allclose(g_uks, g_rks, atol=1e-9,
        err_msg="UKS gradient on closed-shell system diverges from RKS")


# ---------------------------------------------------------------------------
# 2b. GGA closed-shell limit: UKS PBE matches RKS PBE
#
#     Guards the UKS multi-k XC Pulay reroute to the open-shell lattice
#     primitive ``xc_lattice_gradient_contribution_uks`` (full GGA σ-piece).
#     On a closed-shell system the spin-polarized GGA XC Pulay must reduce
#     to the closed-shell one — exercising the σ-coupled path for both.
# ---------------------------------------------------------------------------

def test_uks_pbe_closed_shell_matches_rks():
    sys, basis, opts, kmesh = _h2_closed_shell_box()
    opts.functional = "pbe"
    r_uks = vq.run_uks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    r_rks = vq.run_rks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    assert r_uks.converged
    assert r_rks.converged

    g_uks = np.asarray(vq.compute_gradient_periodic_uks_multi_k(
        sys, basis, r_uks, kmesh, lattice_opts=opts.lattice_opts))
    g_rks = np.asarray(vq.compute_gradient_periodic_rks_multi_k(
        sys, basis, r_rks, kmesh, lattice_opts=opts.lattice_opts))
    np.testing.assert_allclose(g_uks, g_rks, atol=1e-9,
        err_msg="UKS PBE gradient on closed-shell system diverges from RKS "
                "PBE — open-shell GGA σ XC Pulay inconsistent with closed-shell")


# ---------------------------------------------------------------------------
# 3. Hybrid functionals raise NotImplementedError
# ---------------------------------------------------------------------------

def test_hybrid_functional_raises():
    sys, basis, opts, kmesh = _h2_closed_shell_box()
    opts.functional = "b3lyp"
    r = vq.run_uks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    if not r.converged:
        pytest.skip("B3LYP UKS didn't converge in test fixture; "
                    "structural test only.")
    with pytest.raises(NotImplementedError, match="hybrid"):
        vq.compute_gradient_periodic_uks_multi_k(
            sys, basis, r, kmesh, lattice_opts=opts.lattice_opts)


# ---------------------------------------------------------------------------
# 4. Un-converged input rejected
# ---------------------------------------------------------------------------

def test_unconverged_rejected():
    sys, basis, opts, kmesh = _h2_closed_shell_box()
    opts.max_iter = 1
    opts.conv_tol_energy = 1e-30
    r = vq.run_uks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    assert r.converged is False
    with pytest.raises(ValueError, match="not converged"):
        vq.compute_gradient_periodic_uks_multi_k(
            sys, basis, r, kmesh, lattice_opts=opts.lattice_opts)
