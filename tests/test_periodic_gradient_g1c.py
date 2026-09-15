"""Phase G1c — multi-k periodic atomic gradient.

Pinned contracts:

1. **Public API** — ``vq.compute_gradient_periodic_rhf_multi_k`` and
   ``vq.compute_gradient_periodic_rks_multi_k`` are exposed at the
   top level. Both return ``(n_atoms, 3)`` Ha/bohr.

2. **Every finite k mesh differentiates its own SCF energy**. Finite meshes
   have different Born-von-Karman exchange supercells and are not required
   to have identical forces before the cell/k-mesh limit is converged.

3. **Newton's-3rd-law on multi-k periodic** — H₂ chain at 8 Å with
   2×1×1 mesh: equal-and-opposite forces along the periodic axis.

4. **Multi-k XC Pulay** — the periodic LDA/GGA primitive is used.

5. **Refusal on un-converged SCF** — non-converged input raises.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


# ---------------------------------------------------------------------------
# 1. Public API
# ---------------------------------------------------------------------------

def test_compute_gradient_periodic_rhf_multi_k_exposed():
    assert hasattr(vq, "compute_gradient_periodic_rhf_multi_k")


def test_compute_gradient_periodic_rks_multi_k_exposed():
    assert hasattr(vq, "compute_gradient_periodic_rks_multi_k")


# ---------------------------------------------------------------------------
# 2. Every mesh's analytic derivative matches its own energy
# ---------------------------------------------------------------------------


def _run_h2_box_rhf(
    mesh,
    atom0_z_displacement=0.0,
    *,
    box_bohr=None,
    cutoff_bohr=25.0,
    omega=0.0,
    auto_optimize_truncation=True,
):
    R = 1.0 * ANGSTROM_TO_BOHR
    big_box = (
        20.0 * ANGSTROM_TO_BOHR if box_bohr is None else float(box_bohr)
    )
    atoms = [
        vq.Atom(1, [0, 0, atom0_z_displacement]),
        vq.Atom(1, [0, 0, R]),
    ]
    sys = vq.PeriodicSystem(3, np.diag([big_box]*3), atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = float(cutoff_bohr)
    opts.lattice_opts.nuclear_cutoff_bohr = float(cutoff_bohr)
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 100

    kmesh = vq.monkhorst_pack(sys, mesh)
    result = vq.run_rhf_periodic_multi_k_ewald3d(
        sys,
        basis,
        kmesh,
        opts,
        omega=float(omega),
        auto_optimize_truncation=bool(auto_optimize_truncation),
    )
    assert result.converged, f"{mesh} did not converge"
    return sys, basis, kmesh, opts, result


def test_h2_box_each_kmesh_gradient_matches_own_energy_fd():
    """The 1x1x1 and 2x2x2 analytics match their own finite-mesh energies.

    A finite k mesh makes the exact-exchange pair density periodic on an
    effective ``N_k * Omega`` supercell; the residual image interaction is
    therefore mesh dependent (Sundararaman and Arias, Phys. Rev. B 87,
    165122 (2013), Eqs. 10-13).  The old test incorrectly forced the two
    finite-mesh forces to agree at 1e-6 Ha/bohr even though their converged
    energy derivatives differ by 8.27e-6 Ha/bohr in this 20-Angstrom box.

    #575 (2026-09-03): that 8.27e-6 is a genuine *FD-vs-FD* difference
    between the two meshes and is not the defect this test caught.  The
    analytic gradient sat 1.03e-5 Ha/bohr off its own FD at *both* meshes
    (the [2,2,2] leg had never run: the assertion below fires on [1,1,1]
    first), FD-step independent and scaling as 1/L with the box, because
    the shared corrected multi-k derivative summed the erfc Hartree image
    interaction only over the density support (``cutoff_bohr`` = 25 bohr:
    the home cell alone in this 37.8 bohr box) while the SCF's analytic-FT
    J carries every image; with ``alpha = 2.8 / V^(1/3)`` the nearest
    image sits at ``erfc(2.8) = 7.5e-5``.  The derivative now sizes that
    image ball from alpha, and this assertion holds at 5.7e-8 (h = 1e-3).
    The 2e-6 tolerance is unchanged.
    """
    h = 1.0e-3
    for mesh in ([1, 1, 1], [2, 2, 2]):
        sys, basis, kmesh, opts, result = _run_h2_box_rhf(mesh)
        analytic = np.asarray(
            vq.compute_gradient_periodic_rhf_multi_k(
                sys,
                basis,
                result,
                kmesh,
                lattice_opts=opts.lattice_opts,
            )
        )
        # Current results carry the post-optimization lattice support, so the
        # optional caller argument cannot silently fall back to a default
        # DIRECT_TRUNCATED derivative or a stale pre-optimization cutoff.
        # Mutate the caller-owned object after SCF to prove the result retained
        # a separate snapshot rather than an alias.
        opts.lattice_opts.coulomb_method = (
            vq.CoulombMethod.DIRECT_TRUNCATED
        )
        opts.lattice_opts.cutoff_bohr = 1.0
        opts.lattice_opts.nuclear_cutoff_bohr = 1.0
        inferred = np.asarray(
            vq.compute_gradient_periodic_rhf_multi_k(
                sys, basis, result, kmesh
            )
        )
        np.testing.assert_allclose(inferred, analytic, atol=0.0)
        if mesh == [1, 1, 1]:
            explicit_twist = vq.KPoints.from_list(
                sys, [[0.125, 0.0, 0.0]]
            ).to_bloch_kmesh()
            with pytest.raises(
                NotImplementedError,
                match="one-point corrected-Ewald mesh must be Gamma",
            ):
                vq.compute_gradient_periodic_rhf_multi_k(
                    sys, basis, result, explicit_twist
                )
        plus = _run_h2_box_rhf(mesh, +h)[-1].energy
        minus = _run_h2_box_rhf(mesh, -h)[-1].energy
        fd_atom0_z = (plus - minus) / (2.0 * h)

        np.testing.assert_allclose(
            analytic[0, 2],
            fd_atom0_z,
            atol=2.0e-6,
            err_msg=f"{mesh} analytic gradient does not differentiate its SCF energy",
        )
        np.testing.assert_allclose(analytic[1, 2], -fd_atom0_z, atol=2.0e-6)
        np.testing.assert_allclose(analytic[:, :2], 0.0, atol=1.0e-9)


def test_h2_box_gradient_is_box_converged_like_its_energy():
    """The Ewald analytic gradient carries no 1/L term its energy lacks (#575).

    H2/STO-3G at kmesh [1,1,1] in a 30 bohr and a 37.8 bohr box: the SCF
    energies differ by 6e-8 Ha and their central-FD derivatives by 8e-8
    Ha/bohr (both box-converged), so the analytic gradients must agree to the
    same order.  Before the fix they differed by 2.85e-6 Ha/bohr: the
    corrected multi-k derivative summed the erfc Hartree image interaction
    only over the density support (the home cell in both boxes), dropping a
    term proportional to erfc(alpha L)/L = erfc(2.8)/L.  No finite difference
    is involved, so this pins the 1/L signature directly; measured after the
    fix: |dg| = 7.7e-8.
    """
    grads = []
    for box in (30.0, 20.0 * ANGSTROM_TO_BOHR):
        sys, basis, kmesh, opts, result = _run_h2_box_rhf(
            [1, 1, 1], box_bohr=box
        )
        grads.append(
            np.asarray(
                vq.compute_gradient_periodic_rhf_multi_k(
                    sys, basis, result, kmesh, lattice_opts=opts.lattice_opts
                )
            )
        )
    np.testing.assert_allclose(
        grads[0],
        grads[1],
        atol=5.0e-7,
        err_msg="Ewald analytic gradient changes with the box while its "
        "energy does not: a truncated erfc image sum (#575)",
    )


def test_h2_explicit_ewald_alpha_matches_own_energy_fd():
    """An explicit broad Ewald split uses one reciprocal energy/force domain.

    At ``alpha=0.5`` in this 30-bohr box the volume-only reciprocal cutoff
    truncates the exchange Gaussian while it is still large.  The corrected
    SCF widens its reciprocal K domain from 0.794 to 6.438 bohr^-1; both the
    J_LR and K_LR derivatives must use that same alpha-resolved envelope.
    """
    mesh = [2, 1, 1]
    h = 1.0e-3
    run_kwargs = {
        "box_bohr": 30.0,
        "cutoff_bohr": 12.0,
        "omega": 0.5,
        "auto_optimize_truncation": False,
    }
    sys, basis, kmesh, _, result = _run_h2_box_rhf(mesh, **run_kwargs)
    analytic = np.asarray(
        vq.compute_gradient_periodic_rhf_multi_k(sys, basis, result, kmesh)
    )
    plus = _run_h2_box_rhf(
        mesh, +h, **run_kwargs
    )[-1].energy
    minus = _run_h2_box_rhf(
        mesh, -h, **run_kwargs
    )[-1].energy
    fd_atom0_z = (plus - minus) / (2.0 * h)

    np.testing.assert_allclose(analytic[0, 2], fd_atom0_z, atol=2.0e-6)
    np.testing.assert_allclose(analytic[1, 2], -fd_atom0_z, atol=2.0e-6)


def _run_h2_box_pbe0(atom0_z_displacement=0.0):
    bond = 1.0 * ANGSTROM_TO_BOHR
    box = 20.0 * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(1, [0.0, 0.0, atom0_z_displacement]),
        vq.Atom(1, [0.0, 0.0, bond]),
    ]
    system = vq.PeriodicSystem(3, np.diag([box] * 3), atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicKSOptions()
    opts.functional = "pbe0"
    opts.conv_tol_energy = 1.0e-12
    opts.max_iter = 100
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    result = vq.run_rks_periodic_multi_k_ewald3d(
        system, basis, kmesh, opts
    )
    assert result.converged
    return system, basis, kmesh, opts, result


@pytest.mark.slow
def test_pbe0_single_gamma_gradient_matches_own_energy_fd():
    """RKS single Gamma keeps corrected periodic, not molecular, exchange."""
    h = 1.0e-3
    system, basis, kmesh, opts, result = _run_h2_box_pbe0()
    analytic = np.asarray(
        vq.compute_gradient_periodic_rks_multi_k(
            system,
            basis,
            result,
            kmesh,
            grid_options=opts.grid,
        )
    )
    plus = _run_h2_box_pbe0(+h)[-1].energy
    minus = _run_h2_box_pbe0(-h)[-1].energy
    fd_atom0_z = (plus - minus) / (2.0 * h)

    np.testing.assert_allclose(analytic[0, 2], fd_atom0_z, atol=2.0e-6)
    np.testing.assert_allclose(analytic[1, 2], -fd_atom0_z, atol=2.0e-6)


# ---------------------------------------------------------------------------
# 3. Newton's-3rd-law on multi-k H₂ chain
# ---------------------------------------------------------------------------

def test_h2_chain_multi_k_newtons_third_law():
    """H₂ chain at 8-Å spacing with 2×1×1 multi-k: forces equal and
    opposite along the periodic axis."""
    R = 1.0 * ANGSTROM_TO_BOHR
    a_per = 8.0 * ANGSTROM_TO_BOHR
    vac = 20.0 * ANGSTROM_TO_BOHR
    lat = np.diag([a_per, vac, vac])
    atoms = [
        vq.Atom(1, [0, 0, 0]),
        vq.Atom(1, [0, 0, R]),
    ]
    sys = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 100

    kmesh = vq.monkhorst_pack(sys, [2, 1, 1])
    r = vq.run_rhf_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    assert r.converged
    g = vq.compute_gradient_periodic_rhf_multi_k(
        sys, basis, r, kmesh, lattice_opts=opts.lattice_opts)

    # Newton's 3rd along z (the bond direction; periodic axis is x).
    np.testing.assert_allclose(g[0, 2], -g[1, 2], atol=1e-9)
    # Transverse forces (x, y) vanish by symmetry on z-aligned bond.
    np.testing.assert_allclose(g[:, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(g[:, 1], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. Multi-k GGA XC Pulay — lattice-summed primitive (full GGA σ-piece)
#
#    Regression guard for the reroute of the multi-k RKS XC Pulay from the
#    former molecular-grid fallback (LDA σ-piece only) to the lattice-summed
#    periodic primitive ``xc_lattice_gradient_contribution`` shared with the
#    Γ-only G1b path. Before the reroute, multi-k PBE sat ~2e-2 Ha/bohr off
#    the molecular analytic gradient (dropped GGA σ-coupled Pulay term);
#    after, it agrees to the multi-k HF-ish overlap-Lagrangian floor
#    (~1.6e-3, a *separate* pre-existing residual tracked on the multi-k
#    HF assembly — NOT the XC piece).
# ---------------------------------------------------------------------------

def _h2o_pbe_box():
    big = 20.0 * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.85 * ANGSTROM_TO_BOHR, 0.6 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.85 * ANGSTROM_TO_BOHR, 0.6 * ANGSTROM_TO_BOHR]),
    ]
    sys = vq.PeriodicSystem(3, np.diag([big] * 3), atoms)
    return sys, atoms


@pytest.mark.slow
def test_pbe_multi_k_gga_sigma_pulay_matches_molecular():
    """Multi-k PBE gradient picks up the GGA σ-coupled XC Pulay piece via
    the lattice primitive. Tolerance is set above the multi-k HF-ish
    overlap-Lagrangian floor (~1.6e-3) but far below the ~2e-2 error the
    dropped GGA σ-piece produced, so this fails if the XC Pulay regresses
    to the LDA-only fallback."""
    sys, atoms = _h2o_pbe_box()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicKSOptions()
    opts.functional = "pbe"
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 100
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    r = vq.run_rks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    assert r.converged
    g_mk = np.asarray(vq.compute_gradient_periodic_rks_multi_k(
        sys, basis, r, kmesh, lattice_opts=opts.lattice_opts))

    mol = vq.Molecule(atoms, charge=0, multiplicity=1)
    basis_m = vq.BasisSet(mol, "sto-3g")
    rks_opts = vq.RKSOptions()
    rks_opts.functional = "pbe"
    rks_opts.conv_tol_energy = 1e-12
    rks_m = vq.run_rks(mol, basis_m, rks_opts)
    g_m = np.asarray(vq.compute_gradient_rks(mol, basis_m, rks_m))

    np.testing.assert_allclose(g_mk, g_m, atol=5e-3,
        err_msg="multi-k PBE gradient diverges from molecular analytic — "
                "GGA σ-coupled XC Pulay piece missing (LDA-only fallback "
                "regression)")


def test_pbe_multi_k_non_sto3g_basis_runs():
    """A non-STO-3G basis (6-31g) multi-k PBE gradient runs and returns a
    finite (n_atoms, 3) array. The former molecular XC-Pulay fallback hard-
    coded an STO-3G Z→nbf table and raised NotImplementedError on any other
    basis; the lattice primitive derives shell→atom from the basis itself."""
    R = 1.0 * ANGSTROM_TO_BOHR
    big = 20.0 * ANGSTROM_TO_BOHR
    atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, R])]
    sys = vq.PeriodicSystem(3, np.diag([big] * 3), atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "6-31g")
    assert basis.nbasis > 2, "6-31g H2 should have >2 basis functions"
    opts = vq.PeriodicKSOptions()
    opts.functional = "pbe"
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 100
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    r = vq.run_rks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    assert r.converged
    g = np.asarray(vq.compute_gradient_periodic_rks_multi_k(
        sys, basis, r, kmesh, lattice_opts=opts.lattice_opts))
    assert g.shape == (2, 3)
    assert np.all(np.isfinite(g))


# ---------------------------------------------------------------------------
# 5. Un-converged input rejected
# ---------------------------------------------------------------------------

def test_unconverged_rejected():
    R = 1.0 * ANGSTROM_TO_BOHR
    big_box = 20.0 * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(1, [0, 0, 0]),
        vq.Atom(1, [0, 0, R]),
    ]
    sys = vq.PeriodicSystem(3, np.diag([big_box]*3), atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 1
    opts.conv_tol_energy = 1e-30
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    r = vq.run_rhf_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    assert r.converged is False
    with pytest.raises(ValueError, match="not converged"):
        vq.compute_gradient_periodic_rhf_multi_k(
            sys, basis, r, kmesh, lattice_opts=opts.lattice_opts)
