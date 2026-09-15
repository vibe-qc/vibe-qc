"""Phase 12e-c-4a/F2: ω-invariance of the Ewald-3D Hartree J.

The deep contract of the Ewald split is that the total Hartree matrix
is independent of the splitting parameter ω. The production
``build_j_ewald_3d`` path now enforces that contract by evaluating the
periodic ``G != 0`` Hartree matrix directly from the analytical
AO-pair Fourier transform; the historical SR + FFT-Poisson split is
kept only behind ``VIBEQC_J_EWALD3D_BACKEND=grid`` for diagnostics.

For a molecular-limit cell (molecule wrapped in a large-vacuum
periodic box), the composed J differs from the isolated-molecule
J_full by a **scalar-times-overlap shift** ``α · S``, where α is the
Makov-Payne-like correction for the electron density's interaction
with its uniform compensating background. The key observation: α is
**ω-independent** — the shift is a geometric/electrostatic property of
the cell, not of the Ewald decomposition.

Two witnesses exercised here:

1. **ω-invariance of J** (the Ewald identity / F2 regression).
2. **Makov-Payne interpretation**: for a cubic cell, the scalar α ≈
   -α_M · Q / L where α_M ≈ 2.837 is the simple-cubic Madelung
   constant.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2_sysp(box: float = 30.0):
    """Molecular-limit H2 in a cubic box of side ``box`` (bohr),
    centered at (box/2, box/2, box/2)."""
    center = box / 2.0
    atoms = [
        vq.Atom(1, [center, center, center - 0.7]),
        vq.Atom(1, [center, center, center + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    mol = sysp.unit_cell_molecule()
    P = np.asarray(vq.run_rhf(mol, basis).density)
    return sysp, basis, mol, P


def _h2o_sysp(box: float = 30.0):
    """Molecular-limit H2O in a cubic box, for a non-zero-dipole test
    that exercises a slightly more complex density."""
    center = box / 2.0
    atoms = [
        vq.Atom(8, [center, center, center]),
        vq.Atom(1, [center, center + 1.4992, center - 1.1594]),
        vq.Atom(1, [center, center - 1.4992, center - 1.1594]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    mol = sysp.unit_cell_molecule()
    P = np.asarray(vq.run_rhf(mol, basis).density)
    return sysp, basis, mol, P


# ---------------------------------------------------------------------------
# ω-invariance (the Ewald identity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("box", [24.0, 30.0])
def test_ewald_identity_is_omega_invariant_h2(box):
    """The composed J_SR(ω) + J_LR(ω) is independent of ω. We check
    Frobenius-norm stability across a ~7× range of ω.

    At a 0.3-bohr grid and a 30-bohr cubic box the residual is
    ~2 × 10⁻³ of |J|_F. The residual is dominated by higher-order
    Makov–Payne terms (L⁻³ periodic-image couplings that scale with
    the density's second moment); it is *not* a grid-resolution
    artifact — finer grids don't reduce it. Saunders–Dovesi multipolar
    splitting (Phase 12e-c-3c) will cut it further by handling the
    far-field analytically.
    """
    sysp, basis, _, P = _h2_sysp(box=box)
    spacing = 0.3

    J_ref = None
    for omega in (0.3, 0.5, 1.0, 2.0):
        J = vq.build_j_ewald_3d(
            basis, sysp, P, omega=omega, spacing_bohr=spacing,
        )
        if J_ref is None:
            J_ref = J
        else:
            rel_err = np.linalg.norm(J - J_ref) / np.linalg.norm(J_ref)
            # At this box/grid the Ewald identity holds to 0.3% —
            # limited by finite-box effects, not the Poisson or FFT
            # numerics. Saunders–Dovesi will tighten this to <1e-4.
            assert rel_err < 5e-3, (
                f"Ewald identity failed at ω={omega} "
                f"(rel_err = {rel_err:.3e})"
            )


def test_ewald_identity_omega_invariant_h2o():
    """Same ω-invariance check on H2O. This used to fail when J_LR was
    projected through an FFT-Poisson collocation grid that could not
    resolve the STO-3G oxygen 1s core; the analytical AO-pair-FT
    Hartree path has no real-space grid aliasing."""
    sysp, basis, _, P = _h2o_sysp(box=24.0)

    J_ref = vq.build_j_ewald_3d(basis, sysp, P, omega=0.5, spacing_bohr=0.3)
    J_alt = vq.build_j_ewald_3d(basis, sysp, P, omega=1.5, spacing_bohr=0.3)
    rel_err = np.linalg.norm(J_alt - J_ref) / np.linalg.norm(J_ref)
    assert rel_err < 5e-3, f"rel_err = {rel_err:.3e}"


# ---------------------------------------------------------------------------
# Makov–Payne interpretation
# ---------------------------------------------------------------------------

def test_composed_minus_full_is_scalar_times_S():
    """``J_SR + J_LR - J_full`` is (to numerical precision) a scalar
    multiple of the overlap matrix S. Verifies the residual after
    subtracting the best-fit c·S is numerical noise — i.e. the only
    missing physics in the periodic composition vs the isolated
    molecular limit is the Makov-Payne constant shift."""
    sysp, basis, mol, P = _h2_sysp(box=30.0)
    S = np.asarray(vq.compute_overlap(basis))
    eri = vq.compute_eri(basis)
    J_full_iso = np.einsum("mnls,ls->mn", eri, P)

    J_periodic = vq.build_j_ewald_3d(
        basis, sysp, P, omega=0.5, spacing_bohr=0.3,
    )
    diff = J_periodic - J_full_iso
    c = float((S * diff).sum() / (S * S).sum())
    residual = np.linalg.norm(diff - c * S)
    assert residual < 5e-4, (
        f"residual after c·S subtraction = {residual:.3e}, "
        f"c = {c:.6f}"
    )


def test_makov_payne_coefficient_matches_observed_shift():
    """For a cubic cell the analytically-predicted Makov-Payne
    coefficient ``-α_M · Q / L`` matches the empirically-fitted shift
    to within a percent."""
    box = 30.0
    sysp, basis, mol, P = _h2_sysp(box=box)
    S = np.asarray(vq.compute_overlap(basis))
    Q = float(np.trace(P @ S))     # electron count = 2 for H2

    eri = vq.compute_eri(basis)
    J_full_iso = np.einsum("mnls,ls->mn", eri, P)
    J_periodic = vq.build_j_ewald_3d(
        basis, sysp, P, omega=0.5, spacing_bohr=0.3,
    )
    diff = J_periodic - J_full_iso
    alpha_observed = float((S * diff).sum() / (S * S).sum())

    alpha_predicted = vq.makov_payne_coefficient_cubic(Q, box)
    rel_err = abs(alpha_observed - alpha_predicted) / abs(alpha_predicted)
    assert rel_err < 0.02, (
        f"observed α = {alpha_observed:.6f}, "
        f"predicted α = {alpha_predicted:.6f}, "
        f"rel_err = {rel_err:.3e}"
    )


def test_makov_payne_scales_with_q_and_inverse_l():
    """Direct unit checks on the helper: doubling Q doubles |α|,
    halving L (box side) doubles |α|."""
    a0 = vq.makov_payne_coefficient_cubic(Q=2.0, L=30.0)
    a_2Q = vq.makov_payne_coefficient_cubic(Q=4.0, L=30.0)
    a_halfL = vq.makov_payne_coefficient_cubic(Q=2.0, L=15.0)
    assert a_2Q == pytest.approx(2.0 * a0, rel=1e-14)
    assert a_halfL == pytest.approx(2.0 * a0, rel=1e-14)


def test_composed_J_is_symmetric():
    sysp, basis, _, P = _h2_sysp()
    J = vq.build_j_ewald_3d(basis, sysp, P, omega=0.5)
    assert np.allclose(J, J.T, atol=1e-12)


def test_non_orthorhombic_lattice_supported():
    """The composed builder supports skew cells via the FFT metric."""
    atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    lat = np.array([
        [30.0, 2.0, 0.0],
        [0.0, 30.0, 0.0],
        [0.0, 0.0, 30.0],
    ])
    sysp = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    mol = sysp.unit_cell_molecule()
    P = np.asarray(vq.run_rhf(mol, basis).density)
    J = vq.build_j_ewald_3d(
        basis, sysp, P, omega=0.5, grid_shape=(8, 8, 8),
    )
    assert J.shape == P.shape
    assert np.all(np.isfinite(J))
    assert np.allclose(J, J.T, atol=1e-10)


# ---------------------------------------------------------------------------
# GRID-BACKEND-CONVERGES-WRONG: the G=0 finite part of the erf kernel.
#
# The erf kernel's transform is
#
#     FT[erf(w r)/r](G) = 4 pi / G^2  -  pi / w^2  +  O(G^2)
#
# and ``solve_poisson_erf_screened`` pins the WHOLE G=0 component to zero.
# That is right for the 4 pi / G^2 divergence (cancelled by the neutralising
# background) but it also discards the FINITE -pi / w^2. Being a constant
# potential, the omission enters J as a multiple of the overlap, so the
# composed identity J_SR + J_LR = J_full failed by exactly
#
#     dE = pi Q^2 / (2 w^2 V_cell)
#
# Measured 2026-08-03 on H2/STO-3G in a 12-bohr box at the fixed molecular
# RHF density, composed minus analytic_ft, w = 0.20 / 0.35 / 0.50 / 0.80:
# +9.089e-2 / +2.968e-2 / +1.454e-2 / +5.681e-3 Ha -- exactly w^-2, and
# 1.454e-2 at w = 0.5 is the 14.5 mHa the grid backend disagreed with
# analytic_ft by through the SCF. It is NOT discretisation: the gap is flat
# in the grid spacing.
#
# The restoration is opt-in (``restore_g0_finite_part``) because J_LR ALONE
# is legitimately wanted in the pinned gauge -- in it J_LR -> 0 as w -> 0,
# which ``test_small_omega_limit_collapses_to_zero`` pins. Composition sites
# that assert J_SR + J_LR = J_full pass True.
def _h2_box(L=12.0):
    atoms = [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
             vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])]
    sysp = vq.PeriodicSystem(3, np.eye(3) * L, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    D = np.asarray(vq.run_rhf(sysp.unit_cell_molecule(), basis).density)
    return sysp, basis, D


def _composed_minus_full(sysp, basis, D, omega, restore, spacing=0.15):
    from vibeqc import _vibeqc_core as core
    from vibeqc.ewald_composed import compute_j_ewald_3d_ft_lattice
    from vibeqc.periodic_density import build_j_long_range_periodic

    lo = core.LatticeSumOptions()
    D_set = core.compute_overlap_lattice(basis, sysp, lo)
    for i in range(len(D_set)):
        D_set.set_block(i, D if i == 0 else np.zeros_like(D))
    J_full = compute_j_ewald_3d_ft_lattice(
        basis, sysp, D_set, omega, lattice_opts=lo)
    J_sr = vq.build_fock_2e_real_space(basis, sysp, lo, D_set, 0.0, omega)
    J_lr = build_j_long_range_periodic(
        basis, sysp, D_set, omega=omega, spacing_bohr=spacing,
        output_cells=list(range(len(D_set))),
        restore_g0_finite_part=restore,
    )
    f0 = np.asarray(J_full.blocks[0] if hasattr(J_full, "blocks")
                    else J_full[0])
    c0 = np.asarray(J_sr.blocks[0] if hasattr(J_sr, "blocks")
                    else J_sr[0]) + np.asarray(J_lr[0])
    return 0.5 * float(np.einsum("ij,ij->", D, c0 - f0))


@pytest.mark.slow
def test_g0_finite_part_closes_the_composed_identity():
    """With the term restored, J_SR + J_LR reproduces J_full."""
    sysp, basis, D = _h2_box()
    for omega in (0.35, 0.5, 0.8):
        err = _composed_minus_full(sysp, basis, D, omega, restore=True)
        assert abs(err) < 1e-9, (
            f"composed identity broken at w={omega}: {err:+.3e} Ha"
        )


@pytest.mark.slow
def test_g0_omission_is_pi_Q2_over_2w2V_and_scales_as_w_minus_2():
    """Without the term the error is the analytic G=0 self-energy.

    Pinning the closed form (not just 'it is wrong') is what makes this a
    root-cause regression: a future change that reintroduces a DIFFERENT
    error of similar size would not reproduce the w^-2 law or the
    Q^2 / V_cell prefactor.
    """
    L = 12.0
    sysp, basis, D = _h2_box(L)
    from vibeqc import _vibeqc_core as core

    lo = core.LatticeSumOptions()
    S0 = np.asarray(core.compute_overlap_lattice(basis, sysp, lo).blocks[0])
    Q = float(np.einsum("ij,ij->", D, S0))
    V = L ** 3
    for omega in (0.35, 0.5, 0.8):
        err = _composed_minus_full(sysp, basis, D, omega, restore=False)
        predicted = np.pi * Q ** 2 / (2.0 * omega ** 2 * V)
        assert err == pytest.approx(predicted, rel=2e-4), (
            f"w={omega}: observed {err:.6e} vs closed form {predicted:.6e}"
        )


def test_j_lr_default_keeps_the_pinned_g0_gauge():
    """The default is unchanged: callers wanting J_LR alone still get the
    pinned-G=0 object, which is what makes J_LR -> 0 as w -> 0 hold."""
    from vibeqc import _vibeqc_core as core
    from vibeqc.periodic_density import build_j_long_range_periodic

    sysp, basis, D = _h2_box()
    lo = core.LatticeSumOptions()
    D_set = core.compute_overlap_lattice(basis, sysp, lo)
    for i in range(len(D_set)):
        D_set.set_block(i, D if i == 0 else np.zeros_like(D))
    kw = dict(omega=0.5, spacing_bohr=0.30,
              output_cells=list(range(len(D_set))))
    default = np.asarray(build_j_long_range_periodic(
        basis, sysp, D_set, **kw)[0])
    explicit = np.asarray(build_j_long_range_periodic(
        basis, sysp, D_set, restore_g0_finite_part=False, **kw)[0])
    np.testing.assert_allclose(default, explicit, atol=0.0, rtol=0.0)
    restored = np.asarray(build_j_long_range_periodic(
        basis, sysp, D_set, restore_g0_finite_part=True, **kw)[0])
    assert not np.allclose(default, restored), (
        "the opt-in must actually change J_LR"
    )
