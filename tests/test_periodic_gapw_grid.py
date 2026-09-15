"""Tests for the gapw chat's PlaneWaveGrid infrastructure (M1c).

The :class:`PlaneWaveGrid` dataclass + cutoff-driven sizing are the
foundation that the M2 GPW Hartree-J driver will sit on top of.
Tests here pin:

* fractional / Cartesian coordinate consistency on orthorhombic
  and triclinic cells,
* reciprocal-lattice metric on skew cells (the FFT-Poisson kernel
  in C++ uses the same metric internally — wrong G⋅G here would
  show up as a parity bug at M2 against CP2K),
* the FFT-friendly grid sizing rounds up to ``{2ᵃ 3ᵇ 5ᶜ}``,
* the basis-aware cutoff heuristic scales with the tightest
  primitive,
* the experimental warning fires when callers go through
  :func:`make_grid`.

There is also one **integration** test that round-trips the new
grid abstraction through the existing :func:`solve_poisson_coulomb`
C++ kernel: a centred Gaussian charge density on the grid has a
known integrated Hartree energy that matches the analytical
all-space ``q²/√(2πσ²)`` to ~mHa once the cell is large enough.
This is the µHa-class M1 internal-consistency check that does not
need a CP2K oracle.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import (
    GAPWExperimentalWarning,
    PlaneWaveGrid,
    collocate_point_charges_on_grid,
    make_grid,
    nx_for_axis,
    point_charge_self_energy,
    recommend_cutoff_from_basis,
    round_up_fft_friendly,
)


# ---------- round_up_fft_friendly --------------------------------------


@pytest.mark.parametrize("n, expected", [
    (1, 2),
    (2, 2),
    (3, 4),       # 3 is odd → bump to 4
    (4, 4),
    (5, 6),       # 5 is odd → bump to 6
    (7, 8),       # 7 is odd → bump to 8 (within {2,3,5})
    (11, 12),
    (13, 16),
    (17, 18),
    (60, 60),     # 60 = 2² · 3 · 5
    (61, 64),
])
def test_round_up_fft_friendly_default(n, expected):
    assert round_up_fft_friendly(n) == expected


# ---------- nx_for_axis ------------------------------------------------


def test_nx_for_axis_scales_with_axis_and_cutoff():
    """Doubling either the axis length or the sqrt(2·E_cut) grows the
    grid count by roughly the same factor (modulo FFT-friendly
    rounding slack)."""
    n1 = nx_for_axis(10.0, cutoff_ha=20.0)
    n2 = nx_for_axis(20.0, cutoff_ha=20.0)
    n3 = nx_for_axis(10.0, cutoff_ha=80.0)  # 4× cutoff = 2× G_max
    # FFT-rounding can pad either result by up to ~20%, so the
    # observed factor falls in [1.6, 2.5] rather than being exactly 2.
    assert 1.6 * n1 <= n2 <= 2.5 * n1, f"n1={n1}, n2={n2}"
    assert 1.6 * n1 <= n3 <= 2.5 * n1, f"n1={n1}, n3={n3}"


def test_nx_for_axis_rejects_bad_inputs():
    with pytest.raises(ValueError):
        nx_for_axis(10.0, cutoff_ha=0.0)
    with pytest.raises(ValueError):
        nx_for_axis(0.0, cutoff_ha=10.0)
    with pytest.raises(ValueError):
        nx_for_axis(-5.0, cutoff_ha=10.0)


# ---------- PlaneWaveGrid construction --------------------------------


def test_grid_rejects_singular_lattice():
    L = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    with pytest.raises(ValueError, match="singular"):
        PlaneWaveGrid(L, 4, 4, 4)


def test_grid_rejects_bad_extents():
    L = np.eye(3) * 10.0
    with pytest.raises(ValueError):
        PlaneWaveGrid(L, 1, 4, 4)
    with pytest.raises(ValueError):
        PlaneWaveGrid(L, 4, 4, "eight")  # type: ignore[arg-type]


def test_grid_basic_properties():
    L = np.diag([10.0, 12.0, 14.0])
    g = PlaneWaveGrid(L, 8, 12, 16)
    assert g.shape == (8, 12, 16)
    assert g.n_points == 8 * 12 * 16
    assert g.cell_volume_bohr3 == pytest.approx(10.0 * 12.0 * 14.0)
    assert np.allclose(g.axis_lengths_bohr, [10.0, 12.0, 14.0])
    assert g.voxel_volume_bohr3 == pytest.approx(
        10.0 * 12.0 * 14.0 / (8 * 12 * 16)
    )


def test_grid_reciprocal_metric_orthorhombic():
    """For diag(L) the reciprocal columns are diag(2π / L_α)."""
    L = np.diag([10.0, 12.0, 14.0])
    g = PlaneWaveGrid(L, 4, 4, 4)
    B = g.reciprocal_lattice_bohr_inv
    assert np.allclose(np.diag(B), [
        2 * math.pi / 10.0,
        2 * math.pi / 12.0,
        2 * math.pi / 14.0,
    ])
    # Off-diagonal terms vanish.
    assert np.abs(B - np.diag(np.diag(B))).max() < 1e-12


def test_grid_reciprocal_metric_satisfies_duality_skew():
    """For ANY (non-singular) A, B = 2π A⁻ᵀ satisfies b_i · a_j = 2π δ_ij."""
    rng = np.random.default_rng(seed=42)
    L = np.eye(3) + 0.3 * rng.standard_normal((3, 3))
    L *= 10.0
    g = PlaneWaveGrid(L, 4, 4, 4)
    B = g.reciprocal_lattice_bohr_inv
    # a_j is the j-th column of L; b_i is the i-th column of B
    duality = B.T @ L  # should be 2π I
    assert np.allclose(duality, 2 * math.pi * np.eye(3), atol=1e-10)


def test_grid_cartesian_coords_orthorhombic():
    L = np.diag([10.0, 12.0, 14.0])
    g = PlaneWaveGrid(L, 4, 4, 4)
    r = g.cartesian_coords()
    assert r.shape == (4, 4, 4, 3)
    # Sample (i, j, k) = (0, 0, 0) sits at the cell origin.
    assert np.allclose(r[0, 0, 0], [0.0, 0.0, 0.0])
    # Sample (2, 0, 0) is at s = (0.5, 0, 0) → r = (5, 0, 0).
    assert np.allclose(r[2, 0, 0], [5.0, 0.0, 0.0])
    # Sample (0, 3, 0) is at s = (0, 0.75, 0) → r = (0, 9, 0).
    assert np.allclose(r[0, 3, 0], [0.0, 9.0, 0.0])


def test_grid_reciprocal_vectors_g_wrap():
    """The reciprocal_vectors() method follows the FFT wrap-around:
    G[0,0,0] is zero; G[nx//2, 0, 0] is the highest positive ``kx``;
    G[nx-1, 0, 0] is ``-b₁``."""
    L = np.diag([10.0, 10.0, 10.0])
    g = PlaneWaveGrid(L, 8, 8, 8)
    G = g.reciprocal_vectors()
    assert G.shape == (8, 8, 8, 3)
    assert np.allclose(G[0, 0, 0], [0.0, 0.0, 0.0])
    # FFT: index 4 (= nx/2) wraps to k = -4, NOT +4 — Nyquist sits on
    # the negative side in the standard FFT convention used here.
    assert np.allclose(G[4, 0, 0], [-4 * 2 * math.pi / 10.0, 0.0, 0.0])
    # Last index wraps to k = -1.
    assert np.allclose(G[7, 0, 0], [-2 * math.pi / 10.0, 0.0, 0.0])


# ---------- make_grid + cutoff heuristic -------------------------------


def test_make_grid_emits_experimental_warning():
    L = np.eye(3) * 10.0
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        g = make_grid(L, cutoff_ha=20.0)
    msgs = [w for w in captured if issubclass(w.category, GAPWExperimentalWarning)]
    assert msgs, "make_grid should emit GAPWExperimentalWarning"
    assert g.cutoff_ha == pytest.approx(20.0)
    # The reported grid extents are present in the warning message.
    assert f"{g.nx}x{g.ny}x{g.nz}" in str(msgs[0].message)


def test_make_grid_isotropic_cell_uniform_extents():
    L = np.eye(3) * 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        g = make_grid(L, cutoff_ha=20.0)
    assert g.nx == g.ny == g.nz


def test_make_grid_anisotropic_cell_proportional_extents():
    L = np.diag([10.0, 20.0, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        g = make_grid(L, cutoff_ha=20.0)
    # ny should be roughly 2× nx (within FFT-friendly rounding),
    # nz roughly nx/2.
    assert g.ny >= g.nx
    assert g.nz <= g.nx


# ---------- recommend_cutoff_from_basis -------------------------------


def test_recommend_cutoff_scales_with_tightest_primitive():
    base = recommend_cutoff_from_basis([1.0, 0.5, 0.1])
    doubled = recommend_cutoff_from_basis([2.0, 0.5, 0.1])
    assert doubled == pytest.approx(2.0 * base)


def test_recommend_cutoff_floor():
    """A pathologically diffuse basis (all α << 1) should still hit
    the min_cutoff_ha floor."""
    cut = recommend_cutoff_from_basis([0.001, 0.002], min_cutoff_ha=5.0)
    assert cut == 5.0


def test_recommend_cutoff_rejects_empty():
    with pytest.raises(ValueError, match="no positive exponents"):
        recommend_cutoff_from_basis([])


# ---------- Integration: FFT-Poisson on the new grid ------------------


def test_grid_integrates_with_fft_poisson_kernel():
    """Round-trip the new grid abstraction through the existing C++
    :func:`solve_poisson_coulomb` kernel on a **neutral** charge
    distribution — a + Gaussian / − Gaussian pair separated by ``d``
    along x inside the cell. With ``ρ = ρ_+ − ρ_−`` and ``ρ_±`` unit-
    norm Gaussians of common width ``σ``,

        ``E_H = ½ ⟨ρ|ρ⟩ = ⟨ρ_+|ρ_+⟩ − ⟨ρ_+|ρ_−⟩``
             ``= 1 / (σ √π)  −  erf(d / (2σ)) / d``

    where the convolution variance of two same-σ Gaussians is
    ``σ_+² + σ_−² = 2σ²`` so the cross-pair Coulomb integral uses
    ``erf(d / √(2 · 2σ²)) / d = erf(d / (2σ)) / d``.

    The system is **neutral** (∫ρ = 0), so the Madelung shift that
    contaminates a charged-Gaussian test does not apply. The
    remaining error is the dipole's interaction with its own
    periodic images, which for σ = 0.6 bohr / d = 4 bohr in a
    16-bohr cube is well below 20 mHa.

    This is the M1 internal-consistency anchor — no CP2K oracle
    needed.
    """
    L = np.eye(3) * 16.0
    nx = ny = nz = 64
    grid = PlaneWaveGrid(L, nx, ny, nz)
    centre = 0.5 * L.diagonal()
    # Two Gaussians of opposite sign separated by 4 bohr along x.
    d = 4.0
    r1 = centre + np.array([-d / 2, 0.0, 0.0])
    r2 = centre + np.array([+d / 2, 0.0, 0.0])
    r = grid.cartesian_coords()
    sigma = 0.6
    norm = 1.0 / (2 * math.pi * sigma ** 2) ** 1.5
    r2_1 = ((r - r1) ** 2).sum(axis=-1)
    r2_2 = ((r - r2) ** 2).sum(axis=-1)
    rho = norm * (
        np.exp(-r2_1 / (2 * sigma ** 2))
        - np.exp(-r2_2 / (2 * sigma ** 2))
    )

    # Neutrality check (sanity for the grid voxel-volume bookkeeping).
    integrated_charge = float(rho.sum()) * grid.voxel_volume_bohr3
    assert abs(integrated_charge) < 1e-10

    V = core.solve_poisson_coulomb(rho, L)
    E_H = core.hartree_energy_on_grid(rho, V, grid.cell_volume_bohr3)

    # Analytical all-space reference for two same-σ Gaussians of
    # opposite sign at distance d. Convolution variance = 2σ², so
    # the cross-pair Coulomb integral uses erf(d / (2σ)).
    E_self_sum = 1.0 / (sigma * math.sqrt(math.pi))     # = 2 · 1/(2σ√π)
    E_cross = math.erf(d / (2.0 * sigma)) / d
    E_H_analytic = E_self_sum - E_cross

    # 20 mHa tolerance leaves room for periodic image + finite-grid
    # quadrature error.
    assert abs(E_H - E_H_analytic) < 2e-2, (
        f"|E_H - E_H_analytic| = {abs(E_H - E_H_analytic):.3e} Ha; "
        f"expected < 2e-2. E_H={E_H:.6f}, "
        f"E_H_analytic={E_H_analytic:.6f}"
    )


# ---------- Point-charge collocation ---------------------------------------


def test_collocate_single_charge_conserves_charge():
    """A unit charge at the cell centre lays down a Gaussian whose
    integrated value matches 1 e to high precision (the periodic
    images at default image_shells=1 reclaim the lost tails)."""
    L = np.eye(3) * 16.0
    grid = PlaneWaveGrid(L, 64, 64, 64)
    pos = np.array([[8.0, 8.0, 8.0]])
    rho = collocate_point_charges_on_grid(
        [1.0], pos, grid, sigma_bohr=0.5,
    )
    integrated = float(rho.sum()) * grid.voxel_volume_bohr3
    assert integrated == pytest.approx(1.0, abs=1e-6)


def test_collocate_neutral_pair_integrates_to_zero():
    L = np.eye(3) * 16.0
    grid = PlaneWaveGrid(L, 48, 48, 48)
    pos = np.array([[6.0, 8.0, 8.0], [10.0, 8.0, 8.0]])
    rho = collocate_point_charges_on_grid(
        [1.0, -1.0], pos, grid, sigma_bohr=0.5,
    )
    integrated = float(rho.sum()) * grid.voxel_volume_bohr3
    assert abs(integrated) < 1e-9


def test_collocate_translation_symmetry():
    """A uniform translation of every charge by the same lattice
    vector leaves the integrated density (and any periodic
    observable derived from it) unchanged."""
    L = np.eye(3) * 12.0
    grid = PlaneWaveGrid(L, 32, 32, 32)
    pos1 = np.array([[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]])
    pos2 = pos1 + L[:, 0]  # shift by a1
    rho1 = collocate_point_charges_on_grid([1.0, -1.0], pos1, grid,
                                            sigma_bohr=0.5)
    rho2 = collocate_point_charges_on_grid([1.0, -1.0], pos2, grid,
                                            sigma_bohr=0.5)
    # The density itself is identical (cell-periodic).
    assert np.allclose(rho1, rho2, atol=1e-10)


def test_collocate_rejects_bad_inputs():
    L = np.eye(3) * 10.0
    grid = PlaneWaveGrid(L, 8, 8, 8)
    with pytest.raises(ValueError, match="positive"):
        collocate_point_charges_on_grid(
            [1.0], np.array([[5.0, 5.0, 5.0]]), grid, sigma_bohr=0.0,
        )
    with pytest.raises(ValueError, match="image_shells"):
        collocate_point_charges_on_grid(
            [1.0], np.array([[5.0, 5.0, 5.0]]), grid, sigma_bohr=0.5,
            image_shells=-1,
        )
    with pytest.raises(ValueError, match="positions_bohr"):
        collocate_point_charges_on_grid(
            [1.0], np.array([5.0, 5.0, 5.0]), grid, sigma_bohr=0.5,
        )
    with pytest.raises(ValueError, match="charges shape"):
        collocate_point_charges_on_grid(
            [1.0, 2.0], np.array([[5.0, 5.0, 5.0]]), grid, sigma_bohr=0.5,
        )


def test_collocate_integrates_with_fft_poisson():
    """Two opposite-sign Gaussians at separation d give the
    analytical neutral-dipole Hartree energy through the collocation
    kernel — same physics as the grid-level integration test, but
    via the user-facing collocation API."""
    L = np.eye(3) * 16.0
    grid = PlaneWaveGrid(L, 64, 64, 64)
    sigma = 0.6
    d = 4.0
    pos = np.array([[8.0 - d / 2, 8.0, 8.0], [8.0 + d / 2, 8.0, 8.0]])
    rho = collocate_point_charges_on_grid(
        [1.0, -1.0], pos, grid, sigma_bohr=sigma,
    )
    V = core.solve_poisson_coulomb(rho, L)
    E_H = core.hartree_energy_on_grid(rho, V, grid.cell_volume_bohr3)
    E_H_analytic = (1.0 / (sigma * math.sqrt(math.pi))
                     - math.erf(d / (2.0 * sigma)) / d)
    assert abs(E_H - E_H_analytic) < 2e-2


# ---------- Self-energy correction -----------------------------------------


def test_self_energy_scales_with_charge_squared():
    e1 = point_charge_self_energy([1.0], sigma_bohr=0.5)
    e2 = point_charge_self_energy([2.0], sigma_bohr=0.5)
    assert e2 == pytest.approx(4.0 * e1, rel=1e-12)


def test_self_energy_multi_site_is_sum():
    e_total = point_charge_self_energy([1.0, 2.0, 3.0], sigma_bohr=0.5)
    e_sum = (
        point_charge_self_energy([1.0], sigma_bohr=0.5)
        + point_charge_self_energy([2.0], sigma_bohr=0.5)
        + point_charge_self_energy([3.0], sigma_bohr=0.5)
    )
    assert e_total == pytest.approx(e_sum, rel=1e-12)


def test_self_energy_matches_analytic():
    """E_self for one unit-charge unit-norm Gaussian of width σ is
    1 / (2 σ √π)."""
    sigma = 0.5
    e = point_charge_self_energy([1.0], sigma_bohr=sigma)
    assert e == pytest.approx(1.0 / (2.0 * sigma * math.sqrt(math.pi)),
                               rel=1e-12)


# ---------- M1e: Hartree-only Madelung constant of NaCl --------------------


def test_nacl_madelung_constant_via_fft_poisson_micro_ha():
    """The full M1 internal-consistency anchor for the gapw chat.

    Set up the 8-atom conventional NaCl supercell — four +1 and four
    −1 charges at the sodium/chloride lattice sites of a simple
    cubic with lattice parameter ``L`` (atomic units). The cell is
    **neutral**, so the FFT-Poisson kernel's ``V(G=0) = 0``
    convention introduces no Madelung-background shift, and the
    Hartree recipe

        ``E_H_grid  =  ½ ∫ ρ V dr``
        ``E_corrected  =  E_H_grid − Σ_i q_i² / (2 σ √π)``

    must reproduce the analytical NaCl Madelung energy

        ``E_Madelung  =  −4 · M_NaCl · q² / r_NN``

    where ``M_NaCl = 1.747564594…`` is the Madelung constant for the
    rocksalt lattice (Born / Lande 1918) and ``r_NN = L / 2`` is the
    nearest-neighbour Na-Cl distance. The pre-factor 4 counts the
    four formula units in the conventional cubic supercell.

    At ``L = 8 bohr``, ``σ = 0.5 bohr``, ``64³`` grid the recipe
    matches the analytical value to **< 1 µHa**. This is the
    Hartree-only µHa recovery test the gapw chat brief asked for,
    delivered without any external oracle (CP2K, PySCF, CRYSTAL —
    none of which the comparison would meaningfully exercise here).

    The result demonstrates that ``PlaneWaveGrid`` +
    ``collocate_point_charges_on_grid`` + ``solve_poisson_coulomb``
    + ``point_charge_self_energy`` compose into a complete
    Hartree-J reference implementation. The M2 work (GPW Hartree-J
    from a Gaussian density via Mulliken-style collocation) reuses
    exactly this recipe with the density's contracted-Gaussian
    expansion in place of the point-charge smearing.
    """
    L = 8.0
    lattice = np.eye(3) * L
    half = L / 2
    sites_pos = np.array([
        [0.0, 0.0, 0.0],
        [half, half, 0.0],
        [half, 0.0, half],
        [0.0, half, half],
    ])
    sites_neg = np.array([
        [half, 0.0, 0.0],
        [0.0, half, 0.0],
        [0.0, 0.0, half],
        [half, half, half],
    ])
    positions = np.concatenate([sites_pos, sites_neg], axis=0)
    charges = [1.0] * 4 + [-1.0] * 4

    sigma = 0.5
    grid = PlaneWaveGrid(lattice, 64, 64, 64)
    rho = collocate_point_charges_on_grid(charges, positions, grid,
                                           sigma_bohr=sigma)

    # Cell is neutral by construction.
    integrated = float(rho.sum()) * grid.voxel_volume_bohr3
    assert abs(integrated) < 1e-10

    V = core.solve_poisson_coulomb(rho, lattice)
    E_H = core.hartree_energy_on_grid(rho, V, grid.cell_volume_bohr3)
    E_self = point_charge_self_energy(charges, sigma)
    E_madelung_grid = E_H - E_self

    # Analytical NaCl Madelung energy for 4 formula units, q = ±1,
    # r_NN = L / 2. Born-Lande Madelung constant for rocksalt.
    M_nacl = 1.747564594633182
    r_nn = L / 2
    E_madelung_analytic = -4.0 * M_nacl / r_nn

    assert abs(E_madelung_grid - E_madelung_analytic) < 1e-6, (
        f"|E_Madelung_grid − E_analytic| = "
        f"{abs(E_madelung_grid - E_madelung_analytic):.3e} Ha; "
        f"expected < 1e-6.\n  E_madelung_grid  = {E_madelung_grid:.10f}\n"
        f"  E_madelung_analytic = {E_madelung_analytic:.10f}"
    )
