"""Tests for the gapw chat's GPW Hartree-J builder (M2).

The :class:`GpwJBuilder` composes Gaussian-density collocation,
FFT-Poisson, and AO-basis projection into a Hartree-J matrix on
the smooth real-space grid. Tests pin:

* round-trip consistency on the integrated density — collocating
  D on the grid produces ρ(r) whose integral equals ``tr(D @ S)``
  (the total number of electrons for a closed-shell density) to
  finite-grid quadrature precision,
* the J matrix is symmetric (real density → Hermitian J), and a
  zero density gives a zero J,
* convergence in the grid extent — refining ``nx`` should leave
  the integrated Hartree energy stable to ~µHa for a compact
  basis (He STO-3G is the canonical 1-AO closed-shell stress
  case),
* Madelung-shift consistency — the difference between the
  vibe-qc molecular Hartree energy and the periodic GPW Hartree
  energy on the same density matches the analytical
  ``q² · ξ_NaCl / (2 · L)`` Madelung correction within a few mHa
  on a He STO-3G test (q = 2, ξ_cubic ≈ 2.837),
* :class:`GpwJBuilder` and :func:`compute_j_via_gpw` agree
  bit-for-bit (same primitive, two surfaces).

There is no CP2K parity test here — that gates on the M2 full SCF
driver and CP2K being installed; the runner_cp2k smoke tests
already pin the M1f infrastructure. The M3a Ewald-V_ne lift
closes the convention bookkeeping for vacuum-padded neutral
cells (periodic E_HF matches molecular E_HF to < 1 mHa); M3b
(GAPW augmentation) will add the per-atom radial-grid
correction that closes the all-electron gap.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.periodic_gapw_j as gj
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import (
    GAPWExperimentalWarning,
    PlaneWaveGrid,
)
from vibeqc.periodic_gapw_j import (
    GpwJBuilder,
    collocate_density_on_grid,
    compute_j_via_gpw,
    project_potential_to_ao,
)


# Filter the warnings globally — the experimental flag is by design,
# we don't need to assert it on every call here.
pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


# ---------- Fixtures: He / H2 in a cubic box ------------------------------


def _he_in_box(L: float = 16.0):
    mol = vq.Molecule(
        [vq.Atom(2, [L / 2, L / 2, L / 2])], charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    D = np.array([[2.0]])  # He 1s², closed shell
    return mol, basis, D, np.eye(3) * L


def _h2_in_box(L: float = 16.0):
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    result = vq.run_rhf(mol, basis, opts)
    D = np.asarray(result.density)
    return mol, basis, D, np.eye(3) * L


# ---------- Density collocation ------------------------------------------


def test_collocate_density_charge_conservation():
    """∫ρ ≈ tr(D @ S) = N_electrons. For He STO-3G the basis has 1
    s-function; the overlap is unity by construction; the integrated
    density must equal 2 e."""
    _, basis, D, L = _he_in_box(16.0)
    grid = PlaneWaveGrid(L, 64, 64, 64)
    rho = collocate_density_on_grid(basis, D, grid)
    integrated = float(rho.sum()) * grid.voxel_volume_bohr3
    assert integrated == pytest.approx(2.0, rel=1e-4)


def test_collocate_density_zero_matrix_gives_zero():
    _, basis, _, L = _he_in_box(16.0)
    grid = PlaneWaveGrid(L, 32, 32, 32)
    D_zero = np.zeros((basis.nbasis, basis.nbasis))
    rho = collocate_density_on_grid(basis, D_zero, grid)
    assert np.allclose(rho, 0.0)


def test_collocate_density_rejects_shape_mismatch():
    _, basis, _, L = _he_in_box(16.0)
    grid = PlaneWaveGrid(L, 16, 16, 16)
    with pytest.raises(ValueError, match="square"):
        collocate_density_on_grid(basis, np.array([1.0, 2.0]), grid)
    with pytest.raises(ValueError, match="nbasis"):
        collocate_density_on_grid(basis, np.eye(5), grid)


def test_collocate_density_h2_integrates_to_two_electrons():
    """H2 has 2 electrons; the converged density must integrate to 2."""
    _, basis, D, L = _h2_in_box(16.0)
    grid = PlaneWaveGrid(L, 48, 48, 48)
    rho = collocate_density_on_grid(basis, D, grid)
    integrated = float(rho.sum()) * grid.voxel_volume_bohr3
    assert integrated == pytest.approx(2.0, rel=1e-3)


# ---------- Periodic image sum: translation invariance (audit 2026-05-31) -


def _h2_corner_and_center(L: float = 10.0, d: float = 1.4):
    """H2/STO-3G in a cubic cell of side ``L`` at two placements that
    differ by an exact half-cell translation ``(L/2, L/2, L/2)``:

    * **corner** — first atom on the cell origin, so its AO sits on the
      box corner and its Gaussian tail straddles all eight corners (the
      worst case for the molecular-AO charge leak),
    * **center** — the corner placement translated by ``(L/2,L/2,L/2)``.

    The two are a rigid lattice translation of each other, so they share
    one AO-basis density matrix (the SCF density matrix is translation-
    invariant). On an even grid the translation is exactly
    ``(n/2, n/2, n/2)`` voxels, so the two collocated fields are related
    by a clean :func:`numpy.roll`. Returns
    ``(basis_corner, basis_center, D, lattice)``.
    """
    c = L / 2.0
    corner_pos = [[0.0, 0.0, 0.0], [0.0, 0.0, d]]
    center_pos = [[c, c, c], [c, c, c + d]]  # corner + (c, c, c)

    mol_corner = vq.Molecule(
        [vq.Atom(1, p) for p in corner_pos], charge=0, multiplicity=1,
    )
    mol_center = vq.Molecule(
        [vq.Atom(1, p) for p in center_pos], charge=0, multiplicity=1,
    )
    basis_corner = vq.BasisSet(mol_corner, "sto-3g")
    basis_center = vq.BasisSet(mol_center, "sto-3g")

    # The density matrix is identical for the two placements (rigid
    # translation leaves the AO-basis integrals, hence the SCF density,
    # unchanged); compute it once and reuse it for both.
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    D = np.asarray(vq.run_rhf(mol_center, basis_center, opts).density)
    return basis_corner, basis_center, D, np.eye(3) * L


def test_collocate_density_corner_atoms_integrate_to_n():
    """Regression (audit 2026-05-31): a density whose AOs are centred on
    a cell corner must still integrate to ``N = tr(D·S)`` electrons.

    Before the periodic image sum was wired into the AO evaluation, the
    corner AOs leaked their Gaussian tails out of the box and the
    density under-integrated to ~0.4 e instead of 2 e for H₂/STO-3G —
    the cube / ELF / QVF density for any structure with atoms near a
    cell face was wrong by that leak.
    """
    basis_corner, _, D, L = _h2_corner_and_center(10.0)
    S = np.asarray(core.compute_overlap(basis_corner))
    n_elec = float(np.einsum("ij,ij->", D, S))
    grid = PlaneWaveGrid(L, 60, 60, 60)
    rho = collocate_density_on_grid(basis_corner, D, grid)
    integrated = float(rho.sum()) * grid.voxel_volume_bohr3
    assert integrated == pytest.approx(n_elec, rel=1e-3)
    assert integrated == pytest.approx(2.0, rel=1e-3)


def test_collocate_density_is_translation_invariant():
    """The collocated density is invariant under a rigid lattice
    translation of the atoms. Corner and centred placements give the
    same integrated charge, and — since they differ by exactly half a
    cell on an even grid — the same density field up to an
    ``(n/2, n/2, n/2)`` voxel roll. A reference-free probe of periodic-
    image correctness: on the unfixed code the corner field carried only
    ~0.4 e while the centred field carried ~2 e, so neither the integral
    nor the rolled-field comparison held.
    """
    basis_corner, basis_center, D, L = _h2_corner_and_center(10.0)
    n = 60
    grid = PlaneWaveGrid(L, n, n, n)
    rho_corner = collocate_density_on_grid(basis_corner, D, grid)
    rho_center = collocate_density_on_grid(basis_center, D, grid)

    int_corner = float(rho_corner.sum()) * grid.voxel_volume_bohr3
    int_center = float(rho_center.sum()) * grid.voxel_volume_bohr3
    assert int_corner == pytest.approx(int_center, rel=1e-6)

    rolled = np.roll(rho_corner, (n // 2, n // 2, n // 2), axis=(0, 1, 2))
    assert np.max(np.abs(rolled - rho_center)) < 1e-5


def test_gpw_hartree_energy_is_translation_invariant():
    """The GPW Hartree energy ``½ tr(D·J)`` is a periodic observable and
    must not depend on where the atoms sit in the cell. This pins that
    the image sum reaches the J build (``GpwJBuilder._chi`` /
    :func:`project_potential_to_ao`), not only the standalone density
    collocator — on the unfixed code the corner build leaked charge and
    its Hartree energy was off by the missing-charge ``J`` (~0.5 Ha).
    """
    basis_corner, basis_center, D, L = _h2_corner_and_center(10.0)
    grid = PlaneWaveGrid(L, 60, 60, 60)
    e_corner = GpwJBuilder(basis_corner, grid).hartree_energy(D)
    e_center = GpwJBuilder(basis_center, grid).hartree_energy(D)
    assert e_corner == pytest.approx(e_center, rel=1e-6)


# ---------- Potential projection -----------------------------------------


def test_project_potential_is_symmetric():
    """For any real grid potential V, the projected M_μν must be
    symmetric — it's an integral of (real) χ_μ χ_ν V."""
    _, basis, _, L = _h2_in_box(16.0)
    grid = PlaneWaveGrid(L, 32, 32, 32)
    rng = np.random.default_rng(seed=42)
    V = rng.standard_normal(grid.shape)
    M = project_potential_to_ao(basis, V, grid)
    assert np.allclose(M, M.T, atol=1e-12)


def test_project_potential_zero_gives_zero():
    _, basis, _, L = _h2_in_box(16.0)
    grid = PlaneWaveGrid(L, 16, 16, 16)
    V = np.zeros(grid.shape)
    M = project_potential_to_ao(basis, V, grid)
    assert np.allclose(M, 0.0)


def test_project_potential_rejects_shape_mismatch():
    _, basis, _, L = _h2_in_box(16.0)
    grid = PlaneWaveGrid(L, 16, 16, 16)
    with pytest.raises(ValueError, match="potential shape"):
        project_potential_to_ao(basis, np.zeros((8, 8, 8)), grid)


# ---------- GpwJBuilder ---------------------------------------------------


def test_gpw_j_builder_returns_symmetric_matrix():
    _, basis, D, L = _h2_in_box(16.0)
    grid = PlaneWaveGrid(L, 48, 48, 48)
    builder = GpwJBuilder(basis, grid)
    J = builder.build_J(D)
    assert J.shape == (basis.nbasis, basis.nbasis)
    assert np.allclose(J, J.T, atol=1e-10)


def test_gpw_j_zero_density_gives_zero_J():
    _, basis, _, L = _h2_in_box(16.0)
    grid = PlaneWaveGrid(L, 16, 16, 16)
    builder = GpwJBuilder(basis, grid)
    J = builder.build_J(np.zeros((basis.nbasis, basis.nbasis)))
    assert np.allclose(J, 0.0)


def test_gpw_j_rejects_wrong_density_shape():
    _, basis, _, L = _he_in_box(16.0)
    grid = PlaneWaveGrid(L, 16, 16, 16)
    builder = GpwJBuilder(basis, grid)
    with pytest.raises(ValueError, match="shape"):
        builder.build_J(np.zeros((basis.nbasis + 1, basis.nbasis + 1)))


def test_gpw_j_builder_caches_ao_values():
    """A repeat build_J call should not re-evaluate the AO grid; this
    is the cache that makes the SCF cost-amortised."""
    _, basis, D, L = _he_in_box(16.0)
    grid = PlaneWaveGrid(L, 24, 24, 24)
    builder = GpwJBuilder(basis, grid, cache_ao_values=True)
    assert builder._chi_cache is None  # not populated until first call
    builder.build_J(D)
    assert builder._chi_cache is not None
    cached = builder._chi_cache
    builder.build_J(D)
    # Same object (no recomputation).
    assert builder._chi_cache is cached


def test_compute_j_via_gpw_matches_builder():
    """The one-shot ``compute_j_via_gpw`` and the cached
    :class:`GpwJBuilder` must agree bit-for-bit on a fixed
    (basis, D, grid)."""
    _, basis, D, L = _h2_in_box(16.0)
    grid = PlaneWaveGrid(L, 32, 32, 32)
    J_one_shot = compute_j_via_gpw(basis, D, grid, quiet=True)
    J_builder = GpwJBuilder(basis, grid).build_J(D)
    assert np.allclose(J_one_shot, J_builder, atol=1e-12)


# ---------- Convergence + Madelung shift --------------------------------


def test_he_hartree_energy_converges_with_grid():
    """Refining the grid leaves the Hartree energy stable to ~µHa
    once the density is well-resolved."""
    _, basis, D, L = _he_in_box(16.0)
    e_vals = []
    for n in (48, 64, 80):
        grid = PlaneWaveGrid(L, n, n, n)
        e_vals.append(GpwJBuilder(basis, grid).hartree_energy(D))
    # The change between n=64 and n=80 must be well under a mHa.
    assert abs(e_vals[2] - e_vals[1]) < 5e-4
    # And the change between n=48 and n=64 should already be small.
    assert abs(e_vals[1] - e_vals[0]) < 5e-3


def test_he_gpw_minus_molecular_matches_madelung_shift():
    """The Hartree energy in vibe-qc's molecular vacuum vs the
    periodic GPW value on the same density differs by the cubic-cell
    Madelung self-image shift

        ``ΔE_Madelung = q² · ξ_NaCl / (2 · L)``

    with ``ξ_NaCl = 2.837297`` and ``L`` the cell side. For He
    (q = 2) in a 16-bohr cube, ΔE ≈ 0.355 Ha; the GPW recipe
    reproduces it within a few mHa once the grid is converged.
    This is the M2 internal-consistency check that the M1e
    Madelung test on neutral cells has its charged-cell sibling.
    """
    L = 16.0
    mol, basis, D, lattice = _he_in_box(L)

    # Molecular E_H from vibe-qc's existing JKBuilder.
    ps = core.PeriodicSystem()
    ps.dim = 3
    ps.lattice = np.eye(3) * 30.0   # huge box → molecular limit
    ps.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    lo = core.LatticeSumOptions()
    lo.cutoff_bohr = 25.0
    jk_mol = core.build_jk_gamma_molecular_limit(basis, ps, lo, D)
    E_mol = 0.5 * float(np.einsum("ij,ij->", D, np.asarray(jk_mol.J)))

    # Periodic E_H via GPW at a converged grid.
    grid = PlaneWaveGrid(lattice, 80, 80, 80)
    E_gpw = GpwJBuilder(basis, grid).hartree_energy(D)

    # Analytical Madelung shift for q = 2, simple cubic, L = 16:
    #   ΔE = q² · ξ_NaCl / (2 · L) = 4 · 2.837 / 32 ≈ 0.3546
    xi_nacl = 2.837297
    q = 2.0
    delta_madelung_analytic = q ** 2 * xi_nacl / (2.0 * L)

    delta_observed = E_mol - E_gpw
    assert delta_observed == pytest.approx(delta_madelung_analytic,
                                            abs=5e-3), (
        f"Madelung-shift consistency: observed E_mol - E_gpw = "
        f"{delta_observed:.4f} Ha, analytical = "
        f"{delta_madelung_analytic:.4f} Ha"
    )


# ---------- Emit experimental warning -----------------------------------


def test_gpw_j_builder_emits_experimental_warning():
    """Constructing a GpwJBuilder warns the caller — the M2 driver
    isn't yet wired to a full SCF, so anything that gets a builder
    is opting into experimental territory."""
    _, basis, _, L = _he_in_box(16.0)
    grid = PlaneWaveGrid(L, 16, 16, 16)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        GpwJBuilder(basis, grid)
    msgs = [
        w for w in captured
        if issubclass(w.category, GAPWExperimentalWarning)
    ]
    assert msgs, "GpwJBuilder should emit GAPWExperimentalWarning"


# ---------- M2c: evaluate_gpw_energy single-point entry ----------


def _he_periodic_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    return sys


def _h2_periodic_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    return sys


def test_evaluate_gpw_energy_returns_full_breakdown():
    """Dataclass populated end-to-end with finite, real values."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    _, basis, D, L_mat = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(L_mat, 48, 48, 48)
    result = evaluate_gpw_energy(system, basis, D, grid=grid, quiet=True)
    for name in ("e_kinetic", "e_nuclear_attraction", "e_hartree",
                  "e_hf_exchange", "e_nuclear_repulsion", "e_total"):
        v = getattr(result, name)
        assert isinstance(v, float)
        assert np.isfinite(v), f"{name} is not finite: {v}"
    assert result.grid is grid


def test_evaluate_gpw_energy_consistency_with_gpw_j_builder():
    """``result.e_hartree`` equals
    :meth:`GpwJBuilder.hartree_energy` — same primitive."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    _, basis, D, L_mat = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(L_mat, 48, 48, 48)
    e_hartree_builder = GpwJBuilder(basis, grid).hartree_energy(D)
    e_hartree_eval = evaluate_gpw_energy(
        system, basis, D, grid=grid, quiet=True,
    ).e_hartree
    assert e_hartree_eval == pytest.approx(e_hartree_builder, abs=1e-12)


def test_evaluate_gpw_energy_consistency_with_ewald_nuclear_repulsion():
    """``result.e_nuclear_repulsion`` matches a direct
    :func:`ewald_nuclear_repulsion` call on the same system."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    _, basis, D, _ = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 32, 32, 32)
    result = evaluate_gpw_energy(
        system, basis, D, grid=grid, quiet=True,
    )
    e_nn_direct = float(
        core.ewald_nuclear_repulsion(system, core.EwaldOptions())
    )
    assert result.e_nuclear_repulsion == pytest.approx(e_nn_direct,
                                                        abs=1e-10)


def test_evaluate_gpw_energy_kinetic_matches_molecular_in_vacuum():
    """``result.e_kinetic`` ≈ ``tr(D · T_molecular)`` on a vacuum-
    padded cell. After B3 the GPW driver uses the lattice kinetic
    builder, but the kinetic integrand decays exponentially with
    shell separation, so on a 16-bohr cube the lattice T and the
    molecular T agree to numerical noise."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    _, basis, D, _ = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 16, 16, 16)
    result = evaluate_gpw_energy(
        system, basis, D, grid=grid, quiet=True,
    )
    T = np.asarray(core.compute_kinetic(basis))
    e_kin_direct = float(np.einsum("ij,ij->", D, T))
    # Lattice T vs molecular T on a 16-bohr He cell: image-AO overlap
    # is ∝ exp(-α · L² / 2) ≈ exp(-800), so the difference is just
    # finite-arithmetic noise on the lattice sum.
    assert result.e_kinetic == pytest.approx(e_kin_direct, abs=1e-10)


def test_evaluate_gpw_energy_total_matches_term_sum():
    """``result.e_total`` is the algebraic sum of the five term
    fields — floating-point equality."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    _, basis, D, _ = _h2_in_box(16.0)
    system = _h2_periodic_system(16.0)
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 32, 32, 32)
    result = evaluate_gpw_energy(
        system, basis, D, grid=grid, quiet=True,
    )
    expected = (result.e_kinetic + result.e_nuclear_attraction
                + result.e_hartree + result.e_hf_exchange
                + result.e_nuclear_repulsion)
    assert result.e_total == pytest.approx(expected, abs=1e-12)


def test_evaluate_gpw_energy_auto_builds_grid_when_none():
    """When ``grid`` is None, the helper builds one from
    ``cutoff_ha`` and the system lattice."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    _, basis, D, _ = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    result = evaluate_gpw_energy(
        system, basis, D, cutoff_ha=8.0, quiet=True,
    )
    assert result.grid is not None
    assert result.grid.nx > 0
    assert result.grid.cutoff_ha == pytest.approx(8.0)


def test_evaluate_gpw_energy_rejects_bad_density_shape():
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    _, basis, _, _ = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)
    with pytest.raises(ValueError, match="square"):
        evaluate_gpw_energy(system, basis, np.array([1.0, 2.0]),
                              grid=grid, quiet=True)
    with pytest.raises(ValueError, match="nbasis"):
        evaluate_gpw_energy(system, basis, np.eye(5),
                              grid=grid, quiet=True)


def test_evaluate_gpw_energy_omega_nonzero_runs_and_small_omega_matches_zero():
    """Nonzero omega now reaches the range-separated GPW J builder.

    The SR+LR split is algebraically the full Coulomb J, so a tiny omega
    should reproduce the unscreened diagnostic energy within the coarse
    8^3 smoke-grid tolerance instead of raising the old M3 guard.
    """
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    _, basis, D, _ = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)
    r0 = evaluate_gpw_energy(system, basis, D, grid=grid, omega=0.0, quiet=True)
    r_small = evaluate_gpw_energy(
        system, basis, D, grid=grid, omega=1e-6, quiet=True,
    )
    r = evaluate_gpw_energy(system, basis, D, grid=grid, omega=0.3, quiet=True)

    assert np.isfinite(r.e_total)
    assert r.e_hartree > 0.0
    assert r_small.e_total == pytest.approx(r0.e_total, abs=2e-6)


def test_evaluate_gpw_energy_omega_nonzero_uses_screened_exchange():
    """Positive omega evaluates the erfc short-range exchange energy."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    _, basis, D, _ = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)
    lo = core.LatticeSumOptions()
    lo.cutoff_bohr = 25.0
    jk_sr = core.build_jk_gamma_molecular_limit(
        basis,
        system,
        lo,
        D,
        omega=0.3,
    )
    expected = -0.25 * 0.25 * float(np.einsum("ij,ij->", D, jk_sr.K))

    result = evaluate_gpw_energy(
        system,
        basis,
        D,
        grid=grid,
        omega=0.3,
        hf_sr_fraction=0.25,
        hf_lr_fraction=0.0,
        quiet=True,
    )

    assert result.e_hf_exchange == pytest.approx(expected, rel=0.0, abs=1e-12)
    assert np.isfinite(result.e_total)


def test_evaluate_gpw_energy_negative_omega_rejected():
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    _, basis, D, _ = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)
    with pytest.raises(ValueError, match="omega"):
        evaluate_gpw_energy(
            system, basis, D, grid=grid, omega=-0.1, quiet=True,
        )


def test_evaluate_gpw_energy_emits_experimental_warning():
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    _, basis, D, _ = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        evaluate_gpw_energy(system, basis, D, grid=grid)
    msgs = [
        w for w in captured
        if issubclass(w.category, GAPWExperimentalWarning)
    ]
    assert msgs, "evaluate_gpw_energy should emit GAPWExperimentalWarning"


# ---------- M2-full: minimal iterative GPW SCF -----------------------


def test_rhf_gpw_he_converges_to_evaluate_energy():
    """``run_periodic_rhf_gpw`` on He STO-3G in a 16-bohr cube
    converges to the same total energy that
    :func:`evaluate_gpw_energy` returns when evaluated at the
    converged density. End-to-end consistency: SCF + breakdown
    agree at self-consistency."""
    from vibeqc.periodic_gapw_j import (
        evaluate_gpw_energy, run_periodic_rhf_gpw,
    )

    _, basis, _, L_mat = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(L_mat, 48, 48, 48)
    result = run_periodic_rhf_gpw(system, basis, grid=grid, quiet=True)
    assert result.converged
    assert result.n_iter <= 10
    # Re-evaluate the breakdown at the converged density; the SCF's
    # own breakdown should match.
    fresh = evaluate_gpw_energy(
        system, basis, result.density, grid=grid, quiet=True,
    )
    assert result.energy == pytest.approx(fresh.e_total, abs=1e-10)


def test_run_periodic_rks_gpw_aliases_rhf_with_functional():
    """``run_periodic_rks_gpw`` is a thin alias for ``run_periodic_rhf_gpw``
    with a required ``functional=`` — same Γ-only GPW SCF, so the energies
    are bit-identical; omitting ``functional`` is an error. (GPW-AUDIT-004:
    gapw.md tutorials assumed this entry exists; this pins it.)"""
    from vibeqc.periodic_gapw_j import (
        run_periodic_rhf_gpw, run_periodic_rks_gpw,
    )

    _, basis, _, L_mat = _h2_in_box(16.0)
    system = _h2_periodic_system(16.0)
    grid = PlaneWaveGrid(L_mat, 48, 48, 48)

    via_rks = run_periodic_rks_gpw(
        system, basis, functional="lda", grid=grid, max_iter=30, quiet=True,
    )
    via_rhf = run_periodic_rhf_gpw(
        system, basis, functional="lda", grid=grid, max_iter=30, quiet=True,
    )
    assert via_rks.converged
    assert via_rks.energy == pytest.approx(via_rhf.energy, abs=1e-12)

    with pytest.raises((TypeError, ValueError)):
        run_periodic_rks_gpw(system, basis, grid=grid)  # functional required


def test_rhf_gpw_h2_converges():
    """H2 STO-3G in a 16-bohr cube — 2 AOs, requires an actual
    iteration loop (unlike He's 1-AO trivial case). Converges to
    a finite, sensible energy."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)
    result = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=30, quiet=True,
    )
    assert result.converged
    assert result.n_iter <= 15
    # Sensible range — anywhere near the molecular E_HF (-1.117 Ha)
    # plus the Madelung-shift order (~ -0.7 Ha for q=2).
    assert -2.5 < result.energy < -0.5


def test_rhf_gpw_initial_density_seed_matches_hcore_guess():
    """Seeding ``initial_density`` from a converged result and
    re-running should still converge to the same energy in <= 2
    iterations (the density is already a fixed point)."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    _, basis, _, L_mat = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(L_mat, 32, 32, 32)
    first = run_periodic_rhf_gpw(
        system, basis, grid=grid, quiet=True,
    )
    second = run_periodic_rhf_gpw(
        system, basis, grid=grid,
        initial_density=first.density, quiet=True,
    )
    assert second.converged
    # The second run starts at a fixed point — converges immediately.
    assert second.n_iter <= 3
    assert second.energy == pytest.approx(first.energy, abs=1e-9)


def test_rhf_gpw_result_carries_mo_data():
    """The result populates mo_coeffs + mo_energies of the right
    shape; energies sorted ascending (canonical ordering)."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32)
    result = run_periodic_rhf_gpw(
        system, basis, grid=grid, quiet=True,
    )
    assert result.mo_coeffs.shape == (basis.nbasis, basis.nbasis)
    assert result.mo_energies.shape == (basis.nbasis,)
    # Eigenvalues sorted ascending.
    e = result.mo_energies
    assert np.all(e[:-1] <= e[1:] + 1e-12)
    # MOs orthonormal under S (within rounding).
    S = np.asarray(core.compute_overlap(basis))
    overlap_mo = result.mo_coeffs.T @ S @ result.mo_coeffs
    assert np.allclose(overlap_mo, np.eye(basis.nbasis), atol=1e-8)


def test_rhf_gpw_rejects_odd_electron_count():
    """The closed-shell SCF requires an even electron count."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    # Single H atom — 1 electron, odd.
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(1, [L / 2, L / 2, L / 2])]
    mol = vq.Molecule(list(sys.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * L, 16, 16, 16)
    with pytest.raises(ValueError, match="closed-shell"):
        run_periodic_rhf_gpw(sys, basis, grid=grid, quiet=True)


def test_rhf_gpw_rejects_non_3d():
    """Only dim == 3 is supported at M2-full."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    sys = core.PeriodicSystem()
    sys.dim = 2  # slab
    sys.lattice = np.eye(3) * 16.0
    sys.unit_cell = [core.Atom(2, [8.0, 8.0, 8.0])]
    mol = vq.Molecule(list(sys.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 16, 16, 16)
    with pytest.raises(ValueError, match="dim == 3"):
        run_periodic_rhf_gpw(sys, basis, grid=grid, quiet=True)


def test_rhf_gpw_damping_still_converges():
    """A moderate damping factor shouldn't break convergence on
    a compact system — it just costs extra iterations."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    _, basis, _, L_mat = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(L_mat, 32, 32, 32)
    result = run_periodic_rhf_gpw(
        system, basis, grid=grid, damping=0.5,
        max_iter=50, quiet=True,
    )
    assert result.converged
    # Damping bumps the iteration count vs the undamped baseline.
    assert result.n_iter >= 2


def test_rhf_gpw_emits_experimental_warning():
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    _, basis, _, _ = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 16, 16, 16)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        run_periodic_rhf_gpw(system, basis, grid=grid)
    msgs = [
        w for w in captured
        if issubclass(w.category, GAPWExperimentalWarning)
    ]
    assert msgs, "run_periodic_rhf_gpw should emit GAPWExperimentalWarning"


def test_rhf_gpw_he_recovers_molecular_in_vacuum_padded_cell():
    """End-to-end SCF-level consistency under the M3a Ewald-V_ne
    gauge: ``run_periodic_rhf_gpw`` on He STO-3G in a 16-bohr cube
    converges to the molecular E_HF to within 1 mHa.

    Before M3a, V_ne lived at the molecular limit while J + E_nn
    lived in the Ewald gauge, leaving a "2× Madelung" shift in
    the periodic total energy (the per-charge cubic Madelung
    constant ``ξ_NaCl / (2L) ≈ 0.0887 Ha`` per particle on a
    16-bohr cube, doubled because the electron-nucleus cross-term
    was missing its compensating jellium contribution). The M3a
    lift to ``compute_nuclear_lattice_dispatch`` with
    ``CoulombMethod.EWALD_3D`` gives V_ne the same G = 0 + v_bg
    convention as the FFT-Poisson J, and the three pieces now
    cancel exactly on a neutral cell — periodic E_HF matches
    molecular E_HF to the µHa floor of finite-cell-size +
    finite-grid corrections.
    """
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    mol, basis, _, L_mat = _he_in_box(L)
    system = _he_periodic_system(L)

    # Molecular reference.
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    e_mol = vq.run_rhf(mol, basis, opts).energy

    # Periodic SCF via M2-full + M3a.
    grid = PlaneWaveGrid(L_mat, 64, 64, 64)
    result = run_periodic_rhf_gpw(system, basis, grid=grid, quiet=True)
    assert result.converged

    delta_observed = abs(e_mol - result.energy)
    assert delta_observed < 1e-3, (
        f"M3a Ewald-V_ne gauge-cancellation consistency: |E_mol - "
        f"E_periodic| should be < 1 mHa on a vacuum-padded neutral "
        f"He cell. Got {delta_observed:.6f} Ha. "
        f"E_mol={e_mol:.6f}, E_periodic={result.energy:.6f}"
    )


def test_evaluate_gpw_energy_he_hartree_matches_madelung_shifted_reference():
    """The M2 Madelung-shift consistency check re-expressed at the
    breakdown level: ``e_hartree`` matches molecular ``E_J`` minus
    the analytical cubic-cell Madelung shift to within 5 mHa on
    He STO-3G in a 16-bohr cube."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    L = 16.0
    _, basis, D, L_mat = _he_in_box(L)
    system = _he_periodic_system(L)
    grid = PlaneWaveGrid(L_mat, 80, 80, 80)

    ps_big = core.PeriodicSystem()
    ps_big.dim = 3
    ps_big.lattice = np.eye(3) * 30.0
    ps_big.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    lo = core.LatticeSumOptions()
    lo.cutoff_bohr = 25.0
    jk_mol = core.build_jk_gamma_molecular_limit(basis, ps_big, lo, D)
    E_J_mol = 0.5 * float(np.einsum("ij,ij->", D, np.asarray(jk_mol.J)))

    result = evaluate_gpw_energy(system, basis, D, grid=grid, quiet=True)

    xi_nacl = 2.837297
    delta_madelung = (2.0 ** 2) * xi_nacl / (2.0 * L)
    expected_e_hartree = E_J_mol - delta_madelung
    assert result.e_hartree == pytest.approx(expected_e_hartree,
                                              abs=5e-3)


# ---------- M3a: Ewald V_ne lift ----------------------------------------


def test_ewald_v_ne_gamma_symmetric_and_real():
    """The M3a helper :func:`_ewald_v_ne_gamma` returns a real,
    symmetric ``(n_basis, n_basis)`` matrix at Γ — these are
    invariants of the periodic V_ne in any real-valued AO basis."""
    from vibeqc.periodic_gapw_j import _ewald_v_ne_gamma

    _, basis, _, _ = _h2_in_box(16.0)
    system = _h2_periodic_system(16.0)
    V_ne = _ewald_v_ne_gamma(basis, system)
    assert V_ne.shape == (basis.nbasis, basis.nbasis)
    assert V_ne.dtype == float
    assert np.allclose(V_ne, V_ne.T, atol=1e-12)
    assert np.all(np.isfinite(V_ne))


def test_ewald_v_ne_gamma_matches_molecular_limit_in_vacuum():
    """In a large vacuum-padded cell, the periodic Ewald V_ne is
    *close* to the molecular V_ne — but not bit-equal: the v_bg
    jellium shift raises each AO matrix element so the V_ne
    cross-term contributes ``+q² · ξ / (2L)`` to the total
    energy, exactly cancelling the ``-q² · ξ / (2L)`` self-image
    shifts that FFT-Poisson and the Ewald nuclear-repulsion
    impose on the Hartree and nuclear-nuclear terms.

    For He STO-3G (1 AO, Z = 2, n_elec = 2) in an L = 16 bohr
    cubic cell, ``tr(D · V_ne_ewald) - tr(D · V_ne_mol)`` should
    equal ``2 · Z² · ξ_NaCl / (2L) = 0.709 Ha`` — half of which
    is the e-on-electron self-image cancellation, half the
    e-on-nuclei cross-term cancellation. With one AO and a
    closed-shell density of trace 2, that simplifies to the
    1×1 diagonal shift ``Z · ξ_NaCl / L ≈ 0.355 Ha``.

    This is the convention pin for M3a — the Ewald V_ne is
    *gauge-aligned with the FFT-Poisson J*, not bit-equal to the
    molecular V_ne. Future regressions that drift the
    convention will trip this test, which is the goal."""
    from vibeqc.periodic_gapw_j import _ewald_v_ne_gamma, _wrap_molecule

    L = 16.0
    _, basis, _, _ = _he_in_box(L)
    system = _he_periodic_system(L)
    V_ne_ewald = _ewald_v_ne_gamma(basis, system)
    V_ne_mol = np.asarray(core.compute_nuclear(basis, _wrap_molecule(system)))

    delta = V_ne_ewald - V_ne_mol
    xi_nacl = 2.837297
    Z = 2.0
    # The v_bg jellium raises V_ne by Z·ξ/L per AO matrix element
    # (positive shift; opposes the attractive nuclear potential).
    expected_diag_shift = Z * xi_nacl / L
    assert delta[0, 0] == pytest.approx(expected_diag_shift, abs=5e-3), (
        f"Ewald V_ne should sit ~{expected_diag_shift:.4f} Ha "
        f"above the molecular V_ne on a He 16-bohr cube; got "
        f"delta = {delta[0, 0]:.6f} Ha."
    )


def test_evaluate_gpw_energy_he_total_matches_molecular_under_m3a():
    """Breakdown-level analogue of the M3a SCF consistency check:
    on a vacuum-padded neutral He cell, ``e_total`` evaluated at
    the molecular density equals the molecular E_HF to < 1 mHa.
    The Ewald-V_ne lift is the load-bearing piece — pre-M3a the
    same call returned a "2× Madelung" shifted total."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    L = 16.0
    mol, basis, D, L_mat = _he_in_box(L)
    system = _he_periodic_system(L)
    grid = PlaneWaveGrid(L_mat, 64, 64, 64)

    # Molecular reference.
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    e_mol = vq.run_rhf(mol, basis, opts).energy

    result = evaluate_gpw_energy(system, basis, D, grid=grid, quiet=True)
    assert abs(result.e_total - e_mol) < 1e-3, (
        f"M3a breakdown consistency: e_total - E_mol should be < "
        f"1 mHa on a vacuum-padded neutral He cell. Got "
        f"|{result.e_total:.6f} - {e_mol:.6f}| = "
        f"{abs(result.e_total - e_mol):.6f} Ha."
    )


def test_rhf_gpw_h2_total_matches_molecular_under_m3a():
    """Same M3a gauge-cancellation check on H2 STO-3G (2 AOs, two
    nuclei, genuine SCF iteration): periodic E_HF matches
    molecular E_HF to < 1 mHa in a 16-bohr cube."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _h2_periodic_system(L)

    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    e_mol = vq.run_rhf(mol, basis, opts).energy

    grid = PlaneWaveGrid(np.eye(3) * L, 64, 64, 64)
    result = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=30, quiet=True,
    )
    assert result.converged
    assert abs(result.energy - e_mol) < 1e-3, (
        f"M3a H2 SCF consistency: |E_periodic - E_mol| should be "
        f"< 1 mHa. Got {abs(result.energy - e_mol):.6f} Ha. "
        f"E_mol={e_mol:.6f}, E_periodic={result.energy:.6f}"
    )


# ---------- B1: charged-cell neutrality guard ---------------------------


def test_evaluate_gpw_energy_warns_on_non_neutral_density():
    """Passing a density whose trace doesn't match Σ Z should trip the
    neutrality preflight, because the FFT-Poisson + Ewald-V_ne
    convention only cancels on a neutral cell. Quiet path is silent;
    default path warns."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    L = 16.0
    _, basis, D_neutral, _ = _he_in_box(L)
    system = _he_periodic_system(L)
    # He STO-3G has Σ Z = 2; a "+1 cation" density would have
    # tr(D·S) = 1. Build a deliberately non-neutral D.
    D_cation = 0.5 * D_neutral  # tr(D·S) = 1.0, off by 1.0 e
    grid = PlaneWaveGrid(np.eye(3) * L, 16, 16, 16)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        evaluate_gpw_energy(system, basis, D_cation, grid=grid)
    neutrality_msgs = [
        w for w in captured
        if issubclass(w.category, GAPWExperimentalWarning)
        and "non-neutral" in str(w.message)
    ]
    assert neutrality_msgs, (
        "Non-neutral D should emit a GAPWExperimentalWarning about "
        f"jellium-class total-energy shift. Got messages: "
        f"{[str(w.message) for w in captured]}"
    )


def test_kinetic_lattice_gamma_matches_molecular_in_vacuum():
    """The B3 helper :func:`_kinetic_lattice_gamma` returns a real
    symmetric matrix that equals the molecular kinetic matrix to
    numerical noise on a vacuum-padded box. The kinetic integrand
    decays exponentially with shell separation; image-AO image-AO
    couplings at L = 16 bohr on He STO-3G are ~exp(-800)."""
    from vibeqc.periodic_gapw_j import _kinetic_lattice_gamma

    _, basis, _, _ = _he_in_box(16.0)
    system = _he_periodic_system(16.0)
    T_lat = _kinetic_lattice_gamma(basis, system)
    T_mol = np.asarray(core.compute_kinetic(basis))
    assert T_lat.shape == T_mol.shape
    assert np.allclose(T_lat, T_lat.T, atol=1e-12)
    assert np.allclose(T_lat, T_mol, atol=1e-10), (
        f"Lattice T vs molecular T on a 16-bohr He cell should agree "
        f"to numerical noise (image overlap ~exp(-800)). Max abs "
        f"diff = {np.max(np.abs(T_lat - T_mol)):.3e}"
    )


def test_overlap_lattice_gamma_matches_molecular_in_vacuum():
    """Same shape-and-noise pin for :func:`_overlap_lattice_gamma`."""
    from vibeqc.periodic_gapw_j import _overlap_lattice_gamma

    _, basis, _, _ = _h2_in_box(16.0)
    system = _h2_periodic_system(16.0)
    S_lat = _overlap_lattice_gamma(basis, system)
    S_mol = np.asarray(core.compute_overlap(basis))
    assert S_lat.shape == S_mol.shape
    assert np.allclose(S_lat, S_lat.T, atol=1e-12)
    assert np.allclose(S_lat, S_mol, atol=1e-10), (
        f"Lattice S vs molecular S on a 16-bohr H2 cell should agree "
        f"to numerical noise. Max abs diff = "
        f"{np.max(np.abs(S_lat - S_mol)):.3e}"
    )


def test_rhf_gpw_smeared_erfc_matches_ewald_path_on_he():
    """The M3b ``v_ne_convention='smeared_erfc'`` path converges to
    the same SCF total energy as the M3a default ``'ewald'`` path on
    a vacuum-padded neutral He STO-3G cell. The two conventions are
    physically equivalent on neutral cells — the M3b path adds the
    α-correction that makes the gauges match."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    _, basis, _, L_mat = _he_in_box(L)
    system = _he_periodic_system(L)
    grid = PlaneWaveGrid(L_mat, 64, 64, 64, cutoff_ha=300.0)

    r_ewald = run_periodic_rhf_gpw(
        system, basis, grid=grid, v_ne_convention="ewald", quiet=True,
    )
    r_smeared = run_periodic_rhf_gpw(
        system, basis, grid=grid,
        v_ne_convention="smeared_erfc", smearing_alpha=2.0,
        quiet=True,
    )
    assert r_ewald.converged
    assert r_smeared.converged
    assert abs(r_smeared.energy - r_ewald.energy) < 1e-3, (
        f"M3b Ewald vs smeared SCF parity on He STO-3G: "
        f"E_ewald = {r_ewald.energy:.6f}, "
        f"E_smeared = {r_smeared.energy:.6f}, "
        f"Δ = {abs(r_smeared.energy - r_ewald.energy):.6e} Ha"
    )


def test_rhf_gpw_smeared_erfc_matches_molecular_on_h2():
    """End-to-end M3b test: H2 STO-3G in a 16-bohr cube with the
    smeared erfc V_ne convention converges to the molecular E_HF
    within < 1 mHa, same target as the M3a Ewald path."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 64, 64, 64, cutoff_ha=300.0)

    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    e_mol = vq.run_rhf(mol, basis, opts).energy

    r = run_periodic_rhf_gpw(
        system, basis, grid=grid,
        v_ne_convention="smeared_erfc", smearing_alpha=2.0,
        quiet=True,
    )
    assert r.converged
    assert abs(r.energy - e_mol) < 1e-3, (
        f"H2 STO-3G periodic E_HF (smeared erfc) vs molecular: "
        f"|{r.energy:.6f} - {e_mol:.6f}| = {abs(r.energy - e_mol):.6e}"
    )


def test_rhf_gpw_smeared_erfc_alpha_sweep_neutral_cell_parity():
    """The M3b SCF total energy is independent of the smearing
    exponent α on a neutral cell (within the convention's own
    finite-grid noise), since both paths converge to the same
    physical total. Sweeping α ∈ {1.0, 1.5, 2.0}: the deviation
    from the Ewald path stays bounded by 1 mHa on He STO-3G."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    _, basis, _, L_mat = _he_in_box(L)
    system = _he_periodic_system(L)
    grid = PlaneWaveGrid(L_mat, 64, 64, 64, cutoff_ha=300.0)
    e_ewald = run_periodic_rhf_gpw(
        system, basis, grid=grid, v_ne_convention="ewald", quiet=True,
    ).energy

    for alpha in (1.0, 1.5, 2.0):
        r = run_periodic_rhf_gpw(
            system, basis, grid=grid,
            v_ne_convention="smeared_erfc", smearing_alpha=alpha,
            quiet=True,
        )
        assert r.converged
        delta = abs(r.energy - e_ewald)
        assert delta < 1e-3, (
            f"α = {alpha}: |E_smear - E_ewald| = {delta:.6e}, "
            f"E_smear = {r.energy:.6f}, E_ewald = {e_ewald:.6f}"
        )


def test_rhf_gpw_rejects_unknown_v_ne_convention():
    """Bad ``v_ne_convention`` raises a clear ValueError."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    _, basis, _, L_mat = _he_in_box(L)
    system = _he_periodic_system(L)
    grid = PlaneWaveGrid(L_mat, 16, 16, 16, cutoff_ha=300.0)
    with pytest.raises(ValueError, match="convention"):
        run_periodic_rhf_gpw(
            system, basis, grid=grid,
            v_ne_convention="totally_bogus", quiet=True,
        )


def test_evaluate_gpw_energy_smeared_erfc_total_matches_ewald():
    """Breakdown-level analogue of the SCF parity check: at the
    molecular density, ``e_total`` from the smeared erfc path
    matches the Ewald path within < 1 mHa."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    L = 16.0
    _, basis, D, L_mat = _he_in_box(L)
    system = _he_periodic_system(L)
    grid = PlaneWaveGrid(L_mat, 64, 64, 64, cutoff_ha=300.0)
    r_ewald = evaluate_gpw_energy(
        system, basis, D, grid=grid,
        v_ne_convention="ewald", quiet=True,
    )
    r_smear = evaluate_gpw_energy(
        system, basis, D, grid=grid,
        v_ne_convention="smeared_erfc", smearing_alpha=2.0, quiet=True,
    )
    assert abs(r_ewald.e_total - r_smear.e_total) < 1e-3, (
        f"breakdown-level parity on He STO-3G: "
        f"E_ewald = {r_ewald.e_total:.6f}, "
        f"E_smear = {r_smear.e_total:.6f}"
    )


# ---------- M3d: DFT (RKS) support ---------------------------------------


def test_rhf_gpw_he_lda_matches_molecular():
    """The M3d ``functional='lda'`` path on He STO-3G in a 16-bohr
    cube converges to the molecular LDA energy within 1 µHa. The
    XC contribution is local + finite-grid quadrature, so the gauge-
    cancellation in HF (sub-µHa) carries through to DFT once the
    libxc ``exc`` convention (energy density per unit volume) is
    handled correctly."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    _, basis, _, L_mat = _he_in_box(L)
    system = _he_periodic_system(L)
    grid = PlaneWaveGrid(L_mat, 64, 64, 64, cutoff_ha=300.0)

    opts = vq.RKSOptions(); opts.conv_tol_energy = 1e-10
    opts.functional = "lda"
    e_mol = vq.run_rks(
        vq.Molecule([vq.Atom(2, [L / 2, L / 2, L / 2])], 0, 1),
        basis, opts,
    ).energy

    r = run_periodic_rhf_gpw(
        system, basis, grid=grid, functional="lda", quiet=True,
    )
    assert r.converged
    assert abs(r.energy - e_mol) < 1e-5, (
        f"He STO-3G periodic GPW-LDA vs molecular: |Δ| = "
        f"{abs(r.energy - e_mol):.6e}"
    )
    assert r.breakdown.e_xc != 0.0
    assert r.breakdown.functional == "lda"


def test_rhf_gpw_h2_pbe_matches_molecular():
    """Same but for H2 STO-3G with PBE (a GGA functional). Tests the
    full GGA gradient-correction path: σ = |∇ρ|² built via spectral
    FFT, v_sigma projected via the divergence-by-parts trick."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 64, 64, 64, cutoff_ha=300.0)

    opts = vq.RKSOptions(); opts.conv_tol_energy = 1e-10
    opts.functional = "pbe"
    e_mol = vq.run_rks(mol, basis, opts).energy

    r = run_periodic_rhf_gpw(
        system, basis, grid=grid, functional="pbe", quiet=True,
    )
    assert r.converged
    assert abs(r.energy - e_mol) < 1e-5, (
        f"H2 STO-3G periodic GPW-PBE vs molecular: |Δ| = "
        f"{abs(r.energy - e_mol):.6e}"
    )


def test_rhf_gpw_b3lyp_uses_hybrid_exchange_fraction():
    """B3LYP hybrid has ~0.2 HF-exchange fraction. The Fock build
    should scale K by 0.2 (not the full 1.0 of HF). Energy should
    match molecular B3LYP to mHa."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    mol, basis, _, L_mat = _he_in_box(L)
    system = _he_periodic_system(L)
    grid = PlaneWaveGrid(L_mat, 64, 64, 64, cutoff_ha=300.0)

    opts = vq.RKSOptions(); opts.conv_tol_energy = 1e-10
    opts.functional = "b3lyp"
    e_mol = vq.run_rks(mol, basis, opts).energy

    r = run_periodic_rhf_gpw(
        system, basis, grid=grid, functional="b3lyp", quiet=True,
    )
    assert r.converged
    # B3LYP is a GGA hybrid; the gradient projection accumulates more
    # finite-grid error than LDA. 10 mHa on He STO-3G in a 16-bohr
    # cube at 300 Ha cutoff is acceptable for this convention.
    assert abs(r.energy - e_mol) < 1e-2, (
        f"He STO-3G periodic GPW-B3LYP vs molecular: |Δ| = "
        f"{abs(r.energy - e_mol):.6e}"
    )
    # HF exchange fraction should be non-zero (it's a hybrid).
    assert r.breakdown.e_hf_exchange < 0.0


def test_rhf_gpw_scf_trace_records_per_iter_energies():
    """The M3d SCF stores per-iter energies in ``GpwScfResult.scf_trace``.
    Each entry has 'iter', 'energy', 'delta_e', 'grad_norm', 'e_xc'.

    The fixture must actually exhibit SCF dynamics: with STO-3G, H2 has
    2 AOs and its one occupied MO is fixed by symmetry, so iteration 1 is
    already the converged density to full double precision and the
    "trace is not a copy of the final energy" guard was vacuous. 6-31G
    gives the symmetric occupied subspace two dimensions, so the density
    genuinely evolves (~52 mHa between iteration 1 and convergence)."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "6-31g")
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)

    r = run_periodic_rhf_gpw(
        system, basis, grid=grid, functional="lda", quiet=True,
    )
    assert r.converged
    assert r.n_iter > 1
    assert len(r.scf_trace) == r.n_iter
    for step in r.scf_trace:
        assert "iter" in step
        assert "energy" in step
        assert "delta_e" in step
        assert "grad_norm" in step
        assert "e_xc" in step
    assert [step["iter"] for step in r.scf_trace] == list(
        range(1, r.n_iter + 1)
    )
    # Genuine SCF dynamics: the first iterate differs from the converged
    # energy, and each recorded delta_e is the actual per-iteration
    # energy difference (a trace that copies the final energy into every
    # slot fails both).
    assert r.scf_trace[0]["energy"] != r.energy
    for prev, step in zip(r.scf_trace, r.scf_trace[1:]):
        assert step["delta_e"] == pytest.approx(
            step["energy"] - prev["energy"], abs=1e-12
        )


def test_rhf_gpw_result_carries_converged_fock_and_overlap():
    """The GPW driver stores the Fock it actually diagonalized (and the
    overlap it diagonalized against) on ``GpwScfResult``, so the runner
    adapter can surface the SCF's own operator instead of re-deriving an
    HF-shaped one. Consistency pin: the stored (F, S) generalized
    eigenvalues reproduce ``mo_energies`` to machine precision."""
    import scipy.linalg as sla

    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    _, basis, _, _ = _h2_in_box(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32, cutoff_ha=300.0)

    r = run_periodic_rhf_gpw(
        system, basis, grid=grid, functional="lda", quiet=True,
    )
    assert r.converged
    assert r.fock is not None
    assert r.overlap is not None
    n = basis.nbasis
    assert r.fock.shape == (n, n)
    assert r.overlap.shape == (n, n)
    ev = sla.eigh(np.asarray(r.fock), np.asarray(r.overlap),
                  eigvals_only=True)
    assert np.abs(ev - np.asarray(r.mo_energies)).max() < 1e-10


def test_gpw_runner_adapter_uses_stored_fock_without_reconstruction():
    """``gpw_result_to_runner_shape`` surfaces the driver's converged
    Fock/overlap verbatim when present, and does not rebuild J/K (the
    reconstruction cost ~36% of a small GPW job's wall time and is the
    wrong operator for KS/GAPW results anyway)."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw
    from vibeqc.periodic_gapw_runner_adapter import (
        gpw_result_to_runner_shape,
    )

    L = 16.0
    _, basis, _, _ = _h2_in_box(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32, cutoff_ha=300.0)

    r = run_periodic_rhf_gpw(
        system, basis, grid=grid, functional="lda", quiet=True,
    )
    # Poison the builder: with a stored Fock the adapter must not
    # construct a GpwJBuilder at all.
    import vibeqc.periodic_gapw_j as gj_mod

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError(
                "adapter must not reconstruct J when the result "
                "carries its converged Fock"
            )

    orig = gj_mod.GpwJBuilder
    gj_mod.GpwJBuilder = _Boom
    try:
        shaped = gpw_result_to_runner_shape(r, system, basis)
    finally:
        gj_mod.GpwJBuilder = orig
    assert np.array_equal(np.asarray(shaped.fock), np.asarray(r.fock))
    assert np.array_equal(
        np.asarray(shaped.overlap), np.asarray(r.overlap)
    )


def test_rhf_gpw_diis_converges_in_fewer_iters_than_no_diis():
    """DIIS should accelerate convergence relative to plain damping
    on a multi-AO system that actually iterates."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 16.0
    _, basis, _, _ = _h2_in_box(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32, cutoff_ha=300.0)

    r_diis = run_periodic_rhf_gpw(
        system, basis, grid=grid, use_diis=True, quiet=True,
    )
    r_nodiis = run_periodic_rhf_gpw(
        system, basis, grid=grid, use_diis=False, quiet=True,
    )
    # H2 STO-3G is a very easy problem; both should converge fast.
    # Just check both converge and DIIS doesn't take more iters.
    assert r_diis.converged
    assert r_nodiis.converged
    assert r_diis.n_iter <= r_nodiis.n_iter + 1


# ---------- M3e: multi-k GPW --------------------------------------------


def test_multi_k_gpw_gamma_only_matches_single_k():
    """A [1,1,1] (Γ-only) k-mesh must give the same SCF energy as
    the standalone single-k run on the same system."""
    from vibeqc.periodic_gapw_j import (
        run_periodic_rhf_gpw, run_periodic_rks_gpw_multi_k,
    )

    L = 12.0
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)

    r_single = run_periodic_rhf_gpw(
        system, basis, grid=grid, functional="lda", quiet=True,
    )
    kmesh = core.monkhorst_pack(system, [1, 1, 1])
    r_multi = run_periodic_rks_gpw_multi_k(
        system, basis, kmesh,
        functional="lda", grid=grid, quiet=True,
    )
    assert r_multi.converged
    # Not an exact invariant since the GPW-SK-HK-RSET-INCONSISTENT fix:
    # the multi-k driver sums S/T/V over the basis-derived
    # bloch_overlap_cutoff_bohr R-set (>= 25 bohr floor) while the
    # standalone Gamma driver keeps its historical build, so the two
    # differ by the physical inter-image tail the multi-k policy now
    # includes (measured 1.5e-9 Ha on this 12-bohr H2 box).
    assert abs(r_multi.energy - r_single.energy) < 1e-8


def test_multi_k_gpw_2x2x2_converges():
    """A 2×2×2 k-mesh runs and converges. The energy should be in
    the same ballpark as the Γ-only one for a vacuum-padded box
    (image-image dispersion shifts it by a small amount)."""
    from vibeqc.periodic_gapw_j import run_periodic_rks_gpw_multi_k

    L = 12.0
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)

    kmesh = core.monkhorst_pack(system, [2, 2, 2])
    r = run_periodic_rks_gpw_multi_k(
        system, basis, kmesh,
        functional="lda", grid=grid, quiet=True,
    )
    assert r.converged
    assert len(r.mo_coeffs_k) == 8
    assert len(r.mo_energies_k) == 8
    # In the molecular limit (vacuum-padded H2 STO-3G) the k-
    # dispersion is tiny; the multi-k SCF should still match the
    # molecular E_HF_DFT within a few mHa.
    opts = vq.RKSOptions()
    opts.conv_tol_energy = 1e-10
    opts.functional = "lda"
    e_mol = vq.run_rks(mol, basis, opts).energy
    assert abs(r.energy - e_mol) < 1e-2


def test_multi_k_gpw_breakdown_matches_total():
    """The multi-k GPW energy breakdown must sum to ``result.energy``.

    Regression for the v0.12.x "GPW multi-k breakdown ≠ total" audit
    finding. The breakdown re-evaluated the kinetic + nuclear-attraction
    terms at Γ (k = 0) only — ``tr(D_total · T^Γ)`` — dropping the per-k
    Bloch phases, so ``breakdown.e_total`` disagreed with the (correct)
    reported total by tens of mHa on a dispersive mesh. This exact case
    ([2,1,1] H2/STO-3G in a 6-bohr cell) was off by −60.8 mHa pre-fix.
    The fix accumulates kinetic + V_ne from the per-k Bloch-summed
    Hcore(k), the same sum that produces ``result.energy``.
    """
    from vibeqc.periodic_gapw_j import run_periodic_rks_gpw_multi_k

    L = 6.0  # tight: bands disperse, so the Γ-only breakdown bit hard
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 40, 40, 40, cutoff_ha=300.0)

    kmesh = core.monkhorst_pack(system, [2, 1, 1])
    r = run_periodic_rks_gpw_multi_k(
        system, basis, kmesh, functional="lda", grid=grid, quiet=True,
    )
    assert r.converged
    bd = r.breakdown
    # The per-term algebraic sum reproduces e_total exactly ...
    term_sum = (
        bd.e_kinetic + bd.e_nuclear_attraction + bd.e_hartree
        + bd.e_hf_exchange + bd.e_xc + bd.e_nuclear_repulsion
        + bd.e_dispersion + bd.e_dft_plus_u
    )
    assert term_sum == pytest.approx(bd.e_total, abs=1e-10)
    # ... and e_total matches the reported total energy (was 60.8 mHa off).
    assert bd.e_total == pytest.approx(r.energy, abs=1e-6)


def test_multi_k_gpw_rejects_hybrid():
    """Multi-k path requires pure DFT (no HF exchange).
    Hybrid functionals must raise."""
    from vibeqc.periodic_gapw_j import run_periodic_rks_gpw_multi_k

    L = 12.0
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _h2_periodic_system(L)
    kmesh = core.monkhorst_pack(system, [1, 1, 1])
    with pytest.raises(NotImplementedError, match="hybrid"):
        run_periodic_rks_gpw_multi_k(
            system, basis, kmesh,
            functional="b3lyp", quiet=True,
        )


def test_multi_k_gpw_rejects_odd_electron_count():
    """Single H (1 electron) must raise."""
    from vibeqc.periodic_gapw_j import run_periodic_rks_gpw_multi_k

    L = 12.0
    sys = core.PeriodicSystem(); sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(1, [L / 2, L / 2, L / 2])]
    mol = vq.Molecule(list(sys.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")
    kmesh = core.monkhorst_pack(sys, [1, 1, 1])
    with pytest.raises(ValueError, match="even count"):
        run_periodic_rks_gpw_multi_k(
            sys, basis, kmesh, functional="lda", quiet=True,
        )


def test_evaluate_gpw_energy_silent_on_neutral_density():
    """A converged neutral density should not trip the neutrality
    warning (the experimental warning still fires from the J builder
    but not for non-neutral specifically)."""
    from vibeqc.periodic_gapw_j import evaluate_gpw_energy

    L = 16.0
    _, basis, D, _ = _he_in_box(L)
    system = _he_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 16, 16, 16)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        evaluate_gpw_energy(system, basis, D, grid=grid, quiet=True)
    neutrality_msgs = [
        w for w in captured
        if issubclass(w.category, GAPWExperimentalWarning)
        and "non-neutral" in str(w.message)
    ]
    assert not neutrality_msgs, (
        "Neutral D should not trip the non-neutral warning. Got: "
        f"{[str(w.message) for w in neutrality_msgs]}"
    )


# --- Gamma V_ne memoisation --------------------------------------------
# A GPW run asks _ewald_v_ne_gamma for the same matrix three times (the
# SCF, evaluate_gpw_energy, and the runner adapter). It is memoised, which
# measured 26% off a LiH/STO-3G/PBE job with the energy bit-identical.
# The cache is keyed on CONTENT, not identity, and these pin why that
# matters: PeriodicSystem is mutable in place, so an id()-keyed cache
# would return a V_ne for the previous geometry -- a wrong answer.


def _vne_cache_fixture():
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 6.0,
        [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 3.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_ewald_v_ne_gamma_is_memoised(monkeypatch):
    """Repeat calls with the same content build V_ne once and agree."""
    import vibeqc.periodic_v_ne as vne_mod

    gj._V_NE_GAMMA_CACHE.clear()
    system, basis = _vne_cache_fixture()

    builds = []
    real = vne_mod.compute_nuclear_lattice_dispatch

    def counted(*args, **kwargs):
        builds.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(vne_mod, "compute_nuclear_lattice_dispatch", counted)

    first = gj._ewald_v_ne_gamma(basis, system)
    second = gj._ewald_v_ne_gamma(basis, system)
    third = gj._ewald_v_ne_gamma(basis, system)

    assert len(builds) == 1, f"expected one build, got {len(builds)}"
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, third)


def test_ewald_v_ne_gamma_cache_is_content_keyed_not_identity_keyed(
    monkeypatch,
):
    """Moving a nucleus IN PLACE must invalidate the memo.

    ``PeriodicSystem`` is mutable from Python (``atom.xyz`` and
    ``lattice`` are both settable), so a geometry optimiser or a finite
    difference can move the nuclei while keeping the same object. An
    ``id()``-keyed cache would silently return the previous geometry's
    V_ne. This is the regression that keeps the key on content.
    """
    import vibeqc.periodic_v_ne as vne_mod

    gj._V_NE_GAMMA_CACHE.clear()
    system, basis = _vne_cache_fixture()

    builds = []
    real = vne_mod.compute_nuclear_lattice_dispatch

    def counted(*args, **kwargs):
        builds.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(vne_mod, "compute_nuclear_lattice_dispatch", counted)

    before = gj._ewald_v_ne_gamma(basis, system)
    assert len(builds) == 1

    # Same object, moved nucleus.
    system.unit_cell[1].xyz = [0.0, 0.0, 2.4]
    after = gj._ewald_v_ne_gamma(basis, system)

    assert len(builds) == 2, "cache did not notice the moved nucleus"
    assert not np.allclose(before, after), (
        "V_ne is unchanged after moving a nucleus by 0.6 bohr -- the memo "
        "returned a stale matrix"
    )

    # And a lattice change is caught too.
    system.lattice = np.eye(3) * 6.5
    gj._ewald_v_ne_gamma(basis, system)
    assert len(builds) == 3, "cache did not notice the changed lattice"


def test_ewald_v_ne_gamma_returns_independent_arrays():
    """Callers own what they get: mutating one result must not poison
    the next caller's V_ne."""
    gj._V_NE_GAMMA_CACHE.clear()
    system, basis = _vne_cache_fixture()

    first = gj._ewald_v_ne_gamma(basis, system)
    reference = first.copy()
    first *= 2.0  # a caller scaling its own copy in place

    second = gj._ewald_v_ne_gamma(basis, system)
    np.testing.assert_allclose(second, reference, atol=0.0, rtol=0.0)


def test_ewald_v_ne_gamma_cache_is_bounded():
    """The memo must not grow without limit across geometries."""
    gj._V_NE_GAMMA_CACHE.clear()
    system, basis = _vne_cache_fixture()

    for z in (3.0, 2.8, 2.6, 2.4, 2.2):
        system.unit_cell[1].xyz = [0.0, 0.0, z]
        gj._ewald_v_ne_gamma(basis, system)

    assert len(gj._V_NE_GAMMA_CACHE) <= gj._V_NE_GAMMA_CACHE_DEPTH


# ---------- SAP routing (GitLab #667) -----------------------------------


@pytest.mark.parametrize(
    ("driver_name", "functional"),
    [
        ("run_periodic_rhf_gpw", None),
        ("run_periodic_rks_gpw", "lda"),
    ],
    ids=["rhf", "rks"],
)
def test_gamma_gpw_sap_reaches_lattice_potential(
    monkeypatch,
    driver_name,
    functional,
):
    """Both closed-shell Gamma GPW entry points execute periodic SAP."""
    import vibeqc.guess as guess_module

    class SapPotentialReached(RuntimeError):
        pass

    calls = []

    def stop_at_vsap(basis, system, _grid, table, lattice_opts):
        calls.append((basis, system, table, lattice_opts))
        raise SapPotentialReached

    monkeypatch.setattr(guess_module, "compute_vsap_lattice", stop_at_vsap)
    system = _he_periodic_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)
    kwargs = {"functional": functional} if functional is not None else {}

    with pytest.raises(SapPotentialReached):
        getattr(gj, driver_name)(
            system,
            basis,
            grid=grid,
            max_iter=1,
            initial_guess=core.InitialGuess.SAP,
            quiet=True,
            **kwargs,
        )

    assert len(calls) == 1
    assert calls[0][0] is basis
    assert calls[0][1] is system
    assert calls[0][2] == "sap_helfem_large"


def test_multik_gpw_sap_reaches_lattice_potential(monkeypatch):
    """The pure-DFT multi-k GPW entry point builds ``F_SAP(k)``."""
    import vibeqc.guess as guess_module

    class SapPotentialReached(RuntimeError):
        pass

    calls = []

    def stop_at_vsap(_basis, _system, _grid, table, _lattice_opts):
        calls.append(table)
        raise SapPotentialReached

    monkeypatch.setattr(guess_module, "compute_vsap_lattice", stop_at_vsap)
    system = _he_periodic_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)
    kmesh = core.monkhorst_pack(system, [2, 1, 1])

    with pytest.raises(SapPotentialReached):
        gj.run_periodic_rks_gpw_multi_k(
            system,
            basis,
            kmesh,
            functional="lda",
            grid=grid,
            max_iter=1,
            initial_guess=core.InitialGuess.SAP,
            quiet=True,
        )

    assert calls == ["sap_helfem_large"]


@pytest.mark.parametrize(
    "selector",
    [
        core.InitialGuess.SAP,
        core.InitialGuess.PATOM,
        core.InitialGuess.FRAGMO,
    ],
)
def test_gamma_gpw_restart_density_precedes_selector(monkeypatch, selector):
    """A caller-provided Gamma density is the effective READ artifact."""
    import vibeqc.guess as guess_module

    class RestartDensityReached(RuntimeError):
        pass

    def unexpected_vsap(*_args, **_kwargs):
        pytest.fail("guess construction must not run for an explicit density")

    restart_density = np.array([[0.375]])

    def stop_at_first_j(_builder, density):
        assert np.trace(density @ vq.compute_overlap(basis)) == pytest.approx(2., abs=1e-12)
        raise RestartDensityReached

    monkeypatch.setattr(guess_module, "compute_vsap_lattice", unexpected_vsap)
    monkeypatch.setattr(gj.GpwJBuilder, "build_J", stop_at_first_j)
    system = _he_periodic_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)

    expected = NotImplementedError if selector == core.InitialGuess.FRAGMO else RestartDensityReached
    with pytest.raises(expected):
        gj.run_periodic_rhf_gpw(
            system,
            basis,
            grid=grid,
            max_iter=1,
            initial_density=restart_density,
            initial_guess=selector,
            quiet=True,
        )


def test_gamma_gpw_restart_rejects_malformed_selector():
    """A restart density must not hide an invalid guess spelling."""
    system = _he_periodic_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(ValueError, match="unknown initial_guess='bogus'"):
        gj.run_periodic_rhf_gpw(
            system,
            basis,
            initial_density=np.array([[0.375]]),
            initial_guess="bogus",
            quiet=True,
        )


@pytest.mark.parametrize(
    ("selector", "effective"),
    [
        (core.InitialGuess.SAD, core.InitialGuess.SAD),
        (" auto ", core.InitialGuess.SAD),
        ("minao", core.InitialGuess.MINAO),
    ],
    ids=["enum-sad", "string-auto", "string-minao"],
)
def test_gamma_gpw_density_guesses_use_shared_adapter(
    monkeypatch,
    selector,
    effective,
):
    """Direct GPW selectors share normalization and density construction."""
    import vibeqc.guess as guess_module

    class SharedGuessReached(RuntimeError):
        pass

    seen = []

    def stop_at_shared_guess(_mol, _basis, _n_occ, guess, **kwargs):
        seen.append((guess, kwargs["periodic_system"]))
        raise SharedGuessReached

    monkeypatch.setattr(
        guess_module,
        "initial_density_closed_shell",
        stop_at_shared_guess,
    )
    system = _he_periodic_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)

    with pytest.raises(SharedGuessReached):
        gj.run_periodic_rhf_gpw(
            system,
            basis,
            grid=grid,
            max_iter=1,
            initial_guess=selector,
            quiet=True,
        )

    assert seen == [(effective, system)]


def test_multik_gpw_hueckel_uses_shared_fock_adapter(monkeypatch):
    """The multi-k GPW route preserves HUECKEL as a Fock-mode guess."""
    import vibeqc.guess as guess_module

    class SharedFockGuessReached(RuntimeError):
        pass

    seen = []

    def stop_at_shared_fock(_system, _basis, _kpoints, guess, **_kwargs):
        seen.append(guess)
        raise SharedFockGuessReached

    monkeypatch.setattr(
        guess_module,
        "periodic_fock_guess_k",
        stop_at_shared_fock,
    )
    system = _he_periodic_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)
    kmesh = core.monkhorst_pack(system, [2, 1, 1])

    with pytest.raises(SharedFockGuessReached):
        gj.run_periodic_rks_gpw_multi_k(
            system,
            basis,
            kmesh,
            functional="lda",
            grid=grid,
            max_iter=1,
            initial_guess="huckel",
            quiet=True,
        )

    assert seen == [core.InitialGuess.HUECKEL]


@pytest.mark.parametrize(
    ("driver_name", "args", "kwargs"),
    [
        (
            "run_periodic_rks_gpw_multi_k",
            (None, None, None),
            {"functional": "lda"},
        ),
    ],
    ids=["multik"],
)
def test_direct_gpw_guess_selectors_fail_closed_for_patom(
    driver_name,
    args,
    kwargs,
):
    """A recognized guess without a GPW implementation fails early."""
    with pytest.raises(
        NotImplementedError,
        match="initial_guess=PATOM.*not implemented by this route",
    ):
        getattr(gj, driver_name)(
            *args,
            initial_guess=core.InitialGuess.PATOM,
            **kwargs,
        )
