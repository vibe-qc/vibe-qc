"""Periodic-method acceptance across the 7 crystal systems.

The 14 Bravais lattices differ by metric constraints and centering.
The numerical kernels only see a full-rank 3x3 lattice matrix; centering
is represented by the basis atoms. These tests pin the method-level
contract that all crystal-system metrics are accepted.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import vibeqc as vq

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _lattice_from_cellpar(
    a: float,
    b: float,
    c: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> np.ndarray:
    """Return a column-vector lattice matrix from cell parameters."""
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sg = math.sin(gamma)
    if abs(sg) < 1e-14:
        raise ValueError("gamma too close to 0 or 180 degrees")

    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * cg, b * sg, 0.0])
    c_x = c * cb
    c_y = c * (ca - cb * cg) / sg
    c_z_sq = c * c - c_x * c_x - c_y * c_y
    if c_z_sq <= 0.0:
        raise ValueError("cell parameters do not form a positive-volume cell")
    c_vec = np.array([c_x, c_y, math.sqrt(c_z_sq)])
    return np.column_stack([a_vec, b_vec, c_vec])


def _cart(lattice: np.ndarray, frac) -> list[float]:
    return (lattice @ np.asarray(frac, dtype=float)).tolist()


def _fcc_diamond() -> vq.PeriodicSystem:
    """Diamond primitive FCC cell: CRYSTAL space group Fd-3m."""
    a = 3.567 * ANGSTROM_TO_BOHR
    lattice = 0.5 * a * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    return vq.PeriodicSystem(3, lattice, [
        vq.Atom(6, _cart(lattice, [0.0, 0.0, 0.0])),
        vq.Atom(6, _cart(lattice, [0.25, 0.25, 0.25])),
    ])


def _bcc_fe_conventional() -> vq.PeriodicSystem:
    """Body-centred cubic Fe represented as a conventional cell + basis."""
    a = 2.866 * ANGSTROM_TO_BOHR
    lattice = a * np.eye(3)
    return vq.PeriodicSystem(3, lattice, [
        vq.Atom(26, _cart(lattice, [0.0, 0.0, 0.0])),
        vq.Atom(26, _cart(lattice, [0.5, 0.5, 0.5])),
    ])


def _graphene_hexagonal() -> vq.PeriodicSystem:
    """2D hexagonal graphene with a vacuum third axis."""
    a = 2.46 * ANGSTROM_TO_BOHR
    vacuum = 30.0
    lattice = np.column_stack([
        [a, 0.0, 0.0],
        [0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0],
        [0.0, 0.0, vacuum],
    ])
    return vq.PeriodicSystem(2, lattice, [
        vq.Atom(6, _cart(lattice, [0.0, 0.0, 0.5])),
        vq.Atom(6, _cart(lattice, [1.0 / 3.0, 2.0 / 3.0, 0.5])),
    ])


def _al2o3_hexagonal_smoke() -> vq.PeriodicSystem:
    """Small hexagonal Al2O3 stoichiometric cell for infrastructure tests.

    This is deliberately a light-weight BvK/lattice smoke target, not a
    full corundum Wyckoff expansion. The production corundum examples
    carry the R-3c conventional cell; here we only need an Al/O basis in
    a skew hexagonal metric so cubic-only assumptions fail loudly.
    """
    a = 4.7589 * ANGSTROM_TO_BOHR
    c = 12.991 * ANGSTROM_TO_BOHR
    lattice = _lattice_from_cellpar(a, a, c, 90.0, 90.0, 120.0)
    al_frac = [(0.0, 0.0, 0.35216), (0.0, 0.0, 0.64784)]
    o_frac = [
        (0.30624, 0.0, 0.25),
        (0.0, 0.30624, 0.75),
        (0.69376, 0.69376, 0.25),
    ]
    atoms = (
        [vq.Atom(13, _cart(lattice, f)) for f in al_frac]
        + [vq.Atom(8, _cart(lattice, f)) for f in o_frac]
    )
    return vq.PeriodicSystem(3, lattice, atoms)


LATTICE_FAMILY_CASES = [
    ("triclinic", _lattice_from_cellpar(11.0, 12.0, 13.0, 80.0, 75.0, 70.0)),
    ("monoclinic", _lattice_from_cellpar(11.0, 12.0, 13.0, 90.0, 105.0, 90.0)),
    ("orthorhombic", _lattice_from_cellpar(10.0, 12.0, 14.0, 90.0, 90.0, 90.0)),
    ("tetragonal", _lattice_from_cellpar(10.0, 10.0, 14.0, 90.0, 90.0, 90.0)),
    ("trigonal", _lattice_from_cellpar(11.0, 11.0, 11.0, 75.0, 75.0, 75.0)),
    ("hexagonal", _lattice_from_cellpar(10.0, 10.0, 16.0, 90.0, 90.0, 120.0)),
    ("cubic", _lattice_from_cellpar(12.0, 12.0, 12.0, 90.0, 90.0, 90.0)),
]

CRYSTAL_TARGET_CASES = [
    ("fcc_diamond", _fcc_diamond, (3, 3, 3), 227),
    ("bcc_fe", _bcc_fe_conventional, (3, 3, 3), 229),
    ("hexagonal_graphene", _graphene_hexagonal, (3, 3, 1), None),
    ("hexagonal_al2o3", _al2o3_hexagonal_smoke, (2, 2, 2), None),
]


def _h2_system_for_lattice(dim: int, lattice: np.ndarray) -> tuple:
    centre = lattice @ np.array([0.5, 0.5, 0.5])
    if dim == 1:
        axis = lattice[:, 0] / np.linalg.norm(lattice[:, 0])
    elif dim == 2:
        axis = np.cross(lattice[:, 0], lattice[:, 1])
        axis = axis / np.linalg.norm(axis)
    else:
        axis = np.array([0.0, 0.0, 1.0])
    atoms = [
        vq.Atom(1, (centre - 0.7 * axis).tolist()),
        vq.Atom(1, (centre + 0.7 * axis).tolist()),
    ]
    system = vq.PeriodicSystem(dim, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


LOW_DIMENSIONAL_GDF_CASES = [
    (
        "1d-oblique",
        1,
        np.column_stack([
            [9.0, 1.0, 0.0],
            [0.0, 40.0, 2.0],
            [0.0, 1.5, 38.0],
        ]),
    ),
    (
        "2d-hexagonal",
        2,
        _lattice_from_cellpar(12.0, 12.0, 40.0, 90.0, 90.0, 120.0),
    ),
]


@pytest.mark.parametrize(
    "family,lattice",
    LATTICE_FAMILY_CASES,
    ids=[name for name, _ in LATTICE_FAMILY_CASES],
)
def test_fft_poisson_accepts_all_crystal_families(family: str, lattice: np.ndarray):
    """The native FFT Poisson kernel supports every 3D crystal metric."""
    del family
    n = 8
    rho = np.zeros((n, n, n))
    rho[n // 2, n // 2, n // 2] = 1.0

    V = vq.solve_poisson_coulomb(rho, lattice)
    V_lr = vq.solve_poisson_erf_screened(rho, lattice, omega=0.5)

    assert V.shape == rho.shape
    assert V_lr.shape == rho.shape
    assert np.all(np.isfinite(V))
    assert np.all(np.isfinite(V_lr))
    assert abs(float(V.mean())) < 1e-12
    assert abs(float(V_lr.mean())) < 1e-12


@pytest.mark.parametrize(
    "family,lattice",
    LATTICE_FAMILY_CASES,
    ids=[name for name, _ in LATTICE_FAMILY_CASES],
)
def test_periodic_density_grid_accepts_all_crystal_families(
    family: str,
    lattice: np.ndarray,
):
    """Periodic AO density sampling follows arbitrary lattice vectors."""
    del family

    def cart(frac):
        return (lattice @ np.asarray(frac, dtype=float)).tolist()

    system = vq.PeriodicSystem(3, lattice, [
        vq.Atom(1, cart([0.50, 0.50, 0.45])),
        vq.Atom(1, cart([0.50, 0.50, 0.55])),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 4.0
    opts.nuclear_cutoff_bohr = 4.0
    S_lat = vq.compute_overlap_lattice(basis, system, opts)
    rho, grid_shape = vq.evaluate_periodic_density_on_grid(
        basis, system, S_lat, grid_shape=(4, 4, 4),
    )

    assert grid_shape == (4, 4, 4)
    assert rho.shape == grid_shape
    assert np.all(np.isfinite(rho))


@pytest.mark.parametrize(
    "family,lattice",
    LATTICE_FAMILY_CASES,
    ids=[name for name, _ in LATTICE_FAMILY_CASES],
)
def test_gamma_gdf_rhf_accepts_all_3d_crystal_families(
    family: str,
    lattice: np.ndarray,
):
    """Native Γ-GDF SCF must follow the supplied metric, not cubic axes."""
    del family
    system, basis = _h2_system_for_lattice(3, lattice)
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0
    opts.max_iter = 40
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-7
    opts.damping = 0.2
    opts.initial_guess = vq.InitialGuess.HCORE

    result = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        progress=False,
    )

    assert result.converged
    assert result.backend == "native-gamma-gdf"
    assert np.isfinite(result.energy)


@pytest.mark.parametrize(
    "name,dim,lattice",
    LOW_DIMENSIONAL_GDF_CASES,
    ids=[name for name, *_ in LOW_DIMENSIONAL_GDF_CASES],
)
def test_gamma_gdf_rhf_accepts_low_dimensional_nonorthogonal_lattices(
    name: str,
    dim: int,
    lattice: np.ndarray,
):
    """1D/2D native Γ-GDF accepts non-orthogonal embedding cells."""
    del name
    system, basis = _h2_system_for_lattice(dim, lattice)
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0
    opts.max_iter = 40
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-7
    opts.damping = 0.2
    opts.initial_guess = vq.InitialGuess.HCORE

    result = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        progress=False,
    )

    assert int(system.dim) == dim
    assert result.converged
    assert result.backend == "native-gamma-gdf"
    assert np.isfinite(result.energy)


@pytest.mark.parametrize(
    "family,lattice",
    LATTICE_FAMILY_CASES,
    ids=[name for name, _ in LATTICE_FAMILY_CASES],
)
def test_auto_grid_accepts_all_crystal_families(family: str, lattice: np.ndarray):
    del family
    grid_shape = vq.auto_grid(lattice, spacing_bohr=2.5)
    assert len(grid_shape) == 3
    assert all(n > 0 and n % 2 == 0 for n in grid_shape)


@pytest.mark.parametrize(
    "name,builder,mesh,expected_sg",
    CRYSTAL_TARGET_CASES,
    ids=[name for name, *_ in CRYSTAL_TARGET_CASES],
)
def test_general_bvk_infrastructure_on_crystal_targets(
    name: str,
    builder,
    mesh: tuple[int, int, int],
    expected_sg: int | None,
):
    """Reciprocal space, MP grids, and lattice cells are lattice-general.

    These targets mirror the first non-toy structures we want in the
    CRYSTAL/PySCF/vibe-qc parity matrix: FCC diamond, BCC Fe,
    hexagonal graphene, and Al2O3 stoichiometry in a hexagonal metric.
    They exercise primitive cells, conventional centered cells
    represented by the basis, 2D hexagonal PBC, and skew 3D hexagonal
    cells that would catch cubic-only assumptions.
    """
    del name, expected_sg
    system = builder()
    lattice = np.asarray(system.lattice, dtype=float)

    # Born-von Karman reciprocal convention: a_i · b_j = 2πδ_ij.
    reciprocal = np.asarray(system.reciprocal_lattice(), dtype=float)
    np.testing.assert_allclose(
        lattice.T @ reciprocal,
        2.0 * math.pi * np.eye(3),
        atol=1e-12,
    )

    kpts = vq.KPoints.gamma_centred(system, mesh)
    assert kpts.mesh == mesh
    np.testing.assert_allclose(kpts.weights.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        (reciprocal @ kpts.kpoints_frac.T).T,
        kpts.kpoints_cart,
        atol=1e-12,
    )

    periodic_lengths = [
        float(np.linalg.norm(lattice[:, i])) for i in range(system.dim)
    ]
    cutoff = 1.10 * min(periodic_lengths)
    cells = vq.direct_lattice_cells(system, cutoff)
    assert len(cells) > 1

    seen = set()
    for cell in cells:
        idx = tuple(int(x) for x in np.asarray(cell.index).reshape(3))
        seen.add(idx)
        r_cart = np.asarray(cell.r_cart, dtype=float)
        np.testing.assert_allclose(r_cart, lattice @ np.asarray(idx), atol=1e-12)
        assert float(np.linalg.norm(r_cart)) <= cutoff + 1e-12
        if system.dim < 3:
            assert all(idx[i] == 0 for i in range(system.dim, 3))
    for idx in seen:
        assert tuple(-x for x in idx) in seen


@pytest.mark.parametrize(
    "name,builder,mesh,expected_sg",
    [case for case in CRYSTAL_TARGET_CASES if case[3] is not None],
    ids=[case[0] for case in CRYSTAL_TARGET_CASES if case[3] is not None],
)
def test_spglib_ibz_reduction_on_3d_crystal_targets(
    name: str,
    builder,
    mesh: tuple[int, int, int],
    expected_sg: int,
):
    """Space-group detection and IBZ reduction are not cubic special cases."""
    del name
    system = builder()
    vq.attach_symmetry(system, symprec=1e-4)

    assert system.symmetry.number == expected_sg
    assert system.symmetry.order >= 48

    kpts = vq.KPoints.gamma_centred(system, mesh, symmetry=True)
    full_count = int(np.prod(mesh))
    assert 1 <= len(kpts) <= full_count
    np.testing.assert_allclose(kpts.weights.sum(), 1.0, atol=1e-12)


def test_rsgdf_g_mesh_uses_reciprocal_lattice_columns_on_hexagonal_cells():
    """Range-separated GDF LR meshes must follow ``B @ n``.

    A row/column mix-up is invisible for cubic cells but generates
    non-reciprocal vectors for hexagonal or triclinic cells.
    """
    from vibeqc.aux_basis import rsgdf_g_mesh

    system = _al2o3_hexagonal_smoke()
    reciprocal = np.asarray(system.reciprocal_lattice(), dtype=float)
    g_mesh = rsgdf_g_mesh(system, omega=0.8, precision=1e-4)

    # Fundamental reciprocal directions are short enough to be present.
    for i in range(3):
        b_i = reciprocal[:, i]
        assert np.min(np.linalg.norm(g_mesh - b_i, axis=1)) < 1.0e-12
        assert np.min(np.linalg.norm(g_mesh + b_i, axis=1)) < 1.0e-12

    # Every generated vector must have integer fractional coordinates in
    # the reciprocal basis.
    frac = np.linalg.solve(reciprocal, g_mesh.T).T
    np.testing.assert_allclose(frac, np.rint(frac), atol=1.0e-12)


def test_rsgdf_low_dimensional_mesh_requires_explicit_opt_in():
    """A dim<3 RSGDF mesh spans only the periodic axes -- correct support for a
    Bloch phase, wrong support for the ``4π/|G|²`` Coulomb kernel its consumers
    apply to it (the transverse average turns ``1/r`` into a sheet term ``∝1/V``;
    fixed 2026-07-10, see ``tests/test_ccm_lowd_gauge_consistency.py``). The
    builders now refuse unless the caller opts in for a non-Coulomb use.

    Shape assertions below therefore run on the opted-in mesh: the collapse itself
    is still what these builders produce, it just may not be obtained by accident.
    """
    from vibeqc.aux_basis import rsgdf_dense_g_mesh, rsgdf_g_mesh

    chain = vq.PeriodicSystem(
        1,
        np.diag([15.0, 40.0, 40.0]),
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    with pytest.raises(NotImplementedError, match="transverse-collapsed"):
        rsgdf_dense_g_mesh(chain, ke_cutoff=200.0)
    with pytest.raises(NotImplementedError, match="transverse-collapsed"):
        rsgdf_g_mesh(chain, omega=0.8, precision=1.0e-4)

    dense_1d = rsgdf_dense_g_mesh(chain, ke_cutoff=200.0,
                                  allow_transverse_collapse=True)
    omega_1d = rsgdf_g_mesh(chain, omega=0.8, precision=1.0e-4,
                            allow_transverse_collapse=True)
    assert dense_1d.shape[0] < 200
    assert omega_1d.shape[0] < 100
    np.testing.assert_allclose(dense_1d[:, 1:], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(omega_1d[:, 1:], 0.0, atol=1.0e-14)

    slab = _graphene_hexagonal()
    dense_2d = rsgdf_dense_g_mesh(slab, ke_cutoff=200.0,
                                  allow_transverse_collapse=True)
    assert dense_2d.shape[0] < 5000
    np.testing.assert_allclose(dense_2d[:, 2], 0.0, atol=1.0e-14)
