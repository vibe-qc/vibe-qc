"""AO-reach screening of the periodic XC value kernels (vibe-qc#268, #299, #53)
and the grid electron count they report (vibe-qc#385).

``build_xc_periodic`` / ``build_xc_periodic_uks`` evaluate the density on the
periodic Becke grid as a double sum over lattice images of the AOs. They used
to evaluate every AO of every image cell inside ``cutoff_bohr`` at every grid
point and to contract every bra-ket image pair, however far each shell
actually reaches. The screen keeps, per grid batch, only the image shells
whose Gaussian extent (``compute_shell_radial_cutoffs`` at
``ao_screening_tol``) reaches the batch, and skips zero-weight points. The
image domain itself is unchanged, so the screened result must equal the dense
one (``ao_screening_tol=0``) to far below any physical tolerance, while doing
measurably less work. A deliberately loose tolerance must move E_xc: that is
the control showing the screen really drops terms.

The kernels also report ``n_electrons_grid = sum_g w_g rho(r_g)``, the
quantity whose silent shortfall #385 (and #265 before it) reported.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc import monkhorst_pack
from vibeqc._vibeqc_core import (
    GridOptions,
    LatticeSumOptions,
    bloch_sum,
    build_xc_periodic,
    build_xc_periodic_uks,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    direct_lattice_cells,
    periodic_xc_screening_stats,
    real_space_density_from_kpoints,
)
from vibeqc.periodic_grid import build_periodic_becke_grid

_ANG = 1.0 / 0.529177210903
CUTOFF = 12.0
# Screened versus dense: the dropped AO amplitudes are below 1e-12, so the
# energy per cell and every V_xc element agree far below SCF tolerances.
E_TOL = 1e-10
V_TOL = 1e-9


def _fcc(a_bohr, z1, z2):
    lattice = (a_bohr / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    atoms = [vq.Atom(z1, [0.0, 0.0, 0.0]),
             vq.Atom(z2, [a_bohr / 2.0, a_bohr / 2.0, a_bohr / 2.0])]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


SYSTEMS = {
    # Dense ionic cell with compact STO-3G shells: the screen prunes.
    "mgo": lambda: _fcc(4.21 * _ANG, 12, 8),
    # Rocksalt LiH (a = 7.72 bohr, the CCM fixture): the diffuse Li 2sp shell
    # (exponent 0.048) reaches past the cutoff, so little can be pruned; the
    # screen must still be exact.
    "lih": lambda: _fcc(7.72, 3, 1),
}


def _lattice_options(image_radius=CUTOFF):
    opts = LatticeSumOptions()
    opts.cutoff_bohr = CUTOFF
    opts.nuclear_cutoff_bohr = CUTOFF
    opts.becke_image_radius_bohr = image_radius
    return opts


def _grid(system):
    gopts = GridOptions()
    gopts.n_radial = 35
    return build_periodic_becke_grid(system, grid_options=gopts,
                                     image_radius_bohr=CUTOFF)


def _density(system, basis, lat_opts):
    """Closed-shell multi-k density from a kinetic-Hamiltonian fold
    (T C = S C eps), as in test_periodic_xc_cross_cell.py: it carries real
    P(g != 0) blocks without any Coulomb build."""
    kmesh = monkhorst_pack(system, [2, 2, 2], use_symmetry=False)
    kpts = list(kmesh.kpoints)
    S_lat = compute_overlap_lattice(basis, system, lat_opts)
    T_lat = compute_kinetic_lattice(basis, system, lat_opts)
    n_occ = system.n_electrons() // 2
    C_per_k = []
    for k in kpts:
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k = np.asarray(bloch_sum(S_lat, k_arr))
        T_k = np.asarray(bloch_sum(T_lat, k_arr))
        S_k = 0.5 * (S_k + S_k.conj().T)
        T_k = 0.5 * (T_k + T_k.conj().T)
        _, C = sla.eigh(T_k, S_k)
        C_per_k.append(np.asarray(C, dtype=complex))
    cells = list(direct_lattice_cells(system, 2.0 * lat_opts.cutoff_bohr))
    return real_space_density_from_kpoints(
        C_per_k, [n_occ] * len(kpts), kmesh, cells
    )


def _half(D_real):
    return core.make_lattice_matrix_set(
        int(D_real.nbf), list(D_real.cells),
        [0.5 * np.asarray(b, dtype=float) for b in D_real.blocks],
    )


def _max_block_diff(a, b):
    return max(
        float(np.max(np.abs(np.asarray(x) - np.asarray(y)), initial=0.0))
        for x, y in zip(a.blocks, b.blocks)
    )


@pytest.fixture(scope="module", params=sorted(SYSTEMS))
def case(request):
    system, basis = SYSTEMS[request.param]()
    lat_opts = _lattice_options()
    return {
        "name": request.param,
        "system": system,
        "basis": basis,
        "grid": _grid(system),
        "opts": lat_opts,
        "density": _density(system, basis, lat_opts),
    }


@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
def test_screened_rks_matches_dense(case, functional):
    func = core.Functional(functional)
    args = (case["basis"], case["system"], case["grid"], func,
            case["density"], case["opts"])
    dense = build_xc_periodic(*args, ao_screening_tol=0.0)
    screened = build_xc_periodic(*args)  # default tolerance
    assert abs(screened.e_xc - dense.e_xc) <= E_TOL
    assert _max_block_diff(screened.V_xc, dense.V_xc) <= V_TOL
    assert abs(screened.n_electrons_grid - dense.n_electrons_grid) <= E_TOL
    assert screened.periodic_density_bras and dense.periodic_density_bras


@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
def test_screened_uks_matches_dense_and_rks(case, functional):
    half = _half(case["density"])
    func = core.Functional(functional, 2)
    args = (case["basis"], case["system"], case["grid"], func, half, half,
            case["opts"])
    dense = build_xc_periodic_uks(*args, ao_screening_tol=0.0)
    screened = build_xc_periodic_uks(*args)
    assert abs(screened.e_xc - dense.e_xc) <= E_TOL
    assert _max_block_diff(screened.V_alpha, dense.V_alpha) <= V_TOL
    assert _max_block_diff(screened.V_beta, dense.V_beta) <= V_TOL
    # Same total density as the closed-shell evaluation.
    rks = build_xc_periodic(case["basis"], case["system"], case["grid"],
                            core.Functional(functional), case["density"],
                            case["opts"])
    assert abs(screened.n_electrons_grid - rks.n_electrons_grid) <= E_TOL
    assert abs(screened.e_xc - rks.e_xc) <= 1e-8


def test_screen_prunes_compact_images():
    """Non-vacuity: on MgO/STO-3G the screen evaluates fewer image shells and
    bra-ket pairs than the dense kernel, and skips zero-weight points; with
    the screen disabled the counts are the dense ones."""
    system, basis = SYSTEMS["mgo"]()
    opts = _lattice_options()
    grid = _grid(system)
    D = _density(system, basis, opts)
    st = periodic_xc_screening_stats(basis, system, grid, D, opts)
    assert st.image_shells_live < st.image_shells_dense
    assert st.pairs_live < st.pairs_dense
    assert st.n_points_evaluated < st.n_points
    off = periodic_xc_screening_stats(basis, system, grid, D, opts,
                                      ao_screening_tol=0.0)
    assert off.image_shells_live == off.image_shells_dense
    assert off.pairs_live == off.pairs_dense
    assert off.n_points_evaluated == off.n_points
    # The live fraction is reported for the record (pytest -rA).
    print(f"MgO/STO-3G live pairs {st.pairs_live}/{st.pairs_dense}, "
          f"live image shells {st.image_shells_live}/{st.image_shells_dense}")


def test_loose_screen_moves_exc():
    """Control: an AO amplitude bound of 1e-2 drops shells that matter, so the
    energy must move. If it did not, the exactness test above would be
    vacuous (the screen would not be dropping anything)."""
    system, basis = SYSTEMS["mgo"]()
    opts = _lattice_options()
    grid = _grid(system)
    D = _density(system, basis, opts)
    func = core.Functional("lda")
    dense = build_xc_periodic(basis, system, grid, func, D, opts,
                              ao_screening_tol=0.0)
    loose = build_xc_periodic(basis, system, grid, func, D, opts,
                              ao_screening_tol=1e-2)
    assert abs(loose.e_xc - dense.e_xc) > 1e-6


def test_grid_electron_count_sees_truncated_image_reach():
    """#385: the reported grid integral is the electron count per cell when
    the image reach holds the density, and falls short when it does not --
    the silent failure the issue reported."""
    func = core.Functional("lda")
    opts = _lattice_options()
    # MgO: the kinetic-fold density is compact; cutoff 12 holds all 20
    # electrons (20.0019 on this grid).
    system, basis = SYSTEMS["mgo"]()
    xc = build_xc_periodic(basis, system, _grid(system), func,
                           _density(system, basis, opts), opts)
    assert xc.periodic_density_bras
    assert abs(xc.n_electrons_grid - 20.0) / 20.0 < 5e-3
    # LiH: the kinetic fold occupies the most diffuse Li functions, which
    # reach well past 12 bohr, so the grid visibly misses 4 electrons (3.43).
    # (Cutting the bra reach does not shrink it monotonically: image terms of
    # P(g) carry both signs, so only the count against N is diagnostic.)
    system, basis = SYSTEMS["lih"]()
    full = build_xc_periodic(basis, system, _grid(system), func,
                             _density(system, basis, opts), opts)
    assert abs(full.n_electrons_grid - 4.0) / 4.0 > 1e-2


@pytest.mark.parametrize("functional", ["lda", "pbe"])
def test_screened_potential_of_home_cell_density(case, functional):
    """A home-cell (molecular-limit) density has P(g != 0) = 0 but still a
    V_xc(g) block for every image pair: the potential is dE/dP(g). The
    density-term bound max|P(g)| must not drop those pairs from V_xc (the
    multi-k ROKS Ewald driver's first iteration exposed this)."""
    D = case["density"]
    home = [
        np.asarray(b, dtype=float) if not np.any(np.asarray(c.index))
        else np.zeros_like(np.asarray(b, dtype=float))
        for c, b in zip(D.cells, D.blocks)
    ]
    D_home = core.make_lattice_matrix_set(int(D.nbf), list(D.cells), home)
    half = core.make_lattice_matrix_set(int(D.nbf), list(D.cells),
                                        [0.5 * b for b in home])
    base = (case["basis"], case["system"], case["grid"])
    dense = build_xc_periodic(*base, core.Functional(functional), D_home,
                              case["opts"], ao_screening_tol=0.0)
    screened = build_xc_periodic(*base, core.Functional(functional), D_home,
                                 case["opts"])
    assert not dense.periodic_density_bras
    assert abs(screened.e_xc - dense.e_xc) <= E_TOL
    assert _max_block_diff(screened.V_xc, dense.V_xc) <= V_TOL
    f2 = core.Functional(functional, 2)
    dense_u = build_xc_periodic_uks(*base, f2, half, half, case["opts"],
                                    ao_screening_tol=0.0)
    screened_u = build_xc_periodic_uks(*base, f2, half, half, case["opts"])
    assert _max_block_diff(screened_u.V_alpha, dense_u.V_alpha) <= V_TOL
    assert _max_block_diff(screened_u.V_beta, dense_u.V_beta) <= V_TOL


def _torus_xc_case(mesh=(3, 1, 1), shift=(1, 0, 0)):
    """Small quadrature, actual ionic AO images and incomplete difference domain."""
    a = 7.72
    lattice = a / 2 * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    system = vq.PeriodicSystem(3, lattice, [
        vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [a / 2] * 3),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp")
    opts = core.LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    # Unequal bra/ket sets exercise unsymmetrized direct-k projections.
    opts.becke_image_radius_bohr = 9.0
    cells = list(core.direct_lattice_cells(system, 24.0))
    kmesh = vq.monkhorst_pack(system, list(mesh), is_shift=list(shift), use_symmetry=False)
    k = np.asarray(kmesh.kpoints)
    rng = np.random.default_rng(186)
    coeff = rng.normal(size=(3, basis.nbasis, basis.nbasis)) * .04
    orbitals = sum(np.exp(1j * (k @ lattice[:, i]))[:, None, None] * coeff[i]
                   for i in range(3))
    density_k = orbitals @ orbitals.conj().transpose(0, 2, 1)
    origins = np.array([cell.r_cart for cell in cells])
    blocks = np.einsum("k,gk,kij->gij", np.asarray(kmesh.weights),
                       np.exp(-1j * origins @ k.T), density_k, optimize=True).real
    density = core.make_lattice_matrix_set(basis.nbasis, cells, list(blocks))
    points = rng.uniform(-1., 5., size=(23, 3))
    grid = core.Grid(points, np.full(len(points), .03))
    return system, basis, opts, kmesh, density, grid, density_k


@pytest.mark.parametrize("mesh,shift", [((1, 1, 1), (0, 0, 0)),
                                        ((2, 2, 2), (0, 0, 0)),
                                        ((3, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("equal_bra", [False, True])
def test_torus_grouped_xc_matches_real_space_oracle(
    mesh, shift, functional, spin, equal_bra, torus_group_threads,
):
    system, basis, opts, kmesh, density, grid, _ = _torus_xc_case(mesh, shift)
    if equal_bra:
        opts.becke_image_radius_bohr = opts.cutoff_bohr
    fun = core.Functional(functional, spin)
    args = (basis, system, grid, fun)
    if spin == 1:
        old = core.build_xc_periodic(*args, density, opts)
        new = core.build_xc_periodic_kpoints(*args, density, kmesh, opts)
        pairs = [(old.V_xc, new.V_xc)]
    else:
        alpha = core.make_lattice_matrix_set(basis.nbasis, list(density.cells),
                                             [.7 * np.asarray(p) for p in density.blocks])
        beta = core.make_lattice_matrix_set(basis.nbasis, list(density.cells),
                                            [.3 * np.asarray(p) for p in density.blocks])
        old = core.build_xc_periodic_uks(*args, alpha, beta, opts)
        new = core.build_xc_periodic_uks_kpoints(*args, alpha, beta, kmesh, opts)
        pairs = [(old.V_alpha, new.V_alpha), (old.V_beta, new.V_beta)]
    assert new.torus_density_compatible and new.used_torus_grouping
    assert new.periodic_density_bras == old.periodic_density_bras
    assert new.e_xc == pytest.approx(old.e_xc, abs=2e-11, rel=0)
    assert new.n_electrons_grid == pytest.approx(old.n_electrons_grid, abs=2e-11, rel=0)
    for lattice, per_k in pairs:
        for k, got in zip(kmesh.kpoints, per_k):
            np.testing.assert_allclose(got, core.bloch_sum(lattice, k), atol=2e-10, rtol=2e-11)


@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
@pytest.mark.parametrize("spin", [1, 2])
def test_torus_projection_preserves_reordered_density_cells(
    functional, spin, torus_group_threads,
):
    """Image-list/cache ordering must preserve the retained finite operator."""
    system, basis, opts, km, density, grid, _ = _torus_xc_case((3, 1, 1), (1, 0, 0))
    cells, blocks = list(density.cells), list(density.blocks)
    # Keep the home cell first; permute the rest, including residue
    # representatives, instead of changing any Cartesian translation.
    order = np.r_[0, np.random.default_rng(417).permutation(np.arange(1, len(cells)))]

    def reordered(scale):
        return core.make_lattice_matrix_set(
            basis.nbasis, [cells[i] for i in order],
            [scale * np.asarray(blocks[i]) for i in order],
        )

    args = (basis, system, grid, core.Functional(functional, spin))
    if spin == 1:
        reference = core.build_xc_periodic(*args, density, opts)
        result = core.build_xc_periodic_kpoints(*args, reordered(1.), km, opts)
        channels = [(reference.V_xc, result.V_xc)]
    else:
        beta = core.make_lattice_matrix_set(
            basis.nbasis, cells, [.3 * np.asarray(p) for p in blocks],
        )
        reference = core.build_xc_periodic_uks(*args, density, beta, opts)
        result = core.build_xc_periodic_uks_kpoints(*args, reordered(1.), reordered(.3), km, opts)
        channels = [(reference.V_alpha, result.V_alpha), (reference.V_beta, result.V_beta)]
    assert result.torus_density_compatible and result.used_torus_grouping
    assert result.e_xc == pytest.approx(reference.e_xc, abs=2e-11, rel=0)
    assert result.n_electrons_grid == pytest.approx(reference.n_electrons_grid, abs=2e-11, rel=0)
    for lattice, values in channels:
        for k, value in zip(km.kpoints, values):
            np.testing.assert_allclose(value, core.bloch_sum(lattice, k), atol=2e-10, rtol=2e-11)


@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
@pytest.mark.parametrize("grouped", [False, True])
def test_uks_native_phase_timing_preserves_outputs(functional, grouped):
    system, basis, opts, km, alpha, grid, _ = _torus_xc_case()
    beta = core.make_lattice_matrix_set(basis.nbasis, list(alpha.cells),
                                      [.3 * p for p in alpha.blocks])
    args = (basis, system, grid, core.Functional(functional, 2), alpha, beta)

    def evaluate():
        if grouped:
            return core.build_xc_periodic_uks_kpoints(*args, km, opts)
        return core.build_xc_periodic_uks(*args, opts)

    with vq.perf_log() as disabled:
        expected = evaluate()
    with vq.perf_log(native=True) as enabled:
        actual = evaluate()
    assert disabled.native_phases == []
    assert len(enabled.native_phases) == 1
    record = enabled.native_phases[0]
    assert record["kernel"] == "periodic_xc.open_shell"
    assert record["completed"] and record["wall_seconds"] > 0
    phases = {p["name"]: p for p in record["phases"]}
    projection = "torus_potential_projection" if grouped else "potential_projection"
    assert set(phases) == {"setup", "ao_collocation", "pair_plan", "density", "functional", projection}
    assert all(p["measure"] == "elapsed_wall" and p["intervals"] > 0
               and p["seconds"] >= 0 for p in phases.values())
    assert sum(p["seconds"] for p in phases.values()) <= record["wall_seconds"]
    assert actual.e_xc == expected.e_xc
    assert actual.n_electrons_grid == expected.n_electrons_grid
    for channel in ("V_alpha", "V_beta"):
        a, b = getattr(actual, channel), getattr(expected, channel)
        np.testing.assert_array_equal(a if grouped else a.blocks, b if grouped else b.blocks)


@pytest.mark.parametrize("functional,beta_scale", [
    (functional, scale)
    for functional in ("lda", "pbe", "r2scan") for scale in (0., 1e-8, .4)
])
def test_torus_spin_fields_keep_independent_density_channels(
    functional, beta_scale, torus_group_threads,
):
    system, basis, opts, km, alpha, grid, per_k = _torus_xc_case((3, 1, 1), (1, 0, 0))
    # A separate positive density, not a scalar copy of alpha. Include an
    # empty spin and an imbalanced spin to catch shared-field contamination.
    beta_k = beta_scale * per_k[:, ::-1, ::-1]
    cells = list(alpha.cells)
    phases = np.exp(-1j * np.asarray([c.r_cart for c in cells]) @ np.asarray(km.kpoints).T)
    blocks = np.einsum("k,gk,kij->gij", np.asarray(km.weights), phases, beta_k).real
    beta = core.make_lattice_matrix_set(basis.nbasis, cells, list(blocks))
    args = (basis, system, grid, core.Functional(functional, 2))
    old = core.build_xc_periodic_uks(*args, alpha, beta, opts)
    new = core.build_xc_periodic_uks_kpoints(*args, alpha, beta, km, opts)
    assert new.torus_density_compatible and new.used_torus_grouping
    assert new.e_xc == pytest.approx(old.e_xc, rel=0., abs=2e-11)
    assert new.n_electrons_grid == pytest.approx(old.n_electrons_grid, rel=0., abs=2e-11)
    for lattice, values in ((old.V_alpha, new.V_alpha), (old.V_beta, new.V_beta)):
        for k, value in zip(km.kpoints, values):
            np.testing.assert_allclose(value, core.bloch_sum(lattice, k), atol=2e-10, rtol=2e-11)
    # Shared AO operands must not mix the distinct spin-dependent weights.
    swapped = core.build_xc_periodic_uks_kpoints(*args, beta, alpha, km, opts)
    assert swapped.used_torus_grouping
    assert swapped.e_xc == pytest.approx(new.e_xc, rel=0., abs=2e-11)
    np.testing.assert_allclose(swapped.V_alpha, new.V_beta, atol=2e-10, rtol=2e-11)
    np.testing.assert_allclose(swapped.V_beta, new.V_alpha, atol=2e-10, rtol=2e-11)


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("kind", ["nonperiodic", "tiny_nonperiodic", "local_guess", "ad_hoc_mesh"])
def test_torus_xc_falls_back_for_incompatible_density_or_mesh(spin, kind):
    system, basis, opts, kmesh, density, grid, _ = _torus_xc_case()
    blocks = [np.array(p) for p in density.blocks]
    if kind in ("nonperiodic", "tiny_nonperiodic"):
        blocks[0][0, 0] += .02
        if kind == "tiny_nonperiodic":
            blocks = [1e-18 * p for p in blocks]
    elif kind == "local_guess":
        blocks = [p if not np.any(cell.index) else np.zeros_like(p)
                  for p, cell in zip(blocks, density.cells)]
    else:
        kmesh = core.bloch_kmesh_from_lists(list(kmesh.kpoints), list(kmesh.weights))
    altered = core.make_lattice_matrix_set(basis.nbasis, list(density.cells), blocks)
    args = (basis, system, grid, core.Functional("pbe", spin))
    if spin == 1:
        old = core.build_xc_periodic(*args, altered, opts)
        new = core.build_xc_periodic_kpoints(*args, altered, kmesh, opts)
        matrices = [(old.V_xc, new.V_xc)]
    else:
        # Only beta is incompatible: alpha alone must not admit the fast path.
        old = core.build_xc_periodic_uks(*args, density, altered, opts)
        new = core.build_xc_periodic_uks_kpoints(*args, density, altered, kmesh, opts)
        matrices = [(old.V_alpha, new.V_alpha), (old.V_beta, new.V_beta)]
    assert not new.torus_density_compatible and not new.used_torus_grouping
    assert old.e_xc == new.e_xc
    assert old.n_electrons_grid == new.n_electrons_grid
    assert old.periodic_density_bras == new.periodic_density_bras
    for real, values in matrices:
        for k, value in zip(kmesh.kpoints, values):
            np.testing.assert_allclose(value, core.bloch_sum(real, k), atol=2e-12, rtol=2e-13)


@pytest.fixture(params=[1, 2])
def torus_group_threads(request):
    previous = vq.get_num_threads()
    vq.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        vq.set_num_threads(previous)


@pytest.mark.parametrize("tolerance", [0., 1e-12, 1e-3])
@pytest.mark.parametrize("density_scale", [.01, 1., 16.])
def test_torus_grouping_preserves_each_shell_and_pair_screen(
    tolerance, density_scale, torus_group_threads,
):
    system, basis, opts, kmesh, density, grid, _ = _torus_xc_case()
    # Density-weighted and potential-only pair screens need not retain the
    # same graph, even though both use the same AO image tables.
    density = core.make_lattice_matrix_set(
        basis.nbasis, list(density.cells),
        [density_scale * p for p in density.blocks],
    )
    fun = core.Functional("pbe")
    old = core.build_xc_periodic(basis, system, grid, fun, density, opts,
                                ao_screening_tol=tolerance)
    new = core.build_xc_periodic_kpoints(basis, system, grid, fun, density, kmesh, opts,
                                        ao_screening_tol=tolerance)
    assert new.torus_density_compatible
    assert new.e_xc == pytest.approx(old.e_xc, abs=2e-11, rel=0)
    for k, value in zip(kmesh.kpoints, new.V_xc):
        np.testing.assert_allclose(value, core.bloch_sum(old.V_xc, k), atol=2e-10, rtol=2e-11)


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("functional", ["pbe", "pbe0", "r2scan"])
@pytest.mark.parametrize("mesh,shift", [((1, 1, 1), (0, 0, 0)),
                                        ((3, 1, 1), (1, 0, 0))])
def test_torus_density_gradient_preserves_potential_adjoint(spin, functional, mesh, shift):
    system, basis, opts, km, density, grid, per_k = _torus_xc_case(mesh, shift)
    cells = list(density.cells)
    origins = np.asarray([c.r_cart for c in cells])
    phases = np.exp(-1j * origins @ np.asarray(km.kpoints).T)
    weights = np.asarray(km.weights)
    # The shifted case has nonsymmetric real-space blocks: P^T is essential
    # when evaluating the bra contribution to the density gradient.
    if any(shift):
        assert max(np.max(np.abs(p - p.T)) for p in density.blocks) > 1e-5
    direction = .2 * per_k + .01 * np.eye(basis.nbasis)[None, :, :]

    def fold(value):
        blocks = np.einsum("k,gk,kij->gij", weights, phases, value, optimize=True).real
        return core.make_lattice_matrix_set(basis.nbasis, cells, list(blocks))

    fun = core.Functional(functional, spin)
    reference = per_k if spin == 1 else .7 * per_k
    beta = fold(.3 * per_k) if spin == 2 else None

    # Keep the production screened batch size: the larger dense batch can
    # reject torus scratch before it reaches the contraction under test.
    def evaluate(value):
        args = (basis, system, grid, fun, fold(value))
        if spin == 1:
            result = core.build_xc_periodic_kpoints(*args, km, opts, ao_screening_tol=1e-12)
        else:
            result = core.build_xc_periodic_uks_kpoints(*args, beta, km, opts,
                                                       ao_screening_tol=1e-12)
        assert result.torus_density_compatible and result.used_torus_grouping
        return result

    center = evaluate(reference)
    step = 1e-5
    plus, minus = evaluate(reference + step * direction), evaluate(reference - step * direction)
    finite_difference = (plus.e_xc - minus.e_xc) / (2 * step)
    values = center.V_xc if spin == 1 else center.V_alpha
    adjoint = sum(w * np.trace(h @ v).real for w, h, v in zip(weights, direction, values))
    assert adjoint == pytest.approx(finite_difference, abs=2e-8, rel=2e-7)


@pytest.mark.parametrize("spin", [1, 2])
def test_torus_large_uniform_mesh_matches_real_space_oracle(spin):
    # A valid mesh beyond the compact planner index must preserve the same
    # retained graph and k-point order through the general lookup path.
    system, basis, opts, mesh, density, grid, _ = _torus_xc_case(
        (65, 1, 1), (1, 0, 0),
    )
    args = (basis, system, grid, core.Functional("pbe", spin), density)
    if spin == 1:
        old = core.build_xc_periodic(*args, opts)
        new = core.build_xc_periodic_kpoints(*args, mesh, opts)
        matrices = [(old.V_xc, new.V_xc)]
    else:
        beta = core.make_lattice_matrix_set(
            basis.nbasis, list(density.cells), [.3 * p for p in density.blocks],
        )
        old = core.build_xc_periodic_uks(*args, beta, opts)
        new = core.build_xc_periodic_uks_kpoints(*args, beta, mesh, opts)
        matrices = [(old.V_alpha, new.V_alpha), (old.V_beta, new.V_beta)]
    assert new.torus_density_compatible
    assert new.e_xc == pytest.approx(old.e_xc, abs=2e-11, rel=0)
    assert new.n_electrons_grid == pytest.approx(old.n_electrons_grid, abs=2e-11, rel=0)
    for lattice, values in matrices:
        assert len(values) == 65
        for k, value in zip(mesh.kpoints, values):
            np.testing.assert_allclose(value, core.bloch_sum(lattice, k),
                                       atol=2e-10, rtol=2e-11)


def test_compiled_torus_plan_reuse_preserves_screened_covers(tmp_path):
    """Compile the production planner; compare both covers with two builds.

    This is a source-level native regression, not a rebuilt XC/SCF test.
    Only AO-table liveness/width is a fixture; all rectangle-planning code
    and context fields below are taken directly from the translation unit.
    """
    from pathlib import Path
    import shutil
    import subprocess

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the source planner regression")
    source = (Path(__file__).resolve().parents[1] / "cpp/src/periodic_xc.cpp").read_text()

    def section(first, after):
        start = source.index(first)
        return source[start:source.index(after, start)]

    declarations = section("struct BraKetPair {", "// Enumerate the (bra, ket)")
    declarations += section("struct XCTorusContext {", "XCTorusContext xc_torus_context(")
    planner = section("struct XCTorusRectangle {", "struct XCTorusAOView {")
    code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <utility>
#include <vector>
struct ScreenedImageAO {
    std::vector<int> bfs;
    bool live() const { return !bfs.empty(); }
};
''' + declarations + planner + r'''
bool equal(const XCTorusPlan& a, const XCTorusPlan& b) {
    if (a.size()!=b.size()) return false;
    for (std::size_t i=0;i<a.size();++i)
        if (a[i].bra!=b[i].bra || a[i].ket!=b[i].ket ||
            a[i].gp!=b[i].gp || a[i].sign!=b[i].sign) return false;
    return true;
}
XCTorusContext context(int images,int mesh,bool signs,bool complete) {
    XCTorusContext c;c.compatible=true;c.complete=complete;
    c.unit_signs=!signs;c.mesh={mesh,1,1};
    for(int i=0;i<images;++i) {
        std::array<int,3> r{i%mesh,0,0};
        c.residue.push_back(r);c.rep.emplace(r,i);
        c.sign.push_back(signs && (i/mesh)%2 ? -1. : 1.);
    }
    return c;
}
void check(XCTorusContext c,const std::vector<BraKetPair>& pairs,
           const std::vector<int>& d,const std::vector<int>& v,
           const std::vector<ScreenedImageAO>& tab,int nbf,bool null=false) {
    auto old_context=c;
    auto old_d=xc_torus_plan(null?nullptr:&old_context,pairs,d,&tab,nbf);
    auto old_v=xc_torus_plan(null?nullptr:&old_context,pairs,v,&tab,nbf);
    auto result=xc_torus_plans(null?nullptr:&c,pairs,d,v,&tab,nbf);
    assert(equal(old_d,result.density) && equal(old_v,result.potential()));
    assert(c.used==old_context.used);
    assert(result.shared==(d==v));
    assert((&result.density==&result.potential())==(d==v));
    auto moved=std::move(result);
    assert(equal(old_v,moved.potential()));
}
int main() {
    constexpr int n=12;
    std::vector<BraKetPair> pairs;
    for(int a=0;a<n;++a) for(int s=0;s<n;++s) pairs.push_back({a,s,(s-a+n)%n});
    std::vector<int> all(pairs.size());std::iota(all.begin(),all.end(),0);
    int cases=0;
    for(int mesh:{1,2,3}) for(bool signs:{false,true})
    for(bool complete:{false,true}) for(int nbf:{1,6,32}) {
        auto c=context(n,mesh,signs,complete);
        std::vector<ScreenedImageAO> tab(n);
        for(auto& a:tab) a.bfs.resize(nbf);
        auto d=all;d.erase(d.begin()+3);
        auto v=all;v.erase(v.begin()+4); // equal size, different screened edge
        auto reverse=all;std::reverse(reverse.begin(),reverse.end());
        for(const auto& dv:std::vector<std::pair<std::vector<int>,std::vector<int>>>{
              {all,all},{d,all},{d,v},{all,reverse},{{},all},{{},{}}}) {
            check(c,pairs,dv.first,dv.second,tab,nbf);++cases;
        }
        check(c,pairs,all,all,tab,nbf,true);++cases;
        // Reduced AO width and sparse retained graph must preserve the
        // original cost-based ungrouped fallback too.
        for(auto& a:tab)a.bfs.resize(1);
        check(c,pairs,{0},{0},tab,nbf);++cases;
    }
    std::cout<<"cases "<<cases<<"\n";
    // Descriptive microbenchmark, never a noisy wall-time pass/fail gate.
    constexpr int count=80, images=96, nbf=6;
    auto c=context(images,1,false,false);
    pairs.clear();
    for(int a=0;a<images;++a)for(int s=0;s<images;++s)pairs.push_back({a,s,(s-a+images)%images});
    all.resize(pairs.size());std::iota(all.begin(),all.end(),0);
    std::vector<ScreenedImageAO> tab(images);for(auto& a:tab)a.bfs.resize(nbf);
    std::size_t checksum=0;
    for(int mode:{0,1,1,0}) {
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<count;++i) {
            if(mode) {const auto p=xc_torus_plans(&c,pairs,all,all,&tab,nbf);checksum+=p.density.size()+p.potential().size();}
            else {checksum+=xc_torus_plan(&c,pairs,all,&tab,nbf).size();checksum+=xc_torus_plan(&c,pairs,all,&tab,nbf).size();}
        }
        std::cout<<(mode?"reused ":"separate ")
                 <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n";
    }
    assert(checksum>0);
}
'''
    cpp = tmp_path / "plan_reuse.cpp"
    exe = tmp_path / "plan_reuse"
    cpp.write_text(code)
    subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(exe)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert "cases 288" in result.stdout
    print(result.stdout)


@pytest.mark.parametrize("functional", ["tpss", "m06-l", "r2scan"])
@pytest.mark.parametrize("channel", ["restricted", "alpha", "beta"])
@pytest.mark.parametrize("floor_active", [False, True])
def test_mgga_tau_floor_energy_density_and_geometry_adjoint(functional, channel, floor_active):
    """Differentiate the implemented energy on both sides of the tau floor.

    The active-floor fixture deliberately has a non-PSD density, with positive
    sampled rho. It isolates the numerical extension accepted by this kernel;
    it is not a physical converged state. The inactive fixture is positive
    definite. The grid stays fixed for the geometry derivative.
    """
    points = np.array([[.2, .1, .3]])
    grid = core.Grid(points, np.array([1.]))
    opts = core.LatticeSumOptions()
    opts.cutoff_bohr = opts.becke_image_radius_bohr = 2.
    density = np.array([[1., 1.2], [1.2, 1.]]) if floor_active else np.eye(2)
    direction = np.array([[.1, .3], [.3, -.2]])
    other = .5*np.eye(2)
    spin = 1 if channel == "restricted" else 2
    fun = core.Functional(functional, spin)

    def geometry(displacement=0.):
        s = vq.PeriodicSystem(3, np.eye(3)*8., [
            vq.Atom(1, [0., 0., -.7]), vq.Atom(1, [0., 0., .7+displacement]),
        ])
        return s, vq.BasisSet(s.unit_cell_molecule(), "sto-3g")

    system, basis = geometry()
    ao, *grad = (np.asarray(x) for x in core.evaluate_ao_with_gradient(basis, points))
    rho = np.einsum('pi,ij,pj->p', ao, density, ao)
    grad_rho = np.array([2*np.einsum('pi,ij,pj->p', x, density, ao) for x in grad])
    tau = .5*sum(np.einsum('pi,ij,pj->p', x, density, x) for x in grad)
    tau_w = np.sum(grad_rho**2, axis=0)/(8*rho)
    assert rho[0] > 0
    assert bool(tau[0] < tau_w[0]) == floor_active
    assert abs(tau[0]-tau_w[0]) > 1e-3  # differences do not cross the kink

    def operands(s, b, d):
        cells = [core.LatticeCell(s, (0, 0, 0))]
        fold = lambda x: core.make_lattice_matrix_set(b.nbasis, cells, [x])
        if spin == 1:
            return (b, s, grid, fun, fold(d), opts)
        alpha, beta = (d, other) if channel == "alpha" else (other, d)
        return (b, s, grid, fun, fold(alpha), fold(beta), opts)

    build = core.build_xc_periodic if spin == 1 else core.build_xc_periodic_uks
    center = build(*operands(system, basis, density))
    values = center.V_xc if spin == 1 else getattr(center, "V_"+channel)
    adjoint = np.sum(np.asarray(values.blocks[0])*direction)
    step = 1e-5
    ep = build(*operands(system, basis, density+step*direction)).e_xc
    em = build(*operands(system, basis, density-step*direction)).e_xc
    assert adjoint == pytest.approx((ep-em)/(2*step), abs=2e-8, rel=2e-7)

    gradient = (core.xc_lattice_gradient_contribution if spin == 1
                else core.xc_lattice_gradient_contribution_uks)
    analytic = np.asarray(gradient(*operands(system, basis, density)))[1, 2]
    energies = []
    for sign in (1., -1.):
        s, b = geometry(sign*step)
        energies.append(build(*operands(s, b, density)).e_xc)
    assert analytic == pytest.approx((energies[0]-energies[1])/(2*step), abs=2e-8, rel=2e-7)
