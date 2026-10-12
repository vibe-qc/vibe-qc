"""M1 (pair-resolved truncation): ``build_jk_2e_real_space_domains``.

The impl behind every direct J/K entry point has always accepted an
explicit internal cell list + output subset + per-output shell-pair
masks; the wrappers hard-derive the cells radially from
``options.cutoff_bohr``. M1 exposes the impl as-is so Python controls
all three truncation domains (internal summation cells, output triple
set via subset+masks, density support via the density's own cell list)
-- the plumbing every later pair-resolved milestone (M2-M4b) builds on.

Contracts pinned here, all byte-identity (0.0 / 1e-15) against the
existing entry points on the same domains:

* cells = the radial cutoff list, no subset, no masks  ==
  ``build_jk_2e_real_space``.
* + ``output_indices``  ==  ``build_jk_2e_real_space_output_subset``.
* + ``output_shell_masks``  ==
  ``build_jk_2e_real_space_output_subset_masked``.
* ``compute_exchange=False``: J unchanged, K exactly zero.
* A PADDED internal cell list with the cutoff cells as
  ``output_indices`` reproduces the plain wide-ball build on the
  emitted blocks -- proving the caller-supplied list genuinely drives
  the internal (c_lam, c_sig) traversal (the M4a pad semantics,
  anchored transitively to the Poisson pins in
  test_bipole_sr_image_extent.py).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import (
    LatticeSumOptions,
    build_jk_2e_real_space,
    build_jk_2e_real_space_domains,
    build_jk_2e_real_space_output_subset,
    build_jk_2e_real_space_output_subset_masked,
    compute_overlap_lattice,
    direct_lattice_cells,
    make_lattice_matrix_set,
)
from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha

ANG2BOHR = 1.0 / 0.529177210903


@pytest.mark.parametrize("sparse", [False, True])
def test_erfc_empty_internal_domain_returns_empty_blocks(sparse):
    system, basis, omega, density, _ = _mgo_setup(8.0)
    opts = _lat(8.0)
    opts.sr_range_screening = True
    opts.sr_sparse_traversal = sparse
    result = build_jk_2e_real_space_domains(
        basis, system, opts, density, [], omega=omega,
    )
    assert len(result.J.blocks) == len(result.K.blocks) == 0
    assert result.cell_triples_considered == result.shell_quartets_considered == 0


@pytest.mark.parametrize("range_screening", [False, True])
@pytest.mark.parametrize("exchange", [False, True])
def test_sparse_erfc_matches_exhaustive_signed_masked_domains(range_screening, exchange):
    """Same matrix contributions and sum order, with p shells and off-cell density.

    A shuffled, non-spherical explicit domain and ragged output masks prevent
    origin-only or radial-list assumptions from making this comparison vacuous.
    """
    system, basis, omega, _, cells = _mgo_setup(8.0)
    cells = cells[::-1][::2]
    rng = np.random.default_rng(21)
    nbf = int(basis.nbasis)
    density_cells = list(direct_lattice_cells(system, 16.0))
    density = make_lattice_matrix_set(
        nbf, density_cells,
        [rng.normal(size=(nbf, nbf)) * 0.1 for _ in density_cells],
    )
    nsh = len(list(basis.shells()))
    subset = [0, len(cells) - 1]
    masks = [rng.integers(0, 2, nsh * nsh, dtype=np.uint8) for _ in subset]
    opts = _lat(8.0)
    opts.sr_range_screening = range_screening
    opts.schwarz_threshold = 1e-8
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, omega=omega,
            output_indices=subset, output_shell_masks=masks,
            compute_exchange=exchange,
        ))
    dense, sparse = results
    assert sparse.cell_triples_possible <= dense.cell_triples_possible
    assert sparse.cell_triples_considered <= dense.cell_triples_considered
    assert _max_diff(sparse.J, dense.J) == 0.0
    assert _max_diff(sparse.K, dense.K) == 0.0
    assert dense.cell_triples_considered == len(subset) * len(cells) ** 2


@pytest.mark.parametrize("threshold", [1e-12, 1e-7, 1e-3])
@pytest.mark.parametrize("scale", [0.1, -1e-5])
def test_erfc_shell_pair_join_keeps_density_and_screening_boundaries(threshold, scale):
    """Pair traversal preserves the full finite sum for nonsymmetric densities.

    The two-element basis includes diffuse and angular shells. Every output
    block is compared, including distant outputs dominated by exchange.
    """
    a = 4.084 * ANG2BOHR
    system = vq.PeriodicSystem(
        3, a / 2 * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]),
        [vq.Atom(3, [0.17, -0.23, 0.31]), vq.Atom(1, np.full(3, a / 2)
                                                        + [0.17, -0.23, 0.31])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-TZVP")
    cells = list(direct_lattice_cells(system, 8.0))[::-1]
    density_cells = list(direct_lattice_cells(system, 16.0))[::2]
    rng = np.random.default_rng(674)
    nbf = int(basis.nbasis)
    density = make_lattice_matrix_set(
        nbf, density_cells,
        [scale * rng.normal(size=(nbf, nbf)) for _ in density_cells],
    )
    opts = _lat(8.0)
    opts.sr_range_screening = True
    opts.schwarz_threshold = threshold
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, omega=0.5,
        ))
    assert _max_diff(results[0].J, results[1].J) == 0.0
    assert _max_diff(results[0].K, results[1].K) == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("omega", [0.3, 0.6])
def test_charge_pair_screen_against_schwarz_only(omega):
    """Charge-pair envelopes must retain numerical integral accuracy.

    The reference disables distance screening entirely. Diffuse primitive
    and contracted angular pairs see signed off-cell density contributions.
    """
    a = 4.084 * ANG2BOHR
    system = vq.PeriodicSystem(
        3, a / 2 * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]),
        [vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [a / 2] * 3)],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-TZVP")
    cells = list(direct_lattice_cells(system, 12.0))
    nbf = int(basis.nbasis)
    rng = np.random.default_rng(21)
    density = make_lattice_matrix_set(
        nbf, cells, [rng.normal(size=(nbf, nbf)) * 0.05 for _ in cells],
    )
    opts = _lat(12.0)
    opts.schwarz_threshold = 1e-12
    results = []
    for distance_screen in (False, True):
        opts.sr_range_screening = distance_screen
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, omega=omega,
        ))
    assert _max_diff(results[0].J, results[1].J) < 1e-10
    assert _max_diff(results[0].K, results[1].K) < 1e-10
    energies = [sum(
        0.5 * np.sum(np.asarray(d) * (np.asarray(j) - 0.5 * np.asarray(k)))
        for d, j, k in zip(density.blocks, result.J.blocks, result.K.blocks)
    ) for result in results]
    assert abs(energies[0] - energies[1]) < 1e-9


@pytest.mark.parametrize("small_basis", [False, True])
def test_erfc_shell_pair_join_is_bitwise_independent_of_workers(small_basis):
    """Both cell and shell-pair parallelism own complete matrix-element sums."""
    from vibeqc._vibeqc_core import get_num_threads, set_num_threads

    if small_basis:
        system = vq.PeriodicSystem(3, np.eye(3) * 4.0, [vq.Atom(2, [0, 0, 0])])
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        cells = list(direct_lattice_cells(system, 6.0))
        density = _overlap_density(basis, system, 12.0)
        omega = 0.5
    else:
        system, basis, omega, density, cells = _mgo_setup(6.0)
    opts = _lat(6.0)
    opts.sr_range_screening = True
    previous = get_num_threads()
    results = []
    try:
        for workers in (1, 4):
            actual = set_num_threads(workers)
            if workers > 1 and actual < 2:
                pytest.skip("OpenMP workers are unavailable")
            results.append(build_jk_2e_real_space_domains(
                basis, system, opts, density, cells, omega=omega,
            ))
    finally:
        set_num_threads(previous)
    assert _max_diff(results[0].J, results[1].J) == 0.0
    assert _max_diff(results[0].K, results[1].K) == 0.0
    assert results[0].cell_triples_considered == results[1].cell_triples_considered
    assert results[0].cell_triples_possible == results[1].cell_triples_possible
    assert results[0].shell_quartets_considered == results[1].shell_quartets_considered


def test_sparse_erfc_candidate_count_does_not_follow_image_ball_product():
    """Complexity gate independent of wall-clock load, using actual builds.

    The finite-range interaction stays local as padding grows; exhaustive
    enumeration grows as n_internal**2 even for a single output cell.
    """
    system = vq.PeriodicSystem(3, np.eye(3) * 3.0, [vq.Atom(2, [0, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    counts = []
    for cutoff in (18.0, 30.0):
        cells = list(direct_lattice_cells(system, cutoff))
        density_cells = list(direct_lattice_cells(system, 2.0 * cutoff))
        density = make_lattice_matrix_set(
            1, density_cells, [np.ones((1, 1)) for _ in density_cells],
        )
        opts = _lat(cutoff)
        opts.sr_range_screening = True
        results = []
        for sparse in (False, True):
            opts.sr_sparse_traversal = sparse
            results.append(build_jk_2e_real_space_domains(
                basis, system, opts, density, cells,
                omega=0.6, output_indices=[0],
            ))
        dense, sparse = results
        assert _max_diff(sparse.J, dense.J) == 0.0
        assert _max_diff(sparse.K, dense.K) == 0.0
        assert sparse.cell_triples_possible <= dense.cell_triples_possible
        assert dense.cell_triples_considered == len(cells) ** 2
        counts.append((dense.cell_triples_considered, sparse.cell_triples_considered))
    assert counts[1][0] > 20 * counts[0][0]
    assert counts[1][1] < 2 * counts[0][1]
    assert counts[1][1] < counts[1][0] // 20


@pytest.mark.parametrize("omega,threshold", [(0.0, 1e-12), (0.6, 0.0)])
def test_sparse_traversal_falls_back_when_screening_is_inapplicable(omega, threshold):
    system, basis, _, density, cells = _mgo_setup(6.0)
    opts = _lat(6.0)
    opts.schwarz_threshold = threshold
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, omega=omega,
            output_indices=[0],
        ))
    dense, sparse = results
    assert sparse.cell_triples_considered == dense.cell_triples_considered
    assert _max_diff(sparse.J, dense.J) == 0.0
    assert _max_diff(sparse.K, dense.K) == 0.0


@pytest.mark.parametrize("derivative", ["gamma", "density"])
def test_sparse_erfc_preserves_density_adjoints(derivative):
    """The finite-domain energy derivative must retain both contraction arms."""
    import vibeqc._vibeqc_core as core

    system, basis, omega, density, cells = _mgo_setup(6.0)
    opts = _lat(6.0)
    opts.sr_range_screening = True
    builder = getattr(core, f"build_jk_2e_real_space_domains_{derivative}_derivative")
    kwargs = {"omega": omega, "output_indices": [0, len(cells) - 1]}
    if derivative == "gamma":
        kwargs["gamma_density"] = np.eye(int(basis.nbasis))
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(builder(basis, system, opts, density, cells, **kwargs))
    dense, sparse = results
    assert _max_diff(sparse.J, dense.J) == 0.0
    assert _max_diff(sparse.K, dense.K) == 0.0
    if derivative == "gamma":
        np.testing.assert_allclose(sparse.J_gamma_energy_derivative,
                                   dense.J_gamma_energy_derivative, atol=1e-13, rtol=0)
        np.testing.assert_allclose(sparse.K_gamma_energy_derivative,
                                   dense.K_gamma_energy_derivative, atol=1e-13, rtol=0)
    else:
        assert _max_diff(sparse.J_density_energy_derivative,
                         dense.J_density_energy_derivative) < 1e-13
        assert _max_diff(sparse.K_density_energy_derivative,
                         dense.K_density_energy_derivative) < 1e-13


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("domain", ["regular", "duplicate", "other_lattice"])
def test_sparse_erfc_lower_dimensional_and_noncanonical_domains(dimension, domain):
    """Explicit domains retain their multiplicities and actual coordinates."""
    lattice = np.array([[4.0, 1.2, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 12.0]])
    atoms = [vq.Atom(2, [0.3, 0.2, 0.1])]
    system = vq.PeriodicSystem(dimension, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    if domain == "other_lattice":
        cell_system = vq.PeriodicSystem(dimension, lattice * 1.1, atoms)
    else:
        cell_system = system
    cells = list(direct_lattice_cells(cell_system, 7.0))
    if domain == "duplicate":
        cells.append(cells[0])
    density = _overlap_density(basis, system, 14.0)
    opts = _lat(7.0)
    opts.sr_range_screening = True
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=[0], omega=0.5,
        ))
    dense, sparse = results
    assert _max_diff(sparse.J, dense.J) == 0.0
    assert _max_diff(sparse.K, dense.K) == 0.0
    if domain != "regular":
        assert sparse.cell_triples_considered == dense.cell_triples_considered


@pytest.mark.slow
@pytest.mark.parametrize("internal_cutoff,repeat_density", [(20.0, False), (40.0, False), (40.0, True)])
def test_sparse_lih_pobtzvp_home_block(record_property, internal_cutoff, repeat_density):
    """#21 carrier: 13 BFs, including padded cold-build image domains.

    D(h)=S(h) or a repeated home-cell block are deterministic kernel fixtures,
    not converged SCF densities. The repeated block guards against assuming
    decay between periodic density images when selecting a representation.
    Full-node SCF timings must be measured separately on the supported input.
    """
    import time

    # Same geometry as test_bipole_exact_zone._lih_primitive.
    a = 4.084 * ANG2BOHR
    system = vq.PeriodicSystem(
        3, a / 2 * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]),
        [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [a / 2] * 3)],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-TZVP")
    assert basis.nbasis == 13
    cells = list(direct_lattice_cells(system, internal_cutoff))
    density = _overlap_density(basis, system, 2 * internal_cutoff)
    if repeat_density:
        density = make_lattice_matrix_set(
            int(basis.nbasis), density.cells,
            [0.1 * np.asarray(density.blocks[0]) for _ in density.cells],
        )
    record_property("n_internal", len(cells))
    record_property("repeat_density", repeat_density)
    omega = crystal_default_ewald_alpha(abs(np.linalg.det(np.asarray(system.lattice))))
    opts = _lat(20.0)
    opts.sr_range_screening = True
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        start = time.perf_counter()
        result = build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=[0], omega=omega,
        )
        label = "sparse" if sparse else "exhaustive"
        record_property(f"{label}_wall_s", time.perf_counter() - start)
        record_property(f"{label}_cell_triples", result.cell_triples_considered)
        record_property(f"{label}_shell_quartets", result.shell_quartets_considered)
        results.append(result)
    dense, sparse = results
    assert _max_diff(sparse.J, dense.J) == 0.0
    assert _max_diff(sparse.K, dense.K) == 0.0
    assert sparse.cell_triples_possible <= dense.cell_triples_possible
    assert sparse.cell_triples_considered < dense.cell_triples_considered
    # Pin a fivefold reduction in quartet expansion independently of timing.
    assert sparse.shell_quartets_considered < dense.shell_quartets_considered // 5


@pytest.mark.parametrize("angular_momentum", [1, 2, 3])
@pytest.mark.parametrize("pure", [False, True], ids=["cartesian", "spherical"])
@pytest.mark.parametrize("long_contraction", [False, True], ids=["primitive", "signed-long"])
def test_charge_screen_retains_angular_pair_tails(angular_momentum, pure, long_contraction):
    """#755: positive angular charge pairs at 16 bohr defeat the QQR estimate."""
    system = vq.PeriodicSystem(
        3, np.eye(3) * 16.0, [vq.Atom(2, [0., 0., 0.])],
    )
    exponents, coefficients = [0.12], [1.0]
    if long_contraction:
        # Exercise the bounded long-contraction envelope, including signs.
        exponents = [0.12, 0.18, 0.3, 0.5, 0.9, 1.5, 3., 6., 12.]
        coefficients = [1., -0.2, 0.15, -0.1, 0.08, -0.06, 0.04, -0.02, 0.01]
    shell = vq.ShellInfo(
        0, angular_momentum, pure, exponents, coefficients, [0., 0., 0.],
    )
    basis = vq.BasisSet(
        system.unit_cell_molecule(), [shell], "angular-erfc-test", False,
    )
    cells = list(direct_lattice_cells(system, 17.0))
    nbf = int(basis.nbasis)
    # Positive diagonal density isolates (aa|bb) charge couplings; discarded
    # image terms cannot cancel one another in the home-block diagonal J.
    density = make_lattice_matrix_set(nbf, [cells[0]], [np.eye(nbf)])
    opts = _lat(17.0)
    results = []
    for screening in (False, True):
        opts.sr_range_screening = screening
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=[0], omega=0.6,
        ))
    assert _max_diff(results[0].J, results[1].J) < 1e-10
    assert _max_diff(results[0].K, results[1].K) < 1e-10


def test_sparse_erfc_contracted_df_shells_preserve_primitive_order():
    """Cached shell pairs retain primitive order through angular permutations."""
    system = vq.PeriodicSystem(
        3, np.eye(3) * 20.0,
        [vq.Atom(6, [0.17, -0.23, 0.31]), vq.Atom(2, [1.4, 0.5, -0.2])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "cc-pvtz")
    cells = list(direct_lattice_cells(system, 1.0))
    nbf = int(basis.nbasis)
    density = make_lattice_matrix_set(
        nbf, cells, [np.random.default_rng(21).normal(size=(nbf, nbf))],
    )
    opts = _lat(1.0)
    opts.sr_range_screening = True
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, omega=0.5,
        ))
    assert _max_diff(results[0].J, results[1].J) == 0.0
    assert _max_diff(results[0].K, results[1].K) == 0.0


@pytest.mark.parametrize("sparse", [False, True])
def test_erfc_underflowing_product_tails_preserve_nearby_integrals(sparse):
    """Tiny far products share tree nodes with nonzero near-cell products.

    Tight contracted s/p envelopes underflow beyond about 40 bohr. Keep
    those supplied cells, signed nonlocal density and a nonradial ordering,
    and compare the screened result with the ordinary Schwarz-only build.
    """
    system = vq.PeriodicSystem(
        1, np.diag([4., 100., 100.]), [vq.Atom(2, [0., 0., 0.])],
    )
    shells = [
        vq.ShellInfo(0, 0, True, [1., 2.], [0.6, -0.2], [0., 0., 0.]),
        vq.ShellInfo(0, 1, True, [1.2, 2.4], [0.8, 0.1], [0., 0., 0.]),
    ]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "erfc-tail-test", False)
    cells = list(direct_lattice_cells(system, 64.0))[::-1]
    rng = np.random.default_rng(617)
    density_cells = list(direct_lattice_cells(system, 128.0))
    density = make_lattice_matrix_set(
        int(basis.nbasis), density_cells,
        [rng.normal(size=(basis.nbasis, basis.nbasis)) for _ in density_cells],
    )
    subset = [i for i, c in enumerate(cells) if int(c.index[0]) in (0, 1, 16)]
    opts = _lat(64.0)
    opts.schwarz_threshold = 1e-12
    results = []
    for range_screening in (False, True):
        opts.sr_range_screening = range_screening
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=subset, omega=0.5,
        ))
    assert _max_diff(results[0].J, results[1].J) < 1e-11
    assert _max_diff(results[0].K, results[1].K) < 1e-11
    assert np.max(np.abs(np.asarray(results[0].J.blocks))) > 1e-3


@pytest.mark.parametrize("compute_exchange", [False, True])
@pytest.mark.parametrize("only_far", [False, True])
def test_sparse_erfc_screened_outputs_preserve_mask_positions(compute_exchange, only_far):
    """Whole-output rejection retains zero blocks and the caller's mask order."""
    system = vq.PeriodicSystem(
        1, np.diag([4., 100., 100.]), [vq.Atom(2, [0., 0., 0.])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    cells = list(direct_lattice_cells(system, 64.0))[::-1]
    positions = {int(c.index[0]): i for i, c in enumerate(cells)}
    requested = [16, -16] if only_far else [16, 0, 1]
    subset = [positions[i] for i in requested]
    masks = [np.array([i != 0], dtype=np.uint8) for i in requested]
    density_cells = list(direct_lattice_cells(system, 128.0))
    density = make_lattice_matrix_set(
        1, density_cells,
        [np.array([[0.1 * (-1.) ** int(c.index[0])]]) for c in density_cells],
    )
    opts = _lat(64.0)
    opts.schwarz_threshold = 1e-12
    results = []
    for screening in (False, True):
        opts.sr_range_screening = screening
        opts.sr_sparse_traversal = screening
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=subset,
            output_shell_masks=masks, omega=0.5, compute_exchange=compute_exchange,
        ))
    assert _max_diff(results[0].J, results[1].J) < 1e-11
    assert _max_diff(results[0].K, results[1].K) < 1e-11
    assert results[1].shell_pair_outputs == len(subset)
    assert np.count_nonzero(results[1].J.blocks[positions[16]]) == 0
    if only_far:
        assert np.count_nonzero(np.asarray(results[1].J.blocks)) == 0
        assert np.count_nonzero(np.asarray(results[1].K.blocks)) == 0
    else:
        assert np.count_nonzero(results[1].J.blocks[positions[0]]) == 0
        assert np.max(np.abs(results[1].J.blocks[positions[1]])) > 1e-6


def test_sparse_erfc_shell_density_bounds_include_off_home_blocks():
    """Only off-diagonal shell pairs in nonzero density images contribute."""
    system = vq.PeriodicSystem(
        1, np.diag([4., 100., 100.]),
        [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [1.4, 0., 0.])],
    )
    shells = [vq.ShellInfo(i, 0, True, [0.3], [1.], origin)
              for i, origin in enumerate(([0., 0., 0.], [1.4, 0., 0.]))]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "erfc-density-test", False)
    cells = list(direct_lattice_cells(system, 32.0))
    density_cells = list(direct_lattice_cells(system, 64.0))
    blocks = []
    for cell in density_cells:
        block = np.zeros((2, 2))
        if int(cell.index[0]) == 1:
            block[0, 1] = 0.7
        elif int(cell.index[0]) == -1:
            block[1, 0] = 0.7
        blocks.append(block)
    density = make_lattice_matrix_set(2, density_cells, blocks)
    opts = _lat(32.0)
    opts.sr_range_screening = True
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=[0], omega=0.5,
        ))
    assert _max_diff(results[0].J, results[1].J) == 0.0
    assert _max_diff(results[0].K, results[1].K) == 0.0
    assert np.max(np.abs(np.asarray(results[0].K.blocks))) > 1e-4


@pytest.mark.parametrize("threshold", [1e-12, 1e-8, 1e-4])
@pytest.mark.parametrize("density_scale", [1.0, 1e-6])
def test_exchange_work_order_preserves_signed_periodic_density(threshold, density_scale):
    """Sparse work ordering preserves every signed density-image contribution.

    More than 16 cells force internal tree nodes. Angular/diffuse pairs have
    different Schwarz ranges; shuffled cells and unequal signed shell blocks
    expose an incorrect pair-major index or changed contraction order.
    """
    system = vq.PeriodicSystem(
        1, np.diag([4., 100., 100.]), [vq.Atom(2, [0., 0., 0.])],
    )
    shells = [
        vq.ShellInfo(0, 0, True, [0.15, 0.7], [0.8, -0.2], [0., 0., 0.]),
        vq.ShellInfo(0, 1, True, [0.3, 1.2], [0.6, 0.4], [0., 0., 0.]),
    ]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "erfc-tree-test", False)
    cells = list(direct_lattice_cells(system, 64.0))
    rng = np.random.default_rng(719)
    cells = [cells[i] for i in rng.permutation(len(cells))]
    assert len(cells) > 16
    density_cells = list(direct_lattice_cells(system, 128.0))
    blocks = [density_scale * rng.normal(size=(basis.nbasis, basis.nbasis))
              for _ in density_cells]
    # The largest p-p contraction occurs away from the reference cell.
    for cell, block in zip(density_cells, blocks):
        if int(cell.index[0]) == -3:
            block[1:, 1:] *= 7.0
    density = make_lattice_matrix_set(int(basis.nbasis), density_cells, blocks)
    subset = [i for i, cell in enumerate(cells) if int(cell.index[0]) in (0, 1, 4)]
    opts = _lat(64.0)
    opts.schwarz_threshold = threshold
    opts.sr_range_screening = True
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=subset, omega=0.5,
        ))
    assert results[1].shell_pair_outputs == len(subset)
    assert _max_diff(results[0].J, results[1].J) == 0.0
    assert _max_diff(results[0].K, results[1].K) == 0.0
    if threshold < density_scale * 1e-2:
        assert np.max(np.abs(np.asarray(results[0].K.blocks))) > 0.0


def test_sparse_erfc_lazy_pair_gates_preserve_dense_rows():
    """A dense 65-cell domain exercises allocation and multiword gate rows."""
    system = vq.PeriodicSystem(
        1, np.diag([0.25, 100., 100.]), [vq.Atom(2, [0., 0., 0.])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(0, 0, True, [0.02], [1.], [0., 0., 0.]),
    ], "erfc-order-test", False)
    cells = list(direct_lattice_cells(system, 8.0))[::-1]
    assert len(cells) == 65
    density_cells = list(direct_lattice_cells(system, 16.0))
    rng = np.random.default_rng(432)
    density = make_lattice_matrix_set(
        1, density_cells, [rng.normal(size=(1, 1)) for _ in density_cells],
    )
    opts = _lat(8.0)
    opts.sr_range_screening = True
    opts.schwarz_threshold = 1e-30
    subset = [0, len(cells) - 1]
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=subset, omega=0.01,
        ))
    assert results[1].shell_pair_outputs == len(subset)
    # The finite-domain cell gate still applies, with many admitted pairs.
    assert results[1].shell_quartets_considered > 1024 * len(subset)
    assert _max_diff(results[0].J, results[1].J) == 0.0
    assert _max_diff(results[0].K, results[1].K) == 0.0


def test_sparse_erfc_dense_support_keeps_complete_pair_union():
    """Dense support must retain all pairs when the bounded list fills."""
    system = vq.PeriodicSystem(3, np.eye(3) * 1.2, [vq.Atom(2, [0, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    cells = list(direct_lattice_cells(system, 6.0))
    assert len(cells) ** 2 > 262144
    density_cells = list(direct_lattice_cells(system, 12.0))
    density = make_lattice_matrix_set(
        1, density_cells, [np.ones((1, 1)) for _ in density_cells],
    )
    opts = _lat(6.0)
    opts.schwarz_threshold = 1e-40
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=[0], omega=0.5,
        ))
    assert results[1].cell_triples_considered == len(cells) ** 2
    assert _max_diff(results[0].J, results[1].J) == 0.0
    assert _max_diff(results[0].K, results[1].K) == 0.0


def _workspace_case(exponent=0.3, coefficient=0.8, origin=0.0, pure=True):
    system = vq.PeriodicSystem(1, np.diag([4., 100., 100.]),
                              [vq.Atom(2, [origin, 0., 0.])])
    shells = [
        vq.ShellInfo(0, 0, True, [exponent, 1.2], [coefficient, -0.2], [origin, 0., 0.]),
        vq.ShellInfo(0, 1, pure, [0.4], [1.], [origin, 0., 0.]),
    ]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "workspace-test", False)
    cells = list(direct_lattice_cells(system, 12.0))
    density_cells = list(direct_lattice_cells(system, 24.0))
    rng = np.random.default_rng(937)
    density = make_lattice_matrix_set(int(basis.nbasis), density_cells,
        [rng.normal(size=(basis.nbasis, basis.nbasis)) for _ in density_cells])
    opts = _lat(12.0)
    opts.sr_range_screening = True
    return system, basis, opts, density, cells


def _assert_same_jk_work(left, right):
    assert _max_diff(left.J, right.J) == 0.0
    assert _max_diff(left.K, right.K) == 0.0
    for name in ("cell_triples_considered", "cell_triples_possible",
                 "shell_quartets_considered", "shell_pair_outputs", "cell_sparse_outputs",
                 "exhaustive_outputs", "charge_screening_available", "product_screening_available"):
        assert getattr(left, name) == getattr(right, name), name


def test_prepared_workspace_reuses_only_geometry():
    from vibeqc._vibeqc_core import PeriodicJKWorkspace

    system, basis, opts, density, cells = _workspace_case()
    workspace = PeriodicJKWorkspace()
    for step in range(3):
        # Change density values and support, output order/masks, exchange and
        # positive screening threshold. None is cached in geometric preparation.
        positions = [0, len(cells) - 1] if step == 0 else [len(cells) - 1, 0]
        blocks = [np.asarray(b) * (-0.7)**step for b in density.blocks]
        d = make_lattice_matrix_set(int(basis.nbasis), list(density.cells)[step:], blocks[step:])
        opts.schwarz_threshold = 10.0 ** (-12 + step * 3)
        masks = [np.array([1, step != 1, 1, 1], dtype=np.uint8) for _ in positions]
        kw = dict(omega=0.5, output_indices=positions, output_shell_masks=masks,
                  compute_exchange=step != 2)
        plain = build_jk_2e_real_space_domains(basis, system, opts, d, cells, **kw)
        cached = build_jk_2e_real_space_domains(basis, system, opts, d, cells,
                                              workspace=workspace, **kw)
        _assert_same_jk_work(plain, cached)
        assert workspace.preparation_count == 1
        assert workspace.geometry_preparation_count == 1


@pytest.mark.parametrize("change", ["exponent", "coefficient", "origin", "pure",
    "order", "subset", "cell_coordinates", "lattice", "omega", "coulomb", "screen", "range"])
def test_prepared_workspace_invalidates_changed_geometry_or_kernel(change):
    from vibeqc._vibeqc_core import PeriodicJKWorkspace

    system, basis, opts, density, cells = _workspace_case()
    workspace = PeriodicJKWorkspace()
    build_jk_2e_real_space_domains(basis, system, opts, density, cells,
                                  output_indices=[0], omega=0.5, workspace=workspace)
    omega = 0.5
    if change in ("exponent", "coefficient", "origin", "pure"):
        values = dict(exponent=0.35, coefficient=0.75, origin=0.1, pure=False)
        system, basis, opts, density, cells = _workspace_case(**{change: values[change]})
    elif change == "order":
        cells = cells[::-1]
    elif change == "subset":
        cells = cells[:-1]
    elif change in ("cell_coordinates", "lattice"):
        other = vq.PeriodicSystem(1, np.diag([4.1, 100., 100.]), [vq.Atom(2, [0., 0., 0.])])
        if change == "lattice":
            system = other
        else:
            cells = [vq.LatticeCell(other, c.index) for c in cells]
    elif change == "omega":
        omega = 0.6
    elif change == "coulomb":
        omega = 0.0
    elif change == "screen":
        opts.schwarz_threshold = 0.0
    elif change == "range":
        opts.sr_range_screening = False
    kw = dict(output_indices=[0], omega=omega)
    cached = build_jk_2e_real_space_domains(basis, system, opts, density, cells,
                                          workspace=workspace, **kw)
    plain = build_jk_2e_real_space_domains(basis, system, opts, density, cells, **kw)
    _assert_same_jk_work(plain, cached)
    assert workspace.preparation_count == 2
    assert workspace.geometry_preparation_count == (
        1 if change in ("omega", "coulomb", "screen", "range") else 2)


def test_prepared_workspace_concurrent_generations_preserve_snapshots():
    from concurrent.futures import ThreadPoolExecutor
    from vibeqc._vibeqc_core import PeriodicJKWorkspace

    system, basis, opts, density, cells = _workspace_case()
    workspace = PeriodicJKWorkspace()
    omegas = [0.4, 0.6, 0.4, 0.6]
    def build(omega, workspace=None):
        return build_jk_2e_real_space_domains(basis, system, opts, density, cells,
            output_indices=[0], omega=omega, workspace=workspace)
    expected = [build(w) for w in omegas]
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(lambda w: build(w, workspace), omegas))
    for plain, cached in zip(expected, actual):
        _assert_same_jk_work(plain, cached)
    assert workspace.preparation_count == 2
    assert workspace.geometry_preparation_count == 1


def test_prepared_workspace_two_kernels_have_bounded_lru_lifetime():
    from vibeqc._vibeqc_core import PeriodicJKWorkspace

    system, basis, opts, density, cells = _workspace_case()
    workspace = PeriodicJKWorkspace()
    # Revisit the newest two kernels, evict the oldest with a third, then
    # change geometry and verify both old kernel states are invalidated.
    cases = [(0.4, 1), (0.6, 2), (0.4, 2), (0.8, 3), (0.4, 3), (0.6, 4)]
    for omega, count in cases:
        kw = dict(output_indices=[0], omega=omega)
        plain = build_jk_2e_real_space_domains(basis, system, opts, density, cells, **kw)
        cached = build_jk_2e_real_space_domains(basis, system, opts, density, cells,
                                              workspace=workspace, **kw)
        _assert_same_jk_work(plain, cached)
        assert workspace.preparation_count == count
        assert workspace.geometry_preparation_count == 1
    cells = cells[::-1]
    for omega, count in [(0.4, 5), (0.6, 6), (0.4, 6)]:
        kw = dict(output_indices=[0], omega=omega)
        plain = build_jk_2e_real_space_domains(basis, system, opts, density, cells, **kw)
        cached = build_jk_2e_real_space_domains(basis, system, opts, density, cells,
                                              workspace=workspace, **kw)
        _assert_same_jk_work(plain, cached)
        assert workspace.preparation_count == count
        assert workspace.geometry_preparation_count == 2


def _mgo():
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2, a / 2, a / 2])]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _lat(cutoff: float) -> LatticeSumOptions:
    opts = LatticeSumOptions()
    opts.cutoff_bohr = float(cutoff)
    opts.nuclear_cutoff_bohr = float(cutoff)
    return opts


def _overlap_density(basis, system, cutoff: float):
    """Deterministic symmetric density with content on EVERY cell:
    D(h) = S(h). Exercises the cross-cell P(h) lookups the home-only
    density of test_bipole_sr_image_extent.py never touches."""
    opts = _lat(cutoff)
    S_lat = compute_overlap_lattice(basis, system, opts)
    return make_lattice_matrix_set(
        int(basis.nbasis),
        list(S_lat.cells),
        [np.asarray(b, dtype=float) for b in S_lat.blocks],
    )


def _mgo_setup(cutoff: float = 6.0):
    system, basis = _mgo()
    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    omega = float(crystal_default_ewald_alpha(V_cell))
    density = _overlap_density(basis, system, cutoff)
    cells = list(direct_lattice_cells(system, cutoff))
    return system, basis, omega, density, cells


def _max_diff(lat_a, lat_b) -> float:
    assert len(lat_a.blocks) == len(lat_b.blocks)
    worst = 0.0
    for i in range(len(lat_a.blocks)):
        worst = max(
            worst,
            float(
                np.max(
                    np.abs(
                        np.asarray(lat_a.blocks[i], dtype=float)
                        - np.asarray(lat_b.blocks[i], dtype=float)
                    )
                )
            ),
        )
    return worst


def test_domains_full_equals_plain_build():
    system, basis, omega, density, cells = _mgo_setup()
    plain = build_jk_2e_real_space(basis, system, _lat(6.0), density, omega)
    dom = build_jk_2e_real_space_domains(
        basis, system, _lat(6.0), density, cells, omega=omega
    )
    assert _max_diff(dom.J, plain.J) == 0.0
    assert _max_diff(dom.K, plain.K) == 0.0
    # Sanity: the comparison is not vacuous.
    assert float(np.max(np.abs(np.asarray(plain.J.blocks[0])))) > 1e-2


def test_domains_output_subset_equals_subset_binding():
    system, basis, omega, density, cells = _mgo_setup()
    subset = [0, 2, len(cells) - 1]
    ref = build_jk_2e_real_space_output_subset(
        basis, system, _lat(6.0), density, subset, omega
    )
    dom = build_jk_2e_real_space_domains(
        basis, system, _lat(6.0), density, cells,
        output_indices=subset, omega=omega,
    )
    assert _max_diff(dom.J, ref.J) == 0.0
    assert _max_diff(dom.K, ref.K) == 0.0
    # Non-emitted blocks stay zero.
    emitted = set(subset)
    for i in range(len(cells)):
        if i not in emitted:
            assert float(np.max(np.abs(np.asarray(dom.J.blocks[i])))) == 0.0


def test_domains_masked_equals_masked_binding():
    system, basis, omega, density, cells = _mgo_setup()
    n_shells = len(list(basis.shells()))
    subset = [0, 1]
    masks = []
    for k, _ in enumerate(subset):
        mask = np.zeros(n_shells * n_shells, dtype=np.uint8)
        # A deliberately ragged selection, different per output cell.
        for s1 in range(n_shells):
            for s2 in range(n_shells):
                if (s1 + s2 + k) % 2 == 0:
                    mask[s1 * n_shells + s2] = 1
        masks.append(mask)
    ref = build_jk_2e_real_space_output_subset_masked(
        basis, system, _lat(6.0), density, subset, masks, omega
    )
    dom = build_jk_2e_real_space_domains(
        basis, system, _lat(6.0), density, cells,
        output_indices=subset, output_shell_masks=masks, omega=omega,
    )
    assert _max_diff(dom.J, ref.J) == 0.0
    assert _max_diff(dom.K, ref.K) == 0.0


def test_domains_j_only_skips_exchange():
    system, basis, omega, density, cells = _mgo_setup()
    full = build_jk_2e_real_space_domains(
        basis, system, _lat(6.0), density, cells, omega=omega
    )
    j_only = build_jk_2e_real_space_domains(
        basis, system, _lat(6.0), density, cells,
        omega=omega, compute_exchange=False,
    )
    assert _max_diff(j_only.J, full.J) == 0.0
    for i in range(len(cells)):
        assert float(np.max(np.abs(np.asarray(j_only.K.blocks[i])))) == 0.0
    assert float(np.max(np.abs(np.asarray(full.K.blocks[0])))) > 1e-3


def test_sr_range_screening_accuracy_and_noop_cases():
    """Charge-pair separation-aware SR screening
    (LatticeSumOptions.sr_range_screening):

    * default OFF (ordinary Schwarz screening);
    * ON vs OFF at the production ball agrees to the screening noise
      scale (skips only -- measured ~1e-12 at the 1e-12 threshold);
    * omega = 0 (full Coulomb): the flag is a strict no-op (the bound
      is erfc-kernel-specific), bitwise.
    """
    system, basis, omega, density, cells = _mgo_setup()
    assert LatticeSumOptions().sr_range_screening is False

    def _lat_srr(cutoff, srr):
        opts = _lat(cutoff)
        opts.sr_range_screening = srr
        return opts

    off = build_jk_2e_real_space(basis, system, _lat_srr(6.0, False), density, omega)
    on = build_jk_2e_real_space(basis, system, _lat_srr(6.0, True), density, omega)
    assert _max_diff(on.J, off.J) < 1e-9
    assert _max_diff(on.K, off.K) < 1e-9

    off0 = build_jk_2e_real_space(basis, system, _lat_srr(6.0, False), density, 0.0)
    on0 = build_jk_2e_real_space(basis, system, _lat_srr(6.0, True), density, 0.0)
    assert _max_diff(on0.J, off0.J) == 0.0
    assert _max_diff(on0.K, off0.K) == 0.0


def test_sr_range_screening_padded_ball_agrees():
    """The screening's purpose: a padded internal ball (the M4a/M4b
    ket-image pad) with the flag ON reproduces the unscreened padded
    build to the screening noise scale. Performance is measured separately
    on the production carrier; this test checks the numerical result."""
    system, basis, omega, density, cells = _mgo_setup(cutoff=6.0)
    cells_pad = list(direct_lattice_cells(system, 14.0))
    key = lambda c: tuple(int(x) for x in np.asarray(c.index).reshape(3))
    pos = {key(c): i for i, c in enumerate(cells_pad)}
    out_idx = [pos[key(c)] for c in cells]

    def _lat_srr(srr):
        opts = _lat(14.0)
        opts.sr_range_screening = srr
        return opts

    off = build_jk_2e_real_space_domains(
        basis, system, _lat_srr(False), density, cells_pad,
        output_indices=out_idx, omega=omega,
    )
    on = build_jk_2e_real_space_domains(
        basis, system, _lat_srr(True), density, cells_pad,
        output_indices=out_idx, omega=omega,
    )
    worst = 0.0
    for p in out_idx:
        worst = max(
            worst,
            float(np.max(np.abs(np.asarray(on.J.blocks[p]) - np.asarray(off.J.blocks[p])))),
            float(np.max(np.abs(np.asarray(on.K.blocks[p]) - np.asarray(off.K.blocks[p])))),
        )
    assert worst < 1e-8


def test_padded_internal_cells_reproduce_wide_build():
    """Caller-supplied cells genuinely drive the internal traversal.

    Internal list at 12 bohr with the 6-bohr cells as the output subset
    == the plain 12-bohr build on those same cells (identical internal
    quartet set per emitted block). This is the M4a pad expressed
    through the M1 binding.
    """
    system, basis, omega, density, cells = _mgo_setup(cutoff=6.0)
    cells_pad = list(direct_lattice_cells(system, 12.0))
    key = lambda c: tuple(int(x) for x in np.asarray(c.index).reshape(3))
    pad_pos = {key(c): i for i, c in enumerate(cells_pad)}
    output_indices = [pad_pos[key(c)] for c in cells]

    dom = build_jk_2e_real_space_domains(
        basis, system, _lat(12.0), density, cells_pad,
        output_indices=output_indices, omega=omega,
    )
    wide = build_jk_2e_real_space(basis, system, _lat(12.0), density, omega)
    worst = 0.0
    for oi, pos in enumerate(output_indices):
        worst = max(
            worst,
            float(
                np.max(
                    np.abs(
                        np.asarray(dom.J.blocks[pos], dtype=float)
                        - np.asarray(wide.J.blocks[pos], dtype=float)
                    )
                )
            ),
        )
    assert worst == 0.0
    # And the pad matters: the padded home J differs from the unpadded
    # 6-bohr build's (the -515 mHa SR-truncation mechanism).
    unpadded = build_jk_2e_real_space(basis, system, _lat(6.0), density, omega)
    d_home = float(
        np.max(
            np.abs(
                np.asarray(dom.J.blocks[output_indices[0]])
                - np.asarray(unpadded.J.blocks[0])
            )
        )
    )
    assert d_home > 1e-4


@pytest.mark.parametrize("sparse,range_screening,expected", [
    (False, False, "exhaustive_outputs"),
    (False, True, "exhaustive_outputs"),
    (True, False, "cell_sparse_outputs"),
    (True, True, "shell_pair_outputs"),
])
def test_erfc_execution_metadata_reports_selected_traversal(sparse, range_screening, expected):
    system, basis, omega, density, cells = _mgo_setup(8.0)
    opts = _lat(8.0)
    opts.sr_sparse_traversal = sparse
    opts.sr_range_screening = range_screening
    result = build_jk_2e_real_space_domains(
        basis, system, opts, density, cells, output_indices=[0], omega=omega,
    )
    assert getattr(result, expected) == 1
    assert result.shell_pair_outputs + result.cell_sparse_outputs + result.exhaustive_outputs == 1
    assert result.charge_screening_available is range_screening
    assert result.product_screening_available is range_screening
    assert result.shell_pair_fallback_outputs == result.cell_sparse_fallback_outputs == 0


def test_erfc_duplicate_domain_reports_exhaustive_execution():
    system, basis, omega, density, cells = _mgo_setup(8.0)
    opts = _lat(8.0)
    opts.sr_sparse_traversal = opts.sr_range_screening = True
    result = build_jk_2e_real_space_domains(
        basis, system, opts, density, cells+[cells[0]], output_indices=[0], omega=omega,
    )
    assert result.exhaustive_outputs == 1
    assert result.shell_pair_outputs == result.cell_sparse_outputs == 0
    assert result.charge_screening_available
