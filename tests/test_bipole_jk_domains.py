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


@pytest.mark.parametrize("selection", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("second_scale", [0.0, 0.23, 1e-7])
@pytest.mark.parametrize("omega", [0.0, 0.4])
@pytest.mark.parametrize("physical,threshold", [(True, 1e-6), (False, 1e-6), (True, 0.)])
def test_paired_density_domains_preserve_individual_screening(
    selection, sparse, second_scale, omega, physical, threshold
):
    from vibeqc._vibeqc_core import build_jk_2e_real_space_domains_pair

    system = vq.PeriodicSystem(1, np.diag([4., 100., 100.]),
        [vq.Atom(1, [.1, -.2, .3]), vq.Atom(1, [1.5, .1, -.1])])
    shells = [
        vq.ShellInfo(0, 0, True, [.2, 1.1], [.7, -.2], [.1, -.2, .3]),
        vq.ShellInfo(1, 1, True, [.3, 1.4], [.6, .3], [1.5, .1, -.1]),
    ]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "paired-density-test", False)
    cells = list(direct_lattice_cells(system, 4.0))[::-1]
    density_cells = list(direct_lattice_cells(system, 12.0))
    rng = np.random.default_rng(1982)
    first = make_lattice_matrix_set(4, density_cells,
        [rng.normal(size=(4, 4)) for _ in density_cells])
    second = make_lattice_matrix_set(4, density_cells,
        [second_scale * rng.normal(size=(4, 4)) for _ in density_cells])
    opts = _lat(4.0)
    opts.pair_complete_1e = physical
    opts.sr_range_screening = True
    opts.sr_sparse_traversal = sparse
    opts.schwarz_threshold = threshold
    kwargs = dict(output_indices=[0, len(cells) - 1],
        output_shell_masks=[[1, 1, 0, 1], [0, 1, 1, 1]], omega=omega,
        compute_coulomb=selection[0], compute_exchange=selection[1])
    references = [build_jk_2e_real_space_domains(basis, system, opts, density, cells, **kwargs)
                  for density in (first, second)]
    paired = build_jk_2e_real_space_domains_pair(basis, system, opts, first, second, cells, **kwargs)
    for actual, expected in zip(paired, references):
        assert _max_diff(actual.J, expected.J) == 0.0
        assert _max_diff(actual.K, expected.K) == 0.0
    assert paired[0].integral_evaluations == paired[1].integral_evaluations
    separate_count = sum(result.integral_evaluations for result in references)
    assert 0 < paired[0].integral_evaluations <= separate_count
    if second_scale == .23:
        assert paired[0].integral_evaluations < separate_count


@pytest.mark.parametrize("change", ["cell_order", "nonfinite", "dimension"])
def test_paired_density_domains_reject_incompatible_inputs(change):
    from vibeqc._vibeqc_core import build_jk_2e_real_space_domains_pair

    system = vq.PeriodicSystem(1, np.diag([4., 100., 100.]), [vq.Atom(2, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(),
        [vq.ShellInfo(0, 0, True, [1.], [1.], [0., 0., 0.])], "pair-input-test", False)
    cells = list(direct_lattice_cells(system, 4.0))
    first = make_lattice_matrix_set(1, cells, [np.ones((1, 1)) for _ in cells])
    nbf = 2 if change == "dimension" else 1
    blocks = [np.ones((nbf, nbf)) for _ in cells]
    if change == "nonfinite":
        blocks[0][0, 0] = np.nan
    second = make_lattice_matrix_set(nbf, cells[::-1] if change == "cell_order" else cells, blocks)
    with pytest.raises(ValueError, match="paired periodic J/K requires"):
        build_jk_2e_real_space_domains_pair(basis, system, _lat(4.), first, second, cells, omega=.4)


@pytest.mark.parametrize("reverse_cells", [False, True])
def test_coulomb_ket_permutations_preserve_asymmetric_density(reverse_cells):
    """Integral symmetry does not require a symmetric density or output mask."""
    system = vq.PeriodicSystem(1, np.diag([2., 100., 100.]),
        [vq.Atom(2, [.1, -.2, .3])])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(0, 0, True, [.3], [-.7], [.1, -.2, .3]),
        vq.ShellInfo(0, 1, True, [.2, 1.1], [.4, -.2], [.1, -.2, .3]),
    ], "ket-permutation-test", False)
    cells = list(direct_lattice_cells(system, 4.))
    if reverse_cells:
        cells.reverse()
    rng = np.random.default_rng(5104)
    density_cells = list(direct_lattice_cells(system, 8.))
    density = make_lattice_matrix_set(4, density_cells,
        [rng.normal(size=(4, 4)) for _ in density_cells])
    outputs = [i for i, c in enumerate(cells) if c.index[0] in (0, 1)]
    opts = _lat(4.)
    opts.schwarz_threshold = 1e-20
    opts.sr_range_screening = True
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=outputs,
            output_shell_masks=[[1, 1, 1, 1], [1, 1, 0, 1]], omega=.1))
    assert _max_diff(results[0].J, results[1].J) == 0.
    assert _max_diff(results[0].K, results[1].K) == 0.
    assert np.max(np.abs(np.asarray(results[1].J.blocks))) > 1e-5
    assert np.max(np.abs(np.asarray(results[1].K.blocks))) > 1e-5


@pytest.mark.parametrize("exponent", [0.02, 1.0, 50.0])
@pytest.mark.parametrize("omega", [0.0, 1e-6, 0.2, 2.0, 100.0])
def test_scalar_erfc_normalized_s_gaussian_known_limit(exponent, omega):
    """Ahlrichs 2006 Eq. (52), coincident normalized one-primitive s shells."""
    system = vq.PeriodicSystem(3, np.eye(3) * 20., [vq.Atom(2, [0., 0., 0.])])
    shell = vq.ShellInfo(0, 0, True, [exponent], [1.], [0., 0., 0.])
    basis = vq.BasisSet(system.unit_cell_molecule(), [shell], "scalar-erfc-test", False)
    cells = list(direct_lattice_cells(system, 0.1))
    density = make_lattice_matrix_set(1, cells, [np.ones((1, 1))])
    opts = _lat(0.1)
    opts.pair_complete_1e = True
    opts.schwarz_threshold = 1e-20
    result = build_jk_2e_real_space_domains(
        basis, system, opts, density, cells, omega=omega,
    )
    # Rationalized subtraction is stable when omega greatly exceeds sqrt(a).
    root = np.sqrt(omega * omega + exponent)
    expected = 2 * np.sqrt(exponent / np.pi) * exponent / (root * (root + omega))
    assert result.J.blocks[0][0, 0] == pytest.approx(expected, abs=3e-14, rel=0)
    assert result.K.blocks[0][0, 0] == pytest.approx(expected, abs=3e-14, rel=0)
    assert result.shell_pair_outputs == 1


@pytest.mark.parametrize("scale", [0., 1e-200, 1e-310, -1e-310])
@pytest.mark.parametrize("sparse", [False, True])
def test_density_log_screen_preserves_zero_and_subnormal_density(scale, sparse):
    from vibeqc._vibeqc_core import PeriodicJKWorkspace

    exponent, omega = .7, .2
    system = vq.PeriodicSystem(3, np.eye(3) * 20., [vq.Atom(2, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(),
        [vq.ShellInfo(0, 0, True, [exponent], [1.], [0., 0., 0.])], "density-log-test", False)
    cells = list(direct_lattice_cells(system, .1))
    opts = _lat(.1)
    opts.pair_complete_1e = opts.sr_range_screening = True
    opts.sr_sparse_traversal = sparse
    opts.schwarz_threshold = 1e-318
    workspace = PeriodicJKWorkspace()
    root = np.sqrt(omega*omega + exponent)
    integral = 2*np.sqrt(exponent/np.pi)*exponent / (root*(root + omega))
    # Exercise zero/subnormal envelopes and repeated density updates with the
    # same geometry workspace against the normalized Gaussian known limit.
    for value in [1., scale, 1.]:
        density = make_lattice_matrix_set(1, cells, [np.array([[value]])])
        result = build_jk_2e_real_space_domains(basis, system, opts, density, cells,
            omega=omega, workspace=workspace)
        for lattice in [result.J, result.K]:
            actual = lattice.blocks[0][0, 0]
            if value == 0.:
                assert actual == 0.
            else:
                assert actual/value == pytest.approx(integral, abs=2e-12, rel=0)
    assert workspace.preparation_count == 1


@pytest.mark.parametrize("omega", [0.05, 0.5, 3.0])
@pytest.mark.parametrize("threshold", [1e-12, 1e-4])
@pytest.mark.parametrize("exchange", [False, True])
def test_scalar_erfc_contracted_signed_shells_match_engine(omega, threshold, exchange):
    system = vq.PeriodicSystem(1, np.diag([4., 100., 100.]),
        [vq.Atom(1, [.17, -.23, .31]), vq.Atom(1, [1.4, .2, -.1])])
    shells = [
        vq.ShellInfo(0, 0, True, [0.06, 0.4, 2.], [.7, -.3, .1], [.17, -.23, .31]),
        vq.ShellInfo(1, 0, True, [0.2, 1.3], [-.4, .8], [1.4, .2, -.1]),
    ]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "scalar-contracted-test", False)
    cells = list(direct_lattice_cells(system, 8.0))[::-1]
    density_cells = list(direct_lattice_cells(system, 16.0))
    rng = np.random.default_rng(524)
    density = make_lattice_matrix_set(2, density_cells,
        [rng.normal(size=(2, 2)) for _ in density_cells])
    opts = _lat(8.0)
    opts.pair_complete_1e = opts.sr_range_screening = True
    opts.schwarz_threshold = threshold
    results = []
    for sparse in [False, True]:
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, omega=omega,
            output_indices=[0, len(cells) - 1],
            output_shell_masks=[[1, 0, 1, 1], [0, 1, 1, 1]],
            compute_exchange=exchange,
        ))
    assert results[1].shell_pair_outputs > 0
    assert _max_diff(results[0].J, results[1].J) == 0.0
    assert _max_diff(results[0].K, results[1].K) == 0.0


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("omega", [1e-6, .2, 3.])
@pytest.mark.parametrize("selection", [(True, True), (True, False), (False, True)])
def test_one_p_erfc_signed_contractions_match_general_engine(pure, omega, selection):
    system = vq.PeriodicSystem(1, np.diag([4., 100., 100.]),
        [vq.Atom(1, [.17, -.23, .31]), vq.Atom(1, [1.4, .2, -.1])])
    shells = [
        vq.ShellInfo(0, 0, pure, [.06, .4, 2.], [.7, -.3, .1], [.17, -.23, .31]),
        vq.ShellInfo(1, 1, pure, [.2, 1.3], [-.4, .8], [1.4, .2, -.1]),
    ]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "one-p-erfc-test", False)
    cells = list(direct_lattice_cells(system, 4.))[::-1]
    density_cells = list(direct_lattice_cells(system, 12.))
    rng = np.random.default_rng(3072)
    density = make_lattice_matrix_set(4, density_cells,
        [rng.normal(size=(4, 4)) for _ in density_cells])
    opts = _lat(4.)
    opts.pair_complete_1e = opts.sr_range_screening = True
    opts.schwarz_threshold = 1e-12
    results = []
    for sparse in [False, True]:
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(basis, system, opts, density, cells,
            omega=omega, compute_coulomb=selection[0], compute_exchange=selection[1]))
    assert results[1].shell_pair_outputs > 0
    # General Engine handles all quartets in the exhaustive oracle. Nonsymmetric
    # density probes every p-shell position in both J and AC/BD exchange order.
    assert _max_diff(results[0].J, results[1].J) < 2e-13
    assert _max_diff(results[0].K, results[1].K) < 2e-13


@pytest.mark.parametrize("omega", [.05, .5, 3.])
def test_one_p_erfc_matches_normalized_s_center_derivative(omega):
    # For normalized primitives, p_x = (1/sqrt(alpha)) d s / d A_x.
    # This independent finite difference tests the first angular recurrence,
    # Ahlrichs (2006), Eqs. (9)-(11), without a p-integral reference engine.
    origins = np.array([[.2, -.3, .1], [-.4, .1, .3], [.8, .6, -.2], [-.2, -.7, .4]])
    exponents = [.6, .8, .4, 1.2]
    system = vq.PeriodicSystem(3, np.eye(3) * 40., [vq.Atom(1, x) for x in origins])
    cells = list(direct_lattice_cells(system, .1))
    opts = _lat(10.)
    opts.pair_complete_1e = opts.sr_range_screening = True
    opts.schwarz_threshold = 1e-16
    def evaluate(angular, shift=0., axis=0):
        positions = origins.copy()
        positions[0, axis] += shift
        shifted_system = vq.PeriodicSystem(3, np.eye(3) * 40.,
            [vq.Atom(1, x) for x in positions])
        shells = [vq.ShellInfo(i, angular if i == 0 else 0, False,
            [exponents[i]], [1.], positions[i]) for i in range(4)]
        basis = vq.BasisSet(shifted_system.unit_cell_molecule(), shells, "one-p-derivative-test", False)
        nbf = int(basis.nbasis)
        block = np.zeros((nbf, nbf))
        block[-2, -1] = 1.
        density = make_lattice_matrix_set(nbf, cells, [block])
        result = build_jk_2e_real_space_domains(basis, shifted_system, opts, density, cells,
            omega=omega, compute_exchange=False)
        assert result.shell_pair_outputs == 1
        return result.J.blocks[0][:3, 3] if angular else result.J.blocks[0][0, 1]
    actual = evaluate(1)
    h = 2e-4
    for axis in range(3):
        derivative = (evaluate(0, -2*h, axis) - 8*evaluate(0, -h, axis)
            + 8*evaluate(0, h, axis) - evaluate(0, 2*h, axis)) / (12*h)
        assert actual[axis] == pytest.approx(derivative / np.sqrt(exponents[0]), abs=2e-11, rel=0)


@pytest.mark.parametrize("p_positions", [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
@pytest.mark.parametrize("pure_mask", range(4))
@pytest.mark.parametrize("omega", [.05, 3.])
def test_two_p_erfc_all_positions_match_general_engine(p_positions, pure_mask, omega):
    origins = [[.1, -.2, .3], [1.4, .2, -.1], [.7, .6, -.2], [-.2, -.7, .4]]
    system = vq.PeriodicSystem(1, np.diag([4., 100., 100.]),
        [vq.Atom(1, x) for x in origins])
    shells = [vq.ShellInfo(i, int(i in p_positions),
        bool(pure_mask & (1 << (i == p_positions[1]))),
        [.15 + .1*i, 1.2 + .2*i], [.6, -.3], x) for i, x in enumerate(origins)]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "two-p-erfc-test", False)
    cells = list(direct_lattice_cells(system, 4.))[::-1]
    density_cells = list(direct_lattice_cells(system, 12.))
    rng = np.random.default_rng(3077)
    density = make_lattice_matrix_set(8, density_cells,
        [rng.normal(size=(8, 8)) for _ in density_cells])
    opts = _lat(4.)
    opts.pair_complete_1e = opts.sr_range_screening = True
    opts.schwarz_threshold = 1e-12
    results = []
    for sparse in [False, True]:
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(basis, system, opts, density, cells,
            omega=omega, output_indices=[0, len(cells) - 1]))
    assert results[1].shell_pair_outputs == 2
    assert _max_diff(results[0].J, results[1].J) < 2e-13
    assert _max_diff(results[0].K, results[1].K) < 2e-13


@pytest.mark.parametrize("s_position", range(4))
@pytest.mark.parametrize("pure_mask", range(8))
@pytest.mark.parametrize("omega", [.05, 3.])
def test_three_p_erfc_all_positions_match_general_engine(s_position, pure_mask, omega):
    p_positions = [i for i in range(4) if i != s_position]
    origins = [[.1, -.2, .3], [1.4, .2, -.1], [.7, .6, -.2], [-.2, -.7, .4]]
    system = vq.PeriodicSystem(1, np.diag([4., 100., 100.]),
        [vq.Atom(1, x) for x in origins])
    shells = [vq.ShellInfo(i, int(i in p_positions),
        bool(pure_mask & (1 << p_positions.index(i))) if i in p_positions else False,
        [.15 + .1*i, 1.2 + .2*i], [.6, -.3], x) for i, x in enumerate(origins)]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "three-p-erfc-test", False)
    cells = list(direct_lattice_cells(system, 4.))[::-1]
    density_cells = list(direct_lattice_cells(system, 12.))
    rng = np.random.default_rng(3077)
    density = make_lattice_matrix_set(10, density_cells,
        [rng.normal(size=(10, 10)) for _ in density_cells])
    opts = _lat(4.)
    opts.pair_complete_1e = opts.sr_range_screening = True
    opts.schwarz_threshold = 1e-12
    results = []
    for sparse in [False, True]:
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(basis, system, opts, density, cells,
            omega=omega, output_indices=[0, len(cells) - 1]))
    assert results[1].shell_pair_outputs == 2
    assert _max_diff(results[0].J, results[1].J) < 2e-13
    assert _max_diff(results[0].K, results[1].K) < 2e-13


@pytest.mark.parametrize("p_positions", [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
@pytest.mark.parametrize("omega", [.05, 3.])
def test_two_p_erfc_matches_mixed_normalized_s_center_derivative(p_positions, omega):
    # Independent mixed derivative of an all-s integral: p_i p_j equals
    # d_i d_j (s s) / sqrt(alpha_i alpha_j), Ahlrichs (2006), Eqs. (9)-(11).
    origins = np.array([[.2, -.3, .1], [-.4, .1, .3], [.8, .6, -.2], [-.2, -.7, .4]])
    exponents = [.6, .8, .4, 1.2]
    opts = _lat(10.)
    opts.pair_complete_1e = opts.sr_range_screening = True
    opts.schwarz_threshold = 1e-16
    def evaluate(angular, axes, shifts=(0., 0.)):
        positions = origins.copy()
        for atom, axis, shift in zip(p_positions, axes, shifts):
            positions[atom, axis] += shift
        system = vq.PeriodicSystem(3, np.eye(3) * 40., [vq.Atom(1, x) for x in positions])
        shells = [vq.ShellInfo(i, int(angular and i in p_positions), False,
            [exponents[i]], [1.], positions[i]) for i in range(4)]
        basis = vq.BasisSet(system.unit_cell_molecule(), shells, "two-p-derivative-test", False)
        sizes = [3 if angular and i in p_positions else 1 for i in range(4)]
        offsets = np.cumsum([0] + sizes[:-1])
        indices = offsets.copy()
        if angular:
            for atom, axis in zip(p_positions, axes):
                indices[atom] += axis
        nbf = int(basis.nbasis)
        cells = list(direct_lattice_cells(system, .1))
        block = np.zeros((nbf, nbf))
        block[indices[2], indices[3]] = 1.
        density = make_lattice_matrix_set(nbf, cells, [block])
        result = build_jk_2e_real_space_domains(basis, system, opts, density, cells,
            omega=omega, compute_exchange=False)
        assert result.shell_pair_outputs == 1
        return result.J.blocks[0][indices[0], indices[1]]
    h = 2e-3
    stencil = [(-2, 1), (-1, -8), (1, 8), (2, -1)]
    scale = np.sqrt(exponents[p_positions[0]] * exponents[p_positions[1]])
    for i in range(3):
        for j in range(3):
            derivative = sum(ci*cj*evaluate(False, (i, j), (di*h, dj*h))
                for di, ci in stencil for dj, cj in stencil) / (144*h*h)
            assert evaluate(True, (i, j)) == pytest.approx(derivative / scale, abs=3e-10, rel=0)


@pytest.mark.parametrize("s_position", range(4))
@pytest.mark.parametrize("omega", [.05, 3.])
def test_three_p_erfc_matches_analytic_s_center_derivative(s_position, omega):
    """Three normalized p factors are mixed derivatives of analytic s ERIs.

    The reference uses the closed erfc Gaussian integral, Ahlrichs (2006),
    Eq. (52), rather than an angular integral engine. Its fourth-order
    finite-difference stencil has a looser tolerance than the engine check.
    """
    from itertools import product
    from math import erf, exp, fsum, pi, sqrt

    origins = np.array([[.2, -.3, .1], [-.4, .1, .3], [.8, .6, -.2], [-.2, -.7, .4]])
    exponents = np.array([.6, .8, .4, 1.2])
    p_positions = [i for i in range(4) if i != s_position]
    p, q = sum(exponents[:2]), sum(exponents[2:])
    rho = p*q/(p+q)
    ratio = omega/sqrt(omega*omega + rho)
    normalization = np.prod((2*exponents/pi)**.75)

    def fundamental(positions):
        a, b, c, d = positions
        ab = (exponents[0]*a + exponents[1]*b)/p
        cd = (exponents[2]*c + exponents[3]*d)/q
        t = rho * np.dot(ab-cd, ab-cd)
        def boys0(x):
            return .5*sqrt(pi/x)*erf(sqrt(x)) if x else 1.
        attenuation = exp(-exponents[0]*exponents[1]/p*np.dot(a-b, a-b)
                          -exponents[2]*exponents[3]/q*np.dot(c-d, c-d))
        return (normalization*2*pi**2.5/(p*q*sqrt(p+q))*attenuation
                * (boys0(t) - ratio*boys0(t*ratio*ratio)))

    system = vq.PeriodicSystem(3, np.eye(3)*40., [vq.Atom(1, x) for x in origins])
    shells = [vq.ShellInfo(i, int(i in p_positions), False,
        [exponents[i]], [1.], origins[i]) for i in range(4)]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "three-p-derivative", False)
    cells = list(direct_lattice_cells(system, .1))
    sizes = [3 if i in p_positions else 1 for i in range(4)]
    offsets = np.cumsum([0] + sizes[:-1])
    opts = _lat(10.)
    opts.pair_complete_1e = opts.sr_range_screening = True
    opts.schwarz_threshold = 1e-16
    h = .004
    stencil = [(-2, 1), (-1, -8), (1, 8), (2, -1)]
    scale = np.sqrt(np.prod(exponents[p_positions]))
    for axes in [(0, 0, 0), (0, 1, 2), (2, 1, 2)]:
        indices = offsets.copy()
        for atom, axis in zip(p_positions, axes):
            indices[atom] += axis
        block = np.zeros((10, 10))
        block[indices[2], indices[3]] = 1.
        density = make_lattice_matrix_set(10, cells, [block])
        result = build_jk_2e_real_space_domains(basis, system, opts, density, cells,
            omega=omega, compute_exchange=False)
        assert result.shell_pair_outputs == 1
        terms = []
        for shifts in product(stencil, repeat=3):
            positions = origins.copy()
            weight = 1
            for atom, axis, (step, coefficient) in zip(p_positions, axes, shifts):
                positions[atom, axis] += step*h
                weight *= coefficient
            terms.append(weight*fundamental(positions))
        reference = fsum(terms)/(12*h)**3/scale
        assert result.J.blocks[0][indices[0], indices[1]] == pytest.approx(
            reference, abs=2e-8, rel=0)


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
@pytest.mark.parametrize("selection", [(True, True), (True, False), (False, True)])
def test_erfc_shell_pair_join_is_bitwise_independent_of_workers(small_basis, selection):
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
        for workers in (1, 2, 4, 8):
            actual = set_num_threads(workers)
            if workers > 1 and actual < 2:
                pytest.skip("OpenMP workers are unavailable")
            results.append(build_jk_2e_real_space_domains(
                basis, system, opts, density, cells, omega=omega,
                compute_coulomb=selection[0], compute_exchange=selection[1],
            ))
    finally:
        set_num_threads(previous)
    for result in results[1:]:
        _assert_same_jk_work(results[0], result)
        assert results[0].integral_evaluations == result.integral_evaluations


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


@pytest.mark.parametrize("radius_scale", [1. - 1e-9, 1., 1. + 1e-9])
def test_sparse_erfc_sheared_sphere_boundaries_preserve_order(radius_scale):
    """Sparse contraction preserves the finite sum near a lattice sphere boundary."""
    lattice = np.array([[3., 1., 0.], [0., 4., 0.], [0., 0., 100.]])
    system = vq.PeriodicSystem(2, lattice, [vq.Atom(2, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(0, 0, True, [.11], [1.], [0., 0., 0.]),
        vq.ShellInfo(0, 1, True, [.16], [1.], [0., 0., 0.]),
    ], "sphere-boundary", False)
    rng = np.random.default_rng(281)
    indices = [(i, j, 0) for i in range(-2, 3) for j in range(-2, 3)]
    cells = [vq.LatticeCell(system, indices[i]) for i in rng.permutation(len(indices))]
    density_cells = [vq.LatticeCell(system, [i, j, 0])
                     for i in range(-4, 5) for j in range(-4, 5)]
    density = make_lattice_matrix_set(4, density_cells,
        [rng.normal(size=(4, 4)) for _ in density_cells])
    opts = _lat(np.sqrt(17.) * radius_scale)
    opts.pair_complete_1e = True
    opts.sr_range_screening = True
    outputs = [i for i, cell in enumerate(cells)
               if tuple(cell.index) in ((0, 0, 0), (0, 1, 0))][::-1]
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=outputs, omega=.4))
    assert results[1].shell_pair_outputs == len(outputs)
    assert _max_diff(results[0].J, results[1].J) == 0.
    assert _max_diff(results[0].K, results[1].K) == 0.
    assert np.max(np.abs(np.asarray(results[1].J.blocks))) > 1e-5
    assert np.max(np.abs(np.asarray(results[1].K.blocks))) > 1e-5


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_sparse_erfc_large_gapped_cartesian_domains(dimension):
    """Spatial searches preserve signed J/K sums on shuffled, gapped domains."""
    from itertools import product

    lattice = np.array([[3., .7, .2], [.3, 3.5, .4], [.2, .1, 4.]])
    system = vq.PeriodicSystem(dimension, lattice, [vq.Atom(2, [.2, -.1, .3])])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(0, 0, True, [.2, .8], [.7, -.15], [.2, -.1, .3]),
    ], "gapped-cartesian-domain", False)
    extent = {1: 90, 2: 7, 3: 3}[dimension]
    indices = [tuple(x) + (0,) * (3 - dimension)
               for x in product(range(-extent, extent + 1), repeat=dimension)]
    indices = [x for i, x in enumerate(indices) if i % 5 or not any(x)]
    rng = np.random.default_rng(4421)
    cells = [vq.LatticeCell(system, indices[i]) for i in rng.permutation(len(indices))]
    assert len(cells) >= 128
    density = make_lattice_matrix_set(1, cells,
        [rng.normal(size=(1, 1)) for _ in cells])
    outputs = [i for i, cell in enumerate(cells) if not np.any(cell.index)]
    opts = _lat(8.)
    opts.pair_complete_1e = True
    opts.sr_range_screening = True
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=outputs, omega=.5))
    assert results[1].shell_pair_outputs == 1
    assert _max_diff(results[0].J, results[1].J) == 0.
    assert _max_diff(results[0].K, results[1].K) == 0.
    assert np.max(np.abs(np.asarray(results[1].J.blocks))) > 1e-5
    assert np.max(np.abs(np.asarray(results[1].K.blocks))) > 1e-5


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


@pytest.mark.parametrize("threshold", [3e-11, 1e-8])
@pytest.mark.parametrize("scale", [1., -.4])
@pytest.mark.parametrize("length_scale,omega_scale", [
    (1., 1.), (.25, 1.), (4., 1.), (1., .2), (1., 3.),
])
def test_erfc_product_tail_matches_finite_normalized_gaussian_sum(
    threshold, scale, length_scale, omega_scale,
):
    """Independent closed-form J/K for translated, separated s products.

    For unit D_ab(0), J_ab = S_ab**2 sum_R U(|R|) and
    K_ab = U(|A-B|) sum_R exp(-alpha R**2). The normalized erfc
    Gaussian kernel U follows Ahlrichs (2006), Eq. (52),
    doi:10.1039/b605188j; erfc differences avoid far-tail cancellation.
    """
    import math

    # Rescale the normalized Gaussians and vary the kernel independently:
    # the reference remains analytic for diffuse, tight and long-range tails.
    alpha = .7 / length_scale**2
    omega = .7 * omega_scale / length_scale
    separation = 2. * length_scale
    cutoff = 32. * length_scale
    origins = [[-separation / 2, 0., 0.], [separation / 2, 0., 0.]]
    system = vq.PeriodicSystem(1, np.diag([4., 100., 100.]) * length_scale,
                              [vq.Atom(1, r) for r in origins])
    basis = vq.BasisSet(system.unit_cell_molecule(),
        [vq.ShellInfo(i, 0, True, [alpha], [1.], r) for i, r in enumerate(origins)],
        "gaussian-product-tail-test", False)
    cells = list(direct_lattice_cells(system, cutoff))
    home = next(i for i, c in enumerate(cells) if np.all(np.asarray(c.index) == 0))
    density = make_lattice_matrix_set(2, [cells[home]], [np.array([[0., scale], [0., 0.]])])
    theta = alpha * omega**2 / (alpha + omega**2)
    def kernel(distance):
        if distance == 0.:
            return 2. / math.sqrt(math.pi) * (math.sqrt(alpha) - math.sqrt(theta))
        return (math.erfc(math.sqrt(theta) * distance)
                - math.erfc(math.sqrt(alpha) * distance)) / distance
    distances = [float(np.linalg.norm(c.r_cart)) for c in cells]
    exact_j = scale * math.exp(-alpha * separation**2) * sum(kernel(r) for r in distances)
    exact_k = scale * kernel(separation) * sum(math.exp(-alpha * r*r) for r in distances)
    opts = _lat(cutoff)
    opts.schwarz_threshold = threshold
    for screening in (False, True):
        opts.sr_range_screening = screening
        result = build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=[home], omega=omega)
        tolerance = 2*threshold if screening else 3e-14
        assert result.J.blocks[home][0, 1] == pytest.approx(exact_j, abs=tolerance, rel=0)
        assert result.K.blocks[home][0, 1] == pytest.approx(exact_k, abs=tolerance, rel=0)


@pytest.mark.parametrize("count,spacing", [(n, 3.) for n in [1, 7, 8, 9, 16, 17, 32, 33, 65]]
                         + [(n, a) for n in [17, 33, 65] for a in [.999, 1., 1.001, 8.]])
@pytest.mark.parametrize("shuffle", [False, True])
def test_product_tree_leaf_boundaries_preserve_finite_sum(count, spacing, shuffle):
    """Product-tree leaves retain every surviving term and its sum order."""
    system = vq.PeriodicSystem(1, np.diag([spacing, 100., 100.]),
                              [vq.Atom(2, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(0, 0, True, [.2, 1.1], [.7, -.2], [0., 0., 0.]),
        vq.ShellInfo(0, 1, True, [.3], [.8], [0., 0., 0.]),
    ], "product-tree-test", False)
    cells = [vq.LatticeCell(system, [i, 0, 0])
             for i in range(-(count // 2), count - count // 2)]
    rng = np.random.default_rng(937)
    if shuffle:
        cells = [cells[i] for i in rng.permutation(count)]
    density_cells = list(direct_lattice_cells(system, spacing * count))
    density = make_lattice_matrix_set(4, density_cells,
        [.1 * rng.normal(size=(4, 4)) for _ in density_cells])
    outputs = [i for i, c in enumerate(cells) if int(c.index[0]) in (0, 1)]
    opts = _lat(spacing * count)
    opts.schwarz_threshold = 1e-9
    opts.sr_range_screening = True
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=outputs, omega=.5))
    assert results[1].product_screening_available
    assert results[1].shell_pair_outputs == len(outputs)
    assert _max_diff(results[0].J, results[1].J) == 0.
    assert _max_diff(results[0].K, results[1].K) == 0.
    assert np.max(np.abs(np.asarray(results[1].J.blocks))) > 1e-3


@pytest.mark.parametrize("cutoff", [4.3, 9.0, 20.0])
def test_product_tree_charge_spheres_preserve_skew_cell_contractions(cutoff):
    """Contained and boundary-crossing leaves retain the same finite sum."""
    from itertools import product

    system = vq.PeriodicSystem(3,
        np.array([[3.1, .7, -.4], [.2, 3.4, .6], [.3, -.5, 3.7]]),
        [vq.Atom(2, [.2, -.1, .3])])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(0, 0, True, [.12, .8], [.7, -.2], [.2, -.1, .3]),
        vq.ShellInfo(0, 1, True, [.2], [.8], [.2, -.1, .3]),
    ], "skew-product-tree", False)
    rng = np.random.default_rng(619)
    cells = [vq.LatticeCell(system, index)
             for index in product(range(-1, 2), repeat=3)]
    cells = [cells[i] for i in rng.permutation(len(cells))]
    density_cells = [vq.LatticeCell(system, index)
                     for index in product(range(-2, 3), repeat=3)]
    density = make_lattice_matrix_set(4, density_cells,
        [.1 * rng.normal(size=(4, 4)) for _ in density_cells])
    outputs = [i for i, cell in enumerate(cells)
               if tuple(cell.index) in ((0, 0, 0), (1, 1, 1), (-1, 0, 0))]
    opts = _lat(cutoff)
    opts.pair_complete_1e = True
    opts.eri_interaction_cutoff_bohr = 8.0
    opts.sr_range_screening = True
    opts.schwarz_threshold = 1e-10
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=outputs, omega=.5))
    assert results[1].shell_pair_outputs == len(outputs)
    assert _max_diff(results[0].J, results[1].J) == 0.
    assert _max_diff(results[0].K, results[1].K) == 0.
    assert np.max(np.abs(np.asarray(results[1].K.blocks))) > 1e-3


@pytest.mark.parametrize("unused_shells", [64, 81])
def test_product_tree_large_unused_basis_preserves_selected_block(unused_shells):
    """Unused AOs exhaust leaf-copy space without changing the selected sum.

    The 64-shell case fits the original index budget; the 81-shell case
    exercises the compact-node boundary. Both exceed the leaf-copy budget.
    Select the last shell to exercise large global leaf offsets and compare
    its block with the same finite sum in a one-shell basis.
    """
    from itertools import product

    system = vq.PeriodicSystem(3, np.eye(3) * 4., [vq.Atom(2, [0., 0., 0.])])
    cells = [vq.LatticeCell(system, list(index)) for index in product(range(-3, 4), repeat=3)]
    home = next(i for i, cell in enumerate(cells) if np.all(np.asarray(cell.index) == 0))
    shell = vq.ShellInfo(0, 0, True, [.2], [1.], [0., 0., 0.])
    opts = _lat(24.)
    opts.pair_complete_1e = False
    opts.sr_range_screening = True
    opts.schwarz_threshold = 1e-10
    values = []
    for count in (1, unused_shells):
        basis = vq.BasisSet(system.unit_cell_molecule(), [shell] * count, "leaf-layout-budget", False)
        block = np.zeros((count, count))
        selected = count - 1
        block[selected, selected] = .7
        density = make_lattice_matrix_set(count, [cells[home]], [block])
        mask = np.zeros(count * count, dtype=np.uint8)
        mask[selected * count + selected] = 1
        opts.sr_sparse_traversal = count > 1
        result = build_jk_2e_real_space_domains(basis, system, opts, density, cells,
            output_indices=[home], output_shell_masks=[mask.tolist()], omega=.4)
        assert result.product_screening_available
        if count > 1:
            assert result.shell_pair_outputs == 1
        values.append((result.J.blocks[home][selected, selected], result.K.blocks[home][selected, selected]))
    assert values[0] == values[1]
    assert min(values[0]) > 1e-3


@pytest.mark.parametrize("selection", [(True, True), (False, True)])
@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("second_scale", [0., .23])
def test_product_tree_preserves_masked_spin_domains(selection, physical, second_scale):
    """Two signed spin densities keep their individual physical quartet gates."""
    from vibeqc._vibeqc_core import build_jk_2e_real_space_domains_pair

    system = vq.PeriodicSystem(1, np.diag([1.3, 100., 100.]),
        [vq.Atom(1, [.1, -.2, .3]), vq.Atom(1, [.7, .1, -.1])])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(0, 0, True, [.2, 1.1], [.7, -.2], [.1, -.2, .3]),
        vq.ShellInfo(1, 1, True, [.3, 1.4], [.6, -.3], [.7, .1, -.1]),
    ], "product-box-spin-test", False)
    rng = np.random.default_rng(1982)
    cells = [vq.LatticeCell(system, [int(i), 0, 0]) for i in rng.permutation(np.arange(-16, 17))]
    density_cells = list(direct_lattice_cells(system, 43.))
    densities = [make_lattice_matrix_set(4, density_cells,
        [scale * rng.normal(size=(4, 4)) for _ in density_cells]) for scale in [1., second_scale]]
    outputs = [i for i, c in enumerate(cells) if int(c.index[0]) in (0, 3)]
    opts = _lat(5.2)
    opts.pair_complete_1e = physical
    opts.schwarz_threshold = 1e-9
    opts.sr_range_screening = True
    kwargs = dict(output_indices=outputs, output_shell_masks=[[1, 1, 0, 1], [0, 1, 1, 1]],
                  omega=.4, compute_coulomb=selection[0], compute_exchange=selection[1])
    opts.sr_sparse_traversal = False
    references = [build_jk_2e_real_space_domains(basis, system, opts, d, cells, **kwargs)
                  for d in densities]
    opts.sr_sparse_traversal = True
    results = build_jk_2e_real_space_domains_pair(basis, system, opts, *densities, cells, **kwargs)
    for actual, expected in zip(results, references):
        assert actual.shell_pair_outputs == len(outputs)
        assert _max_diff(actual.J, expected.J) == 0.
        assert _max_diff(actual.K, expected.K) == 0.


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
@pytest.mark.parametrize("compute_coulomb", [False, True])
def test_exchange_work_order_preserves_signed_periodic_density(threshold, density_scale, compute_coulomb):
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
            compute_coulomb=compute_coulomb,
        ))
    assert results[1].shell_pair_outputs == len(subset)
    assert _max_diff(results[0].J, results[1].J) == 0.0
    assert _max_diff(results[0].K, results[1].K) == 0.0
    if threshold < density_scale * 1e-2:
        assert np.max(np.abs(np.asarray(results[0].K.blocks))) > 0.0


@pytest.mark.parametrize("shuffle_density", [False, True])
@pytest.mark.parametrize("gapped_cells", [False, True])
def test_density_image_classes_keep_distant_nonperiodic_maxima(shuffle_density, gapped_cells):
    """Same-class images differ by many orders; maxima move between updates."""
    from vibeqc._vibeqc_core import PeriodicJKWorkspace

    system = vq.PeriodicSystem(1, np.diag([.7, 100., 100.]),
                              [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [.3, .2, .1])])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(0, 0, True, [.04, .7], [.8, -.2], [0., 0., 0.]),
        vq.ShellInfo(1, 1, True, [.06], [1.], [.3, .2, .1]),
    ], "nonperiodic-density-images", False)
    rng = np.random.default_rng(521)
    # Gaps and an asymmetric boundary make many translated kets absent.
    # A finite supplied domain must neither wrap nor fill those holes.
    cell_indices = ([-10, -9, -8, -6, -5, -3, -2, 0, 1, 3, 4, 6, 7, 9, 10, 12, 14]
                    if gapped_cells else np.arange(-8, 9))
    cells = [vq.LatticeCell(system, [int(i), 0, 0]) for i in rng.permutation(cell_indices)]
    density_indices = np.arange(-16, 17)
    if shuffle_density:
        density_indices = rng.permutation(density_indices)
    density_cells = [vq.LatticeCell(system, [int(i), 0, 0]) for i in density_indices]
    outputs = [i for i, c in enumerate(cells) if int(c.index[0]) in (-3, 1)]
    opts = _lat(20.)
    opts.pair_complete_1e = False
    opts.sr_range_screening = True
    opts.schwarz_threshold = 1e-9
    workspace = PeriodicJKWorkspace()
    for far in (12, 9):
        blocks = []
        for index in density_indices:
            scale = 500. if index == far else (.2 if index == far - 4 else 1e-9)
            blocks.append(scale * rng.normal(size=(4, 4)))
        density = make_lattice_matrix_set(4, density_cells, blocks)
        kwargs = dict(output_indices=outputs, omega=.4, compute_coulomb=False)
        opts.sr_sparse_traversal = False
        reference = build_jk_2e_real_space_domains(basis, system, opts, density, cells, **kwargs)
        opts.sr_sparse_traversal = True
        actual = build_jk_2e_real_space_domains(basis, system, opts, density, cells,
            workspace=workspace, **kwargs)
        assert actual.shell_pair_outputs == len(outputs)
        assert _max_diff(reference.K, actual.K) == 0.
        assert np.max(np.abs(np.asarray(actual.K.blocks))) > 1.
        assert _max_diff(reference.J, actual.J) == 0.
    assert workspace.preparation_count == 1


@pytest.mark.parametrize("compute_coulomb", [False, True])
@pytest.mark.parametrize("threshold", [1e-14, 1e-5])
def test_exchange_work_order_with_separated_shells(compute_coulomb, threshold):
    """Disconnected shell supports retain signed, nonuniform density terms."""
    positions = [[0., 0., 0.], [3., 0., 0.], [.2, 0., 0.], [6., 0., 0.]]
    system = vq.PeriodicSystem(1, np.diag([8., 100., 100.]),
                              [vq.Atom(1, x) for x in positions])
    shells = [vq.ShellInfo(i, int(i == 1), True, [.3, .9], [.8, -.2], x)
              for i, x in enumerate(positions)]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "separated-shells", False)
    rng = np.random.default_rng(813)
    cells = [vq.LatticeCell(system, [int(i), 0, 0])
             for i in rng.permutation(np.arange(-8, 9))]
    density_cells = [vq.LatticeCell(system, [int(i), 0, 0])
                     for i in range(-16, 17)]
    blocks = []
    for cell in density_cells:
        block = rng.normal(size=(basis.nbasis, basis.nbasis))
        # Nonuniform shell scales and a maximum outside the quartet-cell set.
        block[1:4, :] *= 1e-7
        block[:, 1:4] *= 1e-7
        if int(cell.index[0]) == 12:
            block[-1, -1] *= 1e3
        blocks.append(block)
    density = make_lattice_matrix_set(int(basis.nbasis), density_cells, blocks)
    opts = _lat(.4)
    opts.pair_complete_1e = True
    opts.schwarz_threshold = threshold
    opts.sr_range_screening = True
    outputs = [i for i, cell in enumerate(cells) if int(cell.index[0]) == 0]
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=outputs,
            compute_coulomb=compute_coulomb, omega=.5,
        ))
    assert results[1].shell_pair_outputs == 1
    assert _max_diff(results[0].J, results[1].J) == 0.
    assert _max_diff(results[0].K, results[1].K) == 0.
    assert np.max(np.abs(np.asarray(results[1].K.blocks))) > 1e-3


def test_exchange_work_order_across_large_cell_fields():
    """Many signed contributions span both bytes of packed cell indices."""
    system = vq.PeriodicSystem(1, np.diag([.15, 100., 100.]),
                              [vq.Atom(2, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(0, 0, True, [.04], [1.], [0., 0., 0.]),
    ], "wide-cell-order", False)
    rng = np.random.default_rng(917)
    cells = [vq.LatticeCell(system, [int(i), 0, 0])
             for i in rng.permutation(np.arange(-128, 129))]
    density_cells = [vq.LatticeCell(system, [int(i), 0, 0])
                     for i in range(-256, 257)]
    density = make_lattice_matrix_set(1, density_cells,
        [np.array([[float(rng.normal())]]) for _ in density_cells])
    outputs = [i for i, cell in enumerate(cells) if int(cell.index[0]) in (0, 1, -1)]
    opts = _lat(40.)
    opts.pair_complete_1e = False
    opts.schwarz_threshold = 1e-14
    opts.sr_range_screening = True
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=outputs, omega=.05,
        ))
    assert results[1].shell_pair_outputs == 3
    assert results[1].integral_evaluations > 2048
    assert _max_diff(results[0].J, results[1].J) == 0.
    assert _max_diff(results[0].K, results[1].K) == 0.
    assert np.max(np.abs(np.asarray(results[1].K.blocks))) > 1e-3


@pytest.mark.parametrize("shell_count", [1, 3, 256, 257])
def test_sparse_quartet_order_at_shell_capacity(shell_count):
    """The last shell survives packed ordering and the direct fallback.

    A one-cell finite Gaussian calculation probes sparse field boundaries
    without a large physical basis. Only corner output pairs and density
    elements contribute, but their indices include the final supplied shell.
    """
    system = vq.PeriodicSystem(1, np.diag([10., 100., 100.]),
                             [vq.Atom(2, [0., 0., 0.])])
    shells = [vq.ShellInfo(0, 0, True, [0.2 + i / 500.], [1.],
                          [0., 0., 0.]) for i in range(shell_count)]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "quartet-capacity", False)
    cells = list(direct_lattice_cells(system, 1.0))
    assert len(cells) == 1
    block = np.zeros((shell_count, shell_count))
    block[0, -1] = -0.3
    block[-1, 0] = 0.7
    density = make_lattice_matrix_set(shell_count, cells, [block])
    mask = np.zeros((shell_count, shell_count), dtype=np.uint8)
    mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = 1
    opts = _lat(1.0)
    opts.sr_range_screening = True
    results = []
    for sparse in (False, True):
        opts.sr_sparse_traversal = sparse
        results.append(build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, output_indices=[0],
            output_shell_masks=[mask.ravel().tolist()], omega=0.5,
        ))
    assert results[1].shell_pair_outputs == int(shell_count <= 256)
    assert _max_diff(results[0].J, results[1].J) == 0.0
    assert _max_diff(results[0].K, results[1].K) == 0.0
    assert abs(np.asarray(results[1].K.blocks)[0, -1, -1]) > 1e-3


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


@pytest.mark.parametrize("omega", [0.0, 0.5])
@pytest.mark.parametrize("screen,sparse,range_screen,physical", [
    (False, False, False, False),
    (True, False, True, False),
    (True, True, False, False),
    (True, True, True, False),
    (True, True, True, True),
])
def test_exchange_only_matches_combined_domains(omega, screen, sparse, range_screen, physical):
    from vibeqc._vibeqc_core import PeriodicJKWorkspace

    system, basis, opts, density, cells = _workspace_case()
    opts.schwarz_threshold = 1e-10 if screen else 0.0
    opts.sr_sparse_traversal = sparse
    opts.sr_range_screening = range_screen
    opts.pair_complete_1e = physical
    workspace = PeriodicJKWorkspace()
    kw = dict(omega=omega, output_indices=[len(cells) - 1, 0],
              output_shell_masks=[np.array([1, 0, 1, 1], dtype=np.uint8),
                                  np.ones(4, dtype=np.uint8)], workspace=workspace)
    combined = build_jk_2e_real_space_domains(basis, system, opts, density, cells, **kw)
    exchange = build_jk_2e_real_space_domains(
        basis, system, opts, density, cells, compute_coulomb=False, **kw)
    assert _max_diff(combined.K, exchange.K) == 0.0
    assert any(np.any(b) for b in combined.J.blocks)
    assert all(not np.any(b) for b in exchange.J.blocks)
    assert workspace.preparation_count == 1
    assert exchange.shell_quartets_considered <= combined.shell_quartets_considered
    neither = build_jk_2e_real_space_domains(
        basis, system, opts, density, cells,
        compute_coulomb=False, compute_exchange=False, **kw)
    assert all(not np.any(b) for b in list(neither.J.blocks) + list(neither.K.blocks))


@pytest.mark.parametrize("count", [16383, 16384, 16385, 20000])
@pytest.mark.parametrize("shuffle", [False, True])
def test_sparse_pair_work_preserves_indices_above_fourteen_bits(count, shuffle):
    """Distant, physically excluded images must not force exhaustive work.

    The three active images lie at large vector positions in the ordered
    case. The compact exhaustive reference has precisely the same admitted
    quartets and accumulation order, without the distant padding.
    """
    system = vq.PeriodicSystem(1, np.diag([4., 100., 100.]),
                              [vq.Atom(2, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(0, 0, True, [.7], [1.], [0., 0., 0.]),
    ], "wide-index-test", False)
    indices = list(range(1000, 1000 + count - 3)) + [-1, 0, 1]
    if shuffle:
        np.random.default_rng(31).shuffle(indices)
    cells = [vq.LatticeCell(system, [i, 0, 0]) for i in indices]
    positions = [i for i, index in enumerate(indices) if abs(index) <= 1]
    compact = [cells[i] for i in positions]
    density = make_lattice_matrix_set(1, compact,
        [np.array([[.7 if indices[i] == 0 else -.2]]) for i in positions])
    opts = _lat(6.)
    opts.pair_complete_1e = True
    opts.eri_interaction_cutoff_bohr = 20.
    opts.sr_range_screening = True
    opts.sr_sparse_traversal = False
    reference = build_jk_2e_real_space_domains(
        basis, system, opts, density, compact, omega=.5)
    opts.sr_sparse_traversal = True
    actual = build_jk_2e_real_space_domains(
        basis, system, opts, density, cells, output_indices=positions, omega=.5)
    assert actual.shell_pair_outputs == len(positions)
    for small, large in enumerate(positions):
        assert np.array_equal(reference.J.blocks[small], actual.J.blocks[large])
        assert np.array_equal(reference.K.blocks[small], actual.K.blocks[large])


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


@pytest.mark.parametrize("threshold", [1e-7, 1e-4])
@pytest.mark.parametrize("half_width", [4, 12])
def test_prepared_workspace_product_tree_keeps_kernel_schwarz_separate(threshold, half_width):
    """A nearly screened kernel must not suppress the next kernel's exchange."""
    from vibeqc._vibeqc_core import PeriodicJKWorkspace

    system, basis, opts, density, _ = _workspace_case()
    cells = [vq.LatticeCell(system, [int(i), 0, 0])
             for i in np.random.default_rng(128).permutation(np.arange(-half_width, half_width + 1))]
    density = make_lattice_matrix_set(int(basis.nbasis), list(density.cells),
        [1e-3 * np.asarray(block) for block in density.blocks])
    opts.schwarz_threshold = threshold
    opts.pair_complete_1e = False
    outputs = [i for i, cell in enumerate(cells) if int(cell.index[0]) in (0, 1)]
    workspace = PeriodicJKWorkspace()
    norms = []
    for omega in (100., .08, 100., .08):
        kwargs = dict(output_indices=outputs, omega=omega, compute_coulomb=False)
        opts.sr_sparse_traversal = False
        reference = build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, **kwargs)
        opts.sr_sparse_traversal = True
        actual = build_jk_2e_real_space_domains(
            basis, system, opts, density, cells, workspace=workspace, **kwargs)
        assert _max_diff(reference.K, actual.K) == 0.
        assert _max_diff(reference.J, actual.J) == 0.
        assert actual.product_screening_available
        norms.append(np.max(np.abs(np.asarray(actual.K.blocks))))
    assert norms[1] > 1e-5
    assert norms[1] > 100. * norms[0]
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
