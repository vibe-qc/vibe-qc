"""#429/#724: physical lattice support across integral and SCF consumers.

One-electron and reciprocal product-pair support can be narrower than
exchange output support. Consumers align these lists by integer cell key.
Independent Gaussian sums, image relabelling and finite differences check
the physical nuclear/quartet predicates; coupled split tests separately
check that the real-space and reciprocal terms describe the same operator.

The switch stays off by default. Native regression anchors pin each finite
operator convention; the independent periodic references below test its
convergence separately.
"""

from __future__ import annotations

from itertools import product

from scipy.special import erf

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    compute_multipole_moments_lattice,
    compute_nuclear_erfc_lattice,
    compute_nuclear_lattice,
    compute_nuclear_lattice_with_charges,
    compute_overlap_lattice,
    direct_lattice_cells,
    make_lattice_matrix_set,
    nuclear_erfc_lattice_gradient_contribution,
    nuclear_lattice_gradient_contribution,
    pair_complete_lattice_cells,
)

# H2 in a 6-bohr cubic box, STO-3G, 2x2x2 mesh, DIRECT_TRUNCATED. At an
# 8-bohr cutoff the ball has 7 cells and the pair-complete list 19 (the
# intra-cell offset is 1.4 bohr), so every index-for-index consumer sees
# two lists of different length.
_CUTOFF = 8.0
_E_RHF_OFF = -34.655444157536  # switch off: the historical sum, bit-identical
_E_RHF_ON = -33.87027955873168  # physical AO products and finite source images
# Reviewed after the periodic point-grid and Schwarz self-norm corrections.
# Restoring only the historical grid reproduces the old LDA anchors within
# 1.4e-10 Ha; the wider HF anchor changes only with the Schwarz correction.
# See docs/bipole_erfc_resume.md for the isolated-build receipts.
_E_RKS_OFF = -34.66112304044817
_E_RKS_ON = -33.875342453487875
_E_RHF_RADIAL_15 = -22.939983004503866
_E_RHF_PAIR_15 = -23.19210137102612
_PIN_TOL = 1.0e-7


def _h2_box(a: float = 6.0):
    system = vq.PeriodicSystem(
        3,
        a * np.eye(3),
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])],
        0,
        1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _lat(on: bool, cutoff: float = _CUTOFF):
    opts = vq.LatticeSumOptions()
    opts.pair_complete_1e = bool(on)
    opts.cutoff_bohr = float(cutoff)
    return opts


def _run_cxx(on: bool, *, ks: bool = False, guess=None, cutoff: float = _CUTOFF, relabel=0):
    system, basis = _h2_box()
    if relabel:
        system = vq.PeriodicSystem(3, system.lattice, [
            vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [1.4+6*relabel, 0., 0.]),
        ])
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(system, [2, 2, 2])
    if ks:
        opts = vq.PeriodicKSOptions()
        opts.functional = "lda"
    else:
        opts = vq.PeriodicSCFOptions()
    opts.conv_tol_energy = 1.0e-10
    opts.lattice_opts.pair_complete_1e = bool(on)
    opts.lattice_opts.cutoff_bohr = float(cutoff)
    if guess is not None:
        opts.initial_guess = guess
    driver = vq.run_rks_periodic if ks else vq.run_rhf_periodic
    result = driver(system, basis, kmesh, opts)
    assert result.converged
    return float(result.energy)


def test_lists_differ_in_length_on_this_fixture():
    system, basis = _h2_box()
    n_ball = len(list(direct_lattice_cells(system, _CUTOFF)))
    n_pair = len(list(pair_complete_lattice_cells(basis, system, _CUTOFF)))
    assert (n_ball, n_pair) == (7, 19)


@pytest.mark.parametrize("ks", [False, True], ids=["rhf", "rks-lda"])
def test_cxx_multi_k_fock_assembly_keeps_the_physical_operator_support(ks):
    """Pin each finite-domain convention after independent integral/FD checks."""
    e_off = _run_cxx(False, ks=ks)
    e_on = _run_cxx(True, ks=ks)
    assert e_off == pytest.approx(_E_RKS_OFF if ks else _E_RHF_OFF, rel=0, abs=_PIN_TOL)
    assert e_on == pytest.approx(_E_RKS_ON if ks else _E_RHF_ON, rel=0, abs=_PIN_TOL)
    assert abs(e_on - e_off) > 1.0e-5


@pytest.mark.parametrize("cutoff", [8.0, 15.0])
def test_cxx_multi_k_physical_operator_preserves_cell_relabelling(cutoff):
    reference = _run_cxx(True, cutoff=cutoff)
    relabelled = _run_cxx(True, cutoff=cutoff, relabel=1)
    assert abs(reference-relabelled) < 2e-10
    if cutoff == 15.0:
        # Equal one-electron cell counts do not make the finite nuclear
        # and three-image ERI term sets equal to the historical balls.
        assert reference == pytest.approx(_E_RHF_PAIR_15, rel=0, abs=_PIN_TOL)
        assert _run_cxx(False, cutoff=cutoff) == pytest.approx(_E_RHF_RADIAL_15, rel=0, abs=_PIN_TOL)



@pytest.mark.parametrize(
    "guess",
    [vq.InitialGuess.PATOM, vq.InitialGuess.MINAO, vq.InitialGuess.HUECKEL],
    ids=["patom", "minao", "hueckel"],
)
def test_cxx_guesses_reach_the_same_fixed_point_under_the_switch(guess):
    """PATOM widens its in-field Fock onto Hcore's list; MINAO's cross
    overlap enumerates pair-complete; HUECKEL rides S(g) as is."""
    e_sad = _run_cxx(True, guess=vq.InitialGuess.SAD)
    e_guess = _run_cxx(True, guess=guess)
    assert abs(e_guess - e_sad) < 1.0e-8


def test_cxx_rks_patom_reaches_the_same_fixed_point_under_the_switch():
    e_sad = _run_cxx(True, ks=True, guess=vq.InitialGuess.SAD)
    e_patom = _run_cxx(True, ks=True, guess=vq.InitialGuess.PATOM)
    assert abs(e_patom - e_sad) < 1.0e-8


# ---------------------------------------------------------------------------
# Both halves of the Ewald nuclear split use the same physical pair support.
# ---------------------------------------------------------------------------


def test_ewald_split_nuclear_family_uses_pair_support():
    from vibeqc.lattice_screening import ao_pair_support_mask
    from vibeqc.periodic_v_ne import compute_v_ne_ewald_3d_ft_lattice

    system, basis = _h2_box()
    physical = _lat(True)
    radial = _lat(False, _CUTOFF + 1.4)
    for opts in (physical, radial):
        opts.nuclear_cutoff_bohr = 16.0
    eo = vq.EwaldOptions()
    eo.alpha = 0.6
    for build in (
        lambda opts: compute_nuclear_erfc_lattice(basis, system, eo.alpha, opts),
        lambda opts: compute_v_ne_ewald_3d_ft_lattice(
            basis, system, opts, ewald_options=eo, ke_cutoff=12.0, screen_rel=0.0,
        ),
    ):
        result, reference = build(physical), build(radial)
        assert len(result.cells) == len(reference.cells) == 19
        kept_tail = False
        for cell, got, full in zip(result.cells, result.blocks, reference.blocks):
            keep = ao_pair_support_mask(basis, cell.r_cart, _CUTOFF)
            expected = np.where(keep, np.asarray(full), 0.0)
            np.testing.assert_allclose(got, expected, rtol=0.0, atol=2e-13)
            if np.linalg.norm(cell.r_cart) > _CUTOFF:
                kept_tail |= bool(np.any(np.abs(expected) > 1e-8))
        assert kept_tail, "fixture must exercise a pair outside the historical ball"


@pytest.mark.parametrize("sap", [False, True])
def test_grid_ewald_nuclear_uses_same_pair_support(sap):
    from vibeqc._vibeqc_core import compute_nuclear_lattice_ewald, compute_vsap_lattice
    from vibeqc.lattice_screening import ao_pair_support_mask
    from vibeqc._vibeqc_core import build_grid

    system, basis = _h2_box()
    physical, radial = _lat(True), _lat(False, _CUTOFF + 1.4)
    grid = build_grid(system.unit_cell_molecule())
    eo = vq.EwaldOptions()
    eo.alpha = 0.6
    results = []
    for opts in (physical, radial):
        # Converge both source domains here to isolate the AO-support mask.
        opts.nuclear_cutoff_bohr = 40.0 if sap else 16.0
        if sap:
            value = compute_vsap_lattice(
                basis, system, grid, "sap_helfem_large", opts, eo,
            )
        else:
            value = compute_nuclear_lattice_ewald(basis, system, grid, opts, eo)
        results.append(value)
    assert len(results[0].cells) == len(results[1].cells) == 19
    for cell, got, full in zip(results[0].cells, results[0].blocks, results[1].blocks):
        keep = ao_pair_support_mask(basis, cell.r_cart, _CUTOFF)
        np.testing.assert_allclose(got, np.where(keep, full, 0.0), rtol=0.0, atol=2e-13)


@pytest.mark.parametrize("alpha", [0.4, 0.7])
def test_pair_complete_ewald_nuclear_gradient_matches_fd(alpha):
    from vibeqc.periodic_v_ne import compute_v_ne_ewald_3d_ft_gamma
    from vibeqc.periodic_v_ne_gradient import compute_v_ne_ewald_3d_ft_gamma_gradient

    system, basis = _h2_box()
    opts = _lat(True)
    opts.nuclear_cutoff_bohr = 20.0
    eo = vq.EwaldOptions()
    eo.alpha = alpha
    density = np.array([[0.7, -0.2], [-0.2, 0.4]])
    analytic = compute_v_ne_ewald_3d_ft_gamma_gradient(
        basis, system, opts, density, ewald_options=eo, ke_cutoff=16.0,
    )
    step = 2e-4
    energies = []
    for delta in (-step, step):
        displaced = vq.PeriodicSystem(
            3, 6.0 * np.eye(3),
            [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4 + delta, 0.0, 0.0])],
            0, 1,
        )
        shifted_basis = vq.BasisSet(displaced.unit_cell_molecule(), "sto-3g")
        value = compute_v_ne_ewald_3d_ft_gamma(
            shifted_basis, displaced, opts, ewald_options=eo, ke_cutoff=16.0,
        )
        energies.append(float(np.sum(density * value)))
    assert analytic[1, 0] == pytest.approx((energies[1] - energies[0]) / (2 * step), abs=2e-7)
    np.testing.assert_allclose(analytic.sum(axis=0), 0.0, atol=2e-8)


def test_pair_complete_nuclear_split_is_invariant_to_atom_relabelling():
    """The same crystal must retain its Gamma matrix after one image shift."""
    from vibeqc.periodic_v_ne import compute_v_ne_ewald_3d_ft_gamma

    eo = vq.EwaldOptions()
    eo.alpha = 0.6
    values = {False: [], True: []}
    for displacement in (0.0, 6.0):
        system = vq.PeriodicSystem(
            3, 6.0 * np.eye(3),
            [vq.Atom(1, [0.0, 0.0, 0.0]),
             vq.Atom(1, [1.4 + displacement, 0.0, 0.0])],
            0, 1,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        for on in values:
            opts = _lat(on)
            opts.nuclear_cutoff_bohr = 24.0
            values[on].append(compute_v_ne_ewald_3d_ft_gamma(
                basis, system, opts, ewald_options=eo, ke_cutoff=16.0,
            ))
    np.testing.assert_allclose(*values[True], rtol=0.0, atol=1e-10)
    assert np.max(np.abs(values[False][1] - values[False][0])) > 1e-5


@pytest.mark.parametrize("mismatch", ["overlap", "cache"])
def test_bipole_nuclear_split_rejects_reordered_support(mismatch):
    from types import SimpleNamespace

    from vibeqc.pbc_bipole_common import _compute_nuclear_lattice_ewald_reciprocal_ft

    system, basis = _h2_box()
    opts = _lat(True)
    overlap = compute_overlap_lattice(basis, system, opts)
    eo = vq.EwaldOptions()
    eo.alpha = 0.6
    cache = None
    if mismatch == "overlap":
        overlap = make_lattice_matrix_set(
            basis.nbasis, list(overlap.cells)[::-1], list(overlap.blocks)[::-1],
        )
        reason = "cell ordering differs"
    else:
        cache = SimpleNamespace(cells_r_cart=np.array([
            cell.r_cart for cell in reversed(overlap.cells)
        ]))
        reason = "cache cell list differs"
    with pytest.raises(RuntimeError, match=reason):
        _compute_nuclear_lattice_ewald_reciprocal_ft(
            basis, system, opts, eo, overlap, cache=cache,
        )


@pytest.mark.parametrize("box", [12.0, 30.0])
def test_bipole_3d_accepts_coupled_pair_complete_nuclear_support(box):
    system, basis = _h2_box(box)
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.pair_complete_1e = True
    opts.lattice_opts.cutoff_bohr = box + 2.0
    opts.conv_tol_energy = 1e-9
    result = vq.run_pbc_bipole_rhf(system, basis, kmesh, opts, progress=False)
    assert result.converged
    assert np.isfinite(result.energy)


# ---------------------------------------------------------------------------
# Python consumers: list-agnostic on a pair-complete template.
# ---------------------------------------------------------------------------


def test_copy_lattice_with_blocks_zero_fills_the_pair_complete_tail():
    from vibeqc.pbc_bipole_common import _copy_lattice_with_blocks

    system, basis = _h2_box()
    ball = list(direct_lattice_cells(system, _CUTOFF))
    nbf = basis.nbasis
    blocks = [np.full((nbf, nbf), float(i + 1)) for i in range(len(ball))]
    with pytest.raises(ValueError, match="missing block"):
        _copy_lattice_with_blocks(basis, system, _lat(True), ball, blocks)
    out = _copy_lattice_with_blocks(
        basis, system, _lat(True), ball, blocks, fill_missing=True
    )
    assert len(out.cells) == 19
    for i in range(len(ball)):
        assert np.array_equal(np.asarray(out.blocks[i]), blocks[i])
    for i in range(len(ball), len(out.cells)):
        assert not np.asarray(out.blocks[i]).any()


def test_screened_exchange_blocks_align_to_the_density_list():
    from vibeqc.periodic_screened_exchange import (
        PeriodicExchangeAssembly,
        build_exchange_blocks,
    )

    system, basis = _h2_box()
    exx = PeriodicExchangeAssembly(c_full=1.0, c_sr=0.0, omega_screen=0.0)
    from vibeqc.pbc_bipole_fock import _sr_density_cells

    for on in (False, True):
        D = compute_overlap_lattice(basis, system, _lat(on))  # stand-in density
        if on:
            with pytest.raises(RuntimeError, match="cells the density list does not carry"):
                build_exchange_blocks(basis, system, _lat(on), D, exx)
            by_cell = {tuple(c.index): np.asarray(b) for c, b in zip(D.cells, D.blocks)}
            cells = _sr_density_cells(basis, system, _lat(on))
            D = make_lattice_matrix_set(basis.nbasis, cells, [
                by_cell.get(tuple(c.index), np.zeros((basis.nbasis, basis.nbasis))) for c in cells
            ])
        K = build_exchange_blocks(basis, system, _lat(on), D, exx)
        assert len(K) == len(D.cells)
        assert np.abs(K[0]).max() > 0.0
        native = vq.build_jk_2e_real_space(basis, system, _lat(on), D)
        expected = {tuple(c.index): np.asarray(b) for c, b in zip(native.K.cells, native.K.blocks)}
        for cell, block in zip(D.cells, K):
            assert np.array_equal(block, expected[tuple(cell.index)])
        if on:
            from vibeqc.lattice_screening import ao_pair_support_mask

            assert any(np.asarray(block).any() for block in K[7:])
            assert any(np.any(np.asarray(block)[~ao_pair_support_mask(
                basis, cell.r_cart, _CUTOFF)]) for cell, block in zip(D.cells, K))
        reversed_density = make_lattice_matrix_set(
            basis.nbasis, list(D.cells)[::-1], list(D.blocks)[::-1],
        )
        reversed_blocks = build_exchange_blocks(basis, system, _lat(on), reversed_density, exx)
        for actual, original in zip(reversed_blocks, K[::-1]):
            assert np.array_equal(actual, original)


def test_cell_multipole_moments_read_the_prefix_of_a_longer_density():
    from vibeqc.bipole_cell_moments import compute_cell_multipole_moments

    system, basis = _h2_box()
    M = compute_multipole_moments_lattice(basis, system, _lat(True), 2, (0.0, 0.0, 0.0))
    P = compute_overlap_lattice(basis, system, _lat(True))  # stand-in density
    n_ball = len(M.cells)
    assert (n_ball, len(P.cells)) == (7, 19)
    m_long = np.asarray(compute_cell_multipole_moments(P, M, system=system).moments)
    P_prefix = make_lattice_matrix_set(
        basis.nbasis,
        list(P.cells)[:n_ball],
        [np.asarray(P.blocks[i]) for i in range(n_ball)],
    )
    m_prefix = np.asarray(
        compute_cell_multipole_moments(P_prefix, M, system=system).moments
    )
    assert np.array_equal(m_long, m_prefix)
    P_short = make_lattice_matrix_set(
        basis.nbasis,
        list(P.cells)[: n_ball - 1],
        [np.asarray(P.blocks[i]) for i in range(n_ball - 1)],
    )
    with pytest.raises(ValueError, match="must cover at least"):
        compute_cell_multipole_moments(P_short, M, system=system)


def test_symmetry_reduced_one_electron_sums_follow_the_switch():
    """The reduced S/T reconstruction templates on the same list the full
    builder enumerates, and matches it to machine precision either way."""
    from vibeqc.symmetry_integrals_reduced import (
        compute_kinetic_lattice_reduced,
        compute_overlap_lattice_reduced,
        one_electron_lattice_cells,
    )

    a = 8.0
    system = vq.PeriodicSystem(
        3, np.eye(3) * a, [vq.Atom(11, [0, 0, 0]), vq.Atom(17, [a / 2, a / 2, a / 2])]
    )
    vq.attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    for on, n_expected in ((False, 19), (True, 57)):
        opts = _lat(on, 12.0)
        S_full = vq.compute_overlap_lattice(basis, system, opts)
        T_full = vq.compute_kinetic_lattice(basis, system, opts)
        cells = one_electron_lattice_cells(basis, system, opts)
        assert len(cells) == len(S_full.cells) == n_expected
        _, S_rec = compute_overlap_lattice_reduced(
            basis, system, opts, system.symmetry.operations
        )
        _, T_rec = compute_kinetic_lattice_reduced(
            basis, system, opts, system.symmetry.operations
        )
        assert len(S_rec) == len(T_rec) == n_expected
        for full, rec in ((S_full, S_rec), (T_full, T_rec)):
            err = max(
                float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
                for x, y in zip(full.blocks, rec)
            )
            assert err < 1.0e-12


def test_bipole_1d_direct_nuclear_route_runs_under_the_switch():
    """1D cells take V_ne from the direct sum, so the BIPOLE Python
    consumers (padded J/K re-templating, moment prefix, keyed exchange)
    run end to end with the two lists of different length."""
    a, vac = 6.0, 15.0
    system = vq.PeriodicSystem(
        1,
        np.diag([a, vac, vac]),
        [vq.Atom(1, [0.0, vac / 2, vac / 2]), vq.Atom(1, [1.4, vac / 2, vac / 2])],
        0,
        1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    assert len(list(direct_lattice_cells(system, 11.0))) == 3
    assert len(list(pair_complete_lattice_cells(basis, system, 11.0))) == 5
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    energies = {}
    for on in (False, True):
        opts = vq.PeriodicSCFOptions()
        opts.conv_tol_energy = 1.0e-10
        opts.lattice_opts.pair_complete_1e = on
        opts.lattice_opts.cutoff_bohr = 11.0
        result = vq.run_pbc_bipole_rhf(system, basis, kmesh, opts, progress=False)
        assert result.converged
        energies[on] = float(result.energy)
    assert energies[False] == pytest.approx(-1.839238747730, abs=_PIN_TOL)
    assert energies[True] == pytest.approx(-1.8152364308953906, rel=0, abs=_PIN_TOL)


# Finite nuclear-source images must follow each physical AO product (#724).
def _source_fixture(dim=3, relabel=(0, 0, 0), delta=None, exponents=(.7, 1.1)):
    lattice = np.array([[6.0, 1.1, .3], [0.0, 5.7, .4], [0.0, 0.0, 6.4]])
    positions = np.array([[.1, .2, .3], [1.4, .3, .1], [2.2, 1.6, .4]])
    for atom, shift in enumerate(relabel):
        positions[atom] += shift * lattice[:, 0]
    if delta is not None:
        atom, axis, step = delta
        positions[atom, axis] += step
    system = vq.PeriodicSystem(
        dim, lattice, [vq.Atom(z, r) for z, r in zip([1, 1, 2], positions)],
    )
    shells = [vq.ShellInfo(i, 0, False, [exponents[i]], [1.0], r)
              for i, r in enumerate(positions[:2])]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, 'physical-nuclear-source', False)
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 8.1
    opts.nuclear_cutoff_bohr = 8.3
    opts.pair_complete_1e = True
    return system, basis, opts


def _source_reference(system, opts, omega, charges=None, exponents=(.7, 1.1),
                      physical_sources=True):
    # Independent normalized-s Gaussian integral and an oversized integer cube.
    # Source membership is defined physically, without the native enumerator.
    lattice = np.asarray(system.lattice)
    positions = np.array([a.xyz for a in system.unit_cell])
    ranges = [range(-5, 6) if d < system.dim else [0] for d in range(3)]
    indices = np.array(list(product(*ranges)))
    images = indices @ lattice.T
    sources = (images[:, None, :] + positions[None, :, :]).reshape(-1, 3)
    radial_sources = np.repeat(
        np.sum(images**2, axis=1) <= opts.nuclear_cutoff_bohr**2, len(positions),
    )
    weights = np.tile([1., 1., 2.] if charges is None else charges, len(images))
    result = np.zeros((2, 2))
    def boys(t):
        value = np.ones_like(t)
        nonzero = t > 1e-14
        value[nonzero] = (
            .5 * np.sqrt(np.pi / t[nonzero]) * erf(np.sqrt(t[nonzero]))
        )
        return value
    for a in range(2):
        for b in range(2):
            alpha, beta = exponents[a], exponents[b]
            p = alpha + beta
            norm = (4 * alpha * beta / np.pi**2) ** .75
            ratio = omega**2 / (p + omega**2)
            for image in images:
                A, B = positions[a], positions[b] + image
                distance2 = np.sum((A - B)**2)
                if distance2 > opts.cutoff_bohr**2:
                    continue
                center = .5 * (A + B)
                source_r2 = np.sum((sources - center)**2, axis=1)
                keep = (source_r2 <= opts.nuclear_cutoff_bohr**2
                        if physical_sources else radial_sources)
                product_center = (alpha * A + beta * B) / p
                t = p * np.sum((sources[keep] - product_center)**2, axis=1)
                screened = boys(t) - np.sqrt(ratio) * boys(ratio*t)
                result[a, b] -= (
                    norm * np.exp(-alpha * beta / p * distance2)
                    * 2 * np.pi / p * np.dot(weights[keep], screened)
                )
    return result


@pytest.mark.parametrize('nuclear_cutoff', [8.3, 12.3])
@pytest.mark.parametrize('dim', [1, 3])
@pytest.mark.parametrize('omega', [0.0, 0.4])
@pytest.mark.parametrize(
    "relabel", [(0, 0, 0), (1, 0, 0), (0, -1, 0), (0, 0, 2), (1, -1, 2)],
)
def test_physical_nuclear_source_matches_gaussian_sum(
    dim, omega, relabel, nuclear_cutoff,
):
    system, basis, opts = _source_fixture(dim, relabel)
    opts.nuclear_cutoff_bohr = nuclear_cutoff
    value = (compute_nuclear_lattice(basis, system, opts) if omega == 0
             else compute_nuclear_erfc_lattice(basis, system, omega, opts))
    got = np.sum(value.blocks, axis=0)
    expected = _source_reference(system, opts, omega)
    np.testing.assert_allclose(got, expected, rtol=0, atol=3e-12)
    np.testing.assert_allclose(got, got.T, rtol=0, atol=3e-12)
    base_system, _, base_opts = _source_fixture(dim)
    base_opts.nuclear_cutoff_bohr = nuclear_cutoff
    np.testing.assert_allclose(
        got, _source_reference(base_system, base_opts, omega), rtol=0, atol=3e-12,
    )


def test_physical_nuclear_source_effective_charges():
    system, basis, opts = _source_fixture(3, (1, -1, 2))
    charges = [.3, 0., 1.1]
    value = compute_nuclear_lattice_with_charges(basis, system, opts, charges)
    np.testing.assert_allclose(
        np.sum(value.blocks, axis=0), _source_reference(system, opts, 0., charges),
        rtol=0, atol=3e-12,
    )


@pytest.mark.parametrize('omega', [0.0, .4])
def test_physical_nuclear_source_gradient_matches_fd(omega):
    system, basis, opts = _source_fixture()
    overlap = compute_overlap_lattice(basis, system, opts)
    density = np.array([[.7, -.2], [-.2, .4]])
    D = make_lattice_matrix_set(
        2, list(overlap.cells), [density.copy() for _ in overlap.cells],
    )
    if omega:
        grad = nuclear_erfc_lattice_gradient_contribution(basis, system, D, opts, omega)
    else:
        grad = nuclear_lattice_gradient_contribution(basis, system, D, opts)
    step = 1e-4
    for atom, axis in [(0, 0), (1, 1), (2, 2)]:
        energies = []
        for h in [-step, step]:
            moved, _, moved_opts = _source_fixture(delta=(atom, axis, h))
            energies.append(np.sum(density * _source_reference(moved, moved_opts, omega)))
        assert grad[atom, axis] == pytest.approx(
            (energies[1] - energies[0]) / (2 * step), abs=1e-7,
        )
    np.testing.assert_allclose(np.sum(grad, axis=0), 0, atol=2e-11)


@pytest.mark.parametrize("sap", [False, True])
def test_nuclear_grid_translation_preserves_custom_shells(sap):
    from vibeqc._vibeqc_core import compute_nuclear_lattice_ewald, compute_vsap_lattice
    from vibeqc._vibeqc_core import build_grid

    system, basis = _h2_box()
    custom = vq.BasisSet(
        system.unit_cell_molecule(), list(basis.shells()),
        "custom-nuclear-grid-no-file", True,
    )
    grid = build_grid(system.unit_cell_molecule())
    opts = _lat(True)
    opts.nuclear_cutoff_bohr = 16.0
    eo = vq.EwaldOptions()
    eo.alpha = .6
    values = []
    for ao_basis in (basis, custom):
        if sap:
            value = compute_vsap_lattice(
                ao_basis, system, grid, "sap_helfem_large", opts, eo,
            )
        else:
            value = compute_nuclear_lattice_ewald(ao_basis, system, grid, opts, eo)
        values.append(value)
    for left, right in zip(values[0].blocks, values[1].blocks):
        np.testing.assert_allclose(left, right, rtol=0, atol=2e-13)



def test_sap_finite_source_domain_matches_gaussian_difference():
    from pathlib import Path
    from vibeqc._vibeqc_core import compute_vsap_lattice
    from vibeqc.lattice_screening import ao_pair_support_mask
    from vibeqc._vibeqc_core import build_grid

    system, basis, opts = _source_fixture()
    radial = _lat(False, opts.cutoff_bohr + np.linalg.norm(
        np.asarray(system.unit_cell[0].xyz) - system.unit_cell[1].xyz))
    radial.nuclear_cutoff_bohr = opts.nuclear_cutoff_bohr
    grid = build_grid(system.unit_cell_molecule())
    eo = vq.EwaldOptions()
    eo.alpha = .6
    physical = compute_vsap_lattice(basis, system, grid, "sap_helfem_large", opts, eo)
    historical = compute_vsap_lattice(basis, system, grid, "sap_helfem_large", radial, eo)
    assert [tuple(c.index) for c in physical.cells] == [
        tuple(c.index) for c in historical.cells
    ]
    difference = np.zeros((2, 2))
    for cell, left, right in zip(physical.cells, physical.blocks, historical.blocks):
        keep = ao_pair_support_mask(basis, cell.r_cart, opts.cutoff_bohr)
        difference += np.asarray(left) - np.where(keep, right, 0.0)

    # Read raw fitted coefficients; basis normalization would change them.
    table_path = Path(vq.__file__).parent / "basis_library/basis/sap_helfem_large.g94"
    text = table_path.read_text()
    terms = []
    net = np.zeros(3)
    for symbol, atoms in [("H", [0, 1]), ("He", [2])]:
        lines = text.split(symbol + " 0\n", 1)[1].splitlines()
        count = int(lines[0].split()[1])
        for line in lines[1:1 + count]:
            alpha, coefficient = map(float, line.split())
            charges = np.zeros(3)
            charges[atoms] = coefficient
            net += charges
            terms.append((np.sqrt(alpha), charges))
    terms.append((eo.alpha, -net))
    expected = np.zeros((2, 2))
    for omega, charges in terms:
        expected += _source_reference(system, opts, omega, charges)
        expected -= _source_reference(system, opts, omega, charges, physical_sources=False)
    assert np.max(np.abs(expected)) > 1e-5
    np.testing.assert_allclose(difference, expected, rtol=0, atol=3e-12)


def _physical_quartet_fixture(relabel):
    lattice = np.array([[6., 0., 0.], [.4, 8., 0.], [.1, 0., 9.]])
    positions = np.array([[.1, .2, .3], [1.4, .3, .1]])
    positions += np.array(relabel)[:, None] * lattice[:, 0]
    system = vq.PeriodicSystem(1, lattice, [vq.Atom(1, p) for p in positions])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(i, 0, False, [a], [1.], p)
        for i, (a, p) in enumerate(zip([.7, 1.1], positions))
    ], 'physical-quartet-oracle', False)
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 5.1
    opts.eri_interaction_cutoff_bohr = 7.4
    opts.pair_complete_1e = True
    opts.schwarz_threshold = 0.
    return system, basis, opts


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_physical_eri_enclosure_matches_independent_shifted_spheres(dimension):
    from vibeqc.lattice_screening import physical_eri_cells

    lattice = np.array([[6., .3, .1], [.4, 8., .2], [.1, .2, 9.]])
    positions = np.array([[.1, .2, .3], [1.4, .3, .1]])
    relabel = np.array([4, -1 if dimension > 1 else 0, 1 if dimension > 2 else 0])
    positions[1] += lattice @ relabel
    system = vq.PeriodicSystem(dimension, lattice, [vq.Atom(1, p) for p in positions])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(i, 0, False, [.7], [1.], p) for i, p in enumerate(positions)
    ], "eri-enclosure-oracle", False)
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 3.1
    opts.eri_interaction_cutoff_bohr = 4.2
    expected = set()
    axes = [range(-7, 8) if i < dimension else [0] for i in range(3)]
    for index in product(*axes):
        translation = lattice @ index
        if any(np.linalg.norm(a-b-translation) <= 7.3 for a in positions for b in positions):
            expected.add(index)
    cells = physical_eri_cells(basis, system, opts)
    assert {tuple(cell.index) for cell in cells} == expected
    assert tuple(cells[0].index) == (0, 0, 0)
    assert len(cells) < len(direct_lattice_cells(system, 7.3+np.linalg.norm(positions[1]-positions[0])))


@pytest.mark.parametrize("omega", [0., .4])
def test_distant_image_labels_do_not_expand_eri_storage_or_change_jk(omega):
    from vibeqc.lattice_screening import physical_eri_cells

    system0, _, opts = _physical_quartet_fixture((0, 0))
    opts.cutoff_bohr = opts.eri_interaction_cutoff_bohr = 2.0
    P = np.array([[.8, -.2], [-.2, .3]])
    reference = _physical_quartet_reference(system0, opts, omega, P)
    positions = np.array([atom.xyz for atom in system0.unit_cell])
    # The enclosing integer box has billions of empty entries. Three
    # pair neighborhoods suffice; the density lookup must remain sparse.
    positions[1] += np.asarray(system0.lattice) @ np.array([1000, 1000, 1000])
    system = vq.PeriodicSystem(3, system0.lattice, [vq.Atom(1, p) for p in positions])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(i, 0, False, [a], [1.], p)
        for i, (a, p) in enumerate(zip([.7, 1.1], positions))
    ], "distant-image-oracle", False)
    cells = physical_eri_cells(basis, system, opts)
    assert len(cells) == 3
    density = make_lattice_matrix_set(2, cells, [P for _ in cells])
    result = vq.build_jk_2e_real_space(basis, system, opts, density, omega)
    assert len(result.J.cells) == len(result.K.cells) == 3
    for actual, expected in zip((result.J, result.K), reference):
        np.testing.assert_allclose(np.sum(actual.blocks, axis=0), expected, rtol=0, atol=3e-10)


def test_physical_fock_support_does_not_use_legacy_symmetry_projector():
    from types import SimpleNamespace
    from vibeqc.pbc_bipole_common import resolve_bipole_fock_symmetry

    _, basis, opts = _physical_quartet_fixture((0, 0))
    system = SimpleNamespace(symmetry=SimpleNamespace(operations=[object()]))
    logger = SimpleNamespace(info=lambda message: None)
    assert resolve_bipole_fock_symmetry(
        system, basis, opts, None, None, logger, exchange_split_active=True,
    ) == (None, None)
    with pytest.raises(NotImplementedError, match="wider output support"):
        resolve_bipole_fock_symmetry(system, basis, opts, False, True, logger)


def _physical_quartet_reference(system, opts, omega, density):
    # Independent oversized integer enumeration and normalized Gaussian ERIs.
    positions = np.array([a.xyz for a in system.unit_cell])
    images = np.arange(-6, 7)[:, None] * np.asarray(system.lattice)[:, 0]
    alpha = [.7, 1.1]
    pair2 = opts.cutoff_bohr**2
    interaction2 = opts.eri_interaction_cutoff_bohr**2
    norm2 = lambda x: np.sum(x*x, axis=-1)

    def boys(t):
        root = np.sqrt(np.maximum(t, 1e-300))
        return np.where(t < 1e-12, 1.-t/3.+t*t/10., np.sqrt(np.pi)*erf(root)/(2.*root))

    def eri(a, b, c, d, A, B, C, D):
        p, q = a+b, c+d
        P, Q = (a*A+b*B)/p, (c*C+d*D)/q
        rho = p*q/(p+q)
        t = rho*norm2(P-Q)
        theta = omega**2/(omega**2+rho)
        f = boys(t)-np.sqrt(theta)*boys(theta*t)
        normalization = (2./np.pi)**3*(a*b*c*d)**.75
        return normalization*2.*np.pi**2.5/(p*q*np.sqrt(p+q))*f*np.exp(
            -a*b/p*norm2(A-B)-c*d/q*norm2(C-D))

    J, K = np.zeros((2, 2)), np.zeros((2, 2))
    for a in range(2):
        A = positions[a]
        for b in range(2):
            for g in images:
                B = positions[b]+g
                for c in range(2):
                    C = positions[c]+images[:, None, :]
                    for d in range(2):
                        D = positions[d]+images[None, :, :]
                        density_mask = norm2(C-D) <= pair2
                        jmask = (norm2(A-B) <= pair2) & density_mask & (norm2((A+B-C-D)/2.) <= interaction2)
                        kmask = (norm2(A-C) <= pair2) & (norm2(B-D) <= pair2)
                        kmask &= norm2((A+C-B-D)/2.) <= interaction2
                        J[a, b] += density[c, d]*np.sum(np.where(jmask, eri(alpha[a], alpha[b], alpha[c], alpha[d], A, B, C, D), 0.))
                        K[a, b] += density[c, d]*np.sum(np.where(kmask, eri(alpha[a], alpha[c], alpha[b], alpha[d], A, C, B, D), 0.))
    return J, K


@pytest.mark.parametrize('omega', [0., .4])
@pytest.mark.parametrize('relabel', [(0, 0), (0, 1), (1, 0), (-1, 2)])
def test_physical_quartets_match_gaussian_tensor_and_image_relabelling(omega, relabel):
    system, basis, opts = _physical_quartet_fixture(relabel)
    from vibeqc.pbc_bipole_fock import _sr_internal_cells

    cells = _sr_internal_cells(basis, system, opts, opts.eri_interaction_cutoff_bohr)
    P = np.array([[.8, -.2], [-.2, .3]])
    density = make_lattice_matrix_set(2, cells, [P for _ in cells])
    actual = vq.build_jk_2e_real_space(basis, system, opts, density, omega)
    expected = _physical_quartet_reference(system, opts, omega, P)
    gamma = vq.build_jk_gamma_molecular_limit(basis, system, opts, P, omega)
    for block, reference in zip((gamma.J, gamma.K), expected):
        assert np.max(abs(block-reference)) < 3e-12
    original, _, _ = _physical_quartet_fixture((0, 0))
    unshifted = _physical_quartet_reference(original, opts, omega, P)
    for result, reference, invariant in zip((actual.J, actual.K), expected, unshifted):
        assert [tuple(c.index) for c in result.cells] == [tuple(c.index) for c in cells]
        assert np.max(abs(np.sum(result.blocks, axis=0)-reference)) < 3e-12
        assert np.max(abs(reference-invariant)) < 3e-12


@pytest.mark.parametrize('omega', [0., .4])
def test_physical_sparse_subset_and_masked_outputs_preserve_the_tensor(omega):
    from vibeqc._vibeqc_core import build_jk_2e_real_space_output_subset_masked

    system, basis, opts = _physical_quartet_fixture((0, 1))
    from vibeqc.pbc_bipole_fock import _sr_internal_cells

    cells = _sr_internal_cells(basis, system, opts, opts.eri_interaction_cutoff_bohr)
    rng = np.random.default_rng(724)
    density = make_lattice_matrix_set(2, cells, [rng.normal(size=(2, 2)) for _ in cells])
    opts.schwarz_threshold = 1e-12
    opts.sr_range_screening = True
    opts.sr_sparse_traversal = False
    full = vq.build_jk_2e_real_space(basis, system, opts, density, omega)
    opts.sr_sparse_traversal = True
    sparse = vq.build_jk_2e_real_space(basis, system, opts, density, omega)
    subset = [len(cells)-1, 0]
    masks = [np.array([1, 0, 0, 1], dtype=np.uint8), np.array([0, 1, 1, 0], dtype=np.uint8)]
    masked = build_jk_2e_real_space_output_subset_masked(basis, system, opts, density, subset, masks, omega)
    for name in ('J', 'K'):
        assert np.array_equal(getattr(full, name).blocks, getattr(sparse, name).blocks)
        for index, block in enumerate(getattr(masked, name).blocks):
            mask = masks[subset.index(index)].reshape(2, 2) if index in subset else np.zeros((2, 2))
            assert np.array_equal(block, np.asarray(getattr(full, name).blocks[index])*mask)


@pytest.mark.parametrize('omega', [0.0, 0.4])
def test_physical_quartet_gradient_differentiates_the_finite_gaussian_energy(omega):
    from vibeqc._vibeqc_core import eri_lattice_gradient_contribution

    system, basis, opts = _physical_quartet_fixture((0, 1))
    opts.schwarz_threshold_forces = 0.0
    from vibeqc.pbc_bipole_fock import _sr_internal_cells

    cells = _sr_internal_cells(basis, system, opts, opts.eri_interaction_cutoff_bohr)
    P = np.array([[0.8, -0.2], [-0.2, 0.3]])
    density = make_lattice_matrix_set(2, cells, [P for _ in cells])
    gradient = np.asarray(eri_lattice_gradient_contribution(
        basis, system, density, opts, 1.0, 1.0, omega,
    ))
    h = 2e-5
    finite_difference = np.zeros((2, 3))
    for atom in range(2):
        for axis in range(3):
            energies = []
            for direction in [-1, 1]:
                positions = np.array([a.xyz for a in system.unit_cell])
                positions[atom, axis] += direction*h
                displaced = vq.PeriodicSystem(1, system.lattice, [vq.Atom(1, p) for p in positions])
                J, K = _physical_quartet_reference(displaced, opts, omega, P)
                energies.append(np.sum(P*(0.5*J-0.25*K)))
            finite_difference[atom, axis] = (energies[1]-energies[0])/(2*h)
    assert np.max(abs(gradient-finite_difference)) < 2e-9
    assert np.max(abs(gradient.sum(axis=0))) < 3e-13


@pytest.mark.parametrize('relabel', [(0, 0), (0, 1), (-1, 2)])
def test_direct_nuclear_physical_images_and_gradient_are_relabelling_invariant(relabel):
    from vibeqc._vibeqc_core import nuclear_repulsion_per_cell, nuclear_repulsion_gradient_per_cell

    system, _, opts = _physical_quartet_fixture(relabel)
    opts.nuclear_cutoff_bohr = 8.3
    energy = nuclear_repulsion_per_cell(system, opts)
    gradient = np.asarray(nuclear_repulsion_gradient_per_cell(system, opts))
    original, _, _ = _physical_quartet_fixture((0, 0))
    assert energy == pytest.approx(nuclear_repulsion_per_cell(original, opts), rel=0.0, abs=2e-14)
    assert np.max(abs(gradient-nuclear_repulsion_gradient_per_cell(original, opts))) < 2e-14
    h = 2e-5
    for atom in range(2):
        for axis in range(3):
            energies = []
            for direction in [-1, 1]:
                positions = np.array([a.xyz for a in system.unit_cell])
                positions[atom, axis] += direction*h
                displaced = vq.PeriodicSystem(1, system.lattice, [vq.Atom(1, p) for p in positions])
                energies.append(nuclear_repulsion_per_cell(displaced, opts))
            assert gradient[atom, axis] == pytest.approx((energies[1]-energies[0])/(2*h), rel=0.0, abs=2e-9)


@pytest.mark.parametrize('relabel', [(0, 0), (0, 1), (-1, 2)])
def test_physical_reciprocal_pair_values_and_derivatives_match_gaussians(relabel):
    from vibeqc.bipole_fock_ewald import _build_j_long_range_cache
    from vibeqc.bipole_gradient import _pair_ft_gradient_at_cells

    lattice = np.array([[6.0, 0.3, 0.1], [0.4, 8.0, 0.2], [0.1, 0.2, 9.0]])
    positions = np.array([[0.1, 0.2, 0.3], [1.4, 0.3, 0.1]])
    positions += np.array(relabel)[:, None]*lattice[:, 0]
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, p) for p in positions])
    alpha = [0.7, 1.1]
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(i, 0, False, [a], [1.0], p)
        for i, (a, p) in enumerate(zip(alpha, positions))
    ], 'physical-reciprocal-oracle', False)
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 5.1
    opts.pair_complete_1e = True
    cells = vq.compute_overlap_lattice(basis, system, opts).cells
    shifts = np.array([c.r_cart for c in cells])
    cache = _build_j_long_range_cache(basis, system, shifts, 0.4, 1e-8, K_max=1.3, lattice_opts=opts)
    bra, ket = _pair_ft_gradient_at_cells(basis, cache.K_vectors, shifts, lattice_opts=opts)
    expected = np.zeros_like(cache.ft_per_cell)
    expected_bra, expected_ket = np.zeros_like(bra), np.zeros_like(ket)
    for g, shift in enumerate(shifts):
        for i, a in enumerate(alpha):
            for j, b in enumerate(alpha):
                A, B = positions[i], positions[j]+shift
                delta = A-B
                if delta@delta > opts.cutoff_bohr**2:
                    continue
                center = (a*A+b*B)/(a+b)
                K = cache.K_vectors
                value = ((2*a/np.pi)*(2*b/np.pi))**0.75*(np.pi/(a+b))**1.5
                value = value*np.exp(-a*b/(a+b)*(delta@delta)-np.sum(K*K, axis=1)/(4*(a+b))-1j*(K@center))
                expected[g, i, j] = value
                expected_bra[g, i, j] = value[None, :]*(-2*a*b/(a+b)*delta[:, None]-1j*a/(a+b)*K.T)
                expected_ket[g, i, j] = value[None, :]*(2*a*b/(a+b)*delta[:, None]-1j*b/(a+b)*K.T)
    assert np.max(abs(cache.ft_per_cell-expected)) < 3e-13
    assert np.max(abs(bra-expected_bra)) < 3e-13
    assert np.max(abs(ket-expected_ket)) < 3e-13


def _physical_scf_fixture(method, relabel, delta=0.0):
    lattice = np.array([[6.0, .3, .1], [.4, 8.0, .2], [.1, .2, 9.0]])
    positions = np.array([[.1, .2, .3], [1.4+delta, .3, .1]])
    positions[1] += relabel*lattice[:, 0]
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, p) for p in positions])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(i, 0, False, [a], [1.0], p)
        for i, (a, p) in enumerate(zip([.7, 1.1], positions))
    ], "physical-exchange-split", False)
    opts = vq.PeriodicKSOptions() if method.endswith("ks") else vq.PeriodicSCFOptions()
    if method.endswith("ks"):
        opts.use_periodic_becke = True
    opts.lattice_opts.cutoff_bohr = 5.1
    opts.lattice_opts.nuclear_cutoff_bohr = 20.0
    opts.lattice_opts.pair_complete_1e = True
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-9
    opts.max_iter = 50
    return system, basis, opts, vq.monkhorst_pack(system, [1, 1, 1])


def _run_physical_scf(method, fixture, omega):
    from importlib import import_module

    suffix = "_" + method if method != "rhf" else ""
    run = getattr(import_module(f"vibeqc.pbc_bipole{suffix}"), f"run_pbc_bipole_{method}")
    system, basis, opts, mesh = fixture
    kwargs = {"functional": "lda"} if method.endswith("ks") else {}
    result = run(system, basis, mesh, opts, ewald_omega=omega,
                 ewald_precision=1e-12, sr_image_precision=1e-12,
                 progress=False, **kwargs)
    assert result.converged
    return result


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
def test_physical_exchange_domain_closes_the_ewald_split(method):
    # The first finite-quartet draft also cut the external K and density
    # pairs. This witness moved by 0.792 mHa between the two split values.
    values = [float(_run_physical_scf(method, _physical_scf_fixture(method, relabel), omega).energy)
              for omega, relabel in product([.4, .7], [0, 1])]
    for relabel in [0, 1]:
        assert abs(values[relabel]-values[relabel+2]) < 2e-8
    if not method.endswith("ks"):
        # The HF result also supplies an end-to-end relabelling check.
        # The finite XC image domain is separate from this erfc split test.
        assert abs(values[0]-values[1]) < 2e-11
        assert abs(values[2]-values[3]) < 2e-11


@pytest.mark.parametrize("omega", [.4, .7])
def test_physical_hf_operator_converges_to_independent_periodic_reference(omega):
    from vibeqc.pbc_bipole import run_pbc_bipole_rhf

    # Independent out-of-process PySCF 2.6.2 KRHF/RSJK, Gamma, exxdiv=ewald,
    # normalized primitive s functions, cell precision 1e-12, HCORE guess,
    # energy/gradient SCF tolerances 1e-11/1e-9. Omega .4/.7 energies differ
    # by 5e-15 Ha. The fixture's lattice columns and basis match that input.
    system, basis, opts, mesh = _physical_scf_fixture("rhf", 0)
    opts.lattice_opts.cutoff_bohr = 9.
    result = run_pbc_bipole_rhf(
        system, basis, mesh, opts, ewald_omega=omega, ewald_precision=1e-12,
        sr_image_extent_bohr=18., sr_image_precision=None,
        use_fock_symmetry=False, use_fock_symmetry_reduce=False, progress=False,
    )
    assert result.converged
    assert result.energy == pytest.approx(-.7301966695020943, rel=0, abs=2e-8)
    np.testing.assert_allclose(result.overlap, [[
        [1.0000063548234759, .45741568045731357],
        [.45741568045731357, 1.0000000045855646],
    ]], rtol=0, atol=2e-11)
    np.testing.assert_allclose(result.hcore, [[
        [-.3340570625600636, -.4773139720380655],
        [-.4773139720380655, -.0851591187970735],
    ]], rtol=0, atol=2e-9)
    np.testing.assert_allclose(result.fock, [[
        [-.07389099069321048, -.5209361757235138],
        [-.5209361757235138, .3596320564177773],
    ]], rtol=0, atol=2e-8)


@pytest.mark.parametrize("method", ["rks", "uks"])
@pytest.mark.parametrize("relabel", [0, 1])
def test_physical_lda_operator_matches_independent_periodic_reference(method, relabel):
    from importlib import import_module

    # Independent PySCF 2.6.2 periodic KRKS, LDA_X + VWN5, Gamma, exact
    # same lattice/basis as the HF reference above. Becke levels 7/9 differ
    # by 1.46e-9 Ha; the level-9 grid has 231765 points.
    system, basis, opts, mesh = _physical_scf_fixture(method, relabel)
    opts.lattice_opts.cutoff_bohr = 9.
    opts.becke_image_radius_bohr = 14.
    run = getattr(import_module(f"vibeqc.pbc_bipole_{method}"), f"run_pbc_bipole_{method}")
    result = run(
        system, basis, mesh, opts, functional="lda", ewald_omega=.4,
        ewald_precision=1e-12, sr_image_extent_bohr=18., sr_image_precision=None,
        use_fock_symmetry=False, use_fock_symmetry_reduce=False, progress=False,
    )
    assert result.converged
    assert result.energy == pytest.approx(-.7067490013957616, rel=0, abs=2e-8)
    fock = result.fock if method == "rks" else result.fock_alpha
    np.testing.assert_allclose(fock, [[
        [.1648856501676761, -.21022398530134762],
        [-.21022398530134762, .4117958827541709],
    ]], rtol=0, atol=2e-8)


@pytest.mark.parametrize("omega", [.4, .7])
@pytest.mark.parametrize("relabel", [0, 1])
def test_physical_coupled_scf_gradient_matches_energy_fd(omega, relabel):
    from vibeqc.bipole_gradient import compute_bipole_gradient_rhf

    fixture = _physical_scf_fixture("rhf", relabel)
    system, basis, opts, mesh = fixture
    result = _run_physical_scf("rhf", fixture, omega)
    with pytest.warns(UserWarning, match="preview"):
        gradient = compute_bipole_gradient_rhf(
            system, basis, result, lattice_opts=opts.lattice_opts, kmesh=mesh,
        )
    h = 2e-5
    energies = [_run_physical_scf("rhf", _physical_scf_fixture("rhf", relabel, delta), omega).energy
                for delta in [-h, h]]
    assert abs(gradient[1, 0]-(energies[1]-energies[0])/(2*h)) < 2e-8
    assert np.max(abs(np.sum(gradient, axis=0))) < 2e-11


def test_explicit_physical_gamma_uses_three_image_indices():
    from vibeqc.pbc_bipole_fock import _sr_internal_cells

    system, basis, opts = _physical_quartet_fixture((0, 1))
    cells = _sr_internal_cells(basis, system, opts, opts.eri_interaction_cutoff_bohr)
    P = np.array([[.8, -.2], [-.2, .3]])
    actual = vq.build_jk_gamma_molecular_limit_explicit(basis, system, cells, opts, P, .4)
    expected = _physical_quartet_reference(system, opts, .4, P)
    for block, reference in zip((actual.J, actual.K), expected):
        np.testing.assert_allclose(block, reference, rtol=0, atol=3e-12)
    with pytest.raises(ValueError, match="three-image domains API"):
        vq.build_jk_pair_contributions(basis, system, cells, [(0, 0)], opts, P, .4)


@pytest.mark.parametrize("omega", [0.0, 0.4])
def test_physical_non_gamma_fock_covariance_and_density_derivative(omega, record_property):
    from vibeqc._vibeqc_core import build_jk_2e_real_space_domains_density_derivative
    from vibeqc.pbc_bipole_fock import _sr_internal_cells

    rows = []
    for relabel in [(0, 0), (0, 1), (-1, 2)]:
        system, basis, opts = _physical_quartet_fixture(relabel)
        cells = _sr_internal_cells(basis, system, opts, opts.eri_interaction_cutoff_bohr)
        centers = np.array([s.origin for s in basis.shells()])
        wavevector = np.pi*np.linalg.inv(np.asarray(system.lattice))[0]/2
        P = np.array([[.8, -.2], [-.2, .3]])
        blocks = [P*np.cos((centers[None, :]-centers[:, None]+c.r_cart) @ wavevector)
                  for c in cells]
        density = make_lattice_matrix_set(2, cells, blocks)
        jk = vq.build_jk_2e_real_space(basis, system, opts, density, omega)
        energy = sum(np.sum(p*(.5*j-.25*k)) for p,j,k in zip(blocks,jk.J.blocks,jk.K.blocks))
        # Restore the original integer labels before comparing each raw block.
        covariant = {}
        for cell, j, k in zip(cells, jk.J.blocks, jk.K.blocks):
            for a,b in product(range(2), repeat=2):
                index = np.array(cell.index)
                index[0] += relabel[b]-relabel[a]
                covariant[(*index,a,b)] = (j[a,b],k[a,b])
        rows.append((energy,covariant))
        if relabel == (0, 1):
            result = build_jk_2e_real_space_domains_density_derivative(
                basis, system, opts, density, cells, list(range(len(cells))), omega, True,
            )
            for name, forward, derivative in [
                ("J", result.J, result.J_density_energy_derivative),
                ("K", result.K, result.K_density_energy_derivative),
            ]:
                record_property(name+"_density_adjoint_max_abs", float(np.max(abs(
                    np.asarray(forward.blocks)-np.asarray(derivative.blocks)))))
                np.testing.assert_allclose(forward.blocks, derivative.blocks, rtol=0, atol=3e-12)
            direction = np.random.default_rng(704).normal(size=np.shape(blocks))
            h = 1e-4
            energies = []
            for sign in [-1, 1]:
                displaced = np.asarray(blocks)+sign*h*direction
                displaced_density = make_lattice_matrix_set(2, cells, list(displaced))
                trial = vq.build_jk_2e_real_space(basis, system, opts, displaced_density, omega)
                energies.append(sum(np.sum(p*(.5*j-.25*k))
                                    for p,j,k in zip(displaced,trial.J.blocks,trial.K.blocks)))
            derivative = sum(np.sum(d*(j-.5*k))
                             for d,j,k in zip(direction,jk.J.blocks,jk.K.blocks))
            error = abs(derivative-(energies[1]-energies[0])/(2*h))
            record_property("density_directional_fd_abs", float(error))
            assert error < 3e-10
    reference_energy, reference = rows[0]
    for energy, blocks in rows[1:]:
        assert abs(energy-reference_energy) < 3e-12
        for key in reference.keys() | blocks.keys():
            np.testing.assert_allclose(reference.get(key,(0.,0.)), blocks.get(key,(0.,0.)),
                                       rtol=0, atol=3e-12)


@pytest.mark.parametrize("n_kpoints", [2, 3])
def test_half_translation_exchange_outputs_match_gaussian_character_sum(
    n_kpoints, record_property,
):
    """Independent finite SR operator witness for the #62 output-support gate.

    The pair radius keeps only on-site normalized s products. Their Coulomb
    integral is the interaction of two unit Gaussian charges, so no AO ERI
    builder, quartet-domain planner or symmetry reconstruction enters the
    reference. Exchange still has off-site outputs at both nearest images.
    This finite-sum test does not claim convergence to the periodic limit.
    """
    from math import erf, sqrt
    from vibeqc.pbc_bipole_fock import _sr_density_cells

    lattice = np.diag([8., 16., 16.])
    positions = np.array([[0., 0., 0.], [4., 0., 0.]])
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, p) for p in positions])
    exponent, omega = .7, .4
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(i, 0, False, [exponent], [1.], p)
        for i, p in enumerate(positions)
    ], "half-translation-gaussian-reference", False)
    opts = _lat(True, cutoff=3.1)
    opts.eri_interaction_cutoff_bohr = 5.
    opts.schwarz_threshold = 0.
    density_cells = _sr_density_cells(basis, system, opts, 5.)
    rng = np.random.default_rng(62)
    trial = []
    for k in range(n_kpoints):
        panel = rng.normal(size=(2, 2))
        if n_kpoints == 3 and k == 1:
            panel = panel + 1j*rng.normal(size=(2, 2))
        trial.append(panel@panel.conj().T)
    if n_kpoints == 3:
        trial[-1] = trial[1].conj()
    actions = [np.array([[0., np.exp(-2j*np.pi*k/n_kpoints)], [1., 0.]])
               for k in range(n_kpoints)]
    image = [u@d@u.conj().T for u, d in zip(actions, trial)]
    assert max(np.max(abs(a-b)) for a, b in zip(trial, image)) > .1

    def inverse(matrices, label):
        return sum(np.exp(-2j*np.pi*k*label[0]/n_kpoints)*d
                   for k, d in enumerate(matrices))/n_kpoints

    outputs, cropped = [], []
    for label_name, matrices in (("source", trial), ("image", image)):
        blocks = [inverse(matrices, cell.index) for cell in density_cells]
        assert max(np.max(abs(block.imag)) for block in blocks) < 1e-12
        density = make_lattice_matrix_set(2, density_cells, [b.real for b in blocks])
        actual = vq.build_jk_2e_real_space(basis, system, opts, density, omega)
        expected_j = np.zeros((2, 2), complex)
        expected_k = np.zeros((n_kpoints, 2, 2), complex)
        cropped_k = np.zeros_like(expected_k)
        # Oversized integer cube. Only the on-site and +/- half-cell charge
        # separations survive the declared midpoint interaction radius.
        for label in product(range(-2, 3), repeat=3):
            shift = lattice@label
            for a, b in product(range(2), repeat=2):
                distance = np.linalg.norm(positions[a]-positions[b]-shift)
                if distance > 5.:
                    continue
                rho_sr = exponent*omega**2/(exponent+omega**2)
                value = ((erf(sqrt(exponent)*distance)-erf(sqrt(rho_sr)*distance))/distance
                         if distance else 2./sqrt(np.pi)*(sqrt(exponent)-sqrt(rho_sr)))
                expected_j[a, a] += value*inverse(matrices, (0, 0, 0))[b, b]
                for k in range(n_kpoints):
                    entry = np.exp(2j*np.pi*k*label[0]/n_kpoints)*value*inverse(matrices, label)[a, b]
                    expected_k[k, a, b] += entry
                    if label == (0, 0, 0):
                        cropped_k[k, a, b] += entry
        folded = [np.array([sum(np.exp(2j*np.pi*k*int(cell.index[0])/n_kpoints)*block
                                for cell, block in zip(part.cells, part.blocks))
                            for k in range(n_kpoints)]) for part in (actual.J, actual.K)]
        np.testing.assert_allclose(folded[0], np.broadcast_to(expected_j, folded[0].shape), rtol=0, atol=3e-12)
        np.testing.assert_allclose(folded[1], expected_k, rtol=0, atol=3e-12)
        record_property(label_name+"_J_gaussian_max_abs", float(np.max(abs(folded[0]-expected_j))))
        record_property(label_name+"_K_gaussian_max_abs", float(np.max(abs(folded[1]-expected_k))))
        outputs.append(folded)
        cropped.append(cropped_k)
    for original, transformed in zip(*outputs):
        np.testing.assert_allclose(transformed,
            [u@a@u.conj().T for u, a in zip(actions, original)], rtol=0, atol=3e-12)
    # Cropping only the exchange outputs back to the overlap-radius cell
    # ball loses a required partner even though the retained ERIs are exact.
    cropped_residual = max(np.max(abs(b-u@a@u.conj().T))
                           for a, b, u in zip(*cropped, actions))
    record_property("cropped_K_covariance_max_abs", float(cropped_residual))
    assert cropped_residual > 1e-5


def test_he2_two_character_raw_fock_covariance_and_ewald_split():
    from vibeqc.pbc_bipole import run_pbc_bipole_rhf

    system = vq.PeriodicSystem(3, np.diag([8., 16., 16.]), [
        vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [4., 0., 0.]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "6-31g")
    mesh = vq.monkhorst_pack(system, [2, 1, 1])
    energies = []
    for omega in [.4, .7]:
        opts = vq.PeriodicSCFOptions()
        opts.lattice_opts.cutoff_bohr = 7.0
        opts.lattice_opts.nuclear_cutoff_bohr = 20.0
        opts.lattice_opts.pair_complete_1e = True
        opts.lattice_opts.sr_range_screening = True
        opts.initial_guess = vq.InitialGuess.HCORE
        opts.conv_tol_energy = 1e-10
        opts.conv_tol_grad = 1e-8
        opts.max_iter = 50
        result = run_pbc_bipole_rhf(
            system, basis, mesh, opts, ewald_omega=omega,
            ewald_precision=1e-12, sr_image_precision=1e-12,
            use_fock_symmetry=False, use_fock_symmetry_reduce=False, progress=False,
        )
        assert result.converged
        energies.append(result.energy)
        electronic = 0.0
        for k,w,F,H,C in zip(mesh.kpoints, mesh.weights, result.fock, result.hcore, result.mo_coeffs):
            translation = np.block([
                [np.zeros((2, 2)), np.exp(-1j*k[0]*8.)*np.eye(2)],
                [np.eye(2), np.zeros((2, 2))],
            ])
            # Use the returned raw Fock, with reconstruction disabled.
            np.testing.assert_allclose(F, translation.conj().T@F@translation, rtol=0, atol=2e-11)
            density = 2*np.asarray(C)[:, :2]@np.asarray(C)[:, :2].conj().T
            electronic += .5*w*np.einsum("ij,ji", density, H+F).real
        assert abs(electronic-result.e_electronic) < 2e-11
    # Measured drift is 2.79e-9 Ha at this finite reciprocal resolution.
    assert abs(energies[0]-energies[1]) < 2e-8


@pytest.mark.parametrize("relabel", [0, 1])
def test_physical_unsplit_hartree_cache_and_batches_match_gaussian_sum(relabel):
    from vibeqc.aux_basis import rsgdf_dense_g_mesh
    from vibeqc.ewald_composed import (
        build_j_ewald_3d_ft_gamma_cache,
        build_j_ewald_3d_ft_lattice_cache,
        compute_j_ewald_3d_ft_k_density_to_cells,
        compute_j_ewald_3d_ft_lattice,
        compute_j_ewald_3d_ft_gamma,
    )
    from vibeqc.lattice_screening import physical_eri_cells

    system, basis, scf, _ = _physical_scf_fixture("rhf", relabel)
    opts = scf.lattice_opts
    cells = physical_eri_cells(basis, system, opts)
    ke = 1.5
    vectors = rsgdf_dense_g_mesh(system, ke)
    vectors = vectors[np.sum(vectors*vectors, axis=1) > 0]
    positions = np.array([atom.xyz for atom in system.unit_cell])
    expected = np.zeros((len(cells), 2, 2, len(vectors)), dtype=complex)
    for g, cell in enumerate(cells):
        for i, a in enumerate([.7, 1.1]):
            for j, b in enumerate([.7, 1.1]):
                A, B = positions[i], positions[j]+cell.r_cart
                distance2 = np.dot(A-B, A-B)
                if distance2 > opts.cutoff_bohr**2:
                    continue
                center = (a*A+b*B)/(a+b)
                expected[g, i, j] = (
                    ((2*a/np.pi)*(2*b/np.pi))**.75*(np.pi/(a+b))**1.5
                    * np.exp(-a*b/(a+b)*distance2
                             - np.sum(vectors*vectors, axis=1)/(4*(a+b))
                             - 1j*(vectors@center))
                )
    cache = build_j_ewald_3d_ft_lattice_cache(
        basis, system, cells, lattice_opts=opts, ke_cutoff=ke,
    )
    np.testing.assert_allclose(cache.pair_at_cells, expected, rtol=0, atol=3e-13)
    gamma = build_j_ewald_3d_ft_gamma_cache(
        basis, system, lattice_opts=opts, ke_cutoff=ke,
    )
    np.testing.assert_allclose(gamma.pair_ft, expected.sum(axis=0), rtol=0, atol=3e-13)
    D = np.array([[.7, .2], [.2, .9]])
    density = make_lattice_matrix_set(2, cells, [D.copy() for _ in cells])
    rho = np.einsum("mn,gmnk->k", D, expected)
    exact = np.einsum("k,k,gmnk->gmn", 4*np.pi/np.sum(vectors*vectors, axis=1),
                      rho, expected.conj()).real/abs(np.linalg.det(system.lattice))
    cached = compute_j_ewald_3d_ft_lattice(basis, system, density, .4, cache=cache)
    np.testing.assert_allclose(cached, exact, rtol=0, atol=3e-13)
    for chunk in [1, len(cells)]:
        batched = compute_j_ewald_3d_ft_k_density_to_cells(
            basis, system, [D], [np.zeros(3)], [1.], cells, .4,
            lattice_opts=opts, ke_cutoff=ke, chunk_size=11, cell_chunk_size=chunk,
        )
        np.testing.assert_allclose(batched, exact, rtol=0, atol=3e-13)
    reordered = make_lattice_matrix_set(2, list(reversed(cells)),
                                         [D.copy() for _ in cells])
    with pytest.raises(ValueError, match="cell ordering"):
        compute_j_ewald_3d_ft_lattice(basis, system, reordered, .4, cache=cache)
    changed = vq.LatticeSumOptions()
    changed.pair_complete_1e = True
    changed.cutoff_bohr = opts.cutoff_bohr + 1.
    with pytest.raises(ValueError, match="physical pair support"):
        compute_j_ewald_3d_ft_lattice(basis, system, density, .4,
                                    lattice_opts=changed, cache=cache)
    with pytest.raises(ValueError, match="physical pair support"):
        compute_j_ewald_3d_ft_gamma(basis, system, D, .4,
                                  lattice_opts=changed, cache=gamma)


@pytest.mark.parametrize("method", ["rhf", "rks", "uks"])
@pytest.mark.parametrize("mesh_shape", [[1, 1, 1], [2, 1, 1]])
def test_physical_generic_ewald_drivers_preserve_exchange_support(method, mesh_shape, monkeypatch):
    from importlib import import_module

    monkeypatch.setenv("VIBEQC_J_EWALD3D_KE", "20.0")
    system, basis, opts, _ = _physical_scf_fixture(method, 0)
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.eri_interaction_cutoff_bohr = 14.
    if method.endswith("ks"):
        opts.functional = "pbe0"
    mesh = vq.monkhorst_pack(system, mesh_shape)
    run = getattr(import_module(f"vibeqc.periodic_{method}_multi_k_ewald"),
                  f"run_{method}_periodic_multi_k_ewald3d")
    results = [run(system, basis, mesh, opts, omega=omega,
                   auto_optimize_truncation=False, progress=False)
               for omega in [.4, .7]]
    assert all(result.converged for result in results)
    assert abs(results[0].energy-results[1].energy) < 2e-8


@pytest.mark.parametrize("family", ["gamma", "tight", "multi-k"])
def test_cosx_refuses_unsupported_physical_quartet_contract(family):
    from vibeqc import _vibeqc_core as core
    from vibeqc.periodic_cosx_k import KPointCosxK

    system, basis = _h2_box()
    opts = _lat(True)
    if family == "multi-k":
        with pytest.raises(NotImplementedError, match="physical quartet support"):
            KPointCosxK(basis, system, lat_opts=opts)
    elif family == "tight":
        with pytest.raises(ValueError, match="physical quartet support"):
            core.make_periodic_tight_cosx_jk_builder(
                basis, np.zeros((1, basis.nbasis, basis.nbasis)), system, opts,
            )
    else:
        # The refusal precedes construction of the auxiliary metric.
        with pytest.raises(ValueError, match="physical quartet support"):
            core.make_periodic_gamma_cosx_jk_builder(basis, basis, system, opts)


@pytest.mark.parametrize("family", ["gamma", "lattice"])
def test_diagnostic_grid_hartree_refuses_physical_quartet_contract(family, monkeypatch):
    from vibeqc.ewald_composed import build_j_ewald_3d
    from vibeqc.periodic_fock_multi_k import ewald_3d_j_blocks

    monkeypatch.setenv("VIBEQC_J_EWALD3D_BACKEND", "grid")
    system, basis = _h2_box()
    opts = _lat(True)
    density = np.eye(basis.nbasis)
    with pytest.raises(NotImplementedError, match="physical quartet support"):
        if family == "gamma":
            build_j_ewald_3d(basis, system, density, .4, lattice_opts=opts)
        else:
            cells = list(direct_lattice_cells(system, 0.))
            real_density = make_lattice_matrix_set(basis.nbasis, cells, [density])
            ewald_3d_j_blocks(basis, system, real_density, .4, lattice_opts=opts)


def test_physical_gamma_refuses_explicit_uncorrected_exchange_gauge():
    from vibeqc.periodic_rhf_multi_k_ewald import run_rhf_periodic_multi_k_ewald3d

    system, basis, opts, mesh = _physical_scf_fixture("rhf", 0)
    with pytest.raises(NotImplementedError, match="requires corrected exchange"):
        run_rhf_periodic_multi_k_ewald3d(
            system, basis, mesh, opts, exchange_exxdiv="none",
            auto_optimize_truncation=False, progress=False,
        )
