"""#429 stage 2, Family 2: consumers of the one-electron cell list are list-agnostic.

Under ``LatticeSumOptions.pair_complete_1e`` the overlap / kinetic / direct
nuclear sums ride the pair-complete cell list of
``cpp/include/vibeqc/lattice_pair_cells.hpp`` while the two-electron builds
keep the plain ``|g|`` ball, an exact prefix of it. Every consumer that used
to pair the two lists index-for-index is exercised here with the lists of
different length, and the Ewald-split nuclear family, whose two halves have
not moved together yet, must refuse the switch rather than assemble an Hcore
that keeps T(g) but drops V_ne(g) on the tail cells.

The switch stays off by default; the switch-off numbers are pinned as the
bit-identity contract of this generation.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    compute_multipole_moments_lattice,
    compute_nuclear_erfc_lattice,
    compute_overlap_lattice,
    direct_lattice_cells,
    make_lattice_matrix_set,
    nuclear_erfc_lattice_gradient_contribution,
    pair_complete_lattice_cells,
)

# H2 in a 6-bohr cubic box, STO-3G, 2x2x2 mesh, DIRECT_TRUNCATED. At an
# 8-bohr cutoff the ball has 7 cells and the pair-complete list 19 (the
# intra-cell offset is 1.4 bohr), so every index-for-index consumer sees
# two lists of different length.
_CUTOFF = 8.0
_E_RHF_OFF = -34.655444157536  # switch off: the historical sum, bit-identical
_E_RHF_ON = -34.655500639285  # switch on: the outer one-electron cells survive
_E_RKS_OFF = -34.661376318338
_E_RKS_ON = -34.661460481227
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


def _run_cxx(on: bool, *, ks: bool = False, guess=None, cutoff: float = _CUTOFF):
    system, basis = _h2_box()
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
def test_cxx_multi_k_fock_assembly_keeps_the_outer_one_electron_cells(ks):
    """F(g) = Hcore(g) + F2e(g) starts from Hcore so the tail is not cut back.

    The switch-off value is the historical, bit-identical answer; the
    switch-on value differs by the one-electron tail the ball dropped
    (5.6e-05 Ha for RHF, 8.4e-05 for LDA at this cutoff) and converges
    towards the switch-off value with the cutoff (6e-07 at 10 bohr, 5e-11
    at 15 bohr where the two lists coincide).
    """
    e_off = _run_cxx(False, ks=ks)
    e_on = _run_cxx(True, ks=ks)
    assert e_off == pytest.approx(_E_RKS_OFF if ks else _E_RHF_OFF, abs=_PIN_TOL)
    assert e_on == pytest.approx(_E_RKS_ON if ks else _E_RHF_ON, abs=_PIN_TOL)
    assert abs(e_on - e_off) > 1.0e-5


def test_cxx_multi_k_switch_converges_to_the_ball_with_the_cutoff():
    """Where the pair-complete list equals the ball the switch is a no-op."""
    system, basis = _h2_box()
    assert len(list(direct_lattice_cells(system, 15.0))) == len(
        list(pair_complete_lattice_cells(basis, system, 15.0))
    )
    e_off = _run_cxx(False, cutoff=15.0)
    e_on = _run_cxx(True, cutoff=15.0)
    # Only the per-pair filter differs (pairs past the cutoff inside the ball).
    assert abs(e_on - e_off) < 1.0e-9


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
# The Ewald-split nuclear family refuses the switch (both halves must move
# together; the gauge decision is handed to the route that builds the
# reciprocal half).
# ---------------------------------------------------------------------------


def test_ewald_split_nuclear_family_refuses_the_switch():
    system, basis = _h2_box()
    D = compute_overlap_lattice(basis, system, _lat(True))
    with pytest.raises(ValueError, match="pair_complete_1e.*Ewald-split nuclear"):
        compute_nuclear_erfc_lattice(basis, system, 0.5, _lat(True))
    with pytest.raises(ValueError, match="pair_complete_1e.*Ewald-split nuclear"):
        nuclear_erfc_lattice_gradient_contribution(basis, system, D, _lat(True), 0.5)
    with pytest.raises(ValueError, match="compute_vsap_lattice.*pair_complete_1e"):
        _run_cxx(True, guess=vq.InitialGuess.SAP)


def test_bipole_3d_refuses_the_switch_at_the_nuclear_builder():
    """The 3D BIPOLE drivers take V_ne from the Ewald split, so the refusal
    surfaces there with the reason, before any Fock is assembled."""
    system, basis = _h2_box()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.pair_complete_1e = True
    with pytest.raises(ValueError, match="pair_complete_1e.*Ewald-split nuclear"):
        vq.run_pbc_bipole_rhf(system, basis, kmesh, opts, progress=False)


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
    for on, n_expected in ((False, 7), (True, 19)):
        D = compute_overlap_lattice(basis, system, _lat(on))  # stand-in density
        K = build_exchange_blocks(basis, system, _lat(on), D, exx)
        assert len(K) == len(D.cells) == n_expected
        assert np.abs(K[0]).max() > 0.0
        for g in range(7, len(K)):
            assert not np.asarray(K[g]).any()


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
    assert energies[True] == pytest.approx(-1.839225294147, abs=_PIN_TOL)
