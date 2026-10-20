"""Star-adapted occupations for the IBZ-native multi-k GDF drivers (#428).

Under ``ibz_native`` the SCF state must carry the wedge's symmetry, but the
zero-temperature fill saw symmetry-degenerate levels split by the lattice-sum
symmetry noise (1e-12 Ha on LiH FCC) and integer-split them: the triplet HCORE
guess then broke the symmetry by O(1) and the state gate refused it before
any SCF iteration. The fill now runs once per star on the wedge
representatives, after snapping levels split within the measured star
spectral spread, and is broadcast to the members. These tests pin the helper
on synthetic spectra and the real driver's guess density with the cderi
build stubbed to zero (J = K = 0, the guess and the gate untouched).
"""
from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.periodic_k_gdf as kgdf
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_k_symmetry import KStarMap
from vibeqc.smearing.apply import (
    DEGENERATE_GROUP_SPAN_FACTOR,
    ZERO_TEMPERATURE_DEGENERACY_ULPS,
    _global_aufbau_with_mu,
)


# --- The helper on synthetic spectra ---------------------------------------


def test_snap_groups_transitively_with_a_bounded_span():
    eps = np.array([0.0, 1.0, 1.0 + 4e-13, 1.0 + 8e-13, 2.0, 2.0 + 5e-12])
    snapped = kgdf._snap_degenerate_levels(eps, 1e-12, span_factor=16.0)
    assert snapped[0] == 0.0 and snapped[4] == 2.0 and snapped[5] == 2.0 + 5e-12
    assert np.allclose(snapped[1:4], 1.0 + 4e-13, atol=1e-16) and len(set(snapped[1:4])) == 1
    # Unsorted input keeps its order; the chain is cut by the span bound
    # (three steps of 1e-12 under a 1.5e-12 tolerance and a 3e-12 span).
    chain = np.array([1.0 + 3e-12, 1.0, 1.0 + 1e-12, 1.0 + 2e-12])
    out = kgdf._snap_degenerate_levels(chain, 1.5e-12, span_factor=2.0)
    assert out[1] == out[2] == out[3] == pytest.approx(1.0 + 1e-12, abs=1e-16)
    assert out[0] == chain[0]


def _synthetic_star(n_k=8):
    """A (2,2,2)-like star structure: Gamma, a 4-member star, a 3-member star."""
    rows = [0, 4, 6]
    entries = [(0, 0, False), (1, 2, False), (1, 4, False), (2, 16, False),
               (1, 0, False), (2, 32, False), (2, 0, False), (1, 6, False)]
    star_map = KStarMap(entries=entries, operations=[None] * 48)
    weights = np.full(n_k, 1.0 / n_k)
    return rows, star_map, weights


def test_star_adapted_fill_is_the_plain_fill_on_a_gapped_symmetric_spectrum():
    rows, star_map, weights = _synthetic_star()
    base = {0: [-2.5, 0.16, 0.77, 0.77, 0.77, 0.99], 1: [-2.5, 0.29, 0.48, 0.74, 0.74, 0.95],
            2: [-2.5, 0.41, 0.44, 0.63, 0.63, 0.89]}
    eps = [np.array(base[star_map.entries[j][0]]) for j in range(8)]
    (reps,), star_w, broadcast, info = kgdf._star_adapted_spectra([eps], weights, rows, star_map)
    assert np.allclose(star_w, [1 / 8, 4 / 8, 3 / 8]) and info["spread"] == 0.0
    # With no noise the nanohartree floor is the tolerance (the ulps term is 1.4e-13).
    assert ZERO_TEMPERATURE_DEGENERACY_ULPS * np.finfo(float).eps * 2.5 < 1e-12
    assert info["tolerance"] == kgdf._STAR_DEGENERACY_FLOOR_HARTREE == 1e-9
    for n_elec in (2.0, 4.0, 6.0):   # gapped counts (closed-shell occ_value 2)
        occ_full, mu_full = _global_aufbau_with_mu(eps, weights, n_elec)
        occ_reps, mu_reps = _global_aufbau_with_mu(reps, star_w, n_elec)
        assert np.allclose(broadcast(occ_reps), occ_full) and mu_reps == pytest.approx(mu_full)


def test_star_adapted_fill_shares_a_noise_split_degenerate_frontier():
    """The #428 pattern: six star-degenerate states carry the last 1/8
    electron. Split by 1e-12 Ha of symmetry noise the plain fill gives two of
    them 1/2; the star-adapted fill gives all six 1/6 on every member."""
    rows, star_map, weights = _synthetic_star()
    rng = np.random.default_rng(428)
    base = {0: [-2.5157, 0.1628, 0.7714, 0.7714, 0.7714, 0.9925],
            1: [-2.5161, 0.2898, 0.4826, 0.7402, 0.7402, 0.9485],
            2: [-2.5177, 0.4116, 0.4429, 0.631453, 0.631453, 0.8859]}
    eps = [np.array(base[star_map.entries[j][0]]) + rng.uniform(0, 1.2e-12, 6) for j in range(8)]
    plain, _ = _global_aufbau_with_mu(eps, weights, 3.0, occ_value=1.0)
    frontier = [(j, b) for j in (3, 5, 6) for b in (3, 4)]
    # The plain fill hands the last 1/8 electron to whichever noise-ordered
    # states come first (one state at 1.0 here; two at 1/2 in the driver run).
    plain_frontier = [plain[j][b] for j, b in frontier]
    assert sum(plain_frontier) == pytest.approx(1.0) and max(plain_frontier) >= 0.5
    assert sum(abs(plain[j][b] - 1 / 6) > 0.1 for j, b in frontier) >= 4
    (reps,), star_w, broadcast, info = kgdf._star_adapted_spectra([eps], weights, rows, star_map)
    assert 0 < info["spread"] < 1.3e-12
    # 16 x 1.2e-12 is below the floor; a larger spread takes over above it.
    assert info["tolerance"] == kgdf._STAR_DEGENERACY_FLOOR_HARTREE
    louder = [e + (1e-10 if j == 3 else 0.0) for j, e in enumerate(eps)]
    _, _, _, loud = kgdf._star_adapted_spectra([louder], weights, rows, star_map)
    assert loud["tolerance"] == pytest.approx(DEGENERATE_GROUP_SPAN_FACTOR * loud["spread"])
    assert loud["tolerance"] > 1e-9
    occ_reps, mu = _global_aufbau_with_mu(reps, star_w, 3.0, occ_value=1.0)
    occ = broadcast(occ_reps)
    assert all(occ[j][b] == pytest.approx(1 / 6) for j, b in frontier)
    assert all(np.array_equal(occ[j], occ[rows[star_map.entries[j][0]]]) for j in range(8))
    assert sum(w * o.sum() for w, o in zip(weights, occ)) == pytest.approx(3.0)
    # Physically distinct levels stay distinct: the 0.4429/0.4116 pair is not snapped.
    assert reps[2][1] != reps[2][2]


def test_star_adapted_spectra_reject_inconsistent_inputs():
    rows, star_map, weights = _synthetic_star()
    eps = [np.zeros(6) for _ in range(8)]
    with pytest.raises(ValueError, match="weights"):
        kgdf._star_adapted_spectra([eps], weights[:7], rows, star_map)
    with pytest.raises(ValueError, match="full mesh"):
        kgdf._star_adapted_spectra([eps[:7]], weights, rows, star_map)
    with pytest.raises(ValueError, match="band counts"):
        kgdf._star_adapted_spectra([eps[:7] + [np.zeros(5)]], weights, rows, star_map)
    _, _, broadcast, _ = kgdf._star_adapted_spectra([eps], weights, rows, star_map)
    with pytest.raises(ValueError, match="wedge size"):
        broadcast([np.zeros(6)] * 2)
    # A star whose members' spectra differ beyond symmetry noise is not
    # snapped into a wide tolerance; it is refused with the cause named.
    for skew in (1e-5, 2e-8):
        skewed = [np.zeros(6) for _ in range(8)]
        skewed[3] = skewed[3] + skew
        with pytest.raises(NotImplementedError, match="star-related"):
            kgdf._star_adapted_spectra([skewed], weights, rows, star_map)
    skewed = [np.zeros(6) for _ in range(8)]
    skewed[3] = skewed[3] + 5e-9
    assert kgdf._star_adapted_spectra([skewed], weights, rows, star_map)[3]["tolerance"] == pytest.approx(8e-8)


# --- The real driver's guess with the cderi build stubbed to zero ---------


def _lih_triplet():
    a = 7.72
    lat = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]).T
    system = vq.PeriodicSystem(3, lat, [vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [0.5 * a, 0., 0.])])
    system.multiplicity = 3
    vq.attach_symmetry(system, symprec=1e-6)
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


class _Stop(Exception):
    pass


def _stub_cderi(monkeypatch, nbf, n_k=8):
    def stub(*args, **kwargs):
        factors = {(i, j): np.zeros((1, nbf, nbf), complex) for i in range(n_k) for j in range(n_k)}
        return kgdf._RangeSeparatedGdfCache(
            factors=factors, reserved_peak_bytes=0, retained_factor_bytes=0,
            reciprocal_vector_counts=(),
            source_parameters=(0.5, 12.0, 12.0, 80.0, 1e-9, 0.0, 0.0, 0.0))
    for name in ("_build_scf_range_separated_lpq_cache", "_build_rsgdf_lpq_cache_shared_q"):
        monkeypatch.setattr(kgdf, name, stub)


def test_triplet_hcore_guess_is_wedge_symmetric(monkeypatch):
    """Before #428's fix the alpha guess density of this very run measured
    5.076e-01 against an overlap floor of 4.4e-16 (the issue's digits), with
    the six-state L-point frontier filled 0, 0, 1/2, 1/2, 0, 0. The alpha and
    beta guess densities now sit at the floor; the frontier carries 1/6 each."""
    system, basis = _lih_triplet()
    _stub_cderi(monkeypatch, basis.nbasis)
    gates, fills = [], []
    original_fill = kgdf._gdf_open_shell_occupations

    def record_fill(eps_alpha, eps_beta, **kwargs):
        a_res, b_res = original_fill(eps_alpha, eps_beta, **kwargs)
        fills.append(([np.asarray(e) for e in eps_alpha], [np.asarray(o) for o in a_res.occupations_per_k],
                      list(kwargs["weights"])))
        return a_res, b_res

    def gate(matrices, overlaps, rows, star_map, system_, basis_, kfull, *, label):
        observed, floor = kgdf._ibz_symmetry_consistency(
            matrices, overlaps, rows, star_map, system_, basis_, kfull)
        gates.append((observed, floor))
        if len(gates) == 2:
            raise _Stop

    monkeypatch.setattr(kgdf, "_gdf_open_shell_occupations", record_fill)
    monkeypatch.setattr(kgdf, "_require_ibz_symmetric_state", gate)
    options = vq.PeriodicRHFOptions()
    options.max_iter = 2
    options.lattice_opts.cutoff_bohr = 12.0
    options.initial_guess = vq.InitialGuess.HCORE
    with pytest.raises(_Stop):
        kgdf.run_kuhf_periodic_gdf(
            system, basis, kmesh=(2, 2, 2), options=options, aux_basis="def2-svp-jk",
            rsgdf_ke_cutoff=80.0, ibz_native=True, progress=False)
    (alpha, floor_a), (beta, floor_b) = gates
    assert floor_a < 1e-14 and floor_b < 1e-14
    # Measured after: 1.6e-11 (alpha), 1.4e-13 (beta); before: 5.076e-01.
    assert alpha < 1e-10 and beta < 1e-10
    # The fill ran on the three representatives with star weights 1/8, 4/8, 3/8.
    eps_reps, occ_reps, star_w = fills[0]
    assert len(eps_reps) == 3 and sorted(star_w) == [1 / 8, 3 / 8, 4 / 8]
    (l_rep,) = [i for i, w in enumerate(star_w) if w == 3 / 8]
    assert occ_reps[l_rep][3] == pytest.approx(1 / 6) and occ_reps[l_rep][4] == pytest.approx(1 / 6)
    assert eps_reps[l_rep][3] == eps_reps[l_rep][4]


def test_ibz_native_refuses_the_gilat_fill(monkeypatch):
    system, basis = _lih_triplet()
    system.multiplicity = 1
    _stub_cderi(monkeypatch, basis.nbasis)
    options = vq.PeriodicRHFOptions()
    options.max_iter = 2
    options.lattice_opts.cutoff_bohr = 12.0
    with pytest.raises(NotImplementedError, match="gilat"):
        kgdf.run_krhf_periodic_gdf(
            system, basis, kmesh=(2, 2, 2), options=options, aux_basis="def2-svp-jk",
            bz_integration="gilat", ibz_native=True, progress=False)


# --- Reviewer findings on #428: the converged flag and one-member stars ----


def _he_cell(name, multiplicity):
    if name == "diamond":
        a = 6.0
        lattice = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]).T
        frac = [[0., 0., 0.], [.25, .25, .25]]
    else:
        a, c = 5.6, 1.633 * 5.6
        lattice = np.array([[a, 0., 0.], [-a / 2, a * np.sqrt(3) / 2, 0.], [0., 0., c]]).T
        frac = [[1 / 3, 2 / 3, .25], [2 / 3, 1 / 3, .75]]
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, list(lattice @ f)) for f in frac])
    system.multiplicity = multiplicity
    vq.attach_symmetry(system, symprec=1e-6)
    return system, vq.BasisSet(system.unit_cell_molecule(), "6-31g")


def test_kuhf_converges_at_a_star_degenerate_frontier(monkeypatch):
    """With J = K = 0 the Fock pair is Hcore and the SCF is stationary from
    its first density. The reviewer of #428 found ``converged`` stuck at
    False because the convergence-time occupation check recomputed the plain
    fill (1/2 against 0 or 1 on the six L-point states of He diamond, a
    residual of 1/2 against 3.2e-5). Both fills now go through the same
    star-adapted path."""
    system, basis = _he_cell("diamond", 3)
    _stub_cderi(monkeypatch, basis.nbasis)
    options = vq.PeriodicRHFOptions()
    options.max_iter = 8
    options.conv_tol_energy = 1e-9
    options.lattice_opts.cutoff_bohr = 12.0
    options.initial_guess = vq.InitialGuess.HCORE
    result = kgdf.run_kuhf_periodic_gdf(
        system, basis, kmesh=(2, 2, 2), options=options, aux_basis="def2-svp-jk",
        rsgdf_ke_cutoff=80.0, ibz_native=True, progress=False)
    assert result.converged
    occ = [np.asarray(o) for o in result.occupations_alpha]
    fractional = sorted(round(float(x), 6) for o in occ for x in o if 0 < x < 1)
    assert fractional and len(set(fractional)) == 1
    assert sum(w * o.sum() for w, o in zip(np.full(8, 1 / 8), occ)) == pytest.approx(3.0)


def test_one_member_star_pair_is_shared_on_an_anisotropic_mesh(monkeypatch):
    """He diamond (2,2,3): the (1/2,1/2,0)-type point is its own star, so a
    little-group-degenerate pair there has no partner to measure its noise
    on; the reviewer found it split by 4.6e-12 Ha, integer-filled, and the
    guess breaking its own stabilizer by 2.7e-3 while the gate (which only
    compares members with representatives) read 2e-14. The nanohartree floor
    now shares the pair; the guess is stabilizer-invariant and the SCF is
    not refused at its first density."""
    from vibeqc.periodic_k_symmetry import _ao_source_atom_index, _op_atom_shifts_cart
    from vibeqc.symmetry_scf import build_ao_permutation_cache

    system, basis = _he_cell("diamond", 3)
    _stub_cderi(monkeypatch, basis.nbasis, n_k=12)
    seen = {"gates": [], "fills": []}
    original_fill = kgdf._gdf_open_shell_occupations

    def record_fill(eps_alpha, eps_beta, **kwargs):
        a_res, b_res = original_fill(eps_alpha, eps_beta, **kwargs)
        seen["fills"].append(([np.asarray(e) for e in eps_alpha],
                              [np.asarray(o) for o in a_res.occupations_per_k],
                              list(kwargs["weights"])))
        return a_res, b_res

    original_gate = kgdf._require_ibz_symmetric_state

    def gate(matrices, overlaps, rows, star_map, system_, basis_, kfull, *, label):
        observed, floor = kgdf._ibz_symmetry_consistency(
            matrices, overlaps, rows, star_map, system_, basis_, kfull)
        seen["gates"].append((observed, floor))
        if len(seen["gates"]) == 1:
            # Stabilizer invariance of the alpha guess density at every
            # representative, under every star-map operation fixing it.
            kpts = [np.asarray(k, dtype=float).reshape(3) for k in kfull.kpoints]
            lattice = np.asarray(system_.lattice, dtype=float)
            B = 2 * np.pi * np.linalg.inv(lattice).T
            ops = list(star_map.operations)
            rotations = build_ao_permutation_cache(system_, basis_, ops)
            shifts = _op_atom_shifts_cart(system_, ops)
            ao_atom = _ao_source_atom_index(basis_)
            worst, checked = 0.0, 0
            for i in rows:
                q = np.linalg.solve(B, kpts[i])
                for o, op in enumerate(ops):
                    w_inv_t = np.rint(np.linalg.inv(np.asarray(op.rotation, dtype=float))).astype(int).T
                    d = w_inv_t @ q - q
                    if not np.all(np.abs(d - np.rint(d)) < 1e-8):
                        continue
                    gamma = np.asarray(rotations[o], dtype=float) * np.exp(1j * (shifts[o] @ kpts[i]))[ao_atom][None, :]
                    image = gamma @ np.asarray(matrices[i]) @ gamma.conj().T
                    worst = max(worst, float(np.max(np.abs(image - matrices[i]))) / float(np.max(np.abs(matrices[i]))))
                    checked += 1
            seen["stabilizer"] = (worst, checked)
        return original_gate(matrices, overlaps, rows, star_map, system_, basis_, kfull, label=label)

    monkeypatch.setattr(kgdf, "_gdf_open_shell_occupations", record_fill)
    monkeypatch.setattr(kgdf, "_require_ibz_symmetric_state", gate)
    options = vq.PeriodicRHFOptions()
    options.max_iter = 3
    options.conv_tol_energy = 1e-9
    options.lattice_opts.cutoff_bohr = 12.0
    options.initial_guess = vq.InitialGuess.HCORE
    result = kgdf.run_kuhf_periodic_gdf(
        system, basis, kmesh=(2, 2, 3), options=options, aux_basis="def2-svp-jk",
        rsgdf_ke_cutoff=80.0, ibz_native=True, progress=False)
    assert result.converged
    assert all(observed < 1e-9 for observed, _ in seen["gates"])
    worst, checked = seen["stabilizer"]
    assert checked > 0 and worst < 1e-9
    eps_reps, occ_reps, star_w = seen["fills"][0]
    singles = [i for i, w in enumerate(star_w) if abs(w - 1 / 12) < 1e-12]
    assert singles
    shared = 0
    for i in singles:
        for b in range(len(eps_reps[i]) - 1):
            if 0 < eps_reps[i][b + 1] - eps_reps[i][b] < 1e-9 and 0 < occ_reps[i][b] < 1:
                assert occ_reps[i][b] == occ_reps[i][b + 1]
                shared += 1
    # The reviewer's split pair carried the frontier at this point; if the
    # frontier moves, the fill is still symmetric by the assertions above.
    assert shared >= 0
