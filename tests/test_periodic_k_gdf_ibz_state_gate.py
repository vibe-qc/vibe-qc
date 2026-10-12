"""The IBZ-native state gate measures the little group at every representative (#470).

``_require_ibz_symmetric_state`` compared each star member with its
representative transported through the star map's operation, so a state that
breaks its own little group at a wedge representative and is propagated to
the members through those same operations passed at roundoff (measured
1.6e-11 against a 1e-5 limit for a 0.17-relative break at Gamma on LiH FCC,
while an independent stabilizer residual read 1.1e-1). The gate now also
transports every representative through each admitted operation that fixes
it, time reversal included, and refuses a little-group break like a star
break. These tests build the densities outside the gate and run the real
driver with the cderi build stubbed to zero.
"""
from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.periodic_k_gdf as kgdf
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_k_symmetry import (
    _ao_source_atom_index,
    _op_atom_shifts_cart,
    little_group_entries,
    transport_k_matrix_at,
)
from vibeqc.symmetry_scf import build_ao_permutation_cache


def _cell(name, multiplicity=1):
    if name == "lih":
        a = 7.72
        lattice = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]).T
        atoms = [vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [0.5 * a, 0., 0.])]
        basis_name = "sto-3g"
    elif name == "diamond":
        a = 6.0
        lattice = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]).T
        atoms = [vq.Atom(2, list(lattice @ f)) for f in ([0., 0., 0.], [.25, .25, .25])]
        basis_name = "6-31g"
    else:
        a, c = 5.6, 1.633 * 5.6
        lattice = np.array([[a, 0., 0.], [-a / 2, a * np.sqrt(3) / 2, 0.], [0., 0., c]]).T
        atoms = [vq.Atom(2, list(lattice @ f)) for f in ([1 / 3, 2 / 3, .25], [2 / 3, 1 / 3, .75])]
        basis_name = "6-31g"
    system = vq.PeriodicSystem(3, lattice, atoms)
    system.multiplicity = multiplicity
    vq.attach_symmetry(system, symprec=1e-6)
    return system, vq.BasisSet(system.unit_cell_molecule(), basis_name)


def _mesh(system, mesh):
    kmesh = core.monkhorst_pack(system, list(mesh), [0, 0, 0], False)
    kpts = np.array([np.asarray(k, dtype=float) for k in kmesh.kpoints])
    return kpts, np.full(len(kpts), 1.0 / len(kpts))


# --- Little-group entries ----------------------------------------------------


@pytest.mark.parametrize("name, mesh, expected", [
    ("lih", (2, 2, 2), {0: 96, 4: 24, 6: 32}),
    ("diamond", (2, 2, 3), None),
    ("hcp", (2, 2, 2), None),
])
def test_little_group_entries_match_an_independent_search(name, mesh, expected):
    system, _ = _cell(name)
    kpts, weights = _mesh(system, mesh)
    rows, star_map, kfull = kgdf._resolve_ibz_native_state(
        system, kpts, mesh, weights, None, exchange_producer="sr")
    entries = little_group_entries(star_map, system, kfull, rows)
    B = 2 * np.pi * np.linalg.inv(np.asarray(system.lattice, dtype=float)).T
    for j, fixing in zip(rows, entries):
        q = np.linalg.solve(B, kpts[j])
        own = []
        for o, op in enumerate(star_map.operations):
            w_inv_t = np.rint(np.linalg.inv(np.asarray(op.rotation, dtype=float))).astype(int).T
            for trev, sign in ((False, 1.0), (True, -1.0)):
                d = sign * (w_inv_t @ q) - q
                if np.all(np.abs(d - np.rint(d)) < 1e-8):
                    own.append((o, trev))
        assert fixing == own
        assert (0, False) in fixing or any(not trev for _, trev in fixing)
        if expected is not None:
            assert len(fixing) == expected[j]
    # On a Gamma-centred even mesh every point is its own time-reversal
    # partner, so time reversal fixes every representative.
    if all(m % 2 == 0 for m in mesh):
        assert all(any(trev for _, trev in fixing) for fixing in entries)


# --- Explicit densities through the gate ------------------------------------


class _Stop(Exception):
    pass


def _capture_guess(monkeypatch, system, basis, mesh=(2, 2, 2), driver="run_kuhf_periodic_gdf"):
    """Run the real driver with J = K = 0 and capture the first gate call."""
    nbf = basis.nbasis
    n_k = int(np.prod(mesh))

    def stub(*args, **kwargs):
        factors = {(i, j): np.zeros((1, nbf, nbf), complex) for i in range(n_k) for j in range(n_k)}
        return kgdf._RangeSeparatedGdfCache(
            factors=factors, reserved_peak_bytes=0, retained_factor_bytes=0,
            reciprocal_vector_counts=(), source_parameters=(0.5, 12.0, 12.0, 80.0, 1e-9, 0.0, 0.0, 0.0))

    captured = {}

    def gate(matrices, overlaps, rows, star_map, system_, basis_, kfull, *, label):
        captured.update(matrices=[np.array(m) for m in matrices], overlaps=[np.array(s) for s in overlaps],
                        rows=list(rows), star_map=star_map, kfull=kfull, label=label)
        raise _Stop

    for name in ("_build_scf_range_separated_lpq_cache", "_build_rsgdf_lpq_cache_shared_q"):
        monkeypatch.setattr(kgdf, name, stub)
    monkeypatch.setattr(kgdf, "_require_ibz_symmetric_state", gate)
    options = vq.PeriodicRHFOptions()
    options.max_iter = 2
    options.lattice_opts.cutoff_bohr = 12.0
    options.initial_guess = vq.InitialGuess.HCORE
    with pytest.raises(_Stop):
        getattr(kgdf, driver)(
            system, basis, kmesh=mesh, options=options, aux_basis="def2-svp-jk",
            rsgdf_ke_cutoff=80.0, ibz_native=True, progress=False)
    return captured


def _propagate_from_representative(cap, system, basis, broken_rep):
    """Members of the broken representative's star, through the star map's own ops."""
    rows, star_map, kfull = cap["rows"], cap["star_map"], cap["kfull"]
    ops = star_map.operations
    P = build_ao_permutation_cache(system, basis, ops)
    shifts = _op_atom_shifts_cart(system, ops)
    ao_atom = _ao_source_atom_index(basis)
    kpts = [np.asarray(k, dtype=float).reshape(3) for k in kfull.kpoints]
    out = [m.copy() for m in cap["matrices"]]
    rep_j, block = broken_rep
    i_rep = rows.index(rep_j)
    for j, (i, o, trev) in enumerate(star_map.entries):
        if i == i_rep:
            out[j] = transport_k_matrix_at(block, kpts[j], P[o], shifts[o], ao_atom, time_reversal=trev)
    return out


def test_symmetric_guess_passes_and_a_little_group_break_is_refused(monkeypatch):
    """LiH FCC triplet alpha guess: both residuals at the floor. The same density
    broken at Gamma (little group of 96 operations incl. time reversal) and
    propagated star-consistently: star residual unchanged at 1.6e-11, little
    group 1.1e-1, refused by the production gate naming the little group."""
    system, basis = _cell("lih", 3)
    cap = _capture_guess(monkeypatch, system, basis)
    D, S_k, rows, star_map, kfull = (cap["matrices"], cap["overlaps"], cap["rows"],
                                     cap["star_map"], cap["kfull"])
    (star, little), (star_floor, little_floor) = kgdf._ibz_symmetry_residuals(
        D, S_k, rows, star_map, system, basis, kfull)
    assert star < 1e-9 and little < 1e-9
    assert star_floor < 1e-14 and little_floor < 1e-14
    observed, floor = kgdf._ibz_symmetry_consistency(D, S_k, rows, star_map, system, basis, kfull)
    assert observed == max(star, little) and floor == max(star_floor, little_floor)
    # The real gate accepts it.
    monkeypatch.undo()
    kgdf._require_ibz_symmetric_state(D, S_k, rows, star_map, system, basis, kfull, label="test")

    rng = np.random.default_rng(470)
    nbf = basis.nbasis
    A = rng.normal(size=(nbf, nbf)) + 1j * rng.normal(size=(nbf, nbf))
    gamma_point = rows[0]
    assert np.allclose(kfull.kpoints[gamma_point], 0.0)
    broken_block = D[gamma_point] + 0.05 * (A + A.conj().T)
    broken = _propagate_from_representative(cap, system, basis, (gamma_point, broken_block))
    (star_b, little_b), _ = kgdf._ibz_symmetry_residuals(
        broken, S_k, rows, star_map, system, basis, kfull)
    # Measured: star 1.6e-11 (the blind spot), little group 1.1e-1.
    assert star_b < 1e-9
    assert little_b > 1e-2
    with pytest.raises(NotImplementedError, match="little group") as refusal:
        kgdf._require_ibz_symmetric_state(broken, S_k, rows, star_map, system, basis, kfull, label="test")
    assert "symmetry" in str(refusal.value)
    # A star break (a member changed) is still reported as such.
    member = next(j for j, (i, o, _) in enumerate(star_map.entries) if i == 0 and j != gamma_point) \
        if any(i == 0 and j != gamma_point for j, (i, o, _) in enumerate(star_map.entries)) else None
    member = next(j for j in range(len(D)) if j not in rows)
    star_broken = [m.copy() for m in D]
    star_broken[member] = D[member] + 0.05 * (A + A.conj().T)
    with pytest.raises(NotImplementedError, match="across the stars"):
        kgdf._require_ibz_symmetric_state(star_broken, S_k, rows, star_map, system, basis, kfull, label="test")


@pytest.mark.parametrize("name, mesh", [("diamond", (2, 2, 3)), ("hcp", (2, 2, 2))])
def test_non_symmorphic_guesses_pass_the_little_group_term(monkeypatch, name, mesh):
    """He on diamond and hcp sites, triplet HCORE guess (#428 fill): the
    little-group residual sits at the floor on every representative, including
    the one-member star of the (2,2,3) mesh whose pair the #428 floor shares."""
    system, basis = _cell(name, 3)
    cap = _capture_guess(monkeypatch, system, basis, mesh=mesh)
    D, S_k, rows, star_map, kfull = (cap["matrices"], cap["overlaps"], cap["rows"],
                                     cap["star_map"], cap["kfull"])
    (star, little), (star_floor, little_floor) = kgdf._ibz_symmetry_residuals(
        D, S_k, rows, star_map, system, basis, kfull)
    assert max(star, little) < 1e-9 and max(star_floor, little_floor) < 1e-13
    entries = little_group_entries(star_map, system, kfull, rows)
    if mesh == (2, 2, 3):
        weights = np.full(len(D), 1.0 / len(D))
        singles = [i for i, j in enumerate(rows)
                   if sum(1 for (ii, _, _) in star_map.entries if ii == i) == 1]
        assert singles and all(len(entries[i]) > 1 for i in singles)


# --- Driver: the in-SCF gate refuses a little-group break at a representative --


def test_driver_refuses_a_little_group_break_injected_at_a_representative(monkeypatch):
    """The in-SCF counterpart of the slow symmetry-break test, at a
    representative and with the cderi build stubbed: the alpha and beta
    guesses pass, the third gate call sees a density broken at Gamma and
    propagated through the star map, and refuses it naming the little group.
    Before #470 this injection was invisible to the gate."""
    system, basis = _cell("lih", 3)
    nbf = basis.nbasis

    def stub(*args, **kwargs):
        factors = {(i, j): np.zeros((1, nbf, nbf), complex) for i in range(8) for j in range(8)}
        return kgdf._RangeSeparatedGdfCache(
            factors=factors, reserved_peak_bytes=0, retained_factor_bytes=0,
            reciprocal_vector_counts=(), source_parameters=(0.5, 12.0, 12.0, 80.0, 1e-9, 0.0, 0.0, 0.0))

    for name in ("_build_scf_range_separated_lpq_cache", "_build_rsgdf_lpq_cache_shared_q"):
        monkeypatch.setattr(kgdf, name, stub)
    original = kgdf._require_ibz_symmetric_state
    calls, witnessed = [], []

    def inject(matrices, overlaps, rows, star_map, system_, basis_, kfull, *, label):
        calls.append(label)
        if len(calls) == 3:
            ops = star_map.operations
            P = build_ao_permutation_cache(system_, basis_, ops)
            shifts = _op_atom_shifts_cart(system_, ops)
            ao_atom = _ao_source_atom_index(basis_)
            kpts = [np.asarray(k, dtype=float).reshape(3) for k in kfull.kpoints]
            rep_j = rows[0]
            rng = np.random.default_rng(3)
            A = rng.normal(size=(nbf, nbf)) + 1j * rng.normal(size=(nbf, nbf))
            block = np.asarray(matrices[rep_j]) + 0.05 * (A + A.conj().T)
            i_rep = list(rows).index(rep_j)
            for j, (i, o, trev) in enumerate(star_map.entries):
                if i == i_rep:
                    matrices[j][...] = transport_k_matrix_at(
                        block, kpts[j], P[o], shifts[o], ao_atom, time_reversal=trev)
            (star, little), _ = kgdf._ibz_symmetry_residuals(
                matrices, overlaps, rows, star_map, system_, basis_, kfull)
            witnessed.append((star, little))
        return original(matrices, overlaps, rows, star_map, system_, basis_, kfull, label=label)

    monkeypatch.setattr(kgdf, "_require_ibz_symmetric_state", inject)
    options = vq.PeriodicRHFOptions()
    options.max_iter = 4
    options.lattice_opts.cutoff_bohr = 12.0
    options.initial_guess = vq.InitialGuess.HCORE
    with pytest.raises(NotImplementedError, match="little group"):
        kgdf.run_kuhf_periodic_gdf(
            system, basis, kmesh=(2, 2, 2), options=options, aux_basis="def2-svp-jk",
            rsgdf_ke_cutoff=80.0, ibz_native=True, progress=False)
    assert len(calls) == 3
    (star, little), = witnessed
    assert star < 1e-9 and little > 1e-2
