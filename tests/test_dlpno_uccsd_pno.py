"""Compiled PNO-space open-shell DLPNO-CCSD (``residual_domain="pno"``).

Two independent oracles pin the compiled kernel
``vibeqc._vibeqc_core.dlpno_uccsd_pno_solve``:

* the residual identity. Every contracted virtual index of the spin-orbital
  CCSD residual is carried by an amplitude, so for any PNO truncation, with
  every coupling set the whole occupied space, a pair residual in its PNO
  space equals ``U^T R U`` of the canonical residual ``R`` of
  ``_ccsd_ref.so_residuals`` evaluated with the pair amplitudes embedded in
  the full virtual space, and the singles residual equals ``R1`` itself.
  ``dlpno_uccsd_pno_residuals`` exposes one evaluation, checked here on
  random spin-blocked density-fitting factors;
* the Python "union" route (tests/test_dlpno_uccsd_local.py), which contracts
  each residual in the union of its coupling set's pair spaces with the dense
  spin-orbital domain kernel. The routes build their pair spaces differently
  (combined versus per-spin diagonalisation, the same spaces) and treat the
  singles differently in a pair residual (projected into the union versus
  exact), so their energies agree to the convergence tolerance rather than
  bit for bit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from vibeqc import BasisSet, UHFOptions, run_uhf
from vibeqc._vibeqc_core import Atom, Molecule, dlpno_uccsd_pno_residuals
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno._ccsd_ref import so_residuals
from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions, run_dlpno_uccsd_pilot
from vibeqc.dlpno.uccsd_local_solver import (
    LocalUCCSDOptions,
    run_local_dlpno_uccsd,
)

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"
OH = [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.97])]
SEPARATED_OH_H2 = OH + [(1, [0.0, 0.0, 8.0]), (1, [0.0, 0.0, 8.74])]
CH2_TRIPLET = [
    (6, [0.0, 0.0, 0.0]),
    (1, [0.9913, 0.0, 0.4209]),
    (1, [-0.9913, 0.0, 0.4209]),
]


def _system(atoms, multiplicity, basis_name):
    molecule = Molecule(
        [Atom(z, [c * ANGSTROM_TO_BOHR for c in position]) for z, position in atoms],
        charge=0,
        multiplicity=multiplicity,
    )
    basis = BasisSet(molecule, basis_name)
    scf_options = UHFOptions()
    scf_options.max_iter = 400
    uhf = run_uhf(molecule, basis, scf_options)
    assert uhf.converged
    df = DensityFitting(basis, BasisSet(molecule, AUX), aux_basis_name=AUX)
    return molecule, basis, uhf, df


@pytest.fixture(scope="module")
def oh_sto3g():
    return _system(OH, 2, "sto-3g")


@pytest.fixture(scope="module")
def oh_svp():
    return _system(OH, 2, "def2-svp")


@pytest.fixture(scope="module")
def separated_sto3g():
    return _system(SEPARATED_OH_H2, 2, "sto-3g")


@pytest.fixture(scope="module")
def separated_svp():
    return _system(SEPARATED_OH_H2, 2, "def2-svp")


@pytest.fixture(scope="module")
def ch2_triplet_svp():
    return _system(CH2_TRIPLET, 3, "def2-svp")


# --- the residual identity ---------------------------------------------------


def _random_problem(
    seed, noa, nob, nva, nvb, naux, truncate, drop, spin_structured, layout="sorted"
):
    """Random spin-blocked DF factors, Fock, singles and pair PNO spaces."""
    rng = np.random.default_rng(seed)
    no, nv = noa + nob, nva + nvb
    n = no + nv

    def sym(x):
        return 0.5 * (x + x.transpose(0, 2, 1))

    b_oo = [sym(rng.normal(size=(naux, k, k))) * 0.3 for k in (noa, nob)]
    b_ov = [rng.normal(size=(naux, o, v)) * 0.3 for o, v in ((noa, nva), (nob, nvb))]
    b_vv = [sym(rng.normal(size=(naux, k, k))) * 0.3 for k in (nva, nvb)]
    occ = [np.arange(noa), np.arange(noa, no)]
    vir = [no + np.arange(nva), no + np.arange(nva, nv)]
    b_full = np.zeros((naux, n, n))
    for s in range(2):
        b_full[:, occ[s][:, None], occ[s][None, :]] = b_oo[s]
        b_full[:, occ[s][:, None], vir[s][None, :]] = b_ov[s]
        b_full[:, vir[s][:, None], occ[s][None, :]] = b_ov[s].transpose(0, 2, 1)
        b_full[:, vir[s][:, None], vir[s][None, :]] = b_vv[s]
    coulomb = np.einsum("Ppr,Pqs->pqrs", b_full, b_full)
    eri = coulomb - coulomb.transpose(0, 1, 3, 2)
    spin = np.array([0] * noa + [1] * nob + [0] * nva + [1] * nvb)
    fock = rng.normal(size=(n, n)) * 0.05
    fock = 0.5 * (fock + fock.T) * (spin[:, None] == spin[None, :])
    fock[np.arange(no), np.arange(no)] = -1.0 - rng.random(no)
    fock[np.arange(no, n), np.arange(no, n)] = 0.5 + rng.random(nv)
    occ_spin = [0] * noa + [1] * nob
    t1 = rng.normal(size=(no, nv)) * 0.05
    for i in range(no):
        other = slice(nva, nv) if occ_spin[i] == 0 else slice(0, nva)
        t1[i, other] = 0.0
    pairs = dict(i=[], j=[], U=[], eps=[], K=[], T2=[], L=[])
    t2 = np.zeros((no, no, nv, nv))
    for i in range(no):
        for j in range(i + 1, no):
            if (i, j) in drop:
                continue
            blocks = []
            for s, rows in enumerate((slice(0, nva), slice(nva, nv))):
                if s not in (occ_spin[i], occ_spin[j]):
                    continue
                dim = rows.stop - rows.start
                keep = int(rng.integers(1, dim + 1)) if truncate else dim
                q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
                block = np.zeros((nv, keep))
                block[rows] = q[:, :keep]
                blocks.append(block)
            u = np.hstack(blocks)
            m = u.shape[1]
            amp = rng.normal(size=(m, m)) * 0.05
            amp = amp - amp.T
            if spin_structured and occ_spin[i] != occ_spin[j]:
                n_alpha = blocks[0].shape[1]
                amp[:n_alpha, :n_alpha] = 0.0
                amp[n_alpha:, n_alpha:] = 0.0
            if occ_spin[i] != occ_spin[j] and layout != "sorted":
                # Columns the kernel cannot store compactly: rotated across
                # the two spin blocks, or spin-pure with beta first.
                if layout == "mixed":
                    rot, _ = np.linalg.qr(rng.normal(size=(m, m)))
                else:
                    n_alpha = blocks[0].shape[1]
                    rot = np.roll(np.eye(m), m - n_alpha, axis=1)
                u = u @ rot
                amp = rot.T @ amp @ rot
            pairs["i"].append(i)
            pairs["j"].append(j)
            pairs["U"].append(u)
            pairs["eps"].append(np.ones(m))
            pairs["K"].append(u.T @ eri[i, j, no:, no:] @ u)
            pairs["T2"].append(amp)
            pairs["L"].append(list(range(no)))
            t2[i, j] = u @ amp @ u.T
            t2[j, i] = -t2[i, j]
    o, v = slice(0, no), slice(no, n)
    r1_ref, r2_ref = so_residuals(
        fock, fock, eri, t1, t2, o, v, np.zeros((no, nv)), np.zeros((no, no, nv, nv))
    )

    def flat(x):
        return np.ascontiguousarray(x.reshape(naux, -1))

    r2, r1 = dlpno_uccsd_pno_residuals(
        flat(b_ov[0]), flat(b_ov[1]), flat(b_oo[0]), flat(b_oo[1]),
        flat(b_vv[0]), flat(b_vv[1]),
        fock[o, o], fock[o, v],
        fock[no:no + nva, no:no + nva], fock[no + nva:, no + nva:],
        pairs["i"], pairs["j"], pairs["U"], pairs["eps"], pairs["K"],
        pairs["T2"], pairs["L"], [list(range(no))] * no, t1,
    )
    own_spin = np.zeros((no, nv), dtype=bool)
    for i in range(no):
        own_spin[i, slice(0, nva) if occ_spin[i] == 0 else slice(nva, nv)] = True
    return pairs, r2, r1, r2_ref, r1_ref, own_spin


@pytest.mark.parametrize(
    "seed,shape,truncate,drop,spin_structured",
    [
        (3, (3, 2, 6, 7, 15), False, (), True),
        (3, (3, 2, 6, 7, 15), True, (), True),
        (11, (3, 2, 6, 7, 15), True, ((0, 3), (1, 2)), True),
        (5, (4, 1, 6, 7, 15), True, ((0, 4),), True),
        (8, (2, 3, 4, 9, 21), True, (), True),
        # Amplitudes that break the spin structure still satisfy the pair
        # identity; the singles then leak into the other spin block, which the
        # kernel does not carry, so only the own-spin block is compared.
        (8, (2, 3, 4, 9, 21), True, (), False),
    ],
)
@pytest.mark.parametrize("layout", ["sorted", "mixed", "beta_first"])
def test_residuals_equal_the_projected_canonical_residual(
    seed, shape, truncate, drop, spin_structured, layout
):
    """Sorted spin-pure columns take the compact factor rows; mixed or
    beta-first columns take the full rows. Both must be exact."""
    pairs, r2, r1, r2_ref, r1_ref, own_spin = _random_problem(
        seed,
        *shape,
        truncate=truncate,
        drop=drop,
        spin_structured=spin_structured,
        layout=layout,
    )
    scale = np.abs(r2_ref).max()
    for index, (i, j) in enumerate(zip(pairs["i"], pairs["j"])):
        u = pairs["U"][index]
        np.testing.assert_allclose(
            r2[index], u.T @ r2_ref[i, j] @ u, rtol=0.0, atol=1e-13 * scale
        )
    np.testing.assert_allclose(
        r1[own_spin], r1_ref[own_spin], rtol=0.0, atol=1e-13 * np.abs(r1_ref).max()
    )
    if spin_structured:
        np.testing.assert_allclose(r1, r1_ref, rtol=0.0, atol=1e-13)


# --- molecular ratchets ------------------------------------------------------


def test_full_domain_reproduces_the_dense_pilot_and_the_union_route(oh_svp):
    exact = dict(tcut_pno=0.0, tcut_pairs=0.0, tcut_mkn=0.0, n_frozen=0)
    pno = run_local_dlpno_uccsd(*oh_svp, LocalUCCSDOptions(**exact))
    union = run_local_dlpno_uccsd(
        *oh_svp, LocalUCCSDOptions(residual_domain="union", **exact)
    )
    pilot = run_dlpno_uccsd_pilot(
        *oh_svp,
        DLPNOUCCSDPilotOptions(tcut_pno=0.0, n_frozen=0, localise="boys"),
    )
    assert pno.converged and union.converged and pilot.converged
    assert pno.residual_domain == "pno"
    assert pno.e_corr == pytest.approx(union.e_corr, abs=1e-10)
    assert pno.e_corr == pytest.approx(pilot.e_corr, abs=1e-8)


@pytest.mark.parametrize(
    "fixture,overrides",
    [
        ("oh_svp", {}),
        ("oh_svp", {"compute_triples": True, "triples_mode": "t0"}),
        ("ch2_triplet_svp", {}),
        ("separated_sto3g", {"coupling_radius": 6.0}),
        ("separated_sto3g", {"tcut_pno_singles": 1e-6}),
    ],
)
def test_matches_the_union_route_at_normal_pno(request, fixture, overrides):
    system = request.getfixturevalue(fixture)
    pno = run_local_dlpno_uccsd(*system, LocalUCCSDOptions(**overrides))
    union = run_local_dlpno_uccsd(
        *system, LocalUCCSDOptions(residual_domain="union", **overrides)
    )
    assert pno.converged and union.converged
    assert pno.e_corr == pytest.approx(union.e_corr, abs=1e-9)
    assert pno.e_t == pytest.approx(union.e_t, abs=1e-11)
    assert pno.avg_pno == pytest.approx(union.avg_pno)
    assert pno.n_pairs == union.n_pairs
    assert pno.n_screened == union.n_screened


def test_pair_spaces_are_built_per_spin_block(ch2_triplet_svp):
    """At the full-domain limit a same-spin pair spans exactly its own spin's
    virtuals and an opposite-spin pair both blocks; the union route's combined
    diagonalisation also keeps the zero-occupation directions of the other
    spin for a same-spin pair."""
    molecule, basis, uhf, df = ch2_triplet_svp
    exact = dict(tcut_pno=0.0, tcut_pairs=0.0, tcut_mkn=0.0)
    pno = run_local_dlpno_uccsd(molecule, basis, uhf, df, LocalUCCSDOptions(**exact))
    union = run_local_dlpno_uccsd(
        molecule, basis, uhf, df, LocalUCCSDOptions(residual_domain="union", **exact)
    )
    n_alpha = (molecule.n_electrons() + 2) // 2
    n_beta = molecule.n_electrons() - n_alpha
    n_frozen = 1
    nva, nvb = basis.nbasis - n_alpha, basis.nbasis - n_beta
    noa = n_alpha - n_frozen
    for (p, q), count in pno.pno_per_pair.items():
        spins = {p < noa, q < noa}
        if spins == {True}:
            assert count == nva
        elif spins == {False}:
            assert count == nvb
        else:
            assert count == nva + nvb
        assert union.pno_per_pair[(p, q)] == nva + nvb
    assert pno.e_corr == pytest.approx(union.e_corr, abs=1e-10)


@pytest.mark.parametrize("n_frozen", [0, 1])
@pytest.mark.parametrize("tcut_pno", [0., 1e-5])
def test_pair_fock_boundary_preserves_frozen_and_truncated_results(
    oh_svp, monkeypatch, n_frozen, tcut_pno, record_property,
):
    """Change only the retained pair Fock solve; retain the old NumPy oracle."""
    from types import SimpleNamespace
    import weakref
    from vibeqc.dlpno import uccsd_local_solver as local

    options = LocalUCCSDOptions(
        n_frozen=n_frozen, tcut_pno=tcut_pno, tcut_pairs=0., tcut_mkn=0.,
        coupling_radius=0., compute_triples=True, triples_mode="t0",
        conv_tol_residual=1e-11, conv_tol_energy=1e-13,
    )
    shared = local.solve_dense_fock
    calls, references = [], []

    def checked(matrix, **metadata):
        assert all(ref() is None for ref in references)
        solved = shared(matrix, **metadata)
        if ":pair=" in metadata["basis_id"]:
            references.append(weakref.ref(solved))
            old_eps, old_vectors = np.linalg.eigh(matrix)
            np.testing.assert_array_equal(solved.eigenvalues, old_eps)
            np.testing.assert_array_equal(solved.eigenvectors, old_vectors)
            assert solved.residual_frame == "orthogonal"
            assert max(solved.residuals) < 1e-10
            assert solved.orthogonality_error < 1e-12
            calls.append((matrix.shape[0], metadata))
        return solved

    with monkeypatch.context() as patch:
        patch.setattr(local, "solve_dense_fock", checked)
        actual = run_local_dlpno_uccsd(*oh_svp, options)
    assert all(ref() is None for ref in references)

    def previous(matrix, **metadata):
        if ":pair=" not in metadata["basis_id"]:
            return shared(matrix, **metadata)
        eps, vectors = np.linalg.eigh(matrix)
        return SimpleNamespace(eigenvalues=eps, eigenvectors=vectors)

    with monkeypatch.context() as patch:
        patch.setattr(local, "solve_dense_fock", previous)
        before = run_local_dlpno_uccsd(*oh_svp, options)
    assert actual.converged and before.converged
    assert actual.triples_executed and abs(actual.e_t) > 1e-8
    # Shared results own their arrays; a different memory layout can change
    # the final contraction roundoff even with identical eigenpairs.
    assert actual.e_corr == pytest.approx(before.e_corr, abs=1e-14, rel=0.)
    assert actual.e_t == pytest.approx(before.e_t, abs=1e-14, rel=0.)
    assert actual.e_pno_correction == before.e_pno_correction
    assert len(actual.trace) == len(before.trace)
    for step, old_step in zip(actual.trace, before.trace):
        assert step == pytest.approx(old_step, abs=1e-14, rel=0.)
    assert actual.pno_per_pair == before.pno_per_pair
    assert actual.pao_per_pair == before.pao_per_pair
    assert actual.n_frozen == n_frozen
    assert actual.trace[-1]["r_norm"] < 1e-11
    assert abs(actual.trace[-1]["delta_e"]) < 1e-13
    nva, nvb = oh_svp[1].nbasis - 5, oh_svp[1].nbasis - 4
    dimensions = {}
    for size, metadata in calls:
        prefix = f"dlpno-uccsd:{actual.source_identity}:frozen={n_frozen}:pair="
        assert metadata["basis_id"].startswith(prefix)
        pair_text, spin = metadata["basis_id"][len(prefix):].split(":retained-virtual:")
        pair = tuple(map(int, pair_text.split(",")))
        assert metadata["spin"] == spin
        assert spin in {"alpha" if p < 5-n_frozen else "beta" for p in pair}
        assert (pair, spin) not in dimensions
        dimensions[pair, spin] = size
    assert {pair for pair, _ in dimensions} == set(actual.pno_per_pair)
    for pair, size in actual.pno_per_pair.items():
        assert size == sum(n for (key, _), n in dimensions.items() if key == pair)
    if tcut_pno:
        assert any(n < (nva if spin == "alpha" else nvb)
                   for (_, spin), n in dimensions.items())
        assert abs(actual.e_pno_correction) > 1e-8
    else:
        assert all(n == (nva if spin == "alpha" else nvb)
                   for (_, spin), n in dimensions.items())
    record_property("correlation_Eh", actual.e_corr)
    record_property("triples_Eh", actual.e_t)
    record_property("pno_correction_Eh", actual.e_pno_correction)
    record_property("previous_correlation_error_Eh", abs(actual.e_corr-before.e_corr))
    record_property("previous_triples_error_Eh", abs(actual.e_t-before.e_t))
    record_property("previous_trace_error", max(abs(step[key]-old_step[key])
                    for step, old_step in zip(actual.trace, before.trace)
                    for key in step))
    record_property("pair_fock_solves", len(calls))
    record_property("max_pair_fock_rank", max(size for size, _ in calls))
    record_property("final_iteration", actual.trace[-1])


@pytest.mark.parametrize("n_frozen", [0, 1])
@pytest.mark.parametrize("fixture,pair_cut,singles_cut,atom_cut", [
    ("oh_sto3g", 0., 0., 0.),
    ("oh_sto3g", 1e-5, 0., 0.),
    ("separated_sto3g", 0., 0., 1e-3),
    ("oh_sto3g", 1e-5, 1e-5, 0.),
    ("separated_sto3g", 0., 1e-6, 1e-3),
])
def test_union_pno_fock_preserves_frozen_calculation(
    request, monkeypatch, fixture, pair_cut, singles_cut, atom_cut,
    n_frozen, record_property,
):
    from types import SimpleNamespace
    import weakref
    from vibeqc.dlpno import uccsd_local_solver as local

    system = request.getfixturevalue(fixture)
    options = LocalUCCSDOptions(
        residual_domain="union", n_frozen=n_frozen, tcut_pno=pair_cut,
        tcut_pno_singles=singles_cut, tcut_mkn=atom_cut, tcut_pairs=0.,
        coupling_radius=0., compute_triples=True, triples_mode="t0",
        conv_tol_residual=1e-11, conv_tol_energy=1e-13,
    )
    shared = local.solve_dense_fock
    calls, references = [], []

    def checked(matrix, **metadata):
        assert all(ref() is None for ref in references)
        solved = shared(matrix, **metadata)
        if ":union-" in metadata["basis_id"]:
            references.append(weakref.ref(solved))
            eps, vectors = np.linalg.eigh(matrix)
            np.testing.assert_array_equal(solved.eigenvalues, eps)
            np.testing.assert_array_equal(solved.eigenvectors, vectors)
            assert solved.residual_frame == "orthogonal"
            assert max(solved.residuals) < 1e-10
            assert solved.orthogonality_error < 1e-12
            calls.append((matrix.shape[0], metadata))
        return solved

    with monkeypatch.context() as patch:
        patch.setattr(local, "solve_dense_fock", checked)
        actual = run_local_dlpno_uccsd(*system, options)
    assert all(ref() is None for ref in references)

    def previous(matrix, **metadata):
        if ":union-" not in metadata["basis_id"]:
            return shared(matrix, **metadata)
        eps, vectors = np.linalg.eigh(matrix)
        return SimpleNamespace(eigenvalues=eps, eigenvectors=vectors)

    with monkeypatch.context() as patch:
        patch.setattr(local, "solve_dense_fock", previous)
        before = run_local_dlpno_uccsd(*system, options)
    assert actual.converged and before.converged and actual.triples_executed
    assert actual.source_identity == before.source_identity
    assert actual.n_frozen == before.n_frozen == n_frozen
    for name in ("e_corr", "e_t", "e_pno_correction"):
        assert getattr(actual, name) == pytest.approx(getattr(before, name), abs=1e-14, rel=0.)
    for name in ("pao_per_pair", "pno_per_pair", "singles_per_occ"):
        assert getattr(actual, name) == getattr(before, name)
    assert len(actual.trace) == len(before.trace)
    for step, old_step in zip(actual.trace, before.trace):
        assert step == pytest.approx(old_step, abs=1e-14, rel=0.)
    assert actual.trace[-1]["r_norm"] < 1e-11
    assert abs(actual.trace[-1]["delta_e"]) < 1e-13
    pairs, singles, full_singles = {}, {}, []
    for rank, metadata in calls:
        prefix = f"dlpno-uccsd:{actual.source_identity}:frozen={n_frozen}:union-"
        assert metadata["basis_id"].startswith(prefix)
        label = metadata["basis_id"][len(prefix):]
        if label == "full-singles:alpha-beta":
            assert metadata["spin"] == "alpha-beta"
            full_singles.append(rank)
        elif label.startswith("pair="):
            text, spin = label[len("pair="):].split(":retained-virtual:")
            key = tuple(map(int, text.split(",")))
            assert key not in pairs and spin == metadata["spin"] == "alpha-beta"
            pairs[key] = rank
        else:
            text, spin = label[len("occupied="):].split(":retained-singles:")
            key = int(text)
            noa = (system[0].n_electrons()+1)//2 - n_frozen
            expected = ("alpha" if key < noa else "beta") if atom_cut else "alpha-beta"
            assert key not in singles and spin == metadata["spin"] == expected
            singles[key] = rank
    assert pairs == actual.pno_per_pair
    # Default full singles share one solve, while retained singles keep
    # their independently selected occupied-specific spaces.
    assert singles == (actual.singles_per_occ if singles_cut else {})
    nv = 2*system[1].nbasis-system[0].n_electrons()
    assert full_singles == ([] if singles_cut else [nv])
    if not singles_cut:
        assert set(actual.singles_per_occ.values()) == {nv}
    if pair_cut:
        assert any(rank < nv for rank in pairs.values())
    if singles_cut:
        assert any(rank < nv for rank in singles.values())
    if atom_cut:
        assert actual.avg_pao < 0.8*nv
    if pair_cut == singles_cut == atom_cut == 0.:
        canonical = run_dlpno_uccsd_pilot(*system, DLPNOUCCSDPilotOptions(
            n_frozen=n_frozen, localise="none", tcut_pno=0.,
            conv_tol_residual=1e-11, conv_tol_energy=1e-13,
        ))
        assert canonical.converged
        raw_error = abs(actual.trace[-1]["e_corr"]-canonical.e_corr)
        assert raw_error < 1e-8
        record_property("canonical_raw_error_Eh", raw_error)
    record_property("raw_correlation_Eh", actual.trace[-1]["e_corr"])
    record_property("pno_correction_Eh", actual.e_pno_correction)
    record_property("triples_Eh", actual.e_t)
    record_property("previous_correlation_error_Eh", abs(actual.e_corr-before.e_corr))
    record_property("previous_triples_error_Eh", abs(actual.e_t-before.e_t))
    record_property("previous_trace_error", max(abs(a[k]-b[k])
                    for a, b in zip(actual.trace, before.trace) for k in a))
    record_property("full_singles_fock_ranks", full_singles)
    record_property("pair_fock_ranks", list(pairs.values()))
    record_property("singles_fock_ranks", list(singles.values()))
    record_property("final_iteration", actual.trace[-1])


@pytest.mark.parametrize("threshold,rank", [(0., 4), (0.2, 2), (10., 1)])
@pytest.mark.parametrize("spin", ["beta", "alpha-beta"])
def test_named_singles_fock_preserves_density_selection(threshold, rank, spin):
    from vibeqc.dlpno.uccsd_local_solver import _domain_from_density

    # Strided caller data, a degenerate Fock spectrum and the singleton
    # fallback must preserve the density's selected subspace.
    density = np.diag([0.1, 0., 0.4, 0., 0.8, 0., 0., 0.])[::2, ::2]
    fock = np.diag([2., 0., 1., 0., 1., 0., 3., 0.])[::2, ::2]
    original_density, original_fock = density.copy(), fock.copy()
    before, old_eps = _domain_from_density(density, fock, threshold)
    after, eps = _domain_from_density(
        density, fock, threshold, spin=spin,
        fock_basis_id=f"source:frozen=1:occupied=0:retained-singles:{spin}",
    )
    assert after.shape == (4, rank)
    np.testing.assert_array_equal(density, original_density)
    np.testing.assert_array_equal(fock, original_fock)
    np.testing.assert_array_equal(eps, old_eps)
    np.testing.assert_allclose(after @ after.T, before @ before.T, atol=1e-14, rtol=0.)
    np.testing.assert_allclose(after.T @ after, np.eye(rank), atol=1e-14, rtol=0.)
    np.testing.assert_allclose(after.T @ fock @ after, np.diag(eps), atol=1e-14, rtol=0.)


@pytest.mark.parametrize("n_frozen", [0, 1])
@pytest.mark.parametrize("fixture,singles_cut,atom_cut", [
    ("oh_svp", 1e-30, 0.),
    ("oh_svp", 1e-5, 0.),
    ("separated_sto3g", 1e-6, 1e-3),
])
def test_singles_fock_boundary_preserves_frozen_calculation(
    request, monkeypatch, fixture, singles_cut, atom_cut, n_frozen, record_property,
):
    from types import SimpleNamespace
    import weakref
    from vibeqc.dlpno import uccsd_local_solver as local

    system = request.getfixturevalue(fixture)
    molecule, basis, uhf, df = system
    options = LocalUCCSDOptions(
        n_frozen=n_frozen, tcut_pno=0., tcut_pno_singles=singles_cut,
        tcut_pairs=0., tcut_mkn=atom_cut, coupling_radius=0.,
        compute_triples=True, triples_mode="t0",
        conv_tol_residual=1e-11, conv_tol_energy=1e-13,
    )
    shared = local.solve_dense_fock
    calls, references = [], []

    def checked(matrix, **metadata):
        assert all(ref() is None for ref in references)
        solved = shared(matrix, **metadata)
        if ":retained-singles:" in metadata["basis_id"]:
            references.append(weakref.ref(solved))
            old_eps, old_vectors = np.linalg.eigh(matrix)
            np.testing.assert_array_equal(solved.eigenvalues, old_eps)
            np.testing.assert_array_equal(solved.eigenvectors, old_vectors)
            assert solved.residual_frame == "orthogonal"
            assert max(solved.residuals) < 1e-10
            assert solved.orthogonality_error < 1e-12
            calls.append((matrix.shape[0], metadata))
        return solved

    with monkeypatch.context() as patch:
        patch.setattr(local, "solve_dense_fock", checked)
        actual = run_local_dlpno_uccsd(*system, options)
    assert all(ref() is None for ref in references)

    def previous(matrix, **metadata):
        if ":retained-singles:" not in metadata["basis_id"]:
            return shared(matrix, **metadata)
        eps, vectors = np.linalg.eigh(matrix)
        return SimpleNamespace(eigenvalues=eps, eigenvectors=vectors)

    with monkeypatch.context() as patch:
        patch.setattr(local, "solve_dense_fock", previous)
        before = run_local_dlpno_uccsd(*system, options)
    assert actual.converged and before.converged and actual.triples_executed
    assert actual.source_identity == before.source_identity
    assert actual.n_frozen == before.n_frozen == n_frozen
    assert actual.e_corr == pytest.approx(before.e_corr, abs=1e-14, rel=0.)
    assert actual.e_t == pytest.approx(before.e_t, abs=1e-14, rel=0.)
    assert actual.e_pno_correction == before.e_pno_correction
    assert actual.singles_per_occ == before.singles_per_occ
    assert actual.pno_per_pair == before.pno_per_pair
    assert actual.pao_per_pair == before.pao_per_pair
    assert len(actual.trace) == len(before.trace)
    for step, old_step in zip(actual.trace, before.trace):
        assert step == pytest.approx(old_step, abs=1e-14, rel=0.)
    assert actual.trace[-1]["r_norm"] < 1e-11
    assert abs(actual.trace[-1]["delta_e"]) < 1e-13
    nalpha = (molecule.n_electrons() + molecule.multiplicity - 1) // 2
    nbeta = molecule.n_electrons() - nalpha
    assert len(calls) == nalpha + nbeta - 2*n_frozen
    for p, (rank, metadata) in enumerate(calls):
        spin = "alpha" if p < nalpha-n_frozen else "beta"
        assert metadata == dict(spin=spin, basis_id=(
            f"dlpno-uccsd:{actual.source_identity}:frozen={n_frozen}:"
            f"occupied={p}:retained-singles:{spin}"))
        assert rank == actual.singles_per_occ[p]
    full_ranks = [basis.nbasis - (nalpha if p < nalpha-n_frozen else nbeta)
                  for p in actual.singles_per_occ]
    if singles_cut == 1e-30:
        assert list(actual.singles_per_occ.values()) == full_ranks
        canonical = run_dlpno_uccsd_pilot(*system, DLPNOUCCSDPilotOptions(
            n_frozen=n_frozen, localise="none", tcut_pno=0.,
            conv_tol_residual=1e-11, conv_tol_energy=1e-13,
        ))
        assert canonical.converged
        raw_error = abs(actual.trace[-1]["e_corr"] - canonical.e_corr)
        assert raw_error < 1e-8
        record_property("canonical_raw_error_Eh", raw_error)
    else:
        assert any(rank < full for (rank, _), full in zip(calls, full_ranks))
    record_property("raw_correlation_Eh", actual.trace[-1]["e_corr"])
    record_property("pno_correction_Eh", actual.e_pno_correction)
    record_property("triples_Eh", actual.e_t)
    record_property("previous_correlation_error_Eh", abs(actual.e_corr-before.e_corr))
    record_property("previous_triples_error_Eh", abs(actual.e_t-before.e_t))
    record_property("previous_trace_error", max(abs(step[key]-old_step[key])
                    for step, old_step in zip(actual.trace, before.trace) for key in step))
    record_property("singles_ranks", [rank for rank, _ in calls])
    record_property("final_iteration", actual.trace[-1])


def test_default_singles_do_not_request_a_retained_fock_solve(oh_sto3g, monkeypatch):
    from vibeqc.dlpno import uccsd_local_solver as local

    shared = local.solve_dense_fock

    def checked(matrix, **metadata):
        assert ":retained-singles:" not in metadata["basis_id"]
        return shared(matrix, **metadata)

    monkeypatch.setattr(local, "solve_dense_fock", checked)
    result = run_local_dlpno_uccsd(*oh_sto3g, LocalUCCSDOptions())
    assert result.converged


@pytest.mark.parametrize("spin", ["alpha", "beta"])
@pytest.mark.parametrize("indices,rank", [
    ([0, 1, 2, 3, 4], 2), ([0, 2, 4], 0), ([3], 1), ([], 0),
])
def test_named_pao_fock_preserves_metric_rank(indices, rank, spin, monkeypatch):
    from vibeqc.dlpno import pao

    cutoff = 1e-8
    # The strict PAO cutoff excludes an eigenvalue equal to the threshold.
    # F=2S gives a degenerate retained Fock; the occupied AO is projected out.
    s = np.diag([1., 1., cutoff, cutoff*(1+1e-4), cutoff*(1-1e-4)])
    q = np.diag([0., 1., 1., 1., 1.])
    arrays = []
    for value in (2*s, s, q):
        storage = np.zeros((10, 10))
        storage[::2, ::2] = value
        arrays.append(storage[::2, ::2])
    f, s, q = arrays
    before_arrays = [x.copy() for x in arrays]
    old_v, old_eps = pao.semicanonical_pao_basis(f, s, q, indices, cutoff)
    shared = pao.solve_dense_fock
    calls = []

    def checked(matrix, **metadata):
        assert matrix.shape == (rank, rank)
        assert metadata == dict(spin=spin, basis_id="retained-pao-frame")
        calls.append(matrix.shape)
        return shared(matrix, **metadata)

    monkeypatch.setattr(pao, "solve_dense_fock", checked)
    v, eps = pao.semicanonical_pao_basis(
        f, s, q, indices, cutoff, fock_basis_id="retained-pao-frame", spin=spin,
    )
    assert len(calls) == int(rank > 0)
    assert v.shape == (5, rank) and eps.shape == (rank,)
    for original, after in zip(before_arrays, arrays):
        np.testing.assert_array_equal(after, original)
    np.testing.assert_array_equal(eps, old_eps)
    np.testing.assert_allclose(v @ v.T @ s, old_v @ old_v.T @ s, atol=1e-13, rtol=0.)
    np.testing.assert_allclose(v.T @ s @ v, np.eye(rank), atol=1e-13, rtol=0.)
    np.testing.assert_allclose(v.T @ f @ v, np.diag(eps), atol=1e-13, rtol=0.)
    np.testing.assert_array_equal(q @ v, v)


@pytest.mark.parametrize("n_frozen", [0, 1])
@pytest.mark.parametrize("route,fixture,doi_cut", [
    ("pno", "oh_svp", 0.), ("pno", "separated_sto3g", 1e-2),
    ("union", "oh_sto3g", 0.), ("union", "separated_sto3g", 1e-2),
])
def test_cached_pao_fock_preserves_frozen_calculation(
    request, monkeypatch, fixture, doi_cut, n_frozen, route, record_property,
):
    from types import SimpleNamespace
    import weakref
    from vibeqc.dlpno import pao

    system = request.getfixturevalue(fixture)
    molecule, basis, uhf, df = system
    options = LocalUCCSDOptions(
        residual_domain=route, n_frozen=n_frozen, tcut_pno=0., tcut_pairs=0., tcut_mkn=1e-3,
        tcut_do=doi_cut, tcut_pno_singles=1e-30 if doi_cut == 0. else 1e-6,
        coupling_radius=0., compute_triples=True, triples_mode="t0",
        conv_tol_residual=1e-11, conv_tol_energy=1e-13,
    )
    shared = pao.solve_dense_fock
    calls, references = [], []

    def checked(matrix, **metadata):
        assert all(ref() is None for ref in references)
        solved = shared(matrix, **metadata)
        references.append(weakref.ref(solved))
        old_eps, old_vectors = np.linalg.eigh(matrix)
        np.testing.assert_array_equal(solved.eigenvalues, old_eps)
        np.testing.assert_array_equal(solved.eigenvectors, old_vectors)
        assert solved.residual_frame == "orthogonal"
        assert max(solved.residuals) < 1e-10
        assert solved.orthogonality_error < 1e-12
        calls.append((matrix.shape[0], metadata))
        return solved

    with monkeypatch.context() as patch:
        patch.setattr(pao, "solve_dense_fock", checked)
        actual = run_local_dlpno_uccsd(*system, options)
    assert all(ref() is None for ref in references)

    def previous(matrix, **metadata):
        eps, vectors = np.linalg.eigh(matrix)
        return SimpleNamespace(eigenvalues=eps, eigenvectors=vectors)

    with monkeypatch.context() as patch:
        patch.setattr(pao, "solve_dense_fock", previous)
        before = run_local_dlpno_uccsd(*system, options)
    assert actual.converged and before.converged and actual.triples_executed
    assert actual.source_identity == before.source_identity
    assert actual.n_frozen == before.n_frozen == n_frozen
    assert actual.e_corr == pytest.approx(before.e_corr, abs=1e-14, rel=0.)
    assert actual.e_t == pytest.approx(before.e_t, abs=1e-14, rel=0.)
    assert actual.e_pno_correction == pytest.approx(before.e_pno_correction, abs=1e-14, rel=0.)
    assert actual.pao_per_pair == before.pao_per_pair
    assert actual.pno_per_pair == before.pno_per_pair
    assert actual.singles_per_occ == before.singles_per_occ
    assert len(actual.trace) == len(before.trace)
    for step, old_step in zip(actual.trace, before.trace):
        assert step == pytest.approx(old_step, abs=1e-14, rel=0.)
    assert actual.trace[-1]["r_norm"] < 1e-11
    assert abs(actual.trace[-1]["delta_e"]) < 1e-13
    seen = set()
    for rank, metadata in calls:
        prefix = f"dlpno-uccsd:{actual.source_identity}:frozen={n_frozen}:pao-atoms="
        assert metadata["basis_id"].startswith(prefix)
        atoms_text, spin = metadata["basis_id"][len(prefix):].split(":")
        atoms = tuple(map(int, atoms_text.split(",")))
        assert metadata["spin"] == spin and spin in ("alpha", "beta")
        assert (atoms, spin) not in seen  # Cache reused by pairs and singles.
        seen.add((atoms, spin))
    assert len(calls) < actual.n_pairs
    if doi_cut == 0.:
        assert seen == {((0, 1), "alpha"), ((0, 1), "beta")}
        assert sorted(rank for rank, _ in calls) == [basis.nbasis-5, basis.nbasis-4]
        canonical = run_dlpno_uccsd_pilot(*system, DLPNOUCCSDPilotOptions(
            n_frozen=n_frozen, localise="none", tcut_pno=0.,
            conv_tol_residual=1e-11, conv_tol_energy=1e-13,
        ))
        assert canonical.converged
        raw_error = abs(actual.trace[-1]["e_corr"] - canonical.e_corr)
        assert raw_error < 1e-8
        record_property("canonical_raw_error_Eh", raw_error)
    else:
        assert any(len(atoms) < 4 for atoms, _ in seen)
        assert actual.avg_pao < 0.8*(2*basis.nbasis-molecule.n_electrons())
    record_property("raw_correlation_Eh", actual.trace[-1]["e_corr"])
    record_property("pno_correction_Eh", actual.e_pno_correction)
    record_property("triples_Eh", actual.e_t)
    record_property("previous_correlation_error_Eh", abs(actual.e_corr-before.e_corr))
    record_property("previous_triples_error_Eh", abs(actual.e_t-before.e_t))
    record_property("previous_trace_error", max(abs(step[key]-old_step[key])
                    for step, old_step in zip(actual.trace, before.trace) for key in step))
    record_property("pao_ranks", [rank for rank, _ in calls])
    record_property("final_iteration", actual.trace[-1])


@pytest.mark.parametrize("route", ["pno", "union"])
@pytest.mark.parametrize("n_frozen", [0, 1])
@pytest.mark.parametrize("tcut_tno", [0., 1e-12])
def test_rotated_t1_tno_fock_preserves_frozen_calculation(
    oh_sto3g, monkeypatch, route, n_frozen, tcut_tno, record_property,
):
    from types import SimpleNamespace
    import weakref
    from vibeqc.dlpno import uccsd_local_solver as local

    options = LocalUCCSDOptions(
        residual_domain=route, n_frozen=n_frozen, localise="boys",
        tcut_pno=0., tcut_pairs=0., tcut_mkn=0., tcut_pno_singles=0.,
        coupling_radius=0., compute_triples=True, triples_mode="t1",
        tcut_tno=tcut_tno, conv_tol_residual=1e-11, conv_tol_energy=1e-13,
    )
    shared = local.solve_dense_fock
    calls, references = [], []

    def checked(matrix, **metadata):
        assert all(ref() is None for ref in references)
        solved = shared(matrix, **metadata)
        if ":rotated-t1-triple=" in metadata["basis_id"]:
            references.append(weakref.ref(solved))
            eps, vectors = np.linalg.eigh(matrix)
            np.testing.assert_array_equal(solved.eigenvalues, eps)
            np.testing.assert_array_equal(solved.eigenvectors, vectors)
            assert solved.residual_frame == "orthogonal"
            assert max(solved.residuals) < 1e-10
            assert solved.orthogonality_error < 1e-12
            calls.append((matrix.shape[0], metadata))
        return solved

    with monkeypatch.context() as patch:
        patch.setattr(local, "solve_dense_fock", checked)
        actual = run_local_dlpno_uccsd(*oh_sto3g, options)
    assert all(ref() is None for ref in references)

    def previous(matrix, **metadata):
        if ":rotated-t1-triple=" not in metadata["basis_id"]:
            return shared(matrix, **metadata)
        eps, vectors = np.linalg.eigh(matrix)
        return SimpleNamespace(eigenvalues=eps, eigenvectors=vectors)

    with monkeypatch.context() as patch:
        patch.setattr(local, "solve_dense_fock", previous)
        before = run_local_dlpno_uccsd(*oh_sto3g, options)
    assert actual.converged and before.converged and actual.triples_executed
    assert actual.source_identity == before.source_identity
    assert actual.n_frozen == before.n_frozen == n_frozen
    assert actual.e_corr == before.e_corr and actual.trace == before.trace
    assert actual.e_pno_correction == before.e_pno_correction
    assert actual.e_t == pytest.approx(before.e_t, abs=1e-14, rel=0.)
    for name in ("pno_per_pair", "pao_per_pair", "singles_per_occ", "tno_per_triple",
                 "n_triples", "n_screened_triples", "n_degenerate_tno_triples"):
        assert getattr(actual, name) == getattr(before, name)
    assert actual.trace[-1]["r_norm"] < 1e-11
    assert abs(actual.trace[-1]["delta_e"]) < 1e-13
    seen = {}
    prefix = f"dlpno-uccsd:{actual.source_identity}:frozen={n_frozen}:rotated-t1-triple="
    for rank, metadata in calls:
        assert metadata["basis_id"].startswith(prefix)
        text, spin = metadata["basis_id"][len(prefix):].split(":retained-tno:")
        triple = tuple(map(int, text.split(",")))
        assert len(triple) == 3 and triple[0] < triple[1] < triple[2]
        assert triple not in seen and spin == metadata["spin"] == "alpha-beta"
        seen[triple] = rank
    assert seen == actual.tno_per_triple
    if tcut_tno == 0.:
        assert set(seen.values()) == {3} and actual.n_degenerate_tno_triples == 0
        canonical = run_dlpno_uccsd_pilot(*oh_sto3g, DLPNOUCCSDPilotOptions(
            n_frozen=n_frozen, localise="none", tcut_pno=0., compute_triples=True,
            conv_tol_residual=1e-11, conv_tol_energy=1e-13,
        ))
        assert canonical.converged and abs(actual.e_t) > 1e-8
        raw_error = abs(actual.trace[-1]["e_corr"]-canonical.e_corr)
        triples_error = abs(actual.e_t-canonical.e_t)
        assert raw_error < 1e-8 and triples_error < 1e-11
        record_property("canonical_raw_error_Eh", raw_error)
        record_property("canonical_triples_error_Eh", triples_error)
    else:
        # Minimal-basis truncation collapses triples; this is a preservation
        # check, not an accuracy calibration for a useful TNO cutoff.
        assert actual.avg_tno < 3 and actual.n_degenerate_tno_triples > 0
    record_property("raw_correlation_Eh", actual.trace[-1]["e_corr"])
    record_property("pno_correction_Eh", actual.e_pno_correction)
    record_property("triples_Eh", actual.e_t)
    record_property("previous_triples_error_Eh", abs(actual.e_t-before.e_t))
    record_property("tno_fock_solves", len(calls))
    record_property("tno_rank_range", [min(seen.values()), max(seen.values())])
    record_property("degenerate_triples", actual.n_degenerate_tno_triples)
    record_property("final_iteration", actual.trace[-1])


def test_zero_beta_reference_is_exact_against_canonical_uccsd():
    from vibeqc.cc import run_uccsd

    atoms = [(1, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.6]), (1, [0.0, 0.0, 3.2])]
    molecule, basis, uhf, df = _system(atoms, 4, "def2-svp")
    canonical = run_uccsd(molecule, basis, uhf)
    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(
            localise="none", tcut_pno=0.0, tcut_pairs=0.0, tcut_mkn=0.0
        ),
    )
    assert canonical.e_ccsd_correlation < -1e-4
    assert local.converged
    assert local.e_corr == pytest.approx(canonical.e_ccsd_correlation, abs=1e-9)


def test_one_electron_reference_has_zero_correlation():
    molecule, basis, uhf, df = _system([(1, [0.0, 0.0, 0.0])], 2, "def2-svp")
    result = run_local_dlpno_uccsd(molecule, basis, uhf, df, LocalUCCSDOptions())
    assert result.e_corr == 0.0
    assert result.n_pairs == 0


def test_route_bookkeeping(oh_svp):
    molecule, basis, uhf, df = oh_svp
    result = run_local_dlpno_uccsd(molecule, basis, uhf, df, LocalUCCSDOptions())
    n_occ = sum(1 for _ in result.singles_per_occ)
    assert result.residual_domain == "pno"
    assert result.n_residual_groups == result.n_pairs + n_occ
    assert result.static_factor_bytes > 0
    assert result.setup_seconds >= 0.0 and result.iteration_seconds >= 0.0
    assert len(result.trace) == result.n_iter
    assert result.trace[-1]["e_corr"] == pytest.approx(
        result.e_corr - result.e_pno_correction, abs=1e-12
    )
    # Full-space singles span the virtuals of the occupied's own spin only.
    n_alpha = (molecule.n_electrons() + 1) // 2
    assert set(result.singles_per_occ.values()) <= {
        basis.nbasis - n_alpha,
        basis.nbasis - (molecule.n_electrons() - n_alpha),
    }


def test_route_selects_its_own_basis_function_guard(oh_sto3g):
    molecule, basis, uhf, df = oh_sto3g
    with pytest.raises(ValueError, match=r"max_nbf=3 guard of the 'pno' route"):
        run_local_dlpno_uccsd(
            molecule, basis, uhf, df, LocalUCCSDOptions(max_nbf=3)
        )
    with pytest.raises(ValueError, match=r"max_nbf=3 guard of the 'union' route"):
        run_local_dlpno_uccsd(
            molecule,
            basis,
            uhf,
            df,
            LocalUCCSDOptions(max_nbf=3, residual_domain="union"),
        )


def test_rejects_unknown_residual_domain(oh_sto3g):
    with pytest.raises(ValueError, match="unknown residual_domain"):
        run_local_dlpno_uccsd(
            *oh_sto3g, LocalUCCSDOptions(residual_domain="extended")
        )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"domain_criterion": "mulliken", "tcut_mkn": 2.0}, "empty PAO atom domain"),
        ({"lindep": 1e5}, "empty PAO virtual domain"),
        ({"lindep": 1e5, "tcut_pairs": 1.0, "tcut_pno_singles": 1e-6},
         "empty singles PAO virtual domain"),
        (
            {"domain_criterion": "mulliken", "tcut_mkn": 2.0, "tcut_pairs": 1.0,
             "tcut_pno_singles": 1e-6},
            "empty singles PAO atom domain",
        ),
    ],
)
@pytest.mark.parametrize("route", ["pno", "union"])
def test_empty_domains_fail_closed(separated_sto3g, overrides, message, route):
    with pytest.raises(ValueError, match=message):
        run_local_dlpno_uccsd(
            *separated_sto3g, LocalUCCSDOptions(residual_domain=route, **overrides)
        )


# --- triples -----------------------------------------------------------------


def test_native_t0_hco_pair_spaces_are_finite_and_basis_invariant():
    """A captured rank-deficient union must survive the portable Eigen build.

    The beta PNOs of HCO/cc-pVDZ NormalPNO, pairs (6,9), (6,10), (9,10),
    triggered non-finite BDCSVD left vectors in Eigen 3.4. Keep those spaces
    exactly; synthetic factors and amplitudes avoid an SCF-dependent fixture.
    """
    from vibeqc._vibeqc_core import dlpno_uccsd_local_triples

    with np.load(Path(__file__).parent / "data/dlpno_hco_beta_pair_spaces.npz") as data:
        spaces = [
            np.vstack((np.zeros((25, data[f"pair_{q}"].shape[1])), data[f"pair_{q}"]))
            for q in range(3)
        ]
    rng = np.random.default_rng(190)
    factors = []
    for counts in ((1, 3), (25, 26)):
        factors.append(
            [
                np.ascontiguousarray((block + block.transpose(0, 2, 1)).reshape(2, -1))
                for block in (rng.normal(size=(2, n, n)) * 0.1 for n in counts)
            ]
        )
    ov = [rng.normal(size=(2, o * v)) * 0.1 for o, v in ((1, 25), (3, 26))]
    amplitudes = []
    for space in spaces:
        t = rng.normal(size=(space.shape[1], space.shape[1])) * 0.01
        amplitudes.append(t - t.T)
    energies = []
    for reverse in (False, True):
        out = dlpno_uccsd_local_triples(
            *ov,
            *factors[0],
            *factors[1],
            -np.ones(4),
            np.diag(1.0 + 0.1 * np.arange(25)),
            np.diag(1.0 + 0.1 * np.arange(26)),
            [1, 1, 2],
            [2, 3, 3],
            [u[:, ::-1].copy() for u in spaces] if reverse else spaces,
            [t[::-1, ::-1].copy() for t in amplitudes] if reverse else amplitudes,
            np.zeros((4, 51)),
            [[1, 2, 3]],
        )
        assert out["n_tno"] == [26]
        assert np.isfinite(out["e_t"])
        assert np.all(np.isfinite(out["e_triple"]))
        energies.append(out["e_t"])
    assert energies[0] == pytest.approx(energies[1], abs=1e-12)


@pytest.mark.parametrize("bad_energy", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("per_triple", [True, False])
def test_native_t0_nonfinite_energy_is_not_a_converged_result(
    oh_sto3g, monkeypatch, bad_energy, per_triple
):
    """A converged CCSD step must not conceal a failed native correction."""
    import vibeqc._vibeqc_core as core

    def broken_triples(*args, **kwargs):
        triples = args[14]
        assert triples and tuple(triples[0]) == (0, 1, 2)
        contributions = [0.0] * len(triples)
        if per_triple:
            contributions[0] = bad_energy
        return {
            "e_t": 0.0 if per_triple else bad_energy,
            "e_triple": contributions,
            "n_tno": [3] * len(triples),
        }

    monkeypatch.setattr(core, "dlpno_uccsd_local_triples", broken_triples)
    message = (
        r"non-finite energy for occupied triple \(0, 1, 2\)"
        if per_triple
        else "non-finite total correction"
    )
    with pytest.raises(RuntimeError, match=message):
        run_local_dlpno_uccsd(
            *oh_sto3g,
            LocalUCCSDOptions(
                localise="none",
                tcut_pno=0.0,
                tcut_pairs=0.0,
                tcut_mkn=0.0,
                coupling_radius=0.0,
                n_frozen=0,
                compute_triples=True,
                triples_mode="t0",
            ),
        )


def test_full_domain_t0_reproduces_the_dense_pilot(oh_sto3g):
    """The compiled local (T0) is the dense spin-orbital (T) at full domains."""
    exact = dict(
        localise="none",
        tcut_pno=0.0,
        tcut_pairs=0.0,
        tcut_mkn=0.0,
        tcut_tno=0.0,
        n_frozen=0,
        compute_triples=True,
        triples_mode="t0",
    )
    local = run_local_dlpno_uccsd(*oh_sto3g, LocalUCCSDOptions(**exact))
    pilot = run_dlpno_uccsd_pilot(
        *oh_sto3g,
        DLPNOUCCSDPilotOptions(
            localise="none", tcut_pno=0.0, n_frozen=0, compute_triples=True
        ),
    )
    assert local.converged and pilot.converged and local.triples_executed
    assert local.e_t == pytest.approx(pilot.e_t, abs=1e-11)
    # Spin-pure pair spaces: an all-alpha triple of OH/STO-3G sees the single
    # alpha virtual only, so it cannot host a triple excitation.
    assert local.n_degenerate_tno_triples > 0


@pytest.mark.parametrize(
    "fixture,overrides",
    [
        ("oh_svp", {"tcut_tno": 1e-9}),
        ("ch2_triplet_svp", {"tcut_tno": 1e-9}),
        # Coupling-radius and weak-pair screening both remove triples here.
        ("separated_svp", {"coupling_radius": 6.0, "tcut_tno": 1e-9}),
        ("separated_svp", {"tcut_pairs": 1e-5}),
    ],
)
def test_compiled_t0_matches_the_union_route_triples(request, fixture, overrides):
    system = request.getfixturevalue(fixture)
    common = dict(compute_triples=True, triples_mode="t0", **overrides)
    pno = run_local_dlpno_uccsd(*system, LocalUCCSDOptions(**common))
    union = run_local_dlpno_uccsd(
        *system, LocalUCCSDOptions(residual_domain="union", **common)
    )
    assert pno.triples_executed and union.triples_executed
    assert pno.n_triples == union.n_triples
    assert pno.n_screened_triples == union.n_screened_triples
    assert pno.e_t == pytest.approx(union.e_t, abs=1e-12)
    if "separated" in fixture:
        assert pno.n_screened_triples > 0


def test_dense_t1_modes_still_run_on_the_compiled_ccsd(oh_sto3g):
    """(T1) reads the combined factors, built on demand for this route."""
    common = dict(
        tcut_pairs=0.0, coupling_radius=0.0, compute_triples=True, triples_mode="t1"
    )
    pno = run_local_dlpno_uccsd(*oh_sto3g, LocalUCCSDOptions(**common))
    union = run_local_dlpno_uccsd(
        *oh_sto3g, LocalUCCSDOptions(residual_domain="union", **common)
    )
    assert pno.triples_executed
    assert pno.e_t == pytest.approx(union.e_t, abs=1e-11)


def test_phase_seconds_cover_the_route(oh_svp):
    import time

    molecule, basis, uhf, df = oh_svp
    t0 = time.perf_counter()
    result = run_local_dlpno_uccsd(
        molecule, basis, uhf, df, LocalUCCSDOptions(compute_triples=True)
    )
    wall = time.perf_counter() - t0
    assert result.triples_executed
    assert list(result.phase_seconds) == [
        "reference", "localise", "domains", "factors", "pairs", "singles",
        "amplitudes", "triples",
    ]
    assert all(v >= 0.0 for v in result.phase_seconds.values())
    assert sum(result.phase_seconds.values()) <= wall


def test_spin_orbital_pair_scale_scales_the_screening_threshold(oh_svp):
    molecule, basis, uhf, df = oh_svp
    scaled = run_local_dlpno_uccsd(
        molecule, basis, uhf, df,
        LocalUCCSDOptions(tcut_pairs=4e-2, tcut_pairs_spin_scale=0.25),
    )
    plain = run_local_dlpno_uccsd(
        molecule, basis, uhf, df,
        LocalUCCSDOptions(tcut_pairs=1e-2, tcut_pairs_spin_scale=1.0),
    )
    looser = run_local_dlpno_uccsd(
        molecule, basis, uhf, df,
        LocalUCCSDOptions(tcut_pairs=4e-2, tcut_pairs_spin_scale=1.0),
    )
    assert scaled.n_screened == plain.n_screened > 0
    assert scaled.e_corr == pytest.approx(plain.e_corr, abs=1e-12)
    assert looser.n_screened > scaled.n_screened
    with pytest.raises(ValueError, match="tcut_pairs_spin_scale"):
        run_local_dlpno_uccsd(
            molecule, basis, uhf, df, LocalUCCSDOptions(tcut_pairs_spin_scale=0.0)
        )


def test_run_job_discloses_the_pair_scale(tmp_path, monkeypatch):
    from vibeqc import run_job

    monkeypatch.chdir(tmp_path)
    molecule = Molecule(
        [Atom(z, [c * ANGSTROM_TO_BOHR for c in position]) for z, position in OH],
        charge=0,
        multiplicity=2,
    )
    assert LocalUCCSDOptions().tcut_pairs_spin_scale == 0.25
    run_job(molecule, basis="def2-svp", method="dlpno-ccsd", output="default")
    run_job(
        molecule, basis="def2-svp", method="dlpno-ccsd", output="legacy",
        dlpno_ccsd_options=LocalUCCSDOptions(tcut_pairs_spin_scale=1.0),
    )
    assert "Pair cut, spin orbital   = 0.25 x tcut_pairs" in (
        tmp_path / "default.out"
    ).read_text()
    assert "Pair cut, spin orbital   = 1 x tcut_pairs" in (
        tmp_path / "legacy.out"
    ).read_text()
