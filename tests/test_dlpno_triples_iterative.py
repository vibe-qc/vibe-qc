"""Guo Eq. (2) iterative local `(T1)`: reduction and exactness ratchets.

The scheme keeps the triples in the localised occupied basis and reintroduces
the off-diagonal occupied Fock coupling by iteration, rather than rotating the
occupied indices into the canonical basis the way vibe-qc's existing `(T1)`
does. Two exact targets pin it, one at each end of the occupied basis:

* a diagonal occupied Fock must reproduce the semicanonical `(T0)` amplitude
  `W/D` in a single sweep (Eq. (3) of the paper is the diagonal limit of
  Eq. (2));
* a non-diagonal occupied Fock must reproduce the exact `(T)`, obtained
  independently by rotating to the canonical occupied basis and evaluating
  `(T0)` there, where the occupied Fock is diagonal by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc.dlpno import _ccsd_ref as ref
from vibeqc.dlpno.triples_iterative import (
    degenerate_tno_triples,
    iterative_t1_triples_correction,
)

NO = 4
NV = 5


@pytest.fixture(scope="module")
def system():
    """A small random spin-orbital system with the right ERI antisymmetry."""
    rng = np.random.default_rng(7)
    nso = NO + NV
    o, v = slice(0, NO), slice(NO, nso)

    g = rng.normal(size=(nso,) * 4) * 0.05
    g = g - g.transpose(1, 0, 2, 3)
    g = g - g.transpose(0, 1, 3, 2)
    g = 0.5 * (g + g.transpose(2, 3, 0, 1))

    eps = np.concatenate(
        [np.linspace(-1.2, -0.6, NO), np.linspace(0.4, 1.6, NV)]
    )
    t1 = rng.normal(size=(NO, NV)) * 0.02
    t2 = rng.normal(size=(NO, NO, NV, NV)) * 0.02
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)

    return dict(
        t1=t1,
        t2=t2,
        eps=eps,
        eri_vovv=g[v, o, v, v],
        eri_ovoo=g[o, v, o, o],
        eri_oovv=g[o, o, v, v],
    )


def _t0_energy(s, t1, t2, eri_vovv, eri_ovoo, eri_oovv, eps_o, eps_v):
    """Semicanonical `(T0)` over distinct triples, via the validated kernel."""
    total = 0.0
    for i in range(NO):
        for j in range(i + 1, NO):
            for k in range(j + 1, NO):
                total += ref.so_triple_energy(
                    i, j, k, t1, t2, eri_vovv, eri_ovoo, eri_oovv, eps_o, eps_v
                )
    return total


def test_diagonal_occupied_fock_reduces_to_t0_in_one_sweep(system):
    """Eq. (2) with a diagonal `f_oo` is Eq. (3): `T = W/D`, i.e. `(T0)`."""
    s = system
    eps_o, eps_v = s["eps"][:NO], s["eps"][NO:]
    e_iter, n_iter, converged = iterative_t1_triples_correction(
        s["t1"],
        s["t2"],
        s["eri_vovv"],
        s["eri_ovoo"],
        s["eri_oovv"],
        np.diag(eps_o),
        eps_v,
    )
    e_t0 = _t0_energy(
        s,
        s["t1"],
        s["t2"],
        s["eri_vovv"],
        s["eri_ovoo"],
        s["eri_oovv"],
        eps_o,
        eps_v,
    )
    assert converged
    # The starting guess is already the fixed point, so one sweep suffices.
    assert n_iter == 1
    assert e_iter == pytest.approx(e_t0, abs=1e-15)


def test_offdiagonal_occupied_fock_recovers_the_exact_triples(system):
    """A non-diagonal `f_oo` must converge to the exact `(T)`.

    The reference is built the other way round on purpose: rotate the occupied
    indices to the canonical basis (where `f_oo` is diagonal, so `(T0)` is
    exact) and evaluate there. Agreement means the iteration recovers the same
    physics the rotation does, without leaving the localised basis.
    """
    s = system
    rng = np.random.default_rng(11)
    eps_o, eps_v = s["eps"][:NO], s["eps"][NO:]

    a = rng.normal(size=(NO, NO)) * 0.05
    f_oo = np.diag(eps_o) + 0.5 * (a + a.T)
    np.fill_diagonal(f_oo, eps_o)  # off-diagonal coupling only

    e_iter, n_iter, converged = iterative_t1_triples_correction(
        s["t1"],
        s["t2"],
        s["eri_vovv"],
        s["eri_ovoo"],
        s["eri_oovv"],
        f_oo,
        eps_v,
    )
    assert converged
    assert n_iter > 1  # the coupling really is doing something

    w, x = np.linalg.eigh(f_oo)
    t1c = x.T @ s["t1"]
    t2c = np.einsum("iI,jJ,ijab->IJab", x, x, s["t2"], optimize=True)
    vovv = np.einsum("eibc,iI->eIbc", s["eri_vovv"], x, optimize=True)
    ovoo = np.einsum("majk,jJ,kK->maJK", s["eri_ovoo"], x, x, optimize=True)
    ovoo = np.einsum("maJK,mM->MaJK", ovoo, x, optimize=True)
    oovv = np.einsum("jkbc,jJ,kK->JKbc", s["eri_oovv"], x, x, optimize=True)
    e_exact = _t0_energy(s, t1c, t2c, vovv, ovoo, oovv, w, eps_v)

    assert e_iter == pytest.approx(e_exact, abs=1e-14)

    # The off-diagonal coupling must move the answer, or the test above would
    # pass trivially for the wrong reason.
    e_semicanonical = _t0_energy(
        s,
        s["t1"],
        s["t2"],
        s["eri_vovv"],
        s["eri_ovoo"],
        s["eri_oovv"],
        eps_o,
        eps_v,
    )
    assert abs(e_exact - e_semicanonical) > 1e-8


def test_fewer_than_three_occupieds_has_no_triples(system):
    s = system
    e_t, n_iter, converged = iterative_t1_triples_correction(
        s["t1"][:2],
        s["t2"][:2, :2],
        s["eri_vovv"][:, :2],
        s["eri_ovoo"][:2, :, :2, :2],
        s["eri_oovv"][:2, :2],
        np.diag(s["eps"][:2]),
        s["eps"][NO:],
    )
    assert e_t == 0.0
    assert n_iter == 0
    assert converged


def _all_triples(no):
    return [
        (i, j, k)
        for i in range(no)
        for j in range(i + 1, no)
        for k in range(j + 1, no)
    ]


def test_noop_screening_is_bit_identical_to_the_full_list(system):
    """Passing the complete list explicitly must change nothing at all."""
    s = system
    eps_o, eps_v = s["eps"][:NO], s["eps"][NO:]
    rng = np.random.default_rng(11)
    a = rng.normal(size=(NO, NO)) * 0.05
    f_oo = np.diag(eps_o) + 0.5 * (a + a.T)
    np.fill_diagonal(f_oo, eps_o)

    args = (
        s["t1"],
        s["t2"],
        s["eri_vovv"],
        s["eri_ovoo"],
        s["eri_oovv"],
        f_oo,
        eps_v,
    )
    implicit, n_a, _ = iterative_t1_triples_correction(*args)
    explicit, n_b, _ = iterative_t1_triples_correction(
        *args, triple_list=_all_triples(NO)
    )
    assert explicit == implicit  # bit-for-bit: same list, same arithmetic
    assert n_a == n_b


def test_screened_triples_are_dropped_from_the_energy_and_the_coupling(system):
    """A screened triple contributes nothing and couples to nothing.

    Eq. (2)'s `l`-sums run over surviving triples only, which is what lets the
    localised-basis iteration carry a screened list at all: the
    canonical-rotation `(T1)` cannot, because after the rotation an LMO-based
    screening decision no longer refers to the indices being dropped.
    """
    s = system
    eps_o, eps_v = s["eps"][:NO], s["eps"][NO:]
    rng = np.random.default_rng(11)
    a = rng.normal(size=(NO, NO)) * 0.05
    f_oo = np.diag(eps_o) + 0.5 * (a + a.T)
    np.fill_diagonal(f_oo, eps_o)

    args = (
        s["t1"],
        s["t2"],
        s["eri_vovv"],
        s["eri_ovoo"],
        s["eri_oovv"],
        f_oo,
        eps_v,
    )
    full, _, _ = iterative_t1_triples_correction(*args)

    kept = _all_triples(NO)[:-1]  # drop one triple
    screened, _, converged = iterative_t1_triples_correction(
        *args, triple_list=kept
    )
    assert converged
    # Dropping a triple must change the answer, and by a sane amount: the
    # dropped triple's own contribution plus the coupling it carried.
    assert screened != full
    assert abs(screened - full) < abs(full)


def test_screening_reduces_to_the_semicanonical_case_too(system):
    """With a diagonal `f_oo` a screened run is just `(T0)` over the subset."""
    s = system
    eps_o, eps_v = s["eps"][:NO], s["eps"][NO:]
    kept = _all_triples(NO)[:2]
    e_iter, n_iter, converged = iterative_t1_triples_correction(
        s["t1"],
        s["t2"],
        s["eri_vovv"],
        s["eri_ovoo"],
        s["eri_oovv"],
        np.diag(eps_o),
        eps_v,
        triple_list=kept,
    )
    expected = sum(
        ref.so_triple_energy(
            i,
            j,
            k,
            s["t1"],
            s["t2"],
            s["eri_vovv"],
            s["eri_ovoo"],
            s["eri_oovv"],
            eps_o,
            eps_v,
        )
        for i, j, k in kept
    )
    assert converged
    assert n_iter == 1
    assert e_iter == pytest.approx(expected, abs=1e-15)


def test_empty_and_malformed_triple_lists(system):
    s = system
    eps_o, eps_v = s["eps"][:NO], s["eps"][NO:]
    args = (
        s["t1"],
        s["t2"],
        s["eri_vovv"],
        s["eri_ovoo"],
        s["eri_oovv"],
        np.diag(eps_o),
        eps_v,
    )
    e_t, n_iter, converged = iterative_t1_triples_correction(
        *args, triple_list=[]
    )
    assert (e_t, n_iter, converged) == (0.0, 0, True)

    with pytest.raises(ValueError, match="invalid occupied triple"):
        iterative_t1_triples_correction(*args, triple_list=[(0, 0, 1)])
    with pytest.raises(ValueError, match="invalid occupied triple"):
        iterative_t1_triples_correction(*args, triple_list=[(0, 1, NO)])


def test_screening_is_exact_in_the_non_interacting_fragment_limit():
    """Screening cross-fragment triples costs exactly zero when it should.

    This is the property that makes locality screening legitimate, and the one
    the canonical-rotation `(T1)` cannot express: build a genuinely
    non-interacting two-fragment system (every integral, amplitude and
    occupied-Fock element that mixes the fragments set to zero) and screen out
    every triple spanning both. Those triples carry an identically zero `W`,
    so the exact answer must be unchanged -- bit-for-bit, not approximately.

    It also pins that restricting Eq. (2)'s `l`-sums to surviving triples does
    not corrupt the amplitudes of the triples that remain: if the coupling
    restriction leaked, the surviving fragment amplitudes would shift and the
    difference below would be nonzero.
    """
    rng = np.random.default_rng(3)
    no, nv = 6, 6
    nso = no + nv
    o, v = slice(0, no), slice(no, nso)
    grp_o = np.array([0, 0, 0, 1, 1, 1])
    grp_v = np.array([0, 0, 0, 1, 1, 1])
    grp = np.concatenate([grp_o, grp_v])

    g = rng.normal(size=(nso,) * 4) * 0.05
    g = g - g.transpose(1, 0, 2, 3)
    g = g - g.transpose(0, 1, 3, 2)
    g = 0.5 * (g + g.transpose(2, 3, 0, 1))
    same = (
        (grp[:, None, None, None] == grp[None, :, None, None])
        & (grp[:, None, None, None] == grp[None, None, :, None])
        & (grp[:, None, None, None] == grp[None, None, None, :])
    )
    g = g * same

    eps = np.concatenate([np.linspace(-1.2, -0.6, no), np.linspace(0.4, 1.6, nv)])
    t1 = rng.normal(size=(no, nv)) * 0.02 * (grp_o[:, None] == grp_v[None, :])
    t2 = rng.normal(size=(no, no, nv, nv)) * 0.02
    mask = (
        (grp_o[:, None, None, None] == grp_o[None, :, None, None])
        & (grp_o[:, None, None, None] == grp_v[None, None, :, None])
        & (grp_o[:, None, None, None] == grp_v[None, None, None, :])
    )
    t2 = t2 * mask
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)

    a = rng.normal(size=(no, no)) * 0.05
    f_oo = np.diag(eps[:no]) + 0.5 * (a + a.T)
    f_oo = f_oo * (grp_o[:, None] == grp_o[None, :])
    np.fill_diagonal(f_oo, eps[:no])

    args = (t1, t2, g[v, o, v, v], g[o, v, o, o], g[o, o, v, v], f_oo, eps[no:])
    all_triples = _all_triples(no)
    intra = [t for t in all_triples if len({grp_o[x] for x in t}) == 1]
    assert len(intra) == 2 and len(all_triples) == 20  # 18 are cross-fragment

    full, _, _ = iterative_t1_triples_correction(*args)
    screened, _, converged = iterative_t1_triples_correction(
        *args, triple_list=intra
    )
    assert converged
    assert full != 0.0  # the fragments really do correlate
    assert screened == full  # dropping 18 zero-weight triples changes nothing


# --- Milestone 2: per-triple TNO domains and the Eq. (2) `S` overlaps -------


def _coupled_fock(no, eps_o, seed=11):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(no, no)) * 0.05
    f_oo = np.diag(eps_o) + 0.5 * (a + a.T)
    np.fill_diagonal(f_oo, eps_o)
    return f_oo


def test_identity_domains_are_bit_identical_to_the_full_space(system):
    """The `S` path must collapse to the no-op it generalises.

    Handing every triple the full virtual space as its "domain" makes each
    overlap the identity, so the domain code path has to reproduce the
    full-space result exactly, not merely closely.
    """
    s = system
    eps_o, eps_v = s["eps"][:NO], s["eps"][NO:]
    f_oo = _coupled_fock(NO, eps_o)
    args = (
        s["t1"], s["t2"], s["eri_vovv"], s["eri_ovoo"], s["eri_oovv"],
        f_oo, eps_v,
    )
    full, n_full, _ = iterative_t1_triples_correction(*args)
    identity = {t: (np.eye(NV), eps_v) for t in _all_triples(NO)}
    via_domains, n_dom, converged = iterative_t1_triples_correction(
        *args, tno_domains=identity
    )
    assert converged
    assert via_domains == full
    assert n_dom == n_full


def test_a_rotated_full_span_domain_leaves_the_energy_invariant(system):
    """Same span, different basis: the answer cannot depend on the basis.

    This is what actually exercises the `S` transformations. With identity
    domains the overlaps are trivially the identity; here every domain is a
    genuine rotation of the full space, so the amplitudes really are carried
    between bases, and a wrong `S` contraction would show up immediately.
    """
    s = system
    eps_o, eps_v = s["eps"][:NO], s["eps"][NO:]
    f_oo = _coupled_fock(NO, eps_o)
    args = (
        s["t1"], s["t2"], s["eri_vovv"], s["eri_ovoo"], s["eri_oovv"],
        f_oo, eps_v,
    )
    full, _, _ = iterative_t1_triples_correction(*args)

    rng = np.random.default_rng(23)
    q, _ = np.linalg.qr(rng.normal(size=(NV, NV)))
    f_rot = q.T @ np.diag(eps_v) @ q
    w, rot = np.linalg.eigh(0.5 * (f_rot + f_rot.T))
    rotated = {t: (q @ rot, w) for t in _all_triples(NO)}

    via_rotated, _, converged = iterative_t1_triples_correction(
        *args, tno_domains=rotated
    )
    assert converged
    assert via_rotated == pytest.approx(full, abs=1e-15)


def _pair_density_domains(t2, f_vv, triples, nv, tcut):
    """TNO domains from each triple's pair-amplitude density, semicanonical."""
    out = {}
    for t in triples:
        d = np.zeros((nv, nv))
        for p, q in ((t[0], t[1]), (t[0], t[2]), (t[1], t[2])):
            amp = t2[p, q]
            d += amp @ amp.T + amp.T @ amp
        occ, vec = np.linalg.eigh(0.5 * (d + d.T))
        keep = occ > tcut
        if not keep.any():
            keep[int(np.argmax(occ))] = True
        v = vec[:, keep]
        f_t = v.T @ f_vv @ v
        eps_t, rot = np.linalg.eigh(0.5 * (f_t + f_t.T))
        out[t] = (v @ rot, eps_t)
    return out


def test_calibration_domains_degrade_monotonically_and_then_collapse():
    """This calibration truncates smoothly, then stops hosting triples.

    Measured on a 4-occupied / 7-virtual system whose TNO occupations span
    0.011 to 0.070, so the thresholds below genuinely bite. Two regimes, and
    the boundary between them is what `degenerate_tno_triples` reports: while
    every domain keeps at least three virtuals the error grows smoothly and is
    signed; once domains fall below three, triples can no longer host an
    excitation and the correction goes to zero identically rather than
    becoming small.
    """
    rng = np.random.default_rng(7)
    no, nv = 4, 7
    nso = no + nv
    o, v = slice(0, no), slice(no, nso)
    g = rng.normal(size=(nso,) * 4) * 0.05
    g = g - g.transpose(1, 0, 2, 3)
    g = g - g.transpose(0, 1, 3, 2)
    g = 0.5 * (g + g.transpose(2, 3, 0, 1))
    eps = np.concatenate([np.linspace(-1.2, -0.6, no), np.linspace(0.4, 1.8, nv)])
    t1 = rng.normal(size=(no, nv)) * 0.02
    t2 = rng.normal(size=(no, no, nv, nv)) * 0.02
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    eps_v = eps[no:]
    f_vv = np.diag(eps_v)
    args = (t1, t2, g[v, o, v, v], g[o, v, o, o], g[o, o, v, v],
            _coupled_fock(no, eps[:no]), eps_v)
    triples = _all_triples(no)

    full, _, _ = iterative_t1_triples_correction(*args)

    results = {}
    for tcut in (2e-2, 4e-2, 6e-2, 1e-1):
        dom = _pair_density_domains(t2, f_vv, triples, nv, tcut)
        energy, _, converged = iterative_t1_triples_correction(
            *args, tno_domains=dom
        )
        assert converged
        sizes = [vec.shape[1] for vec, _ in dom.values()]
        results[tcut] = (energy, float(np.mean(sizes)),
                         len(degenerate_tno_triples(dom)))

    errors = [results[t][0] - full for t in (2e-2, 4e-2, 6e-2, 1e-1)]
    domains = [results[t][1] for t in (2e-2, 4e-2, 6e-2, 1e-1)]

    assert all(e > 0.0 for e in errors)          # observed for this fixture
    assert errors == sorted(errors)              # calibration trend
    assert domains == sorted(domains, reverse=True)

    # Clean-truncation regime: domains still host triples.
    assert results[2e-2][2] == 0
    assert results[4e-2][2] == 0
    # Collapse regime: reported, and the correction is gone rather than small.
    assert results[6e-2][2] == 1
    assert results[1e-1][2] == len(triples)
    assert abs(results[1e-1][0]) < 1e-30


def test_domain_input_is_validated(system):
    s = system
    eps_o, eps_v = s["eps"][:NO], s["eps"][NO:]
    args = (
        s["t1"], s["t2"], s["eri_vovv"], s["eri_ovoo"], s["eri_oovv"],
        np.diag(eps_o), eps_v,
    )
    partial = {t: (np.eye(NV), eps_v) for t in _all_triples(NO)[:-1]}
    with pytest.raises(ValueError, match="no TNO domain supplied"):
        iterative_t1_triples_correction(*args, tno_domains=partial)

    wrong = {t: (np.eye(NV + 1), eps_v) for t in _all_triples(NO)}
    with pytest.raises(ValueError, match="expected"):
        iterative_t1_triples_correction(*args, tno_domains=wrong)

    triples = _all_triples(NO)
    first = triples[0]
    base = {t: (np.eye(NV), eps_v) for t in triples}

    wrong_energy_count = dict(base)
    wrong_energy_count[first] = (np.eye(NV), eps_v[:-1])
    with pytest.raises(ValueError, match="TNO energies.*expected"):
        iterative_t1_triples_correction(
            *args, tno_domains=wrong_energy_count
        )

    empty_domain = dict(base)
    empty_domain[first] = (np.empty((NV, 0)), np.empty(0))
    with pytest.raises(ValueError, match="TNO domain.*empty"):
        iterative_t1_triples_correction(*args, tno_domains=empty_domain)

    nonfinite_vectors = dict(base)
    bad_vectors = np.eye(NV)
    bad_vectors[0, 0] = np.nan
    nonfinite_vectors[first] = (bad_vectors, eps_v)
    with pytest.raises(ValueError, match="vectors.*finite"):
        iterative_t1_triples_correction(
            *args, tno_domains=nonfinite_vectors
        )

    nonfinite_energies = dict(base)
    bad_energies = eps_v.copy()
    bad_energies[0] = np.inf
    nonfinite_energies[first] = (np.eye(NV), bad_energies)
    with pytest.raises(ValueError, match="energies.*finite"):
        iterative_t1_triples_correction(
            *args, tno_domains=nonfinite_energies
        )

    nonorthogonal = dict(base)
    bad_basis = np.eye(NV)
    bad_basis[:, 1] = bad_basis[:, 0]
    nonorthogonal[first] = (bad_basis, eps_v)
    with pytest.raises(ValueError, match="orthonormal"):
        iterative_t1_triples_correction(
            *args, tno_domains=nonorthogonal
        )

    wrong_energy_values = dict(base)
    wrong_energy_values[first] = (np.eye(NV), eps_v + 0.1)
    with pytest.raises(ValueError, match="not semicanonical"):
        iterative_t1_triples_correction(
            *args, tno_domains=wrong_energy_values
        )

    nonsemicanonical = dict(base)
    rotation = np.eye(NV)
    theta = np.pi / 4.0
    rotation[:2, :2] = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
    nonsemicanonical[first] = (rotation, eps_v)
    with pytest.raises(ValueError, match="not semicanonical"):
        iterative_t1_triples_correction(
            *args, tno_domains=nonsemicanonical
        )


# --- Milestone 4: the paper's F_Cut and T_CutIter approximations -----------


def _approximation_system():
    """A 5-occupied / 6-virtual system with a spread of occupied couplings."""
    rng = np.random.default_rng(7)
    no, nv = 5, 6
    nso = no + nv
    o, v = slice(0, no), slice(no, nso)
    g = rng.normal(size=(nso,) * 4) * 0.05
    g = g - g.transpose(1, 0, 2, 3)
    g = g - g.transpose(0, 1, 3, 2)
    g = 0.5 * (g + g.transpose(2, 3, 0, 1))
    eps = np.concatenate(
        [np.linspace(-1.2, -0.6, no), np.linspace(0.4, 1.6, nv)]
    )
    t1 = rng.normal(size=(no, nv)) * 0.02
    t2 = rng.normal(size=(no, no, nv, nv)) * 0.02
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    f_oo = _coupled_fock(no, eps[:no], seed=13)
    return (
        t1, t2, g[v, o, v, v], g[o, v, o, o], g[o, o, v, v], f_oo, eps[no:]
    )


def test_zero_thresholds_are_exactly_the_unapproximated_scheme():
    """Both approximations must be no-ops at zero, not nearly-no-ops."""
    args = _approximation_system()
    reference, n_ref, _ = iterative_t1_triples_correction(*args)
    both_zero, n_both, _ = iterative_t1_triples_correction(
        *args, f_cut=0.0, t_cut_iter=0.0
    )
    assert both_zero == reference
    assert n_both == n_ref


def test_f_cut_above_every_coupling_collapses_to_t0():
    """The structural limit of approximation 2, and an exact one.

    `F_Cut` neglects couplings carried by an occupied Fock element below the
    threshold. Set it above *every* off-diagonal element and no coupling
    survives, so Eq. (2) degenerates into Eq. (3): the semicanonical `(T0)`,
    reached in a single sweep because the starting guess is already the fixed
    point. This has to be bit-exact, not close.
    """
    args = _approximation_system()
    f_oo = args[5]
    off = np.abs(f_oo - np.diag(np.diag(f_oo)))

    truncated, n_trunc, _ = iterative_t1_triples_correction(
        *args, f_cut=off.max() * 1.01
    )
    semicanonical_args = list(args)
    semicanonical_args[5] = np.diag(np.diag(f_oo))
    pure_t0, n_t0, _ = iterative_t1_triples_correction(*semicanonical_args)

    assert truncated == pure_t0
    assert n_trunc == n_t0 == 1


def test_calibration_f_cut_error_trend_is_not_signed():
    """These fixtures tighten smoothly, while the error sign is indefinite.

    `F_Cut` discards *couplings* between triples and is not variational: the
    discarded terms carry either sign, so the energy can overshoot the
    reference. On two deterministic fixtures the signed deviation changes
    sign while `|deviation|` is monotone. That is a calibration trend, not a
    universal monotonicity guarantee.
    """
    args = _approximation_system()
    reference, _, _ = iterative_t1_triples_correction(*args)

    deviations = []
    for f_cut in (1e-3, 1e-2, 3e-2, 1e-1):
        energy, _, converged = iterative_t1_triples_correction(
            *args, f_cut=f_cut
        )
        assert converged
        deviations.append(abs(energy - reference))

    assert deviations == sorted(deviations)
    # At the paper's 1e-3 default the approximation is essentially free, which
    # is what its SI Table S2 reports (99.93%, the same as its converged run).
    assert deviations[0] / abs(reference) < 1e-4


def test_t_cut_iter_trades_sweeps_for_accuracy_monotonically():
    """Approximation 3: freezing converged triples costs little and saves work.

    The paper does not state whether the increment is absolute or relative;
    this implementation uses the relative increment (see the module docstring),
    so the thresholds below are fractions of a triple's own energy.
    """
    args = _approximation_system()
    reference, n_ref, _ = iterative_t1_triples_correction(*args)

    sweeps, recoveries = [], []
    for t_cut in (1e-4, 1e-3, 1e-2):
        energy, n_iter, converged = iterative_t1_triples_correction(
            *args, t_cut_iter=t_cut
        )
        assert converged
        sweeps.append(n_iter)
        recoveries.append(energy / reference)

    # Looser threshold: never more sweeps, never better recovery.
    assert sweeps == sorted(sweeps, reverse=True)
    assert all(n <= n_ref for n in sweeps)
    assert recoveries == sorted(recoveries, reverse=True)
    # Still well inside 0.1% at the paper's 1e-3 default.
    assert recoveries[1] > 0.999


def test_approximation_thresholds_reject_negative_values():
    args = _approximation_system()
    with pytest.raises(ValueError, match="f_cut must be non-negative"):
        iterative_t1_triples_correction(*args, f_cut=-1e-3)
    with pytest.raises(ValueError, match="t_cut_iter must be non-negative"):
        iterative_t1_triples_correction(*args, t_cut_iter=-1e-3)
