"""Regression tests for the sparse-state 2-RDM used by the PT2 gradient.

The effective-density CASPT2/NEVPT2 gradient path builds full 1- and
2-RDMs from sparse determinant states via
``gradient._pt2_density._full_rdm12_from_state``. Two invariants are
pinned here:

1. The active (pure-Python) construction matches an independent
   brute-force spin-orbital 2-RDM to machine precision. This locks the
   2026-07-02 fix that made the fallback reachable and correct.

2. The C++ kernel ``state_rdm12_cpp`` (when built) must agree with that
   Python construction. This locks the 2026-07-03 fix that split the
   independent E_rs and bra-side E_sr precompute branches in
   ``cpp/src/nevpt2_ops.cpp``.
"""

import itertools

import numpy as np
import pytest

from vibeqc.gradient._pt2_density import _full_rdm12_from_state
from vibeqc.solvers._mrpt import _ann, _cre, _so


def _random_state(norb: int, nelec: int, seed: int) -> dict:
    """Normalized random CI state over all ``nelec``-electron determinants."""
    n_spinorb = 2 * norb
    masks = []
    for occ in itertools.combinations(range(n_spinorb), nelec):
        m = 0
        for x in occ:
            m |= 1 << x
        masks.append(m)
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=len(masks))
    coeffs /= np.linalg.norm(coeffs)
    return {m: float(c) for m, c in zip(masks, coeffs)}


def _bruteforce_rdm12(state: dict, norb: int) -> tuple[np.ndarray, np.ndarray]:
    """Spin-summed 1- and 2-RDM by explicit second-quantized operator action.

    gamma[p,q]        = <psi| E_pq |psi>
    Gamma[p,q,r,s]    = sum_{st} <psi| a+_{p s} a+_{r t} a_{s t} a_{q s} |psi>
    with E_pq = sum_sigma a+_{p sigma} a_{q sigma}.
    """

    def apply_ops(st: dict, ops) -> dict:
        cur = dict(st)
        for kind, x in reversed(ops):
            nxt: dict = {}
            for m, c in cur.items():
                sign, m2 = (_ann if kind == "a" else _cre)(m, x)
                if sign:
                    nxt[m2] = nxt.get(m2, 0.0) + sign * c
            cur = nxt
        return cur

    def dot(a: dict, b: dict) -> float:
        return sum(c * b.get(m, 0.0) for m, c in a.items())

    rdm1 = np.zeros((norb, norb))
    rdm2 = np.zeros((norb, norb, norb, norb))
    for p in range(norb):
        for q in range(norb):
            for s1 in (0, 1):
                rdm1[p, q] += dot(
                    state,
                    apply_ops(state, [("c", _so(p, s1, norb)), ("a", _so(q, s1, norb))]),
                )
            for r in range(norb):
                for s in range(norb):
                    for s1 in (0, 1):
                        for s2 in (0, 1):
                            ops = [
                                ("c", _so(p, s1, norb)),
                                ("c", _so(r, s2, norb)),
                                ("a", _so(s, s2, norb)),
                                ("a", _so(q, s1, norb)),
                            ]
                            rdm2[p, q, r, s] += dot(state, apply_ops(state, ops))
    return rdm1, rdm2


@pytest.mark.parametrize(
    "norb,nelec,seed",
    [(3, 2, 11), (4, 2, 3), (4, 3, 7), (3, 3, 5), (4, 4, 9)],
)
def test_full_rdm12_matches_bruteforce(norb, nelec, seed):
    state = _random_state(norb, nelec, seed)
    rdm1, rdm2 = _full_rdm12_from_state(state, norb)
    rdm1_bf, rdm2_bf = _bruteforce_rdm12(state, norb)
    assert np.abs(rdm1 - rdm1_bf).max() < 1e-12
    assert np.abs(rdm2 - rdm2_bf).max() < 1e-12
    # Partial-trace identity: sum_k Gamma[p,q,k,k] = (N-1) gamma[p,q].
    assert np.abs(np.einsum("pqkk->pq", rdm2) - (nelec - 1) * rdm1).max() < 1e-12


def test_state_rdm12_cpp_matches_python_2rdm():
    core = pytest.importorskip("vibeqc._vibeqc_core")
    if not hasattr(core, "state_rdm12_cpp"):
        pytest.skip("state_rdm12_cpp not built into this extension")
    norb, nelec = 3, 2
    state = _random_state(norb, nelec, seed=11)
    rdm1_py, rdm2_py = _full_rdm12_from_state(state, norb)
    r1_cpp, r2_cpp = core.state_rdm12_cpp(state, norb)
    r1_cpp = np.asarray(r1_cpp).reshape(norb, norb)
    r2_cpp = np.asarray(r2_cpp).reshape(norb, norb, norb, norb)
    # The 1-RDM is already correct; the 2-RDM is what regresses.
    assert np.abs(r1_cpp - rdm1_py).max() < 1e-12
    assert np.abs(r2_cpp - rdm2_py).max() < 1e-12
