"""Parity tests for the spin-summed RDM builders in ``vibeqc.solvers._rdm``
against PySCF's reference implementation.

These cover the 1/2/3-RDM (``make_rdm123``) and the 4-RDM (``make_rdm1234``).
vibe-qc and PySCF evaluate the RDMs of the *same* CI vector, so the comparison
is exact up to floating-point round-off, not a statistical agreement.

PySCF is used here strictly as an out-of-process reference in the test suite
(the ``importorskip`` gate below), mirroring the existing FCI / MRPT parity
tests.  vibe-qc's own runtime never imports PySCF (CLAUDE.md section 10); the
``cistring`` helper only translates determinant addresses so the identical CI
amplitudes feed both implementations.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from vibeqc.solvers._rdm import make_rdm123, make_rdm1234

pyscf = pytest.importorskip("pyscf")
from pyscf import fci  # noqa: E402
from pyscf.fci import cistring  # noqa: E402


def _random_hamiltonian(norb: int, seed: int):
    """Small random Hermitian Hamiltonian for FCI.

    Returns ``(h1e, eri)`` with ``h1e`` symmetric and ``eri`` in chemist
    notation ``(pq|rs)`` carrying full 8-fold permutational symmetry.
    """
    rng = np.random.default_rng(seed)
    h1e = rng.standard_normal((norb, norb))
    h1e = 0.5 * (h1e + h1e.T)

    g = rng.standard_normal((norb, norb, norb, norb))
    eri = (
        g
        + g.transpose(1, 0, 2, 3)
        + g.transpose(0, 1, 3, 2)
        + g.transpose(1, 0, 3, 2)
        + g.transpose(2, 3, 0, 1)
        + g.transpose(3, 2, 0, 1)
        + g.transpose(2, 3, 1, 0)
        + g.transpose(3, 2, 1, 0)
    ) / 8.0
    return h1e, eri


def _civec_to_det_list(civec, norb, nalpha, nbeta):
    """Map a PySCF FCI vector to (det_list, c) in vibe-qc convention.

    ``civec`` is shaped ``(num_a, num_b)`` and addressed by PySCF's string
    ordering.  vibe-qc's ``det_list`` is a list of ``(alpha_occ, beta_occ)``
    occupation-tuple pairs; ``c`` is indexed by position in that list.  We
    enumerate determinants in ``itertools.combinations`` order (vibe-qc's own
    enumeration) and pull each amplitude from ``civec`` via
    ``cistring.str2addr`` so the identical wavefunction is represented on both
    sides.
    """
    alpha_dets = [tuple(d) for d in itertools.combinations(range(norb), nalpha)]
    beta_dets = [tuple(d) for d in itertools.combinations(range(norb), nbeta)]

    def _addr(occ, nelec):
        bits = 0
        for o in occ:
            bits |= 1 << o
        return cistring.str2addr(norb, nelec, bits)

    addr_a = {a: _addr(a, nalpha) for a in alpha_dets}
    addr_b = {b: _addr(b, nbeta) for b in beta_dets}

    det_list = [(a, b) for a in alpha_dets for b in beta_dets]
    c = np.empty(len(det_list))
    for i, (a, b) in enumerate(det_list):
        c[i] = civec[addr_a[a], addr_b[b]]
    return det_list, c


def _fci_civec(norb, nalpha, nbeta, seed):
    """Run a small PySCF FCI; return (civec, det_list, c) for the same state."""
    h1e, eri = _random_hamiltonian(norb, seed)
    _, civec = fci.direct_spin1.kernel(
        h1e, eri, norb, (nalpha, nbeta), ecore=0.0, nroots=1
    )
    civec = np.asarray(civec)
    det_list, c = _civec_to_det_list(civec, norb, nalpha, nbeta)
    return civec, det_list, c


# (norb, nalpha, nbeta, seed, label)
_CASES = [
    (3, 2, 1, 11, "na2_nb1_norb3"),
    (3, 2, 2, 23, "na2_nb2_norb3"),
    (4, 2, 2, 37, "cas44_na2_nb2"),
]
_CASE_IDS = [c[-1] for c in _CASES]
_TOL = 1e-10


@pytest.mark.parametrize("norb,nalpha,nbeta,seed,label", _CASES, ids=_CASE_IDS)
def test_make_rdm123_matches_pyscf(norb, nalpha, nbeta, seed, label):
    civec, det_list, c = _fci_civec(norb, nalpha, nbeta, seed)
    rdm1, dm2, dm3 = make_rdm123(c, det_list, norb)
    p1, p2, p3 = fci.direct_spin1.make_rdm123(civec, norb, (nalpha, nbeta))
    assert np.max(np.abs(rdm1 - p1)) <= _TOL, f"{label}: rdm1"
    assert np.max(np.abs(dm2 - p2)) <= _TOL, f"{label}: rdm2"
    assert np.max(np.abs(dm3 - p3)) <= _TOL, f"{label}: rdm3"


@pytest.mark.parametrize("norb,nalpha,nbeta,seed,label", _CASES, ids=_CASE_IDS)
def test_make_rdm1234_matches_pyscf(norb, nalpha, nbeta, seed, label):
    civec, det_list, c = _fci_civec(norb, nalpha, nbeta, seed)
    rdm1, dm2, dm3, dm4 = make_rdm1234(c, det_list, norb)
    p1, p2, p3, p4 = fci.direct_spin1.make_rdm1234(civec, norb, (nalpha, nbeta))
    assert np.max(np.abs(rdm1 - p1)) <= _TOL, f"{label}: rdm1"
    assert np.max(np.abs(dm2 - p2)) <= _TOL, f"{label}: rdm2"
    assert np.max(np.abs(dm3 - p3)) <= _TOL, f"{label}: rdm3"
    assert np.max(np.abs(dm4 - p4)) <= _TOL, f"{label}: rdm4"


@pytest.mark.parametrize("norb,nalpha,nbeta,seed,label", _CASES, ids=_CASE_IDS)
def test_make_rdm1234_lower_orders_match_make_rdm123(
    norb, nalpha, nbeta, seed, label
):
    # Adding want4 must not perturb the lower-order RDMs the existing callers
    # rely on.
    _, det_list, c = _fci_civec(norb, nalpha, nbeta, seed)
    r1a, d2a, d3a = make_rdm123(c, det_list, norb)
    r1b, d2b, d3b, _ = make_rdm1234(c, det_list, norb)
    assert np.max(np.abs(r1a - r1b)) <= _TOL, f"{label}: rdm1 drift"
    assert np.max(np.abs(d2a - d2b)) <= _TOL, f"{label}: rdm2 drift"
    assert np.max(np.abs(d3a - d3b)) <= _TOL, f"{label}: rdm3 drift"
