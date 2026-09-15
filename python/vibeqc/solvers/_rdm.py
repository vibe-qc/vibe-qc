"""Reduced density matrices from a CI wavefunction.

Spin-summed (spatial-orbital) 1-, 2- and 3-particle reduced density
matrices, built from the action of the spin-summed single-excitation
generators

    Ê_pq = S_s a+_{ps} a_{qs}

on the CI vector.  The fermionic phases use the **same per-spin-sector
convention** as :mod:`vibeqc.solvers._slater_condon` (the validated FCI
engine), so the energy obtained by contracting these RDMs with the
integrals reproduces the CI eigenvalue exactly -- that identity is the
primary self-test (see :func:`energy_from_rdms`).

Conventions (matching PySCF ``fci.direct_spin1.make_rdm123`` so the RDMs
can be validated elementwise against it):

* ``rdm1[p,q]         = <a+_p a_q>``
* ``rdm2[p,q,r,s]     = <a+_p a+_r a_s a_q>``
* ``rdm3[p,q,r,s,t,u] = <a+_p a+_r a+_t a_u a_s a_q>``

all spin-summed.  The generator products ``<Ê_pq Ê_rs>`` and
``<Ê_pq Ê_rs Ê_tu>`` are reduced to these via the standard Kronecker
contractions (the same reduction PySCF's ``reorder_dm123`` applies).

``h2e`` is physicist's ``g_{pqrs} = <pq|rs> = (pr|qs)``; the energy is
``S h_pq rdm1_pq + 1/2 S (pq|rs) rdm2_pqrs`` with chemist ``(pq|rs) =
g_{prqs}``.
"""

from __future__ import annotations

import numpy as np


def _apply_aq_adp_spin(occ: tuple[int, ...], q: int, p: int):
    """Apply ``a+_p a_q`` within one spin sector to a sorted occupation tuple.

    Returns ``(sign, new_occ)`` or ``(0, None)`` if annihilated.  The sign is
    the Jordan-Wigner phase for this sector's string (number of occupied
    orbitals before the operator's target), matching ``_slater_condon``.
    """
    if q not in occ:
        return 0, None
    lst = list(occ)
    iq = lst.index(q)
    sign = -1 if (iq & 1) else 1  # (-1)^{# electrons before q}
    del lst[iq]
    if p in lst:
        return 0, None
    ip = 0
    n = len(lst)
    while ip < n and lst[ip] < p:
        ip += 1
    if ip & 1:
        sign = -sign
    lst.insert(ip, p)
    return sign, tuple(lst)


def apply_E(p, q, c, det_list, det_index) -> np.ndarray:
    """Return ``(Ê_pq c)`` in the determinant basis.

    ``(Ê_pq c)_I = S_J <D_I| Ê_pq |D_J> c_J`` with
    ``Ê_pq = S_s a+_{ps} a_{qs}``.
    """
    out = np.zeros_like(c)
    for J, (a_occ, b_occ) in enumerate(det_list):
        cj = c[J]
        if cj == 0.0:
            continue
        sign, new_a = _apply_aq_adp_spin(a_occ, q, p)
        if sign:
            out[det_index[(new_a, b_occ)]] += sign * cj
        sign, new_b = _apply_aq_adp_spin(b_occ, q, p)
        if sign:
            out[det_index[(a_occ, new_b)]] += sign * cj
    return out


def _index_map(det_list: list) -> dict:
    return {d: i for i, d in enumerate(det_list)}


def _apply_E_spill(p, q, c, det_list, det_index):
    """``Ê_pq c`` split into in-list vector + out-of-list spill dict.

    Truncated determinant lists (selected CI, CISD, ...) are not closed under
    ``Ê_pq``: components landing on determinants outside the list are
    collected in ``spill`` (keyed by SpinDet) instead of crashing the index
    lookup.  For a closed (full-CAS) list the spill stays empty and the
    vector equals :func:`apply_E`'s.
    """
    out = np.zeros_like(c)
    spill: dict = {}
    for J, (a_occ, b_occ) in enumerate(det_list):
        cj = c[J]
        if cj == 0.0:
            continue
        sign, new_a = _apply_aq_adp_spin(a_occ, q, p)
        if sign:
            key = (new_a, b_occ)
            jj = det_index.get(key)
            if jj is None:
                spill[key] = spill.get(key, 0.0) + sign * cj
            else:
                out[jj] += sign * cj
        sign, new_b = _apply_aq_adp_spin(b_occ, q, p)
        if sign:
            key = (a_occ, new_b)
            jj = det_index.get(key)
            if jj is None:
                spill[key] = spill.get(key, 0.0) + sign * cj
            else:
                out[jj] += sign * cj
    return out, spill


def _spill_dot(sa: dict, sb: dict) -> float:
    if not sa or not sb:
        return 0.0
    if len(sb) < len(sa):
        sa, sb = sb, sa
    return sum(v * sb.get(k, 0.0) for k, v in sa.items())


def _generator_products12(c, det_list, norb, idx):
    """``(rdm1, T2)`` generator products, valid for truncated lists.

    Same quantities as :func:`_generator_products` (``want3=False``) but
    each ``Ê_rs c`` keeps its out-of-list spill, so the resolution of the
    identity in ``T2[p,q,r,s] = (Ê_qp c).(Ê_rs c)`` runs over the *full*
    determinant space:  in-list and spill components live on disjoint
    determinant sets, so the product is ``vec.vec + spill.spill``.  For a
    closed (full-CAS) list every spill is empty and the result is
    bit-identical to :func:`_generator_products`.
    """
    Ec, Es = {}, {}
    EcT, EsT = {}, {}
    rdm1 = np.zeros((norb, norb))
    for r in range(norb):
        for s in range(norb):
            Ec[(r, s)], Es[(r, s)] = _apply_E_spill(r, s, c, det_list, idx)
            EcT[(r, s)], EsT[(r, s)] = _apply_E_spill(s, r, c, det_list, idx)
            rdm1[r, s] = c @ Ec[(r, s)]  # bra is in-list: spill drops out

    T2 = np.zeros((norb, norb, norb, norb))
    for r in range(norb):
        for s in range(norb):
            v = Ec[(r, s)]
            sv = Es[(r, s)]
            for p in range(norb):
                for q in range(norb):
                    T2[p, q, r, s] = EcT[(p, q)] @ v + _spill_dot(EsT[(p, q)], sv)
    return rdm1, T2


def _generator_products(c, det_list, norb, idx, want3: bool, want4: bool = False):
    """Compute generator-product tensors.

    Returns ``(rdm1, T2[, T3[, T4]])`` where
    ``rdm1[p,q]           = <Ê_pq>``,
    ``T2[p,q,r,s]         = <Ê_pq Ê_rs>``,
    ``T3[p,q,r,s,t,u]     = <Ê_pq Ê_rs Ê_tu>`` (only if ``want3``),
    ``T4[p,q,r,s,t,u,v,w] = <Ê_pq Ê_rs Ê_tu Ê_vw>`` (only if ``want4``).

    ``want4`` always produces ``T3`` as well, so the 4-RDM caller has the
    reordered lower-order tensors it needs.  T4 is O(norb**8) in both storage
    and build cost, so it is gated off by default and assembled only for small
    active spaces.
    """
    # Ec[(r,s)] = Ê_rs c ;  EcT[(p,q)] = Ê_qp c = adjoint partner for dots.
    Ec = {}
    EcT = {}
    rdm1 = np.zeros((norb, norb))
    for r in range(norb):
        for s in range(norb):
            Ec[(r, s)] = apply_E(r, s, c, det_list, idx)
            EcT[(r, s)] = apply_E(s, r, c, det_list, idx)  # <.| Ê_rs = (Ê_sr c)|
            rdm1[r, s] = c @ Ec[(r, s)]

    # T2[p,q,r,s] = <Ê_pq Ê_rs> = (Ê_qp c) . (Ê_rs c)
    T2 = np.zeros((norb, norb, norb, norb))
    for r in range(norb):
        for s in range(norb):
            v = Ec[(r, s)]
            for p in range(norb):
                for q in range(norb):
                    T2[p, q, r, s] = EcT[(p, q)] @ v
    if not want3:
        return rdm1, T2

    # T3[p,q,r,s,t,u] = <Ê_pq Ê_rs Ê_tu> = (Ê_qp c) . (Ê_rs Ê_tu c)
    T3 = np.zeros((norb,) * 6)
    for t in range(norb):
        for u in range(norb):
            v1 = Ec[(t, u)]
            for r in range(norb):
                for s in range(norb):
                    v2 = apply_E(r, s, v1, det_list, idx)
                    for p in range(norb):
                        for q in range(norb):
                            T3[p, q, r, s, t, u] = EcT[(p, q)] @ v2
    if not want4:
        return rdm1, T2, T3

    # T4[p,q,r,s,t,u,v,w] = <Ê_pq Ê_rs Ê_tu Ê_vw>
    #                     = (Ê_qp c) . (Ê_rs Ê_tu Ê_vw c).
    # One more nested apply_E level than T3: start from the cached
    # (Ê_vw c) = Ec[(v,w)], apply Ê_tu, then Ê_rs, then dot with the cached
    # bra (Ê_qp c) = EcT[(p,q)].  O(norb**8); built only for small active
    # spaces (gated by want4).
    T4 = np.zeros((norb,) * 8)
    for v in range(norb):
        for w in range(norb):
            w1 = Ec[(v, w)]
            for t in range(norb):
                for u in range(norb):
                    w2 = apply_E(t, u, w1, det_list, idx)
                    for r in range(norb):
                        for s in range(norb):
                            w3 = apply_E(r, s, w2, det_list, idx)
                            for p in range(norb):
                                for q in range(norb):
                                    T4[p, q, r, s, t, u, v, w] = EcT[(p, q)] @ w3
    return rdm1, T2, T3, T4


def _reorder_rdm2(rdm1, T2):
    """<Ê_pq Ê_rs> -> <a+_p a+_r a_s a_q> (PySCF dm2)."""
    dm2 = T2.copy()
    norb = rdm1.shape[0]
    for k in range(norb):
        dm2[:, k, k, :] -= rdm1
    return dm2


def make_rdm1(c, det_list, norb) -> np.ndarray:
    """Spin-summed 1-RDM ``rdm1[p,q] = <a+_p a_q>``.

    Valid for truncated determinant lists (selected CI, CISD, ...): the bra
    projection makes out-of-list ``Ê_pq`` components drop out exactly.
    """
    idx = _index_map(det_list)
    rdm1, _ = _generator_products12(c, det_list, norb, idx)
    return rdm1


def _full_cas_counts(det_list, norb):
    """(n_alpha, n_beta) if ``det_list`` is the canonical full CAS space in
    solvers._determinant.generate_determinants order, else None.

    The C++ direct RDM kernel assumes that exact space + ordering; truncated
    lists (selected CI, CISD, ...) must use the Python path.
    """
    from math import comb

    if not det_list:
        return None
    na_occ, nb_occ = det_list[0]
    n_alpha, n_beta = len(na_occ), len(nb_occ)
    if len(det_list) != comb(norb, n_alpha) * comb(norb, n_beta):
        return None
    if det_list[0] != (
        tuple(range(n_alpha)),
        tuple(range(n_beta)),
    ) or det_list[-1] != (
        tuple(range(norb - n_alpha, norb)),
        tuple(range(norb - n_beta, norb)),
    ):
        return None
    return n_alpha, n_beta


# RDM backend dispatch: full-CAS determinant lists above this size use the
# C++ direct kernel (cpp/src/casci.cpp::casci_direct_rdm12 -- same phase and
# Force with VIBEQC_RDM_BACKEND=python|cpp|auto.


def make_rdm12(c, det_list, norb):
    """Spin-summed (rdm1, rdm2) in PySCF convention.

    Always attempts the C++ path first (``casci_direct_rdm12`` for full-CAS
    lists, ``selected_ci_rdm12`` for truncated/selected-CI lists).  Falls
    back to the spill-aware Python generator-product path only when the C++
    bindings are unavailable.
    """
    import os

    backend = os.environ.get("VIBEQC_RDM_BACKEND", "auto")
    if backend not in ("auto", "cpp", "python"):
        raise ValueError(f"VIBEQC_RDM_BACKEND must be auto|cpp|python, got {backend!r}")
    if backend == "python":
        idx = _index_map(det_list)
        rdm1, T2 = _generator_products12(c, det_list, norb, idx)
        return rdm1, _reorder_rdm2(rdm1, T2)

    # Try C++ kernels in order: full-CAS direct, then selected-CI.
    counts = _full_cas_counts(det_list, norb)
    if counts is not None:
        try:
            from .._vibeqc_core import casci_direct_rdm12 as _direct_rdm12
        except ImportError:
            pass
        else:
            rdm1, rdm2 = _direct_rdm12(
                np.ascontiguousarray(np.asarray(c, dtype=float)),
                norb,
                counts[0],
                counts[1],
            )
            return np.asarray(rdm1), np.asarray(rdm2)

    # Truncated (selected-CI / CISD) list: the C++ pair-based
    # Slater-Condon kernel is exact for arbitrary lists.
    try:
        from .._vibeqc_core import selected_ci_rdm12 as _sel_rdm12
    except ImportError:
        pass
    else:
        from ._selected_ci import _dets_to_masks

        ma, mb = _dets_to_masks(det_list)
        rdm1, rdm2 = _sel_rdm12(
            ma,
            mb,
            np.ascontiguousarray(np.asarray(c, dtype=float)),
            norb,
        )
        return np.asarray(rdm1), np.asarray(rdm2)

    # Python fallback: spill-aware generator products.
    idx = _index_map(det_list)
    rdm1, T2 = _generator_products12(c, det_list, norb, idx)
    return rdm1, _reorder_rdm2(rdm1, T2)


def make_rdm12_sa(
    ci_coeffs_all: np.ndarray,
    det_list: list,
    norb: int,
    weights: list[float] | None = None,
):
    """State-averaged spin-summed (rdm1, rdm2) in PySCF convention.

    Parameters
    ----------
    ci_coeffs_all : (n_det, nroots) ndarray
        CI coefficients for all states (columns).
    det_list : list
        Determinant list.
    norb : int
        Number of active spatial orbitals.
    weights : list[float] or None
        State-averaging weights (must sum to 1).  Defaults to equal weights.

    Returns
    -------
    rdm1 : (norb, norb) ndarray
        Weighted-average 1-RDM.
    rdm2 : (norb, norb, norb, norb) ndarray
        Weighted-average 2-RDM, PySCF convention.
    """
    nroots = ci_coeffs_all.shape[1]
    if weights is None:
        w = [1.0 / nroots] * nroots
    else:
        w = list(weights)
        if len(w) != nroots:
            raise ValueError(f"len(weights)={len(w)} != nroots={nroots}")
        if abs(sum(w) - 1.0) > 1e-12:
            raise ValueError(f"Weights must sum to 1, got sum={sum(w)}")

    rdm1_sa = np.zeros((norb, norb))
    rdm2_sa = np.zeros((norb, norb, norb, norb))
    for k in range(nroots):
        # Per-root build via make_rdm12 so the C++ direct-kernel dispatch
        # (full-CAS spaces) applies to state-averaged runs too.
        r1, r2 = make_rdm12(ci_coeffs_all[:, k], det_list, norb)
        rdm1_sa += w[k] * r1
        rdm2_sa += w[k] * r2
    return rdm1_sa, rdm2_sa


def make_rdm123(c, det_list, norb):
    """Spin-summed (rdm1, rdm2, rdm3) in PySCF convention.

    rdm3[p,q,r,s,t,u] = <a+_p a+_r a+_t a_u a_s a_q>.  The reduction from
    <Ê_pq Ê_rs Ê_tu> replicates PySCF ``reorder_dm123`` exactly.

    Requires a closed (full-CAS) determinant list: the chained
    ``Ê_rs Ê_tu`` intermediates assume the space is closed under single
    excitations (truncated lists would need multi-level spill tracking).
    """
    if len(det_list) > 0:
        from math import comb

        na, nb = len(det_list[0][0]), len(det_list[0][1])
        if len(det_list) != comb(norb, na) * comb(norb, nb):
            raise NotImplementedError(
                "make_rdm123 requires the full CAS determinant list; "
                "truncated lists (selected CI / CISD) support 1-/2-RDMs "
                "only (make_rdm12)"
            )
    idx = _index_map(det_list)
    rdm1, T2, T3 = _generator_products(c, det_list, norb, idx, want3=True)
    dm2 = _reorder_rdm2(rdm1, T2)

    dm3 = T3.copy()
    norb_ = norb
    for q in range(norb_):
        dm3[:, q, q, :, :, :] -= dm2
        dm3[:, :, :, q, q, :] -= dm2
        dm3[:, q, :, :, q, :] -= dm2.transpose(0, 2, 3, 1)
        for s in range(norb_):
            dm3[:, q, q, s, s, :] -= rdm1.T
    return rdm1, dm2, dm3


def make_rdm1234(c, det_list, norb):
    """Spin-summed (rdm1, rdm2, rdm3, rdm4) in PySCF convention.

    rdm4[p,q,r,s,t,u,v,w] = <a+_p a+_r a+_t a+_v a_w a_u a_s a_q> (all
    spin-summed).  The reduction from the generator product
    <Ê_pq Ê_rs Ê_tu Ê_vw> to the normal-ordered 4-RDM replicates PySCF
    ``fci.rdm.reorder_dm1234`` exactly: six dm3-transpose subtractions over
    the q index, seven dm2-transpose subtractions over the (q,s) index pair,
    and a final rdm1.T subtraction over (q,s,u).  This is the standard
    generator-product-to-RDM reduction (Kronecker-delta contractions of the
    excitation operators), the same one ``make_rdm123`` applies one order
    lower.

    The dm2 and dm3 fed into the dm4 reduction are the already-reordered,
    PySCF-convention tensors (their own lower-order reductions are baked in
    and must not be repeated here), matching how ``reorder_dm1234`` chains
    ``reorder_dm123``.

    T4 is O(norb**8) in storage and build cost, so this routine is for small
    active spaces only.

    Requires a closed (full-CAS) determinant list (see :func:`make_rdm123`).
    """
    if len(det_list) > 0:
        from math import comb

        na, nb = len(det_list[0][0]), len(det_list[0][1])
        if len(det_list) != comb(norb, na) * comb(norb, nb):
            raise NotImplementedError(
                "make_rdm1234 requires the full CAS determinant list; "
                "truncated lists (selected CI / CISD) support 1-/2-RDMs "
                "only (make_rdm12)"
            )
    idx = _index_map(det_list)
    rdm1, T2, T3, T4 = _generator_products(
        c, det_list, norb, idx, want3=True, want4=True
    )
    dm2 = _reorder_rdm2(rdm1, T2)

    dm3 = T3.copy()
    for q in range(norb):
        dm3[:, q, q, :, :, :] -= dm2
        dm3[:, :, :, q, q, :] -= dm2
        dm3[:, q, :, :, q, :] -= dm2.transpose(0, 2, 3, 1)
        for s in range(norb):
            dm3[:, q, q, s, s, :] -= rdm1.T

    # T4 -> dm4: replicate fci.rdm.reorder_dm1234, operating on the
    # already-reordered dm3 / dm2 / rdm1.
    dm4 = T4.copy()
    for q in range(norb):
        dm4[:, q, :, :, :, :, q, :] -= dm3.transpose(0, 2, 3, 4, 5, 1)
        dm4[:, :, :, q, :, :, q, :] -= dm3.transpose(0, 1, 2, 4, 5, 3)
        dm4[:, :, :, :, :, q, q, :] -= dm3
        dm4[:, q, :, :, q, :, :, :] -= dm3.transpose(0, 2, 3, 1, 4, 5)
        dm4[:, :, :, q, q, :, :, :] -= dm3
        dm4[:, q, q, :, :, :, :, :] -= dm3
        for s in range(norb):
            dm4[:, q, q, s, :, :, s, :] -= dm2.transpose(0, 2, 3, 1)
            dm4[:, q, q, :, :, s, s, :] -= dm2
            dm4[:, q, :, :, q, s, s, :] -= dm2.transpose(0, 2, 3, 1)
            dm4[:, q, :, s, q, :, s, :] -= dm2.transpose(0, 2, 1, 3)
            dm4[:, q, :, s, s, :, q, :] -= dm2.transpose(0, 2, 3, 1)
            dm4[:, :, :, s, s, q, q, :] -= dm2
            dm4[:, q, q, s, s, :, :, :] -= dm2
            for u in range(norb):
                dm4[:, q, q, s, s, u, u, :] -= rdm1.T
    return rdm1, dm2, dm3, dm4


def energy_from_rdms(
    h1e: np.ndarray,
    h2e: np.ndarray,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    e_const: float = 0.0,
) -> float:
    """Energy ``e_const + S h_pq rdm1_pq + 1/2 S (pq|rs) rdm2_pqrs``.

    ``h2e`` is physicist's ``g``; chemist ``(pq|rs) = g_{prqs}``.  ``rdm2``
    is the PySCF-convention 2-RDM.  Equals the CI eigenvalue when the RDMs
    are those of the CI state and ``e_const`` is the constant (nuclear +
    frozen-core) shift -- the primary RDM self-test.
    """
    eri_chem = h2e.transpose(0, 2, 1, 3)
    e1 = float(np.einsum("pq,pq->", h1e, rdm1))
    e2 = 0.5 * float(np.einsum("pqrs,pqrs->", eri_chem, rdm2))
    return e_const + e1 + e2
