"""Excitation-case-partitioned internally-contracted CASPT2 (roadmap 25i).

The dense IC-CASPT2 in :mod:`._mrpt` builds the whole first-order interacting
space (FOIS) ``{Ê_pr Ê_qs|0>}`` as explicit determinant states and forms one
monolithic overlap ``S``, zeroth-order ``H₀`` and RHS ``V`` over every
determinant that appears.  That is exact but scales with the determinant
count of |0>, so it is a small-active-space method.

OpenMolcas (and CASPT2 generally, Andersson-Malmqvist-Roos 1992;
Celani-Werner 2000) instead partitions the FOIS by *excitation case*: the
pattern of inactive holes / active / secondary particles carried by each
contracted excitation, and contracts each case's ``S`` / ``H₀`` / ``V``
block against the reference *reduced density matrices* rather than the
determinant expansion.  The number of active indices in a case fixes the RDM
order it needs:

    case (created | annihilated, labels I=inactive A=active S=secondary)
      0 active : SS<-II, S<-I            -> no active RDM (MP2-like)
      1 active : SS<-AI, AS<-II, ...     -> 1-RDM
      2 active : AS<-AI, SS<-AA, AA<-II  -> 2-RDM
      3 active : AS<-AA, AA<-AI          -> 3-RDM (S);  4-RDM (H₀) -> F3
      4 active : AA<-AA                  -> internal, projected out of the FOIS

The 4-RDM that the 3-active H₀ block would need is never built: it is
pre-contracted with the diagonal generalized Fock into the intermediate
``F3`` (same size as the 3-RDM).  See the milestone roadmap in
``handovers/HANDOVER_CAS.md``.

This module is built incrementally against the dense engine as an exact
oracle.  **M24 (this milestone)** is the case-classification +
block-organized FOIS framework: it reproduces the dense ``S`` / ``H₀`` /
``V`` and hence the dense second-order energy to machine precision, still
building each case's block from the determinant expansion.  Later milestones
(M25/M26) replace each block's determinant build with the RDM contraction,
case by case, validated element-wise against the dense block; M27 drops the
determinant FOIS entirely and composes with the selected-CI reference.

Nothing here is wired into the public :func:`._mrpt.caspt2` yet; the dense
engine remains the shipped, OpenMolcas-pinned default.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

from ._mrpt import _add, _dot, _so, apply_1body, apply_2body
from ._rdm import _reorder_rdm2, _apply_aq_adp_spin


def classify_case(created, annih, n_core: int, n_act: int, norb: int):
    """Excitation case of one contracted function ``Ê_{created<-annih}|0>``.

    ``created`` are the creation (particle) orbital indices, drawn from the
    active+secondary set; ``annih`` the annihilation (hole) indices, drawn
    from the inactive+active set.  Returns ``(created_labels, annih_labels)``
    as sorted label strings over ``I`` (inactive, ``< n_core``), ``A``
    (active), ``S`` (secondary, ``>= n_core + n_act``): a canonical,
    order-independent case key.
    """
    def lab(o: int) -> str:
        if o < n_core:
            return "I"
        if o < n_core + n_act:
            return "A"
        return "S"

    return (
        "".join(sorted(lab(o) for o in created)),
        "".join(sorted(lab(o) for o in annih)),
    )


def case_active_count(case) -> int:
    """Number of active indices in a case key = the RDM order its overlap needs.

    The zeroth-order ``H₀`` block needs one order higher when the active
    generalized-Fock block contracts in an extra active index (so the
    3-active cases reach the 4-RDM, handled via the ``F3`` intermediate).
    """
    created_labels, annih_labels = case
    return created_labels.count("A") + annih_labels.count("A")


def is_internal_case(case) -> bool:
    """True for the all-active cases that live inside the CAS (projected out)."""
    created_labels, annih_labels = case
    return "S" not in created_labels and "I" not in annih_labels


# ── RDM-contracted FOIS overlap blocks (M25) ────────────────────────────
# The overlap of two contracted excitations factorizes into a Kronecker
# product over the inactive/secondary (spectator) indices times an
# active-space expectation that reduces to a reduced-density-matrix
# contraction.  The pattern (validated per-element against the determinant
# primitive <0|Ê+Ê|0>, the dense engine's apply_1body on the reference):
#
#   an annihilated active orbital contracts the PARTICLE density g
#     (g_tu = <0|Ê_tu|0>, the spin-summed 1-RDM), while a created active
#     orbital contracts the HOLE density (2d - g);
#   inactive holes / secondary particles give factor-2 spin-summed deltas.
#
# The active-index count of the case fixes the RDM order (0 -> scalar,
# 1 -> 1-RDM, 2 -> 2-RDM, 3 -> 3-RDM; the 4-active block is internal).  This
# milestone implements the closed forms for the 0-active and 1-active
# single cases; the 1-active doubles, 2-/3-active blocks, and the V / H₀
# contractions (the last needing F3 for the 3-active 4-RDM) are the
# subsequent M25/M26 increments.  ``rdm1`` is the spin-summed active 1-RDM
# in PySCF convention (``solvers.make_rdm1``).


def overlap_s_i(a, i, ap, ip):
    """<E_{a i}|E_{a' i'}>, the S<-I (0-active single) block.

    Closed shell: ``2 d_{aa'} d_{ii'}`` (a,a' secondary; i,i' inactive).
    """
    return 2.0 if (a == ap and i == ip) else 0.0


def overlap_ss_ii(a, i, b, j, ap, ip, bp, jp):
    """<E_{a i}E_{b j}|E_{a' i'}E_{b' j'}>, the SS<-II (0-active double) block.

    Closed-shell doubles metric (validated):
      4 d_aa'd_bb'd_ii'd_jj' + 4 d_ab'd_ba'd_ij'd_ji'
      - 2 d_ab'd_ba'd_ii'd_jj' - 2 d_aa'd_bb'd_ij'd_ji'.
    a,b,a',b' secondary; i,j,i',j' inactive.
    """
    d = lambda x, y: 1.0 if x == y else 0.0  # noqa: E731
    direct = d(a, ap) * d(b, bp) * d(i, ip) * d(j, jp)
    swap_both = d(a, bp) * d(b, ap) * d(i, jp) * d(j, ip)
    swap_cre = d(a, bp) * d(b, ap) * d(i, ip) * d(j, jp)
    swap_ann = d(a, ap) * d(b, bp) * d(i, jp) * d(j, ip)
    return 4.0 * direct + 4.0 * swap_both - 2.0 * swap_cre - 2.0 * swap_ann


def overlap_s_a(a, t, ap, tp, rdm1, n_core):
    """<E_{a t}|E_{a' t'}>, the S<-A (1-active single) block.

    The annihilated active orbital contracts the particle density:
    ``d_{aa'} g_{t t'}`` (a,a' secondary; t,t' active; g = active 1-RDM).
    """
    if a != ap:
        return 0.0
    return float(rdm1[t - n_core, tp - n_core])


def overlap_a_i(t, i, tp, ip, rdm1, n_core):
    """<E_{t i}|E_{t' i'}>, the A<-I (1-active single) block.

    The created active orbital contracts the hole density:
    ``d_{ii'} (2 d_{t t'} - g_{t t'})`` (i,i' inactive; t,t' active).
    """
    if i != ip:
        return 0.0
    hole = (2.0 if t == tp else 0.0) - float(rdm1[t - n_core, tp - n_core])
    return hole


# ── Generic active E-product expectation engine (M25/M26 foundation) ──────
# The per-case overlap closed forms above were derived one case at a time by
# probing the determinant primitive; that does not scale to the 2-/3-active
# doubles, nor to V and H₀.  Every FOIS quantity vibe-qc's CASPT2 needs is an
# expectation of a *spin-summed E-operator product* over the reference,
#
#     S_IJ  = <0| Ê_I+ Ê_J |0>
#     V_I   = <0| Ê_I+ Ĥ  |0>
#     (H₀)_IJ = <0| Ê_I+ F̂ Ê_J |0>,
#
# so after the external (inactive-hole / secondary-particle) indices factor
# out as Kronecker deltas, the active core of each block is a product
# <0| Ê_{p₁q₁} ... Ê_{pₖqₖ} |0> of purely *active* generators.  This single
# routine reduces that product to the active spin-summed RDMs, replacing the
# per-case guesswork with one validated contraction.
#
# The reduction inverts the generator-product -> normal-ordered-RDM relations
# that :func:`vibeqc.solvers._rdm.make_rdm123` applies in the forward
# direction (the T2/T3 there *are* these generator products; dm2/dm3 are them
# minus the delta terms below).  Validated element-wise against the
# determinant primitive <0|Ê...Ê|0> in tests/test_caspt2_cases.py.


def active_eprod(ops, rdm1, dm2=None, dm3=None, t4=None):
    """``<0| Ê_{p₁q₁} Ê_{p₂q₂} ... Ê_{pₖqₖ} |0>`` over active orbitals.

    ``ops`` is ``[(p₁,q₁), ..., (pₖ,qₖ)]`` with **active-block** indices
    (``0 ... n_act-1``); ``rdm1`` / ``dm2`` / ``dm3`` are the active
    spin-summed RDMs in PySCF / :func:`make_rdm123` convention
    (``g_pq``, ``Γ_pqrs = <a+_p a+_r a_s a_q>``,
    ``Γ3_pqrstu = <a+_p a+_r a+_t a_u a_s a_q>``).  Supports ``k = 1, 2, 3``,
    the orders the FOIS overlap ``S`` and RHS ``V`` reach (the 3-active
    cases need the 3-RDM).

    ``k = 4`` (the H₀ 4-RDM block) is supported only when the raw 4-generator
    product tensor ``t4[p₁q₁p₂q₂p₃q₃p₄q₄] = <Ê Ê Ê Ê>`` is supplied, used to
    *validate* H₀ on small active spaces.  Production H₀ never builds this
    O(n_act⁸) tensor: the 4-RDM only ever appears contracted with the
    diagonal Fock, taken via the F3 intermediate (M26).

    The closed forms (all deltas Kronecker on active indices)::

        k=1: g_{p₁q₁}
        k=2: Γ_{p₁q₁p₂q₂} + d_{q₁p₂} g_{p₁q₂}
        k=3: Γ3_{p₁q₁p₂q₂p₃q₃}
             + d_{q₁p₂} Γ_{p₁q₂p₃q₃}
             + d_{q₂p₃} Γ_{p₁q₁p₂q₃}
             + d_{q₁p₃} Γ_{p₁q₃p₂q₂}
             + d_{q₁p₂} d_{q₂p₃} g_{q₃p₁}
    """
    k = len(ops)
    if k == 0:
        return 1.0
    if k == 1:
        (p1, q1), = ops
        return float(rdm1[p1, q1])
    if k == 2:
        (p1, q1), (p2, q2) = ops
        val = dm2[p1, q1, p2, q2]
        if q1 == p2:
            val += rdm1[p1, q2]
        return float(val)
    if k == 3:
        (p1, q1), (p2, q2), (p3, q3) = ops
        val = dm3[p1, q1, p2, q2, p3, q3]
        if q1 == p2:
            val += dm2[p1, q2, p3, q3]
        if q2 == p3:
            val += dm2[p1, q1, p2, q3]
        if q1 == p3:
            val += dm2[p1, q3, p2, q2]
        if q1 == p2 and q2 == p3:
            val += rdm1[q3, p1]
        return float(val)
    if k == 4 and t4 is not None:
        (p1, q1), (p2, q2), (p3, q3), (p4, q4) = ops
        return float(t4[p1, q1, p2, q2, p3, q3, p4, q4])
    raise NotImplementedError(
        "active_eprod supports products of up to three spin-summed "
        "generators (3-RDM); the 4-generator H₀ block needs the raw 4-RDM "
        "(pass t4, for small-space validation) or the F3 intermediate (M26)"
    )


# ── General mixed-index reducer (external indices -> active engine) ────────
# active_eprod handles a product of purely *active* generators.  A FOIS
# overlap / RHS element is a product of generators with *mixed* I/A/S
# indices (E_ti, E_at, E_ai, ... carry one external index each).  This reducer
# evaluates <0| ∏ E_{p q} |0> for arbitrary index characters by the
# spin-summed E-operator commutator algebra
#
#     [E_pq, E_rs] = d_qr E_ps - d_sp E_rq               (U(n) generators)
#
# together with the closed-shell boundary actions of one generator on the
# reference (i inactive = doubly occupied, a secondary = empty):
#
#     E_pq|0> = 0            if q secondary, or p inactive with p!=q
#             = 2 d_pq |0>   if p,q both inactive
#     <0|E_pq = 0            if p secondary, or q inactive with p!=q
#             = 2 d_pq <0|   if p,q both inactive
#
# (the conjugate pair).  Working entirely in spin-summed generators keeps the
# reduction reference-agnostic (no closed-shell-singlet assumption), and the
# spin coupling (the 2/-1 closed-shell coefficients) falls out of the U(n)
# structure constants automatically.  Reduction strategy: a boundary
# generator that bottoms out is applied immediately; otherwise a non-active
# generator is commuted to the edge where it bottoms, each commutator
# shortening the product, so the recursion terminates in active-only products
# that hand off to active_eprod.  Validated exhaustively against the
# determinant primitive on mixed tuples and against the dense FOIS overlap.


def _lab(o, n_core, n_act):
    if o < n_core:
        return "I"
    if o < n_core + n_act:
        return "A"
    return "S"


def _bottom_right(p, q, lab):
    """``E_pq|0>`` as a scalar x ``|0>``, or ``None`` if a genuine excitation."""
    if lab(q) == "S":
        return 0.0
    if lab(p) == "I":
        return 2.0 if p == q else 0.0
    return None


def _bottom_left(p, q, lab):
    """``<0|E_pq`` as a scalar x ``<0|``, or ``None`` if genuine."""
    if lab(p) == "S":
        return 0.0
    if lab(q) == "I":
        return 2.0 if p == q else 0.0
    return None


def _commutator(a, b, x, y):
    """``[E_ab, E_xy] = d_bx E_ay - d_ya E_xb`` as ``[(coeff, (p,q)), ...]``."""
    out = []
    if b == x:
        out.append((1.0, (a, y)))
    if y == a:
        out.append((-1.0, (x, b)))
    return out


def _G(ops, lab, n_core, rdm1, dm2, dm3, t4=None, s0=1.0):
    if not ops:
        return s0
    if all(lab(p) == "A" and lab(q) == "A" for (p, q) in ops):
        return active_eprod(
            [(p - n_core, q - n_core) for (p, q) in ops], rdm1, dm2, dm3, t4
        )
    # A boundary generator that bottoms out resolves immediately.
    p, q = ops[-1]
    br = _bottom_right(p, q, lab)
    if br is not None:
        if br == 0.0:
            return 0.0
        return br * _G(ops[:-1], lab, n_core, rdm1, dm2, dm3, t4, s0)
    p0, q0 = ops[0]
    bl = _bottom_left(p0, q0, lab)
    if bl is not None:
        if bl == 0.0:
            return 0.0
        return bl * _G(ops[1:], lab, n_core, rdm1, dm2, dm3, t4, s0)
    # Otherwise commute a non-active generator to the edge where it bottoms:
    # left-bottoming ones to the front, right-bottoming ones to the back.
    for j in range(len(ops)):
        if _bottom_left(ops[j][0], ops[j][1], lab) is not None:
            return _commute_to_front(ops, j, lab, n_core, rdm1, dm2, dm3, t4, s0)
    for j in range(len(ops)):
        if _bottom_right(ops[j][0], ops[j][1], lab) is not None:
            return _commute_to_back(ops, j, lab, n_core, rdm1, dm2, dm3, t4, s0)
    raise RuntimeError(f"eprod_expect: no reducible generator in {ops}")


def _commute_to_front(ops, j, lab, n_core, rdm1, dm2, dm3, t4=None, s0=1.0):
    # Move E_xy = ops[j] left past ops[:j]:
    #   (∏_{k<j} E_{a_k}) E_xy = E_xy ∏ E_{a_k}
    #       + S_k (∏_{l<k} E_{a_l}) [E_{a_k}, E_xy] (∏_{l>k} E_{a_l})
    x, y = ops[j]
    left, rest = ops[:j], ops[j + 1:]
    total = _G([(x, y)] + left + rest, lab, n_core, rdm1, dm2, dm3, t4, s0)
    for k in range(len(left)):
        a, b = left[k]
        for coeff, op in _commutator(a, b, x, y):
            newops = left[:k] + [op] + left[k + 1:] + rest
            total += coeff * _G(newops, lab, n_core, rdm1, dm2, dm3, t4, s0)
    return total


def _commute_to_back(ops, j, lab, n_core, rdm1, dm2, dm3, t4=None, s0=1.0):
    # Move E_xy = ops[j] right past ops[j+1:]:
    #   E_xy (∏ E_{c_k}) = (∏ E_{c_k}) E_xy
    #       + S_k (∏_{l<k} E_{c_l}) [E_xy, E_{c_k}] (∏_{l>k} E_{c_l})
    x, y = ops[j]
    head, rest = ops[:j], ops[j + 1:]
    total = _G(head + rest + [(x, y)], lab, n_core, rdm1, dm2, dm3, t4, s0)
    for k in range(len(rest)):
        c, d = rest[k]
        for coeff, op in _commutator(x, y, c, d):
            newops = head + rest[:k] + [op] + rest[k + 1:]
            total += coeff * _G(newops, lab, n_core, rdm1, dm2, dm3, t4, s0)
    return total


def eprod_expect(ops, n_core, n_act, rdm1, dm2=None, dm3=None, t4=None, s0=1.0):
    """``<0| Ê_{p₁q₁} ... Ê_{pₖqₖ} |Φ>`` for generators with mixed I/A/S indices.

    ``ops`` is ``[(p,q), ...]`` in **full-orbital** indices (inactive
    ``< n_core``, active ``n_core ... n_core+n_act-1``, secondary above);
    ``rdm1`` / ``dm2`` / ``dm3`` are the active spin-summed RDMs
    (:func:`make_rdm123` convention).  Reduces the external indices via the
    commutator algebra above to active-only products evaluated by
    :func:`active_eprod`, so the cost is independent of the reference
    determinant count (the scaling win over the dense determinant FOIS).

    By default ``|Φ> = |0>`` (the diagonal expectation, ``s0 = <0|0> = 1`` and
    the RDMs the reference's own).  For a **transition** element
    ``<0| ... |Φ>`` (the F3 route, M26) pass the *transition* RDMs (``<0|Ê...|Φ>``,
    same :func:`make_rdm123` reduction) together with ``s0 = <0|Φ>``: the
    external-index reduction is identical (``|Φ>`` shares ``|0>``'s inactive /
    secondary occupation), only the active expectation and the empty-product
    overlap change.

    ``t4`` (the raw 4-generator product, active indices) is forwarded to
    :func:`active_eprod` for any active product that reaches four generators;
    only the diagonal H₀ block does, and only for small-space validation
    (production H₀ uses F3).
    """
    def lab(o):
        return _lab(o, n_core, n_act)

    return _G(list(ops), lab, n_core, rdm1, dm2, dm3, t4, s0)


def build_fois_by_case(P, n_core: int, n_act: int, n_frozen: int = 0):
    """Group the dense FOIS contracted functions by excitation case.

    Mirrors the candidate generation of :func:`._mrpt._ic_caspt2_solve`
    exactly (same ``Ê_pq`` singles, ``Ê_pq Ê_rs`` doubles, same CAS
    projection), but returns the survivors grouped by
    :func:`classify_case`:

        ``{case_key: [(state_dict, created, annih), ...], ...}``

    plus the bare ``W = H|0>`` and ``E0 = <0|F|0>`` the solve needs.  The
    determinant states are the *oracle* representation; M25/M26 replace each
    group's contribution with an RDM contraction validated against it.
    """
    norb, ref, h1, eri = P["norb"], P["ref"], P["h1"], P["eri"]
    W = _add(apply_1body(ref, h1, norb), apply_2body(ref, eri, norb))  # H|0>
    E0 = _dot(ref, apply_1body(ref, P["F"], norb))                     # <0|F|0>

    inact = list(range(n_frozen, n_core))
    sec = list(range(n_core + n_act, norb))
    actv = list(range(n_core, n_core + n_act))
    occ = inact + actv
    virt = actv + sec
    exc = [(p, q) for p in virt for q in occ if p != q]

    def applyE(state, p, q):
        e = np.zeros((norb, norb))
        e[p, q] = 1.0
        return apply_1body(state, e, norb)

    singles = {(p, q): applyE(ref, p, q) for (p, q) in exc}

    def is_cas(m):
        for ii in inact:
            if not ((m >> _so(ii, 0, norb)) & 1) or not (
                (m >> _so(ii, 1, norb)) & 1
            ):
                return False
        for aa in sec:
            if ((m >> _so(aa, 0, norb)) & 1) or ((m >> _so(aa, 1, norb)) & 1):
                return False
        return True

    groups: dict = {}

    def add(state, created, annih):
        pc = {m: v for m, v in state.items() if abs(v) > 1e-14 and not is_cas(m)}
        if not pc:
            return
        key = classify_case(created, annih, n_core, n_act, norb)
        groups.setdefault(key, []).append((pc, created, annih))

    for (p, q) in exc:
        add(singles[(p, q)], (p,), (q,))
    for (p, q) in exc:
        for (r, s) in exc:
            add(applyE(singles[(r, s)], p, q), (p, r), (q, s))

    return groups, W, E0


def assemble_dense(P, n_core: int, n_act: int, n_frozen: int = 0):
    """Build the case-ordered dense FOIS matrices (the M25 validation oracle).

    Generates the contracted functions grouped by case
    (:func:`build_fois_by_case`), concatenates them into one contracted
    basis ordered by active-index count, and returns the determinant-built
    overlap ``S``, zeroth-order ``H0_raw`` (``MᵀFM``, symmetrized) and RHS
    ``V_raw`` (``MᵀW``) together with the per-column case key and each
    case's column range.  The RDM-contracted forms of M25/M26 validate
    against these blocks element-wise.

    Returns a dict with ``S`` / ``H0_raw`` / ``V_raw`` / ``E0`` (numpy),
    ``col_case`` (length-n_cand list of case keys), ``ordered_cases`` and
    ``case_ranges`` (case key -> (start, stop) columns).
    """
    norb, F = P["norb"], P["F"]
    groups, W, E0 = build_fois_by_case(P, n_core, n_act, n_frozen)

    ordered_cases = sorted(groups, key=lambda k: (case_active_count(k), k))
    cands = []
    col_case = []
    case_ranges: dict = {}
    for key in ordered_cases:
        start = len(cands)
        for (state, _c, _a) in groups[key]:
            cands.append(state)
            col_case.append(key)
        case_ranges[key] = (start, len(cands))
    n_cand = len(cands)
    if n_cand == 0:
        z = np.zeros((0, 0))
        return dict(S=z, H0_raw=z, V_raw=np.zeros(0), E0=E0, col_case=[],
                    ordered_cases=ordered_cases, case_ranges=case_ranges)

    # Global determinant index + unit-normalized candidate matrix M, the
    # identical construction to the dense engine (clean overlap metric).
    det_idx: dict = {}
    rows, cols, vals = [], [], []
    for col, state in enumerate(cands):
        for m, v in state.items():
            j = det_idx.setdefault(m, len(det_idx))
            rows.append(j)
            cols.append(col)
            vals.append(v)
    n_dets = len(det_idx)
    M = sp.csc_matrix((vals, (rows, cols)), shape=(n_dets, n_cand))
    cnorm = np.sqrt(np.asarray(M.multiply(M).sum(axis=0)).ravel())
    cnorm[cnorm < 1e-300] = 1.0
    M = M @ sp.diags(1.0 / cnorm)

    S = (M.T @ M).toarray()

    idx2det = [0] * n_dets
    for m, j in det_idx.items():
        idx2det[j] = m
    fr, fc, fv = [], [], []
    for j, m in enumerate(idx2det):
        for m2, v in apply_1body({m: 1.0}, F, norb).items():
            j2 = det_idx.get(m2)
            if j2 is not None and abs(v) > 1e-14:
                fr.append(j2)
                fc.append(j)
                fv.append(v)
    Fmat = sp.csc_matrix((fv, (fr, fc)), shape=(n_dets, n_dets))
    H0_raw = (M.T @ (Fmat @ M)).toarray()
    H0_raw = 0.5 * (H0_raw + H0_raw.T)

    Wvec = np.zeros(n_dets)
    for m, v in W.items():
        j = det_idx.get(m)
        if j is not None:
            Wvec[j] = v
    V_raw = M.T @ Wvec

    return dict(S=S, H0_raw=H0_raw, V_raw=V_raw, E0=E0, col_case=col_case,
                ordered_cases=ordered_cases, case_ranges=case_ranges)


def overlap_matrix_from_rdms(P, n_core, n_act, rdm1, dm2, dm3, n_frozen=0,
                             cols=None):
    """Case-ordered FOIS overlap ``S`` built purely from the active RDMs.

    The M25 deliverable: each ``S_IJ = <0| Ê_I+ Ê_J |0>`` is evaluated by
    :func:`eprod_expect` (the mixed-index reducer -> :func:`active_eprod`),
    so the overlap never touches the reference determinant expansion, only
    the active spin-summed RDMs ``rdm1`` / ``dm2`` / ``dm3``
    (:func:`make_rdm123` convention).  This reproduces
    :func:`assemble_dense`'s ``S`` to machine precision (validated in
    ``tests/test_caspt2_cases.py``), confirming the overlap of every
    retained case (0- to 3-active, plus the three single<->double couplings)
    is a 1-/2-/3-RDM contraction; no case reaches the 4-RDM.

    By default the column set + ordering mirror :func:`assemble_dense` (via
    :func:`build_fois_by_case`) so the two ``S`` matrices are directly
    comparable.  Pass a precomputed ``cols`` five-tuple (e.g. from
    :func:`_columns_combinatorial`) to build over a determinant-free column
    list instead.  Columns are unit-normalized exactly as the dense engine.
    """
    col_ops, col_case, ordered_cases, case_ranges, cnorm = (
        cols if cols is not None
        else _column_ops_and_norms(P, n_core, n_act, rdm1, dm2, dm3, n_frozen)
    )
    n = len(col_ops)
    if n == 0:
        return dict(S=np.zeros((0, 0)), col_case=[],
                    ordered_cases=ordered_cases, case_ranges=case_ranges)

    def dag(ops):
        return [(q, p) for (p, q) in reversed(ops)]

    def ov(ops_i, ops_j):
        return eprod_expect(dag(ops_i) + ops_j, n_core, n_act, rdm1, dm2, dm3)

    S = np.zeros((n, n))
    for i in range(n):
        S[i, i] = ov(col_ops[i], col_ops[i]) / (cnorm[i] * cnorm[i])
        for j in range(i + 1, n):
            raw = ov(col_ops[i], col_ops[j])
            S[i, j] = S[j, i] = raw / (cnorm[i] * cnorm[j])
    return dict(S=S, col_case=col_case, ordered_cases=ordered_cases,
                case_ranges=case_ranges)


def _column_ops_and_norms(P, n_core, n_act, rdm1, dm2, dm3, n_frozen=0):
    """Shared column enumeration + unit norms for the RDM-built blocks.

    Returns ``(col_ops, col_case, ordered_cases, case_ranges, cnorm)`` with
    the same column set + ordering as :func:`assemble_dense` (so every
    RDM-built matrix is directly comparable to its dense counterpart).
    """
    groups, _W, _E0 = build_fois_by_case(P, n_core, n_act, n_frozen)
    ordered_cases = sorted(groups, key=lambda k: (case_active_count(k), k))
    col_ops, col_case, case_ranges = [], [], {}
    for key in ordered_cases:
        start = len(col_ops)
        for (_state, created, annih) in groups[key]:
            col_ops.append([(created[k], annih[k]) for k in range(len(created))])
            col_case.append(key)
        case_ranges[key] = (start, len(col_ops))

    def dag(ops):
        return [(q, p) for (p, q) in reversed(ops)]

    diag = np.array([
        eprod_expect(dag(c) + c, n_core, n_act, rdm1, dm2, dm3) for c in col_ops
    ]) if col_ops else np.zeros(0)
    cnorm = np.sqrt(np.where(diag > 1e-300, diag, 1.0))
    return col_ops, col_case, ordered_cases, case_ranges, cnorm


def _columns_combinatorial(P, n_core, n_act, rdm1, dm2, dm3, n_frozen=0,
                           tol=1e-12):
    """Determinant-free FOIS column enumeration + unit norms (M27).

    Builds the same retained singles ``Ê_pq|0>`` and doubles
    ``Ê_pq Ê_rs|0>`` as :func:`build_fois_by_case`, but purely from the
    excitation index tuples (``p,r`` virtual, ``q,s`` occupied), with the
    norm and the zero / internal filter coming from the RDM contraction
    (:func:`eprod_expect`) instead of the determinant expansion of |0>.  The
    result is the same five-tuple as :func:`_column_ops_and_norms` and spans
    the same FOIS, so the second-order energy is unchanged, but the cost no
    longer depends on the reference determinant count, the last dense-FOIS
    dependency in the RDM CASPT2 path.  Internal (all-active) columns and
    columns with vanishing norm (annihilated / linearly trivial) are dropped.
    """
    norb = P["norb"]
    inact = list(range(n_frozen, n_core))
    actv = list(range(n_core, n_core + n_act))
    sec = list(range(n_core + n_act, norb))
    occ, virt = inact + actv, actv + sec
    exc = [(p, q) for p in virt for q in occ if p != q]

    def dag(ops):
        return [(q, p) for (p, q) in reversed(ops)]

    by_case: dict = {}
    for (p, q) in exc:
        cand = [(p, q)]
        key = classify_case((p,), (q,), n_core, n_act, norb)
        if is_internal_case(key):
            continue
        nrm = eprod_expect(dag(cand) + cand, n_core, n_act, rdm1, dm2, dm3)
        if nrm > tol:
            by_case.setdefault(key, []).append((cand, nrm))
    for (p, q) in exc:
        for (r, s) in exc:
            cand = [(p, q), (r, s)]
            key = classify_case((p, r), (q, s), n_core, n_act, norb)
            if is_internal_case(key):
                continue
            nrm = eprod_expect(dag(cand) + cand, n_core, n_act, rdm1, dm2, dm3)
            if nrm > tol:
                by_case.setdefault(key, []).append((cand, nrm))

    ordered_cases = sorted(by_case, key=lambda k: (case_active_count(k), k))
    col_ops, col_case, case_ranges, diag = [], [], {}, []
    for key in ordered_cases:
        start = len(col_ops)
        for cand, nrm in by_case[key]:
            col_ops.append(cand)
            col_case.append(key)
            diag.append(nrm)
        case_ranges[key] = (start, len(col_ops))
    cnorm = np.sqrt(np.asarray(diag)) if diag else np.zeros(0)
    return col_ops, col_case, ordered_cases, case_ranges, cnorm


def rhs_from_rdms(P, n_core, n_act, rdm1, dm2, dm3, n_frozen=0, cols=None):
    """Case-ordered FOIS right-hand side ``V_I = <0| Ê_I+ Ĥ |0>`` from the RDMs.

    With ``Ĥ = S_pq h_pq E_pq + 1/2 S_pqrs (pq|rs)(E_pq E_rs - d_qr E_ps)``
    (chemist ``(pq|rs)``, the :func:`_semicanonical_prep` convention), every
    ``V_I`` is an integral-weighted sum of :func:`eprod_expect` calls, so the
    RHS is an RDM contraction with no determinant FOIS (M25).  Reproduces
    :func:`assemble_dense`'s ``V_raw`` to machine precision (validated in
    ``tests/test_caspt2_cases.py``).  Columns are unit-normalized exactly as
    the dense engine does; pass a precomputed ``cols`` five-tuple to build
    over a determinant-free column list.

    This is the correctness-first reference form: the Hamiltonian sum is the
    naive ``O(norb⁴)`` loop per column.  The production per-case integralxRDM
    contractions (which avoid that loop) are an M27 optimization; the math the
    optimization must reproduce is pinned here.
    """
    col_ops, col_case, ordered_cases, case_ranges, cnorm = (
        cols if cols is not None
        else _column_ops_and_norms(P, n_core, n_act, rdm1, dm2, dm3, n_frozen)
    )
    n = len(col_ops)
    if n == 0:
        return dict(V=np.zeros(0), col_case=[], ordered_cases=ordered_cases,
                    case_ranges=case_ranges)
    h1, eri, norb = P["h1"], P["eri"], P["norb"]
    rng = range(norb)
    h1nz = [(p, q, h1[p, q]) for p in rng for q in rng if abs(h1[p, q]) > 1e-13]
    erinz = [(p, q, r, s, eri[p, q, r, s])
             for p in rng for q in rng for r in rng for s in rng
             if abs(eri[p, q, r, s]) > 1e-13]

    def dag(ops):
        return [(q, p) for (p, q) in reversed(ops)]

    V = np.zeros(n)
    for i in range(n):
        bra = dag(col_ops[i])
        acc = 0.0
        for (p, q, h) in h1nz:
            acc += h * eprod_expect(bra + [(p, q)], n_core, n_act, rdm1, dm2, dm3)
        for (p, q, r, s, v) in erinz:
            t1 = eprod_expect(bra + [(p, q), (r, s)], n_core, n_act,
                              rdm1, dm2, dm3)
            t2 = (eprod_expect(bra + [(p, s)], n_core, n_act, rdm1, dm2, dm3)
                  if q == r else 0.0)
            acc += 0.5 * v * (t1 - t2)
        V[i] = acc / cnorm[i]
    return dict(V=V, col_case=col_case, ordered_cases=ordered_cases,
                case_ranges=case_ranges)


def zeroth_order_from_rdms(P, n_core, n_act, rdm1, dm2, dm3, t4, n_frozen=0):
    """Case-ordered FOIS zeroth order ``(H₀)_IJ = <0| Ê_I+ F̂ Ê_J |0>`` from RDMs.

    ``F̂ = S_pq F_pq E_pq`` is the generalized Fock (``P["F"]``, the CASPT2 H₀
    one-body operator).  Each element is ``S_pq F_pq`` of an
    :func:`eprod_expect` over ``Ê_I+ E_pq Ê_J``; for the 3-active cases the
    active product reaches four generators, so this needs the 4-RDM.

    **This is the small-space validation form: the raw 4-RDM ``t4``
    (``<ÊÊÊÊ>`` over active indices, O(n_act⁸)) must be supplied.**  It pins
    the exact quantity the F3 intermediate (M26) reproduces *without* building
    the 4-RDM: the 4-RDM enters H₀ only contracted with the diagonal Fock, so
    ``S_pq F_pq Γ4`` collapses to the 3-RDM-sized ``F3``.  Reproduces
    :func:`assemble_dense`'s ``H0_raw`` to machine precision.

    Columns are unit-normalized and the result symmetrized, exactly as the
    dense engine does.
    """
    col_ops, col_case, ordered_cases, case_ranges, cnorm = _column_ops_and_norms(
        P, n_core, n_act, rdm1, dm2, dm3, n_frozen
    )
    n = len(col_ops)
    if n == 0:
        return dict(H0=np.zeros((0, 0)), col_case=[],
                    ordered_cases=ordered_cases, case_ranges=case_ranges)
    F, norb = P["F"], P["norb"]
    rng = range(norb)
    Fnz = [(p, q, F[p, q]) for p in rng for q in rng if abs(F[p, q]) > 1e-13]

    def dag(ops):
        return [(q, p) for (p, q) in reversed(ops)]

    H0 = np.zeros((n, n))
    for i in range(n):
        bra = dag(col_ops[i])
        for j in range(n):
            ket = col_ops[j]
            acc = 0.0
            for (p, q, f) in Fnz:
                acc += f * eprod_expect(bra + [(p, q)] + ket, n_core, n_act,
                                        rdm1, dm2, dm3, t4=t4)
            H0[i, j] = acc / (cnorm[i] * cnorm[j])
    H0 = 0.5 * (H0 + H0.T)
    return dict(H0=H0, col_case=col_case, ordered_cases=ordered_cases,
                case_ranges=case_ranges)


# ── F3 intermediate: H₀ without ever building the 4-RDM (M26) ─────────────
# The 4-RDM enters H₀ only through the active-active Fock block, and only
# contracted with it.  Split F̂ = F̂_rest (everything with >= 1 external
# index) + F̂_act (active-active).  F̂_rest's contribution is a <= 3-RDM
# contraction (the external index always reduces).  For F̂_act use the
# operator identity (Fhat_act Hermitian, F symmetric):
#
#   <0| Ê_I+ F̂_act Ê_J |0>
#       = <0| Ê_I+ Ê_J |Φ> + <0| Ê_I+ [F̂_act, Ê_J] |0>,   |Φ> = F̂_act|0>.
#
# The first term is a *transition* FOIS overlap between <0| and |Φ>: |Φ> is
# one active-CI application of the active Fock, and the transition needs only
# the 1-/2-/3-particle *transition* RDMs <0|Ê...|Φ> (built by the same
# make_rdm123 reduction) plus the overlap <0|Φ>, never the 4-RDM.  The
# second term's commutator [F̂_act, Ê_J] lowers Ê_J's generator count, so it
# is a <= 3-RDM contraction too.  This is the F3 intermediate (the
# Fock-contracted 4-RDM collapses to 3-RDM size).  Validated against the
# explicit-4-RDM form (:func:`zeroth_order_from_rdms`) and the dense oracle.


# ── Determinant-dict RDM builders (work for ANY list, full or selected) ──
# make_rdm123 / make_rdm1234 / the in-place generator products in _rdm.py
# require a *closed* full-CAS determinant list (the chained Ê_rs Ê_tu
# intermediates assume closure under single excitations).  The
# selected-reference CASPT2 compose (M27a) needs the same 3-RDM (reference +
# transition) for a *truncated* selected-CI list.  The RDM of a truncated
# wavefunction |psi> is just <psi|Ê...Ê|psi> with the bra projecting back onto the
# kept space, so representing every intermediate as a dict over the full
# reachable active determinant set (in-list ∪ out-of-list) and contracting by
# dict overlap is exact for any list, with no closure assumption.  Bit-
# identical to make_rdm123 on a closed full-CAS list (every reachable
# determinant is in the list, so the dict path coincides with the array one).


def _ci_to_dict(ci, dets):
    out: dict = {}
    for i, det in enumerate(dets):
        v = float(ci[i])
        if v != 0.0:
            out[det] = out.get(det, 0.0) + v
    return out


def _apply_e_dict(p, q, state):
    """``Ê_pq |state>`` for ``state`` a dict ``{(a_occ, b_occ): coeff}``."""
    out: dict = {}
    for (a_occ, b_occ), c in state.items():
        sgn, na = _apply_aq_adp_spin(a_occ, q, p)
        if sgn:
            k = (na, b_occ)
            out[k] = out.get(k, 0.0) + sgn * c
        sgn, nb = _apply_aq_adp_spin(b_occ, q, p)
        if sgn:
            k = (a_occ, nb)
            out[k] = out.get(k, 0.0) + sgn * c
    return out


def _dot_dict(a, b):
    if len(b) < len(a):
        a, b = b, a
    return sum(c * b.get(k, 0.0) for k, c in a.items())


def _reorder123(rdm1, T2, T3, norb):
    """Generator products ``<Ê Ê>`` / ``<Ê Ê Ê>`` -> PySCF dm2 / dm3 (the
    identical Kronecker reorder :func:`make_rdm123` applies)."""
    dm2 = _reorder_rdm2(rdm1, T2)
    dm3 = T3.copy()
    for q in range(norb):
        dm3[:, q, q, :, :, :] -= dm2
        dm3[:, :, :, q, q, :] -= dm2
        dm3[:, q, :, :, q, :] -= dm2.transpose(0, 2, 3, 1)
        for s in range(norb):
            dm3[:, q, q, s, s, :] -= rdm1.T
    return dm2, dm3


def _gen_products123_dict(bra, ket, norb):
    """``(g1, T2, T3)`` generator products ``<bra|Ê...Ê|ket>`` over dicts.

    ``g1[r,s]=<bra|Ê_rs|ket>``, ``T2[p,q,r,s]=<bra|Ê_pq Ê_rs|ket>``,
    ``T3[...]=<bra|Ê_pq Ê_rs Ê_tu|ket>``.  Works for any (full or truncated)
    list; ``bra==ket`` gives a state RDM, ``bra!=ket`` a transition RDM.
    """
    Ek, EbT = {}, {}
    g1 = np.zeros((norb, norb))
    for r in range(norb):
        for s in range(norb):
            Ek[(r, s)] = _apply_e_dict(r, s, ket)        # Ê_rs|ket>
            EbT[(r, s)] = _apply_e_dict(s, r, bra)        # (Ê_rs)+|bra>
            g1[r, s] = _dot_dict(bra, Ek[(r, s)])
    T2 = np.zeros((norb,) * 4)
    for r in range(norb):
        for s in range(norb):
            v = Ek[(r, s)]
            for p in range(norb):
                for q in range(norb):
                    T2[p, q, r, s] = _dot_dict(EbT[(p, q)], v)
    T3 = np.zeros((norb,) * 6)
    for t in range(norb):
        for u in range(norb):
            v1 = Ek[(t, u)]
            for r in range(norb):
                for s in range(norb):
                    v2 = _apply_e_dict(r, s, v1)
                    for p in range(norb):
                        for q in range(norb):
                            T3[p, q, r, s, t, u] = _dot_dict(EbT[(p, q)], v2)
    return g1, T2, T3


def rdm123_any(ci, dets, norb):
    """``(rdm1, dm2, dm3)`` for ANY determinant list (full CAS or truncated).

    The selected-reference enabler (M27a): exact for a truncated selected-CI
    list (the RDMs of that truncated wavefunction) and bit-identical to
    :func:`make_rdm123` on a closed full-CAS list, with none of its
    closed-list requirement.
    """
    c = _ci_to_dict(ci, dets)
    g1, T2, T3 = _gen_products123_dict(c, c, norb)
    dm2, dm3 = _reorder123(g1, T2, T3, norb)
    return g1, dm2, dm3


def _transition_rdms123(bra_dict, phi_dict, n_act):
    """Transition (rdm1, dm2, dm3) for ``<0|Ê...|Φ>`` in make_rdm123 convention.

    ``bra_dict`` / ``phi_dict`` are the reference and ``|Φ> = F̂_act|0>`` as
    determinant dicts.  Dict generator products, so it is exact for full *and*
    truncated (selected-CI) lists; ``|Φ>`` may carry determinants outside the
    reference list (the dict spans them).  O(n_act⁶) like a 3-RDM; the 4-RDM
    is never formed.
    """
    na = n_act
    tr1, T2, T3 = _gen_products123_dict(bra_dict, phi_dict, na)
    dm2 = _reorder_rdm2(tr1, T2)
    dm3 = T3.copy()
    for q in range(na):
        dm3[:, q, q, :, :, :] -= dm2
        dm3[:, :, :, q, q, :] -= dm2
        dm3[:, q, :, :, q, :] -= dm2.transpose(0, 2, 3, 1)
        for s in range(na):
            dm3[:, q, q, s, s, :] -= tr1.T
    return tr1, dm2, dm3


def _commute_fact(ops, F, n_core, n_act):
    """``[F̂_act, ∏ ops]`` as ``[(coeff, opstring), ...]`` (F̂_act = active F).

    ``[F̂_act, E_pq] = [pinA] S_t F_{t,p} E_{t,q} - [qinA] S_u F_{q,u} E_{p,u}``
    (active ``t,u``), extended over the product by the Leibniz rule.
    """
    a0, a1 = n_core, n_core + n_act

    def comm_single(p, q):
        out = []
        if a0 <= p < a1:
            for t in range(a0, a1):
                if abs(F[t, p]) > 1e-14:
                    out.append((float(F[t, p]), (t, q)))
        if a0 <= q < a1:
            for u in range(a0, a1):
                if abs(F[q, u]) > 1e-14:
                    out.append((-float(F[q, u]), (p, u)))
        return out

    terms = []
    for k in range(len(ops)):
        for coeff, newop in comm_single(*ops[k]):
            terms.append((coeff, ops[:k] + [newop] + ops[k + 1:]))
    return terms


def zeroth_order_from_rdms_f3(P, n_core, n_act, rdm1, dm2, dm3,
                              ci_active, dets_active, n_frozen=0, cols=None):
    """Case-ordered FOIS ``H0`` from the RDMs **without building the 4-RDM**.

    The M26 deliverable.  Same result as :func:`zeroth_order_from_rdms` (and
    the dense ``H0_raw``) but the active-active Fock contribution is taken via
    the F3 / transition-RDM identity documented above instead of the raw
    4-RDM, so only 1-/2-/3-particle RDMs (reference and transition) are ever
    formed.  ``ci_active`` / ``dets_active`` are the active-space reference CI
    vector + determinant list (for ``|Φ> = F̂_act|0>`` and the transition
    RDMs); ``rdm1`` / ``dm2`` / ``dm3`` are the reference's own active RDMs.
    Pass a precomputed ``cols`` five-tuple to build over a determinant-free
    column list.
    """
    col_ops, col_case, ordered_cases, case_ranges, cnorm = (
        cols if cols is not None
        else _column_ops_and_norms(P, n_core, n_act, rdm1, dm2, dm3, n_frozen)
    )
    n = len(col_ops)
    if n == 0:
        return dict(H0=np.zeros((0, 0)), col_case=[],
                    ordered_cases=ordered_cases, case_ranges=case_ranges)
    F, norb = P["F"], P["norb"]
    a0, a1 = n_core, n_core + n_act

    # F̂_rest: F with the active-active block removed (every element keeps >= 1
    # external index, so its contribution stays at <= 3-RDM).
    Frnz = [(p, q, F[p, q]) for p in range(norb) for q in range(norb)
            if abs(F[p, q]) > 1e-13 and not (a0 <= p < a1 and a0 <= q < a1)]

    # |Φ> = F̂_act|0> (one active-CI application) + the transition RDMs.
    # Dict-based so it is exact for a *selected* (truncated) reference too:
    # |Φ> can carry determinants outside the reference list, which the dict
    # spans (a vector over the reference list would silently drop them).
    bra = _ci_to_dict(ci_active, dets_active)
    phi: dict = {}
    for t in range(n_act):
        for u in range(n_act):
            f = F[a0 + t, a0 + u]
            if abs(f) > 1e-14:
                for k, v in _apply_e_dict(t, u, bra).items():
                    phi[k] = phi.get(k, 0.0) + f * v
    s0 = _dot_dict(bra, phi)  # <0|Φ> = <0|F̂_act|0>
    tr1, tdm2, tdm3 = _transition_rdms123(bra, phi, n_act)

    def dag(ops):
        return [(q, p) for (p, q) in reversed(ops)]

    H0 = np.zeros((n, n))
    for i in range(n):
        bra = dag(col_ops[i])
        for j in range(n):
            ket = col_ops[j]
            acc = 0.0
            # F̂_rest contribution (no 4-RDM)
            for (p, q, f) in Frnz:
                acc += f * eprod_expect(bra + [(p, q)] + ket, n_core, n_act,
                                        rdm1, dm2, dm3)
            # F̂_act contribution: transition overlap + commutator
            acc += eprod_expect(bra + ket, n_core, n_act, tr1, tdm2, tdm3, s0=s0)
            for coeff, string in _commute_fact(ket, F, n_core, n_act):
                acc += coeff * eprod_expect(bra + string, n_core, n_act,
                                            rdm1, dm2, dm3)
            H0[i, j] = acc / (cnorm[i] * cnorm[j])
    H0 = 0.5 * (H0 + H0.T)
    return dict(H0=H0, col_case=col_case, ordered_cases=ordered_cases,
                case_ranges=case_ranges)


def overlap_case_structure(P, n_core: int, n_act: int, n_frozen: int = 0,
                           tol: float = 1e-9):
    """Off-diagonal case pairs that overlap in the FOIS metric ``S``.

    The dense overlap is block-diagonal in the excitation case **except**
    for the single<->double redundancies where a double with an active
    spectator reduces to a net single (``Ê_{a,t}Ê_{t,i}`` overlapping
    ``Ê_{a,i}``): the pairs ``{S<-I, AS<-AI}``, ``{A<-I, AA<-AI}``,
    ``{S<-A, AS<-AA}``.  Those are the only inter-case couplings the RDM
    contraction (M25/M26) must treat jointly; every other case block is
    independent.  Returns the set of frozenset case-pairs whose
    cross-block ``max|S|`` exceeds ``tol``.
    """
    bundle = assemble_dense(P, n_core, n_act, n_frozen)
    S = bundle["S"]
    col_case = bundle["col_case"]
    cases = bundle["ordered_cases"]
    idx = {k: np.array([c == k for c in col_case]) for k in cases}
    coupled = set()
    for a in range(len(cases)):
        for b in range(a + 1, len(cases)):
            blk = S[np.ix_(idx[cases[a]], idx[cases[b]])]
            if blk.size and np.abs(blk).max() > tol:
                coupled.add(frozenset((cases[a], cases[b])))
    return coupled


def caspt2_corr_cases(P, n_core, n_act, n_frozen=0, ipea=0.0, imaginary=0.0,
                      thresh=1e-8):
    """Case-partitioned IC-CASPT2 second-order energy (M24 scaffold).

    Assembles the FOIS overlap ``S``, zeroth-order ``H₀`` and RHS ``V`` from
    the per-case blocks (concatenating the cases into the global contracted
    basis, so ``S`` / ``H₀`` carry an explicit case-block structure) and
    solves the first-order equation exactly as the dense engine does.  Built
    on the determinant representation of each case for now, so it reproduces
    :func:`._mrpt._ic_caspt2_solve` to machine precision; the block structure
    is what M25/M26 RDM-contract.  IPEA is intentionally unsupported here;
    its canonical eight-class contraction is implemented by the explicit
    ``engine='auto'`` path.
    """
    if ipea != 0.0:
        raise NotImplementedError(
            "case-partitioned CASPT2 does not implement the IPEA shift; "
            "use engine='auto' for the canonical eight-class IPEA "
            "contraction"
        )
    bundle = assemble_dense(P, n_core, n_act, n_frozen)
    case_ranges = bundle["case_ranges"]
    S, H0_raw, V_raw, E0 = (
        bundle["S"], bundle["H0_raw"], bundle["V_raw"], bundle["E0"]
    )
    e_corr, nb = _solve_caspt2(S, H0_raw, V_raw, E0, imaginary=imaginary,
                               thresh=thresh)
    return e_corr, nb, case_ranges


def _solve_caspt2(S, H0, V, E0, imaginary=0.0, thresh=1e-8):
    """Forsberg-Malmqvist Hylleraas second-order energy from S / H0 / V / E0.

    Orthonormalizes the FOIS (drops near-linear-dependent directions), then
    sums the second-order energy in the H0 eigenbasis.  Identical to the dense
    engine's solve; shared by :func:`caspt2_corr_cases` (dense blocks) and
    :func:`caspt2_corr_rdm` (RDM blocks).
    """
    if S.shape[0] == 0:
        return 0.0, 0
    w, U = eigh(S)
    keep = w > thresh
    if not np.any(keep):
        return 0.0, 0
    T = U[:, keep] / np.sqrt(w[keep])
    nb = int(keep.sum())
    d, Q = eigh(T.T @ H0 @ T - E0 * np.eye(nb))
    Vp = Q.T @ (T.T @ V)
    if imaginary != 0.0:
        s2 = imaginary**2
        e = float(-np.sum(Vp**2 * d * (d**2 + 2 * s2) / (d**2 + s2) ** 2))
    else:
        e = float(-np.sum(Vp**2 / d))
    return e, nb


def caspt2_corr_rdm(P, n_core, n_act, ci_active, dets_active, rdms=None,
                    n_frozen=0, imaginary=0.0, thresh=1e-8,
                    determinant_free=True):
    """Case-partitioned IC-CASPT2 second-order energy, fully from the RDMs.

    The M27 capstone: assembles the FOIS overlap ``S``, RHS ``V`` and
    zeroth-order ``H0`` (the last via the F3 intermediate, no 4-RDM) entirely
    from the active reduced density matrices and solves the first-order
    equation, reproducing :func:`._mrpt._ic_caspt2_solve` (the OpenMolcas-pinned
    dense engine) to machine precision.

    With ``determinant_free=True`` (default) the FOIS columns are enumerated
    combinatorially (:func:`_columns_combinatorial`) from the excitation index
    tuples, so **nothing** in the energy touches the reference determinant
    expansion, the cost is set by the active RDMs alone.  Set it ``False`` to
    enumerate columns via the dense :func:`build_fois_by_case` instead (the
    direct-comparison path used by the validation tests).

    ``ci_active`` / ``dets_active`` are the active-space reference CI vector +
    determinant list (needed for ``|Φ> = F̂_act|0>`` and the F3 transition
    RDMs).  ``rdms = (rdm1, dm2, dm3)`` may be supplied; otherwise they are
    built with :func:`make_rdm123` from the reference (a full-CAS list).
    IPEA is intentionally unsupported (the dense engine keeps that path).

    Composing with a *selected* reference (the route to large active spaces)
    additionally needs selected-list 3-RDM + transition-3-RDM builders; the
    energy machinery here is reference-agnostic once those RDMs are provided.
    The ``V`` / ``H0`` Hamiltonian sums are still the correctness-first
    O(norb⁴) reference forms (per-case integralxRDM contractions are the
    remaining throughput optimization).
    """
    if rdms is None:
        from ._rdm import make_rdm123
        try:
            rdm1, dm2, dm3 = make_rdm123(ci_active, dets_active, n_act)
        except NotImplementedError:
            # truncated / selected-CI reference: make_rdm123 needs a closed
            # full-CAS list; the dict builder is exact for any list (M27a).
            rdm1, dm2, dm3 = rdm123_any(ci_active, dets_active, n_act)
    else:
        rdm1, dm2, dm3 = rdms
    cols = (_columns_combinatorial(P, n_core, n_act, rdm1, dm2, dm3, n_frozen)
            if determinant_free else None)
    S = overlap_matrix_from_rdms(P, n_core, n_act, rdm1, dm2, dm3, n_frozen,
                                 cols=cols)["S"]
    V = rhs_from_rdms(P, n_core, n_act, rdm1, dm2, dm3, n_frozen,
                      cols=cols)["V"]
    H0 = zeroth_order_from_rdms_f3(P, n_core, n_act, rdm1, dm2, dm3,
                                   ci_active, dets_active, n_frozen,
                                   cols=cols)["H0"]
    F = P["F"]
    E0 = 2.0 * float(sum(F[i, i] for i in range(n_core))) + float(
        np.einsum("tu,tu->", F[n_core:n_core + n_act, n_core:n_core + n_act],
                  rdm1)
    )
    e_corr, nb = _solve_caspt2(S, H0, V, E0, imaginary=imaginary, thresh=thresh)
    return e_corr, nb


def caspt2_corr_rdm_matfree(P, n_core, n_act, ci_active, dets_active, rdms=None,
                            n_frozen=0, rtol=1e-10, maxiter=10000):
    """Matrix-free iterative CASPT2 second-order energy (M27b).

    Solves the first-order amplitude equation ``(H0 - E0 S) x = -V`` with
    MINRES using only the matrix-free :func:`apply_H0` / :func:`apply_S`
    sigma products (never forming ``S`` or ``H0``), then ``e2 = V.x``.  The
    FOIS overlap is singular (linear dependencies), but ``V`` has no component
    on its null space and ``H0``/``S`` annihilate it, so MINRES from ``x0=0``
    stays in the non-null space and the energy is well-defined.  Reproduces
    :func:`caspt2_corr_rdm` (hence the dense ``_ic_caspt2_solve``) to machine
    precision; composes with a selected reference like
    :func:`caspt2_corr_rdm`.

    Unshifted only (imaginary shift needs the H0 eigenbasis or a regularized
    operator; out of scope here).  Cost is still set by the per-element terms
    B/C inside :func:`apply_H0` until those are delta-collapsed; the solver
    framework is what this lands.
    """
    from scipy.sparse.linalg import LinearOperator, minres

    if rdms is None:
        from ._rdm import make_rdm123
        try:
            rdm1, dm2, dm3 = make_rdm123(ci_active, dets_active, n_act)
        except NotImplementedError:
            rdm1, dm2, dm3 = rdm123_any(ci_active, dets_active, n_act)
    else:
        rdm1, dm2, dm3 = rdms
    cols = _columns_combinatorial(P, n_core, n_act, rdm1, dm2, dm3, n_frozen)
    V = rhs_from_rdms(P, n_core, n_act, rdm1, dm2, dm3, n_frozen, cols=cols)["V"]
    n = len(V)
    if n == 0:
        return 0.0, 0
    F = P["F"]
    E0 = 2.0 * float(sum(F[i, i] for i in range(n_core))) + float(
        np.einsum("tu,tu->", F[n_core:n_core + n_act, n_core:n_core + n_act],
                  rdm1)
    )

    def mv(x):
        return (apply_H0(x, cols, P, n_core, n_act, rdm1, dm2, dm3,
                         ci_active, dets_active)
                - E0 * apply_S(x, cols, n_core, n_act, rdm1, dm2, dm3))

    M = LinearOperator((n, n), matvec=mv, dtype=float)
    x, info = minres(M, -V, rtol=rtol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f"caspt2_corr_rdm_matfree: MINRES did not converge "
                           f"(info={info})")
    return float(V @ x), n


# ── Matrix-free FOIS sigma routines (M27b) ───────────────────────────────
# The full n_cand x n_cand S / H0 build is the orbital^8 wall (n_cand ~
# (n_occ n_virt)^2).  A genuinely scalable CASPT2 never forms those matrices:
# it solves (H0 - E0 S) x = -V iteratively, needing only the matrix-vector
# products S.x and H0.x ("sigma" routines).  These exploit the case-block
# structure: S couples only the three single<->double pairs (block-diagonal
# otherwise, overlap_case_structure), and within each (case or coupling) block
# <Ê_i+ Ê_j> is nonzero only when i,j share the same external (inactive-hole +
# secondary-particle) index signature.  So _overlap_sigma_raw groups columns
# by external multiset and contracts each small same-external sub-block (the
# active-RDM contraction, O((orderings x n_act^k)^2)) -- the external indices
# never enter a double sum, giving O(n_act^k x n_external) not n_cand^2.  The
# eprod_expect reducer evaluates each sub-block element (with hand-coded
# closed-form fast paths for the 0-/1-active singles).  Validated == the
# full-matrix S.x / H0.x.

# The three S single<->double couplings (a single and the double that reduces
# to it via an active spectator pair Ê_{a,t}Ê_{t,i} -> Ê_{a,i}); structurally
# fixed, so they need no dense probe to discover.
_S_COUPLINGS = (
    (("S", "I"), ("AS", "AI")),
    (("A", "I"), ("AA", "AI")),
    (("S", "A"), ("AS", "AA")),
)


def _case_members(cols):
    """{case_key: [column indices]} from a column bundle."""
    col_case = cols[1]
    members: dict = {}
    for i, k in enumerate(col_case):
        members.setdefault(k, []).append(i)
    return members


def _overlap_sigma_raw(y, cols, n_core, n_act, rdm1, dm2, dm3, s0=1.0):
    """Raw overlap-structured application ``r_I = S_J <0|Ê_I+ Ê_J|Φ> y_J``.

    Sums only the nonzero case blocks (each diagonal block + the three
    single<->double couplings); ``|Φ> = |0>`` with the reference RDMs and
    ``s0 = 1`` gives the bare overlap, while the *transition* RDMs +
    ``s0 = <0|Φ>`` give the H₀ transition term ``<0|Ê_I+ Ê_J|Φ>`` (same
    operator structure, hence the same block sparsity).  No ``cnorm``: works
    on raw amplitudes ``y`` and returns raw results.
    """
    col_ops, _col_case, ordered_cases, _ranges, _cnorm = cols
    n = len(col_ops)
    raw = np.zeros(n)
    if n == 0:
        return raw
    members = _case_members(cols)
    opmap = {tuple(c): k for k, c in enumerate(map(tuple, col_ops))}

    def dag(ops):
        return [(q, p) for (p, q) in reversed(ops)]

    def collapsed_diag(key):
        # 0-active diagonal blocks via the validated closed forms (the only
        # active dependence is the scalar overlap s0): O(n_case) not n_case^2.
        if key == ("S", "I"):                     # <E_ai|E_a'i'> = 2 d d
            for i in members.get(key, []):
                raw[i] += s0 * 2.0 * y[i]
            return True
        if key == ("SS", "II"):                   # closed-shell doubles metric
            for i in members.get(key, []):
                (a, ii), (b, jj) = col_ops[i]
                v = 4.0 * y[i]
                for coeff, perm in ((4.0, ((b, jj), (a, ii))),
                                    (-2.0, ((b, ii), (a, jj))),
                                    (-2.0, ((a, jj), (b, ii)))):
                    k = opmap.get(perm)
                    if k is not None:
                        v += coeff * y[k]
                raw[i] += s0 * v
            return True
        # 1-active singles: the secondary/inactive index is a spectator delta,
        # the active label contracts the 1-RDM block (g for the bare overlap,
        # the transition 1-RDM for |Φ>).  O(n_case x n_act).
        actv = range(n_core, n_core + n_act)
        if key == ("S", "A"):     # <E_at|E_a't'> = d_aa' g_{tt'} (particle)
            for i in members.get(key, []):
                (a, t), = col_ops[i]
                v = 0.0
                for tp in actv:
                    k = opmap.get(((a, tp),))
                    if k is not None:
                        v += rdm1[t - n_core, tp - n_core] * y[k]
                raw[i] += v
            return True
        if key == ("A", "I"):     # <E_ti|E_t'i'> = d_ii' (2 s0 d_tt' - g_{t't})
            for i in members.get(key, []):
                (t, ii), = col_ops[i]
                v = 0.0
                for tp in actv:
                    k = opmap.get(((tp, ii),))
                    if k is not None:
                        hole = (2.0 * s0 if t == tp else 0.0) \
                            - rdm1[tp - n_core, t - n_core]
                        v += hole * y[k]
                raw[i] += v
            return True
        return False

    def _extkey(i):
        ext = []
        for (p, q) in col_ops[i]:
            if q < n_core:                           # inactive hole
                ext.append(q)
            if p >= n_core + n_act:                  # secondary particle
                ext.append(p)
        return tuple(sorted(ext))

    def collapsed_block_generic(rows_case, cols_case):
        # <Ê_i+ Ê_j> is nonzero only when i,j share the same external
        # (inactive-hole + secondary-particle) index signature, so any block
        # (diagonal OR a single<->double coupling) decomposes into independent
        # sub-blocks of columns that differ only in active labels -- matched by
        # external multiset across the two cases.  Each sub-block is
        # O((orderings x n_act^k)^2) reducer calls (the active-RDM contraction),
        # never an n_case^2 double sum over the external indices.
        rows = members.get(rows_case, [])
        cs = members.get(cols_case, [])
        if not rows or not cs:
            return
        cgroups: dict = {}
        for j in cs:
            cgroups.setdefault(_extkey(j), []).append(j)
        for i in rows:
            g = cgroups.get(_extkey(i))
            if not g:
                continue
            bi = dag(col_ops[i])
            acc = 0.0
            for j in g:
                acc += y[j] * eprod_expect(
                    bi + col_ops[j], n_core, n_act, rdm1, dm2, dm3, s0=s0)
            raw[i] += acc

    for key in ordered_cases:           # diagonal case blocks
        if not collapsed_diag(key):
            collapsed_block_generic(key, key)
    present = set(ordered_cases)
    for sgl, dbl in _S_COUPLINGS:       # the only off-diagonal couplings
        if sgl in present and dbl in present:
            collapsed_block_generic(sgl, dbl)
            collapsed_block_generic(dbl, sgl)
    return raw


def apply_S(x, cols, n_core, n_act, rdm1, dm2, dm3):
    """Matrix-free FOIS overlap application ``(S x)`` (M27b).

    ``cols`` is the column bundle ``(col_ops, col_case, ordered_cases,
    case_ranges, cnorm)`` (from :func:`_columns_combinatorial` or
    :func:`_column_ops_and_norms`).  Computes ``S @ x`` without materializing
    ``S`` by summing only the nonzero case blocks: each case's diagonal block
    plus the three single<->double couplings.  Reproduces ``S_full @ x``
    exactly (validated), at the eventual cost of the per-block contractions
    rather than ``n_cand**2``.

    Columns are unit-normalized as elsewhere: with raw overlaps ``r_IJ`` and
    norms ``c_I``, ``S_IJ = r_IJ/(c_I c_J)``, so ``(S x)_I = (1/c_I) S_J r_IJ
    (x_J/c_J)``, so work in ``y = x/cnorm``, apply the raw blocks, divide by
    ``cnorm``.
    """
    cnorm = cols[4]
    if len(cols[0]) == 0:
        return np.zeros(0)
    y = np.asarray(x, dtype=float) / cnorm
    raw = _overlap_sigma_raw(y, cols, n_core, n_act, rdm1, dm2, dm3)
    return raw / cnorm


def apply_H0(x, cols, P, n_core, n_act, rdm1, dm2, dm3, ci_active, dets_active):
    """Matrix-free FOIS zeroth-order application ``(H0 x)`` (M27b).

    ``H0_IJ = <Ê_I+ F̂ Ê_J>`` with ``F̂`` the generalized Fock (``P["F"]``).
    Reproduces ``zeroth_order_from_rdms_f3``'s ``H0 @ x`` without materializing
    ``H0``, via the F3 split ``F̂ = F̂_act + F̂_rest``:

    * **term A** (``F̂_act`` transition): ``S_J <Ê_I+ Ê_J|Φ> y_J``,
      ``|Φ> = F̂_act|0>``: the block-structured :func:`_overlap_sigma_raw`
      with the transition RDMs + ``s0 = <0|Φ>``.
    * **term B** (``F̂_act`` commutator): ``S_J <Ê_I+ [F̂_act, Ê_J]> y_J``,
      ``[F̂_act, Ê_J]`` via :func:`_commute_fact`, evaluated per element (the
      commutator can produce diagonal active ops ``E_tt`` that are not FOIS
      columns, so it is not a pure amplitude transform).
    * **term C** (``F̂_rest``): ``S_J S_{pqinrest} F_pq <Ê_I+ E_pq Ê_J> y_J``.
      Its *diagonal* external part (the orbital-energy denominators) collapses
      to a single block-structured sigma on the e-weighted amplitude
      ``wodoty`` (``E_pp`` a number operator); its *off-diagonal* external Fock
      (core-active / active-virtual) is collapsed by external-offset grouping
      (matched (I,J) only), ``O(|Foff| x n x n_act^k)`` not ``|Foff| x n^2``.

    Reproduces ``H0 @ x`` exactly (validated).  Every term is now collapsed to
    ``O(poly(orbitals) x n_cand)`` or better -- no ``n_cand^2`` matvec term --
    so the active-RDM contractions (still evaluated through the Python
    :func:`eprod_expect`) are the remaining constant factor (a C++ inner
    kernel is the next step).
    """
    col_ops = cols[0]
    cnorm = cols[4]
    n = len(col_ops)
    if n == 0:
        return np.zeros(0)
    y = np.asarray(x, dtype=float) / cnorm
    F = P["F"]
    a0, a1 = n_core, n_core + n_act

    def dag(ops):
        return [(q, p) for (p, q) in reversed(ops)]

    # term A: F̂_act transition overlap (block-structured)
    bra = _ci_to_dict(ci_active, dets_active)
    phi: dict = {}
    for t in range(n_act):
        for u in range(n_act):
            f = F[a0 + t, a0 + u]
            if abs(f) > 1e-14:
                for k, v in _apply_e_dict(t, u, bra).items():
                    phi[k] = phi.get(k, 0.0) + f * v
    s0 = _dot_dict(bra, phi)
    tr1, tdm2, tdm3 = _transition_rdms123(bra, phi, n_act)
    raw = _overlap_sigma_raw(y, cols, n_core, n_act, tr1, tdm2, tdm3, s0=s0)

    # term C, diagonal external Fock (the orbital-energy denominators): E_pp
    # is a number operator, so <Ê_I+ E_pp Ê_J> = (occupation of p in Ê_J)
    # S_raw_IJ.  The whole diagonal-external contribution therefore collapses
    # to one block-structured overlap sigma on the orbital-energy-weighted
    # amplitude wodoty, w(J) = 2 S_i e_i + S_{sec created in J} e - S_{inact
    # annih in J} e (the MP2-like denominator carried by column J).  Core-core
    # and virtual-virtual blocks are diagonal in the semicanonical basis, so
    # this captures all of them.
    eps = np.diag(F)
    core_sum = 2.0 * float(np.sum(eps[:n_core]))
    w = np.full(n, core_sum)
    for j in range(n):
        for (p, q) in col_ops[j]:
            if p >= a1:                      # secondary particle created
                w[j] += eps[p]
            if q < a0:                       # inactive hole annihilated
                w[j] -= eps[q]
    raw += _overlap_sigma_raw(w * y, cols, n_core, n_act, rdm1, dm2, dm3)

    # term B (F̂_act commutator): <Ê_I+ [F̂_act,Ê_J]> with
    # [F̂_act,Ê_J] = S_m c_m Ê_{K_m} (_commute_fact).  Most K_m are FOIS
    # columns, so they collapse to an overlap sigma on the transformed
    # amplitude S_K (S_{J,m:K_m=K} c_m y_J) <Ê_I+ Ê_K>.  But the commutator
    # can also emit a *diagonal-active* op E_tt times an external excitation
    # (e.g. E_tt E_ai): that is a legitimate, nonzero perturber but is NOT an
    # enumerated column (columns exclude p==q), so those strings are collected
    # and applied per element.  Internal / zero-norm K_m contract to zero
    # against the always-external bra Ê_I and need no handling.  (For a
    # stationary CASSCF reference the off-diagonal external Fock below
    # vanishes, leaving only block-structured sigmas + this small E_tt set.)
    opmap = {tuple(c): k for k, c in enumerate(map(tuple, col_ops))}
    yB = np.zeros(n)
    dropped: dict = {}
    for j in range(n):
        yj = y[j]
        if yj == 0.0:
            continue
        for coeff, kops in _commute_fact(col_ops[j], F, n_core, n_act):
            key = tuple(kops)
            k = opmap.get(key)
            if k is not None:
                yB[k] += coeff * yj
            else:
                dropped[key] = dropped.get(key, 0.0) + coeff * yj
    raw += _overlap_sigma_raw(yB, cols, n_core, n_act, rdm1, dm2, dm3)
    for key, wt in dropped.items():
        if abs(wt) < 1e-14:
            continue
        string = list(key)
        for i in range(n):
            raw[i] += wt * eprod_expect(
                dag(col_ops[i]) + string, n_core, n_act, rdm1, dm2, dm3)

    # term C off-diagonal external Fock (core-active / active-virtual; the
    # core-virtual block vanishes at CASSCF convergence by the generalized
    # Brillouin theorem but the active-coupled blocks do NOT, so this runs for
    # every reference).  <Ê_I+ E_pq Ê_J> is nonzero only when the external
    # (inactive-hole + secondary-particle) signature of I equals that of J
    # *shifted by E_pq's external change* (a+_p adds a secondary / fills an
    # inactive hole; a_q removes a secondary / opens an inactive hole), so
    # group columns by external signature and visit only the matched (I,J):
    # O(|Foff| x n x n_act^k) instead of |Foff| x n^2.
    rng = range(P["norb"])
    Foff = [(p, q, F[p, q]) for p in rng for q in rng
            if abs(F[p, q]) > 1e-13 and p != q
            and not (a0 <= p < a1 and a0 <= q < a1)]
    if Foff:
        def extsig(ops):
            ext = []
            for (pp, qq) in ops:
                if qq < a0:
                    ext.append(qq)
                if pp >= a1:
                    ext.append(pp)
            return tuple(sorted(ext))

        ext_of = [extsig(col_ops[i]) for i in range(n)]
        groups: dict = {}
        for i in range(n):
            groups.setdefault(ext_of[i], []).append(i)
        for (p, q, f) in Foff:
            add, rem = [], []
            if p >= a1:
                add.append(p)            # secondary particle created
            elif p < a0:
                rem.append(p)            # inactive hole filled
            if q < a0:
                add.append(q)            # inactive hole opened
            elif q >= a1:
                rem.append(q)            # secondary particle removed
            for j in range(n):
                yj = y[j]
                if yj == 0.0:
                    continue
                target = list(ext_of[j])
                ok = True
                for r in rem:
                    if r in target:
                        target.remove(r)
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                target = tuple(sorted(target + add))
                for i in groups.get(target, ()):
                    raw[i] += f * yj * eprod_expect(
                        dag(col_ops[i]) + [(p, q)] + col_ops[j],
                        n_core, n_act, rdm1, dm2, dm3)
    return raw / cnorm
