"""Local DLPNO-(T) on converged local DLPNO-CCSD amplitudes (M3c).

The perturbative triples correction evaluated per occupied triple in a
*triple natural orbital* (TNO) domain, reusing the spatial closed-shell
per-triple kernel (native `_vibeqc_core.dlpno_spatial_triple_energy`, with
`_ccsd_cs.cs_triple_energy` as the Python oracle/fallback). Both are validated
to machine precision against the FCI-anchored spin-orbital reference
`_ccsd_ref.so_triples_correction`. This is the (T) analogue of the cs-reuse
move that built the local CCSD solver: rather than hand-transcribe a
spin-adapted (T), drive the validated spatial kernel on a small per-triple
local space.

Locality: the spatial (T) sums over occupied triples (i,j,k); each triple
gets a virtual domain -- the orthonormalised union of the pair PNO spaces
of its distinct occupieds (the TNO precursor, Riplinger, Sandhoefer,
Hansen & Neese, J. Chem. Phys. 139, 134101 (2013)). Amplitudes and DF
integrals are projected into that domain; the connected term's occupied
sum `m` runs over all occupieds (the locality is in the virtual/TNO
domain). Triples are additionally **screened by occupied locality**: a
triple is evaluated only when its three occupieds are mutually within
`coupling_radius` (a distant member couples through a vanishing pair
amplitude) -- O(N^3)->O(N) triples on extended systems, exact at
`coupling_radius=0`.

Exactness anchor: at full PNO domains *and* no TNO truncation every domain
spans the full virtual space, so each triple's contribution is the exact
one and the sum reproduces canonical CCSD(T) -- the (T) analogue of the
CCSD parity ratchet. With truncated TNO domains the dropped cross-domain
pieces are the controlled DLPNO-(T) domain approximation. Occupied
energies in the denominators are the diagonal (semicanonical, "T0")
localised Fock -- exact when the occupieds are canonical
(`localise="none"`), the standard DLPNO-(T0) approximation otherwise.
"""

from __future__ import annotations

from itertools import permutations, product

import numpy as np

from ._ccsd_cs import _blocks, cs_triple_energy, cs_triples_correction

try:  # optional during editable rebuilds of an older core extension
    from vibeqc._vibeqc_core import (
        dlpno_spatial_triples_correction as _cpp_spatial_triples_correction,
    )

    _HAVE_CPP_SPATIAL_TRIPLES = True
except (AttributeError, ImportError):
    _cpp_spatial_triples_correction = None
    _HAVE_CPP_SPATIAL_TRIPLES = False

try:  # optional during editable rebuilds of an older core extension
    from vibeqc._vibeqc_core import (
        dlpno_spatial_triple_energy as _cpp_spatial_triple_energy,
    )

    _HAVE_CPP_SPATIAL_TRIPLE = True
except (AttributeError, ImportError):
    _cpp_spatial_triple_energy = None
    _HAVE_CPP_SPATIAL_TRIPLE = False

try:  # optional during editable rebuilds
    from vibeqc._vibeqc_core import (
        dlpno_project_tno_amplitudes as _cpp_project_tno_amplitudes,
    )

    _HAVE_CPP_PROJECT_TNO = True
except (AttributeError, ImportError):
    _cpp_project_tno_amplitudes = None
    _HAVE_CPP_PROJECT_TNO = False

try:  # optional during editable rebuilds
    from vibeqc._vibeqc_core import (
        dlpno_build_tno_density as _cpp_build_tno_density,
    )

    _HAVE_CPP_TNO_DENSITY = True
except (AttributeError, ImportError):
    _cpp_build_tno_density = None
    _HAVE_CPP_TNO_DENSITY = False


def _local_distinct_triple_keys(n_act, occ_dist, coupling_radius):
    """Yield sorted distinct occupied sets whose ordered triples survive."""
    R = float(coupling_radius)
    screen = R > 0.0 and occ_dist is not None
    d = np.asarray(occ_dist) if screen else None

    for i in range(n_act):
        yield (i,)
    for i in range(n_act):
        for j in range(i + 1, n_act):
            if not screen or d[i, j] <= R:
                yield (i, j)
    for i in range(n_act):
        for j in range(i + 1, n_act):
            if screen and d[i, j] > R:
                continue
            for k in range(j + 1, n_act):
                if screen and (d[i, k] > R or d[j, k] > R):
                    continue
                yield (i, j, k)


def _ordered_triples_for_key(key):
    """Yield ordered triples whose set of distinct occupieds is ``key``."""
    if len(key) == 1:
        i = key[0]
        yield i, i, i
    elif len(key) == 2:
        a, b = key
        for triple in product(key, repeat=3):
            if a in triple and b in triple:
                yield triple
    else:
        yield from permutations(key, 3)


def _spatial_triples_correction(
    t1,
    t2,
    B_ov,
    B_oo,
    B_vv,
    eps_o,
    eps_v,
    *,
    use_cpp=True,
):
    """Closed-shell spatial `(T)` from DF blocks, native when available."""
    t1 = np.ascontiguousarray(np.asarray(t1, dtype=float))
    t2 = np.ascontiguousarray(np.asarray(t2, dtype=float))
    eps_o = np.ascontiguousarray(np.asarray(eps_o, dtype=float))
    eps_v = np.ascontiguousarray(np.asarray(eps_v, dtype=float))

    no, nv = t1.shape

    def coerce_block(block, n_left, n_right, name):
        arr = np.asarray(block, dtype=float)
        if arr.ndim == 3:
            if arr.shape[1:] != (n_left, n_right):
                raise ValueError(
                    f"{name} has trailing shape {arr.shape[1:]}; expected "
                    f"{(n_left, n_right)}"
                )
            tensor = np.ascontiguousarray(arr)
            flat = np.ascontiguousarray(arr.reshape(arr.shape[0], n_left * n_right))
            return flat, tensor
        if arr.ndim == 2:
            if arr.shape[1] != n_left * n_right:
                raise ValueError(
                    f"{name} has {arr.shape[1]} columns; expected "
                    f"{n_left * n_right}"
                )
            flat = np.ascontiguousarray(arr)
            tensor = np.ascontiguousarray(arr.reshape(arr.shape[0], n_left, n_right))
            return flat, tensor
        raise ValueError(f"{name} must be a rank-2 or rank-3 array")

    B_ov_flat, B_ov_tensor = coerce_block(B_ov, no, nv, "B_ov")
    B_oo_flat, B_oo_tensor = coerce_block(B_oo, no, no, "B_oo")
    B_vv_flat, B_vv_tensor = coerce_block(B_vv, nv, nv, "B_vv")

    t2_flat = np.ascontiguousarray(t2.reshape(no * no, nv * nv))
    if use_cpp and _HAVE_CPP_SPATIAL_TRIPLES:
        return float(
            _cpp_spatial_triples_correction(
                t1,
                t2_flat,
                B_ov_flat,
                B_oo_flat,
                B_vv_flat,
                eps_o,
                eps_v,
            )
        )

    V = _blocks(B_ov_tensor, B_vv_tensor, B_oo_tensor)
    return float(
        cs_triples_correction(
            t1, t2, V["ovvv"], V["ooov"], V["ovov"], eps_o, eps_v
        )
    )


def _spatial_triple_energy(
    i,
    j,
    k,
    t1,
    t2,
    ovvv,
    ooov,
    ovov,
    eps_o,
    eps_v,
    *,
    use_cpp=True,
):
    """Closed-shell spatial `(T)` for one ordered occupied triple."""
    t1 = np.ascontiguousarray(np.asarray(t1, dtype=float))
    t2 = np.ascontiguousarray(np.asarray(t2, dtype=float))
    ovvv = np.ascontiguousarray(np.asarray(ovvv, dtype=float))
    ooov = np.ascontiguousarray(np.asarray(ooov, dtype=float))
    ovov = np.ascontiguousarray(np.asarray(ovov, dtype=float))
    eps_o = np.ascontiguousarray(np.asarray(eps_o, dtype=float))
    eps_v = np.ascontiguousarray(np.asarray(eps_v, dtype=float))

    no, nv = t1.shape
    if ovvv.shape != (no, nv, nv, nv):
        raise ValueError(f"ovvv has shape {ovvv.shape}; expected {(no, nv, nv, nv)}")
    if ooov.shape != (no, no, no, nv):
        raise ValueError(f"ooov has shape {ooov.shape}; expected {(no, no, no, nv)}")
    if ovov.shape != (no, nv, no, nv):
        raise ValueError(f"ovov has shape {ovov.shape}; expected {(no, nv, no, nv)}")

    if use_cpp and _HAVE_CPP_SPATIAL_TRIPLE:
        return float(
            _cpp_spatial_triple_energy(
                int(i),
                int(j),
                int(k),
                t1,
                np.ascontiguousarray(t2.reshape(no * no, nv * nv)),
                ovvv.reshape(no * nv, nv * nv),
                ooov.reshape(no * no, no * nv),
                ovov.reshape(no * nv, no * nv),
                eps_o,
                eps_v,
            )
        )

    return float(cs_triple_energy(i, j, k, t1, t2, ovvv, ooov, ovov, eps_o, eps_v))


def _triples_screening_is_noop(occ_dist, coupling_radius):
    if coupling_radius <= 0.0 or occ_dist is None:
        return True
    d = np.asarray(occ_dist, dtype=float)
    if d.size == 0:
        return True
    return bool(np.all(d <= float(coupling_radius)))


def local_triples_correction(
    U,
    T2,
    t1,
    C_loc,
    C_vir,
    S,
    f_vv_full,
    f_dd,
    df,
    n_act,
    tcut_tno=0.0,
    lindep=1e-9,
    occ_dist=None,
    coupling_radius=0.0,
    stats=None,
):
    """(T) correction on the converged local DLPNO-CCSD amplitudes.

    Parameters mirror `run_local_dlpno_ccsd`'s internal state: per-pair PNO
    maps ``U[(i,j)]`` (n_vir x n_pno, in the canonical virtual basis),
    amplitudes ``T2[(i,j)]`` / ``t1[i]`` in those PNO bases, the localised
    occupied / canonical virtual coefficients, the virtual-block Fock
    ``f_vv_full`` and the diagonal occupied Fock ``f_dd``. ``tcut_tno`` > 0
    truncates each triple's TNO domain by occupation number (the eigenvalues
    of the triple's pair-amplitude density); 0 keeps the full union span
    (exact at full PNO domains). ``coupling_radius`` (with ``occ_dist``)
    screens distant triples -- see the module docstring.
    """
    return _local_triples_correction_eps(
        U,
        T2,
        t1,
        C_loc,
        C_vir,
        S,
        f_vv_full,
        f_dd,
        df,
        n_act,
        tcut_tno=tcut_tno,
        lindep=lindep,
        occ_dist=occ_dist,
        coupling_radius=coupling_radius,
        stats=stats,
    )


def _t1_jacobi_eigensystem(f_oo, n_act, max_iter=20, conv_tol=1e-10):
    """Iterative Jacobi diagonalisation of ``f_oo``, returning (evals, evecs).

    Returns ``eps_o`` (eigenvalues on the diagonal after convergence) and
    ``X``, the accumulated rotation matrix such that ``X^T @ f_oo @ X``
    is diagonal.  For the localised-Fock use case ``X`` is the matrix that
    transforms from the canonical-occupied basis to the LMO basis, so
    ``X[i_loc, i_can]`` is the component of canonical orbital ``i_can`` on
    LMO ``i_loc``.

    For a well-localised (nearly diagonal) Fock matrix a handful of sweeps
    suffice; the Jacobi rotation costs O(n_occ^2) per sweep.
    """
    A = f_oo.copy()
    n = A.shape[0]
    X = np.eye(n)
    for _sweep in range(max_iter):
        for i in range(n):
            for j in range(i + 1, n):
                a_ij = A[i, j]
                if abs(a_ij) < 1e-15:
                    continue
                a_ii, a_jj = A[i, i], A[j, j]
                if abs(a_ii - a_jj) < conv_tol:
                    c = np.sqrt(0.5)
                    s = c
                else:
                    theta = 0.5 * np.arctan(2.0 * a_ij / (a_jj - a_ii))
                    c, s = np.cos(theta), np.sin(theta)
                cc, ss = c * c, s * s
                cs = c * s
                A[i, i] = cc * a_ii - 2.0 * cs * a_ij + ss * a_jj
                A[j, j] = ss * a_ii + 2.0 * cs * a_ij + cc * a_jj
                A[i, j] = 0.0
                A[j, i] = 0.0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    aik, ajk = A[i, k], A[j, k]
                    A[i, k] = c * aik - s * ajk
                    A[j, k] = s * aik + c * ajk
                    aki, akj = A[k, i], A[k, j]
                    A[k, i] = c * aki - s * akj
                    A[k, j] = s * aki + c * akj
                for k in range(n):
                    xki, xkj = X[k, i], X[k, j]
                    X[k, i] = c * xki - s * xkj
                    X[k, j] = s * xki + c * xkj
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off = max(off, abs(A[p, q]))
        if off < conv_tol:
            break
    return np.diag(A), X


def _t1_occupied_energies(f_oo, t2_trace, n_act, max_iter=20, conv_tol=1e-10):
    eps_o, _ = _t1_jacobi_eigensystem(f_oo, n_act, max_iter=max_iter, conv_tol=conv_tol)
    return eps_o


def t1_triples_correction(
    U,
    T2,
    t1,
    C_loc,
    C_vir,
    S,
    f_oo,
    f_vv_full,
    df,
    n_act,
    tcut_tno=0.0,
    lindep=1e-9,
    occ_dist=None,
    coupling_radius=0.0,
    max_iter=20,
    conv_tol=1e-10,
    stats=None,
):
    """Compatibility ``(T1)`` correction with a canonical occupied basis.

    ``local_triples_correction`` (DLPNO-(T0)) uses only the *diagonal*
    elements of the localised occupied Fock ``f_dd = diag(f_oo)`` and keeps
    the amplitudes in the localised occupied basis -- two sources of
    semicanonical error worth ~0.1 kcal/mol.  ``exact_triples_correction``
    removes both by expanding to full virtual space + canonical occupied
    rotation, at O(N⁷).

    (T1) removes the occupied-basis error while **keeping the TNO-domain
    scaling**: ``f_oo`` is diagonalised via a localised Jacobi sweep, the
    converged eigenvectors rotate the amplitudes from the localised to the
    canonical occupied basis, and the triples are then evaluated per triple
    in its TNO domain (same as local-(T0)).  The eigenvalue correction alone
    accounts for only ~18% of the (T0) error; the amplitude rotation is
    essential.  This direct occupied-space rotation is deliberately distinct
    from the iterative local-basis Eq. (2) algorithm of Guo et al. (2018),
    which is implemented by :mod:`vibeqc.dlpno.triples_iterative`.

    At full PNO domains the result equals ``exact_triples_correction``
    (the (T1) parity ratchet).  With truncated domains the only error is the
    controlled PNO-domain approximation -- no semicanonical error remains.

    Parameters
    ----------
    f_oo : (n_act, n_act) ndarray, localised occupied Fock matrix
    max_iter, conv_tol : Jacobi convergence controls
    All other parameters mirror ``local_triples_correction``.
    """
    # (T1) step 1: diagonalise f_oo -> eigenvalues e and eigenvectors X.
    # The Jacobi sweep converges in the LMO basis, so X[i,:] is the
    # canonical orbital expressed in the LMO basis (X = U, the localisation
    # rotation from canonical to localised).
    eps_o, X = _t1_jacobi_eigensystem(f_oo, n_act, max_iter=max_iter, conv_tol=conv_tol)

    # (T1) step 2: delegate to the canonical-basis triples engine.
    # _t1_triples_energy handles the full amplitude expansion + rotation
    # using X and the original LMO-basis t1/T2.
    return _t1_triples_energy(
        U,
        T2,
        t1,
        X,
        C_loc,
        C_vir,
        S,
        f_vv_full,
        eps_o,
        df,
        n_act,
        tcut_tno=tcut_tno,
        lindep=lindep,
        occ_dist=occ_dist,
        coupling_radius=coupling_radius,
        stats=stats,
    )


def _t1_occupied_energies(f_oo, t2_trace, n_act, max_iter=20, conv_tol=1e-10):
    eps_o, _ = _t1_jacobi_eigensystem(f_oo, n_act, max_iter=max_iter, conv_tol=conv_tol)
    return eps_o


def _local_triples_correction_eps(
    U,
    T2,
    t1,
    C_loc,
    C_vir,
    S,
    f_vv_full,
    eps_o,
    df,
    n_act,
    tcut_tno=0.0,
    lindep=1e-9,
    occ_dist=None,
    coupling_radius=0.0,
    stats=None,
):
    """Same as ``local_triples_correction`` but with explicit ``eps_o``.

    Used by ``t1_triples_correction`` to inject the Jacobi-converged occupied
    energies and by ``local_triples_correction`` itself (forwarded with
    ``eps_o = f_dd``).
    """
    B_ov = np.ascontiguousarray(np.asarray(df.mo_transform(C_loc, C_vir)))
    B_oo = np.ascontiguousarray(np.asarray(df.mo_transform(C_loc, C_loc)))
    B_vv = np.ascontiguousarray(np.asarray(df.mo_transform(C_vir, C_vir)))

    def has_pair(p, q):
        return (p, q) in U or (q, p) in U

    def Uof(p, q):
        return U[(p, q)] if (p, q) in U else U[(q, p)]

    def Tof(p, q):
        return T2[(p, q)] if (p, q) in T2 else T2[(q, p)].T

    # Both native helpers consume the same complete ordered inventory: the
    # density builder selects the orientation for each distinct pair, while
    # the amplitude projector writes every ordered (m, n) block.
    pair_list = [
        (m, nn)
        for m in range(n_act)
        for nn in range(n_act)
        if has_pair(m, nn)
    ]

    def build_tno(distinct):
        cols = [
            Uof(p, q)
            for ia, p in enumerate(distinct)
            for q in distinct[ia:]
            if has_pair(p, q)
        ]
        M = np.hstack(cols)
        A, sv, _ = np.linalg.svd(M, full_matrices=False)
        V_T = A[:, sv > lindep * max(sv[0], 1e-300)]
        if tcut_tno > 0.0 and V_T.shape[1] > 1:
            if _HAVE_CPP_TNO_DENSITY:
                pi = [p for p, _ in pair_list]
                pj = [q for _, q in pair_list]
                U_list = [np.asfortranarray(Uof(p, q)) for p, q in pair_list]
                T2_list = [np.asfortranarray(Tof(p, q)) for p, q in pair_list]
                V_T = np.asarray(_cpp_build_tno_density(
                    np.asfortranarray(V_T), list(distinct),
                    pi, pj, U_list, T2_list, float(tcut_tno),
                ))
            else:
                D = np.zeros((V_T.shape[1], V_T.shape[1]))
                for ia, p in enumerate(distinct):
                    for q in distinct[ia:]:
                        if has_pair(p, q):
                            Spq = V_T.T @ Uof(p, q)
                            Tpq = Spq @ Tof(p, q) @ Spq.T
                            D += Tpq @ Tpq.T + Tpq.T @ Tpq
                occ, vec = np.linalg.eigh(0.5 * (D + D.T))
                keep = occ > tcut_tno
                if not keep.any():
                    keep[int(np.argmax(occ))] = True
                V_T = V_T @ vec[:, keep]
        F_T = V_T.T @ f_vv_full @ V_T
        eps_v_T, rot = np.linalg.eigh(0.5 * (F_T + F_T.T))
        V_T = V_T @ rot

        B_ov_T = np.einsum("Pmv,va->Pma", B_ov, V_T, optimize=True)
        B_vv_T = np.einsum("Puv,ua,vb->Pab", B_vv, V_T, V_T, optimize=True)
        ovvv = np.einsum("Pia,Pbc->iabc", B_ov_T, B_vv_T, optimize=True)
        ooov = np.einsum("Pij,Pka->ijka", B_oo, B_ov_T, optimize=True)
        ovov = np.einsum("Pia,Pjb->iajb", B_ov_T, B_ov_T, optimize=True)

        n_T = V_T.shape[1]
        if _HAVE_CPP_PROJECT_TNO:
            # Batched C++ kernel replaces the Python double-loops below.
            # The kernel expects t1 vectors already in the full virtual
            # space (length nv), not in the PNO basis; expand them first.
            t1_keys = [m for m in range(n_act) if m in t1 and has_pair(m, m)]
            t1_vecs = [
                np.ascontiguousarray(Uof(m, m) @ t1[m]) for m in t1_keys
            ]
            if pair_list:
                pi = [p for p, _ in pair_list]
                pj = [q for _, q in pair_list]
                U_list = [np.asfortranarray(Uof(p, q)) for p, q in pair_list]
                T2_list = [np.asfortranarray(Tof(p, q)) for p, q in pair_list]
                t1f, T2_flat = _cpp_project_tno_amplitudes(
                    np.asfortranarray(V_T), n_act,
                    pi, pj, U_list, T2_list,
                    t1_keys, t1_vecs,
                )
                T2f = np.asarray(T2_flat).reshape(n_act, n_act, n_T, n_T)
                t1f = np.asarray(t1f)
            else:
                t1f = np.zeros((n_act, n_T))
                T2f = np.zeros((n_act, n_act, n_T, n_T))
        else:
            # Python fallback (kept for cores built before the binding).
            t1f = np.zeros((n_act, n_T))
            for m in range(n_act):
                if m in t1 and has_pair(m, m):
                    t1f[m] = V_T.T @ Uof(m, m) @ t1[m]
            T2f = np.zeros((n_act, n_act, n_T, n_T))
            for m in range(n_act):
                for nn in range(n_act):
                    if has_pair(m, nn):
                        Smn = V_T.T @ Uof(m, nn)
                        T2f[m, nn] = Smn @ Tof(m, nn) @ Smn.T
        return ovvv, ooov, ovov, t1f, T2f, eps_v_T

    e_t = 0.0
    _record_screened_keys(stats, n_act, occ_dist, coupling_radius)
    for key in _local_distinct_triple_keys(n_act, occ_dist, coupling_radius):
        ovvv, ooov, ovov, t1f, T2f, eps_v_T = build_tno(list(key))
        _record_tno_domain(stats, key, len(eps_v_T))
        for i, j, k in _ordered_triples_for_key(key):
            e_t += _spatial_triple_energy(
                i, j, k, t1f, T2f, ovvv, ooov, ovov, eps_o, eps_v_T
            )
    return e_t


def _record_screened_keys(stats, n_act, occ_dist, coupling_radius):
    """Record how many distinct triple keys locality screening dropped.

    ``coupling_radius`` drops occupied triple keys whose members are mutually
    distant, which is the occupied-side analogue of the TNO truncation on the
    virtual side. The counts let a caller see how aggressive that was: on a
    delocalized system a generous-looking radius can discard most of the
    triples energy, and the energy alone does not say so.
    """
    if stats is None:
        return
    kept = sum(1 for _ in _local_distinct_triple_keys(
        n_act, occ_dist, coupling_radius
    ))
    total = sum(1 for _ in _local_distinct_triple_keys(n_act, None, 0.0))
    stats["n_triple_keys"] = int(kept)
    stats["n_triple_keys_screened"] = int(total - kept)


def _record_tno_domain(stats, key, size):
    """Record one triple's retained TNO count into an optional stats sink.

    ``stats`` is a plain dict the caller owns; ``None`` disables collection so
    the hot loop pays nothing. ``key`` is the distinct-occupied tuple the TNO
    space was built for, ``size`` the number of retained virtuals.
    """
    if stats is None:
        return
    stats.setdefault("tno_per_triple", {})[tuple(key)] = int(size)


def summarise_tno_domains(stats):
    """Reduce a raw TNO stats sink to the reported diagnostics.

    Returns ``(mean_domain, n_degenerate)``. A spatial triple excitation
    promotes three electrons into three *distinct* virtuals, so a domain
    holding fewer than three cannot host one and contributes exactly zero
    rather than approximately: ``n_degenerate`` counting above zero means
    ``tcut_tno`` has truncated past the point where the correction exists.
    """
    sizes = list((stats or {}).get("tno_per_triple", {}).values())
    if not sizes:
        return 0.0, 0
    return float(np.mean(sizes)), sum(1 for size in sizes if size < 3)


def _t1_triples_energy(
    U,
    T2,
    t1,
    X,
    C_loc,
    C_vir,
    S,
    f_vv_full,
    eps_o,
    df,
    n_act,
    tcut_tno=0.0,
    lindep=1e-9,
    occ_dist=None,
    coupling_radius=0.0,
    stats=None,
):
    """(T1) triples energy: canonical-basis (T) on Jacobi-rotated amplitudes.

    1. Expand all LMO amplitudes to full virtual space.
    2. Rotate occupied indices LMO->canonical via the Jacobi eigenvectors ``X``.
    3. Build DF integrals directly in the canonical occupied basis.
    4. Apply the validated ``cs_triples_correction``.

    This reproduces ``exact_triples_correction`` when the Jacobi eigensystem
    converges (X == Uo^T where Uo = C_act^T S C_loc).  The TNO truncation
    (``tcut_tno``) and triple screening (``coupling_radius``) are applied on
    top if desired, making this the scaling-preserving, semicanonical-free
    (T1) production path.
    """
    nv = C_vir.shape[1]

    # Step 1: Expand LMO T2 amplitudes to full virtual space
    t2_full_loc = np.zeros((n_act, n_act, nv, nv))
    for (i, j), T in T2.items():
        Uij = U[(i, j)]
        blk = Uij @ T @ Uij.T
        t2_full_loc[i, j] = blk
        if i != j:
            t2_full_loc[j, i] = blk.T
    # Expand t1 too
    t1_full_loc = np.zeros((n_act, nv))
    for i in t1:
        if (i, i) in U:
            t1_full_loc[i] = U[(i, i)] @ t1[i]

    # Step 2: Rotate occupied indices LMO -> canonical via X.
    # X[i_loc, I_can] = component of canonical orbital I_can on LMO i_loc.
    # So X^T maps LMO -> canonical: t1_can = X^T @ t1_loc.
    t1_full_can = X.T @ t1_full_loc
    t2_full_can = np.einsum("iI,jJ,ijab->IJab", X, X, t2_full_loc, optimize=True)

    # Step 3: Build DF integrals in the canonical occupied basis.
    B_ov_can = np.einsum(
        "Pia,iI->PIa",
        np.ascontiguousarray(np.asarray(df.mo_transform(C_loc, C_vir))),
        X,
        optimize=True,
    )
    B_oo_can = np.einsum(
        "Pij,iI,jJ->PIJ",
        np.ascontiguousarray(np.asarray(df.mo_transform(C_loc, C_loc))),
        X,
        X,
        optimize=True,
    )
    B_vv = np.ascontiguousarray(np.asarray(df.mo_transform(C_vir, C_vir)))

    # Step 4: (T) correction.
    # With tcut_tno=0 (default) and no actual occupied-triple screening,
    # full (T) is the validated spatial triples contraction.  Use the native
    # OpenMP kernel when the extension provides it; the helper falls back to
    # the Python oracle for older editable builds.  When tcut_tno>0 or a
    # finite radius really screens triples, use the per-triple TNO loop below.
    if tcut_tno <= 0.0 and _triples_screening_is_noop(
        occ_dist, coupling_radius
    ):
        if stats is not None:
            _record_screened_keys(stats, n_act, occ_dist, coupling_radius)
            for key in _local_distinct_triple_keys(
                n_act, occ_dist, coupling_radius
            ):
                _record_tno_domain(stats, key, nv)
        return _spatial_triples_correction(
            t1_full_can,
            t2_full_can,
            B_ov_can,
            B_oo_can,
            B_vv,
            eps_o,
            np.diag(f_vv_full),
        )

    # With TNO truncation or triple screening: per-triple loop.
    # Each triple (I,J,K) uses the full virtual space chopped into a TNO
    # domain from the pair amplitudes projected to the full space.
    def build_tno(distinct):
        n_can = len(distinct)
        cols = [np.eye(nv) for _ in range(n_can * (n_can + 1) // 2)]
        M = np.hstack(cols)
        A_mat, sv, _ = np.linalg.svd(M, full_matrices=False)
        V_T = A_mat[:, sv > lindep * max(sv[0], 1e-300)]
        if tcut_tno > 0.0 and V_T.shape[1] > 1:
            D = np.zeros((V_T.shape[1], V_T.shape[1]))
            for ia, p in enumerate(distinct):
                for q in distinct[ia:]:
                    Tpq = V_T.T @ t2_full_can[p, q]
                    D += Tpq @ Tpq.T + Tpq.T @ Tpq
            occ, vec = np.linalg.eigh(0.5 * (D + D.T))
            keep = occ > tcut_tno
            if not keep.any():
                keep[int(np.argmax(occ))] = True
            V_T = V_T @ vec[:, keep]
        F_T = V_T.T @ f_vv_full @ V_T
        eps_v_T, rot = np.linalg.eigh(0.5 * (F_T + F_T.T))
        V_T = V_T @ rot
        B_ov_T = np.einsum("PIa,ab->PIb", B_ov_can, V_T, optimize=True)
        B_vv_T = np.einsum("Pab,au,bv->Puv", B_vv, V_T, V_T, optimize=True)
        ovvv = np.einsum("PIa,Pbc->Iabc", B_ov_T, B_vv_T, optimize=True)
        ooov = np.einsum("PIJ,PKa->IJKa", B_oo_can, B_ov_T, optimize=True)
        ovov = np.einsum("PIa,PJb->IaJb", B_ov_T, B_ov_T, optimize=True)
        n_T = V_T.shape[1]
        t1f = np.zeros((n_act, n_T))
        for I in range(n_act):
            t1f[I] = V_T.T @ t1_full_can[I]
        T2f = np.zeros((n_act, n_act, n_T, n_T))
        for I in range(n_act):
            for J in range(n_act):
                T2f[I, J] = V_T.T @ t2_full_can[I, J] @ V_T
        return ovvv, ooov, ovov, t1f, T2f, eps_v_T

    e_t = 0.0
    _record_screened_keys(stats, n_act, occ_dist, coupling_radius)
    for key in _local_distinct_triple_keys(n_act, occ_dist, coupling_radius):
        ovvv, ooov, ovov, t1f, T2f, eps_v_T = build_tno(list(key))
        _record_tno_domain(stats, key, len(eps_v_T))
        for I, J, K in _ordered_triples_for_key(key):
            e_t += _spatial_triple_energy(
                I, J, K, t1f, T2f, ovvv, ooov, ovov, eps_o, eps_v_T
            )
    return e_t


def exact_triples_correction(
    U, T2, t1, C_act, C_loc, C_vir, S, F, f_vv_full, df, n_act
):
    """Canonical (exact) (T) on the converged local DLPNO-CCSD amplitudes.

    The local DLPNO-(T0) above uses the diagonal localised Fock in the triples
    denominators -- a semicanonical approximation worth ~0.1 kcal/mol. This
    removes it entirely: expand the per-pair PNO amplitudes to the full virtual
    space, rotate the occupied from the localised to the canonical basis
    (``Uo = C_actᵀ S C_loc``, the localisation rotation), and apply the
    validated spatial (T) (``cs_triples_correction``) in the *canonical* basis,
    where the occupied Fock is diagonal so the (T) is exact for these
    amplitudes. O(N⁷) (full virtual space, no TNO truncation) -- the
    high-accuracy mode for when the (T) is still affordable; (T1) is the
    scaling-preserving route to the same accuracy.
    """
    nv = C_vir.shape[1]
    # per-pair PNO amplitudes -> full canonical virtual space
    t1f = np.zeros((n_act, nv))
    for i in t1:
        if (i, i) in U:
            t1f[i] = U[(i, i)] @ t1[i]
    t2f = np.zeros((n_act, n_act, nv, nv))
    for (i, j), T in T2.items():
        blk = U[(i, j)] @ T @ U[(i, j)].T
        t2f[i, j] = blk
        if i != j:
            t2f[j, i] = blk.T
    # occupied: localised -> canonical (amplitudes transform with the orbitals)
    Uo = C_act.T @ S @ C_loc
    t1c = Uo @ t1f
    t2c = np.einsum("Ii,Jj,ijab->IJab", Uo, Uo, t2f, optimize=True)
    # canonical-basis chemist integrals + canonical orbital energies
    B_ov = np.asarray(df.mo_transform(C_act, C_vir))
    B_oo = np.asarray(df.mo_transform(C_act, C_act))
    B_vv = np.asarray(df.mo_transform(C_vir, C_vir))
    V = _blocks(B_ov, B_vv, B_oo)
    eps_o = np.diag(C_act.T @ F @ C_act)
    eps_v = np.diag(f_vv_full)
    return cs_triples_correction(
        t1c, t2c, V["ovvv"], V["ooov"], V["ovov"], eps_o, eps_v
    )
