"""Spin-orbital generalization of Guo et al.'s iterative local `(T1)`.

vibe-qc's existing `(T1)` (``triples_local.t1_triples_correction`` and the
open-shell ``triples_mode="t1"``) is **not** the algorithm of that paper. It
diagonalises the occupied Fock and rotates the occupied indices into the
canonical basis, where the `(T)` is exact because the occupied Fock is
diagonal there. That is exact and cheap, but it delocalises: a canonical
occupied is a mixture of every LMO, so a locality-screened triple list cannot
survive the rotation, which is why the open-shell engine fails closed on any
realised screening.

This module generalizes the 2018 closed-shell equation to spin orbitals: it
keeps triples in the **localised** occupied basis and reintroduces the
off-diagonal occupied-Fock coupling by iterating the triples amplitudes.
Because the triple list never leaves the LMO basis, it is the route on which
locality screening is expressible. Guo et al. 2020 is the dedicated
open-shell method-family source; pending a direct equation-by-equation audit
of that paper, this module does not claim to reproduce all of its
implementation-specific approximations.

Reference
---------
Eq. (2) of Guo, Riplinger, Becker, Liakos, Minenkov, Cavallo and Neese,
J. Chem. Phys. 148, 011101 (2018), doi:10.1063/1.5011798:

    R^{a b c}_{ijk} = W^{a b c}_{ijk} + T^{a b c}_{ijk} (f_a + f_b + f_c)
                      - sum_l T^{a b c}_{ijl} f^l_k S
                      - sum_l T^{a b c}_{ilk} f^l_j S
                      - sum_l T^{a b c}_{ljk} f^l_i S

with `S` the overlap between the TNO spaces of two different triples. The
implementation accepts a separate TNO basis for every retained triple and
rotates coupled amplitudes through the corresponding overlaps. Passing no
domains selects the full-virtual oracle, where all overlaps are the identity.

Approximations 2 and 3 of the paper are available as ``f_cut`` and
``t_cut_iter``, both defaulting to zero (exact). ``f_cut`` neglects the
coupling between two triples carried by an occupied Fock element below the
threshold; ``t_cut_iter`` freezes a triple once its energy stops moving
between sweeps. The paper's SI (Tables S2, S3) gives `1e-3` for both, each
recovering 99.93% of the unapproximated result.

One caveat worth stating, because it differs from the TNO truncation the rest
of the DLPNO stack uses: **the `f_cut` error is not sign-definite.** The
retained-energy change from TNO truncation is positive in the calibration
systems exercised here, while `f_cut` drops *couplings*, whose contributions
carry either sign. The energy can therefore overshoot the unapproximated
reference. The focused calibration tests show decreasing error magnitude as
`f_cut` is tightened; this is an observed convergence check, not a universal
variational or monotonicity guarantee.

Reduction check (why the scheme is the right generalisation): for a diagonal
occupied Fock the three `l`-sums collapse to `T (f_i + f_j + f_k)`, so

    R = W + T (f_a + f_b + f_c) - T (f_i + f_j + f_k) = W - T * D

with `D = f_i + f_j + f_k - f_a - f_b - f_c`, and `R = 0` gives `T = W / D`:
exactly the semicanonical `(T0)` amplitude that
``_ccsd_ref.so_triple_energy`` forms implicitly. Eq. (3) of the same paper.
So canonical occupieds must reproduce `(T0)`, and localised occupieds must
reproduce the exact `(T)`, which is what the tests pin.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "degenerate_tno_triples",
    "iterative_t1_triples_correction",
]


def _distinct_triples(no):
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                yield i, j, k


def _connected_w(i, j, k, t2, eri_vovv, eri_ovoo):
    """Connected `W^{abc}_{ijk}`, the source term of Eq. (2).

    Identical construction to the connected buffer of
    ``_ccsd_ref.so_triple_energy``:

        W^c = P(i/jk) P(a/bc) [ sum_e t_{jk}^{ae} <ei||bc>
                                - sum_m t_{im}^{bc} <ma||jk> ]
    """

    def x(i_, j_, k_):
        term1 = np.einsum("ae,ebc->abc", t2[j_, k_], eri_vovv[:, i_])
        term2 = np.einsum("mbc,ma->abc", t2[i_], eri_ovoo[:, :, j_, k_])
        return term1 - term2

    w = x(i, j, k) - x(j, i, k) - x(k, j, i)
    return w - w.transpose(1, 0, 2) - w.transpose(2, 1, 0)


def _disconnected_w(i, j, k, t1, eri_oovv):
    """Disconnected `W^d_{ijk}` = P(i/jk) P(a/bc) [ t_i^a <jk||bc> ]."""

    def y(i_, j_, k_):
        return np.einsum("a,bc->abc", t1[i_], eri_oovv[j_, k_])

    w = y(i, j, k) - y(j, i, k) - y(k, j, i)
    return w - w.transpose(1, 0, 2) - w.transpose(2, 1, 0)


def _antisymmetrised_amplitude(store, i, j, k):
    """Return `T^{abc}_{ijk}` for an arbitrary occupied ordering.

    Amplitudes are stored only for the sorted distinct triple. `T` is
    antisymmetric under occupied exchange, so a permuted request is served by
    the sorted entry times the permutation sign; a repeated occupied index
    gives zero.

    A triple absent from ``store`` was screened out of the list, and Eq. (2)'s
    `l`-sums then simply skip it: the paper considers "only a subset of
    triples", so a screened triple carries no amplitude to couple through.
    That is what makes the localised-basis iteration compatible with locality
    screening, unlike the canonical-rotation `(T1)`.
    """
    idx = (i, j, k)
    if len(set(idx)) < 3:
        return None, 0.0
    order = sorted(range(3), key=lambda p: idx[p])
    key = tuple(idx[p] for p in order)
    # Parity of the permutation that sorts (i, j, k).
    sign = 1.0
    seen = list(order)
    for a in range(3):
        for b in range(a + 1, 3):
            if seen[a] > seen[b]:
                sign = -sign
    block = store.get(key)
    if block is None:
        return None, 0.0
    return block, sign


def _project_blocks(domain, t1, t2, eri_vovv, eri_ovoo, eri_oovv):
    """Project the `(T)` source blocks into one triple's TNO domain.

    ``domain`` is ``V_T`` with shape (n_vir, n_tno), assumed orthonormal and
    already semicanonical (the caller diagonalised the virtual Fock inside it,
    so the denominator stays diagonal). Every virtual index that the connected
    and disconnected `W` contract over is projected, including the internal
    ``e`` and ``a`` sums: restricting those *is* the DLPNO-(T) domain
    approximation, exactly as ``triples_local`` does it for the spatial case.
    """
    v = np.asarray(domain, dtype=float)
    t1_d = t1 @ v
    t2_d = np.einsum("mnab,au,bw->mnuw", t2, v, v, optimize=True)
    vovv_d = np.einsum("eibc,eu,bw,cx->uiwx", eri_vovv, v, v, v, optimize=True)
    ovoo_d = np.einsum("majk,au->mujk", eri_ovoo, v, optimize=True)
    oovv_d = np.einsum("jkbc,bw,cx->jkwx", eri_oovv, v, v, optimize=True)
    return t1_d, t2_d, vovv_d, ovoo_d, oovv_d


def _overlap(domain_target, domain_source):
    """`S = V_T(target)^T V_T(source)`, the Eq. (2) inter-triple overlap."""
    return np.asarray(domain_target, dtype=float).T @ np.asarray(
        domain_source, dtype=float
    )


def _rotate_amplitude(block, s):
    """Carry `T` from the source triple's domain into the target's.

    All three virtual indices transform, since `S` maps one virtual basis onto
    the other: ``T'^{uvw} = sum_{abc} S_{ua} S_{vb} S_{wc} T^{abc}``.
    """
    return np.einsum("ua,vb,wc,abc->uvw", s, s, s, block, optimize=True)


def iterative_t1_triples_correction(
    t1,
    t2,
    eri_vovv,
    eri_ovoo,
    eri_oovv,
    f_oo,
    eps_v,
    *,
    triple_list=None,
    tno_domains=None,
    f_cut=0.0,
    t_cut_iter=0.0,
    max_iter=50,
    conv_tol=1e-11,
):
    """Spin-orbital generalization of Guo Eq. (2), iterative local `(T1)`.

    Parameters mirror the spin-orbital `(T)` kernel: converged CCSD amplitudes
    ``t1``/``t2``, the antisymmetrised integral blocks ``<ei||bc>``,
    ``<ma||jk>`` and ``<jk||bc>``, the **full** occupied Fock block ``f_oo``
    in the localised basis (its off-diagonal part is what this routine
    reintroduces), and the diagonal virtual energies ``eps_v``. Optional
    ``tno_domains`` supplies each retained triple's semicanonical TNO basis;
    coupled amplitudes are transferred with the inter-domain overlaps.

    Returns ``(e_t, n_iter, converged)``. The energy uses the converged
    amplitudes in place of the semicanonical `W/D`:

        E = (1/6) sum_{i<j<k} sum_{abc} T^{abc}_{ijk} (W^c + W^d)^{abc}_{ijk}

    which is the `(T0)` expression with `W^c/D` replaced by `T`, and reduces
    to it exactly when ``f_oo`` is diagonal.
    """
    if f_cut < 0.0:
        raise ValueError("f_cut must be non-negative")
    if t_cut_iter < 0.0:
        raise ValueError("t_cut_iter must be non-negative")
    f_oo = np.asarray(f_oo, dtype=float)
    eps_v = np.asarray(eps_v, dtype=float)
    no = f_oo.shape[0]
    eps_o = np.diag(f_oo)

    if no < 3:
        return 0.0, 0, True

    if triple_list is None:
        triples = list(_distinct_triples(no))
    else:
        # Caller-supplied locality-screened list. Sorted distinct triples only;
        # the l-sums below skip anything absent, which is Eq. (2) restricted to
        # the surviving subset.
        triples = sorted({tuple(sorted(t)) for t in triple_list})
        for t_ in triples:
            if len(set(t_)) != 3 or not all(0 <= x < no for x in t_):
                raise ValueError(f"invalid occupied triple {t_!r}")
    if not triples:
        return 0.0, 0, True
    # Per-triple TNO domains. `None` means the full virtual space for every
    # triple, in which case the Eq. (2) overlaps are the identity and the S
    # path below collapses to the no-op it generalises.
    if tno_domains is None:
        domains = None
        eps_per_triple = {t_: eps_v for t_ in triples}
    else:
        domains = {}
        eps_per_triple = {}
        for t_ in triples:
            entry = tno_domains.get(t_)
            if entry is None:
                raise ValueError(f"no TNO domain supplied for triple {t_!r}")
            vec, eps_t = entry
            vec = np.ascontiguousarray(np.asarray(vec, dtype=float))
            if vec.ndim != 2 or vec.shape[0] != eps_v.shape[0]:
                raise ValueError(
                    f"TNO domain for {t_!r} has shape {vec.shape}; expected "
                    f"(n_vir={eps_v.shape[0]}, n_tno)"
                )
            if vec.shape[1] == 0:
                raise ValueError(f"TNO domain for {t_!r} is empty")
            if not np.all(np.isfinite(vec)):
                raise ValueError(
                    f"TNO domain vectors for {t_!r} must be finite"
                )
            eps_t = np.asarray(eps_t, dtype=float)
            if eps_t.ndim != 1 or eps_t.shape[0] != vec.shape[1]:
                raise ValueError(
                    f"TNO energies for {t_!r} have shape {eps_t.shape}; "
                    f"expected ({vec.shape[1]},)"
                )
            if not np.all(np.isfinite(eps_t)):
                raise ValueError(
                    f"TNO energies for {t_!r} must be finite"
                )
            gram = vec.T @ vec
            if not np.allclose(
                gram,
                np.eye(vec.shape[1]),
                rtol=1e-10,
                atol=1e-10,
            ):
                raise ValueError(
                    f"TNO domain vectors for {t_!r} must be orthonormal"
                )
            projected_fock = vec.T @ (eps_v[:, None] * vec)
            if not np.allclose(
                projected_fock,
                np.diag(eps_t),
                rtol=1e-9,
                atol=1e-10,
            ):
                raise ValueError(
                    f"TNO domain for {t_!r} is not semicanonical or its "
                    "energies do not match the projected virtual Fock"
                )
            domains[t_] = vec
            eps_per_triple[t_] = eps_t

    # Source terms are amplitude-independent: build once, reuse every sweep.
    if domains is None:
        w_c = {t_: _connected_w(*t_, t2, eri_vovv, eri_ovoo) for t_ in triples}
        w_d = {t_: _disconnected_w(*t_, t1, eri_oovv) for t_ in triples}
    else:
        w_c, w_d = {}, {}
        for t_ in triples:
            t1_d, t2_d, vovv_d, ovoo_d, oovv_d = _project_blocks(
                domains[t_], t1, t2, eri_vovv, eri_ovoo, eri_oovv
            )
            w_c[t_] = _connected_w(*t_, t2_d, vovv_d, ovoo_d)
            w_d[t_] = _disconnected_w(*t_, t1_d, oovv_d)

    # Semicanonical denominator per triple, D = f_i + f_j + f_k - f_a - f_b - f_c,
    # with the virtual energies of that triple's own (semicanonical) domain.
    def _denominator(t_):
        e = eps_per_triple[t_]
        return (eps_o[t_[0]] + eps_o[t_[1]] + eps_o[t_[2]]) - (
            e[:, None, None] + e[None, :, None] + e[None, None, :]
        )

    denom = {t_: _denominator(t_) for t_ in triples}

    # Start from the (T0) amplitudes: the exact fixed point when f_oo is
    # diagonal, and the natural guess otherwise.
    amp = {t: w_c[t] / denom[t] for t in triples}

    # Off-diagonal occupied couplings only; the diagonal is already carried by
    # `denom`, so subtracting it here would double-count it.
    f_off = f_oo - np.diag(eps_o)

    def _triple_energy(t_, block):
        return float(np.sum(block * (w_c[t_] + w_d[t_]))) / 6.0

    # Approximation 3 of the paper: once a triple's energy stops moving
    # between sweeps it is frozen and no longer updated. `t_cut_iter=0`
    # freezes nothing, so the default is exact. The paper states the criterion
    # as "its energy increment with respect to the energy in the previous
    # iteration is less than T_CutIter" without saying absolute or relative;
    # its thresholds span 1e-1 to 1e-4, which read as fractions, so the
    # increment here is *relative* to the triple's own energy. Recorded openly
    # rather than silently chosen. Note also that this iteration is Jacobi
    # where the paper's is Gauss-Seidel: the criterion transfers unchanged,
    # but sweep counts are not comparable to the paper's timings.
    frozen: set = set()
    prev_energy = {t_: _triple_energy(t_, amp[t_]) for t_ in triples}

    converged = False
    n_iter = 0
    for sweep in range(max_iter):
        n_iter = sweep + 1
        max_residual = 0.0
        # Jacobi: every triple updates from the previous sweep's amplitudes.
        updated = {}
        for t in triples:
            if t in frozen:
                updated[t] = amp[t]
                continue
            i, j, k = t
            # W + T (f_a + f_b + f_c) - (diagonal part) == W - T * D, using the
            # stored semicanonical denominator.
            residual = w_c[t] - amp[t] * denom[t]
            # The off-diagonal occupied coupling: sum_l over the three slots.
            for slot, (fixed_a, fixed_b) in enumerate(
                ((j, k), (i, k), (i, j))
            ):
                for l in range(no):
                    coupling = f_off[l, t[slot]]
                    # Approximation 2 of the paper: couplings carried by a
                    # Fock element below `f_cut` are neglected. `f_cut=0`
                    # keeps every nonzero coupling, so the default is exact.
                    if abs(coupling) <= f_cut:
                        continue
                    if slot == 0:
                        block, sign = _antisymmetrised_amplitude(
                            amp, l, fixed_a, fixed_b
                        )
                    elif slot == 1:
                        block, sign = _antisymmetrised_amplitude(
                            amp, fixed_a, l, fixed_b
                        )
                    else:
                        block, sign = _antisymmetrised_amplitude(
                            amp, fixed_a, fixed_b, l
                        )
                    if block is None:
                        continue
                    if domains is not None:
                        source = tuple(sorted(
                            (l, fixed_a, fixed_b) if slot == 0
                            else (fixed_a, l, fixed_b) if slot == 1
                            else (fixed_a, fixed_b, l)
                        ))
                        if source != t:
                            block = _rotate_amplitude(
                                block, _overlap(domains[t], domains[source])
                            )
                    residual = residual - coupling * sign * block
            updated[t] = amp[t] + residual / denom[t]
            max_residual = max(max_residual, float(np.max(np.abs(residual))))
        amp = updated
        if t_cut_iter > 0.0:
            for t in triples:
                if t in frozen:
                    continue
                energy = _triple_energy(t, amp[t])
                scale = max(abs(energy), 1e-300)
                if abs(energy - prev_energy[t]) / scale < t_cut_iter:
                    frozen.add(t)
                prev_energy[t] = energy
        if max_residual < conv_tol:
            converged = True
            break

    e_t = 0.0
    for t in triples:
        e_t += float(np.sum(amp[t] * (w_c[t] + w_d[t])))
    return e_t / 6.0, n_iter, converged


def degenerate_tno_triples(tno_domains):
    """Triples whose TNO domain cannot host a triple excitation.

    A triple promotes three electrons into three *distinct* virtuals, so a
    domain retaining fewer than three contributes exactly zero rather than
    approximately. Callers should surface this rather than let a threshold
    silently truncate the correction out of existence, which is the lesson
    already pinned for the `(T0)`/`(T1)` TNO paths.
    """
    return sorted(
        t for t, (vec, _) in (tno_domains or {}).items()
        if np.asarray(vec).shape[1] < 3
    )
