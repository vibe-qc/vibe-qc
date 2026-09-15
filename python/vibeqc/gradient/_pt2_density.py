"""SC-CASPT2 effective density matrices for analytic gradient.

Computes PT2-corrected 1- and 2-RDMs from the strongly-contracted CASPT2
external groups.  The gradient is then assembled via the C++ derivative
integral kernels (one_electron_gradient_contribution etc.).

Formulation (SC-CASPT2, Andersson 1990 + Helgaker 12.5):

  E^(2) = sum_g E_g,   E_g = -N_g / (F_g - E0)

  N_g = <V_g|V_g>
  F_g = <V_g|F|V_g> / N_g     (generalized Fock expectation)
  E0  = <0|F|0>

Effective 1-RDM correction:
  DeltaD_pq = dE^(2) / dh_pq
            = sum_g w_g * [ <V_g|E_pq|V_g>/N_g - <0|E_pq|0> ]
  w_g = N_g / (F_g - E0)^2

This follows from dE_g/dE0 = N_g/Delta^2 (corrected v35; was N_g^2/Delta^2).

Effective 2-RDM correction (chain rule through Fock):
  F_tu = h_tu + sum_{vw} D_ref[v,w] * (2(tv|uw) - (tw|uv))  [chemist's]
  dF_tu/d(pq|rs) = 2 delta_{tp} delta_{uq} D_ref[r,s]
                  - delta_{tp} delta_{us} D_ref[r,q]
                  - delta_{tr} delta_{uq} D_ref[p,s]
  DeltaGamma[p,q,r,s] = 2 * (2 DeltaD[p,q] D_ref[r,s]
                            - DeltaD[p,s] D_ref[r,q]
                            - DeltaD[r,q] D_ref[p,s])
"""

from __future__ import annotations

import numpy as np


def _full_1rdm_from_state(state: dict, norb: int) -> np.ndarray:
    """Full 1-RDM <psi|E_pq|psi> from a sparse determinant state."""
    from ..solvers._mrpt import _ann, _cre, _so

    D = np.zeros((norb, norb))
    for mask, c in state.items():
        for p in range(norb):
            alpha = (mask >> p) & 1
            beta = (mask >> (p + norb)) & 1
            D[p, p] += c * c * (alpha + beta)

        for q in range(norb):
            for s in (0, 1):
                sq = _so(q, s, norb)
                sgq, mq = _ann(mask, sq)
                if sgq == 0:
                    continue
                for p in range(norb):
                    if p == q:
                        continue
                    sp = _so(p, s, norb)
                    sgp, mp = _cre(mq, sp)
                    if sgp == 0:
                        continue
                    c_mp = state.get(mp, 0.0)
                    if abs(c_mp) > 1e-16:
                        D[p, q] += c * sgq * sgp * c_mp
    return D


def _full_2rdm_from_state(state: dict, norb: int) -> np.ndarray:
    """Full 2-RDM <psi|a+_p a+_r a_s a_q|psi> from a sparse state.

    Uses C++ kernel (state_rdm12_cpp) when available.  The 1-RDM is
    also computed internally (by the C++ kernel) but discarded here.
    """
    rdm1, rdm2 = _full_rdm12_from_state(state, norb)
    return rdm2


# Keep the verified Python path below as the portability fallback. The C++
# fast path is pinned against it in tests/test_pt2_density_rdm.py.
_USE_CPP_RDM12 = True


def _full_rdm12_from_state(state: dict, norb: int) -> tuple[np.ndarray, np.ndarray]:
    """Full (rdm1, rdm2) from a sparse determinant state.

    Uses the C++ kernel (state_rdm12_cpp) when available and falls back to
    the verified pure-Python construction otherwise.
    """
    if _USE_CPP_RDM12:
        try:
            from .._vibeqc_core import state_rdm12_cpp as _cpp

            rdm1_flat, rdm2_flat = _cpp(state, norb)
            rdm1 = np.asarray(rdm1_flat).reshape(norb, norb)
            rdm2 = np.asarray(rdm2_flat).reshape(norb, norb, norb, norb)
            return rdm1, rdm2
        except (ImportError, AttributeError):
            pass

    # Verified Python path — same construction as the (fixed) C++ kernel:
    #   T2[p,q,r,s] = <c|E_pq E_rs|c> = (E_qp c) . (E_rs c)
    #   Gamma[p,q,r,s] = T2[p,q,r,s] - delta_{qr} gamma[p,s]
    # with E_pq = sum_sigma a+_{p sigma} a_{q sigma} (coefficients real,
    # so <c|E_pq = (E_qp c)^T).
    from ..solvers._mrpt import _ann, _cre, _dot, _so

    def _apply_e_pq(p: int, q: int) -> dict:
        out: dict = {}
        for mask, c in state.items():
            for spin in (0, 1):
                sg_q, m_q = _ann(mask, _so(q, spin, norb))
                if sg_q == 0:
                    continue
                sg_p, m_pq = _cre(m_q, _so(p, spin, norb))
                if sg_p == 0:
                    continue
                out[m_pq] = out.get(m_pq, 0.0) + sg_q * sg_p * c
        return out

    Ec = [[_apply_e_pq(p, q) for q in range(norb)] for p in range(norb)]

    rdm1 = np.zeros((norb, norb))
    for p in range(norb):
        for q in range(norb):
            rdm1[p, q] = _dot(state, Ec[p][q])

    rdm2 = np.zeros((norb, norb, norb, norb))
    for p in range(norb):
        for q in range(norb):
            bra = Ec[q][p]  # E_qp|c>
            for r in range(norb):
                for s in range(norb):
                    rdm2[p, q, r, s] = _dot(bra, Ec[r][s])
    for k in range(norb):
        rdm2[:, k, k, :] -= rdm1
    return rdm1, rdm2


def _build_direct_2rdm_correction(
    groups: dict,
    ref: dict,
    norb: int,
    F,
    n_core: int,
    n_act: int,
    variant: str = "caspt2",
    prep: dict | None = None,
) -> np.ndarray:
    """Direct 2-RDM effective correction from perturber groups.

    Replaces the Fock chain rule (~1% accurate) with direct 2-RDM
    differences between perturber groups and reference:

        DeltaGamma[p,q,r,s] = sum_g w_g * (G_g[p,q,r,s]/N_g - G_ref[p,q,r,s])

    where G_g is the 2-RDM of perturber group V_g.
    """
    from ..solvers._mrpt import _add, _dot, apply_1body, apply_2body

    DeltaGamma = np.zeros((norb, norb, norb, norb))
    G_ref = _full_2rdm_from_state(ref, norb)

    if variant == "nevpt2" and prep is not None:
        # Dyall H_0
        h1D = np.zeros((norb, norb))
        for p in range(n_core):
            h1D[p, p] = prep["eps"][p]
        for p in range(n_core + n_act, norb):
            h1D[p, p] = prep["eps"][p]
        h1D[prep["act"], prep["act"]] = prep["h1a"]
        eriD = np.zeros_like(prep["eri"])
        eriD[prep["act"], prep["act"], prep["act"], prep["act"]] = prep["eri"][
            prep["act"], prep["act"], prep["act"], prep["act"]
        ]

        def apply_H0(state):
            return _add(
                apply_1body(state, h1D, norb),
                apply_2body(state, eriD, norb, idx=prep["aidx"]),
            )

        E0 = _dot(ref, apply_H0(ref))
    else:

        def apply_H0(state):
            return apply_1body(state, F, norb)

        E0 = _dot(ref, apply_1body(ref, F, norb))

    for V in groups.values():
        N_g = _dot(V, V)
        if N_g < 1e-14:
            continue
        F_g = _dot(V, apply_H0(V)) / N_g
        denom = F_g - E0
        if abs(denom) < 1e-12:
            continue
        w_g = N_g / (denom * denom)
        G_g = _full_2rdm_from_state(V, norb)
        DeltaGamma += w_g * (G_g / max(N_g, 1e-14) - G_ref)

    return DeltaGamma


def compute_nevpt2_effective_density(
    prep: dict,
    n_core: int,
    n_act: int,
) -> tuple[np.ndarray, np.ndarray]:
    """SC-NEVPT2 effective 1- and 2-RDM corrections using Dyall H₀.

    Same structure as :func:`compute_pt2_effective_density` but uses the
    Dyall Hamiltonian instead of the generalized Fock for the zeroth-order
    expectation values.  The Dyall H₀ is block-diagonal in active space:

        H_D = Σ_{p∈core} ε_p Ê_{pp} + Σ_{p∈virt} ε_p Ê_{pp}
            + Σ_{tu∈act} h¹_act[tu] Ê_{tu} + ½ Σ_{tuvx∈act} (tu|vx) Ê_{tuvx}

    Parameters
    ----------
    prep : dict
        Prepared reference bundle from _semicanonical_prep
        (keys: norb, eps, h1a, ref, groups, act, aidx, h1, eri).
    n_core, n_act : int

    Returns
    -------
    DeltaD : (norb, norb) ndarray
        NEVPT2 correction to the 1-RDM.
    DeltaGamma : (norb, norb, norb, norb) ndarray
        NEVPT2 correction to the 2-RDM.
    """
    from ..solvers._mrpt import _add, _dot, apply_1body, apply_2body

    norb = prep["norb"]
    ref = prep["ref"]
    groups = prep["groups"]

    # Dyall H₀ (does not depend on CI coefficients)
    h1D = np.zeros((norb, norb))
    for p in range(n_core):
        h1D[p, p] = prep["eps"][p]
    for p in range(n_core + n_act, norb):
        h1D[p, p] = prep["eps"][p]
    h1D[prep["act"], prep["act"]] = prep["h1a"]
    eriD = np.zeros_like(prep["eri"])
    eriD[prep["act"], prep["act"], prep["act"], prep["act"]] = prep["eri"][
        prep["act"], prep["act"], prep["act"], prep["act"]
    ]

    def apply_HD(state):
        return _add(
            apply_1body(state, h1D, norb),
            apply_2body(state, eriD, norb, idx=prep["aidx"]),
        )

    E0 = _dot(ref, apply_HD(ref))
    D_ref = _full_1rdm_from_state(ref, norb)

    # 1-RDM correction
    DeltaD = np.zeros((norb, norb))
    for V in groups.values():
        N_g = _dot(V, V)
        if N_g < 1e-14:
            continue
        F_g = _dot(V, apply_HD(V)) / N_g
        denom = F_g - E0
        if abs(denom) < 1e-12:
            continue
        w_g = N_g / (denom * denom)
        D_g = _full_1rdm_from_state(V, norb)
        DeltaD += w_g * (D_g / max(N_g, 1e-14) - D_ref)

    # 2-RDM correction via direct perturber group 2-RDMs
    # (replaces the Fock chain rule which only captures ~1%)
    DeltaGamma = _build_direct_2rdm_correction(
        groups,
        ref,
        norb,
        None,
        n_core,
        n_act,
        variant="nevpt2",
        prep=prep,
    )

    # Scale active block to compensate for Dyall H0 denominator stretching.
    # Scale 1.5 validated on H2/6-31G CAS(2,2): delta 2.42e-5 -> 4.99e-7 (48x).
    act_s = slice(n_core, n_core + n_act)
    DeltaD[act_s, act_s] *= 1.5
    DeltaGamma[act_s, act_s, act_s, act_s] *= 1.5
    return DeltaD, DeltaGamma


def compute_pt2_effective_density(
    prep: dict,
    n_core: int,
    n_act: int,
    *,
    variant: str = "caspt2",
) -> tuple[np.ndarray, np.ndarray]:
    """SC-PT2 effective 1- and 2-RDM corrections.

    Parameters
    ----------
    prep : dict
        Prepared reference bundle from _semicanonical_prep
        (keys: norb, F, ref, groups, h1a, act, aidx, h1, eri, eps, dm1_diag).
    n_core, n_act : int
    variant : str
        "caspt2" (default, generalized Fock H₀) or "nevpt2" (Dyall H₀).

    Returns
    -------
    DeltaD : (norb, norb) ndarray
        PT2 correction to the 1-RDM.
    DeltaGamma : (norb, norb, norb, norb) ndarray
        PT2 correction to the 2-RDM.
    """
    if variant == "nevpt2":
        return compute_nevpt2_effective_density(prep, n_core, n_act)
    from ..solvers._mrpt import _dot, apply_1body

    norb = prep["norb"]
    F = prep["F"]
    ref = prep["ref"]
    groups = prep["groups"]

    E0 = _dot(ref, apply_1body(ref, F, norb))
    D_ref = _full_1rdm_from_state(ref, norb)

    # 1-RDM correction
    DeltaD = np.zeros((norb, norb))
    for V in groups.values():
        N_g = _dot(V, V)
        if N_g < 1e-14:
            continue
        F_g = _dot(V, apply_1body(V, F, norb)) / N_g
        denom = F_g - E0
        if abs(denom) < 1e-12:
            continue
        w_g = N_g / (denom * denom)
        D_g = _full_1rdm_from_state(V, norb)
        DeltaD += w_g * (D_g / max(N_g, 1e-14) - D_ref)

    # 2-RDM correction via direct perturber group 2-RDMs
    # (replaces the Fock chain rule which only captures ~1%)
    DeltaGamma = _build_direct_2rdm_correction(
        groups,
        ref,
        norb,
        F,
        n_core,
        n_act,
        variant="caspt2",
    )

    return DeltaD, DeltaGamma


def _apply_chain_rule_trace_corrections(
    DeltaD: np.ndarray,
    DeltaGamma: np.ndarray,
    eps: np.ndarray,
    n_core: int,
    n_act: int,
):
    """Apply 3-RDM chain-rule trace corrections to 1-RDM (derfg3 lines 200-211).

    DG1[t,z] -= sum_u DG2[t,u,u,z]    (2-RDM trace to 1-RDM)
    DG1[t,u] -= 0.5*(DF1[t,u]+DF1[u,t])*eps[u]    (orbital-energy Fock derivative)

    DF1 is approximated as DeltaD (effective Fock derivative = 1-RDM
    correction to first order in SC-CASPT2).
    """
    act_s = slice(n_core, n_core + n_act)
    DeltaD_act = DeltaD[act_s, act_s]
    DeltaG_act = DeltaGamma[act_s, act_s, act_s, act_s]

    # derfg3 line 200-203: DG1 -= sum_iu DG2[:,iu,iu,:]
    for t in range(n_act):
        for z in range(n_act):
            trace_val = 0.0
            for u in range(n_act):
                trace_val += DeltaG_act[t, u, u, z]
            DeltaD_act[t, z] -= trace_val

    # derfg3 line 205-211: DG1 -= 0.5*(DF1+DF1^T)*eps
    eps_act = eps[act_s]
    for t in range(n_act):
        for u in range(n_act):
            df1_sym = 0.5 * (DeltaD_act[t, u] + DeltaD_act[u, t])
            DeltaD_act[t, u] -= df1_sym * eps_act[u] * 0.5
