"""IC-CASPT2 effective densities and response right-hand sides.

The production unshifted explicit-engine gradient uses the exact stationary
Hylleraas density and the orbital/CI response helpers in this module.  The
older CLagDX BDER/SDER component scaffold remains as a diagnostic oracle for
the shifted and direct-engine extensions that are still fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ICCASPT2DensityComponents:
    """Container for IC-CASPT2 density ingredients in semicanonical MO space."""

    e_corr: float
    n_ic: int
    bder: np.ndarray
    sder: np.ndarray
    fg3: dict[str, np.ndarray]
    delta_d: np.ndarray
    delta_gamma: np.ndarray
    depsa: np.ndarray
    psi1: dict[int, float]
    w: dict[int, float]
    clagdx: dict[str, Any]


@dataclass(frozen=True)
class ICCASPT2EffectiveDensity:
    """Exact unshifted IC-CASPT2 Hylleraas effective density."""

    e_corr: float
    n_ic: int
    norm_psi1: float
    delta_d: np.ndarray
    delta_gamma: np.ndarray
    psi1: dict[int, float]


def _transition_rdm12(
    bra: dict[int, float],
    ket: dict[int, float],
    norb: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Spin-summed transition 1-/2-RDMs ``<bra|E...|ket>``.

    The 2-RDM convention matches :func:`._pt2_density._full_rdm12_from_state`
    and therefore contracts directly with chemist-ordered ERIs.
    """
    from ..solvers._mrpt import _dot, apply_1body

    e_ket: list[list[dict[int, float]]] = []
    e_bra_t: list[list[dict[int, float]]] = []
    for p in range(norb):
        ket_row = []
        bra_row = []
        for q in range(norb):
            generator = np.zeros((norb, norb))
            generator[p, q] = 1.0
            ket_row.append(apply_1body(ket, generator, norb))
            generator[p, q] = 0.0
            generator[q, p] = 1.0
            bra_row.append(apply_1body(bra, generator, norb))
        e_ket.append(ket_row)
        e_bra_t.append(bra_row)

    dm1 = np.zeros((norb, norb))
    for p in range(norb):
        for q in range(norb):
            dm1[p, q] = _dot(bra, e_ket[p][q])

    dm2 = np.zeros((norb, norb, norb, norb))
    for p in range(norb):
        for q in range(norb):
            bra_epq = e_bra_t[p][q]
            for r in range(norb):
                for s in range(norb):
                    dm2[p, q, r, s] = _dot(bra_epq, e_ket[r][s])
    for q in range(norb):
        dm2[:, q, q, :] -= dm1
    return dm1, dm2


def compute_ic_caspt2_effective_density(
    prep: dict[str, Any],
    n_core: int,
    n_act: int,
    *,
    n_frozen: int = 0,
    thresh: float = 1e-8,
) -> ICCASPT2EffectiveDensity:
    r"""Return the exact fixed-reference IC-CASPT2 density correction.

    For the unshifted first-order solution, stationarity of the Hylleraas
    functional gives (Celani and Werner, JCP 112, 5546 (2000))::

        L2 = 2 <0|H|1> + <1|(F-E0)|1>.

    Thus its integral derivative needs only transition RDMs and
    ``R = gamma11 - <1|1> gamma00``.  The generalized-Fock chain rule is
    evaluated explicitly, avoiding the historical CLagDX diagonal
    approximation.  Returned tensors are rotated from the semicanonical
    working basis back to the caller's MO basis.
    """
    from ..solvers._mrpt import _dot, _ic_caspt2_solve
    from ._pt2_density import _full_rdm12_from_state

    e_corr, n_ic, psi1, _w, _clagdx = _ic_caspt2_solve(
        prep,
        n_core,
        n_act,
        n_frozen=n_frozen,
        ipea=0.0,
        imaginary=0.0,
        thresh=thresh,
        return_wavefunction=True,
    )
    norb = int(prep["norb"])
    ref = prep["ref"]
    norm_psi1 = float(_dot(psi1, psi1))

    d00, _g00 = _full_rdm12_from_state(ref, norb)
    d11, _g11 = _full_rdm12_from_state(psi1, norb)
    d01, g01 = _transition_rdm12(ref, psi1, norb)
    d10, g10 = _transition_rdm12(psi1, ref, norb)

    # d<1|(F-E0)|1>/dh = D11 - <1|1>D00.  Since F is linear in
    # h and in the reference-density mean field, the same R matrix drives
    # the two-electron chain rule.
    response_d = d11 - norm_psi1 * d00
    delta_d_sc = d01 + d10 + response_d
    delta_g_sc = g01 + g10
    delta_g_sc += 2.0 * np.einsum("pq,rs->pqrs", response_d, d00)
    delta_g_sc -= np.einsum("ps,qr->pqrs", response_d, d00)

    rot = np.asarray(prep.get("rot", np.eye(norb)), dtype=float)
    delta_d = rot @ delta_d_sc @ rot.T
    delta_gamma = np.einsum(
        "pi,qj,rk,sl,ijkl->pqrs",
        rot,
        rot,
        rot,
        rot,
        delta_g_sc,
        optimize=True,
    )
    delta_d = 0.5 * (delta_d + delta_d.T)

    return ICCASPT2EffectiveDensity(
        e_corr=float(e_corr),
        n_ic=int(n_ic),
        norm_psi1=norm_psi1,
        delta_d=delta_d,
        delta_gamma=delta_gamma,
        psi1=psi1,
    )


def _active_reference_rdms_for_fg3(
    ref: dict[int, float],
    norb: int,
    n_core: int,
    n_act: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return active-space G1/G2/G3 in the convention CLagDXA_FG3 expects."""
    from .._vibeqc_core import state_rdm3_cpp
    from ._pt2_density import _full_1rdm_from_state, _full_rdm12_from_state

    act_s = slice(n_core, n_core + n_act)
    g1 = _full_1rdm_from_state(ref, norb)[act_s, act_s]

    _rdm1, rdm2 = _full_rdm12_from_state(ref, norb)
    rdm2_4d = rdm2.reshape(norb, norb, norb, norb)
    g2 = np.zeros((n_act, n_act, n_act, n_act))
    for t in range(n_act):
        for u in range(n_act):
            for v in range(n_act):
                for x in range(n_act):
                    tt = n_core + t
                    uu = n_core + u
                    vv = n_core + v
                    xx = n_core + x
                    g2[t, u, v, x] = rdm2_4d[tt, xx, vv, uu]
                    if u == v:
                        g2[t, u, v, x] -= g1[t, x]

    active_ref = _compact_active_reference_state(ref, norb, n_core, n_act)
    g3 = np.asarray(state_rdm3_cpp(active_ref, n_act, n_act))
    return g1, g2, g3


def _compact_active_reference_state(
    ref: dict[int, float],
    norb: int,
    n_core: int,
    n_act: int,
) -> dict[int, float]:
    """Project a full-space CAS reference onto active-relative determinants."""
    active_ref: dict[int, float] = {}
    for mask, coeff in ref.items():
        active_mask = 0
        for t in range(n_act):
            p = n_core + t
            if (mask >> p) & 1:
                active_mask |= 1 << t
            if (mask >> (p + norb)) & 1:
                active_mask |= 1 << (t + n_act)
        active_ref[active_mask] = active_ref.get(active_mask, 0.0) + coeff
    return {mask: c for mask, c in active_ref.items() if abs(c) > 1e-14}


def compute_ic_caspt2_density_components(
    prep: dict[str, Any],
    n_core: int,
    n_act: int,
    *,
    n_frozen: int = 0,
    ipea: float = 0.0,
    imaginary: float = 0.0,
    thresh: float = 1e-8,
) -> ICCASPT2DensityComponents:
    """Build the currently available IC-CASPT2 gradient density ingredients.

    The one-body correction stored in ``delta_d`` is the active-space
    ``BDER + FG3.DG1`` block described in the CAS gradient handover.  The
    two-body correction stores the active ``FG3.DG2`` block.  These tensors are
    intentionally returned as components for the future IC gradient framework;
    callers must not mix them into the SC-CASPT2 z-vector shortcut.
    """
    from ..solvers._mrpt import _ic_caspt2_solve
    from ._clagdx import compute_bder_sder_from_ic
    from ._clagdxa_fg3 import clagdxa_fg3_contract_full

    norb = int(prep["norb"])
    act_s = slice(n_core, n_core + n_act)

    e_corr, n_ic, psi1, w, clagdx = _ic_caspt2_solve(
        prep,
        n_core,
        n_act,
        n_frozen=n_frozen,
        ipea=ipea,
        imaginary=imaginary,
        thresh=thresh,
        return_wavefunction=True,
    )
    bder, sder = compute_bder_sder_from_ic(clagdx, n_core, n_act, norb)

    g1, g2, g3 = _active_reference_rdms_for_fg3(
        prep["ref"], norb, n_core, n_act
    )
    fg3 = clagdxa_fg3_contract_full(
        bder,
        sder,
        g3,
        g1,
        g2,
        np.asarray(prep["eps"])[act_s],
        n_act,
    )

    delta_d = np.zeros((norb, norb))
    delta_gamma = np.zeros((norb, norb, norb, norb))
    delta_d[act_s, act_s] = bder + fg3["DG1"]
    delta_gamma[act_s, act_s, act_s, act_s] = fg3["DG2"]

    return ICCASPT2DensityComponents(
        e_corr=float(e_corr),
        n_ic=int(n_ic),
        bder=bder,
        sder=sder,
        fg3=fg3,
        delta_d=delta_d,
        delta_gamma=delta_gamma,
        depsa=fg3["DEPSA"],
        psi1=psi1,
        w=w,
        clagdx=clagdx,
    )


def compute_ic_caspt2_orbital_gradient_fd(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_core: int,
    n_act: int,
    n_act_elec: int,
    pairs: list[tuple[int, int]],
    *,
    fd_eps: float = 1e-3,
    n_frozen: int = 0,
    ipea: float = 0.0,
    imaginary: float = 0.0,
    thresh: float = 1e-8,
    reference: tuple[np.ndarray, list] | None = None,
) -> np.ndarray:
    """Compute the IC-CASPT2 orbital RHS ``dE2/dkappa`` by central FD.

    This is the IC analogue of the numerical SC-CASPT2 RHS currently embedded
    in ``_caspt2.py``.  It is intentionally kept as a small deterministic
    oracle until the complete IC Hessian/Z-vector framework is wired.
    """
    from scipy.linalg import expm

    from ..solvers._mrpt import _ic_caspt2_solve, _semicanonical_prep

    nmo = int(h1e_mo.shape[0])
    npr = len(pairs)
    if npr == 0:
        return np.zeros(0)

    g_phys_flat = np.ascontiguousarray(h2e_mo.ravel(), dtype=float)
    xform4 = None
    try:
        from .._vibeqc_core import transform_4index_mo as _xform4

        xform4 = _xform4
    except (ImportError, AttributeError):
        pass

    def e2_at(h1: np.ndarray, h2: np.ndarray) -> float:
        prep = _semicanonical_prep(
            h1,
            h2,
            n_core,
            n_act,
            n_act_elec,
            0,
            reference=reference,
        )
        e2, _nb = _ic_caspt2_solve(
            prep,
            n_core,
            n_act,
            n_frozen=n_frozen,
            ipea=ipea,
            imaginary=imaginary,
            thresh=thresh,
        )
        return float(e2)

    dE2_dk = np.zeros(npr)
    for i, (p, q) in enumerate(pairs):
        K = np.zeros((nmo, nmo))
        K[p, q] = fd_eps
        K[q, p] = -fd_eps
        Up = expm(K)
        Um = expm(-K)
        h1p = Up.T @ h1e_mo @ Up
        h1m = Um.T @ h1e_mo @ Um
        if xform4 is not None:
            gp = np.asarray(
                xform4(g_phys_flat, np.ascontiguousarray(Up, dtype=float), nmo)
            ).reshape(nmo, nmo, nmo, nmo)
            gm = np.asarray(
                xform4(g_phys_flat, np.ascontiguousarray(Um, dtype=float), nmo)
            ).reshape(nmo, nmo, nmo, nmo)
        else:
            gp = np.einsum("ap,bq,cr,ds,abcd->pqrs", Up, Up, Up, Up, h2e_mo)
            gm = np.einsum("ap,bq,cr,ds,abcd->pqrs", Um, Um, Um, Um, h2e_mo)
        dE2_dk[i] = (e2_at(h1p, gp) - e2_at(h1m, gm)) / (2.0 * fd_eps)
    return dE2_dk


def compute_ic_caspt2_ci_gradient_fd(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    ci_coeffs: np.ndarray,
    determinants: list,
    n_core: int,
    n_act: int,
    n_act_elec: int,
    *,
    fd_eps: float = 1e-3,
    n_frozen: int = 0,
    thresh: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``dE2/dc`` in the CASCI eigenbasis and that eigensystem.

    Only the response RHS is differentiated numerically.  Nuclear derivatives
    remain analytic integral contractions; this deterministic small-CAS oracle
    includes the reference dependence of the internally contracted FOIS.
    """
    from ..solvers._mrpt import _ic_caspt2_solve, _semicanonical_prep
    from ._casscf import _build_ci_hamiltonian

    h_ci = _build_ci_hamiltonian(h1e_mo, h2e_mo, determinants, n_core, n_act)
    energies, eigenvectors = np.linalg.eigh(h_ci)
    n_det = len(determinants)
    if n_det <= 1:
        return np.zeros(0), energies, eigenvectors

    c0 = np.asarray(ci_coeffs, dtype=float)
    c0 = c0 / np.linalg.norm(c0)
    gradient = np.zeros(n_det - 1)

    def e2_for(c_ref: np.ndarray) -> float:
        prep = _semicanonical_prep(
            h1e_mo,
            h2e_mo,
            n_core,
            n_act,
            n_act_elec,
            0,
            reference=(c_ref, determinants),
        )
        e2, _n_ic = _ic_caspt2_solve(
            prep,
            n_core,
            n_act,
            n_frozen=n_frozen,
            ipea=0.0,
            imaginary=0.0,
            thresh=thresh,
        )
        return float(e2)

    for k in range(1, n_det):
        direction = eigenvectors[:, k].copy()
        direction -= c0 * float(c0 @ direction)
        direction /= np.linalg.norm(direction)
        c_plus = np.cos(fd_eps) * c0 + np.sin(fd_eps) * direction
        c_minus = np.cos(fd_eps) * c0 - np.sin(fd_eps) * direction
        gradient[k - 1] = (e2_for(c_plus) - e2_for(c_minus)) / (2.0 * fd_eps)
    return gradient, energies, eigenvectors


def build_ic_caspt2_orbital_hessian_fd(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_core: int,
    n_act: int,
    n_act_elec: int,
    pairs: list[tuple[int, int]],
    *,
    fd_eps_hess: float = 1e-3,
    fd_eps_grad: float = 1e-3,
    n_frozen: int = 0,
    ipea: float = 0.0,
    imaginary: float = 0.0,
    thresh: float = 1e-8,
) -> np.ndarray:
    """Build the IC-CASPT2 orbital Hessian contribution by FD of the IC RHS."""
    from scipy.linalg import expm

    nmo = int(h1e_mo.shape[0])
    npr = len(pairs)
    if npr == 0:
        return np.zeros((0, 0))

    H = np.zeros((npr, npr))
    for j, (p, q) in enumerate(pairs):
        Kp = np.zeros((nmo, nmo))
        Km = np.zeros((nmo, nmo))
        Kp[p, q] = fd_eps_hess
        Kp[q, p] = -fd_eps_hess
        Km[p, q] = -fd_eps_hess
        Km[q, p] = fd_eps_hess
        Up = expm(Kp)
        Um = expm(Km)
        h1p = Up.T @ h1e_mo @ Up
        h1m = Um.T @ h1e_mo @ Um
        gp = np.einsum("ap,bq,cr,ds,abcd->pqrs", Up, Up, Up, Up, h2e_mo)
        gm = np.einsum("ap,bq,cr,ds,abcd->pqrs", Um, Um, Um, Um, h2e_mo)
        gp_grad = compute_ic_caspt2_orbital_gradient_fd(
            h1p,
            gp,
            n_core,
            n_act,
            n_act_elec,
            pairs,
            fd_eps=fd_eps_grad,
            n_frozen=n_frozen,
            ipea=ipea,
            imaginary=imaginary,
            thresh=thresh,
        )
        gm_grad = compute_ic_caspt2_orbital_gradient_fd(
            h1m,
            gm,
            n_core,
            n_act,
            n_act_elec,
            pairs,
            fd_eps=fd_eps_grad,
            n_frozen=n_frozen,
            ipea=ipea,
            imaginary=imaginary,
            thresh=thresh,
        )
        H[:, j] = (gp_grad - gm_grad) / (2.0 * fd_eps_hess)
    return 0.5 * (H + H.T)
