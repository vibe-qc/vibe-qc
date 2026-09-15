"""Multi-reference perturbation theory: NEVPT2 and CASPT2.

NEVPT2 (N-Electron Valence Perturbation Theory), strongly-contracted variant
  C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu,
  J. Chem. Phys. 114, 10252 (2001).

CASPT2 (Complete Active Space Perturbation Theory)
  internally-contracted (default): K. Andersson, P.-A. Malmqvist, B. O. Roos,
    J. Chem. Phys. 96, 1218 (1992); P. Celani, H.-J. Werner, J. Chem. Phys.
    112, 5546 (2000).
  strongly-contracted (``variant="sc"``, gated): K. Andersson, P.-A. Malmqvist,
    B. O. Roos, A. J. Sadlej, K. Wolinski, J. Phys. Chem. 94, 5483 (1990).

All add a second-order perturbative correction on top of a CASCI reference.
They differ in the zeroth-order Hamiltonian H₀: NEVPT2 uses Dyall's
Hamiltonian (a true two-body operator in the active space); CASPT2 uses a
one-body generalized-Fock H₀.

Implementation
--------------
The strongly-contracted NEVPT2 and CASPT2 correlation energies are computed
from the defining expression -- one perturber ``|V_g>`` per external (core-hole
/ virtual-particle) excitation pattern ``g``,

    E^(2) = S_g -<V_g|V_g> / ( <V_g|H₀|V_g>/<V_g|V_g> - E₀ ) ,

where ``|V_g>`` is the component of ``H|0>`` with external pattern ``g``
(``H₀`` = Dyall's H_D for NEVPT2, generalized Fock for SC-CASPT2,
``E₀ = <0|H₀|0>``).  The perturbers are built and contracted with a small
spin-orbital second-quantization engine (:func:`apply_1body` /
:func:`apply_2body`), so every fermionic phase, spin-summation and
combinatorial factor is handled by construction rather than by per-class
formulas.  Orbitals are semicanonicalized (the core and virtual blocks of the
generalized Fock are diagonalized); the active CI vector is invariant under
that rotation.

The default **internally-contracted CASPT2** (:func:`_ic_caspt2_corr`) instead
builds the full contracted first-order interacting space ``{Ê_pr Ê_qs|0>}``
explicitly in the determinant basis and solves ``(H₀ - E₀) C = -V`` with the
nondiagonal generalized-Fock H₀ (CASPT2N) -- exact internal contraction, with
optional frozen core, imaginary level shift (Forsberg-Malmqvist 1997) and IPEA
shift (Ghigo 2004).

This is a small-active-space implementation (the spin-orbital / dense
contraction engines scale steeply with the orbital count), consistent with the
rest of :mod:`vibeqc.solvers`.  NEVPT2 is validated against PySCF
``mrpt.NEVPT`` per excitation class to ~1e-9 Ha; internally-contracted CASPT2
against OpenMolcas ``&CASPT2`` to <=2 µHa (see
``tests/test_solvers_mrpt_parity.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

from ._casci import CASCIResult, _frozen_core_dressing, casci
from ._common import require_experimental_multiref
from ._rdm import energy_from_rdms, make_rdm1, make_rdm12


@dataclass
class NEVPT2Result:
    """Multi-reference PT2 result (NEVPT2 / CASPT2).

    Attributes
    ----------
    e_total : float
        Total energy: reference (CASCI) energy + second-order correction.
    e_corr : float
        Second-order correlation energy.
    e_ref : float
        Reference (CASCI) total energy.
    """

    e_total: float
    e_corr: float
    e_ref: float


@dataclass
class CASPT2Options:
    """Options for ``run_job(method="caspt2", caspt2_options=...)``.

    Attributes
    ----------
    variant : str
        ``"ic"`` (internally-contracted, default, validated vs OpenMolcas and
        un-gated) or ``"sc"`` (strongly-contracted, gated).
    ipea : float
        IPEA level shift e (Ghigo 2004).  Non-zero uses the canonical
        eight-class internally contracted basis; see :func:`caspt2`.
    imaginary : float
        Imaginary level shift s for intruder-state removal (Forsberg-Malmqvist
        1997).  ``0.0`` disables it.
    n_frozen : int
        Number of deepest inactive orbitals kept frozen (not correlated).
    multistate : str or None
        ``None`` (single-state, default), ``"ms"`` (MS-CASPT2, Finley 1998)
        or ``"xms"`` (XMS-CASPT2, Granovsky 2011 / Shiozaki 2011): build and
        diagonalize the multi-state effective Hamiltonian over the lowest
        ``nroots`` spin-pure CASCI roots of the reference basis.  See
        :func:`vibeqc.solvers._ms_caspt2.ms_caspt2`.
    nroots : int
        Number of model states for ``multistate``.  ``0`` (default) inherits
        ``casscf_options.nroots`` when a CASSCF reference is composed,
        otherwise it must be set explicitly (>= 2 to be meaningful).
    engine : str
        ``"auto"`` (default) keeps the validated explicit/direct dispatch.
        ``"cases"`` selects the experimental case-partitioned, matrix-free RDM
        CASPT2 (roadmap 25i; ``variant="ic"``, single-state, unshifted,
        ``ipea=0``); see :func:`caspt2`.
    compute_corr_grad : bool
        If True, request the full relaxed correlation gradient.  The
        single-state analytic Lagrangian is production for unshifted
        ``variant="ic"`` closed-shell CASSCF references on the explicit
        small-space engine.  Unsupported shifts, engines, and references
        fail closed in direct gradient calls and use full-energy finite
        differences in geometry optimizers.
    nac_pair : tuple[int, int] or None
        Ordered pair of physical MS/XMS roots for which to compute the
        nonadiabatic derivative-coupling vector.  ``None`` (default) skips
        it.  The current under-review route requires an unshifted, explicit
        IC-CASPT2 calculation on an exact, closed-shell SA-CASSCF reference.
    nac_fd_step : float
        Central nuclear-displacement step in bohr for ``nac_pair``.  Each
        component reoptimizes the SA-CASSCF reference at both displacements.
    nac_gap_tolerance : float
        Minimum physical-state energy gap in Hartree.  Below this threshold
        the adiabatic derivative coupling is gauge singular and fails closed.
    use_zvector : bool
        Compatibility flag retained for callers of the historical gradient
        helpers.  The public analytic IC-CASPT2 route always solves its full
        coupled orbital/CI response equations.
    use_lagrangian : bool
        Compatibility flag retained for the historical SC-CASPT2 gradient
        helper.  It does not change the public analytic IC route.
    """

    variant: str = "ic"
    ipea: float = 0.0
    imaginary: float = 0.0
    n_frozen: int = 0
    multistate: Optional[str] = None
    nroots: int = 0
    engine: str = "auto"
    compute_corr_grad: bool = False
    nac_pair: Optional[tuple[int, int]] = None
    nac_fd_step: float = 1e-3
    nac_gap_tolerance: float = 1e-6
    use_zvector: bool = True
    use_lagrangian: bool = False


@dataclass
class NEVPT2Options:
    """Options for run_job(method="nevpt2", nevpt2_options=...).

    Attributes
    ----------
    compute_corr_grad : bool
        If True, include the SC-NEVPT2 correlation energy gradient via
        numerical FD in SolverResult.gradient.
    use_zvector : bool
        Compatibility flag for the historical semi-analytic Z-vector
        correction.  The public runner differentiates the full relaxed
        NEVPT2 total energy by central finite difference when
        ``compute_corr_grad=True``.
    use_lagrangian : bool
        If True, use the full CP-MCSCF Lagrangian gradient (experimental).
        Default ``False``.
    """

    compute_corr_grad: bool = False
    use_zvector: bool = True
    use_lagrangian: bool = False


# ── Spin-orbital second-quantization engine ─────────────────────────────────
# A state is a dict {bitmask -> coefficient}.  Spin-orbital index of spatial
# orbital ``p`` with spin ``s`` (0=a, 1=b) is ``p + s*norb``.


def _so(p: int, s: int, norb: int) -> int:
    return p + s * norb


def _ann(mask: int, x: int):
    if not (mask >> x) & 1:
        return 0, 0
    sign = -1 if bin(mask & ((1 << x) - 1)).count("1") & 1 else 1
    return sign, mask & ~(1 << x)


def _cre(mask: int, x: int):
    if (mask >> x) & 1:
        return 0, 0
    sign = -1 if bin(mask & ((1 << x) - 1)).count("1") & 1 else 1
    return sign, mask | (1 << x)


def apply_1body(state: dict, h1: np.ndarray, norb: int, idx=None) -> dict:
    """Apply ``S_pq h1_pq S_s a+_{ps} a_{qs}`` to ``state``.

    Uses the C++ kernel when available for performance.
    """
    try:
        from .._vibeqc_core import apply_1body_cpp

        idx_list = list(idx) if idx is not None else None
        return apply_1body_cpp(state, np.ascontiguousarray(h1), norb, idx_list)
    except (ImportError, AttributeError):
        pass
    # Python fallback
    rng = range(norb) if idx is None else idx
    out: dict = {}
    for mask, c in state.items():
        for q in rng:
            for s in (0, 1):
                sgq, mq = _ann(mask, _so(q, s, norb))
                if sgq == 0:
                    continue
                for p in rng:
                    hpq = h1[p, q]
                    if hpq == 0.0:
                        continue
                    sgp, mp = _cre(mq, _so(p, s, norb))
                    if sgp:
                        out[mp] = out.get(mp, 0.0) + sgq * sgp * hpq * c
    return out


def apply_2body(state: dict, eri: np.ndarray, norb: int, idx=None) -> dict:
    """Apply ``1/2 S_pqrs (pq|rs) S_st a+_{ps} a+_{rt} a_{st} a_{qs}``.

    ``eri`` is chemist's ``(pq|rs)``.  Uses the C++ kernel when available.
    """
    try:
        from .._vibeqc_core import apply_2body_cpp

        idx_list = list(idx) if idx is not None else None
        return apply_2body_cpp(state, np.ascontiguousarray(eri).ravel(), norb, idx_list)
    except (ImportError, AttributeError):
        pass
    # Python fallback
    rng = range(norb) if idx is None else idx
    out: dict = {}
    for mask, c in state.items():
        for q in rng:
            for sq in (0, 1):
                sgq, mq = _ann(mask, _so(q, sq, norb))
                if sgq == 0:
                    continue
                for s in rng:
                    for st in (0, 1):
                        sgs, ms = _ann(mq, _so(s, st, norb))
                        if sgs == 0:
                            continue
                        for r in rng:
                            sgr, mr = _cre(ms, _so(r, st, norb))
                            if sgr == 0:
                                continue
                            for p in rng:
                                v = eri[p, q, r, s]
                                if v == 0.0:
                                    continue
                                sgp, mp = _cre(mr, _so(p, sq, norb))
                                if sgp:
                                    out[mp] = out.get(mp, 0.0) + (
                                        0.5 * sgq * sgs * sgr * sgp * v * c
                                    )
    return out


def _add(a: dict, b: dict) -> dict:
    try:
        from .._vibeqc_core import add_cpp

        return add_cpp(a, b)
    except (ImportError, AttributeError):
        pass
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, 0.0) + c
    return out


def _dot(a: dict, b: dict) -> float:
    try:
        from .._vibeqc_core import dot_cpp

        return dot_cpp(a, b)
    except (ImportError, AttributeError):
        pass
    if len(b) < len(a):
        a, b = b, a
    return sum(c * b.get(m, 0.0) for m, c in a.items())


def _build_reference_state(ci, determinants, n_core, norb) -> dict:
    """Reference |0> in the full spin-orbital space (core doubly occupied)."""
    core_mask = 0
    for i in range(n_core):
        core_mask |= (1 << _so(i, 0, norb)) | (1 << _so(i, 1, norb))
    state: dict = {}
    for I, (a_occ, b_occ) in enumerate(determinants):
        ci_I = float(ci[I])
        if abs(ci_I) < 1e-14:
            continue
        m = core_mask
        for t in a_occ:
            m |= 1 << _so(n_core + t, 0, norb)
        for t in b_occ:
            m |= 1 << _so(n_core + t, 1, norb)
        state[m] = state.get(m, 0.0) + ci_I
    return state


# ── Orbital preparation (shared by NEVPT2 and CASPT2) ───────────────────────


def _generalized_fock(h1, eri_chem, dm1_active, n_core, n_act):
    """Generalized Fock F = h + core mean field + active(1-RDM) mean field."""
    F = h1.copy()
    for i in range(n_core):
        F += 2.0 * eri_chem[:, :, i, i] - eri_chem[:, i, i, :]
    aidx = range(n_core, n_core + n_act)
    for ti, t in enumerate(aidx):
        for ui, u in enumerate(aidx):
            d = dm1_active[ti, ui]
            if abs(d) > 1e-14:
                F += d * (eri_chem[:, :, t, u] - 0.5 * eri_chem[:, t, u, :])
    return F


def _semicanonical_prep(
    h1e_mo,
    h2e_mo,
    n_core,
    n_act,
    n_act_elec,
    ms2,
    pseudocanonical_active=False,
    reference=None,
):
    """Semicanonicalize core+virtual and cache everything the PT2 needs.

    By default the active block is left untouched, so the active CI vector is
    unchanged by the rotation (the SC NEVPT2 / CASPT2 H₀ are active-rotation
    invariant).  When ``pseudocanonical_active=True`` the active block of the
    generalized Fock is diagonalized too (fully pseudocanonical orbitals) and
    the CI vector is recomputed in that basis; the IC-CASPT2 IPEA shift
    (Ghigo 2004) is defined in this basis because it modifies the diagonal
    Fock elements using the active occupation numbers there.

    ``reference`` (a ``(ci_coeffs, determinants)`` pair in the *input* orbital
    basis) makes the PT2 build its zeroth order on that explicit reference
    wavefunction instead of re-solving the full CAS via :func:`casci`.  This
    is the selected-CI -> MR-PT2 path (roadmap 25i): the reference may be a
    *truncated* selected-CI wavefunction whose full CAS is intractable.
    Because the semicanonicalization only rotates the core and virtual
    blocks, the active block (and therefore the active CI vector) is
    invariant, so the supplied wavefunction is still the reference in the
    rotated basis (no re-solve, no high-order RDMs).  Not combinable with
    ``pseudocanonical_active`` (the active rotation would change the supplied
    CI vector; IPEA on a selected reference is unsupported and fails closed).

    Returns a dict with the semicanonical integrals, the generalized Fock
    ``F`` (for the CASPT2 H₀), the reference state ``ref``, the bare ``H|0>``
    (``W``) and its external-pattern perturber groups, the active core-dressing
    for the NEVPT2 Dyall H₀, and ``dm1_diag`` (the active 1-RDM diagonal in the
    final basis, used by the IPEA shift).
    """
    if reference is not None and pseudocanonical_active:
        raise ValueError(
            "_semicanonical_prep: an explicit reference is not combinable "
            "with pseudocanonical_active (the active-block rotation would "
            "change the supplied CI vector); IPEA on a selected reference "
            "is unsupported"
        )
    norb = h1e_mo.shape[0]
    h1 = np.asarray(h1e_mo, dtype=float).copy()
    eri = np.asarray(h2e_mo, dtype=float).transpose(0, 2, 1, 3).copy()  # chemist

    if reference is not None:
        ref_ci, ref_dets = reference
        dm1 = make_rdm1(ref_ci, ref_dets, n_act)
    else:
        res0 = casci(
            h1,
            h2e_mo,
            n_active_elec=n_act_elec,
            n_active_orb=n_act,
            n_core=n_core,
            nuclear_repulsion=0.0,
            ms2=ms2,
        )
        dm1 = make_rdm1(res0.ci_coeffs, res0.determinants, n_act)

    F = _generalized_fock(h1, eri, dm1, n_core, n_act)
    blocks = [slice(0, n_core), slice(n_core + n_act, norb)]
    if pseudocanonical_active:
        blocks.append(slice(n_core, n_core + n_act))
    rot = np.eye(norb)
    for blk in blocks:
        if blk.stop - blk.start > 0:
            _, U = eigh(F[blk, blk])
            rot[blk, blk] = U
    h1 = rot.T @ h1 @ rot
    eri = np.einsum("pi,qj,rk,sl,pqrs->ijkl", rot, rot, rot, rot, eri, optimize=True)
    F = rot.T @ F @ rot
    eps = np.diag(F).copy()
    g = eri.transpose(0, 2, 1, 3)  # physicist, semicanonical

    act = slice(n_core, n_core + n_act)
    _, h1a = _frozen_core_dressing(h1, g, n_core, act)  # core-dressed active h

    if reference is not None:
        # Active block untouched by the core/virtual rotation, so the
        # supplied (possibly truncated) wavefunction is still the reference
        # here.  e0_act = <0|Ĥ_act|0> from the reference's own (truncated-
        # list-exact) 1-/2-RDMs and the rotated active integrals.
        ref = _build_reference_state(ref_ci, ref_dets, n_core, norb)
        dm1_act, dm2_act = make_rdm12(ref_ci, ref_dets, n_act)
        g_act = np.ascontiguousarray(g[act, act, act, act])
        e0_act = energy_from_rdms(h1a, g_act, dm1_act, dm2_act, 0.0)
        dm1_diag = np.diag(dm1_act).copy()
    else:
        res = casci(
            h1,
            g,
            n_active_elec=n_act_elec,
            n_active_orb=n_act,
            n_core=n_core,
            nuclear_repulsion=0.0,
            ms2=ms2,
        )
        e0_act = res.e_total - res.e_core
        ref = _build_reference_state(res.ci_coeffs, res.determinants, n_core, norb)
        dm1_diag = np.diag(make_rdm1(res.ci_coeffs, res.determinants, n_act)).copy()

    aidx = list(range(n_core, n_core + n_act))

    # Bare H|0> and its external-pattern perturbers (shared by every H₀).
    W = _add(apply_1body(ref, h1, norb), apply_2body(ref, eri, norb))
    groups = _external_groups(W, n_core, n_act, norb)

    return dict(
        norb=norb,
        h1=h1,
        eri=eri,
        F=F,
        rot=rot,
        eps=eps,
        ref=ref,
        e0_act=e0_act,
        act=act,
        aidx=aidx,
        h1a=h1a,
        n_core=n_core,
        n_act=n_act,
        groups=groups,
        dm1_diag=dm1_diag,
    )


def _pt2_correction(P, apply_H0) -> float:
    """Strongly-contracted second-order energy for a given zeroth-order H₀.

    ``E = S_g -<V_g|V_g> / (<V_g|H0|V_g>/<V_g|V_g> - <0|H0|0>)`` over the
    external-pattern perturber groups.  NEVPT2 passes Dyall's H₀, CASPT2 the
    generalized-Fock H₀.
    """
    e0 = _dot(P["ref"], apply_H0(P["ref"]))
    e_corr = 0.0
    for V in P["groups"].values():
        n = _dot(V, V)
        if n < 1e-14:
            continue
        denom = _dot(V, apply_H0(V)) / n - e0
        if abs(denom) < 1e-12:
            continue
        e_corr -= n / denom
    return e_corr


def _external_groups(W, n_core, n_act, norb):
    """Group a state by external excitation pattern (per-orbital hole/particle
    counts), dropping the purely-internal component."""
    core_set = range(n_core)
    virt_set = range(n_core + n_act, norb)
    groups: dict = {}
    for mask, c in W.items():
        holes = []
        for i in core_set:
            occ = ((mask >> _so(i, 0, norb)) & 1) + ((mask >> _so(i, 1, norb)) & 1)
            if occ < 2:
                holes.append((i, 2 - occ))
        parts = []
        for a in virt_set:
            occ = ((mask >> _so(a, 0, norb)) & 1) + ((mask >> _so(a, 1, norb)) & 1)
            if occ > 0:
                parts.append((a, occ))
        key = (tuple(holes), tuple(parts))
        if key == ((), ()):
            continue
        groups.setdefault(key, {})[mask] = c
    return groups


# ── NEVPT2 ──────────────────────────────────────────────────────────────────


def nevpt2(
    casci_result: CASCIResult,
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_core: int = 0,
    n_virt: int = 0,
    *,
    selected_reference: bool = False,
) -> NEVPT2Result:
    """Strongly-contracted NEVPT2 on top of a CASCI reference.

    Parameters
    ----------
    casci_result : CASCIResult
        Reference from :func:`vibeqc.solvers.casci` (provides the active CI
        vector and determinant list).
    h1e_mo, h2e_mo : ndarray
        Full MO integrals (physicist's ``g``) the CASCI was built from.
    n_core, n_virt : int
        Numbers of inactive (doubly-occupied) and virtual orbitals.  The
        active orbital count is ``norb - n_core - n_virt``.
    selected_reference : bool
        When ``True``, build the PT2 zeroth order on ``casci_result``'s own
        (possibly truncated) wavefunction instead of re-solving the full CAS
        (the selected-CI -> NEVPT2 path, roadmap 25i).  The strongly-
        contracted NEVPT2 is evaluated in determinant space (no high-order
        RDMs), so it scales to the selected regime; the result is the NEVPT2
        of *that* reference and converges to full-CAS NEVPT2 as the
        selection tightens (identical at the full-selection limit).  Default
        ``False`` keeps the exact full-CAS reference (the OpenMolcas-pinned
        path).
    """
    norb = h1e_mo.shape[0]
    n_act = norb - n_core - n_virt
    a_occ0, b_occ0 = casci_result.determinants[0]
    n_act_elec = len(a_occ0) + len(b_occ0)
    ms2 = len(a_occ0) - len(b_occ0)

    reference = (
        (casci_result.ci_coeffs, casci_result.determinants)
        if selected_reference
        else None
    )
    P = _semicanonical_prep(
        h1e_mo, h2e_mo, n_core, n_act, n_act_elec, ms2, reference=reference
    )
    norb = P["norb"]

    # Dyall H₀: diag(e) on core/virtual, core-dressed h on active, active 2e.
    h1D = np.zeros((norb, norb))
    for p in range(n_core):
        h1D[p, p] = P["eps"][p]
    for p in range(n_core + n_act, norb):
        h1D[p, p] = P["eps"][p]
    h1D[P["act"], P["act"]] = P["h1a"]
    eriD = np.zeros_like(P["eri"])
    eriD[P["act"], P["act"], P["act"], P["act"]] = P["eri"][
        P["act"], P["act"], P["act"], P["act"]
    ]

    def apply_HD(state):
        return _add(
            apply_1body(state, h1D, norb),
            apply_2body(state, eriD, norb, idx=P["aidx"]),
        )

    e_corr = _pt2_correction(P, apply_HD)
    return NEVPT2Result(
        e_total=casci_result.e_total + e_corr,
        e_corr=e_corr,
        e_ref=casci_result.e_total,
    )


# ── CASPT2 ──────────────────────────────────────────────────────────────────


def _ipea_icb_specs(inactive, active, secondary):
    """Canonical eight-class IPEA contracted-basis generator labels.

    Each result is ``(case, p, q, r, s)`` for
    ``E_pq E_rs |Psi0>``.  Equal-type commuting pairs are emitted once;
    mixed-type pairs use one fixed orientation.  The active-active generators
    deliberately include ``E_tt``.  Omitting those diagonal generators and
    recovering their span through an overcomplete global singles+doubles
    basis changes the non-invariant IPEA correction.
    """
    specs = []

    def add_same_type(case, generators):
        for p, q in generators:
            for r, s in generators:
                if (r, s) < (p, q):
                    continue
                specs.append((case, p, q, r, s))

    si = [(a, i) for a in secondary for i in inactive]
    sa = [(a, t) for a in secondary for t in active]
    ai = [(t, i) for t in active for i in inactive]

    add_same_type("SS<-II", si)
    specs.extend(
        ("SS<-AI", a, i, b, t)
        for a in secondary
        for i in inactive
        for b in secondary
        for t in active
    )
    specs.extend(
        ("AS<-II", a, i, t, j)
        for a in secondary
        for i in inactive
        for t in active
        for j in inactive
    )
    add_same_type("SS<-AA", sa)
    # AS<-AI has two distinct active-index topologies.
    specs.extend(
        ("AS<-AI", a, i, t, u)
        for a in secondary
        for i in inactive
        for t in active
        for u in active
    )
    specs.extend(
        ("AS<-AI", t, i, a, u)
        for t in active
        for i in inactive
        for a in secondary
        for u in active
    )
    add_same_type("AA<-II", ai)
    specs.extend(
        ("AS<-AA", a, t, u, v)
        for a in secondary
        for t in active
        for u in active
        for v in active
    )
    specs.extend(
        ("AA<-AI", t, i, u, v)
        for t in active
        for i in inactive
        for u in active
        for v in active
    )
    return specs


def _ic_caspt2_solve(
    P,
    n_core,
    n_act,
    n_frozen=0,
    ipea=0.0,
    imaginary=0.0,
    thresh=1e-8,
    return_wavefunction=False,
):
    r"""Solve the IC-CASPT2 first-order equations for one prepared reference.

    ``P`` is a prepared-basis bundle (semicanonical integrals plus the
    reference state) with keys ``norb`` / ``ref`` / ``F`` / ``h1`` (one-
    electron, semicanonical) / ``eri`` (chemist, semicanonical) /
    ``dm1_diag``, as produced by :func:`_semicanonical_prep` (single state)
    or by the multi-state driver (:mod:`._ms_caspt2`, one bundle per model
    state).  See :func:`_ic_caspt2_corr` for the method equations.

    Returns
    -------
    (e_corr, dim_V_SD) : tuple[float, int]
        With ``return_wavefunction=False`` (default).
    (e_corr, dim_V_SD, psi1, W) : tuple[float, int, dict, dict]
        With ``return_wavefunction=True``.  ``psi1`` is the first-order
        wavefunction ``|Ψ⁽¹⁾> = S_mu C_mu |Φ_mu>`` in the determinant basis and
        ``W = H|0>``: the pieces the multi-state effective Hamiltonian
        needs (the <Φ_i|H|Ψ_j⁽¹⁾> couplings of Finley, Malmqvist, Roos &
        Serrano-Andrés, Chem. Phys. Lett. 288, 299 (1998)).
    """
    norb, ref, F, h1, eri = P["norb"], P["ref"], P["F"], P["h1"], P["eri"]
    dm1_diag = P["dm1_diag"]

    W = _add(apply_1body(ref, h1, norb), apply_2body(ref, eri, norb))  # H|0>
    E0 = _dot(ref, apply_1body(ref, F, norb))  # <0|F|0>

    inact = list(range(n_frozen, n_core))  # correlated inactive
    sec = list(range(n_core + n_act, norb))  # secondary (virtual)
    actv = list(range(n_core, n_core + n_act))
    occ = inact + actv  # annihilation sources
    virt = actv + sec  # creation targets
    exc = [(p, q) for p in virt for q in occ if p != q]

    def applyE(state, p, q):
        e = np.zeros((norb, norb))
        e[p, q] = 1.0
        return apply_1body(state, e, norb)

    singles = {(p, q): applyE(ref, p, q) for (p, q) in exc}
    cands, created, annih, candidate_cases = [], [], [], []
    if ipea != 0.0:
        # IPEA is representation-dependent in a nonorthogonal contracted
        # basis.  Use the canonical eight double-excitation classes, including
        # diagonal active generators E_tt, rather than reconstructing that
        # space from the globally overcomplete singles+doubles generator set.
        # This is the clean-room ICB convention of Nishimoto, JCP 158, 174112
        # (2023), DOI 10.1063/5.0147611, Eqs. 23-25 and 29-34.
        for case, p, q, r, s in _ipea_icb_specs(inact, actv, sec):
            right = singles.get((r, s))
            if right is None:  # diagonal active E_tt is absent from ``exc``
                right = applyE(ref, r, s)
            cands.append(applyE(right, p, q))  # Ê_pq Ê_rs |0>
            created.append((p, r))
            annih.append((q, s))
            candidate_cases.append(case)
    else:
        for p, q in exc:
            cands.append(singles[(p, q)])
            created.append((p,))
            annih.append((q,))
            candidate_cases.append(None)
        for p, q in exc:
            for r, s in exc:
                cands.append(applyE(singles[(r, s)], p, q))  # Ê_pq Ê_rs |0>
                created.append((p, r))
                annih.append((q, s))
                candidate_cases.append(None)

    # CAS space = (non-frozen) inactive doubly occupied AND secondary empty;
    # those determinants belong to |0> and are projected out of V_SD.
    def is_cas(m):
        for ii in inact:
            if not ((m >> _so(ii, 0, norb)) & 1) or not ((m >> _so(ii, 1, norb)) & 1):
                return False
        for aa in sec:
            if ((m >> _so(aa, 0, norb)) & 1) or ((m >> _so(aa, 1, norb)) & 1):
                return False
        return True

    # IPEA shift: per-orbital contribution to the H₀ DENOMINATOR.  Ghigo 2004
    # Eqs 10-11 give the Fock shifts s^(EA)_p = +1/2 D_pp e (excited into p) and
    # s^(IP)_r = -1/2(2-D_rr) e (excited out of r).  The hole r enters the energy
    # denominator with a minus sign, so its denominator contribution is
    # -s^(IP)_r = +1/2(2-D_rr) e.  Both are >= 0 (the shift always raises the
    # denominator) and only ACTIVE orbitals contribute (inactive D=2, virtual
    # D=0 give zero), using the active occupations in the pseudocanonical basis.
    a0, a1 = n_core, n_core + n_act

    def orb_shift(o, is_created):
        if a0 <= o < a1:
            d = dm1_diag[o - a0]
            return 0.5 * d * ipea if is_created else 0.5 * (2.0 - d) * ipea
        return 0.0

    # global determinant index + sparse candidate matrix M (n_dets x n_cand)
    det_idx: dict = {}
    rows, cols, vals, dshift = [], [], [], []
    kept = 0
    created_kept, annih_kept, cases_kept = [], [], []
    for c, cr, an, case in zip(cands, created, annih, candidate_cases):
        pc = {m: v for m, v in c.items() if abs(v) > 1e-14 and not is_cas(m)}
        if not pc:
            continue
        col = kept
        kept += 1
        created_kept.append(cr)
        annih_kept.append(an)
        cases_kept.append(case)
        for m, v in pc.items():
            j = det_idx.setdefault(m, len(det_idx))
            rows.append(j)
            cols.append(col)
            vals.append(v)
        dshift.append(
            sum(orb_shift(o, True) for o in cr) + sum(orb_shift(o, False) for o in an)
        )

    n_dets, n_cand = len(det_idx), kept
    if n_cand == 0:
        if return_wavefunction:
            return 0.0, 0, {}, W
        return 0.0, 0
    M = sp.csc_matrix((vals, (rows, cols)), shape=(n_dets, n_cand))
    # normalize the contracted functions to unit norm (clean overlap metric)
    cnorm = np.sqrt(np.asarray(M.multiply(M).sum(axis=0)).ravel())
    cnorm[cnorm < 1e-300] = 1.0
    M = M @ sp.diags(1.0 / cnorm)
    dshift = np.asarray(dshift)

    S = (M.T @ M).toarray()

    # Generalized Fock as a sparse operator on the in-space determinant block.
    # <b_k|F|b_l> needs F only between candidate determinants because the b's
    # span V_SD (so <b_k|F|b_l> = <b_k|P_SD F P_SD|b_l>).
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
    if ipea != 0.0:
        H0_raw += np.diag(dshift)  # IPEA: diagonal shift in the contracted basis
    Wvec = np.zeros(n_dets)
    for m, v in W.items():
        j = det_idx.get(m)
        if j is not None:
            Wvec[j] = v
    V_raw = M.T @ Wvec

    # Orthonormalize V_SD (drop linear dependencies below `thresh`).  IPEA
    # requires a class-local canonical Lowdin transformation: its diagonal
    # correction is not invariant to an arbitrary mixing of nonorthogonal
    # contracted functions.  The unshifted path retains its historical global
    # transformation because H0 itself is representation invariant there.
    if ipea != 0.0:
        transforms = []
        for case in dict.fromkeys(cases_kept):
            indices = np.flatnonzero(np.asarray(cases_kept) == case)
            w, U = eigh(S[np.ix_(indices, indices)])
            keep = w > thresh
            local = U[:, keep] / np.sqrt(w[keep])
            block = np.zeros((n_cand, int(keep.sum())))
            block[indices, :] = local
            transforms.append(block)
        T = np.concatenate(transforms, axis=1)
        nb = T.shape[1]
    else:
        w, U = eigh(S)
        keep = w > thresh
        T = U[:, keep] / np.sqrt(w[keep])
        nb = int(keep.sum())
    H0 = T.T @ H0_raw @ T
    V = T.T @ V_raw

    # Second-order energy in the H₀ eigenbasis.  With an imaginary shift s we
    # use the Hylleraas level-shift-CORRECTED energy (Forsberg & Malmqvist
    # 1997, Eq 11) -- the value OpenMolcas reports, 4th-order insensitive to s:
    #   E2 = -S_i |V_i|^2 Δ_i (Δ_i^2 + 2s^2) / (Δ_i^2 + s^2)^2 ,
    # which reduces to the unshifted -S_i |V_i|^2 / Δ_i at s = 0.
    d, Q = eigh(H0 - E0 * np.eye(nb))
    Vp = Q.T @ V
    if imaginary != 0.0:
        s2 = imaginary**2
        e_corr = float(-np.sum(Vp**2 * d * (d**2 + 2 * s2) / (d**2 + s2) ** 2))
    else:
        e_corr = float(-np.sum(Vp**2 / d))
    if not return_wavefunction:
        return e_corr, nb

    # First-order amplitudes in the H₀ eigenbasis.  Unshifted: C_i = -V_i/Δ_i.
    # With the imaginary shift, the real part of the complex-shifted solve
    # (Forsberg & Malmqvist, Chem. Phys. Lett. 274, 196 (1997), Eq 4):
    #   C_i = -V_i Δ_i / (Δ_i^2 + s^2)
    # i.e. the amplitudes OpenMolcas stores and contracts for the multi-state
    # couplings (the energy above stays the shift-corrected Hylleraas value).
    if imaginary != 0.0:
        c_orth = -Vp * d / (d**2 + imaginary**2)
    else:
        c_orth = -Vp / d
    c_raw = T @ (Q @ c_orth)
    psi1_vec = np.asarray(M @ c_raw).ravel()
    psi1 = {idx2det[j]: float(v) for j, v in enumerate(psi1_vec) if abs(v) > 1e-14}
    if return_wavefunction:
        clagdx = {
            "c_raw": c_raw,
            "V_raw": V_raw,
            "T": T,
            "Q": Q,
            "d": d,
            "E0": E0,
            "created_kept": created_kept,
            "annih_kept": annih_kept,
        }
        return e_corr, nb, psi1, W, clagdx


def _ic_caspt2_corr(
    h1e_mo,
    h2e_mo,
    n_core,
    n_act,
    n_act_elec,
    ms2,
    n_frozen=0,
    ipea=0.0,
    imaginary=0.0,
    thresh=1e-8,
    reference=None,
):
    r"""Internally-contracted CASPT2 second-order correlation energy.

    Builds the first-order interacting space ``V_SD`` explicitly in the
    determinant basis as the contracted double excitations

        |Ψ_pqrs> = Ê_pr Ê_qs |0>

    (Andersson, Malmqvist & Roos, J. Chem. Phys. 96, 1218 (1992), Eqs 1a-1h;
    Celani & Werner, J. Chem. Phys. 112, 5546 (2000)), projects them
    orthogonal to the CAS reference space, orthonormalizes through the overlap
    metric (dropping near-linear-dependent directions below ``thresh``), and
    solves the first-order equation

        (H₀ - E₀) C = -V,   H₀ = P_SD F P_SD,   E₀ = <0|F|0>,   V_i = <i|H|0>

    with the *full* (nondiagonal) generalized Fock ``F`` (the CASPT2N variant
    that is OpenMolcas's default ``&CASPT2``), giving ``E2 = V.C``.

    This is exact internal contraction built on the determinant-level
    spin-orbital engine; it reproduces OpenMolcas IC-CASPT2 to <=2 µHa across
    virtual-, core-hole-, combined- and large-active-space regimes.  The
    contracted space is built densely, so (like the rest of
    :mod:`vibeqc.solvers`) it targets the small-active-space regime.

    Parameters
    ----------
    n_frozen : int
        Number of deepest inactive orbitals kept frozen (not correlated).
    ipea : float
        IPEA level shift e (Ghigo, Roos & Malmqvist, Chem. Phys. Lett. 396,
        142 (2004)); nonzero values use the canonical eight-class internally
        contracted basis described in :func:`caspt2`.
    imaginary : float
        Imaginary level shift s for intruder-state removal (Forsberg &
        Malmqvist, Chem. Phys. Lett. 274, 196 (1997)).

    Returns
    -------
    (e_corr, dim_V_SD) : tuple[float, int]
    """
    P = _semicanonical_prep(
        h1e_mo,
        h2e_mo,
        n_core,
        n_act,
        n_act_elec,
        ms2,
        pseudocanonical_active=(ipea != 0.0),
        reference=reference,
    )
    return _ic_caspt2_solve(
        P,
        n_core,
        n_act,
        n_frozen=n_frozen,
        ipea=ipea,
        imaginary=imaginary,
        thresh=thresh,
    )


_IC_CASPT2_DIRECT_LABEL_THRESHOLD = 5000


def _ic_caspt2_label_count(
    norb: int,
    n_core: int,
    n_act: int,
    n_frozen: int = 0,
) -> int:
    active = list(range(n_core, n_core + n_act))
    inactive = list(range(n_frozen, n_core))
    secondary = list(range(n_core + n_act, norb))
    occ = inactive + active
    virt = active + secondary
    n_exc = sum(1 for p in virt for q in occ if p != q)
    return n_exc + n_exc * n_exc


def _ic_caspt2_uses_direct_active_ci(
    norb: int,
    n_core: int,
    n_act: int,
    n_frozen: int = 0,
    ipea: float = 0.0,
    imaginary: float = 0.0,
) -> bool:
    # The IPEA shift stays on the explicit engine because its canonical
    # eight-class contraction is not implemented by the direct RDM path.
    # The imaginary shift is supported on the direct path since 2026-06-10
    # via the level-shift-corrected SPD solve
    # (solve_signature_block_linear_system_imaginary), so it no longer
    # forces the explicit engine.
    del imaginary
    if ipea != 0.0:
        return False
    return (
        _ic_caspt2_label_count(norb, n_core, n_act, n_frozen=n_frozen)
        > _IC_CASPT2_DIRECT_LABEL_THRESHOLD
    )


def caspt2(
    casci_result: CASCIResult,
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_core: int = 0,
    n_virt: int = 0,
    shift: float = 0.0,
    variant: str = "ic",
    ipea: float = 0.0,
    imaginary: float = 0.0,
    n_frozen: int = 0,
    selected_reference: bool = False,
    engine: str = "auto",
) -> NEVPT2Result:
    r"""CASPT2 on top of a CASCI reference (internally- or strongly-contracted).

    ``variant="ic"`` (default) -- **internally-contracted CASPT2** (Andersson,
    Malmqvist & Roos, J. Chem. Phys. 96, 1218 (1992)), the production method.
    The first-order interacting space is the full set of contracted double
    excitations ``Ê_pr Ê_qs|0>`` and H₀ is the nondiagonal generalized Fock
    (CASPT2N).  Validated against OpenMolcas ``&CASPT2`` to <=2 µHa across
    virtual-, core-hole-, combined- and large-active-space regimes.  Supports a
    frozen core (``n_frozen``) and an imaginary level shift (``imaginary`` s;
    Forsberg & Malmqvist, Chem. Phys. Lett. 274, 196 (1997)) for intruders.

    Imaginary-shift representation caveat (characterized 2026-06-10): vibe-qc
    applies the complex shift to the **full nondiagonal H₀** (the
    basis-independent reading of Forsberg-Malmqvist, implemented identically
    on the explicit and direct paths), whereas OpenMolcas adds ``s^2/Δ`` to
    the diagonal of its case-block-diagonal H₀ representation (``sgmdia``).
    The conventions agree to <1e-8 Ha in intruder-free systems (H2O/cc-pVDZ
    CAS(4,4), s=0.1) but differ at the 1e-5-1e-4 Ha level near the strong
    intruders the shift exists to regularize (e.g. stretched LiH excited
    roots).  s=0 energies carry no such ambiguity.

    ``ipea`` (Ghigo, Roos & Malmqvist, Chem. Phys. Lett. 396, 142 (2004))
    applies the IPEA diagonal shift.  Nonzero values use the canonical eight
    internally contracted double-excitation classes and class-local canonical
    Lowdin orthogonalization (Nishimoto, J. Chem. Phys. 158, 174112 (2023)).
    This fixes the otherwise representation-dependent correction and matches
    OpenMolcas for the supported single-state explicit-engine route.  The
    default remains ``0.0``; IPEA is a semi-empirical correction.

    ``variant="sc"`` -- the older **strongly-contracted** CASPT2 (one perturber
    per external pattern; generalized-Fock H₀, Andersson, Malmqvist, Roos,
    Sadlej & Wolinski, J. Phys. Chem. 94, 5483 (1990)).  Its MP2 and
    full-space limits are exact and it cross-checks against NEVPT2, but its
    active-space terms have no external reference, so it stays gated behind
    ``VIBEQC_EXPERIMENTAL_MULTIREF=1``.  ``shift`` is a plain real level shift
    accepted for this variant only.

    ``selected_reference`` (``variant="ic"`` only) builds the first-order
    interacting space on ``casci_result``'s own (possibly truncated)
    selected-CI wavefunction instead of re-solving the full CAS (the
    selected-CI -> CASPT2 path, roadmap 25i).  The contracted space
    ``Ê_pr Ê_qs|0>`` and the projection out of the CAS reference space are
    evaluated in determinant space (no high-order RDMs), so the reference may
    be a truncated wavefunction whose full CAS is intractable; the result is
    the CASPT2 of *that* reference and converges to full-CAS CASPT2 as the
    selection tightens (identical at the full-selection limit, hence to
    OpenMolcas ``&CASPT2`` there).  Forces the explicit determinant engine
    (the direct active-CI signature path is built on the full CAS).  Not
    combinable with ``ipea!=0`` (the active rotation it needs would change the
    supplied CI vector).

    ``engine`` (``variant="ic"``) selects the solver backend.  ``"auto"``
    (default) keeps the validated explicit/direct dispatch.  ``"cases"`` is the
    case-partitioned, matrix-free RDM CASPT2 (roadmap 25i): it assembles the
    FOIS overlap / RHS / generalized-Fock H₀ as collapsed per-case
    externalxactive-RDM "sigma" contractions (no determinant FOIS, the 4-RDM
    taken via F3) and solves ``(H₀-E₀S)x=-V`` iteratively, so the cost is set
    by the active RDMs and not the reference determinant count.  It reproduces
    the explicit/direct engines (hence OpenMolcas) at s=0 and composes with a
    selected reference; it is unshifted, ``ipea=0`` only.  The overlap, RHS and
    every part of H₀ (including the off-diagonal generalized-Fock coupling, by
    external-offset grouping) are collapsed to block-structured sigmas, so the
    matrix-free matvec has **no** ``n_cand^2`` term for any reference.  The
    active-RDM contractions inside those sigmas are still evaluated through the
    Python reducer, so the present cross-over vs the determinant engines is
    governed by that constant factor (a C++ inner kernel is the next step,
    roadmap 25i M27c).

    See ``handovers/HANDOVER_MULTIREF.md``.
    """
    norb = h1e_mo.shape[0]
    n_act = norb - n_core - n_virt
    a_occ0, b_occ0 = casci_result.determinants[0]
    n_act_elec = len(a_occ0) + len(b_occ0)
    ms2 = len(a_occ0) - len(b_occ0)

    if selected_reference and variant != "ic":
        raise ValueError("selected_reference is only supported for variant='ic'")
    if selected_reference and ipea != 0.0:
        raise ValueError(
            "selected_reference is not combinable with the IPEA shift "
            "(ipea!=0): IPEA needs a pseudocanonical active rotation that "
            "would change the supplied selected CI vector"
        )

    if engine not in ("auto", "cases"):
        raise ValueError(f"caspt2 engine must be 'auto' or 'cases', got {engine!r}")

    if variant == "ic" and engine == "cases":
        # Case-partitioned, matrix-free RDM CASPT2 (roadmap 25i): assembles the
        # FOIS overlap/RHS/H₀ as collapsed per-case externalxactive-RDM sigmas
        # (no determinant FOIS, no 4-RDM via F3) and solves iteratively, so the
        # cost is set by the active RDMs not the reference determinant count.
        # Reproduces the explicit/direct engines (and OpenMolcas) at s=0.
        # Unshifted, ipea=0 only; composes with a selected reference.
        if imaginary != 0.0 or ipea != 0.0:
            raise ValueError(
                "engine='cases' supports only the unshifted, ipea=0 IC-CASPT2 "
                "(imaginary shift / IPEA use engine='auto')"
            )
        from ._caspt2_cases import caspt2_corr_rdm_matfree

        P = _semicanonical_prep(
            h1e_mo,
            h2e_mo,
            n_core,
            n_act,
            n_act_elec,
            ms2,
            reference=(casci_result.ci_coeffs, casci_result.determinants),
        )
        e_corr, _ = caspt2_corr_rdm_matfree(
            P,
            n_core,
            n_act,
            casci_result.ci_coeffs,
            casci_result.determinants,
            n_frozen=n_frozen,
        )
        return NEVPT2Result(
            e_total=casci_result.e_total + e_corr,
            e_corr=e_corr,
            e_ref=casci_result.e_total,
        )

    if variant == "ic":
        _reference = (
            (casci_result.ci_coeffs, casci_result.determinants)
            if selected_reference
            else None
        )
        # A selected reference forces the explicit determinant engine: the
        # direct active-CI signature path is built on the full CAS.
        if not selected_reference and _ic_caspt2_uses_direct_active_ci(
            norb,
            n_core,
            n_act,
            n_frozen=n_frozen,
            ipea=ipea,
            imaginary=imaginary,
        ):
            from ._caspt2_rdm import active_ci_caspt2_corr_signature_iterative

            direct = active_ci_caspt2_corr_signature_iterative(
                h1e_mo,
                h2e_mo,
                n_core,
                n_act,
                n_act_elec,
                ms2,
                n_frozen=n_frozen,
                tol=1e-10,
                imaginary=imaginary,
            )
            if not direct.converged:
                message = (
                    "direct active-CI CASPT2 response did not converge "
                    f"(residual {direct.residual_norm:.3e})"
                )
                raise RuntimeError(message)
            e_corr = direct.energy
        else:
            e_corr, _ = _ic_caspt2_corr(
                h1e_mo,
                h2e_mo,
                n_core,
                n_act,
                n_act_elec,
                ms2,
                n_frozen=n_frozen,
                ipea=ipea,
                imaginary=imaginary,
                reference=_reference,
            )
        return NEVPT2Result(
            e_total=casci_result.e_total + e_corr,
            e_corr=e_corr,
            e_ref=casci_result.e_total,
        )

    if variant != "sc":
        raise ValueError(f"caspt2 variant must be 'ic' or 'sc', got {variant!r}.")

    # ── strongly-contracted variant (gated, no external active-space reference)
    require_experimental_multiref("CASPT2 (strongly-contracted)")
    P = _semicanonical_prep(h1e_mo, h2e_mo, n_core, n_act, n_act_elec, ms2)
    F = P["F"]
    nb = P["norb"]

    def apply_H0(state):
        return apply_1body(state, F, nb)

    if shift == 0.0:
        e_corr = _pt2_correction(P, apply_H0)
    else:
        # Plain (uncorrected) real level shift on the denominators:
        # ΔE -> ΔE + shift.  This is NOT the Roos-Andersson 1995 shift with
        # its back-correction; for proper intruder removal use the IC variant's
        # imaginary shift (Forsberg-Malmqvist 1997).  Validated path: shift=0.0.
        e0 = _dot(P["ref"], apply_H0(P["ref"]))
        e_corr = 0.0
        for V in P["groups"].values():
            n = _dot(V, V)
            if n < 1e-14:
                continue
            dE = _dot(V, apply_H0(V)) / n - e0
            if abs(dE + shift) < 1e-12:
                continue
            e_corr -= n / (dE + shift)
    return NEVPT2Result(
        e_total=casci_result.e_total + e_corr,
        e_corr=e_corr,
        e_ref=casci_result.e_total,
    )
