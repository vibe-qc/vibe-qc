"""Complete active space self-consistent field (CASSCF).

CASSCF optimizes the active-space CI vector **and** the molecular orbitals
together, minimizing the CAS(n,m) energy with respect to orbital rotations.
It removes CASCI's dependence on the (frozen) HF orbitals: at convergence the
orbital gradient vanishes, so the energy is stationary under all
inactive<->active, inactive<->virtual and active<->virtual rotations.  CASSCF is the
standard reference for the multireference perturbation theories built on top
of it (NEVPT2 / CASPT2 then run on the CASSCF reference rather than on HF
orbitals).

State-averaged CASSCF
---------------------
When ``nroots > 1`` and ``weights`` is supplied, the energy being minimized is
the weighted average over the selected CI roots:

    E_SA = S_k w_k E_k

and the orbital gradient is built from the state-averaged 1- and 2-RDMs

    g_SA = S_k w_k g_k,    Γ_SA = S_k w_k Γ_k

(the generalized Fock and gradient are then exactly the same formulas as the
single-state case, just with the SA-RDMs).  State-averaging ensures the
orbitals describe all states in the average fairly -- essential for excited-
state calculations and for avoiding root-flipping in near-degeneracy regions.

Orbital update methods
----------------------
Two orbital-step methods are available:

* **Super-CI** (``orbital_step="superci"``) -- uses a diagonal Hessian
  approximation from the generalized Fock matrix:
  ``H_diag[pq] = F_pp - F_qq``.  No finite-difference Hessian needed, so each
  macro-iteration costs one CASCI solve.  Robust far from convergence.

* **Newton-CG** (``orbital_step="nr"``) -- trust-region (Steihaug-Toint)
  truncated CG on the orbital Hessian, with Hessian-vector products by
  central finite differences of the *analytic* gradient: 2 CASCI solves per
  CG iteration, independent of the rotation-pair count (the historical
  per-pair FD Hessian cost ``2.n_pairs`` CASCI solves per iteration and was
  the large-CAS wall).  Quadratically convergent near the minimum.

The default ``orbital_step="auto"`` starts with Super-CI and switches to
Newton-CG when the gradient norm drops below ``switch_to_nr``.

Algorithm
---------
A two-step (uncoupled) macro-iteration:

1. Solve CASCI in the current orbital basis (:func:`vibeqc.solvers.casci`),
   giving the active CI vector(s), energy and active 1-/2-RDMs.
2. Build the non-symmetric generalized Fock matrix ``F`` and the orbital
   gradient ``g_pq = 2(F_qp - F_pq)`` over the non-redundant rotation pairs.
   Because the CI vector is re-optimized in every basis, the CASSCF energy is
   stationary in the CI parameters, so this orbital gradient is *exact*: no
   orbital-CI coupling term is needed.
3. Take an orbital update step (Super-CI or NR), backtracked so the energy
   can never rise (guaranteed monotone descent).  Exponentiate the
   antisymmetric rotation, rotate the MO integrals, and repeat until ``‖g‖``
   is below threshold.

This is a small-active-space implementation consistent with the rest of
:mod:`vibeqc.solvers` (the underlying CI is the determinant engine).

Like every CASSCF, the energy surface is non-convex: the optimizer converges to
the stationary point in the basin of the starting orbitals, which need not be
the global minimum.  Validated against PySCF ``mcscf.CASSCF`` to <=1e-7 Ha on
unique-minimum systems (H2, LiH, HF, Be) and SA-CASSCF to <=1e-6 Ha.

References
----------
B. O. Roos, P. R. Taylor, P. E. M. Siegbahn, Chem. Phys. 48, 157 (1980).
P. E. M. Siegbahn, J. Almlöf, A. Heiberg, B. O. Roos,
J. Chem. Phys. 74, 2384 (1981).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy.linalg import expm

from vibeqc.output import write

from ._casci import CASCIResult, casci
from ._mrpt import _generalized_fock
from ._rdm import make_rdm12, make_rdm12_sa

OrbitalStep = Literal["auto", "superci", "nr"]


@dataclass
class CASSCFOptions:
    """Options for run_job(method='casscf', casscf_options=...).

    Attributes
    ----------
    nroots : int
        Number of CI roots to include in state-averaging (default 1).
    weights : list[float] or None
        State-averaging weights (must sum to 1, length nroots).
    orbital_step : str
        'auto' (default), 'superci', or 'nr'.
    spin_pure : bool or None
        ``True``: state-average over the lowest ``nroots`` *spin-pure*
        CI roots (<S^2> = S(S+1) with S = ms2/2), skipping the higher-spin
        roots that live in the same M_s determinant sector, the averaging
        a CSF-based code (OpenMolcas, MOLPRO) performs.  ``False``: average
        the raw lowest M_s-sector roots (the vibe-qc default before
        2026-06-11; deliberate mixed-spin ensemble averaging).
        ``None`` (default) resolves to ``True`` for state-averaged runs
        (``nroots > 1``; SA users mean same-spin roots, and the M_s
        sector interleaves higher-spin states on essentially every small
        CAS) and to ``False`` for single-state runs, which keep the raw
        lowest root of the sector (no root buffering on the direct path;
        the .out root table prints per-root <S^2> either way).
    ci_solver : str
        ``"casci"`` (default): exact CI in the full active determinant
        space (dense or direct-Davidson backend).  ``"selected_ci"``:
        CIPSI selected CI (:func:`vibeqc.solvers.selected_casci`), for
        active spaces beyond the determinant wall; the selected space is
        warm-started across macro-iterations.  See ``selected_ci_options``.
    selected_ci_options : SelectedCIOptions or None
        Selection controls for ``ci_solver="selected_ci"`` (target_size,
        pt2_threshold, ...).  ``None`` uses the CASSCF-tuned defaults of
        :func:`vibeqc.solvers.casscf`.
    pt2 : SelectedCIPT2Options or None
        ``ci_solver="selected_ci"`` only: after convergence, compute the
        per-root Epstein-Nesbet PT2 correction on the final selected
        wavefunction (:func:`vibeqc.solvers.selected_ci_pt2`; the SHCI
        perturbative stage).  Surfaces on ``SolverResult.selected_pt2``
        and as a dedicated .out block; the headline ``energy`` stays the
        variational CASSCF value.  ``None`` (default) skips PT2.
    compute_wz : bool or str
        Gradient path selector, kept for backward compatibility:
        - ``False`` (default): the analytic CASSCF gradient. It is the
          complete derivative of the variational CASSCF energy (matches
          full-energy finite differences to ~2e-7 Ha/bohr).
        - ``True``: accepted no-op alias of the analytic gradient with a
          :class:`FutureWarning`. The former experimental "W^z"
          CP-MCSCF correction it selected was a phantom term for a
          variational energy and produced spurious gradient components
          (GitLab #516); it has been removed.
        - ``"numerical"``: central finite difference of the re-converged
          CASSCF energy -- an expensive cross-check oracle.
    """

    nroots: int = 1
    weights: list[float] | None = None
    orbital_step: OrbitalStep = "auto"
    active_orbitals: list[int] | None = None
    spin_pure: bool | None = None
    ci_solver: str = "casci"
    selected_ci_options: object | None = None
    pt2: object | None = None
    compute_wz: bool | str = False


@dataclass
class CASSCFResult:
    """CASSCF result.

    Attributes
    ----------
    e_total : float
        Converged CASSCF total energy (incl. nuclear repulsion + core).
        For state-averaged CASSCF, this is the weighted average over states.
    e_corr : float
        ``e_total - e_hf`` (CASSCF correlation energy relative to the
        reference HF energy, if one was supplied).
    cas : CASCIResult
        The CASCI solution in the *converged* (optimized) orbital basis.
    h1e_cas : np.ndarray
        One-electron MO integrals in the converged CASSCF orbital basis.
    h2e_cas : np.ndarray
        Two-electron MO integrals (physicist's ``g``) in the converged basis.
    mo_rotation : np.ndarray
        Cumulative orbital rotation mapping input to CASSCF basis.
    converged : bool
        Whether the orbital gradient fell below ``conv_tol_grad``.
    n_iter : int
        Number of macro-iterations taken.
    grad_norm : float
        Final orbital-gradient norm.
    energy_trace : list[float]
        Energy at each macro-iteration (weighted-average for SA).
    e_totals : list[float]
        Per-root energies in the converged basis (length ``nroots``).
        Only populated for state-averaged runs.
    """

    e_total: float
    e_corr: float
    cas: CASCIResult
    h1e_cas: np.ndarray
    h2e_cas: np.ndarray
    mo_rotation: np.ndarray
    converged: bool
    n_iter: int
    grad_norm: float
    energy_trace: list = field(default_factory=list)
    e_totals: list = field(default_factory=list)


def _nonredundant_pairs(n_core: int, n_act: int, norb: int) -> list[tuple[int, int]]:
    """Non-redundant orbital-rotation pairs (p, q) with p above q."""
    inact = range(0, n_core)
    act = range(n_core, n_core + n_act)
    virt = range(n_core + n_act, norb)
    pairs: list[tuple[int, int]] = []
    for i in inact:
        for t in act:
            pairs.append((t, i))
        for a in virt:
            pairs.append((a, i))
    for t in act:
        for a in virt:
            pairs.append((a, t))
    return pairs


def _build_kappa(x: np.ndarray, pairs: list[tuple[int, int]], norb: int) -> np.ndarray:
    """Antisymmetric rotation generator kappa from packed parameters."""
    K = np.zeros((norb, norb))
    for (p, q), v in zip(pairs, x):
        K[p, q] = v
        K[q, p] = -v
    return K


def _orbital_gradient_and_fock(
    h1: np.ndarray,
    eri_chem: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    n_core: int,
    n_act: int,
    norb: int,
    pairs: list[tuple[int, int]],
):
    """Orbital gradient g_pq = 2(F_qp - F_pq), generalized Fock F, and
    diagonal Hessian elements H_diag[pq] = Favg_pp - Favg_qq.

    Returns (grad, F, Favg, H_diag).
    """
    # Use the fused C++ kernel when available.
    try:
        from .._vibeqc_core import casscf_orbital_gradient as _casscf_grad
    except ImportError:
        pass
    else:
        # C++ kernel expects flat-packed arrays.
        eri_flat = np.ascontiguousarray(eri_chem.ravel(), dtype=float)
        dm2_flat = np.ascontiguousarray(dm2.ravel(), dtype=float)
        dm1_arr = np.ascontiguousarray(dm1, dtype=float)
        result = _casscf_grad(
            np.ascontiguousarray(h1, dtype=float),
            eri_flat,
            dm1_arr,
            dm2_flat,
            n_core,
            n_act,
            norb,
            pairs,
        )
        return result[0], np.asarray(result[1]), np.asarray(result[2]), result[3]

    # Python fallback.
    A = slice(n_core, n_core + n_act)

    FI = h1.copy()
    for i in range(n_core):
        FI += 2.0 * eri_chem[:, :, i, i] - eri_chem[:, i, i, :]

    Favg = _generalized_fock(h1, eri_chem, dm1, n_core, n_act)

    F = np.zeros((norb, norb))
    if n_core:
        F[:n_core, :] = 2.0 * Favg[:n_core, :]
    if n_act:
        F1 = np.einsum("tu,qu->tq", dm1, FI[:, A], optimize=True)
        F2 = np.einsum("tuvw,quvw->tq", dm2, eri_chem[:, A, A, A], optimize=True)
        F[A, :] = F1 + F2

    grad = np.array([2.0 * (F[q, p] - F[p, q]) for (p, q) in pairs])

    # Super-CI diagonal Hessian from the symmetric Fock: H_diag[pq] = Favg_pp - Favg_qq.
    diag_f = np.diag(Favg)
    H_diag = np.array([diag_f[p] - diag_f[q] for (p, q) in pairs])

    return grad, F, Favg, H_diag


def _superci_step(grad: np.ndarray, H_diag: np.ndarray, trust: float) -> np.ndarray:
    """Super-CI (diagonal-Hessian) orbital step.

    x_pq = -g_pq / max(|H_diag[pq]|, reg) with adaptive regularisation that
    decreases as the gradient shrinks, allowing convergence to tight
    thresholds.  The absolute value matters on stiff surfaces (large active
    spaces): a negative diagonal element would flip that component of the
    step uphill, which the backtracking line search then rejects -- the
    2026-06-10 CAS(10,10) stall.  Clamping to |H_diag| keeps every
    component a descent direction of the local quadratic model.
    """
    g_norm = float(np.linalg.norm(grad))
    reg = max(1e-4, 0.1 * min(1.0, g_norm))
    denom = np.maximum(np.abs(H_diag), reg)
    step = -grad / denom
    norm = float(np.linalg.norm(step))
    if norm > trust:
        step = step * (trust / norm)
    return step


def _rotate_integrals(h1: np.ndarray, g_phys: np.ndarray, U: np.ndarray):
    """Rotate MO integrals by the unitary U (columns = new orbitals)."""
    h1_new = U.T @ h1 @ U
    g_new = np.einsum("ap,bq,cr,ds,abcd->pqrs", U, U, U, U, g_phys, optimize=True)
    return h1_new, g_new


def _to_trust_boundary(x: np.ndarray, p: np.ndarray, trust: float) -> np.ndarray:
    """Walk from ``x`` along ``p`` to the trust-region boundary ‖x+tp‖=trust."""
    px = float(p @ x)
    pp = float(p @ p)
    xx = float(x @ x)
    tau = (-px + np.sqrt(max(px * px + pp * (trust * trust - xx), 0.0))) / max(
        pp, 1e-300
    )
    return x + tau * p


def _newton_cg_step(
    grad: np.ndarray,
    H_diag: np.ndarray,
    hvp,
    trust: float,
    cg_tol: float = 0.1,
    max_cg: int = 20,
) -> np.ndarray:
    """Trust-region (Steihaug-Toint) truncated-CG Newton step.

    Solves ``H x = -g`` approximately with ``hvp(v) = H.v`` supplied as a
    callback (here: central finite differences of the *analytic* orbital
    gradient -- 2 CASCI solves per CG iteration, independent of the number
    of rotation pairs), preconditioned by ``max(|H_diag|, 1e-3)``.  Negative
    curvature or leaving the trust region walks to the boundary (standard
    Steihaug behaviour, Nocedal & Wright, Numerical Optimization, Alg. 7.2).
    """
    x = np.zeros_like(grad)
    r = grad.copy()  # residual of H x + g
    M = np.maximum(np.abs(H_diag), 1e-3)
    z = r / M
    p = -z
    rz = float(r @ z)
    g_norm = float(np.linalg.norm(grad))
    for _ in range(max_cg):
        Hp = hvp(p)
        pHp = float(p @ Hp)
        if pHp <= 1e-14 * float(p @ p):
            return _to_trust_boundary(x, p, trust)
        alpha = rz / pHp
        x_new = x + alpha * p
        if float(np.linalg.norm(x_new)) >= trust:
            return _to_trust_boundary(x, p, trust)
        x = x_new
        r = r + alpha * Hp
        if float(np.linalg.norm(r)) < cg_tol * g_norm:
            break
        z = r / M
        rz_new = float(r @ z)
        beta = rz_new / rz
        p = -z + beta * p
        rz = rz_new
    if not np.any(x):
        # First CG step already left the region / hit negative curvature
        # before accumulating x -- fall back to the preconditioned gradient.
        x = -grad / M
        n = float(np.linalg.norm(x))
        if n > trust:
            x *= trust / n
    return x


def casscf(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_active_elec: int,
    n_active_orb: int,
    n_core: int = 0,
    nuclear_repulsion: float = 0.0,
    e_hf: float = 0.0,
    ms2: Optional[int] = None,
    *,
    nroots: int = 1,
    weights: Optional[list[float]] = None,
    orbital_step: OrbitalStep = "auto",
    active_orbitals: Optional[list[int]] = None,
    spin_pure: Optional[bool] = None,
    ci_solver: str = "casci",
    selected_ci_options=None,
    switch_to_nr: float = 0.01,
    stall_to_nr: float = 1e-4,
    max_macro: int = 100,
    conv_tol_grad: float = 1e-6,
    trust: float = 0.3,
    verbose: int = 0,
) -> CASSCFResult:
    """Run CASSCF in an active space, starting from the supplied MO integrals.

    Parameters
    ----------
    h1e_mo : (norb, norb) ndarray
        One-electron Hamiltonian in the (starting, e.g. HF) MO basis.
    h2e_mo : (norb, norb, norb, norb) ndarray
        Two-electron integrals in physicist's notation, MO basis.
    n_active_elec, n_active_orb : int
        CAS(n, m): active electrons and active spatial orbitals.
    n_core : int
        Number of doubly-occupied inactive (core) orbitals.
    nuclear_repulsion : float
        Nuclear repulsion energy.
    e_hf : float
        Reference HF energy, used only to report ``e_corr``.
    ms2 : int, optional
        ``2 S_z`` of the active electrons (defaults to ``n_active_elec % 2``).
    nroots : int
        Number of CI roots for state-averaging (default 1).
    weights : list[float], optional
        State-averaging weights (must sum to 1, length ``nroots``).
    spin_pure : bool or None
        Average over the lowest ``nroots`` spin-pure roots
        (<S^2>-filtered, S = ms2/2) instead of the raw lowest M_s-sector
        roots; see :class:`CASSCFOptions.spin_pure`.  ``None`` (default)
        resolves to ``True`` for state-averaged runs (``nroots > 1``)
        and ``False`` for single-state runs.  The filter is re-applied
        at every macro-iteration, which also protects the averaged set
        against spin-sector root flipping along the optimization.
    ci_solver : str
        ``"casci"`` (default): exact CI in the full active determinant
        space.  ``"selected_ci"``: CIPSI selected CI
        (:func:`vibeqc.solvers.selected_casci`) inside every
        macro-iteration: the active-space route past the determinant
        wall.  The selected determinant set is warm-started across
        macro-iterations (the previous set is a good variational space
        under a small orbital rotation), and near convergence the
        selection stops changing, so the orbital gradient (exact for the
        energy at fixed selection, since the truncated-list RDMs are
        exact) drives a smooth approach to stationarity.  The converged
        energy is variational and approaches the exact CASSCF from above
        as ``selected_ci_options.target_size`` / ``pt2_threshold`` are
        tightened.  Combines with ``spin_pure``: each macro-iteration
        solves a buffered root set in the selected space and keeps the
        lowest ``nroots`` <S^2>-pure roots
        (:func:`._selected_ci._spin_pure_selected_roots`).
    selected_ci_options : SelectedCIOptions, optional
        Controls for ``ci_solver="selected_ci"``.  Default:
        ``SelectedCIOptions(target_size=20000, conv_tol_energy=1e-10,
        pt2_threshold=1e-9, max_det_per_iter=10000,
        significant_coeff=0.005)``.
    orbital_step : {'auto', 'superci', 'nr'}
        Orbital update method.  ``"auto"`` starts with Super-CI and switches
        to Newton-CG when the gradient norm drops below ``switch_to_nr``.
        ``"superci"`` uses a diagonal Fock-based Hessian (one CASCI solve per
        iteration).  ``"nr"`` uses trust-region truncated-CG Newton with
        Hessian-vector products by central FD of the analytic gradient
        (2 CASCI solves per CG iteration, pair-count independent;
        quadratically convergent near the minimum).
    switch_to_nr : float
        Gradient-norm threshold for switching from Super-CI to NR in
        ``orbital_step="auto"``.
    stall_to_nr : float
        Second ``"auto"`` trigger: hand over to NR when Super-CI's
        average energy progress over the last 3 macro-iterations falls
        below this (Ha/iteration), even though |g| is still above
        ``switch_to_nr``.  Super-CI's diagonal-Hessian step can plateau
        with a large gradient on stiff large-CAS surfaces (the selected
        backend's CAS(14,14) tail sat at |g| ~ 2-6e-2, gaining ~5 µHa
        per iteration); a stall above the gradient threshold is exactly
        the regime where the second-order step is needed.  Near
        convergence the same trigger just starts the NR endgame early.
        0 disables (historical gradient-only switching).
    max_macro : int
        Maximum number of orbital-optimization macro-iterations.
    conv_tol_grad : float
        Convergence threshold on the orbital-gradient norm.
    trust : float
        Trust radius (rad) capping each orbital-rotation step.
    verbose : int
        ``>0`` emits per-iteration energy and gradient norm through the
        active :mod:`vibeqc.output` channel.

    Returns
    -------
    CASSCFResult
    """
    norb = h1e_mo.shape[0]
    n_act = n_active_orb

    # Resolve n_core/n_act from active_orbitals if provided.
    if active_orbitals is not None:
        act_idx = sorted(active_orbitals)
        n_core = act_idx[0]
        n_act = len(act_idx)
    if n_core + n_act > norb:
        raise ValueError(f"n_core ({n_core}) + n_active_orb ({n_act}) > norb ({norb})")
    if ms2 is None:
        ms2 = n_active_elec % 2
    if spin_pure is None:
        # The 2026-06-11 default flip (maintainer-approved): state
        # averaging targets same-spin roots, the averaging a CSF-based
        # code performs by construction.  Single-state runs keep the raw
        # lowest M_s-sector root (no root buffering on the direct path).
        spin_pure = nroots > 1
    if orbital_step not in ("auto", "superci", "nr"):
        raise ValueError(
            f"orbital_step must be 'auto', 'superci', or 'nr', got {orbital_step!r}"
        )
    if ci_solver not in ("casci", "selected_ci"):
        raise ValueError(
            f"ci_solver must be 'casci' or 'selected_ci', got {ci_solver!r}"
        )
    if ci_solver == "selected_ci":
        from ._selected_ci import SelectedCIOptions as _SciOpts

        sci_opts = selected_ci_options or _SciOpts(
            target_size=20000,
            conv_tol_energy=1e-10,
            pt2_threshold=1e-9,
            max_det_per_iter=10000,
            significant_coeff=0.005,
        )

    sa_weights: Optional[list[float]] = None
    if nroots > 1:
        if weights is None:
            sa_weights = [1.0 / nroots] * nroots
        else:
            if len(weights) != nroots:
                raise ValueError(f"len(weights)={len(weights)} != nroots={nroots}")
            if abs(sum(weights) - 1.0) > 1e-12:
                raise ValueError(f"weights must sum to 1, got sum={sum(weights)}")
            sa_weights = list(weights)

    pairs = _nonredundant_pairs(n_core, n_act, norb)
    npr = len(pairs)
    h1 = np.array(h1e_mo, dtype=float)
    g_phys = np.array(h2e_mo, dtype=float)
    U_total = np.eye(norb)
    fd_eps = 1e-4

    # Warm start: each CASCI solve seeds the direct (Davidson) backend with
    # the most recent CI vector(s) -- orbital rotations between evaluations
    # are small, so the previous vector is an excellent starting subspace.
    # The dense backend ignores the guess (exact eigh).  The selected-CI
    # backend warm-starts from the previous selected determinant SET
    # instead ("dets").
    _ci_guess: dict = {"v": None, "dets": None}

    def _solve_casci(h1_, g_, grow=True):
        """One CI solve (chosen backend) in the basis (h1_, g_).

        ``grow=False`` (selected-CI backend only) freezes the selected
        determinant set: solve exactly in the warm-start space without
        selection, and do not update the stored set.  The finite-difference
        Hessian probes need this: the Newton model must differentiate one
        fixed-selection energy surface (the surface the analytic gradient
        is exact for), not a surface that re-selects per probe.
        """
        if ci_solver == "selected_ci":
            from dataclasses import replace as _dc_replace

            from ._selected_ci import selected_casci

            solve_opts = sci_opts if grow else _dc_replace(sci_opts, max_iter=1)
            if spin_pure:
                # Selected-space analogue of the dense spin-pure branch
                # below: buffered solve + <S^2> root filter (see
                # _selected_ci._spin_pure_selected_roots for the
                # truncation/classification notes).
                from ._selected_ci import _spin_pure_selected_roots

                ci_cols, e_tots, _s2s, res_buf = _spin_pure_selected_roots(
                    h1_,
                    g_,
                    n_active_elec,
                    n_act,
                    n_core,
                    nuclear_repulsion=nuclear_repulsion,
                    ms2=ms2,
                    nroots=nroots,
                    options=solve_opts,
                    det_guess=_ci_guess["dets"],
                )
                cas_ = _dc_replace(
                    res_buf,
                    e_total=e_tots[0],
                    e_corr=e_tots[0] - e_hf,
                    ci_coeffs=ci_cols[:, 0].copy(),
                    nroots=nroots,
                    e_totals=list(e_tots),
                    ci_coeffs_all=(ci_cols if nroots > 1 else None),
                )
            else:
                cas_ = selected_casci(
                    h1_,
                    g_,
                    n_active_elec,
                    n_act,
                    n_core,
                    nuclear_repulsion=nuclear_repulsion,
                    e_hf=e_hf,
                    ms2=ms2,
                    nroots=nroots,
                    options=solve_opts,
                    det_guess=_ci_guess["dets"],
                )
            if grow:
                _ci_guess["dets"] = list(cas_.determinants)
            return cas_
        if spin_pure:
            # Lowest-nroots roots of the requested spin only (S = ms2/2):
            # solve with a root buffer and select by <S^2>, the averaging a
            # CSF-based code performs (see CASSCFOptions.spin_pure).
            from dataclasses import replace

            from ._ms_caspt2 import _spin_pure_roots

            ci_cols, e_tots, _s2s, res_buf = _spin_pure_roots(
                h1_,
                g_,
                n_core,
                n_act,
                n_active_elec,
                ms2,
                nroots,
                nuclear_repulsion=nuclear_repulsion,
                ci_guess=_ci_guess["v"],
            )
            return replace(
                res_buf,
                e_total=e_tots[0],
                e_corr=e_tots[0] - e_hf,
                ci_coeffs=ci_cols[:, 0].copy(),
                nroots=nroots,
                e_totals=list(e_tots),
                ci_coeffs_all=(ci_cols if nroots > 1 else None),
            )
        return casci(
            h1_,
            g_,
            n_active_elec,
            n_act,
            n_core,
            nuclear_repulsion=nuclear_repulsion,
            e_hf=e_hf,
            ms2=ms2,
            nroots=nroots,
            ci_guess=_ci_guess["v"],
        )

    def _solve(h1_, g_, grow=True):
        """CASCI + analytic orbital gradient in the basis (h1_, g_)."""
        cas_ = _solve_casci(h1_, g_, grow=grow)
        _ci_guess["v"] = (
            cas_.ci_coeffs_all if cas_.ci_coeffs_all is not None else cas_.ci_coeffs
        )
        if sa_weights is not None:
            dm1, dm2 = make_rdm12_sa(
                cas_.ci_coeffs_all,
                cas_.determinants,
                n_act,
                sa_weights,
            )
            e_sa = float(np.dot(sa_weights, cas_.e_totals))
        else:
            dm1, dm2 = make_rdm12(cas_.ci_coeffs, cas_.determinants, n_act)
            e_sa = cas_.e_total
        grad_, _, _, H_diag_ = _orbital_gradient_and_fock(
            h1_,
            g_.transpose(0, 2, 1, 3),
            dm1,
            dm2,
            n_core,
            n_act,
            norb,
            pairs,
        )
        return cas_, grad_, e_sa, H_diag_

    def _at(step, grow=True):
        U = expm(_build_kappa(step, pairs, norb))
        h1r, gr = _rotate_integrals(h1, g_phys, U)
        cas_, grad_, e_val, H_diag_ = _solve(h1r, gr, grow=grow)
        return cas_, grad_, U, h1r, gr, e_val, H_diag_

    if not pairs:
        cas = _solve_casci(h1, g_phys)
        e_out = cas.e_total
        if sa_weights is not None:
            e_out = float(np.dot(sa_weights, cas.e_totals))
        return CASSCFResult(
            e_total=e_out,
            e_corr=e_out - e_hf,
            cas=cas,
            h1e_cas=h1,
            h2e_cas=g_phys,
            mo_rotation=U_total,
            converged=True,
            n_iter=0,
            grad_norm=0.0,
            energy_trace=[e_out],
            e_totals=cas.e_totals if sa_weights is not None else [],
        )

    cas, grad, e_cur, H_diag = _solve(h1, g_phys)
    grad_norm = float(np.linalg.norm(grad))
    trace: list[float] = []
    converged = False
    it = 0
    e_per_root: list[float] = []

    # Determine whether to use Super-CI or NR in this iteration.
    use_superci = orbital_step in ("auto", "superci")

    for it in range(1, max_macro + 1):
        trace.append(e_cur)
        grad_norm = float(np.linalg.norm(grad))
        if verbose:
            method = "SuperCI" if use_superci else "NR"
            label = "SA-CASSCF" if sa_weights is not None else "CASSCF"
            write(
                f"  {label} iter {it:3d} [{method}] E = {e_cur:.10f}  "
                f"|g| = {grad_norm:.2e}\n"
            )
        if grad_norm < conv_tol_grad:
            converged = True
            if sa_weights is not None:
                e_per_root = list(cas.e_totals)
            break

        # Switch to NR in auto mode if the gradient is low enough, or if
        # Super-CI has stalled above the gradient threshold (energy
        # progress over the last 3 iterations below stall_to_nr each;
        # see the parameter docs).  trace[-1] is this iteration's energy
        # and descent is monotone, so the difference is the window gain.
        if orbital_step == "auto" and use_superci:
            if grad_norm < switch_to_nr:
                use_superci = False
            elif (
                stall_to_nr > 0.0
                and len(trace) >= 4
                and trace[-4] - trace[-1] < 3.0 * stall_to_nr
            ):
                if verbose:
                    write(
                        "    Super-CI stalled "
                        f"(dE over 3 iters = {trace[-4] - trace[-1]:.2e} "
                        f"Ha at |g| = {grad_norm:.2e}); switching to NR\n"
                    )
                use_superci = False

        if use_superci:
            step = _superci_step(grad, H_diag, trust)
        else:
            # Newton-CG: Hessian-vector products by central FD of the
            # analytic gradient (2 CASCI solves per CG iteration -- pair-count
            # independent, unlike the historical per-pair FD Hessian).
            # grow=False freezes the selected-CI determinant set during the
            # probes so both displacements sample the same fixed-selection
            # surface (re-selection per probe breaks the FD antisymmetry
            # and stalls the line search; no-op for the exact-CI backend).
            def _hvp(v: np.ndarray) -> np.ndarray:
                nv = float(np.linalg.norm(v))
                eps = fd_eps / max(nv, 1e-12)
                _, gp, *_ = _at(eps * v, grow=False)
                _, gm, *_ = _at(-eps * v, grow=False)
                return (gp - gm) / (2.0 * eps)

            step = _newton_cg_step(grad, H_diag, _hvp, trust)

        accepted = False
        for _bt in range(30):
            cas_n, grad_n, U, h1n, gn, e_new, H_diag_n = _at(step)
            if e_new <= e_cur + 1e-12:
                accepted = True
                break
            step = step * 0.5
        if not accepted:
            # The (quasi-)Newton direction failed the line search -- retry
            # along plain steepest descent before giving up (the diagonal
            # model can be qualitatively wrong on stiff large-CAS surfaces
            # while -g still descends).
            step = -grad * min(trust / max(grad_norm, 1e-12), 1.0)
            for _bt in range(30):
                cas_n, grad_n, U, h1n, gn, e_new, H_diag_n = _at(step)
                if e_new <= e_cur + 1e-12:
                    accepted = True
                    break
                step = step * 0.5
        if not accepted:
            break

        h1, g_phys, U_total = h1n, gn, U_total @ U
        cas, grad, e_cur, H_diag = cas_n, grad_n, e_new, H_diag_n

    return CASSCFResult(
        e_total=e_cur,
        e_corr=e_cur - e_hf,
        cas=cas,
        h1e_cas=h1,
        h2e_cas=g_phys,
        mo_rotation=U_total,
        converged=converged,
        n_iter=it,
        grad_norm=grad_norm,
        energy_trace=trace,
        e_totals=e_per_root,
    )
