"""Multi-state CASPT2: the MS-CASPT2 and XMS-CASPT2 effective Hamiltonians.

MS-CASPT2
  J. Finley, P.-Å. Malmqvist, B. O. Roos, L. Serrano-Andrés,
  Chem. Phys. Lett. 288, 299 (1998), doi:10.1016/S0009-2614(98)00252-8.
XMS-CASPT2
  A. A. Granovsky, J. Chem. Phys. 134, 214113 (2011), doi:10.1063/1.3596699
  (the model-space Fock rotation, introduced for XMCQDPT2);
  T. Shiozaki, W. Győrffy, P. Celani, H.-J. Werner, J. Chem. Phys. 135,
  081106 (2011), doi:10.1063/1.3633329 (XMS-CASPT2 proper).

State-specific (SS) CASPT2 corrects each CASSCF state independently, so two
interacting states' curves can cross artifactually near avoided crossings or
conical intersections (each state's perturber space treats the *other* state
as just another external configuration).  Multi-state CASPT2 instead builds
an effective Hamiltonian over the model space of N reference states,

    H_eff[i,j] = <Φ_i|Ĥ|Φ_j> + <Φ_i|Ĥ|Ψ_j⁽¹⁾>          (Finley 1998, Eq 11)

where ``Ψ_j⁽¹⁾`` is state j's first-order internally-contracted CASPT2
wavefunction.  ``H_eff`` is symmetrized (``1/2(H_eff + H_effᵀ)``, the
convention of Finley 1998 and of OpenMolcas ``&CASPT2``) and diagonalized;
its eigenvalues are the multi-state energies and its eigenvectors remix the
reference states (the "perturbatively modified CAS" states).

The two modes differ in the zeroth-order Hamiltonian:

* ``mode="ms"``: each state's H₀ uses the generalized Fock built from that
  state's *own* 1-RDM (Finley 1998; OpenMolcas ``MULTistate``,
  Fock operator = "state-specific").
* ``mode="xms"``: every state shares the H₀ built from the *state-averaged*
  1-RDM, and the model states are first rotated to diagonalize that Fock
  within the model space (Granovsky 2011, Sec. II.E; Shiozaki 2011, Eqs 5-8;
  OpenMolcas ``XMULtistate``, Fock operator = "state-average").  This
  restores invariance of the theory under rotations among the model states,
  the correct behavior at near-degeneracies, at the price of relaxing the
  exact SS-CASPT2 limit for well-separated states.

Implementation
--------------
Everything is built on the explicit determinant engine of :mod:`._mrpt` (the
machine-precision-validated small-CAS path): the model space is the lowest
``nroots`` spin-pure CASCI roots (the determinant CI is solved in one M_s
sector, so <S^2> filtering removes the other-multiplicity roots that a
CSF-based code like OpenMolcas never sees); each state's first-order
wavefunction comes from :func:`._mrpt._ic_caspt2_solve`; and every
H-matrix element (reference block <Φ_i|H|Φ_j> and coupling
<Φ_i|H|Ψ_j⁽¹⁾>) is an explicit determinant-space contraction: correct by
construction, no per-class contraction formulas to derive.

Like the rest of the explicit engine this targets the small-active-space /
small-basis regime; couplings cost one extra ``H|Φ_i>`` application per
(bra, ket-basis) pair.  Validated against OpenMolcas ``&CASPT2``
``MULTistate`` / ``XMULtistate`` (see ``tests/test_ms_caspt2.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np
from scipy.linalg import eigh

from ._casci import casci
from ._mrpt import (
    _add,
    _ann,
    _build_reference_state,
    _cre,
    _dot,
    _generalized_fock,
    _ic_caspt2_solve,
    _so,
    apply_1body,
    apply_2body,
)
from ._rdm import make_rdm1


@dataclass
class MSCASPT2Result:
    """Multi-state CASPT2 result (MS or XMS).

    All energies include the nuclear repulsion.  ``mixing`` columns follow
    ``energies`` order (ascending); both are expressed in the model-state
    basis actually perturbed: the CASCI roots for ``mode="ms"``, the
    XMS-rotated states for ``mode="xms"`` (compose with ``xms_rotation`` to
    return to the unrotated CASCI roots).

    Attributes
    ----------
    e_total : float
        Lowest multi-state root (the quantity ``run_job`` reports).
    energies : list[float]
        MS/XMS-CASPT2 total energies, ascending (eigenvalues of ``heff``).
    mixing : np.ndarray
        ``(nroots, nroots)`` eigenvectors of ``heff``; column k mixes the
        model states into multi-state root k.
    heff : np.ndarray
        Symmetrized effective Hamiltonian ``1/2(H+Hᵀ)`` (total energies).
    heff_asym : np.ndarray
        The asymmetric effective Hamiltonian before symmetrization.
    ss_energies : list[float]
        Diagonal of ``heff_asym``: the per-state single-state CASPT2 totals
        (for ``mode="xms"`` these belong to the *rotated* references and have
        no direct physical meaning; Shiozaki 2011).
    e2_corr : list[float]
        Per-state second-order corrections (diagonal couplings).
    ref_energies : list[float]
        Model-state reference energies <Φ_k|H|Φ_k> (CASCI eigenvalues for
        ``mode="ms"``; rotated-state expectation values for ``mode="xms"``).
    ref_hamiltonian : np.ndarray
        The reference block <Φ_i|H|Φ_j> computed *explicitly* in the
        determinant basis, a diagnostic that must equal
        ``diag(E_CASCI)`` (ms) / ``Uᵀ diag(E_CASCI) U`` (xms) to numerical
        precision (pinned in tests).
    xms_rotation : np.ndarray
        The model-space rotation U (identity for ``mode="ms"``); column j
        gives rotated state ``|j̃> = S_k U[k,j] |Φ_k>``.
    model_fock : np.ndarray
        The model-space Fock matrix <Φ_i|F̂|Φ_j> the XMS rotation
        diagonalizes (state-averaged F̂; zeros-shape for ``mode="ms"``).
    s2_values : list[float]
        <S^2> of the selected (spin-filtered) model-space CASCI roots.
    mode : str
        ``"ms"`` or ``"xms"``.
    nroots : int
        Number of model states.
    n_contracted : list[int]
        Retained contracted-space dimension per state.
    """

    e_total: float
    energies: list
    mixing: np.ndarray
    heff: np.ndarray
    heff_asym: np.ndarray
    ss_energies: list
    e2_corr: list
    ref_energies: list
    ref_hamiltonian: np.ndarray
    xms_rotation: np.ndarray
    model_fock: np.ndarray
    s2_values: list
    mode: str = "ms"
    nroots: int = 1
    n_contracted: list = field(default_factory=list)


def _apply_s_plus(state: dict, norb: int) -> dict:
    """Apply ``S₊ = S_p a+_{pa} a_{pb}`` to a spin-orbital-mask state."""
    out: dict = {}
    for mask, c in state.items():
        for p in range(norb):
            sg1, m1 = _ann(mask, _so(p, 1, norb))
            if sg1 == 0:
                continue
            sg2, m2 = _cre(m1, _so(p, 0, norb))
            if sg2 == 0:
                continue
            out[m2] = out.get(m2, 0.0) + sg1 * sg2 * c
    return out


def _s2_expectation(ci: np.ndarray, determinants: list, n_act: int, ms2: int) -> float:
    """<S^2> of an active-space CI vector.

    ``S^2 = S₋S₊ + S_z(S_z + 1)`` with ``S_z = ms2/2``, evaluated as
    ``|S₊psi|^2 + M_s(M_s+1)`` on the active space (the closed-shell core
    contributes nothing to S₊).
    """
    state = _build_reference_state(ci, determinants, 0, n_act)
    sp = _apply_s_plus(state, n_act)
    msz = 0.5 * ms2
    return _dot(sp, sp) + msz * (msz + 1.0)


def _spin_pure_roots(
    h1,
    h2e_phys,
    n_core,
    n_act,
    n_act_elec,
    ms2,
    nroots,
    *,
    nuclear_repulsion=0.0,
    ci_guess=None,
):
    """Lowest ``nroots`` spin-pure CASCI roots with S = ms2/2.

    The determinant CI is solved in the fixed M_s = ms2/2 sector, which
    contains every S >= M_s; a CSF-based reference (OpenMolcas RASSCF) only
    ever sees S = spin-1 roots.  Solve with a root buffer, filter by <S^2>,
    and return the CI columns + energies + <S^2> of the first ``nroots``
    states with S(S+1) ≈ (ms2/2)(ms2/2+1).

    Shared by the multi-state CASPT2 model space (electronic energies,
    ``nuclear_repulsion=0``) and the spin-pure SA-CASSCF root selection
    (which passes the nuclear repulsion and the previous macro-iteration's
    CI columns as a Davidson warm start).

    Returns ``(ci_cols (n_det, nroots), energies list, s2 list,
    casci_result)`` where ``casci_result`` is the buffered (unfiltered)
    :class:`CASCIResult` of the final solve.
    """
    s_target = 0.5 * ms2
    s2_target = s_target * (s_target + 1.0)
    n_alpha = (n_act_elec + ms2) // 2
    n_beta = n_act_elec - n_alpha
    n_det = comb(n_act, n_alpha) * comb(n_act, n_beta)
    n_solve = min(n_det, max(2 * nroots + 4, nroots))
    while True:
        res = casci(
            h1,
            h2e_phys,
            n_active_elec=n_act_elec,
            n_active_orb=n_act,
            n_core=n_core,
            nuclear_repulsion=nuclear_repulsion,
            ms2=ms2,
            nroots=n_solve,
            ci_guess=ci_guess,
        )
        cols = res.ci_coeffs_all if n_solve > 1 else res.ci_coeffs[:, None]
        sel, s2s = [], []
        for k in range(n_solve):
            s2 = _s2_expectation(cols[:, k], res.determinants, n_act, ms2)
            if abs(s2 - s2_target) < 1e-4:
                sel.append(k)
                s2s.append(s2)
            if len(sel) == nroots:
                break
        if len(sel) == nroots:
            ci_cols = np.ascontiguousarray(cols[:, sel])
            energies = [res.e_totals[k] for k in sel]
            return ci_cols, energies, s2s, res
        if n_solve >= n_det:
            raise ValueError(
                f"only {len(sel)} spin-pure roots with S={s_target:g} exist "
                f"in the CAS({n_act_elec},{n_act}) M_s={s_target:g} sector; "
                f"requested nroots={nroots}"
            )
        n_solve = min(n_det, 2 * n_solve)


def _semicanonical_rotation(F, n_core, n_act, norb):
    """Block rotation diagonalizing the core and virtual blocks of ``F``.

    The active block is untouched, so active-space CI vectors are invariant
    (the inactive core dressing of the active Hamiltonian is invariant under
    core-block rotations).
    """
    rot = np.eye(norb)
    for blk in (slice(0, n_core), slice(n_core + n_act, norb)):
        if blk.stop - blk.start > 0:
            _, U = eigh(F[blk, blk])
            rot[blk, blk] = U
    return rot


def _heff_pt2_blocks(
    ci_cols: np.ndarray,
    dets: list,
    h1: np.ndarray,
    eri: np.ndarray,
    n_core: int,
    n_act: int,
    norb: int,
    *,
    mode: str = "ms",
    n_frozen: int = 0,
    imaginary: float = 0.0,
    thresh: float = 1e-8,
    return_wavefunctions: bool = False,
) -> dict:
    """Per-state PT2 solves + effective-Hamiltonian blocks for EXPLICIT
    model states.

    This is the shared physics of the multi-state effective Hamiltonian,
    factored out of :func:`ms_caspt2` so the MS/XMS gradient
    (:mod:`vibeqc.gradient._ms_caspt2_grad`) can evaluate the same
    pipeline on *parametrized* (rotated / externally-mixed) model states
    without re-solving the model-space CASCI.  Single canonical
    implementation: :func:`ms_caspt2` delegates here.

    Parameters
    ----------
    ci_cols : (n_det, nroots) ndarray
        Model-state CI columns (need not be CASCI eigenvectors).
    dets : list
        Active-space determinant list matching ``ci_cols``.
    h1, eri : ndarray
        Full MO one-electron integrals and *chemist's* ERI.
    mode : str
        ``"ms"`` (per-state Fock) or ``"xms"`` (internal model-space
        Fock rotation + one shared SA Fock; Granovsky 2011 / Shiozaki
        2011 Eqs 5-8).

    Returns
    -------
    dict with keys
        ``e2`` (per-state E2 list), ``coup`` (off-diagonal couplings
        <Φ_i|H|Ψ_k⁽¹⁾>), ``ref_expl`` (explicit <Φ_i|H|Φ_k> block,
        includes the core energy), ``u_xms`` (model-space rotation,
        identity for ms), ``fmod`` (model-space Fock matrix), ``nbs``
        (retained contracted dimensions).  With ``return_wavefunctions=True``,
        the internal ``refs`` and ``psi1s`` determinant dictionaries are also
        returned for derivative-coupling overlap evaluation.  They follow the
        model basis used by ``coup`` (XMS-rotated for ``mode="xms"``);
        ``orbital_rotations`` maps each ``psi1s`` determinant expansion from
        its semicanonical core/virtual basis back to the input MO basis.
    """
    nroots = ci_cols.shape[1]
    dm1s = [make_rdm1(ci_cols[:, k], dets, n_act) for k in range(nroots)]

    # ── XMS: rotate the model states to diagonalize the SA Fock ─────────
    # Granovsky 2011 Sec. II.E / Shiozaki 2011 Eq 7: diagonalize
    # <Φ_i|F̂_SA|Φ_j> in the model space; |ĩ> = S_k U[k,i] |Φ_k>.  The
    # equal-weight SA density (hence F̂_SA) is invariant under the unitary
    # model-space rotation, so building it from the unrotated roots is
    # exact.
    u_xms = np.eye(nroots)
    fmod = np.zeros((nroots, nroots))
    f_sa = None
    if mode == "xms":
        dm1_sa = sum(dm1s) / float(nroots)  # equal-weight SA density
        f_sa = _generalized_fock(h1, eri, dm1_sa, n_core, n_act)
        if nroots > 1:
            refs0 = [
                _build_reference_state(ci_cols[:, k], dets, n_core, norb)
                for k in range(nroots)
            ]
            for j in range(nroots):
                fj = apply_1body(refs0[j], f_sa, norb)
                for i in range(nroots):
                    fmod[i, j] = _dot(refs0[i], fj)
            fmod = 0.5 * (fmod + fmod.T)
            _, u_xms = eigh(fmod)
            # deterministic column signs (largest-|component| positive)
            for j in range(nroots):
                jmax = int(np.argmax(np.abs(u_xms[:, j])))
                if u_xms[jmax, j] < 0.0:
                    u_xms[:, j] = -u_xms[:, j]
            ci_cols = ci_cols @ u_xms
            dm1s = [make_rdm1(ci_cols[:, k], dets, n_act) for k in range(nroots)]

    # ── per-state SS-CASPT2 solves ───────────────────────────────────────
    # mode="ms": state-specific Fock + per-state semicanonical basis;
    # mode="xms": one shared SA Fock + one shared semicanonical basis.
    def _prep(f_input):
        rot = _semicanonical_rotation(f_input, n_core, n_act, norb)
        h1_r = rot.T @ h1 @ rot
        eri_r = np.einsum(
            "pi,qj,rk,sl,pqrs->ijkl", rot, rot, rot, rot, eri, optimize=True
        )
        f_r = rot.T @ f_input @ rot
        return h1_r, eri_r, f_r, rot

    if mode == "xms":
        bases = [_prep(f_sa)]
        basis_of = [0] * nroots
    else:
        # F from each (spin-filtered) state's own density (Finley 1998).
        bases = [
            _prep(_generalized_fock(h1, eri, dm1s[k], n_core, n_act))
            for k in range(nroots)
        ]
        basis_of = list(range(nroots))

    # The model states' determinant representation is basis-independent
    # here: the semicanonical rotation touches only the core/virtual
    # blocks, under which the active CI vectors are invariant (the core
    # dressing of the active Hamiltonian is a core-block trace), so one
    # set of reference states serves every per-state basis.
    refs = [
        _build_reference_state(ci_cols[:, i], dets, n_core, norb) for i in range(nroots)
    ]

    e2 = [0.0] * nroots
    nbs = [0] * nroots
    psi1s: list = [None] * nroots
    w_solve: list = [None] * nroots
    for k in range(nroots):
        h1_r, eri_r, f_r, _rot = bases[basis_of[k]]
        P = dict(
            norb=norb,
            ref=refs[k],
            F=f_r,
            h1=h1_r,
            eri=eri_r,
            dm1_diag=np.diag(dm1s[k]).copy(),
        )
        e2_k, nb_k, psi1_k, w_k, _clagdx_k = _ic_caspt2_solve(
            P,
            n_core,
            n_act,
            n_frozen=n_frozen,
            imaginary=imaginary,
            thresh=thresh,
            return_wavefunction=True,
        )
        e2[k], nbs[k], psi1s[k], w_solve[k] = e2_k, nb_k, psi1_k, w_k

    # Couplings <Φ_i|H|Ψ_j⁽¹⁾> (Finley 1998 Eq 11): contract H|Φ_i> with
    # the ket state's first-order wavefunction, both expressed in the ket
    # state's semicanonical basis.  H|Φ_i> for the diagonal (i = ket) is
    # already available from the solve (W); cache the off-diagonal bras
    # per basis.
    w_bra: dict = {}
    for k in range(nroots):
        w_bra[(basis_of[k], k)] = w_solve[k]

    def _w_of(b, i):
        if (b, i) not in w_bra:
            h1_r, eri_r, _f, _rot = bases[b]
            w_bra[(b, i)] = _add(
                apply_1body(refs[i], h1_r, norb),
                apply_2body(refs[i], eri_r, norb),
            )
        return w_bra[(b, i)]

    coup = np.zeros((nroots, nroots))
    ref_explicit = np.zeros((nroots, nroots))
    for k in range(nroots):
        b = basis_of[k]
        for i in range(nroots):
            ref_explicit[i, k] = _dot(refs[i], w_solve[k])
            if i != k:
                coup[i, k] = _dot(_w_of(b, i), psi1s[k])

    result = dict(
        e2=e2, coup=coup, ref_expl=ref_explicit, u_xms=u_xms, fmod=fmod, nbs=nbs
    )
    if return_wavefunctions:
        result.update(
            refs=refs,
            psi1s=psi1s,
            orbital_rotations=[bases[basis_of[k]][3] for k in range(nroots)],
        )
    return result


def ms_caspt2(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_core: int = 0,
    n_virt: int = 0,
    *,
    nroots: int = 2,
    nuclear_repulsion: float = 0.0,
    mode: str = "ms",
    n_frozen: int = 0,
    imaginary: float = 0.0,
    ms2: int = 0,
    n_act_elec: int | None = None,
    thresh: float = 1e-8,
) -> MSCASPT2Result:
    r"""Multi-state (MS / XMS) internally-contracted CASPT2.

    The model space is the lowest ``nroots`` spin-pure CASCI roots
    (S = ``ms2``/2) of the active space in the supplied orbital basis;
    pass the converged integrals of a state-averaged CASSCF for the
    standard SA-CASSCF -> MS-CASPT2 composition.

    Per model state an internally-contracted single-state CASPT2 is solved
    (:func:`._mrpt._ic_caspt2_solve`) and the Finley effective Hamiltonian

        H_eff[i,j] = <Φ_i|Ĥ|Φ_j> + <Φ_i|Ĥ|Ψ_j⁽¹⁾>

    is assembled, symmetrized (``1/2(H+Hᵀ)``) and diagonalized
    (Finley, Malmqvist, Roos & Serrano-Andrés, Chem. Phys. Lett. 288, 299
    (1998), Eq 11).  ``mode="ms"`` uses each state's own generalized Fock
    for H₀; ``mode="xms"`` first rotates the model states to diagonalize
    the state-averaged Fock in the model space and uses that single
    state-averaged H₀ for every state (Granovsky, J. Chem. Phys. 134,
    214113 (2011); Shiozaki, Győrffy, Celani & Werner, J. Chem. Phys. 135,
    081106 (2011), Eqs 5-8).  The XMS state average is equal-weighted over
    the model states (the weighting for which the rotated theory is
    invariant).

    Parameters
    ----------
    h1e_mo, h2e_mo : ndarray
        Full MO integrals (physicist's ``g`` for ``h2e_mo``).
    n_core, n_virt : int
        Inactive / secondary orbital counts; the active space is what
        remains.
    nroots : int
        Number of model states (>= 1; ``nroots=1`` reduces to SS-CASPT2).
    nuclear_repulsion : float
        Added to all returned total energies.
    mode : str
        ``"ms"`` (state-specific Fock) or ``"xms"`` (rotated references +
        state-averaged Fock).
    n_frozen : int
        Deepest inactive orbitals kept uncorrelated in the PT2.
    imaginary : float
        Imaginary level shift s (Forsberg-Malmqvist 1997): per-state
        energies use the shift-corrected Hylleraas value and the couplings
        contract the s-shifted amplitudes (the OpenMolcas convention).
        Note the representation caveat: vibe-qc applies the complex shift
        to the full nondiagonal H₀ (basis-independent), while OpenMolcas
        adds s^2/Δ to the diagonal of its case-block-diagonal H₀
        representation: identical without intruders (<1e-8 Ha observed),
        diverging to ~1e-4 Ha near strong intruders.  s=0 results carry no
        such ambiguity.
    ms2 : int
        2.M_s of the active electrons; the model space is spin-filtered to
        S = ms2/2.
    n_act_elec : int
        Active-electron count (required; keyword-only).
    thresh : float
        Contracted-metric linear-dependence threshold.

    Returns
    -------
    MSCASPT2Result
    """
    if mode not in ("ms", "xms"):
        raise ValueError(f"mode must be 'ms' or 'xms', got {mode!r}")
    if nroots < 1:
        raise ValueError(f"nroots must be >= 1, got {nroots}")

    norb = h1e_mo.shape[0]
    n_act = norb - n_core - n_virt
    if n_act_elec is None:
        raise ValueError("n_act_elec must be supplied (active electron count)")

    h1 = np.asarray(h1e_mo, dtype=float).copy()
    eri = np.asarray(h2e_mo, dtype=float).transpose(0, 2, 1, 3).copy()  # chem

    # ── model space: lowest nroots spin-pure CASCI roots (input basis) ──
    ci_cols, e_ref_elec, s2s, _res = _spin_pure_roots(
        h1, h2e_mo, n_core, n_act, n_act_elec, ms2, nroots
    )
    dets = _res.determinants

    # ── shared physics: XMS rotation + per-state PT2 + couplings ────────
    blocks = _heff_pt2_blocks(
        ci_cols,
        dets,
        h1,
        eri,
        n_core,
        n_act,
        norb,
        mode=mode,
        n_frozen=n_frozen,
        imaginary=imaginary,
        thresh=thresh,
    )
    e2 = blocks["e2"]
    nbs = blocks["nbs"]
    coup = blocks["coup"]
    ref_explicit = blocks["ref_expl"]
    u_xms = blocks["u_xms"]
    fmod = blocks["fmod"]

    # ── effective Hamiltonian (electronic; nuclear added at the end) ────
    # Reference block: the model states diagonalize H in the model space,
    # so <Φ_i|H|Φ_j> = d_ij E_i (ms) and Uᵀ diag(E) U after the XMS
    # rotation (Shiozaki 2011 Eq 6).  The explicitly-contracted block is
    # also computed above as a machine-precision diagnostic.
    ref_mat = u_xms.T @ np.diag(e_ref_elec) @ u_xms

    heff_asym = ref_mat.copy()
    heff_asym[np.diag_indices(nroots)] += np.asarray(e2)
    heff_asym += coup
    # 1/2(H + Hᵀ): Finley 1998 / OpenMolcas mltctl convention.
    heff_sym = 0.5 * (heff_asym + heff_asym.T)
    w_ms, mix = eigh(heff_sym)
    for j in range(nroots):
        jmax = int(np.argmax(np.abs(mix[:, j])))
        if mix[jmax, j] < 0.0:
            mix[:, j] = -mix[:, j]

    nuc = float(nuclear_repulsion)
    return MSCASPT2Result(
        e_total=float(w_ms[0]) + nuc,
        energies=[float(e) + nuc for e in w_ms],
        mixing=mix,
        heff=heff_sym + nuc * np.eye(nroots),
        heff_asym=heff_asym + nuc * np.eye(nroots),
        ss_energies=[float(heff_asym[k, k]) + nuc for k in range(nroots)],
        e2_corr=[float(e) for e in e2],
        ref_energies=[float(ref_mat[k, k]) + nuc for k in range(nroots)],
        ref_hamiltonian=ref_explicit + nuc * np.eye(nroots),
        xms_rotation=u_xms,
        model_fock=fmod,
        s2_values=[float(s) for s in s2s],
        mode=mode,
        nroots=nroots,
        n_contracted=list(nbs),
    )
