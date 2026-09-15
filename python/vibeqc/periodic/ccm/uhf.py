"""Open-shell UHF-CCM self-consistent field (AICCM -- open-shell HF on a cyclic
cluster).

Unrestricted Hartree-Fock on the CCM-weighted integrals: the same WSSC-weighted
``S^CCM``/``h^CCM``/effective-ERI/``V_nn^CCM`` that drive the closed-shell
:func:`vibeqc.periodic.ccm.scf.run_ccm_rhf`, solved with two independent spin
densities so odd-electron / spin-polarised clusters (defect doublets, radical
chains, antiferromagnetic sublattices) are reachable. The effective ERI tensor
is spin-independent, so no new integral work is needed -- only the spin-resolved
Fock build:

    # Closed-shell-consistent UHF (e.g. Szabo & Ostlund, Modern Quantum
    # Chemistry, eqs 3.341-3.348), with every ERI the CCM-weighted one:
    #   J        = S_{rs} (muν|rs) (Pa + Pb)_{rs}        (total Coulomb)
    #   Ks_{muν}  = S_{rs} (mus|rν) Ps_{rs}               (per-spin exchange)
    #   Fs       = h^CCM + J - Ks ,   Ps = Cs_occ Cs_occ+
    #   E        = 1/2 S[(Pa+Pb)h + Pa Fa + Pb Fb] + V_nn^CCM
    # For a closed shell (Pa = Pb) this collapses to run_ccm_rhf exactly.

Spin counts come from the cluster supercell's charge + multiplicity
(``CCMSystem`` sets the multiplicity parity-correct: singlet for an even electron
count, doublet for odd; pass a periodic unit cell with an explicit multiplicity
for higher spin). Scope mirrors the closed-shell molecular route: exact in the
molecular limit, ~1e-5 for 1-D periodic clusters, dense ``n_ref_ao**4`` ERI so
small / 1-D (feasibly 2-D) clusters only (handovers/HANDOVER_AICCM.md).
"""

from __future__ import annotations

from .scf import _ccm_initial_guess, _with_ccm_guess

from dataclasses import dataclass

import numpy as np

from .experimental import _warn_experimental
from .integrals import ccm_overlap
from .padded import ccm_hcore, ccm_nuclear_repulsion
from .scf import (
    _ccm_eri_for_method,
    _diis_extrapolate,
    _orthonormaliser,
    _require_retained_occ,
    _validate_conv_tol_grad,
)

__all__ = ["CCMUHFResult", "run_ccm_uhf"]


@dataclass
class CCMUHFResult:
    """Open-shell HF-CCM result on the finite cyclic cluster.

    ``energy`` is the total energy of the finite cyclic cluster/supercell, NOT
    normalized per reference cell, exactly as
    :class:`~vibeqc.periodic.ccm.scf.CCMSCFResult` documents for the
    closed-shell driver: compare with a primitive-cell method using
    ``energy / ccm.n_cells``. Measured on a Li atom cell, nrep=(2,1,1):
    ``energy`` is twice the (1,1,1) value. The comment on this field
    previously said "per reference cell", which was wrong by a factor of
    ``N_c`` for every cluster above one cell.
    """

    converged: bool
    n_iter: int
    energy: float                 # TOTAL cyclic-cluster energy (Ha), not per cell
    energy_per_atom: float
    e_electronic: float
    e_nuclear: float
    n_alpha: int
    n_beta: int
    s_squared: float              # <S^2> (spin contamination diagnostic)
    mo_energies_alpha: np.ndarray
    mo_energies_beta: np.ndarray
    mo_coeffs_alpha: np.ndarray
    mo_coeffs_beta: np.ndarray
    density_alpha: np.ndarray     # Pa = Ca_occ Ca_occ^T
    density_beta: np.ndarray
    overlap: np.ndarray
    idempotency_error: float      # max_s ||Ps S Ps - Ps||_F
    exchange_q0: str | None = None  # exchange-q=0 convention label (direct route)
    #: Identity of the route/operator that actually produced this energy
    #: (IID 344): e.g. ``"ccm-fourcenter-dense-union12-uhf"`` or
    #: ``"ccm-neutral-direct-uhf"``. Stamped by every public producer so a
    #: record consumer can tell WHICH backend a number came from without
    #: out-of-band knowledge; ``""`` only on results predating the field.
    backend: str = ""
    #: The RSGDF high-``|G|`` tail cutoff (Ha) actually applied when building
    #: this route's neutral cderi, or ``None`` for base-mesh-only (GitLab IID
    #: 307). Recorded because the tail is what makes the direct and multi-k
    #: GDF controls the *same* Hamiltonian: they disagreed by -4.99e-01
    #: Ha/cell on MgO/STO-3G at ``nrep=(1,1,1)`` while one silently tailed and
    #: the other could not. Resolved by
    #: :func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_tail_ke_cutoff`. A
    #: caller-supplied ``cderi`` carries its own tail, which this route cannot
    #: observe, so the field stays ``None`` there.
    rsgdf_tail_ke_cutoff: float | None = None
    guess_selection: object = None

    @property
    def parity_held(self) -> bool:
        """Whether the executing driver held this result for external parity.

        Derived from the ``+PARITY_HELD`` marker on :attr:`backend` -- the
        hold mechanism's canonical carrier
        (:func:`vibeqc.pbc_gdf._gdf_backend_with_parity_hold`) -- so the
        structured flag can never drift from the string (IID 344). The UHF
        cluster routes never touch the hold class, so ``False`` there is an
        affirmative verdict.
        """
        return "+PARITY_HELD" in str(self.backend or "")


def run_ccm_uhf(ccm, *, initial_guess: object = "AUTO", method="union12", max_iter=128, conv_tol=1e-9,
                conv_tol_grad=1e-6, diis_dim=8,
                lindep_tol=1e-7, eri=None, cderi=None):
    """Run open-shell UHF-CCM on ``ccm`` (a :class:`CCMSystem`).

    ``conv_tol_grad`` -- the DIIS-commutator residual bound (default ``1e-6``,
    the historical gate; molecular ``UHFOptions.conv_tol_grad`` convention, as
    on :func:`~vibeqc.periodic.ccm.direct.run_ccm_uhf_direct`).

    Mirrors :func:`vibeqc.periodic.ccm.scf.run_ccm_rhf` with two spin densities.
    ``method`` selects the effective four-center (``"union12"`` default, or
    ``"aiccm2026dev-a"`` for the symmetric Born-von Kármán-torus four-center -- see
    :func:`vibeqc.periodic.ccm.scf.run_ccm_rhf`). ``eri`` may be a precomputed
    effective tensor (shared with an RHF/MP2 run for consistency) and then
    overrides ``method``. Returns a :class:`CCMUHFResult`.

    **Lean neutral UHF (``cderi=L``).** Given the neutral GDF cderi ``L[P,μν]``
    (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi`), J/K are assembled
    straight from ``L`` (``ccm_ri_j_neutral``/``ccm_ri_k_neutral`` per spin), so the
    dense ``n_ref_ao**4`` neutral ``g`` is **never formed** and the MOs share
    ``L``'s supercell-Γ AO basis. This is the open-shell sibling of
    :func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_ri_neutral`: it is the reference
    that **pairs with the open-shell RI correlation** -- ``run_ccm_ump2(ccm, uhf,
    cderi=L)`` and ``ccm_dlpno_ump2``/``ccm_dlpno_uccsd(..., cderi=L)`` -- so the
    whole open-shell correlation reaches moderate 3-D with no dense tensor anywhere.
    Equals ``run_ccm_uhf(eri=ccm_eri_neutral(ccm))`` to machine ε.

    Validated: reduces to ``run_ccm_rhf`` for a closed shell (Pa=Pb), and the
    molecular limit (isolated cluster) reproduces vibe-qc's molecular ``run_uhf``.
    """
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_uhf"
    )
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_uhf',
    )
    _warn_experimental()
    S = ccm_overlap(ccm)
    h, _, _ = ccm_hcore(ccm)
    e_nn = ccm_nuclear_repulsion(ccm)

    # Result-backend identity (IID 344): name the J/K operator that will
    # actually run -- neutral-RI when a cderi is supplied, the caller's
    # injected tensor, or the dense four-center at ``method``.
    if cderi is not None:
        _backend = "ccm-neutral-ri-uhf"
    elif eri is not None:
        _backend = "ccm-fourcenter-dense-injected-uhf"
    else:
        _backend = f"ccm-fourcenter-dense-{method}-uhf"
    if cderi is not None:
        from .ri import ccm_ri_j_neutral, ccm_ri_k_neutral
        _L = np.asarray(cderi, dtype=float)

        def _jk(Da, Db):
            return (ccm_ri_j_neutral(_L, Da + Db),
                    ccm_ri_k_neutral(_L, Da), ccm_ri_k_neutral(_L, Db))
    else:
        eri = _ccm_eri_for_method(ccm, method) if eri is None else eri

        def _jk(Da, Db):
            return (np.einsum("mnrs,rs->mn", eri, Da + Db, optimize=True),
                    np.einsum("msrn,rs->mn", eri, Da, optimize=True),
                    np.einsum("msrn,rs->mn", eri, Db, optimize=True))

    n_elec = ccm.supercell.n_electrons()
    mult = int(ccm.supercell.multiplicity)
    n_unpaired = mult - 1
    if (n_elec - n_unpaired) % 2 != 0 or n_unpaired > n_elec:
        raise ValueError(
            f"cluster electron count {n_elec} and multiplicity {mult} are "
            "incompatible (n_alpha/n_beta non-integer)."
        )
    n_alpha = (n_elec + n_unpaired) // 2
    n_beta = n_elec - n_alpha

    X = _orthonormaliser(S, lindep_tol)
    _require_retained_occ(
        X, max(n_alpha, n_beta), who="run_ccm_uhf", lindep_tol=lindep_tol)

    def diag_fock(F):
        Fp = X.T @ F @ X
        eps, Cp = np.linalg.eigh(Fp)
        return eps, X @ Cp

    def density(C, n_occ):
        Co = C[:, :n_occ]
        return Co @ Co.T                       # UHF spin density (no factor 2)

    # Core-Hamiltonian initial guess (same for both spins; symmetry-broken by SCF).
    eps_a, Ca = diag_fock(h)
    eps_b, Cb = diag_fock(h)
    Da = density(Ca, n_alpha)
    Db = density(Cb, n_beta)

    diis_Fa, diis_Fb, diis_e = [], [], []
    e_last = 0.0
    converged = False
    for it in range(1, max_iter + 1):
        J, Ka, Kb = _jk(Da, Db)
        Fa = h + J - Ka
        Fb = h + J - Kb
        Fa = 0.5 * (Fa + Fa.T)                 # enforce Hermiticity (WIP eff)
        Fb = 0.5 * (Fb + Fb.T)

        e_elec = 0.5 * np.sum((Da + Db) * h + Da * Fa + Db * Fb)
        e_tot = e_elec + e_nn

        # Combined-spin DIIS: stack the per-spin commutator errors so one set of
        # coefficients extrapolates Fa and Fb together.
        err_a = X.T @ (Fa @ Da @ S - S @ Da @ Fa) @ X
        err_b = X.T @ (Fb @ Db @ S - S @ Db @ Fb) @ X
        err = np.stack([err_a, err_b])
        if len(diis_Fa) == diis_dim:
            diis_Fa.pop(0)
            diis_Fb.pop(0)
            diis_e.pop(0)
        diis_Fa.append(Fa)
        diis_Fb.append(Fb)
        diis_e.append(err)
        if len(diis_Fa) >= 2:
            Fa = _diis_extrapolate(diis_Fa, diis_e)
            Fb = _diis_extrapolate(diis_Fb, diis_e)

        eps_a, Ca = diag_fock(Fa)
        eps_b, Cb = diag_fock(Fb)
        Da = density(Ca, n_alpha)
        Db = density(Cb, n_beta)

        de = e_tot - e_last
        e_last = e_tot
        # The energy is stationary at convergence, so |dE| is QUADRATIC in the
        # density error -- an energy-only gate can report converged=True with a
        # ~1e-5-loose minority-spin density. The DIIS-residual bound is
        # therefore an explicit, user-tightenable criterion (conv_tol_grad;
        # default 1e-6 = the historical gate), the molecular
        # UHFOptions/UKSOptions.conv_tol_grad convention.
        if it > 1 and abs(de) < conv_tol and np.max(np.abs(err)) < conv_tol_grad:
            converged = True
            break

    # <S^2> with spin contamination (Szabo & Ostlund eq. 3.291 form):
    #   <S^2> = Sz(Sz+1) + Nb - S_{iina_occ, jinb_occ} |<a_i|S|b_j>|^2
    sz = 0.5 * (n_alpha - n_beta)
    ovlp_ab = Ca[:, :n_alpha].T @ S @ Cb[:, :n_beta]
    s_squared = float(sz * (sz + 1.0) + n_beta - np.sum(ovlp_ab ** 2))

    idem = max(np.linalg.norm(Da @ S @ Da - Da), np.linalg.norm(Db @ S @ Db - Db))
    return _with_ccm_guess(CCMUHFResult(
        converged=converged, n_iter=it, energy=e_tot,
        energy_per_atom=e_tot / ccm.n_atoms, e_electronic=e_elec, e_nuclear=e_nn,
        n_alpha=n_alpha, n_beta=n_beta, s_squared=s_squared,
        mo_energies_alpha=eps_a, mo_energies_beta=eps_b,
        mo_coeffs_alpha=Ca, mo_coeffs_beta=Cb,
        density_alpha=Da, density_beta=Db, overlap=S,
        idempotency_error=float(idem),
        backend=_backend,
    ), guess_selection)
