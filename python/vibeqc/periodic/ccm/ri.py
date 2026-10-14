"""<=3-center Coulomb approximations and fitted-torus controls (experimental).

The four-center CCM Coulomb is O(N⁴). Density fitting (RI) collapses it to three
centers. Two routes are provided, both **experimental**:

1. **Neutral fitted-torus Bloch control** (``run_ccm_rhf_gdf`` /
   ``run_ccm_rks_gdf``; open-shell ``run_ccm_uhf_gdf`` /
   ``run_ccm_uks_gdf``) -- multi-k periodic GDF on the unit cell. Together
   with the corresponding restricted or unrestricted direct control in
   :mod:`vibeqc.periodic.ccm.direct`, this evaluates one specified
   block-circulant neutral Hamiltonian in Fourier-related Bloch and real-Γ
   representations. It is selected by ``"neutral-bloch"`` / ``"gdf"`` /
   ``"aiccm-ri"`` and is not the union-and-weight Γ-CCM construction.

2. **Molecular WSSC RI-J / RIJCOSX** (``run_ccm_rhf_rij`` / ``run_ccm_rhf_rijcosx``)
   -- the *research* route that keeps the explicit WSSC three-center weighting
   (eq-13 union, the V^CCM pattern with the nucleus -> auxiliary function P):

       B_{muν}^P = S_g w_muν.1/2(w_muP+w_νP)(g).(mu_0 ν_0 | P_g),  V^CCM_PQ = S_g w_PQ(g)(P_0|Q_g)
       J = B (V^CCM)⁻¹ Bᵀ D

   It is exact in the isolated limit but ~few-% off periodically: this bare-1/r
   union 3-center weighting is **not RI-consistent** with the four-center --
   because the bare-1/r four-center is *non-separable*, no weighting of its
   3-center can reproduce it (the RI-J floors at ~1-2 %, AICCM_ALGORITHM.md
   Sec.13.7). ``run_ccm_rhf_rijcosx`` adds chain-of-spheres exchange (Neese 2009)
   for K (short-ranged within the WSC).

**RI-consistency -- resolved (Sec.13.7/Sec.13.9).** The RI-consistent CCM three-center
is the **neutral cderi** ``L`` (:func:`ccm_ri_tensors_neutral`, the density fit
of the neutral torus kernel ``v_E``; see :mod:`vibeqc.periodic.ccm.neutral`). Its
RI-J/RI-K (:func:`ccm_ri_j_neutral` / :func:`ccm_ri_k_neutral`) reproduce the
neutral four-center *exactly* (to machine e), where the bare-1/r union RI-J above
is ~1e-3 off the bare four-center. The RI-consistent neutral fitted-torus
**control energy** is the GDF route (``run_ccm_rhf_gdf``): contracting ``L``
reproduces the GDF Fock, and the absolute energy follows the GDF neutral gauge
(the neutralising-background ξ-bookkeeping across e-e/e-n/n-n; not hand-rolled
here -- CLAUDE.md Sec.7). Agreement of its Bloch and real-Γ forms is same-H
representation evidence. The finite Fourier theorem applies to any specified
block-circulant Hamiltonian, but does not equate the Γ-CCM and χ-CCM
constructions.

Small / low-dimensional clusters (dense padded 3-center build). Reference:
Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014); Neese et al., Chem. Phys.
356, 98 (2009) (RIJCOSX).
"""

from __future__ import annotations

from .scf import _ccm_initial_guess, _with_ccm_guess, _ccm_read_density

import numpy as np

from .experimental import _warn_experimental
from .integrals import ccm_overlap
from .neutral import ccm_neutral_cderi
from .padded import (
    _ao_ranges,
    build_padded_cluster,
    ccm_eri,
    ccm_eri_symmetric,
    ccm_hcore,
    ccm_nuclear_repulsion,
    wssc_cells,
)
from .scf import (
    CCMSCFResult,
    _scf_trace_row,
    _commutator_exit_ok,
    _diis_extrapolate,
    _orthonormaliser,
    _require_retained_occ,
    _validate_conv_tol_grad,
)

__all__ = [
    "ccm_ri_tensors",
    "ccm_ri_tensors_neutral",
    "ccm_ri_j_neutral",
    "ccm_ri_k_neutral",
    "run_ccm_rhf_gdf",
    "run_ccm_uhf_gdf",
    "run_ccm_rks_gdf",
    "run_ccm_uks_gdf",
    "run_ccm_rhf_ri_neutral",
    "run_ccm_rhf_rij",
    "run_ccm_rhf_rijcosx",
]


# ----- RI-consistent neutral three-center (density fit of v_E, Sec.13.7) ---------

def ccm_ri_tensors_neutral(ccm, *, ke_cutoff=200.0, aux_basis=None):
    """The RI-consistent CCM three-center: the neutral cderi ``L[P,muν]`` (Sec.13.7).

    Thin wrapper over :func:`vibeqc.periodic.ccm.neutral.ccm_neutral_cderi`. Unlike
    :func:`ccm_ri_tensors` (which returns ``(B, V⁻¹)`` for the bare-1/r union
    weighting), the neutral cderi already folds the metric in, so the single
    tensor ``L`` is all that is needed: ``J = S_P L[P,muν](S_rs L[P,rs] D_rs)``
    (:func:`ccm_ri_j_neutral`), ``K`` via :func:`ccm_ri_k_neutral`. These
    reproduce the neutral four-center exactly (RI-consistent), where the bare
    union RI-J floors at ~1e-3 (non-separable bare-1/r four-center)."""
    return ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff, aux_basis=aux_basis)


def ccm_ri_j_neutral(cderi, D):
    """RI-consistent neutral Coulomb ``J_muν = S_P L[P,muν] (S_rs L[P,rs] D_rs)``."""
    return np.einsum("Pmn,P->mn", cderi,
                     np.einsum("Prs,rs->P", cderi, D, optimize=True), optimize=True)


def ccm_ri_k_neutral(cderi, D):
    """RI-consistent neutral exchange ``K_muν = S_P S_rs L[P,mur] L[P,sν] D_rs``.

    Evaluated as two BLAS-3 passes -- ``W[P,m,s] = S_r L[P,m,r] D[r,s]``
    (one batched GEMM), then ``K = S_{P,s} W[P,m,s] L[P,s,n]`` -- instead
    of the direct three-operand einsum. Bit-exact reformulation (measured
    4e-14 max element on LiH pob-TZVP-REV2 (2,2,2)); ~2x faster there and
    increasingly BLAS-bound at the production ``O(n_aux n^3)`` sizes where
    this contraction is the post-fold wall
    (the original investigation, efficiency program). No intermediate
    beyond the one ``(n_aux, n, n)`` work tensor -- the same footprint the
    einsum path materialised internally.

    For an SCF-loop density (always an aufbau ``D = c X X^T``, positive
    semidefinite with rank = the occupation count) the factored sibling
    :func:`ccm_ri_k_neutral_factored` contracts through the rank instead
    -- an ``n/n_occ`` flop reduction. This dense-D form is the reference
    the factored path is gated against.
    """
    L = np.asarray(cderi)
    W = np.matmul(L, np.asarray(D))           # (P, n, n) batched GEMM
    return np.einsum("Pms,Psn->mn", W, L, optimize=True)


def psd_density_factor(D, *, tol=1e-12):
    """Rank-revealing factor ``X`` of a PSD density: ``D = X @ X.T``
    (``X @ X^H`` for a Hermitian input).

    ``eigh``-based (exact to machine precision; a pivoted-Cholesky
    variant was measured to truncate within its stopping tolerance on
    diffuse pob-class densities and amplify to ~1e-6 in K, so the O(n^3)
    eigh -- still ~n_aux times cheaper than the K contraction it feeds --
    is the deliberate choice). Eigenpairs below ``tol`` relative to the
    largest eigenvalue are dropped, so an aufbau density
    ``D = 2 C_o C_o^T`` returns rank ``n_occ`` regardless of basis size.
    Raises on a significantly indefinite input (a DIIS-extrapolated
    *density* would be one -- the CCM loops extrapolate the Fock, so
    every density they feed is aufbau-PSD by construction; this guard
    keeps the factored exchange from silently mangling any other
    caller's input).
    """
    D = np.asarray(D)
    if not np.iscomplexobj(D):
        # A real symmetric density; a Hermitian mesh block Dh(k) of the
        # block-storage exchange (periodic/ccm/block_cderi.py) is factored
        # by the same rule, D = X X^H, and stays complex.
        D = np.asarray(D, dtype=float)
    w, U = np.linalg.eigh(D)
    w_max = max(float(w[-1]), 0.0)
    if float(w[0]) < -1e-8 * max(w_max, 1.0):
        raise ValueError(
            "psd_density_factor: input is significantly indefinite "
            f"(min eig {float(w[0]):.3e}); the factored exchange requires "
            "an aufbau-PSD density."
        )
    keep = w > tol * max(w_max, 1.0)
    return np.ascontiguousarray(U[:, keep] * np.sqrt(w[keep])[None, :])


def ccm_ri_k_neutral_factored(cderi, X):
    """Neutral exchange from a rank factor ``X`` of the density.

    With ``D = X @ X.T`` (``X`` shape ``(n, k)``, ``k`` = occupation
    rank -- :func:`psd_density_factor`):

        B [P,m,i] = S_r L[P,m,r] X[r,i]          (batched GEMM, O(n_aux n^2 k))
        Bt[P,s,i] = S_n L[P,s,n]-transposed X    (second batched GEMM)
        K_mn      = S_{P,i} B[P,m,i] Bt[P,n,i]   (one GEMM over (P,i))

    The two factors are NOT the same object: the exchange contraction
    uses ``L[P,m,r] ... L[P,s,n]`` -- bra index first on one factor,
    second on the other -- and a fold-built cderi block ``L[P]`` is not
    elementwise symmetric in general (s-only fixtures hide this;
    measured 5.6e-6 on LiH pob-TZVP-REV2 (2,2,2) when the transpose is
    dropped). Exact reformulation of :func:`ccm_ri_k_neutral` for PSD
    ``D`` -- machine-precision agreement gated in
    ``tests/test_ccm_direct.py`` -- at ``~1.5 k/n`` of its flops (the
    efficiency-program K lever: n_occ << n at production bases). The
    seam/zero-mode term ``S D S = (S X)(S X)^T`` factors the same way at
    negligible cost.
    """
    L = np.asarray(cderi)
    X = np.asarray(X, dtype=float)
    n_aux, n, k = L.shape[0], L.shape[1], X.shape[1]
    B = np.matmul(L, X)                              # (P, n, k) = L_P X
    C = np.matmul(X.T, L)                            # (P, k, n) = X^T L_P
    # Column ordering (P outer, i inner) must match between the two flats.
    Bf = np.ascontiguousarray(B.transpose(1, 0, 2).reshape(n, n_aux * k))
    Cf = np.ascontiguousarray(C.transpose(2, 0, 1).reshape(n, n_aux * k))
    return Bf @ Cf.T

_EXACT_K = {"union12": ccm_eri, "aiccm2026dev-a": ccm_eri_symmetric,
            "aiccmdev": ccm_eri_symmetric}   # deprecated alias (back-compat)


# ----- Route 1: neutral fitted-torus control via multi-k GDF -----------------

from dataclasses import dataclass as _dataclass


@_dataclass
class CCMGDFResult:
    """Neutral fitted-torus GDF control, normalised per atom for comparison."""
    converged: bool
    energy: float            # per unit cell (Ha)
    energy_per_atom: float
    n_iter: int
    raw: object              # the underlying Periodic*GDFResult
    #: The q=0 exchange-divergence convention this route applied, recorded as a
    #: result field (mirrors :attr:`CCMDirectResult.exchange_q0`). The HF/hybrid
    #: multi-k GDF path applies ``exxdiv="ewald"`` (:func:`apply_exxdiv_ewald_to_K`),
    #: so a run with exact exchange carries :data:`~vibeqc.periodic.exchange_convention.BVK_EWALD`.
    #: This is a **read** keyed on the driver's computed ``e_hf_exchange`` (nonzero
    #: iff the ewald K-shift was applied), not a re-derivation. Pure DFT (no exact
    #: exchange) carries ``""`` with applicability ``"inactive"``.
    exchange_q0: str = ""
    #: ``"active"`` iff exact exchange is present (RHF, or a hybrid functional) so
    #: the q=0 convention applies; ``"inactive"`` for a pure DFT run.
    exchange_q0_applicability: str = "inactive"
    #: The inner multi-k driver's backend string, carried verbatim so a
    #: consumer of the result can tell a held number from a converged one:
    #: the ``+PARITY_HELD`` suffix (see
    #: :func:`~vibeqc.pbc_gdf._gdf_backend_with_parity_hold`) marks the
    #: dense-core parity-hold class and must never be dropped on its way out
    #: of the CCM wrapper.
    backend: str = ""
    #: Number of expensive per-k-pair fit builds after optional CCM
    #: space-group/time-reversal star reduction (zero when no Lpq cache is used).
    gdf_pair_builds: int = 0
    #: Number of Lpq cache entries required by the unreduced production route
    #: (zero when no Lpq cache is used). Without the star plan this is the
    #: driver's own count (``raw.gdf_cderi_pairs``), not a re-derivation.
    gdf_pair_total: int = 0
    #: ``gdf_pair_total / gdf_pair_builds`` (one when reduction is inactive).
    gdf_pair_reduction_factor: float = 1.0
    #: RSGDF high-``|G|`` tail cutoff (Ha) the inner driver actually
    #: applied, or ``None`` for base-mesh-only. Read from the driver
    #: result -- never re-derived here -- so it cannot drift from what ran
    #: (the IID 344 discipline). This is the field that makes the neutral
    #: control pair auditable: the direct sibling records the same
    #: quantity, and the two disagreeing was worth -4.99e-01 Ha/cell on
    #: MgO/STO-3G at ``nrep=(1,1,1)`` (GitLab IID 307).
    rsgdf_tail_ke_cutoff: float | None = None
    #: Which implementation of ``gdf_method="rsgdf"`` fitted this run:
    #: ``"classic"`` (the all-reciprocal FFT fit, the one the real-Γ fold
    #: builds), ``"sr"`` (the range-separated source that has been the 3-D
    #: default of the multi-k driver since v0.17.1) or ``"not-applicable"``.
    #: Read from the driver's executed fit-source record, never re-derived.
    #: The two sources do not fit the same Coulomb operator (1.4e-4 Ha/cell
    #: apart on LiH/STO-3G (1,1,1)), so a Theorem-1 pair must match here as
    #: well as on the tail (vibe-qc #378).
    fit_source: str = ""
    #: The driver's executed fit-source descriptor
    #: (:class:`~vibeqc.periodic_fit_source.RSGDFFitSource`) when a shared-q
    #: engine built the fit, else ``None``.
    executed_fit_source: object = None
    #: The :class:`~vibeqc.fock_state.SourceProvenance` of the neutral
    #: fitted-torus Hamiltonian this run evaluated, built from the driver's
    #: executed fit-source record (#519, RG-2); ``None`` when the driver
    #: recorded no fit source. The real-Gamma producer stamps the same record
    #: (``result.neutral_source``), so a Theorem-1 pair is checked by
    #: identity (:func:`~vibeqc.periodic.ccm.source.assert_matched_neutral_source`).
    neutral_source: object = None
    #: IID 291 / ruling R6 (mirrors ``CCMSCFResult``): whether the producer was
    #: asked to run a cell measured as vacuum-padded, and the 1-based lattice
    #: directions that measured as vacuum. ``(False, ())`` on a cell measured
    #: as genuinely 3-D.
    vacuum_padding_allowed: bool = False
    vacuum_padded_axes: tuple = ()
    #: The inner multi-k driver's per-cycle trace, carried verbatim (O29,
    #: paper issue #17): the Bloch producer previously exposed only
    #: ``converged``/``n_iter``. Each row is the native ``SCFIteration``
    #: (``iter``, ``energy``, ``delta_e``, ``grad_norm``, ...).
    scf_trace: tuple = ()

    @property
    def xc_lattice_cutoff_bohr(self) -> float:
        """Executed primitive-cell XC reach, read from the Bloch producer."""
        return float(getattr(self.raw, "xc_lattice_cutoff_bohr", 0.0))

    @property
    def achieved_convergence(self) -> dict:
        """The convergence actually reached, read from :attr:`scf_trace`.

        Mirrors ``CCMSCFResult.achieved_convergence`` so the two producers
        of the neutral Hamiltonian expose the same record; the residual
        metric differs (the multi-k driver records a gradient *norm*, the
        real-Γ loop a max-abs commutator element) and is named so a reader
        never compares the two as if they were one quantity.
        """
        tr = tuple(self.scf_trace or ())
        if not tr:
            return {}
        last = tr[-1]
        return {
            "delta_e": float(getattr(last, "delta_e", float("nan"))),
            "grad_max": float(getattr(last, "grad_norm", float("nan"))),
            "grad_metric": "gradient_norm (multi-k driver)",
            "cycles": int(len(tr)),
        }

    @property
    def parity_held(self) -> bool:
        """Whether the inner multi-k driver held this result for parity.

        Structured form of the ``+PARITY_HELD`` marker on :attr:`backend`
        (IID 344): derived from the string -- the hold mechanism's canonical
        carrier -- so the flag can never drift from it. A held absolute
        energy is known-wrong versus the external reference (the dense-core
        P01 class); energy *differences* may still be usable. Consumers must
        separate held rows from converged rows on this field instead of
        grepping the ``.err`` sidecar.
        """
        return "+PARITY_HELD" in str(self.backend or "")

    @property
    def guess_selection(self):
        """The actual SCF reference selection, forwarded without re-resolution."""
        return getattr(self.raw, "guess_selection", None)

    @property
    def fit_cache(self):
        """The multi-k driver's retained per-pair fit, or ``None``.

        Populated when the producer ran with ``retain_fit_cache=True`` and the
        driver took the classic RSGDF multi-k path (an
        :class:`~vibeqc.periodic_k_gdf.RSGDFFitCache`: every ``(k_i, k_j)``
        fit in the eigenmode frame plus the ``M(q)`` eigensystems). The
        real-Γ sibling consumes it instead of refitting when both producers
        run in one process: ``run_ccm_rhf_direct(ccm, fit_cache=g.fit_cache)``.
        ``None`` on the Γ fast path, on the SR/LR engine and on the private
        MDF/compcell sources, which build other objects.
        """
        return getattr(self.raw, "fit_cache", None)


def _ccm_gdf_symmetry_cache_builder(ccm, symmetry, stats):
    """Return the private multi-k Lpq cache builder for a CCM star.

    The generic periodic GDF driver owns the fit implementation and cache
    contractions; this closure supplies only the CCM-specific space-group
    equivalence relation.  Representatives are built in the canonical
    auxiliary frame and other members are reconstructed by AO/auxiliary
    covariance plus time reversal. Which relations exist is decided by
    :func:`~vibeqc.periodic.ccm.symmetry.ccm_symmetry_fold_kpair_plan`, which
    since the vibeqc#337 fix admits only ops that preserve the finite
    Bloch cell list, so every member it emits is exact on the truncated build.
    Fully 3-D replica meshes are still excluded by :func:`_ccm_gdf`, now for
    cost rather than correctness: this per-pair cache would replace the
    generic builder's ``n_k`` shared-q batches with ``n_k**2`` individual pair
    calls.
    """
    def _build_cache(build_pair, kpts, ubasis, aux_modrho, need_k_pairs):
        from .neutral import _ao_atom_indices, _fold_star_reconstruct
        from .symmetry import (
            analyze_ccm_symmetry,
            ccm_symmetry_fold_kpair_plan,
        )

        kpts = np.asarray(kpts, dtype=float)
        n_k = len(kpts)
        nrep = np.asarray(ccm.nrep, dtype=int)
        lattice = np.asarray(ccm.unit_system.lattice, dtype=float)
        reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
        labels = (
            np.round(np.linalg.solve(reciprocal, kpts.T).T * nrep[None, :])
            .astype(int)
            % nrep[None, :]
        )
        index_by_label = {tuple(label): i for i, label in enumerate(labels)}
        zero_q = index_by_label[(0, 0, 0)]
        q_indices = list(range(n_k)) if need_k_pairs else [zero_q]
        sym = (
            symmetry
            if not isinstance(symmetry, (bool, str))
            else analyze_ccm_symmetry(ccm)
        )
        plan = ccm_symmetry_fold_kpair_plan(
            ccm, sym, kpts, q_indices, ubasis, aux_modrho
        )

        stats.update(
            builds=int(plan.n_reps),
            total=int(plan.n_builds),
            factor=float(plan.reduction_factor),
        )
        if plan.n_reps == plan.n_builds:
            if need_k_pairs:
                return {
                    (i, j): build_pair(kpts[i], kpts[j])
                    for i in range(n_k)
                    for j in range(n_k)
                }
            return {(i, i): build_pair(kpts[i], kpts[i]) for i in range(n_k)}

        ao_at = _ao_atom_indices(ubasis)
        aux_at = _ao_atom_indices(aux_modrho)
        referenced = {
            entry[1] for entry in plan.entries.values()
            if entry[0] == "recon"
        }
        representatives = {}
        cache = {}
        for qi in q_indices:
            q_label = labels[qi]
            for ai in range(n_k):
                target_label = tuple((labels[ai] + q_label) % nrep)
                bi = index_by_label[target_label]
                entry = plan.entries[(qi, ai)]
                if entry[0] == "build":
                    tensor = build_pair(
                        kpts[ai],
                        kpts[ai] + kpts[qi],
                        canonical_auxiliary_basis=True,
                    )
                    if (qi, ai) in referenced:
                        representatives[(qi, ai)] = tensor
                else:
                    _, (rqi, rai), op_index, time_reversal = entry
                    tensor = _fold_star_reconstruct(
                        representatives[(rqi, rai)],
                        plan.ops[op_index],
                        kpts[rai],
                        kpts[rai] + kpts[rqi],
                        time_reversal,
                        ao_at,
                        aux_at,
                    )
                cache[(ai, bi)] = tensor
        return cache

    return _build_cache


def _gdf_guess_kwargs(driver_kwargs, *, is_ks):
    """Translate the convenience selector into the native options contract.

    Preserve every supplied option field and leave the caller's object intact.
    Without a convenience keyword, the options selector remains authoritative.
    """
    from ...guess import coerce_initial_guess
    from ..._vibeqc_core import PeriodicKSOptions, PeriodicRHFOptions

    kw = dict(driver_kwargs)
    if "initial_guess" not in kw:
        return kw
    requested = coerce_initial_guess(kw.pop("initial_guess"))
    source = kw.get("options")
    if source is None:
        options = PeriodicKSOptions() if is_ks else PeriodicRHFOptions()
    else:
        options = _copy_options(source)
    options.initial_guess = requested
    kw["options"] = options
    return kw


def _copy_options(source):
    """A same-typed copy of a native options object (pybind value objects
    cannot be copied with :mod:`copy`); nested value fields such as
    ``lattice_opts`` are copied by value."""
    options = type(source)()
    for name in dir(source):
        if not name.startswith("_"):
            value = getattr(source, name)
            if not callable(value):
                setattr(options, name, value)
    return options


def _pin_classic_fit_source(ccm, kw, *, symmetry, who):
    """Pin a neutral Bloch producer to the classic fit and the fold's tail (#378).

    Ruling R1 / Paper 1 Theorem 1: this producer and the real-Γ one must
    evaluate the SAME Hamiltonian. The real-Γ fold
    (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi_fold`) builds the
    classic all-reciprocal fit and mirrors the Γ fast path's auto-sized tail
    (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_tail_ke_cutoff`), so this
    side is pinned to ``fit_source="classic"`` with that same tail passed
    explicitly (the multi-k engine, which owns the Γ limit under an explicit
    source, auto-sizes no tail of its own). Under the driver's shipped default
    (``"auto"``, the SR/LR source on every 3-D cell since v0.17.1) the two
    producers fit different Coulomb operators: the identity broke by 1.4e-4
    Ha/cell on LiH/STO-3G (1,1,1), 1.5e-4 on MgO/STO-3G and 4.4e-5 on the
    paper's 25-bohr chain, and the Bloch side was 20 to 175 times slower
    (bisected to the v0.17.0..v0.17.1 periodic GDF refactor).
    The pin lifts when the SR/LR source is accepted numerically AND the fold is
    rebuilt on it, both halves. An explicit ``fit_source`` in ``driver_kwargs``
    still wins, so the comparison can be made deliberately.

    Returns the requested source. Raises when the CCM k-pair star reduction is
    requested alongside an explicit source: the driver refuses a private fit
    adapter together with one, and the reduction is approximate on 3-D cells
    anyway (+6.7e-5 Ha/cell on LiH/STO-3G (2,2,1)).
    """
    from .neutral import ccm_neutral_tail_ke_cutoff

    unit = ccm.unit_system
    gdf_method = str(kw.get("gdf_method", "rsgdf")).lower()
    # Gradient requests keep the pin too (#404): the multi-k driver
    # differentiates the classic fit on the historical index ball set below,
    # so the real-Γ gradient borrowed behind ``run_ccm_direct_gradient``'s
    # energy-parity gate is the gradient of the Hamiltonian both producers
    # converge.
    if int(unit.dim) == 3 and gdf_method == "rsgdf":
        kw.setdefault("fit_source", "classic")
    fit = str(kw.get("fit_source", "auto"))
    if fit == "classic" and "rsgdf_tail_ke_cutoff" not in kw:
        # #257 made the driver's ``rsgdf_ke_cutoff`` default ``None`` ("no
        # request": the SR/LR planner picks its own mesh, the classic route
        # resolves it to RSGDF_KE_CUTOFF_DEFAULT). The runner arm forwards that
        # ``None`` here, so the mirrored tail must be classified at the mesh
        # the classic source will actually use, exactly as the driver resolves
        # it, never at ``float(None)``.
        from ...periodic_k_gdf import RSGDF_KE_CUTOFF_DEFAULT

        ke = kw.get("rsgdf_ke_cutoff")
        ke = RSGDF_KE_CUTOFF_DEFAULT if ke is None else float(ke)
        kw["rsgdf_tail_ke_cutoff"] = ccm_neutral_tail_ke_cutoff(ccm, ke_cutoff=ke)
    if fit == "classic" and int(unit.dim) == 3:
        # The real-Γ fold builds its classic pairs on the historical index
        # ball, so this side keeps that support too (#466): the drivers'
        # physical-pair default would otherwise split the Theorem-1 operator.
        kw.setdefault("classic_support_policy", "historical_index_ball")
    if symmetry is not None and symmetry is not False and fit != "auto":
        raise ValueError(
            f"{who}: symmetry=True (the CCM k-pair star reduction) cannot be "
            f"combined with fit_source={fit!r}: the multi-k driver refuses a "
            "private fit adapter together with an explicit fit source, and the "
            "reduction is approximate on 3-D cells (+6.7e-5 Ha/cell on "
            "LiH/STO-3G (2,2,1)). Pass symmetry=False (the default).")
    return fit


def _executed_fit(res):
    """``(source, executed tail, descriptor)`` read from a multi-k driver result.

    The multi-k result records the executed tail on its fit-source descriptor
    (``executed_tail_reciprocal_cutoff_ha``), not on ``rsgdf_tail_ke_cutoff``,
    which only the Γ fast path sets; a matched-tail assertion between the two
    producers must read it from there.
    """
    rec = getattr(res, "executed_fit_source", None)
    if rec is not None:
        return (str(getattr(rec, "source", "") or ""),
                getattr(rec, "executed_tail_reciprocal_cutoff_ha", None), rec)
    req = getattr(res, "rsgdf_fit_request", None)
    return (str(getattr(req, "source", "") or ""),
            getattr(res, "rsgdf_tail_ke_cutoff", None), None)


def _bloch_neutral_source(ccm, res, *, functional, active, driver_kwargs):
    """The neutral-source record of a multi-k driver result (#519, RG-2).

    Read the executed fit, never the request: the auxiliary basis, base
    cutoff, tail and cell-list cutoff the engine attested in the record it
    attached to the driver result. ``None`` when the driver recorded no fit
    source, so the wrapper never claims an identity it cannot read.
    """
    from .source import has_recordable_system, neutral_torus_source

    fit_source, tail_executed, fit_record = _executed_fit(res)
    if fit_record is None or not has_recordable_system(ccm):
        return None
    from vibeqc.periodic.exchange_convention import BVK_EWALD

    support = dict(getattr(fit_record, "numerical_support", ()) or ())
    fit_lattice = dict(support.get("fit_lattice", ()) or ())
    options = driver_kwargs.get("options")
    return neutral_torus_source(
        ccm,
        fit_engine=getattr(fit_record, "engine", ""),
        fit_source=fit_source,
        aux_basis=getattr(fit_record, "auxiliary_basis", None),
        ke_cutoff=getattr(fit_record, "base_reciprocal_cutoff_ha", None),
        tail_ke_cutoff=tail_executed,
        lattice_cutoff_bohr=fit_lattice.get("cutoff_bohr"),
        exchange_q0=BVK_EWALD if active else "",
        exchange_q0_applicability="active" if active else "inactive",
        functional=functional,
        xc_lattice_cutoff_bohr=(
            None if functional is None
            else float(getattr(res, "xc_lattice_cutoff_bohr", 0.0))),
        grid_image_radius_bohr=(
            None if functional is None
            else float(getattr(options, "becke_image_radius_bohr", 10.0))),
    )


def _ccm_gdf(ccm, aux_basis, functional, *, symmetry=False,
             who="run_ccm_rhf_gdf", allow_vacuum_padding=False,
             **driver_kwargs):
    """Neutral fitted-torus Bloch control on the ``nrep`` k-mesh.

    This and the real-Γ control are Fourier-related evaluations of the same
    specified block-circulant neutral Hamiltonian. The multi-k route is robust
    where Γ on the supercell would lose overlap positive-definiteness for dense
    clusters. This representation identity does not make the route an
    evaluation of the union-and-weight Γ-CCM construction. ``driver_kwargs``
    are forwarded verbatim to :func:`~vibeqc.run_krhf_periodic_gdf` /
    :func:`~vibeqc.run_krks_periodic_gdf` (e.g. ``k_exchange="cosx"``,
    ``use_compcell``, ``gdf_method``, grid options)."""
    from vibeqc import (make_basis, monkhorst_pack, run_krhf_periodic_gdf,
                        run_krks_periodic_gdf)
    from .direct import _reject_vacuum_padded_direct

    # `who` names the PUBLIC entry point the caller actually used, not this
    # shared helper: run_ccm_rhf_gdf and run_ccm_rks_gdf both land here, and
    # hardcoding one of them made the other's refusal name a function the
    # caller never called (IID 498).
    _padded_axes = _reject_vacuum_padded_direct(
        ccm, who=who, allow_vacuum_padding=allow_vacuum_padding)
    unit = ccm.unit_system
    basis = make_basis(unit.unit_cell_molecule(), ccm.basis_name)
    kmesh = monkhorst_pack(unit, list(ccm.nrep))
    kw = _gdf_guess_kwargs(driver_kwargs, is_ks=functional is not None)
    if aux_basis is not None:
        kw["aux_basis"] = aux_basis
    _pin_classic_fit_source(ccm, kw, symmetry=symmetry, who=who)
    pair_stats = {"builds": 0, "total": 0, "factor": 1.0}
    # Fully 3-D replica meshes use the generic shared-q full-build path. This
    # is now purely a cost decision -- the star plan's relations are exact on
    # the truncated build since the vibeqc#337 fix -- but the generic builder's
    # n_k shared-q batches still beat n_k**2 individual pair calls, so routing
    # 3-D meshes through this cache would cost more than the star saves.
    use_pair_symmetry = (
        symmetry is not None
        and symmetry is not False
        and int(np.prod(ccm.nrep)) > 1
        and not all(int(n) > 1 for n in ccm.nrep)
        and str(kw.get("gdf_method", "rsgdf")).lower() == "rsgdf"
    )
    if use_pair_symmetry:
        kw["_lpq_cache_builder"] = _ccm_gdf_symmetry_cache_builder(
            ccm, symmetry, pair_stats
        )
    if functional is None:
        res = run_krhf_periodic_gdf(unit, basis, kmesh, **kw)
    else:
        res = run_krks_periodic_gdf(unit, basis, kmesh, functional=functional, **kw)
    if (int(unit.charge) == 0 and bool(res.converged)
            and float(res.energy) > 0.0):
        raise ValueError(
            "neutral CCM GDF control converged to a positive total energy "
            f"({float(res.energy):.6f} Ha per cell): unphysical for a neutral "
            "cell, and the signature of a vacuum-padded or "
            "gauge-inconsistent regime (IID 291). Refusing to report this "
            "as a successful result."
        )
    n_unit = ccm.n_atoms // ccm.n_cells
    # q=0 exchange convention as a result FIELD (theory-chat ask 2026-07-16): read
    # from the driver's computed HF-exchange energy -- nonzero iff the ewald K-shift
    # (apply_exxdiv_ewald_to_K) was applied. RHF always has exact exchange; a hybrid
    # KS run has e_hf_exchange != 0; pure DFT has none (convention inactive).
    from vibeqc.periodic.exchange_convention import BVK_EWALD
    e_hf_ex = float(getattr(res, "e_hf_exchange", 0.0) or 0.0)
    active = functional is None or abs(e_hf_ex) > 1e-12
    if pair_stats["total"] == 0:
        # No CCM star plan ran, so every fit pair was a full build. Read the
        # count from the driver result, never re-derive it: a local copy of
        # the driver's use_compcell promotion rules went stale and reported
        # 0 pairs for a pure-DFT dim=3 run that built n_k of them.
        n_pairs = int(res.gdf_cderi_pairs)
        pair_stats.update(builds=n_pairs, total=n_pairs, factor=1.0)
    fit_source, tail_executed, fit_record = _executed_fit(res)
    neutral_source = _bloch_neutral_source(
        ccm, res, functional=functional, active=active,
        driver_kwargs=driver_kwargs)
    return CCMGDFResult(
        converged=bool(res.converged), energy=float(res.energy),
        energy_per_atom=float(res.energy) / n_unit,
        n_iter=int(getattr(res, "n_iter", 0)), raw=res,
        exchange_q0=BVK_EWALD if active else "",
        exchange_q0_applicability="active" if active else "inactive",
        backend=str(getattr(res, "backend", "") or ""),
        gdf_pair_builds=int(pair_stats["builds"]),
        gdf_pair_total=int(pair_stats["total"]),
        gdf_pair_reduction_factor=float(pair_stats["factor"]),
        rsgdf_tail_ke_cutoff=tail_executed,
        fit_source=fit_source, executed_fit_source=fit_record,
        neutral_source=neutral_source,
        vacuum_padding_allowed=bool(allow_vacuum_padding),
        vacuum_padded_axes=tuple(_padded_axes or ()),
        scf_trace=tuple(getattr(res, "scf_trace", ()) or ()))


def _validate_gdf_driver_conv_tol_grad(driver_kwargs, *, who):
    """Fail closed on invalid commutator tolerances before GDF setup."""
    if "conv_tol_grad" in driver_kwargs:
        _validate_conv_tol_grad(driver_kwargs["conv_tol_grad"], who=who)
    options = driver_kwargs.get("options")
    if options is not None and hasattr(options, "conv_tol_grad"):
        _validate_conv_tol_grad(options.conv_tol_grad, who=who)


def _reject_external_xc_neutral_bloch(functional, *, spin, who):
    """Keep the neutral-Bloch control on its validated libxc envelope."""
    if functional is None:
        return

    from ..._vibeqc_core import Functional

    func = Functional(str(functional), int(spin))
    if bool(getattr(func, "is_external", False)):
        raise NotImplementedError(
            f"{who}: full-grid external XC is not available through the "
            "neutral-Bloch CCM control. Its unit-cell atom-block provider "
            "partition has not been proven representation-invariant against "
            "the real-Gamma supercell grouping. Use the real-Gamma control "
            "or the periodic GDF driver directly."
        )


def run_ccm_rhf_gdf(ccm, *, aux_basis=None, symmetry=False,
                    allow_vacuum_padding=False, **driver_kwargs):
    """HF neutral fitted-torus control in the multi-k Bloch representation.

    The unit-cell ``nrep`` mesh and the real-Γ control are related exactly by
    the finite Fourier transform once the same block-circulant neutral
    Hamiltonian is specified. This is not union-and-weight Γ-CCM construction
    evidence. Returns a :class:`CCMGDFResult` (``.energy_per_atom``).

    ``driver_kwargs`` reach :func:`~vibeqc.run_krhf_periodic_gdf` unchanged. For
    **RIJCOSX** exchange on this control (RI-J + seminumerical COSX-K
    instead of exact GDF-K) pass ``k_exchange="cosx", use_compcell=True`` (COSX
    needs the compensated-cell multi-k grid); ``k_exchange="gdf"`` (default) is
    exact exchange.

    **The fit (#378).** On a 3-D cell with ``gdf_method="rsgdf"`` this producer
    pins ``fit_source="classic"`` and passes the real-Γ fold's tail
    (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_tail_ke_cutoff`), so that
    it evaluates the same fitted Hamiltonian as :func:`run_ccm_rhf_direct`
    (Paper 1 Theorem 1; identity at 1e-12 Ha/cell,
    ``tests/test_ccm_theorem1_3d.py``). The driver's own default, the SR/LR
    source, is reachable deliberately with ``fit_source="sr"``; it is a
    different fitted operator (1.4e-4 Ha/cell on LiH/STO-3G) and slower.
    The executed source and tail are recorded on the result
    (``fit_source``, ``executed_fit_source``, ``rsgdf_tail_ke_cutoff``).

    ``symmetry`` (default ``False``) opts into the CCM k-pair star reduction
    on lower-dimensional replica meshes; it is approximate on 3-D cells
    (+6.7e-5 Ha/cell on LiH/STO-3G (2,2,1), the residual ROUTES 10.5 recorded)
    and cannot be combined with an explicit fit source, so it is refused
    together with the classic pin.

    ``retain_fit_cache=True`` (a driver keyword) keeps the multi-k per-pair
    fit on the result (:attr:`CCMGDFResult.fit_cache`) so the real-Γ sibling
    can fold it instead of refitting when both producers run in one process
    (``run_ccm_rhf_direct(ccm, fit_cache=g.fit_cache)``); it is memory the
    SCF would otherwise free, hence opt-in.

    ``allow_vacuum_padding=True`` runs a cell measured as vacuum-padded
    (IID 291) as a warned, recorded vacuum-parameter study (ruling R6); see
    :func:`~vibeqc.periodic.ccm.direct._reject_vacuum_padded_direct`."""
    _validate_gdf_driver_conv_tol_grad(
        driver_kwargs, who="run_ccm_rhf_gdf"
    )
    _warn_experimental()
    return _ccm_gdf(
        ccm, aux_basis, None, symmetry=symmetry,
        who="run_ccm_rhf_gdf", allow_vacuum_padding=allow_vacuum_padding,
        **driver_kwargs
    )


def run_ccm_uhf_gdf(ccm, *, aux_basis=None, allow_vacuum_padding=False,
                    **driver_kwargs):
    """Open-shell HF neutral fitted-torus control via multi-k GDF.

    The spin-polarized sibling of :func:`run_ccm_rhf_gdf`: multi-k KUHF on the
    unit cell with the nrep k-mesh
    (:func:`~vibeqc.run_kuhf_periodic_gdf`). ``driver_kwargs`` reach the
    multi-k driver unchanged. Returns a :class:`CCMGDFResult`.

    .. warning::

       **Spin bookkeeping is per unit cell** in the multi-k driver: the unit
       cell's charge/multiplicity is what the KUHF reference sees, replicated
       over the mesh. The CCM supercell's multiplicity (parity-derived for the
       cluster) matches that convention only at ``nrep = (1,1,1)`` or when the
       supercell spin equals the unit-cell spin times ``N_c``. Cross-route
       open-shell comparisons against :func:`run_ccm_uhf_direct` /
       :func:`run_ccm_uks_direct` (whose spin counts come from the *supercell*
       multiplicity) are convention-matched only at ``(1,1,1)``; multi-cell
       open-shell parity needs an explicit spin-state decision first.
    """
    _validate_gdf_driver_conv_tol_grad(
        driver_kwargs, who="run_ccm_uhf_gdf"
    )
    _warn_experimental()
    from vibeqc import make_basis, monkhorst_pack, run_kuhf_periodic_gdf
    from .direct import _reject_vacuum_padded_direct

    _padded_axes = _reject_vacuum_padded_direct(
        ccm, who="run_ccm_uhf_gdf", allow_vacuum_padding=allow_vacuum_padding)
    unit = ccm.unit_system
    basis = make_basis(unit.unit_cell_molecule(), ccm.basis_name)
    kmesh = monkhorst_pack(unit, list(ccm.nrep))
    kw = _gdf_guess_kwargs(driver_kwargs, is_ks=False)
    if aux_basis is not None:
        kw["aux_basis"] = aux_basis
    _pin_classic_fit_source(ccm, kw, symmetry=False, who="run_ccm_uhf_gdf")
    res = run_kuhf_periodic_gdf(unit, basis, kmesh, **kw)
    if (int(unit.charge) == 0 and bool(res.converged)
            and float(res.energy) > 0.0):
        raise ValueError(
            "neutral CCM GDF control converged to a positive total energy "
            f"({float(res.energy):.6f} Ha per cell): unphysical for a neutral "
            "cell, and the signature of a vacuum-padded or "
            "gauge-inconsistent regime (IID 291). Refusing to report this "
            "as a successful result."
        )
    n_unit = ccm.n_atoms // ccm.n_cells
    # UHF always carries full-range exact exchange, and the multi-k KUHF
    # driver applies exxdiv='ewald' unconditionally -- record the
    # convention like _ccm_gdf does (field added 2026-08-19; the wrapper
    # predated the theory-chat ask that introduced the record).
    from vibeqc.periodic.exchange_convention import BVK_EWALD

    return CCMGDFResult(
        converged=bool(res.converged), energy=float(res.energy),
        energy_per_atom=float(res.energy) / n_unit,
        n_iter=int(getattr(res, "n_iter", 0)), raw=res,
        exchange_q0=BVK_EWALD, exchange_q0_applicability="active",
        backend=str(getattr(res, "backend", "") or ""),
        rsgdf_tail_ke_cutoff=_executed_fit(res)[1],
        fit_source=_executed_fit(res)[0], executed_fit_source=_executed_fit(res)[2],
        neutral_source=_bloch_neutral_source(
            ccm, res, functional=None, active=True, driver_kwargs=driver_kwargs),
        vacuum_padding_allowed=bool(allow_vacuum_padding),
        vacuum_padded_axes=tuple(_padded_axes or ()),
        scf_trace=tuple(getattr(res, "scf_trace", ()) or ()))


def run_ccm_rks_gdf(
    ccm, functional="pbe", *, aux_basis=None, symmetry=False,
    allow_vacuum_padding=False, **driver_kwargs
):
    """KS neutral fitted-torus control in the multi-k Bloch representation.

    As :func:`run_ccm_rhf_gdf` with a libxc ``functional`` (pure or hybrid).
    Hybrids use exact GDF exchange by default; pass ``k_exchange="cosx",
    use_compcell=True`` for **RIJCOSX** hybrid exchange. ``symmetry`` has the
    same RSGDF pair-star semantics as :func:`run_ccm_rhf_gdf`. Returns a
    :class:`CCMGDFResult`.

    **The fit (#378).** Like :func:`run_ccm_rhf_gdf` this producer pins
    ``fit_source="classic"`` with the real-Γ fold's tail on 3-D ``rsgdf`` cells. Its XC
    density-image sum reaches the lattice cutoff (the native kernel's default, #265), and so
    does :func:`run_ccm_rks_direct`'s, so the two producers evaluate one XC term; the identity
    holds at 1e-12 for pure and hybrid functionals.
    """
    _reject_external_xc_neutral_bloch(
        functional, spin=1, who="run_ccm_rks_gdf"
    )
    _validate_gdf_driver_conv_tol_grad(
        driver_kwargs, who="run_ccm_rks_gdf"
    )
    _warn_experimental()
    return _ccm_gdf(
        ccm, aux_basis, functional, symmetry=symmetry,
        who="run_ccm_rks_gdf", allow_vacuum_padding=allow_vacuum_padding,
        **driver_kwargs
    )


def run_ccm_uks_gdf(ccm, functional="pbe", *, aux_basis=None,
                    allow_vacuum_padding=False, **driver_kwargs):
    """Open-shell KS neutral fitted-torus control via multi-k GDF.

    The spin-polarized sibling of :func:`run_ccm_rks_gdf`: multi-k KUKS on the
    unit cell with the nrep k-mesh
    (:func:`~vibeqc.run_kuks_periodic_gdf`). Returns a :class:`CCMGDFResult`.

    .. warning::

       **Spin bookkeeping is per unit cell** in the multi-k driver: the unit
       cell's charge/multiplicity is what the KUKS reference sees, replicated
       over the mesh. The CCM supercell's multiplicity (parity-derived for the
       cluster) matches that convention only at ``nrep = (1,1,1)`` or when the
       supercell spin equals the unit-cell spin times ``N_c``. Cross-route
       open-shell comparisons against :func:`run_ccm_uks_direct` /
       :func:`run_ccm_uhf_direct` (whose spin counts come from the *supercell*
       multiplicity) are convention-matched only at ``(1,1,1)``; multi-cell
       open-shell parity needs an explicit spin-state decision first.
    """
    _reject_external_xc_neutral_bloch(
        functional, spin=2, who="run_ccm_uks_gdf"
    )
    _validate_gdf_driver_conv_tol_grad(
        driver_kwargs, who="run_ccm_uks_gdf"
    )
    _warn_experimental()
    from vibeqc import make_basis, monkhorst_pack, run_kuks_periodic_gdf
    from .direct import _reject_vacuum_padded_direct

    _padded_axes = _reject_vacuum_padded_direct(
        ccm, who="run_ccm_uks_gdf", allow_vacuum_padding=allow_vacuum_padding)
    unit = ccm.unit_system
    basis = make_basis(unit.unit_cell_molecule(), ccm.basis_name)
    kmesh = monkhorst_pack(unit, list(ccm.nrep))
    kw = _gdf_guess_kwargs(driver_kwargs, is_ks=True)
    if aux_basis is not None:
        kw["aux_basis"] = aux_basis
    _pin_classic_fit_source(ccm, kw, symmetry=False, who="run_ccm_uks_gdf")
    res = run_kuks_periodic_gdf(unit, basis, kmesh, functional=functional, **kw)
    if (int(unit.charge) == 0 and bool(res.converged)
            and float(res.energy) > 0.0):
        raise ValueError(
            "neutral CCM GDF control converged to a positive total energy "
            f"({float(res.energy):.6f} Ha per cell): unphysical for a neutral "
            "cell, and the signature of a vacuum-padded or "
            "gauge-inconsistent regime (IID 291). Refusing to report this "
            "as a successful result."
        )
    n_unit = ccm.n_atoms // ccm.n_cells
    # Convention record per _ccm_gdf: active iff exact exchange present
    # (hybrid functional -> nonzero e_hf_exchange); pure KS carries "".
    from vibeqc.periodic.exchange_convention import BVK_EWALD

    _e_hf_ex = float(getattr(res, "e_hf_exchange", 0.0) or 0.0)
    _active = abs(_e_hf_ex) > 1e-12
    return CCMGDFResult(
        exchange_q0=BVK_EWALD if _active else "",
        exchange_q0_applicability="active" if _active else "inactive",
        converged=bool(res.converged), energy=float(res.energy),
        energy_per_atom=float(res.energy) / n_unit,
        n_iter=int(getattr(res, "n_iter", 0)), raw=res,
        backend=str(getattr(res, "backend", "") or ""),
        rsgdf_tail_ke_cutoff=_executed_fit(res)[1],
        fit_source=_executed_fit(res)[0], executed_fit_source=_executed_fit(res)[2],
        neutral_source=_bloch_neutral_source(
            ccm, res, functional=functional, active=_active,
            driver_kwargs=driver_kwargs),
        vacuum_padding_allowed=bool(allow_vacuum_padding),
        vacuum_padded_axes=tuple(_padded_axes or ()),
        scf_trace=tuple(getattr(res, "scf_trace", ()) or ()))


def run_ccm_rhf_ri_neutral(ccm, *, initial_guess: object = "AUTO", cderi=None, ke_cutoff=200.0, aux_basis=None,
                           symmetry=None,
                           max_iter=128, conv_tol=1e-9, conv_tol_grad=1e-6,
                           diis_dim=8, lindep_tol=1e-7,
                           allow_vacuum_padding=False):
    """Closed-shell HF on the **neutral fitted-torus Hamiltonian**, built
    *lean* from the cderi ``L``.

    Unlike :func:`run_ccm_rhf_gdf` (multi-k GDF on the unit cell -> *per-k* KRHF
    MOs), this drives the SCF on the **cluster supercell at Γ**, assembling J and K
    directly from the neutral cderi via :func:`ccm_ri_j_neutral` /
    :func:`ccm_ri_k_neutral` -- so the dense ``n_ref_ao**4`` neutral tensor
    ``g_eff = Σ_P L⊗L`` is **never formed**. The returned MOs live in the same
    supercell-Γ AO basis as ``L``, so this is the reference that **pairs with the
    RI correlation routes** -- ``run_ccm_mp2(ccm, scf, cderi=L)`` /
    ``run_ccm_ccsd(..., cderi=L)`` -- giving neutral canonical correlation that
    reaches moderate 3-D end-to-end (only ``L``, GDF-feasible, is built; no dense
    ``g``, no dense AO four-center).

    ``cderi`` reuses an already-built ``L`` (else it is built via
    :func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi`). Energy equals
    ``run_ccm_rhf(eri=ccm_eri_neutral(ccm))`` to machine ε (``J``/``K`` from ``L``
    reproduce those from ``g_eff = Σ_P L⊗L`` exactly). Passing ``aux_basis``
    together with a prebuilt ``cderi`` raises :class:`ValueError`: the cderi
    already fixes the fitting basis. Returns a ``CCMSCFResult``.

    ``conv_tol`` gates the energy criterion only; ``conv_tol_grad`` is the
    independent DIIS-commutator residual bound (default ``1e-6``, the historical
    gate).

    ``symmetry`` is forwarded to the ``ccm_neutral_cderi`` builder's ``symmetry``
    (space-group pair reduction; ``None``/``False`` off (default), ``True``/"auto"
    to analyze, or a pre-computed :class:`~vibeqc.periodic.ccm.symmetry.CCMSymmetry`)
    -- the same knob :func:`run_ccm_rhf_direct` exposes as ``cderi_symmetry``. The
    reduced fit equals the un-reduced build elementwise (``< 1e-9``; see
    ``tests/test_ccm_symmetry.py``), so the SCF energy is symmetry-invariant.
    Ignored when a prebuilt ``cderi`` is passed.

    .. note::

       This is the **supercell-Γ neutral** (``ccm_eri_neutral``'s ``q=0``
       convention, which drops ``G+q=0``). Its *absolute* energy differs from the
       multi-k :func:`run_ccm_rhf_gdf` control (the KRHF-matching reference) by a
       **Madelung-scale exchange-q0 (exxdiv) term** -- the two share the Coulomb
       operator and differ only in the ``q=0`` exchange self-energy convention.
       Measured 0.239 Ha/atom on H₂/STO-3G ``[6,15,15]`` (2,1,1), ``ke_cutoff``-
       converged, ≈ ξ = ``madelung_constant_for_cell`` (0.2003 Ha). This is the
       exchange-q0 *residual freedom* (D1), not a self-image leak; but it means the
       lean SCF (and the RI/DLPNO correlation built on the same neutral kernel)
       carries the **strict-zero-mode** exchange-q0 convention, **not** the GDF's
       ``ewald`` exxdiv. For correlation it shifts the HF gap. Which convention the
       correlation reference *should* use is resolved by the D2 vq KRHF/KMP2 anchor.

       For a supercell-Γ SCF **in the GDF's ewald convention** use
       :func:`~vibeqc.periodic.ccm.direct.run_ccm_rhf_direct` -- it adds the
       derived exchange-q0 seam ``K -> K + ξ_N·S·D·S`` (plus the folded
       Ewald-gauge one-electron/nuclear terms) and reproduces
       :func:`run_ccm_rhf_gdf` to ≤1e-8 Ha/cell in one real eigenproblem.
    """
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_rhf_ri_neutral"
    )
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_rhf_ri_neutral',
    )
    _warn_experimental()
    from .direct import (
        _reject_low_dimensional_direct,
        _reject_vacuum_padded_direct,
        _stamp_vacuum_padding,
    )

    dim = int(ccm.unit_system.dim)
    if dim != 3:
        _reject_low_dimensional_direct(dim, who="run_ccm_rhf_ri_neutral")
    _padded_axes = _reject_vacuum_padded_direct(
        ccm, who="run_ccm_rhf_ri_neutral",
        allow_vacuum_padding=allow_vacuum_padding)
    if cderi is not None and aux_basis is not None:
        raise ValueError(
            "aux_basis is ignored when a prebuilt cderi is supplied: "
            "the cderi already fixes the fitting basis. Omit aux_basis, "
            "or drop cderi and let the builder construct it at aux_basis."
        )
    L = np.asarray(
        cderi if cderi is not None
        else ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff, aux_basis=aux_basis,
                               symmetry=symmetry),
        dtype=float)
    S, h, e_nn = _ccm_common(ccm)
    res = _rhf_loop(
        ccm, S, h, e_nn,
        lambda D: ccm_ri_j_neutral(L, D),
        lambda D: ccm_ri_k_neutral(L, D),
        max_iter=max_iter, conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
        diis_dim=diis_dim, lindep_tol=lindep_tol)
    res.backend = "ccm-neutral-ri-rhf"    # result-backend identity (IID 344)
    res.cderi = L                          # one build per job (2026-09-24)
    res.cderi_info = {"build": "caller" if cderi is not None else "supercell",
                      "shape": tuple(int(n) for n in np.shape(L))}
    _stamp_vacuum_padding(res, allow_vacuum_padding, _padded_axes)
    return _with_ccm_guess(res, guess_selection)


# ----- Route 2: explicit WSSC three-center RI-J / RIJCOSX (research) ----------

def _aux_cell_to_cols(ccm, pad, aux):
    """home aux-AO -> padded aux-AO column at each WSSC cell (mirrors the orbital
    map in build_padded_cluster, for the auxiliary basis)."""
    aux_of_atom, _ = _ao_ranges(aux)
    aux_atom, local = [], []
    for A in range(ccm.n_atoms):
        cols = aux_of_atom[pad.atom_g_to_pad[(A, (0, 0, 0))]]
        for k in range(len(cols)):
            aux_atom.append(A); local.append(k)
    aux_atom = np.asarray(aux_atom)
    n_aux = aux_atom.size
    home = np.array([aux_of_atom[pad.atom_g_to_pad[(aux_atom[i], (0, 0, 0))]][local[i]]
                     for i in range(n_aux)], dtype=int)
    cell_cols = {}
    for g in wssc_cells(ccm):
        cell_cols[g] = np.array(
            [aux_of_atom[pad.atom_g_to_pad[(aux_atom[i], g)]][local[i]]
             for i in range(n_aux)], dtype=int)
    return home, aux_atom, cell_cols


def ccm_ri_tensors(ccm, aux_basis="def2-universal-jkfit"):
    """CCM RI tensors ``B[mu,ν,P]`` and ``Vinv = (V^CCM)⁻¹`` (eq-13 union 3c)."""
    from vibeqc import BasisSet, compute_2c_eri, compute_3c_eri
    from vibeqc._vibeqc_core import Atom, Molecule

    pad = build_padded_cluster(ccm, wssc_cells(ccm))
    n_el = int(sum(int(z) for z in pad.pad_Z))
    pad_mol = Molecule(
        [Atom(int(z), list(map(float, p)))
         for z, p in zip(pad.pad_Z, pad.pad_positions)], 0, 1 if n_el % 2 == 0 else 2)
    aux = BasisSet(pad_mol, aux_basis)
    three = np.asarray(compute_3c_eri(pad.basis, aux), float)
    two = np.asarray(compute_2c_eri(aux), float)

    weights = ccm.cell_weight_matrices()
    ao = ccm.ao_atom
    n = ccm.nbf
    home_orb = np.arange(n)
    aux_home, aux_atom, aux_cols = _aux_cell_to_cols(ccm, pad, aux)
    n_aux = aux_home.size
    w_home = weights[(0, 0, 0)][ao[:, None], ao[None, :]]

    B = np.zeros((n, n, n_aux))
    for g, w_atom in weights.items():
        cols_P = aux_cols.get(g)
        if cols_P is None:
            continue
        w_oa = w_atom[ao[:, None], aux_atom[None, :]]
        bridge = 0.5 * (w_oa[:, None, :] + w_oa[None, :, :])
        if not np.any(bridge):
            continue
        B += (w_home[:, :, None] * bridge) * three[np.ix_(cols_P, home_orb, home_orb)].transpose(1, 2, 0)
    B = 0.5 * (B + B.transpose(1, 0, 2))

    V = np.zeros((n_aux, n_aux))
    for g, w_atom in weights.items():
        cols_Q = aux_cols.get(g)
        if cols_Q is None:
            continue
        V += w_atom[aux_atom[:, None], aux_atom[None, :]] * two[np.ix_(aux_home, cols_Q)]
    V = 0.5 * (V + V.T)
    return B, np.linalg.pinv(V, rcond=1e-10)


def _ri_j(B, Vinv, D):
    return np.einsum("mnP,P->mn", B,
                     Vinv @ np.einsum("rsQ,rs->Q", B, D, optimize=True), optimize=True)


def _ccm_common(ccm):
    S = ccm_overlap(ccm)
    h, _, _ = ccm_hcore(ccm)
    e_nn = ccm_nuclear_repulsion(ccm)
    # Linear-dependence screening happens in the shared SCF loops via
    # ``_orthonormaliser``; no refusal here.
    return S, h, e_nn


def _rhf_loop(ccm, S, h, e_nn, j_of_D, k_of_D, *, max_iter, conv_tol,
              conv_tol_grad=1e-6, diis_dim, lindep_tol, initial_density=None,
              builder=None):
    """Shared closed-shell SCF loop over injected J/K operators.

    ``builder`` (optional, #472 / #519 RG-3) evaluates one frozen
    :class:`~vibeqc.fock_state.FockBuild` per density; the loop then takes
    ``F[D]`` and ``E[D]`` from the build and runs DIIS on a copy of the
    physical Fock. The terminal build, the last state evaluated, lands on
    ``result.fock_build``. Without a builder the loop assembles
    ``F = h + J - K/2`` and ``E = 1/2 Tr[D(h+F)] + E_nn`` itself, as before.
    Either way a run that stops at ``max_iter`` reports the last evaluated
    state, never a DIIS trial Fock with the density it would have produced
    next.
    """
    conv_tol_grad = _validate_conv_tol_grad(conv_tol_grad, who="_rhf_loop")
    n_elec = ccm.supercell.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError("closed-shell driver needs an even-electron cluster.")
    n_occ = n_elec // 2
    X = _orthonormaliser(S, lindep_tol)
    _require_retained_occ(X, n_occ, who="_rhf_loop", lindep_tol=lindep_tol)

    def diag(F):
        eps, Cp = np.linalg.eigh(X.T @ F @ X)
        return eps, X @ Cp

    eps, C = diag(h)
    D = (2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)
         if initial_density is None
         else _ccm_read_density(initial_density, S, n_elec))
    if initial_density is not None:
        # No orbital spectrum belongs to the supplied density yet.
        eps = np.full_like(eps, np.nan)
    diis_F, diis_e, e_last, conv, it = [], [], 0.0, False, 0
    trace = []
    build = None
    for it in range(1, max_iter + 1):
        if builder is not None:
            build = builder(D)
            F = np.array(build.channels[0].operator.matrix, dtype=float)
            e_tot = float(build.energy.total)
        else:
            F = h + j_of_D(D) - 0.5 * k_of_D(D)
            F = 0.5 * (F + F.T)
            e_tot = 0.5 * np.sum(D * (h + F)) + e_nn
        err = X.T @ (F @ D @ S - S @ D @ F) @ X
        if len(diis_F) == diis_dim:
            diis_F.pop(0); diis_e.pop(0)
        diis_F.append(F); diis_e.append(err)
        # Diagnostic only (#493): recorded before the exit test so the final
        # cycle appears in the trace whether or not it converged.
        trace.append(_scf_trace_row(it, e_tot, e_last, err, eps, n_occ,
                                    len(diis_F)))
        # The energy, residual, density, and raw Fock above describe one SCF
        # state.  Return that fixed point before DIIS produces the *next* trial
        # Fock; otherwise a converged result mixes E/F from this state with D/C
        # from a subsequent extrapolated diagonalisation.
        if it > 1 and _commutator_exit_ok(e_tot - e_last, err, conv_tol, conv_tol_grad):
            conv = True; e_last = e_tot; break
        if it == max_iter:
            # Iteration cap: the state evaluated above is the reported one.
            e_last = e_tot
            break
        if len(diis_F) >= 2:
            F = _diis_extrapolate(diis_F, diis_e)
        eps, C = diag(F)
        D = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)
        e_last = e_tot
    if conv and int(ccm.unit_system.charge) == 0 and e_last > 0.0:
        raise ValueError(
            "closed-shell neutral CCM SCF converged to a positive total "
            f"energy ({e_last:.6f} Ha): unphysical for a neutral cell, and "
            "the signature of a vacuum-padded or gauge-inconsistent regime "
            "(IID 291). Refusing to report this as a successful result."
        )
    return CCMSCFResult(
        converged=conv, n_iter=it, energy=e_last, energy_per_atom=e_last / ccm.n_atoms,
        e_electronic=e_last - e_nn, e_nuclear=e_nn, mo_energies=eps, mo_coeffs=C,
        density=D, fock=F, overlap=S, hcore=h,
        idempotency_error=float(np.linalg.norm(D @ S @ D - 2.0 * D)),
        scf_trace=tuple(trace), fock_build=build)


def _rks_loop(ccm, S, h, e_nn, j_of_D, k_of_D, xc_of_D, alpha, functional, *,
              max_iter, conv_tol, conv_tol_grad=1e-6, diis_dim, lindep_tol, initial_density=None,
              builder=None):
    """Shared closed-shell KS-SCF loop over injected RI operators -- the KS
    sibling of :func:`_rhf_loop`, used by the direct-torus (real-Γ) KS routes.

    Assembles ``F = h + J(D) + V_xc(D) - (alpha/2) K(D)`` with ``alpha`` the
    functional's exact-exchange fraction (``k_of_D`` is only called when
    ``alpha != 0``; pass ``None`` for pure functionals) and
    ``xc_of_D(D) -> (E_xc, V_xc)`` the **supercell** XC energy + potential
    (CCM_THEORY.md Sec. 11.2: F^RKS = h + J - (a_x/2) K + V_xc). The RHF
    ``1/2 Tr[D(h+F)]`` energy identity does not hold once V_xc enters F, so
    the KS energy is assembled explicitly:

        E = Tr[D h] + 1/2 Tr[D J] - (alpha/4) Tr[D K] + E_xc + E_nn .

    Returns a :class:`~vibeqc.periodic.ccm.dft.CCMKSResult` (energies per
    supercell, like every supercell-Γ CCM driver).

    ``builder`` (optional, #472 / #519 RG-3): as in :func:`_rhf_loop`, one
    frozen :class:`~vibeqc.fock_state.FockBuild` per density supplies
    ``F[D]`` and the energy components; the terminal build lands on
    ``result.fock_build``.
    """
    conv_tol_grad = _validate_conv_tol_grad(conv_tol_grad, who="_rks_loop")
    n_elec = ccm.supercell.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError("closed-shell driver needs an even-electron cluster.")
    n_occ = n_elec // 2
    X = _orthonormaliser(S, lindep_tol)
    _require_retained_occ(X, n_occ, who="_rks_loop", lindep_tol=lindep_tol)

    def diag(F):
        eps, Cp = np.linalg.eigh(X.T @ F @ X)
        return eps, X @ Cp

    eps, C = diag(h)
    D = (2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)
         if initial_density is None
         else _ccm_read_density(initial_density, S, n_elec))
    diis_F, diis_e, e_last, conv, it = [], [], 0.0, False, 0
    e_xc = e_coul = e_k = 0.0
    F = np.asarray(h, dtype=float)
    build = None
    for it in range(1, max_iter + 1):
        if builder is not None:
            build = builder(D)
            F = np.array(build.channels[0].operator.matrix, dtype=float)
            parts = dict(build.energy.components)
            e_coul = float(parts.get("coulomb", 0.0))
            e_k = float(parts.get("exchange", 0.0))
            e_xc = float(parts.get("xc", 0.0))
            e_tot = float(build.energy.total)
        else:
            J = j_of_D(D)
            e_xc, V_xc = xc_of_D(D)
            F = h + J + V_xc
            e_coul = 0.5 * float(np.sum(D * J))
            e_k = 0.0
            if alpha != 0.0:
                K = k_of_D(D)
                F = F - 0.5 * alpha * K
                e_k = -0.25 * alpha * float(np.sum(D * K))
            F = 0.5 * (F + F.T)
            e_tot = float(np.sum(D * h)) + e_coul + e_k + e_xc + e_nn
        err = X.T @ (F @ D @ S - S @ D @ F) @ X
        # Everything above describes the accepted density ``D``.  Test that
        # state before DIIS constructs the next trial Fock: DIIS is an
        # accelerator, not a physical operator or a terminal result state.
        if it > 1 and _commutator_exit_ok(
                e_tot - e_last, err, conv_tol, conv_tol_grad):
            conv = True
            e_last = e_tot
            break
        if it == max_iter:
            e_last = e_tot
            break
        if len(diis_F) == diis_dim:
            diis_F.pop(0); diis_e.pop(0)
        diis_F.append(F); diis_e.append(err)
        trial_F = F
        if len(diis_F) >= 2:
            trial_F = _diis_extrapolate(diis_F, diis_e)
        eps, C = diag(trial_F)
        D = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)
        e_last = e_tot

    from .dft import CCMKSResult

    # The accepted density and every reported energy component were evaluated
    # with this physical F[D].  Canonicalise that operator for reporting; the
    # preceding DIIS trial must never leak into the result orbitals.
    eps, C = diag(F)
    if conv and int(ccm.unit_system.charge) == 0 and e_last > 0.0:
        raise ValueError(
            "closed-shell neutral CCM KS SCF converged to a positive total "
            f"energy ({e_last:.6f} Ha): unphysical for a neutral cell, and "
            "the signature of a vacuum-padded or gauge-inconsistent regime "
            "(IID 291). Refusing to report this as a successful result."
        )
    result = CCMKSResult(
        converged=conv, n_iter=it, energy=e_last,
        energy_per_atom=e_last / ccm.n_atoms, e_xc=e_xc, e_coulomb=e_coul,
        e_hf_exchange=e_k, functional=functional, mo_energies=eps, mo_coeffs=C,
        density=D, fock=F, overlap=S, open_shell=False, fock_build=build)
    # The real-Gamma runner adapter exposes the supercell one-electron
    # Hamiltonian alongside F[D].  CCMKSResult predates that optional consumer
    # field, so retain it on the route result without changing the shared
    # four-centre result schema.
    result.hcore = np.asarray(h)
    return result


def run_ccm_rhf_rij(ccm, aux_basis="def2-universal-jkfit", *, initial_guess: object = "AUTO", method="aiccm2026dev-a",
                    max_iter=128, conv_tol=1e-9, conv_tol_grad=1e-6, diis_dim=8,
                    lindep_tol=1e-7, ri=None):
    """Γ-CCM research approximation with explicit WSSC RI-J Coulomb and exact
    four-center exchange. Exact in the isolated limit; ~few-% periodically -- the
    bare-1/r union 3-center is **not RI-consistent** (the bare four-center is
    non-separable, Sec.13.7). For the RI-consistent three-center use the neutral
    cderi :func:`ccm_ri_tensors_neutral` (reproduces the four-center exactly); for
    a neutral fitted-torus **control energy** use :func:`run_ccm_rhf_gdf` (the
    GDF neutral gauge; not Γ-CCM construction evidence). Returns a
    ``CCMSCFResult``."""
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_rhf_rij"
    )
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_rhf_rij',
    )
    _warn_experimental()
    S, h, e_nn = _ccm_common(ccm)
    B, Vinv = ccm_ri_tensors(ccm, aux_basis) if ri is None else ri
    eri_k = _EXACT_K[method](ccm)
    res = _rhf_loop(ccm, S, h, e_nn, lambda D: _ri_j(B, Vinv, D),
                    lambda D: np.einsum("msrn,rs->mn", eri_k, D, optimize=True),
                    max_iter=max_iter, conv_tol=conv_tol,
                    conv_tol_grad=conv_tol_grad, diis_dim=diis_dim,
                    lindep_tol=lindep_tol)
    res.backend = f"ccm-rij-{method}-rhf"  # result-backend identity (IID 344)
    return _with_ccm_guess(res, guess_selection)


def run_ccm_rhf_rijcosx(ccm, aux_basis="def2-universal-jkfit", *, initial_guess: object = "AUTO", max_iter=128,
                        conv_tol=1e-9, conv_tol_grad=1e-6, diis_dim=8,
                        lindep_tol=1e-7, grid_options=None, ri=None):
    """Γ-CCM RIJCOSX research approximation: WSSC RI-J Coulomb plus
    chain-of-spheres (COSX) exchange (Neese 2009) on a supercell grid. Exchange
    is short-ranged (decays within the WSC), so the seminumerical K is a natural
    fit. Exact in the isolated limit (== molecular RIJCOSX); ~few-% periodically
    (RI-J caveat, as :func:`run_ccm_rhf_rij`). Returns a ``CCMSCFResult``."""
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_rhf_rijcosx"
    )
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_rhf_rijcosx',
    )
    _warn_experimental()
    from vibeqc import GridOptions, build_cosx_q, build_grid, compute_cosx_k

    S, h, e_nn = _ccm_common(ccm)
    B, Vinv = ccm_ri_tensors(ccm, aux_basis) if ri is None else ri
    grid = build_grid(ccm.supercell, grid_options or GridOptions())
    q = build_cosx_q(ccm.basis, grid)
    res = _rhf_loop(
        ccm, S, h, e_nn, lambda D: _ri_j(B, Vinv, D),
        lambda D: np.asarray(compute_cosx_k(ccm.basis, D, grid, q)),
        max_iter=max_iter, conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
        diis_dim=diis_dim, lindep_tol=lindep_tol)
    res.backend = "ccm-rijcosx-rhf"       # result-backend identity (IID 344)
    return _with_ccm_guess(res, guess_selection)
