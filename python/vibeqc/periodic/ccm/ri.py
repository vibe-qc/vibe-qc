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

from .scf import _ccm_initial_guess, _with_ccm_guess

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
    (HANDOVER_AICCM_DIRECT_TORUS.md, efficiency program). No intermediate
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
    """Rank-revealing factor ``X`` of a PSD density: ``D = X @ X.T``.

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
    #: (zero when no Lpq cache is used).
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
        options = type(source)()
        for name in dir(source):
            if not name.startswith("_"):
                value = getattr(source, name)
                if not callable(value):
                    setattr(options, name, value)
    options.initial_guess = requested
    kw["options"] = options
    return kw


def _ccm_gdf(ccm, aux_basis, functional, *, symmetry=True,
             who="run_ccm_rhf_gdf", **driver_kwargs):
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
    _reject_vacuum_padded_direct(ccm, who=who)
    unit = ccm.unit_system
    basis = make_basis(unit.unit_cell_molecule(), ccm.basis_name)
    kmesh = monkhorst_pack(unit, list(ccm.nrep))
    kw = _gdf_guess_kwargs(driver_kwargs, is_ks=functional is not None)
    if aux_basis is not None:
        kw["aux_basis"] = aux_basis
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
        n_k = int(np.prod(ccm.nrep))
        uses_lpq_cache = n_k > 1 and (
            bool(kw.get("use_compcell", False))
            or active
            or (functional is not None and int(unit.dim) != 3)
        )
        if uses_lpq_cache:
            need_k_pairs = (
                active and str(kw.get("k_exchange", "gdf")) == "gdf"
            )
            n_pairs = n_k * n_k if need_k_pairs else n_k
            pair_stats.update(builds=n_pairs, total=n_pairs, factor=1.0)
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
        rsgdf_tail_ke_cutoff=getattr(res, "rsgdf_tail_ke_cutoff", None))


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


def run_ccm_rhf_gdf(ccm, *, aux_basis=None, symmetry=True, **driver_kwargs):
    """HF neutral fitted-torus control in the multi-k Bloch representation.

    The unit-cell ``nrep`` mesh and the real-Γ control are related exactly by
    the finite Fourier transform once the same block-circulant neutral
    Hamiltonian is specified. This is not union-and-weight Γ-CCM construction
    evidence. Returns a :class:`CCMGDFResult` (``.energy_per_atom``).

    ``driver_kwargs`` reach :func:`~vibeqc.run_krhf_periodic_gdf` unchanged. For
    **RIJCOSX** exchange on this control (RI-J + seminumerical COSX-K
    instead of exact GDF-K) pass ``k_exchange="cosx", use_compcell=True`` (COSX
    needs the compensated-cell multi-k grid); ``k_exchange="gdf"`` (default) is
    exact exchange. ``symmetry=True`` (default) reuses the currently gated
    space-group and time-reversal Lpq star members on lower-dimensional
    replica meshes; the planner admits only cell-list-preserving ops, so those
    members are exact on the truncated build. Fully 3-D meshes use the
    unreduced shared-q path because its ``n_k`` batched builds are cheaper
    here than ``n_k**2`` per-pair ones, not because the relations are unsafe;
    pass ``symmetry=False`` to request the unreduced path explicitly."""
    _validate_gdf_driver_conv_tol_grad(
        driver_kwargs, who="run_ccm_rhf_gdf"
    )
    _warn_experimental()
    return _ccm_gdf(
        ccm, aux_basis, None, symmetry=symmetry,
        who="run_ccm_rhf_gdf", **driver_kwargs
    )


def run_ccm_uhf_gdf(ccm, *, aux_basis=None, **driver_kwargs):
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

    _reject_vacuum_padded_direct(ccm, who="run_ccm_uhf_gdf")
    unit = ccm.unit_system
    basis = make_basis(unit.unit_cell_molecule(), ccm.basis_name)
    kmesh = monkhorst_pack(unit, list(ccm.nrep))
    kw = _gdf_guess_kwargs(driver_kwargs, is_ks=False)
    if aux_basis is not None:
        kw["aux_basis"] = aux_basis
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
        backend=str(getattr(res, "backend", "") or ""))


def run_ccm_rks_gdf(
    ccm, functional="pbe", *, aux_basis=None, symmetry=True, **driver_kwargs
):
    """KS neutral fitted-torus control in the multi-k Bloch representation.

    As :func:`run_ccm_rhf_gdf` with a libxc ``functional`` (pure or hybrid).
    Hybrids use exact GDF exchange by default; pass ``k_exchange="cosx",
    use_compcell=True`` for **RIJCOSX** hybrid exchange. ``symmetry`` has the
    same RSGDF pair-star semantics as :func:`run_ccm_rhf_gdf`. Returns a
    :class:`CCMGDFResult`."""
    _reject_external_xc_neutral_bloch(
        functional, spin=1, who="run_ccm_rks_gdf"
    )
    _validate_gdf_driver_conv_tol_grad(
        driver_kwargs, who="run_ccm_rks_gdf"
    )
    _warn_experimental()
    return _ccm_gdf(
        ccm, aux_basis, functional, symmetry=symmetry,
        who="run_ccm_rks_gdf", **driver_kwargs
    )


def run_ccm_uks_gdf(ccm, functional="pbe", *, aux_basis=None, **driver_kwargs):
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

    _reject_vacuum_padded_direct(ccm, who="run_ccm_uks_gdf")
    unit = ccm.unit_system
    basis = make_basis(unit.unit_cell_molecule(), ccm.basis_name)
    kmesh = monkhorst_pack(unit, list(ccm.nrep))
    kw = _gdf_guess_kwargs(driver_kwargs, is_ks=True)
    if aux_basis is not None:
        kw["aux_basis"] = aux_basis
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
        backend=str(getattr(res, "backend", "") or ""))


def run_ccm_rhf_ri_neutral(ccm, *, initial_guess: object = "AUTO", cderi=None, ke_cutoff=200.0, aux_basis=None,
                           symmetry=None,
                           max_iter=128, conv_tol=1e-9, conv_tol_grad=1e-6,
                           diis_dim=8, lindep_tol=1e-7):
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
    )

    dim = int(ccm.unit_system.dim)
    if dim != 3:
        _reject_low_dimensional_direct(dim, who="run_ccm_rhf_ri_neutral")
    _reject_vacuum_padded_direct(ccm, who="run_ccm_rhf_ri_neutral")
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
              conv_tol_grad=1e-6, diis_dim, lindep_tol):
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
    D = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)
    diis_F, diis_e, e_last, conv, it = [], [], 0.0, False, 0
    trace = []
    for it in range(1, max_iter + 1):
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
        scf_trace=tuple(trace))


def _rks_loop(ccm, S, h, e_nn, j_of_D, k_of_D, xc_of_D, alpha, functional, *,
              max_iter, conv_tol, conv_tol_grad=1e-6, diis_dim, lindep_tol):
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
    D = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)
    diis_F, diis_e, e_last, conv, it = [], [], 0.0, False, 0
    e_xc = e_coul = e_k = 0.0
    F = np.asarray(h, dtype=float)
    for it in range(1, max_iter + 1):
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
        density=D, fock=F, overlap=S, open_shell=False)
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
