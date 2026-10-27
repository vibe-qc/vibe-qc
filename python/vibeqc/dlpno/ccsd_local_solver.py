"""Reduced-scaling local DLPNO-CCSD solver (M3c).

The genuine per-pair coupled-cluster solver that retires the O(N⁶)
correctness pilot (`dlpno.ccsd`). Amplitudes remain stored in their pair PNO
spaces. By default, each target-pair residual is contracted in Riplinger and
Neese's atom-based extended PAO domain and then projected back into the
target pair's PNO space. The historical ``residual_domain="pair"`` mode
projects coupled amplitudes first and is retained only as an explicit legacy
control. No full-system amplitude tensor is ever formed.

The extended-domain contraction is exact with respect to the full-space
oracle for the same occupied coupling set: every coupled pair's PNO space is
spanned by the PAOs of its own atom domain, that domain is a subset of the
extended domain by construction, and every internal virtual index of the CCSD
residual is carried by an amplitude, so contracting in the extended basis
reproduces the full-space result to ``lindep`` precision (each domain's PAO
metric is orthonormalised independently, dropping eigenvalues below
``lindep``, so a pair direction can leak into a dropped extended direction
only at that level). Pairs that share one coupling set and one extended atom
domain therefore share one contraction: the residual is evaluated once per
distinct (coupling set, extended domain) group in the compiled kernel and each
member pair reads its own block, while the basis, its (ab|cd) factor and
virtual Fock block are shared by every group with the same domain (#689). On
a compact molecule, where every domain is the whole molecule, that is one
canonical-shaped residual per iteration instead of one per pair.

Correctness is by construction: `cs_ccsd_residual` is covariant under
virtual-space rotations, so in the **full-domain limit** (every pair's
PNO space = the full virtual space) the projections are exact rotations
and this solver reproduces canonical closed-shell CCSD bit-for-bit -- the
parity gate in `tests/test_dlpno_ccsd_solver.py` (the M3c analogue of
the M1/M2 DLPNO-MP2 ratchet). With truncated PNO domains the cross-domain
contractions are the controlled DLPNO domain approximation. The retained
approximately 99.9% recovery observation at ``TCutPNO=1e-7`` belongs to the
explicit historical all-electron/full-PAO recipe in
``examples/molecular/benchmark-dlpno-ccsd-t.py``; it is not a claim for the
post-#140/#448 NormalPNO default, whose complete recipe also has nonzero
``TCutMKN`` and frozen chemical cores.

`cs_ccsd_residual` is itself a transcription of the validated C++ kernel
and is FCI-anchored through `_ccsd_ref`; this solver therefore inherits
that anchor (CCSD == FCI for two electrons; canonical CCSD re-validated
to 2e-11 Ha against PySCF).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._ccsd_cs import _blocks, cs_ccsd_pair_ladder, cs_ccsd_residual
from .pao import (
    build_atom_basis_map,
    build_projection_matrix,
    orbital_domain_atom_mask,
    semicanonical_pao_basis,
)
from .pno_density import pair_density, resolve_pno_norm

# Optional C++ per-pair residual kernel (vibeqc::dlpno_pair_residual). It
# reproduces `cs_ccsd_residual` bit-for-bit (<=3e-14) but replaces the
# ~90 small numpy einsums per call with one compiled call. Absent on cores
# built before the binding landed, so the solver falls back to numpy.
try:
    from .._vibeqc_core import dlpno_pair_residual as _cpp_pair_residual

    _HAVE_CPP_RESIDUAL = True
except ImportError:  # pragma: no cover - depends on the compiled core
    _cpp_pair_residual = None
    _HAVE_CPP_RESIDUAL = False

try:
    from .._vibeqc_core import (
        dlpno_project_tno_amplitudes as _cpp_project_amplitudes,
    )

    _HAVE_CPP_PROJECT = True
except ImportError:
    _cpp_project_amplitudes = None
    _HAVE_CPP_PROJECT = False

try:
    from .._vibeqc_core import dlpno_target_pair_residual as _cpp_target_residual

    _HAVE_CPP_TARGET_RESIDUAL = True
except ImportError:  # pragma: no cover - depends on the compiled core
    _cpp_target_residual = None
    _HAVE_CPP_TARGET_RESIDUAL = False

# Compiled PNO-space amplitude solver (vibeqc::dlpno_ccsd_pno_solve): the
# whole iteration with every residual term contracted in the PNO space that
# carries the contracted index (Riplinger and Neese 2013, Sec. II C). It has
# no NumPy transcription; ``residual_domain="pno"`` needs a core that has it.
try:
    from .._vibeqc_core import dlpno_ccsd_pno_solve as _cpp_pno_solve

    _HAVE_CPP_PNO_SOLVE = True
except ImportError:  # pragma: no cover - depends on the compiled core
    _cpp_pno_solve = None
    _HAVE_CPP_PNO_SOLVE = False


# Full-virtual semicanonical MP2 pair energies for the pair screening, in
# parallel (vibeqc::dlpno_sc_mp2_pair_energies). Cores built before it fall
# back to the NumPy loop, which evaluates the same sums on one BLAS thread.
try:
    from .._vibeqc_core import dlpno_sc_mp2_pair_energies as _cpp_sc_pair_energies
except ImportError:  # pragma: no cover - depends on the compiled core
    _cpp_sc_pair_energies = None


# Whether the compiled residual kernels accept `include_ladder` (#700). Cores
# built before that flag landed silently compute the ladder anyway, which would
# double-count it once the pair-space term is added, so probe rather than assume.
# pybind11 embeds the signature in __doc__.
_HAVE_CPP_LADDER_FLAG = bool(
    _cpp_pair_residual is not None
    and "include_ladder" in (_cpp_pair_residual.__doc__ or "")
    and _cpp_target_residual is not None
    and "include_ladder" in (_cpp_target_residual.__doc__ or "")
)

# Same probe for the #700 (b) ring flag. A core without it computes the ring
# terms in the extended basis regardless, so adding the pair-space term would
# DOUBLE-COUNT them; refuse rather than return a wrong energy.
_HAVE_CPP_RING_FLAG = bool(
    _cpp_pair_residual is not None
    and "return_ring" in (_cpp_pair_residual.__doc__ or "")
    and _cpp_target_residual is not None
    and "return_ring" in (_cpp_target_residual.__doc__ or "")
)


#: The local (T) algorithms of ``LocalCCSDOptions.triples_mode``.
TRIPLES_MODES = ("local", "t1", "exact")
#: The residual contraction spaces of ``LocalCCSDOptions.residual_domain``.
RESIDUAL_DOMAINS = ("pno", "pair", "extended", "full")


def _validate_triples_mode(value: object) -> str:
    mode = str(value).strip().lower()
    if mode not in TRIPLES_MODES:
        raise ValueError(
            f"unknown triples_mode: {value!r} "
            '(expected "local", "t1", or "exact")'
        )
    return mode


#: The local MP2 schemes of ``LocalCCSDOptions.pno_guess``.
PNO_GUESSES = ("semicanonical", "lmp2")


def _validate_pno_guess(value: object) -> str:
    guess = str(value).strip().lower()
    if guess not in PNO_GUESSES:
        raise ValueError(
            f'unknown pno_guess: {value!r} (expected "semicanonical" or "lmp2")'
        )
    return guess


@dataclass
class LocalCCSDOptions:
    localise: str = "boys"
    n_frozen: int | None = None
    # This solver implements the full Liakos 2015 NormalPNO triple selected
    # as vibe-qc's cross-route policy. Pinski's tighter MP2-specific PNO
    # setting is intentionally not used as the CCSD default. Zero remains the
    # explicit full-domain/no-truncation setting for exactness tests.
    tcut_pno: float = 3.33e-7
    #: Factor on ``tcut_pno`` for the DIAGONAL pairs only, whose PNO space
    #: also carries the singles. ORCA cuts its singles PNOs at
    #: ``TCutPNOSingles = 0.03 * TCutPNO`` (printed as 9.99e-9 at NormalPNO
    #: by ORCA 6.1.1); this solver keeps one space per diagonal pair, so the
    #: default ``0.03`` gives the singles ORCA's threshold (and the diagonal
    #: doubles a slightly larger space than ORCA's). The semicanonical PNO
    #: correction cannot recover singles truncation, since singles do not
    #: enter MP2: at ``1.0``, the convention of every result recorded before
    #: this default, n-butane/cc-pVDZ NormalPNO CCSD sits 1.62 kcal/mol above
    #: canonical against 1.08 at ``0.03`` (ORCA 6.1.1: 1.16). #65 note 20763
    #: recorded this as one of three ORCA deviations.
    tcut_pno_diag_scale: float = 0.03
    tcut_mkn: float = 1e-3  # 0 -> full PAO domains
    #: How a localised orbital's PAO atom domain is chosen; a pair's domain
    #: is the union of its two orbitals' domains.
    #:
    #: * ``"doi"`` (default): atoms with a projected atomic orbital whose differential
    #:   overlap with the orbital exceeds ``tcut_do`` (Pinski, Riplinger,
    #:   Valeev and Neese, J. Chem. Phys. 143, 034108 (2015), Eq. 4), the
    #:   criterion ORCA uses for its PAO domains. A Mulliken population can be
    #:   tiny next to real density: NH3's Boys lone pair has 1.7e-4 on each
    #:   hydrogen, so at ``tcut_mkn=1e-3`` its diagonal pair builds PNOs from
    #:   nitrogen PAOs alone and the NormalPNO CCSD energy is 0.34 kcal/mol
    #:   above canonical, against 0.02 with the full domain (and 0.02 for
    #:   ORCA 6.1.1, whose DOI domains keep the hydrogens).
    #: * ``"mulliken"``: atoms carrying the orbital's Mulliken population
    #:   above ``tcut_mkn`` (Riplinger and Neese 2013, Sec. II B 1), the
    #:   convention of every result recorded before the DOI default.
    #:
    #: Only the local CCSD route reads this; ``tcut_mkn`` stays the published
    #: coordinate for the other DLPNO routes. Under either criterion
    #: ``tcut_mkn=0`` keeps every atom, the full-domain exactness setting;
    #: under ``"doi"`` a positive ``tcut_mkn`` does not shape the domains.
    domain_criterion: str = "doi"
    #: Differential-overlap cut for ``domain_criterion="doi"`` (ORCA 6.1.1
    #: NormalPNO ``TCutDO``; the Loose and Tight presets set 2e-2 and 5e-3).
    #: 0 keeps every atom.
    tcut_do: float = 1e-2
    # Pair screening threshold (Ha): pairs whose MP2 energy estimate in
    # the full virtual space is below this value are treated at the MP2
    # level (accumulated into e_corr directly) rather than iterated at
    # the CCSD level. 1e-4 is the published NormalPNO TCutPairs coordinate;
    # set 0.0 for the no-screening exactness reference.
    tcut_pairs: float = 1e-4
    # Coupling radius (bohr): occupied m enters pair (ij)'s residual only if
    # its localised centroid is within this distance of i's or j's centroid.
    # This restricts the na^2 occupied coupling sums to a bounded local set
    # (O(N⁴)->O(N^2) in the occupied dimension). The 12-bohr default is a
    # calibrated, conservative cutoff and a **bit-identical no-op on any
    # molecule under ~12-bohr extent** (the common case): single-fragment
    # correlation coupling falls off fast (cc-pVDZ water-dimer sweep: 7 bohr
    # -> 3e-7 Ha, 9 -> 3e-9, 11 -> 1e-12). On *extended* systems the dropped
    # long-range coupling accumulates to a controlled approximation kept
    # well below the PNO truncation error (~7 µHa on a 25-bohr H₂ chain vs
    # the ~0.1% PNO error) -- the same controlled-locality bargain as the
    # default sparse pair list. Set 0.0 for full coupling -- every pair
    # couples to every occupied -- the exact reference the parity ratchet pins.
    coupling_radius: float = 12.0
    # Optional override for the centroid pair distance behind the
    # coupling-radius occupied sets: pair_distance_fn(r_i, r_j) -> float
    # (bohr). None (default) uses the Euclidean norm -- bit-identical to
    # the historical behavior. Periodic toroidal wrappers pass the exact
    # minimum-image distance (vibeqc.periodic_toroidal_mp2.
    # toroidal_pair_distance_fn): the open-cluster Euclidean distance
    # overestimates wrap-around separations on a toroidal reference and
    # would silently under-couple pairs that are adjacent on the torus
    # (the original investigation Stage 5c).
    pair_distance_fn: object = None
    lindep: float = 1e-8
    #: Space in which the pair residual is contracted before being projected
    #: back into the pair's own PNO space.
    #:
    #: * ``"pno"`` (default) contracts every term of the residual in the PNO
    #:   space of the amplitude that carries the contracted index, the term-by-term
    #:   algorithm of Riplinger and Neese, J. Chem. Phys. 138, 034106 (2013),
    #:   Sec. II C, in the compiled ``dlpno_ccsd_pno_solve`` kernel. Free
    #:   indices are projected into the target pair through the PNO overlaps
    #:   (their Eq. 30); singles products are folded into the density-fitting
    #:   factors over the full virtual space, so nothing is truncated before
    #:   it is contracted. Exact against ``"extended"`` at the same coupling
    #:   set and against canonical CCSD at full domains, at a per-iteration
    #:   cost of O(n_pairs nL^2 n_aux n_pno^2) that does not grow with the
    #:   extended-domain dimension. Compiled route only; ``use_cpp_kernel``,
    #:   ``ladder_in_pno_space`` and ``ring_in_pno_space`` do not apply.
    #: * ``"pair"`` projects the coupled pairs' amplitudes into pair
    #:   ``ij``'s space *first* and contracts there. This is the historical
    #:   opt-in behaviour and it is wrong under truncation: it discards the
    #:   parts of those amplitudes lying outside ``ij``'s space that still
    #:   feed back into it through the integrals.
    #: * ``"extended"`` is the extended domain of Riplinger and Neese,
    #:   J. Chem. Phys. 138, 034106 (2013), Sec. II B 1, "necessary in order to
    #:   create a sufficiently large buffer for the accurate calculation of the
    #:   pair-pair interaction terms". It is defined over **atoms**: the union
    #:   of the orbital domains of ``i``, ``j`` and every coupled ``k``,
    #:   spanned by the PAOs on those atoms. Being PAO-based it does not depend
    #:   on ``tcut_pno``, which is the point -- a buffer built from the
    #:   truncated PNO spaces would be built from the very deficiency it exists
    #:   to absorb. At the explicit exactness setting ``tcut_mkn=0`` every
    #:   orbital domain is all atoms, so this spans the whole virtual space and
    #:   coincides with ``"full"``. The NormalPNO default ``tcut_mkn=1e-3``
    #:   instead restricts the atom domains. The domain is taken over the
    #:   pair's whole occupied coupling set (``coupling_radius``), which is
    #:   what makes the contraction exact against ``"full"`` at the same
    #:   coupling set; the paper's smaller pair-list condition suffices only
    #:   for its term-by-term PNO-basis algorithm. Pairs sharing a coupling
    #:   set and an extended domain share one contraction per iteration.
    #: * ``"full"`` expands to the whole virtual space unconditionally,
    #:   contracts, and projects the residual back. That is the exact
    #:   stationarity condition of the subspace-constrained ansatz and what
    #:   ``run_dlpno_ccsd_pilot`` does. O(N^6): the correctness setting, not a
    #:   production one.
    residual_domain: str = "pno"
    #: Pair density whose eigenvalues are compared against ``tcut_pno``, i.e.
    #: what a PNO occupation number means. ``"mp2"`` (default) is Riplinger
    #: and Neese's Eq. 23, ORCA's ``PNONorm MP2Norm`` default, and is the
    #: density the published ``TCutPNO`` presets are calibrated against;
    #: ``"legacy"`` is vibe-qc's historical density of the bare amplitudes and
    #: ``"iepa"`` the pre-2013 LPNO convention. Defaulted to ``"mp2"`` by the
    #: #65 (old #701) ruling so that a preset name means what the literature
    #: means by it: under ``"legacy"`` the same nominal threshold kept 5-19 %
    #: fewer PNOs, i.e. our NormalPNO delivered their LoosePNO. See
    #: :mod:`vibeqc.dlpno.pno_density`.
    pno_norm: str = "mp2"
    max_iter: int = 100
    conv_tol_energy: float = 1e-9
    conv_tol_residual: float = 1e-7
    diis_size: int = 6
    # Use the compiled closed-shell DF-CCSD residual kernels
    # (vibeqc::dlpno_pair_residual / dlpno_target_pair_residual) when the
    # core provides them -- one compiled call instead of ~90 small NumPy
    # einsums per contraction. The kernels take DF B-tensors in whatever
    # orthonormal virtual basis they are handed, so they serve the pair-PNO
    # basis of the legacy "pair" mode and the extended/full basis of the
    # default alike; the extended path also uses the compiled kernel's
    # blocked (ae|bf) ladder instead of retaining a dense n_ext^4 block per
    # domain. Set False to force the NumPy transcription for a parity check.
    use_cpp_kernel: bool = True
    #: Contract the particle-particle (W_abef) ladder in each pair's own PNO
    #: space rather than in the group's extended basis (#700). Exact: the
    #: ladder's contracted indices are carried by ``tau_ij``, which lives
    #: there, and the pair's PNO space is a subspace of the extended basis.
    #: Removes the O(n_ext^4) term that makes ``residual_domain="extended"``
    #: cost a canonical CCSD iteration on any molecule whose domains span it.
    #: ``None`` (default) enables it wherever the residual routine can be
    #: asked to omit the ladder; ``True``/``False`` force it, which is how the
    #: equivalence is pinned in the tests.
    ladder_in_pno_space: bool | None = None

    #: Contract the three ring terms in each SOURCE pair's PNO space rather
    #: than in the group's extended basis (#700, Riplinger and Neese Eqs.
    #: (26)-(28)). Exact, but for a different reason than the ladder: there
    #: the contracted indices are carried by the target pair's own tau, here
    #: they belong to the source pair kj, so the amplitude stays in ITS space
    #: and only the free index is carried across, by the Eq. (30) overlap
    #: S^{ij,kl} = U_ij^T U_kl. Projecting the source amplitude into the
    #: target pair first is the #98 truncation, not this.
    #:
    #: Measured on the compiled kernel, the ring terms are 87% (C3H8) to 94%
    #: (C4H10) of the extended-domain residual once the ladder has moved, so
    #: this is where the remaining cost is. A compiled core predating the
    #: ``include_ring`` flag would contract the ring in the extended basis
    #: anyway, so asking for this there raises rather than double-counting.
    ring_in_pno_space: bool | None = None
    # (T) perturbative triples on the converged local amplitudes (local
    # DLPNO-(T), `triples_local.local_triples_correction`). Each triple's
    # TNO virtual domain is built from the union of its pair PNO spaces,
    # then truncated by **TNO occupation number** -- the triple's pair
    # amplitudes define a density whose eigenvalues are the occupations, and
    # TNOs above tcut_tno are kept (local DLPNO-(T), Riplinger 2013).
    # tcut_tno=0 keeps the full union span (exact at full PNO domains);
    # raising it trades (T) cost for a graceful recovery loss (H₂O/def2-SVP:
    # 1e-5 -> 92.9 %, 1e-4 -> 79.9 % of canonical). The validated default is 0.
    compute_triples: bool = False
    tcut_tno: float = 0.0
    # (T) algorithm: "local" = reduced-scaling DLPNO-(T0) (diagonal localised
    # Fock, ~0.1 kcal/mol semicanonical error); "t1" = compatibility route
    # that diagonalises and rotates the occupied space before the TNO-domain
    # contraction (the default); "exact" = canonical (T) on the converged
    # amplitudes (full virtual space -- O(N⁷), the accuracy oracle).  The
    # iterative local-basis algorithm of Guo et al. (2018) is the distinct
    # open-shell ``triples_mode="t1-iterative"`` route.
    triples_mode: str = "t1"
    # Soft guard: this is a Python reference implementation whose per-pair
    # amplitude coupling still loops all pairs (reduced- but not yet
    # linear-scaling). Deliberately overridable; the C++ port lifts it.
    #
    # Raised 200 -> 1500 for #190. DLPNO's own literature puts the
    # canonical/local crossover for alkane chains at ~C40 and the linear
    # scaling regime beyond ~C60 (Pinski, Becker, Valeev and Neese, JCP 143,
    # 034108 (2015)), so any measurement of this method's scaling has to
    # reach those sizes. The old cap tripped at 202 basis functions -- C8H18
    # -- which is 5x below the crossover, i.e. it excluded the entire regime
    # in which the method is supposed to pay off. cc-pVDZ: C40H82 is 970
    # basis functions, C60H122 is 1450; 1500 covers the range #190 asks for.
    #
    # This is a guard against launching an intractable run by accident, not
    # a statement that everything under it is quick. Extrapolating the
    # measured C3-C5 kernel timings, a converged C40 run is days of laptop
    # CPU even once the extended domain becomes a proper subset. Sizes in
    # this range are cluster work; the cap simply stops being the thing that
    # prevents them.
    max_nbf: int = 1500
    #: Drop the density-fitting object's AO-basis tensors (``df.B`` and the
    #: raw three-centre integrals) once the solver has transformed what it
    #: needs. They are the largest arrays on a large molecule (two copies of
    #: n_aux x nbf^2) and nothing in this route reads them afterwards; the
    #: triples reuse the solver's MO factors. Off by default because it
    #: mutates the caller's ``df``; ``run_job`` constructs a private one.
    #: Ignored for ``compute_triples`` with ``triples_mode="exact"``, which
    #: re-transforms from the AO tensor.
    release_df_tensors: bool = False
    #: Diagnostic only: on the ``"pno"`` and ``"pair"`` routes, also report
    #: the dimension each pair's atom-based ``"extended"`` residual domain
    #: would have, as ``LocalCCSDResult.avg_extended_domain``. Its ratio to
    #: the virtual-space dimension is the virtual-side locality measure of
    #: #190. It costs one overlap eigenvalue problem per distinct domain and
    #: changes no energy. The ``"extended"`` and ``"full"`` routes always
    #: report it.
    report_extended_domain: bool = False
    # RI fitting basis for the correlation integrals. None -> the runner
    # auto-resolves it from the orbital basis (density_fitting.
    # default_aux_basis_for(basis, kind="ri")). Set explicitly (e.g.
    # "cc-pvdz-ri") for orbital bases that ship no default RI aux -- minimal
    # / Pople bases (sto-3g, 6-31g*, ...) have no standard RI fitting set.
    aux_basis: str | None = None

    def __post_init__(self) -> None:
        self.triples_mode = _validate_triples_mode(self.triples_mode)
        self.pno_guess = _validate_pno_guess(self.pno_guess)

    # One-shot CCSD-level diagnostic for the PNO tail discarded by tcut_pno.
    # The retained-PNO amplitudes are converged first; then each pair is
    # augmented with its discarded pair-natural-orbital complement, the CCSD
    # residual is evaluated in that augmented pair space. The returned
    # estimate is deliberately *not* added to the production energy: current
    # calibration shows a one-shot tail delta can worsen compact-molecule
    # recovery when the retained truncated space is already over-correlated.
    # It remains available as an explicitly requested CCSD-amplitude-level
    # diagnostic, without implying that it is a production energy correction.
    estimate_pno_tail: bool = False
    #: Add the semicanonical MP2 correction for the PNO truncation of the
    #: pairs that ARE iterated, ``sum_ij (e_ij^full-PAO - e_ij^PNO)``.
    #:
    #: Riplinger and Neese 2013 apply an MP2 correction for both truncation
    #: errors: the dropped pairs (this solver already adds those, as
    #: ``e_pair_screened``) and the PNO truncation of the kept pairs (this).
    #: Sec. II C, and Sec. III: "the MP2 correction captures the remaining
    #: pair approximation and PNO errors very well". The DLPNO-MP2 route has
    #: always applied it (``DLPNOMP2Result.e_pno_correction``); this route
    #: did not, which is why its truncation errors ran larger.
    #:
    #: Measured on NH3 (S22-01 A) / cc-pVDZ with the domain and pair list
    #: held fixed so that only ``tcut_pno`` varies, it recovers 91.5 % of the
    #: PNO truncation error at ``tcut_pno=1e-5`` and 34.2 % at NormalPNO.
    #: Exactly zero at ``tcut_pno=0``, where nothing is truncated, so the
    #: exactness ratchets are untouched.
    #:
    #: Set False to reproduce a number recorded before this was added.
    pno_correction: bool = True
    #: Local MP2 behind the PNOs, the MP2 truncation corrections and the
    #: CCSD starting amplitudes.
    #:
    #: * ``"semicanonical"`` (default): semicanonical local MP2 in each pair's
    #:   PAO domain, the occupied Fock matrix taken as diagonal (Riplinger and
    #:   Neese 2013, Sec. II B). ORCA's LoosePNO and NormalPNO.
    #: * ``"lmp2"``: full local MP2, the TightPNO scheme of Liakos et al.,
    #:   J. Chem. Theory Comput. 11, 1525 (2015), Sec. 2.2. Semicanonical
    #:   amplitudes first give intermediate PNOs cut at
    #:   ``lmp2_pno_scale * tcut_pno``, for the CCSD pairs and for the weak
    #:   pairs whose semicanonical energy reaches ``lmp2_pair_cut``. The
    #:   coupled LMP2 equations, off-diagonal occupied Fock coupling included,
    #:   are solved in those spaces (:mod:`vibeqc.dlpno.lmp2_guess`), and the
    #:   final PNOs are the natural orbitals of the LMP2 pair densities. The
    #:   weak pairs take their LMP2 energies; the PNO truncation correction is
    #:   the LMP2 energy lost when the CCSD pairs move from intermediate to
    #:   final PNOs, plus the semicanonical estimate of the intermediate cut.
    #:   The CCSD iterations start from LMP2 amplitudes in the final PNOs.
    #:   ``lmp2_pno_scale``, ``lmp2_pair_cut`` and ``lmp2_fock_cut`` default
    #:   to ORCA 6.1.1's TightPNO output (ScaleTCutPNO 1e-2, TCutMP2Pairs
    #:   1e-6, Fock cutoff 1e-5).
    #:
    #: The pair list is the same under both; only ``"lmp2"`` reads the
    #: ``lmp2_*`` fields.
    pno_guess: str = "semicanonical"
    lmp2_pno_scale: float = 1e-2
    lmp2_pair_cut: float = 1e-6
    lmp2_fock_cut: float = 1e-5
    lmp2_max_iter: int = 50
    lmp2_conv_tol_energy: float = 1e-7
    lmp2_conv_tol_residual: float = 5e-7
    #: Memory bound (GB) of the dense LMP2 sweep; past it, or at 0, the
    #: compiled per-link sweep runs instead.
    lmp2_dense_max_gb: float = 16.0


@dataclass
class LocalCCSDResult:
    #: Adapter name and SHA-256 identity of the correlation source the
    #: engine read (:mod:`vibeqc.dlpno.source`).
    source_kind: str = ""
    source_identity: str = ""
    e_hf: float = 0.0
    e_corr: float = 0.0
    e_t: float = 0.0  # 0 for CCSD (no triples); kept for proxy parity
    e_total: float = 0.0
    n_iter: int = 0
    converged: bool = False
    n_pairs: int = 0
    n_frozen: int = 0
    t1_norm: float = 0.0
    avg_coupled_occ: float = 0.0  # mean local occupied-set size per pair
    #: Mean dimension of the basis each pair's residual is contracted in.
    #: Equals the pair's own PNO count for ``residual_domain="pair"``, the
    #: atom-based extended PAO-domain dimension for ``"extended"``, and the
    #: full virtual-space dimension for ``"full"``.
    avg_residual_domain: float = 0.0
    #: Mean dimension of the pairs' atom-based extended domains (the PAOs on
    #: the union of the orbital domains of every occupied in the pair's
    #: coupling set), whether or not the residual was contracted there. Set
    #: on the ``"extended"`` and ``"full"`` routes, and on the others when
    #: ``LocalCCSDOptions.report_extended_domain`` is true; zero otherwise.
    avg_extended_domain: float = 0.0
    #: Pair-density convention the PNO occupation numbers were cut against
    #: (:mod:`vibeqc.dlpno.pno_density`). ``tcut_pno`` only means what the
    #: published presets intend under ``"mp2"``, which is the default.
    pno_norm: str = "mp2"
    #: How many of those groups contracted the particle-particle ladder in
    #: their member pairs' own PNO spaces instead of the extended basis
    #: (#700). Auto-selected per group by the locality ratio n_ext/n_pno, so a
    #: zero here on a compact molecule is the gate working, not a failure.
    n_ladder_pno_groups: int = 0
    #: Number of distinct (occupied coupling set, extended atom domain)
    #: groups the extended/full contraction was evaluated in. Each group costs
    #: one residual contraction per iteration regardless of how many pairs it
    #: holds; on a compact molecule this is 1. Zero for ``"pair"``. Groups
    #: with the same extended atom set share the basis and its (ab|cd) factor.
    n_residual_domains: int = 0
    avg_tno: float = 0.0  # mean retained TNO domain per distinct triple
    #: Distinct occupied triples whose TNO domain retained fewer than three
    #: virtuals. A spatial triple excitation needs three distinct virtuals, so
    #: such a triple contributes exactly zero rather than approximately: a
    #: nonzero count means ``tcut_tno`` truncated past the point where the
    #: correction exists, not that it is small.
    n_degenerate_tno_triples: int = 0
    #: Distinct occupied triple keys evaluated, and how many ``coupling_radius``
    #: screened away. Screening is the occupied-side analogue of the TNO
    #: truncation on the virtual side: on a delocalized system a generous
    #: radius can discard most of the triples energy, which the energy alone
    #: does not reveal.
    n_triple_keys: int = 0
    n_triple_keys_screened: int = 0
    tno_per_triple: dict = field(default_factory=dict)
    pno_per_pair: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    n_screened: int = 0  # pairs treated at MP2 level (below tcut_pairs)
    e_pno_tail_estimate: float = 0.0  # CCSD-level discarded-PNO tail diagnostic
    #: Semicanonical MP2 correction for the PNO truncation of the iterated
    #: pairs, already included in ``e_corr``. Companion to
    #: ``e_pair_screened``'s correction for the dropped pairs.
    e_pno_correction: float = 0.0
    #: ``LocalCCSDOptions.pno_guess`` of this run, and for ``"lmp2"``: the
    #: pairs in the LMP2 solve, its correlation energy in the intermediate
    #: PNO spaces, the iterations of its intermediate and final solves, and
    #: whether both converged.
    pno_guess: str = "semicanonical"
    n_lmp2_pairs: int = 0
    e_lmp2: float = 0.0
    lmp2_iterations: tuple = ()
    lmp2_converged: bool = False
    #: True only when the requested triples contraction was actually entered.
    #: In particular, an all-screened CCSD pair space returns before triples.
    triples_executed: bool = False
    #: ``residual_domain="pno"`` bookkeeping: density-fitting factor rows a
    #: pair had to store for occupieds outside its own coupling set because
    #: another target's contraction needed them, and the resident bytes of
    #: the static per-pair factors. Zero on the other routes.
    n_offlist_rows: int = 0
    static_factor_bytes: int = 0
    #: Wall-clock seconds the compiled route spent on its static factors and
    #: on the amplitude iterations; zero on the other routes.
    setup_seconds: float = 0.0
    iteration_seconds: float = 0.0
    #: Wall-clock seconds per phase of the whole route, in the order run:
    #: ``reference``, ``localise``, ``factors`` (density-fitting transforms),
    #: ``domains``, ``screening``, ``lmp2_guess`` (``pno_guess="lmp2"``
    #: only), ``pair_setup`` (PNO spaces and per-pair data), ``amplitudes``
    #: (the amplitude solve, including the compiled setup) and ``triples``.
    phase_seconds: dict = field(default_factory=dict)


def run_local_dlpno_ccsd(molecule, basis, rhf, df, options=None, *, source=None):
    """Local DLPNO-CCSD in per-pair PNO spaces on a closed-shell RHF reference.

    The engine reads the reference, the AO overlap and the three-centre
    factors through a :class:`~vibeqc.dlpno.source.CorrelationSource`.
    Pass one as ``source`` (the positional ``molecule``, ``basis``, ``rhf``
    and ``df`` may then be ``None``); without it the molecular adapter
    wraps the positional arguments, which leaves every molecular result
    unchanged.
    """
    from vibeqc.correlation_conventions import effective_electron_count

    from .source import CorrelationSource, molecular_source

    if options is None:
        options = LocalCCSDOptions()
    if source is None:
        source = molecular_source(molecule, basis, rhf, df)
    elif not isinstance(source, CorrelationSource):
        raise TypeError(
            "source must be a vibeqc.dlpno.source.CorrelationSource, got "
            f"{type(source).__name__}"
        )
    source.validate_reference()
    molecule, basis = source.molecule, source.basis
    rhf, df = source.reference, source.factors

    triples_mode = _validate_triples_mode(options.triples_mode)
    pno_norm = resolve_pno_norm(getattr(options, "pno_norm", "mp2"))
    if options.residual_domain not in RESIDUAL_DOMAINS:
        raise ValueError(
            f"unknown residual_domain: {options.residual_domain!r} "
            '(expected "pno", "pair", "extended" or "full")'
        )
    _pno_mode = options.residual_domain == "pno"
    if _pno_mode and not _HAVE_CPP_PNO_SOLVE:
        raise ValueError(
            'residual_domain="pno" needs the compiled dlpno_ccsd_pno_solve '
            "kernel, which this core predates; rebuild it or select "
            'residual_domain="extended"'
        )
    _extended = options.residual_domain in ("extended", "full")
    _union_domain = options.residual_domain == "extended"
    # The compiled residual kernel is basis-agnostic: it takes DF B-tensors
    # over an orthonormal virtual basis of any dimension, so it serves the
    # pair-PNO basis of "pair" mode and the extended/full basis alike.
    _use_cpp = bool(options.use_cpp_kernel) and _HAVE_CPP_RESIDUAL
    # #700: contract the particle-particle ladder in each pair's OWN PNO space
    # instead of the group's extended basis. Exact (the ladder's contracted
    # indices are carried by tau_ij, which lives there), and it removes the
    # O(n_ext^4) term that makes the extended default cost a canonical CCSD
    # iteration on any molecule whose domains span it. Only taken where the
    # residual routine can be asked to omit the ladder; the compiled kernel
    # does not yet accept that flag, so the C++ path keeps the old route and
    # stays bit-identical.
    _cpp_can_omit_ladder = (not _use_cpp) or _HAVE_CPP_LADDER_FLAG
    _req_ladder = getattr(options, "ladder_in_pno_space", None)
    if _req_ladder and not _cpp_can_omit_ladder:
        raise ValueError(
            "ladder_in_pno_space=True needs a residual routine that can omit "
            "the ladder; this compiled core predates the include_ladder flag, "
            "so rebuild it or pass use_cpp_kernel=False"
        )
    # Whether ANY group may take the pair-space route. Which ones actually do
    # is decided per group below, once their members' PNO counts are known.
    _ladder_in_pno = _extended and _cpp_can_omit_ladder and _req_ladder is not False

    # #700 (b): the ring terms in each source pair's PNO space. Unlike the
    # ladder there is no compiled counterpart yet -- the kernel has no
    # include_ring flag -- so this is the numpy route only, and asking for it
    # on the compiled route is an error rather than a silent no-op.
    _req_ring = getattr(options, "ring_in_pno_space", None)
    if _req_ring and _use_cpp and not _HAVE_CPP_RING_FLAG:
        raise ValueError(
            "ring_in_pno_space=True needs a residual routine that can omit "
            "the ring terms; this compiled core predates the include_ring "
            "flag, so rebuild it or pass use_cpp_kernel=False"
        )
    if _req_ring and not _extended:
        raise ValueError(
            'ring_in_pno_space=True needs residual_domain="extended" or '
            '"full"; the legacy "pair" route already contracts in the pair '
            "space, truncating the coupled amplitudes as it goes"
        )
    _ring_in_pno = bool(
        _extended and _req_ring
        and ((not _use_cpp) or _HAVE_CPP_RING_FLAG)
    )

    import time as _time

    _phases: dict[str, float] = {}
    _phase_mark = [_time.perf_counter()]

    def _tick(name):
        now = _time.perf_counter()
        _phases[name] = _phases.get(name, 0.0) + (now - _phase_mark[0])
        _phase_mark[0] = now

    F = np.asarray(rhf.fock).copy()
    S = np.asarray(source.overlap_matrix()).copy()
    C = np.asarray(rhf.mo_coeffs).copy()
    n_occ = effective_electron_count(molecule, rhf) // 2
    nbf = C.shape[0]
    if nbf > options.max_nbf:
        raise ValueError(
            f"run_local_dlpno_ccsd: {nbf} basis functions exceeds the "
            f"max_nbf={options.max_nbf} guard. This is a Python reference "
            f"implementation (per-pair coupling still loops all pairs); raise "
            f"LocalCCSDOptions.max_nbf to override (the C++ port lifts it). "
            f"The default covers the C40-C60 range of issue #190 (cc-pVDZ: "
            f"C60H122 is 1450 functions); past that, check the cost first -- "
            f"the per-iteration residual grows steeply with the extended-"
            f"domain dimension."
        )
    from vibeqc.correlation_conventions import resolve_frozen_core_count

    nf = resolve_frozen_core_count(molecule, options.n_frozen, reference=rhf)
    if nf < 0 or nf >= n_occ:
        raise ValueError(f"n_frozen={nf} out of range for n_occ={n_occ}")
    n_act = n_occ - nf
    C_occ_full = C[:, :n_occ]
    C_act = C[:, nf:n_occ]
    C_vir = C[:, n_occ:]
    _tick("reference")

    if options.localise == "boys":
        from vibeqc import compute_dipole
        from vibeqc.localise import foster_boys_localise

        dip = compute_dipole(basis)
        dipoles = np.zeros((nbf, nbf, 3))
        dipoles[:, :, 0] = np.asarray(dip.x)
        dipoles[:, :, 1] = np.asarray(dip.y)
        dipoles[:, :, 2] = np.asarray(dip.z)
        C_loc = foster_boys_localise(C_act, dipoles, max_iter=200)
    else:
        C_loc = C_act

    # Localised-orbital (Boys) centroids -> pairwise distances define each
    # pair's local occupied coupling set. Distances are origin/sign
    # invariant, so the raw <mu|r|ν> dipole integrals suffice.
    from vibeqc import compute_dipole as _compute_dipole

    _dip = _compute_dipole(basis)
    _r = [np.asarray(_dip.x), np.asarray(_dip.y), np.asarray(_dip.z)]
    centroids = np.column_stack(
        [np.einsum("mi,mn,ni->i", C_loc, ax, C_loc, optimize=True) for ax in _r]
    )
    # The option wins over the source's distance (a crystal adapter carries
    # the minimum-image distance; a molecule carries none).
    _pair_distance = (
        options.pair_distance_fn
        if options.pair_distance_fn is not None
        else source.pair_distance_fn
    )
    if _pair_distance is not None:
        occ_dist = np.zeros((n_act, n_act))
        for _i in range(n_act):
            for _j in range(_i + 1, n_act):
                occ_dist[_i, _j] = occ_dist[_j, _i] = float(
                    _pair_distance(centroids[_i], centroids[_j])
                )
    else:
        occ_dist = np.linalg.norm(
            centroids[:, None, :] - centroids[None, :, :], axis=2
        )
    R_couple = float(options.coupling_radius)

    def coupled_occupieds(i, j):
        """Local occupied set for pair (i, j): occupieds within R_couple of
        i or j (always including i, j). R_couple <= 0 -> all occupieds."""
        if R_couple <= 0.0:
            return np.arange(n_act, dtype=int)
        near = (occ_dist[i] < R_couple) | (occ_dist[j] < R_couple)
        near[i] = near[j] = True
        return np.where(near)[0]

    _tick("localise")
    f_oo = C_loc.T @ F @ C_loc
    f_dd = np.diag(f_oo)
    f_vv_full = C_vir.T @ F @ C_vir
    f_ov_full = C_loc.T @ F @ C_vir
    B_ov = np.ascontiguousarray(np.asarray(df.mo_transform(C_loc, C_vir)))
    B_vv = np.ascontiguousarray(np.asarray(df.mo_transform(C_vir, C_vir)))
    B_oo = np.ascontiguousarray(np.asarray(df.mo_transform(C_loc, C_loc)))
    # Occupied-half-transformed factor B^P_{i nu} over the localised active
    # occupieds, built once. Each pair's PAO-basis factors are then one
    # (naux, nbf) x (nbf, n_pao) product instead of a transform of the whole
    # three-index tensor per pair, which is O(n_aux nbf^2 n_pao) each.
    _B_ao = getattr(df, "B", None)
    if _B_ao is not None and np.ndim(_B_ao) == 3:
        B_half = np.einsum("mp,Pmn->Ppn", C_loc, np.asarray(_B_ao), optimize=True)
    else:  # pragma: no cover - DF objects without an exposed AO tensor
        B_half = np.asarray(df.mo_transform(C_loc, np.eye(nbf)))
    _tick("factors")

    Q_vir = build_projection_matrix(C_occ_full, S)
    # C_vir^T S maps every PAO-built virtual (PNOs, extended bases) onto the
    # canonical virtuals. Formed once: the per-pair products below associate
    # as (C_vir^T S) V, so this changes no bit, and it was an n_vir nbf^2
    # product per pair.
    CvS = C_vir.T @ S
    atom_first, _ = build_atom_basis_map(molecule, basis)
    natom = len(atom_first) - 1
    _criterion = str(getattr(options, "domain_criterion", "doi")).strip().lower()
    if float(options.tcut_mkn) <= 0.0:
        # tcut_mkn=0 is the documented full-domain exactness setting; it keeps
        # that meaning under either criterion, so every recipe that pins it
        # (the parity ratchets, the periodic builders, the retained
        # benchmarks) stays exact whatever the domain criterion.
        _criterion = "mulliken"
    # (n_act, natom) atom domain of every localised orbital; a pair's domain
    # is the union of its two rows, which for "mulliken" is exactly the
    # population test select_domain_atoms_mulliken applies to a pair.
    orb_atom_mask = orbital_domain_atom_mask(
        molecule,
        basis,
        C_loc,
        S,
        Q_vir,
        atom_first,
        criterion=_criterion,
        tcut_mkn=float(options.tcut_mkn),
        tcut_do=float(getattr(options, "tcut_do", 1e-2)),
    )

    _orbital_domain_cache: dict[int, tuple[int, ...]] = {}
    _pao_basis_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}

    def orbital_domain(m):
        """Atom domain of localised orbital m under ``domain_criterion``;
        all atoms at the explicit zero-threshold exactness setting. Cached
        per orbital."""
        atoms = _orbital_domain_cache.get(m)
        if atoms is None:
            atoms = tuple(int(a) for a in np.where(orb_atom_mask[m])[0])
            _orbital_domain_cache[m] = atoms
        return atoms

    _tick("domains")
    # ---- pair screening: compute MP2 pair energies for all pairs ----
    # Weak pairs are treated at MP2 level, not iterated at CCSD level.
    # This is the key accuracy lever: strong pairs get full CCSD with
    # truncated PNOs, weak pairs get the exact (full-virtual) MP2 energy.
    e_pair_screened = 0.0
    screened_pairs: set = set()
    _lmp2_guess = _validate_pno_guess(getattr(options, "pno_guess", "semicanonical")) == "lmp2"
    # Full-virtual semicanonical energy of every screened pair, which the
    # LMP2 guess needs to choose and correct its weak pairs.
    e_sc_screened: dict = {}
    if options.tcut_pairs > 0.0:
        eps_v_diag = np.diag(f_vv_full)
        # O(n_act^2 n_vir^2 n_aux): the dominant cost of this route past ~C40
        # when it ran here on one BLAS thread. The compiled kernel evaluates
        # the same sums in parallel over pairs.
        E_sc_full = (
            np.asarray(
                _cpp_sc_pair_energies(
                    B_ov.reshape(B_ov.shape[0], -1),
                    np.ascontiguousarray(f_dd),
                    np.ascontiguousarray(eps_v_diag),
                )
            )
            if _cpp_sc_pair_energies is not None
            else None
        )
        for i in range(n_act):
            for j in range(i, n_act):
                if E_sc_full is not None:
                    e_mp2 = float(E_sc_full[i, j])
                else:
                    w = 2.0 if i != j else 1.0
                    # MP2 pair energy in the full virtual space (canonical
                    # virtual energies, no PAO domain restriction)
                    Bi_full = B_ov[:, i, :]
                    Bj_full = B_ov[:, j, :]
                    K_full = Bi_full.T @ Bj_full
                    denom = (
                        f_dd[i] + f_dd[j] - eps_v_diag[:, None] - eps_v_diag[None, :]
                    )
                    T_full = K_full / denom
                    e_mp2 = w * float(np.sum(T_full * (2.0 * K_full - K_full.T)))
                if abs(e_mp2) < options.tcut_pairs:
                    screened_pairs.add((i, j))
                    e_pair_screened += e_mp2
                    e_sc_screened[(i, j)] = e_mp2

    _tick("screening")
    # ---- per-pair PNO spaces + static local integral data ----
    estimate_pno_tail = bool(options.estimate_pno_tail)
    _pno_correction = bool(getattr(options, "pno_correction", True))
    e_pno_correction = 0.0
    U, eps_pno, pdata = {}, {}, {}
    residual_domain_sizes: list[int] = []
    # Extended/full contraction groups, keyed by (occupied coupling set,
    # extended atom domain). Every pair in a group sees the same coupled
    # amplitudes expanded into the same basis, so the group's residual is
    # contracted once per iteration and each member reads its own block.
    groups: dict[tuple, dict] = {}
    # The extended basis itself, its (ab|cd) DF factor and virtual Fock block
    # depend on the extended atom set alone, so they are shared by every
    # group with that domain whatever its coupling set.
    domains: dict[tuple, dict] = {}
    # Diagnostic (#190): the dimension of the "extended" domain on routes
    # that do not build it. P^T S P for P = Q_vir[:, domain] is a block of
    # Q_vir^T S Q_vir, so each distinct atom union costs one eigvalsh,
    # counted against lindep exactly as semicanonical_pao_basis counts it.
    _report_ext = bool(getattr(options, "report_extended_domain", False)) and (
        not _extended
    )
    extended_domain_sizes: list[int] = []
    _ext_rank_cache: dict[tuple[int, ...], int] = {}
    if _report_ext:
        _S_pao = Q_vir.T @ S @ Q_vir
        _S_pao = 0.5 * (_S_pao + _S_pao.T)

    def extended_domain_rank(L):
        ext = tuple(sorted(set().union(*(orbital_domain(m) for m in L))))
        n = _ext_rank_cache.get(ext)
        if n is None:
            idx = np.concatenate(
                [np.arange(atom_first[a], atom_first[a + 1]) for a in ext]
            )
            ev = np.linalg.eigvalsh(_S_pao[np.ix_(idx, idx)])
            n = int(np.count_nonzero(ev > options.lindep))
            _ext_rank_cache[ext] = n
        return n

    # ---- full local MP2 guess (pno_guess="lmp2") --------------------------
    # Intermediate PNOs from semicanonical amplitudes, cut at
    # lmp2_pno_scale * tcut_pno, for the CCSD pairs and the weak pairs above
    # lmp2_pair_cut; then the coupled LMP2 equations in those spaces. The
    # final PNOs below are the natural orbitals of these LMP2 amplitudes.
    lmp2_int: dict = {}
    _lmp2_sol_int = None
    if _lmp2_guess:
        from .lmp2_guess import pair_energy as _sc_pair_energy
        from .lmp2_guess import solve_local_lmp2

        _lmp2_solve_kw = dict(
            fock_cut=float(options.lmp2_fock_cut),
            max_iter=int(options.lmp2_max_iter),
            conv_tol_energy=float(options.lmp2_conv_tol_energy),
            conv_tol_residual=float(options.lmp2_conv_tol_residual),
            dense_max_gb=float(options.lmp2_dense_max_gb),
            C_vir=C_vir,
            S_ao=S,
        )
        _pair_cut = float(options.lmp2_pair_cut)
        _int_scale = float(options.lmp2_pno_scale)
        for i in range(n_act):
            for j in range(i, n_act):
                if (i, j) in screened_pairs and abs(e_sc_screened[(i, j)]) < _pair_cut:
                    continue
                atoms = np.where(orb_atom_mask[i] | orb_atom_mask[j])[0]
                _dom_key = tuple(int(a) for a in atoms)
                _cached = _pao_basis_cache.get(_dom_key)
                if _cached is None:
                    mask = np.zeros(nbf, dtype=bool)
                    for a in atoms:
                        mask[atom_first[a] : atom_first[a + 1]] = True
                    _cached = semicanonical_pao_basis(
                        F, S, Q_vir, np.where(mask)[0], options.lindep
                    )
                    _pao_basis_cache[_dom_key] = _cached
                V_semi, eps_pao = _cached
                if V_semi.shape[1] == 0:
                    continue
                Bi = B_half[:, i, :] @ V_semi
                Bj = Bi if i == j else B_half[:, j, :] @ V_semi
                K = Bi.T @ Bj
                T = K / (f_dd[i] + f_dd[j] - eps_pao[:, None] - eps_pao[None, :])
                delta = 1.0 if i == j else 0.0
                occ, d_all = np.linalg.eigh(pair_density(T, delta, pno_norm))
                order = np.argsort(-occ)
                occ, d_all = occ[order], d_all[:, order]
                _tcut_int = _int_scale * options.tcut_pno * (
                    float(getattr(options, "tcut_pno_diag_scale", 0.03))
                    if i == j
                    else 1.0
                )
                keep = (
                    occ > _tcut_int
                    if _tcut_int > 0.0
                    else np.ones_like(occ, dtype=bool)
                )
                if not np.any(keep):
                    keep[0] = True
                d = d_all[:, keep]
                Fp = d.T @ np.diag(eps_pao) @ d
                ep, ur = np.linalg.eigh(0.5 * (Fp + Fp.T))
                d = d @ ur
                K_int = d.T @ K @ d
                T_int = K_int / (f_dd[i] + f_dd[j] - ep[:, None] - ep[None, :])
                lmp2_int[(i, j)] = dict(
                    d=d,
                    K=K_int,
                    eps=ep,
                    U=CvS @ (V_semi @ d),
                    T=T_int,
                    e_sc_pao=_sc_pair_energy(K, T, i, j),
                    e_sc_int=_sc_pair_energy(K_int, T_int, i, j),
                )
        _keys = list(lmp2_int)
        _lmp2_sol_int = solve_local_lmp2(
            _keys,
            {k: lmp2_int[k]["K"] for k in _keys},
            {k: lmp2_int[k]["eps"] for k in _keys},
            {k: lmp2_int[k]["U"] for k in _keys},
            {k: lmp2_int[k]["T"] for k in _keys},
            f_oo,
            **_lmp2_solve_kw,
        )
        for k in _keys:
            lmp2_int[k]["T"] = _lmp2_sol_int.T[k]
            if k in screened_pairs:
                # A weak pair takes its LMP2 energy; its truncation to the
                # intermediate PNOs is corrected against the full-virtual
                # semicanonical energy it carried until now.
                e_pair_screened += (
                    _lmp2_sol_int.e_pair[k] - lmp2_int[k]["e_sc_int"]
                )

    if _lmp2_guess:
        _tick("lmp2_guess")
    U_tail, eps_tail = {}, {}
    T2 = {}
    for i in range(n_act):
        for j in range(i, n_act):
            if (i, j) in screened_pairs:
                continue
            atoms = np.where(orb_atom_mask[i] | orb_atom_mask[j])[0]
            mask = np.zeros(nbf, dtype=bool)
            for a in atoms:
                mask[atom_first[a] : atom_first[a + 1]] = True
            # Pairs sharing an atom domain share its semicanonical PAO basis.
            _dom_key = tuple(int(a) for a in atoms)
            _cached = _pao_basis_cache.get(_dom_key)
            if _cached is None:
                _cached = semicanonical_pao_basis(
                    F, S, Q_vir, np.where(mask)[0], options.lindep
                )
                _pao_basis_cache[_dom_key] = _cached
            V_semi, eps_pao = _cached
            if V_semi.shape[1] == 0:
                continue
            Bi = B_half[:, i, :] @ V_semi
            Bj = Bi if i == j else B_half[:, j, :] @ V_semi
            K = Bi.T @ Bj
            T = K / (f_dd[i] + f_dd[j] - eps_pao[:, None] - eps_pao[None, :])
            delta = 1.0 if i == j else 0.0
            if _lmp2_guess:
                # Natural orbitals of the LMP2 pair density, expressed in
                # the semicanonical PAO basis through the intermediate PNOs.
                _g = lmp2_int[(i, j)]
                occ, _q = np.linalg.eigh(pair_density(_g["T"], delta, pno_norm))
                order = np.argsort(-occ)
                occ, d_all = occ[order], _g["d"] @ _q[:, order]
            else:
                D = pair_density(T, delta, pno_norm)
                occ, d_all = np.linalg.eigh(D)
                order = np.argsort(-occ)
                occ, d_all = occ[order], d_all[:, order]
            _tcut_ij = options.tcut_pno * (
                float(getattr(options, "tcut_pno_diag_scale", 0.03))
                if i == j
                else 1.0
            )
            keep = (
                occ > _tcut_ij
                if _tcut_ij > 0.0
                else np.ones_like(occ, dtype=bool)
            )
            if not np.any(keep):
                keep[0] = True
            d = d_all[:, keep]
            Fp = d.T @ np.diag(eps_pao) @ d
            ep, ur = np.linalg.eigh(0.5 * (Fp + Fp.T))
            d = d @ ur
            V_pno = V_semi @ d
            Uij = CvS @ V_pno
            U[(i, j)] = Uij
            eps_pno[(i, j)] = ep
            if estimate_pno_tail and np.any(~keep):
                d_discard = d_all[:, ~keep]
                Fq = d_discard.T @ np.diag(eps_pao) @ d_discard
                eq, uq = np.linalg.eigh(0.5 * (Fq + Fq.T))
                V_discard = V_semi @ (d_discard @ uq)
                U_tail[(i, j)] = CvS @ V_discard
                eps_tail[(i, j)] = eq
            # Only the pair's own occupied rows are needed for its exchange
            # block; the legacy "pair" route transforms every coupled row
            # itself below.
            Bi_pno = B_ov[:, i, :] @ Uij
            Bj_pno = Bi_pno if i == j else B_ov[:, j, :] @ Uij
            # Restrict the occupied coupling space to this pair's local set L;
            # i, j re-index to local positions li, lj within it.
            L = coupled_occupieds(i, j)
            if _report_ext:
                extended_domain_sizes.append(extended_domain_rank(L))
            f_ov_ij = f_ov_full @ Uij
            nL, npno, naux = len(L), Uij.shape[1], B_ov.shape[0]
            pd = dict(
                f_oo=np.ascontiguousarray(f_oo[np.ix_(L, L)]),
                f_vv=np.ascontiguousarray(Uij.T @ f_vv_full @ Uij),
                f_ov=np.ascontiguousarray(f_ov_ij[L]),
                K=Bi_pno.T @ Bj_pno,
                f_ovi=f_ov_ij[i],
                L=L,
                li=int(np.searchsorted(L, i)),
                lj=int(np.searchsorted(L, j)),
            )
            if _pno_mode:
                # The compiled PNO-space solver builds its own factors from
                # the global B-tensors; nothing else per pair is needed.
                pass
            elif not _extended:
                # Legacy "pair" mode contracts in this pair's own PNO basis.
                B_ov_ij = np.einsum("Pmv,va->Pma", B_ov, Uij, optimize=True)
                B_vv_ij = np.einsum(
                    "Puv,ua,vb->Pab", B_vv, Uij, Uij, optimize=True
                )
                if _use_cpp:
                    # L-restricted PNO-basis DF B-tensors, flattened
                    # (n_aux x ...) row-major for the C++ kernel (it rebuilds
                    # the blocks).
                    pd["B_ov_L"] = np.ascontiguousarray(
                        B_ov_ij[:, L, :].reshape(naux, nL * npno)
                    )
                    pd["B_oo_L"] = np.ascontiguousarray(
                        B_oo[:, L][:, :, L].reshape(naux, nL * nL)
                    )
                    pd["B_vv_L"] = np.ascontiguousarray(
                        B_vv_ij.reshape(naux, npno * npno)
                    )
                else:
                    pd["V"] = _blocks(
                        B_ov_ij[:, L, :], B_vv_ij, B_oo[:, L][:, :, L]
                    )
            else:
                # Basis the residual is contracted in before being projected
                # back into this pair's PNO space, so that no coupled
                # amplitude is truncated before it has been contracted.
                #
                # "extended" is Riplinger and Neese's extended domain, defined
                # over ATOMS: the union of the orbital domains of i, j and
                # every coupled k, spanned by the PAOs on those atoms. It is
                # therefore independent of tcut_pno, which is the point: a
                # buffer built out of the truncated PNO spaces would be built
                # out of the very deficiency it exists to absorb. Taking the
                # union over the whole coupling set L (rather than the
                # paper's pair-list condition on k) makes it contain every
                # coupled pair's PAO domain, hence every coupled PNO space, so
                # the contraction reproduces "full" at the same L to lindep
                # precision.
                # "full" spans the whole virtual space unconditionally and is
                # the O(N^6) upper bound on the same idea.
                #
                # The basis, its DF factors and Fock blocks depend only on
                # (L, extended atoms): pairs sharing them share one group and
                # one contraction per iteration (#689).
                if _union_domain:
                    ext_atoms = tuple(
                        sorted(set().union(*(orbital_domain(m) for m in L)))
                    )
                else:
                    ext_atoms = tuple(range(natom))
                dom = domains.get(ext_atoms)
                if dom is None:
                    if _union_domain:
                        ext_mask = np.zeros(nbf, dtype=bool)
                        for a in ext_atoms:
                            ext_mask[atom_first[a] : atom_first[a + 1]] = True
                        V_ext_semi, _eps_ext = semicanonical_pao_basis(
                            F, S, Q_vir, np.where(ext_mask)[0], options.lindep
                        )
                        E = np.ascontiguousarray(CvS @ V_ext_semi)
                    else:
                        E = np.eye(f_vv_full.shape[0])
                    n_ext = int(E.shape[1])
                    B_vv_E = np.einsum(
                        "Puv,ua,vb->Pab", B_vv, E, E, optimize=True
                    )
                    dom = dict(
                        E=E,
                        n_ext=n_ext,
                        f_vv=np.ascontiguousarray(E.T @ f_vv_full @ E),
                    )
                    if _use_cpp:
                        # Flattened row-major (ab|cd) DF factor for the
                        # compiled kernel, which rebuilds the integral blocks
                        # itself and tiles the (ae|bf) ladder when n_ext^4
                        # would not fit in core.
                        dom["B_vv"] = np.ascontiguousarray(
                            B_vv_E.reshape(naux, n_ext * n_ext)
                        )
                    else:
                        dom["B_vv_E"] = B_vv_E
                    domains[ext_atoms] = dom
                E, n_ext = dom["E"], dom["n_ext"]
                gkey = (tuple(int(m) for m in L), ext_atoms)
                grp = groups.get(gkey)
                if grp is None:
                    B_ov_E = np.einsum(
                        "Pmv,va->Pma", B_ov[:, L, :], E, optimize=True
                    )
                    B_oo_L = B_oo[:, L][:, :, L]
                    grp = dict(
                        L=L,
                        domain=dom,
                        members=[],
                        f_oo=pd["f_oo"],
                        f_ov=np.ascontiguousarray((f_ov_full @ E)[L]),
                    )
                    if _use_cpp:
                        grp["B_ov_L"] = np.ascontiguousarray(
                            B_ov_E.reshape(naux, nL * n_ext)
                        )
                        grp["B_oo_L"] = np.ascontiguousarray(
                            B_oo_L.reshape(naux, nL * nL)
                        )
                    else:
                        # Built in the post-pass: whether the O(n_ext^4)
                        # (ab|cd) block is needed depends on this group's
                        # ladder decision, which needs its member list.
                        grp["_B"] = (B_ov_E, dom["B_vv_E"], B_oo_L)
                    groups[gkey] = grp
                grp["members"].append((i, j))
                pd["gkey"] = gkey
                # Extended basis -> this pair's PNO space.
                pd["P"] = np.ascontiguousarray(E.T @ Uij)

                residual_domain_sizes.append(int(n_ext))
            pdata[(i, j)] = pd
            T2[(i, j)] = (d.T @ K @ d) / (f_dd[i] + f_dd[j] - ep[:, None] - ep[None, :])
            if _lmp2_guess:
                # The correction and the starting amplitudes come from the
                # LMP2 solve in the final PNOs, after this loop.
                lmp2_int[(i, j)].update(d_fin=d, K_fin=d.T @ K @ d, eps_fin=ep)
            elif _pno_correction:
                # e_ij^full-PAO - e_ij^PNO, both semicanonical MP2 in this
                # pair's own domain, so the difference is purely what the
                # tcut_pno truncation discarded. Same convention as the pair
                # screening above and as DLPNOMP2's e_pno_correction:
                # w = 2 off-diagonal, e = w * sum(T * (2K - K^T)).
                _w = 2.0 if i != j else 1.0
                _e_full = _w * float(np.sum(T * (2.0 * K - K.T)))
                _K_pno = d.T @ K @ d
                _T_pno = T2[(i, j)]
                _e_trunc = _w * float(
                    np.sum(_T_pno * (2.0 * _K_pno - _K_pno.T))
                )
                e_pno_correction += _e_full - _e_trunc

    _tick("pair_setup")
    # ---- LMP2 in the final PNOs (pno_guess="lmp2") -------------------------
    # The CCSD pairs move to their final PNOs, the weak pairs keep their
    # intermediate ones. The LMP2 energy this loses is the truncation the
    # CCSD pairs suffer, and its amplitudes start the CCSD iterations.
    _lmp2_sol_fin = None
    if _lmp2_guess and T2:
        _keys = list(lmp2_int)
        _Kf, _ef, _Uf, _Tf = {}, {}, {}, {}
        for k in _keys:
            _g = lmp2_int[k]
            if k in T2:
                _R = _g["d"].T @ _g["d_fin"]
                _Kf[k], _ef[k], _Uf[k] = _g["K_fin"], _g["eps_fin"], U[k]
                _Tf[k] = _R.T @ _g["T"] @ _R
            else:
                _Kf[k], _ef[k], _Uf[k], _Tf[k] = _g["K"], _g["eps"], _g["U"], _g["T"]
        if _pno_correction:
            _lmp2_sol_fin = solve_local_lmp2(
                _keys, _Kf, _ef, _Uf, _Tf, f_oo, **_lmp2_solve_kw
            )
            _Tf = _lmp2_sol_fin.T
            e_pno_correction = (
                _lmp2_sol_int.e_corr
                - _lmp2_sol_fin.e_corr
                + sum(
                    lmp2_int[k]["e_sc_pao"] - lmp2_int[k]["e_sc_int"] for k in T2
                )
            )
        for k in T2:
            T2[k] = np.array(_Tf[k], copy=True)

    if _lmp2_guess:
        _tick("lmp2_guess")
    # ----- per-group ladder decision (#700) ------------------------------
    # Moving the particle-particle ladder into each pair's own PNO space trades
    # one O(n_ext^4) contraction per group for one O(naux n_pno^3) contraction
    # per pair, so it pays exactly when the extended basis is much wider than
    # the space the amplitudes occupy. Measured on the compiled path, speedup
    # against the locality ratio n_ext / mean(n_pno):
    #
    #     ratio 1.2 -> 0.33x   2.7 -> 0.80x   3.2 -> 1.04x
    #           3.8 -> 1.04x   5.1 -> 1.31x   6.5 -> 1.23x
    #
    # Break-even sits at ~3.2, and below it this is a REGRESSION (3x slower on
    # water/def2-SVP), so auto-enabling it everywhere would make the common
    # small-molecule case worse. Gate at 4.0, safely past break-even. An
    # explicit ladder_in_pno_space=True overrides the gate, which is how the
    # exactness tests exercise the route on systems too small to trip it.
    _LADDER_LOCALITY_GATE = 4.0
    for _grp in groups.values():
        _members = _grp["members"]
        _mean_pno = (
            float(np.mean([U[m].shape[1] for m in _members])) if _members else 0.0
        )
        _grp["ladder_in_pno"] = bool(
            _ladder_in_pno
            and _members
            and _mean_pno > 0.0
            and (
                _req_ladder is True
                or _grp["domain"]["n_ext"] >= _LADDER_LOCALITY_GATE * _mean_pno
            )
        )
        if not _use_cpp:
            _B_ov_E, _B_vv_E, _B_oo_L = _grp.pop("_B")
            _grp["V"] = _blocks(
                _B_ov_E, _B_vv_E, _B_oo_L,
                include_vvvv=not _grp["ladder_in_pno"],
            )
        if not _grp["ladder_in_pno"]:
            continue
        # Pair-space DF factors for the ladder, built once here rather than per
        # iteration: the "pair interaction" data of Riplinger and Neese 2013
        # Sec. II C.
        for _key in _members:
            _pd, _Uij = pdata[_key], U[_key]
            _pd["B_ov_lad"] = np.ascontiguousarray(
                np.einsum("Pmv,va->Pma", B_ov[:, _pd["L"], :], _Uij, optimize=True)
            )
            _pd["B_vv_lad"] = np.ascontiguousarray(
                np.einsum("Puv,ua,vb->Pab", B_vv, _Uij, _Uij, optimize=True)
            )
    _any_ladder_in_pno = any(g["ladder_in_pno"] for g in groups.values())

    # The occupied-half factor served only the pair loop above.
    del B_half
    # triples_mode="exact" re-transforms from the AO tensor in the canonical
    # occupied basis, so the DF object must survive for it.
    _exact_triples_needs_df = bool(options.compute_triples) and triples_mode == "exact"
    if bool(getattr(options, "release_df_tensors", False)) and not _exact_triples_needs_df:
        for _attr in ("B", "three_center"):
            if hasattr(df, _attr):
                try:
                    setattr(df, _attr, None)
                except AttributeError:  # pragma: no cover - read-only DF objects
                    pass
    _df_factors = (B_ov, B_oo, B_vv)

    _tick("pair_setup")
    result = LocalCCSDResult(e_hf=float(rhf.energy), n_frozen=nf)
    result.source_kind = str(source.kind)
    result.source_identity = str(source.identity)
    result.pno_norm = pno_norm
    result.pno_guess = "lmp2" if _lmp2_guess else "semicanonical"
    if _lmp2_sol_int is not None:
        result.n_lmp2_pairs = len(lmp2_int)
        result.e_lmp2 = float(_lmp2_sol_int.e_corr)
        _sols = [s for s in (_lmp2_sol_int, _lmp2_sol_fin) if s is not None]
        result.lmp2_iterations = tuple(int(s.n_iter) for s in _sols)
        result.lmp2_converged = all(s.converged for s in _sols)
        if not result.lmp2_converged:
            import warnings

            warnings.warn(
                "the coupled LMP2 guess did not converge in "
                f"lmp2_max_iter={options.lmp2_max_iter} iterations "
                f"(iterations {result.lmp2_iterations}); its PNOs and "
                "truncation correction are unconverged",
                RuntimeWarning,
                stacklevel=2,
            )
        # The intermediate data served only the PNO build and the solves.
        lmp2_int.clear()
    result.avg_residual_domain = (
        float(np.mean(residual_domain_sizes))
        if residual_domain_sizes
        else float(np.mean([U[k].shape[1] for k in U])) if U else 0.0
    )
    if _extended and residual_domain_sizes:
        result.avg_extended_domain = float(np.mean(residual_domain_sizes))
    elif extended_domain_sizes:
        result.avg_extended_domain = float(np.mean(extended_domain_sizes))
    result.n_residual_domains = len(groups)
    result.n_ladder_pno_groups = sum(
        1 for g in groups.values() if g.get("ladder_in_pno")
    )
    result.n_pairs = len(T2)
    result.n_screened = len(screened_pairs)
    result.pno_per_pair = {(k[0] + nf, k[1] + nf): U[k].shape[1] for k in U}
    result.avg_coupled_occ = (
        sum(len(pdata[k]["L"]) for k in T2) / len(T2) if T2 else 0.0
    )
    if not T2:
        result.converged = True
        # e_pno_correction accumulates in the loop that fills T2, so it is
        # necessarily 0.0 here; assigned anyway to keep the two exit paths
        # symmetric and the field always populated.
        result.e_pno_correction = e_pno_correction
        result.e_corr = e_pair_screened + e_pno_correction
        result.e_total = result.e_hf + result.e_corr
        result.phase_seconds = dict(_phases)
        return result

    t1 = {i: np.zeros(U[(i, i)].shape[1]) for i in range(n_act) if (i, i) in U}

    def Uof(k, l):
        if (k, l) in U:
            return U[(k, l)]
        if (l, k) in U:
            return U[(l, k)]
        # Screened pair: no PNO map -> use identity in pair's local space
        # (will be skipped by Tof returning zeros)
        return np.eye(0)

    def Tof(k, l):
        if (k, l) in T2:
            return T2[(k, l)]
        if (l, k) in T2:
            return T2[(l, k)].T
        # Screened pair: zero amplitude
        return np.zeros((0, 0))

    def energy():
        e = sum(2.0 * float(np.dot(t1[i], pdata[(i, i)]["f_ovi"])) for i in t1)
        for (i, j), T in T2.items():
            Uij = U[(i, j)]
            t1i = (Uij.T @ U[(i, i)]) @ t1[i] if (i, i) in U else np.zeros(Uij.shape[1])
            t1j = (Uij.T @ U[(j, j)]) @ t1[j] if (j, j) in U else np.zeros(Uij.shape[1])
            tau = T + np.outer(t1i, t1j)
            K = pdata[(i, j)]["K"]
            w = 1.0 if i == j else 2.0
            e += w * float(np.sum(tau * (2.0 * K - K.T)))
        return e

    def _project_amplitudes(V_proj, coupled_set):
        """Project amplitudes from per-pair PNO bases into V_proj.

        Uses the C++ batched kernel when available, falling back to
        numpy double-loops.
        """
        nL = len(coupled_set)
        n_T = V_proj.shape[1]
        if _use_cpp and _HAVE_CPP_PROJECT and nL > 0:
            # Remap actual occupied indices to [0, nL) for the C++ kernel.
            idx_map = {m: a for a, m in enumerate(coupled_set)}
            pi, pj, U_list, T2_list = [], [], [], []
            t1_keys, t1_vecs = [], []
            for m in coupled_set:
                if m in t1 and (m, m) in U:
                    t1_keys.append(idx_map[m])
                    # C++ kernel expects t1 already in full virtual space;
                    # expand from diagonal PNO basis through Uof(m,m).
                    t1_vecs.append(np.ascontiguousarray(Uof(m, m) @ t1[m]))
                for nn in coupled_set:
                    if (m, nn) in U or (nn, m) in U:
                        pi.append(idx_map[m])
                        pj.append(idx_map[nn])
                        U_list.append(np.asfortranarray(Uof(m, nn)))
                        T2_list.append(np.asfortranarray(Tof(m, nn)))
            t1f_flat, T2f_flat = _cpp_project_amplitudes(
                np.asfortranarray(V_proj), nL,
                pi, pj, U_list, T2_list,
                t1_keys, t1_vecs,
            )
            t1_out = np.asarray(t1f_flat)
            T2_out = np.asarray(T2f_flat).reshape(nL, nL, n_T, n_T)
            return t1_out, T2_out
        else:
            t1_out = np.zeros((nL, n_T))
            for a_, m in enumerate(coupled_set):
                if m in t1 and (m, m) in U:
                    t1_out[a_] = V_proj.T @ Uof(m, m) @ t1[m]
            T2_out = np.zeros((nL, nL, n_T, n_T))
            for a_, m in enumerate(coupled_set):
                for b_, nn in enumerate(coupled_set):
                    if (m, nn) in U or (nn, m) in U:
                        Sm = V_proj.T @ Uof(m, nn)
                        T2_out[a_, b_] = Sm @ Tof(m, nn) @ Sm.T
            return t1_out, T2_out

    # DIIS over the concatenated amplitude vector.
    amp_hist, res_hist = [], []

    def flatten():
        return np.concatenate(
            [t1[i].ravel() for i in sorted(t1)] + [T2[k].ravel() for k in sorted(T2)]
        )

    def unflatten(vec):
        off = 0
        for i in sorted(t1):
            n = t1[i].size
            t1[i] = vec[off : off + n].copy()
            off += n
        for k in sorted(T2):
            n = T2[k].size
            T2[k] = vec[off : off + n].reshape(T2[k].shape).copy()
            off += n

    def apply_residual(i, j, r1_pair, R2_pair, new_T2, new_t1):
        """Jacobi update of pair (i, j) from its residual in its PNO basis."""
        den = (
            f_dd[i] + f_dd[j] - eps_pno[(i, j)][:, None] - eps_pno[(i, j)][None, :]
        )
        new_T2[(i, j)] = T2[(i, j)] + R2_pair / den
        if i == j and i in t1:
            new_t1[i] = t1[i] + r1_pair / (f_dd[i] - eps_pno[(i, i)])

    # Per-iteration mixed-basis singles vectors for the #700 ladder. t1_i is
    # expanded in the DIAGONAL pair's PNOs, so the t1_i x t1_j part of tau_ij
    # is NOT inside pair (i,j)'s space; it is contracted in the mixed basis
    # instead. These fold t1 into the DF factors over the FULL virtual space
    # once per occupied per iteration, so nothing is truncated and no pair
    # pays an O(n_vir^2) build of its own.
    _lad_g, _lad_w = {}, {}

    def refresh_ladder_singles():
        _lad_g.clear()
        _lad_w.clear()
        for m in range(n_act):
            if m not in t1 or (m, m) not in U:
                continue
            t1_full = U[(m, m)] @ t1[m]                     # full virtual space
            _lad_g[m] = np.einsum("Pvw,w->Pv", B_vv, t1_full, optimize=True)
            _lad_w[m] = np.einsum("Pmv,v->Pm", B_ov, t1_full, optimize=True)

    def pair_space_ladder(i, j):
        """Particle-particle ladder for one pair, in its own PNO space (#700).

        Riplinger and Neese 2013 Sec. II C contract every residual term in the
        source pair's PNO space. For this term the source is the target pair:
        the contracted indices e,f are carried by tau_ij. Projecting the
        coupled tau^mn into this pair's space is exact here (unlike the ring
        terms, where it is the #98 truncation) because the free indices a,b
        are projected into this space regardless, and the pair's PNO space is
        a subspace of the group's extended basis, so projecting through that
        basis and projecting directly agree identically.
        """
        pd = pdata[(i, j)]
        L, li, lj = pd["L"], pd["li"], pd["lj"]
        Uij = U[(i, j)]
        naux_, nL_, npno_ = B_ov.shape[0], len(L), Uij.shape[1]
        t1L, t2L = _project_amplitudes(Uij, L)
        # a and b are FREE indices, projected into this pair's space either
        # way, so tau_L may be built after projection.
        tau_L = t2L + np.einsum("ma,nb->mnab", t1L, t1L, optimize=True)
        zu, zw = np.zeros((naux_, npno_)), np.zeros((naux_, nL_))
        u_i = _lad_g[i] @ Uij if i in _lad_g else zu
        v_j = _lad_g[j] @ Uij if j in _lad_g else zu
        w_i = _lad_w[i][:, L] if i in _lad_w else zw
        w_j = _lad_w[j][:, L] if j in _lad_w else zw
        return cs_ccsd_pair_ladder(
            t2L[li, lj], t1L, tau_L, pd["B_ov_lad"], pd["B_vv_lad"],
            u_i, v_j, w_i, w_j,
        )

    def ring_source_maps(grp):
        """Each coupled pair's PNO space expressed in THIS group's extended
        basis, built once per group (#700).

        A source pair may belong to a different group, so its own stored
        ``pdata[...]["P"]`` is in that group's basis and cannot be reused
        Both E and U are fixed for the whole run, so this product is loop
        invariant; computing it per iteration repeated an O(n_ext n_vir
        n_pno) matmul per (target pair, source pair) that cannot change.
        Hoisted on that ground alone -- its share of the runtime was not
        measured.
        """
        Qs = grp.get("_ring_Q")
        if Qs is None:
            E, Qs = grp["domain"]["E"], {}
            for m in grp["L"]:
                for n in grp["L"]:
                    key = (min(m, n), max(m, n))
                    if key in Qs:
                        continue
                    U_src = Uof(m, n)
                    if U_src.shape[1]:
                        Qs[key] = E.T @ U_src
            grp["_ring_Q"] = Qs
        return Qs

    def ring_half_in_pno_space(ti, tj, lti, ltj, Q_ij, grp, W):
        """Ring part of the Ph-symmetriser's `half` for the ORDERED pair
        (ti, tj), contracted in each SOURCE pair's PNO space (#700).

        Riplinger and Neese, J. Chem. Phys. 138, 034106 (2013), Eqs.
        (26)-(28). The ladder moved into the target pair's space because its
        contracted indices e,f are carried by tau_ij. These do not: the
        contracted index belongs to the SOURCE pair -- (ti, m) for the W1/W2
        terms, (m, tj) for WX -- so the amplitude is left in its own PNO
        space and only the free virtual index is carried into the target
        pair, by the Eq. (30) overlap

            S^{ij,kl} = U_ij^T U_kl,

        an ordinary matrix product here because the PNOs are expressed in the
        orthonormal canonical virtual basis, so the redundant-PAO metric that
        makes Eq. (30) awkward in the paper is the identity.

        Projecting the source amplitude into the target pair FIRST would look
        like the same thing and is not: that is the #98 truncation, and it
        drops the part of the coupled amplitude lying outside the target
        pair's space which still feeds back through the integrals.
        """
        W1E, W2E, WXE = W
        L = grp["L"]
        Qs = ring_source_maps(grp)
        out = np.zeros((Q_ij.shape[1], Q_ij.shape[1]))
        for lm, m in enumerate(L):
            Q = Qs.get((min(ti, m), max(ti, m)))    # source pair (ti, m)
            if Q is not None:
                t = Tof(ti, m)                      # [a', e'], its own space
                S_pq = Q.T @ Q_ij                   # Eq. (30)
                W1x = Q.T @ W1E[lm, :, ltj, :] @ Q_ij
                W2x = Q.T @ W2E[lm, :, ltj, :] @ Q_ij
                out += S_pq.T @ ((t - t.T) @ W1x + t @ W2x)
            Q = Qs.get((min(m, tj), max(m, tj)))    # source pair (m, tj)
            if Q is not None:
                t = Tof(m, tj)
                S_pq = Q.T @ Q_ij
                WXx = Q.T @ WXE[lm, :, lti, :] @ Q_ij
                out += S_pq.T @ (t @ WXx)
        return out

    def extended_group_residuals(grp):
        """Contract one (coupling set, extended domain) group's residual.

        Every coupled amplitude is expanded into the group's extended basis,
        the closed-shell CCSD residual is contracted there once, and each
        member pair's block is projected back into its own PNO space.
        Projecting first, as the "pair" path does, drops the components of
        the coupled amplitudes that lie outside the target pair's space but
        still feed back into it through the integrals. A group with a single
        member uses the target-selective kernel, which evaluates only that
        pair's residual block. Yields ``(i, j, r1_pair, R2_pair)``.
        """
        L, dom = grp["L"], grp["domain"]
        nL, n_ext = len(L), dom["n_ext"]
        t1E, t2E = _project_amplitudes(dom["E"], L)
        members = grp["members"]
        ring_W = None
        if _use_cpp:
            args = (
                np.ascontiguousarray(t1E),
                np.ascontiguousarray(t2E.reshape(nL * nL, n_ext * n_ext)),
                grp["B_ov_L"],
                grp["B_oo_L"],
                dom["B_vv"],
                grp["f_oo"],
                dom["f_vv"],
                grp["f_ov"],
            )
            cpp_kw = (
                {"include_ladder": False} if grp["ladder_in_pno"] else {}
            )
            if _ring_in_pno:
                # The kernel still BUILDS W1/W2/WX -- they are O(o^2 n_ext^2)
                # and cheap to hand back relative to the residual -- but does
                # not contract them; the source-space contraction happens
                # below (#700 b).
                cpp_kw["include_ring"] = False
                cpp_kw["return_ring"] = True
            if len(members) == 1 and _HAVE_CPP_TARGET_RESIDUAL:
                (i, j) = members[0]
                pd = pdata[(i, j)]
                res = _cpp_target_residual(
                    *args, pd["li"], pd["lj"], **cpp_kw
                )
                if _ring_in_pno:
                    R1t, R2t, ring_W = res
                    ring_W = tuple(
                        np.asarray(x).reshape(nL, n_ext, nL, n_ext)
                        for x in ring_W
                    )
                else:
                    R1t, R2t = res
                blocks = {
                    (i, j): (np.asarray(R1t).reshape(n_ext), np.asarray(R2t))
                }
            else:
                res = _cpp_pair_residual(*args, **cpp_kw)
                if _ring_in_pno:
                    R1E, R2f, ring_W = res
                    ring_W = tuple(
                        np.asarray(x).reshape(nL, n_ext, nL, n_ext)
                        for x in ring_W
                    )
                else:
                    R1E, R2f = res
                R1E = np.asarray(R1E)
                R2E = np.asarray(R2f).reshape(nL, nL, n_ext, n_ext)
                blocks = {
                    (i, j): (
                        R1E[pdata[(i, j)]["li"]],
                        R2E[pdata[(i, j)]["li"], pdata[(i, j)]["lj"]],
                    )
                    for (i, j) in members
                }
        else:
            out = cs_ccsd_residual(
                t1E,
                t2E,
                grp["f_oo"],
                dom["f_vv"],
                grp["f_ov"],
                grp["V"],
                include_ladder=not grp["ladder_in_pno"],
                include_ring=not _ring_in_pno,
                return_ring=_ring_in_pno,
            )
            if _ring_in_pno:
                R1E, R2E, ring_W = out
            else:
                (R1E, R2E), ring_W = out, None
            blocks = {
                (i, j): (
                    R1E[pdata[(i, j)]["li"]],
                    R2E[pdata[(i, j)]["li"], pdata[(i, j)]["lj"]],
                )
                for (i, j) in members
            }
        for (i, j), (r1E, R2E_ij) in blocks.items():
            P = pdata[(i, j)]["P"]
            R2_pair = P.T @ R2E_ij @ P
            if grp["ladder_in_pno"]:
                R2_pair = R2_pair + pair_space_ladder(i, j)
            if ring_W is not None:
                # The kernel left the ring terms out of `half`; put them back
                # in the source pairs' spaces, Ph-symmetrised exactly as
                # cs_ccsd_residual does it: R2 += half + half^T(i<->j, a<->b).
                li, lj = pdata[(i, j)]["li"], pdata[(i, j)]["lj"]
                R2_pair = (
                    R2_pair
                    + ring_half_in_pno_space(i, j, li, lj, P, grp, ring_W)
                    + ring_half_in_pno_space(j, i, lj, li, P, grp, ring_W).T
                )
            yield i, j, P.T @ r1E, R2_pair

    e_prev = energy()
    if _pno_mode:
        # Whole iteration in the compiled PNO-space solver. Pack the pair
        # data in T2's insertion order and unpack the converged amplitudes
        # into the same dicts the (T) and diagnostics below read.
        _keys = list(T2.keys())
        _naux = B_ov.shape[0]
        _res = _cpp_pno_solve(
            np.ascontiguousarray(B_ov.reshape(_naux, n_act * B_ov.shape[2])),
            np.ascontiguousarray(B_oo.reshape(_naux, n_act * n_act)),
            np.ascontiguousarray(B_vv.reshape(_naux, B_vv.shape[1] * B_vv.shape[2])),
            np.ascontiguousarray(f_oo),
            np.ascontiguousarray(f_ov_full),
            np.ascontiguousarray(f_vv_full),
            [int(k[0]) for k in _keys],
            [int(k[1]) for k in _keys],
            [np.asfortranarray(U[k]) for k in _keys],
            [np.ascontiguousarray(eps_pno[k]) for k in _keys],
            [np.asfortranarray(pdata[k]["K"]) for k in _keys],
            [np.asfortranarray(T2[k]) for k in _keys],
            [[int(m) for m in pdata[k]["L"]] for k in _keys],
            max_iter=int(options.max_iter),
            conv_tol_energy=float(options.conv_tol_energy),
            conv_tol_residual=float(options.conv_tol_residual),
            diis_size=int(options.diis_size),
        )
        for _idx, _k in enumerate(_keys):
            T2[_k] = np.ascontiguousarray(np.asarray(_res["T2"][_idx]))
        for _i in list(t1):
            _v = np.asarray(_res["t1"][_i])
            if _v.size:
                t1[_i] = np.ascontiguousarray(_v).reshape(-1)
        _e_prev_trace = e_prev
        for _it, (_e, _r) in enumerate(
            zip(_res["trace_e_corr"], _res["trace_r_norm"])
        ):
            result.trace.append(
                {
                    "iter": _it + 1,
                    "e_corr": float(_e),
                    "delta_e": float(_e) - _e_prev_trace,
                    "r_norm": float(_r),
                }
            )
            _e_prev_trace = float(_e)
        result.converged = bool(_res["converged"])
        result.n_iter = int(_res["n_iter"])
        result.n_offlist_rows = int(_res.get("n_offlist_rows", 0))
        result.static_factor_bytes = int(_res.get("static_factor_bytes", 0))
        result.setup_seconds = float(_res.get("setup_seconds", 0.0))
        result.iteration_seconds = float(_res.get("iteration_seconds", 0.0))
    for iteration in range(0 if _pno_mode else options.max_iter):
        x_old = flatten()  # amplitudes at iteration start (post previous DIIS)
        new_T2, new_t1 = {}, {}
        if _extended:
            if _any_ladder_in_pno:
                refresh_ladder_singles()
            for grp in groups.values():
                for i, j, r1_pair, R2_pair in extended_group_residuals(grp):
                    apply_residual(i, j, r1_pair, R2_pair, new_T2, new_t1)
        for i, j in ([] if _extended else T2):
            # Legacy "pair" mode: contract in this pair's own PNO basis.
            Uij = U[(i, j)]
            n = Uij.shape[1]
            pd = pdata[(i, j)]
            L, li, lj = pd["L"], pd["li"], pd["lj"]
            nL = len(L)
            t2L = np.zeros((nL, nL, n, n))
            for a_, m in enumerate(L):
                for b_, nn in enumerate(L):
                    if not ((m, nn) in U or (nn, m) in U):
                        continue
                    Sm = Uij.T @ Uof(m, nn)
                    t2L[a_, b_] = Sm @ Tof(m, nn) @ Sm.T
            t1L = np.zeros((nL, n))
            for a_, m in enumerate(L):
                if m in t1 and (m, m) in U:
                    t1L[a_] = (Uij.T @ U[(m, m)]) @ t1[m]
            if _use_cpp:
                _args = (
                    np.ascontiguousarray(t1L),
                    np.ascontiguousarray(t2L.reshape(nL * nL, n * n)),
                    pd["B_ov_L"],
                    pd["B_oo_L"],
                    pd["B_vv_L"],
                    pd["f_oo"],
                    pd["f_vv"],
                    pd["f_ov"],
                )
                if _HAVE_CPP_TARGET_RESIDUAL:
                    R1_target, R2_target = _cpp_target_residual(
                        *_args, li, lj
                    )
                    r1_pair = np.asarray(R1_target).reshape(n)
                    R2_pair = np.asarray(R2_target)
                else:
                    R1L, R2f = _cpp_pair_residual(*_args)
                    R1L = np.asarray(R1L)
                    r1_pair = R1L[li]
                    R2_pair = np.asarray(R2f).reshape(nL, nL, n, n)[li, lj]
            else:
                R1L, R2L = cs_ccsd_residual(
                    t1L, t2L, pd["f_oo"], pd["f_vv"], pd["f_ov"], pd["V"]
                )
                r1_pair = R1L[li]
                R2_pair = R2L[li, lj]
            apply_residual(i, j, r1_pair, R2_pair, new_T2, new_t1)
        for k, T in new_T2.items():
            T2[k] = 0.5 * (T + T.T) if k[0] == k[1] else T
        for i, t in new_t1.items():
            t1[i] = t

        amp_hist.append(flatten())
        res_hist.append(amp_hist[-1] - x_old)  # = R/D, the DIIS error vector
        if len(amp_hist) > options.diis_size:
            amp_hist.pop(0)
            res_hist.pop(0)
        nh = len(amp_hist)
        if nh >= 2:
            Bm = np.full((nh + 1, nh + 1), -1.0)
            Bm[-1, -1] = 0.0
            for a_ in range(nh):
                for b_ in range(nh):
                    Bm[a_, b_] = float(np.dot(res_hist[a_], res_hist[b_]))
            rhs = np.zeros(nh + 1)
            rhs[-1] = -1.0
            try:
                c = np.linalg.solve(Bm, rhs)[:nh]
                unflatten(sum(c[k] * amp_hist[k] for k in range(nh)))
            except np.linalg.LinAlgError:
                pass

        e = energy()
        r_norm = float(np.max(np.abs(res_hist[-1])))  # max |R/D| (pre-DIIS)
        result.trace.append(
            {
                "iter": iteration + 1,
                "e_corr": e,
                "delta_e": e - e_prev,
                "r_norm": r_norm,
            }
        )
        if (
            iteration > 0
            and abs(e - e_prev) < options.conv_tol_energy
            and r_norm < options.conv_tol_residual
        ):
            result.converged = True
            result.n_iter = iteration + 1
            break
        e_prev = e
    else:
        if not _pno_mode:
            result.n_iter = options.max_iter

    def pno_tail_estimate():
        if not result.converged or not estimate_pno_tail or not U_tail:
            return 0.0
        e_tail = 0.0
        for (i, j), U_discard in U_tail.items():
            if (i, j) not in T2 or U_discard.size == 0:
                continue
            Uij = U[(i, j)]
            Uaug = np.ascontiguousarray(np.hstack([Uij, U_discard]))
            n_keep = Uij.shape[1]
            n_aug = Uaug.shape[1]
            pd = pdata[(i, j)]
            L, li, lj = pd["L"], pd["li"], pd["lj"]
            nL = len(L)

            B_ov_aug = np.einsum("Pmv,va->Pma", B_ov, Uaug, optimize=True)
            B_vv_aug = np.einsum("Puv,ua,vb->Pab", B_vv, Uaug, Uaug, optimize=True)
            f_ov_aug = f_ov_full @ Uaug
            f_vv_aug = np.ascontiguousarray(Uaug.T @ f_vv_full @ Uaug)

            t2L = np.zeros((nL, nL, n_aug, n_aug))
            for a_, m in enumerate(L):
                for b_, nn in enumerate(L):
                    if not ((m, nn) in U or (nn, m) in U):
                        continue
                    Sm = Uaug.T @ Uof(m, nn)
                    t2L[a_, b_] = Sm @ Tof(m, nn) @ Sm.T
            t1L = np.zeros((nL, n_aug))
            for a_, m in enumerate(L):
                if m in t1 and (m, m) in U:
                    t1L[a_] = (Uaug.T @ U[(m, m)]) @ t1[m]

            if _use_cpp:
                naux = B_ov_aug.shape[0]
                R1L, R2f = _cpp_pair_residual(
                    np.ascontiguousarray(t1L),
                    np.ascontiguousarray(t2L.reshape(nL * nL, n_aug * n_aug)),
                    np.ascontiguousarray(B_ov_aug[:, L, :].reshape(naux, nL * n_aug)),
                    np.ascontiguousarray(B_oo[:, L][:, :, L].reshape(naux, nL * nL)),
                    np.ascontiguousarray(B_vv_aug.reshape(naux, n_aug * n_aug)),
                    pd["f_oo"],
                    f_vv_aug,
                    np.ascontiguousarray(f_ov_aug[L]),
                )
                R2L = np.asarray(R2f).reshape(nL, nL, n_aug, n_aug)
            else:
                Vaug = _blocks(B_ov_aug[:, L, :], B_vv_aug, B_oo[:, L][:, :, L])
                _, R2L = cs_ccsd_residual(
                    t1L,
                    t2L,
                    pd["f_oo"],
                    f_vv_aug,
                    np.ascontiguousarray(f_ov_aug[L]),
                    Vaug,
                )

            eps_aug = np.concatenate([eps_pno[(i, j)], eps_tail[(i, j)]])
            den = f_dd[i] + f_dd[j] - eps_aug[:, None] - eps_aug[None, :]
            dT = R2L[li, lj] / den
            dT[:n_keep, :n_keep] = 0.0
            if i == j:
                dT = 0.5 * (dT + dT.T)
            K_aug = B_ov_aug[:, i, :].T @ B_ov_aug[:, j, :]
            w = 1.0 if i == j else 2.0
            e_tail += w * float(np.sum(dT * (2.0 * K_aug - K_aug.T)))
        return float(e_tail)

    result.e_pno_tail_estimate = pno_tail_estimate()
    result.e_pno_correction = e_pno_correction
    result.e_corr = energy() + e_pair_screened + e_pno_correction
    _tick("amplitudes")
    if options.compute_triples:
        # Collected by the TNO builders so a truncated run can report how far
        # it truncated; "exact" builds no TNO domain and leaves it empty.
        tno_stats: dict = {}
        if triples_mode == "exact":
            from .triples_local import exact_triples_correction

            result.e_t = exact_triples_correction(
                U,
                T2,
                t1,
                C_act,
                C_loc,
                C_vir,
                S,
                F,
                f_vv_full,
                df,
                n_act,
            )
        elif triples_mode == "t1":
            from .triples_local import t1_triples_correction

            result.e_t = t1_triples_correction(
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
                tcut_tno=options.tcut_tno,
                occ_dist=occ_dist,
                coupling_radius=R_couple,
                stats=tno_stats,
                df_factors=_df_factors,
            )
        else:
            from .triples_local import local_triples_correction

            result.e_t = local_triples_correction(
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
                tcut_tno=options.tcut_tno,
                occ_dist=occ_dist,
                coupling_radius=R_couple,
                stats=tno_stats,
                df_factors=_df_factors,
            )
        from .triples_local import summarise_tno_domains

        result.tno_per_triple = dict(tno_stats.get("tno_per_triple", {}))
        result.n_triple_keys = int(tno_stats.get("n_triple_keys", 0))
        result.n_triple_keys_screened = int(
            tno_stats.get("n_triple_keys_screened", 0)
        )
        result.avg_tno, result.n_degenerate_tno_triples = summarise_tno_domains(
            tno_stats
        )
        result.triples_executed = True
        _tick("triples")
    result.e_total = result.e_hf + result.e_corr + result.e_t
    result.t1_norm = float(np.sqrt(sum(float(np.dot(t, t)) for t in t1.values())))
    result.phase_seconds = dict(_phases)
    return result
