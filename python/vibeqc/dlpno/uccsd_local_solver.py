"""Local-domain open-shell DLPNO-UCCSD solver.

This is the reduced-scaling sibling of :mod:`vibeqc.dlpno.uccsd`'s dense
O(N^6) correctness pilot.  Each occupied pair owns a PNO basis, amplitudes
from neighbouring pairs are projected into that basis, and the native
``dlpno_uccsd_pair_residual`` kernel evaluates one bounded spin-orbital
domain at a time. The CCSD phase forms no global four-index
electron-repulsion tensor. The opt-in ``triples_mode="t1-iterative"``
currently materializes dense spin-orbital triples source blocks and is an
experimental, ``max_nbf``-capped correctness/generalization path rather than
a reduced-scaling triples implementation.

The full-domain limit is the correctness ratchet: with ``tcut_mkn=0``,
``tcut_pno=0``, ``tcut_pno_singles=0``, ``tcut_pairs=0``, and
``tcut_tno=0``, and ``coupling_radius=0`` every virtual and occupied is
retained, so the local projections are unitary and the result reproduces the
dense pilot. PAO/PNO, singles-PNO, weak-pair, TNO, and occupied-coupling
projections become exact in that limit. Pair, PNO, and PAO-domain truncation
use the shared NormalPNO default; singles-PNO, TNO, and occupied-coupling
truncation remain opt-in while broader radical benchmarks are accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..solvers._scf import solve_dense_fock
from .pao import (
    build_atom_basis_map,
    build_projection_matrix,
    orbital_domain_atom_mask,
    semicanonical_pao_basis,
)
from .uccsd import _boys_localise

#: Route-specific basis-function guards applied when ``max_nbf`` is None.
_PNO_ROUTE_MAX_NBF = 1500
_UNION_ROUTE_MAX_NBF = 160


@dataclass
class LocalUCCSDOptions:
    """Controls for the local-domain open-shell UCCSD solver.

    ``tcut_pairs=0`` keeps every pair in the CCSD iteration. A positive value
    screens pairs whose full-virtual MP2 estimate is smaller in magnitude than
    the threshold and retains their MP2 energy. ``tcut_mkn=0`` keeps the full
    canonical virtual space; a positive value selects pair atoms by the
    Mulliken populations of the two occupied orbitals and builds separate
    alpha/beta semicanonical PAOs on their union. ``tcut_pno_singles=0`` keeps
    every singles amplitude in the full spin-virtual space. A positive value
    builds occupied-specific, same-spin natural orbitals from the sum of that
    occupied's MP2 pair densities, optionally inside its Mulliken-selected PAO
    domain. ``coupling_radius=0`` keeps the full occupied coupling set and all
    triples. ``compute_triples=False`` leaves the perturbative triples path
    off; when enabled, ``tcut_tno=0`` keeps the full union of the three pair
    spaces for each distinct occupied triple. On the OH/def2-SVP calibration
    under `(T1)`, positive ``tcut_tno`` raises the energy by +12 uHa at
    ``1e-9`` (the ``T_CutTNO`` default of Guo et al. 2018), rising to +759 uHa
    at ``1e-4``; ``1e-12`` already changes it by +11 uHa. These measurements
    are calibration behavior, not a universal sign or monotonicity guarantee
    for a nonvariational triples correction.
    Watch ``LocalUCCSDResult.n_degenerate_tno_triples``: a triple retaining
    fewer than three spin virtuals cannot host a triple excitation at all, so a
    nonzero count means the correction has been truncated out of existence
    rather than approximated. The pair/PNO/domain defaults form the shared
    NormalPNO policy; the open-shell radical benchmark evidence remains less
    extensive than for closed-shell routes. ``triples_mode="t0"`` uses the
    diagonal localised occupied Fock, while the opt-in ``"t1"`` mode restores
    its off-diagonal coupling by spin-block semicanonicalisation. ``"t1"``
    rotates the occupied indices out of the localised basis and so cannot carry
    a screened triple list: it fails closed when ``tcut_pairs`` or
    ``coupling_radius`` actually screens a triple, and runs the exact full
    triple list when they do not. The singles, TNO, and coupling controls
    remain outside the published three-threshold convention.
    """

    localise: str = "boys"
    n_frozen: int | None = None
    # This operational route consumes all three members of the Liakos 2015
    # NormalPNO cross-route project policy. Pinski's MP2-specific threshold
    # set is not used for UCCSD. Explicit zero values retain the exactness
    # settings for callers that need full domains and no pair screening.
    tcut_pno: float = 3.33e-7
    tcut_pairs: float = 1e-4
    #: Factor on ``tcut_pairs`` for the spin-orbital pairs this route
    #: screens. The published ``TCutPairs`` values are calibrated on spatial
    #: pair energies: closed-shell pairs, and in ORCA's open-shell DLPNO the
    #: spin-summed ``e_ij = e_aa + e_bb + 2 e_ab`` (Saitow, Becker, Riplinger,
    #: Valeev and Neese, J. Chem. Phys. 146, 164105 (2017), Eq. 53). A
    #: spin-orbital pair carries only part of that, so at ``1.0`` the same
    #: threshold screens several times more loosely. Measured on all-trans
    #: C6H14/cc-pVDZ through both routes on one SCF, NormalPNO CCSD sits
    #: +0.652 kcal/mol above canonical on the closed-shell route, +0.263 on
    #: this route at ``1.0`` and +0.693 at ``0.25``; on the C6H13 radical,
    #: NormalPNO gives +0.235 at ``1.0`` and +0.631 at ``0.25``. The lower
    #: errors at ``1.0`` are cancellation: the extra screened pairs keep
    #: their full-virtual MP2 energies, which over-correlate. The default
    #: ``0.25`` brings this route in line with the closed-shell one; ``1.0``
    #: reproduces every open-shell result recorded before it (#190).
    tcut_pairs_spin_scale: float = 0.25
    tcut_mkn: float = 1e-3
    #: PAO atom-domain criterion, as for the closed-shell routes: ``"doi"``
    #: (default; differential overlap above ``tcut_do``, Pinski et al. 2015)
    #: or ``"mulliken"`` (population above ``tcut_mkn``). Domains are chosen
    #: per spin orbital against that spin's projected atomic orbitals; a
    #: pair's domain is the union of its two orbitals' domains. ``tcut_mkn=0``
    #: keeps every atom under either criterion.
    domain_criterion: str = "doi"
    tcut_do: float = 1e-2
    #: Add the semicanonical MP2 correction for the PNO truncation of the
    #: iterated pairs, ``sum_pq (e_pq^PAO - e_pq^PNO)``, as the closed-shell
    #: DLPNO-MP2 and local CCSD routes do (#222). Zero at ``tcut_pno=0``.
    pno_correction: bool = True
    #: Density whose eigenvalues tcut_pno cuts, per spin-orbital pair.
    #:
    #: * "two-sided" (default): every pair's PNO space spans the virtual
    #:   directions either occupied index needs, as the closed-shell
    #:   Riplinger-Neese density does. An opposite-spin pair adds each spin
    #:   side's density mapped into the other spin's virtual space through the
    #:   alpha-beta virtual overlap; a same-spin pair adds the two-sided
    #:   density of its direct (Coulomb-only) amplitude, because the
    #:   antisymmetrised MP2 amplitude vanishes wherever it is locally
    #:   symmetric while the CCSD amplitude does not.
    #: * "one-sided": the antisymmetrised amplitude's T T^T + T^T T
    #:   alone, the convention of every result recorded before this option.
    #:
    #: A singlet run through this solver at NormalPNO then matches the
    #: closed-shell route: H2O/cc-pVDZ CCSD 0.022 kcal/mol above canonical
    #: (closed-shell solver 0.016; one-sided 0.274). Radicals: OH 0.279 ->
    #: 0.0025, NH2 0.350 -> 0.0125, CH2 (triplet) 0.197 -> 0.0099, HCO
    #: 0.712 -> 0.071 kcal/mol, for about 25 % more PNOs per pair.
    pno_density: str = "two-sided"
    tcut_pno_singles: float = 0.0
    compute_triples: bool = False
    tcut_tno: float = 0.0
    triples_mode: str = "t0"
    lindep: float = 1e-8
    coupling_radius: float = 0.0
    max_iter: int = 100
    conv_tol_energy: float = 1e-9
    conv_tol_residual: float = 1e-7
    diis_size: int = 6
    #: Residual contraction engine.
    #:
    #: * "pno" (default): the compiled kernel
    #:   ``vibeqc._vibeqc_core.dlpno_uccsd_pno_solve`` contracts every
    #:   spin-orbital residual term in the PNO space of the amplitude that
    #:   carries the contracted virtual index (Riplinger and Neese 2013,
    #:   Sec. II C), with the singles in the full canonical virtual space of
    #:   their spin (or their singles-PNO space). The DF factors stay
    #:   spin-blocked and the pair spaces are built per spin block, so no
    #:   combined spin-orbital tensor is formed for the CCSD. Exact with
    #:   respect to the occupied coupling set: at ``tcut_pno=0`` with full
    #:   domains it reproduces canonical UCCSD.
    #: * "union": the earlier Python route. Each residual is contracted in the
    #:   union of its coupling set's pair spaces by the dense spin-orbital
    #:   domain kernel ``dlpno_uccsd_pair_residual``, which forms every
    #:   antisymmetrised integral of that union; singles entering a pair
    #:   residual are projected into the union. Small systems only.
    residual_domain: str = "pno"
    #: Basis-function guard. ``None`` selects the route's own limit: 1500 on
    #: the compiled "pno" route and 160 on the Python "union" route.
    max_nbf: int | None = None
    # RI fitting basis for the correlation integrals. None lets run_job
    # resolve the orbital-basis default before preflight and execution.
    aux_basis: str | None = None


@dataclass
class LocalUCCSDResult:
    #: Adapter name and SHA-256 identity of the correlation source the
    #: engine read (:mod:`vibeqc.dlpno.source`).
    source_kind: str = ""
    source_identity: str = ""
    e_hf: float = 0.0
    e_corr: float = 0.0
    e_t: float = 0.0
    e_total: float = 0.0
    n_iter: int = 0
    converged: bool = False
    n_pairs: int = 0
    n_frozen: int = 0
    localise: str = "boys"
    triples_mode: str = "t0"
    t1_norm: float = 0.0
    avg_pno: float = 0.0
    avg_pao: float = 0.0
    avg_singles_domain: float = 0.0
    avg_tno: float = 0.0
    avg_coupled_occ: float = 0.0
    n_screened: int = 0
    n_triples: int = 0
    n_screened_triples: int = 0
    #: Evaluated triples whose TNO domain retained fewer than three spin
    #: virtuals.  A triple excitation needs three distinct virtuals, so such a
    #: triple contributes exactly zero by construction rather than
    #: approximately: a nonzero count means ``tcut_tno`` has truncated past the
    #: point where the correction exists at all, not that it is small.
    n_degenerate_tno_triples: int = 0
    e_screened_mp2: float = 0.0
    #: Semicanonical MP2 correction for the PNO truncation of the iterated
    #: pairs, included in ``e_corr`` when ``pno_correction`` is on.
    e_pno_correction: float = 0.0
    #: Distinct residual contractions per iteration (pair groups plus
    #: singles groups); each serves every member of its group.
    n_residual_groups: int = 0
    #: Residual engine that ran ("pno" or "union").
    residual_domain: str = ""
    #: Compiled "pno" route bookkeeping: factor rows stored outside a pair's
    #: own coupling set, resident static factor bytes, and wall seconds of
    #: the static-factor build and of the amplitude iterations.
    n_offlist_rows: int = 0
    static_factor_bytes: int = 0
    setup_seconds: float = 0.0
    iteration_seconds: float = 0.0
    #: Wall-clock seconds per phase of the whole route, in the order run:
    #: ``reference``, ``localise``, ``domains``, ``factors``, ``pairs``
    #: (screening energies and pair PNO spaces), ``singles``, ``amplitudes``
    #: (including the compiled setup) and ``triples``.
    phase_seconds: dict = field(default_factory=dict)
    pno_per_pair: dict[tuple[int, int], int] = field(default_factory=dict)
    pao_per_pair: dict[tuple[int, int], int] = field(default_factory=dict)
    singles_per_occ: dict[int, int] = field(default_factory=dict)
    tno_per_triple: dict[tuple[int, int, int], int] = field(
        default_factory=dict
    )
    screened_pair_energies: dict[tuple[int, int], float] = field(
        default_factory=dict
    )
    trace: list[dict[str, float | int]] = field(default_factory=list)
    #: True only when the requested triples contraction was actually entered.
    #: Non-converged CCSD or insufficient occupied space leaves it false.
    triples_executed: bool = False


def _domain_from_density(
    density: np.ndarray,
    f_vv: np.ndarray,
    threshold: float,
    *,
    fock_basis_id: str | None = None,
    spin: str = "restricted",
) -> tuple[np.ndarray, np.ndarray]:
    """Build and quasi-canonicalise a virtual domain from a PSD density.

    A named Fock frame opts the caller into the shared checked solve.
    Unnamed callers retain their existing boundary pending acceptance.
    Density selection and its largest-occupation fallback are independent
    of the retained Fock solve.
    """
    occ, vectors = np.linalg.eigh(0.5 * (density + density.T))
    order = np.argsort(-occ)
    occ = occ[order]
    vectors = vectors[:, order]
    keep = (
        occ > threshold
        if threshold > 0.0
        else np.ones_like(occ, dtype=bool)
    )
    if not np.any(keep):
        keep[0] = True
    vectors = vectors[:, keep]
    f_domain = vectors.T @ f_vv @ vectors
    f_domain = 0.5 * (f_domain + f_domain.T)
    if fock_basis_id is None:
        eps, rotation = np.linalg.eigh(f_domain)
        return np.ascontiguousarray(vectors @ rotation), eps
    solved = solve_dense_fock(f_domain, spin=spin, basis_id=fock_basis_id)
    # Preserve the NumPy-owned rotation buffer when lifting the retained
    # space; immutable solve buffers can change multiplication roundoff.
    rotation = solved.eigenvectors.copy()
    eps = solved.eigenvalues
    del solved  # Release the snapshot before lifting the retained space.
    return np.ascontiguousarray(vectors @ rotation), eps


def _occupied_triples_rotation(f_oo, n_alpha, *, source_identity, n_frozen):
    """Canonicalise active occupied spins without mixing their coordinates.

    This is the existing rotated-occupied compatibility correction, not the
    ROHF/QRO construction of Saitow et al., JCP 146, 164105 (2017),
    DOI 10.1063/1.4981521, Sec. II A.
    The caller has already removed frozen columns and admitted its source.
    No operator or warm subspace survives this synchronous dense solve.
    """
    no = f_oo.shape[0]
    rotation = np.zeros((no, no))
    energies = np.zeros(no)
    for spin, occupied_slice in (("alpha", slice(0, n_alpha)),
                                 ("beta", slice(n_alpha, no))):
        f_spin = f_oo[occupied_slice, occupied_slice]
        if not f_spin.size:
            continue  # A legitimate empty minority-spin occupied space.
        solved = solve_dense_fock(
            0.5 * (f_spin + f_spin.T), spin=spin,
            basis_id=(f"dlpno-uccsd:{source_identity}:frozen={n_frozen}:"
                      f"active-occupied:{spin}"),
        )
        rotation[occupied_slice, occupied_slice] = solved.eigenvectors
        energies[occupied_slice] = solved.eigenvalues
        del solved  # Release the block solve before processing the next spin.
    return energies, rotation


def run_local_dlpno_uccsd(
    molecule,
    basis,
    uhf,
    df,
    options: LocalUCCSDOptions | None = None,
    *,
    source=None,
) -> LocalUCCSDResult:
    """Run local-domain DLPNO-UCCSD on a converged UHF reference.

    This is the reduced-scaling production engine selected by ``run_job`` for
    open-shell ``dlpno-ccsd`` and ``dlpno-ccsd(t)`` calculations. Local PNO
    assembly and iteration remain ratcheted against the independent dense
    ``run_dlpno_uccsd_pilot`` oracle at the full-domain limit.
    """
    from vibeqc._vibeqc_core import (
        compute_overlap,
        dlpno_spin_orbital_triple_energy,
        dlpno_uccsd_pair_residual,
    )

    if options is None:
        options = LocalUCCSDOptions()
    from .source import CorrelationSource, molecular_source

    if source is None:
        source = molecular_source(molecule, basis, uhf, df)
    elif not isinstance(source, CorrelationSource):
        raise TypeError(
            "source must be a vibeqc.dlpno.source.CorrelationSource, got "
            f"{type(source).__name__}"
        )
    source.validate_reference()
    molecule, basis = source.molecule, source.basis
    uhf, df = source.reference, source.factors
    from vibeqc.correlation_conventions import effective_electron_count
    if options.localise not in ("none", "boys", "external"):
        raise ValueError(f"unknown localise option: {options.localise!r}")
    triples_mode = str(options.triples_mode).lower()
    if triples_mode not in ("t0", "t1", "t1-iterative"):
        raise ValueError(f"unknown triples_mode: {options.triples_mode!r}")
    if options.tcut_pno < 0.0:
        raise ValueError("tcut_pno must be non-negative")
    if options.tcut_pairs < 0.0:
        raise ValueError("tcut_pairs must be non-negative")
    _pair_scale = float(getattr(options, "tcut_pairs_spin_scale", 1.0))
    if not (_pair_scale > 0.0):
        raise ValueError("tcut_pairs_spin_scale must be positive")
    # Spin-orbital pair screening threshold (see tcut_pairs_spin_scale).
    tcut_pairs_so = float(options.tcut_pairs) * _pair_scale
    if options.tcut_mkn < 0.0:
        raise ValueError("tcut_mkn must be non-negative")
    if options.tcut_pno_singles < 0.0:
        raise ValueError("tcut_pno_singles must be non-negative")
    if options.tcut_tno < 0.0:
        raise ValueError("tcut_tno must be non-negative")
    if options.lindep < 0.0:
        raise ValueError("lindep must be non-negative")
    residual_route = str(getattr(options, "residual_domain", "pno")).strip().lower()
    if residual_route not in ("pno", "union"):
        raise ValueError(
            f"unknown residual_domain: {options.residual_domain!r} "
            '(expected "pno" or "union")'
        )
    if residual_route == "pno":
        try:
            from vibeqc._vibeqc_core import dlpno_uccsd_pno_solve
        except ImportError as exc:  # pragma: no cover - stale native core
            raise RuntimeError(
                'residual_domain="pno" needs vibeqc._vibeqc_core.'
                "dlpno_uccsd_pno_solve; rebuild the native core, or select "
                'residual_domain="union" for small systems'
            ) from exc

    import time as _time

    _phases: dict[str, float] = {}
    _phase_mark = [_time.perf_counter()]

    def _tick(name):
        now = _time.perf_counter()
        _phases[name] = _phases.get(name, 0.0) + (now - _phase_mark[0])
        _phase_mark[0] = now

    ca = np.asarray(uhf.mo_coeffs_alpha)
    cb = np.asarray(uhf.mo_coeffs_beta)
    fa_ao = np.asarray(uhf.fock_alpha)
    fb_ao = np.asarray(uhf.fock_beta)
    nbf = ca.shape[0]
    max_nbf = (
        int(options.max_nbf)
        if options.max_nbf is not None
        else (
            _PNO_ROUTE_MAX_NBF
            if residual_route == "pno"
            else _UNION_ROUTE_MAX_NBF
        )
    )
    if nbf > max_nbf:
        raise ValueError(
            f"run_local_dlpno_uccsd: {nbf} basis functions exceeds the "
            f"max_nbf={max_nbf} guard of the {residual_route!r} route"
        )

    ne = effective_electron_count(molecule, uhf)
    two_s = molecule.multiplicity - 1
    nalpha = (ne + two_s) // 2
    nbeta = (ne - two_s) // 2
    from vibeqc.correlation_conventions import resolve_frozen_core_count

    nf = resolve_frozen_core_count(molecule, options.n_frozen, reference=uhf)
    if nf < 0 or nf > nbeta:
        raise ValueError(f"n_frozen={nf} out of range for n_beta={nbeta}")

    noa = nalpha - nf
    nob = nbeta - nf
    nva = nbf - nalpha
    nvb = nbf - nbeta
    no = noa + nob
    nv = nva + nvb
    result = LocalUCCSDResult(
        e_hf=float(uhf.energy),
        n_frozen=nf,
        localise=options.localise,
        triples_mode=triples_mode,
        residual_domain=residual_route,
    )
    result.source_kind = str(source.kind)
    result.source_identity = str(source.identity)
    # The same dict object, so every later tick and every exit path sees it.
    result.phase_seconds = _phases
    if no < 2 or nv == 0:
        result.converged = True
        result.e_total = result.e_hf
        return result

    cao, cav = ca[:, nf:nalpha], ca[:, nalpha:]
    cbo, cbv = cb[:, nf:nbeta], cb[:, nbeta:]
    _tick("reference")
    if options.localise == "boys":
        cao_use = _boys_localise(cao, basis)
        cbo_use = _boys_localise(cbo, basis)
    else:
        cao_use, cbo_use = cao, cbo
    _tick("localise")

    overlap = np.asarray(source.overlap_matrix())
    c_occ = np.hstack([cao_use, cbo_use])
    atom_first, _ = build_atom_basis_map(molecule, basis)
    q_alpha = build_projection_matrix(ca[:, :nalpha], overlap)
    q_beta = build_projection_matrix(cb[:, :nbeta], overlap)
    _criterion = str(getattr(options, "domain_criterion", "doi")).strip().lower()
    if options.tcut_mkn <= 0.0:
        _criterion = "mulliken"
    _mask_kw = dict(
        criterion=_criterion,
        tcut_mkn=float(options.tcut_mkn),
        tcut_do=float(getattr(options, "tcut_do", 1e-2)),
    )
    # Each spin orbital's domain is judged against its own spin's PAOs.
    orb_atom_mask = np.vstack(
        [
            orbital_domain_atom_mask(
                molecule, basis, cao_use, overlap, q_alpha, atom_first, **_mask_kw
            ),
            orbital_domain_atom_mask(
                molecule, basis, cbo_use, overlap, q_beta, atom_first, **_mask_kw
            ),
        ]
    )

    # Occupied-first spin-orbital blocks.  Cross-spin charge-density and
    # Fock blocks are exactly zero; alpha and beta orbitals retain their
    # distinct UHF spatial coefficients.
    f_oo = np.zeros((no, no))
    f_ov = np.zeros((no, nv))
    f_vv = np.zeros((nv, nv))
    f_oo[:noa, :noa] = cao_use.T @ fa_ao @ cao_use
    f_oo[noa:, noa:] = cbo_use.T @ fb_ao @ cbo_use
    f_ov[:noa, :nva] = cao_use.T @ fa_ao @ cav
    f_ov[noa:, nva:] = cbo_use.T @ fb_ao @ cbv
    f_vv[:nva, :nva] = cav.T @ fa_ao @ cav
    f_vv[nva:, nva:] = cbv.T @ fb_ao @ cbv

    _tick("domains")
    # Density-fitting factors per spin over (occupied, virtual), (occupied,
    # occupied) and (virtual, virtual) of that spin. They are block-diagonal
    # in spin: the compiled route and its spin-blocked pair construction read
    # these blocks directly, and the combined spin-orbital tensors are built
    # only for the routes that consume them (the "union" residual and the
    # triples).
    spin_factors = [
        {
            "ov": np.ascontiguousarray(np.asarray(df.mo_transform(c_o, c_v))),
            "oo": np.ascontiguousarray(np.asarray(df.mo_transform(c_o, c_o))),
            "vv": np.ascontiguousarray(np.asarray(df.mo_transform(c_v, c_v))),
        }
        for c_o, c_v in ((cao_use, cav), (cbo_use, cbv))
    ]
    naux = spin_factors[0]["ov"].shape[0]
    _combined_factors: list[np.ndarray] = []

    def combined_factors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combined spin-orbital (b_oo, b_ov, b_vv), built on first use."""
        if not _combined_factors:
            c_oo = np.zeros((naux, no, no))
            c_ov = np.zeros((naux, no, nv))
            c_vv = np.zeros((naux, nv, nv))
            c_oo[:, :noa, :noa] = spin_factors[0]["oo"]
            c_oo[:, noa:, noa:] = spin_factors[1]["oo"]
            c_ov[:, :noa, :nva] = spin_factors[0]["ov"]
            c_ov[:, noa:, nva:] = spin_factors[1]["ov"]
            c_vv[:, :nva, :nva] = spin_factors[0]["vv"]
            c_vv[:, nva:, nva:] = spin_factors[1]["vv"]
            _combined_factors.extend([c_oo, c_ov, c_vv])
        return (
            _combined_factors[0],
            _combined_factors[1],
            _combined_factors[2],
        )

    if residual_route == "union":
        b_oo, b_ov, b_vv = combined_factors()
    else:
        b_oo = b_ov = b_vv = None

    _tick("factors")
    # Localised occupied centroids define bounded occupied coupling sets.
    # Spin does not enter the position expectation value.
    from vibeqc import compute_dipole

    dip = compute_dipole(basis)
    position = [np.asarray(dip.x), np.asarray(dip.y), np.asarray(dip.z)]
    centroids = np.column_stack(
        [
            np.einsum("mi,mn,ni->i", c_occ, axis, c_occ, optimize=True)
            for axis in position
        ]
    )
    distances = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :], axis=2
    )
    radius = float(options.coupling_radius)

    def coupled_occupieds(*centres: int) -> np.ndarray:
        if radius <= 0.0:
            return np.arange(no, dtype=int)
        near = np.zeros(no, dtype=bool)
        for centre in centres:
            near |= distances[centre] < radius
            near[centre] = True
        return np.where(near)[0]

    eps_o = np.diag(f_oo)
    eps_v = np.diag(f_vv)
    all_pair_keys = [(p, q) for p in range(no) for q in range(p + 1, no)]
    pair_u: dict[tuple[int, int], np.ndarray] = {}
    pair_eps: dict[tuple[int, int], np.ndarray] = {}
    pair_k: dict[tuple[int, int], np.ndarray] = {}
    t2: dict[tuple[int, int], np.ndarray] = {}
    pair_pao_size: dict[tuple[int, int], int] = {}
    pair_mp2_energy: dict[tuple[int, int], float] = {}
    screened_pairs: set[tuple[int, int]] = set()
    singles_density = (
        np.zeros((no, nv, nv))
        if options.tcut_pno_singles > 0.0 and residual_route == "union"
        else None
    )
    # Shared spin-blocked bookkeeping: occupied spin and
    # within-spin index, virtual row blocks, per-spin Fock and orbitals.
    occ_spin = [0] * noa + [1] * nob
    occ_local = list(range(noa)) + list(range(nob))
    nv_spin = (nva, nvb)
    vir_rows = (slice(0, nva), slice(nva, nv))
    fvv_spin = (
        np.ascontiguousarray(f_vv[:nva, :nva]),
        np.ascontiguousarray(f_vv[nva:, nva:]),
    )
    eps_v_spin = (np.diag(fvv_spin[0]), np.diag(fvv_spin[1]))
    spin_singles_density = (
        [np.zeros((nv_spin[occ_spin[p]],) * 2) for p in range(no)]
        if options.tcut_pno_singles > 0.0 and residual_route == "pno"
        else None
    )
    _spin_pao_cache: dict[tuple, np.ndarray] = {}

    def spin_pao(atoms: tuple[int, ...], spn: int) -> np.ndarray:
        """Semicanonical PAOs of one spin on ``atoms``, in that spin's
        canonical virtual coordinates; cached per (atoms, spin)."""
        key = (atoms, spn)
        hit = _spin_pao_cache.get(key)
        if hit is None:
            ao_indices = np.concatenate(
                [np.arange(atom_first[a], atom_first[a + 1]) for a in atoms]
            )
            spin = ("alpha", "beta")[spn]
            v_pao, _ = semicanonical_pao_basis(
                (fa_ao, fb_ao)[spn],
                overlap,
                (q_alpha, q_beta)[spn],
                ao_indices,
                options.lindep,
                fock_basis_id=(f"dlpno-uccsd:{result.source_identity}:frozen={nf}:"
                               f"pao-atoms={','.join(map(str, atoms))}:{spin}"),
                spin=spin,
            )
            hit = np.ascontiguousarray((cav, cbv)[spn].T @ overlap @ v_pao)
            _spin_pao_cache[key] = hit
        return hit

    def pair_atom_domain(p: int, q: int) -> tuple[int, ...] | None:
        """Atom domain of a pair; None keeps the full virtual space."""
        if options.tcut_mkn <= 0.0:
            return None
        atoms = np.where(orb_atom_mask[p] | orb_atom_mask[q])[0]
        if atoms.size == 0:
            raise ValueError(
                f"empty PAO atom domain for occupied pair {(p, q)} at "
                f"tcut_mkn={options.tcut_mkn}"
            )
        return tuple(int(a) for a in atoms)

    def spin_pno_space(blocks, threshold: float, *, pair) -> tuple[np.ndarray, np.ndarray]:
        """PNOs of a spin-block-diagonal pair density.

        ``blocks`` holds (spin, density, Fock, PAO or None) per spin block
        the pair touches. Each block is diagonalised on its own, occupations
        above ``threshold`` are kept (all at zero), the kept space of each
        block is semicanonicalised, and the columns are returned alpha first
        in canonical spin-virtual coordinates. As in ``_domain_from_density``
        a pair that would keep nothing keeps its single largest occupation.
        """
        decomposed = []
        for spn, density, f_block, pao in blocks:
            occ, vectors = np.linalg.eigh(0.5 * (density + density.T))
            order = np.argsort(-occ)
            decomposed.append((spn, occ[order], vectors[:, order], f_block, pao))
        keeps = [
            (occ > threshold) if threshold > 0.0 else np.ones_like(occ, dtype=bool)
            for _, occ, _, _, _ in decomposed
        ]
        if not any(np.any(keep) for keep in keeps):
            best = max(
                (b for b in range(len(decomposed)) if decomposed[b][1].size),
                key=lambda b: decomposed[b][1][0],
            )
            keeps[best][0] = True
        columns, energies = [], []
        for (spn, _occ, vectors, f_block, pao), keep in zip(decomposed, keeps):
            if not np.any(keep):
                continue
            kept = vectors[:, keep]
            f_domain = kept.T @ f_block @ kept
            spin = ("alpha", "beta")[spn]
            solved = solve_dense_fock(
                0.5 * (f_domain + f_domain.T), spin=spin,
                basis_id=(f"dlpno-uccsd:{result.source_identity}:frozen={nf}:"
                          f"pair={pair[0]},{pair[1]}:retained-virtual:{spin}"),
            )
            kept = kept @ solved.eigenvectors
            eps_block = solved.eigenvalues
            del solved  # Only the retained coefficients and energies survive.
            block = np.zeros((nv, kept.shape[1]))
            block[vir_rows[spn]] = kept if pao is None else pao @ kept
            columns.append(block)
            energies.append(eps_block)
        return np.ascontiguousarray(np.hstack(columns)), np.concatenate(energies)


    def pair_pao_basis(p: int, q: int) -> np.ndarray | None:
        """Return combined-spin PAOs in canonical virtual coordinates."""
        if options.tcut_mkn <= 0.0:
            return None
        atoms = np.where(orb_atom_mask[p] | orb_atom_mask[q])[0]
        if atoms.size == 0:
            raise ValueError(
                f"empty PAO atom domain for occupied pair {(p, q)} at "
                f"tcut_mkn={options.tcut_mkn}"
            )
        atoms = tuple(int(atom) for atom in atoms)
        va, vb = spin_pao(atoms, 0), spin_pao(atoms, 1)
        domain = np.zeros((nv, va.shape[1] + vb.shape[1]))
        domain[:nva, : va.shape[1]] = va
        domain[nva:, va.shape[1] :] = vb
        if domain.shape[1] == 0:
            raise ValueError(
                f"empty PAO virtual domain for occupied pair {(p, q)} at "
                f"tcut_mkn={options.tcut_mkn}"
            )
        return np.ascontiguousarray(domain)

    def occupied_pao_basis(p: int) -> np.ndarray | None:
        """Return one occupied's same-spin PAOs in spin-virtual coordinates."""
        if options.tcut_mkn <= 0.0:
            return None
        atoms = np.where(orb_atom_mask[p])[0]
        if atoms.size == 0:
            raise ValueError(
                f"empty singles PAO atom domain for occupied {p} at "
                f"tcut_mkn={options.tcut_mkn}"
            )
        spn = occ_spin[p]
        pao = spin_pao(tuple(int(atom) for atom in atoms), spn)
        domain = np.zeros((nv, pao.shape[1]))
        domain[vir_rows[spn]] = pao
        if domain.shape[1] == 0:
            raise ValueError(
                f"empty singles PAO virtual domain for occupied {p} at "
                f"tcut_mkn={options.tcut_mkn}"
            )
        return np.ascontiguousarray(domain)

    pno_correction = bool(getattr(options, "pno_correction", True))
    e_pno_correction = 0.0
    _pno_density = str(getattr(options, "pno_density", "two-sided")).strip().lower()
    if _pno_density not in ("two-sided", "one-sided"):
        raise ValueError(
            f"unknown pno_density {options.pno_density!r} "
            '(expected "two-sided" or "one-sided")'
        )
    two_sided = _pno_density == "two-sided"
    # Canonical-virtual coordinates: alpha virtuals first, then beta; their
    # AO overlap maps one spin side of a pair density into the other's.
    alpha_cols_full = np.arange(nva)
    beta_cols_full = np.arange(nva, nv)
    m_ab_full = cav.T @ overlap @ cbv
    if residual_route == "union":
        for p, q in all_pair_keys:
            coulomb = b_ov[:, p, :].T @ b_ov[:, q, :]
            anti = coulomb - coulomb.T
            denominator = (
                eps_o[p] + eps_o[q] - eps_v[:, None] - eps_v[None, :]
            )
            t_mp2 = anti / denominator
            if singles_density is not None:
                # Occupied-specific MP2 virtual densities. For the ordered pair
                # orientation p < q, the first virtual index belongs to p and the
                # second to q. Summing these pair contributions gives the
                # singles-specific natural-orbital spaces of Riplinger and Neese,
                # J. Chem. Phys. 138, 034106 (2013), doi:10.1063/1.4773581.
                singles_density[p] += t_mp2 @ t_mp2.T
                singles_density[q] += t_mp2.T @ t_mp2
            # The energy of one unique spin-orbital pair carries 1/2; summing both
            # occupied orders in the conventional expression would carry 1/4.
            e_mp2 = 0.5 * float(np.sum(anti * t_mp2))
            pair_mp2_energy[(p, q)] = e_mp2
            if tcut_pairs_so > 0.0 and abs(e_mp2) < tcut_pairs_so:
                screened_pairs.add((p, q))
                continue
            pao = pair_pao_basis(p, q)
            if pao is None:
                pair_pao_size[(p, q)] = nv
                anti_pao = anti
                f_pao = f_vv
            else:
                pair_pao_size[(p, q)] = pao.shape[1]
                anti_pao = pao.T @ anti @ pao
                f_pao = pao.T @ f_vv @ pao
            eps_pao = np.diag(f_pao)
            denominator_pao = (
                eps_o[p]
                + eps_o[q]
                - eps_pao[:, None]
                - eps_pao[None, :]
            )
            t_mp2_pao = anti_pao / denominator_pao
            density = t_mp2_pao @ t_mp2_pao.T + t_mp2_pao.T @ t_mp2_pao
            density = 0.5 * (density + density.T)
            if two_sided:
                if (p < noa) == (q < noa):
                    coulomb_pao = coulomb if pao is None else pao.T @ coulomb @ pao
                    c_amp = coulomb_pao / denominator_pao
                    direct = c_amp @ c_amp.T + c_amp.T @ c_amp
                    density = density + 0.5 * (direct + direct.T)
                else:
                    if pao is None:
                        a_cols, b_cols, m_ab = alpha_cols_full, beta_cols_full, m_ab_full
                    else:
                        a_cols = np.flatnonzero(np.any(pao[:nva] != 0.0, axis=0))
                        b_cols = np.flatnonzero(np.any(pao[nva:] != 0.0, axis=0))
                        m_ab = (cav @ pao[:nva][:, a_cols]).T @ overlap @ (
                            cbv @ pao[nva:][:, b_cols]
                        )
                    d_aa = density[np.ix_(a_cols, a_cols)].copy()
                    d_bb = density[np.ix_(b_cols, b_cols)].copy()
                    density[np.ix_(a_cols, a_cols)] += m_ab @ d_bb @ m_ab.T
                    density[np.ix_(b_cols, b_cols)] += m_ab.T @ d_aa @ m_ab
            d, ep = _domain_from_density(
                density, f_pao, options.tcut_pno,
                fock_basis_id=(f"dlpno-uccsd:{result.source_identity}:frozen={nf}:"
                               f"union-pair={p},{q}:retained-virtual:alpha-beta"),
                spin="alpha-beta",
            )
            u = d if pao is None else np.ascontiguousarray(pao @ d)
            kp = u.T @ anti @ u
            pair_u[(p, q)] = u
            pair_eps[(p, q)] = ep
            pair_k[(p, q)] = kp
            t2[(p, q)] = kp / (
                eps_o[p] + eps_o[q] - ep[:, None] - ep[None, :]
            )
            if pno_correction:
                # e_pq^PAO - e_pq^PNO, both semicanonical MP2 in this pair's own
                # domain, so the difference is what tcut_pno discarded; same
                # 1/2-per-unique-pair convention as the screening estimate above.
                e_pno_correction += 0.5 * float(np.sum(anti_pao * t_mp2_pao)) - (
                    0.5 * float(np.sum(kp * t2[(p, q)]))
                )

    else:
        # Spin-blocked pair construction for the compiled route. Every block
        # below equals the corresponding block of the combined construction
        # above (the pair density is block-diagonal in spin), but each pair
        # space is diagonalised per spin block, so every PNO is spin-pure and
        # a same-spin pair carries no directions of the other spin.
        for p, q in all_pair_keys:
            sp, sq = occ_spin[p], occ_spin[q]
            coulomb = (
                spin_factors[sp]["ov"][:, occ_local[p], :].T
                @ spin_factors[sq]["ov"][:, occ_local[q], :]
            )
            denominator = (
                eps_o[p]
                + eps_o[q]
                - eps_v_spin[sp][:, None]
                - eps_v_spin[sq][None, :]
            )
            if sp == sq:
                anti = coulomb - coulomb.T
                t_full = anti / denominator
                # Same 1/2-per-unique-pair convention as the union route.
                e_mp2 = 0.5 * float(np.sum(anti * t_full))
            else:
                # Opposite spins: <pq||ab> is the Coulomb block alone, and
                # its two orientations each carry half of the pair energy.
                t_full = coulomb / denominator
                e_mp2 = float(np.sum(coulomb * t_full))
            if spin_singles_density is not None:
                spin_singles_density[p] += t_full @ t_full.T
                spin_singles_density[q] += t_full.T @ t_full
            pair_mp2_energy[(p, q)] = e_mp2
            if tcut_pairs_so > 0.0 and abs(e_mp2) < tcut_pairs_so:
                screened_pairs.add((p, q))
                continue
            atoms = pair_atom_domain(p, q)
            if sp == sq:
                spn = sp
                pao = None if atoms is None else spin_pao(atoms, spn)
                if pao is not None and pao.shape[1] == 0:
                    raise ValueError(
                        f"empty PAO virtual domain for occupied pair {(p, q)} "
                        f"at tcut_mkn={options.tcut_mkn}"
                    )
                if pao is None:
                    anti_p, coul_p, f_p = anti, coulomb, fvv_spin[spn]
                else:
                    anti_p = pao.T @ anti @ pao
                    coul_p = pao.T @ coulomb @ pao
                    f_p = pao.T @ fvv_spin[spn] @ pao
                eps_p = np.diag(f_p)
                den_p = eps_o[p] + eps_o[q] - eps_p[:, None] - eps_p[None, :]
                t_p = anti_p / den_p
                density = t_p @ t_p.T + t_p.T @ t_p
                density = 0.5 * (density + density.T)
                if two_sided:
                    c_amp = coul_p / den_p
                    direct = c_amp @ c_amp.T + c_amp.T @ c_amp
                    density = density + 0.5 * (direct + direct.T)
                u, ep = spin_pno_space(
                    [(spn, density, f_p, pao)], options.tcut_pno, pair=(p, q),
                )
                u_s = u[vir_rows[spn]]
                kp = u_s.T @ anti @ u_s
                pair_pao_size[(p, q)] = (
                    int(nv_spin[spn]) if pao is None else int(pao.shape[1])
                )
                e_pair_pao = 0.5 * float(np.sum(anti_p * t_p))
            else:
                pao_a = None if atoms is None else spin_pao(atoms, 0)
                pao_b = None if atoms is None else spin_pao(atoms, 1)
                if atoms is not None and pao_a.shape[1] + pao_b.shape[1] == 0:
                    raise ValueError(
                        f"empty PAO virtual domain for occupied pair {(p, q)} "
                        f"at tcut_mkn={options.tcut_mkn}"
                    )
                if pao_a is None:
                    coul_p = coulomb
                    f_a, f_b = fvv_spin[0], fvv_spin[1]
                    m_ab = m_ab_full
                else:
                    coul_p = pao_a.T @ coulomb @ pao_b
                    f_a = pao_a.T @ fvv_spin[0] @ pao_a
                    f_b = pao_b.T @ fvv_spin[1] @ pao_b
                    m_ab = (cav @ pao_a).T @ overlap @ (cbv @ pao_b)
                eps_a, eps_b = np.diag(f_a), np.diag(f_b)
                den_p = eps_o[p] + eps_o[q] - eps_a[:, None] - eps_b[None, :]
                t_p = coul_p / den_p
                # One-sided density T T^T + T^T T of the antisymmetrised
                # amplitude [[0, t], [-t^T, 0]]: 2 t t^T and 2 t^T t.
                d_aa = 2.0 * (t_p @ t_p.T)
                d_bb = 2.0 * (t_p.T @ t_p)
                if two_sided:
                    d_aa, d_bb = (
                        d_aa + m_ab @ d_bb @ m_ab.T,
                        d_bb + m_ab.T @ d_aa @ m_ab,
                    )
                u, ep = spin_pno_space(
                    [(0, d_aa, f_a, pao_a), (1, d_bb, f_b, pao_b)],
                    options.tcut_pno, pair=(p, q),
                )
                c_pno = u[:nva].T @ coulomb @ u[nva:]
                kp = c_pno - c_pno.T
                pair_pao_size[(p, q)] = (
                    int(nv)
                    if pao_a is None
                    else int(pao_a.shape[1] + pao_b.shape[1])
                )
                e_pair_pao = float(np.sum(coul_p * t_p))
            pair_u[(p, q)] = u
            pair_eps[(p, q)] = ep
            pair_k[(p, q)] = kp
            t2[(p, q)] = kp / (eps_o[p] + eps_o[q] - ep[:, None] - ep[None, :])
            if pno_correction:
                e_pno_correction += e_pair_pao - 0.5 * float(np.sum(kp * t2[(p, q)]))

    pair_keys = [key for key in all_pair_keys if key not in screened_pairs]
    _tick("pairs")

    def triple_is_screened(i: int, j: int, k: int) -> bool:
        """Would local screening drop the LMO triple ``(i, j, k)``?

        Weak-pair and occupied-distance screening are both defined on the
        *localised* occupied indices: a triple survives only when none of its
        three pairs was screened out of the CCSD iteration and all three
        occupied centroids lie inside ``coupling_radius``.
        """
        if {(i, j), (i, k), (j, k)} & screened_pairs:
            return True
        return radius > 0.0 and (
            distances[i, j] >= radius
            or distances[i, k] >= radius
            or distances[j, k] >= radius
        )

    # (T1) rotates the occupied indices into the per-spin canonical basis, in
    # which a "distance" or a weak-pair label has no meaning: each canonical
    # occupied is a mixture of every LMO. So a screened triple list cannot be
    # carried through the rotation, and this route fails closed whenever the
    # requested thresholds actually remove one. Deciding on the realised set
    # rather than on the raw threshold values keeps the exact full-list (T1)
    # reachable when the thresholds happen to screen nothing -- which is what
    # a generous coupling_radius does, though not a positive tcut_pairs on any
    # system carrying symmetry-zero same-spin pair energies (see
    # tests/test_dlpno_uccsd_local.py). Recovering the screened local (T1)
    # itself needs an iterative local-basis treatment. ``t1-iterative`` uses a
    # spin-orbital generalization of Guo et al. 2018 Eq. (2), coupling
    # surviving triples through the off-diagonal occupied Fock *inside* the
    # LMO triple list instead of rotating the list away. Guo et al. 2020 is
    # the dedicated open-shell method-family source.
    screened_triples = (
        [
            (i, j, k)
            for i in range(no)
            for j in range(i + 1, no)
            for k in range(j + 1, no)
            if triple_is_screened(i, j, k)
        ]
        if options.compute_triples and triples_mode == "t1" and no >= 3
        else []
    )
    if screened_triples:
        raise NotImplementedError(
            "open-shell local (T1) does not support weak-pair or "
            "occupied-distance triple screening: the requested "
            f"tcut_pairs={options.tcut_pairs:g} / "
            f"coupling_radius={options.coupling_radius:g} screen "
            f"{len(screened_triples)} of "
            f"{no * (no - 1) * (no - 2) // 6} occupied triples, and the "
            "canonical occupied rotation cannot represent that list. Use "
            'triples_mode="t0" for a screened semicanonical run, or '
            '"t1-iterative", which keeps the triples in the localised basis '
            "and so can carry a screened list, or drop the screening "
            "thresholds."
        )

    singles_u: dict[int, np.ndarray] = {}
    singles_eps: dict[int, np.ndarray] = {}
    t1: dict[int, np.ndarray] = {}
    # Compiled route: singles basis per occupied within its own spin block
    # (None keeps the full canonical virtual space of that spin).
    spin_singles_u: list[np.ndarray | None] = [None] * no
    spin_singles_eps: list[np.ndarray | None] = [None] * no
    if residual_route == "union":
        full_singles = None
        for p in range(no):
            if singles_density is None:
                if full_singles is None:
                    full_singles = _domain_from_density(
                        np.eye(nv), f_vv, 0.0,
                        fock_basis_id=(f"dlpno-uccsd:{result.source_identity}:frozen={nf}:"
                                       "union-full-singles:alpha-beta"),
                        spin="alpha-beta",
                    )
                u, ep = full_singles
            else:
                pao = occupied_pao_basis(p)
                if pao is None:
                    density = singles_density[p]
                    f_singles = f_vv
                else:
                    density = pao.T @ singles_density[p] @ pao
                    f_singles = pao.T @ f_vv @ pao
                spin = "alpha-beta" if pao is None else ("alpha", "beta")[occ_spin[p]]
                d, ep = _domain_from_density(
                    density, f_singles, options.tcut_pno_singles,
                    fock_basis_id=(f"dlpno-uccsd:{result.source_identity}:frozen={nf}:"
                                   f"union-occupied={p}:retained-singles:{spin}"),
                    spin=spin,
                )
                u = d if pao is None else np.ascontiguousarray(pao @ d)
            singles_u[p] = u
            singles_eps[p] = ep
            t1[p] = np.zeros(u.shape[1])

    elif spin_singles_density is not None:
        for p in range(no):
            spn = occ_spin[p]
            if options.tcut_mkn > 0.0:
                atoms = np.where(orb_atom_mask[p])[0]
                if atoms.size == 0:
                    raise ValueError(
                        f"empty singles PAO atom domain for occupied {p} at "
                        f"tcut_mkn={options.tcut_mkn}"
                    )
                pao = spin_pao(tuple(int(a) for a in atoms), spn)
                if pao.shape[1] == 0:
                    raise ValueError(
                        f"empty singles PAO virtual domain for occupied {p} "
                        f"at tcut_mkn={options.tcut_mkn}"
                    )
                density = pao.T @ spin_singles_density[p] @ pao
                f_singles = pao.T @ fvv_spin[spn] @ pao
            else:
                pao = None
                density = spin_singles_density[p]
                f_singles = fvv_spin[spn]
            d, ep = _domain_from_density(
                density, f_singles, options.tcut_pno_singles,
                fock_basis_id=(f"dlpno-uccsd:{result.source_identity}:frozen={nf}:"
                               f"occupied={p}:retained-singles:{('alpha', 'beta')[spn]}"),
                spin=("alpha", "beta")[spn],
            )
            spin_singles_u[p] = d if pao is None else np.ascontiguousarray(pao @ d)
            spin_singles_eps[p] = ep
    # Singles spaces coincide exactly when they are the untruncated full
    # spin-virtual space (tcut_pno_singles=0); otherwise each occupied keeps
    # its own space and its own group.
    singles_key = {
        p: ("full" if singles_density is None else p) for p in range(no)
    }
    _tick("singles")
    result.e_pno_correction = float(e_pno_correction)
    result.n_pairs = len(pair_keys)
    result.n_screened = len(screened_pairs)
    result.screened_pair_energies = {
        key: pair_mp2_energy[key] for key in sorted(screened_pairs)
    }
    result.e_screened_mp2 = float(
        sum(result.screened_pair_energies.values())
    )
    result.pno_per_pair = {key: pair_u[key].shape[1] for key in pair_keys}
    result.pao_per_pair = {key: pair_pao_size[key] for key in pair_keys}
    if residual_route == "union":
        result.singles_per_occ = {p: singles_u[p].shape[1] for p in range(no)}
    else:
        result.singles_per_occ = {
            p: (
                int(nv_spin[occ_spin[p]])
                if spin_singles_u[p] is None
                else int(spin_singles_u[p].shape[1])
            )
            for p in range(no)
        }
    result.avg_pno = (
        float(np.mean(list(result.pno_per_pair.values())))
        if result.pno_per_pair
        else 0.0
    )
    result.avg_pao = (
        float(np.mean(list(result.pao_per_pair.values())))
        if result.pao_per_pair
        else 0.0
    )
    result.avg_singles_domain = float(
        np.mean(list(result.singles_per_occ.values()))
    )

    def source_t2(m: int, n: int) -> tuple[np.ndarray, np.ndarray] | None:
        if m == n:
            return None
        key = (m, n) if m < n else (n, m)
        if key in screened_pairs:
            return None
        sign = 1.0 if m < n else -1.0
        return pair_u[key], sign * t2[key]

    def residual_domain(
        required: np.ndarray,
        occupieds: np.ndarray,
    ) -> np.ndarray:
        """Union pair PNOs needed by a local residual contraction.

        DLPNO residual intermediates need a larger virtual span than the
        target amplitude alone.  Form the orthonormal union of pair domains
        in the local occupied set, always including the target.  Compact
        full-coupling systems may recover the full virtual span; extended
        systems retain a bounded domain once occupied locality is enabled.
        """
        pieces = [required]
        for a, m in enumerate(occupieds):
            for n in occupieds[a + 1 :]:
                pair = pair_u.get((int(m), int(n)))
                if pair is not None:
                    pieces.append(pair)
        stacked = np.hstack(pieces)
        left, singular, _ = np.linalg.svd(stacked, full_matrices=False)
        if singular.size == 0:
            return required
        cutoff = max(1e-12, 1e-10 * singular[0])
        rank = max(required.shape[1], int(np.count_nonzero(singular > cutoff)))
        return np.ascontiguousarray(left[:, : min(rank, nv)])

    def local_problem(
        target_u: np.ndarray,
        occupieds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble amplitudes, DF tensor, and Fock in one target domain."""
        nl = len(occupieds)
        nd = target_u.shape[1]
        t1_local = np.zeros((nl, nd))
        t2_local = np.zeros((nl, nl, nd, nd))
        for a, m in enumerate(occupieds):
            overlap = target_u.T @ singles_u[m]
            t1_local[a] = overlap @ t1[m]
            for b, n in enumerate(occupieds):
                source = source_t2(m, n)
                if source is None:
                    continue
                source_u, source_amp = source
                pair_overlap = target_u.T @ source_u
                t2_local[a, b] = (
                    pair_overlap @ source_amp @ pair_overlap.T
                )

        bov = np.einsum(
            "Piv,va->Pia", b_ov[:, occupieds, :], target_u, optimize=True
        )
        bvv = np.einsum(
            "Puv,ua,vb->Pab", b_vv, target_u, target_u, optimize=True
        )
        size = nl + nd
        b_local = np.zeros((naux, size, size))
        b_local[:, :nl, :nl] = b_oo[:, occupieds][:, :, occupieds]
        b_local[:, :nl, nl:] = bov
        b_local[:, nl:, :nl] = bov.transpose(0, 2, 1)
        b_local[:, nl:, nl:] = bvv

        fov = f_ov[occupieds] @ target_u
        f_local = np.zeros((size, size))
        f_local[:nl, :nl] = f_oo[np.ix_(occupieds, occupieds)]
        f_local[:nl, nl:] = fov
        f_local[nl:, :nl] = fov.T
        f_local[nl:, nl:] = target_u.T @ f_vv @ target_u
        return (
            np.ascontiguousarray(t1_local),
            np.ascontiguousarray(t2_local.reshape(nl * nl, nd * nd)),
            np.ascontiguousarray(b_local.reshape(naux, size * size)),
            np.ascontiguousarray(f_local),
        )

    def energy() -> float:
        value = result.e_screened_mp2
        for p in range(no):
            value += float(np.dot(f_ov[p] @ singles_u[p], t1[p]))
        for p, q in pair_keys:
            u = pair_u[(p, q)]
            tp = (u.T @ singles_u[p]) @ t1[p]
            tq = (u.T @ singles_u[q]) @ t1[q]
            k = pair_k[(p, q)]
            value += 0.5 * float(np.sum(k * t2[(p, q)]))
            value += float(np.sum(k * np.outer(tp, tq)))
        return value

    def flatten() -> np.ndarray:
        return np.concatenate(
            [t1[p].ravel() for p in range(no)]
            + [t2[key].ravel() for key in pair_keys]
        )

    def unflatten(vector: np.ndarray) -> None:
        offset = 0
        for p in range(no):
            count = t1[p].size
            t1[p] = vector[offset : offset + count].copy()
            offset += count
        for key in pair_keys:
            count = t2[key].size
            t2[key] = vector[offset : offset + count].reshape(
                t2[key].shape
            ).copy()
            offset += count

    if residual_route == "union":
        amplitude_history: list[np.ndarray] = []
        residual_history: list[np.ndarray] = []
        coupled_sizes: list[int] = []
        e_prev = energy()
        for iteration in range(options.max_iter):
            old = flatten()
            new_t1: dict[int, np.ndarray] = {}
            new_t2: dict[tuple[int, int], np.ndarray] = {}

            # A residual depends on its target only through the contraction
            # domain and the coupled occupied set. Every pair of one coupling set
            # has the same domain (the union of that set's pair spaces, which
            # contains the target's own), and singles share theirs whenever their
            # natural-orbital spaces coincide, so each group is contracted once
            # per iteration and every member reads its own block (the open-shell
            # counterpart of #689).
            singles_groups: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
            pair_groups: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
            for p in range(no):
                occupieds = coupled_occupieds(p)
                coupled_sizes.append(len(occupieds))
                gkey = (tuple(int(m) for m in occupieds), singles_key[p])
                cached = singles_groups.get(gkey)
                if cached is None:
                    domain = residual_domain(singles_u[p], occupieds)
                    local = local_problem(domain, occupieds)
                    r1, _ = dlpno_uccsd_pair_residual(*local)
                    cached = (domain, np.asarray(r1))
                    singles_groups[gkey] = cached
                domain, r1 = cached
                lp = int(np.searchsorted(occupieds, p))
                projected_r1 = (singles_u[p].T @ domain) @ r1[lp]
                step = projected_r1 / (eps_o[p] - singles_eps[p])
                new_t1[p] = t1[p] + step

            for p, q in pair_keys:
                occupieds = coupled_occupieds(p, q)
                coupled_sizes.append(len(occupieds))
                gkey = tuple(int(m) for m in occupieds)
                cached = pair_groups.get(gkey)
                if cached is None:
                    domain = residual_domain(pair_u[(p, q)], occupieds)
                    local = local_problem(domain, occupieds)
                    _, r2_flat = dlpno_uccsd_pair_residual(*local)
                    nd = domain.shape[1]
                    cached = (
                        domain,
                        np.asarray(r2_flat).reshape(
                            len(occupieds), len(occupieds), nd, nd
                        ),
                    )
                    pair_groups[gkey] = cached
                domain, r2 = cached
                lp = int(np.searchsorted(occupieds, p))
                lq = int(np.searchsorted(occupieds, q))
                denominator = (
                    eps_o[p]
                    + eps_o[q]
                    - pair_eps[(p, q)][:, None]
                    - pair_eps[(p, q)][None, :]
                )
                projection = pair_u[(p, q)].T @ domain
                projected_r2 = projection @ r2[lp, lq] @ projection.T
                updated = t2[(p, q)] + projected_r2 / denominator
                new_t2[(p, q)] = 0.5 * (updated - updated.T)

            result.n_residual_groups = len(singles_groups) + len(pair_groups)
            for p in range(no):
                t1[p] = new_t1[p]
            for key in pair_keys:
                t2[key] = new_t2[key]

            amplitude_history.append(flatten())
            residual_history.append(amplitude_history[-1] - old)
            if len(amplitude_history) > options.diis_size:
                amplitude_history.pop(0)
                residual_history.pop(0)
            nh = len(amplitude_history)
            if nh >= 2:
                metric = np.full((nh + 1, nh + 1), -1.0)
                metric[-1, -1] = 0.0
                for a in range(nh):
                    for b in range(nh):
                        metric[a, b] = float(
                            np.dot(residual_history[a], residual_history[b])
                        )
                rhs = np.zeros(nh + 1)
                rhs[-1] = -1.0
                try:
                    weights = np.linalg.solve(metric, rhs)[:nh]
                    unflatten(
                        sum(
                            weights[k] * amplitude_history[k]
                            for k in range(nh)
                        )
                    )
                except np.linalg.LinAlgError:
                    pass

            e_corr = energy()
            residual_norm = float(np.max(np.abs(residual_history[-1])))
            result.trace.append(
                {
                    "iter": iteration + 1,
                    "e_corr": e_corr,
                    "delta_e": e_corr - e_prev,
                    "r_norm": residual_norm,
                }
            )
            if (
                iteration > 0
                and abs(e_corr - e_prev) < options.conv_tol_energy
                and residual_norm < options.conv_tol_residual
            ):
                result.converged = True
                result.n_iter = iteration + 1
                break
            e_prev = e_corr
        else:
            result.n_iter = options.max_iter

        result.e_corr = energy() + result.e_pno_correction
        result.t1_norm = float(
            np.sqrt(sum(float(np.dot(amplitude, amplitude)) for amplitude in t1.values()))
        )
        result.avg_coupled_occ = float(np.mean(coupled_sizes)) if coupled_sizes else 0.0
    else:
        pair_couplings = [
            [int(m) for m in coupled_occupieds(p, q)] for p, q in pair_keys
        ]
        singles_couplings = [
            [int(m) for m in coupled_occupieds(p)] for p in range(no)
        ]
        e_start = result.e_screened_mp2 + sum(
            0.5 * float(np.sum(pair_k[key] * t2[key])) for key in pair_keys
        )
        empty_basis = np.zeros((0, 0))
        empty_eps = np.zeros(0)
        solve = dlpno_uccsd_pno_solve(
            spin_factors[0]["ov"].reshape(naux, -1),
            spin_factors[1]["ov"].reshape(naux, -1),
            spin_factors[0]["oo"].reshape(naux, -1),
            spin_factors[1]["oo"].reshape(naux, -1),
            spin_factors[0]["vv"].reshape(naux, -1),
            spin_factors[1]["vv"].reshape(naux, -1),
            np.ascontiguousarray(f_oo),
            np.ascontiguousarray(f_ov),
            fvv_spin[0],
            fvv_spin[1],
            [p for p, _ in pair_keys],
            [q for _, q in pair_keys],
            [pair_u[key] for key in pair_keys],
            [pair_eps[key] for key in pair_keys],
            [pair_k[key] for key in pair_keys],
            [t2[key] for key in pair_keys],
            pair_couplings,
            singles_couplings,
            [empty_basis if u is None else u for u in spin_singles_u],
            [empty_eps if e is None else e for e in spin_singles_eps],
            max_iter=int(options.max_iter),
            conv_tol_energy=float(options.conv_tol_energy),
            conv_tol_residual=float(options.conv_tol_residual),
            diis_size=int(options.diis_size),
        )
        for index, key in enumerate(pair_keys):
            t2[key] = np.asarray(solve["T2"][index])
        t1_full = np.asarray(solve["T1"])
        # The triples read singles through (singles_u, t1); one shared
        # identity expresses the full-space singles without per-occupied
        # copies.
        identity_v = np.eye(nv)
        for p in range(no):
            singles_u[p] = identity_v
            singles_eps[p] = eps_v
            t1[p] = np.ascontiguousarray(t1_full[p])
        e_prev = e_start
        for k, (e_iter, r_norm) in enumerate(
            zip(solve["trace_e_corr"], solve["trace_r_norm"])
        ):
            e_now = float(e_iter) + result.e_screened_mp2
            result.trace.append(
                {
                    "iter": k + 1,
                    "e_corr": e_now,
                    "delta_e": e_now - e_prev,
                    "r_norm": float(r_norm),
                }
            )
            e_prev = e_now
        result.converged = bool(solve["converged"])
        result.n_iter = int(solve["n_iter"])
        result.e_corr = (
            float(solve["e_corr"]) + result.e_screened_mp2 + result.e_pno_correction
        )
        result.t1_norm = float(np.linalg.norm(t1_full))
        coupled_sizes = [len(c) for c in singles_couplings] + [
            len(c) for c in pair_couplings
        ]
        result.avg_coupled_occ = (
            float(np.mean(coupled_sizes)) if coupled_sizes else 0.0
        )
        result.n_residual_groups = len(pair_keys) + no
        result.n_offlist_rows = int(solve["n_offlist_rows"])
        result.static_factor_bytes = int(solve["static_factor_bytes"])
        result.setup_seconds = float(solve["setup_seconds"])
        result.iteration_seconds = float(solve["iteration_seconds"])
    _tick("amplitudes")
    compiled_triples = residual_route == "pno" and triples_mode == "t0"
    if options.compute_triples and result.converged and no >= 3:
        if b_ov is None and not compiled_triples:
            # The dense (T1) modes read the combined spin-orbital factors.
            b_oo, b_ov, b_vv = combined_factors()
        t1_full: np.ndarray | None = None
        t2_full: np.ndarray | None = None
        b_ov_triples = b_ov
        b_oo_triples = b_oo
        eps_o_triples = eps_o
        if triples_mode == "t1":
            t1_local = np.zeros((no, nv))
            for m in range(no):
                t1_local[m] = singles_u[m] @ t1[m]
            t2_local = np.zeros((no, no, nv, nv))
            for m in range(no):
                for n in range(no):
                    source = source_t2(m, n)
                    if source is None:
                        continue
                    source_u, source_amp = source
                    t2_local[m, n] = source_u @ source_amp @ source_u.T

            eps_o_triples, rotation = _occupied_triples_rotation(
                f_oo, noa, source_identity=result.source_identity, n_frozen=nf,
            )
            t1_full = np.ascontiguousarray(rotation.T @ t1_local)
            t2_full = np.ascontiguousarray(
                np.einsum(
                    "mM,nN,mnuv->MNuv",
                    rotation,
                    rotation,
                    t2_local,
                    optimize=True,
                )
            )
            b_ov_triples = np.ascontiguousarray(
                np.einsum("Pmv,mM->PMv", b_ov, rotation, optimize=True)
            )
            b_oo_triples = np.ascontiguousarray(
                np.einsum(
                    "Pmn,mM,nN->PMN",
                    b_oo,
                    rotation,
                    rotation,
                    optimize=True,
                )
            )

        # The rotated-occupied ``t1`` route reaches this point only when the
        # screened set is empty (the guard above fails closed otherwise).
        # ``t0`` and ``t1-iterative`` retain this realised local triple list.
        triples: list[tuple[int, int, int]] = []
        for i in range(no):
            for j in range(i + 1, no):
                for k in range(j + 1, no):
                    if triple_is_screened(i, j, k):
                        result.n_screened_triples += 1
                    else:
                        triples.append((i, j, k))

        if compiled_triples:
            # Local (T0) in the compiled kernel: per triple, the TNO space of
            # its three pair spaces, only the pairs that contain one of its
            # occupieds, and only its own DF blocks -- the same contraction
            # as the loop below without the dense per-triple inventories.
            from vibeqc._vibeqc_core import dlpno_uccsd_local_triples

            t1_rows = np.vstack([singles_u[m] @ t1[m] for m in range(no)])
            local_t = dlpno_uccsd_local_triples(
                spin_factors[0]["ov"].reshape(naux, -1),
                spin_factors[1]["ov"].reshape(naux, -1),
                spin_factors[0]["oo"].reshape(naux, -1),
                spin_factors[1]["oo"].reshape(naux, -1),
                spin_factors[0]["vv"].reshape(naux, -1),
                spin_factors[1]["vv"].reshape(naux, -1),
                np.ascontiguousarray(eps_o),
                fvv_spin[0],
                fvv_spin[1],
                [p for p, _ in pair_keys],
                [q for _, q in pair_keys],
                [pair_u[key] for key in pair_keys],
                [t2[key] for key in pair_keys],
                np.ascontiguousarray(t1_rows),
                [list(key) for key in triples],
                tcut_tno=float(options.tcut_tno),
                lindep=float(options.lindep),
            )
            bad_triples = np.flatnonzero(~np.isfinite(local_t["e_triple"]))
            if bad_triples.size:
                key = triples[int(bad_triples[0])]
                raise RuntimeError(
                    "native open-shell (T0) returned a non-finite energy "
                    f"for occupied triple {key}"
                )
            if not np.isfinite(local_t["e_t"]):
                raise RuntimeError(
                    "native open-shell (T0) returned a non-finite total correction"
                )
            result.e_t = float(local_t["e_t"])
            result.tno_per_triple = {
                key: int(size) for key, size in zip(triples, local_t["n_tno"])
            }
            result.n_triples = len(triples)
            result.avg_tno = (
                float(np.mean(list(result.tno_per_triple.values())))
                if result.tno_per_triple
                else 0.0
            )
            result.n_degenerate_tno_triples = sum(
                1 for size in result.tno_per_triple.values() if size < 3
            )
            result.triples_executed = bool(triples)
            result.e_total = result.e_hf + result.e_corr + result.e_t
            _tick("triples")
            return result

        def triple_domain(i: int, j: int, k: int) -> tuple[np.ndarray, np.ndarray]:
            """Build and semicanonicalise one triple's TNO domain."""
            if t2_full is not None:
                density = np.zeros((nv, nv))
                for p, q in ((i, j), (i, k), (j, k)):
                    amplitude = t2_full[p, q]
                    density += amplitude @ amplitude.T + amplitude.T @ amplitude
                # Retained TNO Fock recanonicalization: Riplinger et al.,
                # doi:10.1063/1.4821834, p. 134101-4. Density selection and
                # this rotated-occupied UHF compatibility path stay unchanged.
                return _domain_from_density(
                    density, f_vv, options.tcut_tno,
                    fock_basis_id=(f"dlpno-uccsd:{result.source_identity}:frozen={nf}:"
                                   f"rotated-t1-triple={i},{j},{k}:retained-tno:alpha-beta"),
                    spin="alpha-beta",
                )

            pairs = [(i, j), (i, k), (j, k)]
            stacked = np.hstack([pair_u[key] for key in pairs])
            left, singular, _ = np.linalg.svd(stacked, full_matrices=False)
            cutoff = options.lindep * max(float(singular[0]), 1e-300)
            rank = max(1, int(np.count_nonzero(singular > cutoff)))
            precursor = np.ascontiguousarray(left[:, :rank])
            density = np.zeros((rank, rank))
            for key in pairs:
                overlap_t = precursor.T @ pair_u[key]
                amplitude = overlap_t @ t2[key] @ overlap_t.T
                density += amplitude @ amplitude.T + amplitude.T @ amplitude
            vectors, eps_t = _domain_from_density(
                density,
                precursor.T @ f_vv @ precursor,
                options.tcut_tno,
            )
            return np.ascontiguousarray(precursor @ vectors), eps_t

        if triples_mode == "t1-iterative":
            # Spin-orbital generalization of Guo et al. (2018) Eq. (2) in the
            # LOCALISED occupied basis. Guo et al. (2020) is the dedicated
            # open-shell method-family source. Iterating the off-diagonal
            # occupied-Fock coupling preserves the screened triple list,
            # unlike the canonical-rotation "t1" route.
            from .triples_iterative import iterative_t1_triples_correction

            tno_domains = {}
            for key in triples:
                vectors, eps_key = triple_domain(*key)
                tno_domains[key] = (vectors, eps_key)
                result.tno_per_triple[key] = int(vectors.shape[1])
            b_ov_full = b_ov
            eri_vovv_f = np.einsum(
                "Peb,Pic->eibc", b_vv, b_ov_full, optimize=True
            ) - np.einsum("Pec,Pib->eibc", b_vv, b_ov_full, optimize=True)
            eri_ovoo_f = np.einsum(
                "Pmj,Pka->majk", b_oo, b_ov_full, optimize=True
            ) - np.einsum("Pmk,Pja->majk", b_oo, b_ov_full, optimize=True)
            eri_oovv_f = np.einsum(
                "Pjb,Pkc->jkbc", b_ov_full, b_ov_full, optimize=True
            ) - np.einsum("Pjc,Pkb->jkbc", b_ov_full, b_ov_full, optimize=True)
            t1_it = np.zeros((no, nv))
            for m in range(no):
                t1_it[m] = singles_u[m] @ t1[m]
            t2_it = np.zeros((no, no, nv, nv))
            for m in range(no):
                for n in range(no):
                    source = source_t2(m, n)
                    if source is None:
                        continue
                    source_u, source_amp = source
                    t2_it[m, n] = source_u @ source_amp @ source_u.T
            e_t_iter, n_sweeps, converged_iter = (
                iterative_t1_triples_correction(
                    t1_it,
                    t2_it,
                    eri_vovv_f,
                    eri_ovoo_f,
                    eri_oovv_f,
                    f_oo,
                    eps_v,
                    triple_list=triples,
                    tno_domains=tno_domains,
                )
            )
            if not converged_iter:
                raise RuntimeError(
                    "iterative (T1) triples did not converge in "
                    f"{n_sweeps} sweeps; use the unscreened rotated "
                    'triples_mode="t1" route when applicable, or fall back '
                    'to triples_mode="t0"'
                )
            result.e_t = e_t_iter
            result.n_triples = len(triples)
            result.avg_tno = (
                float(np.mean(list(result.tno_per_triple.values())))
                if result.tno_per_triple
                else float(nv)
            )
            result.n_degenerate_tno_triples = sum(
                1 for s in result.tno_per_triple.values() if s < 3
            )
            result.triples_executed = bool(triples)
            result.e_total = result.e_hf + result.e_corr + result.e_t
            _tick("triples")
            return result

        for i, j, k in triples:
            tno, eps_t = triple_domain(i, j, k)
            nt = tno.shape[1]

            if t1_full is not None and t2_full is not None:
                t1_t = np.ascontiguousarray(t1_full @ tno)
                t2_t = np.ascontiguousarray(
                    np.einsum(
                        "mnuv,ua,vb->mnab",
                        t2_full,
                        tno,
                        tno,
                        optimize=True,
                    )
                )
            else:
                t1_t = np.zeros((no, nt))
                for m in range(no):
                    t1_t[m] = (tno.T @ singles_u[m]) @ t1[m]
                t2_t = np.zeros((no, no, nt, nt))
                for m in range(no):
                    for n in range(no):
                        source = source_t2(m, n)
                        if source is None:
                            continue
                        source_u, source_amp = source
                        overlap_t = tno.T @ source_u
                        t2_t[m, n] = overlap_t @ source_amp @ overlap_t.T

            bov_t = np.einsum(
                "Piv,va->Pia", b_ov_triples, tno, optimize=True
            )
            bvv_t = np.einsum("Puv,ua,vb->Pab", b_vv, tno, tno, optimize=True)
            eri_vovv = np.einsum(
                "Peb,Pic->eibc", bvv_t, bov_t, optimize=True
            ) - np.einsum("Pec,Pib->eibc", bvv_t, bov_t, optimize=True)
            eri_ovoo = np.einsum(
                "Pmj,Pka->majk", b_oo_triples, bov_t, optimize=True
            ) - np.einsum(
                "Pmk,Pja->majk", b_oo_triples, bov_t, optimize=True
            )
            eri_oovv = np.einsum(
                "Pjb,Pkc->jkbc", bov_t, bov_t, optimize=True
            ) - np.einsum("Pjc,Pkb->jkbc", bov_t, bov_t, optimize=True)
            result.e_t += float(
                dlpno_spin_orbital_triple_energy(
                    i,
                    j,
                    k,
                    np.ascontiguousarray(t1_t),
                    np.ascontiguousarray(t2_t.reshape(no * no, nt * nt)),
                    np.ascontiguousarray(eri_vovv.reshape(nt * no, nt * nt)),
                    np.ascontiguousarray(eri_ovoo.reshape(no * nt, no * no)),
                    np.ascontiguousarray(eri_oovv.reshape(no * no, nt * nt)),
                    np.ascontiguousarray(eps_o_triples),
                    np.ascontiguousarray(eps_t),
                )
            )
            result.tno_per_triple[(i, j, k)] = nt
        result.n_triples = len(triples)
        result.avg_tno = (
            float(np.mean(list(result.tno_per_triple.values())))
            if result.tno_per_triple
            else 0.0
        )
        # A triple excitation promotes three electrons into three *distinct*
        # spin virtuals, so a TNO domain holding fewer than three cannot host
        # one and its energy is identically zero. Counting those separates
        # "tcut_tno truncated a little" from "tcut_tno truncated past the point
        # where the correction exists", which the energy alone cannot show.
        result.n_degenerate_tno_triples = sum(
            1 for size in result.tno_per_triple.values() if size < 3
        )
        result.triples_executed = bool(triples)
        _tick("triples")
    result.e_total = result.e_hf + result.e_corr + result.e_t
    return result
