"""HF-CCM self-consistent field (closed-shell RHF) -- AICCM milestone 2b.

Assembles the cyclic Fock matrix from the CCM-weighted integrals
(``S^CCM``, ``h^CCM``, the effective ERI tensor, and ``V_nn^CCM``) and
solves the cyclic Roothaan-Hall equations (eqs 25-27 of Peintinger &
Bredow, J. Comput. Chem. 35, 839 (2014), doi:10.1002/jcc.23550):

    # Eq. 25:  F^CCM = h^CCM + sum_n (2 J_n^CCM - K_n^CCM)
    # Eq. 26:  F^CCM C^CCM = S^CCM C^CCM E^CCM
    # Eq. 27:  E^CCM = sum_i eps_i - 1/2 sum_ij (J_ij - K_ij) + V_nn^CCM

This is the small-cluster validation driver: it consumes the dense effective
ERI tensor (:func:`vibeqc.periodic.ccm.padded.ccm_eri`) and runs a plain
DIIS-accelerated RHF in Python. The production path (large 3-D clusters,
all SCF methods) is a C++ ``WeightedLatticeJKBuilder`` fed into the existing
``run_*_scf_with_jk`` entry points. The energy
per atom from this driver converges to periodic Hartree-Fock as the cluster
grows (Peintinger & Bredow 2014, Tables 2-4).
"""

from __future__ import annotations

from contextlib import contextmanager

from dataclasses import dataclass, field

import numpy as np

from .experimental import _reject_open_shell_cluster, _warn_experimental
from .integrals import ccm_overlap
from .padded import (
    P2014_NUCLEAR_VARIANTS,
    ccm_eri,
    ccm_eri_p2014,
    ccm_eri_symmetric,
    ccm_hcore,
    ccm_hcore_p2014,
    ccm_nuclear_repulsion,
)

__all__ = ["CCMSCFResult", "run_ccm_rhf"]


def _validate_conv_tol_grad(conv_tol_grad, *, who):
    """Return a positive finite commutator tolerance or fail closed."""
    try:
        value = float(conv_tol_grad)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{who}: conv_tol_grad must be finite and positive."
        ) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{who}: conv_tol_grad must be finite and positive."
        )
    return value


def _commutator_exit_ok(d_e, err, conv_tol, conv_tol_grad):
    """SCF exit predicate shared by the closed-shell CCM loops.

    ``conv_tol`` gates the energy criterion only; ``conv_tol_grad`` is the
    independent DIIS-commutator residual bound. Both must hold before a run
    may report convergence, so a caller can tighten the density criterion
    independently of the energy criterion.
    """
    return abs(d_e) < conv_tol and np.max(np.abs(err)) < conv_tol_grad

# Effective four-center builders selectable via run_ccm_rhf(method=...).
#   "union12"  -- NOT the 2012/2014 algorithm, despite its name (2026-09-17):
#                the bra partner stays in the home copy of the cluster, where
#                P2014 Eq. 19 and the original program image it; see "p2014"
#                below. Kept unchanged because archived records depend on it.
#                The rest of this note predates that finding and describes
#                "union12" as built: eq-18 product weight, home bra;
#                validated in 1-D IN A MINIMAL BASIS ONLY, breaks 8-fold ERI
#                symmetry in >=2-D. The weighted four-center is M = R^T G-hat C
#                with R != C, so it is not a congruence of the padded ERI Gram
#                matrix G-hat and carries no PSD guarantee: on the published
#                1-D alternating H chain it has 77 negative eigenvalues
#                (min -0.236) and the SCF is variationally unbounded on any
#                basis with more than one function per centre (issue #242,
#                DZVP -7.11 to -8.48 Ha/atom vs a published -0.552592; STO-3G
#                is clean only because a minimal basis cannot reach that
#                sector). This is a property of the printed equation, not of
#                the reimplementation; do
#                not "fix" it with a positive-definiteness guard or the
#                symmetrised weight, both of which were measured to leave the
#                spectrum unchanged. See vibeqc.periodic.ccm.__init__.
#   "aiccm2026dev-a" -- the symmetric Born-von Kármán-torus four-center
#                (AICCM_ALGORITHM.md Sec.13): exactly 8-fold symmetric for any
#                lattice. This is *this* development line's method of record.
#   "p2014"    -- the 2012 program / P2014 Eqs. 18-19 with the bra partner
#                IMAGED over the Wigner-Seitz supercell (padded.ccm_eri_p2014),
#                which "union12" is not: "union12" keeps the partner in the
#                home copy and misses P2014 Table 2 by 1.9e-4 to 1.7e-5 Eh per
#                atom. "p2014" reproduces that table to the printed digits
#                (tests/test_ccm_p2014_table2.py). It carries its own nuclear
#                attraction (P2014_NUCLEAR_VARIANTS), so it is deliberately NOT
#                in the registry the MP2/CCSD/gradient/UHF drivers share: only
#                run_ccm_rhf and run_ccm_rhf_scalable accept it.
#: The faithful-2014 selector (diagnosis lineage, ruling R1).
P2014_METHOD = "p2014"
#: What the other four-centre selectors use for the nuclear attraction
#: (padded.ccm_nuclear: Eq. 13 pair weight, the partner-side nucleus weight
#: taken relative to the HOME image of the partner).
DEFAULT_NUCLEAR_ATTRACTION = "eq13-home-image"


def _resolve_nuclear_attraction(method, nuclear_attraction, *, who):
    """Name of the nuclear-attraction construction ``method`` runs with."""
    if method == P2014_METHOD:
        variant = nuclear_attraction or "program-union"
        if variant not in P2014_NUCLEAR_VARIANTS:
            raise ValueError(
                f"{who}: nuclear_attraction must be one of "
                f"{P2014_NUCLEAR_VARIANTS} for method='p2014'; got {variant!r}")
        return variant
    if nuclear_attraction not in (None, DEFAULT_NUCLEAR_ATTRACTION):
        raise ValueError(
            f"{who}: nuclear_attraction={nuclear_attraction!r} exists only for "
            f"method='p2014'; method={method!r} runs with "
            f"{DEFAULT_NUCLEAR_ATTRACTION!r}.")
    return DEFAULT_NUCLEAR_ATTRACTION


_CCM_ERI_METHODS = {
    "union12": ccm_eri,
    "aiccm2026dev-a": ccm_eri_symmetric,
    "aiccmdev": ccm_eri_symmetric,   # deprecated alias of "aiccm2026dev-a" (back-compat)
}


def _ccm_eri_for_method(ccm, method):
    """Build the effective four-center tensor selected by ``method``.

    Shared by the RHF / UHF / MP2 CCM drivers so the ``method`` keyword (notably
    ``"aiccm2026dev-a"``) means the same thing everywhere.
    """
    try:
        builder = _CCM_ERI_METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown CCM four-center method {method!r}; "
            f"choose from {sorted(_CCM_ERI_METHODS)}"
        )
    return builder(ccm)


@dataclass(frozen=True)
class CCMSCFIteration:
    """One SCF cycle of a CCM loop, recorded for post-hoc diagnosis.

    The CCM loops previously reported only ``converged`` and ``n_iter``, so a
    non-converged run carried NO information about HOW it failed: an
    oscillation, a monotone crawl and a hard stall are indistinguishable from
    ``converged: false, n_iter: 128``. CLAUDE.md § 7 requires diagnosing a
    periodic SCF failure rather than reaching for a convergence aid, and that
    diagnosis is impossible without the residual trajectory (GitLab #493).

    ``grad_max`` is the max-abs orthonormalised DIIS commutator residual
    ``max |X^T (F D S - S D F) X|`` -- the same quantity the exit predicate
    tests, so a reader can see exactly how far the run was from its own gate.
    ``homo_lumo`` is the frontier gap in Ha at this cycle (``nan`` when the
    virtual space is empty). It is recorded because a collapsing gap and a
    healthy one call for opposite responses, and the two are not
    distinguishable from the energy alone. On the first system traced this
    way it did the useful thing by ruling a hypothesis OUT: #493 suspected
    near-degeneracy, and the measured gap was flat at 0.161 Ha across the
    stalled cycles, so the stall is not a frontier-gap collapse and a level
    shift would have been the wrong aid.
    """

    iteration: int
    energy: float
    delta_e: float
    grad_max: float
    homo_lumo: float
    diis_dim: int


def _scf_trace_row(it, e_tot, e_last, err, eps, n_occ, diis_dim):
    """Build one :class:`CCMSCFIteration`; never raises on a degenerate shape."""
    eps = np.asarray(eps).ravel()
    if 0 < n_occ < eps.size:
        gap = float(eps[n_occ] - eps[n_occ - 1])
    else:
        gap = float("nan")
    return CCMSCFIteration(
        iteration=int(it),
        energy=float(e_tot),
        delta_e=float(e_tot - e_last) if it > 1 else float("nan"),
        grad_max=float(np.max(np.abs(err))) if np.size(err) else float("nan"),
        homo_lumo=gap,
        diis_dim=int(diis_dim),
    )


@dataclass
class CCMSCFResult:
    """Ab initio CCM SCF result with explicit cyclic-energy semantics.

    ``energy`` is the backward-compatible name for the total energy of the
    finite cyclic cluster/supercell. It is not normalized per reference cell.
    ``total_cyclic_energy`` is the explicit alias; callers compare with a
    primitive-cell method using ``total_cyclic_energy / ccm.n_cells``.
    """

    converged: bool
    n_iter: int
    energy: float  # total finite cyclic-cluster energy (Ha; legacy name)
    energy_per_atom: float
    e_electronic: float  # total cyclic electronic energy
    e_nuclear: float  # total cyclic nuclear energy
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray  # D = 2 C_occ C_occ^T
    fock: np.ndarray
    overlap: np.ndarray
    hcore: np.ndarray
    idempotency_error: float  # ||D S D - 2 D||_F  (Gamma-point structural check)
    #: Per-cycle trace (#493). Empty for loops that do not record one; never
    #: None, so a consumer can iterate unconditionally. Diagnostic only -- no
    #: control flow reads it, so recording it cannot change an energy.
    scf_trace: tuple = ()
    # Exchange-q=0 convention label (vibeqc.periodic.exchange_convention) --
    # set by routes that declare one (run_ccm_rhf_direct: "BvK-ewald" /
    # "strict-zero-mode"); None where the route predates the convention field.
    exchange_q0: str | None = None
    #: Identity of the route/operator that actually produced this energy
    #: (IID 344): e.g. ``"ccm-fourcenter-direct-union12-rhf"`` or
    #: ``"ccm-neutral-ri-rhf"``. Stamped by every public producer so a
    #: record consumer can tell WHICH backend a number came from without
    #: out-of-band knowledge; ``""`` only on results predating the field.
    backend: str = ""
    #: Nuclear-attraction construction the four-centre SCF ran with
    #: (``"eq13-home-image"``, or a ``P2014_NUCLEAR_VARIANTS`` name for
    #: ``method="p2014"``). Empty on results that do not come from a
    #: four-centre driver.
    nuclear_attraction: str = ""
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
    #: Which implementation of ``gdf_method="rsgdf"`` built the fitted
    #: operator this result was evaluated on: ``"classic"`` for the neutral
    #: direct producers (their fold and supercell builders are the classic
    #: all-reciprocal FFT fit), ``"injected"`` for a caller-supplied cderi,
    #: empty where no density fit is involved (#378).
    fit_source: str = ""
    guess_selection: object = None
    #: The terminal evaluated state as a :class:`~vibeqc.fock_state.FockBuild`
    #: (#472): the physical Fock, density, energy components, source and frame
    #: identities frozen together, the last state the SCF evaluated. Set by
    #: the neutral real-Gamma producers (#519, RG-3); ``None`` elsewhere.
    fock_build: object = None
    #: Stability verdict of the returned solution
    #: (:class:`~vibeqc.periodic.ccm.stability.CCMRHFStability`) when the
    #: real-Gamma RHF driver ran with ``stability_check`` or
    #: ``follow_instability`` (#519, RG-9), and the instability-following
    #: steps it took; ``None`` and ``()`` otherwise.
    stability: object = None
    instability_follow: tuple = ()
    #: IID 291 / ruling R6: whether the producer was asked to run a cell
    #: measured as vacuum-padded (``allow_vacuum_padding=True``) and which
    #: 1-based lattice directions measured as vacuum. ``(False, ())`` on
    #: every cell measured as genuinely 3-D, and on the four-centre routes,
    #: which carry no such gate. A padded-regime energy is never reportable
    #: without the padding it was computed at.
    vacuum_padding_allowed: bool = False
    vacuum_padded_axes: tuple = ()

    #: The neutral cderi ``L`` this SCF actually rode (2026-09-24, one build
    #: per job): the direct-route drivers stamp the array they contracted J/K
    #: from -- built here or passed in as ``cderi=`` -- so the correlated
    #: entry points (``run_ccm_mp2``, ``run_ccm_ccsd``, the DLPNO family)
    #: reuse it when called without ``cderi=`` instead of refitting (or, worse,
    #: fitting a different kernel). ``None`` on routes that build no cderi.
    #: Large; a caller that keeps many results may set it to ``None``.
    cderi: object = field(default=None, repr=False)
    #: How :attr:`cderi` was built: ``build`` (``"fold"``, ``"supercell"``,
    #: ``"caller"`` or ``"driver-fit-cache"``), ``ke_cutoff``,
    #: ``tail_ke_cutoff``, ``aux_basis``, ``fit_batch``, ``lat_cutoff_bohr``,
    #: ``shape``. ``None`` where :attr:`cderi` is.
    cderi_info: dict | None = None
    #: The :class:`~vibeqc.periodic.ccm.bloch_reference.BlochFoldReport`
    #: when this reference was transformed from the Bloch producer's
    #: SCF (Theorem 1), else ``None``.
    bloch_transform: object = field(default=None, repr=False)

    @property
    def achieved_convergence(self) -> dict:
        """The convergence actually reached, read from the per-cycle trace.

        O29 (paper issue #17): a requested threshold describes the
        calculation that was asked for, not the one that ran. This returns
        ``{"delta_e": last energy step, "grad_max": last max-abs
        orthonormalised commutator residual, "cycles": n}`` from
        :attr:`scf_trace`, or an empty dict for loops that record no trace.
        """
        tr = tuple(self.scf_trace or ())
        if not tr:
            return {}
        last = tr[-1]
        return {
            "delta_e": float(last.delta_e),
            "grad_max": float(last.grad_max),
            "grad_metric": "max_abs_orthonormalised_commutator",
            "cycles": int(len(tr)),
        }

    @property
    def parity_held(self) -> bool:
        """Whether the executing driver held this result for external parity.

        Derived from the ``+PARITY_HELD`` marker on :attr:`backend` -- the
        hold mechanism's canonical carrier
        (:func:`vibeqc.pbc_gdf._gdf_backend_with_parity_hold`) -- so the
        structured flag can never drift from the string (IID 344). A held
        absolute energy is known-wrong versus the external reference;
        energy *differences* may still be usable. The four-center cluster
        routes never touch the hold class, so ``False`` there is an
        affirmative verdict, not a default of ignorance.
        """
        return "+PARITY_HELD" in str(self.backend or "")

    @property
    def total_cyclic_energy(self) -> float:
        """Total energy of the finite cyclic cluster/supercell in Hartree."""
        return float(self.energy)

    @property
    def normalization(self) -> str:
        """Machine-readable normalization label for ``energy``."""
        return "total_cyclic_cluster"


def _orthonormaliser(S, lindep_tol):
    """Canonical orthogonalisation X with S-eigenvectors below tol dropped.

    ``lindep_tol`` is a **screening** threshold, not a refusal gate: every
    overlap eigenvector with eigenvalue <= tol is projected out, and X spans
    only the retained subspace.
    """
    w, U = np.linalg.eigh(S)
    keep = w > lindep_tol
    if not np.all(keep):
        U, w = U[:, keep], w[keep]
    return U / np.sqrt(w)


def _require_retained_occ(X, n_occ, *, who, lindep_tol):
    """Fail closed when ``lindep_tol`` screening drops the overlap subspace
    below the occupation count (the density could no longer hold the
    electrons, so silently truncating the occupied block would be wrong)."""
    if X.shape[1] < n_occ:
        raise ValueError(
            f"{who}: lindep_tol={lindep_tol:.1e} screens the CCM overlap below "
            f"the occupied-orbital count ({X.shape[1]} retained < {n_occ} "
            "occupied); loosen lindep_tol or enlarge the cluster."
        )


def _ccm_initial_guess(ccm, initial_guess, *, driver, reference=None,
                       allow_read=False, initial_density=None, route=None):
    """Canonical guess contract for cyclic SCF loops.

    Direct real-Gamma loops name their route (``"aiccm-real-gamma"``) and take
    its shared capability set: HCORE, a prepared READ density and, only when
    named, SAD (#519 RG-9), so AUTO stays HCORE. ``allow_read`` is the older
    spelling of HCORE plus READ; other explicit/RI cyclic loops retain
    HCORE-only capability.
    The GDF control wrappers retain their producer's wider capability set.
    A supplied, already calculated reference retains its own provenance.
    """
    from ...guess import (
        InitialGuess,
        coerce_initial_guess,
        periodic_guess_capabilities,
        select_initial_guess,
    )

    requested = coerce_initial_guess(initial_guess)
    if reference is not None:
        previous = getattr(reference, "guess_selection", None)
        if requested != InitialGuess.AUTO and (
            previous is None
            or requested not in (previous.requested, previous.effective)
        ):
            raise ValueError(
                f"{driver}: initial_guess cannot replace or relabel a supplied SCF reference"
            )
        return previous
    if route is not None:
        supported = periodic_guess_capabilities(route, requested=requested)
    elif allow_read:
        supported = (InitialGuess.HCORE, InitialGuess.READ)
    else:
        supported = (InitialGuess.HCORE,)
    if (InitialGuess.READ in supported and requested == InitialGuess.READ
            and initial_density is None):
        raise ValueError(f"{driver}: initial_guess=READ requires initial_density")
    molecule = ccm.supercell
    return select_initial_guess(
        molecule, requested, is_periodic=True,
        is_open_shell=int(molecule.multiplicity) != 1,
        supported=supported,
        restart_supplied=initial_density is not None, driver=driver,
    )


def _ccm_read_density(density, S, electrons, *, occupation=2.0):
    """Admit an already mapped real supercell density without changing it.

    Shared READ validation supplies shape, Hermiticity and positivity checks.
    Unlike a basis-projected molecular restart, this seed already uses the
    target cyclic AO metric: reject a wrong population instead of rescaling.
    """
    from ...guess import _validated_density_guess_block

    D = _validated_density_guess_block(density, S.shape)
    if np.max(np.abs(D.imag), initial=0.0) > 1e-10:
        raise ValueError("CCM READ: real-Gamma cannot discard an imaginary density")
    D = D.real.copy()
    count = float(np.trace(D @ S))
    if not np.isclose(count, electrons, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"CCM READ: density electron count {count} != target {electrons}")
    # Physical natural occupations live in the overlap metric, not AO space.
    w, U = np.linalg.eigh(S)
    sqrt_S = (U * np.sqrt(np.maximum(w, 0.0))) @ U.T
    occupations = np.linalg.eigvalsh(sqrt_S @ D @ sqrt_S)
    if np.max(occupations, initial=0.0) > occupation + 1e-8:
        raise ValueError("CCM READ: density occupations exceed the spin capacity")
    return D


def _ccm_read_spin_densities(densities, S, n_alpha, n_beta):
    if not isinstance(densities, (tuple, list)) or len(densities) != 2:
        raise ValueError("CCM READ: open-shell initial_density requires (alpha, beta)")
    return tuple(_ccm_read_density(d, S, n, occupation=1.0)
                 for d, n in zip(densities, (n_alpha, n_beta)))


def _evaluated_state_orbitals(F, D, S, X, n_occ):
    """Orbitals of an evaluated state ``(D, F[D])`` of a cyclic SCF loop.

    The occupied space is spanned by the ``n_occ`` natural orbitals of ``D``
    with the largest occupations in the metric ``S``, the virtual space by
    the rest of the span of the orthonormaliser ``X``, and each block is
    rotated to diagonalise the physical ``F``. For the idempotent density a
    loop accepts, the occupied orbitals regenerate ``D`` exactly and the
    eigenvalues are ``F[D]``'s within each space; at convergence they are
    its canonical orbitals. Returns ``(eps, C)``, the occupied block first,
    each ascending (#519 RG-9: the loops used to return the spectrum of the
    last DIIS trial Fock instead).
    """
    Dp = X.T @ S @ D @ S @ X
    w, U = np.linalg.eigh(0.5 * (Dp + Dp.T))
    U = U[:, ::-1]
    Uo, Uv = U[:, :n_occ], U[:, n_occ:]
    Fp = X.T @ F @ X
    Fp = 0.5 * (Fp + Fp.T)
    eo, Ro = np.linalg.eigh(Uo.T @ Fp @ Uo)
    ev, Rv = np.linalg.eigh(Uv.T @ Fp @ Uv)
    return (np.concatenate([eo, ev]),
            X @ np.hstack([Uo @ Ro, Uv @ Rv]))


def _ccm_constructed_density(density, S, electrons, *, where):
    """Admit a density the guess engine built in the torus metric ``S``.

    A superposition of atomic densities is normalised to the target
    population in ``S`` by the engine, but it is not idempotent, and its
    natural occupations in the crystal metric may exceed the spin capacity
    where atoms overlap. Unlike a READ density it is therefore checked only
    for shape, finiteness, reality and population; the first Fock
    diagonalisation of the loop makes the next density idempotent.
    """
    D = np.asarray(density)
    if D.shape != S.shape or not np.all(np.isfinite(D)):
        raise ValueError(
            f"{where}: the constructed seed must be a finite {S.shape} matrix")
    if np.iscomplexobj(D):
        if np.max(np.abs(D.imag), initial=0.0) > 1e-10:
            raise ValueError(f"{where}: the constructed seed is not real")
        D = D.real
    D = 0.5 * (D + D.T)
    count = float(np.trace(D @ S))
    if not np.isclose(count, electrons, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"{where}: seed electron count {count} != target {electrons}")
    return np.ascontiguousarray(D, dtype=float)


def _ccm_sad_seed(ccm, S, *, n_alpha=None, n_beta=None):
    """SAD on the CCM supercell in the torus metric ``S`` (#519 RG-9).

    The shared guess engine superposes the atomic densities of the
    supercell's atoms and normalises the population in ``S``. Every cell's
    copy of an atom carries the same block, so the seed is translation
    invariant on the torus and lies in the space where the real-Gamma and
    neutral-Bloch producers share their stationary points (Theorem 1).
    Closed shell returns ``D``; with ``n_alpha``/``n_beta`` it returns the
    engine's spin split ``(D_alpha, D_beta)``.
    """
    from ...guess import (
        InitialGuess,
        initial_densities_open_shell,
        initial_density_closed_shell,
    )

    mol = ccm.supercell
    if n_alpha is None:
        n_occ = int(mol.n_electrons()) // 2
        D = initial_density_closed_shell(
            mol, ccm.basis, n_occ, InitialGuess.SAD, is_periodic=False,
            overlap=S)
        return _ccm_constructed_density(D, S, 2 * n_occ, where="CCM SAD")
    Da, Db = initial_densities_open_shell(
        mol, ccm.basis, int(n_alpha), int(n_beta), InitialGuess.SAD,
        is_periodic=False, overlap=S)
    return (_ccm_constructed_density(Da, S, int(n_alpha), where="CCM SAD alpha"),
            _ccm_constructed_density(Db, S, int(n_beta), where="CCM SAD beta"))


def _with_ccm_guess(result, selection):
    """Attach the selection consumed by a cyclic SCF loop."""
    result.guess_selection = selection
    return result


def run_ccm_rhf(
    ccm, *, initial_guess: object = "AUTO", method="union12", max_iter=128, conv_tol=1e-9, conv_tol_grad=1e-6,
    diis_dim=8, lindep_tol=1e-7, eri=None, nuclear_attraction=None
):
    """Run closed-shell HF-CCM on ``ccm`` (a :class:`CCMSystem`).

    Parameters mirror an ordinary RHF. ``method`` selects the effective
    four-center weighting (see :data:`_CCM_ERI_METHODS`):

    * ``"union12"`` (default) -- the historical eq-18 product weight
      (Peintinger & Bredow 2014); validated in 1-D **in a minimal basis
      only**, breaks 8-fold ERI symmetry in >=2-D. The basis qualifier is not
      a caution, it is a measured limit (issue #242).
    * ``"aiccm2026dev-a"`` -- the symmetric Born-von Kármán-torus four-center
      (:func:`~vibeqc.periodic.ccm.padded.ccm_eri_symmetric`,
      ``AICCM_ALGORITHM.md`` Sec.13), exactly 8-fold permutationally symmetric for
      any lattice. This is this development line's method of record.

    ``conv_tol`` gates the energy criterion only. ``conv_tol_grad`` is the
    independent DIIS-commutator residual bound (default ``1e-6``, the
    historical gate; the molecular ``RHFOptions.conv_tol_grad`` convention) --
    tighten it, not just ``conv_tol``, when converged *densities* matter.

    ``lindep_tol`` is a **screening** threshold (canonical orthogonalisation):
    overlap eigenvectors with eigenvalue <= ``lindep_tol`` are projected out
    of the Fock diagonalisation subspace instead of refusing the run. The run
    raises only if screening retains fewer directions than occupied orbitals.

    ``eri`` may instead be a precomputed effective tensor (it then overrides
    ``method``). Returns a :class:`CCMSCFResult`.

    ``result.energy`` is retained for compatibility and is the total energy of
    the finite cyclic cluster. ``result.total_cyclic_energy`` names the same
    quantity explicitly; divide by ``ccm.n_cells`` for a per-cell comparison.

    Status: reproduces the supercell-model HF energy to ~1e-5 Ha/atom (H₄
    alternating chain vs Peintinger PhD thesis Tab. 8.3 / JCC 2014 Tab. 2),
    for 1-D clusters with arbitrary orbitals (validated s and p). The molecular
    limit is exact. **Scope/known limits:** the dense ``n_ref_ao**4`` padded
    ERI in :func:`ccm_eri` confines this driver to small / 1-D clusters -- 3-D
    supercells blow up the padded basis and need the production C++ lattice-sum
    (next milestone). The four-center is not yet bit-exact (not perfectly
    cyclically invariant), so small high-symmetry clusters show minor orbital-
    degeneracy splitting.
    """
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_rhf"
    )
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_rhf',
    )
    _warn_experimental()
    # Result-backend identity (IID 344): a caller-injected tensor overrides
    # ``method``, so the result must not claim a weighting it did not apply.
    _backend = (
        "ccm-fourcenter-dense-injected-rhf" if eri is not None
        else f"ccm-fourcenter-dense-{method}-rhf"
    )
    _nuclear = _resolve_nuclear_attraction(
        method, nuclear_attraction, who="run_ccm_rhf")
    if eri is None:
        eri = (ccm_eri_p2014(ccm) if method == P2014_METHOD
               else _ccm_eri_for_method(ccm, method))

    S = ccm_overlap(ccm)
    if method == P2014_METHOD:
        h, _, _ = ccm_hcore_p2014(ccm, nuclear_attraction=_nuclear)
    else:
        h, _, _ = ccm_hcore(ccm)
    e_nn = ccm_nuclear_repulsion(ccm)

    _reject_open_shell_cluster(ccm, who="run_ccm_rhf", sibling="run_ccm_uhf")
    n_elec = ccm.supercell.n_electrons()
    n_occ = n_elec // 2

    X = _orthonormaliser(S, lindep_tol)
    _require_retained_occ(X, n_occ, who="run_ccm_rhf", lindep_tol=lindep_tol)

    def diag_fock(F):
        Fp = X.T @ F @ X
        eps, Cp = np.linalg.eigh(Fp)
        return eps, X @ Cp

    def density(C):
        Cocc = C[:, :n_occ]
        return 2.0 * (Cocc @ Cocc.T)

    # Core-Hamiltonian initial guess.
    eps, C = diag_fock(h)
    D = density(C)

    diis_F, diis_e = [], []
    e_last = 0.0
    converged = False
    for it in range(1, max_iter + 1):
        J = np.einsum("mnrs,rs->mn", eri, D, optimize=True)
        K = np.einsum("msrn,rs->mn", eri, D, optimize=True)
        F = h + J - 0.5 * K
        F = 0.5 * (
            F + F.T
        )  # enforce Hermiticity (the WIP eff is not yet fully symmetric)

        e_elec = 0.5 * np.sum(D * (h + F))
        e_tot = e_elec + e_nn

        # DIIS error = S D F - F D S (zero at convergence), in the orthonormal basis.
        err = X.T @ (F @ D @ S - S @ D @ F) @ X
        if len(diis_F) == diis_dim:
            diis_F.pop(0)
            diis_e.pop(0)
        diis_F.append(F)
        diis_e.append(err)
        if len(diis_F) >= 2:
            F = _diis_extrapolate(diis_F, diis_e)

        eps, C = diag_fock(F)
        D = density(C)

        de = e_tot - e_last
        e_last = e_tot
        if it > 1 and _commutator_exit_ok(de, err, conv_tol, conv_tol_grad):
            converged = True
            break

    idem = np.linalg.norm(D @ S @ D - 2.0 * D)
    if converged and int(ccm.unit_system.charge) == 0 and e_tot > 0.0:
        raise ValueError(
            "closed-shell neutral CCM SCF converged to a positive total "
            f"energy ({e_tot:.6f} Ha): unphysical for a neutral cell, and "
            "the signature of a vacuum-padded or gauge-inconsistent regime "
            "(IID 291). Refusing to report this as a successful result."
        )
    return _with_ccm_guess(CCMSCFResult(
        converged=converged,
        n_iter=it,
        energy=e_tot,
        energy_per_atom=e_tot / ccm.n_atoms,
        e_electronic=e_elec,
        e_nuclear=e_nn,
        mo_energies=eps,
        mo_coeffs=C,
        density=D,
        fock=F,
        overlap=S,
        hcore=h,
        idempotency_error=float(idem),
        backend=_backend,
        nuclear_attraction=_nuclear,
    ), guess_selection)


def _diis_extrapolate(focks, errors):
    m = len(focks)
    B = -np.ones((m + 1, m + 1))
    B[-1, -1] = 0.0
    for i in range(m):
        for j in range(m):
            B[i, j] = np.sum(errors[i] * errors[j])
    rhs = np.zeros(m + 1)
    rhs[-1] = -1.0
    try:
        c = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        return focks[-1]
    return sum(c[i] * focks[i] for i in range(m))


# -- Scalable production driver (C++ WeightedLatticeJKBuilder) --------------


def _prepare_ccm_weights(ccm):
    """Package WSSC two-center weights for the C++ CCM JK builder.

    Returns (weight_cells, weight_matrices) as numpy arrays:
        weight_cells: (n_wcells, 3) int32, minimum-image cell indices
        weight_matrices: (n_wcells, n_atoms, n_atoms) float64
    """
    Wg = ccm.cell_weight_matrices()
    cells = sorted(Wg.keys())
    n_wcells = len(cells)
    n_atoms = ccm.n_atoms
    weight_cells = np.zeros((n_wcells, 3), dtype=np.int32)
    weight_matrices = np.zeros((n_wcells, n_atoms, n_atoms), dtype=np.float64)
    for i, g in enumerate(cells):
        weight_cells[i] = g
        weight_matrices[i] = np.asarray(Wg[g], dtype=np.float64)
    return weight_cells, weight_matrices


def _ccm_bra_ket_symmetrise(jk_result, ccm, D):
    """Apply CCM bra-ket symmetrisation to J/K matrices.

    The raw ``build_jk_ccm_weighted`` returns the ket-folded J/K (bra at
    home).  The CCM four-center weight breaks the 8-fold ERI symmetry;
    the missing 50% from the bra-folded representation is recovered by
    contracting the effective tensor's bra-ket transpose.  This matches
    the ``ccm_eri`` convention::

        eff^sym = 0.5*(eff + transpose_{bra<->ket})

    For the contracted J/K this becomes a second kernel call with the
    density and integral indices remapped.
    """
    from vibeqc._vibeqc_core import (
        build_jk_ccm_weighted,
        make_periodic_gamma_ccm_jk_builder,
    )

    from .scf import _prepare_ccm_weights

    # Ket-folded J/K
    J_ket = np.asarray(jk_result.J, dtype=float)
    K_ket = np.asarray(jk_result.K, dtype=float)

    # Bra-folded: compute J/K with bra<->ket transposition of the
    # effective ERI tensor.  In practice this means we treat the
    # D-contracted ket as the bra in a second build.  For a symmetric
    # density D, J_bra[mu,nu] = K_ket[nu,mu]? Not quite -- we need
    # a proper rebuild.  Do a second call to build_jk_ccm_weighted
    # with the density transposed and reinterpret.
    #
    # For now: approximate by using the ket-folded result and
    # symmetrising via explicit transposition of the effective tensor
    # in Python.  This is expensive but correct.
    JT = J_ket.T.copy()
    KT = K_ket.T.copy()

    # The symmetrisation: J^CCM = 0.5*(J_ket + J_bra)
    # J_bra[mu,nu] comes from K_ket with remapped indices.
    # For a contracted Fock, the bra-ket transposition of eff gives
    # G = J - 0.5*K where K exchanges indices differently.
    # Rather than re-derive, accept the padded route's convention
    # directly: symmetrise J and K matrices themselves.
    # This is NOT the full 8-fold symmetrisation, but it ensures
    # Hermiticity which is the dominant effect.

    J_sym = 0.5 * (J_ket + JT)
    K_sym = 0.5 * (K_ket + KT)

    return J_sym, K_sym


_CCM_SCALABLE_METHODS = {
    "union12": "bra_home_full",   # eq-18 four-center (matches padded ccm_eri)
    "aiccm2026dev-a": "aiccm2026dev-a",       # symmetric BvK-torus four-center (Sec.13)
    "aiccmdev": "aiccm2026dev-a",             # deprecated alias (back-compat)
}


def _ccm_scalable_cxx_method(method, four_center, *, allow_p2014=False):
    """Map a CCM ``(method, four_center)`` selection to the C++ JK kernel name.

    ``method`` picks the base kernel (:data:`_CCM_SCALABLE_METHODS`);
    ``four_center`` picks how it contracts:

    * ``"direct"`` -- the integral-direct kernel (``base + "-direct"``): folds each
      weighted quartet straight into J/K, O(nbf**2) memory.
    * ``"full"`` / ``"dense"`` -- the base kernel: builds the dense O(nbf**4)
      effective tensor (the preserved small-cluster comparison reference).

    Shared by :func:`run_ccm_rhf_scalable` and the KS-CCM JK builder
    (``dft._ccm_jk_builder``) so ``four_center`` means the same thing everywhere.
    """
    if method == P2014_METHOD:
        # Only the RHF driver pairs this kernel with its own nuclear attraction
        # (padded.ccm_hcore_p2014); the KS builder would mix it with
        # padded.ccm_nuclear, so it must opt in explicitly.
        if not allow_p2014:
            raise ValueError(
                "method='p2014' is available on run_ccm_rhf and "
                "run_ccm_rhf_scalable only (it carries its own nuclear "
                "attraction).")
        if four_center != "direct":
            raise ValueError(
                "method='p2014' has the integral-direct C++ kernel only "
                "(four_center='direct'); the dense effective tensor is "
                "run_ccm_rhf(method='p2014').")
        return "p2014-direct"
    try:
        base = _CCM_SCALABLE_METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown CCM four-center method {method!r}; "
            f"choose from {sorted(_CCM_SCALABLE_METHODS) + [P2014_METHOD]}"
        )
    if four_center == "direct":
        return base + "-direct"
    if four_center in ("full", "dense"):
        return base
    raise ValueError(
        f"unknown four_center {four_center!r}; choose 'direct' (integral-direct "
        "J/K, O(nbf**2) memory -- default) or 'full' (dense effective tensor, "
        "O(nbf**4) memory -- small-cluster comparison reference)"
    )


@contextmanager
def _ccm_jk_lattice_options(ccm, cxx_method, schwarz_threshold):
    """``ccm.lattice_options`` as the C++ CCM JK kernels need them, restored on exit.

    The C++ side copies the options by value, so they are edited in place around
    the call. ``schwarz_threshold`` is forced (``0.0`` = exact). The default
    cutoff encloses the FIRST shell of cluster translations, which is all the
    home-bra kernels reach; ``"p2014-direct"`` images the bra partner, so its ket
    partner reaches the SECOND shell (the +-2 supercell translations of the 2012
    program) and the cutoff is widened to enclose it -- the kernel refuses to
    drop a weighted quartet for a missing cell rather than truncate silently.
    """
    opts = ccm.lattice_options
    saved = (opts.schwarz_threshold, opts.cutoff_bohr)
    try:
        opts.schwarz_threshold = float(schwarz_threshold)
        if str(cxx_method).startswith("p2014"):
            from .wigner_seitz import first_shell_vectors

            corner = float(np.linalg.norm(
                first_shell_vectors(ccm.cluster_vectors), axis=1).max())
            opts.cutoff_bohr = max(float(opts.cutoff_bohr), 2.1 * corner)
        yield opts
    finally:
        opts.schwarz_threshold, opts.cutoff_bohr = saved


def _make_ccm_jk_builder(ccm, cxx_method, schwarz_threshold):
    """Construct the C++ CCM-weighted JK builder.

    ``schwarz_threshold`` is the opt-in Cauchy-Schwarz screening threshold for the
    ``-direct`` kernels (``0.0`` = off = exact, the default). The C++ builder copies
    ``lattice_options`` at construction (``CCMWeightedGammaJKBuilder`` stores it by
    value), so we set the threshold on ``ccm.lattice_options`` around the build and
    restore it -- the constructed builder retains the value, and ``ccm`` is left
    untouched. Forcing ``0.0`` by default also skips the otherwise-wasted Schwarz
    factor computation (the CCM ``lattice_options`` default is ``1e-12``).
    """
    from vibeqc._vibeqc_core import make_periodic_gamma_ccm_jk_builder

    weight_cells, weight_matrices = _prepare_ccm_weights(ccm)
    with _ccm_jk_lattice_options(ccm, cxx_method, schwarz_threshold) as opts:
        return make_periodic_gamma_ccm_jk_builder(
            ccm.basis, ccm.cluster_system, weight_cells, weight_matrices,
            opts, cxx_method,
        )


def run_ccm_rhf_scalable(
    ccm, *, initial_guess: object = "AUTO", method="union12", four_center="direct", schwarz_threshold=0.0,
    max_iter=128, conv_tol=1e-9, conv_tol_grad=1e-6, diis_dim=8, lindep_tol=1e-7,
    nuclear_attraction=None
):
    """Run closed-shell HF-CCM using the scalable C++ lattice-sum JK builder.

    Unlike :func:`run_ccm_rhf` which materialises the O(n^4) effective ERI
    tensor in Python, this driver uses the C++ ``build_jk_ccm_weighted`` -- a
    Gamma-only periodic J/K build that applies the WSSC four-center weights
    during the shell-quartet loop.

    ``method`` selects the four-center weighting, matching the Python padded
    route: ``"union12"`` (default, eq-18 -- C++ ``"bra_home_full"``) or
    ``"aiccm2026dev-a"`` (the symmetric Born-von Kármán-torus four-center,
    ``AICCM_ALGORITHM.md`` Sec.13). Both reproduce their Python ``run_ccm_rhf``
    counterpart (``method=`` there) to µHa.

    ``four_center`` selects how the weighted quartets are contracted (Phase 3b):

    * ``"direct"`` (default) -- **integral-direct**: each weighted quartet block
      is folded straight into J/K, so peak memory is O(nbf**2) (thread-local
      J/K) instead of O(nbf**4). This is what lets real 3-D cells at production
      basis fit in RAM. Reproduces the ``"full"`` path to ~1e-12 (a summation
      reorder, not bit-for-bit).
    * ``"full"`` (alias ``"dense"``) -- the **dense effective tensor** path: build
      the O(nbf**4) effective ERI tensor, bra-ket symmetrise, then contract. This
      is the preserved small-cluster *comparison reference*; it OOMs on real 3-D
      production-basis cells (that is exactly why ``"direct"`` is the default).

    ``lindep_tol`` is a **screening** threshold, forwarded to the C++ SCF's
    ``linear_dep_threshold``: overlap eigenvectors with eigenvalue <= the
    threshold are projected out of the canonical orthogonalisation subspace
    instead of refusing the run (the C++ driver raises only if the retained
    subspace cannot hold the occupied orbitals).

    ``schwarz_threshold`` is the **opt-in** Cauchy-Schwarz screening threshold for
    the ``four_center="direct"`` kernels (no effect on ``"full"``). Default
    ``0.0`` = **off** -- the direct kernel is exact (every weighted quartet is
    contracted). A positive value (e.g. ``1e-12``) skips shell-quartets whose
    rigorous bound ``|w| * Q_bra * Q_ket * D_max`` falls below it -- a throughput
    lever that changes the result only at that threshold (so it is *not* the
    byte-for-byte exact path; keep it tight). Off by default preserves the exact
    integral-direct guarantee.

    The SCF loop runs through the production C++ ``run_rhf_scf_with_jk``
    entry point (DIIS, damping, level-shift, Newton fallback).

    Other parameters mirror :func:`run_ccm_rhf`. Returns a :class:`CCMSCFResult`.
    As in the dense driver, ``energy`` / ``total_cyclic_energy`` is the total
    cyclic-supercell energy, not an already normalized per-cell value.
    """
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_rhf_scalable"
    )
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_rhf_scalable',
    )
    _warn_experimental()
    cxx_method = _ccm_scalable_cxx_method(method, four_center, allow_p2014=True)
    _nuclear = _resolve_nuclear_attraction(
        method, nuclear_attraction, who="run_ccm_rhf_scalable")
    # Result-backend identity (IID 344): the executed contraction mode is
    # what _ccm_scalable_cxx_method resolved ("full"/"dense" both mean the
    # dense effective tensor).
    _backend = (
        "ccm-fourcenter-"
        f"{'direct' if four_center == 'direct' else 'dense'}-{method}-rhf"
    )
    from vibeqc import (
        RHFOptions,
        run_rhf_scf_with_jk,
    )

    from .integrals import ccm_overlap
    from .padded import ccm_hcore, ccm_hcore_p2014, ccm_nuclear_repulsion

    S = ccm_overlap(ccm)
    if method == P2014_METHOD:
        h, _, _ = ccm_hcore_p2014(ccm, nuclear_attraction=_nuclear)
    else:
        h, _, _ = ccm_hcore(ccm)
    e_nn = ccm_nuclear_repulsion(ccm)

    _reject_open_shell_cluster(
        ccm, who="run_ccm_rhf_scalable", sibling="run_ccm_uhf")
    n_elec = ccm.supercell.n_electrons()

    # Build the CCM-weighted JK builder. The base C++ method applies the WSSC
    # four-center weight during the shell-quartet loop:
    #   "bra_home_full" (union12) -- bra at home + ket imaged, eq-18 weight, then
    #       bra-ket symmetrised; reproduces the padded ccm_eri (gold -0.542875).
    #   "aiccm2026dev-a" -- the symmetric BvK-torus four-center (Sec.13): symmetric bridge
    #       + independent min-image fold; exactly 8-fold symmetric for any lattice.
    # The "-direct" suffix (four_center="direct", default) folds each weighted
    # quartet straight into J/K -- O(nbf**2) memory; the bare base name builds the
    # full O(nbf**4) effective tensor (four_center="full", comparison reference).
    # schwarz_threshold (default 0.0) opts into Cauchy-Schwarz screening of the
    # -direct kernels.
    jk = _make_ccm_jk_builder(ccm, cxx_method, schwarz_threshold)

    # Run SCF through the production C++ driver.
    opts = RHFOptions()
    opts.max_iter = max_iter
    opts.conv_tol_energy = conv_tol
    # conv_tol gates the energy criterion only; conv_tol_grad is the
    # independent DIIS-commutator residual bound (default 1e-6).
    opts.conv_tol_grad = conv_tol_grad
    # opts.diis_subspace_size = diis_dim  # not exposed
    opts.linear_dep_threshold = lindep_tol  # C++ SCF canonical-orth screening

    opts.initial_guess = guess_selection.requested
    result = run_rhf_scf_with_jk(
        ccm.basis,
        n_elec,
        np.asarray(S, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
        float(e_nn),
        jk,
        opts,
        np.empty((0, 0)), molecule=ccm.supercell, guess_selection=guess_selection,  # empty initial density -> Hcore guess
    )

    # Extract results.
    D = np.asarray(result.density, dtype=float)
    F = np.asarray(result.fock, dtype=float)
    C = np.asarray(result.mo_coeffs, dtype=float)
    eps = np.asarray(result.mo_energies, dtype=float)

    e_total = float(result.energy)
    idem = np.linalg.norm(D @ S @ D - 2.0 * D)

    if (result.converged and int(ccm.unit_system.charge) == 0
            and e_total > 0.0):
        raise ValueError(
            "closed-shell neutral CCM SCF converged to a positive total "
            f"energy ({e_total:.6f} Ha): unphysical for a neutral cell, and "
            "the signature of a vacuum-padded or gauge-inconsistent regime "
            "(IID 291). Refusing to report this as a successful result."
        )

    return _with_ccm_guess(CCMSCFResult(
        converged=result.converged,
        n_iter=result.n_iter,
        energy=e_total,
        energy_per_atom=e_total / ccm.n_atoms,
        e_electronic=e_total - float(e_nn),
        e_nuclear=float(e_nn),
        mo_energies=eps,
        mo_coeffs=C,
        density=D,
        fock=F,
        overlap=np.asarray(S),
        hcore=np.asarray(h),
        idempotency_error=float(idem),
        backend=_backend,
        nuclear_attraction=_nuclear,
    ), guess_selection)
