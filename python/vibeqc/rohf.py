"""Restricted open-shell Hartree--Fock (ROHF).

vibe-qc ships closed-shell RHF/RKS and unrestricted UHF/UKS as native
C++ drivers.  ROHF is the missing *spin-restricted* open-shell
reference: a single set of spatial orbitals partitioned into doubly
occupied (closed), singly occupied (open) and virtual shells, giving a
spin-pure determinant (``<S^2> = S(S+1)`` exactly, no spin
contamination).  It is the standard reference for ROHF-MP2 / ROHF-CC
and a cleaner CAS starting point than UHF for radicals and high-spin
states.

Architecture (see ``handovers/HANDOVER_ROHF.md``).  The production SCF loops live
in C++ (``run_uhf`` etc.) and cannot host ROHF's Roothaan coupling, so
this is a **pure-Python driver** built on the same two-electron JK seam
the C++ drivers expose to Python (``make_direct_jk_builder`` ->
``build_J`` / ``build_K`` closures).  It is additive (touches no C++),
and the coupling + energy algebra is plain NumPy, so it is verifiable
without the compiled core.

Theory --- Roothaan's single effective Fock operator (Coulson coupling):

    Roothaan, Rev. Mod. Phys. 32, 179 (1960), doi:10.1103/RevModPhys.32.179

Per-spin Fock from the closed/open densities (Coulomb from the *total*
density, exchange per spin)::

    J  = J(Da + Db);  Ka = K(Da);  Kb = K(Db)
    Fa = H + J - Ka;  Fb = H + J - Kb;  Fc = (Fa + Fb)/2

AO-basis shell projectors in the S metric (``Pc + Po + Pv = I``)::

    Pc = Db S          (closed,  doubly occupied)
    Po = (Da - Db) S   (open,    singly occupied)
    Pv = I - Da S      (virtual)

assembled into one symmetric effective Fock with Coulson's coupling
(closed-open block = Fb, open-virtual block = Fa, the rest Fc); see
:func:`roothaan_effective_fock`.  Energy is identical to UHF --- only
the orbital constraint differs::

    E = 1/2 (Tr[Da (H + Fa)] + Tr[Db (H + Fb)]) + E_nuc
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .ecp_metadata import (
    one_electron_hamiltonian,
)

try:
    from vibeqc._vibeqc_core import ADIIS as _ADIIS
    from vibeqc._vibeqc_core import DIIS as _DIIS
    from vibeqc._vibeqc_core import EDIIS as _EDIIS
    from vibeqc._vibeqc_core import KDIIS as _KDIIS
except ImportError:
    _EDIIS = None
    _ADIIS = None
    _KDIIS = None
    _DIIS = None

__all__ = [
    "ROHFOptions",
    "ROHFResult",
    "compute_rohf_gradient",
    "guest_saunders_canonicalize",
    "rohf_energy_weighted_density",
    "roothaan_effective_fock",
    "run_rohf",
]


def require_nonnegative_max_iter(max_iter: Any, *, route: str) -> None:
    """Validate the molecular SCF iteration budget (issue #392).

    Zero is the documented ORCA ``NOITER`` compatibility path: drivers
    evaluate the initial guess once without taking an SCF iteration. Negative,
    boolean, and non-integer budgets are invalid. A boolean needs an explicit
    check because ``bool`` is an ``int`` subclass in Python.
    """
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
        raise ValueError(
            f"{route}: max_iter must be a non-negative integer "
            f"(got {max_iter!r})"
        )
    if max_iter < 0:
        raise ValueError(
            f"{route}: max_iter must be a non-negative integer (got {max_iter})"
        )


# ---------------------------------------------------------------------------
# Options / result containers
# ---------------------------------------------------------------------------


@dataclass
class ROHFOptions:
    """Convergence + acceleration controls for :func:`run_rohf`.

    Field names mirror the C++ ``RHFOptions`` / ``UHFOptions`` surface so
    the molecular runner can forward the same user knobs.  Only the
    subset the pure-Python Roothaan loop honours is exposed; unknown
    extras are ignored by the dispatcher.
    """

    max_iter: int = 128
    # Effective core potentials, both routes, with the same field names as
    # the native RHFOptions so the runner and the sidecar attachment
    # (vibeqc.ecp_metadata.attach_inline_ecp_options_from_basis_sidecar)
    # treat this options object like the native ones.
    ecp_centers: list[Any] = field(default_factory=list)
    ecp_library: str = ""
    ecp_primitive_blocks: list[Any] = field(default_factory=list)
    ecp_primitive_centers: list[Any] = field(default_factory=list)
    ecp_effective_charges: list[float] = field(default_factory=list)
    ecp_total_ncore: int = 0
    conv_tol_energy: float = 1e-9
    conv_tol_grad: float = 1e-6
    use_diis: bool = True
    diis_start_iter: int = 1
    diis_subspace_size: int = 8
    damping: float = 0.5
    # SCF Fock-extrapolation accelerator. See ``SCFAccelerator`` enum.
    # "diis" = Pulay DIIS; "ediis_diis" is the production hybrid
    # (EDIIS for rough convergence, DIIS asymptotically) and is the
    # default (matches C++ RHF/UKS drivers). "r_cdiis" / "ad_cdiis" are
    # the Chupin et al. 2021 adaptive-depth variants -- see
    # ``diis_restart_tau`` / ``diis_adaptive_delta`` below.
    scf_accelerator: str = "ediis_diis"
    # EDIIS to DIIS commutator-norm threshold for "ediis_diis". Use
    # EDIIS while the RMS matrix-element commutator is above this value;
    # switch to plain DIIS once below. Default 1e-1 (same as C++).
    ediis_diis_switch_threshold: float = 1e-1
    # Adaptive-depth commutator-DIIS parameters (Chupin, Dupuy, Legendre
    # & Sere, ESAIM: M2AN 55, 2785 (2021)), consulted only by the
    # matching ``scf_accelerator``: tau (restart aggressiveness) for
    # "r_cdiis", delta (depth window) for "ad_cdiis". Same names,
    # defaults and meaning as ``RHFOptions`` / ``UHFOptions``; resolved
    # into a ``DIISDepthPolicy`` by ``_resolve_diis_depth_policy``, the
    # Python mirror of ``vibeqc::make_diis``.
    diis_restart_tau: float = 1e-4
    diis_adaptive_delta: float = 1e-4
    level_shift: float = 0.0
    # Dynamic (adaptive) damping (Zerner-Hehenberger 1979). When True,
    # ``damping`` is adjusted per-iteration: increased toward
    # ``dynamic_damping_max`` when energy oscillates upward, decreased
    # toward ``dynamic_damping_min`` when decreasing monotonically.
    dynamic_damping: bool = False
    dynamic_damping_min: float = 0.0
    dynamic_damping_max: float = 0.95
    # Saunders-Hillier shift policy, resolved through the same single
    # source of truth as every other molecular and periodic SCF driver
    # (``vibeqc.level_shift_at_iter``, cpp/include/vibeqc/level_shift.hpp):
    #   -1 = auto (shift the first min(5, max_iter-1) cycles, then release)
    #    0 = persistent (hold the shift at every iteration)
    #    N = explicit warm-up length, capped so a tail cycle stays unshifted
    # An explicit ``level_shift_schedule`` (see ``vq.LevelShiftSchedule``)
    # takes precedence over the warm-up logic. The shift never reaches the
    # canonicalising diagonalisation, so the reported orbital energies are
    # those of the unshifted converged Fock either way.
    level_shift_warmup_cycles: int = -1
    level_shift_schedule: List[float] = field(default_factory=list)
    # Second-order convergence thresholds (reserved; Python loop does not
    # yet implement SOSCF/TRAH/Newton). These mirror the C++ RHFOptions
    # fields so the ROHF/ROKS surface stays API-compatible.
    soscf_threshold: float = 0.0
    trah_threshold: float = 0.0
    newton_threshold: float = 0.0
    linear_dep_threshold: float = 1e-8
    # Disabled for ROHF by default: fractional degenerate-shell HF needs
    # general vector-coupling coefficients, while the shipped ROHF path
    # intentionally implements the integer high-spin Coulson form.
    fractional_open_shell: bool = False
    # Canonical molecular selector; READ/FRAGMO carry prepared spin densities.
    initial_guess: Any = "auto"
    read_path: str = ""
    read_density_alpha: Any = None
    read_density_beta: Any = None
    atomic_spins: List[int] = field(default_factory=list)
    # Two-electron build path.  density_fit=True routes J/K through the
    # RI density-fitting builder using ``aux_basis``; otherwise the
    # Schwarz-screened direct libint builder is used.
    density_fit: bool = False
    aux_basis: str = ""
    # RIJCOSX: when cosx=True (also needs density_fit + aux_basis), the
    # exchange K is built with the seminumerical chain-of-spheres kernel
    # (Coulomb J still from RI). cosx_grid_level mirrors the C++ SCF
    # drivers: 0 keeps the legacy sparse COSX grid; 1..4 select the 2021
    # 5-region GridX tiers (Helmich-Paris 2021); -1 auto-selects by basis
    # cardinality. The ROHF driver uses a single COSX grid (no multi-stage
    # progression) — adequate for a reference SCF; for tight gradients use
    # density_fit without cosx.
    cosx: bool = False
    cosx_grid_level: int = 0
    # Davidson/LobPCG iterative diagonalisation (optional).
    # When True and n_bf >= davidson_min_dim, the Fock diagonalisation
    # uses the iterative Davidson solver instead of np.linalg.eigh.
    use_davidson: bool = False
    davidson_min_dim: int = 100
    # Pre-configured DavidsonOptions (created internally when unset).
    davidson: Any = None


@dataclass
class _ScfStep:
    """Duck-typed twin of the C++ ``SCFIteration`` so the shared SCF-log
    formatter (:mod:`vibeqc.output.formats.scf_log`) renders ROHF traces
    with no special-casing."""

    iter: int
    energy: float
    delta_e: float = 0.0
    grad_norm: float = 0.0
    diis_subspace: int = 0


@dataclass(frozen=True)
class _OpenShellModel:
    """Frontier-shell occupation model for one Roothaan SCF run."""

    n_closed: int
    n_open_orbitals: int
    n_open_electrons: int

    @property
    def is_fractional(self) -> bool:
        return (
            self.n_open_orbitals > 0
            and self.n_open_electrons % self.n_open_orbitals != 0
        )


@dataclass
class ROHFResult:
    """Converged ROHF state.

    Exposes the union of the RHF surface (single ``mo_energies`` /
    ``mo_coeffs`` / ``density`` / ``fock``) and the open-shell spin
    densities (``density_alpha`` / ``density_beta``), plus
    ``mo_occupations`` (2 = closed, 1 = integer open, 0 = virtual, with
    uniform fractional values for a detected ROKS frontier ensemble).
    ``s_squared`` is exact for a restricted-open determinant.

    Orbital eigenvalues use the **Guest-Saunders** canonicalisation
    convention by default (``rohf_canonicalization="guest-saunders"``),
    which diagonalises the appropriate Fock sub-blocks independently
    within each subspace (closed / open / virtual).  A detected fractional
    ROKS frontier ensemble instead reports the Roothaan spectrum directly
    (the convention string is ``"roothaan-fractional-shell"``), because the
    integer closed/open/virtual partition is not defined through a
    fractionally occupied symmetry shell.  The original Roothaan eigenvalues
    are always preserved in ``mo_energies_roothaan``.  See
    :func:`guest_saunders_canonicalize` for the algorithm and
    M. F. Guest and V. R. Saunders, Mol. Phys. 28, 819 (1974).
    """

    energy: float
    e_electronic: float
    n_iter: int
    converged: bool
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    mo_occupations: np.ndarray
    density: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    fock: np.ndarray
    fock_alpha: np.ndarray
    fock_beta: np.ndarray
    s_squared: float
    s_squared_ideal: float
    n_alpha: int
    n_beta: int
    scf_trace: List[_ScfStep] = field(default_factory=list)
    guess_selection: Any = None
    restart_basis: Any = None
    e_dft_plus_u: float = 0.0
    method: str = "rohf"
    rohf_canonicalization: str = "guest-saunders"
    mo_energies_roothaan: Optional[np.ndarray] = None
    mo_coeffs_roothaan: Optional[np.ndarray] = None
    # ECP provenance, same names as the native results: which operator the
    # Hamiltonian carried and how many core electrons it removed.
    ecp_operator_applied: bool = False
    ecp_provenance_verified: bool = True
    ecp_xml_centers: tuple = ()
    ecp_xml_library: str = ""
    ecp_primitive_blocks: tuple = ()
    ecp_primitive_centers: tuple = ()
    ecp_effective_charges: tuple = ()
    ecp_total_ncore: int = 0

    @property
    def mo_coefficients(self) -> np.ndarray:
        """Long-form alias for :attr:`mo_coeffs` (read by the QVF writer)."""
        return self.mo_coeffs

    # -- Unrestricted-view aliases ------------------------------------------
    # ROHF uses ONE set of spatial orbitals for both spins, so the alpha
    # and beta orbital sets are identical (only the occupation differs:
    # n_alpha vs n_beta).  Exposing the UHF attribute names lets every
    # existing open-shell-aware consumer (the SCF-log orbital table, the
    # Molden / spin-density writers, population analysis) handle an ROHF
    # result with no special-casing --- the alpha/beta energy columns it
    # prints are correctly identical.
    @property
    def mo_energies_alpha(self) -> np.ndarray:
        return self.mo_energies

    @property
    def mo_energies_beta(self) -> np.ndarray:
        return self.mo_energies

    @property
    def mo_coeffs_alpha(self) -> np.ndarray:
        return self.mo_coeffs

    @property
    def mo_coeffs_beta(self) -> np.ndarray:
        return self.mo_coeffs

    @property
    def mo_coefficients_alpha(self) -> np.ndarray:
        return self.mo_coeffs

    @property
    def mo_coefficients_beta(self) -> np.ndarray:
        return self.mo_coeffs


# ---------------------------------------------------------------------------
# Roothaan coupling (pure function --- the mathematical core)
# ---------------------------------------------------------------------------


def roothaan_effective_fock(
    focka: np.ndarray,
    fockb: np.ndarray,
    dma: np.ndarray,
    dmb: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """Roothaan's single effective Fock matrix (Coulson coupling).

    Combines the per-spin Fock matrices ``focka`` / ``fockb`` into one
    Hermitian operator whose aufbau diagonalisation yields the ROHF
    orbitals.  Block structure in the closed (c) / open (o) / virtual (v)
    partition::

                closed   open    virtual
        closed   Fc       Fb       Fc
        open     Fb       Fc       Fa
        virtual  Fc       Fa       Fc

    with ``Fc = (Fa + Fb)/2``.  Ref. Roothaan, Rev. Mod. Phys. 32, 179
    (1960), Eq. (20); the projector construction follows the standard
    density-matrix form (e.g. PySCF ``scf.rohf.get_roothaan_fock``).

    Parameters
    ----------
    focka, fockb
        Per-spin AO Fock matrices ``Fa = H + J - Ka`` and
        ``Fb = H + J - Kb``.
    dma, dmb
        Alpha / beta AO density matrices (``dma`` includes the open
        shell; ``dmb`` is the closed-shell density, so ``dma - dmb`` is
        the open-shell density).
    s
        AO overlap matrix.
    """
    # Preserve the input dtype: real for molecular / Γ-point matrices,
    # complex-Hermitian for non-Γ k-points (the projector products and the
    # `.conj().T` symmetrisation below are complex-safe). Forcing float here
    # would silently discard the imaginary part at a periodic k-point.
    focka = np.asarray(focka)
    fockb = np.asarray(fockb)
    dma = np.asarray(dma)
    dmb = np.asarray(dmb)
    s = np.asarray(s)

    fc = (focka + fockb) * 0.5
    nao = s.shape[0]
    # Shell projectors (S-metric, idempotent for S-orthonormal MOs):
    #   Pc = Db S  -> closed (doubly occupied)
    #   Po = (Da - Db) S -> open (singly occupied)
    #   Pv = I - Da S    -> virtual
    pc = dmb @ s
    po = (dma - dmb) @ s
    pv = np.eye(nao) - dma @ s

    def proj(left: np.ndarray, mat: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left.conj().T @ mat @ right

    # Diagonal blocks (all Fc) ...
    fock = proj(pc, fc, pc) + proj(po, fc, po) + proj(pv, fc, pv)
    # ... closed-open block = Fb (both orderings -> symmetric) ...
    fock = fock + proj(pc, fockb, po) + proj(po, fockb, pc)
    # ... open-virtual block = Fa ...
    fock = fock + proj(po, focka, pv) + proj(pv, focka, po)
    # ... closed-virtual block = Fc.
    fock = fock + proj(pc, fc, pv) + proj(pv, fc, pc)
    # Symmetrise away any residual asymmetry from finite-precision
    # projector products.
    return 0.5 * (fock + fock.conj().T)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


def guest_saunders_canonicalize(
    mo_coeffs: np.ndarray,
    focka: np.ndarray,
    fockb: np.ndarray,
    n_alpha: int,
    n_beta: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Guest-Saunders canonical ROHF eigenvalues and MO coefficients.

    ROHF has no unique Fock matrix: the occupied-occupied,
    virtual-virtual, and occupied-virtual blocks must be constructed
    from different combinations of Coulomb and exchange.  Every quantum
    chemistry code picks a different "effective" Fock to diagonalise
    for orbital eigenvalues.  This function implements the
    **Guest-Saunders** convention (Mol. Phys. 28, 819, 1974), which is
    ORCA's default and the most common choice for production ROHF.

    Algorithm
    ---------
    In the converged MO basis (where the Roothaan effective Fock is
    diagonal), the per-spin Fock matrices are block-partitioned into
    closed (doubly occupied), open (singly occupied), and virtual
    subspaces.  Within each subspace we diagonalise the appropriate
    Fock sub-block:

    * **Closed** [0, n_closed):  Fc = (Fa + Fb)/2
    * **Open** [n_closed, n_alpha): Fa (alpha Fock — gives Koopmans IP)
    * **Virtual** [n_alpha, :):  Fc = (Fa + Fb)/2

    The resulting eigenvalues are sorted ascending within each subspace.
    The returned vector remains closed/open/virtual-block ordered; it is not
    globally re-sorted because doing so independently of the coefficients
    and occupations would break their positional correspondence.  Because
    the transformation is block-diagonal (each subspace rotated
    independently), the density matrix is invariant and the total energy is
    unchanged.  Only the orbital energies and individual MO shapes differ
    from the Roothaan convention.  This routine implements the integer-shell
    convention only; fractional ensemble shells retain their Roothaan
    orbitals in :func:`run_roothaan_scf`.

    Parameters
    ----------
    mo_coeffs
        Converged ROHF MO coefficients (eigenvectors of the Roothaan
        effective Fock), shape ``(nbf, nbf)``.
    focka, fockb
        Per-spin AO Fock matrices at convergence.
    n_alpha, n_beta
        Number of alpha / beta electrons.  ``n_closed = n_beta``,
        ``n_open = n_alpha - n_beta``.

    Returns
    -------
    eps_canonical : ndarray of shape ``(nbf,)``
        Guest-Saunders eigenvalues: closed (ascending), then open
        (ascending), then virtual (ascending).  Within each subspace
        the ordering is the natural ascending sort from ``eigh``.
    C_canonical : ndarray of shape ``(nbf, nbf)``
        Canonical MO coefficients.  Related to ``mo_coeffs`` by a
        block-diagonal unitary rotation: ``C_canonical = mo_coeffs @ U``
        where ``U`` is the direct sum of the subspace eigenvectors.

    References
    ----------
    * M. F. Guest and V. R. Saunders, Mol. Phys. 28, 819 (1974).
    * Roothaan, Rev. Mod. Phys. 32, 179 (1960) — the SCF operator.
    * ORCA manual, Sec. "Restricted Open-Shell Hartree-Fock".
    """
    mo_coeffs = np.asarray(mo_coeffs, dtype=float)
    focka = np.asarray(focka, dtype=float)
    fockb = np.asarray(fockb, dtype=float)

    nbf = mo_coeffs.shape[0]
    n_closed = n_beta
    n_open = n_alpha - n_beta
    n_virtual = nbf - n_alpha

    # Per-spin Fock in the converged MO basis.
    Fa_mo = mo_coeffs.T @ focka @ mo_coeffs
    Fb_mo = mo_coeffs.T @ fockb @ mo_coeffs
    Fc_mo = 0.5 * (Fa_mo + Fb_mo)

    eps = np.zeros(nbf)
    U = np.eye(nbf)

    # -- Closed subspace: diagonalise Fc (doubly occupied, symmetric) --
    if n_closed > 0:
        cc = slice(0, n_closed)
        eps_c, U_c = np.linalg.eigh(Fc_mo[cc, cc])
        eps[cc] = eps_c
        U[np.ix_(range(n_closed), range(n_closed))] = U_c

    # -- Open subspace: diagonalise Fa (alpha Fock => Koopmans IPs) --
    if n_open > 0:
        oo = slice(n_closed, n_alpha)
        eps_o, U_o = np.linalg.eigh(Fa_mo[oo, oo])
        eps[oo] = eps_o
        U[np.ix_(range(n_closed, n_alpha), range(n_closed, n_alpha))] = U_o

    # -- Virtual subspace: diagonalise Fc --
    if n_virtual > 0:
        vv = slice(n_alpha, nbf)
        eps_v, U_v = np.linalg.eigh(Fc_mo[vv, vv])
        eps[vv] = eps_v
        U[np.ix_(range(n_alpha, nbf), range(n_alpha, nbf))] = U_v

    C_canonical = mo_coeffs @ U
    return eps, C_canonical


def _orthonormaliser(s: np.ndarray, linear_dep_threshold: float) -> np.ndarray:
    """Canonical orthogonaliser ``X`` with ``X^T S X = I``.

    Eigenvalues of ``S`` below ``linear_dep_threshold`` are dropped
    (canonical orthogonalisation), so near-linearly-dependent bases lose
    the offending combinations rather than producing a singular
    ``S^{-1/2}``.  Columns of the returned ``X`` span the kept space;
    ``X`` is ``(nao, n_kept)``.
    """
    s = np.asarray(s, dtype=float)
    w, v = np.linalg.eigh(s)
    keep = w > linear_dep_threshold
    return v[:, keep] / np.sqrt(w[keep])


def _spin_partition(n_electrons: int, multiplicity: int) -> Tuple[int, int]:
    """``(n_alpha, n_beta)`` for a given electron count + multiplicity.

    Matches the molecular runner convention
    ``n_alpha = (N + 2S)/2 = (N + mult - 1)/2``.
    """
    two_s = multiplicity - 1
    if (n_electrons - two_s) % 2 != 0 or two_s < 0 or two_s > n_electrons:
        raise ValueError(
            f"multiplicity={multiplicity} is incompatible with "
            f"{n_electrons} electrons (need N - (mult-1) even and "
            f"0 <= mult-1 <= N)."
        )
    n_alpha = (n_electrons + two_s) // 2
    n_beta = n_electrons - n_alpha
    return n_alpha, n_beta


def _aufbau_densities(
    c: np.ndarray, n_alpha: int, n_beta: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Alpha / beta AO densities from occupied columns of ``c`` (aufbau:
    lowest ``n_beta`` doubly occupied, next ``n_alpha - n_beta`` singly)."""
    ca = c[:, :n_alpha]
    cb = c[:, :n_beta]
    dma = ca @ ca.conj().T
    dmb = cb @ cb.conj().T
    return dma, dmb


def _orbital_expectations(c: np.ndarray, fock: np.ndarray) -> np.ndarray:
    """Diagonal expectation values ``<phi_i|F|phi_i>`` for MO columns."""
    return np.einsum("pi,pq,qi->i", c.conj(), fock, c).real


def _roothaan_occupations(
    mo_energies: np.ndarray,
    mo_energy_alpha: np.ndarray,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    """ROHF/ROKS occupation assignment.

    Roothaan's effective Fock orders the closed-space orbitals reliably, but
    open-shell orbitals should be selected by their alpha orbital energy among
    the non-core candidates.  This mirrors the standard ROHF occupation rule
    and avoids converging open-shell KS/HF cases to the wrong determinant.
    """
    if n_alpha < n_beta:
        raise ValueError("ROHF/ROKS expects n_alpha >= n_beta")
    n_mo = int(np.asarray(mo_energies).size)
    n_open = n_alpha - n_beta
    if n_alpha > n_mo:
        raise RuntimeError(
            f"Failed to assign ROHF occupations: n_alpha={n_alpha} > n_mo={n_mo}"
        )

    occ = np.zeros(n_mo, dtype=float)
    order = np.argsort(mo_energies)
    core_idx = order[:n_beta]
    occ[core_idx] = 2.0
    if n_open:
        candidates = order[n_beta:]
        open_order = np.argsort(np.asarray(mo_energy_alpha)[candidates])
        open_idx = candidates[open_order[:n_open]]
        occ[open_idx] = 1.0
    return occ


def _densities_from_occupations(
    c: np.ndarray, occ: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Alpha/beta AO densities from a single restricted-open MO set."""
    mo_occa = (np.asarray(occ) > 0.0).astype(float)
    mo_occb = (np.asarray(occ) == 2.0).astype(float)
    dma = (c * mo_occa) @ c.conj().T
    dmb = (c * mo_occb) @ c.conj().T
    return dma, dmb


def _occupation_model_from_degeneracy(
    mo_energies: np.ndarray,
    n_alpha: int,
    n_beta: int,
    threshold: float = 1.0e-5,
) -> Optional[_OpenShellModel]:
    """Detect a degenerate frontier shell crossing the RO open boundary.

    The usual ROHF/ROKS occupation is integer: the lowest ``n_beta``
    orbitals are closed and the next ``n_alpha - n_beta`` are open.  Linear
    radicals such as OH(2Pi), however, have a degenerate frontier shell
    spanning the closed/open boundary: three electrons occupy two pi
    orbitals, so the spin-restricted state must use occupations 1.5/1.5.

    This helper inspects the first Roothaan diagonalisation only.  If
    adjacent frontier eigenvalues are degenerate within ``threshold``, it
    returns a locked open-shell model; otherwise the caller stays on the
    standard integer occupation path.  The same expansion also covers the
    complementary electron-type case where the open/virtual boundary is
    degenerate.
    """
    n_open = n_alpha - n_beta
    if n_open <= 0:
        return None
    energies = np.asarray(mo_energies, dtype=float)
    order = np.argsort(energies)
    start = n_beta
    end = n_beta + n_open
    if end > order.size:
        return None

    while start > 0:
        gap = abs(float(energies[order[start]] - energies[order[start - 1]]))
        if gap > threshold:
            break
        start -= 1
    while end < order.size:
        gap = abs(float(energies[order[end]] - energies[order[end - 1]]))
        if gap > threshold:
            break
        end += 1

    if start == n_beta and end == n_beta + n_open:
        return None

    n_open_electrons = 0
    for pos in range(start, end):
        if pos < n_beta:
            n_open_electrons += 2
        elif pos < n_alpha:
            n_open_electrons += 1
    return _OpenShellModel(
        n_closed=start,
        n_open_orbitals=end - start,
        n_open_electrons=n_open_electrons,
    )


def _occupations_from_model(
    mo_energies: np.ndarray,
    model: _OpenShellModel,
) -> np.ndarray:
    """Total spatial-orbital occupations for a locked RO shell model."""
    energies = np.asarray(mo_energies, dtype=float)
    order = np.argsort(energies)
    occ = np.zeros(energies.size, dtype=float)
    occ[order[: model.n_closed]] = 2.0
    if model.n_open_orbitals:
        begin = model.n_closed
        end = begin + model.n_open_orbitals
        occ[order[begin:end]] = float(model.n_open_electrons) / float(
            model.n_open_orbitals
        )
    return occ


def _density_partitions_from_occupations(
    c: np.ndarray,
    occ: np.ndarray,
    n_alpha: int,
    n_beta: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(D_closed, D_open, D_alpha, D_beta)`` for RO occupations.

    Fractional frontier shells distribute the spin excess uniformly over
    the degenerate open space.  For OH(2Pi), for example, two pi orbitals
    carry total occupations 1.5/1.5 with alpha/beta occupations 1.0/0.5.
    """
    occ = np.asarray(occ, dtype=float)
    closed_mask = occ >= (2.0 - 1.0e-12)
    open_mask = (occ > 1.0e-12) & ~closed_mask

    d_closed = (c * closed_mask.astype(float)) @ c.conj().T
    d_open = (c * open_mask.astype(float)) @ c.conj().T

    occ_alpha = np.zeros_like(occ)
    occ_beta = np.zeros_like(occ)
    occ_alpha[closed_mask] = 1.0
    occ_beta[closed_mask] = 1.0
    n_open_orbitals = int(np.count_nonzero(open_mask))
    if n_open_orbitals:
        spin_per_open = float(n_alpha - n_beta) / float(n_open_orbitals)
        occ_alpha[open_mask] = 0.5 * (occ[open_mask] + spin_per_open)
        occ_beta[open_mask] = 0.5 * (occ[open_mask] - spin_per_open)

    dma = (c * occ_alpha) @ c.conj().T
    dmb = (c * occ_beta) @ c.conj().T
    return d_closed, d_open, dma, dmb


def _roothaan_effective_fock_from_partitions(
    focka: np.ndarray,
    fockb: np.ndarray,
    d_closed: np.ndarray,
    d_open: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """Roothaan effective Fock from explicit closed/open projectors."""
    focka = np.asarray(focka)
    fockb = np.asarray(fockb)
    d_closed = np.asarray(d_closed)
    d_open = np.asarray(d_open)
    s = np.asarray(s)

    fc = (focka + fockb) * 0.5
    pc = d_closed @ s
    po = d_open @ s
    pv = np.eye(s.shape[0]) - (d_closed + d_open) @ s

    def proj(left: np.ndarray, mat: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left.conj().T @ mat @ right

    fock = proj(pc, fc, pc) + proj(po, fc, po) + proj(pv, fc, pv)
    fock = fock + proj(pc, fockb, po) + proj(po, fockb, pc)
    fock = fock + proj(po, focka, pv) + proj(pv, focka, po)
    fock = fock + proj(pc, fc, pv) + proj(pv, fc, pc)
    return 0.5 * (fock + fock.conj().T)


def _diis_extrapolate(
    fock_history: List[np.ndarray],
    error_history: List[np.ndarray],
) -> np.ndarray:
    """Pulay DIIS extrapolation of the effective Fock from stored
    (Fock, error) pairs.

    The real Hermitian inner product defines the Pulay metric so complex
    Bloch-orbital residuals are handled without discarding phase information.
    Ref. Pulay, J. Comput. Chem. 3, 556 (1982).
    """
    n = len(fock_history)
    b = np.zeros((n + 1, n + 1))
    b[-1, :] = b[:, -1] = -1.0
    b[-1, -1] = 0.0
    for i in range(n):
        for j in range(i, n):
            val = float(
                np.vdot(error_history[i].ravel(), error_history[j].ravel()).real
            )
            b[i, j] = b[j, i] = val
    rhs = np.zeros(n + 1)
    rhs[-1] = -1.0
    try:
        coeffs = np.linalg.solve(b, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(b, rhs, rcond=None)[0]
    fock = np.zeros_like(fock_history[0])
    for i in range(n):
        fock = fock + coeffs[i] * fock_history[i]
    return fock


def _commutator_error(
    fock_eff: np.ndarray,
    dm_total: np.ndarray,
    s: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Orthonormalised DIIS/convergence error ``X^T (F D S - S D F) X``.

    The commutator of the effective Fock with the *total* density
    vanishes at convergence (the occupied-virtual block of ``F_eff`` is
    zero), so its norm is the SCF gradient norm.
    """
    fds = fock_eff @ dm_total @ s
    err = fds - fds.conj().T
    return x.conj().T @ err @ x


def _ediis_diis_switch_metric(grad_norm: float, n_bf: int) -> float:
    denom = float(max(1, n_bf))
    return grad_norm / denom


def _extrapolated_density_pair(accel) -> Tuple[np.ndarray, np.ndarray]:
    """The (alpha, beta) density whose Fock pair ``accel`` just returned.

    The C++ EDIIS/ADIIS objects return ``sum_i c_i F_i``, which equals
    ``F(sum_i c_i D_i)`` exactly at the HF level (J and K are linear in
    the density), so the Roothaan coupling must project with
    ``D_tilde = sum_i c_i D_i`` -- the state whose Fock the extrapolation
    returned -- not with the current density (IID #119).

    ``D_tilde`` is combined on the C++ side (``last_extrapolated_density``)
    over the history *as the accelerator holds it*.  That matters because
    ``EDIIS::finish_extrapolation`` / ``ADIIS::finish_extrapolation`` may
    erase an interior history entry (the anti-replay guard) and re-solve,
    after which the coefficient vector no longer indexes the sequence of
    pushes.  A Python-side mirror of the density history cannot see that
    erase: one guard fire left the mirror one entry too long, the
    reconstruction was abandoned, and the loop silently fell back to
    projecting the extrapolated Fock with the *current* density -- an
    inconsistent effective Fock that can jump the SCF into another basin
    (GitLab #487).  There is no fallback any more: a block count other
    than two is a programming error and fails loudly.
    """
    blocks = accel.last_extrapolated_density()
    if len(blocks) != 2:
        raise RuntimeError(
            "run_roothaan_scf: the SCF accelerator returned "
            f"{len(blocks)} density block(s) for the extrapolated Fock "
            "pair; the Roothaan coupling needs exactly the (alpha, beta) "
            "pair the extrapolate_uhf call stored. Refusing to couple the "
            "extrapolated Fock with a density it does not belong to "
            "(GitLab #487)."
        )
    return (
        np.asarray(blocks[0], dtype=float),
        np.asarray(blocks[1], dtype=float),
    )


def _update_dynamic_damping(
    alpha: float,
    e_new: float,
    e_prev: float,
    have_prev: bool,
    alpha_min: float = 0.0,
    alpha_max: float = 0.95,
    step: float = 0.1,
    decrease_threshold: float = 1.0e-4,
) -> float:
    """Adaptive density-mixing alpha (Zerner-Hehenberger 1979).

    Python mirror of ``vibeqc::update_dynamic_damping``
    (``cpp/include/vibeqc/dynamic_damping.hpp``), which is header-only and
    not bound; the Roothaan loop is the only Python-side SCF driver that
    needs it.  The heuristic and every constant are taken from that header
    so ROHF/ROKS damp identically to the C++ RHF/UHF/RKS/UKS drivers:

    * energy went up -> damp harder (``alpha += step``, toward ``alpha_max``);
    * substantial decrease (``dE < -decrease_threshold``) -> ease off
      (``alpha -= step/2``, toward ``alpha_min``);
    * small fluctuation -> leave alpha alone, so a near-stationary tail
      does not pump alpha up and down on numerical noise.

    Ref. M. C. Zerner, M. Hehenberger, Chem. Phys. Lett. 62, 550 (1979).
    Cross-checked against the C++ helper by
    ``tests/test_rohf_convergence_controls.py``.
    """
    if not have_prev:
        return min(max(alpha, alpha_min), alpha_max)
    d_e = e_new - e_prev
    if d_e > 0.0:
        alpha = min(alpha_max, alpha + step)
    elif d_e < -decrease_threshold:
        alpha = max(alpha_min, alpha - 0.5 * step)
    return min(max(alpha, alpha_min), alpha_max)


_ACCELERATORS = ("diis", "ediis", "ediis_diis", "adiis", "kdiis", "r_cdiis", "ad_cdiis")


def _resolve_scf_accelerator(options):
    val = getattr(options, "scf_accelerator", "ediis_diis")
    key = str(val).strip().lower().replace("+", "_")
    if key not in _ACCELERATORS:
        raise ValueError(
            f"Unknown scf_accelerator={val!r}; expected one of "
            f"{', '.join(_ACCELERATORS)}"
        )
    return key


def _resolve_diis_depth_policy(accelerator: str, options):
    """``(DIISDepthPolicy, param)`` for the DIIS history depth.

    Python mirror of ``vibeqc::make_diis`` (``cpp/include/vibeqc/ediis.hpp``):
    the plain accelerators keep Pulay's fixed-depth history, while the two
    Chupin et al. 2021 variants shrink or restart it.

    ``RESTART`` (``r_cdiis``) drops the history when the Pulay system goes
    near-linearly-dependent; ``ADAPTIVE`` (``ad_cdiis``) shrinks the depth
    near convergence via a residual-ratio test. Both target the failure
    this loop had no answer to: once the subspace fills with near-identical
    iterates from a long plateau, fixed-depth DIIS stops making progress
    and the SCF crawls (BUG 118).

    Ref. Chupin, Dupuy, Legendre & Sere, ESAIM: M2AN 55, 2785 (2021).
    """
    from vibeqc._vibeqc_core import DIISDepthPolicy

    if accelerator == "r_cdiis":
        return DIISDepthPolicy.RESTART, float(
            getattr(options, "diis_restart_tau", 1.0e-4)
        )
    if accelerator == "ad_cdiis":
        return DIISDepthPolicy.ADAPTIVE, float(
            getattr(options, "diis_adaptive_delta", 1.0e-4)
        )
    return DIISDepthPolicy.FIXED, 1.0e-4


def _accel_subspace_size(accel):
    ss = getattr(accel, "subspace_size", None)
    if callable(ss):
        return int(ss())
    return int(ss) if ss is not None else 0


# ---------------------------------------------------------------------------
# Generic Roothaan SCF loop (shared by ROHF here and ROKS in roks.py)
# ---------------------------------------------------------------------------

# fock_builder(dma, dmb) -> (focka, fockb, e_electronic_without_nuclear)
FockBuilder = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]


def run_roothaan_scf(
    s: np.ndarray,
    hcore: np.ndarray,
    e_nuc: float,
    n_alpha: int,
    n_beta: int,
    fock_builder: FockBuilder,
    options: ROHFOptions,
    init_alpha: np.ndarray,
    init_beta: np.ndarray,
    *,
    iteration_callback: Optional[Callable[[_ScfStep], None]] = None,
) -> ROHFResult:
    """Drive the Roothaan single-matrix SCF given a per-spin Fock builder.

    ``fock_builder(dma, dmb)`` returns ``(focka, fockb, e_elec)`` --- the
    per-spin AO Fock matrices and the electronic energy (excluding
    nuclear repulsion) consistent with the input densities.  This is the
    single seam where ROHF (exact exchange) and ROKS (XC + scaled
    exchange) differ; everything else --- the coupling, DIIS, aufbau
    occupation and convergence test --- is shared.  ``iteration_callback``
    is invoked once per completed SCF iteration with the same trace record
    stored on the result.  It lets periodic callers forward progress and
    divergence checks without coupling this numerical kernel to an output
    implementation.
    """
    require_nonnegative_max_iter(options.max_iter, route="run_roothaan_scf")
    s = np.asarray(s, dtype=float)
    hcore = np.asarray(hcore, dtype=float)
    x = _orthonormaliser(s, options.linear_dep_threshold)

    use_dav = options.use_davidson and s.shape[0] >= options.davidson_min_dim
    dav_opts = options.davidson
    if use_dav and dav_opts is None:
        from vibeqc._vibeqc_core import DavidsonOptions

        dav_opts = DavidsonOptions()

    dma = np.asarray(init_alpha, dtype=float).copy()
    dmb = np.asarray(init_beta, dtype=float).copy()

    # -- Saunders-Hillier level-shift policy --
    # Resolved per iteration through the shared C++ helper so ROHF/ROKS
    # honour level_shift / level_shift_warmup_cycles / level_shift_schedule
    # exactly as the molecular C++ and periodic Python drivers do; see
    # cpp/include/vibeqc/level_shift.hpp for the semantics.
    from vibeqc._vibeqc_core import level_shift_at_iter

    shift_base = float(getattr(options, "level_shift", 0.0) or 0.0)
    shift_warmup = int(getattr(options, "level_shift_warmup_cycles", -1))
    shift_schedule = [
        float(v) for v in (getattr(options, "level_shift_schedule", None) or [])
    ]
    # True once a diagonalisation has been done under a non-zero shift, so
    # the exit path knows the canonical MOs still have to be recomputed
    # from the unshifted Fock (a shifted diagonalisation leaves every
    # virtual eigenvalue high by exactly the shift).
    shift_contaminates_mos = False

    # -- Density-mixing state --
    # Local, not options.damping: with damping=0.0 + dynamic_damping=True
    # the adaptive update is the only source of damping (mirrors the C++
    # UHF/RHF drivers).
    current_damping = float(options.damping)

    # -- SCF accelerator state --
    accelerator = _resolve_scf_accelerator(options)
    have_cxx_accel = _EDIIS is not None and _DIIS is not None and _ADIIS is not None
    if have_cxx_accel:
        _depth_policy, _depth_param = _resolve_diis_depth_policy(accelerator, options)
        _diis_cxx = _DIIS(options.diis_subspace_size, _depth_policy, _depth_param)
    else:
        _diis_cxx = None
    _ediis = _EDIIS(options.diis_subspace_size) if have_cxx_accel else None
    _adiis = _ADIIS(options.diis_subspace_size) if have_cxx_accel else None
    _kdiis = _KDIIS(options.diis_subspace_size) if _KDIIS is not None else None
    # Fallback NumPy DIIS history when C++ is unavailable
    _py_fock_history: List[np.ndarray] = []
    _py_error_history: List[np.ndarray] = []
    # The Roothaan coupling below needs the *extrapolated* density's
    # projectors -- EDIIS/ADIIS return sum_i c_i F_i, which is
    # F(sum_i c_i D_i) at the HF level.  That density is read back from
    # the C++ accelerator (``last_extrapolated_density``), which combines
    # it over the history it actually holds; a Python-side mirror of the
    # density history was abandoned because it could not see the
    # anti-replay guard's erase() and desynced (GitLab #487).

    trace: List[_ScfStep] = []

    energy_prev = 0.0
    converged = False
    focka = fockb = fock_eff = None
    mo_energies = mo_coeffs = None
    mo_eps_prev = None
    mo_occupations = None
    open_shell_model: Optional[_OpenShellModel] = None
    d_closed = d_open = None
    n_iter = 0

    for it in range(1, options.max_iter + 1):
        n_iter = it
        focka, fockb, e_elec = fock_builder(dma, dmb)
        energy = e_elec + e_nuc

        if open_shell_model is not None and d_closed is not None and d_open is not None:
            fock_eff = _roothaan_effective_fock_from_partitions(
                focka, fockb, d_closed, d_open, s
            )
        else:
            fock_eff = roothaan_effective_fock(focka, fockb, dma, dmb, s)

        dm_total = dma + dmb
        err = _commutator_error(fock_eff, dm_total, s, x)
        grad_norm = float(np.linalg.norm(err))

        diis_dim = 0
        if options.use_diis and it >= options.diis_start_iter:
            if accelerator == "diis" or not have_cxx_accel:
                _py_fock_history.append(fock_eff.copy())
                _py_error_history.append(err.copy())
                if len(_py_fock_history) > options.diis_subspace_size:
                    _py_fock_history.pop(0)
                    _py_error_history.pop(0)
                diis_dim = len(_py_fock_history)
                if diis_dim > 1:
                    fock_eff = _diis_extrapolate(_py_fock_history, _py_error_history)
            elif accelerator in ("r_cdiis", "ad_cdiis"):
                # Same Pulay extrapolation as "diis"; the restart /
                # adaptive-depth behaviour lives in the DIIS object's
                # DIISDepthPolicy, set at construction above.
                assert _diis_cxx is not None
                F_diis = np.asarray(_diis_cxx.extrapolate(fock_eff, err))
                diis_dim = _accel_subspace_size(_diis_cxx)
                if diis_dim > 1:
                    fock_eff = F_diis
            elif accelerator == "ediis_diis":
                assert _diis_cxx is not None and _ediis is not None
                F_diis = np.asarray(_diis_cxx.extrapolate(fock_eff, err))
                fa_ediis, fb_ediis = _ediis.extrapolate_uhf(
                    focka, fockb, dma, dmb, energy
                )
                diis_dim = _accel_subspace_size(_diis_cxx)
                switch = _ediis_diis_switch_metric(grad_norm, fock_eff.shape[0])
                # The EDIIS pair reaches the SCF only when the switch
                # metric selects it AND the history is deep enough to
                # extrapolate; every other path discards it, which the
                # anti-replay guard must be told about (see
                # EDIIS::finish_extrapolation in cpp/src/ediis.cpp).
                if (
                    switch > options.ediis_diis_switch_threshold
                    and _accel_subspace_size(_diis_cxx) > 1
                ):
                    # Couple the EDIIS Focks with the projectors of the
                    # hull-optimal density D_tilde = sum_i c_i D_i, not
                    # the current density.  The EDIIS return is
                    # F(D_tilde); projecting it with a different density's
                    # subspaces makes the Roothaan effective Fock
                    # inconsistent and can jump the SCF into the wrong
                    # stationary point (IID #119).  D_tilde comes from
                    # the accelerator itself (GitLab #487).
                    dma_e, dmb_e = _extrapolated_density_pair(_ediis)
                    fock_eff = roothaan_effective_fock(
                        np.asarray(fa_ediis), np.asarray(fb_ediis),
                        dma_e, dmb_e, s,
                    )
                else:
                    _ediis.discard_last_extrapolation()
                    if _accel_subspace_size(_diis_cxx) > 1:
                        fock_eff = F_diis
            elif accelerator == "ediis":
                assert _ediis is not None
                fa_ediis, fb_ediis = _ediis.extrapolate_uhf(
                    focka, fockb, dma, dmb, energy
                )
                diis_dim = _accel_subspace_size(_ediis)
                if _accel_subspace_size(_ediis) > 1:
                    # Same hull-optimal projector coupling as the hybrid
                    # branch (IID #119, #487).
                    dma_e, dmb_e = _extrapolated_density_pair(_ediis)
                    fock_eff = roothaan_effective_fock(
                        np.asarray(fa_ediis), np.asarray(fb_ediis),
                        dma_e, dmb_e, s,
                    )
                else:
                    _ediis.discard_last_extrapolation()
            elif accelerator == "adiis":
                assert _adiis is not None
                fa_ad, fb_ad = _adiis.extrapolate_uhf(focka, fockb, dma, dmb)
                diis_dim = _accel_subspace_size(_adiis)
                if _accel_subspace_size(_adiis) > 1:
                    # Same hull-optimal projector coupling as EDIIS
                    # (IID #119, #487); ADIIS shares the QP +
                    # returned-Fock contract (Hu & Yang 2010).
                    dma_e, dmb_e = _extrapolated_density_pair(_adiis)
                    fock_eff = roothaan_effective_fock(
                        np.asarray(fa_ad), np.asarray(fb_ad),
                        dma_e, dmb_e, s,
                    )
                else:
                    _adiis.discard_last_extrapolation()
            elif accelerator == "kdiis":
                if _kdiis is not None and mo_eps_prev is not None:
                    try:
                        fock_eff = np.asarray(
                            _kdiis.extrapolate(
                                fock_eff, mo_coeffs, mo_eps_prev, n_alpha
                            )
                        )
                        diis_dim = _accel_subspace_size(_kdiis)
                    except Exception:
                        pass

        delta_e = energy - energy_prev if it > 1 else 0.0
        step = _ScfStep(
            iter=it,
            energy=energy,
            delta_e=delta_e,
            grad_norm=grad_norm,
            diis_subspace=diis_dim,
        )
        trace.append(step)
        if iteration_callback is not None:
            iteration_callback(step)

        if (
            it > 1
            and abs(delta_e) < options.conv_tol_energy
            and (grad_norm < options.conv_tol_grad)
        ):
            converged = True
            # Diagonalise the converged Fock once more for canonical MOs.
            # Unshifted, always: the Saunders-Hillier operator is inert on
            # the occupied block but raises every virtual eigenvalue by the
            # shift, so canonicalising through it would report orbital
            # energies (and a gap) that are wrong by exactly that amount.
            mo_energies, mo_coeffs = _solve_fock(
                fock_eff,
                x,
                s,
                dma,
                0.0,
                use_dav=use_dav,
                dav_opts=dav_opts,
            )
            shift_contaminates_mos = False
            if open_shell_model is None and options.fractional_open_shell:
                open_shell_model = _occupation_model_from_degeneracy(
                    mo_energies, n_alpha, n_beta
                )
            if open_shell_model is not None:
                mo_occupations = _occupations_from_model(mo_energies, open_shell_model)
            else:
                mo_occupations = _roothaan_occupations(
                    mo_energies,
                    _orbital_expectations(mo_coeffs, focka),
                    n_alpha,
                    n_beta,
                )
            break

        # Optional virtual level shift (diagonalisation only).
        shift_now = float(
            level_shift_at_iter(
                shift_base,
                shift_warmup,
                shift_schedule,
                int(options.max_iter),
                it,
            )
        )
        mo_energies, mo_coeffs = _solve_fock(
            fock_eff,
            x,
            s,
            dma,
            shift_now,
            use_dav=use_dav,
            dav_opts=dav_opts,
        )
        shift_contaminates_mos = shift_now > 0.0
        mo_eps_prev = mo_energies
        if open_shell_model is None and options.fractional_open_shell:
            open_shell_model = _occupation_model_from_degeneracy(
                mo_energies, n_alpha, n_beta
            )
        if open_shell_model is not None:
            mo_occupations = _occupations_from_model(mo_energies, open_shell_model)
            d_closed, d_open, dma_new, dmb_new = _density_partitions_from_occupations(
                mo_coeffs, mo_occupations, n_alpha, n_beta
            )
        else:
            mo_occupations = _roothaan_occupations(
                mo_energies,
                _orbital_expectations(mo_coeffs, focka),
                n_alpha,
                n_beta,
            )
            dma_new, dmb_new = _densities_from_occupations(mo_coeffs, mo_occupations)
            d_closed = d_open = None

        # Density damping gate: damp only while the DIIS history has at
        # most one entry, i.e. before Pulay extrapolation can take over
        # the update.  The first DIIS iteration has nothing to
        # extrapolate from, so a damped first step smooths the startup
        # transient; once diis_dim > 1 the extrapolation owns the
        # update and damping is skipped.
        if current_damping > 0.0 and not (
            options.use_diis and it >= options.diis_start_iter and diis_dim > 1
        ):
            d = current_damping
            dma_new = (1.0 - d) * dma_new + d * dma
            dmb_new = (1.0 - d) * dmb_new + d * dmb

        dma, dmb = dma_new, dmb_new

        # Adaptive mixing for the *next* iteration, ordered as in the C++
        # UHF driver: the alpha used at iteration N reflects the energy
        # change measured through iteration N-1.
        if options.dynamic_damping:
            current_damping = _update_dynamic_damping(
                current_damping,
                energy,
                energy_prev,
                it > 1,
                options.dynamic_damping_min,
                options.dynamic_damping_max,
            )

        energy_prev = energy

    # Final energy / Fock consistent with the converged density.
    focka, fockb, e_elec = fock_builder(dma, dmb)
    energy = e_elec + e_nuc
    if open_shell_model is not None and d_closed is not None and d_open is not None:
        fock_eff = _roothaan_effective_fock_from_partitions(
            focka, fockb, d_closed, d_open, s
        )
    else:
        fock_eff = roothaan_effective_fock(focka, fockb, dma, dmb, s)
    if mo_energies is None or shift_contaminates_mos:
        # ``shift_contaminates_mos`` covers the max_iter-exhausted exit: the
        # last in-loop diagonalisation ran under a live shift, so the MOs it
        # produced carry the shift on every virtual eigenvalue. Redo it
        # unshifted -- an unconverged orbital table is diagnostic output and
        # has to show the orbital energies of the Fock actually reached.
        mo_energies, mo_coeffs = _solve_fock(
            fock_eff,
            x,
            s,
            dma,
            0.0,
            use_dav=use_dav,
            dav_opts=dav_opts,
        )
    if open_shell_model is None and options.fractional_open_shell:
        open_shell_model = _occupation_model_from_degeneracy(
            mo_energies, n_alpha, n_beta
        )
    if open_shell_model is not None:
        mo_occupations = _occupations_from_model(mo_energies, open_shell_model)
    else:
        mo_occupations = _roothaan_occupations(
            mo_energies,
            _orbital_expectations(mo_coeffs, focka),
            n_alpha,
            n_beta,
        )

    spin = 0.5 * (n_alpha - n_beta)
    s_squared = spin * (spin + 1.0)

    # Guest-Saunders is an integer closed/open/virtual convention.  Applying
    # those n_beta/n_alpha boundaries through a detected fractional frontier
    # ensemble would put symmetry partners in different subspaces and split
    # them.  Report the already sorted, symmetry-preserving Roothaan spectrum
    # for that case; ordinary integer ROHF/ROKS remains Guest-Saunders.
    mo_energies_roothaan = np.asarray(mo_energies).copy()
    mo_coeffs_roothaan = np.asarray(mo_coeffs).copy()
    fractional_shell = open_shell_model is not None and open_shell_model.is_fractional
    if not fractional_shell:
        reported_energies, reported_coeffs = guest_saunders_canonicalize(
            mo_coeffs,
            focka,
            fockb,
            n_alpha,
            n_beta,
        )
        canonicalization = "guest-saunders"
    else:
        reported_energies = mo_energies_roothaan.copy()
        reported_coeffs = mo_coeffs_roothaan.copy()
        canonicalization = "roothaan-fractional-shell"

    return ROHFResult(
        energy=float(energy),
        e_electronic=float(e_elec),
        n_iter=n_iter,
        converged=converged,
        mo_energies=reported_energies,
        mo_coeffs=reported_coeffs,
        mo_occupations=mo_occupations,
        density=dma + dmb,
        density_alpha=dma,
        density_beta=dmb,
        fock=fock_eff,
        fock_alpha=focka,
        fock_beta=fockb,
        s_squared=s_squared,
        s_squared_ideal=s_squared,
        n_alpha=n_alpha,
        n_beta=n_beta,
        scf_trace=trace,
        rohf_canonicalization=canonicalization,
        mo_energies_roothaan=mo_energies_roothaan,
        mo_coeffs_roothaan=mo_coeffs_roothaan,
    )


def _solve_fock(
    fock_eff: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
    dma: np.ndarray,
    level_shift: float,
    use_dav: bool = False,
    dav_opts=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalise the effective Fock in the orthonormal basis.

    With ``level_shift > 0`` the virtual block is raised by
    ``level_shift`` Hartree (operator ``shift * (S - S Da S)`` adds the
    shift to every orbital outside the alpha-occupied space), which damps
    rotations into low-lying virtuals.  The shift affects only the
    diagonalisation, not the energy or DIIS error.

    ``level_shift`` here is the shift *for this call*, not
    ``options.level_shift``: the caller resolves the per-iteration value
    through ``level_shift_at_iter`` and passes ``0.0`` for the
    canonicalising diagonalisation, whose eigenvalues are reported to the
    user (the shift would leave every virtual one high by exactly ``b``).

    When ``use_dav`` is True and ``dav_opts`` is provided, the Davidson
    iterative solver is used for the leading eigenvalues/vectors instead
    of a dense diagonalisation.
    """
    f = fock_eff
    if level_shift > 0.0:
        f = f + level_shift * (s - s @ dma @ s)
    f_orth = x.conj().T @ f @ x
    f_orth = 0.5 * (f_orth + f_orth.T)
    if use_dav and dav_opts is not None:
        from vibeqc._vibeqc_core import davidson_solve

        if dav_opts.n_eig == 0:
            dav_opts.n_eig = f_orth.shape[0]
        if dav_opts.guess_vectors is not None:
            pass  # already set from previous iteration
        dres = davidson_solve(f_orth, dav_opts)
        if not dres.converged:
            raise RuntimeError(f"Davidson did not converge after {dres.n_iter} iters")
        eps, c_orth = dres.eigenvalues, dres.eigenvectors
        dav_opts.guess_vectors = c_orth
    else:
        eps, c_orth = np.linalg.eigh(f_orth)
    c = x @ c_orth
    return eps, c


# ---------------------------------------------------------------------------
# ROHF Fock builder + public driver
# ---------------------------------------------------------------------------


def _make_hf_fock_builder(
    hcore: np.ndarray,
    build_j: Callable[[np.ndarray], np.ndarray],
    build_k: Callable[[np.ndarray], np.ndarray],
) -> FockBuilder:
    """Exact-exchange (Hartree--Fock) per-spin Fock builder.

    ``Fa = H + J(Da+Db) - K(Da)``, ``Fb = H + J(Da+Db) - K(Db)`` and the
    UHF/ROHF electronic energy
    ``E = Tr[Dt H] + 1/2 Tr[Da(J-Ka)] + 1/2 Tr[Db(J-Kb)]``.
    """
    hcore = np.asarray(hcore, dtype=float)

    def builder(
        dma: np.ndarray, dmb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        dm_total = dma + dmb
        j = np.asarray(build_j(dm_total))
        ka = np.asarray(build_k(dma))
        kb = np.asarray(build_k(dmb))
        focka = hcore + j - ka
        fockb = hcore + j - kb
        e_elec = float(
            np.einsum("ij,ij->", dm_total, hcore)
            + 0.5 * np.einsum("ij,ij->", dma, j - ka)
            + 0.5 * np.einsum("ij,ij->", dmb, j - kb)
        )
        return focka, fockb, e_elec

    return builder


def _resolve_jk_builder(molecule: Any, basis: Any, options: ROHFOptions):
    """Return ``(build_J, build_K)`` closures for the chosen path.

    Mirrors :func:`vibeqc.parity._make_jk_builders`: RI density fitting
    when ``options.density_fit`` (needs ``aux_basis``), else the
    Schwarz-screened direct libint builder.
    """
    from . import _vibeqc_core as _core
    from .density_fitting import DensityFitting

    if options.density_fit:
        if not options.aux_basis:
            raise ValueError("density_fit=True requires options.aux_basis")
        aux_basis = type(basis)(molecule, options.aux_basis)
        df = DensityFitting(
            basis,
            aux_basis,
            aux_basis_name=options.aux_basis,
            molecule=molecule,
        )
        if getattr(options, "cosx", False):
            # RIJCOSX: RI-J for Coulomb, seminumerical chain-of-spheres for
            # exchange. The COSX grid honours cosx_grid_level (0 = legacy
            # sparse grid; 1..4 / -1 = 2021 5-region GridX tiers). Single
            # grid (no multi-stage) — the ROHF reference is fed to ROHF-CC /
            # ROHF-MP2; a converged reference within the COSX fit band is the
            # goal. K is symmetric per spin density, so the same kernel
            # serves Ka = K(Da), Kb = K(Db).
            from . import (
                cosx_basis_cardinality_from_name,
                cosx_grid_options_for_level,
                resolve_cosx_grid_level,
            )

            gl = int(getattr(options, "cosx_grid_level", 0))
            if gl == 0:
                # Legacy sparse COSX grid (mirrors C++
                # default_cosx_grid_options(): 35 radial × 9×18 product).
                cosx_opts = _core.GridOptions()
                cosx_opts.n_radial = 35
                cosx_opts.n_theta = 9
                cosx_opts.n_phi = 18
            else:
                card = cosx_basis_cardinality_from_name(basis.name)
                cosx_opts = cosx_grid_options_for_level(
                    resolve_cosx_grid_level(gl, card)
                )
            grid = _core.build_grid(molecule, cosx_opts)
            return (
                lambda d: np.asarray(df.build_J(d)),
                lambda d: np.asarray(_core.compute_cosx_k(basis, d, grid)),
            )
        return (
            lambda d: np.asarray(df.build_J(d)),
            lambda d: np.asarray(df.build_K_density(d)),
        )
    jk = _core.make_direct_jk_builder(basis)
    return (
        lambda d: np.asarray(jk.build_J(d)),
        lambda d: np.asarray(jk.build_K(d)),
    )


def _supported_initial_guesses() -> Tuple[Any, ...]:
    """The molecular Roothaan adapter supplies every engine requirement."""
    from ._vibeqc_core import InitialGuess

    return tuple(InitialGuess.__members__.values())


def _resolve_initial_guess(options: ROHFOptions) -> Any:
    """Canonical selector, validated even when a restart density is supplied."""
    from ._initial_guess import coerce_initial_guess

    return coerce_initial_guess(options.initial_guess)


def _prepare_initial_guess(molecule, options, initial_density):
    from .guess import select_initial_guess

    return select_initial_guess(
        molecule, options.initial_guess, is_open_shell=True,
        atomic_spins=options.atomic_spins,
        restart_supplied=initial_density is not None,
        driver="ROHF/ROKS",
    )


def _initial_densities(
    molecule: Any,
    basis: Any,
    n_alpha: int,
    n_beta: int,
    options: ROHFOptions,
    s: np.ndarray,
    hcore: np.ndarray,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Consume the selected native construction; builder failures propagate."""
    from . import _vibeqc_core as _core

    from .guess import guess_ecp_context, validate_guess_ecp

    selection = _prepare_initial_guess(molecule, options, None)
    ecp_context = validate_guess_ecp(selection.effective, guess_ecp_context(options), molecule)
    kind = selection.effective
    if kind in (_core.InitialGuess.READ, _core.InitialGuess.FRAGMO):
        da, db = options.read_density_alpha, options.read_density_beta
        if da is None or db is None:
            if kind == _core.InitialGuess.FRAGMO:
                raise ValueError("FRAGMO requires prepared fragment spin densities")
            from .guess_read import resolve_read_densities_open
            da, db = resolve_read_densities_open(options, molecule, basis, None)
        return np.asarray(da), np.asarray(db)
    if kind != _core.InitialGuess.HCORE:
        # The same engine used by UHF gets the actual molecular pencil. SAD
        # remains density-mode, preserving its Hund spatial pattern; PATOM
        # alone needs the in-field J/K step (Van Lenthe 2006, Procedure).
        jk = _core.make_direct_jk_builder(basis)
        g = _core._guess_open_shell_density_with_jk(
            molecule, basis, n_alpha, n_beta, kind, s, hcore, jk,
            False, options.atomic_spins, options.linear_dep_threshold, ecp_context,
        )
        if g is None:
            raise RuntimeError(f"{kind.name} builder returned no initial density")
        da, db = g
        return np.asarray(da), np.asarray(db)
    # Core (Hcore) guess: diagonalise Hcore in the orthonormal basis.
    f_orth = x.conj().T @ np.asarray(hcore, dtype=float) @ x
    _eps, c_orth = np.linalg.eigh(f_orth)
    c = x @ c_orth
    return _aufbau_densities(c, n_alpha, n_beta)


def _renormalise_guess_spin_density(
    da: np.ndarray,
    db: np.ndarray,
    n_alpha: int,
    n_beta: int,
    s: np.ndarray,
    tol: float = 1.0e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility adapter for the common population normalization contract."""
    from .guess import normalize_spin_density_guess

    return normalize_spin_density_guess(da, db, s, n_alpha, n_beta)


def run_rohf(
    molecule: Any,
    basis: Any,
    options: Optional[ROHFOptions] = None,
    *,
    initial_density: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    read_from: Optional[object] = None,
    fragments: Optional[object] = None,
) -> ROHFResult:
    """Restricted open-shell Hartree--Fock single point.

    Open-shell counterpart of :func:`vibeqc.run_rhf` that keeps a single
    spin-restricted orbital set (spin-pure determinant).  Drop-in for the
    molecular runner: same ``(molecule, basis, options)`` signature.

    Parameters
    ----------
    molecule
        :class:`vibeqc.Molecule`; ``multiplicity`` sets the open-shell
        count (``n_alpha - n_beta = multiplicity - 1``).
    basis
        :class:`vibeqc.BasisSet` for ``molecule``.
    options
        :class:`ROHFOptions` (defaults applied when ``None``).
    initial_density
        Optional ``(D_alpha, D_beta)`` warm-start densities.
    """
    if options is None:
        options = ROHFOptions()
    require_nonnegative_max_iter(options.max_iter, route="run_rohf")
    _prepare_ecp_options(options, molecule, basis, route="run_rohf")
    if read_from is not None or fragments is not None:
        if initial_density is not None:
            raise ValueError("ROHF: pass either initial_density or a READ/FRAGMO source")
        from .guess import prepare_molecular_guess_source
        prepare_molecular_guess_source(
            "rohf", options, molecule, basis, read_from=read_from, fragments=fragments,
        )
    selection = _prepare_initial_guess(molecule, options, initial_density)
    from .guess import guess_ecp_context, validate_guess_ecp
    validate_guess_ecp(selection.effective, guess_ecp_context(options), molecule)

    from . import _vibeqc_core as _core

    s = np.asarray(_core.compute_overlap(basis), dtype=float)
    hcore, e_nuc, n_valence, _z_eff = one_electron_hamiltonian(
        molecule, basis, options
    )
    hcore = np.asarray(hcore, dtype=float)
    e_nuc = float(e_nuc)

    n_alpha, n_beta = _spin_partition(int(n_valence), molecule.multiplicity)

    build_j, build_k = _resolve_jk_builder(molecule, basis, options)
    fock_builder = _make_hf_fock_builder(hcore, build_j, build_k)

    x = _orthonormaliser(s, options.linear_dep_threshold)
    if initial_density is not None:
        init_a, init_b = (
            np.asarray(initial_density[0], dtype=float),
            np.asarray(initial_density[1], dtype=float),
        )
    else:
        init_a, init_b = _initial_densities(
            molecule, basis, n_alpha, n_beta, options, s, hcore, x
        )

    result = run_roothaan_scf(
        s, hcore, e_nuc, n_alpha, n_beta, fock_builder, options, init_a, init_b
    )
    _stamp_ecp_provenance(result, molecule, options)
    result.guess_selection = selection
    result.restart_basis = basis
    return result


def _prepare_ecp_options(options, molecule, basis, *, route: str) -> None:
    """Attach the basis sidecar (inline) and enforce the BUG 99 guard, exactly
    as the native SCF wrappers do, so ROHF/ROKS solve the same Hamiltonian."""
    from .ecp_metadata import (
        attach_inline_ecp_options_from_basis_sidecar,
        validate_ecp_required,
    )

    attach_inline_ecp_options_from_basis_sidecar(options, molecule, basis)
    basis_name = str(getattr(basis, "name", "") or "").strip()
    if basis_name:
        validate_ecp_required(options, molecule, basis_name)
    has_xml = bool(getattr(options, "ecp_centers", None))
    has_inline = bool(getattr(options, "ecp_primitive_blocks", None))
    if has_xml and has_inline:
        raise ValueError(
            f"{route}: ecp_centers (XML library) and ecp_primitive_blocks "
            "(inline primitives) are mutually exclusive"
        )
    if not has_xml and not has_inline:
        # A core count or effective-charge vector without an operator is
        # orphan provenance: the native drivers refuse it, and so does this
        # one, rather than run bare-Z with a valence electron count.
        stray = int(getattr(options, "ecp_total_ncore", 0) or 0) != 0 or bool(
            getattr(options, "ecp_effective_charges", None)
        ) or bool(getattr(options, "ecp_primitive_centers", None))
        if stray:
            raise ValueError(
                f"{route}: ECP metadata (ecp_total_ncore / "
                "ecp_effective_charges / ecp_primitive_centers) is set but "
                "no ECP operator (ecp_centers or ecp_primitive_blocks) is; "
                "supply the operator or clear the metadata"
            )


def _stamp_ecp_provenance(result, molecule, options) -> None:
    """Record on ``result`` which ECP operator the Hamiltonian carried."""
    from .ecp_metadata import ecp_core_electrons_per_atom

    blocks = list(getattr(options, "ecp_primitive_blocks", None) or [])
    centers = list(getattr(options, "ecp_centers", None) or [])
    if blocks:
        result.ecp_operator_applied = True
        result.ecp_primitive_blocks = tuple(blocks)
        result.ecp_primitive_centers = tuple(
            tuple(float(v) for v in c)
            for c in (getattr(options, "ecp_primitive_centers", None) or [])
        )
        result.ecp_effective_charges = tuple(
            float(q) for q in (getattr(options, "ecp_effective_charges", None) or [])
        )
    elif centers:
        result.ecp_operator_applied = True
        result.ecp_xml_centers = tuple(centers)
        result.ecp_xml_library = str(
            getattr(options, "ecp_library", "") or "ecp10mdf"
        )
    else:
        result.ecp_operator_applied = False
    result.ecp_provenance_verified = True
    result.ecp_total_ncore = int(sum(ecp_core_electrons_per_atom(molecule, options)))


def rohf_energy_weighted_density(
    dma: np.ndarray,
    dmb: np.ndarray,
    focka: np.ndarray,
    fockb: np.ndarray,
) -> np.ndarray:
    """ROHF energy-weighted (Lagrangian) density ``W`` for the Pulay force.

    ``W = Da Fa Da + Db Fb Db`` in the AO basis, where ``Fa``/``Fb`` are the
    per-spin Fock matrices.  This is the term that contracts with ``dS/dR``
    in the nuclear gradient.  Unlike a closed-shell SCF, the ROHF orbitals
    do *not* diagonalise the per-spin Fock matrices (they diagonalise the
    Roothaan effective Fock), so the naive "orbital-energy" form
    ``S_i n_i e_i C_i C_iᵀ`` (with ``e_i`` the effective-Fock eigenvalues)
    is **wrong** -- it drops the within-occupied off-diagonal Fock elements.
    The ``Ds Fs Ds`` form is exact (verified by finite difference of the
    ROHF energy to ~1e-10; it is also the AO form UHF reduces to, since
    there ``Ds Fs Ds = S_i es_i Cs_i Cs_iᵀ``).
    """
    dma = np.asarray(dma, dtype=float)
    dmb = np.asarray(dmb, dtype=float)
    focka = np.asarray(focka, dtype=float)
    fockb = np.asarray(fockb, dtype=float)
    w = dma @ focka @ dma + dmb @ fockb @ dmb
    return 0.5 * (w + w.conj().T)  # symmetrise finite-precision asymmetry


def compute_rohf_gradient(
    molecule: Any,
    basis: Any,
    result: "ROHFResult",
    *,
    gradient_options: Any = None,
) -> np.ndarray:
    """Analytic nuclear gradient ``dE/dR`` (Ha/bohr, shape ``(n_atoms, 3)``)
    of a converged ROHF state.

    Assembled from the same building blocks as the UHF gradient
    (:func:`vibeqc.compute_gradient_uhf`) --- one-electron + two-electron
    (J + full exchange) + nuclear repulsion --- but with the *ROHF*
    energy-weighted density (:func:`rohf_energy_weighted_density`) in the
    overlap (Pulay) term.  The two-electron piece uses
    ``two_electron_gradient_contribution_uhf`` with ``alpha_hf=1.0`` (exact
    exchange), exactly as ROHF/UHF share the same two-electron energy.

    Density-fitted SCF is supported when ``gradient_options`` carries the
    matching ``density_fit=True`` and ``aux_basis`` settings. RIJCOSX and ECP
    gradients remain unsupported and fail closed.

    Manually supplied ECP-derived ROHF results paired with an
    all-electron-named basis are unsupported. This API can reject ECP
    provenance attached to ``basis``, ``gradient_options``, or ``result``,
    but cannot infer the Hamiltonian provenance of otherwise unmarked arrays.
    """
    from .ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        molecule,
        basis,
        options=gradient_options,
        result=result,
        route="compute_rohf_gradient",
    )
    from . import _vibeqc_core as _core

    if getattr(result, "method", "rohf") not in ("rohf",):
        raise ValueError(
            "compute_rohf_gradient expects an ROHF result "
            f"(got method={getattr(result, 'method', None)!r})."
        )

    dma = np.asarray(result.density_alpha, dtype=float)
    dmb = np.asarray(result.density_beta, dtype=float)
    p_total = dma + dmb
    w = rohf_energy_weighted_density(dma, dmb, result.fock_alpha, result.fock_beta)

    use_df = bool(getattr(gradient_options, "density_fit", False))
    if use_df and bool(getattr(gradient_options, "cosx", False)):
        raise NotImplementedError(
            "Analytic ROHF RIJCOSX gradients are not yet supported; use "
            "density fitting without COSX. See handovers/HANDOVER_ROHF.md."
        )

    grad = np.asarray(
        _core.one_electron_gradient_contribution(basis, molecule, p_total),
        dtype=float,
    )
    if use_df:
        aux_basis_name = str(getattr(gradient_options, "aux_basis", "") or "")
        if not aux_basis_name:
            raise ValueError(
                "ROHF density-fitted gradient requires gradient_options.aux_basis"
            )
        aux_basis = type(basis)(molecule, aux_basis_name)
        coeff = np.asarray(result.mo_coeffs, dtype=float)
        c_occ_alpha = coeff[:, : int(result.n_alpha)]
        c_occ_beta = coeff[:, : int(result.n_beta)]
        grad_two = np.asarray(
            _core.compute_df_j_gradient(basis, aux_basis, molecule, p_total),
            dtype=float,
        )
        # The DF-K kernel differentiates a doubly occupied spatial space.
        # Each ROHF spin space contributes half of that convention:
        # E_K = -1/2 sum_sigma Tr[D_sigma K_sigma].
        grad_two += 0.5 * np.asarray(
            _core.compute_df_k_gradient(
                basis, aux_basis, molecule, c_occ_alpha, 1.0
            ),
            dtype=float,
        )
        grad_two += 0.5 * np.asarray(
            _core.compute_df_k_gradient(
                basis, aux_basis, molecule, c_occ_beta, 1.0
            ),
            dtype=float,
        )
        grad = grad + grad_two
    else:
        grad = grad + np.asarray(
            _core.two_electron_gradient_contribution_uhf(
                basis, molecule, dma, dmb, 1.0
            ),
            dtype=float,
        )
    grad = grad + np.asarray(
        _core.overlap_gradient_contribution(basis, molecule, w), dtype=float
    )
    grad = grad + np.asarray(_core.nuclear_repulsion_gradient(molecule), dtype=float)
    return grad


def _nuclear_repulsion(molecule: Any) -> float:
    """Nuclear repulsion energy (Hartree).

    Prefers ``Molecule.nuclear_repulsion()`` (the C++ method the native
    SCF drivers use --- it honours ECP-reduced core charges); falls back
    to the plain ``Z_i Z_j / r_ij`` sum over ``molecule.atoms`` (the same
    construction as ``neb._nuclear_repulsion_molecular``).
    """
    fn = getattr(molecule, "nuclear_repulsion", None)
    if callable(fn):
        return float(fn())
    e = 0.0
    atoms = list(molecule.atoms)
    for i in range(len(atoms)):
        zi = int(atoms[i].Z)
        xi = np.asarray(atoms[i].xyz, dtype=float)
        for j in range(i + 1, len(atoms)):
            zj = int(atoms[j].Z)
            xj = np.asarray(atoms[j].xyz, dtype=float)
            r = float(np.linalg.norm(xi - xj))
            if r > 0.0:
                e += zi * zj / r
    return e
