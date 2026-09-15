"""Common interfaces and data structures for non-mean-field solvers.

All solvers in this package share a common protocol:
    result = solver.solve(hamiltonian, nelec, norb, options) -> SolverResult

A Hamiltonian carries one- and two-electron integrals in an orthonormal
(spatial-orbital) basis.  The orbital provider is responsible for
delivering the transformation from AO -> MO; the solvers themselves are
orbital-basis-agnostic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

# ── Hamiltonian ───────────────────────────────────────────────────────────


@dataclass
class Hamiltonian:
    """One- and two-electron integrals in an orthonormal spatial-orbital basis.

    Attributes
    ----------
    h1e : (norb, norb) ndarray
        One-electron (core) Hamiltonian h_{pq}.
    h2e : (norb, norb, norb, norb) ndarray
        Two-electron integrals in **physicist's** notation:
        g_{pqrs} = (pr|qs) = ∫∫ phi_p*(r₁) phi_q*(r₂) r₁₂⁻¹ phi_r(r₁) phi_s(r₂).
        Stored as a 4-index tensor with anti-symmetry NOT folded in --
        the solvers apply Slater-Condon rules that need the full tensor.
    nuclear_repulsion : float
        Nuclear-nuclear repulsion energy E_nuc (Hartree).
    norb : int
        Number of spatial orbitals.
    nelec : int
        Number of electrons.
    ms2 : int
        2 x S_z = n_alpha - n_beta.  0 for closed-shell singlets.
    description : str
        Human-readable label for logging / diagnostics.
    """

    h1e: np.ndarray
    h2e: np.ndarray
    nuclear_repulsion: float = 0.0
    norb: int = 0
    nelec: int = 0
    ms2: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        if self.norb == 0 and self.h1e is not None:
            self.norb = self.h1e.shape[0]

    def active_space(
        self,
        n_active_orb: int,
        n_active_elec: int,
        *,
        ms2: Optional[int] = None,
    ) -> "Hamiltonian":
        """Project onto a frozen-core-dressed CAS(``n_active_elec``, ``n_active_orb``).

        Standard complete-active-space partition (Roos, Taylor & Siegbahn,
        Chem. Phys. 48, 157 (1980); textbook form in Helgaker, Jorgensen &
        Olsen, *Molecular Electronic-Structure Theory*, Sec.11/Sec.12), identical
        to the convention used by ``mcscf.CASCI`` and by vibe-qc's own
        ``casci``/``casscf``/``nevpt2``/``caspt2`` family: the lowest

            n_core = (self.nelec - n_active_elec) // 2

        molecular orbitals form a doubly-occupied **inactive** core, the next
        ``n_active_orb`` are **active**, and the remainder are virtual.  The
        returned Hamiltonian lives in the active orbital block but folds the
        inactive electrons back in exactly, so a CI over the active
        determinants reproduces the full-space energy:

        * ``h1e`` -- active block dressed by the inactive mean field
          (physicist's ``g_{pqrs} = <pq|rs>``):

              h̃_pq = h_pq + S_c (2 g_{pcqc} - g_{pccq})   (p, q active)

        * ``h2e`` -- the active-active-active-active sub-tensor;
        * ``nuclear_repulsion`` -- the original E_nuc **plus** the constant
          inactive energy

              E_core = 2 S_c h_cc + S_cd (2 g_{cdcd} - g_{cddc})

          so that ``E_total = (active CI eigenvalue) + nuclear_repulsion``
          equals the untruncated CASCI total energy.

        This is the proper frozen-core truncation: the dressing term and the
        ``E_core`` offset are what a bare integral slice
        (``h1e[active, active]``) silently drops -- without them the reported
        energy is the active-only contribution, off by ~``E_core`` Hartree.

        Parameters
        ----------
        n_active_orb : int
            Number of active spatial orbitals (``1 <= n_active_orb <= norb``).
        n_active_elec : int
            Number of electrons in the active space.  The inactive count
            ``self.nelec - n_active_elec`` must be even and non-negative.
        ms2 : int, optional
            ``2 S_z`` of the active electrons.  Defaults to the parent
            Hamiltonian's ``ms2`` -- the doubly-occupied inactive core carries
            no net spin, so all of the molecule's ``2 S_z`` lives in the
            active window.

        Returns
        -------
        Hamiltonian
            ``norb = n_active_orb``, ``nelec = n_active_elec``, with the
            dressed one-electron term and the inactive energy folded into
            ``nuclear_repulsion``.  ``self`` must be the **full** MO-basis
            Hamiltonian (all orbitals) -- the dressing needs the core-active
            integrals that a pre-sliced Hamiltonian would no longer carry.

        Notes
        -----
        The dressing is the same validated routine the CASCI path uses
        (``vibeqc.solvers._casci._frozen_core_dressing``); ``active_space``
        is the entry point for the determinant solvers (``fci``,
        ``selected_ci``, ``dmrg``, ``v2rdm``, ``cisd``, ``transcorrelated_ci``)
        that operate directly on a truncated :class:`Hamiltonian`.
        """
        # Lazy import: keeps _common import-light and avoids front-loading the
        # CASCI module (which pulls scipy + the optional C++ direct backend)
        # just to define this method.  Reusing _frozen_core_dressing keeps the
        # dressing bit-for-bit identical to mcscf-validated CASCI.
        from ._casci import _frozen_core_dressing

        norb = self.norb
        if ms2 is None:
            ms2 = self.ms2
        if not (1 <= n_active_orb <= norb):
            raise ValueError(
                f"n_active_orb ({n_active_orb}) must be in [1, norb={norb}]"
            )
        if n_active_elec < 0:
            raise ValueError(f"n_active_elec ({n_active_elec}) must be >= 0")
        n_inactive_elec = self.nelec - n_active_elec
        if n_inactive_elec < 0 or n_inactive_elec % 2 != 0:
            raise ValueError(
                "active_space needs a non-negative, even inactive-electron "
                f"count; self.nelec={self.nelec}, n_active_elec={n_active_elec} "
                f"gives {n_inactive_elec}. The inactive orbitals are "
                "doubly occupied, so (nelec - n_active_elec) must be even."
            )
        n_core = n_inactive_elec // 2
        if n_core + n_active_orb > norb:
            raise ValueError(
                f"frozen core ({n_core}) + active ({n_active_orb}) exceeds "
                f"norb ({norb}); reduce the active space."
            )
        if abs(ms2) > n_active_elec or (n_active_elec - ms2) % 2 != 0:
            raise ValueError(
                f"ms2={ms2} is incompatible with {n_active_elec} active "
                "electrons (need |ms2| <= n_active_elec and matching parity)."
            )

        active = slice(n_core, n_core + n_active_orb)
        e_core, h1e_active = _frozen_core_dressing(self.h1e, self.h2e, n_core, active)
        h2e_active = np.ascontiguousarray(self.h2e[active, active, active, active])
        return Hamiltonian(
            h1e=np.ascontiguousarray(h1e_active),
            h2e=h2e_active,
            nuclear_repulsion=self.nuclear_repulsion + e_core,
            norb=n_active_orb,
            nelec=n_active_elec,
            ms2=ms2,
            description=(
                f"{self.description} | CAS({n_active_elec}e,{n_active_orb}o)"
            ).lstrip(" |"),
        )


# ── Solver options ────────────────────────────────────────────────────────


@dataclass
class SolverOptions:
    """Base options shared by all non-mean-field solvers.

    Sub-classes add method-specific controls.
    """

    #: Target number of determinants / bond-dimension / constraint set size.
    #: Interpretation is method-specific.
    target_size: int = 100

    #: Energy convergence threshold (Hartree).
    conv_tol_energy: float = 1e-6

    #: Maximum number of macro-iterations.
    max_iter: int = 50

    #: Verbosity: 0 = silent, 1 = per-iteration, 2 = debug.
    verbose: int = 0

    #: Random seed for reproducibility.
    random_seed: int = 42


# ── Solver result ─────────────────────────────────────────────────────────


@dataclass
class SolverResult:
    """Common result container for all non-mean-field solvers.

    Fields marked ``method-specific`` are populated by individual backends
    and may be ``None``.
    """

    #: Total energy (Hartree).  Always includes nuclear repulsion.
    energy: float

    #: Method name, e.g. ``"selected_ci"``, ``"dmrg"``, ``"v2rdm"``.
    method: str

    #: Whether the solver considers itself converged.
    converged: bool = True

    #: Number of macro-iterations / sweeps taken.
    n_iter: int = 0

    #: Energy trace per macro-iteration (if tracked).
    energy_trace: list[float] = field(default_factory=list)

    # ── method-specific ──

    #: Wavefunction coefficients (determinant x configuration), CI only.
    ci_coeffs: Optional[np.ndarray] = None

    #: Determinant / configuration labels, CI only.
    ci_labels: Optional[list[tuple[int, ...]]] = None

    #: 1-RDM (norb, norb), if computed by the solver.
    rdm1: Optional[np.ndarray] = None

    #: 2-RDM (norb, norb, norb, norb), if computed by the solver.
    rdm2: Optional[np.ndarray] = None

    #: Truncation error / discarded weight, DMRG.
    truncation_error: Optional[float] = None

    #: Bond dimension, DMRG.
    bond_dim: Optional[int] = None

    #: N-representability residual norm, v2RDM.
    constraint_residual: Optional[float] = None

    #: Perturbative correction, Selected-CI.
    pt2_correction: Optional[float] = None

    #: Why the solver stopped, Selected-CI.  One of ``"energy"`` (the
    #: delta-E criterion fired), ``"target_size"`` (the requested
    #: determinant-space size was reached), ``"space_exhausted"`` (the
    #: candidate pool emptied below ``target_size`` -- distinguishable
    #: from an energy-converged run), or ``"max_iter"`` (iteration cap).
    #: ``None`` for solvers that do not report a reason.
    stop_reason: Optional[str] = None

    #: Transcorrelated diagnostics dict.
    tc_diagnostics: Optional[dict] = None

    #: Per-root total energies (Hartree), multi-root CASCI / SA-CASSCF /
    #: multi-state CASPT2 only.  ``energy`` remains the headline value
    #: (ground root, or the weighted state average for SA-CASSCF).
    root_energies: Optional[list[float]] = None

    #: Per-root <S^2> alongside ``root_energies`` (same ordering), so the
    #: .out root table shows which spin sector each averaged root lives
    #: in.  None when not computed (diagnostic; degrades to omission).
    root_s2: Optional[list[float]] = None

    #: Per-root Epstein-Nesbet PT2 on a selected-CI wavefunction
    #: (:func:`vibeqc.solvers.selected_ci_pt2` dicts: ``e_pt2``,
    #: ``e_total``, ``stderr``, ``n_perturbers``), requested via
    #: ``CASSCFOptions.pt2``.  ``energy`` stays variational; this is the
    #: SHCI perturbative estimate on top.
    selected_pt2: Optional[list[dict]] = None

    #: Multi-state PT2 diagnostics (MS/XMS-CASPT2): ``mode``, ``heff``
    #: (symmetrized effective Hamiltonian), ``heff_asym``, ``mixing``,
    #: ``ss_energies``, ``ref_energies``, ``e2_corr``, ``xms_rotation``.
    multistate: Optional[dict] = None

    #: Analytic nuclear gradient (n_atoms, 3) in Hartree/bohr.
    #: Computed for CASSCF; None for other methods.
    gradient: Optional[np.ndarray] = None

    #: Ordered physical-root pair for ``nonadiabatic_coupling``.
    nonadiabatic_pair: Optional[tuple[int, int]] = None

    #: State-pair nonadiabatic derivative coupling (n_atoms, 3) in bohr^-1.
    #: Currently available for the under-review MS/XMS-CASPT2 route.
    nonadiabatic_coupling: Optional[np.ndarray] = None

    #: Natural orbitals in the AO basis, shape ``(n_ao, n_mo)``; columns are
    #: orbitals, matching the mean-field ``mo_coeffs`` convention. Populated
    #: for CASCI/CASSCF, where they are the eigenvectors of the converged
    #: one-particle density matrix: inactive orbitals pass through unchanged
    #: (they are already natural, with occupancy 2), active orbitals are the
    #: converged CAS orbitals rotated by the eigenvectors of the active-block
    #: 1-RDM, and virtuals pass through unchanged. ``None`` for solvers that
    #: do not optimize orbitals.
    natural_orbitals: Optional[np.ndarray] = None

    #: Occupation numbers matching :attr:`natural_orbitals` column for
    #: column, in the same order. Electron counts per orbital, so they sum
    #: to the number of electrons. Integer for the inactive (2.0) and
    #: virtual (0.0) blocks; genuinely fractional across the active space,
    #: where the departure from {0, 2} is the multireference character.
    natural_occupations: Optional[np.ndarray] = None

    #: Exact orbital/CI provenance for optional TREXIO export. Kept separate
    #: from natural-orbital visualization because CI coefficients refer to
    #: the optimization/CI orbital basis, not its natural-orbital rotation.
    trexio_wavefunction: Optional[dict] = field(default=None, repr=False)

    @property
    def energy_total(self) -> float:
        """Alias for ``energy``."""
        return self.energy


# ── Solver protocol ───────────────────────────────────────────────────────


class SolverProtocol(Protocol):
    """Structural protocol for a non-mean-field solver."""

    def solve(
        self,
        hamiltonian: Hamiltonian,
        options: SolverOptions | None = None,
    ) -> SolverResult: ...


# ── Utility ───────────────────────────────────────────────────────────────


def _physicist_to_chemist(h2e: np.ndarray) -> np.ndarray:
    """Convert physicist's ERI g_{pqrs} = (pr|qs) -> chemist's (pq|rs).

    (pq|rs) = g_{prqs}
    """
    return h2e.transpose(0, 2, 1, 3)


def _chemist_to_physicist(eri_chem: np.ndarray) -> np.ndarray:
    """Convert chemist's (pq|rs) -> physicist's g_{pqrs} = (pr|qs)."""
    return eri_chem.transpose(0, 2, 1, 3)


def _antisymmetrize_h2e(h2e_phys: np.ndarray) -> np.ndarray:
    """Build the antisymmetrized two-electron tensor <pq||rs>.

    <pq||rs> = g_{pqrs} - g_{pqsr}
    """
    return h2e_phys - h2e_phys.transpose(0, 1, 3, 2)


# ── Experimental-method gating ──────────────────────────────────────────────

#: Opt-in environment variable for the strongly-contracted CASPT2 variant,
#: which is not separately validated against an external reference.  CASCI /
#: CASSCF / NEVPT2 (validated vs PySCF) and internally-contracted CASPT2,
#: including its single-state IPEA route (validated vs OpenMolcas), run
#: **ungated**.
#: Same opt-in pattern as the former ``VIBEQC_EXPERIMENTAL_CCSD`` gate
#: (lifted 2026-06-10 when the molecular CCSD kernel was anchored).
EXPERIMENTAL_MULTIREF_ENV = "VIBEQC_EXPERIMENTAL_MULTIREF"


def require_experimental_multiref(method: str) -> None:
    """Refuse to run a not-externally-validated CASPT2 path unless opted in.

    CASCI / CASSCF / NEVPT2 are validated against PySCF (``mcscf.CASCI`` /
    ``mcscf.CASSCF`` / ``mrpt.NEVPT``) and run **ungated**.  The default
    ``method="caspt2"`` is the **internally-contracted** variant, validated
    against OpenMolcas ``&CASPT2`` to <=2 µHa, and is also **ungated**.

    The **strongly-contracted** variant (``caspt2(variant="sc")``) stays gated:
    its one-perturber-per-external-pattern implementation is superseded by the
    IC variant and is not separately validated against an external CASPT2
    reference.

    Set ``VIBEQC_EXPERIMENTAL_MULTIREF=1`` to proceed.

    Parameters
    ----------
    method : str
        Human-readable name of the gated path, used in the error message.

    Raises
    ------
    RuntimeError
        If ``VIBEQC_EXPERIMENTAL_MULTIREF`` is not set to ``"1"``.
    """
    if os.environ.get(EXPERIMENTAL_MULTIREF_ENV, "").strip() != "1":
        raise RuntimeError(
            f"{method} is gated: it is not sub-microhartree reproducible "
            "against OpenMolcas. The default internally-contracted CASPT2 "
            "(method='caspt2') is validated and un-gated; see "
            f"handovers/HANDOVER_MULTIREF.md. Set {EXPERIMENTAL_MULTIREF_ENV}=1 "
            "to proceed."
        )
