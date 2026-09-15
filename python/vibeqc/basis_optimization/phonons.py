"""Gamma-point phonons and zero-point energy on the Gaussian (BIPOLE) route.

This is the Gaussian-route counterpart of
:mod:`vibeqc.periodic_gapw_phonon`, written for one purpose: to give the
cohesive-energy pipeline (``vibe_basis.pipeline.CohesivePipeline``) a
zero-point correction on the same BIPOLE route the single point and the
relaxation already ride, so that all three stages describe one energy
surface.

Why the Hessian comes from **energies**, not from forces
--------------------------------------------------------
The GAPW module differentiates the *analytic* GAPW force, which is
production on that route. On the BIPOLE route it is not:
:mod:`vibeqc.bipole_optimize`'s module docstring states that the analytic
BIPOLE gradient is a research-preview surface (RHF/UHF Gamma and
maintained KS Gamma pinned; broader KS, multi-k, finite-temperature and
meta-GGA certification open) and that finite differences of the BIPOLE
*energy* are the production force path.

So the production BIPOLE force is itself a finite difference, at ~6N SCFs
per gradient. Building a finite-difference Hessian on top of it would
difference a difference: two stacked FD layers whose step sizes have to
be tuned against each other, compounding noise, at ~36N^2 SCFs.

This module therefore builds the Hessian **directly from total energies**:

* off-diagonal
  ``H_ij = (E(+i,+j) - E(+i,-j) - E(-i,+j) + E(-i,-j)) / (4 h^2)``
* diagonal
  ``H_ii = (E(+i) - 2 E(0) + E(-i)) / h^2``

One differencing layer, resting only on the total energy -- the quantity
that is correct by construction on this route -- at ``18 N^2 + 1`` SCFs
(73 for a 2-atom primitive cell). :func:`estimate_scf_count` reports the
cost before a run starts, because it grows quadratically.

``hessian_mode="analytic_force"`` differentiates the research-preview
analytic BIPOLE gradient instead, at ``6N`` SCFs. It is offered for
research on that gradient and emits a warning, exactly as
``bipole_optimize`` does for ``force_mode="analytic"``. It is not the
default.

**The M1 cross-route gate has not been run yet.** No frequency here has
been compared against the GAPW route; the gate driver exists
(``examples/basisset_dev/gamma_phonon_gate.py``) and the submission
recipe is in ``handovers/HANDOVER_BASISOPT.md``, but the numbers do not
exist. Nothing in this module should be read as gate-validated.

Scope
-----
* **Gamma-point only.** A Gamma-only ZPE is an approximation for a solid:
  it samples one q-point of the phonon Brillouin zone, so it misses the
  dispersion of the acoustic branches away from Gamma entirely. It is the
  standard cheap estimate and it is what this module computes; phonon-
  dispersion ZPE (supercell or q-mesh) is out of scope. Treat the number
  as a correction of the right size, not a converged thermodynamic value.
* The lattice is held fixed. Only atomic (internal) displacements are
  differenced, so these are the zone-centre optic modes plus the three
  translational acoustic modes -- not the elastic/strain degrees of
  freedom.
* Convention for imaginary modes matches :mod:`vibeqc.hessian` and
  :mod:`vibeqc.thermo`: a negative ``omega^2`` is reported as a
  **negative wavenumber** and is excluded from the ZPE sum, with the
  count carried on the result as ``n_imaginary_modes_excluded``.

Acoustic modes
--------------
Translational invariance makes the three Gamma acoustic modes exactly
zero in exact arithmetic. In a finite-difference Hessian from a finite
basis they are not, and the residual is a useful diagnostic of the whole
calculation's noise floor.

**By default this module does not impose the acoustic sum rule.** It
identifies the acoustic modes by projection onto the mass-weighted
uniform-translation subspace, reports their residual frequencies on the
result, and excludes them from the ZPE (their true frequency is zero, so
they carry no zero-point energy). ``asr="rowsum"`` or ``asr="project"``
imposes it; ``GammaPhonons.asr`` records which was done, so a residual
that has been zeroed can never be mistaken for one that was measured.

Usage
-----
::

    from vibeqc.basis_optimization.phonons import gamma_phonons

    ph = gamma_phonons(system, "pob-tzvp-rev2", method="RKS",
                       functional="pbe", step_bohr=0.02)
    print(ph.frequencies_cm1)
    print(ph.zero_point_kj_per_mol_per_formula_unit(n_formula_units=1))
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import warnings
from typing import Any, Optional, Sequence

import numpy as np

from .._vibeqc_core import Atom, BasisSet, PeriodicSystem, attach_symmetry
from ..output import Level, write
from ..periodic_gapw_phonon import (
    _AMU_TO_ELECTRON_MASS,
    _HARTREE_TO_CM_INV,
    _HARTREE_TO_THZ,
    _resolve_masses,
    phonon_eigenvalues,
)

_log = logging.getLogger("vibeqc.basis_optimization.phonons")

__all__ = [
    "DEFAULT_STEP_BOHR",
    "HARTREE_TO_KJ_PER_MOL",
    "GammaPhonons",
    "apply_acoustic_sum_rule",
    "compute_hessian_fd_energy",
    "compute_hessian_fd_force",
    "estimate_scf_count",
    "gamma_phonons",
    "phonons_from_hessian",
]


#: 1 cm^-1 expressed in Hartree. Same constant, inverted, that
#: :mod:`vibeqc.thermo` uses -- kept consistent so a periodic ZPE and a
#: molecular ZPE cannot disagree in the last digits.
_CM_INV_TO_HARTREE: float = 1.0 / _HARTREE_TO_CM_INV

#: CODATA-derived; matches ``vibe_basis.pipeline.HARTREE_TO_KJ_PER_MOL``
#: and the 2625.4996 quoted in :mod:`vibeqc.thermo`.
HARTREE_TO_KJ_PER_MOL: float = 2625.4996394798254

#: Frequencies with ``|omega| <= _ZERO_CM1`` are treated as numerically
#: zero and contribute nothing. Matches :mod:`vibeqc.thermo`'s 1 cm^-1
#: threshold so the two modules classify a mode the same way.
_ZERO_CM1: float = 1.0

#: Default central-difference half-step on atomic positions, in bohr.
#:
#: 0.02 bohr (~0.011 Angstrom) -- deliberately **twice** the GAPW
#: module's 0.01, and the reason is the extra differencing this route
#: does not do. GAPW differences an analytic force once, so its error is
#: ``O(h^2) + eps/h``. An energy Hessian divides by ``h^2``, so its SCF-
#: noise term is ``eps/h^2``: at a converged-SCF energy noise of
#: ~1e-9 Ha, h = 0.01 bohr already puts ~1e-5 Ha/bohr^2 of noise into a
#: matrix element, while the truncation error stays well under a cm^-1
#: at 0.02 for these stiff ionic modes.
#:
#: **That is a reasoned estimate, not a measurement.** No step-size scan
#: has been run on this route yet. Verify with one on a new system class
#: rather than assuming this transfers -- ``--steps`` on
#: ``examples/basisset_dev/gamma_phonon_gate.py`` does it, and the
#: numbers belong in ``handovers/HANDOVER_BASISOPT.md`` once they exist.
DEFAULT_STEP_BOHR: float = 0.02


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GammaPhonons:
    """Gamma-point phonons of one periodic cell on the BIPOLE route.

    Attributes
    ----------
    frequencies_cm1, frequencies_thz
        All ``3N`` harmonic frequencies, ascending. Imaginary modes are
        reported as negative wavenumbers.
    eigenvectors
        Mass-weighted normal modes as columns, matching ``frequencies_*``.
    hessian
        The ``(3N, 3N)`` Cartesian force-constant matrix in Ha/bohr^2,
        after symmetrisation and after any acoustic sum rule.
    dynamical_matrix
        ``M^{-1/2} H M^{-1/2}`` in atomic units.
    masses_amu
        Per-atom masses used for the weighting.
    acoustic_indices
        Indices into ``frequencies_cm1`` of the three modes identified as
        acoustic by projection onto the uniform-translation subspace.
    acoustic_residual_cm1
        Their frequencies. **The honest residual when ``asr == "none"``**;
        with an ASR imposed they are zero by construction and say nothing
        about the calculation's noise floor.
    acoustic_projection
        For each acoustic mode, the fraction of its norm lying in the
        translation subspace (1.0 = pure translation). A value well below
        ~0.9 means the acoustic modes are mixing with optical ones and
        the identification should not be trusted.
    zero_point_hartree
        ZPE **per unit cell**, in Hartree, positive. Acoustic and
        imaginary modes are excluded; see the module docstring.
    n_imaginary_modes_excluded, n_acoustic_modes_excluded
        How many modes the sum dropped, and why.
    fd_step_bohr, hessian_mode, asr
        The settings that produced this result. Provenance, not decoration
        -- a frequency without its step size is not reproducible.
    n_scf
        Number of SCF evaluations spent.
    detail
        Free-form provenance (method, functional, k-mesh, ...).
    """

    frequencies_cm1: np.ndarray
    frequencies_thz: np.ndarray
    eigenvectors: np.ndarray
    hessian: np.ndarray
    dynamical_matrix: np.ndarray
    masses_amu: np.ndarray
    acoustic_indices: tuple[int, ...]
    acoustic_residual_cm1: tuple[float, ...]
    acoustic_projection: tuple[float, ...]
    zero_point_hartree: float
    n_imaginary_modes_excluded: int
    n_acoustic_modes_excluded: int
    fd_step_bohr: float
    hessian_mode: str
    asr: str
    n_scf: int
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        """Total number of modes, ``3N``."""
        return int(self.frequencies_cm1.shape[0])

    @property
    def optical_frequencies_cm1(self) -> np.ndarray:
        """Frequencies with the acoustic modes removed, ascending."""
        mask = np.ones(self.n_modes, dtype=bool)
        for idx in self.acoustic_indices:
            mask[idx] = False
        return self.frequencies_cm1[mask]

    @property
    def zero_point_kj_per_mol(self) -> float:
        """ZPE per unit cell, kJ/mol."""
        return self.zero_point_hartree * HARTREE_TO_KJ_PER_MOL

    def zero_point_per_formula_unit(self, n_formula_units: int = 1) -> float:
        """ZPE per formula unit, in Hartree."""
        n = int(n_formula_units)
        if n < 1:
            raise ValueError(
                f"n_formula_units must be >= 1; got {n_formula_units!r}"
            )
        return self.zero_point_hartree / n

    def zero_point_kj_per_mol_per_formula_unit(
        self, n_formula_units: int = 1
    ) -> float:
        """ZPE per formula unit, kJ/mol -- the pipeline's reporting unit."""
        return (
            self.zero_point_per_formula_unit(n_formula_units)
            * HARTREE_TO_KJ_PER_MOL
        )

    def summary(self, n_formula_units: int = 1) -> str:
        """A short human-readable block. Rendered by the caller."""
        lines = [
            f"Gamma phonons ({self.hessian_mode}, h = "
            f"{self.fd_step_bohr:g} bohr, ASR = {self.asr}, "
            f"{self.n_scf} SCF)",
        ]
        for k, w in enumerate(self.frequencies_cm1):
            tag = " (acoustic)" if k in self.acoustic_indices else ""
            lines.append(f"  mode {k:>3d}  {w:12.3f} cm^-1{tag}")
        if self.asr == "none":
            worst = max((abs(w) for w in self.acoustic_residual_cm1), default=0.0)
            lines.append(f"  acoustic residual (not corrected): {worst:.3f} cm^-1")
        if self.n_imaginary_modes_excluded:
            lines.append(
                f"  imaginary modes excluded from the ZPE: "
                f"{self.n_imaginary_modes_excluded}"
            )
        lines.append(
            f"  ZPE = {self.zero_point_hartree:.8f} Ha/cell = "
            f"{self.zero_point_kj_per_mol_per_formula_unit(n_formula_units):.3f} "
            f"kJ/mol per formula unit"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def estimate_scf_count(n_atoms: int, hessian_mode: str = "energy") -> int:
    """SCF evaluations a Hessian on *n_atoms* will cost.

    ``"energy"`` -> ``18 N^2 + 1``; ``"analytic_force"`` -> ``6 N``.

    Call this before starting: the energy route is quadratic in the cell
    size, so what is 73 SCFs for a 2-atom primitive cell is 1153 for an
    8-atom one.
    """
    n = int(n_atoms)
    if n < 1:
        raise ValueError(f"n_atoms must be >= 1; got {n_atoms!r}")
    mode = str(hessian_mode)
    if mode == "energy":
        n_dof = 3 * n
        # 1 central + 2 per DOF (singles) + 4 per unordered DOF pair.
        return 1 + 2 * n_dof + 4 * (n_dof * (n_dof - 1)) // 2
    if mode == "analytic_force":
        return 6 * n
    raise ValueError(
        f"hessian_mode must be 'energy' or 'analytic_force'; got {hessian_mode!r}"
    )


# ---------------------------------------------------------------------------
# Geometry + SCF plumbing
# ---------------------------------------------------------------------------


def _displaced_system(
    system: PeriodicSystem,
    displacements: dict[tuple[int, int], float],
) -> PeriodicSystem:
    """``system`` with ``{(atom, axis): delta_bohr}`` applied. Lattice fixed."""
    new_atoms = []
    for i, atom in enumerate(system.unit_cell):
        xyz = list(atom.xyz)
        for (a, cart), delta in displacements.items():
            if a == i:
                xyz[cart] += float(delta)
        new_atoms.append(Atom(int(atom.Z), xyz))
    out = PeriodicSystem(
        system.dim,
        np.asarray(system.lattice, dtype=np.float64),
        new_atoms,
        charge=system.charge,
        multiplicity=system.multiplicity,
    )
    # Preserve the caller's attached-symmetry *calculation mode*, exactly
    # as `compute_bipole_gradient_fd` does. A displacement can lower the
    # point group, but even its C1 operation keeps the group-invariant
    # pair-resolved Fock domain active; dropping symmetry entirely would
    # switch displaced points back to the radial domain and difference two
    # different energy surfaces.
    symmetry = getattr(system, "symmetry", None)
    if getattr(symmetry, "operations", None):
        attach_symmetry(out)
    return out


def _run_bipole(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh,
    options,
    *,
    method: str,
    functional: Optional[str],
    bipole_kwargs: dict,
):
    """One BIPOLE SCF. Returns the result object."""
    method_upper = method.upper()
    if method_upper == "RHF":
        from ..pbc_bipole import run_pbc_bipole_rhf

        return run_pbc_bipole_rhf(
            system, basis, kmesh, options, progress=False, **bipole_kwargs
        )
    if method_upper == "UHF":
        from ..pbc_bipole_uhf import run_pbc_bipole_uhf

        return run_pbc_bipole_uhf(
            system, basis, kmesh, options, progress=False, **bipole_kwargs
        )
    if method_upper == "RKS":
        from ..pbc_bipole_rks import run_pbc_bipole_rks

        return run_pbc_bipole_rks(
            system,
            basis,
            kmesh,
            options,
            functional=functional,
            progress=False,
            **bipole_kwargs,
        )
    if method_upper == "UKS":
        from ..pbc_bipole_uks import run_pbc_bipole_uks

        return run_pbc_bipole_uks(
            system,
            basis,
            kmesh,
            options,
            functional=functional,
            progress=False,
            **bipole_kwargs,
        )
    raise ValueError(
        f"phonons: unknown method {method!r} (expected RHF, UHF, RKS or UKS)"
    )


class _EnergySampler:
    """Caches ``E(displacement pattern)`` across the Hessian sweep.

    The mixed diagonal/off-diagonal stencil reuses every single-DOF point
    ``E(+i)`` / ``E(-i)`` for the diagonal, so the cache is not an
    optimisation detail: without it the sweep would pay for those ``6N``
    points twice.
    """

    def __init__(
        self,
        system: PeriodicSystem,
        basis_name: str,
        kmesh,
        options,
        *,
        method: str,
        functional: Optional[str],
        bipole_kwargs: dict,
    ) -> None:
        self._system = system
        self._basis_name = str(basis_name)
        self._kmesh = kmesh
        self._options = options
        self._method = method
        self._functional = functional
        self._bipole_kwargs = dict(bipole_kwargs)
        self._cache: dict[tuple, float] = {}
        self.n_scf = 0

    @staticmethod
    def _key(displacements: dict[tuple[int, int], float]) -> tuple:
        return tuple(sorted((a, c, float(d)) for (a, c), d in displacements.items()))

    def energy(self, displacements: dict[tuple[int, int], float]) -> float:
        key = self._key(displacements)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        sys_disp = _displaced_system(self._system, displacements)
        basis_disp = BasisSet(sys_disp.unit_cell_molecule(), self._basis_name)
        result = _run_bipole(
            sys_disp,
            basis_disp,
            self._kmesh,
            self._options,
            method=self._method,
            functional=self._functional,
            bipole_kwargs=self._bipole_kwargs,
        )
        self.n_scf += 1

        if not bool(getattr(result, "converged", True)):
            n_iter = getattr(result, "n_iter", "unknown")
            raise RuntimeError(
                "phonons: SCF did not converge at displacement "
                f"{key} (n_iter={n_iter}). Refusing to finite-difference a "
                "non-converged energy -- a Hessian divides by h^2, so a "
                "half-converged point is amplified, not averaged out."
            )

        # Finite-temperature (Fermi-smeared) KS: the variationally
        # consistent quantity is the Mermin free energy A = E - T.S, which
        # is what the production FD force differentiates
        # (`compute_bipole_gradient_fd`). Differencing E_total instead
        # would carry the non-variational occupation-response term. At
        # T = 0 the two coincide.
        if float(getattr(result, "smearing_temperature", 0.0) or 0.0) > 0.0:
            energy = float(getattr(result, "free_energy", result.energy))
        else:
            energy = float(result.energy)

        self._cache[key] = energy
        return energy


# ---------------------------------------------------------------------------
# Hessians
# ---------------------------------------------------------------------------


def compute_hessian_fd_energy(
    system: PeriodicSystem,
    basis_name: str,
    kmesh=None,
    options=None,
    *,
    method: str = "RHF",
    functional: Optional[str] = None,
    step_bohr: float = DEFAULT_STEP_BOHR,
    symmetrize: bool = True,
    progress: bool = False,
    **bipole_kwargs,
) -> tuple[np.ndarray, int]:
    """Cartesian force-constant matrix from central differences of the energy.

    Returns ``(H, n_scf)`` with ``H`` of shape ``(3N, 3N)`` in Ha/bohr^2
    and ``n_scf`` the number of SCFs actually run.

    Parameters
    ----------
    system
        Equilibrium-geometry cell. **Displacements are meaningful only at
        a stationary point**: a Hessian at a geometry with residual forces
        still diagonalises, and its low modes are then contaminated by the
        gradient. Relax first.
    basis_name
        Basis name; the basis is rebuilt at every displaced geometry.
    kmesh
        A ``BlochKMesh``. ``None`` means Gamma only.
    options
        Periodic SCF options object, forwarded unchanged to every point so
        that all ``18 N^2 + 1`` energies come off the same settings.
    method, functional
        ``"RHF"`` / ``"UHF"`` / ``"RKS"`` / ``"UKS"``, plus the XC
        functional for the KS branches.
    step_bohr
        Central-difference half-step. See :data:`DEFAULT_STEP_BOHR` for
        why it is larger than the GAPW module's.
    symmetrize
        Average ``H`` with its transpose. On by default: the stencil is
        symmetric in exact arithmetic, so the antisymmetric part is pure
        numerical noise, and halving it is free.
    """
    if system.dim != 3:
        raise NotImplementedError(
            f"compute_hessian_fd_energy: only 3D periodic cells are "
            f"supported; got dim = {system.dim}."
        )
    h = float(step_bohr)
    if not (h > 0.0) or not math.isfinite(h):
        raise ValueError(f"step_bohr must be a positive finite float; got {step_bohr!r}")

    n_atoms = len(system.unit_cell)
    n_dof = 3 * n_atoms
    sampler = _EnergySampler(
        system,
        basis_name,
        kmesh,
        options,
        method=method,
        functional=functional,
        bipole_kwargs=bipole_kwargs,
    )

    def dof(q: int) -> tuple[int, int]:
        return divmod(q, 3)

    if progress:
        write(
            f"Gamma-point Hessian from BIPOLE energies: {n_atoms} atoms, "
            f"{n_dof} DOF, h = {h:g} bohr, "
            f"{estimate_scf_count(n_atoms, 'energy')} SCF evaluations.\n",
            Level.VERBOSE,
        )

    e0 = sampler.energy({})
    H = np.zeros((n_dof, n_dof), dtype=np.float64)

    # Diagonal: standard 3-point second derivative.
    #   H_ii = (E(+i) - 2 E(0) + E(-i)) / h^2
    for q in range(n_dof):
        a, c = dof(q)
        e_p = sampler.energy({(a, c): +h})
        e_m = sampler.energy({(a, c): -h})
        H[q, q] = (e_p - 2.0 * e0 + e_m) / (h * h)
        _log.info("phonons: diagonal DOF %d/%d done", q + 1, n_dof)

    # Off-diagonal: the 4-point mixed second derivative.
    #   H_ij = (E(+i,+j) - E(+i,-j) - E(-i,+j) + E(-i,-j)) / (4 h^2)
    for q in range(n_dof):
        aq, cq = dof(q)
        for r in range(q + 1, n_dof):
            ar, cr = dof(r)

            def _e(sq: float, sr: float, _aq=aq, _cq=cq, _ar=ar, _cr=cr) -> float:
                # (atom, axis) keys are distinct whenever q != r, so a
                # same-atom / different-axis pair stays two dict entries.
                return sampler.energy({(_aq, _cq): sq * h, (_ar, _cr): sr * h})

            H[q, r] = (
                _e(+1.0, +1.0) - _e(+1.0, -1.0) - _e(-1.0, +1.0) + _e(-1.0, -1.0)
            ) / (4.0 * h * h)
            H[r, q] = H[q, r]
        _log.info("phonons: off-diagonal row %d/%d done", q + 1, n_dof)

    if symmetrize:
        H = 0.5 * (H + H.T)
    return H, sampler.n_scf


def compute_hessian_fd_force(
    system: PeriodicSystem,
    basis_name: str,
    kmesh=None,
    options=None,
    *,
    method: str = "RHF",
    functional: Optional[str] = None,
    step_bohr: float = 0.01,
    symmetrize: bool = True,
    progress: bool = False,
    **bipole_kwargs,
) -> tuple[np.ndarray, int]:
    """Force-constant matrix from central differences of the *analytic* force.

    ``6N`` SCFs instead of ``18 N^2 + 1``, and it is **research-only**.
    The analytic BIPOLE gradient is a research-preview surface (see
    :mod:`vibeqc.bipole_optimize`); the KS, multi-k, finite-temperature
    and meta-GGA cases a cohesive-energy campaign needs are exactly the
    ones that are not certified. A :class:`UserWarning` is emitted for the
    same reason ``bipole_optimize`` emits one for
    ``force_mode="analytic"``.

    Use it to study the analytic gradient. Do not use it to produce a ZPE
    you intend to publish, and do not use it to satisfy a gate.
    """
    warnings.warn(
        "compute_hessian_fd_force differentiates the research-preview "
        "analytic BIPOLE gradient. Broader KS, multi-k, finite-temperature "
        "and meta-GGA cases are uncertified; the production route is "
        "compute_hessian_fd_energy (hessian_mode='energy').",
        UserWarning,
        stacklevel=2,
    )
    if system.dim != 3:
        raise NotImplementedError(
            f"compute_hessian_fd_force: only 3D periodic cells are "
            f"supported; got dim = {system.dim}."
        )
    from ..bipole_gradient import (
        compute_bipole_gradient_rhf,
        compute_bipole_gradient_rks,
        compute_bipole_gradient_uhf,
        compute_bipole_gradient_uks,
    )

    grad_fn = {
        "RHF": compute_bipole_gradient_rhf,
        "UHF": compute_bipole_gradient_uhf,
        "RKS": compute_bipole_gradient_rks,
        "UKS": compute_bipole_gradient_uks,
    }.get(method.upper())
    if grad_fn is None:
        raise ValueError(f"phonons: unknown method {method!r}")

    h = float(step_bohr)
    n_atoms = len(system.unit_cell)
    n_dof = 3 * n_atoms
    H = np.zeros((n_dof, n_dof), dtype=np.float64)
    n_scf = 0

    if progress:
        write(
            f"Gamma-point Hessian from the analytic BIPOLE force "
            f"(research preview): {n_atoms} atoms, {6 * n_atoms} SCF.\n",
            Level.VERBOSE,
        )

    for q in range(n_dof):
        a, c = divmod(q, 3)
        rows = []
        for sign in (+1.0, -1.0):
            sys_disp = _displaced_system(system, {(a, c): sign * h})
            basis_disp = BasisSet(sys_disp.unit_cell_molecule(), basis_name)
            result = _run_bipole(
                sys_disp,
                basis_disp,
                kmesh,
                options,
                method=method,
                functional=functional,
                bipole_kwargs=bipole_kwargs,
            )
            n_scf += 1
            if not bool(getattr(result, "converged", True)):
                raise RuntimeError(
                    f"phonons: SCF did not converge at atom {a}, axis {c}, "
                    f"step {sign * h:+g} bohr."
                )
            rows.append(
                np.asarray(
                    grad_fn(sys_disp, basis_disp, result), dtype=np.float64
                ).reshape(-1)
            )
        H[q, :] = (rows[0] - rows[1]) / (2.0 * h)

    if symmetrize:
        H = 0.5 * (H + H.T)
    return H, n_scf


# ---------------------------------------------------------------------------
# Acoustic sum rule + mode classification
# ---------------------------------------------------------------------------


#: Iterations the ``"rowsum"`` ASR takes to satisfy translational
#: invariance and symmetry *simultaneously*. One pass of the row-sum
#: correction leaves a matrix whose column sums are still nonzero, and
#: re-symmetrising then reintroduces a row-sum residual equal to half the
#: row/column asymmetry. That asymmetry is itself numerical error, so the
#: alternation contracts quickly -- but it does need to be alternated
#: rather than done once and declared correct.
_ASR_ROWSUM_ITERATIONS: int = 64
_ASR_ROWSUM_TOL: float = 1e-14


def _row_sums(H: np.ndarray, n_atoms: int) -> np.ndarray:
    """``S[i, a, b] = sum_j Phi[3i+a, 3j+b]`` -- zero iff the ASR holds."""
    return H.reshape(n_atoms, 3, n_atoms, 3).sum(axis=2)


def apply_acoustic_sum_rule(hessian: np.ndarray, mode: str = "rowsum") -> np.ndarray:
    """Impose translational invariance on a force-constant matrix.

    ``"rowsum"`` (the usual convention) enforces ``sum_j Phi_ij^ab = 0``
    by absorbing the residual into each atom's self term ``Phi_ii``. It
    leaves the inter-atomic constants untouched, which is what makes it
    the standard choice: the self term is the one a finite-difference
    stencil determines least well. Applied alternately with
    symmetrisation until both hold (see
    :data:`_ASR_ROWSUM_ITERATIONS`), because one pass of either breaks
    the other.

    ``"project"`` instead removes the uniform-translation subspace from
    the matrix entirely (``P H P`` with ``P = 1 - T T^T`` over
    *unweighted* Cartesian translations -- the unweighted ones are right,
    since the ASR is a statement about ``Phi``, not about ``D``). Exact in
    one step and symmetric by construction, but more aggressive: it also
    removes any translational admixture an optical mode had.

    ``"none"`` returns the matrix unchanged. It is the module default, so
    that the acoustic residual stays a measurement.

    Returns a new array; the input is not modified.
    """
    H = np.array(hessian, dtype=np.float64, copy=True)
    if mode == "none":
        return H
    n_dof = H.shape[0]
    if H.ndim != 2 or H.shape[1] != n_dof or n_dof % 3:
        raise ValueError(
            f"apply_acoustic_sum_rule: expected a (3N, 3N) matrix; got {H.shape}."
        )
    n_atoms = n_dof // 3

    if mode == "rowsum":
        H = 0.5 * (H + H.T)
        for _ in range(_ASR_ROWSUM_ITERATIONS):
            residual = _row_sums(H, n_atoms)
            if float(np.abs(residual).max()) < _ASR_ROWSUM_TOL:
                break
            for i in range(n_atoms):
                H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] -= residual[i]
            H = 0.5 * (H + H.T)
        return H

    if mode == "project":
        T = np.zeros((n_dof, 3), dtype=np.float64)
        for i in range(n_atoms):
            for a in range(3):
                T[3 * i + a, a] = 1.0
        Q, _ = np.linalg.qr(T)
        P = np.eye(n_dof) - Q @ Q.T
        H = P @ H @ P
        return 0.5 * (H + H.T)

    raise ValueError(
        f"apply_acoustic_sum_rule: mode must be 'none', 'rowsum' or "
        f"'project'; got {mode!r}"
    )


def _identify_acoustic(
    eigenvectors: np.ndarray, masses_amu: Sequence[float]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """The three modes closest to a uniform translation, and their weights.

    Identification is by **projection**, not by "the three smallest
    frequencies". Picking the smallest is wrong exactly when it matters:
    a genuinely imaginary optical mode, or a soft optic branch, sorts
    below a noisy acoustic residual, and the ZPE would then drop the
    physical mode and keep the spurious one.

    The mass-weighted eigenvectors of ``D = M^{-1/2} H M^{-1/2}`` see a
    uniform Cartesian translation as ``u_i = sqrt(m_i) e_alpha``, so that
    is the subspace projected onto.
    """
    n_dof = eigenvectors.shape[0]
    n_atoms = n_dof // 3
    T = np.zeros((n_dof, 3), dtype=np.float64)
    for i in range(n_atoms):
        w = math.sqrt(float(masses_amu[i]))
        for a in range(3):
            T[3 * i + a, a] = w
    Q, _ = np.linalg.qr(T)

    weights = np.einsum("ij,ij->j", Q.T @ eigenvectors, Q.T @ eigenvectors)
    order = np.argsort(-weights)[:3]
    order = tuple(sorted(int(i) for i in order))
    return order, tuple(float(weights[i]) for i in order)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def gamma_phonons(
    system: PeriodicSystem,
    basis_name: str,
    kmesh=None,
    options=None,
    *,
    method: str = "RHF",
    functional: Optional[str] = None,
    step_bohr: float = DEFAULT_STEP_BOHR,
    hessian_mode: str = "energy",
    asr: str = "none",
    progress: bool = False,
    **bipole_kwargs,
) -> GammaPhonons:
    """Gamma-point frequencies and ZPE for one periodic cell.

    Parameters
    ----------
    system, basis_name, kmesh, options, method, functional
        As :func:`compute_hessian_fd_energy`.
    step_bohr
        Central-difference half-step, bohr. See :data:`DEFAULT_STEP_BOHR`.
    hessian_mode
        ``"energy"`` (default, production) or ``"analytic_force"``
        (research; see :func:`compute_hessian_fd_force`).
    asr
        ``"none"`` (default) reports the acoustic residual without
        touching it. ``"rowsum"`` / ``"project"`` impose the acoustic sum
        rule. The choice is recorded on the result.
    progress
        Emit a cost line through :mod:`vibeqc.output` before starting.

    Returns
    -------
    GammaPhonons
    """
    if hessian_mode == "energy":
        H, n_scf = compute_hessian_fd_energy(
            system,
            basis_name,
            kmesh,
            options,
            method=method,
            functional=functional,
            step_bohr=step_bohr,
            progress=progress,
            **bipole_kwargs,
        )
    elif hessian_mode == "analytic_force":
        H, n_scf = compute_hessian_fd_force(
            system,
            basis_name,
            kmesh,
            options,
            method=method,
            functional=functional,
            step_bohr=step_bohr,
            progress=progress,
            **bipole_kwargs,
        )
    else:
        raise ValueError(
            f"hessian_mode must be 'energy' or 'analytic_force'; "
            f"got {hessian_mode!r}"
        )

    return phonons_from_hessian(
        H,
        system,
        fd_step_bohr=float(step_bohr),
        hessian_mode=str(hessian_mode),
        asr=str(asr),
        n_scf=int(n_scf),
        detail={
            "method": method,
            "functional": functional,
            "basis": str(basis_name),
            "kmesh": repr(kmesh),
        },
    )


def phonons_from_hessian(
    hessian: np.ndarray,
    system: PeriodicSystem,
    *,
    fd_step_bohr: float = float("nan"),
    hessian_mode: str = "energy",
    asr: str = "none",
    n_scf: int = 0,
    detail: Optional[dict] = None,
) -> GammaPhonons:
    """Mass-weight, diagonalise, classify and sum a force-constant matrix.

    Split out from :func:`gamma_phonons` so the analysis half is testable
    against an analytic Hessian without running a single SCF.
    """
    H_raw = np.asarray(hessian, dtype=np.float64)
    n_dof = H_raw.shape[0]
    n_atoms = len(system.unit_cell)
    if H_raw.ndim != 2 or H_raw.shape != (n_dof, n_dof) or n_dof != 3 * n_atoms:
        raise ValueError(
            f"phonons_from_hessian: expected a ({3 * n_atoms}, {3 * n_atoms}) "
            f"matrix for {n_atoms} atoms; got {H_raw.shape}."
        )

    H = apply_acoustic_sum_rule(H_raw, asr)
    masses_amu = _resolve_masses(system)

    masses_me = np.empty(n_dof, dtype=np.float64)
    for i, m in enumerate(masses_amu):
        masses_me[3 * i : 3 * i + 3] = float(m) * _AMU_TO_ELECTRON_MASS
    inv_sqrt_m = 1.0 / np.sqrt(masses_me)
    D = H * np.outer(inv_sqrt_m, inv_sqrt_m)
    D = 0.5 * (D + D.T)

    freqs_cm1, freqs_thz, modes = phonon_eigenvalues(D, masses_amu=masses_amu)

    acoustic_idx, acoustic_weight = _identify_acoustic(modes, masses_amu)
    acoustic_residual = tuple(float(freqs_cm1[i]) for i in acoustic_idx)

    # --- the ZPE sum -------------------------------------------------
    # ZPE = (1/2) sum hbar omega, positive, per unit cell. The pipeline
    # SUBTRACTS it from the cohesive energy (`CohesiveResult`), because a
    # vibrating solid is less bound than a static one.
    #
    # Excluded: the three acoustic modes (their exact frequency is zero,
    # so the residual is numerical and carries no zero-point energy) and
    # any imaginary mode. Imaginary modes are information, not noise --
    # they say the geometry is not a minimum -- so they are counted on the
    # result rather than silently dropped, matching
    # `vibeqc.thermo`'s `n_imaginary_modes_excluded`.
    keep = np.ones(n_dof, dtype=bool)
    for i in acoustic_idx:
        keep[i] = False
    n_imaginary = int(np.sum((freqs_cm1 < -_ZERO_CM1) & keep))
    vib = keep & (freqs_cm1 > _ZERO_CM1)
    zpe = 0.5 * float(np.sum(freqs_cm1[vib] * _CM_INV_TO_HARTREE))

    return GammaPhonons(
        frequencies_cm1=freqs_cm1,
        frequencies_thz=freqs_thz,
        eigenvectors=modes,
        hessian=H,
        dynamical_matrix=D,
        masses_amu=masses_amu,
        acoustic_indices=acoustic_idx,
        acoustic_residual_cm1=acoustic_residual,
        acoustic_projection=acoustic_weight,
        zero_point_hartree=zpe,
        n_imaginary_modes_excluded=n_imaginary,
        n_acoustic_modes_excluded=len(acoustic_idx),
        fd_step_bohr=float(fd_step_bohr),
        hessian_mode=str(hessian_mode),
        asr=str(asr),
        n_scf=int(n_scf),
        detail=dict(detail or {}),
    )
