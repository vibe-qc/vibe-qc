"""ASE Calculator wrapper for the periodic GPW route.

The :class:`VibeqcGPW` calculator drives vibe-qc's M3 GPW periodic
SCF (`run_periodic_rhf_gpw` for Γ-only RHF / RKS, and
`run_periodic_rks_gpw_multi_k` for multi-k pure-DFT RKS) from the
standard ASE Calculator API. This matches GPAW's user-facing
shape, so a user can swap calculators with no other code change::

    from ase.build import bulk
    from vibeqc.ase_periodic_gpw import VibeqcGPW

    atoms = bulk(...)          # any 3D periodic ase.Atoms
    atoms.calc = VibeqcGPW(basis='sto-3g', functional='lda',
                            cutoff_ha=300.0, kmesh=[2, 2, 2])
    print(atoms.get_potential_energy())   # eV

Scope (v0.11.0, M3e-aligned)

* Closed-shell only -- the underlying GPW drivers require an even
  electron count. Open-shell atoms raise :class:`ValueError` from
  the SCF driver; the wrapper additionally pre-checks before
  building basis sets and gives a clear error.
* Forces are computed by central-difference finite differences
  on the SCF energy (no analytic gradient yet -- the analytic
  GPW gradient is post-M3 work). The displaced SCFs are warm-
  started from the central-density solution to keep the cost
  manageable. Step size is configurable via ``fd_step_bohr``
  (default 0.01 bohr).
* Multi-k requires a pure functional (no HF exchange); hybrids on
  the multi-k path raise ``NotImplementedError`` from the C++
  driver, propagated to the caller.
* ``functional=None`` selects Hartree-Fock (Γ-only RHF). Setting
  both ``functional=None`` and ``kmesh=[a, b, c]`` with a, b, c
  > 1 is not supported -- multi-k GPW is pure-DFT only (see
  :func:`vibeqc.run_periodic_rks_gpw_multi_k`).

Unit conventions: ASE works in Å + eV, vibe-qc in bohr + Hartree.
Conversions use ASE's own constants (:attr:`ase.units.Bohr`,
:attr:`ase.units.Hartree`).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Bohr, Hartree
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "vibeqc.ase_periodic_gpw requires ASE. Install it with "
        "`pip install ase` into your vibe-qc venv, or "
        "`pip install -e '.[ase]'` from the repo checkout."
    ) from exc

from . import (
    Atom,
    BasisSet,
    Molecule,
    PeriodicSystem,
    monkhorst_pack,
)
from .periodic_gapw_j import (
    run_periodic_rhf_gpw,
    run_periodic_rks_gpw_multi_k,
)
from .kpoints import _integer_counts

_log = logging.getLogger("vibeqc.ase_periodic_gpw")


__all__ = ["VibeqcGPW"]


class VibeqcGPW(Calculator):
    """ASE Calculator wrapping vibe-qc's GPW periodic SCF drivers.

    Parameters (all via kwargs at construction)
    -------------------------------------------
    basis : str
        libint-recognized basis-set name (default ``"sto-3g"``).
    cutoff_ha : float
        Plane-wave grid cutoff in Hartree (default ``300.0``). Sets
        the FFT grid density used for the Hartree-J build.
    functional : str | None
        ``None`` (default) selects Hartree-Fock (Γ-only RHF only).
        A functional name (e.g. ``"lda"``, ``"pbe"``) selects RKS.
    kmesh : list[int] | None
        ``None`` (default) or ``[1, 1, 1]`` selects the Γ-only
        path (:func:`run_periodic_rhf_gpw`). Anything else routes
        through :func:`run_periodic_rks_gpw_multi_k` and requires
        ``functional`` to be a pure-DFT functional.
    """

    implemented_properties = ["energy", "forces", "stress"]

    default_parameters = {
        "basis": "sto-3g",
        "cutoff_ha": 300.0,
        "functional": None,
        "kmesh": None,
        "initial_guess": "AUTO",
        # Central-difference step (bohr) for the numerical-force
        # loop. 0.01 bohr ≈ 5e-3 Å gives well-balanced
        # truncation vs. round-off for STO-3G-scale energies
        # (~1e-9 Ha SCF convergence).
        "fd_step_bohr": 0.01,
        # v0.12 R2: analytic-where-possible per-atom forces. Set
        # True to force the legacy 6N-SCF central-difference path
        # (useful for cross-checking the analytic gradient).
        "use_numerical_forces": False,
        # V_ne convention forwarded from run_periodic_rhf_gpw. The
        # default Ewald convention permits a fully analytic V_ne
        # gradient via nuclear_lattice_gradient_contribution.
        "v_ne_convention": "ewald",
        "smearing_alpha": None,
    }

    nolabel = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # ---- ASE Atoms -> vibe-qc PeriodicSystem ------------------------

    @staticmethod
    def _atoms_to_periodic_system(atoms) -> PeriodicSystem:
        """Convert an ASE :class:`Atoms` to a vibe-qc PeriodicSystem.

        Cell + positions are converted from Å to bohr. Only fully
        3D-periodic Atoms are supported by the GPW drivers.
        """
        if atoms.pbc.sum() != 3:
            raise NotImplementedError(
                "VibeqcGPW: only fully 3D-periodic Atoms (pbc=[True, "
                "True, True]) are supported by the GPW route; got "
                f"pbc.sum() = {int(atoms.pbc.sum())}."
            )
        # ASE rows = lattice vectors a_i; PeriodicSystem columns = a_i
        # (periodic.hpp:32). Transpose at the boundary, else non-orthogonal cells
        # are silently transposed (cubic cells are transpose-invariant and hide it).
        lattice_bohr = np.asarray(atoms.cell.array, dtype=np.float64).T / Bohr
        positions_bohr = atoms.positions / Bohr
        zs = atoms.get_atomic_numbers()

        sys = PeriodicSystem()
        sys.dim = 3
        sys.lattice = lattice_bohr
        sys.unit_cell = [Atom(int(z), list(pos)) for z, pos in zip(zs, positions_bohr)]
        return sys

    # ---- ASE integration -------------------------------------------

    def _run_scf(
        self,
        atoms,
        *,
        initial_density: Optional[np.ndarray] = None,
        source_result=None,
    ):
        """Drive one GPW SCF on ``atoms`` and return the result.

        Internal helper shared between the central single-point and
        the displaced runs used by the numerical-force loop. The
        ``initial_density`` kwarg lets the displaced calls warm-
        start from the converged central density (Γ-only path only;
        the multi-k driver doesn't take an initial_density kwarg).
        Returns the underlying SCF result object.
        """
        kmesh_param = self.parameters["kmesh"]
        if kmesh_param is not None:
            kmesh_param = _integer_counts(kmesh_param, name="GPW kmesh")
        system = self._atoms_to_periodic_system(atoms)

        # The libint BasisSet constructor wants a Molecule; build one
        # from the unit-cell atoms (positions in bohr).
        mol = Molecule(list(system.unit_cell), 0, 1)
        basis_obj = BasisSet(mol, self.parameters["basis"])

        functional: Optional[str] = self.parameters["functional"]
        cutoff_ha: float = float(self.parameters["cutoff_ha"])

        # Decide the route. ``None`` or ``[1, 1, 1]`` -> Γ-only;
        # otherwise multi-k (pure-DFT only).
        is_gamma_only = kmesh_param is None or tuple(kmesh_param) == (
            1,
            1,
            1,
        )

        from .guess import coerce_initial_guess
        initial_guess = coerce_initial_guess(self.parameters["initial_guess"])
        method_label = f"RKS / {functional}" if functional else "RHF"

        if is_gamma_only:
            # Γ-only path: run_periodic_rhf_gpw handles both
            # functional=None (HF) and functional=<name> (RKS).
            gamma_kwargs: dict[str, Any] = {
                "cutoff_ha": cutoff_ha,
                "functional": functional,
                "quiet": True,
                "initial_guess": initial_guess,
            }
            if source_result is not None:
                if initial_density is not None:
                    raise ValueError("supply a source result or a current-basis density")
                from .guess_read import resolve_periodic_read_density_closed
                initial_density = resolve_periodic_read_density_closed(
                    basis=basis_obj, read_from=source_result,
                )
            if initial_density is not None:
                gamma_kwargs["initial_density"] = initial_density
            result = run_periodic_rhf_gpw(
                system,
                basis_obj,
                **gamma_kwargs,
            )
        else:
            # Multi-k: pure-DFT RKS only.
            if functional is None:
                raise NotImplementedError(
                    "VibeqcGPW: multi-k GPW is pure-DFT only; pass a "
                    "functional (e.g. 'lda', 'pbe'). HF on a multi-k "
                    "k-mesh is not wired in the GPW route."
                )
            kmesh_obj = monkhorst_pack(system, list(kmesh_param))
            result = run_periodic_rks_gpw_multi_k(
                system,
                basis_obj,
                kmesh_obj,
                functional=functional,
                cutoff_ha=cutoff_ha,
                initial_guess=initial_guess,
                quiet=True,
            )

        if source_result is not None and is_gamma_only:
            from ._vibeqc_core import GuessSelection, InitialGuess
            source_selection = getattr(source_result, "guess_selection", None)
            if source_selection is not None:
                from dataclasses import replace
                result = replace(result, guess_selection=GuessSelection(
                    source_selection.requested, source_selection.effective, InitialGuess.READ,
                ))
        if not result.converged:
            raise RuntimeError(
                f"VibeqcGPW {method_label} GPW SCF did not converge "
                f"after {result.n_iter} iterations "
                f"(energy = {result.energy} Ha)"
            )
        return result

    def _compute_numerical_forces(self, central_result) -> np.ndarray:
        """Central-difference numerical forces on ``self.atoms``.

        For each atom *i* and Cartesian axis *j* we evaluate the SCF
        energy at ``r_i ± h e_j`` (with ``h = fd_step_bohr`` converted
        to Å) and form ``F[i, j] = -(E(+h) - E(-h)) / (2 h)``. The
        displaced SCFs warm-start from ``central_result.density`` so
        each displaced run converges in only a handful of additional
        iterations.

        Returned units: eV / Å (standard ASE convention).
        """
        h_bohr = float(self.parameters["fd_step_bohr"])
        h_ang = h_bohr * Bohr  # bohr -> Å

        # Gamma warm starts project from the source AO basis. Multi-k
        # displaced bases require a Bloch cross-basis projector; those runs
        # execute the selected physical guess afresh.

        n_atoms = len(self.atoms)
        forces = np.zeros((n_atoms, 3), dtype=np.float64)

        base_positions = self.atoms.get_positions().copy()

        for i in range(n_atoms):
            for j in range(3):
                # +h displacement
                disp_atoms = self.atoms.copy()
                pos = base_positions.copy()
                pos[i, j] += h_ang
                disp_atoms.set_positions(pos)
                res_plus = self._run_scf(
                    disp_atoms,
                    source_result=central_result,
                )
                e_plus = float(res_plus.energy)  # Hartree

                # -h displacement
                disp_atoms = self.atoms.copy()
                pos = base_positions.copy()
                pos[i, j] -= h_ang
                disp_atoms.set_positions(pos)
                res_minus = self._run_scf(
                    disp_atoms,
                    source_result=central_result,
                )
                e_minus = float(res_minus.energy)  # Hartree

                # F = -dE/dr. Step is h_bohr in vibe-qc bohr, so the
                # derivative in Ha/bohr is (e_plus - e_minus) / (2 h_bohr).
                # Convert to eV / Å via Hartree / Bohr.
                dE_dq_ha_per_bohr = (e_plus - e_minus) / (2.0 * h_bohr)
                forces[i, j] = -dE_dq_ha_per_bohr * (Hartree / Bohr)
        return forces

    def _compute_analytic_forces(self, central_result) -> np.ndarray:
        """Analytic-where-possible forces on ``self.atoms``.

        Delegates to :func:`vibeqc.periodic_gapw_gradient.compute_gradient_gpw`,
        which uses the periodic gradient C++ primitives for the kinetic,
        overlap-Pulay, V_ne (Ewald), and E_nn pieces and a fixed-density
        FD on the FFT grid for the Hartree and XC pieces. Returns the
        force ``F = -dE/dR`` in ASE's eV/Å convention.
        """
        from .periodic_gapw_gradient import compute_gradient_gpw

        system = self._atoms_to_periodic_system(self.atoms)
        mol = Molecule(list(system.unit_cell), 0, 1)
        basis_obj = BasisSet(mol, self.parameters["basis"])

        grad_ha_per_bohr = compute_gradient_gpw(
            system,
            basis_obj,
            central_result,
            basis_name=str(self.parameters["basis"]),
            v_ne_convention=str(self.parameters.get("v_ne_convention", "ewald")),
            smearing_alpha=self.parameters.get("smearing_alpha"),
            functional=self.parameters.get("functional"),
            grid=getattr(central_result, "grid", None),
        )
        # F = -dE/dR; convert Ha/bohr -> eV/Å.
        forces = -np.asarray(grad_ha_per_bohr, dtype=np.float64) * (Hartree / Bohr)
        return forces

    def _compute_stress(self, central_result) -> np.ndarray:
        """Compute the periodic stress tensor via finite-difference on
        the fixed-density GPW energy.

        Returns a (3, 3) stress tensor in ASE's eV/Å^3 convention.
        """
        from .periodic_gapw_stress import compute_stress_gpw

        system = self._atoms_to_periodic_system(self.atoms)
        mol = Molecule(list(system.unit_cell), 0, 1)
        basis_obj = BasisSet(mol, self.parameters["basis"])

        stress_ha_per_bohr3 = compute_stress_gpw(
            system,
            basis_obj,
            central_result,
            basis_name=str(self.parameters["basis"]),
            v_ne_convention=str(self.parameters.get("v_ne_convention", "ewald")),
            smearing_alpha=self.parameters.get("smearing_alpha"),
            functional=self.parameters.get("functional"),
            grid=getattr(central_result, "grid", None),
        )
        # Convert Hartree/bohr^3 -> eV/Å^3.
        return np.asarray(stress_ha_per_bohr3, dtype=np.float64) * (Hartree / Bohr**3)

    def calculate(  # type: ignore[override]
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        # Pre-flight: open-shell rejection. The GPW SCF drivers
        # would raise on odd electron counts anyway; doing it here
        # gives a clearer error pointing at the wrapper.
        zs = self.atoms.get_atomic_numbers()
        n_elec = int(sum(int(z) for z in zs))
        if n_elec % 2 != 0:
            raise ValueError(
                f"VibeqcGPW: cell has {n_elec} electrons (odd); the "
                "GPW route is closed-shell only. UHF / UKS GPW "
                "support is post-M3 work."
            )

        functional: Optional[str] = self.parameters["functional"]
        kmesh_param: Optional[Sequence[int]] = self.parameters["kmesh"]
        if kmesh_param is not None:
            kmesh_param = _integer_counts(kmesh_param, name="GPW kmesh")
        cutoff_ha: float = float(self.parameters["cutoff_ha"])
        is_gamma_only = kmesh_param is None or tuple(kmesh_param) == (
            1,
            1,
            1,
        )
        from .guess import coerce_initial_guess
        initial_guess = coerce_initial_guess(self.parameters["initial_guess"])
        method_label = f"RKS / {functional}" if functional else "RHF"
        _log.info(
            "GPW %s / %s  n_atoms=%d  n_electrons=%d  cutoff_ha=%.1f  kmesh=%s",
            method_label,
            self.parameters["basis"],
            len(self.atoms),
            n_elec,
            cutoff_ha,
            "gamma-only" if is_gamma_only else list(kmesh_param),
        )

        # Central single-point SCF.
        result = self._run_scf(self.atoms)

        self.results["energy"] = float(result.energy) * Hartree
        self.results["free_energy"] = self.results["energy"]
        # Expose the underlying result for downstream inspection.
        self.results["scf_result"] = result

        if "forces" in properties:
            use_numerical = bool(self.parameters.get("use_numerical_forces", False))
            # The analytic gradient is only wired for the Γ-only GPW
            # path. Multi-k falls back to the numerical loop.
            if use_numerical or not is_gamma_only:
                self.results["forces"] = self._compute_numerical_forces(
                    result,
                )
            else:
                self.results["forces"] = self._compute_analytic_forces(
                    result,
                )

        if "stress" in properties:
            self.results["stress"] = self._compute_stress(result)
