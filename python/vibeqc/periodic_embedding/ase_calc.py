"""ASE Calculator for Green's-function embedded surfaces.

Provides an :class:`ase.calculators.calculator.Calculator` that wraps
:func:`~vibeqc.periodic_embedding.runner.run_embedded_surface` so that
ASE can drive geometry relaxations, vibrational analyses, and workflow
integration.

The calculator is intentionally minimal in this first landing:
* Single-point energy (band + nuclear).
* Central-difference **numerical forces** (``F = -dE/dR``).  These are
  the gradient of whatever energy expression the calculator returns;
  as the energy model matures (full Ishida two-step SCF, double-counting
  corrections) the forces follow automatically.  Analytical lattice-
  integral + S_emb gradients are future work -- see
  ``handovers/HANDOVER_GF_EMBEDDING.md`` remaining-work item 1.
* Gated as experimental until validated against thick-slab GDF limit.

Unit conventions: ASE works in Å + eV, vibe-qc in bohr + Hartree.
Conversions use ASE's own constants (:attr:`ase.units.Bohr`,
:attr:`ase.units.Hartree`).

Usage::

    from ase import Atoms
    from vibeqc.periodic_embedding.ase_calc import EmbeddedSurfaceCalculator

    atoms = ...  # ASE Atoms with tags array
    calc = EmbeddedSurfaceCalculator(
        basis_name="sto-3g",
        surface_k_mesh=(2, 2),
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()      # eV
    forces = atoms.get_forces()                # eV / Å
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .._vibeqc_core import (
    Atom,
    BasisSet,
    PeriodicSystem,
    nuclear_repulsion_per_cell,
)
from .._vibeqc_core import LatticeSumOptions as _CoreLatOpts
from .region import TAG_REGION_I, TAG_SUBSTRATE, RegionPartition
from .runner import EmbeddedSurfaceResult, run_embedded_surface

# Guard the ASE import -- vibe-qc core does not depend on ASE.
try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Bohr, Hartree
except ImportError:
    Calculator = object
    all_changes = []
    # Fallback constants so the module still imports without ASE
    # (the class is then a stub and the tests skip).  CODATA 2018.
    Bohr = 0.52917721090380  # Å per bohr
    Hartree = 27.211386245988  # eV per Hartree


class EmbeddedSurfaceCalculator(Calculator):
    """ASE Calculator for embedded-surface calculations.

    Wraps :func:`run_embedded_surface` to provide ``get_potential_energy()``
    and ``get_forces()``.  Forces are central-difference numerical
    gradients of the returned energy (``F = -dE/dR``); the analytical
    lattice-integral + S_emb gradient is future work.

    Parameters
    ----------
    basis_name
        Orbital basis set name (e.g. ``"sto-3g"``, ``"6-31g*"``).
    surface_k_mesh
        In-plane Monkhorst-Pack subdivisions ``(nx, ny)``.
    contour_n_nodes
        Number of contour nodes.
    e_fermi, e_bottom
        Fermi level and band bottom in Ha.  If None, auto-estimated.
        For geometry optimisation, passing explicit values is strongly
        recommended: the numerical-force loop then holds the contour
        window fixed across displacements (see below).
    lat_cutoff_bohr
        Lattice-sum cutoff in bohr.
    fd_step_bohr
        Central-difference step (bohr) for the numerical-force loop.
        0.01 bohr balances truncation vs. round-off for STO-3G-scale
        energies.
    force_atom_indices
        Restrict the numerical-force loop to these atom indices (others
        get zero force).  ``None`` (default) differentiates every atom.
        For a slab, pass the mobile atoms (adsorbate + relaxed surface
        shell) -- the frozen substrate then costs no displaced runs.
    tags
        Per-atom tags for region partition.  If None, the tags are read
        from ``atoms.get_tags()``.  Default convention: 0 = substrate,
        1 = region I.
    charge
        Total charge per unit cell (default 0).
    multiplicity
        Spin multiplicity ``2S + 1`` of the unit cell (default 1,
        closed-shell singlet).  An open-shell cell (odd electron count,
        or a requested non-singlet) needs ``multiplicity > 1`` **and**
        a spin-polarised density method -- pass ``scf_method="uhf"``
        (forwarded to :func:`run_embedded_surface`).  ``charge`` and
        ``multiplicity`` must satisfy the usual parity rule: the cell's
        electron count and ``multiplicity`` must share parity, else the
        :class:`PeriodicSystem` constructor raises before any embedding
        work runs.
    kwargs
        Additional keyword arguments forwarded to
        :func:`run_embedded_surface` (e.g. ``scf_method="uhf"`` for the
        spin-polarised path).
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        *,
        basis_name: str = "sto-3g",
        surface_k_mesh: Tuple[int, int] = (2, 2),
        contour_n_nodes: int = 40,
        e_fermi: Optional[float] = None,
        e_bottom: Optional[float] = None,
        lat_cutoff_bohr: float = 15.0,
        fd_step_bohr: float = 0.01,
        force_atom_indices: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[int]] = None,
        charge: int = 0,
        multiplicity: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.basis_name = basis_name
        self.surface_k_mesh = surface_k_mesh
        self.contour_n_nodes = contour_n_nodes
        self.e_fermi = e_fermi
        self.e_bottom = e_bottom
        self.lat_cutoff_bohr = lat_cutoff_bohr
        self.fd_step_bohr = fd_step_bohr
        self.force_atom_indices = (
            None if force_atom_indices is None else list(force_atom_indices)
        )
        self._tags = tags
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self._kwargs = kwargs
        self._last_result: Optional[EmbeddedSurfaceResult] = None

    def _atoms_to_system(self, atoms) -> PeriodicSystem:
        """Convert an ASE Atoms object to a PeriodicSystem."""
        positions_ang = atoms.get_positions()  # Å
        # ASE rows = lattice vectors a_i; PeriodicSystem columns = a_i
        # (cpp/include/vibeqc/periodic.hpp:32).  Transpose at the
        # boundary, else non-orthogonal cells are silently transposed
        # (cubic cells are transpose-invariant and hide it).
        cell_rows_bohr = np.array(atoms.get_cell()) / Bohr  # rows = a_i

        positions_bohr = positions_ang / Bohr

        unit_cell = [
            Atom(int(z), pos.tolist())
            for z, pos in zip(atoms.get_atomic_numbers(), positions_bohr)
        ]

        # Determine dimensionality: if the 3rd lattice vector is much
        # larger than the in-plane ones, it's dim=2 (slab), else dim=3.
        # Simple heuristic: dim=2 if the cell has a long c-axis.  Measure
        # the row norms (each row is a lattice vector) before transposing.
        a_len = np.linalg.norm(cell_rows_bohr[0])
        c_len = np.linalg.norm(cell_rows_bohr[2])
        dim = 2 if c_len > 3 * a_len else 3

        return PeriodicSystem(
            dim=dim,
            lattice=cell_rows_bohr.T,  # columns = lattice vectors
            unit_cell=unit_cell,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )

    def _get_tags(self, atoms) -> Sequence[int]:
        if self._tags is not None:
            return self._tags
        raw = atoms.get_tags()
        if raw is not None and len(raw) == len(atoms):
            return [int(t) for t in raw]
        # Default: all atoms are region I (tag=1).
        return [TAG_REGION_I] * len(atoms)

    def _lat_opts(self):
        from .._vibeqc_core import LatticeSumOptions

        opts = LatticeSumOptions()
        opts.cutoff_bohr = self.lat_cutoff_bohr
        opts.nuclear_cutoff_bohr = self.lat_cutoff_bohr
        return opts

    def _compute_energy(
        self,
        atoms,
        *,
        e_fermi: Optional[float] = None,
        e_bottom: Optional[float] = None,
    ) -> Tuple[float, EmbeddedSurfaceResult]:
        """Embedded-surface total energy for ``atoms``, in **Hartree**.

        Returns ``(energy_ha, result)``.  Shared by the single-point
        path (:meth:`calculate`) and the finite-difference force loop
        (:meth:`_compute_numerical_forces`).

        ``e_fermi`` / ``e_bottom`` override the calculator defaults so
        the force loop can hold the complex-energy contour window fixed
        across displacements.

        The energy is ``band_energy + nuclear_repulsion``.  The band
        term is the contour band-structure energy of region I; the
        nuclear term is the full-cell repulsion.  This is the same
        placeholder total used by the single-point path -- double-
        counting / electrostatic corrections arrive with the full
        Ishida two-step SCF.  The numerical forces below are the exact
        gradient of *this* expression, so they track it as it matures.
        """
        system = self._atoms_to_system(atoms)
        mol = system.unit_cell_molecule()
        basis = BasisSet(mol, self.basis_name)
        tags = self._get_tags(atoms)

        ef = e_fermi if e_fermi is not None else self.e_fermi
        eb = e_bottom if e_bottom is not None else self.e_bottom

        result = run_embedded_surface(
            system,
            basis,
            tags,
            surface_k_mesh=self.surface_k_mesh,
            contour_n_nodes=self.contour_n_nodes,
            e_fermi=ef,
            e_bottom=eb,
            lat_opts=self._lat_opts(),
            **self._kwargs,
        )
        e_nuc = nuclear_repulsion_per_cell(system, self._lat_opts())
        energy_ha = result.band_energy + e_nuc
        return energy_ha, result

    def _compute_numerical_forces(
        self, central_result: EmbeddedSurfaceResult
    ) -> np.ndarray:
        """Central-difference numerical forces on ``self.atoms``.

        For each atom *i* and Cartesian axis *j* we evaluate the
        embedded energy at ``r_i ± h e_j`` (``h = fd_step_bohr``) and
        form ``F[i, j] = -(E(+h) - E(-h)) / (2 h)`` in ASE's eV/Å
        convention.

        The contour window (``e_fermi``, ``e_bottom``) is **held fixed**
        to the central geometry's values across all displaced runs.
        Physically this is the force at fixed Fermi level -- the
        semi-infinite substrate is an electron reservoir -- and it also
        removes the spurious noise that an auto-re-estimated window
        would inject into the finite difference.

        Atoms outside ``force_atom_indices`` (when set) keep zero force,
        so a slab calculation only pays for its mobile atoms.
        """
        h_bohr = float(self.fd_step_bohr)
        h_ang = h_bohr * Bohr  # bohr -> Å

        # Hold the contour edges fixed to the central run.
        ef = central_result.e_fermi
        eb = central_result.e_bottom

        n_atoms = len(self.atoms)
        forces = np.zeros((n_atoms, 3), dtype=np.float64)

        if self.force_atom_indices is None:
            active = range(n_atoms)
        else:
            active = self.force_atom_indices

        base_positions = self.atoms.get_positions().copy()

        for i in active:
            for j in range(3):
                # +h displacement
                disp_atoms = self.atoms.copy()
                pos = base_positions.copy()
                pos[i, j] += h_ang
                disp_atoms.set_positions(pos)
                e_plus, _ = self._compute_energy(
                    disp_atoms, e_fermi=ef, e_bottom=eb
                )

                # -h displacement
                disp_atoms = self.atoms.copy()
                pos = base_positions.copy()
                pos[i, j] -= h_ang
                disp_atoms.set_positions(pos)
                e_minus, _ = self._compute_energy(
                    disp_atoms, e_fermi=ef, e_bottom=eb
                )

                # F = -dE/dr.  Step is h_bohr in bohr, so the derivative
                # in Ha/bohr is (e_plus - e_minus) / (2 h_bohr).
                dE_dq_ha_per_bohr = (e_plus - e_minus) / (2.0 * h_bohr)
                forces[i, j] = -dE_dq_ha_per_bohr * (Hartree / Bohr)
        return forces

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=None,
    ):
        if properties is None:
            properties = ["energy"]
        super().calculate(atoms, properties, system_changes)

        energy_ha, result = self._compute_energy(self.atoms)
        self._last_result = result

        # vibe-qc works in Hartree; ASE expects eV.
        self.results["energy"] = energy_ha * Hartree
        self.results["free_energy"] = self.results["energy"]

        if "forces" in properties:
            self.results["forces"] = self._compute_numerical_forces(result)

    @property
    def last_result(self) -> Optional[EmbeddedSurfaceResult]:
        """The most recent :class:`EmbeddedSurfaceResult`, or None."""
        return self._last_result
