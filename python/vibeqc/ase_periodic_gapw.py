"""ASE Calculator wrapper for the periodic GAPW (all-electron) route.

The :class:`VibeqcGAPW` calculator drives vibe-qc's GAPW periodic
SCF (`run_periodic_rhf_gapw` for Γ-only RHF / RKS, and
`run_periodic_rks_gapw_multi_k` for multi-k pure-DFT RKS) from the
standard ASE Calculator API. This is the all-electron counterpart to
:class:`vibeqc.ase_periodic_gpw.VibeqcGPW` and matches GPAW's user-facing
shape::

    from ase.build import bulk
    from vibeqc.ase_periodic_gapw import VibeqcGAPW

    atoms = bulk(...)          # any 3D periodic ase.Atoms
    atoms.calc = VibeqcGAPW(basis='sto-3g', functional='lda',
                             cutoff_ha=300.0, kmesh=[2, 2, 2],
                             gapw_kwargs=dict(lmax=3, soft_cutoff=3.0))
    print(atoms.get_potential_energy())   # eV

Scope (v0.12, GAPW M3c-aligned)

* Closed-shell only -- the underlying GAPW drivers require an even
  electron count. Open-shell atoms raise :class:`ValueError` from
  the SCF driver; the wrapper additionally pre-checks before
  building basis sets and gives a clear error.
* Forces are computed via the analytic GAPW gradient by default
  (:func:`vibeqc.periodic_gapw_gradient.compute_gradient_gapw`).
  The legacy central-difference numerical-force path is available
  via ``use_numerical_forces=True``.
* Multi-k requires a pure functional (no HF exchange); hybrids on
  the multi-k path raise ``NotImplementedError`` from the C++
  driver, propagated to the caller.
* ``functional=None`` selects Hartree-Fock (Γ-only RHF). Setting
  both ``functional=None`` and ``kmesh=[a, b, c]`` with a, b, c
  > 1 is not supported -- multi-k GAPW is pure-DFT only (see
  :func:`vibeqc.run_periodic_rks_gapw_multi_k`).

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
        "vibeqc.ase_periodic_gapw requires ASE. Install it with "
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
from .periodic_gapw_augment import (
    run_periodic_rhf_gapw,
    run_periodic_rks_gapw,
    run_periodic_rks_gapw_multi_k,
)

_log = logging.getLogger("vibeqc.ase_periodic_gapw")


__all__ = ["VibeqcGAPW"]


class VibeqcGAPW(Calculator):
    """ASE Calculator wrapping vibe-qc's GAPW periodic SCF drivers.

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
        path (:func:`run_periodic_rhf_gapw`). Anything else routes
        through :func:`run_periodic_rks_gapw_multi_k` and requires
        ``functional`` to be a pure-DFT functional.
    gapw_kwargs : dict | None
        Extra keyword arguments forwarded to the GAPW J builder.
        Supported keys: ``lmax`` (default 3), ``soft_cutoff``
        (default 3.0), ``n_radial`` (default 80), ``lebedev_order``
        (default 17), ``one_centre``, and ``molecular_limit``. The grid controls
        apply to both routes. ``one_centre`` and ``molecular_limit`` are
        Gamma-only scientific controls and are rejected for multi-k runs. HF
        with the method-aware one-centre default requires
        ``molecular_limit=True``; compact crystals should use a periodic
        GDF/BIPOLE calculator.
    use_gapw_forces : bool
        When ``True`` (default), use the analytic GAPW gradient for the
        Gamma block functional. Set ``False`` to use central differences.
        Multi-k and fit-free analytic one-centre results always use central
        differences of their own SCF energy.
    """

    implemented_properties = ["energy", "forces"]

    default_parameters = {
        "basis": "sto-3g",
        "cutoff_ha": 300.0,
        "functional": None,
        "kmesh": None,
        # GAPW augmentation parameters.
        "gapw_kwargs": None,
        # When True (default), use the analytic GAPW gradient for
        # forces. When False, fall back to central-difference FD
        # on the SCF energy (same as the GPW Calculator).
        "use_gapw_forces": True,
        # Central-difference step (bohr) for the numerical-force
        # loop (only used when use_gapw_forces=False).
        "fd_step_bohr": 0.01,
        # V_ne convention forwarded to the SCF driver and gradient.
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
        3D-periodic Atoms are supported by the GAPW drivers.
        """
        if atoms.pbc.sum() != 3:
            raise NotImplementedError(
                "VibeqcGAPW: only fully 3D-periodic Atoms (pbc=[True, "
                "True, True]) are supported by the GAPW route; got "
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
    ):
        """Drive one GAPW SCF on ``atoms`` and return the result.

        Internal helper shared between the central single-point and
        the displaced runs used by the numerical-force loop. The
        ``initial_density`` kwarg lets the displaced calls warm-
        start from the converged central density (Γ-only path only;
        the multi-k driver doesn't take an initial_density kwarg).
        Returns the underlying SCF result object
        (:class:`GapwScfResult` or :class:`GpwMultiKScfResult`).
        """
        system = self._atoms_to_periodic_system(atoms)

        # The libint BasisSet constructor wants a Molecule; build one
        # from the unit-cell atoms (positions in bohr).
        mol = Molecule(list(system.unit_cell), 0, 1)
        basis_obj = BasisSet(mol, self.parameters["basis"])

        functional: Optional[str] = self.parameters["functional"]
        kmesh_param: Optional[Sequence[int]] = self.parameters["kmesh"]
        cutoff_ha: float = float(self.parameters["cutoff_ha"])

        # Extract GAPW-specific kwargs.
        gapw_kwargs: dict[str, Any] = dict(self.parameters.get("gapw_kwargs") or {})
        gapw_kwargs.setdefault("lmax", 3)
        gapw_kwargs.setdefault("soft_cutoff", 3.0)
        gapw_kwargs.setdefault("n_radial", 80)
        gapw_kwargs.setdefault("lebedev_order", 17)

        # Decide the route. ``None`` or ``[1, 1, 1]`` -> Γ-only;
        # otherwise multi-k (pure-DFT only).
        is_gamma_only = kmesh_param is None or tuple(int(k) for k in kmesh_param) == (
            1,
            1,
            1,
        )

        method_label = f"RKS / {functional}" if functional else "RHF"

        if is_gamma_only:
            # Γ-only path: run_periodic_rhf_gapw handles both
            # functional=None (HF) and functional=<name> (RKS).
            gamma_kwargs: dict[str, Any] = {
                "cutoff_ha": cutoff_ha,
                "functional": functional,
                "quiet": True,
                "lmax": gapw_kwargs["lmax"],
                "soft_cutoff": gapw_kwargs["soft_cutoff"],
                "n_radial": gapw_kwargs["n_radial"],
                "lebedev_order": gapw_kwargs["lebedev_order"],
            }
            if "one_centre" in gapw_kwargs:
                gamma_kwargs["one_centre"] = gapw_kwargs["one_centre"]
            if "molecular_limit" in gapw_kwargs:
                gamma_kwargs["molecular_limit"] = gapw_kwargs[
                    "molecular_limit"
                ]
            if initial_density is not None:
                gamma_kwargs["initial_density"] = initial_density

            # The GAPW SCF is a superset of the GPW route -- even
            # when functional=None (RHF), the GAPW driver is the
            # correct entry point.
            result = run_periodic_rhf_gapw(
                system,
                basis_obj,
                **gamma_kwargs,
            )
        else:
            # Multi-k: pure-DFT RKS only.
            unsupported_scientific_controls = sorted(
                {"one_centre", "molecular_limit"}.intersection(gapw_kwargs)
            )
            if unsupported_scientific_controls:
                controls = ", ".join(unsupported_scientific_controls)
                raise NotImplementedError(
                    "VibeqcGAPW: one_centre and molecular_limit are "
                    "Gamma-only scientific controls and cannot be used on "
                    f"the multi-k GAPW route (got {controls})."
                )
            if functional is None:
                raise NotImplementedError(
                    "VibeqcGAPW: multi-k GAPW is pure-DFT only; pass a "
                    "functional (e.g. 'lda', 'pbe'). HF on a multi-k "
                    "k-mesh is not wired in the GAPW route."
                )
            kmesh_obj = monkhorst_pack(system, list(kmesh_param))
            result = run_periodic_rks_gapw_multi_k(
                system,
                basis_obj,
                kmesh_obj,
                functional=functional,
                cutoff_ha=cutoff_ha,
                quiet=True,
                lmax=gapw_kwargs["lmax"],
                soft_cutoff=gapw_kwargs["soft_cutoff"],
                n_radial=gapw_kwargs["n_radial"],
                lebedev_order=gapw_kwargs["lebedev_order"],
            )

        if not result.converged:
            raise RuntimeError(
                f"VibeqcGAPW {method_label} GAPW SCF did not converge "
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

        # Warm-start density (Γ-only path provides one; multi-k
        # falls back to the driver's Hcore guess).
        initial_density = getattr(central_result, "density", None)

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
                    initial_density=initial_density,
                )
                e_plus = float(res_plus.energy)  # Hartree

                # -h displacement
                disp_atoms = self.atoms.copy()
                pos = base_positions.copy()
                pos[i, j] -= h_ang
                disp_atoms.set_positions(pos)
                res_minus = self._run_scf(
                    disp_atoms,
                    initial_density=initial_density,
                )
                e_minus = float(res_minus.energy)  # Hartree

                # F = -dE/dr. Step is h_bohr in vibe-qc bohr, so the
                # derivative in Ha/bohr is (e_plus - e_minus) / (2 h_bohr).
                # Convert to eV / Å via Hartree / Bohr.
                dE_dq_ha_per_bohr = (e_plus - e_minus) / (2.0 * h_bohr)
                forces[i, j] = -dE_dq_ha_per_bohr * (Hartree / Bohr)
        return forces

    def _compute_gapw_forces(self, central_result) -> np.ndarray:
        """GAPW analytic forces on ``self.atoms``.

        Delegates to :func:`vibeqc.periodic_gapw_gradient.compute_gradient_gapw`,
        which uses the GAPW gradient with per-atom augmentation correction
        for all-electron accuracy. Returns the force ``F = -dE/dR`` in
        ASE's eV/Å convention.
        """
        from .periodic_gapw_gradient import compute_gradient_gapw

        system = self._atoms_to_periodic_system(self.atoms)
        mol = Molecule(list(system.unit_cell), 0, 1)
        basis_obj = BasisSet(mol, self.parameters["basis"])

        gapw_kwargs: dict[str, Any] = dict(self.parameters.get("gapw_kwargs") or {})

        grad_ha_per_bohr = compute_gradient_gapw(
            system,
            basis_obj,
            central_result,
            basis_name=str(self.parameters["basis"]),
            v_ne_convention=str(self.parameters.get("v_ne_convention", "ewald")),
            smearing_alpha=self.parameters.get("smearing_alpha"),
            functional=self.parameters.get("functional"),
            grid=getattr(central_result, "grid", None),
            gapw_kwargs=gapw_kwargs,
        )
        # F = -dE/dR; convert Ha/bohr -> eV/Å.
        forces = -np.asarray(grad_ha_per_bohr, dtype=np.float64) * (Hartree / Bohr)
        return forces

    def calculate(  # type: ignore[override]
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        # Pre-flight: open-shell rejection. The GAPW SCF drivers
        # would raise on odd electron counts anyway; doing it here
        # gives a clearer error pointing at the wrapper.
        zs = self.atoms.get_atomic_numbers()
        n_elec = int(sum(int(z) for z in zs))
        if n_elec % 2 != 0:
            raise ValueError(
                f"VibeqcGAPW: cell has {n_elec} electrons (odd); the "
                "GAPW route is closed-shell only. UHF / UKS GAPW "
                "support is post-M3c work."
            )

        functional: Optional[str] = self.parameters["functional"]
        kmesh_param: Optional[Sequence[int]] = self.parameters["kmesh"]
        cutoff_ha: float = float(self.parameters["cutoff_ha"])
        is_gamma_only = kmesh_param is None or tuple(int(k) for k in kmesh_param) == (
            1,
            1,
            1,
        )
        method_label = f"RKS / {functional}" if functional else "RHF"
        _log.info(
            "GAPW %s / %s  n_atoms=%d  n_electrons=%d  cutoff_ha=%.1f  kmesh=%s",
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
            use_gapw_forces = bool(self.parameters.get("use_gapw_forces", True))
            one_centre = str(getattr(result, "one_centre", "block"))
            # The GAPW analytic gradient is wired for the Gamma block path.
            # Multi-k and the fit-free analytic-ERI Hartree mode use exact
            # central differences of their own SCF energy instead.
            if (
                not is_gamma_only
                or not use_gapw_forces
                or one_centre != "block"
            ):
                self.results["forces"] = self._compute_numerical_forces(
                    result,
                )
            else:
                self.results["forces"] = self._compute_gapw_forces(
                    result,
                )
