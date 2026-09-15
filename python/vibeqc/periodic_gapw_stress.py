"""Analytic stress tensor for the GAPW / GPW periodic SCF path.

The stress tensor s_ab is the derivative of the total energy with
respect to lattice strain e_ab:

    s_ab = (1/V) . dE / de_ab

Following the Nielsen & Martin formalism (Phys. Rev. B 32, 3780 (1985)),
we decompose into:

1. **Hellmann-Feynman stress** -- finite-difference on the total energy
   with respect to cell deformations at fixed atomic positions (in
   fractional coordinates) and fixed density.

2. **Pulay stress** -- from the basis-set dependence on the cell vectors
   (the dchi/dh terms where h is the cell dimension). This is the
   cell-response analogue of the Pulay force correction.

For GPW / GAPW, the same gauge-consistency argument applies as for the
gradient: all Ewald / FFT-Poisson pieces must be deformed together to
preserve the gauge cancellation.

Implementation
--------------

The Hellmann-Feynman piece is computed via central-difference on the
full GAPW energy at fixed fractional coordinates and fixed density,
applying a small strain d to each independent component of the lattice
matrix. The Pulay stress uses the energy-weighted density W and the
cell-response of the overlap matrix (dS/dh).

Returns
-------
Stress tensor in Hartree/bohr^3 (same units as internal pressure, which
is -s_ab). The ASE calculator convention expects eV/Å^3; multiply by
``(Hartree / Bohr**3) -> (eV / Å**3)`` conversion factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from . import _vibeqc_core as _core
from ._vibeqc_core import (
    BasisSet,
    LatticeSumOptions,
    PeriodicSystem,
)
from .periodic_gapw_grid import PlaneWaveGrid, make_grid
from .periodic_gapw_j import (
    collocate_density_on_grid,
    evaluate_gpw_energy,
)

__all__ = [
    "compute_stress_gpw",
    "compute_stress_gapw",
    "STRESS_CITATION_KEYS",
    "method_citations",
]


# Citation keys for the periodic stress tensor, resolved against
# python/vibeqc/output/citations/database.toml. The stress is an offline
# post-SCF property that returns a bare (3, 3) array, not an SCF-job feature,
# so it has no `[routes.*]` entry; the module surfaces its provenance
# explicitly the way vibeqc.basis_optimization.bdiis does (a run that reports a
# stress should cite the method the same way a paper's Methods section does).
# Nielsen-Martin 1985 is the Hellmann-Feynman + Pulay decomposition the
# finite-difference implementation follows (see the module docstring). The
# resolve-in-database guard lives in tests/test_periodic_stress_citation.py.
STRESS_CITATION_KEYS: tuple[str, ...] = ("nielsen_martin_1985",)


def method_citations() -> tuple[str, ...]:
    """Citation-database keys for the periodic stress tensor."""
    return STRESS_CITATION_KEYS


def _strain_cell(
    lattice: np.ndarray,
    strain: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Apply a small strain ``eps`` to component ``(i, j)`` of the lattice.

    The strain is applied symmetrically: ``L -> (I + e) . L`` where e
    has only component (i, j) set.

    Parameters
    ----------
    lattice
        ``(3, 3)`` lattice vectors (columns = a₁, a₂, a₃) in bohr.
    strain
        ``(3, 3)`` strain direction matrix (typically a one-hot matrix).
    eps
        Strain magnitude (dimensionless).

    Returns
    -------
    strained_lattice
        ``(3, 3)`` strained lattice matrix.
    """
    strain_tensor = np.eye(3) + eps * strain
    return strain_tensor @ lattice


def _stress_fd(
    system,
    basis: BasisSet,
    basis_name: str,
    D: np.ndarray,
    *,
    grid: PlaneWaveGrid,
    v_ne_convention: str,
    smearing_alpha: Optional[float],
    functional: Optional[str],
    eps: float = 1e-4,
    use_gapw: bool = False,
    gapw_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Central-difference stress via fixed-density energy evaluations.

    For each independent component (i, j) with i <= j of the 3x3 symmetric
    strain tensor, applies ±e to that component and evaluates the
    fixed-density total energy. The stress is

        s_ij = (1/V) . (E(+e_ij) - E(-e_ij)) / (4 . e)   (i != j)
        s_ii = (1/V) . (E(+e_ii) - E(-e_ii)) / (2 . e)   (diagonal)

    The factor 4 for off-diagonal comes from applying the strain
    symmetrically: e_ij = e_ji = e, so the total strain energy change
    is 2x the individual component, and the FD divides by 2e, giving
    an overall factor of 1/(4e).

    Returns
    -------
    stress : (3, 3) ndarray in Hartree/bohr^3.
    """
    from .molecule import Molecule

    lattice = np.asarray(system.lattice, dtype=float)
    V_cell = abs(np.linalg.det(lattice))

    stress = np.zeros((3, 3), dtype=float)
    h = float(eps)

    # Pre-compute fractional coordinates.
    frac_coords = []
    Z_list = []
    for atom in system.unit_cell:
        frac = np.linalg.solve(lattice, np.asarray(atom.xyz))
        frac_coords.append(frac)
        Z_list.append(int(atom.Z))

    for i in range(3):
        for j in range(i, 3):
            direction = np.zeros((3, 3), dtype=float)
            direction[i, j] = 1.0
            if i != j:
                direction[j, i] = 1.0  # symmetric strain

            def _energy_at_strain(sign: float):
                lat_s = _strain_cell(lattice, direction, sign * h)
                atoms_s = [
                    _core.Atom(Z, lat_s @ frac) for Z, frac in zip(Z_list, frac_coords)
                ]
                sys_s = PeriodicSystem(dim=3, lattice=lat_s, unit_cell=atoms_s)
                mol_s = Molecule(list(atoms_s), 0, 1)
                basis_s = BasisSet(mol_s, basis_name)
                # Use the same grid dimensions as the reference cell so
                # the energy difference comes purely from strain, not
                # from a grid-resolution change.
                grid_s = PlaneWaveGrid(lat_s, grid.nx, grid.ny, grid.nz)

                # One-electron pieces.
                from .periodic_gapw_j import (
                    _build_v_ne,
                    _kinetic_lattice_gamma,
                )

                T_s = _kinetic_lattice_gamma(basis_s, sys_s)
                V_ne_s = _build_v_ne(
                    basis_s,
                    sys_s,
                    v_ne_convention,
                    smearing_alpha,
                    grid_s,
                )
                E_nn_s = float(
                    _core.ewald_nuclear_repulsion(sys_s, _core.EwaldOptions())
                )

                # Two-electron.
                if use_gapw:
                    from .periodic_gapw_augment import GapwJBuilder

                    gkw = dict(gapw_kwargs or {})
                    builder_s = GapwJBuilder(
                        basis_s,
                        sys_s,
                        grid_s,
                        lmax=gkw.get("lmax", 3),
                        soft_cutoff=gkw.get("soft_cutoff", 3.0),
                        n_radial=gkw.get("n_radial", 80),
                        quiet=True,
                    )
                    J_s = builder_s.build_J(D)
                    # XC augmentation energy.
                    func_obj = None
                    if functional is not None:
                        func_obj = _core.Functional(functional, 1)
                    _h_aug, xc_aug = builder_s._aug.compute_augmentation_energy(
                        D, functional=func_obj
                    )
                else:
                    from .periodic_gapw_j import GpwJBuilder

                    J_s = GpwJBuilder(basis_s, grid_s).build_J(D)
                    xc_aug = 0.0

                # Exchange (Γ-only, direct AO-image sum).
                lo = LatticeSumOptions()
                lo.cutoff_bohr = 25.0
                jk_s = _core.build_jk_gamma_molecular_limit(basis_s, sys_s, lo, D)
                K_s = np.asarray(jk_s.K)

                if functional is not None:
                    func_s = _core.Functional(functional, 1)
                    ex_frac = float(func_s.hf_exchange_fraction)
                else:
                    ex_frac = 1.0

                e_kin = float(np.einsum("ij,ij->", D, T_s))
                e_ne = float(np.einsum("ij,ij->", D, V_ne_s))
                e_hfx = -0.25 * ex_frac * float(np.einsum("ij,ij->", D, K_s))
                e_hart = 0.5 * float(np.einsum("ij,ij->", D, J_s))
                e_xc_grid = 0.0
                if functional is not None:
                    from .periodic_gapw_j import (
                        _evaluate_xc_on_grid,
                        collocate_density_on_grid,
                    )

                    rho_s = collocate_density_on_grid(basis_s, D, grid_s)
                    e_xc_grid, _, _, _, _ = _evaluate_xc_on_grid(rho_s, grid_s, func_s)
                return e_kin + e_ne + e_hart + e_hfx + e_xc_grid + xc_aug + E_nn_s

            e_plus = _energy_at_strain(+1)
            e_minus = _energy_at_strain(-1)

            # Diag: factor 2.h; off-diag: factor 4.h (symmetric strain
            # doubles the energy change).
            denom = (4.0 * h) if i != j else (2.0 * h)
            stress_ij = (e_plus - e_minus) / denom / V_cell
            stress[i, j] = stress_ij
            if i != j:
                stress[j, i] = stress_ij

    return stress


def compute_stress_gpw(
    system: PeriodicSystem,
    basis: BasisSet,
    result: Any,
    *,
    basis_name: str,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    functional: Optional[str] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute the GPW periodic stress tensor via finite difference.

    Parameters
    ----------
    system, basis, result
        The converged periodic system + basis + GPW SCF result.
    basis_name
        Basis-set name for rebuilding on strained cells.
    v_ne_convention, smearing_alpha, functional
        Same kwargs the SCF was driven with.
    grid
        Pre-built :class:`PlaneWaveGrid`. Defaults to result.grid.
    cutoff_ha
        Fallback cutoff.
    eps
        Strain magnitude for central difference.

    Returns
    -------
    stress
        ``(3, 3)`` stress tensor in Hartree/bohr^3.
    """
    if not result.converged:
        raise ValueError("compute_stress_gpw: result is not converged.")

    D = np.asarray(result.density, dtype=float)
    if grid is None:
        grid = getattr(result, "grid", None)
    if grid is None:
        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = make_grid(
            np.asarray(system.lattice, dtype=float),
            cutoff_ha=cutoff_ha,
        )

    return _stress_fd(
        system,
        basis,
        basis_name,
        D,
        grid=grid,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        functional=functional,
        eps=eps,
        use_gapw=False,
    )


def compute_stress_gapw(
    system: PeriodicSystem,
    basis: BasisSet,
    result: Any,
    *,
    basis_name: str,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    functional: Optional[str] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    eps: float = 1e-4,
    gapw_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Compute the GAPW periodic stress tensor via finite difference.

    Uses the same finite-strain approach as :func:`compute_stress_gpw`
    but evaluates the GAPW (augmented) energy under strain. The
    augmentation correction responds to cell deformation through the
    per-atom radial grids.

    Parameters
    ----------
    system, basis, result
        The converged periodic system + basis + GAPW SCF result.
    basis_name
        Basis-set name for rebuilding on strained cells.
    v_ne_convention, smearing_alpha, functional
        Same kwargs the SCF was driven with.
    grid
        Pre-built :class:`PlaneWaveGrid`.
    cutoff_ha
        Fallback cutoff.
    eps
        Strain magnitude for central difference.
    gapw_kwargs
        Extra kwargs forwarded to the GAPW evaluator.

    Returns
    -------
    stress
        ``(3, 3)`` stress tensor in Hartree/bohr^3.
    """
    if not result.converged:
        raise ValueError("compute_stress_gapw: result is not converged.")
    one_centre = str(getattr(result, "one_centre", "block"))
    if one_centre != "block":
        raise NotImplementedError(
            "compute_stress_gapw: finite-strain derivatives are implemented "
            "only for one_centre='block'; the SCF result used "
            f"one_centre={one_centre!r}. Rerun explicitly with "
            "one_centre='block'."
        )

    D = np.asarray(result.density, dtype=float)
    if grid is None:
        grid = getattr(result, "grid", None)
    if grid is None:
        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = make_grid(
            np.asarray(system.lattice, dtype=float),
            cutoff_ha=cutoff_ha,
        )

    return _stress_fd(
        system,
        basis,
        basis_name,
        D,
        grid=grid,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        functional=functional,
        eps=eps,
        use_gapw=True,
        gapw_kwargs=gapw_kwargs,
    )
