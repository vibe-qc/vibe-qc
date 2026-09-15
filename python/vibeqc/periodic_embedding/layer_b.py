"""Layer B -- lateral impurity embedding (Dyson + Lloyd) on G0.

Given the clean-surface Green function ``G0`` from Layer A and a localized
perturbation ``Delta_V`` (the adsorbate + perturbed surface atoms), solves
the Dyson equation on the finite-support region and computes the adsorption
energy through Lloyd's formula.

The clean-surface G0 is the region-I GF from
:func:`~vibeqc.periodic_embedding.scf2step.compute_region_i_gf_at_kz`,
already embedded in the semi-infinite substrate via S_emb.  Layer B
breaks lateral periodicity *locally* and reaches the isolated (zero-
coverage) adsorbate limit.

Two modes:
* **Mock ΔV** -- a user-supplied potential shift on selected AOs (test path).
* **Full ΔV** -- Hamiltonian difference between the adsorbed and clean
  systems (production path, see :func:`compute_layer_b_delta_v`).

References
----------
* P. Lloyd, Proc. Phys. Soc. 90, 207 (1967).
* J. E. Inglesfield, "The Embedding Method for Electronic Structure,"
  IOP Publishing (2015), doi:10.1088/978-0-7503-1042-0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .._vibeqc_core import Atom, BasisSet, LatticeSumOptions, Molecule, PeriodicSystem
from .dyson import dyson_solve
from .lloyd import (
    lloyd_band_energy_change,
    lloyd_friedel_sum,
    lloyd_integrated_dos_change,
)
from .region import RegionPartition
from .substrate_gf import _build_hk_sk


@dataclass
class LayerBResult:
    """Result of a Layer-B impurity embedding calculation.

    Attributes
    ----------
    delta_n
        Integrated DOS change ``ΔN(E)`` along the energy sweep
        (Lloyd's formula), shape ``(n_energies,)``.
    delta_n_total
        Total displaced charge (Friedel sum) at the top of the sweep.
    band_energy_change
        Band-structure contribution to the adsorption energy (Ha).
    energies
        Real-energy grid for the sweep (Ha).
    g_perturbed
        Perturbed GF matrices at the sweep energies, shape
        ``(n_energies, n_dv, n_dv)``.
    """

    delta_n: NDArray[np.float64]
    delta_n_total: float
    band_energy_change: float
    energies: NDArray[np.float64]
    g_perturbed: NDArray[np.complex128]


def _g0_on_sweep(
    g0_fn,  # callable: complex -> (n_dv, n_dv) complex matrix
    energies: NDArray[np.float64],
    eta: float,
) -> NDArray[np.complex128]:
    """Evaluate G0(z) on a real-energy sweep ``E + i*eta``.

    Returns shape ``(n_energies, n_dv, n_dv)``.
    """
    n_e = len(energies)
    g0_first = np.atleast_2d(
        np.asarray(g0_fn(energies[0] + 1j * eta), dtype=np.complex128)
    )
    n_dv = g0_first.shape[0]
    result = np.empty((n_e, n_dv, n_dv), dtype=np.complex128)
    result[0] = g0_first
    for i in range(1, n_e):
        result[i] = np.atleast_2d(
            np.asarray(g0_fn(energies[i] + 1j * eta), dtype=np.complex128)
        )
    return result


def compute_layer_b_mock_delta_v(
    g0_fn,  # callable: complex z -> G0_II(z) as (n_i, n_i) complex matrix
    delta_v_diag: NDArray[np.float64],  # on-site shifts on the perturbed AOs
    *,
    dv_loc: Optional[NDArray[np.int64]] = None,
    region: Optional[RegionPartition] = None,
    energies: Optional[NDArray[np.float64]] = None,
    e_range: tuple[float, float] = (-3.0, 0.0),
    n_energies: int = 2000,
    eta: float = 0.01,
    e_fermi: Optional[float] = None,
) -> LayerBResult:
    """Layer B with a mock diagonal ΔV on specified AOs within region I.

    Parameters
    ----------
    g0_fn
        Callable ``z -> G0_II(z)`` returning the clean-surface GF in the
        full region-I AO basis, shape ``(n_i, n_i)``.
    delta_v_diag
        On-site energy shifts, shape ``(n_dv,)``.  ``ΔV = diag(delta_v_diag)``.
    dv_loc
        Integer positions of the perturbed AOs within the region-I basis
        (0-based into ``G0_II``).  If None, derived from ``region.dv_ao``.
    region
        Region partition.  Required if ``dv_loc`` is None.
    energies
        Real-energy sweep for Lloyd's formula.  If None, auto-generated.
    e_range
        ``(e_min, e_max)`` for auto-generated sweep.
    n_energies
        Number of energy points.
    eta
        Imaginary broadening (Ha).
    e_fermi
        Fermi level for band-energy integral.  If None, uses last energy.

    Returns
    -------
    LayerBResult
        ΔN, Friedel sum, band-energy change, and perturbed GF stack.
    """
    if dv_loc is None:
        if region is None:
            raise ValueError("Either dv_loc or region must be provided.")
        if region.n_dv == 0:
            raise ValueError("No ΔV support AOs (dv_ao is empty).")
        i_to_pos = {int(idx): pos for pos, idx in enumerate(region.i_ao)}
        dv_loc = np.array([i_to_pos[int(idx)] for idx in region.dv_ao], dtype=int)
    else:
        dv_loc = np.asarray(dv_loc, dtype=int)

    n_dv = len(dv_loc)
    delta_v_diag = np.asarray(delta_v_diag, dtype=float)
    if delta_v_diag.shape != (n_dv,):
        raise ValueError(
            f"delta_v_diag shape {delta_v_diag.shape} != expected ({n_dv},)"
        )

    dv_full = np.diag(delta_v_diag.astype(np.complex128))

    if energies is None:
        energies = np.linspace(e_range[0], e_range[1], n_energies, dtype=float)
    else:
        energies = np.asarray(energies, dtype=float)

    def _g0_dv(z: complex) -> NDArray[np.complex128]:
        g0_full = np.atleast_2d(np.asarray(g0_fn(z), dtype=np.complex128))
        return g0_full[np.ix_(dv_loc, dv_loc)]

    g0_stack = _g0_on_sweep(_g0_dv, energies, eta)

    delta_n = lloyd_integrated_dos_change(g0_stack, dv_full)
    delta_n_total = lloyd_friedel_sum(g0_stack, dv_full)

    ef = float(e_fermi) if e_fermi is not None else float(energies[-1])
    e_band = lloyd_band_energy_change(energies, delta_n, ef)

    # Compute perturbed GF at each energy for diagnostics.
    g_pert = np.empty_like(g0_stack)
    for i in range(len(energies)):
        g_pert[i] = dyson_solve(g0_stack[i], dv_full)

    return LayerBResult(
        delta_n=delta_n,
        delta_n_total=float(delta_n_total),
        band_energy_change=e_band,
        energies=energies,
        g_perturbed=g_pert,
    )


def compute_layer_b_delta_v(
    system_clean: PeriodicSystem,
    system_adsorbed: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    k_cart: NDArray[np.float64],
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
) -> NDArray[np.complex128]:
    """Build ΔV(k) as the core-Hamiltonian difference on the ΔV support.

    Computes ``H_adsorbed(k) - H_clean(k)`` restricted to the dv_ao
    indices.  The two systems must have congruent AO spaces -- use
    :func:`make_ghost_clean_system` to build a clean system with ghost
    (Z=0) atoms at the adsorbate positions.

    Parameters
    ----------
    system_clean
        Clean system with ghost atoms (same atom count as adsorbed).
    system_adsorbed
        Adsorbed periodic system.
    basis
        Orbital basis set (same for both systems, built from the
        adsorbed system's molecule).
    region
        Region partition with dv_ao non-empty.
    k_cart
        3D Cartesian k-vector.
    lat_opts
        Lattice-sum options.

    Returns
    -------
    delta_v
        ``ΔV(k)`` on the ΔV support AOs, shape ``(n_dv, n_dv)``.
    """
    if lat_opts is None:
        lat_opts = LatticeSumOptions()
        lat_opts.cutoff_bohr = 30.0
        lat_opts.nuclear_cutoff_bohr = 30.0

    H_clean, S_clean = _build_hk_sk(system_clean, basis, k_cart, lat_opts)
    H_ads, S_ads = _build_hk_sk(system_adsorbed, basis, k_cart, lat_opts)

    dv_ao = region.dv_ao
    H_clean_dv = H_clean[np.ix_(dv_ao, dv_ao)]
    H_ads_dv = H_ads[np.ix_(dv_ao, dv_ao)]

    return H_ads_dv - H_clean_dv


def make_ghost_clean_system(
    system_adsorbed: PeriodicSystem,
    basis: BasisSet,
    ghost_indices: list[int],
) -> tuple[PeriodicSystem, BasisSet]:
    """Build a clean system with ghost (Z=0) atoms at specified positions.

    Replaces the atoms at ``ghost_indices`` in ``system_adsorbed`` with
    Z=0 atoms while keeping the same basis functions.  This allows
    ``compute_layer_b_delta_v`` to compute ``ΔV = H_ads - H_clean``
    on congruent AO spaces.

    Parameters
    ----------
    system_adsorbed
        The adsorbed periodic system.
    basis
        The orbital basis for the adsorbed system.
    ghost_indices
        Which atom indices to make into ghosts (Z -> 0).  These are
        typically the adsorbate atoms.

    Returns
    -------
    (system_clean, basis_clean)
        The clean system with Z=0 at the specified positions, and a
        BasisSet constructed from the same shells on the clean molecule.
    """
    atoms_clean = []
    for i, a in enumerate(system_adsorbed.unit_cell):
        if i in ghost_indices:
            atoms_clean.append(Atom(0, [a.xyz[0], a.xyz[1], a.xyz[2]]))
        else:
            atoms_clean.append(a)

    # Multiplicity: same as adsorbed but with ghost electrons removed.
    n_elec = sum(a.Z for a in atoms_clean)
    mult = system_adsorbed.multiplicity
    # Keep multiplicity consistent: if n_elec goes from even->odd, adjust.
    if n_elec % 2 != mult % 2:
        mult = 2 if n_elec % 2 == 1 else 1

    system_clean = PeriodicSystem(
        dim=system_adsorbed.dim,
        lattice=system_adsorbed.lattice,
        unit_cell=atoms_clean,
        charge=0,
        multiplicity=mult,
    )

    # Build BasisSet from the adsorbed system's shells on the clean molecule.
    mol_clean = Molecule(atoms_clean, charge=0, multiplicity=mult)
    shells = basis.shells()
    basis_clean = BasisSet(mol_clean, shells, name=basis.name)

    return system_clean, basis_clean
