"""Region partition for Green's-function embedding on tagged slabs.

Divides a slab+adsorbate :class:`~vibeqc._vibeqc_core.PeriodicSystem` into
the three embedded regions needed by Layer A and Layer B using per-atom
integer tags:

* ``0`` -- substrate (semi-infinite bulk, region II). Not treated explicitly;
  its effect enters through the embedding potential ``Sigma_emb``.
* ``1`` -- region I (surface atoms solved explicitly in the embedded Dyson
  equation). Region I sits above the dividing plane *S*.
* ``2`` -- ΔV support (adsorbate + locally perturbed surface atoms, Layer B).
  The localized perturbation whose Dyson equation is solved on top of the
  clean-surface Green function ``G0`` from Layer A.

The module maps these atom-level tags to AO index ranges through the
supplied :class:`~vibeqc._vibeqc_core.BasisSet` and defines a dividing
plane *S* (a plane of constant *z* in Cartesian bohr) separating region I
from the substrate.

The AO ordering follows vibe-qc's canonical layout: shells grouped by atom
in unit-cell order, each shell contributing ``2l+1`` (spherical) or
``(l+1)(l+2)/2`` (Cartesian) functions. The atom->AO mapping is recomputed
from :meth:`BasisSet.shells` each time, so a basis change or a different
system needs a fresh :class:`RegionPartition`.

References
----------
* J. E. Inglesfield, J. Phys. C 14, 3795 (1981), doi:10.1088/0022-3719/14/26/015.
* H. Ishida, Phys. Rev. B 63, 165409 (2001), doi:10.1103/PhysRevB.63.165409.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .._vibeqc_core import BasisSet, PeriodicSystem

# Tag convention -- kept as module constants so every consumer shares one
# source of truth.
TAG_SUBSTRATE = 0
TAG_REGION_I = 1
TAG_DELTA_V = 2

_VALID_TAGS = frozenset({TAG_SUBSTRATE, TAG_REGION_I, TAG_DELTA_V})


@dataclass(frozen=True)
class _AtomAOMap:
    """Atom index -> AO range mapping for one :class:`BasisSet`.

    Not part of the public API -- consumed by :meth:`RegionPartition.from_tags`
    and re-derivable from any ``(system, basis)`` pair.
    """

    ao_to_atom: NDArray[np.int64]  # (n_ao,) -- which atom owns each AO
    atom_ao_start: NDArray[np.int64]  # (n_atoms+1,) -- AO range starts

    @classmethod
    def from_basis(cls, basis: BasisSet, n_atoms: int) -> "_AtomAOMap":
        """Build the atom->AO map from a basis set for a system of ``n_atoms``.

        The map is computed from :meth:`BasisSet.shells`, which returns
        :class:`~vibeqc._vibeqc_core.ShellInfo` entries with ``atom_index``
        and angular momentum ``l``. A shell contributes ``2l+1`` functions
        when ``pure=True`` (spherical harmonics) or ``(l+1)(l+2)/2`` when
        ``pure=False`` (Cartesian).
        """
        nbf = basis.nbasis
        ao_to_atom = np.empty(nbf, dtype=np.int64)
        ao_to_atom[:] = -1

        # Two-pass: first, build ao_to_atom; second, build atom_ao_start.
        shells = basis.shells()
        bf = 0
        for s in shells:
            n_func = _n_func(s.l, s.pure)
            for _ in range(n_func):
                ao_to_atom[bf] = s.atom_index
                bf += 1

        atom_start = np.zeros(n_atoms + 1, dtype=np.int64)
        for ai in range(n_atoms):
            atom_start[ai + 1] = atom_start[ai] + int(np.sum(ao_to_atom == ai))
        return cls(ao_to_atom=ao_to_atom, atom_ao_start=atom_start)


def _n_func(l: int, pure: bool) -> int:
    """Number of basis functions contributed by a shell of angular momentum ``l``."""
    if pure:
        return 2 * l + 1
    return (l + 1) * (l + 2) // 2


def _ao_indices_for_atoms(
    atom_mask: NDArray[np.bool_], atom_ao_map: _AtomAOMap
) -> NDArray[np.int64]:
    """Return a flat, sorted array of AO indices belonging to masked atoms."""
    starts = atom_ao_map.atom_ao_start
    indices = []
    for ai in np.flatnonzero(atom_mask):
        indices.extend(range(starts[ai], starts[ai + 1]))
    return np.array(indices, dtype=np.int64)


@dataclass(frozen=True)
class RegionPartition:
    """Atom and AO partition of a tagged slab for Green's-function embedding.

    Construct with :meth:`from_tags`. The dataclass is frozen -- each
    ``(system, basis, tags)`` triple maps to exactly one partition.

    Attributes
    ----------
    i_ao
        AO indices for region I (solved explicitly). Shape ``(n_i,)``.
    s_ao
        AO indices on/near the dividing plane *S*. A subset of ``i_ao``.
        Shape ``(n_s,)``.
    dv_ao
        AO indices for the ΔV support region (Layer B). Shape ``(n_dv,)``.
    i_atoms
        Atom indices of region I. Shape ``(n_i_atoms,)``.
    s_atoms
        Atom indices whose AOs are on plane *S*. Shape ``(n_s_atoms,)``.
    dv_atoms
        Atom indices of ΔV support. Shape ``(n_dv_atoms,)``.
    substrate_atoms
        Atom indices of the substrate (tag 0). Shape ``(n_sub_atoms,)``.
    s_plane_z
        Dividing-plane *z* coordinate in bohr (Cartesian).
    n_ao_total
        Total number of AOs in the full basis.
    basis_name
        Name of the basis set used.
    """

    i_ao: NDArray[np.int64]
    s_ao: NDArray[np.int64]
    dv_ao: NDArray[np.int64]

    i_atoms: NDArray[np.int64]
    s_atoms: NDArray[np.int64]
    dv_atoms: NDArray[np.int64]
    substrate_atoms: NDArray[np.int64]

    s_plane_z: float

    n_ao_total: int
    basis_name: str

    # -- derived short-hands -------------------------------------------------
    @property
    def n_i(self) -> int:
        return int(self.i_ao.shape[0])

    @property
    def n_s(self) -> int:
        return int(self.s_ao.shape[0])

    @property
    def n_dv(self) -> int:
        return int(self.dv_ao.shape[0])

    @property
    def n_i_atoms(self) -> int:
        return int(self.i_atoms.shape[0])

    @property
    def n_substrate_atoms(self) -> int:
        return int(self.substrate_atoms.shape[0])

    # -- construction --------------------------------------------------------
    @classmethod
    def from_tags(
        cls,
        system: PeriodicSystem,
        basis: BasisSet,
        tags: NDArray[np.int64] | list[int] | tuple[int, ...],
        *,
        s_plane_offset: float = 1.0,
    ) -> "RegionPartition":
        """Build a partition from per-atom integer tags.

        Parameters
        ----------
        system
            Slab + adsorbate system. ``system.unit_cell`` provides the atom
            positions; ``system.dim`` is expected to be 2 or 3 (slab).
        basis
            Orbital basis set. Its shells must match the atom count and order
            in ``system``.
        tags
            Per-atom integer tags, one entry per atom in ``system.unit_cell``.
            Allowed values: 0 (substrate), 1 (region I), 2 (ΔV support).
            Every atom must carry exactly one tag.
        s_plane_offset
            Distance in bohr that the dividing plane *S* is placed *below*
            the lowest (smallest *z*) region-I atom centre. Used only when
            there are no substrate atoms below region I (i.e. the slab's
            bottom is tagged as region I -- an unusual but valid setup).
            When substrate atoms sit below region I, the plane is placed at
            the midpoint of the widest gap between the region-I and substrate
            atom *z*-coordinate ranges.  Default 1.0 bohr (~ 0.53 Å).

        Returns
        -------
        RegionPartition
            Frozen partition with AO index arrays, atom index arrays, and the
            dividing plane geometry. Ready for consumption by
            :mod:`~vibeqc.periodic_embedding.substrate_gf` and
            :mod:`~vibeqc.periodic_embedding.surface_sigma`.

        Raises
        ------
        ValueError
            If tags contain invalid values, don't match the atom count, or
            the system has no region-I atoms.
        """
        tags = np.asarray(tags, dtype=np.int64)
        n_atoms = len(system.unit_cell)
        if tags.shape != (n_atoms,):
            raise ValueError(f"tags length {tags.shape[0]} != n_atoms {n_atoms}")
        invalid = set(np.unique(tags)) - _VALID_TAGS
        if invalid:
            raise ValueError(
                f"Invalid tag values {invalid}; allowed: {sorted(_VALID_TAGS)}"
            )

        mask_i = tags == TAG_REGION_I
        mask_dv = tags == TAG_DELTA_V
        mask_sub = tags == TAG_SUBSTRATE

        if not mask_i.any():
            raise ValueError("No atoms tagged as region I (tag=1)")

        i_atoms = np.flatnonzero(mask_i).astype(np.int64)
        dv_atoms = np.flatnonzero(mask_dv).astype(np.int64)
        sub_atoms = np.flatnonzero(mask_sub).astype(np.int64)

        # --- AO map ---
        ao_map = _AtomAOMap.from_basis(basis, n_atoms)
        i_ao = _ao_indices_for_atoms(mask_i, ao_map)
        dv_ao = _ao_indices_for_atoms(mask_dv, ao_map)

        # --- Dividing plane S ---
        # Extract z-coordinates of all unit-cell atoms in bohr.
        z_all = np.array([a.xyz[2] for a in system.unit_cell], dtype=float)
        z_i = z_all[i_atoms]
        s_plane_z = cls._compute_s_plane_z(
            z_i=z_i,
            z_sub=z_all[sub_atoms] if len(sub_atoms) > 0 else None,
            offset=s_plane_offset,
        )

        # AOs on plane S: currently all region-I AOs whose atom centres are
        # within a neighbourhood of the plane (default: all region-I AOs,
        # since the first implementation projects S_emb onto the full region-I
        # basis -- the typical approach for thin region I).
        s_atoms = cls._resolve_s_atoms(
            z_all=z_all,
            i_atoms=i_atoms,
            s_plane_z=s_plane_z,
        )
        s_ao = _ao_indices_for_atoms(np.isin(np.arange(n_atoms), s_atoms), ao_map)

        return cls(
            i_ao=i_ao,
            s_ao=s_ao,
            dv_ao=dv_ao,
            i_atoms=i_atoms,
            s_atoms=s_atoms,
            dv_atoms=dv_atoms,
            substrate_atoms=sub_atoms,
            s_plane_z=s_plane_z,
            n_ao_total=basis.nbasis,
            basis_name=basis.name,
        )

    # -- plane geometry helpers ----------------------------------------------
    @staticmethod
    def _compute_s_plane_z(
        z_i: NDArray[np.float64],
        z_sub: Optional[NDArray[np.float64]],
        offset: float,
    ) -> float:
        """Choose the dividing-plane *z* coordinate.

        * If substrate atoms sit below region I (``z_sub`` given and
          ``z_sub.max() < z_i.min()``), place *S* at the midpoint of the
          gap: ``(z_i.min() + z_sub.max()) / 2``.
        * Otherwise (no substrate below, or only region I), place *S* at
          ``z_i.min() - offset``. This is the case for a bare surface
          without explicit substrate atoms -- the embedding potential
          will supply the missing half-space.
        """
        z_i_min = float(z_i.min())
        if z_sub is not None and len(z_sub) > 0:
            z_sub_max = float(z_sub.max())
            if z_sub_max < z_i_min:
                return 0.5 * (z_i_min + z_sub_max)
        return z_i_min - float(offset)

    @staticmethod
    def _resolve_s_atoms(
        z_all: NDArray[np.float64],
        i_atoms: NDArray[np.int64],
        s_plane_z: float,
        *,
        margin: float = 3.0,
    ) -> NDArray[np.int64]:
        """Select region-I atoms whose *z*-coordinate is within ``margin``
        bohr of the dividing plane.

        For thin region I (1-2 layers) this typically returns all region-I
        atoms. For thicker region I it picks only the bottom layer(s) --
        the ones whose basis functions have significant amplitude at *S*.
        """
        z_i = z_all[i_atoms]
        close = np.abs(z_i - s_plane_z) <= margin
        return i_atoms[close].astype(np.int64)
