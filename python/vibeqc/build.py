"""Slab and adsorbate construction utilities -- native, no ASE.

This module is the pure-geometry helper for surface-catalysis workflows.
It produces genuine ``dim=2`` :class:`PeriodicSystem` slabs (two in-plane
lattice vectors + Cartesian-z atoms; the internal ``a3`` is auto-synthesized
bookkeeping, not a vacuum-as-periodicity parameter). The vacuum-free 2D
Coulomb treatment (``jk_method='auto'`` → ``SLAB_EWALD_2D``) is a3-invariant.
Pass ``slab(..., periodic_z=True)`` for the legacy ``dim=3``-with-vacuum cell,
or build directly from lattice vectors with :func:`slab_2d`. Quantum-chemistry
parameters (basis, functional, k-mesh, ...) are decided downstream by the
caller.

Why native (no ASE runtime dep)
-------------------------------

Per CLAUDE.md Sec. 10, vibe-qc does not depend on other QC programs at
runtime, and ASE is treated as a library used through the `ase`
bridge module only. The slab builder is pure geometry -- there is no
reason to require ASE for it. The construction here is deliberately
simple (closed-form surface unit cells for the common facets) and
covers the 90 % use case: clean low-index slabs of fcc, bcc, and hcp
metals.

Supported facets
----------------

============ ==================  =====================================
Structure    Miller index        Notes
============ ==================  =====================================
``"fcc"``    (1,0,0)             centred-square surface cell, AB stack
``"fcc"``    (1,1,0)             rectangular, ABAB
``"fcc"``    (1,1,1)             hexagonal, ABCABC
``"bcc"``    (1,0,0)             square, AB
``"bcc"``    (1,1,0)             rectangular, AB (closest-packed)
``"bcc"``    (1,1,1)             hexagonal, ABCABC
``"hcp"``    (0,0,0,1)           basal plane, AB
============ ==================  =====================================

The c/a ratio for hcp defaults to the ideal :math:`\\sqrt{8/3}`; pass
``c_over_a=`` to override.

Lateral supercells, vacuum padding, and a layered ``layer_index``
metadata array (used by :func:`vibeqc.bipole_optimize.relax_atoms`
to freeze the bottom layers) are all supported.

Examples
--------

5-layer Fe(100) with 12 Å vacuum, 2x2 lateral cell::

    from vibeqc.build import slab
    fe100 = slab("Fe", "bcc", (1, 0, 0), n_layers=5,
                 vacuum=12.0, supercell=(2, 2))

Drop an N₂ on a bridge site, side-on, 2.0 Å above the top layer::

    from vibeqc.build import place_adsorbate, molecule
    n2 = molecule("N2", bond_length=1.10)  # Å
    fe100_n2 = place_adsorbate(fe100, n2, site="bridge",
                               orientation="side-on", height=2.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import Atom, PeriodicSystem

__all__ = [
    "slab",
    "slab_2d",
    "synthesize_slab_a3",
    "place_adsorbate",
    "molecule",
    "SlabInfo",
    "BULK_LATTICE_CONSTANTS",
]


_ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# Experimental room-temperature lattice constants in Å, for the listed
# bulk structure. These are *defaults* -- pass ``a=`` to override. The
# table is short by design (the metals most commonly used as catalysts /
# substrates); extend as needed.
BULK_LATTICE_CONSTANTS: dict[str, tuple[str, float, Optional[float]]] = {
    # element : (structure, a [Å], c [Å] or None for cubic)
    "Li": ("bcc", 3.491, None),
    "Na": ("bcc", 4.225, None),
    "K": ("bcc", 5.225, None),
    "V": ("bcc", 3.030, None),
    "Cr": ("bcc", 2.880, None),
    "Fe": ("bcc", 2.866, None),
    "Nb": ("bcc", 3.301, None),
    "Mo": ("bcc", 3.147, None),
    "Ta": ("bcc", 3.303, None),
    "W": ("bcc", 3.165, None),
    "Al": ("fcc", 4.046, None),
    "Ni": ("fcc", 3.524, None),
    "Cu": ("fcc", 3.615, None),
    "Rh": ("fcc", 3.803, None),
    "Pd": ("fcc", 3.890, None),
    "Ag": ("fcc", 4.085, None),
    "Ir": ("fcc", 3.839, None),
    "Pt": ("fcc", 3.924, None),
    "Au": ("fcc", 4.078, None),
    "Pb": ("fcc", 4.951, None),
    "Be": ("hcp", 2.286, 3.585),
    "Mg": ("hcp", 3.209, 5.211),
    "Sc": ("hcp", 3.309, 5.273),
    "Ti": ("hcp", 2.951, 4.684),
    "Co": ("hcp", 2.507, 4.069),
    "Zn": ("hcp", 2.665, 4.947),
    "Y": ("hcp", 3.648, 5.732),
    "Zr": ("hcp", 3.232, 5.147),
    "Ru": ("hcp", 2.706, 4.282),
    "Cd": ("hcp", 2.979, 5.620),
    "Re": ("hcp", 2.760, 4.458),
    "Os": ("hcp", 2.734, 4.319),
}


_SYMBOL_TO_Z = {
    s: z
    for z, s in enumerate(
        [
            "",
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
        ]
    )
}


def _z(symbol_or_z: Union[str, int]) -> int:
    if isinstance(symbol_or_z, (int, np.integer)):
        return int(symbol_or_z)
    s = symbol_or_z.strip()
    s = s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()
    if s not in _SYMBOL_TO_Z:
        raise ValueError(
            f"build: unknown element symbol {symbol_or_z!r}. "
            f"Extend _SYMBOL_TO_Z in python/vibeqc/build.py."
        )
    return _SYMBOL_TO_Z[s]


@dataclass(frozen=True)
class SlabInfo:
    """Metadata produced alongside :func:`slab`.

    Attributes
    ----------
    element : str
        Element symbol, e.g. ``"Fe"``.
    structure : str
        Bulk structure tag, ``"fcc"`` / ``"bcc"`` / ``"hcp"``.
    facet : tuple[int, int, int]
        Miller index of the cleaved facet (3-tuple even for hcp; the
        4-tuple form ``(0,0,0,1)`` is collapsed to ``(0,0,1)``).
    n_layers : int
        Number of atomic layers in the slab.
    supercell : tuple[int, int]
        Lateral repeats applied to the primitive surface cell.
    a : float
        Bulk lattice constant in Å used to build the slab.
    c : float | None
        c parameter in Å for hcp, ``None`` for cubic.
    vacuum : float
        Vacuum thickness in Å added along the surface normal.
    layer_index : tuple[int, ...]
        One entry per atom in the slab, giving its layer ordinal
        (0 = bottom, ``n_layers - 1`` = top). Used by relaxation
        helpers (``freeze_indices=``) to fix the bottom N layers.
    """

    element: str
    structure: str
    facet: Tuple[int, int, int]
    n_layers: int
    supercell: Tuple[int, int]
    a: float
    c: Optional[float]
    vacuum: float
    layer_index: Tuple[int, ...]

    def bottom_layer_indices(self, n_bottom: int) -> Tuple[int, ...]:
        """Atom indices of the bottom ``n_bottom`` layers.

        Handy for the ``freeze_indices=`` argument to
        :func:`vibeqc.bipole_optimize.relax_atoms` -- frozen-substrate
        relaxation is the standard surface-catalysis pattern.
        """
        if n_bottom <= 0:
            return ()
        return tuple(i for i, layer in enumerate(self.layer_index) if layer < n_bottom)


def _primitive_surface_cell(
    structure: str,
    facet: Tuple[int, int, int],
    a: float,
    c: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (a1, a2, basis, d) for the primitive surface unit cell.

    * ``a1``, ``a2`` are the in-plane primitive surface lattice vectors
      (Å, shape (2,)).
    * ``basis`` is a (n_basis_per_layer, 3) array of in-plane offsets
      for each atom in the primitive layer; the third coordinate (z)
      is always 0 (each entry is one atom in *one* layer, stacking is
      handled separately).
    * ``d`` is the inter-layer spacing in Å along the surface normal.

    The stacking sequence (AB, ABC, ...) is encoded by lateral offsets
    applied to successive layers; the offsets themselves are returned
    by :func:`_layer_stacking_offsets`.
    """
    s = structure.lower()
    h, k, l = facet

    if s == "fcc":
        if (h, k, l) == (1, 0, 0):
            d = a / 2.0
            # Primitive surface cell: square with side a/√2, rotated 45°.
            a1 = np.array([a / np.sqrt(2), 0.0])
            a2 = np.array([0.0, a / np.sqrt(2)])
            basis = np.array([[0.0, 0.0, 0.0]])
            return a1, a2, basis, d
        if (h, k, l) == (1, 1, 0):
            d = a / (2 * np.sqrt(2))
            a1 = np.array([a / np.sqrt(2), 0.0])
            a2 = np.array([0.0, a])
            basis = np.array([[0.0, 0.0, 0.0]])
            return a1, a2, basis, d
        if (h, k, l) == (1, 1, 1):
            d = a / np.sqrt(3)
            # In-plane lattice = (a/√2) x (a/√2) at 60°.
            ax = a / np.sqrt(2)
            a1 = np.array([ax, 0.0])
            a2 = np.array([ax / 2.0, ax * np.sqrt(3) / 2.0])
            basis = np.array([[0.0, 0.0, 0.0]])
            return a1, a2, basis, d

    if s == "bcc":
        if (h, k, l) == (1, 0, 0):
            d = a / 2.0
            a1 = np.array([a, 0.0])
            a2 = np.array([0.0, a])
            basis = np.array([[0.0, 0.0, 0.0]])
            return a1, a2, basis, d
        if (h, k, l) == (1, 1, 0):
            d = a / np.sqrt(2)
            # Closest-packed BCC plane. Rectangular surface cell:
            # a1 along [001] of length a; a2 along [1̄10] of length a√2.
            a1 = np.array([a, 0.0])
            a2 = np.array([0.0, a * np.sqrt(2)])
            basis = np.array([[0.0, 0.0, 0.0]])
            return a1, a2, basis, d
        if (h, k, l) == (1, 1, 1):
            d = a * np.sqrt(3) / 6.0
            # 2D hexagonal surface lattice with parameter a√2.
            ax = a * np.sqrt(2)
            a1 = np.array([ax, 0.0])
            a2 = np.array([ax / 2.0, ax * np.sqrt(3) / 2.0])
            basis = np.array([[0.0, 0.0, 0.0]])
            return a1, a2, basis, d

    if s == "hcp":
        # (0001) basal plane; facet=(0,0,1) accepted as shorthand.
        if (h, k, l) in ((0, 0, 1),):
            if c is None:
                raise ValueError(
                    "hcp slab requires c lattice constant (pass c=... "
                    "or use an element with a default in "
                    "BULK_LATTICE_CONSTANTS)"
                )
            d = c / 2.0
            a1 = np.array([a, 0.0])
            a2 = np.array([-a / 2.0, a * np.sqrt(3) / 2.0])
            basis = np.array([[0.0, 0.0, 0.0]])
            return a1, a2, basis, d

    raise ValueError(
        f"slab: facet {facet!r} on structure {structure!r} is not "
        f"in the built-in table. Supported: fcc(100/110/111), "
        f"bcc(100/110/111), hcp(0001). Extend "
        f"_primitive_surface_cell in vibeqc/build.py if you need "
        f"more."
    )


def _layer_stacking_offsets(
    structure: str,
    facet: Tuple[int, int, int],
    a1: np.ndarray,
    a2: np.ndarray,
) -> Sequence[np.ndarray]:
    """Per-layer in-plane offsets for the stacking sequence.

    Returns the cycle (length 2 for AB, 3 for ABC) of 2-vectors in Å
    that shift each successive layer relative to the bottom one.
    """
    s = structure.lower()
    h, k, l = facet
    zero = np.zeros(2)

    if s == "fcc" and (h, k, l) == (1, 0, 0):
        # AB stacking: layer 1 sits at the centre of the (a1,a2) cell.
        return (zero, 0.5 * a1 + 0.5 * a2)
    if s == "fcc" and (h, k, l) == (1, 1, 0):
        return (zero, 0.5 * a1 + 0.5 * a2)
    if s == "fcc" and (h, k, l) == (1, 1, 1):
        # ABC stacking.
        return (zero, (a1 + a2) / 3.0, 2.0 * (a1 + a2) / 3.0)
    if s == "bcc" and (h, k, l) == (1, 0, 0):
        return (zero, 0.5 * a1 + 0.5 * a2)
    if s == "bcc" and (h, k, l) == (1, 1, 0):
        return (zero, 0.5 * a1 + 0.5 * a2)
    if s == "bcc" and (h, k, l) == (1, 1, 1):
        return (zero, (a1 + a2) / 3.0, 2.0 * (a1 + a2) / 3.0)
    if s == "hcp" and (h, k, l) == (0, 0, 1):
        return (zero, (a1 + a2) / 3.0)
    raise ValueError(f"build: no stacking sequence for {structure!r}{facet!r}")


def _normalise_facet(facet: Sequence[int], structure: str) -> Tuple[int, int, int]:
    """Accept 3- or 4-index Miller; collapse 4-index hcp to 3-index."""
    if len(facet) == 4:
        if structure.lower() != "hcp":
            raise ValueError(
                f"build: 4-index Miller (Bravais) notation is for hcp "
                f"only, got structure={structure!r}"
            )
        h, k, _i, l = facet
        return (int(h), int(k), int(l))
    if len(facet) == 3:
        return (int(facet[0]), int(facet[1]), int(facet[2]))
    raise ValueError(f"build: facet must be a 3- or 4-tuple, got {facet!r}")


def _as_3vec(v: Sequence[float]) -> np.ndarray:
    """Return ``v`` as a length-3 float array; a 2-vector is embedded in the
    ``z = 0`` plane (slab normal along +z by convention)."""
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size == 2:
        return np.array([arr[0], arr[1], 0.0])
    if arr.size == 3:
        return arr.astype(float)
    raise ValueError(f"slab_2d: lattice vector must have 2 or 3 components, got {arr.size}")


def synthesize_slab_a3(
    a1: Sequence[float],
    a2: Sequence[float],
    atom_positions_bohr: Sequence[Sequence[float]] = (),
    *,
    normal_padding_bohr: float = 15.0,
    min_length_bohr: float = 30.0,
) -> np.ndarray:
    """Synthesize the internal (non-physical) ``a3`` for a ``dim=2`` slab.

    A genuine 2D slab is defined by only the two in-plane lattice vectors
    ``a1, a2``; the vibe-qc slab physics (the ``SLAB_EWALD_2D`` /
    Parry–de Leeuw–Perram gauge) is provably invariant to the third vector.
    But the AO-integral machinery and spglib both require a full-rank 3×3
    lattice, so we synthesize an ``a3`` along the slab normal
    ``n̂ = (a1×a2)/|a1×a2|``, long enough that it plays no physical role:

        ``|a3| = max(min_length_bohr, z_extent + 2·normal_padding_bohr)``

    where ``z_extent`` is the span of the atom positions projected onto ``n̂``.
    The length only affects bookkeeping (spglib cell, AO pair-list cutoff cost);
    it is **not** a vacuum-gap periodicity parameter and the SCF energy does not
    depend on it (see ``tests/test_build_slab.py`` a3-invariance).
    """
    v1 = _as_3vec(a1)
    v2 = _as_3vec(a2)
    normal = np.cross(v1, v2)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError(
            "slab_2d: in-plane vectors a1, a2 are collinear; cannot define a "
            "slab normal."
        )
    n_hat = normal / norm
    positions = [np.asarray(r, dtype=float) for r in atom_positions_bohr]
    if positions:
        proj = np.array([float(np.dot(r, n_hat)) for r in positions])
        z_extent = float(proj.max() - proj.min())
    else:
        z_extent = 0.0
    length = max(float(min_length_bohr), z_extent + 2.0 * float(normal_padding_bohr))
    return n_hat * length


def slab_2d(
    a1: Sequence[float],
    a2: Sequence[float],
    atoms: Sequence[Atom],
    *,
    charge: int = 0,
    multiplicity: int = 1,
    normal_padding_bohr: float = 15.0,
    min_a3_bohr: float = 30.0,
) -> PeriodicSystem:
    """Build a genuine ``dim=2`` (slab) :class:`PeriodicSystem` from two
    in-plane lattice vectors and Cartesian atoms with **real** z.

    Parameters
    ----------
    a1, a2 : length-2 or length-3 sequences (bohr)
        The in-plane lattice vectors. A 2-vector is embedded in the ``z=0``
        plane. These are the *only* periodicity the slab has.
    atoms : sequence of :class:`Atom`
        Unit-cell atoms with Cartesian positions in bohr. Their coordinate
        along the slab normal is physical and preserved as given — there is no
        vacuum centring and no ``a3`` the caller must supply.
    charge, multiplicity : int
        Forwarded to :class:`PeriodicSystem`.
    normal_padding_bohr, min_a3_bohr : float
        Control only the auto-synthesized bookkeeping ``a3`` (see
        :func:`synthesize_slab_a3`); the SCF is invariant to them.

    Returns
    -------
    PeriodicSystem
        With ``dim=2``. The lattice columns are ``a1``, ``a2``, and the
        synthesized normal ``a3``. Route it through ``jk_method='auto'`` (or
        ``'slab_ewald_2d'``) for the vacuum-free 2D Coulomb treatment.

    Notes
    -----
    This retires the old "slab = ``dim=3`` cell with a big vacuum gap"
    convention. For an explicit 3D-with-vacuum cell (e.g. a plane-wave-style
    reference), use :func:`slab` with ``periodic_z=True``.
    """
    positions = [np.asarray(at.xyz, dtype=float) for at in atoms]
    a3 = synthesize_slab_a3(
        a1,
        a2,
        positions,
        normal_padding_bohr=normal_padding_bohr,
        min_length_bohr=min_a3_bohr,
    )
    lattice = np.column_stack([_as_3vec(a1), _as_3vec(a2), a3])
    return PeriodicSystem(
        2, lattice, list(atoms), charge=charge, multiplicity=multiplicity
    )


def slab(
    element: Union[str, int],
    structure: Optional[str] = None,
    facet: Sequence[int] = (1, 0, 0),
    n_layers: int = 4,
    *,
    vacuum: float = 12.0,
    supercell: Tuple[int, int] = (1, 1),
    a: Optional[float] = None,
    c: Optional[float] = None,
    charge: int = 0,
    multiplicity: int = 1,
    periodic_z: bool = False,
) -> Tuple[PeriodicSystem, SlabInfo]:
    """Build a thin-film slab with vacuum padding along the third axis.

    Parameters
    ----------
    element : str | int
        Element symbol (``"Fe"``) or atomic number. Used both to look
        up the default lattice constant from
        :data:`BULK_LATTICE_CONSTANTS` (if ``a`` is not given) and to
        populate the ``Z`` field of each atom.
    structure : str, optional
        ``"fcc"`` / ``"bcc"`` / ``"hcp"``. Defaults to the structure
        listed for ``element`` in :data:`BULK_LATTICE_CONSTANTS`.
    facet : 3- or 4-tuple of int
        Miller index (``(1, 0, 0)``, ``(1, 1, 1)``, ...). For hcp the
        4-index Bravais notation is accepted (``(0, 0, 0, 1)``) and
        collapsed.
    n_layers : int
        Number of atomic layers in the slab (>= 1).
    vacuum : float, default 12.0
        For ``periodic_z=True`` (legacy dim=3): the physical vacuum gap in Å
        added along the third lattice vector. For the ``dim=2`` default it is
        **non-physical** — recorded only as viewer / grid-extent metadata in
        :class:`SlabInfo`; the SCF does not use it (the internal ``a3`` is
        auto-synthesized and the 2D physics is a3-invariant).
    supercell : (int, int)
        Lateral repeats of the primitive surface cell along ``a1`` and
        ``a2``. ``(2, 2)`` is the most common.
    a, c : float, optional
        Bulk lattice constants in Å. Defaults from
        :data:`BULK_LATTICE_CONSTANTS`. ``c`` is only used for hcp.
    charge, multiplicity : int
        Forwarded to :class:`PeriodicSystem`. The default ``mult=1`` is
        usually wrong for metals -- set explicitly for spin-polarized
        SCF.
    periodic_z : bool, default False
        If ``False`` (default), build a genuine ``dim=2`` slab (vacuum-free 2D
        physics; ``vacuum`` is metadata only). If ``True``, build the legacy
        ``dim=3`` cell with a physical vacuum gap along ``c`` and the slab
        centred in z — an explicit escape hatch for 3D-with-vacuum references.

    Returns
    -------
    system : PeriodicSystem
        ``dim=2`` by default (lattice columns ``a1``, ``a2``, and the
        auto-synthesized bookkeeping ``a3``; atoms keep their real z). With
        ``periodic_z=True``, ``dim=3`` and the third vector spans slab+vacuum.
    info : SlabInfo
        Layer index per atom + metadata.

    Notes
    -----
    The slab is centred along the third lattice vector: atoms occupy
    a contiguous slice of thickness ``(n_layers - 1) * d`` with vacuum
    split half-and-half above and below. This is the canonical surface-
    catalysis layout (no dipole moment across the cell when the slab
    is symmetric).
    """
    sym = element if isinstance(element, str) else None
    if (
        structure is None
        or a is None
        or (BULK_LATTICE_CONSTANTS.get(sym, (None,))[0] == "hcp" and c is None)
    ):
        # Pull defaults from the bulk table.
        if sym is None:
            raise ValueError(
                "slab: pass structure= and a= (and c= for hcp) when "
                "element is given as an atomic number."
            )
        if sym not in BULK_LATTICE_CONSTANTS:
            raise ValueError(
                f"slab: no default structure / lattice constant for "
                f"{sym!r}. Pass structure= and a= explicitly, or "
                f"extend BULK_LATTICE_CONSTANTS in "
                f"python/vibeqc/build.py."
            )
        default_struct, default_a, default_c = BULK_LATTICE_CONSTANTS[sym]
        structure = structure or default_struct
        if a is None:
            a = default_a
        if c is None and default_c is not None:
            c = default_c

    if structure.lower() == "hcp" and c is None:
        raise ValueError("slab: hcp requires c lattice constant")
    if n_layers < 1:
        raise ValueError(f"slab: n_layers must be >= 1, got {n_layers}")
    if vacuum < 0.0:
        raise ValueError(f"slab: vacuum must be >= 0, got {vacuum}")
    nx, ny = supercell
    if nx < 1 or ny < 1:
        raise ValueError(f"slab: supercell entries must be >= 1, got {supercell}")

    facet3 = _normalise_facet(facet, structure)
    a1_prim, a2_prim, _basis, d = _primitive_surface_cell(
        structure, facet3, float(a), c if c is None else float(c)
    )
    stack_offsets = _layer_stacking_offsets(structure, facet3, a1_prim, a2_prim)

    # Build the n_layers x (primitive in-plane) atom list, in Å.
    Z = _z(element)
    coords_ang: list[np.ndarray] = []
    layer_index: list[int] = []
    for layer in range(n_layers):
        off = stack_offsets[layer % len(stack_offsets)]
        z = layer * d
        coords_ang.append(np.array([off[0], off[1], z]))
        layer_index.append(layer)

    # Lateral supercell.
    if nx != 1 or ny != 1:
        new_coords: list[np.ndarray] = []
        new_layers: list[int] = []
        for ix in range(nx):
            for iy in range(ny):
                shift = ix * np.array([a1_prim[0], a1_prim[1], 0.0]) + iy * np.array(
                    [a2_prim[0], a2_prim[1], 0.0]
                )
                for r, li in zip(coords_ang, layer_index):
                    new_coords.append(r + shift)
                    new_layers.append(li)
        coords_ang = new_coords
        layer_index = new_layers
        a1 = a1_prim * nx
        a2 = a2_prim * ny
    else:
        a1 = a1_prim
        a2 = a2_prim

    slab_thickness = (n_layers - 1) * d

    if periodic_z:
        # Legacy dim=3 cell: a physical vacuum gap along c, slab centred in z.
        # Retained as an explicit escape hatch (plane-wave-style references, or
        # a 3D-with-vacuum comparison); this is NOT the vacuum-free 2D physics.
        # Prefer the dim=2 default below for real slab calculations.
        c_len = slab_thickness + vacuum
        z_origin = (c_len - slab_thickness) / 2.0
        coords_ang3 = [r + np.array([0.0, 0.0, z_origin]) for r in coords_ang]
        lattice_ang = np.zeros((3, 3))
        lattice_ang[:2, 0] = a1
        lattice_ang[:2, 1] = a2
        lattice_ang[2, 2] = c_len
        lattice_bohr = lattice_ang * _ANGSTROM_TO_BOHR
        atoms = [Atom(Z, list(r * _ANGSTROM_TO_BOHR)) for r in coords_ang3]
        system = PeriodicSystem(
            3, lattice_bohr, atoms, charge=charge, multiplicity=multiplicity
        )
    else:
        # Default: a genuine dim=2 slab. Atoms keep their natural z (0 ..
        # slab_thickness); the internal a3 is auto-synthesized and non-physical
        # (see slab_2d / synthesize_slab_a3). There is no vacuum-as-periodicity;
        # `vacuum` is retained only as viewer / grid-extent metadata in SlabInfo.
        a1_bohr = _as_3vec(a1) * _ANGSTROM_TO_BOHR
        a2_bohr = _as_3vec(a2) * _ANGSTROM_TO_BOHR
        atoms = [Atom(Z, list(r * _ANGSTROM_TO_BOHR)) for r in coords_ang]
        system = slab_2d(
            a1_bohr, a2_bohr, atoms, charge=charge, multiplicity=multiplicity
        )
    info = SlabInfo(
        element=sym or f"Z{Z}",
        structure=structure.lower(),
        facet=facet3,
        n_layers=n_layers,
        supercell=(nx, ny),
        a=float(a),
        c=None if c is None else float(c),
        vacuum=float(vacuum),
        layer_index=tuple(layer_index),
    )
    return system, info


# ---------------------------------------------------------------------------
# Tiny molecule library for adsorbate convenience.
#
# Bonds are arranged along +z so that "side-on" adsorption (axis parallel
# to the surface) and "end-on" (axis perpendicular to the surface) can be
# selected by an orientation argument in place_adsorbate.
# ---------------------------------------------------------------------------


_BUILTIN_MOLECULES: dict[str, tuple[tuple[str, tuple[float, float, float]], ...]] = {
    # Bond lengths in Å. These are reasonable starting geometries --
    # users typically relax the adsorbate against the surface.
    "H": (("H", (0.0, 0.0, 0.0)),),
    "H2": (("H", (0.0, 0.0, -0.371)), ("H", (0.0, 0.0, 0.371))),
    "N2": (("N", (0.0, 0.0, -0.550)), ("N", (0.0, 0.0, 0.550))),
    "O2": (("O", (0.0, 0.0, -0.604)), ("O", (0.0, 0.0, 0.604))),
    "CO": (("C", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.128))),
    "OH": (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.970))),
    "NH": (("N", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.036))),
    "NO": (("N", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.151))),
    "NH2": (
        ("N", (0.0, 0.0, 0.0)),
        ("H", (0.8012, 0.0, 0.6376)),
        ("H", (-0.8012, 0.0, 0.6376)),
    ),
    "NH3": (
        ("N", (0.0, 0.0, 0.0)),
        ("H", (0.9377, 0.0, 0.3816)),
        ("H", (-0.4688, 0.8120, 0.3816)),
        ("H", (-0.4688, -0.8120, 0.3816)),
    ),
    "H2O": (
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.7572, 0.0, 0.5860)),
        ("H", (-0.7572, 0.0, 0.5860)),
    ),
    "CH4": (
        ("C", (0.0, 0.0, 0.0)),
        ("H", (0.6276, 0.6276, 0.6276)),
        ("H", (-0.6276, -0.6276, 0.6276)),
        ("H", (0.6276, -0.6276, -0.6276)),
        ("H", (-0.6276, 0.6276, -0.6276)),
    ),
}


def molecule(
    name_or_atoms: Union[str, Iterable[tuple]],
    *,
    bond_length: Optional[float] = None,
) -> Tuple[Tuple[str, Tuple[float, float, float]], ...]:
    """Return a small adsorbate geometry as a tuple of (symbol, xyz_Å).

    Parameters
    ----------
    name_or_atoms : str | iterable
        Built-in name (``"H"``, ``"H2"``, ``"N2"``, ``"O2"``,
        ``"CO"``, ``"OH"``, ``"NH"``, ``"NO"``, ``"NH2"``,
        ``"NH3"``, ``"H2O"``, ``"CH4"``) or a sequence of
        ``(symbol, (x, y, z))`` tuples in Å.
    bond_length : float, optional
        For diatomics (``H2``, ``N2``, ``O2``, ``CO``, ``OH``,
        ``NH``, ``NO``), override the bond length in Å. Ignored
        otherwise.

    The geometry is centred on the first atom (or the heavy atom for
    diatomics where it matters) so that :func:`place_adsorbate` can
    drop it onto a surface site with a simple shift.
    """
    if not isinstance(name_or_atoms, str):
        return tuple(
            (s, (float(r[0]), float(r[1]), float(r[2]))) for s, r in name_or_atoms
        )
    name = name_or_atoms.strip()
    if name not in _BUILTIN_MOLECULES:
        raise KeyError(
            f"molecule: {name!r} not in the built-in library "
            f"({sorted(_BUILTIN_MOLECULES)}). Pass an explicit "
            f"sequence of (symbol, xyz) tuples instead."
        )
    geom = _BUILTIN_MOLECULES[name]
    if bond_length is not None and len(geom) == 2:
        # Rebuild the diatomic centred on its midpoint.
        half = float(bond_length) / 2.0
        s1, _ = geom[0]
        s2, _ = geom[1]
        return (
            (s1, (0.0, 0.0, -half)),
            (s2, (0.0, 0.0, half)),
        )
    return geom


def _wrap_in_cell(pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Wrap the in-plane (x,y) components back into [0, a) x [0, b).

    The z-coordinate (surface normal) is left unchanged -- wrapping it
    would risk placing the adsorbate below the surface.

    Parameters
    ----------
    pos : np.ndarray, shape (3,)
        Cartesian position in Å.
    cell : np.ndarray, shape (3, 3)
        Lattice vectors as columns in Å.

    Returns
    -------
    np.ndarray, shape (3,)
        Position with in-plane components wrapped into the cell in Å.
    """
    frac = np.linalg.solve(cell.T, pos)  # Å -> fractional
    # Only wrap the in-plane fractional coordinates (indices 0, 1).
    frac[0] = frac[0] % 1.0
    frac[1] = frac[1] % 1.0
    return cell.T @ frac  # fractional -> Å


def _site_offset_for_slab(
    info: SlabInfo,
    site: str,
) -> np.ndarray:
    """In-plane offset (in Å, 2-vector) for a named site on the slab.

    The offset is relative to the position of the topmost-layer atom
    in the (0, 0) primitive surface unit cell after stacking -- i.e.
    the upper-left atom in the top layer. ``place_adsorbate`` adds
    this to that atom's projected (x, y) to produce the adsorbate's
    lateral position.

    Sites are defined per facet; "bridge" and "hollow" semantics vary.
    """
    a = float(info.a)
    c = info.c if info.c is not None else 0.0
    s = info.structure
    h, k, l = info.facet
    site = site.lower()

    # The in-plane primitive vectors of the *primitive* surface cell
    # (we never multiplied them by supercell -- sites are placed within
    # one primitive cell).
    a1, a2, _basis, _d = _primitive_surface_cell(s, info.facet, a, c)

    if site == "top":
        return np.zeros(2)
    if site == "bridge":
        return 0.5 * a1
    if site == "long-bridge":
        return 0.5 * a2
    if site == "short-bridge":
        return 0.5 * a1
    if site == "hollow":
        if s == "fcc" and (h, k, l) == (1, 0, 0):
            # 4-fold hollow at the cell centre.
            return 0.5 * a1 + 0.5 * a2
        if s == "bcc" and (h, k, l) == (1, 0, 0):
            return 0.5 * a1 + 0.5 * a2
        if (s, (h, k, l)) in (("fcc", (1, 1, 1)), ("hcp", (0, 0, 1))):
            # fcc hollow = 1/3, 2/3.
            return (a1 + a2) / 3.0
        if (s, (h, k, l)) == ("bcc", (1, 1, 0)):
            # Long-bridge ≈ 3-fold hollow on bcc(110).
            return 0.5 * a1 + 0.25 * a2
        return 0.5 * a1 + 0.5 * a2
    if site == "fcc-hollow":
        return (a1 + a2) / 3.0
    if site == "hcp-hollow":
        return 2.0 * (a1 + a2) / 3.0
    raise ValueError(
        f"place_adsorbate: site {site!r} not recognised. Try "
        f"'top', 'bridge', 'hollow' (or 'fcc-hollow' / "
        f"'hcp-hollow' for (111) / (0001))."
    )


def _orient_molecule(
    coords: np.ndarray,
    orientation: str,
    molecule_name: str = "",
) -> np.ndarray:
    """Rotate molecular coordinates into the requested orientation.

    Parameters
    ----------
    coords : np.ndarray, shape (n_atoms, 3)
        Input coordinates (Å) in the molecule's local frame, with the
        anchor / binding atom at the lowest z (z=0 after a z_min shift)
        and the rest of the molecule extending along +z.
    orientation : str
        One of ``"end-on"``, ``"upright"``, ``"side-on"``,
        ``"tilted"``, ``"flat"``, ``"binding-atom-down"``, ``"auto"``.
    molecule_name : str
        Molecule label for ``"auto"`` resolution.

    Returns
    -------
    np.ndarray
        Rotated coordinates, with the lowest atom at z=0.
    """
    orient = orientation.lower()

    # Aliases.
    if orient in ("end-on", "upright", "binding-atom-down"):
        # Already in the correct frame: anchor at bottom, molecule up.
        out = coords.copy()
        out[:, 2] -= out[:, 2].min()
        return out

    if orient == "side-on":
        # Rotate 90° about y: (x, y, z) -> (-z, y, x) so the bond
        # lies along +x and both atoms end up at the same z.
        new = coords.copy()
        new[:, 0] = -coords[:, 2]
        new[:, 2] = coords[:, 0]
        new[:, 2] -= new[:, 2].min()
        return new

    if orient == "tilted":
        # Tilt 45° from the surface normal (z) toward +x.
        # Rotate by -45° about y-axis.
        c45 = np.cos(np.radians(45.0))
        s45 = np.sin(np.radians(45.0))
        R = np.array([[c45, 0.0, -s45], [0.0, 1.0, 0.0], [s45, 0.0, c45]])
        new = coords @ R.T
        new[:, 2] -= new[:, 2].min()
        return new

    if orient == "flat":
        n_atoms = coords.shape[0]
        if n_atoms <= 2:
            # Diatomic / single atom -> side-on.
            return _orient_molecule(coords, "side-on")
        # Polyatomic: rotate so the molecule lies flat on the surface.
        # For NH3 / H2O / CH4, rotate 90° about the x-axis so that
        # the principal axis (z in the built-in geometry) goes into
        # the y-direction.
        new = coords.copy()
        # Rotate 90° about x: (x, y, z) -> (x, -z, y).
        new[:, 1] = -coords[:, 2]
        new[:, 2] = coords[:, 1]
        new[:, 2] -= new[:, 2].min()
        return new

    if orient == "auto":
        # Smart default per molecule.
        name = molecule_name.upper().strip()
        side_on = ("N2", "H2", "O2")
        if name in side_on:
            return _orient_molecule(coords, "side-on")
        # Everything else defaults to end-on / binding-atom-down.
        return _orient_molecule(coords, "end-on")

    raise ValueError(
        f"place_adsorbate: orientation must be one of 'end-on', "
        f"'upright', 'side-on', 'tilted', 'flat', "
        f"'binding-atom-down', 'auto', got {orientation!r}"
    )


def _resolve_anchor(
    syms: list[str],
    anchor: int | str | None,
    molecule_name: str,
) -> int:
    """Return the 0-based index of the anchor (binding) atom.

    The anchor atom is the one placed closest to the surface.
    """
    if anchor is None:
        # Default anchor: for heteronuclear diatomics, the heavy atom
        # (the one that typically binds); for homonuclear, atom 0.
        return 0
    if isinstance(anchor, int):
        if anchor < 0 or anchor >= len(syms):
            raise ValueError(
                f"place_adsorbate: anchor index {anchor} out of range "
                f"for {len(syms)}-atom molecule"
            )
        return anchor
    # String: element symbol.
    for i, s in enumerate(syms):
        if s.upper() == anchor.upper():
            return i
    raise ValueError(
        f"place_adsorbate: anchor symbol {anchor!r} not found in "
        f"molecule {molecule_name!r} (atoms: {syms})"
    )


def place_adsorbate(
    slab_system: PeriodicSystem,
    adsorbate: Union[
        str, Iterable[tuple], Tuple[Tuple[str, Tuple[float, float, float]], ...]
    ],
    *,
    info: Optional[SlabInfo] = None,
    site: str = "top",
    height: Optional[float] = None,
    distance: Optional[float] = None,
    orientation: str = "auto",
    position: Optional[Tuple[float, float]] = None,
    bond_length: Optional[float] = None,
    anchor: int | str | None = None,
    coverage: float = 1.0,
) -> PeriodicSystem:
    """Attach an adsorbate to a slab surface.

    Parameters
    ----------
    slab_system : PeriodicSystem
        Slab produced by :func:`slab` (or any 3D-periodic system with
        vacuum along the third lattice vector).
    adsorbate : str | iterable | tuple
        Built-in name (``"N2"``, ``"CO"``, ``"NH3"``, ...) -- see
        :func:`molecule` -- or an explicit
        ``((symbol, (x, y, z)), ...)`` sequence in Å.
    info : SlabInfo, optional
        Slab metadata from :func:`slab`. Required when using a named
        ``site`` rather than explicit ``position``.
    site : str
        Named adsorption site: ``"top"``, ``"bridge"``, ``"hollow"``,
        ``"fcc-hollow"``, ``"hcp-hollow"``, ``"long-bridge"``,
        ``"short-bridge"``. Resolution requires ``info``.
    height : float, optional
        Perpendicular distance (Å) from the surface plane to the
        anchor atom's centre. Exactly one of ``height`` or
        ``distance`` must be given.
    distance : float, optional
        Alias for ``height``. Exactly one must be set.
    orientation : str, default ``"auto"``
        For diatomics: ``"end-on"`` / ``"upright"`` (axis perpendicular
        to surface), ``"side-on"`` (parallel), ``"tilted"`` (45°
        from normal).
        For polyatomics: ``"binding-atom-down"`` (anchor points
        toward surface), ``"flat"`` (molecular plane parallel to
        surface), ``"auto"`` (most common bonding mode).
    position : (float, float), optional
        Direct in-plane position in Å (relative to lattice origin),
        overriding ``site`` + ``info``.
    bond_length : float, optional
        Forwarded to :func:`molecule` to override the built-in bond
        length for diatomics.
    anchor : int | str | None, optional
        Which atom binds to the surface. By index (0-based) or element
        symbol. Default: the first atom in the molecule (typically
        the heavy / binding atom).
    coverage : float, default 1.0
        Fractional surface coverage. With ``coverage < 1.0``, the
        adsorbate is placed once in the cell (the caller is
        responsible for using a suitably sized supercell).

    Returns
    -------
    PeriodicSystem
        New system with the adsorbate atoms appended to the slab.

    Raises
    ------
    ValueError
        If the molecule name is unknown, the site is unrecognised,
        both ``height`` and ``distance`` are set, or the anchor
        index is out of range.
    """
    # --- Resolve height vs distance ---
    if height is not None and distance is not None:
        raise ValueError(
            "place_adsorbate: use only one of 'height' or 'distance', not both."
        )
    if height is None and distance is None:
        raise ValueError("place_adsorbate: one of 'height' or 'distance' must be set.")
    h = float(height if height is not None else distance)

    # --- Resolve molecule name (for auto-orientation) ---
    molecule_name = adsorbate if isinstance(adsorbate, str) else ""

    # --- Get geometry ---
    geom = molecule(adsorbate, bond_length=bond_length)
    coords = np.array([list(r) for _, r in geom], dtype=float)
    syms = [s for s, _ in geom]

    if len(coords) == 0:
        raise ValueError("place_adsorbate: adsorbate must have at least one atom.")

    # --- Centre -> orient -> re-anchor ---
    # First shift so the lowest atom is at z=0.
    z_min = coords[:, 2].min()
    coords[:, 2] -= z_min

    # Orient.
    coords = _orient_molecule(coords, orientation, molecule_name)

    # Determine which atom goes to the surface (anchor).
    anchor_idx = _resolve_anchor(syms, anchor, molecule_name)

    # Shift so the anchor atom is at z=0.
    anchor_z = coords[anchor_idx, 2]
    coords[:, 2] -= anchor_z

    # --- Decide the in-plane position ---
    if position is not None:
        in_plane = np.array(position, dtype=float)
    else:
        if info is None:
            raise ValueError(
                "place_adsorbate: pass info= (from slab(...)) to "
                "resolve named sites, or pass position=(x, y)."
            )
        in_plane = _site_offset_for_slab(info, site)

    # --- Top-layer z ---
    z_top_bohr = max(float(a.xyz[2]) for a in slab_system.unit_cell)
    z_top_ang = z_top_bohr / _ANGSTROM_TO_BOHR
    z0 = z_top_ang + h

    # --- Lattice cell in Å for wrapping ---
    lat_bohr = np.asarray(slab_system.lattice, dtype=float)
    lat_ang = lat_bohr / _ANGSTROM_TO_BOHR

    # --- Build adsorbate atoms (in Å) ---
    ads_atoms = []
    for s, r in zip(syms, coords):
        x = r[0] + in_plane[0]
        y = r[1] + in_plane[1]
        z = r[2] + z0
        # Wrap (x, y) into the cell in case the site offset pushed
        # the position outside [0, a) x [0, b).
        wrapped = _wrap_in_cell(np.array([x, y, z], dtype=float), lat_ang)
        ads_atoms.append(Atom(_z(s), list(wrapped * _ANGSTROM_TO_BOHR)))

    new_atoms = list(slab_system.unit_cell) + ads_atoms
    return PeriodicSystem(
        slab_system.dim,
        lat_bohr,
        new_atoms,
        charge=slab_system.charge,
        multiplicity=slab_system.multiplicity,
    )
