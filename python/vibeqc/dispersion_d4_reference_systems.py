"""D4 reference-system catalogue -- Phase D4b-3.

The native D4 C6 model (Phase D4b-4) interpolates a per-element
reference C6 dataset over the coordination number ``CN`` and the EEQ
partial charge ``q``. Each reference data point is computed from a
*reference system*: the element embedded in a small molecule that
fixes it at one particular coordination environment.

This module is the **catalogue** of those reference systems. For each
element in scope it provides a list of closed-shell host molecules
with standard geometries, the index of the target (reference) atom,
and the target's nominal coordination number. Phase D4b-4 feeds each
system through the :func:`vibeqc.molecular_c6` pipeline (RHF ->
coupled ``a(iw)`` -> Casimir-Polder) to produce the reference
``a(iw)`` / C6 table.

Scope of this milestone
-----------------------
Period-2 covalent main-group elements -- **H, B, C, N, O, F** -- plus
the closed-shell noble-gas atoms **He** and **Ne**. These are the
elements the D4b-2b pilot validated and the ones whose coordination
chemistry needs more than one reference system. Deferred to a later
milestone:

* **Free-atom references** for the open-shell first-row atoms
  (C ^3P, N ⁴S, O ^3P, F ^2P, B ^2P, H ^2S). The reference-C6 pipeline
  is RHF-only; the free-atom limit needs the UHF/ROHF response path.
  Closed-shell free atoms (He, Ne) *are* included.
* **Period-3 and below** (Si, P, S, Cl, ...) and the transition
  metals. Same machinery, more systems -- a later sweep.

See ``handovers/HANDOVER_D4_NATIVE.md`` Sec. 3 (the re-derivation campaign)
and Sec. 6 (current state / next action).

Geometry provenance
-------------------
All geometries are standard experimental or well-established computed
equilibrium structures -- bond lengths and angles from the NIST
Computational Chemistry Comparison and Benchmark Database (CCCBDB)
and the CRC Handbook of Chemistry and Physics. Molecular geometries
are physical facts, not copyrightable. Each :class:`ReferenceSystem`
records its ``geometry_source``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._vibeqc_core import Atom, Molecule

__all__ = [
    "ReferenceSystem",
    "ANGSTROM_TO_BOHR",
    "reference_systems_for",
    "all_reference_elements",
    "all_reference_systems",
    "build_molecule",
]

# CODATA 2018 -- 1 Å = 1.8897259886 a0.
ANGSTROM_TO_BOHR = 1.8897259886

# Element symbol <-> Z, for the elements this catalogue covers.
_SYMBOL = {1: "H", 2: "He", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
           10: "Ne"}


@dataclass(frozen=True)
class ReferenceSystem:
    """One D4 reference system: an element in a fixed coordination
    environment, given by a closed-shell host molecule.

    Attributes
    ----------
    element
        Atomic number ``Z`` of the element this reference is *for*.
    label
        Short unique identifier, ``"<symbol>/<host>-<motif>"``,
        e.g. ``"C/CH4-sp3"``.
    atoms_angstrom
        The host molecule geometry as ``((Z, x, y, z), ...)`` with
        coordinates in **Ångström**. :func:`build_molecule` converts
        to the bohr that :class:`Molecule` expects.
    target_index
        Index into ``atoms_angstrom`` of the reference atom. Its
        ``Z`` must equal ``element``.
    nominal_cn
        The count of bonded (first-shell) neighbours of the target
        atom -- the integer coordination number the reference system
        was chosen to populate. Note this counts *neighbours*, so a
        multiply-bonded partner counts once (N in N₂ has
        ``nominal_cn = 1``, not 3). The D4 model uses the continuous
        erf coordination number computed at run time (Phase D4b-4);
        for these single-neighbour-per-bond molecules it tracks
        ``nominal_cn`` closely.
    charge
        Total molecular charge. ``0`` for every system in this
        milestone; the non-neutral references that calibrate the
        D4 charge dependence are a later addition.
    multiplicity
        Spin multiplicity. ``1`` (closed shell) for every system
        here -- a hard requirement of the RHF-only reference
        pipeline.
    geometry_source
        Provenance string for the geometry.
    """

    element: int
    label: str
    atoms_angstrom: tuple[tuple[int, float, float, float], ...]
    target_index: int
    nominal_cn: float
    charge: int = 0
    multiplicity: int = 1
    geometry_source: str = ""

    def __post_init__(self) -> None:
        if not self.atoms_angstrom:
            raise ValueError(f"{self.label}: empty geometry")
        if not (0 <= self.target_index < len(self.atoms_angstrom)):
            raise ValueError(
                f"{self.label}: target_index {self.target_index} out of "
                f"range for {len(self.atoms_angstrom)} atoms")
        target_z = self.atoms_angstrom[self.target_index][0]
        if target_z != self.element:
            raise ValueError(
                f"{self.label}: target atom Z={target_z} but element="
                f"{self.element}")
        if self.multiplicity != 1:
            raise ValueError(
                f"{self.label}: only closed-shell (multiplicity=1) "
                f"reference systems are supported in Phase D4b-3")
        n_elec = sum(z for z, *_ in self.atoms_angstrom) - self.charge
        if n_elec % 2 != 0:
            raise ValueError(
                f"{self.label}: odd electron count ({n_elec}) is "
                f"inconsistent with multiplicity=1")

    @property
    def symbol(self) -> str:
        return _SYMBOL.get(self.element, f"Z{self.element}")

    @property
    def n_atoms(self) -> int:
        return len(self.atoms_angstrom)


# ---------------------------------------------------------------------
# Geometry builders -- small helpers so the catalogue stays readable.
# All return ``((Z, x, y, z), ...)`` tuples in Ångström, target atom
# first by convention.
# ---------------------------------------------------------------------

def _atom_only(z: int) -> tuple[tuple[int, float, float, float], ...]:
    """A single free atom at the origin."""
    return ((z, 0.0, 0.0, 0.0),)


def _diatomic(z_a: int, z_b: int, r: float):
    """A-B along z; A at the origin."""
    return ((z_a, 0.0, 0.0, 0.0), (z_b, 0.0, 0.0, r))


def _linear_triatomic(z_centre: int, z_end: int, r: float):
    """Symmetric linear E-centre-E (e.g. CO2); centre at the origin."""
    return ((z_centre, 0.0, 0.0, 0.0),
            (z_end, 0.0, 0.0, r),
            (z_end, 0.0, 0.0, -r))


def _linear_abc(z_a: int, z_b: int, z_c: int, r_ab: float, r_bc: float):
    """Linear A-B-C along z, B at the origin (e.g. HCN: H-C-N)."""
    return ((z_a, 0.0, 0.0, -r_ab),
            (z_b, 0.0, 0.0, 0.0),
            (z_c, 0.0, 0.0, r_bc))


def _tetrahedral(z_centre: int, z_lig: int, r: float):
    """E centre + 4 identical ligands at tetrahedral vertices."""
    d = r / math.sqrt(3.0)
    return ((z_centre, 0.0, 0.0, 0.0),
            (z_lig, d, d, d),
            (z_lig, -d, -d, d),
            (z_lig, -d, d, -d),
            (z_lig, d, -d, -d))


def _trigonal_planar(z_centre: int, z_lig: int, r: float):
    """E centre + 3 identical ligands, D3h, in the xy-plane."""
    out = [(z_centre, 0.0, 0.0, 0.0)]
    for k in range(3):
        ang = 2.0 * math.pi * k / 3.0
        out.append((z_lig, r * math.cos(ang), r * math.sin(ang), 0.0))
    return tuple(out)


def _pyramidal_xh3(z_centre: int, r: float, angle_deg: float):
    """C3v E-H3 (e.g. NH3): centre on +z, 3 H below in a ring.

    ``angle_deg`` is the H-E-H angle.  The geometry is solved so
    every E-H bond length equals ``r`` and every H-E-H angle equals
    ``angle_deg``.
    """
    half = math.radians(angle_deg) / 2.0
    # H-H ring radius r and the E->ring-plane offset h satisfy
    #   sin(half) = (r.sqrt(3)/2) / r        (half the H-H chord / r)
    #   r^2 + h^2 = r^2
    rho = r * math.sin(half) * 2.0 / math.sqrt(3.0)
    h = math.sqrt(max(r * r - rho * rho, 0.0))
    out = [(z_centre, 0.0, 0.0, 0.0)]
    for k in range(3):
        ang = 2.0 * math.pi * k / 3.0
        out.append((1, rho * math.cos(ang), rho * math.sin(ang), -h))
    return tuple(out)


def _bent_xh2(z_centre: int, r: float, angle_deg: float):
    """C2v H-E-H (e.g. H2O); centre at the origin, opens toward +z."""
    half = math.radians(angle_deg) / 2.0
    x = r * math.sin(half)
    z = r * math.cos(half)
    return ((z_centre, 0.0, 0.0, 0.0),
            (1, x, 0.0, z),
            (1, -x, 0.0, z))


# ---------------------------------------------------------------------
# The reference-system catalogue.
#
# Bond lengths (Å) / angles: NIST CCCBDB experimental geometries
# unless noted. nominal_cn = count of bonded neighbours of the target.
# ---------------------------------------------------------------------

_CCCBDB = "NIST CCCBDB, experimental equilibrium geometry"

_REFERENCE_SYSTEMS: tuple[ReferenceSystem, ...] = (
    # --- Hydrogen (Z=1) -- terminal in every common environment ----
    ReferenceSystem(
        element=1, label="H/H2",
        atoms_angstrom=_diatomic(1, 1, 0.7414),
        target_index=0, nominal_cn=1.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=1, label="H/CH4",
        atoms_angstrom=_tetrahedral(6, 1, 1.0870),
        target_index=1, nominal_cn=1.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=1, label="H/H2O",
        atoms_angstrom=_bent_xh2(8, 0.9578, 104.48),
        target_index=1, nominal_cn=1.0, geometry_source=_CCCBDB),

    # --- Helium (Z=2) -- closed-shell free atom --------------------
    ReferenceSystem(
        element=2, label="He/atom",
        atoms_angstrom=_atom_only(2),
        target_index=0, nominal_cn=0.0,
        geometry_source="free atom"),

    # --- Boron (Z=5) -- trigonal-planar sp^2 ------------------------
    ReferenceSystem(
        element=5, label="B/BH3-sp2",
        atoms_angstrom=_trigonal_planar(5, 1, 1.1900),
        target_index=0, nominal_cn=3.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=5, label="B/BF3-sp2",
        atoms_angstrom=_trigonal_planar(5, 9, 1.3070),
        target_index=0, nominal_cn=3.0, geometry_source=_CCCBDB),

    # --- Carbon (Z=6) -- sp^3 / sp^2 / sp ----------------------------
    ReferenceSystem(
        element=6, label="C/CH4-sp3",
        atoms_angstrom=_tetrahedral(6, 1, 1.0870),
        target_index=0, nominal_cn=4.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=6, label="C/C2H4-sp2",
        atoms_angstrom=(
            (6, 0.0, 0.0, 0.6695), (6, 0.0, 0.0, -0.6695),
            (1, 0.0, 0.9289, 1.2321), (1, 0.0, -0.9289, 1.2321),
            (1, 0.0, 0.9289, -1.2321), (1, 0.0, -0.9289, -1.2321)),
        target_index=0, nominal_cn=3.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=6, label="C/C2H2-sp",
        atoms_angstrom=(
            (6, 0.0, 0.0, 0.6015), (6, 0.0, 0.0, -0.6015),
            (1, 0.0, 0.0, 1.6625), (1, 0.0, 0.0, -1.6625)),
        target_index=0, nominal_cn=2.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=6, label="C/CO2-sp",
        atoms_angstrom=_linear_triatomic(6, 8, 1.1600),
        target_index=0, nominal_cn=2.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=6, label="C/CO",
        atoms_angstrom=_diatomic(6, 8, 1.1283),
        target_index=0, nominal_cn=1.0, geometry_source=_CCCBDB),

    # --- Nitrogen (Z=7) -- CN 1 / 2 / 3 ----------------------------
    ReferenceSystem(
        element=7, label="N/NH3-sp3",
        atoms_angstrom=_pyramidal_xh3(7, 1.0124, 106.67),
        target_index=0, nominal_cn=3.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=7, label="N/N2H2-sp2",
        # trans-diimide (E-HN=NH), C2h planar -- N at CN 2.
        atoms_angstrom=(
            (7, -0.626, 0.0, 0.0), (7, 0.626, 0.0, 0.0),
            (1, -0.924, 0.984, 0.0), (1, 0.924, -0.984, 0.0)),
        target_index=0, nominal_cn=2.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=7, label="N/N2-sp",
        atoms_angstrom=_diatomic(7, 7, 1.0977),
        target_index=0, nominal_cn=1.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=7, label="N/HCN-sp",
        atoms_angstrom=_linear_abc(1, 6, 7, 1.0655, 1.1532),
        target_index=2, nominal_cn=1.0, geometry_source=_CCCBDB),

    # --- Oxygen (Z=8) -- sp^3 (water) / sp^2 (carbonyl) --------------
    ReferenceSystem(
        element=8, label="O/H2O-sp3",
        atoms_angstrom=_bent_xh2(8, 0.9578, 104.48),
        target_index=0, nominal_cn=2.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=8, label="O/CO2-sp2",
        atoms_angstrom=_linear_triatomic(6, 8, 1.1600),
        target_index=1, nominal_cn=1.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=8, label="O/CO",
        atoms_angstrom=_diatomic(6, 8, 1.1283),
        target_index=1, nominal_cn=1.0, geometry_source=_CCCBDB),

    # --- Fluorine (Z=9) -- terminal in every environment -----------
    ReferenceSystem(
        element=9, label="F/HF",
        atoms_angstrom=_diatomic(1, 9, 0.9168),
        target_index=1, nominal_cn=1.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=9, label="F/F2",
        atoms_angstrom=_diatomic(9, 9, 1.4119),
        target_index=0, nominal_cn=1.0, geometry_source=_CCCBDB),
    ReferenceSystem(
        element=9, label="F/BF3",
        atoms_angstrom=_trigonal_planar(5, 9, 1.3070),
        target_index=1, nominal_cn=1.0, geometry_source=_CCCBDB),

    # --- Neon (Z=10) -- closed-shell free atom ---------------------
    ReferenceSystem(
        element=10, label="Ne/atom",
        atoms_angstrom=_atom_only(10),
        target_index=0, nominal_cn=0.0,
        geometry_source="free atom"),
)


# Index by element for O(1) lookup.
_BY_ELEMENT: dict[int, tuple[ReferenceSystem, ...]] = {}
for _rs in _REFERENCE_SYSTEMS:
    _BY_ELEMENT.setdefault(_rs.element, ())
    _BY_ELEMENT[_rs.element] = _BY_ELEMENT[_rs.element] + (_rs,)


def reference_systems_for(element: int) -> tuple[ReferenceSystem, ...]:
    """Return every reference system catalogued for atomic number
    ``element``. Raises :class:`KeyError` for an element not yet in
    scope (see the module docstring for the covered set)."""
    if element not in _BY_ELEMENT:
        covered = ", ".join(_SYMBOL[z] for z in sorted(_BY_ELEMENT))
        raise KeyError(
            f"no D4 reference systems for element Z={element}; "
            f"Phase D4b-3 covers: {covered}")
    return _BY_ELEMENT[element]


def all_reference_elements() -> tuple[int, ...]:
    """Atomic numbers with at least one reference system, ascending."""
    return tuple(sorted(_BY_ELEMENT))


def all_reference_systems() -> tuple[ReferenceSystem, ...]:
    """The full catalogue, catalogue order."""
    return _REFERENCE_SYSTEMS


def build_molecule(refsys: ReferenceSystem) -> Molecule:
    """Construct the :class:`Molecule` for ``refsys`` -- geometry
    converted Å -> bohr, with the system's charge and multiplicity."""
    atoms = [
        Atom(z, [x * ANGSTROM_TO_BOHR,
                 y * ANGSTROM_TO_BOHR,
                 z_ * ANGSTROM_TO_BOHR])
        for (z, x, y, z_) in refsys.atoms_angstrom
    ]
    return Molecule(atoms, charge=refsys.charge,
                    multiplicity=refsys.multiplicity)
