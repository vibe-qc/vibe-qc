"""Molecular graph construction from 3D atomic coordinates.

Detects bonds using covalent radii + tolerance, then builds a
molecular graph with adjacency, ring detection, and functional-group
analysis. This is the first stage of structure-to-name conversion.

References
----------
- Cordero et al., *Dalton Trans.* 2008, 2832 (covalent radii)
- IUPAC Red Book 2005, Sec. IR-4 (stoichiometric names)
- IUPAC Blue Book 2013 (organic nomenclature rules)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ── Element data ──────────────────────────────────────────────────────────────

ATOMIC_NUMBER_TO_SYMBOL: dict[int, str] = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
}

# Cordero et al. (2008) covalent radii in Å (single-bond).
# Values for common elements; fall back to 1.5 Å for missing.
COVALENT_RADII_ANGSTROM: dict[int, float] = {
    1: 0.31,
    2: 0.28,
    3: 1.28,
    4: 0.96,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    10: 0.58,
    11: 1.66,
    12: 1.41,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    18: 1.06,
    19: 2.03,
    20: 1.76,
    21: 1.70,
    22: 1.60,
    23: 1.53,
    24: 1.39,
    25: 1.39,
    26: 1.32,
    27: 1.26,
    28: 1.24,
    29: 1.32,
    30: 1.22,
    31: 1.22,
    32: 1.20,
    33: 1.19,
    34: 1.20,
    35: 1.20,
    36: 1.16,
    37: 2.20,
    38: 1.95,
    46: 1.39,
    47: 1.45,
    48: 1.44,
    53: 1.39,
    54: 1.40,
    78: 1.36,
    79: 1.36,
    80: 1.32,
    82: 1.46,
}

# Typical bond-order radii ratios: double ~0.87× single, triple ~0.78×
BOND_ORDER_FACTOR = {1: 1.00, 2: 0.87, 3: 0.78}

# Tolerance factor for bond detection (× sum of covalent radii)
BOND_TOLERANCE = 1.25


@dataclass
class AtomInfo:
    """Per-atom information for graph construction."""

    index: int
    z: int  # atomic number
    symbol: str  # element symbol
    x: float  # position in Å
    y: float
    z_coord: float


@dataclass
class BondInfo:
    """A detected bond between two atoms."""

    i: int
    j: int
    order: int = 1  # estimated bond order (1, 2, 3)
    distance: float = 0.0  # in Å


@dataclass
class MolecularGraph:
    """Full molecular graph built from 3D coordinates.

    Attributes
    ----------
    atoms : list of AtomInfo
    bonds : list of BondInfo
    adjacency : list of list of int  (neighbour indices per atom)
    rings : list of list of int      (atom indices in each ring)
    formula : str                    (Hill-system formula)
    n_atoms : int
    n_electrons : int
    is_organic : bool                (contains C and H as majority)
    """

    atoms: list[AtomInfo]
    bonds: list[BondInfo] = field(default_factory=list)
    adjacency: list[list[int]] = field(default_factory=list)
    rings: list[list[int]] = field(default_factory=list)
    formula: str = ""
    n_atoms: int = 0
    n_electrons: int = 0
    is_organic: bool = False

    def atom_degree(self, idx: int) -> int:
        """Number of bonded neighbours for atom *idx*."""
        return len(self.adjacency[idx]) if idx < len(self.adjacency) else 0

    def atom_neighbours(self, idx: int) -> list[int]:
        """Indices of atoms bonded to *idx*."""
        return self.adjacency[idx] if idx < len(self.adjacency) else []

    def bond_between(self, i: int, j: int) -> Optional[BondInfo]:
        """Return the BondInfo connecting atoms *i* and *j*, or None."""
        for b in self.bonds:
            if (b.i == i and b.j == j) or (b.i == j and b.j == i):
                return b
        return None

    def connected_components(self) -> list[list[int]]:
        """Partition the graph into disconnected subgraphs via BFS.

        Returns a list of atom-index lists, one per component. The
        largest component (by atom count) is first; this is typically
        the surface/slab in an adsorbed system.
        """
        visited: set[int] = set()
        components: list[list[int]] = []
        for i in range(self.n_atoms):
            if i not in visited:
                # BFS from this atom
                comp: list[int] = []
                queue = [i]
                visited.add(i)
                while queue:
                    v = queue.pop(0)
                    comp.append(v)
                    for nb in self.adjacency[v]:
                        if nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
                components.append(comp)

        # Sort by size descending — largest component first
        components.sort(key=len, reverse=True)
        return components

    def subgraph(self, indices: list[int]) -> MolecularGraph:
        """Build a new MolecularGraph restricted to the given atom indices.

        Atom indices are remapped 0..n-1 in the returned subgraph.
        Positions, bonds, and adjacency are preserved.
        """
        idx_set = set(indices)
        new_atoms = [self.atoms[i] for i in indices]
        # Remap indices
        index_map = {old: new for new, old in enumerate(indices)}
        new_bonds = []
        for b in self.bonds:
            if b.i in idx_set and b.j in idx_set:
                new_bonds.append(
                    BondInfo(
                        i=index_map[b.i],
                        j=index_map[b.j],
                        order=b.order,
                        distance=b.distance,
                    )
                )
        new_adj = [[] for _ in range(len(indices))]
        for b in new_bonds:
            new_adj[b.i].append(b.j)
            new_adj[b.j].append(b.i)

        # Rebuild formula from sub-atoms
        formula = _hill_formula(new_atoms)

        return MolecularGraph(
            atoms=new_atoms,
            bonds=new_bonds,
            adjacency=new_adj,
            rings=[],
            formula=formula,
            n_atoms=len(new_atoms),
            n_electrons=sum(a.z for a in new_atoms),
            is_organic=self.is_organic,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Bond detection
# ═══════════════════════════════════════════════════════════════════════════════


def covalent_radius(z: int, order: int = 1) -> float:
    """Covalent radius (Å) for element *z* at given bond order."""
    r1 = COVALENT_RADII_ANGSTROM.get(z, 1.50)
    return r1 * BOND_ORDER_FACTOR.get(order, 1.0)


def _estimate_bond_order(z_i: int, z_j: int, distance: float, aromatic: bool = False) -> int:
    """Bond-order estimate from distance vs covalent radii.

    When *aromatic* is True, returns 1 (the ring-perception layer in
    smiles.py handles aromatic display). Ratio thresholds widened
    to 0.80/0.90 to catch C=C at ~1.34 A (ratio 0.88)."""
    if aromatic:
        return 1
    # Hydrogen bonds are always single
    if z_i == 1 or z_j == 1:
        return 1
    r_single = covalent_radius(z_i) + covalent_radius(z_j)
    if r_single <= 0:
        return 1
    ratio = distance / r_single
    if ratio < 0.80:
        return 3
    if ratio < 0.90:
        return 2
    return 1


def detect_bonds(
    atoms: list[AtomInfo],
    tolerance: float = BOND_TOLERANCE,
) -> list[BondInfo]:
    """Detect chemical bonds from atomic positions and covalent radii.

    Parameters
    ----------
    atoms :
        Atom list with positions in Å.
    tolerance :
        Multiply the sum of covalent radii by this factor to get the
        bond-detection cutoff. Default 1.25 accommodates polar / weak bonds.

    Returns
    -------
    List of detected bonds with estimated bond orders.
    """
    bonds: list[BondInfo] = []
    n = len(atoms)

    for i in range(n):
        for j in range(i + 1, n):
            dx = atoms[i].x - atoms[j].x
            dy = atoms[i].y - atoms[j].y
            dz = atoms[i].z_coord - atoms[j].z_coord
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            r_max = (
                covalent_radius(atoms[i].z) + covalent_radius(atoms[j].z)
            ) * tolerance

            if dist <= r_max:
                order = _estimate_bond_order(atoms[i].z, atoms[j].z, dist)
                bonds.append(BondInfo(i=i, j=j, order=order, distance=dist))

    return bonds


# ═══════════════════════════════════════════════════════════════════════════════
# Ring detection
# ═══════════════════════════════════════════════════════════════════════════════


def _find_cycles(adjacency: list[list[int]], max_size: int = 7) -> list[list[int]]:
    """Find all rings (cycles) up to *max_size* using DFS."""
    n = len(adjacency)
    rings: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def dfs(start: int, current: int, path: list[int], depth: int) -> None:
        if depth > max_size:
            return
        for neighbour in adjacency[current]:
            if neighbour == start and depth >= 3:
                cycle = tuple(sorted(path))
                if cycle not in seen:
                    seen.add(cycle)
                    rings.append(list(path))
                continue
            if neighbour in path:
                continue
            path.append(neighbour)
            dfs(start, neighbour, path, depth + 1)
            path.pop()

    for i in range(n):
        dfs(i, i, [i], 1)

    return rings


def _is_aromatic_ring(
    ring: list[int], atoms: list[AtomInfo], bonds: list[BondInfo]
) -> bool:
    """Heuristic: 5- or 6-membered ring of C/N with alternating single/double bonds."""
    if len(ring) not in (5, 6):
        return False
    for idx in ring:
        if atoms[idx].z not in (6, 7):  # C or N
            return False
    # Check alternating bond orders in the ring
    orders: list[int] = []
    ring_set = set(ring)
    for b in bonds:
        if b.i in ring_set and b.j in ring_set:
            orders.append(b.order)
    if not orders:
        return False
    # Count double bonds
    n_double = sum(1 for o in orders if o >= 2)
    ring_size = len(ring)
    if ring_size == 6 and n_double >= 3:  # benzene-like
        return True
    if ring_size == 5 and n_double >= 2:  # pyrrole/furan-like
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════════════


def _hill_formula(atoms: list[AtomInfo]) -> str:
    """Hill-system formula: C first, then H, then rest alphabetically."""
    counts: dict[str, int] = {}
    for a in atoms:
        counts[a.symbol] = counts.get(a.symbol, 0) + 1

    parts: list[str] = []
    # Carbon first
    if "C" in counts:
        parts.append(f"C{counts['C']}" if counts["C"] > 1 else "C")
        del counts["C"]
        # H after C
        if "H" in counts:
            parts.append(f"H{counts['H']}" if counts["H"] > 1 else "H")
            del counts["H"]
    # Rest alphabetically
    for sym in sorted(counts):
        c = counts[sym]
        parts.append(f"{sym}{c}" if c > 1 else sym)

    return "".join(parts)


def build_graph(
    z_values: list[int],
    positions_bohr: list[tuple[float, float, float]],
    tolerance: float = BOND_TOLERANCE,
) -> MolecularGraph:
    """Build a full molecular graph from atomic numbers and positions.

    Parameters
    ----------
    z_values :
        Atomic numbers.
    positions_bohr :
        Atom positions in Bohr (will be converted to Å for bond detection).
    tolerance :
        Bond-detection tolerance factor.

    Returns
    -------
    MolecularGraph with bonds, adjacency, rings, and metadata.
    """
    # Convert Bohr to Å for bond detection
    bohr_to_ang = 0.529177210903
    atoms = [
        AtomInfo(
            index=i,
            z=z,
            symbol=ATOMIC_NUMBER_TO_SYMBOL.get(z, "X"),
            x=pos[0] * bohr_to_ang,
            y=pos[1] * bohr_to_ang,
            z_coord=pos[2] * bohr_to_ang,
        )
        for i, (z, pos) in enumerate(zip(z_values, positions_bohr))
    ]

    bonds = detect_bonds(atoms, tolerance)

    # Build adjacency
    adjacency: list[list[int]] = [[] for _ in atoms]
    for b in bonds:
        adjacency[b.i].append(b.j)
        adjacency[b.j].append(b.i)

    rings = _find_cycles(adjacency)
    formula = _hill_formula(atoms)
    n_electrons = sum(a.z for a in atoms)

    # Heuristic: organic = contains C + H, and C is substantial fraction
    symbols = [a.symbol for a in atoms]
    n_c = symbols.count("C")
    n_h = symbols.count("H")
    is_organic = n_c > 0 and n_h > 0 and (n_c + n_h) > len(atoms) * 0.5

    # Mark aromatic rings
    aromatic_rings = [r for r in rings if _is_aromatic_ring(r, atoms, bonds)]

    return MolecularGraph(
        atoms=atoms,
        bonds=bonds,
        adjacency=adjacency,
        rings=aromatic_rings if aromatic_rings else rings,
        formula=formula,
        n_atoms=len(atoms),
        n_electrons=n_electrons,
        is_organic=is_organic,
    )


def from_molecule(molecule) -> MolecularGraph:
    """Build a MolecularGraph from a vibe-qc ``Molecule`` or list of ``Atom``.

    Parameters
    ----------
    molecule :
        Either a ``vibeqc.Molecule`` (has ``.atoms`` with ``.Z`` and
        ``.position`` in Bohr) or a list of ``Atom`` objects.

    Returns
    -------
    MolecularGraph
    """
    # Handle list of atoms
    if hasattr(molecule, "__iter__") and hasattr(molecule[0], "Z"):
        atoms_list = molecule
    elif hasattr(molecule, "atoms"):
        atoms_list = molecule.atoms
    else:
        raise TypeError(f"Expected Molecule or list of Atom, got {type(molecule)}")

    z_vals = [a.Z for a in atoms_list]
    positions = [(a.position[0], a.position[1], a.position[2]) for a in atoms_list]
    return build_graph(z_vals, positions)


def from_atoms_list(atoms: list[tuple[int, float, float, float]]) -> MolecularGraph:
    """Build from a list of (Z, x, y, z) tuples with positions in Å.

    This is the convenience entry-point for the input library, where
    geometries are typically stored as ``[(Z, x_ang, y_ang, z_ang), ...]``.
    """
    z_vals = [a[0] for a in atoms]
    # Convert Å to Bohr for build_graph (which expects Bohr)
    ang_to_bohr = 1.0 / 0.529177210903
    positions_bohr = [
        (a[1] * ang_to_bohr, a[2] * ang_to_bohr, a[3] * ang_to_bohr) for a in atoms
    ]
    return build_graph(z_vals, positions_bohr)
