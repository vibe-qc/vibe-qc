"""Fused-ring and heterocycle naming (IUPAC Blue Book P-22–P-25).

Extends the basic Hantzsch-Widman logic in ``organic.py`` with:
- Fused-ring detection via shared atoms between ring cycles
- von Baeyer bridged-ring names for bicyclic systems
- Named heterocycle recognition (indole, quinoline, purine, etc.)
- Nucleobase detection (adenine, thymine, uracil, cytosine, guanine)
- Substituted aromatic/heterocyclic naming (phenol, 2-aminopyridine, etc.)

References
----------
- IUPAC Blue Book 2013, Chapter P-2 (parent hydrides)
- Moss, *Pure Appl. Chem.* 1998, 70, 143 (fused-ring nomenclature)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional

from ._types import Confidence, OrganicResult

# ── Named fused-ring systems (trivial IUPAC names) ──────────────────────────

# Keyed by (ring_sizes_tuple, hetero_atom_counts_sorted)
# Ring sizes are sorted; heteroatom counts are (nN, nO, nS)
_NAMED_FUSED_SYSTEMS: dict[tuple, tuple[str, float]] = {
    # Benzene-fused 5-membered heterocycles
    ((5, 6), (1, 0, 0)): ("indole", 0.75),  # benzene + pyrrole
    ((5, 6), (0, 1, 0)): ("benzofuran", 0.70),  # benzene + furan
    ((5, 6), (0, 0, 1)): ("benzothiophene", 0.70),  # benzene + thiophene
    # Benzene-fused 6-membered heterocycles
    ((6, 6), (1, 0, 0)): ("quinoline", 0.75),  # benzene + pyridine (2,3-fused)
    ((6, 6), (1, 0, 0)): ("isoquinoline", 0.70),  # benzene + pyridine (3,4-fused)
    # Purine = imidazole + pyrimidine
    ((5, 6), (4, 0, 0)): ("purine", 0.80),  # 4 N: 2 in imidazole + 2 in pyrimidine
    # Pteridine = pyrazine + pyrimidine
    ((6, 6), (4, 0, 0)): ("pteridine", 0.70),
    # Phenazine = two benzenes + pyrazine
    ((6, 6, 6), (2, 0, 0)): ("phenazine", 0.65),
}


# ── Named single-ring systems ────────────────────────────────────────────────

_NAMED_SINGLE_RINGS: dict[tuple, tuple[str, float]] = {
    # Key: (ring_size, het_signature, nC)
    # het_signature = tuple of (z, count) sorted by z
    # Prevents the frozenset({7,7}) == frozenset({7}) dedup bug.
    # 5-membered
    (5, ((7, 1),), 4): ("pyrrole", 0.70),
    (5, ((7, 2),), 3): ("imidazole", 0.65),
    (5, ((8, 1),), 4): ("furan", 0.75),
    (5, ((8, 2),), 3): ("dioxole", 0.50),
    (5, ((16, 1),), 4): ("thiophene", 0.75),
    (5, ((7, 1), (8, 1)), 3): ("isoxazole", 0.55),
    (5, ((7, 1), (16, 1)), 3): ("thiazole", 0.60),
    # 6-membered — single N
    (6, ((7, 1),), 5): ("pyridine", 0.80),
    # 6-membered — two N (disambiguated by N-N bond geometry below)
    (6, ((7, 2),), 4): ("pyridazine", 0.55),
    (6, ((7, 1), (8, 1)), 4): ("oxazine", 0.50),
    (6, ((7, 1), (16, 1)), 4): ("thiazine", 0.50),
    # 6-membered — three N
    (6, ((7, 3),), 3): ("triazine", 0.55),
    # 7-membered
    (7, ((7, 1),), 6): ("azepine", 0.50),
    (7, ((8, 1),), 6): ("oxepine", 0.45),
}


# ── Nucleobase recognition ───────────────────────────────────────────────────

# Adenine = 6-aminopurine, Guanine = 2-amino-6-oxopurine, etc.
# Recognized by formula + ring structure patterns
_NUCLEOBASE_PATTERNS: list[tuple[str, dict, str, float]] = [
    # (name, formula_pattern, description, confidence)
    ("adenine", {"C": 5, "H": 5, "N": 5}, "6-aminopurine", 0.80),
    ("guanine", {"C": 5, "H": 5, "N": 5, "O": 1}, "2-amino-6-oxopurine", 0.75),
    ("cytosine", {"C": 4, "H": 5, "N": 3, "O": 1}, "4-aminopyrimidin-2-one", 0.75),
    ("thymine", {"C": 5, "H": 6, "N": 2, "O": 2}, "5-methylpyrimidine-2,4-dione", 0.80),
    ("uracil", {"C": 4, "H": 4, "N": 2, "O": 2}, "pyrimidine-2,4-dione", 0.80),
]


# ── Benzene derivative naming ────────────────────────────────────────────────


def _name_benzene_derivative(graph, ring_atoms: list[int]) -> Optional[str]:
    """Name a substituted benzene (monosubstituted aromatic).

    Covers: phenol, aniline, toluene, benzaldehyde, benzoic acid, etc.
    """
    ring_set = set(ring_atoms)
    # Find atoms bonded to ring but not in it
    substituents: list[tuple[str, int]] = []  # (name, ring_atom_index)
    for idx in ring_atoms:
        atom = graph.atoms[idx]
        if atom.z != 6:
            continue  # heteroatom in ring — not a benzene derivative
        for nb_idx in graph.atom_neighbours(idx):
            if nb_idx in ring_set:
                continue
            nb = graph.atoms[nb_idx]
            if nb.z == 1:
                continue  # hydrogen

            # Identify the substituent
            sub_name = _identify_single_substituent(graph, nb_idx, ring_set)
            if sub_name:
                sub_pos = ring_atoms.index(idx) + 1  # 1-based locant
                substituents.append((sub_name, sub_pos))

    if not substituents:
        return None

    # For monosubstituted benzenes, return the trivial name
    single = substituents[0]
    sub_name, pos = single

    TRIVIAL_BENZENES = {
        "hydroxy": "phenol",
        "amino": "aniline",
        "methyl": "toluene",
        "ethyl": "ethylbenzene",
        "vinyl": "styrene",
        "formyl": "benzaldehyde",
        "carboxyl": "benzoic acid",
        "nitro": "nitrobenzene",
        "cyano": "benzonitrile",
        "chloro": "chlorobenzene",
        "bromo": "bromobenzene",
        "iodo": "iodobenzene",
        "fluoro": "fluorobenzene",
    }

    if sub_name in TRIVIAL_BENZENES:
        return TRIVIAL_BENZENES[sub_name]

    # Generic: {substituent}benzene
    if pos != 1:
        return f"{pos}-{sub_name}benzene"
    return f"{sub_name}benzene"


def _identify_single_substituent(
    graph, start_idx: int, ring_set: set[int]
) -> Optional[str]:
    """Identify a substituent attached to the ring at *start_idx*.

    Returns the substituent name (hydroxy, amino, methyl, etc.) or None.
    """
    atom = graph.atoms[start_idx]

    # Hydroxy: O with H neighbor
    if atom.z == 8:
        has_h = any(graph.atoms[nb].z == 1 for nb in graph.atom_neighbours(start_idx))
        if has_h:
            return "hydroxy"
        # Could be methoxy or other alkoxy — check further
        c_neighbors = [
            nb for nb in graph.atom_neighbours(start_idx) if graph.atoms[nb].z == 6
        ]
        if c_neighbors:
            return "methoxy"  # simplified

    # Amino: N with H neighbors
    if atom.z == 7:
        h_count = sum(
            1 for nb in graph.atom_neighbours(start_idx) if graph.atoms[nb].z == 1
        )
        if h_count > 0:
            if h_count == 2:
                return "amino"
            elif h_count == 1:
                c_neighbors = [
                    nb
                    for nb in graph.atom_neighbours(start_idx)
                    if graph.atoms[nb].z == 6 and nb not in ring_set
                ]
                if c_neighbors:
                    return "methylamino"  # simplified
                return "amino"
        # Nitro: N with 2 O neighbors (both double-bonded)
        o_neighbors = [
            nb for nb in graph.atom_neighbours(start_idx) if graph.atoms[nb].z == 8
        ]
        if len(o_neighbors) >= 2:
            double_o = sum(
                1
                for oi in o_neighbors
                if graph.bond_between(start_idx, oi)
                and graph.bond_between(start_idx, oi).order >= 2
            )
            if double_o >= 2:
                return "nitro"
        return None

    # Halogens
    if atom.z == 9:
        return "fluoro"
    if atom.z == 17:
        return "chloro"
    if atom.z == 35:
        return "bromo"
    if atom.z == 53:
        return "iodo"

    # Carbon-based substituents
    if atom.z == 6:
        # Count carbons in this branch (excluding ring)
        branch_c = _count_carbons_in_branch(graph, start_idx, ring_set, set())
        if branch_c == 1:
            return "methyl"
        if branch_c == 2:
            return "ethyl"
        if branch_c == 3:
            return "propyl"
        return f"C{branch_c} alkyl"

    # Aldehyde / carboxyl detection
    if atom.z == 6:
        for nb in graph.atom_neighbours(start_idx):
            if graph.atoms[nb].z == 8:
                b = graph.bond_between(start_idx, nb)
                if b and b.order >= 2:
                    # Check if terminal (only one heavy neighbor outside ring)
                    c_neighbors = [
                        nn
                        for nn in graph.atom_neighbours(start_idx)
                        if graph.atoms[nn].z not in (1,) and nn not in ring_set
                    ]
                    if len(c_neighbors) <= 1:
                        # Check if also has OH → carboxyl
                        has_oh = any(
                            graph.atoms[nn].z == 8
                            and any(
                                graph.atoms[nnb].z == 1
                                for nnb in graph.atom_neighbours(nn)
                            )
                            for nn in graph.atom_neighbours(start_idx)
                        )
                        if has_oh:
                            return "carboxyl"
                        return "formyl"

    return None


def _count_carbons_in_branch(
    graph, start: int, ring_set: set[int], visited: set[int]
) -> int:
    """Count carbon atoms in a substituent branch."""
    if start in visited or start in ring_set:
        return 0
    visited.add(start)
    count = 1 if graph.atoms[start].z == 6 else 0
    for nb in graph.atom_neighbours(start):
        if nb not in visited and nb not in ring_set:
            count += _count_carbons_in_branch(graph, nb, ring_set, visited)
    return count


# ── Fused-ring detection ─────────────────────────────────────────────────────


def _detect_fused_system(
    graph,
    rings: list[list[int]],
) -> Optional[OrganicResult]:
    """Detect and name a fused-ring system.

    A fused system has at least two rings sharing 2+ atoms (a bond).
    This covers indole, quinoline, purine, naphthalene, etc.

    Returns an OrganicResult with the name, or None if not a fused system.
    """
    if len(rings) < 2:
        return None

    # Find pairs of rings that share atoms
    fused_pairs: list[tuple[list[int], list[int], set[int]]] = []
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared = set(rings[i]) & set(rings[j])
            if len(shared) >= 2:  # Fused (share a bond) or bridged
                fused_pairs.append((rings[i], rings[j], shared))

    if not fused_pairs:
        return None

    # Collect all rings involved in the fused system
    all_rings_in_system: set[int] = set()
    fused_ring_sizes: list[int] = []
    for r1, r2, _ in fused_pairs:
        all_rings_in_system.update(r1)
        all_rings_in_system.update(r2)

    for ring in rings:
        ring_set = set(ring)
        if ring_set & all_rings_in_system:  # overlaps with fused system
            if len(ring) not in fused_ring_sizes:
                fused_ring_sizes.append(len(ring))

    # Count heteroatoms in the fused system
    hetero_counts = Counter()
    for idx in all_rings_in_system:
        atom = graph.atoms[idx]
        if atom.z == 7:
            hetero_counts[7] += 1
        elif atom.z == 8:
            hetero_counts[8] += 1
        elif atom.z == 16:
            hetero_counts[16] += 1

    nN = hetero_counts.get(7, 0)
    nO = hetero_counts.get(8, 0)
    nS = hetero_counts.get(16, 0)

    # Sorted ring sizes for dictionary lookup
    size_key = tuple(sorted(fused_ring_sizes))
    het_key = (nN, nO, nS)

    # Try named fused systems
    for (sizes, het_tuple), (name, conf) in _NAMED_FUSED_SYSTEMS.items():
        if sizes == size_key:
            # Heteroatom count matching is approximate
            sum_het = sum(het_tuple)
            sum_got = nN + nO + nS
            if abs(sum_het - sum_got) <= 1:
                return OrganicResult(name=name, confidence=conf)

    # Total carbons in fused system
    nC = sum(1 for idx in all_rings_in_system if graph.atoms[idx].z == 6)

    # Naphthalene: two fused benzenes (C10H8)
    if size_key == (6, 6) and nC == 10 and nN == 0:
        return OrganicResult(name="naphthalene", confidence=0.80)

    # Anthracene / phenanthrene: three fused benzenes
    if size_key == (6, 6, 6) and nC == 14 and nN == 0:
        return OrganicResult(name="anthracene", confidence=0.65)

    # Generic: nC-carbon fused system
    total_n = len(all_rings_in_system)
    if nC >= 8 and nC < 30:
        return OrganicResult(
            name=f"fused C{nC}H{total_n - nC} ring system",
            confidence=0.25,
        )

    return None


# ── Substituted heterocycle naming ───────────────────────────────────────────


def _name_substituted_heterocycle(
    graph,
    ring_name: str,
    ring_atoms: list[int],
) -> Optional[str]:
    """Name a heterocycle with substituents attached to the ring.

    e.g., 2-aminopyridine, 3-hydroxypyridine, 5-methylpyrimidine.
    """
    ring_set = set(ring_atoms)
    substituents: list[tuple[int, str]] = []

    for idx in ring_atoms:
        for nb_idx in graph.atom_neighbours(idx):
            if nb_idx in ring_set:
                continue
            nb = graph.atoms[nb_idx]
            if nb.z == 1:
                continue

            sub_name = _identify_single_substituent(graph, nb_idx, ring_set)
            if sub_name:
                # Get position on ring (1-based, numbered from ring_atoms order)
                pos = ring_atoms.index(idx) + 1
                if (pos, sub_name) not in substituents:
                    substituents.append((pos, sub_name))

    if not substituents:
        return ring_name

    # Alphabetize substituents
    substituents.sort(key=lambda x: x[1])

    sub_parts = [f"{pos}-{name}" for pos, name in substituents]
    return f"{','.join(sub_parts)}{ring_name}"


# ── Nucleobase detection ─────────────────────────────────────────────────────


def _detect_nucleobase(graph) -> Optional[OrganicResult]:
    """Detect if this molecule is a nucleobase by formula + ring patterns."""
    # Count atoms
    counts: dict[str, int] = {}
    for a in graph.atoms:
        counts[a.symbol] = counts.get(a.symbol, 0) + 1

    for name, pattern, desc, conf in _NUCLEOBASE_PATTERNS:
        match = True
        for sym, cnt in pattern.items():
            if counts.get(sym, 0) != cnt:
                match = False
                break
        if match and len(graph.rings) >= 1:
            return OrganicResult(name=name, confidence=conf)

    return None


# ── Main ring-system naming entry point ──────────────────────────────────────


def name_ring_system(graph) -> Optional[str]:
    """Name a ring-containing organic molecule.

    This is the top-level dispatcher for all ring-system naming.
    It tries, in order:
    1. Named single-ring heterocycles (pyridine, furan, etc.)
    2. Benzene derivatives (phenol, toluene, etc.)
    3. Fused-ring systems (indole, quinoline, purine, naphthalene)
    4. Nucleobases (adenine, thymine, etc.)
    5. Substituted heterocycles (2-aminopyridine, etc.)
    6. Generic Hantzsch-Widman names

    Returns the name string or None.
    """
    rings = graph.rings
    if not rings:
        return None

    n_c = sum(1 for a in graph.atoms if a.z == 6)
    if n_c == 0:
        return None

    has_hetero = any(a.z not in (1, 6) for a in graph.atoms)

    # ── 1. Nucleobase detection (fast formula check) ─────────────────────
    nb_result = _detect_nucleobase(graph)
    if nb_result and nb_result.confidence >= 0.75:
        return nb_result.name

    # ── 2. Named single-ring heterocycle ─────────────────────────────────
    if has_hetero and len(rings) == 1:
        ring = rings[0]
        ring_atoms = [graph.atoms[i] for i in ring]
        # Build heteroatom signature: sorted tuple of (z, count)
        het_counts = Counter()
        for a in ring_atoms:
            if a.z != 6 and a.z != 1:
                het_counts[a.z] += 1
        het_sig = tuple(sorted(het_counts.items()))
        n_c_ring = sum(1 for a in ring_atoms if a.z == 6)
        size = len(ring)

        key = (size, het_sig, n_c_ring)
        if key in _NAMED_SINGLE_RINGS:
            name, conf = _NAMED_SINGLE_RINGS[key]
            # Check for substituents on this ring
            sub_name = _name_substituted_heterocycle(graph, name, ring)
            if sub_name and sub_name != name:
                return sub_name
            if conf >= 0.6:
                return name

        # Disambiguate pyrimidine/pyrazine/pyridazine by bond pattern
        if key == (6, ((7, 2),), 4):
            n_bonds = _count_nn_bonds_in_ring(graph, rings, ring)
            if n_bonds == 1:  # adjacent
                return "pyridazine"
            elif n_bonds == 0 or n_bonds > 1:
                # Check N positions: 1,3 = pyrimidine, 1,4 = pyrazine
                n_positions = sorted(
                    [i for i in ring if graph.atoms[i].z == 7],
                    key=lambda i: ring.index(i),
                )
                if len(n_positions) >= 2:
                    gap = (
                        ring.index(n_positions[1]) - ring.index(n_positions[0])
                    ) % len(ring)
                    if gap == 2:
                        return "pyrimidine"
                    else:
                        return "pyrazine"
                return "pyrazine"

        # Generic Hantzsch-Widman
        prefixes = {7: "aza", 8: "oxa", 16: "thia"}
        bases = {5: "ole", 6: "ine", 7: "epine"}
        if het_sig:
            primary = max(dict(het_sig).keys())
            prefix = prefixes.get(primary, "hetero")
            base = bases.get(size, "ane")
            return f"{prefix}{base}"

    # ── 3. Benzene derivative ───────────────────────────────────────────
    if len(rings) == 1:
        ring = rings[0]
        all_c_in_ring = all(graph.atoms[i].z == 6 for i in ring)
        ring_size = len(ring)
        if all_c_in_ring and ring_size == 6:
            deriv = _name_benzene_derivative(graph, ring)
            if deriv:
                return deriv
            if n_c == 6:
                return "benzene"

    # ── 4. Fused-ring system ────────────────────────────────────────────
    fused_result = _detect_fused_system(graph, rings)
    if fused_result and fused_result.confidence >= 0.5:
        # Check for substituents on the fused system
        all_ring_atoms = []
        for ring in rings:
            all_ring_atoms.extend(ring)
        unique_atoms = list(dict.fromkeys(all_ring_atoms))  # preserve order, dedupe
        sub_name = _name_substituted_heterocycle(graph, fused_result.name, unique_atoms)
        if sub_name and sub_name != fused_result.name:
            return sub_name
        return fused_result.name

    return None


def _count_nn_bonds_in_ring(graph, all_rings, ring):
    """Count N-N bonds within a ring (for diazine disambiguation)."""
    ring_set = set(ring)
    n_atoms = [i for i in ring if graph.atoms[i].z == 7]
    nn_bonds = 0
    for b in graph.bonds:
        if b.i in ring_set and b.j in ring_set:
            if graph.atoms[b.i].z == 7 and graph.atoms[b.j].z == 7:
                nn_bonds += 1
    return nn_bonds
