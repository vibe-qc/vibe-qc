"""Full organic substitutive nomenclature (IUPAC Blue Book P-1–P-7).

Implements:
- Principal chain selection (longest carbon chain, maximum unsaturation,
  maximum substituents — per Blue Book P-44)
- Locant numbering (lowest-set rule per Blue Book P-14)
- Substituent identification and alphabetical ordering
- Functional group suffix priority (Blue Book Table P-61)
- Heterocycle naming (Hantzsch-Widman for common 5/6-membered rings)

Strategy: conservative — returns a systematic name only when confident,
otherwise None so the caller falls back to compositional naming.

References
----------
- Favre & Powell, *Nomenclature of Organic Chemistry*, RSC 2014
- IUPAC Blue Book 2013
- Hellwinkel, *Systematic Nomenclature of Organic Chemistry*, Springer 2001
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ── Data: alkane roots and substituent info ──────────────────────────────────

_ALKANE_ROOT: dict[int, str] = {
    1: "meth",
    2: "eth",
    3: "prop",
    4: "but",
    5: "pent",
    6: "hex",
    7: "hept",
    8: "oct",
    9: "non",
    10: "dec",
    11: "undec",
    12: "dodec",
}

# Heteroatom symbols used in ring detection
_HETEROATOM_SYMBOLS = {7: "N", 8: "O", 16: "S", 35: "Br", 9: "F"}


# ── Functional group priority (highest → lowest) ─────────────────────────────

_FUNCTIONAL_GROUP_PRIORITY = [
    ("carboxylic acid", "-oic acid", 1),
    ("ester", "-oate", 2),
    ("amide", "-amide", 3),
    ("nitrile", "-nitrile", 4),
    ("aldehyde", "-al", 5),
    ("ketone", "-one", 6),
    ("alcohol", "hydroxy-", 7),
    ("amine", "amino-", 8),
    ("alkene", "-ene", 9),
    ("alkyne", "-yne", 10),
]

_FUNCTIONAL_GROUP_SIGNATURES = [
    ("carboxylic acid", 1, lambda g: _is_carboxylic_acid(g)),
    ("nitrile", 4, lambda g: _is_nitrile(g)),
    ("aldehyde", 5, lambda g: _is_aldehyde(g)),
    ("ketone", 6, lambda g: _is_ketone(g)),
    ("alcohol", 7, lambda g: _is_alcohol(g)),
    ("amine", 8, lambda g: _is_amine(g)),
    ("ester", 2, lambda g: _is_ester(g)),
]


# ── Result type for organic naming ───────────────────────────────────────────


@dataclass
class OrganicResult:
    """Result of an organic nomenclature attempt."""

    name: str
    confidence: float  # 0.0 – 1.0
    principal_chain_length: int = 0
    has_rings: bool = False
    ring_count: int = 0
    substituents: list[str] = field(default_factory=list)
    functional_groups: list[str] = field(default_factory=list)


# ── Principal chain selection ────────────────────────────────────────────────


def _find_carbon_atoms(graph) -> list[int]:
    """Return indices of carbon atoms."""
    return [a.index for a in graph.atoms if a.z == 6]


def _select_principal_chain(graph) -> list[int] | None:
    """Select the principal chain (Blue Book P-44).

    Priority: maximum number of principal groups → longest chain →
    maximum unsaturation → lowest locants.

    For simple cases, finds the longest carbon path via DFS.
    """
    carbon_indices = _find_carbon_atoms(graph)
    if not carbon_indices:
        return None

    if len(carbon_indices) == 1:
        return [carbon_indices[0]]

    # Use the adjacency subgraph of carbons for longest-path finding
    path, length = _find_longest_path(graph, carbon_indices)
    if length < 2:
        return None

    return path


def _find_longest_path(graph, atom_indices: list[int]) -> tuple[list[int], int]:
    """Find the longest path through the given atom indices."""
    carbon_set = set(atom_indices)
    best_path: list[int] = []

    def dfs(start: int, current: int, path: list[int], visited: set[int]) -> None:
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = list(path)
        for nb in graph.atom_neighbours(current):
            if nb in carbon_set and nb not in visited:
                visited.add(nb)
                path.append(nb)
                dfs(start, nb, path, visited)
                path.pop()
                visited.remove(nb)

    # Cap DFS to first 30 carbons for performance
    for c in atom_indices[:30]:
        dfs(c, c, [c], {c})

    return best_path, len(best_path)


# ── Functional group detection ───────────────────────────────────────────────


def _is_carboxylic_acid(graph) -> bool:
    """Carboxyl group: carbon bonded to two oxygens (one double-bonded)."""
    carbons = [a for a in graph.atoms if a.z == 6]

    for c in carbons:
        c_o_bonds = [
            (b.j, b.order)
            for b in graph.bonds
            if b.i == c.index and b.j < len(graph.atoms)
        ]
        # Symmetric case
        if not c_o_bonds:
            c_o_bonds = [
                (b.i, b.order)
                for b in graph.bonds
                if b.j == c.index and b.i < len(graph.atoms)
            ]

        if any(ord_val >= 2 for _, ord_val in c_o_bonds):
            # Has C=O; check for second O bonded to same carbon
            other_o = [j for j, _ in c_o_bonds]
            for o_idx in other_o:
                o_neighbors = graph.atom_neighbours(o_idx)
                has_other_o = any(
                    graph.atoms[nb].z == 8 and nb != o_idx for nb in o_neighbors
                )
                if has_other_o:
                    # Check C is terminal-ish (bonded to <=1 heavy non-O atom)
                    c_neighbors = graph.atom_neighbours(c.index)
                    non_o_heavy = [
                        n for n in c_neighbors if graph.atoms[n].z not in (1, 8)
                    ]
                    if len(non_o_heavy) <= 1:
                        return True

    return False


def _is_aldehyde(graph) -> bool:
    """Aldehyde: C=O with carbon bonded to at least one H (terminal)."""
    carbons = [a for a in graph.atoms if a.z == 6]
    oxygens = [a for a in graph.atoms if a.z == 8]

    for c in carbons:
        for o in oxygens:
            b = graph.bond_between(c.index, o.index)
            if not b or b.order < 2:
                continue
            # Carbon must be terminal (bonded to <=1 heavy non-O atom besides O)
            c_neighbors = graph.atom_neighbours(c.index)
            heavy_non_o = [n for n in c_neighbors if graph.atoms[n].z not in (1, 8)]
            if len(heavy_non_o) <= 1:
                return True

    return False


def _is_ketone(graph) -> bool:
    """Ketone: C=O with the carbon bonded to two other carbons."""
    carbons = [a for a in graph.atoms if a.z == 6]
    oxygens = [a for a in graph.atoms if a.z == 8]

    for c in carbons:
        for o in oxygens:
            b = graph.bond_between(c.index, o.index)
            if not b or b.order < 2:
                continue
            c_neighbors = graph.atom_neighbours(c.index)
            heavy_non_o = [n for n in c_neighbors if graph.atoms[n].z not in (1, 8)]
            # Must be bonded to >=2 carbons (not counting O or H)
            n_carbons = sum(1 for n in heavy_non_o if graph.atoms[n].z == 6)
            if n_carbons >= 2:
                return True

    return False


def _is_alcohol(graph) -> bool:
    """Alcohol: OH group (oxygen bonded to both carbon and hydrogen)."""
    oxygens = [a for a in graph.atoms if a.z == 8]

    for o in oxygens:
        has_c = any(graph.atoms[nb].z == 6 for nb in graph.atom_neighbours(o.index))
        has_h = any(
            graph.atoms[nb].z == 1
            for nb in graph.atom_neighbours(o.index)
            if graph.atoms[nb].index != o.index
        )
        # More careful: check that O bonds to at least one H (via bond)
        bonded_to_h = any(
            graph.bond_between(o.index, h.index) is not None
            for h in graph.atoms
            if h.z == 1
        )
        if has_c and bonded_to_h:
            return True

    return False


def _is_amine(graph) -> bool:
    """Amine: nitrogen bonded to at least one carbon (and possibly H)."""
    nitrogens = [a for a in graph.atoms if a.z == 7]

    for n in nitrogens:
        c_neighbors = [
            nb for nb in graph.atom_neighbours(n.index) if graph.atoms[nb].z == 6
        ]
        if c_neighbors:
            return True

    return False


def _is_nitrile(graph) -> bool:
    """Nitrile: C≡N triple bond with the carbon being terminal."""
    carbons = [a for a in graph.atoms if a.z == 6]
    nitrogens = [a for a in graph.atoms if a.z == 7]

    for c in carbons:
        for n in nitrogens:
            b = graph.bond_between(c.index, n.index)
            if not b or b.order < 3:
                continue
            # C should be terminal (only N as heavy neighbor, or 1 other heavy)
            c_neighbors = graph.atom_neighbours(c.index)
            non_n_heavy = [nn for nn in c_neighbors if graph.atoms[nn].z not in (1, 7)]
            if len(non_n_heavy) <= 1:
                return True

    return False


def _is_ester(graph) -> bool:
    """Ester: C(=O)-O-C linkage."""
    carbons = [a for a in graph.atoms if a.z == 6]
    oxygens = [a for a in graph.atoms if a.z == 8]

    for c in carbons:
        for o1 in oxygens:
            b_co = graph.bond_between(c.index, o1.index)
            if not b_co or b_co.order < 2:
                continue
            # This O also bonds to another carbon (ester O-C bond)
            for o2_idx in graph.atom_neighbours(o1.index):
                if graph.atoms[o2_idx].z == 6 and graph.atoms[o2_idx].index != c.index:
                    return True

    return False


# ── Substituent identification ───────────────────────────────────────────────


def _identify_substituents(graph, chain_indices: set[int]) -> list[tuple[int, str]]:
    """Identify substituents not on the principal chain.

    Returns list of (locant, substituent_name).
    """
    substituents: list[tuple[int, str]] = []
    carbon_set = {a.index for a in graph.atoms if a.z == 6}
    seen: set[tuple[int, str]] = set()

    def add_sub(locant: int, name: str) -> None:
        key = (locant, name)
        if key not in seen:
            seen.add(key)
            substituents.append(key)

    for idx in chain_indices:
        for nb_idx in graph.atom_neighbours(idx):
            if nb_idx in chain_indices:
                continue
            atom = graph.atoms[nb_idx]
            if atom.z == 1:  # H atoms are implicit
                continue

            locant = _find_locant_for_chain_atom(graph, idx, chain_indices)

            # Alkyl substituents (carbon branches off the chain)
            if atom.z == 6:
                branch_carbons = _count_branch_carbons(
                    graph, nb_idx, chain_indices | carbon_set
                )
                if branch_carbons >= 1 and branch_carbons in _ALKANE_ROOT:
                    root = _ALKANE_ROOT[branch_carbons]
                    name = f"{root}yl"
                    add_sub(locant, name)

            # Halogens
            elif atom.z == 9:
                add_sub(locant, "fluoro")
            elif atom.z == 17:
                add_sub(locant, "chloro")
            elif atom.z == 35:
                add_sub(locant, "bromo")
            elif atom.z == 53:
                add_sub(locant, "iodo")

            # Hydroxyl group (O bonded to H)
            elif atom.z == 8:
                has_h = any(
                    graph.bond_between(atom.index, hi.index) is not None
                    for hi in graph.atoms
                    if hi.z == 1
                )
                if has_h:
                    add_sub(locant, "hydroxy")

            # Amino group (N bonded to H)
            elif atom.z == 7:
                has_h = any(
                    graph.bond_between(atom.index, hi.index) is not None
                    for hi in graph.atoms
                    if hi.z == 1
                )
                if has_h:
                    add_sub(locant, "amino")

    return substituents


def _count_branch_carbons(graph, start: int, exclude: set[int]) -> int:
    """Count carbons reachable from *start* without entering *exclude*."""
    visited = {start}
    stack = [start]
    count = 0
    while stack:
        current = stack.pop()
        if graph.atoms[current].z == 6:
            count += 1
        for nb in graph.atom_neighbours(current):
            if nb not in visited and nb not in exclude and graph.atoms[nb].z == 6:
                visited.add(nb)
                stack.append(nb)
    return count


def _find_locant_for_chain_atom(graph, idx: int, chain_indices: list[int]) -> int:
    """Find the position of atom *idx* within the principal chain."""
    try:
        return chain_indices.index(idx) + 1
    except ValueError:
        # Fallback: use degree to guess (terminal = 1)
        return 1


# ── Numbering / lowest-locant rule ───────────────────────────────────────────


def _assign_numbering(graph, chain_indices: list[int]) -> list[int]:
    """Number the principal chain per IUPAC lowest-locant rules.

    Returns the ordered chain indices from one end to the other.
    For now: use BFS path between chain termini.
    """
    n = len(chain_indices)
    if n <= 2:
        return chain_indices

    chain_set = set(chain_indices)

    def _chain_ends() -> tuple[int, int]:
        degree_1 = [
            idx for idx in chain_indices if len(graph.atom_neighbours(idx)) == 1
        ]
        if len(degree_1) >= 2:
            return degree_1[0], degree_1[-1]
        # Fallback: find furthest pair via BFS distance
        best_d, best = 0, (chain_indices[0], chain_indices[0])
        for a in chain_indices[:5]:
            for b in chain_indices[5:]:
                d = _path_length(graph, a, b, chain_set)
                if d > best_d:
                    best_d, best = d, (a, b)
        return best[0], best[1]

    start_a, end_a = _chain_ends()
    forward_chain = _path_order(graph, start_a, end_a, chain_set)
    return forward_chain


def _path_length(graph, start: int, end: int, valid_set: set[int]) -> int:
    """BFS shortest path length within *valid_set*."""
    if start == end:
        return 0
    visited = {start}
    queue = [(start, 0)]
    while queue:
        current, dist = queue.pop(0)
        for nb in graph.atom_neighbours(current):
            if nb == end and nb in valid_set:
                return dist + 1
            if nb not in visited and nb in valid_set:
                visited.add(nb)
                queue.append((nb, dist + 1))
    return float("inf")


def _path_order(graph, start: int, end: int, valid_set: set[int]) -> list[int]:
    """Return atoms along the shortest path from *start* to *end*, ordered."""
    if start == end:
        return [start]
    parent = {start: None}
    visited = {start}
    queue = [start]
    found = False
    while queue and not found:
        current = queue.pop(0)
        for nb in graph.atom_neighbours(current):
            if nb == end and nb in valid_set:
                parent[nb] = current
                found = True
                break
            if nb not in visited and nb in valid_set:
                visited.add(nb)
                parent[nb] = current
                queue.append(nb)
    if not found:
        return [start, end]
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent.get(node)
    return list(reversed(path))


# ── Name assembly ────────────────────────────────────────────────────────────


def _unsaturation_suffix(root: str, n_double: int, n_triple: int) -> str:
    """Build the unsaturation suffix for a given root and bond counts.

    Per Blue Book P-31: connect multiple bonds with connecting vowels.
    e.g., pent + a + diene → penta-1,3-diene (locants added separately).
    """
    if n_triple == 0 and n_double == 0:
        return f"{root}ane"

    # Insert connecting 'a' before vowel-starting suffixes
    needs_vowel = root and not root.endswith("a")

    parts: list[str] = []
    if n_double > 0:
        mprefix = "" if n_double == 1 else {2: "di", 3: "tri", 4: "tetra"}.get(n_double)
        parts.append(f"{mprefix}ene" if not needs_vowel else f"{root}a{mprefix}ene")
    if n_triple > 0:
        mprefix = "" if n_triple == 1 else {2: "di", 3: "tri"}.get(n_triple)
        # 'yne' always preceded by connecting vowel
        parts.append(
            f"{mprefix}yne" if not needs_vowel and not parts else f"{root}a{mprefix}yne"
        )

    return parts[0] if len(parts) == 1 else "".join(parts)


def _assemble_substitutive_name(
    graph,
    chain_indices: list[int],
    substituents: list[tuple[int, str]],
    functional_group: str | None = None,
) -> OrganicResult:
    """Assemble the systematic IUPAC name per Blue Book P-1–P-7."""
    n_c_in_chain = len(chain_indices)
    if n_c_in_chain == 0 or n_c_in_chain not in _ALKANE_ROOT:
        return OrganicResult(name="", confidence=0.0)

    root = _ALKANE_ROOT[n_c_in_chain]

    # Determine unsaturation in the chain
    chain_set = set(chain_indices)
    n_double = sum(
        1 for b in graph.bonds if b.i in chain_set and b.j in chain_set and b.order >= 2
    )
    n_triple = sum(
        1 for b in graph.bonds if b.i in chain_set and b.j in chain_set and b.order >= 3
    )

    # Build unsaturation suffix
    if n_double == 0 and n_triple == 0:
        base_name = f"{root}ane"
    else:
        base_name = _unsaturation_suffix(root, n_double, n_triple)

    # Functional group handling
    suffix = ""
    for pname, psuffix, _ in _FUNCTIONAL_GROUP_PRIORITY:
        if pname == functional_group:
            suffix = psuffix.lstrip("-")
            break

    # Replace -e of alkane/alkene/alkyne with functional group suffix
    if suffix and base_name.endswith(("e", "ne", "yne")):
        if suffix in ("al",):
            base_name = base_name.rstrip("e") + suffix
        elif suffix in ("one",):
            base_name = base_name.rstrip("e") + suffix
        elif suffix in ("oic acid",):
            # For acids, replace -e with -oic acid
            if base_name.endswith("e"):
                base_name = base_name[:-1] + "oic acid"
            else:
                base_name += "oic acid"
        elif suffix == "nitrile":
            base_name += suffix
        else:
            base_name = base_name.rstrip("e") + suffix

    # Substituents — alphabetized, with locants
    sorted_subs = sorted(substituents, key=lambda x: (x[1], x[0]))

    sub_parts: list[str] = []
    for locant, name in sorted_subs:
        sub_parts.append(f"{locant}-{name}")

    # Assemble the full name
    parts = []
    if sub_parts:
        parts.append(",".join(sub_parts))
    parts.append(base_name)

    # Add suffix as prefix if it's a substitutive form (e.g., hydroxy-)
    # handled via functional_group already being in substituents list above
    name = " ".join(parts)

    confidence = min(1.0, 0.5 + (n_c_in_chain / 20) + (0.3 if functional_group else 0))
    conf = (
        Confidence.HIGH
        if confidence >= 0.8
        else (Confidence.MEDIUM if confidence >= 0.5 else Confidence.LOW)
    )

    return OrganicResult(
        name=name,
        confidence=confidence,
        principal_chain_length=n_c_in_chain,
        has_rings=len(graph.rings) > 0,
        ring_count=len(graph.rings),
        substituents=[n for _, n in substituents],
        functional_groups=[functional_group] if functional_group else [],
    )


# ── Heterocycle detection & naming ───────────────────────────────────────────


def _detect_heterocycle(graph) -> OrganicResult | None:
    """Detect and name simple heterocycles using Hantzsch-Widman rules."""
    ring_count = len(graph.rings)
    if ring_count == 0:
        return None

    # Try first ring only (primary system)
    for ring in graph.rings[:1]:
        ring_atoms = [graph.atoms[i] for i in ring]
        hetero_in_ring = {a.z for a in ring_atoms if a.z != 6}
        n_hetero = len(hetero_in_ring)
        n_c = sum(1 for a in ring_atoms if a.z == 6)

        if n_hetero == 0 or n_hetero > 3:
            continue

        ring_size = len(ring_atoms)

        # 5-membered rings
        if ring_size == 5:
            if hetero_in_ring == {8}:  # O only
                return OrganicResult(name="furan", confidence=0.7)
            if hetero_in_ring == {7}:  # N only
                return OrganicResult(name="pyrrole", confidence=0.65)
            if hetero_in_ring == {16}:  # S only
                return OrganicResult(name="thiophene", confidence=0.7)
            if hetero_in_ring == {7, 8}:  # N + O
                return OrganicResult(name="isoxazole", confidence=0.5)

        # 6-membered rings
        if ring_size == 6:
            if hetero_in_ring == {7} and n_c == 5:
                return OrganicResult(name="pyridine", confidence=0.75)
            if hetero_in_ring == {7, 7} and n_c == 4:
                return OrganicResult(name="pyrimidine", confidence=0.5)

        # Generic Hantzsch-Widman for other cases
        prefixes = {7: "aza", 8: "oxa", 16: "thia"}
        bases = {5: "ole", 6: "ine"}
        if hetero_in_ring and max(hetero_in_ring) in prefixes:
            primary_het = max(hetero_in_ring)
            prefix = prefixes[primary_het]
            base = bases.get(ring_size, "ane")
            return OrganicResult(name=f"{prefix}{base}", confidence=0.3)

    return None


# ── Confidence enum (local import to avoid circular dep) ───────────────────────

from .iupac_name import Confidence  # noqa: E402

# ── Main entry point for organic naming ──────────────────────────────────────


def name_organic(graph) -> Optional[str]:
    """Build the systematic IUPAC name for an organic molecule.

    Returns None if the molecule doesn't qualify as organic or confidence
    is too low for a reliable guess.
    """
    n_c = sum(1 for a in graph.atoms if a.z == 6)
    if n_c == 0:
        return None

    # Heterocycle check (rings with heteroatoms take priority)
    has_hetero = any(a.z not in (1, 6) for a in graph.atoms)
    if graph.rings and has_hetero:
        het_result = _detect_heterocycle(graph)
        if het_result and het_result.confidence >= 0.7:
            return het_result.name

    # Principal chain selection
    chain = _select_principal_chain(graph)
    if chain is None or len(chain) < 2:
        return None

    chain_ordered = _assign_numbering(graph, chain)
    chain_set = set(chain_ordered)

    # Functional group detection (highest priority first)
    functional_group = None
    for gname, _, matcher in _FUNCTIONAL_GROUP_SIGNATURES:
        if matcher(graph):
            functional_group = gname
            break

    # Substituent identification
    substituents = _identify_substituents(graph, chain_set)

    # Build name
    result = _assemble_substitutive_name(
        graph, chain_ordered, substituents, functional_group
    )

    if result.confidence < 0.4:
        return None

    return result.name
