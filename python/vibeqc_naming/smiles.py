"""SMILES generation — pre-scanned ring edges, aromatic perception."""

from __future__ import annotations
from typing import Optional

def graph_to_smiles(graph) -> Optional[str]:
    if graph.n_atoms == 0:
        return None
    heavy = [i for i in range(graph.n_atoms) if graph.atoms[i].z != 1]
    if not heavy:
        return None
    remap = {old: new for new, old in enumerate(heavy)}
    n = len(heavy)
    adj = [[] for _ in range(n)]
    orders = {}
    for b in graph.bonds:
        if b.i in remap and b.j in remap:
            ni, nj = remap[b.i], remap[b.j]
            adj[ni].append(nj); adj[nj].append(ni)
            orders[(min(ni,nj), max(ni,nj))] = b.order
    zs = [graph.atoms[i].z for i in heavy]

    terminals = [i for i in range(n) if len(adj[i]) == 1]
    start = min(terminals, key=lambda i: zs[i]) if terminals else 0

    # Use graph.rings to find ring-closing bonds.
    # Each ring contributes one ring-closure bond: the edge from the
    # last atom in the DFS traversal back to the first atom in that ring.
    ring_next = 1
    ring_digit = {}  # atom -> list of digits
    seen_rings = set()
    for ring in graph.rings:
        rs = set(ring)
        # Map ring atoms to heavy indices
        ring_heavy = [remap[i] for i in ring if i in remap]
        if len(ring_heavy) < 3:
            continue
        # The ring closure is between the two atoms that are adjacent
        # in the ring but would be farthest apart in the DFS from start.
        # Simplification: use the smallest-index pair as the closure.
        a, b = min(ring_heavy), max(ring_heavy)
        # Only keep rings where a-b are actually adjacent in the ring
        if b in adj[a]:
            key = (a, b)
            if key not in seen_rings:
                seen_rings.add(key)
                d = ring_next
                ring_next += 1
                ring_digit.setdefault(a, []).append(d)
                ring_digit.setdefault(b, []).append(d)

    visited = [False] * n
    parts: list[str] = []

    def dfs(idx: int, parent: int | None):
        visited[idx] = True
        z = zs[idx]
        aromatic = _aromatic(graph, heavy[idx])
        if aromatic:
            syms = {6: 'c', 7: 'n', 8: 'o', 16: 's'}
        else:
            syms = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
        parts.append(syms.get(z, f'[{graph.atoms[heavy[idx]].symbol}]'))

        # Emit ring digits at this atom
        if idx in ring_digit:
            for d in ring_digit[idx]:
                parts.append(str(d) if d < 10 else f'%{d}')

        nbrs = adj[idx]
        unvisited = [n for n in nbrs if not visited[n]]
        unvisited.sort(key=lambda n: (len(adj[n]), zs[n]))
        if unvisited:
            first = unvisited[0]; rest = unvisited[1:]
            order = orders.get((min(idx,first), max(idx,first)), 1)
            if order == 2: parts.append('=')
            elif order == 3: parts.append('#')
            dfs(first, idx)
            for child in rest:
                if not visited[child]:
                    parts.append('(')
                    order = orders.get((min(idx,child), max(idx,child)), 1)
                    if order == 2: parts.append('=')
                    elif order == 3: parts.append('#')
                    dfs(child, idx)
                    parts.append(')')

    dfs(start, None)
    return ''.join(parts)

def _aromatic(graph, atom_idx):
    """Detect if an atom is in an aromatic ring.

    Uses two heuristics:
    1. Bond-order pattern: >=3 double bonds in a 6-ring, >=2 in a 5-ring.
    2. Bond-distance uniformity: if all ring bonds are between 1.35-1.45 A
       (typical aromatic range) and no bonds are clearly single (>1.48 A),
       treat as aromatic. This catches benzene when bond orders from the
       distance estimator all read as single.
    """
    for ring in graph.rings:
        if atom_idx in ring:
            rs = set(ring)
            nd = sum(1 for b in graph.bonds if b.i in rs and b.j in rs and b.order >= 2)
            sz = len(ring)
            if (sz == 6 and nd >= 3) or (sz == 5 and nd >= 2):
                return True
            # Fallback: all ring bonds in aromatic distance range
            ring_bonds = [b for b in graph.bonds if b.i in rs and b.j in rs]
            if ring_bonds and sz in (5, 6):
                all_in_range = all(1.30 <= b.distance <= 1.50 for b in ring_bonds)
                zs_in_ring = {graph.atoms[i].z for i in ring}
                # Only C, N, O, S in aromatic rings
                valid_zs = zs_in_ring <= {6, 7, 8, 16}
                if all_in_range and valid_zs and len(ring_bonds) >= sz:
                    return True
    return False

def smiles_from_atoms(atoms):
    from .molecular_graph import from_atoms_list
    return graph_to_smiles(from_atoms_list(atoms))

def smiles_from_name(name):
    from .structure_db import structure_from_name
    s = structure_from_name(name)
    return smiles_from_atoms(s) if s else None
