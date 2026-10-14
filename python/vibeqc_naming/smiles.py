"""SMILES generation — pre-scanned ring edges, aromatic perception."""

from __future__ import annotations
from typing import Optional

def graph_to_smiles(graph) -> Optional[str]:
    """Serialize a spanning forest, attaching every branch at its parent atom."""
    if graph.n_atoms == 0 or sum(a.z for a in graph.atoms) % 2:
        return None
    heavy = [i for i, atom in enumerate(graph.atoms) if atom.z != 1]
    if not heavy:
        return None
    if any(a.z == 1 and len(graph.adjacency[i]) != 1 for i, a in enumerate(graph.atoms)):
        return None
    adj = {i: [j for j in graph.adjacency[i] if graph.atoms[j].z != 1] for i in heavy}
    orders = {tuple(sorted((b.i, b.j))): b.order for b in graph.bonds}
    aromatic = {i: _aromatic(graph, i) for i in heavy}
    children = {i: [] for i in heavy}
    seen = set()
    tree_edges = set()
    roots = []
    traversal = []

    def visit(i):
        seen.add(i)
        traversal.append(i)
        for j in sorted(adj[i], key=lambda j: (len(adj[j]), graph.atoms[j].z, j)):
            if j not in seen:
                children[i].append(j)
                tree_edges.add(tuple(sorted((i, j))))
                visit(j)

    terminals = [i for i in heavy if len(adj[i]) == 1]
    start = min(terminals, key=lambda i: (graph.atoms[i].z, i)) if terminals else heavy[0]
    for i in [start] + heavy:
        if i not in seen:
            roots.append(i)
            visit(i)
    closures = sorted({tuple(sorted((i, j))) for i in heavy for j in adj[i]} - tree_edges)
    rings = {i: [] for i in heavy}

    def bond_symbol(i, j):
        order = orders[tuple(sorted((i, j)))]
        if aromatic[i] and aromatic[j]:
            if any(i in ring and j in ring for ring in graph.rings):
                return ''
            return '-' if order == 1 else {2: '=', 3: '#'}[order]
        return {1: '', 2: '=', 3: '#'}.get(order, '')

    positions = {i: k for k, i in enumerate(traversal)}
    for digit, (i, j) in enumerate(closures, 1):
        if positions[i] > positions[j]:
            i, j = j, i
        label = str(digit) if digit < 10 else f'%{digit:02d}'
        rings[i].append(bond_symbol(i, j) + label)
        rings[j].append(label)

    def emit(i):
        atom = graph.atoms[i]
        symbol = atom.symbol
        if aromatic[i]:
            symbol = symbol.lower()
            if atom.z == 7 and any(graph.atoms[j].z == 1 for j in graph.adjacency[i]):
                symbol = '[nH]'
        elif atom.z not in {6, 7, 8, 9, 15, 16, 17, 35, 53}:
            symbol = f'[{symbol}]'
        result = symbol + ''.join(rings[i])
        # Branches precede the continuation: after emitting the continuation,
        # SMILES parentheses would attach to its last atom, not to this atom.
        for child in children[i][:-1]:
            result += '(' + bond_symbol(i, child) + emit(child) + ')'
        if children[i]:
            child = children[i][-1]
            result += bond_symbol(i, child) + emit(child)
        return result

    return '.'.join(emit(i) for i in roots)

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
            if any(graph.atoms[i].z == 6 and len(graph.adjacency[i]) > 3 for i in ring):
                continue  # Saturated ring carbons cannot be aromatic.
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
