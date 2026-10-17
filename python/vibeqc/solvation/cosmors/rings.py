"""Ring-atom perception for the COSMO-RS gas-phase reference.

Klamt 1998 eq. 21 lowers a molecule's gas-phase chemical potential by
``omega * n_ring`` with ``n_ring`` "the number of ring atoms in molecule X"
(section 5.1: ``omega = -0.21`` kcal/mol, "for a six-membered ring this
corresponds to -1.26 kcal/mol and it reflects the difference in dG_hydr
between hexane and cyclohexane"). The paper counts atoms that belong to a
ring; nothing about the ring's size, aromaticity or heteroatoms enters, and
Klamt reports the term working "even for rings of such size" as 16 atoms.

That count is a property of the molecular graph: an atom is a ring atom when
it lies on a cycle, i.e. when at least one of its bonds is not a bridge. The
bonds come from the covalent-radius perception in
:mod:`vibeqc.naming.molecular_graph` (Cordero et al. 2008 radii, tolerance
1.25), and the cycles from the bridge classification of Tarjan 1974, which
needs no enumeration of rings and so has no size cap. A fused system counts
every atom once; an atom on a chain *between* two rings is not on any cycle
and is not counted, which is what "ring atom" means.

Perceived bonds are a heuristic, so the perception is exposed, not silently
applied: ``CosmoRSOptions(n_ring_atoms="auto")`` asks for it and the output
states the count and that it was perceived.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = ["ring_atoms", "ring_atom_count", "perceive_bonds"]

BOHR_TO_ANG = 0.529177210903


def perceive_bonds(
    atomic_numbers: Sequence[int], positions_bohr: np.ndarray, *, tolerance: float = 1.25
) -> list[tuple[int, int]]:
    """Bonds as ``(i, j)`` pairs from covalent radii, positions in bohr."""
    from vibeqc.naming.molecular_graph import (
        ATOMIC_NUMBER_TO_SYMBOL,
        AtomInfo,
        detect_bonds,
    )

    pos = np.asarray(positions_bohr, dtype=np.float64).reshape(-1, 3) * BOHR_TO_ANG
    atoms = [
        AtomInfo(
            index=i, z=int(z), symbol=ATOMIC_NUMBER_TO_SYMBOL.get(int(z), "X"),
            x=float(p[0]), y=float(p[1]), z_coord=float(p[2]),
        )
        for i, (z, p) in enumerate(zip(atomic_numbers, pos))
    ]
    return [(int(b.i), int(b.j)) for b in detect_bonds(atoms, tolerance=tolerance)]


def _bridges(n_atoms: int, bonds: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    """Bridge edges of the graph (Tarjan 1974), as ``(min, max)`` pairs.

    Iterative depth-first search with discovery times and low-links; an edge
    ``(u, v)`` to a child ``v`` is a bridge when no back edge from ``v``'s
    subtree reaches ``u`` or above, ``low[v] > disc[u]``.
    """
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for k, (i, j) in enumerate(bonds):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    disc = [-1] * n_atoms
    low = [0] * n_atoms
    bridges: set[tuple[int, int]] = set()
    time = 0
    for root in range(n_atoms):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = time
        time += 1
        # (node, parent edge id, iterator over its adjacency)
        stack = [(root, -1, iter(adjacency[root]))]
        while stack:
            u, parent_edge, it = stack[-1]
            advanced = False
            for v, edge in it:
                if edge == parent_edge:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = time
                    time += 1
                    stack.append((v, edge, iter(adjacency[v])))
                    advanced = True
                    break
                low[u] = min(low[u], disc[v])
            if advanced:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                low[p] = min(low[p], low[u])
                if low[u] > disc[p]:
                    bridges.add((min(p, u), max(p, u)))
    return bridges


def ring_atoms(
    atomic_numbers: Sequence[int], positions_bohr: np.ndarray, *, tolerance: float = 1.25
) -> list[int]:
    """Indices of the atoms that lie on a cycle of the perceived bond graph."""
    atomic_numbers = [int(z) for z in atomic_numbers]
    bonds = perceive_bonds(atomic_numbers, positions_bohr, tolerance=tolerance)
    bridges = _bridges(len(atomic_numbers), bonds)
    on_cycle: set[int] = set()
    for i, j in bonds:
        if (min(i, j), max(i, j)) not in bridges:
            on_cycle.update((i, j))
    return sorted(on_cycle)


def ring_atom_count(molecule, *, tolerance: float = 1.25) -> int:
    """``n_ring`` of Klamt 1998 eq. 21 for a :class:`vibeqc.Molecule`."""
    atoms = list(molecule.atoms)
    return len(
        ring_atoms(
            [int(a.Z) for a in atoms],
            np.array([list(a.xyz) for a in atoms], dtype=np.float64),
            tolerance=tolerance,
        )
    )
