"""Conservative constitutional-identity checks for supported molecular names.

These explicit neutral connectivity patterns are chemical structure definitions,
not formula aliases or coordinate templates. Hydrogen counts and bond orders
participate in matching. No stereochemical identity or general IUPAC coverage
is claimed, and no optional chemistry backend is required at runtime.
"""
from collections import Counter
from numbers import Integral

# Heavy-atom labels (atomic number, attached H), then (i, j, bond order).
# Aromatic edges use 1.5 to avoid dependence on a particular Kekule form.
_PATTERNS = {
    'methane': (((6, 4),), ()),
    'ethane': (((6, 3), (6, 3)), ((0, 1, 1.0),)),
    'propane': (((6, 3), (6, 2), (6, 3)), ((0, 1, 1.0), (1, 2, 1.0))),
    'butane': (((6, 3), (6, 2), (6, 2), (6, 3)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0))),
    'pentane': (((6, 3), (6, 2), (6, 2), (6, 2), (6, 3)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0))),
    'hexane': (((6, 3), (6, 2), (6, 2), (6, 2), (6, 2), (6, 3)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0))),
    '2-methylpentane': (((6, 3), (6, 1), (6, 3), (6, 2), (6, 2), (6, 3)), ((0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0))),
    'acetylene': (((6, 1), (6, 1)), ((0, 1, 3.0),)),
    'ethylene': (((6, 2), (6, 2)), ((0, 1, 2.0),)),
    'but-2-ene': (((6, 3), (6, 1), (6, 1), (6, 3)), ((0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0))),
    '1,3-butadiene': (((6, 2), (6, 1), (6, 1), (6, 2)), ((0, 1, 2.0), (1, 2, 1.0), (2, 3, 2.0))),
    'cyclopropane': (((6, 2), (6, 2), (6, 2)), ((0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0))),
    'cyclobutane': (((6, 2), (6, 2), (6, 2), (6, 2)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0))),
    'cyclohexane': (((6, 2), (6, 2), (6, 2), (6, 2), (6, 2), (6, 2)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 0, 1.0))),
    'benzene': (((6, 1), (6, 1), (6, 1), (6, 1), (6, 1), (6, 1)), ((0, 1, 1.5), (1, 2, 1.5), (2, 3, 1.5), (3, 4, 1.5), (4, 5, 1.5), (5, 0, 1.5))),
    'toluene': (((6, 3), (6, 0), (6, 1), (6, 1), (6, 1), (6, 1), (6, 1)), ((0, 1, 1.0), (1, 2, 1.5), (2, 3, 1.5), (3, 4, 1.5), (4, 5, 1.5), (5, 6, 1.5), (6, 1, 1.5))),
    'phenol': (((8, 1), (6, 0), (6, 1), (6, 1), (6, 1), (6, 1), (6, 1)), ((0, 1, 1.0), (1, 2, 1.5), (2, 3, 1.5), (3, 4, 1.5), (4, 5, 1.5), (5, 6, 1.5), (6, 1, 1.5))),
    'aniline': (((7, 2), (6, 0), (6, 1), (6, 1), (6, 1), (6, 1), (6, 1)), ((0, 1, 1.0), (1, 2, 1.5), (2, 3, 1.5), (3, 4, 1.5), (4, 5, 1.5), (5, 6, 1.5), (6, 1, 1.5))),
    'pyridine': (((7, 0), (6, 1), (6, 1), (6, 1), (6, 1), (6, 1)), ((0, 1, 1.5), (1, 2, 1.5), (2, 3, 1.5), (3, 4, 1.5), (4, 5, 1.5), (5, 0, 1.5))),
    'naphthalene': (((6, 1), (6, 1), (6, 1), (6, 0), (6, 1), (6, 1), (6, 1), (6, 1), (6, 0), (6, 1)), ((0, 1, 1.5), (1, 2, 1.5), (2, 3, 1.5), (3, 4, 1.5), (4, 5, 1.5), (5, 6, 1.5), (6, 7, 1.5), (7, 8, 1.5), (8, 9, 1.5), (9, 0, 1.5), (8, 3, 1.5))),
    'chlorobenzene': (((17, 0), (6, 0), (6, 1), (6, 1), (6, 1), (6, 1), (6, 1)), ((0, 1, 1.0), (1, 2, 1.5), (2, 3, 1.5), (3, 4, 1.5), (4, 5, 1.5), (5, 6, 1.5), (6, 1, 1.5))),
    'biphenyl': (((6, 1), (6, 1), (6, 1), (6, 0), (6, 0), (6, 1), (6, 1), (6, 1), (6, 1), (6, 1), (6, 1), (6, 1)), ((0, 1, 1.5), (1, 2, 1.5), (2, 3, 1.5), (3, 4, 1.0), (4, 5, 1.5), (5, 6, 1.5), (6, 7, 1.5), (7, 8, 1.5), (8, 9, 1.5), (3, 10, 1.5), (10, 11, 1.5), (11, 0, 1.5), (9, 4, 1.5))),
    'formaldehyde': (((6, 2), (8, 0)), ((0, 1, 2.0),)),
    'methanol': (((6, 3), (8, 1)), ((0, 1, 1.0),)),
    'acetaldehyde': (((6, 3), (6, 1), (8, 0)), ((0, 1, 1.0), (1, 2, 2.0))),
    'ethanol': (((6, 3), (6, 2), (8, 1)), ((0, 1, 1.0), (1, 2, 1.0))),
    'dimethyl ether': (((6, 3), (8, 0), (6, 3)), ((0, 1, 1.0), (1, 2, 1.0))),
    'acetone': (((6, 3), (6, 0), (8, 0), (6, 3)), ((0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0))),
    'propanal': (((6, 3), (6, 2), (6, 1), (8, 0)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 2.0))),
    'formic acid': (((6, 1), (8, 0), (8, 1)), ((0, 1, 2.0), (0, 2, 1.0))),
    'acetic acid': (((6, 3), (6, 0), (8, 0), (8, 1)), ((0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0))),
    'propanoic acid': (((6, 3), (6, 2), (6, 0), (8, 0), (8, 1)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 2.0), (2, 4, 1.0))),
    'ethyl acetate': (((6, 3), (6, 2), (8, 0), (6, 0), (6, 3), (8, 0)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (3, 5, 2.0))),
    'glycine': (((7, 2), (6, 2), (6, 0), (8, 0), (8, 1)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 2.0), (2, 4, 1.0))),
    'acetonitrile': (((6, 3), (6, 0), (7, 0)), ((0, 1, 1.0), (1, 2, 3.0))),
    'hydrogen cyanide': (((6, 1), (7, 0)), ((0, 1, 3.0),)),
    'methylamine': (((6, 3), (7, 2)), ((0, 1, 1.0),)),
    'trimethylamine': (((6, 3), (7, 0), (6, 3), (6, 3)), ((0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0))),
    'cyclohexanol': (((8, 1), (6, 1), (6, 2), (6, 2), (6, 2), (6, 2), (6, 2)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 1, 1.0))),
    'butan-2-ol': (((6, 3), (6, 1), (8, 1), (6, 2), (6, 3)), ((0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0), (3, 4, 1.0))),
    'ethane-1,2-diol': (((8, 1), (6, 2), (6, 2), (8, 1)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0))),
    'benzaldehyde': (((8, 0), (6, 1), (6, 0), (6, 1), (6, 1), (6, 1), (6, 1), (6, 1)), ((0, 1, 2.0), (1, 2, 1.0), (2, 3, 1.5), (3, 4, 1.5), (4, 5, 1.5), (5, 6, 1.5), (6, 7, 1.5), (7, 2, 1.5))),
    'benzoic acid': (((8, 0), (6, 0), (8, 1), (6, 0), (6, 1), (6, 1), (6, 1), (6, 1), (6, 1)), ((0, 1, 2.0), (1, 2, 1.0), (1, 3, 1.0), (3, 4, 1.5), (4, 5, 1.5), (5, 6, 1.5), (6, 7, 1.5), (7, 8, 1.5), (8, 3, 1.5))),
    'tetrahydrofuran': (((6, 2), (6, 2), (6, 2), (8, 0), (6, 2)), ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 0, 1.0))),
    'dichloromethane': (((17, 0), (6, 2), (17, 0)), ((0, 1, 1.0), (1, 2, 1.0))),
}

# Locant of each heavy pattern atom in the name, None where the name does not
# number it (the hydroxy oxygen of butan-2-ol). Stereodescriptors (#537) are
# placed only on entries listed here; both numberings of but-2-ene put the
# double bond at 2.
_LOCANTS = {
    'butan-2-ol': (1, 2, None, 3, 4),
    'but-2-ene': (1, 2, 3, 4),
}


def validate_state(charge, multiplicity):
    """Refuse electronic states the neutral molecular-name model cannot encode."""
    if (isinstance(charge, bool) or not isinstance(charge, Integral)
            or isinstance(multiplicity, bool) or not isinstance(multiplicity, Integral)
            or multiplicity < 1):
        raise ValueError("charge must be an integer and multiplicity a positive integer")
    if charge != 0 or multiplicity != 1:
        raise ValueError("molecular naming does not represent charged or open-shell species")


def matched_name(graph):
    """Return a supported constitutional name only after labeled graph matching."""
    match = match_catalog(graph)
    return None if match is None else match[0]


def catalog_locants(name):
    """Heavy-pattern locants of a catalog name, or None if it records none."""
    return _LOCANTS.get(name)


def match_catalog(graph):
    """Match ``graph`` against the catalog.

    Returns ``(name, heavy, mapping)`` or None: ``heavy`` lists the graph's
    non-hydrogen atom indices and ``mapping[k]`` is the pattern atom that
    ``heavy[k]`` matched.
    """
    from .smiles import _aromatic

    if not graph.atoms:
        return None
    seen, pending = set(), [0]
    while pending:
        i = pending.pop()
        if i not in seen:
            seen.add(i)
            pending.extend(graph.adjacency[i])
    if len(seen) != len(graph.atoms):
        return None
    if any(a.z == 1 and len(graph.adjacency[i]) != 1 for i, a in enumerate(graph.atoms)):
        return None
    heavy = [i for i, a in enumerate(graph.atoms) if a.z != 1]
    remap = {old: new for new, old in enumerate(heavy)}
    labels = tuple((graph.atoms[i].z, sum(graph.atoms[j].z == 1 for j in graph.adjacency[i])) for i in heavy)
    observed = [[0.0] * len(heavy) for _ in heavy]
    for bond in graph.bonds:
        if bond.i not in remap or bond.j not in remap:
            continue
        i, j = remap[bond.i], remap[bond.j]
        # A single bond between aromatic rings is not itself aromatic.
        in_ring = any(bond.i in ring and bond.j in ring for ring in graph.rings)
        order = 1.5 if in_ring and _aromatic(graph, bond.i) and _aromatic(graph, bond.j) else bond.order
        observed[i][j] = observed[j][i] = order
    counts = Counter(labels)
    for name, (expected_labels, edges) in _PATTERNS.items():
        if counts != Counter(expected_labels):
            continue
        expected = [[0.0] * len(heavy) for _ in heavy]
        for i, j, order in edges:
            expected[i][j] = expected[j][i] = order
        candidates = {i: [j for j in range(len(heavy)) if labels[i] == expected_labels[j]
                         and sorted(observed[i]) == sorted(expected[j])]
                      for i in range(len(heavy))}
        order = sorted(candidates, key=lambda i: len(candidates[i]))
        mapping = {}
        used = set()

        def match(depth):
            if depth == len(order):
                return True
            i = order[depth]
            for j in candidates[i]:
                if j in used or any(observed[i][k] != expected[j][v] for k, v in mapping.items()):
                    continue
                mapping[i] = j
                used.add(j)
                if match(depth + 1):
                    return True
                used.remove(j)
                del mapping[i]
            return False

        if match(0):
            return name, heavy, dict(mapping)
    return None
