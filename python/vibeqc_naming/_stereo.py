"""Caller-supplied stereodescriptors on a molecular graph (#537).

The engine does not assign CIP labels. A caller that knows them, such as a
sketcher whose server runs the new CIP algorithm (Hanson et al., J. Chem.
Inf. Model. 2018, 58, 1755), attaches ``R``/``S``/``r``/``s`` to atoms and
``E``/``Z`` to double bonds. This module checks that each label sits on an
atom or bond that can carry one, and writes the descriptors in front of a
name with their locants in ascending order, as in ``(2R)-butan-2-ol`` (IUPAC
2013 recommendations, chapter P-9).
"""

ATOM_LABELS = ("R", "S", "r", "s")
BOND_LABELS = ("E", "Z")


def _atom_index(graph, value, what):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(graph.atoms):
        raise ValueError(f"{what} {value!r} is not an atom index of this graph")
    return value


def has_stereodescriptors(graph):
    return bool(getattr(graph, "atom_stereo", None) or getattr(graph, "bond_stereo", None))


def validate_stereodescriptors(graph):
    """Refuse a label on an atom or bond that cannot carry it.

    Checks structure only: a labelled atom has four neighbours and at most one
    hydrogen; a labelled bond is a double bond whose ends each carry two other
    substituents, not both hydrogen. Whether the label is the right one is the
    caller's responsibility.
    """
    for atom, label in (getattr(graph, "atom_stereo", None) or {}).items():
        _atom_index(graph, atom, "stereocentre")
        if label not in ATOM_LABELS:
            raise ValueError(f"atom {atom}: stereodescriptor {label!r} is not one of {', '.join(ATOM_LABELS)}")
        neighbours = graph.adjacency[atom]
        if graph.atoms[atom].z == 1 or len(neighbours) != 4:
            raise ValueError(
                f"atom {atom} has {len(neighbours)} neighbours; a CIP stereocentre needs four"
            )
        n_hydrogen = sum(graph.atoms[k].z == 1 for k in neighbours)
        if n_hydrogen > 1:
            raise ValueError(f"atom {atom} carries {n_hydrogen} hydrogens and cannot be a stereocentre")
    for key, label in (getattr(graph, "bond_stereo", None) or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"double-bond descriptor key {key!r} must be an (i, j) atom pair")
        i = _atom_index(graph, key[0], "double-bond atom")
        j = _atom_index(graph, key[1], "double-bond atom")
        if label not in BOND_LABELS:
            raise ValueError(f"bond {i}-{j}: stereodescriptor {label!r} is not E or Z")
        bond = graph.bond_between(i, j)
        if bond is None or bond.order != 2:
            raise ValueError(f"atoms {i} and {j} are not joined by a double bond")
        for end, other in ((i, j), (j, i)):
            substituents = [k for k in graph.adjacency[end] if k != other]
            if len(substituents) != 2 or all(graph.atoms[k].z == 1 for k in substituents):
                raise ValueError(
                    f"atom {end} of the {i}={j} double bond needs two substituents, "
                    "not both hydrogen, for an E/Z descriptor"
                )


def stereo_name(graph, name, heavy, mapping, locants):
    """Prefix ``name`` with the graph's descriptors and their locants.

    ``heavy``, ``mapping`` and ``locants`` come from the catalog match of
    ``graph`` to ``name``. Returns ``name`` unchanged when the graph carries no
    descriptors.
    """
    if not has_stereodescriptors(graph):
        return name
    validate_stereodescriptors(graph)
    if locants is None:
        raise ValueError(
            f"stereodescriptors are not supported for {name!r}: no locant numbering is recorded for it"
        )
    position = {atom: k for k, atom in enumerate(heavy)}

    def locant(atom):
        value = locants[mapping[position[atom]]] if atom in position else None
        if value is None:
            raise ValueError(f"atom {atom} is not numbered in {name!r}, so its descriptor has no locant")
        return value

    atom_stereo = getattr(graph, "atom_stereo", None) or {}
    bond_stereo = getattr(graph, "bond_stereo", None) or {}
    entries = [(locant(atom), f"{locant(atom)}{label}") for atom, label in atom_stereo.items()]
    for (i, j), label in bond_stereo.items():
        first = min(locant(i), locant(j))
        entries.append((first, f"{first}{label}"))
    entries.sort(key=lambda entry: entry[0])
    return "(" + ",".join(text for _, text in entries) + ")-" + name
