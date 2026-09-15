"""Shared validation and record flattening for SECCM method adapters.

Private to the ``seccm`` package; not part of the reviewed public surface.
"""

from __future__ import annotations

import numpy as np

from vibeqc._vibeqc_core import Molecule
from vibeqc.molecule import ANGSTROM_TO_BOHR

from .topology import SECCMTopology, SECCMTopologyError


def flatten_topology_records(
    molecule: Molecule,
    topology: SECCMTopology,
    *,
    route_name: str,
) -> tuple[
    np.ndarray,
    list[int],
    list[int],
    np.ndarray,
    list[float],
    list[int],
    np.ndarray,
    np.ndarray,
    tuple[int, int, int],
]:
    """Validate the shared SECCM contract and flatten the WS records.

    Returns ``(translations, central, origin, shell_labels, weights,
    multiplicities, displacements, primitive_vectors, replicas)`` for the
    C++ record adapters.  All Cartesian arrays are copied in bohr so the
    private native seam can re-attest the finite-group normalization.
    """
    if not isinstance(molecule, Molecule):
        raise TypeError(f"{route_name} requires a Molecule")
    if not isinstance(topology, SECCMTopology):
        raise TypeError(f"{route_name} requires an SECCMTopology")
    if not topology.finite_group_bound:
        raise SECCMTopologyError(
            f"{route_name} requires an explicitly bound finite translation "
            "group"
        )
    if not topology.current_lattice_group_compatible:
        raise SECCMTopologyError(
            f"{route_name} cyclic translations are incompatible with the "
            "bound finite group"
        )
    if topology.dimensionality not in (1, 2, 3):
        raise SECCMTopologyError(
            f"{route_name} requires one to three cyclic dimensions"
        )
    if topology.length_unit is None:
        raise SECCMTopologyError(
            f"{route_name} requires explicit topology length-unit metadata"
        )
    if len(topology.cells) != len(molecule.atoms):
        raise SECCMTopologyError(
            f"{route_name} topology atom count does not match the molecule"
        )
    if not topology.is_valid(len(molecule.atoms)):
        raise SECCMTopologyError(
            f"{route_name} topology fails the directed ownership validity "
            "rule"
        )
    topology.reversal_map()
    if topology.record_orbits_currently_usable:
        # A bound action is a stronger claim than pairwise reversibility: each
        # record must also possess every simultaneous finite-group translate.
        # Only validate that optional reduction while it is usable.  The
        # native SECCM adapters consume the complete directed record list, so
        # a symmetry-breaking geometry must not be rejected merely because a
        # previously bound orbit reduction is no longer available.
        topology.record_translation_orbits()

    unit_scale = 1.0 if topology.length_unit == "bohr" else ANGSTROM_TO_BOHR
    translations = np.asarray(topology.translations, dtype=float) * unit_scale
    group = topology.finite_group
    assert group is not None
    primitive_vectors = (
        np.asarray(group.primitive_vectors, dtype=float) * unit_scale
    )
    replicas = tuple(int(value) for value in group.replicas)
    central: list[int] = []
    origin: list[int] = []
    shell_labels: list[tuple[int, int, int]] = []
    weights: list[float] = []
    multiplicities: list[int] = []
    displacements: list[np.ndarray] = []
    for central_index, cell in enumerate(topology.cells):
        for image in cell:
            central.append(central_index)
            origin.append(int(image.origin))
            shell_labels.append(
                tuple(int(value) for value in image.image_shell_label)
            )
            weights.append(float(image.weight))
            multiplicities.append(int(image.ownership_multiplicity))
            displacements.append(
                np.asarray(image.disp, dtype=float) * unit_scale
            )
    shell_label_matrix = np.asarray(
        shell_labels,
        dtype=np.int32,
    ).reshape((-1, 3))
    displacement_matrix = np.asarray(
        displacements,
        dtype=float,
    ).reshape((-1, 3))
    return (
        translations,
        central,
        origin,
        shell_label_matrix,
        weights,
        multiplicities,
        displacement_matrix,
        primitive_vectors,
        replicas,
    )


def topology_length_unit_scale(topology: SECCMTopology) -> float:
    """Return the bohr-per-unit scale for a topology's length unit."""
    return 1.0 if topology.length_unit == "bohr" else ANGSTROM_TO_BOHR


def translation_orbits(
    molecule: Molecule,
    topology: SECCMTopology,
) -> tuple[tuple[int, ...], ...]:
    """Group atom indices into orbits of the bound finite translation group.

    The cyclic cluster carries a finite translation group by construction,
    so two atoms whose separation is a primitive lattice vector of that
    group are the *same* site of the underlying crystal, differing only by
    which replica cell they sit in. Every physical property that is a
    function of the site alone - the Mulliken charge among them - is
    therefore required to be equal across an orbit. This is a property of
    the model, not a convergence heuristic.

    Two atoms must have the same species and their full Cartesian
    displacement must be an integer combination of the primitive vectors,
    within the bound group's Cartesian geometry tolerance. In 1-D/2-D the
    nonperiodic displacement is retained: different chains or slab layers
    are not translation-equivalent merely because their projections match.

    Returns a tuple of index tuples, one per orbit, in first-appearance
    order. Orbits of size one are included: a defect or surface site is
    its own orbit and simply carries no equality constraint.
    """
    group = topology.finite_group
    if group is None:
        raise SECCMTopologyError(
            "translation orbits require an explicitly bound finite "
            "translation group"
        )
    scale = topology_length_unit_scale(topology)
    primitive = np.asarray(group.primitive_vectors, dtype=float) * scale
    if primitive.ndim != 2 or primitive.shape[0] < 1:
        raise SECCMTopologyError(
            "translation orbits require at least one primitive vector"
        )
    coords = np.array([atom.xyz for atom in molecule.atoms], dtype=float)

    inverse = np.linalg.pinv(primitive)
    tolerance = float(group.geometry_tolerance) * scale
    species = [int(atom.Z) for atom in molecule.atoms]
    orbits: list[list[int]] = []
    for index, coordinate in enumerate(coords):
        for members in orbits:
            representative = members[0]
            if species[index] != species[representative]:
                continue
            displacement = coordinate - coords[representative]
            lattice_shift = np.rint(displacement @ inverse) @ primitive
            if np.allclose(
                displacement, lattice_shift, rtol=0.0, atol=tolerance
            ):
                members.append(index)
                break
        else:
            orbits.append([index])
    return tuple(tuple(members) for members in orbits)


def translation_symmetry_charge_spread(
    molecule: Molecule,
    topology: SECCMTopology,
    charges,
) -> float:
    """Largest charge spread within any translation orbit (electrons).

    Zero for a state that respects the cyclic translation symmetry the
    model is defined to have. A nonzero value is a symmetry violation: the
    converged density distinguishes atoms that the finite group makes
    equivalent. This is reported, never gated - see the note at the call
    site and issue #421.

    Returns NaN for absent or nonfinite charges. Failed runs that expose
    finite partial charges can still be diagnosed; the spread alone does
    not imply convergence. A present vector of the wrong length raises.
    """
    values = np.asarray(charges, dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan")
    if values.size != len(molecule.atoms):
        raise ValueError(
            "charge vector length does not match the molecule atom count"
        )
    if not np.isfinite(values).all():
        return float("nan")
    worst = 0.0
    for members in translation_orbits(molecule, topology):
        if len(members) < 2:
            continue
        block = values[list(members)]
        worst = max(worst, float(block.max() - block.min()))
    return worst
