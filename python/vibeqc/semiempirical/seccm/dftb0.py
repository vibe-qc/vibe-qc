"""DFTB0 energy and analytic gradient over a frozen SECCM topology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibeqc._vibeqc_core import Molecule
from vibeqc._vibeqc_core import semiempirical as _se_cxx
from vibeqc.molecule import ANGSTROM_TO_BOHR
from vibeqc.semiempirical.parameters import default_parameters
from vibeqc.semiempirical.routes import SemiempiricalRoutePlan

from .topology import SECCMTopology, SECCMTopologyError


DFTB0_SECCM_PARAMETER_SET = "vibeqc-inhouse-dftb-screening-v1"


@dataclass(frozen=True)
class DFTB0SECCMResult:
    """One finite-cluster DFTB0-SECCM energy and optional gradient."""

    energy: float
    electronic_energy: float
    repulsive_energy: float
    long_range_energy: float
    dispersion_energy: float
    specific_energy: float
    total_cyclic_energy: float
    cyclic_electronic_energy: float
    cyclic_repulsive_energy: float
    homo_lumo_gap: float
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    overlap: np.ndarray
    hamiltonian: np.ndarray
    gradient: np.ndarray | None
    n_basis: int
    n_occ: int
    group_order: int
    n_records: int
    normalization: str
    parameter_set: str
    parameter_identity: str
    parameter_sha256: str
    topology_fingerprint: str
    reference_geometry_fingerprint: str
    route_plan: SemiempiricalRoutePlan


def _frozen_array(value) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def run_dftb0_seccm(
    molecule: Molecule,
    topology: SECCMTopology,
    *,
    parameter_set: str = DFTB0_SECCM_PARAMETER_SET,
    compute_gradient: bool = False,
) -> DFTB0SECCMResult:
    """Evaluate the gated DFTB0-SECCM finite-cluster route.

    The route accepts neutral, closed-shell, insulating systems with one,
    two, or three cyclic dimensions and explicit repulsive-pair coverage
    (currently H, C, N, O, F, P, S, and Cl). It evaluates the complete frozen
    WS record set and normalizes energy and, when requested, its fixed-topology analytic
    nuclear gradient by the finite-group order. No SCC, Madelung, dispersion,
    stress, topology derivative, or orbit reduction is implied.
    """
    if not isinstance(molecule, Molecule):
        raise TypeError("DFTB0-SECCM requires a Molecule")
    if not isinstance(topology, SECCMTopology):
        raise TypeError("DFTB0-SECCM requires an SECCMTopology")

    route_plan = SemiempiricalRoutePlan.from_request(
        "dftb0",
        boundary="seccm",
        properties=("energy", "gradient") if compute_gradient else ("energy",),
        charge=int(molecule.charge),
        multiplicity=int(molecule.multiplicity),
    ).with_seccm_runtime(periodic_dimension=topology.dimensionality)
    if parameter_set != DFTB0_SECCM_PARAMETER_SET:
        raise ValueError(
            "DFTB0-SECCM T3a supports only parameter_set="
            f"{DFTB0_SECCM_PARAMETER_SET!r}"
        )
    if not topology.finite_group_bound:
        raise SECCMTopologyError(
            "DFTB0-SECCM requires an explicitly bound finite translation group"
        )
    if not topology.current_lattice_group_compatible:
        raise SECCMTopologyError(
            "DFTB0-SECCM cyclic translations are incompatible with the bound "
            "finite group"
        )
    # One, two, and three cyclic dimensions are supported; the native
    # seam and every stage below it are dimension-generic (see the note
    # in cpp/src/semiempirical/seccm/dftb0.cpp).
    if topology.length_unit is None:
        raise SECCMTopologyError(
            "DFTB0-SECCM requires explicit topology length-unit metadata"
        )
    if len(topology.cells) != len(molecule.atoms):
        raise SECCMTopologyError(
            "DFTB0-SECCM topology atom count does not match the molecule"
        )
    if not topology.is_valid(len(molecule.atoms)):
        raise SECCMTopologyError(
            "DFTB0-SECCM topology fails the directed ownership validity rule"
        )
    topology.reversal_map()

    # Scope is the explicit repulsive-pair table, not a hardcoded
    # element list: the shipped set covers H, C, N, O, F, P, S, Cl and
    # every one of their 36 unordered pairs. The native seam applies the
    # same predicate and raises with the covered set named.
    parameters = default_parameters()
    atomic_numbers = sorted({int(atom.Z) for atom in molecule.atoms})
    for index, first in enumerate(atomic_numbers):
        for second in atomic_numbers[index:]:
            if not parameters.has_repulsive_pair(first, second):
                raise NotImplementedError(
                    "DFTB0-SECCM requires an explicit repulsive pair for "
                    "every element pair in the cell; the shipped "
                    "vibeqc-inhouse-dftb-screening-v1 set covers "
                    "H, C, N, O, F, P, S, Cl"
                )

    unit_scale = 1.0 if topology.length_unit == "bohr" else ANGSTROM_TO_BOHR
    translations = np.asarray(topology.translations, dtype=float) * unit_scale
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
            shell_labels.append(tuple(int(value) for value in image.image_shell_label))
            weights.append(float(image.weight))
            multiplicities.append(int(image.ownership_multiplicity))
            displacements.append(np.asarray(image.disp, dtype=float) * unit_scale)

    group = topology.finite_group
    assert group is not None
    primitive_vectors = (
        np.asarray(group.primitive_vectors, dtype=float) * unit_scale
    )
    replicas = tuple(int(value) for value in group.replicas)
    shell_label_matrix = np.asarray(
        shell_labels,
        dtype=np.int32,
    ).reshape((-1, 3))
    displacement_matrix = np.asarray(
        displacements,
        dtype=float,
    ).reshape((-1, 3))
    params = default_parameters()
    native = _se_cxx._run_dftb0_seccm_from_records(
        molecule,
        params,
        np.asarray(translations, dtype=float),
        central,
        origin,
        shell_label_matrix,
        weights,
        multiplicities,
        displacement_matrix,
        primitive_vectors,
        replicas,
        float(group.geometry_tolerance) * unit_scale,
        bool(compute_gradient),
    )
    return DFTB0SECCMResult(
        energy=float(native.energy),
        electronic_energy=float(native.electronic_energy),
        repulsive_energy=float(native.repulsive_energy),
        long_range_energy=float(native.long_range_energy),
        dispersion_energy=float(native.dispersion_energy),
        specific_energy=float(native.specific_energy),
        total_cyclic_energy=float(native.total_cyclic_energy),
        cyclic_electronic_energy=float(native.cyclic_electronic_energy),
        cyclic_repulsive_energy=float(native.cyclic_repulsive_energy),
        homo_lumo_gap=float(native.homo_lumo_gap),
        mo_energies=_frozen_array(native.mo_energies),
        mo_coeffs=_frozen_array(native.mo_coeffs),
        density=_frozen_array(native.density),
        overlap=_frozen_array(native.overlap),
        hamiltonian=_frozen_array(native.hamiltonian),
        gradient=(
            _frozen_array(native.gradient)
            if bool(native.has_gradient)
            else None
        ),
        n_basis=int(native.n_basis),
        n_occ=int(native.n_occ),
        group_order=int(native.group_order),
        n_records=int(native.n_records),
        normalization="per_primitive_cell",
        parameter_set=parameter_set,
        parameter_identity=str(native.parameter_identity),
        parameter_sha256=str(native.parameter_sha256),
        topology_fingerprint=topology.topology_fingerprint,
        reference_geometry_fingerprint=topology.reference_geometry_fingerprint,
        route_plan=route_plan,
    )


__all__ = [
    "DFTB0_SECCM_PARAMETER_SET",
    "DFTB0SECCMResult",
    "run_dftb0_seccm",
]
