"""PM6 NDDO energy over a frozen SECCM topology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibeqc._vibeqc_core import Molecule
from vibeqc._vibeqc_core import semiempirical as _se_cxx
from vibeqc.molecule import Atom
from vibeqc.semiempirical.routes import SemiempiricalRoutePlan

from ._adapter_common import flatten_topology_records, topology_length_unit_scale
from .topology import SECCMTopology


PM6_SECCM_PARAMETER_SET = "pm6-auto-parameter-loader"


@dataclass(frozen=True)
class PM6SECCMResult:
    """One finite-cluster PM6-SECCM energy, per primitive cell."""

    energy: float
    e_electronic: float
    e_core: float
    #: CCM Madelung self-energy per primitive cell, 0.0 unless the opt-in
    #: embedding is enabled. MSINDO convention (madelsum.f + ccmfockcl.f):
    #: the diagonal Fock deposit carries the electronic half, this field
    #: the core-charge half, together 1/2 sum_I q_I V_mad(I).
    e_madelung: float
    total_cyclic_energy: float
    cyclic_core_energy: float
    cyclic_madelung_energy: float
    homo_lumo_gap: float
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    gradient: np.ndarray | None
    gradient_method: str | None
    n_basis: int
    n_occ: int
    n_iter: int
    group_order: int
    n_records: int
    converged: bool
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


def _gradient_fd(
    molecule: Molecule,
    topology: SECCMTopology,
    options: dict,
    step: float,
    parameter_identity: str,
    parameter_sha256: str,
) -> np.ndarray:
    """Central-difference gradient of the converged per-cell energy.

    No analytic derivative of the current PM6 Hamiltonian is exposed. The
    SECCM boundary therefore uses the exact-by-construction option: central
    differences over frozen topology displacements
    (topology.rebuild_displacements keeps the WS record set fixed while the
    displacements track the displaced geometry).
    """
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("gradient_fd_step must be positive and finite")
    coords = np.array([atom.xyz for atom in molecule.atoms])
    rebuild_options = (
        {"max_tie_score_excursion": 1.0e-3}
        if topology.has_reference_ties
        else {}
    )
    energy_options = {**options, "compute_gradient": False}
    topology_unit_scale = topology_length_unit_scale(topology)

    def energy_at(geometry: np.ndarray) -> float:
        # Molecule coordinates and ``gradient_fd_step`` are always expressed in
        # bohr. ``rebuild_displacements`` instead consumes coordinates in the
        # topology's declared unit, so convert only at that boundary and keep
        # the displaced Molecule in bohr.
        displaced = topology.rebuild_displacements(
            geometry / topology_unit_scale, **rebuild_options
        )
        displaced_molecule = Molecule(
            [
                Atom(atom.Z, coordinate.tolist())
                for atom, coordinate in zip(molecule.atoms, geometry)
            ],
            molecule.charge,
            molecule.multiplicity,
        )
        result = run_pm6_seccm(
            displaced_molecule, displaced, **energy_options
        )
        if (
            result.parameter_identity != parameter_identity
            or result.parameter_sha256 != parameter_sha256
        ):
            raise RuntimeError(
                "PM6-SECCM finite differences used different immutable "
                "parameter snapshots"
            )
        return result.energy

    gradient = np.zeros((len(coords), 3))
    for atom in range(len(coords)):
        for axis in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            gradient[atom, axis] = (
                energy_at(plus) - energy_at(minus)
            ) / (2.0 * step)
    return gradient


def run_pm6_seccm(
    molecule: Molecule,
    topology: SECCMTopology,
    *,
    parameter_set: str = PM6_SECCM_PARAMETER_SET,
    max_iter: int = 100,
    conv_tol: float = 1.0e-7,
    madelung: bool = False,
    allow_truncated_electrostatics: bool = False,
    compute_gradient: bool = False,
    gradient_fd_step: float = 1.0e-4,
) -> PM6SECCMResult:
    """Evaluate the PM6-SECCM finite-cluster route.

    The supercell Fock is the molecular PM6 NDDO Fock with every directed
    two-center term accumulated through the WS record set with its fractional
    weight. Neutral closed-shell clusters with one to three cyclic dimensions.
    ``madelung=True`` enables the opt-in CCM Madelung/Ewald embedding for
    one and two cyclic dimensions: the classical point-charge field of the
    lattice beyond the Wigner-Seitz cell, added to the diagonal of the Fock
    matrix in the MSINDO CCM convention (``ccmfockcl.f``:
    ``FA(K,K) = H(K,K) - MADELATOM(I)``) with the potential taken as the
    full Ewald lattice sum minus the WS-internal bare ``1/r``, WS-weighted
    with the self image excluded (``madelsum.f``, SMADEL branch), because
    the NDDO two-centre integrals already carry everything inside the WS
    cell. Three dimensions fail closed: the WS-folded remainder there has
    no thermodynamic limit (issues #211, #425, #444).

    Without the embedding a nontrivial cyclic topology is **not
    quantitative** - the WS-truncated monopole sum it leaves reproduces the
    exact rocksalt Madelung constant only to -43%..+38%, with the sign
    flipping on replica parity - so such a run requires an explicit
    ``allow_truncated_electrostatics=True`` acknowledgement and is an
    algorithm-mechanics probe rather than a solid-state prediction.

    Note the embedded route legitimately differs from the isolated molecule
    at a trivial one-cell topology whenever the cell is polar, because a
    polar cell really does interact with its periodic images; the
    bit-for-bit molecular-limit contract is a property of the unembedded
    route. The shipped SCC-DFTB-SECCM embedding behaves identically.

    The current SECCM Hamiltonian implements only s/p AO channels. Elements
    whose bundled PM6 parameter record requires d orbitals fail closed before
    SCF; accepting their nine-orbital records with p-channel stand-ins would
    not be PM6.

    ``compute_gradient=True`` returns central differences of the converged
    per-cell energy over the frozen topology displacements (six converged
    SCFs per atom, ``gradient_fd_step`` controls the displacement); the
    result reports ``gradient_method="finite_difference"``.
    """
    if parameter_set != PM6_SECCM_PARAMETER_SET:
        raise ValueError(
            "PM6-SECCM supports only parameter_set="
            f"{PM6_SECCM_PARAMETER_SET!r}"
        )
    route_plan = SemiempiricalRoutePlan.from_request(
        "pm6",
        boundary="seccm",
        properties=("energy", "gradient")
        if compute_gradient
        else ("energy",),
        charge=int(molecule.charge),
        multiplicity=int(molecule.multiplicity),
    ).with_seccm_runtime(periodic_dimension=topology.dimensionality)
    (
        translations,
        central,
        origin,
        shell_labels,
        weights,
        multiplicities,
        displacements,
        primitive_vectors,
        replicas,
    ) = flatten_topology_records(molecule, topology, route_name="PM6-SECCM")

    group = topology.finite_group
    assert group is not None
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

    params = load_pm6_params_auto([int(atom.Z) for atom in molecule.atoms])
    native = _se_cxx._run_pm6_seccm_from_records(
        molecule,
        params,
        translations,
        central,
        origin,
        np.asarray(shell_labels, dtype=np.int32),
        weights,
        multiplicities,
        np.asarray(displacements, dtype=float),
        primitive_vectors,
        replicas,
        float(group.geometry_tolerance) * topology_length_unit_scale(topology),
        int(max_iter),
        float(conv_tol),
        bool(madelung),
        bool(allow_truncated_electrostatics),
    )
    if not bool(native.converged):
        raise RuntimeError(
            "C++ PM6-SECCM did not converge within the iteration budget"
        )
    if compute_gradient:
        gradient = _gradient_fd(
            molecule,
            topology,
            {
                "parameter_set": parameter_set,
                "max_iter": max_iter,
                "conv_tol": conv_tol,
                "madelung": madelung,
                "allow_truncated_electrostatics": (
                    allow_truncated_electrostatics
                ),
            },
            gradient_fd_step,
            str(native.parameter_identity),
            str(native.parameter_sha256),
        )
    else:
        gradient = None
    return PM6SECCMResult(
        energy=float(native.energy),
        e_electronic=float(native.e_electronic),
        e_core=float(native.e_core),
        e_madelung=float(native.e_madelung),
        total_cyclic_energy=float(native.total_cyclic_energy),
        cyclic_core_energy=float(native.cyclic_core_energy),
        cyclic_madelung_energy=float(native.cyclic_madelung_energy),
        homo_lumo_gap=float(native.homo_lumo_gap),
        mo_energies=_frozen_array(native.mo_energies),
        mo_coeffs=_frozen_array(native.mo_coeffs),
        density=_frozen_array(native.density),
        gradient=(
            _frozen_array(gradient)
            if gradient is not None
            else None
        ),
        gradient_method="finite_difference" if compute_gradient else None,
        n_basis=int(native.n_basis),
        n_occ=int(native.n_occ),
        n_iter=int(native.n_iter),
        group_order=int(native.group_order),
        n_records=int(native.n_records),
        converged=bool(native.converged),
        normalization="per_primitive_cell",
        parameter_set=parameter_set,
        parameter_identity=str(native.parameter_identity),
        parameter_sha256=str(native.parameter_sha256),
        topology_fingerprint=topology.topology_fingerprint,
        reference_geometry_fingerprint=topology.reference_geometry_fingerprint,
        route_plan=route_plan,
    )


__all__ = [
    "PM6_SECCM_PARAMETER_SET",
    "PM6SECCMResult",
    "run_pm6_seccm",
]
