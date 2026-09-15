"""OMx (OM2/OM3) energy over a frozen SECCM topology.

The supercell Fock is the molecular OMx Hamiltonian evaluated over the
Wigner-Seitz record set with fractional image weights, the same
Bredow-Geudtner-Jug construction as the PM6-SECCM adapter. Three-center
terms (the eq-9 G1/G2 VORT corrections and the OM2/OM3 ECP) carry the
Peintinger-Bredow 2014 image weighting: the production eq-13 scheme over
the union WSSC(MN) by default, Janetzko's original eq-10 scheme on request.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibeqc._vibeqc_core import Molecule
from vibeqc._vibeqc_core import semiempirical as _se_cxx
from vibeqc.semiempirical.routes import SemiempiricalRoutePlan

from ._adapter_common import flatten_topology_records, topology_length_unit_scale
from .topology import SECCMTopology

_VARIANTS = frozenset({"om2", "om3"})

_THREE_CENTER_WEIGHTINGS = frozenset({"peintinger_eq13", "janetzko_eq10"})

_TRIVIAL_RECORD_WEIGHT_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class OMxSECCMResult:
    """One finite-cluster OMx-SECCM energy, per primitive cell."""

    energy: float
    e_electronic: float
    e_core: float
    total_cyclic_energy: float
    cyclic_core_energy: float
    homo_lumo_gap: float
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
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
    three_center_weighting: str
    topology_fingerprint: str
    reference_geometry_fingerprint: str
    route_plan: SemiempiricalRoutePlan
    truncated_electrostatics_acknowledged: bool = False


def _frozen_array(value) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def run_omx_seccm(
    molecule: Molecule,
    topology: SECCMTopology,
    *,
    variant: str = "om2",
    max_iter: int = 200,
    conv_tol: float = 1.0e-7,
    three_center_weighting: str = "peintinger_eq13",
    allow_truncated_electrostatics: bool = False,
) -> OMxSECCMResult:
    """Evaluate the OMx-SECCM route (WS-weighted supercell Fock).

    ``variant`` selects the OM2/OM3 parameter set. Neutral closed-shell
    systems only. The molecular limit (one replica) reproduces the
    molecular ``run_omx_v2`` driver.

    OM1-SECCM fails closed because the published analytic core-valence ECP is
    not implemented. The molecular OM1 development route remains separately
    warning-gated; extending its incomplete Hamiltonian over a cyclic cluster
    would silently amplify that missing term.

    ``three_center_weighting`` selects the image weighting of the
    three-center terms (the G1/G2 VORT corrections and the OM2/OM3 ECP):

    - ``"peintinger_eq13"`` (default): ``x_muNuC = x_muNu * (x_muC +
      x_nuC)/2`` with the center weights from the union
      ``WSSC(MN) = WSSC(M) u WSSC(N)`` (Peintinger-Bredow 2014, eq 13).
    - ``"janetzko_eq10"`` (opt-in comparison): ``x_muNuC = x_muNu *
      x_muC * x_nuC/(x_muC + x_nuC)`` with the center weights from the
      individual WSSCs (eq 10).

    Cyclic-image calculations fail closed by default because this adapter does
    not yet include the self-consistent Madelung/Ewald terms required for ionic
    crystals. This includes an order-one finite group if any record wraps
    across the cyclic boundary or has fractional ownership.
    ``allow_truncated_electrostatics=True`` explicitly acknowledges an
    algorithm-mechanics-only calculation with that long-range contribution
    omitted; such a result is not a quantitative solid-state prediction. The
    acknowledgement is recorded on both the result and its immutable route
    plan.
    """
    key = str(variant).strip().lower()
    if key == "om1":
        raise NotImplementedError(
            "OM1-SECCM is unavailable because the published analytic "
            "core-valence ECP is not implemented; use OM2- or OM3-SECCM"
        )
    if key not in _VARIANTS:
        raise ValueError(
            f"OMx-SECCM variant must be one of {sorted(_VARIANTS)!r}; got "
            f"{variant!r}"
        )
    if three_center_weighting not in _THREE_CENTER_WEIGHTINGS:
        raise ValueError(
            "OMx-SECCM three_center_weighting must be one of "
            f"{sorted(_THREE_CENTER_WEIGHTINGS)!r}; got "
            f"{three_center_weighting!r}"
        )
    if type(allow_truncated_electrostatics) is not bool:
        raise ValueError(
            "OMx-SECCM allow_truncated_electrostatics must be boolean"
        )
    base_route_plan = SemiempiricalRoutePlan.from_request(
        key,
        boundary="seccm",
        properties=("energy",),
        charge=int(molecule.charge),
        multiplicity=int(molecule.multiplicity),
    )
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
    ) = flatten_topology_records(molecule, topology, route_name="OMx-SECCM")

    group = topology.finite_group
    assert group is not None
    non_molecular_records = bool(np.any(shell_labels != 0)) or any(
        abs(float(weight) - 1.0) > _TRIVIAL_RECORD_WEIGHT_TOLERANCE
        for weight in weights
    )
    truncated_electrostatics_required = (
        group.order > 1 or non_molecular_records
    )
    truncated_electrostatics_acknowledged = (
        allow_truncated_electrostatics
        and truncated_electrostatics_required
    )
    if (
        truncated_electrostatics_required
        and not allow_truncated_electrostatics
    ):
        raise NotImplementedError(
            "OM2-/OM3-SECCM cyclic-image calculations require "
            "self-consistent long-range Madelung/Ewald electrostatics, which "
            "this adapter does not implement. The exact molecular limit "
            "(one replica, zero-translation records with full weights) "
            "remains available. Set "
            "allow_truncated_electrostatics=True only to acknowledge an "
            "algorithm-mechanics-only calculation that is not a quantitative "
            "solid-state result."
        )
    route_plan = base_route_plan.with_seccm_runtime(
        periodic_dimension=topology.dimensionality,
        three_center_weighting=three_center_weighting,
        truncated_electrostatics_acknowledged=(
            truncated_electrostatics_acknowledged
        ),
    )
    from vibeqc.semiempirical.methods.omx_params import load_omx_params

    params = load_omx_params(key)
    native = _se_cxx._run_omx_seccm_from_records(
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
        three_center_weighting,
        allow_truncated_electrostatics,
    )
    if not bool(native.converged):
        raise RuntimeError(
            "C++ OMx-SECCM did not converge within the iteration budget"
        )
    native_weighting = str(native.three_center_weighting)
    if native_weighting != three_center_weighting:
        raise RuntimeError(
            "C++ OMx-SECCM returned inconsistent three-center weighting "
            "provenance"
        )
    native_acknowledgement = bool(
        native.truncated_electrostatics_acknowledged
    )
    if (
        native_acknowledgement
        != truncated_electrostatics_acknowledged
    ):
        raise RuntimeError(
            "C++ OMx-SECCM returned inconsistent truncated-electrostatics "
            "provenance"
        )
    return OMxSECCMResult(
        energy=float(native.energy),
        e_electronic=float(native.e_electronic),
        e_core=float(native.e_core),
        total_cyclic_energy=float(native.total_cyclic_energy),
        cyclic_core_energy=float(native.cyclic_core_energy),
        homo_lumo_gap=float(native.homo_lumo_gap),
        mo_energies=_frozen_array(native.mo_energies),
        mo_coeffs=_frozen_array(native.mo_coeffs),
        density=_frozen_array(native.density),
        n_basis=int(native.n_basis),
        n_occ=int(native.n_occ),
        n_iter=int(native.n_iter),
        group_order=int(native.group_order),
        n_records=int(native.n_records),
        converged=bool(native.converged),
        normalization="per_primitive_cell",
        parameter_set=f"OM{key[-1]} published parameters (Dral 2016)",
        parameter_identity=str(native.parameter_identity),
        parameter_sha256=str(native.parameter_sha256),
        three_center_weighting=native_weighting,
        truncated_electrostatics_acknowledged=native_acknowledgement,
        topology_fingerprint=topology.topology_fingerprint,
        reference_geometry_fingerprint=topology.reference_geometry_fingerprint,
        route_plan=route_plan,
    )


__all__ = [
    "OMxSECCMResult",
    "run_omx_seccm",
]
