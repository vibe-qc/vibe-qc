"""Normalized reference ingestion and explicit numerical qualification.

External engines run as separate jobs. These functions consume their normalized
results, never import an external QC package or infer convergence from an energy.
"""

import json
from dataclasses import asdict
from pathlib import Path

from .datasets import ReferenceProtocol, ReferenceRecord
from .schema import System, digest, finite


def read_references(path):
    """Read versioned electronic-energy records; reject implicit unit conversion."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "vibeqc.parameter-opt.references.v1":
        raise ValueError("unsupported reference schema")
    if data.get("coordinate_unit") != "bohr" or data.get("energy_unit") != "hartree":
        raise ValueError("reference input must explicitly use bohr and hartree")
    result = []
    for item in data["records"]:
        item = dict(item)
        result.append(
            ReferenceRecord(
                system=System(**item.pop("system")),
                protocol=ReferenceProtocol(**item.pop("protocol")),
                **item,
            )
        )
    if not result or len({r.sha256 for r in result}) != len(result):
        raise ValueError("nonempty unique reference records required")
    return tuple(result)


def convergence_check(coarse, fine, *, varied, tolerance):
    """One-pair sensitivity evidence, not a general accuracy certification.

    `varied` names changed protocol fields (basis or settings keys). All other
    settings must match. Auxiliary-basis changes can accompany basis studies.
    """
    finite(tolerance, "convergence tolerance")
    if tolerance <= 0:
        raise ValueError("positive convergence tolerance required")
    if coarse.system.geometry_sha256 != fine.system.geometry_sha256:
        raise ValueError("convergence check requires identical physical systems")
    varied = frozenset(varied)
    allowed = {"basis", "grid", "pno", "scf", "aux_basis"}
    if not varied or not varied <= allowed:
        raise ValueError("unsupported convergence study dimension")

    def flattened(protocol):
        data = asdict(protocol)
        settings = json.loads(data.pop("settings_json"))
        # Namespaced settings avoid collision with the protocol identity.
        return {**data, **{"settings." + k: v for k, v in settings.items()}}

    a, b = flattened(coarse.protocol), flattened(fine.protocol)
    changed = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    expected = {k if k == "basis" else "settings." + k for k in varied}
    if changed != expected:
        raise ValueError(
            f"changed protocol dimensions {sorted(changed)} differ from declared study"
        )
    difference = abs(fine.energy - coarse.energy)
    return {
        "coarse_sha256": coarse.sha256,
        "fine_sha256": fine.sha256,
        "varied": sorted(varied),
        "absolute_difference_hartree": difference,
        "tolerance_hartree": tolerance,
        "within_tolerance": difference <= tolerance,
        "scope": "single geometry electronic energy sensitivity",
    }


def reference_request(system, protocol):
    """Portable calculation specification for existing vq job producers.

    This is a spec, not a runnable vq payload. Job producers must translate all
    protocol settings and return explicitly normalized, converged records.
    """
    spec = {
        "schema": "vibeqc.parameter-opt.reference-request.v1",
        "system": asdict(system),
        "protocol": asdict(protocol),
        "coordinate_unit": "bohr",
        "observable": "energy",
        "energy_unit": "hartree",
    }
    return {**spec, "request_sha256": digest(spec)}
