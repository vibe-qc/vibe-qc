"""Parameter file I/O for semiempirical registries.

Load and save DFTB element parameter sets using the TOML format.  Repulsive
data can be loaded from the same format; the DFTB saver currently writes the
element block only.

Example:
    # Save
    from vibeqc.semiempirical.io import save_parameters, load_production_parameters
    params = load_production_parameters()
    save_parameters(params, "my_params.toml")

    # Load
    from vibeqc.semiempirical.io import load_parameters
    params = load_parameters("my_params.toml")
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.basis_crystal import _ELEMENT_SYMBOLS
from vibeqc.output import Level, write
from vibeqc.output.formats.toml import write_toml

_METHODS_DIR = Path(__file__).with_name("methods")
_MSINDO_INDO_PATH = _METHODS_DIR / "msindo_params.json"
_MSINDO_NDDO_PATH = _METHODS_DIR / "msindo_params_nddo.json"


def save_parameters(params, path: str | Path) -> None:
    """Save element parameters to TOML.

    Repulsive blocks can be read by :func:`load_parameters`, but this writer
    intentionally preserves the long-standing element-only DFTB export.
    """
    path = Path(path)
    with open(path, "w") as f:
        f.write("# vibe-qc DFTB parameter file\n")
        f.write(f"kappa = {params.kappa}\n\n")

        for Z in range(1, 87):
            if not params.has_element(Z):
                continue
            f.write("[[element]]\n")
            f.write(f"Z = {Z}\n")

            eps = []
            for li in range(3):
                e = params.on_site_energy(Z, li)
                eps.append(e)
            f.write(f"on_site = [")
            f.write(", ".join(f"{v:.6f}" for v in eps))
            f.write("]\n")

            zs = []
            for li in range(3):
                z = params.sto_exponent(Z, li)
                zs.append(z)
            f.write(f"zeta = [")
            f.write(", ".join(f"{v:.6f}" for v in zs))
            f.write("]\n")

            f.write(f"hubbard_u = {params.hubbard_u(Z):.6f}\n")
            f.write(f"valence_electrons = {params.valence_electrons(Z)}\n\n")

    n_elements = len([Z for Z in range(1, 87) if params.has_element(Z)])
    write(f"Saved {n_elements} semiempirical elements to {path}\n", Level.VERBOSE)


def load_parameters(path: str | Path) -> _se.SemiempiricalParameters:
    """Load parameters from a TOML file.

    Handles element data and spline-fitted repulsive pairs.
    """
    import tomllib

    path = Path(path)
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    params = _se.SemiempiricalParameters()

    if "kappa" in data:
        params.kappa = float(data["kappa"])

    # Load elements
    for elem in data.get("element", []):
        Z = int(elem["Z"])
        on_site = [float(v) for v in elem["on_site"]]
        zeta = [float(v) for v in elem["zeta"]]
        U = float(elem.get("hubbard_u", 0.3))
        nval = int(elem.get("valence_electrons", 0))
        params.add_element(Z, on_site, zeta, U, nval)

    # Load repulsive pairs (spline or analytic)
    for rp_data in data.get("repulsive", []):
        Z1 = int(rp_data["Z1"])
        Z2 = int(rp_data["Z2"])

        if "R_bohr" in rp_data and "V_Ha" in rp_data:
            R = [float(v) for v in rp_data["R_bohr"]]
            V = [float(v) for v in rp_data["V_Ha"]]
            params.set_repulsive_pair_spline(Z1, Z2, R, V)
        elif "A" in rp_data:
            A = float(rp_data["A"])
            B = float(rp_data.get("B", 0.0))
            params.set_repulsive_pair_analytic(Z1, Z2, A, B)

    return params


def load_production_parameters():
    """Return the DFTB parameter set named by the production loader.

    Currently the same parameter set as :func:`load_default_parameters`:
    the spline-fitted production repulsives are deferred pending a
    validated DFT reference dataset, and every pair outside the
    {H, C, N, O, F, P, S, Cl} table falls back to the combining-rule
    ``A/R^12`` estimate (issue #306).
    """
    return _se.SemiempiricalParameters.dftb0_production()


def load_default_parameters():
    """Return default DFTB0 parameters (R^-12 repulsive)."""
    return _se.SemiempiricalParameters.dftb0_default()


def msindo_parameter_registry() -> dict[str, Any]:
    """Return the full MSINDO INDO/NDDO parameter registry.

    The returned mapping is TOML-serialisable and includes the complete bundled
    INDO H-Xe parameter table plus the NDDO override table for H, Li-F, and
    Na-Cl.  It is intended for article/supporting-data exports; runtime MSINDO
    still reads the JSON bundles directly.
    """
    indo = _load_json(_MSINDO_INDO_PATH)
    nddo = _load_json(_MSINDO_NDDO_PATH)
    indo_elements = _msindo_element_records(indo["elements"])
    nddo_elements = _msindo_element_records(nddo["elements"])

    return {
        "schema_version": 1,
        "kind": "vibeqc.msindo.parameters",
        "metadata": {
            "implementation": "vibe-qc independent MSINDO reimplementation",
            "scope": (
                "Full bundled INDO parameter table plus NDDO overrides; "
                "runtime source of truth remains the JSON bundles."
            ),
            "permission": (
                "MSINDO method and bundled parameters are used by permission of "
                "the Mulliken Center for Theoretical Chemistry, University of "
                "Bonn, Prof. T. Bredow."
            ),
            "indo_provenance": str(indo.get("_provenance", "")),
            "nddo_provenance": str(nddo.get("_provenance", "")),
            "nddo_note": str(nddo.get("note", "")),
        },
        "source": {
            "indo_json": "python/vibeqc/semiempirical/methods/msindo_params.json",
            "indo_sha256": _sha256(_MSINDO_INDO_PATH),
            "nddo_json": "python/vibeqc/semiempirical/methods/msindo_params_nddo.json",
            "nddo_sha256": _sha256(_MSINDO_NDDO_PATH),
        },
        "units": {
            "indo": str(indo.get("units", "")),
            "nddo": str(nddo.get("units", "")),
        },
        "al_groups": {
            "indo": indo.get("_al_groups", []),
            "nddo": nddo.get("al_groups", []),
        },
        "supported": {
            "indo_z": [row["Z"] for row in indo_elements],
            "nddo_z": [row["Z"] for row in nddo_elements],
            "indo_element_count": len(indo_elements),
            "nddo_element_count": len(nddo_elements),
        },
        "indo_element": indo_elements,
        "nddo_element": nddo_elements,
    }


def save_msindo_parameter_registry(path: str | Path) -> None:
    """Write the MSINDO INDO/NDDO parameter registry as TOML."""
    path = Path(path)
    write_toml(msindo_parameter_registry(), path)
    write(f"Saved MSINDO parameter registry to {path}\n", Level.VERBOSE)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _msindo_element_records(elements: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in sorted(elements, key=lambda item: int(item)):
        z = int(key)
        record = {"Z": z, "symbol": _element_symbol(z)}
        record.update(elements[key])
        records.append(record)
    return records


def _element_symbol(z: int) -> str:
    if 0 < z < len(_ELEMENT_SYMBOLS):
        return _ELEMENT_SYMBOLS[z]
    return f"Z{z}"
