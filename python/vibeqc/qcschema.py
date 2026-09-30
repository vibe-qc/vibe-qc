"""Molecular QCSchema v1 input and result interchange.

Coordinates are in bohr and energies in Hartree, as in QCSchema. The
adapter deliberately rejects molecule features the native ``Molecule``
cannot represent rather than silently dropping them.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ._vibeqc_core import Atom, BasisSet, Molecule
from .output.formats.xyz import _ATOMIC_SYMBOLS


def read_qcschema(path: str | Path) -> dict[str, Any]:
    """Read a QCSchema JSON document from disk."""
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON number in QCSchema document: {value}")

    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream, parse_constant=reject_constant)
    if not isinstance(document, dict):
        raise ValueError("QCSchema document must be a JSON object")
    return document


def molecule_from_qcschema(data: Mapping[str, Any]) -> Molecule:
    """Construct a native molecule from QCSchema molecule data."""
    if (
        data.get("schema_name") != "qcschema_molecule"
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 2
    ):
        raise ValueError("expected qcschema_molecule schema_version 2")
    supported = {
        "schema_name", "schema_version", "symbols", "geometry", "atomic_numbers",
        "molecular_charge", "molecular_multiplicity", "real", "fix_com",
        "fix_orientation", "provenance",
    }
    present = sorted(set(data) - supported)
    if present:
        raise ValueError(f"unsupported QCSchema molecule fields: {', '.join(present)}")
    real = data.get("real", [])
    if not isinstance(real, list) or any(value is not True for value in real):
        raise ValueError("QCSchema ghost atoms or invalid real flags are not supported")
    symbols = data.get("symbols")
    geometry = data.get("geometry")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("QCSchema molecule needs a nonempty symbols array")
    if not isinstance(geometry, list) or len(geometry) != 3 * len(symbols):
        raise ValueError("QCSchema geometry must contain three coordinates per atom")
    if "real" in data and len(real) != len(symbols):
        raise ValueError("QCSchema real must contain one flag per atom")
    for flag in ("fix_com", "fix_orientation"):
        if flag in data and not isinstance(data[flag], bool):
            raise ValueError(f"QCSchema {flag} must be a boolean")
    numbers = data.get("atomic_numbers")
    if numbers is not None and (not isinstance(numbers, list) or len(numbers) != len(symbols)):
        raise ValueError("QCSchema atomic_numbers must contain one number per atom")
    atoms = []
    for index, symbol in enumerate(symbols):
        if not isinstance(symbol, str) or symbol not in _ATOMIC_SYMBOLS[1:]:
            raise ValueError(f"unsupported QCSchema element symbol: {symbol!r}")
        number = _ATOMIC_SYMBOLS.index(symbol)
        if numbers is not None and (
            isinstance(numbers[index], bool) or numbers[index] != number
        ):
            raise ValueError(f"atomic number does not match symbol {symbol!r}")
        coords = geometry[3 * index:3 * index + 3]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in coords
        ):
            raise ValueError("QCSchema geometry must contain finite numbers")
        atoms.append(Atom(number, [float(value) for value in coords]))
    charge = data.get("molecular_charge", 0)
    multiplicity = data.get("molecular_multiplicity", 1)
    if (
        isinstance(charge, bool)
        or not isinstance(charge, (int, float))
        or not math.isfinite(charge)
        or not float(charge).is_integer()
    ):
        raise ValueError("vibe-qc requires an integer molecular charge")
    if (
        isinstance(multiplicity, bool)
        or not isinstance(multiplicity, (int, float))
        or not math.isfinite(multiplicity)
        or not float(multiplicity).is_integer()
    ):
        raise ValueError("vibe-qc requires an integer molecular multiplicity")
    return Molecule(atoms, int(charge), int(multiplicity))


def molecule_to_qcschema(molecule: Molecule) -> dict[str, Any]:
    """Return a QCSchema molecule with the native atom order and Bohr positions."""
    atoms = list(molecule.atoms)
    symbols = []
    geometry = []
    for atom in atoms:
        number = int(atom.Z)
        if not 1 <= number < len(_ATOMIC_SYMBOLS):
            raise ValueError(f"QCSchema symbol unavailable for atomic number {number}")
        symbols.append(_ATOMIC_SYMBOLS[number])
        geometry.extend(float(value) for value in atom.xyz)
    return {
        "schema_name": "qcschema_molecule",
        "schema_version": 2,
        "symbols": symbols,
        "geometry": geometry,
        "molecular_charge": int(molecule.charge),
        "molecular_multiplicity": int(molecule.multiplicity),
        "fix_com": True,
        "fix_orientation": True,
    }


def _method_route(method: str, multiplicity: int) -> tuple[str, str | None]:
    name = method.strip().lower()
    if name in {"hf", "scf"}:
        return ("uhf" if multiplicity > 1 else "rhf"), None
    if name in {"rhf", "uhf"}:
        return name, None
    if name in {"mp2", "ccsd", "ccsd(t)", "cisd", "fci"}:
        return name, None
    return ("uks" if multiplicity > 1 else "rks"), method


def _run_atomic_input(data: Mapping[str, Any], output_stem: Path) -> dict[str, Any]:
    if (
        data.get("schema_name") not in {"qcschema_input", "qc_schema_input"}
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
    ):
        raise ValueError("expected qcschema_input schema_version 1")
    unsupported = sorted(set(data) - {
        "schema_name", "schema_version", "molecule", "driver", "model",
        "keywords", "provenance",
    })
    if unsupported:
        raise ValueError(f"unsupported QCSchema input fields: {', '.join(unsupported)}")
    driver = data.get("driver")
    if driver not in {"energy", "gradient", "hessian"}:
        raise ValueError(f"unsupported QCSchema driver: {driver!r}")
    molecule_data = data.get("molecule")
    if not isinstance(molecule_data, Mapping):
        raise ValueError("QCSchema input needs a molecule object")
    molecule = molecule_from_qcschema(molecule_data)
    model = data.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("method"), str) or not model["method"].strip():
        raise ValueError("QCSchema model needs a method string")
    if set(model) - {"method", "basis"}:
        raise ValueError("unsupported QCSchema model fields")
    basis_name = model.get("basis")
    if not isinstance(basis_name, str) or not basis_name:
        raise ValueError("QCSchema model needs a basis name string")
    keywords = data.get("keywords", {})
    if not isinstance(keywords, Mapping) or keywords:
        raise ValueError("QCSchema keywords are not supported by this adapter")
    method, functional = _method_route(model["method"], molecule.multiplicity)
    if driver != "energy" and method not in {"rhf", "uhf"}:
        raise NotImplementedError(f"QCSchema {driver} currently supports HF only")

    from . import __version__, compute_gradient, compute_gradient_uhf
    from .hessian import compute_hessian_fd
    from .runner import run_job

    result = run_job(
        molecule, basis=basis_name, method=method, functional=functional,
        output=output_stem, output_qvf=False, write_molden_file=False,
        write_xyz_file=False, write_population_file=False, citations=False,
        progress=False,
    )
    if not result.converged or (
        hasattr(result, "ccsd") and not result.ccsd.converged
    ):
        raise RuntimeError("QCSchema job did not converge")
    total_energy = getattr(result, "energy_total", result.energy)
    energy = float(total_energy() if callable(total_energy) else total_energy)
    if driver == "energy":
        return_result: float | list[float] = energy
    elif driver == "gradient":
        basis = BasisSet(molecule, basis_name)
        gradient_fn = compute_gradient if method == "rhf" else compute_gradient_uhf
        return_result = np.asarray(gradient_fn(molecule, basis, result), dtype=float).ravel().tolist()
    else:
        hessian_result = compute_hessian_fd(molecule, basis_name, method=method.upper())
        return_result = np.asarray(hessian_result.hessian, dtype=float).ravel().tolist()
    electrons = molecule.n_electrons()
    nalpha = (electrons + molecule.multiplicity - 1) // 2
    properties = {
        "return_energy": energy,
        "calcinfo_natom": len(list(molecule.atoms)),
        "calcinfo_nalpha": nalpha,
        "calcinfo_nbeta": electrons - nalpha,
    }
    if method in {"rhf", "uhf", "rks", "uks"} or energy != float(result.energy):
        properties["scf_total_energy"] = float(result.energy)
    return {
        "schema_name": "qcschema_output",
        "schema_version": 1,
        "molecule": dict(molecule_data),
        "driver": driver,
        "model": dict(model),
        "keywords": dict(keywords),
        "provenance": {"creator": "vibe-qc", "version": __version__, "routine": "run_qcschema"},
        "success": True,
        "return_result": return_result,
        "properties": properties,
    }


def run_qcschema(
    input_data: Mapping[str, Any] | str | Path,
    *,
    output_path: str | Path | None = None,
    output_stem: str | Path | None = None,
) -> dict[str, Any]:
    """Run a QCSchema atomic input and optionally write an atomic result JSON.

    If ``output_stem`` is omitted, native job sidecars are temporary. Errors
    in unsupported inputs are raised before a calculation begins.
    """
    data = read_qcschema(input_data) if isinstance(input_data, (str, Path)) else input_data
    if not isinstance(data, Mapping):
        raise ValueError("QCSchema input must be an object")
    if output_path is not None and Path(output_path).suffix.lower() != ".json":
        raise ValueError("QCSchema output path must end in .json")
    if output_stem is None:
        with tempfile.TemporaryDirectory(prefix="vibeqc-qcschema-") as directory:
            document = _run_atomic_input(data, Path(directory) / "job")
    else:
        document = _run_atomic_input(data, Path(output_stem))
    if output_path is not None:
        from .output.formats.qcschema import write_qcschema
        write_qcschema(output_path, document)
    return document
