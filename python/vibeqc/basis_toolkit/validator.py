"""Validator for QVF-Basis data against the canonical JSON Schema.

Provides three entry points -- dict, JSON string, and file -- plus a
standalone CLI: ``python -m vibeqc.basis_toolkit.validator <file>``.

When ``jsonschema`` is installed, validation is full and exact.  When
it is not available, a lightweight structural check is used instead;
the first element of the returned error list is always the warning
``"jsonschema not available; full schema validation skipped (structural checks only)"``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ._element_data import VALID_SYMBOLS

# ---------------------------------------------------------------------------
# jsonschema availability
# ---------------------------------------------------------------------------

try:
    import jsonschema  # noqa: F401

    _has_jsonschema = True
except ImportError:
    _has_jsonschema = False

# ---------------------------------------------------------------------------
# Schema path
# ---------------------------------------------------------------------------


def _schema_path() -> Path:
    """Absolute path to the canonical QVF-Basis v1 JSON Schema file."""
    return Path(__file__).resolve().parent / "schemas" / "qvf_basis_v1.schema.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_basis_dict(data: dict) -> list[str]:
    """Validate a ``BasisSetData`` dict against the QVF-Basis v1 schema.

    Parameters
    ----------
    data : dict
        Plain dict conforming to (or purporting to conform to) the
        QVF-Basis v1 structure.

    Returns
    -------
    list[str]
        List of human-readable error messages.  An empty list means the
        data passed validation.  When ``jsonschema`` is not installed,
        the first message is always a warning about the lightweight
        fallback.
    """
    if _has_jsonschema:
        return _validate_jsonschema(data)
    else:
        return _validate_lightweight(data)


def validate_basis_json(json_str: str) -> list[str]:
    """Parse *json_str* as JSON and validate the result.

    Parameters
    ----------
    json_str : str
        A JSON string representing a basis-set dict.

    Returns
    -------
    list[str]
        List of error messages (empty = valid).  JSON parse errors are
        returned as a single-element list.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return ["Top-level value must be a JSON object (dict)."]
    return validate_basis_dict(data)


def validate_basis_file(path: str | Path) -> list[str]:
    """Read *path* as JSON and validate its contents.

    Parameters
    ----------
    path : str or Path
        Path to a ``.qvf.json`` (or plain ``.json``) file.

    Returns
    -------
    list[str]
        List of error messages (empty = valid).  File-not-found and
        JSON-parse errors are returned as single-element lists.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [f"File not found: {path}"]
    except OSError as exc:
        return [f"Cannot read {path}: {exc}"]
    return validate_basis_json(raw)


# ---------------------------------------------------------------------------
# jsonschema path
# ---------------------------------------------------------------------------


def _validate_jsonschema(data: dict) -> list[str]:
    """Full validation via ``jsonschema`` (when available)."""
    schema_path = _schema_path()
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)

    # Re-import jsonschema inside the guarded path so type checkers that
    # see the ImportError branch don't flag it as possibly-unbound.
    import jsonschema  # type: ignore[import-not-found]

    # Use the modern referencing.Registry for jsonschema >= 4.18.
    # Fall back to the deprecated RefResolver for older versions.
    try:
        from referencing import Registry, Resource

        schema_resource = Resource.from_contents(schema)
        registry = Registry().with_resource(schema_path.as_uri(), schema_resource)
        validator_cls = jsonschema.validators.validator_for(schema)
        v = validator_cls(schema, registry=registry)
    except ImportError:
        # jsonschema < 4.18 path (RefResolver, deprecated but functional)
        validator_cls = jsonschema.validators.validator_for(schema)
        try:
            resolver = jsonschema.RefResolver(
                base_uri=schema_path.as_uri(),
                referrer=schema,
            )
        except Exception:
            resolver = None
        v = (
            validator_cls(schema, resolver=resolver)
            if resolver
            else validator_cls(schema)
        )

    errors: list[str] = []
    for err in sorted(v.iter_errors(data), key=lambda e: e.path):
        path_str = "/" + "/".join(str(p) for p in err.path) if err.path else "(root)"
        errors.append(f"{path_str}: {err.message}")
    return errors


# ---------------------------------------------------------------------------
# Lightweight structural validation (no jsonschema)
# ---------------------------------------------------------------------------


def _validate_lightweight(data: dict) -> list[str]:
    """Structural checks that don't require the ``jsonschema`` package.

    These cover the most-common mistakes (missing required fields, wrong
    types, invalid element symbols, exponents <= 0) but are not a
    complete substitute for the JSON Schema.
    """
    errors: list[str] = [
        "jsonschema not available; full schema validation skipped (structural checks only)"
    ]

    if not isinstance(data, dict):
        errors.append("Top-level value must be a dict.")
        return errors

    # --- top-level required fields ------------------------------------------
    for field in ("schema_version", "name", "role", "elements"):
        if field not in data:
            errors.append(f"Missing required top-level field: '{field}'.")

    if "schema_version" in data:
        sv = data["schema_version"]
        if not isinstance(sv, str):
            errors.append("'schema_version' must be a string.")
        elif not sv.startswith("1."):
            errors.append(f"'schema_version' must start with '1.' (got {sv!r}).")

    if "name" in data:
        if not isinstance(data["name"], str) or data["name"] == "":
            errors.append("'name' must be a non-empty string.")

    if "role" in data:
        role = data["role"]
        if role not in ("orbital", "auxiliary", "fitting", "ecp"):
            errors.append(
                f"'role' must be one of orbital/auxiliary/fitting/ecp (got {role!r})."
            )

    # --- elements -----------------------------------------------------------
    if "elements" in data:
        elements = data["elements"]
        if not isinstance(elements, dict):
            errors.append("'elements' must be a dict (string -> element basis).")
        elif len(elements) == 0:
            errors.append("'elements' must contain at least one entry.")
        else:
            for symbol, ed in elements.items():
                if symbol not in VALID_SYMBOLS:
                    errors.append(f"Invalid element symbol as key: {symbol!r}.")
                if not isinstance(ed, dict):
                    errors.append(
                        f"Element '{symbol}' value must be a dict, got {type(ed).__name__}."
                    )
                else:
                    errors.extend(_check_element(symbol, ed))

    # --- ecps (optional) ----------------------------------------------------
    if "ecps" in data and data["ecps"] is not None:
        ecps = data["ecps"]
        if not isinstance(ecps, dict):
            errors.append("'ecps' must be a dict (string -> ECP entry).")
        else:
            for symbol, ecp in ecps.items():
                if symbol not in VALID_SYMBOLS:
                    errors.append(f"Invalid element symbol in ecps: {symbol!r}.")
                if not isinstance(ecp, dict):
                    errors.append(
                        f"ECP '{symbol}' must be a dict, got {type(ecp).__name__}."
                    )
                else:
                    errors.extend(_check_ecp(symbol, ecp))

    # --- references (optional) ----------------------------------------------
    if "references" in data:
        refs = data["references"]
        if not isinstance(refs, list):
            errors.append("'references' must be a list.")
        else:
            for i, ref in enumerate(refs):
                if not isinstance(ref, dict):
                    errors.append(f"references[{i}] must be a dict.")
                elif "key" not in ref:
                    errors.append(f"references[{i}] missing required field 'key'.")
                elif not isinstance(ref["key"], str) or ref["key"] == "":
                    errors.append(f"references[{i}].key must be a non-empty string.")

    # --- provenance (optional) ----------------------------------------------
    if "provenance" in data and data["provenance"] is not None:
        prov = data["provenance"]
        if not isinstance(prov, dict):
            errors.append("'provenance' must be a dict.")
        elif "origin" not in prov:
            errors.append("'provenance' missing required field 'origin'.")

    return errors


def _check_element(symbol: str, ed: dict) -> list[str]:
    """Validate a single ``ElementBasis`` dict (lightweight)."""
    errors: list[str] = []

    if "element" not in ed:
        errors.append(f"Element '{symbol}' missing required field 'element'.")
    elif ed["element"] != symbol:
        errors.append(
            f"Element key '{symbol}' does not match 'element' field value {ed['element']!r}."
        )

    if "shells" not in ed:
        errors.append(f"Element '{symbol}' missing required field 'shells'.")
        return errors

    shells = ed["shells"]
    if not isinstance(shells, list):
        errors.append(f"Element '{symbol}'.shells must be a list.")
        return errors
    if len(shells) == 0:
        errors.append(f"Element '{symbol}'.shells must contain at least one shell.")
        return errors

    for i, shell in enumerate(shells):
        if not isinstance(shell, dict):
            errors.append(f"{symbol}.shells[{i}] must be a dict.")
            continue
        errors.extend(_check_shell(f"{symbol}.shells[{i}]", shell))

    if "ecp_id" in ed and not isinstance(ed["ecp_id"], str):
        errors.append(f"Element '{symbol}'.ecp_id must be a string.")

    return errors


def _check_shell(path: str, shell: dict) -> list[str]:
    """Validate a single ``Shell`` dict (lightweight)."""
    errors: list[str] = []

    # angular_momentum
    if "angular_momentum" not in shell:
        errors.append(f"{path}: missing required field 'angular_momentum'.")
    else:
        am = shell["angular_momentum"]
        if not isinstance(am, list):
            errors.append(f"{path}.angular_momentum must be a list.")
        elif len(am) < 1 or len(am) > 2:
            errors.append(
                f"{path}.angular_momentum must have 1 or 2 elements, got {len(am)}."
            )
        else:
            for j, l_val in enumerate(am):
                if not isinstance(l_val, int):
                    errors.append(
                        f"{path}.angular_momentum[{j}] must be an integer, got {type(l_val).__name__}."
                    )
                elif l_val < 0 or l_val > 12:
                    errors.append(
                        f"{path}.angular_momentum[{j}] = {l_val} out of range [0, 12]."
                    )

    # harmonic_type
    if "harmonic_type" not in shell:
        errors.append(f"{path}: missing required field 'harmonic_type'.")
    else:
        ht = shell["harmonic_type"]
        if ht not in ("spherical", "cartesian"):
            errors.append(
                f"{path}.harmonic_type must be 'spherical' or 'cartesian', got {ht!r}."
            )

    # primitives
    if "primitives" not in shell:
        errors.append(f"{path}: missing required field 'primitives'.")
    else:
        prims = shell["primitives"]
        if not isinstance(prims, list):
            errors.append(f"{path}.primitives must be a list.")
        elif len(prims) == 0:
            errors.append(f"{path}.primitives must contain at least one primitive.")
        else:
            for k, prim in enumerate(prims):
                if not isinstance(prim, dict):
                    errors.append(f"{path}.primitives[{k}] must be a dict.")
                    continue
                errors.extend(_check_primitive(f"{path}.primitives[{k}]", prim))

    return errors


def _check_primitive(path: str, prim: dict) -> list[str]:
    """Validate a single ``Primitive`` dict (lightweight)."""
    errors: list[str] = []

    if "exponent" not in prim:
        errors.append(f"{path}: missing required field 'exponent'.")
    else:
        exp = prim["exponent"]
        if not isinstance(exp, (int, float)):
            errors.append(
                f"{path}.exponent must be a number, got {type(exp).__name__}."
            )
        elif exp <= 0:
            errors.append(f"{path}.exponent must be > 0, got {exp}.")

    if "coefficient" not in prim:
        errors.append(f"{path}: missing required field 'coefficient'.")
    else:
        coeff = prim["coefficient"]
        if not isinstance(coeff, (int, float)):
            errors.append(
                f"{path}.coefficient must be a number, got {type(coeff).__name__}."
            )

    return errors


def _check_ecp(symbol: str, ecp: dict) -> list[str]:
    """Validate a single ``ECPEntry`` dict (lightweight)."""
    errors: list[str] = []

    for field in ("element", "n_core_electrons", "angular_momentum_max", "potentials"):
        if field not in ecp:
            errors.append(f"ECP '{symbol}' missing required field '{field}'.")

    if "potentials" in ecp:
        pots = ecp["potentials"]
        if isinstance(pots, list):
            for i, pot in enumerate(pots):
                if not isinstance(pot, dict):
                    errors.append(f"ECP '{symbol}'.potentials[{i}] must be a dict.")
                else:
                    if "angular_momentum" not in pot:
                        errors.append(
                            f"ECP '{symbol}'.potentials[{i}] missing 'angular_momentum'."
                        )
                    if "terms" in pot and isinstance(pot["terms"], list):
                        for j, term in enumerate(pot["terms"]):
                            if isinstance(term, dict):
                                for tf in ("coefficient", "exponent", "power"):
                                    if tf not in term:
                                        errors.append(
                                            f"ECP '{symbol}'.potentials[{i}].terms[{j}] missing '{tf}'."
                                        )

    return errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m vibeqc.basis_toolkit.validator <file>``."""
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        print(
            "Usage: python -m vibeqc.basis_toolkit.validator <file.qvf.json>",
            file=sys.stderr,
        )
        sys.exit(2)

    path = argv[1]
    errors = validate_basis_file(path)

    if not errors:
        print(f"✓ Valid: {path}")
        sys.exit(0)

    print(f"✗ Validation errors in {path}:")
    for msg in errors:
        print(f"  - {msg}")
    sys.exit(1)


if __name__ == "__main__":
    main()
