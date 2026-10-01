"""QCSchema JSON file writer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .._text_safety import safe_json_bytes


def write_qcschema(path: str | Path, document: Mapping[str, Any]) -> Path:
    """Write a QCSchema molecule, atomic input, or atomic result to JSON."""
    target = Path(path)
    if target.suffix.lower() != ".json":
        raise ValueError("QCSchema output path must end in .json")
    if document.get("schema_name") not in {
        "qcschema_molecule", "qcschema_input", "qc_schema_input",
        "qcschema_output", "qc_schema_output",
    }:
        raise ValueError("document is not a QCSchema molecule, input, or output")
    json.dumps(document, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(safe_json_bytes(dict(document), indent=2) + b"\n")
    return target
