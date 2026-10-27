"""Atomic JSON artifacts for parameter-opt; no partial checkpoints on failure."""

import json
import os
import tempfile
from pathlib import Path


def write_parameter_opt(path, document):
    target = Path(path)
    if target.suffix != ".json" or document.get("schema") != "vibeqc.parameter-opt.v1":
        raise ValueError("parameter-opt requires a versioned .json artifact")
    payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target
