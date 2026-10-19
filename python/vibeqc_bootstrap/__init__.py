"""Console boundary for basis-integrity failures before package initialization."""
from __future__ import annotations

import importlib
import sys


def _dispatch(module: str) -> int:
    try:
        return int(importlib.import_module(module).main() or 0)
    except RuntimeError as exc:
        # Package initialization can fail before its exception class can be
        # imported. Compare the defining module and class, not message text.
        if (type(exc).__module__ == "vibeqc._basis_overlay"
                and type(exc).__name__ == "BasisOverlayIntegrityError"):
            print(str(exc), file=sys.stderr)
            return exc.exit_status
        raise


def main() -> int:
    return _dispatch("vibeqc._cli")


def branded_main() -> int:
    return _dispatch("vibeqc.cli")
