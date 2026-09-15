"""Test bootstrap: add src/ to sys.path without requiring install.

Mirrors vibe-queue's pattern — running ``pytest`` inside vibe-basis/
finds ``vibe_basis`` via this conftest, so contributors don't have
to ``pip install -e .`` before iterating on tests.

Production / CI runs DO ``pip install -e .`` (which is the path
the ``vb`` CLI entry point also takes).
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
