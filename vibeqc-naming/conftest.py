"""Test bootstrap: run the distribution tests from a checkout without an install.

Mirrors ``vibe-basis/conftest.py``. When ``vibeqc_naming`` is not installed,
the repository's ``python/`` directory, which holds the package source, goes
on ``sys.path``. An installed copy (CI installs the built wheel) is left in
charge, so the tests then exercise what was installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("vibeqc_naming") is None:
    SOURCE = Path(__file__).resolve().parent.parent / "python"
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
