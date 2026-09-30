"""Locate the suite's companion checkouts, loudly on demand.

The 2026-09 repository split moved vibe-view and vibe-queue out of this tree.
They are siblings of the vibe-qc checkout (``../vibe-view``), never children
of it, so joining a companion name onto the repository root names a path that
cannot exist in any checkout -- developer, release, or lane mirror.
vibe-basis is the one component that is still co-located.

``scripts/_companion_paths.sh`` is the product's single answer to "where does
a companion live". This module asks that resolver instead of repeating the
join, so the shell scripts and the test suite cannot drift apart: if the
layout changes again, only the shell table moves.

A missing companion is a supported configuration -- a user who installed only
vibe-qc has none of them -- so the cross-component contract tests skip rather
than fail. The skip names the coverage that did not run and how to make it a
failure, following the same rule as :mod:`tests.trexio_reference` (#253):
``VIBEQC_REQUIRE_COMPANION_CHECKOUTS`` turns every such skip into a failure,
which is what a gate carrying the companion checkouts should set.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOLVER = REPO_ROOT / "scripts" / "_companion_paths.sh"
REQUIRE_VAR = "VIBEQC_REQUIRE_COMPANION_CHECKOUTS"
OVERRIDE_VARS = {
    "vibe-view": "VIBE_VIEW_ROOT",
    "vibe-queue": "VIBE_QUEUE_ROOT",
    "vibe-basis": "VIBE_BASIS_ROOT",
}

_FALSE = frozenset({"", "0", "false", "no", "off"})

_QUERY = """
. "$1"
if vibeqc_companion_root "$2" resolved; then present=1; else present=0; fi
printf '%s\\n%s\\n' "$present" "$resolved"
"""


def companions_required(environ: Mapping[str, str] | None = None) -> bool:
    """True when ``VIBEQC_REQUIRE_COMPANION_CHECKOUTS`` is set to a truthy value."""
    environ = os.environ if environ is None else environ
    return environ.get(REQUIRE_VAR, "").strip().lower() not in _FALSE


@lru_cache(maxsize=None)
def _resolve(component: str) -> tuple[bool, Path]:
    probe = subprocess.run(
        ["/bin/bash", "-c", _QUERY, "resolve", str(RESOLVER), component],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = probe.stdout.splitlines()
    if probe.returncode != 0 or len(lines) != 2 or not lines[1]:
        raise RuntimeError(
            f"{RESOLVER} could not resolve {component!r}: "
            f"{(probe.stderr or probe.stdout).strip()!r}"
        )
    return lines[0] == "1", Path(lines[1])


def expected_root(component: str) -> Path:
    """Where this checkout looks for ``component``, present or not."""
    return _resolve(component)[1]


def companion_root(component: str) -> Path | None:
    """The companion checkout, or ``None`` when it is not there."""
    present, root = _resolve(component)
    return root if present else None


def require(component: str, coverage: str) -> Path:
    """Return the companion checkout, or stop the test loudly.

    ``coverage`` names what does not run without it, so a reader of a skipped
    run can tell which contract went unchecked rather than seeing a bare
    "not installed".
    """
    root = companion_root(component)
    if root is not None:
        return root
    override = OVERRIDE_VARS[component]
    reason = (
        f"{coverage} did not run: no {component} checkout at "
        f"{expected_root(component)}. {component} is a separate repository "
        f"since the 2026-09 split; clone it there or set {override}"
    )
    if companions_required():
        pytest.fail(
            f"{reason}. {REQUIRE_VAR} is set, so a missing companion "
            "checkout is a failure rather than a skip."
        )
    pytest.skip(f"{reason}. Set {REQUIRE_VAR}=1 to make this a failure.")
    raise AssertionError("unreachable")
