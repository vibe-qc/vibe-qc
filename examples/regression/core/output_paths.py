from __future__ import annotations

import datetime as _dt
import os
import uuid
from pathlib import Path
from typing import Optional

RUNS_DIR_ENV = "VIBEQC_RUNS_DIR"
DEFAULT_OUTPUT_ROOT = Path("~/vibeqc-runs")


def resolve_output_root(
    cli_output_root: Optional[str] = None,
    *,
    environ: Optional[dict[str, str]] = None,
    create: bool = False,
) -> Path:
    """Resolve the root directory that owns regression run artifacts."""
    env = os.environ if environ is None else environ
    raw = cli_output_root or env.get(RUNS_DIR_ENV) or str(DEFAULT_OUTPUT_ROOT)
    root = Path(raw).expanduser().resolve(strict=False)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def make_run_id(prefix: str = "") -> str:
    """Return a timestamped run id, optionally prefixed by a tool name."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    base = f"{stamp}-{suffix}"
    return f"{prefix}-{base}" if prefix else base
