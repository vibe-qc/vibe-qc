"""``vibeqc-outputs`` -- inspect a job's ``.system`` manifest.

Operator-visible counterpart of ``vibeqc-cite``: walks the
``[plan]`` + ``[outputs]`` sections of a ``{stem}.system`` manifest
(Phase O1's schema extension) and prints a human-readable summary --
which files were *declared* to be emitted, which actually landed,
the SHA-256 + byte counts, the run status (running / complete /
crashed / dry_run), and the wall-time budget.

Use cases:

* Post-``vq fetch`` sanity check: did everything come back?
* Debug a crashed job: which writers had finished before the
  exception?
* Audit a generated output set:
  ``vibeqc-outputs output-h2o-opt.system``
  shows every file declared by that run.

Reads only the manifest -- no need for vibeqc's compiled core or
even an installed ``vibeqc`` package. Plain stdlib ``tomllib``,
nothing else.

Exit codes
----------

* ``0`` -- manifest read + summary printed.
* ``1`` -- manifest missing or malformed.
* ``2`` -- manifest has no ``[plan]`` section (pre-Phase-O1 format).
* ``3`` -- used with ``--strict`` and at least one declared file is
  missing on disk.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Sequence


__all__ = ["main"]


_STATUS_DECORATIONS = {
    "complete":  "[OK]    ",
    "running":   "[RUN]   ",
    "crashed":   "[CRASH] ",
    "dry_run":   "[DRY]   ",
    "":          "[?]     ",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vibeqc-outputs",
        description=(
            "Inspect a vibe-qc job's .system manifest. Shows the "
            "declared output plan, which files actually landed on "
            "disk, and the overall job status. Reads only the "
            "manifest (no vibe-qc install required)."
        ),
    )
    p.add_argument(
        "stem",
        type=Path,
        help=(
            "Job stem or manifest path. Accepts 'output-h2o' / "
            "'output-h2o.out' / 'output-h2o.system' -- the path is "
            "normalised to the .system sibling internally."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero (3) if any declared 'always' file is "
            "missing or has written=false. Useful for CI gates "
            "verifying generated output sets are complete."
        ),
    )
    p.add_argument(
        "--paths-only",
        action="store_true",
        help=(
            "Print just the declared file paths, one per line. "
            "Useful for piping into other tools "
            "(e.g. `vibeqc-outputs job.system --paths-only | "
            "xargs ls -la`)."
        ),
    )
    p.add_argument(
        "--missing-only",
        action="store_true",
        help=(
            "Restrict the summary table to files whose written "
            "flag is False or whose path doesn't exist on disk."
        ),
    )
    return p


def _resolve_manifest_path(stem: Path) -> Path:
    """Accept ``output-h2o`` / ``output-h2o.out`` /
    ``output-h2o.system`` -- normalise to the ``.system`` sibling."""
    return stem.with_suffix(".system")


def _format_size(n: int | None) -> str:
    if n is None or n <= 0:
        return ""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _declared_paths(workspace: Path, declared_path: str) -> tuple[Path, ...]:
    """Candidate interpretations for old and current relative path rows."""
    p = Path(declared_path)
    if p.is_absolute():
        return (p,)
    # Plans historically preserved the caller's relative spelling, which may
    # be relative either to the inspection cwd or to the manifest directory.
    return (p, workspace / p)


def _exists_on_disk(workspace: Path, declared_path: str) -> bool:
    return any(
        path.is_file()
        for path in _declared_paths(workspace, declared_path)
    )


def _is_manifest_path(
    manifest_path: Path,
    workspace: Path,
    declared_path: str,
) -> bool:
    """Recognise the manifest's own row, including legacy path spellings."""
    for candidate in _declared_paths(workspace, declared_path):
        try:
            if candidate.resolve(strict=False) == manifest_path.resolve(
                strict=False,
            ):
                return True
        except OSError:
            if os.path.abspath(os.fspath(candidate)) == os.path.abspath(
                os.fspath(manifest_path),
            ):
                return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest_path = _resolve_manifest_path(args.stem)
    if not manifest_path.is_file():
        sys.stderr.write(
            f"vibeqc-outputs: manifest not found at {manifest_path}\n"
        )
        return 1
    try:
        body = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        sys.stderr.write(
            f"vibeqc-outputs: failed to parse {manifest_path}: "
            f"{type(exc).__name__}: {exc}\n"
        )
        return 1
    plan = body.get("plan")
    outputs = body.get("outputs")
    if not isinstance(plan, dict):
        sys.stderr.write(
            f"vibeqc-outputs: {manifest_path} has no [plan] section "
            "(pre-Phase-O1 manifest format). Re-run the job to get "
            "a current manifest.\n"
        )
        return 2

    declared = plan.get("files") or []
    outputs_files = (outputs or {}).get("files") or []
    # Index outputs by path so we can look up written-status fast.
    by_path: dict[str, dict[str, Any]] = {
        str(row.get("path", "")): dict(row) for row in outputs_files
    }
    workspace = manifest_path.parent

    if args.paths_only:
        for row in declared:
            sys.stdout.write(f"{row.get('path', '')}\n")
        return 0

    # --- Header ----------------------------------------------------- #
    job_kind = plan.get("job_kind") or "?"
    method = plan.get("method") or "?"
    basis = plan.get("basis") or "?"
    functional = plan.get("functional") or ""
    status = (outputs or {}).get("status") or ""
    finished = (outputs or {}).get("finished_at_iso") or ""
    wall_s = (body.get("run") or {}).get("wall_seconds")
    label_parts = [method]
    if functional:
        label_parts.append(f"/ {functional}")
    label_parts.append(f"basis={basis}")
    label_parts.append(f"({job_kind})")
    sys.stdout.write(
        f"manifest: {manifest_path}\n"
        f"  job:    {' '.join(label_parts)}\n"
        f"  status: {status or '(no status)'}\n"
    )
    if finished:
        sys.stdout.write(f"  finished_at: {finished}\n")
    if wall_s is not None:
        sys.stdout.write(f"  wall_seconds: {float(wall_s):.3f}\n")
    sys.stdout.write("\n")

    # --- Files table ----------------------------------------------- #
    sys.stdout.write(
        f"  {'status':<9}{'role':<12}{'format':<14}"
        f"{'size':<10}{'checksum':<15}path\n"
    )
    sys.stdout.write("  " + "-" * 79 + "\n")

    missing_count = 0
    shown = 0
    for row in declared:
        path = str(row.get("path", ""))
        role = str(row.get("role", ""))
        fmt = str(row.get("format", ""))
        always = bool(row.get("always", True))
        out_row = by_path.get(path) or {}
        written = bool(out_row.get("written", False))
        on_disk = _exists_on_disk(workspace, path)
        bytes_ = out_row.get("bytes")
        sha = str(out_row.get("sha256") or "")
        checksum_status = str(out_row.get("checksum_status") or "")
        if _is_manifest_path(manifest_path, workspace, path):
            # A self-digest can never describe the final bytes containing it.
            # Also override stale hashes from pre-contract manifests so this
            # inspector never presents one as trustworthy.
            checksum = "self-excluded"
        elif sha:
            checksum = sha[:8]
        else:
            checksum = checksum_status

        if written and on_disk:
            tag = "OK"
        elif always:
            tag = "MISS"
            missing_count += 1
        else:
            tag = "skip"

        if args.missing_only and tag not in ("MISS",):
            continue

        sys.stdout.write(
            f"  {tag:<9}{role:<12}{fmt:<14}"
            f"{_format_size(bytes_):<10}{checksum:<15}{path}\n"
        )
        shown += 1

    if args.missing_only and shown == 0:
        sys.stdout.write("  (no missing files -- every always-on "
                         "declared artefact is on disk)\n")

    if args.strict and missing_count > 0:
        sys.stderr.write(
            f"\nvibeqc-outputs: --strict -- {missing_count} declared "
            "always-on file(s) missing\n"
        )
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
