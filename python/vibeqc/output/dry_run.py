"""Dry-run support -- pre-flight manifest without running anything.

``vibeqc.output.dry_run_manifest(...)`` writes ``{stem}.system`` with
the ``[plan]`` section populated and ``[outputs].status = "dry_run"``,
then returns. No SCF runs, no molecule is required, no compute is
done. Companion paths in drivers such as :func:`vibeqc.runner.run_job`
and :func:`vibeqc.neb.run_neb` (via the ``VIBEQC_DRY_RUN=1`` env var
or the ``dry_run=True`` kwarg) short-circuit before any work and call
this helper.

Use cases:

* The ``vq submit`` pre-flight asks each Python job script what files
  it will produce. With ``VIBEQC_DRY_RUN=1`` set in the environment,
  the script's vibe-qc driver call writes a dry-run manifest and exits;
  vq parses the ``[plan]`` section and records the expected output set
  in the job spec.
* CI dashboards verifying that a renamed kwarg / method / basis still
  produces the documented file set without paying for an SCF.
* Tutorials wanting to introduce the artefact list before showing the
  full pipeline.

The manifest produced by this helper carries every section the
regular manifest does (``[vibeqc]``, ``[host]``, ``[cpu]``,
``[memory]``, ``[python]``, ``[libraries]``, ``[validation]``,
``[run]``, ``[plan]``, ``[outputs]``); the difference is purely
``outputs.status = "dry_run"`` and the absence of any actual
``[[outputs.files]]`` rows beyond the placeholder (``written=false``)
entries seeded from the plan.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, TextIO

from .manifest import ManifestUpdater
from .plan import OutputPlan


__all__ = [
    "dry_run_manifest",
    "is_dry_run_requested",
    "is_dry_run_estimate_requested",
    "print_dry_run_summary",
]


def is_dry_run_requested() -> bool:
    """``True`` if ``VIBEQC_DRY_RUN`` is set in the environment to a
    truthy value (anything other than empty / ``"0"`` / ``"false"`` /
    ``"no"``). Case-insensitive. Used by :func:`vibeqc.runner.run_job`
    to detect the env-var trigger when the caller did not pass an
    explicit ``dry_run=`` kwarg."""
    val = os.environ.get("VIBEQC_DRY_RUN", "").strip().lower()
    return val not in ("", "0", "false", "no")


def is_dry_run_estimate_requested() -> bool:
    """``True`` if ``VIBEQC_DRY_RUN_ESTIMATE`` is set in the environment
    to a truthy value (anything other than empty / ``"0"`` / ``"false"``
    / ``"no"``). Case-insensitive, mirroring :func:`is_dry_run_requested`.

    Gates the opt-in peak-memory estimate pass in the dry-run
    short-circuit of :func:`vibeqc.runner.run_job`. When set *alongside*
    ``VIBEQC_DRY_RUN``, the dry-run additionally builds the basis, calls
    ``estimate_memory(...)``, and records ``.total_bytes`` (the configured
    headroom is already applied) as ``[memory].estimate_bytes`` in the
    ``.system`` manifest. ``vq submit auto`` reads that figure for
    memory-aware RAM-fit host placement. Off by default so the cheap
    output-discovery dry-run stays cheap."""
    val = os.environ.get("VIBEQC_DRY_RUN_ESTIMATE", "").strip().lower()
    return val not in ("", "0", "false", "no")


def dry_run_manifest(
    plan: OutputPlan,
    *,
    record_hostname: bool = True,
    print_summary: bool = True,
    stream: TextIO | None = None,
    estimate_bytes: int | None = None,
) -> Path:
    """Write a one-shot dry-run manifest and return its path.

    Parameters
    ----------
    plan
        The :class:`OutputPlan` describing what *would* have been
        produced. Build it via
        :meth:`OutputPlan.from_run_job_kwargs` with the same kwargs
        the real job would receive.
    record_hostname
        Forwarded to :class:`ManifestUpdater`. ``False`` (or
        ``VIBEQC_NO_HOSTNAME=1``) emits ``hostname = "<redacted>"``.
    print_summary
        If ``True`` (default), also call :func:`print_dry_run_summary`
        on the given stream -- useful for the
        ``VIBEQC_DRY_RUN=1 python input.py`` workflow where the
        operator wants the artefact list on stdout.
    stream
        Where to print the summary; defaults to ``sys.stdout``.
    estimate_bytes
        Optional peak-memory estimate in bytes (``estimate_memory(...)
        .total_bytes``, with the configured headroom already applied). When
        not ``None``, recorded as ``[memory].estimate_bytes`` in the
        manifest for memory-aware ``vq submit auto`` placement. Computed
        by the caller only under the ``VIBEQC_DRY_RUN_ESTIMATE`` opt-in
        (see :func:`is_dry_run_estimate_requested`); ``None`` leaves the
        ``[memory]`` section untouched so the default dry-run manifest is
        unchanged.

    Returns
    -------
    pathlib.Path
        The on-disk path of the written ``.system`` manifest.
    """
    updater = ManifestUpdater(
        plan,
        record_hostname=record_hostname,
        wall_seconds=0.0,
        estimate_bytes=estimate_bytes,
    )
    updater.mark_dry_run()
    if print_summary:
        print_dry_run_summary(plan, stream=stream)
    return updater.path


def print_dry_run_summary(
    plan: OutputPlan,
    *,
    stream: TextIO | None = None,
) -> None:
    """Print a human-readable summary of the declared artefacts to
    ``stream`` (default stdout). One line per file, plus a header
    line identifying the job and a footer indicating dry-run mode."""
    out = stream if stream is not None else sys.stdout

    def _w(s: str) -> None:
        out.write(s + "\n")

    fn = plan.functional or "(none)"
    _w(f"vibe-qc dry-run: {plan.method} / basis={plan.basis} "
       f"/ functional={fn}")
    _w(f"  output stem    = {plan.stem}")
    _w(f"  options digest = {plan.options_digest}")
    _w(f"  declared files ({len(plan.files)}):")
    for f in plan.files:
        if not f.always:
            tag = "conditional"
        else:
            tag = "always-on"
        _w(f"    {tag:<11s}  {f.role:<11s}  {f.path}")
    _w("  manifest written; no SCF will run.")
    out.flush()
