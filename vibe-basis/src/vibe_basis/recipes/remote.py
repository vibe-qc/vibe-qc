"""Remote optimizer — submit an entire recipe as a single vq job.

This module provides the "submit optimizer to compute-host" pattern
from GOAL4_DESIGN.md §"Where work runs":

    Laptop                              compute-host (vq daemon)
    ──────                              ─────────────────────
    submit_recipe_to_vq()
      │
      ├─ emit_recipe_script ────→       vq job: optimizer.py
      ├─ vq submit               →       │
      │                                  ├─ LocalTransport
      │                                  │  └─ subprocess: CRYSTAL14 MgO.d12
      │                                  │  └─ subprocess: CRYSTAL14 CaO.d12
      │                                  │  └─ ...
      │                                  ├─ write results.json
      │                                  └─ exit 0
      │
      ├─ vq status (poll loop)
      ├─ vq fetch               ←──     results.json
      └─ load ParityReport

The recipe runs on compute-host with :class:`LocalTransport` — CRYSTAL14
is a local subprocess on compute-host, no queue latency per evaluation.
The laptop only submits + polls + fetches the final result.

Usage
-----

    from vibe_basis.recipes.remote import submit_recipe_to_vq

    handle = submit_recipe_to_vq(
        recipe="pob_parity",
        structures=["PT2013-T4"],        # table id or list of names
        basis="pob-tzvp",
        method="rhf",
        crystal_wrapper="~/crystal/run-crystal.sh",
        vq_transport=VqTransport(),
        wall_time_s=172800,              # 48 hours
    )
    # Poll until done:
    state = vq_transport.wait(handle)
    # Fetch results:
    report_dir = vq_transport.fetch(handle, "./results")
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from vibe_basis.io.references import get_ref
from vibe_basis.io.structures import STRUCTURES, in_table
from vibe_basis.transports.base import (
    JobHandle,
    JobResult,
    Transport,
    TransportError,
)
from vibe_basis.transports.vq import VqTransport

# ---------------------------------------------------------------------------
# Recipe-script emitter
# ---------------------------------------------------------------------------

_RECIPE_SCRIPT_TEMPLATE = '''\
"""Auto-generated recipe runner — submitted to vq by vibe-basis."""
import json
import sys
from pathlib import Path

# Ensure vibe_basis is importable (compute-host has it in the venv).
# If running from a dedicated clone without editable install:
REPO = Path("{repo_root}").expanduser()
if str(REPO / "vibe-basis" / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "vibe-basis" / "src"))

from vibe_basis.io.structures import STRUCTURES, in_table
from vibe_basis.recipes.pob_parity import run_pob_parity
from vibe_basis.transports.local import LocalTransport

# ---- Recipe input ----------------------------------------------------------

RECIPE = {recipe_name!r}
STRUCTURE_SOURCE = {structure_source!r}
STRUCTURE_NAMES = {structure_names!r}
BASIS = {basis!r}
METHOD = {method!r}
WORKDIR_ROOT = {workdir_root!r}
CPUS = {cpus}
WALL_TIME_S = {wall_time_s}
CRYSTAL_WRAPPER = {crystal_wrapper!r}
SHRINK = {shrink}
TOLDEE = {toldee}

# ---- Resolve structures ----------------------------------------------------

if STRUCTURE_SOURCE == "table":
    structures = in_table(STRUCTURE_NAMES)
elif STRUCTURE_SOURCE == "names":
    structures = [STRUCTURES[n] for n in STRUCTURE_NAMES if n in STRUCTURES]
else:
    structures = list(STRUCTURES.values())

if not structures:
    print("ERROR: no structures resolved", file=sys.stderr)
    sys.exit(1)

# ---- Run recipe ------------------------------------------------------------

transport = LocalTransport()
report = run_pob_parity(
    structures=structures,
    transport=transport,
    basis=BASIS,
    method=METHOD,
    workdir_root=Path(WORKDIR_ROOT),
    cpus=CPUS,
    wall_time_s=WALL_TIME_S,
    crystal_wrapper=CRYSTAL_WRAPPER,
    shrink=SHRINK,
    toldee=TOLDEE,
)

# ---- Serialize result ------------------------------------------------------

result_path = Path("{result_path}")
result_path.parent.mkdir(parents=True, exist_ok=True)
payload = {{
    "recipe": RECIPE,
    "basis": report.basis,
    "method": report.method,
    "compounds_total": report.compounds_total,
    "compounds_emitted": report.compounds_emitted,
    "compounds_converged": report.compounds_converged,
    "compounds_compared": report.compounds_compared,
    "sum_abs_delta_mha": report.sum_abs_delta_mha,
    "passes_acceptance": report.passes_acceptance,
    "results": [
        {{
            "compound": r.compound,
            "formula": r.formula,
            "ok": r.ok,
            "energy": r.energy,
            "method": r.method,
            "last_cycle": r.last_cycle,
            "ref_energy": r.ref_energy,
            "delta_mha": r.delta_mha,
            "failure_mode": r.failure_mode,
            "notes": r.notes,
        }}
        for r in report.results
    ],
}}
result_path.write_text(json.dumps(payload, indent=2))

print(report.summary())
if not report.passes_acceptance:
    print("\\nGATE FAILED — Σ|ΔE| >= 0.1 mHa", file=sys.stderr)
    sys.exit(2)
sys.exit(0)
'''


def emit_recipe_script(
    *,
    recipe: str = "pob_parity",
    structures: Optional[Sequence[str]] = None,
    structure_source: str = "table",
    basis: str = "pob-tzvp",
    method: str = "rhf",
    workdir_root: str = "recipe_runs",
    cpus: int = 14,
    wall_time_s: int = 1800,
    crystal_wrapper: str = "crystal",
    shrink: int = 8,
    toldee: int = 8,
    repo_root: str = "~/gitlab/vibeqc-basis",
) -> str:
    """Emit a self-contained Python script that runs a recipe on compute-host.

    The script uses :class:`LocalTransport` (CRYSTAL14 as local
    subprocess) and writes results to a JSON file.  It is designed
    to be submitted as a vq job.

    Parameters
    ----------
    recipe
        Recipe name — currently only ``"pob_parity"`` is supported.
    structures
        If *structure_source* is ``"table"``, a list of paper-table
        ids (e.g. ``["PT2013-T4"]``).  If ``"names"``, a list of
        compound names (e.g. ``["MgO", "CaO"]``).
    structure_source
        ``"table"`` or ``"names"``.
    basis, method, shrink, toldee
        Passed to :func:`run_pob_parity`.
    workdir_root
        Directory inside the vq job's workdir where per-compound
        subdirs are created.
    cpus, wall_time_s
        Passed to :func:`run_pob_parity`.
    crystal_wrapper
        Path to the ``run-crystal.sh`` script on compute-host.
    repo_root
        Path to the vibe-qc monorepo checkout on compute-host.

    Returns
    -------
    str
        The Python script text.  Write it into the work directory
        before submitting.
    """
    structure_names: list[str] = list(structures) if structures else []
    return _RECIPE_SCRIPT_TEMPLATE.format(
        recipe_name=recipe,
        structure_source=structure_source,
        structure_names=structure_names,
        basis=basis,
        method=method,
        workdir_root=workdir_root,
        cpus=cpus,
        wall_time_s=wall_time_s,
        crystal_wrapper=crystal_wrapper,
        shrink=shrink,
        toldee=toldee,
        repo_root=repo_root,
        result_path="recipe_result.json",
    )


# ---------------------------------------------------------------------------
# Submission helpers
# ---------------------------------------------------------------------------


def submit_recipe_to_vq(
    *,
    recipe: str = "pob_parity",
    structures: Optional[Sequence[str]] = None,
    structure_source: str = "table",
    basis: str = "pob-tzvp",
    method: str = "rhf",
    workdir_root: str = "recipe_runs",
    cpus: int = 14,
    wall_time_s: int = 172800,  # 48 hours default for multi-hour optimizations
    crystal_wrapper: str = "crystal",
    shrink: int = 8,
    toldee: int = 8,
    repo_root: str = "~/gitlab/vibeqc-basis",
    host: Optional[str] = None,
    vq_transport: Optional[VqTransport] = None,
    label: Optional[str] = None,
    working_dir: Optional[Path] = None,
) -> JobHandle:
    """Package a recipe as a self-contained script and submit it via vq.

    The recipe runs on the remote host (remote compute hosts) with
    :class:`LocalTransport`, so CRYSTAL14 is a local subprocess.
    The laptop only submits, polls, and fetches.

    Parameters
    ----------
    recipe
        Recipe name — ``"pob_parity"`` (Stage 0) is the first
        supported recipe.
    structures
        Paper-table ids (if ``structure_source="table"``) or
        compound names (if ``structure_source="names"``).
    structure_source
        ``"table"`` or ``"names"``.
    basis, method, shrink, toldee
        Passed through to the recipe.
    workdir_root
        Directory inside the vq job's workdir for per-compound
        subdirs.
    cpus
        CPUs allocated to the vq job (and shared across CRYSTAL14
        subprocesses).
    wall_time_s
        Wall-time budget for the vq job in seconds.  Default 48 h
        for multi-hour optimizations.
    crystal_wrapper
        Path to ``run-crystal.sh`` on the remote host.
    repo_root
        Path to the vibe-qc monorepo checkout on the remote host.
    host
        Target host: ``"compute-host"``, ``"second-compute-host"``, or ``None`` for
        the vq config's ``default_host``.
    vq_transport
        A :class:`VqTransport` instance.  Created with defaults
        if not provided.
    label
        Label for the vq job (appears in ``vq status``).
    working_dir
        Directory to stage the job files in.  Created if not
        provided (a temp dir under the current directory).

    Returns
    -------
    JobHandle
        The vq job handle.  Use ``vq_transport.wait(handle)`` and
        ``vq_transport.fetch(handle, dest)`` to retrieve results.
    """
    transport = vq_transport or VqTransport(host=host)
    wd = working_dir or Path(f"vb-{recipe}-vq-job")
    wd.mkdir(parents=True, exist_ok=True)

    # Emit the self-contained recipe script.
    script = emit_recipe_script(
        recipe=recipe,
        structures=structures,
        structure_source=structure_source,
        basis=basis,
        method=method,
        workdir_root=workdir_root,
        cpus=cpus,
        wall_time_s=wall_time_s,  # note: this is LOCAL wall_time per CRYSTAL14 run
        crystal_wrapper=crystal_wrapper,
        shrink=shrink,
        toldee=toldee,
        repo_root=repo_root,
    )
    (wd / "run_recipe.py").write_text(script)

    # Submit the recipe as a vq job.
    cmd = ["python3", "run_recipe.py"]
    return transport.submit(
        work_dir=wd,
        command=cmd,
        cpus=cpus,
        wall_time_s=wall_time_s,  # VQ wall_time: total budget for the job
        label=label or f"vb-{recipe}",
    )


# ---------------------------------------------------------------------------
# Result retrieval
# ---------------------------------------------------------------------------


@dataclass
class RemoteRecipeResult:
    """Result of a remotely-executed recipe, fetched from vq."""

    job_handle: JobHandle
    recipe: str
    report_dict: dict[str, Any]
    summary: str
    passes: bool

    @classmethod
    def from_fetched(
        cls,
        handle: JobHandle,
        output_dir: Path,
    ) -> "RemoteRecipeResult":
        """Load a recipe result from a fetched vq job output."""
        json_path = output_dir / "recipe_result.json"
        if not json_path.exists():
            raise TransportError(
                f"recipe result not found at {json_path} — "
                f"the job may have failed before writing results. "
                f"Check {output_dir / 'run_recipe.py.stdout'} and "
                f"{output_dir / 'run_recipe.py.stderr'}."
            )
        payload = json.loads(json_path.read_text())
        # Rebuild a summary string from the dict.
        lines = [f"Remote recipe: {payload['recipe']}"]
        lines.append(f"  basis: {payload['basis']}  method: {payload['method']}")
        lines.append(
            f"  {payload['compounds_emitted']} emitted / "
            f"{payload['compounds_converged']} converged / "
            f"{payload['compounds_compared']} compared"
        )
        for r in payload.get("results", []):
            if r["ok"] and r["delta_mha"] is not None:
                lines.append(
                    f"  {r['compound']:<8} {r['energy']:>14.6f}  "
                    f"Δ = {r['delta_mha']:+.3f} mHa"
                )
            elif r["ok"]:
                lines.append(
                    f"  {r['compound']:<8} {r['energy']:>14.6f}  (no reference)"
                )
            else:
                lines.append(f"  {r['compound']:<8} FAILED — {r['failure_mode']}")
        lines.append(f"  Σ|ΔE| = {payload['sum_abs_delta_mha']:.3f} mHa")
        lines.append(f"  gate = {'PASS' if payload['passes_acceptance'] else 'FAIL'}")
        return cls(
            job_handle=handle,
            recipe=payload["recipe"],
            report_dict=payload,
            summary="\n".join(lines),
            passes=payload["passes_acceptance"],
        )


def fetch_and_load_recipe_result(
    handle: JobHandle,
    vq_transport: Optional[VqTransport] = None,
    dest_dir: Optional[Path] = None,
) -> RemoteRecipeResult:
    """Fetch a completed recipe job and load its result.

    Convenience wrapper around ``vq_transport.fetch()`` +
    :meth:`RemoteRecipeResult.from_fetched`.

    Parameters
    ----------
    handle
        The JobHandle returned by :func:`submit_recipe_to_vq`.
    vq_transport
        The same VqTransport instance used for submission.
    dest_dir
        Directory to fetch outputs into.  Created if not provided.

    Returns
    -------
    RemoteRecipeResult
    """
    transport = vq_transport or VqTransport()
    dest = dest_dir or Path(f"vb-fetched-{handle.job_id}")
    output_dir = transport.fetch(handle, dest)
    return RemoteRecipeResult.from_fetched(handle, output_dir)


# ---------------------------------------------------------------------------
# Convenience: submit, wait, fetch in one call
# ---------------------------------------------------------------------------


def run_recipe_remote(
    *,
    recipe: str = "pob_parity",
    structures: Optional[Sequence[str]] = None,
    structure_source: str = "table",
    basis: str = "pob-tzvp",
    method: str = "rhf",
    cpus: int = 14,
    wall_time_s: int = 172800,
    crystal_wrapper: str = "crystal",
    shrink: int = 8,
    toldee: int = 8,
    repo_root: str = "~/gitlab/vibeqc-basis",
    vq_transport: Optional[VqTransport] = None,
    poll_interval_s: float = 60.0,
    timeout_s: float = 200_000.0,
) -> RemoteRecipeResult:
    """Submit a recipe to vq, wait for completion, fetch results.

    This is the all-in-one convenience entry point.  For a long
    production run you may prefer to call :func:`submit_recipe_to_vq`
    and :func:`fetch_and_load_recipe_result` separately so you
    can disconnect and reconnect.

    Parameters
    ----------
    (All parameters are the same as :func:`submit_recipe_to_vq`
    plus polling controls.)

    poll_interval_s
        How often to poll ``vq status`` (default 60 s).
    timeout_s
        Maximum total wait time (default ~55 hours).

    Returns
    -------
    RemoteRecipeResult
    """
    transport = vq_transport or VqTransport()

    handle = submit_recipe_to_vq(
        recipe=recipe,
        structures=structures,
        structure_source=structure_source,
        basis=basis,
        method=method,
        cpus=cpus,
        wall_time_s=wall_time_s,
        crystal_wrapper=crystal_wrapper,
        shrink=shrink,
        toldee=toldee,
        repo_root=repo_root,
        vq_transport=transport,
        working_dir=Path(f"vb-{recipe}-vq-job"),
    )

    state = transport.wait(
        handle,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )

    if state not in ("completed",):
        raise TransportError(
            f"recipe job {handle.job_id!r} ended in state {state!r} — "
            f"not completed. Fetch outputs anyway to inspect logs."
        )

    return fetch_and_load_recipe_result(
        handle,
        vq_transport=transport,
        dest_dir=Path(f"vb-fetched-{handle.job_id}"),
    )
