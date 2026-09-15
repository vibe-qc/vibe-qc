"""Orchestrate one ORCA cell on compute-host-d through the ``vq`` queue.

vibe-qc never imports ORCA (CLAUDE.md § 10); ORCA runs out-of-process.
This module is the ORCA analogue of ``runner_pyscf.py``'s
subprocess-runner pattern — except the "subprocess" is a remote ``vq``
job: serialize the request (an ORCA input deck), run it out-of-process
on compute-host-d, parse one machine-readable artefact (the ORCA ``.out``)
back into a decomposition dict.

The job workspace bundles ``run-orca.sh`` alongside the input deck, so
a fresh ORCA cell runs without ``run-orca.sh`` needing to be on the
compute-host-d clone first — the wrapper ships *with* the job.

Nothing here imports vibe-qc; it only shells out to the ``vq`` CLI.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .cases import Cell
from .orca_input import build_orca_input
from .parse_orca import parse_orca_mp2_output, parse_orca_output

# vq job states that mean "this job is done, one way or another"
# (vibe-queue/src/vq/spec.py::TERMINAL_STATES). COMPLETED is the only
# success.
_TERMINAL_STATES = frozenset({
    "completed", "failed", "killed", "interrupted",
    "oom_killed", "starved", "time_exceeded", "aborted_by_queue",
})


class OrcaVqError(RuntimeError):
    """An ORCA-via-vq job failed to submit, run, or come back parseable."""


def _inject_host(args: tuple[str, ...]) -> list[str]:
    """Insert the explicit host (from ``VIBEQC_BENCHMARK_HOST``) into
    a ``vq`` argv when set, otherwise return args unchanged.

    Subcommand placement:
      * ``vq submit ...`` → ``vq submit --host HOST ...`` (flag form,
        because ``vq submit`` parses an early positional as the
        workspace dir under ``-d``).
      * ``vq status / fetch / programs / queue / kill / wait``
        → ``vq SUB HOST ...`` (positional first arg).
      * Anything else → unchanged (admin verbs, etc).

    Used when the calling chat's local box can't reach the configured
    ``default_host`` (compute-host-d down, network blip) and wants to retarget
    a benchmark run to a sibling host like compute-host-a without editing the
    global ``~/.config/vq/config.toml`` ``default_host`` entry.
    """
    host = os.environ.get("VIBEQC_BENCHMARK_HOST", "").strip()
    if not host or not args:
        return list(args)
    sub = args[0]
    if sub == "submit":
        return [sub, "--host", host, *args[1:]]
    if sub in ("status", "fetch", "programs", "queue", "kill", "wait", "tail"):
        return [sub, host, *args[1:]]
    return list(args)


def _vq(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run the ``vq`` CLI. Raises OrcaVqError on a non-zero exit if check."""
    final_args = _inject_host(args)
    proc = subprocess.run(
        ["vq", *final_args], capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        raise OrcaVqError(
            f"`vq {' '.join(final_args)}` failed (exit {proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    return proc


def _vq_read(*args: str, retries: int = 3) -> subprocess.CompletedProcess:
    """Run a *read-only* ``vq`` command, retrying transient SSH failures.

    ``vq status`` / ``vq fetch`` / ``vq programs`` reach compute-host-d over
    SSH; a backed-up box drops the odd connection ("Connection reset by
    peer", "kex_exchange_identification") — a transient that a short
    backoff clears. Only safe for *idempotent* commands — ``vq submit``
    must never auto-retry (it would double-submit), so it uses plain
    :func:`_vq`.
    """
    last: Optional[subprocess.CompletedProcess] = None
    for attempt in range(retries):
        proc = _vq(*args, check=False)
        if proc.returncode == 0:
            return proc
        last = proc
        if attempt < retries - 1:
            time.sleep(5 * (attempt + 1))  # 5s, 10s backoff
    raise OrcaVqError(
        f"`vq {' '.join(args)}` failed {retries}x (exit "
        f"{last.returncode if last else '?'}):\n"
        f"{last.stdout if last else ''}\n{last.stderr if last else ''}"
    )


_orca_bin_cache: Optional[str] = None


def orca_binary_path() -> str:
    """The absolute ORCA binary path on the queue host, from ``vq programs``.

    The vq daemon dispatches jobs with a bare PATH — it does *not* put
    registered-program dirs on PATH (despite what an earlier handover
    rev assumed; the failing-job stderr showed
    ``PATH=/usr/local/sbin:/usr/local/bin:/usr/bin``). So we read the
    path out of ``vq programs --json`` and hand it to ``run-orca.sh``
    via an explicit ``ORCA_BIN`` env var — the same pattern
    ``tests/integration_smoke.py`` uses for the CRYSTAL binaries.
    """
    global _orca_bin_cache
    if _orca_bin_cache is not None:
        return _orca_bin_cache
    proc = _vq_read("programs", "--json")
    try:
        programs = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise OrcaVqError(
            f"could not parse `vq programs --json`: {exc}\n{proc.stdout}"
        ) from exc
    for prog in programs:
        if prog.get("name") == "orca":
            if prog.get("status") != "OK":
                raise OrcaVqError(
                    f"ORCA is registered but not OK on the queue host: "
                    f"{prog.get('status')} — {prog.get('reason')}"
                )
            binary = prog.get("binary")
            if not binary:
                raise OrcaVqError(
                    "`vq programs` lists orca but with no binary path")
            _orca_bin_cache = binary
            return binary
    raise OrcaVqError(
        "no `orca` program registered on the queue host — "
        "`vq programs` to check"
    )


def submit_orca_cell(
    cell: Cell,
    *,
    workspace_root: Path,
    nprocs: int = 1,
    cpus: int = 1,
    wall_time_s: int = 1800,
    priority: int = 0,
) -> Tuple[str, Path]:
    """Stage `cell`'s workspace and ``vq submit`` it. Returns (jobid, workspace).

    The workspace holds the ORCA input deck and a copy of
    ``run-orca.sh``; the job command is ``bash run-orca.sh <deck>``,
    run with cwd = the workspace on compute-host-d.

    `cpus` is what the job claims against the daemon's budget; keep it
    equal to `nprocs` so a parallel ORCA run has the cores it asks for.
    `wall_time_s` is mandatory (the v0.5.9 watchdog mis-kills
    bash-wrapped jobs as STARVED without it).
    """
    queue_checkout = Path(os.environ.get(
        "VIBEQC_QUEUE_CHECKOUT", str(Path.home() / "gitlab" / "vibe-queue"),
    )).expanduser()
    run_orca_sh = queue_checkout / "contrib" / "run-orca.sh"
    if not run_orca_sh.is_file():
        raise OrcaVqError(
            f"run-orca.sh not found at {run_orca_sh}. Set VIBEQC_QUEUE_CHECKOUT "
            "to the separate vibe-queue checkout on this submission machine."
        )

    workspace = workspace_root / cell.cell_id
    workspace.mkdir(parents=True, exist_ok=True)
    deck_name = f"{cell.cell_id}.inp"
    (workspace / deck_name).write_text(
        build_orca_input(cell, nprocs=nprocs), encoding="utf-8")
    shutil.copy2(run_orca_sh, workspace / "run-orca.sh")

    orca_bin = orca_binary_path()
    args = [
        "submit", "-d", str(workspace),
        "--cpus", str(cpus),
        "--wall-time-seconds", str(wall_time_s),
    ]
    if priority:
        args += ["--priority", str(priority)]
    # `env ORCA_BIN=...` hands the binary path to run-orca.sh: the
    # daemon's job PATH doesn't include the ORCA install dir, so the
    # wrapper can't resolve `orca` on its own.
    args += ["--", "env", f"ORCA_BIN={orca_bin}",
             "bash", "run-orca.sh", deck_name]

    proc = _vq(*args)
    jobid = proc.stdout.strip().splitlines()[-1].strip()
    if not jobid:
        raise OrcaVqError(f"`vq submit` produced no jobid:\n{proc.stdout}")
    return jobid, workspace


def _parse_status(status_text: str) -> Dict[str, Any]:
    """Pull state / timing / exit_code out of `vq status` text."""
    info: Dict[str, Any] = {}
    for line in status_text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if key == "state":
            # `vq status` may suffix an archive marker, e.g.
            # "failed (archived)" — keep only the bare state token.
            info["state"] = val.split()[0] if val else ""
        elif key in ("submitted", "started", "finished", "exit_code"):
            info[key] = val
    return info


def job_status(jobid: str) -> Dict[str, Any]:
    """Return the parsed ``vq status`` of `jobid` (state, timing, exit_code)."""
    proc = _vq_read("status", jobid, "-n", "0")
    info = _parse_status(proc.stdout)
    info["_raw"] = proc.stdout
    if "state" not in info:
        raise OrcaVqError(
            f"could not read job state for {jobid} from `vq status`:\n"
            f"{proc.stdout}"
        )
    return info


def wait_for_terminal(
    jobid: str,
    *,
    poll_s: int = 20,
    timeout_s: int = 7200,
    on_poll: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Poll `jobid` until it reaches a vq terminal state (or `timeout_s`).

    `on_poll(state, elapsed_s)` is called once per poll for progress
    reporting. Raises OrcaVqError on timeout — the job is left running
    on compute-host-d (the caller can `vq status` / `vq kill` it).
    """
    t0 = time.monotonic()
    while True:
        info = job_status(jobid)
        state = info["state"]
        elapsed = time.monotonic() - t0
        if on_poll is not None:
            on_poll(state, elapsed)
        if state in _TERMINAL_STATES:
            return info
        if elapsed > timeout_s:
            raise OrcaVqError(
                f"job {jobid} still {state!r} after {elapsed:.0f}s "
                f"(timeout {timeout_s}s) — left running on compute-host-d; "
                f"`vq status {jobid}` to check, `vq kill {jobid}` to stop"
            )
        time.sleep(poll_s)


def fetch_workspace(jobid: str, dest_root: Path) -> Path:
    """``vq fetch`` `jobid`'s workspace; return the local ``<dest>/<jobid>/``."""
    dest_root.mkdir(parents=True, exist_ok=True)
    _vq_read("fetch", jobid, "-o", str(dest_root))
    workspace = dest_root / jobid
    if not workspace.is_dir():
        raise OrcaVqError(
            f"`vq fetch {jobid}` did not produce {workspace}")
    return workspace


def _wall_seconds(info: Dict[str, Any]) -> Optional[float]:
    """finished - started, in seconds — the on-box execution wall time.

    This is queue-wait-free (it brackets the actual ORCA process), so
    it is the honest number for the speed benchmark.
    """
    started, finished = info.get("started"), info.get("finished")
    if not started or not finished:
        return None
    from datetime import datetime

    try:
        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(finished)
    except ValueError:
        return None
    return (t1 - t0).total_seconds()


def run_orca_cell(
    cell: Cell,
    *,
    workspace_root: Path,
    fetch_root: Path,
    nprocs: int = 1,
    cpus: int = 1,
    wall_time_s: int = 1800,
    priority: int = 0,
    poll_s: int = 20,
    timeout_s: int = 7200,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Full submit -> wait -> fetch -> parse loop for one ORCA cell.

    Returns the decomposition dict — :func:`parse_orca_output` for an
    SCF cell, :func:`parse_orca_mp2_output` for an MP2 cell
    (``cell.is_mp2``) — extended with provenance: ``vq_jobid``,
    ``vq_state``, ``vq_wall_s`` (on-box execution wall time),
    ``nprocs``, ``cell_id``.

    Raises :class:`OrcaVqError` if the job didn't reach COMPLETED, or
    :class:`~.parse_orca.OrcaParseError` if the ORCA output won't parse
    / fails its self-check.
    """
    def _say(msg: str) -> None:
        if verbose:
            print(f"  [orca-vq {cell.cell_id}] {msg}", flush=True)

    jobid, _workspace = submit_orca_cell(
        cell, workspace_root=workspace_root, nprocs=nprocs, cpus=cpus,
        wall_time_s=wall_time_s, priority=priority,
    )
    _say(f"submitted -> jobid {jobid} (cpus={cpus}, nprocs={nprocs}, "
         f"wall_limit={wall_time_s}s)")

    last_state = ""

    def _on_poll(state: str, elapsed: float) -> None:
        nonlocal last_state
        if state != last_state:
            _say(f"state={state} (t+{elapsed:.0f}s)")
            last_state = state

    info = wait_for_terminal(
        jobid, poll_s=poll_s, timeout_s=timeout_s, on_poll=_on_poll)
    state = info["state"]
    if state != "completed":
        raise OrcaVqError(
            f"ORCA cell {cell.cell_id} job {jobid} ended in state "
            f"{state!r} (exit_code={info.get('exit_code')}). Inspect: "
            f"`vq status {jobid}` / `vq fetch {jobid}`."
        )

    workspace = fetch_workspace(jobid, fetch_root)
    out_path = workspace / f"{cell.cell_id}.out"
    if not out_path.is_file():
        candidates = sorted(p.name for p in workspace.glob("*.out"))
        raise OrcaVqError(
            f"ORCA output {out_path.name} not in fetched workspace "
            f"{workspace} (found .out files: {candidates})"
        )
    _say(f"completed, fetched -> {out_path}")

    # MP2 cells parse a post-SCF MP2 block (e_hf / e_corr / e_total +
    # channels); SCF cells parse the per-intermediate decomposition.
    parsed = (parse_orca_mp2_output(out_path) if cell.is_mp2
              else parse_orca_output(out_path))
    wall = _wall_seconds(info)
    parsed.update({
        "cell_id": cell.cell_id,
        "vq_jobid": jobid,
        "vq_state": state,
        "vq_wall_s": wall,
        "nprocs": nprocs,
    })
    _say(f"parsed OK (E_total={parsed['e_total']:.8f} Eh, "
         f"self-check residual={parsed['e_total_residual']:.2e}, "
         f"wall={wall:.1f}s)" if wall else f"parsed OK")
    return parsed
