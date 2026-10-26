"""vq transport — submit external-SCF jobs through the vibe-queue CLI.

vibe-queue (``vq``) is the cross-machine job queue co-located with
vibe-qc. A laptop ``vq submit`` ships a job to a remote daemon
(compute-host for this lab); the daemon runs it and ``vq fetch`` brings
the outputs back.

This transport wraps three ``vq`` subcommands:

* ``vq submit -d <work_dir> --cpus N --wall-time-seconds T -- <cmd>``
    Submit. Returns a job id on stdout.
* ``vq status <job_id>``
    Poll. Prints a ``state:`` line we scrape.
* ``vq fetch <job_id> -o <dest>``
    Download. Lands outputs under ``<dest>/<job_id>/``.

The ``--wall-time-seconds`` flag is always passed: the vq v0.5.9
daemon watchdog mis-kills bash-wrapped jobs as ``STARVED`` without
it (documented in the chat memory + the CRYSTAL14 README recipe).

vibe-basis treats ``vq`` as an external CLI — we shell out to it
and parse its text output; we never ``import vq``. The actual
subprocess call is injected (``runner=``) so tests run without
``vq`` installed and without compute-host access.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

from .base import (
    JobHandle,
    JobSubmitError,
    Transport,
    TransportError,
)

# A runner is anything call-compatible with the subset of
# subprocess.run we use: runner(argv, capture_output=True, text=True)
# returning an object with .stdout / .stderr / .returncode.
Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _default_runner(argv: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(  # noqa: S603 — argv is built from known tokens
        list(argv),
        capture_output=True,
        text=True,
        check=False,
    )


class VqTransport(Transport):
    """Submit / poll / fetch external-SCF jobs through the ``vq`` CLI.

    Parameters
    ----------
    vq_bin
        Path or name of the ``vq`` executable. Default ``"vq"``
        (assumes it is on ``$PATH`` after installing vq through its own
        lifecycle, or through ``vibe-basis/scripts/install.sh --with-vq``).
    host
        Target host for ``vq submit``.  ``None`` (default) uses
        ``default_host`` from ``~/.config/vq/config.toml``.  Set to
        ``"compute-host"`` or ``"second-compute-host"`` to target a specific machine.
    default_cpus, default_wall_time_s
        Used when ``submit`` is called without explicit values. The
        defaults (14 CPUs / 7200 s) match the CRYSTAL14-on-compute-host
        recipe — compute-host is a 16-core box, leaving 2 cores free is
        the courtesy default.
    runner
        Injected subprocess callable, for testing. Defaults to a
        thin wrapper over ``subprocess.run``.
    """

    def __init__(
        self,
        *,
        vq_bin: str = "vq",
        host: Optional[str] = None,
        default_cpus: int = 14,
        default_wall_time_s: int = 7200,
        runner: Optional[Runner] = None,
    ) -> None:
        self.vq_bin = vq_bin
        self.host = host  # None = use default_host from vq config
        self.default_cpus = default_cpus
        self.default_wall_time_s = default_wall_time_s
        self._runner: Runner = runner or _default_runner

    # ------------------------------------------------------------------
    # Internal: run a vq subcommand, raise on non-zero exit.
    # ------------------------------------------------------------------
    def _run(self, argv: Sequence[str]) -> "subprocess.CompletedProcess[str]":
        proc = self._runner(argv)
        if proc.returncode != 0:
            raise TransportError(
                f"`{' '.join(argv)}` exited {proc.returncode}\n"
                f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
            )
        return proc

    @staticmethod
    def _parse_job_id(stdout: str) -> str:
        """Extract the job id from ``vq submit`` stdout.

        ``vq submit`` is used in the README as ``JOBID=$(vq submit
        ...)``, i.e. the id is the captured stdout. We take the last
        non-empty line and its last whitespace token — that handles
        both a bare ``abc123`` and a ``Submitted job abc123`` style
        line. Verified against vq's actual format on the first
        compute-host run; adjust here if vq changes its submit output.
        """
        lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
        if not lines:
            raise JobSubmitError(
                f"vq submit produced no parseable job id (stdout={stdout!r})"
            )
        token = lines[-1].split()[-1]
        if not token:
            raise JobSubmitError(f"vq submit job id is empty (stdout={stdout!r})")
        return token

    @staticmethod
    def _parse_state(stdout: str) -> str:
        """Extract the state token from ``vq status`` stdout.

        ``vq status`` prints a line ``state: <token> [extra...]``.
        We return ``<token>`` lowercased. If no ``state:`` line is
        found, return ``"unknown"`` — the caller's wait-loop treats
        anything not in TERMINAL_STATES as still-in-flight, so an
        unparseable status just means "keep polling" rather than a
        crash.
        """
        for line in stdout.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("state:"):
                rest = stripped.split(":", 1)[1].strip()
                if rest:
                    return rest.split()[0].lower()
        return "unknown"

    # ------------------------------------------------------------------
    # Transport interface
    # ------------------------------------------------------------------
    def submit(
        self,
        work_dir: Path | str,
        command: Sequence[str],
        *,
        cpus: Optional[int] = None,
        wall_time_s: Optional[int] = None,
        label: Optional[str] = None,
    ) -> JobHandle:
        work_dir = Path(work_dir)
        argv = [self.vq_bin, "submit"]
        # Insert host as first positional if specified.
        if self.host is not None:
            argv.append(self.host)
        argv += [
            "-d",
            str(work_dir),
            "--cpus",
            str(cpus if cpus is not None else self.default_cpus),
            "--wall-time-seconds",
            str(wall_time_s if wall_time_s is not None else self.default_wall_time_s),
            "--",
            *command,
        ]
        proc = self._run(argv)
        job_id = self._parse_job_id(proc.stdout)
        return JobHandle(
            job_id=job_id,
            work_dir=work_dir,
            command=tuple(command),
            label=label,
        )

    def poll(self, handle: JobHandle) -> str:
        proc = self._run([self.vq_bin, "status", handle.job_id])
        return self._parse_state(proc.stdout)

    def fetch(self, handle: JobHandle, dest: Path | str) -> Path:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self._run([self.vq_bin, "fetch", handle.job_id, "-o", str(dest)])
        # vq lands outputs under <dest>/<job_id>/.
        return dest / handle.job_id
