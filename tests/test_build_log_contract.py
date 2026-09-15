"""Contract tests for scripts/_build_log.sh — the quiet-build logging helper.

Born from the v0.15.131 release-gate post-mortem: three release-candidate
``build-test`` jobs failed undiagnosably because libint's codegen chatter
overflowed the GitLab runner's 4 MB trace cap ~35 minutes before the actual
error (which was then never collected — three times over). The helper routes
bulk build output to a log file, emits heartbeat lines, and on failure
replays the log tail with a resource snapshot, so the error always lands in
the visible trace.

These are pure shell-contract tests: no compiler, no network, no native
build. They drive the real helper through a bash child process.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = _REPO_ROOT / "scripts"
HELPER = SCRIPTS / "_build_log.sh"
NATIVE_STAMP = SCRIPTS / "_native_stamp.sh"


def _run(
    snippet: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run a snippet in bash with the helper sourced, set -euo pipefail on —
    the same shell regime build_libint.sh calls it under."""
    full_env = dict(os.environ)
    full_env.pop("VIBEQC_BUILD_VERBOSE", None)
    full_env.pop("VIBEQC_BUILD_LOG_HEARTBEAT_SECS", None)
    full_env.update(env or {})
    script = f"set -euo pipefail\n. {shlex.quote(str(HELPER))}\n{snippet}"
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=cwd,
    )


def test_success_path_redirects_bulk_output_and_restores_stdout(tmp_path: Path):
    log = tmp_path / "build.log"
    res = _run(
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "echo BULK-CHATTER\n"
        "vibeqc_build_log_end\n"
        "echo AFTER-END\n"
    )
    assert res.returncode == 0, res.stderr
    assert "BULK-CHATTER" not in res.stdout, res.stdout
    assert "BULK-CHATTER" in log.read_text(encoding="utf-8")
    assert "AFTER-END" in res.stdout, res.stdout
    assert re.search(r"demo build finished in \d+ min", res.stdout), res.stdout


def test_failure_replays_log_tail_and_preserves_exit_code(tmp_path: Path):
    log = tmp_path / "build.log"
    res = _run(
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "echo THE-ACTUAL-ERROR-LINE\n"
        "exit 7\n"
    )
    assert res.returncode == 7, (res.returncode, res.stdout, res.stderr)
    assert "demo build failed (exit 7)" in res.stdout, res.stdout
    assert "THE-ACTUAL-ERROR-LINE" in res.stdout, res.stdout
    assert "last 200 lines" in res.stdout, res.stdout
    # The post-mortem numbers the v0.15.131 incident lacked:
    assert "disk_avail=" in res.stdout, res.stdout


def test_failed_command_under_set_e_hits_the_trap(tmp_path: Path):
    log = tmp_path / "build.log"
    res = _run(
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "echo before-death\n"
        "false\n"
        "vibeqc_build_log_end\n"
    )
    assert res.returncode == 1, (res.returncode, res.stdout, res.stderr)
    assert "demo build failed (exit 1)" in res.stdout, res.stdout
    assert "before-death" in res.stdout, res.stdout


def test_verbose_opt_out_streams_and_writes_no_log(tmp_path: Path):
    log = tmp_path / "build.log"
    res = _run(
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "echo BULK-CHATTER\n"
        "vibeqc_build_log_end\n",
        env={"VIBEQC_BUILD_VERBOSE": "1"},
    )
    assert res.returncode == 0, res.stderr
    assert "BULK-CHATTER" in res.stdout, res.stdout
    assert not log.exists()


def test_heartbeat_reports_elapsed_and_last_ninja_step(tmp_path: Path):
    log = tmp_path / "build.log"
    res = _run(
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "echo '[12/100] Building CXX object foo.o'\n"
        "sleep 2.5\n"
        "vibeqc_build_log_end\n",
        env={"VIBEQC_BUILD_LOG_HEARTBEAT_SECS": "1"},
    )
    assert res.returncode == 0, res.stderr
    heartbeat = re.search(r"\[demo build \+\d+min\].*", res.stdout)
    assert heartbeat, res.stdout
    assert "[12/100]" in heartbeat.group(0), res.stdout
    assert "disk_avail=" in heartbeat.group(0), res.stdout


def test_heartbeat_orphans_pin_no_inherited_descriptors(tmp_path: Path):
    """An orphaned heartbeat sleep must not inherit the caller's long-lived
    descriptors. The concrete incident: build_libint.sh holds the build
    lock's FIFO write end at a high fd; an orphan inheriting it kept the
    lock-holder process — and with it the caller's stdout pipe — alive for
    a full heartbeat interval after the script had exited. Simulated here
    with a FIFO write end at fd 9 and a `cat` whose stdout is our pipe and
    which exits only when every write end of the FIFO is closed."""
    log = tmp_path / "build.log"
    fifo = tmp_path / "control"
    os.mkfifo(fifo)
    t0 = time.monotonic()
    res = _run(
        f"exec 9<> {shlex.quote(str(fifo))}\n"
        # 9>&-: the holder must not inherit the write end it is waiting on
        # (the real lock holder doesn't), or the test deadlocks on itself.
        f"cat {shlex.quote(str(fifo))} 9>&- &\n"
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "sleep 0.3\n"
        "vibeqc_build_log_end\n",
        env={"VIBEQC_BUILD_LOG_HEARTBEAT_SECS": "30"},
    )
    elapsed = time.monotonic() - t0
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert elapsed < 10, f"caller stayed pinned for {elapsed:.1f}s"


def test_quiet_build_plumbing_is_lifecycle_only():
    """The logging must not rotate libint's recipe hash: a rotation would
    rebuild libint (~hours at the 5_4_4 default) on every dev box and fleet
    host for a change that cannot alter the built artifact. The plumbing
    therefore lives inside vibeqc-recipe-hash lifecycle-only markers, and
    the hashed recipe content must keep the real build command while never
    seeing the logging calls."""
    res = subprocess.run(
        [
            "bash",
            "-c",
            "set -uo pipefail\n"
            f". {shlex.quote(str(SCRIPTS / '_libint_max_am.sh'))}\n"
            f". {shlex.quote(str(NATIVE_STAMP))}\n"
            f"_vibeqc_recipe_content {shlex.quote(str(SCRIPTS / 'build_libint.sh'))}\n",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    stripped = res.stdout
    assert "cmake --build" in stripped  # the real recipe stays hashed
    assert "vibeqc_build_log" not in stripped  # the logging never is
    assert "_build_log.sh" not in stripped


@pytest.mark.parametrize("fn", ["vibeqc_build_log_begin", "vibeqc_build_log_end"])
def test_helper_defines_its_entry_points(fn: str):
    text = HELPER.read_text(encoding="utf-8")
    assert f"{fn}()" in text
