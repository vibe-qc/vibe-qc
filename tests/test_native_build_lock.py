"""Cross-platform safety regressions for the native dependency build lock."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_LOCK = REPO_ROOT / "scripts" / "_build_lock.sh"
LIFECYCLE_LOCK = REPO_ROOT / "scripts" / "_lifecycle_lock.sh"
NATIVE_UPDATER = REPO_ROOT / "scripts" / "update_native_deps.sh"
BUILDERS = (
    "build_libint.sh",
    "build_libxc.sh",
    "build_spglib.sh",
    "build_fftw.sh",
    "build_libecpint.sh",
    "build_openblas.sh",
)
BASE_PYTHON = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
BASH = "/bin/bash" if Path("/bin/bash").is_file() else shutil.which("bash")

pytestmark = pytest.mark.skipif(
    BASH is None or os.name != "posix",
    reason="the build lock requires Bash and POSIX fcntl",
)


def _lock_fixture(tmp_path: Path) -> tuple[Path, Path]:
    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    helper = scripts / BUILD_LOCK.name
    shutil.copy2(BUILD_LOCK, helper)
    return checkout, helper


def _lock_env(*, fake_bin: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("VIBEQC_BUILD_LOCK_") or name == "VIBEQC_BUILD_LOCK":
            env.pop(name)
    env["VIBEQC_BUILD_LOCK_PYTHON"] = str(BASE_PYTHON)
    if fake_bin is not None:
        env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    return env


def _start_holder(
    helper: Path, *, env: dict[str, str], nested: bool = False
) -> subprocess.Popen[str]:
    if nested:
        snippet = """
set -e
. "$1"
vibeqc_acquire_build_lock
printf 'outer-ready\n'
"$2" -c '
    set -e
    . "$1"
    vibeqc_acquire_build_lock
    printf "nested-ready\\n"
    IFS= read -r _
' nested "$1"
"""
    else:
        snippet = """
set -e
. "$1"
vibeqc_acquire_build_lock
printf 'ready\n'
IFS= read -r _
"""
    process = subprocess.Popen(
        [str(BASH), "-c", snippet, "holder", str(helper), str(BASH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert process.stdout is not None
    expected = ("outer-ready\n", "nested-ready\n") if nested else ("ready\n",)
    observed = tuple(process.stdout.readline() for _ in expected)
    if observed != expected:
        assert process.stderr is not None
        stderr = process.stderr.read()
        process.kill()
        process.wait(timeout=5)
        pytest.fail(f"build-lock holder failed: {observed!r}; {stderr}")
    return process


def _stop_holder(process: subprocess.Popen[str]) -> None:
    assert process.stdin is not None
    process.stdin.write("\n")
    process.stdin.flush()
    process.communicate(timeout=10)
    assert process.returncode == 0


def _acquire_once(
    helper: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(BASH),
            "-c",
            '. "$1"; vibeqc_acquire_build_lock',
            "contender",
            str(helper),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_no_flock_command_still_serializes_concurrent_builds(tmp_path: Path) -> None:
    _, helper = _lock_fixture(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    flock_marker = tmp_path / "flock-was-called"
    fake_flock = fake_bin / "flock"
    fake_flock.write_text(
        f"#!/bin/sh\n: > {shlex.quote(str(flock_marker))}\nexit 99\n",
        encoding="utf-8",
    )
    fake_flock.chmod(0o755)
    env = _lock_env(fake_bin=fake_bin)

    holder = _start_holder(helper, env=env)
    try:
        contender = _acquire_once(helper, env)
        assert contender.returncode != 0
        assert "already running" in contender.stderr
        assert not flock_marker.exists()
    finally:
        _stop_holder(holder)


def test_nested_subprocess_reuses_verified_process_tree_lock(tmp_path: Path) -> None:
    _, helper = _lock_fixture(tmp_path)
    env = _lock_env()
    holder = _start_holder(helper, env=env, nested=True)
    try:
        contender = _acquire_once(helper, env)
        assert contender.returncode != 0
        assert "already running" in contender.stderr
    finally:
        _stop_holder(holder)


def test_lifecycle_lock_can_release_before_build_lock(tmp_path: Path) -> None:
    checkout, helper = _lock_fixture(tmp_path)
    lifecycle_helper = checkout / "scripts" / LIFECYCLE_LOCK.name
    shutil.copy2(LIFECYCLE_LOCK, lifecycle_helper)
    target = tmp_path / "environment"
    target.parent.mkdir(parents=True, exist_ok=True)
    snippet = """
set -e
. "$1"
. "$2"
vibe_toolset_acquire_lifecycle_lock "$3" "$4" "$5" test
vibeqc_acquire_build_lock
vibe_toolset_release_lifecycle_lock
printf 'lifecycle-released\n'
"""
    result = subprocess.run(
        [
            str(BASH),
            "-c",
            snippet,
            "dual-lock",
            str(lifecycle_helper),
            str(helper),
            str(BASE_PYTHON),
            str(checkout),
            str(target),
        ],
        env=_lock_env(),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "lifecycle-released\n"


def test_build_lock_releases_after_holder_crash(tmp_path: Path) -> None:
    _, helper = _lock_fixture(tmp_path)
    env = _lock_env()
    holder = _start_holder(helper, env=env)
    holder.kill()
    holder.wait(timeout=5)

    deadline = time.monotonic() + 5
    while True:
        result = _acquire_once(helper, env)
        if result.returncode == 0:
            break
        if time.monotonic() >= deadline:
            pytest.fail(
                result.stderr or "kernel build lock did not release after crash"
            )
        time.sleep(0.05)


@pytest.mark.parametrize("unsafe_kind", ("third-party-symlink", "lock-symlink"))
def test_unsafe_lock_resource_fails_closed(tmp_path: Path, unsafe_kind: str) -> None:
    checkout, helper = _lock_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    if unsafe_kind == "third-party-symlink":
        (checkout / "third_party").symlink_to(outside, target_is_directory=True)
    else:
        third_party = checkout / "third_party"
        third_party.mkdir()
        (third_party / ".build.lock").symlink_to(outside / "foreign-lock")

    result = _acquire_once(helper, _lock_env())
    assert result.returncode != 0
    assert "could not acquire the native build lock" in result.stderr


def test_forged_inherited_state_fails_closed(tmp_path: Path) -> None:
    checkout, helper = _lock_fixture(tmp_path)
    env = _lock_env()
    env.update(
        {
            "VIBEQC_BUILD_LOCK_ACTIVE": "1",
            "VIBEQC_BUILD_LOCK_HELPER_PID": str(os.getpid()),
            "VIBEQC_BUILD_LOCK_STATE_DIR": ("/tmp/vibeqc-build-lock-state.forged"),
            "VIBEQC_BUILD_LOCK_ROOT": str(checkout.resolve()),
        }
    )
    result = _acquire_once(helper, env)
    assert result.returncode != 0
    assert "helper/state is not active and valid" in result.stderr


def test_missing_explicit_lock_python_fails_closed(tmp_path: Path) -> None:
    _, helper = _lock_fixture(tmp_path)
    env = _lock_env()
    env["VIBEQC_BUILD_LOCK_PYTHON"] = str(tmp_path / "missing-python")
    result = _acquire_once(helper, env)
    assert result.returncode != 0
    assert "not a usable isolated external Python" in result.stderr


def test_libxc_direct_helper_checks_lock_before_installed_sentinel(
    tmp_path: Path,
) -> None:
    checkout, helper = _lock_fixture(tmp_path)
    scripts = checkout / "scripts"
    shutil.copy2(REPO_ROOT / "scripts" / "build_libxc.sh", scripts)
    (scripts / "_safe_build_env.sh").write_text("true\n", encoding="utf-8")
    (scripts / "_verify_source.sh").write_text("true\n", encoding="utf-8")
    sentinel = (
        checkout
        / "third_party"
        / "libxc"
        / "install"
        / "lib"
        / "cmake"
        / "Libxc"
        / "LibxcConfig.cmake"
    )
    sentinel.parent.mkdir(parents=True)
    sentinel.touch()
    env = _lock_env()

    holder = _start_holder(helper, env=env)
    try:
        result = subprocess.run(
            [str(BASH), str(scripts / "build_libxc.sh")],
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "already running" in result.stderr
        assert "already installed" not in result.stdout
    finally:
        _stop_holder(holder)


def test_native_updater_locks_before_reading_current_state(tmp_path: Path) -> None:
    checkout, helper = _lock_fixture(tmp_path)
    scripts = checkout / "scripts"
    shutil.copy2(NATIVE_UPDATER, scripts)
    shutil.copy2(REPO_ROOT / "scripts" / "_libint_max_am.sh", scripts)
    stamp_read = tmp_path / "stamp-was-read"
    (scripts / "_native_stamp.sh").write_text(
        """
: > "$VIBEQC_TEST_STAMP_READ"
VIBEQC_STAMP_PATH="$PWD/third_party/.native-build-stamp"
_vibeqc_stamp_pairs() { :; }
vibeqc_stamp_drifted_deps() { :; }
vibeqc_stamp_write() { :; }
""",
        encoding="utf-8",
    )
    env = _lock_env()
    env["VIBEQC_TEST_STAMP_READ"] = str(stamp_read)

    holder = _start_holder(helper, env=env)
    try:
        result = subprocess.run(
            [str(BASH), str(scripts / "update_native_deps.sh")],
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "already running" in result.stderr
        assert "all current" not in result.stdout
        assert not stamp_read.exists()
    finally:
        _stop_holder(holder)


def test_native_updater_dry_run_remains_lock_free(tmp_path: Path) -> None:
    checkout, helper = _lock_fixture(tmp_path)
    scripts = checkout / "scripts"
    shutil.copy2(NATIVE_UPDATER, scripts)
    shutil.copy2(REPO_ROOT / "scripts" / "_libint_max_am.sh", scripts)
    stamp_read = tmp_path / "dry-run-stamp-was-read"
    (scripts / "_native_stamp.sh").write_text(
        """
: > "$VIBEQC_TEST_STAMP_READ"
VIBEQC_STAMP_PATH="$PWD/third_party/.native-build-stamp"
_vibeqc_stamp_pairs() { :; }
vibeqc_stamp_drifted_deps() { :; }
vibeqc_stamp_write() { :; }
""",
        encoding="utf-8",
    )
    env = _lock_env()
    env["VIBEQC_TEST_STAMP_READ"] = str(stamp_read)

    holder = _start_holder(helper, env=env)
    try:
        result = subprocess.run(
            [str(BASH), str(scripts / "update_native_deps.sh"), "--dry-run"],
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "all current" in result.stdout
        assert stamp_read.is_file()
    finally:
        _stop_holder(holder)


@pytest.mark.parametrize("builder", BUILDERS)
def test_every_native_builder_locks_before_probe_or_mutation(builder: str) -> None:
    source = (REPO_ROOT / "scripts" / builder).read_text(encoding="utf-8")
    begin_marker = "# vibeqc-recipe-hash: lifecycle-only-begin"
    end_marker = "# vibeqc-recipe-hash: lifecycle-only-end"
    lifecycle_begin = source.index(f"\n{begin_marker}\n")
    acquire = source.index("\nvibeqc_acquire_build_lock\n")
    lifecycle_end = source.index(f"\n{end_marker}\n")
    version_assignment = source.index('_VERSION="')
    assert lifecycle_begin < acquire < lifecycle_end < version_assignment

    block = source.split(begin_marker, 1)[1].split(end_marker, 1)[0]
    executable = [
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert executable == [
        '. "$SCRIPT_DIR/_build_lock.sh"',
        "vibeqc_acquire_build_lock",
    ]
