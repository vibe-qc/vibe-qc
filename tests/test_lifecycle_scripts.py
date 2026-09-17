"""Regression tests for the root vibe-qc install lifecycle scripts.

The lifecycle entry points are intentionally shell-only so they work before
vibe-qc has a Python environment. These tests exercise their destructive-path
guards and transaction state directly without compiling the native core.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests import companion_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_HELPERS = REPO_ROOT / "scripts" / "_venv_helpers.sh"
INSTALL = REPO_ROOT / "scripts" / "install.sh"
UPDATE = REPO_ROOT / "scripts" / "update.sh"
UNINSTALL = REPO_ROOT / "scripts" / "uninstall.sh"
OPTIONAL_TOOLS = REPO_ROOT / "scripts" / "install_optional_tools.sh"
CAPTURE_UPDATE = REPO_ROOT / "scripts" / "update_vibeview_capture_env.sh"
SETUP_HELPERS = REPO_ROOT / "scripts" / "_setup_helpers.sh"
LIFECYCLE_LOCK = REPO_ROOT / "scripts" / "_lifecycle_lock.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="bash is required for lifecycle script tests",
)


@pytest.fixture
def vibe_view_checkout() -> Path:
    """The sibling vibe-view checkout the capture updater builds from.

    vibe-view left this tree in the 2026-09 split (#320). A checkout without
    it is a supported configuration, so the tests that need the real
    lifecycle helpers skip with a reason naming the coverage that did not
    run; VIBEQC_REQUIRE_COMPANION_CHECKOUTS turns that skip into a failure.
    """
    return companion_paths.require(
        "vibe-view", "vibe-view capture-environment coverage"
    )


@pytest.fixture
def vibe_queue_checkout() -> Path:
    """The sibling vibe-queue checkout, resolved the same way."""
    return companion_paths.require(
        "vibe-queue", "cross-component lifecycle-lock coverage"
    )


_LIFECYCLE_LOCK_ROOT = Path(f"/tmp/vibe-toolset-lifecycle-locks-{os.geteuid()}")


def _lifecycle_lock_entries() -> set[str]:
    try:
        return {entry.name for entry in _LIFECYCLE_LOCK_ROOT.iterdir()}
    except FileNotFoundError:
        return set()


def _remove_unheld_lifecycle_lock(path: Path) -> bool:
    """Delete one lock file unless a live process holds it.

    ``flock`` locks the open file, not the name, so a lock file is only safe
    to delete while nobody holds it. The file is opened without following
    symlinks, a non-blocking exclusive ``flock`` is attempted, and on success
    the name is unlinked *while the lock is still held*, so no acquirer can
    open this inode in between. A held lock (``flock`` refused) is left alone
    and reported as not removed. Returns True when the file is gone.
    """
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return True
    finally:
        os.close(fd)


@pytest.fixture(autouse=True)
def _prune_lifecycle_locks_minted_by_this_test():
    """Remove the lock files a test leaves in the uid-global lock directory.

    ``scripts/_lifecycle_lock.sh`` mints one ``checkout-<sha256>.lock`` and
    one ``target-<sha256>.lock`` per resource under
    ``/tmp/vibe-toolset-lifecycle-locks-<uid>`` and never deletes them. Real
    installs make one pair per checkout and venv; the tests here mint a fresh
    pair per ``tmp_path`` resource on every run, so the directory grew by
    thousands of empty files a day (#204: 10,655 on one machine). Nine of the
    27 files one run of this module leaves are for targets the scripts never
    created, so they cannot be recomputed from what is left on disk; the
    directory is snapshotted instead and only names that appeared during
    this test are considered.

    Safety: only files that did not exist before the test are touched, and
    each is deleted only if a non-blocking ``flock`` succeeds, so a lock a
    live process holds (this test's own holder that was not stopped, or a
    concurrent real install on this machine) is never removed. Deleting an
    unheld file is harmless: the next acquirer creates a fresh one with
    ``O_EXCL``. The production helper's own check-then-``flock`` window
    (#204, option 2) is not widened here; closing it needs the helper to
    re-check the inode after locking, which is a separate change.
    """
    before = _lifecycle_lock_entries()
    yield
    for name in sorted(_lifecycle_lock_entries() - before):
        _remove_unheld_lifecycle_lock(_LIFECYCLE_LOCK_ROOT / name)


def _bash(snippet: str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", snippet],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _source_helpers(snippet: str) -> str:
    return f". {shlex.quote(str(VENV_HELPERS))}\n{snippet}\n"


def _recognisable_venv(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "pyvenv.cfg").write_text(
        f"executable = {sys.executable}\nversion = {sys.version_info.major}."
        f"{sys.version_info.minor}.{sys.version_info.micro}\n",
        encoding="utf-8",
    )


def _optional_python_metadata_case(path: Path) -> str:
    """Shell case arm emulating the optional helper's isolated probe."""
    return (
        "  *VIBEQC_OPTIONAL_PYTHON_V1*)\n"
        "    printf '%s\\n' VIBEQC_OPTIONAL_PYTHON_V1 "
        f"{shlex.quote(str(path.resolve()))} "
        f"{shlex.quote(str(Path(sys.executable).resolve()))} 1\n"
        "    exit 0 ;;\n"
        "  *sysconfig.get_path*)\n"
        f"    printf '%s\\n' {shlex.quote(str((path / 'bin').resolve()))}\n"
        "    exit 0 ;;\n"
    )


def _start_toolset_lock_holder(
    target: Path, checkout: Path = REPO_ROOT
) -> subprocess.Popen[str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    checkout.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "/bin/bash",
            "-c",
            (
                f". {shlex.quote(str(LIFECYCLE_LOCK))}; "
                "vibe_toolset_acquire_lifecycle_lock "
                f"{shlex.quote(str(Path(sys.executable).resolve()))} "
                f"{shlex.quote(str(checkout))} {shlex.quote(str(target))} test; "
                "trap 'vibe_toolset_release_lifecycle_lock' EXIT; "
                "printf 'ready\\n'; IFS= read -r _"
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    ready = process.stdout.readline()
    if ready != "ready\n":
        assert process.stderr is not None
        pytest.fail(process.stderr.read() or "lifecycle lock holder did not start")
    return process


def _toolset_lock_path(scope: str, resource: Path) -> Path:
    digest = hashlib.sha256(f"{scope}:{resource.resolve()}".encode()).hexdigest()
    current = resource.resolve(strict=False)
    while not current.exists():
        current = current.parent
    owner_uid = current.lstat().st_uid
    return Path(f"/tmp/vibe-toolset-lifecycle-locks-{owner_uid}") / (
        f"{scope}-{digest}.lock"
    )


def _mark_owned_venv(path: Path, checkout: Path = REPO_ROOT) -> None:
    (path / ".vibeqc-lifecycle.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "vibe-qc-lifecycle-venv",
                "project": str(checkout.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _add_vibeqc_pep610_record(path: Path, checkout: Path = REPO_ROOT) -> None:
    record = (
        path
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "vibe_qc-test.dist-info"
        / "direct_url.json"
    )
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps({"url": checkout.resolve().as_uri(), "dir_info": {}}) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("relative_target", [".", "scripts", ".git/env"])
def test_dangerous_checkout_targets_are_rejected(
    tmp_path: Path, relative_target: str
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / ".git").mkdir()
    target = checkout / relative_target
    result = _bash(
        _source_helpers(
            "vibeqc_assert_safe_venv_target "
            f"{shlex.quote(str(target.resolve(strict=False)))} "
            f"{shlex.quote(str(checkout))}"
        )
    )
    assert result.returncode != 0
    assert "refusing" in result.stderr.lower()


@pytest.mark.parametrize("target", ["/", "/tmp", "/usr", "/var"])
def test_top_level_system_targets_are_rejected(target: str) -> None:
    result = _bash(
        _source_helpers(
            "vibeqc_assert_safe_venv_target "
            f"{shlex.quote(target)} {shlex.quote(str(REPO_ROOT))}"
        )
    )
    assert result.returncode != 0
    assert "unsafe virtualenv target" in result.stderr


@pytest.mark.parametrize(
    "target",
    (
        "/etc/hosts",
        "/etc/vibeqc-environment-that-does-not-exist",
        "/boot/vibeqc-environment-that-does-not-exist",
        "/nix/vibeqc-environment-that-does-not-exist",
        "/snap/vibeqc-environment-that-does-not-exist",
        "/usr/local/share/vibeqc-environment-that-does-not-exist",
        "/private/etc/vibeqc-environment-that-does-not-exist",
    ),
)
def test_protected_system_descendants_are_rejected(target: str) -> None:
    result = _bash(
        _source_helpers(
            "vibeqc_assert_safe_venv_target "
            f"{shlex.quote(target)} {shlex.quote(str(REPO_ROOT))}"
        )
    )
    assert result.returncode != 0
    assert "protected system location" in result.stderr


def test_foreign_git_metadata_descendants_are_rejected_existing_or_missing(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "other-repository" / ".git"
    existing = metadata / "existing-environment"
    existing.mkdir(parents=True)
    missing = metadata / "missing-environment"

    for target in (existing, missing):
        result = _bash(
            _source_helpers(
                "vibeqc_assert_safe_venv_target "
                f"{shlex.quote(str(target))} {shlex.quote(str(REPO_ROOT))}"
            )
        )
        assert result.returncode != 0
        assert "Git metadata" in result.stderr


def test_safe_temp_user_and_xdg_descendants_remain_allowed(tmp_path: Path) -> None:
    home = tmp_path / "user-home"
    xdg = home / ".local" / "share"
    existing = xdg / "existing-environment"
    existing.mkdir(parents=True)
    missing = xdg / "missing-environment"
    env = dict(os.environ, HOME=str(home), XDG_DATA_HOME=str(xdg))

    for target in (existing, missing, tmp_path / "ordinary-temp-environment"):
        result = subprocess.run(
            [
                "/bin/bash",
                "-c",
                _source_helpers(
                    "vibeqc_assert_safe_venv_target "
                    f"{shlex.quote(str(target))} {shlex.quote(str(REPO_ROOT))}"
                ),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_caller_controlled_home_cannot_exempt_a_system_location() -> None:
    poisoned_home = Path("/etc").resolve()
    target = poisoned_home / "vibeqc-environment-that-does-not-exist"
    env = dict(os.environ, HOME=str(poisoned_home), XDG_DATA_HOME=str(poisoned_home))
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            _source_helpers(
                "vibeqc_assert_safe_venv_target "
                f"{shlex.quote(str(target))} {shlex.quote(str(REPO_ROOT))}"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "protected system location" in result.stderr


def test_home_owned_by_another_uid_cannot_exempt_its_descendants(
    tmp_path: Path,
) -> None:
    home = Path.home().resolve()
    home_text = str(home)
    if not (
        home_text == "/root"
        or home_text.startswith(("/Users/", "/home/", "/var/home/"))
    ):
        pytest.skip("test account does not use a standard account-home path")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_stat = fake_bin / "stat"
    fake_stat.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"${FAKE_STAT_UID:?}\"\n",
        encoding="utf-8",
    )
    fake_stat.chmod(0o755)
    target = home / f"vibeqc-other-owner-probe-{os.getpid()}"
    env = dict(
        os.environ,
        HOME=str(home),
        PATH=f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        FAKE_STAT_UID=str(os.geteuid() + 1),
    )
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            _source_helpers(
                "vibeqc_assert_safe_venv_target "
                f"{shlex.quote(str(target))} {shlex.quote(str(REPO_ROOT))}"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "protected system location" in result.stderr


def test_target_lock_contends_across_different_checkouts(tmp_path: Path) -> None:
    checkout_a = tmp_path / "checkout-a"
    checkout_b = tmp_path / "checkout-b"
    (checkout_a / "scripts").mkdir(parents=True)
    (checkout_b / "scripts").mkdir(parents=True)
    target = tmp_path / "shared-environment"
    helper = shlex.quote(str(VENV_HELPERS))
    target_arg = shlex.quote(str(target))
    checkout_a_arg = shlex.quote(str(checkout_a))
    checkout_b_arg = shlex.quote(str(checkout_b))
    holder = subprocess.Popen(
        [
            "/bin/bash",
            "-c",
            (
                f". {helper}; "
                f"vibeqc_acquire_lifecycle_lock {target_arg} {checkout_a_arg} test; "
                "trap 'vibeqc_release_lifecycle_lock' EXIT; "
                "printf 'ready\\n'; IFS= read -r _"
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline() == "ready\n"
    contender_snippet = (
        f". {helper}; "
        f"if vibeqc_acquire_lifecycle_lock {target_arg} {checkout_b_arg} test; then "
        "vibeqc_release_lifecycle_lock; exit 0; fi; exit 1"
    )

    try:
        blocked = _bash(contender_snippet)
        assert blocked.returncode != 0
        assert "lifecycle operation already active" in blocked.stderr
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0

    acquired = _bash(contender_snippet)
    assert acquired.returncode == 0, acquired.stderr


def test_shared_lock_contends_across_components_for_the_same_target(
    tmp_path: Path,
    vibe_view_checkout: Path,
    vibe_queue_checkout: Path,
) -> None:
    """Across repositories, the target is the shared resource, not the checkout.

    This test used to assert that a held vibe-view target lock also blocked a
    vq lifecycle lock on an unrelated target, under "active on this checkout".
    That was true only while both components lived in one tree: the checkout
    lock is keyed by checkout, and since the 2026-09 split vibe-view and
    vibe-queue have two. Serialising two separate repositories' lifecycle
    operations would be wrong, so the checkout half is inverted here and the
    target half -- two components must never mutate one environment at once --
    is kept as the cross-component contract that survived the split.
    """
    view_helper = vibe_view_checkout / "scripts" / "_venv_helpers.sh"
    vq_helper = vibe_queue_checkout / "scripts" / "_venv_helpers.sh"
    target_a = tmp_path / "component-a-environment"
    target_b = tmp_path / "component-b-environment"

    def hold_view_target(target: Path) -> subprocess.Popen[str]:
        process = subprocess.Popen(
            [
                "/bin/bash",
                "-c",
                (
                    f". {shlex.quote(str(view_helper))}; "
                    "vibe_view_acquire_target_lock "
                    f"{shlex.quote(str(target))} test "
                    f"{shlex.quote(sys.executable)}; "
                    "trap 'vibe_view_release_target_lock' EXIT; "
                    "printf 'ready\\n'; IFS= read -r _"
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        assert process.stdout.readline() == "ready\n"
        return process

    holder = hold_view_target(target_a)
    try:
        other_checkout_allowed = subprocess.run(
            [
                "/bin/bash",
                "-c",
                (
                    f". {shlex.quote(str(vq_helper))}; "
                    f"vq_acquire_lifecycle_lock {shlex.quote(str(target_b))} test"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert other_checkout_allowed.returncode == 0, (
            other_checkout_allowed.stdout + other_checkout_allowed.stderr
        )
        assert "active on this checkout" not in other_checkout_allowed.stderr
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0

    other_checkout = tmp_path / "other-checkout"
    (other_checkout / "scripts").mkdir(parents=True)
    holder = hold_view_target(target_a)
    try:
        target_blocked = _bash(
            _source_helpers(
                "vibeqc_acquire_lifecycle_lock "
                f"{shlex.quote(str(target_a))} "
                f"{shlex.quote(str(other_checkout))} test "
                f"{shlex.quote(sys.executable)}"
            )
        )
        assert target_blocked.returncode != 0
        assert "active on this target" in target_blocked.stderr
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0


def test_shared_lock_survives_exec_handoff_without_contention_gap(
    tmp_path: Path,
) -> None:
    target = tmp_path / "handoff-environment"
    helper = shlex.quote(str(LIFECYCLE_LOCK))
    python = shlex.quote(str(Path(sys.executable).resolve()))
    checkout = shlex.quote(str(REPO_ROOT))
    target_arg = shlex.quote(str(target))
    child = (
        f"set -e; . {helper}; "
        '[ "$VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE" = 1 ]; '
        "trap 'vibe_toolset_release_lifecycle_lock' EXIT; "
        "printf 'ready\\n'; IFS= read -r _"
    )
    outer = (
        f"set -e; . {helper}; "
        f"vibe_toolset_acquire_lifecycle_lock {python} {checkout} {target_arg} reinstall; "
        "vibe_toolset_prepare_lifecycle_lock_handoff; "
        f"exec /bin/bash -c {shlex.quote(child)}"
    )
    holder = subprocess.Popen(
        ["/bin/bash", "-c", outer],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline() == "ready\n"
    try:
        contender = _bash(
            f". {helper}; vibe_toolset_acquire_lifecycle_lock "
            f"{python} {checkout} {target_arg} contender"
        )
        assert contender.returncode != 0
        assert "lifecycle operation already active" in contender.stderr
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0

    source = (REPO_ROOT / "scripts" / "reinstall.sh").read_text(encoding="utf-8")
    handoff = source.index("vibe_toolset_prepare_lifecycle_lock_handoff")
    delegated_install = source.index('exec "$SCRIPT_DIR/install.sh"', handoff)
    assert handoff < delegated_install
    assert "vibeqc_release_lifecycle_lock\ntrap - EXIT" not in source


def test_shared_lock_rejects_partial_inherited_environment() -> None:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("VIBE_TOOLSET_INHERITED_"):
            env.pop(name)
    env["VIBE_TOOLSET_INHERITED_CHECKOUT"] = str(REPO_ROOT)
    result = subprocess.run(
        ["/bin/bash", "-c", f". {shlex.quote(str(LIFECYCLE_LOCK))}"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "incomplete inherited" in result.stderr


@pytest.mark.parametrize("helper_pid", ("$$", "999999999"))
def test_shared_lock_rejects_forged_or_dead_legacy_fifo_helper_handoff(
    helper_pid: str,
) -> None:
    helper = shlex.quote(str(LIFECYCLE_LOCK))
    checkout = shlex.quote(str(REPO_ROOT))
    result = _bash(
        f"""
state_dir="$(mktemp -d "/tmp/vibe-toolset-lifecycle-state.$EUID.XXXXXX")"
mkfifo "$state_dir/control"
: > "$state_dir/status"
trap 'exec 198>&-; rm -f "$state_dir/control" "$state_dir/status"; rmdir "$state_dir"' EXIT
exec 198</dev/null
VIBE_TOOLSET_INHERITED_PID=$$
VIBE_TOOLSET_INHERITED_HELPER_PID={helper_pid}
VIBE_TOOLSET_INHERITED_STATE="$state_dir"
VIBE_TOOLSET_INHERITED_CHECKOUT={checkout}
VIBE_TOOLSET_INHERITED_TARGET=/tmp/forged-vibeqc-target
VIBE_TOOLSET_INHERITED_ACTION=forged
export VIBE_TOOLSET_INHERITED_PID VIBE_TOOLSET_INHERITED_HELPER_PID \
    VIBE_TOOLSET_INHERITED_STATE VIBE_TOOLSET_INHERITED_CHECKOUT \
    VIBE_TOOLSET_INHERITED_TARGET VIBE_TOOLSET_INHERITED_ACTION
. {helper}
"""
    )
    assert result.returncode != 0
    assert "refusing legacy FIFO/helper" in result.stderr


def test_shared_lock_rejects_wrong_claimed_lock_paths(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    target = tmp_path / "venv"
    checkout.mkdir()
    fake_checkout = tmp_path / "fake-checkout.lock"
    fake_target = tmp_path / "fake-target.lock"
    result = _bash(
        f"""
exec 194<>{shlex.quote(str(fake_checkout))}
exec 195<>{shlex.quote(str(fake_target))}
VIBE_TOOLSET_INHERITED_PID=$$
VIBE_TOOLSET_INHERITED_CHECKOUT={shlex.quote(str(checkout.resolve()))}
VIBE_TOOLSET_INHERITED_TARGET={shlex.quote(str(target.resolve()))}
VIBE_TOOLSET_INHERITED_ACTION=forged
VIBE_TOOLSET_INHERITED_CHECKOUT_PATH={shlex.quote(str(fake_checkout))}
VIBE_TOOLSET_INHERITED_TARGET_PATH={shlex.quote(str(fake_target))}
VIBE_TOOLSET_INHERITED_CHECKOUT_FD=194
VIBE_TOOLSET_INHERITED_TARGET_FD=195
export VIBE_TOOLSET_INHERITED_PID VIBE_TOOLSET_INHERITED_CHECKOUT \
    VIBE_TOOLSET_INHERITED_TARGET VIBE_TOOLSET_INHERITED_ACTION \
    VIBE_TOOLSET_INHERITED_CHECKOUT_PATH VIBE_TOOLSET_INHERITED_TARGET_PATH \
    VIBE_TOOLSET_INHERITED_CHECKOUT_FD VIBE_TOOLSET_INHERITED_TARGET_FD
. {shlex.quote(str(LIFECYCLE_LOCK))}
"""
    )
    assert result.returncode != 0
    assert "descriptors are not active and valid" in result.stderr


def test_shared_lock_rejects_wrong_handoff_descriptor_numbers(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    target = tmp_path / "venv"
    checkout.mkdir()
    checkout_lock = _toolset_lock_path("checkout", checkout)
    target_lock = _toolset_lock_path("target", target)
    result = _bash(
        f"""
VIBE_TOOLSET_INHERITED_PID=$$
VIBE_TOOLSET_INHERITED_CHECKOUT={shlex.quote(str(checkout.resolve()))}
VIBE_TOOLSET_INHERITED_TARGET={shlex.quote(str(target.resolve()))}
VIBE_TOOLSET_INHERITED_ACTION=forged
VIBE_TOOLSET_INHERITED_CHECKOUT_PATH={shlex.quote(str(checkout_lock))}
VIBE_TOOLSET_INHERITED_TARGET_PATH={shlex.quote(str(target_lock))}
VIBE_TOOLSET_INHERITED_CHECKOUT_FD=193
VIBE_TOOLSET_INHERITED_TARGET_FD=195
export VIBE_TOOLSET_INHERITED_PID VIBE_TOOLSET_INHERITED_CHECKOUT \
    VIBE_TOOLSET_INHERITED_TARGET VIBE_TOOLSET_INHERITED_ACTION \
    VIBE_TOOLSET_INHERITED_CHECKOUT_PATH VIBE_TOOLSET_INHERITED_TARGET_PATH \
    VIBE_TOOLSET_INHERITED_CHECKOUT_FD VIBE_TOOLSET_INHERITED_TARGET_FD
. {shlex.quote(str(LIFECYCLE_LOCK))}
"""
    )
    assert result.returncode != 0
    assert "checkout descriptor is invalid" in result.stderr


def test_shared_lock_rejects_changed_named_lock_inode(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    target = tmp_path / "venv"
    checkout.mkdir()
    lock_root = Path(f"/tmp/vibe-toolset-lifecycle-locks-{os.geteuid()}")
    lock_root.mkdir(mode=0o700, exist_ok=True)
    lock_root.chmod(0o700)
    checkout_lock = _toolset_lock_path("checkout", checkout)
    target_lock = _toolset_lock_path("target", target)
    result = _bash(
        f"""
umask 077
exec 194<>{shlex.quote(str(checkout_lock))}
exec 195<>{shlex.quote(str(target_lock))}
rm -f {shlex.quote(str(target_lock))}
: > {shlex.quote(str(target_lock))}
VIBE_TOOLSET_INHERITED_PID=$$
VIBE_TOOLSET_INHERITED_CHECKOUT={shlex.quote(str(checkout.resolve()))}
VIBE_TOOLSET_INHERITED_TARGET={shlex.quote(str(target.resolve()))}
VIBE_TOOLSET_INHERITED_ACTION=forged
VIBE_TOOLSET_INHERITED_CHECKOUT_PATH={shlex.quote(str(checkout_lock))}
VIBE_TOOLSET_INHERITED_TARGET_PATH={shlex.quote(str(target_lock))}
VIBE_TOOLSET_INHERITED_CHECKOUT_FD=194
VIBE_TOOLSET_INHERITED_TARGET_FD=195
export VIBE_TOOLSET_INHERITED_PID VIBE_TOOLSET_INHERITED_CHECKOUT \
    VIBE_TOOLSET_INHERITED_TARGET VIBE_TOOLSET_INHERITED_ACTION \
    VIBE_TOOLSET_INHERITED_CHECKOUT_PATH VIBE_TOOLSET_INHERITED_TARGET_PATH \
    VIBE_TOOLSET_INHERITED_CHECKOUT_FD VIBE_TOOLSET_INHERITED_TARGET_FD
. {shlex.quote(str(LIFECYCLE_LOCK))}
"""
    )
    assert result.returncode != 0
    assert "descriptors are not active and valid" in result.stderr


def test_shared_lock_rejects_independent_descriptors_while_owner_holds(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    target = tmp_path / "venv"
    holder = _start_toolset_lock_holder(target, checkout)
    checkout_lock = _toolset_lock_path("checkout", checkout)
    target_lock = _toolset_lock_path("target", target)
    try:
        result = _bash(
            f"""
exec 194<>{shlex.quote(str(checkout_lock))}
exec 195<>{shlex.quote(str(target_lock))}
VIBE_TOOLSET_INHERITED_PID=$$
VIBE_TOOLSET_INHERITED_CHECKOUT={shlex.quote(str(checkout.resolve()))}
VIBE_TOOLSET_INHERITED_TARGET={shlex.quote(str(target.resolve()))}
VIBE_TOOLSET_INHERITED_ACTION=forged
VIBE_TOOLSET_INHERITED_CHECKOUT_PATH={shlex.quote(str(checkout_lock))}
VIBE_TOOLSET_INHERITED_TARGET_PATH={shlex.quote(str(target_lock))}
VIBE_TOOLSET_INHERITED_CHECKOUT_FD=194
VIBE_TOOLSET_INHERITED_TARGET_FD=195
export VIBE_TOOLSET_INHERITED_PID VIBE_TOOLSET_INHERITED_CHECKOUT \
    VIBE_TOOLSET_INHERITED_TARGET VIBE_TOOLSET_INHERITED_ACTION \
    VIBE_TOOLSET_INHERITED_CHECKOUT_PATH VIBE_TOOLSET_INHERITED_TARGET_PATH \
    VIBE_TOOLSET_INHERITED_CHECKOUT_FD VIBE_TOOLSET_INHERITED_TARGET_FD
. {shlex.quote(str(LIFECYCLE_LOCK))}
"""
        )
        assert result.returncode != 0
        assert "descriptors are not active and valid" in result.stderr
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0


def test_shell_owned_lifecycle_lock_has_no_killable_keeper(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    target = tmp_path / "venv"
    checkout.mkdir()
    result = _bash(
        f"""
. {shlex.quote(str(LIFECYCLE_LOCK))}
vibe_toolset_acquire_lifecycle_lock \
    {shlex.quote(str(Path(sys.executable).resolve()))} \
    {shlex.quote(str(checkout))} {shlex.quote(str(target))} test
[ -z "${{VIBE_TOOLSET_LIFECYCLE_LOCK_PID:-}}" ]
[ -z "${{VIBE_TOOLSET_LIFECYCLE_LOCK_STATE_DIR:-}}" ]
[ "$VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_FD" = 194 ]
[ "$VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_FD" = 195 ]
vibe_toolset_release_lifecycle_lock
"""
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lifecycle_lock_namespace_is_derived_from_resource_owner() -> None:
    source = LIFECYCLE_LOCK.read_text(encoding="utf-8")

    assert '"vibe-toolset-lifecycle-locks-%d" % uid' in source
    assert "return current.lstat().st_uid" in source
    assert "root_stat.st_uid != uid" in source
    assert "opened.st_uid != uid" in source
    assert "0o700" in source
    assert "0o600" in source


def _lifecycle_lock_path(scope: str, resource: Path) -> Path:
    """The lock file ``scripts/_lifecycle_lock.sh`` would use for a resource.

    Mirrors the naming in that script: ``sha256("<scope>:<abs path>")`` under
    ``/tmp/vibe-toolset-lifecycle-locks-<uid>``. Kept in step by
    ``test_lifecycle_lock_path_matches_the_shell_helper`` below, so a rename in
    the shell helper fails loudly here rather than quietly making the
    dry-run assertions vacuous.
    """
    digest = hashlib.sha256(
        f"{scope}:{resource}".encode("utf-8")
    ).hexdigest()
    return (Path(f"/tmp/vibe-toolset-lifecycle-locks-{os.geteuid()}")
            / f"{scope}-{digest}.lock")


def test_reinstall_dry_run_creates_no_lifecycle_lock_artifacts(
    tmp_path: Path,
) -> None:
    target = tmp_path / "owned-environment"
    _recognisable_venv(target)
    _mark_owned_venv(target)
    # Assert on the artifacts THIS invocation would create, not on the whole
    # of /tmp/vibe-toolset-lifecycle-locks-<uid>. That directory is global to
    # the uid, shared by every checkout and process on the machine, and
    # nothing ever prunes it -- it had 9926 entries when this test was
    # rewritten. Seventeen other tests in this file mint locks there from
    # their own tmp_path checkout/venv pairs, so a whole-directory snapshot
    # made this test pass or fail on what its neighbours happened to do.
    checkout_lock = _lifecycle_lock_path("checkout", REPO_ROOT)
    target_lock = _lifecycle_lock_path("target", target)
    # The target is a fresh tmp_path, so its lock cannot pre-exist; the
    # checkout lock may, left by an earlier run, so record it.
    checkout_lock_existed = checkout_lock.exists()
    state_pattern = f"vibe-toolset-lifecycle-state.{os.geteuid()}.*"
    before_states = set(Path("/tmp").glob(state_pattern))
    started = time.time()

    result = subprocess.run(
        [
            "/bin/bash",
            str(REPO_ROOT / "scripts" / "reinstall.sh"),
            "--venv",
            str(target),
            "--python",
            sys.executable,
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not target_lock.exists(), (
        f"dry run created a target lock: {target_lock}")
    assert checkout_lock.exists() == checkout_lock_existed, (
        f"dry run changed the checkout lock: {checkout_lock}")
    # Only state directories created during the subprocess window are this
    # run's doing; anything older belongs to a neighbour and is not ours to
    # assert on.
    new_states = [
        path for path in set(Path("/tmp").glob(state_pattern)) - before_states
        if path.stat().st_ctime >= started
    ]
    assert not new_states, f"dry run created lifecycle state: {new_states}"


def test_minted_lifecycle_locks_are_pruned_only_when_unheld(tmp_path: Path) -> None:
    """The autouse pruner never deletes a held lock and removes it once free.

    A real ``vibe_toolset_acquire_lifecycle_lock`` holder takes this test's
    checkout and target locks. While it lives, the pruning helper must refuse
    both files; after it exits they are unheld and must be removed. The
    autouse fixture would do the same at teardown; this pins the decision it
    relies on (#204).
    """
    checkout = tmp_path / "checkout"
    target = tmp_path / "venv"
    holder = _start_toolset_lock_holder(target, checkout)
    try:
        locks = [
            _toolset_lock_path("checkout", checkout),
            _toolset_lock_path("target", target),
        ]
        for lock in locks:
            assert lock.exists(), lock
            assert _remove_unheld_lifecycle_lock(lock) is False, (
                f"pruner deleted a held lock: {lock}")
            assert lock.exists(), lock
    finally:
        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.close()
        holder.wait(timeout=30)
    for lock in locks:
        assert _remove_unheld_lifecycle_lock(lock) is True, lock
        assert not lock.exists(), lock
    # Idempotent on an already-missing name.
    assert _remove_unheld_lifecycle_lock(locks[0]) is True


def test_lifecycle_lock_path_matches_the_shell_helper() -> None:
    """``_lifecycle_lock_path`` reproduces the shell helper's lock naming.

    The dry-run assertions above check for named lock files, so they would
    silently pass against the wrong names if the shell helper's scheme
    changed. Pin the scheme against its source of truth.
    """
    source = LIFECYCLE_LOCK.read_text(encoding="utf-8")
    assert 'hashlib.sha256((scope + ":" + resource).encode("utf-8"))' in source
    assert 'root / (scope + "-" + digest + ".lock")' in source
    assert '("vibe-toolset-lifecycle-locks-%d" % uid)' in source


def test_target_containing_checkout_is_rejected(tmp_path: Path) -> None:
    checkout = tmp_path / "container" / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    target = tmp_path / "container"
    result = _bash(
        _source_helpers(
            "vibeqc_assert_safe_venv_target "
            f"{shlex.quote(str(target))} {shlex.quote(str(checkout))}"
        )
    )
    assert result.returncode != 0
    assert "contains the checkout" in result.stderr


def test_unrecognised_directory_is_never_removed(tmp_path: Path) -> None:
    target = tmp_path / "not-a-venv"
    target.mkdir()
    sentinel = target / "keep-me"
    sentinel.write_text("user data", encoding="utf-8")
    result = _bash(
        _source_helpers(
            f"vibeqc_remove_venv {shlex.quote(str(target))} "
            f"{shlex.quote(str(REPO_ROOT))}"
        )
    )
    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "user data"
    assert "not recognisably a virtualenv" in result.stderr


def test_symlink_target_is_rejected(tmp_path: Path) -> None:
    real_venv = tmp_path / "real"
    _recognisable_venv(real_venv)
    link = tmp_path / "linked"
    link.symlink_to(real_venv, target_is_directory=True)
    result = _bash(
        _source_helpers(f"vibeqc_resolve_venv_path resolved {shlex.quote(str(link))}")
    )
    assert result.returncode != 0
    assert link.is_symlink()
    assert "symbolic link" in result.stderr


def test_failed_replacement_restores_previous_venv(tmp_path: Path) -> None:
    target = tmp_path / "environment"
    _recognisable_venv(target)
    _mark_owned_venv(target)
    sentinel = target / "old-state"
    sentinel.write_text("still usable", encoding="utf-8")
    snippet = _source_helpers(
        "set -euo pipefail\n"
        f"target={shlex.quote(str(target))}\n"
        f"repo={shlex.quote(str(REPO_ROOT))}\n"
        f"python={shlex.quote(str(Path(sys.executable).resolve()))}\n"
        'vibeqc_begin_venv_replacement "$target" "$repo" 1\n'
        "trap vibeqc_abort_venv_replacement EXIT\n"
        'vibeqc_start_venv_replacement "$target" "$repo" "$python" 0\n'
        'printf partial > "$target/new-state"\n'
        "false"
    )
    result = _bash(snippet)
    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "still usable"
    assert not (target / "new-state").exists()
    assert not list(tmp_path.glob("environment.previous.*"))


def test_replacement_refuses_foreign_target_appearing_after_begin(
    tmp_path: Path,
) -> None:
    target = tmp_path / "late foreign environment"
    sentinel = target / "foreign-state"
    snippet = _source_helpers(
        "set -euo pipefail\n"
        f"target={shlex.quote(str(target))}\n"
        f"repo={shlex.quote(str(REPO_ROOT))}\n"
        f"python={shlex.quote(str(Path(sys.executable).resolve()))}\n"
        'vibeqc_begin_venv_replacement "$target" "$repo" 0\n'
        'mkdir -p "$target"\n'
        'printf "version = 3.12\\n" > "$target/pyvenv.cfg"\n'
        'printf foreign > "$target/foreign-state"\n'
        "rc=0\n"
        'vibeqc_start_venv_replacement "$target" "$repo" "$python" 0 || rc=$?\n'
        "vibeqc_abort_venv_replacement\n"
        'printf "RC=%s\\n" "$rc"\n'
    )
    result = _bash(snippet)
    assert result.returncode == 0, result.stderr
    assert "RC=0" not in result.stdout
    assert "target changed before replacement mutation" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "foreign"
    assert not list(tmp_path.glob("late foreign environment.previous.*"))


def test_failed_commit_keeps_transaction_available_for_rollback(
    tmp_path: Path,
) -> None:
    target = tmp_path / "environment"
    _recognisable_venv(target)
    _mark_owned_venv(target)
    (target / "old-state").write_text("old", encoding="utf-8")
    snippet = _source_helpers(
        f"target={shlex.quote(str(target))}\n"
        f"repo={shlex.quote(str(REPO_ROOT))}\n"
        f"python={shlex.quote(str(Path(sys.executable).resolve()))}\n"
        'vibeqc_begin_venv_replacement "$target" "$repo" 1\n'
        'vibeqc_start_venv_replacement "$target" "$repo" "$python" 0\n'
        'printf partial > "$target/new-state"\n'
        "rc=0\n"
        "vibeqc_commit_venv_replacement || rc=$?\n"
        "vibeqc_abort_venv_replacement\n"
        'printf "RC=%s\\n" "$rc"'
    )
    result = _bash(snippet)
    assert result.returncode == 0, result.stderr
    assert "RC=1" in result.stdout
    assert (target / "old-state").read_text(encoding="utf-8") == "old"
    assert not (target / "new-state").exists()


def test_broken_venv_preserves_recorded_base_interpreter(tmp_path: Path) -> None:
    target = tmp_path / "broken"
    _recognisable_venv(target)
    result = _bash(
        _source_helpers(
            f"vibeqc_venv_base_python found {shlex.quote(str(target))}\n"
            'printf "%s\\n" "$found"'
        )
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == Path(sys.executable).resolve()


def test_python_inside_replaced_path_resolves_to_external_base(
    tmp_path: Path,
) -> None:
    target = tmp_path / "environment"
    python = target / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    result = _bash(
        _source_helpers(
            "vibeqc_resolve_base_python base "
            f"{shlex.quote(str(python))}\n"
            'printf "%s\\n" "$base"'
        )
    )
    assert result.returncode == 0, result.stderr
    resolved = Path(result.stdout.strip()).resolve()
    assert resolved == Path(sys.executable).resolve()
    assert target not in resolved.parents


def test_uninstall_removes_only_recognisable_venv_and_allows_root_ci(
    tmp_path: Path,
) -> None:
    target = tmp_path / "environment"
    _recognisable_venv(target)
    _mark_owned_venv(target)
    (target / "installed-file").write_text("x", encoding="utf-8")
    env = os.environ.copy()
    env["VIBEQC_BUILD_LOCK"] = "0"
    result = subprocess.run(
        ["bash", str(UNINSTALL), "--venv", str(target)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not target.exists()
    assert "Uninstall complete" in result.stdout


def test_uninstall_refuses_foreign_virtualenv_before_removal(tmp_path: Path) -> None:
    target = tmp_path / "shared-environment"
    _recognisable_venv(target)
    sentinel = target / "must-survive"
    sentinel.write_text("foreign\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(UNINSTALL),
            "--python",
            sys.executable,
            "--venv",
            str(target),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ownership is not proven" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "foreign\n"


@pytest.mark.parametrize(
    "command",
    (
        (INSTALL, "--current", "--force"),
        (UPDATE, "--recreate-venv"),
        (REPO_ROOT / "scripts" / "reinstall.sh",),
    ),
)
def test_replacement_commands_refuse_foreign_virtualenv_including_dry_run(
    tmp_path: Path, command: tuple[Path | str, ...]
) -> None:
    target = tmp_path / f"foreign-{Path(command[0]).stem}"
    _recognisable_venv(target)
    sentinel = target / "must-survive"
    sentinel.write_text("shared\n", encoding="utf-8")
    args = [
        "bash",
        str(command[0]),
        *(str(value) for value in command[1:]),
        "--python",
        sys.executable,
        "--venv",
        str(target),
        "--dry-run",
    ]

    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert "ownership is not proven" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "shared\n"


@pytest.mark.parametrize(
    "command",
    (
        (INSTALL, "--current", "--force"),
        (UPDATE, "--recreate-venv"),
        (REPO_ROOT / "scripts" / "reinstall.sh",),
    ),
)
def test_replacement_commands_accept_checkout_owned_virtualenv_dry_run(
    tmp_path: Path, command: tuple[Path | str, ...]
) -> None:
    target = tmp_path / f"owned-{Path(command[0]).stem}"
    _recognisable_venv(target)
    _mark_owned_venv(target)
    args = [
        "bash",
        str(command[0]),
        *(str(value) for value in command[1:]),
        "--python",
        sys.executable,
        "--venv",
        str(target),
        "--dry-run",
    ]

    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert target.exists()


def test_legacy_uninstall_requires_explicit_matching_pep610_adoption(
    tmp_path: Path,
) -> None:
    target = tmp_path / "legacy-environment"
    _recognisable_venv(target)
    _add_vibeqc_pep610_record(target)

    refused = subprocess.run(
        [
            "bash",
            str(UNINSTALL),
            "--python",
            sys.executable,
            "--venv",
            str(target),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "--adopt-legacy" in refused.stderr
    assert target.exists()

    adopted = subprocess.run(
        [
            "bash",
            str(UNINSTALL),
            "--python",
            sys.executable,
            "--venv",
            str(target),
            "--adopt-legacy",
        ],
        cwd=REPO_ROOT,
        env=dict(os.environ, VIBEQC_BUILD_LOCK="0"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert adopted.returncode == 0, adopted.stdout + adopted.stderr
    assert not target.exists()


def test_foreign_marker_is_never_legacy_adopted(tmp_path: Path) -> None:
    target = tmp_path / "foreign-marked-environment"
    _recognisable_venv(target)
    _mark_owned_venv(target, tmp_path / "other-checkout")
    _add_vibeqc_pep610_record(target)

    result = subprocess.run(
        [
            "bash",
            str(UNINSTALL),
            "--python",
            sys.executable,
            "--venv",
            str(target),
            "--adopt-legacy",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ownership is not proven" in result.stderr
    assert target.exists()


def test_uninstall_uses_external_inspector_not_activated_or_recorded_python(
    tmp_path: Path,
) -> None:
    target = tmp_path / "hostile-environment"
    _recognisable_venv(target)
    _mark_owned_venv(target)
    executed = tmp_path / "target-python-ran"
    recorded_executed = tmp_path / "recorded-python-ran"
    recorded = tmp_path / "recorded-python"
    recorded.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(recorded_executed))}\nexit 91\n",
        encoding="utf-8",
    )
    recorded.chmod(0o755)
    (target / "pyvenv.cfg").write_text(
        f"home = hostile\nexecutable = {recorded}\n",
        encoding="utf-8",
    )
    hostile = target / "bin" / "python3"
    hostile.parent.mkdir()
    hostile.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(executed))}\nexit 0\n",
        encoding="utf-8",
    )
    hostile.chmod(0o755)
    env = dict(os.environ, PATH=f"{hostile.parent}{os.pathsep}{os.environ['PATH']}")

    result = subprocess.run(
        ["bash", str(UNINSTALL), "--venv", str(target), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not executed.exists()
    assert not recorded_executed.exists()
    assert target.exists()


def test_ordinary_update_lock_accepts_activated_target_without_running_it(
    tmp_path: Path,
) -> None:
    target = tmp_path / "activated-environment"
    _recognisable_venv(target)
    _mark_owned_venv(target)
    executed = tmp_path / "activated-python-ran"
    activated = target / "bin" / "python3"
    activated.parent.mkdir()
    activated.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(executed))}\nexit 90\n",
        encoding="utf-8",
    )
    activated.chmod(0o755)
    env = dict(os.environ, PATH=f"{activated.parent}{os.pathsep}{os.environ['PATH']}")
    snippet = _source_helpers(
        f"target={shlex.quote(str(target))}\n"
        'vibe_toolset_find_external_python inspector "$target"\n'
        f'vibeqc_acquire_lifecycle_lock "$target" {shlex.quote(str(REPO_ROOT))} update "$inspector"\n'
        "vibeqc_release_lifecycle_lock"
    )

    result = subprocess.run(
        ["/bin/bash", "-c", snippet],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not executed.exists()


def test_uninstall_accepts_metadata_python_below_package_minimum(
    tmp_path: Path,
) -> None:
    target = tmp_path / "owned-environment"
    _recognisable_venv(target)
    _mark_owned_venv(target)
    inspector = tmp_path / "metadata-python"
    inspector.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *'sys.version_info < tuple'*) exit 0 ;;\n"
        "  *'sys.version_info < (3, 11)'*) exit 91 ;;\n"
        "esac\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    inspector.chmod(0o755)

    result = subprocess.run(
        [
            "/bin/bash",
            str(UNINSTALL),
            "--python",
            str(inspector),
            "--venv",
            str(target),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert target.exists()


def test_reinstall_proves_ownership_before_recorded_python_is_executed(
    tmp_path: Path,
) -> None:
    target = tmp_path / "foreign-with-hostile-config"
    _recognisable_venv(target)
    executed = tmp_path / "recorded-python-ran"
    hostile = tmp_path / "hostile-python"
    hostile.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(executed))}\nexit 0\n",
        encoding="utf-8",
    )
    hostile.chmod(0o755)
    (target / "pyvenv.cfg").write_text(
        f"executable = {hostile}\nversion = 3.13.0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "reinstall.sh"),
            "--venv",
            str(target),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ownership is not proven" in result.stderr
    assert not executed.exists()
    assert target.exists()


@pytest.mark.parametrize(
    ("script", "mode_args"),
    (
        (INSTALL, ("--current", "--force")),
        (REPO_ROOT / "scripts" / "reinstall.sh", ()),
        (UNINSTALL, ()),
        (REPO_ROOT / "scripts" / "update.sh", ("--dev", "--recreate-venv")),
    ),
)
def test_destructive_inspectors_ignore_target_pythonpath_sitecustomize(
    tmp_path: Path,
    script: Path,
    mode_args: tuple[str, ...],
) -> None:
    target = tmp_path / f"foreign-{script.stem}"
    _recognisable_venv(target)
    executed = tmp_path / f"{script.stem}-sitecustomize-ran"
    (target / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(executed)!r}).touch()\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(target))

    result = subprocess.run(
        [
            "bash",
            str(script),
            *mode_args,
            "--python",
            sys.executable,
            "--venv",
            str(target),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ownership is not proven" in result.stderr
    assert not executed.exists()
    assert target.exists()


def test_root_install_writes_ownership_only_after_verification() -> None:
    source = INSTALL.read_text(encoding="utf-8")
    verification = source.index(
        'vibeqc_verify_install "$VENV_PATH/bin/python" "$EXTRAS_GROUP"'
    )
    marker = source.index("vibeqc_mark_owned_venv", verification)
    commit = source.index("vibeqc_commit_venv_replacement", marker)
    assert verification < marker < commit


def test_root_mpi_install_verification_never_uses_singleton_init() -> None:
    helper = SETUP_HELPERS.read_text(encoding="utf-8")
    verify = helper.split("vibeqc_verify_install() {", 1)[1].split("\n}", 1)[0]
    assert "*,mpi,*)" in verify
    assert "mpi4py.rc.initialize = False" in verify
    assert verify.index("mpi4py.rc.initialize = False") < verify.index("import vibeqc")
    assert "if MPI.Is_initialized():" in verify
    assert (
        'raise SystemExit("generic MPI verification unexpectedly initialized MPI")'
        in verify
    )
    assert "MPI.Get_library_version()" in verify
    install = INSTALL.read_text(encoding="utf-8")
    update = UPDATE.read_text(encoding="utf-8")
    assert "vibeqc_verify_install" in install
    assert "vibeqc_verify_install" in update


def test_root_ownership_marker_is_checkout_specific_and_atomic(tmp_path: Path) -> None:
    target = tmp_path / "environment"
    _recognisable_venv(target)
    result = _bash(
        _source_helpers(
            "vibeqc_mark_owned_venv "
            f"{shlex.quote(str(target))} {shlex.quote(str(REPO_ROOT))} "
            f"{shlex.quote(sys.executable)}"
        )
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((target / ".vibeqc-lifecycle.json").read_text())
    assert payload == {
        "schema": 1,
        "kind": "vibe-qc-lifecycle-venv",
        "project": str(REPO_ROOT.resolve()),
    }
    assert not list(target.glob(".vibeqc-lifecycle.*.tmp"))


@pytest.mark.parametrize(
    ("script", "option"),
    [
        (INSTALL, "--branch"),
        (INSTALL, "--extras"),
        (INSTALL, "--python"),
        (INSTALL, "--venv"),
        (UPDATE, "--branch"),
        (UPDATE, "--ref"),
        (UPDATE, "--extras"),
        (UPDATE, "--python"),
        (UPDATE, "--venv"),
        (REPO_ROOT / "scripts" / "reinstall.sh", "--venv"),
        (UNINSTALL, "--venv"),
        (UNINSTALL, "--python"),
    ],
)
def test_lifecycle_value_options_reject_empty_and_option_like_values(
    script: Path, option: str
) -> None:
    empty = subprocess.run(
        ["/bin/bash", str(script), option, ""],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert empty.returncode != 0, (script, option)
    assert "non-empty" in empty.stderr

    option_like = subprocess.run(
        ["/bin/bash", str(script), option, "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert option_like.returncode != 0, (script, option)
    assert "not option" in option_like.stderr


def test_update_without_venv_exits_before_any_git_operation(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git_log = tmp_path / "git.log"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$VIBEQC_TEST_GIT_LOG"\nexit 97\n',
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "VIBEQC_BUILD_NICED": "1",
            "VIBEQC_BUILD_LOCK": "0",
            "VIBEQC_TEST_GIT_LOG": str(git_log),
            "VIBEQC_TOOLING_CHECKED": "1",
            "VIBEQC_VQ_COOPERATE": "0",
        }
    )
    missing = tmp_path / "missing-venv"
    result = subprocess.run(
        ["bash", str(UPDATE), "--dev", "--venv", str(missing)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Nothing was changed" in result.stderr
    assert not git_log.exists(), git_log.read_text(encoding="utf-8")


def test_install_dry_run_defaults_to_release_and_current_is_explicit(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env["VIBEQC_BUILD_NICED"] = "1"
    target = tmp_path / "new-venv"
    default = subprocess.run(
        ["bash", str(INSTALL), "--dry-run", "--venv", str(target)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert default.returncode == 0, default.stdout + default.stderr
    assert "branch:       release" in default.stdout

    current = subprocess.run(
        [
            "bash",
            str(INSTALL),
            "--current",
            "--dry-run",
            "--venv",
            str(target),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert current.returncode == 0, current.stdout + current.stderr
    assert "branch:       <leave working tree as-is>" in current.stdout


@pytest.mark.parametrize("script", [OPTIONAL_TOOLS, CAPTURE_UPDATE])
def test_auxiliary_lifecycle_help_needs_no_environment(script: Path) -> None:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    result = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "USAGE" in result.stdout


def test_capture_update_reports_the_sibling_layout_when_vibe_view_is_absent(
    tmp_path: Path,
) -> None:
    """#320: the refusal names a reachable path and how to satisfy it.

    Before the fix the updater looked under ``<checkout>/vibe-view/scripts``
    and exited 1 for every user, because no checkout of this repository has
    contained vibe-view since the split.
    """
    absent = tmp_path / "no-vibe-view-here"
    env = dict(os.environ, VIBE_VIEW_ROOT=str(absent))
    result = subprocess.run(
        ["/bin/bash", str(CAPTURE_UPDATE), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert str(absent) in result.stderr
    assert "separate repository" in result.stderr
    assert "VIBE_VIEW_ROOT=/path/to/vibe-view" in result.stderr
    # Spelled by concatenation, not by a path join, so the guard in
    # tests/test_toolset_lifecycle_contract.py does not read this negative
    # assertion as a new offender.
    dead_pre_split_path = f"{REPO_ROOT}/vibe-view"
    assert dead_pre_split_path not in result.stderr


def test_capture_update_rejects_an_empty_vibe_view_root() -> None:
    result = subprocess.run(
        ["/bin/bash", str(CAPTURE_UPDATE), "--vibe-view-root"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--vibe-view-root requires a path" in result.stderr


def test_capture_update_builds_from_an_explicitly_named_checkout(
    tmp_path: Path, vibe_view_checkout: Path
) -> None:
    result = subprocess.run(
        [
            "/bin/bash",
            str(CAPTURE_UPDATE),
            "--vibe-view-root",
            str(vibe_view_checkout),
            "--python",
            str(Path(sys.executable).resolve()),
            "--venv",
            str(tmp_path / "explicit-root-capture"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"source:   {vibe_view_checkout}" in result.stdout


@pytest.mark.parametrize("selector", ("--ref", "--branch"))
def test_capture_update_accepts_matching_source_assertion(
    tmp_path: Path, selector: str, vibe_view_checkout: Path,
) -> None:
    ref = "HEAD"
    if selector == "--ref":
        ref = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    target = tmp_path / f"capture-{selector.removeprefix('--')}"
    result = subprocess.run(
        [
            "/bin/bash",
            str(CAPTURE_UPDATE),
            selector,
            ref,
            "--python",
            str(Path(sys.executable).resolve()),
            "--venv",
            str(target),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"source ref: {ref} ({selector})" in result.stdout
    assert "source commit:" in result.stdout
    assert not target.exists()


@pytest.mark.parametrize("selector", ("--ref", "--branch"))
def test_capture_update_source_assertion_requires_value(
    selector: str,
) -> None:
    result = subprocess.run(
        ["/bin/bash", str(CAPTURE_UPDATE), selector],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"{selector} requires a source ref" in result.stderr


def test_capture_update_still_rejects_unknown_argument() -> None:
    result = subprocess.run(
        ["/bin/bash", str(CAPTURE_UPDATE), "--source", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unknown argument '--source'" in result.stderr


def test_capture_update_rejects_conflicting_source_assertions() -> None:
    result = subprocess.run(
        [
            "/bin/bash",
            str(CAPTURE_UPDATE),
            "--ref",
            "HEAD",
            "--branch",
            "main",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "choose only one of --ref or --branch" in result.stderr


@pytest.mark.parametrize(
    ("selector", "ref", "message"),
    (
        ("--ref", "HEAD^", "does not match checkout HEAD"),
        ("--branch", "vq-capture-ref-that-does-not-exist", "cannot resolve"),
    ),
)
def test_capture_update_rejects_invalid_source_before_target_mutation(
    tmp_path: Path, selector: str, ref: str, message: str,
) -> None:
    target = tmp_path / "must-not-be-created"
    result = subprocess.run(
        [
            "/bin/bash",
            str(CAPTURE_UPDATE),
            selector,
            ref,
            "--venv",
            str(target),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert not target.exists()


def test_capture_update_source_assertion_preserves_ownership_gate(
    tmp_path: Path, vibe_view_checkout: Path,
) -> None:
    target = tmp_path / "unowned-capture-environment"
    _recognisable_venv(target)
    sentinel = target / "must-survive"
    sentinel.write_text("preserve\n", encoding="utf-8")
    expected_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = subprocess.run(
        [
            "/bin/bash",
            str(CAPTURE_UPDATE),
            "--ref",
            expected_sha,
            "--python",
            str(Path(sys.executable).resolve()),
            "--venv",
            str(target),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "refusing to replace an unmarked vibe-view environment" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_optional_tools_dry_run_needs_no_preexisting_venv(tmp_path: Path) -> None:
    missing = tmp_path / "not-created"
    result = subprocess.run(
        [
            "bash",
            str(OPTIONAL_TOOLS),
            "--venv",
            str(missing),
            "--yes",
            "--dry-run",
            "moltui",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Would run:" in result.stdout
    assert "moltui>=0.1" in result.stdout
    assert not missing.exists()


def test_auxiliary_dry_runs_do_not_take_the_mutation_lock(tmp_path: Path) -> None:
    """The optional-tool dry run ignores a held mutation lock.

    Split from the capture half below: the optional-tool installer is owned by
    this repository outright, so it must keep being checked on a checkout that
    has no vibe-view sibling (#320).
    """
    holder = _start_toolset_lock_holder(tmp_path / "held-target")
    env = dict(os.environ, CI="true", GITLAB_CI="true")
    try:
        optional_target = tmp_path / "optional-dry-run"
        optional = subprocess.run(
            [
                "/bin/bash",
                str(OPTIONAL_TOOLS),
                "--venv",
                str(optional_target),
                "--yes",
                "--dry-run",
                "moltui",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert optional.returncode == 0, optional.stdout + optional.stderr
        assert not optional_target.exists()
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0


def test_capture_dry_run_does_not_take_the_mutation_lock(
    tmp_path: Path, vibe_view_checkout: Path
) -> None:
    holder = _start_toolset_lock_holder(tmp_path / "held-target")
    env = dict(os.environ, CI="true", GITLAB_CI="true")
    try:
        capture_target = tmp_path / "capture-dry-run"
        capture = subprocess.run(
            [
                "/bin/bash",
                str(CAPTURE_UPDATE),
                "--python",
                str(Path(sys.executable).resolve()),
                "--venv",
                str(capture_target),
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert capture.returncode == 0, capture.stdout + capture.stderr
        assert not capture_target.exists()
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0


def test_optional_tools_upgrade_refreshes_installed_tool(tmp_path: Path) -> None:
    fake_bin = tmp_path / "venv" / "bin"
    _recognisable_venv(fake_bin.parent)
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    call_log = tmp_path / "calls.log"
    fake_python.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$OPTIONAL_TOOL_CALL_LOG"\n'
        'case "$*" in\n' + _optional_python_metadata_case(fake_bin.parent) + "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_cli = fake_bin / "moltui"
    fake_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_cli.chmod(0o755)
    env = os.environ.copy()
    env["OPTIONAL_TOOL_CALL_LOG"] = str(call_log)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    result = subprocess.run(
        [
            "bash",
            str(OPTIONAL_TOOLS),
            "--python",
            "python",
            "--upgrade",
            "--yes",
            "moltui",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = call_log.read_text(encoding="utf-8")
    assert calls.count("-c import moltui") >= 2
    assert "-m pip install --upgrade moltui>=0.1" in calls


def test_optional_tools_repairs_import_without_console_command(tmp_path: Path) -> None:
    fake_bin = tmp_path / "venv" / "bin"
    _recognisable_venv(fake_bin.parent)
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    call_log = tmp_path / "calls.log"
    fake_python.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$OPTIONAL_TOOL_CALL_LOG"\n'
        'case "$*" in\n' + _optional_python_metadata_case(fake_bin.parent) + "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["OPTIONAL_TOOL_CALL_LOG"] = str(call_log)

    result = subprocess.run(
        [
            "bash",
            str(OPTIONAL_TOOLS),
            "--python",
            str(fake_python),
            "--yes",
            "moltui",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "incomplete; repairing" in result.stdout
    assert "-m pip install --upgrade moltui>=0.1" in call_log.read_text(
        encoding="utf-8"
    )
    assert "was not installed in the venv" in result.stderr


def test_optional_tools_uses_interpreter_reported_scripts_directory(
    tmp_path: Path,
) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    shim_bin = tmp_path / "shim"
    _recognisable_venv(venv_bin.parent)
    venv_bin.mkdir()
    shim_bin.mkdir()
    fake_python = venv_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        + _optional_python_metadata_case(venv_bin.parent)
        + "  *sysconfig.get_path*) printf '%s\\n' \"$OPTIONAL_TOOL_SCRIPTS_DIR\" ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    (shim_bin / "python").symlink_to(fake_python)
    fake_cli = venv_bin / "moltui"
    fake_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_cli.chmod(0o755)
    env = os.environ.copy()
    env["OPTIONAL_TOOL_SCRIPTS_DIR"] = str(venv_bin)

    result = subprocess.run(
        [
            "bash",
            str(OPTIONAL_TOOLS),
            "--python",
            str(shim_bin / "python"),
            "--yes",
            "moltui",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Already installed: moltui" in result.stdout


def test_optional_tools_refuses_false_successful_pip_install(tmp_path: Path) -> None:
    fake_bin = tmp_path / "venv" / "bin"
    _recognisable_venv(fake_bin.parent)
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        + _optional_python_metadata_case(fake_bin.parent)
        + "  '-c import moltui') exit 1 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_cli = fake_bin / "moltui"
    fake_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_cli.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(OPTIONAL_TOOLS),
            "--python",
            str(fake_python),
            "--yes",
            "moltui",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "pip returned success" in result.stderr


def test_optional_tools_isolates_python_probe_and_locks_before_health(
    tmp_path: Path,
) -> None:
    target = tmp_path / "selected-venv"
    subprocess.run([sys.executable, "-m", "venv", str(target)], check=True)
    other_checkout = tmp_path / "other-checkout"
    sentinel = tmp_path / "sitecustomize-ran"
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        'Path(os.environ["OPTIONAL_SENTINEL"]).write_text("ran")\n',
        encoding="utf-8",
    )
    holder = _start_toolset_lock_holder(target, other_checkout)
    env = dict(
        os.environ,
        PYTHONPATH=str(poison),
        OPTIONAL_SENTINEL=str(sentinel),
    )
    try:
        result = subprocess.run(
            [
                "/bin/bash",
                str(OPTIONAL_TOOLS),
                "--python",
                str(target / "bin" / "python"),
                "--yes",
                "moltui",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "active on this target" in result.stderr
        assert not sentinel.exists(), "sitecustomize ran before lock acquisition"
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0


def test_optional_tools_venv_selection_never_executes_target_before_lock(
    tmp_path: Path,
) -> None:
    target = tmp_path / "selected-venv"
    _recognisable_venv(target)
    sentinel = tmp_path / "target-python-ran"
    python = target / "bin" / "python"
    python.parent.mkdir()
    python.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(sentinel))}\nexit 0\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    holder = _start_toolset_lock_holder(target, tmp_path / "other-checkout")
    try:
        result = subprocess.run(
            [
                "/bin/bash",
                str(OPTIONAL_TOOLS),
                "--venv",
                str(target),
                "--yes",
                "moltui",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "active on this target" in result.stderr
        assert not sentinel.exists()
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0


def test_optional_tools_holds_target_lock_through_cli_verification(
    tmp_path: Path,
) -> None:
    target = tmp_path / "fake-venv"
    _recognisable_venv(target)
    fake_bin = target / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        + _optional_python_metadata_case(target)
        + f"  *sysconfig.get_path*) printf '%s\\n' {shlex.quote(str(fake_bin))} ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_cli = fake_bin / "moltui"
    verify_ready = tmp_path / "verify-ready"
    verify_release = tmp_path / "verify-release"
    cli_count = tmp_path / "cli-count"
    fake_cli.write_text(
        "#!/bin/sh\n"
        'count="$(sed -n \'1p\' "$OPTIONAL_CLI_COUNT" 2>/dev/null || true)"\n'
        'count="${count:-0}"\n'
        "count=$((count + 1))\n"
        'printf \'%s\\n\' "$count" > "$OPTIONAL_CLI_COUNT"\n'
        '[ "$count" -gt 1 ] || exit 1\n'
        ': > "$OPTIONAL_VERIFY_READY"\n'
        'while [ ! -e "$OPTIONAL_VERIFY_RELEASE" ]; do sleep 0.05; done\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    env = dict(
        os.environ,
        OPTIONAL_CLI_COUNT=str(cli_count),
        OPTIONAL_VERIFY_READY=str(verify_ready),
        OPTIONAL_VERIFY_RELEASE=str(verify_release),
    )
    process = subprocess.Popen(
        [
            "/bin/bash",
            str(OPTIONAL_TOOLS),
            "--python",
            str(fake_python),
            "--yes",
            "moltui",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(200):
            if verify_ready.exists() or process.poll() is not None:
                break
            time.sleep(0.05)
        assert verify_ready.exists(), "optional tool never reached CLI verification"
        other_checkout = tmp_path / "other-checkout"
        other_checkout.mkdir()
        contender = _bash(
            f". {shlex.quote(str(LIFECYCLE_LOCK))}; "
            "vibe_toolset_acquire_lifecycle_lock "
            f"{shlex.quote(str(Path(sys.executable).resolve()))} "
            f"{shlex.quote(str(other_checkout))} {shlex.quote(str(target))} test"
        )
        assert contender.returncode != 0
        assert "active on this target" in contender.stderr
    finally:
        verify_release.touch()
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stdout + stderr


def test_optional_tools_accepts_activated_target_with_external_lock_python(
    tmp_path: Path,
) -> None:
    target = tmp_path / "activated-venv"
    subprocess.run([sys.executable, "-m", "venv", "--copies", str(target)], check=True)
    python = target / "bin" / "python"
    purelib = subprocess.run(
        [str(python), "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    Path(purelib, "vibeqc.py").write_text("\n", encoding="utf-8")
    Path(purelib, "moltui.py").write_text("\n", encoding="utf-8")
    cli = target / "bin" / "moltui"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    env = dict(
        os.environ,
        VIRTUAL_ENV=str(target),
        PATH=f"{target / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    )
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["/bin/bash", str(OPTIONAL_TOOLS), "--yes", "moltui"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Already installed: moltui" in result.stdout


@pytest.mark.parametrize(
    "target",
    [
        "/etc/vibeqc-optional-environment-that-does-not-exist",
        str(REPO_ROOT / ".git" / "optional-environment"),
    ],
)
def test_optional_tools_refuses_unsafe_targets_before_execution(target: str) -> None:
    result = subprocess.run(
        [
            "/bin/bash",
            str(OPTIONAL_TOOLS),
            "--venv",
            target,
            "--yes",
            "moltui",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "refusing" in result.stderr.lower()


@pytest.mark.parametrize(
    ("profile", "pip_suffix", "companion"),
    [
        ("viewer-gpu", "", "viewer-gpu"),
        ("basisopt", "", "basisopt"),
        ("test", "[test]", ""),
        ("mpi", "[mpi]", ""),
        ("dispersion,mpi", "[dispersion,mpi]", ""),
    ],
)
def test_setup_helper_resolves_all_documented_component_extras(
    profile: str, pip_suffix: str, companion: str
) -> None:
    snippet = (
        f". {shlex.quote(str(SETUP_HELPERS))}\n"
        f"vibeqc_extras_to_pip_spec {shlex.quote(profile)} resolved\n"
        'printf "%s|%s\\n" "$resolved" "$VIBEQC_COLOCATED_EXTRA"'
    )
    result = _bash(snippet, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{pip_suffix}|{companion}"


@pytest.mark.parametrize("profile", ["basisopt"])
def test_setup_helper_preflights_colocated_component(
    tmp_path: Path, profile: str
) -> None:
    """Only vibe-basis is still co-located; see the viewer-gpu test below."""
    project_name = "vibe-basis"
    project = tmp_path / project_name
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    snippet = (
        f". {shlex.quote(str(SETUP_HELPERS))}\n"
        "vibeqc_assert_colocated_extra_available "
        f"{shlex.quote(str(tmp_path))} {shlex.quote(profile)}"
    )
    available = _bash(snippet)
    assert available.returncode == 0, available.stderr

    (project / "pyproject.toml").unlink()
    missing = _bash(snippet)
    assert missing.returncode != 0
    assert "co-located project is missing" in missing.stderr


def test_setup_helper_refuses_viewer_gpu_as_a_colocated_extra(
    tmp_path: Path,
) -> None:
    """viewer-gpu has no co-located project to preflight since the split.

    An in-tree ``vibe-view/pyproject.toml`` is planted here deliberately: the
    refusal is a statement about the layout, so it must not be talked out of
    it by a directory that happens to be there.
    """
    project = tmp_path / "vibe-view"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    snippet = (
        f". {shlex.quote(str(SETUP_HELPERS))}\n"
        "vibeqc_assert_colocated_extra_available "
        f"{shlex.quote(str(tmp_path))} viewer-gpu"
    )
    refused = _bash(snippet)
    assert refused.returncode != 0
    assert "no longer co-located" in refused.stderr
    assert "../vibe-view" in refused.stderr


@pytest.mark.parametrize(
    ("profile", "project_name", "install_suffix", "verification"),
    [
        ("basisopt", "vibe-basis", "vibe-basis", "import vibe_basis"),
    ],
)
def test_colocated_install_runs_pip_then_verification_and_propagates_failure(
    tmp_path: Path,
    profile: str,
    project_name: str,
    install_suffix: str,
    verification: str,
) -> None:
    repo = tmp_path / "checkout"
    project = repo / project_name
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    fake_python = tmp_path / "venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$COMPANION_CALL_LOG"\n'
        'if [ "$1" = "-c" ] && [ "${FAIL_COMPANION_VERIFY:-0}" = "1" ]; then\n'
        "    exit 42\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    log = tmp_path / "calls.log"
    snippet = (
        f". {shlex.quote(str(SETUP_HELPERS))}\n"
        "vibeqc_install_colocated_extra "
        f"{shlex.quote(str(fake_python.parents[1]))} "
        f"{shlex.quote(str(repo))} {shlex.quote(profile)}"
    )
    env = os.environ.copy()
    env["COMPANION_CALL_LOG"] = str(log)
    success = subprocess.run(
        ["/bin/bash", "-c", snippet],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert success.returncode == 0, success.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls == [
        f"-m pip install --upgrade -e {repo / install_suffix}",
        f"-c {verification}",
    ]

    log.unlink()
    env["FAIL_COMPANION_VERIFY"] = "1"
    failed = subprocess.run(
        ["/bin/bash", "-c", snippet],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 42
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2


def test_colocated_install_refuses_viewer_gpu_without_running_pip(
    tmp_path: Path,
) -> None:
    """viewer-gpu cannot be installed from this checkout since the split.

    The refusal must come before pip is reached, so the fake interpreter's
    call log stays empty.
    """
    repo = tmp_path / "checkout"
    project = repo / "vibe-view"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    fake_python = tmp_path / "venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$COMPANION_CALL_LOG\"\nexit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    log = tmp_path / "calls.log"
    log.write_text("", encoding="utf-8")
    snippet = (
        f". {shlex.quote(str(SETUP_HELPERS))}\n"
        "vibeqc_install_colocated_extra "
        f"{shlex.quote(str(fake_python.parents[1]))} "
        f"{shlex.quote(str(repo))} viewer-gpu"
    )
    env = os.environ.copy()
    env["COMPANION_CALL_LOG"] = str(log)
    refused = subprocess.run(
        ["/bin/bash", "-c", snippet],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "separate project" in refused.stderr
    assert "../vibe-view[viewer]" in refused.stderr
    assert log.read_text(encoding="utf-8") == ""


def test_component_preflight_runs_before_expensive_install_steps() -> None:
    install_source = INSTALL.read_text(encoding="utf-8")
    update_source = UPDATE.read_text(encoding="utf-8")
    assert install_source.index("vibeqc_assert_colocated_extra_available") < (
        install_source.index('echo "==> Running setup_native_deps.sh')
    )
    assert update_source.index("vibeqc_assert_colocated_extra_available") < (
        update_source.index('echo "==> Rebuilding native deps')
    )


def test_capture_updater_uses_canonical_transaction_and_never_uninstalls() -> None:
    source = CAPTURE_UPDATE.read_text(encoding="utf-8")
    assert "vibe_view_begin_venv_replacement" in source
    assert "vibe_view_commit_venv_replacement" in source
    assert "vibe_view_abort_venv_replacement" in source
    assert "capture-selftest" in source
    assert '"$PYTHON_BIN" "$ADOPT_LEGACY" capture' in source
    assert "pip uninstall" not in source
    assert source.index(
        'vibe_view_acquire_target_lock "$VENV_PATH" "capture update"'
    ) < source.index('vibe_view_begin_venv_replacement "$VENV_PATH"')
    assert source.rindex("vibe_view_release_target_lock") > source.index(
        'capture_write_marker "$VENV_PATH"'
    )


def test_capture_only_ownership_is_reproved_inside_replacement_start(
    tmp_path: Path, vibe_view_checkout: Path,
) -> None:
    target = tmp_path / "capture-only-environment"
    _recognisable_venv(target)
    marker = target / ".vibe-view-capture.json"
    marker.write_text("{}\n", encoding="utf-8")
    sentinel = target / "capture-owned-state"
    sentinel.write_text("preserve\n", encoding="utf-8")
    helper = shlex.quote(str(vibe_view_checkout / "scripts" / "_venv_helpers.sh"))
    script = f"""
set -euo pipefail
. {helper}
capture_assert_replaceable() {{
    [ -f "$1/.vibe-view-capture.json" ]
}}
vibe_view_begin_venv_replacement "$1" 1
trap 'vibe_view_abort_venv_replacement' EXIT
vibe_view_start_venv_replacement "$1" "$2" 0 capture
[ -f "$VIBE_VIEW_VENV_TX_BACKUP/capture-owned-state" ]
"""
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            script,
            "capture-transaction-test",
            str(target),
            str(Path(sys.executable).resolve()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    assert marker.is_file()


def test_capture_marker_inspection_ignores_pythonpath_sitecustomize(
    tmp_path: Path, vibe_view_checkout: Path,
) -> None:
    target = tmp_path / "capture-environment"
    _recognisable_venv(target)
    marker = target / ".vibe-view-capture.json"
    marker.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "vibe-view-capture-environment",
                "project": str(vibe_view_checkout.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    poison = tmp_path / "poison"
    poison.mkdir()
    sentinel = tmp_path / "sitecustomize-ran"
    (poison / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        'Path(os.environ["CAPTURE_SENTINEL"]).write_text("ran")\n',
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        CI="true",
        GITLAB_CI="true",
        PYTHONPATH=str(poison),
        CAPTURE_SENTINEL=str(sentinel),
    )
    result = subprocess.run(
        [
            "/bin/bash",
            str(CAPTURE_UPDATE),
            "--python",
            str(Path(sys.executable).resolve()),
            "--venv",
            str(target),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "capture-owned" in result.stdout
    assert not sentinel.exists()


def test_capture_update_contends_on_the_canonical_target_before_creation(
    tmp_path: Path, vibe_view_checkout: Path,
) -> None:
    target = tmp_path / "capture-environment"
    holder = _start_toolset_lock_holder(target, tmp_path / "other-checkout")
    env = dict(os.environ, CI="true", GITLAB_CI="true")
    try:
        result = subprocess.run(
            [
                "/bin/bash",
                str(CAPTURE_UPDATE),
                "--python",
                str(Path(sys.executable).resolve()),
                "--venv",
                str(target),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "active on this target" in result.stderr
        assert not target.exists()
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0


def test_capture_dry_run_accepts_activated_target_with_external_base_python(
    tmp_path: Path, vibe_view_checkout: Path,
) -> None:
    target = tmp_path / "activated-capture-environment"
    subprocess.run([sys.executable, "-m", "venv", "--copies", str(target)], check=True)
    (target / ".vibe-view-capture.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "vibe-view-capture-environment",
                "project": str(vibe_view_checkout.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        CI="true",
        GITLAB_CI="true",
        VIRTUAL_ENV=str(target),
        PATH=f"{target / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    )
    env.pop("PYTHON", None)
    result = subprocess.run(
        [
            "/bin/bash",
            str(CAPTURE_UPDATE),
            "--venv",
            str(target),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "capture-owned" in result.stdout


def test_capture_explicit_target_python_is_rejected_without_execution(
    tmp_path: Path, vibe_view_checkout: Path,
) -> None:
    target = tmp_path / "capture-environment"
    _recognisable_venv(target)
    (target / ".vibe-view-capture.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "vibe-view-capture-environment",
                "project": str(vibe_view_checkout.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sentinel = tmp_path / "target-python-ran"
    target_python = target / "bin" / "python"
    target_python.parent.mkdir()
    target_python.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(sentinel))}\nexit 0\n",
        encoding="utf-8",
    )
    target_python.chmod(0o755)
    env = dict(os.environ, CI="true", GITLAB_CI="true")
    result = subprocess.run(
        [
            "/bin/bash",
            str(CAPTURE_UPDATE),
            "--python",
            str(target_python),
            "--venv",
            str(target),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "external interpreter" in result.stderr
    assert not sentinel.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="capture lifecycle refuses root")
def test_capture_updater_requires_explicit_custom_legacy_adoption(
    tmp_path: Path, vibe_view_checkout: Path,
) -> None:
    target = tmp_path / "full-viewer-environment"
    subprocess.run([sys.executable, "-m", "venv", str(target)], check=True)
    project = vibe_view_checkout.resolve()
    marker = target / ".vibe-view-standalone.json"
    marker.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "vibe-view-standalone-venv",
                "project": str(project),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    refused = subprocess.run(
        [
            "bash",
            str(CAPTURE_UPDATE),
            "--python",
            str(Path(sys.executable).resolve()),
            "--venv",
            str(target),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "unmarked vibe-view environment" in refused.stderr
    assert target.is_dir()

    adopted = subprocess.run(
        [
            "bash",
            str(CAPTURE_UPDATE),
            "--python",
            str(Path(sys.executable).resolve()),
            "--venv",
            str(target),
            "--adopt-legacy",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert adopted.returncode == 0, adopted.stdout + adopted.stderr
    assert "legacy-adopted" in adopted.stdout
    assert target.is_dir()


def test_capture_xvfb_shim_delegates_to_the_next_path_entry(tmp_path: Path) -> None:
    source = CAPTURE_UPDATE.read_text(encoding="utf-8")
    shim_source = source.split("<<'SH'\n", 1)[1].split("\nSH\n", 1)[0]
    shim_dir = tmp_path / "capture bin"
    system_bin = tmp_path / "system bin"
    shim_dir.mkdir()
    system_bin.mkdir()
    shim = shim_dir / "xvfb-run"
    shim.write_text(shim_source, encoding="utf-8")
    shim.chmod(0o755)
    delegated = system_bin / "xvfb-run"
    delegated.write_text(
        "#!/bin/sh\nprintf 'delegated:%s\\n' \"$*\"\n", encoding="utf-8"
    )
    delegated.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{shim_dir}{os.pathsep}{system_bin}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        [str(shim), "-a", "/bin/echo", "ready"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "delegated:-a /bin/echo ready"
