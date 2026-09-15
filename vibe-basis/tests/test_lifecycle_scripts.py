"""Lifecycle contract for the standalone vibe-basis installation."""

from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
SCRIPTS = PACKAGE_ROOT / "scripts"
LIFECYCLE = ("install.sh", "update.sh", "reinstall.sh", "uninstall.sh")


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(SCRIPTS / script), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _owned_fake_venv(path: Path) -> None:
    (path / "bin").mkdir(parents=True)
    (path / "pyvenv.cfg").write_text(
        f"home = test\nexecutable = {sys.executable}\n", encoding="utf-8"
    )
    os.symlink(sys.executable, path / "bin" / "python")
    (path / ".vibe-basis-standalone").write_text(
        f"project={PACKAGE_ROOT}\nprofile=core\nwith_vq=0\npython={sys.executable}\n",
        encoding="utf-8",
    )


def test_lifecycle_scripts_are_executable_and_bash_syntax_clean():
    for name in LIFECYCLE:
        script = SCRIPTS / name
        assert script.stat().st_mode & stat.S_IXUSR
        result = subprocess.run(
            ["/bin/bash", "-n", str(script)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    helper = SCRIPTS / "_venv_helpers.sh"
    result = subprocess.run(
        ["/bin/bash", "-n", str(helper)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_every_lifecycle_help_path_works_without_an_installation():
    for name in LIFECYCLE:
        result = _run(name, "--help")
        assert result.returncode == 0, (name, result.stderr)
        assert "USAGE" in result.stdout


def test_reinstall_rejects_git_branch_selectors():
    for selector in ("--dev", "--release", "--branch=main", "--skip-git"):
        result = _run("reinstall.sh", selector)
        assert result.returncode != 0, selector
        assert "current checkout" in result.stderr


def test_value_options_reject_empty_and_option_like_values():
    cases = (
        ("install.sh", "--venv"),
        ("install.sh", "--python"),
        ("install.sh", "--extras"),
        ("update.sh", "--venv"),
        ("update.sh", "--branch"),
        ("uninstall.sh", "--venv"),
        ("uninstall.sh", "--python"),
    )
    for script, option in cases:
        empty = _run(script, option, "")
        assert empty.returncode != 0, (script, option)
        assert "non-empty" in empty.stderr
        option_like = _run(script, option, "--dry-run")
        assert option_like.returncode != 0, (script, option)
        assert "not option" in option_like.stderr


def test_update_rejects_conflicting_or_ignored_selectors():
    cases = (
        ("--dev", "--release"),
        ("--branch", "main", "--dev"),
        ("--skip-git", "--dev"),
        ("--dev", "--skip-git"),
        ("--with-vq", "--without-vq"),
    )
    for args in cases:
        result = _run("update.sh", *args)
        assert result.returncode != 0, args
        assert "conflict" in result.stderr


def test_uninstall_rejects_symlink_target_with_trailing_slash(tmp_path: Path):
    target = tmp_path / "owned"
    _owned_fake_venv(target)
    sentinel = target / "keep"
    sentinel.write_text("user data\n", encoding="utf-8")
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)

    result = _run(
        "uninstall.sh",
        "--python",
        sys.executable,
        "--venv",
        f"{link}/",
        "--dry-run",
    )
    assert result.returncode != 0
    assert "symlink virtualenv target" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "user data\n"


def test_update_dirty_preflight_ignores_untracked_custom_venvs():
    source = (SCRIPTS / "update.sh").read_text(encoding="utf-8")
    assert 'git -C "$VIBE_BASIS_REPO_ROOT" diff --quiet' in source
    assert 'git -C "$VIBE_BASIS_REPO_ROOT" diff --cached --quiet' in source
    assert "status --porcelain" not in source


def test_update_accepts_linked_git_worktree(tmp_path: Path):
    if shutil.which("git") is None:
        return
    source = tmp_path / "source"
    package = source / "vibe-basis"
    shutil.copytree(
        PACKAGE_ROOT,
        package,
        ignore=shutil.ignore_patterns(".venv", "__pycache__"),
    )
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Lifecycle Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "add", "vibe-basis"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-q", "-m", "fixture"], check=True
    )
    worktree = tmp_path / "linked-worktree"
    subprocess.run(
        ["git", "-C", str(source), "worktree", "add", "-q", str(worktree)],
        check=True,
    )
    linked_package = worktree / "vibe-basis"
    venv = linked_package / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
    os.symlink(sys.executable, venv / "bin" / "python")
    (venv / ".vibe-basis-standalone").write_text(
        f"project={linked_package}\nprofile=core\nwith_vq=0\npython={sys.executable}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "/bin/bash",
            str(linked_package / "scripts" / "update.sh"),
            "--dry-run",
            "--python",
            sys.executable,
            "--venv",
            str(venv),
        ],
        cwd=worktree,
        text=True,
        capture_output=True,
        check=False,
    )
    assert (worktree / ".git").is_file()
    assert result.returncode == 0, result.stdout + result.stderr


def test_install_dry_run_resolves_every_profile(tmp_path: Path):
    for profile in ("core", "standard", "optimizers", "all", "test"):
        result = _run(
            "install.sh",
            "--dry-run",
            "--python",
            sys.executable,
            "--venv",
            str(tmp_path / profile),
            "--extras",
            profile,
        )
        assert result.returncode == 0, (profile, result.stderr)
        assert "Dry-run complete" in result.stdout


def test_force_install_resolves_target_interpreter_to_external_base(tmp_path: Path):
    venv = tmp_path / "venv"
    _owned_fake_venv(venv)
    result = _run(
        "install.sh",
        "--force",
        "--dry-run",
        "--python",
        str(venv / "bin" / "python"),
        "--venv",
        str(venv),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"python:       {Path(sys.executable).resolve()}" in result.stdout


def test_install_records_stable_base_from_transient_environment(tmp_path: Path):
    transient_python = tmp_path / "transient" / "bin" / "python"
    transient_python.parent.mkdir(parents=True)
    transient_python.symlink_to(sys.executable)
    result = _run(
        "install.sh",
        "--dry-run",
        "--python",
        str(transient_python),
        "--venv",
        str(tmp_path / "new-venv"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"python:       {Path(sys.executable).resolve()}" in result.stdout


def test_update_and_uninstall_dry_runs_preserve_owned_environment(tmp_path: Path):
    venv = tmp_path / "venv"
    _owned_fake_venv(venv)

    update = _run(
        "update.sh",
        "--skip-git",
        "--dry-run",
        "--python",
        sys.executable,
        "--venv",
        str(venv),
    )
    assert update.returncode == 0, update.stderr

    uninstall = _run(
        "uninstall.sh",
        "--dry-run",
        "--python",
        sys.executable,
        "--venv",
        str(venv),
    )
    assert uninstall.returncode == 0, uninstall.stderr
    assert venv.is_dir()


def test_reinstall_and_uninstall_accept_a_broken_owned_environment(tmp_path: Path):
    venv = tmp_path / "broken"
    _owned_fake_venv(venv)
    (venv / "bin" / "python").unlink()
    sentinel = venv / "keep"
    sentinel.write_text("previous install\n", encoding="utf-8")

    reinstall = _run(
        "reinstall.sh",
        "--dry-run",
        "--python",
        sys.executable,
        "--venv",
        str(venv),
    )
    assert reinstall.returncode == 0, reinstall.stdout + reinstall.stderr

    uninstall = _run(
        "uninstall.sh",
        "--dry-run",
        "--python",
        sys.executable,
        "--venv",
        str(venv),
    )
    assert uninstall.returncode == 0, uninstall.stdout + uninstall.stderr
    assert sentinel.read_text(encoding="utf-8") == "previous install\n"


def test_update_uses_recorded_python_when_path_python_is_old(tmp_path: Path):
    venv = tmp_path / "venv"
    _owned_fake_venv(venv)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "source = sys.stdin.read()\n"
        "if 'sys.version_info <' in source:\n"
        "    raise SystemExit(88)\n"
        "sys.argv = sys.argv[1:]\n"
        "exec(compile(source, '<stdin>', 'exec'))\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "/bin/bash",
            str(SCRIPTS / "update.sh"),
            "--skip-git",
            "--dry-run",
            "--venv",
            str(venv),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"python:       {Path(sys._base_executable).resolve()}" in result.stdout


def test_uninstall_refuses_an_unowned_venv(tmp_path: Path):
    venv = tmp_path / "not-ours"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
    os.symlink(sys.executable, venv / "bin" / "python")

    result = _run(
        "uninstall.sh",
        "--dry-run",
        "--python",
        sys.executable,
        "--venv",
        str(venv),
    )
    assert result.returncode != 0
    assert "refusing unowned virtualenv" in result.stderr
    assert venv.is_dir()


def test_force_install_does_not_execute_unowned_target_python(tmp_path: Path):
    venv = tmp_path / "foreign"
    (venv / "bin").mkdir(parents=True)
    sentinel = tmp_path / "must-not-run"
    malicious = venv / "bin" / "python"
    malicious.write_text(
        f"#!/bin/sh\nprintf 'ran\\n' >{shlex.quote(str(sentinel))}\nexit 91\n",
        encoding="utf-8",
    )
    malicious.chmod(0o755)
    (venv / "pyvenv.cfg").write_text(
        f"home = foreign\nexecutable = {malicious}\n", encoding="utf-8"
    )

    result = _run(
        "install.sh",
        "--force",
        "--dry-run",
        "--python",
        str(malicious),
        "--venv",
        str(venv),
    )
    assert result.returncode != 0
    assert "refusing unowned virtualenv" in result.stderr
    assert not sentinel.exists()


def test_failed_replacement_restores_previous_environment(tmp_path: Path):
    venv = tmp_path / "venv"
    _owned_fake_venv(venv)
    sentinel = venv / "previous-install"
    sentinel.write_text("keep me\n", encoding="utf-8")

    helper = shlex.quote(str(SCRIPTS / "_venv_helpers.sh"))
    target = shlex.quote(str(venv))
    partial = shlex.quote(str(venv / "new-install"))
    program = f"""
set -euo pipefail
. {helper}
vibe_basis_acquire_target_lock {target} {shlex.quote(sys.executable)}
trap 'vibe_basis_abort_replacement; vibe_basis_release_target_lock' EXIT
vibe_basis_begin_replacement {target}
mkdir -p {target}
printf 'partial\n' > {partial}
exit 42
"""
    result = subprocess.run(
        ["/bin/bash", "-c", program], text=True, capture_output=True, check=False
    )
    assert result.returncode == 42
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert not (venv / "new-install").exists()
    assert not list(tmp_path.glob("venv.vibe-basis-backup.*"))


def test_lifecycle_lock_rejects_contention_and_recovers_after_exit(tmp_path: Path):
    target = tmp_path / "venv"
    ready = tmp_path / "ready"
    helper = shlex.quote(str(SCRIPTS / "_venv_helpers.sh"))
    quoted_target = shlex.quote(str(target))
    quoted_python = shlex.quote(sys.executable)
    quoted_ready = shlex.quote(str(ready))
    holder_program = f"""
set -euo pipefail
. {helper}
vibe_basis_acquire_target_lock {quoted_target} {quoted_python}
trap 'vibe_basis_release_target_lock' EXIT
trap 'exit 0' TERM INT
printf 'ready\n' > {quoted_ready}
while :; do sleep 1; done
"""
    holder = subprocess.Popen(
        ["/bin/bash", "-c", holder_program],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(100):
            if ready.exists():
                break
            if holder.poll() is not None:
                break
            time.sleep(0.02)
        assert ready.exists(), f"lock holder exited with {holder.poll()}"
        contender_program = (
            f"set -euo pipefail; . {helper}; "
            f"vibe_basis_acquire_target_lock {quoted_target} {quoted_python}"
        )
        contender = subprocess.run(
            ["/bin/bash", "-c", contender_program],
            text=True,
            capture_output=True,
            check=False,
        )
        assert contender.returncode != 0
        assert "toolset lifecycle operation already active" in contender.stderr
    finally:
        holder.terminate()
        holder.wait(timeout=5)

    retry_program = (
        f"set -euo pipefail; . {helper}; "
        f"vibe_basis_acquire_target_lock {quoted_target} {quoted_python}; "
        "vibe_basis_release_target_lock"
    )
    retry = subprocess.run(
        ["/bin/bash", "-c", retry_program],
        text=True,
        capture_output=True,
        check=False,
    )
    assert retry.returncode == 0, retry.stdout + retry.stderr


def test_lifecycle_lock_survives_exec_handoff(tmp_path: Path):
    helper = shlex.quote(str(SCRIPTS / "_venv_helpers.sh"))
    target = shlex.quote(str(tmp_path / "venv"))
    python = shlex.quote(sys.executable)
    child_program = (
        f"set -euo pipefail; . {helper}; "
        '[ "$VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE" = 1 ]; '
        "trap 'vibe_basis_release_target_lock' EXIT; "
        "printf 'ready\\n'; IFS= read -r _"
    )
    outer_program = (
        f"set -euo pipefail; . {helper}; "
        f"vibe_basis_acquire_target_lock {target} {python}; "
        "trap 'vibe_basis_release_target_lock' EXIT; "
        "vibe_basis_prepare_target_lock_handoff; "
        f"exec /bin/bash -c {shlex.quote(child_program)}"
    )
    holder = subprocess.Popen(
        ["/bin/bash", "-c", outer_program],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline() == "ready\n"
    try:
        contender = subprocess.run(
            [
                "/bin/bash",
                "-c",
                (
                    f"set -euo pipefail; . {helper}; "
                    f"vibe_basis_acquire_target_lock {target} {python}"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert contender.returncode != 0
        assert "lifecycle operation already active" in contender.stderr
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0


def test_bundled_vq_source_marker_is_written_and_verified(tmp_path: Path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    marker = tmp_path / "source-sha"
    fake_vq = venv / "bin" / "vq"
    fake_vq.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        f"marker={shlex.quote(str(marker))}\n"
        'if [ "${1-}" = source-sha ] && [ "${2-}" = --write-marker ]; then\n'
        '    printf \'%s\\n\' "$3" >"$marker"\n'
        'elif [ "${1-}" = source-sha ]; then\n'
        '    cat "$marker"\n'
        "else\n"
        "    exit 2\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_vq.chmod(0o755)
    helper = shlex.quote(str(SCRIPTS / "_venv_helpers.sh"))
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f"set -euo pipefail; . {helper}; "
                f"vibe_basis_stamp_vq_source_marker {shlex.quote(str(venv))}"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    expected = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_text(encoding="utf-8").strip() == expected


def test_bundled_vq_source_marker_failure_is_fatal(tmp_path: Path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    fake_vq = venv / "bin" / "vq"
    fake_vq.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
    fake_vq.chmod(0o755)
    helper = shlex.quote(str(SCRIPTS / "_venv_helpers.sh"))
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f"set -euo pipefail; . {helper}; "
                f"vibe_basis_stamp_vq_source_marker {shlex.quote(str(venv))}"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "could not write" in result.stderr


def test_install_refuses_repository_as_target():
    result = _run(
        "install.sh",
        "--dry-run",
        "--python",
        sys.executable,
        "--venv",
        str(REPO_ROOT),
        "--force",
    )
    assert result.returncode != 0
    assert "source tree" in result.stderr or "repository" in result.stderr


def test_install_refuses_git_metadata_as_target():
    result = _run(
        "install.sh",
        "--dry-run",
        "--python",
        sys.executable,
        "--venv",
        str(REPO_ROOT / ".git" / "vibe-basis-venv"),
    )
    assert result.returncode != 0
    assert "Git metadata" in result.stderr


@pytest.mark.parametrize(
    "target",
    (
        "/etc/hosts",
        "/etc/vibe-basis-environment-that-does-not-exist",
        "/boot/vibe-basis-environment-that-does-not-exist",
        "/nix/vibe-basis-environment-that-does-not-exist",
        "/snap/vibe-basis-environment-that-does-not-exist",
        "/usr/local/share/vibe-basis-environment-that-does-not-exist",
        "/private/etc/vibe-basis-environment-that-does-not-exist",
    ),
)
def test_protected_system_descendants_are_rejected(target: str):
    helper = shlex.quote(str(SCRIPTS / "_venv_helpers.sh"))
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            f". {helper}; vibe_basis_assert_safe_target {shlex.quote(target)}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "protected system location" in result.stderr


def test_foreign_git_metadata_descendants_are_rejected_existing_or_missing(
    tmp_path: Path,
):
    helper = shlex.quote(str(SCRIPTS / "_venv_helpers.sh"))
    metadata = tmp_path / "other-repository" / ".git"
    existing = metadata / "existing-environment"
    existing.mkdir(parents=True)
    missing = metadata / "missing-environment"

    for target in (existing, missing):
        result = subprocess.run(
            [
                "/bin/bash",
                "-c",
                f". {helper}; vibe_basis_assert_safe_target {shlex.quote(str(target))}",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "Git metadata" in result.stderr


def test_temp_user_and_xdg_descendants_remain_allowed(tmp_path: Path):
    helper = shlex.quote(str(SCRIPTS / "_venv_helpers.sh"))
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
                f". {helper}; vibe_basis_assert_safe_target {shlex.quote(str(target))}",
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_poisoned_home_and_xdg_cannot_exempt_a_system_location():
    helper = shlex.quote(str(SCRIPTS / "_venv_helpers.sh"))
    poisoned_home = Path("/etc").resolve()
    target = poisoned_home / "vibe-basis-environment-that-does-not-exist"
    env = dict(os.environ, HOME=str(poisoned_home), XDG_DATA_HOME=str(poisoned_home))
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            f". {helper}; vibe_basis_assert_safe_target {shlex.quote(str(target))}",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "protected system location" in result.stderr


def test_home_owned_by_another_uid_cannot_exempt_its_descendants(tmp_path: Path):
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
    target = home / f"vibe-basis-other-owner-probe-{os.getpid()}"
    env = dict(
        os.environ,
        HOME=str(home),
        PATH=f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        FAKE_STAT_UID=str(os.geteuid() + 1),
    )
    helper = shlex.quote(str(SCRIPTS / "_venv_helpers.sh"))
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            f". {helper}; vibe_basis_assert_safe_target {shlex.quote(str(target))}",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "protected system location" in result.stderr


def test_target_lock_contends_across_different_checkouts(tmp_path: Path):
    checkout_a = tmp_path / "checkout-a" / "vibe-basis" / "scripts"
    checkout_b = tmp_path / "checkout-b" / "vibe-basis" / "scripts"
    checkout_a.mkdir(parents=True)
    checkout_b.mkdir(parents=True)
    helper_a = checkout_a / "_venv_helpers.sh"
    helper_b = checkout_b / "_venv_helpers.sh"
    shutil.copy2(SCRIPTS / "_venv_helpers.sh", helper_a)
    shutil.copy2(SCRIPTS / "_venv_helpers.sh", helper_b)
    for checkout in (checkout_a.parents[1], checkout_b.parents[1]):
        shared_scripts = checkout / "scripts"
        shared_scripts.mkdir()
        shutil.copy2(
            REPO_ROOT / "scripts" / "_lifecycle_lock.sh",
            shared_scripts / "_lifecycle_lock.sh",
        )
    target = tmp_path / "shared-environment"
    target_arg = shlex.quote(str(target))
    holder = subprocess.Popen(
        [
            "/bin/bash",
            "-c",
            (
                f". {shlex.quote(str(helper_a))}; "
                f"vibe_basis_acquire_target_lock {target_arg} {shlex.quote(sys.executable)}; "
                "trap 'vibe_basis_release_target_lock' EXIT; "
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
    contender = (
        f". {shlex.quote(str(helper_b))}; "
        f"if vibe_basis_acquire_target_lock {target_arg} {shlex.quote(sys.executable)}; then "
        "vibe_basis_release_target_lock; exit 0; fi; exit 1"
    )

    try:
        blocked = subprocess.run(
            ["/bin/bash", "-c", contender],
            text=True,
            capture_output=True,
            check=False,
        )
        assert blocked.returncode != 0
        assert "lifecycle operation already active" in blocked.stderr
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0

    acquired = subprocess.run(
        ["/bin/bash", "-c", contender],
        text=True,
        capture_output=True,
        check=False,
    )
    assert acquired.returncode == 0, acquired.stderr


def test_external_python_resolution_ignores_pythonpath_sitecustomize(tmp_path: Path):
    injected = tmp_path / "injected"
    injected.mkdir()
    executed = tmp_path / "sitecustomize-ran"
    (injected / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(executed)!r}).touch()\n",
        encoding="utf-8",
    )
    target = tmp_path / "missing-environment"
    helper = shlex.quote(str(SCRIPTS / "_venv_helpers.sh"))
    env = dict(os.environ, PYTHONPATH=str(injected))

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f". {helper}; vibe_basis_resolve_base_python "
                f"{shlex.quote(sys.executable)} {shlex.quote(str(target))}"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not executed.exists()


def test_optional_dependency_contract_is_resolvable_without_published_vq():
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as stream:
        extras = tomllib.load(stream)["project"]["optional-dependencies"]
    assert extras["vq"] == []
    assert not any(requirement.startswith("vq") for requirement in extras["all"])
    assert any(requirement.startswith("scipy") for requirement in extras["test"])
