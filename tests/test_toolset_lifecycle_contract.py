"""Cross-component contract for the co-located toolset lifecycle scripts."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_SCRIPT_DIRS = {
    "vibe-qc": REPO_ROOT / "scripts",
    "vibe-view": REPO_ROOT / "vibe-view" / "scripts",
    "vq": REPO_ROOT / "vibe-queue" / "scripts",
    "vibe-basis": REPO_ROOT / "vibe-basis" / "scripts",
}
OPERATIONS = ("install", "update", "reinstall", "uninstall")
LIFECYCLE_HELPERS = (
    REPO_ROOT / "scripts" / "_lifecycle_lock.sh",
    REPO_ROOT / "scripts" / "_build_lock.sh",
    REPO_ROOT / "scripts" / "_native_stamp.sh",
    REPO_ROOT / "scripts" / "_safe_build_env.sh",
    REPO_ROOT / "scripts" / "_setup_helpers.sh",
    REPO_ROOT / "scripts" / "_venv_helpers.sh",
    REPO_ROOT / "scripts" / "_vq_cooperation.sh",
    REPO_ROOT / "vibe-view" / "scripts" / "_venv_helpers.sh",
    REPO_ROOT / "vibe-queue" / "scripts" / "_venv_helpers.sh",
    REPO_ROOT / "vibe-basis" / "scripts" / "_venv_helpers.sh",
)


@pytest.mark.parametrize("component", COMPONENT_SCRIPT_DIRS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_component_has_executable_syntax_clean_lifecycle_entry(
    component: str, operation: str
) -> None:
    script = COMPONENT_SCRIPT_DIRS[component] / f"{operation}.sh"
    assert script.is_file(), f"missing {component} {operation} entry point"
    assert os.access(script, os.X_OK), f"not executable: {script}"
    result = subprocess.run(
        ["/bin/bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("component", COMPONENT_SCRIPT_DIRS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_lifecycle_help_is_environment_independent(
    component: str, operation: str
) -> None:
    script = COMPONENT_SCRIPT_DIRS[component] / f"{operation}.sh"
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    result = subprocess.run(
        ["/bin/bash", str(script), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "USAGE" in result.stdout
    assert "install" in result.stdout.lower()


@pytest.mark.parametrize(
    "helper",
    LIFECYCLE_HELPERS,
    ids=lambda path: str(path.parent.name + "/" + path.name),
)
def test_sourced_lifecycle_helpers_are_bash_syntax_clean(helper: Path) -> None:
    result = subprocess.run(
        ["/bin/bash", "-n", str(helper)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_documented_default_environments_are_distinct() -> None:
    expected = {
        "vibe-qc": ".venv",
        "vibe-view": "vibe-view/.venv",
        "vq": "vibe-queue/.venv",
        "vibe-basis": "vibe-basis/.venv",
    }
    for component, default in expected.items():
        help_text = subprocess.run(
            [
                "/bin/bash",
                str(COMPONENT_SCRIPT_DIRS[component] / "install.sh"),
                "--help",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert default in help_text


def test_mac_and_linux_portability_contract_avoids_bash4_maps() -> None:
    scripts = list(LIFECYCLE_HELPERS)
    scripts.extend(
        script_dir / f"{operation}.sh"
        for script_dir in COMPONENT_SCRIPT_DIRS.values()
        for operation in OPERATIONS
    )
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        assert "declare -A" not in source
        assert "#!/usr/bin/env zsh" not in source


def test_shared_lock_accepts_nested_missing_target_parents(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "nested" / "environment"
    helper = shlex.quote(str(REPO_ROOT / "scripts" / "_lifecycle_lock.sh"))
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f". {helper}; vibe_toolset_acquire_lifecycle_lock "
                f"{shlex.quote(str(Path(sys.executable).resolve()))} "
                f"{shlex.quote(str(REPO_ROOT))} {shlex.quote(str(target))} test; "
                "vibe_toolset_release_lifecycle_lock"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not target.parent.exists()


@pytest.mark.parametrize(
    ("script", "lifecycle_call"),
    (
        (
            REPO_ROOT / "vibe-basis" / "scripts" / "update.sh",
            "vibe_basis_acquire_target_lock",
        ),
        (
            REPO_ROOT / "vibe-queue" / "scripts" / "update.sh",
            "vq_acquire_lifecycle_lock",
        ),
    ),
)
def test_git_updaters_take_lifecycle_then_native_build_lock(
    script: Path, lifecycle_call: str
) -> None:
    source = script.read_text(encoding="utf-8")
    lifecycle = source.index(lifecycle_call)
    build = source.index("vibeqc_acquire_build_lock", lifecycle)
    fetch = source.index(" fetch ", build)
    assert lifecycle < build < fetch
