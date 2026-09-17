"""Cross-component contract for the suite's toolset lifecycle scripts.

Only vibe-basis is still co-located. The 2026-09 split moved vibe-view and
vibe-queue into sibling repositories, so their script directories are resolved
through :mod:`tests.companion_paths` -- which asks the product's own
``scripts/_companion_paths.sh`` -- and the parametrisations that need them skip
loudly when the sibling is not checked out. Nothing here joins a companion
onto ``REPO_ROOT``: that path cannot exist.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tests import companion_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
# Component key -> companion name for the resolver, or None when the component
# is this repository itself.
COMPONENT_COMPANIONS = {
    "vibe-qc": None,
    "vibe-view": "vibe-view",
    "vq": "vibe-queue",
    "vibe-basis": "vibe-basis",
}
COMPONENTS = tuple(COMPONENT_COMPANIONS)
OPERATIONS = ("install", "update", "reinstall", "uninstall")
VIBEQC_HELPERS = (
    "_lifecycle_lock.sh",
    "_build_lock.sh",
    "_native_stamp.sh",
    "_safe_build_env.sh",
    "_setup_helpers.sh",
    "_venv_helpers.sh",
    "_vq_cooperation.sh",
    "_companion_paths.sh",
)
LIFECYCLE_HELPERS = tuple(
    [("vibe-qc", name) for name in VIBEQC_HELPERS]
    + [(component, "_venv_helpers.sh") for component in COMPONENTS[1:]]
)
COMPANION_RESOLVER = REPO_ROOT / "scripts" / "_companion_paths.sh"
# Each component's documented default environment, as its own install help
# states it, relative to that component's own checkout.
DOCUMENTED_DEFAULT_VENVS = {
    "vibe-qc": ".venv",
    "vibe-view": ".venv",
    "vq": "vibe-queue/.venv",
    "vibe-basis": "vibe-basis/.venv",
}


def component_root(component: str, coverage: str) -> Path:
    """The checkout that owns ``component``; skips when it is not present."""
    companion = COMPONENT_COMPANIONS[component]
    if companion is None:
        return REPO_ROOT
    return companion_paths.require(companion, coverage)


def component_script_dir(component: str, coverage: str) -> Path:
    return component_root(component, coverage) / "scripts"


def present_component_roots() -> dict[str, Path]:
    """Every component this checkout can actually see, vibe-qc included."""
    roots = {"vibe-qc": REPO_ROOT}
    for component, companion in COMPONENT_COMPANIONS.items():
        if companion is None:
            continue
        root = companion_paths.companion_root(companion)
        if root is not None:
            roots[component] = root
    return roots


def _bash(
    snippet: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-c", snippet],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_component_has_executable_syntax_clean_lifecycle_entry(
    component: str, operation: str
) -> None:
    script = (
        component_script_dir(
            component, f"{component} {operation} entry-point contract"
        )
        / f"{operation}.sh"
    )
    assert script.is_file(), f"missing {component} {operation} entry point"
    assert os.access(script, os.X_OK), f"not executable: {script}"
    result = subprocess.run(
        ["/bin/bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_lifecycle_help_is_environment_independent(
    component: str, operation: str
) -> None:
    script = (
        component_script_dir(component, f"{component} {operation} help contract")
        / f"{operation}.sh"
    )
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
    ("component", "helper"),
    LIFECYCLE_HELPERS,
    ids=lambda value: str(value),
)
def test_sourced_lifecycle_helpers_are_bash_syntax_clean(
    component: str, helper: str
) -> None:
    script = (
        component_script_dir(
            component, f"{component} lifecycle-helper syntax contract"
        )
        / helper
    )
    result = subprocess.run(
        ["/bin/bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_documented_default_environments_are_distinct() -> None:
    """No two components document a default environment at the same path.

    Before the split this compared relative strings, because every component
    shared one tree and a shared string meant a shared directory. Each has its
    own checkout now -- vibe-qc and vibe-view both document ``.venv`` -- so the
    collision that matters is between the resolved absolute paths.
    """
    resolved: dict[str, Path] = {}
    for component, root in present_component_roots().items():
        default = DOCUMENTED_DEFAULT_VENVS[component]
        help_text = subprocess.run(
            ["/bin/bash", str(root / "scripts" / "install.sh"), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert default in help_text, component
        resolved[component] = Path(os.path.normpath(root / default))
    # vibe-qc and vibe-basis are always present, so this never degenerates to
    # a single-entry comparison however few companions are checked out.
    assert {"vibe-qc", "vibe-basis"} <= set(resolved)
    assert len(set(resolved.values())) == len(resolved), resolved


def test_mac_and_linux_portability_contract_avoids_bash4_maps() -> None:
    scripts: list[Path] = []
    for component, root in present_component_roots().items():
        scripts.extend(
            root / "scripts" / helper
            for helper_component, helper in LIFECYCLE_HELPERS
            if helper_component == component
        )
        scripts.extend(
            root / "scripts" / f"{operation}.sh" for operation in OPERATIONS
        )
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        assert "declare -A" not in source, script
        assert "#!/usr/bin/env zsh" not in source, script


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
    ("component", "lifecycle_call"),
    (
        ("vibe-basis", "vibe_basis_acquire_target_lock"),
        ("vq", "vq_acquire_lifecycle_lock"),
    ),
)
def test_git_updaters_take_lifecycle_then_native_build_lock(
    component: str, lifecycle_call: str
) -> None:
    script = (
        component_script_dir(
            component, f"{component} updater lock-ordering contract"
        )
        / "update.sh"
    )
    source = script.read_text(encoding="utf-8")
    lifecycle = source.index(lifecycle_call)
    build = source.index("vibeqc_acquire_build_lock", lifecycle)
    fetch = source.index(" fetch ", build)
    assert lifecycle < build < fetch


# --- the companion resolver itself ------------------------------------------
# These run on every checkout, companion present or not: they are the product's
# own answer to "where does a companion live", which this repository owns
# outright, so none of them may skip.


def _unoverridden_env() -> dict[str, str]:
    """The ambient environment with every companion override removed.

    These tests pin the *default* layout, so an override exported by whoever
    launched pytest -- a gate that names its companion checkouts, say -- must
    not decide the answer.
    """
    env = os.environ.copy()
    for variable in companion_paths.OVERRIDE_VARS.values():
        env.pop(variable, None)
    return env


def _resolver_probe(
    checkout: Path, component: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = _unoverridden_env() if env is None else env
    snippet = (
        f'. {shlex.quote(str(checkout / "scripts" / "_companion_paths.sh"))}\n'
        f"if vibeqc_companion_root {shlex.quote(component)} root; "
        "then present=1; else present=0; fi\n"
        'printf "%s %s\\n" "$present" "$root"\n'
    )
    return _bash(snippet, env=env)


def _fabricate_checkout(tmp_path: Path) -> Path:
    """A throwaway checkout carrying only the resolver, for layout tests."""
    checkout = tmp_path / "suite" / "vibe-qc"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "scripts" / "_companion_paths.sh").write_text(
        COMPANION_RESOLVER.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return checkout


def test_companion_resolver_places_split_components_beside_the_checkout(
    tmp_path: Path,
) -> None:
    checkout = _fabricate_checkout(tmp_path)
    sibling = checkout.parent / "vibe-view"
    sibling.mkdir()

    result = _resolver_probe(checkout, "vibe-view")
    assert result.returncode == 0, result.stderr
    present, root = result.stdout.split()
    assert present == "1"
    assert Path(root) == sibling.resolve()
    assert not root.startswith(str(checkout.resolve()) + os.sep)


def test_companion_resolver_keeps_vibe_basis_in_tree(tmp_path: Path) -> None:
    checkout = _fabricate_checkout(tmp_path)
    (checkout / "vibe-basis").mkdir()

    result = _resolver_probe(checkout, "vibe-basis")
    assert result.returncode == 0, result.stderr
    present, root = result.stdout.split()
    assert present == "1"
    assert Path(root) == (checkout / "vibe-basis").resolve()


def test_companion_resolver_reports_an_absent_sibling_without_failing(
    tmp_path: Path,
) -> None:
    checkout = _fabricate_checkout(tmp_path)

    result = _resolver_probe(checkout, "vibe-view")
    present, root = result.stdout.split()
    assert present == "0"
    assert Path(root) == checkout.parent / "vibe-view"
    assert ".." not in root


def test_companion_resolver_honours_the_override_variable(tmp_path: Path) -> None:
    checkout = _fabricate_checkout(tmp_path)
    elsewhere = tmp_path / "somewhere-else" / "vibe-view"
    elsewhere.mkdir(parents=True)
    env = _unoverridden_env()
    env["VIBE_VIEW_ROOT"] = str(elsewhere)

    result = _resolver_probe(checkout, "vibe-view", env=env)
    present, root = result.stdout.split()
    assert present == "1"
    assert Path(root) == elsewhere.resolve()


def test_companion_resolver_refuses_an_unknown_component(tmp_path: Path) -> None:
    checkout = _fabricate_checkout(tmp_path)

    result = _resolver_probe(checkout, "agentic-loop")
    assert result.stdout.split()[0] == "0"
    assert "unknown companion component 'agentic-loop'" in result.stderr


def test_no_test_joins_a_split_component_onto_the_repository_root() -> None:
    """#320: the join that made 35 tests red, kept out by construction."""
    dead_joins = (
        '"vibe-view"',
        '"vibe-queue"',
        "'vibe-view'",
        "'vibe-queue'",
        "'agentic-loop/",
        '"agentic-loop/',
    )
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "tests").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for dead in dead_joins:
            for prefix in ("REPO_ROOT / ", "parents[1] / "):
                if f"{prefix}{dead}" in source:
                    offenders.append(f"{path.name}: {prefix}{dead}")
    assert not offenders, offenders
