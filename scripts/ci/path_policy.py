#!/usr/bin/env python3
"""Classify repository changes into the CI jobs they require.

GitLab decides whether to create a job before that job can run, so the
``rules:changes`` lists in ``.gitlab-ci.yml`` remain the execution surface.
This module is the testable policy surface: its pattern lists are compared
byte-for-byte with named YAML anchors by :func:`verify_gitlab_policy`.

CI policy (maintainer decision, 2026-07-26): CI runs once per release, on
the exact release tree. An ordinary ``main`` push selects **no** jobs — the
``main`` context below selects nothing, and the ``workflow:`` rules in
``.gitlab-ci.yml`` refuse to create the pipeline at all — except for the
decoupled ``website/**`` lane, which does carry one test gate:
``website-test``, which both website deploys ``needs:``. That lane
publishes to the public root on every matching push, so it cannot wait for
release-candidate evidence. Test evidence for day-to-day commits is produced
locally before landing; release evidence comes from the single
release-candidate pipeline, which runs every component gate. The immutable
tag reuses that exact-SHA evidence and creates no duplicate test pipeline.

The path-gated selection logic survives for the ``merge_request`` context.
Any shared build/test surface must still be added to ``NATIVE_PATHS`` in
the same commit that introduces it.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]

NATIVE_PATHS: Final = (
    ".gitlab-ci.yml",
    "CMakeLists.txt",
    "MANIFEST.in",
    "pyproject.toml",
    "cpp/**/*",
    "python/vibeqc/**/*",
    "python/vibeqc_naming/**/*",
    "tests/**/*",
    "scripts/_build_lock.sh",
    "scripts/_lifecycle_lock.sh",
    "scripts/_safe_build_env.sh",
    "scripts/_setup_helpers.sh",
    "scripts/_venv_helpers.sh",
    "scripts/build_*.sh",
    "scripts/gen_qvf_v2_schema.py",
    "scripts/install.sh",
    "scripts/setup_basis_library.sh",
    "scripts/setup_native_deps.sh",
    "scripts/test_gate/**/*",
    "scripts/update.sh",
    "scripts/update_native_deps.sh",
)

VIBE_BASIS_PATHS: Final = (
    ".gitlab-ci.yml",
    "pyproject.toml",
    "scripts/_build_lock.sh",
    "scripts/_lifecycle_lock.sh",
    "vibe-basis/**/*",
)

WEBSITE_PATHS: Final = ("website/**/*",)

DOCS_PATHS: Final = (
    ".gitlab-ci.yml",
    "CHANGELOG.md",
    "changelog.d/**/*",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "README.md",
    "docs/**/*",
    "scripts/build_site.sh",
)

# vq_paths and vibe_view_paths are gone: those components are separate
# repositories with their own pipelines. vibe-basis stays in this one.
ANCHOR_POLICIES: Final = {
    "native_paths": NATIVE_PATHS,
    "vibe_basis_paths": VIBE_BASIS_PATHS,
    "docs_paths": DOCS_PATHS,
    "website_paths": WEBSITE_PATHS,
}

ALWAYS_JOBS: Final = ("ci-path-policy",)
RELEASE_JOBS: Final = (
    "build-test",
    "ci-path-policy",
    "docs-build",
    "qvf-conformance",
    "vibe-basis-test",
)
NATIVE_CACHE_KEY: Final = (
    "vibeqc-core-debian12-gcc12-python313-cp313-linux-x86_64-release-v1"
)


def _matches(path: str, pattern: str) -> bool:
    """Match the GitLab patterns used by this policy.

    ``/**/*`` means every file below the named directory, including files
    directly in it. The remaining patterns use shell-style ``*`` only.
    """

    normalized = path.removeprefix("./")
    if pattern.endswith("/**/*"):
        return normalized.startswith(pattern[:-4])
    return fnmatch.fnmatchcase(normalized, pattern)


def matches_any(paths: Iterable[str], patterns: Sequence[str]) -> bool:
    return any(_matches(path, pattern) for path in paths for pattern in patterns)


def select_jobs(paths: Sequence[str], *, context: str = "main") -> tuple[str, ...]:
    """Return the sorted CI jobs required for ``paths`` and pipeline context."""

    if context == "release_candidate":
        return RELEASE_JOBS
    if context == "tag":
        return ()
    if context == "release":
        return ("ci-path-policy", "docs-build")
    if context == "web":
        return ("ci-path-policy", "docs-build")
    if context == "main":
        # Ordinary main commits run no CI test jobs: evidence is local
        # (scripts/test_gate/ + component suites) and the release-gate
        # pipeline. The website lane is the single exception, because it
        # publishes on the push rather than on the release: its deploys
        # gate on website-test, so that job is the one main selection.
        if matches_any(paths, WEBSITE_PATHS):
            return ("website-test",)
        return ()

    jobs = set(ALWAYS_JOBS)
    if matches_any(paths, NATIVE_PATHS):
        jobs.add("build-test")
    if matches_any(paths, VIBE_BASIS_PATHS):
        jobs.add("vibe-basis-test")
    if matches_any(paths, DOCS_PATHS):
        jobs.add("docs-build")
    return tuple(sorted(jobs))


def _yaml_anchor_paths(text: str, anchor: str) -> tuple[str, ...]:
    lines = text.splitlines()
    marker = re.compile(rf"^(?P<indent>\s*)changes:\s*&{re.escape(anchor)}\s*$")
    for index, line in enumerate(lines):
        match = marker.match(line)
        if match is None:
            continue
        base_indent = len(match.group("indent"))
        paths: list[str] = []
        for candidate in lines[index + 1 :]:
            if not candidate.strip():
                continue
            indent = len(candidate) - len(candidate.lstrip())
            if indent <= base_indent:
                break
            item = re.match(r'^\s*-\s+"([^"]+)"\s*$', candidate)
            if item is None:
                raise ValueError(
                    f"unsupported entry below &{anchor}: {candidate.strip()!r}"
                )
            paths.append(item.group(1))
        return tuple(paths)
    raise ValueError(f".gitlab-ci.yml has no changes anchor &{anchor}")


def verify_gitlab_policy(path: Path | None = None) -> None:
    """Fail if GitLab's real job-selection lists drift from this module."""

    ci_path = path or REPO_ROOT / ".gitlab-ci.yml"
    text = ci_path.read_text()
    errors: list[str] = []
    for anchor, expected in ANCHOR_POLICIES.items():
        actual = _yaml_anchor_paths(text, anchor)
        if actual != expected:
            errors.append(
                f"&{anchor} differs:\n"
                f"  policy={list(expected)!r}\n"
                f"  gitlab={list(actual)!r}"
            )
    if errors:
        raise ValueError("\n".join(errors))


def verify_build_reuse_policy(root: Path | None = None) -> None:
    """Guard the ABI-scoped cache and the clean release-build exception."""

    repo_root = root or REPO_ROOT
    ci_text = (repo_root / ".gitlab-ci.yml").read_text()
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    cmake_text = (repo_root / "CMakeLists.txt").read_text()
    venv_helper_text = (repo_root / "scripts" / "_venv_helpers.sh").read_text()
    errors: list[str] = []

    if NATIVE_CACHE_KEY not in ci_text:
        errors.append(f"native cache key is not the reviewed value: {NATIVE_CACHE_KEY}")
    if pyproject.get("tool", {}).get("scikit-build", {}).get("build-dir") != (
        "build/{wheel_tag}"
    ):
        errors.append("scikit-build build-dir must be build/{wheel_tag}")
    if "CMAKE_CXX_COMPILER_LAUNCHER" not in cmake_text:
        errors.append("CMake does not configure ccache as a compiler launcher")
    if "--no-build-isolation" not in venv_helper_text:
        errors.append("editable installs must use stable in-venv build requirements")
    for release_guard in (
        'SKBUILD_BUILD_DIR="$CI_PROJECT_DIR/build-clean/$CI_PIPELINE_ID"',
        "export CCACHE_DISABLE=1",
        "--tier T1",
        "regression pytest deferred: ordinary main push",
    ):
        if release_guard not in ci_text:
            errors.append(f"clean release-build guard is missing: {release_guard}")
    for concurrency_guard in (
        "resource_group: native-build-test",
        "interruptible: true",
    ):
        if concurrency_guard not in ci_text:
            errors.append(
                f"native build concurrency guard is missing: {concurrency_guard}"
            )
    if "CI_COMMIT_TAG" in ci_text:
        errors.append("version tags must reuse release-candidate evidence, not run CI")
    if "pytest /tmp/vibe-view/tests -q" in ci_text:
        errors.append("build-test must not duplicate the full vibe-view suite")
    for coupled_file in (
        "/tmp/vibe-view/tests/test_container_submission.py",
        "/tmp/vibe-view/tests/test_job_container_lifecycle.py",
        "/tmp/vibe-view/tests/test_kind_drift.py",
        "/tmp/vibe-view/tests/test_wavefunction_libint_parity.py",
    ):
        if coupled_file not in ci_text:
            errors.append(f"producer-coupled viewer guard is missing: {coupled_file}")
    if errors:
        raise ValueError("\n".join(errors))


REPRESENTATIVE_CASES: Final = (
    ("python-core", "main", ("python/vibeqc/runner.py",)),
    ("cpp-core", "main", ("cpp/src/rhf.cpp",)),
    ("website-only", "main", ("website/src/pages/index.astro",)),
    ("basis-only-mr", "merge_request", ("vibe-basis/src/vibe_basis/cli.py",)),
    ("docs-only-mr", "merge_request", ("docs/quickstart.md",)),
    ("handover-only-mr", "merge_request", ("handovers/HANDOVER_TEST_HEALTH.md",)),
    ("python-core-mr", "merge_request", ("python/vibeqc/runner.py",)),
    ("cpp-core-mr", "merge_request", ("cpp/src/rhf.cpp",)),
    ("build-config-mr", "merge_request", (".gitlab-ci.yml",)),
    ("release-candidate", "release_candidate", ("handovers/NOTE.md",)),
    ("tag", "tag", ("handovers/NOTE.md",)),
)


def representative_selections() -> list[dict[str, object]]:
    return [
        {
            "case": name,
            "context": context,
            "paths": list(paths),
            "jobs": list(select_jobs(paths, context=context)),
        }
        for name, context, paths in REPRESENTATIVE_CASES
    ]


def _git_changed_paths(base: str, head: str) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--name-only", base, head],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context",
        choices=("main", "merge_request", "release_candidate", "tag", "release", "web"),
        default="main",
    )
    parser.add_argument("--paths", nargs="*")
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--verify-gitlab", action="store_true")
    parser.add_argument("--verify-build-cache", action="store_true")
    parser.add_argument("--show-cases", action="store_true")
    args = parser.parse_args()

    if args.verify_gitlab:
        verify_gitlab_policy()
    if args.verify_build_cache:
        verify_build_reuse_policy()
    if args.show_cases:
        print(json.dumps(representative_selections(), indent=2, sort_keys=True))

    if args.paths is not None:
        paths = args.paths
    elif args.base and args.head:
        paths = _git_changed_paths(args.base, args.head)
    elif args.verify_gitlab or args.verify_build_cache or args.show_cases:
        return 0
    else:
        parser.error("pass --paths, --base/--head, --verify-gitlab, or --show-cases")

    print(
        json.dumps(
            {
                "context": args.context,
                "paths": paths,
                "jobs": list(select_jobs(paths, context=args.context)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
