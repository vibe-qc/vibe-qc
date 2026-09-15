from __future__ import annotations

import unittest
from pathlib import Path

from scripts.ci.path_policy import (
    select_jobs,
    verify_build_reuse_policy,
    verify_gitlab_policy,
)


class PathPolicyTests(unittest.TestCase):
    """Release-only CI policy (2026-07-26).

    Ordinary ``main`` commits select NO jobs — test evidence is local, and
    the single release-candidate pipeline proves every component gate on the
    exact release tree. The tag reuses that evidence. Path-gated selection survives
    only for merge-request pipelines.
    """

    def test_ordinary_main_commits_select_no_jobs(self) -> None:
        for paths in (
            ["vibe-basis/src/vibe_basis/cli.py"],
            ["docs/quickstart.md"],
            ["handovers/HANDOVER_TEST_HEALTH.md"],
            ["python/vibeqc/runner.py"],
            ["cpp/src/rhf.cpp"],
            [".gitlab-ci.yml"],
        ):
            with self.subTest(paths=paths):
                self.assertEqual(select_jobs(paths, context="main"), ())

    def test_website_main_pushes_select_their_own_gate(self) -> None:
        """The website lane publishes on the push, so it gates on the push.

        Every other main path defers to release-candidate evidence; this one
        cannot, because website-deploy has already reached the public root by
        then. Both deploys `needs: ["website-test"]`, so selecting it here is
        what keeps the deploy from running unverified.
        """
        self.assertEqual(
            select_jobs(["website/src/pages/index.astro"], context="main"),
            ("website-test",),
        )
        self.assertEqual(
            select_jobs(["website/tests/products.verify.mjs"], context="main"),
            ("website-test",),
        )
        # The exception is scoped to the lane: a neighbouring path stays empty.
        self.assertEqual(select_jobs(["websites/other.md"], context="main"), ())

    def test_basis_only_mr_avoids_native_build(self) -> None:
        self.assertEqual(
            select_jobs(["vibe-basis/src/vibe_basis/cli.py"], context="merge_request"),
            ("ci-path-policy", "vibe-basis-test"),
        )

    def test_docs_only_mr_avoids_native_build(self) -> None:
        self.assertEqual(
            select_jobs(["docs/quickstart.md"], context="merge_request"),
            ("ci-path-policy", "docs-build"),
        )

    def test_handover_only_mr_runs_policy_guard(self) -> None:
        self.assertEqual(
            select_jobs(["handovers/HANDOVER_TEST_HEALTH.md"], context="merge_request"),
            ("ci-path-policy",),
        )

    def test_python_core_mr_requires_native_build(self) -> None:
        self.assertIn(
            "build-test",
            select_jobs(["python/vibeqc/runner.py"], context="merge_request"),
        )
        self.assertIn(
            "build-test",
            select_jobs(
                ["python/vibeqc_naming/normalization.py"],
                context="merge_request",
            ),
        )

    def test_cpp_core_mr_requires_native_build(self) -> None:
        self.assertIn(
            "build-test",
            select_jobs(["cpp/src/rhf.cpp"], context="merge_request"),
        )

    def test_test_change_mr_requires_native_build(self) -> None:
        self.assertIn(
            "build-test",
            select_jobs(["tests/test_rhf.py"], context="merge_request"),
        )

    def test_native_build_metadata_mr_requires_native_build(self) -> None:
        for path in (
            "CMakeLists.txt",
            "pyproject.toml",
            "scripts/setup_native_deps.sh",
            "scripts/build_libint.sh",
            "scripts/gen_qvf_v2_schema.py",
            "scripts/_venv_helpers.sh",
            "scripts/test_gate/suite_manifest.json",
            ".gitlab-ci.yml",
        ):
            with self.subTest(path=path):
                self.assertIn(
                    "build-test", select_jobs([path], context="merge_request")
                )

    def test_shared_lifecycle_helpers_gate_every_component(self) -> None:
        expected = (
            "build-test",
            "ci-path-policy",
            "vibe-basis-test",
        )
        for path in ("scripts/_build_lock.sh", "scripts/_lifecycle_lock.sh"):
            with self.subTest(path=path):
                self.assertEqual(select_jobs([path], context="merge_request"), expected)

    def test_qvf_producer_and_schema_surface_mr_requires_native_build(
        self,
    ) -> None:
        for path in (
            "python/vibeqc/output/formats/qvf.py",
            "python/vibeqc/output/formats/qvf_manifest.schema.json",
        ):
            with self.subTest(path=path):
                self.assertIn(
                    "build-test", select_jobs([path], context="merge_request")
                )

    def test_release_candidate_requires_all_gates_and_tag_reuses_it(self) -> None:
        expected = (
            "build-test",
            "ci-path-policy",
            "docs-build",
            "qvf-conformance",
            "vibe-basis-test",
        )
        self.assertEqual(
            select_jobs(["handovers/NOTE.md"], context="release_candidate"),
            expected,
        )
        self.assertEqual(select_jobs(["handovers/NOTE.md"], context="tag"), ())

    def test_release_branch_is_docs_only(self) -> None:
        self.assertEqual(
            select_jobs(["docs/quickstart.md"], context="release"),
            ("ci-path-policy", "docs-build"),
        )

    def test_manual_web_pipeline_builds_docs(self) -> None:
        self.assertEqual(
            select_jobs([], context="web"),
            ("ci-path-policy", "docs-build"),
        )

    def test_gitlab_rules_match_the_tested_policy(self) -> None:
        verify_gitlab_policy()


    def test_website_gate_checks_both_published_bases(self) -> None:
        config = (Path(__file__).resolve().parents[3] / ".gitlab-ci.yml").read_text()
        self.assertIn("\nwebsite-test:\n", config)
        # The gate must run the suite, not just rebuild the site.
        gate = config.split("\nwebsite-test:\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn("npm --prefix website run verify", gate)
        # The lane publishes two bases; the gate must cover both.
        self.assertIn("PREVIEW_BASE=/preview/", gate)


    def test_native_reuse_and_clean_release_guards_are_present(self) -> None:
        verify_build_reuse_policy()


if __name__ == "__main__":
    unittest.main()
