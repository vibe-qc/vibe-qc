from __future__ import annotations

import unittest
from pathlib import Path

from scripts.ci.path_policy import (
    select_jobs,
    verify_build_reuse_policy,
    verify_gitlab_policy,
)


class PathPolicyTests(unittest.TestCase):
    """Cheap privacy preflight on main; expensive tests stay path/candidate gated."""

    def test_ordinary_main_commits_select_privacy_only(self) -> None:
        for paths in (
            ["vibe-basis/src/vibe_basis/cli.py"],
            ["vibeqc-naming/pyproject.toml"],
            ["docs/quickstart.md"],
            ["studies/skala-showcase/README.md"],
            ["python/vibeqc/runner.py"],
            ["cpp/src/rhf.cpp"],
            [".gitlab-ci.yml"],
        ):
            with self.subTest(paths=paths):
                self.assertEqual(select_jobs(paths, context="main"), ("source-privacy",))

    def test_website_main_pushes_select_their_own_gate(self) -> None:
        """The website lane publishes on the push, so it gates on the push.

        Every other main path defers to release-candidate evidence; this one
        cannot, because website-deploy has already reached the public root by
        then. Both deploys `needs: ["website-test"]`, so selecting it here is
        what keeps the deploy from running unverified.
        """
        self.assertEqual(
            select_jobs(["website/src/pages/index.astro"], context="main"),
            ("source-privacy", "website-test"),
        )
        self.assertEqual(
            select_jobs(["website/tests/products.verify.mjs"], context="main"),
            ("source-privacy", "website-test"),
        )
        # The exception is scoped to the lane: a neighbouring path stays empty.
        self.assertEqual(select_jobs(["websites/other.md"], context="main"), ("source-privacy",))

    def test_trah_regression_is_automatic_and_bounded_on_mrs(self) -> None:
        for path in ("cpp/src/trah.cpp", "cpp/include/vibeqc/trah.hpp",
                     "tests/test_trah_noncanonical.cpp", "tests/native/CMakeLists.txt"):
            with self.subTest(path=path):
                jobs = select_jobs([path], context="merge_request")
                self.assertIn("native-regressions", jobs)
                self.assertNotIn("build-test", jobs)
        self.assertNotIn("native-regressions", select_jobs(
            ["python/vibeqc/runner.py"], context="merge_request"))

    def test_basis_only_mr_avoids_native_build(self) -> None:
        self.assertEqual(
            select_jobs(["vibe-basis/src/vibe_basis/cli.py"], context="merge_request"),
            ("ci-path-policy", "source-privacy", "vibe-basis-test"),
        )

    def test_docs_only_mr_avoids_native_build(self) -> None:
        self.assertEqual(
            select_jobs(["docs/quickstart.md"], context="merge_request"),
            ("ci-path-policy", "source-privacy"),
        )

    def test_study_only_mr_runs_policy_guard(self) -> None:
        self.assertEqual(
            select_jobs(["studies/skala-showcase/README.md"], context="merge_request"),
            ("ci-path-policy", "source-privacy"),
        )

    def test_naming_source_or_packaging_mr_runs_the_naming_gate(self) -> None:
        """The standalone distribution (#536) ships python/vibeqc_naming, so
        either side gates on the pure-Python job, never on a native build."""
        for path in ("python/vibeqc_naming/organic.py", "vibeqc-naming/hatch_build.py"):
            with self.subTest(path=path):
                self.assertEqual(
                    select_jobs([path], context="merge_request", build_native=True),
                    ("ci-path-policy", "source-privacy", "vibeqc-naming-test"),
                )
        self.assertNotIn("vibeqc-naming-test", select_jobs(["python/vibeqc/runner.py"], context="merge_request"))

    def test_python_only_mr_uses_cheap_checks(self) -> None:
        self.assertNotIn(
            "build-test",
            select_jobs(["python/vibeqc/runner.py"], context="merge_request"),
        )
        self.assertNotIn(
            "build-test",
            select_jobs(
                ["python/vibeqc_naming/normalization.py"],
                context="merge_request",
            ),
        )

    def test_ci_only_changes_do_not_build_but_explicit_full_does(self) -> None:
        self.assertNotIn("build-test", select_jobs([".gitlab-ci.yml"], context="merge_request"))
        self.assertIn("build-test", select_jobs([], context="web", full_validation=True))
        config = (Path(__file__).resolve().parents[3] / ".gitlab-ci.yml").read_text()
        self.assertIn("$VIBEQC_FULL_VALIDATION == \"1\"", config)
        self.assertIn('${VIBEQC_FULL_VALIDATION:-0}', config)

    def test_cpp_core_mr_build_is_opt_in(self) -> None:
        self.assertIn(
            "build-test",
            select_jobs(["cpp/src/rhf.cpp"], context="merge_request", build_native=True),
        )

    def test_test_only_mr_avoids_compile_only_native_job(self) -> None:
        for path in ("tests/test_rhf.py", "scripts/test_gate/suite_manifest.json",
                     "scripts/test_gate/run_full_suite.py"):
            with self.subTest(path=path):
                self.assertNotIn(
                    "build-test", select_jobs([path], context="merge_request", build_native=True)
                )
        self.assertIn("build-test", select_jobs(
            ["tests/test_rhf.py", "cpp/src/rhf.cpp"], context="merge_request", build_native=True
        ))

    def test_native_build_metadata_mr_requires_native_build(self) -> None:
        for path in (
            "CMakeLists.txt",
            "pyproject.toml",
            "scripts/setup_native_deps.sh",
            "scripts/build_libint.sh",
            "scripts/_venv_helpers.sh",
        ):
            with self.subTest(path=path):
                self.assertIn(
                    "build-test", select_jobs([path], context="merge_request", build_native=True)
                )

    def test_shared_lifecycle_helpers_gate_every_component(self) -> None:
        expected = (
            "build-test",
            "ci-path-policy",
            "source-privacy",
            "vibe-basis-test",
        )
        for path in ("scripts/_build_lock.sh", "scripts/_lifecycle_lock.sh"):
            with self.subTest(path=path):
                self.assertEqual(select_jobs([path], context="merge_request", build_native=True), expected)

    def test_qvf_python_surface_does_not_compile_unchanged_cpp(
        self,
    ) -> None:
        for path in (
            "python/vibeqc/output/formats/qvf.py",
            "python/vibeqc/output/formats/qvf_manifest.schema.json",
        ):
            with self.subTest(path=path):
                self.assertNotIn(
                    "build-test", select_jobs([path], context="merge_request", build_native=True)
                )

    def test_release_candidate_requires_all_gates_and_tag_reuses_it(self) -> None:
        expected = (
            "build-test",
            "changelog-audit",
            "ci-path-policy",
            "docs-build",
            "qvf-conformance",
            "source-privacy",
            "vibe-basis-test",
            "vibeqc-naming-test",
        )
        self.assertEqual(
            select_jobs(["studies/skala-showcase/README.md"], context="release_candidate"),
            expected,
        )
        self.assertEqual(select_jobs(["studies/skala-showcase/README.md"], context="tag"), ())

    def test_release_branch_is_docs_only(self) -> None:
        self.assertEqual(
            select_jobs(["docs/quickstart.md"], context="release"),
            ("ci-path-policy", "docs-build", "source-privacy"),
        )

    def test_manual_web_pipeline_builds_docs(self) -> None:
        self.assertEqual(
            select_jobs([], context="web"),
            ("ci-path-policy", "docs-build", "qvf-conformance", "source-privacy"),
        )

    def test_privacy_gate_precedes_builds_and_trusts_only_this_checkout(self) -> None:
        config = (Path(__file__).resolve().parents[3] / ".gitlab-ci.yml").read_text()
        gate = config.split("\nsource-privacy:\n", 1)[1].split("\nci-path-policy:", 1)[0]
        self.assertIn("stage: privacy", gate)
        self.assertIn("stages:\n  - privacy\n  - test", config)
        # A main pipeline containing only .pre/.post jobs would not run.
        self.assertNotIn("stage: .pre", gate)
        trust = 'git config --global --add safe.directory "$CI_PROJECT_DIR"'
        scan = 'python .githooks/source_privacy.py --tree "$CI_COMMIT_SHA"'
        self.assertIn(trust, gate)
        self.assertLess(gate.index(trust), gate.index(scan))
        self.assertNotIn('safe.directory "*"', gate)
        website = config.split("\nwebsite-test:\n", 1)[1]
        self.assertIn('needs: ["source-privacy"]', website)

    def test_routine_native_and_prose_changes_keep_a_bounded_gate(self) -> None:
        for path in ("cpp/src/rhf.cpp", "CMakeLists.txt", "docs/quickstart.md"):
            jobs = select_jobs([path], context="merge_request")
            self.assertNotIn("build-test", jobs)
            self.assertNotIn("docs-build", jobs)
        self.assertIn("docs-build", select_jobs(["docs/conf.py"], context="merge_request"))
        self.assertIn("qvf-conformance", select_jobs(
            ["python/vibeqc/output/formats/qvf_manifest.schema.json"], context="merge_request"
        ))
        self.assertNotIn("vibe-basis-test", select_jobs([".gitlab-ci.yml"], context="merge_request"))
        config = (Path(__file__).resolve().parents[3] / ".gitlab-ci.yml").read_text()
        self.assertIn('$CI_PIPELINE_SOURCE == "merge_request_event" && $VIBEQC_BUILD_NATIVE == "1"', config)

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
