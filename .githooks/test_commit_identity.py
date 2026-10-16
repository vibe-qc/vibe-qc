"""Exercise the commit-identity hook in isolated disposable repositories."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HOOKS = Path(os.environ.get("PRIVACY_HOOKS_DIR", Path(__file__).resolve().parent))
PUBLIC = "mpei@vibe-qc.com"
# Synthetic value on a reserved example domain; no real address is stored.
OTHER = "private-fixture@example.org"


class CommitIdentityHookTests(unittest.TestCase):
    def run_check(self, email, extra=(), env_overrides=None, through_pre_commit=False):
        with tempfile.TemporaryDirectory(prefix="identity-hook-") as temporary:
            repo = Path(temporary)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            shutil.copytree(
                HOOKS, repo / ".githooks", ignore=shutil.ignore_patterns("__pycache__")
            )
            git = ["git", "-C", str(repo)]
            subprocess.run([*git, "config", "user.name", "Example"], check=True)
            subprocess.run([*git, "config", "user.email", email], check=True)
            for value in extra:
                subprocess.run(
                    [*git, "config", "--add", "privacy.commitEmail", value], check=True
                )
            env = os.environ.copy()
            for key in (
                "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                "GIT_COMMITTER_EMAIL", "VIBE_PRIVACY_REQUIRE_TERMS",
                "VIBE_PRIVACY_TERMS_FILE",
            ):
                env.pop(key, None)
            env.update(env_overrides or {})
            if through_pre_commit:
                (repo / "example.txt").write_text("example\n")
                subprocess.run([*git, "add", "example.txt"], check=True)
                cmd = ["sh", str(repo / ".githooks/pre-commit")]
            else:
                cmd = ["python3", str(repo / ".githooks/check_commit_identity.py")]
            return subprocess.run(
                cmd, cwd=repo, env=env, capture_output=True, text=True, check=False
            )

    def test_public_addresses_pass(self):
        for email in ("mpei@vibe-qc.com", "project@vibe-qc.com", "MPEI@vibe-qc.com"):
            with self.subTest(email=email):
                result = self.run_check(email)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_other_address_is_refused_without_being_repeated(self):
        result = self.run_check(OTHER)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(OTHER, result.stderr)
        self.assertIn("redacted", result.stderr)
        self.assertIn("git config --local user.email", result.stderr)

    def test_locally_added_public_address_passes(self):
        result = self.run_check("someone@example.org", extra=("someone@example.org",))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_committer_identity_is_checked_too(self):
        result = self.run_check(PUBLIC, env_overrides={"GIT_COMMITTER_EMAIL": OTHER})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("committer", result.stderr)
        self.assertNotIn(OTHER, result.stderr)

    def test_pre_commit_runs_the_check(self):
        passed = self.run_check(PUBLIC, through_pre_commit=True)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        refused = self.run_check(OTHER, through_pre_commit=True)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("commit identity check", refused.stderr)


if __name__ == "__main__":
    unittest.main()
