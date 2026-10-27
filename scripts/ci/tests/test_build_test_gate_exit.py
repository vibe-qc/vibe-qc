"""Every step of a gated ``build-test`` script item fails the job (#446).

GitLab runs a job's script items in one shell under ``set -eo pipefail`` and
fails the job only when that shell exits non-zero. Inside
``if ...; then a && b && c; else ...; fi`` errexit is ignored for every
command of the ``&&`` chain except the last, and the ``if`` that then returns
non-zero does not stop the shell either. A failing ``b`` (the T1 gate verdict)
therefore left the job green. This test replays each script item that is gated
on a full-validation or release-candidate pipeline, with every external step
replaced by a stub, and fails each step in turn.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[3] / ".gitlab-ci.yml"
GATE = 'if [ "${VIBEQC_FULL_VALIDATION:-0}" = 1 ]'
NEXT = "next script item ran"
# Commands the gated items run by name; .venv/bin/{python,pip} are stubbed too.
# A new external command in a gated item makes the green replay exit 127: add it here.
STUBBED = ("apt-get", "bash", "ccache", "chown", "git", "install", "runuser", "useradd")
STUB = """#!/bin/sh
n=0
[ -f "$STUB_COUNTER" ] && read -r n < "$STUB_COUNTER"
n=$((n + 1))
echo "$n" > "$STUB_COUNTER"
[ "$n" != "${STUB_FAIL_AT:-0}" ]
"""
CONTEXTS = {
    "release candidate": {"CI_COMMIT_BRANCH": "release-candidate/v0.0.0"},
    "full validation": {"VIBEQC_FULL_VALIDATION": "1", "CI_COMMIT_BRANCH": "main"},
    "ordinary main push": {"CI_COMMIT_BRANCH": "main"},
}


def _fold(lines: list[str]) -> str:
    """YAML ``>-`` folding: a break between two base-indented lines becomes a space."""
    text = lines[0]
    for previous, line in zip(lines, lines[1:]):
        if not previous or not line:
            raise ValueError("blank lines inside a folded build-test item are not modelled")
        text += ("\n" if previous[0] == " " or line[0] == " " else " ") + line
    return text


def gated_build_test_items(config: str) -> list[str]:
    job = config.split("\nbuild-test:\n", 1)[1].split("\n  rules:\n", 1)[0]
    script = job.split("\n  script:\n", 1)[1]
    items: list[list[str]] = []
    for line in script.splitlines():
        if line.startswith("    - "):
            items.append([line[6:]])
        elif line.startswith("      ") and items:
            items[-1].append(line[6:])
    folded = [_fold(item[1:]) for item in items if item[0] == ">-"]
    return [text for text in folded if text.startswith(GATE)]


def replay(item: str, context: dict[str, str], fail_at: int = 0) -> tuple[int, int, str]:
    """Run ``item`` as GitLab would, then a following item; return (rc, stub calls, output)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stubs, venv = root / "stubs", root / ".venv" / "bin"
        stubs.mkdir()
        venv.mkdir(parents=True)
        for stub in [stubs / name for name in STUBBED] + [venv / "python", venv / "pip"]:
            stub.write_text(STUB)
            stub.chmod(0o755)
        (stubs / "grep").symlink_to(shutil.which("grep"))
        counter = root / "calls"
        env = {"PATH": str(stubs), "STUB_COUNTER": str(counter),
               "STUB_FAIL_AT": str(fail_at), "VIBEQC_RELEASE_BASE": "v0.0.0", **context}
        proc = subprocess.run(
            [shutil.which("bash"), "-c", f"set -eo pipefail\n{item}\necho {NEXT!r}"],
            cwd=root, env=env, capture_output=True, text=True, timeout=60,
        )
        calls = int(counter.read_text()) if counter.exists() else 0
        return proc.returncode, calls, proc.stdout + proc.stderr


@unittest.skipUnless(shutil.which("bash") and shutil.which("grep"), "needs bash and grep")
class BuildTestGateExitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.items = gated_build_test_items(CONFIG.read_text())

    def test_the_gated_chains_are_found(self) -> None:
        chains = [item for item in self.items if " && " in item]
        self.assertTrue(any("gate_verdict.py" in item for item in chains))
        self.assertTrue(any("/tmp/vibe-view/tests/" in item for item in chains))

    def test_every_failing_step_fails_the_job(self) -> None:
        for name, context in CONTEXTS.items():
            for index, item in enumerate(self.items):
                rc, steps, output = replay(item, context)
                with self.subTest(context=name, item=index):
                    self.assertEqual(rc, 0, output)
                    self.assertIn(NEXT, output)
                    if name != "ordinary main push" and " && " in item:
                        self.assertGreaterEqual(steps, 2, output)
                for fail_at in range(1, steps + 1):
                    rc, calls, output = replay(item, context, fail_at)
                    with self.subTest(context=name, item=index, failing_step=fail_at):
                        self.assertNotEqual(rc, 0, output)
                        self.assertNotIn(NEXT, output)
                        self.assertEqual(calls, fail_at, output)

    def test_native_build_modes_and_install_failure(self) -> None:
        config = CONFIG.read_text()
        script = config.split("    - |\n      mkdir -p test-results\n", 1)[1]
        script = "mkdir -p test-results\n" + script.split("    - |\n", 1)[0]
        script = "\n".join(line[6:] if line.startswith("      ") else line
                           for line in script.splitlines())
        for clean in ("0", "1"):
            for fail in ("0", "9"):
                with self.subTest(clean=clean, install_status=fail), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    stubs = root / "stubs"
                    stubs.mkdir()
                    for command in ("mkdir", "date", "tee"):
                        (stubs / command).symlink_to(shutil.which(command))
                    (stubs / "ccache").write_text("#!/bin/sh\necho 'cache statistics'\n")
                    (stubs / "ccache").chmod(0o755)
                    (stubs / "bash").write_text("#!/bin/sh\nexit \"$INSTALL_STATUS\"\n")
                    (stubs / "bash").chmod(0o755)
                    env = {"PATH": str(stubs), "CI_PROJECT_DIR": tmp,
                           "CI_PIPELINE_ID": "123", "VIBEQC_CLEAN_BUILD": clean,
                           "CCACHE_DISABLE": "1", "INSTALL_STATUS": fail}
                    command = script + '\nprintf "MODE=%s CACHE=%s" "$SKBUILD_BUILD_DIR" "${CCACHE_DISABLE:-enabled}"'
                    result = subprocess.run([shutil.which("bash"), "-euc", command],
                                            cwd=root, env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode == 0, fail == "0", result.stderr)
                    self.assertTrue((root / "test-results/native-build-seconds.txt").is_file())
                    self.assertTrue((root / "test-results/compile-cache-after.txt").is_file())
                    if fail == "0":
                        directory = "build-clean/123" if clean == "1" else "build/cp313-cp313-linux_x86_64"
                        self.assertIn(f"MODE={tmp}/{directory}", result.stdout)
                        self.assertIn("CACHE=" + ("1" if clean == "1" else "enabled"), result.stdout)

    def test_ordinary_main_pushes_defer_the_regression_steps(self) -> None:
        for item in self.items:
            if " && " in item:
                rc, steps, output = replay(item, CONTEXTS["ordinary main push"])
                self.assertEqual((rc, steps), (0, 0), output)
                self.assertIn("deferred: ordinary main push", output)


if __name__ == "__main__":
    unittest.main()
