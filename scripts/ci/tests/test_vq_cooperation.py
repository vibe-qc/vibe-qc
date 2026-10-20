"""Bounded integration tests of the sourced build helper using a fake queue.

Run directly with Python; VIBEQC_COOPERATION_HELPER selects a retained control.
No queue process or native build is started.
"""
import json
import os
import resource
from pathlib import Path
import subprocess
import tempfile
import unittest

HELPER = Path(os.environ.get("VIBEQC_COOPERATION_HELPER", Path(__file__).resolve().parents[2] / "_vq_cooperation.sh"))
FAKE = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['FAKE_LOG'], 'a') as f:
    f.write(json.dumps(args) + '\n')
mode = os.environ.get('FAKE_MODE', 'normal')
if args[:2] == ['daemon', 'ping']:
    if mode == 'no-daemon': sys.exit(1)
    if mode == 'unhealthy': sys.exit(2)
    print('daemon reachable')
    sys.exit(0)
verb = args[0]
if mode == verb + '-failure':
    print('command failed', file=sys.stderr)
    sys.exit(2)
state_path = Path(os.environ['FAKE_STATE'])
state = json.loads(state_path.read_text())
host = 'localhost' if 'localhost' in args else 'remote-default'
tag = args[args.index('--paused-by') + 1] if '--paused-by' in args else None
count = 0
for job in state:
    if job['host'] != host: continue
    if verb == 'pause' and job['state'] == 'running':
        job.update(state='suspended', tag=tag)
        count += 1
    elif verb == 'resume' and job['state'] == 'suspended' and (tag is None or job['tag'] == tag):
        job.update(state='running', tag=None)
        count += 1
state_path.write_text(json.dumps(state))
if mode == verb + '-malformed': print('unexpected response')
elif mode == verb + '-truncated': print(f'{verb}d {count} jobs (')
elif mode == verb + '-errors': print(f'{verb}d {count} jobs (1 error(s))')
else: print(f'{verb}d {count} job' + ('s' if count != 1 else '') + ' (0 already suspended, 1 not RUNNING)')
'''


def child_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))


class CooperationTests(unittest.TestCase):
    def run_helper(self, body, mode="normal", jobs=None, extra=None):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = root / "vq"
            fake.write_text(FAKE)
            fake.chmod(0o755)
            log = root / "argv.jsonl"
            state = root / "state.json"
            state.write_text(json.dumps(jobs if jobs is not None else [
                dict(host="localhost", state="running", tag=None),
                dict(host="localhost", state="suspended", tag="operator"),
                dict(host="remote-default", state="running", tag=None),
            ]))
            env = os.environ.copy()
            for key in list(env):
                if key.startswith("VIBEQC_VQ_"): env.pop(key)
            env.update(VQ_BIN=str(fake), FAKE_LOG=str(log), FAKE_STATE=str(state), FAKE_MODE=mode, HELPER=str(HELPER))
            env.update(extra or {})
            result = subprocess.run(["bash", "-c", 'set -euo pipefail; . "$HELPER"; ' + body], env=env, text=True, capture_output=True, timeout=30, preexec_fn=child_limits)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return calls, json.loads(state.read_text()), result.stderr, result.stdout

    def test_local_host_and_matching_tag_preserve_other_pauses(self):
        calls, state, stderr, _ = self.run_helper("vibeqc_vq_pause; vibeqc_vq_resume; vibeqc_vq_resume")
        self.assertEqual(calls[0], ["daemon", "ping", "localhost", "--timeout", "2"])
        self.assertEqual(len(calls), 3)
        pause, resume = calls[1:]
        self.assertEqual(pause[:4], ["pause", "localhost", "--all", "--paused-by"])
        self.assertRegex(pause[4], r"^vibeqc-build-[0-9]+$")
        self.assertEqual(resume, ["resume", "localhost", "--all", "--paused-by", pause[4]])
        self.assertEqual(state[1]["state"], "suspended")
        self.assertEqual(state[1]["tag"], "operator")
        self.assertEqual(state[2]["state"], "running")
        self.assertNotIn("failed", stderr)

    def test_nested_source_preserves_ownership(self):
        calls, _, _, _ = self.run_helper('vibeqc_vq_pause; . "$HELPER"; vibeqc_vq_pause; vibeqc_vq_resume')
        self.assertEqual([c[0] for c in calls], ["daemon", "pause", "resume"])

    def test_nested_child_does_not_resume_parent(self):
        calls, _, _, stdout = self.run_helper('vibeqc_vq_install_trap; vibeqc_vq_pause; bash -c \'set -euo pipefail; . "$HELPER"; vibeqc_vq_install_trap; vibeqc_vq_pause\'; cat "$FAKE_LOG"; vibeqc_vq_resume')
        self.assertEqual([c[0] for c in calls], ["daemon", "pause", "resume"])
        self.assertNotIn('"resume"', stdout)

    def test_trap_composes_once(self):
        calls, _, _, stdout = self.run_helper("trap 'echo previous' EXIT; vibeqc_vq_install_trap; vibeqc_vq_install_trap; vibeqc_vq_pause")
        self.assertEqual([c[0] for c in calls], ["daemon", "pause", "resume"])
        self.assertEqual(stdout, "previous\n")

    def test_no_daemon_skips_queue_commands(self):
        calls, _, stderr, _ = self.run_helper("vibeqc_vq_pause; vibeqc_vq_resume", "no-daemon")
        self.assertEqual(len(calls), 1)
        self.assertIn("no local daemon", stderr)
        self.assertNotIn("failed", stderr)

    def test_unhealthy_daemon_is_reported(self):
        calls, _, stderr, _ = self.run_helper("vibeqc_vq_pause; vibeqc_vq_resume", "unhealthy")
        self.assertEqual(len(calls), 1)
        self.assertIn("probe failed", stderr)

    def test_zero_jobs_is_success(self):
        calls, _, stderr, _ = self.run_helper("vibeqc_vq_pause; vibeqc_vq_resume", jobs=[])
        self.assertEqual(len(calls), 3)
        self.assertIn("paused 0 jobs", stderr)
        self.assertIn("resumed 0 jobs", stderr)
        self.assertNotIn("failed", stderr)

    def test_opt_out(self):
        calls, _, _, _ = self.run_helper("vibeqc_vq_pause; vibeqc_vq_resume", extra={"VIBEQC_VQ_COOPERATE": "0"})
        self.assertEqual(calls, [])

    def test_pause_error_still_cleans_only_own_tag(self):
        for mode in ("pause-failure", "pause-malformed", "pause-truncated", "pause-errors"):
            with self.subTest(mode=mode):
                calls, state, stderr, _ = self.run_helper("vibeqc_vq_install_trap; vibeqc_vq_pause", mode)
                self.assertEqual([c[0] for c in calls], ["daemon", "pause", "resume"])
                self.assertEqual(state[1]["tag"], "operator")
                self.assertIn("warning", stderr)

    def test_resume_failure_can_be_retried(self):
        calls, _, stderr, _ = self.run_helper("vibeqc_vq_pause; vibeqc_vq_resume; export FAKE_MODE=normal; vibeqc_vq_resume", "resume-failure")
        self.assertEqual([c[0] for c in calls], ["daemon", "pause", "resume", "resume"])
        self.assertIn("warning", stderr)

    def test_resume_malformed_and_partial_errors_are_reported(self):
        for mode in ("resume-malformed", "resume-truncated", "resume-errors"):
            with self.subTest(mode=mode):
                _, _, stderr, _ = self.run_helper("vibeqc_vq_pause; vibeqc_vq_resume", mode)
                self.assertIn("warning", stderr)

    def test_quoted_existing_trap(self):
        calls, _, _, stdout = self.run_helper("trap \"printf '%s\\n' \\\"it's preserved\\\"\" EXIT; vibeqc_vq_install_trap; vibeqc_vq_pause")
        self.assertEqual([c[0] for c in calls], ["daemon", "pause", "resume"])
        self.assertEqual(stdout, "it's preserved\n")

    def test_missing_binary(self):
        calls, _, stderr, _ = self.run_helper("vibeqc_vq_pause; vibeqc_vq_resume", extra={"VQ_BIN": "/nonexistent-vq"})
        self.assertEqual(calls, [])
        self.assertIn("probe failed", stderr)


if __name__ == "__main__":
    unittest.main()
