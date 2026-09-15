"""Portable configuration and dispatch checks; no numerical core is required.

Run directly with Python for the hermetic launcher boundary checks.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

STUDY = Path(__file__).resolve().parents[1] / "studies" / "aiccm-2026"
spec = importlib.util.spec_from_file_location("isolated_site_settings", STUDY / "site_settings.py")
site = importlib.util.module_from_spec(spec)
with patch.dict(os.environ):
    os.environ.pop("AICCM_SITE_CONFIG", None)
    spec.loader.exec_module(site)


class SiteSettingsTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("AICCM_SITE_CONFIG", None)
        self.profile = patch.object(site, "_profile", {})
        self.profile.start()
        self.addCleanup(self.profile.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def load(self, data):
        path = self.root / "site.json"
        path.write_text(data)
        os.environ["AICCM_SITE_CONFIG"] = str(path)
        return site.load_profile()

    def test_missing_profile_permits_local_metadata_but_blocks_remote_defaults(self):
        self.assertEqual(site.load_profile(), {})
        self.assertIsNone(site.system_host("sample"))
        self.assertEqual(site.system_host("sample", "local"), "local")
        for operation in (site.molecular_hosts, lambda: site.required_host("paper_host")):
            with self.assertRaisesRegex(ValueError, "explicitly enabled"):
                operation()
        self.assertEqual(site.explicit_remote_host("compute-1"), "compute-1")

    def test_external_example_requires_deliberate_enablement(self):
        profile = self.load((STUDY / "site-profile.example.json").read_text())
        site._profile = profile
        hosts = site.host_map("tier_hosts")
        with self.assertRaises(ValueError):
            hosts["A"]
        profile["allow_submission"] = True
        self.assertEqual(hosts["A"], "compute-small")
        self.assertEqual(site.molecular_hosts(), ("compute-medium", "compute-small"))
        self.assertEqual(site.system_host("mgo"), "compute-large")

    def test_aliases_cannot_be_local_addresses_options_or_shell_fragments(self):
        site._profile = {"local_host_aliases": ["workstation", "workstation.local"]}
        for host in ("localhost", "LOCALHOST.", "local", "127.0.0.2", "0.0.0.0",
                     "::1", "workstation", "WORKSTATION.local", "-oProxyCommand=x",
                     "node;touch marker", "node\nother", "node path", "user@node"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                site.explicit_remote_host(host)

    def test_current_hostname_mdns_alias_is_local_without_a_profile(self):
        for current in ("workstation", "workstation.example.invalid"):
            with patch.object(site.socket, "gethostname", return_value=current):
                for alias in (current, "workstation", "workstation.local",
                              "WORKSTATION.LOCAL.", "localhost.localdomain"):
                    with self.subTest(current=current, alias=alias), self.assertRaises(ValueError):
                        site.explicit_remote_host(alias)

    def test_schema_unknown_duplicate_and_invalid_values_fail_closed(self):
        for data in ('{}', '[]', '{"schema_version":true}',
                     '{"schema_version":1,"schema_version":1}',
                     '{"schema_version":1,"command":"do-something"}',
                     '{"schema_version":1,"allow_submission":"true"}',
                     '{"schema_version":1,"tier_hosts":{"A":"-node"}}',
                     '{"schema_version":1,"tier_hosts":{"A":"node","A":"other"}}',
                     '{"schema_version":1,"molecular_hosts":[]}',
                     '{"schema_version":1,"paper_host":null}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.load(data)

    def test_profile_must_stay_outside_checkout_including_symlinks(self):
        for path in (STUDY / "site-profile.example.json", self.root / "link.json"):
            if path.name == "link.json":
                path.symlink_to(STUDY / "site-profile.example.json")
            os.environ["AICCM_SITE_CONFIG"] = str(path)
            with self.assertRaisesRegex(ValueError, "outside the source checkout"):
                site.load_profile()

    def test_missing_empty_oversized_and_invalid_json_are_rejected(self):
        for value in ("", str(self.root / "missing-private-config")):
            os.environ["AICCM_SITE_CONFIG"] = value
            with self.assertRaises(ValueError) as error:
                site.load_profile()
            self.assertNotIn(str(self.root), str(error.exception))
        with self.assertRaisesRegex(ValueError, "size limit"):
            self.load(" " * 65537)
        with self.assertRaises(ValueError):
            self.load("not JSON")

    def test_private_attestation_alias_is_opt_in_and_generic_mode_remains_valid(self):
        self.assertEqual(site.accepted_bundle_modes(), (site.PUBLIC_BUNDLE_MODE,))
        site._profile = site.validate_profile({"schema_version": 1,
            "bundle_attestation_mode": "legacy-test-attestation/v1"})
        self.assertEqual(site.bundle_mode(), "legacy-test-attestation/v1")
        self.assertEqual(site.accepted_bundle_modes(),
                         (site.PUBLIC_BUNDLE_MODE, "legacy-test-attestation/v1"))

    def test_preflight_with_no_profile_never_invokes_queue(self):
        preflight_spec = importlib.util.spec_from_file_location(
            "isolated_preflight", STUDY / "preflight_mol_batch.py")
        module = importlib.util.module_from_spec(preflight_spec)
        with patch.dict(sys.modules, {"site_settings": site}):
            preflight_spec.loader.exec_module(module)
        with patch.object(module.subprocess, "run") as run:
            self.assertFalse(module.check_vq())
            run.assert_not_called()


class LauncherBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.study = self.root / "studies" / "aiccm-2026"
        self.study.mkdir(parents=True)
        shutil.copyfile(STUDY / "run.sh", self.study / "run.sh")
        self.log = self.root / "calls"
        self.env = dict(os.environ)
        for key in tuple(self.env):
            if key.startswith(("AICCM_", "VIBEQC_")):
                self.env.pop(key)
        self.env["TEST_DISPATCH_LOG"] = str(self.log)
        self.env["AICCM_PROBE_TMPDIR"] = str(self.root)

    def interpreter(self, path, label):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/bin/sh\nprintf "%s\\n" "' + label + ':$*" >> "$TEST_DISPATCH_LOG"\n')
        path.chmod(0o755)
        return path

    def run_launcher(self):
        return subprocess.run(["bash", str(self.study / "run.sh"), "sample", "rhf-ri"],
                              env=self.env, text=True, capture_output=True)

    def test_explicit_interpreter_wins_and_receives_probe_before_producer(self):
        self.interpreter(self.root / ".venv/bin/python", "checkout")
        override = self.interpreter(self.root / "operator bin/python", "explicit")
        self.env["VIBEQC_PYTHON"] = str(override)
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(line.startswith("explicit:") for line in calls))
        self.assertIn("probe_host.py", calls[0])
        self.assertIn("run_case.py sample rhf-ri", calls[1])
        self.assertEqual(list(self.root.glob("aiccm-probe-attestation.*")), [])
        self.assertNotIn(str(override), result.stderr)

    def test_invalid_explicit_interpreter_never_falls_back(self):
        self.interpreter(self.root / ".venv/bin/python", "checkout")
        for selected in ("", str(self.root / "missing")):
            self.env["VIBEQC_PYTHON"] = selected
            result = self.run_launcher()
            self.assertEqual(result.returncode, 2)
            self.assertFalse(self.log.exists())

    def test_checkout_interpreter_remains_available_without_site_config(self):
        self.interpreter(self.root / ".venv/bin/python", "checkout")
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.log.read_text().startswith("checkout:"))


if __name__ == "__main__":
    unittest.main()
