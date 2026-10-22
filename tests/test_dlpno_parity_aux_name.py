"""Source-only CLI forwarding controls; no native or reference calculation."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class TestParityAuxName(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "parity_aux_test", ROOT / "examples/regression/parity_periodic_dlpno_vs_pyscf.py"
        )
        self.driver = importlib.util.module_from_spec(spec)
        # No geometry/array work is needed to exercise argument forwarding.
        with patch.dict(sys.modules, {"numpy": SimpleNamespace()}):
            spec.loader.exec_module(self.driver)

    def test_canonical_record_and_preserved_choices(self):
        library = ROOT / "python/vibeqc/basis_library/qvf"
        canonical = "def2-universal-jkfit"
        self.assertFalse((library / "def2-universal-jk.qvf.json").exists())
        record = json.loads((library / f"{canonical}.qvf.json").read_text())
        self.assertEqual(record["name"], canonical)
        for symbol in ("H", "Li", "F"):
            self.assertTrue(record["elements"][symbol]["shells"])
        self.assertEqual(self.driver._canonical_aux_name("def2-universal-jk"), canonical)
        for name in (None, canonical, "def2-svp-jk", "unknown-aux"):
            self.assertEqual(self.driver._canonical_aux_name(name), name)

    def test_cli_forwards_same_name_to_full_default_fold_and_reference(self):
        for name in ("def2-universal-jk", "def2-universal-jkfit",
                     "def2-svp-jk", "def2-tzvp-jk", "def2-tzvpp-jk",
                     "def2-qzvp-jk", "def2-qzvpp-jk", None):
            with self.subTest(name=name):
                calls, payloads = [], []

                def run(*args, **kwargs):
                    calls.append(kwargs["aux_basis"])
                    return SimpleNamespace(
                        hf_result=SimpleNamespace(aux_basis_name=kwargs["aux_basis"]),
                        n_cells=2, n_bf=2, n_occ=1, e_hf_per_cell=-1.,
                        e_corr_per_cell=-.01, e_t_per_cell=-.001, converged=True,
                        cc_result=SimpleNamespace(n_pairs=1), localisation="none",
                        source_kind="test-double", same_source=True, exchange_q0="ewald",
                        e_hf=-2., e_corr=-.02, e_t=-.002,
                    )

                def child(command, **kwargs):
                    payloads.append(json.loads(command[-1]))
                    if command[2] == self.driver._PYSCF_AUX_PREFLIGHT:
                        self.assertEqual(calls, [])
                        return SimpleNamespace(returncode=0, stderr="")
                    row = dict(e_hf=-1., e_corr_keep=-.01, e_corr_bare=-.01,
                               e_t_keep=-.001, e_t_bare=-.001, converged=True)
                    return SimpleNamespace(stdout="PYSCF-RESULT:" + json.dumps(
                        {"aux": payloads[-1]["aux"], "1x1x2": row}), stderr="")

                modules = {
                    "vibeqc.aux_basis": SimpleNamespace(default_aux_for=lambda basis: "def2-tzvp-jk"),
                    "vibeqc.periodic_dlpno": SimpleNamespace(run_periodic_dlpno_ccsd=run),
                    "vibeqc.periodic_toroidal_mp2": SimpleNamespace(
                        build_toroidal_supercell=lambda system, mesh: system),
                }
                with patch.dict(sys.modules, modules), \
                     patch.object(self.driver, "_full_options", return_value=None), \
                     patch.object(self.driver, "build_system", return_value=(object(), {}, "test")), \
                     patch.object(self.driver.subprocess, "run", side_effect=child), \
                     patch.object(sys, "argv", ["parity", "--meshes", "1,1,2"] +
                                  (["--aux", name] if name else [])), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(self.driver.main(), 0)
                selected = self.driver._canonical_aux_name(name) if name else "def2-tzvp-jk"
                self.assertEqual(calls, [selected] * 3)
                reference = selected + "fit" if selected.endswith("-jk") else selected
                self.assertEqual([p["aux"] for p in payloads], [reference] * 2)

    def test_preflight_failure_stops_before_vibeqc(self):
        with patch.object(self.driver, "build_system", return_value=(object(), {}, "test")), \
             patch.dict(sys.modules, {"vibeqc.periodic_toroidal_mp2": SimpleNamespace(
                 build_toroidal_supercell=Mock())}), \
             patch.object(self.driver, "vibeqc_runs") as run, \
             patch.object(self.driver.subprocess, "run", return_value=SimpleNamespace(
                 returncode=1, stderr="BasisNotFoundError: def2-tzvp-jkfit")), \
             patch.object(sys, "argv", ["parity", "--aux", "def2-tzvp-jk"]):
            with self.assertRaisesRegex(RuntimeError, "auxiliary preflight failed"):
                self.driver.main()
            run.assert_not_called()

    def test_preflight_loads_each_element_and_refuses_missing_blocks(self):
        payload = {"atoms": [["Li", [0, 0, 0]], ["H", [1, 0, 0]], ["H", [2, 0, 0]]],
                   "aux": "def2-tzvp-jkfit"}
        loader = Mock(return_value=[[0, [1., 1.]]])
        with patch.dict(sys.modules, {"pyscf.gto": SimpleNamespace(
                basis=SimpleNamespace(load=loader))}), \
             patch.object(sys, "argv", ["child", json.dumps(payload)]):
            exec(self.driver._PYSCF_AUX_PREFLIGHT, {})
            self.assertEqual([call.args for call in loader.call_args_list],
                             [("def2-tzvp-jkfit", "H"), ("def2-tzvp-jkfit", "Li")])
            for response in ([], ValueError("unknown basis")):
                loader.reset_mock()
                loader.side_effect = response if isinstance(response, Exception) else None
                loader.return_value = response
                with self.assertRaisesRegex(RuntimeError, "No default auxiliary substitution"):
                    exec(self.driver._PYSCF_AUX_PREFLIGHT, {})

    def test_unresolved_default_is_not_forwarded(self):
        with self.assertRaisesRegex(ValueError, "resolve the Vibe-QC auxiliary"):
            self.driver._pyscf_payload({}, "pob-tzvp", [], None)


if __name__ == "__main__":
    unittest.main()
