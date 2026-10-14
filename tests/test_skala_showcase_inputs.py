"""Standalone CLI/occupation controls; no native core or checkpoint import.

Run with a NumPy-capable Python: python tests/test_skala_showcase_inputs.py
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "studies/skala-showcase/beh_bipole_uks.py"
spec = importlib.util.spec_from_file_location("beh_input", INPUT)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def result(mesh_size=1):
    return SimpleNamespace(
        converged=True, energy=-15.0,
        kpoint_weights=np.ones(mesh_size) / mesh_size,
        occupations_alpha=np.tile([1., 1., 1., 0.], (mesh_size, 1)),
        occupations_beta=np.tile([1., 1., 0., 0.], (mesh_size, 1)),
    )


class TestSkalaShowcaseInput(unittest.TestCase):
    def test_weighted_spin_counts(self):
        for n in (1, 2, 8):
            with self.subTest(kpoints=n):
                self.assertEqual(example.spin_occupation_counts(result(n)), (3., 2.))
        value = result(2)
        value.kpoint_weights = np.array([0.25, 0.75])
        value.occupations_alpha = np.array([[0., 0., 0., 0.], [1., 1., 1., 1.]])
        value.occupations_beta = np.array([[1., 1., 1., 1.], [1., 1. / 3, 0., 0.]])
        np.testing.assert_allclose(example.spin_occupation_counts(value), [3., 2.], rtol=0, atol=1e-14)

    def test_wrong_or_malformed_spin_state_refused(self):
        base = result(2)
        mutations = [
            {"occupations_beta": base.occupations_alpha.copy()},
            {"occupations_alpha": base.occupations_beta.copy(),
             "occupations_beta": base.occupations_alpha.copy()},
            {"occupations_alpha": np.full((2, 4), np.nan)},
            {"occupations_beta": np.full((2, 4), np.inf)},
            {"kpoint_weights": np.array([1., 1.])},
            {"kpoint_weights": np.array([np.nan, 0.5])},
            {"kpoint_weights": np.array([-0.5, 1.5])},
            {"occupations_alpha": np.array([[2., 1., 0., 0.]] * 2)},
            {"occupations_beta": np.array([[-1., 1., 1., 1.]] * 2)},
            {"occupations_beta": np.ones((1, 4))},
            {"kpoint_weights": np.ones((2, 1)) / 2},
            {"occupations_alpha": [], "occupations_beta": []},
        ]
        for changes in mutations:
            with self.subTest(changes=tuple(changes)):
                value = copy.deepcopy(base)
                for key, replacement in changes.items():
                    setattr(value, key, replacement)
                with self.assertRaises(RuntimeError):
                    example.spin_occupation_counts(value)

    def test_bookkeeping_tolerance_has_no_relative_slack(self):
        value = result()
        value.occupations_alpha[0, 3] = 2e-8
        with self.assertRaises(RuntimeError):
            example.spin_occupation_counts(value)

    def test_invalid_cli_refused(self):
        cases = [[], *[["--bipole-cutoff-bohr", v] for v in ("nan", "inf", "0", "-1")],
                 ["--bipole-cutoff-bohr", "12", "--kpoints", "0", "1", "1"],
                 ["--bipole-cutoff-bohr", "12", "--kpoints", "1", "2"],
                 ["--bipole-cutoff-bohr", "12", "--kpoints", "1.5", "1", "1"]]
        for args in cases:
            with self.subTest(args=args), patch.object(sys, "argv", [str(INPUT), *args]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    example.parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_runtime_refuses_unsupported_platform_and_python(self):
        for version, platform in [((3, 14), "linux"), ((3, 13), "darwin")]:
            with self.subTest(version=version, platform=platform):
                with patch.object(sys, "version_info", version), patch.object(sys, "platform", platform):
                    with self.assertRaises(SystemExit):
                        example.require_supported_runtime()

    def test_complete_mesh_and_spin_forwarding(self):
        class System:
            def __init__(self, **kwargs): self.data = kwargs
            def unit_cell_molecule(self): return self

        calls = []
        fake = SimpleNamespace(
            PeriodicSystem=System, Atom=lambda z, xyz: (z, xyz),
            BasisSet=lambda system, name: (system, name),
            run_periodic_job=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        for mesh in ([1, 1, 1], [2, 1, 1], [2, 2, 2]):
            with self.subTest(mesh=mesh), tempfile.TemporaryDirectory() as tmp:
                args = [str(INPUT), "--dry-run", "--bipole-cutoff-bohr", "12",
                        "--kpoints", *map(str, mesh), "--output", str(Path(tmp) / "result")]
                with patch.object(sys, "argv", args), patch.dict(sys.modules, {"vibeqc": fake}):
                    with contextlib.redirect_stdout(io.StringIO()):
                        example.main()
                positional, options = calls[-1]
                self.assertEqual(positional[0].data["multiplicity"], 2)
                self.assertEqual(sum(z for z, _ in positional[0].data["unit_cell"]), 5)
                np.testing.assert_array_equal(positional[0].data["lattice"], np.eye(3) * 12)
                self.assertEqual(positional[1][1], "sto-3g")
                for key, value in {"method": "UKS", "functional": "skala-1.1",
                                   "jk_method": "bipole", "kpoints": mesh,
                                   "smearing_temperature": 0., "use_exchange_ewald_split": True,
                                   "bipole_cutoff_bohr": 12., "bipole_nuclear_cutoff_bohr": 12.,
                                   "write_population_file": False, "dry_run": True}.items():
                    self.assertEqual(options[key], value)

    def test_mesh_output_separation_and_reuse_refusal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VQ_WORKDIR": tmp}):
            gamma = example.reserve_output_stem(None, [1, 1, 1])
            multi = example.reserve_output_stem(None, [2, 1, 1])
            self.assertNotEqual(gamma.parent, multi.parent)
            with self.assertRaises(FileExistsError):
                example.reserve_output_stem(None, [2, 1, 1])

    def test_real_run_requires_convergence_finite_energy_and_spin_counts(self):
        class System:
            def __init__(self, **kwargs): pass
            def unit_cell_molecule(self): return self

        cases = [result(), None, result(), result(), result()]
        cases[2].converged = False
        cases[3].energy = float("nan")
        cases[4].occupations_beta = cases[4].occupations_alpha.copy()
        for index, value in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as tmp:
                fake = SimpleNamespace(
                    PeriodicSystem=System, Atom=lambda z, xyz: (z, xyz),
                    BasisSet=lambda *args: None, run_periodic_job=lambda *a, **kw: value,
                )
                args = [str(INPUT), "--bipole-cutoff-bohr", "12", "--output", str(Path(tmp) / "result")]
                with patch.object(sys, "argv", args), patch.dict(sys.modules, {"vibeqc": fake}):
                    with patch.object(example, "require_supported_runtime"), contextlib.redirect_stdout(io.StringIO()):
                        if index == 0:
                            example.main()
                        else:
                            with self.assertRaises(RuntimeError):
                                example.main()


if __name__ == "__main__":
    unittest.main()
