"""Import and lifecycle contracts without loading the native package."""

from __future__ import annotations

import builtins
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy  # Load once before restoring the isolated module registry.


_MODULE_PATH = Path(__file__).resolve().parents[1] / "python/vibeqc/mpi.py"
_SERIAL_STEP = {
    "SLURM_PROCID": "0",
    "SLURM_STEP_ID": "0",
    "SLURM_STEP_NUM_TASKS": "1",
}


def _load_module(environ, mpi=None):
    """Execute the maintained module with isolated launch/import boundaries."""
    name = "_mpi_environment_test_module"
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__
    imports = []

    def isolated_import(name, *args, **kwargs):
        if name == "mpi4py":
            imports.append(name)
            if mpi is None:
                raise ImportError("mpi4py intentionally unavailable")
            return SimpleNamespace(MPI=mpi)
        return original_import(name, *args, **kwargs)

    with patch.dict(os.environ, environ, clear=True), patch.dict(
        sys.modules, {name: module}
    ), patch("builtins.__import__", side_effect=isolated_import):
        spec.loader.exec_module(module)
    return module, imports


def _fake_mpi(rank=0, size=1, *, initialized=True, finalized=False, thread=1):
    comm = SimpleNamespace(Get_rank=lambda: rank, Get_size=lambda: size)
    return SimpleNamespace(
        COMM_WORLD=comm,
        THREAD_FUNNELED=1,
        Query_thread=lambda: thread,
        Is_initialized=lambda: initialized,
        Is_finalized=lambda: finalized,
    )


class MPIEnvironmentTests(unittest.TestCase):
    def assert_serial(self, env):
        module, imports = _load_module(env)
        world = module.mpi_world()
        self.assertEqual((world.rank, world.size, world.active), (0, 1, False))
        self.assertIsNone(world.comm)
        self.assertEqual(imports, [])
        self.assertFalse(module.mpi_available())
        self.assertEqual(module.mpi_bcast("serial"), "serial")
        self.assertEqual(module.mpi_gather("serial"), ["serial"])

    def test_single_task_slurm_only_import_is_serial_without_mpi4py(self):
        self.assert_serial(_SERIAL_STEP)

    def test_single_task_step_uses_step_size_in_larger_allocation(self):
        self.assert_serial({**_SERIAL_STEP, "SLURM_NTASKS": "8"})

    def test_plain_and_batch_import_are_serial(self):
        for env in ({}, {"SLURM_PROCID": "0", "SLURM_STEP_ID": "batch"}):
            with self.subTest(env=env):
                self.assert_serial(env)

    def test_single_task_slurm_does_not_initialize_installed_mpi(self):
        module, imports = _load_module(_SERIAL_STEP, _fake_mpi())
        self.assertIsNone(module.mpi_world().comm)
        self.assertEqual(imports, [])

    def test_explicit_and_mpi_rank_evidence_still_require_mpi4py(self):
        controls = [
            {"VIBEQC_MPI_REQUIRED": "1", "VIBEQC_MPI_EXPECTED_SIZE": "1"},
            {"PMI_RANK": "0", "PMI_SIZE": "1"},
            {"OMPI_COMM_WORLD_RANK": "0", "OMPI_COMM_WORLD_SIZE": "1"},
            {"MV2_COMM_WORLD_RANK": "0", "MV2_COMM_WORLD_SIZE": "1"},
            {"PMIX_RANK": "0", "PMIX_NAMESPACE": "test", "PMIX_SERVER_URI2": "local://server"},
        ]
        for control in controls:
            for env in (control, {**_SERIAL_STEP, **control}):
                with self.subTest(env=env):
                    with self.assertRaisesRegex(RuntimeError, "mpi4py.MPI could not be loaded"):
                        _load_module(env)

    def test_multitask_slurm_cannot_fall_back_to_serial(self):
        for rank in ("0", "1"):
            with self.subTest(rank=rank):
                with self.assertRaisesRegex(RuntimeError, "mpi4py.MPI could not be loaded"):
                    _load_module({**_SERIAL_STEP, "SLURM_PROCID": rank, "SLURM_STEP_NUM_TASKS": "2"})

    def test_malformed_incomplete_and_conflicting_metadata_refuse(self):
        controls = [
            ({**_SERIAL_STEP, "SLURM_STEP_ID": "bad"}, "SLURM_STEP_ID"),
            ({**_SERIAL_STEP, "SLURM_PROCID": "bad"}, "SLURM_PROCID"),
            ({**_SERIAL_STEP, "SLURM_PROCID": "1"}, "outside world size"),
            ({**_SERIAL_STEP, "SLURM_STEP_NUM_TASKS": "0"}, "SLURM_STEP_NUM_TASKS"),
            ({"SLURM_PROCID": "0", "SLURM_STEP_ID": "0"}, "incomplete"),
            ({**_SERIAL_STEP, "PMI_RANK": "0"}, "incomplete"),
            ({**_SERIAL_STEP, "PMIX_RANK": "0"}, "incomplete"),
            ({**_SERIAL_STEP, "PMI_RANK": "0", "PMI_SIZE": "2"}, "world-size variables disagree"),
            ({**_SERIAL_STEP, "PMI_RANK": "1", "PMI_SIZE": "2"}, "rank variables disagree"),
            ({**_SERIAL_STEP, "VIBEQC_MPI_REQUIRED": "1"}, "EXPECTED_SIZE"),
            ({**_SERIAL_STEP, "VIBEQC_MPI_REQUIRED": "invalid"}, "must be 0 or 1"),
        ]
        for env, message in controls:
            with self.subTest(env=env):
                with self.assertRaisesRegex(RuntimeError, message):
                    _load_module(env)

    def test_complete_mpi_still_tolerates_partial_auxiliary_family(self):
        env = {"OMPI_COMM_WORLD_RANK": "1", "OMPI_COMM_WORLD_SIZE": "2", "PMI_RANK": "1"}
        module, imports = _load_module(env, _fake_mpi(rank=1, size=2))
        world = module.mpi_world()
        self.assertEqual((world.rank, world.size, world.active), (1, 2, True))
        self.assertEqual(imports, ["mpi4py"])

    def test_detected_launch_lifecycle_refuses_invalid_runtime(self):
        env = {"PMI_RANK": "0", "PMI_SIZE": "2"}
        controls = [
            (_fake_mpi(size=2, initialized=False), "automatic initialization"),
            (_fake_mpi(size=2, finalized=True), "was finalized"),
            (_fake_mpi(size=2, thread=0), "inadequate thread level"),
            (_fake_mpi(rank=1, size=2), "rank contradicts"),
            (_fake_mpi(size=1), "size contradicts"),
        ]
        for mpi, message in controls:
            with self.subTest(message=message):
                module, _ = _load_module(env, mpi)
                with self.assertRaisesRegex(RuntimeError, message):
                    module.mpi_world()

    def test_cached_world_rejects_finalized_runtime(self):
        mpi = _fake_mpi(size=2)
        module, _ = _load_module({"PMI_RANK": "0", "PMI_SIZE": "2"}, mpi)
        self.assertTrue(module.mpi_world().active)
        mpi.Is_finalized = lambda: True
        with self.assertRaisesRegex(RuntimeError, "finalized after"):
            module.mpi_world()


if __name__ == "__main__":
    unittest.main()
