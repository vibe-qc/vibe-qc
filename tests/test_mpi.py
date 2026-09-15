"""Tests for vibeqc.mpi — MPI parallelization infrastructure.

Tests run correctly in both serial (no MPI) and parallel (mpirun)
modes. When MPI is not active, all distribution functions return
the full range to rank 0.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import vibeqc as vq


mpi_module = importlib.import_module("vibeqc.mpi")

_MPI_LAUNCH_ENV_NAMES = {
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
    "MV2_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_PROCID",
    "SLURM_STEP_ID",
    "SLURM_STEP_NUM_TASKS",
    "SLURM_NTASKS",
    "PMIX_RANK",
    "PMIX_NAMESPACE",
    "VIBEQC_MPI_REQUIRED",
    "VIBEQC_MPI_EXPECTED_SIZE",
}


def _unlaunched_environment() -> dict[str, str]:
    env = dict(os.environ)
    for name in _MPI_LAUNCH_ENV_NAMES:
        env.pop(name, None)
    for name in tuple(env):
        if name.startswith("PMIX_SERVER_URI"):
            env.pop(name)
    return env


def _run_python(code: str, *, env: dict[str, str], cwd=None):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=cwd,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None,
    reason="needs the [mpi] extra to prove it remains inactive",
)
def test_plain_import_and_serial_helpers_do_not_load_or_initialize_mpi():
    """Installing mpi4py must not turn a plain process into MPI."""
    result = _run_python(
        """
        import sys
        import vibeqc as vq

        assert "mpi4py.MPI" not in sys.modules
        world = vq.mpi_world()
        assert world.comm is None
        assert world.rank == 0
        assert world.size == 1
        assert world.active is False
        assert vq.mpi_rank() == 0
        assert vq.mpi_size() == 1
        assert vq.mpi_available() is False
        assert vq.mpi_bcast({"serial": True}) == {"serial": True}
        assert vq.mpi_gather("x") == ["x"]
        assert "mpi4py.MPI" not in sys.modules

        import mpi4py
        mpi4py.rc.initialize = False
        from mpi4py import MPI

        assert MPI.Is_initialized() is False
        """,
        env=_unlaunched_environment(),
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    ("env", "source", "rank", "size"),
    [
        (
            {"OMPI_COMM_WORLD_RANK": "1", "OMPI_COMM_WORLD_SIZE": "3"},
            "Open MPI",
            1,
            3,
        ),
        (
            {"PMI_RANK": "0", "PMI_SIZE": "2"},
            "PMI/Intel MPI/MPICH",
            0,
            2,
        ),
        (
            {"MV2_COMM_WORLD_RANK": "2", "MV2_COMM_WORLD_SIZE": "4"},
            "MVAPICH",
            2,
            4,
        ),
        (
            {
                "SLURM_PROCID": "3",
                "SLURM_STEP_ID": "0",
                "SLURM_STEP_NUM_TASKS": "5",
                "SLURM_NTASKS": "8",
            },
            "Slurm step",
            3,
            5,
        ),
        (
            {
                "PMIX_RANK": "1",
                "PMIX_NAMESPACE": "job.1",
                "PMIX_SERVER_URI2": "local://server",
            },
            "PMIx",
            1,
            None,
        ),
        (
            {
                "VIBEQC_MPI_REQUIRED": "1",
                "VIBEQC_MPI_EXPECTED_SIZE": "2",
            },
            "vibe-qc required launcher",
            None,
            2,
        ),
    ],
)
def test_mpi_launch_environment_accepts_strong_rank_evidence(
    env, source, rank, size
):
    evidence = mpi_module._mpi_launch_environment(env)
    assert evidence is not None
    assert evidence.sources == (source,)
    assert evidence.rank == rank
    assert evidence.size == size


@pytest.mark.parametrize(
    "env",
    [
        {"PBS_NP": "8"},
        {"SLURM_NTASKS": "8"},
        {"SLURM_PROCID": "0"},
        {
            "SLURM_PROCID": "0",
            "SLURM_STEP_ID": "batch",
            "SLURM_NTASKS": "8",
        },
        {
            "SLURM_PROCID": "0",
            "SLURM_STEP_ID": "",
            "SLURM_NTASKS": "8",
        },
        {"SLURM_PROCID": "0", "SLURM_STEP_ID": "interactive"},
        {"SLURM_PROCID": "0", "SLURM_STEP_ID": "pending"},
        # Current Slurm uses 0xfffffffa..fd for special steps; older
        # releases also used 0xfffffffe/ff. All values above the stable
        # SLURM_MAX_NORMAL_STEP_ID boundary are reserved.
        {"SLURM_PROCID": "0", "SLURM_STEP_ID": "4294967281"},
        {"SLURM_PROCID": "0", "SLURM_STEP_ID": "4294967290"},
        {"SLURM_PROCID": "0", "SLURM_STEP_ID": "4294967293"},
        {"SLURM_PROCID": "0", "SLURM_STEP_ID": "4294967294"},
        {"SLURM_PROCID": "0", "SLURM_STEP_ID": "4294967295"},
        {"PMI_SIZE": "8"},
        {"OMPI_COMM_WORLD_SIZE": "8"},
        {"I_MPI_ROOT": "/opt/intel"},
        {"OMPI_VERSION": "5"},
        {"PMIX_VERSION": "5"},
    ],
)
def test_mpi_launch_environment_ignores_allocation_only_evidence(env):
    assert mpi_module._mpi_launch_environment(env) is None


def test_complete_launcher_evidence_tolerates_partial_auxiliary_family():
    evidence = mpi_module._mpi_launch_environment(
        {
            "OMPI_COMM_WORLD_RANK": "1",
            "OMPI_COMM_WORLD_SIZE": "2",
            "PMI_RANK": "1",
        }
    )
    assert evidence is not None
    assert evidence.sources == ("Open MPI",)
    assert evidence.rank == 1
    assert evidence.size == 2


def test_scheduler_pmi_evidence_and_required_size_must_agree():
    evidence = mpi_module._mpi_launch_environment(
        {
            "PMI_RANK": "1",
            "PMI_SIZE": "2",
            "VIBEQC_MPI_REQUIRED": "1",
            "VIBEQC_MPI_EXPECTED_SIZE": "2",
        }
    )
    assert evidence is not None
    assert evidence.rank == 1
    assert evidence.size == 2
    assert evidence.sources == (
        "PMI/Intel MPI/MPICH",
        "vibe-qc required launcher",
    )

    with pytest.raises(RuntimeError, match="world-size variables disagree"):
        mpi_module._mpi_launch_environment(
            {
                "PMI_RANK": "1",
                "PMI_SIZE": "2",
                "VIBEQC_MPI_REQUIRED": "1",
                "VIBEQC_MPI_EXPECTED_SIZE": "3",
            }
        )


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PMI_RANK": "0"}, "incomplete"),
        ({"PMI_RANK": "rank", "PMI_SIZE": "2"}, "PMI_RANK"),
        ({"PMI_RANK": "2", "PMI_SIZE": "2"}, "outside world size"),
        (
            {
                "SLURM_PROCID": "0",
                "SLURM_STEP_ID": "not-a-step",
                "SLURM_STEP_NUM_TASKS": "2",
            },
            "SLURM_STEP_ID",
        ),
        ({"VIBEQC_MPI_REQUIRED": "1"}, "EXPECTED_SIZE"),
        (
            {
                "PMI_RANK": "0",
                "PMI_SIZE": "2",
                "OMPI_COMM_WORLD_RANK": "1",
                "OMPI_COMM_WORLD_SIZE": "2",
            },
            "rank variables disagree",
        ),
    ],
)
def test_mpi_launch_environment_rejects_unsafe_evidence(env, message):
    with pytest.raises(RuntimeError, match=message):
        mpi_module._mpi_launch_environment(env)


def test_detected_launcher_without_mpi4py_fails_closed(tmp_path):
    """A launched rank must never degrade to an independent serial job."""
    poison = tmp_path / "mpi4py"
    poison.mkdir()
    (poison / "__init__.py").write_text(
        'raise ImportError("poison mpi4py")\n', encoding="utf-8"
    )
    env = _unlaunched_environment()
    env.update({"PMI_RANK": "0", "PMI_SIZE": "1"})
    result = _run_python(
        """
        try:
            import vibeqc  # noqa: F401
        except RuntimeError as exc:
            assert "mpi4py.MPI could not be loaded" in str(exc)
        else:
            raise AssertionError("detected MPI launch silently became serial")
        """,
        env=env,
        cwd=tmp_path,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None,
    reason="needs the [mpi] extra to exercise disabled initialization",
)
def test_detected_launcher_with_disabled_auto_init_fails_closed():
    env = _unlaunched_environment()
    env.update(
        {
            "MPI4PY_RC_INITIALIZE": "0",
            "PMI_RANK": "0",
            "PMI_SIZE": "1",
        }
    )
    result = _run_python(
        """
        import vibeqc as vq
        from mpi4py import MPI

        assert MPI.Is_initialized() is False
        try:
            vq.mpi_world()
        except RuntimeError as exc:
            assert "automatic initialization is disabled or failed" in str(exc)
        else:
            raise AssertionError("uninitialized MPI launch silently became serial")
        assert MPI.Is_initialized() is False
        """,
        env=env,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_cached_mpi_world_rejects_finalized_runtime(monkeypatch):
    class FinalizedMPI:
        @staticmethod
        def Is_finalized():
            return True

    monkeypatch.setattr(
        mpi_module,
        "_MPI_LAUNCH_EVIDENCE",
        mpi_module._MPILaunchEvidence(("test launcher",), 0, 2),
    )
    monkeypatch.setattr(mpi_module, "_MPI", FinalizedMPI())
    monkeypatch.setattr(
        mpi_module,
        "_mpi_world",
        vq.MPIWorld(comm=None, rank=0, size=2, active=True),
    )
    with pytest.raises(RuntimeError, match="finalized after"):
        mpi_module.mpi_world()


def test_mpi_available_returns_bool():
    """mpi_available() returns a boolean."""
    assert isinstance(vq.mpi_available(), bool)


def test_mpi_rank_is_non_negative():
    """mpi_rank() returns a non-negative integer."""
    r = vq.mpi_rank()
    assert isinstance(r, int)
    assert r >= 0


def test_mpi_size_is_positive():
    """mpi_size() returns a positive integer (>= 1)."""
    s = vq.mpi_size()
    assert isinstance(s, int)
    assert s >= 1


def test_mpi_world_is_mpi_world():
    """mpi_world() returns an MPIWorld instance."""
    w = vq.mpi_world()
    assert isinstance(w, vq.MPIWorld)
    assert w.rank == vq.mpi_rank()
    assert w.size == vq.mpi_size()


def test_mpi_barrier_noop_when_inactive():
    """mpi_barrier() is a no-op when MPI is not active."""
    vq.mpi_barrier()  # Should not raise


def test_mpi_allreduce_identity_serial():
    """In serial mode, allreduce returns the input unchanged."""
    data = np.array([1.0, 2.0, 3.0])
    result = vq.mpi_allreduce(data, op="sum")
    np.testing.assert_array_equal(result, data)


def test_mpi_reduce_sum_serial():
    """In serial mode, reduce_sum returns the input."""
    data = np.array([1.0, 2.0])
    result = vq.mpi_reduce_sum(data)
    np.testing.assert_array_equal(result, data)


def test_mpi_bcast_serial():
    """In serial mode, bcast returns the input unchanged."""
    data = {"key": [1, 2, 3]}
    result = vq.mpi_bcast(data)
    assert result == data


def test_mpi_gather_serial():
    """In serial mode, gather returns a single-element list."""
    data = np.array([42.0])
    result = vq.mpi_gather(data)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], data)


def test_mpi_allgather_serial():
    """In serial mode, allgather returns a single-element list."""
    data = np.array([7.0])
    result = vq.mpi_allgather(data)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], data)


# ---------------------------------------------------------------------------
# Shell pair distribution
# ---------------------------------------------------------------------------


def test_distribute_shell_pairs_coverage():
    """Shell pair distribution covers all pairs exactly once."""
    if not vq.mpi_available():
        pytest.skip("MPI not active — serial-only test")
    n_shells = 20
    ranges = vq.mpi_allgather(vq.distribute_shell_pairs(n_shells))
    # Reconstruct coverage
    covered = set()
    for sp_range in ranges:
        for idx in range(sp_range.start, sp_range.end):
            covered.add(idx)
    n_pairs = n_shells * (n_shells + 1) // 2
    assert len(covered) == n_pairs
    assert covered == set(range(n_pairs))


def test_distribute_shell_pairs_serial():
    """In serial mode, all pairs go to rank 0."""
    sp = vq.distribute_shell_pairs(10)
    assert sp.rank == 0
    assert sp.start == 0
    n_pairs = 10 * 11 // 2  # 55
    assert sp.end == n_pairs
    assert sp.total == n_pairs
    assert sp.count == n_pairs


def test_distribute_shell_pairs_zero_shells():
    """Zero shells → zero pairs."""
    sp = vq.distribute_shell_pairs(0)
    assert sp.total == 0
    assert sp.count == 0


# ---------------------------------------------------------------------------
# Grid slab partition
# ---------------------------------------------------------------------------


def test_grid_slab_partition_serial():
    """In serial mode, the slab is the entire grid."""
    slab = vq.grid_slab_partition(64, 64, 128)
    assert slab.rank == 0
    assert slab.nx == 64
    assert slab.ny == 64
    assert slab.nz_local == 128
    assert slab.z_start == 0
    assert slab.z_end == 128
    assert slab.nz_global == 128
    assert slab.local_shape == (64, 64, 128)


def test_grid_slab_partition_even():
    """Even division of z-planes."""
    n_ranks = vq.mpi_size()
    nz = 120
    # Just test that the slab is well-formed
    slab = vq.grid_slab_partition(32, 32, nz)
    assert slab.nx == 32
    assert slab.ny == 32
    assert slab.nz_local >= 0
    assert slab.z_end - slab.z_start == slab.nz_local
    assert slab.z_start >= 0
    assert slab.z_end <= nz


# ---------------------------------------------------------------------------
# Task distribution
# ---------------------------------------------------------------------------


def test_distribute_tasks_serial():
    """In serial mode, all tasks go to rank 0."""
    start, end = vq.distribute_tasks(100)
    assert start == 0
    assert end == 100


def test_distribute_tasks_zero():
    """Zero tasks → empty range."""
    start, end = vq.distribute_tasks(0)
    assert start == 0
    assert end == 0


def test_distribute_tasks_coverage():
    """Task distribution covers all tasks exactly once."""
    if not vq.mpi_available():
        pytest.skip("MPI not active — serial-only test")
    n_tasks = 97  # prime, to exercise remainder logic
    ranges = vq.mpi_allgather(vq.distribute_tasks(n_tasks))
    covered = set()
    for s, e in ranges:
        for t in range(s, e):
            covered.add(t)
    assert len(covered) == n_tasks
    assert covered == set(range(n_tasks))


# ---------------------------------------------------------------------------
# Allreduce edge cases
# ---------------------------------------------------------------------------


def test_allreduce_invalid_op():
    """Invalid reduction operation raises ValueError when MPI is active."""
    if not vq.mpi_available():
        pytest.skip("MPI not active — invalid op passes through in serial")
    data = np.array([1.0])
    with pytest.raises(ValueError, match="Unknown.*reduction"):
        vq.mpi_allreduce(data, op="invalid_op")


# ---------------------------------------------------------------------------
# KPointPartition
# ---------------------------------------------------------------------------
# The split is the easy half; the reassembly is where k-point farming goes
# wrong, because a plain allgather returns results in RANK order while
# callers need GLOBAL k order. These pin both, in serial here and under a
# real mpirun below.


@pytest.mark.parametrize("strategy", ["block", "cyclic"])
def test_kpoint_partition_serial_is_identity(strategy):
    """Serial mode owns everything, in order, and gathers unchanged --
    so a non-MPI run behaves exactly as it did before."""
    part = vq.KPointPartition.create(7, strategy=strategy)
    assert part.n_kpoints == 7
    assert part.n_local == 7
    assert part.local_indices == tuple(range(7))
    assert part.allgather_ordered(list("abcdefg")) == list("abcdefg")
    assert all(part.owns(k) for k in range(7))


@pytest.mark.parametrize("strategy", ["block", "cyclic"])
@pytest.mark.parametrize("n_kpoints", [0, 1, 5, 8, 13])
def test_kpoint_partition_owner_matches_membership(
    monkeypatch, strategy, n_kpoints
):
    """owner_of() is computed arithmetically rather than by
    communication, so it must agree with the membership every rank would
    compute for itself. Emulate each rank by overriding the world."""
    for size in (1, 2, 3, 5):
        assignments: dict[int, int] = {}
        for rank in range(size):
            monkeypatch.setattr(
                vq.mpi, "_mpi_world",
                vq.MPIWorld(comm=None, rank=rank, size=size, active=size > 1),
            )
            part = vq.KPointPartition.create(n_kpoints, strategy=strategy)
            for k in part.local_indices:
                assert k not in assignments, f"k={k} owned twice"
                assignments[k] = rank
                assert part.owns(k)
        # Every point assigned to exactly one rank, none dropped.
        assert sorted(assignments) == list(range(n_kpoints))
        # owner_of is arithmetic, not communication: asked on rank 0 it
        # must still name the rank that actually took each point.
        monkeypatch.setattr(
            vq.mpi, "_mpi_world",
            vq.MPIWorld(comm=None, rank=0, size=size, active=size > 1),
        )
        part0 = vq.KPointPartition.create(n_kpoints, strategy=strategy)
        for k, owning_rank in assignments.items():
            assert part0.owner_of(k) == owning_rank


def test_kpoint_partition_rejects_mismatched_local_values():
    """A local list that does not line up with local_indices is a
    programming error and must not be silently zipped short."""
    part = vq.KPointPartition.create(4)
    with pytest.raises(ValueError, match="local values"):
        part.allgather_ordered([1, 2])


def test_kpoint_partition_rejects_bad_strategy_and_negative():
    with pytest.raises(ValueError, match="strategy"):
        vq.KPointPartition.create(4, strategy="spiral")
    with pytest.raises(ValueError, match="n_kpoints"):
        vq.KPointPartition.create(-1)


def test_kpoint_partition_owner_of_range_checked():
    part = vq.KPointPartition.create(3)
    with pytest.raises(IndexError):
        part.owner_of(3)
    with pytest.raises(IndexError):
        part.owner_of(-1)


def test_kpoint_partition_oversubscribed_ranks_get_empty(monkeypatch):
    """More ranks than k-points is a scheduling choice, not an error."""
    monkeypatch.setattr(
        vq.mpi, "_mpi_world",
        vq.MPIWorld(comm=None, rank=3, size=4, active=True),
    )
    part = vq.KPointPartition.create(2, strategy="block")
    assert part.local_indices == ()
    assert part.n_local == 0


# ---------------------------------------------------------------------------
# Complete real-space output-block distribution
# ---------------------------------------------------------------------------


def _cell_task_keys(n_tasks):
    return [(index, -index, index % 3) for index in range(n_tasks)]


@pytest.mark.parametrize("strategy", ["block", "cyclic"])
def test_lattice_output_partition_serial_is_identity(strategy):
    keys = _cell_task_keys(7)
    part = vq.LatticeOutputPartition.create(keys, strategy=strategy)
    assert part.task_keys == tuple(keys)
    assert part.local_indices == tuple(range(7))
    assert part.local_task_keys == tuple(keys)
    assert part.task_counts == (7,)
    assert part.allgather_ordered(list("abcdefg")) == list("abcdefg")


@pytest.mark.parametrize("strategy", ["block", "cyclic"])
@pytest.mark.parametrize("n_tasks", [0, 1, 2, 5, 7, 13])
def test_lattice_output_partition_exact_coverage(
    monkeypatch, strategy, n_tasks
):
    keys = _cell_task_keys(n_tasks)
    for size in (1, 2, 3, 4, 8):
        owners = {}
        counts = []
        for rank in range(size):
            monkeypatch.setattr(
                vq.mpi,
                "_mpi_world",
                vq.MPIWorld(
                    comm=None,
                    rank=rank,
                    size=size,
                    active=size > 1,
                ),
            )
            part = vq.LatticeOutputPartition.create(
                keys, strategy=strategy
            )
            counts.append(part.n_local)
            for task in part.local_indices:
                assert task not in owners
                owners[task] = rank
                assert part.owner_of(task) == rank
        assert sorted(owners) == list(range(n_tasks))
        assert tuple(counts) == part.task_counts


def test_lattice_output_partition_rejects_bad_contract():
    with pytest.raises(ValueError, match="unique"):
        vq.LatticeOutputPartition.create([(0, 0, 0), (0, 0, 0)])
    with pytest.raises(ValueError, match="three integers"):
        vq.LatticeOutputPartition.create([(0, 0)])
    with pytest.raises(ValueError, match="three integers"):
        vq.LatticeOutputPartition.create([(0, True, 0)])
    with pytest.raises(ValueError, match="strategy"):
        vq.LatticeOutputPartition.create([(0, 0, 0)], strategy="random")


class _ParcelComm:
    def __init__(self, parcels):
        self.parcels = parcels

    def allgather(self, _local):
        return self.parcels


def _partition_parcel(part, source, values):
    return (
        part._SCHEMA,
        part.n_tasks,
        part.strategy,
        part.task_fingerprint,
        source,
        part._indices_for_rank(
            part.n_tasks, source, part.size, part.strategy
        ),
        values,
    )


def test_lattice_output_partition_gathers_by_global_task(monkeypatch):
    world = vq.MPIWorld(comm=None, rank=0, size=3, active=True)
    monkeypatch.setattr(vq.mpi, "_mpi_world", world)
    part = vq.LatticeOutputPartition.create(
        _cell_task_keys(7), strategy="cyclic"
    )
    parcels = []
    for source in range(3):
        indices = part._indices_for_rank(7, source, 3, "cyclic")
        parcels.append(
            _partition_parcel(
                part, source, [f"task-{index}" for index in indices]
            )
        )
    world.comm = _ParcelComm(list(reversed(parcels)))
    local = [f"task-{index}" for index in part.local_indices]
    assert part.allgather_ordered(local) == [
        f"task-{index}" for index in range(7)
    ]


@pytest.mark.parametrize(
    "mutate",
    ["duplicate-source", "wrong-ownership", "short-values", "fingerprint"],
)
def test_lattice_output_partition_rejects_adversarial_parcels(
    monkeypatch, mutate
):
    world = vq.MPIWorld(comm=None, rank=0, size=2, active=True)
    monkeypatch.setattr(vq.mpi, "_mpi_world", world)
    part = vq.LatticeOutputPartition.create(
        _cell_task_keys(4), strategy="cyclic"
    )
    parcels = [
        list(_partition_parcel(part, source, [source, source]))
        for source in range(2)
    ]
    if mutate == "duplicate-source":
        parcels[1][4] = 0
    elif mutate == "wrong-ownership":
        parcels[1][5] = (0, 3)
    elif mutate == "short-values":
        parcels[1][6] = [1]
    else:
        parcels[1][3] = "not-the-same-cell-order"
    world.comm = _ParcelComm([tuple(parcel) for parcel in parcels])
    with pytest.raises(RuntimeError):
        part.allgather_ordered([0, 0])
