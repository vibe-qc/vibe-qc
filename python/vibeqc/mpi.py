"""MPI parallelization infrastructure for vibe-qc.

Provides domain-decomposition, shell-pair distribution, and
collective operations for parallel SCF, gradients, and TDDFT
calculations across multiple MPI ranks.

Architecture
------------

The MPI layer is optional -- vibe-qc works perfectly fine without
it (single-rank, OpenMP-only). Installing ``mpi4py`` provides the
capability but does not initialize MPI in an ordinary serial process.
When validated launcher-rank evidence or an explicit wrapper contract
identifies a process started by ``mpirun`` / ``srun``, the
:class:`MPIWorld` singleton binds to ``MPI.COMM_WORLD`` automatically.

**Parallel strategies** (chosen per-calculation):

1. **Shell-pair distribution** -- AO integral pairs are distributed
   across ranks with a round-robin scheduler. Each rank evaluates
   a subset of ``(shell_pair_i, shell_pair_j)`` pairs for ERIs,
   J, and K builds. Results are allreduced at the end.

2. **Grid domain decomposition** -- for GPW/GAPW FFT-grid
   collocation, the 3D real-space grid is partitioned into slabs
   along the z-axis. Each rank collocates density on its local
   slab, the slabs are assembled into the full-grid density
   (zero-pad + allreduce), the FFT-Poisson solve is replicated on
   the full grid, and each rank projects its slab of the potential
   back to the AO basis with a final allreduce over the disjoint
   slab contributions. (A distributed FFT with slab transposition
   would remove the replicated solve; not implemented.)

3. **State/k-point distribution** -- for TDDFT and multi-k SCF,
   independent k-points or excitation states are farmed across
   ranks. Each rank computes its assigned subset and sends results
   to rank 0 for collection.

4. **Complete lattice-output distribution** -- direct periodic builders can
   farm complete real-space AO output blocks while retaining the full internal
   translation sum inside every task. Results are allgathered in a
   fingerprinted canonical cell order; this scheduler is independent of the
   physical construction that declared the tasks.

References
----------
* mpi4py: Dalcin et al., *Comput. Sci. Eng.* **13**, 58 (2011).
* MPI-3 Standard (MPI Forum, 2012).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "MPIWorld",
    "GridSlab",
    "KPointPartition",
    "LatticeOutputPartition",
    "ShellPairRange",
    "collocate_density_mpi",
    "distribute_shell_pairs",
    "distribute_tasks",
    "grid_slab_partition",
    "mpi_allgather",
    "mpi_allreduce",
    "mpi_available",
    "mpi_barrier",
    "mpi_bcast",
    "mpi_gather",
    "mpi_rank",
    "mpi_reduce_sum",
    "mpi_size",
    "mpi_world",
    "project_potential_mpi",
]


# ---------------------------------------------------------------------------
# Singleton world
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MPILaunchEvidence:
    """Validated evidence that this process must bind an MPI world."""

    sources: tuple[str, ...]
    rank: int | None
    size: int | None


def _decimal_environment(
    environ: Mapping[str, str],
    name: str,
    *,
    positive: bool,
) -> int:
    value = environ.get(name, "")
    if not value.isascii() or not value.isdecimal():
        qualifier = "positive" if positive else "non-negative"
        raise RuntimeError(f"{name} must be a {qualifier} decimal integer")
    parsed = int(value)
    if (positive and parsed < 1) or (not positive and parsed < 0):
        qualifier = "positive" if positive else "non-negative"
        raise RuntimeError(f"{name} must be {qualifier}")
    return parsed


def _mpi_launch_environment(
    environ: Mapping[str, str] | None = None,
) -> _MPILaunchEvidence | None:
    """Return validated launch evidence, without importing ``mpi4py.MPI``.

    Rank variables are normally the activation signal. Slurm can expose its
    process id in allocation and special-step contexts, so it additionally
    requires an actual task-step id and step-scoped size. Size-only scheduler
    variables are deliberately insufficient.
    """

    env = os.environ if environ is None else environ
    evidence: list[_MPILaunchEvidence] = []
    incomplete_rank_sources: list[str] = []

    def add_rank_size(source: str, rank_name: str, size_name: str) -> None:
        if rank_name not in env:
            return
        if size_name not in env:
            incomplete_rank_sources.append(f"{source} ({size_name} missing)")
            return
        rank = _decimal_environment(env, rank_name, positive=False)
        size = _decimal_environment(env, size_name, positive=True)
        if rank >= size:
            raise RuntimeError(
                f"{source} launcher rank {rank} is outside world size {size}"
            )
        evidence.append(_MPILaunchEvidence((source,), rank, size))

    add_rank_size(
        "Open MPI",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
    )
    add_rank_size("PMI/Intel MPI/MPICH", "PMI_RANK", "PMI_SIZE")
    add_rank_size(
        "MVAPICH",
        "MV2_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_SIZE",
    )

    if "SLURM_PROCID" in env:
        step_id = env.get("SLURM_STEP_ID")
        non_task_step_names = {
            None,
            "",
            "batch",
            "extern",
            "interactive",
            "pending",
        }
        reserved_step_id = False
        if step_id not in non_task_step_names:
            parsed_step_id = _decimal_environment(
                env, "SLURM_STEP_ID", positive=False
            )
            # Slurm reserves every value above SLURM_MAX_NORMAL_STEP_ID.
            # The individual batch/extern encodings changed between releases,
            # so the stable boundary is safer than copying one version's list.
            reserved_step_id = parsed_step_id > 0xFFFFFFF0
        if step_id in non_task_step_names or reserved_step_id:
            # A batch/allocation process can carry SLURM_PROCID=0 without
            # being an MPI participant. Only an actual srun task step counts.
            pass
        elif "SLURM_STEP_NUM_TASKS" not in env:
            incomplete_rank_sources.append(
                "Slurm step (SLURM_STEP_ID/SLURM_STEP_NUM_TASKS missing)"
            )
        else:
            rank = _decimal_environment(
                env, "SLURM_PROCID", positive=False
            )
            size = _decimal_environment(
                env, "SLURM_STEP_NUM_TASKS", positive=True
            )
            if rank >= size:
                raise RuntimeError(
                    f"Slurm launcher rank {rank} is outside world size {size}"
                )
            evidence.append(_MPILaunchEvidence(("Slurm step",), rank, size))

    if "PMIX_RANK" in env:
        has_namespace = bool(env.get("PMIX_NAMESPACE"))
        has_server = any(
            name.startswith("PMIX_SERVER_URI") and bool(value)
            for name, value in env.items()
        )
        if not has_namespace or not has_server:
            incomplete_rank_sources.append(
                "PMIx (PMIX_NAMESPACE/server URI missing)"
            )
        else:
            rank = _decimal_environment(env, "PMIX_RANK", positive=False)
            evidence.append(_MPILaunchEvidence(("PMIx",), rank, None))

    required = env.get("VIBEQC_MPI_REQUIRED")
    if required not in (None, "", "0", "1"):
        raise RuntimeError("VIBEQC_MPI_REQUIRED must be 0 or 1")
    if required == "1":
        if "VIBEQC_MPI_EXPECTED_SIZE" not in env:
            raise RuntimeError(
                "VIBEQC_MPI_REQUIRED=1 requires VIBEQC_MPI_EXPECTED_SIZE"
            )
        expected_size = _decimal_environment(
            env, "VIBEQC_MPI_EXPECTED_SIZE", positive=True
        )
        evidence.append(
            _MPILaunchEvidence(("vibe-qc required launcher",), None, expected_size)
        )

    if not evidence:
        if incomplete_rank_sources:
            detail = ", ".join(incomplete_rank_sources)
            raise RuntimeError(f"incomplete MPI launcher-rank evidence: {detail}")
        return None

    ranks = {item.rank for item in evidence if item.rank is not None}
    sizes = {item.size for item in evidence if item.size is not None}
    if len(ranks) > 1:
        raise RuntimeError("MPI launcher rank variables disagree")
    if len(sizes) > 1:
        raise RuntimeError("MPI launcher world-size variables disagree")

    return _MPILaunchEvidence(
        tuple(source for item in evidence for source in item.sources),
        next(iter(ranks), None),
        next(iter(sizes), None),
    )


_MPI_LAUNCH_EVIDENCE = _mpi_launch_environment()
_MPI = None
_HAS_MPI = False

if _MPI_LAUNCH_EVIDENCE is not None:
    try:
        from mpi4py import MPI as _MPI  # type: ignore[import-untyped]
    except Exception as exc:
        sources = ", ".join(_MPI_LAUNCH_EVIDENCE.sources)
        raise RuntimeError(
            "MPI launcher-rank evidence was detected "
            f"({sources}), but mpi4py.MPI could not be loaded"
        ) from exc
    _HAS_MPI = True


@dataclass
class MPIWorld:
    """Singleton representing the MPI communication world.

    With no launch evidence the world is the serial identity at rank 0 and
    size 1. A detected launch must bind MPI successfully and never falls back
    to that identity.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm or None
        The MPI communicator (``MPI.COMM_WORLD``).
    rank : int
        Rank of this process (0-based).
    size : int
        Total number of MPI processes.
    active : bool
        True if MPI is initialized and size > 1.
    """

    comm: object = None
    rank: int = 0
    size: int = 1
    active: bool = False

    def __post_init__(self):
        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.active = self.size > 1


def _init_mpi_world() -> MPIWorld:
    """Create the MPI world singleton from validated launcher evidence."""
    if _MPI_LAUNCH_EVIDENCE is None:
        return MPIWorld()
    if not _HAS_MPI or _MPI is None:
        raise RuntimeError("MPI launch was detected but mpi4py is unavailable")
    if _MPI.Is_finalized():
        raise RuntimeError("MPI launch was detected after MPI was finalized")
    if not _MPI.Is_initialized():
        raise RuntimeError(
            "MPI launch was detected but mpi4py automatic initialization "
            "is disabled or failed"
        )
    thread_level = _MPI.Query_thread()
    if thread_level < _MPI.THREAD_FUNNELED:
        raise RuntimeError(
            "MPI provides an inadequate thread level; THREAD_FUNNELED "
            "or stronger is required"
        )
    try:
        comm = _MPI.COMM_WORLD
        world = MPIWorld(comm=comm)
    except Exception as exc:
        raise RuntimeError("MPI.COMM_WORLD could not be bound") from exc

    expected_rank = _MPI_LAUNCH_EVIDENCE.rank
    expected_size = _MPI_LAUNCH_EVIDENCE.size
    if expected_rank is not None and world.rank != expected_rank:
        raise RuntimeError(
            "MPI.COMM_WORLD rank contradicts launcher-rank evidence: "
            f"communicator={world.rank}, launcher={expected_rank}"
        )
    if expected_size is not None and world.size != expected_size:
        raise RuntimeError(
            "MPI.COMM_WORLD size contradicts launcher-rank evidence: "
            f"communicator={world.size}, launcher={expected_size}"
        )
    return world


# Module-level singleton -- lazy communicator binding
_mpi_world: Optional[MPIWorld] = None


def mpi_world() -> MPIWorld:
    """Return the MPI world singleton (lazy-init)."""
    global _mpi_world
    if _mpi_world is None:
        _mpi_world = _init_mpi_world()
    elif _MPI_LAUNCH_EVIDENCE is not None:
        if _MPI is None:
            raise RuntimeError("bound MPI world lost its mpi4py runtime")
        if _MPI.Is_finalized():
            raise RuntimeError("MPI was finalized after communicator binding")
    return _mpi_world


def mpi_available() -> bool:
    """True if MPI is active with more than 1 rank."""
    return mpi_world().active


def mpi_rank() -> int:
    """Current MPI rank."""
    return mpi_world().rank


def mpi_size() -> int:
    """Total number of MPI ranks."""
    return mpi_world().size


def mpi_barrier() -> None:
    """Barrier synchronization across all ranks."""
    w = mpi_world()
    if w.active:
        w.comm.Barrier()


# ---------------------------------------------------------------------------
# Collective operations
# ---------------------------------------------------------------------------


def mpi_allreduce(
    data: np.ndarray,
    op: str = "sum",
) -> np.ndarray:
    """Allreduce across MPI ranks.

    Parameters
    ----------
    data : np.ndarray
        Local data array.
    op : str
        Operation: ``"sum"``, ``"max"``, ``"min"``.

    Returns
    -------
    np.ndarray
        Reduced array (same shape, summed across ranks).
    """
    w = mpi_world()
    if not w.active:
        return data

    result = np.empty_like(data)
    if op == "sum":
        w.comm.Allreduce(data, result, op=_MPI.SUM)
    elif op == "max":
        w.comm.Allreduce(data, result, op=_MPI.MAX)
    elif op == "min":
        w.comm.Allreduce(data, result, op=_MPI.MIN)
    else:
        raise ValueError(f"Unknown MPI reduction op: {op!r}")
    return result


def mpi_reduce_sum(data: np.ndarray, root: int = 0) -> Optional[np.ndarray]:
    """Reduce-sum to a root rank.

    Returns None on non-root ranks.
    """
    w = mpi_world()
    if not w.active:
        return data

    if w.rank == root:
        result = np.empty_like(data)
        w.comm.Reduce(data, result, op=_MPI.SUM, root=root)
        return result
    else:
        w.comm.Reduce(data, None, op=_MPI.SUM, root=root)
        return None


def mpi_bcast(data, root: int = 0):
    """Broadcast data from root to all ranks."""
    w = mpi_world()
    if not w.active:
        return data
    return w.comm.bcast(data, root=root)


def mpi_gather(data, root: int = 0) -> Optional[List]:
    """Gather data from all ranks to root."""
    w = mpi_world()
    if not w.active:
        return [data]

    gathered = w.comm.gather(data, root=root)
    if w.rank == root:
        return gathered
    return None


def mpi_allgather(data) -> List:
    """Allgather -- every rank gets data from all ranks."""
    w = mpi_world()
    if not w.active:
        return [data]
    return w.comm.allgather(data)


# ---------------------------------------------------------------------------
# k-point distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KPointPartition:
    """Assignment of Brillouin-zone sampling points to MPI ranks.

    Periodic k-point work is the natural coarse-grained MPI target in
    vibe-qc (``handovers/HANDOVER_MPI.md``): the per-k (or per-momentum-
    transfer) tasks are independent, and the coupling is one reduction at
    the end rather than a chatty exchange, which is what makes it viable
    on Ethernet.

    The awkward part of farming k-points is not splitting them -- it is
    putting the answers back together. Each rank ends up holding results
    for *its own* subset, in its own order, and a plain
    :func:`mpi_allgather` returns a list-of-lists ordered by rank, not by
    k. Reassembling that by hand at each call site is exactly how
    off-by-one and misordering bugs get in, so this class owns both
    halves: the split (:attr:`local_indices`) and the reassembly
    (:meth:`allgather_ordered`).

    Serial (``size == 1``) is the identity partition: every index is
    local, in order, and :meth:`allgather_ordered` just returns the
    caller's list. Non-MPI runs therefore behave exactly as before.

    .. warning::

       With ``strategy="block"`` a naive rank-order flatten of the
       gathered parcels *happens to* produce global k order, because
       contiguous blocks are handed out in rank order. That coincidence
       hides the bug until someone selects ``"cyclic"`` (or an
       unbalanced split), at which point results silently attach to the
       wrong k. A mutation check on 2026-07-28 confirmed it: replacing
       the reassembly below with a rank-order flatten still passed every
       block-strategy test and failed the cyclic ones. Always go through
       :meth:`allgather_ordered`; never flatten an
       :func:`mpi_allgather` of per-k work yourself.

    Attributes
    ----------
    n_kpoints : int
        Total number of sampling points being distributed.
    local_indices : tuple of int
        Global indices owned by this rank, ascending.
    rank, size : int
        This rank and the world size the partition was built for.
    strategy : str
        ``"block"`` (contiguous) or ``"cyclic"`` (round-robin).
    """

    n_kpoints: int
    local_indices: Tuple[int, ...]
    rank: int
    size: int
    strategy: str

    @classmethod
    def create(
        cls,
        n_kpoints: int,
        *,
        strategy: str = "block",
    ) -> "KPointPartition":
        """Partition ``n_kpoints`` across the current MPI world.

        ``strategy="block"`` hands each rank one contiguous run, matching
        :func:`distribute_tasks`. ``strategy="cyclic"`` deals them
        round-robin, which balances better when per-k cost varies
        systematically with the index (for example when a q = 0 group is
        cheaper than the rest), at the cost of contiguity.

        Ranks beyond ``n_kpoints`` get an empty assignment rather than an
        error -- oversubscription is a scheduling choice, not a bug, and
        the collectives below stay correct for empty locals.
        """
        if int(n_kpoints) < 0:
            raise ValueError(
                f"KPointPartition: n_kpoints must be >= 0, got {n_kpoints}"
            )
        n_kpoints = int(n_kpoints)
        w = mpi_world()
        rank = int(w.rank)
        size = max(int(w.size), 1)

        if strategy == "block":
            base, rem = divmod(n_kpoints, size)
            start = rank * base + min(rank, rem)
            stop = start + base + (1 if rank < rem else 0)
            local = tuple(range(start, stop))
        elif strategy == "cyclic":
            local = tuple(range(rank, n_kpoints, size))
        else:
            raise ValueError(
                "KPointPartition: strategy must be 'block' or 'cyclic'; "
                f"got {strategy!r}"
            )
        return cls(
            n_kpoints=n_kpoints,
            local_indices=local,
            rank=rank,
            size=size,
            strategy=str(strategy),
        )

    @property
    def n_local(self) -> int:
        """Number of sampling points owned by this rank."""
        return len(self.local_indices)

    def owns(self, k: int) -> bool:
        """True if global index ``k`` belongs to this rank."""
        return int(k) in self.local_indices

    def owner_of(self, k: int) -> int:
        """Rank that owns global index ``k``.

        Computed from the strategy rather than by communication, so it is
        callable on any rank without collective participation.
        """
        k = int(k)
        if not 0 <= k < self.n_kpoints:
            raise IndexError(
                f"KPointPartition: k={k} out of range for "
                f"n_kpoints={self.n_kpoints}"
            )
        if self.strategy == "cyclic":
            return k % self.size
        base, rem = divmod(self.n_kpoints, self.size)
        # The first `rem` ranks carry one extra point.
        boundary = (base + 1) * rem
        if k < boundary:
            return k // (base + 1)
        return rem + (k - boundary) // base

    def allgather_ordered(self, local_values: Sequence) -> List:
        """Collect per-k results from every rank into GLOBAL k order.

        ``local_values`` must line up with :attr:`local_indices`. The
        returned list has length :attr:`n_kpoints` on *every* rank, with
        entry ``k`` produced by whichever rank owned ``k`` -- so callers
        index it by k, never by rank.

        Serial mode returns ``list(local_values)`` unchanged.
        """
        values = list(local_values)
        if len(values) != self.n_local:
            raise ValueError(
                "KPointPartition.allgather_ordered: expected "
                f"{self.n_local} local values to match local_indices, "
                f"got {len(values)}"
            )
        w = mpi_world()
        if not w.active:
            return values

        parcels = w.comm.allgather((self.local_indices, values))
        ordered: List = [None] * self.n_kpoints
        seen = 0
        for indices, chunk in parcels:
            for k, value in zip(indices, chunk):
                ordered[k] = value
                seen += 1
        if seen != self.n_kpoints:
            # A rank contributed a partition inconsistent with this one
            # (different n_kpoints or strategy). Fail loudly: a silently
            # short result would look like a converged answer.
            raise RuntimeError(
                "KPointPartition.allgather_ordered: reassembled "
                f"{seen} of {self.n_kpoints} points; ranks disagree on "
                "the partition."
            )
        return ordered


# ---------------------------------------------------------------------------
# Complete real-space output-block distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatticeOutputPartition:
    """MPI ownership of complete real-space AO output blocks.

    This is a construction-neutral scheduler.  In particular, its task keys
    are neither Gamma-CCM union-and-weight WSCs nor, by themselves, the
    residue classes of the finite translation group used by chi-CCM.  A task
    is exactly one caller-declared output block.  The caller remains
    responsible for keeping the complete internal translation sum inside
    that task and for aggregating tied Wigner-Seitz representatives before a
    finite-group residue is ever presented as one.

    The gathered parcel carries the complete scheduling contract -- task
    count, strategy, ordered-key fingerprint, source rank, owned indices and
    values.  Reassembly validates all of it.  This is intentionally stricter
    than a count-only gather: one duplicated index plus one missing index has
    the right total count but is still a silent wrong answer.
    """

    task_keys: Tuple[Tuple[int, int, int], ...]
    local_indices: Tuple[int, ...]
    rank: int
    size: int
    strategy: str
    task_fingerprint: str

    _SCHEMA = "vibeqc.mpi.lattice-output-partition/v1"

    @staticmethod
    def _indices_for_rank(
        n_tasks: int,
        rank: int,
        size: int,
        strategy: str,
    ) -> Tuple[int, ...]:
        if strategy == "cyclic":
            return tuple(range(rank, n_tasks, size))
        base, rem = divmod(n_tasks, size)
        start = rank * base + min(rank, rem)
        stop = start + base + (1 if rank < rem else 0)
        return tuple(range(start, stop))

    @staticmethod
    def _fingerprint(keys: Sequence[Tuple[int, int, int]]) -> str:
        digest = sha256()
        digest.update(b"vibeqc-lattice-output-keys-v1\0")
        for key in keys:
            for value in key:
                digest.update(int(value).to_bytes(8, "big", signed=True))
        return digest.hexdigest()

    @classmethod
    def create(
        cls,
        task_keys: Sequence[Sequence[int]],
        *,
        strategy: str = "cyclic",
    ) -> "LatticeOutputPartition":
        """Partition an ordered, unique list of three-index cell keys."""
        if strategy not in ("block", "cyclic"):
            raise ValueError(
                "LatticeOutputPartition: strategy must be 'block' or "
                f"'cyclic'; got {strategy!r}"
            )
        keys_list: List[Tuple[int, int, int]] = []
        for position, raw_key in enumerate(task_keys):
            try:
                values = tuple(raw_key)
            except TypeError as exc:
                raise ValueError(
                    "LatticeOutputPartition: every task key must contain "
                    f"three integers; task {position} is {raw_key!r}"
                ) from exc
            if len(values) != 3 or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in values
            ):
                raise ValueError(
                    "LatticeOutputPartition: every task key must contain "
                    f"three integers; task {position} is {raw_key!r}"
                )
            keys_list.append(tuple(int(value) for value in values))
        keys = tuple(keys_list)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "LatticeOutputPartition: task keys must be unique; "
                "duplicate output cells would be counted twice"
            )
        w = mpi_world()
        rank = int(w.rank)
        size = max(int(w.size), 1)
        local = cls._indices_for_rank(len(keys), rank, size, strategy)
        return cls(
            task_keys=keys,
            local_indices=local,
            rank=rank,
            size=size,
            strategy=strategy,
            task_fingerprint=cls._fingerprint(keys),
        )

    @property
    def n_tasks(self) -> int:
        """Total number of declared output-block tasks."""
        return len(self.task_keys)

    @property
    def n_local(self) -> int:
        """Number of tasks owned by this rank."""
        return len(self.local_indices)

    @property
    def local_task_keys(self) -> Tuple[Tuple[int, int, int], ...]:
        """Ordered cell keys owned by this rank."""
        return tuple(self.task_keys[index] for index in self.local_indices)

    @property
    def task_counts(self) -> Tuple[int, ...]:
        """Deterministic task count for every rank."""
        return tuple(
            len(
                self._indices_for_rank(
                    self.n_tasks,
                    rank,
                    self.size,
                    self.strategy,
                )
            )
            for rank in range(self.size)
        )

    def owns(self, task_index: int) -> bool:
        """True if ``task_index`` belongs to this rank."""
        return int(task_index) in self.local_indices

    def owner_of(self, task_index: int) -> int:
        """Rank that owns ``task_index``, without communication."""
        task_index = int(task_index)
        if not 0 <= task_index < self.n_tasks:
            raise IndexError(
                "LatticeOutputPartition: task index "
                f"{task_index} out of range for n_tasks={self.n_tasks}"
            )
        if self.strategy == "cyclic":
            return task_index % self.size
        base, rem = divmod(self.n_tasks, self.size)
        boundary = (base + 1) * rem
        if task_index < boundary:
            return task_index // (base + 1)
        return rem + (task_index - boundary) // base

    def allgather_ordered(self, local_values: Sequence) -> List:
        """Return complete task values in declared cell-key order.

        Every active rank must call this collective.  Empty ranks contribute
        an empty parcel; callers must not turn that empty assignment into a
        native ``output_indices=[]`` call when the native API interprets an
        empty subset as all outputs.
        """
        values = list(local_values)
        if len(values) != self.n_local:
            raise ValueError(
                "LatticeOutputPartition.allgather_ordered: expected "
                f"{self.n_local} local values to match local_indices, "
                f"got {len(values)}"
            )
        w = mpi_world()
        if not w.active:
            return values

        parcel = (
            self._SCHEMA,
            self.n_tasks,
            self.strategy,
            self.task_fingerprint,
            self.rank,
            self.local_indices,
            values,
        )
        parcels = list(w.comm.allgather(parcel))
        if len(parcels) != self.size:
            raise RuntimeError(
                "LatticeOutputPartition.allgather_ordered: collective "
                f"returned {len(parcels)} rank parcels for size={self.size}"
            )

        ordered: List = [None] * self.n_tasks
        seen_ranks = set()
        seen_indices = set()
        for raw in parcels:
            if not isinstance(raw, (tuple, list)) or len(raw) != 7:
                raise RuntimeError(
                    "LatticeOutputPartition.allgather_ordered: malformed "
                    "rank parcel"
                )
            schema, n_tasks, strategy, fingerprint, source, indices, chunk = raw
            if (
                schema != self._SCHEMA
                or int(n_tasks) != self.n_tasks
                or strategy != self.strategy
                or fingerprint != self.task_fingerprint
            ):
                raise RuntimeError(
                    "LatticeOutputPartition.allgather_ordered: ranks "
                    "disagree on the task count, strategy, or ordered-cell "
                    "fingerprint"
                )
            source = int(source)
            if not 0 <= source < self.size or source in seen_ranks:
                raise RuntimeError(
                    "LatticeOutputPartition.allgather_ordered: duplicate "
                    "or out-of-range source rank"
                )
            seen_ranks.add(source)
            indices = tuple(int(index) for index in indices)
            chunk = list(chunk)
            expected = self._indices_for_rank(
                self.n_tasks, source, self.size, self.strategy
            )
            if indices != expected:
                raise RuntimeError(
                    "LatticeOutputPartition.allgather_ordered: source rank "
                    f"{source} reported indices {indices}, expected {expected}"
                )
            if len(chunk) != len(indices):
                raise RuntimeError(
                    "LatticeOutputPartition.allgather_ordered: source rank "
                    f"{source} reported {len(indices)} indices but "
                    f"{len(chunk)} values"
                )
            for index, value in zip(indices, chunk):
                if index in seen_indices:
                    raise RuntimeError(
                        "LatticeOutputPartition.allgather_ordered: duplicate "
                        f"global task index {index}"
                    )
                seen_indices.add(index)
                ordered[index] = value
        if seen_ranks != set(range(self.size)) or seen_indices != set(
            range(self.n_tasks)
        ):
            raise RuntimeError(
                "LatticeOutputPartition.allgather_ordered: gathered parcels "
                "do not cover every rank and task exactly once"
            )
        return ordered


# ---------------------------------------------------------------------------
# Shell-pair distribution
# ---------------------------------------------------------------------------


@dataclass
class ShellPairRange:
    """Range of shell-pair indices assigned to a single MPI rank.

    Attributes
    ----------
    rank : int
        MPI rank owning this range.
    start : int
        Start index (inclusive) in the flattened shell-pair list.
    end : int
        End index (exclusive).
    total : int
        Total number of shell pairs across all ranks.
    """

    rank: int
    start: int
    end: int
    total: int

    @property
    def count(self) -> int:
        return self.end - self.start


def distribute_shell_pairs(
    n_shells: int,
) -> ShellPairRange:
    """Distribute shell-shell pairs across MPI ranks.

    Shell pairs ``(i, j)`` with ``i <= j`` (upper triangle) are
    flattened and distributed in contiguous blocks to each rank.

    Parameters
    ----------
    n_shells : int
        Number of contracted shells in the basis set.

    Returns
    -------
    ShellPairRange
        The local range of pairs to compute.
    """
    w = mpi_world()
    rank = w.rank
    n_ranks = max(w.size, 1)

    n_pairs = n_shells * (n_shells + 1) // 2

    # Contiguous block distribution
    base = n_pairs // n_ranks
    rem = n_pairs % n_ranks
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)

    return ShellPairRange(rank=rank, start=start, end=end, total=n_pairs)


# ---------------------------------------------------------------------------
# Grid domain decomposition (3D slab along z-axis)
# ---------------------------------------------------------------------------


@dataclass
class GridSlab:
    """A z-axis slab of a 3D real-space grid assigned to one MPI rank.

    Attributes
    ----------
    rank : int
        MPI rank owning this slab.
    nx, ny : int
        Full grid dimensions in x and y.
    nz_local : int
        Number of z-planes owned by this rank.
    z_start : int
        Global z-index of the first plane in this slab.
    z_end : int
        Global z-index one past the last plane.
    nz_global : int
        Total number of z-planes across all ranks.
    """

    rank: int
    nx: int
    ny: int
    nz_local: int
    z_start: int
    z_end: int
    nz_global: int

    @property
    def local_shape(self) -> Tuple[int, int, int]:
        """Shape of the local grid chunk ``(nx, ny, nz_local)``."""
        return (self.nx, self.ny, self.nz_local)


def grid_slab_partition(
    nx: int,
    ny: int,
    nz: int,
) -> GridSlab:
    """Partition a 3D grid into z-axis slabs for MPI domain decomposition.

    Parameters
    ----------
    nx, ny, nz : int
        Full grid dimensions.

    Returns
    -------
    GridSlab
        The local slab owned by this rank.
    """
    w = mpi_world()
    rank = w.rank
    n_ranks = max(w.size, 1)

    # Distribute z-planes as evenly as possible
    base = nz // n_ranks
    rem = nz % n_ranks
    z_start = rank * base + min(rank, rem)
    z_end = z_start + base + (1 if rank < rem else 0)
    nz_local = z_end - z_start

    return GridSlab(
        rank=rank,
        nx=nx,
        ny=ny,
        nz_local=nz_local,
        z_start=z_start,
        z_end=z_end,
        nz_global=nz,
    )


# ---------------------------------------------------------------------------
# Periodic TDDFT k-point / state distribution
# ---------------------------------------------------------------------------


def distribute_tasks(
    n_tasks: int,
) -> Tuple[int, int]:
    """Distribute ``n_tasks`` independent work items across MPI ranks.

    Returns ``(local_start, local_end)`` -- the range of tasks
    assigned to this rank.

    Parameters
    ----------
    n_tasks : int
        Total number of independent work items.

    Returns
    -------
    local_start : int
        First task index for this rank.
    local_end : int
        One past the last task index.
    """
    w = mpi_world()
    rank = w.rank
    n_ranks = max(w.size, 1)

    base = n_tasks // n_ranks
    rem = n_tasks % n_ranks
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


# ---------------------------------------------------------------------------
# MPI-aware density collocation (GPW/GAPW grid parallelisation)
# ---------------------------------------------------------------------------


def collocate_density_mpi(
    chi_local: np.ndarray,
    density_matrix: np.ndarray,
) -> np.ndarray:
    """Collocate the density on this rank's local grid slab.

    Collocation is pointwise in the grid index -- ``r(r_g) = S_muν
    D_muν chi_mu(r_g) chi_ν(r_g)`` depends only on the density matrix and
    the AO values at ``r_g`` -- so the local slab density is complete
    as computed and there is **no cross-rank reduction here**. The
    caller assembles the full grid from the disjoint slabs (see
    ``GpwJBuilder._build_J_mpi``: zero-pad + allreduce of the full
    grid).

    A pre-v0.12.1 version allreduce-summed the per-rank results,
    which added *different* z-slabs together elementwise -- garbage
    for any world size > 1 (and shape-mismatched on uneven slab
    partitions). Pinned by ``tests/test_mpi_gpw_parity.py``.

    Parameters
    ----------
    chi_local : np.ndarray
        Local AO values on this rank's grid slab, shape
        ``(nx * ny * nz_local, n_basis)``.
    density_matrix : np.ndarray
        AO density matrix, shape ``(n_basis, n_basis)``.

    Returns
    -------
    np.ndarray
        Density on the local slab, shape ``(nx * ny * nz_local,)``.
    """
    D = np.asarray(density_matrix, dtype=float)
    return np.einsum("gm,mn,gn->g", chi_local, D, chi_local, optimize=True)


def project_potential_mpi(
    chi_local: np.ndarray,
    V_local: np.ndarray,
    dV: float,
) -> np.ndarray:
    """Project the Hartree potential on a local slab to the AO basis.

    Computes J^{local}_{mu nu} = dV * sum_g chi_mu(r_g) chi_nu(r_g) V(r_g)
    on the local slab, then allreduces across ranks. Unlike the
    density collocation, this reduction is correct *by construction*:
    the grid sum is partitioned over disjoint slabs, every rank's
    partial sum is the same ``(n_basis, n_basis)`` shape, and the
    allreduce adds each grid point's contribution exactly once. The
    correctness precondition is that the slabs tile the grid without
    overlap -- which :func:`grid_slab_partition` guarantees.

    Parameters
    ----------
    chi_local : np.ndarray
        Local AO values, shape ``(n_grid_local, n_basis)``.
    V_local : np.ndarray
        Local Hartree potential, shape ``(n_grid_local,)``.
    dV : float
        Voxel volume in bohr^3.

    Returns
    -------
    np.ndarray
        Full AO-basis J matrix, shape ``(n_basis, n_basis)``,
        summed across all ranks.
    """
    chi = np.asarray(chi_local, dtype=float)
    V = np.asarray(V_local, dtype=float)
    # J_{mu nu} = dV * sum_g chi_{mu}(r_g) * V(r_g) * chi_{nu}(r_g)
    #           = dV * chi^T @ diag(V) @ chi
    chi_weighted = chi * V[:, None]  # (n_grid_local, n_basis)
    J_local = dV * (chi.T @ chi_weighted)
    return mpi_allreduce(J_local, op="sum")
