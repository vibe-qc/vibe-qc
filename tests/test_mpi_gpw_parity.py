"""MPI GPW Hartree-J parity — simulated multi-rank world vs serial.

Regression suite for the v0.12 post-release audit finding
(2026-06-12): ``GpwJBuilder(mpi_aware=True)`` produced garbage for
any MPI world size > 1. Three compounding errors, every one of which
degenerates to correct behaviour at world size 1 (which is why the
serial CI suite never caught it):

1. ``vibeqc.mpi.collocate_density_mpi`` allreduce-summed the
   per-rank slab densities *elementwise* — adding rank 0's bottom-
   of-cell density to rank 1's top-of-cell density (and shape-
   mismatching outright on uneven slab partitions). Collocation is
   pointwise in the grid index; the local slab density is complete
   as computed and there is nothing to reduce.
2. ``GpwJBuilder._build_J_mpi`` zero-padded only *its own* slab into
   the full-grid density and ran FFT-Poisson on it — every other
   rank's slab was zero. The full grid was never reduced.
3. The slab points were sliced from the flattened coordinate array
   as ``r_all[z_start·nx·ny : z_end·nx·ny]`` — but the C-order
   flattening of the ``(nx, ny, nz, 3)`` coordinate grid has z as
   the *fastest* axis, so the slice actually selected x-planes and
   the subsequent ``reshape(nx, ny, nz_local)`` scrambled them.

CI has no ``mpirun``, so multi-rank execution is emulated in-process
by :class:`SimulatedMPI` below: the code under test runs once per
simulated rank with ``vibeqc.mpi.mpi_world`` / ``mpi_allreduce``
monkeypatched, sharing per-call contribution buffers across the
runs. The invariant pinned here is route-internal and exact: the
MPI-decomposed J build must reproduce the serial J build on the
same (basis, D, grid) to float-roundoff, on every rank.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.mpi as vq_mpi
from vibeqc.mpi import MPIWorld
from vibeqc.periodic_gapw_grid import PlaneWaveGrid
from vibeqc.periodic_gapw_j import GpwJBuilder, _ao_values_on_grid

pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


class SimulatedMPI:
    """Deterministic in-process emulation of an N-rank MPI world.

    The code under test runs once per rank *sequentially*; the
    patched ``mpi_allreduce`` deposits each rank's contribution into
    a per-call-index buffer shared across the runs and returns the
    sum of everything deposited so far. A single sequential pass
    cannot hand every rank the complete reduction (rank 0 runs
    before rank 1 has deposited anything), so :meth:`run_lockstep`
    sweeps the rank loop until every rank's result is bitwise
    stable — at that point every ``mpi_allreduce`` call has returned
    the true all-rank sum, exactly as a lockstep MPI world would.
    Convergence takes (number of chained collective stages + 1)
    sweeps; the cap only guards a non-converging pathology.

    Emulates ``op="sum"`` only — the GPW J build uses nothing else.
    Mismatched per-rank shapes deposited into the same collective
    call raise immediately, mirroring the abort a real mpi4py
    ``Allreduce`` on mismatched buffers would produce.
    """

    def __init__(self, n_ranks: int):
        self.n_ranks = n_ranks
        self._rank = 0
        self._call_idx = 0
        self._slots: dict[int, dict[int, np.ndarray]] = {}

    # ---- patched entry points -----------------------------------

    def fake_mpi_world(self) -> MPIWorld:
        return MPIWorld(rank=self._rank, size=self.n_ranks, active=True)

    def fake_allreduce(self, data: np.ndarray, op: str = "sum") -> np.ndarray:
        if op != "sum":
            raise NotImplementedError(
                f"SimulatedMPI only emulates op='sum'; got op={op!r}"
            )
        idx = self._call_idx
        self._call_idx += 1
        slot = self._slots.setdefault(idx, {})
        slot[self._rank] = np.array(data, dtype=float, copy=True)
        shapes = {a.shape for a in slot.values()}
        if len(shapes) > 1:
            raise AssertionError(
                f"mpi_allreduce call #{idx}: ranks deposited mismatched "
                f"shapes {sorted(shapes)} — a real MPI allreduce would "
                f"abort the job here."
            )
        return sum(slot.values())

    # ---- driver ---------------------------------------------------

    def run_lockstep(self, fn, max_sweeps: int = 6) -> list:
        """Run ``fn(rank)`` for every rank, sweeping until stable.

        Returns the per-rank results of the stabilised sweep.
        """
        prev = None
        for _ in range(max_sweeps):
            results = []
            for rank in range(self.n_ranks):
                self._rank = rank
                self._call_idx = 0
                results.append(fn(rank))
            if prev is not None and all(
                np.array_equal(a, b) for a, b in zip(results, prev)
            ):
                return results
            prev = results
        raise AssertionError(
            f"SimulatedMPI: per-rank results did not stabilise within "
            f"{max_sweeps} sweeps — the code under test is not a "
            f"deterministic function of the collective inputs."
        )


def _h2_offcentre_system(L: float = 10.0, nx: int = 12, ny: int = 12, nz: int = 12):
    """H2/STO-3G along z, deliberately off-centre.

    The bond axis runs along z (the slab axis) and the molecule sits
    away from the cell centre, so the density differs strongly
    between slabs — a slab-symmetric density could let a wrong
    reduction cancel by accident. The density matrix is synthetic
    (symmetric, not idempotent): the J build is linear-algebraic in
    D, so MPI-vs-serial parity needs no SCF.
    """
    mol = vq.Molecule(
        [
            vq.Atom(1, [0.4 * L, 0.5 * L, 0.30 * L]),
            vq.Atom(1, [0.4 * L, 0.5 * L, 0.30 * L + 1.4]),
        ],
        charge=0,
        multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    D = np.array([[1.6, 0.5], [0.5, 0.4]])
    grid = PlaneWaveGrid(np.eye(3) * L, nx, ny, nz)
    return basis, D, grid


def _install(sim: SimulatedMPI, monkeypatch) -> None:
    monkeypatch.setattr(vq_mpi, "mpi_world", sim.fake_mpi_world)
    monkeypatch.setattr(vq_mpi, "mpi_allreduce", sim.fake_allreduce)


# ---------------------------------------------------------------------------
# collocate_density_mpi — slab locality
# ---------------------------------------------------------------------------


def test_collocate_density_mpi_returns_local_slab_density(monkeypatch):
    """Each rank's collocated slab equals the serial density's slab.

    Collocation is pointwise in the grid index — ρ(r_g) depends only
    on D and the AO values at r_g — so the local slab density is
    complete as computed. Pre-fix the function allreduce-summed the
    per-rank slabs elementwise, returning slab₀+slab₁ on every rank.
    """
    basis, D, grid = _h2_offcentre_system()
    sim = SimulatedMPI(2)
    _install(sim, monkeypatch)

    r_grid = grid.cartesian_coords()  # (nx, ny, nz, 3)
    chi_full = _ao_values_on_grid(
        basis, r_grid.reshape(-1, 3), grid.lattice_bohr
    )
    rho_ref = np.einsum(
        "gm,mn,gn->g", chi_full, D, chi_full, optimize=True
    ).reshape(grid.shape)

    def per_rank(rank):
        slab = vq_mpi.grid_slab_partition(grid.nx, grid.ny, grid.nz)
        # z is the FASTEST axis of the C-order flattening, so the
        # slab points are sliced on axis 2, not as a flat range.
        r_local = r_grid[:, :, slab.z_start : slab.z_end, :].reshape(-1, 3)
        chi_local = _ao_values_on_grid(basis, r_local, grid.lattice_bohr)
        rho_local = vq_mpi.collocate_density_mpi(chi_local, D)
        return rho_local.reshape(grid.nx, grid.ny, slab.nz_local)

    rho_slabs = sim.run_lockstep(per_rank)
    # nz = 12 over 2 ranks → slabs [0, 6) and [6, 12).
    np.testing.assert_allclose(rho_slabs[0], rho_ref[:, :, :6], atol=1e-12, rtol=0)
    np.testing.assert_allclose(rho_slabs[1], rho_ref[:, :, 6:], atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# _build_J_mpi — end-to-end parity vs the serial build
# ---------------------------------------------------------------------------


def test_build_j_mpi_matches_serial_two_ranks(monkeypatch):
    """J from the 2-rank MPI build == J from the serial build.

    The decomposition is route-internal: the MPI build must
    reproduce the serial J on the same (basis, D, grid) to float
    roundoff. Pre-fix the Poisson solve saw a mostly-zero, partially
    scrambled density and the J was wrong by O(0.1) Ha-scale matrix
    elements.
    """
    basis, D, grid = _h2_offcentre_system()  # nz=12 → even 6/6 split
    J_serial = GpwJBuilder(basis, grid).build_J(D)

    sim = SimulatedMPI(2)
    _install(sim, monkeypatch)
    builder = GpwJBuilder(basis, grid, mpi_aware=True)

    J_per_rank = sim.run_lockstep(lambda rank: builder.build_J(D))

    # Allreduce semantics: the result is replicated — identical on
    # every rank.
    np.testing.assert_array_equal(J_per_rank[0], J_per_rank[1])
    for J_mpi in J_per_rank:
        np.testing.assert_allclose(J_mpi, J_serial, atol=1e-10, rtol=0)


def test_build_j_mpi_matches_serial_three_ranks_uneven_slabs(monkeypatch):
    """Same parity with 3 ranks and an uneven slab split (4/3/3).

    Pre-fix the slab allreduce was shape-mismatched outright on
    uneven partitions (each rank contributed an nx·ny·nz_local-sized
    buffer with differing nz_local) — the case a real mpirun would
    abort on rather than silently mis-converge.
    """
    basis, D, grid = _h2_offcentre_system(nz=10)  # 10 planes / 3 ranks
    J_serial = GpwJBuilder(basis, grid).build_J(D)

    sim = SimulatedMPI(3)
    _install(sim, monkeypatch)
    builder = GpwJBuilder(basis, grid, mpi_aware=True)

    J_per_rank = sim.run_lockstep(lambda rank: builder.build_J(D))

    for J_mpi in J_per_rank:
        np.testing.assert_allclose(J_mpi, J_serial, atol=1e-10, rtol=0)


def test_build_j_mpi_inactive_world_degenerates_to_serial():
    """At world size 1 (no patching) the MPI path equals serial.

    This is the degenerate case that made the multi-rank bug
    invisible to the serial CI suite — pinned so the fix keeps it.
    """
    basis, D, grid = _h2_offcentre_system()
    J_mpi = GpwJBuilder(basis, grid, mpi_aware=True).build_J(D)
    J_serial = GpwJBuilder(basis, grid).build_J(D)
    np.testing.assert_allclose(J_mpi, J_serial, atol=1e-10, rtol=0)
