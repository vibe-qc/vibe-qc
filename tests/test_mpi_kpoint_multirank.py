"""Real multi-rank tests for the k-point partition (HANDOVER_MPI.md step 1).

Every other MPI test in the suite runs single-process, which exercises
only the serial fallbacks -- the branches that matter under ``mpirun``
(the collectives, and the rank-order-to-k-order reassembly) are never
touched. These spawn an actual ``mpirun -np N`` and assert the invariant
that matters for k-point farming: **every rank ends up with the same
list, in global k order, whoever computed each entry**.

Skipped cleanly when ``mpi4py`` (the ``[mpi]`` optional extra) or an MPI
launcher is absent, so the default gate is unaffected. They only *mean*
something in an environment that has both.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _mpi_launcher() -> str | None:
    for name in ("mpirun", "mpiexec"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _have_mpi4py() -> bool:
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return False
    return True


requires_mpi = pytest.mark.skipif(
    _mpi_launcher() is None or not _have_mpi4py(),
    reason="needs the [mpi] extra (mpi4py) and an MPI launcher",
)


def test_mpi_environment_is_not_half_installed(request) -> None:
    """An MPI launcher without ``mpi4py`` is a broken environment, not an
    absent capability -- and it must not pass silently (issue #465).

    Every other test in this module skips when MPI is unavailable, which is
    right when MPI is *genuinely* absent: a laptop with no launcher cannot run
    these and should not pretend to. But the skip cannot tell that case apart
    from a machine that plainly intends to run MPI -- ``mpirun`` is installed
    -- and is merely missing one pip package. Both print ``23 skipped``.

    That is worse than a red lane. A red lane is at least visible once someone
    runs it; this one produces output indistinguishable from a passing lane in
    every summary a human or an agent reads. Nothing separates "23 tests
    verified the MPI paths" from "23 tests declined to look". The failure it
    enables is specific: a fixer changes MPI-adjacent code, runs the lane, sees
    no failures, and lands -- evidence real in form and empty in content.
    Measured on such a machine: 23 skipped in 0.32 s before installing
    ``mpi4py``, 23 passed in 133.56 s after.

    So this one test always runs, and fails loudly on the half-present case
    while staying quiet on the genuinely-absent one. The reverse asymmetry
    (``mpi4py`` present, no launcher) is deliberately *not* an error: there is
    nothing to run and nothing the developer needs to fix.
    """
    launcher = _mpi_launcher()
    if launcher is None:
        return  # genuinely no MPI on this machine; the skips are correct
    if not _have_mpi4py():
        # Count what this session actually collected from this module rather
        # than quoting a literal: a literal misreports the moment a
        # multi-rank test is added, parametrized, or deselected.
        here = Path(__file__).resolve()
        silent = [
            item
            for item in request.session.items
            if Path(str(item.path)).resolve() == here
            and item.nodeid != request.node.nodeid
        ]
        pytest.fail(
            f"broken MPI test environment: an MPI launcher is installed "
            f"({launcher}) but mpi4py is not importable, so every multi-rank "
            "test in this module skips while appearing green "
            f"({len(silent)} collected in this session).\n\n"
            "Install the extra that makes them run:\n"
            "    pip install -e '.[mpi]'\n\n"
            "If this machine genuinely must not run MPI tests, remove the "
            "launcher from PATH rather than leaving the environment half "
            "installed -- see issue #465."
        )


def _run_under_mpi(
    body: str,
    n_ranks: int,
    tmp_path,
    *,
    expected_world_size: int | None = None,
):
    script = tmp_path / "mpi_body.py"
    script.write_text(textwrap.dedent(body))
    launcher = _mpi_launcher()
    assert launcher is not None
    env = dict(os.environ)
    # Keep each rank single-threaded so the box is not oversubscribed by
    # ranks x OpenMP threads while the test runs.
    env["OMP_NUM_THREADS"] = "1"
    if expected_world_size is not None:
        env["VIBEQC_MPI_REQUIRED"] = "1"
        env["VIBEQC_MPI_EXPECTED_SIZE"] = str(expected_world_size)
    # OpenMPI refuses to oversubscribe by default on small machines.
    return subprocess.run(
        [launcher, "--oversubscribe", "-np", str(n_ranks),
         sys.executable, str(script)],
        capture_output=True, text=True, timeout=600, env=env,
    )


@requires_mpi
def test_single_launched_rank_binds_initialized_world(tmp_path):
    """A one-rank launch binds COMM_WORLD although it is not distributed."""
    result = _run_under_mpi(
        """
        import sys
        import vibeqc as vq
        from mpi4py import MPI

        world = vq.mpi_world()
        assert MPI.Is_initialized() is True
        assert "mpi4py.MPI" in sys.modules
        assert world.comm is not None
        assert world.rank == 0
        assert world.size == 1
        assert world.active is False
        assert vq.mpi_available() is False
        print("OK")
        """,
        1,
        tmp_path,
        expected_world_size=1,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.count("OK") == 1


@requires_mpi
@pytest.mark.parametrize("n_ranks", [2, 3])
@pytest.mark.parametrize("strategy", ["block", "cyclic"])
def test_allgather_ordered_returns_global_k_order(
    n_ranks, strategy, tmp_path
):
    """Each rank computes only its own k-points, yet every rank must end
    up with the full list indexed by k -- not by rank.

    The payload is deliberately order-sensitive: entry k is ``k * 10``,
    so any rank-ordered or shifted reassembly changes the result rather
    than merely permuting equal values.
    """
    result = _run_under_mpi(
        f"""
        import sys
        from vibeqc.mpi import KPointPartition, mpi_size, mpi_rank

        n_k = 7
        part = KPointPartition.create(n_k, strategy="{strategy}")
        assert mpi_size() == {n_ranks}, (mpi_size(), {n_ranks})
        assert part.size == {n_ranks}

        # Only this rank's points are computed here.
        local = [k * 10 for k in part.local_indices]
        gathered = part.allgather_ordered(local)

        assert gathered == [k * 10 for k in range(n_k)], (
            mpi_rank(), part.local_indices, gathered
        )
        # Ownership is disjoint and complete across ranks.
        assert all(part.owner_of(k) == part.rank
                   for k in part.local_indices)
        print("OK", mpi_rank(), part.local_indices)
        """,
        n_ranks, tmp_path,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.count("OK") == n_ranks, result.stdout


@requires_mpi
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_partition_covers_every_kpoint_exactly_once(n_ranks, tmp_path):
    """Across ranks the local index sets must tile the full range with
    no gap and no overlap -- a dropped k would silently under-sample the
    Brillouin zone, which is far worse than a crash."""
    result = _run_under_mpi(
        f"""
        from vibeqc.mpi import KPointPartition, mpi_world

        for strategy in ("block", "cyclic"):
            for n_k in (0, 1, 4, 9, 16):
                part = KPointPartition.create(n_k, strategy=strategy)
                owned = part.allgather_ordered(
                    [part.rank] * part.n_local
                )
                assert len(owned) == n_k, (strategy, n_k, owned)
                # Every entry was written by the rank owner_of names.
                for k, writer in enumerate(owned):
                    assert writer == part.owner_of(k), (
                        strategy, n_k, k, writer, part.owner_of(k)
                    )
        print("OK", mpi_world().rank)
        """,
        n_ranks, tmp_path,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.count("OK") == n_ranks


@requires_mpi
@pytest.mark.parametrize("n_ranks", [2, 3])
@pytest.mark.parametrize("strategy", ["block", "cyclic"])
def test_lattice_output_partition_real_collective(
    n_ranks, strategy, tmp_path
):
    """Complete output-block tasks reassemble in cell-key order."""
    result = _run_under_mpi(
        f"""
        from vibeqc.mpi import LatticeOutputPartition, mpi_rank, mpi_size

        keys = [(i, -i, i % 2) for i in range(7)]
        part = LatticeOutputPartition.create(keys, strategy="{strategy}")
        assert mpi_size() == {n_ranks}
        local = [(index, keys[index]) for index in part.local_indices]
        gathered = part.allgather_ordered(local)
        assert gathered == [(index, keys[index]) for index in range(7)]
        assert part.task_counts == tuple(
            sum(1 for index in range(7) if part.owner_of(index) == rank)
            for rank in range({n_ranks})
        )
        print("OK", mpi_rank(), part.local_indices)
        """,
        n_ranks,
        tmp_path,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.count("OK") == n_ranks


@requires_mpi
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_bipole_complete_output_cell_farming_matches_native(
    n_ranks, tmp_path
):
    """Farmed erfc J/K blocks retain the complete internal cell sum.

    The second pass emits only the home block, so ``np=3`` includes two
    empty ranks. Entering the native kernel there would trigger its legacy
    ``output_indices=[]`` means ALL behavior and this test would fail.
    """
    result = _run_under_mpi(
        f"""
        import numpy as np
        import vibeqc as vq
        from vibeqc._vibeqc_core import (
            LatticeSumOptions,
            build_jk_2e_real_space_domains,
            direct_lattice_cells,
            make_lattice_matrix_set,
        )
        from vibeqc.mpi import mpi_rank, mpi_size
        from vibeqc.pbc_bipole_fock import (
            _build_jk_domains_output_cells_mpi,
        )

        length = 8.0
        system = vq.PeriodicSystem(
            3,
            np.diag([length, length, length]),
            [
                vq.Atom(1, [4.0, 4.0, 3.3]),
                vq.Atom(1, [4.0, 4.0, 4.7]),
            ],
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        opts = LatticeSumOptions()
        opts.cutoff_bohr = 8.1
        opts.nuclear_cutoff_bohr = 8.1
        cells = list(direct_lattice_cells(system, 8.1))
        nbf = int(basis.nbasis)
        density_blocks = [
            np.eye(nbf) * (0.5 if tuple(cell.index) == (0, 0, 0) else 0.0)
            for cell in cells
        ]
        density = make_lattice_matrix_set(nbf, cells, density_blocks)
        assert mpi_size() == {n_ranks}

        for output_indices in (list(range(len(cells))), [0]):
            for compute_exchange in (False, True):
                reference = build_jk_2e_real_space_domains(
                    basis,
                    system,
                    opts,
                    density,
                    cells,
                    output_indices,
                    [],
                    0.3,
                    compute_exchange,
                )
                farmed = _build_jk_domains_output_cells_mpi(
                    basis,
                    system,
                    opts,
                    density,
                    cells,
                    output_indices,
                    omega=0.3,
                    compute_exchange=compute_exchange,
                    task_kind="chi-direct-output-cell",
                    strategy="cyclic",
                )
                for index in range(len(cells)):
                    np.testing.assert_array_equal(
                        np.asarray(farmed.J.blocks[index]),
                        np.asarray(reference.J.blocks[index]),
                    )
                    if compute_exchange:
                        np.testing.assert_array_equal(
                            np.asarray(farmed.K.blocks[index]),
                            np.asarray(reference.K.blocks[index]),
                        )
                executed = farmed.output_cell_farming_execution
                assert executed.world_size == {n_ranks}
                assert executed.global_task_count == len(output_indices)
                assert sum(executed.local_task_counts) == len(output_indices)
                assert executed.complete_internal_translation_sum is True
        print("OK", mpi_rank())
        """,
        n_ranks,
        tmp_path,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.count("OK") == n_ranks


@requires_mpi
def test_chi_rhf_output_cell_farming_matches_replicated_driver(tmp_path):
    """χ RHF selector parity against the same BIPOLE run without farming."""
    result = _run_under_mpi(
        """
        import warnings
        import numpy as np
        import vibeqc as vq
        from vibeqc.mpi import mpi_rank, mpi_size
        from vibeqc.periodic.chi.scf import cyclic_gamma_mesh

        system = vq.PeriodicSystem(
            3,
            np.diag([12.0, 12.0, 12.0]),
            [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        mesh = (2, 1, 1)
        kmesh = cyclic_gamma_mesh(system, mesh).to_bloch_kmesh()

        reference_options = vq.PeriodicRHFOptions()
        reference_options.max_iter = 1
        reference_options.lattice_opts.cutoff_bohr = 8.0
        reference_options.lattice_opts.nuclear_cutoff_bohr = 8.0
        reference = vq.run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            reference_options,
            fock_mixing=0.0,
            use_ewald_j_split=True,
            use_exchange_ewald_split=True,
            exchange_exxdiv="ewald",
            use_multipole_far_field=False,
            sr_image_precision=1.0e-6,
            farm_output_cells=False,
            progress=False,
        )

        chi_options = vq.PeriodicRHFOptions()
        chi_options.max_iter = 1
        chi_options.lattice_opts.cutoff_bohr = 8.0
        chi_options.lattice_opts.nuclear_cutoff_bohr = 8.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            farmed = vq.run_aiccm2026dev_b_rhf(
                system,
                basis,
                mesh,
                chi_options,
                backend="four_center",
                fock_mixing=0.0,
                progress=False,
            )

        assert mpi_size() == 2
        assert farmed.n_iter == reference.n_iter == 1
        assert farmed.energy == reference.energy
        assert farmed.e_electronic == reference.e_electronic
        for got, want in zip(farmed.fock, reference.fock):
            np.testing.assert_array_equal(got, want)
        for got, want in zip(farmed.density.blocks, reference.density.blocks):
            np.testing.assert_array_equal(got, want)
        executed = farmed.output_cell_farming_execution
        assert executed.active is True
        assert executed.world_size == 2
        assert executed.task_kind == "chi-direct-output-cell"
        assert executed.complete_internal_translation_sum is True
        assert farmed.aiccm2026dev_b.direct_output_cell_farming == executed
        print("OK", mpi_rank(), executed.local_task_count)
        """,
        2,
        tmp_path,
        expected_world_size=2,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.count("OK") == 2


@requires_mpi
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_chi_rks_output_cell_farming_matches_replicated_driver(
    tmp_path,
    n_ranks,
):
    """chi RKS selector parity against the same unfarmed PBE calculation."""
    result = _run_under_mpi(
        f"""
        import warnings
        import numpy as np
        import vibeqc as vq
        from vibeqc.mpi import mpi_rank, mpi_size
        from vibeqc.periodic.chi.scf import (
            _distributed_residue_inverse_bloch_transform,
            cyclic_gamma_mesh,
            inverse_bloch_transform,
            wigner_seitz_representatives,
        )

        system = vq.PeriodicSystem(
            3,
            np.diag([12.0, 12.0, 12.0]),
            [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        mesh = (2, 1, 1)
        character_mesh = cyclic_gamma_mesh(system, mesh)
        kmesh = character_mesh.to_bloch_kmesh()

        reference_options = vq.PeriodicKSOptions()
        reference_options.functional = "pbe"
        reference_options.max_iter = 1
        reference_options.lattice_opts.cutoff_bohr = 8.0
        reference_options.lattice_opts.nuclear_cutoff_bohr = 8.0
        reference = vq.run_pbc_bipole_rks(
            system,
            basis,
            kmesh,
            reference_options,
            functional="pbe",
            fock_mixing=0.0,
            use_ewald_j_split=True,
            use_exchange_ewald_split=True,
            exchange_exxdiv="ewald",
            use_multipole_far_field=False,
            sr_image_precision=1.0e-6,
            farm_output_cells=False,
            progress=False,
        )

        chi_options = vq.PeriodicKSOptions()
        chi_options.functional = "pbe"
        chi_options.max_iter = 1
        chi_options.lattice_opts.cutoff_bohr = 8.0
        chi_options.lattice_opts.nuclear_cutoff_bohr = 8.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            farmed = vq.run_aiccm2026dev_b_rks(
                system,
                basis,
                "pbe",
                mesh,
                chi_options,
                backend="four_center",
                fock_mixing=0.0,
                progress=False,
            )

        assert mpi_size() == {n_ranks}
        assert farmed.n_iter == reference.n_iter == 1
        assert farmed.energy == reference.energy
        assert farmed.e_electronic == reference.e_electronic
        assert farmed.e_xc == reference.e_xc
        for got, want in zip(farmed.fock, reference.fock):
            np.testing.assert_array_equal(got, want)
        for got, want in zip(farmed.density.blocks, reference.density.blocks):
            np.testing.assert_array_equal(got, want)
        executed = farmed.output_cell_farming_execution
        assert executed.active is True
        assert executed.world_size == {n_ranks}
        assert executed.task_kind == "chi-direct-output-cell"
        assert executed.complete_internal_translation_sum is True
        assert farmed.aiccm2026dev_b.direct_output_cell_farming == executed
        residue_execution = farmed.residue_inverse_transform_execution
        assert farmed.aiccm2026dev_b.residue_inverse_transform == residue_execution
        assert residue_execution.task_kind == "chi-residue-inverse-transform"
        assert residue_execution.active is True
        assert residue_execution.world_size == {n_ranks}
        assert residue_execution.n_residues == 2
        assert residue_execution.n_representatives == 3
        assert residue_execution.representative_multiplicities == (1, 2)
        assert residue_execution.task_counts == (
            {', '.join('1' if rank < 2 else '0' for rank in range(n_ranks))},
        )
        assert len(residue_execution.local_residue_indices) == (
            residue_execution.task_counts[mpi_rank()]
        )
        assert residue_execution.complete_character_sum is True
        assert residue_execution.supplied_representatives_grouped is True
        assert residue_execution.representative_scope == "zero-offset-cell"
        assert len(residue_execution.construction_fingerprint) == 64
        assert residue_execution.extra_wigner_weight_applied is False
        synthetic = [
            np.asarray([[2.0, 1.0j], [-1.0j, 3.0]], dtype=np.complex128),
            np.asarray([[1.0, -1.0j], [1.0j, 0.5]], dtype=np.complex128),
        ]
        distributed_blocks, synthetic_execution = (
            _distributed_residue_inverse_bloch_transform(
                synthetic,
                character_mesh.kpoints_frac,
                wigner_seitz_representatives(system, mesh),
                character_mesh.weights,
                mesh=mesh,
                representative_scope="zero-offset-cell",
            )
        )
        serial_blocks = inverse_bloch_transform(
            synthetic,
            character_mesh.kpoints_frac,
            synthetic_execution.residue_keys,
            character_mesh.weights,
        )
        np.testing.assert_array_equal(distributed_blocks, serial_blocks)
        assert synthetic_execution == residue_execution
        print("OK", mpi_rank(), executed.local_task_count)
        """,
        n_ranks,
        tmp_path,
        expected_world_size=n_ranks,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.count("OK") == n_ranks


@requires_mpi
def test_chi_residue_transform_rejects_cross_rank_construction_drift(tmp_path):
    result = _run_under_mpi(
        """
        import numpy as np
        import vibeqc as vq
        from vibeqc.mpi import mpi_rank
        from vibeqc.periodic.chi.scf import (
            _distributed_residue_inverse_bloch_transform,
            cyclic_gamma_mesh,
            wigner_seitz_representatives,
        )

        system = vq.PeriodicSystem(
            3,
            np.diag([12.0, 12.0, 12.0]),
            [vq.Atom(1, [0.0, 0.0, 0.0])],
        )
        mesh = (2, 1, 1)
        character_mesh = cyclic_gamma_mesh(system, mesh)
        try:
            _distributed_residue_inverse_bloch_transform(
                [np.asarray([[1.0]]), np.asarray([[0.5]])],
                character_mesh.kpoints_frac,
                wigner_seitz_representatives(system, mesh),
                character_mesh.weights,
                mesh=mesh,
                representative_scope=f"rank-specific-{mpi_rank()}",
            )
        except RuntimeError as exc:
            assert "construction fingerprint" in str(exc)
        else:
            raise AssertionError("cross-rank construction drift was accepted")
        print("OK", mpi_rank())
        """,
        2,
        tmp_path,
        expected_world_size=2,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.count("OK") == 2


@requires_mpi
def test_chi_uhf_runs_separate_spin_residue_collectives(tmp_path):
    result = _run_under_mpi(
        """
        import warnings
        import numpy as np
        import vibeqc as vq
        from vibeqc.mpi import mpi_rank, mpi_world

        system = vq.PeriodicSystem(
            3,
            np.diag([12.0, 12.0, 12.0]),
            [vq.Atom(1, [0.0, 0.0, 0.0])],
            multiplicity=2,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        options = vq.PeriodicRHFOptions()
        options.max_iter = 1
        options.lattice_opts.cutoff_bohr = 8.0
        options.lattice_opts.nuclear_cutoff_bohr = 8.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vq.run_aiccm2026dev_b_uhf(
                system,
                basis,
                (2, 1, 1),
                options,
                backend="four_center",
                fock_mixing=0.0,
                progress=False,
            )

        executed = result.residue_inverse_transform_execution
        assert executed.world_size == 2
        assert executed.task_counts == (1, 1)
        assert executed.representative_scope == "zero-offset-cell"
        assert executed.complete_character_sum is True
        assert result.aiccm2026dev_b.residue_inverse_transform == executed
        energies = mpi_world().comm.allgather(result.energy)
        np.testing.assert_array_equal(energies, [energies[0], energies[0]])
        print("OK", mpi_rank())
        """,
        2,
        tmp_path,
        expected_world_size=2,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.count("OK") == 2


@requires_mpi
def test_required_world_size_mismatch_fails_before_work(tmp_path):
    result = _run_under_mpi(
        """
        import vibeqc as vq
        vq.mpi_world()
        """,
        2,
        tmp_path,
        expected_world_size=3,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert (
        "world-size variables disagree" in output
        or "COMM_WORLD size contradicts" in output
    ), output


@requires_mpi
def test_inconsistent_partition_fails_loudly(tmp_path):
    """If ranks disagree on the partition the reassembly is short, and a
    silently short result would read as a converged answer. It must
    raise instead."""
    result = _run_under_mpi(
        """
        from vibeqc.mpi import KPointPartition, mpi_world

        w = mpi_world()
        # Rank 0 claims a larger mesh than its peers: the gathered
        # parcels cannot cover it.
        n_k = 8 if w.rank == 0 else 4
        part = KPointPartition.create(n_k)
        try:
            part.allgather_ordered([0] * part.n_local)
        except RuntimeError as exc:
            assert "ranks disagree" in str(exc), exc
            if w.rank == 0:
                print("RAISED")
        else:
            if w.rank == 0:
                raise SystemExit("expected RuntimeError on rank 0")
        """,
        2, tmp_path,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "RAISED" in result.stdout, result.stdout


@requires_mpi
@pytest.mark.parametrize("n_ranks", [2])
def test_multik_gdf_cderi_farming_matches_serial(n_ranks, tmp_path):
    """The first production k-point farming path (HANDOVER_MPI.md step 4).

    The multi-k GDF cderi cache is built by farming momentum-transfer
    groups across ranks and reassembling with one ordered gather. Every
    rank must end up with the *same complete* cache it would have built
    alone -- a dropped or misattached group would silently corrupt the
    exchange operator rather than crash.

    Compares the farmed cache against a serial rebuild of the same
    groups, element by element, inside the same run.
    """
    result = _run_under_mpi(
        """
        import numpy as np
        import vibeqc as vq
        from vibeqc.mpi import mpi_size, mpi_rank
        from vibeqc._vibeqc_core import LatticeSumOptions
        from vibeqc.aux_basis import (
            make_aux_basis_set, make_modrho_aux_basis,
            build_lpq_bloch_native_fft,
        )
        from vibeqc.periodic_k_gdf import _build_rsgdf_lpq_cache_shared_q

        a = 7.72
        lattice = 0.5 * a * np.array([[0., 1, 1], [1, 0, 1], [1, 1, 0]]).T
        system = vq.PeriodicSystem(
            3, lattice,
            [vq.Atom(3, [0., 0, 0]), vq.Atom(1, [0.5 * a, 0, 0])],
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        mol = system.unit_cell_molecule()
        aux = make_modrho_aux_basis(
            make_aux_basis_set(mol, aux_name="def2-svp-jk"), mol
        )
        recip = np.asarray(system.reciprocal_lattice(), dtype=float)
        kpoints = np.array([np.zeros(3), 0.5 * recip[:, 2]])
        opts = LatticeSumOptions()

        assert mpi_size() > 1, "expected a real multi-rank run"

        farmed = _build_rsgdf_lpq_cache_shared_q(
            system, basis, aux, kpoints, True,
            ke_cutoff=20.0, lat_opts=opts, linear_dep_thr=1e-9,
            fit_screen_threshold=0.0, progress=None, q_metric_cache={},
        )

        # Every rank holds the complete cache, not just its own share.
        assert set(farmed) == {(0, 0), (0, 1), (1, 0), (1, 1)}, (
            mpi_rank(), sorted(farmed)
        )

        # And every block equals an independent per-pair build.
        for (i, j), lpq in farmed.items():
            want = build_lpq_bloch_native_fft(
                system, basis, aux, kpoints[i], kpoints[j],
                ke_cutoff=20.0, lat_opts=opts, linear_dep_thr=1e-9,
                fit_screen_threshold=0.0,
            )
            assert lpq.shape == want.shape, ((i, j), lpq.shape, want.shape)
            assert np.allclose(lpq, want, atol=2e-12, rtol=2e-12), (
                mpi_rank(), (i, j), float(np.max(np.abs(lpq - want)))
            )
        print("OK", mpi_rank())
        """,
        n_ranks, tmp_path,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.count("OK") == n_ranks, result.stdout


@requires_mpi
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_multik_gdf_exchange_farming_matches_serial(n_ranks, tmp_path):
    """Per-iteration GDF exchange farming (HANDOVER_MPI.md step 4c).

    Distinct from the cderi farming above, which runs once per SCF
    *setup*. This is the O(n_k^2) fitted-exchange contraction every
    multi-k GDF iteration runs, farmed over the **bra** index: each rank
    computes complete ``K(k_i)`` matrices for its own share, so nothing
    is partially summed across ranks and the farmed result must be
    *exactly* the serial one, not merely close.

    The failure this guards is the one an ordered gather exists to
    prevent: a rank-order flatten would attach ``K`` to the wrong
    k-point, which produces a plausible-looking but wrong exchange
    operator rather than a crash.
    """
    result = _run_under_mpi(
        """
        import numpy as np
        from vibeqc.mpi import mpi_size, mpi_rank
        from vibeqc.periodic_k_gdf import (
            _build_k_from_lpq_cache,
            _k_from_densities_dense,
        )

        assert mpi_size() > 1, "expected a real multi-rank run"

        rng = np.random.default_rng(20260805)
        n_k, naux, nbf, n_occ = 6, 14, 7, 2
        lpq = {}
        for i in range(n_k):
            for j in range(n_k):
                lpq[(i, j)] = (
                    rng.standard_normal((naux, nbf, nbf))
                    + 1j * rng.standard_normal((naux, nbf, nbf))
                )
        W = [
            rng.standard_normal((nbf, n_occ))
            + 1j * rng.standard_normal((nbf, n_occ))
            for _ in range(n_k)
        ]
        dens = [w @ w.conj().T for w in W]
        wts = np.full(n_k, 1.0 / n_k)

        farmed = _build_k_from_lpq_cache(lpq, dens, wts, nbasis=nbf)
        # Every rank holds the COMPLETE list, in global k order.
        assert len(farmed) == n_k, (mpi_rank(), len(farmed))

        # Serial reference computed independently on every rank via the
        # historical dense contraction, which does no farming at all.
        ref = _k_from_densities_dense(lpq, dens, wts, range(n_k), nbasis=nbf)
        scale = max(float(np.max(np.abs(k))) for k in ref)
        worst = max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(ref, farmed)
        )
        assert worst < 1e-12 * scale, (mpi_rank(), worst)

        # Order matters: shifting the farmed list by one k must break it,
        # so the test would fail if the gather ever reassembled by rank.
        shifted = farmed[1:] + farmed[:1]
        misordered = max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(ref, shifted)
        )
        assert misordered > 1e-6 * scale, "fixture too symmetric to detect order"

        print("OK", mpi_rank(), mpi_size())
        """,
        n_ranks,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("OK") == n_ranks, result.stdout
