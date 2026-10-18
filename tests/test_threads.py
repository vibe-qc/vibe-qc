"""Phase P1.1: thread-count control + run_job timing block."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq


def _supports_exact_df_blocking(core) -> bool:
    """Return whether fixed DF blocks have a byte-exact positive control."""
    info = core.blas_info()
    backend = info["libraries"].lower()
    return info["blas_enabled"] and (
        "openblas" in backend or "accelerate" in backend
    )


# Cross-thread tolerances for the two routes that are not engineered for
# bitwise thread invariance (#323). The shared-builder J and paired K, the
# fixed-block DF half transform and the XC batch reduction are, and their
# tests below stay bitwise. What varies is dense LAPACK work outside those
# kernels: the Cholesky factor of the Coulomb metric while a builder is
# constructed (by elimination: the metric, the 3c tensor, the fixed-block
# transform and the shared-builder J are all bitwise), and the in-field Fock
# diagonalisations of the PATOM guess (its J and paired K are bitwise). A
# LAPACK that threads these calls reassociates its sums with the worker
# count; builds whose LAPACK does not are bitwise identical.
# Measured on an x86 build with a threaded system OpenBLAS and LAPACKE, at 2,
# 4 and 8 workers against 1, on the decks of the two tests below:
#   J from builders constructed at N workers: max |dJ| = 9.70e-13
#     (394 u * max|J|, u = 2**-53, max|J| = 22.1);
#   PATOM densities: max |dD| = 6.62e-12 elementwise; spectral norm
#     2.48e-11 = 439 u * ||F||_2 / gap (||F||_2 = 152.7, HOMO-LUMO gaps
#     0.333 and 0.300 Ha), the Davis-Kahan scale of a rounding-level Fock
#     perturbation.
# Each tolerance is ten times the largest measured spread, rounded up
# (margins 10.3 and 10.6).
_DF_J_CROSS_THREAD_ATOL = 1.0e-11
_PATOM_DENSITY_CROSS_THREAD_ATOL = 7.0e-11


# ---------------------------------------------------------------------------
# Thread-count API
# ---------------------------------------------------------------------------

def test_get_num_threads_returns_positive():
    n = vq.get_num_threads()
    assert isinstance(n, int)
    assert n >= 1


def test_set_num_threads_changes_count():
    initial = vq.get_num_threads()
    try:
        assert vq.set_num_threads(2) == 2
        assert vq.get_num_threads() == 2

        assert vq.set_num_threads(1) == 1
        assert vq.get_num_threads() == 1
    finally:
        # Restore so we don't affect later tests.
        vq.set_num_threads(initial)


def test_set_num_threads_zero_restores_default():
    initial = vq.get_num_threads()
    try:
        vq.set_num_threads(2)
        assert vq.get_num_threads() == 2
        # n <= 0 should restore a sensible default (either OMP_NUM_THREADS
        # or omp_get_num_procs). At minimum it should not leave us pinned
        # at 2 — it's always ≥ 1.
        restored = vq.set_num_threads(0)
        assert restored >= 1
        # On any non-uniprocessor box the restored default is > 2.
        # We don't assume that's always the case (CI can run with
        # OMP_NUM_THREADS=1 or on a single-core runner), so just check
        # the "not stuck at 2 when num_procs > 2" behavior softly.
    finally:
        vq.set_num_threads(initial)


def test_set_num_threads_negative_restores_default():
    initial = vq.get_num_threads()
    try:
        vq.set_num_threads(3)
        vq.set_num_threads(-1)    # treated as "restore default"
        assert vq.get_num_threads() >= 1
    finally:
        vq.set_num_threads(initial)


# ---------------------------------------------------------------------------
# run_job integration: num_threads, timing block in .out
# ---------------------------------------------------------------------------

def _h2o() -> vq.Molecule:
    return vq.Molecule([
        vq.Atom(8, [0, 0, 0]),
        vq.Atom(1, [0, 1.43, -0.98]),
        vq.Atom(1, [0, -1.43, -0.98]),
    ])


def test_run_job_num_threads_pins_thread_count():
    initial = vq.get_num_threads()
    try:
        with tempfile.TemporaryDirectory() as d:
            out_stem = Path(d) / "run"
            vq.run_job(
                _h2o(), basis="sto-3g", method="rhf", output=out_stem,
                num_threads=2, write_molden_file=False,
            )
            text = (out_stem.with_suffix(".out")).read_text()
        assert "Threads: 2" in text
        # After the run, the thread count should still be what we asked
        # for (run_job doesn't undo the set).
        assert vq.get_num_threads() == 2
    finally:
        vq.set_num_threads(initial)


def test_run_job_writes_timing_block():
    with tempfile.TemporaryDirectory() as d:
        out_stem = Path(d) / "run"
        vq.run_job(
            _h2o(), basis="sto-3g", method="rhf", output=out_stem,
            write_molden_file=False,
        )
        text = (out_stem.with_suffix(".out")).read_text()
    assert "Timings (wall clock, seconds)" in text
    assert "SCF total" in text
    assert "SCF avg. per iteration" in text
    assert "Job total" in text
    assert "OpenMP thread" in text


def test_run_job_num_threads_none_preserves_setting():
    """Passing num_threads=None should NOT mutate the current thread
    count. The recorded thread count is whatever the caller set before
    the run."""
    initial = vq.get_num_threads()
    try:
        vq.set_num_threads(3)
        with tempfile.TemporaryDirectory() as d:
            out_stem = Path(d) / "run"
            vq.run_job(
                _h2o(), basis="sto-3g", method="rhf", output=out_stem,
                write_molden_file=False,
            )
            text = (out_stem.with_suffix(".out")).read_text()
        assert "Threads: 3" in text
        # Thread count still 3 after the job.
        assert vq.get_num_threads() == 3
    finally:
        vq.set_num_threads(initial)


@pytest.mark.parametrize("method", ["rks", "uks"])
def test_molecular_xc_batches_use_requested_threads_without_changing_result(
    method: str,
):
    """Independent XC batches must use the team and reduce in grid order."""
    initial = vq.get_num_threads()
    try:
        if method == "rks":
            molecule = vq.Molecule([
                vq.Atom(1, [0.0, 0.0, -0.7]),
                vq.Atom(1, [0.0, 0.0, 0.7]),
            ])
            options = vq.RKSOptions()
            runner = vq.run_rks
        else:
            molecule = vq.Molecule(
                [vq.Atom(1, [0.0, 0.0, 0.0])],
                multiplicity=2,
            )
            options = vq.UKSOptions()
            options.stability_check = False
            runner = vq.run_uks

        options.functional = "PBE"
        options.grid.n_radial = 32
        options.grid.n_theta = 17
        options.grid.n_phi = 36
        options.max_iter = 80
        options.conv_tol_energy = 1e-10
        options.conv_tol_grad = 1e-7
        basis = vq.BasisSet(molecule, "sto-3g")

        vq.set_num_threads(1)
        serial = runner(molecule, basis, options)
        vq.set_num_threads(4)
        parallel = runner(molecule, basis, options)

        assert serial.converged and parallel.converged
        assert serial.xc_batch_workers_used == 1
        assert parallel.xc_batch_workers_used == 4
        assert parallel.energy.hex() == serial.energy.hex()
        assert parallel.e_xc.hex() == serial.e_xc.hex()
        if method == "rks":
            assert np.array_equal(parallel.density, serial.density)
        else:
            assert np.array_equal(
                parallel.density_alpha, serial.density_alpha
            )
            assert np.array_equal(
                parallel.density_beta, serial.density_beta
            )
    finally:
        vq.set_num_threads(initial)


def test_density_fitting_j_uses_requested_threads_without_changing_result():
    """RI-J must use outer workers without changing auxiliary ordering."""
    from vibeqc import _vibeqc_core as core

    initial = vq.get_num_threads()
    try:
        molecule = vq.Molecule([
            vq.Atom(6, [0.0, 0.0, z])
            for z in (-3.75, -1.25, 1.25, 3.75)
        ])
        basis = vq.BasisSet(molecule, "def2-qzvpp")
        aux = vq.BasisSet(molecule, "def2-qzvpp-j")
        builder = core.make_df_jk_builder(basis, aux)

        rng = np.random.default_rng(204)
        raw = rng.standard_normal((basis.nbasis, basis.nbasis))
        density = 0.5 * (raw + raw.T)

        vq.set_num_threads(1)
        serial = np.asarray(builder.build_J(density))
        assert builder.last_j_workers_used == 1

        vq.set_num_threads(4)
        parallel = np.asarray(builder.build_J(density))
        assert builder.last_j_workers_used == 4
        assert np.array_equal(parallel, serial)
        repeated = np.asarray(builder.build_J(density))
        assert builder.last_j_workers_used == 4
        assert np.array_equal(repeated, serial)
    finally:
        vq.set_num_threads(initial)


def test_density_fitting_transform_uses_threads_without_changing_j():
    """The multi-RHS B transform must preserve every fitted coefficient."""
    from vibeqc import _vibeqc_core as core

    initial = vq.get_num_threads()
    try:
        molecule = vq.Molecule([
            vq.Atom(6, [0.0, 0.0, z])
            for z in (-3.75, -1.25, 1.25, 3.75)
        ])
        basis = vq.BasisSet(molecule, "def2-qzvpp")
        aux = vq.BasisSet(molecule, "def2-qzvpp-j")

        vq.set_num_threads(1)
        serial_builder = core.make_df_jk_builder(basis, aux)
        assert serial_builder.last_df_transform_workers_used == 1
        assert serial_builder.last_df_pack_workers_used == 1
        assert serial_builder.last_df_unpack_workers_used == 1

        rng = np.random.default_rng(2204)
        raw = rng.standard_normal((basis.nbasis, basis.nbasis))
        density = 0.5 * (raw + raw.T)
        serial_j = np.asarray(serial_builder.build_J(density)).copy()
        del serial_builder

        vq.set_num_threads(4)
        parallel_builder = core.make_df_jk_builder(basis, aux)
        expected_transform_workers = (
            4 if _supports_exact_df_blocking(core) else 1
        )
        assert (
            parallel_builder.last_df_transform_workers_used
            == expected_transform_workers
        )
        assert parallel_builder.last_df_pack_workers_used == 4
        assert parallel_builder.last_df_unpack_workers_used == 4
        parallel_j = np.asarray(parallel_builder.build_J(density))
        # Not bitwise across worker counts (#323): the metric Cholesky may
        # run in threaded LAPACK. See _DF_J_CROSS_THREAD_ATOL.
        np.testing.assert_allclose(
            parallel_j, serial_j, rtol=0.0, atol=_DF_J_CROSS_THREAD_ATOL
        )
    finally:
        vq.set_num_threads(initial)


def test_density_fitting_blocked_transform_is_bitwise_thread_invariant():
    """Fixed RHS boundaries keep identical bytes regardless of their owner."""
    from vibeqc import _vibeqc_core as core

    if not _supports_exact_df_blocking(core):
        pytest.skip("fixed DF blocks require a byte-exact BLAS backend")

    initial = vq.get_num_threads()
    try:
        rng = np.random.default_rng(3204)
        lower = np.tril(0.01 * rng.standard_normal((96, 96)))
        lower[np.diag_indices_from(lower)] += 2.0
        lower = np.asarray(lower, order="F")
        rhs = np.asarray(rng.standard_normal((96, 2049)), order="F")

        vq.set_num_threads(1)
        legacy, blocked, workers = core._df_half_transform_for_test(
            lower, rhs
        )
        assert workers == 1
        serial_blocked = np.asarray(blocked).copy()
        serial_bits = serial_blocked.view(np.uint64)
        assert np.array_equal(
            serial_bits, np.asarray(legacy).view(np.uint64)
        )

        for requested in (4, 8):
            vq.set_num_threads(requested)
            _legacy, blocked, workers = core._df_half_transform_for_test(
                lower, rhs
            )
            assert workers == requested
            assert np.array_equal(
                np.asarray(blocked).view(np.uint64), serial_bits
            )
            _, repeated, repeated_workers = core._df_half_transform_for_test(
                lower, rhs
            )
            assert repeated_workers == requested
            assert np.array_equal(
                np.asarray(repeated).view(np.uint64), serial_bits
            )
    finally:
        vq.set_num_threads(initial)


def test_density_fitting_transform_small_problem_stays_serial():
    """Small B transforms avoid teams and their blocked-BLAS overhead."""
    from vibeqc import _vibeqc_core as core

    initial = vq.get_num_threads()
    try:
        molecule = vq.Molecule([
            vq.Atom(1, [0.0, 0.0, -0.7]),
            vq.Atom(1, [0.0, 0.0, 0.7]),
        ])
        basis = vq.BasisSet(molecule, "sto-3g")
        aux = vq.BasisSet(molecule, "def2-svp-jk")

        vq.set_num_threads(4)
        builder = core.make_df_jk_builder(basis, aux)
        assert builder.last_df_transform_workers_used == 1
        assert builder.last_df_pack_workers_used == 1
        assert builder.last_df_unpack_workers_used == 1
    finally:
        vq.set_num_threads(initial)


def test_density_fitting_j_small_problem_stays_serial():
    """Small RI contractions must avoid an OpenMP team-launch regression."""
    from vibeqc import _vibeqc_core as core

    initial = vq.get_num_threads()
    try:
        molecule = vq.Molecule([
            vq.Atom(1, [0.0, 0.0, -0.7]),
            vq.Atom(1, [0.0, 0.0, 0.7]),
        ])
        basis = vq.BasisSet(molecule, "sto-3g")
        aux = vq.BasisSet(molecule, "def2-svp-jk")
        builder = core.make_df_jk_builder(basis, aux)
        density = np.eye(basis.nbasis)

        vq.set_num_threads(1)
        serial = np.asarray(builder.build_J(density))
        vq.set_num_threads(4)
        parallel = np.asarray(builder.build_J(density))

        assert builder.last_j_workers_used == 1
        assert np.array_equal(parallel, serial)
    finally:
        vq.set_num_threads(initial)


def test_density_fitting_paired_k_uses_workers_and_preserves_spin_chains():
    """Supported BLAS RI-K blocks preserve both beta=1 accumulation streams."""
    from vibeqc import _vibeqc_core as core

    initial = vq.get_num_threads()
    try:
        molecule = vq.Molecule([
            vq.Atom(6, [0.0, 0.0, z])
            for z in (-6.25, -3.75, -1.25, 1.25, 3.75, 6.25)
        ])
        basis = vq.BasisSet(molecule, "def2-qzvpp")
        aux = vq.BasisSet(molecule, "def2-qzvpp-j")
        builder = core.make_df_jk_builder(basis, aux)

        rng = np.random.default_rng(1204)
        raw_alpha = rng.standard_normal((basis.nbasis, basis.nbasis))
        raw_beta = rng.standard_normal((basis.nbasis, basis.nbasis))
        density_alpha = 0.5 * (raw_alpha + raw_alpha.T)
        density_beta = 0.5 * (raw_beta + raw_beta.T)

        vq.set_num_threads(1)
        serial_alpha = np.asarray(builder.build_K(density_alpha))
        assert builder.last_k_workers_used == 1
        serial_beta = np.asarray(builder.build_K(density_beta))

        vq.set_num_threads(4)
        parallel_single = np.asarray(builder.build_K(density_alpha))
        expected_workers = 4 if _supports_exact_df_blocking(core) else 1
        assert builder.last_k_workers_used == expected_workers
        assert np.array_equal(parallel_single, serial_alpha)
        repeated_single = np.asarray(builder.build_K(density_alpha))
        assert builder.last_k_workers_used == expected_workers
        assert np.array_equal(repeated_single, serial_alpha)

        parallel_alpha, parallel_beta = builder.build_K_pair(
            density_alpha, density_beta
        )
        assert builder.last_k_pair_workers_used == expected_workers
        assert np.array_equal(np.asarray(parallel_alpha), serial_alpha)
        assert np.array_equal(np.asarray(parallel_beta), serial_beta)
        repeated_alpha, repeated_beta = builder.build_K_pair(
            density_alpha, density_beta
        )
        assert builder.last_k_pair_workers_used == expected_workers
        assert np.array_equal(np.asarray(repeated_alpha), serial_alpha)
        assert np.array_equal(np.asarray(repeated_beta), serial_beta)
    finally:
        vq.set_num_threads(initial)


def test_density_fitting_paired_k_small_problem_stays_serial():
    """Small paired RI-K builds retain the sequential generic path."""
    from vibeqc import _vibeqc_core as core

    initial = vq.get_num_threads()
    try:
        molecule = vq.Molecule([
            vq.Atom(1, [0.0, 0.0, -0.7]),
            vq.Atom(1, [0.0, 0.0, 0.7]),
        ])
        basis = vq.BasisSet(molecule, "sto-3g")
        aux = vq.BasisSet(molecule, "def2-svp-jk")
        builder = core.make_df_jk_builder(basis, aux)
        density = np.eye(basis.nbasis)

        vq.set_num_threads(1)
        serial_single = np.asarray(builder.build_K(density))
        serial = builder.build_K_pair(density, 0.5 * density)
        vq.set_num_threads(4)
        parallel_single = np.asarray(builder.build_K(density))
        paired = builder.build_K_pair(density, 0.5 * density)

        assert builder.last_k_workers_used == 1
        assert np.array_equal(parallel_single, serial_single)
        assert builder.last_k_pair_workers_used == 1
        assert np.array_equal(np.asarray(paired[0]), np.asarray(serial[0]))
        assert np.array_equal(np.asarray(paired[1]), np.asarray(serial[1]))
    finally:
        vq.set_num_threads(initial)


def test_open_shell_patom_uses_density_fitting_paired_k():
    """The production PATOM guess must consume the paired DF-K seam."""
    from vibeqc import _vibeqc_core as core

    initial = vq.get_num_threads()
    try:
        molecule = vq.Molecule([
            vq.Atom(6, [0.0, 0.0, z])
            for z in (-6.25, -3.75, -1.25, 1.25, 3.75, 6.25)
        ])
        basis = vq.BasisSet(molecule, "def2-qzvpp")
        aux = vq.BasisSet(molecule, "def2-qzvpp-j")
        builder = core.make_df_jk_builder(basis, aux)
        overlap = core.compute_overlap(basis)
        hcore = (
            core.compute_kinetic(basis)
            + core.compute_nuclear(basis, molecule)
        )

        vq.set_num_threads(1)
        serial_alpha, serial_beta = (
            core._guess_open_shell_density_with_jk(
                molecule,
                basis,
                19,
                17,
                core.InitialGuess.PATOM,
                overlap,
                hcore,
                builder,
                True,
            )
        )
        assert builder.last_k_pair_workers_used == 1

        vq.set_num_threads(4)
        density_alpha, density_beta = (
            core._guess_open_shell_density_with_jk(
                molecule,
                basis,
                19,
                17,
                core.InitialGuess.PATOM,
                overlap,
                hcore,
                builder,
                True,
            )
        )
        assert builder.last_k_pair_workers_used == 4
        # Not bitwise across worker counts (#323): the in-field Fock
        # diagonalisations may run in threaded LAPACK. See
        # _PATOM_DENSITY_CROSS_THREAD_ATOL.
        np.testing.assert_allclose(
            density_alpha, serial_alpha, rtol=0.0,
            atol=_PATOM_DENSITY_CROSS_THREAD_ATOL,
        )
        np.testing.assert_allclose(
            density_beta, serial_beta, rtol=0.0,
            atol=_PATOM_DENSITY_CROSS_THREAD_ATOL,
        )
        assert not np.array_equal(density_alpha, density_beta)
    finally:
        vq.set_num_threads(initial)


def test_open_shell_patom_dispatches_python_paired_k_override():
    """The C++ PATOM path must honor a Python paired-K implementation."""
    from vibeqc import _vibeqc_core as core

    molecule = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, -0.7]),
        vq.Atom(1, [0.0, 0.0, 0.7]),
    ])
    basis = vq.BasisSet(molecule, "sto-3g")
    overlap = core.compute_overlap(basis)
    hcore = core.compute_kinetic(basis) + core.compute_nuclear(
        basis, molecule
    )

    class _PairedBuilder(core.JKBuilder):
        def __init__(self):
            super().__init__()
            self.j_calls = 0
            self.k_calls = 0
            self.pair_calls = 0

        def build_J(self, density):
            self.j_calls += 1
            return np.zeros_like(np.asarray(density))

        def build_K(self, density):
            self.k_calls += 1
            return np.zeros_like(np.asarray(density))

        def build_K_pair(self, density_alpha, density_beta):
            self.pair_calls += 1
            return (
                np.zeros_like(np.asarray(density_alpha)),
                np.zeros_like(np.asarray(density_beta)),
            )

    builder = _PairedBuilder()
    density_alpha, density_beta = core._guess_open_shell_density_with_jk(
        molecule,
        basis,
        1,
        1,
        core.InitialGuess.PATOM,
        overlap,
        hcore,
        builder,
        True,
    )

    assert density_alpha.shape == (basis.nbasis, basis.nbasis)
    assert density_beta.shape == (basis.nbasis, basis.nbasis)
    assert builder.pair_calls > 0
    assert builder.k_calls == 0


def test_open_shell_sad_dispatches_single_k_override():
    """The SAD trampoline must dispatch through the single-K seam."""
    from vibeqc import _vibeqc_core as core

    molecule = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, -0.7]),
        vq.Atom(1, [0.0, 0.0, 0.7]),
    ])
    basis = vq.BasisSet(molecule, "sto-3g")
    overlap = core.compute_overlap(basis)
    hcore = core.compute_kinetic(basis) + core.compute_nuclear(
        basis, molecule
    )

    class _SingleBuilder(core.JKBuilder):
        def __init__(self):
            super().__init__()
            self.j_calls = 0
            self.k_calls = 0
            self.pair_calls = 0

        def build_J(self, density):
            self.j_calls += 1
            return np.zeros_like(np.asarray(density))

        def build_K(self, density):
            self.k_calls += 1
            return np.zeros_like(np.asarray(density))

        def build_K_pair(self, density_alpha, density_beta):
            self.pair_calls += 1
            return (
                np.zeros_like(np.asarray(density_alpha)),
                np.zeros_like(np.asarray(density_beta)),
            )

    builder = _SingleBuilder()
    density_alpha, density_beta = core._guess_open_shell_density_with_jk(
        molecule,
        basis,
        1,
        1,
        core.InitialGuess.SAD,
        overlap,
        hcore,
        builder,
        True,
    )

    assert density_alpha.shape == (basis.nbasis, basis.nbasis)
    assert density_beta.shape == (basis.nbasis, basis.nbasis)
    assert builder.j_calls == 1
    assert builder.k_calls == 1
    assert builder.pair_calls == 0
