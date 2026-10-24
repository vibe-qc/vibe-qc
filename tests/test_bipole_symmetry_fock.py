"""Tests for SYM3b Fock symmetry enforcement in BIPOLE."""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    InitialGuess,
    LatticeSumOptions,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    attach_symmetry,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    monkhorst_pack,
)
from vibeqc.pbc_bipole import run_pbc_bipole_rhf


def _he_sc():
    a = 4.0
    lattice = np.eye(3) * a
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    attach_symmetry(system)
    return system


@pytest.mark.parametrize("copy_budget", [0, 64 * 1024**2])
def test_symmetry_fock_dense_scatter_owns_writable_disjoint_blocks(monkeypatch, copy_budget):
    from vibeqc import bipole_symmetry_fock as symmetry
    from vibeqc.bipole_symmetry_fock import cell_orbit_mapping, symmetrize_fock_blocks
    from vibeqc._vibeqc_core import direct_lattice_cells

    system = _he_sc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "6-31g*")
    cells = list(direct_lattice_cells(system, 4.1))
    mapping = cell_orbit_mapping(system, basis, cells)
    assert mapping is not None
    monkeypatch.setattr(symmetry, "_DENSE_SCATTER_MAX_BYTES", copy_budget)
    original = np.random.default_rng(901).normal(size=(len(cells), basis.nbasis, basis.nbasis))
    saved = original.copy()
    original.flags.writeable = False
    blocks = list(original)
    identity = id(blocks)
    symmetrize_fock_blocks(blocks, mapping)
    assert id(blocks) == identity
    assert all(b.flags.writeable and not np.shares_memory(b, original) for b in blocks)
    np.testing.assert_array_equal(original, saved)
    dense = saved.copy()
    symmetrize_fock_blocks(dense, mapping)
    np.testing.assert_array_equal(dense, blocks)
    rest = np.asarray(blocks[1:]).copy()
    blocks[0][:] = 99.
    np.testing.assert_array_equal(blocks[1:], rest)


def test_symmetry_fock_enforcement_is_projection():
    """Symmetry enforcement is a projection: a no-op on already-symmetric
    blocks, and idempotent on perturbed ones.

    The mapping tuple is opaque to callers — this test deliberately does
    not unpack it (the 2026-06-09 version exposed internals that encoded
    a whole-matrix transformation law only valid for single-atom-at-origin
    cells)."""
    from vibeqc.bipole_symmetry_fock import cell_orbit_mapping, symmetrize_fock_blocks

    system = _he_sc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 6.0
    cells = list(compute_overlap_lattice(basis, system, opts).cells)

    mapping = cell_orbit_mapping(system, basis, cells)
    assert mapping is not None, "cell_orbit_mapping returned None"

    # T + S blocks of the symmetric crystal are already symmetry-adapted:
    # enforcement must reproduce them at machine precision.
    S_lat = compute_overlap_lattice(basis, system, opts)
    T_lat = compute_kinetic_lattice(basis, system, opts)
    fock = [
        np.asarray(T_lat.blocks[c], dtype=float)
        + np.asarray(S_lat.blocks[c], dtype=float)
        for c in range(len(S_lat.cells))
    ]
    fock_before = [np.copy(b) for b in fock]
    symmetrize_fock_blocks(fock, mapping)
    for c, (before, after) in enumerate(zip(fock_before, fock)):
        max_diff = float(np.max(np.abs(before - after)))
        assert max_diff < 1e-11, (
            f"symmetric input changed at cell {c}: {max_diff:.2e}"
        )

    # Perturbed (symmetry-broken) blocks: applying the enforcement twice
    # gives the same result as once (projection property).
    rng = np.random.default_rng(11)
    fock_pert = [
        b + 1e-3 * rng.standard_normal(b.shape) for b in fock_before
    ]
    once = [np.copy(b) for b in fock_pert]
    symmetrize_fock_blocks(once, mapping)
    twice = [np.copy(b) for b in once]
    symmetrize_fock_blocks(twice, mapping)
    for c, (b1, b2) in enumerate(zip(once, twice)):
        max_diff = float(np.max(np.abs(b1 - b2)))
        assert max_diff < 1e-12, f"not idempotent at cell {c}: {max_diff:.2e}"


@pytest.mark.parametrize("attach", [False, True])
def test_legacy_mgo_symmetry_fixture_refuses_truncated_overlap(attach):
    """Attaching symmetry must not bypass the overlap-fold refusal (#66).

    The historical cutoff10 fixture cannot establish energy invariance:
    its S(Gamma) drift is 0.015, and SCF must refuse it before iteration.
    S/T reconstruction itself is tested above on a complete orbit domain.
    """
    from vibeqc.pbc_bipole_common import BipoleFoldUnreliableError

    a = 4.21 / 0.529177210903
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3, lattice,
        [vq.Atom(12, [0.0, 0.0, 0.0]),
         vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0])],
    )
    if attach:
        attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.max_iter = 50
    opts.use_diis = True
    opts.conv_tol_energy = 1e-8
    opts.initial_guess = InitialGuess.SAD
    with pytest.raises(BipoleFoldUnreliableError, match="S.*fold"):
        run_pbc_bipole_rhf(
            system, basis, monkhorst_pack(system, [1, 1, 1]), opts,
            historical_domain=True,  # Preserve this historical cutoff-10 witness.
            use_ewald_j_split=True,
            use_exchange_ewald_split=False,
            use_fock_symmetry_reduce=False,
            sr_image_precision=None,
            ewald_precision=1e-8,
            progress=False,
        )


def test_symmetry_fock_reconstruction_runs_with_bipole():
    from vibeqc.bipole_symmetry_fock import cell_orbit_mapping, symmetrize_fock_blocks

    system = _he_sc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False

    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
        use_fock_symmetry_reduce=False,
        sr_image_precision=None,
    )
    assert result.n_iter == 1

    # The orbit mapping must be built on the OPERATOR cell list (what
    # the driver symmetrizes). Under the Ewald exchange split the
    # density set is intentionally WIDER (2× cutoff) than the operator
    # support, so ``result.density.cells`` is no longer the right list
    # to pair with operator blocks.
    S_lat = compute_overlap_lattice(basis, system, opts.lattice_opts)
    cells = list(S_lat.cells)
    mapping = cell_orbit_mapping(system, basis, cells)

    assert mapping is not None, "cell_orbit_mapping returned None for He sc"
    # Build test Fock blocks from overlap (simple test of the symmetry rotation).
    mapping2 = cell_orbit_mapping(system, basis, cells)
    assert mapping2 is not None
    fock_blocks = [
        np.asarray(S_lat.blocks[c], dtype=float) for c in range(len(S_lat.cells))
    ]
    symmetrize_fock_blocks(fock_blocks, mapping2)
    for b in fock_blocks:
        assert np.all(np.isfinite(b))


def test_bipole_symmetry_fock_preserves_energy():
    """SYM3b Fock symmetry enforcement must not change the SCF energy
    on a symmetric system (He in simple cubic)."""
    # The physical pair default (#316) refuses the legacy SYM3b projector;
    # this test pins the historical composition, so it replays it explicitly
    # (#503).
    system = _he_sc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False

    result_no_sym = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
        historical_domain=True,
        use_fock_symmetry=False,
        use_fock_symmetry_reduce=False,
        sr_image_precision=None,
    )
    result_with_sym = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
        historical_domain=True,
        use_fock_symmetry=True,
        use_fock_symmetry_reduce=False,
        sr_image_precision=None,
    )
    assert result_no_sym.n_iter == result_with_sym.n_iter == 1
    assert abs(result_no_sym.energy - result_with_sym.energy) < 1e-12, (
        f"symmetry Fock changed energy: "
        f"{result_no_sym.energy:.12f} vs {result_with_sym.energy:.12f}"
    )


def _mgo_sym(cutoff: float):
    ang2bohr = 1.0 / 0.529177210903
    a = 4.21 * ang2bohr
    lattice = (a / 2.0) * np.array([
        [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2] * 3)])
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lat = LatticeSumOptions()
    lat.cutoff_bohr = cutoff
    lat.nuclear_cutoff_bohr = cutoff
    return system, basis, lat


@pytest.mark.parametrize("compute_coulomb", [False, True])
@pytest.mark.parametrize("internal_extent", [None, 10.0])
def test_sym3b_reduced_build_equals_symmetrized_full_build(compute_coulomb, internal_extent):
    """SYM3b core: ``build_jk_reduced_symmetrized`` (build J/K only at orbit-
    representative cells with the full internal lattice sum, then reconstruct
    by rotation) is BIT-IDENTICAL to ``symmetrize_fock_blocks`` applied to a
    full ``build_jk_2e_real_space``.

    This is the |G|-fold compute reduction (Phase SYM3b) agreeing with the
    SYM3 storage round-trip to machine precision (``symmetry_integrals``), so
    the reduced direct-ERI build is a pure speedup, not an approximation
    relative to the symmetrized operator.
    """
    from vibeqc._vibeqc_core import build_jk_2e_real_space
    from vibeqc.bipole_symmetry_fock import (
        build_jk_reduced_symmetrized,
        cell_orbit_mapping,
        representative_cell_indices,
        symmetrize_fock_blocks,
    )

    system, basis, lat = _mgo_sym(8.0)
    density = _stand_in_density(basis, system, lat)
    if internal_extent is None:
        full = build_jk_2e_real_space(basis, system, lat, density, 0.30)
    else:
        from vibeqc.pbc_bipole_fock import _sr_image_padded_jk

        full = _sr_image_padded_jk(basis, system, lat, density, 0.30, internal_extent)
    # Use the JK build's OWN cell list so rep indices line up with the output.
    mapping = cell_orbit_mapping(system, basis, list(full.J.cells))
    assert mapping is not None
    reps = representative_cell_indices(mapping)
    assert 0 < len(reps) < len(list(full.J.cells))  # a genuine reduction

    j_ref = [np.asarray(b, dtype=float) for b in full.J.blocks]
    k_ref = [np.asarray(b, dtype=float) for b in full.K.blocks]
    symmetrize_fock_blocks(j_ref, mapping)
    symmetrize_fock_blocks(k_ref, mapping)

    red = build_jk_reduced_symmetrized(
        basis, system, lat, _stand_in_density(basis, system, lat),
        0.30, mapping, reps, compute_coulomb=compute_coulomb,
        internal_extent_bohr=internal_extent)
    farmed = build_jk_reduced_symmetrized(
        basis,
        system,
        lat,
        _stand_in_density(basis, system, lat),
        0.30,
        mapping,
        reps,
        output_cell_farming_task_kind="chi-direct-output-cell",
        compute_coulomb=compute_coulomb,
        internal_extent_bohr=internal_extent,
    )
    for c in range(len(j_ref)):
        np.testing.assert_allclose(
            np.asarray(red.J.blocks[c]), j_ref[c] if compute_coulomb else 0.0,
            atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(
            np.asarray(red.K.blocks[c]), k_ref[c], atol=1e-12, rtol=0.0)
        np.testing.assert_array_equal(
            np.asarray(farmed.J.blocks[c]), np.asarray(red.J.blocks[c])
        )
        np.testing.assert_array_equal(
            np.asarray(farmed.K.blocks[c]), np.asarray(red.K.blocks[c])
        )
    executed = farmed.output_cell_farming_execution
    assert executed.task_kind == "chi-direct-output-cell"
    assert executed.global_task_count == len(reps)
    assert executed.complete_internal_translation_sum is True


def _stand_in_density(basis, system, lat):
    """A point-group-symmetric lattice density for operator tests (overlap)."""
    return compute_overlap_lattice(basis, system, lat)


@pytest.mark.parametrize("extent", [None, 6.0])
@pytest.mark.parametrize("farming", [None, "chi-direct-output-cell"])
def test_reduced_j_only_matches_combined_build(monkeypatch, extent, farming):
    import vibeqc.pbc_bipole_fock as fock
    from vibeqc._vibeqc_core import direct_lattice_cells, make_lattice_matrix_set
    from vibeqc.bipole_symmetry_fock import (
        build_jk_reduced_symmetrized, cell_orbit_mapping, representative_cell_indices,
    )

    system = _he_sc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 4.1
    opts.pair_complete_1e = False
    cells = list(direct_lattice_cells(system, opts.cutoff_bohr))
    mapping = cell_orbit_mapping(system, basis, cells)
    reps = representative_cell_indices(mapping)
    assert len(reps) < len(cells)
    density_cells = list(direct_lattice_cells(system, 12.0))
    density = make_lattice_matrix_set(
        1, density_cells, [np.ones((1, 1)) for _ in density_cells],
    )
    kwargs = dict(internal_extent_bohr=extent, output_cell_farming_task_kind=farming)
    combined = build_jk_reduced_symmetrized(
        basis, system, opts, density, 0.4, mapping, reps, **kwargs)
    assert any(np.any(block) for block in combined.K.blocks)
    original = fock._build_jk_domains_output_cells_mpi
    native_selections = []

    def selected_domains(*args, **kwargs):
        native_selections.append(kwargs.get("compute_exchange", True))
        return original(*args, **kwargs)

    monkeypatch.setattr(fock, "_build_jk_domains_output_cells_mpi", selected_domains)
    selected = build_jk_reduced_symmetrized(
        basis, system, opts, density, 0.4, mapping, reps,
        compute_exchange=False, **kwargs)
    assert selected.K is None
    assert native_selections == [False]
    for actual, expected in zip(selected.J.blocks, combined.J.blocks):
        np.testing.assert_array_equal(actual, expected)
    if farming:
        assert selected.output_cell_farming_execution == combined.output_cell_farming_execution


@pytest.mark.parametrize("uks", [False, True])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
@pytest.mark.parametrize("incremental", [False, True])
def test_reduced_pure_dft_skips_exchange_through_complete_scf(
    monkeypatch, uks, mesh, incremental
):
    import vibeqc.bipole_symmetry_fock as symmetry
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    system = vq.PeriodicSystem(
        3, np.eye(3) * 20.0,
        [vq.Atom(1, [0.0, 0.0, -0.7]), vq.Atom(1, [0.0, 0.0, 0.7])],
    )
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    original = symmetry.build_jk_reduced_symmetrized
    calls = []

    def selected(*args, **kwargs):
        calls.append(kwargs.get("compute_exchange", True))
        result = original(*args, **kwargs)
        assert result.K is None
        return result

    monkeypatch.setattr(symmetry, "build_jk_reduced_symmetrized", selected)
    driver = run_pbc_bipole_uks if uks else run_pbc_bipole_rks

    def run():
        opts = PeriodicKSOptions()
        opts.functional = "svwn"
        # Retain coverage of the historical pair-resolved truncation.
        result = driver(
            system, basis, monkhorst_pack(system, mesh), opts,
            use_fock_symmetry_reduce=True, use_incremental_fock=incremental,
            historical_domain=True,
            progress=False,
        )
        assert result.converged
        return result

    j_only = run()
    assert len(calls) > 1 and not any(calls)

    def combined(*args, **kwargs):
        kwargs["compute_exchange"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(symmetry, "build_jk_reduced_symmetrized", combined)
    reference = run()
    assert j_only.energy == reference.energy
    assert j_only.e_xc == reference.e_xc
    assert j_only.n_iter == reference.n_iter


def test_sym3b_reduce_matches_full_symmetry_uhf_rks_uks():
    """SYM3b wiring for the other three BIPOLE drivers: ``use_fock_symmetry_
    reduce=True`` gives the SAME SCF energy as ``use_fock_symmetry=True`` for
    UHF / RKS / UKS (the reduced build reconstructs the same symmetrised J/K,
    so the SCF trajectories are identical). H2 box, cubic symmetry (7 cells →
    4 orbit-representative)."""
    # The physical pair default (#316) refuses the legacy SYM3b projector;
    # this test pins the historical composition, so it replays it explicitly
    # (#503).
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])

    def _hf_opts():
        o = PeriodicRHFOptions()
        o.lattice_opts.cutoff_bohr = 8.0
        o.lattice_opts.nuclear_cutoff_bohr = 8.0
        o.max_iter = 4
        o.use_diis = True
        o.initial_guess = InitialGuess.SAD
        return o

    def _ks_opts():
        o = PeriodicKSOptions()
        o.functional = "svwn"
        o.lattice_opts.cutoff_bohr = 8.0
        o.lattice_opts.nuclear_cutoff_bohr = 8.0
        o.max_iter = 4
        o.use_diis = True
        o.initial_guess = InitialGuess.SAD
        return o

    for name, fn, mk in (
        ("UHF", run_pbc_bipole_uhf, _hf_opts),
        ("RKS", run_pbc_bipole_rks, _ks_opts),  # pure functional: J-only branch
        ("UKS", run_pbc_bipole_uks, _ks_opts),
    ):
        e_sym = float(fn(
            system, basis, kmesh, mk(), use_exchange_ewald_split=True,
            use_fock_symmetry=True, use_fock_symmetry_reduce=False,
            sr_image_precision=None, ewald_precision=1e-8,
            historical_domain=True, progress=False).energy)
        e_red = float(fn(
            system, basis, kmesh, mk(), use_exchange_ewald_split=True,
            use_fock_symmetry_reduce=True, sr_image_precision=None,
            ewald_precision=1e-8,
            historical_domain=True, progress=False).energy)
        assert e_red == pytest.approx(e_sym, abs=1e-7), (
            f"{name}: SYM3b reduce {e_red:.10f} != full-symmetry {e_sym:.10f}")


@pytest.mark.parametrize("method", ["UHF", "UKS"])
def test_unrestricted_auto_symmetry_does_not_project_atomspin_state(method):
    """Attached structural symmetry cannot certify a broken spin density."""
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 30.0,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [3.0, 0.0, 0.0])],
    )
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])

    def _run(reduce):
        opts = PeriodicRHFOptions() if method == "UHF" else PeriodicKSOptions()
        if method == "UKS":
            opts.functional = "svwn"
        opts.initial_guess = InitialGuess.SAD
        opts.atomic_spins = [1, -1]
        opts.max_iter = 1
        opts.use_diis = False
        driver = run_pbc_bipole_uhf if method == "UHF" else run_pbc_bipole_uks
        return driver(
            system,
            basis,
            kmesh,
            opts,
            use_fock_symmetry_reduce=reduce,
            progress=False,
        )

    auto = _run(None)
    disabled = _run(False)
    assert auto.physical_fock_symmetry_reduced_builds == 0
    assert not auto.pair_resolved_fock_domain
    assert auto.energy == pytest.approx(disabled.energy, abs=1.0e-12)


@pytest.mark.parametrize("method", ["UHF", "UKS"])
def test_unrestricted_explicit_symmetry_rejects_atomspin_state(method):
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 20.0,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [3.0, 0.0, 0.0])],
    )
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicRHFOptions() if method == "UHF" else PeriodicKSOptions()
    opts.initial_guess = InitialGuess.SAD
    opts.atomic_spins = [1, -1]
    if method == "UKS":
        opts.functional = "svwn"
    driver = run_pbc_bipole_uhf if method == "UHF" else run_pbc_bipole_uks

    with pytest.raises(NotImplementedError, match="atomic_spins"):
        driver(
            system,
            basis,
            kmesh,
            opts,
            use_fock_symmetry_reduce=True,
            progress=False,
        )


def test_sym3b_reduce_falls_back_without_symmetry():
    """``use_fock_symmetry_reduce`` on a cell with no usable symmetry must not
    crash — it falls back to the full build (no rep-cell reduction)."""
    # The physical pair default (#316) refuses the legacy SYM3b projector;
    # this test pins the historical composition, so it replays it explicitly
    # (#503).
    # 1-atom cubic He still has symmetry; use a deliberately asymmetric 2-atom
    # cell so the orbit machinery may decline. The run must still converge.
    a = 6.0
    lattice = np.eye(3) * a
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.1, 0.3, 0.0])])
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 8.0
    opts.lattice_opts.nuclear_cutoff_bohr = 8.0
    opts.max_iter = 30
    opts.use_diis = True
    opts.initial_guess = InitialGuess.SAD
    r = run_pbc_bipole_rhf(
        system, basis, kmesh, opts,
        use_exchange_ewald_split=True,
        use_fock_symmetry_reduce=True,
        ewald_precision=1e-8, historical_domain=True, progress=False)
    assert r.converged


def test_sym3b_shell_mask_skips_nonrep_subblocks():
    """The shell-pair mask is the finer reduction over the whole-cell subset:
    it restricts the build to the orbit-representative atom-pair sub-blocks, so
    the masked raw build zeros non-rep sub-blocks that the whole-cell subset
    fills — yet both reconstruct to the SAME symmetrised operator (the mask
    changes only WHICH sub-blocks are computed)."""
    from vibeqc._vibeqc_core import (
        build_jk_2e_real_space_output_subset,
        build_jk_2e_real_space_output_subset_masked,
    )
    from vibeqc.bipole_symmetry_fock import (
        cell_orbit_mapping,
        representative_cell_indices,
        representative_shell_masks,
        symmetrize_fock_blocks,
    )

    system, basis, lat = _mgo_sym(8.0)
    D = _stand_in_density(basis, system, lat)
    mapping = cell_orbit_mapping(system, basis, list(D.cells))
    assert mapping is not None
    reps = representative_cell_indices(mapping)
    masks = representative_shell_masks(mapping, reps)
    assert len(masks) == len(reps)

    n_shells = len(list(basis.shells()))
    # The mask is a strict subset of all (s1, s2) output pairs in the rep cells
    # — the whole point of the finer reduction.
    total_set = sum(int(m.sum()) for m in masks)
    assert 0 < total_set < len(reps) * n_shells * n_shells

    jk_whole = build_jk_2e_real_space_output_subset(
        basis, system, lat, D, reps, 0.30)
    jk_mask = build_jk_2e_real_space_output_subset_masked(
        basis, system, lat, D, reps, masks, 0.30)
    nz_whole = sum(int(np.count_nonzero(np.asarray(b)))
                   for b in jk_whole.J.blocks)
    nz_mask = sum(int(np.count_nonzero(np.asarray(b)))
                  for b in jk_mask.J.blocks)
    # The masked raw build genuinely skips non-rep sub-blocks the whole-cell
    # build fills.
    assert nz_mask < nz_whole

    # Both reconstruct to the same symmetrised J and K.
    for whole, mask in ((jk_whole.J, jk_mask.J), (jk_whole.K, jk_mask.K)):
        bw = [np.asarray(b, dtype=float) for b in whole.blocks]
        bm = [np.asarray(b, dtype=float) for b in mask.blocks]
        symmetrize_fock_blocks(bw, mapping)
        symmetrize_fock_blocks(bm, mapping)
        for a, b in zip(bw, bm):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("element,pair,interaction", [(2, 2.1, 3.1), (6, 1.1, 1.6)])
@pytest.mark.parametrize("selection", ["jk", "j", "k"])
@pytest.mark.parametrize("image_shift", [0, 1])
@pytest.mark.parametrize("angle", [0., .37])
def test_physical_reduction_matches_unprojected_operator(element, pair, interaction, selection, image_shift, angle):
    """Off-center images, angular rotations and the wider exchange support."""
    from vibeqc import _vibeqc_core as core
    from vibeqc.physical_bipole_symmetry import physical_fock_mapping
    from vibeqc.bipole_symmetry_fock import build_jk_reduced_symmetrized, representative_cell_indices

    system = vq.PeriodicSystem(
        3, np.eye(3) * 4., [vq.Atom(element, [0., 0., 0.]), vq.Atom(element, [2. + 4. * image_shift, 0., 0.])],
    )
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    system = vq.PeriodicSystem(
        3, rotation @ np.asarray(system.lattice),
        [vq.Atom(atom.Z, rotation @ np.asarray(atom.xyz)) for atom in system.unit_cell],
    )
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = LatticeSumOptions()
    opts.pair_complete_1e = True
    opts.cutoff_bohr = pair
    opts.eri_interaction_cutoff_bohr = interaction
    opts.schwarz_threshold = 0.
    mapping = physical_fock_mapping(system, basis, opts)
    assert mapping is not None and not mapping.pair_resolved
    reps = representative_cell_indices(mapping)
    assert len(reps) < len(mapping.cells)
    density = core.make_lattice_matrix_set(
        basis.nbasis, mapping.cells, [np.eye(basis.nbasis) for _ in mapping.cells],
    )
    assert mapping.physical_plan.density_is_covariant(density)
    full = core.build_jk_2e_real_space(basis, system, opts, density, .4)
    reduced = build_jk_reduced_symmetrized(
        basis, system, opts, density, .4, mapping, reps,
        compute_coulomb=selection != "k", compute_exchange=selection != "j",
        output_cell_farming_task_kind="chi-direct-output-cell",
    )
    assert reduced.output_cell_farming_execution.global_task_count == len(reps)
    np.testing.assert_allclose(reduced.J.blocks, full.J.blocks if selection != "k" else 0., atol=2e-13, rtol=0.)
    if selection != "j":
        np.testing.assert_allclose(reduced.K.blocks, full.K.blocks, atol=2e-13, rtol=0.)
    else:
        assert reduced.K is None
    if element == 2 and pair == 2.1:
        centers = np.array([atom.xyz for atom in system.unit_cell])
        exterior = [np.linalg.norm(centers[None] + np.asarray(c.r_cart) - centers[:, None], axis=2) > pair
                    for c in full.K.cells]
        assert np.max(np.abs(np.asarray(full.K.blocks)[exterior])) > 1e-6


def test_physical_reduction_checks_density_stabilizers_and_falls_back():
    from vibeqc import _vibeqc_core as core
    from vibeqc.physical_bipole_symmetry import physical_fock_mapping
    from vibeqc.bipole_symmetry_fock import build_jk_reduced_symmetrized, representative_cell_indices

    system = vq.PeriodicSystem(3, np.eye(3) * 4., [vq.Atom(6, [0., 0., 0.])])
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = LatticeSumOptions()
    opts.pair_complete_1e = True
    opts.cutoff_bohr = 2.1
    opts.eri_interaction_cutoff_bohr = 2.1
    opts.schwarz_threshold = 1e-12
    mapping = physical_fock_mapping(system, basis, opts)
    # The home atom-pair orbit has one member. Unequal p occupations break
    # its stabilizer, invisible to a representative/member-only comparison.
    blocks = [np.eye(basis.nbasis) for _ in mapping.cells]
    home = next(i for i, cell in enumerate(mapping.cells) if not np.any(cell.index))
    blocks[home][-1, -1] = .7
    density = core.make_lattice_matrix_set(basis.nbasis, mapping.cells, blocks)
    before = np.asarray(density.blocks).copy()
    assert not mapping.physical_plan.density_is_covariant(density)
    full = core.build_jk_2e_real_space(basis, system, opts, density, .4)
    actual = build_jk_reduced_symmetrized(
        basis, system, opts, density, .4, mapping, representative_cell_indices(mapping),
        output_cell_farming_task_kind="chi-direct-output-cell",
    )
    assert actual.output_cell_farming_execution.global_task_count == len(mapping.cells)
    np.testing.assert_array_equal(actual.J.blocks, full.J.blocks)
    np.testing.assert_array_equal(actual.K.blocks, full.K.blocks)
    np.testing.assert_array_equal(density.blocks, before)
    opts.cutoff_bohr += .1
    with pytest.raises(ValueError, match="does not match"):
        build_jk_reduced_symmetrized(
            basis, system, opts, density, .4, mapping, representative_cell_indices(mapping),
        )


@pytest.mark.parametrize("selection", ["jk", "j", "k"])
@pytest.mark.parametrize("threshold", [0., 1e-12, 1e-20])
@pytest.mark.parametrize("image_shift", [None, 0, 1])
def test_physical_symmetry_retains_small_asymmetric_residual(monkeypatch, selection, threshold, image_shift):
    """A small broken p occupation must survive the reduced-plus-full split."""
    from vibeqc import _vibeqc_core as core
    from vibeqc.physical_bipole_symmetry import physical_fock_mapping
    from vibeqc.bipole_symmetry_fock import build_jk_reduced_symmetrized

    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    atoms = [vq.Atom(6, [0., 0., 0.])]
    if image_shift is not None:
        atoms.append(vq.Atom(6, rotation @ np.array([2. + 4. * image_shift, 0., 0.])))
    system = vq.PeriodicSystem(3, 4. * rotation, atoms)
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = LatticeSumOptions()
    opts.pair_complete_1e = True
    opts.cutoff_bohr = opts.eri_interaction_cutoff_bohr = 2.1
    opts.schwarz_threshold = threshold
    mapping = physical_fock_mapping(system, basis, opts)
    plan = mapping.physical_plan
    blocks = [np.eye(basis.nbasis) for _ in mapping.cells]
    home = next(i for i, c in enumerate(mapping.cells) if not np.any(c.index))
    blocks[home][-1, -1] += 1e-8
    density = core.make_lattice_matrix_set(basis.nbasis, mapping.cells, blocks)
    symmetric, residual, _ = plan._split_nearly_covariant_density(mapping, density)
    assert not plan.density_is_covariant(density)
    assert plan.density_is_covariant(symmetric)
    np.testing.assert_allclose(np.asarray(symmetric.blocks) + residual.blocks, blocks,
                               atol=2e-16, rtol=0.)
    full = core.build_jk_2e_real_space(basis, system, opts, density, .4)
    projected = core.build_jk_2e_real_space(basis, system, opts, symmetric, .4)
    # Pin that discarding the remainder would fail this test.
    for field in ("J", "K"):
        assert np.max(np.abs(np.asarray(getattr(full, field).blocks)
                             - getattr(projected, field).blocks)) > 1e-10
    from vibeqc import pbc_bipole_fock
    original = pbc_bipole_fock._build_jk_domains_output_cells_mpi
    thresholds = []

    def checked(*args, **kwargs):
        thresholds.append(args[2].schwarz_threshold)
        return original(*args, **kwargs)

    monkeypatch.setattr(pbc_bipole_fock, "_build_jk_domains_output_cells_mpi", checked)
    actual = build_jk_reduced_symmetrized(
        basis, system, opts, density, .4, mapping, plan.representatives,
        compute_coulomb=selection != "k", compute_exchange=selection != "j",
        output_cell_farming_task_kind="chi-direct-output-cell",
    )
    assert (plan.reduced_builds, plan.full_builds, plan.residual_builds) == (
        (1, 0, 1) if threshold > 0. else (0, 1, 0)
    )
    assert actual.output_cell_farming_execution.global_task_count == len(mapping.cells)
    if threshold > 0.:
        assert len(thresholds) == 2 and 0. < thresholds[1] < thresholds[0]
        assert sum(thresholds) == pytest.approx(threshold, rel=2e-16, abs=0.)
    else:
        assert thresholds == [0.]
    np.testing.assert_allclose(actual.J.blocks, full.J.blocks if selection != "k" else 0.,
                               atol=2e-13, rtol=0.)
    if selection != "j":
        np.testing.assert_allclose(actual.K.blocks, full.K.blocks, atol=2e-13, rtol=0.)
    else:
        assert actual.K is None
    np.testing.assert_array_equal(density.blocks, blocks)
    assert opts.schwarz_threshold == threshold


@pytest.mark.parametrize("method", ["rhf", "uhf", "svwn", "pbe0", "hse06", "uks-svwn", "uks-pbe0", "uks-hse06"])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
@pytest.mark.parametrize("reduction", [True, None])
def test_physical_symmetry_reduction_complete_scf(monkeypatch, method, mesh, reduction):
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks
    from vibeqc.physical_bipole_symmetry import _PhysicalPlan

    system = vq.PeriodicSystem(
        3, np.eye(3) * 12., [vq.Atom(1, [0., 0., -.7]), vq.Atom(1, [0., 0., .7])],
    )
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    driver = {"rhf": run_pbc_bipole_rhf, "uhf": run_pbc_bipole_uhf}.get(method, run_pbc_bipole_rks)
    if method.startswith("uks-"):
        driver = run_pbc_bipole_uks
    opts = PeriodicRHFOptions() if method in ("rhf", "uhf") else PeriodicKSOptions()
    if method not in ("rhf", "uhf"):
        opts.functional = method.removeprefix("uks-")
    opts.lattice_opts.cutoff_bohr = 8.
    opts.lattice_opts.nuclear_cutoff_bohr = 8.
    opts.conv_tol_energy = 1e-9
    calls = []
    original = _PhysicalPlan.density_is_covariant

    def checked(self, density):
        result = original(self, density)
        calls.append(result)
        return result

    monkeypatch.setattr(_PhysicalPlan, "density_is_covariant", checked)
    kwargs = dict(progress=False, use_incremental_fock=True)
    reference = driver(system, basis, monkhorst_pack(system, mesh), opts,
                       use_fock_symmetry_reduce=False, **kwargs)
    actual = driver(system, basis, monkhorst_pack(system, mesh), opts,
                    use_fock_symmetry_reduce=reduction, **kwargs)
    assert reference.converged and actual.converged
    assert calls and any(calls)
    residuals = actual.physical_fock_symmetry_residual_builds
    assert actual.physical_fock_symmetry_reduced_builds == sum(calls) + residuals
    assert actual.physical_fock_symmetry_full_builds == len(calls) - sum(calls) - residuals
    assert not actual.pair_resolved_fock_domain
    assert actual.energy == pytest.approx(reference.energy, abs=2e-10)
    assert actual.n_iter == reference.n_iter
    for field in (("fock_alpha", "fock_beta") if method == "uhf" or method.startswith("uks-")
                  else ("fock",)):
        np.testing.assert_allclose(getattr(actual, field), getattr(reference, field), atol=2e-10, rtol=0.)


@pytest.mark.parametrize("pair,interaction", [(4., 3.1), (3.1, 4.)])
def test_physical_symmetry_refuses_ambiguous_cutoff_boundaries(pair, interaction):
    from vibeqc.physical_bipole_symmetry import physical_fock_mapping

    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    system = vq.PeriodicSystem(3, 4. * rotation, [vq.Atom(2, [0., 0., 0.])])
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = LatticeSumOptions()
    opts.pair_complete_1e = True
    opts.cutoff_bohr = pair
    opts.eri_interaction_cutoff_bohr = interaction
    # A hard distance tie is not a floating-point-invariant quartet domain.
    # In the unguarded R=L=4 witness, rotated reconstruction moved K by
    # 9.3e-5 Ha. The native operator must remain unchanged: decline reduction.
    assert physical_fock_mapping(system, basis, opts) is None
