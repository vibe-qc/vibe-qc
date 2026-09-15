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


@pytest.mark.xfail(
    reason="MgO/STO-3G needs cutoff >= 16 bohr for S(Γ) fold convergence")
def test_symmetry_integrals_mgo_energy_invariance_when_fock_reduce_disabled():
    """Attached symmetry alone leaves the energy invariant when M5 Fock
    reduction and padding are explicitly disabled.

    Two regressions in one:

    * The 2026-06-09 Fock enforcement transformed whole-matrix blocks
      with one uniform cell map g → R·g — only correct for
      single-atom-at-origin cells; on MgO primitive it scattered Mg–O
      cross blocks into wrong cells (~0.5 Ha shift) AND was auto-on
      whenever symmetry was attached.
    * Even with the correct atom-pair-resolved group action, enforcing
      orbit symmetry on the historical truncated radial J/K domain moved
      energies. M5 fixes that domain and makes reduction the attached-
      symmetry default; this test retains the narrower SYM2c S/T identity
      contract by explicitly opting out of M5 and selecting the legacy-gauge
      diagnostic associated with that historical domain.

    NOTE: xfail — MgO/STO-3G requires cutoff >= 16 bohr for S(Γ) fold
    convergence (< 1e-4 drift), which is too slow for the standard test
    suite.  The fold gate correctly refuses smaller cutoffs.  Use the
    dedicated MgO parity scripts in examples/regression/ for full
    fold-converged validation.
    """
    ANG2BOHR = 1.0 / 0.529177210903
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )

    def _mgo(attach: bool):
        system = vq.PeriodicSystem(
            3,
            lattice,
            [
                vq.Atom(12, [0.0, 0.0, 0.0]),
                vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0]),
            ],
        )
        if attach:
            attach_symmetry(system)
        return system

    def _opts():
        o = PeriodicRHFOptions()
        o.lattice_opts.cutoff_bohr = 10.0
        o.lattice_opts.nuclear_cutoff_bohr = 10.0
        o.max_iter = 50
        o.use_diis = True
        o.conv_tol_energy = 1e-8
        o.initial_guess = InitialGuess.SAD
        return o

    energies = {}
    for attach in (False, True):
        system = _mgo(attach)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        kmesh = monkhorst_pack(system, [1, 1, 1])
        result = run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            _opts(),
            use_ewald_j_split=True,
            use_exchange_ewald_split=False,
            use_fock_symmetry_reduce=False,
            sr_image_precision=None,
            ewald_precision=1e-8,
            progress=False,
        )
        assert result.converged, f"attach={attach} did not converge"
        energies[attach] = result.energy

    # The S/T-reduced path reconstructs the explicit integrals to
    # machine precision, so the SCF trajectory is essentially identical.
    assert abs(energies[True] - energies[False]) < 1e-9, (
        f"symmetry changed the converged MgO energy: "
        f"{energies[False]:.10f} (off) vs {energies[True]:.10f} (on)"
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


def test_sym3b_reduced_build_equals_symmetrized_full_build():
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
    full = build_jk_2e_real_space(basis, system, lat, _stand_in_density(basis, system, lat), 0.30)
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
        0.30, mapping, reps)
    farmed = build_jk_reduced_symmetrized(
        basis,
        system,
        lat,
        _stand_in_density(basis, system, lat),
        0.30,
        mapping,
        reps,
        output_cell_farming_task_kind="chi-direct-output-cell",
    )
    for c in range(len(j_ref)):
        np.testing.assert_allclose(
            np.asarray(red.J.blocks[c]), j_ref[c], atol=1e-12, rtol=0.0)
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


def test_sym3b_reduce_matches_full_symmetry_uhf_rks_uks():
    """SYM3b wiring for the other three BIPOLE drivers: ``use_fock_symmetry_
    reduce=True`` gives the SAME SCF energy as ``use_fock_symmetry=True`` for
    UHF / RKS / UKS (the reduced build reconstructs the same symmetrised J/K,
    so the SCF trajectories are identical). H2 box, cubic symmetry (7 cells →
    4 orbit-representative)."""
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
            progress=False).energy)
        e_red = float(fn(
            system, basis, kmesh, mk(), use_exchange_ewald_split=True,
            use_fock_symmetry_reduce=True, sr_image_precision=None,
            ewald_precision=1e-8,
            progress=False).energy)
        assert e_red == pytest.approx(e_sym, abs=1e-7), (
            f"{name}: SYM3b reduce {e_red:.10f} != full-symmetry {e_sym:.10f}")


def test_unrestricted_auto_symmetry_does_not_project_atomspin_state():
    """Attached structural symmetry cannot certify a broken spin density."""
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 30.0,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [3.0, 0.0, 0.0])],
    )
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])

    def _run(reduce):
        opts = PeriodicRHFOptions()
        opts.initial_guess = InitialGuess.SAD
        opts.atomic_spins = [1, -1]
        opts.max_iter = 1
        opts.use_diis = False
        return run_pbc_bipole_uhf(
            system,
            basis,
            kmesh,
            opts,
            use_fock_symmetry_reduce=reduce,
            progress=False,
        )

    auto = _run(None)
    disabled = _run(False)
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
        ewald_precision=1e-8, progress=False)
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
