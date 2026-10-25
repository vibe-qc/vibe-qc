"""M1 contract for the BIPOLE shared-helper extraction.

``python/vibeqc/pbc_bipole_common.py`` holds the helpers that the four
BIPOLE drivers (RHF/RKS/UHF/UKS) share. This module pins the two things
the extraction must preserve:

* **Single source of truth** — every driver resolves a given shared
  helper to the *same* object in ``pbc_bipole_common`` (notably the
  previously-duplicated ``_density_set_gamma_or_lattice``).
* **Backward-compatible re-exports** — the helpers stay importable from
  their historical modules (``pbc_bipole`` / ``pbc_bipole_uhf``), which
  ``bipole_gradient`` and several tests still import from.

These are cheap identity checks (no SCF run); they guard the refactor's
import graph, not BIPOLE numerics — that stays covered by the driver
regression suites.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as _np
import pytest

import vibeqc as vq
import vibeqc.pbc_bipole as rhf
import vibeqc.pbc_bipole_common as common
import vibeqc.pbc_bipole_rks as rks
import vibeqc.pbc_bipole_uhf as uhf
import vibeqc.pbc_bipole_uks as uks
from vibeqc import Atom

# The full shared surface that lives in pbc_bipole_common.
_COMMON_ALL = {
    "PBCBipoleEnergyComponents",
    "_zero_cross_cell_density",
    "_bloch_sum_blocks",
    "_cell_key",
    "_lattice_contract",
    "_lattice_contract_blocks",
    "_crystal_ewald_options",
    "_expand_ibz_kmesh_for_ewald_j",
    "_default_bipole_v_ne_grid_options",
    "_compute_nuclear_lattice_ewald_reciprocal_ft",
    "_spin_occupations",
    "_combine_density_sets",
    "_copy_lattice_with_blocks",
    "_density_set_gamma_or_lattice",
}

# Helpers historically defined in pbc_bipole.py (RHF was the accidental
# common module); they must stay importable from there.
_RHF_REEXPORTS = {
    "PBCBipoleEnergyComponents",
    "_zero_cross_cell_density",
    "_bloch_sum_blocks",
    "_cell_key",
    "_lattice_contract",
    "_lattice_contract_blocks",
    "_crystal_ewald_options",
    "_expand_ibz_kmesh_for_ewald_j",
    "_default_bipole_v_ne_grid_options",
    "_compute_nuclear_lattice_ewald_reciprocal_ft",
}

# Spin helpers historically defined in pbc_bipole_uhf.py.
_UHF_REEXPORTS = {
    "_spin_occupations",
    "_combine_density_sets",
    "_copy_lattice_with_blocks",
}


def test_common_exposes_full_shared_surface():
    # All 14 M1 helpers are present and exported; every __all__ entry is real.
    assert _COMMON_ALL <= set(common.__all__)
    for name in common.__all__:
        assert hasattr(common, name), f"__all__ lists missing attr: {name}"
    for name in _COMMON_ALL:
        assert hasattr(common, name), f"missing from pbc_bipole_common: {name}"


def test_rhf_module_reexports_are_identity():
    # e.g. `from vibeqc.pbc_bipole import _crystal_ewald_options` (bipole_gradient, tests).
    for name in _RHF_REEXPORTS:
        assert getattr(rhf, name) is getattr(common, name), name


def test_uhf_module_reexports_are_identity():
    # e.g. `from vibeqc.pbc_bipole_uhf import _combine_density_sets` (bipole_gradient).
    for name in _UHF_REEXPORTS:
        assert getattr(uhf, name) is getattr(common, name), name


def test_all_drivers_share_one_helper_object():
    # Helpers used by every driver resolve to one object, not per-driver copies.
    for name in (
        "_crystal_ewald_options",
        "_lattice_contract",
        "_compute_nuclear_lattice_ewald_reciprocal_ft",
        "_zero_cross_cell_density",
    ):
        objs = {getattr(m, name) for m in (rhf, rks, uhf, uks)}
        assert objs == {getattr(common, name)}, (name, objs)


@pytest.mark.parametrize("requested", [True, _np.bool_(True)])
def test_quartet_far_field_guard_rejects_all_boolean_true_values(requested):
    with pytest.raises(NotImplementedError, match="three-translation"):
        common.reject_bipole_quartet_far_field(requested, driver="unit-test")


def test_quartet_far_field_guard_rejects_non_boolean_truthy_values():
    with pytest.raises(TypeError, match="must be bool or None"):
        common.reject_bipole_quartet_far_field(1, driver="unit-test")


def test_density_set_gamma_or_lattice_is_deduplicated():
    # Previously duplicated verbatim in pbc_bipole_rks and pbc_bipole_uks;
    # M1 consolidates them into a single pbc_bipole_common object.
    assert (
        rks._density_set_gamma_or_lattice
        is uks._density_set_gamma_or_lattice
        is common._density_set_gamma_or_lattice
    )


def test_spin_helpers_shared_between_uhf_and_uks():
    for name in _UHF_REEXPORTS:
        assert getattr(uhf, name) is getattr(uks, name) is getattr(common, name), name


def _single_atom_periodic_cell(Z: int, multiplicity: int = 1):
    return common.PeriodicSystem(
        3,
        _np.diag([12.0, 12.0, 12.0]),
        [Atom(Z, [0.0, 0.0, 0.0])],
        0,
        multiplicity,
    )


@pytest.mark.parametrize(
    ("Z", "expected"),
    [
        (13, (7, 6)),   # Al FCC benchmark cell
        (29, (15, 14)), # Cu FCC benchmark cell
    ],
)
def test_spin_occupations_default_odd_periodic_cell_is_minimum_open_shell(
    Z, expected
):
    # Periodic library inputs default to multiplicity=1. For odd-electron
    # metallic primitive cells, BIPOLE UHF/UKS must start the valid minimum
    # open-shell split instead of rejecting before SCF.
    assert common._spin_occupations(_single_atom_periodic_cell(Z)) == expected


def test_spin_occupations_rejects_nondefault_parity_mismatch():
    with pytest.raises(ValueError, match="cannot be split"):
        common._spin_occupations(_single_atom_periodic_cell(13, multiplicity=3))


# ---------------------------------------------------------------------------
# smearing_basin_warning — ionic-Γ basin-straddle diagnostic (Gap-B hardening)
# ---------------------------------------------------------------------------
from vibeqc.pbc_bipole_common import smearing_basin_warning  # noqa: E402

# eps with a small (~0.14 eV) gap above n_occ=3: HOMO=-0.40, LUMO=-0.395.
_EPS_SMALL_GAP = [_np.array([-0.50, -0.41, -0.40, -0.395, 0.20])]
_N_OCC = 3
_OCC_FRACTIONAL = [_np.array([1.0, 1.0, 0.7, 0.3, 0.0])]  # straddling
_OCC_INTEGER = [_np.array([1.0, 1.0, 1.0, 0.0, 0.0])]  # gapped, no straddle


def test_basin_warning_fires_when_smearing_straddles_small_gap():
    # T = 0.01 Ha (0.27 eV) >> ½·gap(0.14 eV); fractional occs present.
    msg = smearing_basin_warning(
        0.01, [(_EPS_SMALL_GAP, _OCC_FRACTIONAL, _N_OCC, 1.0)], 2.0, "drv"
    )
    assert msg is not None
    assert "near-metallic basin" in msg and "HOMO-LUMO gap" in msg


def test_basin_warning_silent_for_subgap_smearing():
    # T = 0.001 Ha (0.027 eV) < ½·gap; no straddle even if occs fractional.
    assert (
        smearing_basin_warning(
            0.001, [(_EPS_SMALL_GAP, _OCC_FRACTIONAL, _N_OCC, 1.0)], 0.1, "drv"
        )
        is None
    )


def test_basin_warning_silent_without_smearing():
    assert (
        smearing_basin_warning(
            0.0, [(_EPS_SMALL_GAP, _OCC_FRACTIONAL, _N_OCC, 1.0)], 0.0, "drv"
        )
        is None
    )


def test_basin_warning_silent_for_integer_occupations():
    # Gapped insulator converged to integer occupations: no fractional occ.
    assert (
        smearing_basin_warning(
            0.01, [(_EPS_SMALL_GAP, _OCC_INTEGER, _N_OCC, 1.0)], 0.0, "drv"
        )
        is None
    )


def test_basin_warning_silent_for_genuine_metal():
    # Vanishing gap (true metal): fractional occupation is physical → silent.
    eps_metal = [_np.array([-0.50, -0.41, -0.4001, -0.3999, 0.20])]
    occ = [_np.array([1.0, 1.0, 0.6, 0.4, 0.0])]
    assert (
        smearing_basin_warning(
            0.01, [(eps_metal, occ, _N_OCC, 1.0)], 3.0, "drv"
        )
        is None
    )


def test_basin_warning_uks_two_spin_channels_min_gap():
    # Alpha gapped, beta straddling: the min-gap-across-spins triggers.
    eps_a = [_np.array([-0.5, -0.4, 0.3, 0.4])]  # big gap above n_occ=2
    occ_a = [_np.array([1.0, 1.0, 0.0, 0.0])]
    eps_b = [_np.array([-0.5, -0.405, -0.40, 0.2])]  # ~0.14 eV gap above 2
    occ_b = [_np.array([1.0, 0.7, 0.3, 0.0])]
    msg = smearing_basin_warning(
        0.01,
        [(eps_a, occ_a, 2, 1.0), (eps_b, occ_b, 2, 1.0)],
        1.5,
        "run_pbc_bipole_uks",
    )
    assert msg is not None


# ---------------------------------------------------------------------------
# exact-FT J core reciprocal-tail warning (7cd0fd3a; magnitude measured
# by the Gap-B validation 2026-07-13: ~1.0 Ha on MgO at ke=200)
# ---------------------------------------------------------------------------
from vibeqc.pbc_bipole_common import warn_bipole_exact_j_core_tail  # noqa: E402


class _SilentPlog:
    def info(self, *_a, **_k):
        pass


def _basis_for(z_atoms):
    import numpy as np
    from vibeqc import BasisSet, PeriodicSystem

    lattice = np.eye(3) * 8.0
    system = PeriodicSystem(
        3, lattice, [Atom(z, [i * 1.4, 0.0, 0.0]) for i, z in enumerate(z_atoms)]
    )
    return BasisSet(system.unit_cell_molecule(), "sto-3g")


def test_core_tail_warns_on_dense_core_at_default_ke():
    """Mg 1s (gamma_max ~ 299 in STO-3G) needs ke ~ 8300 Ha for 1e-6
    coverage; the production default 200 must warn (measured ~1.0 Ha
    absolute deficit on MgO, historical design record Sec. 0a 2026-07-13)."""
    with pytest.warns(RuntimeWarning, match="VIBEQC_J_EWALD3D_KE"):
        ke_needed = warn_bipole_exact_j_core_tail(
            _basis_for([12, 8]), 200.0, _SilentPlog()
        )
    assert ke_needed is not None and ke_needed > 200.0


def test_core_tail_silent_for_light_cores(recwarn):
    """H2/STO-3G's tightest primitive is covered by the ke=200 default."""
    assert (
        warn_bipole_exact_j_core_tail(_basis_for([1, 1]), 200.0, _SilentPlog())
        is None
    )
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_core_tail_silent_once_converged(recwarn):
    assert (
        warn_bipole_exact_j_core_tail(
            _basis_for([12, 8]), 20000.0, _SilentPlog()
        )
        is None
    )
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# ---------------------------------------------------------------------------
# prepare_bipole_lattice_options — user screening thresholds must propagate
# ---------------------------------------------------------------------------


def _user_lattice_opts():
    opts = common.LatticeSumOptions()
    # cutoffs equal so the neutral-cell nuclear clamp is a no-op here.
    opts.cutoff_bohr = 9.0
    opts.nuclear_cutoff_bohr = 9.0
    opts.schwarz_threshold = 3.5e-9
    opts.schwarz_threshold_forces = 7.5e-13
    opts.screening_overlap_threshold = 1.5e-8
    opts.screening_exchange_threshold = 2.5e-8
    opts.slab_ewald_alpha = 0.31
    opts.becke_image_radius_bohr = 11.0
    # M4b QQR separation-aware SR screening flag: non-default so the
    # allowlist-driven propagation test below covers it non-vacuously
    # (run_periodic_job(sr_range_screening=True) sets exactly this).
    if hasattr(opts, "sr_range_screening"):
        opts.sr_range_screening = True
    opts.sr_sparse_traversal = False
    # Deliberately non-default: must NOT leak into the derived sets.
    opts.coulomb_method = common.CoulombMethod.EWALD_3D
    return opts


def test_prepare_lattice_options_propagates_user_screening_thresholds():
    # Regression (2026-07-13 truncation audit): the derived option sets
    # were fresh LatticeSumOptions() carrying only the two cutoffs, so a
    # user-set schwarz_threshold (or any other screening threshold on
    # options.lattice_opts) silently never reached the BIPOLE 2-e build
    # in any of the four drivers.
    system = _single_atom_periodic_cell(2)
    opts = _user_lattice_opts()
    _, _, lat_2e, lat_1e = common.prepare_bipole_lattice_options(
        system, opts, None, _SilentPlog()
    )
    for name in common._LATTICE_PASSTHROUGH_FIELDS:
        assert getattr(lat_2e, name) == getattr(opts, name), name
        assert getattr(lat_1e, name) == getattr(opts, name), name


def test_prepare_lattice_options_still_forces_coulomb_method():
    # The CRYSTAL-gauge channel split survives the passthrough: F^2e is
    # always DIRECT_TRUNCATED and the 1e channel is EWALD_3D on a 3-D
    # cell, whatever the user put on coulomb_method.
    system = _single_atom_periodic_cell(2)
    _, _, lat_2e, lat_1e = common.prepare_bipole_lattice_options(
        system, _user_lattice_opts(), None, _SilentPlog()
    )
    assert lat_2e.coulomb_method == common.CoulombMethod.DIRECT_TRUNCATED
    assert lat_1e.coulomb_method == common.CoulombMethod.EWALD_3D


def test_m5_driver_defaults_are_precision_pad_and_auto_reduce():
    for fn in (
        rhf.run_pbc_bipole_rhf,
        rks.run_pbc_bipole_rks,
        uhf.run_pbc_bipole_uhf,
        uks.run_pbc_bipole_uks,
    ):
        params = inspect.signature(fn).parameters
        assert params["sr_image_precision"].default == pytest.approx(1e-8)
        assert params["use_fock_symmetry_reduce"].default is None


def test_m5_symmetry_reduce_auto_enables_only_on_corrected_split():
    system = vq.PeriodicSystem(
        3,
        _np.eye(3) * 6.0,
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    vq.attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = common.LatticeSumOptions()
    opts.cutoff_bohr = 6.0
    opts.nuclear_cutoff_bohr = 6.0

    mapping, reps = common.resolve_bipole_fock_symmetry(
        system,
        basis,
        opts,
        None,
        None,
        _SilentPlog(),
        exchange_split_active=True,
    )
    assert mapping is not None and mapping.pair_resolved
    assert reps is not None and len(reps) > 0

    assert common.resolve_bipole_fock_symmetry(
        system,
        basis,
        opts,
        None,
        False,
        _SilentPlog(),
        exchange_split_active=True,
    ) == (None, None)
    assert common.resolve_bipole_fock_symmetry(
        system,
        basis,
        opts,
        None,
        None,
        _SilentPlog(),
        exchange_split_active=False,
    ) == (None, None)
    assert common.resolve_bipole_fock_symmetry(
        system,
        basis,
        opts,
        None,
        None,
        _SilentPlog(),
        exchange_split_active=True,
        auto_reduce_safe=False,
    ) == (None, None)


@pytest.mark.parametrize("attached,reduction,safe,cutoff,expected", [
    (True, None, True, 2.1, True),
    (True, False, True, 2.1, False),
    (True, None, False, 2.1, False),
    (False, None, True, 2.1, False),
    (True, None, True, 4., False),  # Physical distance exactly at the cutoff.
])
def test_physical_auto_symmetry_admission(attached, reduction, safe, cutoff, expected):
    system = vq.PeriodicSystem(3, _np.eye(3) * 4., [vq.Atom(2, [0., 0., 0.])])
    if attached:
        vq.attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = common.LatticeSumOptions()
    opts.pair_complete_1e = True
    opts.cutoff_bohr = cutoff
    opts.eri_interaction_cutoff_bohr = 3.1
    mapping, reps = common.resolve_bipole_fock_symmetry(
        system, basis, opts, None, reduction, _SilentPlog(), auto_reduce_safe=safe,
    )
    if expected:
        assert mapping.physical_plan is not None and not mapping.pair_resolved
        assert len(reps) < len(mapping.cells)
    else:
        assert (mapping, reps) == (None, None)


def test_terminal_trace_refresh_compares_against_evaluated_terminal_state():
    trace = []
    for iteration, energy, delta, grad in (
        (1, -1.0, 0.0, 2.0e-2),
        (2, -1.2, -0.2, 3.0e-3),
    ):
        trace.append(
            vq.SCFIteration(
                iter=iteration,
                energy=energy,
                delta_e=delta,
                grad_norm=grad,
            )
        )

    delta, grad = common.refresh_bipole_terminal_trace(
        trace,
        -1.25,
        grad_norm=4.0e-4,
    )

    assert delta == pytest.approx(-0.05)
    assert grad == pytest.approx(4.0e-4)
    assert trace[-1].energy == pytest.approx(-1.25)
    assert trace[-1].delta_e == pytest.approx(-0.05)
    assert trace[-1].grad_norm == pytest.approx(4.0e-4)


def test_one_representative_ibz_is_not_a_lone_twist():
    """A reduced two-point MP mesh may store one non-Gamma representative."""
    system = vq.PeriodicSystem(
        3,
        _np.eye(3) * 8.0,
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    vq.attach_symmetry(system)
    kmesh = vq.KPoints.monkhorst_pack(
        system,
        (2, 1, 1),
        symmetry=True,
    ).to_bloch_kmesh()

    metadata = common.validate_bipole_kmesh(
        kmesh,
        driver="test",
        require_complete=True,
    )
    common.reject_bipole_lone_non_gamma_kpoint(kmesh, driver="test")

    assert metadata == (1, 2, True, True)
    assert _np.linalg.norm(_np.asarray(kmesh.kpoints)[0]) > 0.0


def test_malformed_ibz_mapping_and_weights_fail_closed():
    bad_size = SimpleNamespace(
        kpoints=[[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]],
        weights=[0.5, 0.5],
        mesh=(2, 2, 2),
        ir_mapping=[0],
    )
    with pytest.raises(ValueError, match="ir_mapping has 1 entries"):
        common.validate_bipole_kmesh(bad_size, driver="test")

    bad_weights = SimpleNamespace(
        kpoints=[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
        weights=[0.5, 0.5],
        mesh=(3, 1, 1),
        ir_mapping=[0, 1, 1],
    )
    with pytest.raises(ValueError, match="IBZ weights"):
        common.validate_bipole_kmesh(bad_weights, driver="test")


def test_terminal_commutator_helpers_match_manual_weighted_norms():
    overlap = [
        _np.eye(2),
        _np.array([[1.0, 0.1], [0.1, 1.2]]),
    ]
    fock_a = [
        _np.diag([0.2, 0.8]),
        _np.array([[0.1, 0.3], [0.3, 0.9]]),
    ]
    fock_b = [0.7 * block for block in fock_a]
    density_a = [
        _np.array([[1.0, 0.2], [0.2, 0.1]]),
        _np.array([[0.8, -0.1], [-0.1, 0.2]]),
    ]
    density_b = [0.4 * block for block in density_a]
    weights = [0.25, 0.75]

    errors_a = []
    errors_b = []
    for f_a, f_b, d_a, d_b, s in zip(
        fock_a, fock_b, density_a, density_b, overlap
    ):
        fds_a = f_a @ d_a @ s
        fds_b = f_b @ d_b @ s
        errors_a.append(fds_a - fds_a.T)
        errors_b.append(fds_b - fds_b.T)

    restricted_expected = sum(
        weight * _np.linalg.norm(error)
        for weight, error in zip(weights, errors_a)
    )
    unrestricted_expected = sum(
        weight
        * _np.sqrt(_np.linalg.norm(error_a) ** 2 + _np.linalg.norm(error_b) ** 2)
        for weight, error_a, error_b in zip(weights, errors_a, errors_b)
    )

    assert common.restricted_bipole_commutator_norm(
        fock_a,
        density_a,
        overlap,
        weights,
    ) == pytest.approx(restricted_expected)
    assert common.unrestricted_bipole_commutator_norm(
        fock_a,
        fock_b,
        density_a,
        density_b,
        overlap,
        weights,
    ) == pytest.approx(unrestricted_expected)


@pytest.mark.parametrize("pair_complete", [False, True])
def test_fold_drift_keeps_one_electron_domain(pair_complete, monkeypatch):
    from vibeqc import _vibeqc_core as core
    options = vq.LatticeSumOptions()
    options.cutoff_bohr = 8.0
    options.pair_complete_1e = pair_complete
    seen = []
    def overlap(basis, system, lat):
        seen.append((lat.cutoff_bohr, lat.pair_complete_1e))
        return SimpleNamespace(
            cells=[SimpleNamespace(r_cart=_np.zeros(3))], blocks=[_np.eye(2)]
        )
    monkeypatch.setattr(core, "compute_overlap_lattice", overlap)
    assert common.s_fold_truncation_drift(None, None, options) == 0.0
    assert seen == [(8.0, pair_complete), (12.0, pair_complete)]


@pytest.mark.parametrize("molecular", [False, True])
def test_ecp_guard_recognizes_physical_charges_against_parent(molecular):
    system = vq.PeriodicSystem(3, _np.eye(3) * 20., [
        vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [0., 0., 3.]),
    ])
    if molecular:
        system = system.unit_cell_molecule()
    options = SimpleNamespace(ecp_effective_charges=[3., 1.])
    common.reject_bipole_ecp_options(options, driver="test", system=system)
    assert options.ecp_effective_charges == [3., 1.]


@pytest.mark.parametrize("fault", [
    "reduced", "reordered", "size", "roundoff", "nonfinite", "core",
    "fractional_core", "home_centers", "primitive_centers", "xml_library",
    "missing_parent", "unknown_parent", "primitive_blocks", "xml_centers",
])
def test_ecp_guard_does_not_mask_incomplete_operator_metadata(fault):
    system = vq.PeriodicSystem(3, _np.eye(3) * 20., [
        vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [0., 0., 3.]),
    ])
    options = SimpleNamespace(ecp_effective_charges={
        "reduced": [1., 1.], "reordered": [1., 3.], "size": [3.],
        "roundoff": [3.000000001, 1.], "nonfinite": [3., float("nan")],
    }.get(fault, [3., 1.]))
    if fault in ("core", "fractional_core"):
        options.ecp_total_ncore = 0.5 if fault == "fractional_core" else 2
    elif fault in ("home_centers", "primitive_centers"):
        setattr(options, f"ecp_{fault}", [[0., 0., 0.]])
    elif fault in ("primitive_blocks", "xml_centers"):
        setattr(options, "ecp_primitive_blocks" if fault == "primitive_blocks"
                else "ecp_centers", [object()])
    elif fault == "xml_library":
        options.ecp_library = "ecp10mdf"
    elif fault == "missing_parent":
        system = None
    elif fault == "unknown_parent":
        system = SimpleNamespace(unit_cell=system.unit_cell)
    with pytest.raises(NotImplementedError, match="ECP metadata"):
        common.reject_bipole_ecp_options(options, driver="test", system=system)


def test_ecp_guard_does_not_truncate_fractional_orphan_core_count():
    with pytest.raises(NotImplementedError, match="ECP metadata"):
        common.reject_bipole_ecp_options(
            SimpleNamespace(ecp_total_ncore=0.5), driver="test")


def test_ecp_guard_physical_charges_do_not_override_valence_only_basis():
    system = vq.PeriodicSystem(3, _np.eye(3) * 20., [vq.Atom(79, [0., 0., 0.])],
                               multiplicity=2)
    with pytest.raises(NotImplementedError, match="ECP metadata"):
        common.reject_bipole_ecp_options(
            SimpleNamespace(ecp_effective_charges=[79.]), driver="test",
            basis=SimpleNamespace(name="lanl2dz"), system=system)


def _lattice_set(cells_and_blocks):
    return SimpleNamespace(
        cells=[SimpleNamespace(index=list(c)) for c, _ in cells_and_blocks],
        blocks=[_np.asarray(b, dtype=float) for _, b in cells_and_blocks],
    )


def test_lattice_contract_skips_zero_padding_cells_missing_from_density():
    # The pair-complete one-electron list pads the |g| ball by the largest
    # intra-cell offset, so an atom outside the home cell adds operator cells
    # whose blocks are exactly zero; the density need not carry them (#502).
    d0 = _np.array([[1.0, 0.5], [0.5, 2.0]])
    m0 = _np.array([[3.0, -1.0], [-1.0, 0.25]])
    density = _lattice_set([((0, 0, 0), d0)])
    padded = _lattice_set([((0, 0, 0), m0), ((-1, -1, 0), _np.zeros((2, 2)))])
    assert common._lattice_contract(
        density, padded, operator_name="T"
    ) == pytest.approx(float(_np.sum(d0 * m0)), abs=0.0)


def test_lattice_contract_still_rejects_nonzero_block_missing_from_density():
    density = _lattice_set([((0, 0, 0), _np.eye(2))])
    operator = _lattice_set([
        ((0, 0, 0), _np.eye(2)),
        ((-1, -1, 0), _np.array([[0.0, 1e-14], [0.0, 0.0]])),
    ])
    with pytest.raises(ValueError, match=r"missing cell \(-1, -1, 0\)"):
        common._lattice_contract(density, operator, operator_name="T")


def test_native_lattice_cell_keys_preserve_integer_order_and_snapshots():
    from vibeqc import _vibeqc_core as core

    system = vq.PeriodicSystem(3, _np.eye(3) * 4., [Atom(2, [0., 0., 0.])])
    keys = [(7, -3, 0), (0, 0, 0), (7, -3, 0), (-(2**31), 2**31 - 1, -1)]
    cells = [core.LatticeCell(system, key) for key in keys]
    lattice = core.make_lattice_matrix_set(1, cells, [_np.ones((1, 1))] * len(cells))
    for source in (cells, tuple(cells), lattice):
        actual = core.lattice_cell_keys(source)
        assert actual == keys
        assert all(type(value) is int for key in actual for value in key)
        actual[0] = (99, 99, 99)
        assert core.lattice_cell_keys(source) == keys
    assert core.lattice_cell_keys([]) == []
    assert core.lattice_cell_keys(core.make_lattice_matrix_set(1, [], [])) == []


def test_native_selected_blocks_are_ordered_independent_live_copies():
    from vibeqc import _vibeqc_core as core

    system = vq.PeriodicSystem(3, _np.eye(3), [Atom(2, [0., 0., 0.])])
    lattice = core.make_lattice_matrix_set(
        2, [core.LatticeCell(system, (i, 0, 0)) for i in range(3)],
        [_np.eye(2) * (i + 1) for i in range(3)],
    )
    selected = lattice.get_blocks([2, 0, 2])
    _np.testing.assert_array_equal(selected, [_np.eye(2) * 3, _np.eye(2), _np.eye(2) * 3])
    selected[0][0, 0] = 99.
    assert selected[2][0, 0] == 3. and lattice.get_blocks([2])[0][0, 0] == 3.
    lattice.set_block(2, _np.eye(2) * 7.)
    assert selected[2][0, 0] == 3. and lattice.get_blocks([2])[0][0, 0] == 7.
    assert lattice.get_blocks([]) == []
    for indices in ([-1], [3], [0, -1], [2**63 - 1]):
        with pytest.raises(IndexError, match="out of range"):
            lattice.get_blocks(indices)
    _np.testing.assert_array_equal(lattice.get_blocks([0, 1, 2]),
                                   [_np.eye(2), _np.eye(2) * 2, _np.eye(2) * 7])


@pytest.mark.parametrize("source_kind", ["native", "python", "iterator"])
@pytest.mark.parametrize("complex_operator", [False, True])
def test_lattice_contract_batched_keys_preserve_support_and_arithmetic(
    source_kind, complex_operator,
):
    from vibeqc import _vibeqc_core as core

    system = vq.PeriodicSystem(3, _np.eye(3) * 4., [Atom(2, [0., 0., 0.])])
    rng = _np.random.default_rng(182)
    # Repeated density keys keep their last block; repeated operator keys
    # contribute each time, in supplied order. Extra density support is unused.
    d_keys = [(1, -2, 3), (0, 0, 0), (-4, 2, 1), (1, -2, 3)]
    m_keys = [(0, 0, 0), (1, -2, 3), (9, 9, 9), (0, 0, 0)]
    d_blocks = [rng.normal(size=(3, 3)) for _ in d_keys]
    m_blocks = [rng.normal(size=(3, 3)) for _ in m_keys]
    if complex_operator:
        m_blocks = [block + 1j * rng.normal(size=(3, 3)) for block in m_blocks]
    m_blocks[2] *= 0  # A zero padding block need not have a density partner.
    dc = [core.LatticeCell(system, key) for key in d_keys]
    mc = [core.LatticeCell(system, key) for key in m_keys]
    if source_kind == "native":
        density = core.make_lattice_matrix_set(3, dc, d_blocks)
    else:
        density = SimpleNamespace(
            cells=[SimpleNamespace(index=key) for key in d_keys], blocks=d_blocks)
        mc = [SimpleNamespace(index=key) for key in m_keys]
    expected = 0.0
    lookup = dict(zip(d_keys, d_blocks))
    for key, block in zip(m_keys, m_blocks):
        if key in lookup:
            expected += float(_np.real(_np.sum(lookup[key] * block)))
    assert common._lattice_contract_blocks(
        density, iter(mc) if source_kind == "iterator" else mc, m_blocks,
        operator_name="test",
    ) == expected
    if source_kind == "native" and not complex_operator:
        operator = core.make_lattice_matrix_set(3, mc, m_blocks)
        assert common._lattice_contract(density, operator, operator_name="test") == expected
        # No cached density values: set_block changes the next energy.
        density.set_block(1, 2 * d_blocks[1])
        expected = 0.0
        lookup[(0, 0, 0)] = 2 * d_blocks[1]
        for key, block in zip(m_keys, m_blocks):
            if key in lookup:
                expected += float(_np.real(_np.sum(lookup[key] * block)))
        assert common._lattice_contract(density, operator, operator_name="test") == expected


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (13, 13), (16, 16), (31, 31), (93, 93), (2, 7)])
@pytest.mark.parametrize("layout", ["C", "F", "mixed", "strided"])
def test_lattice_contract_batch_preserves_cancellation_and_cell_order(shape, layout):
    rng = _np.random.default_rng(982)
    keys = [(i, 0, 0) for i in range(257)]
    density_blocks, operator_blocks = [], []
    for i in range(len(keys)):
        # Span multiple batches, with cancellation both inside blocks and
        # between cells. Changing either reduction order can change the result.
        d = rng.normal(size=shape) * rng.choice([1.e-12, 1., 1.e12], size=shape)
        m = rng.normal(size=shape)
        if layout == "F" or (layout == "mixed" and i % 2):
            d = _np.asfortranarray(d)
        if layout == "F":
            m = _np.asfortranarray(m)
        if layout == "strided":
            d, m = d[::-1, ::-1], m[::-1, ::-1]
        density_blocks.append(d)
        operator_blocks.append(m)
    density = SimpleNamespace(
        cells=[SimpleNamespace(index=key) for key in keys], blocks=density_blocks)
    # A different operator order and repeated key must not be sorted or merged.
    order = list(range(256, -1, -1)) + [128, 0]
    expected = 0.
    for i in order:
        expected += float(_np.real(_np.sum(density_blocks[i] * operator_blocks[i])))
    actual = common._lattice_contract_keyed_blocks(
        density, [keys[i] for i in order], [operator_blocks[i] for i in order],
        operator_name="test",
    )
    assert actual.hex() == expected.hex()


def test_lattice_contract_batch_flushes_for_different_shapes_and_types():
    blocks = [_np.full((3, 3), 1.e16)] * 129
    blocks += [_np.ones((2, 7), dtype=_np.float32), _np.ones((2, 2), dtype=complex) * 1j]
    blocks += [_np.full((3, 3), -1.e16)] * 129
    blocks += [_np.empty((0, 0)), _np.ones((725, 725)), _np.eye(3)]
    keys = [(i, 0, 0) for i in range(len(blocks))]
    density = SimpleNamespace(
        cells=[SimpleNamespace(index=key) for key in keys],
        blocks=[_np.ones_like(b) for b in blocks],
    )
    expected = 0.
    for d, m in zip(density.blocks, blocks):
        expected += float(_np.real(_np.sum(d * m)))
    assert common._lattice_contract_keyed_blocks(
        density, keys, iter(blocks), operator_name="test",
    ).hex() == expected.hex()


def test_lattice_contract_batched_keys_retain_shape_and_support_errors():
    from vibeqc import _vibeqc_core as core

    system = vq.PeriodicSystem(3, _np.eye(3) * 4., [Atom(2, [0., 0., 0.])])
    home, other = [core.LatticeCell(system, key) for key in [(0, 0, 0), (-1, 0, 2)]]
    density = core.make_lattice_matrix_set(2, [home], [_np.eye(2)])
    with pytest.raises(ValueError, match=r"shape mismatch for T at cell \(0, 0, 0\)"):
        common._lattice_contract_blocks(density, [home], [_np.ones((1, 2))], operator_name="T")
    with pytest.raises(ValueError, match=r"missing cell \(-1, 0, 2\) required by T"):
        common._lattice_contract_blocks(density, [other], [_np.eye(2)], operator_name="T")
    assert common._lattice_contract_blocks(density, [], [], operator_name="T") == 0.


def test_bipole_rhf_energy_is_invariant_when_atoms_leave_the_home_cell():
    # #502: relax_atoms trial steps leave atoms outside the home cell. The
    # pair-complete T/V_ne lists then carry zero padding cells the density
    # domain lacks, which used to raise; the energy must equal the wrapped
    # and home-cell descriptions of the same crystal.
    from vibeqc._vibeqc_core import PeriodicRHFOptions

    def energy(z1, z2):
        system = vq.PeriodicSystem(
            3, _np.diag([8.0, 8.0, 8.0]),
            [Atom(1, [0.0, 0.0, z1]), Atom(1, [0.0, 0.0, z2])],
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        opts = PeriodicRHFOptions()
        opts.max_iter = 80
        opts.conv_tol_energy = 1e-10
        opts.lattice_opts.cutoff_bohr = 4.0
        opts.lattice_opts.nuclear_cutoff_bohr = 4.0
        result = rhf.run_pbc_bipole_rhf(
            system, basis, vq.monkhorst_pack(system, (1, 1, 1)), opts,
            progress=False, sr_image_precision=None,
        )
        assert result.converged
        return float(result.energy)

    unwrapped = energy(5.6569, -3.8569)
    assert unwrapped == pytest.approx(energy(5.6569, 4.1431), abs=1e-10)
    assert unwrapped == pytest.approx(energy(0.0, 1.5138), abs=1e-10)


class _CountedLatticeExports:
    """Model native copy-out accessors while keeping mutation independent."""

    def __init__(self, cells, blocks):
        self._cells = cells
        self._blocks = [block.copy() for block in blocks]
        self.cell_reads = self.block_reads = 0

    @property
    def cells(self):
        self.cell_reads += 1
        return list(self._cells)

    @property
    def blocks(self):
        self.block_reads += 1
        return [block.copy() for block in self._blocks]

    def set_block(self, index, block):
        self._blocks[index] = block.copy()


def test_overlap_fold_exports_once_for_all_kpoints(monkeypatch):
    from vibeqc import _vibeqc_core as core

    cells = [SimpleNamespace(r_cart=_np.array([x, 0., 0.])) for x in [-1., 0., 1.]]
    kpoints = [_np.zeros(3), _np.array([.4, 0., 0.])]
    records = []

    def overlap(basis, system, options):
        record = _CountedLatticeExports(cells, [_np.eye(2) * options.cutoff_bohr for _ in cells])
        records.append(record)
        return record

    monkeypatch.setattr(core, "compute_overlap_lattice", overlap)
    options = vq.LatticeSumOptions()
    options.cutoff_bohr = 4.
    assert common.s_fold_truncation_drift(None, None, options, k_points=kpoints) == pytest.approx(6.)
    assert [(r.cell_reads, r.block_reads) for r in records] == [(1, 1), (1, 1)]


def test_nuclear_ewald_assembly_exports_once_and_keeps_overlap(monkeypatch):
    cells = [SimpleNamespace(index=_np.array([x, 0, 0]), r_cart=_np.array([float(x), 0., 0.]))
             for x in [-1, 0, 1]]
    overlap_blocks = [_np.array([[1. + i, .1], [.1, 2.]]) for i in range(3)]
    short_blocks = [-2. * block for block in overlap_blocks]
    overlap = _CountedLatticeExports(cells, overlap_blocks)
    short = _CountedLatticeExports(cells, short_blocks)
    monkeypatch.setattr(common, "compute_nuclear_erfc_lattice", lambda *args: short)
    system = vq.PeriodicSystem(3, 4. * _np.eye(3), [vq.Atom(2, [0., 0., 0.])])
    options = vq.LatticeSumOptions()
    options.pair_complete_1e = False
    ewald = SimpleNamespace(alpha=.5)
    ft = _np.arange(24).reshape(3, 2, 2, 2).astype(complex) * (.01 + .02j)
    cache = SimpleNamespace(
        cells_r_cart=_np.array([c.r_cart for c in cells]), pair_cutoff_bohr=None,
        K_vectors=_np.array([[1., 0., 0.], [-1., 0., 0.]]),
        kernel=_np.array([.2, .2]), ft_per_cell=ft,
    )
    result, returned_cache = common._compute_nuclear_lattice_ewald_reciprocal_ft(
        None, system, options, ewald, overlap, cache=cache, resolved_erfc_options=options,
    )
    expected = [v - _np.real(_np.einsum("k,mnk->mn", 2. * cache.kernel, f.conj()))
                + _np.pi * 2. / (.5**2 * 64.) * s
                for v, s, f in zip(short_blocks, overlap_blocks, ft)]
    assert result is short and returned_cache is cache
    _np.testing.assert_allclose(result._blocks, expected, rtol=0, atol=1e-15)
    _np.testing.assert_array_equal(overlap._blocks, overlap_blocks)
    assert (short.cell_reads, short.block_reads) == (1, 1)
    assert (overlap.cell_reads, overlap.block_reads) == (1, 1)


@pytest.mark.parametrize("pair_complete", [False, True])
@pytest.mark.parametrize("alpha", [.6, 2.])
def test_nuclear_ewald_source_screen_matches_full_image_sum(
    monkeypatch, pair_complete, alpha,
):
    from vibeqc.periodic_v_ne import DEFAULT_NUCLEAR_SOURCE_TOL

    system = vq.PeriodicSystem(
        3, 6. * _np.eye(3),
        [vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [3., 0., 0.])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.LatticeSumOptions()
    options.cutoff_bohr = 6.
    options.nuclear_cutoff_bohr = 16.
    options.pair_complete_1e = pair_complete
    overlap = common.compute_overlap_lattice(basis, system, options)
    ncell, nbf = len(overlap.cells), basis.nbasis
    # Isolate short-range source selection while exercising the real assembly,
    # AO support mask and background. Native source-bound tests cover d shells.
    cache = SimpleNamespace(
        cells_r_cart=_np.array([c.r_cart for c in overlap.cells]),
        pair_cutoff_bohr=6. if pair_complete else None,
        K_vectors=_np.array([[1., 0., 0.]]), kernel=_np.ones(1),
        ft_per_cell=_np.zeros((ncell, nbf, nbf, 1), dtype=complex),
    )
    native = common.compute_nuclear_erfc_lattice
    seen = []

    def build(*args):
        seen.append(args[4:])
        return native(*args)

    monkeypatch.setattr(common, "compute_nuclear_erfc_lattice", build)
    screened, _ = common._compute_nuclear_lattice_ewald_reciprocal_ft(
        basis, system, options, SimpleNamespace(alpha=alpha), overlap,
        cache=cache, resolved_erfc_options=options,
    )
    assert seen == [(DEFAULT_NUCLEAR_SOURCE_TOL,) if pair_complete else ()]
    monkeypatch.setattr(
        common, "compute_nuclear_erfc_lattice", lambda *args: native(*args[:4]),
    )
    full, _ = common._compute_nuclear_lattice_ewald_reciprocal_ft(
        basis, system, options, SimpleNamespace(alpha=alpha), overlap,
        cache=cache, resolved_erfc_options=options,
    )
    assert [tuple(c.index) for c in screened.cells] == [tuple(c.index) for c in full.cells]
    reference = _np.asarray(full.blocks)
    rounding = 64. * _np.finfo(float).eps * float(_np.abs(reference).max())
    _np.testing.assert_allclose(
        screened.blocks, reference, rtol=0., atol=DEFAULT_NUCLEAR_SOURCE_TOL + rounding,
    )


@pytest.mark.parametrize("displacement", [1e-8, 2e-6])
def test_physical_auto_declines_approximate_attached_symmetry(displacement):
    system = vq.PeriodicSystem(
        3, _np.eye(3) * 6.,
        [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [3. + displacement, 0., 0.])],
    )
    vq.attach_symmetry(system, symprec=1e-4)
    assert len(system.symmetry.operations) > 2
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = common.LatticeSumOptions()
    opts.pair_complete_1e = True
    opts.cutoff_bohr = 2.1
    opts.eri_interaction_cutoff_bohr = 3.1
    assert common.resolve_bipole_fock_symmetry(
        system, basis, opts, None, None, _SilentPlog(),
    ) == (None, None)
