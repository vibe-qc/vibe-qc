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
    absolute deficit on MgO, handover Sec. 0a 2026-07-13)."""
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
        assert params["sr_image_precision"].default == pytest.approx(1e-6)
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
