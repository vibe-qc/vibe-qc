"""Phase K8 — AUTO k-point recommender (``KPoints.recommend``).

``recommend`` is the electronic-structure-aware sibling of the K5
density constructors: instead of taking a target spacing it *decides*
Δk from metal/insulator character, builds a symmetry-reduced mesh, and
bundles a recommended smearing + a human-readable rationale.

Pinned contracts exercised here:

1. **Public API + ready-to-use output** — ``KPoints.recommend`` returns
   a ``kind="monkhorst-pack"`` :class:`KPoints` that drops straight into
   the periodic SCF drivers, carrying ``.smearing`` / ``.rationale`` /
   ``.verification``.
2. **Character → Δk** — band-gap thresholds map to insulator / small-gap
   / metal spacings; ``is_metal`` is the secondary signal; with no hint
   AUTO falls back to the SAFE metallic default and warns.
3. **Smearing follows character** — clean insulators get none; small-gap
   and metals get the matching preset width.
4. **Grid type** — Γ-centred for hexagonal/trigonal lattices, MP shift
   otherwise.
5. **Edge cases** — large supercell collapses to Γ-only; slabs pin the
   vacuum axis to 1.
6. **verify=True** — convergence ladder confirms / refines / reports.
7. **predictor** — ML hook picks the conservative end; ``"ml"`` without a
   model raises.
8. **Config block** — Δk + gap cutoffs are named, tunable constants.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.kpoints import (
    DELTA_K_INSULATOR,
    DELTA_K_METAL,
    DELTA_K_SMALL_GAP,
    GAP_INSULATOR_EV,
    GAP_METAL_EPS_EV,
    KPointConvergence,
    _is_hexagonal_like,
    _spacegroup_number,
)
from vibeqc.smearing import VALID_FLAVORS, SmearingOptions

ANGSTROM_TO_BOHR = 1.8897261339213


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _si_diamond():
    """Cubic insulator — Si in the diamond structure (SG 227)."""
    a = 5.43 * ANGSTROM_TO_BOHR
    lat = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    p2 = 0.25 * (lat[:, 0] + lat[:, 1] + lat[:, 2])
    return vq.PeriodicSystem(
        3, lat, [vq.Atom(14, [0.0, 0.0, 0.0]), vq.Atom(14, p2.tolist())]
    )


def _w_bcc():
    """Simple metal — tungsten BCC (SG 229)."""
    a = 3.16 * ANGSTROM_TO_BOHR
    lat = np.array(
        [[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]]
    )
    return vq.PeriodicSystem(3, lat, [vq.Atom(74, [0.0, 0.0, 0.0])])


def _hexagonal():
    """Primitive hexagonal cell (γ = 120°) — exercises Γ-centring."""
    a = 2.5 * ANGSTROM_TO_BOHR
    c = 4.0 * ANGSTROM_TO_BOHR
    lat = np.array([[a, -a / 2, 0.0], [0.0, a * np.sqrt(3) / 2, 0.0], [0.0, 0.0, c]])
    return vq.PeriodicSystem(3, lat, [vq.Atom(4, [0.0, 0.0, 0.0])])


def _big_supercell():
    """A large cubic box — BZ tiny, should collapse to Γ-only."""
    L = 22.0 * ANGSTROM_TO_BOHR
    return vq.PeriodicSystem(3, np.diag([L, L, L]), [vq.Atom(18, [0, 0, 0])])


def _slab_2d():
    """2D slab with vacuum along z — vacuum axis must stay at N = 1."""
    a = 3.0 * ANGSTROM_TO_BOHR
    return vq.PeriodicSystem(2, np.diag([a, a, 100.0]), [vq.Atom(12, [0, 0, 0])])


def _h2_gdf_smoke_system():
    """Tiny dilute molecular-limit cell for runner forwarding smoke tests."""
    box = 18.0
    c = box / 2.0
    return vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )


# ---------------------------------------------------------------------------
# 1. Public API + ready-to-use output
# ---------------------------------------------------------------------------


def test_recommend_is_public():
    assert hasattr(vq.KPoints, "recommend")


def test_recommend_returns_monkhorst_pack_kpoints():
    kp = vq.KPoints.recommend(_si_diamond(), band_gap=1.1)
    assert isinstance(kp, vq.KPoints)
    assert kp.kind == "monkhorst-pack"
    # Ready to feed the SCF drivers via the boundary helper.
    assert vq.as_bloch_kmesh(kp) is not None


def test_recommend_populates_spec_fields():
    kp = vq.KPoints.recommend(_si_diamond(), band_gap=1.1)
    assert kp.mesh is not None and len(kp.mesh) == 3
    assert kp.shift is not None
    assert kp.rationale  # non-empty human-readable string
    assert kp.grid_type in ("gamma", "monkhorst-pack")
    assert kp.full_mesh_size >= kp.n_kpoints  # IR set ≤ full mesh


def test_kpointconvergence_exported():
    assert hasattr(vq, "KPointConvergence")
    assert vq.KPointConvergence is KPointConvergence


# ---------------------------------------------------------------------------
# 2. Cubic insulator — moderate mesh, no smearing, not hex-flagged
# ---------------------------------------------------------------------------


def test_cubic_insulator_moderate_mesh_no_smearing():
    kp = vq.KPoints.recommend(_si_diamond(), band_gap=1.1)
    assert kp.rationale.startswith("insulator")
    assert kp.smearing is None
    # "moderate" — neither collapsed to Γ nor absurdly dense.
    assert kp.mesh != (1, 1, 1)
    assert all(1 <= n <= 16 for n in kp.mesh)


def test_cubic_insulator_not_flagged_hexagonal():
    """Si diamond is cubic; its FCC primitive cell carries 60° vector
    angles, but the spacegroup (227) is authoritative — it must not be
    treated as hexagonal."""
    sys = _si_diamond()
    assert _is_hexagonal_like(sys, _spacegroup_number(sys)) is False
    assert "hexagonal" not in vq.KPoints.recommend(sys, band_gap=1.1).rationale


def test_insulator_delta_k_matches_from_kspacing():
    """recommend delegates the Δk → mesh step to the validated K5 path."""
    sys = _si_diamond()
    auto = vq.KPoints.recommend(sys, band_gap=1.1)
    direct = vq.KPoints.from_kspacing(sys, DELTA_K_INSULATOR, units="angstrom")
    assert auto.mesh == direct.mesh


# ---------------------------------------------------------------------------
# 3. Simple metal — denser mesh + smearing
# ---------------------------------------------------------------------------


def test_metal_denser_than_insulator_same_system():
    sys = _w_bcc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metal = vq.KPoints.recommend(sys, is_metal=True)
        insulator = vq.KPoints.recommend(sys, is_metal=False)
    assert np.prod(metal.mesh) > np.prod(insulator.mesh)


def test_metal_recommends_smearing():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = vq.KPoints.recommend(_w_bcc(), is_metal=True)
    assert isinstance(kp.smearing, SmearingOptions)
    assert kp.smearing.temperature > 0.0
    assert kp.smearing.flavor in VALID_FLAVORS
    assert "smearing" in kp.rationale


def test_metal_delta_k_matches_from_kspacing():
    sys = _w_bcc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto = vq.KPoints.recommend(sys, is_metal=True)
    direct = vq.KPoints.from_kspacing(sys, DELTA_K_METAL, units="angstrom")
    assert auto.mesh == direct.mesh


# ---------------------------------------------------------------------------
# 4. Band-gap thresholds → character → Δk
# ---------------------------------------------------------------------------


def test_band_gap_thresholds_select_character():
    sys = _si_diamond()
    wide = vq.KPoints.recommend(sys, band_gap=GAP_INSULATOR_EV + 0.5)
    small = vq.KPoints.recommend(sys, band_gap=GAP_INSULATOR_EV / 2.0)
    metal = vq.KPoints.recommend(sys, band_gap=0.0)
    assert wide.rationale.startswith("insulator")
    assert small.rationale.startswith("small-gap")
    assert metal.rationale.startswith("metal")
    # Denser as the gap closes.
    assert np.prod(wide.mesh) <= np.prod(small.mesh) <= np.prod(metal.mesh)


def test_zero_gap_is_metal_with_smearing():
    kp = vq.KPoints.recommend(_si_diamond(), band_gap=GAP_METAL_EPS_EV / 2.0)
    assert kp.rationale.startswith("metal")
    assert kp.smearing is not None and kp.smearing.temperature > 0.0


def test_small_gap_gets_gentle_smearing():
    kp = vq.KPoints.recommend(_si_diamond(), band_gap=0.3)
    assert kp.rationale.startswith("small-gap")
    assert kp.smearing is not None
    assert 0.0 < kp.smearing.temperature < 0.005  # gentler than metal


# ---------------------------------------------------------------------------
# 5. Hexagonal — Γ-centred
# ---------------------------------------------------------------------------


def test_hexagonal_is_detected():
    sys = _hexagonal()
    assert _is_hexagonal_like(sys, _spacegroup_number(sys)) is True


def test_hexagonal_uses_gamma_centred_grid():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = vq.KPoints.recommend(_hexagonal(), is_metal=True)
    assert kp.shift == (0, 0, 0)
    assert kp.grid_type == "gamma"


def test_non_hexagonal_even_mesh_uses_mp_shift():
    """Contrast case: a non-hex cell with an even mesh keeps the classical
    MP shift, proving the Γ-override is specific to hex lattices."""
    kp = vq.KPoints.recommend(_si_diamond(), band_gap=0.3)  # 12×12×12
    assert all(n % 2 == 0 for n in kp.mesh)
    assert kp.grid_type == "monkhorst-pack"
    assert kp.shift == (1, 1, 1)


# ---------------------------------------------------------------------------
# 6. Large supercell — Γ-only collapse
# ---------------------------------------------------------------------------


def test_large_supercell_collapses_to_gamma():
    kp = vq.KPoints.recommend(_big_supercell(), band_gap=5.0)
    assert kp.mesh == (1, 1, 1)
    assert "Γ-only" in kp.rationale


# ---------------------------------------------------------------------------
# 7. Slab / 2D — vacuum axis pinned to 1
# ---------------------------------------------------------------------------


def test_slab_vacuum_axis_pinned():
    kp = vq.KPoints.recommend(_slab_2d(), band_gap=2.0)
    assert kp.mesh[2] == 1
    assert kp.mesh[0] > 1 and kp.mesh[1] > 1


def test_periodicity_hint_mismatch_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vq.KPoints.recommend(_slab_2d(), band_gap=2.0, periodicity="3d")
    assert any("periodicity" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# 8. SAFE fallback — no hint ⇒ metal + warning
# ---------------------------------------------------------------------------


def test_no_hint_falls_back_to_metal_and_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kp = vq.KPoints.recommend(_si_diamond())
    assert kp.rationale.startswith("metal")
    assert kp.smearing is not None and kp.smearing.temperature > 0.0
    assert any(
        issubclass(w.category, UserWarning) and "metallic" in str(w.message).lower()
        for w in caught
    )


def test_classifier_hook_used_when_no_hint():
    sys = _si_diamond()
    kp_metal = vq.KPoints.recommend(sys, classifier=lambda s: "metal")
    kp_ins = vq.KPoints.recommend(sys, classifier=lambda s: ("insulator", None))
    assert kp_metal.rationale.startswith("metal")
    assert kp_ins.rationale.startswith("insulator")
    assert np.prod(kp_metal.mesh) > np.prod(kp_ins.mesh)


# ---------------------------------------------------------------------------
# 9. Δk override + gamma_centred override
# ---------------------------------------------------------------------------


def test_delta_k_override_bypasses_classification():
    sys = _si_diamond()
    kp = vq.KPoints.recommend(sys, band_gap=1.1, delta_k=0.15)
    assert kp.mesh == vq.KPoints.from_kspacing(sys, 0.15, units="angstrom").mesh
    assert "override" in kp.rationale


def test_gamma_centred_override_forces_zero_shift():
    kp = vq.KPoints.recommend(_si_diamond(), band_gap=0.3, gamma_centred=True)
    assert kp.shift == (0, 0, 0)


# ---------------------------------------------------------------------------
# 10. verify=True — convergence ladder
# ---------------------------------------------------------------------------


def test_verify_converges_immediately_for_flat_energy():
    """A flat (mesh-independent) energy is already converged at the base
    AUTO mesh — chosen mesh equals the base mesh, no refinement needed."""
    sys = _si_diamond()
    base = vq.KPoints.recommend(sys, band_gap=1.1).mesh
    kp = vq.KPoints.recommend(
        sys,
        band_gap=1.1,
        verify=True,
        scf_energy_fn=lambda kpts: -7.5,
    )
    assert isinstance(kp.verification, KPointConvergence)
    assert kp.verification.converged is True
    assert kp.verification.chosen_mesh == base
    assert kp.mesh == base


def test_verify_refines_then_converges():
    """A slowly-converging energy forces at least one refinement before
    the tolerance is met; the chosen mesh is no coarser than the base."""
    sys = _si_diamond()
    base = vq.KPoints.recommend(sys, band_gap=1.1).mesh

    def energy(kpts):
        return -7.5 + 0.05 / float(np.prod(kpts.mesh))

    kp = vq.KPoints.recommend(
        sys,
        band_gap=1.1,
        verify=True,
        scf_energy_fn=energy,
        tolerance_meV_per_atom=1.0,
    )
    assert kp.verification.converged is True
    assert np.prod(kp.verification.chosen_mesh) >= np.prod(base)
    assert len(kp.verification.rungs) >= 3


def test_verify_reports_non_convergence():
    """An energy that keeps drifting never meets tolerance — reported as
    not converged with the finest rung tried."""

    def energy(kpts):
        return -7.5 - 0.01 * max(kpts.mesh)

    kp = vq.KPoints.recommend(
        _si_diamond(),
        band_gap=1.1,
        verify=True,
        scf_energy_fn=energy,
        tolerance_meV_per_atom=1.0,
    )
    assert kp.verification.converged is False
    assert "NOT converged" in kp.rationale


def test_verify_requires_energy_fn():
    with pytest.raises(ValueError, match="scf_energy_fn"):
        vq.KPoints.recommend(_si_diamond(), band_gap=1.1, verify=True)


# ---------------------------------------------------------------------------
# 11. predictor — ML Δk hook
# ---------------------------------------------------------------------------


def test_predictor_callable_picks_conservative_end():
    sys = _si_diamond()
    # mean 0.25, σ 0.05 → conservative Δk = 0.20 (denser).
    kp = vq.KPoints.recommend(sys, band_gap=1.1, predictor=lambda f: (0.25, 0.05))
    assert kp.mesh == vq.KPoints.from_kspacing(sys, 0.20, units="angstrom").mesh
    assert "predictor" in kp.rationale


def test_predictor_accepts_dict_and_scalar():
    sys = _si_diamond()
    kp_dict = vq.KPoints.recommend(
        sys,
        band_gap=1.1,
        predictor=lambda f: {"delta_k": 0.25, "uncertainty": 0.05},
    )
    kp_scalar = vq.KPoints.recommend(sys, band_gap=1.1, predictor=lambda f: 0.20)
    target = vq.KPoints.from_kspacing(sys, 0.20, units="angstrom").mesh
    assert kp_dict.mesh == target
    assert kp_scalar.mesh == target


def test_predictor_features_are_passed():
    captured = {}

    def predictor(features):
        captured.update(features)
        return 0.2

    vq.KPoints.recommend(_si_diamond(), band_gap=1.1, predictor=predictor)
    assert captured["n_atoms"] == 2
    assert captured["dim"] == 3
    assert "reciprocal_lengths_bohr" in captured


def test_predictor_ml_without_model_raises():
    with pytest.raises(NotImplementedError, match="bundled ML k-spacing model is gated"):
        vq.KPoints.recommend(_si_diamond(), band_gap=1.1, predictor="ml")


def test_predictor_ml_with_user_model():
    sys = _si_diamond()
    kp = vq.KPoints.recommend(
        sys,
        band_gap=1.1,
        predictor="ml",
        ml_predictor=lambda f: (0.30, 0.10),
    )
    assert kp.mesh == vq.KPoints.from_kspacing(sys, 0.20, units="angstrom").mesh


def test_unknown_predictor_string_raises():
    with pytest.raises(ValueError, match="unknown predictor"):
        vq.KPoints.recommend(_si_diamond(), band_gap=1.1, predictor="magic")


# ---------------------------------------------------------------------------
# 12. Validation
# ---------------------------------------------------------------------------


def test_negative_band_gap_raises():
    with pytest.raises(ValueError, match="band_gap"):
        vq.KPoints.recommend(_si_diamond(), band_gap=-1.0)


def test_non_positive_tolerance_raises():
    with pytest.raises(ValueError, match="tolerance"):
        vq.KPoints.recommend(_si_diamond(), band_gap=1.1, tolerance_meV_per_atom=0.0)


def test_non_positive_delta_k_raises():
    with pytest.raises(ValueError, match="delta_k"):
        vq.KPoints.recommend(_si_diamond(), band_gap=1.1, delta_k=-0.1)


def test_unknown_periodicity_raises():
    with pytest.raises(ValueError, match="periodicity"):
        vq.KPoints.recommend(_si_diamond(), band_gap=1.1, periodicity="4d")


# ---------------------------------------------------------------------------
# 13. bz_integration="gilat" — parameter-free tetrahedron-family path
# ---------------------------------------------------------------------------


def test_gilat_no_smearing_on_efficient_ibz_mesh():
    """Gilat–Raubenheimer is parameter-free (no smearing) and — since the
    driver expands an IBZ mesh internally — AUTO keeps the efficient
    symmetry-reduced mesh rather than forcing the full BZ."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = vq.KPoints.recommend(_w_bcc(), is_metal=True, bz_integration="gilat")
    assert kp.bz_integration == "gilat"
    assert kp.smearing is None
    assert kp.is_symmetry_reduced is True  # IBZ, not the full mesh
    assert kp.n_kpoints < kp.full_mesh_size
    assert "Gilat" in kp.rationale


def test_gilat_respects_symmetry_setting():
    """gilat composes with the symmetry flag like any other backend:
    default reduces to the IBZ; symmetry=False yields the full mesh."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reduced = vq.KPoints.recommend(_w_bcc(), is_metal=True, bz_integration="gilat")
        full = vq.KPoints.recommend(
            _w_bcc(), is_metal=True, bz_integration="gilat", symmetry=False
        )
    assert reduced.is_symmetry_reduced is True
    assert full.is_symmetry_reduced is False
    assert full.n_kpoints == full.full_mesh_size
    assert reduced.n_kpoints < full.n_kpoints
    assert reduced.bz_integration == full.bz_integration == "gilat"
    assert reduced.smearing is None and full.smearing is None


def test_default_and_explicit_smearing_backends():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = vq.KPoints.recommend(_w_bcc(), is_metal=True)
        explicit = vq.KPoints.recommend(
            _w_bcc(), is_metal=True, bz_integration="smearing"
        )
    assert default.bz_integration is None
    assert default.smearing is not None and default.smearing.temperature > 0.0
    assert explicit.bz_integration == "smearing"
    assert explicit.smearing is not None


def test_invalid_bz_integration_raises():
    with pytest.raises(ValueError, match="bz_integration"):
        vq.KPoints.recommend(_w_bcc(), is_metal=True, bz_integration="bogus")


def test_gilat_full_mesh_is_consumable_by_the_low_level_backend():
    """With symmetry=False the spec is a full regular mesh — exactly one
    k-point per grid cell, accepted by the Gilat backend's grid mapping
    (a reduced mesh would raise here; the driver expands those instead)."""
    from vibeqc.bz_integration import fractional_kpoints, kpoint_grid_indices

    sys = _w_bcc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Coarse Δk keeps the full mesh small for the test.
        kp = vq.KPoints.recommend(
            sys, is_metal=True, bz_integration="gilat", symmetry=False, delta_k=1.0
        )
    assert kp.is_symmetry_reduced is False
    bm = vq.as_bloch_kmesh(kp)
    frac = fractional_kpoints(
        sys.reciprocal_lattice(), np.asarray(bm.kpoints, dtype=float)
    )
    idx = kpoint_grid_indices(frac, kp.mesh)
    n_full = int(np.prod(kp.mesh))
    assert len(idx) == n_full
    assert len({tuple(i) for i in idx}) == n_full  # all cells distinct


def test_gilat_ibz_mesh_is_consumable_by_the_driver_entry_point():
    """The whole point of keeping the IBZ mesh: AUTO's reduced gilat mesh
    must be accepted by the driver-facing ``gilat_occupations_for_kmesh``
    auto-dispatch (which expands the IBZ to the full BZ internally)."""
    from vibeqc.bz_integration import gilat_occupations_for_kmesh

    a = 3.5 * ANGSTROM_TO_BOHR
    lat = np.array(
        [[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]]
    )
    li = vq.PeriodicSystem(3, lat, [vq.Atom(3, [0, 0, 0])])  # bcc Li
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = vq.KPoints.recommend(
            li, is_metal=True, bz_integration="gilat", delta_k=1.0
        )
    assert spec.is_symmetry_reduced is True  # IBZ, not full mesh
    assert spec.n_kpoints < spec.full_mesh_size

    bm = vq.as_bloch_kmesh(spec)
    nk = len(bm.kpoints)
    eps = [
        np.array([-0.5 + 0.01 * i, -0.1 + 0.02 * i, 0.3 + 0.01 * i, 0.8])
        for i in range(nk)
    ]
    occ, e_fermi = gilat_occupations_for_kmesh(li, bm, eps, 3.0, 2.0)
    assert len(occ) == nk
    assert np.isfinite(e_fermi)
    weights = np.asarray(bm.weights, dtype=float)
    electrons = float(sum(weights[k] * float(np.sum(occ[k])) for k in range(nk)))
    assert abs(electrons - 3.0) < 0.5  # ~3 e⁻ on a coarse GR net


def test_gilat_with_verify_keeps_backend():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = vq.KPoints.recommend(
            _w_bcc(),
            is_metal=True,
            bz_integration="gilat",
            verify=True,
            scf_energy_fn=lambda kpts: -7.5,
        )
    assert kp.bz_integration == "gilat"
    assert kp.is_symmetry_reduced is True  # IBZ mesh retained through verify
    assert kp.smearing is None
    assert kp.verification.converged is True


# ---------------------------------------------------------------------------
# 14. Config block — named, tunable constants
# ---------------------------------------------------------------------------


def test_delta_k_constants_ordered_by_density():
    # Metals demand the smallest spacing (densest mesh), insulators the
    # largest. This ordering is the whole point of the heuristic.
    assert DELTA_K_METAL < DELTA_K_SMALL_GAP < DELTA_K_INSULATOR


def test_gap_cutoffs_sane():
    assert 0.0 < GAP_METAL_EPS_EV < GAP_INSULATOR_EV


# ---------------------------------------------------------------------------
# E — Auto-read smearing / bz_integration from KPoints metadata
# ---------------------------------------------------------------------------


def test_recommend_sets_smearing_for_metal():
    """recommend(is_metal=True) sets .smearing with metal preset."""
    spec = vq.KPoints.recommend(_w_bcc(), is_metal=True)
    assert spec.smearing is not None
    assert spec.smearing.temperature > 0.0
    assert spec.smearing.flavor == "methfessel-paxton"
    assert spec.smearing.source == "auto"


def test_recommend_sets_no_smearing_for_insulator():
    """recommend(band_gap=1.1) leaves .smearing as None."""
    spec = vq.KPoints.recommend(_si_diamond(), band_gap=1.1)
    assert spec.smearing is None


def test_recommend_sets_bz_integration_gilat():
    """recommend(bz_integration='gilat') sets .bz_integration and clears smearing."""
    spec = vq.KPoints.recommend(_w_bcc(), is_metal=True, bz_integration="gilat")
    assert spec.bz_integration == "gilat"
    assert spec.smearing is None
    assert "Gilat" in spec.rationale


def test_recommend_default_bz_integration_is_none():
    """recommend() without bz_integration leaves .bz_integration as None."""
    spec = vq.KPoints.recommend(_w_bcc(), is_metal=True)
    assert spec.bz_integration is None


def test_runner_dispatch_rhf_accepts_bz_integration():
    """run_rhf_periodic_scf accepts bz_integration kwarg on DIRECT path."""
    from vibeqc._vibeqc_core import (
        BasisSet,
        CoulombMethod,
        PeriodicSCFOptions,
    )
    from vibeqc.periodic_rhf_dispatch import run_rhf_periodic_scf

    sys = _h2_gdf_smoke_system()
    basis = BasisSet(sys.unit_cell_molecule(), "sto-3g")
    kmesh = vq.KPoints.monkhorst_pack(sys, (1, 1, 1))
    opts = PeriodicSCFOptions()
    opts.lattice_opts.coulomb_method = CoulombMethod.DIRECT_TRUNCATED
    opts.max_iter = 1
    result = run_rhf_periodic_scf(sys, basis, kmesh, opts, bz_integration=None)
    assert result is not None


def test_runner_dispatch_rhf_rejects_bad_bz_integration():
    """run_rhf_periodic_scf passes bad bz_integration through; Ewald rejects."""
    from vibeqc._vibeqc_core import (
        BasisSet,
        CoulombMethod,
        PeriodicRHFOptions,
    )
    from vibeqc.periodic_rhf_dispatch import run_rhf_periodic_scf

    sys = _si_diamond()
    basis = BasisSet(sys.unit_cell_molecule(), "sto-3g")
    kmesh = vq.KPoints.monkhorst_pack(sys, (2, 2, 2))
    opts = PeriodicRHFOptions()
    opts.lattice_opts.coulomb_method = CoulombMethod.EWALD_3D
    opts.max_iter = 1
    with pytest.raises(ValueError, match="bz_integration"):
        run_rhf_periodic_scf(sys, basis, kmesh, opts, bz_integration="bogus")


def test_runner_rejects_bad_bz_integration_early(tmp_path):
    """run_periodic_job validates bz_integration early (before any SCF)."""
    from vibeqc.periodic_runner import run_periodic_job

    sys = _si_diamond()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    # "Early" is before the SCF, not before the output writer: the manifest
    # is already open when the guard fires, so the stem must be a tmp_path
    # one or the refusal leaves ``output.system`` in the caller's tree (#508).
    with pytest.raises(ValueError, match="bz_integration"):
        run_periodic_job(
            sys,
            basis,
            method="RHF",
            bz_integration="bogus",
            max_iter=1,
            output=str(tmp_path / "bad_bz_integration"),
        )


# The three runner smoke tests below drive a real SCF to convergence.
# They used to pass ``max_iter=1`` as a speed hack; since the fail-closed
# non-convergence gate (3fc4b068f) a one-iteration SCF raises RuntimeError
# before any forwarding assertion can run (IID 190, same repair shape as
# IID 104 / 771cc3458). The dilute H2 box converges in 2 iterations at
# this tolerance, so the cap below is a safety net, not a budget.
_SMOKE_SCF_KWARGS = dict(max_iter=10, conv_tol_energy=1e-6)


def test_runner_gilat_gdf_closed_shell_smoke(tmp_path):
    """run_periodic_job forwards Gilat occupations to closed-shell GDF."""
    from vibeqc.periodic_runner import run_periodic_job

    sys = _h2_gdf_smoke_system()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "gilat_gdf"
    result = run_periodic_job(
        sys,
        basis,
        method="RHF",
        kpoints=(2, 1, 1),
        bz_integration="gilat",
        jk_method="gdf",
        output=stem,
        output_qvf=False,
        write_density=False,
        citations=False,
        progress=False,
        **_SMOKE_SCF_KWARGS,
    )
    assert result is not None
    assert result.converged
    # Gilat occupations: one occupation vector per resolved k-point, each
    # integrating to the two electrons of the closed-shell H2 cell.
    occupations = getattr(result, "occupations", None)
    assert occupations
    assert len(occupations) == 2
    for occ in occupations:
        assert np.sum(np.asarray(occ, dtype=float)) == pytest.approx(2.0)
    out_text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "bz_integration" in out_text
    assert "gilat" in out_text


def test_runner_explicit_bz_integration_none_allowed(tmp_path):
    """run_periodic_job(bz_integration=None) with GDF multi-k works."""
    from vibeqc.periodic_runner import run_periodic_job

    sys = _h2_gdf_smoke_system()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    result = run_periodic_job(
        sys,
        basis,
        method="RHF",
        kpoints=(2, 1, 1),
        bz_integration=None,
        jk_method="gdf",
        output=tmp_path / "test_bz_none",
        output_qvf=False,
        write_density=False,
        citations=False,
        progress=False,
        **_SMOKE_SCF_KWARGS,
    )
    assert result is not None
    assert result.converged
    assert np.isfinite(result.energy)


def test_recommend_kpoints_passthrough_to_run_periodic_job(tmp_path):
    """A BIPOLE-compatible recommended metal mesh flows through the runner."""
    from vibeqc.periodic_runner import run_periodic_job

    sys = _h2_gdf_smoke_system()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    # Force metal to get smearing metadata; use a coarse Δk to keep the
    # mesh small enough for this API-forwarding smoke test.
    spec = vq.KPoints.recommend(
        sys,
        is_metal=True,
        delta_k=1.0,
        smearing_method="fermi-dirac",
    )
    assert spec.smearing is not None
    stem = tmp_path / "test_auto_smear"
    result = run_periodic_job(
        sys,
        basis,
        method="RKS",
        functional="lda",
        kpoints=spec,
        jk_method="bipole",
        bipole_cutoff_bohr=18.0,
        output=stem,
        output_qvf=False,
        write_density=False,
        citations=False,
        progress=False,
        **_SMOKE_SCF_KWARGS,
    )
    assert result is not None
    assert result.converged
    assert np.isfinite(result.energy)
    # The recommender's smearing metadata must reach the runner's options
    # block, i.e. the KPoints spec was consumed, not just its mesh.
    out_text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "smearing_source" in out_text
    assert "auto" in out_text
    assert "fermi-dirac" in out_text


# ---------------------------------------------------------------------------
# B — Classify from a prior SCF result
# ---------------------------------------------------------------------------

def _mock_result(eps_per_k, fermi_level=0.0):
    """Minimal duck-typed periodic SCF result for testing."""
    class _Mock:
        pass
    r = _Mock()
    r.mo_energies = [np.asarray(eps, dtype=float) for eps in eps_per_k]
    r.fermi_level = float(fermi_level)
    return r


def _band_insulator(n_occ=2, n_virt=3):
    """Return eigenvalues for an insulator with clear gap."""
    occ = np.linspace(-2.0, -0.5, n_occ)
    virt = np.linspace(0.5, 2.0, n_virt)
    return np.concatenate([occ, virt])


def _band_metal(n_occ=2, n_virt=3):
    """Return eigenvalues for a metal (bands overlap -> gap <= 0)."""
    # The last occupied and first virtual are degenerate so after
    # sorting eps[n_occ-1] == eps[n_occ] -> gap == 0 -> metal.
    occ = np.zeros(n_occ + n_virt)
    occ[:n_occ] = -np.arange(n_occ, 0, -1, dtype=float)
    occ[n_occ] = occ[n_occ - 1]  # degenerate
    if n_virt > 1:
        occ[n_occ + 1:] = np.arange(1, n_virt, dtype=float)
    return occ


def _band_small_gap(n_occ=2, n_virt=3):
    """Return eigenvalues with a small gap (~0.01 Ha)."""
    occ = np.linspace(-2.0, -0.8, n_occ)
    virt = np.linspace(-0.79, 1.0, n_virt)
    return np.concatenate([occ, virt])


def test_scf_result_insulator_classifies_correctly():
    """A clear insulator result classifies as insulator."""
    sys = _si_diamond()
    result = _mock_result([_band_insulator(n_occ=14, n_virt=3)])
    spec = vq.KPoints.recommend(sys, scf_result=result)
    assert spec.rationale.startswith("insulator")
    assert spec.smearing is None


def test_scf_result_metal_classifies_correctly():
    """A metallic result classifies as metal."""
    sys = _si_diamond()
    result = _mock_result([_band_metal(n_occ=14, n_virt=3)])
    spec = vq.KPoints.recommend(sys, scf_result=result)
    assert spec.rationale.startswith("metal")
    assert spec.smearing is not None


def test_scf_result_small_gap_classifies_correctly():
    """A small-gap result classifies as small-gap semiconductor."""
    sys = _si_diamond()
    result = _mock_result([_band_small_gap(n_occ=14, n_virt=3)])
    spec = vq.KPoints.recommend(sys, scf_result=result)
    assert spec.rationale.startswith("small-gap semiconductor")
    assert spec.smearing is not None
    assert spec.smearing.temperature < 0.01


def test_scf_result_explicit_hints_win():
    """Explicit band_gap trumps scf_result."""
    sys = _si_diamond()
    result = _mock_result([_band_metal(n_occ=14, n_virt=3)])
    spec = vq.KPoints.recommend(sys, band_gap=2.0, scf_result=result)
    assert spec.rationale.startswith("insulator")
    assert spec.smearing is None


def test_scf_result_unusable_falls_through_to_safe_default():
    """A result with no eigenvalues falls through to SAFE metal."""
    sys = _si_diamond()
    class _BadResult:
        pass
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = vq.KPoints.recommend(sys, scf_result=_BadResult())
    assert spec.rationale.startswith("metal")
    assert spec.smearing is not None
    assert any("metallic" in str(w.message).lower() for w in caught)


def test_scf_result_multik_classifies_correctly():
    """Multi-k result: gap computed as indirect across k-points."""
    sys = _si_diamond()
    eps1 = _band_insulator(n_occ=14, n_virt=3)
    eps2 = _band_insulator(n_occ=14, n_virt=3)
    result = _mock_result([eps1, eps2])
    spec = vq.KPoints.recommend(sys, scf_result=result)
    assert spec.rationale.startswith("insulator")


# ---------------------------------------------------------------------------
# C — Gated cheap pre-SCF classifier (classifier="pre-scf")
# ---------------------------------------------------------------------------

def test_pre_scf_classifier_insulator():
    """classifier='pre-scf' classifies Si diamond as insulator."""
    sys = _si_diamond()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    spec = vq.KPoints.recommend(sys, classifier="pre-scf", basis=basis)
    assert spec.rationale.startswith("insulator")
    assert spec.smearing is None


def test_pre_scf_classifier_without_basis_warns():
    """classifier='pre-scf' without basis warns and falls back to SAFE."""
    sys = _si_diamond()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = vq.KPoints.recommend(sys, classifier="pre-scf")
    assert spec.rationale.startswith("metal")
    assert spec.smearing is not None
    assert any("basis" in str(w.message).lower() for w in caught)


def test_pre_scf_explicit_hints_win():
    """Explicit band_gap trumps pre-scf classifier."""
    sys = _si_diamond()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    spec = vq.KPoints.recommend(
        sys, band_gap=2.0, classifier="pre-scf", basis=basis,
    )
    assert spec.rationale.startswith("insulator")
    assert spec.smearing is None


def test_pre_scf_bad_classifier_string_raises():
    """An unknown classifier string raises TypeError."""
    sys = _si_diamond()
    with pytest.raises(TypeError, match="classifier"):
        vq.KPoints.recommend(sys, classifier="bogus")
