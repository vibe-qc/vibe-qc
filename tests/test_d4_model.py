"""Phase D4b-4 — the native D4 C6 model.

Part 1: :func:`vibeqc.dispersion_d4_model.cn_gaussian_weights`, the
coordination-number Gaussian reference weighting.

Part 2: :func:`vibeqc.dispersion_d4_model.zeta_charge_scaling`,
:func:`~vibeqc.dispersion_d4_model.zeta_hardness`,
:func:`~vibeqc.dispersion_d4_model.effective_nuclear_charge`, and
:func:`~vibeqc.dispersion_d4_model.d4_reference_weights` — the
charge-scaling factor ζ(q) and the combined CN+charge weights.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc.dispersion_d4_model import (
    D4_CHARGE_HEIGHT,
    D4_CHARGE_STEEPNESS,
    D4_WEIGHTING_FACTOR,
    R4R2_MAX_Z,
    ZETA_MAX_Z,
    cn_gaussian_weights,
    compute_c8,
    compute_c9,
    compute_d4_c6_pair,
    d4_atm_triple_energy,
    d4_bj_damping_energy,
    d4_reference_weights,
    effective_nuclear_charge,
    r4r2_val,
    zeta_charge_scaling,
    zeta_hardness,
)

# ---------------------------------------------------------------------
# Normalisation + basic shape.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cn", [0.0, 1.0, 2.3, 3.0, 4.0, 7.5])
def test_weights_sum_to_one(cn):
    w = cn_gaussian_weights(cn, [1.0, 2.0, 3.0, 4.0])
    assert w.shape == (4,)
    assert np.all(w >= 0.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_single_reference_gets_unit_weight():
    w = cn_gaussian_weights(2.7, [3.0])
    assert w.shape == (1,)
    assert w[0] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------
# The weight peaks at the nearest reference.
# ---------------------------------------------------------------------


def test_exact_match_dominates():
    """An atom whose CN equals a reference CN puts almost all weight
    on that reference."""
    refs = [1.0, 2.0, 3.0, 4.0]
    w = cn_gaussian_weights(3.0, refs)
    assert np.argmax(w) == 2
    assert w[2] > 0.99


def test_weight_peaks_at_nearest_reference():
    refs = [2.0, 3.0, 4.0]
    # CN 3.6 is closest to reference 4.0 (index 2).
    w = cn_gaussian_weights(3.6, refs)
    assert np.argmax(w) == 2


def test_midpoint_is_balanced():
    """Halfway between two references the two weights are equal and
    dominate the rest."""
    w = cn_gaussian_weights(2.5, [2.0, 3.0])
    assert w[0] == pytest.approx(w[1], abs=1e-12)
    assert w[0] == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------
# Monotonicity — moving toward a reference raises its weight.
# ---------------------------------------------------------------------


def test_weight_increases_toward_reference():
    refs = [1.0, 4.0]
    prev = -1.0
    for cn in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        w = cn_gaussian_weights(cn, refs)
        # weight on the CN-4 reference must rise monotonically as cn → 4
        assert w[1] >= prev
        prev = w[1]


# ---------------------------------------------------------------------
# Sharpness — a larger weighting factor concentrates the weight.
# ---------------------------------------------------------------------


def test_larger_weighting_factor_sharpens():
    refs = [2.0, 3.0]
    cn = 2.2  # nearer reference 2.0
    w_soft = cn_gaussian_weights(cn, refs, weighting_factor=1.0)
    w_sharp = cn_gaussian_weights(cn, refs, weighting_factor=20.0)
    # sharper → more weight on the nearer reference
    assert w_sharp[0] > w_soft[0]


def test_default_weighting_factor_is_d4_value():
    assert D4_WEIGHTING_FACTOR == 6.0
    # The default-argument path must match an explicit wf = 6.0.
    refs = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(
        cn_gaussian_weights(2.4, refs),
        cn_gaussian_weights(2.4, refs, weighting_factor=6.0),
    )


# ---------------------------------------------------------------------
# Robustness — far-from-everything underflow fallback.
# ---------------------------------------------------------------------


def test_far_from_all_references_falls_back_to_nearest():
    """An absurd CN where every Gaussian underflows still returns a
    normalised vector with unit weight on the nearest reference."""
    refs = [1.0, 2.0, 3.0]
    w = cn_gaussian_weights(500.0, refs, weighting_factor=50.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.argmax(w) == 2  # reference 3.0 is nearest
    assert w[2] == pytest.approx(1.0, abs=1e-12)


def test_n_gaussian_one_is_pure_single_gaussian():
    """n_gaussian=1 reproduces the textbook normalised single
    Gaussian exp(-wf ΔCN²) / Σ."""
    refs = np.array([1.0, 2.0, 3.0, 4.0])
    cn, wf = 2.7, 6.0
    expected = np.exp(-wf * (cn - refs) ** 2)
    expected /= expected.sum()
    np.testing.assert_allclose(
        cn_gaussian_weights(cn, refs, weighting_factor=wf, n_gaussian=1),
        expected,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------


def test_empty_references_raises():
    with pytest.raises(ValueError, match="non-empty"):
        cn_gaussian_weights(2.0, [])


def test_nonpositive_weighting_factor_raises():
    with pytest.raises(ValueError, match="weighting_factor"):
        cn_gaussian_weights(2.0, [1.0, 2.0], weighting_factor=0.0)


def test_bad_n_gaussian_raises():
    with pytest.raises(ValueError, match="n_gaussian"):
        cn_gaussian_weights(2.0, [1.0, 2.0], n_gaussian=0)


# ---------------------------------------------------------------------
# Integration with the D4b-3 reference catalogue.
# ---------------------------------------------------------------------


def test_weights_over_carbon_reference_cns():
    """Carbon's catalogue spans CN 1-4; an sp³ carbon (CN ≈ 4) must
    weight the CN-4 reference highest."""
    from vibeqc.dispersion_d4_reference_systems import reference_systems_for

    carbon_refs = [s.nominal_cn for s in reference_systems_for(6)]
    w = cn_gaussian_weights(3.95, carbon_refs)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    # the heaviest weight is on a CN-4 reference
    assert carbon_refs[int(np.argmax(w))] == 4.0


# =====================================================================
# D4b-4 part 2 — ζ(q) charge-scaling factor
# =====================================================================


# --- ζ(q) basic properties ---


def test_zeta_identity_at_reference():
    """ζ = 1 when qmod = qref (no charge deviation)."""
    assert zeta_charge_scaling(3.0, 0.4, 1.0, 1.0) == pytest.approx(1.0)
    assert zeta_charge_scaling(3.0, 0.4, 6.0, 6.0) == pytest.approx(1.0)
    assert zeta_charge_scaling(1.5, 0.2, 0.5, 0.5) == pytest.approx(1.0)


def test_zeta_suppression_when_charge_differs():
    """ζ < 1 when qmod ≠ qref (charge deviation suppresses the
    reference)."""
    z = zeta_charge_scaling(3.0, 0.4, 1.0, 2.0)
    assert 0.0 < z < 1.0


def test_zeta_anionic_floor():
    """ζ = exp(ga) when qmod ≤ 0 (deep anionic extreme)."""
    ga = 3.0
    z_negexp = zeta_charge_scaling(ga, 0.4, 1.0, -0.1)
    z_zero = zeta_charge_scaling(ga, 0.4, 1.0, 0.0)
    assert z_negexp == pytest.approx(np.exp(ga))
    assert z_zero == pytest.approx(np.exp(ga))


def test_zeta_cationic_trend():
    """As qmod → ∞ (strongly cationic), ζ → exp(ga*(1−exp(gi))).
    This is the asymptotic floor for cations."""
    ga, gi = 3.0, 0.4
    asymp = np.exp(ga * (1.0 - np.exp(gi)))
    # Use a very large qmod so qref/qmod is negligible at this tolerance.
    z = zeta_charge_scaling(ga, gi, 1.0, 1.0e12)
    assert z == pytest.approx(asymp, rel=1e-6)
    # Also verify the trend: z decreases monotonically toward the asymptote
    z_nearer = zeta_charge_scaling(ga, gi, 1.0, 10.0)
    assert z_nearer > z  # farther from asymptote = larger ζ


def test_zeta_monotone_with_qmod():
    """For fixed qref, the D4 ζ branch suppresses cationic deviations
    and enhances anionic deviations toward the qmod <= 0 saturation."""
    ga, gi, qref = 3.0, 0.4, 6.0
    z_ref = zeta_charge_scaling(ga, gi, qref, qref)
    # cations: qmod > qref
    z_hi = zeta_charge_scaling(ga, gi, qref, qref + 1.0)
    z_hihi = zeta_charge_scaling(ga, gi, qref, qref + 2.0)
    assert z_hi < z_ref
    assert z_hihi < z_hi
    # anions: qmod < qref
    z_lo = zeta_charge_scaling(ga, gi, qref, qref - 1.0)
    assert z_lo > z_ref
    # but anions above -z_eff
    if qref - 2.0 > 0:
        z_lolo = zeta_charge_scaling(ga, gi, qref, qref - 2.0)
        assert z_lolo > z_lo


def test_zeta_zero_gi_is_flat():
    """gi = 0 → ζ = 1 everywhere (no charge sensitivity)."""
    for qmod in [0.5, 1.0, 2.0, 10.0]:
        z = zeta_charge_scaling(3.0, 0.0, 1.0, qmod)
        assert z == pytest.approx(1.0)


def test_zeta_large_gi_is_steep():
    """Larger gi makes ζ drop faster as charge deviates."""
    z_soft = zeta_charge_scaling(3.0, 0.2, 1.0, 1.5)
    z_hard = zeta_charge_scaling(3.0, 0.8, 1.0, 1.5)
    assert z_hard < z_soft  # steeper decay


def test_zeta_with_varies_ga():
    """Larger ga changes the floor — still 1 at qref, but flattening
    differs."""
    assert zeta_charge_scaling(3.0, 0.4, 1.0, 1.0) == pytest.approx(1.0)
    assert zeta_charge_scaling(5.0, 0.4, 1.0, 1.0) == pytest.approx(1.0)
    # different ga → different suppression away from qref
    z3 = zeta_charge_scaling(3.0, 0.4, 1.0, 3.0)
    z5 = zeta_charge_scaling(5.0, 0.4, 1.0, 3.0)
    assert z3 != z5


# --- Data-table sanity ---


def test_zeta_hardness_light_elements():
    """Spot-checks against the dftd4 hardness.f90 table."""
    assert zeta_hardness(1) == pytest.approx(0.47259288)
    assert zeta_hardness(6) == pytest.approx(0.42195412)
    assert zeta_hardness(7) == pytest.approx(0.50438193)
    assert zeta_hardness(8) == pytest.approx(0.58691863)


def test_zeta_hardness_is_distinct_from_eeq():
    """The D4 ζ hardness is NOT the EEQ chemical hardness — confirm
    by direct spot-check."""
    # EEQ η for C is 0.19408787 (from eeq_charges_data.cpp)
    # D4 ζ hardness for C is 0.42195412
    assert zeta_hardness(6) != pytest.approx(0.19408787)
    # EEQ η for O is 0.03151644
    assert zeta_hardness(8) != pytest.approx(0.03151644)
    # EEQ η for H is −0.35015861 (yes, negative!)
    assert zeta_hardness(1) != pytest.approx(-0.35015861)


def test_zeta_hardness_all_nonnegative():
    """All hardness values for Z=1..86 are positive."""
    for z in range(1, 87):
        assert zeta_hardness(z) > 0.0, f"zeta_hardness({z}) <= 0"


def test_zeta_hardness_bounds():
    with pytest.raises(ValueError, match="outside"):
        zeta_hardness(0)
    with pytest.raises(ValueError, match="outside"):
        zeta_hardness(ZETA_MAX_Z + 1)


def test_effective_charge_light_elements():
    """For H–Ar, Z_eff = Z (bare nuclear charge)."""
    for z in range(1, 19):
        assert effective_nuclear_charge(z) == float(z)


def test_effective_charge_rb_is_screened():
    """Rb (Z=37) has def2-ECP with 28 core e⁻ → Z_eff = 9."""
    assert effective_nuclear_charge(37) == pytest.approx(9.0)


def test_effective_charge_cs_is_screened():
    """Cs (Z=55) has def2-ECP with 46 core e⁻ → Z_eff = 9."""
    assert effective_nuclear_charge(55) == pytest.approx(9.0)


def test_effective_charge_bounds():
    with pytest.raises(ValueError, match="outside"):
        effective_nuclear_charge(0)
    with pytest.raises(ValueError, match="outside"):
        effective_nuclear_charge(ZETA_MAX_Z + 1)


# --- Combined d4_reference_weights ---


def test_d4_weights_shape():
    """Shape matches the number of references."""
    w = d4_reference_weights(
        cn=3.5,
        q=0.0,
        reference_cns=[1.0, 2.0, 3.0, 4.0],
        reference_qs=[0.0, 0.0, 0.0, 0.0],
        z=6,
    )
    assert w.shape == (4,)
    assert np.all(w >= 0.0)


def test_d4_weights_with_zero_charge_equal_cn_weights():
    """When all reference charges equal the atom's charge, ζ = 1 for
    every reference, so the d4 weights equal the CN weights."""
    ref_cns = [1.0, 2.0, 3.0, 4.0]
    ref_qs = [0.0, 0.0, 0.0, 0.0]
    w_d4 = d4_reference_weights(
        cn=2.7,
        q=0.0,
        reference_cns=ref_cns,
        reference_qs=ref_qs,
        z=6,
    )
    w_cn = cn_gaussian_weights(2.7, ref_cns)
    np.testing.assert_allclose(w_d4, w_cn, rtol=1e-12)


def test_d4_weights_charge_suppression():
    """When q differs from a reference's q_ref, ζ < 1 suppresses
    that reference relative to the pure-CN weight."""
    ref_cns = [2.0, 3.0, 4.0]
    # reference 1 (CN 2.0) has q_ref = 0.0 (neutral)
    # references 2 and 3 have q_ref matching atom's charge q=0.8
    ref_qs = [0.0, 0.8, 0.8]
    w = d4_reference_weights(
        cn=3.5,
        q=0.8,
        reference_cns=ref_cns,
        reference_qs=ref_qs,
        z=6,
    )
    # The CN-only weight for CN=2.0 at CN=3.5 would be the raw gw[0].
    # With ζ < 1, the actual weight should be lower.
    gw_cn = cn_gaussian_weights(3.5, ref_cns)
    # reference 1 (q_ref=0.0, mismatch with q=0.8) gets ζ-attenuated
    assert w[0] < gw_cn[0] * 0.99  # at least some attenuation
    # references 2,3 (q_ref=0.8, matches q=0.8) are not attenuated
    assert w[1] == pytest.approx(gw_cn[1], rel=1e-12)
    assert w[2] == pytest.approx(gw_cn[2], rel=1e-12)


def test_d4_weights_mismatched_length_raises():
    with pytest.raises(ValueError, match="same length"):
        d4_reference_weights(
            cn=3.0,
            q=0.0,
            reference_cns=[1.0, 2.0, 3.0],
            reference_qs=[0.0, 0.0],  # too short
            z=6,
        )


def test_d4_weights_default_ga_gc():
    """Default ga/gc values are D4_CHARGE_HEIGHT / D4_CHARGE_STEEPNESS."""
    ref_cns = [2.0, 4.0]
    ref_qs = [0.0, 0.0]
    w_default = d4_reference_weights(
        cn=3.0,
        q=0.0,
        reference_cns=ref_cns,
        reference_qs=ref_qs,
        z=6,
    )
    w_explicit = d4_reference_weights(
        cn=3.0,
        q=0.0,
        reference_cns=ref_cns,
        reference_qs=ref_qs,
        z=6,
        ga=D4_CHARGE_HEIGHT,
        gc=D4_CHARGE_STEEPNESS,
    )
    np.testing.assert_allclose(w_default, w_explicit, rtol=1e-15)


def test_d4_weights_custom_gc_affects_charge_sensitivity():
    """Larger gc → more charge sensitivity → weights shift more
    when charge mismatches."""
    ref_cns = [2.0, 4.0]
    # q_atom = 0.5 matches ref[1] (q_ref=0.5), mismatches ref[0] (q_ref=0.0)
    # For carbon: z_eff=6, so qmod=6.5, qref0=6.0 (mismatch), qref1=6.5 (match)
    # ref[0] mismatch: qmod > qref0 → ζ < 1, larger gc → smaller ζ
    ref_qs = [0.0, 0.5]
    q_atom = 0.5
    w_small_gc = d4_reference_weights(
        cn=3.0,
        q=q_atom,
        reference_cns=ref_cns,
        reference_qs=ref_qs,
        z=6,
        gc=0.5,
    )
    w_large_gc = d4_reference_weights(
        cn=3.0,
        q=q_atom,
        reference_cns=ref_cns,
        reference_qs=ref_qs,
        z=6,
        gc=5.0,
    )
    # At CN=3.0, the pure-CN weight splits 50:50.
    # ref[1] (q_ref=0.5) gets ζ=1 always (match).
    # ref[0] (q_ref=0.0, mismatch) gets ζ < 1; larger gc → ζ smaller.
    # So ratio ref[0]/(ref[0]+ref[1]) decreases with larger gc.
    ratio_small = w_small_gc[0] / (w_small_gc[0] + w_small_gc[1])
    ratio_large = w_large_gc[0] / (w_large_gc[0] + w_large_gc[1])
    assert ratio_large < ratio_small


def test_d4_weights_vs_cn_only_neq():
    """With non-zero charge, d4_weights differ from cn_gaussian_weights
    when some reference_qs differ from the atom's q."""
    ref_cns = [2.0, 3.0, 4.0]
    ref_qs = [-0.3, 0.0, 0.3]
    w_d4 = d4_reference_weights(
        cn=3.0,
        q=0.2,
        reference_cns=ref_cns,
        reference_qs=ref_qs,
        z=6,
    )
    w_cn = cn_gaussian_weights(3.0, ref_cns)
    # at least one weight differs
    assert not np.allclose(w_d4, w_cn, rtol=1e-12)


# =====================================================================
# D4b-4 part 3 — D4ReferenceDataset + compute_d4_c6_pair
# =====================================================================


def test_dataset_construction():
    from vibeqc.dispersion_d4_reference_data import D4ReferenceDataset, _PerElement

    alpha = np.ones((2, 23)) * 10.0
    c6s = np.array([[120.0, 115.0], [115.0, 120.0]])
    elem = _PerElement(z=6, cns=[4.0, 2.0], qs=[-0.3, 0.1], alpha_iw=alpha, c6_self=c6s)
    ds = D4ReferenceDataset(elements={6: elem})
    assert ds.supported_z == [6]
    assert ds.get_cns(6) == [4.0, 2.0]
    assert ds.get_c6_ref(6, 0, 6, 1) == pytest.approx(115.0)


def test_dataset_json_roundtrip(tmp_path):
    from vibeqc.dispersion_d4_reference_data import D4ReferenceDataset, _PerElement

    alpha = np.array([[1.0, 0.5], [0.8, 0.4]])
    c6s = np.array([[10.0, 5.0], [5.0, 8.0]])
    elem = _PerElement(z=1, cns=[1.0], qs=[0.1], alpha_iw=alpha, c6_self=c6s)
    ds = D4ReferenceDataset(elements={1: elem})
    p = tmp_path / "test.json"
    ds.save_json(p)
    loaded = D4ReferenceDataset.load_json(p)
    np.testing.assert_allclose(loaded.elements[1].c6_self, c6s)


def test_compute_d4_c6_single_ref():
    from vibeqc.dispersion_d4_reference_data import D4ReferenceDataset, _PerElement

    alpha = np.ones((1, 23)) * 10.0
    c6s = np.array([[100.0]])
    elem = _PerElement(z=6, cns=[4.0], qs=[-0.3], alpha_iw=alpha, c6_self=c6s)
    ds = D4ReferenceDataset(elements={6: elem})
    c6 = compute_d4_c6_pair(
        cn_a=2.0, q_a=-0.3, z_a=6, cn_b=2.0, q_b=-0.3, z_b=6, ref_data=ds
    )
    assert c6 == pytest.approx(100.0)


def test_compute_d4_c6_interpolation():
    from vibeqc.dispersion_d4_reference_data import D4ReferenceDataset, _PerElement

    alpha = np.ones((2, 23)) * 10.0
    c6s = np.array([[200.0, 100.0], [100.0, 50.0]])
    elem = _PerElement(
        z=6, cns=[4.0, 2.0], qs=[-0.3, -0.3], alpha_iw=alpha, c6_self=c6s
    )
    ds = D4ReferenceDataset(elements={6: elem})
    c6_hi = compute_d4_c6_pair(
        cn_a=4.0, q_a=-0.3, z_a=6, cn_b=4.0, q_b=-0.3, z_b=6, ref_data=ds
    )
    c6_lo = compute_d4_c6_pair(
        cn_a=2.0, q_a=-0.3, z_a=6, cn_b=2.0, q_b=-0.3, z_b=6, ref_data=ds
    )
    c6_mid = compute_d4_c6_pair(
        cn_a=3.0, q_a=-0.3, z_a=6, cn_b=3.0, q_b=-0.3, z_b=6, ref_data=ds
    )
    assert c6_hi > c6_mid > c6_lo


def test_compute_d4_c6_symmetric():
    from vibeqc.dispersion_d4_reference_data import D4ReferenceDataset, _PerElement

    alpha = np.ones((1, 23)) * 10.0
    c6s = np.array([[42.0]])
    elem6 = _PerElement(
        z=6, cns=[4.0], qs=[-0.3], alpha_iw=alpha.copy(), c6_self=c6s.copy()
    )
    elem1 = _PerElement(
        z=1, cns=[1.0], qs=[0.1], alpha_iw=alpha.copy() * 0.2, c6_self=np.array([[5.0]])
    )
    ds = D4ReferenceDataset(elements={1: elem1, 6: elem6})
    # Add cross c6
    from vibeqc.dispersion_d4_refdata import casimir_polder_c6, imaginary_frequency_grid

    _, wts = imaginary_frequency_grid(23, 0.5)
    cross = np.array([[casimir_polder_c6(alpha[0], alpha[0] * 0.2, wts)]])
    ds.c6_cross[(1, 6)] = cross
    c6_ab = compute_d4_c6_pair(
        cn_a=1.0, q_a=0.1, z_a=1, cn_b=4.0, q_b=-0.3, z_b=6, ref_data=ds
    )
    c6_ba = compute_d4_c6_pair(
        cn_a=4.0, q_a=-0.3, z_a=6, cn_b=1.0, q_b=0.1, z_b=1, ref_data=ds
    )
    assert c6_ab == pytest.approx(c6_ba)


# =====================================================================
# D4b-5 — r4r2 table, C8, BJ damping energy, per-functional parameters
# =====================================================================


def test_r4r2_spots():
    """r4r2 values must match the dftd4 pre-computed table
    (``sqrt_z_r4_over_r2`` = sqrt(0.5 * sqrt(Z) * <r⁴>/<r²>)).

    Regression for the 2026-06-26 fix: 117/118 entries had shipped
    without the sqrt(Z) factor (only H, where sqrt(Z)=1, was right),
    inflating C8 / shrinking the BJ radius R0 and over-binding native
    D4 energies regardless of the C6 data. These values are pinned, not
    just asserted positive, so the transcription cannot silently break
    again.
    """
    # dftd4 stores r4r2_st(Z) = sqrt(0.5 * sqrt(Z) * <r⁴>/<r²>(Z)).
    assert r4r2_val(1) == pytest.approx(2.00734900, abs=1e-6)   # H
    assert r4r2_val(2) == pytest.approx(1.56637132, abs=1e-6)   # He
    assert r4r2_val(6) == pytest.approx(3.10492822, abs=1e-6)   # C
    assert r4r2_val(7) == pytest.approx(2.71175242, abs=1e-6)   # N
    assert r4r2_val(8) == pytest.approx(2.59361682, abs=1e-6)   # O
    assert r4r2_val(9) == pytest.approx(2.38825250, abs=1e-6)   # F
    assert r4r2_val(54) == pytest.approx(5.25477002, abs=1e-6)  # Xe


def test_r4r2_bounds():
    with pytest.raises(ValueError, match="outside"):
        r4r2_val(0)
    with pytest.raises(ValueError, match="outside"):
        r4r2_val(R4R2_MAX_Z + 1)


def test_compute_c8_proportional():
    """C8 is proportional to C6."""
    c8_1 = compute_c8(10.0, 6, 6)
    c8_2 = compute_c8(20.0, 6, 6)
    assert c8_2 == pytest.approx(2.0 * c8_1, rel=1e-12)


def test_bj_damping_energy():
    """BJ damped energy is negative and decays with distance."""
    c6, c8 = 20.0, compute_c8(20.0, 6, 6)
    e1 = d4_bj_damping_energy(3.0, c6, c8, 1.0, 0.96, 0.386, 4.807)
    e2 = d4_bj_damping_energy(6.0, c6, c8, 1.0, 0.96, 0.386, 4.807)
    assert e1 < 0.0
    assert e2 < 0.0
    assert abs(e2) < abs(e1)  # decays with distance


def test_d4_parameters_pbe():
    from vibeqc.dispersion_d4_parameters import get_d4_params

    p = get_d4_params("pbe")
    assert p.s6 == 1.0
    assert p.s8 == pytest.approx(0.95948085)
    assert p.a1 == pytest.approx(0.38574991)


def test_d4_parameters_b2plyp():
    from vibeqc.dispersion_d4_parameters import get_d4_params

    p = get_d4_params("B2PLYP")
    assert p.s6 == 0.64  # double-hybrid scaling
    assert p.s8 > 0.0


def test_d4_parameters_case_insensitive():
    from vibeqc.dispersion_d4_parameters import get_d4_params

    p1 = get_d4_params("pbe")
    p2 = get_d4_params("PBE")
    p3 = get_d4_params("Pbe")
    assert p1 == p2 == p3


def test_d4_parameters_missing():
    from vibeqc.dispersion_d4_parameters import get_d4_params

    with pytest.raises(KeyError, match="No D4 parameters"):
        get_d4_params("nonexistent_functional_xyz")


def test_d4_parameters_bhlyp_pinned():
    """BHLYP gets its own dftd4 row, not a neighbour's.

    Regression for the v0.10.0 transcription error where the dftd4
    "bhlyp" row landed under the key "hcth120" — and from the
    bj-eeq-mbd column at that — with an alias "bhlyp" -> "hcth120"
    on top, so BHLYP-D4 silently used MBD-variant damping.
    """
    from vibeqc.dispersion_d4_parameters import get_d4_params

    # dftd4 parameters.toml [parameter.bhlyp.d4.bj-eeq-atm]
    # (Caldeweyher et al. 2019, doi:10.1063/1.5090222).
    for spelling in ("bhlyp", "BHLYP", "BH-LYP", "b-h-lyp"):
        p = get_d4_params(spelling)
        assert p.s6 == 1.0
        assert p.s8 == pytest.approx(1.65281646, abs=1e-8)
        assert p.a1 == pytest.approx(0.27263660, abs=1e-8)
        assert p.a2 == pytest.approx(5.48634586, abs=1e-8)
        assert p.s9 == 1.0
        assert p.doi == "10.1063/1.5090222"


def test_d4_parameters_hcth120_removed():
    """dftd4 ships no HCTH-120 D4 parametrisation — the lookup must
    fail loudly rather than return another functional's damping."""
    from vibeqc.dispersion_d4_parameters import get_d4_params

    with pytest.raises(KeyError, match="No D4 parameters"):
        get_d4_params("hcth120")


def test_d4_parameters_exact_rows_replacing_aliases():
    """mpwpw / b1lyp / revpbe0dh have their exact dftd4 rows (they
    were approximate or dangling aliases before), and revdsdpbep86
    carries its own s6, not revdodpbep86's.

    Pinned to dftd4 parameters.toml bj-eeq-atm rows
    (doi:10.1063/1.5090222; revDSD/revDOD rows from Santra,
    Sylvetsky & Martin 2019, doi:10.1021/acs.jpca.9b03157).
    """
    from vibeqc.dispersion_d4_parameters import get_d4_params

    p = get_d4_params("mpwpw")  # was a dangling alias -> "mpw1pw"
    assert (p.s6, p.s9) == (1.0, 1.0)
    assert p.s8 == pytest.approx(1.82596836, abs=1e-8)
    assert p.a1 == pytest.approx(0.34526745, abs=1e-8)
    assert p.a2 == pytest.approx(4.84620734, abs=1e-8)

    p = get_d4_params("b1lyp")  # was approximated by blyp's row
    assert p.s8 == pytest.approx(1.98553711, abs=1e-8)
    assert p.a1 == pytest.approx(0.39309040, abs=1e-8)
    assert p.a2 == pytest.approx(4.55465145, abs=1e-8)

    p = get_d4_params("revpbe0dh")  # was approximated by pbe0dh's row
    assert p.s6 == pytest.approx(0.8750, abs=1e-8)
    assert p.s8 == pytest.approx(1.24456037, abs=1e-8)
    assert p.a1 == pytest.approx(0.36730560, abs=1e-8)
    assert p.a2 == pytest.approx(4.71126482, abs=1e-8)

    # s6 = 0.5132 (revDSD-PBEP86) vs 0.5552 (revDOD-PBEP86): the two
    # rows differ only in s6, which is exactly how the original
    # copy-paste error slipped in.
    assert get_d4_params("revdsdpbep86").s6 == pytest.approx(0.5132)
    assert get_d4_params("revdodpbep86").s6 == pytest.approx(0.5552)


def test_d4_parameters_full_table_matches_dftd4_toml():
    """Sweep the entire native table against the canonical
    parameters.toml shipped by the dftd4 package: every entry must
    equal its bj-eeq-atm row (the D4 default). This is the test that
    would have caught the bhlyp/hcth120 mis-key at transcription
    time — it guards both wrong-row and wrong-variant errors.
    """
    import os

    dftd4 = pytest.importorskip("dftd4")
    tomllib = pytest.importorskip("tomllib")

    toml_path = os.path.join(os.path.dirname(dftd4.__file__), "parameters.toml")
    if not os.path.exists(toml_path):
        pytest.skip("dftd4 package does not ship parameters.toml")
    with open(toml_path, "rb") as fh:
        reference = tomllib.load(fh)["parameter"]

    from vibeqc.dispersion_d4_parameters import _PARAMS

    def norm(name):
        return name.lower().replace("-", "").replace("_", "").replace(" ", "")

    ref_by_norm = {norm(k): v for k, v in reference.items()}
    # vibe-qc key -> dftd4 TOML key, where the spelling differs.
    renames = {"bp86": "bp", "bpw91": "bpw"}
    # xTB Hamiltonians carry their own published D4 fits (JCTC 2019 /
    # 2017) and are not part of the DFT parameter file.
    not_in_toml = {"gfn1xtb", "gfn2xtb"}

    for key, p in _PARAMS.items():
        if key in not_in_toml:
            continue
        entry = ref_by_norm.get(norm(renames.get(key, key)))
        assert entry is not None, f"{key}: no dftd4 reference entry"
        row = entry["d4"]["bj-eeq-atm"]
        for field, ours in (
            ("s6", p.s6), ("s8", p.s8), ("a1", p.a1), ("a2", p.a2), ("s9", p.s9)
        ):
            theirs = row.get(field, 1.0)
            assert ours == pytest.approx(theirs, abs=1e-8), (
                f"{key}.{field}: vibe-qc has {ours}, dftd4 bj-eeq-atm has {theirs}"
            )


# ---------------------------------------------------------------------
# Three-body Axilrod-Teller-Muto term (Phase D4b-6).
# ---------------------------------------------------------------------


def test_compute_c9_geometric_mean():
    """C9 = − sqrt(C6_ab · C6_ac · C6_bc) — negative, symmetric."""
    from vibeqc.dispersion_d4_model import compute_c9

    c9 = compute_c9(10.0, 40.0, 90.0)
    assert c9 == pytest.approx(-np.sqrt(10.0 * 40.0 * 90.0))
    assert c9 < 0.0
    # Symmetric under any permutation of the three pair C6 values.
    assert compute_c9(10.0, 40.0, 90.0) == pytest.approx(compute_c9(90.0, 10.0, 40.0))


def test_atm_triple_energy_equilateral_closed_form():
    """For an equilateral triangle (all sides r, all angles 60°) the
    angle factor 3·cos³60° + 1 = 3/8 + 1 = 1.375, so the undamped
    triple energy is exactly C9 · 1.375 / r⁹. Verified at a distance
    large enough that the zero-damping factor is ≈ 1."""
    from vibeqc.dispersion_d4_model import d4_atm_triple_energy

    r = 30.0  # far apart → f_damp ≈ 1
    c9 = -50.0
    r0 = 4.0  # small vs r → f_damp ≈ 1
    e = d4_atm_triple_energy(r, r, r, c9, r0, r0, r0)
    expected = c9 * 1.375 / r**9
    assert e == pytest.approx(expected, rel=1e-6)


def test_atm_damping_suppresses_short_range():
    """The zero-damping factor kills the 1/r⁹ divergence: at a triple
    distance well inside the critical radius the damped energy is a
    tiny fraction of the undamped 1/r⁹ value."""
    from vibeqc.dispersion_d4_model import d4_atm_triple_energy

    c9, r0 = -50.0, 6.0
    r = 2.0  # well inside r0 → strong damping
    e_damped = d4_atm_triple_energy(r, r, r, c9, r0, r0, r0)
    e_undamped = c9 * 1.375 / r**9
    assert abs(e_damped) < 0.05 * abs(e_undamped)


def test_atm_triple_energy_far_limit_undamped():
    """At large separation the damping factor → 1 and the triple
    energy → the bare ATM kernel; it also decays as 1/r⁹."""
    from vibeqc.dispersion_d4_model import d4_atm_triple_energy

    c9, r0 = -100.0, 5.0
    e1 = d4_atm_triple_energy(20.0, 20.0, 20.0, c9, r0, r0, r0)
    e2 = d4_atm_triple_energy(40.0, 40.0, 40.0, c9, r0, r0, r0)
    # Doubling every side scales 1/r⁹ by 2⁻⁹ = 1/512.
    assert e1 / e2 == pytest.approx(512.0, rel=1e-4)


def test_atm_angle_factor_collinear():
    """A (near-)collinear triple has angle factor 3·cosθ_a·cosθ_b·cosθ_c
    + 1 → −2 (one 180° angle, two 0° angles), so the triple energy
    flips sign relative to the compact equilateral case — the
    geometry dependence the ATM term exists to capture."""
    from vibeqc.dispersion_d4_model import d4_atm_triple_energy

    c9, r0 = -50.0, 3.0
    # Collinear: B between A and C, sides r_ab = r_bc = d, r_ac = 2d.
    d = 25.0
    e = d4_atm_triple_energy(d, 2.0 * d, d, c9, r0, r0, r0)
    # angle factor → −2 ⇒ sign opposite the equilateral (+1.375) case.
    e_equilateral = d4_atm_triple_energy(d, d, d, c9, r0, r0, r0)
    assert e * e_equilateral < 0.0


def test_compute_d4_atm_energy_exported():
    """The full-molecule ATM driver is importable from the model
    module and the top-level surface lists the three D4b-6 symbols."""
    from vibeqc.dispersion_d4_model import (
        compute_c9,
        compute_d4_atm_energy,
        d4_atm_triple_energy,
    )

    assert callable(compute_c9)
    assert callable(d4_atm_triple_energy)
    assert callable(compute_d4_atm_energy)


# ---------------------------------------------------------------------
# Native-backend dftd4 parity (the un-gating bar, now MET).
#
# The shipped d4_reference_data.json is built from the per-atom Eq.-6
# extraction over a correlated CPKS (PBE38/TD-DFT) alpha(iw), and the
# r4r2 / C8 table is dftd4-consistent (both landed 2026-06-26). Native
# pairwise C6 now agree with dftd4 to within a few percent and the
# CH4-dimer D4 energy to <0.05 kcal/mol, so the parity tests below are
# real (no longer strict-xfail) and the experimental warning gate is
# removed.
#
# Parity bars are handovers/HANDOVER_D4_NATIVE.md § 5: pairwise C6 within
# a few % (10% asserted here) and dimer dispersion energy within
# 0.1 kcal/mol of the dftd4 backend. These guard against a regression
# in the reference data, the extraction, the CPKS provider, or the
# r4r2 table; if a future regeneration cannot meet them, loosen
# deliberately with the maintainer -- don't widen silently.
# ---------------------------------------------------------------------

import warnings as _warnings
from pathlib import Path as _Path

from vibeqc import Atom as _Atom
from vibeqc import Molecule as _Molecule
from vibeqc.dispersion_d4 import (
    D4NativeExperimentalWarning,
    compute_d4,
    dftd4_available,
)

from .conftest import GEOMETRIES as _GEOMETRIES

_REFDATA_JSON = _Path(__file__).resolve().parents[1] / "d4_reference_data.json"

_needs_refdata = pytest.mark.skipif(
    not _REFDATA_JSON.is_file(),
    reason="d4_reference_data.json not present (repo-root checkout only)",
)
_needs_dftd4 = pytest.mark.skipif(
    not dftd4_available(),
    reason="dftd4 not installed; pip install -e '.[dispersion]'",
)


def _mol_from_geometry(name):
    return _Molecule([_Atom(z, list(xyz)) for z, xyz in _GEOMETRIES[name]])


def _ch4_dimer():
    """Two conftest CH4 monomers, second shifted +7 bohr in z."""
    atoms = [_Atom(z, list(xyz)) for z, xyz in _GEOMETRIES["CH4"]]
    atoms += [
        _Atom(z, [xyz[0], xyz[1], xyz[2] + 7.0]) for z, xyz in _GEOMETRIES["CH4"]
    ]
    return _Molecule(atoms)


def _dftd4_c6_matrix(mol):
    """Pairwise C6 matrix from the dftd4 package (positions in bohr)."""
    from dftd4.interface import DispersionModel

    numbers = np.array([atom.Z for atom in mol.atoms], dtype=np.int32)
    positions = np.array([atom.xyz for atom in mol.atoms], dtype=np.float64)
    model = DispersionModel(numbers=numbers, positions=positions, charge=0.0)
    return np.asarray(model.get_properties()["c6 coefficients"], dtype=float)


@_needs_refdata
def test_native_backend_no_experimental_warning():
    """backend='native' is un-gated: no experimental warning is emitted
    (regression for the 2026-06-26 un-gate)."""
    mol = _mol_from_geometry("H2")
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", D4NativeExperimentalWarning)
        result = compute_d4(
            mol, "pbe", backend="native", ref_data_path=str(_REFDATA_JSON)
        )
    # Smoke: the native path computes a bound dispersion energy.
    assert result.energy < 0.0


@_needs_refdata
@_needs_dftd4
def test_native_c6_parity_vs_dftd4():
    """Native pairwise C6 vs the dftd4 package within 10% (CH4 + H2O)."""
    from vibeqc.dispersion_d4_model import _compute_c6_matrix
    from vibeqc.dispersion_d4_reference_data import D4ReferenceDataset

    ref_data = D4ReferenceDataset.load_json(_REFDATA_JSON)
    for name in ("CH4", "H2O"):
        mol = _mol_from_geometry(name)
        c6_native = _compute_c6_matrix(mol, ref_data)
        c6_dftd4 = _dftd4_c6_matrix(mol)
        n = len(mol.atoms)
        for a in range(n):
            for b in range(a):
                assert c6_native[a, b] == pytest.approx(
                    c6_dftd4[a, b], rel=0.10
                ), (
                    f"{name} C6[{a},{b}]: native {c6_native[a, b]:.3f} vs "
                    f"dftd4 {c6_dftd4[a, b]:.3f}"
                )


@_needs_refdata
@_needs_dftd4
def test_native_energy_parity_vs_dftd4():
    """Native CH4-dimer D4 energy within 0.1 kcal/mol of dftd4 (PBE)."""
    mol = _ch4_dimer()
    e_native = compute_d4(
        mol, "pbe", backend="native", ref_data_path=str(_REFDATA_JSON)
    ).energy
    e_dftd4 = compute_d4(mol, "pbe", backend="dftd4").energy
    # 0.1 kcal/mol in Hartree — the handover § 5 chemical-accuracy bar.
    assert abs(e_native - e_dftd4) < 0.1 / 627.5094740631, (
        f"E_disp native {e_native:.8f} vs dftd4 {e_dftd4:.8f} Ha"
    )
