"""Unit-level tests for the v0.10.x ``vibeqc.smearing`` package.

The package is the new home for fractional-occupation smearing
utilities. These tests cover the new public surfaces
(``SmearingOptions``, ``SmearingResult``, ``apply_smearing``,
``closed_shell_periodic_occupations``) plus the back-compat
guarantees:

* every symbol that used to live in ``vibeqc.occupations`` is still
  importable from there;
* the new package re-exports the same symbols on
  ``vibeqc.smearing``;
* both paths produce the same numerical results.

Per-backend integration coverage lives in
``test_periodic_smearing.py`` (existing) and grows per-cell as
M2–M8 land.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc

import vibeqc as vq
from vibeqc import smearing as sm
from vibeqc.periodic_k_gdf import (
    _is_per_k_integer_aufbau,
    _occupations_per_k as _gdf_occupations_per_k,
)
from vibeqc.smearing.apply import _aufbau_with_midgap_mu
from vibeqc.smearing.marzari_vanderbilt import (
    _mv_entropy_integrand,
    _mv_occupation,
)
from vibeqc.smearing.methfessel_paxton import (
    _mp_entropy_integrand,
    _mp_occupation,
)


# ---------------------------------------------------------------------------
# Back-compat: vibeqc.occupations is still importable and re-exports
# every symbol that lived there at v0.4.0 / v0.9.0.
# ---------------------------------------------------------------------------

def test_occupations_module_still_imports():
    """The shim keeps every public name from the old module."""
    from vibeqc import occupations
    expected = {
        "EV_PER_HARTREE",
        "HARTREE_PER_RYDBERG",
        "KB_HARTREE_PER_K",
        "SMEARING_PRESETS",
        "SmearingResolution",
        "aufbau_occupations_per_k",
        "electronvolt_to_hartree_temperature",
        "fermi_dirac_occupations_per_k",
        "guess_smearing_temperature",
        "hartree_to_kelvin_temperature",
        "kelvin_to_hartree_temperature",
        "resolve_smearing_temperature",
        "rydberg_to_hartree_temperature",
    }
    for name in expected:
        assert hasattr(occupations, name), f"occupations shim missing {name!r}"


def test_occupations_and_smearing_are_the_same_objects():
    """Shim re-exports — identity-equal, not just equivalent."""
    from vibeqc import occupations
    for name in (
        "SmearingResolution",
        "fermi_dirac_occupations_per_k",
        "aufbau_occupations_per_k",
        "resolve_smearing_temperature",
        "guess_smearing_temperature",
        "kelvin_to_hartree_temperature",
        "KB_HARTREE_PER_K",
    ):
        assert getattr(occupations, name) is getattr(sm, name), (
            f"{name!r} should be the same object on both modules"
        )


def test_vibeqc_namespace_exposes_new_surfaces():
    """The top-level ``vq.*`` surface gains the new smearing API."""
    assert vq.SmearingOptions is sm.SmearingOptions
    assert vq.SmearingResult is sm.SmearingResult
    assert vq.apply_smearing is sm.apply_smearing
    assert vq.closed_shell_periodic_occupations is sm.closed_shell_periodic_occupations
    # Unchanged surfaces are still on ``vq`` too.
    assert vq.SmearingResolution is sm.SmearingResolution
    assert vq.kelvin_to_hartree_temperature is sm.kelvin_to_hartree_temperature
    assert vq.temperature_in_hartree is sm.temperature_in_hartree


# ---------------------------------------------------------------------------
# SmearingOptions dataclass
# ---------------------------------------------------------------------------

def test_smearing_options_defaults_disabled():
    so = vq.SmearingOptions()
    assert so.temperature == 0.0
    assert so.flavor == "fermi-dirac"
    assert so.mp_order == 1
    assert not so.enabled


def test_smearing_options_temperature_positive():
    so = vq.SmearingOptions(temperature=0.005)
    assert so.enabled
    assert so.temperature == 0.005


def test_smearing_options_rejects_negative_temperature():
    with pytest.raises(ValueError, match="must be >= 0"):
        vq.SmearingOptions(temperature=-1e-3)


def test_smearing_options_rejects_unknown_flavor():
    with pytest.raises(ValueError, match="flavor must be one of"):
        vq.SmearingOptions(flavor="gaussian")


def test_smearing_options_accepts_alias_spellings_for_flavor():
    # "fermi_dirac" with underscore normalises to canonical form.
    so = vq.SmearingOptions(flavor="fermi_dirac")
    assert so.flavor == "fermi-dirac"


def test_legacy_mermin_smearing_method_alias():
    res = vq.resolve_smearing_temperature(
        "metal",
        method="meremin",
        metallic=True,
    )
    # "meremin" normalises to the canonical "mermin" flavour.
    assert res.method == "mermin"
    assert res.temperature == pytest.approx(0.005)


def test_smearing_options_methfessel_paxton_validates_mp_order():
    vq.SmearingOptions(flavor="methfessel-paxton", mp_order=1)
    vq.SmearingOptions(flavor="methfessel-paxton", mp_order=2)
    with pytest.raises(ValueError, match="mp_order must be 1 or 2"):
        vq.SmearingOptions(flavor="methfessel-paxton", mp_order=5)


def test_smearing_options_from_user_preset():
    so = vq.SmearingOptions.from_user("metal")
    assert so.temperature == pytest.approx(0.005)
    assert so.source == "preset:metal"


def test_smearing_options_from_user_kelvin():
    so = vq.SmearingOptions.from_user(1000.0, unit="kelvin")
    assert so.temperature == pytest.approx(
        1000.0 * vq.KB_HARTREE_PER_K
    )
    assert so.source == "explicit:kelvin"


def test_smearing_options_from_legacy_kwarg_zero():
    so = vq.SmearingOptions.from_legacy_kwarg(0.0)
    assert so.temperature == 0.0
    assert so.source == "explicit"


def test_smearing_options_from_legacy_kwarg_nonzero():
    so = vq.SmearingOptions.from_legacy_kwarg(0.005)
    assert so.temperature == pytest.approx(0.005)
    assert so.source == "legacy_kwarg"


# ---------------------------------------------------------------------------
# apply_smearing — closed-shell, multi-k
# ---------------------------------------------------------------------------

@pytest.fixture
def closed_shell_h2_like_eps():
    """4-band, 2 k-point eigenvalue set, 2 electrons (closed-shell)."""
    return [
        np.array([-0.50, -0.30, 0.10, 0.20]),
        np.array([-0.40, -0.20, 0.15, 0.25]),
    ]


def test_apply_smearing_t_zero_returns_aufbau(closed_shell_h2_like_eps):
    res = vq.apply_smearing(
        closed_shell_h2_like_eps,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    )
    assert np.allclose(res.occupations_per_k[0], [2.0, 0.0, 0.0, 0.0])
    assert np.allclose(res.occupations_per_k[1], [2.0, 0.0, 0.0, 0.0])
    assert res.entropy == 0.0
    assert res.free_energy_correction == 0.0
    # μ = midgap = (max HOMO + min LUMO) / 2
    assert res.mu == pytest.approx(-0.35)


def test_gdf_t_zero_uses_one_global_fermi_level():
    """Regression for PERIODIC-K6-NONCONV.

    Independent per-k Aufbau fills the 0.30-Ha state at k0 and leaves the
    -0.20-Ha state at k1 empty. A periodic zero-temperature state instead
    has one chemical potential across the Brillouin zone, so the two lowest
    states are both occupied at k1 while conserving two electrons per cell.

    The archived Si/def2-SVP/PBE 6x6x6 failure had the same signature at
    production scale: a -0.080825-Ha indirect gap with exactly 14 occupied
    bands forced at every k point.
    """
    occ, mu, entropy = _gdf_occupations_per_k(
        [np.array([0.30, 0.40]), np.array([-0.30, -0.20])],
        np.array([0.5, 0.5]),
        n_elec_per_cell=2,
        smearing_T=0.0,
        n_occ_each=1,
    )

    np.testing.assert_allclose(occ[0], [0.0, 0.0])
    np.testing.assert_allclose(occ[1], [2.0, 2.0])
    assert mu == pytest.approx(0.05)
    assert entropy == 0.0
    assert sum(
        w * float(np.sum(occ_k))
        for w, occ_k in zip([0.5, 0.5], occ)
    ) == pytest.approx(2.0)


def test_gdf_t_zero_preserves_fermi_degeneracy():
    """A partially filled Fermi group must not be split by array order."""
    occ, mu, entropy = _gdf_occupations_per_k(
        [np.array([-1.0, 0.0]), np.array([0.0, 1.0])],
        np.array([0.5, 0.5]),
        n_elec_per_cell=2,
        smearing_T=0.0,
        n_occ_each=1,
    )

    np.testing.assert_allclose(occ[0], [2.0, 1.0])
    np.testing.assert_allclose(occ[1], [1.0, 0.0])
    assert mu == pytest.approx(0.0)
    assert entropy == 0.0
    assert sum(
        w * float(np.sum(occ_k))
        for w, occ_k in zip([0.5, 0.5], occ)
    ) == pytest.approx(2.0)


def test_gdf_fixed_subspace_consumers_reject_global_band_overlap():
    """Only the legacy equal-band-count pattern has one fixed subspace."""
    assert _is_per_k_integer_aufbau(
        [np.array([2.0, 0.0]), np.array([2.0, 0.0])],
        n_occ_each=1,
    )
    assert not _is_per_k_integer_aufbau(
        [np.array([0.0, 0.0]), np.array([2.0, 2.0])],
        n_occ_each=1,
    )
    assert not _is_per_k_integer_aufbau(
        [np.array([2.0, 1.0]), np.array([1.0, 0.0])],
        n_occ_each=1,
    )


def test_apply_smearing_t_positive_conserves_electrons(
    closed_shell_h2_like_eps,
):
    res = vq.apply_smearing(
        closed_shell_h2_like_eps,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=vq.SmearingOptions(temperature=0.01),
    )
    n_total = sum(
        float(w) * float(occ.sum())
        for w, occ in zip([0.5, 0.5], res.occupations_per_k)
    )
    assert n_total == pytest.approx(2.0, abs=1e-10)
    assert res.entropy >= 0.0


def test_apply_smearing_free_energy_matches_minus_t_s(
    closed_shell_h2_like_eps,
):
    so = vq.SmearingOptions(temperature=0.01)
    res = vq.apply_smearing(
        closed_shell_h2_like_eps,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=so,
    )
    assert res.free_energy_correction == pytest.approx(
        -so.temperature * res.entropy
    )


def test_apply_smearing_accepts_bare_float_for_back_compat(
    closed_shell_h2_like_eps,
):
    """Drivers can pass a plain float as a shortcut."""
    res_float = vq.apply_smearing(
        closed_shell_h2_like_eps,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=0.01,
    )
    res_obj = vq.apply_smearing(
        closed_shell_h2_like_eps,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=vq.SmearingOptions(temperature=0.01),
    )
    for o_f, o_o in zip(res_float.occupations_per_k, res_obj.occupations_per_k):
        assert np.allclose(o_f, o_o)


@pytest.mark.parametrize(
    "so",
    [
        vq.SmearingOptions(temperature=0.01, flavor="methfessel-paxton", mp_order=1),
        vq.SmearingOptions(temperature=0.01, flavor="methfessel-paxton", mp_order=2),
        vq.SmearingOptions(temperature=0.01, flavor="marzari-vanderbilt"),
    ],
    ids=["mp-order1", "mp-order2", "marzari-vanderbilt"],
)
def test_apply_smearing_methfessel_paxton_and_marzari_vanderbilt(
    closed_shell_h2_like_eps, so
):
    """Methfessel-Paxton (M6, orders 1 & 2) and Marzari-Vanderbilt (M7) are
    *implemented* — ``apply_smearing`` computes their occupations, it does not
    raise. The decisive physical invariant is electron conservation; the
    free-energy correction must equal ``-T·S``.
    """
    res = vq.apply_smearing(
        closed_shell_h2_like_eps,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=so,
    )
    occ = res.occupations_per_k
    # Electron conservation (the robust invariant for any valid smearing).
    n_total = sum(float(w) * float(o.sum()) for w, o in zip([0.5, 0.5], occ))
    assert n_total == pytest.approx(2.0, abs=1e-9)
    # Occupations are finite. The generalized kernels can overshoot their
    # nominal [0, g=2] interval; cold smearing reaches about 2.17 at g=2.
    allocc = np.concatenate([np.asarray(o, dtype=float) for o in occ])
    assert np.all(np.isfinite(allocc))
    assert allocc.min() >= -0.05 and allocc.max() <= 2.2
    # Free-energy correction is exactly -T·S.
    assert res.free_energy_correction == pytest.approx(
        -float(so.temperature) * res.entropy
    )
    assert np.isfinite(res.entropy)


@pytest.mark.parametrize("order", [1, 2])
def test_methfessel_paxton_entropy_matches_published_formula(order):
    """MP generalized entropy is the terminal Hermite contribution."""
    x = np.array([-1.25, -0.25, 0.5, 1.5])
    if order == 1:
        expected = (
            (1.0 - 2.0 * x * x) * np.exp(-x * x) / (4.0 * np.sqrt(np.pi))
        )
    else:
        expected = (
            (4.0 * x**4 - 12.0 * x * x + 3.0)
            * np.exp(-x * x)
            / (16.0 * np.sqrt(np.pi))
        )
    assert np.allclose(_mp_entropy_integrand(x, order), expected)


def test_marzari_vanderbilt_kernel_matches_published_formula():
    """MV Eq. (1)/(2), transformed to eps-minus-mu per-spin variables."""
    x = np.array([-1.25, -0.25, 0.5, 1.5])
    shifted = x + 1.0 / np.sqrt(2.0)
    expected_occ = (
        0.5 * erfc(shifted)
        + np.exp(-shifted * shifted) / np.sqrt(2.0 * np.pi)
    )
    expected_entropy = (
        np.exp(-shifted * shifted) * (1.0 + np.sqrt(2.0) * x)
        / (2.0 * np.sqrt(np.pi))
    )
    assert np.allclose(_mv_occupation(x), expected_occ)
    assert np.allclose(_mv_entropy_integrand(x), expected_entropy)


@pytest.mark.parametrize(
    ("occupation", "entropy"),
    [
        pytest.param(
            lambda x: _mp_occupation(x, 1),
            lambda x: _mp_entropy_integrand(x, 1),
            id="methfessel-paxton-order-1",
        ),
        pytest.param(
            lambda x: _mp_occupation(x, 2),
            lambda x: _mp_entropy_integrand(x, 2),
            id="methfessel-paxton-order-2",
        ),
        pytest.param(
            _mv_occupation,
            _mv_entropy_integrand,
            id="marzari-vanderbilt",
        ),
    ],
)
def test_generalized_entropy_is_variational_conjugate(occupation, entropy):
    """The free-energy kernels obey s'(x) = x f'(x) pointwise."""
    x = np.array([-2.0, -0.75, -0.2, 0.4, 1.1, 2.0])
    h = 1.0e-5
    ds_dx = (entropy(x + h) - entropy(x - h)) / (2.0 * h)
    df_dx = (occupation(x + h) - occupation(x - h)) / (2.0 * h)
    assert np.allclose(ds_dx, x * df_dx, rtol=2.0e-9, atol=2.0e-10)


def test_apply_smearing_t_zero_is_aufbau_for_any_flavor(
    closed_shell_h2_like_eps,
):
    # temperature == 0 fast-paths to integer Aufbau filling regardless of the
    # requested flavor (the flavor only matters at T > 0).
    for flavor in ("methfessel-paxton", "marzari-vanderbilt"):
        so = vq.SmearingOptions(temperature=0.0, flavor=flavor)
        res = vq.apply_smearing(
            closed_shell_h2_like_eps,
            weights=[0.5, 0.5],
            n_electrons_per_cell=2.0,
            n_occ_each=1,
            smearing=so,
        )
        assert np.allclose(res.occupations_per_k[0], [2.0, 0.0, 0.0, 0.0])
        assert res.entropy == 0.0


# ---------------------------------------------------------------------------
# closed_shell_periodic_occupations — driver-friendly tuple-returning wrapper
# ---------------------------------------------------------------------------

def test_closed_shell_periodic_occupations_matches_apply_smearing(
    closed_shell_h2_like_eps,
):
    occ, mu, entropy = vq.closed_shell_periodic_occupations(
        closed_shell_h2_like_eps,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing_temperature=0.01,
    )
    res = vq.apply_smearing(
        closed_shell_h2_like_eps,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=vq.SmearingOptions(temperature=0.01),
    )
    for o_t, o_r in zip(occ, res.occupations_per_k):
        assert np.allclose(o_t, o_r)
    assert mu == pytest.approx(res.mu)
    assert entropy == pytest.approx(res.entropy)


def test_closed_shell_periodic_occupations_t_zero_branch(
    closed_shell_h2_like_eps,
):
    occ, mu, entropy = vq.closed_shell_periodic_occupations(
        closed_shell_h2_like_eps,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing_temperature=0.0,
    )
    assert np.allclose(occ[0], [2.0, 0.0, 0.0, 0.0])
    assert entropy == 0.0
    assert mu == pytest.approx(-0.35)


# ---------------------------------------------------------------------------
# GitLab #85 -- the T = 0 multi-k branch of the shared dispatcher
#
# ``apply_smearing`` is the entry point every backend driver calls, and its
# ``temperature == 0`` branch filled the lowest ``n_occ_each`` bands at every
# k point independently. It computed one global midgap ``mu`` and then did not
# use it, so on any mesh with a band crossing that mu the occupied set is
# wrong: a state above mu at one k is filled while a state below mu at another
# k is left empty. ``periodic_runner._band_summary`` classifies band edges
# from exactly these occupations, so it then reports CBM - VBM < 0 -- the
# nonphysical negative indirect gap of #85.
#
# The native multi-k GDF route was moved to one global Fermi level in
# 60104fc01 (pinned by test_gdf_t_zero_uses_one_global_fermi_level above), and
# the C++ k-point routine takes the same branch
# (compute_closed_shell_kpoint_occupations, global fill for more than one
# spectrum). The shared Python dispatcher was not, so every driver reaching
# T = 0 through it -- Ewald-3D multi-k RHF/RKS, BIPOLE RKS multi-k, and the
# GAPW multi-k routes -- kept the per-k fill.
# ---------------------------------------------------------------------------


# Two k points, three bands, two electrons per cell. Band 0 at k0 (-0.20) lies
# ABOVE bands 0 and 1 at k1 (-0.60, -0.50), so a per-k fill occupies -0.20 and
# leaves -0.50 empty; one global mu occupies both k1 states instead.
_CROSSING_EPS = [
    np.array([-0.20, 0.30, 0.90]),
    np.array([-0.60, -0.50, 0.80]),
]
_CROSSING_WEIGHTS = [0.5, 0.5]


def _band_edges_from_occupations(eps_per_k, occ_per_k, max_occ=2.0):
    """VBM / CBM / indirect gap exactly as ``_band_summary`` reads them.

    ``periodic_runner._band_summary`` calls a state occupied when its
    occupation exceeds 1e-8 and virtual when it is below ``max_occ - 1e-8``,
    takes the VBM as the maximum occupied energy over the whole mesh and the
    CBM as the minimum virtual one, and reports ``CBM - VBM`` unclamped.
    Reproduced here so this test measures the reported observable.
    """
    occupied = [
        float(e)
        for eps, occ in zip(eps_per_k, occ_per_k)
        for e, o in zip(np.asarray(eps), np.asarray(occ))
        if o > 1e-8
    ]
    virtual = [
        float(e)
        for eps, occ in zip(eps_per_k, occ_per_k)
        for e, o in zip(np.asarray(eps), np.asarray(occ))
        if o < max_occ - 1e-8
    ]
    vbm = max(occupied)
    cbm = min(virtual)
    return vbm, cbm, cbm - vbm


def test_apply_smearing_t_zero_multi_k_uses_one_global_fermi_level():
    """#85: the occupied set follows the global energy ordering.

    Parent behaviour: ``[[2, 0, 0], [2, 0, 0]]`` -- the lowest band filled at
    each k independently, which occupies -0.20 at k0 while -0.50 at k1 stays
    empty.
    """
    res = vq.apply_smearing(
        _CROSSING_EPS,
        weights=_CROSSING_WEIGHTS,
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    )

    np.testing.assert_allclose(res.occupations_per_k[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(res.occupations_per_k[1], [2.0, 2.0, 0.0])
    # One sharp zero-temperature ensemble: no entropy, no free-energy shift.
    assert res.entropy == 0.0
    assert res.free_energy_correction == 0.0
    # Particle count is conserved against the weighted constraint.
    assert sum(
        w * float(np.sum(occ_k))
        for w, occ_k in zip(_CROSSING_WEIGHTS, res.occupations_per_k)
    ) == pytest.approx(2.0)


def test_apply_smearing_t_zero_multi_k_never_reports_a_negative_indirect_gap():
    """#85's reported symptom, measured on the reporting observable.

    At the parent this returns VBM = -0.200000, CBM = -0.500000 and an
    indirect gap of -0.300000 Ha: an occupied state 0.3 Ha ABOVE the state
    called the conduction-band minimum. The archived production instance is
    Si/def2-SVP/PBE at 6x6x6 -- a -0.080825-Ha indirect gap with exactly 14
    occupied bands forced at every k point.
    """
    res = vq.apply_smearing(
        _CROSSING_EPS,
        weights=_CROSSING_WEIGHTS,
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    )

    vbm, cbm, gap = _band_edges_from_occupations(
        _CROSSING_EPS, res.occupations_per_k
    )
    assert vbm == pytest.approx(-0.50)
    assert cbm == pytest.approx(-0.20)
    assert gap == pytest.approx(0.30)
    # The classification is self-consistent: every occupied state lies at or
    # below every virtual one, which is what makes the gap non-negative.
    assert gap >= 0.0


def test_apply_smearing_t_zero_multi_k_drops_the_equal_band_count_signature():
    """#85: a band-overlap mesh must not carry the same count at every k.

    "Exactly N occupied bands at every k point" is the fingerprint of the
    per-k fill. Under one global Fermi level the count is free to differ by k,
    which is the whole point.
    """
    res = vq.apply_smearing(
        _CROSSING_EPS,
        weights=_CROSSING_WEIGHTS,
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    )

    counts = [int(np.count_nonzero(o > 1e-8)) for o in res.occupations_per_k]
    assert counts == [0, 2]
    assert len(set(counts)) > 1
    assert not _is_per_k_integer_aufbau(res.occupations_per_k, n_occ_each=1)


def test_apply_smearing_t_zero_gapped_mesh_is_unchanged_by_the_global_fill():
    """A mesh gapped at every k keeps the legacy arrays and mu, bit-for-bit.

    This is what bounds the blast radius: the global fill differs from the
    per-k fill only where the per-k fill was wrong. When the electron count
    lands exactly on a group boundary, the occupied set and the midgap mu are
    identical, so no gapped-insulator result moves.
    """
    gapped = [
        np.array([-0.50, -0.30, 0.10, 0.20]),
        np.array([-0.40, -0.20, 0.15, 0.25]),
    ]
    res = vq.apply_smearing(
        gapped,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    )
    legacy = sm.aufbau_occupations_per_k(gapped, 1, occ_value=2.0)

    for got, want in zip(res.occupations_per_k, legacy):
        assert np.array_equal(np.asarray(got), np.asarray(want))
    # Exactly the midgap value the historical branch returned, not merely
    # close to it.
    assert res.mu == 0.5 * (-0.40 + -0.30)


def test_apply_smearing_t_zero_single_spectrum_keeps_the_historical_path():
    """Negative control: the same route with the global fill inactive.

    One spectrum has nothing for a shared mu to reorder, so ``apply_smearing``
    keeps the historical hard-Aufbau branch. Same entry point, same T = 0
    argument shape, feature not engaged -- the result must be bit-identical to
    ``_aufbau_with_midgap_mu`` on the same input, including for the
    band-crossing spectrum that moves the multi-k case above.
    """
    for eps in (_CROSSING_EPS[0], _CROSSING_EPS[1]):
        res = vq.apply_smearing(
            [eps],
            weights=[1.0],
            n_electrons_per_cell=2.0,
            n_occ_each=1,
            smearing=None,
        )
        legacy_occ, legacy_mu = _aufbau_with_midgap_mu([eps], 1)

        assert np.array_equal(
            np.asarray(res.occupations_per_k[0]), np.asarray(legacy_occ[0])
        )
        assert res.mu == legacy_mu
        assert res.entropy == 0.0


def test_shared_dispatcher_and_gdf_route_agree_at_t_zero():
    """The two T = 0 implementations must now be one contract.

    ``periodic_k_gdf._occupations_per_k`` went to one global Fermi level in
    60104fc01; the shared dispatcher every other periodic driver calls did
    not, so the same mesh occupied differently depending on which backend ran
    it. Pin that the disagreement is gone.
    """
    shared_occ, shared_mu, shared_entropy = sm.closed_shell_periodic_occupations(
        _CROSSING_EPS,
        _CROSSING_WEIGHTS,
        2.0,
        1,
        0.0,
    )
    gdf_occ, gdf_mu, gdf_entropy = _gdf_occupations_per_k(
        _CROSSING_EPS,
        np.asarray(_CROSSING_WEIGHTS),
        n_elec_per_cell=2,
        smearing_T=0.0,
        n_occ_each=1,
    )

    for shared_k, gdf_k in zip(shared_occ, gdf_occ):
        assert np.array_equal(np.asarray(shared_k), np.asarray(gdf_k))
    assert shared_mu == gdf_mu
    assert shared_entropy == gdf_entropy == 0.0


def test_apply_smearing_t_zero_multi_k_shares_a_degenerate_fermi_group():
    """A partially filled group at mu is split evenly, not by array order.

    The band-overlap ensemble is the sharp T -> 0 limit of the Fermi function,
    so a degenerate group cut by the particle constraint takes one shared
    fractional occupation. Mesh ordering must not decide which member wins.
    """
    degenerate = [np.array([-1.0, 0.0]), np.array([0.0, 1.0])]
    res = vq.apply_smearing(
        degenerate,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    )

    np.testing.assert_allclose(res.occupations_per_k[0], [2.0, 1.0])
    np.testing.assert_allclose(res.occupations_per_k[1], [1.0, 0.0])
    assert res.mu == pytest.approx(0.0)
    assert res.entropy == 0.0


# ---------------------------------------------------------------------------
# The fixed-occupied-subspace guard (#85, density half tracked as #509)
#
# The Ewald-3D and BIPOLE multi-k routes build their T = 0 density from a
# single ``C[:, :n_occ]`` slice at every k. That slice cannot represent a
# band-overlap fill, so pairing it with the global occupations would converge
# one density and report the band edges of another. Those sites refuse instead.
# ---------------------------------------------------------------------------


def test_per_k_integer_predicate_accepts_only_the_fixed_subspace_pattern():
    assert sm.occupations_are_per_k_integer_aufbau(
        [np.array([2.0, 0.0]), np.array([2.0, 0.0])], 1
    )
    # Different occupied count by k -- the band-overlap fill.
    assert not sm.occupations_are_per_k_integer_aufbau(
        [np.array([0.0, 0.0]), np.array([2.0, 2.0])], 1
    )
    # Shared fractional occupation across a degenerate Fermi group.
    assert not sm.occupations_are_per_k_integer_aufbau(
        [np.array([2.0, 1.0]), np.array([1.0, 0.0])], 1
    )
    # Right count, wrong bands: occupation must sit on the LOWEST states.
    assert not sm.occupations_are_per_k_integer_aufbau(
        [np.array([0.0, 2.0]), np.array([2.0, 0.0])], 1
    )
    # occ_value is explicit, not inferred: a mesh whose only occupied states
    # are a HALF-filled degenerate group presents 1.0 everywhere, and must not
    # be mistaken for a full closed-shell subspace the integer builder could
    # reproduce.
    assert not sm.occupations_are_per_k_integer_aufbau(
        [np.array([1.0, 0.0]), np.array([1.0, 0.0])], 1
    )
    # The same arrays ARE the fixed subspace for a single spin channel.
    assert sm.occupations_are_per_k_integer_aufbau(
        [np.array([1.0, 0.0]), np.array([1.0, 0.0])], 1, occ_value=1.0
    )


def test_fixed_subspace_guard_passes_a_gapped_mesh_silently():
    """Negative control: the same guard, on occupations the feature leaves
    alone. A gapped mesh still returns the legacy pattern, so nothing raises
    and no route changes behaviour."""
    gapped = [
        np.array([-0.50, -0.30, 0.10, 0.20]),
        np.array([-0.40, -0.20, 0.15, 0.25]),
    ]
    res = vq.apply_smearing(
        gapped,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    )
    # Returns None; the assertion is that it does not raise.
    assert (
        sm.require_fixed_occupied_subspace(
            res.occupations_per_k, 1, entry="test"
        )
        is None
    )


def test_fixed_subspace_guard_refuses_a_band_overlap_fill():
    """A band-overlap T = 0 fill must fail closed, naming what to do."""
    res = vq.apply_smearing(
        _CROSSING_EPS,
        weights=_CROSSING_WEIGHTS,
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    )
    with pytest.raises(NotImplementedError) as excinfo:
        sm.require_fixed_occupied_subspace(
            res.occupations_per_k, 1, entry="run_rhf_periodic_multi_k_ewald3d"
        )

    message = str(excinfo.value)
    assert "run_rhf_periodic_multi_k_ewald3d" in message
    # The message must carry the evidence and the way out, not just a refusal.
    assert "[0, 2]" in message
    assert "different number of bands" in message
    assert "smearing_temperature" in message
    assert "#509" in message


def test_fixed_subspace_guard_names_the_degenerate_shape_separately():
    """The two ways a global fill escapes the fixed subspace read differently.

    A band-overlap mesh occupies a different COUNT by k; a degenerate group cut
    by the particle constraint keeps the count and goes FRACTIONAL. Reporting
    the second as the first would send a reader looking for the wrong thing.
    """
    degenerate = [np.array([-1.0, 0.0]), np.array([0.0, 1.0])]
    res = vq.apply_smearing(
        degenerate,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    )
    with pytest.raises(NotImplementedError) as excinfo:
        sm.require_fixed_occupied_subspace(
            res.occupations_per_k, 1, entry="run_pbc_bipole_rks"
        )

    message = str(excinfo.value)
    assert "fractional occupation across a degenerate group" in message
    assert "different number of bands" not in message


# ---------------------------------------------------------------------------
# #725: a caller-chosen (MOM) occupied subspace survives the T = 0 fill
# ---------------------------------------------------------------------------

# The verifier's reproducer from #725: two k points after a MOM column
# permutation, so each eps(k) lists the MOM-selected occupied state FIRST and
# is not in energy order. Equal weights, 2 electrons per cell, n_occ_each = 1.
_MOM_PERMUTED_EPS = [np.array([0.2, -0.5]), np.array([0.3, -0.4])]
_MOM_WEIGHTS = [0.5, 0.5]


def test_global_fill_undoes_a_mom_permutation_and_the_guard_names_it():
    """Documents the defect shape (#725) rather than the fix: handed a
    MOM-permuted spectrum, the energy-ordered fill re-picks the lower column
    at every k, and the guard must describe that as an out-of-block
    selection with equal counts, not as a per-k count mismatch."""
    res = vq.apply_smearing(
        _MOM_PERMUTED_EPS,
        weights=_MOM_WEIGHTS,
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    )
    assert [o.tolist() for o in res.occupations_per_k] == [[0.0, 2.0], [0.0, 2.0]]
    with pytest.raises(NotImplementedError) as excinfo:
        sm.require_fixed_occupied_subspace(
            res.occupations_per_k, 1, entry="run_rhf_periodic_multi_k_ewald3d"
        )
    message = str(excinfo.value)
    assert "outside the leading n_occ_each entries" in message
    assert "[1, 1]" in message
    assert "different number of bands" not in message
    assert "fractional occupation" not in message


def test_fixed_occupied_subspace_fill_is_positional_and_passes_the_guard():
    """With ``fixed_occupied_subspace=True`` the T = 0 fill occupies the
    leading ``n_occ_each`` entries at every k (Gilbert, Besley, Gill 2008:
    MOM occupies the orbitals of largest overlap, not of lowest energy), mu
    is the midpoint of the highest occupied / lowest empty energy under that
    partition, no frontier cut is reported, and the fixed-slice guard
    accepts the result."""
    res = vq.apply_smearing(
        _MOM_PERMUTED_EPS,
        weights=_MOM_WEIGHTS,
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
        fixed_occupied_subspace=True,
    )
    assert [o.tolist() for o in res.occupations_per_k] == [[2.0, 0.0], [2.0, 0.0]]
    assert res.mu == pytest.approx(0.5 * (0.3 + (-0.5)))
    assert res.entropy == 0.0
    assert res.free_energy_correction == 0.0
    assert res.frontier_cut_unresolved is False
    assert np.isnan(res.frontier_gap)
    assert sm.occupations_are_per_k_integer_aufbau(res.occupations_per_k, 1)
    assert (
        sm.require_fixed_occupied_subspace(res.occupations_per_k, 1, entry="x")
        is None
    )
    # The driver-facing tuple wrapper forwards the flag unchanged.
    occ, mu, entropy = sm.closed_shell_periodic_occupations(
        _MOM_PERMUTED_EPS, _MOM_WEIGHTS, 2.0, 1, 0.0, fixed_occupied_subspace=True
    )
    assert [o.tolist() for o in occ] == [[2.0, 0.0], [2.0, 0.0]]
    assert mu == pytest.approx(res.mu)
    assert entropy == 0.0


def test_fixed_occupied_subspace_is_a_zero_temperature_contract():
    """MOM holds a sharp subspace; a smeared Fermi function has none."""
    with pytest.raises(ValueError, match="zero-temperature"):
        vq.apply_smearing(
            _MOM_PERMUTED_EPS,
            weights=_MOM_WEIGHTS,
            n_electrons_per_cell=2.0,
            n_occ_each=1,
            smearing=0.01,
            fixed_occupied_subspace=True,
        )


def test_fixed_occupied_subspace_default_is_byte_neutral_for_aufbau_meshes():
    """Negative control: on an energy-ordered gapped mesh the positional and
    the global fill agree exactly, and the default flag keeps every existing
    caller on the global fill (the crossing mesh still reselects)."""
    gapped = [
        np.array([-0.50, -0.30, 0.10, 0.20]),
        np.array([-0.40, -0.20, 0.15, 0.25]),
    ]
    common = dict(weights=[0.5, 0.5], n_electrons_per_cell=4.0, n_occ_each=2)
    a = vq.apply_smearing(gapped, smearing=None, **common)
    b = vq.apply_smearing(
        gapped, smearing=None, fixed_occupied_subspace=True, **common
    )
    for x, y in zip(a.occupations_per_k, b.occupations_per_k):
        np.testing.assert_array_equal(x, y)
    assert a.mu == pytest.approx(b.mu)
    crossing = vq.apply_smearing(
        _CROSSING_EPS,
        weights=_CROSSING_WEIGHTS,
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    )
    assert not sm.occupations_are_per_k_integer_aufbau(
        crossing.occupations_per_k, 1
    )
