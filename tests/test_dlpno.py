"""Unit tests for the DLPNO building blocks that remain after M3b.

Pair classification, PAO/domain machinery, and threshold presets. The
pre-M2 placeholder modules (pno/integrals/provider/solver/df_bridge/
pair_pair/triples) were retired in M3b — their physics lives on,
validated, in dlpno.mp2 and dlpno.ccsd (see tests/test_dlpno_mp2.py and
tests/test_dlpno_ccsd.py; remaining production gates live in
handovers/HANDOVER_GATED_ITEMS.md).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from vibeqc.dlpno import (
    DLPNO_DEFAULTS,
    DLPNO_LOOSE,
    DLPNO_TIGHT,
    DLPNOThresholdProvenance,
    DLPNOThresholds,
    apply_dlpno_thresholds,
    describe_dlpno_thresholds,
    options_from_dlpno_thresholds,
    resolve_dlpno_thresholds,
)
from vibeqc.dlpno.ccsd import DLPNOCCSDPilotOptions
from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions
from vibeqc.dlpno.mp2 import DLPNOMP2Options
from vibeqc.dlpno.pairs import (
    PairClass,
    PairInfo,
    classify_pairs,
    compute_orbital_centroids,
    compute_pair_distances,
    estimate_pair_energy_dipole,
    estimate_pair_energy_overlap,
)
from vibeqc.dlpno.pao import (
    PAODomain,
    build_pao_coeffs,
    build_projection_matrix,
    select_domain_atoms_mulliken,
)
from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions
from vibeqc.dlpno.uccsd_local_solver import LocalUCCSDOptions
from vibeqc.dlpno.ump2 import DLPNOUMP2Options


class TestOrbitalCentroids:
    """Compute centroids for simple systems."""

    def test_two_atom_system(self):
        """H2-like: two s-orbitals, centroids at atom positions."""
        nbf = 2
        C_occ = np.eye(2)  # one orbital per basis function
        S = np.eye(2)
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])

        centroids = compute_orbital_centroids(C_occ, coords, S)
        assert centroids.shape == (2, 3)
        np.testing.assert_allclose(centroids[0], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(centroids[1], [0.0, 0.0, 1.4], atol=1e-12)

    def test_distance_matrix(self):
        """Distance between two centroids along z."""
        centroids = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        dist = compute_pair_distances(centroids)
        assert dist.shape == (2, 2)
        assert dist[0, 0] == pytest.approx(0.0)
        assert dist[0, 1] == pytest.approx(2.0)
        assert dist[1, 0] == pytest.approx(2.0)


class TestPairClassification:
    """End-to-end pair classification."""

    def test_classify_two_orbitals(self):
        """Two occupied orbitals — one pair."""
        nbf, nocc = 4, 2
        C_occ = np.zeros((nbf, nocc))
        C_occ[0, 0] = 1.0  # orbital on atom 0
        C_occ[2, 1] = 1.0  # orbital on atom 1
        S = np.eye(nbf)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],  # AO 0
                [0.0, 0.0, 0.0],  # AO 1
                [0.0, 0.0, 1.4],  # AO 2
                [0.0, 0.0, 1.4],  # AO 3
            ]
        )

        result = classify_pairs(C_occ, S, coords, tcut_pairs=1e-4)
        assert result.n_occ == 2
        assert result.n_total == 1  # one pair

    def test_classify_far_apart_is_distant(self):
        """Orbitals 10 Bohr apart -> distant pair."""
        nbf, nocc = 4, 2
        C_occ = np.zeros((nbf, nocc))
        C_occ[0, 0] = 1.0
        C_occ[2, 1] = 1.0
        S = np.eye(nbf)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 10.0],
            ]
        )

        result = classify_pairs(C_occ, S, coords, tcut_pairs=1e-4)
        assert result.n_total == 1
        assert result.n_distant == 1
        assert result.n_strong == 0
        assert result.all_pairs[0].pair_class == PairClass.DISTANT

    def test_classify_close_is_strong(self):
        """Orbitals overlapping -> strong pair."""
        nbf, nocc = 2, 2
        C_occ = np.array([[1.0, 0.5], [0.0, 0.866]])  # overlap
        S = np.eye(nbf)
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])

        result = classify_pairs(C_occ, S, coords, tcut_pairs=1e-4)
        assert result.n_total == 1
        # Should be classified based on overlap metric
        assert result.all_pairs[0].pair_class in (PairClass.STRONG, PairClass.WEAK)

    def test_overlap_metric_nonnegative(self):
        """Differential overlap should be nonnegative."""
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        S = np.eye(2)
        m = estimate_pair_energy_overlap(C, S, 0, 1)
        assert m >= 0.0


# ---------------------------------------------------------------------------
# PAO construction tests
# ---------------------------------------------------------------------------


class TestPAOConstruction:
    """Test the virtual projector and PAO building."""

    def test_projector_idempotent(self):
        """Q² = Q for the virtual projector."""
        nbf, nocc = 4, 2
        C_occ = np.eye(nbf, nocc)  # first two columns
        S = np.eye(nbf)
        Q = build_projection_matrix(C_occ, S)
        # Check idempotence: Q @ Q = Q
        np.testing.assert_allclose(Q @ Q, Q, atol=1e-12)

    def test_projector_blocks_occupied(self):
        """Q projects out the occupied space."""
        nbf, nocc = 6, 2
        C_occ = np.eye(nbf, nocc)
        S = np.eye(nbf)
        Q = build_projection_matrix(C_occ, S)

        # C_occ should be in the nullspace of Q
        proj = Q @ C_occ
        np.testing.assert_allclose(proj, 0.0, atol=1e-12)

    def test_build_pao_coeffs_simple(self):
        """Build PAOs for a 4-AO system with 2 occupied."""
        nbf, nocc = 6, 2
        C_occ = np.eye(nbf, nocc)
        S = np.eye(nbf)
        Q = build_projection_matrix(C_occ, S)

        # Domain: all AOs
        mask = np.ones(nbf, dtype=bool)
        C_pao, n_pao = build_pao_coeffs(Q, mask, S, lindep_thresh=1e-12)
        assert n_pao == nbf - nocc  # 4 PAOs
        assert C_pao.shape == (nbf, n_pao)

    def test_build_pao_coeffs_subset(self):
        """Build PAOs for a subset of AOs."""
        nbf, nocc = 8, 2
        C_occ = np.eye(nbf, nocc)
        S = np.eye(nbf)
        Q = build_projection_matrix(C_occ, S)

        # Domain: first 4 AOs
        mask = np.zeros(nbf, dtype=bool)
        mask[:4] = True
        C_pao, n_pao = build_pao_coeffs(Q, mask, S, lindep_thresh=1e-12)
        assert n_pao >= 1
        assert C_pao.shape == (4, n_pao)


# ---------------------------------------------------------------------------
# PNO construction tests
# ---------------------------------------------------------------------------



# Published DLPNO threshold sets.  Each row is (TCutPairs, TCutPNO, TCutMKN)
# exactly as tabulated in Liakos, Sparta, Kesharwani, Martin & Neese,
# J. Chem. Theory Comput. 11, 1525 (2015), doi:10.1021/ct501129s, Table 1
# ("Definition of the Three Default Thresholds Controlling the DLPNO"):
#
#     LoosePNO    1e-3   1e-6      1e-3    rapid estimates
#     NormalPNO   1e-4   3.33e-7   1e-3    general thermochemistry / kinetics
#     TightPNO    1e-5   1e-7      1e-4    noncovalent interactions,
#                                          conformational equilibria
#
# NormalPNO coincides with the original DLPNO-CCSD defaults of Riplinger &
# Neese, J. Chem. Phys. 138, 034106 (2013), doi:10.1063/1.4773581 (stated
# there in the Fig. 9 caption: "Default thresholds were used
# (TCutPNO = 3.33e-7; TCutPairs = 1e-4 Eh, TCutMKN = 1e-3)").
#
# These triples are the contract of the three exported presets.  A preset
# that carries a published name must carry the published numbers: see
# GitLab issue #416, where a "TightPNO"-labelled S22 wave in fact ran
# TCutPairs 5e-5 / TCutPNO 1e-8 / TCutMKN 1e-3 — zero of three matching.
PUBLISHED_LOOSEPNO = (1e-3, 1e-6, 1e-3)
PUBLISHED_NORMALPNO = (1e-4, 3.33e-7, 1e-3)
PUBLISHED_TIGHTPNO = (1e-5, 1e-7, 1e-4)


class TestThresholds:
    """DLPNOThresholds configuration and published-preset fidelity."""

    @staticmethod
    def _triple(t):
        """(TCutPairs, TCutPNO, TCutMKN) in the publications' order."""
        return (t.tcut_pairs, t.tcut_pno, t.tcut_mkn)

    def test_defaults_are_published_normalpno(self):
        # Riplinger 2013 / Liakos 2015 Table 1 "NormalPNO".
        assert self._triple(DLPNO_DEFAULTS) == PUBLISHED_NORMALPNO

    def test_tight_is_published_tightpno(self):
        # Liakos 2015 Table 1 "TightPNO" — the set the paper recommends
        # for noncovalent interactions.  Regression for issue #416.
        assert self._triple(DLPNO_TIGHT) == PUBLISHED_TIGHTPNO

    def test_loose_is_published_loosepno(self):
        # Liakos 2015 Table 1 "LoosePNO".
        assert self._triple(DLPNO_LOOSE) == PUBLISHED_LOOSEPNO

    def test_preset_ordering_is_monotone(self):
        # Tightening a preset may never loosen any of the three knobs.
        for tight, loose in ((DLPNO_TIGHT, DLPNO_DEFAULTS),
                             (DLPNO_DEFAULTS, DLPNO_LOOSE)):
            for a, b in zip(self._triple(tight), self._triple(loose)):
                assert a <= b

    def test_weak_pair_companions_track_their_strong_partner(self):
        # Not fixed by Table 1 — vibe-qc's own convention, pinned so it
        # cannot drift silently: the weak-pair PNO cutoff is one decade
        # looser than the strong-pair one, and the weak-pair energy
        # threshold equals the strong-pair one.
        for t in (DLPNO_DEFAULTS, DLPNO_TIGHT, DLPNO_LOOSE):
            assert t.tcut_pno_weak == pytest.approx(10.0 * t.tcut_pno)
            assert t.tcut_pairs_weak == t.tcut_pairs

    def test_custom(self):
        t = DLPNOThresholds(tcut_pno=1e-9, tcut_mkn=5e-4)
        assert t.tcut_pno == 1e-9
        assert t.tcut_mkn == 5e-4
        # Unchanged defaults
        assert t.tcut_c == 1e-3

    @pytest.mark.parametrize(
        "options_type,supported",
        [
            (DLPNOMP2Options, ("tcut_pairs", "tcut_pno", "tcut_mkn")),
            (DLPNOUMP2Options, ("tcut_pairs", "tcut_pno")),
            (DLPNOCCSDPilotOptions, ("tcut_pno", "tcut_mkn")),
            (DLPNOUCCSDPilotOptions, ("tcut_pno",)),
            (LocalCCSDOptions, ("tcut_pairs", "tcut_pno", "tcut_mkn")),
            (LocalUCCSDOptions, ("tcut_pairs", "tcut_pno", "tcut_mkn")),
        ],
    )
    def test_every_solver_defaults_to_its_normalpno_supported_subset(
        self, options_type, supported
    ):
        """Every moved default is the published NormalPNO value (#448).

        Missing algorithmic consumers remain absent rather than becoming
        decorative metadata; their supported-subset status is tested through
        the shared resolver separately.
        """
        options = options_type()
        published = {
            "tcut_pairs": PUBLISHED_NORMALPNO[0],
            "tcut_pno": PUBLISHED_NORMALPNO[1],
            "tcut_mkn": PUBLISHED_NORMALPNO[2],
        }
        for name in supported:
            assert getattr(options, name) == pytest.approx(published[name])

    @pytest.mark.parametrize(
        "spelling,canonical,expected",
        [
            ("loose", "LoosePNO", PUBLISHED_LOOSEPNO),
            ("Loose_PNO", "LoosePNO", PUBLISHED_LOOSEPNO),
            ("normal", "NormalPNO", PUBLISHED_NORMALPNO),
            ("default", "NormalPNO", PUBLISHED_NORMALPNO),
            ("Tight-PNO", "TightPNO", PUBLISHED_TIGHTPNO),
        ],
    )
    def test_named_resolver_is_case_and_separator_insensitive(
        self, spelling, canonical, expected
    ):
        resolved = resolve_dlpno_thresholds(spelling)
        assert resolved.name == canonical
        assert self._triple(resolved) == expected

    def test_named_resolver_rejects_an_unknown_name(self):
        with pytest.raises(ValueError, match="unknown DLPNO threshold preset"):
            resolve_dlpno_thresholds("NearlyTightPNO")

    def test_named_record_rejects_false_weak_pair_companions(self):
        with pytest.raises(ValueError, match="derived weak-pair companions"):
            DLPNOThresholds(
                tcut_pairs=PUBLISHED_TIGHTPNO[0],
                tcut_pno=PUBLISHED_TIGHTPNO[1],
                tcut_mkn=PUBLISHED_TIGHTPNO[2],
                tcut_pairs_weak=9e-4,
                tcut_pno_weak=8e-4,
                name="TightPNO",
            )

    @pytest.mark.parametrize(
        "options,expected_applied,expected_unsupported,expected_inactive",
        [
            (
                DLPNOMP2Options(),
                {"tcut_pairs", "tcut_pno", "tcut_mkn"},
                set(),
                {"tcut_pairs_weak", "tcut_pno_weak"},
            ),
            (
                DLPNOUMP2Options(),
                {"tcut_pno"},
                {"tcut_mkn"},
                {"tcut_pairs"},
            ),
            (
                DLPNOCCSDPilotOptions(),
                {"tcut_pno", "tcut_mkn"},
                {"tcut_pairs"},
                set(),
            ),
            (
                DLPNOUCCSDPilotOptions(),
                {"tcut_pno"},
                {"tcut_pairs", "tcut_mkn"},
                set(),
            ),
            (
                LocalCCSDOptions(),
                {"tcut_pairs", "tcut_pno", "tcut_mkn"},
                set(),
                set(),
            ),
            (
                LocalUCCSDOptions(),
                {"tcut_pairs", "tcut_pno", "tcut_mkn"},
                set(),
                set(),
            ),
        ],
    )
    def test_apply_reports_each_solver_capability_truthfully(
        self,
        options,
        expected_applied,
        expected_unsupported,
        expected_inactive,
    ):
        provenance = apply_dlpno_thresholds(options, "tight")
        assert provenance.preset == "TightPNO"
        assert dict(provenance.requested) == {
            "tcut_pairs": PUBLISHED_TIGHTPNO[0],
            "tcut_pno": PUBLISHED_TIGHTPNO[1],
            "tcut_mkn": PUBLISHED_TIGHTPNO[2],
        }
        assert set(dict(provenance.applied)) >= expected_applied
        assert set(provenance.unsupported) == expected_unsupported
        assert set(provenance.inactive) == expected_inactive
        for name in expected_applied:
            assert getattr(options, name) == pytest.approx(
                dict(provenance.applied)[name]
            )
        for name in expected_unsupported:
            assert not hasattr(options, name)

    def test_ump2_pair_cutoff_is_applied_only_when_localised(self):
        canonical = DLPNOUMP2Options(localise="none", tcut_pairs=9e-4)
        provenance = apply_dlpno_thresholds(canonical, "tight")
        assert canonical.tcut_pairs == pytest.approx(9e-4)
        assert "tcut_pairs" in provenance.inactive
        assert "tcut_pairs" not in dict(provenance.applied)

        local = DLPNOUMP2Options(localise="boys", tcut_pairs=9e-4)
        provenance = apply_dlpno_thresholds(local, "tight")
        assert local.tcut_pairs == pytest.approx(PUBLISHED_TIGHTPNO[0])
        assert "tcut_pairs" not in provenance.inactive
        assert dict(provenance.applied)["tcut_pairs"] == pytest.approx(
            PUBLISHED_TIGHTPNO[0]
        )

    def test_closed_mp2_stores_but_does_not_apply_unreachable_weak_tier(self):
        options = DLPNOMP2Options()
        provenance = apply_dlpno_thresholds(options, "loose")
        assert options.tcut_pairs_weak == pytest.approx(DLPNO_LOOSE.tcut_pairs)
        assert options.tcut_pno_weak == pytest.approx(
            DLPNO_LOOSE.tcut_pno_weak
        )
        assert {
            "tcut_pairs_weak",
            "tcut_pno_weak",
        } == set(provenance.inactive)
        assert not {
            "tcut_pairs_weak",
            "tcut_pno_weak",
        } & set(dict(provenance.applied))

    def test_closed_mp2_custom_weak_interval_is_reported_as_active(self):
        options = DLPNOMP2Options(
            tcut_pairs=1e-6,
            tcut_pairs_weak=1e-4,
            tcut_pno_weak=1e-5,
        )
        provenance = describe_dlpno_thresholds(options)
        assert not {
            "tcut_pairs_weak",
            "tcut_pno_weak",
        } & set(provenance.inactive)
        assert {
            "tcut_pairs_weak",
            "tcut_pno_weak",
        } <= set(dict(provenance.applied))

    def test_factory_applies_after_route_overrides(self):
        options = options_from_dlpno_thresholds(
            DLPNOUMP2Options,
            "loose",
            localise="boys",
            max_iter=17,
        )
        assert options.localise == "boys"
        assert options.max_iter == 17
        assert options.tcut_pairs == pytest.approx(PUBLISHED_LOOSEPNO[0])
        assert options.tcut_pno == pytest.approx(PUBLISHED_LOOSEPNO[1])
        assert not hasattr(options, "tcut_mkn")

    def test_factory_preserves_an_explicit_threshold_exception(self):
        options = options_from_dlpno_thresholds(
            DLPNOMP2Options,
            "tight",
            tcut_pno=2.5e-8,
        )
        assert options.tcut_pairs == pytest.approx(PUBLISHED_TIGHTPNO[0])
        assert options.tcut_pno == pytest.approx(2.5e-8)
        assert describe_dlpno_thresholds(options).preset == "custom"

    def test_factory_preserves_an_inactive_threshold_exception(self):
        options = options_from_dlpno_thresholds(
            DLPNOUMP2Options,
            "tight",
            localise="none",
            tcut_pairs=9e-4,
        )
        provenance = describe_dlpno_thresholds(options)
        assert provenance.preset == "custom"
        assert dict(provenance.requested) == {
            "tcut_pairs": pytest.approx(9e-4),
            "tcut_pno": pytest.approx(PUBLISHED_TIGHTPNO[1]),
            "tcut_mkn": pytest.approx(PUBLISHED_TIGHTPNO[2]),
        }
        assert dict(provenance.applied) == {
            "tcut_pno": pytest.approx(PUBLISHED_TIGHTPNO[1]),
        }
        assert provenance.inactive == ("tcut_pairs",)

    def test_bare_options_disclose_an_inactive_threshold_exception(self):
        options = DLPNOUMP2Options(
            localise="none",
            tcut_pairs=9e-4,
        )
        provenance = describe_dlpno_thresholds(options)
        assert provenance.preset == "custom"
        assert dict(provenance.requested) == {
            "tcut_pairs": pytest.approx(9e-4),
            "tcut_pno": pytest.approx(PUBLISHED_NORMALPNO[1]),
        }
        assert dict(provenance.applied) == {
            "tcut_pno": pytest.approx(PUBLISHED_NORMALPNO[1]),
        }
        assert provenance.inactive == ("tcut_pairs",)
        assert provenance.unsupported == ("tcut_mkn",)

    def test_factory_preserves_custom_full_request_on_partial_route(self):
        custom = DLPNOThresholds(
            tcut_pairs=7e-4,
            tcut_pno=PUBLISHED_NORMALPNO[1],
            tcut_mkn=8e-4,
        )
        options = options_from_dlpno_thresholds(
            DLPNOUCCSDPilotOptions,
            custom,
        )
        provenance = describe_dlpno_thresholds(options)
        assert provenance.preset == "custom"
        assert dict(provenance.requested) == {
            "tcut_pairs": pytest.approx(7e-4),
            "tcut_pno": pytest.approx(PUBLISHED_NORMALPNO[1]),
            "tcut_mkn": pytest.approx(8e-4),
        }
        assert dict(provenance.applied) == {
            "tcut_pno": pytest.approx(PUBLISHED_NORMALPNO[1]),
        }
        assert set(provenance.unsupported) == {"tcut_pairs", "tcut_mkn"}

    def test_describe_tracks_mutation_after_applying_a_named_request(self):
        options = DLPNOUMP2Options(localise="none")
        apply_dlpno_thresholds(options, "tight")
        options.tcut_pairs = 9e-4

        provenance = describe_dlpno_thresholds(options)
        assert provenance.preset == "custom"
        assert dict(provenance.requested)["tcut_pairs"] == pytest.approx(9e-4)
        assert "tcut_pairs" in provenance.inactive

    def test_describe_does_not_mislabel_newly_active_stored_cutoff(self):
        options = options_from_dlpno_thresholds(
            DLPNOUMP2Options,
            "tight",
            localise="none",
        )
        assert options.tcut_pairs == pytest.approx(PUBLISHED_NORMALPNO[0])

        options.localise = "boys"
        provenance = describe_dlpno_thresholds(options)
        assert provenance.preset == "custom"
        assert dict(provenance.requested)["tcut_pairs"] == pytest.approx(
            PUBLISHED_TIGHTPNO[0]
        )
        assert dict(provenance.applied)["tcut_pairs"] == pytest.approx(
            PUBLISHED_NORMALPNO[0]
        )
        assert provenance.inactive == ()

    def test_describe_recovers_named_request_when_newly_active_value_matches(self):
        options = options_from_dlpno_thresholds(
            DLPNOUMP2Options,
            "tight",
            localise="none",
        )

        options.localise = "boys"
        options.tcut_pairs = PUBLISHED_TIGHTPNO[0]
        provenance = describe_dlpno_thresholds(options)

        assert provenance.preset == "TightPNO"
        assert dict(provenance.requested) == {
            "tcut_pairs": pytest.approx(PUBLISHED_TIGHTPNO[0]),
            "tcut_pno": pytest.approx(PUBLISHED_TIGHTPNO[1]),
            "tcut_mkn": pytest.approx(PUBLISHED_TIGHTPNO[2]),
        }
        assert dict(provenance.applied) == {
            "tcut_pairs": pytest.approx(PUBLISHED_TIGHTPNO[0]),
            "tcut_pno": pytest.approx(PUBLISHED_TIGHTPNO[1]),
        }
        assert provenance.inactive == ()

    @pytest.mark.parametrize(
        "options_type",
        [
            DLPNOMP2Options,
            DLPNOUMP2Options,
            DLPNOCCSDPilotOptions,
            DLPNOUCCSDPilotOptions,
            LocalCCSDOptions,
            LocalUCCSDOptions,
        ],
    )
    def test_describe_attributes_all_six_defaults_to_normalpno(
        self, options_type
    ):
        provenance = describe_dlpno_thresholds(options_type())
        assert provenance.preset == "NormalPNO"

    def test_unmarked_partial_route_does_not_invent_a_named_request(self):
        options = DLPNOUCCSDPilotOptions(
            tcut_pno=PUBLISHED_TIGHTPNO[1],
        )
        provenance = describe_dlpno_thresholds(options)

        assert provenance.preset == "custom"
        assert dict(provenance.requested) == {
            "tcut_pno": pytest.approx(PUBLISHED_TIGHTPNO[1]),
        }
        assert dict(provenance.applied) == {
            "tcut_pno": pytest.approx(PUBLISHED_TIGHTPNO[1]),
        }
        assert set(provenance.unsupported) == {"tcut_pairs", "tcut_mkn"}

    def test_local_uccsd_solver_specific_cutoffs_are_machine_disclosed(self):
        from vibeqc.runner import (
            _dlpno_manifest_fields,
            _dlpno_threshold_settings,
        )

        options = LocalUCCSDOptions(
            tcut_pno_singles=2e-8,
            compute_triples=True,
            tcut_tno=4e-7,
        )
        provenance = describe_dlpno_thresholds(options)

        settings = _dlpno_threshold_settings(options, provenance)
        assert settings["tcut_pno_singles"] == pytest.approx(2e-8)
        assert settings["tcut_tno"] == pytest.approx(4e-7)
        manifest = _dlpno_manifest_fields(options, provenance)
        assert manifest["dlpno_tcut_pno_singles"] == pytest.approx(2e-8)
        assert manifest["dlpno_tcut_tno"] == pytest.approx(4e-7)

    @pytest.mark.parametrize(
        "options",
        [
            LocalCCSDOptions(compute_triples=False, tcut_tno=4e-7),
            LocalCCSDOptions(
                compute_triples=True,
                triples_mode="exact",
                tcut_tno=4e-7,
            ),
        ],
    )
    def test_inactive_tno_cutoff_is_not_disclosed_as_effective(self, options):
        from vibeqc.runner import (
            _dlpno_manifest_fields,
            _dlpno_threshold_settings,
        )

        provenance = describe_dlpno_thresholds(options)
        settings = _dlpno_threshold_settings(options, provenance)
        assert "tcut_tno" not in settings
        manifest = _dlpno_manifest_fields(options, provenance)
        assert manifest["dlpno_tcut_tno"] == ""

    def test_describe_does_not_overwrite_custom_options(self):
        options = LocalCCSDOptions(tcut_pno=2.5e-8)
        before = options.tcut_pno
        provenance = describe_dlpno_thresholds(options)
        assert provenance.preset == "custom"
        assert options.tcut_pno == before

    def test_provenance_is_immutable(self):
        provenance = apply_dlpno_thresholds(DLPNOMP2Options(), "normal")
        assert isinstance(provenance, DLPNOThresholdProvenance)
        with pytest.raises(FrozenInstanceError):
            provenance.preset = "custom"

    def test_closed_shell_triples_mode_typo_fails_closed(self):
        with pytest.raises(ValueError, match="unknown triples_mode"):
            LocalCCSDOptions(triples_mode="tx")


# ---------------------------------------------------------------------------
# Integral transformation tests
# ---------------------------------------------------------------------------


class TestPNODensityConventions:
    """`tcut_pno` cuts occupation numbers, so the density defines the recipe.

    Three conventions ship (`vibeqc.dlpno.pno_density`): vibe-qc's historical
    `"legacy"` density of the bare amplitudes, Riplinger and Neese's `"mp2"`
    norm (their Eq. 23; ORCA's `PNONorm MP2Norm` default) and the pre-2013
    LPNO `"iepa"` norm (Neese, Wennmohs and Hansen 2009, Eqs. 18-19; ORCA's
    `PNONorm IEPANorm`). The default is `"legacy"` so no shipped number moves;
    which one *should* ship is GitLab #701.
    """

    @staticmethod
    def _amplitudes(seed, n=7, symmetric=False):
        rng = np.random.default_rng(seed)
        T = 0.05 * rng.standard_normal((n, n))
        return 0.5 * (T + T.T) if symmetric else T

    def test_resolution_accepts_the_orca_spellings(self):
        from vibeqc.dlpno import PNO_NORMS, resolve_pno_norm

        assert PNO_NORMS == ("legacy", "mp2", "iepa")
        assert resolve_pno_norm(None) == "legacy"
        for spelling in ("mp2", "MP2", "MP2Norm", "mp2_norm", "MP2 Norm"):
            assert resolve_pno_norm(spelling) == "mp2"
        for spelling in ("iepa", "IEPANorm", "iepa-norm"):
            assert resolve_pno_norm(spelling) == "iepa"
        with pytest.raises(ValueError, match="unknown pno_norm"):
            resolve_pno_norm("meyer")
        with pytest.raises(TypeError, match="must be a string"):
            resolve_pno_norm(3.33e-7)

    def test_legacy_reproduces_the_historical_density(self):
        """The default must be byte-identical to what the builders did before."""
        from vibeqc.dlpno import pair_density

        for symmetric, delta in ((False, 0.0), (True, 1.0)):
            T = self._amplitudes(11, symmetric=symmetric)
            historical = (T @ T.T + T.T @ T) / (1.0 + delta)
            historical = 0.5 * (historical + historical.T)
            assert np.allclose(
                pair_density(T, delta, "legacy"), historical, rtol=0, atol=0
            )

    def test_mp2_norm_is_the_paper_equation(self):
        """Riplinger and Neese 2013, Eq. 23 and the line under it."""
        from vibeqc.dlpno import pair_density

        for symmetric, delta in ((False, 0.0), (True, 1.0)):
            T = self._amplitudes(12, symmetric=symmetric)
            T_tilde = (4.0 * T - 2.0 * T.T) / (1.0 + delta)
            expected = T_tilde @ T.T + T_tilde.T @ T
            assert np.allclose(pair_density(T, delta, "mp2"), expected)

    def test_iepa_norm_is_the_mp2_norm_times_a_per_pair_scalar(self):
        """LPNO 2009 Eqs. 18-19: prefactor (1+delta)/N_ij, N = 1 + <Tt+ T>."""
        from vibeqc.dlpno import pair_density

        for symmetric, delta in ((False, 0.0), (True, 1.0)):
            T = self._amplitudes(13, symmetric=symmetric)
            T_tilde = (4.0 * T - 2.0 * T.T) / (1.0 + delta)
            n_ij = 1.0 + float(np.sum(T_tilde * T))  # <A B> = sum_pq A_pq B_qp
            d_mp2 = pair_density(T, delta, "mp2")
            assert np.allclose(
                pair_density(T, delta, "iepa"), d_mp2 * (1.0 + delta) / n_ij
            )
            # N_ij is a norm, so it exceeds 1 for any nonzero amplitude. On
            # physical amplitudes it is 1.003-1.007, which is why the two
            # published norms agree so closely and why Riplinger and Neese
            # could switch between them ("insignificantly small, on average
            # 1.4% per PNO included", their Fig. 2). That claim is measured on
            # a real molecule in tests/test_dlpno_ccsd_solver.py; these
            # synthetic amplitudes are orders of magnitude larger.
            assert n_ij > 1.0

    def test_every_convention_is_symmetric_and_positive_semidefinite(self):
        from vibeqc.dlpno import PNO_NORMS, pair_density

        for norm in PNO_NORMS:
            for symmetric, delta in ((False, 0.0), (True, 1.0)):
                D = pair_density(self._amplitudes(14, symmetric=symmetric), delta, norm)
                assert np.allclose(D, D.T)
                assert np.linalg.eigvalsh(D).min() > -1e-13

    def test_mp2_occupations_are_two_to_six_times_the_legacy_ones(self):
        """The exact algebra that makes a threshold rescaling impossible.

        With T = S + A (symmetric plus antisymmetric),
            D_mp2    = [4 S^2 + 12 A^T A] / (1 + delta)
            D_legacy = [2 S^2 +  2 A^T A] / (1 + delta)
        so the ratio is 2 along a purely symmetric direction and 6 along a
        purely antisymmetric one. A diagonal pair has symmetric T and is
        therefore exactly 2. Because the factor varies *within* one pair, no
        scalar `tcut_pno` reproduces the other convention's PNO set.
        """
        from vibeqc.dlpno import pair_density

        for delta in (0.0, 1.0):
            T = self._amplitudes(15, symmetric=(delta == 1.0))
            S = 0.5 * (T + T.T)
            A = 0.5 * (T - T.T)
            d_mp2 = pair_density(T, delta, "mp2")
            d_legacy = pair_density(T, delta, "legacy")
            assert np.allclose(d_mp2, (4 * S @ S + 12 * A.T @ A) / (1.0 + delta))
            assert np.allclose(d_legacy, (2 * S @ S + 2 * A.T @ A) / (1.0 + delta))
            assert np.allclose(d_mp2, 2 * d_legacy + 8 * A.T @ A / (1.0 + delta))
            ratio = np.linalg.eigvalsh(d_mp2) / np.linalg.eigvalsh(d_legacy)
            assert ratio.min() >= 2.0 - 1e-9
            assert ratio.max() <= 6.0 + 1e-9
            if delta == 1.0:  # diagonal pair: T symmetric, A = 0
                assert np.allclose(d_mp2, 2.0 * d_legacy)

    def test_mp2_norm_retains_at_least_as_many_pnos(self):
        """Same amplitudes and threshold: `"legacy"` never keeps more.

        Follows from the 2x-to-6x eigenvalue relation above. The size of the
        gap on real systems (4.6-18.5 % at NormalPNO) is measured in
        tests/test_dlpno_ccsd_solver.py.
        """
        from vibeqc.dlpno import pair_density

        tcut = 3.33e-7
        for seed in (21, 22, 23):
            T = self._amplitudes(seed)
            counts = {
                norm: int(np.sum(np.linalg.eigvalsh(pair_density(T, 0.0, norm)) > tcut))
                for norm in ("legacy", "mp2")
            }
            assert counts["mp2"] >= counts["legacy"]

    def test_route_options_default_to_the_published_density(self):
        """Every closed-shell route defaults to Riplinger and Neese Eq. 23.

        The #65 (old #701) ruling. `tcut_pno` is a threshold on pair-density
        eigenvalues, so a preset value only means what it was calibrated
        against; the published ladder is calibrated against this density.
        Under the previous `"legacy"` default the same nominal threshold kept
        5-19 % fewer PNOs, i.e. our NormalPNO delivered their LoosePNO.
        """
        from vibeqc.dlpno.ccsd import DLPNOCCSDPilotOptions
        from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions
        from vibeqc.dlpno.mp2 import DLPNOMP2Options

        for cls in (LocalCCSDOptions, DLPNOMP2Options, DLPNOCCSDPilotOptions):
            assert cls().pno_norm == "mp2", cls.__name__

    def test_open_shell_routes_do_not_expose_the_choice(self):
        """The 4T - 2T^T adaptation is a closed-shell spin-summation identity.

        ORCA builds the open-shell PNOs from T T^dagger / T^dagger T with no
        such factor, so the open-shell option classes deliberately carry no
        `pno_norm` and are untouched.
        """
        from vibeqc.dlpno.uccsd_local_solver import LocalUCCSDOptions
        from vibeqc.dlpno.ump2 import DLPNOUMP2Options

        for cls in (LocalUCCSDOptions, DLPNOUMP2Options):
            assert not hasattr(cls(), "pno_norm"), cls.__name__
