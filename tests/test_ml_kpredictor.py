"""Tests for the K8-F ML k-spacing predictor (vibeqc.data_library.ml_kpredictor)."""

from __future__ import annotations

import os
import pickle
import tempfile
from unittest import mock

import numpy as np
import pytest
from vibeqc.data_library.ml_kpredictor import (
    _CHARACTER_MAP,
    _FEATURE_NAMES,
    ML_PREDICTOR_ENV_VAR,
    KSpacingPredictor,
    _features_to_array,
    build_ml_predictor,
    is_ml_predictor_enabled,
)

# scikit-learn is an optional [ml] extra (see pyproject.toml).
# KSpacingPredictor lazily imports it; guard tests that need it.
try:
    import sklearn  # noqa: F401

    _has_sklearn = True
except ImportError:
    _has_sklearn = False

_requires_sklearn = pytest.mark.skipif(
    not _has_sklearn,
    reason="scikit-learn not installed — pip install -e '.[ml]'",
)


class TestFeatureExtraction:
    """Unit tests for _features_to_array and feature schema."""

    def test_feature_names_length(self):
        assert len(_FEATURE_NAMES) == 11

    def test_character_map_coverage(self):
        for val in ["metal", "small-gap semiconductor", "insulator", None]:
            assert val in _CHARACTER_MAP

    def test_full_feature_dict_round_trip(self):
        features = {
            "n_atoms": 8,
            "dim": 3,
            "cell_volume_bohr3": 300.0,
            "reciprocal_lengths_bohr": [0.5, 0.5, 0.4],
            "character": "insulator",
            "band_gap_eV": 2.5,
            "n_electrons": 48,
        }
        arr = _features_to_array(features)
        assert arr.shape == (1, 11)
        assert arr.dtype == np.float64
        assert arr[0, 9] == 1.0  # character_insulator
        assert arr[0, 7] == 0.0  # character_metal
        assert arr[0, 8] == 0.0  # character_small_gap

    def test_character_none_treated_as_metal(self):
        features = {
            "n_atoms": 2,
            "dim": 3,
            "cell_volume_bohr3": 100.0,
            "reciprocal_lengths_bohr": [1.0, 1.0, 1.0],
            "character": None,
            "band_gap_eV": None,
            "n_electrons": 10,
        }
        arr = _features_to_array(features)
        assert arr[0, 7] == 1.0  # metal one-hot
        assert arr[0, 8] == 0.0
        assert arr[0, 9] == 0.0
        assert arr[0, 6] == 0.0  # gap unknown -> 0

    def test_2d_system_pads_reciprocal_lengths(self):
        features = {
            "n_atoms": 4,
            "dim": 2,
            "cell_volume_bohr3": 500.0,
            "reciprocal_lengths_bohr": [0.6, 0.7],
            "character": "metal",
            "band_gap_eV": 0.0,
            "n_electrons": 20,
        }
        arr = _features_to_array(features)
        assert arr[0, 2] == 0.6
        assert arr[0, 3] == 0.7
        assert arr[0, 4] == 0.0

    def test_gap_none_becomes_zero(self):
        features = {
            "n_atoms": 1,
            "dim": 3,
            "cell_volume_bohr3": 50.0,
            "reciprocal_lengths_bohr": [1.0, 1.0, 1.0],
            "character": "metal",
            "band_gap_eV": None,
            "n_electrons": 5,
        }
        arr = _features_to_array(features)
        assert arr[0, 6] == 0.0


class TestEnvGate:
    """Tests for is_ml_predictor_enabled()."""

    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert not is_ml_predictor_enabled()

    def test_enabled_with_one(self):
        with mock.patch.dict(os.environ, {ML_PREDICTOR_ENV_VAR: "1"}):
            assert is_ml_predictor_enabled()

    def test_other_values_disabled(self):
        with mock.patch.dict(os.environ, {ML_PREDICTOR_ENV_VAR: "true"}):
            assert not is_ml_predictor_enabled()

    def test_whitespace_handled(self):
        with mock.patch.dict(os.environ, {ML_PREDICTOR_ENV_VAR: " 1 "}):
            assert is_ml_predictor_enabled()

    def test_empty_string_disabled(self):
        with mock.patch.dict(os.environ, {ML_PREDICTOR_ENV_VAR: ""}):
            assert not is_ml_predictor_enabled()


@_requires_sklearn
class TestKSpacingPredictor:
    """Tests for the predictor class."""

    def test_load_bundled_model(self):
        pred = KSpacingPredictor()
        assert not pred._loaded
        pred._ensure_loaded()
        assert pred._loaded
        assert pred._model is not None

    def test_predict_returns_tuple(self):
        pred = KSpacingPredictor()
        features = {
            "n_atoms": 8,
            "dim": 3,
            "cell_volume_bohr3": 300.0,
            "reciprocal_lengths_bohr": [0.5, 0.5, 0.4],
            "character": "insulator",
            "band_gap_eV": 2.5,
            "n_electrons": 48,
        }
        mean, sigma = pred.predict(features)
        assert isinstance(mean, float)
        assert isinstance(sigma, float)

    def test_predict_range_sanity(self):
        pred = KSpacingPredictor()
        metal_features = {
            "n_atoms": 2,
            "dim": 3,
            "cell_volume_bohr3": 80.0,
            "reciprocal_lengths_bohr": [0.8, 0.8, 0.8],
            "character": "metal",
            "band_gap_eV": 0.0,
            "n_electrons": 12,
        }
        mean_m, sigma_m = pred.predict(metal_features)
        assert 0.03 < mean_m < 0.50
        assert sigma_m >= 0.0

        ins_features = {
            "n_atoms": 8,
            "dim": 3,
            "cell_volume_bohr3": 300.0,
            "reciprocal_lengths_bohr": [0.5, 0.5, 0.4],
            "character": "insulator",
            "band_gap_eV": 5.0,
            "n_electrons": 64,
        }
        mean_i, sigma_i = pred.predict(ins_features)
        assert 0.05 < mean_i < 0.80
        assert sigma_i >= 0.0

    def test_metal_denser_than_insulator(self):
        pred = KSpacingPredictor()
        base = {
            "n_atoms": 4,
            "dim": 3,
            "cell_volume_bohr3": 200.0,
            "reciprocal_lengths_bohr": [0.5, 0.5, 0.6],
            "n_electrons": 24,
        }
        dk_metal, _ = pred.predict({**base, "character": "metal", "band_gap_eV": 0.0})
        dk_ins, _ = pred.predict({**base, "character": "insulator", "band_gap_eV": 4.0})
        assert dk_metal <= dk_ins * 1.5

    def test_larger_cell_coarser_k(self):
        pred = KSpacingPredictor()
        small = {
            "n_atoms": 2,
            "dim": 3,
            "cell_volume_bohr3": 80.0,
            "reciprocal_lengths_bohr": [0.8, 0.8, 0.8],
            "character": "insulator",
            "band_gap_eV": 3.0,
            "n_electrons": 16,
        }
        large = {
            "n_atoms": 2,
            "dim": 3,
            "cell_volume_bohr3": 800.0,
            "reciprocal_lengths_bohr": [0.3, 0.3, 0.3],
            "character": "insulator",
            "band_gap_eV": 3.0,
            "n_electrons": 16,
        }
        dk_small, _ = pred.predict(small)
        dk_large, _ = pred.predict(large)
        assert dk_large >= dk_small * 0.5

    def test_conservative_end_non_negative(self):
        pred = KSpacingPredictor()
        features = {
            "n_atoms": 1,
            "dim": 3,
            "cell_volume_bohr3": 50.0,
            "reciprocal_lengths_bohr": [1.5, 1.5, 1.5],
            "character": "metal",
            "band_gap_eV": 0.0,
            "n_electrons": 1,
        }
        mean, sigma = pred.predict(features)
        assert mean > 0
        assert sigma >= 0

    def test_missing_model_file_raises(self):
        pred = KSpacingPredictor(model_path="/nonexistent/path/model.pkl")
        with pytest.raises(FileNotFoundError, match="not found"):
            pred._ensure_loaded()

    def test_build_ml_predictor_returns_callable(self):
        fn = build_ml_predictor()
        assert callable(fn)
        features = {
            "n_atoms": 4,
            "dim": 3,
            "cell_volume_bohr3": 200.0,
            "reciprocal_lengths_bohr": [0.5, 0.5, 0.6],
            "character": "insulator",
            "band_gap_eV": 2.0,
            "n_electrons": 24,
        }
        mean, sigma = fn(features)
        assert isinstance(mean, float)
        assert isinstance(sigma, float)
        assert mean > 0


@_requires_sklearn
@pytest.mark.slow
class TestRecommendIntegration:
    """Smoke tests exercising the full recommend(..., predictor="ml") path."""

    def test_predictor_ml_without_env_var_raises(self):
        import numpy as np
        import vibeqc as vq

        atoms = [vq.Atom(14, [0.0, 0.0, 0.0])]
        sys = vq.PeriodicSystem(3, np.eye(3) * 5.43, atoms)

        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(NotImplementedError, match="VIBEQC_ML_KPOINTS=1"):
                vq.KPoints.recommend(sys, predictor="ml")

    def test_predictor_ml_with_env_var_produces_valid_kpoints(self):
        import numpy as np
        import vibeqc as vq

        atoms = [vq.Atom(14, [0.0, 0.0, 0.0])]
        sys = vq.PeriodicSystem(3, np.eye(3) * 5.43, atoms)

        with mock.patch.dict(os.environ, {ML_PREDICTOR_ENV_VAR: "1"}):
            spec = vq.KPoints.recommend(sys, band_gap=1.1, predictor="ml")

        assert spec is not None
        assert len(spec.mesh) == 3
        assert all(m >= 1 for m in spec.mesh)
        assert "ml" in spec.rationale.lower()

    def test_predictor_ml_metal_produces_dense_mesh(self):
        import numpy as np
        import vibeqc as vq

        atoms = [vq.Atom(74, [0.0, 0.0, 0.0])]
        cell = (
            np.array(
                [[1.585, 1.585, -1.585], [-1.585, 1.585, 1.585], [1.585, -1.585, 1.585]]
            )
            * 2.0
        )
        sys = vq.PeriodicSystem(3, cell, atoms)

        with mock.patch.dict(os.environ, {ML_PREDICTOR_ENV_VAR: "1"}):
            spec = vq.KPoints.recommend(sys, is_metal=True, predictor="ml")

        assert all(m >= 2 for m in spec.mesh)

    def test_predictor_ml_insulator_produces_reasonable_mesh(self):
        import numpy as np
        import vibeqc as vq

        atoms = [vq.Atom(11, [0.0, 0.0, 0.0]), vq.Atom(17, [0.5, 0.5, 0.5])]
        sys = vq.PeriodicSystem(3, np.eye(3) * 5.64, atoms)

        with mock.patch.dict(os.environ, {ML_PREDICTOR_ENV_VAR: "1"}):
            spec = vq.KPoints.recommend(sys, band_gap=5.0, predictor="ml")

        assert all(m >= 1 for m in spec.mesh)
        assert spec.smearing is None or spec.smearing.temperature == 0.0

    def test_explicit_ml_predictor_overrides_bundled(self):
        import numpy as np
        import vibeqc as vq

        atoms = [vq.Atom(14, [0.0, 0.0, 0.0])]
        sys = vq.PeriodicSystem(3, np.eye(3) * 5.43, atoms)

        def my_predictor(features):
            return 0.25, 0.02

        with mock.patch.dict(os.environ, {}, clear=True):
            spec = vq.KPoints.recommend(
                sys,
                band_gap=1.1,
                predictor="ml",
                ml_predictor=my_predictor,
            )

        assert spec is not None
        assert "ml" in spec.rationale.lower()


class TestParsePredictorOutput:
    """Tests for _parse_predictor_output in kpoints.py."""

    def test_tuple_output(self):
        from vibeqc.kpoints import _parse_predictor_output

        mean, sigma = _parse_predictor_output((0.30, 0.05))
        assert mean == 0.30 and sigma == 0.05

    def test_dict_output(self):
        from vibeqc.kpoints import _parse_predictor_output

        mean, sigma = _parse_predictor_output({"delta_k": 0.30, "uncertainty": 0.05})
        assert mean == 0.30 and sigma == 0.05

    def test_dict_alt_keys(self):
        from vibeqc.kpoints import _parse_predictor_output

        mean, sigma = _parse_predictor_output({"mean": 0.25, "sigma": 0.03})
        assert mean == 0.25 and sigma == 0.03

    def test_scalar_output(self):
        from vibeqc.kpoints import _parse_predictor_output

        mean, sigma = _parse_predictor_output(0.30)
        assert mean == 0.30 and sigma == 0.0


@_requires_sklearn
class TestModelRoundTrip:
    """Verify train -> save -> load works."""

    def test_round_trip(self):
        from sklearn.ensemble import RandomForestRegressor
        from vibeqc.data_library.ml_kpredictor import _generate_synthetic_training_data

        X, y = _generate_synthetic_training_data(n_samples=100, seed=42)
        model = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
        )
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path = f.name

        try:
            pred = KSpacingPredictor(model_path=tmp_path)
            features = {
                "n_atoms": 8,
                "dim": 3,
                "cell_volume_bohr3": 300.0,
                "reciprocal_lengths_bohr": [0.5, 0.5, 0.4],
                "character": "insulator",
                "band_gap_eV": 2.5,
                "n_electrons": 48,
            }
            mean, sigma = pred.predict(features)
            assert mean > 0 and sigma >= 0
        finally:
            os.unlink(tmp_path)


@_requires_sklearn
class TestElectronConservation:
    """n_electrons maps sensibly through the model."""

    def test_more_electrons_same_geometry(self):
        pred = KSpacingPredictor()
        base = {
            "n_atoms": 4,
            "dim": 3,
            "cell_volume_bohr3": 200.0,
            "reciprocal_lengths_bohr": [0.5, 0.5, 0.6],
            "character": "insulator",
            "band_gap_eV": 3.0,
        }
        dk_few, _ = pred.predict({**base, "n_electrons": 4})
        dk_many, _ = pred.predict({**base, "n_electrons": 400})
        assert 0.03 < dk_few < 0.80
        assert 0.03 < dk_many < 0.80
