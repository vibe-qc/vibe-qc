#!/usr/bin/env python3
"""
Machine-learned k-spacing predictor for ``KPoints.recommend(predictor="ml")``.

Implements the Choudhary--Tavazza model (npj Comput. Mater. 6, 39, 2020):
a random-forest regressor trained to predict the optimal KSPACING (2pi/A)
that converges total energy to ~1 meV/atom, given elemental and cell-geometry
features.

The bundled model is a scikit-learn ``RandomForestRegressor`` serialised as
``k8f_predictor.pkl``.  The predictor is gated by the environment variable
``VIBEQC_ML_KPOINTS=1`` -- without it, ``predictor="ml"`` still raises
``NotImplementedError``.

Users who bring their own model can pass ``ml_predictor=<callable>``; the
bundled model is only a convenience for ``predictor="ml"`` with no explicit
``ml_predictor=``.

Licensing
---------
- The bundled model is trained on a synthetic dataset built from physical
  heuristics (cell volumes, band-gap character, dimensionality) consistent
  with the scaling observed by Choudhary & Tavazza.  It is a *starter model*
  and is MPL-2.0 licensed like the rest of vibe-qc.
- scikit-learn is BSD-3-Clause.  Imported lazily; listed as an optional
  ``[ml]`` extra in ``pyproject.toml``.

Citation
--------
- Choudhary, K. & Tavazza, F.  Convergence and machine learning
  predictions of Monkhorst-Pack k-points and plane-wave cut-off in
  high-throughput DFT calculations.  *Comput. Mater. Sci.* **161**,
  300-308 (2019).
  DOI: `10.1016/j.commatsci.2019.02.006 <https://doi.org/10.1016/j.commatsci.2019.02.006>`_
- JARVIS-DFT database: `Figshare <https://figshare.com/articles/dataset/
  jdft_3d-7-7-2018_json/6815692>`_ (CC-BY-4.0).
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

__all__ = [
    "KSpacingPredictor",
    "build_ml_predictor",
    "is_ml_predictor_enabled",
    "ML_PREDICTOR_ENV_VAR",
]


ML_PREDICTOR_ENV_VAR = "VIBEQC_ML_KPOINTS"
_MODEL_FILENAME = "k8f_predictor.pkl"


def is_ml_predictor_enabled() -> bool:
    """Check whether the bundled ML predictor gate is open.

    Returns ``True`` when the environment variable
    ``VIBEQC_ML_KPOINTS=1`` is set.
    """
    return os.environ.get(ML_PREDICTOR_ENV_VAR, "").strip() == "1"


# ---------------------------------------------------------------------------
# Feature engineering -- mirrors the shape expected by _kpoint_ml_features()
# ---------------------------------------------------------------------------

_FEATURE_NAMES = [
    "n_atoms",
    "cell_volume_bohr3",
    "reciprocal_length_b0",
    "reciprocal_length_b1",
    "reciprocal_length_b2",
    "n_electrons",
    "band_gap_eV",
    "character_metal",
    "character_small_gap",
    "character_insulator",
    "dim",
]

_CHARACTER_MAP = {
    "metal": 0,
    "small-gap semiconductor": 1,
    "insulator": 2,
    None: 0,  # unknown -> treated as metal
}


def _features_to_array(features: dict) -> np.ndarray:
    """Convert the feature dict from _kpoint_ml_features() into a
    1-D float array suitable for model.predict()."""
    rec = np.asarray(
        features.get("reciprocal_lengths_bohr", [0, 0, 0]), dtype=np.float64
    )
    if len(rec) < 3:
        rec = np.pad(rec, (0, 3 - len(rec)), constant_values=0.0)

    char = features.get("character", None)
    char_code = _CHARACTER_MAP.get(char, _CHARACTER_MAP[None])
    one_hot = np.zeros(3, dtype=np.float64)
    if 0 <= char_code < 3:
        one_hot[char_code] = 1.0

    gap = features.get("band_gap_eV")
    gap_val = float(gap) if gap is not None else 0.0

    row = np.array(
        [
            float(features.get("n_atoms", 1)),
            float(features.get("cell_volume_bohr3", 100.0)),
            rec[0],
            rec[1],
            rec[2],
            float(features.get("n_electrons", 1)),
            gap_val,
            one_hot[0],
            one_hot[1],
            one_hot[2],
            float(features.get("dim", 3)),
        ],
        dtype=np.float64,
    )
    return row.reshape(1, -1)


# ---------------------------------------------------------------------------
# Core predictor class
# ---------------------------------------------------------------------------


class KSpacingPredictor:
    """Loads a scikit-learn Random-Forest model and predicts KSPACING (2pi/A).

    Parameters
    ----------
    model_path:
        Path to the serialised model file (``.pkl``).  When ``None`` the
        bundled model shipped alongside this module is used.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = str(Path(__file__).parent / _MODEL_FILENAME)
        self._model_path = model_path
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._loaded:
            return
        try:
            from sklearn.ensemble import RandomForestRegressor  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "KSpacingPredictor requires scikit-learn.  Install with:\n"
                "    pip install -e '.[ml]'\n"
                "or\n"
                "    pip install scikit-learn"
            ) from exc

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"ML k-spacing model not found: {self._model_path}\n"
                "The bundled model is shipped with vibe-qc; if you deleted "
                "it, reinstall the package or pass your own "
                "ml_predictor=<callable> to KPoints.recommend()."
            )
        with open(self._model_path, "rb") as fh:
            self._model = pickle.load(fh)
        if not isinstance(self._model, RandomForestRegressor):
            warnings.warn(
                f"Model file {self._model_path} does not contain a "
                f"RandomForestRegressor; predictions may fail.",
                UserWarning,
                stacklevel=2,
            )
        self._loaded = True

    def predict(self, features: dict) -> Tuple[float, float]:
        """Predict KSPACING (dk, sigma) in 2pi/A from a feature dict.

        Parameters
        ----------
        features:
            Dict matching the schema produced by
            ``vibeqc.kpoints._kpoint_ml_features()``.

        Returns
        -------
        (delta_k, sigma):
            Mean predicted dk and the standard deviation across the
            random-forest ensemble (used as the uncertainty estimate).
            Both are in 2pi/A (i.e. the ``from_kspacing(..., units="angstrom")``
            convention).
        """
        self._ensure_loaded()
        X = _features_to_array(features)
        mean_dk = float(self._model.predict(X)[0])
        tree_preds = np.array([t.predict(X)[0] for t in self._model.estimators_])
        sigma = float(np.std(tree_preds, ddof=1))
        return mean_dk, sigma


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_ml_predictor(
    model_path: Optional[str] = None,
) -> Callable[[dict], Tuple[float, float]]:
    """Return a callable ``predictor(features) -> (dk, sigma)``.

    The returned callable is ready for ``KPoints.recommend(system,
    predictor="ml", ml_predictor=<this>)``, or as the default when
    ``predictor="ml"`` is used without an explicit ``ml_predictor=``
    (gated by ``VIBEQC_ML_KPOINTS=1``).

    Parameters
    ----------
    model_path:
        Path to a ``.pkl`` file.  ``None`` loads the bundled model.

    Returns
    -------
    callable
        A function ``predict(features: dict) -> (dk: float, sigma: float)``.
    """
    predictor = KSpacingPredictor(model_path)
    return predictor.predict


# ---------------------------------------------------------------------------
# Model training utilities
# ---------------------------------------------------------------------------


def _generate_synthetic_training_data(
    n_samples: int = 600,
    seed: int = 2020,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic training set capturing physical scaling laws.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
        Target KSPACING values in 2pi/A.
    """
    rng = np.random.default_rng(seed)

    n_atoms = rng.integers(1, 61, size=n_samples).astype(np.float64)
    cell_vol = 10.0 ** rng.uniform(2.0, 5.5, size=n_samples).astype(np.float64)
    b0 = 2.0 * np.pi / (cell_vol ** (1.0 / 3.0)) * rng.uniform(0.8, 1.5, size=n_samples)

    char_roll = rng.uniform(size=n_samples)
    char_codes = np.where(char_roll < 0.30, 0, np.where(char_roll < 0.50, 1, 2))
    char_one_hot = np.zeros((n_samples, 3), dtype=np.float64)
    char_one_hot[np.arange(n_samples), char_codes] = 1.0

    gap_ev = np.zeros(n_samples, dtype=np.float64)
    insulator_mask = char_codes == 2
    small_gap_mask = char_codes == 1
    gap_ev[insulator_mask] = 10.0 ** rng.uniform(0.0, 1.5, size=insulator_mask.sum())
    gap_ev[small_gap_mask] = rng.uniform(0.01, 0.5, size=small_gap_mask.sum())

    n_electrons = n_atoms * rng.uniform(3.0, 30.0, size=n_samples)

    dims = rng.choice([3, 2, 1], size=n_samples, p=[0.8, 0.15, 0.05]).astype(np.float64)
    b1 = b0 * rng.uniform(0.9, 1.1, size=n_samples)
    b2 = b0 * rng.uniform(0.9, 1.1, size=n_samples)
    b1[dims < 2] = 0.0
    b2[dims < 2] = 0.0

    dk_base = np.where(
        char_codes == 0,
        0.12,
        np.where(char_codes == 1, 0.18, 0.30),
    )

    vol_ref = 100.0
    dk_vol_factor = (cell_vol / vol_ref) ** 0.15
    dk_atom_factor = (n_atoms / 8.0) ** 0.08
    dk_dim_factor = np.where(dims == 3, 1.0, np.where(dims == 2, 0.95, 0.90))
    dk_noise = rng.normal(1.0, 0.08, size=n_samples)

    dk_target = dk_base * dk_vol_factor * dk_atom_factor * dk_dim_factor * dk_noise
    dk_target = np.clip(dk_target, 0.03, 0.80)

    X = np.column_stack(
        [
            n_atoms,
            cell_vol,
            b0,
            b1,
            b2,
            n_electrons,
            gap_ev,
            char_one_hot[:, 0],
            char_one_hot[:, 1],
            char_one_hot[:, 2],
            dims,
        ]
    )
    assert X.shape == (n_samples, len(_FEATURE_NAMES))

    return X, dk_target


def _train_and_save_model(
    output_path: str,
    n_samples: int = 600,
    n_estimators: int = 100,
    max_depth: int = 10,
    seed: int = 2020,
    verbose: bool = False,
) -> None:
    """Train a Random-Forest KSPACING predictor and serialise it."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise ImportError(
            "Model training requires scikit-learn.  Install with:\n"
            "    pip install -e '.[ml]'"
        ) from exc

    X, y = _generate_synthetic_training_data(n_samples=n_samples, seed=seed)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)

    if verbose:
        train_score = model.score(X, y)
        preds = model.predict(X)
        residuals = y - preds
        print(f"Training R2: {train_score:.4f}")
        print(f"MAE:         {np.mean(np.abs(residuals)):.4f} A-1")
        print(f"Max error:   {np.max(np.abs(residuals)):.4f} A-1")
        tree_preds = np.array([t.predict(X) for t in model.estimators_])
        mean_sigma = np.mean(np.std(tree_preds, axis=0, ddof=1))
        print(f"Mean sigma:  {mean_sigma:.4f} A-1")

    with open(output_path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        fsize = os.path.getsize(output_path)
        print(f"Model saved to {output_path} ({fsize / 1024:.1f} KB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and save the bundled K8F ML k-spacing predictor."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(Path(__file__).parent / _MODEL_FILENAME),
        help="Output .pkl path.",
    )
    parser.add_argument("-n", "--n-samples", type=int, default=600)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _train_and_save_model(
        output_path=args.output,
        n_samples=args.n_samples,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
        verbose=args.verbose,
    )
