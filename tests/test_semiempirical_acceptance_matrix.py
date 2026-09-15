"""Regression tests for the semiempirical acceptance-status matrix."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_matrix_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "semiempirical_acceptance_matrix.py"
    )
    spec = importlib.util.spec_from_file_location(
        "semiempirical_acceptance_matrix", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gfn2_acceptance_matrix_reflects_native_d4_scope():
    matrix = _load_matrix_module()
    issues = matrix.ACCEPTANCE["gfn2_xtb"]["gating_issues"]
    joined = " ".join(issues)

    assert "Self-consistent GFN2-D4 dispersion not implemented" not in joined
    assert "Native post-SCF D4 is limited to H, He, B, C, N, O, F, Ne" in joined
    assert "GFN2D4UnsupportedWarning" in joined
    assert "Periodic AES image-cell multipole Ewald still pending" in issues


def test_pm6_acceptance_matrix_reflects_heavy_heavy_validation_gap():
    matrix = _load_matrix_module()
    pm6 = matrix.ACCEPTANCE["pm6"]
    issues = pm6["gating_issues"]
    joined = " ".join(issues)

    assert pm6["status"] == "development_heavy_heavy_tensor_open"
    assert pm6["validated"] is False
    assert "No MOPAC parity demonstrated" not in joined
    assert "PM6-like total energy" in joined
    assert "not a MOPAC heat of formation" in joined
    assert "full PM6 NDDO multipole tensor" in joined
    assert "Periodic PM6 uses native batched finite differences" in joined
