"""Status-text regressions for semiempirical user-guide pages."""

from __future__ import annotations

import inspect
from pathlib import Path


_SEMIEMPIRICAL_GUIDE = (
    Path(__file__).resolve().parents[1] / "docs" / "user_guide" / "semiempirical.md"
)
_MSINDO_GUIDE = (
    Path(__file__).resolve().parents[1] / "docs" / "user_guide" / "msindo.md"
)
_CCM_GUIDE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "user_guide"
    / "cyclic_cluster_model.md"
)


def test_msindo_gradient_status_is_current():
    text = _SEMIEMPIRICAL_GUIDE.read_text(encoding="utf-8")

    assert "MSINDO's analytic gradients are partially implemented" not in text
    assert "full assembly pending" not in text
    assert "molecular closed-shell analytic gradients" in text
    assert "closed-shell CCM" in text
    assert "native route" in text
    assert "excited-state/root-tracking gradients" in text


def test_msindo_user_guide_feature_matrix_is_current():
    text = _MSINDO_GUIDE.read_text(encoding="utf-8")

    assert "H-Br (Z 1-35)" not in text
    assert "Kr + 4th row onward" not in text
    assert "analytic partially implemented" not in text
    assert "full assembly pending" not in text
    assert "H-Xe (Z 1-54)" in text
    assert "closed-shell INDO analytic gradients via C++" in text
    assert "Cs and heavier (Z > 54)" in text


def test_periodic_semiempirical_public_support_is_not_overclaimed():
    text = _SEMIEMPIRICAL_GUIDE.read_text(encoding="utf-8")

    assert "All families support Gamma-point periodic calculations" not in text
    assert "GFN2-xTB | Gated experimental" in text
    assert "Gamma experimental; full-k gated" in text
    assert "PM6 / UPM6 | Molecular development" in text
    assert "full heavy-heavy two-center tensor parity remains open" in text
    assert "Gamma experimental, closed-shell" in text
    assert "molecular yes; periodic no" in text
    assert "Public full-k semiempirical support is DFTB0 and" in text
    assert "SCC-DFTB only, closed-shell only" in text
    assert "MSINDO periodic" in text
    assert "uses the SECCM cyclic-cluster route" in text


def test_ccm_guide_distinguishes_high_level_and_direct_madelung_defaults():
    from vibeqc.semiempirical.methods.msindo_ccm import (
        CCMOptions,
        ccm_gradient_fd,
        ccm_optimize,
        run_ccm,
    )
    from vibeqc.semiempirical.methods.msindo_ccm_gradient_analytic import (
        ccm_gradient_analytic,
    )

    assert CCMOptions().madelung is True
    for direct_helper in (
        run_ccm,
        ccm_gradient_fd,
        ccm_gradient_analytic,
        ccm_optimize,
    ):
        assert (
            inspect.signature(direct_helper).parameters["madelung"].default
            is False
        )

    text = " ".join(_CCM_GUIDE.read_text(encoding="utf-8").split())
    assert "defaults `madelung=True`" in text
    assert "direct `run_ccm`, gradient, and optimization helpers" in text
    assert "must opt in explicitly" in text
    assert "`True` in `CCMOptions`; `False` in `run_ccm`" in text
