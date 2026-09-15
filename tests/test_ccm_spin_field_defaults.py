"""Gamma-CCM result classes default both spin-density fields to ``None``.

This is the class side of the contract whose consumer side is pinned in
``tests/test_spin_channel_detection.py``.  The runner-contract adapters for the
supercell-Gamma CCM routes (:class:`CCMRealGammaResult`,
:class:`CCMFourCentreResult`) and the CCM KS result (:class:`CCMKSResult`)
declare ``density_alpha`` / ``density_beta`` as dataclass *fields* that are
``None`` on a closed-shell run.  ``hasattr(result, "density_alpha")`` is
therefore true for a closed-shell result and cannot detect open shell, which is
why every consumer must test spin-density *values*.

It is split from the consumer tests because a test file that imports
``vibeqc.periodic.ccm`` carries the experimental marker, while the consumer
guard protects shipped property code and stays verified.  Keeping the import
here also means a broken CCM import can no longer fail collection of that
guard.
"""

from __future__ import annotations

import dataclasses as dc

import pytest

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _nullable_spin_result_classes():
    from vibeqc.periodic.ccm.dft import CCMKSResult
    from vibeqc.periodic.ccm.four_center_runner import CCMFourCentreResult
    from vibeqc.periodic.ccm.real_gamma_runner import CCMRealGammaResult

    return [CCMRealGammaResult, CCMFourCentreResult, CCMKSResult]


@pytest.mark.parametrize(
    "cls", _nullable_spin_result_classes(), ids=lambda c: c.__name__
)
def test_spin_fields_default_to_none_so_hasattr_is_useless(cls) -> None:
    fields = {f.name: f for f in dc.fields(cls)}
    for name in ("density_alpha", "density_beta"):
        assert name in fields, f"{cls.__name__} should declare {name}"
        assert fields[name].default is None, (
            f"{cls.__name__}.{name} must default to None -- the closed-shell "
            "case is what makes a value-based open-shell test necessary"
        )
