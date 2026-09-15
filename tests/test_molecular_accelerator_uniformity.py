"""Molecular SCF — accelerator + dynamic-damping option-surface uniformity.

Verifies that RHFOptions / UHFOptions / RKSOptions / UKSOptions all
carry the same SCF-accelerator + dynamic-damping selection surface,
with consistent defaults. Mirror of `test_periodic_accelerator_uniformity.py`
for the molecular side — catches future drift when a field is added
to one molecular options class but not the others.

Note: the molecular default is EDIIS_DIIS (Garza/Scuseria 2012 hybrid,
v0.8.0), unlike periodic which still defaults to DIIS. See
[docs/user_guide/scf_convergence.md](../docs/user_guide/scf_convergence.md)
for the rationale.
"""

from __future__ import annotations

import pytest

from vibeqc import (
    RHFOptions,
    UHFOptions,
    RKSOptions,
    UKSOptions,
    SCFAccelerator,
)


_MOLECULAR_OPTION_CLASSES = (
    RHFOptions,
    UHFOptions,
    RKSOptions,
    UKSOptions,
)


# ---------------------------------------------------------------------------
# scf_accelerator — selectable on every molecular options class.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("OptionsCls", _MOLECULAR_OPTION_CLASSES)
def test_molecular_scf_accelerator_default_is_ediis_diis(OptionsCls):
    """All molecular options default to EDIIS_DIIS (v0.8.0 production hybrid)."""
    assert OptionsCls().scf_accelerator == SCFAccelerator.EDIIS_DIIS


@pytest.mark.parametrize("OptionsCls", _MOLECULAR_OPTION_CLASSES)
@pytest.mark.parametrize("accel", [
    SCFAccelerator.DIIS,
    SCFAccelerator.KDIIS,
    SCFAccelerator.EDIIS,
    SCFAccelerator.EDIIS_DIIS,
    SCFAccelerator.ADIIS,
    SCFAccelerator.ADIIS_DIIS,
])
def test_molecular_scf_accelerator_field_is_settable(OptionsCls, accel):
    """Every accelerator-family choice round-trips on every molecular options."""
    o = OptionsCls()
    o.scf_accelerator = accel
    assert o.scf_accelerator == accel


@pytest.mark.parametrize("OptionsCls", _MOLECULAR_OPTION_CLASSES)
def test_molecular_ediis_diis_switch_threshold_default(OptionsCls):
    """Default threshold matches the PySCF convention (1e-1)."""
    assert OptionsCls().ediis_diis_switch_threshold == pytest.approx(1e-1)


@pytest.mark.parametrize("OptionsCls", _MOLECULAR_OPTION_CLASSES)
def test_molecular_ediis_diis_switch_threshold_is_settable(OptionsCls):
    o = OptionsCls()
    o.ediis_diis_switch_threshold = 5e-2
    assert o.ediis_diis_switch_threshold == pytest.approx(5e-2)


# ---------------------------------------------------------------------------
# Dynamic damping — selectable on every molecular options class.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("OptionsCls", _MOLECULAR_OPTION_CLASSES)
def test_molecular_dynamic_damping_default_is_true(OptionsCls):
    # ORCA-style adaptive damping is on by default for molecular SCF (v0.15.x).
    assert OptionsCls().dynamic_damping is True


@pytest.mark.parametrize("OptionsCls", _MOLECULAR_OPTION_CLASSES)
def test_molecular_dynamic_damping_bounds_defaults(OptionsCls):
    o = OptionsCls()
    assert o.dynamic_damping_min == pytest.approx(0.0)
    assert o.dynamic_damping_max == pytest.approx(0.95)


@pytest.mark.parametrize("OptionsCls", _MOLECULAR_OPTION_CLASSES)
def test_molecular_dynamic_damping_fields_are_settable(OptionsCls):
    o = OptionsCls()
    o.dynamic_damping = True
    o.dynamic_damping_min = 0.1
    o.dynamic_damping_max = 0.9
    assert o.dynamic_damping is True
    assert o.dynamic_damping_min == pytest.approx(0.1)
    assert o.dynamic_damping_max == pytest.approx(0.9)
