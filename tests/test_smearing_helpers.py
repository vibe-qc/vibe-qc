"""User-facing smearing-temperature helper tests.

The SCF drivers store canonical ``k_B T`` in Hartree. These helpers are
the small policy layer that lets user inputs stay readable (Kelvin/eV/Ry,
named presets, and a conservative auto guess).
"""
from __future__ import annotations

import pytest

import vibeqc as vq


def test_smearing_unit_conversions():
    assert vq.electronvolt_to_hartree_temperature(1.0) == pytest.approx(
        1.0 / vq.EV_PER_HARTREE
    )
    assert vq.rydberg_to_hartree_temperature(0.1) == pytest.approx(0.05)

    r_k = vq.resolve_smearing_temperature(300.0, unit="kelvin")
    assert r_k.temperature == pytest.approx(vq.kelvin_to_hartree_temperature(300.0))
    assert r_k.source == "explicit:kelvin"

    r_ev = vq.resolve_smearing_temperature(0.2, unit="eV")
    assert r_ev.temperature == pytest.approx(0.2 / vq.EV_PER_HARTREE)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1000 K", 1000.0 * vq.KB_HARTREE_PER_K),
        ("0.1 eV", 0.1 / vq.EV_PER_HARTREE),
        ("0.02 Ha", 0.02),
        ("0.01 hartree", 0.01),
        ("0.2 Ry", 0.1),
    ],
)
def test_smearing_numeric_strings_can_carry_units(value, expected):
    r = vq.resolve_smearing_temperature(value)
    assert r.temperature == pytest.approx(expected)
    assert r.source.startswith("explicit:")


def test_smearing_presets_and_auto_policy():
    metal = vq.resolve_smearing_temperature("metal")
    assert metal.temperature == pytest.approx(0.005)
    assert metal.source == "preset:metal"

    small_gap = vq.resolve_smearing_temperature("auto", band_gap_hartree=0.01)
    assert small_gap.temperature == pytest.approx(0.002)
    assert small_gap.source == "auto"

    metallic = vq.resolve_smearing_temperature("auto", metallic=True)
    assert metallic.temperature == pytest.approx(0.005)

    conservative = vq.resolve_smearing_temperature("auto")
    assert conservative.temperature == pytest.approx(0.0)
    assert "conservative" in conservative.reason


def test_smearing_rejects_unknown_method_or_preset():
    with pytest.raises(NotImplementedError, match="Smearing method"):
        vq.resolve_smearing_temperature(0.01, method="gaussian")

    with pytest.raises(ValueError, match="unknown smearing preset"):
        vq.resolve_smearing_temperature("mystery-metal")
