"""Automatic smearing-temperature guess.

Conservative-by-default: if no metallic hint and no band-gap
estimate are available, returns zero smearing rather than
silently turning a likely insulator into a smeared metallic SCF.
"""

from __future__ import annotations

from typing import Optional

from .resolution import SMEARING_PRESETS, SmearingResolution


def guess_smearing_temperature(
    *,
    metallic: Optional[bool] = None,
    band_gap_hartree: Optional[float] = None,
    n_electrons: Optional[float] = None,
) -> SmearingResolution:
    """Conservative automatic Fermi-Dirac smearing guess.

    The auto policy is deliberately cautious: if no metallic hint
    or previous band gap is known, it returns zero smearing rather
    than changing the physics of a likely insulator. For known
    metals, the default is 0.005 Ha (about 0.136 eV / 1580 K),
    matching the value used in vibe-qc's periodic smearing
    regression tests and tutorials.
    """
    if band_gap_hartree is not None:
        gap = float(band_gap_hartree)
        if gap < 0.0:
            raise ValueError(
                "guess_smearing_temperature: band_gap_hartree must be >= 0"
            )
        if gap < 1e-5:
            return SmearingResolution(
                SMEARING_PRESETS["metal"],
                source="auto",
                reason="band_gap_hartree is effectively zero",
            )
        if gap < 0.05:
            return SmearingResolution(
                SMEARING_PRESETS["small-gap"],
                source="auto",
                reason="band_gap_hartree is small; using gentle smearing",
            )
        return SmearingResolution(
            0.0,
            source="auto",
            reason="band_gap_hartree is comfortably insulating",
        )

    if metallic is True:
        return SmearingResolution(
            SMEARING_PRESETS["metal"],
            source="auto",
            reason="metallic=True",
        )
    if metallic is False:
        return SmearingResolution(
            0.0,
            source="auto",
            reason="metallic=False",
        )

    if n_electrons is not None and int(round(float(n_electrons))) % 2 == 1:
        return SmearingResolution(
            SMEARING_PRESETS["metal"],
            source="auto",
            reason=(
                "odd electron count per cell suggests a "
                "metallic/open-shell case"
            ),
        )

    return SmearingResolution(
        0.0,
        source="auto",
        reason=(
            "no metallic hint or band gap supplied; "
            "conservative no-smearing default"
        ),
    )
