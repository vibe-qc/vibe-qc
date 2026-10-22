"""Experimental-status marker for COSMO-RS and Direct COSMO-RS.

``run_job(cosmors=...)`` and every Direct COSMO-RS SCF
(``SolventModel(variant="dcosmo-rs")``) fire :func:`_warn_experimental` once
per call, as the status contract of ``docs/methods.md`` requires of an
experimental route. Filter it with::

    warnings.filterwarnings(
        "ignore", category=vibeqc.CosmoRSExperimentalWarning
    )

Why experimental: the only parameter set, ``klamt1998``, was fitted on a
DMol/BPW91/DNP protocol and is transferred, not refitted, to vibe-qc's cavity,
methods and basis sets. Measured 2026-10-22 on the eight pure-liquid vapour
pressures of Klamt 1995 Table 2 at the recommended BPW91/def2-TZVP protocol:
rms 1.76 kcal/mol with a mean of -1.43 on the conductor density, rms 2.93 on
the Direct COSMO-RS density, which overpolarizes water, methanol and ethanol
by 1.5 to 3.2 kcal/mol. The catalog entry is ``docs/experimental/catalog.md``.
"""

from __future__ import annotations

import warnings

__all__ = ["CosmoRSExperimentalWarning"]


class CosmoRSExperimentalWarning(UserWarning):
    """COSMO-RS thermodynamics and Direct COSMO-RS are experimental."""


def _warn_experimental(what: str, *, stacklevel: int = 3) -> None:
    warnings.warn(
        f"{what} is experimental: the klamt1998 COSMO-RS constants are "
        "transferred, not refitted, to vibe-qc's protocol (pure-liquid vapour "
        "pressures at BPW91/def2-TZVP: rms 1.8 kcal/mol on the conductor "
        "density, 2.9 on the Direct COSMO-RS density). Treat the numbers as "
        "research output; see docs/user_guide/solvation.md.",
        category=CosmoRSExperimentalWarning,
        stacklevel=stacklevel,
    )
