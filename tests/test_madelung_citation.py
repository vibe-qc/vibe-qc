"""The Makov-Payne finite-box correction cites its source paper.

``vibeqc.madelung`` offers the monopole (Makov-Payne) charge correction as a
library capability -- ``madelung_energy_correction``,
``makov_payne_coefficient_cubic``, ``apply_madelung_correction``. It is *not* a
``run_periodic_job`` feature: every Coulomb method reachable through the runner
zeroes it (EWALD_3D and SLAB return 0.0; the Γ-only DIRECT_TRUNCATED user route
``FFT_POISSON`` was retired in v0.13.0 and raises), and the default GDF path
applies the exxdiv='ewald' exchange-divergence Madelung constant instead. So
like the BDIIS optimiser and the periodic stress tensor, the module surfaces
its provenance through a citation constant rather than a ``[routes.*]`` row
(CLAUDE.md § 8).

This pin keeps ``MADELUNG_CITATION_KEYS`` from decaying into dead weight: the
key it names must resolve in the database. It replaces makov_payne_1995's slot
in the repo-wide dead-entry allow-list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import vibeqc as _vq
from vibeqc.madelung import MADELUNG_CITATION_KEYS, method_citations

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover -- repo requires 3.11+ per AGENTS.md
    import tomli as tomllib


DATABASE_TOML = (
    Path(_vq.__file__).resolve().parent / "output" / "citations" / "database.toml"
)


def _database() -> dict:
    with DATABASE_TOML.open("rb") as fh:
        return tomllib.load(fh)


def test_madelung_citation_keys_resolve_in_database() -> None:
    entries = _database()["entries"]
    missing = [k for k in MADELUNG_CITATION_KEYS if k not in entries]
    assert not missing, (
        f"the Makov-Payne correction cites citation-database keys that do not "
        f"exist in database.toml: {missing}. Add the [entries.<key>] blocks "
        f"(CLAUDE.md § 8)."
    )
    assert method_citations() == MADELUNG_CITATION_KEYS
    assert "makov_payne_1995" in MADELUNG_CITATION_KEYS
