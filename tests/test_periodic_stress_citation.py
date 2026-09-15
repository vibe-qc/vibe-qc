"""The periodic stress tensor cites the method it implements.

``vibeqc.periodic_gapw_stress`` computes the GPW/GAPW stress by finite
difference following the Nielsen-Martin (1985) Hellmann-Feynman + Pulay
decomposition. The stress is an offline post-SCF property that returns a bare
(3, 3) array, so like the BDIIS optimiser it has no ``[routes.*]`` entry and
surfaces its provenance through a module citation constant instead (CLAUDE.md
§ 8).

This is the pin that keeps ``STRESS_CITATION_KEYS`` from becoming dead weight:
every key it names must resolve in the citation database, and it must name the
paper the module docstring actually credits -- not Doll 2004, whose *analytic*
LCAO cell gradient is a different method than the finite-difference stress that
ships.
"""

from __future__ import annotations

import sys
from pathlib import Path

import vibeqc as _vq
from vibeqc.periodic_gapw_stress import STRESS_CITATION_KEYS, method_citations

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


def test_stress_citation_keys_resolve_in_database() -> None:
    entries = _database()["entries"]
    missing = [k for k in STRESS_CITATION_KEYS if k not in entries]
    assert not missing, (
        f"the periodic stress tensor cites citation-database keys that do not "
        f"exist in database.toml: {missing}. Add the [entries.<key>] blocks "
        f"(CLAUDE.md § 8)."
    )
    assert method_citations() == STRESS_CITATION_KEYS


def test_stress_cites_nielsen_martin_not_doll() -> None:
    """The shipped stress is Nielsen-Martin finite difference. Doll 2004 is the
    analytic cell gradient -- a different method -- and must not be cited for
    it (that misattribution is exactly why doll_stress_2004 was removed)."""
    assert "nielsen_martin_1985" in STRESS_CITATION_KEYS
    assert "doll_stress_2004" not in STRESS_CITATION_KEYS
    assert "doll_stress_2004" not in _database()["entries"]
