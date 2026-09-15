"""Structural invariants of the citation database.

These are the checks that do not need a network call, do not need to know
what a paper says, and would each have caught a defect that survived review.
They complement the per-feature pinning files
([`test_scf_accelerator_citation_coverage.py`](test_scf_accelerator_citation_coverage.py),
[`test_functional_citation_coverage.py`](test_functional_citation_coverage.py),
[`test_dispersion_citation_coverage.py`](test_dispersion_citation_coverage.py),
[`test_ecp_citation_coverage.py`](test_ecp_citation_coverage.py)),
which pin *which paper* a feature cites. This file pins the *shape* of the
database, per [CLAUDE.md § 8](../CLAUDE.md).

What each test is defending against, with the defect that motivated it:

* `test_no_two_entries_share_a_doi` -- `assemble()` deduplicates by entry
  key, not by DOI. Two keys for one paper therefore print it twice. Before
  2026-07-10 `method="dftb0", basis="sto-3g"` emitted Hehre 1969 twice, under
  two BibTeX keys, because `hehre_sto_ng_1969` and `hehre_sto3g_1969` were
  the same DOI.

* `test_every_article_has_a_doi` -- an article without a DOI cannot be
  mechanically verified against Crossref, which is how the KDIIS phantom
  ("C. Kollmar, Int. J. Quantum Chem. 105, 685 (2005)") went unnoticed.

* `test_doi_syntax_is_well_formed` -- a typo'd suffix silently 404s. Three
  DOIs in this database did (`10.1063/1.5006075`, `10.1039/b001167k`,
  and one that resolved to an unrelated paper).

* `test_every_route_resolves_to_an_entry` -- a dangling route resolves to
  nothing the moment it fires.

* `test_bibtex_keys_are_unique` -- a collision silently drops one of the two
  references from the user's bibliography.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

import vibeqc as _vq


DATABASE_TOML = (
    Path(_vq.__file__).resolve().parent
    / "output" / "citations" / "database.toml"
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — repo requires 3.11+ per AGENTS.md
    import tomli as tomllib


# A DOI is "10." + registrant + "/" + suffix (ANSI/NISO Z39.104-2022).
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

# Entry-key pairs that legitimately share a DOI, with the reason. Each is a
# real duplicate awaiting a merge that this audit deliberately did not make.
# Shrink this set; do not grow it. A new duplicate is a bug, not an entry.
#
# Empty since 2026-07-10: the last two pairs were merged rather than tolerated.
# `vandevondele_molopt_2007` folded into `vandevondele_basisopt_2007`, and
# `daga_bdiis_2020` into `daga_optbasis_2020`. Prefer a merge; an allow-list row
# only defers the double-print that `test_no_two_entries_share_a_doi` guards.
_KNOWN_DUPLICATE_DOIS: dict[str, frozenset[str]] = {}

# Entries with `kind = "article"` that genuinely have no DOI. Keep empty if
# possible: an article without a DOI cannot be Crossref-verified.
_ARTICLES_WITHOUT_DOI: frozenset[str] = frozenset()


def _database() -> dict:
    with DATABASE_TOML.open("rb") as fh:
        return tomllib.load(fh)


def _entries() -> dict:
    return _database()["entries"]


def _routed_keys() -> set[str]:
    routed: set[str] = set()
    for table in _database()["routes"].values():
        for value in table.values():
            routed.update(value if isinstance(value, list) else [value])
    return routed


def test_no_two_entries_share_a_doi() -> None:
    by_doi: dict[str, list[str]] = defaultdict(list)
    for key, entry in _entries().items():
        doi = entry.get("doi")
        if doi:
            by_doi[doi.strip().lower()].append(key)

    duplicates = {
        doi: sorted(keys) for doi, keys in by_doi.items() if len(keys) > 1
    }
    unexpected = {
        doi: keys
        for doi, keys in duplicates.items()
        if frozenset(keys) != _KNOWN_DUPLICATE_DOIS.get(doi.lower())
    }
    assert not unexpected, (
        "two entry keys share one DOI, so a job whose routes reach both will "
        "print the same paper twice (assemble() deduplicates by entry key, "
        "not by DOI):\n"
        + "\n".join(f"  {doi} -> {keys}" for doi, keys in unexpected.items())
        + "\nMerge the entries, or add them to _KNOWN_DUPLICATE_DOIS with a "
        "reason."
    )


def test_known_duplicate_dois_still_exist() -> None:
    """If a merge lands, the allow-list entry must go with it."""
    by_doi: dict[str, set[str]] = defaultdict(set)
    for key, entry in _entries().items():
        if entry.get("doi"):
            by_doi[entry["doi"].strip().lower()].add(key)
    for doi, expected in _KNOWN_DUPLICATE_DOIS.items():
        actual = by_doi.get(doi.lower(), set())
        assert actual == set(expected), (
            f"_KNOWN_DUPLICATE_DOIS is stale for {doi}: expected keys "
            f"{sorted(expected)}, database has {sorted(actual)}. If the "
            f"duplicate was merged, drop the allow-list row."
        )


@pytest.mark.parametrize(
    "key",
    sorted(k for k, v in _entries().items() if v.get("kind") == "article"),
)
def test_every_article_has_a_doi(key: str) -> None:
    entry = _entries()[key]
    if key in _ARTICLES_WITHOUT_DOI:
        pytest.skip(f"{key} is a documented DOI-less article")
    assert entry.get("doi"), (
        f"[entries.{key}] has kind='article' but no DOI, so nothing can "
        f"check mechanically that the paper exists. This is how the KDIIS "
        f"phantom reference survived review."
    )


@pytest.mark.parametrize(
    "key",
    sorted(k for k, v in _entries().items() if v.get("doi")),
)
def test_doi_syntax_is_well_formed(key: str) -> None:
    doi = _entries()[key]["doi"]
    assert _DOI_RE.match(doi), (
        f"[entries.{key}] DOI {doi!r} is not a syntactically valid DOI "
        f"(expected '10.<registrant>/<suffix>')."
    )
    assert not doi.startswith(("http", "doi:")), (
        f"[entries.{key}] DOI {doi!r} must be the bare DOI, not a URL."
    )


def test_every_route_resolves_to_an_entry() -> None:
    db = _database()
    entries = db["entries"]
    dangling: list[str] = []
    for table_name, table in db["routes"].items():
        for route_key, value in table.items():
            for entry_key in value if isinstance(value, list) else [value]:
                if entry_key not in entries:
                    dangling.append(
                        f"routes.{table_name}.{route_key!r} -> {entry_key!r}"
                    )
    assert not dangling, (
        "dangling routes -- the citation resolves to nothing when it "
        "fires:\n" + "\n".join(f"  {d}" for d in dangling)
    )


@pytest.mark.parametrize("key", sorted(_entries()))
def test_every_entry_declares_a_bibtex_key(key: str) -> None:
    assert _entries()[key].get("bibtex_key"), (
        f"[entries.{key}] declares no bibtex_key, so it cannot be emitted "
        f"into a .bibtex file."
    )


def test_bibtex_keys_are_unique() -> None:
    """Two entries sharing a BibTeX key collide in the emitted .bibtex:
    the user's bibliography silently keeps only one of them.

    Note this is *not* the same as `bibtex_key == entry_key`. Twenty-five
    entries deliberately differ (`vibeqc_software` -> `peintinger_vibeqc`,
    `libint_valeev` -> `valeev_libint`, ...), because the entry key names
    the routed feature while the BibTeX key names the publication.
    """
    by_bibtex: dict[str, list[str]] = defaultdict(list)
    for key, entry in _entries().items():
        by_bibtex[entry["bibtex_key"]].append(key)
    collisions = {b: sorted(k) for b, k in by_bibtex.items() if len(k) > 1}
    assert not collisions, (
        "entries share a BibTeX key and would collide in the emitted "
        ".bibtex:\n"
        + "\n".join(f"  {b} <- {keys}" for b, keys in collisions.items())
    )
