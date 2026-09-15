"""No citation entry is reachable from nothing.

CLAUDE.md § 8 step 2: "a citation without a route is silently dead weight". An
entry that no route reaches can never appear in a user's ``.out`` references
block, ``.bibtex``, ``.references`` or ``.system`` manifest, so the attribution
it was added to preserve is not actually preserved.

Reachability has **two** channels, and checking only the first is what let this
class of rot accumulate:

1. **Static routes** -- a key named in some ``[routes.*]`` table. Fired by
   ``assemble()`` from the job's plan.

2. **Dynamic entries** -- a key passed to ``assemble(extra_entries=[...])``, or
   exported as a module citation constant the way
   ``vibeqc.basis_optimization.bdiis.BDIIS_CITATION_KEYS`` is. These are real
   citation surfaces (they reach ``.bibtex`` / ``res.citations``) but they name
   the key in Python source, not in the routes table.

A naive audit that walks only ``[routes.*]`` reports every dynamic entry as dead
weight. On 2026-07-10 that produced a list of 26 "unrouted" entries, of which 10
were in fact reachable -- the five MS/XMS-CASPT2 papers and the UNO-CAS
reference fired from ``runner.py``, the two MACE foundation-model papers from
``mlip/_mace_models.py``, and two basis-optimisation papers from
``BDIIS_CITATION_KEYS``.

So channel 2 is detected here by parsing every module under ``python/vibeqc``
and collecting string literals that *exactly* equal an entry key. Exact match,
via ``ast``: a key mentioned in a comment does not count (comments are not in
the AST, and a comment is not a citation surface), and a docstring can never
compare equal to a short key. A false positive would require someone to write
the key as a bare string literal and then not use it.

The converse hole -- a ``[routes.*]`` row that exists but that no production
caller can ever trigger -- is not detectable from the entry side at all. The
whole ``[routes.numerics]`` table was in that state until 2026-07-10: every row
was routed, so every entry looked alive, while no production code ever passed
``numerics=`` to ``assemble()``. Guarding that requires a producer-side pin, as
in ``tests/test_kpoints.py::test_every_kpoints_citation_key_resolves_to_a_route``
and the runner-wiring pins in ``tests/test_feature_citation_end_to_end.py``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import vibeqc as _vq

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover -- repo requires 3.11+ per AGENTS.md
    import tomli as tomllib


_PACKAGE_ROOT = Path(_vq.__file__).resolve().parent
DATABASE_TOML = _PACKAGE_ROOT / "output" / "citations" / "database.toml"


# Entries that no route and no Python source reference reaches. Each needs a
# reason and an owner: either wire it, or delete it. Shrink this set; do not
# grow it. Adding an entry with no route in the same commit is the bug CLAUDE.md
# § 8 step 2 exists to prevent.
_KNOWN_DEAD_ENTRIES: dict[str, str] = {
    # Translation-invariant periodic two-centre lattice sums, the theory behind
    # LatticeSumOptions.pair_complete_1e. Added by accae7ff9 (#429) with no
    # route and nothing emitting a routes.numerics key for the flag, so wiring
    # it means emitting that key from the periodic one-electron path -- the
    # #429 lane's call, not a basis/ECP one. Owner: the pbc lane (#429, #731).
    "sharma_beylkin_two_center_2021": (
        "added by #429 without a route; needs a routes.numerics key emitted "
        "when pair_complete_1e is active"
    ),
    # Everything else the 2026-07-10/-11 audit flagged is now reachable:
    #   - the stress paper is nielsen_martin_1985 via STRESS_CITATION_KEYS, and
    #     the Doll / phonopy / Parlinski-Li-Kawazoe entries were removed as
    #     misattributions (the shipped phonons are a Gamma-point mass-weighted
    #     Hessian, no supercell);
    #   - makov_payne_1995 is surfaced via MADELUNG_CITATION_KEYS;
    #   - pisani_crystal_1988 fires for every periodic job (routes.methods
    #     ._periodic_lcao) and pritchard_bse_2019 for every bundled basis
    #     (routes.basis_sets._bse_bundled_provenance).
    # andrae_ecp_1990 left this list on 2026-09-06: 6664da63d (#88) routes the
    # Stuttgart-Koln ECP set from every def2 basis that carries it, so the
    # per-element decision it was waiting on now exists.
}


def _database() -> dict:
    with DATABASE_TOML.open("rb") as fh:
        return tomllib.load(fh)


def _statically_routed(db: dict) -> set[str]:
    routed: set[str] = set()
    for table in db["routes"].values():
        for value in table.values():
            routed.update(value if isinstance(value, list) else [value])
    return routed


def _dynamically_referenced(entry_keys: set[str]) -> set[str]:
    """Entry keys named as exact string literals anywhere under python/vibeqc.

    Catches ``assemble(extra_entries=["pulay_hamilton_uno_1988"])`` and module
    constants like ``BDIIS_CITATION_KEYS``. Parsed rather than grepped so that
    a key appearing only in a comment is *not* counted as a citation surface.
    """
    found: set[str] = set()
    for path in _PACKAGE_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover -- vendored/generated oddities
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in entry_keys
            ):
                found.add(node.value)
    return found


def test_no_entry_is_reachable_from_nothing() -> None:
    db = _database()
    entries = set(db["entries"])
    reachable = _statically_routed(db) | _dynamically_referenced(entries)
    dead = entries - reachable

    unexpected = sorted(dead - set(_KNOWN_DEAD_ENTRIES))
    assert not unexpected, (
        "citation entries that no [routes.*] row and no python/vibeqc source "
        "reference can reach, so they can never appear in a user's references "
        "block, .bibtex or .system manifest:\n"
        + "\n".join(f"  {k}" for k in unexpected)
        + "\n\nPer CLAUDE.md § 8 step 2, wire a route (or an extra_entries "
        "emit), or delete the entry. If it is genuinely blocked, add it to "
        "_KNOWN_DEAD_ENTRIES with the reason."
    )


def test_known_dead_allow_list_is_not_stale() -> None:
    """A key that got wired (or deleted) must leave the allow-list with it.

    Without this, the allow-list silently absolves entries that are fine, and
    the next audit re-derives the same list from scratch.
    """
    db = _database()
    entries = set(db["entries"])
    reachable = _statically_routed(db) | _dynamically_referenced(entries)

    gone = sorted(set(_KNOWN_DEAD_ENTRIES) - entries)
    assert not gone, (
        f"_KNOWN_DEAD_ENTRIES names entries that no longer exist: {gone}. "
        "Drop the allow-list rows."
    )
    now_reachable = sorted(set(_KNOWN_DEAD_ENTRIES) & reachable)
    assert not now_reachable, (
        f"_KNOWN_DEAD_ENTRIES names entries that are now reachable: "
        f"{now_reachable}. Drop the allow-list rows -- the debt is paid."
    )


def _ecp_allow_list() -> set[str]:
    """Read ``_UNROUTED_ECP_ENTRIES`` out of the sibling test's source.

    Parsed rather than imported: pytest does not put ``tests/`` on ``sys.path``,
    and importing a sibling test module for one constant would couple their
    collection order.
    """
    src = (Path(__file__).parent / "test_ecp_citation_coverage.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_UNROUTED_ECP_ENTRIES"
                for t in node.targets
            )
        ):
            return {
                n.value
                for n in ast.walk(node.value)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            }
    raise AssertionError("_UNROUTED_ECP_ENTRIES not found in the sibling test")


def test_ecp_allow_list_agrees_with_the_repo_wide_one() -> None:
    """tests/test_ecp_citation_coverage.py keeps an ECP-specific view of the
    same debt. Two allow-lists drift; pin that the narrow one is a subset."""
    extra = sorted(_ecp_allow_list() - set(_KNOWN_DEAD_ENTRIES))
    assert not extra, (
        f"_UNROUTED_ECP_ENTRIES lists {extra}, absent from _KNOWN_DEAD_ENTRIES. "
        "Either the entry is reachable (drop it there) or it is dead (add it "
        "here with a reason)."
    )


def test_dynamic_reference_channel_is_actually_detected() -> None:
    """Self-check on the AST scan.

    ``pulay_hamilton_uno_1988`` is reachable *only* through
    ``assemble(extra_entries=...)`` in runner.py, and ``daga_optbasis_2020`` is
    reachable both statically and through ``BDIIS_CITATION_KEYS``. If the scan
    silently stopped finding string literals, the dead-entry test above would
    start failing with a pile of false positives -- or worse, a future refactor
    would quietly make it vacuous. Pin one key per channel.
    """
    db = _database()
    entries = set(db["entries"])
    dynamic = _dynamically_referenced(entries)
    assert "pulay_hamilton_uno_1988" in dynamic
    assert "daga_optbasis_2020" in dynamic
    assert "pulay_hamilton_uno_1988" not in _statically_routed(db)
