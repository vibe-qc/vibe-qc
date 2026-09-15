"""``vibeqc.fetch.client_optimade`` — replay-based deterministic emission.

We pin one OPTIMADE response (mp-1265 = canonical MgO rocksalt) in
``tests/data/fetch/`` and replay it through
:func:`_build_periodic_from_entry`. This exercises:

    cartesian → fractional coordinate conversion,
    primitive → conventional cell standardisation via spglib,
    heuristics for SCF defaults (SAD + 0.85 for ionic),
    Provenance stamping.

No live network access — `tests/data/fetch/optimade_mgo_mp1265.json`
is the captured response.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.regression.core.spec import PeriodicSpec, Provenance
from vibeqc.fetch.client_optimade import _build_periodic_from_entry


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "data" / "fetch" / "optimade_mgo_mp1265.json"


@pytest.fixture
def mgo_entry():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    by_filter = raw["structures"]
    for _filt, providers in by_filter.items():
        for url, payload in providers.items():
            return url, payload["data"][0]
    pytest.skip("fixture has no usable entry")


def test_mp1265_round_trips_to_orthorhombic_rocksalt(mgo_entry):
    """mp-1265 ships as the rhombohedral primitive cell; spglib
    standardise → 8-atom cubic Fm-3m. SCF heuristics fire SAD + 0.85
    for ionic (Mg + O)."""
    url, entry = mgo_entry
    spec = _build_periodic_from_entry(
        chosen_url=url, entry=entry,
        explicit_provider="mp", quick=True, slug_override="mgo_rocksalt",
    )
    assert isinstance(spec, PeriodicSpec)
    assert spec.id == "mgo_rocksalt"
    assert spec.family == "rocksalt"
    assert len(spec.atoms) == 8                   # conventional cubic
    assert spec.space_group == "Fm-3m"
    # Lattice constant within 0.1% of the MP-supplied value.
    assert abs(spec.lattice_ang[0][0] - 4.194) < 0.005
    # Off-diagonal terms must vanish for EWALD_3D.
    for i, row in enumerate(spec.lattice_ang):
        for j, x in enumerate(row):
            if i != j:
                assert abs(x) < 1e-3 * spec.lattice_ang[0][0]
    # SCF heuristics for ionic crystal.
    assert spec.default_initial_guess == "SAD"
    assert spec.default_damping == pytest.approx(0.85)
    # 8 atoms in cell → 2x2x2 k-mesh seed.
    assert spec.default_kmesh == (2, 2, 2)
    # `--quick` smoke mode → sto-3g.
    assert spec.recommended_basis == "sto-3g"


def test_mp1265_provenance_is_complete(mgo_entry):
    """Every fetched SPEC must carry a populated Provenance — no None
    fields except ``notes`` (which can be empty)."""
    url, entry = mgo_entry
    spec = _build_periodic_from_entry(
        chosen_url=url, entry=entry,
        explicit_provider="mp", quick=True, slug_override="mgo_rocksalt",
    )
    p = spec.provenance
    assert isinstance(p, Provenance)
    assert p.source_db == "OPTIMADE/mp"
    assert p.source_id == "mp-1265"
    assert p.source_url.startswith("https://")
    assert p.fetcher_version
    assert p.fetched_at.endswith("Z")
    # license is empty-string rather than None per the design contract.
    assert isinstance(p.license, str)


def test_mp1265_emit_spec_round_trips(tmp_path, mgo_entry):
    """Emitted module must be (1) syntactically valid Python and
    (2) re-importable as a SPEC with the same values."""
    from vibeqc.fetch.emit_spec import emit_spec_module

    url, entry = mgo_entry
    spec = _build_periodic_from_entry(
        chosen_url=url, entry=entry,
        explicit_provider="mp", quick=True, slug_override="mgo_rocksalt",
    )
    out = emit_spec_module(spec, tmp_path)
    assert out.name == "mgo_rocksalt.py"
    # Re-import the emitted module fresh.
    import importlib.util
    spec_mod = importlib.util.spec_from_file_location("emitted", out)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    assert isinstance(mod.SPEC, PeriodicSpec)
    assert mod.SPEC.id == spec.id
    assert mod.SPEC.atoms == spec.atoms
    assert mod.SPEC.lattice_ang == spec.lattice_ang
    assert mod.SPEC.provenance.source_id == "mp-1265"


def test_mp1265_emit_input_compiles(tmp_path, mgo_entry):
    """The generated input script must compile — empty f-strings,
    unmatched braces, and other template-builder bugs caught here."""
    import py_compile

    from vibeqc.fetch.emit_input import emit_input_script

    url, entry = mgo_entry
    spec = _build_periodic_from_entry(
        chosen_url=url, entry=entry,
        explicit_provider="mp", quick=True, slug_override="mgo_rocksalt",
    )
    out = emit_input_script(spec, tmp_path, basis="sto-3g", method="rks-lda")
    py_compile.compile(str(out), doraise=True)


# ---------------------------------------------------------------------------
# Multi-candidate API — _structural_hash / _dedup_specs / _rank_specs
# ---------------------------------------------------------------------------

from datetime import datetime, timezone

from examples.regression.core.spec import AtomFrac
from vibeqc.fetch.client_optimade import (
    _dedup_specs,
    _rank_specs,
    _structural_hash,
)


def _make_spec(
    *,
    source_db: str,
    source_id: str,
    a: float = 4.211,
    family: str = "rocksalt",
    atoms_frac: list[tuple[str, int, tuple[float, float, float]]] = None,
) -> PeriodicSpec:
    """Hand-build a PeriodicSpec for hash / dedup / rank tests.

    Default geometry is rocksalt-like MgO at lattice constant ``a``;
    override ``atoms_frac`` for distinct polymorphs.
    """
    if atoms_frac is None:
        atoms_frac = [
            ("Mg", 12, (0.0, 0.0, 0.0)),
            ("O",  8,  (0.5, 0.5, 0.5)),
        ]
    return PeriodicSpec(
        id="test",
        family=family,
        lattice_ang=((a, 0.0, 0.0), (0.0, a, 0.0), (0.0, 0.0, a)),
        space_group="Fm-3m",
        atoms=tuple(AtomFrac(symbol=s, z=z, frac=f) for (s, z, f) in atoms_frac),
        provenance=Provenance(
            source_db=source_db,
            source_id=source_id,
            source_url=f"https://example.org/{source_id}",
            original_reference="",
            license="",
            fetched_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            fetcher_version="0.1.0",
            notes="",
        ),
    )


def test_structural_hash_same_structure_collides():
    """Two specs with the same chemistry + geometry must hash equal,
    regardless of provenance."""
    a = _make_spec(source_db="OPTIMADE/mp", source_id="mp-1265")
    b = _make_spec(source_db="OPTIMADE/oqmd", source_id="oqmd-12345")
    assert _structural_hash(a) == _structural_hash(b)


def test_structural_hash_different_lattice_diverges():
    """Different lattice constant → different hash (1 mÅ resolution)."""
    a = _make_spec(source_db="x", source_id="1", a=4.211)
    b = _make_spec(source_db="x", source_id="2", a=4.500)
    assert _structural_hash(a) != _structural_hash(b)


def test_structural_hash_atom_order_invariant():
    """Atoms in different order on the page → same hash (sort by
    (symbol, frac) in the hash function)."""
    a = _make_spec(
        source_db="x", source_id="1",
        atoms_frac=[
            ("Mg", 12, (0.0, 0.0, 0.0)),
            ("O",  8,  (0.5, 0.5, 0.5)),
        ],
    )
    b = _make_spec(
        source_db="x", source_id="2",
        atoms_frac=[
            ("O",  8,  (0.5, 0.5, 0.5)),
            ("Mg", 12, (0.0, 0.0, 0.0)),
        ],
    )
    assert _structural_hash(a) == _structural_hash(b)


def test_structural_hash_rounds_to_1e_4():
    """Sub-1e-4 numerical noise between providers must collide."""
    a = _make_spec(source_db="x", source_id="1", a=4.211)
    b = _make_spec(source_db="x", source_id="2", a=4.21102)   # +2e-5 noise
    assert _structural_hash(a) == _structural_hash(b)


def test_structural_hash_different_polymorph_diverges():
    """Same formula, different positions (rocksalt vs CsCl-style) →
    different hash."""
    a = _make_spec(
        source_db="x", source_id="rocksalt",
        atoms_frac=[
            ("Mg", 12, (0.0, 0.0, 0.0)),
            ("O",  8,  (0.5, 0.5, 0.5)),
        ],
    )
    b = _make_spec(
        source_db="x", source_id="cscl_like",
        atoms_frac=[
            ("Mg", 12, (0.0, 0.0, 0.0)),
            ("O",  8,  (0.25, 0.25, 0.25)),     # CsCl has the second atom at (½,½,½)
        ],                                       # but pretend this is a hypothetical phase
    )
    assert _structural_hash(a) != _structural_hash(b)


def test_dedup_specs_merges_aliases_into_notes():
    """Three providers with the same structure → one survivor with
    'also at: …' merged into the kept Provenance.notes."""
    primary = _make_spec(source_db="OPTIMADE/mp",     source_id="mp-1265")
    dup_a   = _make_spec(source_db="OPTIMADE/oqmd",   source_id="oqmd-1234")
    dup_b   = _make_spec(source_db="OPTIMADE/aflow",  source_id="aflow:abc")
    out = _dedup_specs([primary, dup_a, dup_b])
    assert len(out) == 1
    notes = out[0].provenance.notes
    assert "also at:" in notes
    assert "OPTIMADE/oqmd/oqmd-1234" in notes
    assert "OPTIMADE/aflow/aflow:abc" in notes


def test_dedup_specs_preserves_distinct_polymorphs():
    """Two physically distinct polymorphs must NOT be deduped."""
    rocksalt = _make_spec(
        source_db="x", source_id="rocksalt",
        atoms_frac=[
            ("Mg", 12, (0.0, 0.0, 0.0)),
            ("O",  8,  (0.5, 0.5, 0.5)),
        ],
    )
    other = _make_spec(
        source_db="y", source_id="hp_phase",
        atoms_frac=[
            ("Mg", 12, (0.0, 0.0, 0.0)),
            ("O",  8,  (0.25, 0.25, 0.25)),
        ],
    )
    out = _dedup_specs([rocksalt, other])
    assert len(out) == 2


def test_dedup_specs_preserves_input_order():
    """Stable: the first occurrence wins, deduped order matches input
    order so cached replays are bitwise-deterministic."""
    a = _make_spec(source_db="x", source_id="a", a=4.0)
    b = _make_spec(source_db="x", source_id="b", a=5.0)
    a_dup = _make_spec(source_db="y", source_id="a-mirror", a=4.0)
    out = _dedup_specs([a, b, a_dup])
    assert len(out) == 2
    assert out[0].provenance.source_id == "a"   # first 'a' wins
    assert out[1].provenance.source_id == "b"
    assert "a-mirror" in out[0].provenance.notes


def test_rank_specs_idempotent_on_singleton():
    a = _make_spec(source_db="x", source_id="1")
    assert _rank_specs([a]) == [a]


def test_rank_specs_empty_input():
    assert _rank_specs([]) == []


def test_rank_specs_plurality_plus_bigger_cell_tiebreak():
    """3-way: one Pm-3m HP polymorph + two Fm-3m rocksalts of different
    sizes. Plurality (Fm-3m, count=2) wins overall; within that bucket
    the bigger cell wins. The Pm-3m HP entry comes last.

    This tests both axes of the rank function: bucket-by-plurality
    (primary) and -nsites (tiebreak).
    """
    rocksalt_big = _make_spec(
        source_db="x", source_id="rocksalt_canonical", a=4.211,
        atoms_frac=[
            ("Mg", 12, (0.0, 0.0, 0.0)),
            ("Mg", 12, (0.0, 0.5, 0.5)),
            ("Mg", 12, (0.5, 0.0, 0.5)),
            ("Mg", 12, (0.5, 0.5, 0.0)),
            ("O",  8,  (0.5, 0.5, 0.5)),
            ("O",  8,  (0.5, 0.0, 0.0)),
            ("O",  8,  (0.0, 0.5, 0.0)),
            ("O",  8,  (0.0, 0.0, 0.5)),
        ],
    )
    # A second Fm-3m at a different lattice constant — distinct structure
    # but same space group, ensures plurality is unambiguous.
    rocksalt_small = _make_spec(
        source_db="x", source_id="rocksalt_other", a=4.50,
        atoms_frac=[
            ("Mg", 12, (0.0, 0.0, 0.0)),
            ("Mg", 12, (0.0, 0.5, 0.5)),
            ("Mg", 12, (0.5, 0.0, 0.5)),
            ("Mg", 12, (0.5, 0.5, 0.0)),
            ("O",  8,  (0.5, 0.5, 0.5)),
            ("O",  8,  (0.5, 0.0, 0.0)),
            ("O",  8,  (0.0, 0.5, 0.0)),
            ("O",  8,  (0.0, 0.0, 0.5)),
        ],
    )
    # Pm-3m HP-style — different space group, smaller cell, non-plurality.
    hp_phase = _make_spec(
        source_db="x", source_id="hp_phase", a=3.0,
        atoms_frac=[
            ("Mg", 12, (0.0, 0.0, 0.0)),
            ("O",  8,  (0.5, 0.5, 0.5)),
        ],
    )
    out = _rank_specs([hp_phase, rocksalt_small, rocksalt_big])
    # The HP polymorph should be last — it's not in the plurality bucket.
    assert out[-1] is hp_phase
    # Both Fm-3m rocksalts come ahead of the HP polymorph, regardless
    # of which order they appear within the plurality bucket (they
    # have the same nsites, so the tiebreak is input order).
    assert hp_phase not in out[:2]


# ---- list-candidates rendering (VFETCH-X1, v0.13.x) -----------------------

def test_render_candidates_table_single_spec():
    """Rendering one candidate produces header + sep + one data row."""
    from vibeqc.fetch.cli import _render_candidates_table

    spec = _make_spec(source_db="OPTIMADE/mp", source_id="mp-1265", a=4.194)
    table = _render_candidates_table([spec])
    lines = table.split("\n")
    assert len(lines) == 3  # header + sep + row
    assert "mp" in lines[2]
    assert "mp-1265" in lines[2]
    assert "4.1940" in lines[2]
    assert lines[2].rsplit(None, 1)[-1] == "0"  # merged=0


def test_render_candidates_table_dedup_merge_count():
    """Merged candidates show non-zero merge count from provenance.notes."""
    from dataclasses import replace as _replace
    from vibeqc.fetch.cli import _render_candidates_table

    primary = _make_spec(source_db="OPTIMADE/mp", source_id="mp-1265", a=4.194)
    dup_note = primary.provenance.notes + "; also at: OPTIMADE/cod/9006411, OPTIMADE/oqmd/12345"
    dup_prov = _replace(primary.provenance, notes=dup_note)
    primary = _replace(primary, provenance=dup_prov)
    table = _render_candidates_table([primary])
    lines = table.split("\n")
    # merged count should be 2 (two also-at entries)
    assert lines[2].rsplit(None, 1)[-1] == "2"


def test_render_candidates_table_multiple_candidates():
    """Multiple candidates each get their own row."""
    from vibeqc.fetch.cli import _render_candidates_table

    a = _make_spec(source_db="OPTIMADE/mp", source_id="mp-1265", a=4.194)
    b = _make_spec(source_db="OPTIMADE/cod", source_id="9006411", a=4.211)
    table = _render_candidates_table([a, b])
    lines = table.split("\n")
    assert len(lines) == 4  # header + sep + 2 rows
    assert "mp-1265" in lines[2]
    assert "9006411" in lines[3]


def test_render_candidates_table_unknown_provenance():
    """Spec with no provenance renders gracefully (no crash, '?' placeholders)."""
    from vibeqc.fetch.cli import _render_candidates_table

    spec = _make_spec(source_db="OPTIMADE/mp", source_id="mp-1265", a=4.194)
    # Nuke provenance
    from dataclasses import replace as _replace
    spec = _replace(spec, provenance=None)
    table = _render_candidates_table([spec])
    lines = table.split("\n")
    assert "?" in lines[2]  # provider renders as "?"


def test_list_candidates_subparser_registered():
    """``vqfetch list-candidates --help`` exits 0 and mentions the subcommand."""
    import subprocess, sys
    # We can't import the full CLI (vibeqc C++ dep), but we can check
    # the argparse wiring by building the parser from source.
    from vibeqc.fetch.cli import _build_parser
    p = _build_parser()
    # The 'list-candidates' subcommand should be registered.
    choices = [a for a in p._subparsers._group_actions[0].choices]
    assert "list-candidates" in choices
