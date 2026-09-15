"""``vibeqc.fetch.client_cod`` — offline CIF parsing path.

We pin a small CIF (mirroring the canonical COD 9006411 MgO entry) at
``tests/data/cod/9006411_synthetic.cif`` and route the fetcher through
the ``cif_path=...`` escape hatch so no network is needed.

Live COD round-trips are exercised by the ``--quick`` smoke harness
when the implementer has network reachability. § 11 of the design doc
covers the alternative offline path explicitly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from examples.regression.core.spec import PeriodicSpec, Provenance


HERE = Path(__file__).resolve().parent
FIXTURE_CIF = HERE / "data" / "cod" / "9006411_synthetic.cif"


def test_cif_parses_to_orthorhombic_rocksalt(tmp_path):
    """A synthetic Fm-3m MgO CIF round-trips through ASE → vibe-qc
    PeriodicSpec with the expected 8-atom conventional cell."""
    from vibeqc.fetch.cache import FetchCache
    from vibeqc.fetch.client_cod import fetch_cod

    cache = FetchCache(root=tmp_path / "cache")
    spec = fetch_cod(
        cod_id="9006411_synthetic",
        cif_path=FIXTURE_CIF,
        cache=cache,
        use_cache=False,
        quick=True,
        slug_override="mgo_cod_smoke",
    )
    assert isinstance(spec, PeriodicSpec)
    assert spec.id == "mgo_cod_smoke"
    assert spec.family == "rocksalt"
    assert len(spec.atoms) == 8                     # ASE expands Fm-3m
    # Lattice constant matches the CIF.
    assert spec.lattice_ang[0][0] == pytest.approx(4.211, rel=1e-3)
    # Space group extracted from CIF (loose check — CIF whitespace
    # variation makes a strict match brittle).
    assert "m-3m" in spec.space_group.lower().replace(" ", "")
    # Heuristics fire correctly.
    assert spec.default_initial_guess == "SAD"
    assert spec.default_damping == pytest.approx(0.85)
    assert spec.recommended_basis == "sto-3g"


def test_cod_provenance_carries_doi(tmp_path):
    from vibeqc.fetch.client_cod import fetch_cod

    spec = fetch_cod(
        cod_id="9006411_synthetic",
        cif_path=FIXTURE_CIF,
        use_cache=False,
        quick=True,
    )
    p = spec.provenance
    assert isinstance(p, Provenance)
    assert p.source_db == "COD"
    assert p.source_id == "9006411_synthetic"
    assert p.license == "CC0"
    assert p.original_reference        # CIF has _journal_paper_doi
    assert "crystallography.net" in p.source_url


def test_cod_emit_input_compiles(tmp_path):
    """Generated input script must be valid Python."""
    import py_compile

    from vibeqc.fetch.client_cod import fetch_cod
    from vibeqc.fetch.emit_input import emit_input_script

    spec = fetch_cod(
        cod_id="9006411_synthetic",
        cif_path=FIXTURE_CIF,
        use_cache=False,
        quick=True,
        slug_override="mgo_cod_smoke",
    )
    out = emit_input_script(spec, tmp_path, basis="sto-3g", method="rks-lda")
    py_compile.compile(str(out), doraise=True)


def test_cod_cache_round_trip(tmp_path):
    """Writing through the cache and reading it back yields the same SPEC."""
    from vibeqc.fetch.cache import FetchCache
    from vibeqc.fetch.client_cod import fetch_cod

    cache = FetchCache(root=tmp_path / "cache")
    spec_a = fetch_cod(
        cod_id="9006411_synthetic",
        cif_path=FIXTURE_CIF,
        cache=cache,
        use_cache=True,
        quick=True,
    )
    # Now fetch again — should come from cache, not re-parse the cif_path.
    spec_b = fetch_cod(
        cod_id="9006411_synthetic",
        cache=cache,
        use_cache=True,
        quick=True,
    )
    assert spec_a == spec_b


def test_cod_cache_only_misses_raise(tmp_path):
    """``--cache-only`` against a fresh cache must error rather than fetch."""
    from vibeqc.fetch.cache import FetchCache, FetchCacheMiss
    from vibeqc.fetch.client_cod import fetch_cod

    cache = FetchCache(root=tmp_path / "cache")
    with pytest.raises(FetchCacheMiss):
        fetch_cod(
            cod_id="never_fetched",
            cache=cache,
            use_cache=True,
            cache_only=True,
            quick=True,
        )
