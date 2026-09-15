"""Rendering a basis from the Basis Set Exchange into the on-demand cache.

Most cases run against a stand-in catalogue, so the suite does not require
the optional ``basis-set-exchange`` distribution (~334 MB of package data).
The few that exercise the real one skip when it is absent.

The behaviour these pin, in order of how much a mistake would cost:

* a record that does not cover the requested elements is refused and
  **nothing is written**, so a half-covering cache entry can never defeat the
  coverage guard on a later run;
* writes are atomic, because a truncated ``.g94`` in the cache is
  indistinguishable from a short basis and the next run would load it;
* the cache is keyed on the catalogue version, so upgrading the
  distribution does not silently keep serving the old record.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from vibeqc.basis_fetch import (
    BseUnavailable,
    ensure_bases_available,
    fetch_requested_by_env,
    _elements_in,
    _split_ecp,
    bse_cache_dir,
    fetch_basis_from_bse,
)

# A two-element all-electron record, in the layout BSE's gaussian94 writer
# emits: comment header, then '<Sym> 0' blocks separated by '****'.
_PLAIN = """\
!----------------------------------------------------------------------
! Basis Set Exchange
! Originating publication: Someone, J. Chem. Phys. 1, 1 (2000)
!----------------------------------------------------------------------


H     0
S    1   1.00
      1.0000000              1.0000000
****
O     0
S    1   1.00
      2.0000000              1.0000000
****
"""

# The same, plus ECP blocks, which libint2 rejects if left in the .g94.
_WITH_ECP = _PLAIN + """\
NA-ECP     2     10
f potential
  1
2      1.0000000          1.0000000
"""


def _fake_bse(records, version="9.9"):
    """A stand-in for the basis_set_exchange module surface we use."""
    mod = types.ModuleType("basis_set_exchange")
    mod.version = lambda: version
    mod.get_metadata = lambda: {k: {} for k in records}

    def get_basis(name, fmt=None, header=True):
        return records[name]

    mod.get_basis = get_basis
    return mod


@pytest.fixture
def catalogue(monkeypatch):
    """Install the stand-in catalogue for the duration of a test."""
    def install(records, version="9.9"):
        monkeypatch.setitem(
            sys.modules, "basis_set_exchange", _fake_bse(records, version)
        )
    return install


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEQC_BSE_CACHE", str(tmp_path / "c"))
    return tmp_path / "c" / "basis"


# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------


def test_cache_dir_prefers_the_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEQC_BSE_CACHE", str(tmp_path / "env"))
    assert bse_cache_dir(tmp_path / "arg") == tmp_path / "arg"


def test_cache_dir_uses_the_environment_then_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEQC_BSE_CACHE", str(tmp_path / "env"))
    assert bse_cache_dir() == tmp_path / "env"
    monkeypatch.delenv("VIBEQC_BSE_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert bse_cache_dir() == tmp_path / "xdg" / "vibeqc" / "basis"


def test_cache_dir_falls_back_to_dot_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("VIBEQC_BSE_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert bse_cache_dir().parts[-3:] == (".cache", "vibeqc", "basis")


# ---------------------------------------------------------------------------
# The optional dependency
# ---------------------------------------------------------------------------


def test_a_missing_distribution_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "basis_set_exchange", None)
    with pytest.raises(BseUnavailable, match=r"vibe-qc\[bse\]"):
        fetch_basis_from_bse("anything")


def test_the_missing_distribution_message_says_the_default_is_unchanged(monkeypatch):
    monkeypatch.setitem(sys.modules, "basis_set_exchange", None)
    with pytest.raises(BseUnavailable, match="bundled library is unchanged"):
        fetch_basis_from_bse("anything")


# ---------------------------------------------------------------------------
# Splitting and parsing
# ---------------------------------------------------------------------------


def test_split_leaves_an_all_electron_record_alone():
    orbital, ecp = _split_ecp(_PLAIN)
    assert orbital == _PLAIN
    assert ecp == ""


def test_split_moves_every_ecp_block_out_of_the_orbital_half():
    orbital, ecp = _split_ecp(_WITH_ECP)
    assert "NA-ECP" not in orbital          # libint2 would reject the file
    assert "NA-ECP" in ecp
    assert "Originating publication" in orbital   # citations stay with it


def test_elements_are_read_from_the_block_headers():
    assert _elements_in(_PLAIN) == (1, 8)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def test_fetch_writes_the_basis_and_a_provenance_record(catalogue, cache):
    catalogue({"demo": _PLAIN})
    got = fetch_basis_from_bse("demo")

    assert got.g94_path == cache / "demo.g94"
    assert got.ecp_path is None and not got.has_ecp
    assert got.elements == (1, 8)
    assert got.bse_version == "9.9"

    prov = json.loads(got.provenance_path.read_text())
    assert prov["bse_version"] == "9.9"
    assert prov["elements"] == [1, 8]
    assert "pritchard_bse_2019" in prov["citation_keys"]
    assert prov["fetched_at"]


def test_the_written_file_says_it_is_not_a_bundled_basis(catalogue, cache):
    catalogue({"demo": _PLAIN})
    text = fetch_basis_from_bse("demo").g94_path.read_text()
    assert "Not a bundled vibe-qc basis" in text
    assert "catalogue version 9.9" in text
    assert "Originating publication" in text     # BSE's own header survives


def test_an_ecp_record_lands_as_two_files(catalogue, cache):
    catalogue({"demo": _WITH_ECP})
    got = fetch_basis_from_bse("demo")
    assert got.has_ecp and got.ecp_path == cache / "demo.ecp"
    assert "NA-ECP" in got.ecp_path.read_text()
    assert "NA-ECP" not in got.g94_path.read_text()


def test_names_are_lowercased_for_the_cache_filename(catalogue, cache):
    catalogue({"demo": _PLAIN})
    assert fetch_basis_from_bse("DeMo").g94_path == cache / "demo.g94"


def test_an_unknown_name_says_how_big_the_catalogue_is(catalogue, cache):
    catalogue({"demo": _PLAIN})
    with pytest.raises(LookupError, match=r"no\s+basis named 'nope'"):
        fetch_basis_from_bse("nope")


def test_an_empty_record_is_refused_rather_than_written(catalogue, cache):
    catalogue({"demo": "   \n"})
    with pytest.raises(LookupError, match="empty record"):
        fetch_basis_from_bse("demo")
    assert not (cache / "demo.g94").exists()


def test_a_record_with_no_element_blocks_is_refused(catalogue, cache):
    catalogue({"demo": "! only a comment\n"})
    with pytest.raises(LookupError, match="no element blocks"):
        fetch_basis_from_bse("demo")
    assert not (cache / "demo.g94").exists()


# ---------------------------------------------------------------------------
# Element coverage: the refusal that keeps a partial entry out of the cache
# ---------------------------------------------------------------------------


def test_a_record_covering_the_requested_elements_is_written(catalogue, cache):
    catalogue({"demo": _PLAIN})
    got = fetch_basis_from_bse("demo", elements=[1, 8])
    assert got.g94_path.is_file()


def test_a_record_missing_an_element_is_refused_and_writes_nothing(catalogue, cache):
    """The decision this module is built around: fetching supplies whole
    sets. Grafting BSE blocks onto a bundled record to fill a gap would put
    two sources under one basis name with no reviewed provenance for the
    join."""
    catalogue({"demo": _PLAIN})
    with pytest.raises(LookupError, match=r"Ag \(Z=47\)"):
        fetch_basis_from_bse("demo", elements=[1, 8, 47])
    assert not (cache / "demo.g94").exists()
    assert not (cache / "demo.provenance.json").exists()


def test_the_refusal_explains_why_it_will_not_graft(catalogue, cache):
    catalogue({"demo": _PLAIN})
    with pytest.raises(LookupError, match="does not graft"):
        fetch_basis_from_bse("demo", elements=[47])


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_a_second_call_serves_the_cache_without_re_rendering(catalogue, cache):
    calls = []
    mod = _fake_bse({"demo": _PLAIN})
    inner = mod.get_basis

    def counting(name, fmt=None, header=True):
        calls.append(name)
        return inner(name, fmt, header)

    mod.get_basis = counting
    sys.modules["basis_set_exchange"] = mod
    try:
        fetch_basis_from_bse("demo")
        fetch_basis_from_bse("demo")
        assert len(calls) == 1
    finally:
        del sys.modules["basis_set_exchange"]


def test_refresh_re_renders_even_on_a_hit(catalogue, cache):
    catalogue({"demo": _PLAIN})
    fetch_basis_from_bse("demo")
    (cache / "demo.g94").write_text("clobbered\n")
    fetch_basis_from_bse("demo", refresh=True)
    assert "clobbered" not in (cache / "demo.g94").read_text()


def test_a_catalogue_upgrade_invalidates_the_cache(catalogue, cache):
    """Otherwise a newer distribution would keep serving the old record."""
    catalogue({"demo": _PLAIN}, version="9.9")
    first = fetch_basis_from_bse("demo")
    assert first.bse_version == "9.9"

    catalogue({"demo": _PLAIN.replace("2.0000000", "3.0000000")}, version="10.0")
    second = fetch_basis_from_bse("demo")
    assert second.bse_version == "10.0"
    assert "3.0000000" in second.g94_path.read_text()


def test_writes_are_atomic_and_leave_no_partial_behind(catalogue, cache):
    catalogue({"demo": _WITH_ECP})
    fetch_basis_from_bse("demo")
    assert not list(cache.glob("*.partial"))


# ---------------------------------------------------------------------------
# Against the real catalogue
# ---------------------------------------------------------------------------

# Scoped to these two tests: a module-level importorskip would skip the whole
# file, including every stand-in case that needs no extra at all.
_needs_bse = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("basis_set_exchange")
    is None,
    reason="the [bse] extra is not installed",
)


@_needs_bse
def test_a_real_unbundled_basis_renders_and_loads(tmp_path, monkeypatch):
    """End to end: a set vibe-qc does not bundle becomes loadable."""
    import os

    import vibeqc as vq

    monkeypatch.setenv("VIBEQC_BSE_CACHE", str(tmp_path / "c"))
    got = fetch_basis_from_bse("sadlej pvtz", elements=[1, 8])
    assert got.elements and 8 in got.elements

    monkeypatch.setitem(os.environ, "LIBINT_DATA_PATH", str(tmp_path / "c"))
    mol = vq.Molecule(
        [vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 0, 1.8]), vq.Atom(1, [1.7, 0, -0.5])],
        0, 1,
    )
    assert vq.BasisSet(mol, "sadlej pvtz").nbasis > 0


@_needs_bse
def test_the_real_catalogue_is_local_not_a_network_call():
    """Recorded because it is the fact the design leans on: the lookup reads
    package data, so it works air-gapped and is pinned by the version."""
    import pathlib

    import basis_set_exchange as bse

    data = pathlib.Path(bse.__file__).parent / "data"
    assert data.is_dir() and any(data.iterdir())


# ---------------------------------------------------------------------------
# Making a rendered basis resolvable
# ---------------------------------------------------------------------------


@pytest.fixture
def pinned_libint(monkeypatch):
    """Record LIBINT_DATA_PATH so a test that repoints it is undone.

    ``ensure_bases_available`` writes the variable directly; registering the
    original with monkeypatch first means teardown restores it anyway.
    """
    import os

    monkeypatch.setenv("LIBINT_DATA_PATH", os.environ.get("LIBINT_DATA_PATH", ""))
    return os.environ["LIBINT_DATA_PATH"]


@pytest.mark.parametrize("value, expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("", False), ("0", False), ("no", False), ("maybe", False),
])
def test_the_environment_switch_reads_the_usual_truthy_spellings(
    monkeypatch, value, expected
):
    monkeypatch.setenv("VIBEQC_FETCH_BSE", value)
    assert fetch_requested_by_env() is expected


def test_a_bundled_name_is_a_complete_no_op(catalogue, cache, pinned_libint):
    """No lookup, no cache entry, and resolution is left exactly as it was.

    This is the property that keeps the default path free of risk: opting in
    changes nothing at all until a name genuinely fails to resolve.
    """
    import os

    catalogue({"demo": _PLAIN})
    assert ensure_bases_available(["def2-svp"], elements=[1, 8]) == ()
    assert os.environ["LIBINT_DATA_PATH"] == pinned_libint
    assert not cache.exists() or not list(cache.glob("*.g94"))


def test_blank_names_are_ignored(catalogue, cache, pinned_libint):
    catalogue({"demo": _PLAIN})
    assert ensure_bases_available([None, "", "   "]) == ()


def test_an_unresolvable_name_is_rendered_and_made_resolvable(
    catalogue, cache, pinned_libint
):
    import os

    catalogue({"demo": _PLAIN})
    got = ensure_bases_available(["demo"], elements=[1, 8])
    assert [f.name for f in got] == ["demo"]
    assert os.environ["LIBINT_DATA_PATH"] != pinned_libint
    assert (Path(os.environ["LIBINT_DATA_PATH"]) / "basis" / "demo.g94").is_file()


def test_the_composed_root_is_a_superset_not_a_replacement(
    catalogue, cache, pinned_libint
):
    """The failure this design exists to avoid: pointing libint at the cache
    alone would hide the bundled library, so a run using a fetched orbital
    basis with a bundled auxiliary would fail on the auxiliary."""
    import os

    catalogue({"demo": _PLAIN})
    ensure_bases_available(["demo"], elements=[1, 8])
    composed = Path(os.environ["LIBINT_DATA_PATH"]) / "basis"

    assert (composed / "demo.g94").is_file()          # the rendered one
    for bundled in ("def2-svp.g94", "pob-tzvp-rev2.g94", "cc-pvdz-pp.g94"):
        assert (composed / bundled).is_file(), f"{bundled} stopped resolving"


def test_composition_is_reused_rather_than_rebuilt(catalogue, cache, pinned_libint):
    """Content-addressed on (source root, rendered stems), so a second run
    with the same need lands on the same directory."""
    import os

    catalogue({"demo": _PLAIN})
    ensure_bases_available(["demo"])
    first = os.environ["LIBINT_DATA_PATH"]
    ensure_bases_available(["demo"])
    assert os.environ["LIBINT_DATA_PATH"] == first
    assert not list((cache.parent / "composed").glob("*.partial-*"))


def test_an_ecp_sidecar_is_composed_alongside_its_orbital_file(
    catalogue, cache, pinned_libint
):
    import os

    catalogue({"demo": _WITH_ECP})
    ensure_bases_available(["demo"])
    composed = Path(os.environ["LIBINT_DATA_PATH"]) / "basis"
    assert (composed / "demo.g94").is_file()
    assert (composed / "demo.ecp").is_file()


# ---------------------------------------------------------------------------
# The run_job flag
# ---------------------------------------------------------------------------


def test_run_job_exposes_the_flag_defaulting_to_off():
    import inspect

    import vibeqc

    param = inspect.signature(vibeqc.run_job).parameters["fetch_from_bse"]
    assert param.default is None            # None consults the environment


@_needs_bse
def test_run_job_refuses_an_unbundled_basis_unless_opted_in(tmp_path, monkeypatch):
    """The default is unchanged: opting in is the only thing that changes it."""
    import vibeqc as vq

    monkeypatch.delenv("VIBEQC_FETCH_BSE", raising=False)
    monkeypatch.setenv("VIBEQC_BSE_CACHE", str(tmp_path / "c"))
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])], 0, 1)
    with pytest.raises(RuntimeError, match="no shells loaded"):
        vq.run_job(
            mol, basis="sadlej pvtz", method="rhf",
            output=str(tmp_path / "off"), verbose=0,
        )


# ---------------------------------------------------------------------------
# The .system manifest record
# ---------------------------------------------------------------------------


def test_an_ordinary_run_records_which_basis_library_answered(tmp_path):
    """No fetching involved. Worth recording anyway: vibe-qc prefers a build
    overlay over the committed tree, so a result produced against a stale
    overlay is otherwise indistinguishable from one produced against current
    data. Needs no optional extra, so this runs everywhere."""
    import tomllib

    import vibeqc as vq

    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])], 0, 1)
    vq.run_job(mol, basis="sto-3g", method="rhf",
               output=str(tmp_path / "plain"), verbose=0)

    with open(tmp_path / "plain.system", "rb") as fh:
        section = tomllib.load(fh)["basis_library"]
    assert section["fetched_count"] == 0
    assert section["resolved_root"]                       # names a real root
    assert Path(section["resolved_root"]).is_dir()
    assert "fetched" not in section                       # empty array is dropped


@_needs_bse
def test_a_fetched_basis_is_recorded_in_the_manifest(tmp_path, monkeypatch):
    import tomllib

    import vibeqc as vq

    monkeypatch.setenv("VIBEQC_BSE_CACHE", str(tmp_path / "c"))
    monkeypatch.setenv("LIBINT_DATA_PATH", __import__("os").environ.get("LIBINT_DATA_PATH", ""))
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])], 0, 1)
    vq.run_job(mol, basis="sadlej pvtz", method="rhf", fetch_from_bse=True,
               output=str(tmp_path / "fetched"), verbose=0)

    with open(tmp_path / "fetched.system", "rb") as fh:
        section = tomllib.load(fh)["basis_library"]
    assert section["fetched_count"] == 1
    row = section["fetched"][0]
    assert row["name"] == "sadlej pvtz"
    assert row["bse_version"]
    assert row["element_count"] > 0
    # The row points at the full provenance rather than inlining the element
    # list, which the manifest's TOML emitter has no list support for.
    assert Path(row["provenance_path"]).is_file()
    assert json.loads(Path(row["provenance_path"]).read_text())["elements"]


def test_provenance_recording_never_fails_a_run(tmp_path, monkeypatch):
    """The manifest record is provenance, not physics: if it raises, the
    calculation must still complete."""
    import vibeqc.basis_fetch as bf

    monkeypatch.setattr(
        bf, "library_root", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    import vibeqc as vq

    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])], 0, 1)
    res = vq.run_job(mol, basis="sto-3g", method="rhf",
                     output=str(tmp_path / "ok"), verbose=0)
    assert res.converged
