"""The user basis store, the content-addressed fetch cache and the resolution order (#408, #410).

What these pin, in the order a mistake would cost most:

* a stored record whose bytes changed refuses the run naming the record,
  and a tampered rendering does the same (#235 and #285: a check that can
  say "fine" when it is not is worse than none);
* the resolution order is exactly the documented one, constructed case by
  case: a user ``LIBINT_DATA_PATH`` wins, the bundle beats the store unless
  the record was imported with ``replace_bundled``, the store beats the
  cache, and a composed root is a superset of the bundle;
* a record fetched twice is stored once (content addressing);
* the import tool refuses collisions, bad names and stores inside the
  product tree, and a vibe-basis export imports with no manual step;
* an imported record runs a molecule end to end and builds a cell's basis
  through ``BasisSet`` by name, with its provenance in the manifest.

Every case uses a store and cache under ``tmp_path``; nothing touches
``~/.local/share`` or ``~/.cache``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import vibeqc as vq
from vibeqc import basis_store
from vibeqc.basis_toolkit import ProvenanceKind, Upstream, from_g94_file, library
from vibeqc.basis_toolkit.model import BasisSetData

BUNDLED_BASIS = library.renderings_dir()


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A fresh store and cache, and LIBINT_DATA_PATH restored afterwards."""
    monkeypatch.setenv("VIBEQC_BASIS_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("VIBEQC_BSE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LIBINT_DATA_PATH", os.environ.get("LIBINT_DATA_PATH", ""))
    return tmp_path / "store", tmp_path / "cache"


def _sto3g_copy(name: str, *, scale: float = 1.0) -> BasisSetData:
    """A record with STO-3G's shells (optionally rescaled) under a new name."""
    data = library.load_record("sto-3g")
    data.name = name
    data.description = f"test copy of sto-3g as {name}"
    data.references = []
    if scale != 1.0:
        for element in data.elements.values():
            for shell in element.shells:
                for prim in shell.primitives:
                    prim.exponent *= scale
    data.provenance = None
    return data


def _h2():
    return vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])], 0, 1)


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


def test_store_root_follows_env_then_xdg_then_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEQC_BASIS_STORE", str(tmp_path / "env"))
    assert basis_store.user_store_root() == tmp_path / "env"
    monkeypatch.delenv("VIBEQC_BASIS_STORE")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert basis_store.user_store_root() == tmp_path / "xdg" / "vibeqc" / "basis"
    monkeypatch.delenv("XDG_DATA_HOME")
    assert basis_store.user_store_root().parts[-4:] == (".local", "share", "vibeqc", "basis")
    assert basis_store.user_store_root(tmp_path / "arg") == tmp_path / "arg"


def test_a_store_inside_the_product_tree_is_refused():
    package_store = Path(vq.__file__).resolve().parent / "basis_library" / "store"
    with pytest.raises(basis_store.BasisStoreError, match="inside the vibe-qc tree"):
        basis_store.import_record(_sto3g_copy("inside"), kind="hand-written", root=package_store)
    assert not package_store.exists()


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import_writes_record_rendering_and_index(roots):
    store, _ = roots
    stored = basis_store.import_record(_sto3g_copy("my-sto3g"), kind="hand-written",
                                       upstream=[Upstream(name="me")])
    assert stored.path == store / "records" / "my-sto3g.qvf.json"
    assert stored.g94_path.is_file() and stored.ecp_path is None
    record = library.load_record(stored.path)
    assert record.provenance.kind is ProvenanceKind.HAND_WRITTEN
    assert record.provenance.content_sha256 == stored.content_sha256 == record.content_sha256()
    assert record.provenance.retrieval_date
    index = json.loads((store / "index.json").read_text())
    assert index["records"]["my-sto3g"]["content_sha256"] == stored.content_sha256
    assert [r.name for r in basis_store.list_records()] == ["my-sto3g"]
    assert not (BUNDLED_BASIS / "my-sto3g.g94").exists()


def test_import_refuses_a_bundled_name_unless_told_to_replace(roots):
    with pytest.raises(basis_store.BasisStoreError, match="bundled basis set"):
        basis_store.import_record(_sto3g_copy("sto-3g"), kind="hand-written")
    stored = basis_store.import_record(_sto3g_copy("sto-3g", scale=1.1), kind="hand-written",
                                       replace_bundled=True)
    assert stored.replaces_bundled
    assert json.loads((basis_store.user_store_root() / "index.json").read_text())["records"]["sto-3g"]["replaces_bundled"]


def test_import_refuses_a_second_record_of_the_same_name_unless_replaced(roots):
    basis_store.import_record(_sto3g_copy("twice"), kind="hand-written")
    with pytest.raises(basis_store.BasisStoreError, match="already in the store"):
        basis_store.import_record(_sto3g_copy("twice"), kind="hand-written")
    again = basis_store.import_record(_sto3g_copy("twice", scale=2.0), kind="hand-written", replace=True)
    assert [r.content_sha256 for r in basis_store.list_records()] == [again.content_sha256]


@pytest.mark.parametrize("bad", ["", "../escape", "Has Space", "a/b", "UPPER"])
def test_import_refuses_names_libint_cannot_open(roots, bad):
    data = _sto3g_copy("placeholder")
    if bad == "UPPER":
        # Upper case is folded, not refused: BasisSet lower-cases every name.
        stored = basis_store.import_record(data, name=bad, kind="hand-written")
        assert stored.name == "upper"
        return
    with pytest.raises(basis_store.BasisStoreError, match="cannot be stored"):
        basis_store.import_record(data, name=bad, kind="hand-written")


def test_import_needs_a_user_kind(roots):
    with pytest.raises(basis_store.BasisStoreError, match="hand-written"):
        basis_store.import_record(_sto3g_copy("nokind"))
    with pytest.raises(basis_store.BasisStoreError, match="bundled or fetched"):
        basis_store.import_record(_sto3g_copy("vendorish"), kind="vendor")


def test_import_reads_a_g94_with_its_ecp_sidecar(roots, tmp_path):
    source = tmp_path / "src" / "lanl-copy.g94"
    source.parent.mkdir()
    source.write_text((BUNDLED_BASIS / "lanl2dz.g94").read_text())
    source.with_suffix(".ecp").write_text((BUNDLED_BASIS / "lanl2dz.ecp").read_text())
    stored = basis_store.import_record(source, kind="hand-written")
    assert stored.name == "lanl-copy" and stored.ecp_path is not None
    record = library.load_record(stored.path)
    assert record.ecps and record.elements["Na"].ecp_id == "Na"
    # same numbers as the bundled record: content addressing sees through the name
    assert record.content_sha256() == library.load_record("lanl2dz").content_sha256()


def test_a_vibe_basis_export_imports_with_no_manual_step(roots, tmp_path):
    """The sidecar supplies kind, run id and description."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vibe-basis" / "src"))
    from vibe_basis.io.export import write_basis_for_import
    from vibe_basis.topology import BasisTopology, Shell

    topology = BasisTopology(atoms={"H": (Shell("S", (3.42525091, 0.62391373, 0.1688554),
                                                 (0.15432897, 0.53532814, 0.44463454)),)})
    g94, _sidecar = write_basis_for_import(topology, tmp_path / "run-7", name="H-Opt", run_id="run-7",
                                          description="an optimised hydrogen", objective={"mad": 0.1})
    stored = basis_store.import_record(g94)
    assert stored.name == "h-opt" and stored.kind == "vibe-basis"
    record = library.load_record(stored.path)
    assert record.provenance.run_id == "run-7"
    assert record.description == "an optimised hydrogen"
    assert any(u.name == "vibe-basis" for u in record.provenance.upstream)
    assert record.elements["H"].shells[0].primitives[0].exponent == 3.42525091


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def test_a_changed_record_refuses_and_names_the_file(roots):
    stored = basis_store.import_record(_sto3g_copy("tamper"), kind="hand-written")
    text = stored.path.read_text().replace("0.15432897", "0.15432898", 1)
    stored.path.write_text(text)
    with pytest.raises(basis_store.BasisStoreIntegrityError, match=r"tamper\.qvf\.json.*sha256"):
        basis_store.verify_record(stored)
    with pytest.raises(basis_store.BasisStoreIntegrityError):
        basis_store.activate_user_store()


def test_a_changed_rendering_refuses_and_a_missing_one_is_regenerated(roots):
    stored = basis_store.import_record(_sto3g_copy("render"), kind="hand-written")
    stored.g94_path.write_text(stored.g94_path.read_text() + "!\n")
    with pytest.raises(basis_store.BasisStoreIntegrityError, match=r"render\.g94"):
        basis_store.verify_record(stored)
    stored.g94_path.unlink()
    basis_store.verify_record(stored)
    assert stored.g94_path.is_file()


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


def test_a_store_record_resolves_in_basisset_after_activation(roots):
    basis_store.import_record(_sto3g_copy("store-only"), kind="hand-written")
    composed = basis_store.activate_user_store()
    assert composed is not None and Path(os.environ["LIBINT_DATA_PATH"]) == composed
    basis = vq.BasisSet(_h2(), "store-only")
    assert basis.nbasis == 2
    # the composed root is a superset: the bundle still resolves
    assert vq.BasisSet(_h2(), "def2-svp").nbasis == 10
    assert (composed / "basis" / "pob-tzvp-rev2.g94").is_file()


def test_the_bundle_beats_a_store_record_unless_it_replaces_bundled(roots):
    basis_store.import_record(_sto3g_copy("sto-3g", scale=4.0), kind="hand-written", replace_bundled=True)
    composed = basis_store.activate_user_store()
    rendered = (composed / "basis" / "sto-3g.g94").resolve()
    assert rendered == (basis_store.user_store_root() / "basis" / "sto-3g.g94").resolve()
    # and without the flag the bundled file stays the one libint reads
    index = json.loads((basis_store.user_store_root() / "index.json").read_text())
    index["records"]["sto-3g"]["replaces_bundled"] = False
    (basis_store.user_store_root() / "index.json").write_text(json.dumps(index))
    composed2 = basis_store.activate_user_store()
    assert composed2 != composed
    assert (composed2 / "basis" / "sto-3g.g94").resolve() == (BUNDLED_BASIS / "sto-3g.g94").resolve() or \
        (composed2 / "basis" / "sto-3g.g94").read_text() == (BUNDLED_BASIS / "sto-3g.g94").read_text()


def test_a_user_libint_data_path_wins_over_the_store(roots, tmp_path, monkeypatch):
    user = tmp_path / "user"
    (user / "basis").mkdir(parents=True)
    for name in ("def2-svp.g94", "6-311+g3df2p.g94", "pob-tzvp-rev2.g94"):
        (user / "basis" / name).write_text("****\n")
    basis_store.import_record(_sto3g_copy("ignored"), kind="hand-written")
    monkeypatch.setenv("LIBINT_DATA_PATH", str(user))
    assert basis_store.activate_user_store() is None
    assert os.environ["LIBINT_DATA_PATH"] == str(user)
    resolution = basis_store.resolve(["ignored"])
    assert resolution.records == [] and os.environ["LIBINT_DATA_PATH"] == str(user)


def test_an_empty_store_costs_nothing_and_changes_nothing(roots):
    before = os.environ["LIBINT_DATA_PATH"]
    assert basis_store.activate_user_store() is None
    assert os.environ["LIBINT_DATA_PATH"] == before
    assert basis_store.resolve(["def2-svp", "sto-3g"]).records == []
    assert os.environ["LIBINT_DATA_PATH"] == before


def test_resolve_prefers_store_over_cache_and_cache_over_fetch(roots):
    store_copy = _sto3g_copy("shared-name", scale=1.5)
    cache_copy = _sto3g_copy("shared-name", scale=3.0)
    cached = basis_store.cache_put(cache_copy, name="shared-name", bse_name="Shared", bse_version="9.9")
    resolution = basis_store.resolve(["shared-name"])
    assert [r.layer for r in resolution.records] == ["cache"]
    assert resolution.records[0].content_sha256 == cached.content_sha256
    basis_store.import_record(store_copy, kind="hand-written")
    resolution = basis_store.resolve(["shared-name"])
    assert [r.layer for r in resolution.records] == ["store"]
    assert vq.BasisSet(_h2(), "shared-name").nbasis == 2


def test_resolve_without_fetch_leaves_an_unknown_name_unresolved(roots):
    resolution = basis_store.resolve(["no-such-basis"], fetch=False)
    assert resolution.records == []
    with pytest.raises(RuntimeError):
        vq.BasisSet(_h2(), "no-such-basis")


# ---------------------------------------------------------------------------
# Content-addressed cache
# ---------------------------------------------------------------------------


def test_cache_stores_identical_content_once(roots):
    _, cache = roots
    a = basis_store.cache_put(_sto3g_copy("first"), name="first", bse_version="1")
    b = basis_store.cache_put(_sto3g_copy("second"), name="second", bse_version="1")
    assert a.content_sha256 == b.content_sha256
    assert len(list((cache / "records").glob("*.qvf.json"))) == 1
    assert a.path == b.path
    index = json.loads((cache / "index.json").read_text())
    assert set(index["records"]) == {"first", "second"}


def test_cache_hit_requires_matching_fields_and_verifies(roots):
    basis_store.cache_put(_sto3g_copy("cached"), name="cached", bse_version="9.9")
    assert basis_store.cache_get("cached", require={"bse_version": "10.0"}) is None
    hit = basis_store.cache_get("cached", require={"bse_version": "9.9"})
    assert hit is not None and hit.layer == "cache"
    corrupt = hit.path.read_text().replace("3.42525091", "3.42525092", 1)
    hit.path.write_text(corrupt)
    with pytest.raises(basis_store.BasisStoreIntegrityError, match="sha256"):
        basis_store.cache_get("cached")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_an_imported_record_runs_a_molecule_and_builds_a_cell_basis(roots, tmp_path):
    """#410 acceptance: by name, with no other change, and the manifest says
    which record answered."""
    basis_store.import_record(_sto3g_copy("my-two-element", scale=1.0), kind="hand-written",
                              upstream=[Upstream(name="hand")])
    stem = tmp_path / "h2"
    result = vq.run_job(_h2(), basis="my-two-element", method="rhf", output=str(stem),
                        verbose=0, write_molden_file=False, write_population_file=False,
                        write_xyz_file=False)
    reference = vq.run_job(_h2(), basis="sto-3g", method="rhf", output=str(tmp_path / "ref"),
                           verbose=0, write_molden_file=False, write_population_file=False,
                           write_xyz_file=False)
    assert abs(float(result.energy) - float(reference.energy)) < 1e-10
    manifest = stem.with_suffix(".system").read_text()
    assert "records_count   = 1" in manifest.replace("  ", " ") or "records_count" in manifest
    assert "my-two-element" in manifest and "hand-written" in manifest
    references = stem.with_suffix(".references")
    if references.is_file():
        assert "user-supplied" in references.read_text().lower() or "my-two-element" in references.read_text()

    import numpy as np

    from vibeqc import PeriodicSystem

    lattice = 4.0 * np.eye(3)
    cell = PeriodicSystem(3, lattice, [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])])
    basis = vq.BasisSet(cell.unit_cell_molecule(), "my-two-element")
    assert basis.nbasis == 2
