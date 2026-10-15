"""The user's basis store and the content-addressed fetch cache (#408, #410).

One store, one namespace: a hand-written set, a vibe-basis product and a
set rendered from the Basis Set Exchange all resolve by name through the
same door as a bundled one, and nothing downstream branches on what kind
of record answered (#295 decision 2). This module is that door.

Resolution order (documented in ``docs/user_guide/basis_sets.md``; the
test ``tests/test_basis_store.py`` constructs each case):

1. ``LIBINT_DATA_PATH`` set by the user with a complete library wins, and
   nothing below is consulted.
2. The bundled library: the verified ``build/basis_library`` overlay when
   present, else the package copy (``vibeqc._install_basis_library``).
3. The user store, :func:`user_store_root`: records imported with
   ``vibe-qc-basis import``. A user record never shadows a bundled name
   unless it was imported with ``--replace-bundled``, which is recorded in
   the store index and makes that record win for that name.
4. The fetch cache, :func:`fetch_cache_root`: records rendered from the
   Basis Set Exchange on demand, consulted for a name neither the bundle
   nor the store resolves.
5. Fetching from the catalogue, opt-in (``run_job(fetch_from_bse=True)``
   or ``VIBEQC_FETCH_BSE=1``), which writes layer 4.

Layers 3 and 4 are made visible to libint by composing a directory of
symlinks over the bundled library (:func:`compose`) and pointing
``LIBINT_DATA_PATH`` at it; a composed root is a strict superset of the
bundled one, so a fetched orbital basis with a bundled auxiliary keeps
working. Both layers live outside every product checkout and are never
written by anything under ``python/vibeqc/basis_library/``.

Integrity: a record is identified by :meth:`BasisSetData.content_sha256`.
Before a stored record is used, its file is re-hashed and its rendering is
compared byte for byte with what the record renders to. A mismatch raises
:class:`BasisStoreIntegrityError` naming the file and nothing is
substituted silently (#235 and #285 are the lessons: a check that can
answer "fine" when it is not is worse than none).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .basis_toolkit import library as _library
from .basis_toolkit.model import (
    BasisSetData,
    CitationRoute,
    Provenance,
    ProvenanceKind,
    Upstream,
)

__all__ = [
    "BasisStoreError",
    "BasisStoreIntegrityError",
    "Resolution",
    "StoredRecord",
    "activate_user_store",
    "cache_get",
    "cache_put",
    "compose",
    "fetch_cache_root",
    "import_record",
    "list_records",
    "load_source",
    "remove_record",
    "resolve",
    "user_store_root",
    "verify_record",
]

_INDEX = "index.json"
_RECORDS = "records"
_RENDER = "render"
_BASIS = "basis"
_COMPOSED = "composed"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9+()*,._-]*$")
_PROVENANCE_SIDECAR = ".provenance.json"


class BasisStoreError(RuntimeError):
    """A store operation was refused (collision, bad name, bad source)."""


class BasisStoreIntegrityError(BasisStoreError):
    """A stored record or its rendering does not match its recorded identity."""

    exit_status = 78


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


def user_store_root(root: Optional[Path | str] = None) -> Path:
    """Where imported records live.

    ``$VIBEQC_BASIS_STORE`` first, then ``$XDG_DATA_HOME/vibeqc/basis``,
    then ``~/.local/share/vibeqc/basis``. Data, not cache: an imported
    record is the user's own work.
    """
    if root is not None:
        return Path(root).expanduser()
    if override := os.environ.get("VIBEQC_BASIS_STORE"):
        return Path(override).expanduser()
    if xdg := os.environ.get("XDG_DATA_HOME"):
        return Path(xdg).expanduser() / "vibeqc" / "basis"
    return Path("~/.local/share/vibeqc/basis").expanduser()


def fetch_cache_root(root: Optional[Path | str] = None) -> Path:
    """Where fetched records are cached: :func:`vibeqc.basis_fetch.bse_cache_dir`."""
    from .basis_fetch import bse_cache_dir

    return bse_cache_dir(root)


def _refuse_inside_product_tree(path: Path, what: str) -> None:
    """A store inside the package or its checkout is not a store (#295)."""
    package = Path(__file__).resolve().parent
    checkout = package.parent.parent
    resolved = Path(path).expanduser().resolve()
    for forbidden in (package, checkout):
        if resolved == forbidden or forbidden in resolved.parents:
            raise BasisStoreError(
                f"{what} {path} lies inside the vibe-qc tree {forbidden}; a basis "
                "store belongs under the user's data or cache directory, never in "
                "a product checkout or the installed package"
            )


# ---------------------------------------------------------------------------
# Records and the index
# ---------------------------------------------------------------------------


@dataclass
class StoredRecord:
    """One record in a store layer and where its rendering is."""

    name: str
    kind: str
    content_sha256: str
    path: Path
    rendering_dir: Path
    layer: str  # "store" or "cache"
    replaces_bundled: bool = False
    recorded_at: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def g94_path(self) -> Path:
        return self.rendering_dir / f"{self.name}.g94"

    @property
    def ecp_path(self) -> Optional[Path]:
        path = self.rendering_dir / f"{self.name}.ecp"
        return path if path.is_file() else None

    def manifest_row(self) -> dict[str, object]:
        """Flat scalar row for the ``.system`` manifest's ``[[basis_library.records]]``."""
        return {
            "name": self.name,
            "kind": self.kind,
            "layer": self.layer,
            "content_sha256": self.content_sha256,
            "replaces_bundled": bool(self.replaces_bundled),
            "record_path": str(self.path),
        }


def _load_index(root: Path) -> dict:
    path = root / _INDEX
    if not path.is_file():
        return {"version": 1, "records": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BasisStoreIntegrityError(f"{path}: unreadable store index ({exc})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("records"), dict):
        raise BasisStoreIntegrityError(f"{path}: malformed store index")
    return data


def _save_index(root: Path, index: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (_INDEX + ".partial")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, root / _INDEX)


def _write_rendering(data: BasisSetData, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in _library.render_record(data).items():
        tmp = directory / (name + ".partial")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, directory / name)


def verify_record(stored: StoredRecord) -> BasisSetData:
    """Load ``stored``, prove its identity and its rendering, return it.

    The record file must hash to ``stored.content_sha256`` and every
    rendered file must equal what the record renders to. A missing
    rendering is written; a differing one is a hard error naming the file.
    """
    if not stored.path.is_file():
        raise BasisStoreIntegrityError(f"{stored.path}: record file is missing")
    try:
        data = _library.load_record(stored.path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BasisStoreIntegrityError(f"{stored.path}: record does not parse ({exc})") from exc
    actual = data.content_sha256()
    if actual != stored.content_sha256:
        raise BasisStoreIntegrityError(
            f"{stored.path}: content sha256 {actual} does not match the recorded "
            f"{stored.content_sha256}; the record was altered after it was stored. "
            "Remove it and import or fetch it again."
        )
    if stored.layer == "store" and data.name != stored.name:
        raise BasisStoreIntegrityError(
            f"{stored.path}: record is named {data.name!r} but is stored as {stored.name!r}"
        )
    data.name = stored.name  # a cache record may serve several names; render under this one
    expected = _library.render_record(data)
    missing = [n for n in expected if not (stored.rendering_dir / n).is_file()]
    if missing:
        _write_rendering(data, stored.rendering_dir)
    for name, text in expected.items():
        on_disk = stored.rendering_dir / name
        if on_disk.read_text(encoding="utf-8") != text:
            raise BasisStoreIntegrityError(
                f"{on_disk}: rendering differs from the record {stored.path}; "
                "the rendered text was altered after it was written. Remove the "
                "rendering and it is regenerated from the record."
            )
    return data


# ---------------------------------------------------------------------------
# The user store
# ---------------------------------------------------------------------------


def _store_record(name: str, entry: dict, root: Path) -> StoredRecord:
    return StoredRecord(
        name=name,
        kind=str(entry.get("kind", "")),
        content_sha256=str(entry.get("content_sha256", "")),
        path=root / _RECORDS / f"{name}{_library.RECORD_SUFFIX}",
        rendering_dir=root / _BASIS,
        layer="store",
        replaces_bundled=bool(entry.get("replaces_bundled", False)),
        recorded_at=str(entry.get("imported_at", "")),
        extra={k: v for k, v in entry.items()
               if k not in ("kind", "content_sha256", "replaces_bundled", "imported_at")},
    )


def list_records(root: Optional[Path | str] = None) -> list[StoredRecord]:
    """Every record in the user store, sorted by name."""
    store = user_store_root(root)
    index = _load_index(store)
    return [_store_record(name, entry, store) for name, entry in sorted(index["records"].items())]


def _bundled_names() -> set[str]:
    return set(_library.stems())


def _normalise_name(name: str) -> str:
    text = str(name or "").strip().lower()
    if not _NAME_RE.match(text) or text in (".", "..") or "/" in text or "\\" in text:
        raise BasisStoreError(
            f"basis name {name!r} cannot be stored: use lower-case letters, digits "
            "and the punctuation libint accepts (+ ( ) * , . _ -)"
        )
    return text


def load_source(source: Path | str, *, name: Optional[str] = None) -> tuple[BasisSetData, dict]:
    """Read a basis from any supported file and return ``(record, sidecar)``.

    Accepts ``.qvf.json`` / ``.qvf``, Gaussian94 ``.g94`` (an ``.ecp`` of
    the same stem beside it is merged), BSE ``.json``, a CRYSTAL per-element
    file, or a directory of CRYSTAL per-element files. ``sidecar`` is the
    content of ``<stem>.provenance.json`` next to the source when present
    (what :mod:`vibe_basis.io.export` writes), else ``{}``.
    """
    from .basis_toolkit import from_bse_file, from_crystal_file, from_g94, from_g94_file
    from .basis_toolkit.qvf_basis import load_any

    path = Path(source).expanduser()
    if not path.exists():
        raise BasisStoreError(f"basis source {path} does not exist")
    sidecar_path = path.with_name(path.stem + _PROVENANCE_SIDECAR) if path.is_file() else path / ("provenance" + ".json")
    if path.name.endswith(".qvf.json") or path.suffix == ".qvf":
        data = load_any(path)
    elif path.is_dir():
        parts = sorted(p for p in path.iterdir() if p.is_file() and re.match(r"^\d+_", p.name))
        if not parts:
            raise BasisStoreError(f"{path}: no CRYSTAL per-element files (NN_Sym) found")
        data = None
        for part in parts:
            atom = from_crystal_file(part, name=name or path.name)
            if data is None:
                data = atom
            else:
                data.elements.update(atom.elements)
                if atom.ecps:
                    merged = dict(data.ecps or {})
                    merged.update(atom.ecps)
                    data.ecps = merged
        for sym, element in data.elements.items():
            element.ecp_id = sym if sym in (data.ecps or {}) else None
    elif path.suffix.lower() == ".g94":
        data = from_g94_file(path, name=name or path.stem)
        ecp = path.with_suffix(".ecp")
        if ecp.is_file():
            sidecar_ecp = from_g94(ecp.read_text(encoding="utf-8"), name=data.name, role="ecp")
            merged = dict(data.ecps or {})
            merged.update(sidecar_ecp.ecps or {})
            data.ecps = merged or None
            for sym, element in data.elements.items():
                element.ecp_id = sym if sym in merged else None
    elif path.suffix.lower() == ".json":
        data = from_bse_file(path)
        if name:
            data.name = name
    else:
        try:
            data = from_crystal_file(path, name=name or path.stem)
        except Exception as exc:  # noqa: BLE001 - the message names the file
            raise BasisStoreError(f"{path}: not a basis file this tool reads ({exc})") from exc
    sidecar: dict = {}
    if sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise BasisStoreError(f"{sidecar_path}: provenance sidecar is not JSON ({exc})") from exc
    return data, sidecar


def import_record(
    source: Path | str | BasisSetData,
    *,
    name: Optional[str] = None,
    kind: Optional[ProvenanceKind | str] = None,
    replace_bundled: bool = False,
    replace: bool = False,
    description: Optional[str] = None,
    upstream: Sequence[Upstream] = (),
    citation_keys: Sequence[str] = (),
    run_id: Optional[str] = None,
    notes: Optional[str] = None,
    root: Optional[Path | str] = None,
) -> StoredRecord:
    """Put a basis set into the user store so it resolves by name (#410).

    ``source`` is a file :func:`load_source` reads, or a record. ``kind``
    must be ``hand-written`` or ``vibe-basis`` (a provenance sidecar next to
    the source may supply it, with ``run_id`` and ``description``). The
    import refuses a name that is bundled unless ``replace_bundled`` is
    set, and a name already in the store unless ``replace`` is set.
    Nothing under ``python/vibeqc/basis_library/`` is written.
    """
    store = user_store_root(root)
    _refuse_inside_product_tree(store, "basis store")
    sidecar: dict = {}
    if isinstance(source, BasisSetData):
        data = source
    else:
        data, sidecar = load_source(source, name=name)
    # An explicit name, however empty, is checked as given; only ``None``
    # falls back to the sidecar's or the record's own name.
    stem = _normalise_name(name if name is not None else (sidecar.get("name") or data.name))
    kind_value = kind or sidecar.get("kind")
    if kind_value is None:
        raise BasisStoreError(
            f"{stem}: say what this record is with kind='hand-written' or 'vibe-basis'"
        )
    kind_enum = ProvenanceKind(str(kind_value))
    if kind_enum not in (ProvenanceKind.HAND_WRITTEN, ProvenanceKind.VIBE_BASIS):
        raise BasisStoreError(
            f"{stem}: a user store holds hand-written and vibe-basis records; "
            f"{kind_enum.value!r} records are bundled or fetched"
        )
    if stem in _bundled_names() and not replace_bundled:
        raise BasisStoreError(
            f"{stem!r} is a bundled basis set. Import under another name, or pass "
            "replace_bundled=True (--replace-bundled) to make your record win for "
            "this name; the choice is recorded in the store index."
        )
    index = _load_index(store)
    if stem in index["records"] and not replace:
        raise BasisStoreError(
            f"{stem!r} is already in the store at {store}. Pass replace=True "
            "(--replace) to overwrite it, or import under another name."
        )

    data.name = stem
    placeholder = not data.description.strip() or data.description.startswith("Imported from ")
    if description is not None:
        data.description = description
    elif sidecar.get("description") and placeholder:
        data.description = str(sidecar["description"])
    elif placeholder:
        data.description = ""
    prov = data.provenance or Provenance(origin="manual")
    prov.kind = kind_enum
    prov.upstream = list(prov.upstream) + list(upstream)
    for up in sidecar.get("upstream", []) or []:
        prov.upstream.append(Upstream.from_dict(up))
    if citation_keys:
        prov.citations.append(CitationRoute(keys=list(citation_keys)))
    prov.run_id = run_id or sidecar.get("run_id") or prov.run_id
    if notes:
        prov.notes = notes
    if not prov.retrieval_date:
        prov.retrieval_date = str(sidecar.get("generated_at", "") or "")[:10] or _dt.date.today().isoformat()
    data.provenance = prov

    record_path = store / _RECORDS / f"{stem}{_library.RECORD_SUFFIX}"
    _library.save_record(data, record_path)
    _write_rendering(data, store / _BASIS)
    index["records"][stem] = {
        "kind": kind_enum.value,
        "content_sha256": data.content_sha256(),
        "replaces_bundled": bool(replace_bundled),
        "imported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "run_id": prov.run_id or "",
    }
    _save_index(store, index)
    return _store_record(stem, index["records"][stem], store)


def remove_record(name: str, *, root: Optional[Path | str] = None) -> bool:
    """Drop ``name`` from the user store; returns whether it was there."""
    store = user_store_root(root)
    index = _load_index(store)
    stem = _normalise_name(name)
    if stem not in index["records"]:
        return False
    del index["records"][stem]
    _save_index(store, index)
    for path in (store / _RECORDS / f"{stem}{_library.RECORD_SUFFIX}",
                 store / _BASIS / f"{stem}.g94", store / _BASIS / f"{stem}.ecp"):
        path.unlink(missing_ok=True)
    return True


# ---------------------------------------------------------------------------
# The content-addressed fetch cache
# ---------------------------------------------------------------------------


def _cache_record(name: str, entry: dict, root: Path) -> StoredRecord:
    sha = str(entry.get("content_sha256", ""))
    return StoredRecord(
        name=name,
        kind=str(entry.get("kind", "bse")),
        content_sha256=sha,
        path=root / _RECORDS / f"{sha}{_library.RECORD_SUFFIX}",
        rendering_dir=root / _RENDER / sha / _BASIS,
        layer="cache",
        recorded_at=str(entry.get("fetched_at", "")),
        extra={k: v for k, v in entry.items()
               if k not in ("kind", "content_sha256", "fetched_at")},
    )


def cache_put(
    data: BasisSetData,
    *,
    name: str,
    root: Optional[Path | str] = None,
    **index_fields: object,
) -> StoredRecord:
    """Store ``data`` by its content hash and point ``name`` at it.

    A record fetched twice is stored once: the record file is named by its
    content hash. ``index_fields`` (``bse_name``, ``bse_version``, ...) are
    kept in the name index for cache-hit decisions and the manifest.
    """
    cache = fetch_cache_root(root)
    _refuse_inside_product_tree(cache, "basis fetch cache")
    stem = _normalise_name(name)
    data.name = stem
    sha = data.content_sha256()
    record_path = cache / _RECORDS / f"{sha}{_library.RECORD_SUFFIX}"
    if not record_path.is_file():
        # Named by content, not by basis name, so ``save_record``'s
        # name-equals-stem rule does not apply; same content under a second
        # name shares the file and the index tells the names apart.
        data.schema_version = _library.SCHEMA_VERSION
        if data.provenance is not None:
            data.provenance.content_sha256 = sha
        record_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = record_path.with_name(record_path.name + ".partial")
        tmp.write_text(_library.record_json(data), encoding="utf-8")
        os.replace(tmp, record_path)
    rendering = cache / _RENDER / sha / _BASIS
    if not (rendering / f"{stem}.g94").is_file():
        _write_rendering(data, rendering)
    index = _load_index(cache)
    kind = data.provenance.kind.value if data.provenance and data.provenance.kind else "bse"
    entry = {
        "kind": kind,
        "content_sha256": sha,
        "fetched_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    entry.update({k: v for k, v in index_fields.items() if v is not None})
    index["records"][stem] = entry
    _save_index(cache, index)
    return _cache_record(stem, entry, cache)


def cache_get(
    name: str,
    *,
    root: Optional[Path | str] = None,
    require: Optional[dict] = None,
) -> Optional[StoredRecord]:
    """The cached record for ``name``, verified, or ``None``.

    ``require`` is a dict of index fields that must match (for instance
    ``{"bse_version": "0.12"}``); a mismatch is a miss, not an error. A hit
    whose record or rendering fails :func:`verify_record` is an error.
    """
    cache = fetch_cache_root(root)
    index = _load_index(cache)
    entry = index["records"].get(_normalise_name(name))
    if entry is None:
        return None
    for key, value in (require or {}).items():
        if entry.get(key) != value:
            return None
    stored = _cache_record(_normalise_name(name), entry, cache)
    verify_record(stored)
    return stored


# ---------------------------------------------------------------------------
# Composition and resolution
# ---------------------------------------------------------------------------


def _bundled_root() -> Optional[Path]:
    """Layer 2: the overlay or package copy, whatever ``LIBINT_DATA_PATH`` says.

    Composition must start from the bundle itself; starting from the root
    currently selected would fold an earlier composition's links (and the
    store records they point at) into the next one.
    """
    from . import _resolve_basis_library

    return _resolve_basis_library()


def compose(
    source_root: Path,
    records: Sequence[StoredRecord],
    *,
    root: Path,
) -> Path:
    """A libint root carrying ``source_root`` plus every record's rendering.

    Symlinks, content-addressed on the source root and the records' names
    and hashes, so repeated resolutions reuse one directory. A record wins
    over a same-named bundled file only when it ``replaces_bundled``;
    otherwise the bundled file stays and the record is skipped (the import
    tool refuses that collision up front, so it only arises when a bundled
    set gains a name a store already has).
    """
    key_material = [str(Path(source_root).resolve())] + sorted(
        f"{r.layer}:{r.name}:{r.content_sha256}:{int(r.replaces_bundled)}" for r in records
    )
    key = hashlib.sha256("\0".join(key_material).encode()).hexdigest()[:16]
    target = Path(root) / _COMPOSED / key
    if (target / _BASIS).is_dir():
        return target
    tmp = target.with_name(f"{target.name}.partial-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / _BASIS).mkdir(parents=True)
    bundled_files = {p.name: p for p in (Path(source_root) / _BASIS).iterdir() if p.is_file()}
    for name, path in bundled_files.items():
        (tmp / _BASIS / name).symlink_to(path.resolve())
    for record in records:
        for rendered in (record.g94_path, record.ecp_path):
            if rendered is None:
                continue
            link = tmp / _BASIS / rendered.name
            if rendered.name in bundled_files and not record.replaces_bundled:
                continue
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(rendered.resolve())
    try:
        tmp.replace(target)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
    return target


@dataclass
class Resolution:
    """What :func:`resolve` did: the root libint reads and the records it added."""

    root: Path
    records: list[StoredRecord]
    fetched: tuple = ()

    @property
    def changed(self) -> bool:
        return bool(self.records)


def _user_env_pinned() -> bool:
    """True when the user set a complete LIBINT_DATA_PATH of their own (layer 1)."""
    from . import _basis_library_has_standard_payload, _resolve_basis_library

    current = os.environ.get("LIBINT_DATA_PATH")
    if not current:
        return False
    chosen = _resolve_basis_library()
    if chosen is not None and Path(current).resolve() == chosen.resolve():
        return False
    composed_roots = (user_store_root() / _COMPOSED, fetch_cache_root() / _COMPOSED)
    current_path = Path(current).resolve()
    if any(parent.resolve() in current_path.parents for parent in composed_roots if parent.exists()):
        return False
    return _basis_library_has_standard_payload(current)


def activate_user_store(*, root: Optional[Path | str] = None) -> Optional[Path]:
    """Make the user store's records resolvable (layer 3), if there are any.

    Called at import when the user did not pin ``LIBINT_DATA_PATH``. Costs
    one ``stat`` when the store has no index; otherwise every record is
    verified and a composed root is selected. Returns the composed root or
    ``None`` when nothing changed.
    """
    store = user_store_root(root)
    if not (store / _INDEX).is_file():
        return None
    records = list_records(root)
    if not records:
        return None
    if _user_env_pinned():
        return None
    source = _bundled_root()
    if source is None:
        return None
    for record in records:
        verify_record(record)
    composed = compose(source, records, root=store)
    os.environ["LIBINT_DATA_PATH"] = str(composed)
    return composed


def resolve(
    names: Iterable[Optional[str]],
    *,
    elements: Optional[Iterable[int]] = None,
    fetch: bool = False,
    root: Optional[Path | str] = None,
    cache_root: Optional[Path | str] = None,
) -> Resolution:
    """Resolve ``names`` through layers 2 to 5 and point libint at the result.

    A name the bundled library or an active user store already answers is
    left alone. A name neither answers is looked up in the fetch cache,
    and, with ``fetch``, rendered from the Basis Set Exchange into it.
    Every store or cache record used is verified first. The returned
    :class:`Resolution` lists the records added for the run's manifest.
    """
    from .basis_registry import basis_file_path

    wanted: list[str] = []
    for name in names:
        text = str(name or "").strip()
        if text and text not in wanted:
            wanted.append(text)
    store_records = list_records(root) if (user_store_root(root) / _INDEX).is_file() else []
    used: list[StoredRecord] = []
    fetched: list = []
    pinned = _user_env_pinned()
    if store_records and not pinned:
        for record in store_records:
            verify_record(record)
        used.extend(store_records)
    for name in wanted:
        if basis_file_path(name) is not None and not any(
            r.replaces_bundled and r.name == _normalise_name(name) for r in used
        ):
            continue
        if any(r.name == _normalise_name(name) for r in used):
            continue
        hit = cache_get(name, root=cache_root)
        if hit is None and fetch:
            from .basis_fetch import fetch_basis_from_bse

            got = fetch_basis_from_bse(name, elements=elements, cache_dir=cache_root)
            fetched.append(got)
            hit = cache_get(name, root=cache_root)
        if hit is not None:
            used.append(hit)
    source = _bundled_root()
    if not used or source is None or pinned:
        return Resolution(root=Path(os.environ.get("LIBINT_DATA_PATH", "")), records=[], fetched=tuple(fetched))
    compose_root = fetch_cache_root(cache_root) if any(r.layer == "cache" for r in used) else user_store_root(root)
    composed = compose(source, used, root=compose_root)
    os.environ["LIBINT_DATA_PATH"] = str(composed)
    return Resolution(root=composed, records=used, fetched=tuple(fetched))
