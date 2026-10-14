"""Read bundled basis sets from, and add records to, the bundled library.

The bundled library is ``python/vibeqc/basis_library/``: ``qvf/`` holds the
QVF-Basis records (the source of truth) and ``basis/`` their renderings
for libint and libecpint (:mod:`.library`). Adding a bundled basis means
adding a record and re-rendering; the renderings are never written by
hand. A user's own basis does not belong here at all: the import tool
puts it in the user's store, where it resolves by name like a bundled one
(#410).

::

    from vibeqc.basis_toolkit import from_bse_file, write_to_library

    data = from_bse_file("my-basis.bse.json")
    data.name = "my-basis"
    write_to_library(data)          # qvf/my-basis.qvf.json + basis/my-basis.{g94,ecp}
"""

from __future__ import annotations

from pathlib import Path

from . import library
from .model import BasisSetData

__all__ = ["write_to_library", "read_from_library", "BASIS_LIBRARY_DIR", "library_formats"]

BASIS_LIBRARY_DIR: Path = library.library_root()
"""Absolute path to ``python/vibeqc/basis_library/``."""


def library_formats() -> list[str]:
    """Formats :func:`read_from_library` serves: the record, or its renderings."""
    return ["qvf_json", "g94", "ecp"]


def write_to_library(
    data: BasisSetData,
    fmt: str = "qvf_json",
    *,
    render: bool = True,
    root: Path | None = None,
) -> Path:
    """Add or replace the bundled record ``data.name`` and re-render the library.

    ``fmt`` must be ``"qvf_json"``: the record is the only thing written by
    hand; ``basis/<name>.g94`` and ``<name>.ecp`` are rendered from it (and
    from every other record, so a stale rendering is refreshed too) when
    ``render`` is true. The record must carry a provenance block with a
    kind, upstream and retrieval date, which the parity test requires of
    every bundled record.

    ``root`` is the library directory (``qvf/`` and ``basis/`` under it);
    the default is the bundled library. Returns the record's path. Raises
    ``ValueError`` for another ``fmt`` or a record without provenance.
    """
    if fmt != "qvf_json":
        raise ValueError(
            f"write_to_library writes records only (fmt='qvf_json'); {fmt!r} "
            "files are rendered from the record by vibeqc.basis_toolkit.library"
        )
    prov = data.provenance
    if prov is None or prov.kind is None or not prov.upstream or not prov.retrieval_date:
        raise ValueError(
            f"record {data.name!r} needs provenance.kind, provenance.upstream and "
            "provenance.retrieval_date before it can join the bundled library"
        )
    path = library.save_record(data, library.record_path(data.name, root))
    if render:
        library.render_library(library.renderings_dir(root), root=root)
    return path


def read_from_library(
    name: str,
    fmt: str = "qvf_json",
    *,
    root: Path | None = None,
) -> BasisSetData | str:
    """Read a bundled basis set.

    ``fmt="qvf_json"`` (the default) returns the record as
    :class:`BasisSetData`; ``"g94"`` and ``"ecp"`` return the rendered text
    libint or libecpint reads (``"ecp"`` raises ``FileNotFoundError`` for a
    record without potentials).
    """
    if fmt == "qvf_json":
        return library.load_record(name, root)
    if fmt in ("g94", "ecp"):
        path = library.renderings_dir(root) / f"{name}.{fmt}"
        if not path.is_file():
            raise FileNotFoundError(f"no {fmt} rendering for bundled basis {name!r}")
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unknown format {fmt!r}. Supported: {', '.join(library_formats())}")
