"""Write basis sets directly into vibe-qc's bundled basis library.

The basis library lives at ``python/vibeqc/basis_library/`` and is what
libint reads at runtime.  This module writes :class:`BasisSetData` into
that directory in any supported format (G94, ORCA, NWChem, QVF).

The canonical workflow for adding a new basis set::

    from vibeqc.basis_toolkit import from_bse_file, write_to_library

    data = from_bse_file("my-basis.bse.json")
    write_to_library(data, fmt="g94")   # writes to basis_library/basis/<name>.g94
    write_to_library(data, fmt="qvf_json")  # writes to basis_library/custom/<name>.qvf.json
"""

from __future__ import annotations

from pathlib import Path

from .model import BasisSetData

__all__ = ["write_to_library", "BASIS_LIBRARY_DIR", "library_formats"]


def _find_library_root() -> Path:
    """Locate the basis_library/ directory relative to this package."""
    # basis_toolkit/ lives inside python/vibeqc/, same parent as basis_library/
    return Path(__file__).resolve().parent.parent / "basis_library"


BASIS_LIBRARY_DIR: Path = _find_library_root()
"""Absolute path to ``python/vibeqc/basis_library/``."""


def library_formats() -> list[str]:
    """Return the list of format codes supported by :func:`write_to_library`."""
    return ["g94", "orca", "nwchem", "qvf_json", "qvf"]


def write_to_library(
    data: BasisSetData,
    fmt: str = "g94",
    *,
    subdir: str = "custom",
    filename: str | None = None,
) -> Path:
    """Write *data* into the bundled basis library.

    Parameters
    ----------
    data:
        The basis set to write.
    fmt:
        Output format: ``"g94"``, ``"orca"``, ``"nwchem"``, ``"qvf_json"``,
        or ``"qvf"``.
    subdir:
        Which subdirectory to write to.  Default ``"custom"`` (vibe-qc's
        custom additions).  Use ``"basis"`` to overwrite a libint standard
        set (requires re-running ``setup_basis_library.sh`` afterward).
    filename:
        Output filename (without extension).  Defaults to the lowercased
        basis name with non-alphanumeric chars replaced by hyphens.

    Returns
    -------
    Path
        The path of the written file.

    Raises
    ------
    ValueError
        If *fmt* is not supported.
    """
    ext_map = {
        "g94": ".g94",
        "orca": ".orca",
        "nwchem": ".nwchem",
        "qvf_json": ".qvf.json",
        "qvf": ".qvf",
    }
    ext = ext_map.get(fmt)
    if ext is None:
        raise ValueError(
            f"Unknown format {fmt!r}.  Supported: {', '.join(library_formats())}"
        )

    import re

    safe_name = re.sub(r"[^a-z0-9-]", "-", data.name.lower()).strip("-")
    out_name = filename or safe_name
    # .qvf.json special case: the extension is .qvf.json, not just .json
    out_path = BASIS_LIBRARY_DIR / subdir / f"{out_name}{ext}"

    # Ensure the directory exists.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "g94":
        from .exporter_g94 import to_g94

        out_path.write_text(to_g94(data), encoding="utf-8")
    elif fmt == "orca":
        from .exporter_orca import to_orca

        out_path.write_text(to_orca(data), encoding="utf-8")
    elif fmt == "nwchem":
        from .exporter_nwchem import to_nwchem

        out_path.write_text(to_nwchem(data), encoding="utf-8")
    elif fmt == "qvf_json":
        from .qvf_basis import save_qvf_json

        save_qvf_json(data, out_path)
    elif fmt == "qvf":
        from .qvf_basis import save_qvf

        save_qvf(data, out_path)

    return out_path


# ---------------------------------------------------------------------------
# Convenience: read a basis back from the library
# ---------------------------------------------------------------------------


def read_from_library(
    name: str,
    fmt: str = "g94",
    *,
    subdir: str = "basis",
) -> BasisSetData:
    """Read a basis set from the bundled library.

    Parameters
    ----------
    name:
        Basis set name (matches the filename stem, e.g. ``"sto-3g"``).
    fmt:
        File format to expect (``"g94"``, ``"orca"``, ``"nwchem"``,
        ``"qvf_json"``, ``"qvf"``).
    subdir:
        Subdirectory to read from.  Default ``"basis"`` (the assembled
        runtime directory).

    Returns
    -------
    BasisSetData
    """
    ext_map = {
        "g94": ".g94",
        "orca": ".orca",
        "nwchem": ".nwchem",
        "qvf_json": ".qvf.json",
        "qvf": ".qvf",
    }
    ext = ext_map.get(fmt)
    if ext is None:
        raise ValueError(f"Unknown format {fmt!r}")

    path = BASIS_LIBRARY_DIR / subdir / f"{name}{ext}"

    if fmt == "g94":
        from .importer_g94 import from_g94_file

        return from_g94_file(path, name=name)
    elif fmt == "orca":
        from .importer_orca import from_orca_file

        return from_orca_file(path, name=name)
    elif fmt == "nwchem":
        from .importer_nwchem import from_nwchem_file

        return from_nwchem_file(path, name=name)
    elif fmt in ("qvf_json", "qvf"):
        from .qvf_basis import load_any

        return load_any(path)
    else:
        raise ValueError(f"Unsupported read format: {fmt!r}")
