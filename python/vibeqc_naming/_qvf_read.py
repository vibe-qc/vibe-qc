"""Minimal QVF structure reader — enough for IUPAC naming, nothing more.

vibe-qc writes QVF; this reads back the one section naming needs. It is
deliberately minimal and lives here, in a package that imports only the
standard library and numpy, so `vibeqc_naming` keeps working without the
compiled `_vibeqc_core` extension and without vibe-view installed.

Rich consumption of a .qvf archive is vibe-view's job. The normative spec,
the JSON schema and the reference implementation live in the `qvf`
repository; vibe-qc implements the format rather than depending on that
toolkit, and is validated against its conformance corpus.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


class QVFReadError(Exception):
    """A .qvf archive is unreadable, or its declared digests do not match."""


@dataclass(frozen=True)
class QVFAtom:
    symbol: str
    atomic_number: int
    position: np.ndarray  # (3,) float64, angstrom (QVF structure convention)
    occupancy: float = 1.0


@dataclass(frozen=True)
class QVFStructure:
    atoms: list[QVFAtom]
    pbc: tuple[bool, bool, bool]
    lattice_vectors: Optional[np.ndarray]  # (3,3) or None


class QVFReader:
    """Minimal reader for the structure section of a .qvf archive."""

    def __init__(self, path):
        self._path = Path(path)
        try:
            self._zf = zipfile.ZipFile(self._path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise QVFReadError(f"cannot open {self._path}: {exc}") from None
        try:
            self.manifest: dict[str, Any] = json.loads(
                self._zf.read("manifest.json").decode("utf-8")
            )
        except KeyError:
            self.close()
            raise QVFReadError(f"{self._path} has no manifest.json") from None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.close()
            raise QVFReadError(f"{self._path}: manifest.json is not valid JSON: {exc}") from None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        zf = getattr(self, "_zf", None)
        if zf is not None:
            zf.close()

    def _member_bytes(self, section_id: str, member_name: str) -> bytes:
        for section in self.manifest.get("sections", []):
            if section.get("id") != section_id:
                continue
            member = (section.get("members") or {}).get(member_name)
            if member is None:
                raise QVFReadError(
                    f"section {section_id!r} has no member {member_name!r}"
                )
            try:
                data = self._zf.read(member["path"])
            except KeyError:
                raise QVFReadError(
                    f"member {member['path']!r} is declared in the manifest "
                    f"but absent from the archive"
                ) from None
            declared = member.get("sha256")
            if declared is not None:
                actual = hashlib.sha256(data).hexdigest()
                if actual != declared:
                    raise QVFReadError(
                        f"sha256 mismatch for {member['path']!r}: "
                        f"manifest declares {declared}, archive contains {actual}"
                    )
            return data
        raise QVFReadError(f"no section with id {section_id!r}")

    def read_structure(self, section_id: str = "structure") -> QVFStructure:
        raw = self._member_bytes(section_id, "structure")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QVFReadError(f"structure member is not valid JSON: {exc}") from None
        atoms = [
            QVFAtom(
                symbol=str(a["symbol"]),
                atomic_number=int(a["atomic_number"]),
                position=np.asarray(a["position"], dtype=np.float64),
                occupancy=float(a.get("occupancy", 1.0)),
            )
            for a in payload.get("atoms", [])
        ]
        pbc_raw = payload.get("pbc") or (False, False, False)
        lattice = payload.get("lattice_vectors")
        return QVFStructure(
            atoms=atoms,
            pbc=tuple(bool(x) for x in pbc_raw),
            lattice_vectors=(
                np.asarray(lattice, dtype=np.float64) if lattice is not None else None
            ),
        )
