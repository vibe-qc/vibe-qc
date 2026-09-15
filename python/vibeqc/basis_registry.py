"""One registry for basis-set metadata.

The bundled library answers three questions that used to be answered in
four different places, each with its own name normalisation and its own
notion of what a basis is:

* **Which auxiliary basis should a job default to?**  The molecular
  density-fitting module and the periodic auxiliary module each kept a
  table; the two disagreed (``def2-qzvp`` mapped to two different JK fits,
  ``pob-*`` refused in one and silently substituted in the other).
* **Does this basis need an effective core potential on this atom?**  A
  name pattern (``^dhf-``, ``-pp$``, ``^lanl`` ...) decided, so a light
  molecule in LANL2DZ was refused although LANL2DZ is all-electron there,
  ``x2c-TZVPall`` was refused although it is all-electron everywhere, and
  the valence-only ``pob-tzvp-rev2`` / ``def2-m*`` blocks beyond Kr passed
  because their names carry no such marker.
* **Which elements does the file actually cover?**  Nothing asked; libint
  hands back an atom with zero shells.

This module reads ``basis_library/registry.toml`` for the curated part
(default fits, family rules) and the ``.g94`` / ``.ecp`` files themselves
for the factual part (element coverage, per-element core counts), and
exposes one set of functions every consumer goes through.  The ECP
decision is *per element*: a basis "replaces core" on an atom when its
sidecar carries a positive-core block for that element, or, for a family
that is valence-only beyond a threshold but ships no sidecar, when the
atom lies beyond that threshold (in which case there is no data to attach
and the guard must refuse).
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = [
    "BasisRecord",
    "EcpRequirement",
    "basis_file_path",
    "basis_replaces_core",
    "canonical_basis_name",
    "default_aux_basis",
    "ecp_requirements",
    "element_coverage",
    "lookup",
    "sidecar_core_electrons",
]

_REGISTRY_PATH = Path(__file__).resolve().parent / "basis_library" / "registry.toml"
_BUNDLED_BASIS_DIR = Path(__file__).resolve().parent / "basis_library" / "basis"

_ELEMENT_SYMBOLS = (
    "X",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
)
_SYMBOL_TO_Z = {s.lower(): z for z, s in enumerate(_ELEMENT_SYMBOLS)}


def element_symbol(z: int) -> str:
    """Symbol for ``z``; ``"Z<n>"`` beyond the table so messages never crash."""
    z = int(z)
    if 0 < z < len(_ELEMENT_SYMBOLS):
        return _ELEMENT_SYMBOLS[z]
    return f"Z{z}"


def element_number(symbol: str) -> int:
    """Atomic number for ``symbol``, or 0 when it is not an element.

    Case-insensitive, so it accepts both the mixed-case spelling a ``.g94``
    element header uses (``Na``) and the all-caps Pople-era one (``NA``).
    Returns 0 rather than raising: callers scan text where a non-element
    token is an ordinary miss, not an error.
    """
    return _SYMBOL_TO_Z.get(str(symbol).strip().lower(), 0)


# ---------------------------------------------------------------------------
# Registry file
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasisRecord:
    """What the registry knows about one basis name."""

    name: str
    family: str
    role: str
    default_jk: str | None
    default_ri: str | None
    standin_jk: str | None
    standin_ri: str | None
    ecp_required_above_z: int | None
    all_electron: bool
    note: str

    @property
    def registered(self) -> bool:
        return self.family != "unknown"


@dataclass(frozen=True)
class _Family:
    name: str
    pattern: re.Pattern[str]
    role: str
    all_electron: bool
    ecp_required_above_z: int | None
    note: str


@dataclass(frozen=True)
class _Registry:
    aliases: dict[str, str]
    bases: dict[str, dict]
    families: tuple[_Family, ...]


@lru_cache(maxsize=1)
def _registry() -> _Registry:
    with open(_REGISTRY_PATH, "rb") as fh:
        raw = tomllib.load(fh)
    aliases = {
        str(k).strip().lower(): str(v).strip().lower()
        for k, v in (raw.get("aliases") or {}).items()
    }
    bases = {
        str(k).strip().lower(): dict(v) for k, v in (raw.get("bases") or {}).items()
    }
    families = tuple(
        _Family(
            name=str(f["name"]),
            pattern=re.compile(str(f["pattern"])),
            role=str(f.get("role", "orbital")),
            all_electron=bool(f.get("all_electron", False)),
            ecp_required_above_z=(
                int(f["ecp_required_above_z"])
                if f.get("ecp_required_above_z") is not None
                else None
            ),
            note=str(f.get("note", "")),
        )
        for f in (raw.get("families") or [])
    )
    return _Registry(aliases=aliases, bases=bases, families=families)


def canonical_basis_name(name: object) -> str:
    """Lowercase, stripped, alias-resolved basis name.

    This is the key every metadata lookup uses.  It does not change the
    file libint opens: the bundled files are already stored under this
    spelling, and the aliases table lists only spellings that resolve to a
    file that ships under another stem.
    """
    key = str(getattr(name, "name", name) or "").strip().lower()
    return _registry().aliases.get(key, key)


def lookup(name: object) -> BasisRecord:
    """Return the registry record for ``name`` (a name or a ``BasisSet``).

    Always returns a record; an unregistered basis comes back with
    ``family="unknown"`` and no defaults so callers can decide how to
    refuse.
    """
    key = canonical_basis_name(name)
    reg = _registry()
    family: _Family | None = None
    for fam in reg.families:
        if fam.pattern.search(key):
            family = fam
            break
    rec = reg.bases.get(key, {})
    role = str(rec.get("role") or (family.role if family else "orbital"))
    threshold = rec.get("ecp_required_above_z")
    if threshold is None and family is not None:
        threshold = family.ecp_required_above_z
    all_electron = bool(
        rec.get("all_electron", family.all_electron if family else False)
    )
    return BasisRecord(
        name=key,
        family=str(rec.get("family") or (family.name if family else ("registered" if rec else "unknown"))),
        role=role,
        default_jk=rec.get("default_jk"),
        default_ri=rec.get("default_ri"),
        standin_jk=rec.get("standin_jk"),
        standin_ri=rec.get("standin_ri"),
        ecp_required_above_z=int(threshold) if threshold is not None else None,
        all_electron=all_electron,
        note=str(rec.get("note") or (family.note if family else "")),
    )


# ---------------------------------------------------------------------------
# Files: element coverage and sidecar core counts
# ---------------------------------------------------------------------------


def basis_file_path(name: object) -> Path | None:
    """Path of the ``.g94`` file libint will open for ``name``, or ``None``.

    Mirrors libint's own resolution: ``$LIBINT_DATA_PATH/basis/<name>.g94``
    (the variable ``vibeqc.__init__`` points at the bundled library) with
    the bundled directory as the fallback.
    """
    key = canonical_basis_name(name)
    if not key:
        return None
    root = os.environ.get("LIBINT_DATA_PATH")
    if root:
        candidate = Path(root) / "basis" / f"{key}.g94"
        if candidate.is_file():
            return candidate
    bundled = _BUNDLED_BASIS_DIR / f"{key}.g94"
    return bundled if bundled.is_file() else None


_G94_ELEMENT_HEADER = re.compile(r"^\s*([A-Z][A-Za-z]?)\s+0\s*$", re.MULTILINE)


@lru_cache(maxsize=256)
def _element_coverage_of_file(path: str) -> frozenset[int]:
    text = Path(path).read_text(errors="replace")
    zs = set()
    for m in _G94_ELEMENT_HEADER.finditer(text):
        z = _SYMBOL_TO_Z.get(m.group(1).lower())
        if z:
            zs.add(z)
    return frozenset(zs)


def element_coverage(name: object) -> frozenset[int]:
    """Atomic numbers the bundled ``.g94`` file carries a block for.

    Empty when the basis is not bundled (libint may still find it on a
    user-supplied ``LIBINT_DATA_PATH``).
    """
    path = basis_file_path(name)
    if path is None:
        return frozenset()
    return _element_coverage_of_file(str(path))


def sidecar_core_electrons(name: object) -> dict[int, int]:
    """``{Z: n_core}`` for every element block in the basis' ``.ecp`` sidecar.

    Empty when the basis ships no sidecar.  A zero-core block (a model
    potential that replaces no electrons) is reported with ``n_core == 0``.
    """
    from .ecp_metadata import parse_sidecar_path, sidecar_path_for

    sidecar = sidecar_path_for(canonical_basis_name(name))
    if sidecar is None:
        return {}
    return {int(h.Z): int(h.ncore) for h in parse_sidecar_path(sidecar)}


# ---------------------------------------------------------------------------
# ECP requirement, per element
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EcpRequirement:
    """One atom type of a job that must run under an ECP in this basis."""

    Z: int
    symbol: str
    n_core: int | None
    has_data: bool

    def __str__(self) -> str:  # pragma: no cover - message formatting
        return self.symbol


def ecp_requirements(name: object, atomic_numbers: Iterable[int]) -> list[EcpRequirement]:
    """Which of ``atomic_numbers`` need an ECP in basis ``name``.

    Sidecar first: an element whose sidecar block replaces core electrons
    needs the ECP and has the data to attach.  Then the family rule: a
    valence-only family beyond ``ecp_required_above_z`` needs an ECP on
    every heavier atom; if the sidecar has no block for such an atom (or
    the basis ships no sidecar at all) the requirement carries
    ``has_data=False`` and the caller must refuse rather than run
    all-electron in a valence basis.  An all-electron family never returns
    a requirement.
    """
    zs = sorted({int(z) for z in atomic_numbers if int(z) > 0})
    if not zs:
        return []
    record = lookup(name)
    if record.all_electron or record.role != "orbital":
        return []
    cores = sidecar_core_electrons(name)
    out: list[EcpRequirement] = []
    for z in zs:
        if z in cores:
            if cores[z] > 0:
                out.append(EcpRequirement(z, element_symbol(z), cores[z], True))
            continue
        if record.ecp_required_above_z is not None and z > record.ecp_required_above_z:
            out.append(EcpRequirement(z, element_symbol(z), None, False))
    return out


def basis_replaces_core(name: object, atomic_numbers: Iterable[int]) -> bool:
    """``True`` when any of ``atomic_numbers`` must run under an ECP."""
    return bool(ecp_requirements(name, atomic_numbers))


# ---------------------------------------------------------------------------
# Auxiliary basis defaults
# ---------------------------------------------------------------------------

_AUX_KINDS = ("jk", "ri")


def default_aux_basis(name: object, kind: str = "jk", *, allow_standin: bool = False) -> str:
    """Default fitting basis for orbital basis ``name``.

    ``kind`` is ``"jk"`` (Coulomb + exchange fit for SCF) or ``"ri"``
    (correlation fit for MP2 / CC).  Only a published fit is returned by
    default.  ``allow_standin=True`` additionally accepts the registry's
    zeta-matched stand-in for bases that have no published fit (the
    periodic GDF drivers rely on this; a molecular fit refuses instead so a
    user never gets a def2 fit under a pob or Pople basis without asking).

    Raises ``ValueError`` for an unknown ``kind`` and
    ``NotImplementedError`` when nothing is registered.
    """
    if kind not in _AUX_KINDS:
        raise ValueError(f"kind must be 'jk' or 'ri', got {kind!r}")
    record = lookup(name)
    published = record.default_jk if kind == "jk" else record.default_ri
    if published:
        return published
    standin = record.standin_jk if kind == "jk" else record.standin_ri
    if standin and allow_standin:
        return standin
    display = str(getattr(name, "name", name))
    if record.name.startswith("pob-"):
        raise NotImplementedError(
            f"No bundled JKfit / RIfit auxiliary basis for orbital basis "
            f"'{display}'. Designing a pob-* fitting basis is a separate "
            "research project (publication-worthy) and is tracked outside "
            "the molecular density-fitting feature work. Pass aux_basis= "
            "explicitly to override (any def2 fit is a reasonable quick-test "
            "fallback but accuracy is not guaranteed)."
        )
    raise NotImplementedError(
        f"No default {kind}-aux registered for orbital basis '{display}'. "
        "Minimal / Pople-style bases (sto-ng, 3-21g, 6-31g*, ...) ship no "
        "standard fitting aux -- pass aux_basis= explicitly (any bundled "
        "aux, or switch to a correlation-consistent / def2 orbital basis). "
        "For DLPNO methods set aux_basis on the DLPNO options object "
        "(e.g. LocalCCSDOptions(aux_basis='cc-pvdz-ri')). Bundled JKfit "
        "families: def2-svp..qzvpp, cc-pvDZ..5Z. Bundled RI families: "
        "def2-svp..qzvppd, cc-pvDZ..5Z. The registry lives in "
        "python/vibeqc/basis_library/registry.toml."
    )
