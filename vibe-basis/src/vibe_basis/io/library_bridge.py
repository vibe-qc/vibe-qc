"""Bridge between vibe-basis and the qc-input-library.

The separately installed qc-input-library
contains 78+ standalone CRYSTAL14 `.d12` input files with inline
POB basis sets, reference outputs, and OPTBASIS templates.

This module:
* Discovers systems by compound, method, and basis
* Parses geometry + reference energies from existing inputs/outputs
* Converts them to vibe-basis :class:`Structure` objects
* Provides pre-built optimization test sets
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vibe_basis.io.structures import Structure, StructureAtom

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass
class LibraryEntry:
    """One compound/method/basis combination from the library."""

    compound: str
    lattice: str  # e.g. "rocksalt", "diamond", "fcc"
    method: str  # "rhf", "pbe", "b3lyp", "pbe0", …
    basis: str  # "pob-tzvp-rev2", "sto-3g", …
    input_path: Path
    output_path: Optional[Path] = None
    reference_energy: Optional[float] = None


class LibraryIndex:
    """Index of all CRYSTAL inputs in the qc-input-library."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._entries: list[LibraryEntry] = []
        self._scan()

    def _scan(self) -> None:
        crystal_dir = self.root / "crystal"
        if not crystal_dir.exists():
            return

        for system_dir in sorted(crystal_dir.iterdir()):
            if not system_dir.is_dir():
                continue
            # Parse compound-lattice from directory name.
            parts = system_dir.name.split("-", 1)
            compound = parts[0]
            lattice = parts[1] if len(parts) > 1 else ""

            for calc_dir in sorted(system_dir.iterdir()):
                if not calc_dir.is_dir() or calc_dir.name.startswith("_"):
                    continue
                method, _, basis = calc_dir.name.partition("_")
                if not basis:
                    continue

                inp = calc_dir / "INPUT.d12"
                if not inp.exists():
                    continue

                entry = LibraryEntry(
                    compound=compound,
                    lattice=lattice,
                    method=method,
                    basis=basis.replace("_", "-"),
                    input_path=inp,
                )

                out = calc_dir / "output.out"
                if out.exists():
                    entry.output_path = out
                    entry.reference_energy = _extract_energy(out)

                self._entries.append(entry)

    @property
    def entries(self) -> list[LibraryEntry]:
        return list(self._entries)

    @property
    def compounds(self) -> set[str]:
        return {e.compound for e in self._entries}

    def find(
        self,
        compound: str | None = None,
        method: str | None = None,
        basis: str | None = None,
    ) -> list[LibraryEntry]:
        """Filter entries by compound, method, and/or basis."""
        result = self._entries
        if compound:
            result = [e for e in result if e.compound == compound]
        if method:
            result = [e for e in result if e.method == method]
        if basis:
            result = [e for e in result if e.basis == basis]
        return result

    def test_set(
        self,
        method: str = "rhf",
        basis: str = "pob-tzvp-rev2",
        *,
        exclude: set[str] | None = None,
    ) -> list[LibraryEntry]:
        """Return a default optimization test set.

        Filters to a given method + basis, excluding slabs, 0D, 1D, and
        2D systems (the optimizer needs 3D periodic bulk for now).
        """
        entries = self.find(method=method, basis=basis)
        # Exclude non-bulk and user-specified compounds.
        non_bulk = {
            "001-slab",
            "molecule",
            "dimer",
            "trimer",
            "chain-1d",
            "polymer-1d",
            "graphene-2d",
            "peierls-1d",
            "adsorbate",
        }
        result = []
        for e in entries:
            parts = e.input_path.parent.parent.name.split("-")
            if any(p in non_bulk for p in parts):
                continue
            if exclude and e.compound in exclude:
                continue
            result.append(e)
        return result


# ---------------------------------------------------------------------------
# Geometry parsing from .d12 files
# ---------------------------------------------------------------------------


def parse_d12_geometry(path: Path) -> Optional[Structure]:
    """Extract geometry from a qc-input-library .d12 file.

    Returns a vibe-basis :class:`Structure`, or None if the .d12
    doesn't contain a standard periodic geometry block.
    """
    text = path.read_text()
    lines = text.splitlines()
    title = lines[0].strip()

    # CRYSTAL type: line 2
    crystal_type = lines[1].strip() if len(lines) > 1 else "CRYSTAL"
    if crystal_type not in ("CRYSTAL",):
        return None

    # Print flags: line 3
    # Space group: line 4
    # Lattice params: line 5
    # N atoms: line 6
    try:
        sg = int(lines[3].strip())
        lattice = lines[4].strip().split()
        natoms = int(lines[5].strip())
    except (ValueError, IndexError):
        return None

    # Parse lattice parameters.
    nlat = len(lattice)
    a = float(lattice[0]) if nlat > 0 else 1.0
    b = float(lattice[1]) if nlat > 1 else a
    c = float(lattice[2]) if nlat > 2 else a
    alpha = float(lattice[3]) if nlat > 3 else 90.0
    beta = float(lattice[4]) if nlat > 4 else 90.0
    gamma = float(lattice[5]) if nlat > 5 else 90.0

    # Determine crystal system from lattice parameters.
    if len(lattice) == 1:
        system = "cubic"
    elif len(lattice) == 2:
        system = "hexagonal"
    elif len(lattice) == 3:
        system = "orthorhombic"
        if abs(a - b) < 1e-6 and abs(b - c) < 1e-6:
            system = "cubic"
    elif len(lattice) == 4:
        system = "monoclinic"
    else:
        system = "triclinic"

    # Parse atom positions.
    unit_cell: list[StructureAtom] = []
    asymm_unit: list[StructureAtom] = []
    for i in range(natoms):
        line_idx = 6 + i
        if line_idx >= len(lines):
            break
        parts = lines[line_idx].strip().split()
        if len(parts) < 4:
            break
        Z = int(parts[0])
        fx = float(parts[1])
        fy = float(parts[2])
        fz = float(parts[3])
        atom = StructureAtom(Z=Z, fxyz=(fx, fy, fz))
        asymm_unit.append(atom)
        unit_cell.append(atom)

    # Determine spacegroup symbol.
    sg_map = {
        225: "Fm-3m",
        227: "Fd-3m",
        216: "F-43m",
        221: "Pm-3m",
        224: "Pn-3m",
        1: "P1",
        166: "R-3m",
        167: "R-3c",
        186: "P6_3mc",
        194: "P6_3/mmc",
        136: "P4_2/mnm",
    }

    return Structure(
        name=f"{title.lower().replace(' ', '-')}",
        formula=title,
        spacegroup=sg_map.get(sg, f"sg{sg}"),
        crystal_system=system,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        unit_cell=tuple(unit_cell),
        crystal_spacegroup=sg,
        crystal_asymm_unit=tuple(asymm_unit),
    )


# ---------------------------------------------------------------------------
# Energy extraction from output files
# ---------------------------------------------------------------------------


_TOTAL_ENERGY_RE = re.compile(
    r"TOTAL\s+ENERGY\s*\([^)]+\)\s*\([^)]*\)\s*\(\s*\d+\s*\)\s+"
    r"(-?\d+\.\d+[Ee][+\-]?\d+)"
)


def _extract_energy(path: Path) -> Optional[float]:
    """Extract the last TOTAL ENERGY from a CRYSTAL output file."""
    text = path.read_text()
    last_energy = None
    for line in text.splitlines():
        m = _TOTAL_ENERGY_RE.search(line)
        if m:
            last_energy = float(m.group(1))
    return last_energy


# ---------------------------------------------------------------------------
# Quick bootstrap
# ---------------------------------------------------------------------------


def discover_library(
    root: str | Path = ".",
) -> Optional[LibraryIndex]:
    """Auto-discover the qc-input-library.

    Looks for a directory named ``qc-input-library`` relative to
    the current working directory, or at a user-specified path.
    Returns None if not found.
    """
    root = Path(root)
    if not root.exists():
        # Try common locations.
        candidates = [
            Path("../qc-input-library"),
            Path.home() / "gitlab/qc-input-library",
        ]
        for c in candidates:
            if c.exists():
                return LibraryIndex(c)
        return None
    return LibraryIndex(root)
