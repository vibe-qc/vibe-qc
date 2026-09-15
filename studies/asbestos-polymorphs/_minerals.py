"""Geometry library for the asbestos-polymorphs study.

Each mineral lives in cifs/<name>.cif (drop the CIF, then call the
matching builder here). The builder converts CIF → ASE Atoms → vibe-qc
PeriodicSystem in bohr.

If a CIF is missing, the builder raises FileNotFoundError pointing at
cifs/CIFS_NEEDED.md, which lists the recommended source per mineral.

Mirror of examples/periodic/_systems.py — same pattern, just for the
amphibole / serpentine paper rather than the parity-test set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903
HERE = Path(__file__).resolve().parent
CIFS = HERE / "cifs"


def _load_from_cif(cif_name: str) -> Tuple[vq.PeriodicSystem, str]:
    """Load a CIF from cifs/, convert to vibe-qc PeriodicSystem."""
    try:
        from ase.io import read as ase_read
    except ImportError as exc:
        raise RuntimeError(
            "asbestos-polymorphs/_minerals.py requires ASE. "
            "pip install ase"
        ) from exc

    cif_path = CIFS / cif_name
    if not cif_path.is_file():
        raise FileNotFoundError(
            f"CIF missing: {cif_path}.\n"
            f"See {CIFS / 'CIFS_NEEDED.md'} for the recommended source."
        )

    atoms_ase = ase_read(str(cif_path))
    cell_bohr = np.asarray(atoms_ase.cell.array, dtype=float) * ANG2BOHR

    Z_BY_SYMBOL = {
        "H": 1, "Li": 3, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11,
        "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
        "K": 19, "Ca": 20, "Fe": 26,
    }
    cell_atoms = []
    for sym, pos_ang in zip(atoms_ase.get_chemical_symbols(),
                            atoms_ase.get_positions()):
        if sym not in Z_BY_SYMBOL:
            raise NotImplementedError(
                f"Z lookup for {sym!r} not in _minerals.py — extend "
                f"Z_BY_SYMBOL."
            )
        cell_atoms.append(vq.Atom(Z_BY_SYMBOL[sym],
                                  list(pos_ang * ANG2BOHR)))

    sys_p = vq.PeriodicSystem(3, cell_bohr, cell_atoms)
    label = f"{atoms_ase.get_chemical_formula()} ({len(atoms_ase)} atoms)"
    return sys_p, label


# ============================================================
# Builders — one per mineral
# ============================================================

def lizardite_1T() -> Tuple[vq.PeriodicSystem, str]:
    """Lizardite-1T (P3̄1m, trigonal). Mg₃Si₂O₅(OH)₄.
    Reference: Mellini & Zanazzi 1987 (AMCSD #0011073)."""
    return _load_from_cif("lizardite-1T.cif")


def chrysotile_clino() -> Tuple[vq.PeriodicSystem, str]:
    """Chrysotile, clino polytype (Cc, monoclinic). Mg₃Si₂O₅(OH)₄.
    Bulk-crystal reference (not the cylindrical scroll).
    Reference: Whittaker 1956 (AMCSD #0009829)."""
    return _load_from_cif("chrysotile-clino.cif")


def antigorite_m17() -> Tuple[vq.PeriodicSystem, str]:
    """Antigorite, m=17 superstructure (Pm, monoclinic). ~290 atoms.
    Reference: Capitani & Mellini 2004 (AMCSD #0006097).

    Heads up: this is a stress-test cell. Start with smaller m for
    early runs; document which m was used."""
    return _load_from_cif("antigorite-m17.cif")


def tremolite() -> Tuple[vq.PeriodicSystem, str]:
    """Tremolite (C2/m, monoclinic). Ca₂Mg₅Si₈O₂₂(OH)₂.
    Mg endmember of the tremolite-actinolite series — closed-shell.
    Reference: Hawthorne & Grundy 1976 (AMCSD #0005135)."""
    return _load_from_cif("tremolite.cif")


def anthophyllite_Mg() -> Tuple[vq.PeriodicSystem, str]:
    """Anthophyllite, Mg endmember (Pnma, orthorhombic).
    Mg₇Si₈O₂₂(OH)₂. Closed-shell.
    Reference: Walitzi 1965 / Sueno et al. 1972."""
    return _load_from_cif("anthophyllite-Mg.cif")


def riebeckite() -> Tuple[vq.PeriodicSystem, str]:
    """Riebeckite (C2/m, monoclinic).
    Na₂Fe₃²⁺Fe₂³⁺Si₈O₂₂(OH)₂. Open-shell — needs UKS.
    Reference: Whittaker 1949."""
    return _load_from_cif("riebeckite.cif")


def grunerite() -> Tuple[vq.PeriodicSystem, str]:
    """Grunerite (C2/m, monoclinic). Fe₇²⁺Si₈O₂₂(OH)₂.
    Open-shell — needs UKS. Reference: Hawthorne 1983."""
    return _load_from_cif("grunerite.cif")


# Lookup-by-name for the orchestrator.
MINERALS = {
    "lizardite-1T":      lizardite_1T,
    "chrysotile-clino":  chrysotile_clino,
    "antigorite-m17":    antigorite_m17,
    "tremolite":         tremolite,
    "anthophyllite":     anthophyllite_Mg,
    "riebeckite":        riebeckite,
    "grunerite":         grunerite,
}
