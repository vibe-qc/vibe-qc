"""PT2013 supporting information reference energies for vibe-basis.

Single source of truth for the HF total energies per unit cell
listed in the Supporting Information Table 2 of:

  M. F. Peintinger, D. Vilela Oliveira, T. Bredow,
  *J. Comput. Chem.* **34**, 451 (2013).  DOI 10.1002/jcc.23153

These energies were computed by CRYSTAL09 with the pob-TZVP basis
at SHRINK 8 8 and the paper's published lattice constants (relaxed
geometries).  Stage 0 of Goal 8 compares against these numbers as
an *infrastructure smoke test* — showing the pipeline (emit → submit
→ parse) is correct, not that the numerics match to µHa.

NaF is listed as ``None`` because the HF SCF did not converge in
the paper (the SI Table 2 entry is a dash).
"""

from __future__ import annotations

from typing import Optional

# ---- Reference energies (Hartree per unit cell) ---------------------------
#
# Migrated from `examples/basisset_dev/_generator.py` on 2026-05-14
# with the same numeric literals.

PT2013_T2_HF: dict[str, Optional[float]] = {
    "LiCl": -467.087468,
    "NaCl": -621.495944,
    "LiF": -107.055717,
    "NaF": None,  # HF SCF did not converge in PT2013 SI Table 2
    "KF": -698.704515,
    "CaF2": -875.945290,
    "K2O": -1273.185420,
    "MgO": -274.681754,
    "CaO": -751.806904,
    "LiH": -8.062837,
    "NaH": -162.453809,
    "KH": -599.723256,
}


# ---- Convenience accessors ------------------------------------------------


def get_ref(name: str) -> Optional[float]:
    """Return the PT2013 SI Table 2 HF reference energy for *name*,
    or ``None`` if (a) the compound is not in the table or (b) the
    table entry is a dash (convergence failure).
    """
    return PT2013_T2_HF.get(name)


def has_ref(name: str) -> bool:
    """``True`` iff *name* has a valid (non-None, non-missing) reference."""
    v = PT2013_T2_HF.get(name)
    return v is not None


def compounds_with_ref() -> list[str]:
    """Return the list of compound names that have a valid reference energy."""
    return sorted(k for k, v in PT2013_T2_HF.items() if v is not None)
