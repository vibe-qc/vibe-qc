"""Export BasisSetData to ORCA ``%basis`` block format.

ORCA reads basis sets from its ``%basis`` directive, which uses a
``NewGTO`` / ``end`` block structure.  This exporter writes a single
``%basis ... end`` block that can be pasted into an ORCA input file.

Lossy fields
------------
Same as :mod:`.exporter_g94`: name, description, role, basis_family,
references, provenance, and harmonic_type are all dropped.
SP shells are split into separate ``S`` and ``P`` blocks sharing the
same exponents, which is the ORCA convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .model import LossReport

if TYPE_CHECKING:
    from .model import BasisSetData

__all__ = ["to_orca", "to_orca_file", "loss_report_orca"]

# Angular-momentum -> ORCA shell label.
_AM_LABEL: dict[int, str] = {
    0: "S",
    1: "P",
    2: "D",
    3: "F",
    4: "G",
    5: "H",
    6: "I",
}


# ---------------------------------------------------------------------------
# Loss report
# ---------------------------------------------------------------------------


def loss_report_orca(_data: BasisSetData) -> LossReport:
    """Pre-compute the information-loss report for an ORCA export."""
    report = LossReport()
    report.add_loss("name", "ORCA %basis block has no name field")
    report.add_loss("description", "no description field in ORCA %basis")
    report.add_loss("role", "no role field in ORCA %basis")
    report.add_loss("basis_family", "no family field in ORCA %basis")
    report.add_loss("references", "no citation block in ORCA %basis")
    report.add_loss("provenance", "no provenance metadata in ORCA %basis")
    report.add_loss("harmonic_type", "ORCA %basis does not carry spherical/cartesian")
    return report


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def _fmt_exponent_orca(x: float) -> str:
    """Format an exponent for ORCA output -- 12 significant digits."""
    return f"{x:>20.12g}"


def _fmt_coefficient_orca(c: float) -> str:
    """Format a contraction coefficient -- 12 significant digits."""
    return f"{c:>20.12g}"


# ---------------------------------------------------------------------------
# ORCA emission
# ---------------------------------------------------------------------------


def _emit_orca(data: BasisSetData) -> str:
    """Return the complete ORCA ``%basis`` block for *data*."""
    lines: list[str] = ["%basis"]

    for sym in data.elems_sorted():
        elem = data.elements[sym]
        lines.append(f"  NewGTO {sym}")

        for shell in elem.shells:
            am = shell.angular_momentum
            n_prim = len(shell.primitives)
            n_unique = n_prim // 2 if am == [0, 1] else n_prim
            label = _AM_LABEL.get(am[0], f"?{am[0]}")

            if am == [0, 1]:
                # SP shell -> emit separate S and P blocks.
                # S block (first N primitives carry s-coefficients).
                lines.append(f"    {label}  {n_unique}")
                for i in range(n_unique):
                    exp = _fmt_exponent_orca(shell.primitives[i].exponent)
                    coeff = _fmt_coefficient_orca(shell.primitives[i].coefficient)
                    lines.append(f"    {i}  {exp}  {coeff}")
                # P block (next N primitives carry p-coefficients,
                # same exponents).
                lines.append(f"    P  {n_unique}")
                for i in range(n_unique):
                    exp = _fmt_exponent_orca(shell.primitives[i + n_unique].exponent)
                    coeff = _fmt_coefficient_orca(
                        shell.primitives[i + n_unique].coefficient
                    )
                    lines.append(f"    {i}  {exp}  {coeff}")
            else:
                lines.append(f"    {label}  {n_prim}")
                for i, prim in enumerate(shell.primitives):
                    exp = _fmt_exponent_orca(prim.exponent)
                    coeff = _fmt_coefficient_orca(prim.coefficient)
                    lines.append(f"    {i}  {exp}  {coeff}")

        lines.append("  end")
    lines.append("end")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_orca(data: BasisSetData) -> str:
    """Convert *data* to an ORCA ``%basis`` block string."""
    return _emit_orca(data)


def to_orca_file(data: BasisSetData, path: str | Path) -> LossReport:
    """Write *data* to *path* as an ORCA ``%basis`` block and return a loss report."""
    path = Path(path)
    _ = path.write_text(_emit_orca(data), encoding="utf-8")
    return loss_report_orca(data)
