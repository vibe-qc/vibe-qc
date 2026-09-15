"""Export BasisSetData to Gaussian94 (.g94) text format.

G94 is the format parsed by libint -- the native integral engine for
vibe-qc.  This exporter writes the subset of the canonical model that
G94 can represent: element blocks with shells and primitives.

Lossy fields
------------
G94 carries no metadata beyond a comment header.  The following fields
of :class:`~.model.BasisSetData` are **always dropped**::

    name, description, role, basis_family, references,
    provenance, harmonic_type

SP shells (``angular_momentum == [0, 1]``) are preserved via the G94
``SP`` shell label convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .model import LossReport

if TYPE_CHECKING:
    from .model import BasisSetData

__all__ = ["to_g94", "to_g94_file", "loss_report_g94"]

# Angular-momentum -> G94 shell label for pure (non-SP) shells.
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


def loss_report_g94(_data: BasisSetData) -> LossReport:
    """Pre-compute the information-loss report for a G94 export.

    This is a pure function -- it does not write anything.
    """
    report = LossReport()
    report.add_loss("name", "G94 header carries the name as a comment only")
    report.add_loss("description", "no description field in G94")
    report.add_loss("role", "no role field in G94")
    report.add_loss("basis_family", "no family field in G94")
    report.add_loss("references", "no citation block in G94")
    report.add_loss("provenance", "no provenance metadata in G94")
    report.add_loss("harmonic_type", "G94 does not encode spherical vs Cartesian")
    return report


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

_EXP_THRESHOLD = 0.01


def _fmt_exponent(x: float) -> str:
    """Format an exponent for G94 output.

    Exponents >= 0.01 use fixed-point with 6 decimal places;
    smaller exponents use scientific notation.
    """
    if abs(x) >= _EXP_THRESHOLD:
        return f"{x:>14.6f}"
    return f"{x:>14.6e}"


def _fmt_coefficient(c: float) -> str:
    """Format a contraction coefficient with 12 decimal places."""
    return f"{c:>18.12f}"


# ---------------------------------------------------------------------------
# G94 emission
# ---------------------------------------------------------------------------


def _emit_g94(data: BasisSetData) -> str:
    """Return the complete G94 text for *data*."""
    lines: list[str] = []

    # Comment header -- two mandatory comment lines.
    lines.append(f"! Basis: {data.name}")
    role_str = data.role.value
    family_str = data.basis_family or ""
    lines.append(f"! Role: {role_str} {family_str}".rstrip())

    # Element blocks, sorted by atomic number.
    for sym in data.elems_sorted():
        elem = data.elements[sym]
        lines.append("****")
        lines.append(f"{sym}     0")

        for shell in elem.shells:
            am = shell.angular_momentum
            n_prim = len(shell.primitives)

            if am == [0, 1]:
                # SP shell: 2*N primitives (first N = s, next N = p).
                n_unique = n_prim // 2
                lines.append(f"SP   {n_unique}   1.00")
                for i in range(n_unique):
                    exp = _fmt_exponent(shell.primitives[i].exponent)
                    s_coeff = _fmt_coefficient(shell.primitives[i].coefficient)
                    p_coeff = _fmt_coefficient(
                        shell.primitives[i + n_unique].coefficient
                    )
                    lines.append(f"  {exp}  {s_coeff}  {p_coeff}")
            else:
                # Pure shell.
                label = _AM_LABEL.get(am[0], f"?{am[0]}")
                lines.append(f"{label}   {n_prim}   1.00")
                for prim in shell.primitives:
                    exp = _fmt_exponent(prim.exponent)
                    coeff = _fmt_coefficient(prim.coefficient)
                    lines.append(f"  {exp}  {coeff}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_g94(data: BasisSetData) -> str:
    """Convert *data* to a G94-formatted string."""
    return _emit_g94(data)


def to_g94_file(data: BasisSetData, path: str | Path) -> LossReport:
    """Write *data* to *path* as G94 text and return a loss report."""
    path = Path(path)
    _ = path.write_text(_emit_g94(data), encoding="utf-8")
    return loss_report_g94(data)
