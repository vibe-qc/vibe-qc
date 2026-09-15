"""Export BasisSetData to NWChem basis block format.

NWChem reads basis sets from a ``basis`` directive with element-tagged
shell blocks.  This exporter writes the ``basis ... end`` block.

Lossy fields
------------
Same as :mod:`.exporter_g94`: name, description, role, basis_family,
references, provenance, and harmonic_type are all dropped.
SP shells are split into separate ``S`` and ``P`` blocks sharing the
same exponents, which matches NWChem's convention.
ECP data is omitted -- NWChem ECP blocks are a separate directive
(``ecp``) and are not emitted here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .model import HarmonicType, LossReport

if TYPE_CHECKING:
    from .model import BasisSetData

__all__ = ["to_nwchem", "to_nwchem_file", "loss_report_nwchem"]

# Angular-momentum -> NWChem shell label.
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


def loss_report_nwchem(data: BasisSetData) -> LossReport:
    """Pre-compute the information-loss report for an NWChem export."""
    report = LossReport()
    report.add_loss("name", "NWChem basis header carries the name as a comment only")
    report.add_loss("description", "no description field in NWChem basis")
    report.add_loss("role", "no role field in NWChem basis")
    report.add_loss("basis_family", "no family field in NWChem basis")
    report.add_loss("references", "no citation block in NWChem basis")
    report.add_loss("provenance", "no provenance metadata in NWChem basis")
    report.add_loss(
        "harmonic_type",
        "NWChem basis header declares spherical/cartesian globally; "
        + "per-shell harmonic_type is not preserved",
    )
    if data.ecps:
        report.add_loss("ecps", "ECP block is a separate NWChem directive, not emitted")
    return report


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def _fmt_nwchem(x: float) -> str:
    """Format a float for NWChem -- 12 significant digits, right-aligned."""
    return f"{x:>20.12g}"


# ---------------------------------------------------------------------------
# NWChem emission
# ---------------------------------------------------------------------------


def _emit_nwchem(data: BasisSetData) -> str:
    """Return the complete NWChem basis block for *data*."""
    # Determine the global harmonic type for the header.
    # Walk all shells; if any shell is Cartesian, declare "cartesian".
    harmonic = _infer_global_harmonic(data)

    lines: list[str] = [f'basis "{data.name}" {harmonic}']

    for sym in data.elems_sorted():
        elem = data.elements[sym]

        for shell in elem.shells:
            am = shell.angular_momentum
            n_prim = len(shell.primitives)
            n_unique = n_prim // 2 if am == [0, 1] else n_prim
            label = _AM_LABEL.get(am[0], f"?{am[0]}")

            if am == [0, 1]:
                # SP shell -> emit separate S and P blocks.
                # S block.
                lines.append(f"  {sym}    {label}")
                for i in range(n_unique):
                    exp = _fmt_nwchem(shell.primitives[i].exponent)
                    coeff = _fmt_nwchem(shell.primitives[i].coefficient)
                    lines.append(f"    {exp}    {coeff}")
                # P block.
                lines.append(f"  {sym}    P")
                for i in range(n_unique):
                    exp = _fmt_nwchem(shell.primitives[i + n_unique].exponent)
                    coeff = _fmt_nwchem(shell.primitives[i + n_unique].coefficient)
                    lines.append(f"    {exp}    {coeff}")
            else:
                lines.append(f"  {sym}    {label}")
                for prim in shell.primitives:
                    exp = _fmt_nwchem(prim.exponent)
                    coeff = _fmt_nwchem(prim.coefficient)
                    lines.append(f"    {exp}    {coeff}")

    lines.append("end")
    return "\n".join(lines)


def _infer_global_harmonic(data: BasisSetData) -> str:
    """Infer the global harmonic type from all shells.

    Returns ``"spherical"`` unless any shell is Cartesian.
    """
    for elem in data.elements.values():
        for shell in elem.shells:
            if shell.harmonic_type == HarmonicType.CARTESIAN:
                return "cartesian"
    return "spherical"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_nwchem(data: BasisSetData) -> str:
    """Convert *data* to an NWChem basis block string."""
    return _emit_nwchem(data)


def to_nwchem_file(data: BasisSetData, path: str | Path) -> LossReport:
    """Write *data* to *path* as an NWChem basis block and return a loss report."""
    path = Path(path)
    _ = path.write_text(_emit_nwchem(data), encoding="utf-8")
    return loss_report_nwchem(data)
