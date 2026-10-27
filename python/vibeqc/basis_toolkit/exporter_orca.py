"""Export BasisSetData to an ORCA ``%basis`` block, shells and ECPs.

ORCA reads per-element basis sets from ``NewGTO <element> ... end`` and
per-element effective core potentials from ``NewECP <element> ... end``
inside one ``%basis ... end`` block (ORCA 6.1 manual, section 2.7.5
"Manual Input of ECP Parameters" and 2.7.6 "Assigning or Adding Basis
Functions to an Element"). Paste the block into an ORCA input with no
basis keyword on the simple-input line, so ORCA assigns nothing of its
own.

Conventions
-----------
* Primitive rows carry a running index starting at 1, then the exponent
  and the unnormalised contraction coefficient.
* Shell labels are ``s p d f g h i k`` for l = 0..7 (ORCA has no ``j``
  shell); l >= 8 is written as the integer, as the manual requires.
* An SP shell becomes an S and a P shell over the same exponents.
* An ECP shell row is ``index exponent coefficient power``. ``power`` is
  the Gaussian-format n of ``r^(n-2)``, the convention the records keep
  (the manual's SD(10,MDF) vanadium example writes n = 2 for its r^0
  terms). ``lmax`` names the local channel; the channels below it are the
  projected differences, as in the Gaussian ``s-f potential`` layout.
* Every number is written with :func:`repr`, the shortest decimal that
  reads back as the same double, so nothing is rounded.

Lossy fields
------------
ORCA input carries no metadata: name, description, role, basis_family,
references and provenance are dropped. ORCA always uses spherical
harmonics, so a Cartesian record changes meaning (the loss report says so).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .model import HarmonicType, LossReport

if TYPE_CHECKING:
    from .model import BasisSetData, ECPEntry

__all__ = ["to_orca", "to_orca_file", "loss_report_orca", "orca_label"]

#: ORCA's shell letters for l = 0..7; there is no ``j``.
_ORCA_LETTERS = "spdfghik"


def orca_label(l: int) -> str:
    """The shell label ORCA accepts for angular momentum ``l``."""
    l = int(l)
    if l < 0:
        raise ValueError(f"negative angular momentum {l}")
    return _ORCA_LETTERS[l].upper() if l < len(_ORCA_LETTERS) else str(l)


# ---------------------------------------------------------------------------
# Loss report
# ---------------------------------------------------------------------------


def loss_report_orca(data: BasisSetData) -> LossReport:
    """What an ORCA ``%basis`` block cannot carry."""
    report = LossReport()
    for field in ("name", "description", "role", "basis_family", "references", "provenance"):
        report.add_loss(field, "ORCA %basis input carries no metadata")
    if any(
        shell.harmonic_type == HarmonicType.CARTESIAN
        for element in data.elements.values()
        for shell in element.shells
    ):
        report.add_loss(
            "harmonic_type",
            "ORCA always uses spherical harmonics; Cartesian shells change meaning",
        )
    return report


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _num(x: float, width: int = 22) -> str:
    return f"  {repr(float(x)):>{width}s}"


def _shell_rows(label: str, exponents, coefficients) -> list[str]:
    rows = [f"    {label}  {len(exponents)}"]
    for i, (exponent, coefficient) in enumerate(zip(exponents, coefficients), start=1):
        rows.append(f"    {i:3d}{_num(exponent)}{_num(coefficient)}")
    return rows


def _gto_block(sym: str, element) -> list[str]:
    lines = [f"  NewGTO {sym}"]
    for shell in element.shells:
        am = list(shell.angular_momentum)
        prims = shell.primitives
        if am == [0, 1]:
            n = len(prims) // 2
            exps = [p.exponent for p in prims[:n]]
            lines += _shell_rows("S", exps, [p.coefficient for p in prims[:n]])
            lines += _shell_rows("P", exps, [p.coefficient for p in prims[n:]])
        elif len(am) == 1:
            lines += _shell_rows(
                orca_label(am[0]), [p.exponent for p in prims], [p.coefficient for p in prims]
            )
        else:
            raise ValueError(f"{sym}: cannot write shell with angular momentum {am}")
    lines.append("  end")
    return lines


def _ecp_block(sym: str, entry: ECPEntry) -> list[str]:
    lmax = int(entry.angular_momentum_max)
    by_l = {int(p.angular_momentum): list(p.terms) for p in entry.potentials}
    if max(by_l, default=-1) != lmax:
        raise ValueError(f"{sym}: the local channel must be the highest one (#207)")
    lines = [
        f"  NewECP {sym}",
        f"    N_core {int(entry.n_core_electrons)}",
        f"    lmax {orca_label(lmax).lower()}",
    ]
    for l in range(lmax + 1):
        terms = by_l.get(l)
        if not terms:
            continue
        lines.append(f"    {orca_label(l).lower()} {len(terms)}")
        for i, term in enumerate(terms, start=1):
            lines.append(f"    {i:3d}{_num(term.exponent)}{_num(term.coefficient)}  {int(term.power):d}")
    lines.append("  end")
    return lines


def _emit_orca(data: BasisSetData) -> str:
    """The complete ``%basis`` block of *data*: every shell, every ECP."""
    from ._element_data import ATOMIC_NUMBER

    lines: list[str] = ["%basis"]
    for sym in data.elems_sorted():
        lines += _gto_block(sym, data.elements[sym])
    for sym in sorted(data.ecps or {}, key=lambda s: ATOMIC_NUMBER.get(s, 999)):
        lines += _ecp_block(sym, data.ecps[sym])
    lines.append("end")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_orca(data: BasisSetData) -> str:
    """The ORCA ``%basis ... end`` block of *data*, ECPs included."""
    return _emit_orca(data)


def to_orca_file(data: BasisSetData, path: str | Path) -> LossReport:
    """Write the ``%basis`` block to *path* and return the loss report."""
    Path(path).write_text(_emit_orca(data), encoding="utf-8")
    return loss_report_orca(data)
