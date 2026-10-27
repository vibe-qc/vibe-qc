"""Export BasisSetData to NWChem ``basis`` and ``ECP`` blocks.

NWChem reads shells from a ``basis "<name>" spherical ... end`` block and
effective core potentials from a separate ``ECP ... END`` block::

    basis "pob-tzvp" spherical
      Ag    S
         12.567714    1.0
      ...
    end
    ECP
      Ag nelec 28
      Ag ul
        2    1.0    0.0
      Ag S
        2    12.567714    255.054771
      ...
    END

Conventions
-----------
* Shell labels are ``S P D F G H I K L M ...``: NWChem has no ``J``.
* An SP shell becomes an S and a P shell over the same exponents.
* An ECP row is ``power exponent coefficient`` with ``power`` the
  Gaussian-format n of ``r^(n-2)`` the records keep; ``ul`` is the local
  channel and the lettered channels are the projected differences. This
  is the layout PySCF's NWChem reader consumes, which
  ``tests/test_basis_formats_ecp.py`` uses as the independent check.
* Every number is written with :func:`repr`, so nothing is rounded.

Lossy fields
------------
NWChem input carries only the basis name: description, role,
basis_family, references and provenance are dropped. NWChem has no field
for the local channel's angular momentum; a reader infers it as one above
the highest projector, which is the same operator.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .model import HarmonicType, LossReport

if TYPE_CHECKING:
    from .model import BasisSetData, ECPEntry

__all__ = ["to_nwchem", "to_nwchem_file", "loss_report_nwchem", "nwchem_label"]

#: NWChem's shell letters by angular momentum (no ``J``), as PySCF reads them.
NWCHEM_LETTERS = "SPDFGHIKLMNORTU"


def nwchem_label(l: int) -> str:
    """The shell label NWChem accepts for angular momentum ``l``."""
    l = int(l)
    if not 0 <= l < len(NWCHEM_LETTERS):
        raise ValueError(f"NWChem has no label for angular momentum {l}")
    return NWCHEM_LETTERS[l]


# ---------------------------------------------------------------------------
# Loss report
# ---------------------------------------------------------------------------


def loss_report_nwchem(data: BasisSetData) -> LossReport:
    """What NWChem input cannot carry."""
    report = LossReport()
    for field in ("description", "role", "basis_family", "references", "provenance"):
        report.add_loss(field, "NWChem input carries only the basis name")
    if data.ecps and not all(_projectors_contiguous(e) for e in data.ecps.values()):
        report.add_loss(
            "angular_momentum_max",
            "NWChem cannot state the local channel's l; a reader infers the "
            "highest projector plus one (the same operator)",
        )
    return report


def _projectors_contiguous(entry: ECPEntry) -> bool:
    projectors = {int(p.angular_momentum) for p in entry.potentials} - {int(entry.angular_momentum_max)}
    return int(entry.angular_momentum_max) == (max(projectors) + 1 if projectors else 0)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _num(x: float, width: int = 22) -> str:
    return f"  {repr(float(x)):>{width}s}"


def _infer_global_harmonic(data: BasisSetData) -> str:
    """``"cartesian"`` if any shell is Cartesian, else ``"spherical"``."""
    for elem in data.elements.values():
        for shell in elem.shells:
            if shell.harmonic_type == HarmonicType.CARTESIAN:
                return "cartesian"
    return "spherical"


def _basis_block(data: BasisSetData) -> list[str]:
    lines = [f'basis "{data.name}" {_infer_global_harmonic(data)}']
    for sym in data.elems_sorted():
        # The per-element comment the Basis Set Exchange's NWChem files carry;
        # NWChem ignores it, PySCF's reader splits elements on it.
        lines.append(f"#BASIS SET: {sym}")
        for shell in data.elements[sym].shells:
            am = list(shell.angular_momentum)
            prims = shell.primitives
            if am == [0, 1]:
                n = len(prims) // 2
                groups = [("S", prims[:n]), ("P", prims[n:])]
            elif len(am) == 1:
                groups = [(nwchem_label(am[0]), prims)]
            else:
                raise ValueError(f"{sym}: cannot write shell with angular momentum {am}")
            for label, group in groups:
                lines.append(f"  {sym}    {label}")
                lines += [f"  {_num(p.exponent)}{_num(p.coefficient)}" for p in group]
    lines.append("end")
    return lines


def _ecp_block(data: BasisSetData) -> list[str]:
    from ._element_data import ATOMIC_NUMBER

    lines = ["ECP"]
    for sym in sorted(data.ecps, key=lambda s: ATOMIC_NUMBER.get(s, 999)):
        entry = data.ecps[sym]
        lmax = int(entry.angular_momentum_max)
        by_l = {int(p.angular_momentum): list(p.terms) for p in entry.potentials}
        if max(by_l, default=-1) != lmax:
            raise ValueError(f"{sym}: the local channel must be the highest one (#207)")
        lines.append(f"  {sym} nelec {int(entry.n_core_electrons)}")
        for l in [lmax] + list(range(lmax)):
            terms = by_l.get(l)
            if not terms:
                continue
            lines.append(f"  {sym} {'ul' if l == lmax else nwchem_label(l)}")
            lines += [f"    {int(t.power):d}{_num(t.exponent)}{_num(t.coefficient)}" for t in terms]
    lines.append("END")
    return lines


def _emit_nwchem(data: BasisSetData) -> str:
    lines = _basis_block(data)
    if data.ecps:
        lines += _ecp_block(data)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_nwchem(data: BasisSetData) -> str:
    """NWChem ``basis`` block of *data*, followed by its ``ECP`` block if any."""
    return _emit_nwchem(data)


def to_nwchem_file(data: BasisSetData, path: str | Path) -> LossReport:
    """Write the NWChem text to *path* and return the loss report."""
    Path(path).write_text(_emit_nwchem(data), encoding="utf-8")
    return loss_report_nwchem(data)
