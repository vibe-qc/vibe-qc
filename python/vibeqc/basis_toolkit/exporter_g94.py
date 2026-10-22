"""Render :class:`BasisSetData` as the Gaussian94 text libint and libecpint read.

Two files come out of one record: ``<name>.g94`` carries the orbital
shells (libint's ``BasisSet`` parser), and ``<name>.ecp`` the effective
core potentials (``vibeqc.ecp_metadata.parse_inline_ecp_sidecar``). The
record is the source of truth; this module is the only writer of either
file under ``basis_library/`` (#406).

Precision
---------
Every number is written with :func:`repr`, the shortest decimal that
parses back to the identical double. Nothing is rounded: a record and its
rendering carry the same values to the last bit, which is what the
round-trip pin in ``tests/test_basis_qvf_parity.py`` checks and what
#291 asked for after a fixed-width writer lost 1,013 digits.

Header
------
The ``!`` comment header is derived from the record's metadata, so the
originating publications, the provenance and the content hash travel with
the text the way the Basis Set Exchange's own renderers do. A header
cannot be edited by hand; change the record and re-render.

Layout of the ECP sidecar
-------------------------
The BSE / Gaussian layout, with the local channel (``angular_momentum ==
angular_momentum_max``) first and the projectors in ascending ``l``::

    Rb     0
    Rb-ECP     3    28
    f potential
      1
    2      3.843114      -12.3169
    s-f potential
      3
    ...

``power`` is the Gaussian-format integer ``n`` of ``r^(n-2)``; libecpint
subtracts two when it reads it (#207).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .model import LossReport

if TYPE_CHECKING:
    from .model import BasisSetData, ECPEntry

__all__ = [
    "to_g94",
    "to_g94_file",
    "to_ecp_sidecar",
    "render_header",
    "loss_report_g94",
    "AM_LETTERS",
]

# Angular momentum -> G94 shell label for pure (non-SP) shells.
_AM_LABEL: dict[int, str] = {
    0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H", 6: "I", 7: "J", 8: "K",
}

#: Lower-case channel letters of the ECP sidecar, indexed by ``l``.
AM_LETTERS = "spdfghik"

_RULE = "! " + "-" * 70
_WRAP = 76


# ---------------------------------------------------------------------------
# Loss report
# ---------------------------------------------------------------------------


def loss_report_g94(data: BasisSetData) -> LossReport:
    """What a G94 rendering cannot carry as structured data.

    Metadata survives only as comment text; ``harmonic_type`` is not
    encoded at all (libint decides spherical versus Cartesian itself).
    """
    report = LossReport()
    report.add_loss("harmonic_type", "G94 does not encode spherical vs Cartesian")
    report.add_loss("provenance", "carried as comment text only")
    report.add_loss("references", "carried as comment text only")
    if data.ecps:
        report.add_loss("ecps", "written to the .ecp sidecar, not the .g94")
    return report


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _num(x: float, width: int = 22) -> str:
    """The shortest round-trip decimal of ``x``, right-aligned in a column.

    Two spaces always separate columns: ``repr`` of a double runs to 24
    characters (``-1.2345678901234567e-05``), so a fixed field alone could
    glue a negative coefficient onto the exponent before it.

    libint parses with ``std::stod`` after mapping Fortran ``D`` exponents
    to ``E``, so Python's ``1e-05`` spelling is read exactly.
    """
    return f"  {repr(float(x)):>{width}s}"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _element_ranges(symbols: Iterable[str]) -> str:
    """``"H-Kr, Rb-Rn"`` for a run of symbols, in atomic-number order."""
    from ._element_data import ATOMIC_NUMBER, SYMBOL

    zs = sorted(ATOMIC_NUMBER[s] for s in symbols if s in ATOMIC_NUMBER)
    if not zs:
        return ""
    runs: list[tuple[int, int]] = []
    for z in zs:
        if runs and z == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], z)
        else:
            runs.append((z, z))
    return ", ".join(
        SYMBOL[a] if a == b else f"{SYMBOL[a]}-{SYMBOL[b]}" for a, b in runs
    )


def _comment(text: str, indent: str = "") -> list[str]:
    """Wrap ``text`` into ``! `` comment lines."""
    out: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph.strip():
            out.append("!")
            continue
        body = textwrap.wrap(
            paragraph, width=_WRAP - len(indent),
            subsequent_indent="  ", break_long_words=False, break_on_hyphens=False,
        ) or [""]
        out.extend(f"! {indent}{line}".rstrip() for line in body)
    return out


def render_header(data: BasisSetData, *, sidecar: bool = False) -> list[str]:
    """The ``!`` comment block both rendered files start with.

    Order: identity, generation notice, provenance, originating
    publications, element coverage, then the record's free-text
    description. The publications come before the description so a reader
    (or a header-scanning test) finds every DOI within the first lines.
    """
    lines = [_RULE]
    role = data.role.value if hasattr(data.role, "value") else str(data.role)
    if sidecar:
        lines.append(f"! ECP sidecar for {data.name}")
    # The two lines ``importer_g94`` reads back, so name, role and family
    # survive a text round trip.
    lines.append(f"! Basis: {data.name}")
    lines.append(f"! Role: {role} {data.basis_family or ''}".rstrip())
    lines.extend(_comment(
        f"Rendered by vibeqc.basis_toolkit from the QVF-Basis record "
        f"{data.name}.qvf.json (schema {data.schema_version}, content sha256 "
        f"{data.content_sha256()}). Generated text: do not edit, change the "
        f"record and re-render."
    ))
    prov = data.provenance
    if prov is not None:
        parts: list[str] = []
        if prov.kind is not None:
            kind = prov.kind.value if hasattr(prov.kind, "value") else str(prov.kind)
            parts.append(kind)
        for up in prov.upstream:
            piece = up.name
            if up.version:
                piece += f" {up.version}"
            if up.detail:
                piece += f" ({up.detail})"
            parts.append(piece)
        if prov.retrieval_date:
            parts.append(f"recorded {prov.retrieval_date}")
        if parts:
            lines.extend(_comment("Provenance: " + "; ".join(parts)))
        if prov.run_id:
            lines.extend(_comment(f"vibe-basis run: {prov.run_id}"))
    for ref in data.references:
        text = ref.description or ref.key
        if ref.doi:
            text += f"; DOI: {ref.doi}"
        lines.extend(_comment(f"Originating publication: {text}"))
    if not data.references:
        lines.extend(_comment(
            "Originating publication: none recorded (user-supplied data); "
            "cite the record's provenance."
        ))
    coverage = f"Elements: {_element_ranges(data.elements)} ({len(data.elements)})"
    if data.ecps:
        coverage += (
            f"; ECP: {_element_ranges(data.ecps)} ({len(data.ecps)} blocks"
            f"{'' if sidecar else f' in {data.name}.ecp'})"
        )
    lines.extend(_comment(coverage))
    description = (data.description or "").strip()
    if description:
        lines.append("!")
        lines.extend(_comment(description))
    lines.append(_RULE)
    return lines


# ---------------------------------------------------------------------------
# Orbital shells
# ---------------------------------------------------------------------------


def _shell_lines(shell) -> list[str]:
    am = list(shell.angular_momentum)
    n_prim = len(shell.primitives)
    if am == [0, 1]:
        # SP shell: the model stores N s-contraction primitives followed by
        # the N p-contraction primitives over the same exponents.
        n = n_prim // 2
        if 2 * n != n_prim:
            raise ValueError("SP shell needs an even primitive count")
        out = [f"SP   {n}   1.00"]
        for i in range(n):
            s_prim, p_prim = shell.primitives[i], shell.primitives[i + n]
            if s_prim.exponent != p_prim.exponent:
                raise ValueError("SP shell s and p exponents must agree")
            out.append(f"{_num(s_prim.exponent)}{_num(s_prim.coefficient)}{_num(p_prim.coefficient)}")
        return out
    if len(am) != 1 or am[0] not in _AM_LABEL:
        raise ValueError(f"cannot render shell with angular momentum {am}")
    out = [f"{_AM_LABEL[am[0]]}   {n_prim}   1.00"]
    for prim in shell.primitives:
        out.append(f"{_num(prim.exponent)}{_num(prim.coefficient)}")
    return out


def to_g94(data: BasisSetData, *, header: bool = True) -> str:
    """The orbital ``.g94`` text of ``data``; ECPs go to :func:`to_ecp_sidecar`."""
    lines: list[str] = render_header(data) if header else []
    for sym in data.elems_sorted():
        lines.append("****")
        lines.append(f"{sym}     0")
        for shell in data.elements[sym].shells:
            lines.extend(_shell_lines(shell))
    lines.append("****")
    return "\n".join(lines) + "\n"


def to_g94_file(data: BasisSetData, path: str | Path) -> LossReport:
    """Write the orbital rendering to ``path`` and return the loss report."""
    Path(path).write_text(to_g94(data), encoding="utf-8")
    return loss_report_g94(data)


# ---------------------------------------------------------------------------
# ECP sidecar
# ---------------------------------------------------------------------------


def _ecp_block(entry: ECPEntry) -> list[str]:
    by_l = {int(p.angular_momentum): list(p.terms) for p in entry.potentials}
    lmax = int(entry.angular_momentum_max)
    present = max(by_l) if by_l else -1
    if present != lmax:
        raise ValueError(
            f"{entry.element}: angular_momentum_max {lmax} but the highest "
            f"channel present is {present}; the local channel must be the "
            f"highest one (#207)"
        )
    if lmax >= len(AM_LETTERS):
        raise ValueError(f"{entry.element}: ECP l={lmax} beyond {AM_LETTERS!r}")
    local = AM_LETTERS[lmax]
    lines = [f"{entry.element:<6s} 0", f"{entry.element}-ECP {lmax:5d} {int(entry.n_core_electrons):5d}"]

    def channel(label: str, terms) -> None:
        lines.append(f"{label} potential")
        lines.append(f"  {len(terms)}")
        for t in terms:
            lines.append(f"{int(t.power):d}{_num(t.exponent, 20)}{_num(t.coefficient)}")

    channel(local, by_l[lmax])
    for ell in range(lmax):
        if ell in by_l:
            channel(f"{AM_LETTERS[ell]}-{local}", by_l[ell])
    return lines


def to_ecp_sidecar(data: BasisSetData, *, header: bool = True) -> str | None:
    """The ``.ecp`` sidecar text of ``data``, or ``None`` without ECPs."""
    if not data.ecps:
        return None
    from ._element_data import ATOMIC_NUMBER

    lines: list[str] = render_header(data, sidecar=True) if header else []
    for sym in sorted(data.ecps, key=lambda s: ATOMIC_NUMBER.get(s, 999)):
        lines.extend(_ecp_block(data.ecps[sym]))
    return "\n".join(lines) + "\n"
