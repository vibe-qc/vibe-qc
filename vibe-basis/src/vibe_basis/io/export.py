"""Write an optimised basis so vibe-qc's import tool takes it with no manual step.

vibe-basis must not import vibe-qc, so the hand-over is two files:

* ``<name>.g94``: the shells in Gaussian94 text, the format every QC code
  and the Basis Set Exchange exchange, written at full precision
  (``repr`` of every double);
* ``<name>.provenance.json``: what the record is (``kind``
  ``"vibe-basis"``), the run identity, when it was generated and a
  description of the objective.

``vibe-qc-basis import <name>.g94`` reads both and puts the record into the
user's basis store, where ``BasisSet(mol, "<name>")`` resolves it like a
bundled set (vibe-qc #410).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from ..topology import BasisTopology, Shell

__all__ = ["basis_to_g94", "write_basis_for_import"]

_G94_LABEL = {"S": "S", "SP": "SP", "P": "P", "D": "D", "F": "F", "G": "G"}


def _num(x: float) -> str:
    return f"  {repr(float(x)):>22s}"


def basis_to_g94(topology: BasisTopology, *, name: str, header: Optional[str] = None) -> str:
    """Gaussian94 text of ``topology``; exponents carry the CRYSTAL scale factor."""
    lines = ["! " + line for line in (header or f"Basis: {name}\nRole: orbital vibe-basis").splitlines()]
    from ..backends.crystal_atom import _ELEMENT_SYMBOLS

    z_of = {symbol: z for z, symbol in _ELEMENT_SYMBOLS.items()}
    for element in sorted(topology.atoms, key=lambda s: (z_of.get(s, 999), s)):
        lines.append("****")
        lines.append(f"{element}     0")
        for shell in topology.atoms[element]:
            lines.extend(_shell_lines(shell))
    lines.append("****")
    return "\n".join(lines) + "\n"


def _shell_lines(shell: Shell) -> list[str]:
    label = _G94_LABEL[shell.shell_type]
    scale = float(shell.scale_factor or 1.0)
    out = [f"{label}   {shell.n_primitives}   1.00"]
    for i, exponent in enumerate(shell.exponents):
        row = _num(exponent * scale) + _num(shell.coefficients[i])
        if shell.shell_type == "SP":
            row += _num(shell.coefficients_p[i])
        out.append(row)
    return out


def write_basis_for_import(
    topology: BasisTopology,
    directory: Path | str,
    *,
    name: str,
    run_id: str,
    description: str = "",
    objective: Optional[Mapping[str, Any]] = None,
) -> tuple[Path, Path]:
    """Write ``<name>.g94`` and ``<name>.provenance.json`` under ``directory``.

    Returns the two paths. ``objective`` is any JSON-serialisable summary of
    what was optimised (systems, reference, final value); it lands in the
    sidecar and, through the import tool, in the record's notes.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = name.strip().lower()
    g94 = directory / f"{stem}.g94"
    sidecar = directory / f"{stem}.provenance.json"
    generated = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    g94.write_text(
        basis_to_g94(
            topology, name=stem,
            header=f"Basis: {stem}\nRole: orbital vibe-basis\nvibe-basis run {run_id}, generated {generated}",
        ),
        encoding="utf-8",
    )
    payload: dict[str, Any] = {
        "name": stem,
        "kind": "vibe-basis",
        "run_id": run_id,
        "generated_at": generated,
        "description": description or f"Optimised by vibe-basis, run {run_id}.",
        "upstream": [{"name": "vibe-basis", "identifier": run_id}],
    }
    if objective is not None:
        payload["objective"] = dict(objective)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return g94, sidecar
