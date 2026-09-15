"""Extended-XYZ writer -- XYZ with lattice carried in the comment line.

Extended XYZ is a de-facto extension of the plain XYZ format, popular
with ASE, pymatgen, OVITO, and i-PI. The atom-count line + per-atom
rows are identical to plain XYZ (so vanilla viewers still load the
file as a non-periodic molecule); the lattice is encoded in the
comment-line key/value pairs::

    n_atoms
    Lattice="ax ay az bx by bz cx cy cz" Properties=species:S:1:pos:R:3 energy=<Ha>
    <sym> <x> <y> <z>
    ...

The ``Lattice`` value is a 9-element row-major flattening of the
lattice in Ångström. The ``Properties`` token mirrors the ASE
convention so ``ase.io.read("output.xyz")`` recovers a PBC ``Atoms``
object with the right cell. ``energy=<Ha>`` is the conventional
single-frame energy tag.

For molecular (non-periodic) jobs use
:func:`vibeqc.output.formats.xyz.write_xyz` -- that path emits plain
XYZ without the lattice extension so the file is unambiguously
non-periodic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .xyz import BOHR_TO_ANGSTROM, _symbol
from .._text_safety import scrub_output_text
from .._stem_paths import stem_sibling


__all__ = ["write_extended_xyz", "format_extended_xyz"]


def write_extended_xyz(
    stem: os.PathLike | str,
    system: Any,
    *,
    energy_ha: float | None = None,
    comment: str | None = None,
) -> Path:
    """Write ``{stem}.xyz`` in Extended-XYZ form with the periodic
    lattice encoded in the comment line.

    Parameters
    ----------
    stem
        Path stem; the ``.xyz`` suffix is appended.
    system
        Anything with ``.lattice`` (3x3 numpy array, columns are
        lattice vectors in bohr) and ``.unit_cell`` (iterable of atoms
        each carrying ``.Z`` and ``.xyz`` in bohr). The native
        :class:`vibeqc.PeriodicSystem` satisfies this; so does a
        duck-typed test stub.
    energy_ha
        Optional SCF total energy in Hartree. Embedded as
        ``energy=<value>`` in the comment line.
    comment
        Extra text appended to the comment line *before* the standard
        ``Lattice=...`` / ``Properties=...`` / ``energy=...`` tags.
        Defaults to a short vibe-qc provenance string.
    """
    target = stem_sibling(stem, ".xyz")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        format_extended_xyz(system, energy_ha=energy_ha,
                             comment=comment),
        encoding="utf-8",
    )
    return target


def format_extended_xyz(
    system: Any,
    *,
    energy_ha: float | None = None,
    comment: str | None = None,
) -> str:
    """Return the Extended-XYZ string for ``system``."""
    lattice = system.lattice  # (3, 3) bohr, columns = vectors
    atoms = list(system.unit_cell)

    # Convert lattice to Å, row-major flattening (a_x a_y a_z b_x b_y ...).
    # The ASE convention is rows = vectors; vibe-qc stores columns =
    # vectors. Transpose to row-vectors first.
    lat_rows: list[list[float]] = []
    for i in range(3):
        lat_rows.append([float(lattice[0, i]) * BOHR_TO_ANGSTROM,
                         float(lattice[1, i]) * BOHR_TO_ANGSTROM,
                         float(lattice[2, i]) * BOHR_TO_ANGSTROM])
    lat_str = " ".join(f"{v:.10f}" for row in lat_rows for v in row)

    bits: list[str] = [f'Lattice="{lat_str}"']
    bits.append("Properties=species:S:1:pos:R:3")
    bits.append("pbc=\"T T T\"")
    if energy_ha is not None:
        bits.append(f"energy={float(energy_ha):.10f}")
    # Scrub bidi / zero-width / control chars from the user comment before
    # the lattice/properties tags are appended (see vibeqc.output._text_safety).
    head = scrub_output_text(comment or "vibe-qc periodic geometry")
    head = f"{head} | " + " ".join(bits)

    lines: list[str] = [str(len(atoms)), head]
    for atom in atoms:
        sym = _symbol(int(atom.Z))
        x_b, y_b, z_b = atom.xyz
        x = float(x_b) * BOHR_TO_ANGSTROM
        y = float(y_b) * BOHR_TO_ANGSTROM
        z = float(z_b) * BOHR_TO_ANGSTROM
        lines.append(f"{sym:<2s}  {x: .10f}  {y: .10f}  {z: .10f}")
    return "\n".join(lines) + "\n"
