"""``{stem}.xyz`` writer -- final geometry in standard XYZ format.

For a molecular ``run_job(..., write_xyz=True)`` (the default) the
final geometry is emitted alongside the existing ``.out`` /
``.system`` / ``.molden`` siblings. The format is the canonical XYZ
that every viewer (Avogadro, Jmol, VMD, PyMOL, MolTUI), every
chem-toolkit (RDKit, OpenBabel, ASE), and every QC code reads.

For optimization runs (``optimize=True``) the ``.xyz`` carries the
optimised geometry; the per-step trajectory remains in ``.traj`` (ASE
binary). For periodic jobs the lattice is *not* emitted here --
``poscar.py`` / ``xsf.py`` / a forthcoming ``.cif`` writer are the
right surfaces; the periodic-driver wiring lands in Phase O5.

Public API
----------

``write_xyz(stem, molecule, *, comment=None, energy_ha=None) -> Path``
    Render the molecule's atoms to ``{stem}.xyz``. Positions are
    converted from bohr (vibe-qc's internal unit) to Ångström.

``format_xyz(molecule, *, comment=None, energy_ha=None) -> str``
    The same content as a string, for callers that want to embed an
    XYZ block in a larger artefact.

XYZ format (per the de-facto specification -- no formal RFC):

    n_atoms
    <comment line -- single line, any printable ASCII>
    <symbol>  <x>  <y>  <z>
    ...

Common ad-hoc extension: append ``energy=<E_in_Ha>`` (or eV) to the
comment line. We do that when the caller passes ``energy_ha=`` so
downstream parsers (ASE, Open Babel) pick it up. Coordinates are
emitted with 10 decimal places -- overkill for visualisation, but
keeps round-trip parity with the bohr -> Å conversion within float64
precision.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .._text_safety import scrub_output_text
from .._stem_paths import stem_sibling


__all__ = ["write_xyz", "format_xyz"]


# Bohr -> Ångström. CODATA 2018: a₀ = 0.529177210903 Å (matches
# `vibeqc.molecule.ANGSTROM_TO_BOHR` for round-trip parity).
BOHR_TO_ANGSTROM = 0.529177210903


# Elements 1-103. Reverse of ``vibeqc.molecule._ATOMIC_NUMBERS`` plus
# the rest of the periodic table that bundled basis sets can reach
# (def2-* covers H-Rn). Beyond Z=103 we fall through to ``"X"`` (the
# XYZ-convention placeholder for an unknown element) rather than
# raising -- a writer should never crash a finished SCF.
_ATOMIC_SYMBOLS: tuple[str, ...] = (
    "X",  # 0 -- placeholder / ghost atom
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr",
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
)


def _symbol(z: int) -> str:
    """Atomic-number -> element symbol. Out-of-range Z falls through
    to ``"X"`` (the XYZ-convention placeholder for an unknown
    element)."""
    if 0 <= z < len(_ATOMIC_SYMBOLS):
        return _ATOMIC_SYMBOLS[z]
    return "X"


def write_xyz(
    stem: os.PathLike | str,
    molecule: Any,
    *,
    comment: str | None = None,
    energy_ha: float | None = None,
) -> Path:
    """Write ``{stem}.xyz`` with the molecule's final geometry.

    Parameters
    ----------
    stem
        Path stem; the ``.xyz`` suffix is appended.
    molecule
        Anything with a ``.atoms`` iterable, each item carrying a
        ``.Z`` (atomic number, int) and a ``.xyz`` (3-element sequence
        of bohr coordinates). The native ``vibeqc.Molecule`` /
        ``Atom`` types satisfy this; so does a duck-typed test stub.
    comment
        Second line of the XYZ file. Defaults to a vibe-qc provenance
        line.
    energy_ha
        Optional total energy in Hartree. When given, appended to the
        comment line as ``energy=<value>`` so ASE / Open Babel can
        recover the SCF result from the geometry file alone.

    Returns
    -------
    pathlib.Path
        The on-disk path of the written ``.xyz``.
    """
    target = stem_sibling(stem, ".xyz")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        format_xyz(molecule, comment=comment, energy_ha=energy_ha),
        encoding="utf-8",
    )
    return target


def format_xyz(
    molecule: Any,
    *,
    comment: str | None = None,
    energy_ha: float | None = None,
) -> str:
    """Return the XYZ-formatted string for ``molecule``. Same content
    that :func:`write_xyz` writes; useful for embedding the geometry
    in a larger artefact.
    """
    atoms = list(molecule.atoms)
    head_comment = comment if comment is not None else (
        "vibe-qc final geometry (bohr -> Å, CODATA 2018 a₀)"
    )
    if energy_ha is not None:
        # Strip any trailing punctuation the caller might have left
        # before appending the energy tag.
        head_comment = head_comment.rstrip()
        if head_comment.endswith("."):
            head_comment = head_comment[:-1]
        head_comment = f"{head_comment} energy={float(energy_ha):.10f}"

    # The comment line is user-controlled (run_job passes the output
    # basename / a caller string); neutralise bidi / zero-width / control
    # chars before it becomes XYZ line 2. See vibeqc.output._text_safety.
    lines: list[str] = [str(len(atoms)), scrub_output_text(head_comment)]
    for atom in atoms:
        sym = _symbol(int(atom.Z))
        x_b, y_b, z_b = atom.xyz
        x = float(x_b) * BOHR_TO_ANGSTROM
        y = float(y_b) * BOHR_TO_ANGSTROM
        z = float(z_b) * BOHR_TO_ANGSTROM
        # %-2s pads single-character symbols so columns align with
        # two-character ones -- purely cosmetic, every reader is fine
        # with arbitrary whitespace.
        lines.append(f"{sym:<2s}  {x: .10f}  {y: .10f}  {z: .10f}")
    # XYZ files conventionally end with a trailing newline.
    return "\n".join(lines) + "\n"
