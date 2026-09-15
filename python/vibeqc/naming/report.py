"""Emit the IUPAC name of a molecule into the active output channel.

This exists so the SCF runner does not have to know how to name a
molecule. Before the output channel, naming was inlined in
``runner.py``'s job-header block for one reason only: that was the only
place with the ``.out`` file handle in scope. Naming has nothing to do
with SCF.

Now the naming module writes its own line. When no channel is active the
write is a no-op, so calling this from a test or a notebook is safe.

Lives under ``vibeqc.naming`` (the shim that already requires a built
vibe-qc) rather than in the standalone ``vibeqc_naming`` distribution: a
naming library should not know that quantum-chemistry output channels
exist. This module is the seam between the two.
"""

from __future__ import annotations

from ..output import write


__all__ = ["write_iupac_name"]

# vibe-qc carries geometry in bohr; vibeqc_naming's perception code wants
# angstrom. CODATA 2018.
_BOHR_TO_ANGSTROM = 0.529177210903


def write_iupac_name(molecule) -> bool:
    """Write ``  IUPAC name: <name>`` for *molecule*. Returns success.

    Best-effort by contract: naming walks a perception pipeline that can
    fail on exotic coordination, radicals, or anything the organic rules
    do not cover. A job must never die because its molecule could not be
    named, so every failure is swallowed and reported as ``False`` rather
    than raised. That mirrors the ``try/except Exception: pass`` this
    replaces; the return value is the only new information, and it lets a
    caller that cares tell "unnamed" from "named".
    """
    try:
        from vibeqc_naming import name_from_atoms

        atoms = [
            (
                int(atom.Z),
                atom.xyz[0] * _BOHR_TO_ANGSTROM,
                atom.xyz[1] * _BOHR_TO_ANGSTROM,
                atom.xyz[2] * _BOHR_TO_ANGSTROM,
            )
            for atom in molecule.atoms
        ]
        iupac = name_from_atoms(atoms, prefer_trivial=True)
    except Exception:
        return False
    write(f"  IUPAC name: {iupac}\n")
    return True
