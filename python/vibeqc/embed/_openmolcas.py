"""OpenMolcas XFIELD out-of-process runner for embedded-cluster parity.

Productionised from ``studies/embedded-cluster-cas/parity_openmolcas_xfield.py``.
Runs OpenMolcas SCF + RASSCF with a GATEWAY ``XField`` point-charge embedding,
parses the printed energies, and returns them for comparison with vibe-qc's
``vq.embed_cluster``.

OpenMolcas is driven OUT OF PROCESS (CLAUDE.md s10); nothing here imports it.
Per CLAUDE.md s1, no OpenMolcas code or data is bundled with vibe-qc.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_SYM: dict[int, str] = {
    1: "H",
    3: "Li",
    4: "Be",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    11: "Na",
    12: "Mg",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    19: "K",
    20: "Ca",
}


def _discover_openmolcas() -> tuple[str, str, str]:
    """Find pymolcas, MOLCAS root, and Python executable."""
    pym = os.environ.get(
        "OPENMOLCAS_PYMOLCAS",
        os.path.expanduser("~/gitlab/OpenMolcas/build/pymolcas"),
    )
    mol = os.environ.get("MOLCAS", os.path.expanduser("~/gitlab/OpenMolcas/build"))
    py = os.environ.get("OPENMOLCAS_PYTHON", sys.executable)
    if not (os.path.exists(pym) and os.path.isdir(mol)):
        raise RuntimeError(
            f"OpenMolcas not found (pymolcas={pym}, MOLCAS={mol}). "
            f"Set OPENMOLCAS_PYMOLCAS and MOLCAS env vars."
        )
    return pym, mol, py


def run_openmolcas_xfield(
    atoms_bohr: list[tuple[int, tuple[float, float, float]]],
    basis: str,
    ext_pos: list[list[float]],
    ext_q: list[float],
    n_core: int,
    n_act: int,
    n_elec: int,
    spin: int = 1,
    charge: int = 0,
    tight: bool = False,
    cionly: bool = False,
    timeout: int = 900,
) -> tuple[float | None, float | None, str]:
    """Run OpenMolcas SCF + RASSCF with an XFIELD point-charge embedding.

    Parameters
    ----------
    atoms_bohr : list[(Z, (x, y, z))]
        QM cluster atoms in bohr.
    basis : str
        Basis set name (e.g. ``"6-31G"``).
    ext_pos : list[list[float]]
        External point-charge positions (bohr).
    ext_q : list[float]
        External point-charge values (e).
    n_core : int
        Number of inactive (frozen-core) orbitals.
    n_act : int
        Number of active orbitals (RAS2).
    n_elec : int
        Number of active electrons.
    spin : int
        Spin multiplicity.
    charge : int
        Net charge of the QM cluster (passed to ``&SCF Charge=``).
    tight : bool
        Tighten RASSCF convergence to 1e-10/1e-8/1e-8.
    cionly : bool
        RASSCF CIonly (fixed SCF orbitals, CASCI).
    timeout : int
        Subprocess timeout in seconds.

    Returns
    -------
    (E_scf, E_rasscf, note)
        Energies in Hartree, or None if parsing failed.  ``note``
        describes any problems.
    """
    pym, mol, py = _discover_openmolcas()
    wd = tempfile.mkdtemp(prefix="om_xfield_")
    scr = os.path.join(wd, "scr")
    os.makedirs(scr)
    try:
        xyz = f"{len(atoms_bohr)}\nBohr\n" + "\n".join(
            f"{_SYM[z]} {x:.10f} {y:.10f} {zz:.10f}" for z, (x, y, zz) in atoms_bohr
        )
        with open(os.path.join(wd, "job.xyz"), "w") as fh:
            fh.write(xyz + "\n")

        xfield = ""
        if ext_q:
            xfield = (
                f"XField\n{len(ext_q)} 0\n"
                + "\n".join(
                    f"{p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {float(q):.10f}"
                    for p, q in zip(ext_pos, ext_q)
                )
                + "\n"
            )
        thrs = "THRS\n1.0d-10 1.0d-8 1.0d-8\n" if tight else ""
        ci = "CIOnly\n" if cionly else ""
        rasscf = (
            f"&RASSCF\nSpin = {spin}\nnActEl = {n_elec} 0 0\n"
            f"Inactive = {n_core}\nRAS2 = {n_act}\nCIRoot = 1 1 1\n"
            f"{thrs}{ci}"
        )
        scf_charge = f"Charge = {int(charge)}\n" if charge else ""
        inp = (
            f"&GATEWAY\nCoord = job.xyz\nBasis = {basis}\nGroup = C1\n"
            f"{xfield}&SEWARD\n&SCF\n{scf_charge}{rasscf}"
        )
        with open(os.path.join(wd, "job.input"), "w") as fh:
            fh.write(inp)

        env = dict(os.environ, MOLCAS=mol, MOLCAS_WORKDIR=scr, OMP_NUM_THREADS="4")
        proc = subprocess.run(
            [py, pym, "job.input"],
            cwd=wd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        e_scf: float | None = None
        e_ras: float | None = None
        note = ""
        for line in proc.stdout.splitlines():
            m = re.search(r"Total SCF energy\s+(-?\d+\.\d+)", line)
            if m:
                e_scf = float(m.group(1))
            m = re.search(r"RASSCF root number\s+1 Total energy:\s+(-?\d+\.\d+)", line)
            if m:
                e_ras = float(m.group(1))
        if e_scf is None or e_ras is None:
            note = "missing energy (scf=%s ras=%s); stderr tail: %s" % (
                e_scf,
                e_ras,
                (proc.stderr or "")[-400:],
            )
        return e_scf, e_ras, note
    finally:
        shutil.rmtree(wd, ignore_errors=True)
