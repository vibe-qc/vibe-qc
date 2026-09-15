"""Milestone 1 parity: vibe-qc molecular CAS in an external point-charge
field vs OpenMolcas GATEWAY XFIELD (the rigorous oracle).

Same LiH/6-31G CAS(2,2) + same +/-1 point charges as
spike_external_field.py. OpenMolcas is run OUT OF PROCESS (CLAUDE.md
s10); nothing here imports it -- it writes a GATEWAY/SEWARD/SCF/RASSCF
input, runs pymolcas, and parses the printed energies.

XField convention (OpenMolcas gateway.rst): first line `nXF nOrd`,
nOrd=0 => monopole only (`x y z q` per line); ALL entries are in atomic
units (bohr) by default, matching vibe-qc's internal bohr. Coord is fed
via a "Bohr"-header xyz. Energy convention: the nucleus<->XField term is
included; the XField<->XField self-energy is excluded -- the same
convention as vibe-qc's E_nuc = E_nuc(QM-QM) + sum Z_A q_B / R_AB. The
script reports the ext-ext term so any convention offset is visible.

Run (needs the local OpenMolcas build):
    .venv/bin/python studies/embedded-cluster-cas/parity_openmolcas_xfield.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vibeqc as vq  # noqa: E402
from spike_external_field import BOHR, embedded_cas  # noqa: E402

_SYM = {1: "H", 3: "Li", 4: "Be", 6: "C", 7: "N", 8: "O", 9: "F",
        11: "Na", 12: "Mg", 17: "Cl"}


def _discover():
    pym = os.environ.get(
        "OPENMOLCAS_PYMOLCAS",
        os.path.expanduser("~/gitlab/OpenMolcas/build/pymolcas"))
    mol = os.environ.get(
        "MOLCAS", os.path.expanduser("~/gitlab/OpenMolcas/build"))
    py = os.environ.get("OPENMOLCAS_PYTHON", sys.executable)
    if not (os.path.exists(pym) and os.path.isdir(mol)):
        raise RuntimeError(f"OpenMolcas not found (pymolcas={pym}, MOLCAS={mol})")
    return pym, mol, py


def openmolcas_xfield(atoms_bohr, basis, ext_pos, ext_q,
                      n_core, n_act, n_elec, spin=1, charge=0, tight=False,
                      cionly=False, timeout=600):
    """Run OpenMolcas SCF + RASSCF with an XFIELD point-charge embedding.

    Returns (E_scf, E_rasscf, note).  atoms_bohr: [(Z,(x,y,z)) bohr];
    ext_pos: [[x,y,z] bohr]; ext_q: [q]. ``charge`` is the QM cluster net
    charge (e.g. +10 for an [OMg6] ionic cluster); it is passed to &SCF and
    is consistent with the RASSCF Inactive=n_core electron count. ``tight``
    tightens the RASSCF convergence thresholds (THRS) from the loose defaults
    (energy 1e-8, orbital rotation/gradient 1e-4) to 1e-10 / 1e-8 / 1e-8, for
    a clean cross-code CASSCF comparison.
    """
    pym, mol, py = _discover()
    wd = tempfile.mkdtemp(prefix="om_xfield_")
    scr = os.path.join(wd, "scr")
    os.makedirs(scr)
    try:
        xyz = f"{len(atoms_bohr)}\nBohr\n" + "\n".join(
            f"{_SYM[z]} {x:.10f} {y:.10f} {zz:.10f}"
            for z, (x, y, zz) in atoms_bohr)
        with open(os.path.join(wd, "job.xyz"), "w") as fh:
            fh.write(xyz + "\n")

        xfield = ""
        if ext_q:
            xfield = (f"XField\n{len(ext_q)} 0\n" + "\n".join(
                f"{p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {float(q):.10f}"
                for p, q in zip(ext_pos, ext_q)) + "\n")
        thrs = "THRS\n1.0d-10 1.0d-8 1.0d-8\n" if tight else ""
        ci = "CIOnly\n" if cionly else ""
        rasscf = (f"&RASSCF\nSpin = {spin}\nnActEl = {n_elec} 0 0\n"
                  f"Inactive = {n_core}\nRAS2 = {n_act}\nCIRoot = 1 1 1\n"
                  f"{thrs}{ci}")
        scf_charge = f"Charge = {int(charge)}\n" if charge else ""
        inp = (f"&GATEWAY\nCoord = job.xyz\nBasis = {basis}\nGroup = C1\n"
               f"{xfield}&SEWARD\n&SCF\n{scf_charge}{rasscf}")
        with open(os.path.join(wd, "job.input"), "w") as fh:
            fh.write(inp)

        env = dict(os.environ, MOLCAS=mol, MOLCAS_WORKDIR=scr,
                   OMP_NUM_THREADS="4")
        proc = subprocess.run([py, pym, "job.input"], cwd=wd, env=env,
                              capture_output=True, text=True, timeout=timeout)
        e_scf = e_ras = None
        note = ""
        for line in proc.stdout.splitlines():
            m = re.search(r"Total SCF energy\s+(-?\d+\.\d+)", line)
            if m:
                e_scf = float(m.group(1))
            m = re.search(
                r"RASSCF root number\s+1 Total energy:\s+(-?\d+\.\d+)", line)
            if m:
                e_ras = float(m.group(1))
        if e_scf is None or e_ras is None:
            note = ("missing energy (scf=%s ras=%s); stderr tail: %s"
                    % (e_scf, e_ras, (proc.stderr or "")[-400:]))
        return e_scf, e_ras, note
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def ext_ext_energy(ext_pos, ext_q):
    e = 0.0
    for i in range(len(ext_q)):
        for j in range(i + 1, len(ext_q)):
            r = np.linalg.norm(np.asarray(ext_pos[i], float)
                               - np.asarray(ext_pos[j], float))
            e += ext_q[i] * ext_q[j] / r
    return e


def main():
    mol = vq.Molecule([vq.Atom(3, [0.0, 0.0, 0.0]),
                       vq.Atom(1, [0.0, 0.0, 1.60 * BOHR])])
    basis = vq.BasisSet(mol, "6-31g")
    atoms_bohr = [(3, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.60 * BOHR))]
    nae, nao, ncore = 2, 2, 1

    d = 4.0 * BOHR
    ext_pos = [[0.0, 0.0, -d], [0.0, 0.0, 1.60 * BOHR + d]]
    ext_q = [+1.0, -1.0]

    print("=" * 70)
    print("Milestone 1 parity: vibe-qc embedded CAS vs OpenMolcas XFIELD")
    print("=" * 70)
    print(f"  ext-ext self-energy (excluded by both conventions): "
          f"{ext_ext_energy(ext_pos, ext_q):+.8f} Ha")

    for label, ep, eq in (("bare (no field)", [], []),
                          ("XFIELD +/-1 cage", ext_pos, ext_q)):
        vqe = embedded_cas(mol, basis, ep, eq, nae, nao)
        om_scf, om_ras, note = openmolcas_xfield(
            atoms_bohr, "6-31G", ep, eq, ncore, nao, nae)
        print(f"\n-- {label} --")
        if note:
            print(f"   OpenMolcas note: {note}")
        print(f"   {'':10s} {'vibe-qc':>15s} {'OpenMolcas':>15s} {'|Δ|':>11s}")
        for vk, om, nm in (("rhf", om_scf, "RHF/SCF"),
                           ("casscf", om_ras, "CASSCF/RASSCF")):
            if om is None:
                print(f"   {nm:10s} {vqe[vk]:15.8f} {'--':>15s}")
                continue
            dd = abs(vqe[vk] - om)
            tag = "PASS" if dd < 5e-6 else ("~" if dd < 1e-3 else "FAIL")
            print(f"   {nm:13s} {vqe[vk]:15.8f} {om:15.8f} {dd:11.2e}  {tag}")


if __name__ == "__main__":
    main()
