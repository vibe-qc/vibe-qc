#!/usr/bin/env python
"""GAPW all-electron parity — vibe-qc vs external oracles (OUT OF PROCESS).

Reproduces the historical GPW-AUDIT-005 / 006 / 007 parity checks. For each
core-bearing closed-shell system it runs three *vibe-qc*
routes in-process —

    GDF  = run_rhf_periodic_gamma_gdf      (analytic all-electron reference)
    GPW  = run_periodic_rhf_gpw            (soft FFT grid, NO augmentation)
    GAPW = run_periodic_rhf_gapw           (soft FFT grid + augmentation)

— and compares them against external all-electron oracles launched in a
**separate interpreter** (CLAUDE.md §10: vibe-qc never imports PySCF / GPAW /
CP2K in-process; they are GPLv3 and run as subprocess oracles only, exactly as
``examples/regression/core/runner_{pyscf,gpaw}.py`` do):

    PySCF.pbc.RHF.density_fit()  — absolute all-electron Gaussian periodic HF
    GPAW (grid-PAW, PW mode)     — PAW-referenced -> energy DIFFERENCES only

CP2K is GAPW's native code and the closest absolute oracle, but was
unavailable (no executable) in the audit environment; PySCF.pbc substitutes as
the absolute all-electron reference and vibe-qc's own GDF as the in-process,
same-gauge reference.

Run (threads pinned, per the audit recipe)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        .venv/bin/python examples/regression/gapw_parity/run_gapw_external_parity.py

Set ``VIBEQC_PYSCF_PYTHON`` / ``VIBEQC_GPAW_PYTHON`` to pick the external
interpreter (default: the current one — keeps the subprocess boundary while
staying convenient in a dev venv). Systems containing C/N/O/F/Ne are EXPECTED
to crash the GAPW route (GPW-AUDIT-006); the script catches and reports that.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import warnings

import numpy as np

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_augment import run_periodic_rhf_gapw
from vibeqc.periodic_gapw_grid import PlaneWaveGrid
from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

warnings.filterwarnings("ignore")

_BOHR = 0.52917721067  # Å per bohr
_ZSYM = {1: "H", 2: "He", 3: "Li", 4: "Be", 6: "C", 8: "O", 10: "Ne"}


# --------------------------------------------------------------------------- #
# External oracle scripts — run in a SEPARATE interpreter (never imported).   #
# --------------------------------------------------------------------------- #

_PYSCF_SCRIPT = r"""
import json, sys
import numpy as np
def main():
    spec = json.loads(sys.argv[1])
    from pyscf.pbc import gto as pbcgto, scf as pbcscf
    BOHR = 0.52917721067
    a = np.eye(3) * spec["L"] * BOHR
    lines = [f'{at[0]} {at[1]*BOHR:.10f} {at[2]*BOHR:.10f} {at[3]*BOHR:.10f}'
             for at in spec["atoms"]]
    cell = pbcgto.M(atom="; ".join(lines), a=a, basis=spec["basis"],
                    unit="A", verbose=0)
    mf = pbcscf.RHF(cell).density_fit(); mf.conv_tol = 1e-10; mf.max_cycle = 100
    e = float(mf.kernel())
    print("PYSCF-RESULT:" + json.dumps({"E_ha": e, "converged": bool(mf.converged)}))
main()
"""

_GPAW_SCRIPT = r"""
import json, sys
import numpy as np
def main():
    spec = json.loads(sys.argv[1])
    from ase import Atoms
    from ase.units import Hartree
    from gpaw import GPAW, PW
    BOHR = 0.52917721067
    syms = [at[0] for at in spec["atoms"]]
    pos = np.array([[c*BOHR for c in at[1:]] for at in spec["atoms"]], float)
    atoms = Atoms(symbols=syms, positions=pos,
                  cell=np.eye(3)*spec["L"]*BOHR, pbc=True)
    atoms.calc = GPAW(mode=PW(spec.get("cutoff_ev", 600.0)), xc="LDA",
                      kpts=(1, 1, 1), txt=None, convergence={"energy": 1e-7})
    e = float(atoms.get_potential_energy()) / Hartree
    print("GPAW-RESULT:" + json.dumps({"E_ha": e, "reference": "gpaw_paw"}))
main()
"""


def _external(script: str, marker: str, spec: dict, env_var: str):
    """Run an oracle script in a separate interpreter; parse its result line."""
    python = os.environ.get(env_var, sys.executable)
    try:
        proc = subprocess.run(
            [python, "-c", script, json.dumps(spec)],
            capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):]), None
    if "No module named" in proc.stderr:
        return None, "unavailable"
    return None, f"no result (rc={proc.returncode})"


# --------------------------------------------------------------------------- #
# Systems (positions in bohr).                                                 #
# --------------------------------------------------------------------------- #

def _atom(Z, L=12.0):
    s = core.PeriodicSystem(); s.dim = 3; s.lattice = np.eye(3) * L
    s.unit_cell = [core.Atom(Z, [L / 2, L / 2, L / 2])]
    mol = vq.Molecule([vq.Atom(Z, [L / 2, L / 2, L / 2])], 0, 1)
    atoms = [[_ZSYM[Z], L / 2, L / 2, L / 2]]
    return s, vq.BasisSet(mol, "sto-3g"), L, atoms


def _h2o(L=14.0):
    c = L / 2; r = 1.8088; th = math.radians(104.5)
    O = [c, c, c]
    H1 = [c + r * math.sin(th / 2), c + r * math.cos(th / 2), c]
    H2 = [c - r * math.sin(th / 2), c + r * math.cos(th / 2), c]
    s = core.PeriodicSystem(); s.dim = 3; s.lattice = np.eye(3) * L
    s.unit_cell = [core.Atom(8, O), core.Atom(1, H1), core.Atom(1, H2)]
    mol = vq.Molecule([vq.Atom(8, O), vq.Atom(1, H1), vq.Atom(1, H2)], 0, 1)
    atoms = [["O"] + O, ["H"] + H1, ["H"] + H2]
    return s, vq.BasisSet(mol, "sto-3g"), L, atoms


_SYSTEMS = {
    "He (control)": lambda: _atom(2),
    "Be (1s core, survives pruning)": lambda: _atom(4),
    "Ne (deep 1s, pruned)": lambda: _atom(10),
    "O atom (deep 1s, pruned)": lambda: _atom(8),
    "H2O (O 1s pruned)": _h2o,
}


def _vibeqc_route(fn, *a, **kw):
    try:
        return float(fn(*a, **kw).energy), None
    except Exception as exc:  # noqa: BLE001 — report, do not fix (§7/§9)
        return None, f"{type(exc).__name__}: {str(exc)[:60]}"


def main():
    N = 24
    print(f"# GAPW all-electron parity (STO-3G, Γ, {N}³ grid; threads should "
          f"be pinned)\n# vibe-qc GDF/GPW/GAPW in-process; PySCF/GPAW "
          f"out-of-process (§10)\n")
    for label, builder in _SYSTEMS.items():
        system, basis, L, atoms = builder()
        grid = PlaneWaveGrid(np.eye(3) * L, N, N, N)
        print(f"== {label} ==")

        e_gdf, err = _vibeqc_route(
            vq.run_rhf_periodic_gamma_gdf, system, basis,
            _rhf_opts(), progress=False)
        e_gpw, _ = _vibeqc_route(
            run_periodic_rhf_gpw, system, basis, grid=grid,
            conv_tol_energy=1e-9, max_iter=60, quiet=True)
        e_gapw, gapw_err = _vibeqc_route(
            run_periodic_rhf_gapw, system, basis, grid=grid,
            conv_tol_energy=1e-9, max_iter=60,
            molecular_limit=True, quiet=True)

        pyscf, pe = _external(_PYSCF_SCRIPT, "PYSCF-RESULT:",
                              {"L": L, "basis": "sto-3g", "atoms": atoms},
                              "VIBEQC_PYSCF_PYTHON")

        def fmt(x):
            return f"{x:.8f}" if isinstance(x, float) else str(x)

        print(f"  vibe-qc GDF  (ref) = {fmt(e_gdf)} Ha")
        print(f"  vibe-qc GPW        = {fmt(e_gpw)} Ha")
        print(f"  vibe-qc GAPW       = {fmt(e_gapw if e_gapw is not None else 'CRASH: ' + (gapw_err or ''))}"
              + (f"  (GAPW−GDF = {(e_gapw - e_gdf) * 1e3:+.2f} mHa)"
                 if (e_gapw is not None and e_gdf is not None) else ""))
        if pyscf is not None:
            print(f"  PySCF.pbc (ext)    = {pyscf['E_ha']:.8f} Ha  "
                  f"[exxdiv/Madelung offset vs vibe-qc GDF is HF-convention]")
        else:
            print(f"  PySCF.pbc (ext)    = {pe}")
        print()

    print("# GPAW DeltaE bond-scan corroboration: run the LiH scan here when "
          "GPAW is available (historically GAPW ΔE was off by ~2.76 Ha).")


def _rhf_opts():
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-9
    return opts


if __name__ == "__main__":
    main()
