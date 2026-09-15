"""Run OpenMolcas CASPT2 as an external program (the IC-CASPT2 oracle).

OpenMolcas is *the* local CASPT2 reference for vibe-qc: PySCF ships no
CASPT2 and ORCA's multireference PT2 is NEVPT2, not CASPT2.  Per CLAUDE.md
§10 it is driven **out-of-process** — this runner writes a `&GATEWAY /
&SEWARD / &SCF / &RASSCF / &CASPT2` input, launches ``pymolcas`` in a
subprocess, and parses the printed total energies.  Nothing here imports
OpenMolcas, and vibe-qc never imports it either.

It regenerates the OpenMolcas reference values recorded as constants in
``tests/test_solvers_mrpt_parity.py`` (``_OM_IC``).  Run it directly to
print the table; the test suite does **not** depend on it (the references
are recorded, so the tests run without OpenMolcas installed).

Configuration (no paths are hard-coded — set these in your environment):

* ``OPENMOLCAS_PYMOLCAS`` — path to the ``pymolcas`` driver script.
* ``OPENMOLCAS_PYTHON``  — python that can import ``pyparsing`` (OpenMolcas
  ships its own venv; its ``pymolcas`` shebang often points at a system
  python3 that lacks pyparsing).  Defaults to ``python3``.
* ``MOLCAS``            — the OpenMolcas build/install root (``$MOLCAS``).

By default a fixed CASCI(HF) reference (``&SCF`` then ``&RASSCF ... CIonly``)
is used so the comparison is apples-to-apples with a vibe-qc ``casci``-on-HF
reference; ``ci_only=False`` instead lets RASSCF optimize the orbitals (full
CASSCF reference, matching vibe-qc's ``casscf_options`` composition).
``Group = C1`` and ``IPEAshift`` / ``Frozen`` / ``Imaginary`` are set
explicitly.  The xyz comment line is set to ``Bohr`` so coordinates are read
in bohr (OpenMolcas parses a unit keyword from that line). Embedded-cluster
parity jobs can additionally pass a charged QM cluster plus GATEWAY ``XField``
monopoles; their positions and charges are likewise atomic units (bohr, e).
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

_SYM = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
}

_NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DdEe][-+]?\d+)?)"


def _as_float(value: str) -> float:
    """Parse an OpenMolcas decimal, including Fortran ``D`` exponents."""
    return float(value.replace("D", "E").replace("d", "e"))


def _stream_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


@dataclass
class OpenMolcasResult:
    """Parsed OpenMolcas energies (Hartree) for one CASPT2 job.

    ``casci`` / ``caspt2`` keep the historical root-1 values; multi-root /
    multi-state jobs additionally fill the per-root lists (index = root−1).
    """

    casci: Optional[float]      # RASSCF (CIonly) total energy, root 1
    caspt2: Optional[float]     # (SS-)CASPT2 total energy, root 1
    note: str = ""
    rasscf_roots: list = field(default_factory=list)   # RASSCF per root
    caspt2_roots: list = field(default_factory=list)   # SS-CASPT2 per root
    ms_roots: list = field(default_factory=list)       # MS/XMS-CASPT2 roots
    scf: Optional[float] = None                         # preceding SCF total
    nac_pair: Optional[tuple[int, int]] = None  # zero-based physical roots
    nac: list = field(default_factory=list)  # (n_atoms, 3), bohr^-1


def _discover() -> tuple[str, str, str]:
    pymolcas = os.environ.get("OPENMOLCAS_PYMOLCAS")
    molcas = os.environ.get("MOLCAS")
    python = os.environ.get("OPENMOLCAS_PYTHON", "python3")
    if not pymolcas or not molcas:
        raise RuntimeError(
            "Set OPENMOLCAS_PYMOLCAS (the pymolcas driver), MOLCAS (the build "
            "root), and optionally OPENMOLCAS_PYTHON (a python with pyparsing)."
        )
    return pymolcas, molcas, python


def run_caspt2(
    atoms: Sequence[tuple[int, Sequence[float]]],
    basis: str,
    n_core: int,
    n_act: int,
    n_elec: int,
    spin: int = 1,
    frozen: int = 0,
    ipea: float = 0.0,
    imaginary: float = 0.0,
    timeout: int = 900,
    threads: int = 4,
    ci_only: bool = True,
    nroots: int = 1,
    multistate: Optional[str] = None,
    artifact_dir: Optional[Path] = None,
    *,
    xfield_positions: Sequence[Sequence[float]] = (),
    xfield_charges: Sequence[float] = (),
    charge: int = 0,
    nac_pair: Optional[tuple[int, int]] = None,
) -> OpenMolcasResult:
    """Run one OpenMolcas CASPT2 job (bohr geometry) and parse the energies.

    ``atoms`` is a list of ``(Z, (x, y, z))`` in bohr.  ``n_core`` inactive,
    ``n_act`` active orbitals, ``n_elec`` active electrons.  ``frozen`` is the
    deep-core freeze count (``Frozen=0`` correlates all inactive, matching
    vibe-qc's default IC-CASPT2).

    ``ci_only=True`` (default) keeps the historical fixed-orbital reference
    (``&RASSCF … CIonly`` on the HF orbitals — apples-to-apples with vibe-qc's
    CASCI-on-HF CASPT2).  ``ci_only=False`` lets RASSCF optimize the orbitals,
    i.e. a full CASSCF reference — apples-to-apples with vibe-qc's
    ``method="caspt2"`` + ``casscf_options`` composition; the parsed ``casci``
    field is then the converged CASSCF total energy.

    ``nroots > 1`` requests an equal-weight multi-root reference
    (``CIRoot = N N 1``).  ``multistate`` adds the multi-state CASPT2 step on
    all roots: ``"ms"`` → ``MULTistate = all`` (state-specific Fock,
    Finley 1998), ``"xms"`` → ``XMULtistate = all`` (rotated references +
    state-averaged Fock, Granovsky 2011 / Shiozaki 2011); the final
    MS/XMS-CASPT2 eigenvalues land in ``ms_roots``.

    ``nac_pair=(Q, P)`` additionally requests the zero-based physical-state
    derivative coupling. OpenMolcas requires a full SA-CASSCF reference and
    RI/CD integrals for this analytic route; the runner therefore adds
    ``RICD`` plus the CASPT2, MCLR, and ALASKA NAC directives and returns the
    Cartesian vector in bohr^-1 as :attr:`OpenMolcasResult.nac`.

    ``xfield_positions`` and ``xfield_charges`` describe GATEWAY ``XField``
    monopoles in atomic units (bohr, e). They must have equal length.
    ``charge`` is the integer net charge of the finite QM cluster and is
    passed to ``&SCF``; the external monopoles are not included in that
    electron-count declaration. The preceding SCF total is returned in
    :attr:`OpenMolcasResult.scf`, enabling embedded-versus-bare stabilization
    comparisons without a second parser.
    """
    if multistate not in (None, "ms", "xms"):
        raise ValueError(f"multistate must be None|'ms'|'xms', got {multistate!r}")
    if nac_pair is not None:
        if multistate is None or nroots < 2:
            raise ValueError(
                "nac_pair requires multistate='ms'|'xms' and nroots >= 2"
            )
        if ci_only:
            raise ValueError(
                "nac_pair requires a full SA-CASSCF reference (ci_only=False)"
            )
        if (
            len(nac_pair) != 2
            or nac_pair[0] == nac_pair[1]
            or min(nac_pair) < 0
            or max(nac_pair) >= nroots
        ):
            raise ValueError(
                f"nac_pair must contain two distinct zero-based roots in "
                f"[0, {nroots}), got {nac_pair!r}"
            )
    xfield_positions = tuple(tuple(float(x) for x in p) for p in xfield_positions)
    xfield_charges = tuple(float(q) for q in xfield_charges)
    if len(xfield_positions) != len(xfield_charges):
        raise ValueError(
            "xfield_positions and xfield_charges must have equal length "
            f"({len(xfield_positions)} != {len(xfield_charges)})"
        )
    for index, position in enumerate(xfield_positions):
        if len(position) != 3:
            raise ValueError(
                f"xfield_positions[{index}] must contain three coordinates"
            )
        if not all(math.isfinite(x) for x in position):
            raise ValueError(f"xfield_positions[{index}] must be finite")
    if not all(math.isfinite(q) for q in xfield_charges):
        raise ValueError("xfield_charges must be finite")
    charge_int = int(charge)
    if charge_int != charge:
        raise ValueError(f"charge must be an integer, got {charge!r}")
    pymolcas, molcas, python = _discover()
    if artifact_dir is None:
        wd = tempfile.mkdtemp(prefix="om_caspt2_")
        cleanup = True
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        wd = str(artifact_dir)
        cleanup = False
    scr = os.path.join(wd, "scr")
    os.makedirs(scr, exist_ok=True)
    try:
        xyz = f"{len(atoms)}\nBohr\n" + "\n".join(
            f"{_SYM[z]} {x:.10f} {y:.10f} {zz:.10f}"
            for z, (x, y, zz) in atoms
        )
        with open(os.path.join(wd, "job.xyz"), "w", encoding="utf-8") as fh:
            fh.write(xyz + "\n")
        caspt2 = f"&CASPT2\nIPEAshift = {ipea}\nFrozen = {frozen}\n"
        if imaginary:
            caspt2 += f"Imaginary = {imaginary}\n"
        if multistate == "ms":
            caspt2 += "MULTistate = all\n"
        elif multistate == "xms":
            caspt2 += "XMULtistate = all\n"
        if nac_pair is not None:
            caspt2 += f"NAC = {nac_pair[0] + 1} {nac_pair[1] + 1}\n"
        rasscf = (
            f"&RASSCF\nSpin = {spin}\nnActEl = {n_elec} 0 0\n"
            f"Inactive = {n_core}\nRAS2 = {n_act}\n"
            f"CIRoot = {nroots} {nroots} 1\n"
        )
        if ci_only:
            rasscf += "CIonly\n"

        xfield = ""
        if xfield_charges:
            rows = "\n".join(
                f"{x:.10f} {y:.10f} {z:.10f} {q:.10f}"
                for (x, y, z), q in zip(xfield_positions, xfield_charges)
            )
            xfield = f"XField\n{len(xfield_charges)} 0\n{rows}\n"

        scf = "&SCF\n"
        if charge_int:
            scf += f"Charge = {charge_int}\n"
        inp = (
            f"&GATEWAY\nCoord = job.xyz\nBasis = {basis}\nGroup = C1\n"
            + xfield
            + ("RICD\n" if nac_pair is not None else "")
            + "&SEWARD\n"
            + scf
            + rasscf
            + caspt2
        )
        if nac_pair is not None:
            roots = f"{nac_pair[0] + 1} {nac_pair[1] + 1}"
            inp += f"&MCLR\n\n&ALASKA\nNAC = {roots}\n"
        with open(os.path.join(wd, "job.input"), "w", encoding="utf-8") as fh:
            fh.write(inp)
        env = dict(os.environ, MOLCAS=molcas, MOLCAS_WORKDIR=scr,
                   OMP_NUM_THREADS=str(threads))
        scf_energy = ci = pt = None
        note = ""
        ras_roots: dict = {}
        pt_roots: dict = {}
        ms_states: dict = {}
        nac_rows: list[list[float]] = []
        in_total_nac = False
        try:
            proc = subprocess.run(
                [python, pymolcas, "job.input"], cwd=wd, env=env,
                capture_output=True, text=True, timeout=timeout,
            )
            if artifact_dir is not None:
                (artifact_dir / "stdout.log").write_text(
                    proc.stdout or "",
                    encoding="utf-8",
                )
                (artifact_dir / "stderr.log").write_text(
                    proc.stderr or "",
                    encoding="utf-8",
                )
            for line in (proc.stdout or "").splitlines():
                if "Total derivative coupling" in line:
                    in_total_nac = True
                    nac_rows = []
                    continue
                if in_total_nac and len(nac_rows) < len(atoms):
                    nac_match = re.match(
                        rf"^\s*\S+\s+{_NUMBER}\s+{_NUMBER}\s+{_NUMBER}\s*$",
                        line,
                    )
                    if nac_match:
                        nac_rows.append(
                            [_as_float(nac_match.group(i)) for i in range(1, 4)]
                        )
                m = re.search(r"Total SCF energy\s+" + _NUMBER, line)
                if m:
                    scf_energy = _as_float(m.group(1))
                m = re.search(
                    r"RASSCF root number\s+(\d+) Total energy:\s+" + _NUMBER,
                    line)
                if m:
                    ras_roots[int(m.group(1))] = _as_float(m.group(2))
                # "CASPT2 Root  k" (per-state SS-CASPT2) but NOT the final
                # "MS-CASPT2 Root  k" / "XMS-CASPT2 Root  k" lines; a
                # negative lookbehind keeps the two families apart.
                m = re.search(
                    r"(?<!-)CASPT2 Root\s+(\d+)\s+Total energy:\s+" + _NUMBER,
                    line)
                if m:
                    pt_roots[int(m.group(1))] = _as_float(m.group(2))
                m = re.search(
                    r"X?MS-CASPT2 Root\s+(\d+)\s+Total energy:\s+" + _NUMBER,
                    line)
                if m:
                    ms_states[int(m.group(1))] = _as_float(m.group(2))
            ci = ras_roots.get(1)
            pt = pt_roots.get(1)
            problems: list[str] = []
            if proc.returncode != 0:
                problems.append(
                    f"OpenMolcas exited with status {proc.returncode}"
                )
            missing: list[str] = []
            if scf_energy is None:
                missing.append("SCF")
            if ci is None:
                missing.append("RASSCF")
            if multistate:
                # XMS deliberately prints no per-root SS lines (rotated
                # references have no well-defined SS energy); judge the
                # run by the final multi-state roots instead.
                if len(ms_states) != nroots:
                    missing.append(
                        f"expected {nroots} {multistate.upper()}-CASPT2 "
                        f"roots, parsed {len(ms_states)}"
                    )
            elif pt is None:
                missing.append("CASPT2")
            if nac_pair is not None and len(nac_rows) != len(atoms):
                missing.append(
                    f"expected {len(atoms)} derivative-coupling rows, "
                    f"parsed {len(nac_rows)}"
                )
            if missing:
                problems.append("missing/invalid result: " + ", ".join(missing))
            if problems:
                note = "; ".join(problems)
                note += "; stderr tail: " + (proc.stderr or "")[-200:]
        except subprocess.TimeoutExpired as exc:
            note = "OpenMolcas timed out"
            if artifact_dir is not None:
                stdout = getattr(exc, "stdout", None) or getattr(exc, "output", None)
                (artifact_dir / "stdout.log").write_text(
                    _stream_text(stdout),
                    encoding="utf-8",
                )
                (artifact_dir / "stderr.log").write_text(
                    _stream_text(getattr(exc, "stderr", None)),
                    encoding="utf-8",
                )
        result = OpenMolcasResult(
            casci=ci, caspt2=pt, note=note,
            rasscf_roots=[ras_roots[k] for k in sorted(ras_roots)],
            caspt2_roots=[pt_roots[k] for k in sorted(pt_roots)],
            ms_roots=[ms_states[k] for k in sorted(ms_states)],
            scf=scf_energy,
            nac_pair=nac_pair,
            nac=nac_rows,
        )
        if artifact_dir is not None:
            import json

            (artifact_dir / "parsed.json").write_text(
                json.dumps(
                    {
                        "scf": result.scf,
                        "casci": result.casci,
                        "caspt2": result.caspt2,
                        "note": result.note,
                        "rasscf_roots": result.rasscf_roots,
                        "caspt2_roots": result.caspt2_roots,
                        "ms_roots": result.ms_roots,
                        "nac_pair": result.nac_pair,
                        "nac": result.nac,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return result
    finally:
        if cleanup:
            shutil.rmtree(wd, ignore_errors=True)


# The cases recorded in tests/test_solvers_mrpt_parity.py::_OM_IC, plus the
# IPEA / imaginary-shift / larger-active-space cases used to validate the IC
# engine during the un-gating audit (2026-06-03).
_H2 = [(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.4))]
_H2O = [(8, (0.0, 0.0, 0.0)), (1, (0.0, 1.43, -0.93)), (1, (0.0, -1.43, -0.93))]
_N2_eq = [(7, (0.0, 0.0, 0.0)), (7, (0.0, 0.0, 2.074))]
_LIH = [(3, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 3.0))]
_LIH_STRETCH = [(3, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 6.0))]

_CASES = [
    # (label, atoms, basis, n_core, n_act, n_elec, frozen, ipea, imaginary)
    ("H2/6-31g  CAS(2,2)            ", _H2, "6-31G", 0, 2, 2, 0, 0.0, 0.0),
    ("H2O/STO-3G CAS(4,4) Frozen=0  ", _H2O, "STO-3G", 3, 4, 4, 0, 0.0, 0.0),
    ("H2O/STO-3G CAS(4,4) Frozen=1  ", _H2O, "STO-3G", 3, 4, 4, 1, 0.0, 0.0),
    ("H2O/6-31g CAS(4,4) Frozen=0   ", _H2O, "6-31G", 3, 4, 4, 0, 0.0, 0.0),
    ("N2/STO-3G CAS(6,6) Frozen=0   ", _N2_eq, "STO-3G", 4, 6, 6, 0, 0.0, 0.0),
    ("H2O/6-31g CAS(4,4) IPEA=0.25  ", _H2O, "6-31G", 3, 4, 4, 0, 0.25, 0.0),
    ("N2/STO-3G CAS(6,6) Imag=0.1   ", _N2_eq, "STO-3G", 4, 6, 6, 0, 0.0, 0.1),
]

# The multi-state cases recorded in tests/test_ms_caspt2.py::_OM_MS
# (CIonly references on HF orbitals, SA2 equal weights, IPEA=0, Frozen=0).
_MS_CASES = [
    # (label, atoms, basis, n_core, n_act, n_elec, multistate, imaginary)
    ("LiH/STO-3G R=3 CAS(2,2) MS2  ", _LIH, "STO-3G", 1, 2, 2, "ms", 0.0),
    ("LiH/STO-3G R=3 CAS(2,2) XMS2 ", _LIH, "STO-3G", 1, 2, 2, "xms", 0.0),
    ("LiH/STO-3G R=6 CAS(2,2) MS2  ", _LIH_STRETCH, "STO-3G", 1, 2, 2, "ms", 0.0),
    ("LiH/STO-3G R=6 CAS(2,2) XMS2 ", _LIH_STRETCH, "STO-3G", 1, 2, 2, "xms", 0.0),
    ("H2/6-31g CAS(2,2) MS2        ", _H2, "6-31G", 0, 2, 2, "ms", 0.0),
    ("LiH/STO-3G R=6 MS2 Imag=0.1  ", _LIH_STRETCH, "STO-3G", 1, 2, 2, "ms", 0.1),
    ("LiH/STO-3G R=6 XMS2 Imag=0.1 ", _LIH_STRETCH, "STO-3G", 1, 2, 2, "xms", 0.1),
]


def main() -> None:
    print(f"{'case':<32}{'CASCI(OM)':>16}{'CASPT2(OM)':>16}  note")
    print("-" * 88)
    for label, atoms, basis, nc, na, ne, fz, ip, im in _CASES:
        res = run_caspt2(atoms, basis, nc, na, ne, frozen=fz, ipea=ip,
                         imaginary=im)
        ci = f"{res.casci:.8f}" if res.casci is not None else "None"
        pt = f"{res.caspt2:.8f}" if res.caspt2 is not None else "None"
        print(f"{label:<32}{ci:>16}{pt:>16}  {res.note}")
    print()
    print("multi-state (SA2, CIonly, IPEA=0, Frozen=0)")
    print("-" * 88)
    for label, atoms, basis, nc, na, ne, mst, im in _MS_CASES:
        res = run_caspt2(atoms, basis, nc, na, ne, imaginary=im,
                         nroots=2, multistate=mst)
        ras = " ".join(f"{e:.8f}" for e in res.rasscf_roots)
        ms = " ".join(f"{e:.8f}" for e in res.ms_roots)
        print(f"{label:<32} rasscf [{ras}]  {mst} [{ms}]  {res.note}")


if __name__ == "__main__":
    main()
