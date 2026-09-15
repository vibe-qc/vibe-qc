"""Run GPAW as an external program (out-of-process parity reference).

GPAW is **GPLv3+**. The parent suite — and all of ``python/vibeqc/`` —
must never import GPAW or call it in-process (that would create a
GPL-derivative; vibe-qc is MPL-2.0, CLAUDE.md §1/§10). This runner is the
sanctioned boundary: it serialises the requested calculation to JSON,
launches a *separate* Python interpreter that imports and runs GPAW, then
parses one machine-readable result line back into :class:`CodeRow`. We
reproduce GPAW's *numbers*, never its code.

GPAW is grid-PAW: its total energy is referenced to the sum of per-atom
PAW reference energies, so it is **not** directly comparable to a Gaussian
all-electron or GAPW total. Use this reference for energy **differences**
(binding curves, relative energies, bond-length minima) where the per-atom
reference cancels — that is the meaningful all-electron parity for the GAPW
augmentation work.

Set ``VIBEQC_GPAW_PYTHON=/path/to/python`` to choose the external
interpreter (default: the current one — keeps the subprocess boundary while
staying convenient in dev envs).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

from .case import CodeRow
from .spec import MethodSpec, PeriodicSpec


_RESULT_MARKER = "VIBEQC-GPAW-RESULT:"


_GPAW_EXTERNAL_SCRIPT = r'''
import json
import importlib
import sys
import time
import traceback

RESULT_MARKER = "VIBEQC-GPAW-RESULT:"

# vibe-qc xc name -> GPAW xc name (GPAW routes through libxc).
XC_MAP = {
    "lda": "LDA",
    "pbe": "PBE",
    "blyp": "BLYP",
    "pbe0": "PBE0",
    # FLAVOR FIX: GPAW's "B3LYP" routes through libxc ->
    # HYB_GGA_XC_B3LYP, the Gaussian/VWN-RPA flavor; vibe-qc's bare
    # "b3lyp" is the ORCA/VWN5 flavor, so it must map to the libxc
    # B3LYP5 name. The old bare-name pairing silently crossed the
    # ~10-15 mHa/heavy-atom flavor gap.
    "b3lyp": "HYB_GGA_XC_B3LYP5",
    "b3lyp5": "HYB_GGA_XC_B3LYP5",
    "b3lypg": "B3LYP",
    "b3lyp/g": "B3LYP",
}


def emit(payload):
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)


def gpaw_xc(xc):
    if xc is None:
        return "LDA"
    return XC_MAP.get(str(xc).lower(), str(xc).upper())


def run_periodic(payload):
    import numpy as np
    from ase import Atoms
    from ase.units import Hartree
    gpaw = importlib.import_module("gpaw")
    GPAW = gpaw.GPAW
    PW = gpaw.PW

    lattice_ang = np.asarray(payload["lattice_ang"], dtype=float)
    symbols = [a["symbol"] for a in payload["atoms"]]
    frac = np.asarray([a["frac"] for a in payload["atoms"]], dtype=float)
    positions = frac @ lattice_ang  # fractional -> cartesian (Å)

    atoms = Atoms(symbols=symbols, positions=positions,
                  cell=lattice_ang, pbc=True)

    method = payload["method"]
    if method["scf"] not in ("rhf", "rks", "uhf", "uks"):
        raise NotImplementedError(f"gpaw: scf={method['scf']!r}")
    # GPAW is a DFT code — pure HF is not its target. Map RHF/RKS->spin-paired,
    # UHF/UKS->spin-polarised; xc per the method (HF maps to EXX only if asked).
    spinpol = method["scf"] in ("uhf", "uks")
    xc = "EXX" if method["scf"] in ("rhf", "uhf") and not method.get("xc") \
        else gpaw_xc(method.get("xc"))

    kmesh = tuple(int(x) for x in payload["kmesh"])
    cutoff_ev = float(payload.get("cutoff_ev", 400.0))

    calc = GPAW(
        mode=PW(cutoff_ev),
        xc=xc,
        kpts=kmesh,
        spinpol=spinpol,
        txt=None,
        convergence={"energy": float(payload["conv_tol_energy"]) * Hartree},
        maxiter=int(payload["max_iter"]),
    )
    atoms.calc = calc

    print(f"external GPAW: {len(symbols)} atoms, PW({cutoff_ev} eV), "
          f"xc={xc}, kpts={kmesh}, spinpol={spinpol}", flush=True)
    print("external GPAW backend: grid-PAW (PW mode); energy is "
          "PAW-referenced -> use for DIFFERENCES, not absolute totals",
          flush=True)

    t0 = time.perf_counter()
    e_ev = float(atoms.get_potential_energy())
    wall = time.perf_counter() - t0

    emit({
        "status": "ok",
        "code_version": getattr(gpaw, "__version__", "unknown"),
        "energy_ha": e_ev / Hartree,
        "energy_ev": e_ev,
        "energy_reference": "gpaw_paw",  # NOT an absolute all-electron total
        "wall_s": wall,
        "converged": bool(calc.scf.converged),
        "n_iter": int(getattr(calc.scf, "niter", 0)) or None,
    })


def main():
    payload = json.loads(sys.stdin.read())
    try:
        if payload["kind"] == "periodic":
            run_periodic(payload)
        else:
            raise ValueError(f"unknown payload kind {payload['kind']!r}")
    except ModuleNotFoundError as exc:
        if exc.name in ("gpaw", "ase"):
            emit({"status": "unavailable", "code_version": "unknown",
                  "note": f"{exc.name} not importable in external process: {exc}"})
        else:
            emit({"status": "error", "code_version": "unknown",
                  "note": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()})
    except Exception as exc:
        emit({"status": "error", "code_version": "unknown",
              "note": f"{type(exc).__name__}: {str(exc)[:200]}",
              "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
'''


def _gpaw_python() -> str:
    return os.environ.get("VIBEQC_GPAW_PYTHON", sys.executable)


def _method_payload(method: MethodSpec) -> dict[str, Any]:
    return {"id": method.id, "scf": method.scf, "xc": method.xc}


def _append_log(log_path: Path, text: str) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
        fh.flush()


def _parse_external_result(stdout: str) -> Optional[dict[str, Any]]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    return None


def _run_external_gpaw(payload: dict[str, Any], log_path: Path) -> dict[str, Any]:
    python = _gpaw_python()
    _append_log(log_path, f"  external GPAW command: {python} -c <script>")
    proc = subprocess.run(
        [python, "-c", _GPAW_EXTERNAL_SCRIPT],
        input=json.dumps(payload), text=True, capture_output=True,
    )
    if proc.stdout:
        _append_log(log_path, proc.stdout)
    if proc.stderr:
        _append_log(log_path, "  external GPAW stderr:")
        _append_log(log_path, proc.stderr)
    result = _parse_external_result(proc.stdout)
    if result is not None:
        result.setdefault("returncode", proc.returncode)
        return result
    if "No module named 'gpaw'" in proc.stderr or "No module named 'ase'" in proc.stderr:
        return {"status": "unavailable", "code_version": "unknown",
                "note": "gpaw/ase not importable in external process",
                "returncode": proc.returncode}
    return {"status": "error", "code_version": "unknown",
            "note": f"external GPAW emitted no result marker (rc={proc.returncode})",
            "returncode": proc.returncode}


def _apply_result(row: CodeRow, result: dict[str, Any], n_atoms: int) -> CodeRow:
    row.n_atoms = int(n_atoms) if n_atoms > 0 else None
    row.code_version = str(result.get("code_version") or "unknown")
    status = str(result.get("status") or "error")
    if status == "unavailable":
        row.status = "unavailable"
        row.note = str(result.get("note") or "external GPAW unavailable")
        return row
    if status != "ok":
        row.status = "error"
        row.note = str(result.get("note") or "external GPAW failed")
        return row
    row.energy_ha = float(result["energy_ha"])
    row.wall_s = float(result.get("wall_s") or 0.0)
    row.converged = bool(result.get("converged", False))
    row.n_iter = result.get("n_iter")
    if n_atoms > 0:
        row.energy_per_atom_ha = row.energy_ha / n_atoms
    # GPAW total is PAW-referenced; flag it so comparisons use differences.
    row.note = "gpaw PAW-referenced energy (compare differences, not absolute)"
    if not row.converged:
        row.note = "gpaw: SCF did not converge within maxiter"
    return row


def run_periodic_case(
    *, run_id: str, target: str, spec: PeriodicSpec, basis_name: str,
    method: MethodSpec, kmesh: Tuple[int, int, int],
    conv_tol_energy: Optional[float] = None, max_iter: Optional[int] = None,
    cutoff_ev: float = 400.0, log_path: Path,
) -> CodeRow:
    """Run the external GPAW grid-PAW reference and return its CodeRow.

    ``basis_name`` is accepted for API parity with the other runners but is
    unused — GPAW uses PAW setups + a plane-wave basis (``cutoff_ev``), not a
    Gaussian basis set.
    """
    if conv_tol_energy is None:
        conv_tol_energy = spec.default_conv_tol_energy
    if max_iter is None:
        max_iter = spec.default_max_iter

    row = CodeRow(
        run_id=run_id, target=target, system_id=spec.id, family=spec.family,
        basis=f"PW({cutoff_ev}eV)", method_id=method.id,
        kmesh="x".join(str(k) for k in kmesh),
        code="gpaw", code_version="unknown",
        n_atoms=len(spec.atoms),
    )

    _append_log(log_path, "\n" + "=" * 78)
    _append_log(log_path, (
        f"  gpaw external | {spec.id} | PW({cutoff_ev}eV) | {method.id} | "
        f"kmesh={kmesh} | target={target}"))
    _append_log(log_path, "=" * 78)

    payload = {
        "kind": "periodic",
        "lattice_ang": spec.lattice_ang,
        "atoms": [{"symbol": at.symbol, "frac": at.frac} for at in spec.atoms],
        "method": _method_payload(method),
        "kmesh": kmesh,
        "cutoff_ev": cutoff_ev,
        "conv_tol_energy": conv_tol_energy,
        "max_iter": max_iter,
    }
    result = _run_external_gpaw(payload, log_path)
    row = _apply_result(row, result, len(spec.atoms))
    if row.energy_ha is not None:
        _append_log(log_path, (
            f"  E/cell = {row.energy_ha:.10f} Ha (PAW-ref)   "
            f"({row.n_iter} iters, wall {row.wall_s:.1f} s, "
            f"converged={row.converged})"))
    else:
        _append_log(log_path, f"  {row.status}: {row.note}")
    return row
