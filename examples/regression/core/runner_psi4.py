"""Run the Psi4 molecular reference as an external program.

The parent regression suite must not import Psi4 or call its Python
API in-process (CLAUDE.md § 10). Psi4 is a validation reference, so
this runner serializes the requested calculation to JSON, launches a
separate Python interpreter, lets that external process import and
run Psi4, then parses one machine-readable result line back into
:class:`CodeRow`.

Set ``VIBEQC_PSI4_PYTHON=/path/to/python`` to choose the external
interpreter. The default is the current interpreter, which is
convenient when Psi4 has been pip-installed into the vibe-qc venv.
For the conda distribution, point this at the psi4conda env's
``python``.

Psi4 is the third independent reference (alongside ORCA and PySCF)
for vibe-qc molecular parity. Three independent codes triangulate
better than two: when vibe-qc disagrees with two of three at the
same magnitude, the third tells us whether the disagreement is a
vibe-qc bug or a Psi4-vs-{ORCA,PySCF} library quirk.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .case import CodeRow
from .spec import MethodSpec, MoleculeSpec


_RESULT_MARKER = "VIBEQC-PSI4-RESULT:"


# Map (scf, xc) to Psi4 method string. Psi4 understands the same set
# of LibXC keywords PySCF does. Kept in module scope so the parent
# process can sanity-check coverage without spawning a subprocess.
_PSI4_METHOD_MAP = {
    ("rhf", None):     "hf",
    ("uhf", None):     "hf",                 # Psi4 picks UHF from reference=uhf when mult>1
    ("rks", "lda"):    "svwn",               # Slater + VWN5 == ORCA's "LDA" / vibeqc's "LDA"
    ("rks", "pbe"):    "pbe",
    ("rks", "blyp"):   "blyp",
    # FLAVOR FIX: Psi4's "b3lyp" is the Gaussian/VWN-RPA flavor
    # (since Psi4 1.2); vibe-qc's bare "b3lyp" is the ORCA/VWN5
    # flavor, so the matching Psi4 spelling is "b3lyp5". The old
    # bare-name pairing silently crossed the ~10-15 mHa/heavy-atom
    # flavor gap. vibe-qc's "b3lyp/g" / "b3lypg" pair with Psi4's
    # bare "b3lyp".
    ("rks", "b3lyp"):   "b3lyp5",
    ("rks", "b3lyp5"):  "b3lyp5",
    ("rks", "b3lyp/g"): "b3lyp",
    ("rks", "b3lypg"):  "b3lyp",
    ("uks", "lda"):    "svwn",
    ("uks", "pbe"):    "pbe",
    ("uks", "blyp"):   "blyp",
    ("uks", "b3lyp"):   "b3lyp5",
    ("uks", "b3lyp5"):  "b3lyp5",
    ("uks", "b3lyp/g"): "b3lyp",
    ("uks", "b3lypg"):  "b3lyp",
}


def _psi4_method(method: MethodSpec) -> str:
    if method.post == "mp2":
        return "mp2"
    key = (method.scf, (method.xc or "").lower() or None)
    psi4_kw = _PSI4_METHOD_MAP.get(key)
    if psi4_kw is None:
        raise NotImplementedError(
            f"runner_psi4: no Psi4 method mapping for "
            f"scf={method.scf!r} xc={method.xc!r} post={method.post!r}"
        )
    return psi4_kw


_PSI4_EXTERNAL_SCRIPT = r'''
import json
import sys
import time
import traceback

RESULT_MARKER = "VIBEQC-PSI4-RESULT:"


def emit(payload):
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)


def run_molecule(payload):
    import psi4

    psi4.core.be_quiet()

    atom_lines = []
    for atom in payload["atoms"]:
        xyz = atom["xyz_ang"]
        atom_lines.append(
            f"{atom['symbol']} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
        )
    geom = "\n".join(
        [
            f"{int(payload['charge'])} {int(payload['multiplicity'])}",
            *atom_lines,
            "units angstrom",
            "symmetry c1",
            "no_reorient",
            "no_com",
        ]
    )
    psi4.geometry(geom)

    psi4.set_options({
        "basis": payload["basis"],
        "reference": payload["reference"],
        "scf_type": payload["scf_type"],
        "e_convergence": float(payload["conv_tol_energy"]),
        "maxiter": int(payload["max_iter"]),
    })

    method = payload["psi4_method"]
    t0 = time.perf_counter()
    energy = float(psi4.energy(method))
    wall = time.perf_counter() - t0

    emit({
        "status": "ok",
        "code_version": psi4.__version__,
        "energy_ha": energy,
        "wall_s": wall,
        "converged": True,
        "n_iter": None,
    })


def main():
    payload = json.loads(sys.stdin.read())
    try:
        if payload["kind"] == "molecule":
            run_molecule(payload)
        else:
            raise ValueError(f"unknown payload kind {payload['kind']!r}")
    except ModuleNotFoundError as exc:
        if exc.name == "psi4":
            emit({
                "status": "unavailable",
                "code_version": "unknown",
                "note": f"psi4 not importable in external process: {exc}",
            })
        else:
            emit({
                "status": "error",
                "code_version": "unknown",
                "note": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
    except Exception as exc:
        emit({
            "status": "error",
            "code_version": "unknown",
            "note": f"{type(exc).__name__}: {str(exc)[:200]}",
            "traceback": traceback.format_exc(),
        })


if __name__ == "__main__":
    main()
'''


def _psi4_python() -> str:
    return os.environ.get("VIBEQC_PSI4_PYTHON", sys.executable)


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


def _run_external_psi4(
    payload: dict[str, Any], log_path: Path, cwd: Path
) -> dict[str, Any]:
    python = _psi4_python()
    _append_log(log_path, f"  external Psi4 command: {python} -c <script>")
    proc = subprocess.run(
        [python, "-c", _PSI4_EXTERNAL_SCRIPT],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=str(cwd),
    )
    if proc.stdout:
        _append_log(log_path, proc.stdout)
    if proc.stderr:
        _append_log(log_path, "  external Psi4 stderr:")
        _append_log(log_path, proc.stderr)

    result = _parse_external_result(proc.stdout)
    if result is not None:
        result.setdefault("returncode", proc.returncode)
        return result

    if "No module named 'psi4'" in proc.stderr:
        return {
            "status": "unavailable",
            "code_version": "unknown",
            "note": "psi4 not importable in external process",
            "returncode": proc.returncode,
        }
    return {
        "status": "error",
        "code_version": "unknown",
        "note": (
            "external Psi4 process emitted no result marker "
            f"(returncode={proc.returncode})"
        ),
        "returncode": proc.returncode,
    }


def _apply_result(row: CodeRow, result: dict[str, Any], n_atoms: int) -> CodeRow:
    row.n_atoms = int(n_atoms) if n_atoms > 0 else None
    row.code_version = str(result.get("code_version") or "unknown")
    status = str(result.get("status") or "error")
    if status == "unavailable":
        row.status = "unavailable"
        row.note = str(result.get("note") or "external Psi4 unavailable")
        return row
    if status != "ok":
        row.status = "error"
        row.note = str(result.get("note") or "external Psi4 failed")
        return row

    row.energy_ha = float(result["energy_ha"])
    row.wall_s = float(result.get("wall_s") or 0.0)
    row.converged = bool(result.get("converged", False))
    row.n_iter = result.get("n_iter")
    if n_atoms > 0:
        row.energy_per_atom_ha = row.energy_ha / n_atoms
    if not row.converged:
        row.note = "psi4: SCF did not converge"
    return row


def run_molecule_case(
    *, run_id: str, target: str, spec: MoleculeSpec, basis_name: str,
    method: MethodSpec,
    conv_tol_energy: Optional[float] = None,
    max_iter: Optional[int] = None,
    log_path: Path,
    workdir: Path,
) -> CodeRow:
    """Run one Psi4 molecular case via an external Psi4 subprocess.

    Returns a CodeRow with status='unavailable' if Psi4 isn't
    importable in the external interpreter.
    """
    if conv_tol_energy is None:
        conv_tol_energy = spec.default_conv_tol_energy
    if max_iter is None:
        max_iter = spec.default_max_iter

    row = CodeRow(
        run_id=run_id, target=target, system_id=spec.id, family=spec.family,
        basis=basis_name, method_id=method.id, kmesh="mol",
        code="psi4", code_version="unknown",
        n_atoms=len(spec.atoms),
        n_electrons=sum(at.z for at in spec.atoms) - int(spec.charge),
    )

    _append_log(log_path, "\n" + "=" * 78)
    _append_log(
        log_path,
        (
            f"  psi4 external | {spec.id} | {basis_name} | {method.id} | "
            f"mol | target={target}"
        ),
    )
    _append_log(log_path, "=" * 78)

    try:
        psi4_method = _psi4_method(method)
    except NotImplementedError as exc:
        row.status = "error"
        row.note = str(exc)
        _append_log(log_path, f"  {row.note}")
        return row

    # Spin reference. Psi4 picks UHF/UKS automatically when
    # reference='uhf' is set; for closed-shell stick with rhf/rks.
    if method.scf in ("uhf", "uks") or int(spec.multiplicity) > 1:
        reference = "uhf"
    elif method.scf in ("rhf", "rks"):
        reference = "rhf"
    else:
        reference = "rhf"

    case_workdir = workdir / f"{spec.id}__{basis_name}__{method.id}"
    case_workdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "kind": "molecule",
        "atoms": [
            {"symbol": at.symbol, "xyz_ang": list(at.xyz_ang)}
            for at in spec.atoms
        ],
        "charge": int(spec.charge),
        "multiplicity": int(spec.multiplicity),
        "basis": basis_name,
        "psi4_method": psi4_method,
        "reference": reference,
        "scf_type": "df" if method.df else "direct",
        "conv_tol_energy": float(conv_tol_energy),
        "max_iter": int(max_iter),
    }

    _append_log(
        log_path,
        (
            f"  psi4 method: {psi4_method}, basis: {basis_name}, "
            f"reference: {reference}, scf_type: {payload['scf_type']}"
        ),
    )

    result = _run_external_psi4(payload, log_path, case_workdir)
    row = _apply_result(row, result, len(spec.atoms))
    if row.energy_ha is not None:
        _append_log(
            log_path,
            (
                f"  E = {row.energy_ha:.10f} Ha   "
                f"(wall {row.wall_s:.2f} s, converged={row.converged}, "
                f"df={method.df})"
            ),
        )
    else:
        _append_log(log_path, f"  {row.status}: {row.note}")
    return row
