"""Run the PySCF reference as an external program.

The parent regression suite must not import PySCF or call PySCF
functions in-process. PySCF is a validation reference, so this runner
serializes the requested calculation to JSON, launches a separate
Python interpreter, lets that external process import and run PySCF,
then parses one machine-readable result line back into :class:`CodeRow`.

Set ``VIBEQC_PYSCF_PYTHON=/path/to/python`` to choose the external
interpreter. The default is the current interpreter, which is convenient
for developer environments while preserving the subprocess boundary.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from .case import CodeRow
from .spec import MethodSpec, MoleculeSpec, PeriodicSpec


_RESULT_MARKER = "VIBEQC-PYSCF-RESULT:"


_PYSCF_EXTERNAL_SCRIPT = r'''
import json
import importlib
import sys
import time
import traceback

RESULT_MARKER = "VIBEQC-PYSCF-RESULT:"

XC_MAP = {
    "lda": "slater,vwn5",
    "pbe": "pbe,pbe",
    "blyp": "b88,lyp",
    # B3LYP flavor pairing: vibe-qc's bare "b3lyp" is the ORCA/VWN5
    # flavor (maintainer ruling, 2026-06-11) — PySCF's matching
    # spelling is "b3lyp5" (PySCF's plain "b3lyp" is the Gaussian/
    # VWN-RPA variant == vibe-qc's "b3lyp/g" / "b3lypg").
    "b3lyp": "b3lyp5",
    "b3lyp5": "b3lyp5",
    "b3lypg": "b3lyp",
    "b3lyp/g": "b3lyp",
    "hf": "hf",
}


def emit(payload):
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)


def pyscf_xc(xc):
    if xc is None:
        return "slater,vwn5"
    return XC_MAP.get(str(xc).lower(), str(xc).lower())


def make_periodic_mf(method, cell, kmesh, pbc_dft, pbc_scf):
    gamma = tuple(int(x) for x in kmesh) == (1, 1, 1)
    if gamma:
        if method["scf"] == "rks":
            mf = pbc_dft.RKS(cell)
            mf.xc = pyscf_xc(method.get("xc"))
        elif method["scf"] == "rhf":
            mf = pbc_scf.RHF(cell)
        elif method["scf"] == "uks":
            mf = pbc_dft.UKS(cell)
            mf.xc = pyscf_xc(method.get("xc"))
        elif method["scf"] == "uhf":
            mf = pbc_scf.UHF(cell)
        else:
            raise NotImplementedError(f"pyscf periodic: scf={method['scf']!r}")
    else:
        kpts = cell.make_kpts(list(kmesh))
        if method["scf"] == "rks":
            mf = pbc_dft.KRKS(cell, kpts)
            mf.xc = pyscf_xc(method.get("xc"))
        elif method["scf"] == "rhf":
            mf = pbc_scf.KRHF(cell, kpts)
        elif method["scf"] == "uks":
            mf = pbc_dft.KUKS(cell, kpts)
            mf.xc = pyscf_xc(method.get("xc"))
        elif method["scf"] == "uhf":
            mf = pbc_scf.KUHF(cell, kpts)
        else:
            raise NotImplementedError(f"pyscf periodic: scf={method['scf']!r}")
    return mf.density_fit()


def run_periodic(payload):
    import numpy as np
    pyscf = importlib.import_module("pyscf")
    pbc_dft = importlib.import_module("pyscf.pbc.dft")
    pbc_gto = importlib.import_module("pyscf.pbc.gto")
    pbc_scf = importlib.import_module("pyscf.pbc.scf")

    lattice_ang = np.asarray(payload["lattice_ang"], dtype=float)
    atom_lines = []
    for atom in payload["atoms"]:
        frac = np.asarray(atom["frac"], dtype=float)
        xyz = lattice_ang @ frac
        atom_lines.append(
            f"{atom['symbol']} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
        )
    cell_kwargs = dict(
        atom="; ".join(atom_lines),
        a=lattice_ang.tolist(),
        basis=payload["basis"],
        unit="A",
        verbose=0,
    )
    # Set spin for open-shell: PySCF periodic uses cell.spin = n_alpha - n_beta
    # (not multiplicity).  For multi-k open-shell, override mf.nelec in the
    # caller to pin the correct per-cell occupation; see the OPEN-SHELL TRAP
    # note in multik_exchange_split_refs_pyscf.py.
    if payload.get("multiplicity", 1) > 1:
        cell_kwargs["spin"] = int(payload["multiplicity"]) - 1
    # Low-dimensional PBC. A 1-D chain / 2-D slab needs the Coulomb operator
    # truncated in the non-periodic directions (cell.dimension < 3) so that
    # spurious inter-image Coulomb cannot contaminate the (correlation) energy.
    # PySCF's convention: the periodic axes are the FIRST `dimension` lattice
    # vectors, and low-dimensional GDF needs low_dim_ft_type='inf_vacuum'.
    if payload.get("dimension") is not None:
        cell_kwargs["dimension"] = int(payload["dimension"])
    if payload.get("low_dim_ft_type") is not None:
        cell_kwargs["low_dim_ft_type"] = payload["low_dim_ft_type"]
    cell = pbc_gto.M(**cell_kwargs)
    print(
        f"external PySCF cell: nbas={cell.nbas}, nelectron={cell.nelectron}, "
        f"dimension={cell.dimension}",
        flush=True,
    )
    print("external PySCF backend: pbc density_fit() / GDF", flush=True)

    mf = make_periodic_mf(
        payload["method"], cell, payload["kmesh"], pbc_dft, pbc_scf
    )
    mf.conv_tol = float(payload["conv_tol_energy"])
    mf.max_cycle = int(payload["max_iter"])
    mf.verbose = int(payload.get("verbose", 3))

    t0 = time.perf_counter()
    e_scf = float(mf.kernel())
    wall = time.perf_counter() - t0

    result_energy = e_scf
    e_corr = None
    if payload["method"].get("post") == "mp2":
        # Periodic MP2 post-step on the chosen oracle (§10, out of process).
        # KMP2 for a sampled k-mesh; the molecular MP2 driver for a Gamma-only
        # single-k pbc mean-field (pyscf.pbc exposes no non-K MP2 class). Both
        # read the converged pbc DF mean-field's orbitals/energies.
        gamma = tuple(int(x) for x in payload["kmesh"]) == (1, 1, 1)
        t1 = time.perf_counter()
        if gamma:
            mp2 = importlib.import_module("pyscf.mp").MP2(mf, frozen=0)
        else:
            mp2 = importlib.import_module("pyscf.pbc.mp").KMP2(mf, frozen=0)
        mp2.verbose = 0
        e_corr = float(mp2.kernel()[0])
        wall += time.perf_counter() - t1
        result_energy = e_scf + e_corr
        print(
            f"external PySCF periodic post-MP2: e_corr/cell={e_corr:.10f} Ha "
            f"({'MP2@Gamma' if gamma else 'KMP2'})",
            flush=True,
        )

    emit({
        "status": "ok",
        "code_version": pyscf.__version__,
        "energy_ha": result_energy,
        "e_hf_ha": e_scf,
        "e_corr_ha": e_corr,
        "n_basis_functions": int(cell.nao_nr()),
        "n_basis_shells": int(cell.nbas),
        "n_electrons": int(cell.nelectron),
        "wall_s": wall,
        "converged": bool(getattr(mf, "converged", False)),
        "n_iter": int(getattr(mf, "cycles", 0)) or None,
    })


def run_molecule(payload):
    pyscf = importlib.import_module("pyscf")
    mol_dft = importlib.import_module("pyscf.dft")
    mol_gto = importlib.import_module("pyscf.gto")
    mol_scf = importlib.import_module("pyscf.scf")

    atom_lines = [
        (
            f"{atom['symbol']} {atom['xyz_ang'][0]:.10f} "
            f"{atom['xyz_ang'][1]:.10f} {atom['xyz_ang'][2]:.10f}"
        )
        for atom in payload["atoms"]
    ]
    mol = mol_gto.M(
        atom="; ".join(atom_lines),
        basis=payload["basis"],
        unit="A",
        charge=int(payload["charge"]),
        spin=int(payload["multiplicity"]) - 1,
        verbose=0,
    )
    print(
        f"external PySCF molecule: nbas={mol.nbas}, nelectron={mol.nelectron}",
        flush=True,
    )

    method = payload["method"]
    if method["scf"] == "rhf":
        mf = mol_scf.RHF(mol)
    elif method["scf"] == "rks":
        mf = mol_dft.RKS(mol)
        mf.xc = pyscf_xc(method.get("xc"))
    elif method["scf"] == "uhf":
        mf = mol_scf.UHF(mol)
    elif method["scf"] == "uks":
        mf = mol_dft.UKS(mol)
        mf.xc = pyscf_xc(method.get("xc"))
    else:
        raise NotImplementedError(f"pyscf molecule: scf={method['scf']!r}")

    if method.get("df"):
        mf = mf.density_fit(auxbasis=method.get("aux_basis"))
        print(
            f"external PySCF molecular density_fit: aux={method.get('aux_basis')!r}",
            flush=True,
        )

    mf.conv_tol = float(payload["conv_tol_energy"])
    mf.max_cycle = int(payload["max_iter"])
    mf.verbose = int(payload.get("verbose", 0))

    t0 = time.perf_counter()
    e_scf = float(mf.kernel())
    scf_wall = time.perf_counter() - t0
    result_energy = e_scf
    post_wall = 0.0

    if method.get("post") == "mp2":
        mol_mp = importlib.import_module("pyscf.mp")

        t0 = time.perf_counter()
        # Keep the archived three-code protocol explicit on the PySCF side;
        # zero frozen orbitals matches vibe-qc n_frozen_core=0 and ORCA
        # NoFrozenCore rather than relying on a library default.
        mp2 = (
            mol_mp.UMP2(mf, frozen=0)
            if method["scf"] == "uhf"
            else mol_mp.MP2(mf, frozen=0)
        )
        mp2.verbose = 0
        mp2.kernel()
        post_wall = time.perf_counter() - t0
        result_energy = float(mp2.e_tot)

    emit({
        "status": "ok",
        "code_version": pyscf.__version__,
        "energy_ha": result_energy,
        "n_basis_functions": int(mol.nao_nr()),
        "n_basis_shells": int(mol.nbas),
        "n_electrons": int(mol.nelectron),
        "wall_s": scf_wall + post_wall,
        "converged": bool(getattr(mf, "converged", False)),
        "n_iter": int(getattr(mf, "cycles", 0)) or None,
    })


def main():
    payload = json.loads(sys.stdin.read())
    try:
        if payload["kind"] == "periodic":
            run_periodic(payload)
        elif payload["kind"] == "molecule":
            run_molecule(payload)
        else:
            raise ValueError(f"unknown payload kind {payload['kind']!r}")
    except ModuleNotFoundError as exc:
        if exc.name == "pyscf":
            emit({
                "status": "unavailable",
                "code_version": "unknown",
                "note": f"pyscf not importable in external process: {exc}",
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


def _pyscf_python() -> str:
    return os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)


def _method_payload(method: MethodSpec) -> dict[str, Any]:
    return {
        "id": method.id,
        "scf": method.scf,
        "xc": method.xc,
        "post": method.post,
        "df": bool(method.df),
        "aux_basis": method.aux_basis,
    }


def _append_log(log_path: Path, text: str) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
        fh.flush()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_external_result(stdout: str) -> Optional[dict[str, Any]]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    return None


def _run_external_pyscf(
    payload: dict[str, Any],
    log_path: Path,
    artifact_dir: Optional[Path] = None,
) -> dict[str, Any]:
    python = _pyscf_python()
    _append_log(log_path, f"  external PySCF command: {python} -c <script>")
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json(artifact_dir / "input.json", payload)
        (artifact_dir / "runner.py").write_text(
            _PYSCF_EXTERNAL_SCRIPT,
            encoding="utf-8",
        )
    proc = subprocess.run(
        [python, "-c", _PYSCF_EXTERNAL_SCRIPT],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
    )
    if artifact_dir is not None:
        (artifact_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (artifact_dir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    if proc.stdout:
        _append_log(log_path, proc.stdout)
    if proc.stderr:
        _append_log(log_path, "  external PySCF stderr:")
        _append_log(log_path, proc.stderr)

    result = _parse_external_result(proc.stdout)
    if result is not None:
        result.setdefault("returncode", proc.returncode)
        if artifact_dir is not None:
            _write_json(artifact_dir / "parsed.json", result)
        return result

    note = (
        "external PySCF process emitted no result marker "
        f"(returncode={proc.returncode})"
    )
    if "No module named 'pyscf'" in proc.stderr:
        result = {
            "status": "unavailable",
            "code_version": "unknown",
            "note": "pyscf not importable in external process",
            "returncode": proc.returncode,
        }
        if artifact_dir is not None:
            _write_json(artifact_dir / "parsed.json", result)
        return result
    result = {
        "status": "error",
        "code_version": "unknown",
        "note": note,
        "returncode": proc.returncode,
    }
    if artifact_dir is not None:
        _write_json(artifact_dir / "parsed.json", result)
    return result


def _apply_result(row: CodeRow, result: dict[str, Any], n_atoms: int) -> CodeRow:
    row.n_atoms = int(n_atoms) if n_atoms > 0 else None
    row.code_version = str(result.get("code_version") or "unknown")
    status = str(result.get("status") or "error")
    if status == "unavailable":
        row.status = "unavailable"
        row.note = str(result.get("note") or "external PySCF unavailable")
        return row
    if status != "ok":
        row.status = "error"
        row.note = str(result.get("note") or "external PySCF failed")
        return row

    row.energy_ha = float(result["energy_ha"])
    row.wall_s = float(result.get("wall_s") or 0.0)
    row.converged = bool(result.get("converged", False))
    row.n_iter = result.get("n_iter")
    if result.get("n_electrons") is not None:
        row.n_electrons = int(result["n_electrons"])
    if result.get("n_basis_functions") is not None:
        row.n_basis_functions = int(result["n_basis_functions"])
    if result.get("n_basis_shells") is not None:
        row.n_basis_shells = int(result["n_basis_shells"])
    if n_atoms > 0:
        row.energy_per_atom_ha = row.energy_ha / n_atoms
    if not row.converged:
        row.note = "pyscf: SCF did not converge within max_cycle"
    return row


def run_periodic_case(
    *, run_id: str, target: str, spec: PeriodicSpec, basis_name: str,
    method: MethodSpec, kmesh: Tuple[int, int, int],
    conv_tol_energy: Optional[float] = None,
    max_iter: Optional[int] = None,
    log_path: Path,
    artifact_dir: Optional[Path] = None,
) -> CodeRow:
    """Run the external PySCF.pbc reference and return its CodeRow."""
    if conv_tol_energy is None:
        conv_tol_energy = spec.default_conv_tol_energy
    if max_iter is None:
        max_iter = spec.default_max_iter

    row = CodeRow(
        run_id=run_id, target=target, system_id=spec.id, family=spec.family,
        basis=basis_name, method_id=method.id,
        kmesh="x".join(str(k) for k in kmesh),
        code="pyscf", code_version="unknown",
        n_atoms=len(spec.atoms),
    )

    _append_log(log_path, "\n" + "=" * 78)
    _append_log(
        log_path,
        (
            f"  pyscf external | {spec.id} | {basis_name} | {method.id} | "
            f"kmesh={kmesh} | target={target}"
        ),
    )
    _append_log(log_path, "=" * 78)

    payload = {
        "kind": "periodic",
        "lattice_ang": spec.lattice_ang,
        "atoms": [
            {"symbol": at.symbol, "frac": at.frac}
            for at in spec.atoms
        ],
        "basis": basis_name,
        "method": _method_payload(method),
        "kmesh": kmesh,
        "conv_tol_energy": conv_tol_energy,
        "max_iter": max_iter,
        "verbose": 3,
    }
    result = _run_external_pyscf(payload, log_path, artifact_dir=artifact_dir)
    row = _apply_result(row, result, len(spec.atoms))
    if row.energy_ha is not None:
        _append_log(
            log_path,
            (
                f"  E/cell = {row.energy_ha:.10f} Ha   "
                f"({row.n_iter} iters, wall {row.wall_s:.1f} s, "
                f"converged={row.converged})"
            ),
        )
    else:
        _append_log(log_path, f"  {row.status}: {row.note}")
    return row


@dataclass(frozen=True)
class PeriodicMP2Ref:
    """One out-of-process PySCF.pbc periodic MP2 (KMP2) reference point.

    Per-unit-cell energies (PySCF normalizes the k-sampled energy per cell), so
    ``e_corr`` is directly comparable to the megacell route's per-cell value.
    """

    status: str
    code_version: str
    kmesh: Tuple[int, int, int]
    dimension: Optional[int]
    e_hf: Optional[float]
    e_corr: Optional[float]
    e_tot: Optional[float]
    converged: bool
    note: str = ""


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def run_periodic_mp2(
    *,
    lattice_ang: Any,
    atoms_frac: Any,
    basis: str,
    kmesh: Tuple[int, int, int],
    scf: str = "rhf",
    dimension: Optional[int] = None,
    low_dim_ft_type: Optional[str] = None,
    conv_tol_energy: float = 1e-10,
    max_iter: int = 100,
    log_path: Optional[Path] = None,
) -> PeriodicMP2Ref:
    """Out-of-process PySCF.pbc periodic MP2 (KMP2) reference (§10).

    Runs KRHF + KMP2 (or RHF + MP2 at Gamma) on the requested periodic cell in
    a separate interpreter — PySCF is never imported in-process — and returns
    the per-cell HF / correlation / total energies. ``dimension`` /
    ``low_dim_ft_type`` enable low-dimensional PBC (``dimension=1`` for an
    isolated chain: Coulomb truncated transverse, no spurious inter-chain
    interaction, the apples-to-apples partner of the open-boundary megacell).

    This is the §10-clean oracle for the megacell periodic-correlation
    thermodynamic-limit cross-check: the megacell route (megacell -> inf) and
    PySCF KMP2 (k-mesh -> inf) are two independent routes to the same per-cell
    correlation TDL.

    Parameters
    ----------
    lattice_ang : (3, 3) array-like
        Lattice row-vectors in Angstrom.
    atoms_frac : sequence of (symbol, (f1, f2, f3))
        Atoms as element symbol + fractional coordinates.
    basis, kmesh, scf
        Orbital basis, Monkhorst-Pack mesh, and SCF kind (``"rhf"``).
    """
    method = MethodSpec(id="mp2", scf=scf, post="mp2")
    payload: dict[str, Any] = {
        "kind": "periodic",
        "lattice_ang": [[float(x) for x in row] for row in lattice_ang],
        "atoms": [
            {"symbol": str(sym), "frac": [float(x) for x in frac]}
            for sym, frac in atoms_frac
        ],
        "basis": basis,
        "method": _method_payload(method),
        "kmesh": tuple(int(k) for k in kmesh),
        "conv_tol_energy": float(conv_tol_energy),
        "max_iter": int(max_iter),
        "verbose": 0,
    }
    if dimension is not None:
        payload["dimension"] = int(dimension)
    if low_dim_ft_type is not None:
        payload["low_dim_ft_type"] = low_dim_ft_type

    if log_path is None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_external_pyscf(payload, Path(tmp) / "pyscf_periodic_mp2.log")
    else:
        result = _run_external_pyscf(payload, log_path)

    km = tuple(int(k) for k in kmesh)
    status = str(result.get("status") or "error")
    if status != "ok":
        return PeriodicMP2Ref(
            status=status,
            code_version=str(result.get("code_version") or "unknown"),
            kmesh=km, dimension=dimension,
            e_hf=None, e_corr=None, e_tot=None,
            converged=False, note=str(result.get("note") or ""),
        )
    return PeriodicMP2Ref(
        status="ok",
        code_version=str(result.get("code_version") or "unknown"),
        kmesh=km, dimension=dimension,
        e_hf=_opt_float(result.get("e_hf_ha")),
        e_corr=_opt_float(result.get("e_corr_ha")),
        e_tot=_opt_float(result.get("energy_ha")),
        converged=bool(result.get("converged", False)),
    )


def run_molecule_case(
    *, run_id: str, target: str, spec: MoleculeSpec, basis_name: str,
    method: MethodSpec,
    conv_tol_energy: Optional[float] = None,
    max_iter: Optional[int] = None,
    log_path: Path,
    artifact_dir: Optional[Path] = None,
) -> CodeRow:
    """Run the external PySCF molecular reference and return its CodeRow."""
    if conv_tol_energy is None:
        conv_tol_energy = spec.default_conv_tol_energy
    if max_iter is None:
        max_iter = spec.default_max_iter

    row = CodeRow(
        run_id=run_id, target=target, system_id=spec.id, family=spec.family,
        basis=basis_name, method_id=method.id, kmesh="mol",
        code="pyscf", code_version="unknown",
        n_atoms=len(spec.atoms),
    )

    _append_log(log_path, "\n" + "=" * 78)
    _append_log(
        log_path,
        (
            f"  pyscf external | {spec.id} | {basis_name} | {method.id} | "
            f"mol | target={target}"
        ),
    )
    _append_log(log_path, "=" * 78)

    payload = {
        "kind": "molecule",
        "atoms": [
            {"symbol": at.symbol, "xyz_ang": at.xyz_ang}
            for at in spec.atoms
        ],
        "charge": spec.charge,
        "multiplicity": spec.multiplicity,
        "basis": basis_name,
        "method": _method_payload(method),
        "conv_tol_energy": conv_tol_energy,
        "max_iter": max_iter,
        "verbose": 0,
    }
    result = _run_external_pyscf(payload, log_path, artifact_dir=artifact_dir)
    row = _apply_result(row, result, len(spec.atoms))
    if row.energy_ha is not None:
        _append_log(
            log_path,
            (
                f"  E = {row.energy_ha:.10f} Ha   "
                f"({row.n_iter} SCF iters + post={method.post}, "
                f"total wall {row.wall_s:.2f} s, converged={row.converged}, "
                f"df={method.df})"
            ),
        )
    else:
        _append_log(log_path, f"  {row.status}: {row.note}")
    return row
