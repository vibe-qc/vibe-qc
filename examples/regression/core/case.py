"""Case execution records + CSV / env.json io."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CodeRow:
    """One row of the results CSV — one (case, code, target) triple."""

    run_id: str
    target: str  # "dev" | "release"
    system_id: str
    family: str  # "rocksalt" | "molecule_closed_shell" | ...
    basis: str
    method_id: str
    kmesh: str  # "(1,1,1)" or "molecule"
    code: str  # "vibeqc" | "pyscf" | "orca"
    code_version: str
    converged: Optional[bool] = None
    n_iter: Optional[int] = None
    n_retries: int = 0
    retry_strategies: str = ""  # ";"-joined strategy names from the ladder
    energy_ha: Optional[float] = None
    energy_per_atom_ha: Optional[float] = None
    n_atoms: Optional[int] = None
    n_electrons: Optional[int] = None
    n_basis_functions: Optional[int] = None
    n_basis_shells: Optional[int] = None
    wall_s: float = 0.0
    severity: str = ""  # eigs_preflight worst severity (vibeqc only)
    min_eigval_S: Optional[float] = None
    delta_ha_vs_ref: Optional[float] = None
    delta_mha_vs_ref: Optional[float] = None
    abs_delta_mha_vs_ref: Optional[float] = None
    delta_mha_per_atom_vs_ref: Optional[float] = None
    abs_delta_mha_per_atom_vs_ref: Optional[float] = None
    delta_mha_per_electron_vs_ref: Optional[float] = None
    abs_delta_mha_per_electron_vs_ref: Optional[float] = None
    delta_mha_per_basis_function_vs_ref: Optional[float] = None
    abs_delta_mha_per_basis_function_vs_ref: Optional[float] = None
    ref_code: str = ""  # which CodeRow in this case is the reference
    status: str = "pending"  # pass | marginal | fail | unavailable | error
    note: str = ""


CSV_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "target",
    "system_id",
    "family",
    "basis",
    "method_id",
    "kmesh",
    "code",
    "code_version",
    "converged",
    "n_iter",
    "n_retries",
    "retry_strategies",
    "energy_ha",
    "energy_per_atom_ha",
    "n_atoms",
    "n_electrons",
    "n_basis_functions",
    "n_basis_shells",
    "wall_s",
    "severity",
    "min_eigval_S",
    "delta_ha_vs_ref",
    "delta_mha_vs_ref",
    "abs_delta_mha_vs_ref",
    "delta_mha_per_atom_vs_ref",
    "abs_delta_mha_per_atom_vs_ref",
    "delta_mha_per_electron_vs_ref",
    "abs_delta_mha_per_electron_vs_ref",
    "delta_mha_per_basis_function_vs_ref",
    "abs_delta_mha_per_basis_function_vs_ref",
    "ref_code",
    "status",
    "note",
)


def write_csv(path: Path, rows: List[CodeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            d = asdict(r)
            w.writerow([d[c] for c in CSV_COLUMNS])


@dataclass
class CaseRecord:
    """All rows from running one (system, basis, method) case across codes."""

    system_id: str
    family: str
    basis: str
    method_id: str
    kmesh: Tuple[int, int, int]
    rows: List[CodeRow] = field(default_factory=list)
    verbose_log_path: Optional[Path] = None
    # Phase 2 — optional NIST CCCBDB experimental reference attached at
    # run-suite time (`--include-experimental-reference cccbdb`). The
    # report renderer surfaces atomization energy / IE / vibrational
    # fundamentals / dipole / polarizability when present, with full
    # NIST-DOI citation. ``None`` when the run wasn't asked to attach
    # references or when the molecule isn't in the CCCBDB canonical
    # set.
    experimental_reference: Optional["object"] = None
    """When set, an :class:`examples.regression.core.spec.ExperimentalReference`
    pulled from NIST CCCBDB. Stored as ``object`` here to avoid a
    forward-import (the schema is defined in ``.spec`` and CaseRecord
    is consumed by ``.spec`` callers — keep the reverse direction
    type-erased at the boundary)."""

    @property
    def label(self) -> str:
        km = "x".join(str(k) for k in self.kmesh) if self.kmesh != (0, 0, 0) else "mol"
        return f"{self.system_id}__{self.basis}__{self.method_id}__{km}"


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _git_branch(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def collect_env(repo_root: Path) -> Dict[str, Any]:
    """Snapshot environment metadata for the run."""
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "git_sha": _git_sha(repo_root),
        "git_branch": _git_branch(repo_root),
    }
    try:
        import vibeqc  # noqa: WPS433

        env["vibeqc_version"] = getattr(vibeqc, "__version__", "n/a")
        env["vibeqc_path"] = getattr(vibeqc, "__file__", "n/a")
    except ImportError:
        env["vibeqc_version"] = "import_failed"
    pyscf_python = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    try:
        out = subprocess.run(
            [
                pyscf_python,
                "-c",
                (
                    "import importlib; "
                    "m = importlib.import_module('pyscf'); "
                    "print(m.__version__)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        env["pyscf_version"] = out.stdout.strip() or "unknown"
        env["pyscf_command"] = pyscf_python
    except Exception:
        env["pyscf_version"] = "not_installed"
        env["pyscf_command"] = pyscf_python
    try:
        import ase  # noqa: WPS433

        env["ase_version"] = ase.__version__
    except ImportError:
        env["ase_version"] = "not_installed"
    orca = shutil.which("orca")
    env["orca_command"] = orca if orca else "not_on_path"

    # CP2K — same subprocess-only boundary as ORCA.
    cp2k_exe = (
        os.environ.get("VIBEQC_CP2K_EXECUTABLE")
        or shutil.which("cp2k.psmp")
        or shutil.which("cp2k.popt")
        or shutil.which("cp2k.sopt")
        or shutil.which("cp2k")
    )
    env["cp2k_command"] = cp2k_exe if cp2k_exe else "not_on_path"
    return env


def write_env_json(path: Path, env: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(env, fh, indent=2, sort_keys=True)
