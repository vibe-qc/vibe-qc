"""External-program comparison runner — out-of-process parity testing.

Runs ORCA and CRYSTAL geometry optimizations via subprocess at
identical method/basis/fmax and parses the output to extract step
counts, timings, and final energies.  Complies with vibe-qc's §2
rule (no runtime imports from other QC programs — all interaction
is through subprocess).

The external programs must be installed on the host and available
on PATH.  This module only *runs* them — it does not bundle,
vendor, or import them.

Usage::

    from examples.regression.optimizer_benchmark.external import (
        run_orca_optimization, run_crystal_optimization,
    )

    result = run_orca_optimization(
        xyz_path="h2o.xyz",
        method="RKS", functional="PBE", basis="def2-SVP",
        fmax_eva=0.01, max_steps=200,
    )
    print(f"ORCA: {result['n_steps']} steps, E={result['energy_ha']} Ha")
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExternalResult:
    """Result of an external geometry optimization."""

    program: str
    system_name: str
    method: str
    functional: Optional[str]
    basis: str
    converged: bool
    n_steps: int
    wall_time_s: float
    energy_initial_ha: float
    energy_final_ha: float
    grad_final_eva: float
    output_path: str
    error: Optional[str] = None
    raw_energies: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# XYZ file writer (for external programs)
# ---------------------------------------------------------------------------


def _write_xyz(path: Path, system: Any, comment: str = "") -> None:
    """Write an XYZ file from a vibe-qc Molecule or TestSystem.

    Converts bohr → Ångström for external consumption.
    """
    from ase.units import Bohr

    if hasattr(system, "molecule"):
        mol = system.molecule
    else:
        mol = system

    atoms = list(mol.atoms)
    lines = [str(len(atoms)), comment]
    for atom in atoms:
        x = atom.xyz[0] * Bohr
        y = atom.xyz[1] * Bohr
        z = atom.xyz[2] * Bohr
        symbol = _ATOMIC_SYMBOLS.get(atom.Z, "X")
        lines.append(f"{symbol:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    path.write_text("\n".join(lines) + "\n")


_ATOMIC_SYMBOLS: dict[int, str] = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
}


# ---------------------------------------------------------------------------
# ORCA input generation + output parsing
# ---------------------------------------------------------------------------


def _orca_input(
    method: str,
    functional: Optional[str],
    basis: str,
    charge: int,
    multiplicity: int,
    fmax_eva: float,
    max_steps: int,
) -> str:
    """Generate an ORCA input file string."""
    method_upper = method.upper()
    if method_upper == "RKS":
        method_line = f"! {functional or 'PBE'} {basis} OPT"
    elif method_upper == "RHF":
        method_line = f"! HF {basis} OPT"
    elif method_upper == "UHF":
        method_line = f"! UHF {basis} OPT"
    else:
        raise ValueError(f"Unsupported ORCA method: {method}")

    return f"""{method_line}
%geom
  MaxIter {max_steps}
  TolMaxForce {fmax_eva:.6f}
  convergence tight
end
%scf
  Convergence Tight
end
* xyz {charge} {multiplicity}
"""


def _parse_orca_output(output: str) -> dict:
    """Parse ORCA output for optimization results."""
    result: dict = {
        "converged": False,
        "n_steps": 0,
        "energy_final_ha": float("nan"),
        "energy_initial_ha": float("nan"),
        "grad_final_eva": float("nan"),
        "raw_energies": [],
        "wall_time_s": 0.0,
    }

    # Check convergence
    if re.search(r"THE OPTIMIZATION HAS CONVERGED", output):
        result["converged"] = True

    # Extract energies (Ha)
    energies = re.findall(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)", output)
    if len(energies) >= 2:
        result["energy_initial_ha"] = float(energies[1])
        result["energy_final_ha"] = float(energies[-1])
    result["raw_energies"] = [float(e) for e in energies]

    # Count geometry steps
    steps = re.findall(r"Geometry optimization step\s+(\d+)", output)
    if steps:
        result["n_steps"] = int(steps[-1])

    # Wall time
    time_match = re.search(
        r"TOTAL RUN TIME:\s+(\d+)\s+days\s+(\d+)\s+hours\s+(\d+)", output
    )
    if time_match:
        d, h, m = map(int, time_match.groups())
        result["wall_time_s"] = d * 86400 + h * 3600 + m * 60
    else:
        time_match = re.search(r"TOTAL RUN TIME:\s+(\d+)\s+hours\s+(\d+)", output)
        if time_match:
            h, m = map(int, time_match.groups())
            result["wall_time_s"] = h * 3600 + m * 60

    return result


# ---------------------------------------------------------------------------
# CRYSTAL input generation + output parsing
# ---------------------------------------------------------------------------


def _crystal_input(
    method: str,
    functional: Optional[str],
    basis: str,
    charge: int,
    multiplicity: int,
    fmax_eva: float,
    max_steps: int,
) -> str:
    """Generate a CRYSTAL17 input file (molecular run)."""
    if method.upper() == "RHF":
        dft_line = "HF"
    elif method.upper() == "RKS":
        dft_line = functional or "PBE"
    else:
        raise ValueError(f"Unsupported CRYSTAL method: {method}")

    return f"""{dft_line}
OPTGEOM
FINALRUN 0
MAXCYCLE {max_steps}
TOLDEG 0.0003
END
{basis}
END
{charge} {multiplicity}
"""


def _parse_crystal_output(output: str) -> dict:
    """Parse CRYSTAL output for optimization results."""
    result: dict = {
        "converged": False,
        "n_steps": 0,
        "energy_final_ha": float("nan"),
        "energy_initial_ha": float("nan"),
        "grad_final_eva": float("nan"),
        "raw_energies": [],
        "wall_time_s": 0.0,
    }

    # Check convergence
    if re.search(r"OPT END - CONVERGED", output):
        result["converged"] = True

    # Extract energies
    energies = re.findall(r"TOTAL ENERGY\s+\S+\s+\S+\s+=\s+([-\d.]+)", output)
    if energies:
        result["raw_energies"] = [float(e) for e in energies]
        result["energy_initial_ha"] = float(energies[0])
        result["energy_final_ha"] = float(energies[-1])

    # Count steps
    steps = re.findall(r"CYCLE\s+(\d+)", output)
    if steps:
        result["n_steps"] = int(steps[-1])

    # Wall time
    time_match = re.search(r"CP_PROC_TIME\s+([\d.]+)", output, re.IGNORECASE)
    if time_match:
        result["wall_time_s"] = float(time_match.group(1))

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_orca_optimization(
    xyz_path: str | Path,
    *,
    method: str = "RKS",
    functional: Optional[str] = "PBE",
    basis: str = "def2-SVP",
    charge: int = 0,
    multiplicity: int = 1,
    fmax_eva: float = 0.01,
    max_steps: int = 200,
    orca_exe: str = "orca",
    work_dir: Optional[str | Path] = None,
    timeout_s: float = 3600.0,
) -> ExternalResult:
    """Run an ORCA geometry optimization and parse results.

    Parameters
    ----------
    xyz_path : Path
        Path to an XYZ file with the starting geometry.
    method : str
        SCF method: ``"RHF"``, ``"UHF"``, ``"RKS"``.
    functional : str or None
        XC functional for KS methods.
    basis : str
        ORCA-recognized basis set name.
    charge, multiplicity : int
        Molecular charge and spin multiplicity.
    fmax_eva : float
        Maximum force convergence (eV/Å).
    max_steps : int
        Maximum geometry optimization steps.
    orca_exe : str
        Path or name of the ORCA executable.
    work_dir : Path or None
        Working directory. Uses a temp dir if None.
    timeout_s : float
        Maximum wall-clock time (seconds).

    Returns
    -------
    ExternalResult
    """
    xyz_path = Path(xyz_path)
    stem = xyz_path.stem

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="vq_orca_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    inp_path = work_dir / f"{stem}.inp"
    out_path = work_dir / f"{stem}.out"

    # Read XYZ and write ORCA input
    xyz_content = xyz_path.read_text()
    inp = (
        _orca_input(
            method, functional, basis, charge, multiplicity, fmax_eva, max_steps
        )
        + xyz_content
        + "\n*\n"
    )
    inp_path.write_text(inp)

    try:
        subprocess.run(
            [orca_exe, str(inp_path)],
            cwd=str(work_dir),
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return ExternalResult(
            program="orca",
            system_name=stem,
            method=method,
            functional=functional,
            basis=basis,
            converged=False,
            n_steps=0,
            wall_time_s=timeout_s,
            energy_initial_ha=float("nan"),
            energy_final_ha=float("nan"),
            grad_final_eva=float("inf"),
            output_path=str(out_path),
            error="Timeout",
        )
    except FileNotFoundError:
        return ExternalResult(
            program="orca",
            system_name=stem,
            method=method,
            functional=functional,
            basis=basis,
            converged=False,
            n_steps=0,
            wall_time_s=0.0,
            energy_initial_ha=float("nan"),
            energy_final_ha=float("nan"),
            grad_final_eva=float("inf"),
            output_path=str(out_path),
            error=f"Executable not found: {orca_exe}",
        )

    if not out_path.exists():
        return ExternalResult(
            program="orca",
            system_name=stem,
            method=method,
            functional=functional,
            basis=basis,
            converged=False,
            n_steps=0,
            wall_time_s=0.0,
            energy_initial_ha=float("nan"),
            energy_final_ha=float("nan"),
            grad_final_eva=float("inf"),
            output_path=str(out_path),
            error="Output file not produced",
        )

    output = out_path.read_text()
    parsed = _parse_orca_output(output)

    return ExternalResult(
        program="orca",
        system_name=stem,
        method=method,
        functional=functional,
        basis=basis,
        converged=parsed["converged"],
        n_steps=parsed["n_steps"],
        wall_time_s=parsed["wall_time_s"],
        energy_initial_ha=parsed["energy_initial_ha"],
        energy_final_ha=parsed["energy_final_ha"],
        grad_final_eva=parsed["grad_final_eva"],
        output_path=str(out_path),
        raw_energies=parsed["raw_energies"],
    )


def run_crystal_optimization(
    xyz_path: str | Path,
    *,
    method: str = "RHF",
    functional: Optional[str] = None,
    basis: str = "6-31G**",
    charge: int = 0,
    multiplicity: int = 1,
    fmax_eva: float = 0.01,
    max_steps: int = 200,
    crystal_exe: str = "crystal",
    work_dir: Optional[str | Path] = None,
    timeout_s: float = 3600.0,
) -> ExternalResult:
    """Run a CRYSTAL geometry optimization and parse results.

    Same interface as :func:`run_orca_optimization`.
    """
    xyz_path = Path(xyz_path)
    stem = xyz_path.stem

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="vq_crystal_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    inp_path = work_dir / f"{stem}.d12"
    out_path = work_dir / f"{stem}.out"

    inp = _crystal_input(
        method, functional, basis, charge, multiplicity, fmax_eva, max_steps
    )
    # CRYSTAL reads geometry from a separate file or inline.  For
    # molecular runs we use EXTERNAL coordinates referencing the XYZ.
    # Simplified: embed coordinates directly.
    xyz_content = xyz_path.read_text()
    lines = xyz_content.strip().split("\n")
    # Write geometry block after the input
    inp += "\n".join(lines[2:]) + "\n"
    inp_path.write_text(inp)

    try:
        subprocess.run(
            [crystal_exe],
            cwd=str(work_dir),
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return ExternalResult(
            program="crystal",
            system_name=stem,
            method=method,
            functional=functional,
            basis=basis,
            converged=False,
            n_steps=0,
            wall_time_s=timeout_s,
            energy_initial_ha=float("nan"),
            energy_final_ha=float("nan"),
            grad_final_eva=float("inf"),
            output_path=str(out_path),
            error="Timeout",
        )
    except FileNotFoundError:
        return ExternalResult(
            program="crystal",
            system_name=stem,
            method=method,
            functional=functional,
            basis=basis,
            converged=False,
            n_steps=0,
            wall_time_s=0.0,
            energy_initial_ha=float("nan"),
            energy_final_ha=float("nan"),
            grad_final_eva=float("inf"),
            output_path=str(out_path),
            error=f"Executable not found: {crystal_exe}",
        )

    if not out_path.exists():
        return ExternalResult(
            program="crystal",
            system_name=stem,
            method=method,
            functional=functional,
            basis=basis,
            converged=False,
            n_steps=0,
            wall_time_s=0.0,
            energy_initial_ha=float("nan"),
            energy_final_ha=float("nan"),
            grad_final_eva=float("inf"),
            output_path=str(out_path),
            error="Output file not produced",
        )

    output = out_path.read_text()
    parsed = _parse_crystal_output(output)

    return ExternalResult(
        program="crystal",
        system_name=stem,
        method=method,
        functional=functional,
        basis=basis,
        converged=parsed["converged"],
        n_steps=parsed["n_steps"],
        wall_time_s=parsed["wall_time_s"],
        energy_initial_ha=parsed["energy_initial_ha"],
        energy_final_ha=parsed["energy_final_ha"],
        grad_final_eva=parsed["grad_final_eva"],
        output_path=str(out_path),
        raw_energies=parsed["raw_energies"],
    )
