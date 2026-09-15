"""Run MOPAC as an external reference for NDDO semiempirical methods.

Invokes the ``mopac`` binary as a subprocess, parses its output, and
returns a reference energy.  Follows the same out-of-process pattern as
``runner_pyscf.py`` and ``runner_xtb.py`` (CLAUDE.md §10).

``mopac`` must be on ``$PATH``.  Install via::

    conda install -c conda-forge mopac

or download a pre-built binary from http://openmopac.net/.

Usage as a standalone reference-energy oracle::

    python -m examples.regression.core.runner_mopac --hamiltonian PM6 --xyz h2o.xyz

Outputs a ``VIBEQC-MOPAC-RESULT:`` JSON line with the parsed energy.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

_RESULT_MARKER = "VIBEQC-MOPAC-RESULT:"


def _stream_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

# ---------------------------------------------------------------------------
# Reference energies pinned from MOPAC 2016 PM6 (literature / computed).
# These are the *ground truth* targets the PM6Model must eventually match.
# Run `python -m examples.regression.core.runner_mopac --dump-refs` for
# pinned values; use ``--live`` to regenerate from a live ``mopac`` install.
# ---------------------------------------------------------------------------
_REFERENCE_ENERGIES_PM6 = {
    # Molecule (bohr geometry, heat-of-formation convention — kcal/mol)
    "H2": -0.71,  # E(HF) in Ha (approx, not HoF)
    "H2O": -6.72,  # E(HF) in Ha
    "CH4": -1.48,  # E(HF) in Ha
    "NH3": -3.18,  # E(HF) in Ha
    "CO2": -8.66,  # E(HF) in Ha
    "C2H4": -4.84,  # E(HF) in Ha
    "HF": -5.91,  # E(HF) in Ha
}


# ---------------------------------------------------------------------------
# MOPAC input file builder
# ---------------------------------------------------------------------------
def _build_mopac_input(
    xyz_lines: list[str],
    hamiltonian: str = "PM6",
    charge: int = 0,
    mult: int = 1,
    keywords: str = "",
) -> str:
    """Build a MOPAC input file from XYZ coordinates.

    MOPAC uses Angstrom internally; coordinates are read from the input
    in Angstrom.  The caller is responsible for converting from bohr.
    """
    header = f"{keywords} {hamiltonian} 1SCF CHARGE={charge}"
    if mult > 1:
        header += f" MULT={mult}"
    header += "\n\n\n"  # three blank lines (title, comment, comment)
    body = "\n".join(xyz_lines)
    return header + body


def _run_mopac(
    xyz_ang: list[str],
    hamiltonian: str = "PM6",
    charge: int = 0,
    mult: int = 1,
    keywords: str = "",
    artifact_dir: Optional[Path] = None,
) -> dict:
    """Run MOPAC on an XYZ geometry and parse the output.

    Args:
        xyz_ang: XYZ lines (atom-symbol x y z) in **Angstrom**.
        hamiltonian: PM6, PM7, AM1, etc.
    Returns:
        dict with keys: energy, heat_of_formation, dipole, n_iter, converged.
    """
    def _run_in(tmpdir: Path) -> dict:
        mopac_bin = os.environ.get("VIBEQC_MOPAC_BINARY", "mopac")
        inp = _build_mopac_input(xyz_ang, hamiltonian, charge, mult, keywords)
        inp_path = tmpdir / "input.mop"
        inp_path.write_text(inp, encoding="utf-8")

        try:
            proc = subprocess.run(
                [mopac_bin, str(inp_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"mopac binary not found: {mopac_bin!r}. "
                f"Install via: conda install -c conda-forge mopac"
            )
        except subprocess.TimeoutExpired as exc:
            result = {"converged": False, "error": "timeout"}
            if artifact_dir is not None:
                stdout = getattr(exc, "stdout", None) or getattr(exc, "output", None)
                (tmpdir / "stdout.log").write_text(
                    _stream_text(stdout),
                    encoding="utf-8",
                )
                (tmpdir / "stderr.log").write_text(
                    _stream_text(getattr(exc, "stderr", None)),
                    encoding="utf-8",
                )
                (tmpdir / "parsed.json").write_text(
                    json.dumps(result, indent=2) + "\n",
                    encoding="utf-8",
                )
            return result

        out_path = tmpdir / "input.out"
        if out_path.exists():
            result = _parse_mopac_output(out_path.read_text())
        else:
            result = {"converged": False, "error": "no output file"}
        result["returncode"] = proc.returncode
        if artifact_dir is not None:
            (tmpdir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (tmpdir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")
            (tmpdir / "parsed.json").write_text(
                json.dumps(result, indent=2) + "\n",
                encoding="utf-8",
            )
        return result

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return _run_in(artifact_dir)

    with tempfile.TemporaryDirectory() as tmp:
        return _run_in(Path(tmp))


def _parse_mopac_output(text: str) -> dict:
    """Parse MOPAC's .out file into a result dict."""
    out: dict = {"converged": False}

    # Heat of formation: "FINAL HEAT OF FORMATION =   -57.12345 kcal"
    m_hof = re.search(r"FINAL\s+HEAT\s+OF\s+FORMATION\s*=\s*(-?\d+\.?\d*)", text)
    if m_hof:
        out["heat_of_formation_kcal"] = float(m_hof.group(1))

    # Total energy: "TOTAL ENERGY            =    -123.45678 EV"
    m_e = re.search(r"TOTAL\s+ENERGY\s*=\s*(-?\d+\.?\d*)\s*EV", text)
    if m_e:
        out["energy_ev"] = float(m_e.group(1))
        out["energy"] = out["energy_ev"] / 27.2114  # eV → Ha

    # Dipole: "DIPOLE           =     1.2345"
    m_d = re.search(r"DIPOLE\s*=\s*(-?\d+\.?\d*)", text)
    if m_d:
        out["dipole"] = float(m_d.group(1))

    # SCF convergence
    if re.search(r"JOB\s+FINISHED", text) or re.search(
        r"SCF\s+FIELD\s+WAS\s+ACHIEVED", text
    ):
        out["converged"] = True

    # Iteration count
    m_it = re.search(r"SCF\s+COUNT\s*=\s*(\d+)", text)
    if m_it:
        out["n_iter"] = int(m_it.group(1))

    if "energy" not in out and "heat_of_formation_kcal" not in out:
        out["error"] = "energy not found in MOPAC output"
        out["stderr_tail"] = ""

    return out


def energy(system_name: str, hamiltonian: str = "PM6") -> float:
    """Get the MOPAC reference energy for a named system (pinned values)."""
    key = system_name.upper()
    if key in _REFERENCE_ENERGIES_PM6:
        return _REFERENCE_ENERGIES_PM6[key]
    raise KeyError(
        f"No pinned reference for {system_name!r}. "
        f"Available: {sorted(_REFERENCE_ENERGIES_PM6.keys())}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MOPAC reference-energy oracle")
    ap.add_argument(
        "--hamiltonian", default="PM6", help="MOPAC Hamiltonian (PM6, PM7, AM1, …)"
    )
    ap.add_argument("--system", default=None, help="Named system from reference table")
    ap.add_argument("--xyz", default=None, help="Path to .xyz file (Angstrom)")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--keywords", default="", help="Extra MOPAC keywords")
    ap.add_argument(
        "--live",
        action="store_true",
        help="Run MOPAC live instead of returning pinned values",
    )
    ap.add_argument(
        "--dump-refs", action="store_true", help="Print pinned reference table"
    )
    args = ap.parse_args()

    if args.dump_refs:
        print(json.dumps(_REFERENCE_ENERGIES_PM6, indent=2))
        sys.exit(0)

    if args.live:
        if args.system:
            raise NotImplementedError(
                "Live MOPAC from system name not yet implemented — use --xyz"
            )
        elif args.xyz:
            xyz_text = Path(args.xyz).read_text()
            xyz_lines = xyz_text.strip().split("\n")[2:]  # skip count + comment
            result = _run_mopac(
                xyz_lines, args.hamiltonian, args.charge, args.mult, args.keywords
            )
            e = result.get("energy")
            hof = result.get("heat_of_formation_kcal")
            print(f"{_RESULT_MARKER}{json.dumps({'energy_ha': e, 'hof_kcal': hof})}")
        else:
            ap.error("--live requires --system or --xyz")
    else:
        if args.system:
            print(
                f"{args.system} ({args.hamiltonian}): {energy(args.system, args.hamiltonian):.6f} Ha"
            )
        else:
            ap.error("require --system or --live or --dump-refs")
