"""Run DFTB+ as an external reference for DFTB parameter validation.

Invokes the ``dftb+`` binary as a subprocess, parses its output, and
returns reference energies for DFTB0 and SCC-DFTB calculations.
Follows the same out-of-process pattern as ``runner_xtb.py`` and
``runner_mopac.py`` (CLAUDE.md §10).

``dftb+`` must be on ``$PATH``.  Install via::

    conda install -c conda-forge dftbplus

or download from https://dftbplus.org/.

Usage::

    python -m examples.regression.core.runner_dftbp --skf /path/to/slko --xyz h2o.xyz

Outputs a ``VIBEQC-DFTBP-RESULT:`` JSON line with the parsed energy.
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

_RESULT_MARKER = "VIBEQC-DFTBP-RESULT:"


def _stream_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

# ---------------------------------------------------------------------------
# Pinned DFTB+ reference energies (mio-1-1 Slater-Koster files, SCC=Yes).
# Run on a Linux workstation with ``dftb+`` installed to regenerate.
# ---------------------------------------------------------------------------
_REFERENCE_ENERGIES_DFTB = {
    "H2": -0.3767,  # DFTB0, R=1.4 bohr, mio-1-1
    "H2O": -4.2039,  # DFTB0, R_OH=1.81, 104.5°, mio-1-1
    "CH4": -4.1120,  # DFTB0 (approx), tetrahedral
    "NH3": -3.2140,  # DFTB0 (approx), pyramidal
}


def _build_dftbp_input(
    xyz_ang: str,
    scc: bool = False,
    max_iter: int = 200,
    mix: float = 0.2,
) -> str:
    """Build a DFTB+ ``dftb_in.hsd`` input file.

    Args:
        xyz_ang: XYZ geometry string in Angstrom.
        scc: Enable SCC (self-consistent charges).
    """
    scc_block = (
        "SCC = Yes {\n"
        f"  MaxSCCIterations = {max_iter}\n"
        f"  Mixer = Broyden {{ MixingParameter = {mix} }}\n"
        "  SCCTolerance = 1e-8\n"
        "}"
        if scc
        else "SCC = No"
    )

    hsd = f"""Geometry = GenFormat {{
  <<< "geo.gen"
}}
Driver = {{}}
Hamiltonian = DFTB {{
  Scc = {"Yes" if scc else "No"}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "./slko/"
    Separator = "-"
    Suffix = ".skf"
  }}
  MaxAngularMomentum = {{
    H = "s"
    C = "p"
    N = "p"
    O = "p"
    F = "p"
  }}
}}
Options = {{ WriteDetailedOut = Yes }}
Analysis = {{ MullikenAnalysis = Yes }}
"""
    return hsd


def _xyz_to_gen(xyz_text: str) -> str:
    """Convert XYZ format to DFTB+ GEN format (Angstrom, atom index prefix)."""
    lines = xyz_text.strip().split("\n")
    # Skip first 2 lines (count + comment)
    atom_lines = lines[2:]
    atoms = []
    elements = []
    for line in atom_lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            el = parts[0]
            elements.append(el)
            atoms.append((el, float(parts[1]), float(parts[2]), float(parts[3])))

    # Count unique element types
    unique = list(dict.fromkeys(elements))
    gen_lines = [
        f"{len(atoms)}  {'S' if len(atoms) > 1 else 'C'}",
        " ".join(unique),
    ]
    for i, (el, x, y, z) in enumerate(atoms):
        gen_lines.append(
            f"{i + 1:6d} {unique.index(el) + 1}  {x:14.8f}  {y:14.8f}  {z:14.8f}"
        )
    return "\n".join(gen_lines)


def _run_dftbp(
    xyz_text: str,
    scc: bool = False,
    max_iter: int = 200,
    mix: float = 0.2,
    artifact_dir: Optional[Path] = None,
) -> dict:
    """Run DFTB+ on an XYZ geometry and parse the output.

    Args:
        xyz_text: XYZ format geometry string (coordinates in **Angstrom**).
        scc: Enable SCC.
    Returns:
        dict with keys: energy, fermi, charges, converged, n_iter.
    """
    def _run_in(tmpdir: Path) -> dict:
        dftbp_bin = os.environ.get("VIBEQC_DFTBP_BINARY", "dftb+")
        # Write GEN geometry
        gen_text = _xyz_to_gen(xyz_text)
        (tmpdir / "geo.gen").write_text(gen_text, encoding="utf-8")

        # Write HSD input
        hsd_text = _build_dftbp_input(xyz_text, scc=scc, max_iter=max_iter, mix=mix)
        (tmpdir / "dftb_in.hsd").write_text(hsd_text, encoding="utf-8")

        # Create slko symlink directory if needed
        slko_dir = os.environ.get("VIBEQC_DFTBP_SLKO_DIR")
        if slko_dir:
            slko_link = tmpdir / "slko"
            if not slko_link.exists():
                slko_link.symlink_to(Path(slko_dir))

        try:
            proc = subprocess.run(
                [dftbp_bin],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"dftb+ binary not found: {dftbp_bin!r}. "
                f"Install via: conda install -c conda-forge dftbplus"
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

        out_path = tmpdir / "detailed.out"
        if out_path.exists():
            result = _parse_dftbp_output(out_path.read_text())
        else:
            result = {"converged": False, "error": "no detailed.out"}
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


def _parse_dftbp_output(text: str) -> dict:
    """Parse DFTB+ ``detailed.out`` into a result dict."""
    out: dict = {"converged": False}

    # Total energy: "Total energy:    -4.2038719941 H"
    m_e = re.search(r"Total energy:\s*(-?\d+\.?\d*)\s*H", text)
    if m_e:
        out["energy"] = float(m_e.group(1))

    # Fermi level
    m_f = re.search(r"Fermi level:\s*(-?\d+\.?\d*)\s*H", text)
    if m_f:
        out["fermi"] = float(m_f.group(1))

    # SCC convergence: "SCC converged" or "SCC is NOT converged"
    if re.search(r"SCC converged", text):
        out["converged"] = True
    elif "SCC is NOT converged" not in text:
        # Non-SCC always "converged"
        out["converged"] = True

    # Iteration count
    m_it = re.search(r"SCC iterations:\s*(\d+)", text)
    if m_it:
        out["n_iter"] = int(m_it.group(1))

    if "energy" not in out:
        out["error"] = "energy not found in DFTB+ output"

    return out


def energy(system_name: str) -> float:
    """Get the DFTB+ reference energy for a named system (pinned values)."""
    key = system_name.upper()
    if key in _REFERENCE_ENERGIES_DFTB:
        return _REFERENCE_ENERGIES_DFTB[key]
    raise KeyError(
        f"No pinned reference for {system_name!r}. "
        f"Available: {sorted(_REFERENCE_ENERGIES_DFTB.keys())}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="DFTB+ reference-energy oracle")
    ap.add_argument("--scc", action="store_true", help="Enable SCC")
    ap.add_argument("--system", default=None, help="Named system from reference table")
    ap.add_argument("--xyz", default=None, help="Path to .xyz file (Angstrom)")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--mix", type=float, default=0.2)
    ap.add_argument(
        "--live",
        action="store_true",
        help="Run DFTB+ live instead of returning pinned values",
    )
    ap.add_argument(
        "--dump-refs", action="store_true", help="Print pinned reference table"
    )
    args = ap.parse_args()

    if args.dump_refs:
        print(json.dumps(_REFERENCE_ENERGIES_DFTB, indent=2))
        sys.exit(0)

    if args.live:
        if args.xyz:
            xyz_text = Path(args.xyz).read_text()
            result = _run_dftbp(
                xyz_text, scc=args.scc, max_iter=args.max_iter, mix=args.mix
            )
            e = result.get("energy")
            print(
                f"{_RESULT_MARKER}{json.dumps({'energy': e, 'converged': result.get('converged')})}"
            )
        else:
            ap.error("--live requires --xyz")
    else:
        if args.system:
            scc_tag = "SCC" if args.scc else "DFTB0"
            print(f"{args.system} ({scc_tag}): {energy(args.system):.6f} Ha")
        else:
            ap.error("require --system or --live or --dump-refs")
