"""Run the xTB reference as an external program.

Invokes the ``xtb`` binary (Grimme group, https://github.com/grimme-lab/xtb)
as a subprocess, parses its output, and returns a :class:`CodeRow`.  Follows
the same out-of-process pattern as ``runner_pyscf.py`` (CLAUDE.md §10).

``xtb`` must be on ``$PATH``.  On Linux install via conda-forge::

    conda install -c conda-forge xtb

On macOS, xTB has no pre-built binary — use a Linux CI runner or a
conda environment on a Linux workstation.

Usage as a standalone reference-energy oracle::

    python -m examples.regression.core.runner_xtb --gfn 2 --xyz h2o.xyz

Outputs a single ``VIBEQC-XTB-RESULT:`` JSON line with the parsed energy.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

_RESULT_MARKER = "VIBEQC-XTB-RESULT:"

# ---------------------------------------------------------------------------
# Reference energies pinned from xTB v6.7.1 GFN2-xTB (literature / computed).
# These are the *ground truth* targets the GFN2Model must eventually match.
# Run `python -m examples.regression.core.runner_xtb --dump-refs` to
# regenerate from a live ``xtb`` installation.
#
# IMPORTANT (2026-08-05 fix): the _GEOMETRIES_XYZ values are now in ångström
# (the XYZ standard).  The 2026-06-26 pinned molecule references were generated
# with the pre-fix XYZ that stored bohr numbers in the XYZ strings — xtb then
# interpreted them as ångström, yielding references at ~1.89× the intended
# bond lengths.  The atom references (single atom at origin) are unaffected.
# Molecule references must be regenerated on a Linux host.
# ---------------------------------------------------------------------------
_REFERENCE_ENERGIES_GFN2 = {
    # Atom (ångström)          xTB GFN2 total energy (Eh)
    # Single atoms at origin — geometry is (0,0,0) regardless of units.
    "H": -0.502396,  # ¹H, unrestricted (doublet)
    "He": -0.0,  # noble — no valence
    "C": -1.785981,  # ³P, unrestricted (triplet)
    "N": -2.707235,  # ⁴S, unrestricted (quartet)
    "O": -3.900936,  # ³P, unrestricted (triplet)
    "F": -5.499780,  # ²P, unrestricted (doublet)
    "Ne": -0.0,  # noble — no valence
    # Molecules (ångström geometry) — REGENERATION NEEDED on a Linux host.
    # Current values are STALE: they were generated with xtb reading the
    # old bohr-labeled XYZ as ångström, i.e. at ~1.89× the intended distances.
    # The H2 value is repinned from the ASCENT external-parity task below.
    "H2": -0.982017,  # R=0.740848 Å (=1.4 bohr), live xtb GFN2 2026-08-05
                       # (vibe-qc gap: +14.6 mEh, pre-dispersion electronic model)
    "H2O": -5.070374,  # Live xtb GFN2 2026-08-05 at _water() geometry (OH=0.958 Å)
    "CH4": -4.182964,  # STALE — was C-H=1.155 Å; correct ref for C-H=0.611 Å needed
    "NH3": -3.288569,  # STALE — was N-H≈1.91 Å; correct ref for N-H≈1.01 Å needed
    "CO2": -7.788907,  # STALE — was C-O=2.196 Å; correct ref for C-O=1.162 Å needed
    "C2H4": -5.585929,  # STALE — was C=C=2.5 Å; correct ref for C=C=1.323 Å needed
    # NOTE: the heavy single atoms added to _GEOMETRIES_XYZ below (Ge…Lu) are
    # deliberately absent here — xtb has no macOS binary and is not installed in
    # this dev environment, so their reference energies must be generated on a
    # Linux host: `python -m examples.regression.core.runner_xtb --live
    # --system <El> --dump-refs`.  They are NOT fabricated (CLAUDE.md §8).
}


# ---------------------------------------------------------------------------
# Bohr-to-ångström conversion factor (XYZ standard).
# The test fixtures (test_gfn2_xtb.py) define geometries in bohr; XYZ files
# consumed by xtb must be in ångström.  This factor converts the former to the
# latter when writing XYZ for the oracle.
# ---------------------------------------------------------------------------
_BOHR_TO_ANGSTROM = 1.0 / 1.8897259885789233

# ---------------------------------------------------------------------------
# Geometries in ångström (XYZ standard).
# Converted from tests/test_gfn2_xtb.py bohr fixtures with _BOHR_TO_ANGSTROM.
# ---------------------------------------------------------------------------
_GEOMETRIES_XYZ = {
    "H": "1\n\nH  0.0 0.0 0.0\n",
    "C": "1\n\nC  0.0 0.0 0.0\n",
    "N": "1\n\nN  0.0 0.0 0.0\n",
    "O": "1\n\nO  0.0 0.0 0.0\n",
    "F": "1\n\nF  0.0 0.0 0.0\n",
    "H2": ("2\n\nH  0.000000  0.000000  0.000000\n"
           "H  0.740848  0.000000  0.000000\n"),
    "H2O": (
        "3\n\n"
        "O   0.000000  0.000000  0.000000\n"
        "H   0.757331  0.586388  0.000000\n"
        "H  -0.757331  0.586388  0.000000\n"
    ),
    "CH4": (
        "5\n\n"
        "C   0.000000  0.000000  0.000000\n"
        "H   0.611041  0.611041  0.611041\n"
        "H  -0.611041 -0.611041  0.611041\n"
        "H  -0.611041  0.611041 -0.611041\n"
        "H   0.611041 -0.611041 -0.611041\n"
    ),
    "NH3": (
        "4\n\n"
        "N   0.000000  0.000000  0.116419\n"
        "H   0.936644  0.000000 -0.275172\n"
        "H  -0.468851  0.811758 -0.275172\n"
        "H  -0.468851 -0.811758 -0.275172\n"
    ),
    "CO2": (
        "3\n\n"
        "C   0.000000  0.000000  0.000000\n"
        "O   1.162073  0.000000  0.000000\n"
        "O  -1.162073  0.000000  0.000000\n"
    ),
    "C2H4": (
        "6\n\n"
        "C   0.000000  0.000000  0.000000\n"
        "C   1.322943  0.000000  0.000000\n"
        "H  -0.547698  0.000000  0.926060\n"
        "H  -0.547698  0.000000 -0.926060\n"
        "H   1.870642  0.000000  0.926060\n"
        "H   1.870642  0.000000 -0.926060\n"
    ),
    # ---- Heavy single atoms (frozen-core valence audit) -------------------
    # These exercise gfn2_valence_electrons() past a completed d/f series,
    # where GFN2 freezes the (n−1)d¹⁰ (and, in period 6, the 4f¹⁴) in the
    # core.  Wired here so a live xtb on a Linux host can supply quantitative
    # references for tests/test_gfn2_xtb.py::test_heavy_element_valence_*;
    # vibe-qc's own gate already pins the bound/neutral/converged invariants.
    # Run each with the multiplicity in _ATOM_GFN2_MULT below.
    "Ge": "1\n\nGe 0.0 0.0 0.0\n",
    "Br": "1\n\nBr 0.0 0.0 0.0\n",
    "Kr": "1\n\nKr 0.0 0.0 0.0\n",
    "Sn": "1\n\nSn 0.0 0.0 0.0\n",
    "I": "1\n\nI  0.0 0.0 0.0\n",
    "Xe": "1\n\nXe 0.0 0.0 0.0\n",
    "W": "1\n\nW  0.0 0.0 0.0\n",
    "Os": "1\n\nOs 0.0 0.0 0.0\n",
    "Pt": "1\n\nPt 0.0 0.0 0.0\n",
    "Pb": "1\n\nPb 0.0 0.0 0.0\n",
    "Po": "1\n\nPo 0.0 0.0 0.0\n",
    "Rn": "1\n\nRn 0.0 0.0 0.0\n",
    "Pr": "1\n\nPr 0.0 0.0 0.0\n",
    "Eu": "1\n\nEu 0.0 0.0 0.0\n",
    "Lu": "1\n\nLu 0.0 0.0 0.0\n",
}

# Multiplicity (2S+1) to run each heavy single atom at, matching the
# vibe-qc comparison in test_gfn2_xtb.py: even GFN2-valence atoms run as the
# closed-shell singlet the closed-shell driver uses; odd-valence atoms
# (Br/I 4s²4p⁵-type, the trivalent f-in-core lanthanides) run as doublets.
# (xtb's --uhf flag takes the unpaired-electron count, i.e. mult − 1.)
_ATOM_GFN2_MULT = {
    "Ge": 1, "Br": 2, "Kr": 1, "Sn": 1, "I": 2, "Xe": 1,
    "W": 1, "Os": 1, "Pt": 1, "Pb": 1, "Po": 1, "Rn": 1,
    "Pr": 2, "Eu": 2, "Lu": 2,
}


def _find_xtb() -> str:
    """Locate the ``xtb`` binary on PATH."""
    xtb = os.environ.get("VIBEQC_XTB_BINARY", "xtb")
    return xtb


def _stream_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_xtb(
    xyz: str,
    gfn: int = 2,
    charge: int = 0,
    mult: int = 1,
    acc: float = 1.0,
    artifact_dir: Optional[Path] = None,
) -> dict:
    """Run ``xtb`` on an XYZ geometry string and parse the output.

    Returns a dict with keys: energy, gradient_norm, dipole, charges,
    homo, lumo, gap, n_iter, converged, runtime_s.
    """
    def _run_in(tmpdir: Path) -> dict:
        xtb_bin = _find_xtb()
        xyz_path = tmpdir / "input.xyz"
        xyz_path.write_text(xyz, encoding="utf-8")

        cmd = [
            xtb_bin,
            str(xyz_path),
            "--gfn",
            str(gfn),
            "--chrg",
            str(charge),
            "--uhf",
            str(mult - 1),
            "--acc",
            str(acc),
            "--parallel",
            "1",
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"xtb binary not found: {xtb_bin!r}. "
                f"Install via: conda install -c conda-forge xtb"
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

        result = _parse_xtb_output(proc.stdout, proc.stderr)
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


def _parse_xtb_output(stdout: str, stderr: str) -> dict:
    """Parse xTB's text output into a result dict."""
    out: dict = {"converged": False}

    for line in stdout.splitlines():
        line_stripped = line.strip()

        # Total energy line: "   * total energy  :   -5.070488964533 Eh"
        if "total energy" in line_stripped.lower() and ":" in line_stripped:
            parts = line_stripped.split(":")
            if len(parts) >= 2:
                token = parts[-1].strip().split()[0]
                try:
                    out["energy"] = float(token)
                except ValueError:
                    pass

        # Gradient norm
        if "gradient norm" in line_stripped.lower() and ":" in line_stripped:
            parts = line_stripped.split(":")
            if len(parts) >= 2:
                token = parts[-1].strip().split()[0]
                try:
                    out["gradient_norm"] = float(token)
                except ValueError:
                    pass

        # HOMO/LUMO gap
        if "HOMO-LUMO gap" in line_stripped and "/" in line_stripped:
            # "   -> HOMO-LUMO gap   8.234 / eV"
            parts = line_stripped.split()
            for i, w in enumerate(parts):
                if w == "gap" and i + 1 < len(parts):
                    try:
                        out["gap_ev"] = float(parts[i + 1])
                    except ValueError:
                        pass

        # SCF convergence
        if "converged" in line_stripped.lower() and "scf" in line_stripped.lower():
            out["converged"] = True
            # Try to extract iteration count
            for w in line_stripped.split():
                try:
                    out["n_iter"] = int(w)
                except ValueError:
                    pass

    if "energy" not in out:
        out["error"] = "energy not found in xtb output"
        out["stderr_tail"] = stderr[-500:] if stderr else ""

    return out


# ---------------------------------------------------------------------------
# Water-dimer SCC-only interaction references (Eh): E_dim − 2·E_mono, for the
# EXACT tests/test_gfn2_xtb.py::_water geometries.
#
# Pinned from a live xtb 6.7.1 GFN2 run (vq job, compute-study, 2026-06-26; xtb fetched
# into the job workdir from the official linux-x86_64 release). For these
# neutral water pairs D4 dispersion is ~0 at the printed precision (total energy
# = SCC + repulsion exactly), so the total interaction IS the SCC-only one —
# directly comparable to vibe-qc's run_gfn2_xtb (which omits the post-SCF D4)
# and immune to vibe-qc's known-defective native D4 dataset (C6 4–32× too large,
# audit 2026-05-31). NOT fabricated (CLAUDE.md §8).
#
# These are POSITIVE (repulsive): _water(0)+_water(x0) is two parallel-translated
# waters (∥ dipoles, side by side), so GFN2 gives Pauli/overlap repulsion, not an
# H-bond. xtb monomer total on this geometry is −5.070374 Eh.
# ---------------------------------------------------------------------------
_REFERENCE_DIMER_SCC_INT_GFN2: dict = {
    "trans_5.6": 0.011705,  # xtb 6.7.1 GFN2
    "trans_6.0": 0.007006,
    "trans_7.0": 0.002915,
}


def dimer_scc_interaction_ref(name: str) -> float:
    """xtb GFN2 SCC-only (total − dispersion) interaction energy for a named
    water-dimer geometry.  Raises KeyError until pinned from a live xtb run."""
    return _REFERENCE_DIMER_SCC_INT_GFN2[name]


def geometry_xyz(system_name: str) -> str:
    """Return the pinned XYZ geometry for a named xtb reference system."""
    try:
        return _GEOMETRIES_XYZ[system_name]
    except KeyError as exc:
        raise KeyError(
            f"No geometry defined for {system_name!r}. "
            f"Available: {sorted(_GEOMETRIES_XYZ.keys())}"
        ) from exc


def energy(system_name: str, gfn: int = 2, charge: int = 0, mult: int = 1) -> float:
    """Get the xTB GFN2 total energy for a named system.

    Uses the pinned reference values.  Pass ``--live`` to run xtb instead.
    """
    key = system_name.upper() if system_name in ("h2",) else system_name
    if key in _REFERENCE_ENERGIES_GFN2:
        return _REFERENCE_ENERGIES_GFN2[key]
    raise KeyError(
        f"No pinned reference for {system_name!r}. "
        f"Available: {sorted(_REFERENCE_ENERGIES_GFN2.keys())}"
    )


def energy_live(
    system_name: str, gfn: int = 2, charge: int = 0, mult: Optional[int] = None
) -> float:
    """Run xtb live and return the total energy.

    ``mult`` defaults to the entry in :data:`_ATOM_GFN2_MULT` for heavy single
    atoms (so ``--live --system Pb`` runs the right spin state), else 1.
    """
    xyz = _GEOMETRIES_XYZ.get(system_name)
    if xyz is None:
        raise KeyError(f"No geometry defined for {system_name!r}")
    if mult is None:
        mult = _ATOM_GFN2_MULT.get(system_name, 1)
    result = _run_xtb(xyz, gfn=gfn, charge=charge, mult=mult)
    if not result.get("converged"):
        raise RuntimeError(
            f"xtb did not converge for {system_name}: {result.get('error', 'unknown')}"
        )
    return result["energy"]


# ---------------------------------------------------------------------------
# CLI — ``python -m examples.regression.core.runner_xtb``
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="xTB reference-energy oracle")
    ap.add_argument("--gfn", type=int, default=2, help="GFN level (0,1,2)")
    ap.add_argument("--system", default=None, help="Named system from reference table")
    ap.add_argument("--xyz", default=None, help="Path to .xyz file")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument(
        "--mult",
        type=int,
        default=None,
        help="spin multiplicity 2S+1 (default: per-atom hint for heavy atoms, else 1)",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Run xtb live instead of returning pinned values",
    )
    ap.add_argument(
        "--dump-refs",
        action="store_true",
        help="Print pinned reference table (for regeneration)",
    )
    ap.add_argument(
        "--acc", type=float, default=1.0, help="Accuracy (1.0=normal, 0.1=tight)"
    )
    args = ap.parse_args()

    if args.dump_refs:
        print(json.dumps(_REFERENCE_ENERGIES_GFN2, indent=2))
        sys.exit(0)

    if args.live:
        if args.system:
            e = energy_live(
                args.system, gfn=args.gfn, charge=args.charge, mult=args.mult
            )
        elif args.xyz:
            xyz = Path(args.xyz).read_text()
            result = _run_xtb(
                xyz, gfn=args.gfn, charge=args.charge,
                mult=args.mult if args.mult is not None else 1,
            )
            e = result.get("energy")
        else:
            ap.error("--live requires --system or --xyz")
        print(f"{_RESULT_MARKER}{json.dumps({'energy': e})}")
    else:
        if args.system:
            print(f"{args.system}: {energy(args.system):.6f} Eh")
        else:
            ap.error("require --system or --live or --dump-refs")
