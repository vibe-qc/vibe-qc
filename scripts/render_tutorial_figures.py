#!/usr/bin/env python3
"""Generate vibe-view tutorial screenshots from vibe-qc calculations.

Run this from the project root. Requires vibe-qc and vibe-view installed.

Usage:
    .venv/bin/python scripts/render_tutorial_figures.py

Outputs screenshots into docs/_static/plots/vibe_view/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure offscreen rendering
os.environ["PYVISTA_OFF_SCREEN"] = "True"

_OUTPUT_DIR = Path("docs/_static/plots/vibe_view")


def ensure_dir() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def molecular_h2o() -> str:
    """Run H2O RKS/PBE/6-31G* and produce a QVF. Returns path to QVF."""
    from vibeqc import Atom, Molecule, run_job

    mol = Molecule(
        [
            Atom(8, [0.0, 0.00, 0.00]),
            Atom(1, [0.0, 1.43, -0.98]),
            Atom(1, [0.0, -1.43, -0.98]),
        ]
    )

    run_job(
        mol,
        basis="sto-3g",
        method="rks",
        functional="PBE",
        output="h2o_tutorial",
        output_qvf=True,
        write_cube=["density", "homo", "lumo"],
        write_molden_file=True,
        hessian=True,
        optimize=True,
    )
    return "h2o_tutorial.qvf"


def capture_h2o_screenshots(qvf_path: str) -> None:
    """Generate screenshots from the H2O QVF."""
    from vibeview.capture import (
        capture_bond_orders,
        capture_dos,
        capture_energy_diagram,
        capture_scf_history,
        capture_spectra,
        capture_structure,
        capture_volume,
    )
    from vibeview.qvf import QVFReader

    reader = QVFReader(qvf_path)

    # Structure
    capture_structure(
        reader,
        str(_OUTPUT_DIR / "tutorial-h2o-structure.png"),
        representation="ball_and_stick",
        show_labels=False,
    )
    print("  [OK] structure")

    # Density (section id auto-generated as vol_dens_0)
    ok = capture_volume(
        reader,
        "vol_dens_0",
        str(_OUTPUT_DIR / "tutorial-h2o-density.png"),
        isovalue=0.05,
        colormap="viridis",
        opacity=0.6,
    )
    print(f"  [{'OK' if ok else 'FAIL'}] density")

    # HOMO (first orbital: vol_mo_0)
    ok = capture_volume(
        reader,
        "vol_mo_0",
        str(_OUTPUT_DIR / "tutorial-h2o-homo.png"),
        isovalue=0.04,
        colormap="RdBu",
        opacity=0.6,
    )
    print(f"  [{'OK' if ok else 'FAIL'}] HOMO")

    # SCF history
    capture_scf_history(reader, str(_OUTPUT_DIR / "tutorial-h2o-scf.html"))
    print("  [OK] SCF history")

    # IR spectrum
    capture_spectra(reader, str(_OUTPUT_DIR / "tutorial-h2o-ir.html"))
    print("  [OK] IR spectrum")

    # Bond orders
    capture_bond_orders(reader, str(_OUTPUT_DIR / "tutorial-h2o-bond-orders.html"))
    print("  [OK] bond orders")

    # Energy diagram
    capture_energy_diagram(reader, str(_OUTPUT_DIR / "tutorial-h2o-grotrian.html"))
    print("  [OK] energy diagram")

    reader.close()


def periodic_si() -> str:
    """Run Si diamond periodic RHF/sto-3g and produce a QVF. Returns path."""
    import numpy as np
    import vibeqc as vq

    a_bohr = 3.567 / 0.529177  # diamond cubic
    cell = np.eye(3) * a_bohr
    frac = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.25, 0.25, 0.25],
        ]
    )
    si = vq.PeriodicSystem(
        3,
        cell,
        [vq.Atom(14, frac[0] @ cell), vq.Atom(14, frac[1] @ cell)],
    )
    basis = vq.BasisSet(si.unit_cell_molecule(), "sto-3g")

    kpath = vq.kpath_from_segments(
        si,
        segments=[
            ([0.0, 0.0, 0.0], "G", [0.5, 0.0, 0.5], "X"),
            ([0.5, 0.0, 0.5], "X", [0.5, 0.25, 0.75], "W"),
            ([0.5, 0.25, 0.75], "W", [0.375, 0.375, 0.75], "K"),
            ([0.375, 0.375, 0.75], "K", [0.0, 0.0, 0.0], "G"),
        ],
        points_per_segment=15,
    )

    vq.run_periodic_job(
        si,
        basis=basis,
        method="RHF",
        output="si_tutorial",
        output_qvf=True,
        write_density=True,
        band_structure=vq.band_structure_hcore(si, basis, kpath),
        dos_kmesh=[8, 8, 8],
    )
    return "si_tutorial.qvf"


def capture_si_screenshots(qvf_path: str) -> None:
    """Generate screenshots from the Si diamond QVF."""
    from vibeview.capture import (
        capture_bands,
        capture_dos,
        capture_structure,
        capture_volume,
    )
    from vibeview.qvf import QVFReader

    reader = QVFReader(qvf_path)

    # Structure with 2x2x2 replication
    ok = capture_structure(
        reader,
        str(_OUTPUT_DIR / "tutorial-si-structure.png"),
        representation="ball_and_stick",
        replication=(2, 2, 2),
    )
    print(f"  [{'OK' if ok else 'FAIL'}] Si structure (2x2x2)")

    # Density
    ok = capture_volume(
        reader,
        "vol_dens_0",
        str(_OUTPUT_DIR / "tutorial-si-density.png"),
        isovalue=0.03,
        colormap="viridis",
        replication=(2, 2, 2),
    )
    print(f"  [{'OK' if ok else 'FAIL'}] Si density")

    # Bands
    capture_bands(reader, str(_OUTPUT_DIR / "tutorial-si-bands.png"))
    print("  [OK] Si bands")

    # DOS
    capture_dos(reader, str(_OUTPUT_DIR / "tutorial-si-dos.html"))
    print("  [OK] Si DOS")

    reader.close()


def main() -> int:
    ensure_dir()
    print("==> vibe-view tutorial figure generator")
    print()

    # Molecular: H2O
    print("-- Molecular: H2O RKS/PBE/sto-3g --")
    try:
        qvf = molecular_h2o()
        capture_h2o_screenshots(qvf)
        # Clean up the QVF
        Path(qvf).unlink(missing_ok=True)
        for f in Path().glob("h2o_tutorial.*"):
            f.unlink(missing_ok=True)
    except Exception as e:
        print(f"  [SKIP] Molecular: {e}", file=sys.stderr)

    print()

    # Periodic: Si diamond
    print("-- Periodic: Si diamond RHF/sto-3g --")
    try:
        qvf = periodic_si()
        capture_si_screenshots(qvf)
        Path(qvf).unlink(missing_ok=True)
        for f in Path().glob("si_tutorial.*"):
            f.unlink(missing_ok=True)
    except Exception as e:
        print(f"  [SKIP] Periodic: {e}", file=sys.stderr)

    print()
    print(f"==> Done. Screenshots in {_OUTPUT_DIR.resolve()}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
