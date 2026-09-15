"""Cross-validate NH3 umbrella inversion barrier via NEB.

The classic NEB textbook problem: ammonia inverts through a planar
D3h transition state. The barrier height is well-known (~5.8 kcal/mol
experimentally; HF/STO-3G overshoots to ~10 kcal/mol). This script
runs the full NEB through vibe-qc's calculator + ORCA's calculator
and compares the barriers.

Run:
    .venv/bin/python examples/ase_compare/compare-nh3-neb-barrier.py

Wall time: ~3-5 minutes per calculator (ASE NEB on 5 intermediate
images, climbing-image enabled). Skip ORCA branch if not available.

Produces:
    output-compare-nh3-neb-barrier.csv
    output-compare-nh3-neb-barrier.png  — MEP plot side-by-side

Cross-validation succeeds when both barriers agree to within
1 kcal/mol — they should, since the underlying SCF agrees to µHa
and NEB is a deterministic optimization on top.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.mep import NEB
from ase.optimize import FIRE, BFGSLineSearch

from vibeqc.ase import VibeQC
from vibeqc.benchmark import (
    make_orca_calculator,
    print_calculator_availability,
)

HERE = Path(__file__).resolve().parent

EV_PER_KCAL = 1.0 / 23.06054783  # eV per kcal/mol


def nh3_endpoints() -> tuple[Atoms, Atoms]:
    """Two enantiomers of pyramidal NH3 — Hs above vs below N.

    Geometry from optimized HF/STO-3G; close enough that endpoint
    relaxation converges in a handful of steps.
    """
    # Pyramidal: H's at z = -0.31, N at z = 0.06.
    n_z, h_z = 0.06, -0.31
    initial = Atoms(
        symbols=["N", "H", "H", "H"],
        positions=[
            (0.0, 0.0, n_z),
            (0.940, 0.0, h_z),
            (-0.470, 0.814, h_z),
            (-0.470, -0.814, h_z),
        ],
    )
    final = initial.copy()
    final.positions[:, 2] *= -1.0  # flip across xy-plane
    return initial, final


def _make_neb_with(calc_factory, n_images: int = 5) -> NEB:
    """Build a fresh NEB with `n_images` intermediates between the two
    pyramidal endpoints. Each image gets its own calculator instance
    (calc_factory()) — sharing one Calculator across images would
    silently overwrite state."""
    initial, final = nh3_endpoints()

    # Endpoint relaxation — keeps the band's endpoints honest (they're
    # the reference for the barrier).
    initial.calc = calc_factory()
    BFGSLineSearch(initial, logfile=None).run(fmax=0.05, steps=30)
    final.calc = calc_factory()
    BFGSLineSearch(final, logfile=None).run(fmax=0.05, steps=30)

    # Build the band of images.
    images = [initial]
    for _ in range(n_images):
        img = initial.copy()
        img.calc = calc_factory()
        images.append(img)
    images.append(final)

    # Linear interpolation between endpoints; the climbing-image
    # algorithm refines from there.
    neb = NEB(images, climb=True)
    neb.interpolate()
    # Re-attach calculators after interpolate() (it doesn't propagate them).
    for img in images[1:-1]:
        img.calc = calc_factory()
    return neb


def run_neb(label: str, calc_factory) -> dict | None:
    """Run a full NEB + climbing-image refinement, return the MEP
    energies (eV) relative to the endpoint."""
    print(f"\n--- {label} ---", flush=True)
    try:
        neb = _make_neb_with(calc_factory, n_images=5)
        FIRE(neb, logfile=None).run(fmax=0.10, steps=50)
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {e}")
        return None
    energies = np.array([img.get_potential_energy() for img in neb.images])
    e0 = energies[0]
    barrier_eV = energies.max() - e0
    barrier_kcal = barrier_eV / EV_PER_KCAL
    print(f"  endpoint energy: {e0:.6f} eV")
    print(f"  TS energy:       {energies.max():.6f} eV")
    print(f"  barrier:         {barrier_eV*1000:.3f} meV  "
          f"({barrier_kcal:.3f} kcal/mol)")
    return {
        "endpoint_eV": float(e0),
        "energies_eV": energies.tolist(),
        "barrier_eV": float(barrier_eV),
        "barrier_kcal_mol": float(barrier_kcal),
    }


def main() -> None:
    print("=" * 72)
    print(" Cross-validation:  NH3 umbrella-inversion barrier via NEB-CI")
    print("=" * 72)
    print()
    print("Calculator availability:")
    print_calculator_availability()
    print()
    print("Method: HF / STO-3G  (qualitative only — HF + minimal basis "
          "overshoots\n         the experimental ~5.8 kcal/mol barrier "
          "by ~2×)")
    print()

    rows: dict[str, dict] = {}

    rows["vibe-qc"] = run_neb(
        "vibe-qc / RHF / STO-3G",
        lambda: VibeQC(basis="sto-3g"),
    )

    def _orca_factory():
        return make_orca_calculator(
            orcasimpleinput="HF STO-3G EnGrad",
            label="orca-nh3-neb",
        )
    if _orca_factory() is not None:
        rows["ORCA"] = run_neb(
            "ORCA / RHF / STO-3G",
            _orca_factory,
        )
    else:
        print("\n(ORCA not available — set ORCA_COMMAND or add orca to "
              "$PATH and re-run for the cross-comparison.)")

    # Side-by-side summary
    print("\n" + "=" * 72)
    print(" Barriers")
    print("=" * 72)
    for code, r in rows.items():
        if r is None:
            continue
        print(f"  {code:<10}  barrier = {r['barrier_kcal_mol']:.3f} kcal/mol")

    if "vibe-qc" in rows and "ORCA" in rows and rows["vibe-qc"] and rows["ORCA"]:
        gap = abs(
            rows["vibe-qc"]["barrier_kcal_mol"]
            - rows["ORCA"]["barrier_kcal_mol"]
        )
        print(f"\nvibe-qc vs ORCA disagreement on barrier: {gap*1000:.2f} cal/mol")
        # Tolerance: 1 kcal/mol. Both codes start the NEB from the same
        # initial linear interpolation, but the FIRE optimizer can
        # land in slightly different points along the MEP — the TS
        # itself agrees to within whatever the SCF/CI does.
        assert gap < 1.0, (
            f"NH3 barriers disagree by {gap:.3f} kcal/mol — investigate "
            f"NEB convergence or SCF settings"
        )
        print("✓ vibe-qc and ORCA agree on the NH3 inversion barrier "
              "to within 1 kcal/mol on the same NEB protocol.")

    # Plot MEP if matplotlib's around.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for code, r in rows.items():
        if r is None:
            continue
        e = np.array(r["energies_eV"]) - r["endpoint_eV"]
        ax.plot(range(len(e)), e * 1000, marker="o", label=code)
    ax.set_xlabel("NEB image index")
    ax.set_ylabel("ΔE / meV")
    ax.set_title("NH3 umbrella inversion — MEP via NEB-CI")
    ax.legend()
    ax.grid(alpha=0.3)
    png_path = HERE / "output-compare-nh3-neb-barrier.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"\nMEP plot written to "
          f"{png_path.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
