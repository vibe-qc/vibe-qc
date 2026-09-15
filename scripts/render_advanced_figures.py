#!/usr/bin/env python3
"""Generate advanced tutorial screenshots: NEB reaction path, compare mode."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PYVISTA_OFF_SCREEN"] = "True"

_OUT = Path("docs/_static/plots/vibe_view")
_OUT.mkdir(parents=True, exist_ok=True)


def neb_nh3() -> str:
    """NH3 umbrella NEB. Returns QVF path."""
    import vibeqc as vq

    # Planar NH3
    r = vq.Molecule(
        [
            vq.Atom(7, [0.0000, 0.0000, 0.1196]),
            vq.Atom(1, [0.0000, 0.9384, -0.2790]),
            vq.Atom(1, [0.8127, -0.4692, -0.2790]),
            vq.Atom(1, [-0.8127, -0.4692, -0.2790]),
        ]
    )
    # Inverted
    p = vq.Molecule(
        [
            vq.Atom(7, [0.0000, 0.0000, -0.1196]),
            vq.Atom(1, [0.0000, 0.9384, 0.2790]),
            vq.Atom(1, [0.8127, -0.4692, 0.2790]),
            vq.Atom(1, [-0.8127, -0.4692, 0.2790]),
        ]
    )

    result = vq.run_neb(
        r, p, basis="sto-3g", method="RHF", n_images=5, progress=True, max_iter=20
    )
    result.write_qvf("nh3-neb")
    return "nh3-neb.qvf"


def capture_neb(qvf_path: str) -> None:
    from vibeview.capture import capture_structure
    from vibeview.qvf import QVFReader

    reader = QVFReader(qvf_path)
    for s in reader.sections:
        print(f"  {s.id}: {s.kind}")

    capture_structure(
        reader,
        str(_OUT / "tutorial-nh3-neb-structure.png"),
        representation="ball_and_stick",
        show_labels=True,
    )
    print("  [OK] NEB structure")
    reader.close()


def compare_h2o() -> None:
    """Generate two H2O QVFs at different levels and compare."""
    from vibeqc import Atom, Molecule, run_job

    mol = Molecule(
        [
            Atom(8, [0.0, 0.00, 0.00]),
            Atom(1, [0.0, 1.43, -0.98]),
            Atom(1, [0.0, -1.43, -0.98]),
        ]
    )

    # HF/sto-3g
    run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output="h2o_hf",
        output_qvf=True,
        write_cube=["density"],
    )
    # PBE/sto-3g
    run_job(
        mol,
        basis="sto-3g",
        method="rks",
        functional="PBE",
        output="h2o_pbe",
        output_qvf=True,
        write_cube=["density"],
    )
    print("  [OK] Two H2O QVFs generated")

    # Capture structure from each
    from vibeview.capture import capture_structure, capture_volume
    from vibeview.qvf import QVFReader

    for tag, qvf in [("hf", "h2o_hf.qvf"), ("pbe", "h2o_pbe.qvf")]:
        reader = QVFReader(qvf)
        capture_structure(reader, str(_OUT / f"tutorial-compare-{tag}.png"))
        print(f"  [OK] {tag} structure")
        reader.close()


def main() -> int:
    print("==> Advanced tutorial screenshots")
    print()

    # NEB
    print("-- NH3 umbrella NEB --")
    try:
        qvf = neb_nh3()
        capture_neb(qvf)
        Path(qvf).unlink(missing_ok=True)
        for f in Path().glob("nh3-neb.*"):
            f.unlink(missing_ok=True)
    except Exception as e:
        print(f"  [SKIP] NEB: {e}")

    print()

    # Compare
    print("-- H2O HF vs PBE compare --")
    try:
        compare_h2o()
        for pat in ["h2o_hf.*", "h2o_pbe.*"]:
            for f in Path().glob(pat):
                if f.suffix != ".png":
                    f.unlink(missing_ok=True)
    except Exception as e:
        print(f"  [SKIP] Compare: {e}")

    print(f"\n==> Done: {_OUT.resolve()}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
