"""vibe-view showcase: NEB reaction path (+ a DFT+U variant).

Produces .qvf archives whose headline section is ``reaction.path`` — the
section vibe-view's reaction renderer animates frame-by-frame with
reactant / transition-state / product waypoints and an energy profile.

Two calculations, two archives:

  1. ``neb_h3.qvf``        — collinear H + H₂ → H₂ + H exchange via a
                             climbing-image NEB at UHF / STO-3G. The
                             classic 3-atom hydrogen-exchange barrier; the
                             climbing image drives an image toward the saddle
                             so the TS waypoint sits at the barrier top of the
                             energy profile. (max_iter is capped for a fast
                             demo, so it lands near-converged — fine for
                             visualising the path; bump max_iter for a tight
                             saddle.) Sections: structure, reaction.path,
                             citations.

  2. ``neb_h3_plus_u.qvf`` — the SAME reaction with a Hubbard +U correction
                             applied (Dudarev rotationally-invariant
                             functional) via ``dft_plus_u=[HubbardSite(...)]``.
                             DFT+U adds no new QVF section kind — its visible
                             effect in vibe-view is in the *citations* panel,
                             which now also lists Dudarev (1998) and
                             Cococcioni & de Gironcoli (2005). A modest
                             U = 2 eV on the H 1s shell keeps the NEB well
                             behaved; this is a demonstration of the +U
                             routing, not a physically motivated value.

Why H + H₂? It is the smallest system with a genuine reaction barrier,
converges in seconds, and exercises the full reaction-path surface
(interpolated images, energy profile, climbing-image TS).

Run:

    .venv/bin/python examples/vibe_view/showcase_neb_reaction.py

Output (under examples/vibe_view/runs/neb_reaction/):

    neb_h3.qvf            <- open this; walk the reaction.path frames
    neb_h3_plus_u.qvf     <- same path; check the citations panel for +U refs

Then:

    vibe-view open examples/vibe_view/runs/neb_reaction/neb_h3.qvf

In vibe-view: activate the reaction.path section. Step through the images
(reactant → TS → product), watch the energy profile, and confirm the TS
waypoint sits at the barrier top. Open the citations panel and compare the
two archives — the +U run carries the two extra DFT+U references.
"""

from __future__ import annotations

from pathlib import Path

import vibeqc as vq
from vibeqc import Atom, Molecule, UHFOptions, run_neb

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs" / "neb_reaction"
RUN_DIR.mkdir(parents=True, exist_ok=True)


def _h3(z_positions: list[float]) -> Molecule:
    """Three collinear H atoms (doublet) along z, positions in bohr."""
    return Molecule([Atom(1, [0.0, 0.0, float(z)]) for z in z_positions], 0, 2)


# Reactant: H_a + H_b–H_c  (H_a far, H_b–H_c bonded ~1.4 bohr)
# Product:  H_a–H_b + H_c  (H_a–H_b bonded, H_c far)
REACTANT = _h3([0.0, 1.4, 4.4])
PRODUCT = _h3([0.0, 3.0, 4.4])


def _run_neb(*, dft_plus_u=None, stem: str, label: str) -> None:
    print(f"\n--- {label} ---")
    uhf = UHFOptions()
    uhf.max_iter = 200
    result = run_neb(
        REACTANT,
        PRODUCT,
        basis="sto-3g",
        n_images=5,
        method="UHF",
        uhf_options=uhf,
        interpolation="idpp",
        climbing_image=True,
        max_iter=60,
        conv_tol_force=2e-3,
        n_jobs=1,
        dft_plus_u=dft_plus_u,
    )
    converged = getattr(result, "converged", None)
    max_force = getattr(result, "max_force", None)
    print(f"  converged={converged}  max_force={max_force}")
    path = result.write_qvf(RUN_DIR / stem)
    print(f"  wrote {path}")

    import json
    import zipfile

    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        kinds = sorted({s["kind"] for s in manifest["sections"]})
        print(f"  sections: {kinds}")


def main() -> None:
    print("=" * 72)
    print(" vibe-view showcase: NEB reaction path (H + H₂ exchange) + DFT+U")
    print("=" * 72)

    # 1. Plain climbing-image NEB.
    _run_neb(stem="neb_h3", label="NEB: H + H₂ → H₂ + H (UHF / STO-3G, CI-NEB)")

    # 2. Same reaction with a Hubbard +U correction (Dudarev). U on the
    #    H 1s shell (l=0). Demonstrates the DFT+U citation routing.
    _run_neb(
        dft_plus_u=[vq.HubbardSite(atom_index=0, l=0, U_ev=2.0)],
        stem="neb_h3_plus_u",
        label="NEB + DFT+U: U=2 eV on H(0) 1s shell",
    )

    print()
    print("=" * 72)
    print(" Done. Open either archive in vibe-view:")
    print(f"   vibe-view open {RUN_DIR / 'neb_h3.qvf'}")
    print(f"   vibe-view open {RUN_DIR / 'neb_h3_plus_u.qvf'}")
    print(" Activate reaction.path; step reactant → TS → product. Compare")
    print(" the citations panels — the +U run lists Dudarev & Cococcioni.")
    print("=" * 72)


if __name__ == "__main__":
    main()
