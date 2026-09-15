"""vibe-view showcase: formaldehyde (H2CO) at RHF / 6-31G*.

Single calculation that produces a .qvf carrying every section
vibe-view knows how to render today. Formaldehyde is the right
system for a showcase because it is small enough to finish in
under a minute on any laptop and chemically rich enough to show
something interesting in every panel:

  * structure         a planar 4-atom molecule (C, O, two Hs); good
                      view for the unit-cell-off / bonds-on rendering.
  * volume.density    the C=O bond density plus the lone pairs on
                      oxygen are clearly visible at the default
                      isovalue.
  * volume.orbital    HOMO is the in-plane oxygen lone pair (n);
                      LUMO is the C=O pi* with one node through the
                      C=O bond. Distinct lobes, perfect for the
                      isosurface renderer.
  * wavefunction.gto  full MO basis embedded so vibe-view can
                      resample any orbital on demand.
  * vibrations        H2CO has six normal modes; three are IR-bright:
                      C=O stretch, HCH scissoring, C-H symmetric
                      stretch. Each mode animates with believable
                      amplitudes. (RHF/6-31G* puts the frequencies
                      5 to 10 percent above experiment because there
                      is no electron correlation; mode shapes are
                      still correct.)
  * spectra.ir        the three IR peaks land on the plot with
                      hover tooltips for frequency + intensity.
  * atom_properties   Mulliken / Loewdin charges show the expected
                      polarity: O strongly negative, C positive,
                      Hs slightly positive.
  * citations         the BibTeX bundle the runtime assembled:
                      vibe-qc itself, libint, libxc, 6-31G* (Hariharan
                      and Pople 1973). Ready to paste into a paper.

Why RHF and not PBE / B3LYP? Two reasons:

* H2CO + PBE / 6-31G* trips the documented AUTO / SAP / DIIS
  oscillation pattern (homepage Known-issues admonition); SAD
  helps at the equilibrium but the SCF dynamics are fragile to
  optimiser-step intermediate geometries. RHF converges in a
  dozen iters on the same system without any handholding.
* The showcase is a UI / output-format demonstration, not a
  chemistry benchmark. Skipping correlation costs a few mHa on
  the SCF energy and bumps the harmonic frequencies up by 5 to
  10 percent. Both are fine for screenshots; the QVF section
  surface is identical.

If you want a DFT version, swap ``method="rhf"`` for
``method="rks"`` and add ``functional="b3lyp"``. B3LYP is more
robust than PBE on this fixture; the IR frequencies come down
close to experiment (~1750 / 1500 / 2900 cm^-1).

Run:

    ~/path/to/vibeqc/.venv/bin/python examples/vibe_view/showcase_formaldehyde.py

Output (all under ``examples/vibe_view/runs/h2co_showcase/``):

    h2co.qvf       <- open this with vibe-view
    h2co.out
    h2co.molden
    h2co.bibtex
    h2co.references
    h2co.population.{txt,json}
    h2co.density.cube
    h2co.homo.cube
    h2co.lumo.cube
    h2co.xyz
    h2co.system        (TOML provenance manifest)

Then:

    vibe-view open examples/vibe_view/runs/h2co_showcase/h2co.qvf

vibe-view will print a startup banner listing every section in
the archive as ``rendered``, open the default browser at
http://127.0.0.1:8080 once uvicorn has actually bound the
listener, and auto-activate the density section.

Walking the panels:

    1. structure       spin / zoom. H-C-H angle ~116 deg, C=O bond
                       ~1.18 A (RHF is a bit short vs the 1.21 A
                       experimental). Bonds drawn from covalent radii.
    2. density         start at the default isovalue (~0.05 e/bohr^3),
                       walk it down to 0.02 to see the lone-pair pocket
                       behind oxygen, walk up to 0.10 to squeeze the
                       surface in toward the nuclei.
    3. homo            switch the colormap to a divergent map. The +/-
                       lobes are the two halves of the oxygen sp2 lone
                       pair, sitting in the molecular plane.
    4. lumo            same view, then notice the node running through
                       the C=O bond and the lobes above/below the
                       molecular plane. That's the pi*.
    5. wavefunction.gto    scroll the MO list. Below HOMO you should
                       see another lone-pair orbital and, deeper, the
                       C=O sigma bond. Above LUMO are progressively
                       antibonding combinations.
    6. vib             mode-by-mode: 1, 2, 3 are the three IR-bright
                       ones. The C=O stretch is unmistakable.
    7. ir              the same three modes as a Plotly stem chart with
                       hover tooltips. The C=O stretch around
                       1900 to 2000 cm^-1 at RHF/6-31G* is the
                       tallest peak (experiment is around 1750; the
                       gap is the correlation deficiency).
    8. atom_properties color-coded charges in the 3D viewport.
                       Oxygen is dark red, carbon yellow-ish, hydrogens
                       slightly white-leaning.
    9. citations       BibTeX block ready to copy.

For the full guide see docs/user_guide/vibe_view.md and
docs/tutorial/vibe_view_walkthrough.md.
"""

from __future__ import annotations

from pathlib import Path

from vibeqc import Atom, Molecule, RHFOptions, run_job

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs" / "h2co_showcase"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Formaldehyde at near-equilibrium geometry (planar C2v):
# C=O ~ 2.301 bohr, C-H ~ 2.094 bohr, HCO angle ~ 122 deg. Atom
# coordinates are in bohr (the vibe-qc native unit). Numbers are
# DFT-equilibrium; RHF will polish them slightly on the FD Hessian
# step but the geometry is close enough that the harmonic-mode
# decomposition is meaningful.
mol = Molecule(
    [
        Atom(6, [0.000, 0.0, 0.000]),  # C at origin
        Atom(8, [0.000, 0.0, 2.301]),  # O along +z
        Atom(1, [1.776, 0.0, -1.109]),  # H below
        Atom(1, [-1.776, 0.0, -1.109]),  # H below (symmetric)
    ],
)


def main() -> None:
    print("=" * 72)
    print(" vibe-view showcase: H2CO / RHF / 6-31G*")
    print("=" * 72)
    print()
    print(f"Output directory: {RUN_DIR}")
    print("Producing h2co.qvf with:")
    print("  - structure, density, HOMO, LUMO, wavefunction.gto")
    print("  - vibrational modes + IR spectrum (via FD Hessian)")
    print("  - Mulliken / Loewdin atom_properties")
    print("  - citations")
    print()

    # Roomy SCF iteration cap is cheap insurance; H2CO + RHF actually
    # converges in ~17 iters at this geometry but the Hessian FD path
    # touches displaced geometries that can take a few more.
    rhf_opts = RHFOptions()
    rhf_opts.max_iter = 200

    run_job(
        mol,
        basis="6-31g*",
        method="rhf",
        rhf_options=rhf_opts,
        optimize=False,  # already at near-equilibrium
        hessian=True,  # writes vibrations + spectra.ir
        write_cube=["density", "homo", "lumo"],  # writes density + two volume.orbital
        cube_spacing=0.2,  # bohr, default
        cube_padding=4.0,  # bohr, default
        write_molden_file=True,  # writes wavefunction.gto
        write_population_file=True,  # writes atom_properties
        write_xyz_file=True,  # final geometry as xyz
        citations=True,  # writes citations + .bibtex
        output=RUN_DIR / "h2co",
        output_qvf=True,  # the headline artefact
    )

    print()
    print("=" * 72)
    print(" Done. Now open the QVF in vibe-view:")
    print()
    print(f"   vibe-view open {RUN_DIR / 'h2co.qvf'}")
    print()
    print(" The browser opens at http://127.0.0.1:8080. Walk the sidebar")
    print(" sections in order; the docstring of this script has a panel-")
    print(" by-panel guide of what to look at in each one.")
    print("=" * 72)


if __name__ == "__main__":
    main()
