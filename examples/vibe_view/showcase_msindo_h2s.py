"""vibe-view showcase: H2S with MSINDO (semiempirical INDO, 3rd-row d shell).

MSINDO (Ahlswede & Jug, J. Comput. Chem. 20, 563/572 (1999)) is a basis-set-free
semiempirical method — no Gaussian basis, just built-in Slater orbitals.  This
showcase runs H2S (the sulphur carries a 3d polarization shell + a Ne frozen
core, so it exercises vibe-qc's d-shell MSINDO path) and writes a ``.qvf``
archive for vibe-view.

Because MSINDO works in a minimal orthogonal Slater basis (not a Gaussian basis),
the QVF carries the **structure** and the **citations** (the Ahlswede-Jug method
papers, assembled automatically — CLAUDE.md § 8).  Gaussian-grid sections
(density / MO isosurfaces, ``wavefunction.gto``) do not apply to a Slater
semiempirical wavefunction.  The script also prints the MSINDO total energy and
the atomic charges (which coincide for Mulliken/Löwdin in MSINDO's orthogonal
basis) so you can read the chemistry directly.

Run:

    .venv/bin/python examples/vibe_view/showcase_msindo_h2s.py

Output (under ``examples/vibe_view/runs/h2s_msindo/``):

    h2s.qvf        <- open this with vibe-view
    h2s.out
    h2s.bibtex / h2s.references   (the MSINDO method papers, paper-ready)
    h2s.system     (TOML provenance manifest)
    h2s.xyz

Then:

    vibe-view open examples/vibe_view/runs/h2s_msindo/h2s.qvf

vibe-view opens the browser at http://127.0.0.1:8080 with the **structure**
panel (sulphur centre, two hydrogens, H-S-H ~92°) and the **citations** panel
(BibTeX ready to paste).  Spin/zoom the molecule; the bonds are drawn from
covalent radii.
"""

from __future__ import annotations

from pathlib import Path

from vibeqc import Atom, Molecule, run_job
from vibeqc.semiempirical.methods import msindo

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs" / "h2s_msindo"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# H2S near equilibrium (S-H ~1.336 Å, H-S-H ~92°).  Atom coords are bohr
# (vibe-qc's native unit); 0.9627 Å = 1.819 bohr, 0.9264 Å = 1.751 bohr.
A2B = msindo.ANGSTROM_TO_BOHR
h2s = Molecule([
    Atom(16, [0.0, 0.0, 0.0]),
    Atom(1, [0.9627 * A2B, 0.0, 0.9264 * A2B]),
    Atom(1, [-0.9627 * A2B, 0.0, 0.9264 * A2B]),
])


def main() -> None:
    print("=" * 72)
    print(" vibe-view showcase: H2S / MSINDO (semiempirical, 3rd-row d shell)")
    print("=" * 72)
    print(f"\nOutput directory: {RUN_DIR}\n")

    result = run_job(
        h2s,
        method="msindo",
        write_xyz_file=True,
        citations=True,      # writes citations + .bibtex (Ahlswede-Jug 1999)
        output=RUN_DIR / "h2s",
        output_qvf=True,     # the vibe-view artefact (structure + citations)
    )
    print(f"\nMSINDO total energy: {float(result.energy):.10f} Ha"
          f"  (reference MSINDO -11.2354666980)")

    # Atomic charges from the orthogonal MSINDO density: q_A = CZ_A − Σ P_μμ.
    r = msindo.run_msindo([16, 1, 1],
                          [[0.0, 0.0, 0.0], [0.9627, 0.0, 0.9264],
                           [-0.9627, 0.0, 0.9264]])
    blocks, _ = msindo._atom_blocks([16, 1, 1])
    print("\nAtomic charges (Mulliken = Löwdin in MSINDO's orthogonal basis):")
    for sym, z, (lo, hi) in zip(["S", "H", "H"], [16, 1, 1], blocks):
        q = msindo.eff_core_charge(z) - float(r.density.diagonal()[lo:hi].sum())
        print(f"   {sym:<2}  {q:+.3f} e")

    print("\n" + "=" * 72)
    print(" Open it in vibe-view:")
    print(f"\n   vibe-view open {RUN_DIR / 'h2s.qvf'}\n")
    print(" The browser opens at http://127.0.0.1:8080 with the structure and")
    print(" citations panels.  (Density/MO isosurfaces don't apply to a Slater")
    print(" semiempirical wavefunction.)")
    print("=" * 72)


if __name__ == "__main__":
    main()
