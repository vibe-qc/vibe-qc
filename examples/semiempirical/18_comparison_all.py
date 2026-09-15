"""Three-tier comparison: all methods on H2O.

Part of the vibe-qc semiempirical-vs-MLIP comparison study
(manuscript §5, SI §S4).  Runs every method tier on water:

  Tier 1 — Ab-initio DFT:  HF, PBE, PBE0, B3LYP (def2-SVP)
  Tier 2 — Semiempirical:  DFTB0, SCC-DFTB, GFN2-xTB, PM6,
                           OM1/OM2/OM3, MSINDO
  Tier 3 — MLIP:          MACE-MPA-0 (MIT), MACE-OFF23 (ASL)

IMPORTANT: absolute total energies are NOT comparable across tiers.
MACE energies are reference-shifted DFT-surface values. Semiempirical
energies are method-internal. Compare geometries, forces, frequencies,
and relative energies within each tier.

Run:
    python examples/semiempirical/18_comparison_all.py

Requires the [mace] extra for Tier 3 (pip install 'vibe-qc[mace]').
"""

from pathlib import Path

from vibeqc import Molecule, run_job

HERE = Path(__file__).resolve().parent
mol = Molecule.from_xyz(HERE.parent / "h2o.xyz")

# =========================================================================
# Tier 1 — Ab-initio DFT (reference)
# =========================================================================

for method, basis in [
    ("hf", "def2-svp"),
    ("pbe", "def2-svp"),
    ("pbe0", "def2-svp"),
    ("b3lyp", "def2-svp"),
]:
    label = f"{method}-{basis.replace('-', '')}"
    run_job(mol, method=method, basis=basis, output=HERE / f"output-h2o-{label}")
    print(f"DFT {method}/{basis}: {HERE / f'output-h2o-{label}.out'}")

# =========================================================================
# Tier 2 — Semiempirical
# =========================================================================

for method in ("dftb0", "scc_dftb", "gfn2_xtb", "pm6", "om1", "om2", "om3", "msindo"):
    run_job(mol, method=method, output=HERE / f"output-h2o-{method}")
    print(f"SE {method}: {HERE / f'output-h2o-{method}.out'}")

# =========================================================================
# Tier 3 — MLIP (MACE)
# =========================================================================

# Default MIT model (MACE-MPA-0, materials).
run_job(mol, method="mace", output=HERE / "output-h2o-mace-mpa0")
print(f"MLIP mace-mpa0: {HERE / 'output-h2o-mace-mpa0.out'}")

# ASL-gated organic model (requires acknowledgment; uncomment to run):
# from vibeqc.mlip import MLIPOptions
# run_job(
#     mol, method="mace",
#     mlip_options=MLIPOptions(model="off23-medium", accept_academic_license=True),
#     output=HERE / "output-h2o-mace-off23",
# )

print("\nThree-tier comparison complete.")
print("Compare geometries, forces, and vibrational frequencies across tiers.")
print("DO NOT compare absolute energies across tiers — see manuscript §5.1.")
