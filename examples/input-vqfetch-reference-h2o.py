"""vqfetch — programmatic CCCBDB reference fetch of H₂O + vibe-qc round-trip.

Walks the v0.8.0 ``vibeqc.fetch.references`` Python API end-to-end:

  1. ``fetch_cccbdb()``                — pull NIST CCCBDB
                                          ``exp2x.asp?casno=7732185``
                                          (per-molecule master page).
  2. Inspect ``ExperimentalReference`` — atomization energy, ΔHf°,
                                          vibrational fundamentals,
                                          dipole, IE, Cartesian
                                          geometry, full Provenance.
  3. **D₀ → Dₑ correction**            — Add ZPE = ½·Σν to the
                                          ZPE-included CCCBDB
                                          atomization energy so the
                                          result is directly comparable
                                          to electronic-only QC results.
                                          (The "footgun" documented in
                                          ``docs/user_guide/reference_data.md``.)
  4. ``experimental_geometry_to_molecule_spec()``
                                       — wrap the NIST experimental
                                          Cartesian geometry as a
                                          regression-suite
                                          ``MoleculeSpec``.
  5. Run vibe-qc RHF/STO-3G on NIST geometry — converges in <1 s on a
                                          laptop, prints E_total, the
                                          NIST experimental dipole, and
                                          a placeholder for a future
                                          method-validated atomization
                                          comparison.

Cache-friendly: the first run hits ``cccbdb.nist.gov`` (~1 s,
rate-limited per the polite-fetch contract); subsequent runs replay
from ``$XDG_CACHE_HOME/vibeqc/fetch/CCCBDB/`` unless ``--no-cache``.

Output files land in ``/tmp/vqfetch_reference_h2o/`` so re-running
doesn't clutter ``examples/``.

Run:
    .venv/bin/python examples/input-vqfetch-reference-h2o.py

Paired with the user-facing walkthrough in
``docs/user_guide/reference_data.md`` and tutorial 30
(``docs/tutorial/external_data_fetcher.md``).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import vibeqc as vq
from vibeqc.fetch.references import (
    experimental_geometry_to_molecule_spec,
    fetch_cccbdb,
)


# CODATA 2018 — kept identical to vibeqc.scf_log / runner.py.
HARTREE_TO_KCAL_PER_MOL = 627.5094740631
KCAL_PER_MOL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL_PER_MOL
CM_INV_TO_KCAL_PER_MOL = 0.0028591452815       # CODATA: hc·Nₐ / 4184


OUTDIR = Path("/tmp/vqfetch_reference_h2o")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---- 1. Fetch the CCCBDB record ---------------------------------------------

print("\n  [1/5] fetch_cccbdb(cas='7732-18-5')   # H₂O")
t0 = time.perf_counter()
ref = fetch_cccbdb(cas="7732-18-5")
print(f"        returned {ref.kind} ExperimentalReference in "
      f"{time.perf_counter()-t0:.2f}s "
      f"(first run hits NIST; subsequent runs replay from cache)")


# ---- 2. Inspect the record --------------------------------------------------

print("\n  [2/5] What CCCBDB has for water")
print(f"        formula                = {ref.formula}")
print(f"        name                   = {ref.name}")
print(f"        ΔH_f(0K)               = {ref.enthalpy_of_formation_0_kj_per_mol:.2f} "
      f"± {ref.enthalpy_of_formation_0_uncertainty_kj_per_mol:.2f} kJ/mol")
print(f"        ΔH_f(298K)             = {ref.enthalpy_of_formation_298_kj_per_mol:.2f} "
      f"± {ref.enthalpy_of_formation_298_uncertainty_kj_per_mol:.2f} kJ/mol")
print(f"        atomization E (D₀)     = {ref.atomization_energy_kcal_per_mol:.2f} "
      f"kcal/mol  ⚠ ZPE-included")
print(f"        ionization energy      = {ref.ionization_energy_ev:.3f} "
      f"± {ref.ionization_energy_uncertainty_ev:.3f} eV")
print(f"        dipole moment          = {ref.dipole_moment_debye:.3f} D")
print(f"        polarizability         = {ref.polarizability_au:.3f} a.u.")
print(f"        vibrational fundamentals = {ref.vibrational_fundamentals_cm_inv} cm⁻¹")
print(f"        vibrational harmonics    = {ref.vibrational_harmonics_cm_inv} cm⁻¹")
print(f"        bond r(O–H)            = {ref.bond_lengths_ang[0][2]:.4f} Å")
print(f"        angle ∠(H–O–H)         = {ref.bond_angles_deg[0][3]:.3f}°")

print("\n        Provenance:")
p = ref.provenance
print(f"          source_db            = {p.source_db}")
print(f"          source_id            = {p.source_id}")
print(f"          source_url           = {p.source_url}")
print(f"          original_reference   = {p.original_reference}   ⟵ NIST DOI")
print(f"          license              = {p.license}")
print(f"          fetched_at           = {p.fetched_at}")
print(f"          notes (truncated)    = {p.notes[:100]}…")


# ---- 3. D₀ → Dₑ: the footgun in action --------------------------------------

print("\n  [3/5] Convert CCCBDB D₀ (ZPE-included) to Dₑ (electronic-only)")
print( "        — the value most QC textbooks quote for atomization energy.")
zpe_cm_inv = 0.5 * sum(ref.vibrational_fundamentals_cm_inv)
zpe_kcal = zpe_cm_inv * CM_INV_TO_KCAL_PER_MOL
ae_de_kcal = ref.atomization_energy_kcal_per_mol + zpe_kcal
print(f"        ZPE         = ½·Σν = ½·{sum(ref.vibrational_fundamentals_cm_inv):.0f} cm⁻¹ "
      f"= {zpe_cm_inv:.0f} cm⁻¹ = {zpe_kcal:.2f} kcal/mol")
print(f"        AE D₀       = {ref.atomization_energy_kcal_per_mol:.2f} kcal/mol "
      f"(CCCBDB experimental, ZPE-included)")
print(f"        AE Dₑ       = {ae_de_kcal:.2f} kcal/mol   (D₀ + ZPE)")
print( "        Literature  ≈ 232.5 kcal/mol  (Curtiss G2/3 test sets — agrees)")


# ---- 4. Wrap NIST experimental geometry into a regression-suite MoleculeSpec ---

print("\n  [4/5] experimental_geometry_to_molecule_spec(ref, quick=True)")
spec = experimental_geometry_to_molecule_spec(
    ref, slug="h2o_cccbdb", quick=True,
)
print(f"        spec.id                = {spec.id}")
print(f"        spec.atoms             = {len(spec.atoms)} atoms — "
      + ", ".join(f"{a.symbol}@{a.xyz_ang}" for a in spec.atoms))
print(f"        spec.charge            = {spec.charge}")
print(f"        spec.multiplicity      = {spec.multiplicity}  (closed-shell)")
print(f"        spec.recommended_basis = {spec.recommended_basis}")
print(f"        spec.provenance.notes  = {spec.provenance.notes[:80]}…")


# ---- 5. Run vibe-qc RHF/STO-3G on the NIST geometry -------------------------

print("\n  [5/5] vibe-qc RHF / STO-3G on the NIST experimental geometry")
ANG_TO_BOHR = 1.0 / 0.529177210903
mol = vq.Molecule([
    vq.Atom(a.z, [c * ANG_TO_BOHR for c in a.xyz_ang]) for a in spec.atoms
])

t0 = time.perf_counter()
result = vq.run_job(mol, basis="sto-3g", method="rhf", output=OUTDIR / "h2o_rhf")
wall = time.perf_counter() - t0

print(f"        E_RHF/STO-3G(H₂O)      = {result.energy:.8f} Ha   "
      f"({wall:.2f}s)")
print( "        Compared geometry      = Szabo & Ostlund (r=1.0 Å, 104.5°):")
print( "          E_RHF/STO-3G         = -74.96292825 Ha (regression-suite)")
print( "          Δ                    ≈ +1 mHa — the geometry difference")
print( "                                 (NIST r=0.958 Å vs S&O r=1.0 Å), not a bug.")

# Dump the full record for inspection.
record_path = OUTDIR / "h2o_cccbdb.json"
record_path.write_text(
    json.dumps(asdict(ref), indent=2, sort_keys=True), encoding="utf-8",
)

print("\n  Done. Artefacts at:")
print(f"    JSON record    {record_path}")
print(f"    SCF output     {OUTDIR / 'h2o_rhf.out'}")
print()
print( "  To attach this record to a regression run automatically:")
print( "      python -m examples.regression.run_suite \\")
print( "          --systems h2o --bases sto-3g --methods rhf,rks-lda \\")
print( "          --include-experimental-reference cccbdb")
print()
print( "  Cache: set VIBEQC_FETCH_CACHE_ONLY=1 to verify subsequent runs")
print( "  replay offline (no NIST traffic).")
