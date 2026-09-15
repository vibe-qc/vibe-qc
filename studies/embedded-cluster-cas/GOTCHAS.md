# MEMORY.md: cross-cutting gotchas for embedded-cluster CAS

Embedded-cluster CASSCF/CASPT2 workstream (MR6-MR10).
Keep this current; the handover carries milestone progress, this carries gotchas.

---

## OpenMolcas parity basis gotcha

**Mg/O OpenMolcas parity MUST use 6-31G, never STO-3G.**

vibe-qc and OpenMolcas carry different STO-3G parameters for Mg/O.
STO-3G absolute |delta| ~1.1e-6 Ha -- basis mismatch, not embedding bug.
6-31G gives ~6e-8 Ha absolute floor and ~4e-9 Ha Madelung-stabilization dE.

Source: MR6c (parity_mr6c.py), MR10 (test_embed_parity.py).

---

## AIMP licensing (LGPL 2.1 vs MPL 2.0)

NEVER bundle AIMP data.  LGPL 2.1 + MPL 2.0 = incompatible for bundling.
Parser reads from user's OpenMolcas install at runtime.  ECP XML in temp dir.

Source: CLAUDE.md s1, HANDOVER_PERIODIC_MULTIREF.md s8.

---

## External programs are oracles only (CLAUDE.md s10)

OpenMolcas/PySCF: out-of-process, never imported.  NON-NEGOTIABLE.

---

## Embedded energy conventions

- Hcore = T + V_nuc(QM) + V_ext(pc) + V_aimp (if present)
- E_nuc = E_nuc(QM-QM) + sum Z_A q_B / R_AB
- ext-ext self-energy EXCLUDED (matches OpenMolcas XField)
- V_ext via compute_nuclear_with_charges
- AIMP ECP: coefficients negated (A(AIMP) = -Zeff * A(ECP))

---

## Probe ball sensitivity

MR6c used probe_radius = 0.60*a.  Driver defaults to 1.2*qm_radius.
Override probe_factor=0.60/0.51 for exact MR6c reproduction.

---

## Ionic cluster carve: odd-Z Molecule bug

Fixed: multiplicity=2 when n_elec odd.  Applies to _carve.py.

---

## Database.toml auto-generation

download-doi.txt is AUTO-GENERATED.  Manual appends are ephemeral.
Use [entries.*] + routes in database.toml.
