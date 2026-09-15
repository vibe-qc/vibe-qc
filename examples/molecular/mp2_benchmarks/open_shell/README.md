# Open-shell MP2 benchmark complement (UMP2 / SCS-UMP2 / SOS-UMP2)

The S22 dimer set is all closed-shell. This directory adds a small
complement of open-shell systems so the UMP2 / SCS-UMP2 / SOS-UMP2
variants of vibe-qc's MP2 surface get an apples-to-apples comparison
against ORCA at the same converger settings.

The checked-in inputs pin this historical comparison to all-electron
correlation on both sides.  vibe-qc uses `n_frozen_core=0` or
`frozen_core=False`; ORCA uses `NoFrozenCore` plus `FrozenCore FC_NONE`.
The resulting rows are therefore explicit old-convention evidence, not
samples of vibe-qc's current unqualified frozen-core default.

## Systems

| Tag           | System                  | Multiplicity | Atoms | Why                                                       |
|---------------|-------------------------|--------------|-------|-----------------------------------------------------------|
| OH-radical    | OH radical, ⊥e geom.    | 2 (doublet)  | 2     | Smallest open-shell test; <S²> = 0.75 exactly             |
| O2-triplet    | O₂, exp. r=1.2075 Å     | 3 (triplet)  | 2     | <S²> = 2.0; spin-polarised diatomic                       |
| CH3-radical   | Methyl radical, exp.    | 2 (doublet)  | 4     | Sp² centre, planar, π-density on C                        |
| OH-H2O        | OH•...H₂O H-bonded      | 2 (doublet)  | 5     | Open-shell intermolecular; pair-correlation on radical    |

All use cc-pVTZ orbital basis + cc-pVTZ-RI MP2 aux, matching the S22
benchmark side. Same converger profile (plain DIIS, conv_tol_energy
1e-8, max_iter 200) — see `../s22/converger.py`.

## Variants

| Variant   | vibe-qc                                | ORCA                                |
|-----------|----------------------------------------|-------------------------------------|
| ump2      | `run_ump2(opts.density_fit=False)`     | `! MP2`                             |
| riump2    | `run_ump2(opts.density_fit=True, ...)` | `! RI-MP2 cc-pVTZ/C`                |
| scsump2   | `run_scs_ump2(...)`                    | `! SCS-MP2 cc-pVTZ/C`               |
| sosump2   | `run_sos_ump2(...)`                    | `! SOS-MP2 cc-pVTZ/C`               |

B2PLYP on open-shell radicals goes through `run_b2plyp` with a UKS
SCF reference; we omit it from this complement to keep the variant
set focused on UMP2 itself. Coverage of B2PLYP-on-radicals lives in
the existing `tests/test_scs_mp2.py::test_scs_ump2_*` PySCF parity
suite.

## Layout

Mirrors `../s22/`:

    geometries/
      oh-radical.xyz
      o2-triplet.xyz
      ch3-radical.xyz
      oh-h2o.xyz
    inputs/
      vibeqc/<tag>/<variant>.py
      orca/<tag>/<variant>.inp

Running the drivers writes `.log`, `.result`, and ORCA `.out` files beside the
inputs.  The 16 checked-in vibe-qc `.result` files are retained unchanged as
old all-electron numerical evidence; newly run payloads also disclose
`n_frozen_core: 0`.  New logs and ORCA outputs remain local run records.
