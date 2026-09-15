# S22 MP2 benchmark — closed-shell dimer comparison

vibe-qc ↔ ORCA at matched converger settings (plain DIIS, SCFCONV8)
across every closed-shell MP2 variant vibe-qc exposes.

The checked-in inputs preserve the suite's historical **all-electron**
comparison explicitly: vibe-qc uses `n_frozen_core=0` or
`frozen_core=False`, while ORCA uses `NoFrozenCore` plus
`FrozenCore FC_NONE`.  These rows remain old-convention evidence after the
#140 default change; regenerating them does not reinterpret their energies as
published-default frozen-core results.

See the parent [`../README.md`](../README.md) for methodology, scope,
and reference-energy citations. This README only covers things that
are S22-specific.

## The S22 set in one table

| # | Subset | System                              | Atoms | Cost note          |
|---|--------|--------------------------------------|------:|--------------------|
| 01| HB     | Ammonia dimer                       | 8     | small              |
| 02| HB     | Water dimer                         | 6     | smallest           |
| 03| HB     | Formic acid dimer                   | 10    | small              |
| 04| HB     | Formamide dimer                     | 12    | small              |
| 05| HB     | Uracil dimer HB                     | 24    | RI-MP2 only        |
| 06| HB     | 2-Pyridone-2-aminopyridine          | 25    | RI-MP2 only        |
| 07| HB     | Adenine-thymine WC                  | 30    | RI-MP2 only; biggest|
| 08| DD     | Methane dimer                       | 10    | small              |
| 09| DD     | Ethene dimer                        | 12    | small              |
| 10| DD     | Benzene-CH₄                         | 17    | RI-MP2 dimer only  |
| 11| DD     | Benzene dimer stack                 | 24    | RI-MP2 only        |
| 12| DD     | Pyrazine dimer                      | 20    | RI-MP2 only        |
| 13| DD     | Uracil dimer stack                  | 24    | RI-MP2 only        |
| 14| DD     | Indole-benzene stack                | 28    | RI-MP2 only        |
| 15| DD     | Adenine-thymine stack               | 30    | RI-MP2 only        |
| 16| MX     | Ethene-ethyne                       | 10    | small              |
| 17| MX     | Benzene-H₂O                         | 15    | small              |
| 18| MX     | Benzene-NH₃                         | 16    | RI-MP2 dimer only  |
| 19| MX     | Benzene-HCN                         | 15    | small              |
| 20| MX     | Benzene dimer T-shape               | 24    | RI-MP2 only        |
| 21| MX     | Indole-benzene T-shape              | 28    | RI-MP2 only        |
| 22| MX     | Phenol dimer                        | 26    | RI-MP2 only        |

"RI-MP2 only" means canonical MP2 (the O(N⁵) 4-index path) is skipped
on that body — the test still exercises RI-MP2 + SCS-RI-MP2 +
SOS-RI-MP2 + B2PLYP, which are the production paths anyway.

## What the comparator reports

After `python ../compare.py s22` walks your local outputs, the local
`REPORT.md` it writes has three sections:

1. **Per-system table** — (system, body, variant) ΔE_corr in Ha and
   kcal/mol, plus SCF iter counts both sides. The kcal/mol column
   makes "is this chemically significant?" visible at a glance.
2. **Interaction energies** — without counterpoise: E_int =
   E_total[dimer] − E_total[monoA] − E_total[monoB]. Compared
   side-by-side to the S22B Marshall 2011 CCSD(T)/CBS reference.
   Without CP correction these will be over-bound (BSSE-affected) —
   document but don't conflate with "vibe-qc bug".
3. **Aggregate ΔE_corr per variant** — max/mean/median |Δ|.
   Acceptance bar: <1e-7 Eh for RI-MP2 / SCS-MP2 / SOS-MP2, <1e-5 Eh
   for B2PLYP (XC-grid sensitivity).
