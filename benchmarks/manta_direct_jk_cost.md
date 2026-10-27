# Matched MANTA versus libint direct J/K cost

Development tracker #638, first increment. `manta_direct_jk_cost.py` runs one
selected RHF with exact J and K per fixture on the libint reference and on
MANTA with identical options (HCORE guess, 1e-10/1e-8, single thread) and
records wall time, the engine receipt's shell calls and primitive integrals,
and the energy. Both backends digest the same shell-quartet schedule; the
receipts agree on every shell-call count and the energies agree to 3e-12 Ha.
Run it from a development environment on an otherwise idle machine; background
indexing or scanning distorts both columns.

## 2026-10-24, arm64 laptop, single thread

Baseline at main `5e9b3702`, then after the shell-quartet reuse change in
`cpp/src/manta_shells.cpp` and `cpp/src/manta_md.cpp`: every MANTA tile of
H2O/6-31G, HBr/STO-3G and H2O/def2-SVP (33,858 shell quartets) is bit-identical
before and after, and the primitive-integral counts are unchanged.

| fixture | nbf | libint s | MANTA before s | ratio before | MANTA after s | ratio after |
| --- | --- | --- | --- | --- | --- | --- |
| H2O/STO-3G | 7 | 0.01 | 0.25 | 21 | 0.20 (loaded run) | - |
| H2O/6-31G | 13 | 0.08 | 1.64 | 21 | 1.23 (loaded run) | - |
| HBr/STO-3G | 19 | 0.26 | 5.66 | 22 | 2.51 | 9.5 |
| ZnH+/STO-3G | 19 | 0.61 | 13.26 | 22 | 5.81 | 9.7 |

The two light fixtures were re-measured only while another process loaded the
machine (their libint column doubled in that run), so they carry no after
ratio; the heavy fixtures were re-measured on a quiet machine.

## What was measured and changed

A sampled profile of the baseline ZnH+ run spent about 40 % of its samples in
malloc/free, 16 % rebuilding primitive-pair expansions (index tables, the three
axis recurrences and the E matrix) and 8 % in memset/memmove, against about
10 % in the Boys function, the Hermite kernel and the two E-R-E products. The
driver rebuilt the ket pair expansion inside the bra primitive loop and
allocated the auxiliary seeds and cache plus three matrices per primitive
quartet. The change builds each shell quartet's pair expansions once, serves
the Cartesian and Hermite index tables from shared read-only tables, and reuses
one scratch for the auxiliary seeds/cache and the kernel/intermediate products.
The arithmetic and its accumulation order are unchanged.

After the change the profile is dominated by the Hermite R-tensor recursion and
the kernel assembly inside `coulomb_into`, with the pair build and the Boys
function next; malloc/free is below 5 %. The remaining factor of about 9.5 to
the libint reference is kernel arithmetic, not heap traffic, and is the next
#638 increment. Screening is not involved: the provenance records
`screening: none` on both backends by design.
