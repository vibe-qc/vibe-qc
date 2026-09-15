# pob-* basis-set verification report

**Date:** 2026-05-08 (work conducted on `basissetdev`)
**Scope:** verify the three pob-* basis files bundled in vibe-qc
against the upstream Bredow-group sources and the supporting
information of the cited papers.
**Tool:** `scripts/basisset_dev/pob_basis_verify.py`

## Authoritative sources

* **pob-TZVP**, `python/vibeqc/basis_library/sources/pob-TZVP/`, 
  32 per-element CRYSTAL-format files distributed by the Bredow
  group as `pob-tzvp-tar.gz`
  (`https://www.chemie.uni-bonn.de/bredow/de/software/pob-tzvp-tar.gz`,
  retrieved 2026-05-07). Spot-checked element-by-element against
  Section 2 of the supporting information of M. F. Peintinger,
  D. Vilela Oliveira, T. Bredow, *J. Comput. Chem.* **34**, 451
  (2013) [DOI 10.1002/jcc.23153] for H, Li, Be, B, C, N, O, F, Na,
  Mg, Al, Si, P, S, Cl, K, Ca, Sc, Ti, V, all exponents and
  coefficients agree to printed precision.
* **pob-TZVP-rev2**, `sources/pob-TZVP-rev2/`, 32 files from
  `pob-tzvp-rev2-tar.gz`, retrieved 2026-05-07.
* **pob-DZVP-rev2**, `sources/pob-DZVP-rev2/`, 19 per-element files
  scraped from the per-element HTML pages at
  `https://www.chemie.uni-bonn.de/bredow/de/software/pob-dzvp-rev2/<NN_x>`
  (the Bredow group serves dzvp-rev2 as a Plone web folder rather
  than a tarball). Retrieved 2026-05-07.

## Findings

### Bug: sulfur d-polarisation column-swap in shipped pob-TZVP and pob-TZVP-rev2

Both shipped `pob-tzvp.g94` and `pob-tzvp-rev2.g94` had the d-shell
of sulfur in the wrong columns:

```
D    1   1.00
        0.0000000000        0.5207010100   ← bug: exponent zero
```

vs. authoritative source:

```
D    1   1.00
        0.5207010100        1.0000000000   ← correct (pob-TZVP)
        0.4107010100        1.0000000000   ← correct (pob-TZVP-rev2)
```

**Root cause.** The Bredow source files for sulfur in both bases
contain a typo, the d-shell scale `1.0` is mistyped as two
whitespace-separated tokens `1 0`:

```
0 3 1 0.0 1 0
  0.5207010100     1.00000000000000
```

CRYSTAL itself parses this correctly (it's whitespace-tolerant
across newlines). The previous CRYSTAL-to-`.g94` conversion that
seeded vibe-qc's bundled file was *not* tolerant: it took the `0`
on the header line as the start of the next primitive's exponent,
shifting all subsequent fields by one slot. Result: exponent column
ended up holding `0` and the coefficient column held the d-exponent
`0.5207010100`. The actual `1.0000000000000` coefficient was orphaned
on the next line and silently truncated.

**Chemistry impact.** A d-polarisation function with exponent zero
contributes nothing to the orbital expansion. Sulfur calculations
using the previously bundled pob-TZVP or pob-TZVP-rev2 silently ran
with **no d-polarisation on S**, equivalent to running pob-TZ rather
than pob-TZVP for the sulfur centres. Quantitative impact on
sulfur-containing systems (S, B₄C with no S, no, but ZnS, MnS, K₂S,
CoS, CuS, β-ZnS in the paper test set) is yet to be measured, 
expect lattice-constant errors O(0.01-0.05 Å) and total-energy errors
several kJ/mol per S centre, since the d-function is structurally
important on second-row p-block elements.

**Fix.** `pob_basis_verify.py` parser is line-aware and detects the
"`1 0`" typo, reconstructing the scale as `1.0`. `--regenerate`
re-emits the affected `.g94` files. Both `pob-tzvp.g94` and
`pob-tzvp-rev2.g94` now carry the correct d-polarisation on sulfur.

### Precision of all numbers

Beyond the sulfur fix, the main change is preserving full source
precision. The previous shipped files rounded contraction coefficients
to 10 decimal places; the Bredow CRYSTAL files publish 11 dp for
pob-TZVP and pob-TZVP-rev2 and 8 dp for pob-DZVP-rev2. The regenerated
files preserve every digit the source publishes.

### Element coverage

| Basis | Source files | Shipped `.g94` blocks | Match |
|-------|--------------|-----------------------|-------|
| pob-TZVP | 32 (H-Br excluding noble gases) | 32 | ✅ |
| pob-TZVP-rev2 | 32 (same range) | 32 | ✅ |
| pob-DZVP-rev2 | 19 (H-V excluding Si and noble gases) | 19 | ✅ |

## Verification command

```bash
$ python3 scripts/basisset_dev/pob_basis_verify.py --atol 1e-15
=== pob-TZVP ===
  source elements:  32
  in shipped g94:   32
  parity:           CLEAN within tolerance

=== pob-TZVP-rev2 ===
  source elements:  32
  in shipped g94:   32
  parity:           CLEAN within tolerance

=== pob-DZVP-rev2 ===
  source elements:  19
  in shipped g94:   19
  parity:           CLEAN within tolerance
```

All three basis sets are byte-clean against their authoritative
upstream sources to 1e-15 absolute tolerance, i.e. exact agreement
to all printed digits.

## Follow-ups

* Tell Bredow group about the `16_S` "`1 0`" typo so it can be fixed
  upstream (CRYSTAL parses it but other consumers may not).
* Add a regression test that runs `pob_basis_verify.py --atol 1e-15`
  in CI.
* Once sulfur-relevant features land in vibe-qc periodic
  (multi-k + open-shell + hexagonal), regenerate the pob-paper
  benchmark numbers for sulfides (ZnS, MnS, K₂S, CoS, CuS, β-ZnS)
  with the corrected basis and compare to SI Tables 1 / 8 / 9. The
  delta between previously shipped (broken d on S) and corrected
  basis is the chemistry impact of this bug.
