# Authoritative basis-set sources

Per-element CRYSTAL-format basis files distributed by the Bredow group
at the Mulliken Center for Theoretical Chemistry, University of Bonn.
These are the **canonical source** for the bundled `pob-*.g94` files.

| Directory | Source URL | Retrieved |
|-----------|------------|-----------|
| `pob-TZVP/` | <https://www.chemie.uni-bonn.de/bredow/de/software/pob-tzvp-tar.gz> (H-Br); `pob-tzvp-rb-i-tar.gz` (`pob-TZVP-Rb-I.tar.gz`, 7280 bytes, sha256 `2b341b8f…d434bf194e`: the 16 ECP-bearing `Z+200` records, Laun 2018) | 2026-05-07 (H-Br), 2026-09-13 (Rb-I) |
| `pob-TZVP-rev2/` | <https://www.chemie.uni-bonn.de/bredow/de/software/pob-tzvp-rev2-tar.gz> (H-Br); `pob-tzvp-ref2-rb-i-tar.gz`, `pob-tzvp-rev2-cs-po-tar.gz`, `pob-tzv-rev2-la-lu-tar.gz` (the 46 ECP-bearing `Z+200` records) | 2026-05-07 (H-Br), 2026-09-05 (Rb-Po, La-Lu) |
| `pob-DZVP-rev2/` | <https://www.chemie.uni-bonn.de/bredow/de/software/pob-dzvp-rev2/{NN_x}> (per-element HTML pages, scraped) | 2026-05-07 |

The pob-TZVP source files have been spot-checked against the basis-set
listings in **Section 2** of the supporting information of
*J. Comput. Chem.* **34**, 451 (2013) [DOI 10.1002/jcc.23153] for H,
Li, Be, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, Sc, Ti, V —
exponents and contraction coefficients agree to all printed digits.

## CRYSTAL format quick reference

Each per-element file:

```
Z    N_SHELLS
ITYP L NPRIM CHARGE SCALE        ← shell header
exponent_1 coefficient_1
exponent_2 coefficient_2
...
ITYP L NPRIM CHARGE SCALE        ← next shell header
...
```

where `L`: 0=s, 1=sp, 2=p, 3=d, 4=f. `ITYP=0` denotes all-electron
(no ECP). SP shells (`L=1`) take two coefficients per primitive (s and p
shares of a shared exponent).

The heavy pob-TZVP-rev2 records (Rb-Po, La-Lu) carry a `200+Z` header and
an `INPUT` ECP block before the shells; `vibeqc.basis_crystal.
parse_crystal_atom_basis` reads both, the periodic runner converts the
ECP through `crystal_ecp_to_libecpint_arrays`, and
`basis/pob-tzvp-rev2.ecp` is the same data emitted as a Gaussian-style
sidecar for the molecular route. Bundling these records means ECP
resolution never touches the network.

## Regenerating the .g94 files

```bash
python3 scripts/basisset_dev/pob_basis_verify.py --regenerate
```

emits `basis/<basis>.g94` and `custom/<basis>.g94` at full source
precision and runs a parity check against the existing files. Run
without `--regenerate` to verify only.

## Known upstream typo

The Bredow `16_S` files (both `pob-TZVP/` and `pob-TZVP-rev2/`) carry
the d-shell scale `1.0` mistyped as two tokens `1 0`:

```
0 3 1 0.0 1 0
  0.5207010100     1.00000000000000
```

CRYSTAL itself is whitespace-tolerant and parses this correctly. The
parser in `pob_basis_verify.py` matches that tolerance. The previously
shipped `pob-{tzvp,tzvp-rev2}.g94` files did **not** — they suffered
a column-shift bug at this position, putting the d-exponent into the
coefficient column and zero into the exponent column. That bug is now
fixed by regeneration; see `docs/basisset_dev/VERIFICATION_REPORT.md`
for details.
