# vibe-qc bundled basis library

Basis sets shipped with vibe-qc as Gaussian `.g94` text files — the same
format [libint](https://github.com/evaleev/libint) parses natively.

This directory lives **inside the Python package** so it ships in every
wheel and source distribution. A fresh `pip install -e .` from the
vibe-qc checkout already has every standard libint basis plus vibe-qc's
custom additions; users do **not** need to set `LIBINT_DATA_PATH` or
run any separate setup step.

## Layout

```
python/vibeqc/basis_library/
├── basis/         ← assembled and committed: every basis set ships here
│                    (<name>.g94 orbital blocks + <name>.ecp ECP sidecars)
├── custom/        ← source of truth for vibe-qc's custom additions,
│                    orbital .g94 files and pre-split .ecp sidecars alike
├── sources/       ← per-element CRYSTAL records the pob-* files derive from
├── qvf/           ← QVF-Basis structured sidecars (.qvf.json), orbital + ECP
├── registry.toml  ← curated metadata: default fitting sets, ECP-requirement
│                    and all-electron family rules (read by vibeqc.basis_registry)
└── README.md
```

- **`basis/`** is what libint reads at runtime. `vibeqc/__init__.py`
  points `LIBINT_DATA_PATH` at the parent `basis_library/` directory at
  import time, so `BasisSet(mol, "<name>")` resolves any `.g94` file
  in `basis/`. The contents are:
  - libint's standard set (sto-3g, 6-31g\*, cc-pVxZ, def2-\*, ...)
  - vibe-qc's custom additions (POB crystal bases, BSE-sourced additions,
    and release-paper exact reproduction bases)
- **`custom/`** is the source of truth for the custom overlay. Edit files
  here, then re-run `scripts/setup_basis_library.sh` to refresh `basis/`.
  (The standard set in `basis/` is overwritten with libint's current bundle;
  custom files are layered on top so they win on a name conflict.) A
  pre-split `custom/<name>.ecp` sidecar is copied into `basis/` verbatim;
  this is the only way an ECP survives regeneration for a basis whose
  orbital file carries no embedded ECP block (vDZP, def2-mSVP, def2-mTZVP,
  def2-mTZVPP, pob-TZVP-rev2).
- **`<name>.ecp` is the ECP for `<name>.g94`, per element.** The molecular
  SCF wrappers read it and attach its primitives inline whenever an atom of
  the molecule has a block there (`vibeqc.ecp_metadata.
  attach_inline_ecp_options_from_basis_sidecar`); an atom without a block
  runs all-electron in that basis. `vibeqc.basis_registry.ecp_requirements`
  is the per-element question every guard asks.
- **`registry.toml`** names the default JK / RI fitting set per orbital
  basis (one table for the molecular and the periodic callers), the
  valence-only families that need an ECP beyond a given element, and the
  all-electron relativistic families. Add a row when you add a basis that
  has a published fit or that is valence-only anywhere.

## End-user perspective

```python
import vibeqc as vq
mol = vq.Molecule([vq.Atom(8, [0, 0, 0]),
                   vq.Atom(1, [0, 1.43, -0.98]),
                   vq.Atom(1, [0, -1.43, -0.98])])

# All of these "just work" out of a vibe-qc editable install:
vq.BasisSet(mol, "sto-3g")        # libint standard
vq.BasisSet(mol, "6-31g*")        # libint standard
vq.BasisSet(mol, "cc-pvtz")       # libint standard
vq.BasisSet(mol, "pob-tzvp")      # vibe-qc custom (CRYSTAL pob set)
vq.BasisSet(mol, "6-311+g3df2p")  # exact PBE 1996 reproduction basis
```

The user override (`LIBINT_DATA_PATH=/path/to/your/own/library`) still
wins if you set it explicitly; vibe-qc only sets the variable when the
user hasn't.

## Adding a new basis (developers)

1. Format the basis as a Gaussian `.g94` text file. If you have it in
   another format (CRYSTAL, ORCA, Turbomole, PySCF, JSON from
   [BSE](https://www.basissetexchange.org)), convert to `.g94` — BSE
   exports Gaussian format directly.
2. Save as `python/vibeqc/basis_library/custom/<name>.g94`, where
   `<name>` is the case-folded identifier users will pass to
   `BasisSet(mol, name)` (libint normalises to lowercase internally).
3. Re-run `./scripts/setup_basis_library.sh` so the file lands in
   `basis/` (overwriting any libint standard with the same name).
4. Commit both `custom/<name>.g94` *and* `basis/<name>.g94` so the
   wheel ships the new file.
5. Add an entry to the inventory below documenting source, revision,
   and citation.

## Inventory

### vibe-qc custom sets (committed in `custom/`)

| File | Purpose | Source | Citation |
|------|---------|--------|----------|
| `pob-tzvp.g94` | Triple-ζ valence + polarisation, optimised for periodic crystals | `sources/pob-TZVP/` (Bredow group archive) | M. F. Peintinger, D. Vilela Oliveira, T. Bredow, *J. Comput. Chem.* **34**, 451 (2013), DOI 10.1002/jcc.23153 |
| `pob-tzvp-rev2.g94` | Triple-ζ valence + polarisation, revision 2 (BSSE-aware re-optimisation) | `sources/pob-TZVP-rev2/` (Bredow group archive) | D. Vilela Oliveira, J. Laun, M. F. Peintinger, T. Bredow, *J. Comput. Chem.* **40**, 2364 (2019), DOI 10.1002/jcc.26013 |
| `pob-dzvp-rev2.g94` | Double-ζ valence + polarisation, revision 2 (H–V) | `sources/pob-DZVP-rev2/` (Bredow group archive) | D. Vilela Oliveira, J. Laun, M. F. Peintinger, T. Bredow, *J. Comput. Chem.* **40**, 2364 (2019), DOI 10.1002/jcc.26013 |

The `.g94` files in `custom/` and `basis/` are auto-generated by
`scripts/basisset_dev/pob_basis_verify.py --regenerate` from the
authoritative per-element CRYSTAL-format sources in `sources/`. Edit
the source files, never the `.g94` directly. The verifier doubles as
parity-check tool; see `docs/basisset_dev/VERIFICATION_REPORT.md` for
the audit history.

### BSE-derived custom sets

| File | Purpose | Source | Citation |
|------|---------|--------|----------|
| `6-311+g3df2p.g94` | Exact `6-311+G(3df,2p)` basis for the PBE 1996 reproduction benchmark | Constructed from BSE 0.12 Pople component bases: H-He from `6-311G(2df,2pd)` with D shells removed; Li-Ar from `6-311++G(3df,3pd)` heavy-atom blocks | Krishnan/Binkley/Seeger/Pople 1980; McLean/Chandler 1980; Hariharan/Pople 1973; Frisch/Pople/Binkley 1984 |

### ECP sidecars shipped for valence-only custom bases

| Sidecar | Elements | Source | Notes |
|------|---------|--------|-------|
| `vdzp.ecp` | B-F, Al-Ar, Ga-Kr, Rb-Rn (82 blocks) | BSE 0.12 vDZP record, split by `split_ecp_g94.py` | Müller, Hansen, Grimme 2023 (ωB97X-3c); custom cores, inline only |
| `def2-msvp.ecp` | Rb-Rn (36 blocks) | Psi4 / BSE def2-mSVP record | The def2-ECP set; PBEh-3c, HSE-3c, B3LYP-3c |
| `def2-mtzvp.ecp`, `def2-mtzvpp.ecp` | Rb-Rn (36 blocks) | Same def2-ECP data as `def2-msvp.ecp` | B97-3c and r²SCAN-3c pair with the def2-ECP beyond Kr like the parent def2 sets; the BSE records ship without the ECP blocks |
| `pob-tzvp-rev2.ecp` | Rb-I, Cs-Po, La-Lu (46 blocks) | `sources/pob-TZVP-rev2/` CRYSTAL Z+200 records via `vibeqc.basis_crystal.emit_ecp_sidecar` | One ECP for molecular and periodic runs; round-trips to the arrays the periodic bridge feeds libecpint |
| `def2-{sv(p),svp,tzvp,tzvpp,qzvp,qzvpp}.ecp` | Rb-Rn (50 blocks) | BSE def2 records, split by `scripts/basisset_dev/merge_def2_heavy_blocks.py` | The def2-ECP; the matching `custom/def2-*.g94` are libint's H-Kr files with the BSE Rb-Rn valence blocks appended |
| `def2-{svpd,tzvpd,tzvppd,qzvpd,qzvppd}.ecp` | Rb-La, Hf-Rn (36 blocks) | same | The diffuse sets carry no lanthanide block |

### QVF-Basis sidecars

The `qvf/` directory contains structured JSON sidecars for every basis
set, generated from the `.g94` files by `scripts/convert_basis_library_to_qvf.py`.
These are schema-validated, diff-friendly, and carry metadata (name, role,
family, provenance) that G94 comments can't encode.  Regenerate after any
change to the `.g94` files:

```bash
.venv/bin/python scripts/convert_basis_library_to_qvf.py --write
.venv/bin/python -m vibeqc.basis_toolkit.cli validate python/vibeqc/basis_library/qvf/
```

The `.g94` files remain the runtime source for libint; the `.qvf.json` files
are the canonical structured representation.

### Standard sets (assembled from libint 2.13.1)

`sto-3g`, `3-21g`, `6-31g`, `6-31g*`, `6-31g**`, `6-311g`, `6-311g*`,
`6-311g**`, `cc-pvdz`, `cc-pvtz`, `cc-pvqz`, `cc-pv5z`, `aug-cc-pv{d,t,q,5}z`,
`def2-{sv,svp,svpd,tzv,tzvp,tzvpd,tzvpp,qzvp,qzvpp}`, JKFIT / CABS / RI
variants, ANO-RCC, and others. Full list: `ls basis/`.

## Licensing + attribution

The standard sets in `basis/` are inherited from libint 2.13.1's
upstream distribution and ship under the same redistribution
terms libint applies to its bundled basis-set data (libint
itself is LGPL 3.0 for the library). The POB custom sets in
`custom/` are originated by the Bredow group at the University
of Bonn and Mike Peintinger; no external clearance required.
BSE-derived custom files carry their source and originating
publication references in the file header.

**For published work that uses any of these basis sets**, cite
the originating publication for the specific basis set you
used — the per-publication references are preserved in each
`.g94` file's header.

The full inventory, per-family citation hints, and redistribution
notes for the 239 bundled `.g94` runtime basis files are documented in
[`docs/license.md`](https://vibe-qc.com/docs/license.html#bundled-basis-sets).
