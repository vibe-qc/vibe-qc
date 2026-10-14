# vibe-qc bundled basis library

Every basis set vibe-qc ships is one **QVF-Basis record**,
`qvf/<name>.qvf.json`: its shells, its effective core potentials, its
originating publications and its provenance in one schema-validated JSON
file. The `.g94` and `.ecp` text files libint and libecpint read are
**renderings** of those records, written by
`vibeqc.basis_toolkit.library` and pinned byte for byte to the records
by `tests/test_basis_qvf_parity.py`. Edit a record and re-render; never
edit the text.

This directory lives inside the Python package so it ships in every
wheel and source distribution. A fresh `pip install -e .` has every
record immediately; users do not set `LIBINT_DATA_PATH` or run a setup
step.

## Layout

```
python/vibeqc/basis_library/
├── qvf/           ← the records: one <name>.qvf.json per basis set (source of truth)
├── basis/         ← rendered for libint and libecpint: <name>.g94 shells, <name>.ecp potentials
├── sources/       ← per-element CRYSTAL records the pob-* sets derive from (archive of record)
├── registry.toml  ← curated metadata: default fitting sets, ECP-requirement and
│                    all-electron family rules (read by vibeqc.basis_registry)
└── README.md
```

- **`qvf/`** is what you change. A record carries `elements[Sym].shells`,
  `ecps[Sym]` (`n_core_electrons`, `angular_momentum_max`, `potentials`
  with Gaussian-format powers, see #207), `references` with DOIs, and a
  `provenance` block (schema 1.1.0): `kind` (`vendor`, `bse`,
  `hand-written`, `vibe-basis`), the `upstream` sources, the
  `retrieval_date`, `content_sha256` of the numerical content, and
  `citations` routed per element range (#288).
- **`basis/`** is what libint reads at runtime. `vibeqc/__init__.py`
  points `LIBINT_DATA_PATH` at a verified rendering of this library (the
  `build/basis_library/` overlay when `scripts/setup_basis_library.sh`
  made one, else this committed copy), so `BasisSet(mol, "<name>")`
  resolves any record. `<name>.ecp` is the ECP for `<name>.g94`, per
  element: the molecular SCF wrappers attach its primitives inline for
  every atom with a block there, and `vibeqc.basis_registry.ecp_requirements`
  is the per-element question every guard asks.
- **`sources/`** stays untouched: it is the archive the independent pob
  checks read (#228), and `scripts/basisset_dev/pob_basis_verify.py`
  compares it with the records.
- **`registry.toml`** names the default JK / RI fitting set per orbital
  basis, the valence-only families that need an ECP beyond a given
  element, and the all-electron relativistic families.

## End-user perspective

```python
import vibeqc as vq
mol = vq.Molecule([vq.Atom(8, [0, 0, 0]),
                   vq.Atom(1, [0, 1.43, -0.98]),
                   vq.Atom(1, [0, -1.43, -0.98])])

vq.BasisSet(mol, "sto-3g")        # libint standard
vq.BasisSet(mol, "6-31g*")        # libint standard
vq.BasisSet(mol, "cc-pvtz")       # libint standard
vq.BasisSet(mol, "pob-tzvp")      # Bredow group crystal basis
vq.BasisSet(mol, "6-311+g3df2p")  # exact PBE 1996 reproduction basis
```

A user-set `LIBINT_DATA_PATH` still wins; vibe-qc only sets the variable
when the user has not. Your own records live in the user store
(`vibe-qc-basis import`, `vibeqc.basis_store`), never here, and resolve by
name after the bundle. The five-layer resolution order is documented in
[the basis-set guide](https://vibe-qc.com/docs/user_guide/basis_sets.html).

## Changing or adding a bundled basis (developers)

1. Produce the record. `vibe-qc-basis convert <input> -o qvf/<name>.qvf.json`
   accepts BSE JSON, Gaussian94 (with or without ECP blocks), ORCA, NWChem
   and CRYSTAL per-element files; `<name>` is the lower-case identifier
   users pass to `BasisSet`. Fill `references` and `provenance` (kind,
   upstream, retrieval date, citation routes) before committing: the
   parity test refuses a record without them.
2. Render: `./scripts/setup_basis_library.sh --target-dir python/vibeqc/basis_library`
   (or `python scripts/render_basis_library.py --target python/vibeqc/basis_library/basis`).
3. Route the citation in `python/vibeqc/output/citations/database.toml`
   (`[routes.basis_sets]` must equal the record's citation keys) and record
   the component in `docs/license.md`.
4. Commit the record and its rendering together. `python
   scripts/render_basis_library.py --check` says whether they agree.

The runtime overlay under `build/basis_library/` is rendered from the same
records by `scripts/setup_basis_library.sh` and carries a content receipt;
an overlay that disagrees with the committed rendering refuses to load
(#409). The import tool for hand-written and vibe-basis records writes to
the user's own store, never here (#410).

## Inventory

### Bredow group pob-* sets

| Record | Purpose | Source | Citation |
|------|---------|--------|----------|
| `pob-tzvp` (ECP on Rb-I) | Triple-zeta valence + polarisation, optimised for periodic crystals | `sources/pob-TZVP/` (Bredow group archives, H-Br and Rb-I) | M. F. Peintinger, D. Vilela Oliveira, T. Bredow, *J. Comput. Chem.* **34**, 451 (2013), DOI 10.1002/jcc.23153; Rb-I: J. Laun, D. Vilela Oliveira, T. Bredow, *J. Comput. Chem.* **39**, 1285 (2018), DOI 10.1002/jcc.25195 |
| `pob-tzvp-rev2` (ECP on Rb-Po, La-Lu) | Revision 2 (BSSE-aware re-optimisation) | `sources/pob-TZVP-rev2/` (Bredow group archive) | 2013 parent; H-Br: D. Vilela Oliveira, J. Laun, M. F. Peintinger, T. Bredow, *J. Comput. Chem.* **40**, 2364 (2019), DOI 10.1002/jcc.26013; Rb-I: Laun, Bredow 2022, DOI 10.1002/jcc.26839; Cs-Ba, Hf-Po: Laun, Bredow 2021, DOI 10.1002/jcc.26521; La-Lu: Seidler, Laun, Bredow 2023, DOI 10.1002/jcc.27097 |
| `pob-dzvp-rev2` | Double-zeta valence + polarisation, revision 2 (H-Br) | `sources/pob-DZVP-rev2/` plus BSE for Si, Cr-Br | D. Vilela Oliveira, J. Laun, M. F. Peintinger, T. Bredow, *J. Comput. Chem.* **40**, 2364 (2019), DOI 10.1002/jcc.26013 |

The pob records derive from the per-element CRYSTAL sources in
`sources/`; the ECP radial powers are converted from CRYSTAL's literal
power of r to libecpint's Gaussian convention by
`vibeqc.basis_crystal.crystal_ecp_to_libecpint_arrays` (#207), and the
verifier `scripts/basisset_dev/pob_basis_verify.py` compares both sides.
See `docs/basisset_dev/VERIFICATION_REPORT.md` for the audit history.

### libint's standard set, completed where libint's files are short

The `vendor` records whose upstream is libint 2.13.1 carry libint's own
element data. A few are libint files plus reviewed Basis Set Exchange
blocks for elements libint omits because a later publication added them:
`def2-*` beyond Kr (with the def2-ECP as the sidecar), `cc-pvqz` (Ca),
`cc-pv6z` (Be) and `sto-3g` (Xe). Each record's `provenance.upstream`
names both sources and which elements came from which.

### Basis Set Exchange records

The `bse` records (the PP correlation-consistent family, vDZP, the 3c
composite bases, LANL, dhf-*, the Jensen and ANO families, and the rest)
are whole catalogue records; `provenance.upstream` carries the catalogue
version and the BSE name, and the `references` the publications the
catalogue lists.

### ECP sidecars

Every record with `ecps` renders a sidecar: vDZP (B-F, Al-Ar, Ga-Kr,
Rb-Rn; custom cores, inline only), the def2 family and the def2-m*
composite bases beyond Kr (the def2-ECP), the LANL family, dhf-*, the
Peterson/Figgen PP sets, and pob-TZVP(-rev2). The sidecar's channels are
the arrays libecpint receives; the molecular and the periodic route read
the same record.

## Licensing and attribution

Records inherited from libint 2.13.1 ship under the terms libint applies
to its bundled basis-set data (libint itself is LGPL 3.0). The pob sets
are bundled with the Bredow group's agreement. Basis Set Exchange records
are numerical data from published papers distributed by BSE under
CC-BY-4.0. Every record names its originating publications in
`references`, the rendered headers repeat them, and a run's `.references`
file cites what the record's citation routes say.

**For published work that uses any of these basis sets**, cite the
originating publications of the basis set you used. The full inventory
and redistribution notes are in
[`docs/license.md`](https://vibe-qc.com/docs/license.html#bundled-basis-sets).
