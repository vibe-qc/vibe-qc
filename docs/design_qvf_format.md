# Design - QVF container format for visualization data

> **Status**: QVF v1 is implemented with the current v1.2 feature set:
> the writer emits 40 canonical section kinds, and vibe-view renders 39
> first-class section kinds plus explicit `bonds` via the structure renderer.
> The canonical machine-readable contract is
> [`python/vibeqc/output/formats/qvf_manifest.schema.json`](../python/vibeqc/output/formats/qvf_manifest.schema.json).
> This document is the human tech spec for that schema and for the
> current vibe-qc writer / vibe-view reader contract.

QVF (`.qvf`, "Quantum Visualization Format") is a ZIP-based container
for quantum-chemistry visualization and analysis data. It is designed
to keep one calculation's structure, scalar fields, spectra, bands,
trajectories, provenance, and viewer hints in one random-access file
instead of scattering them across Cube, XYZ, Molden, XSF, log, and
program-specific sidecar files.

The format is intended for the whole quantum-chemistry ecosystem, not
only vibe-qc. A QVF producer may be any quantum-chemistry code. A QVF
consumer may be a structure viewer, a band plotter, a spectrum tool, a
validator, a notebook script, or the reference GPU viewer `vibe-view`.

## 0. Changes Since The May 2026 Blog Post

The May 21, 2026 blog post
["Quantum chemistry needs a modern file format"](https://vibe-qc.com/2026/05/21/qc-needs-a-modern-file-format-qvf/)
described the motivation and early direction. The implementation has
moved since then:

| Area | Blog-era statement | Current QVF v1 state |
|---|---|---|
| Reference viewer | A viewer was on the roadmap. | The separate vibe-view project renders all current v1 section kinds (`bonds` is folded into the structure renderer). |
| Writer | QVF writer was described as part of the output plan. | `write_qvf()` and `qvf_bytes()` are implemented in `python/vibeqc/output/formats/qvf.py`. |
| Validator | `qvf-validate` was part of the commitment. | `validate_qvf()` validates schema, refs, checksums, JSON parseability, binary byte size, duplicate ids, and selected cross-references. |
| Wavefunctions | Wavefunction coefficient matrices were originally scoped out. | `wavefunction.gto` section carries GTO basis + MO coefficients for molecular systems **and the Γ-point (k=0) of periodic systems**. Full k-resolved Bloch wavefunctions remain out of the visualization contract; vibe-qc may additionally write `x_vibeqc.bloch_wavefunction` as a restart-only vendor section. |
| Provenance | The blog used "provenance" broadly. | `source` is mandatory; `provenance`, `citations`, `scf_history`, `thermochemistry`, `dipole_moment`, and `constraints` are optional. |
| Extensibility | The blog sketched semver-by-kind. | **Closed in v1.x.** Formal `extensions` block, per-section `critical` flag, optional `schema_uri`, and a minor-version policy are now specified (see § 5). |
| DOS/PDOS | Not mentioned. | **Added in v1.x:** `dos.total` and `dos.projected` canonical kinds. |
| Electrostatic potential | Not mentioned. | **Added in v1.x:** `volume.potential` emitted by writer and rendered by vibe-view (§ 4.10). |
| NCI / RDG surfaces | Not mentioned. | **Added in v1.x:** `volume.rdg` emitted by writer and rendered by vibe-view (§ 4.11). |
| Fermi surface | Not mentioned. | **Added in v1.x:** `fermi_surface` emitted by writer and rendered by vibe-view (§ 4.12). |
| Phonon bands + DOS | Not mentioned. | **Added in v1.x:** `phonon_bands` + `phonon_dos` emitted by writer and rendered by vibe-view (§ 4.13). |
| Equation of state | Not mentioned. | **Added in v1.x:** `equation_of_state` emitted by writer and rendered by vibe-view (§ 4.14). |
| Metadata | The blog used "provenance" broadly. | **Added in v1.x:** optional root metadata for `thermochemistry`, `dipole_moment`, `constraints`. |

This document is therefore both a specification and a correction point
for the next public write-up: QVF is no longer only a proposal, but its
extension governance should still be described honestly.

## 1. Design Goals

QVF v1 is built around six goals:

1. **One shareable artifact.** A calculation's visualization data
   should travel as one `.qvf` file.
2. **Random access.** A consumer should read only the members it needs.
   A structure-only viewer should not load a 500 MB density grid.
3. **Typed payloads.** Every member has a declared format, dtype, shape,
   and checksum where applicable.
4. **Stable common vocabulary.** Common data types use canonical section
   kinds such as `structure`, `volume.density`, `bands`, and
   `spectra.ir`.
5. **Partial support.** A consumer can support a subset of kinds and
   still open the file. Unsupported vendor sections are reported rather
   than interpreted incorrectly.
6. **Producer neutrality.** The `source.program` field can name any
   code. The core section kinds are not vibe-qc-specific.

QVF v1 is deliberately **not** a restart format. It does not attempt to
encode integral caches, SCF internal state, grids used internally by a
particular DFT engine, or every possible wavefunction representation.
The optional `wavefunction.gto` section is a visualization-oriented
Molden-like molecular orbital carrier, not a universal restart record.

## 2. Archive Model

A QVF file is a ZIP archive. The only mandatory ZIP member is
`manifest.json`. Every other member is named from the manifest, so
member paths are part of the manifest contract rather than hard-coded
by the container itself.

A typical vibe-qc-produced archive looks like this:

```text
example.qvf
├── manifest.json
├── structure/
│   ├── structure.json
│   └── symmetry.json
├── bonds/
│   └── connectivity.json
├── volumes/
│   ├── Electron_density.dat
│   ├── Electron_density_grid.json
│   ├── HOMO.dat
│   └── HOMO_grid.json
├── bands/
│   ├── kpath.json
│   └── eigenvalues.bin
├── spectra/
│   ├── ir.json
│   └── raman.json
├── trajectories/
│   ├── opt.json
│   └── opt_coords.bin
├── wavefunction/
│   ├── basis.json
│   ├── mo_metadata.json
│   └── mo_coefficients.bin
└── citations/
    └── references.bib
```

The exact paths may differ. Consumers must follow the `path` fields in
`manifest.json`. Producers should use stable, readable directories for
debuggability, but path layout is not semantic.

There is no separate `viewer_defaults.json`; viewer hints live in the
root `viewer_defaults` object in `manifest.json`.

### 2.1 `manifest.json`

Minimal manifest shape:

```jsonc
{
  "qvf_version": 1,
  "source": {
    "program": "vibe-qc",
    "version": "0.9.0",
    "calculation": "h2o_rhf_sto3g"
  },
  "sections": [
    {
      "id": "structure",
      "kind": "structure",
      "members": {
        "structure": {
          "path": "structure/structure.json",
          "format": "json",
          "sha256": "..."
        }
      }
    }
  ]
}
```

Richer manifests may add:

```jsonc
{
  "schema_uri": "https://vibe-qc.org/spec/qvf/1/manifest.schema.json",
  "provenance": {
    "method": "RKS",
    "functional": "PBE",
    "basis": "def2-SVP",
    "charge": 0,
    "multiplicity": 1,
    "scf_converged": true,
    "scf_energy": {"value": -76.3371, "units": "Eh"}
  },
  "thermochemistry": {
    "zpve_eh": 0.0294,
    "enthalpy_eh": -76.3077,
    "entropy_cal_mol_k": 52.3,
    "gibbs_free_energy_eh": -76.3321,
    "temperature_k": 298.15,
    "pressure_atm": 1.0
  },
  "dipole_moment": {
    "total_debye": 1.85,
    "vector_debye": [0.0, 0.0, 1.85],
    "origin": [0.0, 0.0, 0.2217]
  },
  "constraints": {
    "frozen_atoms": [3, 4],
    "distance_constraints": [
      {"atoms": [0, 1], "target_angstrom": 1.5}
    ]
  },
  "extensions": {
    "x_vendor_ecp": {
      "version": "1.0",
      "schema_uri": "https://vendor.example.org/qvf/ecp.schema.json",
      "critical": false
    }
  },
  "viewer_defaults": {
    "auto_open": ["vol_dens_0"],
    "vol_dens_0": {"isovalue": 0.05, "colormap": "viridis", "opacity": 0.6},
    "bookmarks": [
      {
        "name": "front",
        "camera": {
          "position": [0.0, 0.0, 12.0],
          "focal_point": [0.0, 0.0, 0.0],
          "view_up": [0.0, 1.0, 0.0],
          "view_angle": 30.0
        }
      }
    ]
  }
}
```

`schema_uri` is optional. The schema's `$id` is still the
canonical identifier for the machine contract.

The manifest root may also carry optional metadata blocks:

* **`thermochemistry`**, Thermodynamic corrections computed from the
  Hessian / frequency calculation. Fields: `zpve_eh` (zero-point
  vibrational energy in Hartree), `enthalpy_eh`, `entropy_cal_mol_k`,
  `gibbs_free_energy_eh`, `temperature_k`, `pressure_atm`. All fields
  are optional; a producer may emit only what it computed.

* **`dipole_moment`**, Electric dipole moment. Fields: `total_debye`,
  `vector_debye` (3-element array in Debye), `origin` (either a legacy
  string naming the reference point, or an explicit 3-element coordinate
  array in bohr). vibe-qc writes the numeric coordinate so charged-system
  dipoles retain their exact reference point.

* **`constraints`**, Geometry optimization constraints. May carry
  `frozen_atoms` (array of zero-based atom indices), `frozen_lattice`
  (bool), and constraint arrays: `distance_constraints`,
  `angle_constraints`, `torsion_constraints`. Each constraint has
  `atoms` (indices) and a `target_*` field with the target value.

* **`extensions`**, Extension governance block (see § 5). A mapping
  from vendor namespace to `{"version": "...", "schema_uri": "...",
  "critical": true|false}`. If a section with a matching `x_<vendor>.*`
  kind has `critical: true`, a consumer must either support it or
  refuse to open the file.

### 2.2 Section Objects

Every section has:

| Field | Required | Meaning |
|---|---:|---|
| `id` | yes | Unique section identifier within the archive. |
| `kind` | yes | Section kind from the canonical registry or `x_<vendor>.*`. |
| `members` | yes | Mapping from member role to JSON or binary member spec. |
| `label` | no | Human-readable display label. |
| `component` | no | Component hint for sections such as `volume.orbital`. |
| `critical` | no | If `true`, consumers that do not support this kind MUST refuse to open the archive (see § 5). Default `false`. |
| `schema_uri` | no | URI of a JSON Schema for this section's members, overriding the kind's canonical contract for vendor sections. |

JSON member spec:

```jsonc
{
  "path": "structure/structure.json",
  "format": "json",
  "sha256": "64 lowercase hex characters"
}
```

Binary member spec:

```jsonc
{
  "path": "volumes/rho.dat",
  "format": "binary",
  "dtype": "float32",
  "shape": [80, 80, 80],
  "sha256": "64 lowercase hex characters"
}
```

Binary members are raw C-contiguous NumPy-style arrays. The dtype is a
NumPy dtype name from the schema enum. Current writer/reader behavior
assumes ordinary little-endian platforms; adding an explicit endian tag
is an open portability hardening item.

### 2.3 Validation

The canonical validator checks:

- ZIP readability and a per-member uncompressed-size cap.
- `manifest.json` existence and JSON parseability.
- manifest conformance to the JSON Schema.
- section id uniqueness.
- declared member paths exist in the archive.
- every declared member checksum matches the bytes on disk.
- every JSON member parses as UTF-8 JSON.
- every binary member's byte length equals `dtype.itemsize * prod(shape)`.
- `volume.difference.operand_a` / `operand_b` resolve when present.
- `reaction.waypoints.trajectory_ref` resolves to a `trajectory` section.

The schema catches structural errors. `validate_qvf()` adds semantic
checks that JSON Schema cannot express cleanly.

## 3. Units And Numeric Conventions

QVF v1 uses fixed units per field. Producers convert at write time.
Consumers may convert for display but should treat the on-disk units as
the contract.

| Field | Unit / convention |
|---|---|
| `structure.atoms[].position` | Angstrom |
| `structure.lattice_vectors` | Angstrom, row vectors |
| `volume.*.grid.origin` | bohr |
| `volume.*.grid.voxel_vectors` | bohr per grid step |
| `volume.density.data` | electron density on a bohr grid, usually e / bohr^3 |
| `volume.orbital.data` | orbital amplitude on a bohr grid |
| `trajectory.coords` | Angstrom |
| `trajectory.metadata.energies` | Hartree |
| `reaction.path.coords` | Angstrom |
| `reaction.path.waypoints[].energy_eh` | Hartree |
| `bands.eigenvalues` | eV |
| `bands.kpath.fermi`, `fermi_energy_ev` | eV |
| `spectra.ir.frequencies` | cm^-1 |
| `spectra.ir.intensities` | km / mol |
| `spectra.uvvis.frequencies` / `energies_ev` | eV |
| `vibrations.metadata.frequencies` | cm^-1 |
| `vibrations.displacements` | normal-mode displacement vectors, current writer convention |
| `wavefunction.gto.basis.shells[].exponents` | bohr^-2 |
| `wavefunction.gto.mo_metadata.energies` | Hartree |
| `dos.total.energies` | eV |
| `dos.total.dos` | states / eV / cell (or spin channel) |
| `dos.total.n_electrons` | unitless (integrated count) |
| `dos.projected.energies` | eV |
| `dos.projected.dos` | states / eV / channel |
| `dos.projected.channels[].projection` | states / eV |
| `volume.potential.data` | hartree / e (atomic units of potential) |
| `volume.rdg.data` | dimensionless (|∇ρ| / (2(3π²)^⅓ ρ^⅔)) |
| `fermi_surface.energies` | eV (signed: E(k) − E_F) |
| `phonon_bands.frequencies` | cm^-1 |
| `phonon_bands.qpath.segments[].k_start/k_end` | fractional reciprocal coordinates |
| `phonon_dos.frequencies` | cm^-1 |
| `phonon_dos.total` | states / cm^-1 |
| `equation_of_state.volumes` | Angstrom^3 |
| `equation_of_state.energies` | eV |
| `equation_of_state.fit.V0` | Angstrom^3 |
| `equation_of_state.fit.B0` | GPa |
| `equation_of_state.fit.B0_prime` | unitless |
| `equation_of_state.fit.E0` | eV |

When adding fields, prefer unit-bearing names such as `energy_eh`,
`energy_ev`, `frequency_cm`, or a `{"value", "units"}` object.

## 4. Canonical Section Kinds

The current v1 registry is the set of schema `Section.*` branches plus the
vendor namespace branch. The **Writer** and **vibe-view** columns record
what is *implemented today*: `yes` means the vibe-qc writer emits the kind
and vibe-view renders it; `via structure renderer` means the section is
consumed by the structure panel rather than getting its own panel. A future
`no` would mean the writer emits the kind but vibe-view does not yet render it.

| Kind | Payload | Writer | vibe-view |
|---|---|:---:|:---:|
| `structure` | Atoms, PBC flags, optional lattice | yes | yes |
| `bonds` | Explicit connectivity table | yes | via structure renderer |
| `bond_orders` | Bond-order table (Mayer / Wiberg) with distances | yes | yes |
| `volume.density` | Scalar density grid | yes | yes |
| `volume.orbital` | Pre-sampled orbital grid | yes | yes |
| `volume.spin` | Spin-density grid | yes | yes |
| `volume.elf` | Electron localization function grid | yes | yes |
| `volume.difference` | Difference scalar field with optional operand refs | yes | yes |
| `volume.generic` | Generic scalar field escape hatch | yes | yes |
| `bands` | k-path metadata + rank-3 eigenvalue array | yes | yes |
| `spectra.ir` | IR frequency / intensity spectrum | yes | yes |
| `spectra.raman` | Raman spectrum | yes | yes |
| `spectra.uvvis` | UV/vis spectrum | yes | yes |
| `spectra.ecd` | ECD spectrum | yes | yes |
| `spectra.vcd` | VCD spectrum | yes | yes |
| `spectra.nmr` | NMR chemical shifts / tensors / couplings | yes | yes |
| `spectra.epr` | EPR g-tensor / hyperfine (A) / zero-field-splitting (D) | yes | yes |
| `spectra.generic` | Generic 1D spectrum | yes | yes |
| `trajectory` | Frames + optional energies | yes | yes |
| `reaction.path` | Self-contained reaction path | yes | yes |
| `reaction.waypoints` | Waypoints over an existing trajectory | yes | yes |
| `scan.surface` | 2D relaxed-scan energy surface | yes | yes |
| `vibrations` | Frequencies + displacement tensor | yes | yes |
| `atom_properties` | Mulliken / Loewdin / Hirshfeld charges | yes | yes |
| `structure.symmetry` | spglib-style symmetry summary | yes | yes |
| `scf_history` | SCF iteration records | yes | yes |
| `citations` | BibTeX bibliography bytes | yes | yes |
| `run.record` | Verbatim program input + full log of one invocation | yes | yes |
| `job.spec` | Declarative specification of the requested calculation | yes | yes |
| `wavefunction.gto` | GTO basis + MO coefficients (molecular, or Γ-point of a periodic system) | yes | yes, with limitations |
| `basis.ao` | Pre-sampled atomic-orbital (basis-function) grid | yes | yes |
| `dos.total` | Total density of states (energy grid + DOS) | yes | yes |
| `dos.projected` | Projected DOS (per-atom, per-l-channel) | yes | yes |
| `dos.coop` | Crystal Orbital Overlap Population bonding analysis | yes | yes |
| `dos.cohp` | Crystal Orbital Hamilton Population bonding analysis | yes | yes |
| `volume.potential` | Electrostatic potential on a grid | yes | yes |
| `volume.rdg` | Reduced density gradient (NCI analysis) | yes | yes |
| `fermi_surface` | 3D k-space energy grid near Fermi level | yes | yes |
| `phonon_bands` | Phonon band structure (q-path + frequencies) | yes | yes |
| `phonon_dos` | Phonon density of states | yes | yes |
| `equation_of_state` | Volume-energy curve + fitted EOS params | yes | yes |
| `topology.qtaim` | QTAIM critical points + bond paths | yes | yes |
| `x_<vendor>.*` | Vendor extension | external producers | listed; known overlays may render |

¹ **All writer kinds now render.** The six kinds the writer promoted from
reserved on 2026-06-09 (`volume.potential`, `volume.rdg`, `fermi_surface`,
`phonon_bands`, `phonon_dos`, `equation_of_state`) were briefly emitted ahead of
their viewer renderers (roadmap Phase J, now complete); as of 2026-06 vibe-view
renders all of them. A consumer that encounters a future not-yet-rendered kind
should still silently skip it per Rule 2.

The 40 canonical writer kinds above are what the writer emits and the viewer
renders today: 39 first-class viewer kinds plus `bonds`, rendered via the
structure view. See
[The QVF file format, end to end](tutorial/qvf_file_format.md).

The reserved names `volume.orbital_projection`,
`topology.elf_basins`, and `projections.lcao` are
planning names. They are not portable v1 viewer contracts yet. External
producers should use `x_<vendor>.*` for experimental data until a kind
is promoted into the schema.

### 4.1 `structure` And `bonds`

`structure` carries atoms in Angstrom and optional lattice vectors:

```jsonc
{
  "atoms": [
    {"symbol": "O", "position": [0.0, 0.0, 0.1173], "atomic_number": 8},
    {"symbol": "H", "position": [0.0, 0.7572, -0.4692], "atomic_number": 1}
  ],
  "pbc": [false, false, false],
  "lattice_vectors": null
}
```

For periodic systems, `lattice_vectors` is a `[3, 3]` array of row
vectors in Angstrom and `pbc` marks periodic axes. The payload is
schema'd as `$defs/StructurePayload`; its contract, normative in
[`spec/qvf-format-spec.md` in QVF](https://github.com/vibe-qc/qvf/blob/main/spec/qvf-format-spec.md) § 5.1, is:

* `pbc` is REQUIRED whenever `lattice_vectors` is non-null, and is the
  only carrier of per-axis periodicity. Absent `pbc` means a molecule.
  The periodic axes need not be the leading ones.
* `dimensionality` (payload, not section object) is optional and
  derived: `dimensionality == sum(pbc)`. It carries no axis identity.
* Row `i` of `lattice_vectors` pairs with `pbc[i]`. Where `pbc[i]` is
  false, row `i` is non-physical bookkeeping kept so the matrix stays
  full-rank `3x3` -- never a cell edge, a tiling axis, or a vacuum
  extent. A synthesized 30-bohr slab normal and a real 30-bohr vacuum
  gap are numerically identical, so `pbc` is the only thing that tells
  them apart.

Explicit connectivity is carried as a separate `bonds` section:

```jsonc
{
  "pairs": [
    {"i": 0, "j": 1, "order": 1.0},
    {"i": 0, "j": 2, "order": 1.0}
  ]
}
```

If no `bonds` section is present, viewers may infer bonds from covalent
radii. Explicit bonds take precedence.

#### Biomolecule metadata (optional)

For proteins and other biomolecules the `structure` **section object**
(peer to `id`/`kind`/`members`, *not* inside `structure.json`) may carry
optional biomolecule metadata for ribbon/cartoon rendering. They are
schema'd as optional properties of `$defs/SectionStructure`: **additive,
no `qvf_version` bump**, per the versioning policy in
[`GOVERNANCE.md` in QVF](https://github.com/vibe-qc/qvf/blob/main/GOVERNANCE.md) (an optional member is the additive case). All
four are independently optional:

```jsonc
{
  "id": "structure",
  "kind": "structure",
  "members": { "structure": { "path": "structure/structure.json", "...": "..." } },
  "chains": ["A", "B"],
  "residues": [
    {"name": "GLY", "seq": 1, "chain": "A", "atom_indices": [0, 1, 2, 3]},
    {"name": "ALA", "seq": 2, "chain": "A", "atom_indices": [4, 5, 6]}
  ],
  "secondary_structure": [
    {"type": "helix", "chain": "A", "start_seq": 1, "end_seq": 12},
    {"type": "sheet", "chain": "A", "start_seq": 15, "end_seq": 22}
  ],
  "b_factors": [12.4, 15.9, 11.2, 30.7, 28.1, 22.0, 19.6]
}
```

`atom_indices` are 0-based into the `structure.json` `atoms` list.
`secondary_structure[].type` is one of `helix`, `sheet`, `coil`; the
schema pins that enum, so the writer drops a range with any other
descriptor rather than failing the whole archive at the validate gate.
`b_factors` holds one temperature factor per atom, parallel to the
payload's atom order.

The writer accepts all four via the `biomolecule_data` context key on
`write_qvf`/`qvf_bytes` and normalizes numpy ints/arrays to plain
`int`/`list`. Consumers that don't understand the fields ignore them;
vibe-view uses them to draw cartoon representations.

A structure payload's atoms may additionally carry the per-atom
biomolecular fields `atom_name` (PDB cols 13-16), `residue_name` (cols
18-20), `residue_seq` (cols 23-26), `chain_id` (col 22) and `b_factor`
(cols 61-66). Two precedence rules make the two carriers unambiguous:

* A consumer that can derive residues, chains, or secondary structure
  itself (vibe-view does, from the per-atom fields and from CA geometry)
  **prefers the producer-supplied section fields** where they exist. A
  producer knows what geometry cannot reveal, such as an alpha from a pi
  from a 3-10 helix.
* Per-atom `b_factor` **wins** over the section-level `b_factors` array,
  because it cannot desynchronize from its own atom. A `b_factors` array
  whose length differs from the atom count is ignored outright rather
  than applied partially.

### 4.2 `volume.*`

All volume kinds use the same member shape:

```jsonc
{
  "id": "vol_dens_0",
  "kind": "volume.density",
  "label": "Electron density",
  "members": {
    "grid": {"path": "volumes/Electron_density_grid.json", "format": "json", "sha256": "..."},
    "data": {
      "path": "volumes/Electron_density.dat",
      "format": "binary",
      "dtype": "float32",
      "shape": [80, 80, 80],
      "sha256": "..."
    }
  }
}
```

Grid JSON:

```jsonc
{
  "origin": [-8.0, -8.0, -8.0],
  "voxel_vectors": [
    [0.2, 0.0, 0.0],
    [0.0, 0.2, 0.0],
    [0.0, 0.0, 0.2]
  ],
  "shape": [80, 80, 80]
}
```

`voxel_vectors` is `[v_i, v_j, v_k]`, three per-voxel step vectors in
bohr. For a data index `(i, j, k)`, the point position is:

```text
origin + i * v_i + j * v_j + k * v_k
```

This is point-centered data: the number of points in the rendered grid
matches `shape`, not `shape + 1`. Non-orthogonal grids are supported by
using non-zero off-diagonal vector components.

`volume.difference` may add `operand_a` and `operand_b` section ids.
If one operand is present, both are required. The sign convention is:

```text
data = operand_a - operand_b
```

#### Periodic BvK torus grids

For periodic visualization archives the `structure.lattice_vectors` field is
the displayed cell. Producers that visualize a finite Born-von Karman torus
must write the full BvK supercell there, not the primitive cell, when the
volume data are sampled on the supercell. The `structure.pbc` flags and any
`structure.dimensionality` metadata identify which axes are physical periodic
axes; inactive axes may still carry vacuum-box vectors for display.

The grid descriptor remains unchanged: `origin` and `voxel_vectors` are in
bohr, while the structure and lattice are in Angstrom. A torus-aligned grid
whose shape times voxel vectors spans the BvK cell tiles without gaps when a
viewer replicates it along the lattice vectors. For example, the documented
chi-CCM-B fixtures include:

- a 1D H-chain with `pbc=[true,false,false]`, lattice lengths
  `10.583544 x 10.583544 x 10.583544` Angstrom, and a `40 x 40 x 40`
  density grid spanning `20 x 20 x 20` bohr;
- a 3D H2-pair with `pbc=[true,true,true]`, lattice lengths
  `12.700253 x 6.350127 x 6.350127` Angstrom, and a `48 x 24 x 24`
  density grid spanning `24 x 12 x 12` bohr.

Those archives deliberately omit periodic `wavefunction.gto`; the portable
visualization payload is the precomputed `volume.density` and
`volume.orbital` grids. A producer may add vendor sections such as
`x_vibeqc.aiccm2026dev_b_convention` for finite-torus convention metadata and
`x_ccm.wannier_centers` for centre/spread overlays. Consumers that do not
understand those vendor sections still open the canonical structure and volume
sections; vibe-view recognizes `x_ccm.wannier_centers` as an optional marker
overlay. Downloadable sanitized examples are summarized in the
{download}`chi-CCM-B fixture README <_static/examples/chi-ccm-b-qvf/README.md>`.

### 4.3 `bands`

Band eigenvalues are always rank 3:

```jsonc
{
  "id": "bands0",
  "kind": "bands",
  "members": {
    "kpath": {"path": "bands/kpath.json", "format": "json", "sha256": "..."},
    "eigenvalues": {
      "path": "bands/eigenvalues.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [1, 100, 26],
      "sha256": "..."
    }
  }
}
```

Shape is `[n_spin, n_kpoints, n_bands]`. Restricted calculations use
`n_spin = 1`. Energies are stored in eV.

The `kpath` JSON should include:

```jsonc
{
  "kind": "bands",
  "version": "1.0",
  "n_spin": 1,
  "n_kpoints": 100,
  "n_bands": 26,
  "fermi": -4.71,
  "fermi_energy_ev": -4.71,
  "reciprocal_space": true,
  "segments": [
    {
      "label_start": "G",
      "label_end": "X",
      "k_start": [0.0, 0.0, 0.0],
      "k_end": [0.5, 0.0, 0.0],
      "n_points": 50
    }
  ]
}
```

Consumers should accept either explicit segment `start` / `end`
indices or writer-style `n_points` segments.

**Fat bands (band-character projections).** When band-character weights are
available, the section carries an optional `projections` binary member of
shape `[n_kpoints, n_bands, n_channels]` and the `kpath` JSON adds a
`channels` array naming each channel:

```jsonc
{
  "id": "bands0",
  "kind": "bands",
  "members": {
    "kpath": {"path": "bands/kpath.json", "format": "json", "sha256": "..."},
    "eigenvalues": {
      "path": "bands/eigenvalues.bin", "format": "binary",
      "dtype": "float64", "shape": [1, 100, 26], "sha256": "..."
    },
    "projections": {
      "path": "bands/projections.bin", "format": "binary",
      "dtype": "float64", "shape": [100, 26, 8], "sha256": "..."
    }
  }
}
```

…with the `kpath` JSON extended:

```jsonc
{
  "channels": ["Si-3s", "Si-3p", "Si-3d", "O-2s", "O-2p"],
  "weights_kind": "mulliken"
}
```

Consumers render fat bands by colouring each band line segment according to
the dominant channel weight. Without `projections`, bands are monochromatic.

### 4.4 Spectra

`spectra.ir`, `spectra.raman`, `spectra.uvvis`, `spectra.ecd`,
`spectra.vcd`, and `spectra.generic` each carry one JSON member named
`spectrum`. The common shape is:

```jsonc
{
  "frequencies": [1000.0, 1500.0],
  "intensities": [12.0, 4.5]
}
```

The physical meaning and units depend on kind. For example, IR uses
cm^-1 and km/mol; UV/vis and ECD use eV on the x axis.

`spectra.nmr` is intentionally different. Its `spectrum` member is an
object with conventional optional keys:

```jsonc
{
  "isotope": "1H",
  "reference": "TMS",
  "solvent": "gas",
  "chemical_shifts": [
    {"atom_index": 0, "symbol": "H", "isotropic_shift_ppm": 4.65}
  ],
  "shielding_tensors": [],
  "j_couplings": []
}
```

The schema intentionally keeps NMR payload shape loose in v1 while the
field vocabulary settles.

`spectra.epr` follows the same object-shaped pattern for electron
paramagnetic resonance parameters. Its `spectrum` member carries the
g-tensor, per-nucleus hyperfine (A) tensors, and the zero-field-splitting
(D) tensor. This schematic example uses ellipses for omitted tensor
entries; replace them with computed numbers before serializing valid JSON:

```text
{
  "g_tensor": {
    "matrix": [[2.0023, 0.0, 0.0], [0.0, 2.0021, 0.0], [0.0, 0.0, 2.0089]],
    "isotropic": 2.0044,
    "principal": [2.0023, 2.0021, 2.0089]
  },
  "hyperfine": [
    {"atom_index": 0, "symbol": "N", "isotope": "14N",
     "a_iso_mhz": 45.2, "a_tensor_mhz": [[...], [...], [...]]}
  ],
  "zero_field_splitting": {"d_mhz": 1200.0, "e_mhz": 30.0}
}
```

Units follow EPR convention: g-values are dimensionless; hyperfine and
zero-field-splitting parameters are in MHz. Like `spectra.nmr`, the payload
shape is intentionally loose in v1, a producer emits whichever subset it
computed, and consumers render defensively. `spectra.epr` was promoted to a
canonical kind by maintainer decision (2026-07-08) ahead of the usual
two-producer threshold in § 5.7, to give ORCA a canonical target for its EPR
output.

### 4.5 Trajectories, Reactions, And Vibrations

`trajectory` has:

- `metadata` JSON with atom identities and optional energies.
- `coords` binary `float64` with shape `[n_frames, n_atoms, 3]` in
  Angstrom.

`reaction.path` uses the same coordinate layout and adds waypoint
records in metadata:

```jsonc
{
  "waypoints": [
    {"frame_index": 0, "label": "R", "kind": "reactant"},
    {"frame_index": 7, "label": "TS", "kind": "transition_state", "energy_eh": -75.1}
  ],
  "reaction_coordinate": [0.0, 0.2, 0.5]
}
```

**Periodic reaction paths.** `reaction.path` may carry an optional
`lattice` binary member: float64 `[3, 3]` (fixed cell) or
`[n_frames, 3, 3]` (variable cell), columns = a, b, c, in **bohr**
(matching `vibeqc.PeriodicSystem.lattice`). Its **presence marks the
path periodic**: a `reaction.path` without `lattice` is unambiguously
molecular and renderers must draw no cell. This is capability detection
by member presence; it does **not** bump `qvf_version` (see § 5.1).

`reaction.waypoints` carries only the annotations and references an
existing `trajectory` section through section-level `trajectory_ref`.
The validator checks that the reference resolves to a `trajectory`
section. Frame-bound validation is a desired future tightening.

`vibrations` carries:

- `metadata` JSON with atoms and frequencies in cm^-1.
- `displacements` binary with shape `[n_modes, n_atoms, 3]`.

### 4.6 `wavefunction.gto`

`wavefunction.gto` carries an atom-centered Gaussian basis and MO
coefficient matrices. It describes molecular wavefunctions and the
**Γ-point (k=0) crystalline orbitals of periodic systems**, at Γ the Bloch
phase `e^{ik·r}` is unity, so the orbitals are real linear combinations of
the cell AOs and are exactly representable by real MO coefficients over the
same GTO basis. Full k-resolved Bloch wavefunctions (complex, k-dependent)
are **not** carried by this kind; they remain out of the v1 visualization
contract.

```jsonc
{
  "id": "wf",
  "kind": "wavefunction.gto",
  "members": {
    "basis": {"path": "wavefunction/basis.json", "format": "json", "sha256": "..."},
    "mo_metadata": {"path": "wavefunction/mo_metadata.json", "format": "json", "sha256": "..."},
    "mo_coefficients": {
      "path": "wavefunction/mo_coefficients.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [41, 41],
      "sha256": "..."
    }
  }
}
```

Unrestricted wavefunctions replace `mo_coefficients` with
`mo_coefficients_alpha` and `mo_coefficients_beta`.

`basis` JSON conventions:

- `structure_ref` names the referenced `structure` section.
- `center` is a zero-based atom index.
- `l` is shell angular momentum.
- `exponents` are in bohr^-2.
- `coefficients` apply to **`N_i`-normalized** primitive Gaussians, i.e.
    each contracted basis function is reconstructed as
    ``Σ_i c_i · φ_i(r)`` where
    ``φ_i(r) = N_i · (x−A_x)^{l_x} (y−A_y)^{l_y} (z−A_z)^{l_z} e^{−α_i r²}``
    with ``N_i = (2α_i/π)^{3/4} · (4α_i)^{l/2} / √((2l−1)!!)``,
    ``l = l_x + l_y + l_z``. ``N_i`` depends only on the total ``l`` (it is
    the *axial* norm): mixed Cartesian components are deliberately **not**
    individually unit-normalized (QVF spec Appendix A.1).
    **Producers that source coefficients from an engine that stores them
    pre-multiplied by ``N_i`` (e.g. libint, PySCF's ``libcint``) MUST
    divide each coefficient by ``N_i`` before writing.** vibe-qc's writer
    does this automatically in ``qvf_wf_data``.
- `pure: true` means spherical harmonics; `false` means Cartesian.
- AO order is part of the contract: spherical `m = -l ... +l`;
  Cartesian uses libint lexicographic ordering.

`mo_metadata` JSON conventions:

- `spin` is `"restricted"` or `"unrestricted"`; `orbital_kind` is
  `"canonical"` / `"natural"` / `"localized"`.
- `energies` are in Hartree. `occupations` are electron counts per MO when
  `occupation_semantics` is `"electron_occupation"`. Natural transition
  orbital sections use the same numeric slot for transition weights and MUST
  set `occupation_semantics` to `"transition_weight"`. New producers MUST
  write one of those two values whenever `orbital_kind` is `"natural"`;
  consumers must treat an unmarked legacy natural section as ambiguous rather
  than inferring the meaning from its section id or from the sum of its
  values.
- `k_point` (optional) records the fractional reciprocal-space coordinate
  the MO coefficients describe. **Absent (or `[0, 0, 0]`) means the Γ-point**
  - the only k-point a `wavefunction.gto` section may carry in v1. Producers
  emitting a periodic wavefunction MUST evaluate at Γ and SHOULD write
  `k_point: [0.0, 0.0, 0.0]` so the archive is self-describing.

Periodic notes + limits in the current reference viewer:

- **Periodic systems are supported at Γ only.** The MO coefficients are the
  real Γ-point crystalline orbitals over the cell's GTO basis. A producer
  that has only multi-k results writes the Γ-point set (vibe-qc's periodic
  runner derives a Γ proxy when the SCF stored per-k coefficients).
- The reference viewer evaluates the orbital as `Σ_µ c_µ χ_µ(r)` using the
  central-cell atom centers; it does **not** add periodic-image AO tails, so
  density that would wrap across a cell face is truncated at the boundary.
  Acceptable for localized orbitals; a known approximation for diffuse ones.
- vibe-qc may write `x_vibeqc.bloch_wavefunction` as a first-party vendor
  section for all-k READ restarts. Its version 1.0 restricted payload stores
  per-k complex coefficients, occupations, energies and fractional k-point
  metadata. Version 1.1 adds `restart_kind="spin_density_bloch_kpoints"`,
  `spin="unrestricted"` for physical alpha/beta densities. Each metadata block
  carries `k_index`, `k_point`, `k_weight`, `density_alpha` and `density_beta`;
  the last two fields name distinct binary members. Density members are
  float64 `[n_ao, n_ao, 2]` arrays in AO row/column order with trailing
  `[real, imag]` components. `density_encoding="complex_split_last_axis"`
  and `density_layout="ao_row_ao_column"` make this convention explicit.
  Source basis shells and structure use the same references as version 1.0.
  Both physical spin channels are stored even when one is empty; canonical
  orbitals from a later diagonalization must not replace a returned SCF
  density. The core READ adapter checks member hashes, quadrature, Hermiticity
  and positivity before using the state. This remains a restart-only vendor
  section, independent of the QVF manifest version and canonical visualization
  contract. Older vibe-qc restart readers reject the spin variant; other
  consumers may ignore the section under the normal `x_*` rules.
- vibe-view currently evaluates shells up to `l = 3` for display.
  Higher-`l` basis functions need a renderer extension or pre-sampled
  `volume.orbital` sections.

### 4.8 `dos.total`, Total Density of States

Carries the total electronic density of states, typically broadened with
a Gaussian or Lorentzian smearing function:

```jsonc
{
  "id": "dos_total",
  "kind": "dos.total",
  "members": {
    "energies": {
      "path": "dos/energies.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [500],
      "sha256": "..."
    },
    "dos": {
      "path": "dos/total.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [500],
      "sha256": "..."
    }
  }
}
```

Both `energies` and `dos` are rank-1 arrays of the same length. For
spin-polarized calculations, `dos` is shape `[2, n_points]` (alpha,
beta channels). Energies are in eV, referenced to the Fermi level
(which is at 0.0 eV).

Metadata JSON (role `meta`) may carry:

```jsonc
{
  "smearing": 0.05,
  "smearing_type": "gaussian",
  "fermi_energy_ev": -4.71,
  "n_electrons": 64.0,
  "n_spin": 1
}
```

`n_electrons` is the integrated count under the DOS, useful for
cross-checking and for Fermi-level interpolation.

### 4.9 `dos.projected`, Projected Density of States

Carries atom-projected and/or angular-momentum-projected DOS. The data
is a flat rank-2 array where each row is one projection channel:

```jsonc
{
  "id": "dos_pdos",
  "kind": "dos.projected",
  "members": {
    "energies": {
      // Distinct from dos.total's "dos/energies.bin": when both sections
      // ship in one archive a shared path would collide as a duplicate zip
      // entry, so dos.projected uses its own "dos/projected_energies.bin".
      "path": "dos/projected_energies.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [500],
      "sha256": "..."
    },
    "projections": {
      "path": "dos/projections.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [12, 500],
      "sha256": "..."
    }
  }
}
```

Metadata JSON (role `meta`) carries the channel descriptions:

```jsonc
{
  "energies_units": "eV",
  "n_spin": 1,
  "fermi_energy_ev": -4.71,
  "channels": [
    {"atom_index": 0, "symbol": "Mg", "l": 0, "label": "Mg-3s"},
    {"atom_index": 0, "symbol": "Mg", "l": 1, "label": "Mg-3p"},
    {"atom_index": 1, "symbol": "O",  "l": 0, "label": "O-2s"},
    {"atom_index": 1, "symbol": "O",  "l": 1, "label": "O-2p"}
  ]
}
```

Shape is `[n_channels, n_points]` for restricted or `[n_spin, n_channels, n_points]`
for spin-polarized. Consumers should check `n_spin` from the metadata
and interpret the leading dimension accordingly.

The `dos.total` and `dos.projected` kinds are deliberately separate
(rather than a single combined kind) so that a consumer doing only
total-DOS plotting can ignore the (potentially large) projection tensor.

### 4.10 `volume.potential`, Electrostatic Potential Grid

**Writer-emitted and rendered by vibe-view** (electrostatic-potential
renderer added 2026-06-20). When
a producer emits it, the member structure is identical to `volume.density`
(grid JSON + binary data), but the data field carries the total
electrostatic potential in hartree/e (atomic units of potential):

```jsonc
{
  "id": "esp",
  "kind": "volume.potential",
  "label": "Electrostatic potential",
  "members": {
    "grid": {"path": "volumes/esp_grid.json", "format": "json", "sha256": "..."},
    "data": {
      "path": "volumes/esp.dat",
      "format": "binary",
      "dtype": "float64",
      "shape": [80, 80, 80],
      "sha256": "..."
    }
  }
}
```

Visually, consumers typically render a single isosurface colored by the
potential value (blue = positive, red = negative) or map the potential
onto the solvent-accessible surface. vibe-view may offer both.

### 4.11 `volume.rdg`, Reduced Density Gradient (NCI Analysis)

The reduced density gradient s(r) = |∇ρ(r)| / (2(3π²)^⅓ ρ(r)^⅔) is the
key scalar field for Non-Covalent Interaction (NCI) analysis. Its
member structure matches `volume.density`:

```jsonc
{
  "id": "rdg",
  "kind": "volume.rdg",
  "label": "Reduced density gradient",
  "members": {
    "grid": {"path": "volumes/rdg_grid.json", "format": "json", "sha256": "..."},
    "data": {
      "path": "volumes/rdg.dat",
      "format": "binary",
      "dtype": "float32",
      "shape": [80, 80, 80],
      "sha256": "..."
    }
  }
}
```

The data field carries the dimensionless s(r) value. Consumers that
also have a `volume.density` section in the archive should compute
sign(λ₂)ρ from the density (where λ₂ is the second eigenvalue of the
Hessian of ρ) and use it as the coloring field for the NCI isosurface
(standard convention: blue = attractive, green = van der Waals,
red = repulsive).

If the producer has already computed sign(λ₂)ρ, it may additionally
write a `volume.density` section with the sign(λ₂)ρ data for consumer
convenience.

### 4.12 `fermi_surface`, 3D Fermi Surface in Reciprocal Space

For metallic periodic systems, the Fermi surface is a 3D scalar field
in reciprocal space: the signed distance from the Fermi level as a
function of k-point for a selected band. The archive carries raw band
energies on a Monkhorst-Pack mesh, and the consumer extracts isosurfaces
at E = E_F:

```jsonc
{
  "id": "fermi0",
  "kind": "fermi_surface",
  "members": {
    "mesh": {
      "path": "fermi/mesh.json",
      "format": "json",
      "sha256": "..."
    },
    "energies": {
      "path": "fermi/energies.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [10, 10, 10, 2],
      "sha256": "..."
    }
  }
}
```

`mesh` JSON:

```jsonc
{
  "nk1": 10,
  "nk2": 10,
  "nk3": 10,
  "n_spin": 1,
  "fermi_energy_ev": -4.71,
  "band_indices": [3, 4],
  "lattice_vectors": [[4.2, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 4.2]]
}
```

`energies` shape is `[nk1, nk2, nk3, n_bands_near_fermi]`, only bands
that cross or are within a window of ±2 eV (producer's choice) of E_F.
The consumer renders a 3D isosurface at value = 0.0 (E_F) for each band,
optionally colored by band index or by spin channel.

The reciprocal-space grid is assumed gamma-centered and uniformly
spaced. `lattice_vectors` (in Angstrom, real space) enables the consumer
to draw the reciprocal cell wireframe.

### 4.13 `phonon_bands` And `phonon_dos`, Phonon Structure

#### `phonon_bands`

Parallel structure to electronic `bands`, but storing phonon frequencies
along a q-path in the Brillouin zone:

```jsonc
{
  "id": "phonon_bands",
  "kind": "phonon_bands",
  "members": {
    "qpath": {"path": "phonons/qpath.json", "format": "json", "sha256": "..."},
    "frequencies": {
      "path": "phonons/frequencies.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [1, 100, 24],
      "sha256": "..."
    }
  }
}
```

`frequencies` shape is `[n_qpts, n_modes]` (3 × n_atoms modes).
Frequencies are in cm⁻¹. Producers should ensure acoustic modes start
near zero; consumers should not display or highlight negative
(imaginary) frequencies unless explicitly requested.

`qpath` JSON:

```jsonc
{
  "n_atoms": 8,
  "n_modes": 24,
  "has_eigenvectors": false,
  "segments": [
    {
      "label_start": "G",
      "label_end": "X",
      "k_start": [0.0, 0.0, 0.0],
      "k_end": [0.5, 0.0, 0.0],
      "n_points": 50
    }
  ]
}
```

Optional: if eigenvectors are available, an additional `eigenvectors`
binary member (shape `[n_qpts, n_modes, n_atoms, 3]`, float64) carries
the displacement vectors for animated display. `has_eigenvectors: true`
in the metadata tells the consumer to load them.

#### `phonon_dos`

```jsonc
{
  "id": "phonon_dos",
  "kind": "phonon_dos",
  "members": {
    "meta": {"path": "phonons/dos_meta.json", "format": "json", "sha256": "..."},
    "frequencies": {
      "path": "phonons/dos_freq.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [500],
      "sha256": "..."
    },
    "dos": {
      "path": "phonons/dos_total.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [500],
      "sha256": "..."
    }
  }
}
```

Metadata `meta` may carry `smearing`, `smearing_type`, `n_atoms`,
`n_modes`. Optionally, a `projected` binary member (shape
`[n_atoms, n_points]`) carries atom-projected phonon DOS.

### 4.14 `equation_of_state`, EOS Curve And Fit

Carries the raw volume-energy points from a series of periodic
calculations at varying unit-cell volumes, plus the fitted EOS
parameters:

```jsonc
{
  "id": "eos",
  "kind": "equation_of_state",
  "members": {
    "volumes": {
      "path": "eos/volumes.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [15],
      "sha256": "..."
    },
    "energies": {
      "path": "eos/energies.bin",
      "format": "binary",
      "dtype": "float64",
      "shape": [15],
      "sha256": "..."
    },
    "fit": {
      "path": "eos/fit.json",
      "format": "json",
      "sha256": "..."
    }
  }
}
```

`fit` JSON:

```jsonc
{
  "model": "birch_murnaghan",
  "V0": 120.4,
  "E0": -5000.23,
  "B0": 74.2,
  "B0_prime": 4.1,
  "energy_unit": "eV",
  "volume_unit": "angstrom^3",
  "pressure_unit": "GPa",
  "residual_rms": 1.2e-5,
  "pressures_gpa": [null, -0.2, 0.0, 0.3]
}
```

The `pressures_gpa` array (same length as volumes) reports the
pressure at each point from the EOS fit, or `null` if the point was
excluded. `residual_rms` is the RMS deviation of the fit from the
raw points.

Supported `model` values: `"birch_murnaghan"` (3rd order,
E(V) = E0 + 9V0B0/16 · {[(V0/V)^(2/3) − 1]³ · B0' + [(V0/V)^(2/3) − 1]² · [6 − 4(V0/V)^(2/3)]}),
`"murnaghan"`, `"vinet"`.

### 4.15 `bond_orders`

Carries a computed bond-order table (Mayer, Wiberg, etc.) with interatomic
distances. Distinct from the `bonds` connectivity section: `bond_orders`
is a quantitative per-pair table, `bonds` is a qualitative connectivity list.

```jsonc
{
  "id": "bond_orders",
  "kind": "bond_orders",
  "members": {
    "bond_orders": {
      "path": "bonds/orders.json",
      "format": "json",
      "sha256": "..."
    }
  }
}
```

The JSON payload:

```jsonc
{
  "method": "mayer",
  "pairs": [
    {"i": 0, "j": 1, "order": 0.98, "distance_ang": 1.42, "symbol_i": "O", "symbol_j": "H"},
    {"i": 0, "j": 2, "order": 0.03, "distance_ang": 2.85, "symbol_i": "O", "symbol_j": "H"}
  ]
}
```

- `method` is the bond-order definition: `"mayer"`, `"wiberg"`, etc.
- Each pair must include `i`, `j` (0-based atom indices) and `order`.
  `distance_ang`, `symbol_i`, and `symbol_j` are optional but recommended.
- Consumers should render a sortable table; paired with a `bonds` section
  a viewer may also colour bond cylinders by order.

### 4.16 `topology.qtaim`

Carries the result of a QTAIM (Quantum Theory of Atoms in Molecules)
topological analysis: critical points of the electron density and (optionally)
gradient-path bond paths connecting atoms through bond critical points.

```jsonc
{
  "id": "qtaim",
  "kind": "topology.qtaim",
  "members": {
    "critical_points": {
      "path": "topology/critical_points.json",
      "format": "json",
      "sha256": "..."
    }
  }
}
```

The JSON payload:

```jsonc
{
  "points": [
    {"type": "bcp", "position": [0.71, 0.0, 0.0], "rho": 0.26,
     "laplacian": -0.54, "ellipticity": 0.12, "atom_pair": [0, 1]},
    {"type": "rcp", "position": [0.0, 0.0, 0.5], "rho": 0.08,
     "laplacian": 0.12},
    {"type": "ccp", "position": [0.0, 0.0, 0.0], "rho": 0.01,
     "laplacian": 0.05}
  ],
  "bond_paths": [
    {"atoms": [0, 1], "path": [[0.0, 0.0, 0.0], [0.1, 0.05, 0.0],
                                "...", [0.71, 0.0, 0.0]]}
  ]
}
```

- Each point has a `type`: `"bcp"` (bond CP, (3,-1)), `"rcp"` (ring CP,
  (3,+1)), `"ccp"` (cage CP, (3,+3)), or `"ncp"` (nuclear CP, (3,-3)).
- `position` is in Angstrom; `rho` in e/Å³; `laplacian` in e/Å⁵.
- `bond_paths` is optional: a list of gradient-path polylines, each with
  `atoms: [i, j]` and `path: [[x, y, z], …]` in Angstrom.
- Consumers render: coloured spheres at CP positions (orange for BCP,
  yellow for RCP, green for CCP), polylines for bond paths, and tooltips
  with ρ / ∇²ρ / ε.

### 4.7 Provenance, Citations, And SCF History (Updated)

Root `source` is mandatory and intentionally small:

```jsonc
{"program": "vibe-qc", "version": "0.9.0", "calculation": "job_name"}
```

Root `provenance` is optional and carries calculation-level metadata
such as method, basis, functional, charge, multiplicity, convergence,
the SCF iteration count (`n_scf_iterations`), and unit-tagged energies.

`citations` carries one binary member, `references`, containing UTF-8
BibTeX bytes.

`run.record` (spec § 5.8) makes the archive a self-contained record of
the calculation: opaque UTF-8 binary members `input` (the verbatim job
input; for vibe-qc, the driving Python script) and `log` (the full
`.out`), an optional JSON `files` index with original basenames (never
absolute paths), and a required section-level `program` identity. The
runners assemble it after the output channel closes
(`assemble_run_record` in `output/formats/qvf.py`), so the embedded log
is the complete on-disk log at archive time. The periodic runner writes
its final QVF only after any geometry-optimization epilogue, so the log
member is byte-complete and does not need a truncated marker.

Terminal archives state their lifecycle whether or not they came from a
container. Every vibe-qc run that settles -- `run_job`,
`run_periodic_job` (both the basis-driven and the semiempirical route),
and the double-hybrid driver -- stamps
`provenance.run_status` `"converged"` / `"failed"` on its final archive
via `terminal_run_status(result)` (`output/formats/qvf.py`), using the
same rule the container finalizer applies, so a calculation settles to
the same status either way. This matters because § 3.2 makes an absent
`run_status` mean *"not stated"* and forbids a consumer from inferring
one: an archive that omits it never declares it is done, and vibe-view
shows it no lifecycle status. Archives that are not a settled run --
reaction paths, relaxed scans, `qvf_bytes` -- deliberately stay silent.

`"failed"` reaches a molecular archive by a different route than
`"converged"`. A non-converged SCF makes `run_job` **raise**, refusing to
treat the energy as a successful calculation, so no `{stem}.qvf` result
archive is published at all -- the terminal-dispatch
`terminal_run_status` call is never reached on that path. The lifecycle
is still settled, by the checkpointer: `_checkpointer.finalize("failed")`
writes a terminal archive stamped `run_status = "failed"` (to the
`checkpoint_qvf` path when one was requested), so a consumer watching a
live job sees it end instead of a `"running"` snapshot frozen forever.
The container runner settles the same way, catching the runner error and
finalizing the container as `failed` with whatever log exists. Pinned
end-to-end by `test_divergent_scf_settles_the_failed_lifecycle`
(`rhf_options.max_iter = 1` is the supported way to force divergence;
`run_job` exposes no top-level iteration cap).

`job.spec` (spec § 5.9) is the request-side complement of `run.record`:
one JSON member, `spec`, carrying the declarative JobSpecPayload:
`job_type` (`"molecular"` | `"periodic"`, required) plus optional
`method`, `basis`, `functional`, `charge`, `multiplicity`, `kpoints`,
`tasks`, and an open `options` object. Together with a `structure`
section and `provenance.run_status = "pending"` (§ 3.2 of the spec) it
makes an archive an executable input: a runner reconstructs the job,
marks it `"running"`, executes it, and updates the same container in
place; result sections plus a `run.record` are added and `run_status`
moves to
`"converged"` / `"failed"`, while the `job.spec` section is preserved
byte-for-byte. A failed invocation also receives a complete run record
with the executed spec, available log, timestamps, and nonzero exit
status. A settled container is refused unless the caller explicitly
forces a re-run; each forced re-run appends a sequenced run record. The
payload is data, never code; a runner must not execute
`run.record.input` as a side effect of opening or running an archive
(producer: `_write_job_spec_section` via the `job_spec` context key of
`write_qvf`; container execution: `vibeqc.qvf_job.run_container` /
`vibeqc run job.qvf`).

A settled container is self-contained: no sidecar file is needed to
understand the run:

* the full `.out` log is the `run.record` `log` member (re-embedded at
  container-update time, after any post-archive epilogue, so it is
  byte-complete);
* the `.system` manifest is the `run.record` `attachment.system`
  member (declared in the `files` index with its original basename);
* optional performance and structured-event sidecars are
  `attachment.perf` and `attachment.structured`;
* the `.references` / `.bibtex` content is the `citations` section.

The sidecar files themselves still land at the output stem; the
container makes them redundant, it does not suppress them. Per-artifact
runner switches (`write_xyz_file=False`, `write_molden_file=False`, …)
remain the way to turn individual sidecars off, passed as `options` in
the job.spec or as local overrides.

Containers have a first-class single-file vq path: the container is the whole
payload, and artifact-only fetch returns only the updated container:

```sh
# build the pending container (locally, from Python)
#   vq.write_pending_qvf(mol, "job", method="rhf", basis="sto-3g")
vq submit HOST job.qvf --program vibeqc-dev
vq fetch HOST JOBID --name job.qvf -o results/
```

vq recognizes the `.qvf` suffix, records the artifact on the job spec, and
resolves the managed runtime to its installed CLI and snapshots its full git
SHA on the job automatically
(`python -m vibeqc._cli run job.qvf`, equivalent to `vibeqc run job.qvf`);
it never
tries `python job.qvf`. The in-place update settles the same file.
`fetch --name` publishes that one file atomically and leaves operational logs
server-side. `--expected-sha FULL_SHA` is an optional reproducibility
assertion, not routine operator input.

The directory compatibility form remains supported. Put runner sidecars in
the per-job scratch directory so the workspace's user artifact remains the
container:

```sh
vq submit HOST -d jobdir/ --program vibeqc-dev \
  --expected-sha 0123456789abcdef0123456789abcdef01234567 \
  -- bash -lc 'exec "$VQ_PROGRAM_BIN/vibeqc" run job.qvf \
      --output "$VQ_WORKDIR/job"'
```

See [`docs/agent_interaction.md` in vibe-queue](https://github.com/vibe-qc/vibe-queue/blob/main/docs/agent_interaction.md) for scheduler-wrapper routing,
forced reruns, QVF lifecycle JSON, and the chemistry-failure versus
queue-process-failure distinction.

`scf_history` carries one JSON member, `iterations`:

```jsonc
{
  "iterations": [
    {"iter": 1, "energy_eh": -75.0, "delta_e": 1.0, "diis_error": 1e-2}
  ]
}
```

Fields are conventional in v1. Consumers should degrade gracefully when
a solver does not report DIIS error or energy deltas.

Beyond section-level records, the manifest root may carry additional
metadata as documented in § 2.1:

* **`thermochemistry`**, ZPVE, enthalpy, entropy, Gibbs free energy.
* **`dipole_moment`**, Total and vector dipole moment in Debye.
* **`constraints`**, Geometry optimization constraints (frozen atoms,
  distance/angle/torsion targets).

These are root-level objects, not sections, because they describe the
calculation as a whole rather than carrying binary payloads. A consumer
may display them in the application bar, a property panel, or omit them
gracefully.

## 5. Extension And Versioning Model

QVF v1 establishes a formal extension governance model built on four
mechanisms: the root `extensions` block, the per-section `critical`
flag, the per-section `schema_uri`, and a minor-version convention for
adding canonical optional kinds.

### 5.1 `qvf_version` Policy

`qvf_version` is an integer. v1.0 = `1`; v1.1 = `1`; v1.2 = `1`.
The extension governance and new canonical kinds are backward-compatible
additions.
Major version bumps (`qvf_version: 2`, …) indicate breaking changes:
removed kinds, mandatory-become-optional inversions, or incompatible
unit changes. Minor version is not encoded in the integer; instead,
consumers detect capability from the extension block and supported-kind
checking.

```{admonition} `qvf_version: 2` was withdrawn (governance ruling, 2026-07-10)
:class: warning

vibe-qc v0.10.0 through v0.15.x stamped `qvf_version: 2` on archives containing
a periodic `reaction.path`. That bump was a policy violation: the entire
schema-visible delta was one optional member (`lattice`, § 4.5), which is the
additive case and must not bump the version. It also obliged conforming v1-only
consumers to refuse archives vibe-qc routinely wrote.

Ruling: `lattice` is an optional member of v1; a periodic reaction path is
detected by its presence. Current producers must emit `qvf_version: 1`.
Consumers should accept existing archives labelled `qvf_version: 2` and treat
them as v1-compatible when their only extra contract is the optional periodic
path lattice metadata. See
[[`GOVERNANCE.md` in QVF](https://github.com/vibe-qc/qvf/blob/main/GOVERNANCE.md)](https://github.com/vibe-qc/qvf/blob/main/GOVERNANCE.md)
§ Version history.
```

If future major versions are needed, a `qvf_version` value > `1` means
consumers that only understand `qvf_version = 1` MUST refuse to open
the file.

### 5.2 Canonical Kinds

Canonical kinds appear in the JSON Schema and are described in § 4.
The set of canonical kinds may grow in minor revisions (new kinds
added, existing kinds extended with optional members). No canonical
kind may be removed or have its required members changed without a
major version bump.

### 5.3 Vendor Namespace (`x_<vendor>.*`)

Any section with a kind matching `x_<vendor>.<specific>` is a vendor
extension. The `<vendor>` part should be a DNS-like identifier
(e.g. `x_vibeqc`, `x_pyscf`, `x_vasp`, `x_orca`).

- Vendor members must still be valid JSON or binary member specs.
- A consumer that does not implement a vendor section must list it as present
  but unsupported, unless the extension is declared as `critical: true` (see
  § 5.5).
- A consumer may opt in to a known vendor section and render it directly or use
  it as overlay metadata. For example, vibe-view recognizes
  `x_ccm.wannier_centers` as optional markers on structure and volume panels.
- Unknown non-vendor kinds (no `x_` prefix, not in the canonical
  registry) are not schema-valid QVF and should be reported as errors.

### 5.4 Root `extensions` Block

The manifest root may carry an `extensions` object that declares which
vendor extensions are in use and their contracts:

```jsonc
{
  "extensions": {
    "x_vendor_ecp": {
      "version": "1.0",
      "schema_uri": "https://vendor.example.org/qvf/ecp.schema.json",
      "critical": false
    },
    "x_other_plugin": {
      "version": "2.1",
      "critical": true
    }
  }
}
```

Each key is a vendor namespace prefix (without the `x_`, the `x_` is
implied from the section kinds). The value object has:

| Field | Required | Meaning |
|---|---:|---|
| `version` | yes | Vendor's own version string for this extension. |
| `schema_uri` | no | URI of a JSON Schema that validates this extension's section members. |
| `critical` | no | If `true`, any consumer that does not support this extension MUST refuse to open the archive. Default `false`. |

### 5.5 Per-section `critical` Flag

Every section object in `sections[]` may carry a `critical` boolean
(default `false`). If `critical: true` and the consumer does not
recognize or support that `kind`, the consumer MUST refuse to open
the file entirely rather than silently ignoring the section.

This prevents data loss in workflows where an extension carries
information essential to correct interpretation (e.g., an ECP
specification that changes the effective number of electrons, or
a custom energy correction that alters displayed energies).

### 5.6 Minor-Version Convention

Canonical kinds may be added in minor revisions. A producer targeting
`qvf_version: 1` may emit any canonical kind defined in the current
schema. A consumer targeting `qvf_version: 1` must accept any file
where it either supports the kind or the kind is non-critical.

There is no explicit `minor_version` field. Extensions are the
mechanism for signaling non-canonical capability.

### 5.7 Registry Process For Promoting Vendor Kinds

To promote an `x_<vendor>.*` kind to canonical status:

1. The kind is used in production by at least two independent
   producers for at least one minor release cycle.
2. A specification PR adds the kind to § 4, the JSON Schema
   `Section.oneOf` branches, and updates the validator.
3. The kind's member contract is stable across the two producers.
4. At least one reference consumer implements rendering support.

This process is intentionally lightweight in v1.x to encourage rapid
adoption, and may be formalized in v2.

## 6. Producer Contract

A conforming producer must:

- write a valid ZIP archive with `manifest.json`;
- set `qvf_version` to `1`;
- include `source.program`, `source.version`, and `source.calculation`;
- use unique section ids;
- use canonical kinds or `x_<vendor>.*`;
- include only manifest member paths that exist in the ZIP;
- compute sha256 over the exact bytes stored in the ZIP;
- write numeric fields in the units specified above;
- write binary payloads matching declared dtype and shape;
- ensure JSON members parse as UTF-8 JSON;
- declare every `x_<vendor>` namespace used in `sections[]` in the root
  `extensions` block when `critical: true` is set on any section in
  that namespace;
- set `critical: true` on a section only if the producer truly requires
  consumer support for correct interpretation (see § 5.5).

A producer should:

- include `provenance` whenever the calculation context is known;
- include `citations` when results depend on citable methods, bases,
  functionals, ECPs, libraries, or external data;
- use `volume.generic` only when no specific `volume.*` kind applies;
- use vendor namespaces for experimental payloads;
- ship small examples and run `validate_qvf()` in tests.

## 7. Consumer Contract

A conforming consumer must:

- read `manifest.json` by name rather than assuming ZIP entry order;
- validate or at least sanity-check `qvf_version`, `source`, and
  `sections`;
- verify sha256 before using a member's bytes;
- interpret only section kinds it supports;
- report unsupported vendor sections without corrupting or
  misinterpreting them;
- respect declared dtype, shape, and units;
- check per-section `critical` flag: if `critical: true` and the kind
  is not supported, refuse to open the file;
- check root `extensions` block for `critical: true` extensions that
  are not supported and refuse accordingly.

A consumer may require specific kinds for a particular operation. For
example, a band plotter may refuse a file without `bands`, while a
structure viewer may open the same file and ignore all scalar fields.

## 8. vibe-view Reference Implementation

[vibe-view](https://github.com/vibe-qc/vibe-view) is an independent
consumer in its own repository. The engine, viewer, and QVF reference toolkit
implement the published format independently and agree through conformance
checks. QVF is a format reference, not a runtime dependency of vibe-qc.

Install the viewer from its own checkout:

```sh
git clone https://github.com/vibe-qc/vibe-view.git
cd vibe-view
./scripts/install.sh
.venv/bin/vibe-view open my-calculation.qvf
```



Viewer modules in the separate vibe-view repository:

```text
src/vibeview/                 # in the vibe-view repository
├── qvf.py                  # QVFReader: zip, schema, sha256, lazy reads
├── kinds.py                # SUPPORTED_KINDS and lazy-kind registry
├── schema.json             # checked copy of the published manifest schema
├── viewer_defaults.py      # camera, volume hints, replication
├── renderers/
│   ├── structure.py
│   ├── volume.py
│   ├── bands.py
│   ├── spectra.py
│   ├── trajectory.py
│   ├── vibrations.py
│   ├── reaction.py
│   ├── wavefunction.py
│   ├── atom_properties.py
│   ├── nmr.py
│   ├── scf_history.py
│   ├── symmetry.py
│   └── citations.py
└── app.py                  # Trame / PyVista UI
```

vibe-view uses PyVista, VTK, Trame, Plotly, Pydantic, jsonschema, and
NumPy. It lazy-loads volume binary blobs and verifies member checksums
before use.

## 9. Implementation Pointers

- Writer: [`python/vibeqc/output/formats/qvf.py`](../python/vibeqc/output/formats/qvf.py)
- Schema: [`python/vibeqc/output/formats/qvf_manifest.schema.json`](../python/vibeqc/output/formats/qvf_manifest.schema.json)
- Viewer reader: [`vibe-view/src/vibeview/qvf.py`](https://github.com/vibe-qc/vibe-view/blob/main/src/vibeview/qvf.py)
- Viewer kind registry: [`vibe-view/src/vibeview/kinds.py`](https://github.com/vibe-qc/vibe-view/blob/main/src/vibeview/kinds.py)
- User guide: [`docs/user_guide/vibe_view.md`](user_guide/vibe_view.md)
- Reference consumer example: [`docs/consumer_qvf_reference.md`](consumer_qvf_reference.md)

Relevant tests in the core checkout:

- `tests/test_qvf_writer.py`
- `tests/test_qvf_round_trip.py`
- `tests/test_qvf_writer_to_viewer.py`

Consumer tests live in the independent vibe-view checkout:

- [`tests/test_qvf.py`](https://github.com/vibe-qc/vibe-view/blob/main/tests/test_qvf.py)
- [`tests/test_renderer_behaviour.py`](https://github.com/vibe-qc/vibe-view/blob/main/tests/test_renderer_behaviour.py)
- [`tests/test_qvf_schema_identity.py`](https://github.com/vibe-qc/vibe-view/blob/main/tests/test_qvf_schema_identity.py)

Core writer-to-viewer integration tests need an explicitly installed viewer
for their consumer checks; those checks skip when it is absent. The release
`qvf-conformance` gate checks the core schema against the pinned specification
repository independently of a viewer checkout.

## 10. Open Design Items

These are intentionally not hidden:

1. **Endianness.** The current dtype names do not carry byte order.
   v1 assumes the normal little-endian scientific-Python ecosystem.
   A byte-order prefix in dtype (e.g., `">f4"`) should be added for
   cross-platform portability in a future v1.x tightening pass.
2. **Canonical k-resolved Bloch wavefunctions.** `wavefunction.gto` now covers
   molecular systems and the Γ-point of periodic systems (§ 4.6), and vibe-qc
   writes `x_vibeqc.bloch_wavefunction` for its own restricted and spin-resolved READ
   restarts, but a cross-consumer canonical visualization section
   (e.g. `wavefunction.bloch` or `wavefunction.kohn_sham`) remains future work.
3. **Richer topology payloads.** `topology.qtaim` is now a canonical
   rendered section for critical points and bond paths. ELF basin data and
   basin-integration conventions remain future v1.x work.
4. **Large time-dependent volumes.** v1 handles individual scalar
   fields well; long volumetric movies may need chunking conventions,
   a time axis in the grid descriptor, or an explicit frame-based
   volume kind.
5. **NMR schema tightening.** `spectra.nmr` is intentionally loose in
   v1 while producer conventions settle. A future v1.x tightening pass should
   define chemical-shift tensor, J-coupling tensor, and relaxation-time
   structures.
6. **Fermi-surface grid conventions.** `fermi_surface` uses a
   gamma-centered Monkhorst-Pack grid in v1. A future revision may
   need to support arbitrary k-mesh geometries (tetrahedron integration
   meshes, band-structure k-paths cross-sampled onto the FS grid).
7. **Phonon eigenvectors.** The optional `eigenvectors` member in
   `phonon_bands` uses displacement vectors in Cartesian coordinates.
   The exact normalization and phase conventions should be documented
   explicitly in a future tightening pass.
8. **EOS fit validation.** The `equation_of_state` section carries
   only the fit parameters, not the raw derivative data (pressure,
   bulk modulus as a function of volume). A future revision may add
   optional derivative arrays for interactive EOS exploration.

The format is useful now. These items define the next layer needed for
QVF to become a durable multi-code standard rather than only a strong
vibe-qc/vibe-view interchange format.
