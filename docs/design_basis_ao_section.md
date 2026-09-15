# Design, `basis.ao` section for basis-function visualization

Viewer paths in this design refer to the independent [vibe-view repository](https://github.com/vibe-qc/vibe-view). The core writer remains in vibe-qc.

> **Status**: Implemented and shipped: writer, reader, and the
> vibe-view sidebar/AO-picker renderer all exist; the § 10 "Reserved
> future kinds" table below (`basis.overlap_map`, `basis.completeness`,
> `basis.difference`) remains genuinely unimplemented.
> **Target**: QVF v1 (backward-compatible addition as an optional canonical kind).
> **Depends on**: existing QVF writer (`qvf.py`), cube grid evaluators (`cube.py`),
>               vibe-view volume renderer + sidebar grouping pattern.

## 1. Motivation

Basis-set optimization workflows need to answer spatial questions
that energy tables alone cannot:

- *"Is this diffuse exponent bleeding 3-4 unit cells away?"*
- *"Does the contraction give enough angular resolution at the atom?"*
- *"Are there pathological shell overlaps across cell boundaries?"*
- *"Did the optimizer shrink the rogue diffuse, or did it lose needed flexibility?"*

Answering these requires **per-AO isosurfaces**, the ability to select
a single atomic orbital (primitive or contracted shell), render its
isosurface in the crystal lattice, and compare before/after optimization.

This document defines the `basis.ao` section kind that delivers that
capability through the existing QVF + vibe-view pipeline.

## 2. Section kind: `basis.ao`

`basis.ao` is a volume-like section that carries **one** atomic orbital's
scalar field on a 3-D grid, plus structured metadata that describes
which basis function it represents.

### 2.1 Manifest shape

```jsonc
{
  "id": "ao_H_1s_s0_p0",
  "kind": "basis.ao",
  "label": "H 1s — exp=0.12  coeff=0.15",
  "ao_metadata": {
    "atom_index": 0,
    "atom_symbol": "H",
    "shell_index": 0,
    "primitive_index": 0,
    "angular_momentum": [0, 0],
    "shell_type": "s",
    "exponent": 0.12,
    "coefficient": 0.15,
    "is_primitive": true,
    "is_contracted": false,
    "ao_index": 0,
    "basis_label": "pob-TZVP"
  },
  "members": {
    "grid": {
      "path": "basis_ao/H_1s_s0_p0_grid.json",
      "format": "json",
      "sha256": "…"
    },
    "data": {
      "path": "basis_ao/H_1s_s0_p0.dat",
      "format": "binary",
      "dtype": "float32",
      "shape": [80, 80, 80],
      "sha256": "…"
    }
  }
}
```

### 2.2 `ao_metadata` fields

All fields are required unless marked optional.  This is a section-level
object (sibling to `id`, `kind`, `members`), **not** a zip member, the
metadata is embedded directly in `manifest.json` for random-access
discovery without reading any payload bytes.

| Field | Type | Description |
|-------|------|-------------|
| `atom_index` | `int ≥ 0` | Zero-based index into the `structure.atoms` array. The AO is centered at this atom. |
| `atom_symbol` | `string` | Element symbol (denormalized convenience, e.g. `"O"`). |
| `shell_index` | `int ≥ 0` | Which contracted shell within the atom's basis block (0-based). |
| `primitive_index` | `int ≥ 0` | Which primitive Gaussian within the shell. For contracted shells (when `is_contracted` is true), this is the first primitive's index. |
| `angular_momentum` | `[int, int]` | `[l, m]` quantum numbers. `l ∈ {0,1,2,3,4,5}` for s/p/d/f/g/h. `m ∈ [-l, +l]` is the real-spherical-harmonic index and is meaningful **only** when `cartesian_powers` is absent; when it is present the component is a Cartesian monomial with no `m`, this element is `0` by convention, and a consumer must ignore it. |
| `cartesian_powers` | `[int, int, int]` (optional) | `[lx, ly, lz]` of the Cartesian monomial this AO is, with `lx+ly+lz == angular_momentum[0]`. Present **iff** the AO belongs to a Cartesian (`pure: false`) shell, which has `(l+1)(l+2)/2` components rather than `2l+1` and therefore no magnetic quantum number. Ordered per QVF spec Appendix A.2, so a d shell is `xx, xy, xz, yy, yz, zz`. |
| `shell_type` | `string` | One of `"s"`, `"p"`, `"sp"`, `"d"`, `"f"`, `"g"`, `"h"`. The `"sp"` label denotes a CRYSTAL-style shared-exponent SP shell. |
| `exponent` | `float > 0` | Gaussian exponent α in bohr⁻². |
| `coefficient` | `float` | Normalized contraction coefficient for this primitive.  For contracted shells this is the contraction coefficient of the first primitive. |
| `is_primitive` | `bool` | `true` if this AO is a single primitive Gaussian (χ = c · N · r^l · Y_{lm} · exp(-α r²)). |
| `is_contracted` | `bool` | `true` if this AO is the full contracted shell (χ = Σ_p c_p · N_p · r^l · Y_{lm} · exp(-α_p r²)). |
| `ao_index` | `int ≥ 0` | Global AO index (0-based) within the full basis set. Used to cross-reference the `wavefunction.gto` section when both are present. |
| `basis_label` | `string` (optional) | Human-readable basis set name, e.g. `"pob-TZVP"`. |

**Invariants:**

- `is_primitive` and `is_contracted` are mutually exclusive.  Exactly one is `true`.
- `cartesian_powers` is present exactly when the shell is Cartesian.  Its
  absence is what tells a consumer that `angular_momentum[1]` is a real `m`.
  Deriving `m` as *in-shell index* − `l` is correct only for a pure shell;
  on a Cartesian shell it runs past `+l` (a d shell yields `-2 … +3`).
- When `is_primitive` is `true`, `primitive_index` names the specific primitive within the shell.
- When `is_contracted` is `true`, `primitive_index` is informational (identifies the first primitive of that shell), and the `exponent` / `coefficient` values refer to that first primitive as a representative.
- `ao_index` is globally unique across all `basis.ao` sections referencing the same basis set.

### 2.3 Members

| Member | Format | Required | Description |
|--------|--------|----------|-------------|
| `grid` | `json` | yes | Same grid descriptor as all `volume.*` sections: `origin` (bohr, 3-vector), `voxel_vectors` (bohr, 3×3), `shape` (3-integers). |
| `data` | `binary`, `float32` or `float64` | yes | Rank-3 array, point-centered.  The scalar field χ(r) evaluated at each voxel. |

The grid + data contract is identical to `volume.orbital` (design § 4.2).
This means vibe-view's `VolumeRenderer` can render a `basis.ao` section
with zero code changes, the renderer dispatch just needs to map
`basis.ao` → `VolumeRenderer`.

### 2.4 Id convention

Producers should use stable, human-readable ids:

```
ao_{symbol}_{shell_type}[_{component}]_s{shell_idx}_p{prim_idx}
```

`{component}` disambiguates the AOs of one shell, which otherwise share
every other field of the id: `m{m}` for a spherical component, the monomial
(`xx`, `xy`, …) for a Cartesian one.  An `s` shell has a single component
and omits the part entirely.

Examples:
- `ao_H_s_s0_p0`, H atom, s shell, primitive 0
- `ao_O_sp_s2_p1`, O atom, sp shell, primitive 1
- `ao_O_d_m-2_s3_contracted`, O atom, spherical d shell, `m = -2`, contracted
- `ao_O_d_xy_s3_contracted`, O atom, Cartesian d shell, `xy`, contracted

These ids appear in the vibe-view AO picker dropdown and in
`viewer_defaults.auto_open`.  Consumers must tolerate any valid
`^[A-Za-z0-9_.-]+$` id.  Note the charset admits `-` but not `+`, which is
why a non-negative `m` carries no sign (`m0`, `m1`).

## 3. Viewer defaults

Per-section hints in `viewer_defaults`:

```jsonc
{
  "viewer_defaults": {
    "auto_open": ["ao_O_sp_s1_p0"],
    "ao_H_s_s0_p0": {
      "isovalue": 0.03,
      "colormap": "RdBu",
      "opacity": 0.5
    }
  }
}
```

### 3.1 Sensible defaults

The producer should compute per-AO default isovalues based on the
exponent, so the first render is informative:

| Exponent range | Default isovalue | Rationale |
|---|---|---|
| α < 0.1 (very diffuse) | 0.005 | Diffuse functions have low peak amplitude; a high isovalue shows nothing. |
| 0.1 ≤ α < 1.0 (diffuse) | 0.02 |  |
| 1.0 ≤ α < 10 (moderate) | 0.05 | Standard. |
| α ≥ 10 (tight) | 0.10 | Tight functions have high peak amplitude at the nucleus. |

Default colormap: `"RdBu"` (diverging), essential for p, d, f functions
that have positive and negative lobes.

Default opacity: 0.5.

Default grid spacing: **0.15 bohr** (finer than the 0.25 bohr density
default, because individual primitives can be spatially compact).

Default grid padding: **8.0 bohr** (wider than the 4.0 bohr molecular
default, to capture diffuse-functions extending across unit cells).

### 3.2 Grid for periodic systems

For periodic systems the grid should:

- Span the **unit cell** (or a user-specified N×M×L supercell).
- Have `origin` at the unit-cell origin (atoms in the reference cell
  are inside the grid).
- Use axis-aligned `voxel_vectors` for orthogonal cells; follow lattice
  vectors for non-orthogonal cells.
- Evaluate the AO **only at the reference atom position**, do not
  replicate AOs across image cells.  vibe-view's periodic replication
  controls handle the visual tiling.

Grid origin and voxel vectors remain in **bohr** per the QVF v1 contract
(design § 1.3a).

## 4. Which AOs to export by default

The producer should accept a filter specification.  Sensible presets:

### 4.1 "Optimised" mode (default for basis-optimization workflows)

Export only the primitives that are **free parameters** in the
optimization.  The `FreeSpec` list from `vibeqc.basis_optimization.parametrise`
tells us exactly which exponents/coefficients the optimizer varied.

Additionally export the most diffuse and most contracted primitive per
shell as reference points.

### 4.2 "Valence" mode

Export only shells with principal quantum number ≥ the highest core
shell (typically: valence + polarisation shells).

### 4.3 "All" mode

Export every primitive and every contracted shell for every atom.
This produces the largest QVF, for a pob-TZVP basis on a 2-atom cell
(~30 shells, ~4 primitives/shell, ~150 AOs total), at 80³ grid points
× 4 bytes × 150 AOs ≈ 307 MB in the zip (typically ~100 MB compressed).

### 4.4 Filtering API

The convenience builder should accept:

```python
qvf_ao_data(
    basis, system_or_mol,
    *,
    atom_indices: list[int] | None = None,   # which atoms
    shell_types: list[str] | None = None,     # e.g. ["s", "p", "d"]
    ao_indices: list[int] | None = None,      # explicit AO indices
    primitive_only: bool = False,             # skip contracted shells
    contracted_only: bool = False,            # skip individual primitives
    mode: str = "all",                        # "all" | "optimised" | "valence"
    spacing: float = 0.15,                    # bohr
    padding: float = 8.0,                     # bohr
)
```

## 5. vibe-view integration

### 5.1 Sidebar

All `basis.ao` sections collapse into a single sidebar entry:

```
Basis Functions (37)         ✓
```

The "(N)" count is the number of `basis.ao` sections in the file.
Clicking it activates the first AO and reveals the AO picker in the
right panel.

### 5.2 Right panel, AO picker

The right panel shows a card with:

```
Basis Functions
┌─────────────────────────────────┐
│ Filter by atom:  [All ▼]       │
│ Filter by l:     [All ▼]       │
│ Mode:            [Primitives ▼]│
│                                 │
│ AO: [H 2s — α=0.12 ▼]          │
│ Exponent: 0.120 bohr⁻²         │
│ Coefficient: 0.150             │
│ Center: H (atom 0)             │
│ Shell type: s  (l=0, m=0)      │
│ Global AO index: 0             │
└─────────────────────────────────┘
```

The AO dropdown items have descriptive labels:

```
H 1s — α=0.120                (primitive)
H 1s — α=0.350                (primitive)
H 1s (contracted)             (contracted shell)
O 2sp — α=0.185               (primitive)
O 3d_z² — α=3.500             (primitive)
O 3d (contracted)             (contracted shell)
```

Filter dropdowns let the user narrow by atom, angular momentum, and
primitive vs. contracted.

The isosurface controls (isovalue, colormap, opacity, clip planes,
replication) are the same shared panel used by all `volume.*` sections.

### 5.3 Renderer dispatch

`basis.ao` maps to `VolumeRenderer` in `renderers/__init__.py`.
The rendering path is identical to `volume.orbital` since the member
layout (grid + data) is the same.

### 5.4 Before/after comparison

The planned multi-file support (open user request #3 from the vibe-view
handover) is the natural comparison mechanism: open two QVFs (pre- and
post-optimization), select the same AO in each, switch between them in
the file dropdown.

## 6. Producer contract

A conforming producer must:

- Use kind `"basis.ao"`.
- Include `ao_metadata` with all required fields at the section root.
- Provide `grid` (JSON) and `data` (binary, 3-D) members following the
  `volume.*` contract.
- Set `is_primitive` xor `is_contracted` to `true`.
- Follow the id convention or use any valid `^[A-Za-z0-9_.-]+$` id.

A producer should:

- Compute per-AO `viewer_defaults` hints with sensible isovalues
  (see § 3.1).
- Include `basis_label` when the basis set has a known name.
- Use `float32` for the data member unless `float64` precision is
  genuinely needed.
- Export only the AOs relevant to the workflow (see § 4).

## 7. Consumer contract

A conforming consumer must:

- Treat `basis.ao` as a volume section with per-AO metadata.
- Verify sha256 before rendering.
- Respect the `ao_metadata` fields for labeling and grouping.

A consumer should:

- Collapse multiple `basis.ao` sections into a single navigable group.
- Provide filtering by atom, angular momentum, and primitive/contracted.
- Display the AO metadata alongside the isosurface.

## 8. Relationship to `wavefunction.gto`

| Aspect | `wavefunction.gto` | `basis.ao` |
|--------|---------------------|------------|
| What it carries | Full basis spec + MO coefficient matrix | One pre-computed AO scalar field |
| Compute location | Consumer evaluates AOs internally | Producer evaluates, consumer renders |
| Scope | Molecular only (v1) | Molecular + periodic |
| Use case | Interactive MO explorer | Basis inspection, optimization diagnostics |
| File size | ~KB (basis JSON + MO matrix) | ~MB per AO (3-D grid) |
| Viewer renders | Isosurface on demand | Pre-computed isosurface |

The two are complementary.  A QVF may contain both.  When both are
present, `ao_metadata.ao_index` cross-references the AO in
`wavefunction.gto.basis`.

## 9. JSON Schema addition

A new `SectionBasisAO` definition branch is added to the `Section.oneOf`
array in `qvf_manifest.schema.json`:

```jsonc
"SectionBasisAO": {
  "properties": {
    "kind": {"const": "basis.ao"},
    "ao_metadata": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "atom_index", "atom_symbol", "shell_index", "primitive_index",
        "angular_momentum", "shell_type", "exponent", "coefficient",
        "is_primitive", "is_contracted", "ao_index"
      ],
      "properties": {
        "atom_index":     {"type": "integer", "minimum": 0},
        "atom_symbol":    {"type": "string", "minLength": 1, "maxLength": 3},
        "shell_index":    {"type": "integer", "minimum": 0},
        "primitive_index":{"type": "integer", "minimum": 0},
        "angular_momentum":{
          "type": "array",
          "items": {"type": "integer"},
          "minItems": 2, "maxItems": 2
        },
        "cartesian_powers":{
          "type": "array",
          "items": {"type": "integer", "minimum": 0},
          "minItems": 3, "maxItems": 3
        },
        "shell_type":     {"type": "string", "enum": ["s","p","sp","d","f","g","h"]},
        "exponent":       {"type": "number", "exclusiveMinimum": 0},
        "coefficient":    {"type": "number"},
        "is_primitive":   {"type": "boolean"},
        "is_contracted":  {"type": "boolean"},
        "ao_index":       {"type": "integer", "minimum": 0},
        "basis_label":    {"type": "string"}
      }
    },
    "members": {
      "type": "object",
      "additionalProperties": false,
      "required": ["grid", "data"],
      "properties": {
        "grid": {"$ref": "#/$defs/JsonMember"},
        "data": {"$ref": "#/$defs/Volume3DBinary"}
      }
    }
  }
}
```

## 10. Reserved future kinds

The following related kind names are reserved for forward compatibility
but are not implemented:

| Kind | Purpose |
|------|---------|
| `basis.overlap_map` | 2-D AO overlap matrix heatmap (S_{μν}) |
| `basis.completeness` | Real-space basis completeness map (sum of all AO densities on a grid) |
| `basis.difference` | Difference field between two basis sets (ρ_basis_A − ρ_basis_B) |

## 11. Implementation plan

| Step | What | Where | Depends on |
|------|------|-------|------------|
| 1 | `evaluate_ao_column(basis, ao_index, points)`, single-AO grid evaluator | `cube.py` | - |
| 2 | `make_periodic_grid(system, spacing, padding)`, unit-cell grid builder | `cube.py` | - |
| 3 | `qvf_ao_data(basis, system, ...)`, QVF-ready AO list builder | `qvf.py` | 1, 2 |
| 4 | `_write_basis_ao_section()`, section writer | `qvf.py` | 3 |
| 5 | `SectionBasisAO`, JSON Schema branch | `qvf_manifest.schema.json` | 4 |
| 6 | `basis.ao` → `_IMPLEMENTED_KINDS` + validator | `qvf.py` | 4, 5 |
| 7 | `basis.ao` → `SUPPORTED_KINDS` + `VolumeRenderer` dispatch | `src/vibeview/kinds.py` in vibe-view, `renderers/__init__.py` | - |
| 8 | Sidebar grouping ("Basis Functions (N)") + AO picker panel | `src/vibeview/app.py` in vibe-view | 7 |
| 9 | Showcase example script | `examples/vibe_view/` | 4, 8 |
| 10 | Tests: writer round-trip, schema validation, viewer activation | `tests/`, `tests/` in vibe-view | 4, 5, 8 |

Steps 1-6 are producer-side; 7-8 are consumer-side; they can proceed in
parallel once step 4 (the manifest contract) is finalized.
