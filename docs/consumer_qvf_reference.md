# QVF consumer reference: reading a .qvf file in Python

The manifest shape matches the vibe-view consumer ([`src/vibeview/qvf.py` in vibe-view](https://github.com/vibe-qc/vibe-view/blob/main/src/vibeview/qvf.py)).

```{important}
**Viewer support tracks writer support, with one rule.** vibe-view
renders every implemented writer kind it has a renderer for
(see `SUPPORTED_KINDS` in the viewer repository's `src/vibeview/kinds.py`). Unknown
vendor-namespace (`x_<vendor>.*`) sections, and any reserved kinds not
yet wired into a renderer, are classified as "skipped, unsupported" by
the viewer and do not prevent the archive from opening (§ 2.5 Rule 2 of
`design_qvf_format.md`). Known vendor overlays, such as
`x_ccm.wannier_centers`, may still be rendered by a consumer that
recognizes their schema. The full writer / viewer matrix is in the
content-kinds table of the design doc.

Use the manifest as the single source of truth for "what is in
this archive". The viewer's status panel reports which of those
sections are actually being rendered in the current session.
```

```python
import zipfile, json, hashlib
import numpy as np

# ── Open ───────────────────────────────────────────────────────────────
path = "h2o.qvf"
zf = zipfile.ZipFile(path, "r")
manifest = json.loads(zf.read("manifest.json"))

print(f"QVF v{manifest['qvf_version']} from {manifest['source']['program']}")

# ── Verify sha256 of every member ──────────────────────────────────────
for section in manifest["sections"]:
    for _key, member in section.get("members", {}).items():
        sha = member.get("sha256")
        if sha is not None:
            got = hashlib.sha256(zf.read(member["path"])).hexdigest()
            assert got == sha, f"sha256 mismatch for {member['path']}"

# ── Structure → atoms ──────────────────────────────────────────────────
for s in manifest["sections"]:
    if s["kind"] == "structure":
        struct = json.loads(zf.read(s["members"]["structure"]["path"]))
        for a in struct["atoms"]:
            print(f"  {a['symbol']} at {a['position']}")
        print(f"  pbc={struct['pbc']}")
        if "lattice_vectors" in struct:
            print(f"  lattice={struct['lattice_vectors']}")

# ── Volume.density → numpy array ──────────────────────────────────────
for s in manifest["sections"]:
    if s["kind"] == "volume.density":
        dm = s["members"]["data"]
        raw = zf.read(dm["path"])
        grid_data = np.frombuffer(raw, dtype=np.float32).reshape(dm["shape"])
        # Grid descriptor is a JSON member
        g = json.loads(zf.read(s["members"]["grid"]["path"]))
        print(f"Density: {dm['shape']}, origin={g['origin']}")

# ── Vibrations ────────────────────────────────────────────────────────
for s in manifest["sections"]:
    if s["kind"] == "vibrations":
        meta = json.loads(zf.read(s["members"]["metadata"]["path"]))
        print(f"Frequencies: {len(meta['frequencies'])} modes")

zf.close()
```

## Sample archives for consumer tests

The documentation tree includes a small static fixture bundle for periodic QVF
consumers:

* {download}`fixture README <_static/examples/chi-ccm-b-qvf/README.md>`
* [`3D vacuum-padded chi-CCM-B H-chain QVF`](_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-hchain-ri-n4-wannier.qvf)
* [`3D chi-CCM-B H2-pair QVF`](_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-h2pair-3d-ri-n2-wannier.qvf)

These archives are useful regression inputs because they combine periodic
`structure.lattice_vectors`, torus-aligned `volume.density` and
`volume.orbital` grids, finite-BvK convention metadata, and
`x_ccm.wannier_centers` overlays. The same bundle includes the input scripts,
full `.out` logs, sanitized `.system` manifests, and headless vibe-view PNG
captures linked from [the example-output catalog](example_outputs.md#static-reference-bundles).

## Consumer contract (v1)

The normative contract belongs to the [QVF format repository](https://github.com/vibe-qc/qvf).
The core's schema at
[`python/vibeqc/output/formats/qvf_manifest.schema.json`](../python/vibeqc/output/formats/qvf_manifest.schema.json)
and the viewer's schema are separate checked copies. Each implementation is
validated against the published specification and conformance corpus, without
a shared runtime library or cross-repository symlink. This list is a
human-readable summary, not a replacement specification.

1. **Kind strings** must come from the registered v1 set (§ 1.4 of
   `design_qvf_format.md`) or the `x_<vendor>.*` namespace.
2. **Every member** in `members` has `path`, `format` (`"json"` |
   `"binary"`), and `sha256`. Binary members additionally have `dtype`
   (a numpy dtype name) and `shape` (rank-N integer array).
3. **Structure** has a JSON member `"structure"`; periodic
   structures also have `lattice_vectors` and `pbc=[true,true,true]`.
4. **Volumes (`volume.density`, `volume.orbital`, `volume.spin`,
   `volume.elf`, `volume.difference`, `volume.generic`)** have binary
   `"data"` + JSON `"grid"` members. The grid JSON carries `origin`,
   `voxel_vectors`, `shape` (bohr units per design § 1.3a; vibe-view
   converts to Å at its PyVista boundary). `volume.difference` may
   additionally carry `operand_a` and `operand_b` (string section
   ids that must resolve, per `dependentRequired`). `volume.generic`
   is an escape hatch for fields that don't fit the purpose-built
   kinds, producers should prefer a more specific kind when one
   applies.
5. **Spectra (`spectra.ir`, `spectra.raman`, `spectra.uvvis`,
   `spectra.ecd`, `spectra.vcd`, `spectra.nmr`, `spectra.generic`)**
   have a JSON `"spectrum"` member with `frequencies` and
   `intensities`.
6. **Vibrations** have JSON `"metadata"` (with `atoms` and
   `frequencies`) + binary `"displacements"` (`float64`,
   `[n_modes, n_atoms, 3]`).
7. **Trajectory** has JSON `"metadata"` (with `atoms` and `energies`)
   + binary `"coords"` (`float64`, `[n_frames, n_atoms, 3]`, Å).
8. **`reaction.path`** has the same binary layout as `trajectory`;
   the metadata JSON additionally carries `waypoints` (each with
   `frame_index`, `label`, `kind ∈ {reactant,
   transition_state, intermediate, product, point}`, optional
   `energy_eh`) and an optional `reaction_coordinate` array.
9. **`reaction.waypoints`** carries one JSON `"waypoints"` member
   plus a section-level `trajectory_ref` string naming the
   `trajectory` section it annotates (validator-checked).
10. **Bands** has JSON `"kpath"` (with `fermi` key) + binary
    `"eigenvalues"` (`float64`, `[n_spin, n_kpoints, n_bands]`, eV).
11. **`atom_properties`** carries one or more of `mulliken_charge`,
    `loewdin_charge`, `hirshfeld_charge`, `spin_population`, each
    `float64 [n_atoms]`.
12. **`citations`** carries a binary `"references"` member (BibTeX
    bytes, UTF-8).
    **`run.record`** likewise carries opaque binary text members:
    `"input"` (the verbatim program input) and `"log"` (the full
    program output), both UTF-8, plus an optional JSON `"files"`
    index and `attachment.*` members (arbitrary bytes). The
    section-level `program` string (required) names the code that
    ran, independent of root `source.program`.
    **`job.spec`** carries one JSON `"spec"` member: the declarative
    JobSpecPayload (`job_type` required, `"molecular"` |
    `"periodic"`; optional `method`, `basis`, `functional`,
    `charge`, `multiplicity`, `kpoints`, `tasks`, `options`)
    describing the calculation the archive *requests*. Paired with
    `provenance.run_status = "pending"` it marks a container that
    has not yet run. The payload is data, never code.
13. **`bonds`** carries one JSON `"bonds"` member with
    `{"pairs": [{"i", "j", "order"}, ...]}`.
14. **`scf_history`** carries one JSON `"iterations"` member with
    `{"iterations": [{"iter", "energy_eh", ...}, ...]}`.
15. **`structure.symmetry`** carries one JSON `"data"` member
    (spglib output).
16. **Wavefunction (`wavefunction.gto`)** has JSON `"basis"` (with
    `structure_ref`, `pure`, `n_ao`, `shells`) + JSON
    `"mo_metadata"` (with `spin`, `orbital_kind`, energies /
    occupations either at top level for `restricted` or under
    `alpha`/`beta` for `unrestricted`) + binary `"mo_coefficients"`
    (restricted) **or** `"mo_coefficients_alpha"` +
    `"mo_coefficients_beta"` (unrestricted), each row-major
    `float64` of shape `[n_mo, n_ao]`. Molecular and Gamma-point
    periodic in v1. Shell coefficients apply to **normalized**
    primitive Gaussians (see design doc Sec. 4.6 for the formula).
17. **Manifest root** may carry `viewer_defaults` with `auto_open`
    (list of section ids), per-section render hints (isovalue,
    colormap, opacity, replication), and `bookmarks` (ordered list
    of `{name, camera}` using the VTK camera model).
18. **Unknown / vendor sections** (Rule 2): consumers list unknown
    sections as "skipped, unsupported" and continue. They don't crash the
    open. Consumers may opt in to rendering known vendor overlays such as
    `x_ccm.wannier_centers`.

## Live / streaming checkpoints

A producer can rewrite a QVF *while the job runs* so a viewer can
hot-reload it (SCF convergence climbing, an optimization trajectory
growing, the geometry morphing to the relaxed structure). vibe-qc's
runners do this on demand: pass `checkpoint_qvf=<path>` +
`checkpoint_every=N` to `run_job` / `run_periodic_job`. A consumer
watching that path reads three optional fields:

* **`provenance.run_status`** is `"running"` while the job is in flight,
  then `"converged"` (success) or `"failed"` (crash). Watch until it is
  no longer `"running"`, then read the settled archive.
* **`provenance.checkpoint`** is `{"seq": int, "wall_time_s": float,
  "written_at": ISO-8601}` (mid-SCF snapshots may also carry
  `"scf_iteration"` and `"energy_eh"`). `seq` is **monotonic**: use it to
  tell a fresh snapshot from a stale one and to drop out-of-order reads
  without diffing bytes.
* **per-section `partial`** is `true` on a section still growing (e.g. an
  optimization `trajectory`), `absent`/`false` when settled.

These live under the open `provenance` block and on sections, so they
validate against the **current v1 manifest schema**: a v1 consumer that
ignores them still opens the file correctly. Every checkpoint write is
atomic (temp file + `os.replace`), so a reader never observes a
half-written zip; a plain "reload on mtime change" watcher is safe.

Cadence is route-dependent. Per-iteration frames (`checkpoint_every=N`)
appear on the periodic routes that stream through the shared progress
logger (Ewald, GDF, BIPOLE, and GPW), which refresh every `N` SCF cycles.
Molecular single-point SCFs expose their compiled-C++ iteration callback to
the terminal, structured-log, and `.system` manifest surfaces, but QVF
checkpointing does not consume that callback yet. Molecular jobs therefore
emit a start + terminal QVF frame only. A consumer should treat the frame
count as informative, not guaranteed, and rely on `run_status` for lifecycle.
