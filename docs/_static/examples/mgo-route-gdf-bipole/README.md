# MgO GDF/BIPOLE route-comparison fixtures

```{admonition} Stale: the two `.molden` files here are corrupt, regeneration pending
:class: warning

**Do not use `output-mgo-rhf-sto3g-gdf-k222.molden` or
`output-mgo-rhf-sto3g-bipole-k222.molden`.** Each carries 196 coefficient
lines of the form `9.93E-01+2.83E-20j`, which no Molden reader can parse.
The bug that produced them is **fixed on `main`**; only this bundle is
stale, because it was generated at v0.15.41 and has not been re-run since.

Two separate defects are visible in these files. Periodic Bloch MO
coefficients are `complex128` even at Gamma, and numpy renders a complex
scalar as `a+bj`, so the writer emitted unparsable text under a cheerful
"Molecular orbitals written to ..." log line. On top of that, the GDF file
is 62 % imaginary: it predates `_gamma_index_for_multi_k`, so it exported
whichever k-point sat at index 0 as if it were Gamma. It was never a Gamma
block at all.

**To regenerate**, submit `input-mgo-rhf-sto3g-bipole-k222.py` through `vq`
against a `vibeqc-dev` program at or after the script's `MIN_MAIN_SHA`
(`3e3eef829`, the commit that re-expresses degenerate Gamma blocks on a
real basis). The script drives both cases. A run was queued as
`f9dd6daf949d` on compute-reference and had not dispatched at the time of writing;
check with `vq status compute-reference f9dd6daf949d` or just resubmit.

**Three things about the regenerated bundle will differ from what is
committed here, all of them intended:**

* **The GDF case will have no `.population.txt` / `.population.json`.** The
  two files currently committed are not a crystal population: they are a
  molecular Mulliken analysis of a single Bloch block. Multi-k population is
  now supported only on BIPOLE, which contracts the real-space lattice
  density and the full SCF k-mesh. `artifact-status.json` needs those two
  GDF rows dropped.
* **Both `.molden` files become real** and are labelled in their `[Title]`
  as Gamma-block orbitals, home cell only. Molden has no periodic
  representation, so the file is the home-cell truncation of the Gamma
  crystalline orbital and clips where an orbital straddles a cell face. The
  `.qvf` is the faithful artefact; see
  [QVF and vibe-view](../../../visualization.md).
* **The provenance paragraph and every number under "Validation summary"
  below must be re-taken from the new run.** A local reproduction at
  v0.15.112 gave GDF `-271.7175698639` Ha (1.5e-7 from the committed value,
  14 iterations both) and BIPOLE `-271.2145297544` Ha (**1.5e-5** from the
  committed value, 23 iterations both). The BIPOLE drift is 100x the GDF
  control on the same cell and is tracked separately as
  `BIPOLE-MGO-K222-DRIFTS-100X-MORE-THAN-GDF` in
  `agentic-loop/bug-claims.md`; it is not a Molden matter.
```

These files are the public, documentation-safe copies of the MgO primitive
RHF/STO-3G route-comparison runs used by the periodic-method tutorial.

The completed GDF run was submitted through `vq` on the managed
`vibeqc-dev` program and executed with vibe-qc `0.15.41` at source
`a5ac682`. The matched BIPOLE run uses the same managed program, the
symmetry-attached payload, and the parity-test `12 bohr` cutoff contract. The
full verbose `.out`, captured stdout/stderr, input snapshots, `.system`
manifests, QVF archives, structure files, population files, and citation
files are kept for static download.

The `.system` sidecar is copied with transient scheduler paths replaced by
relative file names or `<redacted-python>`. The QVF archive and numerical
output payloads are otherwise unchanged, and the job-side `validate_qvf`
checks returned `True`.

The committed BIPOLE QVF carries structure, atom-property, bond-order, and
SCF-history sections. This historical run predates exact returned-density
export: its density-grid and DOS optional artifacts warned because the old
density output template omitted converged lattice cells, so no BIPOLE density
surface is linked here. Current full-mesh BIPOLE runs no longer use that
template and write a density volume after exact refold, AO-image, and electron
count checks succeed. This static archive remains unchanged until the fixture
is regenerated. The SCF itself converged normally.

## Files

- [artifact-status.json](artifact-status.json): machine-readable artifact status
- [screenshot-status.json](screenshot-status.json): machine-readable vibe-view capture status

| Case | Input | Output log | System manifest | QVF | Summary |
|---|---|---|---|---|---|
| MgO primitive RHF/STO-3G GDF, Monkhorst-Pack `2 x 2 x 2` | [input](input-mgo-rhf-sto3g-gdf-k222.py) | [`.out`](output-mgo-rhf-sto3g-gdf-k222.out), [stdout](stdout.txt), [stderr](stderr.txt) | [`.system`](output-mgo-rhf-sto3g-gdf-k222.system) | [`.qvf`](output-mgo-rhf-sto3g-gdf-k222.qvf) | [JSON](summary-mgo-rhf-sto3g-gdf-k222.json) |
| MgO primitive RHF/STO-3G BIPOLE, Monkhorst-Pack `2 x 2 x 2` | [input](input-mgo-rhf-sto3g-bipole-k222.py) | [`.out`](output-mgo-rhf-sto3g-bipole-k222.out), [stdout](stdout-bipole.txt), [stderr](stderr-bipole.txt) | [`.system`](output-mgo-rhf-sto3g-bipole-k222.system) | [`.qvf`](output-mgo-rhf-sto3g-bipole-k222.qvf) | [JSON](summary-mgo-rhf-sto3g-bipole-k222.json) |

## vibe-view captures

- GDF: [structure](captures/mgo-gdf-structure.png), [density](captures/mgo-gdf-density.png)
- BIPOLE: [structure](captures/mgo-bipole-structure.png)

## Validation summary

- GDF energy: `-271.7175700123166` Ha per primitive cell.
- GDF SCF: converged in `14` iterations.
- GDF HOMO/LUMO gap: `0.510904035204234` Ha.
- GDF QVF sections rendered nonblank in headless vibe-view for structure
  and density.
- BIPOLE energy: `-271.21454466172247` Ha per primitive cell.
- BIPOLE SCF: converged in `23` iterations.
- BIPOLE HOMO/LUMO gap: `0.5081894111790204` Ha.
- BIPOLE QVF rendered nonblank in headless vibe-view for structure. In this
  historical archive, density and DOS sections are absent because their
  optional artifact steps warned after the converged SCF.
