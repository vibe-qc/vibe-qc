# chi-CCM-B periodic QVF visualization fixtures

These files are the public, documentation-safe copies of the two
chi-CCM-B / `aiccm2026dev-b` QVF archives validated with vibe-view on
2026-07-15.

They were generated at pre-D89 source `d746d238` (reported as `d746d23` in
the archived output). Their density/orbital payloads and viewer captures are
retained only as visualization fixtures. The `.out`, `.system`, and QVF
convention metadata predate the normative D89 `ccm_approach`,
`ccm_construction`, and `evaluation_representation` fields. Consequently these
artifacts are not current construction-comparison or numerical-validation
evidence and must not be read as such.

The `.qvf` archives were copied with the numerical payloads unchanged and
with local host metadata in `manifest.json` replaced by `<redacted>`. The
`.system` sidecars are likewise path- and host-sanitized for the public
documentation tree. The `.out` files are the full verbose SCF logs.

## Files

- [artifact-status.json](artifact-status.json): machine-readable artifact status
- [screenshot-status.json](screenshot-status.json): machine-readable vibe-view capture status

| Case | Input | Output log | System manifest | QVF |
|---|---|---|---|---|
| 3D vacuum-padded H-chain, RI, `aiccm_lattice_extension=(4,1,1)` | [input-chi-ccm-b-hchain-ri-n4-wannier.py](input-chi-ccm-b-hchain-ri-n4-wannier.py) | [output-chi-ccm-b-hchain-ri-n4-wannier.out](output-chi-ccm-b-hchain-ri-n4-wannier.out) | [output-chi-ccm-b-hchain-ri-n4-wannier.system](output-chi-ccm-b-hchain-ri-n4-wannier.system) | [output-chi-ccm-b-hchain-ri-n4-wannier.qvf](output-chi-ccm-b-hchain-ri-n4-wannier.qvf) |
| 3D H2-pair, RI, `aiccm_lattice_extension=(2,1,1)` | [input-chi-ccm-b-h2pair-3d-ri-n2-wannier.py](input-chi-ccm-b-h2pair-3d-ri-n2-wannier.py) | [output-chi-ccm-b-h2pair-3d-ri-n2-wannier.out](output-chi-ccm-b-h2pair-3d-ri-n2-wannier.out) | [output-chi-ccm-b-h2pair-3d-ri-n2-wannier.system](output-chi-ccm-b-h2pair-3d-ri-n2-wannier.system) | [output-chi-ccm-b-h2pair-3d-ri-n2-wannier.qvf](output-chi-ccm-b-h2pair-3d-ri-n2-wannier.qvf) |

## vibe-view captures

- [vibe-view-contact-sheet.png](vibe-view-contact-sheet.png)
- H-chain: [structure](hchain-structure.png), [replicated structure](hchain-structure-replicated.png), [density](hchain-density.png), [replicated density](hchain-density-replicated.png), [HOMO](hchain-homo.png), [LUMO](hchain-lumo.png), [Wannier overlay](hchain-wannier-overlay.png)
- H2-pair: [structure](h2pair-structure.png), [replicated structure](h2pair-structure-replicated.png), [density](h2pair-density.png), [replicated density](h2pair-density-replicated.png), [HOMO](h2pair-homo.png), [LUMO](h2pair-lumo.png), [Wannier overlay](h2pair-wannier-overlay.png)

## Validation summary

- H-chain: `pbc=[true,true,true]`, dimensionality `3`, full BvK
  supercell lattice `10.583544 x 10.583544 x 10.583544` Angstrom, density
  grid `40 x 40 x 40`, density integral `8.001083` electrons, two
  orbital volumes, four `x_ccm.wannier_centers`.
- H2-pair: `pbc=[true,true,true]`, dimensionality `3`, full BvK
  supercell lattice `12.700253 x 6.350127 x 6.350127` Angstrom, density
  grid `48 x 24 x 24`, density integral `4.000538` electrons, two
  orbital volumes, two `x_ccm.wannier_centers`.
- vibe-view `capture-selftest` exited 0. The section captures, replicated
  captures, orbital surfaces, and Wannier overlays rendered nonblank.
