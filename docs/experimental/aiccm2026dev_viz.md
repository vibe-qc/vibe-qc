---
myst:
  html_meta:
    "description": "How to generate and inspect AICCM visualization outputs: QVF orbital archives for vibe-view, Gaussian cube files for VMD/VESTA, and Molden files. Covers the aiccm-viz route, the aiccm-properties route, and the QVF consumer workflow."
    "og:title": "vibe-qc - AICCM visualization (QVF, cube, Molden)"
---

# AICCM visualization

Both Γ-CCM and χ-CCM[^xccm-convention] write standard visualization formats
that can be opened in vibe-view, VMD, VESTA, or any cube/Molden reader. Γ-CCM
visualizes states from the union-and-weight construction; χ-CCM visualizes
states from the finite-character construction, evaluated on its Γ-centred
character mesh. Similar file formats do not imply that the two constructions
supplied the same finite Hamiltonian. This page covers what each route writes
and how to inspect it.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM approaches. Γ-CCM uses
    the union-and-weight/Wigner--Seitz integral-weighting construction; χ-CCM
    uses the finite-translation-group character construction. They are compared
    at a declared common exchange-q=0 convention. Their distinction is not a
    choice of Coulomb kernels, and equality for a specified operator/route is
    evidence to establish, not a naming premise.

## Γ-CCM visualization

### The `aiccm-viz` route (full orbital export)

The `aiccm-viz` route runs an HF SCF and writes **two** complete orbital
sets - the canonical crystalline orbitals (COs) and the localized
Wannier orbitals - plus the density:

```bash
cd studies/aiccm-2026
python run_case.py c-diamond aiccm-viz
```

Outputs (written to the output directory):

| file | contents | open with |
|---|---|---|
| `<system>_canonical.qvf` | Crystalline orbitals (canonical HF) | vibe-view |
| `<system>_wannier.qvf` | Localized Wannier orbitals (Pipek-Mezey) | vibe-view |
| `<system>_density.cube` | HF density on a uniform grid | VMD, VESTA |
| `<system>_co_<i>.cube` | Canonical orbital *i* on a grid | VMD, VESTA |
| `<system>_wannier_<i>.cube` | Wannier orbital *i* on a grid | VMD, VESTA |
| `<system>.molden` | All orbitals in Molden format | Molden, Jmol |

The JSON result (`<system>__aiccm-viz.json`) records the Wannier
centers and spreads, the localization objective, and the
unitary-invariance density residual (must be ~1e-15).

### The `aiccm-properties` route (density + properties .qvf)

```bash
python run_case.py c-diamond aiccm-properties
```

Writes:

| file | contents |
|---|---|
| `<system>_props_canonical.qvf` | Crystalline orbitals + density + per-atom properties |
| `<system>_props_density.cube` | HF density cube |

### Python API

```python
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.scf import run_ccm_rhf
from vibeqc.output.formats.qvf import write_qvf

ccm = CCMSystem(system, (2, 2, 2), "sto-3g")
scf = run_ccm_rhf(ccm, method="aiccm2026dev-a")

# Write a QVF archive with the HF density and orbitals
write_qvf(
    "my_system.qvf",
    molecule=ccm.supercell,
    mo_coefficients=scf.mo_coeffs,
    mo_energies=scf.mo_energies,
    density_matrix=scf.density,
    basis_name="sto-3g",
)
```

The QVF format is documented in the
[QVF specification](../design_qvf_format.md) and
[consumer reference](../consumer_qvf_reference.md).

## χ-CCM visualization

χ-CCM writes standard output formats through `run_periodic_job` with the
usual output flags:

```python
result = vq.run_periodic_job(
    system, basis,
    method="RHF",
    jk_method="aiccm2026dev-b",
    aiccm_lattice_extension=(2, 2, 2),
    aiccm_backend="ri",
    write_molden_file=True,
    output_qvf=True,
)
```

Enable `write_molden_file=True` and `output_qvf=True` on any
`run_periodic_job` call. The output directory receives:

| file | contents |
|---|---|
| `<jobname>.molden` | Orbitals in Molden format |
| `<jobname>.qvf` | QVF orbital archive |

The `aiccm2026dev_b_band_structure` function also writes band-structure
data that can be plotted with the standard band-structure tools
(`vibeqc.bands`).

Two small chi-CCM-B QVF fixtures are bundled with the documentation:
a 3D vacuum-padded H-chain and a 3D H2-pair, both run with the RI backend and emitted with
Wannier-centre overlays. Their reproducible inputs, full `.out` logs,
sanitized `.system` manifests, QVFs, and vibe-view PNG captures are in
the {download}`chi-CCM-B fixture README <../_static/examples/chi-ccm-b-qvf/README.md>`.

These fixtures were generated at pre-D89 source `d746d238` (shown as
`d746d23` in the archived output). They are visualization-only fixtures. Their
records lack the normative D89 approach identity fields, so they are not
construction-comparison or numerical-validation evidence.
Open them directly with:

```bash
vibe-view open docs/_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-hchain-ri-n4-wannier.qvf
vibe-view open docs/_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-h2pair-3d-ri-n2-wannier.qvf
```

## Opening a QVF in vibe-view

vibe-view is the GPU-accelerated 3-D viewer for QVF archives. From the
vibe-qc checkout:

```bash
vibe-view open <path/to/file.qvf>
```

Or from within Python:

```python
from vibeview.qvf import QVFReader

reader = QVFReader("c-diamond_canonical.qvf")
print(len(reader.sections), "sections")
```

The vibe-view [handover](https://github.com/vibe-qc/vibe-view/blob/main/HANDOVER.md) documents the
current capabilities (orbital isosurfaces, density, bond analysis,
COOP/COHP).

## Opening a cube file in VMD / VESTA

Cube files are plain-text volumetric grids. Any standard tool reads them:

```bash
# VMD
vmd c-diamond_density.cube

# VESTA (GUI)
open c-diamond_density.cube
```

For orbital cubes, VMD can display isosurfaces of individual orbitals:

```tcl
# VMD Tcl console
mol new c-diamond_co_0.cube
mol addrep top
mol modstyle 0 top Isosurface 0.05 0 0 0 1 1
```

## Rendering for publication

The QVF → vibe-view pipeline is the recommended path for paper and cover
figures:

1. Run `aiccm-viz` on the target system
2. Open the `.qvf` in vibe-view
3. Choose the orbital, isovalue, and colour map
4. Export a high-resolution PNG or the camera state for reproducibility

The `.cube` files are the fallback when vibe-view is not available or
when you need VESTA's crystallographic rendering (bonds, polyhedra,
unit-cell outline).

## See also

- [QVF format specification](../design_qvf_format.md)
- [QVF consumer reference](../consumer_qvf_reference.md)
- [Archived QVF writer handover](https://vibe-qc.com/docs/)
- [vibe-view handover](https://github.com/vibe-qc/vibe-view/blob/main/HANDOVER.md)
- [Orbital visualization tutorial](../tutorial/orbital_visualization.md)
- [Periodic orbital cubes tutorial](../tutorial/periodic_orbital_cubes.md)
- [vibe-view walkthrough](../tutorial/vibe_view_walkthrough.md)
