# QTAIM: topological analysis of the electron density

QTAIM (Quantum Theory of Atoms in Molecules, Bader 1985/1990) is a
topological analysis of the electron density ρ(r). It locates
stationary points (critical points) of ρ(r), classifies them by the
Hessian eigenvalue sign pattern (nuclear, bond, ring, cage), and
traces gradient paths (bond paths) connecting bonded atoms.

```{admonition} Feature status
:class: note

`vibeqc.qtaim.qtaim_analysis()` ships critical-point search via
finite-difference Hessian and gradient-path bond tracing. The
implementation is pure Python with NumPy, calling vibe-qc's
`evaluate_ao_with_gradient` native binding for density and gradient
evaluation. QVF output (`topology.qtaim` section) is supported
through `qtaim_result_to_qvf()`.
```

## Algorithm sketch

1. Build a uniform grid around the molecule.
2. Evaluate ρ(r) and ∇ρ(r) on the grid via `evaluate_ao_with_gradient`.
3. Finite-difference the gradient to get the density Hessian H_ρ(r).
4. Scan grid cells for sign changes in ∇ρ -- seed points.
5. Newton-Raphson refine each seed to a stationary point.
6. Classify by Hessian eigenvalues: (3,-3) = ncp, (3,-1) = bcp,
   (3,+1) = rcp, (3,+3) = ccp.
7. For each BCP, trace gradient paths to the two connected atoms.

## Quick start

```python
from vibeqc.qtaim import qtaim_analysis, qtaim_result_to_qvf
import vibeqc as vq

# Run a molecular SCF first
mol = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0]),
                   vq.Atom(1, [0.0, 0.757, -0.469]),
                   vq.Atom(1, [0.0, -0.757, -0.469])])
basis = vq.BasisSet(mol, "6-31g*")
result = vq.run_rhf(mol, basis)

# Run QTAIM analysis
qtaim = qtaim_analysis(result, basis, mol, grid_spacing=0.1)

for cp in qtaim.critical_points:
    print(f"{cp.type}: rho={cp.rho:.4f} e/A^3, laplacian={cp.laplacian:.4f} e/A^5")

for bp in qtaim.bond_paths:
    print(f"Bond {bp.atoms[0]}-{bp.atoms[1]}: {len(bp.path)} points")

# Export as QVF-ready dict
qvf_data = qtaim_result_to_qvf(qtaim)
```

## Data types

| Class | Attributes |
|---|---|
| `CriticalPoint` | `type` ("ncp"/"bcp"/"rcp"/"ccp"), `position` (A), `rho` (e/A^3), `laplacian` (e/A^5), `ellipticity` (BCP only), `atom_pair` (BCP only) |
| `BondPath` | `atoms` ((i, j) indices), `path` ((N,3) points in A) |
| `QTAIMResult` | `critical_points` (list), `bond_paths` (list), `grid_spacing` (bohr) |

## Where to look in the code

| Layer | File |
|---|---|
| Python API | `python/vibeqc/qtaim.py` |
| AO evaluation (density + gradient) | `cpp/src/bindings.cpp` (`evaluate_ao_with_gradient`) |
| Grid infrastructure | `python/vibeqc/cube.py` |
| QVF topology writer | `python/vibeqc/output/formats/qvf.py` (`topology.qtaim` section) |
| Citation route | `python/vibeqc/output/citations/database.toml` (routes: `qtaim`, `bader`) |
| Tests | `tests/test_qtaim.py` |

## Known limitations

- The Hessian uses finite differences on the gradient grid. An
  `evaluate_ao_with_hessian` native binding would replace this with
  analytic second derivatives, improving accuracy near nuclei.
- Atomic basin integration (the full AIM partitioning) is not yet
  implemented; the current scope is critical-point search and bond-path
  tracing.

## References

- Bader, R. F. W., *Atoms in Molecules*, Oxford University Press (1990).
- Bader, R. F. W., *Acc. Chem. Res.* **18**, 9-15 (1985) -- original
  QTAIM formulation.
- Lu, T. & Chen, F., *J. Comput. Chem.* **33**, 580-592 (2012) --
  MultiWFN reference implementation.

Citations are automatically included in the output manifest when
QTAIM analysis is performed; see `docs/user_guide/citations.md`.

## See also

- [Volumetric data (cube / molden)](volumetric_data.md) -- density and
  orbital export.
- [Post-SCF properties](properties.md) -- Mulliken / Lowdin / Mayer
  bond-order analysis.
- [QVF file format](../tutorial/qvf_file_format.md) -- the container
  that carries QTAIM topology data.
