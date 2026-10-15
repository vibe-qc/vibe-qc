# QTAIM: topological analysis of the electron density

QTAIM (Quantum Theory of Atoms in Molecules, {vibeqc-cite}`bader_qtaim_1985`)
is a topological analysis of the electron density ρ(r). It locates the
stationary points (critical points) of ρ(r), classifies them by the sign
pattern of the density Hessian (nuclear, bond, ring, cage), and traces the
gradient paths (bond paths) that connect bonded atoms. Critical-point search
and bond-path tracing are production on molecular SCF densities; see
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
Atomic-basin integration is not implemented (see the limitations below).

```{admonition} Feature status
:class: note

`vibeqc.qtaim.qtaim_analysis()` ships critical-point search and gradient-path
bond tracing on a uniform grid. The implementation is pure Python with NumPy
and calls vibe-qc's `evaluate_ao_with_hessian` native binding, so the density,
its gradient and its Hessian are all analytic at every grid point. QVF output
(`topology.qtaim` section) is supported through `qtaim_result_to_qvf()`.
```

## Theory

Bader's partitioning starts from the gradient vector field of the density.
Every gradient path of ρ terminates at a maximum, which in almost all cases
sits at a nucleus; the set of all paths ending at one nucleus is the atomic
basin $\Omega_A$, and neighbouring basins are separated by zero-flux
surfaces,

$$
\nabla\rho(\mathbf r)\cdot\mathbf n(\mathbf r) = 0
\quad\text{for every } \mathbf r \text{ on the surface } S(\Omega_A),
$$

so no gradient path crosses them. Integrating ρ over a basin gives the
QTAIM atomic charge, and the basins are the "atoms in molecules"; vibe-qc
does not yet integrate them, but it locates the points that define them.

Critical points are the points where $\nabla\rho = 0$. They are classified
by the eigenvalues $\lambda_1 \le \lambda_2 \le \lambda_3$ of the density
Hessian $H_{ij} = \partial^2\rho/\partial r_i\partial r_j$ through the pair
(rank, signature): the rank $\omega$ is the number of non-zero eigenvalues
and the signature $\sigma$ the sum of their signs. For a stable molecular
density every critical point has rank 3, leaving four types:

| $(\omega, \sigma)$ | name | meaning |
|---|---|---|
| (3, -3) | nuclear critical point, `ncp` | local maximum; the attractor of a basin, at or very near a nucleus |
| (3, -1) | bond critical point, `bcp` | maximum in two directions, minimum along the third: the saddle between two bonded atoms |
| (3, +1) | ring critical point, `rcp` | minimum inside a ring of bonded atoms |
| (3, +3) | cage critical point, `ccp` | local minimum enclosed by rings |

The counts obey the Poincare-Hopf relation
$n_{\mathrm{ncp}} - n_{\mathrm{bcp}} + n_{\mathrm{rcp}} - n_{\mathrm{ccp}} = 1$
for an isolated molecule, a useful completeness check on any search. A bond
path is the pair of gradient paths that leave a bcp along its unique
positive-curvature eigenvector and end at the two neighbouring nuclei; its
existence is Bader's criterion for a bonding interaction, and the bcp
properties characterise that interaction. The density $\rho_b$ at the bcp
correlates with bond order and strength; the ellipticity

$$
\varepsilon = \frac{\lambda_1}{\lambda_2} - 1
$$

measures how far the density departs from cylindrical symmetry about the
bond (near zero for single and triple bonds, large for double bonds and for
bonds about to break); and the Laplacian

$$
\nabla^2\rho = \lambda_1 + \lambda_2 + \lambda_3
$$

tells where charge is concentrated or depleted. $\nabla^2\rho < 0$ at the
bcp (the two negative curvatures dominate) marks a shared, covalent
interaction; $\nabla^2\rho > 0$ (the positive curvature along the bond
dominates) marks a closed-shell interaction such as an ionic bond, a
hydrogen bond or a van der Waals contact. Through the local virial theorem,
$\tfrac14\nabla^2\rho = 2G + V$ with the kinetic energy density $G > 0$ and
the potential energy density $V < 0$, the sign of the Laplacian says which
energy dominates at the bcp.

QTAIM is defined from the density alone, so once the density is converged
its bond definition is unambiguous and only weakly method dependent. Its
limits are equally well known: gradient paths can exist without a chemical
bond (H...H contacts in crowded molecules), a bcp lying close to a ring
critical point annihilates with it under small geometry changes, and near
the nuclei a Gaussian basis has no cusp, so ncp positions and Laplacians
there depend on the basis set.

## Algorithm sketch

1. Build a uniform grid around the molecule (`grid_spacing`, `padding`).
2. Evaluate ρ(r), ∇ρ(r) and the Hessian H_ρ(r) on the grid with the
   analytic `evaluate_ao_with_hessian` binding.
3. Select seeds: voxels where |∇ρ| is a local minimum over its 26
   neighbours and lies below `gradient_tol`.
4. Newton-Raphson refine each seed to a stationary point.
5. Classify by Hessian eigenvalues: (3,-3) = ncp, (3,-1) = bcp,
   (3,+1) = rcp, (3,+3) = ccp.
6. For each BCP, trace gradient paths to the two connected atoms.

## Quick start

`Atom` coordinates are in bohr, as everywhere in vibe-qc; a geometry typed
in Angstrom values is a compressed molecule whose density has no bond
critical points at all.

```python
from vibeqc.qtaim import qtaim_analysis, qtaim_result_to_qvf
import vibeqc as vq

# Run a molecular SCF first (coordinates in bohr)
mol = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0]),
                   vq.Atom(1, [0.0, 1.43, -0.98]),
                   vq.Atom(1, [0.0, -1.43, -0.98])])
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

On this geometry at RHF/6-31G* the search finds three nuclear critical
points and two bond critical points, one per O-H bond, with
$\rho_b \approx 2.66$ e/A³, $\nabla^2\rho_b \approx -57$ e/A⁵ (a shared,
covalent interaction) and $\varepsilon \approx 0.02$ (a cylindrically
symmetric single bond), and traces both bond paths; the Poincare-Hopf count
is $3 - 2 = 1$. The analysis takes about three seconds at a 0.1 bohr grid
(vibe-qc 0.17.x development build on an Apple M-series CPU); the default
0.15 bohr grid finds the same points in about a second.

## Data types

| Class | Attributes |
|---|---|
| `CriticalPoint` | `type` ("ncp"/"bcp"/"rcp"/"ccp"), `position` (A), `rho` (e/A^3), `laplacian` (e/A^5), `ellipticity` (BCP only), `atom_pair` (BCP only) |
| `BondPath` | `atoms` ((i, j) indices), `path` ((N,3) points in A, running from atom `i` through the bond critical point to atom `j`) |
| `QTAIMResult` | `critical_points` (list), `bond_paths` (list), `grid_spacing` (bohr) |

## Where to look in the code

| Layer | File |
|---|---|
| Python API | `python/vibeqc/qtaim.py` |
| AO evaluation (density, gradient and Hessian) | `cpp/src/bindings.cpp` (`evaluate_ao_with_hessian`) |
| Grid infrastructure | `python/vibeqc/cube.py` |
| QVF topology writer | `python/vibeqc/output/formats/qvf.py` (`topology.qtaim` section) |
| Citation route | `python/vibeqc/output/citations/database.toml` (routes: `qtaim`, `bader`) |
| Tests | `tests/test_qtaim.py` |

## Known limitations

- Atomic basin integration (the full AIM partitioning, and with it QTAIM
  charges and the exact delocalisation index) is not implemented; the
  current scope is critical-point search and bond-path tracing.
- Ring and cage critical points are only found when the seed grid resolves
  them; refine `grid_spacing` and check the Poincare-Hopf count when a ring
  system returns fewer points than expected. No automatic count check is
  performed.
- Molecular SCF densities only; periodic densities are not analysed.

## References

- Bader, R. F. W., *Atoms in Molecules*, Oxford University Press (1990).
- Bader, R. F. W., *Acc. Chem. Res.* **18**, 9-15 (1985), the original
  QTAIM formulation ({vibeqc-cite}`bader_qtaim_1985`).
- Lu, T. & Chen, F., *J. Comput. Chem.* **33**, 580-592 (2012), the
  MultiWFN reference implementation.

Citations are automatically included in the output manifest when
QTAIM analysis is performed; see `docs/user_guide/citations.md`.

## See also

- [Volumetric data (cube / molden)](volumetric_data.md), density and
  orbital export.
- [Post-SCF properties](properties.md), Mulliken / Loewdin / Mayer
  bond-order analysis.
- [Bond analysis](bond_analysis.md), the AO-approximated delocalisation
  index that the basin integration would replace.
- [QVF file format](../tutorial/qvf_file_format.md), the container
  that carries QTAIM topology data.
