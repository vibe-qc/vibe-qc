# Band structure and density of states

Band structures and densities of states describe the same eigenvalue problem
from two different sampling strategies. A band plot follows selected lines
through reciprocal space. A density of states (DOS) integrates over a mesh
covering the Brillouin zone.

Use the [worked band tutorial](../tutorial/band_structure.md) for the small H2
chain example and the [PDOS tutorial](../tutorial/pdos.md) for orbital
projections. This page explains which calculation to run, what the returned
numbers mean, and what must be converged before interpreting a plot.

## The eigenvalue problem

For an LCAO periodic calculation, each sampled wave vector solves

$$
\mathbf{F}(\mathbf{k})\mathbf{C}_n(\mathbf{k})
= \varepsilon_n(\mathbf{k})
\mathbf{S}(\mathbf{k})\mathbf{C}_n(\mathbf{k}),
$$

where the Bloch matrices are assembled from real-space lattice blocks,

$$
\mathbf{F}(\mathbf{k}) = \sum_g
e^{i\mathbf{k}\cdot\mathbf{R}_g}\mathbf{F}^{(g)},
\qquad
\mathbf{S}(\mathbf{k}) = \sum_g
e^{i\mathbf{k}\cdot\mathbf{R}_g}\mathbf{S}^{(g)}.
$$

The energies depend on the operator used for $\mathbf{F}$:

| Operator | What it contains | Appropriate use |
|---|---|---|
| Hcore | kinetic plus nuclear attraction, $\mathbf{T}+\mathbf{V}$ | fast infrastructure check and teaching example |
| SCF Fock or Kohn-Sham | Hcore plus the converged electron-electron terms | physical band, DOS, and PDOS interpretation |

An Hcore spectrum is not a Hartree-Fock or DFT prediction. It can illustrate
Bloch sums and orbital character, but it must not be reported as a
self-consistent band gap.

## A path and a mesh answer different questions

| Sampling | Use it for | Converge |
|---|---|---|
| Labeled k-path | dispersion and crossings along chosen symmetry lines | path convention and points per segment |
| Monkhorst-Pack mesh | DOS, PDOS, occupations, and Brillouin-zone averages | mesh density in every periodic direction |

A smooth line on a k-path does not prove that the global valence maximum or
conduction minimum lies on that path. Determine an indirect gap from a
converged mesh or from a path known to include the relevant extrema.

Fractional k coordinates are expressed in the reciprocal basis. vibe-qc
constructs that basis from the direct-lattice columns as
$\mathbf{B}=2\pi\mathbf{A}^{-T}$. The path point `(0.5, 0, 0)` therefore
means half of the first reciprocal vector, not a Cartesian value of
$0.5\ \mathrm{bohr}^{-1}$.

## Fast Hcore sanity check

The following complete example builds a one-dimensional H2 chain, samples
$\Gamma\rightarrow X$, computes a DOS on a separate mesh, and writes a plot:

```python
import numpy as np
import vibeqc as vq
from vibeqc.plot import bands_dos_figure

system = vq.PeriodicSystem(
    dim=1,
    lattice=[[6.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
    unit_cell=[
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [1.4, 0.0, 0.0]),
    ],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

kpath = vq.kpath_from_segments(
    system,
    [((0.0, 0.0, 0.0), "Γ", (0.5, 0.0, 0.0), "X")],
    points_per_segment=40,
)
bands = vq.band_structure_hcore(
    system,
    basis,
    kpath,
    n_electrons_per_cell=2,
)
dos = vq.density_of_states_hcore(
    system,
    basis,
    mesh=[80, 1, 1],
    sigma=0.02,
    n_electrons_per_cell=2,
)

area = np.trapezoid(dos.dos, dos.energies)
print(f"bands shape: {bands.energies.shape}")
print(f"integrated DOS: {area:.6f}; expected {basis.nbasis}")

fig = bands_dos_figure(bands, dos, title="H2 chain (STO-3G, Hcore)")
fig.savefig("h-chain-hcore-bands-dos.png", dpi=180)
```

The returned energies and `sigma` are in Hartree. The plotters accept
`units="eV"` and shift the displayed energy zero to `e_fermi` by default.

## The DOS equation and normalization

vibe-qc broadens the sampled eigenvalues with unit-area Gaussians,

$$
g_\sigma(E) = \sum_{\mathbf{k}} w_{\mathbf{k}}\sum_n
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left[-\frac{(E-\varepsilon_{n\mathbf{k}})^2}{2\sigma^2}\right],
$$

with normalized k-point weights $\sum_{\mathbf{k}}w_{\mathbf{k}}=1$.
Over a sufficiently wide energy grid,

$$
\int g_\sigma(E)\,dE = N_\mathrm{bands}.
$$

This is a state-count DOS, not an electron-count DOS. Occupations and spin
degeneracy are separate. The integral is a useful numerical check: a narrow
energy window or excessive tail truncation makes it smaller than the number
of bands.

Broadening changes appearance, not the underlying eigenvalues. Useful
starting values are:

| `sigma` | Approximate width | Typical effect |
|---:|---:|---|
| 0.002 Ha | 0.054 eV | exposes discrete mesh spikes |
| 0.005 Ha | 0.136 eV | resolves many semiconductor features |
| 0.010 Ha | 0.272 eV | smoother survey plot |
| 0.020 Ha | 0.544 eV | may hide narrow edge structure |

Converge the k mesh before choosing the broadening for presentation. A large
`sigma` must not be used to conceal an under-sampled DOS.

## DOS and PDOS in QVF output

For a converged Gaussian-basis periodic job, the high-level runner builds total
DOS and atom/angular-momentum projected DOS for the default QVF output. Use
`dos_kmesh` to control the post-SCF mesh:

```python
import vibeqc as vq


def run_bulk_with_spectrum(system, basis):
    return vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="pbe",
        kpoints=[4, 4, 4],
        dos_kmesh=[12, 12, 12],
        output_qvf=True,
        output="output-bulk-pbe",
    )
```

Open `output-bulk-pbe.qvf` in vibe-view and select the DOS or PDOS section.
The runner uses a 0.05 eV Gaussian width for these QVF sections. Choose a mesh
appropriate to dimensionality, for example `[64, 1, 1]` for a 1D chain or
`[24, 24, 1]` for a 2D sheet. A default `[8, 8, 8]` is only a starting point
for a 3D crystal.

```{warning}
For RKS and UKS, the current QVF post-processor does not diagonalize the
converged Kohn-Sham operator. It reconstructs an operator from the converged
density without the exchange-correlation potential; the restricted path also
uses full exact exchange. Treat that QVF spectrum as a diagnostic only. Use a
route-specific helper that is documented to consume the converged operator for
quantitative DFT bands or DOS.
```

The `band_structure=` argument is different: it embeds a band object that you
have already computed. Do not pass an Hcore band object beside an SCF DOS and
then label both as one self-consistent spectrum. Use a route-specific SCF band
helper when one is available.

## Programmatic SCF routes

The generic `band_structure` and `density_of_states` functions consume
real-space `LatticeMatrixSet` objects. They are post-processing primitives;
the generic periodic result objects do not all expose `fock_real` and
`overlap_real` attributes.

Current route-specific choices include:

| SCF route | Band or DOS helper | Notes |
|---|---|---|
| GPW/GAPW | `band_path_eigenvalues`, `compute_dos_from_result` | fixed-density route; `compute_dos_from_result` is occupation weighted, integrates to the electron count, and gives virtual states zero weight; see [GAPW](gapw.md) |
| AICCM2026DEV-B | `aiccm2026dev_b_band_structure` | finite-torus interpolation; converge lattice extension |
| CCM A stream | `vibeqc.periodic.ccm.ccm_band_structure` | exact on folded character points; converge `nrep` |
| Explicit real-space F/S workflow | `band_structure`, `density_of_states`, `density_of_states_projected` | advanced API when the caller owns the lattice blocks |

For projected fat bands on explicit lattice blocks, import
`band_structure_projected` from `vibeqc.bands`. The top-level namespace
currently exports the DOS projection helpers but not this fat-band function.

## Fermi reference and band gaps

When `n_electrons_per_cell` is an even integer, `BandStructure.e_fermi` is
the highest occupied eigenvalue found on the sampled path. For an odd
electron count it is left unset because separate spin channels are required.

This reference is convenient for plotting, but it is not a finite-temperature
chemical potential and it is not reliable for a metal. For a semiconductor or
insulator, report

$$
E_g = \min_{n\in\mathrm{unocc},\mathbf{k}}\varepsilon_{n\mathbf{k}}
- \max_{n\in\mathrm{occ},\mathbf{k}}\varepsilon_{n\mathbf{k}},
$$

using a converged mesh or a path that includes both extrema. State whether the
gap is direct or indirect and identify the k points. For a metal, converge the
k mesh, occupation model, electronic temperature, entropy term, and
energy/free-energy convention together; see [Smearing](smearing.md).

Combined plotting helpers shift each panel by that object's own `e_fermi`.
A path-derived occupied reference and a mesh-derived reference can differ even
when the operator is the same. Before sharing an energy axis, verify that the
references agree. If they do not, align the data to one stated reference and
plot with `shift_to_fermi=False`.

### Reading the gap off a converged SCF result

Do not reach into a result object's eigenvalue attributes yourself. The layout
differs by driver: `PBCBipoleRHFResult` carries `mo_energies` as a *per-k*
`list` and has no `mo_energies_k` at all, while the GPW/GAPW multi-k results
carry `mo_energies_k` and use `mo_energies` for the Γ-only case. So a reader
that guesses gets it wrong in one of two ways, and only one of them is loud
(issues #15, #135, #505).

Use the accessors:

```python
import vibeqc as vq

bands = vq.per_k_band_energies(result)      # list of 1-D arrays, one per k
frontier = vq.frontier_bands(result, n_occ=n_electrons // 2)
print(frontier.homo, frontier.lumo, frontier.gap)
print(frontier.homo_k, frontier.lumo_k)     # direct vs indirect
```

`frontier_bands` takes the VBM as a maximum over every sampled k and the CBM as
a minimum, so an indirect gap is found correctly and a degenerate manifold
reduces to a scalar rather than raising. `n_occ` is validated against the number
of **bands**, not the number of k-points, and a ragged per-k set is refused
rather than silently misaligned.

A worked example of what guessing costs: on H₂/STO-3G in a 6-bohr cube at
k = (2,1,1), flattening the per-k list into one array and reading positions
`n_occ - 1` and `n_occ` reports a gap of 0.0235 Ha: the spacing between the
two k-points' *occupied* levels, not the true 1.3834 Ha. The SCF
converged, the run exits 0, and nothing in the log says otherwise.

## CPU time and memory

Diagonalizing an $N_\mathrm{bf}\times N_\mathrm{bf}$ generalized eigenproblem
at $N_k$ independent points has the leading trend

$$
T \sim \mathcal{O}(N_k N_\mathrm{bf}^3).
$$

Serial processing can keep each generalized-eigenproblem workspace near
$\mathcal{O}(N_\mathrm{bf}^2)$. The current PDOS implementation also retains
Mulliken weights of shape $(N_k,N_\mathrm{bf},N_\mathrm{bf})$, so its leading
post-processing storage is $\mathcal{O}(N_kN_\mathrm{bf}^2)$. Stored curves add
$\mathcal{O}(GN_E)$ for $G$ projection groups and $N_E$ energy points. The
preceding SCF can still set the process peak, while a dense mesh can make
diagonalization dominate CPU time.

Record the basis size, SCF mesh, DOS mesh, path, `sigma`, thread count, wall
time, CPU time, and peak RSS. Do not compare two spectral routes when these
inputs differ.

## Interpretation checklist

Before publishing a band or DOS figure, verify:

1. The underlying SCF converged to a physically credible solution.
2. The basis, Coulomb route, SCF mesh, and numerical thresholds are converged.
3. The DOS mesh is converged independently of the SCF mesh.
4. The path follows a stated reciprocal-space convention.
5. The plot identifies Hcore versus SCF data unambiguously.
6. The energy reference, occupations, spin channels, and broadening are stated.
7. The DOS integral passes its state- or electron-count check as appropriate;
   PDOS groups form a verified, non-overlapping AO partition and agree with a
   separately computed unprojected DOS on the same grid.
8. The input, `.out`, `.system`, `.qvf`, and citations remain together.

The complete runnable Hcore example is
[`examples/periodic/input-h-chain-bands.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-h-chain-bands.py).
