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

For an LCAO periodic calculation (the crystalline-orbital formulation of
{vibeqc-cite}`pisani_crystal_1988`), each sampled wave vector solves

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

The uniform mesh is the Monkhorst-Pack construction ({vibeqc-cite}`monkhorst_pack_1976`);
standard path conventions are discussed in [k-point meshes](k_points.md).

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
currently exports the DOS projection helpers but not this fat-band function;
the next section shows the call.

## Fat bands: projected band weights

A fat-band plot draws every band with a width, or a colour, proportional to
its character on a chosen atom or angular-momentum channel. vibe-qc computes
that character with the same Mulliken partition ({vibeqc-cite}`mulliken_1955`)
it uses for the projected DOS, delivered band by band at every sampled k point instead of integrated
over energy. For band $n$ at $\mathbf k$ with coefficient vector
$\mathbf C_n(\mathbf k)$, the weight on AO $\mu$ and on a channel $c$ (an
(atom, $l$) group of AOs) are

$$
w_{\mu,n\mathbf k} = \operatorname{Re}\Big[\,C^{*}_{\mu n}(\mathbf k)\,
\big(\mathbf S(\mathbf k)\,\mathbf C_n(\mathbf k)\big)_{\mu}\Big],
\qquad
W_{c,n\mathbf k} = \sum_{\mu\in c} w_{\mu,n\mathbf k},
\qquad
\sum_{c} W_{c,n\mathbf k} = \mathbf C_n^\dagger\mathbf S\,\mathbf C_n = 1 .
$$

The weights sum to one per band because the channels partition the AO set
exactly once, which `band_structure_projected` checks before it runs.
Mulliken weights can be negative for strongly overlapping diffuse functions;
that is a property of the partition, not a bug, and a large negative weight
is a hint to look at the basis rather than at the band.

`band_structure_projected(F_real, S_real, system, basis, kpath,
n_electrons_per_cell=None)` in `vibeqc.bands` takes the real-space lattice
blocks of the operator and the overlap, a `KPath`, and returns a
`BandStructure` whose `projections` array has shape `(n_kpoints, n_bands,
n_channels)` and whose `channels` list carries labels such as `"Si1-s"` or
`"O2-p"` (atom symbol, one-based atom index, angular momentum). The
following complete example builds the Hcore fat bands of the 1-D H chain from
the page's first example; the kinetic and nuclear blocks are added cell by
cell with the core binding `make_lattice_matrix_set`, since the projected
helper takes one operator rather than a list of terms:

```python
import numpy as np
import vibeqc as vq
from vibeqc._vibeqc_core import make_lattice_matrix_set
from vibeqc.bands import band_structure_projected

# 1-D H chain: two H atoms per cell, 6 bohr period, vacuum along y and z.
system = vq.PeriodicSystem(
    dim=1,
    lattice=[[6.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
    unit_cell=[vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# Real-space lattice blocks of S, T and V; Hcore = T + V block by block.
opts = vq.LatticeSumOptions()
S_real = vq.compute_overlap_lattice(basis, system, opts)
T_real = vq.compute_kinetic_lattice(basis, system, opts)
V_real = vq.compute_nuclear_lattice(basis, system, opts)
H_real = make_lattice_matrix_set(
    S_real.nbf, list(T_real.cells),
    [np.asarray(t) + np.asarray(v) for t, v in zip(T_real.blocks, V_real.blocks)],
)

kpath = vq.kpath_from_segments(
    system, [((0.0, 0.0, 0.0), "Γ", (0.5, 0.0, 0.0), "X")], points_per_segment=20,
)
bands = band_structure_projected(H_real, S_real, system, basis, kpath, n_electrons_per_cell=2)
print(bands.channels)                       # ['H1-s', 'H2-s']
print(bands.projections.shape)              # (21, 2, 2): (n_k, n_bands, n_channels)
print(np.allclose(bands.projections.sum(axis=2), 1.0))   # True: Mulliken weights sum to 1 per band
print(np.round(bands.projections[0], 3))    # band characters at Γ: 0.5 / 0.5 on each H
```

On the vibe-qc 0.17.x development build (Apple M2) this prints the two
channel labels, the `(21, 2, 2)` shape, `True`, and a $2\times 2$ block of
`0.5`: both Hcore bands of the symmetric chain are shared equally between the
two hydrogens at every k, as they must be. For a self-consistent fat band,
pass the converged Fock lattice blocks of a route that exposes them (the
explicit real-space F/S workflow in the table above) in place of `H_real`; the
weights are then the character of the converged crystal orbitals.

The projections travel with the band object. A `BandStructure` carrying
`projections` and `channels`, passed to `run_periodic_job(...,
band_structure=bands, output_qvf=True)`, is written into the QVF `bands`
section as `bands/projections.bin` (float64, `(n_kpoints, n_bands,
n_channels)`) next to `bands/eigenvalues.bin`, with the channel labels stored
in `bands/kpath.json` under `"channels"`. That is the payload vibe-view's
band panel reads to colour bands by character; a band object without
projections produces the same section without those two members. Keep the
operator conventions straight when you do this: an Hcore fat band beside an
SCF DOS is two different spectra, and the QVF does not merge them.

## Fermi reference and band gaps

When `n_electrons_per_cell` is an even integer, `BandStructure.e_fermi` is
the highest occupied eigenvalue found on the sampled path, and
`DensityOfStates.e_fermi` the highest occupied eigenvalue on the mesh under
one Fermi level for the whole mesh (the same zero-temperature fill the SCF
uses). On an insulator both are the valence-band maximum. For an odd electron
count they are left unset because separate spin channels are required. A
path is not a Brillouin-zone sample, so the path reference is not reliable
for a metal.

For a semiconductor or insulator, with $n$ occupied bands, report

$$
E_g = \min_{\mathbf{k}}\varepsilon_{n\mathbf{k}}
- \max_{\mathbf{k}}\varepsilon_{n-1,\mathbf{k}},
$$

bands counted from zero in the energy-sorted spectrum, using a converged mesh
or a path that includes both extrema. State whether the gap is direct or
indirect and identify the k points.

Do not compute it as "lowest empty state minus highest occupied state" of a
zero-temperature SCF on a mesh. With one Fermi level for the whole mesh, a
metal whose bands overlap has every occupied sampled level below every empty
one, so that difference is positive: it is the spacing of the sampled levels
around the Fermi level, and it shrinks to zero as the mesh is refined. A band
that is occupied at some k-points and empty at others is the signature of a
metal; the band-index formula above then returns the negative band overlap.
For a metal, converge the k mesh, occupation model, electronic temperature,
entropy term, and energy/free-energy convention together; see
[Smearing](smearing.md).

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
frontier = vq.frontier_bands(result)        # reads the result's occupations
print(frontier.homo, frontier.lumo, frontier.gap, frontier.is_metallic)
print(frontier.homo_k, frontier.lumo_k)     # direct vs indirect
```

`frontier_bands(result)` classifies the states from the occupations the SCF
actually used, through `vibeqc.band_edges.result_band_edges`. That is the
same classifier the `.out` "Band extrema (multi-k)" block, the `.system`
`[run]` band-edge fields (`indirect_gap`, `direct_gap`, `valence_band_max`,
`conduction_band_min`, `gap_above_fermi_manifold`, `is_metallic`, the same
keys on every periodic route) and the semiempirical k-route result fields read,
so every surface reports the same numbers. A metal reports
`is_metallic=True` and a gap `<= 0`: the band overlap when the Fermi level
crosses a band, or minus the spread of a fractionally occupied Fermi group.
Screen on `is_metallic`, not on the sign of the gap. `gap_above_fermi_manifold`
is the distance to the next entirely empty state; it is never the band gap.

`frontier_bands(result, n_occ=n)` is the band-index reading, bands `n - 1` and
`n` at every k, for a result without occupations. It agrees with the default
on every insulator and every band-overlap metal. `n_occ` is validated against
the number of **bands**, not the number of k-points, and a ragged per-k set is
refused rather than silently misaligned. Both forms take the VBM as a maximum
over every sampled k and the CBM as a minimum, so an indirect gap is found
correctly and a degenerate manifold reduces to a scalar rather than raising.

For an unrestricted result, `frontier_bands(result)` combines both spin
channels (the per-spin edges are in `frontier.edges.channels`), and
`spin="alpha"` or `spin="beta"` selects one; `per_k_band_energies` requires
`spin=` there, since the two channels are different spectra.

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
