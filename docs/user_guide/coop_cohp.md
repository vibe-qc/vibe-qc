# COOP and COHP bonding analysis

Crystal Orbital Overlap Population (COOP) and Crystal Orbital Hamilton
Population (COHP) resolve periodic bonding interactions by energy and atom
pair. They complement a band or PDOS plot:

- PDOS asks which AO groups contribute to a state.
- COOP asks whether the overlap population between two atoms is bonding or
  antibonding.
- COHP weights that interaction by a Hamiltonian matrix element.

vibe-qc returns and plots the Lobster-style **negative COHP**, written
`-COHP`, so positive values indicate bonding in both the COOP and displayed
COHP curves.

## Recommended workflow

The high-level periodic runner is the supported way to compute COOP, COHP,
periodic Mayer bond orders, total DOS, and PDOS as one post-SCF output family.
The following complete example uses a 1D H2 chain:

```python
import vibeqc as vq

a = 6.0
vacuum = 30.0
system = vq.PeriodicSystem(
    dim=1,
    lattice=[[a, 0.0, 0.0], [0.0, vacuum, 0.0], [0.0, 0.0, vacuum]],
    unit_cell=[
        vq.Atom(1, [0.0, vacuum / 2, vacuum / 2]),
        vq.Atom(1, [1.4, vacuum / 2, vacuum / 2]),
    ],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_job(
    system,
    basis,
    method="RKS",
    functional="lda",
    kpoints=[8, 1, 1],
    dos_kmesh=[32, 1, 1],
    output_qvf=True,
    coop_cohp=True,
    output="h2-chain-coop-cohp",
)

print(result.converged, result.energy)
```

The runner embeds `dos.coop`, `dos.cohp`, total DOS, PDOS, and periodic bond
data in `h2-chain-coop-cohp.qvf`. It also records the route and required
citations in the `.out`, `.system`, `.references`, and `.bibtex` siblings.

For bulk GDF, the payload uses the returned SCF eigenpairs, spin Fock
matrices, overlaps and accepted densities. This retains XC and ECP contributions
and fractional occupations. The default spectrum uses the SCF k mesh, recorded
in the manifest. A different `dos_kmesh` currently requires a target-k
Hamiltonian provider and is explicitly refused; it is not produced by
interpolating a finite-mesh Fock. Spectral storage is admitted under a 128 MiB
budget before allocation.

ROHF has shared orbitals of an effective Roothaan operator and separate
physical spin Fock matrices. The QVF writer preserves explicit operator and
validation labels when supplied by private ROHF diagnostics. Public GDF
ROHF spectral generation remains guarded pending validation of consumer
interpretation and the numerical projections. A physical spin-Fock
projection on effective energies needs to retain that stated meaning.

```{warning}
Other periodic routes still use the legacy reconstructed post-SCF operator,
which can omit XC. Their COHP weights now use that same analyzed operator in
each spin channel, but this does not make it the converged Kohn-Sham spectrum.
```

Inspect the archive without writing a custom parser:

```sh
vibeqc coop h2-chain-coop-cohp.qvf
vibeqc coop --json h2-chain-coop-cohp.qvf
vibeqc mayer h2-chain-coop-cohp.qvf
vibeqc mayer --json --threshold 0.1 h2-chain-coop-cohp.qvf
vibe-view open h2-chain-coop-cohp.qvf
```

`coop_cohp=True` requires QVF output: the archive is the only place the
analysis is written. Passing `coop_cohp=True` with `output_qvf=False` raises a
`ValueError` before the SCF starts, naming both knobs, instead of running the
SCF and silently skipping the analysis.

## Definitions and sign convention

For atom pair $A,B$, band $n$, and wave vector $\mathbf{k}$, define the
pair-resolved overlap and chosen-Hamiltonian populations

$$
P^{S}_{AB,n\mathbf{k}}=
\operatorname{Re}\sum_{\mu\in A}\sum_{\nu\in B}
S_{\mu\nu}(\mathbf{k})
C^*_{\mu n}(\mathbf{k})C_{\nu n}(\mathbf{k}),
$$

$$
P^{H}_{AB,n\mathbf{k}}=
\operatorname{Re}\sum_{\mu\in A}\sum_{\nu\in B}
H_{\mu\nu}(\mathbf{k})
C^*_{\mu n}(\mathbf{k})C_{\nu n}(\mathbf{k}).
$$

After Gaussian broadening,

$$
\mathrm{COOP}_{AB}(E)=
\sum_\mathbf{k}w_\mathbf{k}\sum_n
P^{S}_{AB,n\mathbf{k}}\,\delta_\sigma(E-\varepsilon_{n\mathbf{k}}),
$$

$$
\mathrm{-COHP}_{AB}(E)=
-\sum_\mathbf{k}w_\mathbf{k}\sum_n
P^{H}_{AB,n\mathbf{k}}\,\delta_\sigma(E-\varepsilon_{n\mathbf{k}}).
$$

For standard COHP, $\mathbf{H}$ is the same spin Hamiltonian used to generate
$\mathbf{C}$ and $\varepsilon$. A projection of any other operator must be
identified as an explicit operator projection, rather than standard COHP. The returned COHP
array already includes the leading minus sign. Interpret the displayed
convention as:

| Curve value | COOP | Returned `-COHP` |
|---|---|---|
| positive | bonding overlap population | bonding Hamilton population |
| negative | antibonding overlap population | antibonding Hamilton population |

The integrated values through the reference energy are ICOOP and `-ICOHP`.
They summarize the occupied contribution for each pair. Always state the sign
convention because some programs plot raw COHP instead.

## Result shape

The lower-level `COOPCOHPResult` contains:

| Attribute | Restricted shape | Unrestricted shape | Meaning |
|---|---:|---:|---|
| `energies` | `(n_energy,)` | `(n_energy,)` | Hartree energy grid |
| `coop` | `(n_pairs, n_energy)` | `(2, n_pairs, n_energy)` | alpha/beta split when unrestricted |
| `cohp` | same as `coop`, or `None` | same as `coop`, or `None` | returned `-COHP`; absent without `H_terms` |
| `integrated_coop` | `(n_pairs,)` | `(2, n_pairs)` | ICOOP through the reference energy |
| `integrated_cohp` | `(n_pairs,)`, or `None` | `(2, n_pairs)`, or `None` | returned `-ICOHP` |
| `pairs` | list | list | atom indices, symbols, and distance in Angstrom |

The high-level workflow serializes these arrays into QVF rather than attaching
real-space Fock terms to the generic SCF result object.

## Pair selection

`ao_pairs_per_atom_pair(system, basis, pair_distance_cutoff=8.0)` creates AO
index lists only for distinct atom pairs $A<B$ whose raw home-cell coordinate
distance is within the cutoff. The cutoff is in bohr. The high-level runner
does not currently expose this cutoff.

This selector does not apply a minimum-image convention, enumerate
image-resolved bonds, or include a bond from an atom to one of its own periodic
images. A one-atom primitive cell therefore produces no pairs. Bloch-summed
weights include lattice translations, but the pair label and distance still
describe the raw home-cell atoms. Inspect that limitation before assigning a
curve to a bond crossing the cell boundary.

## Advanced explicit-block API

`compute_coop_cohp` is also available when a workflow already owns the
real-space lattice blocks. Its positional arguments are Fock terms, overlap,
system, basis, and k mesh. Set `include_cohp=True` to project the analyzed
Hamiltonian in each spin channel:

```python
import vibeqc as vq


def analyze_lattice_blocks(
    fock_terms,
    overlap_real,
    system,
    basis,
    kmesh,
):
    return vq.compute_coop_cohp(
        fock_terms,
        overlap_real,
        system,
        basis,
        kmesh,
        include_cohp=True,
        pair_distance_cutoff=8.0,
        sigma=0.01,
        n_electrons_per_cell=system.n_electrons(),
    )
```

The compatibility `H_terms` input also enables COHP, but is accepted only
when its Bloch sum equals the analyzed Hamiltonian for every spin. A different
operator is refused. The low-level `cohp_weights_k` kernel remains available
for explicitly named arbitrary-operator projections.

For an unrestricted operator, pass the beta-spin lattice blocks as
`F_terms_beta=`. Do not copy older examples that access
`result.fock_terms`, `result.overlap_real`, or `result.hcore_terms`: generic
periodic results do not promise those fields.

`periodic_mayer_bond_orders` uses the same explicit-block model for a scalar
bond-order matrix. The production runner computes its own consistent version
for QVF output.

## Plotting an explicit result

The plotters accept a `COOPCOHPResult` directly:

```python
from vibeqc.plot import cohp_figure, coop_figure


def save_pair_plots(analysis):
    coop = coop_figure(analysis, max_pairs=8, title="COOP")
    coop.savefig("coop.png", dpi=180)

    cohp = cohp_figure(analysis, max_pairs=8, title="-COHP")
    cohp.savefig("cohp.png", dpi=180)
```

`bands_cohp_figure(bands, analysis)` produces a combined view when `bands` is a
valid `BandStructure` computed from the same operator and energy convention.
The helper shifts each panel by its own stored reference, so first verify that
those references agree. Otherwise align both objects to one stated reference
and use `shift_to_fermi=False`. Do not combine Hcore bands with SCF-derived
COHP and present them as one spectrum.

## Convergence and interpretation

COOP and COHP inherit every approximation in the underlying SCF and add their
own reconstruction, mesh, broadening, and pair-selection choices. Use this
order:

1. Converge the SCF basis, Coulomb route, k mesh, and numerical thresholds.
2. Converge `dos_kmesh` for the pair curves and integrated values.
3. Hold the mesh fixed while choosing Gaussian broadening for presentation.
4. Inspect the available home-cell pairs and exclude claims that require an
   image-resolved or same-atom image bond.
5. Compare atom-pair trends across at least one reasonable basis change.
6. Retain the QVF and all provenance/citation sidecars.

An integrated value is not a universal bond energy. It depends on the basis,
Hamiltonian, occupations, reference energy, pair partition, and sign
convention. Use it for controlled comparisons with all of those choices held
fixed.

For metals, the k mesh, occupations, electronic temperature, entropy term,
and energy/free-energy convention must be converged together. See
[Smearing](smearing.md).

## CPU time and memory

At each of $N_\mathbf{k}$ k points, the analysis diagonalizes an
$N_\mathrm{bf}\times N_\mathrm{bf}$ operator and evaluates the requested pair
weights. The leading diagonalization trend is

$$
T_\mathrm{diag}\sim\mathcal{O}(N_\mathbf{k}N_\mathrm{bf}^3).
$$

Pair accumulation adds work proportional to the number of selected pairs and
energy-grid points. In addition to $\mathcal{O}(N_\mathrm{bf}^2)$ matrix
workspace, the current implementation forms transient pair weights of shape
$(N_\mathrm{pair},N_\mathbf{k},N_\mathrm{bf})$ per spin. Stored curves scale as
$\mathcal{O}(N_\mathrm{spin}N_\mathrm{pair}N_E)$.

Record CPU time, wall time, peak RSS, basis count, SCF and analysis meshes,
pair-selection rule, number of pairs, energy-grid size, `sigma`, spin treatment,
threads, host, and version.

## References and related guides

The runner selects the defining references automatically when the analysis
executes:

- Hughbanks and Hoffmann, *J. Am. Chem. Soc.* **105**, 3528 (1983), COOP.
- Dronskowski and Blochl, *J. Phys. Chem.* **97**, 8617 (1993), COHP.
- Deringer, Tchougreeff, and Dronskowski, *J. Phys. Chem. A* **115**, 5461
  (2011), projection framework.
- Maintz et al., *J. Comput. Chem.* **37**, 1030 (2016), Lobster convention.

Continue with:

- [Projected DOS](../tutorial/pdos.md) for atom and angular-momentum character.
- [Band structure and DOS](band_structure.md) for path/mesh sampling and
  self-consistent route helpers.
- [Bond analysis](bond_analysis.md) for molecular and periodic scalar bond
  measures.
