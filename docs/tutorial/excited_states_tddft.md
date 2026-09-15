# Excited states with TDDFT

SCF supplies a reference electronic state; convergence alone does not prove
it is the ground state. To predict a UV/Vis
spectrum, a photochemical pathway, or the colour of a dye, you need the
*excited* states and how brightly each one absorbs. Time-dependent
density-functional theory (TDDFT) is the workhorse for this: it takes a
converged ground-state Kohn-Sham reference and returns vertical
excitation energies and oscillator strengths at a cost only a small
multiple of the ground-state SCF. This page runs TDDFT on formaldehyde
two ways, the full Casida linear-response equation and its
Tamm-Dancoff approximation (TDA), compares them on the same molecule,
and the theory section below derives where the eigenproblem comes from
and what the approximation drops.

vibe-qc builds the excitation problem from the ground-state molecular
orbitals you already have, so the workflow is always two steps: converge
an SCF, then hand its MO energies, MO coefficients, and density to the
TDDFT solver.

```python
import math
import numpy as np
from vibeqc import (
    Atom, Molecule, BasisSet, RKSOptions, run_rks,
    run_tddft_tda, run_tddft_casida,
)

# Formaldehyde H2CO: C at origin, O up the z-axis, two H in the xz-plane.
ang2bohr = 1.8897259886
rCO = 1.205 * ang2bohr          # C=O bond
rCH = 1.111 * ang2bohr          # C-H bond
half = math.radians(116.5) / 2  # half the H-C-H angle
Hx, Hz = rCH * math.sin(half), -rCH * math.cos(half)

mol = Molecule([
    Atom(8, [0.0,  0.0, rCO]),   # O
    Atom(6, [0.0,  0.0, 0.0]),   # C
    Atom(1, [ Hx,  0.0, Hz]),    # H
    Atom(1, [-Hx,  0.0, Hz]),    # H
])
n_occ = 8                        # 16 electrons / 2
basis = BasisSet(mol, "6-31g*")

# Step 1: converged ground-state reference (B3LYP).
opts = RKSOptions()
opts.functional = "B3LYP"
gs = run_rks(mol, basis, opts)
assert gs.converged, "ground state must converge before TDDFT"

mo_e = np.asarray(gs.mo_energies)
mo_c = np.asarray(gs.mo_coeffs)
dens = np.asarray(gs.density)

# Step 2a: Tamm-Dancoff approximation.
tda = run_tddft_tda(
    mol, basis, mo_e, mo_c, n_occ,
    n_states=5, functional="B3LYP", density_ao=dens, grid_options=opts.grid,
)

# Step 2b: full Casida linear response on the same reference.
cas = run_tddft_casida(
    mol, basis, mo_e, mo_c, n_occ,
    n_states=5, functional="B3LYP", density_ao=dens, grid_options=opts.grid,
)

for res in (tda, cas):
    print(f"\n=== {res.method} (B3LYP kernel) ===")
    print(f"{'state':>5} {'E/eV':>8} {'lambda/nm':>10} {'f_osc':>9}")
    for s in res.states:
        print(f"{s.index:>5} {s.excitation_energy_ev:8.4f} "
              f"{s.wavelength_nm:10.1f} {s.oscillator_strength:9.5f}")
```

The following historical output illustrates the lowest five singlet
excitations on this geometry with the supplied low-level legacy grid.
It is not a new validation run of the current source. The recorded
B3LYP/6-31G\* reference converged in 34 iterations to
`E = -114.43887772` Ha (HOMO-LUMO gap 6.16 eV):

```
=== TDA (B3LYP kernel) ===
state     E/eV  lambda/nm     f_osc
    1   4.1116      301.5   0.00000
    2   9.1021      136.2   0.36210
    3   9.2420      134.2   0.00432
    4  10.2013      121.5   0.03364
    5  10.3771      119.5   0.00000

=== Casida (B3LYP kernel) ===
state     E/eV  lambda/nm     f_osc
    1   4.0906      303.1   0.00000
    2   9.0529      137.0   0.42018
    3   9.1606      135.3   0.00914
    4   9.8107      126.4   0.31225
    5  10.3709      119.6   0.00000
```

## CLI and unrestricted references

`vibeqc tddft` applies the molecular mid-level grid policy to its KS reference
and passes that same grid and physical density to the response calculation.
`--casida` selects full Casida for both restricted and unrestricted references;
without it the CLI uses TDA. For open shells it derives alpha/beta occupations
from the effective electron count and requested multiplicity, including ECP
core removal.

In the direct Python API, open-shell TDA and Casida take separate spin MOs,
occupations and `density_alpha_ao` / `density_beta_ao`, plus `grid_options`.
Use the same functional and grid as the SCF reference. This does not expand
the supported functional kernel or excited-state gradient envelope; those
limits remain independent of CLI routing.

## Reading the spectrum

A TDDFT table is two columns of physics: *where* a state sits (the
excitation energy, equivalently a wavelength) and *how visible* it is
(the oscillator strength `f`, which is what an absorption spectrum
actually plots). A state with `f = 0` is dark; it exists but does not
absorb a photon to first order.

The lowest excitation of formaldehyde, 4.09 eV in Casida (303 nm,
near-UV) with `f = 0`, is the textbook **n -> pi\*** transition: an
electron promoted from the oxygen lone pair (the HOMO) into the empty
C=O pi antibonding orbital (the LUMO). It is symmetry-forbidden, hence
dark, and it is the state that makes formaldehyde photochemistry
interesting. Experiment places it near 3.94 eV and TD-B3LYP benchmarks
land at 3.9 to 4.0 eV, so 4.09 eV is the expected GGA-hybrid result for
this basis.

The second state, 9.05 eV with `f = 0.42`, is the bright **pi -> pi\***
transition (HOMO into the next virtual). Its large oscillator strength
is the dominant feature any computed formaldehyde UV spectrum would
show. The `dominant_amplitudes` field on each state names the leading
orbital pair behind the excitation, so you can label a peak as
"HOMO -> LUMO" rather than guessing:

```python
state = cas.states[0]
occ, virt, amp = state.dominant_amplitudes[0]
print(f"leading: occ {occ} -> virt {virt}, weight {amp:.3f}")
# leading: occ 8 -> virt 1, weight -0.999   (HOMO -> LUMO)
```

## TDA versus full Casida

The two solvers describe the same physics at two levels of theory.
Casida solves the complete linear-response problem; TDA throws away the
coupling between excitations and de-excitations (the `B` matrix in the
theory below). The practical question is how much that costs you, and
the answer on formaldehyde is "very little for the states that matter":

| State | Character | TDA / eV | Casida / eV | TDA - Casida |
|---|---|---|---|---|
| 1 | n -> pi\* (dark) | 4.1116 | 4.0906 | +0.021 |
| 2 | pi -> pi\* (bright) | 9.1021 | 9.0529 | +0.049 |
| 3 | sigma -> pi\* | 9.2420 | 9.1606 | +0.081 |
| 4 | mixed | 10.2013 | 9.8107 | +0.391 |
| 5 | dark | 10.3771 | 10.3709 | +0.006 |

Two patterns are worth internalizing. First, **TDA sits systematically
above Casida**: dropping the de-excitation coupling removes a downward
shift, so TDA energies are upper bounds relative to the full response.
Second, **the gap is tiny for the low, well-separated valence states**
(21 meV on the n -> pi\*) and grows only where states couple strongly
or mix (state 4, where the 0.39 eV difference flags that the de-excitation
block is doing real work).

This is why TDA is the default in much of the literature for valence
excitations: it is cheaper, it is numerically more robust because the
eigenproblem is a plain symmetric diagonalization rather than the
squared Casida form, and it is immune to the triplet-instability
collapse that can drive a full-response excitation energy imaginary near
a near-degenerate reference. Reach for full Casida when you want
oscillator strengths right to a few percent or when comparing against
experimental intensities; reach for TDA for fast, stable state orderings
and excited-state geometries.

## The XC kernel and why the reference matters

TDDFT excitation energies inherit every strength and weakness of the
ground-state functional, plus one extra ingredient: the
exchange-correlation *kernel* `f_xc`, the response of the XC potential to
a density perturbation. In vibe-qc you switch the kernel on by passing
both `functional=` and `density_ao=`. Pass `grid_options=opts.grid` as well
to build the adiabatic LDA/GGA kernel (ALDA/AGGA) on the reference SCF grid.
The four low-level molecular response drivers retain their legacy grid when
`grid_options` is omitted; supplying a density alone does not transfer a grid.
Omitting `density_ao` omits the XC kernel `f_xc`, but retains the orbital gaps,
Hartree/Coulomb coupling and any exact-exchange coupling. This is *not* a full
TDDFT-functional result. The missing-density warning applies to hybrid
functionals with nonzero exact exchange; pure-functional calls do not receive
that warning.

The choice of B3LYP here is deliberate. Pure GGAs such as PBE have a
narrow HOMO-LUMO gap on formaldehyde and the ground-state SCF oscillates
rather than converging cleanly (a real SCF difficulty of the small-gap
GGA on this system, not a vibe-qc bug). The 20 percent exact exchange in
B3LYP opens the gap, the SCF converges in 34 iterations, and the hybrid
TDDFT kernel (exact exchange plus the ALDA part of the functional, the
Bauernschmitt-Ahlrichs composition) is the canonical, well-benchmarked
choice for organic chromophores. If you do want a pure-functional
spectrum, converge the ground state first with a level shift or a tighter
accelerator, then feed the converged MOs to the solver exactly as above.

One sharp edge: **range-separated hybrids are refused, not run wrong.**
Their position-dependent exact exchange cannot be folded into a single
global coefficient over ordinary two-electron integrals, so passing, for
example, `functional="wb97x"` raises `NotImplementedError` rather than
quietly returning meaningless energies. Use a global hybrid or a pure
functional.

## Open-shell and periodic variants

The closed-shell pair above has two siblings for the cases where a single
RKS reference is not enough. Both build the same excitation problem from a
different ground state:

- `run_tddft_tda_uhf(...)` and `run_tddft_casida_uhf(...)` take separate
  alpha and beta MO energies and coefficients from a UHF or UKS ground state
  and solve the spin-unrestricted TDA or full Casida response. They are the
  routes to excitations of radicals, triplets, and other open-shell species.
  LDA, GGA, and global-hybrid references include the spin-polarised `f_xc`
  response kernel when the converged alpha and beta density matrices are
  supplied. Range-separated hybrids are still refused because their
  attenuated exchange needs range-separated ERIs.
- `run_tddft_tda_periodic(result, basis, n_occ=..., ...)` takes a converged
  Gamma-point periodic SCF result and runs the same TDA solver on its
  Bloch orbitals. It is intended for vacuum-padded molecular cells and
  large supercells, where periodic-image interactions in the excitation
  kernel are negligible.

Both return the same `TDDFTResult` container, so the reading code above
is unchanged.

## Excited-state gradients

`run_job(tddft=True, tddft_gradient=True)` prints a finite-difference
nuclear gradient of the lowest excited state, and it is worth knowing
exactly which surface is differentiated. The runner wires one
excited-state energy function into the reference-agnostic driver in
`vibeqc.excited_gradient`: the closed-shell **RHF ground state plus the
CIS (TDA) excitation energy**, followed across the displaced geometries
by amplitude-overlap state tracking and central-differenced with a
1e-3 Å step. The `.out` block is headed
`Excited-state gradient (S₁, RHF/CIS(TDA) surface, FD, step=1e-3 Å)` and
carries a `Surface:` line saying so.

A run that computed a different surface (an RKS reference, a Casida
full-linear-response excitation, or an open-shell reference) is refused
before any calculation starts, with a `NotImplementedError` naming the
missing energy function, rather than being handed the HF-CIS gradient
under a generic header, which is what happened before issue #570. To
differentiate a TDDFT surface today, build the energy function yourself
on top of `run_tddft_tda` / `run_tddft_casida` and feed it to
`cis_state_gradients_fd`; the driver only needs a callable returning the
ground-state energy and the excitation vectors at each geometry.

## Citations

Because TDDFT routes several distinct papers, running with an `output=`
path writes the usual `.bibtex` and `.references` siblings carrying the
exact methods to cite:

```python
tda = run_tddft_tda(
    mol, basis, mo_e, mo_c, n_occ,
    n_states=3, functional="B3LYP", density_ao=dens, grid_options=opts.grid,
    output="h2co_tddft",
)
# writes h2co_tddft.bibtex + h2co_tddft.references
```

The TDDFT-specific keys that land are Runge-Gross 1984 (the TDDFT
existence theorem), Casida 1995 (the linear-response working equations),
Hirata-Head-Gordon 1999 (the Tamm-Dancoff approximation), and
Bauernschmitt-Ahlrichs 1996 (the hybrid kernel), alongside the B3LYP
functional and 6-31G\* basis citations the ground state already carried.
See the [citations guide](../user_guide/citations.md) for the full
provenance surface.

## Theory

The sections below build the excitation problem from the ground up: why a
ground-state functional can predict excited states at all, the
linear-response derivation that produces the Casida eigenproblem, the
Tamm-Dancoff approximation that simplifies it, and where the oscillator
strength comes from.

### From ground-state DFT to excitations

Static Kohn-Sham DFT (see [Molecular DFT](molecular_dft.md)) is a
ground-state theory: the Hohenberg-Kohn theorems are about the lowest
state of a given external potential. The bridge to excited states is the
**Runge-Gross theorem** (1984), the time-dependent analogue: for a system
evolving from a fixed initial state, the time-dependent external potential
is a unique functional of the time-dependent density. That legitimizes a
time-dependent Kohn-Sham system whose density matches the true one, and
the *poles* of its density response to a weak perturbation are the exact
vertical excitation energies. TDDFT is the practical realization: perturb
the ground-state density at frequency omega, ask where the response
diverges, and read off the spectrum.

### The Casida eigenproblem

Linearizing the time-dependent Kohn-Sham equations about the ground state,
and expanding the response in the space of occupied-to-virtual orbital
transitions (labeled by an occupied index `i` and a virtual index `a`),
turns "find the poles of the response" into a matrix eigenvalue problem.
The excitation energies omega are the eigenvalues of the coupled
non-Hermitian system

$$
\begin{pmatrix} \mathbf{A} & \mathbf{B} \\ -\mathbf{B}^{*} & -\mathbf{A}^{*} \end{pmatrix}
\begin{pmatrix} \mathbf{X} \\ \mathbf{Y} \end{pmatrix}
= \omega
\begin{pmatrix} \mathbf{X} \\ \mathbf{Y} \end{pmatrix},
$$

where `X` collects the excitation amplitudes (occupied -> virtual) and `Y`
the de-excitation amplitudes (virtual -> occupied). The two orbital-rotation
Hessian blocks in the canonical MO basis are

$$
A_{ia,jb} = \delta_{ij}\delta_{ab}(\varepsilon_a - \varepsilon_i)
  + 2\,(ia|jb) - c_x\,(ij|ab) + (ia|f_\text{xc}|jb),
$$

$$
B_{ia,jb} = 2\,(ia|jb) - c_x\,(ib|ja) + (ia|f_\text{xc}|jb).
$$

The diagonal of `A` is just the orbital-energy difference of the bare
transition; the integral terms are the response corrections. The Hartree
term `2(ia|jb)` couples transitions through the Coulomb field, `c_x` is
the fraction of exact exchange (1 for Hartree-Fock, 0 for a pure
functional, the admixture alpha for a global hybrid such as 0.20 for
B3LYP), and `(ia|f_xc|jb)` is the adiabatic XC kernel, the piece that
distinguishes TDDFT from time-dependent Hartree-Fock. Because the
adiabatic kernel is local and frequency-independent, the same matrix
element enters both `A` and `B`.

For real orbitals the non-Hermitian problem is recast into the symmetric
form vibe-qc actually solves,

$$
(\mathbf{A}-\mathbf{B})^{1/2}\,(\mathbf{A}+\mathbf{B})\,(\mathbf{A}-\mathbf{B})^{1/2}\,\mathbf{Z}
  = \omega^{2}\,\mathbf{Z},
$$

whose eigenvalues are the squared excitation energies. The square root of
`A - B` requires that matrix to be positive definite, which holds for a
stable ground state; a negative eigenvalue there signals a triplet or
singlet instability of the reference, and the squared form makes that
failure visible as an imaginary omega.

### The Tamm-Dancoff approximation

The Tamm-Dancoff approximation (TDA) sets `B = 0`: it discards the coupling
between excitations and de-excitations entirely. The coupled non-Hermitian
problem then collapses to a single Hermitian eigenvalue problem

$$
\mathbf{A}\,\mathbf{X} = \omega\,\mathbf{X},
$$

a plain symmetric diagonalization of the `A` block alone. When the kernel
has no `f_xc` (pure exact exchange, `c_x = 1`) this is exactly
configuration interaction singles (CIS); with the DFT kernel it is
TDA-DFT. Dropping `B` is rarely a large error for low valence states (the
formaldehyde table above quantifies it at tens of meV), and it buys three
things: lower cost, a numerically more stable eigenproblem, and immunity to
the triplet-instability collapse, because the missing `A - B` square root
can no longer go imaginary. The trade is slightly worse oscillator
strengths and energies that are upper bounds relative to full response.

### Oscillator strengths

An excitation energy tells you where a peak is; the **oscillator
strength** tells you how tall it is. It is built from the transition dipole
moment between the ground state and excited state `I`, which in the AO
basis comes from the transition density assembled from the excitation
amplitudes,

$$
P^{0I}_{\mu\nu} = \sum_{ia} X^{I}_{ia}\,(C_{\mu i} C_{\nu a} + C_{\mu a} C_{\nu i}),
\qquad
\boldsymbol{\mu}_{0I} = \operatorname{tr}\!\big(\mathbf{P}^{0I}\,\hat{\boldsymbol{\mu}}\big).
$$

The dimensionless oscillator strength in the length gauge is then

$$
f_I = \tfrac{2}{3}\,\omega_I\,|\boldsymbol{\mu}_{0I}|^{2},
$$

with omega and the dipole in atomic units. A symmetry-forbidden
transition has a vanishing transition dipole and so `f = 0` (the
formaldehyde n -> pi\* state), while a strongly dipole-allowed transition
such as pi -> pi\* carries most of the spectral weight. The length-gauge
form above is accurate to roughly 5 percent for valence excitations, which
is why intensities are usually quoted to two significant figures rather
than more.

## Resources

The formaldehyde example runs in well under a second on one core after the
ground-state SCF, on the same order as the RKS reference itself. The cost
driver is the four-index AO-to-MO integral transformation and the
diagonalization of the transition-space matrix, both of which scale with
the number of occupied-virtual pairs; the ALDA/AGGA kernel adds one grid
pass per occupied-virtual pair. For the small 6-31G\* basis here that is
negligible, but it grows quickly with basis size, so request only as many
`n_states` as you need.

## References

- **TDDFT existence theorem.** E. Runge and E. K. U. Gross,
  "Density-Functional Theory for Time-Dependent Systems," *Phys. Rev.
  Lett.* **52**, 997 (1984). The time-dependent analogue of the
  Hohenberg-Kohn theorem that makes TDDFT well-defined.
- **Casida linear-response equations.** M. E. Casida, "Time-Dependent
  Density Functional Response Theory for Molecules," in *Recent Advances
  in Density Functional Methods*, Part I, ed. D. P. Chong, 155-192 (World
  Scientific, 1995). The working matrix formulation solved here.
- **Tamm-Dancoff approximation.** S. Hirata and M. Head-Gordon,
  "Time-dependent density functional theory within the Tamm-Dancoff
  approximation," *Chem. Phys. Lett.* **314**, 291 (1999). Introduces and
  benchmarks the `B = 0` approximation; the source of the TDA-B3LYP
  reference values.
- **Adiabatic hybrid kernel.** R. Bauernschmitt and R. Ahlrichs,
  "Treatment of electronic excitations within the adiabatic approximation
  of time dependent density functional theory," *Chem. Phys. Lett.* **256**,
  454 (1996). The composition of the hybrid TDDFT kernel as exact exchange
  plus `f_xc`.
- **Excited-states review.** A. Dreuw and M. Head-Gordon, "Single-Reference
  ab Initio Methods for the Calculation of Excited States of Large
  Molecules," *Chem. Rev.* **105**, 4009 (2005). A practical survey of
  TDDFT, its failures (charge-transfer, Rydberg, doubly excited states),
  and where TDA helps.

## Next

- [Natural orbitals](natural_orbitals.md) cover the ground-state density
  analysis that complements the natural transition orbitals you can extract
  from a TDDFT state with `compute_nto`, the most compact orbital picture of
  an excitation.
- [Molecular DFT](molecular_dft.md) is the ground-state reference every
  TDDFT run starts from; the functional and grid choices there set the
  accuracy ceiling for the excitations.
- [Open-shell SCF](open_shell.md) is the UHF/UKS reference that feeds
  `run_tddft_tda_uhf` for radicals and triplets.
