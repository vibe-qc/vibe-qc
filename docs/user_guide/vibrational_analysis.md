---
myst:
  html_meta:
    "description": "Reference for vibrational analysis in vibe-qc: finite-difference and analytic Hessians, harmonic frequencies, IR intensities, rigid-rotor harmonic-oscillator thermochemistry, partial Hessians, isotopes, the surface a run_job Hessian describes, periodic unit-cell Hessians and the preview Gamma-point GAPW phonons, with theory, options, output and every gate."
    "og:title": "vibe-qc user guide - vibrational analysis, IR intensities and thermochemistry"
---

# Vibrational analysis: Hessians, frequencies, IR intensities and thermochemistry

vibe-qc builds the nuclear Hessian of a molecule, diagonalises it in
mass-weighted coordinates for harmonic frequencies and normal modes, turns
dipole derivatives into infrared intensities, and feeds the frequencies into a
rigid-rotor harmonic-oscillator (RRHO) ideal-gas thermochemistry block. For
molecules this is production on the mean-field references RHF, UHF, ROHF, RKS
and UKS. `run_job(hessian=True)` uses finite differences of the analytic
gradient; the analytic coupled-perturbed Hessians are library functions. A
periodic `hessian=True` is the Hessian of the isolated unit-cell molecule, not a
phonon calculation, and the Gamma-point phonons of the GAPW route are a
preview. Read [Methods: theory and status](../methods.md#how-to-read-the-status-labels)
for the labels. Three behaviours of the current build deserve attention before
you use the numbers: the Hessian ignores `solvent=` and `dispersion=` silently,
the partial-Hessian mode table prints too few rows, and `phonon_dos` fails under
NumPy 2. They are described under [Limits and gates](#limits-and-gates).

The tutorials [Vibrational frequencies](../tutorial/vibrational_frequencies.md)
and [Thermodynamics at temperature](../tutorial/thermodynamics.md) work through
examples; this page is the reference.

## Quick start

Optimise water at RHF/STO-3G and take the harmonic frequencies, IR intensities
and thermochemistry at the optimised geometry, in one job.

```python
from vibeqc import Atom, Molecule, ThermoOptions, run_job

ang = 1.8897259886
mol = Molecule([
    Atom(8, [0.0, 0.0, 0.1173 * ang]),
    Atom(1, [0.0, 0.7572 * ang, -0.4692 * ang]),
    Atom(1, [0.0, -0.7572 * ang, -0.4692 * ang]),
], charge=0, multiplicity=1)

result = run_job(
    mol, basis="sto-3g", method="rhf",
    optimize=True, hessian=True,
    thermo_options=ThermoOptions(temperature=298.15, pressure=101325.0,
                                 symmetry_number=2),
    output="h2o_freq",
)
print(result.energy)
```

On a vibe-qc 0.17.10 development build (Apple M5 Max) the job takes about 3 s on an idle machine
(18 displaced SCF and gradient evaluations for the Hessian) and prints, after
the optimisation:

```text
  ## Vibrational Frequencies
  Surface: RHF/sto-3g
  Finite-difference Hessian  (step = 0.005 bohr,  18 displacements)
  Imaginary modes: 0
  Linear molecule: False

  Mode  Freq/cm⁻¹  IR/(km/mol)
  1        2169.9         7.23
  2        4139.3        44.28
  3        4390.3        29.97

  ## Thermochemistry (RRHO ideal gas)
  T = 298.15 K   p = 101325 Pa   s_rot = 2   rotor = nonlinear
  Zero-point energy        =     0.0243751537 Ha  (   15.296 kcal/mol)
  Thermal corr. to U       =     0.0272079885 Ha
  Thermal corr. to H       =     0.0281521734 Ha
  Thermal corr. to G       =     0.0066367761 Ha
  Total entropy S          =     0.0000721630 Ha/K  (  45.283 cal/mol/K)
  E(RHF) + ZPE             =   -74.9415260363 Ha
  H = E(RHF) + H_corr      =   -74.9377490165 Ha
  G = E(RHF) + G_corr      =   -74.9592644138 Ha
```

The frequencies are the three vibrations (bend, symmetric stretch, asymmetric
stretch); the six translational and rotational modes are projected out and
not listed. STO-3G stretches are far too high, as expected for a minimal basis.
Set `symmetry_number` yourself: the default 1 is right only for an asymmetric top
and would overstate the rotational entropy of water by $k_B\ln 2$. The
`.system` manifest gains a `[hessian]` section and the QVF archive gains
`vibrations`, `spectra.ir` and a root `thermochemistry` record.

The same calculation through the library, with a check against the analytic
Hessian:

```python
import numpy as np
from vibeqc import (Atom, Molecule, BasisSet, RHFOptions, run_rhf,
                    HessianFDOptions, compute_hessian_fd, ir_intensities,
                    compute_thermochemistry, ThermoOptions,
                    compute_hessian_rhf_analytic)

ang = 1.8897259886
lines = open("h2o_freq.xyz").read().strip().splitlines()[2:]   # optimised geometry from the job above
sym = {"O": 8, "H": 1}
mol = Molecule([Atom(sym[l.split()[0]], [float(x) * ang for x in l.split()[1:4]])
                for l in lines], 0, 1)

hess = compute_hessian_fd(
    mol, "sto-3g", method="RHF",
    hessian_options=HessianFDOptions(step_bohr=0.005, include_dipole_derivatives=True),
)
print(np.round(hess.frequencies_cm1[6:], 1), np.round(ir_intensities(hess)[6:], 2))
th = compute_thermochemistry(mol, hess, options=ThermoOptions(symmetry_number=2))
print(f"ZPE {th.zpe:.6f}  H_corr {th.h_thermal:.6f}  G_corr {th.g_thermal:.6f} Ha, "
      f"S {th.s_total * 627.509474 * 1000:.3f} cal/mol/K, rotor {th.rotor_type}")

basis = BasisSet(mol, "sto-3g")
opts = RHFOptions(); opts.conv_tol_energy = 1e-10
ana = compute_hessian_rhf_analytic(mol, basis, run_rhf(mol, basis, opts), basis_name="sto-3g")
print(np.round(ana.frequencies_cm1[6:], 1),
      f"max |H_fd - H_analytic| = {np.max(np.abs(hess.hessian - ana.hessian)):.1e} Ha/bohr^2")
```

```text
[2169.9 4139.3 4390.3] [ 7.23 44.28 29.97]
ZPE 0.024375  H_corr 0.028152  G_corr 0.006637 Ha, S 45.283 cal/mol/K, rotor nonlinear
[2169.9 4139.3 4390.3] max |H_fd - H_analytic| = 1.0e-05 Ha/bohr^2
```

The finite-difference and analytic routes agree on the frequencies to 0.1
cm$^{-1}$ (the matrices differ by $10^{-5}$ Ha/bohr$^2$, the finite-difference
noise). For this three-atom molecule the analytic route (1.6 s) is not faster
than the finite-difference one (1.0 s); it wins for larger molecules because it
needs one SCF instead of $6N$.

## Theory

Near a stationary point the energy is quadratic in the nuclear displacements,
$E(\mathbf R)\approx E_0+\tfrac12\,\Delta\mathbf x^{\mathsf T}\mathbf H\,\Delta\mathbf x$,
with the Cartesian Hessian $H_{ia,jb}=\partial^2E/\partial R_{ia}\partial R_{jb}$
($i,j$ atoms, $a,b\in\{x,y,z\}$). Newton's equations of motion in mass-weighted
coordinates $q_{ia}=\sqrt{m_i}\,x_{ia}$ separate into normal modes: the
eigenvalues $\omega_k^2$ of the mass-weighted Hessian
$\mathbf F=\mathbf M^{-1/2}\mathbf H\,\mathbf M^{-1/2}$ give the harmonic
frequencies $\tilde\nu_k=\omega_k/(2\pi c)$ and its eigenvectors the mode
shapes (Wilson, Decius and Cross, {vibeqc-cite}`wilson_decius_cross_1955`). At a
true stationary point six of the $3N$ eigenvalues (five for a linear molecule)
belong to translations and rotations and are zero. The code projects those
directions out of $\mathbf F$ before diagonalising, so they come out as exact
zeros at the head of `frequencies_cm1` and the vibrational modes are the entries
after them. A negative $\omega^2$ is reported as a negative wavenumber
(an imaginary mode, a saddle direction).

**Finite-difference Hessian.** `compute_hessian_fd` differentiates the analytic
gradient $\mathbf g=\partial E/\partial\mathbf R$ by central differences,
$H_{\cdot,jb}\approx[\mathbf g(\mathbf R+\delta\,\mathbf e_{jb})-\mathbf g(\mathbf R-\delta\,\mathbf e_{jb})]/2\delta$, with the basis rebuilt at every
displaced geometry, then symmetrises $\mathbf H\leftarrow(\mathbf H+\mathbf H^{\mathsf T})/2$.
It costs $6N$ converged SCFs and gradients and reproduces the Hessian of whatever
Hamiltonian the gradient differentiates (ECP included). The step $\delta$ is
0.005 bohr by default: smaller steps amplify SCF noise, larger ones admit
anharmonic contamination.

**Analytic Hessian.** Differentiating the SCF stationarity condition once more
gives the coupled-perturbed Hartree-Fock (CPHF) or Kohn-Sham (CPKS) equations for
the orbital response ({vibeqc-cite}`pople_derivatives_1979`), and
$\mathbf H=\mathbf H^{\text{skeleton}}+\mathbf H^{\text{response}}$. In vibe-qc the
skeleton second derivatives of the integrals come from libint, the derivative of
the AO Fock matrix at fixed density is obtained by central differences of Fock
builds (about $6N$ of them, far cheaper than $6N$ SCFs), and the CPHF/CPKS linear
system is solved once per Cartesian perturbation. The result is returned in the
same `HessianResult` shape, so frequencies, IR intensities and thermochemistry
consume either route.

**IR intensities.** The harmonic intensity of mode $p$ is
$I_p=\dfrac{N_A\pi}{3c^2}\,\Big|\dfrac{\partial\boldsymbol\mu}{\partial Q_p}\Big|^2$,
where $Q_p$ is the mass-weighted normal coordinate. The dipole derivatives
$\partial\mu_\beta/\partial R_{ia}$ are taken from the same displaced geometries as
the Hessian at the cost of one dipole evaluation each, and rotated into the normal
coordinates with $\partial\mu_\beta/\partial Q_p=\sum_k(\partial\mu_\beta/\partial R_k)\,L_{kp}/\sqrt{M_k}$,
where $\mathbf L$ holds the mode eigenvectors. Atomic units convert to km/mol with
the usual factor 974.864. Modes with $|\tilde\nu|<1$ cm$^{-1}$ are set to zero
intensity.

**Thermochemistry.** For an ideal gas of non-interacting molecules in the
rigid-rotor harmonic-oscillator approximation the partition function factorises,
$q=q_\text{trans}\,q_\text{rot}\,q_\text{vib}\,q_\text{elec}$, and every
thermodynamic function follows from its logarithm. With
$x_k=\hbar\omega_k/k_BT$:

$$
\text{ZPE}=\tfrac12\sum_k\hbar\omega_k,\qquad
E_\text{vib}=\sum_k\frac{\hbar\omega_k}{e^{x_k}-1},\qquad
S_\text{vib}=k_B\sum_k\Big[\frac{x_k}{e^{x_k}-1}-\ln\big(1-e^{-x_k}\big)\Big],
$$

$$
S_\text{trans}=k_B\Big[\ln\Big(\frac{(2\pi Mk_BT/h^2)^{3/2}\,k_BT}{P}\Big)+\tfrac52\Big],\qquad
S_\text{rot}=k_B\Big[\ln q_\text{rot}+\tfrac32\Big],\quad
q_\text{rot}=\frac{\sqrt\pi}{\sigma}\frac{(k_BT/h)^{3/2}}{\sqrt{B_aB_bB_c}},
$$

with $\sigma$ the rotational symmetry number, $B_i=h/(8\pi^2I_i)$ the rotational
constants (for a linear molecule $q_\text{rot}=k_BT/(\sigma hB)$ and the constant
is $+1$ instead of $+\tfrac32$), and $S_\text{elec}=k_B\ln g$ with $g$ the spin
degeneracy. The corrections that are added to the electronic energy are
$U=\text{ZPE}+E_\text{trans}+E_\text{rot}+E_\text{vib}$, $H=U+k_BT$ and
$G=H-TS$. Modes below $-1$ cm$^{-1}$ (imaginary) are left out of $q_\text{vib}$ and
counted in `n_imaginary_modes_excluded`; modes between $-1$ and $1$ cm$^{-1}$ are
treated as projected zeros. No quasi-harmonic treatment of soft modes is applied.

Harmonic frequencies overestimate experimental fundamentals by roughly 10 to 15
percent at Hartree-Fock and 3 to 5 percent for common hybrid functionals; scale
them before comparing to spectra (for example {vibeqc-cite}`wong_scaling_1996`).
The harmonic approximation fails for soft, anharmonic modes below about 100
cm$^{-1}$ (torsions, floppy dimers), for saddle points (imaginary modes have no
partition function), and for the entropy of hindered rotors, where the RRHO
result is the wrong physical model.

**Gamma-point phonons.** For a periodic cell the analogue of the Hessian is the
dynamical matrix at $\mathbf q=0$,
$D_{ia,jb}=\dfrac{1}{\sqrt{M_iM_j}}\dfrac{\partial^2E}{\partial u_{ia}\,\partial u_{jb}}$,
with $\mathbf u$ the displacements of the atoms of one unit cell. The GAPW route
builds it by central differences of the analytic GAPW atomic force and
diagonalises it; only $\mathbf q=0$ is sampled.

## Options

`run_job` keywords:

| Keyword | Default | Effect |
|---|---|---|
| `hessian` | `False` | After the SCF (and after `optimize=True`, if given), build the FD Hessian, print frequencies, IR intensities and thermochemistry, and store them in the QVF. Cost about $6N$ SCFs. |
| `hessian_frozen_indices` | `None` | 0-based atom indices to hold fixed. Only the unfrozen atoms are displaced ($6M$ SCFs) and the result is the $(3M)\times(3M)$ partial-Hessian block. |
| `thermo_options` | `None` | A `ThermoOptions`; `None` uses 298.15 K, 101325 Pa, symmetry number 1. |

`ThermoOptions(temperature=298.15, pressure=101325.0, symmetry_number=1, electronic_degeneracy=None)`:
temperature in K, pressure in Pa, the rotational symmetry number (set it: H$_2$,
H$_2$O, N$_2$ are 2, NH$_3$ 3, CH$_4$ 12, benzene 12), and the electronic
degeneracy, which defaults to the spin multiplicity $2S+1$ and needs an override for
orbitally degenerate states such as $^2\Pi$ NO.

`HessianFDOptions(step_bohr=0.005, sym_strict=True, project_trans_rot=True, atomic_masses_amu=None, include_dipole_derivatives=False, frozen_indices=None)`:
`atomic_masses_amu` gives isotope masses (replacing H by D on the water above gives
1584.9, 2990.7 and 3211.0 cm$^{-1}$); `include_dipole_derivatives=True` is
required by `ir_intensities`; `frozen_indices` is the partial Hessian.

Library functions, all importable from `vibeqc`:

| Function | Purpose |
|---|---|
| `compute_hessian_fd(mol, basis_name, method="RHF", *, scf_options, grid_options, grid_level, gradient_options, hessian_options)` | FD Hessian for `RHF`, `UHF`, `ROHF`, `RKS`, `UKS`; returns `HessianResult`. |
| `compute_hessian_{rhf,uhf,rks,uks}_analytic(mol, basis, result, *, basis_name, ...)` | CPHF/CPKS Hessians from a converged result; `basis_name` is required. The KS versions take `functional` and `grid_options`. |
| `ir_intensities(hessian_result)` | IR intensities in km/mol, indexed like `frequencies_cm1`. |
| `compute_thermochemistry(mol, hessian_result, *, options)` | `ThermoResult` with `zpe`, `e_trans`, `e_rot`, `e_vib`, `u_thermal`, `h_thermal`, `g_thermal`, entropies `s_trans`, `s_rot`, `s_vib`, `s_elec`, `s_total`, heat capacities, `n_imaginary_modes_excluded`, `rotor_type`. |
| `compute_hessian_gpw(system, basis, result, *, basis_name, fd_step_bohr=0.02, ...)` and `compute_vibrational_frequencies(hessian, masses_amu)` | GPW Gamma-point atomic Hessian and its wavenumbers. |
| `vibeqc.periodic_gapw_phonon.PhononCalculator`, `compute_dynamical_matrix_fd`, `phonon_eigenvalues`, `phonon_dos` | Preview GAPW Gamma phonons. |

`HessianResult` holds `hessian` and `hessian_mw` (Ha/bohr$^2$ and mass-weighted),
`frequencies_cm1` (ascending, imaginary negative, projected zeros exactly 0),
`normal_modes` (mass-weighted columns; the Cartesian pattern of atom $i$ in mode
$k$ is `normal_modes[3i:3i+3, k] / sqrt(M_i)`), `imaginary_count`,
`n_displacements`, `is_linear`, `dipole_derivatives`, `dipole_origin`,
`masses_amu` and `unfrozen_indices`. Energies are in Hartree and entropies in
Hartree per Kelvin per molecule.

## Output

- **`.out`**: `## Vibrational Frequencies` with the `Surface:` line, the step and
  displacement count, the imaginary-mode count and the mode table (the six or five
  projected zeros are not listed; imaginary modes carry a trailing `i`), then
  `## Thermochemistry (RRHO ideal gas)`. The last three rows name the electronic
  energy they add to (`E(RHF)`, `E(RKS)`, ...). If an IR column cannot be computed
  the table falls back to frequencies only with a warning line.
- **`.system`**: a `[hessian]` section with `requested_method`, `surface`,
  `surface_method`, `surface_is_requested_method` (a machine-readable flag) and
  `available`.
- **QVF**: `vibrations` (frequencies, un-mass-weighted displacement patterns,
  `surface` in the metadata), `spectra.ir`, and a root `thermochemistry` record with
  `zpve_eh`, `enthalpy_eh` (the total $E+H_\text{corr}$), `entropy_cal_mol_k`,
  `gibbs_free_energy_eh`, `temperature_k` and `pressure_atm`. vibe-view animates
  the modes.
- **Citations**: the Wilson-Decius-Cross entry fires with the harmonic analysis.

### Which surface the frequencies describe

`run_job(hessian=True)` differentiates the analytic gradient of a **mean-field**
reference. A post-SCF request such as `method="mp2"`, `"ccsd"`, `"dlpno-ccsd"` or
`"ovgf"` resolves to its RHF or UHF reference first, so the frequencies, modes and
thermochemistry describe the reference surface, not the correlated one. The
`.out` says so:

```text
  Surface: RHF/sto-3g  (NOT MP2)
  This job's energy is MP2, but the Hessian finite-differences
  the analytic RHF gradient. The frequencies, normal modes and
  thermochemistry below describe the RHF reference surface;
  no MP2 second derivatives were computed.
```

and the thermochemistry rows read `E(RHF) + ZPE`, not the MP2 total. To get a
correlated enthalpy add the printed thermal correction to the correlated energy
yourself. A method that resolves to no mean-field reference (CASSCF, CISD, CASPT2,
NEVPT2, FCI, selected CI, DMRG) prints `SKIPPED -- no Hessian is available on the
<method> surface` and reports no frequencies.

## Limits and gates

- **Methods**: the Hessian needs a Gaussian-basis mean-field reference
  (`rhf`, `uhf`, `rohf`, `rks`, `uks`). Semiempirical and MLIP methods print
  `SKIPPED` with a message. `method="roks"` raises `NotImplementedError` before the
  SCF because the FD Hessian needs the ROKS analytic gradient, which is missing
  (ROHF Hessians work). An unconverged SCF prints `SKIPPED -- SCF did not converge`.
  Full-grid external XC functionals (for example SKALA) raise
  `NotImplementedError`. Functionals without a complete analytic gradient
  (range-separated hybrids, VV10; GitLab #571) fail closed on the FD Hessian with
  the same message the optimisers use.
- **`solvent=` and `dispersion=` are not part of the Hessian.** The displaced
  SCFs and gradients that build the Hessian use the plain SCF options of the run,
  not the solvent or dispersion wrappers. Verified on water at RKS/PBE/STO-3G
  (a fixed geometry): the run with `dispersion="d3bj"` and the run with
  `solvent="water"` print frequencies (1894.3, 3986.5, 4259.5 cm$^{-1}$),
  intensities and thermal corrections that are identical to the gas-phase,
  dispersion-free run, and nothing in the `.out` says so. The thermochemistry
  rows then add those corrections to `result.energy`, which is the bare SCF
  energy under D3 (the dispersion energy is missing from `E + ZPE`, `H` and
  `G`; add `result.e_dispersion`) and the in-solvent total under CPCM (a
  solvated energy combined with gas-phase corrections). Treat frequencies and
  thermochemistry from such a job as gas-phase and dispersion-free, and combine
  the pieces by hand.
- **ECPs**: the finite-difference Hessian keeps the ECP operator (centres follow
  their atoms). The analytic Hessians refuse molecular ECPs
  (`compute_hessian_*_analytic` raise before computing).
- **Analytic Hessians** exist for RHF, UHF, RKS and UKS. Meta-GGA and
  range-separated analytic Hessians are not implemented. Their accuracy differs by
  reference. RHF matches the finite-difference Hessian to $10^{-5}$ Ha/bohr$^2$
  (above). For closed-shell RKS on water at STO-3G, LDA agrees with the
  finite-difference matrix to $1.1\times10^{-5}$ Ha/bohr$^2$ and identical
  frequencies, but PBE differs by $3.4\times10^{-3}$ Ha/bohr$^2$ (frequencies
  1890.7, 3978.8, 4248.0 against 1894.3, 3986.5, 4259.5 cm$^{-1}$) and PBE0 by
  $5.0\times10^{-3}$ Ha/bohr$^2$ (shifts of 7 to 12 cm$^{-1}$). The RKS module
  documents that the GGA part of the CPKS response uses LDA $f_{xc}$ terms only. For
  quantitative GGA or hybrid frequencies use `compute_hessian_fd` or
  `run_job(hessian=True)`, which differentiate the full analytic gradient. The UKS
  route was not compared here.
- **Stationary point**: the harmonic formulas assume $\mathbf g=0$. Pass a
  tightly optimised geometry; loose thresholds show up as spurious low-frequency
  modes of tens of cm$^{-1}$. The FD Hessian does not check this for you.
- **Partial Hessians** (`hessian_frozen_indices`): the frozen atoms anchor the
  system, so there is no translation or rotation. Thermochemistry treats the
  result as vibrational-only (`rotor = frozen`, no translational, rotational or $PV$
  terms). The mode table in the `.out` currently skips its first six rows as if
  they were translations and rotations, so a partial Hessian with three
  unfrozen atoms prints three of its nine modes, and water with the oxygen frozen
  prints none, although the thermochemistry uses all six. Read the frequencies
  from the QVF or from `compute_hessian_fd(..., hessian_options=HessianFDOptions(frozen_indices=[0]))`,
  which returns all six (18.2, 23.1, 25.1, 2087.9, 4053.9, 4236.4 cm$^{-1}$ on the
  unoptimised water above; the three low modes are the frozen-oxygen artefacts).
- **Periodic `hessian=True`** runs `compute_hessian_fd` on `system.unit_cell_molecule()`
  with default molecular SCF options for the method (RHF, UHF, RKS with `functional`,
  UKS). It ignores the lattice, the k mesh and the Coulomb route, and prints
  `Finite-difference Hessian (unit cell)`. Verified for an H$_2$ molecule in a 12
  bohr GDF Gamma cell: one mode, 5027.1 cm$^{-1}$, the value of the isolated
  molecule at RHF/STO-3G, 12 displacements. There is no thermochemistry block. Use
  it for vacuum-padded molecules only. It raises `NotImplementedError` on GAPW
  RHF/UHF, BIPOLE, the AICCM variants, slab GDF, periodic ROHF/ROKS, full-grid
  external XC and the basis-free semiempirical routes.
- **GPW Hessian**: `compute_hessian_gpw` takes a converged Gamma `GpwScfResult` of a
  3-D cell (`NotImplementedError` otherwise) and differences the analytic GPW
  gradient, step 0.02 bohr. Use `compute_vibrational_frequencies` to get wavenumbers.

### Preview: Gamma-point GAPW phonons

```{admonition} Status: preview
:class: note

The GAPW Gamma-point phonon route is a maintained preview: `PhononCalculator`
differences the analytic GAPW atomic force of a closed-shell RHF or RKS cell
(odd electron counts raise; RHF needs the explicit `one_centre="block"` option)
and diagonalises the dynamical matrix at $\mathbf q=0$ only. It fails closed for
non-3-D cells. `get_bandstructure` returns a flat surface that repeats the Gamma
frequencies at every $\mathbf q$; it is not a force-constant interpolation. See
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc.periodic_gapw_grid import PlaneWaveGrid
from vibeqc.periodic_gapw_phonon import PhononCalculator

warnings.simplefilter("ignore")                     # GAPWExperimentalWarning per build
L = 6.0                                             # bohr, cubic box
lat = np.array([[L, 0, 0], [0, L, 0], [0, 0, L]], dtype=float)
system = vq.PeriodicSystem(3, lat.tolist(),
    [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]), vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])])
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
grid = PlaneWaveGrid(lat.T, 24, 24, 24)             # coarse grid, for speed

pc = PhononCalculator(system, basis, "sto-3g", functional="lda", grid=grid)
pc.compute_phonons(disp=0.01)
freqs_cm1, freqs_thz, modes = pc.diagonalize_dynamical_matrix()
print(np.round(freqs_cm1, 1))
```

This is a slow call: on the build above, with a $24^3$ grid, it took about 570 s
(24 displaced GAPW SCFs and forces for two atoms). The output for that run is

```text
[-1400.7 -1400.7    -0.8    -0.8     3.6  4752.6]
```

Read it as a demonstration of the API, not of physical phonons. The geometry
(bond 1.4 bohr, coarse grid, STO-3G, LDA) is not a stationary point of this
energy surface, and the dynamical matrix is not projected for translations and
rotations, so the two rotations of the diatomic appear as large imaginary values
($-1400$ cm$^{-1}$) instead of zeros; the stretch is the last entry. For a real
calculation start from a relaxed cell, a converged grid and a converged basis.

```{warning}
`phonon_dos` currently fails under NumPy 2: it calls `numpy.trapz`, which was removed
in NumPy 2.0, and raises `AttributeError: module 'numpy' has no attribute 'trapz'`
(verified with NumPy 2.5.3; `pyproject.toml` allows `numpy>=1.26`). Build the
density of states yourself until this is fixed, for example as a sum of Gaussians
of the entries of `freqs_cm1`.
```

## See also

- [Vibrational frequencies](../tutorial/vibrational_frequencies.md) and
  [Thermodynamics at temperature](../tutorial/thermodynamics.md): worked examples
  with ASE.
- [Geometry optimization](geometry_optimization.md): the tight stationary point the
  harmonic formulas need, and [IRC and the dimer method](irc_and_dimer.md) for
  saddle points, where the Hessian must show exactly one imaginary mode.
- [Solvation](solvation.md), [ECPs](ecp.md), [Functionals](functionals.md): what
  each option means, and which of them the Hessian does not see.
- [Output files](output_files.md), [QVF and vibe-view](../visualization.md) and
  [Citations](citations.md).
- [GAPW](gapw.md) and [Periodic methods](periodic_methods.md) for the phonon route.
- [Methods: theory and status](../methods.md) and the [feature matrix](../features.md).
