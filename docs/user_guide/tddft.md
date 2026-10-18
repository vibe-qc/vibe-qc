---
myst:
  html_meta:
    "description": "Reference for linear-response TDDFT, the Tamm-Dancoff approximation and CIS in vibe-qc: run_job(tddft=True), run_tddft_tda and run_tddft_casida, unrestricted, ROHF and Gamma-point periodic variants, natural transition orbitals, finite-difference excited-state gradients, the vibeqc tddft CLI, output files, and every gate the code enforces."
    "og:title": "vibe-qc user guide - TDDFT, TDA and CIS excited states"
---

# TDDFT, TDA and CIS excited states

vibe-qc computes vertical excitation energies, oscillator strengths, transition
dipoles and natural transition orbitals (NTOs) from a converged ground-state
reference by linear-response time-dependent density-functional theory (TDDFT),
in its full Casida form or in the Tamm-Dancoff approximation (TDA). On a
Hartree-Fock reference the TDA is configuration interaction singles (CIS). The
energies are production for restricted and unrestricted references with LDA,
GGA and global-hybrid kernels; the finite-difference excited-state gradient is a
preview that covers only the closed-shell RHF/CIS surface. Two defects in the
current build affect printed numbers and are described under
[Known defects](#known-defects-in-this-build). The label vocabulary is defined
in [Methods: theory and status](../methods.md#how-to-read-the-status-labels).
The tutorial [Excited states with TDDFT](../tutorial/excited_states_tddft.md)
works one molecule through the physics; this page is the reference.

## Quick start

Water at PBE0/6-31G\*, the TDA and the full Casida equation, through `run_job`.
The excitation problem is solved after the SCF converges and written into the
`.out`.

```python
from vibeqc import Atom, Molecule, run_job

ang = 1.8897259886
mol = Molecule([
    Atom(8, [0.0, 0.0, 0.1173 * ang]),
    Atom(1, [0.0, 0.7572 * ang, -0.4692 * ang]),
    Atom(1, [0.0, -0.7572 * ang, -0.4692 * ang]),
], charge=0, multiplicity=1)

result = run_job(
    mol, basis="6-31g*", method="rks", functional="pbe0",
    tddft=True, tddft_type="tda", tddft_n_states=4, tddft_spectrum=True,
    output="h2o_tda",
)
run_job(
    mol, basis="6-31g*", method="rks", functional="pbe0",
    tddft=True, tddft_type="casida", tddft_n_states=4,
    output="h2o_casida",
)
```

On a vibe-qc 0.17.10 development build (Apple M5 Max) each job takes on the order of
ten seconds on an otherwise idle machine (several times that on a busy one), most of
it the XC-kernel evaluation (see [Cost](#cost)). The SCF energy is -76.3238384517 Ha and
the `.out` carries this block (Casida below it):

```text
  ## TD-DFT excited states (TDA (pbe0))
  State   E (eV)  λ (nm)   f_osc  Dominant transition
      1   8.4355   147.0  0.0330  5→1 (0.999), 5→6 (0.016), 5→4 (0.015)
      2  10.4117   119.1  0.0000  5→2 (0.999), 5→7 (0.033), 5→3 (0.026)
      3  10.9988   112.7  0.2089  4→1 (0.990), 3→2 (0.125), 5→5 (0.042)
      4  13.1764    94.1  0.1700  4→2 (0.982), 3→1 (0.176), 3→4 (0.036)

  ## TD-DFT excited states (Casida (pbe0))
      1   8.4037   147.5  0.0279  5→1 (0.999), 5→4 (0.027), 5→8 (0.016)
      2  10.4037   119.2  0.0000  5→2 (0.999), 5→7 (0.037), 5→3 (0.030)
      3  10.9262   113.5  0.2429  4→1 (0.996), 3→2 (0.078), 5→5 (0.018)
      4  13.1031    94.6  0.2248  4→2 (0.987), 3→1 (0.151), 4→7 (0.032)
```

In the dominant-transition column the occupied index counts orbitals from the
lowest (5 is the HOMO) and the virtual index from the LUMO (1 is the LUMO), both
1-based. State 1 is therefore HOMO to LUMO. State 2 has $f = 0$ because the
transition is dipole-forbidden by symmetry. The full Casida energies lie 0.03 to
0.08 eV below the TDA here, the usual size of the $\mathbf B$ matrix effect for
valence states. The oscillator-strength column is printed as the code computes
it; see the defect note before quoting it.

The run also writes the usual `h2o_tda.bibtex` and `h2o_tda.references`
siblings. `tddft_spectrum=True` adds a Gaussian-broadened text spectrum to the
`.out` (see [Output](#output)).

```{note}
`run_job` cannot attach the TDDFT result to a pybind11 SCF result object (the
RHF/RKS/UHF/UKS result types have no `__dict__`), so `result.tddft` is usually
not available. Read the `.out` and the QVF archive, or call the solvers
directly as shown in [The library route](#the-library-route).
```

## Theory

A weak time-dependent field perturbs the ground-state density, and the poles of
the linear density response are the excitation energies. Written in the basis of
occupied-virtual orbital pairs $(ia)$ for a closed-shell reference, the poles
solve the non-Hermitian Casida equation
({vibeqc-cite}`casida_tddft_1995`; the underlying existence theorem is
{vibeqc-cite}`runge_gross_1984`)

$$
\begin{pmatrix} \mathbf A & \mathbf B \\ \mathbf B & \mathbf A \end{pmatrix}
\begin{pmatrix} \mathbf X \\ \mathbf Y \end{pmatrix}
= \omega
\begin{pmatrix} \mathbf 1 & \mathbf 0 \\ \mathbf 0 & -\mathbf 1 \end{pmatrix}
\begin{pmatrix} \mathbf X \\ \mathbf Y \end{pmatrix},
$$

with the spin-adapted singlet blocks that `vibeqc.tddft` builds,

$$
\begin{aligned}
A_{ia,jb} &= \delta_{ij}\delta_{ab}(\varepsilon_a-\varepsilon_i)
  + 2(ia|jb) - c_x\,(ij|ab) + (ia|f_{xc}|jb),\\
B_{ia,jb} &= 2(ia|jb) - c_x\,(ib|ja) + (ia|f_{xc}|jb).
\end{aligned}
$$

Here $\varepsilon$ are the ground-state orbital energies, $(pq|rs)$ are
two-electron integrals in chemist's notation over the ground-state orbitals,
$c_x$ is the exact-exchange fraction and $f_{xc}$ is the adiabatic
exchange-correlation kernel. The approximations that enter are the adiabatic
approximation (the kernel is local in time, so it has no frequency dependence
and cannot describe states of double-excitation character) and the choice of
functional in $f_{xc}$. The exchange fraction is $c_x = 1$ for a Hartree-Fock
reference, $0$ for a pure functional, and the functional's admixture for a
global hybrid, following {vibeqc-cite}`bauernschmitt_ahlrichs_1996` (B3LYP 0.20,
PBE0 0.25). The kernel is the LDA or GGA second functional derivative from
libxc, contracted on the reference's DFT grid.

The code solves the equation in the reduced form
$(\mathbf A-\mathbf B)^{1/2}(\mathbf A+\mathbf B)(\mathbf A-\mathbf B)^{1/2}\mathbf Z=\omega^2\mathbf Z$, which needs $\mathbf A-\mathbf B$ to be positive
definite and clamps its negative eigenvalues to zero. That holds for a stable
ground state; an unstable reference (triplet or orbital instability) makes the
result meaningless, so run the SCF stability analysis first.

Setting $\mathbf B=\mathbf 0$ gives the **Tamm-Dancoff approximation**
({vibeqc-cite}`hirata_headgordon_tda_1999`): a Hermitian eigenproblem for
$\mathbf A$ alone, which is faster, immune to the triplet-instability collapse of
the full equation, and usually within 0.1 to 0.3 eV of it for valence states. On
a Hartree-Fock reference the TDA is **CIS**
({vibeqc-cite}`foresman_cis_1992`), the variational configuration interaction of
singly excited determinants. The singlet and triplet CIS matrices, implemented
reference-agnostically in `vibeqc.excited.cis_excitations`, are

$$
A^{\text{singlet}}_{ia,jb} = \delta_{ij}\delta_{ab}(\varepsilon_a-\varepsilon_i) + 2(ia|jb) - (ij|ab),
\qquad
A^{\text{triplet}}_{ia,jb} = \delta_{ij}\delta_{ab}(\varepsilon_a-\varepsilon_i) - (ij|ab).
$$

The **oscillator strength** in the length gauge is
$f_I=\tfrac23\,\omega_I\,|\boldsymbol\mu_{0I}|^2$ in atomic units, where the
transition dipole is contracted from the transition density built from the
eigenvector (for the full equation, the vector $\mathbf X+\mathbf Y$, normalised
to unit length). **Natural transition orbitals**
({vibeqc-cite}`martin_nto_2003`) compress an excitation into its most compact
orbital description: the singular value decomposition
$\mathbf X=\mathbf U\,\boldsymbol\sigma\,\mathbf V^{\mathsf T}$ of the
$n_\text{occ}\times n_\text{virt}$ amplitude matrix gives hole orbitals
$\mathbf C_\text{occ}\mathbf U$, particle orbitals $\mathbf C_\text{virt}\mathbf V$
and weights $\sigma_k$ (with $\sum_k\sigma_k^2=1$ for a normalised TDA vector).
A single weight near 1 means the excitation is one hole-particle pair.

TDDFT is the standard tool for low-lying valence excitations of closed-shell
molecules. In this implementation the cost is dominated by the XC kernel, not by the
integrals (see [Cost](#cost)). It fails for
charge-transfer and long-range states with a pure or low-exchange functional
(the kernel lacks the $-1/R$ asymptote), for Rydberg states in bases without
diffuse functions, for states with double-excitation character, and near
conical intersections. Range-separated hybrids, the usual remedy for
charge-transfer states, are not available in the response kernel here.

## Options

`run_job` keywords (molecular):

| Keyword | Default | Effect |
|---|---|---|
| `tddft` | `False` | Run the excitation calculation after the SCF converges. Needs an SCF that produces MO coefficients (RHF, RKS, UHF, UKS, ROHF TDA). Skipped without a message if the SCF did not converge. |
| `tddft_n_states` | `5` | Number of roots. |
| `tddft_type` | `"tda"` | `"tda"` or `"casida"`. |
| `tddft_spectrum` | `False` | Append a Gaussian-broadened text spectrum to the `.out` (sigma 0.15 eV, 200 grid points, normalised to a maximum of 1). |
| `nto` | `False` | Compute NTOs for every state and embed them in the QVF archive. Runs inside the QVF writer, so it needs `output_qvf=True`. |
| `tddft_molden` | `False` | With `nto=True`, also write one Molden file per state and role, `<stem>.nto_S<n>_hole.molden` and `<stem>.nto_S<n>_electron.molden`. |
| `tddft_gradient` | `False` | Finite-difference S1 gradient on the RHF/CIS surface only. See [Excited-state gradients](#excited-state-gradients). |

`run_periodic_job` accepts `tddft` and `tddft_n_states` only; the solver is always
the TDA at the Gamma point. The library entry points, all importable from
`vibeqc`:

| Function | Reference | Notes |
|---|---|---|
| `run_tddft_tda(molecule, basis, mo_energies, mo_coeff, n_occ, *, n_states=5, functional=None, density_ao=None, output=None, grid_options=None)` | RHF, RKS | Singlets. `functional=None` is CIS. |
| `run_tddft_casida(...)` | RHF, RKS | Same signature; full linear response. |
| `run_tddft_tda_uhf`, `run_tddft_casida_uhf` | UHF, UKS | Take separate alpha and beta energies, coefficients, occupations and `density_alpha_ao` / `density_beta_ao`. |
| `run_tddft_tda_rohf(molecule, basis, mo_energies, mo_coeff, n_docc, n_socc, *, n_states=5, output=None)` | ROHF | Doubly-occupied to virtual space only. |
| `run_tddft_tda_periodic(result, basis, *, n_occ, n_states=5, functional=None, output=None)` | periodic Gamma | See [Limits and gates](#limits-and-gates). |
| `compute_nto(state, mo_coeff, n_occ)`, `compute_nto_uhf(...)` | | Return `NTOResult`; `compute_nto_uhf` returns a dict with `"alpha"` and `"beta"` entries. |
| `vibeqc.excited.cis_excitations(eps, n_occ, ovov, oovv, *, spin="singlet", n_states=5)` | any | Singlet or triplet CIS from orbital energies and the $(ia\vert jb)$, $(ij\vert ab)$ blocks. |
| `vibeqc.excited_gradient.cis_state_gradients_fd`, `make_hf_cis_energy_fn` | RHF | State-tracked finite-difference gradients. |

Pass `density_ao` (and, for the KS functional, the reference's `grid_options`) so
the $f_{xc}$ kernel is included. Without a density and with a hybrid functional
the solver includes only the $c_x$ exchange term and emits a `UserWarning` that
the energies are not full TDA/TDDFT values; with a pure functional and no density
the kernel is silently absent. The TDA switches to an iterative Davidson-type
solver when the pair space exceeds 2000 and at most 10 percent of the roots are
requested (`RuntimeWarning` if a root residual exceeds $10^{-4}$); otherwise it
diagonalises densely. The full Casida equation is always dense.

The command line offers `vibeqc tddft <xyz> [--basis def2-svp] [--functional ""]
[--n-states 5] [--casida] [--charge 0] [--multiplicity 1]`. With no functional
it runs HF/CIS; with `--multiplicity` greater than 1 it uses UHF or UKS and the
unrestricted solvers. Quote a basis name that contains `*` in the shell.

## Cost

For a DFT reference the time is spent in the exchange-correlation kernel, not in the
response matrices. The kernel is evaluated one occupied-virtual pair at a time (one
grid contraction per pair, then a loop over pairs of pairs), so it grows with roughly
the square of $n_\text{occ}n_\text{virt}$ and with the grid size. Measured for the
water PBE0 case above, one TDA solve for four states, on a machine that was also busy
with another job (so read the ratios, not the absolute seconds):

| Basis | Functions | Occupied-virtual pairs | SCF | TDA with $f_{xc}$ | TDA without $f_{xc}$ |
|---|---|---|---|---|---|
| STO-3G | 7 | 10 | 10 s | 4 s | 0.01 s |
| 6-31G\* | 18 | 65 | 27 s | 38 s | 0.01 s |

Without the kernel (the Hartree-Fock/CIS case, or `density_ao` omitted) the same solve
took a hundredth of a second, so the whole difference is the kernel. A separate profile of a
water TDA solve attributes more than 99 percent of the
time to the kernel routine and shows wall time growing from seconds in STO-3G to
minutes in def2-SVP. Expect a DFT excitation calculation to cost several times the SCF
and to grow much faster than it as the basis grows; a HF/CIS calculation is close to
free once the reference exists. Time a small basis first, keep `tddft_n_states`
modest (the kernel cost does not depend on it, but the memory for the dense response
matrices does), and do not extrapolate a STO-3G timing to a production basis.

## The library route

Use the solvers directly when you want the result objects, NTOs or triplets.
Same water molecule, PBE0/6-31G\*, plus HF/CIS singlets and triplets:

```python
import numpy as np
import vibeqc as vq
from vibeqc.runner import apply_ks_grid_default
from vibeqc.tddft import make_eri_provider
from vibeqc.excited import cis_excitations

ANG = 1.8897259886
mol = vq.Molecule([
    vq.Atom(8, [0.0, 0.0, 0.1173 * ANG]),
    vq.Atom(1, [0.0, 0.7572 * ANG, -0.4692 * ANG]),
    vq.Atom(1, [0.0, -0.7572 * ANG, -0.4692 * ANG]),
], 0, 1)
basis = vq.BasisSet(mol, "6-31g*")
n_occ = mol.n_electrons() // 2

opts = vq.RKSOptions()
opts.functional = "pbe0"
apply_ks_grid_default(opts)          # the grid run_job would use
gs = vq.run_rks(mol, basis, opts)

td = vq.run_tddft_tda(
    mol, basis, np.asarray(gs.mo_energies), np.asarray(gs.mo_coeffs), n_occ,
    n_states=4, functional="pbe0", density_ao=np.asarray(gs.density),
    grid_options=opts.grid,
)
for s in td.states:
    print(s.index, f"{s.excitation_energy_ev:.4f}", s.dominant_amplitudes[0][:2])

nto = vq.compute_nto(td.states[0], np.asarray(gs.mo_coeffs), n_occ)
print("NTO weights", [f"{w:.4f}" for w in nto.weights[:3]])

# CIS on a Hartree-Fock reference, singlet and triplet
hf = vq.run_rhf(mol, basis)
prov = make_eri_provider(np.asarray(vq.compute_eri(basis)),
                         np.asarray(hf.mo_coeffs), n_occ)
args = (np.asarray(hf.mo_energies), n_occ, prov.ovov(), prov.oovv())
print("singlet", np.round(cis_excitations(*args, spin="singlet", n_states=3).excitation_energies_ev, 4))
print("triplet", np.round(cis_excitations(*args, spin="triplet", n_states=3).excitation_energies_ev, 4))
```

```text
1 8.4355 (5, 1)
2 10.4117 (5, 2)
3 10.9988 (4, 1)
4 13.1764 (4, 2)
NTO weights ['0.9997', '0.0001', '0.0001']
singlet [ 9.6338 11.457  12.4118]
triplet [ 8.6546 10.7573 10.8161]
```

The library numbers reproduce the `run_job` block above. HOMO-LUMO is a single
NTO pair: its weight λ = σ² is 0.9997 (`NTOResult.weights`; `hole_weights`
holds the singular values σ, here 0.9998). The HF/CIS singlet energies agree
with an independent Hartree-Fock TDA reference calculation to better than
$10^{-3}$ eV. Each
`TDDFTState` carries `excitation_energy`, `excitation_energy_ev`,
`wavelength_nm`, `oscillator_strength`, `transition_dipole` (atomic units),
`dominant_amplitudes` and the full `excitation_vector`.

## Output

- **`.out`**: a `## TD-DFT excited states (<TDA|Casida|TDA-ROHF> (<functional>))`
  table with state, energy in eV, wavelength in nm, oscillator strength and the
  three leading transitions. With `tddft_spectrum=True`, a `## UV/Vis spectrum`
  block. If the solver raises, the block reads `FAILED: <ExceptionType>: <message>`
  and the job continues; the SCF result is unaffected.
- **QVF** (with `output_qvf=True`): an `excited_states` section (QVF spec
  § 5.12) with each state's index, excitation energy in eV, oscillator strength
  and transition dipole (multiplicity 1 on a closed-shell reference), and a
  `spectra.uvvis` section with energies, intensities, wavelengths and the method
  label that names it as `states_ref`. With `nto=True`, each state also gets a
  hole and a particle `wavefunction.gto` section, `wf_nto_S<n>_hole` and
  `wf_nto_S<n>_electron`, with `orbital_kind: "nto"`, `nto_role`, and
  `state_ref` / `state_index` pointing at the state, whose `nto_ref` names the
  hole section (§ 5.3). Their `occupations` are the NTO weights
  λₖ = σₖ², which sum to 1, and their `energies` are zeros. On an unrestricted
  reference the pair carries `spin: "unrestricted"` with α and β sets, and the
  weights of both spins together sum to 1. vibe-view renders the pairs as
  orbitals; `vibeqc nto <file>.qvf` prints the weights. The periodic route does
  not write TDDFT data into the QVF.
- **Citations** (`.bibtex`, `.references`): Runge-Gross 1984 and Casida 1995 for
  any TDDFT run, Hirata-Head-Gordon 1999 for the TDA, Bauernschmitt-Ahlrichs 1996
  when a hybrid exchange fraction enters the kernel, and Martin 2003 when `nto`
  is on, next to the functional and basis entries.
- **Files for NTOs**: the Molden files above when `tddft_molden=True`.

## Excited-state gradients

The gradient of an excited state is the derivative of $E_\text{SCF}+\omega_I$.
`vibeqc.excited_gradient.cis_state_gradients_fd` differentiates that sum by
central differences (default step $10^{-3}$ Å) and follows the root across the
displaced geometries by maximum overlap of the CIS amplitude vector, so a root
flip does not corrupt the slope. It costs about $6N$ additional SCF plus CIS
evaluations. The only energy function shipped is `make_hf_cis_energy_fn`, the
closed-shell RHF reference with singlet or triplet CIS and no ECP. Its
coordinates are in **Ångström**. Verified on H$_2$ at 0.74 Å, RHF/6-31G\*:

```python
import numpy as np
from vibeqc.excited_gradient import make_hf_cis_energy_fn, cis_state_gradients_fd

fn = make_hf_cis_energy_fn([1, 1], "6-31g*", n_states=2)
coords_angstrom = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
g = cis_state_gradients_fd(fn, coords_angstrom, states=[0, 1])   # 0 = ground, 1 = S1
print(np.round(g[0][:, 2], 6), np.round(g[1][:, 2], 6))
```

```text
[-0.007561  0.007561] [ 0.198837 -0.198837]
```

The values are Ha/bohr. `states` uses 0 for the ground state and $k\ge1$ for the
$k$-th CIS root. For your own surface (a DFT kernel, say), build a callable that
maps Ångström coordinates to a `CISStateSet` and pass it as `energy_fn`.

`run_job(tddft=True, tddft_gradient=True)` wraps this for the S1 state and prints
a block headed `Excited-state gradient (S1, RHF/CIS(TDA) surface, FD, step=1e-3 Å)`;
see the defect note below before using it. `vibeqc tddft-opt <xyz>` minimises the
S1 energy with L-BFGS-B on the same HF/CIS energy function. Its `--functional`
option is printed but not used: the surface is always HF/CIS, whatever the option
says. Treat it as an exploration convenience.

## Known defects in this build

```{warning}
Two defects in vibe-qc 0.17.10 development builds change printed numbers. Both were
found by comparing against an independent program and by re-running the library
calls; the excitation energies are not affected.

1. **Oscillator strengths are too large.** The transition dipole is contracted as
   $2\sum_{ia}X_{ia}\langle i|\mathbf r|a\rangle$, whereas the spin-adapted
   closed-shell singlet is $\sqrt2\sum_{ia}X_{ia}\langle i|\mathbf r|a\rangle$.
   Restricted (RHF/RKS) oscillator strengths are therefore a factor of 2 too
   large, and unrestricted (UHF/UKS) ones a factor of 4, relative to Hartree-Fock
   TDA in PySCF on the same geometry (water RHF/6-31G\*: 0.0375 versus 0.0187 for
   the first state; OH radical UHF/6-31G\*: 0.0122 versus 0.0031). Excitation
   energies agree to $10^{-4}$ eV. Ratios of intensities within one run, and which
   states are bright or dark, are unaffected. Do not quote absolute `f` values or
   molar absorptivities until this is fixed.
2. **`run_job(tddft_gradient=True)` differentiates the wrong geometry.** The runner
   passes bohr coordinates to a function that expects Ångström, so the gradient
   is evaluated at a geometry stretched by a factor of 1.8897. For H$_2$ at
   0.74 Å it prints 0.0131 Ha/bohr where the library call above, with the
   correct Ångström input, gives 0.1988 Ha/bohr. Use
   `cis_state_gradients_fd` directly with Ångström coordinates until the runner
   is fixed.
```

## Limits and gates

- **Range-separated hybrids** (for example wB97X, CAM-B3LYP): not implemented.
  The response would need error-function-attenuated $(ij|ab)$ integrals, not a
  single global scale on the regular ones. The library raises
  `NotImplementedError`; under `run_job` the SCF completes and the `.out` block
  reads `FAILED: NotImplementedError: TDDFT/TDA with the range-separated hybrid
  'wb97x' is not implemented ...` while the job exits normally. Use a global
  hybrid (PBE0, B3LYP) or a pure functional.
- **Double hybrids**: the perturbative doubles correction is not applied; the
  result is the bare response of the hybrid reference and a `UserWarning` says so.
- **Kernels**: LDA, GGA and global-hybrid exchange fractions only. The kernel
  builder documents that meta-GGA kernels are unavailable.
- **ROHF**: TDA only, over doubly-occupied to virtual excitations, labelled
  `TDA-ROHF` in the `.out` with a note that excitations out of or into the
  singly-occupied orbitals are not included. `tddft_type="casida"` on ROHF and any
  `tddft=True` on ROKS raise `NotImplementedError` before the SCF starts (issue
  #361). Verified for ROKS.
- **Spin**: `run_tddft_tda` and `run_tddft_casida` return singlets. Triplets are
  available through `cis_excitations(spin="triplet")` on an HF reference.
- **Full-grid external XC functionals** (for example SKALA, which is experimental):
  `NotImplementedError`, because the provider exposes no pointwise second
  functional derivative. **OpenTrustRegion** with `tddft=True`: `ValueError`.
- **Skipped runs**: the block is skipped after an unconverged SCF and for any
  result that carries no MO-energy attributes.
- **Excited-state gradient**: `tddft_gradient=True` requires an RHF reference,
  `tddft_type="tda"` and multiplicity 1; an RKS, Casida or open-shell request
  raises `NotImplementedError` before the calculation (issue #570), as does a
  molecular ECP basis. A CIS energy function for an odd electron count raises
  `NotImplementedError`.
- **Periodic**: `run_periodic_job(tddft=True)` calls `run_tddft_tda_periodic` on the
  Gamma-point orbitals and writes the table under
  `## TD-DFT excited states (TDA, Gamma-point)`. Three limits. First, the wrapper builds
  the response from open-space integrals of the unit-cell basis and passes no
  density, so there is no periodic image sum and no $f_{xc}$ kernel; it is meaningful
  for vacuum-padded molecular cells and large supercells only, and a pure functional
  gives a bare response without exchange or correlation kernel. Second, it is refused
  with `NotImplementedError` on BIPOLE, on slab GDF, on periodic ROHF/ROKS and for
  full-grid external XC, and the basis-free semiempirical periodic routes reject the
  keyword. Third, and verified: on the GDF Gamma route the result object
  (`PeriodicKRHFGDFResult`) stores `mo_coeffs` with a leading k-point axis (shape
  `(1, 2, 2)` for H$_2$ in a 12 bohr cell at STO-3G), and the wrapper reads that first
  axis as the basis size. The `run_periodic_job` call then writes
  `FAILED: ValueError: No virtual orbitals available for TDDFT (n_basis == n_occ)`
  into the `.out`, and so does a direct `run_tddft_tda_periodic(result, basis, n_occ=1)`.
  Passing an object that carries the $k=0$ block (`mo_energies[0]`, `mo_coeffs[0]`)
  works around it. Treat periodic TDDFT as unverified for GDF results until this is
  fixed.
- **Cost and memory**: see [Cost](#cost). The solvers hold the dense AO two-electron tensor and its MO
  blocks, $O(n_\text{bf}^4)$ and $O(n_\text{occ}^2n_\text{virt}^2)$. The
  pre-flight memory estimator includes the TDDFT terms.

## See also

- [Excited states with TDDFT](../tutorial/excited_states_tddft.md): worked
  formaldehyde example, theory derivation and the open-shell and periodic
  variants.
- [Ionisation potentials from the electron propagator](../tutorial/ovgf_ionization.md)
  and [Green's-function ionisation energies](ovgf.md): the ionisation counterpart.
- [Functionals](functionals.md): which exchange fraction each hybrid carries.
- [Open-shell references](rohf.md) and [SCF convergence](scf_convergence.md):
  the reference the response is built on.
- [Citations](citations.md), [Output files](output_files.md) and
  [QVF and vibe-view](../visualization.md).
- [Methods: theory and status](../methods.md) and the
  [feature matrix](../features.md).
