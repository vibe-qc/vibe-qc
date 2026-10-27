# Open-shell systems (Unrestricted, Restricted-Open-Shell)

A system with unpaired electrons, a radical, an odd-electron ion, a triplet
state, a partially filled d shell, breaks the closed-shell RHF picture of one
doubly-occupied orbital per electron pair. There are two standard open-shell
formalisms, and both run in vibe-qc: **unrestricted** (the U methods), which
give the α and β electrons their own independent spatial orbitals, and
**restricted open-shell** (the RO methods), which keep the doubly-occupied
orbitals spin-restricted. The choice carries across the whole method stack
(UHF/ROHF, UKS/ROKS, UMP2/ROMP2, and onward to coupled cluster). This page
walks the foundational ones on the doublet OH radical.

## Run the UHF reference

Use the installed engine environment and a run directory outside the checkout.
The first block is standalone; the UKS, UMP2 and ROHF blocks continue in the
same script. Coordinates passed to `Atom` are in bohr and energies are in
hartree. This fixed OH geometry is illustrative, not an optimized structure.

Neutral OH has nine electrons. `multiplicity=2` gives five alpha and four
beta electrons; `charge=-1, multiplicity=1` would describe hydroxide instead.
See [Molecular charge and spin](../user_guide/molecules.md#configuring-open-shell-systems)
for the electron-count checks.

```python
from vibeqc import Atom, Molecule, BasisSet, UHFOptions, run_uhf

mol = Molecule(
    [Atom(8, [0, 0, 0]), Atom(1, [0, 0, 1.832])],
    charge=0,
    multiplicity=2,
)
basis = BasisSet(mol, "6-31g*")

opts = UHFOptions()
opts.conv_tol_energy = 1e-10
opts.conv_tol_grad = 1e-8
opts.max_iter = 200            # iteration budget, not a convergence guarantee
result_uhf = run_uhf(mol, basis, opts)
if not result_uhf.converged:
    raise RuntimeError("OH UHF did not converge; inspect result_uhf.scf_trace")

print(f"E_UHF  = {result_uhf.energy:.10f}")
print(f"<S^2>  = {result_uhf.s_squared:.6f}")
print(f"Ideal  = {result_uhf.s_squared_ideal:.6f}")
```

Keep this result as `result_uhf`: UMP2 below needs its UHF orbitals and
orbital energies. A converged flag establishes SCF convergence, not the
correct electronic state. Inspect the spin diagnostic and occupations as
well as the energy; see [SCF convergence](../user_guide/scf_convergence.md).

## UKS, open-shell DFT

Open-shell DFT is the unrestricted-Kohn-Sham counterpart of UHF: the same
independent α and β orbitals, with an exchange-correlation functional in
place of exact exchange. `run_uks` takes a `UKSOptions` carrying the
functional name:

```python
# doc-audit: skip - continues with mol and basis from the UHF block
from vibeqc import UKSOptions, run_uks

opts = UKSOptions()
opts.functional = "PBE"
result_uks = run_uks(mol, basis, opts)
if not result_uks.converged:
    raise RuntimeError("OH PBE UKS did not converge; inspect result_uks.scf_trace")
print(f"E_UKS = {result_uks.energy:.10f}")
```

UKS uses the same functional interface as RKS. Here the supplied
`UKSOptions` keeps its default legacy product grid. The grid selection rules
for supplied options differ from `run_job`; see
[Molecular integration grids](../user_guide/functionals.md#molecular-integration-grids).
The separate result name prevents the UKS reference from replacing the UHF
reference needed for the next step.

## UMP2, second-order correlation on a UHF reference

With a converged UHF reference in hand, `run_ump2` adds second-order
Møller-Plesset correlation on top, split into the same-spin (αα, ββ) and
opposite-spin (αβ) channels:

```python
# doc-audit: skip - continues with mol, basis and result_uhf from the UHF block
from vibeqc import UMP2Options, run_ump2

mp2_opts = UMP2Options()
mp2_opts.n_frozen_core = 0  # explicitly correlate all electrons for this example
ump2 = run_ump2(mol, basis, result_uhf, mp2_opts)
print(f"E_UMP2 total  = {ump2.e_total:.10f}")
print(f"E_correlation = {ump2.e_correlation:.10f}")
print(f"  αα same-spin = {ump2.e_aa:.10f}")
print(f"  ββ same-spin = {ump2.e_bb:.10f}")
print(f"  αβ opposite  = {ump2.e_ab:.10f}")
```

Use the same geometry and basis that produced `result_uhf`. The driver
rejects an unconverged UHF reference. It does not accept the UKS or ROHF
result as a replacement for that reference.

`UMP2Options.n_frozen_core` defaults to `-1`, selecting the published
chemical-core policy; `0` explicitly selects all-electron correlation.
Record this choice when comparing energies. At the default unscaled spin
coefficients, `e_correlation = e_aa + e_bb + e_ab` and
`e_total = e_hf + e_correlation`. The reported UHF spin diagnostic remains a
property of the reference, not a newly computed UMP2 expectation value.
See [MP2 and double hybrids](../user_guide/mp2_and_double_hybrids.md)
for frozen-core, density-fitting and memory controls.

Closed-shell MP2 on an RHF reference is `run_mp2`.

### Write a complete MP2 job

As an alternative to the low-level UHF/UMP2 calls above, let `run_job`
compute the reference and correction together and record both in the output:

```python
# doc-audit: skip - reuses mol from the UHF block; alternative workflow
from vibeqc import run_job

job = run_job(
    mol,
    basis="6-31g*",
    method="mp2",             # multiplicity=2 selects UMP2
    mp2_reference="uhf",
    frozen_core=0,             # match the all-electron example above
    output="oh_ump2",
    write_molden_file=True,
)
if not job.converged:
    raise RuntimeError("OH MP2 reference did not converge; inspect oh_ump2.out")
print(f"E_UHF        = {job.energy:.10f}")
print(f"E_correlation = {job.mp2.e_correlation:.10f}")
print(f"E_UMP2 total = {job.energy_total:.10f}")
```

`job.energy` remains the UHF energy; `job.energy_total` includes the MP2
correction and `job.mp2` exposes the spin channels. This call runs a new job,
so choose it instead of the preceding low-level calls when you only need
one calculation with output files. The Molden file contains reference
orbitals, not correlated natural orbitals. The complete script is
[`examples/molecular/input-oh-ump2.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-oh-ump2.py).

## ROHF: a spin-pure open shell

UHF buys a lower energy by giving the α and β electrons their own spatial
orbitals, and pays for it with spin contamination. ROHF makes the opposite
trade: it keeps the open shell spin-restricted (one shared spatial orbital
set with closed, open, and virtual blocks), so the reference is a **pure
spin eigenstate** by construction. Ordinary radicals have a single SOMO.
`run_rohf` takes the same molecule, basis, and options shape as `run_uhf`:

```python
# doc-audit: skip - continues with mol and basis from the UHF block
from vibeqc import ROHFOptions, run_rohf

opts = ROHFOptions(conv_tol_energy=1e-10, conv_tol_grad=1e-8)
rohf = run_rohf(mol, basis, opts)
if not rohf.converged:
    raise RuntimeError("OH ROHF did not converge; inspect rohf.scf_trace")

print(f"E_ROHF = {rohf.energy:.10f}")
print(f"<S^2>  = {rohf.s_squared:.6f}  (exact 0.75, by construction)")
```

Compare ROHF and UHF only at the same geometry, basis and numerical
settings. The UHF variational space contains the ROHF space, but an SCF
iteration can stop at different stationary solutions; a single calculation
does not guarantee the lowest energy in either space. ROHF supplies the
spin-restricted reference used by its own correlated routes; see
[ROHF and ROKS](../user_guide/rohf.md) for their supported combinations.

Open-shell **DFT** has the matching spin-restricted form, `run_roks` with
`ROKSOptions(functional=...)`, the spin-pure counterpart to `run_uks`.
For degenerate terms, ROKS averages the frontier shell; OH(2Pi), for
example, uses occupations `2, 2, 2, 1.5, 1.5, 0, ...` across the pi pair.
Analytic ROHF nuclear gradients come from `compute_rohf_gradient`, and the
periodic side ships ROHF at $\Gamma$ and multi-k
(`run_rohf_periodic_gamma_ewald3d`, `run_rohf_periodic_multi_k_ewald3d`).

## Theory

The sections below are the theory behind the recipes above: why a
closed-shell method cannot describe an open shell, the coupled UHF
equations, the spin contamination the unrestricted formalism introduces,
and the spin-resolved UMP2 correction.

### Why open-shell needs its own method

Restricted Hartree-Fock (RHF) assumes every spatial orbital carries
exactly **two** electrons: one α-spin, one β-spin, occupying the
*same* orbital $\phi_i(\mathbf{r})$. This is the right ansatz for
closed-shell ground states (singlets: He, H₂, H₂O, benzene) but
breaks any system with an unpaired electron, radicals, odd-electron
molecules, triplet states, transition-metal complexes with partially
filled d shells.

For $N_\alpha$ alpha-spin and $N_\beta$ beta-spin electrons
with $N_\alpha \neq N_\beta$, we need at least two paths forward:

- **UHF (unrestricted HF).** Give α and β electrons *different*
  spatial orbitals, $\phi^\alpha_i$ and $\phi^\beta_i$ (J. A. Pople
  and R. K. Nesbet, *J. Chem. Phys.* **22**, 571 (1954),
  doi:10.1063/1.1740120). Double the
  SCF unknowns, but now every electron has full variational freedom
  in its spatial part.
- **ROHF (restricted open-shell HF).** Keep the doubly-occupied
  orbitals spin-restricted (same $\phi_i$ for α and β when both are
  occupied), and let only the singly-occupied orbitals be separate
  ({vibeqc-cite}`roothaan_rohf_1960`).
  The determinant stays a pure spin eigenstate, at a small energy cost
  versus UHF. vibe-qc ships this as `run_rohf` (above).

### UHF equations

UHF solves two coupled generalised eigenvalue problems, the
Pople-Nesbet equations:

$$
\mathbf{F}^\alpha \, \mathbf{C}^\alpha = \mathbf{S} \, \mathbf{C}^\alpha \, \boldsymbol{\varepsilon}^\alpha,
\qquad
\mathbf{F}^\beta  \, \mathbf{C}^\beta  = \mathbf{S} \, \mathbf{C}^\beta  \, \boldsymbol{\varepsilon}^\beta,
$$

with the Fock matrices coupled through the total density:

$$
F^\sigma_{\mu\nu} = h_{\mu\nu}
  + \sum_{\lambda\tau} (P^\alpha_{\lambda\tau} + P^\beta_{\lambda\tau}) \, (\mu\nu|\lambda\tau)
  - \sum_{\lambda\tau} P^\sigma_{\lambda\tau} \, (\mu\tau|\lambda\nu),
\qquad \sigma \in \{\alpha, \beta\}.
$$

The Coulomb ($J$) term sees the *total* density; the exchange ($K$)
term is **spin-diagonal**, α electrons only exchange with α, β with
β. This is the physics of Fermi correlation that gives UHF an energy
at or below ROHF's for any open-shell state.

### Spin contamination, the price you pay

UHF's freedom has a catch: the resulting wavefunction is **not** a
pure spin eigenstate. For a nominal doublet ($S = 1/2$, target
$\langle S^2 \rangle = 0.75$), the UHF determinant can mix in
higher-multiplicity contributions, and $\langle S^2 \rangle$
measured on the UHF wavefunction comes out larger than $S(S+1)$.
The diagnostic is computed as

$$
\langle \hat{S}^2 \rangle_\text{UHF} = S_z(S_z + 1) + N_\beta
  - \sum_i^\text{α occ} \sum_j^\text{β occ} \bigl|\langle \phi^\alpha_i | \phi^\beta_j \rangle\bigr|^2,
$$

and vibe-qc reports it as `result.s_squared` alongside the ideal
value `result.s_squared_ideal`. Check the deviation from the target alongside
the occupations, SCF residual and sensitivity to the starting guess.
A small deviation alone does not validate the energy or the intended state.
Adding a UMP2 correction is not a spin-purification step.

Large spin contamination often signals the real problem, the
reference determinant isn't a good starting point for a
single-determinant method, and you need a multireference approach.
Active-space FCI / selected-CI / DMRG and orbital-optimised CASSCF /
CASPT2 / NEVPT2 ({vibeqc-cite}`white_dmrg_1992`,
{vibeqc-cite}`roos_taylor_siegbahn_casci_1980`,
{vibeqc-cite}`andersson_caspt2_1990`, {vibeqc-cite}`angeli_nevpt2_2001`)
all ship via `vibeqc.solvers`, with CASSCF also exposed
as `run_job(method="casscf", active_space=(...))`.

### UMP2

Second-order Møller-Plesset perturbation theory
({vibeqc-cite}`moller_plesset_1934`) on a UHF reference partitions correlation energy into **same-spin** and **opposite-spin**
channels:

$$
E^{(2)} = E_{\alpha\alpha}^{(2)} + E_{\beta\beta}^{(2)} + E_{\alpha\beta}^{(2)},
$$

with

$$
E_{\sigma\tau}^{(2)} = c_{\sigma\tau} \sum_{ij \in \text{occ}} \sum_{ab \in \text{virt}}
  \frac{\langle ij \| ab \rangle^2}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b},
\qquad c_{\sigma\sigma} = \tfrac14, \quad c_{\alpha\beta} = 1,
$$

summed over orbitals where $i$ and $a$ have spin $\sigma$ and $j$ and
$b$ have spin $\tau$. The factor ¼ removes the double counting of
$i \leftrightarrow j$ and $a \leftrightarrow b$ in the same-spin sums;
in the opposite-spin channel the exchange integral vanishes, so
$\langle ij \| ab \rangle = \langle ij | ab \rangle$. The `ump2.e_aa`, `.e_bb`, and `.e_ab` fields surface this
decomposition directly. The same-spin/opposite-spin split is the
starting point for **spin-component-scaled MP2** (SCS-MP2,
{vibeqc-cite}`grimme_scs_mp2_2003`), where the two channels get different empirical
prefactors; it ships as `vibeqc.run_scs_mp2` (RHF reference) and
`vibeqc.run_scs_ump2` (UHF reference).

## Resources

UHF builds separate alpha and beta densities and Fock matrices. Cost depends
on the basis, integral mode, starting guess and convergence history. UMP2
adds an integral transformation and correlation workspace beyond the SCF
cost; use the [memory guide](../user_guide/memory.md) before scaling this
small-molecule example. No current-build timing is implied by this recipe.

## References

- **UHF, original.** J. A. Pople and R. K. Nesbet, "Self-consistent
  orbitals for radicals," *J. Chem. Phys.* **22**, 571 (1954).
- **ROHF formulation.** C. C. J. Roothaan, "Self-consistent field
  theory for open shells of electronic systems," *Rev. Mod. Phys.*
  **32**, 179 (1960).
- **Spin contamination.** J. A. Pople, P. M. W. Gill, and N. C.
  Handy, "Spin-unrestricted character of Kohn-Sham orbitals for
  open-shell systems," *Int. J. Quantum Chem.* **56**, 303 (1995).
- **Textbook.** A. Szabo and N. S. Ostlund, *Modern Quantum
  Chemistry*, Dover (1996), chapters 3.8 (UHF) and 6.5 (UMP2).
- **SCS-MP2.** S. Grimme, "Improved second-order Møller-Plesset
  perturbation theory by separate scaling of parallel- and
  antiparallel-spin pair correlation energies," *J. Chem. Phys.*
  **118**, 9095 (2003).
- **Pauncz on spin eigenfunctions.** R. Pauncz, *Spin
  Eigenfunctions*, Plenum (1979). The comprehensive treatment of
  spin states, projection operators, and multireference
  constructions if you want the full theory.
