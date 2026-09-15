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

We start with UHF:

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
opts.max_iter = 200            # OH/6-31G* UHF needs ~120 iterations to settle
result = run_uhf(mol, basis, opts)

print(f"E_UHF  = {result.energy:.10f}")
print(f"<S^2>  = {result.s_squared:.6f}  (doublet, ideal 0.75)")
```

```text
E_UHF  = -75.3809427974
<S^2>  = 0.755290          # mildly spin-contaminated (ideal 0.75)
```

## UKS, open-shell DFT

Open-shell DFT is the unrestricted-Kohn-Sham counterpart of UHF: the same
independent α and β orbitals, with an exchange-correlation functional in
place of exact exchange. `run_uks` takes a `UKSOptions` carrying the
functional name:

```python
from vibeqc import UKSOptions, run_uks

opts = UKSOptions()
opts.functional = "PBE"
result = run_uks(mol, basis, opts)
```

Same functional library as RKS.

## UMP2, second-order correlation on a UHF reference

With a converged UHF reference in hand, `run_ump2` adds second-order
Møller-Plesset correlation on top, split into the same-spin (αα, ββ) and
opposite-spin (αβ) channels:

```python
from vibeqc import run_ump2

ump2 = run_ump2(mol, basis, result_uhf)   # pass a converged UHF result
print(f"E_UMP2 total  = {ump2.e_total:.10f}")
print(f"E_correlation = {ump2.e_correlation:.10f}")
print(f"  αα same-spin = {ump2.e_aa:.10f}")
print(f"  ββ same-spin = {ump2.e_bb:.10f}")
print(f"  αβ opposite  = {ump2.e_ab:.10f}")
```

Closed-shell MP2 on an RHF reference is `run_mp2`.

## ROHF: a spin-pure open shell

UHF buys a lower energy by giving the α and β electrons their own spatial
orbitals, and pays for it with spin contamination. ROHF makes the opposite
trade: it keeps the open shell spin-restricted (one shared spatial orbital
set with closed, open, and virtual blocks), so the reference is a **pure
spin eigenstate** by construction. Ordinary radicals have a single SOMO.
`run_rohf` takes the same molecule, basis, and options shape as `run_uhf`:

```python
from vibeqc import ROHFOptions, run_rohf

opts = ROHFOptions(conv_tol_energy=1e-10, conv_tol_grad=1e-8)
rohf = run_rohf(mol, basis, opts)

print(f"E_ROHF = {rohf.energy:.10f}")
print(f"<S^2>  = {rohf.s_squared:.6f}  (exact 0.75, by construction)")
```

```text
E_ROHF = -75.3770312007
<S^2>  = 0.750000          # exactly 0.75, no contamination to correct
```

On this OH radical ROHF sits about 3.9 mHa *above* UHF: UHF's extra
variational freedom always wins on raw energy. What ROHF gives back is exact
spin purity, $\langle S^2\rangle = 0.75$ to every digit, with no
contamination to diagnose. Reach for it when a clean spin state matters more
than the last few mHa, or when you need a spin-pure reference for a
correlated treatment.

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

For $N_\alpha$ unpaired-spin-up and $N_\beta$ spin-down electrons
with $N_\alpha \neq N_\beta$, we need at least two paths forward:

- **UHF (unrestricted HF).** Give α and β electrons *different*
  spatial orbitals, $\phi^\alpha_i$ and $\phi^\beta_i$. Double the
  SCF unknowns, but now every electron has full variational freedom
  in its spatial part.
- **ROHF (restricted open-shell HF).** Keep the doubly-occupied
  orbitals spin-restricted (same $\phi_i$ for α and β when both are
  occupied), and let only the singly-occupied orbitals be separate.
  The determinant stays a pure spin eigenstate, at a small energy cost
  versus UHF. vibe-qc ships this as `run_rohf` (above).

### UHF equations

UHF solves two coupled generalised eigenvalue problems:

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
β. This is the physics of Fermi correlation that makes UHF give a
lower energy than RHF for any open-shell reference.

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
value `result.s_squared_ideal`. Heuristic thresholds:

| $\langle S^2 \rangle - S(S+1)$ | Interpretation |
| --- | --- |
| $< 0.02$ | Clean open shell, trust UHF |
| $0.02 - 0.10$ | Mild contamination, numbers still useful |
| $> 0.10$ | Severe contamination, consider ROHF, CASSCF, or MP2/CCSD on top |

Large spin contamination often signals the real problem, the
reference determinant isn't a good starting point for a
single-determinant method, and you need a multireference approach.
Active-space FCI / selected-CI / DMRG and orbital-optimised CASSCF /
CASPT2 / NEVPT2 all ship via `vibeqc.solvers`, with CASSCF also exposed
as `run_job(method="casscf", active_space=(...))`.

### UMP2

Second-order Møller-Plesset perturbation theory on a UHF reference
partitions correlation energy into **same-spin** and **opposite-spin**
channels:

$$
E^{(2)} = E_{\alpha\alpha}^{(2)} + E_{\beta\beta}^{(2)} + E_{\alpha\beta}^{(2)},
$$

with

$$
E_{\sigma\tau}^{(2)} = \sum_{ij \in \text{occ}} \sum_{ab \in \text{virt}}
  \frac{\langle ij \| ab \rangle^2}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
$$

summed over pairs where orbital $i$ has spin $\sigma$ and $j$ has
$\tau$. The `ump2.e_aa`, `.e_bb`, and `.e_ab` fields surface this
decomposition directly. The same-spin/opposite-spin split is the
starting point for **spin-component-scaled MP2** (SCS-MP2,
Grimme 2003), where the two channels get different empirical
prefactors, roadmap item for the future.

## Resources

~70 MB peak RAM, <0.1 s on one core (Apple M2 baseline) for the
OH-radical / 6-31G\* example. UHF roughly doubles the cost of the
matching closed-shell HF, separate α / β Fock builds and densities
each iteration. Triplet O₂ at the same basis: ~150 MB, ~0.5 s.

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
