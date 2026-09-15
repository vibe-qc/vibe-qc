# Molecular DFT (RKS)

Run closed-shell Kohn-Sham density-functional theory. Same shape as
the [HF tutorial](molecular_hf.md), Fock-matrix build, SCF loop,
DIIS acceleration, with the HF exchange operator replaced by an
exchange-correlation functional $v_\text{xc}[\rho]$. The theory
section below explains the Kohn-Sham mapping, where the grid comes
from, and what DFT can and cannot tell you.

```python
from vibeqc import Atom, Molecule, BasisSet, RKSOptions, run_rks

mol = Molecule([
    Atom(8, [0.0,  0.0,  0.0]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])
basis = BasisSet(mol, "6-31g*")

opts = RKSOptions()
opts.functional = "PBE"     # or "LDA", "BLYP", "B3LYP", "PBE0", etc.
# Supplied low-level options keep their grid, here the legacy product grid.
result = run_rks(mol, basis, opts)

print(f"E_KS        = {result.energy:.10f}")
print(f"E_Coulomb   = {result.e_coulomb:.10f}")
print(f"E_xc        = {result.e_xc:.10f}")
print(f"E_HF_exch   = {result.e_hf_exchange:.10f}")
```

Any functional name libxc understands is accepted, there are over
500. Common choices:

| Name | Type | Use case |
|---|---|---|
| `LDA` / `SVWN` | LDA | Fast sanity check; too exchange-heavy for real chemistry |
| `PBE` | GGA | Default for solid-state; good cost/accuracy balance |
| `BLYP` | GGA | Classic molecular GGA |
| `B3LYP` | hybrid GGA | de-facto standard for organics |
| `PBE0` | hybrid GGA | Semiconductors, molecular properties |
| `TPSS` / `TPSSh` | meta-GGA / hybrid meta-GGA | Better barriers than GGAs; TPSSh = 10 % HF |
| `M06-L` / `M06-2X` | meta-GGA / 54 % hybrid meta-GGA | Minnesota family; M06-2X for thermochemistry + non-covalent |

Hybrids automatically enable the HF-exchange mixing through libint
under the hood; no extra option needed.

## Grid quality

The exchange-correlation energy has no closed form, so vibe-qc
evaluates it numerically: around every atom it lays down a cloud of
weighted sample points and, each SCF iteration, evaluates the density
and the XC functional at each one. Three knobs set how dense that
cloud is:

```python
opts.grid.n_radial = 99   # radial shells per atom
opts.grid.n_theta  = 29   # polar (θ) points per shell
opts.grid.n_phi    = 58   # azimuthal (φ) points per shell
```

These are controls for a custom product grid. Fresh options supplied to the
low-level `run_rks(mol, basis, opts)` keep the legacy 75 x 17 x 36 grid,
about 46,000 points per atom. The custom 99 x 29 x 58 grid above has about
167,000 points per atom. More points increase the XC integration work; the
Coulomb and exact-exchange builds have separate costs.

The mid-level `run_job` and molecular optimization interfaces instead apply
`grid_level="orca-defgrid3"` to absent or untouched options. They preserve
customized grid fields. The named `fine` preset uses a pruned Lebedev grid,
so it is different from the custom product grid above. Pass
`grid_level="legacy"` explicitly to request the old grid through a mid-level
API. See [Molecular integration grids](../user_guide/functionals.md#molecular-integration-grids)
for the entry-point rules and complete preset settings.

**What it buys.** A denser grid shrinks the quadrature error, pushing
the energy toward the value the functional would give with a perfect
integral. It does **not** make the functional itself more accurate
against experiment; it only removes integration noise. Check grid convergence
for tight energy *differences*, benchmark totals and smooth forces along a reaction path, especially for grid-sensitive
meta-GGAs such as SCAN (see the
[functionals guide](../user_guide/functionals.md)). The quadrature
scheme behind these knobs (Becke partitioning, the Treutler-Ahlrichs
radial map, and the product-vs-Lebedev angular choice) is spelled out
in the **Theory** section below (*Numerical integration: the grid*).

## Validation against PySCF

The vibe-qc regression suite runs the H2O / PBE / 6-31G* total energy
through both vibe-qc and PySCF and asserts agreement to 3e-6 Ha (a
comfortable bound around the ~1e-6 Ha grid-accuracy expectation). See
`tests/test_rks.py` for the list of cases.

## Theory

The sections below build up the Kohn-Sham picture from the ground: why
DFT exists, the Hohenberg-Kohn theorems it rests on, the mapping onto a
non-interacting reference, how the equations look in the AO basis, the
numerical grid that integrates the XC energy, and where the method
helps and hurts.

### Why DFT exists at all

HF is tractable but misses electron correlation. Explicit correlated
wavefunction methods (MP2, CCSD, CCSD(T), CI) recover it at increasing
cost, CCSD(T) scales $\mathcal{O}(N^7)$. DFT trades exactness for
scaling: it's **in principle exact** (Hohenberg-Kohn 1964) but only
**in practice tractable** (Kohn-Sham 1965) with approximations to the
unknown exchange-correlation functional. Typical cost $\mathcal{O}(N^3)$
for pure DFT, $\mathcal{O}(N^4)$ for hybrids.

### The Hohenberg-Kohn theorems

Two foundational results:

1. For a non-degenerate ground state, the external potential
   $v_\text{ext}(\mathbf{r})$ is a unique functional of the density
   $\rho(\mathbf{r})$ (up to a trivial constant). Everything about
   the system, wavefunction, observables, is therefore encoded in
   $\rho$.
2. There exists a universal functional $F_\text{HK}[\rho] =
   T[\rho] + V_\text{ee}[\rho]$ such that the exact ground-state
   energy comes from minimizing

   $$
   E[\rho] = F_\text{HK}[\rho] + \int v_\text{ext}(\mathbf{r}) \, \rho(\mathbf{r}) \, \mathrm{d}^3 r
   $$

   over all $N$-representable densities.

$F_\text{HK}$ is universal (same functional for every molecule) but
its closed form is *unknown*. Every DFT approximation is a model for
$F_\text{HK}$.

### The Kohn-Sham mapping

Kohn-Sham side-stepped the unknown $T[\rho]$ by mapping the
interacting system onto a **fictitious non-interacting system** of
$N$ electrons moving in an effective potential $v_\text{eff}(\mathbf{r})$
that reproduces the same density. The density comes from
non-interacting orbitals:

$$
\rho(\mathbf{r}) = \sum_i |\phi_i(\mathbf{r})|^2,
\qquad
\Bigl( -\tfrac{1}{2} \nabla^2 + v_\text{eff}(\mathbf{r}) \Bigr) \phi_i = \varepsilon_i \, \phi_i,
$$

with

$$
v_\text{eff}(\mathbf{r}) = v_\text{ext}(\mathbf{r})
  + \int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, \mathrm{d}^3 r'
  + v_\text{xc}(\mathbf{r}),
\qquad
v_\text{xc}(\mathbf{r}) = \frac{\delta E_\text{xc}[\rho]}{\delta \rho(\mathbf{r})}.
$$

The exchange-correlation potential $v_\text{xc}$ holds every bit of
physics HF misses, and since the kinetic-energy functional of the
non-interacting system is computed exactly from the orbitals, the
error in DFT is entirely in $E_\text{xc}$.

### In the AO basis

Expand $\phi_i = \sum_\mu C_{\mu i} \chi_\mu$ as in HF. The generalised
eigenvalue problem is formally identical:

$$
\mathbf{F}^\text{KS} \, \mathbf{C} = \mathbf{S} \, \mathbf{C} \, \boldsymbol{\varepsilon},
$$

with the Kohn-Sham Fock matrix

$$
F^\text{KS}_{\mu\nu} = h_{\mu\nu}
  + \sum_{\lambda\sigma} P_{\lambda\sigma} \,(\mu\nu \,|\, \lambda\sigma)
  + V^\text{xc}_{\mu\nu}
  - \tfrac{\alpha}{2} \sum_{\lambda\sigma} P_{\lambda\sigma} \,(\mu\sigma \,|\, \lambda\nu).
$$

The first two terms are $h$ + Coulomb (same as HF). $V^\text{xc}_{\mu\nu}
= \int \chi_\mu(\mathbf{r}) \, v_\text{xc}[\rho](\mathbf{r}) \, \chi_\nu(\mathbf{r}) \, \mathrm{d}^3 r$
is the XC matrix built by numerical quadrature on the molecular grid
(see below). For **hybrid functionals** the fraction $\alpha$ of HF
exchange is mixed in, e.g., $\alpha = 0.20$ for B3LYP, $0.25$ for
PBE0, and libint evaluates the same four-index integrals the HF
driver uses.

### Numerical integration: the grid

$V^\text{xc}$ has no closed form and must be integrated numerically.
vibe-qc uses the **Becke partitioning** scheme: atom-centered fuzzy
Voronoi cells smeared by a polynomial $p_k(s)$ (Becke 1988) so that
nuclear cusps in $v_\text{xc}$ are cleanly resolved. Around each atom
the integration uses a direct product:

- **Radial grid**: Treutler-Ahlrichs M4 mapping $u \to r$ with
  Chebyshev (2nd-kind) nodes (`n_radial`, default 75).
- **Angular grid**: by default a direct product of Gauss-Legendre
  nodes in $\cos\theta$ (`n_theta`) and equally spaced $\phi$ nodes
  (`n_phi`), giving 17 × 36 = 612 points per shell at the default quality.
  A Lebedev-Laikov path is also available (`opts.grid.angular =
  AngularScheme.Lebedev` with `opts.grid.lebedev_order`); it reaches
  the same accuracy in roughly half the points (order 29 = 302
  points/shell).

Total per atom: `n_radial × n_theta × n_phi ≈ 46 000` grid points at
the default quality. Tighten `n_radial` to 99 and the angular grid to
29 × 58 for benchmark-quality integration; the step from `medium`
to `fine` can move total energies by a few tens of µHa.

### Jacob's ladder

Perdew's hierarchy of DFT approximations climbs a metaphorical ladder
from "Hartree world" (no XC at all) toward exactness, each rung
adding a new ingredient:

| Rung | Functional input | Examples |
|---|---|---|
| 1 | $\rho$ only | LDA (SVWN, PW92) |
| 2 | + $\nabla\rho$ | GGA (PBE, BLYP) |
| 3 | + $\tau = \sum_i \lvert\nabla\phi_i\rvert^2 / 2$ | meta-GGA (TPSS, SCAN, r²SCAN) |
| 4 | + exact HF exchange (constant fraction) | hybrid (B3LYP, PBE0) |
| 5 | + range-separated HF exchange | RSH (ωB97X, CAM-B3LYP) |

Higher rung ≠ automatically better, it's an accuracy/cost trade-off
that depends on the property and system class. See
[DFT functional comparison](functional_comparison.md) for a practical comparison
on water; vibe-qc currently supports rungs 1, 2, and 4.

### Where DFT helps and hurts

- **Helps.** Energies, geometries, and vibrational frequencies for
  main-group thermochemistry, organic chemistry, and most of
  transition-metal chemistry, within 1-3 kcal/mol and 0.01 Å at a
  GGA/hybrid level for ordinary molecules.
- **Hurts, self-interaction error.** An electron interacts with
  itself through the Coulomb term. HF cancels this exactly (the
  diagonal $K$ term); local DFT only partially cancels it. That's
  why pure-DFT HOMOs are too shallow and charge-transfer states
  have the wrong trend with distance. Hybrids help by mixing in
  exact exchange.
- **Hurts, no long-range dispersion.** GGAs and hybrids are blind
  to the $-C_6/R^6$ tail (see [Dispersion corrections (D3-BJ)](dispersion.md) for
  the D3(BJ) fix).
- **Hurts, strongly correlated systems.** Any system where a single
  Slater determinant is a poor starting point (biradicals, some
  transition-metal complexes, bond-breaking, Mott insulators). DFT
  gives qualitatively wrong answers here; you want CASSCF /
  multireference methods.

## Resources

~250 MB peak RAM, <0.5 s on one core (Apple M2 baseline) for the
H₂O / 6-31G\* PBE example. The DFT XC grid (default 75-radial ×
302-angular Lebedev shells per atom, ~20 k points for H₂O) dominates
the per-iteration cost over the Fock build at this size; tighten or
loosen via the grid options on `RKSOptions.grid`.

## References

- **Hohenberg-Kohn theorems.** P. Hohenberg and W. Kohn,
  "Inhomogeneous electron gas," *Phys. Rev.* **136**, B864 (1964).
- **Kohn-Sham equations.** W. Kohn and L. J. Sham,
  "Self-consistent equations including exchange and correlation
  effects," *Phys. Rev.* **140**, A1133 (1965).
- **Becke atomic partitioning / grid.** A. D. Becke, "A multicenter
  numerical integration scheme for polyatomic molecules," *J. Chem.
  Phys.* **88**, 2547 (1988).
- **Treutler-Ahlrichs radial grid.** O. Treutler and R. Ahlrichs,
  "Efficient molecular numerical integration schemes," *J. Chem.
  Phys.* **102**, 346 (1995).
- **Lebedev angular grid.** V. I. Lebedev and D. N. Laikov,
  "A quadrature formula for the sphere of the 131st algebraic order
  of accuracy," *Dokl. Math.* **59**, 477 (1999).
- **Jacob's ladder.** J. P. Perdew and K. Schmidt, "Jacob's ladder of
  density functional approximations for the exchange-correlation
  energy," *AIP Conf. Proc.* **577**, 1 (2001).
- **Textbook.** R. G. Parr and W. Yang, *Density-Functional Theory of
  Atoms and Molecules*, Oxford University Press (1989). The canonical
  DFT reference, with complete derivations.
- **Practical DFT textbook.** W. Koch and M. C. Holthausen, *A
  Chemist's Guide to Density Functional Theory*, 2nd ed., Wiley-VCH
  (2001). Less formal than Parr-Yang, more oriented toward "which
  functional should I use for my chemistry."
