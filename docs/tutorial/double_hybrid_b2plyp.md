# Double hybrid: B2PLYP step by step

The [DFT functional comparison tutorial](functional_comparison.md)
walks Jacob's ladder up to rung 4 (hybrid DFT, B3LYP, PBE0). The next
rung is **double-hybrid DFT**: a hybrid-DFT SCF step plus a scaled
MP2 correlation correction on the converged KS orbitals. Double hybrids
are a **production** route in vibe-qc (closed-shell energies validated to
1e-7 Ha against PySCF; the label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels)),
and the [Theory](#theory) section below explains what the energy expression
contains and why it is a `method=` rather than a `functional=`.

| Rung | Input | Cost | Examples |
|---|---|---|---|
| 4. hybrid | + fraction of HF exchange | +~2× | B3LYP, PBE0 |
| **5. range-separated** | + long-range HF exchange | +~3× | ωB97X, ωB97M-V (see [the ωB97X tutorial](range_separated_wb97x.md)) |
| **6. double hybrid** | + post-SCF scaled MP2 correlation | +~10× | **B2PLYP**, DSD-PBEP86 |

This tutorial runs **B2PLYP** (Grimme, *J. Chem. Phys.* **124**,
034108 (2006)) on water and compares against the rungs below it. The
core recipe is

```
E_B2PLYP = E_RKS[0.53·HF + 0.47·B88, 0.73·LYP] + 0.27 · E_MP2_corr
```

- two SCF runs' worth of work (hybrid DFT + RI-MP2 on top), but a
documented improvement in reaction barriers, conformer ordering, and
weak-interaction energies vs. plain hybrids.

## A single-line dispatch

`run_b2plyp` handles both steps of the recipe for you. This runs B2PLYP
on water in a cc-pVDZ basis and reads back the total energy, the hybrid
SCF piece, and the scaled MP2 correction separately:

```python
from vibeqc import Atom, Molecule, BasisSet, run_b2plyp

mol = Molecule([
    Atom(8, [0.0,  0.00,  0.00]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])
basis = BasisSet(mol, "cc-pvdz")

result = run_b2plyp(mol, basis)

print(result.e_total)               # the published B2PLYP energy
print(result.rks.energy)            # the hybrid SCF step (= total − MP2 corr)
print(result.mp2.e_correlation)     # 0.27 × (E_os + E_ss) on the KS orbitals
print(result.functional)            # "b2plyp"
```

A run on a vibe-qc 0.17.10 development build (Apple M-series laptop, about
30 s) gave `e_total = -76.348110` Ha, of which the hybrid SCF step is
`-76.284154` Ha and the scaled MP2 correction `-0.063956` Ha. The same
calculation is reachable through `run_job`, which returns the same
`DoubleHybridResult`:

```python
import vibeqc as vq

result = vq.run_job(mol, basis="cc-pvdz", method="b2plyp", output="h2o_b2plyp")
print(result.e_total)               # -76.348110
```

`run_b2plyp` defaults to **density fitting on both steps**: RI-J +
RI-K for the hybrid SCF, RI-MP2 for the correction. Auxiliary bases
are auto-resolved from the orbital basis name (JKfit for SCF, RIfit
for MP2), which needs a correlation-consistent or def2 orbital basis:
a Pople basis such as 6-31G\* has no registered fitting basis and the
default call raises `NotImplementedError: No default jk-aux registered`.
For those bases, or for tight parity-validation against another code with
non-DF inputs, pass `density_fit=False, density_fit_mp2=False` (or name the
auxiliary bases explicitly with `aux_basis=` and `aux_basis_mp2=`).

### A double hybrid is a method, not a functional

A double hybrid names a complete method, so it is selected with `method=`.
Asking the plain Kohn-Sham runner for the same functional is refused,
because that call would return only the SCF half and silently omit the
perturbative correlation:

```python
try:
    vq.run_job(mol, basis="cc-pvdz", method="rks", functional="b2plyp")
except NotImplementedError as exc:
    print(exc)
# RKS with the double hybrid functional 'b2plyp' would return only its SCF
# component and omit the required perturbative correlation. Use
# vibeqc.run_double_hybrid(..., 'b2plyp') or the matching complete wrapper
# (run_b2plyp, run_dsd_pbep86, run_revdsd_pbep86, or run_pwpb95).
```

The accepted spellings are `method="b2plyp"`, `"dsd-pbep86"`,
`"revdsd-pbep86"` and `"pwpb95"`. Because this route is a separate
runner, `run_job` refuses options it does not carry (for example
`optimize=True` or `solvent=`) instead of dropping them.

## Comparing across the ladder

Computing total energies for water at the same geometry in cc-pVDZ across
HF → hybrid → MP2 → double-hybrid (all-electron correlation, the MP2 step of
the double hybrids uses no frozen core; the double hybrids carry no
dispersion here):

| Method | $E$ (Ha) | $\Delta E$ vs HF (mHa) |
|---|---:|---:|
| HF                                | −76.024314 | 0       |
| B3LYP (hybrid)                    | −76.377207 | −352.89 |
| PBE0 (hybrid)                     | −76.333343 | −309.03 |
| MP2 (post-SCF on HF)              | −76.225072 | −200.76 |
| SCS-MP2 (Grimme 2003)             | −76.221051 | −196.74 |
| SOS-MP2 (Jung et al. 2004)        | −76.219041 | −194.73 |
| **B2PLYP** (double hybrid)        | **−76.348110** | **−323.80** |
| DSD-PBEP86                        | −76.274960 | −250.65 |
| revDSD-PBEP86                     | −76.271466 | −247.15 |
| PWPB95                            | −76.355617 | −331.30 |

The MP2 rows are `E_HF + E_os + E_ss` with the scaling of each variant
applied to the two spin components of the MP2 result (`c_os = 6/5, c_ss = 1/3`
for SCS, `c_os = 1.3, c_ss = 0` for SOS). Total energies of different
functionals are not comparable as accuracies; the table shows how much
correlation each recipe recovers, not which is right.

Reading across:

* **Hybrids beat raw MP2 on this small system.** The DFT correlation
  recovered by B3LYP / PBE0 outpaces MP2's, because MP2 captures
  only dynamic correlation while a hybrid functional's GGA piece
  models both dynamic and static contributions semi-empirically.
* **B2PLYP sits between MP2 and B3LYP.** It pulls in 27 % of the MP2
  correlation on top of a hybrid SCF, so the correction is smaller
  than full MP2 but more rigorous than the GGA correlation alone.
* **The 0.27 MP2 fraction is system-dependent in its impact.** On
  this small molecule it adds ~64 mHa to the SCF step; on larger
  weakly-interacting complexes the relative weight grows.

## What's inside the `DoubleHybridResult`

The result object exposes both stages and their energy decomposition. The
attributes below reach the hybrid `RKSResult`, the post-SCF `MP2Result`,
and the assembled B2PLYP total:

```python
result.rks                       # full RKSResult of the hybrid step
result.rks.energy                # SCF total energy (the 0.53·HF + 0.47·B88 + 0.73·LYP step)
result.rks.e_hf_exchange         # −α/2 · tr(D K) with α = 0.53
result.rks.e_xc                  # E_x^B88 (0.47·) + E_c^LYP (0.73·)

result.mp2                       # full MP2Result of the post-SCF correction
result.mp2.e_os                  # unscaled opposite-spin energy on KS orbitals
result.mp2.e_ss                  # unscaled same-spin energy on KS orbitals
result.mp2.e_correlation         # 0.27 · (e_os + e_ss), the scaled MP2 piece

result.e_total                   # result.rks.energy + result.mp2.e_correlation
result.functional                # "b2plyp"
```

The `Functional` resolver carries the recipe too:

```python
from vibeqc import Functional

fn = Functional("b2plyp")
print(fn.hf_exchange_fraction)   # 0.53
print(fn.is_double_hybrid)       # True
print(fn.mp2_c_os, fn.mp2_c_ss)  # (0.27, 0.27)
```

## When to use B2PLYP

* **Reaction barriers, isomerisation energies, conformer ordering**, 
  B2PLYP is markedly more accurate than B3LYP on average across the
  GMTKN55 benchmark suites (Goerigk et al. 2017, in the references).
* **Weak interactions (vdW dimers, hydrogen bonds)**, pairs well
  with dispersion. The dispatcher folds D4 in directly:

  ```python
  result = run_b2plyp(mol, basis, dispersion="d4")  # B2PLYP-D4
  print(result.dispersion.energy)  # the D4 piece (Hartree)
  print(result.e_total)            # XC + MP2 + D4 total
  ```

  For water in cc-pVDZ the D4 term is -0.000128 Ha (total -76.348238 Ha)
  and the D3(BJ) term, `dispersion="d3bj"`, is -0.000234 Ha (total
  -76.348345 Ha). `D4` needs the optional `dftd4` package and `D3(BJ)` the
  `dftd3` package.

  See the
  [MP2 and double hybrids reference page](../user_guide/mp2_and_double_hybrids.md#dispersion-d3bj-and-d4)
  for the full dispersion section. D3(BJ) is also available via
  `vibeqc.compute_d3bj`, call it after `run_b2plyp(...)` and add
  the energies.
* **Large molecules**, the RI-MP2 step is formally O(N⁵) but with a much
  smaller prefactor than canonical MP2 and without the four-index integral
  storage, so B2PLYP on a 60-atom system with def2-TZVP is tractable on a
  workstation. RI is the default; canonical MP2 would be untractable past
  ~def2-SVP at that size.

When **not** to use B2PLYP:

* Multi-reference problems (transition-metal bond-breaking, biradical
  intermediates), MP2 (and double-hybrids inheriting it) collapse
  near degeneracies. Use a multi-reference method instead.
* Very large systems where the O(N⁵) MP2 step is still too expensive, pure
  hybrids (PBE0, B3LYP) at O(N³) with RIJCOSX are the right tool.

## What the `.out` reports

The dispatcher writes a compact `.out` (plus the usual Molden, XYZ,
population, citation and QVF siblings) whose central block names the model
components:

```text
Double-hybrid DFT dispatcher
  Functional:        b2plyp
  Basis:             cc-pvdz
  Charge:            0
  Multiplicity:      1
  MP2 frozen core:   0 (explicit all-electron model component)
  Construction:      hybrid RKS SCF + scaled RI-MP2 (c_os=0.27, c_ss=0.27)

  Energy components (Ha)
  ---------------------------------------------------
  RKS reference                        -76.2841541376
  MP2 correlation                       -0.0639561563
  Total energy                         -76.3481102939
  ---------------------------------------------------
```

The unscaled spin components behind the MP2 line are available
programmatically: `result.mp2.e_os = -0.176810` and
`result.mp2.e_ss = -0.060064` Ha, so
$0.27\,(E_\text{os} + E_\text{ss}) = -0.063956$ Ha, and `result.e_total`
is the sum of the two lines above.

### Open-shell molecules

For an open-shell molecule the same call runs the spin-pure ROKS SCF as the
hybrid half and a semicanonical restricted-open-shell RI-MP2 correction on
top of it:

```python
oh = Molecule([Atom(8, [0, 0, 0]), Atom(1, [0, 0, 1.83])], multiplicity=2)
oh_result = vq.run_job(oh, basis="cc-pvdz", method="b2plyp")
print(oh_result.e_total)   # -75.576872
print(type(oh_result.rks).__name__, type(oh_result.mp2).__name__)
# ROKSResult RefMP2Result
```

## Theory

A double hybrid keeps the mixing idea of a hybrid functional, exact
exchange in place of part of the semilocal exchange, and applies it to
the correlation as well: a fraction of the semilocal correlation is
replaced by a second-order perturbation term. The paragraphs below give the
energy expression, explain where the orbitals for the perturbation term
come from, and describe how the DSD family and the dispersion corrections
refine it.

### The energy expression

With $a_x$ the fraction of exact exchange and $c$ the weight of the
second-order term, the B2PLYP energy ({vibeqc-cite}`grimme_b2plyp_2006`) is

$$
E_\text{B2PLYP} = E_\text{ref}[\{\phi\}] \;+\; c\,E_c^{(2)},
\qquad
E_\text{ref} = E_\text{core+J} + (1-a_x)\,E_x^\text{B88} + a_x\,E_x^\text{HF} + (1-c)\,E_c^\text{LYP},
$$

with $a_x = 0.53$ and $c = 0.27$. The first part is an ordinary hybrid
Kohn-Sham energy (the "SCF step"), minimised self-consistently over the
orbitals $\{\phi\}$ and reported as `result.rks.energy`. The second part
is the MP2-type correlation energy evaluated once, non-self-consistently,
with the converged orbitals $\phi_p$ and orbital energies $\varepsilon_p$
of that SCF step,

$$
E_c^{(2)} = \sum_{i<j}\sum_{a<b}
  \frac{|\langle ij\|ab\rangle|^2}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
  = E_\text{os} + E_\text{ss},
$$

with $i,j$ occupied, $a,b$ virtual, and $E_\text{os}$, $E_\text{ss}$ its
opposite-spin and same-spin components. Because $c$ multiplies both spin
components equally, B2PLYP is a spin-component-unresolved double hybrid,
$c_\text{os} = c_\text{ss} = c$.

### Why the MP2 term sits on Kohn-Sham orbitals

The second-order term is the leading correction of the Goerling-Levy
adiabatic-connection expansion around the Kohn-Sham determinant, so the
natural orbitals for it are Kohn-Sham ones, not Hartree-Fock ones. B2PLYP
evaluates it with the orbitals of the 53 percent hybrid, a compromise
between the two, and compensates for the resulting overcounting by the
weights: the factor $(1-c)$ on the semilocal correlation and $c$ on the
perturbative term together account for the correlation once, while the
fitted $a_x$ and $c$ absorb what the truncation misses. The perturbation
term is not variationally coupled back into the orbitals, which is why the
result is an SCF step plus a post-SCF step and why the cost is that of a
hybrid SCF plus one MP2 (RI-MP2 with the default density fitting). The same
structure explains the failures: where MP2 breaks down because the
occupied-virtual gap closes (near-degeneracy, stretched bonds, some
transition-metal compounds), the double hybrid inherits the problem, only
damped by the factor $c$.

### The DSD family and dispersion

The DSD functionals ({vibeqc-cite}`kozuch_martin_dsd_2011`) drop the
requirement $c_\text{os} = c_\text{ss}$: the same-spin and opposite-spin
components of the perturbation term are scaled separately, in the spirit of
SCS-MP2,

$$
E_c^{(2)} \;\to\; c_\text{os}\,E_\text{os} + c_\text{ss}\,E_\text{ss},
$$

and the parameters are fitted together with a dispersion correction, so
that D3(BJ) or D4 is part of the definition of the method rather than an
add-on. In vibe-qc `Functional("dsd-pbep86")` carries $a_x = 0.70$,
$c_\text{os} = 0.53$, $c_\text{ss} = 0.25$; the revDSD-PBEP86-D4 refit
({vibeqc-cite}`santra_martin_revdsd_2019`) uses $a_x = 0.69$,
$c_\text{os} = 0.5922$, $c_\text{ss} = 0.0636$ and is tied to D4, not
D3(BJ); and PWPB95 ({vibeqc-cite}`goerigk_grimme_pwpb95_2011`) is a
meta-GGA double hybrid with $a_x = 0.50$, $c_\text{os} = 0.269$ and
$c_\text{ss} = 0$, a pure spin-opposite-scaled form. For any of them the
`dispersion=` keyword adds the correction (`"d3bj"` or `"d4"`), and the
energy reported without it is the "no-D" energy of the published model,
not its production value. Open-shell molecules run the ROKS SCF half
followed by a semicanonical restricted-open-shell MP2 correction, which
keeps the reference spin-pure.

## References

- **B2PLYP original.** S. Grimme, "Semiempirical hybrid density
  functional with perturbative second-order correlation,"
  *J. Chem. Phys.* **124**, 034108 (2006).
  [doi:10.1063/1.2148954](https://doi.org/10.1063/1.2148954).
  The defining paper for the 0.53 HF + 0.47 B88 / 0.73 LYP /
  0.27 MP2 mixing.
- **B3LYP components, for context.** A. D. Becke,
  *J. Chem. Phys.* **98**, 5648 (1993) (B3 mixing); C. Lee, W.
  Yang, R. G. Parr, *Phys. Rev. B* **37**, 785 (1988) (LYP); and
  P. J. Stephens, F. J. Devlin, C. F. Chabalowski, M. J. Frisch,
  *J. Phys. Chem.* **98**, 11623 (1994) (the canonical B3LYP recipe).
- **SCS-MP2.** S. Grimme, "Improved second-order Møller-Plesset
  perturbation theory by separate scaling of parallel- and
  antiparallel-spin pair correlation energies," *J. Chem. Phys.*
  **118**, 9095 (2003).
  [doi:10.1063/1.1569242](https://doi.org/10.1063/1.1569242).
- **SOS-MP2.** Y. Jung, R. C. Lochan, A. D. Dutoi, M. Head-Gordon,
  "Scaled opposite-spin second order Møller-Plesset correlation
  energy: An economical electronic structure method,"
  *J. Chem. Phys.* **121**, 9793 (2004).
  [doi:10.1063/1.1809602](https://doi.org/10.1063/1.1809602).
- **DSD double hybrids.** S. Kozuch and J. M. L. Martin, "DSD-PBEP86: in
  search of the best double-hybrid DFT with spin-component scaled MP2 and
  dispersion corrections," *Phys. Chem. Chem. Phys.* **13**, 20104 (2011);
  G. Santra, N. Sylvetsky, J. M. L. Martin, *J. Phys. Chem. A* **123**, 5129
  (2019) (revDSD-PBEP86-D4); L. Goerigk and S. Grimme, *J. Chem. Theory
  Comput.* **7**, 291 (2011) (PWPB95).
- **GMTKN benchmark of B2PLYP.** L. Goerigk, A. Hansen, C. Bauer,
  S. Ehrlich, A. Najibi, S. Grimme, "A look at the density
  functional theory zoo with the advanced GMTKN55 database for
  general main group thermochemistry, kinetics, and noncovalent
  interactions," *Phys. Chem. Chem. Phys.* **19**, 32184 (2017).
  [doi:10.1039/C7CP04913G](https://doi.org/10.1039/C7CP04913G).
  Pins B2PLYP at the top of the double-hybrid pack on
  general-purpose chemistry.

The bundled [citation database](../user_guide/citations.md) routes
`b2plyp` to Grimme 2006 plus the underlying B3LYP components, so
every `run_b2plyp` job auto-emits these in the `.bibtex`.

## Further reading

* [MP2 and double hybrids](../user_guide/mp2_and_double_hybrids.md)
  - full API reference for all MP2 variants and double hybrids.
* [DFT functional comparison](functional_comparison.md), the
  Jacob's-ladder tutorial that this one builds on.
* [Cross-code validation](cross_validation.md), how vibe-qc's
  results are validated against PySCF / ORCA references.
* [`examples/molecular/mp2_benchmarks/`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/mp2_benchmarks/README.md)
  - vibe-qc ↔ ORCA S22 + open-shell benchmark suite, including
  B2PLYP across all 22 S22 dimers.
