# Double hybrid: B2PLYP step by step

The [DFT functional comparison tutorial](functional_comparison.md)
walks Jacob's ladder up to rung 4 (hybrid DFT, B3LYP, PBE0). The next
rung is **double-hybrid DFT**: a hybrid-DFT SCF step plus a scaled
MP2 correlation correction on the converged KS orbitals.

| Rung | Input | Cost | Examples |
|---|---|---|---|
| 4. hybrid | + fraction of HF exchange | +~2× | B3LYP, PBE0 |
| **5. range-separated** | + long-range HF exchange | +~3× | ωB97X (not in vibe-qc yet) |
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
on water in a 6-31G\* basis and reads back the total energy, the hybrid
SCF piece, and the scaled MP2 correction separately:

```python
from vibeqc import Atom, Molecule, BasisSet, run_b2plyp

mol = Molecule([
    Atom(8, [0.0,  0.00,  0.00]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])
basis = BasisSet(mol, "6-31g*")

result = run_b2plyp(mol, basis)

print(result.e_total)               # the published B2PLYP energy
print(result.rks.energy)            # the hybrid SCF step (= total − MP2 corr)
print(result.mp2.e_correlation)     # 0.27 × (E_os + E_ss) on the KS orbitals
print(result.functional)            # "b2plyp"
```

`run_b2plyp` defaults to **density fitting on both steps**: RI-J +
RI-K for the hybrid SCF, RI-MP2 for the correction. Auxiliary bases
are auto-resolved from the orbital basis name (JKfit for SCF, RIfit
for MP2). For tight parity-validation against another code with
non-DF inputs, pass `density_fit=False, density_fit_mp2=False`.

## Comparing across the ladder

Computing total energies for water at the same geometry across
HF → GGA → hybrid → double-hybrid:

| Method | $E$ (Ha) | $\Delta E$ vs HF (mHa) |
|---|---:|---:|
| HF                                | −76.006678 | 0       |
| B3LYP (hybrid)                    | −76.363872 | −357.19 |
| PBE0 (hybrid)                     | −76.318570 | −311.89 |
| MP2 (post-SCF on HF)              | −76.189440 | −182.76 |
| SCS-MP2 (Grimme 2003)             | −76.185086 | −178.41 |
| SOS-MP2 (Jung et al. 2004)        | −76.182910 | −176.23 |
| **B2PLYP** (double hybrid)        | **−76.327746** | **−321.07** |

Reading across:

* **Hybrids beat raw MP2 on this small system.** The DFT correlation
  recovered by B3LYP / PBE0 outpaces MP2's, because MP2 captures
  only dynamic correlation while a hybrid functional's GGA piece
  models both dynamic and static contributions semi-empirically.
* **B2PLYP sits between MP2 and B3LYP.** It pulls in 27 % of the MP2
  correlation on top of a hybrid SCF, so the correction is smaller
  than full MP2 but more rigorous than the GGA correlation alone.
* **The 0.27 MP2 fraction is system-dependent in its impact.** On
  this small molecule it adds ~59 mHa to the SCF step; on larger
  weakly-interacting complexes the relative weight grows.

## What's inside the `DoubleHybridResult`

The result object exposes both stages and their energy decomposition. The
attributes below reach the hybrid `RKSResult`, the post-SCF `MP2Result`,
and the assembled B2PLYP total:

```python
result.rks                       # full RKSResult of the hybrid step
result.rks.energy                # SCF total energy (= 0.53·HF + 0.47·B88 + 0.73·LYP step)
result.rks.e_hf_exchange         # −α/2 · tr(D K) with α = 0.53
result.rks.e_xc                  # E_x^B88 (0.47·) + E_c^LYP (0.73·)

result.mp2                       # full MP2Result of the post-SCF correction
result.mp2.e_os                  # unscaled opposite-spin energy on KS orbitals
result.mp2.e_ss                  # unscaled same-spin energy on KS orbitals
result.mp2.e_correlation         # 0.27 · (e_os + e_ss) — the scaled MP2 piece

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
  B2PLYP is roughly half a kcal/mol better than B3LYP on the GMTKN
  benchmark suites.
* **Weak interactions (vdW dimers, hydrogen bonds)**, pairs well
  with dispersion. The dispatcher folds D4 in directly:

  ```python
  result = run_b2plyp(mol, basis, dispersion="d4")  # B2PLYP-D4
  print(result.dispersion.energy)  # the D4 piece (Hartree)
  print(result.e_total)            # XC + MP2 + D4 total
  ```

  See the
  [MP2 and double hybrids reference page](../user_guide/mp2_and_double_hybrids.md#dispersion-d3bj-and-d4)
  for the full dispersion section. D3(BJ) is also available via
  `vibeqc.compute_d3bj`, call it after `run_b2plyp(...)` and add
  the energies.
* **Large molecules**, the RI-MP2 step scales O(N⁴), so B2PLYP on a
  60-atom system with def2-TZVP is tractable on a workstation. RI is
  the default; canonical MP2 would be O(N⁵) and untractable past
  ~def2-SVP at that size.

When **not** to use B2PLYP:

* Multi-reference problems (transition-metal bond-breaking, biradical
  intermediates), MP2 (and double-hybrids inheriting it) collapse
  near degeneracies. Use a multi-reference method instead.
* Very large systems where O(N⁴) is still too expensive, pure
  hybrids (PBE0, B3LYP) at O(N³) with RIJCOSX are the right tool.

## What the `.out` reports

The `.out` carries both the hybrid SCF trace and the post-SCF MP2
block:

```text
═══════════════════════════════════════════════════════════════
Step 1 — Hybrid SCF (0.53 HF + 0.47 B88, 0.73 LYP, 0.27 LDA-VWN)
═══════════════════════════════════════════════════════════════
SCF trace
  iter   energy (Ha)        dE          ||[F,DS]||  DIIS
   ...
    11   -76.268783       4.1e-09       1.8e-08      7  → converged

Energy components
  Nuclear repulsion         +  9.193867 Ha
  Electronic kinetic        + 75.926413 Ha
  Coulomb (J)               + 46.806121 Ha
  HF exchange (α=0.53)      −  4.882104 Ha
  XC (B88 0.47 + LYP 0.73)  −  9.281240 Ha
  ─────────────────────────────────────────────
  E_RKS step                = -76.268783 Ha

═══════════════════════════════════════════════════════════════
Step 2 — Scaled MP2 correlation on the KS orbitals
═══════════════════════════════════════════════════════════════
RI-MP2 (aux basis: 6-31g*-rifit, 56 aux functions)
  E_os (opposite spin)      = -0.165743 Ha
  E_ss (same spin)          = -0.052126 Ha
  E_corr unscaled           = -0.217869 Ha
  E_corr scaled (×0.27)     = -0.058824 Ha

═══════════════════════════════════════════════════════════════
B2PLYP total
═══════════════════════════════════════════════════════════════
  E_RKS step                = -76.268922 Ha
  + 0.27 · E_MP2_corr       = -0.058824 Ha
  ─────────────────────────────────────────────
  E_B2PLYP                  = -76.327746 Ha
```

The same numbers are reachable programmatically (`result.rks.energy`,
`result.mp2.e_os`, `result.mp2.e_correlation`, `result.e_total`).

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
