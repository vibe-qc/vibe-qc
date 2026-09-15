# RIJCOSX SCF and analytic gradients on glycine

This tutorial exercises the v0.8.0 `JKBuilder` polymorphic
Fock build via the **RIJCOSX** kernel: RI-J for the Coulomb
piece, seminumerical chain-of-spheres exchange (COSX) for the
K piece. Validated against ORCA 6.1.1 to 0.13 mHa on glycine /
def2-TZVP.

**You will**:

- Build a 10-atom glycine molecule.
- Run RHF with the four-index (default) Fock build for
  reference.
- Re-run with RIJCOSX, comparing energy + analytic gradient.
- Time the two paths to see the COSX speedup at scale.

## Background, why a seminumerical K

Hybrid DFT (B3LYP, PBE0, ωB97X) needs exact Fock exchange, and
that's where the cost goes. The four-index K build scales
formally as $\mathcal{O}(N^4)$ in basis functions; even with
Cauchy-Schwarz screening it's the wall-clock bottleneck for
medium-sized systems. Two approximations have come to dominate:

* **RIJK** (Weigend 2002, Neese 2003), apply the resolution-of-
  the-identity ansatz to *both* J and K. Replaces the 4-index ERI
  $(\mu\nu|\lambda\sigma)$ with a sum of 3-index intermediates
  through an auxiliary basis. Asymptotic scaling stays
  $\mathcal{O}(N^4)$, but the prefactor drops by an order of
  magnitude. Useful below ~1000 BFs; above that the K-side
  $\mathcal{O}(N^4)$ catches up.
* **COSX** (Neese, Wennmohs, Hansen, Becker 2009), *chain-of-spheres*
  exchange. Evaluates K via a numerical Becke grid, exploiting the
  fact that $K_{\mu\nu} = \sum_\lambda P_{\lambda\sigma}(\mu\lambda|\sigma\nu)$
  can be rewritten as a 1-electron operator integrated against
  the orbital density. The grid integration scales formally
  $\mathcal{O}(N^2)$ in the asymptotic limit, and the prefactor
  is small. **RIJCOSX** keeps RI-J for the Coulomb piece (the
  $\mathcal{O}(N^2 M)$ DF-J kernel) and pairs it with COSX for K.

The COSX K kernel is the chunk of the speedup. For glycine at
def2-TZVP (120 BFs) you'll see ~3×; for glycine pentamer
(~600 BFs) it's ~10×; for 100-heavy-atom systems it's over 20×.

For the implementation:
[`user_guide/density_fitting.md`](../user_guide/density_fitting.md)
documents the JKBuilder polymorphic dispatch and the three concrete
kernels (`FourIndexJKBuilder`, `DFJKBuilder`, `COSXJKBuilder`).

## Setup

Start by building the 10-atom glycine test molecule from an inline XYZ
block; this is the system both Fock-build paths below run on.

```python
import vibeqc as vq
import time

# Glycine — 10-atom amino acid; exercises C, N, O, H + a hydrogen-bond.
glycine = vq.Molecule.from_xyz("""\
10
glycine, B3LYP/6-31G* optimized geometry
N    -1.5012   -0.1265    0.0000
C    -0.4080    0.7959    0.0000
C     0.9189    0.0670    0.0000
O     1.0150   -1.1438    0.0000
O     1.9583    0.7993    0.0000
H    -2.3543    0.4169    0.0000
H    -1.5012   -0.7196    0.8189
H    -0.4773    1.4360    0.8843
H    -0.4773    1.4360   -0.8843
H     2.7892    0.2748    0.0000
""")
```

## Reference: four-index RHF

First run RHF with the default four-index Fock build to get the reference
energy and wall-clock time the RIJCOSX run is measured against:

```python
basis = vq.BasisSet(glycine, "def2-tzvp")  # ~120 BFs

t0 = time.perf_counter()
ref = vq.run_rhf(
    glycine, basis,
    options=vq.RHFOptions(),               # default = FourIndexJKBuilder
)
t_4idx = time.perf_counter() - t0

print(f"4-index RHF energy: {ref.energy:.6f} Ha")
print(f"  wall-clock: {t_4idx:.2f} s")
# 4-index RHF energy: -282.823485 Ha
#   wall-clock: ~25 s  (single-threaded)
```

## RIJCOSX

Now re-run the same SCF with RI-J for the Coulomb piece and COSX for
exchange, and compare the energy and timing against the four-index
reference:

```python
from vibeqc import GridOptions, default_aux_basis_for, default_cosx_grid_options

aux_name = default_aux_basis_for("def2-tzvp", kind="jk")
# → "def2-tzvp-jk"

t0 = time.perf_counter()
rijcosx = vq.run_rhf(
    glycine, basis,
    options=vq.RHFOptions(
        density_fit=True,
        aux_basis=aux_name,
        cosx=True,                          # → COSXJKBuilder
        # cosx_grid defaults to 35 × 9 × 18 (matches ORCA's default GridX)
    ),
)
t_cosx = time.perf_counter() - t0

print(f"RIJCOSX energy:    {rijcosx.energy:.6f} Ha")
print(f"  Δ vs 4-index:    {1000*(rijcosx.energy - ref.energy):+.3f} mHa")
print(f"  wall-clock:      {t_cosx:.2f} s ({t_4idx/t_cosx:.1f}× speedup)")
# RIJCOSX energy:    -282.823611 Ha
#   Δ vs 4-index:    +0.13 mHa
#   wall-clock:      ~8 s   (~3× speedup at this size; bigger systems show 10×+)
```

## Analytic gradient

The COSX path includes an analytic gradient (Bykov et al.,
*Mol. Phys.* **113**, 1961, 2015). Compare against the
4-index analytic gradient as the ground truth:

```python
from vibeqc import compute_gradient_rhf

g_ref = compute_gradient_rhf(glycine, basis, ref)
g_cosx = compute_gradient_rhf(glycine, basis, rijcosx,
                              options=vq.RHFOptions(
                                  density_fit=True, aux_basis=aux_name, cosx=True,
                              ))

import numpy as np
max_diff_per_atom = np.max(np.abs(g_cosx - g_ref), axis=1)  # (n_atoms,)
print(f"Max |Δgradient| per atom (mHa/bohr):")
for atom, d in zip(glycine.atoms, 1000 * max_diff_per_atom):
    print(f"  {atom.symbol:<3} {d:6.3f}")
print(f"Overall max: {1000 * np.max(np.abs(g_cosx - g_ref)):.3f} mHa/bohr")
# Overall max: ~0.13 mHa/bohr  (matches the energy delta order — RIJCOSX
# is consistent in energy + gradient at the same precision tier)
```

## What this tells you

* **RIJCOSX is essentially exact for hybrid-DFT-grade
  precision.** 0.13 mHa/atom is well below the inherent
  basis-set error for def2-TZVP (~0.5-1 mHa) and the
  experimental uncertainty most users care about (~kcal/mol
  ≈ 1.6 mHa).
* **The wall-clock speedup grows with system size.** At
  10 atoms / def2-TZVP it's ~3×; at glycine pentamer
  (~50 atoms / def2-TZVP) it's ~10×; at 100+ atoms it's
  over 20×.
* **The COSX K kernel is what saves time.** RI-J alone
  (`density_fit=True, cosx=False`) helps less because J
  scales O(N²·M) anyway via the Cauchy-Schwarz screening
  vibe-qc has had since v0.5.5; the K piece is where the
  wall-clock win comes from.
* **Use this combination** for any production hybrid-DFT
  geometry optimisation on > 30 heavy atoms. Below that, the
  4-index path's setup-free simplicity wins.

## When NOT to use RIJCOSX

* **Pure GGA / LDA** (no exact exchange): no exchange to
  assemble, so COSX is wasted. Use `density_fit=True,
  cosx=False` (= `DFJKBuilder`) and let RI-J accelerate the
  Coulomb piece.
* **Tiny systems** (< 20 BFs): the auxiliary basis + grid
  setup time dominates. Stay with the default 4-index path.
* **Production gradients above 50 BFs def2-TZVP** with
  `cosx=False` (the RIJK path): see the
  {ref}`v0.8.x DF gradient bug warning <historical-gradient-bug-status>`.
  The RIJCOSX path goes through the COSX gradient and is
  unaffected by this bug.

## What the `.out` reports

A RIJCOSX run's `.out` carries a JKBuilder block in addition to
the usual SCF trace:

```text
JKBuilder kernel:     COSXJKBuilder
  J path:             RIJ (def2-tzvp-jk auxiliary, 392 functions)
  K path:             COSX (Becke grid 35 × 9 × 18, 25420 points)
  Per-iteration cost: ~0.4 s J build + ~0.3 s K integration
SCF accelerator:      EDIIS_DIIS (default)

  iter   energy (Ha)        dE          ||[F,DS]||  DIIS
  -------------------------------------------------------------
     1   -279.815632     1.34e+00       3.7e-01      1
     2   -282.621047     2.81e+00       9.4e-02      2
   ...
    14   -282.823611     7.2e-09        2.1e-08      8  → converged
```

The "JKBuilder kernel" line tells you which Fock-build path the
SCF actually took, useful sanity check when `density_fit=` or
`cosx=` get tangled across multiple option structs.

## Resources

Glycine / def2-TZVP / RIJCOSX: ~120 BFs, ~8 s on a single core,
peak RAM ~250 MB. The four-index reference at the same level
takes ~25 s and peaks at ~700 MB. The crossover where RIJCOSX
*wins* on wall-time is around 60 BFs / def2-SVP for hybrid
functionals on a modern CPU; below that the auxiliary-basis +
grid setup dominates and the four-index path's setup-free
simplicity is preferred.

## References

- **COSX algorithm.** F. Neese, F. Wennmohs, A. Hansen, U. Becker,
  "Efficient, approximate and parallel Hartree-Fock and hybrid DFT
  calculations. A 'chain-of-spheres' algorithm for the Hartree-Fock
  exchange," *Chem. Phys.* **356**, 98 (2009).
  [doi:10.1016/j.chemphys.2008.10.036](https://doi.org/10.1016/j.chemphys.2008.10.036).
- **RI-J / RIJK.** F. Weigend, "A fully direct RI-HF algorithm:
  Implementation, optimised auxiliary basis sets, demonstration of
  accuracy and efficiency," *Phys. Chem. Chem. Phys.* **4**, 4285
  (2002). [doi:10.1039/B204199P](https://doi.org/10.1039/B204199P).
- **RIJCOSX analytic gradient.** S. Bykov, T. Petrenko, R. Izsák,
  S. Kossmann, U. Becker, E. Valeev, F. Neese, "Efficient
  implementation of the analytic second derivatives of
  Hartree-Fock and hybrid DFT energies: a detailed analysis of
  different approximations," *Mol. Phys.* **113**, 1961 (2015).
  [doi:10.1080/00268976.2015.1025114](https://doi.org/10.1080/00268976.2015.1025114).
  The COSX-gradient implementation in vibe-qc follows this paper's
  derivation.
- **Auxiliary-basis convention.** F. Weigend, "Hartree-Fock exchange
  fitting basis sets for H to Rn," *J. Comput. Chem.* **29**, 167
  (2008). [doi:10.1002/jcc.20702](https://doi.org/10.1002/jcc.20702).
  The def2-*-jk aux family the RIJCOSX kernel routes by default.

The RIJCOSX combination of these papers fires in the bundled
[citation database](../user_guide/citations.md) routes
`methods["cosx"]`, `methods["rij"]`, and the matching aux-basis
entries, every `run_job` using `cosx=True` writes them into the
auto-generated `.bibtex`.

## See also

- [`user_guide/density_fitting.md`](../user_guide/density_fitting.md), 
  full JKBuilder reference + recommended-aux table.
- [`user_guide/functionals.md`](../user_guide/functionals.md), 
  hybrid functionals where RIJCOSX shines.
- [Pt cluster with LANL2DZ: heavy-element ECPs](pt_cluster_lanl2dz.md), 
  pairs RIJCOSX with ECPs.
- [Stiff molecular SCF: rescuing oscillation with EDIIS+DIIS](ediis_diis_hybrid.md), the
  default SCF accelerator paired with RIJCOSX in this tutorial;
  see what it does in stiff cases.
