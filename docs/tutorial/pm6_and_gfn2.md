---
myst:
  html_meta:
    "description": "Two more semiempirical engines in vibe-qc: PM6 (molecular NDDO development route) and GFN2-xTB (extended tight-binding, experimental). How to run each with run_job, real H2O energies, and why semiempirical totals live on each method's own reference scale."
    "og:title": "vibe-qc tutorial - PM6 and GFN2-xTB semiempirical methods"
---

# More semiempirical methods: PM6 and GFN2-xTB

Beyond [DFTB](semiempirical_dftb.md) and [MSINDO](msindo.md), vibe-qc ships
two more semiempirical engines for fast energies and gradients on large
systems: **PM6**, a molecular NDDO development route, and **GFN2-xTB**, an
extended tight-binding method that is experimental in vibe-qc. Both are far cheaper
than a minimal-basis Hartree-Fock and are the right tool for
preoptimization, conformer screening, and systems too large for ab initio.

## PM6

PM6 is selected with `method="pm6"` and needs no basis-set argument: it
carries its own minimal valence basis and parameter set.

```python
import vibeqc as vq

mol = vq.Molecule([vq.Atom(8, [0, 0, 0]),
                   vq.Atom(1, [0,  1.43, -0.98]),
                   vq.Atom(1, [0, -1.43, -0.98])])
result = vq.run_job(mol, method="pm6", output="h2o_pm6")
print(result.energy)
```

```text
SCF converged in 8 iterations;  E = -10.21015659 Ha
```

PM6 provides energies and finite-difference gradients, so it plugs into
geometry optimization the same way the ab-initio methods do.

The current source-correct scope covers H-only and H-heavy s/p interactions.
The full heavy-heavy two-center NDDO tensor is still open, so molecules with
more than one non-hydrogen atom are not a quantitative PM6 reference yet.

## GFN2-xTB (experimental)

GFN2-xTB is Grimme's extended semiempirical tight-binding method. It is
**experimental** in vibe-qc and emits a `GFN2ExperimentalWarning`, which you
opt past explicitly:

```python
import warnings
import vibeqc as vq

with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # opt in to the experimental method
    result = vq.run_job(mol, method="gfn2_xtb", output="h2o_gfn2")
print(result.energy)
```

```text
E = -4.85491960 Ha
```

## Reading the numbers

Semiempirical total energies live on **each method's own parametrised
reference scale**. The PM6 number (−10.21 Ha) and the GFN2-xTB number
(−4.85 Ha) are not comparable to each other, nor to an ab-initio total (the
same water at HF / 6-31G* is −76.01 Ha). What *is* meaningful is **relative**
energies within one method (conformers, reaction energies) and the
geometries and properties the method produces. The standard workflow is to
get a structure or a trend fast with a semiempirical method, then refine the
energetics with DFT or a correlated method.

## See also

- [Semiempirical DFTB](semiempirical_dftb.md) and [MSINDO](msindo.md): the
  other two molecular semiempirical engines.
- [Machine-learning interatomic potentials with MACE](mace_mlip.md): the
  other fast, non-ab-initio route.
- [The cyclic cluster model](cyclic_cluster_model.md): MSINDO carried to
  periodic systems.
