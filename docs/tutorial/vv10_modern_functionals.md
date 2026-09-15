# Modern functionals: VV10 and the ωB97X-V / ωB97M-V wave

vibe-qc's range-separated hybrids spent a release without their
intended correlation partner: the "-V" members (ωB97X-V, ωB97M-V) are
defined *with* the VV10 nonlocal correlation term, and without it they
cannot be run as their authors meant. That gap is now closed. With the
**VV10** kernel in the C++ core, those functionals evaluate completely
and self-consistently, and the same wave adds the **r2SCAN01** meta-GGA
and the **revDSD-PBEP86-D4** double hybrid. This page is the worked
companion to the [functionals user guide](../user_guide/functionals.md);
it shows how to invoke each new functional and what it is for. For the
older, conventional range-separated hybrid ωB97X (which carries no VV10
term), start with [Range-separated hybrids: ωB97X](range_separated_wb97x.md).

Every snippet below uses water at a fixed geometry so you can paste and
run them directly:

```python
import vibeqc as vq

mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.00, 0.00]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
)
```

## VV10: the nonlocal correlation piece

VV10 (Vydrov and Van Voorhis 2010) adds a genuinely *nonlocal*
correlation term: a double integral over all pairs of grid points that
recovers long-range dispersion from the density alone, with no empirical
pairwise C6 table. It is the ingredient that completes the "-V"
functionals below. vibe-qc ships it both as a standalone functional and
as a kernel you can call on any grid:

```python
basis = vq.BasisSet(mol, "cc-pvdz")

opts = vq.RKSOptions()
opts.functional = "vv10"          # rPW86 exchange + PBE correlation + the VV10 nonlocal term
result = vq.run_rks(mol, basis, opts)
print(f"E(VV10) = {result.energy:.6f} Ha")
```

The nonlocal potential is folded into the SCF exchange-correlation
build, so both the energy and its potential are exact (not a post-hoc
add-on). The C++ kernel is exposed directly as
`vibeqc.compute_vv10(coords, weights, rho, sigma, b, C)` for evaluation
on an arbitrary density, with the published parameters b = 5.9,
C = 0.0093. vibe-qc's kernel matches PySCF's `_vv10nlc` on a shared
water / cc-pVDZ grid to within 1e-8 Ha (`tests/test_vv10.py`).

Inside an SCF the nonlocal double integral runs on its own, coarser grid:
`GridOptions.vv10_grid_factor` (default `3.0`) divides the radial and
angular resolution of the XC grid for the VV10 term only, the same
practice Vydrov and Van Voorhis used in their implementation paper (a
coarse SG-1 grid for the nonlocal term under a (75, 302) grid for the
semilocal part, because the nonlocal term is far less sensitive to grid
fineness). Set `opts.grid.vv10_grid_factor = 1.0` to evaluate VV10 on the
full XC grid when you need the dense reference.

## ωB97X-V and ωB97M-V: the complete range-separated hybrids

These are the functionals that VV10 unlocks. **ωB97X-V** is a
range-separated *GGA* hybrid plus VV10; **ωB97M-V** is a range-separated
*meta-GGA* hybrid plus VV10, and is among the best-performing functionals
on broad main-group benchmarks. Both compose three advanced pieces at
once: the erf-attenuated long-range exact exchange, the semilocal
(meta-)GGA, and the VV10 nonlocal term. You select them by name, exactly
like any other functional:

```python
opts = vq.RKSOptions()
opts.functional = "wb97m-v"       # range-separated meta-GGA + VV10 in one functional
result = vq.run_rks(mol, basis, opts)
print(f"E(ωB97M-V) = {result.energy:.6f} Ha")
```

vibe-qc reproduces PySCF's `dft.RKS(xc="wb97m-v", nlc="VV10")` to within
5e-5 Ha at the same grid (`tests/test_wb97m_v.py`); ωB97X-V matches its
PySCF reference to the same tolerance (`tests/test_vv10.py`). The
`Functional` resolver carries the recipe, so you can confirm what you are
running:

```python
fn = vq.Functional("wb97m-v")
print(fn.is_range_separated)   # True
print(fn.rsh_omega)            # 0.3  (bohr^-1)
print(fn.needs_vv10)           # True
print(fn.vv10_b, fn.vv10_C)    # 6.0 0.01  (the -V functionals re-fit b and C)
```

```{important}
**The "-V" range-separated hybrids run through direct SCF only.** As with
plain ωB97X, the erf-attenuated exchange has no 3-centre density-fitting
path yet, so the molecular driver selects the direct Fock builder
automatically; leave `density_fit` at its default. They are **molecular
RKS / UKS only** (periodic RSH and periodic VV10 are queued), and have
**no analytic gradient yet**, finite-difference geometry optimization
works via `run_job(..., optimize=True)`.
```

## r2SCAN01: a re-regularized meta-GGA

r2SCAN01 (Furness et al. 2022) targets a known weakness of r²SCAN: it
restores the fourth-order term of the slowly-varying gradient expansion
that r²SCAN drops, by re-regularizing with a larger η. The practical
effect is better behaviour for slowly-varying densities, bulk metals and
bonding regions, where r²SCAN slightly underbinds. It is a pure meta-GGA
(no exact exchange), so it is cheap, and it is τ-only (laplacian-free),
which fits vibe-qc's meta-GGA grid:

```python
basis_svp = vq.BasisSet(mol, "def2-svp")

opts = vq.RKSOptions()
opts.functional = "r2scan01"
result = vq.run_rks(mol, basis_svp, opts)
print(f"E(r2SCAN01) = {result.energy:.6f} Ha")
```

For this water geometry in def2-SVP, vibe-qc returns **E = -76.3151 Ha**,
matching PySCF's `MGGA_X/C_R2SCAN01` to 3.9 µHa (the asserted reference
in `tests/test_xc.py`). Unlike the range-separated hybrids above,
r2SCAN01 has **analytic molecular gradients** (the Pulay force includes
the τ term), so native geometry optimization works.

## revDSD-PBEP86-D4: a re-parametrized double hybrid

A double hybrid runs a hybrid-DFT SCF step and adds a scaled MP2
correlation correction. **revDSD-PBEP86-D4** (Santra, Sylvetsky and
Martin 2019) is a careful re-fit of DSD-PBEP86 with two distinguishing
features: an *asymmetric* MP2 scaling (different coefficients for
opposite-spin and same-spin pairs), and D4 as its **native** dispersion,
the functional coefficients were fit jointly with the D4 damping, so D4
is part of the definition rather than an afterthought. Use the dedicated
dispatcher:

```python
basis = vq.BasisSet(mol, "cc-pvdz")
result = vq.run_revdsd_pbep86(mol, basis)

print(f"E(hybrid SCF step)   = {result.rks.energy:.6f} Ha")
print(f"E(MP2 correction)    = {result.mp2.e_correlation:.6f} Ha")   # 0.5922*E_os + 0.0636*E_ss
print(f"E(revDSD-PBEP86)     = {result.e_total:.6f} Ha")
```

The SCF base is 0.69 HF + 0.31 PBE exchange + 0.4210 P86 correlation; the
MP2 correction uses c_os = 0.5922 and c_ss = 0.0636. The dispatcher
returns a `DoubleHybridResult` carrying `.rks`, `.mp2`, and `.e_total`,
and matches a PySCF reference to within 5e-5 Ha
(`tests/test_revdsd_pbep86.py`). To obtain the published
revDSD-PBEP86-**D4** total, add the native dispersion:

```python
result = vq.run_revdsd_pbep86(mol, basis, dispersion="d4")
```

```{note}
`dispersion="d4"` needs the optional `dftd4` package (`pip install
dftd4`); without it, drop the keyword for the SCF + MP2 total. Note also
that the older revDSD-PBEP86-**D3BJ** uses a *different* coefficient fit
and is not reachable through this alias, the `revdsd-pbep86` name is the
D4-native parametrization.
```

## What ships, and what is still queued

The table is the snapshot for this functional wave:

| Functional | `xc=` string | Status |
|---|---|---|
| VV10 (standalone) | `vv10` | ✅ shipped, molecular RKS / UKS, self-consistent |
| ωB97X-V | `wb97x-v` | ✅ shipped, molecular, direct SCF, FD gradient |
| ωB97M-V | `wb97m-v` | ✅ shipped, molecular, direct SCF, FD gradient |
| r2SCAN01 | `r2scan01` | ✅ shipped, molecular, analytic gradient |
| revDSD-PBEP86(-D4) | `revdsd-pbep86` | ✅ shipped via `run_revdsd_pbep86`, D4 native |
| ωB97M(2) | `wb97m(2)` | ⏳ gated, needs a from-scratch B97M meta-GGA base (not in libxc) |
| Periodic RSH / periodic VV10 | (n/a) | ⏳ queued |
| RSH analytic gradients, RSH + density fitting | (n/a) | ⏳ queued |

`xc="wb97m(2)"` raises a clear message explaining why it is held back: it
is the one member of this family that is not a composition of stock libxc
parts (its SCF base is a re-optimised B97M power series that libxc does
not ship).

## References

- **VV10.** O. A. Vydrov, T. Van Voorhis, *J. Chem. Phys.* **133**,
  244103 (2010).
- **ωB97X-V.** N. Mardirossian, M. Head-Gordon, *Phys. Chem. Chem. Phys.*
  **16**, 9904 (2014).
- **ωB97M-V.** N. Mardirossian, M. Head-Gordon, *J. Chem. Phys.* **144**,
  214110 (2016).
- **r2SCAN01.** J. W. Furness, A. D. Kaplan, J. Ning, J. P. Perdew,
  J. Sun, *J. Chem. Phys.* **156**, 034109 (2022).
- **revDSD-PBEP86.** G. Santra, N. Sylvetsky, J. M. L. Martin,
  *J. Phys. Chem. A* **123**, 5129 (2019).
- **D4 dispersion.** E. Caldeweyher, C. Bannwarth, S. Grimme,
  *J. Chem. Phys.* **150**, 154122 (2019).

Each functional's route is registered in the bundled
[citation database](../user_guide/citations.md), so a job that uses one
auto-emits the right reference in its `.bibtex` and `.references`
siblings; the entries above are the authoritative form.

## Next

- [User guide: functionals](../user_guide/functionals.md), the API
  reference this tutorial is the worked companion to.
- [Range-separated hybrids: ωB97X](range_separated_wb97x.md), the
  conventional (no-VV10) long-range-corrected hybrid and the HOMO ≈ −IP
  demonstration.
- [Double hybrids: B2PLYP](double_hybrid_b2plyp.md), the annotated
  walk-through of the hybrid-SCF-plus-scaled-MP2 machinery that
  revDSD-PBEP86 rides.
- [Dispersion corrections](dispersion.md), the D3 and D4 story behind the
  `dispersion="d4"` keyword.
