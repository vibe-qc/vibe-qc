# Functionals

Most conventional XC functionals are resolved through
[libxc](https://libxc.gitlab.io/) at runtime. Any functional name libxc
accepts works: short names
(`"LDA"`, `"PBE"`, `"B3LYP"`, `"PBE0"`), libxc XC\_… identifiers,
**weighted-sum custom strings** (`"0.2*HF + 0.8*GGA_X_PW91, GGA_C_PW91"`
for PW1PW), and vibe-qc's own aliases. External full-grid functionals such
as Microsoft SKALA-1.1 use the same SCF drivers through a separate
differentiable provider interface and do not call libxc.

> For open-shell DFT you choose between **UKS** (unrestricted) and
> **ROKS** (spin-pure restricted open-shell). ROKS supports LDA, GGA,
> meta-GGA, global-hybrid, and range-separated-hybrid functionals; see
> the dedicated [ROHF and ROKS](rohf.md) page for the UKS-vs-ROKS
> decision and the invocation. That family statement covers libxc-backed
> functionals; external providers declare their own ROKS capability below.

## ⚠️ B3LYP convention, vibe-qc ships the ORCA definition

**The `b3lyp` keyword resolves to libxc id 475
(`HYB_GGA_XC_B3LYP5`, VWN5 local correlation, the ORCA / ADF
convention), as it has since v0.8.0.** v0.12.0 adds explicit
flavor spellings (`b3lyp5`, `b3lypg`) and the cross-code audit
below, but the default is unchanged.

"B3LYP" is a fixed hybrid recipe (Stephens-Devlin-Chabalowski-
Frisch 1994); codes differ only in *which* Vosko-Wilk-Nusair
parametrisation fills the 0.19-weighted LSDA-correlation slot
(Hertwig & Koch, *Chem. Phys. Lett.* **268**, 345 (1997)). The
field is split into two camps:

| Camp | Codes | vibe-qc spelling |
|------|-------|------------------|
| VWN5 (Ceperley-Alder fit) | ORCA, TURBOMOLE, ADF, CRYSTAL | `b3lyp` (= `b3lyp5`) |
| VWN-RPA ("VWN(III)" in Gaussian's naming) | Gaussian, libxc, PySCF (`b3lyp` *and* `b3lypg`, ≥ 2.3), Psi4 (≥ 1.2), NWChem, Q-Chem | `b3lyp/g` (= `b3lypg`) |

vibe-qc follows the ORCA camp: it is internally consistent with
`lda` / `svwn` (also VWN5), and it is the flavor CRYSTAL, vibe-qc's
primary periodic parity reference, computes.

Flavor gap: **~6.8 mHa on H₂/STO-3G, ~10-15 mHa per heavy atom**
(it is exactly 0.19·(E_c[VWN-RPA] − E_c[VWN5])). Measured
cross-code on H₂/STO-3G at 1.4 bohr: `b3lyp` = −1.1586001 Ha,
matching CRYSTAL14's `B3LYP` keyword to 1e-9 Ha and ORCA's bare
`B3LYP` to ~1 µHa; `b3lyp/g` = −1.1654009 Ha, matching PySCF 2.13's
`b3lyp` to <1e-9 Ha and ORCA's `B3LYP/G` to ~1 µHa.

Practical guidance:

* **For Gaussian / PySCF / Psi4 parity**, pass
  `functional="b3lyp/g"` (or the PySCF-style spelling
  `"b3lypg"`; `functional="402"` and
  `functional="XC_HYB_GGA_XC_B3LYP"` resolve to the same
  functional). PySCF evaluates the VWN-RPA flavor for **both**
  `xc="b3lyp"` and `xc="b3lypg"`.
* **For ORCA / TURBOMOLE / CRYSTAL / CP2K parity**, the default
  is already right; `functional="b3lyp5"` is an explicit spelling
  of the same functional for unambiguous cross-code tables.
* When publishing, say *which* B3LYP you used (cite Hertwig &
  Koch 1997, the auto-citation block does this for you, see
  § Citations).
* There is deliberately **no `b3lyp3` alias**: libxc's
  `LDA_C_VWN_3` is the Ceperley-Alder fit III, numerically
  identical to VWN5 on closed-shell densities, *not* Gaussian's
  "VWN(III)" (the RPA fit), so the obvious name would select the
  wrong flavor exactly when users reach for it.

## Recipe table

vibe-qc resolves these short names in `xc.cpp::resolve_alias()`
(plus everything libxc accepts directly):

| Name | Type | HF-exchange | libxc id | Notes |
|------|------|-------------|----------|-------|
| `lda` / `svwn` | LDA | 0 % | 1 + 7 | Slater exchange + VWN5 correlation. Fast sanity check. |
| `pbe` | GGA | 0 % | 101 + 130 | Workhorse for solid state. |
| `blyp` | GGA | 0 % | 106 + 131 | Classic molecular GGA. |
| `b3lyp` (= `b3lyp5`) | hybrid | 20 % | 475 | de-facto for organics; VWN5 / ORCA convention, **see the flavor box above**. |
| `b3lyp/g` (= `b3lypg`) | hybrid | 20 % | 402 | Gaussian-compatible B3LYP (VWN-RPA), what Gaussian / PySCF / Psi4 mean by the bare name. |
| `pbe0` / `pbeh` | hybrid | 25 % | 406 | PBE + 25% Fock exchange. |
| `pw1pw` | hybrid | 20 % | weighted-sum | Bredow's 1-parameter periodic hybrid; cite Bredow & Gerson PRB 61, 5194 (2000). |
| `tpss` | meta-GGA | 0 % | 202 + 231 | Non-empirical τ-dependent MGGA; Tao-Perdew-Staroverov-Scuseria PRL 91, 146401 (2003). |
| `tpssh` | hybrid meta-GGA | 10 % | 457 | 10 % global hybrid of TPSS; Staroverov-Scuseria-Tao-Perdew JCP 119, 12129 (2003). |
| `m06-l` (`m06l`) | meta-GGA | 0 % | 203 + 233 | Pure Minnesota meta-GGA; Zhao-Truhlar JCP 125, 194101 (2006). |
| `m06-2x` (`m062x`) | hybrid meta-GGA | 54 % | 450 + 236 | 54 % global hybrid for thermochemistry + non-covalent interactions; Zhao-Truhlar TCA 120, 215 (2008). |
| `scan` | meta-GGA | 0 % | 263 + 267 | Strongly-constrained meta-GGA; Sun-Ruzsinszky-Perdew PRL 115, 036402 (2015). **Grid-sensitive, use a fine grid** (prefer `r2scan`). |
| `r2scan` | meta-GGA | 0 % | 497 + 498 | Re-regularized SCAN, smooth, grid-friendly; Furness et al JPCL 11, 8208 (2020). Recommended default of the family. |
| `r2scan01` | meta-GGA | 0 % | 645 + 642 | r²SCAN with the fourth-order gradient expansion restored (larger η); Furness et al JCP 156, 034109 (2022). Aimed at slowly-varying densities where r²SCAN slightly underbinds. |
| `r2scan0` | hybrid meta-GGA | 25 % | 660 | 25 % global hybrid of r²SCAN (PBE0-like). |
| `r2scanh` | hybrid meta-GGA | 10 % | 659 | 10 % global hybrid of r²SCAN (TPSSh-like). |
| `wb97x` | range-sep hybrid | 15.77 %→100 % | 464 | Long-range-corrected hybrid GGA; Chai-Head-Gordon 2008. ω = 0.3. Direct SCF only (see § Range-separated hybrids). |
| `wb97x-d` | range-sep hybrid + D | 22.2 %→100 % | 471 | ωB97X-D, Chai-Head-Gordon 2008. ω = 0.2 + intrinsic CHG dispersion. Use `vibeqc.run_wb97x_d` for the full XC + dispersion total. |
| `hse06` | screened RSH | 25 %→0 % | 428 | Heyd-Scuseria-Ernzerhof screened hybrid; 25 % HF short-range, 0 % long-range, ω = 0.11. The solid-state band-gap workhorse. |
| `skala-1.1` (`skala`) | neural non-local meta-GGA | 0 % | external | Microsoft SKALA-1.1 rev1; optional PyTorch runtime and immutable model download. See the dedicated section below. |

The full libxc functional set is also reachable directly. Use
the libxc identifier if your shorthand isn't in the table:
`functional="XC_HYB_GGA_XC_M06_2X"`, `functional="XC_GGA_X_RPBE,XC_GGA_C_PBE"`,
etc.

## PW1PW: Bredow's periodic hybrid

`pw1pw` is the first **non-libxc-built-in custom hybrid** in
vibe-qc. The functional is:

```
PW1PW = 0.20 × HF exchange  +  0.80 × GGA_X_PW91 exchange  +  GGA_C_PW91 correlation
```

Implemented via libxc's custom-string path (the alias `"pw1pw"`
expands to `".2*HF + .8*GGA_X_PW91, GGA_C_PW91"`).

```python
import vibeqc as vq

result = vq.run_periodic_job(
    mgo,
    basis="pob-tzvp",
    method="RKS",
    functional="pw1pw",
    kmesh=(4, 4, 4),
    output="mgo-pw1pw-444",
)
```

Cite: Bredow, T.; Gerson, A. R. *Phys. Rev. B* **61**, 5194
(2000). PW1PW is the strict-comparison reference for the
[Peintinger 2013 SI](https://doi.org/10.1002/jcc.23153) basis-
set paper's HF + PW1PW totals on ~60 compounds.

## Defining custom functionals at runtime (`define_functional`)

Added in v0.11.3: you can register a CRYSTAL-style one-parameter hybrid
(or any weighted-sum functional) **from the Python input without a C++
edit**. This is the same mechanism that ships PW1PW, PBEh-3c, and the
double-hybrid SCF pieces internally.

```python
import vibeqc as vq

vq.define_functional(
    "my-hybrid",
    [("GGA_X_PW91", 0.80), ("GGA_C_PW91", 1.00)],
    hf_exchange_fraction=0.20,
)

# From this point on, "my-hybrid" resolves exactly like a built-in:
f = vq.Functional("my-hybrid")
assert f.hf_exchange_fraction == 0.20
assert f.is_hybrid

# Use it in any driver:
result = vq.run_rks(mol, basis, vq.RKSOptions(functional="my-hybrid"))
result = vq.run_periodic_job(system, basis, method="RKS",
                             functional="my-hybrid", kpoints=[8,8,8])
```

``components`` is a list of ``(libxc_name, weight)`` pairs. Any libxc
name that `xc_functional_get_number` recognises works, `"GGA_X_PBE"`,
`"GGA_C_LYP"`, `"MGGA_C_BC95"`, etc. Weights scale each component's
contribution to the XC energy; they need not sum to 1.0.

``hf_exchange_fraction`` is the **global** HF-exchange fraction for
hand-mixed hybrids (PW1PW-style). Set it to 0.0 for a pure-DFT custom
mix; non-zero makes the functional a hybrid and routes it through the
exact-exchange code path automatically.

**Thread-safe.** Call `define_functional` once at startup, before any
SCF job that references the alias. Registration is case-insensitive;
"MY-HYBRID" and "my-hybrid" resolve to the same alias.

A full example is in `examples/molecular/define_functional_demo.py`.

### Full-grid providers (`define_external_functional`)

Machine-learned or otherwise non-pointwise XC models can register a Python
callback with `vq.define_external_functional`. This complete toy functional
integrates a quadratic of the total density. It is an API example, not a
scientific functional:

```python
import numpy as np
import vibeqc as vq


def quadratic_density_xc(values):
    weights = np.asarray(values["grid_weights"], dtype=float)
    rho_alpha = np.asarray(values["rho_alpha"], dtype=float)
    rho_beta = np.asarray(values["rho_beta"], dtype=float)
    rho = rho_alpha + rho_beta
    coefficient = -0.01

    n_points = weights.size
    zero_scalar = np.zeros(n_points)
    zero_vector = np.zeros((n_points, 3))
    v_rho = 2.0 * coefficient * weights * rho

    return {
        "energy": float(coefficient * np.dot(weights, rho * rho)),
        "v_rho_alpha": v_rho,
        "v_rho_beta": v_rho.copy(),
        "v_grad_alpha": zero_vector,
        "v_grad_beta": zero_vector.copy(),
        "v_tau_alpha": zero_scalar,
        "v_tau_beta": zero_scalar.copy(),
    }


vq.define_external_functional(
    "toy-full-grid-xc",
    quadratic_density_xc,
    capability_version=1,
    required_grid_profile="",  # Accept the caller's canonical profile.
)
```

The callback input mapping contains `functional`, `grid_profile`,
atom-major `points`, final `grid_weights`, raw
`atomic_grid_weights`, `atom_coords`, `atomic_grid_sizes`, both spin
densities, their Cartesian gradients, and both kinetic-energy densities.
Periodic calls also provide periodic metadata. Returned adjoints are
derivatives of the already integrated energy and must therefore include
quadrature weights. Do not multiply them by weights a second time.

Capability version 1 can set `required_grid_profile` to a canonical profile
such as `"pyscf-level3"`. That declaration validates the incoming grid but
does not select it. Only the built-in SKALA aliases receive automatic SKALA
grid selection; a custom provider's caller must choose its required grid.

Registration is process-lifetime, case-insensitive, and insert-only. A custom
name cannot shadow a built-in or replace another provider. External providers
are not attributed to libxc; the application is responsible for supplying the
provider's defining citation.

## Microsoft SKALA-1.1 neural XC

`skala-1.1` is Microsoft's differentiable, non-local neural
exchange-correlation functional. vibe-qc supplies an independent full-grid
adapter for the official CPU TorchScript checkpoint; it does not import the
upstream PySCF-based runtime. The aliases `skala` and `skala-1.1-rev1`
select the same immutable artifact.

The high-level molecular and periodic runners automatically select vibe-qc's
pinned PySCF 2.14 level-3 parity profile. This reproduces the grid used by the
upstream PySCF integration and benchmark setup; it is a vibe-qc compatibility
choice, not a constraint encoded by the checkpoint. Molecular RKS, UKS, and
ROKS fixed-geometry energies and potentials are available. Selected periodic
AO-grid routes are experimental. Derivatives, optimization, response, and
incompatible periodic XC backends fail closed.

The checkpoint contains the XC model only. Its metadata recommends the
separate D3(BJ) setting `b3lyp5`; request
`dispersion="b3lyp5"` explicitly when that molecular protocol is intended.

See the dedicated [Microsoft SKALA-1.1 guide](skala.md) for a complete
runnable example, optional-runtime and platform requirements, exact grid and
cache contracts, D3 result semantics, offline staging, provenance, memory
planning, the qualified periodic route matrix, scientific scope, and
troubleshooting.

## Choosing a functional (decision tree)

* **Periodic ionic insulator** (oxides, halides, sulphides):
  `pw1pw` or `b3lyp` or `pbe0` for hybrids; `pbe` for pure GGA.
  Cite the published reference for the system class.
* **Periodic semiconductor / metal**: `pbe` (most common).
  Hybrids hit the dense-Lpq RAM ceiling fast, see
  [`multi_k_scf.md`](multi_k_scf.md) § Open issues.
* **Molecular organic geometry optimisation**: `r²SCAN-3c`, 
  the turnkey "Swiss army knife" composite (see
  [`composites.md`](composites.md)), or `b3lyp` with D3(BJ)
  (see [`molecules.md`](molecules.md)) / `pbe0` for more
  reactive systems.
* **Molecular thermochemistry**: the `r²SCAN-3c` / `ωB97X-3c` /
  `HSE-3c` composite methods are runnable today
  ([`composites.md`](composites.md)); `pbe0` + def2-TZVP +
  D3(BJ) is the conventional mid-tier workhorse.
* **Open-shell radicals**: prefer `uks` driver over `rks`.
  Same functional table works.

## ⚠️ Plain B3LYP for noncovalent interactions

Plain B3LYP (no dispersion correction) systematically
**under-binds** noncovalent interactions by 20-50% on the
S22 set. **Always pair B3LYP with a dispersion correction**
when noncovalent interactions matter:

```python
result = vq.run_job(
    mol,
    basis="def2-tzvp",
    method="rks",
    functional="b3lyp",
    dispersion="d3bj",  # or "d4"
)
```

D3(BJ) is the standard. D4 (Caldeweyher-Bannwarth-Grimme, 2019)
ships alongside D3(BJ): use `dispersion="d4"` to opt in. There
are two D4 backends, the optional `dftd4` package (production
default, bit-exact to upstream) and an in-tree native MPL-2.0
implementation (`compute_d4(mol, func, backend="native")`) that
is parity-validated for **H-Ne** (correlated CPKS/PBE38 reference
C6 within a few percent of dftd4); use the `dftd4` backend for
elements outside that set. Both are documented in
[`molecules.md`](molecules.md) § Dispersion.

## Meta-GGA functionals (TPSS / TPSSh, M06-L / M06-2X, SCAN / r²SCAN)

vibe-qc evaluates the kinetic-energy density
τ(r) = ½ Σ_i |∇ψ_i(r)|² on the DFT grid (using the AO gradients
that already power the GGA path), so the τ-dependent meta-GGA
family is now available in **molecular RKS + UKS + ROKS**:

```python
opts = vq.RKSOptions()
opts.functional = "r2scan"    # or tpss / tpssh / m06-l / m06-2x /
                              # scan / r2scan01 / r2scan0 / r2scanh
result = vq.run_rks(mol, basis, opts)
```

**SCAN vs r²SCAN.** SCAN (`scan`) satisfies all 17 exact
constraints a semilocal functional can, but its sharp enhancement
factor is notoriously grid-sensitive, on vibe-qc's default medium
grid the H₂O/def2-SVP energy sits ~1 mHa off the fine-grid limit.
**For SCAN, set a fine grid** (`opts.grid.n_radial = 120`,
`opts.grid.lebedev_order = 41` or finer). **r²SCAN** (`r2scan`) is
the re-regularized SCAN, same constraint satisfaction, a smooth
enhancement factor, grid-stable to ~µHa on the default grid. It is
the recommended default of the family; `r2scan0` (25 % HF) and
`r2scanh` (10 % HF) are its global hybrids. **r²SCAN01** (`r2scan01`)
re-regularizes r²SCAN with a larger η so the fourth-order term of
the slowly-varying gradient expansion (dropped by r²SCAN) is
restored; it targets slowly-varying densities such as bulk metals
and bonding regions where r²SCAN slightly underbinds.

Coverage:

* **Molecular RKS / UKS / ROKS**, closed-shell + open-shell SCF live;
  parity to PySCF.dft within grid-quadrature noise (~µHa on def2-SVP /
  medium grid for closed-shell H₂O; ~10-50 µHa on STO-3G /
  `grids.level=5` for O₂ triplet UKS). ROKS retains PySCF-parity coverage
  for the integer-occupation convention and has dedicated fractional
  frontier-shell coverage for degenerate OH(2Pi).
* **Periodic SCF**, *available*: `cpp/src/periodic_xc.cpp`'s shared XC
  kernel (`build_xc_periodic`) carries meta-GGA through, with a von
  Weizsäcker τ floor fixing the earlier TPSS/M06-L SCF + analytic
  gradient divergence. Closed-shell multi-k GAPW and open-shell (Γ) GPW
  meta-GGA are both wired (`run_periodic_rks_gapw_multi_k`,
  `run_periodic_uks_gpw` / `run_periodic_uhf_gpw`), and r²SCAN01 ships
  across the GDF/Ewald/multi-k/BIPOLE-Γ periodic routes too.
* **Analytic gradients**, *available* for molecular RKS + UKS
  (`vibeqc.compute_gradient_rks` / `compute_gradient_uks`). The XC
  Pulay force includes the τ term; finite-difference validated to
  ~3e-9 (TPSS/TPSSh) / ~1e-7 (M06-2X) on H₂/STO-3G. Geometry
  optimisation with meta-GGAs works end-to-end.
* **Analytic Hessian / CPKS**, *not yet*. The meta-GGA
  second-derivative kernel (`eval_unpolarised_fxc`) raises on MGGA;
  finite-difference Hessians work as a workaround.
* **Laplacian-dependent meta-GGAs** (TB09, …), *not supported*.
  vibe-qc populates τ but not ∇²ρ on the grid; libxc functionals
  with `XC_FLAGS_NEEDS_LAPLACIAN` are refused at `Functional`
  construction with a clear error.

## Range-separated hybrids (ωB97X, ωB97X-D)

A range-separated (CAM) hybrid makes the exact-exchange admixture
*position-dependent*:

```
EXX(r₁₂) = cam_alpha + cam_beta · erf(ω · r₁₂)
```

so the exchange operator is
`cam_alpha·(1/r₁₂) + cam_beta·(erf(ω·r₁₂)/r₁₂)`. vibe-qc reads the
(ω, α, β) triple from libxc at construction, 
`Functional("wb97x").is_range_separated / rsh_omega / cam_alpha /
cam_beta`, and the SCF driver builds the second, erf-attenuated
exchange matrix K_erf(ω) alongside the ordinary K.

`wb97x` (Chai-Head-Gordon 2008) is the first range-separated hybrid
in vibe-qc: a long-range-corrected functional, 15.77 % HF at short
range ramping to **100 % HF at long range**, ω = 0.3 bohr⁻¹.

```python
opts = vq.RKSOptions()
opts.functional = "wb97x"
result = vq.run_rks(mol, basis, opts)
```

### ωB97X-D, range-separated hybrid + dispersion

`wb97x-d` (Chai-Head-Gordon 2008) is the dispersion-corrected
re-parameterisation of ωB97X (ω = 0.2 bohr⁻¹, 22.2 % HF short-range →
100 % long-range). It has **two pieces**: the range-separated-hybrid
XC functional (libxc `XC_HYB_GGA_XC_WB97X_D`), and an intrinsic
empirical "-D" dispersion correction of the older "DFT-D2" family.
The dispersion is geometry-only, no SCF coupling, so the complete
ωB97X-D energy is `E_SCF + E_disp`. Use the dispatcher
:func:`vibeqc.run_wb97x_d` for the full total:

```python
result = vq.run_wb97x_d(mol, basis)          # closed- or open-shell
print(result.e_total)        # E_SCF + E_dispersion
print(result.scf.energy, result.dispersion.energy)
```

`run_wb97x_d` and `run_double_hybrid` apply `grid_level="orca-defgrid3"`
to absent or untouched SCF options, for closed and open shells. Custom grid
fields take precedence. Pass `grid_level="legacy"` to reproduce the old
integration grid. The low-level `run_rks` and `run_uks` wrappers continue
to use a supplied options object's grid as given.


The CHG dispersion is
`E_disp = −Σ_{A<B} C6_AB/R^6 · 1/(1 + 6·(R/R0_AB)^-12)` with the
Grimme-2006 D2 atomic C6 / vdW-radius tables (H-Xe); a heavier
element raises a clear error. `vibeqc.compute_chg_dispersion(mol)`
exposes the standalone term. The bare `functional="wb97x-d"` alias
through `run_rks` / `run_uks` gives the **XC/SCF part only** (the
range-separated hybrid), the same alias-vs-dispatcher split as
`b2plyp` / `run_b2plyp`; use `run_wb97x_d` for the published total.

Coverage and caveats:

* **Molecular RKS / UKS / ROKS energies**, live; parity to PySCF.dft at
  grid-quadrature precision (~µHa, def2-SVP; ROKS coverage includes the
  ωB97X exchange split and the ωB97M-V meta-GGA/VV10 combination).
* **Direct SCF only**, the erf-attenuated K is built by the
  direct-SCF Fock kernel. `density_fit=true` with a range-separated
  functional is rejected with a clear error (the RI path has no
  erf-attenuated 3-centre integrals yet). The molecular drivers
  select the direct builder automatically for RSH functionals, so no
  user action is needed, just leave `density_fit` at its default.
* **Hard open-shell radicals**, ωB97X's 100 % long-range HF makes
  open-shell SCF stiffer. O₂, CH₃, and atomic radicals converge with
  the defaults; an orbital-near-degenerate case such as the OH ²Π
  radical can plateau near the gradient tolerance with plain DIIS, 
  set `opts.level_shift = 0.5` to converge it cleanly (standard QC
  practice for stiff open-shell SCF).
* **Newton / TRAH** second-order acceleration is disabled for RSH
  functionals (their orbital-Hessian exchange response is not yet
  range-separation aware); DIIS + SOSCF + the quadratic fallback
  carry convergence.
* **Analytic gradients** for RSH functionals are *not yet* available
  (finite-difference works). ωB97X-V and periodic RSH are queued, 
  see below.

### HSE06, screened range-separated hybrid

`hse06` (Heyd, Scuseria, Ernzerhof; the Krukau-Vydrov-Izmaylov-
Scuseria 2006 re-parameterisation, libxc `XC_HYB_GGA_XC_HSE06`) is a
*screened* RSH: 25 % HF at short range tapering to **0 % at long
range**, the opposite ramp direction to ωB97X. In vibe-qc's erf-form
CAM representation this is `cam_alpha = +0.25`, `cam_beta = −0.25`
(`EXX(r→0) = 0.25`, `EXX(r→∞) = 0`), so it runs through the *same*
erf-attenuated-K machinery as ωB97X, just with a negative long-range
coefficient. No separate erfc kernel is needed.

```python
opts = vq.RKSOptions()
opts.functional = "hse06"
result = vq.run_rks(mol, basis, opts)
```

HSE06 is primarily a solid-state band-gap functional. Molecular HSE06
is available as above, and **periodic HSE06 ships now** on the routes
that build a genuine erfc short-range exchange kernel: the Ewald
family (`jk_method="slab_ewald_2d"` / AUTO on dim=2 slabs, plus the
Γ and multi-k `EWALD_3D` drivers) and `jk_method="bipole"`. The
periodic assembly is `K_HF = 0.25 · K_erfc(ω = 0.11)`, a pure
short-range direct lattice sum with **no** Madelung/exxdiv seam
(the erfc kernel has no G → 0 divergence; this matches CRYSTAL's
SR-exchange treatment). Routes whose exact exchange is full-range
only (GDF at Γ and multi-k, GPW/GAPW, COSX) **fail closed** on any
range-separated functional rather than silently running HSE06 as
PBE0, which is what they did before 2026-07-12. Long-range-corrected
RSH (ωB97X, CAM-B3LYP, LC-ωPBE) fail closed on *every* periodic
route: their full-range exchange arm needs an exxdiv treatment that
is not yet validated. Analytic periodic gradients for screened
hybrids are not available (the gradient entry points fail closed
too); use finite differences.

### Analytic gradients: what the RSH / VV10 kernels are missing (GitLab #571)

The molecular analytic RKS / UKS gradient (`cpp/src/gradient.cpp`) carries
exact exchange through `Functional.hf_exchange_fraction()` only, which for a
range-separated hybrid is the short-range constant `cam_alpha`. The
erf-attenuated arm `cam_beta · K_erf(ω)` that the SCF builds for the energy
has no gradient counterpart, and neither has the VV10 nonlocal correlation
(Vydrov and Van Voorhis, *J. Chem. Phys.* **133**, 244103 (2010)). Measured on
a bent O/H/H molecule in STO-3G against the full-energy central-difference
gradient (step 0.005 bohr): PBE 2.1e-6 Ha/bohr (the noise floor), HSE06
5.6e-3, VV10 7.0e-4, ωB97X-V 5.6e-2, with max |g| about 0.1 Ha/bohr.

What that means in practice:

* `vibeqc.functional_gradient_terms_missing("wb97x-v")` returns the two
  missing terms (an empty list for LDA / GGA / meta-GGA / global hybrids).
* `vibeqc.compute_gradient_rks` / `compute_gradient_uks` raise
  `NotImplementedError` for such a `result.functional`, naming the terms.
  `allow_incomplete=True` returns the partial analytic gradient knowingly;
  it is not the derivative of the reported energy.
* `run_job(optimize=True)`, `optimize_molecule`, `optimize_molecule_brent`,
  the `geomopt` provider and the `VibeQC` ASE calculator switch to
  full-energy central finite differences (step 0.005 bohr) for these
  functionals and write one `Note:` line to the `.out` saying so. Global
  hybrids (PBE0, B3LYP) keep their analytic gradient.
* Frequencies (`compute_hessian_fd`, which finite-differences the analytic
  gradient), the analytic-Hessian skeletons, CPCM gradients and NEB fail
  closed with the same message for these functionals.

## Recently landed, and still-queued refinements

Several families below landed after the v0.8.0 cut; each bullet marks
what ships now and what is still queued. Track the remainder at
[`docs/roadmap.md`](../roadmap.md) § XC functional library
expansion:

* **Range-separated hybrids**, molecular `wb97x`, `wb97x-d` (via
  `vibeqc.run_wb97x_d`), `hse06`, and now **`wb97x-v` and `wb97m-v`**
  **ship now** (see § Range-separated hybrids above). The VV10
  non-local correlation that ωB97X-V / ωB97M-V depend on landed
  (`vibeqc.compute_vv10`; `xc="vv10"` is the standalone functional),
  so these are the *complete* functionals, evaluated self-consistently
  on the SCF grid (worked walk-through:
  [Modern functionals: VV10 and the ωB97X-V / ωB97M-V wave](../tutorial/vv10_modern_functionals.md)).
  Still queued: RSH *analytic gradients*, RSH + *density fitting*, and
  *periodic* RSH. Until the RSH exchange and VV10 gradient terms land,
  the molecular gradient entry points **refuse** these functionals and
  the optimizers **fall back to finite differences** (see the next
  paragraph; GitLab #571).
* **r²SCAN family**, molecular `scan` / `r2scan` / `r2scan0` /
  `r2scanh` **ship now** (see § Meta-GGA functionals above).
  **Periodic** SCAN / r²SCAN also ship (the periodic meta-GGA
  XC kernel landed in v0.12.x); the r²SCAN gradient (cell-opt
  Pulay τ term) is queued.
* **Composite "3c" methods**, **r²SCAN-3c, ωB97X-3c, and HSE-3c
  are runnable today** (turnkey: functional + matching basis +
  gCP + D4 in one call); B3LYP-3c / B97-3c / PBEh-3c are
  `PENDING_GCP_DATA` (the framework is wired, the per-basis gCP
  tables are pending). See [`composites.md`](composites.md) for
  the recipe table, status flags, and worked examples.
* **Double hybrids**, **B2PLYP** (Grimme 2006), **DSD-PBEP86**
  (Kozuch-Martin 2011), **revDSD-PBEP86-D4** (Santra-Sylvetsky-Martin
  2019), and **PWPB95** (Goerigk-Grimme 2011) ship now via
  [`vibeqc.run_b2plyp`](mp2_and_double_hybrids.md#b2plyp) /
  [`vibeqc.run_dsd_pbep86`](mp2_and_double_hybrids.md#dsd-pbep86) /
  `vibeqc.run_revdsd_pbep86` /
  [`vibeqc.run_pwpb95`](mp2_and_double_hybrids.md#pwpb95), with
  `vibeqc.run_double_hybrid(name, …)` as the generic dispatcher.
  PWPB95 is vibe-qc's first **meta-GGA** double hybrid and its first
  **spin-opposite-scaled** one (the same-spin MP2 coefficient is
  zero). The SCS-MP2 / SOS-MP2 machinery the double-hybrid line rides
  is also live; see the
  [MP2 and double hybrids](mp2_and_double_hybrids.md) page.
  **ωB97M(2)** (range-separated meta-GGA double hybrid) remains
  blocked, but no longer on VV10 (that landed): unlike the other
  double hybrids it is not a composition of stock libxc components,
  its SCF base is a re-optimised B97M power-series functional that
  libxc does not ship, so it needs a dedicated from-scratch meta-GGA
  kernel. `xc="wb97m(2)"` raises a message saying so. **D4 dispersion**
  ships via the optional
  `dftd4` package, pass `dispersion="d4"` to `run_b2plyp` /
  `run_dsd_pbep86` / `run_revdsd_pbep86` / `run_pwpb95` for the
  published `X-D4` total (for revDSD-PBEP86, D4 is the native
  parametrisation), or call `vibeqc.compute_d4` directly for arbitrary
  functionals.
  D3(BJ) is also wired through the molecular runner via the existing
  `vibeqc.compute_d3bj` machinery.

## Programmatic access

```python
from vibeqc import Functional

fn = Functional("PBE0")
print(fn.hf_exchange_fraction())    # 0.25
print(fn.kind())                    # XCKind.GGA   (well, GGA-hybrid)

# Custom libxc composition strings (comma-separated; X then C):
fn = Functional("XC_LDA_X,XC_LDA_C_VWN")            # explicit Slater + VWN
fn = Functional("XC_GGA_X_PBE,XC_GGA_C_PBE")        # PBE from separate parts

# Weighted-sum custom strings (PW1PW pattern):
fn = Functional("0.2*HF + 0.8*GGA_X_PW91, GGA_C_PW91")
```

The `Functional` ctor docstring lists every known short alias
the resolver accepts.

## Citations

Every `run_job` call writes a [`{stem}.bibtex`](citations.md) and
`{stem}.references` sibling alongside `{stem}.out`, pre-assembled
with all the per-functional papers your job needs to cite. The list
below is the human-readable equivalent of what the runtime emits;
look at the auto-generated files first.

* **B3LYP** (all flavor spellings, `b3lyp`, `b3lyp5`, `b3lyp/g`,
  `b3lypg`): Becke, *J. Chem. Phys.* **98**, 5648 (1993);
  Lee, Yang, Parr, *Phys. Rev. B* **37**, 785 (1988);
  Stephens, Devlin, Chabalowski, Frisch, *J. Phys. Chem.* **98**,
  11623 (1994); Vosko, Wilk, Nusair, *Can. J. Phys.* **58**,
  1200 (1980) (both the VWN5 fit used by the default `b3lyp` and
  the RPA fit used by `b3lyp/g` are from this paper); Hertwig,
  Koch, *Chem. Phys. Lett.* **268**, 345 (1997) (which flavor is
  which, cite it so your Methods section pins the variant).
* **PBE**: Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**,
  3865 (1996).
* **PBE0**: PBE 1996 above plus Adamo, Barone, *J. Chem. Phys.*
  **110**, 6158 (1999).
* **PW91 / PW1PW**: Perdew, Chevary, Vosko, Jackson, Pederson,
  Singh, Fiolhais, *Phys. Rev. B* **46**, 6671 (1992); Bredow,
  Gerson, *Phys. Rev. B* **61**, 5194 (2000) for the PW1PW
  hybrid mixing.
* **B2PLYP** (and other double hybrids): Grimme, *J. Chem. Phys.*
  **124**, 034108 (2006), plus the B3LYP components above.
* **TPSS / TPSSh**: Tao, Perdew, Staroverov, Scuseria,
  *Phys. Rev. Lett.* **91**, 146401 (2003); Staroverov, Scuseria,
  Tao, Perdew, *J. Chem. Phys.* **119**, 12129 (2003) for the 10 %
  global-hybrid (TPSSh) variant.
* **M06-L / M06-2X**: Zhao, Truhlar, *J. Chem. Phys.* **125**,
  194101 (2006) (M06-L, pure meta-GGA); Zhao, Truhlar,
  *Theor. Chem. Acc.* **120**, 215 (2008) (Minnesota M06 family,
  including the 54 %-hybrid M06-2X).
* **SCAN**: Sun, Ruzsinszky, Perdew, *Phys. Rev. Lett.* **115**,
  036402 (2015).
* **ωB97X**: Chai, Head-Gordon, *J. Chem. Phys.* **128**, 084106
  (2008).
* **ωB97X-D** (and its CHG / DFT-D2-type dispersion): Chai,
  Head-Gordon, *Phys. Chem. Chem. Phys.* **10**, 6615 (2008); the
  D2 atomic C6 / vdW-radius set is Grimme, *J. Comput. Chem.* **27**,
  1787 (2006).
* **HSE06**: Heyd, Scuseria, Ernzerhof, *J. Chem. Phys.* **118**,
  8207 (2003); erratum *ibid.* **124**, 219906 (2006); Krukau,
  Vydrov, Izmaylov, Scuseria, *J. Chem. Phys.* **125**, 224106
  (2006) (the "06" re-parameterisation).
* **r²SCAN** (and `r2scan0` / `r2scanh`): Furness, Kaplan, Ning,
  Perdew, Sun, *J. Phys. Chem. Lett.* **11**, 8208 (2020);
  erratum *ibid.* **11**, 9248 (2020).
* **r²SCAN01**: Furness, Kaplan, Ning, Perdew, Sun, *J. Chem.
  Phys.* **156**, 034109 (2022).
* **D3(BJ) / D4** dispersion: Grimme, Antony, Ehrlich, Krieg,
  *J. Chem. Phys.* **132**, 154104 (2010); Grimme, Ehrlich,
  Goerigk, *J. Comput. Chem.* **32**, 1456 (2011) (BJ damping
  extension); Caldeweyher, Bannwarth, Grimme, *J. Chem. Phys.*
  **150**, 154122 (2019).
* **libxc** itself: Lehtola, Steigemann, Oliveira, Marques,
  *SoftwareX* **7**, 1 (2018).

Pre-v0.8.x workflows hand-curated this list from
[`docs/citing.md`](../citing.md). The runtime database now does it
for you, see the
[citations user guide](citations.md) for the full schema, the
[auto-citations tutorial](../tutorial/auto_citations.md) for the
end-to-end manuscript workflow, and
[Archived `AGENTS.md` § "Citation database ownership"](https://vibe-qc.com/docs/)
for the rule on keeping the database in sync when adding a new
functional.

## See also

* [`density_fitting.md`](density_fitting.md), JKBuilder
  + RIJCOSX (the right Fock build for big-system hybrid DFT).
* [`multi_k_scf.md`](multi_k_scf.md), KRHF / KRKS multi-k
  periodic SCF. PW1PW + B3LYP + PBE0 work at multi-k via the
  KRKS driver.
* [`molecules.md`](molecules.md), molecular SCF +
  dispersion + analytic gradients.
* [`scf_convergence.md`](scf_convergence.md), DIIS / EDIIS
  hybrid + level-shift / quadratic-fallback for stiff
  functionals.
