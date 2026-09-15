# MP2 and double hybrids

vibe-qc ships second-order Møller-Plesset perturbation theory (MP2)
as a post-SCF correlation method for molecules, plus the
spin-component-scaled variants (SCS-MP2, SOS-MP2) and the first
double-hybrid density functional (B2PLYP). All variants compose with
density fitting (RI-MP2) and with the unrestricted (UHF) reference.

## At a glance

| Variant | Entry point | Reference | Recommended default |
|---|---|---|---|
| Canonical MP2 | `run_mp2(mol, basis, rhf)` | Szabo-Ostlund eq. 6.71 | Parity validation only |
| RI-MP2 | `run_mp2(..., MP2Options(density_fit=True, aux_basis="cc-pvtz-ri"))` | Vahtras-Almlöf-Feyereisen 1993 | Preferred accuracy route; production-size coverage pending BUG 63 demonstrations |
| SCS-MP2 | `run_scs_mp2(mol, basis, rhf)` | Grimme 2003 (c_os=6/5, c_ss=1/3) | Thermochemistry, reactions |
| SOS-MP2 | `run_sos_mp2(mol, basis, rhf)` | Jung-Lochan-Dutoi-Head-Gordon 2004 (c_os=1.3, c_ss=0) | Even cheaper than SCS |
| Canonical / RI / SCS / SOS UMP2 | `run_ump2`, `run_scs_ump2`, `run_sos_ump2` | UHF reference | Open-shell |
| B2PLYP | `run_b2plyp(mol, basis)` | Grimme 2006 | First double hybrid |

The high-level molecular dispatcher `run_job(..., method="mp2")`
runs the same **canonical, four-index** MP2 kernel as
`run_mp2(mol, basis, rhf)` on an RHF reference for closed-shell
systems, or UMP2 on a UHF reference for open-shell systems. It does
not auto-select RI by default, so use the default route for like-for-like
canonical parity checks against another code's non-RI MP2 route. To run
RI-MP2 through `run_job`, pass `mp2_options` with `density_fit=True` and
`aux_basis` set; open-shell UMP2 uses `ump2_options` the same way.
If `aux_basis` is empty, `run_job` auto-detects the per-zeta RI auxiliary
basis when one is registered for the orbital basis.

All public canonical/RI/SCS/SOS MP2 routes now freeze the chemical core by
default, for restricted and unrestricted references alike. The count follows
ORCA 6.1 Table 2.69: ORCA's frozen-electron count is converted to closed-shell
spatial orbitals and summed over the atoms. Use the same high-level escape for
every route when matching `NoFrozenCore` or an archived all-electron value:

```python
from vibeqc import run_job

all_electron = run_job(
    mol,
    basis="cc-pvtz",
    method="mp2",
    frozen_core=False,
)
```

The resolved count is printed in `.out`; compare that count, the reference
type, and the RI/conventional choice before comparing energies. Low-level
option objects expose an explicit orbital count for tests and specialist
workflows, but `frozen_core=False` is the preferred public spelling.

This is deliberately a **count-only** implementation of the ORCA table.
vibe-qc removes that many lowest-energy occupied orbitals; it does not yet
reproduce ORCA's additional atomic-orbital-character selection when canonical
orbitals have an unusual ordering. ECP-bearing calculations need particular
care: molecular ECP references are currently rejected by every correlated
route because their variational electron count differs from the molecule's
physical count. `frozen_core=False` controls correlation-space freezing; it
does not restore electrons removed by an ECP, so it does not bypass this
fail-closed boundary.

The four `run_*` convenience wrappers (`run_scs_mp2`, `run_sos_mp2`,
`run_scs_ump2`, `run_sos_ump2`) **default to RI-MP2 with the per-zeta
RIfit aux auto-resolved** from the orbital basis name. Canonical MP2
is reachable only via the explicit `density_fit=False` flag, the
brief's recommended setup throughout, since canonical MP2 scales as
O(N⁵) in the AO→MO ERI transform while RI-MP2 scales as O(N⁴) with
sub-µHa drift on the standard bases.

## Canonical MP2 + RI-MP2

```python
from vibeqc import Atom, BasisSet, Molecule, RHFOptions, MP2Options
from vibeqc import run_rhf, run_mp2, default_aux_basis_for

mol = Molecule.from_xyz("h2o.xyz")
basis = BasisSet(mol, "cc-pvtz")

# RHF reference. MP2 requires a converged RHF result.
rhf_opts = RHFOptions(); rhf_opts.conv_tol_energy = 1e-10
hf = run_rhf(mol, basis, rhf_opts)

# Canonical MP2 (O(N⁵)) - useful for parity validation.
mp2 = run_mp2(mol, basis, hf)
print(mp2.e_total, mp2.e_correlation)

# RI-MP2 (O(N⁴)); preferred over the canonical parity route.
mp2_opts = MP2Options()
mp2_opts.density_fit = True
mp2_opts.aux_basis = default_aux_basis_for(basis.name, kind="ri")
mp2_df = run_mp2(mol, basis, hf, mp2_opts)

# High-level runner using the same RI-MP2 options.
from vibeqc import run_job
run_job(mol, basis="cc-pvtz", method="mp2", mp2_options=mp2_opts)
```

Both canonical and RI kernels expose an explicit storage policy. `auto`
selects an in-core or bounded direct contraction from the requested workspace;
it never selects disk implicitly. `direct` recomputes exact canonical
occupied-`i` slabs or contracts RI occupied-`i` panels, while an explicit
`disk` request writes those completed slabs to the selected scratch directory
before the energy pass. The result records
the resolved policy and modeled workspace so scaling tests do not infer the
algorithm from molecule size:

```python
mp2_opts.memory_mode = "auto"       # auto | incore | direct | disk
mp2_opts.requested_memory_bytes = 2 * 1024**3
mp2_opts.scratch_directory = "/local/scratch"  # only used by disk mode

result = run_job(
    mol,
    basis="cc-pvtz",
    method="mp2",
    mp2_options=mp2_opts,
    memory_budget_bytes=2 * 1024**3,
)
print(result.mp2.memory_mode_used, result.mp2.workspace_bytes)
```

The options budget bounds the MP2 kernel's planned workspace; the `run_job`
process budget also reserves estimator headroom and live SCF result matrices
before deriving that smaller native allowance. RI direct mode removes the
global OVOV tensor, but the current generic density-fitting constructor still
builds and retains eager `n_aux * n_basis**2` factors. Large crambin/cc-pVTZ
and comparable inputs therefore remain outside the production claim until
real constrained-RSS runs complete. A dry-run estimate is not that evidence.

`MP2Result` carries the per-component Grimme decomposition (see next
section). Sum invariant: `e_os + e_ss == e_correlation` at canonical
`c_os = c_ss = 1`.

## Spin-component-scaled MP2

The MP2 correlation energy splits into an opposite-spin (αβ singlet
pair) piece and a same-spin (αα + ββ triplet pair) piece. The
**Grimme decomposition** (Grimme, *J. Chem. Phys.* **118**, 9095 (2003))
sets:

```
e_os = Σ (ia|jb)² / Δ                          (singlet pair)
e_ss = Σ [(ia|jb)² − (ia|jb)(ib|ja)] / Δ       (triplet pair)
e_correlation(c_os, c_ss) = c_os · e_os + c_ss · e_ss
```

with Δ = ε_i + ε_j − ε_a − ε_b. Setting different (c_os, c_ss) gives
the named recipes:

| Recipe | c_os | c_ss | Origin |
|---|---|---|---|
| Canonical MP2 | 1 | 1 | Szabo-Ostlund eq. 6.71 |
| **SCS-MP2** | **6/5** | **1/3** | Grimme 2003 (semi-empirical fit) |
| **SOS-MP2** | **1.3** | **0** | Jung et al. 2004 (drop same-spin) |
| B2PLYP correction | 0.27 | 0.27 | Grimme 2006 (double hybrid) |
| DSD double hybrids | varies | varies | Kozuch-Martin 2011 |

vibe-qc's `MP2Options` exposes `c_os` and `c_ss` (defaults 1.0, 1.0).
Both **compose with `density_fit=True`** without code duplication, 
the same kernel computes (e_os, e_ss); the scales apply at the final
sum.

```python
from vibeqc import run_scs_mp2, run_sos_mp2

scs = run_scs_mp2(mol, basis, hf)            # RI by default, c_os=6/5, c_ss=1/3
sos = run_sos_mp2(mol, basis, hf)            # RI by default, c_os=1.3, c_ss=0

# Custom scaling factors:
mine = run_scs_mp2(mol, basis, hf, c_os=1.10, c_ss=0.50)
```

The unrestricted variants (`run_scs_ump2`, `run_sos_ump2`) apply the
same coefficients to UMP2's channel decomposition: `c_os` scales the
αβ channel (`UMP2Result.e_ab`), `c_ss` scales the αα + ββ sum
(`UMP2Result.e_aa + UMP2Result.e_bb`).

### When to use which

* **SCS-MP2** is a well-validated improvement over MP2 across
  thermochemistry, reaction barriers, conformer ordering. Default if
  you don't have a specific reason to pick something else.
* **SOS-MP2** drops the more expensive same-spin term entirely; with
  Laplace-transformed denominators it is O(N⁴) without RI. vibe-qc's
  vanilla implementation is O(N⁴) (via RI) for both; SOS still
  outperforms SCS on cost-per-accuracy in some regimes (Jung et al.).
* **Canonical MP2** is for parity validation against reference codes
  (PySCF.MP2, ORCA `! MP2`), every published MP2 number used the
  unscaled formula.

## Verifying the RI fit: `report_ri_residual`

RI-MP2 replaces the canonical four-index (ia|jb) with the Coulomb-metric
Dunlap fit

$$(ia|jb)_{\mathrm{RI}} = \sum_{PQ} (ia|P)\, [V^{-1}]_{PQ}\, (Q|jb).$$

For any finite auxiliary basis this introduces a per-bucket residual.
The residual is **structurally biased**: $E_{\mathrm{os}}$ is
over-estimated (less negative than canonical) and $E_{\mathrm{ss}}$ is
under-estimated (more negative than canonical), and the two errors
partially cancel in the unscaled sum but survive SCS / SOS scaling.
Concretely, on H₂O / cc-pVTZ:

| Aux basis | $n_{\mathrm{aux}}$ | $\Delta E_{\mathrm{os}}$ | $\Delta E_{\mathrm{ss}}$ | $\Delta E_{\mathrm{SCS}}$ |
|---|---:|---:|---:|---:|
| cc-pvtz-ri | 141 | +5.05e-5 | −2.43e-5 | +5.25e-5 |
| cc-pvqz-ri | 242 | +2.09e-5 | −4.56e-6 | +2.35e-5 |
| def2-qzvpp-rifit | 253 | +1.56e-5 | −5.42e-6 | +1.69e-5 |
| cc-pvtz-jkfit (mis-pair) | 139 | +3.33e-4 | −4.21e-4 | +2.60e-4 |

(Bucket deltas in Hartree; SCS uses Grimme's $c_{\mathrm{os}}=6/5$,
$c_{\mathrm{ss}}=1/3$.)

`MP2Options.report_ri_residual = True` turns on an opt-in diagnostic
that, alongside the RI build, also computes the canonical (ia|jb) and
returns the per-bucket residuals on `MP2Result`:

```python
opts = MP2Options()
opts.density_fit = True
opts.aux_basis = "cc-pvtz-ri"
opts.report_ri_residual = True              # opt-in

mp2 = run_mp2(mol, basis, hf, opts)

assert mp2.ri_residual_reported             # True iff the flag fired
print(mp2.e_os_ri_residual)                 # +5.048e-05 on H2O/cc-pvtz
print(mp2.e_ss_ri_residual)                 # -2.430e-05
```

The diagnostic **doubles the cost of the RI-MP2 call** (it performs an
additional exact integral-direct canonical contraction). It is intended for verification of
aux-basis adequacy in benchmarks and unit tests, not production. The
flag is silently no-op when `density_fit = False`
(`ri_residual_reported` stays `False`).

The four SCS/SOS convenience wrappers accept the flag directly, so you
don't have to hand-build an options struct:

```python
scs = run_scs_mp2(mol, basis, hf, report_ri_residual=True)
print(scs.e_os_ri_residual, scs.e_ss_ri_residual)
```

`run_sos_mp2`, `run_scs_ump2`, and `run_sos_ump2` take the same
`report_ri_residual=False` keyword.

### When the diagnostic earns its keep

* **Aux-basis sizing for a new system class.** Run once on a small
  representative geometry; if `|e_os_ri_residual|` exceeds your
  tolerance (rule of thumb: ≤10 µHa for SCS work), bump to the next
  zeta of the matching RIfit family.
* **Catching a wrong aux family.** Pairing a JKfit aux with MP2 is
  technically valid but over-fit, the bucket residuals are typically
  10× larger than with the matching RIfit family. The diagnostic
  surfaces this immediately.
* **Cross-code parity setup.** Use it to confirm both codes are
  running RI (not canonical) before chasing sub-µHa OS/SS differences
  - if both codes report similar OS+/SS− residuals against their own
  canonicals, the comparison is well-framed.

### What the diagnostic does *not* tell you

The residuals are vs. canonical at the **same RHF reference**, not vs.
the basis-set limit. A canonical-MP2 number itself carries
basis-set-incompleteness error of ~mHa on cc-pVTZ; the RI residual is
strictly the fit-on-top-of-that. For total-energy accuracy budgets,
combine the RI residual with a basis-set-extrapolation estimate.

## Open-shell: UMP2 and its scaled variants

`run_ump2(mol, basis, uhf, UMP2Options(...))` takes a converged UHF
reference. `UMP2Result.e_aa / e_bb / e_ab` are the three spin-channel
energies (already in the natural Grimme convention).

```python
from vibeqc import run_uhf, run_scs_ump2, UHFOptions

mol = Molecule([Atom(8,[0,0,0]),Atom(1,[0,0,1.83])], multiplicity=2)
basis = BasisSet(mol, "cc-pvdz")
uhf_opts = UHFOptions(); uhf_opts.conv_tol_energy = 1e-10
uhf = run_uhf(mol, basis, uhf_opts)

scs_uhf = run_scs_ump2(mol, basis, uhf)
# scs_uhf.e_correlation == 6/5 · e_ab + 1/3 · (e_aa + e_bb)
```

The RI fit-residual diagnostic works identically for UMP2:
`UMP2Options.report_ri_residual = True` (alongside `density_fit =
True`) populates **per-channel** residuals on the result, 
`UMP2Result.e_aa_ri_residual`, `e_bb_ri_residual`,
`e_ab_ri_residual`, plus the `ri_residual_reported` flag. The αβ
channel carries the opposite-spin-like bias and the αα / ββ channels
the same-spin-like bias, mirroring the closed-shell signature
documented above. Same opt-in cost caveat: the diagnostic doubles
the runtime of the RI-UMP2 call.

```python
opts = UMP2Options()
opts.density_fit = True
opts.aux_basis = "def2-svp-rifit"
opts.report_ri_residual = True

ump2 = run_ump2(mol, basis, uhf, opts)
assert ump2.ri_residual_reported
print(ump2.e_aa_ri_residual, ump2.e_bb_ri_residual, ump2.e_ab_ri_residual)
```

## Double hybrids

vibe-qc ships four double hybrids out of the box:

| Functional | SCF mix | MP2 mix (c_os, c_ss) | Reference |
|---|---|---|---|
| **B2PLYP** | 0.53·HF + 0.47·B88, 0.73·LYP | 0.27, 0.27 | Grimme 2006 |
| **DSD-PBEP86** | 0.70·HF + 0.30·PBE, 0.43·P86(VWN5) | 0.53, 0.25 | Kozuch-Martin 2011 |
| **revDSD-PBEP86-D4** | 0.69·HF + 0.31·PBE, 0.4210·P86 | 0.5922, 0.0636 | Santra-Sylvetsky-Martin 2019 |
| **PWPB95** | 0.50·HF + 0.50·mPW(PW6), 0.731·B95 | 0.269, 0.0 | Goerigk-Grimme 2011 |

All four are dispatched via name-specific wrappers (`run_b2plyp`,
`run_dsd_pbep86`, `run_revdsd_pbep86`, `run_pwpb95`) or the generic
`run_double_hybrid(mol, basis, "name", ...)`, and return a
:class:`DoubleHybridResult` with the same shape.

For open-shell molecules, the double-hybrid dispatcher runs the
spin-pure **ROKS** SCF half and then a semicanonical **ROHF-MP2**
doubles correction. This includes the meta-GGA PWPB95 route; a direct
`run_roks(..., functional="pwpb95")` still raises because the SCF
energy alone is not the complete double-hybrid method.

### B2PLYP

A **double hybrid** combines a hybrid-DFT SCF step with a scaled MP2
correlation correction on the converged KS orbitals:

```
E_DH = E_RKS[xc_scf] + c_os · E_os + c_ss · E_ss
```

B2PLYP (Grimme, *J. Chem. Phys.* **124**, 034108 (2006)) is the first
double hybrid wired up in vibe-qc:

```
E_B2PLYP = E_RKS[0.53·HF + 0.47·B88, 0.73·LYP] + 0.27 · E_MP2_corr
```

The orchestrator does both steps in one call:

```python
from vibeqc import run_b2plyp

result = run_b2plyp(mol, basis)
print(result)
# DoubleHybridResult(functional='b2plyp', e_total=..., e_rks=..., e_mp2_corr=...)

print(result.e_total)         # the published double-hybrid energy
print(result.rks.energy)      # the hybrid SCF step alone
print(result.mp2.e_correlation)  # 0.27 · (e_os + e_ss)
```

`run_b2plyp` defaults to **DF on both steps**: RI-J + RI-K for the
hybrid SCF, RI-MP2 for the correction. Aux bases are auto-resolved
from the orbital basis name (JKfit for SCF, RIfit for MP2). Pass
`density_fit_mp2=False` for an explicit canonical-MP2 parity
validation run.

### Anatomy

| `Functional("b2plyp")` accessor | Value |
|---|---|
| `hf_exchange_fraction` | 0.53 |
| `is_hybrid` | `True` |
| `is_double_hybrid` | `True` |
| `mp2_c_os` | 0.27 |
| `mp2_c_ss` | 0.27 |
| `kind` | `XCKind.GGA` |

Non-double-hybrid functionals (LDA, GGA, hybrid-GGA: PBE0, B3LYP,
PW1PW, …) carry `is_double_hybrid = False` and `mp2_c_os = mp2_c_ss
= 0`. The same `Functional` accessors carry the dispatcher pattern
across B2PLYP, DSD-PBEP86, and PWPB95, and will extend to ωB97M(2)
once it is unblocked (see § Adding more double hybrids).

### DSD-PBEP86

The second double hybrid, Kozuch-Martin 2011 (PCCP 13, 20104), using
the final recommended D3(BJ)-fit parameter set from that paper:

```
E_DSD-PBEP86 = E_RKS[0.70·HF + 0.30·PBE, 0.43·P86(VWN5)]
             + 0.53 · E_os + 0.25 · E_ss
```

```python
from vibeqc import run_dsd_pbep86, Functional

result = run_dsd_pbep86(mol, basis)
print(result.e_total)

fn = Functional("dsd-pbep86")  # or "dsdpbep86"
print(fn.hf_exchange_fraction)   # 0.70
print(fn.mp2_c_os, fn.mp2_c_ss)  # (0.53, 0.25) - asymmetric scaling
```

!!! note "2011 vs 2013 parameter revisions, and the P86 local part"
    Two same-name traps affect cross-code comparisons:

    1. **Parameter revision.** Kozuch & Martin re-optimized every DSD
       coefficient in their 2013 follow-up (J. Comput. Chem. 34, 2327;
       for PBE-P86: 0.69 HF, 0.44 P86, c_os 0.52, c_ss 0.22, D3BJ
       s6 = 0.48). vibe-qc's `dsd-pbep86` is the **2011** set, matching
       ORCA's keyword of the same name and the 2011 D3(BJ) damping fit
       (s6 = 0.418, a2 = 5.65) that `dispersion="d3bj"` applies.
    2. **P86 local-part flavor.** "P86 correlation" can mean Perdew's
       original 1986 form on the PZ81 local correlation (libxc
       `GGA_C_P86`) or the same gradient correction on the VWN5 local
       part (libxc `GGA_C_P86VWN`). ORCA composes the VWN5 flavor;
       vibe-qc's `dsd-pbep86` follows it. The flavors differ by
       ~1.9 mHa (scaled by c_c) on H2O/def2-SVP.

    With both conventions matched, the vibe-qc SCF part reproduces a
    conventional (NORI, DEFGRID3) ORCA 6.1.1 reference to ~0.005 mHa
    on H2O/def2-SVP. If you compare against a code using the 2013
    revision or the PZ81-local P86, expect mHa-scale differences in
    both the SCF and MP2 parts. Generic vibe-qc MP2 and ORCA now share the
    ORCA 6.1 chemical-core counting convention by default; use
    `frozen_core=False` / ORCA `NoFrozenCore` on both sides for an
    all-electron comparison. Double-hybrid convenience functions retain an
    explicitly all-electron internal scaled-MP2 component as part of their
    calibrated model definition; the ORCA regression recipes therefore add
    `NoFrozenCore`. Match that composite recipe explicitly rather than
    inferring it from the generic MP2 default.

DSD-PBEP86 is the prototypical **asymmetric**-c_os/c_ss double
hybrid, c_os ≠ c_ss is what distinguishes the DSD family from
B2PLYP-style symmetric mixing. vibe-qc's `MP2Options.c_os` /
`c_ss` cover this case natively; the `Functional` resolver carries
the published coefficients.

The published DSD-PBEP86 method usually pairs with the D3(BJ)
dispersion correction; the dispatcher returns the un-dispersed XC +
MP2 total. Add D3(BJ) post-hoc via `vibeqc.compute_d3bj` when you
need the published "with-D" energy. A dedicated D4 parameter set for
DSD-PBEP86 is a separate upcoming item.

### revDSD-PBEP86

The GMTKN55-retrained revision of DSD-PBEP86 (Santra, Sylvetsky &
Martin, *J. Phys. Chem. A* **123**, 5129 (2019)). Same primitive
pieces as DSD-PBEP86, re-optimised jointly with D4 dispersion:

```
E_revDSD-PBEP86 = E_RKS[0.69·HF + 0.31·PBE, 0.4210·P86(VWN5)]
                + 0.5922 · E_os + 0.0636 · E_ss   (+ D4)
```

```python
from vibeqc import run_revdsd_pbep86, Functional

# Un-dispersed XC + MP2 total:
result = run_revdsd_pbep86(mol, basis)

# Published revDSD-PBEP86-D4 total (folds in the D4 dispersion):
result_d4 = run_revdsd_pbep86(mol, basis, dispersion="d4")
print(result_d4.e_total)

fn = Functional("revdsd-pbep86")  # or "revdsdpbep86"
print(fn.mp2_c_os, fn.mp2_c_ss)   # (0.5922, 0.0636)
```

This alias is the **D4** member specifically (the D4 damping
s6 = 0.5132, a1 = 0.44, a2 = 3.60 was fit jointly with these
coefficients). The -D3BJ member uses a *different* XC/MP2 fit and is
not reachable through this name.

!!! note "2019 vs 2021 parameters, and the P86 local part"
    The same two traps as DSD-PBEP86 apply here, with different
    numbers:

    1. **P86 local-part flavor.** revDSD-PBEP86 uses the **VWN5**-local
       P86 (libxc `GGA_C_P86VWN`), like `dsd-pbep86`. The 2019 refit was
       run in Q-Chem, but the paper's own supporting information settles
       the intended composition: its ORCA sample deck specifies
       `Exchange X_PBE` / `Correlation C_P86`, which ORCA composes on
       VWN-5. Building the recipe on PZ81-local `GGA_C_P86` instead
       costs ~1.8 mHa in the SCF part on H2O/cc-pVTZ.
    2. **Parameter revision.** vibe-qc ships the **2019** Table 4 D4 set
       (c_c = 0.4210). ORCA's `REVDSD-PBEP86-D4/2021` keyword is a later
       re-parametrisation using c_c = 0.4224, worth a further ~0.5 mHa
       in the SCF part. The two are not interchangeable, and vibe-qc
       cites the 2019 paper it implements.

    Comparing vibe-qc against an ORCA `/2021` reference on H2O/cc-pVTZ,
    the total-energy gap is ~0.40 mHa, of which the coefficient revision
    is the largest share; the rest is ORCA's RIJCOSX/RI-MP2 protocol
    (its own final-integration exchange correction on that run is
    ~0.16 mHa). Against a 2019-parameterized reference with matched
    conventions, expect the SCF part to agree to grid accuracy.

### PWPB95

The third double hybrid, Goerigk-Grimme 2011 (*J. Chem. Theory
Comput.* **7**, 291), vibe-qc's first **meta-GGA** double hybrid and
its first **spin-opposite-scaled** (SOS) one:

```
E_PWPB95 = E_RKS[0.50·HF + 0.50·mPW(PW6), 0.731·B95]
         + 0.269 · E_os    (c_ss = 0 - same-spin MP2 dropped)
```

```python
from vibeqc import run_pwpb95, Functional

result = run_pwpb95(mol, basis)
print(result.e_total)

fn = Functional("pwpb95")
print(fn.hf_exchange_fraction)   # 0.50
print(fn.mp2_c_os, fn.mp2_c_ss)  # (0.269, 0.0) - opposite-spin only
print(fn.kind)                   # XCKind.MGGA
```

Two things set PWPB95 apart from B2PLYP / DSD-PBEP86:

* **It is a meta-GGA.** The B95 correlation component is τ-dependent,
  so `Functional("pwpb95").kind` is `XCKind.MGGA` and the SCF step
  runs the τ-dependent Kohn-Sham path. The dispatcher handles that
  transparently, no caller-side change versus the GGA double
  hybrids.
* **It is spin-opposite-scaled.** The same-spin MP2 coefficient is
  zero, so the correction is `0.269 · E_os` alone. The same-spin
  energy is still computed and reported on `result.mp2.e_ss`; it just
  does not enter `e_total`.

PWPB95 also does not mix *stock* libxc components the way B2PLYP and
DSD-PBEP86 do: Goerigk-Grimme reparametrise both semilocal pieces, 
the "PW6" modification of mPW91 exchange and a re-tuned B95
correlation. vibe-qc carries those reparametrised values as libxc
external-parameter overrides inside the `Functional("pwpb95")`
resolver, so callers never see them.

The published method is **PWPB95-D3(BJ)**. Pass `dispersion="d3bj"`
(or `"d4"`) to fold the dispersion correction into `e_total`; the
default returns the un-dispersed XC + MP2 total.

### Dispersion: D3(BJ) and D4

Double hybrids inherit the dispersion question from regular hybrid
DFT: the published B2PLYP / DSD-PBEP86 totals in the literature
almost always include a **D3(BJ)** or **D4** dispersion correction
on top of the XC + MP2 piece. The dispatcher folds it in when you
ask for it:

```python
# Un-dispersed B2PLYP (the "B2PLYP / no-D" total):
no_d = run_b2plyp(mol, basis)

# Full B2PLYP-D4 (the published total - Caldeweyher-Bannwarth-Grimme 2019):
d4   = run_b2plyp(mol, basis, dispersion="d4")
print(d4.e_total)         # XC + MP2 + D4
print(d4.dispersion.energy)  # the D4 piece alone (Hartree)
```

`dispersion="d4"` reaches the reference Caldeweyher-Bannwarth-Grimme
2019 implementation via the optional ``dftd4`` Python package
(install with ``pip install -e '.[dispersion]'``). When ``dftd4`` is
not installed the dispatcher raises with an actionable hint;
``dispersion=None`` (the default) returns the un-dispersed total
and never touches the dispersion path.

For arbitrary (non-double-hybrid) functionals, call
[`vibeqc.compute_d4(mol, functional)`](../api/generated/vibeqc.compute_d4.rst)
directly and add the energy to your SCF total. Any functional name
``dftd4`` catalogs is accepted (B3LYP, PBE0, PW1PW, r²SCAN, …).

D3(BJ) is wired directly into the dispatcher too, pass
`dispersion="d3bj"` for the published `X-D3(BJ)` total (vibe-qc's own
framework + optional ``dftd3`` backend, per-functional parameters
auto-resolved from the functional name). See
:func:`vibeqc.compute_d3bj` and the
[Density fitting](density_fitting.md) page for the standalone
dispersion API.

### Adding more double hybrids

The same Functional + dispatcher pattern generalises to other
double-hybrid families. To register a new one:

1. Add an alias entry to `cpp/src/xc.cpp::resolve_alias` with the
   SCF mix (HF fraction + weighted libxc components) and the MP2
   `c_os` / `c_ss` pair.
2. Add a thin Python wrapper `run_<name>` in `python/vibeqc/__init__.py`
   that delegates to `run_double_hybrid(mol, basis, "<name>", ...)`.
3. Add a parity test against the reference code's same-recipe
   composition.

PWPB95 followed exactly this recipe, with one twist. Its semilocal
pieces are *reparametrised* (the "PW6" mPW91 exchange and a re-tuned
B95 correlation), which stock libxc components cannot express, so
step 1 also extended `AliasComponent` to carry per-component libxc
**external-parameter overrides** (`xc_func_set_ext_params_name`).
That machinery is now in place for any future reparametrised
functional.

**Still blocked**: ωB97M(2) (Mardirossian-Head-Gordon 2018) is a
range-separated meta-GGA double hybrid. The τ-density path, the
range-separated Coulomb-attenuation machinery, VV10 non-local
correlation, and even the bespoke B97M semilocal energy/potential
kernel (validated to machine precision against libxc's ωB97M-V) are
all live now; the maintainer has scoped the remaining work to a
**self-consistent** ωB97M(2) variant (optimising the functional's own
orbitals, not the published xDH form on ωB97M-V orbitals, which has no
external parity anchor to validate against). What's left is the
self-consistent SCF driver itself; see `handovers/HANDOVER_WB97M2.md`
for the full recipe. `xc="wb97m(2)"` keeps its honest gate until that
lands.

### Cross-code parity

`tests/test_b2plyp.py` and `tests/test_dsd_pbep86.py` pin vibe-qc's
dispatchers against PySCF's hand-composed equivalents (matching
`dft.RKS(xc='...')` for the SCF +
`mp.MP2(mf, frozen=0).density_fit(auxbasis=aux_basis_name).kernel()` for the
explicitly all-electron correlation correction with the matched RI auxiliary,
scaled by the published coefficients).
`tests/test_pwpb95.py` instead pins against **ORCA 6.1** `! PWPB95`:
PWPB95's reparametrised libxc components cannot be expressed in
PySCF's xc-string parser, so a hand-rolled PySCF recipe would
silently use stock mPW91 / B95, ORCA is the honest cross-code
reference here.

| Method | H₂O/cc-pVDZ gap | Tolerance | Reference |
|---|---|---|---|
| B2PLYP    | 1.93e-7 Ha | 1e-5 Ha (grid-coupling slack) | PySCF |
| DSD-PBEP86 | 1.07e-5 Ha | 5e-5 Ha (P86 grid-sensitive) | PySCF |
| PWPB95    | 1.14e-6 Ha | 5e-5 Ha (conventional, all-electron) | ORCA 6.1 |

The gap is set by the DFT-grid difference between vibe-qc and the
reference code; total energies at our own grid are reproducible to
machine precision.

For a full multi-system, multi-variant validation harness against
ORCA, see [`examples/molecular/mp2_benchmarks/`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/mp2_benchmarks/README.md)
- S22 (closed shell) + open-shell complement, same-converger
discipline, automated comparator.

## Citations

For published work that uses these methods:

* **Canonical MP2**, Møller, Plesset, *Phys. Rev.* **46**, 618
  (1934); operational form in Szabo-Ostlund, *Modern Quantum
  Chemistry* (eq. 6.71).
* **RI-MP2**, Vahtras, Almlöf, Feyereisen, *Chem. Phys. Lett.* **213**,
  514 (1993); Weigend, Häser, Patzelt, Ahlrichs, *Chem. Phys. Lett.*
  **294**, 143 (1998) (the RIfit auxiliary basis family vibe-qc uses).
* **SCS-MP2**, Grimme, *J. Chem. Phys.* **118**, 9095 (2003).
* **SOS-MP2**, Jung, Lochan, Dutoi, Head-Gordon, *J. Chem. Phys.*
  **121**, 9793 (2004).
* **B2PLYP**, Grimme, *J. Chem. Phys.* **124**, 034108 (2006).
* **DSD-PBEP86**, Kozuch, Martin, *Phys. Chem. Chem. Phys.* **13**,
  20104 (2011).
* **PWPB95**, Goerigk, Grimme, *J. Chem. Theory Comput.* **7**, 291
  (2011).
* **D4 dispersion**, Caldeweyher, Bannwarth, Grimme, *J. Chem. Phys.*
  **150**, 154122 (2019); reference implementation: Grimme group
  ``dftd4`` Python package.

## See also

* [DLPNO-MP2](dlpno_mp2.md), near-linear-scaling local MP2
  (`method="dlpno-mp2"`); tracks canonical RI-MP2 with controllable
  truncation thresholds.
* [Functionals](functionals.md), the `Functional` resolver, libxc
  composition, hybrid mixing.
* [Density fitting](density_fitting.md), RI-J / RI-K / RIJCOSX
  setup; same auxiliary-basis families MP2 consumes.
* [Molecules](molecules.md), molecular SCF (RHF / UHF / RKS / UKS),
  dispersion, analytic gradients.
* [`examples/molecular/mp2_benchmarks/`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/mp2_benchmarks/README.md)
  - vibe-qc ↔ ORCA cross-code S22 + open-shell suite.
