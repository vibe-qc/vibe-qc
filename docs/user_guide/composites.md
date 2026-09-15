# Composite "3c" methods

The Grimme-school **composite 3c methods** bundle a tuned small basis,
a modern dispersion correction (D3-BJ or D4), a geometric counterpoise
correction (gCP), and, for some recipes, a modified short-range
correction (3c-SRB) into a single user-facing keyword. The result is
that one call gets you GMTKN55 / S22 / X23 accuracy comparable to a
much-more-expensive parent functional + triple-zeta calculation at
small-basis cost.

This chapter documents the v0.9.0 composite surface: the seven
registered recipes, their current availability status, the
`vq.compute_gcp` standalone API, and the contribution pathway for
extending coverage to new bases / elements.

## Quick start

```python
import vibeqc as vq

mol = vq.Molecule.from_xyz("water_dimer.xyz")

# One call: bundles HF + MINIX basis + D3(BJ) + gCP(MINIX) + SRB
res = vq.run_job(mol, method="hf-3c", output="water_dimer_hf3c")

# Read the composite total off the output file or programmatically:
#   E_SCF + E_disp + E_gCP + E_SRB
```

Composite recipes are case-insensitive and accept the unicode-omega
spelling: `"ωB97X-3c"` and `"wb97x-3c"` resolve to the same recipe.

## Recipe catalogue

```{list-table}
:header-rows: 1
:widths: 14 28 14 12 14 18

* - Composite
  - Best for
  - Basis
  - Disp.
  - SRB?
  - Availability today
* - **HF-3c**
  - Very large systems; mean-field qualitative
  - MINIX
  - D3(BJ)
  - yes
  - **`RUNNABLE`** (end-to-end)
* - **PBEh-3c**
  - Strong noncovalent geometries; geometry pre-step
  - def2-mSVP
  - D3(BJ)
  - no
  - **`RUNNABLE`** (turnkey)
* - **B97-3c**
  - Workhorse GGA; transition metals
  - def2-mTZVP
  - D3(BJ)
  - yes
  - **`RUNNABLE`** (turnkey)
* - **B3LYP-3c**
  - IR / vibrational frequencies (B3LYP's strength)
  - def2-mSVP
  - D3(BJ)
  - no
  - **`RUNNABLE`** (turnkey)
* - **r²SCAN-3c**
  - "Swiss army knife"; best general-purpose
  - def2-mTZVPP
  - D4
  - no
  - **`RUNNABLE`** (first turnkey 3c)
* - **ωB97X-3c**
  - Drug-like / pharma; barrier heights
  - vDZP
  - D4
  - no
  - **`RUNNABLE`** (inline vDZP ECPs)
* - **HSE-3c**
  - Solid-state range-separated; embedded clusters
  - def2-mSVP
  - D3(BJ,ATM)
  - no
  - **`RUNNABLE`** (turnkey)
```

:::{note}
**HSE-3c follows the paper, not Grimme's `gcp` program.** PBEh-3c and
HSE-3c share the def2-mSVP basis but not their empirical constants:
HSE-3c has its own D3(BJ) damping *and* its own gCP fit, both from
Table S1 of the ESI to
[Brandenburg, Caldeweyher & Grimme (2016)](https://doi.org/10.1039/c6cp01697a).

For gCP the two published sources disagree. That paper's Table S1 gives
`eta = 1.40858`, while `gcp.f90` (`case ('hse3c')`) in every release of
Grimme's standalone `gcp` program gives `eta = 1.32378`. vibe-qc uses the
paper's value, because only it reproduces the paper's own worked example:
the ESI publishes `E_gCP = 0.0148400663` a.u. for gas-phase benzene, which
vibe-qc reproduces to 2e-10 Ha with Table S1's constants and misses by
1.1e-4 Ha with `gcp.f90`'s. This matches CRYSTAL, which has a self-contained
internal gCP and produced that published number. It means an `hse-3c` gCP
energy from vibe-qc differs from one computed by a code that calls Grimme's
`gcp` binary (such as ORCA) by roughly 0.07 kcal/mol on benzene. The
discrepancy is acknowledged upstream in
[grimme-lab/gcp issue #17](https://github.com/grimme-lab/gcp/issues/17),
where the stated intent is to correct the code to the paper.
:::

### The three-body dispersion term

PBEh-3c and HSE-3c define their D3 energy to include the
Axilrod-Teller-Muto (ATM) three-body term, so their recipes carry
`s9 = 1.0`:

> "To have a consistent description of geometry and energy, the
> three-body dispersion is always included in the PBEh-3c method."
> --- Grimme, Brandenburg, Bannwarth & Hansen, *J. Chem. Phys.* **143**,
> 054107 (2015)

The term is repulsive and grows with density. On an isolated benzene it
shifts the D3 energy by only about +0.0094 mHa, but the same paper
reports it contributing 4--7% of a molecular-crystal lattice energy ---
which is exactly the regime these composites target.

The other 3c recipes (HF-3c, B97-3c, B3LYP-3c) are two-body only
(`s9 = 0.0`), as is every plain `dispersion="d3bj"` calculation: the
published per-functional `(s8, a1, a2)` damping sets were fit without a
three-body term. See {doc}`../tutorial/dispersion` for how to switch it
on by hand.

r2SCAN-3c uses D4 instead of D3. Its composite fit sets `s9 = 2.0` and
also changes D4's charge-scaling parameters from the general
`(beta, gamma) = (3, 2)` model to `(2, 1)`. Both D4 backends select this
charge model only for `r2scan-3c`; an ordinary r2SCAN-D4 calculation keeps
its separately fitted damping constants and the general charge model.

## Availability, what runs today

The v0.9.0 release lands the **composite scaffolding** (gCP standalone
API, recipe registry, keyword-shortcut dispatcher, SRB pairwise
correction). End-to-end runnability of a given composite depends on
its prerequisites:

`RUNNABLE`
:  Every ingredient is wired today. `vq.run_job(method="...")` works
   end-to-end.

`NEEDS_BASIS`
:  The functional + dispersion + gCP framework are wired, but the
   target basis is not bundled in `python/vibeqc/basis_library/basis/`
   even after the v0.9 basis-library expansion. You can still run the
   composite by supplying the basis-set definition locally and pointing
   `BasisSet(molecule, "/path/to/your_basis.g94")` at it.
   The per-element gCP data tables (`e_mis`, `n_virt`, `zeta`) for the
   composite-specific bases are also pending, see [Extending the gCP
   coverage](#extending-the-gcp-coverage) below.

`PENDING_F1`
:  *No composite carries this flag any more, F1 landed in v0.9.0.*
   The CAM K-build (split into K_full + K_LR using libint's
   `Operator::erf_coulomb`) is wired through `FourIndexJKBuilder` in
   v0.9.0; RSH SCF works for systems up to ~250 basis functions. The
   `DirectJKBuilder` / `DFJKBuilder` / `COSXJKBuilder` long-range
   K-build paths are queued as v0.9.x patches.
   Until those land, RSH SCF auto-selects `SCFMode::CONVENTIONAL` to
   route through FourIndex; if a user explicitly forces a non-FourIndex
   builder with an RSH functional, the first K-build call raises with
   the actionable "not yet wired for <builder>" message.

`PENDING_F3`
:  *No composite carries this flag any more, F3 landed in v0.9.0.*
   The slot is retained for documentation completeness. The mGGA
   τ-density pipeline (τ assembly, v_τ Fock contribution) is wired
   in `cpp/src/rks.cpp` and `cpp/src/uks.cpp`; the τ-only mGGA
   aliases (TPSS, r²SCAN, SCAN, M06-L) are registered in
   `cpp/src/xc.cpp`. r²SCAN-3c is now `NEEDS_BASIS`. mGGA *analytic
   gradient*, *Hessian*, and *periodic-SCF* paths are explicitly
   gated for the v0.9.x patches with actionable errors at the call
   site (use `vibeqc.gradient_fd(...)` for the mGGA gradient today).

`PENDING_ECP`
:  The recipe's basis carries effective-core potentials whose
   per-element ncores match no bundled libecpint XML library, and no
   inline primitive feed is available for that recipe yet. Running the
   basis all-electron would return a meaningless energy, so the
   dispatcher raises `CompositeUnavailable` rather than a wrong total.
   **No bundled composite currently carries this flag:** **ωB97X-3c**
   now uses the vDZP sidecar ECPs through the molecular inline primitive
   path.

`PENDING_GCP_DATA`
:  *No composite carries this flag any more, the per-element gCP
   tables (`e_mis`, `n_virt`) for the composite bases (MINIX,
   def2-mSVP, def2-mTZVPP) are bundled and verified against the
   `grimme-lab/gcp` Fortran reference (H-Kr).* The slot is retained
   for documentation completeness; see [Extending the gCP
   coverage](#extending-the-gcp-coverage) for adding new bases.

You can introspect this at runtime:

```python
for r in vq.list_composites():
    print(f"{r.name}  ->  {r.availability.value}  ({r.notes})")
```

## The gCP standalone API

The geometric counterpoise correction is exposed as a standalone
function, independent of the composite-recipe dispatcher:

```python
import vibeqc as vq

mol = vq.Molecule.from_xyz("water_dimer.xyz")
res = vq.compute_gcp(mol, "def2-svp")
print(f"E_gCP = {res.energy:.6f} Ha = {res.energy * 627.5:.4f} kcal/mol")
```

Returns a `GCPResult` with `.energy` (Hartree), `.basis_name`, the
`.params` actually used (so SCF logs can record provenance), and an
optional `.gradient` when `with_gradient=True` is passed.

### Behind the scenes, Kruse-Grimme 2012 form

```{math}
E_\mathrm{gCP} = \sigma \sum_a \sum_{b \neq a}
    \frac{ e^{-\alpha R_{ab}^\beta} \, e_a^\mathrm{MIS} }
         { \sqrt{ N_b^\mathrm{virt} \, S_{ab} } }
```

where `σ, α, β` are basis-set-specific fit parameters
(Kruse-Grimme 2012 Table 2), `e_a^MIS` is the per-atom "missing
basis-set incompleteness" energy (Hartree, always negative; per-
element tabulation in Table 3 / SI of the paper), `N_b^virt` is the
number of virtual orbitals of atom b in the target basis, and `S_ab`
is the geometric-mean 1s-STO overlap of atoms a and b. The sign of
E_gCP is **always positive**, gCP opposes the BSSE over-binding of
incomplete basis sets.

For full details and the per-basis citation chain, see
[`vibeqc.compute_gcp`](../api/generated/vibeqc.compute_gcp.rst).

## Extending the gCP coverage

The v0.9.0 first cut bundles per-element data for H / C / N / O / F at
def2-SVP and def2-TZVP. To extend coverage, follow the workflow in
[the data_library standard](data_library.md):

1. **For a new element at a bundled basis**, add a new
   `[elements.<symbol>]` table to the matching
   [`python/vibeqc/data_library/gcp/<basis>.toml`](https://github.com/vibe-qc/vibe-qc/tree/main/python/vibeqc/data_library/gcp)
   file. Populate `e_mis` / `n_virt` / `zeta` from the cited source
   (the file's `metadata.citation`). Add a regression test in
   [`tests/test_gcp.py`](https://github.com/vibe-qc/vibe-qc/blob/main/tests/test_gcp.py)
   that hits the new element.

2. **For a new basis**, copy
   [`data_library/gcp/_template.toml`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/data_library/gcp/_template.toml)
   to a new `<name>.toml`, fill in the `[metadata]` block (citation,
   DOI, license posture), transcribe the `(σ, α, β)` constants and
   per-element data, then expose the new keyword under a recipe in
   [`python/vibeqc/composites.py`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/composites.py)
   if it's part of a composite.

3. **For a new method on an already-bundled basis**, add a
   `[variants.<method>]` block rather than a new file. The four fit
   constants `(σ, η, α, β)` are per `(method, basis)` pair, not per
   basis: `gcp.f90` selects them on the method while the per-element
   `e_mis` / `n_virt` tables are selected on the basis. Three composites
   share def2-mSVP, and they do **not** share a gCP fit. Set
   `gcp_variant=` on the recipe to select the block; leaving it `None`
   uses the file's basis-level `[parameters]`. A `gcp_variant` naming a
   block that does not exist raises rather than falling back, so a typo
   cannot silently substitute another composite's constants.

   ```python
   >>> vq.available_gcp_variants("def2-msvp")
   ['hse-3c']
   >>> vq.gcp_params_for("def2-msvp", variant="hse-3c").eta
   1.40858
   >>> vq.gcp_params_for("def2-msvp").eta          # pbeh-3c's fit
   1.32492
   ```

4. **For a one-off run with un-bundled data**, pass an explicit
   `params=` to `vq.compute_gcp`:

   ```python
   custom = vq.GCPParams(
       basis_name="my-basis",
       sigma=0.3845, alpha=0.7400, beta=1.4366,
       e_mis={6: 0.0273, 7: 0.0457, ...},
       n_virt={6: 9.0, 7: 9.0, ...},
       zeta={6: 5.67, 7: 6.66, ...},
       citation="my reference",
   )
   res = vq.compute_gcp(mol, params=custom)
   ```

## Citations

When you publish results from a 3c composite, cite **all four** of:

1. **vibe-qc** per `CITATION.cff` + the JCC release paper once
   published.
2. **The composite-method paper**:

   * HF-3c: Sure & Grimme, *J. Comput. Chem.* **34**, 1672 (2013).
   * PBEh-3c: Grimme, Brandenburg, Bannwarth, Hansen, *J. Chem. Phys.*
     **143**, 054107 (2015).
   * B97-3c: Brandenburg, Bannwarth, Hansen, Grimme, *J. Chem. Phys.*
     **148**, 064104 (2018).
   * r²SCAN-3c: Grimme, Hansen, Ehlert, Mewes, *J. Chem. Phys.*
     **154**, 064103 (2021).
   * ωB97X-3c: Müller, Hansen, Grimme, *J. Chem. Phys.* **158**,
     014103 (2023).

3. **gCP**: Kruse & Grimme, *J. Chem. Phys.* **136**, 154101 (2012).
4. **Dispersion**:

   * D3-BJ: Grimme, Ehrlich, Goerigk, *J. Comput. Chem.* **32**, 1456
     (2011).
   * D4: Caldeweyher, Bannwarth, Grimme, *J. Chem. Phys.* **150**,
     154122 (2019).

The recipe's `.citation` field surfaces the canonical originator
paper in the SCF log, and `compute_gcp` records the parameter-source
publication in `GCPResult.params.citation`.
