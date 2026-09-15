# Experimental features

vibe-qc ships a handful of features behind an **experimental gate**. They
are reachable, you can turn one on and get a result, but they are not yet
production-certified: a feature may be incomplete, may change or break
between releases without the usual deprecation cycle, or is known to be
inaccurate in some regime. Each one emits an `ExperimentalWarning` (or a
feature-specific subclass) every time you use it, so an experimental result
never slips into a script unnoticed.

This page catalogs warning-gated experimental features. Methods labeled for
screening or validated prescreening on their own status pages are not thereby
external-parity production methods merely because they do not emit a warning.

## What "experimental" means here

There are two reasons a feature carries the gate, and the difference matters
for whether you can trust its numbers:

- **Production-quality in its supported regime, warned because the broader
  feature is still being built.** GPW and GAPW are the examples: the shipped
  paths reproduce their references, but the surrounding capability (the
  all-electron augmentation default, open-shell multi-k, and so on) is
  incomplete, so the whole route warns. Safe to use within the documented
  regime.
- **Not yet accurate enough for published numbers.** GFN2-xTB and OM1 are in
  this group: they run, but a known defect or a missing term makes them
  development and qualitative only. The native D4
  backend is narrower: it is quantitative for its documented H, He, B, C, N, O,
  F, Ne scope and fails closed outside it. The "Quantitative" column below
  flags these.

Roadmap features are different: those raise `NotImplementedError` and are
not reachable at all (see [the roadmap](../roadmap.md)). Experimental features
*are* reachable; they just warn.

The BIPOLE quartet multipole far-field belongs with unavailable features, not
with the reachable experimental catalog. It is unavailable from every SCF
driver. The supported exact Ewald-J route defaults to
`use_multipole_far_field=False`, and explicit `True` raises before setup.

## Catalog

In this table, Γ-CCM and χ-CCM[^xccm-convention] are distinct
union-and-weight and finite-translation-group character approaches.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM construction approaches:
    union-and-weight/Wigner-Seitz integral weighting and finite-translation-group
    characters, respectively. A declared common exchange-q=0 convention is a
    comparison constraint, not an identification of the constructions. Their
    distinction is not a choice of Coulomb kernels, and equality for a specified
    operator and route must be established rather than inferred from the labels.

| Feature | Area | Opt in | Status and key caveat | Quantitative |
|---|---|---|---|---|
| **GPW** (Gaussian plane waves) | periodic | `run_periodic_job(jk_method="gpw")` | Gamma RHF/maintained-preview ROHF/ROKS/RKS/UHF/UKS and multi-k pure-DFT RKS/ROKS/UKS ship; ROHF/ROKS use one integer-occupation restricted orbital set and reject gradients, smearing, and unsupported envelopes; multi-k ROKS hybrids remain gated; emits `GAPWExperimentalWarning` | Yes, in the supported regime |
| **GAPW** (all-electron augmentation) | periodic | `run_periodic_job(jk_method="gapw")` (`gapw_molecular_limit=True` required for RHF/UHF) | not yet the JK AUTO default; declared molecular-limit Gamma RHF/UHF uses fit-free analytic one-centre augmentation, while Gamma RKS/UKS and multi-k pure-DFT RKS use the block construction; multi-k HF/hybrids and compact-crystal augmentation validation remain gated; converge atom and plane-wave grids for absolute totals | Partial |
| **BIPOLE multi-k KS analytic gradient** | periodic | `relax_atoms(..., force_mode="analytic")` or direct `compute_bipole_gradient_*` calls | maintained preview; the corrected gauge is finite-difference-validated, broader KS certification is pending. High-level optimization defaults to the exact finite-difference force path | Partial |
| **MPI GPW grid overlay** | periodic | `GpwJBuilder(mpi_aware=True)` (not via `run_periodic_job`) | distributed-memory Hartree-J; the multi-rank build was fixed after v0.12 and is pending validation on a real multi-rank run | No |
| **Surface embedding** (Green's function) | periodic | the embedded-surface calculator, see [surface embedding](../user_guide/surface_embedding.md) | emits an experimental warning; the 3D Gaussian foundation is validated, thick-slab GDF parity is pending, so absolute energies and densities are qualitative | No |
| **Γ-CCM / aiccm2026dev-a** | periodic | `run_periodic_job(method="aiccm", variant="four-center")`; library `vibeqc.periodic.ccm`, `method="aiccm2026dev-a"` | union-and-weight/Wigner-Seitz integral-weighting CCM development line with HF, KS, MP2, and CCSD(T); the runner arm supports 3-D RHF/RKS/UHF/UKS and every SCF driver emits `AICCM2026DevAExperimentalWarning`. The `Γ-CCM` name is for prose; the library selector stays `method="aiccm2026dev-a"` and the runner selector is `variant="four-center"`. See its [separate page](../aiccm2026dev_a.md) and the [AICCM page](../user_guide/aiccm.md) | Partial |
| **χ-CCM / aiccm2026dev-b** | periodic | `run_periodic_job(method="aiccm", variant="chi")` (legacy `jk_method="aiccm2026dev-b"` warns); restricted/unrestricted χ-CCM APIs | finite-character (Γ-centred character-mesh) CCM with independent finite-torus RHF/RKS/UHF/UKS in 3D via four-center, RI, and RIJCOSX; all 1D/2D absolute-energy routes fail closed pending a shared wire/slab Coulomb convention; restricted and unrestricted real-torus correlation in 3D; the real-space lattice extension is primary and the character net is derived; occupied [localization](../tutorial/aiccm2026dev_b_localization.md) and [space-group diagnostics](aiccm2026dev_b_symmetry.md) are χ-CCM-only; symmetry integral reduction, mixed-boundary Green functions, minimum-image local screens, and production reduced-scaling CC remain open; see its [separate page](../user_guide/aiccm2026dev_b.md) | Partial |
| **GFN2-xTB** | semiempirical | `run_job(method="gfn2_xtb")` | not quantitative; external parity and periodic AES/molecular-limit terms are production gates. Post-SCF native D4 is validated only for H, He, B, C, N, O, F, Ne; outside that set GFN2 returns zero D4 with `GFN2D4UnsupportedWarning` | No |
| **OM1** | semiempirical | `run_job(method="om1")` | the analytic core-valence ECP is not implemented, so bonds to heavy atoms come out roughly 0.3 angstrom short; OM2/OM3 do not share this missing-ECP warning but remain validated prescreening routes with open external parity | No |
| **D4 dispersion, native backend** | dispersion | `compute_d4(..., backend="native")` | parity-validated for H, He, B, C, N, O, F, Ne; raises outside that catalogue. The default `backend="dftd4"` (optional package) remains the full-periodic-table route | Yes, in the supported regime |

The periodic GPW/GAPW family is documented in full in [GPW and GAPW](../user_guide/gapw.md)
and the [GAPW design note](../design_periodic_gapw.md). The native-D4-backend
status, the BIPOLE gradient preview, and the GFN2-xTB / OM1 limits are in
[troubleshooting](../troubleshooting.md); the semiempirical models also have a
[user-guide page](../user_guide/semiempirical.md).

The χ-CCM research interfaces have separate pages for
[occupied localization](../tutorial/aiccm2026dev_b_localization.md),
[PAO/PNO local correlation](aiccm2026dev_b_pno.md), and
[space-group diagnostics](aiccm2026dev_b_symmetry.md).

The full AICCM documentation suite covers both lines:
[quickstart](../tutorial/aiccm_quickstart.md),
[troubleshooting](aiccm2026dev_troubleshooting.md),
[basis sets](aiccm2026dev_basis.md),
[cross-stream comparison](aiccm2026dev_compare.md),
[open-shell](aiccm2026dev_open_shell.md),
[visualization](aiccm2026dev_viz.md), and the
[documentation map](index.md#aiccm-documentation-map).

## Keeping competing versions on purpose

Some experimental features exist in more than one version on purpose, because
the right approach is itself an open research question and we want to measure
which one wins before committing to it. These are not duplicates to be merged
away; all are maintained until the comparison is settled.

The active example is the **ab-initio CCM (AICCM)**. Three constructions are
kept visible while two development streams remain independent:

- **`method="union12"`** (default), the historical product weight from
  Peintinger and Bredow 2014 (eq. 18). Validated in 1-D, but it breaks the
  8-fold permutational symmetry of the electron-repulsion integrals in two and
  three dimensions.
- **`method="aiccm2026dev-a"`**, Γ-CCM, the union-and-weight/Wigner-Seitz
  integral-weighting construction inside `vibeqc.periodic.ccm`.
- **`jk_method="aiccm2026dev-b"`**, χ-CCM, the finite-character
  (Γ-centred character-mesh) CCM evaluation in
  `vibeqc.periodic.chi`. It
  periodizes the declared finite-torus Hamiltonian first, records
  `coulomb_kernel="3d-periodic-g0"` and `exchange_q0="bvk-ewald"`, and uses
  Wigner--Seitz ties only to choose equivalent representatives; it does not
  import the Γ-CCM core.

Archived pre-guard 1D B tests are defect evidence only; current χ-CCM-B
absolute energies fail closed in every 1D/2D backend. Γ-CCM and χ-CCM are
developed independently on
purpose: if they converge to the same answer under a matched Coulomb gauge,
that is evidence for the common limit; if they diverge, the tensor symmetries
and scalar energy derivatives decide which construction is admissible. The
duplication is the experiment, not an oversight.

The fleet study uses one shared geometry registry but separate executable
drivers. Its current cross-approach status is `not-defined`. See the
[χ-CCM fleet inputs](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/README_B.md) for
the complete RHF/RKS backend matrix and the 3D post-HF inputs.

## Internal and research-only paths (not for users)

A few experimental code paths are research scaffolding rather than
user-facing features. They have no `run_job` / `run_periodic_job` entry, or
are superseded, so they live in the roadmap rather than here: the GAPW
orbital-transformation (OT) direct-minimisation solver, the landed-but-unwired
Anderson density mixer, the BIPOLE legacy-gauge diagonal-Z gradient preview,
and omega-B97M(2) (the functional exists in libxc but has no user alias yet).

## Silencing an experimental warning

The warnings exist so an experimental path never runs unnoticed. Once you
have read the caveat and accept it, silence it the standard Python way by
filtering on the feature's warning class (for example `GAPWExperimentalWarning`,
importable from the module that raises it):

```python
import warnings
warnings.filterwarnings("ignore", category=GAPWExperimentalWarning)
```

Some constructors also take a `warn=False` argument, for example
`GFN2Model(mol, warn=False)`.

## Adding to this page

This is the designated home for experimental-feature status. When you ship a
feature behind an `ExperimentalWarning` (or a subclass), add a row to the
catalog with its opt-in and its one key caveat, and link to its detailed
page. A feature that warns at runtime but is invisible here is exactly the
gap this page exists to close.

```{toctree}
:hidden:

aiccm2026dev_b_pno
aiccm2026dev_b_symmetry
aiccm2026dev_diamond_compare
aiccm2026dev_open_shell
aiccm2026dev_compare
aiccm2026dev_basis
aiccm2026dev_viz
aiccm2026dev_troubleshooting
```
