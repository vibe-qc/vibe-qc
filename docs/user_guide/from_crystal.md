---
myst:
  html_meta:
    "description": "Translation guide for users coming from CRYSTAL14 / CRYSTAL23 to vibe-qc. Maps SHRINK / TOLINTEG / SCFDIR / FMIXING / LEVSHIFT / BIPOLE / ATOMSPIN to the corresponding vibe-qc API."
    "og:title": "CRYSTAL → vibe-qc translation guide"
    "og:description": "A keyword-by-keyword crosswalk for CRYSTAL users moving to vibe-qc."
---

# Moving from CRYSTAL

vibe-qc's roadmap targets feature-for-feature parity with
[CRYSTAL](https://www.crystal.unito.it/) on the v1.0 / v2.0 timeline,
the Peintinger-Vilela Oliveira-Bredow pob-* basis-set family is
bundled, and the periodic SCF gauge convention is being aligned
with CRYSTAL's BIPOLE construction (see
[user_guide/bipole](bipole.md)). For CRYSTAL users, the *physics*
maps directly; the **input idiom** is different, CRYSTAL takes a
fixed-grammar `.d12` deck while vibe-qc takes a Python script. This
page is a keyword-by-keyword crosswalk.

## Input file shape

### CRYSTAL `.d12`

```text
TiO2 rutile
CRYSTAL
0 0 0
136
4.5937  2.9587
2
22  0.0     0.0     0.0
8   0.30530 0.30530 0.0
END
BASISSET
POB-TZVP
END
DFT
EXCHANGE
PBE
CORRELAT
PBE
END
SHRINK
8 8
TOLINTEG
7 7 7 7 14
FMIXING
30
LEVSHIFT
2 1
END
```

### vibe-qc Python

```python
import numpy as np
import vibeqc as vq

# Rutile lattice - fetched from a CIF / Materials Project / hand-built.
a, c = 4.5937 * 1.8897259886, 2.9587 * 1.8897259886    # → bohr

sysp = vq.PeriodicSystem(
    dim=3,
    lattice=np.diag([a, a, c]),         # tetragonal P4_2/mnm
    unit_cell=[
        vq.Atom(22, [0.00000 * a, 0.00000 * a, 0.0 * c]),     # Ti
        vq.Atom(22, [0.50000 * a, 0.50000 * a, 0.5 * c]),
        vq.Atom( 8, [0.30530 * a, 0.30530 * a, 0.0 * c]),     # O
        vq.Atom( 8, [0.69470 * a, 0.69470 * a, 0.0 * c]),
        vq.Atom( 8, [0.80530 * a, 0.19470 * a, 0.5 * c]),
        vq.Atom( 8, [0.19470 * a, 0.80530 * a, 0.5 * c]),
    ],
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
kmesh = vq.monkhorst_pack(sysp, [8, 8, 8])              # SHRINK 8 8

opts = vq.PeriodicRKSOptions()
opts.functional   = "PBE"
opts.lattice_opts.cutoff_bohr = 12.0                    # ≈ TOLINTEG row 5 control
opts.fock_mixing  = 0.30                                # FMIXING 30
opts.level_shift  = 0.2                                 # LEVSHIFT 2 1; vibe-qc uses Ha
                                                        # → 0.2 Ha ≈ 5 eV ≈ "2"
                                                        # in CRYSTAL's units (×10⁻¹ Ha)

result = vq.run_rks_periodic(sysp, basis, kmesh, opts)
print(result.energy)
```

The structural difference is that vibe-qc's input is **executable
Python** rather than declarative text. That trades CRYSTAL's
grammar guard-rails for the full flexibility of a programming
language, you can sweep over functionals in a `for` loop, generate
geometries from a CIF, splice in custom basis sets, etc., without
input-file pre-processing.

For high-volume CRYSTAL ports the
[`vq submit -d <dir>`](queue.md#your-first-calculation) directory-submit
pattern lets you keep one Python "driver" plus several geometry
files alongside it, mimicking CRYSTAL's `.d12` + `INPUT` layout.

## Keyword crosswalk

```{important}
Many CRYSTAL keywords accept integer values that map to scaled
SI / atomic units (the `LEVSHIFT 2 1` means $0.2$ Ha applied for
1 iteration). vibe-qc uses **Hartree directly** for every
energy-like field. Multiplying CRYSTAL's integer by $10^{-1}$
generally lands the same number.
```

### Geometry + symmetry

| CRYSTAL | vibe-qc |
|---|---|
| `CRYSTAL` (3D), `SLAB` (2D), `POLYMER` (1D), `MOLECULE` (0D) | `PeriodicSystem(dim=3 / 2 / 1)` or just `Molecule(...)` |
| Space-group number + lattice + asymmetric unit | Full lattice matrix + full unit-cell atom list (vibe-qc reconstructs symmetry via spglib) |
| `KEEPSYMM` | (default, vibe-qc keeps the input cell) |
| `EXTPRT` (print extended geometry) | Manifested as `.xyz` / `.POSCAR` / `.xsf` siblings by `run_periodic_job`, see [output_files](output_files.md). |
| `SUPERCEL` | Build the supercell in Python before calling `PeriodicSystem(...)` (e.g. via ASE `make_supercell`). |
| `ATOMSPIN` (initial spin per atom) | The UHF / UKS drivers carry `multiplicity` on the `Molecule`; per-atom spin density isn't a CRYSTAL-only concept and lands via the SAD initial guess. |
| `OPTGEOM` (geometry optimisation) | Route-preserving fixed-cell optimization: BIPOLE uses production finite-difference forces, while supported closed-shell GDF envelopes use analytic gradients, including explicit slab GDF. Variable-cell optimization fails closed. |

```{note}
**Lattice direction matters.** CRYSTAL stores lattice vectors
as rows; vibe-qc as columns. The conversion is one transpose:
`lattice_vibeqc = lattice_crystal.T`. Both are in the same units
once you convert Å → bohr.
```

### Basis sets

| CRYSTAL | vibe-qc |
|---|---|
| `BASISSET / POB-TZVP / END` block | `BasisSet(mol, "pob-tzvp")`, same name |
| Per-element CRYSTAL-format basis lines | Drop the file under `basis_library/custom/` and re-run `./scripts/setup_basis_library.sh`. The CRYSTAL-format parser (`vibeqc.basis_crystal`) handles the per-element layout. |
| Mixed BSE + custom basis | Same pattern, drop in `custom/`, rebuild. See [basis_sets](basis_sets.md) § "Custom basis sets". |
| `INPUT` block with explicit primitive coefficients | Same `custom/` mechanism. |

The bundled pob-* family on `main` covers H-Br (the same coverage
as the original Peintinger 2013 paper). Current installs ship the
expanded packaged library as well: 255 Gaussian `.g94` runtime files,
combining the libint-inherited standard files, pob-* bases, and the
BSE-derived overlay files documented in [basis_sets](basis_sets.md).

### ECPs

| CRYSTAL | vibe-qc |
|---|---|
| `ECP` keyword inside the basis block | `opts.ecp_centers = [vq.ECPCenter(Z=..., xyz=...)]` + `opts.ecp_library = "lanl2dz"` or `"ecp46mdf"` etc. |
| `HAYWLC` / `HAYWSC` | `ecp_library="lanl2dz"` |
| Stuttgart MDF/MWB ECPs from external libraries | One of the bundled `ecp{10,28,46,60,78}mdf` libraries, picked by element row (see [ecp](ecp.md) § Library-selection table). |

See [Pt cluster with LANL2DZ: heavy-element ECPs](../tutorial/pt_cluster_lanl2dz.md) for the
ECPCenter recipe in detail.

### k-mesh

| CRYSTAL | vibe-qc |
|---|---|
| `SHRINK n n` | `vq.monkhorst_pack(sysp, [n, n, n])` |
| `SHRINK IS ISP` (MP net `IS` + denser Gilat net `ISP` for the Fermi energy / DOS) | The `IS` net is `vq.monkhorst_pack(sysp, [IS]*3)`. The `ISP` Gilat net, parameter-free Fermi-surface / DOS integration, is the `bz_integration="gilat"` backend (note below). |
| `KNETOUT` (k-net diagnostic) | The result object carries `result.kmesh.kpoints_cart` and `result.kmesh.weights`. |
| `GAMMA` (Γ-only) | `vq.monkhorst_pack(sysp, [1, 1, 1])` or use the Γ-only drivers `run_*_periodic_gamma*`. |

```{note}
**Gilat-Raubenheimer net (`bz_integration="gilat"`)** is the integrator behind
CRYSTAL's `SHRINK` second (`ISP`) net, implemented in `vibeqc.bz_integration`.
Pass `bz_integration="gilat"` to the multi-k drivers
(`run_{rhf,rks,uhf,uks}_periodic_multi_k_ewald3d`) for the parameter-free
Gilat-Raubenheimer cell quadrature (no smearing width to converge). It is
self-consistent for **insulators** (full or symmetry-reduced meshes), and
parameter-free **post-SCF** for **metals**: converge with
`smearing_temperature`, then take the GR Fermi level / occupations / `gilat_dos`
on the converged eigenvalues. Self-consistent GR-driven *metals* await the
periodic density-mixing program (Kerker / Anderson / Broyden, roadmap v0.10.x);
until then a metal total energy comes from smearing.
```

### Coulomb / exchange truncation (TOLINTEG)

CRYSTAL's `TOLINTEG T1 T2 T3 T4 T5` controls five different screening
tolerances. There is no one-to-one map onto the supported exact BIPOLE route:
its quartet penetration dispatcher is fail-closed. Converge the vibe-qc
real-space support and Schwarz screening independently.

| CRYSTAL TOLINTEG row | vibe-qc analogue |
|---|---|
| `T1` (Coulomb overlap, bipolar) | No direct analogue on the exact route; converge `cutoff_bohr` and the fold diagnostic |
| `T2` (Coulomb penetration) | No active analogue; quartet bipolar dispatch is unavailable |
| `T3` (exchange overlap) | Converge the direct-ERI overlap/Schwarz thresholds independently |
| `T4` (exchange pseudo-overlap) | No one-to-one analogue |
| `T5` (pseudopotential) | ECP-bearing BIPOLE jobs are currently unsupported |

Use the overlap-fold preflight and cutoff ladders to establish real-space
support, then tighten the Schwarz thresholds as a separate convergence test. See
[linear_dependence](linear_dependence.md) for the diagnostic stack
when they need tightening.

### SCF convergence, direct keyword map

| CRYSTAL | vibe-qc |
|---|---|
| `MAXCYCLE n` | `opts.max_iter = n` |
| `TOLDEE n` (energy threshold = $10^{-n}$ Ha) | `opts.conv_tol_energy = 10**(-n)` |
| `TOLDEG n` (gradient threshold) | `opts.conv_tol_grad = 10**(-n)` |
| `FMIXING p` (Fock-matrix mixing, p in %) | `opts.fock_mixing = p / 100`, same Pulay mixing |
| `LEVSHIFT n 1` (level shift = `n * 0.1` Ha for one iter) | `opts.level_shift = n / 10` (in Hartree) |
| `LEVSHIFT n 0` (persistent shift, never removed) | `opts.level_shift = n / 10; opts.level_shift_warmup_cycles = 0` |
| `BROYDEN` | Not implemented in vibe-qc; vibe-qc uses Pulay DIIS / EDIIS by default. |
| `ANDERSON` | Not implemented, same comment. |
| `SMEAR width` | `opts.smearing_temperature = width` (Ha; named presets `"metal"`, `"small-gap"`, `"auto"` are also accepted). See [Fermi-Dirac smearing for metals and small-gap solids](../tutorial/fermi_dirac_smearing.md). |
| `DIIS` | Default on (`opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS` in v0.8.0; `DIIS` available as the alternative). See [Stiff molecular SCF: rescuing oscillation with EDIIS+DIIS](../tutorial/ediis_diis_hybrid.md). |
| `SPINLOCK n nstep` (lock the spin) | `opts.spinlock_mode = vq.SpinlockMode.SPIN_SCHEDULE`, `opts.spinlock_value = n`, `opts.spinlock_iterations = nstep` (molecular UHF / UKS). For a per-atom broken-symmetry seed, set `opts.atomic_spins` (the CRYSTAL `ATOMSPIN` analogue). See [initial_guess](initial_guess.md). |
| `GUESSP` (read previous density) | `opts.initial_guess = vq.InitialGuess.READ` with `read_from=` a prior result, or `opts.read_path = "prev.qvf"` / `"prev.molden"`. See [initial_guess](initial_guess.md). |

### Coulomb / Exchange engine

| CRYSTAL | vibe-qc |
|---|---|
| `BIPOLE` (default periodic Gaussian Coulomb route) | [`run_periodic_job(..., jk_method="bipole")`](bipole.md) or the direct `run_pbc_bipole_*` drivers, production exact Ewald-J energy route for RHF/UHF/RKS/UKS at Γ and multi-k; production forces use finite differences. The quartet multipole far-pair replacement is unavailable, and explicit `use_multipole_far_field=True` raises before setup. |
| `BIPOLAR n` (bipolar expansion order) | Not exposed. `multipole_l_max` is inactive on the supported exact route because the quartet multipole replacement fails closed. |
| Implicit Ewald-V_ne / Ewald-E_nn | Default for `run_rhf_periodic_*` and `run_rks_periodic_*` (`opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D`). |
| `OPTGEOM` (periodic gradient) | `run_periodic_job(..., jk_method="bipole", optimize=True)` for fixed-cell atomic relaxation with production finite-difference forces. Variable-cell optimization fails closed; analytic BIPOLE gradients are a gated research preview. |
| `FREQCALC` (periodic vibrations) | Periodic Hessians/vibrations are still roadmap. Molecular vibrations via `run_job(..., compute_hessian=True)` work today. |

### Functionals

| CRYSTAL | vibe-qc |
|---|---|
| `EXCHANGE PBE / CORRELAT PBE` | `opts.functional = "PBE"` (libxc-routed) |
| `B3LYP` | `opts.functional = "B3LYP"`, same flavor in both codes: CRYSTAL's B3LYP keyword computes the **VWN5** variant (verified vs CRYSTAL14 to 1e-9 Ha on H₂/STO-3G), which is exactly vibe-qc's bare `b3lyp` (ORCA definition). Do **not** use vibe-qc's `b3lyp/g` / `b3lypg` (the Gaussian / VWN-RPA flavor) against a CRYSTAL reference, CRYSTAL14 cannot express it, and the comparison would carry a systematic ~10-15 mHa/heavy-atom flavor offset. See [functionals](functionals.md) § B3LYP. |
| `EXCHANGE BECKE / CORRELAT LYP / HYBRID 20` | A hand-built Becke+LYP 20 % hybrid (no VWN slot), *not* B3LYP; compose the same mix via `vq.define_functional` if you really ran that. |
| `PBE0` | Same name, same libxc id |
| `PW1PW` | `opts.functional = "pw1pw"`, Bredow's periodic hybrid. See [functionals](functionals.md). |
| `DFTD3` / `XLYP-D3` style suffix | Pass `opts.dispersion = "d3"` / `"d3bj"` / `"d4"`. Independent from the functional name. |
| `HSE06` (screened hybrid) | `opts.functional = "hse06"`. Short-range exact exchange is supported by the BIPOLE, Ewald, and slab routes, including supported multi-k meshes. |
| `LC-ωPBE` and other long-range-corrected hybrids | Fail closed on periodic routes until the long-range exact-exchange arm is implemented. |

## What CRYSTAL does that vibe-qc doesn't (yet)

Use CRYSTAL for these for now:

* **Analytic periodic forces and cell optimization.** Fixed-cell BIPOLE
  optimization uses production finite-difference forces. The analytic
  gradient remains a gated preview, and variable-cell optimization fails
  closed pending a certified stress and coupled optimizer.
* **Vibrational frequencies for solids** (`FREQCALC`). Depends on
  analytic periodic Hessians; molecular `FREQ` works via
  `run_job(compute_hessian=True)`.
* **Phonon dispersion** (`PHONON`). Roadmap.
* **Elastic constants** (`ELASTCON`). Roadmap.
* **Long-range-corrected hybrids for solids** (`LC-ωPBE` and related
  functionals). HSE06 is supported because its exact-exchange arm is
  short-ranged; long-range exact exchange still fails closed on the
  periodic routes.
* **Coupled-cluster periodic** (`MP2`, `CCSD`). vibe-qc's molecular
  MP2 / RI-MP2 is solid; periodic post-HF correlation is v2.x scope.

No longer on this list (shipped since this page was first
written): production multi-k BIPOLE SCF energies for
RHF/UHF/RKS/UKS/ROKS in the corrected Ewald exchange gauge,
open-shell multi-k SCF, SYM3b symmetry-reduced Fock assembly on
attached-symmetry crystals, and periodic geometry optimisation
with FD forces.

## What vibe-qc does that CRYSTAL doesn't (or does differently)

Things you gain by switching:

* **Auto-citations.** Every `run_periodic_job` writes a
  `.bibtex` / `.references` pair listing every paper to cite,
  including the pob-* basis-set papers, libxc, spglib, and the
  per-functional papers. See [citations](citations.md).
* **Python scripting.** Sweeps over functionals / k-mesh / lattice
  constant are a few-line `for` loop, not multiple `.d12` files.
* **ASE-bridged geometry I/O.** Read CIF / POSCAR / Quantum
  Espresso `.pwi` directly; pass to `PeriodicSystem` via the ASE
  bridge.
* **`vq` queue with vibe-qc-aware preflight.** `vq submit
  --vibeqc-preflight` knows which artefacts the job will write, so
  remote fetch is selective. See [Submitting a job to a remote machine with `vq`](../tutorial/vq_queue_remote_job.md).
* **Open-source.** MPL-2.0; no per-seat license; no proprietary
  binary distribution.

## Validating vibe-qc against CRYSTAL

vibe-qc treats CRYSTAL14 as an out-of-process validation reference
(per [Archived CLAUDE.md § 10](https://vibe-qc.com/docs/):
no QC-code imports). The
[`vq submit`](queue.md#crystal-orca-and-other-external-programs)
runner spawns CRYSTAL14 as a subprocess on the configured remote;
the
[parity matrix](https://github.com/vibe-qc/vibe-qc/blob/main/examples/regression/parity_matrix_orca/)
collects results across multiple codes for the 15 demo systems
([Archived commit `39c5309`](https://vibe-qc.com/docs/)).

For ad-hoc validation, the typical workflow is:

1. Write the same system as a vibe-qc Python file and a
   CRYSTAL14 `.d12`.
2. `vq submit input.py` and `vq submit -- bash run-crystal.sh`
   (see [queue](queue.md) § CRYSTAL14 workflows).
3. Compare the two `.out` files' total energies.

### Comparing total energies: mesh and exchange-convention rules

Three rules prevent the most common apparent "gauge offsets"
(0.5-1.2 Ha-scale) when comparing against CRYSTAL:

1. **Match the k-mesh first.** vibe-qc's zone-centred
   `kpoints=(n,n,n)` Monkhorst-Pack net is the same net as
   CRYSTAL's `SHRINK n n`. A Γ-point vibe-qc run
   (`kpoints=(1,1,1)`) compared against a `SHRINK 2` CRYSTAL run
   differs by the *k-sampling error*, which on a compact ionic
   cell (MgO/STO-3G) is several hundred mHa: that is not a gauge
   difference, and no convention arithmetic will remove it.

2. **For HF and hybrids, mind the exchange G=0 convention.**
   vibe-qc's default `exchange_exxdiv="ewald"` adds the
   probe-charge Madelung correction ξ_M to the exact-exchange
   G=0 channel (the Gygi-Baldereschi / probe-charge-Ewald
   treatment, same as PySCF `exxdiv='ewald'`); its finite-mesh
   energies are therefore already close to the k-converged limit.
   **CRYSTAL applies no such correction**: its coarse-mesh
   exchange samples the q→0 region at the mesh's smallest |q| and
   over-binds its own k-converged value (measured **638 mHa** on
   MgO/STO-3G RHF at `SHRINK 2 2` vs `SHRINK 8 8`). A
   coarse-mesh CRYSTAL total is *neither* vibe-qc's
   `exchange_exxdiv="ewald"` *nor* `"none"`, so for absolute
   totals, **compare at k-convergence** (`SHRINK 8 8`-class
   meshes), where the convention difference vanishes. The
   convention identity at fixed density,

   `E(exxdiv="none") = E(exxdiv="ewald") + ½ · N_e · ξ_M(BvK supercell)`

   (MgO/STO-3G `(2,2,2)`: 2.87669 Ha), converts between vibe-qc's
   own two conventions but does **not** reproduce CRYSTAL's
   coarse-mesh number.

3. **Pure DFT (PBE, LDA, …) has no exchange-convention seam.**
   Any large offset there is the k-mesh (rule 1), the basis, or a
   route defect, not a gauge.

Practical expectations at matched basis, matched (dense) mesh:

* **BIPOLE route** (fold-converged `bipole_cutoff_bohr`, i.e.
  reported fold drift < 1e-4): a few mHa from k-converged
  CRYSTAL totals (measured MgO +0.4 to +3.4 mHa, diamond
  −0.145 mHa, Si −0.659 mHa) and ~0.1 mHa from PySCF at the
  matched `exxdiv='ewald'` convention. BIPOLE is the
  method-family match for CRYSTAL (both are direct-space
  Gaussian truncation + Ewald codes), so prefer it for CRYSTAL
  parity work.
* **GDF route on compact-core ionic cells (MgO, NaCl-class) at
  multi-k**: the default RSGDF fit truncates a dense-core tail
  worth **~0.5 Ha** on MgO/STO-3G `(2,2,2)` (run is tagged
  `+PARITY_HELD` and warns). This is a documented fitting-route
  defect, not a gauge difference; for parity either use the
  BIPOLE route or set the opt-in `rsgdf_tail_ke_cutoff`
  (≈ `11·ζ_max`; ~70× cost).
* **CRYSTAL-side truncation noise** at default `TOLINTEG`
  (7 7 7 7 14) is ~1e-4 Ha: tightening to `TOLINTEG` 12/24 +
  `NOBIPOLA` moved NaCl/STO-3G `(4,4,4)` by only −0.13 mHa.
  Use `TOLDEE 9`; treat ~0.1 mHa as the CRYSTAL-side parity
  floor.
* **Component energies do not compare one-to-one.** vibe-qc
  places the neutralising-background (jellium) terms explicitly
  in V_ne and E_nn (they cancel in the total for a neutral
  cell); CRYSTAL folds them differently into its `ENECYCLE`
  components. Component-wise differences of ~16 Ha on
  MgO/STO-3G are expected while the totals agree; only compare
  totals (and total-energy differences).

Known CRYSTAL practicalities from the parity campaign: LiH/STO-3G
diverges in CRYSTAL at `SHRINK 1` and `2` (conducting-state
oscillation; use `SHRINK 4` + `LEVSHIFT`), and CRYSTAL23 v1.0.1 does
not accept the `BASISSET` keyword form some deck generators emit, so
inline the basis block instead.

## See also

- [user_guide/bipole](bipole.md), BIPOLE periodic-RHF driver status,
  the 1988 quartet-expansion source, and the Saunders 1992 electrostatics
  reference.
- [user_guide/external_codes](external_codes.md) §
  "CRYSTAL14, periodic regression reference", the subprocess
  runner pattern.
- [user_guide/queue](queue.md) § CRYSTAL14 workflows, how to
  submit CRYSTAL14 jobs through the same `vq` queue used for
  vibe-qc.
- [user_guide/basis_sets](basis_sets.md) § "CRYSTAL-format basis
  parser", drop a CRYSTAL `.bas` file under `custom/`.
- [Why solid-state calculations use `pob-TZVP`](../tutorial/pob_tzvp.md), why
  CRYSTAL-style basis sets matter for periodic calculations.
- [MgO from the Materials Project: `vqfetch` end-to-end](../tutorial/mgo_from_materials_project.md)
  - end-to-end periodic SCF using a Materials Project / CIF
  starting structure.
