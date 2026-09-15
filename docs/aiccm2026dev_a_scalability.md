# Γ-CCM (`aiccm2026dev-a`) dense four-center scalability - analysis & strategy

> **D89 construction-boundary notice (2026-07-15).** This completed analysis
> preserves its measured memory and timing results, but its earlier route labels
> are superseded. Γ-CCM is the union-and-weight/Wigner-Seitz integral-weighting
> construction; χ-CCM is the distinct finite-translation-group character
> construction. Neutral GDF/Bloch and real-Gamma calculations are
> same-Hamiltonian representation controls once that block-circulant Hamiltonian
> is specified. They are not Γ-CCM production routes, and their finite Fourier
> equivalence does not prove Γ-CCM equals χ-CCM. Accordingly, statements below
> that call GDF "exact CCM", a Γ-CCM production path, or mere routing are retained
> as historical planning language only. The resource measurements remain valid
> for the named implementation routes.

Completed Γ-CCM scalability record. This is the **Phase 1
(analysis) + Phase 2 (strategy)** deliverable for the `aiccm2026dev-a` (Γ-CCM)
memory workstream: why the dense symmetric four-center OOMs on real 3-D crystals,
and the surveyed phased plan to make it fit in RAM.

> **Status: analysis + strategy, for review.** No code-behaviour change yet. The
> recommended plan must be reviewed before any implementation (Phase 3) - see the
> handover mandate (CLAUDE.md §11/§14).

All numbers below are **analytical** (cheap: `N_pad` and the implied tensor bytes
need no ERI build) or from **tiny-cell** (1-D H-chain) memory profiles run under a
hard RSS cap. The genuine c-diamond `(2,2,2)` reproduction is a declared-cap **vq**
job (id recorded in the handover), not a local run - a parallel run already spiked
this shared box to 137 GB.

---

## 1. Phase 1 - analysis: where the memory goes

### 1.1 The failure path (confirmed by reading the source)

```
run_ccm_rhf (periodic/ccm/scf.py:117)
  -> _ccm_eri_for_method (scf.py:45)
    -> ccm_eri_symmetric / ccm_eri (periodic/ccm/padded.py:361 / :289)
      -> eri_pad = np.asarray(compute_eri(pad.basis))    <-- the killer
         # pad.basis is the PADDED cluster: home cell + every ±2t image cell.
```

There are **two** distinct large allocations in the pipeline, in order of severity:

| # | Allocation | Site | Size | Severity |
|---|---|---|---|---|
| 1 | **Padded ERI** `compute_eri(pad.basis)` | `padded.py:321` / `:401` (C++ core) | `N_pad⁴ × 8` B | **the wall** |
| 2 | Folded effective tensor `eff` | `padded.py:333`/`:411`, returned to `scf.py` | `N_ref⁴ × 8` B | secondary wall (pob 3-D) |
| 3 | J/K build `einsum("mnrs,rs->mn", eff, D)` | `scf.py:157-158` | works on `eff`, O(N_ref²) extra | not a wall |

* `N_pad = N_ref_ao × N_eri_cells`, where `N_eri_cells = len(eri_cells(ccm))` is the
  number of ±2t image cells the four-center fold materialises (`padded.eri_cells`).
* The padded ERI (#1) is allocated **inside the C++ libint core** (`compute_eri`),
  so a Python `tracemalloc` does not see it - process RSS does. It is dense and
  **screening-free** (a validation builder).
* The folded `eff` (#2) is the object `scf.py` actually contracts; it is `N_ref⁴`,
  independent of the image count, and is what survives once the padded ERI is freed.

### 1.2 Analytical scaling table (the wall)

`N_ref_ao` (= `ccm.nbf`), `N_eri_cells`, `N_pad`, dense-padded-ERI (`N_pad⁴·8`),
folded-effective-ERI (`N_ref⁴·8`) for the canonical test cells. STO-3G unless noted.
Computed from geometry only - **no ERI built**.

| system | nrep | basis | n_atoms | N_ref | wssc cells | eri cells | **N_pad** | **padded ERI** | folded ERI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| h-chain | (1,1,1) | sto-3g | 2 | 2 | 1 | 1 | 2 | 0.1 µiB | 0.1 µiB |
| h-chain | (2,1,1) | sto-3g | 4 | 4 | 3 | 5 | 20 | 1.2 MiB | 2 KiB |
| h-chain | (4,1,1) | sto-3g | 8 | 8 | 3 | 5 | 40 | 19.5 MiB | 31 KiB |
| h-chain | (8,1,1) | sto-3g | 16 | 16 | 3 | 5 | 80 | 0.31 GiB | 0.5 MiB |
| **c-diamond** | (1,1,1) | sto-3g | 2 | 10 | 7 | 25 | 250 | **29.1 GiB** | 76 KiB |
| **c-diamond** | (2,1,1) | sto-3g | 4 | 20 | 11 | 45 | 900 | **4.77 TiB** | 1.2 MiB |
| **c-diamond** | (2,2,1) | sto-3g | 8 | 40 | 25 | 117 | 4680 | **3490 TiB** | 19.5 MiB |
| **c-diamond** | (2,2,2) | sto-3g | 16 | 80 | 25 | 117 | 9360 | **55 800 TiB** | 0.31 GiB |
| si-diamond | (1,1,1) | sto-3g | 2 | 18 | 7 | 25 | 450 | 306 GiB | 0.8 MiB |
| si-diamond | (2,1,1) | sto-3g | 4 | 36 | 11 | 45 | 1620 | 50.1 TiB | 12.5 MiB |
| si-diamond | (2,2,2) | sto-3g | 16 | 144 | 25 | 117 | 16848 | 586 000 TiB | 3.2 GiB |
| mgo | (1,1,1) | sto-3g | 2 | 14 | 13 | 57 | 798 | 2.95 TiB | 0.3 MiB |
| mgo | (2,1,1) | sto-3g | 4 | 28 | 17 | 79 | 2212 | 174 TiB | 4.6 MiB |
| mgo | (2,2,2) | sto-3g | 16 | 112 | 27 | 125 | 14000 | 280 000 TiB | 1.17 GiB |
| cscl | (2,2,2) | pob-tzvp-rev2 | 16 | 176 | - | 125 | 22000 | 1.7e6 TiB | 7.2 GiB |
| **c-diamond** | (2,2,2) | pob-tzvp-rev2 | 16 | 288 | - | 117 | 33696 | 9.4e6 TiB | **51.3 GiB** |
| **mgo** | (2,2,2) | pob-tzvp-rev2 | 16 | 296 | - | 125 | 37000 | 1.4e7 TiB | **57.2 GiB** |

AO density: STO-3G is 1 (H) / 5 (C) / 7 (MgO avg) / 9 (Si) AO/atom; pob-tzvp-rev2
is ~11-18 AO/atom.

**Where the wall is.**

* **Dense padded ERI (#1) dies at the first real 3-D cell.** c-diamond `(1,1,1)`
  STO-3G already needs 29 GiB; `(2,1,1)` needs 4.8 TiB; `(2,2,2)` needs 55 800 TiB.
  Every dense-four-center route (`aiccm-hf`, `-ks`, `-viz`, `-localize`, `-mp2`,
  `-ccsd`) inherits this and OOMs identically - this is the reproduced blocker.
* **Folded `eff` (#2) is fine at STO-3G but is the *second* wall at pob-tzvp-rev2
  3-D.** c-diamond/MgO `(2,2,2)` pob-tzvp-rev2 fold to a **51-57 GiB** `eff` tensor
  even if #1 were solved - so a fix that only removes the padded ERI still leaves a
  near-64-GB ceiling at production basis. An integral-direct J/K (never form `eff`)
  removes **both** walls; anything that still materialises `eff` only removes #1.

### 1.3 Tiny-cell memory profile (confirms the dominant allocation + exponent)

1-D H-chain STO-3G, the only family safe to actually build locally (largest dense
ERI here is 80⁴·8 = 312 MiB), under a 4 GB in-process RSS watchdog:

| nrep | N_ref | N_pad | `compute_eri` tensor (exact) | process RSS high-water |
|---|---:|---:|---:|---:|
| (2,1,1) | 4 | 20 | 1.221 MiB | 115.6 MiB |
| (4,1,1) | 8 | 40 | 19.531 MiB | 135.6 MiB |
| (8,1,1) | 16 | 80 | 312.500 MiB | 448.3 MiB |

* The dominant allocation is unambiguously `compute_eri`'s **`N_pad⁴` padded
  tensor**: its exact size scales `N_pad → 2·N_pad ⇒ tensor ×16`, a fitted
  **exponent of 4.000** (`tensor ∝ N_pad⁴`). Process RSS tracks it (the ~313 MiB
  tensor adds ~313 MiB to the high-water at the largest case).
* `tracemalloc` (Python allocator) reports ~0 for `compute_eri` because the tensor
  is a C++-core allocation - RSS / `.nbytes` are the correct instruments, and both
  agree with the `N_pad⁴` law.

### 1.4 The lean neutral-control counter-example (quantified)

The memory target is not hypothetical: a separately constructed neutral GDF
control already avoids the `N_pad⁴` tensor, and symmetry supplies an independent
constant-factor lever. The GDF route demonstrates a lean periodic fit, not a
construction-preserving Γ-CCM fix:

1. **`run_ccm_rhf_gdf` (neutral multi-k GDF control, `ri.py`).** This
   runs the validated native multi-k GDF **on the unit cell** with the `nrep`
   k-mesh - so its working AO dimension is the **unit-cell** `nbf` (10 for
   c-diamond, not the supercell's 80), and its 3-index RI tensor is
   `N_aux × n_unit² × n_k` - a **few MiB** k-resolved cderi, never `N_pad⁴`:

   | system | unit nbf | GDF 3-index RI (order) | dense padded ERI |
   |---|---:|---:|---:|
   | c-diamond (2,2,2) | 10 | ~3 MiB | 55 800 TiB |
   | mgo (2,2,2) | 14 | ~8 MiB | 280 000 TiB |

   The handover records it ran c-diamond `(2,2,2)` in **~6 GB / ~23 min**
   (E/atom −37.4155 Ha). This is evidence that a lean neutral fitted-torus
   control is achievable. It is not an exact Γ-CCM result or an oracle for the
   union-and-weight construction. (The submitted vq job re-confirms the peak
   RSS on a fresh box.)

   The production RHF and hybrid-RKS RSGDF cache also reduces its expensive
   `(k_i,k_j)` fits over the currently gated cluster space-group and
   time-reversal stars.
   The SCF still uses the complete k-mesh: only symmetry-equivalent Lpq
   factors are reconstructed, in the canonical auxiliary frame. The returned
   `CCMGDFResult` records `gdf_pair_builds`, `gdf_pair_total`, and
   `gdf_pair_reduction_factor`; `symmetry=False` retains the full-build
   validation path. Fully 3-D replica meshes currently retain that exact
   full-build path even when symmetry is enabled: GitLab IID 337 showed that
   the finite production Bloch cell list is not covariant under every cubic
   star operation on diffuse Li/STO-3G. The generic shared-q batching remains
   active, and the result reports a reduction factor of one. The `(3,1,1)` H2
   chain gate still measures 8 builds for 9 cache entries with an energy
   change below `1e-15 Ha`.

2. **Symmetry-unique atom pairs (`symmetry.py` `ccm_symmetry_unique_atom_pairs`).**
   The cluster-invariant space group reduces the ordered home atom-pair count that
   the ERI/Fock build must touch - measured on the `(2,2,2)` cells:

   | system | pairs | unique | reduction | cluster group order |
   |---|---:|---:|---:|---:|
   | c-diamond | 256 | 19 | **13.47×** | 48 |
   | si-diamond | 256 | 19 | **13.47×** | 48 |
   | mgo | 256 | 32 | **8.0×** | 48 |

   An unused lever: only one representative pair per orbit needs its block built,
   the rest follow by the AO rotation `P`. It is a constant-factor (not
   complexity-class) win, so it **stacks on top of** an integral-direct fix rather
   than replacing it.

> Aside (not in scope here): `build_padded_cluster` raises `KeyError(0)` on a
> pob-tzvp-rev2 cell (cscl) - an ECP/ghost-atom basis-mapping bug in the padded
> assembly, independent of memory. It belongs to the method/basis chain
> ([Archived `HANDOVER_AICCM_FOLLOWON.md`](https://vibe-qc.com/docs/)), not
> this workstream; flagged for them. The pob
> `N_pad` rows above use the analytical `N_pad = nbf × eri_cells`, which is
> unaffected.

---

## 2. Phase 2 - strategy: candidates, adversarial check, recommended plan

The goal (handover §1): the dense-four-center routes (hf/ks/viz/localize/post-HF)
run light-tier 3-D cells (c-diamond/Si/MgO `(2,2,2)` and larger) within ~64 GB -
ideally far less - with **zero change to any existing `-a` energy/property**
(this is a memory/throughput refactor, **not** a numerics change).

### 2.1 Candidate approaches

For each: what it is, expected memory/throughput, implementation cost, **how it
could fail / what would falsify it**.

#### C1 - Integral-direct J/K (never materialise the 4-index ERI)

The standard N⁴-memory fix: contract shell quartets into J/K **on the fly** with
the WSSC weight applied per quartet, and accumulate into the `N_ref²` Fock blocks
- never building `eri_pad` (#1) *or* `eff` (#2). Memory drops from `O(N_pad⁴)` /
`O(N_ref⁴)` to `O(N_ref²)` (the Fock + density) plus a bounded shell-quartet
working set. This is the architecture `run_ccm_rhf_scalable`'s C++
`build_jk_ccm_weighted` already *aims* at (it removes the Python padded tensor),
**but** that path (a) still materialises a tensor in C++ (`bra_home_full` is noted
O(nbf⁴) in C++) and (b) carries a **separate over-binding correctness bug** owned
by the testing chat - so it is a memory *reference*, not a drop-in. The clean C1 is
a screened, integral-direct contraction that reproduces `ccm_eri` / `ccm_eri_symmetric`
**byte-for-byte**.

* **Memory:** `O(N_ref²)` + working set. c-diamond `(2,2,2)` STO-3G: from 55 800 TiB
  to well under 1 GiB. pob-tzvp-rev2 `(2,2,2)`: removes the 51 GiB `eff` wall too.
* **Throughput:** comparable to the dense build once screened (Schwarz); the dense
  builder is itself O(N_pad⁴) FLOPs, so direct is *faster* (it skips negligible
  quartets).
* **Cost:** high (the genuine engineering). Needs the WSSC `ω_{μνρσ}` weight applied
  inside the quartet loop with the **bra-ket symmetrisation** (`ccm_eri`) /
  **symmetric bridge + independent min-image fold** (`ccm_eri_symmetric`)
  reproduced exactly - the M2b magnitude⊕symmetry subtlety lives here.
* **Could fail / falsifier:** the symmetric four-center weight couples the bra
  output indices to the contracted ket atom (`¼(ω_μr+ω_νr+ω_μs+ω_νs)`), so a naive
  per-quartet scalar weight loses resolution; if the direct contraction cannot
  reproduce `ccm_eri_symmetric(ccm)` to ~1e-12 on the 1-D/2-D validation cells, C1
  is falsified for the symmetric method and must fall back to building `eff` from
  screened blocks (removes #1 only). **Gate:** byte-for-byte vs the dense `eff` on
  every cell currently in the test suite.

#### C2 - Shell-pair / block batching with a bounded working set

Build the folded `eff` (#2) in **batches** of ket shell-pairs, streaming the padded
ERI block-by-block so the padded `N_pad⁴` tensor (#1) is never fully resident - a
fixed-size working buffer (e.g. one `(N_ref, N_ref, block, block)` slab) is folded
into `eff` and discarded. Removes wall #1; leaves wall #2 (`eff` is still `N_ref⁴`).

* **Memory:** `O(N_ref⁴)` (the `eff` that survives) + a tunable block buffer.
  c-diamond `(2,2,2)` STO-3G: 0.31 GiB (just `eff`); pob-tzvp-rev2 3-D: still
  **51 GiB** (`eff`) - so C2 *alone* does not reach pob 3-D.
* **Throughput:** similar FLOPs to dense; more passes over shells.
* **Cost:** medium. Reuses the existing fold logic per block; the hard part is a
  blocked `compute_eri` (per-shell-quartet) entry - vibe-qc's core may already
  expose shell-quartet ERIs (libint), else a new binding.
* **Could fail / falsifier:** if the core only exposes the whole-basis `compute_eri`
  (no per-shell-quartet call), C2 needs a new C++ binding and collapses into C1's
  cost. Falsified as a *cheap* milestone if no batched ERI entry exists. **Gate:**
  same byte-for-byte `eff` reproduction.

#### C3 - Retired as a Γ-CCM scalability solution; retain as a control

The GDF path (`run_ccm_rhf_gdf`) and neutral cderi belong to a separately
specified neutral fitted-torus Hamiltonian. They cannot replace the
union-and-weight tensor while retaining Γ-CCM identity. The measurements below
remain useful for sizing a control or a separately labelled post-HF research
route, but C3 is not a Γ-CCM memory solution:

* **HF/KS** already have the lean GDF control. Wiring `aiccm-hf`/`-ks` to it
  would silently change the construction and is forbidden.
* **viz/localize** need orbitals + density, which the GDF result provides.
* **post-HF (MP2/CCSD)** need MO ERIs; the RI-consistent neutral cderi `L`
  (`ccm_neutral_cderi`) gives `(ia|jb)` via `L`-contraction without ever forming
  the `N_ref⁴` AO tensor - the standard RI-MP2 memory profile (`O(N_aux·N_occ·N_virt)`).
* **Memory:** the GDF/RI footprint (few MiB-few GiB), the lean profile.
* **Throughput:** the GDF route's ~23 min on c-diamond `(2,2,2)`; RI-MP2 is cheaper
  than dense-AO MP2.
* **Cost:** low-medium for HF/KS/viz/localize (routing + result-shape adaptation);
  medium for post-HF (RI-MO transform from `L`).
* **Could fail / falsifier:** **the consumer boundary is owned by the method chain**
  (`mp2/ccsd/dlpno/neutral/properties/localize` - explicitly *not this chat's
  files*). C3 is therefore a **cross-chat** change: this chat can provide the lean
  J/K / `L` source; the method chat must adapt the consumers. It is falsified as a
  *self-contained* milestone - it requires coordination (handover §0). Also: RI is
  exact *to the fitting error*, not byte-for-byte vs the dense four-center, so C3
  changes the numbers at the ~0.05 mHa/atom level - it does **not** pass the
  byte-for-byte gate and so is a *new route*, not a refactor of the existing one.
  Keep it as an **opt-in lean path**, not a silent replacement.

#### C4 - Exploit the 13.5× symmetry-unique pairs

Build only one representative atom-pair block per space-group orbit, scatter the
rest via the AO rotation `P`. Measured 13.47× (c-diamond/Si) / 8.0× (MgO) on
`(2,2,2)`. A **constant-factor** throughput + working-set win that **stacks on** C1
or C2 (it reduces the number of quartet blocks built, not the asymptotic memory of
the result).

* **Memory:** reduces the working set / build time, not the `eff` / Fock size.
* **Throughput:** up to ~13× fewer ERI blocks on high-symmetry cells.
* **Cost:** medium (petite-list scatter; the `symmetry.py` orbits already exist).
* **Could fail / falsifier:** the reduction approaches `|G_c|` only for generic
  pairs; tiny / low-symmetry cells get less. Falsified as a *primary* fix (it never
  removes the N⁴ wall alone). The symmetry machinery is in `symmetry.py` (method
  chain) - using it from the build is a coordinate-before-edit boundary.

#### C5 - Out-of-core streaming to disk

Spill the padded ERI / `eff` to disk and stream blocks during the J/K contraction.

* **Memory:** `O(N_ref²)` RAM; disk holds the `N_pad⁴` / `N_ref⁴` data.
* **Throughput:** I/O-bound; for c-diamond `(2,2,2)` STO-3G the *padded* tensor is
  55 800 TiB - **un-storable even on disk**, so out-of-core does not help wall #1.
  It could hold the `N_ref⁴` `eff` (0.3-57 GiB) but that is exactly what C1/C2
  avoid entirely.
* **Cost:** medium (memmap plumbing).
* **Could fail / falsifier:** falsified for wall #1 by the table - you cannot stream
  a 55 800 TiB object. Only ever a fallback for the `N_ref⁴` `eff`, which the
  better candidates eliminate. **Not recommended.**

#### C6 - GPU integrals (LONG-TERM - scope only)

Offload the quartet contraction to GPU. Orthogonal to the memory question (a GPU
integral-direct engine is still integral-direct); it is a throughput lever once C1
exists. **Scope only - do not plan to build first.** Falsifier: irrelevant to the
RAM blocker, which C1 already solves on CPU.

### 2.2 Adversarial / panel summary

| candidate | removes wall #1 (padded N_pad⁴) | removes wall #2 (eff N_ref⁴) | byte-for-byte? | self-contained (this chat's files)? | primary or stacking |
|---|:--:|:--:|:--:|:--:|---|
| **C1 integral-direct** | ✅ | ✅ | ✅ (the gate) | ✅ scf.py/padded.py + new module/C++ | **primary** |
| C2 block-batched fold | ✅ | ❌ | ✅ | ✅ | bridge / fallback |
| C3 RI/GDF consumers | ✅ | ✅ | ❌ (RI error) | ❌ needs method chain | opt-in lean route |
| C4 symmetry pairs | ❌ (factor) | ❌ | ✅ | ⚠ uses symmetry.py | stacking |
| C5 out-of-core | ❌ (un-storable) | ⚠ | ✅ | ✅ | not recommended |
| C6 GPU | ❌ | ❌ | ✅ | ✅ | long-term throughput |

The single fact that orders everything: **wall #1 is un-storable** (55 800 TiB for
c-diamond `(2,2,2)` STO-3G), so any candidate that materialises the padded ERI -
even to disk - is dead. The fix must **never form the 4-index tensor**. That is C1
(and C2 as a partial bridge). C3 is the already-proven lean route but it is a
*different numerical answer* (RI error) and crosses the chat boundary, so it is an
opt-in path, not a refactor of the dense one. C4 stacks. C5/C6 are out.

### 2.3 Recommended phase ordering

Each phase is a small, independently-shippable, **byte-for-byte-gated** milestone
(CLAUDE.md §14). Land green, full C++ + Python suite, CHANGELOG accurate.

**Phase 3a (first milestone) - block-batched fold (C2), removes wall #1.**
Stream the padded ERI in ket-shell-pair blocks into the existing fold, so
`ccm_eri` / `ccm_eri_symmetric` produce the **identical** `eff` without the
`N_pad⁴` resident tensor. Smallest landable step that makes c-diamond `(2,2,2)`
STO-3G (and every STO-3G 3-D light-tier cell) run, because once #1 is gone the
surviving `eff` is only 0.3 GiB at STO-3G. **Gate:** `eff_blocked == eff_dense` to
~1e-12 on every suite cell; energies/properties unchanged. **Risk:** needs a
per-shell-quartet (or blocked) ERI entry in the core - *first task is to confirm it
exists*; if not, this phase merges into 3b.

**Phase 3b (second milestone) - integral-direct J/K (C1), removes wall #2.**
Contract quartets straight into J/K with the WSSC weight + symmetrisation applied
on the fly - never forming `eff`. This is what unlocks **pob-tzvp-rev2 3-D**
(the 51-57 GiB `eff` wall) and larger cells. Reuse the 3a block machinery; add the
J/K accumulation. **Gate:** `J,K` match the dense-`eff` contraction byte-for-byte
on every suite cell. **This is the master deliverable.**

**Phase 3c - stack the 13.5× symmetry-unique pairs (C4)** onto 3b's quartet loop
(petite-list scatter via `symmetry.py` orbits; coordinate the read with the method
chain). Throughput win on high-symmetry cells; **gate:** symmetry-on == symmetry-off
energy/Fock.

**Phase 3d (parallel, opt-in) - expose the lean RI/GDF source (C3)** for the
consumers that can accept the RI answer (post-HF via the neutral cderi `L`, viz via
GDF orbitals). This chat provides the lean J/K / `L`; the **method chain** wires the
consumers (handover §0 boundary). Documented as a *separate route* (RI error, not
byte-for-byte), not a replacement.

**Deferred:** C5 (out-of-core - un-storable for #1), C6 (GPU - long-term throughput
once C1 lands).

### 2.4 The byte-for-byte numerics gate (every phase)

Non-negotiable (handover §3, CLAUDE.md §7 - no papering over):

* Every existing `-a` energy and property is **unchanged** by 3a/3b/3c. The
  reference is the current dense `ccm_eri` / `ccm_eri_symmetric` `eff` (and the J/K
  it yields) on every cell in the test suite (`test_ccm_*`, `test_aiccm2026dev_a`).
  Tolerance: ~1e-12 on `eff` / J / K (machine round-off of the contraction order),
  These are control-route resource estimates, not a bit-identical Γ-CCM
  refactor claim.
* **Add memory-regression coverage:** a cell that used to trip the
  `VIBEQC_CCM_PADDED_ERI_MAX_GB` guard (e.g. c-diamond `(2,2,2)` STO-3G) now runs
  under a declared cap and returns the same energy.
* **Out of scope (do not touch):** the `run_ccm_rhf_scalable` over-binding
  correctness bug (testing chat owns it); the `-b` line; the method layer's files.
  C3's RI route is a *new* path at RI accuracy, explicitly **not** gated against the
  dense four-center byte-for-byte - keep it opt-in and labelled.

### 2.5 What would falsify the recommendation as a whole

The plan rests on one claim: **the fold/contraction can be reproduced block-wise or
quartet-wise to machine precision without the full 4-index tensor.** It is falsified
if the symmetric four-center weight (`ccm_eri_symmetric`'s
`¼(ω_μr+ω_νr+ω_μs+ω_νs)` bridge with independent min-image folds of r and s) cannot
be expressed per-quartet without re-materialising cross-index data of size `N_pad⁴`
- i.e. if the weight is genuinely non-separable across the streaming boundary. The
1-D/2-D byte-for-byte gate in Phase 3a is the early decisive test: if a blocked fold
cannot reproduce `eff` there, the whole "never form the tensor" premise is wrong and
the Γ-CCM route remains gated while the neutral GDF calculation stays a
separately labelled control. Accepting the control as the Γ-CCM production
answer is not an allowed fallback.

---

## 3. Phase 3 - implementation

### 3b - integral-direct J/K in the C++ kernel (DONE, 2026-06-25)

Maintainer-approved ordering: **3b first** (the review chose the integral-direct
C++ kernel over the Phase-3a block-batched Python fold). After FR-2 routed
`aiccm-viz`/`-localize`/`-pao` to `run_ccm_rhf_scalable`, that driver is the
production HF path and already ran STO-3G 3-D - so the sharp remaining blocker is
**wall #2 inside its C++ kernel** (`build_jk_ccm_weighted`): both production methods
(`bra_home_full` for `union12`, `aiccm2026dev-a` for the method of record) allocated
a thread-local effective tensor `Vj_tls[n_threads]` of `nbf²×nbf²`, then reduced to
`V_full`, symmetrised to `V_sym`, and contracted - a peak of `(n_threads+2)·N_ref⁴·8`
bytes (~300 GiB on a pob-tzvp-rev2 c-diamond (2,2,2) cell at `nbf=288`).

**What landed.** Two new C++ kernels in
[`cpp/src/periodic_fock.cpp`](../cpp/src/periodic_fock.cpp) -
`aiccm2026dev-a-direct` and `bra_home_full-direct` - reuse the *identical* verified
quartet loop and WSSC weight, but fold each weighted block straight into thread-local
J/K instead of `Vj`. The fold is exact: with `V_sym = ½(V + Vᵀ)` and the full-branch
contractions `J[μν]=Σ P[λσ]V_sym[μν,λσ]`, `K[μν]=Σ P[λσ]V_sym[μσ,λν]`, a single block
`t = (μν|λσ)·w` contributes

```
J[μ,ν] += ½ t P[λ,σ]      J[λ,σ] += ½ t P[μ,ν]      (the V and Vᵀ halves)
K[μ,σ] += ½ t P[λ,ν]      K[λ,ν] += ½ t P[μ,σ]
```

followed by the same final Hermitisation. Peak JK-build memory drops from
`(n_threads+2)·N_ref⁴·8` to `n_threads·2·N_ref²·8`. The builder is already rebuilt
every SCF iteration (no `V` cache - `CCMWeightedGammaJKBuilder::build_g_rhf` calls
`build_jk_ccm_weighted(…D…)` each call), so this is **zero recompute penalty** - same
integrals, no tensor, and it drops the separate `nbf⁴` contraction pass too.

`run_ccm_rhf_scalable` gains a `four_center=` keyword:
[`scf.py`](../python/vibeqc/periodic/ccm/scf.py) maps `"direct"` (default → the
`-direct` kernels) or `"full"`/`"dense"` (→ the **preserved** full-tensor kernels,
the small-cluster comparison reference). The dense Python `run_ccm_rhf` and the full
C++ branches are untouched - the full four-center path stays runnable for comparison
(maintainer constraint, 2026-06-25).

**Verification (gated, byte-for-byte).**

| cell | basis | direct vs full | note |
|---|---|---|---|
| He square (2,2,1) 2-D | sto-3g | `|ΔE| = 0` (exact) | + full == padded |
| H₄ chain (4,1,1) 1-D | sto-3g | `|ΔE| < 1e-9` | both `union12` + `aiccm2026dev-a` |
| **c-diamond (2,2,2) 3-D** | sto-3g | `|ΔE| = 0` (exact) | peak RSS **0.23 GiB** (direct) vs **1.55 GiB** (full), `OMP=2` |

Tests in [`tests/test_ccm_scalable.py`](../tests/test_ccm_scalable.py):
`test_scalable_direct_matches_full` (parametrised, the byte-for-byte gate) and
`test_scalable_direct_memory_regression` (slow; the c-diamond (2,2,2) RSS contrast -
direct stays O(nbf²) while full allocates the O(nbf⁴) tensor, same energy). Full CCM
suite green (185 tests). The authoritative production-basis (pob-tzvp-rev2, ~300 GiB
full vs lean direct) reproduction is a vq job (testing chat, §0).

**Consumer adoption.** `properties.py` (viz figures) and `convergence.py` route
through `run_ccm_rhf_scalable` → they pick up `four_center="direct"` automatically.
`dft.py` (`run_ccm_rks`/`run_ccm_uks`) builds the JK builder directly; it now takes
the same `four_center="direct"` default via the shared `scf._ccm_scalable_cxx_method`
helper, so KS-CCM scales too (landed with maintainer authorization to cross the §0
method-layer boundary for this one-liner). The post-HF stack (mp2/ccsd) still rides
the dense Python `run_ccm_rhf` - out of scope here (small-cluster + the byte-for-byte
reference); RI-MP2 via the neutral cderi is the 3d route for those.

### 3a / 3c / 3d - status

* **3a (block-batched Python fold):** demoted. With `scalable` as the production
  path, the dense Python `run_ccm_rhf` now serves only mp2/ccsd small-clusters + the
  reference; 3b solved the production blocker directly. 3a remains available if the
  dense Python path itself needs STO-3G 3-D.
* **3c (symmetry-unique pairs, 13.5×):** unchanged plan - a constant-factor
  throughput win that stacks on the 3b quartet loop; deferred.
* **3d (RI/GDF for consumers):** cross-chat, opt-in; the GDF route is the 3-D
  *accuracy* path (the bare four-center is a 3-D model number). Unchanged.
