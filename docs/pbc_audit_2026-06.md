# PBC routes audit, efficiency, symmetry, convergence (2026-06-10)

Maintainer-requested full audit of the periodic routes, feeding three
workstreams: (1) per-route efficiency improvements, (2) systematic
symmetry utilization, (3) intelligent default convergence strategies
(`convergence="auto"`, landed for BIPOLE the same day, see
`periodic_convergence_auto.py`). Sources: code sweep of every
`jk_method` route on main @ `8ac3fbbd` + the 2026-05/06 profiling
memories. File:line references are as of that SHA.

## 1. Convergence-aid coverage matrix

Which aids each route's drivers actually wire (✓ = wired, with
default;, = absent; ✗ = explicitly rejected):

| Route / driver | DIIS family | damping | dyn. damping | FMIXING | level shift | LS warmup | smearing | MOM | ODA | quad. fallback |
|---|---|---|---|---|---|---|---|---|---|---|
| **BIPOLE** RHF/UHF/RKS/UKS (Γ+multi-k) | full family (DIIS/KDIIS/EDIIS/ADIIS via `MultiKPeriodicSCFAccelerator`) | ✓ 0 | ✓ off | ✓ 0 (KS auto-30% for DFT functionals) | ✓ 0 (+schedule) | - | ✓ KS only | ✓ | ✓ | - |
| **EWALD Γ** RHF | Pulay only (family ✗ `NotImplementedError`) | ✓ 0 | ✓ off | ✓ 0 | ✓ 0 | - | - | - | - | ✓ |
| **EWALD Γ** RKS/UHF/UKS | Pulay only | ✓ 0 | ✓ off | - | ✓ 0 | - | - | - | - | ✓ |
| **EWALD multi-k** RHF | per-k Pulay | ✓ 0 | - | - | ✓ 0 (+schedule) | - | ✓ | ✓ | ✓ | ✓ |
| **EWALD multi-k** RKS | per-k Pulay | ✓ 0 | - | - | ✓ 0 | - | ✓ | - | - | ✓ |
| **EWALD multi-k** UHF | per-spin per-k Pulay | ✓ 0 | - | - | ✓ 0 | - | - | - | - | ✓ |
| **EWALD multi-k** UKS | per-spin per-k Pulay | ✓ 0 | - | - | ✓ 0 | - | ✗ raises | - | - | ✓ |
| **GDF Γ** (`periodic_rhf_gdf`) | Pulay | ✓ 0 | - | ✓ 0 | ✓ 0 | ✓ auto-5-cycles | ✓ | - | - | - |
| **GDF multi-k** (`periodic_k_gdf`) | per-k Pulay | ✓ 0 | - | ✓ 0 | ✓ 0 | ✓ auto-5-cycles | ✓ | - | - | - |
| **GDF compcell Γ** (`pbc_gdf`) | accelerator family | ✓ 0 | ✓ off | - | - | - | - | - | - | - |
| **RIJCOSX** | C++ Gamma RHF engine; multi-k GDF/COSX backend | ✓ | ✓ | - | ✓ | ✓ closed-shell multi-k | ✓ multi-k | ✓ | - | - |
| **GPW / GAPW** | **ignores the options struct entirely** (own hardcoded defaults; OT experimental, unwired) | - | - | - | - | - | ✗ own module | - | - | - |

**Findings (convergence):**

* C1. Coverage is wildly inconsistent across routes, same physics,
  different aid sets. The new `periodic_convergence_auto` resolver is
  the single place to converge on; routes should adopt it as they are
  touched (BIPOLE done; GDF natural next, it already has the most
  complete per-route aid set).
* C2. GPW/GAPW silently ignore `use_diis`/`damping`/`max_iter`-class
  runner options, a transparency bug by the new standard (the .out
  prints options the driver never sees). Owner: GPW chat.
* C3. EWALD multi-k UKS *raises* on smearing while RKS supports it, 
  arbitrary asymmetry.
* C4. `level_shift` was entirely silent in the .out until today
  (now printed when non-zero); GDF's auto level-shift-warmup
  (5 cycles when shifted) is good prior art for strategy v2.
* C5. Existing scattered heuristics now centralized or to centralize:
  tight-cell test (GDF `pbc_gdf` L379 + `periodic_rijcosx` L99, 
  adopted by the classifier), smearing auto-resolver
  (`smearing/auto.py`, composed), BIPOLE KS FMIXING-30 in-driver
  default (reported by the resolver as a floor), GDF conducting-state
  gap warning at convergence (good v2 signal: re-classify post-SCF and
  WARN when the guess was wrong).

## 2. Symmetry-utilization matrix

| Route | IBZ k-reduction | reduced 1e integrals | reduced 2e / Fock | other |
|---|---|---|---|---|
| BIPOLE | ✓ (`monkhorst_pack(use_symmetry=True)` + `ir_mapping` expansion for J^LR) | ✓ auto, bit-exact (SYM2c atom-pair orbits; MgO invariance <1e-9) | opt-in Fock projection only (`use_fock_symmetry=True`; off by default, truncation-asymmetry, see CHANGELOG 2026-06-10) | spheropole/Ewald symmetric by construction |
| EWALD Γ + multi-k |, (none) | - | - | - |
| GDF Γ | n/a | ✓ conditional (`compute_*_lattice_reduced` when symmetry attached) | `symmetry_stabilize` (J/K projection) + `symmetry_reduce_fock` opt-ins | - |
| GDF multi-k |, (full mesh) | - | - | - |
| GPW / GAPW | ✓ multi-k (`KPoints.monkhorst_pack(symmetry=True)`) | - | - | - |
| RIJCOSX | no (multi-k route follows GDF full mesh) | - | - | Gamma RHF still n/a |

**Findings (symmetry):**

* S1. IBZ k-reduction exists in two independent implementations
  (BIPOLE `ir_mapping`, GPW `KPoints`) and is absent from the EWALD
  multi-k and GDF multi-k drivers, the single biggest cheap win for
  those routes (8× on cubic [2,2,2] full meshes, 48× asymptotically).
* S2. Symmetry-reduced **two-electron** builds (the real SYM3
  wall-clock goal) exist nowhere. The atom-pair-orbit machinery
  (`symmetry_lattice_c`) is the proven foundation; the J/K builders
  would need orbit-aware ERI loops in C++. Large project; the
  reduced-1e path shows the transformation layer is sound.
* S3. The BIPOLE Fock-symmetrization lesson generalizes: *operators
  assembled from truncated lattice sums are not exactly
  orbit-symmetric*, any symmetry enforcement on truncated J/K
  reshuffles truncation error. Symmetry must be used to *skip
  redundant computation of exactly-equivalent quantities* (integrals,
  k-points), not to project nearly-symmetric operators.

## 3. Efficiency findings

* E1. **Multi-k BIPOLE/EWALD Fock build dominates** (2026-05-25
  py-spy: 99.5 % of multi-k iteration wall in C++
  `build_fock_2e_real_space`). The EWALD multi-k UHF builder makes
  **6 lattice-ERI calls per iteration** (2×J_full + 2×F_full per spin
  via the K = 2(J−F) extraction; `periodic_uhf_multi_k_ewald.py`
  L174-213). Fusing J+K into one traversal (a `build_jk` that returns
  both) is the top per-iteration win, route-wide.
* E2. **Per-iteration rebuild waste:** EWALD Γ rebuilds the full Ewald
  J machinery each iteration; GDF caches Lpq correctly (good model);
  BIPOLE caches the J^LR FT (good); the BIPOLE SYM3b `_fock_sym_map`
  AO-rotation caches are rebuilt per `symmetrize_fock_blocks` call
  when opted in (minor, opt-in only).
* E3. **Python-FT pin**: `VIBEQC_AOPAIR_FT_BACKEND=python` (the
  deterministic test pin) is catastrophically slow on production-size
  cells (2 h vs 20 min observed on the same suite 2026-06-10). Tests
  that don't need bit-determinism should drop the pin.
* E4. GDF's conducting-state warning, energy-sanity guard, and
  ionic-cell guard are cheap post-SCF checks the other routes lack.

## 4. Action queue (priority-ordered)

| # | Action | Owner / status |
|---|---|---|
| 1 | `convergence="auto"` resolver + BIPOLE wiring + transparency block | **landed 2026-06-10** (this audit's companion commit) |
| 2 | Adopt the resolver on the GDF routes | **done 2026-06-10**, `convergence="auto"` active on jk_method="gdf" (no BIPOLE-style KS FMIXING floor; GDF drivers honor resolved values verbatim) |
| 3 | IBZ k-reduction, **2026-06-10 finding: the BIPOLE "IBZ-native" shortcut was broken on non-trivial crystals** (replicated D(k) without the star AO rotations; MgO [2,2,2] IBZ: non-convergent, 8.25 Ha off; He-only validation hid it). BIPOLE now expands IBZ inputs to the full mesh up front (correct, no savings; MgO full≡IBZ regression pinned). True IBZ-native transport is OPEN: `periodic_k_symmetry` carries the validated star-matching + probe record, P·D·Pᵀ exact for a subset of star ops, residual unexplained by G-phases/shift-phases/parity/time-reversal; suspected truncated-sum Fock asymmetry interaction. EWALD/GDF multi-k IBZ blocked behind the same question. | groundwork landed; transport question escalated |
| 4 | Fused J+K, **done at the call sites (2026-06-10)**: the fused C++ `build_jk_2e_real_space` already returned J and K from one traversal; the EWALD multi-k UHF/UKS builders now use it instead of the 2-traversals-per-spin `K = 2(J_full − F_full)` reconstruction (6 → 4 lattice-ERI passes per UHF Fock build; operators pinned bit-identical). **New finding while validating:** the RHF and UHF multi-k EWALD drivers disagree by 9.3e-4 Ha on closed-shell H₂ [1,1,1] with or without screening, pre-existing (different J machinery: `build_periodic_fock_ewald3d_k` vs J_SR+grid-Poisson J_LR); xfailed with diagnosis, EWALD-owner item. Also restored the UHF multi-k smearing rejection gate (was silently ignoring smearing_temperature). **2026-06-11: J-consistency CLOSED**, the J split was the b4a6faba merge-drop (UHF reverted to the pre-f7ee5832 split J); the merge-drop restoration routes both drivers through `ewald_3d_j_blocks` again, the fused-K call sites are kept on top, and `test_closed_shell_uhf_matches_rhf_at_111_mesh` is un-xfailed (strict-XPASS confirmed UHF ≡ RHF). | done (J-consistency closed 2026-06-11) |
| 5 | GPW/GAPW honor (or honestly reject) the runner SCF options | GPW chat (flagged) |
| 6 | EWALD multi-k UKS smearing parity with RKS | unowned, small |
| 7 | Post-SCF re-classification warning (gap vs assumed profile, GDF's conducting-state check generalized; auto-strategy v2 signal) | with strategy v2 |
| 8 | Strategy v2: guess-gap refinement from the first diagonalization (drivers know it for free), level-shift warmup adoption, open-shell profiles | with maintainer feedback on v1 |

## 5. Known open physics issues (not efficiency)

* The BIPOLE MgO absolute-parity bug (+2.5/+4.4 Ha vs PySCF at matched
  mesh; spheropole-dominant), escalated, see
  `handovers/HANDOVER_BIPOLE_PRODUCTION.md` §0a 2026-06-10 and
  `docs/troubleshooting.md`.
* Γ-only compcell GDF on tight ionic cells (documented +576 Ha
  non-physical guard; multi-k GDF is the production route there).
* RSGDF 3D (~231 mHa over-binding, architectural, guarded).
