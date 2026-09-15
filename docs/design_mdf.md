# Design: Mixed Density Fitting (MDF) for all-electron µHa periodic GDF

**Status:** **Γ + multi-k MDF complete + validated (RHF/UHF/UKS),
2026-06-14, GDF chat.** All 5 increments landed + green on `main`: cderi
builder `build_lpq_mdf` (Gaussian fit + PW residual + the Eq-23 G=0 term
`V̄_P = −π^{5/2}·(A@m_fused)`), `gdf_method='mdf'` wired through
`run_pbc_gdf_{rhf,uhf,uks}` (Γ) and `run_k{rhf,uhf,uks,rks}_periodic_gdf`
(multi-k, via `build_lpq_bloch_mdf`) with complex-aware J/K; the Ne-in-box
all-electron gate (MDF mesh-converged by ke≈60; matches PySCF MDF to
0.14 mHa; rsgdf ke=200 ~190 mHa off); H2 kmesh=(2,1,1) = published KRHF to
21 µHa; and `run_periodic_job` `gdf_method`/`mdf_ke_cutoff` threading.
**The G=0 term is monopole-only and validated EXACT** (H2 gap−monopole =
7.5e-9; compensated aux have no dipole), there is no higher-multipole
term; the small residual is a mesh/aux convergence knob, not missing
physics (§4). MDF is complete. See §4 + HANDOVER §5.
**Goal:** close the all-electron / heavy-atom GDF accuracy floor. Pure
Gaussian DF (compcell / rsgdf) cannot represent steep core densities to
µHa at a tractable plane-wave cutoff, MgO/sto-3g lands −515 mHa at
ke=200, and brute-force µHa needs ke≈6400 (hours/SCF). MDF reaches
all-electron µHa at a *modest* mesh by handling the steep part in real
space (exact libint Gaussian integrals) and only the smooth residual via
plane waves.

**Source:** Sun, Berkelbach, McClain & Chan, *J. Chem. Phys.* **147**,
164119 (2017), doi:10.1063/1.4998644, §II A-C, Appendices A/B. PDF in
`library/pbc/2017_sun_gaussian-plane-wave-mixed-density.pdf`. PySCF
reference implementation: `pyscf.pbc.df.MDF`.

---

## 1. The method (paper → vibe-qc terms)

MDF expands each periodic AO pair density (Eq 4) in **compensated
Gaussians + plane waves**:

    ρ_μν(r) ≈ Σ_Q φ_Q(r) d_{Q,μν}  +  Σ_{G≠0} e^{iG·r}/√(NΩ) c_{G,μν}  +  ρ̄_μν

* `φ_Q = χ_Q − ξ_Q` (Eq 5): a compact fitting Gaussian χ_Q minus a smooth
  compensating Gaussian ξ_Q, carrying zero net charge + zero multipoles
  (Eq 8). **This is exactly vibe-qc's compcell "fused" basis**, 
  `make_compensating_basis` / `make_fused_basis` / `fuse_transform_matrix`
  already build the χ_Q − ξ_Q combination.
* `ρ̄_μν = S_μν/Ω` (Eq 9): the G=0, k_μ=k_ν constant (overlap term).

The periodic ERI (Eq 22, Γ specialisation, real AO pairs so ρ(−G)=ρ(G)*):

    W_μν,κλ = Σ_i L_{i,μν} L_{i,κλ}              ← Gaussian part (factorised)
            + Σ_{G≠0} coul(G) ρ_μν(G) ρ_κλ(G)*  ← PW residual

with `coul(G) = 4π / (Ω |G|²)`, `ρ_μν(G)` the AO-pair FT, and (Eq 23)

    L_{i,μν} = Σ_P t_{Pi} [ T_{P,μν} − V̄_P ρ̄_μν
                           − Σ_{G≠0} coul(G) ρ_P(−G) ρ_μν(G) ]

where:
* `T_{P,μν} = (φ_P | μν)`, the **real-space** 3c Coulomb integral on the
  fused basis (libint, `compute_3c_eri_lattice`). Exact for steep cores, 
  no mesh. **vibe-qc already builds this** (compcell `T_fused`).
* `ρ_P(G)`, FT of the compensated fitting function φ_P
  (`rsgdf_aux_fourier_transform` on the fused basis; the χ−ξ FT).
* `ρ_μν(G)`, AO-pair FT (`_rsgdf_dense_pair_ft` /
  `ao_pair_fourier_transform_bloch`).
* `V̄_P` (Eq 14): `π/α_χ − π/α_ξ` for s-type fitting functions at G=0,
  else 0. The G=0 self-term.
* `t`, the J̃-orthogonaliser (Eq 19-21) that removes Gaussian↔PW linear
  dependence. `t† J̃ t = 1` with the **PW-dressed 2c metric** (Eq 20):

      J̃_PQ = M_{PQ} − Σ_{G≠0} coul(G) ρ_P(−G) ρ_Q(G)

  `M_{PQ} = (φ_P | φ_Q)` is the real-space 2c Coulomb metric (compcell
  `M_fused`, libint). Diagonalise J̃, drop eigenvalues < `thr·max`
  (default `1e-9`, Eq §II B), `t = U_keep / √λ_keep`.

**Reading of the structure.** The Gaussian L is *the compcell 3c tensor
with the PW-projection subtracted and the PW-dressed metric*. The PW
residual is what the Gaussians cannot represent, the part that makes it
all-electron-exact. Pure GDF/compcell = the Gaussian part with NO PW
residual *and* the bare (un-dressed) metric; that is its accuracy floor.

## 2. Building blocks (all already in the tree)

| Need | Existing |
|---|---|
| fused basis χ−ξ + A transform | `make_modrho_aux_basis`, `make_compensating_basis`, `make_fused_basis`, `fuse_transform_matrix` |
| real-space `M_{PQ}` = (φ_P|φ_Q) | `compute_2c_eri_lattice(fused, …)` |
| real-space `T_{P,μν}` = (φ_P|μν) | `compute_3c_eri_lattice(ao, fused, …)` |
| `ρ_P(G)` fused-basis FT | `rsgdf_aux_fourier_transform(fused, G)` |
| `ρ_μν(G)` AO-pair FT | `_rsgdf_dense_pair_ft` / `ao_pair_fourier_transform_bloch` |
| dense G-mesh, `coul(G)` | `rsgdf_dense_g_mesh`, `4π/(Ω|G|²)` |
| eigendecompose/threshold | the shared pattern in `build_lpq_*` |

**Net new code is small:** the J̃ assembly (M − PW projection), the
t-orthogonalise, the L assembly (T − V̄ρ̄ − PW projection, then t), and
the PW residual handling in J/K. Most pieces are wiring existing FTs.

## 3. cderi representation + J/K

`W = L·L (real, n_kept_gauss) + Σ_G coul ρ_μν(G) ρ_κλ(G)* (complex PW)`.

The Gaussian L slots straight into the existing real Lpq J/K
(`_build_j_from_lpq` / `_build_k_from_lpq`). The **PW residual** is a
separate term:

* **PW-J** `J_μν += Σ_G coul(G) ρ_μν(G) ρ_D(G)*`, `ρ_D(G)=Σ_κλ ρ_κλ(G)D_κλ`
  - this is the analytic-FT Hartree vibe-qc already has
  (`compute_j_ewald_3d_ft_gamma` / `build_j_ewald_3d`); reuse it on the
  same mesh. Cheap (density FT).
* **PW-K** `K_μν += Σ_G coul(G) Σ_κλ ρ_μκ(G) D_κλ ρ_νλ(G)*`, the FFT-style
  exchange. New; O(n²·n_G) per iteration. The expensive piece; the AO-pair
  FT `ρ_μκ(G)` is mesh-cached (density-independent), so per-iter cost is
  the two contractions with D.

The exxdiv Madelung treatment carries over from the existing GDF K.

## 4. Increment plan (land each green; PySCF-MDF-pinned)

1. **`build_lpq_mdf` Gaussian part + J̃/t, DONE (2026-06-14, 221211a5).**
   The L vectors + the PW residual cderi. Verified
   (`tests/test_pbc_gdf_mdf.py`): the no-PW limit (`mdf_ke_cutoff<=0`) is
   **bit-identical** to `build_lpq_compcell(apply_aft_correction=False)`;
   PW-on gives a Hermitian + PSD reconstructed W; the PW-dressing
   re-ranks the aux space (36→26 on H2/def2-svp-jk, the Eq 19-21
   linear-dependence removal).

2. **PW residual + the Eq-23 G=0 term `V̄ρ̄`, DONE (2026-06-14).** The
   real-space libint 3c `T` and the reciprocal PW projection differ by a
   *constant, ke-independent* G=0 term (`4.87e-3` on H2/def2-svp-jk; the
   2c metric matches to 1e-12, only the 3c carries G=0, since the AO
   pair has `ρ_μν(0)=S/Ω≠0`). Omitting it blows up the L vectors as the
   mesh tightens. The closed form is the **Ewald-regularised** second-
   moment term (NOT the naïve Fourier limit `(4π/Ω)lim ρ_P/|G|²`, off by
   ~10⁴): `V̄_P = −π^{5/2}·(A @ m_fused)`, `m_fused[f] = Σ_k c_{f,k}
   α_{f,k}^{-5/2}` over s-type fused functions, `ρ̄_μν = S_μν/Ω`. The
   `−π^{5/2}` derives as `−(2π/3)·(3/2)π^{3/2}` and matches the exact
   `T_real − T_recip(dense)` gap to 1e-5 (constant to machine precision
   across aux). Verified: reconstructed W matches rsgdf to **1.6e-4 and
   is mesh-stable** (ke=20/40/80). *(The naïve `α^{-3/4}` charge weight
   tried first was 2× off on contracted shells, the right weight is the
   un-normalised primitive charge ∝ `c α^{-3/2}`, giving the `α^{-5/2}`
   second moment above.)*
3. **Wire J/K, DONE (2026-06-14).** Combined complex cderi
   `[L_gauss; cderi_pw]` through complex-aware `_build_{j,k}_from_lpq`
   (`iscomplexobj` guard ⇒ real path bit-identical, 42 GDF tests green).
   `gdf_method='mdf'` + `mdf_ke_cutoff` (default 40 Ha) in
   `_pbc_gdf_gamma_setup` / `run_pbc_gdf_{rhf,uhf,uks}`. H2 RHF/MDF
   within 45 µHa of rsgdf.
4. **All-electron gate, DONE (2026-06-14).** MgO-at-Γ is NOT usable (the
   documented Γ-only tight-ionic instability diverges compcell too). The
   valid Γ-stable gate is a **heavy atom in a vacuum box**: Ne/sto-3g/
   10-bohr, MDF mesh-converged by ke≈60, matches PySCF MDF (matched aux
   `def2-svp-jk`==`def2-svp-jkfit`) to **0.14 mHa**, while rsgdf ke=200 is
   ~190 mHa over-bound (the floor MDF removes). Plus triplet H2 UHF/MDF =
   PySCF UHF Γ to 9.5 µHa. `tests/test_pbc_gdf_mdf.py` (7, the Ne one
   `@slow`).

   **Fail-closed on the compact-dense-core class (2026-07-29, G-GDF-001
   MDF sub-gate).** The "MgO-at-Γ NOT usable" note above is now
   enforced, not just documented: measured on MgO primitive FCC/STO-3G,
   Γ mdf is non-convergent and trial-dependent at ke=40/60
   (+1703/+2143/+418 Ha) and *falsely converges* at ke=80 to −1651 Ha
   (PySCF MDF: −271.0499, evading the energy-sanity guard), and multi-k
   (1,1,2) mdf lands −11741 Ha non-converged. All five GDF drivers now
   raise on the compact-dense-core class (Z≥8 in <250 bohr³/atom;
   `pbc_gdf._reject_dense_core_mdf`). In the same closure the
   dense-core classifier stopped mis-tagging MDF's *validated* envelope:
   the tight-basis/unresolved-reciprocal-mesh hold is an rsgdf concept,
   so the Ne-in-box gate above no longer carries `+PARITY_HELD` (its
   energy is unchanged, pinned). Regressions:
   `test_dense_core_mdf_fails_closed_gamma_and_multik`,
   `test_classifier_mdf_not_held_on_tight_basis_vacuum_box`,
   `test_ne_box_mdf_backend_untagged`.
5. **Multi-k + open-shell, DONE (2026-06-14).** Open-shell Γ (UHF/UKS)
   via the shared setup + complex per-spin J/K. Multi-k via
   `build_lpq_bloch_mdf` (the per-(k_bra,k_ket) cderi: q-only real-space
   Gaussian, only steep/local aux survive J̃, so q-only ≈ ket-resolved, 
   + ket-resolved PW residual on `{G+q}` + V̄ρ̄ only on diagonal pairs)
   wired into `run_k{rhf,uhf,uks,rks}_periodic_gdf`. H2 kmesh=(2,1,1) =
   published KRHF (−1.12013988) to 21 µHa; `build_lpq_bloch_mdf(0,0)` ≡ Γ
   to 3.6e-5.

**`run_periodic_job` pass-through, DONE (2026-06-14).** `gdf_method`
(default `None` = unchanged behavior) + `mdf_ke_cutoff` threaded through
the runner to all GDF sub-paths; Γ closed-shell RHF with an explicit
`gdf_method` routes through `run_pbc_gdf_rhf`. (This surfaced a
pre-existing ~5.8 mHa Γ-RHF discrepancy between the legacy
`run_rhf_periodic_gamma_gdf` runner default and the PySCF-validated
`run_pbc_gdf_rhf`, flagged separately per §7, NOT an MDF issue.)

### The G=0 term is monopole-only, and that is EXACT (validated, no higher-multipole term)

An earlier reading assumed the ~1% MDF residual was a neglected
higher-multipole (dipole/quadrupole) G=0 term. A validation probe
(`examples/debug/mdf_g0_multipole_probe.py`) **disproves that**, the
monopole term is the *complete* G=0 self-term:

* **H2 / def2-svp-jk** (where the dense-mesh gap is mesh-converged to
  1.6e-13): `|gap|max = 4.871e-3`, monopole `|−π^{5/2}·A@m · S/Ω|max =
  4.871e-3`, and **`|gap − monopole|max = 7.5e-9`**, the monopole
  reproduces the entire G=0 term to 8 significant figures.
* **The compensated aux carry no dipole**: `|d_P|max = 4.7e-7 ≈ 0` (the
  χ−ξ compensation kills the low multipoles by construction). So the
  derived dipole candidate `(4π/3Ω)·d_P·D_μν` is ~1e-9 and does *not*
  reduce the residual, there is no dipole/quadrupole G=0 term to add.

So the ~1.6e-4 W-vs-rsgdf gap (and the 45 µHa H2 / 0.14 mHa Ne energy
residuals) is **not** a missing G=0 term, it is the **modest-mesh +
Gaussian-fit representation difference** between MDF and the dense-mesh
reference: the compensated-Gaussian fit leaves a residual whose high-G
content the *modest* PW mesh does not fully resolve. It is a **convergence
knob (`mdf_ke_cutoff`)**, not missing physics. (The probe's apparent 93%
Ne "residual" is an artifact: the *dense* reference mesh itself is not
converged for Ne's steep core, `|gap(400)−gap(200)| = 0.15`, so the Ne
gap can't be cleanly decomposed; the clean H2 evidence above stands.)

**Implication for sub-µHa all-electron:** push `mdf_ke_cutoff` (and/or a
steeper aux), a mesh/aux convergence study, not a new analytic term. The
monopole MDF is already a strong, complete result (all-electron floor cut
from ~1-9 Ha to ≤0.14 mHa).

## 5. Validation (out-of-process, CLAUDE.md §10)

`pyscf.pbc.df.MDF(cell)` SCF on MgO/Γ (all-electron, `exxdiv='ewald'`,
`cell.unit='B'`); compare vibe-qc `gdf_method='mdf'`. Also keep the LiH
light-atom µHa (MDF must not regress the cases GDF already nails).

**Measured target + floor (2026-06-14, out-of-process; MgO 2-atom /
sto-3g / Γ, default JK aux):**

    PYSCF GDF = -261.40073318 Ha
    PYSCF MDF = -270.67898594 Ha   ← target
    gap       =   9.28 Ha

The ~9 Ha GDF↔MDF gap (NOT the ~0.5 Ha the rsgdf-vs-PySCF-GDF probe
suggested, that compared two *Gaussian* DFs sharing the aux floor)
confirms the all-electron Gaussian-DF floor is large with a minimal aux:
the default JK aux for sto-3g has no functions steep enough for the Mg/O
1s cores, so the pure-Gaussian fit of the core pair density is badly
incomplete. MDF's PW residual fixes it. vibe-qc `gdf_method='mdf'` must
reach the MDF value; `rsgdf`/`compcell` sit on the floor. (NB: pin the
exact MgO geometry/aux in the test, the absolute numbers above are for
the scoping cell, recompute for the committed test cell.)

## 6. Citations (§8)

The GDF/rsgdf citation route already fires Sun-Berkelbach 2017 (the MDF
paper) for every GDF SCF, MDF is the *same* paper, so no new database
entry is needed; the inline Eq references above are the WHAT/HOW record
per CLAUDE.md §8. If MDF lands its own `gdf_method='mdf'` route, confirm
the existing `routes` row still fires (it keys on the GDF/rsgdf backend).
