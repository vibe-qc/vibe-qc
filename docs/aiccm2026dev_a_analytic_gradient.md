# Analytic WSSC gradient (Γ-CCM forces) — design + implementation spec

**Status:** design complete + feasibility confirmed (2026-06-26). Implementation
pending (intricate, all-or-nothing against the finite-difference gate). Target
file `python/vibeqc/periodic/ccm/gradient_analytic.py`; gate
`ccm_numerical_gradient` (`properties.py`).

## Goal

Analytic `dE_total/dR` per unit-cell atom for the HF-CCM, matching the existing
finite-difference `ccm_numerical_gradient` (which displaces each unit-cell atom
±h, propagating to **every** periodic image — the cyclic constraint). Default
method `"aiccm2026dev-a"` (the bare symmetric four-center `ccm_eri_symmetric`),
RHF first.

## Feasibility — confirmed Python-only, NO new C++

The blocker looked like the 2-e term: the molecular `two_electron_gradient_
contribution(basis, mol, D)` builds the HF tensor `Γ = ½DD − ¼DD` **internally**,
so the WSSC weights can't be woven in. **Resolution:**
`two_electron_gradient_casscf(basis, mol, gamma_ao_flat)` (`cpp/src/gradient.cpp:1473`,
bound at `bindings.cpp:2840`) contracts `d(μν|λσ)/dR` with an **arbitrary** AO
2-particle density `Γ_μνλσ` (Helgaker Eq. 12.5.7, row-major flat). That is exactly
the general-Γ ERI-gradient we need — feed it the WSSC-weighted `Γ^CCM` scattered
onto the padded layout. Dense `n_pad_ao**4` → small / 1-D / 2-D path (same regime
as the gate and `ccm_eri_symmetric`).

## Term structure

`dE/dR = dE_nn/dR + Σ P^CCM·dh^CCM/dR + Σ Γ^CCM·d(eri^CCM)/dR − Σ W^CCM·dS^CCM/dR`,
each integral being the WSSC-folded padded integral, so each derivative is the
**same fold applied to the padded integral derivative**, then **image-summed**
(`grad[A] = Σ_{p : pad_atom_of[p]=(A,g)} grad_pad[p]`, from
`PaddedCluster.pad_atom_of`).

The CCM densities on the home (nbf) basis from a converged `run_ccm_rhf`:
`P = 2·C_occ C_occᵀ`, `W = 2·Σ_i ε_i C_i C_iᵀ`, `Γ = ½P⊗P − ¼ (P⊗P) swapped`
(closed-shell HF 2-RDM, AO index order `μνλσ`).

### The fold adjoint (the crux)

Every CCM matrix is a **linear** fold of the padded integral: `M^CCM = L(M_pad)`
(`_fold_two_center` for S/T; `ccm_nuclear` for V; `ccm_eri_symmetric` for the
four-center). The energy term `Σ C^CCM · M^CCM = Σ C^CCM · L(M_pad) = Σ Lᵀ(C^CCM) ·
M_pad`. So the gradient term `Σ C^CCM · dM^CCM/dR = Σ Lᵀ(C^CCM) · dM_pad/dR` — feed
the molecular gradient routine the **adjoint-scattered** contraction coefficient
`C_pad = Lᵀ(C^CCM)` on the padded basis.

- **2-center adjoint** (P→P_pad, W→W_pad): mirror `_fold_two_center`
  (`padded.py:217`) — for each cell `g`, `M_pad[home_rows, cols_g] +=
  0.5·w_g[ao]·C^CCM`; then symmetrise (`+ .T`) because the fold does
  `0.5(out+out.T)`.
- **4-center adjoint** (Γ→Γ_pad): mirror `ccm_eri_symmetric` (`padded.py:361`)
  loop **exactly** (symmetric bridge `¼(w_ar+w_br+w_as+w_bs)`, independent
  minimum-image fold over `(g_c, g_e)`, closing transpose) — but **scatter**:
  `Γ_pad[home_a, cols_b(g), cols_c(g_c), cols_e(g_e)] += w4 · Γ^CCM[a,b,c,e]`
  instead of `eff += w4 · eri_pad[...]`. The closing `0.5(eff + eff.transpose(2,3,0,1))`
  adjoints to symmetrising `Γ^CCM` the same way before the scatter.
- **V_ne (hardest)**: `ccm_nuclear` (`padded.py:247`) weights each **nucleus image**
  `c=(C0,g_c)` by `ω_{A,c}` and folds the bra/ket pair too. The gradient has (i) the
  bra/ket basis-center derivatives (fold like the 2-center adjoint) AND (ii) the
  **Hellmann-Feynman nucleus-position** derivative `∫φ ∂(−Z/r_c)/∂R_{C0} φ`,
  weighted by `ω_{A,c}` and image-summed onto the unit-cell atom `C0` belongs to.
  Build V's contribution per nucleus image with `compute_nuclear` derivative on a
  single-charge `Molecule` (mirroring `ccm_nuclear`'s per-`c` construction), or via
  one padded `Molecule` carrying the weighted effective charges.

### Nuclear repulsion `dE_nn/dR`

`ccm_nuclear_repulsion` (`padded.py:437`) — its derivative is the per-cell
lattice-summed `−Σ Z_A Z_B (R_A−R_B)/|·|³` over the padded/WSSC nuclear pairs;
reuse `nuclear_repulsion_gradient` on the padded nuclear arrangement + image-sum,
or differentiate `ccm_nuclear_repulsion`'s own pair list directly (simplest,
isolated, finite-diff-gateable first).

## Build order (each step finite-diff checkable)

1. `dE_nn/dR` alone vs FD of `ccm_nuclear_repulsion` — isolated, simplest.
2. 1-e `Σ P·dh` (T + V) — V needs the HF-nucleus term; check vs FD of
   `Σ P^CCM·h^CCM(R)` at fixed P (Hellmann-Feynman part) then add Pulay.
3. Overlap Pulay `−Σ W·dS`.
4. 2-e `Σ Γ·d(eri)` via `two_electron_gradient_casscf` + the 4-center adjoint.
5. Total vs `ccm_numerical_gradient` on H₂ chain (2,1,1)/(4,1,1) STO-3G to ~1e-6;
   per-cell forces sum ≈ 0 (translational invariance, built-in check).

## Risks / notes

- All-or-nothing: the total only matches FD once **every** term + fold adjoint is
  exact. Build + check term-by-term (step list) to localize errors.
- `ccm_eri` (eq-18) is not bit-exact cyclically (`padded.py:311`); the default
  `ccm_eri_symmetric` is 8-fold symmetric in 1-D/2-D — gate on those.
- Dense `n_pad_ao**4`: 1-D/2-D/small only (the `_check_padded_eri_size` guard
  applies). 3-D analytic forces would need the scalable weighted ERI-gradient
  kernel in C++ (a `build_jk_ccm_weighted` sibling) — out of scope here.
- UHF gradient: `two_electron_gradient_contribution_uhf` / a casscf-style general-Γ
  per spin; mirror once RHF lands.
