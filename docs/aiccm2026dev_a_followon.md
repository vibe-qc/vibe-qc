# AICCM `aiccm2026dev-a` follow-on - derivation document

> **D89 construction-boundary notice (2026-07-15).** Γ-CCM
> (`aiccm2026dev-a`) is the union-and-weight/Wigner-Seitz integral-weighting
> construction; χ-CCM (`aiccm2026dev-b`) is the distinct
> finite-translation-group character construction. Neutral GDF/Bloch and
> real-Gamma routes are same-Hamiltonian representation controls only after a
> block-circulant finite Hamiltonian has been specified. They are not Γ-CCM
> production routes and do not prove Γ-CCM equals χ-CCM. In the historical text
> below, `CCM == SCM-Γ` and Fourier/Wannier statements apply only within the
> explicitly chosen Hamiltonian and occupied space. The geometry, localization,
> symmetry, and local-correlation measurements remain route-specific validation
> records; they are not cross-construction equivalence evidence.
>
> **D90 binding clarification (2026-07-16).** The neutral-only UCCSD(T) and
> DLPNO implementations described below are A-namespace neutral fitted-torus
> correlation controls. Their API names, module location, and solver ancestry
> assign neither Γ-CCM nor χ-CCM construction identity. Historical uses of
> "CCM UCCSD(T)" or "CCM DLPNO" below name that cyclic-cluster control stack,
> not the union-and-weight construction.

Math + validation record for the three follow-on features on the
`aiccm2026dev-a` cyclic-cluster line: **(1) localization**, **(2) DLPNO natural
orbitals**, **(3) space-group symmetry**. Each sits behind its own option and
does not change existing behavior. Decisions + open-questions log:
[Archived `handovers/HANDOVER_AICCM_FOLLOWON.md`](https://vibe-qc.com/docs/).

Every numerical claim here is grounded in a real run (test file named per
section). SI/atomic units (bohr, hartree).

---

## Interaction-range parameterization - the real-space dual of k-point sampling

### A.1 The object

Historically the cluster is given as an explicit supercell mesh
`nrep = (N₁, N₂, N₃)`. The physically meaningful knob, however, is *how far the
interactions reach* - a real-space cutoff `R_c`. The finite translation group
for a cluster of size `nrep` has a geometrically dual Γ-centred
`(N₁,N₂,N₃)` character mesh. Choosing the cluster by a real-space radius is
therefore a mesh-density parameterization, without asserting that two different
construction maps produce the same Hamiltonian. We let the user supply the unit
cell + `R_c` and derive the minimal `nrep`.

### A.2 What "reach `R_c`" means geometrically - the WS inscribed sphere

In the CCM, atom A interacts with the minimum image of atom B inside A's
Wigner-Seitz supercell (WSSC). The set of displacements captured *at full weight,
in every direction* is the largest sphere inscribed in the WS cell. So "the
interactions reach `R_c` around every atom" is precisely

```
r_in(L_c) ≥ R_c ,   L_c = {N_i a_i} the cluster lattice.
```

**The inscribed-sphere radius equals half the shortest lattice vector,
`r_in = λ₁/2`.** Proof: `x` lies in the WS (Voronoi) cell iff `x·R ≤ |R|²/2` for
every lattice vector `R`; for `|x| = r` the binding constraint is along `R̂`,
giving `r ≤ |R|/2`, minimised by the shortest `R`. The nearest Voronoi face is
the perpendicular bisector of the shortest lattice vector. Hence the condition is

```
λ₁(L_c) ≥ 2 R_c .
```

### A.3 The per-direction formula and why it is rigorous (not a guess)

Let `d_i = V/|a_j × a_k|` be the unit-cell interplanar spacing along `a_i` (the
spacing between successive `(a_j,a_k)` planes; `d_i = 1/|b_i|` with `b_i` the
2π-free reciprocal vector). The cluster interplanar spacing is `D_i = N_i d_i`.
Project any nonzero cluster vector `R = Σ_i m_i N_i a_i` onto the unit reciprocal
direction `b̂_i`: since `a_i·b̂_i = d_i` and `a_j·b̂_i = 0` (j≠i),

```
|R| ≥ |R·b̂_i| = |m_i| D_i   ⟹   λ₁(L_c) = min_{R≠0}|R| ≥ min_i D_i .
```

Therefore choosing

```
N_i = ⌈ 2 R_c / d_i ⌉      ⟹   D_i ≥ 2R_c ∀i   ⟹   λ₁ ≥ 2R_c   ⟹   r_in ≥ R_c .   ✓
```

The bound `λ₁ ≥ min_i D_i` makes the per-direction formula a **sufficient
condition** for the inscribed-sphere guarantee - it can never under-shoot. It is
**tight** (equality `λ₁ = min_i D_i`) for orthogonal cells, and conservative for
oblique cells (where `λ₁` can exceed `min_i D_i`, so the achieved `r_in` is
slightly larger than `R_c` - never smaller). This is the rigorous content behind
the heuristic `N_i ≈ ⌈2R_c/d_i⌉`.

### A.4 The duality constant

A cyclic cluster of WS "diameter" `2 R_c` is a BvK torus, so its allowed k-points
are spaced by `Δk = 2π/(2R_c) = π/R_c` (2π reciprocal convention). Equivalently,
per direction, `N_i = ⌈2R_c/d_i⌉ = ⌈|b_i|/(π/R_c)⌉ = ⌈|b_i|/Δk⌉` - exactly the
uniform-k-density rule. So:

```
real-space cutoff R_c   ⟺   k-spacing Δk = π/R_c   ⟺   cluster (N₁,N₂,N₃) ≡ MP mesh (N₁,N₂,N₃).
```

Larger `R_c` ⇔ denser k-mesh. A radius scan is the CCM analogue of a k-point
convergence study.

### A.5 Implementation + validation

Geometry (`vibeqc.periodic.ccm.wigner_seitz`): `interplanar_spacings`,
`shortest_lattice_vector_length`, `wsc_inscribed_radius`,
`nrep_for_interaction_range`, `kspacing_for_interaction_range`. API
(`CCMSystem`): `interaction_range=` (bohr) / `interaction_range_ang=` (Å), the
`from_interaction_range(unit, R_c, basis, units=...)` constructor, and the stored
diagnostics `interaction_range`, `wsc_inscribed_radius`, `kspacing_equiv`;
explicit `nrep` remains the advanced override. Scan helper
(`ccm_interaction_range_scan`) → `InteractionRangeScan` (radius → energy/atom,
with `converged_radius(tol)` and JSON `as_records()`).

Validated (`tests/test_ccm_interaction_range.py`):

* `shortest_lattice_vector_length` / `wsc_inscribed_radius` match a brute-force
  ±6-shell oracle to ≤1e-9 on cubic / ortho / tetragonal / fcc-prim / bcc-prim /
  hexagonal / triclinic / vacuum-chain lattices.
* **Inscribed-sphere gate** `r_in(L_c) ≥ R_c` *and* the per-direction bound
  `min_i N_i d_i ≥ 2R_c` hold for every (lattice, R_c) in the battery above -
  the numerical proof of A.3, not an assertion.
* Orthogonal bound is tight: ortho(3,5,7) at R_c=7 → `nrep=(5,3,2)`.
* `from_interaction_range` and explicit `nrep` give bit-identical `S^CCM` and HF
  energy; the Å convenience matches bohr; ambiguous specs raise.
* H₂-chain (6-bohr cell, STO-3G) radius scan converges monotonically; the dual
  k-mesh per point equals the cluster mesh.

---

## Task 1 - Localization of the canonical crystalline orbitals (Wannier)

### 1.1 The object

A converged Γ-CCM SCF returns occupied orbitals `ψ` for its selected finite
Hamiltonian on the `N = N₁N₂N₃`-cell supercell. They are
`S^CCM`-orthonormal:
`Σ_{μν} C†_{μi} S^CCM_{μν} C_{νj} = δ_{ij}`.

### 1.2 Localization is a unitary rotation within the occupied space

The textbook cell-periodic Wannier construction is the discrete back-transform
over the cluster net,

```
w_n(r − R) = (1/N) Σ_k e^{−i k·R} ψ_{n,k}(r),     k ∈ equivalent net,  R ∈ L_c,
```

which produces `N` translationally-equivalent Wannier functions per band. The
block-Fourier matrix `U_{(nk),(mR)} = (1/√N) e^{−ik·R} δ_{nm}` is **unitary**, so
the Wannier set spans the *same* occupied space as `{ψ_{n,k}}`. For one already
specified block-circulant finite Hamiltonian, the real-supercell occupied set
and its character blocks are related by that unitary. Localizing the
real-supercell occupieds therefore chooses a localized gauge within the same
specified occupied space, with no explicit k-sum. This conditional same-H
statement does not equate Γ-CCM with χ-CCM or a neutral GDF control.

Because the rotation `C_loc = C_occ U` stays inside the occupied space, the
one-particle density `P = Σ_i^{occ} C_i C_i†` is invariant, and with it every
ground-state observable, including the total energy. This is the hard
correctness gate.

### 1.3 Gauge fixing (spread minimization)

The Marzari-Vanderbilt spread functional is
`Ω = Σ_i [⟨w_i|r²|w_i⟩ − ⟨w_i|r|w_i⟩²]`. Two gauges are offered, reusing the
validated molecular/periodic localizer
(`vibeqc.periodic_localise.localise_periodic_gamma`):

* **Pipek-Mezey** - maximize `Σ_i Σ_A (Q_i^A)²` (Mulliken populations). No
  position operator ⇒ **PBC-safe in every regime**, including orbitals that wrap
  the cluster boundary. Default.
* **Foster-Boys** - maximize `Σ_i ⟨w_i|r|w_i⟩²` via the home-cell dipole
  integrals. Faithful (with the reported centroids/spreads) only for orbitals
  that do **not** wrap the boundary (molecule-in-a-box / dilute / large cell).

The reported `centroids = ⟨w_i|r|w_i⟩` and `spreads = ⟨w_i|r²⟩ − ⟨w_i|r⟩²` use
the plain (home-cell) position operator, so for a dense cell they **fold into the
home cell** (a Wannier on cell `R` reports its in-cell center). The Resta
periodic operator `⟨μ|e^{−iG·r}|ν⟩` that would unwrap them is an open item
(§ decisions log); Pipek-Mezey localization itself is unaffected.

### 1.4 Implementation + validation

`vibeqc.periodic.ccm.localise_ccm(ccm_result, ccm, method=...)` →
`PeriodicWannierResult` (`C_loc`, `U`, `centers`, `centroids`, `spreads`,
`charges`, `localization`). `localization_density_residual(ccm_result, w)` is the
gate `‖P_loc − P_canonical‖_F`.

Validated (`tests/test_ccm_localize.py`), dense 1-D H₂ chain (6-bohr cell,
STO-3G, nrep (3,1,1), 3 occupied):

| quantity | Pipek-Mezey | Boys |
|---|---|---|
| density residual `‖P_loc−P_can‖` | 2.4e-16 | 3.0e-16 |
| unitarity `‖C_locᵀ S C_loc − I‖` | 6.7e-16 | 4.9e-16 |
| objective (initial → final) | 0.644 → 1.500 | 109.5 → 181.5 |
| spreads Ω_i (bohr²) | ~2.39 | ~2.39 |
| Wannier center z (bohr) | 0.7 (H₂ bond midpoint) | 0.7 |
| real `C_loc` (centrosymmetric) | yes | yes |

So the rotation is exactly density/energy-preserving (machine ε), genuinely
localizes, gives real orbitals for the centrosymmetric cell, and places the
Wannier centers on the H₂ bond midpoint - the expected Wyckoff/bond position.

### 1.5 Open (M2-M4, see decisions log)
Explicit multi-k `U(k)`/`M(k,b)` route; Resta operator (unwrapped
centroids/spreads for dense crystals); IAO/IBO path (Knizia); aliasing
diagnostics + spread-vs-cluster-size convergence; emit in the exact rep Task 2
(DLPNO) consumes.

---

## Task 3 - Space-group symmetry (spglib)

### 3.1 The cluster-invariant subgroup

spglib analyzes the **crystal** (unit cell) space group `G`, giving operations
`{W|w}` (integer rotation `W` in the unit-cell fractional basis + fractional
translation `w`). Each op is converted to Cartesian, `R = L_u W L_u⁻¹`,
`t = L_u w` (`L_u` = unit lattice, columns aⱼ). The operations usable on the
cyclic cluster are the **cluster-invariant subgroup** `G_c ⊆ G`: those `(R,t)`
that map the cluster lattice `L_c = {N_j a_j}` onto itself modulo `L_c`. For a
point op, `R(N_j a_j) = N_j Σ_k W_{kj} a_k ∈ L_c` iff `N_j W_{kj}/N_k ∈ ℤ` - so
for equal `nrep` in symmetry-related directions **all** crystal point ops survive;
unequal `nrep` keeps the compatible subset. Tested operationally by
`symmetry_ao.atom_permutation_under_op` on the supercell (it raises iff `(R,t)`
is not a cluster symmetry).

### 3.2 AO representation

For each `g ∈ G_c` the AO matrix is `P = (atom permutation) × (Wigner-D blocks)`
from `symmetry_ao.build_ao_permutation_matrix(basis, R, perm)` - real solid
harmonics, `wigner_d_real(ℓ, R)` per shell, with improper rotations factored
through the `ℓ`-parity. At Γ the fractional translation contributes phase 1, so
`P` (permutation + rotation) is the full cluster-AO representation.

### 3.3 Validation + a proven finding (`tests/test_ccm_symmetry.py`)

The AO maps are **correct**: `S^CCM` and `T^CCM` are invariant `Pᵀ M P = M` to
machine precision under every `g ∈ G_c`, including the non-symmorphic glide/screw
ops of diamond (Fd-3m):

| system | SG | crystal/cluster order | ‖PᵀSP−S‖ | ‖PᵀTP−T‖ | ‖PᵀVP−V‖ (rel) |
|---|---|---|---|---|---|
| MgO (Fm-3m) | 225 | 48 / 48 | 2e-15 | 1e-14 | 0.077 (4.6e-4) |
| C-diamond (Fd-3m, non-symmorphic) | 227 | 48 / 48 | 2e-15 | 2e-15 | 0.44 (7.9e-3) |
| NaCl (Fm-3m) | 225 | 48 / 48 | 5e-15 | - | 0.044 |

**Finding.** The union three-center `V^CCM` (bare-1/r, weight
`ω_μν·½(ω_μC+ω_νC)`) is **not** crystal-symmetric - ~5e-4 (MgO) → ~8e-3
(C-diamond) relative. S and T being exactly invariant isolate V as the sole
culprit (the maps are validated); the anchor-bridge breaks *spatial* symmetry
just as eq-18 broke *permutation* symmetry.

**Resolution (proven).** The **neutral Ewald `V_ne` IS crystal-symmetric**
(`‖PᵀVP−V‖/‖V‖` ≈ 2e-8, lattice-sum tolerance) - last rows below:

| nuclear operator | MgO rel breaking | C-diamond rel breaking |
|---|---|---|
| union three-center (bare-1/r) | 4.6e-4 | 7.9e-3 |
| **neutral Ewald `V_ne`** | **2.3e-8** | **5.0e-9** |

This is the **spatial analogue of the Madelung gap** (#16): bare-1/r union
weights break both permutation symmetry (four-center, fixed by §13's
`ccm_eri_symmetric`) and spatial symmetry (three-center V), and the neutral Ewald
kernel fixes both. So the symmetric Hamiltonian already exists - `h = T + V_ne`
with the neutral cderi four-center - and is the natural target for symmetry
exploitation.

### 3.4 Plan (M2-M3, decisions log)
Symmetry exploitation rides the **neutral / GDF route** (symmetric `h`): petite-
list symmetry-unique shell pairs/quartets → representative integrals → scatter
via `P` + symmetrize Fock; fold the k-net to the IBZ with weights; the exact
`E`/Fock == symmetry-off gate then holds because `h` is symmetric. (The bare-union
route stays available but its `h` is not crystal-symmetric, so its symmetrized
energy would differ - a documented limitation, not a target.)

## Task 2 - DLPNO natural orbitals (PAO → PNO)

Foundations now in place: localized occupieds (Task 1), the symmetric neutral
route + symmetry-unique pair orbits (Task 3).

### 2.1 Projected atomic orbitals (M1)

The local virtual basis is the AO space projected out of the occupied space
(Pulay; Riplinger-Neese §II.D):

```
C_PAO = (I − C_occ C_occᵀ S^CCM) · A ,
```

canonical-orthogonalized within a domain (eigenvectors of `Q S Q` above a
linear-dependence floor). The occupied **space** projector `C_occ C_occᵀ` is
invariant under occupied rotations, so the full-cell PAO set is identical whether
the canonical or the localized (Task 1) occupieds are used - the localized
occupieds enter at the pair-domain stage (M2). `ccm_pao(ccm_result, ccm)` reuses
`vibeqc.dlpno.pao`.

Validated (`tests/test_ccm_dlpno.py`), dense 1-D H₂ chain (STO-3G, nrep (3,1,1)):
PAOs S^CCM-orthogonal to the occupied space (`max|C_occᵀ S C_pao|` = 2.9e-16);
`n_pao = 3 = nbf − n_occ` (full virtual, no linear dependence); occupieds + PAOs
span the full space (rank 6/6).

### 2.2 DLPNO-MP2 (M2, DONE)

`ccm_dlpno_mp2(ccm, scf_result, *, cderi=, localize=, tcut_pno=, tcut_mkn=, …)`
→ `CCMDLPNOMP2Result`. It mirrors the validated molecular DLPNO-MP2
(`vibeqc.dlpno.mp2`), **reusing its component functions** (`build_atom_basis_map`,
`build_projection_matrix`, `select_domain_atoms_mulliken`,
`semicanonical_pao_basis`) with `S^CCM` / `F^CCM` injected and the `(ia|jb)`
integrals density-fit from the **neutral cderi** `L[P,μν]` (`B_half[P,i,ν] =
Σ_μ C_loc[μ,i] L[P,μν]`, so `K_ab = Σ_P B[P,i,a] B[P,j,b]`).

Pipeline per pair `(i,j)`: Mulliken domain (`tcut_mkn`) → semicanonical PAOs
(eigh of `F` in the domain, `S^CCM`-orthonormal) → first-order amplitudes
`T_ab = K_ab/(f_ii+f_jj−ε_a−ε_b)` → PNOs (eigvecs of the pair density
`½(TTᵀ+TᵀT)`, occupation-threshold `tcut_pno`) → quasi-canonicalize → coupled
LMP2 residual `R_ij = K_ij + (ε_a+ε_b)T_ij − Σ_k[F_ik P(T_kj) + F_kj P(T_ik)]`
with neighbour amplitudes projected via `S_pq = V_pᵀ S^CCM V_q`. For canonical
occupieds `F_oo` is diagonal and the residual converges in one step; for
localized occupieds the `S^CCM`-link coupling restores the canonical sum.

**Reference = neutral (per the `-b`-critique correction).** The Madelung
background shifts the occ-virt denominators, so bare-vs-neutral correlation do
*not* agree (§ decision log): the DLPNO-MP2 reference is the neutral SCF
(`run_ccm_rhf(ccm, eri=ccm_eri_neutral(ccm))`) + neutral cderi. No
Madelung-robustness is claimed.

**Validated (`tests/test_ccm_dlpno_mp2.py`, 11/11; H₂-chain STO-3G).**

| gate | result |
|---|---|
| no-truncation, canonical occupieds | `Δ = 0.0` vs canonical CCM MP2 (neutral) |
| no-truncation, Pipek-Mezey occupieds | `Δ = 8.5e-15` (coupled LMP2, 4 iters) |
| no-truncation, Boys occupieds | `Δ = 8.5e-15` |
| no-truncation, (4,1,1) = 10 pairs | `Δ ≲ 1e-13` (coupled iteration at scale) |
| PNO truncation `tcut_pno` 1e-4→1e-8 | monotone error 5e-6 → <1e-7; `e_pno_correction` tracks the gap |

The no-truncation limit reproducing canonical CCM MP2 to machine ε - for
canonical *and* localized occupieds - is the hard correctness proof (not an
assertion).

### 2.3 DLPNO-CCSD(T) (M3, DONE)

`ccm_dlpno_ccsd(ccm, scf_result, *, cderi=, g_neutral=, localize=, tcut_pno=,
tcut_mkn=, compute_triples=, n_frozen=)` → `CCMDLPNOCCSDResult`
(`vibeqc.periodic.ccm.dlpno_ccsd`). **Subspace-projected** DLPNO-CCSD(T): the
per-pair PNOs (`_build_ccm_pno_pairs`, shared with MP2) are merged into a single
**union virtual space** `V_trunc`, and vibe-qc's own CCM CCSD engine
(`ccsd.py` `_ccsd_iterate` + `_triples`) runs in `{occ, V_trunc}`.

* **Union space.** Stack the pair PNO coefficients into `W`,
  canonical-orthogonalize against `S^CCM` (eigvecs of `WᵀS^CCM W` above
  `pno_lindep`), then **semicanonicalize** (diagonalize the Fock within the
  space) → `V_trunc`, `ε_v`.
* **Why exact at no truncation.** The CCSD engine assumes a diagonal Fock, so we
  use the *canonical* occupieds (occ-occ block diagonal) and the semicanonical
  `V_trunc`. Every PNO is built from PAOs projected out of the full occupied
  space, so `V_trunc ⊥_S occ` and the occ-virt Fock block vanishes: the MO set
  `C_sub = [C_occ | V_trunc]` is orthonormal with a block-diagonal Fock. At
  `tcut_pno = tcut_mkn = 0` each pair's PNOs span the full virtual block, so the
  union does too - `V_trunc` is then a unitary rotation of the canonical
  virtuals, and CCSD(T) (invariant to occ/virt unitary rotations) equals the
  canonical CCM CCSD(T).
* **Reference = neutral** (the `g_eff = Σ_P L⊗L` four-center, per the decision
  log): both the SCF reference and the CCSD MO integrals.

**Validated (`tests/test_ccm_dlpno_ccsd.py`, 9/9; H₂-chain STO-3G).**

| gate | result |
|---|---|
| no-truncation, canonical occupieds | `Δtot = 7e-18`; CCSD *and* (T) each == canonical |
| no-truncation, Pipek-Mezey | `Δtot = 0.0` |
| no-truncation, Boys | `Δtot = 7e-18` |
| no-truncation, (4,1,1) | `Δtot = 3e-16` |
| PNO truncation | `n_vpno ≤ n_full`; (T)-toggle / total consistency |

The no-truncation == canonical CCM CCSD(T) limit (CCSD + (T), all gauges) is the
hard correctness proof.

**Scope note (honest).** The union-PNO space truncates the *global* virtual
space, so it shrinks only when a virtual direction is unimportant to **every**
pair; on the small, well-coupled validation chains the union stays
full-dimensional (correct, not a bug). Per-pair-coupled DLPNO-CCSD (separate
per-pair PNO spaces, the more aggressive truncation) is the M3b refinement.

### 2.4 Open (M3b-): per-pair-coupled DLPNO-CCSD; symmetry-unique pairs
`(i, jR)` from Task 3's orbit reduction; open-shell U-DLPNO (Task D). Refs:
Nejad et al. 2025; periodic DLPNO literature.

## Task C - Derivable properties on the BvK torus

`vibeqc.periodic.ccm.properties`. Which one-particle observables are well defined
on a neutral finite Born-von-Karman torus, and which need care - implemented in
priority order of how cleanly they are defined.

### C.1 HOMO-LUMO / fundamental gap (well defined)

`ccm_homo_lumo_gap(scf_result, ccm)` → `CCMGap`. It reports
`gap = ε[n_occ] − ε[n_occ−1]` for the selected finite Hamiltonian. A
block-circulant Hamiltonian has the same spectrum in its real-supercell and
character representations, but a bulk-gap interpretation requires
route-specific convergence. Validated: equals the raw-spectrum gap to machine
ε; H₂-chain (3,1,1) finite-cluster gap 1.14 Ha.

### C.2 Mulliken / Löwdin populations (well defined)

`ccm_mulliken_charges` / `ccm_lowdin_charges` → `CCMPopulation`. Basis-local
partitions of the cluster density with the cyclic overlap `S^CCM`:
`q_A = Z_A − Σ_{μ∈A}(D S^CCM)_μμ` (Mulliken) and the `S^{1/2} D S^{1/2}`
analogue (Löwdin). Charges sum to the cluster charge (≤1e-9). The per-cell charge
is the translational image-average; `translational_spread` (max image-to-image
variation) is a symmetry diagnostic - ≈ 6e-4 for the homonuclear H₂ chain, but
~0.3 for the *ionic* LiH chain on the bare four-center, concrete evidence that
ionic systems need the neutral reference (the bare-1/r four-center breaks cyclic
symmetry for ionic cells; § 2.0 decision log). LiH per-cell charges have the
correct sign (Li⁺ / H⁻).

### C.3 Electric dipole (care needed - Resta)

`ccm_dipole(scf_result, ccm)` returns the **finite-cluster** electric dipole
`μ = Σ_A Z_A R_A − Tr(D ⟨r⟩)`. This is *not* the bulk polarization: the position
operator is not legitimate on a torus (Resta 1998) and the plain home-cell
`⟨μ|r|ν⟩` wraps for boundary-crossing orbitals (the Wannier-centroid aliasing of
§ 1.3). Faithful only for non-wrapping (dilute) clusters; for a dense crystal it
is a symmetry indicator (≈ 0 for centrosymmetric/neutral - verified ≤1e-8 on the
H₂ chain). Bulk polarization needs the Berry-phase / Resta operator (open item).

### C.4 Nuclear gradient / forces (numerical; analytic deferred)

`ccm_numerical_gradient(ccm, *, runner=, method=, h=)` → `(n_cell_atoms, 3)`
central-difference `dE_total/dR`. Displacing a unit-cell atom propagates to every
image (the supercell is built by replication), so this is the gradient under the
cyclic constraint; the force is `−gradient`. Well defined and tractable (no
analytic WSSC-integral derivatives), `6·n_cell_atoms` SCF evaluations.

Validated: (i) the per-cell forces sum to ≈ 0 (translational invariance, exact);
(ii) in the isolated limit (`(1,1,1)`, 80-bohr box) it reproduces vibe-qc's
molecular analytic RHF gradient to **8.9e-8** (finite-difference accuracy). The
analytic WSSC-weighted gradient (differentiating `S^CCM`, `h^CCM`, the neutral
four-center) is the open follow-on.

### C.5 Open
Resta periodic position operator (bulk polarization + unwrapped dipoles, shared
with Task 1 M4); analytic CCM gradient; open-shell (UHF/UKS) gap + populations
(Task D - populations/dipole already accept an unrestricted density via the
total-density accessor; only the spin-resolved gap needs wiring).

## Task D - Open-shell parity

The unrestricted analogues of the RHF-path methods, on the neutral reference.

### D.1 UCCSD(T) (`run_ccm_uccsd`, DONE)

`vibeqc.periodic.ccm.uccsd.run_ccm_uccsd(ccm, uhf_result=, *, cderi=,
compute_triples=)` → `CCMUCCSDResult`. The open-shell sibling of `run_ccm_ccsd`:
it runs vibe-qc's own spin-orbital UCCSD(T) engine
(`vibeqc.dlpno._ccsd_ref.run_ref_uccsd`, the in-repo anchor for `cpp/uccsd.cpp`)
on the UHF-CCM reference, fed the CCM **neutral cderi** transformed into the α / β
MO bases (`B^σ[P,p,q] = Σ_μν C^σ[μ,p] L[P,μν] C^σ[ν,q]`, so
`(pq|rs)^{σσ'} = Σ_P B^σ_pq B^{σ'}_rs`) plus the diagonal canonical UHF Fock.

The neutral route is the *natural* one: `run_ref_uccsd` needs an RI factorization
of the four-center, and the neutral `g_eff = Σ_P L⊗L` is exactly that (the
bare-1/r four-center is non-separable) and, per § 2.0, is the required
correlation reference for ionic systems.

**Validated (`tests/test_ccm_uccsd.py`):**

* **Closed-shell consistency** - on an even-electron closed-shell cluster
  `run_ccm_uccsd` reproduces the closed-shell `run_ccm_ccsd` on the same neutral
  four-center to ≤1e-8 (CCSD and (T) each; the UHF collapses to RHF, αα = ββ).
  This exercises the full spin-orbital machinery + the α/β B-tensor transform.
* **Open-shell** - a genuine doublet cluster (3-H chain, `n_α ≠ n_β`) converges
  with negative correlation.

### D.2 Unrestricted properties (DONE)

The population (`ccm_mulliken_charges` / `ccm_lowdin_charges`) and dipole
(`ccm_dipole`) routines already consume an unrestricted density (the total-density
accessor sums `density_alpha + density_beta`). The **spin-resolved gap** is now
wired in `ccm_homo_lumo_gap`: for a UHF/UKS result it returns
`LUMO − HOMO` with `HOMO = max(ε^α[n_α−1], ε^β[n_β−1])`,
`LUMO = min(ε^α[n_α], ε^β[n_β])` (`spin="unrestricted"`). Validated on the
doublet 3-H chain (`tests/test_ccm_properties.py`).

### D.3 U-DLPNO (DONE - DLPNO-UMP2 + DLPNO-UCCSD(T))

The open-shell local-correlation analogue of `ccm_dlpno_mp2` / `ccm_dlpno_ccsd`,
on the **neutral**-reference UHF-CCM wavefunction. Both reuse the molecular
open-shell DLPNO PNO machinery (`vibeqc.dlpno.ump2._pnos`) and the spin-orbital
UCCSD(T) engine (`vibeqc.dlpno._ccsd_ref.run_ref_uccsd`, the same one
`run_ccm_uccsd` runs) with the CCM neutral B-tensors injected.

**DLPNO-UMP2 - `ccm_dlpno_ump2(ccm, uhf_result=, *, cderi=, tcut_pno=,
tcut_mkn=, n_frozen=, ss_scale=, os_scale=)` -> `CCMDLPNOUMP2Result`**
(`dlpno_ump2.py`).
The correlation is resolved into the three unrestricted-MP2 spin channels (Szabo &
Ostlund Sec. 6.7, as in `run_ccm_ump2`):

* same-spin (αα, ββ): `E_σσ = Σ_{i<j} Σ_ab ½ |⟨ij||ab⟩|² / D`, antisymmetrised
  exchange `⟨ij||ab⟩ = (ia|jb) − (ib|ja)`, one PNO set per pair;
* opposite-spin (αβ): `E_ab = Σ_{ij} Σ_ab (ia|jb)² / D`, Coulomb only, separate
  α-PNO and β-PNO set per pair.

Every `(ia|jb)` is density-fit from the neutral cderi (`g_eff = Σ_P L⊗L`); each
pair carries PNOs built from its pair density and truncated by occupation number
(`tcut_pno`).

**DLPNO-UCCSD(T) - `ccm_dlpno_uccsd(ccm, uhf_result=, *, cderi=, tcut_pno=,
tcut_mkn=, n_frozen=, compute_triples=)` -> `CCMDLPNOUCCSDResult`**
(`dlpno_uccsd.py`).
The per-channel PNO pairs are merged into a **per-spin union PNO virtual space**
(`V_trunc^α` from the αα pairs + the α side of the αβ pairs; `V_trunc^β`
symmetrically), each `S^CCM`-orthonormal and semicanonical (the Fock diagonalised
within the union), so `[occ^σ | V_trunc^σ]` is orthonormal with a diagonal Fock
(the occ-virt block vanishes - every PNO is a rotation of the canonical virtuals,
which are `S^CCM`-orthogonal to the occupieds). `run_ref_uccsd` then solves the
unrestricted CC within `{occ^σ, V_trunc^σ}`. One implementation detail: the
spin-orbital engine indexes α/β spatial MOs by a single shared range, so the
smaller per-spin truncated MO set is padded with extra (real, semicanonical)
complement virtuals - exact-safe, since enlarging a truncated space only reduces
truncation.

**Why it is exact at no truncation.** At `tcut_pno = 0` every PNO set spans its
full virtual space, so each per-spin union is a unitary rotation of the canonical
σ-virtuals; UMP2 reproduces the canonical channels term-by-term, and UCCSD(T) -
invariant to rotations within the occupied/virtual blocks - equals the canonical
CCM UCCSD(T). With `tcut_mkn > 0`, each occupied pair selects atoms from its
`S^CCM` Mulliken populations. Separate alpha and beta virtual projectors remove
the complete occupied space of each spin, and the selected projected atomic
orbitals are canonical-orthogonalized and semicanonicalized with the
corresponding converged UHF Fock matrix. Pair densities and PNOs are then built
inside those bounded spaces and mapped back into canonical virtual coordinates
for the UCCSD union. `tcut_mkn = 0` bypasses PAO construction exactly.

**Validated (`tests/test_ccm_dlpno_ump2.py` 13/13, `tests/test_ccm_dlpno_uccsd.py`
8/8):** the no-truncation limit (`tcut_pno = 0`) reproduces

* canonical CCM UMP2 (`run_ccm_ump2` on the neutral four-center) to machine ε
  (≤2e-17 on the closed-shell (2,1,1) H₂ chain; ≤1e-17 on the genuine doublet
  (3,1,1) H chain, n_α ≠ n_β), per spin channel; and
* canonical CCM UCCSD(T) (`run_ccm_uccsd`) - CCSD to machine ε (closed-shell
  ≤4e-17; doublet ≤8e-14), (T) to ≤6e-21.

Truncation recovers canonically: on a 6-31g doublet the UMP2 error falls
monotonically (1.2e-3 → 1.0e-5 → 4.4e-7 → ε) as `tcut_pno` → 0, and the
unequal-per-spin **padding path** is exercised (β padded back to full when the
α union needs the larger common MO count) and stays exact-safe.

**Reference = neutral** (per § 2.0 / the `-b`-critique correction): the SCF
reference and the four-center are the neutral kernel - the Madelung background
shifts the occ-virt denominators and the bare-1/r four-center has no consistent
cderi. References: Pinski et al. 2015 for DLPNO-MP2; Riplinger et al. 2013 for
DLPNO-CCSD(T); Stanton et al. 1991 for spin-orbital UCCSD. Saitow et al. 2017
is qualified background for the open-shell CC route only: these independent
UHF alpha/beta spaces do not implement its single-spatial-orbital NEV-PNO
construction or SOMO-pair threshold scale. The applicable entries are already
in the citation DB and routed for the reused molecular components.

**Remaining locality follow-ons:** localized occupied orbitals require the
coupled LMP2 residual rather than the present canonical occupied denominators.
Per-pair-coupled DLPNO-UCCSD and symmetry-unique pairs also remain Task 3 work.
