# Mixed-boundary (wire/slab) Green's function for the low-D CCM four-center

*The `aiccm2026dev-a` side of joint deliverable **item 6** (the open construction of
the converged `-a`/`-b` theory result, `docs/manuscripts/aiccm_a_position.md`
§0.5). Derived **Hamiltonian-first**: define the mixed-boundary Poisson problem,
then read **both** the Coulomb kernel and the exchange-`q=0` seam off the **same**
Green's function `G` -- not a minimum-image truncation. Built as `-a`'s own
implementation. A future χ-CCM comparison must bind the separately derived B
Hamiltonian; current χ-CCM-B 1-D/2-D absolute energies fail closed.*

> **D90 construction boundary.** This document describes an A-line
> mixed-boundary wire construction/control. It is not currently classified as
> union-and-weight Γ-CCM because no binding from this Hamiltonian to the
> Wigner--Seitz integral-weighting map has been derived. A-line namespace
> ownership and real-Gamma evaluation of the finite cluster do not confer
> Γ-CCM identity. This wire/slab Green-function work also does not define a
> χ-CCM kernel, and a same-Green-function representation control would not
> prove equality of the two constructions.

Status: M1 (this document) → **M2a/M3a scalar kernels, M2b/M4/M5 d=1 (wire) DONE**
(2026-07-04): the full wire four-center + `V_ne`/`E_nn` + SCF
(`run_ccm_rhf_wire`) with the neutral-cell Hartree gauge cancellation gated and
the M5 4c-vs-RI gap collapsed to machine ε. **d=2 (slab)** four-center + the
wire-Madelung exchange seam remain. Progress tracked in
`handovers/HANDOVER_AICCM_FOLLOWON.md`.

---

## 0. Why the 3-D neutral control is wrong for genuine 1-D/2-D

The historical neutral-torus control object is the charge-neutral **3-D-torus**
Ewald Green's function `v_E^{3D}` -- the unique
solution of the torus Poisson equation with a *uniform* compensating background
`−1/V_c` over the **3-torus** `𝕋³ = ℝ³/L_c`, with the `G=0` mode dropped
(`python/vibeqc/periodic/ccm/neutral.py`; `ccm_neutral_cderi` is its Γ density
fit). It was previously mislabelled as a shared Γ-CCM/χ-CCM production object;
D89 withdraws that claim. A vacuum-padded 1-D chain or 2-D slab is *modelled*
with this same 3-D
kernel by inflating the open directions with vacuum. That is a **supercell
model**, not the right Hamiltonian: the neutralizing background is spread through
the vacuum, the long-range field in the open directions is the wrong functional
form, and the resulting four-center carries a system-size-dependent **gauge
offset** that is **not** an RI fitting error. Historical pre-guard comparisons
showed four-center-vs-RI gaps of *hundreds of kJ/mol* in low-D, including a
vacuum-padded B H₄ control with `4c - RI = 22.2 mHa`. Those B values are defect
evidence, not model or accuracy numbers (Q-lowD). Current χ-CCM-B blocks every
lower-dimensional SCF backend before SCF.

The honest fix is a Green's function that is periodic in the `d` lattice
directions and **open** (decaying, free-space) in the `3−d` transverse
directions, with the neutralizing background a uniform charge **only over the
periodic directions** (a sheet for `d=2`, a line for `d=1`). This is the
**mixed-boundary** kernel `G^{(d)}`. The 3-D kernel is recovered as the
transverse-periodic sum of `G^{(d)}` -- the hard, dependency-free validation
anchor (§5).

### 0.1 Correction (2026-07-10) - the low-D production kernel was not `v_E^{3D}`

The paragraph above understates the defect, and the correction changes how low-D
numbers must be read. A vacuum-padded low-D cell was **not** being modelled with
the 3-D torus kernel `v_E^{3D}`. `rsgdf_dense_g_mesh` (and `rsgdf_g_mesh`) size the
reciprocal mesh over the *periodic* axes only (`axis < dim`) and pin every
non-periodic axis at `G_⊥ = 0`; consumers then apply `4π/|G+q|²/V` to that mesh.
Dropping the transverse Fourier components replaces each AO-pair density by its
**transverse average**, so the kernel is not `1/r` at all but a transverse-uniform
**sheet term `∝ 1/V`**, which *vanishes* as the vacuum grows.

Measured (H₂ chain, `dim=1`, sto-3g, 6-bohr period): `|g_neutral|_max · V = 92.46`
is **constant** across transverse vacuum `D = 12/18/24` bohr, while the wire
four-center is `D`-invariant at `1.895512`; the two differ by 14 % of the wire
scale even after removing the best-fit `c·S⊗S`. At SCF level the `dim=1` route ran
`-6.917 → -6.568` Ha for `D = 12 → 30` bohr, and the production multi-k
`run_krhf_periodic_gdf` ran `-3.458 → -3.324` Ha on the same sweep.

So the low-D four-center-vs-RI gap is **not** merely "a gauge offset rather than an
accuracy error". Combined with the *bare* (conditionally convergent) `V_ne`/`E_nn`
lattice sums the `dim<3` route also used, the total additionally diverged with the
nuclear lattice-sum cutoff. Both the 3-D-torus route and the multi-k GDF therefore
**fail closed** for `dim < 3` as of 2026-07-10; the gauge-consistent low-D
Hamiltonians (`run_ccm_rhf_wire`, the four-center route) are selected by name, never
substituted silently - they are different operators, coinciding only in the
non-interacting limit. See `tests/test_ccm_lowd_gauge_consistency.py` and
`handovers/HANDOVER_AICCM_FOLLOWON.md` finding #6.

---

## 1. The mixed-boundary Poisson problem (Hamiltonian-first)

Let the cell be periodic in `d` directions with in-plane/along-axis cell area /
length `A` (`d=2`) or `L` (`d=1`), and open in the transverse coordinate(s),
written `z` (a scalar for `d=2`; `ρ=(x,y)` for `d=1`). Write `r = (R∥, z)` with
`R∥` the periodic coordinates. The electrostatic kernel `G(r,r')` is the periodic
solution of Poisson's equation **with the periodicity's neutralizing background**:

```
−∇² G(r, 0) = 4π [ δ^{(3)}(r) − b_d(z) ] ,      (Gaussian units, e=1)
```

where `b_d` is the compensating background that makes the source charge-neutral
**per periodic cell**, smeared uniformly over the periodic directions and
*localized in the transverse coordinate exactly as the unit point source is*:

* `d=3`: `b_3 = 1/V_c` (uniform over the 3-torus) → the production `v_E^{3D}`
  (`G=0` dropped). The transverse background is absent (no open direction).
* `d=2` (slab): `b_2(z) = δ(z)/A` -- a neutralizing **sheet** in the source plane.
  The in-plane average density vanishes; the `z`-profile is a `δ` at the source.
* `d=1` (wire): `b_1(ρ) = δ^{(2)}(ρ)/L` -- a neutralizing **line** along the axis.

Boundary conditions (the part that is *declared*, not derived -- they fix the
otherwise-undetermined harmonic piece):

* **Periodic** in `R∥` (Born-von-Kármán): `G(R∥+T, z) = G(R∥, z)` for lattice `T`.
* **Open / decaying** in the transverse direction(s): the *oscillatory*
  (`G∥≠0` / `G_z≠0`) Fourier components decay as `e^{−|G∥||z|}` (`d=2`) /
  `K₀(|G_z|ρ)` (`d=1`); the **zero transverse-mean** (`G∥=0`/`G_z=0`) channel is
  the field of the neutralizing sheet/line and is only *conditionally* convergent
  -- its linear-in-`|z|` (`d=2`) / logarithmic-in-`ρ` (`d=1`) growth is the low-D
  analogue of the 3-D `G=0` conditional convergence (§4). A finite constant `C`
  (the gauge) is fixed by a reference (`z→0` / `ρ→ρ₀`).
* **Dipole / asymptotic BC (polar systems).** A slab with a net normal dipole
  `M_z ≠ 0` (or a wire with a net transverse dipole) carries a **dipole-layer**
  term in the conditional channel -- a discontinuity in the asymptotic potential
  across the slab, the low-D analogue of the surface term in the 3-D conditionally
  convergent lattice sum. It **must be declared** and is the physical content of
  the `M_z≠0` correction; see §4 and the known limitation in §6.

---

## 2. The 2-D slab kernel `G^{2D}` (periodic in the plane, open in `z`)

Solving §1 for `d=2` in a plane-wave basis over the in-plane reciprocal lattice
`{G∥}` (area `A` per cell), each `G∥≠0` mode solves
`(|G∥|² − ∂_z²) ĝ_{G∥}(z) = 4π δ(z)`, giving the decaying Green's function
`ĝ_{G∥}(z) = (2π/|G∥|) e^{−|G∥||z|}`. The `G∥=0` mode solves
`−∂_z² ĝ_0(z) = 4π[δ(z) − 1/A]`·`A` per area, the field of a sheet pair, giving
`ĝ_0(z) = −2π|z|/A` (up to the gauge constant). Summing:

```
G^{2D}(ρ, z) = (2π/A) Σ_{G∥≠0} (e^{iG∥·ρ} e^{−|G∥||z|}) / |G∥|   −  (2π/A)|z|  +  C
                └──────── oscillatory, exponentially screened ────────┘   └ sheet ┘
```

(Parry, *Surf. Sci.* **49**, 433 (1975); de Leeuw-Perram, *Mol. Phys.* **37**,
1313 (1979); Bertaut.) This is **exactly** the gauge realized in vibe-qc's
already-landed slab Hartree: `python/vibeqc/ewald_composed_slab.py` builds the
oscillatory `G∥≠0` channel as the `L_z→∞` continuous-`G_z` 3-D-FT and the
`G∥=0` channel as the analytic `−(2π/A)∫dz∫dz' n(z)|z−z'|n(z')` `|z|`-convolution,
with `V_ne` (`periodic_v_ne_slab.py`) sharing the *same* gauge so the bilinear
total `E = E_nn + Tr[D·V_ne] + ½Tr[D·J]` is gauge-consistent
(`tests/test_j_slab_ewald_2d.py`). **The 2-D *Coulomb/Hartree* side of `G^{2D}`
therefore already exists and is validated against the 3-D kernel** (§5). What D3
adds for `d=2`: the **four-center / ERI** form of the same `G^{2D}` and the
matching **exchange-`q=0` seam** (§3, §6), plus closing the polar-slab dipole BC
(§4/§6).

---

## 3. The 1-D wire kernel `G^{1D}` (periodic along `z`, open in `ρ`)

For `d=1` (axis `z`, length `L` per cell, transverse `ρ=(x,y)`), each `G_z≠0`
mode solves `(G_z² − ∇_ρ²) ĝ_{G_z}(ρ) = 4π δ^{(2)}(ρ)`, the 2-D screened-Poisson
(Yukawa-in-2-D) equation, whose decaying solution is the modified Bessel function
`ĝ_{G_z}(ρ) = 2 K₀(|G_z|ρ)`. The `G_z=0` mode solves the 2-D Poisson equation of
a neutralized line, `−∇_ρ² ĝ_0 = 4π[δ^{(2)}(ρ) − 1/(L·∞)]`·`L`, giving the
logarithmic line potential `ĝ_0(ρ) = −2 ln(ρ/ρ₀)` (up to gauge). Summing:

```
G^{1D}(ρ, z) = (2/L) Σ_{G_z≠0} K₀(|G_z|ρ) e^{iG_z z}   −  (2/L) ln(ρ/ρ₀)  +  C
                └──── oscillatory, Bessel-screened ────┘   └─── line ───┘
```

This kernel is **new** to vibe-qc (the slab workstream marked `NEUTRALIZED_1D`
explicitly out of scope, `handovers/HANDOVER_SLAB_EWALD_2D.md`). M2 implements it.
`K₀` is available via `scipy.special.k0`; the `G_z≠0` sum converges
exponentially (`K₀(x) ~ √(π/2x) e^{−x}`), and the `ρ→0` short-range singularity
is `K₀(|G_z|ρ) → −ln(|G_z|ρ/2) − γ`, which combines with the `−(2/L)ln(ρ/ρ₀)`
line term to give the correct bare `1/r` Coulomb singularity as `ρ,z→0` (the
short-range `K` exchange piece is built molecular-limit/real-space, as in the 3-D
and slab paths). `ρ₀` (the gauge radius) and `C` are fixed by the reduction
anchor (§5).

---

## 4. The conditional (zero-transverse-mean) channel and the dipole BC

The `G∥=0` (`d=2`) / `G_z=0` (`d=1`) channel is where all the finite-size physics
and the convention freedom live -- it is the low-D image of the 3-D `G=0` term:

* Its **growing** part (`−(2π/A)|z|`; `−(2/L)ln(ρ/ρ₀)`) is *forced* by the
  neutralizing sheet/line -- the analogue of the forced `J`/Coulomb part in 3-D.
* Its **constant** `C` is the gauge, fixed by a reference point; differences of
  energies (and the MO-basis correlation numerator) are `C`-independent.
* For a **polar** cell (net normal dipole `M_z`, slab; net transverse dipole,
  wire) the asymptotic potential is **discontinuous** across the system -- a
  **dipole-layer** term `Δφ = 4π M_z/A` (slab). This is the genuine low-D analogue
  of the 3-D conditionally-convergent surface term, and it is a **declared
  boundary condition**: the modeller chooses tin-foil (short-circuit, `Δφ=0`) or
  open (the dipole layer is kept). `-a` declares the **open** BC (the dipole layer
  is physical for an isolated slab/wire) and records it alongside `exchange_q0`.
  The existing slab Hartree carries a **known inaccuracy here** (the `M_z≠0`
  z-profile first moment is off ~6.7e-3, blocking the slab SCF un-gate, §6) -- D3
  must get the dipole-layer term right, because it is exactly the term that
  distinguishes a polar slab's Hamiltonian.

---

## 5. The transverse-periodic-reduction gate (M3 -- the hard, dependency-free check)

`v_E^{3D}` is the transverse-periodic sum of the mixed-boundary kernel: stacking
`G^{2D}` at every transverse image `z → z + nL_z` (a 3-D lattice with the open
direction re-periodized at period `L_z`) must reproduce the **already-validated**
`v_E^{3D}` up to the gauge constant:

```
Σ_{n∈ℤ} G^{2D}(ρ, z + nL_z)  ≟  v_E^{3D}(ρ, z; L_z)  +  c·(gauge)        [d=2 → 3]
Σ_{m,n}  G^{1D}(ρ + m a_x + n a_y, z)  ≟  v_E^{3D}(·; a_x,a_y)  + c       [d=1 → 3]
```

Realized on the **four-center / kernel** level, the gate is: the low-D effective
four-center `g_eff^{(d)} = (μν| G^{(d)} |rs)` re-periodized in the transverse
directions equals `ccm_eri_neutral`/`ccm_neutral_cderi` (`v_E^{3D}`) up to a
rank-1 `c·S⊗S` gauge shift -- the four-center analogue of the **already-passing**
`tests/test_j_slab_ewald_2d.py::test_j_matches_3d_vacuum_limit_up_to_gauge`
(`J_2D − J_3D → c·S + resid`, residual → 0 as the gap grows). This needs **no
external dependency**: the right-hand side is `-a`'s own validated 3-D kernel.
Passing it for both `d=2` and `d=1`, on the four-center, is M3.

---

## 6. The exchange-`q=0` seam from the same `G` (M4) -- and the D1 tie-in

The exchange contribution `K_{μν} = Σ_{rs} D_{rs} (μs| G |rν)` carries a
`q=0` (`G∥=0`/`G_z=0`) self-interaction seam -- the low-D analogue of the 3-D
`exxdiv` term. **It is read off the same `G`, not chosen independently:** the
seam is the signed probe-charge limit of the conditional channel of `G^{(d)}`
(the `C`-gauge value at the probe-charge self-distance), i.e. the low-D analogue
of `madelung.apply_exxdiv_ewald_to_K` / `exxdiv_ewald_energy_shift` (the 3-D
BvK-Madelung exchange). Because the seam shifts the HF occupied/virtual spectrum,
**it propagates into correlation through the gap** -- exactly the dependence D1
made a machine-recorded provenance field (`exchange_q0`,
`python/vibeqc/periodic/exchange_convention.py`). **As landed (2026-07-04)** the
wire SCF (`run_ccm_rhf_wire`) records `exchange_q0 = strict-zero-mode` -- the
well-defined, `g_perp_min`-independent default (the `S⊗S` monopole projected out
of `K`; §10). The `BvK-ewald` low-D seam (the signed-probe-charge limit of the
conditional channel of `G^{(d)}`, whose 3-D sibling is the `ξ_N·S D S` term of
`run_ccm_rhf_direct`) is the residual freedom -- the D2 low-D exchange-`q=0`
question -- to be added, recorded, and asserted matched in the A/B/KMP2 comparison
alongside the dipole BC.

---

## 7. Milestones

| M | Deliverable | Gate | State |
|---|---|---|---|
| **M1** | Poisson problem + BCs + kernels declared (this doc) | derivation + dependency-free reduction anchor stated | **DONE** |
| **M2a** | Scalar mixed-boundary kernels: `G^{1D}` (new, `K₀`/`ln`) + `G^{2D}` (Parry, `e^{−|G∥||z|}`/`|z|`) | **harmonic off-source** (`∇²G=0` for `ρ>0`/`z≠0`) + **unit source strength** (`G − 1/r →` finite self-energy ξ, i.e. `r·G → 1` -- a non-unit source would diverge) + lattice-periodic + transverse-even -- all dependency-free | **DONE** (`periodic/lowd_greens.py`, `test_lowd_greens.py` 17/17; the self-energy ξ is the M4 exchange-seam foundation) |
| **M2b** | Four-center routing `(μν\|G^{(d)}\|rs)` | short-range → bare `1/r` pair; contracts as chemists' ERIs | **d=1 (wire) DONE** (`periodic/ccm/lowd_four_center.py` `ccm_wire_cderi`/`ccm_eri_wire`, the §9 quadrature; factorises as a real cderi → the whole RI/DLPNO stack rides it. Gate: exact erf-image-sum four-center matched to **2.8e-8** mod one `c·S⊗S`, `test_ccm_lowd_four_center.py` 3/3. NB the 3-D vacuum-ladder is *not* a clean gate: `c` drifts as `−(2/L)ln D` ✓ physics, but the non-gauge residual is the 3-D route's own finite-`D` artifact.) **d=2 (slab): the wire's construction does NOT carry over** (finding 2026-07-05). The naive full-3-D-FT + open-axis quadrature works for the wire because the open direction is 2-D and its `G∥=0` conditional channel is a mild **log** that reduces to a `c·S⊗S` monopole gauge. The slab's open direction is 1-D, so its `G∥=0` channel is a **`|z|` sheet** with a `1/g_z_min` **power** divergence: it is *dipole*-coupling, not a pure monopole gauge. Measured on a non-polar H₂/sto-3g (2,2,1) slab: (i) J from the naive slab cderi disagrees with the validated `compute_j_slab_ewald_2d_gamma` by **24%** after `c·S` removal; (ii) RI-MP2 on it is catastrophically gauge-dependent (**−276.8 Ha at g_z_min=1e-4 vs −0.13 at 1e-2**, vs the 3-D-neutral's −0.0014). The correct slab four-center needs the §9 **partial in-plane FT** `ρ̃_μν(G∥;z)` (open `z` retained) + the analytic `|z|`-convolution for `G∥=0` (as vibe-qc's slab-J builder already does its `J_g0`), a dedicated build, not a swap of the wire quadrature |
| **M3a** | Scalar reduction: each kernel `== lim` (large transverse period) of `v_E^{3D}` up to a gauge `C` | reference Ewald `v_E^{3D}` self-validated (η-invariant + Madelung); residual ↓ monotonically (wire ~1/D² to <2e-3; slab ~1/L_z) | **DONE** (`test_lowd_greens_reduction.py` 3/3) |
| **M3b** | Four-center reduction: `g_eff^{(d)}` re-periodized `== ccm_eri_neutral` up to `c·S⊗S` | residual → 0 with gap | **SUPERSEDED** by a stronger, reference-free gate: the wire four-center is matched to the **exact `erf`-image-sum** four-center (`Σ_n erf(μ d_n)/d_n`) to 2.8e-8 mod `c·S⊗S` (M2b). The 3-D vacuum ladder is explicitly *not* a clean gate (§0/§7-M2b: its non-gauge residual is the 3-D route's own finite-`D` artifact), so exact image sums replaced it throughout M2b/M4/M5 |
| **M4-foundation** | Low-D self-energy ξ = `lim_{r→0}[G − 1/r]` (the Madelung-constant analog) | r² Richardson, gauge-shift exact `(2/L)ln(ρ₀'/ρ₀)`; **anchored to the validated 3-D `madelung_constant_for_cell`**: `−madelung(D⊥) − ξ_wire − (2/L)ln(D⊥)` is D⊥-independent (the gauge is *exactly* the line-vs-uniform `(2/L)ln(D⊥)`) | **DONE** (`wire_self_energy`/`slab_self_energy`, `test_lowd_greens{,_reduction}.py`) |
| **M4** | Wire `V_ne`/`E_nn` on the shared gauge + the exchange-`q=0` seam | neutral-cell Hartree total `g_perp_min`-independent (gauge cancels); exchange `q=0` handled by the strict-zero-mode | **d=1 DONE** (`lowd_four_center.py` `ccm_wire_v_ne` (`Δc'/Δc=−N_nuc`), `ccm_wire_e_nn` (1-D mini-Ewald, β-independent); Hartree total invariant to 1.1e-5 while each term swings ~6 Ha, `test_ccm_lowd_four_center.py` 10/10). The **wire-Madelung** seam (`exxdiv="ewald"` analogue) is the residual D2 low-D question; the strict-zero-mode (`exxdiv="strict"`, S⊗S projected out of `K`) is the well-defined default |
| **M5** | Re-run 1-D H-chain control | 4c-vs-RI gap (`-b` H₄ `4c−RI=22.2 mHa`) collapses to the **RI fitting error** | **d=1 DONE** (`lowd_scf.py` `run_ccm_rhf_wire`: dense four-center SCF == RI-cderi SCF to **5e-13** -- the wire `4c−RI` is machine-zero; and the strict total == the SCF from **exact image-sum integrals** to 8.4e-7, `test_ccm_lowd_scf.py` 4/4) |

**For the declared A-line wire Hamiltonian, d=1 now yields accuracy numbers,
not model numbers.** This statement does not bind the implementation to Γ-CCM.
d=2 (slab) still awaits its four-center + reduction gate.

## 8. Reuse / boundaries

* Reuse vibe-qc's own slab Ewald (`ewald_composed_slab.py`,
  `periodic_v_ne_slab.py`) and the 3-D `v_E` (`neutral.py`) -- vibe-qc code, not an
  external program (CLAUDE.md §10). The 1-D wire kernel is new.
* The slab *SCF un-gate* (the polar-dipole §7 gate) is the **periodic-SCF chat's**
  workstream ([Archived `HANDOVER_SLAB_EWALD_2D.md`](https://vibe-qc.com/docs/),
  blocked on `M_z≠0`); D3 builds the
  **CCM four-center + exchange-seam** layer on top and contributes the corrected
  dipole-layer term back to that gate.
* Any future parity check against a separately specified `-b` mixed-boundary
  construction must run **out-of-process** (subprocess), never by importing
  `-b` modules; `-a` keeps its own implementation. Current χ-CCM-B 1-D/2-D
  absolute energies remain fail-closed, and parity would not assign either
  implementation to an approach by ancestry.
* **Heavy runs (the dense `g_eff` / all-FT GDF) go to vq, never local** (the
  137 GB hazard); M3/M5 numerics are vq jobs.

*References:* Parry 1975; de Leeuw-Perram 1979; Bertaut; Rozzi *et al.*,
*Phys. Rev. B* **73**, 205119 (2006) (mixed-boundary Coulomb cutoffs);
Sun-Berkelbach-McClain-Chan 2017 (the 3-D periodic GDF this reduces to);
Peintinger & Bredow 2014 (CCM). Joint spec: `aiccm_a_position.md` §0.5 item 6,
`aiccm_a_proposed_comparison_text.md` point 5.

## 9. M2b -- concrete four-center construction (design, 2026-06-28)

The scalar kernels (M2a) are **translation-invariant** (1-D/2-D periodic in *free*
transverse/normal space), so `wire_greens(|Δρ⃗|, Δz)` / `slab_greens(Δρ⃗, Δz)` are
already the two-point source→field kernels `G^{(d)}(r−r')`. The four-center
`(μν|G^{(d)}|rs) = ∫∫ χ_μν(r) G^{(d)}(r−r') χ_rs(r') dr dr'` is therefore a **mixed
reciprocal(periodic)/real(open) object** -- not a scalar reciprocal contraction:

**Wire (d=1, periodic `z`, open transverse `ρ⃗`).** With `g_m = 2πm/L`,
```
(μν|G^{1D}|rs) = (4/L) Σ_{m≥1} ∫dρ⃗ dρ⃗' ρ̃_μν(g_m; ρ⃗) K₀(g_m|ρ⃗−ρ⃗'|) ρ̃_rs(−g_m; ρ⃗')
                 − (2/L) [ln-line / neutralising term],
   ρ̃_μν(g_m; ρ⃗) = ∫dz χ_μ(ρ⃗,z) χ_ν(ρ⃗,z) e^{i g_m z}   (1-D FT in z, RETAINING ρ⃗).
```
Per longitudinal mode `m` it is a **2-D transverse convolution** of the bra/ket AO-pair
profiles with the mode's `K₀(g_m·)` (the transverse screened-Poisson Green's function).

**Slab (d=2, periodic in-plane `ρ⃗`, open normal `z`).** With `G∥` the in-plane
reciprocal vectors,
```
(μν|G^{2D}|rs) = (2π/A) Σ_{G∥≠0} ∫dz dz' ρ̃_μν(G∥; z) e^{−|G∥||z−z'|}/|G∥| ρ̃_rs(−G∥; z')
                 − (2π/A) [sheet |z−z'| / neutralising term],
   ρ̃_μν(G∥; z) = ∫dρ⃗ χ_μ(ρ⃗,z) χ_ν(ρ⃗,z) e^{i G∥·ρ⃗}   (in-plane FT, RETAINING z).
```
Per in-plane mode `G∥` it is a **1-D `z`-convolution** with `e^{−|G∥||z−z'|}`.

> **Why the wire quadrature must NOT be reused here (finding 2026-07-05).** It is
> tempting to build the slab four-center the way the wire one is built: a full
> 3-D AO-pair FT `ρ̃(G∥,G_z)` on a `{G∥}×G_z` grid with `4π/(|G∥|²+G_z²)` folded
> into the weight, cut at `g_z_min`. That is **wrong for the slab**. For the wire
> the open direction is 2-D and its `G∥=0`-analogue conditional channel is a
> *log*, so the cut leaves a pure `c·S⊗S` monopole gauge. For the slab the open
> direction is 1-D: its `G∥=0` channel is the `−(2π/A)|z|` **sheet**, whose small-
> `G_z` weight `4π/G_z²` is a `1/g_z_min` **power** divergence coupling to AO-pair
> *dipoles*, not a monopole. Empirically (H₂/sto-3g (2,2,1) non-polar slab): the
> naive slab cderi's J misses `compute_j_slab_ewald_2d_gamma` by 24% (after `c·S`),
> and its RI-MP2 swings −276.8→−0.13 Ha as `g_z_min` goes 1e-4→1e-2. The `G∥=0`
> channel therefore **must** be built as the analytic `|z|`-convolution below
> (mirroring the slab-J builder's `J_g0`), and only the `G∥≠0` channels ride a
> reciprocal/`z`-grid factorisation; the partial-FT primitive is not optional.

**The key primitive -- a *partial* AO-pair FT.** Both need the AO-pair density
Fourier-transformed **only in the periodic direction(s)**, keeping the open
direction real. `python/vibeqc/_aopair_ft.py` (general-L McMurchie-Davidson, Bloch
variants) provides the full/Bloch AO-pair FT; M2b needs its **partial** form
(periodic dims transformed, open dim retained). Two routes:
1. extract the partial FT from the Bloch primitive (fix the open-dim reciprocal
   component to a real-space slice), or
2. do the open-direction integral analytically -- Gaussian × `K₀` (wire) and
   Gaussian × `e^{−|G∥||z|}` (slab) are closed-form in the MD/Boys machinery
   `_aopair_ft` already uses -- avoiding a real-space grid.

**Screening.** The transverse/normal kernels decay (`K₀` exponentially in `g_m ρ`,
`e^{−|G∥||z|}`), so the mode sums truncate at the same `n_recip`/`n_shell` as the
scalar kernels; only the leading modes carry weight for compact AO pairs.

**M3b gate (unchanged).** Re-periodise the open direction(s) -- make the cell fully
3-D-periodic -- and `g_eff^{(d)}` must reduce to `ccm_eri_neutral` (the validated
3-D neutral four-center) up to a `c·S⊗S` gauge, mirroring the passing slab-`J`
reduction test. Contracts as chemists' ERIs into the existing CCM correlation stack.

**Status:** design only. Implementation is a dedicated build (the partial-FT
primitive + the mode convolution + the M3b reduction gate); dense at first
(small/1-D/2-D), then the scalable weighted route. Heavy runs → vq (§8).

## 10. M4 seam -- the conditional gauge in the SCF energy (algebra, pinned 2026-06-29)

The wire four-center's conditional constant enters the closed-shell SCF energy
through two exactly known channels (verified against the measured tensor gauge
shift `Δc` between two IR cutoffs; `tests/test_ccm_lowd_four_center.py`):

```
E_J[c]  = ½ c (Tr PS)²   = ½ c N_e²      (Hartree;  measured to 1e-6)
E_K[c]  = ¼ c Tr[(PS)²]  ≈ ½ c N_e       (exchange; measured to 9e-5)
```

* The **Hartree** gauge term cancels for a **neutral cell** against the matching
  nuclear-side gauges (`½c(N_e² − 2N_eZ + Z²) = ½c(N_e−Z)² = 0`) -- provided
  `V_ne` and `E_nn` are built with the *same* wire kernel and regularization
  (the M4 build item).
* The **exchange** gauge term `−½cN_e` does **not** cancel: it *is* the
  exchange-`q=0` seam. Fixing the exchange-q0 convention ⇔ choosing `c`
  (strict-zero vs ewald-like are specific `c` choices) -- connecting directly to
  D1/D2 (`exchange_q0`, the 0.24 Ha/atom 3-D finding).
* Correlation is seam-free to O(mismatch): the gauge ~vanishes in `(ov|ov)` MO
  integrals (occ-virt S-orthogonality; residual only via `S^CCM`-vs-plain-`S`,
  measured 1.1e-5 on H₂ (2,1,1)).

**Landed (2026-07-04), the full wire SCF (M4/M5).** The wire `V_ne`/`E_nn`
builders (same quadrature, two-center/zero-center forms) and `run_ccm_rhf_wire`
close this:

* `ccm_wire_v_ne` rides the *same* real cderi `B`, so its gauge is `−N_nuc·c·S`
  with the four-center's `c` (`Δc'/Δc = −N_nuc` verified to 6 digits).
* `ccm_wire_e_nn` is a 1-D mini-Ewald (bare-`1/r` `erfc` real + self, Gaussian
  reciprocal on the shared quadrature); β drops out; gauge `½c·N_nuc²`.
* **Hartree cancellation realised numerically:** `E_nn + Tr[D·V_ne] + ½Tr[D·J]`
  is `g_perp_min`-invariant to 1.1e-5 while each term swings ~6 Ha across a 100×
  IR sweep (`½c(N_e−N_nuc)²=0`).
* **S/T convention decided:** plain cluster-supercell overlap/kinetic
  (short-ranged, gauge-free) -- the periodicity lives *entirely* in the Coulomb
  kernel, consistent with the wire four-center / `V_ne` / `E_nn` all built on the
  plain supercell AO pairs. (The alternative, WSSC-folded `S^CCM`/`T^CCM`, would
  double-count periodicity against the kernel-periodic Coulomb.)
* **Exchange** has no nuclear partner, so the exchange-`q=0` `½c·N_e` is the one
  surviving convention. `run_ccm_rhf_wire(exxdiv="strict")` projects the `S⊗S`
  monopole out of `K` (`K → K − c*·S D S`) → a `g_perp_min`-independent HF total
  (the low-D strict-zero-mode). The wire-Madelung seam (`exxdiv="ewald"`
  analogue) is the residual D2 low-D question.

See `python/vibeqc/periodic/ccm/lowd_scf.py` and `tests/test_ccm_lowd_scf.py`.
