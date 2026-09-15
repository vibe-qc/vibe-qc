# Periodic-method requirements from `basissetdev`

**Audience:** the periodic-methods main development chat.
**Authors:** `basissetdev`, M. F. Peintinger.
**Date:** 2026-05-08; last refresh 2026-05-13.
**Status:** living document; rebased onto `main` as features land.

> **2026-05-13 refresh.** Since the initial audit on `2c7c7df`,
> Codex landed the **native GDF backend** (`periodic_rhf_gdf.py`,
> `periodic_k_gdf.py`, `periodic_gdf_blocks.py`), replacing the
> retired in-process PySCF periodic backend, which CLAUDE.md § 10
> now formally forbids. **CRYSTAL-style FMIXING** (previous-Fock
> mixing) and **Fermi-Dirac smearing** are wired through the GDF
> and Ewald multi-k paths. Shared SCF mixing helpers
> (`validate_fraction_01`, `mix_fock_matrices`) live in
> `cpp/include/vibeqc/scf_mixing.hpp`, **new basissetdev work
> must reuse these**, never duplicate (manifest rule).
> See per-requirement status notes below.

This is the punch list of periodic capabilities `basissetdev` needs
in order to reproduce the published pob-paper test sets in vibe-qc.
The papers in scope are:

* M. F. Peintinger, D. Vilela Oliveira, T. Bredow, *J. Comput. Chem.*
  **34**, 451 (2013). [pob-TZVP, **PT2013**]
* D. Vilela Oliveira, J. Laun, M. F. Peintinger, T. Bredow,
  *J. Comput. Chem.* **40**, 2364 (2019). [pob-TZVP-rev2 +
  pob-DZVP-rev2, **VO2019**]

Together they enumerate roughly 70 binary and ternary ionic /
semiconducting / TM-oxide / metallic systems plus a 9-compound
subset of the X23 molecular crystals. Each requirement here is
tagged with the table(s) it gates and the count of compounds that
become runnable when it lands.

## Conventions

- **P0**: hard blocker for ≥10 compounds in the test set.
- **P1**: blocker for 1-10 compounds, or for cross-method validation.
- **P2**: nice-to-have / faithful reproduction of CRYSTAL workflow.
- **vibe-qc state** originally audited on commit `2c7c7df` of
  `basissetdev`; refreshed 2026-05-13 on commit `1bf32cc`.
- **Feature counts** below are conservative, orthorhombic is the
  most permissive cell shape vibe-qc currently handles.

## Test-set overview

| Source | Table | Cell shape | Compounds | Listed |
|--------|-------|------------|-----------|--------|
| PT2013 | T4    | cubic      | 12 | LiCl, NaCl, LiF, NaF, KF, CaF₂, K₂O, MgO, CaO, LiH, NaH, KH |
| PT2013 | T5    | hexagonal  | 10 | BeF₂, ScCl₃, MgBr₂, BeO, α-SiO₂, B₂O₃, Al₂O₃, NaNO₃, MgCO₃, FePO₄ |
| PT2013 | T6    | orthorhombic | 5 | BeBr₂, NaNO₂, CaH₂, CrCl₂, GeSe |
| PT2013 | T8    | cubic semiconductor | 22 | C(diamond), Si, Ge, AlP, AlN, Na₂Se, GaAs, GaP, ZnS, MnSe, ZnSe, β-BN, β-SiC, TiC, Cu₃N, VC, VN, TiN, CrN, K₂S, MnS |
| PT2013 | T9    | hexagonal semicond. | 9 | NiAs, α-SiC, α-BN, B₄C, ScB₂, CoS, CuS, GaF₃, GeO₂ |
| PT2013 | T10   | cubic TM oxide | 7 | Sc₂O₃, MnO, FeO, CoO, NiO, Cu₂O, ZnCr₂O₄ |
| PT2013 | T11   | hexagonal/tetragonal TM oxide | 4 | V₂O₃, Cr₂O₃, ZnO, TiO₂(rutile) |
| PT2013 | T12   | cubic metal | 11 | Li, Na, K, Ca, Sc, V, Cr, Fe, Ni, Cu, Al, Ni₃Al |
| PT2013 | T14   | HF cubic ionic | 12 (subset of T4 + K₂S) | LiCl, NaCl, LiF, NaF, KF, CaF₂, K₂O, MgO, CaO, LiH, NaH, KH |
| VO2019 | T2    | cubic ionic   | 14 | + KBr, SiF₄ |
| VO2019 | T3    | hexagonal ionic | 10 | (overlap with PT2013 T5) |
| VO2019 | T4    | orthorhombic ionic | 4 | NaNO₂, CaH₂, CrCl₂, GeSe |
| VO2019 | T6    | cubic semicond. | 14 | (subset of PT2013 T8) |
| VO2019 | T7    | hex semicond. | 9 | (subset of PT2013 T9) |
| VO2019 | T9    | cubic TM oxide | 7 | (overlap with PT2013 T10) |
| VO2019 | T10   | hex/tet TM oxide | 5 | + TiO₂(anatase) |
| VO2019 | T14   | molecular crystal | 9 | from X23, dispersion-dominated |
| VO2019 | Figs 2-7 | molecular hydride dimer | ~30 | H_n A-AH_n at 1.4-5 Å, A=Li-Br |

**Total unique compounds:** ≈75. **Counted by cell shape:**
roughly 40 cubic, 20 hexagonal, 10 orthorhombic/tetragonal, plus
the molecular-crystal subset.

## Hard requirements

### R1, Non-orthorhombic Ewald [P0] — SHIPPED

The FFT Poisson kernel (`cpp/include/vibeqc/fft_poisson.hpp`) now
works in fractional grid coordinates and applies the full
reciprocal-lattice metric to G·G, so orthorhombic, monoclinic, and
fully triclinic cells use the same FFT kernel — the orthorhombic-only
rejection this section originally described is gone.
`tests/test_periodic_lattice_families.py` (added 2026-07-10) exercises
hexagonal, tetragonal, and other non-orthorhombic families end to end,
including a dedicated `test_fft_poisson_accepts_all_crystal_families`
and a hexagonal Al₂O₃ smoke test — Al₂O₃ is one of the ≈32 compounds
this requirement's "blocked" list (below) names. A separate
ASE-boundary lattice-convention bug affecting non-orthogonal cells was
also fixed and validated against PySCF KRHF/GDF on a hexagonal cell to
3.05 mHa (2026-06-17,
`examples/regression/parity_hexagonal_lattice_vs_pyscf.py`).

The test-set fraction this unblocks:

- **Hexagonal:** PT2013 T5 (10), PT2013 T9 (9), PT2013 T11 (3),
  VO2019 T3 (10), VO2019 T7 (9), VO2019 T10 (4), ≈32 unique
  compounds.
- **Tetragonal:** TiO₂(rutile), TiO₂(anatase), 2 compounds.

**Basissetdev-specific acceptance still to confirm:**
`run_rhf_periodic_multi_k_ewald3d` accepting hexagonal `(a, b=a, c,
α=β=90°, γ=120°)` cells with total energy on α-Al₂O₃ (R-3c) at
experimental geometry agreeing with PySCF.pbc.GDF to 100 µHa per
formula unit specifically (the shipped fix is validated at the
FFT-Poisson-kernel and ASE-boundary level above, not yet re-run
against this exact acceptance number here).

**Note for `kpoints.py`:** the hexagonal/trigonal Γ-centring guard
is already in place; the work is on the Poisson side. No changes to
the k-mesh API required.

### R2, Multi-k MP-grid SCF on production basis sizes [P0]

**Status as of 2026-05-13: partially landed.**

Native GDF backend live: `periodic_rhf_gdf.py` (Γ-only),
`periodic_k_gdf.py` (multi-k), `periodic_gdf_blocks.py` (Bloch
assembly from lattice blocks). The in-process PySCF periodic
backend was retired (commits `545db04` / `7ab9384`), manifest
§ 10 now forbids importing PySCF inside vibe-qc; parity testing
against PySCF happens out-of-process via `examples/regression/
runner_pyscf.py`. The CaF₂ pob-TZVP RKS-PBE CRYSTAL14 regression
fixture (`e46211c`) is the first cubic-ionic parity test
exercising basissetdev's .d12 sidecar pipeline (Phase 14h).

**Already landed in the multi-k path:**

* `smearing_temperature` Fermi-Dirac occupations
  (`periodic_rhf_multi_k_ewald.py` lines 33-34, 156, 445;
  `periodic_k_gdf.py` line 76). Unit-bearing strings ("0.01 Ha",
  "300 K") accepted via the GDF gamma path (`7eacfd3`). **Free
  energy** with `E - T·S` correction surfaced in the result
  bundle (`periodic_rhf_gdf.py` line 548).
* `fock_mixing` (CRYSTAL FMIXING-equivalent) wired through both
  Γ-GDF and multi-k. Validation against MgO PBE/POB-TZVP
  CRYSTAL14 totals via `examples/regression` (commit `7c02666`).
* Low-dimensional MP meshes (1D / 2D systems) accepted by the
  legacy path (`f4d95f9`).
* Periodic dimensionality is logged in GDF jobs (`56fd2e7`).

**Still open:**

* Validation against CRYSTAL k-mesh density at **production
  sizes**. PT2013 / VO2019 use Pack-Monkhorst grids around
  8×8×8 for rocksalt ionics, 4×4×4 for the larger sesquioxides.
  vibe-qc tests have so far focused on small (2×2×2 to 4×4×4)
  grids; the linear-system size at full pob-TZVP × 8×8×8 is
  unmeasured. The Phase-14h `.d12` sidecars now make this
  measurable: same compound, same basis, same k-mesh, on both
  codes via `vq`. **Bench needed.**
* Smoke test on metallic Cu (FCC, T12). Smearing is in; the
  metal-specific SCF flow at production size hasn't been
  benchmarked. T12 metals (Li, Na, K, Ca, Sc, V, Cr, Fe, Ni,
  Cu, Al, Ni₃Al) inherit any remaining issues.

**Acceptance:** MgO total energy at Γ + 8×8×8 Pack-Monkhorst with
pob-TZVP/PW1PW agrees with PT2013 SI Table 1 (E = −275.477 595 Ha
per cell) to 1 mHa. Metallic Cu (FCC) converges with smearing
σ = 0.01 Ha and reproduces the 2013 Table 12 lattice constant
(3.506 Å) to 0.01 Å.

**Reuse, don't duplicate.** Shared mixing helpers
(`validate_fraction_01`, `mix_fock_matrices`) in
`cpp/include/vibeqc/scf_mixing.hpp` are the canonical
implementation across molecular + periodic, per manifest § 10.
Any new basissetdev SCF-control work plugs into this header.

### R3, Stable open-shell UKS / UHF for transition-metal oxides [P0]

**Status update: the initial-guess blocker has shipped.** ATOMSPIN, a
per-atom broken-symmetry spin seed (`atomic_spins`), landed 2026-06-14
and was extended to the periodic case (READ / ATOMSPIN / SPINLOCK) for
periodic UHF/UKS/ROHF on 2026-06-16, threaded through
`periodic_uks_ewald.py`, `periodic_uhf_ewald.py`,
`periodic_uhf_multi_k_ewald.py`, and `periodic_rohf_ewald.py`, and
validated against PySCF.pbc via
`examples/regression/parity_periodic_atomspin_vs_pyscf.py`. What's not
yet independently re-confirmed here is the specific basissetdev
acceptance criterion below (NiO vs. PT2013 SI Table 1 through the
CRYSTAL comparison) — Basissetdev's Goal 3 AFM-II compound inputs
(MnO, FeO, CoO, NiO, α-MnS, α-MnSe) should be revisited now that the
initial-guess hook they were waiting to fill in actually exists. The
`.d12` sidecars (Phase 14h) still skip emission for AFM compounds
pending ATOMSPIN-aware generator support on that side.

The multi-k UKS (`periodic_uks_multi_k_ewald.py`) and UHF
(`periodic_uhf_multi_k_ewald.py`) modules exist. The blocker is
not the kernel but the **initial-guess strategy** for
antiferromagnetic ordering. Currently both default to Hcore →
identical α/β densities → SCF converges to the closed-shell
solution by symmetry. This is the wrong basin for AFM systems.

Compounds blocked (counted once across PT2013 + VO2019): MnO, FeO,
CoO, NiO (T10/T9), Cr₂O₃, V₂O₃ (T11/T10), CrCl₂ (T6/T4), CrN, MnS,
MnSe (T8/T6), **10 transition-metal compounds**.

**Acceptance:** broken-symmetry initial guess (Mulliken-localised
single-atom UKS densities tiled into the AFM unit cell, optionally
with a small spin-symmetry-breaking field on the magnetic sublattice).
NiO total energy with pob-TZVP/PW1PW reproduces PT2013 SI Table 1
(−3167.495 768 Ha per cell, two formula units in the AFM-II unit
cell) to 1 mHa.

### R4, PW1PW hybrid functional [P0]

PW1PW is the workhorse functional for PT2013 and VO2019. It is
the **1-parameter global hybrid of Bredow & Gerson** (T. Bredow,
A. R. Gerson, *Phys. Rev. B* **61**, 5194 (2000), DOI
[10.1103/PhysRevB.61.5194](https://doi.org/10.1103/PhysRevB.61.5194)),
originally introduced as the "HF+PWGGA hybrid approach" in §III.D
of that paper. The mixing α = 0.20 was empirically optimised on
MgO / NiO / CoO bulk properties and *coincidentally* matched the
B3LYP exchange-mixing fraction. The "PW1PW" name came into use
in subsequent CRYSTAL-community papers (Peintinger 2013 onward).

Definition:

* `E_x = 0.20 · E_x^HF + 0.80 · E_x^PW91`
* `E_c = E_c^PW91`

Where PW91 = Perdew-Wang 91 GGA (libxc IDs `XC_GGA_X_PW91` = 109
and `XC_GGA_C_PW91` = 134). The `hf_exchange_fraction` mechanism
in `periodic_rks_multi_k_ewald.py` already supports α = 0.20
mixing; what's missing is the named entry in vibe-qc's functional
registry that resolves `"PW1PW"` to this triplet. **Status
2026-05-13:** still missing, no registry alias landed since the
audit. The 33 `.d12` parity sidecars in Phase 14h currently emit
the no-DFT (HF) form; once PW1PW resolves, the generator just
flips `method="pw1pw"` and CRYSTAL's `PW1PW` keyword does the
matching parity run.

Note: PW1PW is **distinct** from PBE0 (PBE-based hybrid, α = 0.25),
B3LYP (three-parameter Becke88 + LYP + VWN5), and plain PW91
(no HF mixing). Do not silently substitute any of these for PW1PW
in benchmarks against the pob SI tables.

**Acceptance:** `functional="PW1PW"` resolves to the Bredow-Gerson
hybrid with α = 0.20 and PWGGA (X + C) DFT components. RKS total
energy on MgO with pob-TZVP at converged k-mesh equals PT2013 SI
Table 1 (E = −275.477 595 Ha per cell) to 1 mHa. (Without R1+R2
this is testable only for cubic ionics and TM oxides.)

The Bredow-Gerson 2000 paper additionally benchmarks HF+LYP, HF
alone, B3LYP, BLYP, PBE, and plain PWGGA on MgO / NiO / CoO. If
the periodic chat wants a smaller cross-validation acceptance
target, the seven-functional comparison from that paper's
Tables I-IV is a clean test fixture: ≈30 (compound × functional)
data points at known geometry. **PDF reference:**
`/private/tmp/bredow2000.pdf` on the user's machine; user-shipped
2026-05-08.

### R5, Geometry optimisation in periodic [P0]

**Status as of 2026-05-13: still blocked.** Gradients are
present and have been extended (RKS gradients,
`periodic_gradient_rks.py`; open-shell gradients,
`periodic_gradient_open_shell.py`; multi-k gradients,
`periodic_gradient_multi_k.py`), but no atom-or-lattice
relaxation driver. Phase 14h `.d12` sidecars fix the lattice at
the experimental value so the parity test isn't *also* gated on
this requirement.

The molecular path has `_optimize_geometry` (`runner.py`); the
periodic path has gradients (`periodic_gradient*.py`) but **no
relaxation driver**. Every line of every `pob-paper` table is a
*relaxed* geometry: lattice parameters and atomic positions are
optimised against the basis. Without relaxation, comparison to
the paper tables is impossible.

Two flavours needed:

1. **Atomic-position-only relax** at fixed lattice. Required for
   T5/T6/T9/T10/T11/T8 (anything with internal coordinates), ≈40
   compounds.
2. **Lattice + atomic-position simultaneous relax**. Required for
   every cell in the test set, ≈75 compounds.

**Acceptance:** wrap the existing periodic-gradient drivers behind
an ASE BFGS / FIRE optimiser that updates both atomic positions and
the lattice tensor. NaCl (Fm-3m) with pob-TZVP/PW1PW relaxes from
5.5 Å starting cell to 5.609 ± 0.005 Å (PT2013 Table 4), atomic
positions stay at (0, 0, 0) and (½, ½, ½) by symmetry.

For the lattice-tensor side, the stress tensor is the natural
primitive but is currently *not* in vibe-qc (memory: deferred to
v0.8). A first cut can use **finite-difference lattice gradient**
(perturb each lattice parameter by ε, recompute total energy, divide)
and feed that to the BFGS optimiser. Slower than analytical stress
but unblocks the test set.

### R6, Counterpoise / BSSE for periodic crystals [P1]

VO2019's central novelty is the BSSE-aware re-optimisation. They
use CRYSTAL's `ATOMBSSE` to compute the per-atom counterpoise
correction in a periodic crystal (ghost-atom basis on the heavy
atom, no nuclei, no electrons, in the otherwise-real lattice).

vibe-qc has no equivalent yet. To **reproduce** VO2019 numbers,
needed; to **use** the rev2 basis sets, not strictly needed (the
basis already incorporates the BSSE-aware exponent shift).

For Goal 4 of `basissetdev` (re-running the rev2 optimisation in
vibe-qc), counterpoise IS required as part of the objective
function. Hence P1 not P0.

**Acceptance:** `ghost_atoms=[(symbol, position), …]` parameter on
`PeriodicSystem` that places basis functions but no nuclei / no
electrons. NaH ATOMBSSE matches CRYSTAL within 1 mHa.

## Cross-method validations

### R7, Functional zoo: PBE, PBE0, B3LYP, BLYP, HSE [P1]

PT2013 and VO2019 cite cross-validation against B3LYP and PBE0 in
side notes. vibe-qc's libxc-based functional registry should
already accept these strings (the runner advertises them). What's
missing is **end-to-end verification** at converged k-mesh +
multi-k SCF for the cubic ionic subset (which is the easy 12-compound
benchmark). Listing here so the periodic chat doesn't deprioritise.

**Acceptance:** the 12 cubic ionics from PT2013 T4 run with PBE,
PBE0, B3LYP, BLYP, HSE06 (each), in addition to PW1PW.

### R8, Wu-Cohen GGA (WC) [P2]

WC (`XC_GGA_X_WC` = 118; correlation typically PBE) is used in
PT2013 for the metallic systems (preliminary basis-set
optimisation). Easy to add, libxc has it; the question is the
registry alias.

**Acceptance:** `functional="WC"` resolves to WC-X + PBE-C.

## Numerics & screening

### R9, TOLINTEG-equivalent screening control [P1]

`linear_dependence.py` and `eigs_preflight.py` already point at
`LatticeSumOptions.schwarz_threshold` as the TOLINTEG-equivalent
knob. PT2013 explicitly mentions ITOL5 = 27 (vs default 14) for
ionic SCF convergence, i.e. extremely tight Coulomb-tail screening.
Need to verify `schwarz_threshold` covers ITOL4 (Schwarz
two-electron) AND ITOL5 (penetration / overlap-based truncation
distance). If they're conflated in vibe-qc, surface separately.

**Acceptance:** running the sulfide subset (ZnS, MnS, K₂S, CoS,
CuS) with default screening converges; bumping the equivalent of
ITOL5 to 27 closes any residual SCF oscillation.

### R10, Atomic-reference HF energies for cohesive / atomization [P1]

PT2013 Table 13 reports atomization enthalpies. To compute these
you need (a) the periodic total energy and (b) the isolated-atom
total energy in the same basis. The molecular path handles (b) at
the symmetry-broken UHF level; the question is *exposing* a
single-atom reference path that accepts a Mulliken-style fractional-
occupation specification (atoms in PT2013 are computed with
"neutral atomic configurations from our website", the .single
files in the Bredow archive).

**Acceptance:** `vq.atomic_reference_energy(symbol, basis,
configuration="neutral")` returns the SCF energy for the open-shell
atom with the published occupation, for every element in pob-TZVP.

## Lower-priority polish

### R11, D3(BJ) for periodic [P2]

`dispersion.py` exists in molecular path. VO2019 Table 14 applies
D3(BJ) on top of pob-TZVP / pob-TZVP-rev2 for the X23-subset
molecular crystals; the deltas drive ≈2-3 % lattice-constant
improvements. Periodic D3 is an additive post-SCF step on the
relaxed geometry, needs lattice summation of the C₆/C₈ pairs but
no SCF intrusion.

**Acceptance:** ZnO + D3(BJ) with pob-TZVP-rev2 reproduces VO2019
Table 14 D3-corrected `c` to 0.01 Å.

### R12, `ATOMBSSE` for the molecular hydride dimers [P2]

The 2019 BSSE-correction recipe uses **molecular** counterpoise on
H_n A-AH_n dimers (computed in ORCA 3.0 with MP2). vibe-qc's
molecular DF MP2 was wired in v0.7.3. A native vibe-qc version of
this step is needed for Goal 4 (re-running the rev2 optimisation
without external ORCA dependencies).

**Acceptance:** counterpoise(H₂O)₂ at 2.5 Å with vibe-qc/MP2
matches ORCA 3.0 MP2 to 1 mHa.

## Basis-optimization gradient infrastructure

### R13, Lattice-summed exponent/coefficient-derivative 3c/2c kernel [P1]

**Status as of 2026-06-21: ask raised; not started.** New requirement from the
periodic basis-optimization gradient effort (the crystal OPTBASIS analogue; see
[`PERIODIC_GRADIENT_DESIGN.md`](PERIODIC_GRADIENT_DESIGN.md)).

The periodic basis-parameter gradient assembles, per k-point,

    dE/d(eta) = Sum_k w_k Re[ tr(P dHcore/d(eta)) + 1/2 tr(P dG/d(eta))
                              - tr(W dS/d(eta)) ].

The clean one-electron pieces dS(k)/d(eta) and dT(k)/d(eta) are **done** (Phase
P1) in Python, by Bloch-summing the molecular `overlap_/kinetic_exponent_
derivative` bindings between a home-cell shell and its image. The hard remaining
piece (Phase P2) is the two-electron Coulomb derivative dG(k)/d(eta) and the
nuclear derivative dV(k)/d(eta). Both are built from the **lattice-summed**
GDF/Ewald 3-index (P|mu nu) and 2-index (P|Q) integrals with the Ewald-3D gauge,
so their basis-parameter derivative needs the d/d(eta) sibling of the existing
d/dR gradient kernels `compute_3c_eri_lattice_gradient_weighted` /
`compute_2c_eri_lattice_gradient_weighted` (used today by
[`periodic_gdf_gradient.py`](../../python/vibeqc/periodic_gdf_gradient.py) for
nuclear gradients).

This is the one piece of the periodic basis-gradient that lives in the
periodic-SCF integral layer (`cpp/`), which basissetdev does not own, hence this
ask (CLAUDE.md s11). It is **not** a hard blocker: a first working basis-gradient
can use an integral-FD dG(k)/dV(k) behind the assembly's provider with no C++
change. R13 enables the fully analytic, production route. Marked P1 (blocks
cross-method validation of the periodic optimizer), not P0.

**Acceptance:** a binding analogous to
`compute_3c_eri_lattice_gradient_weighted` that, given a target shell and a
primitive index (plus a coefficient variant), returns the weighted contraction
of the lattice-summed 3c/2c integral derivative w.r.t. that primitive's exponent
(or coefficient). Validated to 1e-6 or better vs a central integral FD of the
Bloch-summed 3c/2c tensor at fixed k, on a small cell. Until it lands,
basissetdev proceeds with the integral-FD fallback behind the provider.

## Out of scope here (other chats / future versions)

- **Periodic stress tensor.** Memory: reserved for vibe-qc v0.8.
  Once landed, R5 lattice opt becomes analytical instead of FD.
- **Slab / surface support.** Same: v0.8 deliverable.
- **AutoAux periodic aux basis.** Owned by the DF dev chat (memory:
  `feedback_aux_basis_routing.md`).
- **CCM / cyclic-cluster method.** Memory: deferred to a later
  vibe-qc version.

## Suggested implementation order

The order that unblocks the most of the test set per landed feature:

1. **R5 atomic-position relax** + **R4 PW1PW** + **R7 functional
   verification** → unlocks 12 cubic ionics from PT2013 T4 +
   12 cubic ionics from VO2019 T2 (overlap) + 22 cubic
   semiconductors from PT2013 T8 + 7 cubic TM oxides from PT2013
   T10 = ≈40 compounds runnable.
2. **R3 broken-symmetry UKS guess** → unlocks the 10 TM oxide /
   chloride / nitride compounds.
3. **R1 non-orthorhombic Ewald** → unlocks the 32 hexagonal/
   tetragonal compounds. Big single-feature payoff.
4. **R5b lattice relax** (FD lattice gradient) → enables the actual
   lattice-constant comparison vs. paper tables for everything
   above.
5. **R2 metallic k-mesh smearing** → unlocks the 12 cubic metals.
6. **R6 ATOMBSSE periodic** + **R12 molecular counterpoise** →
   unlocks the rev2 re-optimisation pipeline for Goal 4.

## Status table (as of refresh on `1bf32cc`, 2026-05-13)

| Req | Title | State |
|-----|-------|-------|
| R1 | Non-orthorhombic Ewald | shipped (FFT Poisson kernel handles orthorhombic/monoclinic/triclinic uniformly; hexagonal/tetragonal covered by `tests/test_periodic_lattice_families.py`) -- basissetdev's specific Al2O3-vs-PySCF acceptance number not yet re-confirmed |
| R2 | Multi-k MP-grid SCF | partial, native GDF + smearing + FMIXING landed; production-size CRYSTAL14 parity bench outstanding |
| R3 | Open-shell UKS for AFM TM oxides | partial, kernels exist, BS guess (ATOMSPIN) shipped 2026-06-14/16 -- basissetdev PT2013 parity check not yet re-confirmed |
| R4 | PW1PW hybrid functional | partial, `hf_exchange_fraction` exists, registry alias still missing |
| R5 | Periodic geometry / lattice optimisation | **MISSING**, gradients exist (RKS / open-shell / multi-k), no relax driver |
| R6 | Counterpoise / BSSE periodic | **MISSING** |
| R7 | PBE/PBE0/B3LYP/BLYP/HSE verification | partial, registry exists, end-to-end untested at production size |
| R8 | WC GGA functional | partial, libxc has it, registry alias TBD |
| R9 | TOLINTEG-equivalent screening control | partial, `schwarz_threshold` exists, ITOL4/5 conflation TBD |
| R10 | Atomic-reference HF for atomisation | partial, molecular UHF works, fractional-occ wrapper missing |
| R11 | D3(BJ) for periodic | partial, molecular path exists, periodic glue missing |
| R12 | ATOMBSSE molecular dimer | partial, molecular MP2 works, counterpoise glue missing |
| R13 | Lattice-summed exponent/coeff-deriv 3c/2c kernel | **NOT STARTED**, ask raised 2026-06-21; integral-FD-behind-provider fallback unblocks P2 meanwhile |

### Shared infrastructure now in `main` (basissetdev reuses, never duplicates)

| Header / module | Provides | Notes |
|-----------------|----------|-------|
| `cpp/include/vibeqc/scf_mixing.hpp` | `validate_fraction_01`, `mix_fock_matrices` | Canonical Fock-mixing primitives; molecular UHF/UKS + periodic GDF/Ewald all dispatch here. |
| `python/vibeqc/periodic_rhf_gdf.py` | Γ-only RHF GDF backend | Native; no PySCF runtime dependency. |
| `python/vibeqc/periodic_k_gdf.py` | Multi-k GDF | Cubic / orthorhombic / low-dim lattice families covered. |
| `python/vibeqc/periodic_gdf_blocks.py` | Bloch assembly from lattice blocks | Underpins both Γ and multi-k. |
| `examples/regression/runner_pyscf.py` | Out-of-process PySCF parity | Manifest § 10: PySCF is **external**; vibe-qc never imports it. |
| `examples/basisset_dev/inputs/*.d12` (Phase 14h) | CRYSTAL14 parity sidecars | 33 cubic ionic / semiconductor / TM-compound `.d12` decks; runs out-of-process via `vq submit` + `run-crystal.sh` (memory: `reference_crystal14_via_vq.md`). |

## Hand-off note

This document is the deliverable from `basissetdev` to the
periodic-methods chat. It does not prescribe how each requirement
is implemented; that is the periodic chat's call. When a
requirement lands, mark it ✅ here and update
[PLAN.md](PLAN.md) Goal 2 status. `basissetdev` rebases onto
`main` after each landing.
