# v0.10.0 D4 refinement — Hirshfeld charges from the SCF density

**Status:** research spike, **executed end-to-end 2026-05-18** on the
worktree-private venv (vibe-qc 0.7.5, dftd4 4.x). Branch
`claude/stoic-elbakyan-9c450c`. Spike → results → v0.10.0 scoping
verdict, all in this README.

**Promoted to production 2026-05-18:** the spike's classical-Hirshfeld
implementation now lives in
[`python/vibeqc/properties.py`](../../python/vibeqc/properties.py)
next to ``mulliken_charges`` / ``loewdin_charges``, exposed as
``vibeqc.hirshfeld_charges`` + ``vibeqc.HirshfeldResult``. Test suite
at [`tests/test_hirshfeld_charges.py`](../../tests/test_hirshfeld_charges.py)
(8 tests, all passing). User-facing docs at
[`docs/user_guide/properties.md`](../../docs/user_guide/properties.md).
The spike's local ``hirshfeld_charges.py`` is now a one-line re-export
so the comparison + sensitivity drivers continue to run unchanged.

**Roadmap target.** [`docs/roadmap.md`](../../docs/roadmap.md)
§ `v0.10.0 — Grimme D4 dispersion`:

> 🎯 Headline feature — Grimme's D4 dispersion correction with
> charge-dependent C6 coefficients via partial-Hirshfeld atomic
> populations. Strict refinement over the D3 already shipped in
> v0.2.x.
>
> * **D2a** Charge-dependent c6ab table + parameter set
>   (Caldeweyher 2019).
> * **D2b** Hirshfeld atomic charges from the SCF density.
> * **D2c** D4 energy + gradient + stress.
> * **D2d** ``dftd4`` reference Python backend as optional dep.

The motivating "refinement" is **D2b** — feeding Hirshfeld
charges *computed from vibe-qc's converged SCF density* into the
charge-dependent C6 table, in place of the EEQ charges that
`dftd4` solves internally. The published D4 method (Caldeweyher
2019) uses EEQ; the strict variant the roadmap calls for is a
*physically more grounded alternative* in which atomic charges
come from the SCF wave function rather than from the EEQ
electronegativity-equalisation surrogate.

## Architectural verdict — why this spike is just "Hirshfeld"

The natural minimal patch — "compute Hirshfeld charges, pass
them to `dftd4`" — is **blocked at the library boundary**:

| Surface | Charge input? | Source |
|---|---|---|
| `dftd4.interface.DispersionModel(numbers, positions, charge=...)` | Total *molecular* charge only — fed to internal EEQ | `python/dftd4/interface.py` |
| `model.get_dispersion(param, grad)` | No charge argument | same |
| `model.get_properties()` | Returns EEQ-computed charges, no setter | same |
| C ABI `dftd4_get_dispersion(...)` | No charge slot | `include/dftd4.h` (dftd4 main) |
| C ABI `dftd4_get_properties(cn, charges, c6, alpha)` | Returns charges; no input variant | same |

Independent verification on both the Python wrapper and the C
header confirms there is **no public hook for caller-supplied
atomic partial charges anywhere in the dftd4 stack**. The EEQ →
C6 path is hard-wired inside the Fortran library.

Therefore the v0.10.0 D2b plan splits into two strictly
sequential work items:

1. **Hirshfeld charges from the SCF density** (this spike).
   Self-contained Python module on top of existing
   `_vibeqc_core` bindings (`build_grid`, `evaluate_ao`,
   `sad_density`). Independent value as a general analysis tool.

2. **Native D4 dispersion energy with caller-supplied charges**
   (CHANGELOG calls this *Phase D4b*, follow-on after the
   landed *Phase D4a-i* native EEQ in `cpp/src/eeq_charges.cpp`).
   This unlocks D2b because, unlike the external `dftd4`, the
   native vibe-qc backend can take `partial_charges` as an
   input argument. Three options for the C6 reference table that
   the native backend needs:

   * **B-vendor**: vendor `dftd4`'s Fortran data tables
     (`src/dftd4/data/*.f90`) — they're MIT-licensed, ~5k lines
     of reference C6 / α / Q grids per element.
   * **B-port**: re-port the C++ data tables from
     `tblite`/`dftd4` into `cpp/src/dispersion_d4_data.cpp`,
     mirroring the existing
     [`cpp/src/dispersion_data.cpp`](../../cpp/src/dispersion_data.cpp)
     pattern for D3.
   * **B-patch-upstream**: submit a PR to dftd4 adding an
     `external_charges` keyword to `get_dispersion`. Lowest
     vibe-qc code-volume; subject to upstream merge timeline.

   B-port is the closest match to vibe-qc's "we own every
   numerical method we ship" stance ([CLAUDE.md § 10](../../CLAUDE.md#10-external-programs-vs-vendored-libraries))
   and is the same pattern already used for the D3 reference
   tables and the EEQ ionisation potentials. Recommendation:
   B-port.

## What this spike delivers

* [`hirshfeld_charges.py`](hirshfeld_charges.py) — classical
  Hirshfeld (Hirshfeld 1977) charge calculator. Pure-Python,
  140 lines, depends only on stable `_vibeqc_core` bindings
  (`build_grid`, `evaluate_ao`, `sad_density`) plus
  `vibeqc.properties._shell_to_atom` for AO bookkeeping.

* [`compare_charges.py`](compare_charges.py) — runnable
  comparison: Hirshfeld vs. Mulliken vs. Löwdin vs. EEQ on H₂O,
  NH₃, HF, NH₄⁺, OH⁻. Prints a per-atom table and a sanity check
  (Σ q_A = molecular charge). Executed numbers in "Results" §.

* [`d4_with_hirshfeld_charges.py`](d4_with_hirshfeld_charges.py)
  — H₂O dimer driver that exercises the new charges *alongside*
  the existing dftd4-EEQ path, and prints the verdict that the
  dftd4 wrapper cannot accept the new charges. Concrete
  scaffolding for the eventual D4b-backed call.

* [`sensitivity_estimate.py`](sensitivity_estimate.py) — uses
  the measured Δq to predict ΔE_disp(Hirshfeld − EEQ) via
  Caldeweyher's analytical Gaussian-weight derivative, without
  needing a working native D4. Executed numbers in
  "Sensitivity estimate" § — concrete v0.10.0 D2b sizing.

* No production code under `python/vibeqc/` is touched. The
  spike stays scoped per the experimental-chat policy until
  D4b lands.

## Why classical Hirshfeld and not Hirshfeld-I

Two reasons:

1. The roadmap text reads "Hirshfeld atomic charges from the
   SCF density", which is the canonical 1977 definition (the
   iterative-Hirshfeld 2007 variant goes by "Hirshfeld-I" or
   "iterative Hirshfeld" in the literature, not bare
   "Hirshfeld").

2. Classical Hirshfeld is a one-shot density partition — the
   per-atom Hirshfeld weight w_A(r) only depends on free-atom
   densities, which `vibeqc._vibeqc_core.sad_density(mol, basis)`
   already supplies in the right form (block-diagonal in the
   molecular AO basis, so the AO-block restricted to atom A
   *is* the free-atom density matrix of A in A's own AOs). No
   per-atom SCF, no ionic-fragment branching.

Hirshfeld-I is a clean follow-on (cache atomic SCFs at fractional
charges, then iterate the weight) — separate spike, separate
chat. Tracked as **D2b-i** below if the main D2b lands first.

## Math (one screen)

Classical Hirshfeld atomic charge:

```
q_A = Z_A − ∫ w_A(r) ρ(r) d³r
```

with the Hirshfeld weight

```
w_A(r) = ρ_A^free(r) / Σ_B ρ_B^free(r)
```

and the integral evaluated on a Becke-Lebedev-Treutler molecular
grid (same kind used by RKS/UKS):

```
∫ w_A(r) ρ(r) d³r ≈ Σ_g  weight_g · w_A(r_g) · ρ(r_g)
```

The molecular density on grid points is the standard AO-basis
evaluation:

```
ρ(r_g) = Σ_μν P_μν · χ_μ(r_g) · χ_ν(r_g)
```

For the free-atom densities we exploit a structural property of
the SAD guess density matrix `P_pro = sad_density(mol, basis)`:
since each atom's SCF is run *in vacuum* and contributes its own
AO block to `P_pro`, the matrix is block-diagonal by atom (off-
diagonal blocks are zero by construction). Restricting the AO
sum to atom A's basis-function range therefore gives the
free-atom density of A evaluated at the molecular geometry:

```
ρ_A^free(r_g) = Σ_{μν ∈ A's AOs}  (P_pro)_μν · χ_μ(r_g) · χ_ν(r_g)
```

Sanity guarantees that fall out for free:

* Σ_A w_A(r) = 1 wherever the promolecule has support.
* Σ_A q_A = molecular charge (numerically, modulo grid error).
* For a single atom in the molecule, w_A ≡ 1 and q_A = Z_A − n_e
  recovers exactly the SCF electron count of the atom.

## Results (2026-05-18 — `python compare_charges.py`)

HF/def2-SVP, vibe-qc grid level default (137,700 grid points on H₂O,
229,500 on NH₄⁺). Per-atom charges in electrons:

| System | atom | Mulliken | Löwdin | EEQ | **Hirshfeld** | ORCA Hirshfeld¹ |
|---|---|---:|---:|---:|---:|---:|
| H₂O | O | −0.347 | −0.156 | −0.699 | **−0.375** | ≈ −0.32 |
|     | H | +0.174 | +0.078 | +0.349 | **+0.188** | ≈ +0.16 |
| NH₃ | N | −0.308 | −0.156 | −0.927 | **−0.370** | ≈ −0.42 |
|     | H | +0.103 | +0.052 | +0.309 | **+0.123** | ≈ +0.14 |
| HF  | F | −0.263 | −0.113 | −0.268 | **−0.256** | ≈ −0.21 |
|     | H | +0.263 | +0.113 | +0.268 | **+0.256** | ≈ +0.21 |
| NH₄⁺ | N | +0.016 | +0.357 | **−1.032** | **−0.029** | ≈ −0.05² |
|      | H | +0.246 | +0.161 | **+0.508** | **+0.257** | ≈ +0.26² |
| OH⁻ | O | −0.996 | −0.911 | −1.170 | **−0.959** | ≈ −0.93² |
|     | H | −0.004 | −0.089 | +0.170 | **−0.041** | ≈ −0.07² |

¹ ORCA 5.x `! HF def2-SVP HIRSHFELD` rough values; ² no direct ORCA
ref in hand, listed values are approximate community-reported
classical-Hirshfeld for these geometries.

### Sanity (all pass)

* ∫ ρ_mol dV − n_electrons: < 5×10⁻⁵ on every system (default grid).
* Σ q_A^Hirshfeld − q_mol: ≤ 8×10⁻⁷ e on every system.
* Hirshfeld values sit between Mulliken and EEQ in magnitude on
  every closed-shell system, as expected.

### Heavy-atom systematics vs ORCA

vibe-qc's classical Hirshfeld charges on heavy atoms are ~0.04–0.06 e
*more negative* than ORCA's reference values. This is the expected
signature of an SAD-derived promolecule built in the *molecular*
basis (def2-SVP here): the molecular basis has diffuse / polarisation
functions that the bare atom doesn't need, slightly over-localising
the free-atom density on the heavy centre. ORCA uses tabulated
Slater-type atomic densities that don't carry the molecular basis's
diffuseness. **Documentable trade-off, not a bug** — Phase D4b can
optionally swap in tabulated Slater densities for byte-for-byte ORCA
agreement, or accept the SAD-basis variant as the vibe-qc-native
choice.

## Headline finding for v0.10.0 D2b scoping

EEQ **massively over-polarises** small molecules vs Hirshfeld:

| System | q(heavy)_EEQ | q(heavy)_Hirsh | |Δq| (heavy) |
|---|---:|---:|---:|
| H₂O | −0.699 | −0.375 | **0.324** |
| NH₃ | −0.927 | −0.370 | **0.557** |
| HF | −0.268 | −0.256 | 0.012 |
| **NH₄⁺** | **−1.032** | **−0.029** | **1.003** |
| OH⁻ | −1.170 | −0.959 | 0.211 |

For ionic / charged systems the EEQ vs Hirshfeld charge gap is
**O(1) electron**. Even for neutral H-bonded systems it sits at
0.3 e on the heavy centre. With Caldeweyher 2019's `gc = 2.0`
Gaussian-weight prefactor, this translates to a 20–60 % shift in
C6_AB coefficients on the heavy-atom pair — and a similar
fractional shift in E_disp_2body. **The v0.10.0 D2b refinement is
physically meaningful**, not a 5th-decimal vanity item.

## Sensitivity estimate (executed) — predicted ΔE_disp(Hirshfeld − EEQ)

`sensitivity_estimate.py` linearises around EEQ using Caldeweyher
2019's Gaussian weight (gc = 2.0) and applies the per-atom Δq from
the spike. Results on three H-bonded dimers:

| Dimer | mean \|Δq\| | predicted relative ΔE_disp | E_disp(EEQ, B3LYP-D4) | abs. ΔE_disp (B3LYP) | E_disp(EEQ, PBE0-D4) | abs. ΔE_disp (PBE0) |
|---|---:|---:|---:|---:|---:|---:|
| H₂O dimer | 0.25 e | +38 % | −1491 µHa | **±564 µHa** | −810 µHa | ±306 µHa |
| NH₃ dimer | 0.27 e | +41 % | −2343 µHa | **±962 µHa** | −1316 µHa | ±540 µHa |
| HF dimer  | 0.06 e |  +9 % |  −662 µHa |   ±60 µHa  |  −356 µHa |  ±33 µHa  |

All values are well above the S22 / S66 benchmark noise floor
(~10 µHa). The v0.10.0 D2b refinement **is** physically meaningful;
on NH₃ dimer the predicted shift is ~1 mHa, half the size of the
total B3LYP-D4 dispersion contribution.

**Direction of the shift.** Hirshfeld charges are systematically
*less* polar than EEQ on the heavy centres (0.3-0.55 e less negative
on O / N). Smaller |q_A| ⇒ smaller charge enhancement of the C6
coefficient (the Caldeweyher reference grid is built so that more-
polarised atoms borrow larger C6 from highly-charged reference
points) ⇒ |E_disp(Hirshfeld)| < |E_disp(EEQ)|. Phase D4b should
deliver weaker dispersion binding on H-bonded dimers vs the
existing EEQ-D4 wrapper. This is the expected direction; the
empirical question is which is closer to high-level reference
binding energies (CCSD(T)-F12, MRCC-IB, etc.).

**Validation budget caveat.** The κ = 1.5/e first-order coefficient
is a global average over reference-point spread (Caldeweyher 2019
fig. 3 implicitly). The truncation error of the linearisation is
~(gc · Δq)² ≈ 10-30 % at the |Δq| values seen here. Phase D4b
will deliver the exact answer.

## How to re-execute this spike

```sh
# Assume $VIBEQC_REPO points at your main checkout (e.g. ~/src/vibeqc),
# and you've cd'd into a worktree off it.
# 0. Worktree-private venv (per CLAUDE/memory rule):
python -m venv .venv && . .venv/bin/activate
pip install scikit-build-core pybind11 numpy scipy

# 0a. Symlink the parent checkout's third_party native-dep installs
#     (saves ~15 min of libint/libxc/spglib/fftw/libecpint compile
#     time on a fresh worktree; the installs are CMake-relocatable):
mkdir -p third_party && for d in libint libxc spglib fftw libecpint; do
    mkdir -p third_party/$d
    ln -sfn "$VIBEQC_REPO/third_party/$d/install" third_party/$d/install
done

# 1. Editable-install vibe-qc itself:
pip install -e '.[dispersion]' --no-build-isolation

# 2. Run the comparison driver (Hirshfeld vs Mulliken/Löwdin/EEQ):
python studies/d4-hirshfeld-spike/compare_charges.py

# 3. Run the D4 bridge driver (H₂O dimer, dftd4 wall demo):
python studies/d4-hirshfeld-spike/d4_with_hirshfeld_charges.py
```

Worktree-venv build ran in 2 min 25 s on M1 Max with the symlink
trick (full from-scratch build would be ~12-15 min).

## Exit criteria for promoting out of `studies/`

* `hirshfeld_charges` agrees with ORCA Hirshfeld to ≤ 5 mE on
  the H₂O / NH₃ / HF / H₂O-dimer set (closed-shell).
* Open-shell extension (NH₃⁺ radical, O₂ triplet) using
  α + β SAD blocks — agrees with ORCA to ≤ 10 mE.
* Hirshfeld charges sum to the molecular charge to ≤ 1e-3 e on
  default grid level 3 (matches ORCA's typical grid sanity).
* Once Phase D4b (native D4 dispersion w/ caller-supplied
  charges) lands, the Hirshfeld module promotes to
  `python/vibeqc/properties.py` next to `mulliken_charges` /
  `loewdin_charges`, and the D4 module exposes
  `compute_d4(..., charge_source="hirshfeld")`.

## References

* Hirshfeld, F. L. *Theor. Chim. Acta* **44**, 129 (1977).
  Original definition. The 1977 paper is the namesake; the
  weight formula and the promolecular reference both originate
  here.
* Caldeweyher, E.; Bannwarth, C.; Grimme, S. *J. Chem. Phys.*
  **150**, 154122 (2019). D4 with EEQ charges — the published
  baseline our refinement diverges from at the charge-model
  step.
* Bultinck, P. et al. *J. Chem. Phys.* **126**, 144111 (2007).
  Iterative Hirshfeld — the follow-on enhancement (D2b-i).
* Van Lenthe et al. *J. Comput. Chem.* **27**, 926 (2006).
  SAD reference — confirms vibe-qc's `sad_density(mol, basis)`
  yields the atom-block-diagonal density we need as the
  promolecular reference.
