# Design proposal, analytic SCF-energy gradient w.r.t. basis parameters

**Status: Phase 1 complete and validated (RHF).** Both phases of the phased approach
(§4) have landed. Phase 0 = integral-level FD + analytic assembly; Phase 1 = the
closed-form analytic exponent-derivative integrals feeding the same assembly.

* **Phase 0**, the build-independent **assembly**
  (`vibeqc.basis_optimization.energy_gradient`) verified against a self-contained mock
  RHF (`test_energy_gradient_assembly.py`) and, on a build (**compute-small, 2026-06-17**,
  compute-reference down), via the real `VibeqcIntegralProvider`: H₂ / pob-TZVP / RHF matched the
  full re-SCF total-energy FD to **|Δ| ≤ 1.5×10⁻⁷ Ha**.
* **Phase 1**, analytic ∂S/∂T/∂V/∂(μν|λσ) w.r.t. a primitive exponent
  (`cpp/src/basis_param_gradient.cpp`, bound as `*_exponent_derivative`), validated to
  ≤1e-9 vs FD on H/C s,p,d. `energy_gradient_analytic` assembles them into the full RHF
  gradient and matches full-energy FD to ≤5e-9 Ha on Be / H₂ / H₂O (single-atom,
  multi-atom summing, multi-element + d-shell). Wired into the optimiser via
  `make_rhf_native_objective_gradient`, `optimize_bdiis(obj, x0, grad=…)` is a fully
  analytic in-process loop (one SCF per gradient vs the FD path's 2·N_param). Developed
  on a **local build** (`scripts/update.sh --dev`), which superseded the earlier
  build-gate. Exponent-only so far; coefficient derivatives, and the KS (∂E_xc) +
  periodic extensions, remain.

## 1. Why

`optimize_bdiis` currently finite-differences the SCF energy term of the objective
Ω = E + penalty. FD costs `2·n_params` SCF solves per gradient and is noisy. The
penalty term already has an analytic gradient
([`gradients.py`](../../python/vibeqc/basis_optimization/gradients.py), commit
`50d73b99`). The remaining piece is the **analytic energy gradient** dE/dη w.r.t.
each free basis parameter η (a Gaussian exponent α or a contraction coefficient c).
This is what makes BDIIS cheap on real periodic SCF and is the headline accuracy
claim for the basis-optimisation paper.

## 2. The math, it is the nuclear gradient with ∂/∂R → ∂/∂η

For RHF (closed shell), with density P = 2·Σ_i^occ CᵢCᵢᵀ and energy-weighted
density W = 2·Σ_i^occ εᵢ CᵢCᵢᵀ:

    dE/dη = Σ_μν P_μν (∂T_μν/∂η + ∂V_μν/∂η)
          + ½ Σ_μνλσ P_μν P_λσ ∂(μν‖λσ)/∂η
          − Σ_μν W_μν ∂S_μν/∂η
          + ∂E_xc/∂η            [KS only]

where (μν‖λσ) = (μν|λσ) − ½(μσ|λν) for RHF (hybrid-scaled for KS), and the nuclear
repulsion E_nn does not depend on basis parameters. The `−Σ W ∂S/∂η` Pulay term
captures the wavefunction response exactly (variational HF/KS ⇒ no CPHF needed), the
same reason it works for nuclear gradients.

**This is structurally identical to the existing nuclear gradient.** vibe-qc already
assembles W and the Pulay term in [`cpp/src/gradient.cpp`](../../cpp/src/gradient.cpp)
(`W = 2 Σ ε_i C_μi C_νi`, line ~607) and contracts P/W against libint
`deriv_order=1` integral derivatives ([`gradient.hpp`](../../cpp/include/vibeqc/gradient.hpp)).
The periodic analogue is [`periodic_gradient.cpp`](../../cpp/include/vibeqc/periodic_gradient.hpp)
(lattice-sum Pulay). **We reuse all of that.** Only the integral-derivative *source*
changes: `∂/∂R` → `∂/∂η`.

## 3. The only new piece, integral derivatives w.r.t. η

libint provides geometric derivatives `∂/∂R` (`deriv_order=1`) natively; it does **not**
provide exponent derivatives. Two sub-cases:

* **Contraction coefficients (easy).** A contracted integral is *linear* in its
  contraction coefficients: I = Σ_p c_p·(primitive_p term). So ∂I/∂c_p is just the
  integral re-contracted with a unit coefficient on primitive `p` and zero elsewhere
  (carrying the per-primitive normalisation). **No new integral type**, reuse the
  existing libint path with a modified contraction. This already matches the
  closed-form coefficient gradient validated for the *penalty* in `gradients.py`.

* **Exponents (the real work).** ∂/∂α of a normalised primitive splits into a
  normalisation-derivative term plus −r²·(primitive). Since r²·g_l raises angular
  momentum, ∂I/∂α is a linear combination of integrals over the same shells with the
  varied shell promoted l → l+2 (plus l and lower terms from normalisation). libint
  computes arbitrary-L integrals, so the l+2 integrals are available; the exponent
  derivative is assembled in our layer from them. This is the standard
  shell-incrementation route used by derivative-Gaussian theory.

## 4. Decision point, integral-derivative strategy (needs maintainer input)

| Option | What | Cost / accuracy | Effort |
|--------|------|-----------------|--------|
| **A. Integral-level central FD** | perturb α, recompute integrals via libint, FD | ~2 integral builds/param; FD-grade | low |
| **B. Analytic shell-incrementation** | assemble ∂I/∂α from libint l+2 integrals + normalisation term | exact; reusable | high (C++) |
| **C. Coefficient derivatives** | unit-recontraction (both A and B need this; trivial) | exact | trivial |

**Recommendation:** ship in two phases.

* **Phase 0, Option A (integral-level FD) + the analytic assembly.** Gets a *correct,
  validated* W-Pulay assembly pipeline end-to-end fast, reusing `gradient.cpp`'s P/W
  contraction. Even at FD-grade integral derivatives this validates everything except
  the exponent-derivative integrals themselves, and is the reference for Phase 1.
* **Phase 1, Option B (analytic shell-incrementation)** for the exponent integrals →
  the production gradient. Coefficient derivatives (C) are exact in both phases.

Resolved (investigation 2026-06-17 against the C++ tree):

1. **A-then-B**, A (Phase 0) is done and verified; B is Phase 1.
2. **Where: C++, confirmed required.** The exponent derivative needs per-shell /
   per-primitive control over `libint2::Shell` objects (build l+2-promoted auxiliary
   shells, combine with the −r² + normalisation terms). vibe-qc drives libint2 directly
   - `basis.libint()` returns the `libint2::BasisSet`, `gradient.cpp` builds
   `libint2::Engine(Operator, max_nprim, max_l, deriv_order)` and loops shell pairs,
   and `ao_eval.cpp` reads `shell.alpha` / `shell.contr`. The **Python bindings expose
   only contracted, whole-molecule integrals** (`compute_overlap(basis)` etc.), no
   per-shell hook, so a Python l+2 route is *not* available. Phase 1 lands in a new
   `cpp/src/basis_param_gradient.cpp` reusing the `gradient.cpp` engine+shell-loop
   pattern, exposed via a binding, behind an `AnalyticIntegralProvider` swappable for
   `VibeqcIntegralProvider`.
3. **First cut: HF molecular**, deferring KS (`∂E_xc/∂η`) and periodic (reuse
   `periodic_gradient`'s lattice-sum Pulay).
4. **libint capability: yes.** The engine is constructed with `max_l`, and shells are
   built/iterated as `libint2::Shell` (see `gradient.cpp`, `ao_eval.cpp`, `schwarz.hpp`
   `shift_shells_to_cell`), so arbitrary-L / ad-hoc shells are available in C++.

### Build constraint (the gate on Phase 1)

Phase 1 is integral-engine **C++** that must **compile + pass tests before it lands on
`main`** (CLAUDE.md § 14: the `build-test` CI job is the build-break gate). Verifying C++
on compute-reference/compute-small via `vq` does **not** help here: `vq` runs the *already-deployed* build,
and `--refresh` rebuilds *from `main`*, so the C++ would have to be pushed to `main`
*before* it could be compiled there, which is exactly the pattern § 14 forbids (an
un-compiling push reds the build gate for everyone). Phase 0 was developable build-free
because it is pure Python; Phase 1 is not.

**Therefore Phase 1 needs a build environment for develop+validate** (a local toolchain,
or the maintainer building on a dev host with CI gating), not the `vq` payload path that
served Phase 0. The design above is complete and grounded; the implementation is the
build-gated next step. The Phase-0 integral-FD gradient is the ready-made numerical
reference to validate it against.

## 5. Integration with what exists

The result plugs straight into the optimiser as a `grad=` callable. The composed
gradient is `energy_grad(x) + penalty_grad(x)`, where:

* `penalty_grad` = `vibeqc.basis_optimization.penalty_gradient(...)` (already landed,
  analytic).
* `energy_grad` = new analytic energy gradient (this proposal), mapped to optimiser
  space by the same log/linear chain rule the penalty gradient uses, and to the
  per-(shell, primitive, field) free-spec layout of `BasisParametrisation`.

The reusable kernel from `gradients.py`, the analytic same-center `∂S/∂α`, is exactly
the atomic block of the molecular `∂S/∂η` Pulay term, so it doubles as an independent
unit check on the new multi-centre overlap derivative.

## 6. Validation plan

1. **Integral level:** chosen `∂I/∂η` vs central FD of libint integrals; the
   `gradients.py` same-center analytic `∂S/∂α` as an independent check on the atomic block.
2. **Energy level:** dE/dη vs central FD of the SCF energy (the current BDIIS FD path)
   to ≤ 1e-6 Ha per unit η, on single-atom HF then a small molecule.
3. **Parity:** optimise a known case and compare to CRYSTAL OPTBASIS (the `vibe-basis`
   parity runs drive CRYSTAL out-of-process for the reference, per CLAUDE.md § 10).

## 7. Phasing summary

```
P0  HF, integral-level FD exponent derivs + analytic W-Pulay assembly  (validates pipeline)
P1  HF, analytic shell-incrementation exponent derivs                  (production gradient)
P2  KS (add ∂E_xc/∂η)  →  periodic (lattice-sum Pulay)                  (coverage)
```

Each phase is independently landable and FD-validated before the next. All phases need a
build; implementation waits on the §4 decisions.
