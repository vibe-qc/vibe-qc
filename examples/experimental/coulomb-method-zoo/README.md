# coulomb-method-zoo: alternative periodic Coulomb J/K methods

Experimental cross-pollination spike. Throwaway. See
`POSTMORTEM.md` (written at the end) for the verdict.

## What this benchmarks

The release-path periodic-SCF driver `run_rhf_periodic_gamma_ewald3d`
builds the Hartree matrix as

    J(D) = J_SR(ω, D) + J_LR(ω, D)

with **J_SR = erfc(ω·r)/r** in real space (analytic Gaussian ERIs at
the Γ point, lattice-summed) and **J_LR = erf(ω·r)/r** in reciprocal
space via FFT-Poisson on a real-space density grid (`build_j_long_range`
in `python/vibeqc/ewald_j.py`). Exchange K is full-range, real-space.

This spike asks: **what are the alternative ways to build J for a
periodic Gaussian-AO density, and how do they compare?** All four
candidates from the system prompt:

| label        | math                                                     | ref                       |
|--------------|----------------------------------------------------------|---------------------------|
| `EWALD3D`    | erfc·r real-space + erf·r FFT-Poisson (existing)         | vibe-qc release           |
| `WOLF`       | full 1/r real-space, smooth cutoff, no reciprocal        | Wolf JCP 110, 8254 (1999) |
| `PLAIN_EWALD`| analytic reciprocal-space Gaussian Ewald (no FFT)        | de Leeuw-Perram-Smith     |
| `ADFT`       | variational Coulomb fitting on aux basis (no Ewald)      | Mintmire-Dunlap, Köster   |

All exchange K is built with the same molecular-limit real-space
builder (`build_jk_gamma_molecular_limit`) so we are comparing the
Coulomb math, not the exchange math.

## Test systems

We are deliberately not using MgO sto-3g, per
`project_periodic_rhf_gdf_spike` memory, the existing `_ewald3d`
driver is structurally broken on MgO (oscillating, +241 Ha off PySCF,
6229 s wall). A spike comparing alternative J builders against a
broken reference would teach us nothing. Instead:

- **H₂ in a 12-bohr vacuum cubic box / sto-3g**. Smallest molecular-
  limit test where periodic and molecular SCF are both well-defined
  and the existing Ewald-3D path is converged. This is the
  primary benchmark.
- **LiH in a 16-bohr vacuum cubic box / sto-3g**. One element heavier,
  ionic, still molecular-limit. A check that the methods that work
  on H₂ also handle a Li 1s core.

The molecular-limit choice is forced by the molecular-DF integral
machinery, `compute_3c_eri` is molecular, not lattice-summed, so
true bulk geometries (tight unit cells with cell-to-cell density
overlap) need a separate native-periodic 3c-ERI binding that doesn't
exist yet (per `docs/handover_periodic_scf_followup.md`).

## Math summary

### Plain Ewald (analytic reciprocal-space)

Same erf/erfc split as `EWALD3D`, but the long-range piece is

    J_LR_μν(D) = Σ_{G ≠ 0} (4π / |G|²) e^{-|G|²/(4ω²)} ρ̂*(G) ρ̂_μν(G) / V

where ρ̂_μν(G) = ∫ χ_μ χ_ν e^{-iG·r} dr is the analytic Fourier
transform of an AO product (Gaussian-product theorem), and ρ̂(G) =
Σ_κλ D_κλ ρ̂_κλ(G). For sto-3g this is closed-form per primitive.
G=0 is dropped (zero-mean gauge). No FFT, no real-space density
grid, just a sum over a finite shell of G vectors.

Compared to `EWALD3D`: replaces the FFT-Poisson grid with an analytic
Gaussian-Fourier sum. Should match `EWALD3D` to high accuracy at
sufficient G-cutoff.

### Wolf summation

Truncate the full-range real-space Coulomb at a cutoff R_c with a
smooth shifted-force kernel:

    K_W(r) = erfc(α r)/r - erfc(α R_c)/R_c   for r < R_c
           = 0                                for r ≥ R_c

For Gaussian AO products this is `J_SR(α)` minus a constant ×
overlap-like term. Cheap, no reciprocal space at all. Wolf reports
~10⁻³ to 10⁻⁴ accuracy on point-charge systems.

### ADFT (variational Coulomb fitting)

The Coulomb energy is written as a functional of an auxiliary
density ρ̃ = Σ_P c_P ξ_P(r):

    E_J[ρ̃] = ∫ρ̃ V_ext + ∫∫ρ̃(r)ρ̃(r')/|r-r'| dr dr' / 2 + ...

Minimizing the **fit error** ½ ∫∫(ρ-ρ̃)(r)(ρ-ρ̃)(r')/|r-r'| gives the
linear system

    A c = b     A_PQ = (P|Q),  b_P = (P|μν) D_μν

and the variational Coulomb energy

    E_J = c·b - ½ c·A·c   (= ½ ∫∫ρ̃ρ̃/|r-r'| at optimum)

The Fock contribution is J̃_μν = Σ_P c_P (P|μν).

For periodic at molecular-limit, A and b are computed with vibe-qc's
existing molecular DF tooling (`compute_2c_eri`, `compute_3c_eri`)
because the AO density has no significant overlap across cell
boundaries. For tight bulk this would need a periodic 3c-ERI binding.

## How to run

    .venv/bin/python examples/experimental/coulomb-method-zoo/bench_h2.py
    .venv/bin/python examples/experimental/coulomb-method-zoo/bench_lih.py
    .venv/bin/python examples/experimental/coulomb-method-zoo/run_all_methods.py

Outputs: `results/*.csv` and `results/*.json`.

## Throwaway scope reminder

Per `feedback_experimental_chat_policy`: every commit prefixed
`experiment:`, no CHANGELOG, no docs, no backwards-compat. One
demonstration test per method. If a method doesn't pan out, the
postmortem says so and we move on.
