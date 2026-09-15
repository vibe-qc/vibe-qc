---
orphan: true
---

# Design, periodic smearing as a shared backend-agnostic utility

> **Status:** Shipped on `main`. Core open-shell smearing landed
> 2026-06-15; GDF multi-k UHF/UKS smearing landed; `run_periodic_job`
> exposure complete. The `smearing_temperature` kwarg deprecation
> targeted at v0.11.0 was deferred. Decisions log in §10.
>
> **Original target:** v0.10.x SCF-convergence program
> ([`docs/roadmap.md`](roadmap.md) § "SCF guess +
> convergence program (multi-release)").
>
> **Chat:** `smear`. Drop-box:
> `.release-status/v0.10.x/smear.md`
> (CLAUDE.md §3).

This document defines the contract by which fractional-occupation
smearing becomes a **single shared utility** consumed uniformly by
every periodic SCF backend (Ewald-3D, GDF, BIPOLE, GAPW) at every
spin × k-mesh combination vibe-qc ships. Today smearing is partially
duplicated across drivers, Fermi-Dirac only, and missing entirely
for UHF / UKS and for the C++ direct-truncated Γ paths. This design
unifies the surface and admits Methfessel-Paxton and Marzari-
Vanderbilt without touching backend SCF loops again.

The numerical kernels themselves are textbook, Mermin (1965) for
the finite-T free-energy formalism `A = E − TS`; Methfessel-Paxton
(1989) for the Hermite-polynomial smoothed step; Marzari, Vanderbilt,
De Vita, Payne (1999) for cold smearing; dos Santos-Marzari (2023)
for the modern T → 0 extrapolation story. None of that is up for
design discussion here. What this doc designs is the **vibe-qc-side
plumbing**: the dataclass shape, the function signature, the
deprecation path, the C++ contract, and how GAPW composes with it.

## 1. Scope, what lands at M0/M1, what's deferred

| | M0/M1 (this push) | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Design doc + drop-box baseline | ✅ | | | | | | | | | |
| `python/vibeqc/smearing/` package (renamed from `occupations.py`) | ✅ | | | | | | | | | |
| `SmearingOptions` dataclass | ✅ | | | | | | | | | |
| `smearing_temperature` kwarg → deprecation alias | ✅ | | | | | | | | | |
| Drivers folded onto shared utility (no behavior change) | ✅ | | | | | | | | | |
| C++ `real_space_density_from_kpoints_fractional` generalised (open-shell shape) | | ✅ | | | | | | | | |
| Fermi-Dirac on Ewald-3D direct-C++ Γ (RHF + RKS) | | ✅ | | | | | | | | |
| Fermi-Dirac on Ewald-3D UHF + UKS (Γ + multi-k) | | | ✅ | | | | | | | |
| Fermi-Dirac on GDF Γ-only (closed-shell) | | | | ✅ | | | | | | |
| Fermi-Dirac on BIPOLE RKS/UHF/UKS (Γ + multi-k; RHF rejects) | | | | | ✅ | | | | | |
| Methfessel-Paxton N=1, N=2 on every cell | | | | | | ✅ | | | | |
| Marzari-Vanderbilt cold smearing on every cell | | | | | | | ✅ | | | |
| GAPW smearing | | | | | | | | ✅ | | |
| `vq.extrapolate_t_zero(...)` UX | | | | | | | | | ✅ | |
| Production hardening; legacy kwarg removal | | | | | | | | | | ✅ |

> **Status, open-shell per-spin smearing core + GDF multi-k (2026-06-15).**
> The M3 open-shell machinery has landed *core-first*, on the GDF backend
> (where the multi-k open-shell driver lives):
> * **Per-spin core**, `smearing.apply_smearing_open_shell` runs the shared
>   μ-bisection once per spin channel with `spin_degeneracy = 1` and an
>   independent particle-count constraint, giving separate μ_α, μ_β (the
>   CP2K/VASP/QE convention, §3 "Spin handling"). `fermi_dirac_occupations_per_k`
>   / `aufbau_occupations_per_k` are parametrized by degeneracy (default 2.0
>   ⇒ closed-shell bit-identical). Empty/full channels (n_β=0, or n_α=full)
>   short-circuit the bisection. Unit tests: `tests/test_smearing_open_shell.py`.
> * **GDF multi-k UHF/UKS**, `run_kuhf/kuks_periodic_gdf` honor
>   `smearing_temperature > 0`: per-spin global μ_σ across the BZ, Mermin free
>   energy `A = E - T(S_α+S_β)`, result carries `fermi_level_alpha/beta`,
>   `entropy`, `free_energy`, `occupations_alpha/beta`. T=0 stays the exact
>   per-k Aufbau (UHF(M=1)≡KRHF machine-precision anchors green). Tests:
>   `tests/test_periodic_kuhf_gdf.py` (M=1≡KRHF+smear, triplet conservation,
>   boron 2p¹ fractional degenerate shell).
>
> * **`run_periodic_job` exposure**, `_validate_smearing_dispatch` now
>   permits open-shell `smearing_temperature > 0` for `jk_method='gdf'` with
>   a `kpoints` mesh (routed to `run_kuhf/kuks_periodic_gdf`); `kpoints=(1,1,1)`
>   gives a Γ-point open-shell smeared run through the k-resolved driver.
>
> **Landed since (2026-06-17, GDF chat):**
> * **Smearing-consistent `⟨S²⟩`, DONE** (`514876fa`). `_multi_k_s_squared`
>   takes the fractional per-spin occupations (`occ_alpha_k`/`occ_beta_k`) and
>   computes the ensemble-UHF ⟨S²⟩ = Sz(Sz+1) + n_β − Σ_k w_k Σ_ij n^α_i n^β_j
>   |⟨α|S|β⟩|²; integer filling (occ omitted) falls back **bit-identical**.
> * **Γ-only open-shell GDF (`run_pbc_gdf_uhf/uks`), DONE.** Per-spin global
>   μ_α/μ_β over the degenerate Γ levels via `apply_smearing_open_shell`
>   (weights=[1]); the result carries the same smearing fields as the multi-k
>   driver; T=0 stays bit-identical (UHF(M=1)≡RHF, all Γ gates green).
>   `run_periodic_job` open-shell GDF smearing now accepts `kpoints=None` (Γ)
>   as well as a k-mesh. Tests: boron 2p¹ fractional degenerate shell in
>   `tests/test_pbc_gdf_compcell.py::test_pbc_gdf_u{hf,ks}_gamma_smearing_fractional`.
>
> **Still pending for M3 / open-shell smearing completeness:**
> * Ewald-3D multi-k UHF/UKS (the table row above), `periodic_uhf/uks_multi_k_ewald.py`
>   still call `reject_unsupported_smearing_temperature`. (PBC-chat-owned.)

**Hard non-goals of this chat (owned elsewhere):**

* Backend SCF correctness. Each backend chat owns its own SCF loop
  - if a backend oscillates or converges to a different stationary
  point under smearing than without, that's the backend's bug
  (CLAUDE.md §7) and gets filed back to the owning chat. Smearing
  is not a damping hack.
* Density mixing (Anderson / Broyden / Kerker / Pulay-Kerker /
  periodic Pulay). The scf-mix chat owns that. Smearing and mixing
  compose; `SmearingOptions` and `MixingOptions` are designed as
  orthogonal surfaces and the metallic-system test set is shared.
* Molecular UHF / UKS smearing. Useful but not periodic; out of
  scope for this chat unless the maintainer rescopes.
* Charged-cell Madelung corrections under smearing. Coordinate with
  whichever chat owns that.
* Other-program runtime imports. Parity validation is via the
  subprocess oracle pattern (`examples/regression/runner_*.py`,
  CLAUDE.md §10). No `import pyscf` / `import cp2k`.

## 2. The five pre-flight questions

### Q1, Where does the shared utility live and how does it relate to `occupations.py`?

**Answer:** New package
`python/vibeqc/smearing/` becomes the single source of truth. The
existing [`python/vibeqc/occupations.py`](../python/vibeqc/occupations.py)
contents move into per-flavor modules; `occupations.py` is reduced
to a thin back-compat re-export shim with a `DeprecationWarning` on
import.

```
python/vibeqc/smearing/
    __init__.py        # Public API surface
    options.py         # SmearingOptions dataclass + unit helpers
    _mu_bisection.py   # Shared chemical-potential bisection
    fermi_dirac.py     # Fermi-Dirac occupations + entropy
    mermin.py          # Mermin finite-T free-energy occupations
    methfessel_paxton.py   # MP N=1, N=2 (M6)
    marzari_vanderbilt.py  # MV cold smearing (M7)
    auto.py            # guess_smearing_temperature + presets
    extrapolation.py   # vq.extrapolate_t_zero (M9)
```

**Public symbols re-exported from `vibeqc.__init__`** (every existing
public symbol stays public; new symbols added):

```python
# unchanged - keep working through occupations.py shim AND through
# the new package
from vibeqc.smearing import (
    SmearingResolution, resolve_smearing_temperature,
    guess_smearing_temperature,
    kelvin_to_hartree_temperature, hartree_to_kelvin_temperature,
    electronvolt_to_hartree_temperature, rydberg_to_hartree_temperature,
    fermi_dirac_occupations_per_k,
    aufbau_occupations_per_k,
    KB_HARTREE_PER_K, EV_PER_HARTREE, HARTREE_PER_RYDBERG,
)

# new at M0/M1
from vibeqc.smearing import SmearingOptions, apply_smearing
```

**Why move rather than wrap.** `occupations.py` already shipped
public API in v0.4.0, users have `vq.resolve_smearing_temperature`
in their scripts. The move-with-shim pattern keeps user code
working bit-for-bit while giving the new code a clean home that can
grow to MP / MV without making `occupations.py` a 2000-line god
module. Shim removal is **not** scheduled, it stays indefinitely
as a back-compat surface.

### Q2, What is the `SmearingOptions` shape?

```python
@dataclass(frozen=True)
class SmearingOptions:
    """Canonical user-facing smearing configuration.

    Stored on every periodic SCF options dataclass (PeriodicRHFOptions,
    PeriodicSCFOptions, PeriodicKSOptions, BipoleOptions, GdfOptions, …).
    Consumed by vibeqc.smearing.apply_smearing(eigenvalues, ..., smearing=...).

    All numeric widths are k_B T in Hartree (the canonical unit on the
    SCF options surface, matching the v0.4.0 contract). Use the
    resolve_smearing_temperature(...) helper to accept user-facing
    inputs in K / eV / Ry or named presets.
    """

    # Electronic temperature k_B T in Hartree. 0.0 disables smearing.
    temperature: float = 0.0

    # "fermi-dirac" | "mermin" | "methfessel-paxton" | "marzari-vanderbilt"
    flavor: str = "fermi-dirac"

    # For Methfessel-Paxton only. 1 or 2 - higher orders are not
    # numerically useful in practice (MP-1 and MP-2 are what every
    # production code ships).
    mp_order: int = 1

    # Provenance for user-facing logs. Mirrors SmearingResolution.source
    # so SCF logs can say "preset:metal" or "explicit:kelvin" rather
    # than just a number.
    source: str = "explicit"
    reason: str = ""
```

**Hard rules baked in:**
* `temperature` is in **Hartree**. Always. Conversion happens at the
  user-facing boundary (`SmearingOptions.from_user(...)` factory or
  `resolve_smearing_temperature(...)`). No driver ever sees Kelvin.
* `temperature == 0.0` is the sentinel for "disabled". The driver
  fast-path is `if smearing.temperature > 0.0: ...`, same as today.
* `mp_order` is silently ignored unless `flavor == "methfessel-paxton"`.
  No bookkeeping noise.
* The dataclass is frozen so the SCF driver can hold a reference
  through the entire SCF loop without worrying about mutation.

**Factory for user input.**
```python
@classmethod
def from_user(cls, value, *, unit="hartree", flavor="fermi-dirac",
              mp_order=1, **resolve_kwargs) -> SmearingOptions:
    """Resolve user-facing temperature input to canonical SmearingOptions."""
    res = resolve_smearing_temperature(value, unit=unit, **resolve_kwargs)
    return cls(
        temperature=res.temperature,
        flavor=flavor,
        mp_order=int(mp_order),
        source=res.source,
        reason=res.reason,
    )
```

### Q3, What is the `apply_smearing` contract?

```python
def apply_smearing(
    eigenvalues_per_k: Sequence[np.ndarray],
    *,
    weights: Sequence[float],
    n_electrons_per_cell: float,
    smearing: SmearingOptions,
    spin: str = "closed-shell",  # "closed-shell" | "alpha" | "beta"
) -> SmearingResult:
    """Single entry point every backend driver calls.

    Returns occupations per k-point, chemical potential μ, electronic
    entropy S/k_B per cell, and the flavor's free-energy contribution
    (so the driver can assemble A = E − TS with the correct flavor-
    dependent entropy term).
    """
```

```python
@dataclass(frozen=True)
class SmearingResult:
    occupations_per_k: List[np.ndarray]   # per-band occupations in [0, n_max]
    mu: float                             # chemical potential, Hartree
    entropy: float                        # S/k_B per unit cell, dimensionless
    free_energy_correction: float         # = -temperature * entropy, Hartree
    smearing: SmearingOptions             # echoed back for log surfaces
```

**Spin handling.**
* `spin="closed-shell"`: occupations are in `[0, 2]`; one
  `apply_smearing(...)` call solves for one μ that conserves
  `sum_k w_k sum_i n_i(k) = n_electrons`. This is the v0.4.0
  contract.
* `spin="alpha"` / `spin="beta"`: occupations in `[0, 1]`; the
  driver calls `apply_smearing(...)` twice, once per spin channel,
  with **separate** electron counts. Each channel solves for its
  own μ_σ. This is the standard UHF/UKS pattern.

**Why per-spin separate μ rather than one shared μ.** With separate
μ_σ, the per-spin occupations relax to the right populations even
when the SCF lands an antiferromagnetic / ferromagnetic stationary
point with substantially different Fermi levels in the two channels
- required for Fe / Ni / Cr / Mn metallics. CP2K, VASP, Quantum
Espresso all use the per-spin μ convention for spin-polarized
metallic SCF; matching them keeps the parity contract honest.

**μ-bisection is shared.** A single
`_mu_bisection.py.find_mu(particle_count_fn, target, lo, hi)`
implements the bisection. Each flavor's `apply_smearing` builds the
right `particle_count_fn` from its occupation formula and hands it
to the shared bisector. This is how MP and MV slot in without
re-implementing μ-search logic.

**Entropy formulas (the per-flavor difference).**
| Flavor | Occupation `n(x)` (x = (ε−μ)/T) | Entropy contribution `s(x)` |
|---|---|---|
| Fermi-Dirac | `1/(1+exp(x))` | `−[f ln f + (1−f) ln(1−f)]`, f = n / (closed-shell 2 or open-shell 1) |
| MP-1 | step regularised by erf + Hermite | per Methfessel & Paxton (1989) eq. 9 |
| MP-2 | as MP-1 with second-order Hermite | per same paper, higher-order term |
| MV cold | `[2 − √2·x − √(2π)·erf(x − 1/√2)] / 2` | per Marzari et al. (1999) appendix |

Each flavor's module owns its own `_occupation(x, T)` and `_entropy(x)`
functions; `apply_smearing` is the same dispatch+bisection wrapper
for all of them.

### Q4, How does each backend embed `SmearingOptions`, and what is the deprecation path for the existing `smearing_temperature` kwarg?

**Constraint discovered during M0 audit.** The periodic options
classes (`PeriodicRHFOptions`, `PeriodicSCFOptions`,
`PeriodicKSOptions`) are **pybind11-bound C++ structs**, not Python
dataclasses
([`cpp/src/bindings.cpp:3815`](../cpp/src/bindings.cpp),
[`cpp/include/vibeqc/periodic_rhf.hpp:97`](../cpp/include/vibeqc/periodic_rhf.hpp)).
Adding a Python `__post_init__` adapter is not possible without
re-wrapping every options class in a Python subclass, which would
break user code that constructs `vq.PeriodicRHFOptions()` directly.

**Decision (§9 #8 revised).** Stage the embedding across two
releases to keep M1 truly behavior-neutral and to avoid C++
struct churn:

* **v0.10.x (M0/M1):** `SmearingOptions` is a **Python-side dataclass
  only**. The C++ options structs keep their existing
  `smearing_temperature: float` field unchanged, that field stays
  the single source of truth on the options surface through v0.10.x.
  Users who want the new flavor / mp_order / extrapolation surface
  go through the user-facing high-level wrappers (`run_periodic_job`),
  which accept `smearing=SmearingOptions(...)` and set
  `opts.smearing_temperature = smearing.temperature` internally.
  Driver-internal SCF loops convert
  `opts.smearing_temperature → SmearingOptions(temperature=...,
  flavor="fermi-dirac")` at the top of the SCF loop and feed it
  to `apply_smearing(...)`.
* **v0.11.0:** Add the `SmearingOptions` field to the C++ options
  structs (binding it as a nested pybind11 class), remove the
  legacy `smearing_temperature: float` field. Hard breaking change,
  flagged in v0.10.x deprecation warnings.

This staging keeps M1 a Python-only refactor, no `.so` rebuild
needed for any consuming chat to pick up the new surface.

**Driver-internal pattern (M1, every driver):**
```python
# At the top of the SCF loop, normalise to the new shape:
smearing = SmearingOptions.from_legacy_kwarg(opts.smearing_temperature)
# ... SCF loop calls apply_smearing(eps_per_k, ..., smearing=smearing)
```

**User-facing pattern (v0.10.x, optional new surface):**
```python
# Old (keeps working through v0.10.x; will be removed in v0.11.0):
vq.run_periodic_job(sys, basis, smearing_temperature=0.005, ...)

# New (recommended at v0.10.x; required at v0.11.0):
vq.run_periodic_job(sys, basis,
                    smearing=SmearingOptions(temperature=0.005,
                                             flavor="fermi-dirac"),
                    ...)
```

The high-level wrapper `run_periodic_job` does the resolution:
```python
def run_periodic_job(sys, basis, *,
                     smearing=None,
                     smearing_temperature=None,  # legacy
                     ...):
    if smearing is not None and smearing_temperature is not None:
        raise ValueError(
            "Pass either smearing= (new) or smearing_temperature= (legacy), "
            "not both."
        )
    if smearing_temperature is not None:
        warnings.warn(
            "smearing_temperature= is deprecated; pass "
            "smearing=SmearingOptions(temperature=...) instead. "
            "Removed in v0.11.0.",
            DeprecationWarning, stacklevel=2,
        )
        smearing = SmearingOptions.from_user(smearing_temperature, ...)
    elif smearing is None:
        smearing = SmearingOptions()  # temperature=0.0, disabled
    ...
    opts.smearing_temperature = smearing.temperature  # bridge to C++
```

**Per-backend wiring (M0/M1 commit touches):**

| Driver | Internal SmearingOptions setup | Reads `opts.smearing_temperature` (unchanged) |
|---|---|---|
| `periodic_rhf_multi_k_ewald.py:567` | `smearing = SmearingOptions.from_legacy_kwarg(...)` at top of SCF | yes |
| `periodic_rks_multi_k_ewald.py:411` | same | yes |
| `periodic_k_gdf.py:496` | same | yes |
| `periodic_uhf_multi_k_ewald.py:281` | currently raises on T > 0; **M3** lifts | n/a until M3 |
| `periodic_uks_multi_k_ewald.py:281` | same, **M3** lifts | n/a until M3 |
| BIPOLE options structs (**M5**) | when wired | n/a until M5 |
| GAPW options struct (**M8**) | when wired | n/a until M8 |

User-visible: zero behavior change at M1. Legacy kwarg keeps
working through v0.10.x as today.

### Q5, What is the C++-side contract, and how does GAPW compose with it?

**Current C++ surface.** `cpp/src/periodic_scf.cpp:104`, 
`real_space_density_from_kpoints_fractional` consumes per-k
fractional occupations (closed-shell shape: per band → `n ∈ [0, 2]`)
and performs the inverse-Bloch fold to produce the real-space
density. The kernel is **flavor-agnostic** by construction, it sees
occupations as numbers, not as a Fermi distribution. Good.

**Generalisation needed at M2 (UHF/UKS unlock).** The same kernel
must accept a per-spin shape:

```cpp
// Current (closed-shell): occ_per_k[k][i] ∈ [0, 2]
LatticeMatrixSet real_space_density_from_kpoints_fractional(
    const std::vector<Eigen::MatrixXd>& C_per_k,        // MO coeffs per k
    const std::vector<Eigen::VectorXd>& occ_per_k,      // per-MO occupation
    const Crystal& crystal,
    const std::vector<Eigen::Vector3d>& kpoints_cartesian,
    const std::vector<double>& weights);

// Target (any spin): caller passes per-spin per-k occupations and
// gets back per-spin real-space density. Closed-shell collapses to
// the alpha-only call with weights doubled - same kernel, no spin
// awareness in C++.
```

**The C++ kernel stays flavor-agnostic and spin-agnostic.** The
Python `apply_smearing` produces occupations; the C++ folds them.
Smearing flavor lives entirely in Python. This is the right boundary
because (a) the bisection + entropy formulas are small Python code
that doesn't benefit from C++, (b) C++ stays out of policy decisions
about MP order or MV vs FD, (c) testability is much higher in Python.

**GAPW composability (coordination input for the GAPW chat).** GAPW's
plane-wave grid composes with smeared occupations the same way GDF's
density does: the driver assembles the density matrix `D_μν = Σ_k w_k
Σ_i n_i(k) C_iμ(k) C_iν(k)*` and hands it to the J/K builder, which
projects to the FFT grid. **Nothing in the smearing surface needs to
know about FFT grids.** The GAPW chat must:
* embed `SmearingOptions` in its options struct from M0,
* call `vibeqc.smearing.apply_smearing(...)` in its SCF loop with
  the eigenvalues from its Bloch diagonalisation,
* feed the resulting `occupations_per_k` to the same density-matrix
  assembly that the existing GDF / Ewald-3D paths use.

The smearing surface is **identical** for GAPW. If the GAPW chat
finds it isn't, that's a bug in this design, flag it during M0
sign-off, not later.

**Cross-check at M0:** The current C++ kernel is closed-shell-only;
when M2 generalises it for open shells, the M0 commit lands the
generalised signature even though only closed-shell callers exist
in M0/M1. (Closed-shell collapses to a one-channel call with α-only
occupations.) This avoids a churn cycle when M3 ships UHF/UKS.
*Decision pending §9, could also defer the signature change to M2
to keep M0/M1 truly behavior-neutral. Default: defer to M2.*

## 3. T → 0 extrapolation UX (M9, but designed-for in M0)

The user-facing recommendation matrix for metallic systems (per
Marzari et al. 1999, refined dos Santos-Marzari 2023):

| Flavor | Error vs T → 0 | Recommended T window | Extrapolation |
|---|---|---|---|
| Fermi-Dirac | linear in T | 100-600 K | linear in T, ladder of 3-5 T values |
| MP-1 | linear in T² | 0.005-0.02 Ha | quadratic in T |
| MP-2 | as MP-1 with smaller prefactor | as MP-1 | quadratic in T |
| Marzari-Vanderbilt cold | cubic in T (no quadratic term) | 0.005-0.02 Ha | cubic in T, ladder of 3-4 T values |

The UX (M9) exposes:
```python
results_T = [vq.run_periodic_job(sys, basis, smearing_temperature=T,
                                  flavor="marzari-vanderbilt") for T in T_grid]
energy_T0, fit_diag = vq.extrapolate_t_zero(results_T, flavor="marzari-vanderbilt")
```

The fit-diagnostic surface (`fit_diag.r_squared`, `fit_diag.coeffs`,
`fit_diag.residuals_per_T`) is mandatory, the user must be able to
see whether the ladder of T values is in the asymptotic regime
before they trust the extrapolated energy.

This is M9; only mentioned here to confirm the `SmearingOptions`
surface admits it (it does, every flavor's free-energy contribution
is in `SmearingResult.free_energy_correction`, which is what the
extrapolator fits over a ladder of T).

## 4. Test strategy

**Unit-level (`tests/test_smearing_*.py`):**
* Each flavor's `_occupation(x, T)` matches its closed-form
  formula on a grid.
* Each flavor's `_entropy(x)` matches its closed-form formula.
* `_mu_bisection.find_mu` solves the particle-count constraint to
  `< 1e-12` electrons on every flavor.
* `apply_smearing` round-trips: T → 0 collapses to integer Aufbau,
  electron-count conservation, entropy ≥ 0.

**Per-backend integration (`tests/test_periodic_<backend>_smearing.py`):**
* T = 0 reproduces the pre-smearing SCF dynamics bit-for-bit for gapped
  spectra. Native closed-shell multi-k GDF instead corrects the historical
  fixed-band-count behavior when bands overlap between k points: one weighted
  particle constraint selects the globally lowest states.
* Wide-gap insulator at low T: energy / free-energy / occupations
  match the T = 0 result to µHa (smearing inert when the gap
  exceeds k_B T by ≫ 1).
* Electron-count conservation under any T.
* `⟨S²⟩ = 0.75` on an isolated H atom (UKS, M3+).
* Closed-shell H₂ at any k-mesh, any T: matches integer-Aufbau RHF
  / RKS bit-for-bit at T → 0.

**CP2K subprocess oracle parity (`examples/regression/`):**

| System | Flavor | Bar |
|---|---|---|
| Na bcc (3³ k-mesh) | Fermi-Dirac, T = 300 K | sub-mHa/atom; μ to ~10⁻⁴ Ha vs CP2K |
| Al fcc (4³) | Marzari-Vanderbilt, σ = 0.01 Ha | sub-mHa/atom; **cubic** T-dependence of `A − E₀` verified (vs linear for FD) |
| Fe bcc | Fermi-Dirac, UKS | per-spin μ_σ matches CP2K; ⟨S²⟩ honored under smearing |
| Cu fcc | MP-1, σ = 0.01 Ha | matches CP2K to sub-mHa/atom |
| Al fcc (T → 0 extrapolation) | MV cold ladder | extrapolated E₀ matches CP2K T → 0 to ~µHa |

CP2K is invoked via a subprocess runner per CLAUDE.md §10, no
`import cp2k` anywhere in `python/vibeqc/` or `cpp/`. The runner
pattern lives at `examples/regression/runner_cp2k.py` (to land in
M2-M6 as the CP2K parity oracle).

## 5. User-guide story

Single canonical page: `docs/user_guide/smearing.md` (lands at M0/M1,
covers Fermi-Dirac only; gains MP / MV sections as M6 / M7 ship).
Cross-linked from:
* `docs/user_guide/scf_convergence.md`, when to enable smearing
  vs other convergence aids
* `docs/user_guide/periodic_basics.md`, "metallic systems" section
* `docs/user_guide/<backend>.md` per backend, per-backend
  availability row
* `docs/features.md`, per-backend smearing column

The page must say, prominently: **smearing is for partial occupations
near a metallic Fermi surface, not for damping a divergent SCF.**
If your SCF oscillates with smearing off but converges with smearing
on, you're masking a backend bug (CLAUDE.md §7). The page links the
v0.7.0 Madelung self-image leak (over-bound H₂/STO-3G by ~0.587 Ha
in a 30-bohr box) as the canonical cautionary tale.

## 6. CHANGELOG entry (M0/M1 commit)

Provisional `[Unreleased]` entry:

```
### Added
- `vibeqc.smearing` - new package consolidating Fermi-Dirac
  occupation utilities (μ-bisection, electronic entropy,
  unit-flexible smearing-temperature parsing) into a single
  backend-agnostic surface. `SmearingOptions` dataclass lands
  on every periodic options struct; designed to admit
  Methfessel-Paxton + Marzari-Vanderbilt without further
  refactors.

### Changed
- Periodic SCF drivers (multi-k Ewald RHF/RKS, multi-k GDF KRHF)
  now consume `vibeqc.smearing.apply_smearing` rather than
  driver-local `_occupations_from_eps` / `_occupations_per_k`
  closures. No behavior change - bit-for-bit parity with the
  v0.4.0 / v0.7.x results.

### Deprecated
- `PeriodicRHFOptions.smearing_temperature` (and siblings on
  every periodic options struct) - use
  `PeriodicRHFOptions.smearing = SmearingOptions(temperature=...)`
  instead. Legacy kwarg keeps working through v0.10.x with a
  `DeprecationWarning`; removal targeted at v0.11.0.

### Fixed
- `docs/roadmap.md` smearing coverage rows reconciled -
  line 3683 claimed C1b shipped on every periodic backend
  (false) and line 3693 said multi-k Ewald only (true at
  v0.4.0, but multi-k GDF KRHF shipped smearing later). Both
  rows now reflect the audit-confirmed truth.
```

## 7. Roadmap reconciliation (M0/M1 commit)

Both [`docs/roadmap.md:3683`](roadmap.md) and
[`docs/roadmap.md:3693`](roadmap.md) rewritten to reflect the
audited matrix in §1 of the drop-box. The smearing row in the
"SCF guess + convergence program" matrix gains an explicit
"covered today vs. planned at v0.10.x" split rather than a single
"partial" cell.

## 8. Coordination, sibling-chat sign-off requests

| Chat | What we're asking | Why now |
|---|---|---|
| **release** | Sign off on `SmearingOptions` contract + v0.11.0 deprecation timeline for `smearing_temperature` | Establishes the contract before downstream user docs cite it |
| **GDF** ([Archived `handovers/HANDOVER_GDF_V0_11_2026_05_29.md`](https://vibe-qc.com/docs/)) | Review M0 for compat with existing `periodic_k_gdf.py` smearing wiring; confirm UHF/UKS GDF schedule | M4 sequencing depends on it |
| **BIPOLE** ([Archived `handovers/HANDOVER_BIPOLE_PRODUCTION.md`](https://vibe-qc.com/docs/) / [`bipole_status.md`](bipole_status.md)) | Confirm BIPOLE RKS/UHF/UKS smearing coverage on Γ + multi-k; BIPOLE RHF remains rejected | M5 sequencing depends on keeping the method gates explicit |
| **GAPW** ([`docs/design_periodic_gapw.md`](design_periodic_gapw.md)) | Review M0, `SmearingOptions` is the surface their M4 (metallic GAPW) consumes; pre-commit alignment avoids a re-design later | Their design doc is live; coordinate before either freezes |
| **scf-mix** | Confirm `SmearingOptions` ↔ `MixingOptions` composition; coordinate metallic-system test set | Both chats target the same metallic test cells |

## 9. Decisions log

| # | Decision | When | Status |
|---|---|---|---|
| 1 | Move `occupations.py` contents into `vibeqc/smearing/` (rename + thin shim) rather than wrapping or staying flat | 2026-05-25 | per maintainer (combined M0/M1 direction) |
| 2 | Combine M0 design doc + M1 no-behavior-change refactor in one push | 2026-05-25 | per maintainer |
| 3 | Audit per-backend coverage as part of M0; rewrite roadmap lines 3683 + 3693 | 2026-05-25 | per maintainer |
| 4 | `SmearingOptions.temperature` in Hartree (matching v0.4.0 contract), user conversion at boundary | 2026-05-25 | author |
| 5 | Per-spin separate μ_σ for UHF/UKS (matches CP2K / VASP / QE convention) | 2026-05-25 | author, sign-off pending sibling chats |
| 6 | C++ kernel stays flavor-agnostic; smearing flavor entirely in Python | 2026-05-25 | author, pending sign-off |
| 7 | Deprecation cutoff for legacy `smearing_temperature` kwarg = v0.11.0 | 2026-05-25 | author, **release chat sign-off pending** |
| 8 | `SmearingOptions` is Python-side-only at v0.10.x, C++ struct embedding deferred to v0.11.0 because periodic options are pybind11-bound C++ structs (constraint discovered during M0 audit; see Q4) | 2026-05-25 | author, **release chat sign-off pending** |
| 8a | C++ open-shell `real_space_density_from_kpoints_fractional` signature generalisation: defer to M2 (keep M0/M1 truly behavior-neutral) | 2026-05-25 | author, pending |
| 9 | CP2K is the parity oracle for periodic metallics (subprocess runner per CLAUDE.md §10) | 2026-05-25 | author |

## 10. References

* Mermin, N. D. *Thermal properties of the inhomogeneous electron
  gas.* Phys. Rev. **137**, A1441 (1965).
* Methfessel, M. & Paxton, A. T. *High-precision sampling for
  Brillouin-zone integration in metals.* Phys. Rev. B **40**,
  3616 (1989).
* Marzari, N., Vanderbilt, D., De Vita, A. & Payne, M. C.
  *Thermal contraction and disordering of the Al(110) surface.*
  Phys. Rev. Lett. **82**, 3296 (1999).
* dos Santos, F. P. & Marzari, N. *Fermi-surface effects and the
  electronic entropy of crystals at finite temperature.* Phys. Rev.
  B **107**, 195122 (2023).

## See also

* `.release-status/v0.10.x/smear.md`
  - chat drop-box (baseline + plan + pending asks)
* [`docs/user_guide/smearing.md`](user_guide/smearing.md), user
  guide page (lands with M0/M1)
* [`docs/roadmap.md`](roadmap.md) § "SCF guess + convergence
  program (multi-release)", smearing row
* [`docs/release_process.md`](release_process.md), branch model,
  documentation cadence
* CLAUDE.md §3 (drop-box), §7 (smearing-is-not-a-damping-hack),
  §10 (no external QC program imports), §14 (small-landable)
