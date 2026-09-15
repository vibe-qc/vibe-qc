# Simple-systems CRYSTAL14 baseline (STO-3G RHF)

Six small periodic crystals at the **smallest possible basis** (STO-3G)
under **plain RHF, no FMIXING / no LEVSHIFT** — the goal is to pick
systems that converge cleanly in CRYSTAL14 under default SCF settings.
The converging subset becomes the **v0.8.0 vibe-qc parity baseline**:
known-good reference numbers we can chase without simultaneously
debugging SCF acceleration, k-point sampling, level-shift recipes,
or basis-set linear dependence.

The principle is the user's 2026-05-13 directive:

> "We will chase bugs forever if we at the same time try to debug
> SCF convergers, K-point sampling, levelshift and other bugs."

POB-TZVP is deliberately avoided here for two reasons:

1. **POB-TZVP is optimised for DFT / hybrid DFT, not for pure HF**
   or correlated methods. The Peintinger 2013 / Vilela-Oliveira
   2019 parameter sweeps targeted DFT and hybrid functionals;
   the basis is not a recommended HF basis for solids.
2. MgO RHF/POB-TZVP does not converge in CRYSTAL14 even with
   persistent `LEVSHIFT 50 0` and 200 cycles — both as a
   consequence of (1) and because at the time we'd be confounding
   gauge bugs with SCF-acceleration bugs (Anderson / Broyden / SOSCF
   were on the v0.8.x roadmap then; all three have since shipped —
   worth re-checking convergence with them available).

When POB-TZVP enters parity testing it will be for
**hybrid-DFT** (B3LYP, PBE0, PW1PW) — the regime it was
designed for. See `docs/user_guide/basis_sets.md` § "Method
scope" for the canonical reasoning.

## Inputs

| File | System | SG | Lattice (Å) | Notes |
|---|---|---|---|---|
| `lih-rhf-sto3g.d12` | LiH rocksalt | 225 | 4.084 | Smallest ionic; 4 electrons / FU |
| `mgo-rhf-sto3g.d12` | MgO rocksalt | 225 | 4.21 | The user's known-good test recipe |
| `nacl-rhf-sto3g.d12` | NaCl rocksalt | 225 | 5.640 | Heavier ions, larger cell |
| `lif-rhf-sto3g.d12` | LiF rocksalt | 225 | 4.020 | Wider gap; small-LiH-like |
| `c-diamond-rhf-sto3g.d12` | C diamond | 227 | 3.567 | Wide-gap covalent |
| `si-diamond-rhf-sto3g.d12` | Si diamond | 227 | 5.430 | Smaller-gap covalent |

All inputs use:

* `SHRINK 8 8` — 8×8×8 Monkhorst-Pack + Gilat secondary mesh.
* `TOLDEE 8` — 10⁻⁸ Hartree SCF tolerance.
* No FMIXING, no LEVSHIFT, no BROYDEN — defaults only.

## Submission (compute-reference via vq)

Submit each input from a clean local checkout:

```bash
# Once per checkout
git push origin <branch>
ssh compute-reference 'cd ~/gitlab/vibeqc-dev && git pull --ff-only'

# Submit one input — wrapper handles PATH for CRYSTAL14 binaries
cd examples/regression/crystal_parity
vq submit -d . --cpus 14 --wall-time-seconds 7200 -- \
    bash run-reference.sh baseline_sto3g/<input>.d12
```

Or use `submit_all.sh` to fan out all 6 in one go (they execute
serially under the daemon's `--max-jobs 1` cap).

## Acceptance criteria for "v0.8.0 baseline candidate"

A system goes into the baseline iff CRYSTAL14 reports
`SCF ENDED - CONVERGENCE ON ENERGY` (the canonical converged
state) under **defaults-only** within 30 cycles. Anything that needs
LEVSHIFT or FMIXING isn't a clean test problem for the v0.8.0
gauge-fix work — it confounds gauge bugs with SCF-acceleration bugs.

## What this baseline gives us

Once curated, the converging subset becomes:

1. A reference parity table — committed `(system, E_total, E_nuclear,
   n_iter)` numbers that vibe-qc must match on the same recipe.
2. A regression test fixture for `tests/test_periodic_crystal_parity.py`
   that pulls the parity numbers and asserts against future vibe-qc
   runs. (Not yet delivered — that test file does not exist in the
   tree as of this update; the plan below was never completed.)
3. The user-facing "this works today" set for the v0.8.0 release
   notes — explicitly distinguished from harder cases that need
   SCF acceleration.
