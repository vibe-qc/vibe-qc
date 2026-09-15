# Periodic SCF convergence: damping, DIIS, and level shifts

A periodic Hartree-Fock (or KS-DFT) SCF iterates the Roothaan-Hall
equations until the density is self-consistent: at each step build a
Fock matrix from the current density, diagonalize it for new
orbitals, build a new density, repeat. For molecules and large unit
cells this just works in 10-20 iterations. For **tight-cell crystals**
(lattice $\sim$ 5 bohr) where bands sit close at the Fermi level, the
naive iteration can oscillate, the density swaps periodically
between two states without ever settling on the fixed point. vibe-qc
ships three knobs that stabilise the SCF in this regime:

1. **Density damping** (`damping`), linear mix of old and new
   density before the next Fock build. Default 0.5.
2. **Pulay DIIS** (`use_diis`), extrapolate to the SCF fixed point
   using a history of past densities. Default `True`.
3. **Level shift** (`level_shift`, Phase **C1a**, v0.4.0 new), 
   raise virtual MO eigenvalues by $b$ during iteration, which
   smoothly biases each step toward the lower-energy state without
   changing the SCF fixed point. The headliner of v0.4.0's tight-
   cell convergence pass.

This tutorial walks through the level-shift mechanism, when to use
it, and how it composes with the existing damping + DIIS machinery.

## The basic call

`level_shift` lives on `PeriodicSCFOptions` /
`PeriodicRHFOptions` / `PeriodicKSOptions` (default `0.0`). Set it
to a positive Hartree value:

```python
import vibeqc as vq
import numpy as np

# Tight 3D LiH cubic cell — a notoriously hard convergence case.
a = 4.5
sysp = vq.PeriodicSystem(
    3, np.eye(3) * a,
    [vq.Atom(3, [0.05,         0.05,         0.05]),
     vq.Atom(1, [0.5*a + 0.05, 0.5*a + 0.05, 0.5*a + 0.05])],
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

opts = vq.PeriodicSCFOptions()
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
opts.use_diis     = False         # disable to see level shift cleanly
opts.damping      = 0.1
opts.level_shift  = 0.3           # ← Hartree
opts.max_iter     = 30
opts.conv_tol_grad = 1e-5

result = vq.run_rhf_periodic_gamma_scf(
    sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
)
print(f"converged in {result.n_iter} iters at E = {result.energy:.6f} Ha")
```

The dispatcher (`run_rhf_periodic_scf` for multi-k,
`run_rhf_periodic_gamma_scf` for Γ-only) propagates `level_shift`
through to whichever Coulomb-method backend it routes to.

## Auto-reducing the shift

A constant shift helps the opening iterations but slows the last-mile
approach, so you rarely want to hold it forever. vibe-qc reduces it for
you. Two controls, both available on every molecular and periodic
options struct with identical names and semantics:

```python
opts.level_shift = 0.3
opts.level_shift_warmup_cycles = -1   # default: auto
```

* `-1` (default) shifts the first few startup cycles, then releases,
  always leaving at least one unshifted tail cycle so the reported
  energy is the true unshifted fixed point.
* `0` holds the shift at every iteration (the legacy constant behaviour
  used in the example above).
* `N > 0` shifts iterations `1..N`, then releases.

For full control, give an explicit per-iteration decay curve (CRYSTAL's
`LEVSHIFT B IRESET` idea). It supersedes `level_shift` + the warm-up:

```python
from vibeqc import LevelShiftSchedule

opts = LevelShiftSchedule.crystal_default().apply_to(vq.PeriodicSCFOptions())
# [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0] — decays to zero over seven iterations

opts.level_shift_schedule = [0.6, 0.4, 0.2, 0.0]   # or set it directly
```

The last entry is reused for any later iteration, so ending the list in
`0.0` releases the shift permanently.

**Every backend honours these.** The four molecular drivers, the C++
direct-truncated periodic drivers (`run_rhf_periodic_gamma`,
`run_rhf_periodic`, `run_rks_periodic`, where the shift is applied per
k-point as `F(k) + b·S(k) − (b/2)·S(k)·P(k)·S(k)`), and every Python
periodic backend (Γ-Ewald, multi-k Ewald, BIPOLE, GDF) resolve the
shift through one shared helper, `vibeqc.level_shift_at_iter`. When the
resolved shift for an iteration is `0.0` the `S·D·S` products are
skipped entirely, so a released shift costs nothing.

## Inertness at convergence

The Saunders-Hillier level shift transforms the Fock matrix during
iteration but **does not change the SCF fixed point**. Every value
of $b$ converges to the same total energy, the same MO eigenvalues
(reported as the *physical* unshifted values in the result), and
the same density. The only thing that differs is the iteration
*trajectory*:

![Level shift SCF trace: orbital-gradient norm versus iteration on log-y for tight 3D LiH cubic cell at three level_shift values 0.0 / 0.3 / 0.7 Ha. The b=0 trajectory is steepest and reaches conv_tol_grad in 9 iterations; b=0.3 in 11 iterations; b=0.7 in 13 iterations. All three converge to the same final total energy.](../_static/plots/level-shift-scf-trace.png)

Three Γ-Ewald SCFs on the same tight LiH cell, varying only
`level_shift`. The orbital-gradient norm $\|\nabla L\|$ drops
monotonically toward `conv_tol_grad` (grey dashed line) for all
three, but the *slope* changes: $b = 0$ (blue) is steepest and
fastest (9 iterations); $b = 0.3$ (green) is shallower (11);
$b = 0.7$ (red) shallower still (13). All three converge to the
same total energy, the energy spread across the three runs is
$7.6 \times 10^{-9}$ Ha, well under the $10^{-6}$ "indistinguishable"
threshold the test suite uses. Reproduce with
``python3 examples/plots/level-shift-scf-trace.py``.

The trade-off is **wall clock vs stability**. Larger $b$ takes more
iterations because each step is smaller; smaller $b$ takes fewer
iterations but each step is more aggressive, and the more
aggressive step can overshoot, bouncing between two near-degenerate
density states. For a near-metallic tight cell or a small-gap
semiconductor where the canonical iteration *would* oscillate, the
small extra wall-clock cost of $b = 0.3$ buys you guaranteed
monotone progress.

## When you need a level shift

The table below maps the symptom you read off the SCF trace to its likely cause and the level-shift (plus damping or DIIS) setting that fixes it:

| Symptom (in the SCF trace) | Most likely cause | Fix |
|---|---|---|
| Energy oscillates between two values without converging | Near-degenerate occupied/virtual at $E_F$ | `level_shift = 0.3` |
| `delta_e` cycles +,−,+,−,... but `\|grad\|` stays bounded | Density oscillation; DIIS extrapolation can't escape the cycle | `level_shift = 0.3` *and* increase `damping` to 0.7 |
| `\|grad\|` plateau at ~$10^{-3}$, energy stable | DIIS subspace is contaminated by an unphysical mixture | Restart with `level_shift = 0.5` and `diis_start_iter = 5` |
| `\|grad\|` *increases* from one iteration to the next | Catastrophic oscillation; the canonical iteration is diverging | `level_shift = 1.0` and lower `damping` to 0.3 |

The default `level_shift = 0.0` reproduces v0.1.0 behavior exactly,
so existing scripts get the same results. Flip it on opportunistically
when you see one of the symptoms above. Production CRYSTAL workflows
on tight cells routinely use $b = 0.5$ as a safety default.

## Composing with DIIS and damping

The three convergence aids work in different parts of the iteration:

- **Damping** averages *densities* before the Fock build:
  $D_{i} \leftarrow (1 - \alpha) D_{i-1} + \alpha D_i^{\text{new}}$.
- **DIIS** extrapolates *Fock matrices* using a history of past
  iterations to minimize the residual norm.
- **Level shift** modifies the *Fock matrix* before diagonalisation
  in each iteration, raising virtual eigenvalues so the
  diagonalisation doesn't easily mix occupieds and virtuals.

They compose: turning all three on at once is the standard hard-case
recipe. **Order of stabilisation strength**: start with damping
(cheap, no effect on converged density), add DIIS (essentially free,
on by default), and finally enable level shift only when the trace
shows real oscillation. With DIIS on, the level shift is most
effective during startup before DIIS has built a useful subspace, 
the figure above intentionally turns DIIS off so the level-shift
effect is unambiguous.

## Debugging convergence problems, the [`examples/debug/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/debug/) kit

When a periodic SCF doesn't converge with the aids on this page,
the [`examples/debug/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/debug)
directory ships nine focused substrate scripts, each answers
one diagnostic question. Read the
[README](https://github.com/vibe-qc/vibe-qc/blob/main/examples/debug/README.md)
for the decision tree; the most useful starting points:

| Symptom | Run this |
|---|---|
| "Is anything in the periodic-SCF stack working?" | `scf_easy_periodic.py`, Ne / Ar / diamond C must converge |
| "Why is my specific system diverging?" | `scf_iteration_recorder.py`, 4-panel forensics plot of \|ΔE\| / ‖[F,DS]‖ / E / DIIS |
| "Which combination of aids fixes it?" | `scf_convergence_sweep.py`, 16-config grid scan |
| "Does the new SAD guess (v0.6.1) help?" | `scf_sad_vs_hcore.py`, A/B test on LiH / NaCl / MgO |
| "Is this a vibe-qc bug or just a hard system?" | `scf_vs_pyscf.py`, head-to-head against `pyscf.pbc.dft.RKS` |
| "What does the wall-time vs precision look like?" | `scf_perf_scaling.py`, knob scan |

Every script wraps the SCF in [`vq.crash_dump_context`](../user_guide/output_files.md)
so failures leave a TOML dump + last-iter density on disk, 
inspect post-mortem via `vq.load_dump("output/<label>.dump")`
without re-running.

## Theory

The sections below give the mechanism behind the level shift: the Saunders-Hillier shifted-Fock transformation, why raising the virtual eigenvalues during iteration kills the near-degenerate occupation swap that drives oscillation, how it composes with Fermi-Dirac smearing for genuinely metallic cells, and what remains on the roadmap.

### The Saunders-Hillier transformation

At each SCF iteration, instead of diagonalizing $\mathbf{F}$ directly,
diagonalize the **shifted Fock matrix**

$$
\tilde{\mathbf{F}} = \mathbf{F}
+ b \, \mathbf{S}
- \tfrac{b}{2} \, \mathbf{S}\mathbf{D}\mathbf{S}
$$

where $\mathbf{S}$ is the AO overlap, $\mathbf{D}$ is the current
density matrix, and $b > 0$ (Hartree) is the level-shift parameter.
Saunders and Hillier (1973) showed this transformation:

1. **Raises virtual eigenvalues** by $b$ at the SCF fixed point
   (since $\mathbf{S}\mathbf{D}\mathbf{S} = \mathbf{S}$ on virtuals
   and $\mathbf{S}\mathbf{D}\mathbf{S} = 2\mathbf{S}$ on occupieds
   for a closed-shell idempotent density).
2. **Leaves the SCF fixed point unchanged** because at convergence
   the ground-state density $\mathbf{D}^*$ satisfies
   $\tilde{\mathbf{F}}(\mathbf{D}^*) \mathbf{C}^* = (\mathbf{F}^*
   + b\,\mathbf{P}_\text{virt})\mathbf{C}^*$ with $\mathbf{P}_\text{virt}$
   the projector onto virtuals, and the *occupied* eigenspace is
   unchanged.
3. **Increases the HOMO-LUMO gap** during iteration by exactly $b$,
   making the diagonalisation more numerically stable.

The shifted-eigenvalues are *not* what vibe-qc reports, `result.mo_energies`
reports the un-shifted physical eigenvalues, recovered with one
final Fock construction *without* the shift after convergence. The
shift is purely an iteration aid.

### Why this fixes oscillation

Periodic SCF oscillation typically arises from a **near-degenerate
occupied/virtual pair** at the Fermi level. Without level shift, the
diagonalisation at iteration $i$ may swap the occupations of the two
states, giving a wildly different density at iteration $i+1$, which
swaps them back, ad infinitum. The level shift increases the gap by
$b$ during iteration, so the two states are no longer near-degenerate
in the diagonalisation eigensystem and the swap doesn't happen. By
the time convergence is reached, the *physical* gap (without shift)
may still be small, but the iteration found the right ordering and
the small gap is no longer a numerical problem.

### Composing with C1b smearing

**C1b (Fermi-Dirac smearing + fractional occupations)** has shipped
in the multi-k Ewald RHF/RKS drivers and the native Γ-GDF RHF/RKS
driver. Low-level option objects still store
``opts.smearing_temperature = T_Ha`` (Hartree, default 0.0), but the
recommended user-facing resolver accepts units, presets, and a
conservative automatic guess:

```python
opts.smearing_temperature = vq.resolve_smearing_temperature(
    "auto",
    metallic=True,
).temperature

opts.smearing_temperature = vq.resolve_smearing_temperature(
    1000.0,
    unit="kelvin",
).temperature
```

When nonzero, occupations follow

$$
n_i \;=\; \frac{2}{1 + \exp((\varepsilon_i - \mu) / T)}
$$

with bisection on the chemical potential $\mu$ to satisfy the
per-cell electron count. The free energy ``A = E − T·S`` is the
quantity the SCF converges (variational under smearing); the
result object exposes ``free_energy``, ``entropy``, ``fermi_level``,
and per-k ``occupations`` arrays.

**Smearing vs level shift**: level shift fixes *near-degenerate*
occupied/virtual swap oscillation when there's still a clear
occupation pattern. Smearing fixes *genuinely metallic* systems
where the very notion of a clean occupied/virtual partition
disappears at the Fermi level. They compose: a tight metallic cell
typically benefits from both ``level_shift = 0.3`` and
``smearing_temperature ≈ 0.005`` Ha (~1500 K, well above any
"physical" electronic temperature but below the gap scale that
matters for total energies). C1b smearing currently lives in the
native Γ-GDF RHF/RKS path and periodic multi-k Ewald RHF/RKS paths, 
molecular and periodic UKS smearing land in follow-up sub-phases.

### Still on the roadmap

- **C1c, second-order / quadratically-convergent SCF** as a fallback
  when DIIS + smearing still oscillate. Used for the very hardest
  tight-cell cases (3D defect supercells, transition-metal oxides).
  Tutorial will follow when shipped.

## Resources

- 3D LiH cubic cell at $a = 4.5$ bohr, sto-3g, EWALD_3D Γ-only,
  no DIIS: ~50 s per SCF on one core (Apple M2). Three runs
  (level_shift = 0.0 / 0.3 / 0.7) total ~150 s for the embedded
  figure.
- Per-iteration cost is dominated by the Ewald Fock build; level
  shift adds one $\mathbf{S} \cdot \mathbf{D} \cdot \mathbf{S}$
  matrix product per iteration ($\mathcal{O}(N_\text{bf}^3)$ FLOPs)
  - typically <1% wall-time overhead.
- Memory peak is one Fock + one overlap + one density per k-point:
  $\sim 3 \cdot N_\text{bf}^2 \cdot N_k \cdot 16$ bytes. For LiH
  sto-3g on a 4×4×4 mesh: ~1 MB.

## Caveats

- **`level_shift` only on the periodic drivers in v0.4.0.** Molecular
  RHF / UHF / RKS / UKS get the level-shift hook in C1a-2 (a
  follow-up phase). For molecular cases use `damping` + `use_diis`,
  which already handle nearly all cases.
- **Don't over-shift.** $b > 1$ Ha for a system with a gap of $\sim$
  1 Ha effectively pushes virtuals far enough that the iteration
  becomes a pure damping loop with no real Fock-matrix dynamics, 
  numerically stable but extremely slow.
- **DIIS subspace contamination.** If you flip `level_shift` on
  mid-run after DIIS has built up a subspace from oscillating
  iterations, the contaminated subspace can fight the level shift.
  Usually faster to restart from `initial_guess = vq.InitialGuess.SAD`
  with the level shift on from iteration 1.
- **Reported MO energies are physical**, not shifted. Don't subtract
  $b$ from them by hand.

## References

- **Saunders, V. R.; Hillier, I. H.** "A 'Level-Shifting' method for
  converging closed shell Hartree-Fock wave functions," *Int. J.
  Quantum Chem.* **7**, 699 (1973). The original level-shift paper.
  vibe-qc's $\tilde{\mathbf{F}} = \mathbf{F} + b\mathbf{S} -
  (b/2)\mathbf{SDS}$ is the form Saunders and Hillier introduced.
- **Pulay, P.** "Convergence acceleration of iterative sequences.
  The case of SCF iteration," *Chem. Phys. Lett.* **73**, 393 (1980).
  The DIIS algorithm vibe-qc's `use_diis` implements.
- **Pulay, P.** "Improved SCF convergence acceleration,"
  *J. Comput. Chem.* **3**, 556 (1982). Refinements to the DIIS
  weight calculation (the "C2" residual norm) that vibe-qc's
  multi-k DIIS uses.
- **Helgaker, T.; Jørgensen, P.; Olsen, J.** *Molecular Electronic-
  Structure Theory*. Wiley (2000), Chapter 10. Standard textbook
  treatment of HF iteration, level shifts, and DIIS, the source
  many later QC textbooks adapt.
- **Pisani, C.; Dovesi, R.; Roetti, C.** *Hartree-Fock Ab Initio
  Treatment of Crystalline Systems*. Lecture Notes in Chemistry
  vol. 48, Springer (1988). The CRYSTAL convergence-acceleration
  apparatus vibe-qc's periodic SCF mirrors. Section 4 covers the
  level-shift / smearing / damping triad in periodic context.

## Next

- [User guide: SCF convergence](../user_guide/scf_convergence.md)
  - same machinery applied to molecular SCFs (level shift in
  molecular drivers ships in C1a-2).
- [Tight-cell DFT with periodic Becke partition](tight_cell_dft.md)
  - the *other* tight-cell pathology (DFT integration weights);
  pairs naturally with this tutorial when running periodic KS-DFT
  on small unit cells.
- [Ewald user guide](../user_guide/ewald.md), the long-range
  Coulomb story that goes hand-in-hand with the SCF-iteration story.
