# Smearing for metals: Fermi-Dirac, Methfessel-Paxton, and cold smearing

The molecular-limit SCF assumes a clean separation between the
occupied and virtual manifolds, every orbital below the HOMO is
filled, every orbital above the LUMO is empty. **Metals don't work
that way.** Bands cross the Fermi level; at finite k-mesh
resolution different k-points integer-occupy slightly different
band counts, and the Aufbau iteration oscillates between near-
degenerate occupation patterns. The classic fix, due to Mermin
1965 in the DFT context, is to give every orbital a *fractional*
Fermi-Dirac occupation

$$
f_i(\varepsilon_F, T) \;=\; \frac{1}{1 + \exp\bigl[(\varepsilon_i - \varepsilon_F)/k_B T\bigr]}
$$

at a small electronic temperature $T$ and integrate to the right
total electron count by adjusting $\varepsilon_F$ each iteration.
This smears the band-crossing into a soft step that the SCF can
descend without oscillation. The trade-off is an
**electronic-entropy contribution** to the free energy that the SCF
minimises in place of the bare total energy, physically harmless
at small $T$, and the relevant thermodynamic potential for any
metallic system anyway.

vibe-qc wires three smearing functions through the shared
`vibeqc.smearing` utility and into the closed-shell periodic drivers:
**Fermi-Dirac** (the thermal occupation itself), **Methfessel-Paxton** (a
Hermite-polynomial expansion whose entropy term nearly vanishes), and
**Marzari-Vanderbilt** cold smearing. This tutorial leads with Fermi-Dirac
on a representative case (a 1D hydrogen chain at half-filling, the simplest
metallic-band testbed in the bundled examples), covers how to enable it,
what the output looks like, and how to choose the temperature, then shows
how to switch to the other two, which share the same temperature handling.

## The recipe

The high-level `run_periodic_job(..., smearing_temperature=...)`
wrapper accepts readable forms:

```python
vq.run_periodic_job(..., smearing_temperature=0.005)       # Hartree
vq.run_periodic_job(..., smearing_temperature="1580 K")    # Kelvin
vq.run_periodic_job(..., smearing_temperature="0.136 eV")  # eV
vq.run_periodic_job(..., smearing_temperature="0.01 Ry")   # Rydberg
vq.run_periodic_job(..., smearing_temperature="metal")     # 0.005 Ha
vq.run_periodic_job(..., smearing_temperature="small-gap") # 0.002 Ha
vq.run_periodic_job(..., smearing_temperature="auto")      # conservative
```

Direct driver option objects store only the canonical electronic
$k_B T$ in Hartree:

```python
opts = vq.PeriodicKSOptions()
opts.smearing_temperature = vq.resolve_smearing_temperature("metal").temperature
```

The unit-aware parser is a high-level/user-facing convenience; the
SCF drivers consume Hartree.

The `"auto"` mode is deliberately **conservative**, without a
metallic hint it returns zero smearing rather than silently
changing the physics of a likely insulator. To opt into the
metallic guess, pass `metallic=True` or `n_electrons=<odd>` to
`vq.resolve_smearing_temperature` (or just use the `"metal"` preset
directly).

## Worked example: 1D hydrogen chain at half filling

The infinite 1D H chain at uniform bond length is the simplest
metallic-band testbed in vibe-qc. With one electron per H atom and
two H atoms per cell at *uniform* spacing, the band is exactly
half-filled, Mott would dimerise it via Peierls (see
[Peierls dimerisation of a 1D H-chain](peierls.md)), but at the RKS/PBE level with no
symmetry breaking, you get a 1D metal.

Working directory:

```
~/vibeqc-runs/h-chain-metallic/
    input-h-chain-smear.py
```

`input-h-chain-smear.py`:

```python
from pathlib import Path
import numpy as np
import vibeqc as vq

# Two H atoms per cell, uniformly spaced at 1.4 bohr (compressed
# regime — well inside the metallic phase, well away from the
# Peierls dimerisation that opens a gap at larger spacing).
a = 2.8  # bohr — cell length so atoms sit at 0 and a/2
sysp = vq.PeriodicSystem(
    dim=1,
    lattice=np.diag([a, 30.0, 30.0]),
    unit_cell=[vq.Atom(1, [0.0, 0.0, 0.0]),
               vq.Atom(1, [a / 2, 0.0, 0.0])],
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")

# Vanilla RKS/PBE without smearing — expect oscillation /
# convergence failure on a 16-point mesh.
res_no = vq.run_periodic_job(
    sysp, basis,
    method="RKS",
    functional="PBE",
    output="h-chain-nosmear",
    kpoints=[16, 1, 1],
    smearing_temperature=0.0,
    max_iter=80,
    write_molden_file=False,
)

# Same input, "metal" preset for k_B T.
res_sm = vq.run_periodic_job(
    sysp, basis,
    method="RKS",
    functional="PBE",
    output="h-chain-smear",
    kpoints=[16, 1, 1],
    smearing_temperature="metal",  # 0.005 Ha ≈ 0.136 eV
    max_iter=80,
    write_molden_file=False,
)

print(f"No smearing : converged={res_no.converged}, "
      f"iters={res_no.n_iter}, E={res_no.energy:.8f}")
print(f"Smearing    : converged={res_sm.converged}, "
      f"iters={res_sm.n_iter}, E={res_sm.energy:.8f}")
print(f"  electronic entropy S/k_B:        {res_sm.entropy:.6f}")
print(f"  free energy A = E - TS:          {res_sm.free_energy:.8f} Ha")
print(f"  Fermi level:                      {res_sm.fermi_level:.6f} Ha")
```

Expected behaviour:

- **Without smearing** the SCF oscillates between integer-occupation
  patterns at the band crossing, often never converging within 80
  iterations (or converging to a higher-energy stationary point
  that depends on the k-mesh).
- **With smearing** the SCF settles in 12-20 iterations to the
  free-energy minimum. The reported `entropy` is non-negative,
  the `fermi_level` sits inside the bath of bands at the crossing.

## What appears in the output

The `.out` log carries a smearing block right after the SCF trace:

```text
Finite-temperature occupations (Fermi-Dirac)
----------------------------------------------------
  source            "metal" preset
  k_B T (Hartree)   5.000e-03      (1577.8 K, 0.1361 eV)
  E_free            -1.143217  Ha
  T * S             -0.001094  Ha    ← electronic entropy
  E_extrapolated    -1.144311  Ha    ← 0 K limit (E_free + ½ T S)
  E_internal        -1.145404  Ha    ← E_free - T*S, reported in `result.energy`
  Fermi level (ε_F) -0.137820  Ha
```

The result object carries the same fields:

```python
res.energy           # E_internal — the usual SCF total
res.free_energy      # E_internal - T*S  (the *minimised* potential)
res.entropy          # S/k_B (always non-negative)
res.fermi_level      # ε_F at the converged occupations
res.occupations      # per-k-point Fermi-Dirac fractions, shape (nk, nbf)
```

For metallic systems the right number to report in a paper is
**E_extrapolated**, the $T \to 0$ extrapolation of the free
energy, which is the same total energy you would have gotten from
an infinitely fine k-mesh with no smearing. The other two energies
are intermediates the SCF needed.

## Choosing the temperature

The smearing temperature is a methodological knob, not a physical
temperature, the system isn't actually at 1580 K. Two rules of
thumb:

1. **Pick the smallest $k_B T$ that gives stable convergence.** Too
   small and the SCF oscillates the same way it did without
   smearing; too large and the bandwidth gets smeared into
   meaninglessness. For most metals, 0.005 Ha ($k_B T$ ≈ 0.136 eV ≈
   1580 K) is comfortably in the right regime, this is what
   `"metal"` resolves to.

2. **Verify the $T \to 0$ extrapolation is well-behaved.** Run the
   same calculation at 0.001, 0.002, 0.005, 0.01 Ha and check that
   `E_extrapolated` converges with $k_B T \to 0$ faster than
   `E_free` or `E_internal`. If the extrapolation is unstable, you
   need a denser k-mesh, not a different temperature.

The `"small-gap"` preset (0.002 Ha) suits systems with a small but
finite gap, small-bandgap semiconductors at large unit cells, or
defective insulators where the defect state sits just below the
conduction band. The `"debug"` preset (0.010 Ha) is intentionally
on the high side for code-correctness tests; don't use it for
production.

The `"auto"` mode is conservative on purpose: without a metallic
hint or recorded band gap it returns zero smearing. To opt into
the metallic guess automatically pass `metallic=True` or
`band_gap_hartree=0.0`:

```python
from vibeqc import resolve_smearing_temperature
guess = resolve_smearing_temperature("auto", metallic=True)
print(guess.temperature, guess.reason)
# 0.005, 'metallic=True'
```

## Beyond Fermi-Dirac: Methfessel-Paxton and cold smearing

Fermi-Dirac is the most physical choice (it is the actual thermal
occupation), but its electronic-entropy term shrinks only linearly in
$k_B T$, so the zero-temperature extrapolation wants a fairly small width
and a fine k-mesh. Two alternatives trade a little physical transparency
for a flatter, faster extrapolation, and both run through the same
drivers:

- **Methfessel-Paxton** (Methfessel and Paxton 1989) replaces the
  Fermi-Dirac step with a truncated Hermite-polynomial expansion of order
  $N$. The entropy contribution then vanishes to high order in $k_B T$, so
  the free energy is nearly width-independent and a coarser mesh with a
  larger smearing suffices. It is the standard metals choice in plane-wave
  codes, and it is what the AUTO recommender now picks for a metal.
- **Marzari-Vanderbilt** cold smearing (Marzari et al. 1999) shapes the
  broadened density to stay positive even though individual occupations
  can overshoot their formal $[0, 1]$ range. Its generalized entropy is
  conjugate to that occupation kernel, making the reported free energy a
  well-behaved minimization target. It takes no order parameter.

Select either by name; the temperature handling is identical to
Fermi-Dirac:

```python
res = vq.run_periodic_job(
    sysp, basis,
    method="RKS", functional="PBE",
    kpoints=[16, 1, 1],
    smearing_method="methfessel-paxton",   # or "marzari-vanderbilt"
    smearing_temperature="metal",
    output="h-chain-mp",
)
```

The accepted strings are `"fermi-dirac"`, `"methfessel-paxton"`, and
`"marzari-vanderbilt"`. To raise the Methfessel-Paxton expansion order
(1 by default; 2 is the common alternative), pass an explicit options
object instead of the bare method name:

```python
res = vq.run_periodic_job(
    sysp, basis,
    method="RKS", functional="PBE", kpoints=[16, 1, 1],
    smearing=vq.SmearingOptions(
        temperature=vq.resolve_smearing_temperature("metal").temperature,
        flavor="methfessel-paxton",
        mp_order=2,
    ),
)
```

```{note}
The AUTO k-point recommender defaults a **metal** to Methfessel-Paxton:
`vq.KPoints.recommend(system, is_metal=True)` returns a spec whose
`.smearing.flavor` is `"methfessel-paxton"`. Pass
`smearing_method="marzari-vanderbilt"` to `recommend(...)` to override it,
then feed the spec into `run_periodic_job(..., kpoints=spec)` to carry the
choice through.
```

All three methods report the same `result.entropy`, `result.free_energy`,
and `result.fermi_level` fields and the same `.out` smearing block (only
the method name in the header changes). For a routine metal,
Methfessel-Paxton order 1 or cold smearing at $k_B T$ near 0.005 Ha is a
good default; keep Fermi-Dirac when you specifically want the
finite-temperature physics (a genuine electronic-temperature study)
rather than just a convergence aid.

## Compatibility matrix

Smearing is wired into the closed-shell periodic drivers today; the table
below records which entry points support it and which are still on the
roadmap:

| Driver | Smearing supported? |
|---|---|
| `run_rhf_periodic_gamma_gdf(..., functional=...)` | ✓ Γ-only RHF/RKS |
| `run_rhf_periodic_multi_k_ewald3d` | ✓ multi-k RHF |
| `run_rks_periodic_multi_k_ewald3d` | ✓ multi-k RKS |
| `run_uhf_periodic_*` / `run_uks_periodic_*` | ✓ open-shell (Γ GDF, multi-k GDF, multi-k Ewald, and BIPOLE) |
| Molecular `run_rhf` / `run_rks` / `run_uhf` / `run_uks` | not yet (periodic-only today) |

Molecular smearing is rarely needed (open-shell molecules use UHF
with explicit α/β occupations; small-gap molecular systems use the
[stiff-convergence recipe](ediis_diis_hybrid.md)). Multi-k UHF
/ UKS smearing arrives once the multi-k open-shell drivers
themselves land, see [`multi_k_scf`](../user_guide/multi_k_scf.md).

## Resources

The 1D H chain example above with a 16-point k-mesh converges in
~5 s on a modern laptop. Wider chains (32-64 k-points) or 2D
graphene at a [12, 12, 1] mesh push the wall-time into the 30 s -
2 min range.

## References

- **Mermin's variational principle**, N. D. Mermin, "Thermal
  Properties of the Inhomogeneous Electron Gas," *Phys. Rev.*
  **137**, A1441 (1965). The foundational statement that
  finite-temperature DFT minimises the Mermin free energy
  $F = E - T S$, not the bare total energy.
- **Fermi-Dirac smearing in plane-wave DFT**, M. Methfessel,
  A. T. Paxton, "High-precision sampling for Brillouin-zone
  integration in metals," *Phys. Rev. B* **40**, 3616 (1989).
  Compares Fermi-Dirac with Methfessel-Paxton smearing; the
  half-T-S extrapolation rule comes from this paper.
- **Cold smearing**, N. Marzari, D. Vanderbilt, A. De Vita, M. C. Payne,
  "Thermal Contraction and Disordering of the Al(110) Surface,"
  *Phys. Rev. Lett.* **82**, 3296 (1999). The bounded-occupation
  ("cold") smearing function.
- **Smearing in Gaussian-basis periodic codes**, R. Dovesi et al.,
  "CRYSTAL14: A program for the *ab initio* investigation of
  crystalline solids," *Int. J. Quantum Chem.* **114**, 1287
  (2014). The CRYSTAL `SMEAR` directive, the lineage vibe-qc's
  periodic smearing follows.

## Next

- [Peierls dimerisation of a 1D H-chain](peierls.md), the
  *other* fix for the 1D H chain at half filling: open a gap by
  breaking the unit-cell translation symmetry.
- [Periodic SCF convergence: damping, DIIS, and level shifts](periodic_scf_convergence.md)
  - damping, DIIS, level shifts. Smearing composes freely with all
  three.
- [Band structure and density of states](band_structure.md), visualise
  the band crossing the smearing is taming.
- [User guide, SCF convergence](../user_guide/scf_convergence.md)
  § Smearing, full API reference, unit conversions, the
  `SmearingResolution` dataclass.
