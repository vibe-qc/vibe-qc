---
myst:
  html_meta:
    "description": "Smearing for metals in vibe-qc: the theory of fractional occupations (Fermi-Dirac, Methfessel-Paxton, Marzari-Vanderbilt), the Mermin free energy, the electronic entropy and the zero-temperature extrapolation, when each function is appropriate, and a worked Be metal example through run_periodic_job with smearing_temperature and smearing_method."
    "og:title": "vibe-qc tutorial - smearing for metals: Fermi-Dirac, Methfessel-Paxton and cold smearing"
---

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
metallic system anyway. Smearing on the periodic routes listed in the
compatibility matrix below is production; the Theory section derives
the three occupation functions and says when to use which.

## Theory

**Why integer occupations fail.** Write the band energy of a periodic
mean-field calculation as a Brillouin-zone sum,
$E_{\text{band}} = \sum_{\mathbf k} w_{\mathbf k}\sum_i n_i(\mathbf k)\,\varepsilon_i(\mathbf k)$.
For an insulator the occupations $n_i(\mathbf k)$ are 2 or 0 for every
$\mathbf k$, the integrand is smooth, and the Monkhorst-Pack sum converges
exponentially. In a metal the occupation is a step at the Fermi surface
$\varepsilon_i(\mathbf k) = \varepsilon_F$, the integrand is discontinuous,
the sum converges as a power of the mesh spacing, and on any finite mesh
the map from density to occupations is itself discontinuous: a tiny change
of the potential can move a band across $\varepsilon_F$ at one k-point and
flip two electrons between bands. No density mixer converges a
discontinuous fixed-point map, which is the "charge sloshing" every metal
shows without smearing.

**Mermin's finite-temperature functional.** Mermin
({vibeqc-cite}`mermin_finite_temperature_dft_1965`) showed that at a finite
electronic temperature $T$ the variational quantity of density-functional
theory is not the energy but the Helmholtz free energy

$$
A[n] = E[n] - T\,S ,
\qquad
S = -k_B \sum_{\mathbf k} w_{\mathbf k}\sum_i \Bigl[\,f_i \ln f_i + (1 - f_i)\ln(1 - f_i)\Bigr],
\qquad f_i = n_i/2 ,
$$

with $S$ the entropy of independent fermions, and that minimising $A$ over
the occupations at fixed electron count $N_e = \sum_{\mathbf k} w_{\mathbf k}\sum_i n_i(\mathbf k)$
gives exactly the Fermi-Dirac distribution above, with $\varepsilon_F$ the
Lagrange multiplier of the constraint. Every smearing scheme is this
variational principle with a chosen occupation function: because $A$ is
stationary with respect to the occupations, the SCF sees a smooth,
minimisable objective, the Hellmann-Feynman forces are derivatives of
$A$ (not of $E$), and the code reports three energies, the internal energy
$E$ (`result.energy`), the free energy $A = E - TS$ (`result.free_energy`,
the quantity actually minimised) and the entropy $S/k_B$ (`result.entropy`),
together with $\varepsilon_F$ (`result.fermi_level`). The default
`"fermi-dirac"` flavour and the `"mermin"` alias produce identical
occupations; the alias exists to make explicit that the reported energy is
a free energy.

**Three occupation functions.** Writing $x = (\varepsilon_i - \varepsilon_F)/\sigma$
with the width $\sigma = k_BT$, vibe-qc implements, per spin channel,

$$
f^{\mathrm{FD}}(x) = \frac{1}{1 + e^{x}},
\qquad
f^{\mathrm{MP}}_N(x) = \tfrac12\operatorname{erfc}(x) + \sum_{n=1}^{N} A_n\, H_{2n-1}(x)\, e^{-x^2},
\quad A_n = \frac{(-1)^n}{n!\,4^n\sqrt\pi},
\qquad
f^{\mathrm{MV}}(x) = \tfrac12\operatorname{erfc}\bigl(x + \tfrac{1}{\sqrt2}\bigr) + \frac{e^{-(x + 1/\sqrt2)^2}}{\sqrt{2\pi}} ,
$$

where $H_m$ are Hermite polynomials and $N$ (`mp_order`, 1 or 2) the
Methfessel-Paxton order. Each function has a conjugate generalised entropy
defined by $s'(x) = x\,f'(x)$, so that $A = E - \sigma\sum s$ is
stationary at fixed particle number for every flavour; for Fermi-Dirac
$s$ is the fermion entropy above, for Methfessel-Paxton
$s_N(x) = \tfrac12 A_N H_{2N}(x)e^{-x^2}$, and for Marzari-Vanderbilt
$s(x) = \tfrac{1}{2\sqrt\pi}\,(1 + \sqrt2\,x)\,e^{-(x + 1/\sqrt2)^2}$
(the per-spin forms of the published expressions, {vibeqc-cite}`methfessel_paxton_1989`
and {vibeqc-cite}`marzari_vanderbilt_1999`). The generalised entropies of
the last two are not positive definite, and the Methfessel-Paxton
occupations can be slightly negative or exceed one in the tails; the
Marzari-Vanderbilt function was constructed so that the *broadened density
of states* stays positive even though individual occupations may still
overshoot, which is what "cold" refers to.

**The zero-temperature extrapolation.** The smearing width is a
numerical device, so what one wants is the $\sigma \to 0$ limit. For a
free-electron-like DOS both $E$ and $A$ deviate from the true $E(0)$ by
terms of order $\sigma^2$ with opposite signs and equal magnitude, so their
average cancels the leading error:

$$
E(0) \approx \tfrac12\bigl(E + A\bigr) = A + \tfrac12\,TS = E - \tfrac12\,TS ,
$$

the number to quote for a Fermi-Dirac metal calculation (it is not printed in
the `.out` block; form it from `result.energy` and `result.entropy`). Methfessel-Paxton's expansion is designed
so that the leading error of $A$ vanishes to order $\sigma^{2N+2}$ for a DOS
that is a polynomial of degree $2N$ near $\varepsilon_F$, which is why its
free energy is nearly width-independent and needs no extrapolation at
moderate $\sigma$; Marzari-Vanderbilt achieves the same at first order
while keeping the density positive. Both therefore tolerate a larger
$\sigma$, and hence a coarser mesh, than Fermi-Dirac for the same
total-energy error.

**When each is appropriate.** Use **Fermi-Dirac** when the electronic
temperature is physical (warm dense matter, temperature-dependent
properties, a genuine finite-$T$ free energy for phonons or dynamics),
when a small-gap system needs a monotone, strictly bounded occupation, or
whenever the extrapolation formula above is going to be applied; it is
the most robust SCF driver but converges slowest in $\sigma$. Use
**Methfessel-Paxton** (order 1, occasionally 2) as the workhorse for total
energies and forces of ordinary metals on a converged mesh; it is what
`KPoints.recommend(..., is_metal=True)` selects, and its only hazards are
the unphysical negative occupations at very large $\sigma$ and a DOS that
should not be read literally. Use **Marzari-Vanderbilt** cold smearing when
you want Methfessel-Paxton's fast convergence but a positive broadened
density, for instance for surfaces and defects where the local DOS is
inspected, or when the first-order MP occupations misbehave. For a
semiconductor or insulator none of them is needed and all of them give
integer occupations and $S = 0$ once $\sigma$ is below the gap; the
[Gilat-Raubenheimer net](gilat_net_metals.md) is the parameter-free
alternative for the post-SCF Fermi level and DOS of a converged metal.

vibe-qc wires three smearing functions through the shared
`vibeqc.smearing` utility and into the periodic drivers:
**Fermi-Dirac** (the thermal occupation itself), **Methfessel-Paxton** (a
Hermite-polynomial expansion whose entropy term nearly vanishes), and
**Marzari-Vanderbilt** cold smearing. This tutorial leads with Fermi-Dirac
on a representative case (face-centred cubic beryllium, a one-atom
free-electron-like metal), covers how to enable it, what the output looks
like, and how to choose the temperature, then shows how to switch to the
other two, which share the same temperature handling.

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

## Worked example: fcc beryllium, a one-atom metal

Beryllium in the fcc structure at a compressed lattice constant of 3.2 Å
is the smallest metallic test case in vibe-qc: one atom per primitive cell,
four electrons, and $2s$/$2p$ bands that cross the Fermi level. The same cell
is the metal of the [Gilat-Raubenheimer tutorial](gilat_net_metals.md). The
script below runs restricted Kohn-Sham PBE through `run_periodic_job` on the
GDF route at a 2x2x2 mesh with the `"metal"` smearing preset.

```{note}
Earlier versions of this page used a uniformly spaced 1-D hydrogen chain
sampled at 16 k-points. In the current build `run_periodic_job` refuses a
`dim=1` cell *with a k-mesh* on the GDF route (a `NotImplementedError` about
the 1-D Coulomb gauge; Gamma-only 1-D cells still run), and a metallic chain
needs the mesh, so the chain is no longer reachable here; the
[Peierls tutorial](peierls.md) still discusses its physics.
```

Working directory:

```
~/vibeqc-runs/be-fcc-metallic/
    input-be-fcc-smear.py
```

`input-be-fcc-smear.py`:

```python
import numpy as np
import vibeqc as vq

# fcc Be, a = 3.2 A (compressed, metallic density); one atom per
# primitive cell, four electrons.
a = 3.2 / 0.529177210903                       # bohr
lattice = (a / 2) * np.array([[0.0, 1.0, 1.0],
                              [1.0, 0.0, 1.0],
                              [1.0, 1.0, 0.0]])
sysp = vq.PeriodicSystem(3, lattice, [vq.Atom(4, [0.0, 0.0, 0.0])])
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

res = vq.run_periodic_job(
    sysp, basis,
    method="RKS",
    functional="PBE",
    jk_method="gdf",
    output="be-fcc-smear",
    kpoints=[2, 2, 2],
    smearing_temperature="metal",  # 0.005 Ha, about 0.136 eV
    max_iter=60,
    write_molden_file=False,
)

print(f"converged={res.converged}, iters={res.n_iter}, E={res.energy:.8f}")
print(f"  electronic entropy S/k_B:  {res.entropy:.6f}")
print(f"  free energy A = E - TS:    {res.free_energy:.8f} Ha")
print(f"  Fermi level:               {res.fermi_level:.6f} Ha")
```

Output from a vibe-qc 0.17.10 development build (the run took about 20 CPU
minutes, most of it the GDF build for the mesh, so plan for minutes rather than
seconds; a smaller auxiliary basis or looser cutoffs give a quicker test):

```text
converged=True, iters=4, E=-14.54890214
  electronic entropy S/k_B:  0.000044
  free energy A = E - TS:    -14.54890236 Ha
  Fermi level:               0.297100 Ha
```

What this run shows, and what it does not:

- The SCF converges in four iterations with the smeared occupations. The
  free energy lies below the energy by $TS = 2.2\times10^{-7}$ Ha, which is
  exactly $k_B T \cdot S/k_B$ with $k_B T = 0.005$ Ha and
  $S/k_B = 4.4\times10^{-5}$.
- The entropy is tiny. On this coarse 2x2x2 mesh no sampled level lies within a
  few $k_B T$ of the Fermi level, so the occupations are close to integer and
  smearing changes almost nothing here. The mesh was chosen for cost; the
  Fermi surface is resolved only on denser meshes, which is where smearing
  matters and where the [Gilat-Raubenheimer tutorial](gilat_net_metals.md)
  works (a 4x4x4 mesh on the same cell).
- Without smearing, a partially filled band forces the SCF to choose integer
  occupations at every k-point, and on a mesh that does sample the Fermi
  surface it can oscillate between occupation patterns or land on a stationary
  point that depends on the mesh and the starting guess. That is the general
  motivation for smearing; this particular coarse-mesh run does not exhibit
  it, and the no-smearing comparison was not run here.

## What appears in the output

The `.out` log carries a smearing block right after the energy breakdown:

```text
  Finite-temperature (smearing)
  ---------------------------------------------
       kBT_smearing (Ha) =         0.0050000000
              T_elec (K) =             1578.875
            entropy S/kB =         0.0000440528
        free_energy (Ha) =       -14.5489023565
        fermi_level (Ha) =         0.2970999961
```

The preceding energy breakdown reports the internal energy $E$ as the total
energy, and the job header records `smearing_method = fermi-dirac`,
`smearing_source = preset:metal` and the temperature in Hartree and Kelvin.
The result object carries the same fields:

```python
res.energy           # E, the internal energy, the usual SCF total
res.free_energy      # A = E - T*S  (the *minimised* potential)
res.entropy          # S/k_B (always non-negative)
res.fermi_level      # the Fermi level at the converged occupations
```

The $T \to 0$ extrapolation $E(0) \approx E - \tfrac12 TS$ from the theory
section is not printed; form it from `res.energy` and the entropy,

```python
kT = 0.005                                     # Ha, the "metal" preset
e_zero = res.energy - 0.5 * kT * res.entropy   # E(0) = E - TS/2
```

and quote it for a Fermi-Dirac metal calculation, since it is the estimate
that approaches the result of an infinitely fine mesh without smearing. The
other two energies are intermediates the SCF needed.

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
   $E - \tfrac12 TS$ converges with $k_B T \to 0$ faster than
   the free energy `A` or the internal energy `E`. If the extrapolation is unstable, you
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
    method="RKS", functional="PBE", jk_method="bipole",
    kpoints=[2, 2, 2],
    smearing_method="methfessel-paxton",   # or "marzari-vanderbilt"
    smearing_temperature="metal",
    output="be-fcc-mp",
)
```

The accepted strings are `"fermi-dirac"`, `"methfessel-paxton"`, and
`"marzari-vanderbilt"`. To raise the Methfessel-Paxton expansion order
(1 by default; 2 is the common alternative), pass an explicit options
object instead of the bare method name:

```python
res = vq.run_periodic_job(
    sysp, basis,
    method="RKS", functional="PBE", jk_method="bipole", kpoints=[2, 2, 2],
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

Through `run_periodic_job`, `smearing_temperature > 0` is accepted on the
combinations below and refused with a `NotImplementedError` naming the
missing route everywhere else (the rule lives in the runner's smearing
dispatch check):

| `method` | `jk_method` | Smearing through `run_periodic_job` |
|---|---|---|
| RHF, RKS | `gdf`, `gpw`, `gapw` | accepted (Γ and multi-k) |
| RKS, UHF, UKS | `bipole` | accepted (Γ and multi-k); **BIPOLE RHF refuses** (integer occupations) |
| UHF, UKS | `gdf` | accepted (Γ and multi-k) |
| RHF, RKS, UHF, UKS | `rijcosx` | accepted on the multi-k COSX route only; Γ RIJCOSX refuses |
| ROHF, ROKS | any | refused (integer 2/1/0 occupations) |

The low-level drivers carry the same option directly:
`run_rhf_periodic_gamma_gdf`, `run_krhf_periodic_gdf`, `run_krks_periodic_gdf`,
`run_kuhf_periodic_gdf`, `run_kuks_periodic_gdf`, the multi-k Ewald
`run_rhf_periodic_multi_k_ewald3d` / `run_rks_periodic_multi_k_ewald3d` and
their UHF/UKS siblings (which support per-spin smearing but have no
`run_periodic_job` dispatch branch), and the BIPOLE RKS/UHF/UKS drivers.
Molecular `run_rhf` / `run_rks` / `run_uhf` / `run_uks` do not smear:
open-shell molecules use UHF with explicit α/β occupations, and small-gap
molecular systems use the [stiff-convergence recipe](ediis_diis_hybrid.md).

## Resources

The fcc Be example above is a one-atom, five-function cell, but the cost is
dominated by the BIPOLE lattice sums rather than by the diagonalisation, and
it grows with the number of k-points (a 4x4x4 mesh, which a converged Be
energy needs, has eight times as many as 2x2x2). Time a single point on your
own machine before committing to a mesh scan.

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
  *other* fix for a half-filled band: open a gap by breaking the
  unit-cell translation symmetry.
- [Periodic SCF convergence: damping, DIIS, and level shifts](periodic_scf_convergence.md)
  - damping, DIIS, level shifts. Smearing composes freely with all
  three.
- [Band structure and density of states](band_structure.md), visualise
  the band crossing the smearing is taming.
- [User guide, SCF convergence](../user_guide/scf_convergence.md)
  § Smearing, full API reference, unit conversions, the
  `SmearingResolution` dataclass.
