---
myst:
  html_meta:
    "description": "Hands-on vibe-qc tutorial for quasiparticle ionisation potentials with the electron propagator: run_job(method='ovgf') on water at RHF/6-31G*, compare Koopmans, second-order (GF2), renormalised third-order and Delta-SCF ionisation energies with experiment, read result.ovgf and the pole strengths, and derive the Dyson equation, the second- and third-order self-energy and the renormalisation."
    "og:title": "vibe-qc tutorial - ionisation potentials from the electron propagator"
---

# Ionisation potentials from the electron propagator

Koopmans' theorem says the ionisation potential (IP) of a closed-shell
molecule is minus the orbital energy, $I_p \approx -\varepsilon_p$. It is
the first thing everyone reads off an SCF output and it is systematically
wrong by one to two electronvolts: it ignores the relaxation of the
remaining electrons when one is removed, which lowers the IP, and the
loss of correlation energy in the cation, which raises it. The two errors
partly cancel, which is why Koopmans works at all. A $\Delta$SCF
calculation of the cation recovers the relaxation but not the
correlation; a correlated calculation of both species recovers both, one
state at a time. The electron propagator, or one-particle Green's
function, does better: it corrects *all* the orbital energies of one
reference calculation at once, through a frequency-dependent
**self-energy** $\Sigma(\omega)$, and gives every ionisation a **pole
strength** that says whether the one-electron picture still holds.

vibe-qc exposes this as `run_job(method="ovgf")`. The route is production
for HF references (closed shell here, open shell through the spin-orbital
kernel; see [Methods: theory and status](../methods.md#how-to-read-the-status-labels)),
and it is important to know exactly what it computes: `result.ovgf` holds
the quasiparticle energies of the **diagonal second-order self-energy**
(GF2), solved self-consistently in the frequency, for the outer valence
(up to the three highest occupied orbitals plus the LUMO); the `.out`
additionally prints a **renormalised** HOMO ionisation potential that
includes the full diagonal third-order self-energy through a geometric
resummation, in the spirit of the outer-valence Green's function (OVGF)
of Cederbaum and von Niessen. It is not a literal implementation of the
OVGF A/B/C recipes; the module docstring of `vibeqc.propagator` says so,
and this page follows it.

## Setup

Water at RHF/6-31G*, followed by a $\Delta$SCF cation for comparison.

```python
import vibeqc as vq

ANG = 1.8897259886
EV = 27.211386245988
atoms = [
    vq.Atom(8, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.0,  0.7572 * ANG, 0.5865 * ANG]),
    vq.Atom(1, [0.0, -0.7572 * ANG, 0.5865 * ANG]),
]
h2o = vq.Molecule(atoms, 0, 1)

# RHF, then the diagonal second-order self-energy for the outer valence.
res = vq.run_job(h2o, basis="6-31g*", method="ovgf", output="h2o_ovgf")
print(res)                                    # RHFResult+OVGF(4 quasiparticle orbitals)
print(f"E(RHF) = {res.energy:.8f} Ha")

n_occ = h2o.n_electrons() // 2
print(f"{'MO':>3} {'label':>5} {'Koopmans/eV':>12} {'GF2/eV':>8} {'Z':>6} {'Dyson iters':>12}")
for q in res.ovgf:                            # QuasiparticleResult per orbital
    label = "HOMO" if q.orbital == n_occ - 1 else "LUMO" if q.orbital == n_occ else ""
    print(f"{q.orbital:3d} {label:>5} {-q.eps_scf * EV:12.3f} {-q.eps_qp * EV:8.3f} "
          f"{q.pole_strength:6.3f} {q.iterations:12d}")

# Delta-SCF: vertical IP from the UHF cation at the neutral geometry.
cation = vq.run_job(vq.Molecule(atoms, 1, 2), basis="6-31g*", method="uhf",
                    output="h2o_cation")
print(f"Delta-SCF(UHF) IP = {(cation.energy - res.energy) * EV:.3f} eV")
```

On a vibe-qc 0.17.10 development build (Apple M5 Max) the OVGF job takes
under four seconds and the cation under three:

```
RHFResult+OVGF(4 quasiparticle orbitals)
E(RHF) = -76.00910804 Ha
 MO label  Koopmans/eV   GF2/eV      Z  Dyson iters
  2             19.221   17.977  0.930            3
  3             15.536   13.448  0.914            3
  4  HOMO       13.539   11.004  0.907            4
  5  LUMO       -5.839   -5.292  0.980            3
Delta-SCF(UHF) IP = 10.847 eV
```

The `.out` carries the same table under the header
`OVGF / GF2 quasiparticle energies (diagonal second-order self-energy)`,
followed by three summary lines:

```
  OVGF ionization potential (HOMO) = 11.0040 eV
  OVGF electron affinity   (LUMO) = -5.2917 eV   (small-basis EA -- qualitative only)
  GF2(renorm) ionization potential (HOMO) = 11.6628 eV   (renormalized; pole Z=0.925)
```

## What just happened

`run_job` converged the RHF reference, transformed the two-electron
integrals to the MO basis (the full $n_\text{bf}^4$ tensor, 18 functions
here), and for each of the four outer-valence orbitals solved the diagonal
Dyson equation $\omega = \varepsilon_p + \Sigma_{pp}(\omega)$ by Newton
iteration, three or four steps each. `result.ovgf` is a list of
`QuasiparticleResult` records with `orbital` (0-based MO index),
`eps_scf` (the Koopmans value), `eps_qp` (the quasiparticle energy),
`sigma` (the self-energy at the solution), `pole_strength`, `iterations`
and `converged`, all energies in Hartree; `q.correction` is
`eps_qp - eps_scf`. IPs are $-\varepsilon^\text{qp}$ for occupied
orbitals and the electron affinity is $-\varepsilon^\text{qp}$ of the LUMO.

Line up the HOMO (the $1b_1$ oxygen lone pair) against experiment, whose
vertical IP is 12.62 eV:

| Estimate | IP / eV | Error / eV | What it contains |
|---|---|---|---|
| Koopmans, $-\varepsilon_\text{HOMO}$ | 13.54 | +0.92 | frozen orbitals: no relaxation, no correlation |
| $\Delta$SCF (UHF cation) | 10.85 | -1.77 | full orbital relaxation, no correlation |
| GF2, `result.ovgf` | 11.00 | -1.62 | second-order self-energy: relaxation and correlation to second order |
| GF2 renormalised (third order), `.out` | 11.66 | -0.96 | plus the diagonal third-order self-energy, geometrically resummed |
| experiment (vertical) | 12.62 | | |

The pattern is the standard one. Koopmans overshoots because the cation
is not allowed to relax. $\Delta$SCF undershoots by even more: the
relaxation is now complete but the neutral molecule has more correlation
energy than the cation, and dropping correlation altogether costs the
neutral more. Bare second-order GF2 also undershoots, because at second
order the $2h1p$ intermediate states (two holes, one particle) are treated
as non-interacting and the relaxation part of the self-energy is
overestimated; this "bare-GF2 overcorrection" is well documented. The
third-order terms couple the intermediate states and pull the IP back up
toward experiment; the residual 1 eV is mostly the 6-31G* basis, which
has no diffuse or higher polarisation functions, and the diagonal
approximation. With a triple-zeta basis the renormalised value typically
lands within a few tenths of an electronvolt.

The pole strengths carry a second message. $Z = 0.907$ for the HOMO means
that 91 percent of the ionisation intensity sits in one line and the
Koopmans one-hole picture is sound. Values below about 0.8 mean the hole
state is mixing strongly with shake-up ($2h1p$) configurations, the
quasiparticle peak is fragmenting into satellites, and a diagonal
propagator (or any single-configuration picture) is no longer adequate;
the variation below shows such a case in water's own inner valence.

The electron affinity is printed but flagged: $-5.3$ eV means water binds
no extra electron, which is correct, but the number itself is the energy
of a 6-31G* virtual orbital with no diffuse functions and should not be
quoted.

## Variation: all valence orbitals with the propagator API

The runner corrects the three highest occupied orbitals and prints the
renormalised value only for the HOMO. `vibeqc.propagator` gives you the
kernels directly, so you can run the third-order treatment on every
valence orbital.

```python
import numpy as np
import vibeqc as vq
from vibeqc.propagator import (quasiparticle_energies,
                               renormalized_quasiparticle_energies)

ANG = 1.8897259886
EV = 27.211386245988
h2o = vq.Molecule([vq.Atom(8, [0, 0, 0]),
                   vq.Atom(1, [0,  0.7572 * ANG, 0.5865 * ANG]),
                   vq.Atom(1, [0, -0.7572 * ANG, 0.5865 * ANG])], 0, 1)
basis = vq.BasisSet(h2o, "6-31g*")
ref = vq.run_rhf(h2o, basis)
C, eps = np.asarray(ref.mo_coeffs), np.asarray(ref.mo_energies)
n_occ = h2o.n_electrons() // 2

eri_ao = np.asarray(vq.compute_eri(basis))                       # (mu nu|la si)
eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, C, C, C, C, optimize=True)
valence = list(range(1, n_occ))                                  # skip the O 1s core

gf2 = quasiparticle_energies(eps, n_occ, eri_mo, valence, iterate=True)
ren = renormalized_quasiparticle_energies(eri_ao, C, C, eps, eps, n_occ, n_occ,
                                          [("alpha", p) for p in valence])
expt = {1: 32.2, 2: 18.55, 3: 14.74, 4: 12.62}                   # vertical IPs / eV
print(f"{'MO':>3} {'Koopmans':>9} {'GF2':>8} {'Z2':>6} {'GF2(ren)':>9} {'Z3':>6} {'expt':>6}")
for q2, (_, q3) in zip(gf2, ren):
    print(f"{q2.orbital:3d} {-q2.eps_scf * EV:9.3f} {-q2.eps_qp * EV:8.3f} {q2.pole_strength:6.3f} "
          f"{-q3.eps_qp * EV:9.3f} {q3.pole_strength:6.3f} {expt[q2.orbital]:6.2f}")

# The non-iterative estimate eps_p + Sigma(eps_p), one self-energy evaluation.
homo = quasiparticle_energies(eps, n_occ, eri_mo, [n_occ - 1], iterate=False)[0]
print(f"non-iterative GF2 HOMO IP = {-homo.eps_qp * EV:.3f} eV")
```

```
 MO  Koopmans      GF2     Z2  GF2(ren)     Z3   expt
  1    36.493   33.396  0.296    33.916  0.405  32.20
  2    19.221   17.977  0.930    18.360  0.943  18.55
  3    15.536   13.448  0.914    14.011  0.930  14.74
  4    13.539   11.004  0.907    11.663  0.925  12.62
non-iterative GF2 HOMO IP = 10.728 eV
```

Three things to take from this table. The three outer-valence ionisations
($1b_1$, $3a_1$, $1b_2$; MOs 4, 3, 2) all follow the HOMO pattern:
Koopmans high, GF2 low, the renormalised value in between and closer to
experiment, with pole strengths above 0.9. The inner-valence $2a_1$ (MO 1)
has $Z = 0.30$ at second order and $0.41$ at third: this is the famous
breakdown of the orbital picture in the inner valence, where the $2a_1$
hole mixes with many $2h1p$ shake-up states and the photoelectron
spectrum shows a broad band of satellites instead of one line. A diagonal
propagator cannot describe that region and the low $Z$ tells you so; do
not quote 33.9 eV as "the" $2a_1$ IP. Finally, the non-iterative estimate
$\varepsilon_p + \Sigma_{pp}(\varepsilon_p)$ differs from the converged
Dyson solution by 0.3 eV here; the runner always iterates.

The `("alpha", p)` orbital labels are the open-shell interface: for a
UHF reference `run_job(method="ovgf")` on a molecule with
`multiplicity > 1` builds the spin-orbital second-order self-energy and
corrects the spin-HOMO and spin-LUMO of each channel, and the same
function accepts different `Ca`, `Cb` and occupations.

## Limits and guards

The diagonal self-energy needs the full MO two-electron tensor, so the
routes are guarded rather than allowed to run out of memory. The
closed-shell runner refuses more than 100 basis functions with a
`NotImplementedError` ("builds the full MO two-electron tensor ... limited
to small/medium systems (n_bf <= 100)"); the open-shell runner builds a
$(2n_\text{bf})^4$ spin-orbital tensor and refuses more than 40 basis
functions (2 n_bf <= 80). The renormalised third-order HOMO value is
printed only when $2n_\text{bf} \le 60$, and
`renormalized_quasiparticle_energies` raises beyond its
`max_spinorbitals` (default 60) because the third-order contractions
scale steeply. For larger molecules use a smaller basis, or the
`vibeqc.propagator` API with a truncated orbital window. The same route
serves the MSINDO semiempirical reference through `msindo_ovgf`
(see [MSINDO](../user_guide/msindo.md)).

## Theory

The one-particle Green's function $G(\omega)$ describes the propagation
of an added electron or hole. Its poles in $\omega$ are the exact
ionisation potentials and electron affinities, and their residues are the
pole strengths. For the Hartree-Fock reference the zeroth-order Green's
function $G_0$ has poles at the orbital energies, which is Koopmans'
theorem, and the exact $G$ is related to it by the **Dyson equation**

$$
G(\omega) = G_0(\omega) + G_0(\omega)\,\Sigma(\omega)\,G(\omega),
$$

where the self-energy $\Sigma(\omega)$ collects every correlation and
relaxation effect. In the **diagonal approximation** the off-diagonal
elements $\Sigma_{pq}$, $p \ne q$, are dropped, the ionised state keeps
the shape of one canonical orbital, and each pole is found from the
one-dimensional **quasiparticle equation**

$$
\omega_p = \varepsilon_p + \Sigma_{pp}(\omega_p),
\qquad
Z_p = \Big(1 - \frac{\partial\Sigma_{pp}}{\partial\omega}\Big|_{\omega_p}\Big)^{-1},
$$

which `vibeqc.propagator._solve_dyson` solves by Newton's method. The
pole strength $Z_p \in (0,1]$ is the weight of the one-hole configuration
in the final state; $1 - Z_p$ has leaked into shake-up satellites.

To **second order** in the fluctuation potential the self-energy of a
closed-shell reference, in spatial orbitals with $i,j$ occupied and $a,b$
virtual and chemist's notation for the integrals, is

$$
\Sigma^{(2)}_{pp}(\omega) =
\sum_{iaj} \frac{\big[2(pi|aj) - (pj|ai)\big](pi|aj)}{\omega + \varepsilon_a - \varepsilon_i - \varepsilon_j}
+ \sum_{aib} \frac{\big[2(pa|ib) - (pb|ia)\big](pa|ib)}{\omega + \varepsilon_i - \varepsilon_a - \varepsilon_b}.
$$

The first sum runs over $2h1p$ intermediate states (two holes, one
particle): it is the relaxation and correlation channel that dominates
for ionisations and its denominators are small near an IP, so it drives
the IP *down* from Koopmans. The second, $2p1h$ sum matters for electron
attachment. This is the GF2 self-energy that `result.ovgf` reports; its
cost is one $O(N^5)$ integral transformation plus $O(o^2 v + o v^2)$ per
orbital and Newton step. Because the intermediate states are treated as
free, the second-order relaxation is too large, and GF2 IPs sit below
experiment, as the water table shows.

The **third-order** self-energy $\Sigma^{(3)}$ couples the intermediate
states: it contains static terms (ladder and ring diagrams with
$\omega$-independent denominators, the $W$ terms of electron-propagator
theory) and dynamic terms (the $U$ terms, with $\omega$ in the
denominator), each antisymmetrised over the two holes, and the
particle-hole mirror for the $2p1h$ channel. Evaluated as such, third
order alone would overcorrect in the other direction. The OVGF idea of
Cederbaum ({vibeqc-cite}`cederbaum_ovgf_1975`) and of von Niessen,
Schirmer and Cederbaum ({vibeqc-cite}`vonniessen_ovgf_1984`) is to treat
the ratio of third to second order as the first term of a geometric
series and resum it,

$$
\Sigma_c(\omega) \approx \frac{\Sigma^{(2)}_c(\omega)}{1 - \Sigma^{(3)}_c(\omega)/\Sigma^{(2)}_c(\omega)},
\qquad c \in \{2h1p,\;2p1h\},
$$

applied per channel, with a fall-back to the bare $\Sigma^{(2)} +
\Sigma^{(3)}$ when the ratio leaves the radius of convergence
(`screen_cap`, default 0.5). This is what `renormalized_quasiparticle_energies`
computes, with the derivative $\partial\Sigma/\partial\omega$ taken
analytically through the screening so that $Z$ is exact for the resummed
self-energy. The literal OVGF of the 1984 review defines three specific
screening partitions (versions A, B and C) and a selection rule between
them; vibe-qc implements the per-channel geometric screening of the full
diagonal third-order self-energy and labels it "renormalised GF2" for
that reason. The third-order contractions scale as $O(o^2 v^3)$ and
$O(o^3 v^2)$ per orbital and evaluation, which is why the routine is
guarded by a spin-orbital count.

Where the method helps and where it fails follows from the
approximations. Outer-valence IPs of closed-shell molecules with
$Z > 0.85$ are its home ground: one calculation gives the whole valence
photoelectron spectrum with errors of a few tenths of an electronvolt in
a good basis, at a cost comparable to MP2. It fails, and says so through
a small $Z$, for inner-valence and core ionisations, for satellite
structure and for strongly correlated references; there the
non-diagonal ADC(3) or an explicit $\Delta$CCSD(T) treatment is needed.
The same diagonal self-energy corrections work on top of semiempirical
NDDO references, as Danovich showed ({vibeqc-cite}`danovich_ovgf_1997`);
that is the route vibe-qc's `msindo_ovgf` follows.

## Resources

- [Post-SCF properties](post_scf_properties.md) and the
  [properties guide](../user_guide/properties.md): the orbital energies
  and Koopmans values the propagator starts from.
- [Open-shell SCF](open_shell.md): the UHF references that feed the
  spin-orbital kernel.
- [MSINDO](../user_guide/msindo.md): the same GF2 self-energy on the
  semiempirical integral set (`msindo_ovgf`).
- [Feature matrix](../features.md) and
  [Methods: theory and status](../methods.md#greens-function-ionisation-energies-gf2-and-ovgf).
- Tests: `tests/test_propagator.py` pins the self-energy against a
  definitional loop and central differences; `tests/test_runner_ovgf.py`
  pins the runner block and citations.

Cost: the RHF plus GF2 job on water/6-31G* runs in under four seconds;
the integral transformation is $O(N^5)$ and the tensor $n_\text{bf}^4$
doubles (about 0.8 MB here, 800 MB at the 100-function guard).

## References

- **OVGF.** L. S. Cederbaum, "One-body Green's function for atoms and
  molecules: theory and application," *J. Phys. B* **8**, 290 (1975),
  {vibeqc-cite}`cederbaum_ovgf_1975`. The renormalised outer-valence
  self-energy.
- **The 1984 review.** W. von Niessen, J. Schirmer and L. S. Cederbaum,
  "Computational methods for the one-particle Green's function,"
  *Comput. Phys. Rep.* **1**, 57 (1984), {vibeqc-cite}`vonniessen_ovgf_1984`.
  Second- and third-order self-energies, the OVGF A/B/C screenings and
  the ADC(3) alternative.
- **Semiempirical references.** D. Danovich, "NDDO semiempirical
  approximations coupled with Green's function technique: a reliable
  approach for calculating ionization potentials," *J. Mol. Struct.
  THEOCHEM* **401**, 235 (1997), {vibeqc-cite}`danovich_ovgf_1997`. The
  precedent for `msindo_ovgf`.
- **Textbook derivation.** A. Szabo and N. S. Ostlund, *Modern Quantum
  Chemistry* (Macmillan, 1982), Chapter 7: the second-order self-energy
  and Koopmans' theorem. J. V. Ortiz, "Electron propagator theory: an
  approach to prediction and interpretation in quantum chemistry,"
  *WIREs Comput. Mol. Sci.* **3**, 123 (2013), for the $W$ and $U$
  third-order terms.

## Next

- [Excited states with TDDFT](excited_states_tddft.md): neutral
  excitations from the same response-theory family.
- [Natural orbitals](natural_orbitals.md): a correlated density view of
  the reference whose orbitals were corrected here.
- [Coupled cluster](coupled_cluster_ccsd_t.md): the $\Delta$CCSD(T) route
  when one IP has to be right to 0.1 eV.
