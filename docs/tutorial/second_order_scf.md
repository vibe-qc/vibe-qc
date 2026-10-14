---
myst:
  html_meta:
    "description": "Second-order SCF in vibe-qc for difficult convergence: full-Hessian Newton, SOSCF, and trust-region augmented Hessian (TRAH). How to turn each on with its threshold option, a worked H2O comparison against DIIS, the orbital-rotation theory behind each method, a caution about hard starting densities, and how to choose among them."
    "og:title": "vibe-qc tutorial - second-order SCF (Newton, SOSCF, TRAH)"
---

# Second-order SCF: Newton, SOSCF, and TRAH

DIIS, the default accelerator, is fast and robust for most closed-shell
molecules, but it can stall on **difficult SCF problems**: transition-metal
complexes with near-degenerate d orbitals, stretched bonds near
dissociation, diffuse anions. When the density oscillates and the energy
will not settle, the cure is a **second-order** method that uses the
curvature of the energy (the orbital Hessian) to take a Newton step toward
the nearest stationary point.

vibe-qc ships three, all as fields on the standard options object and all
**off by default**. They are production finalizers (label vocabulary in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels)),
and the [Theory](#theory) section below derives what each one solves:

| Option | Method | Character |
|---|---|---|
| `newton_threshold` | Full-Hessian Newton | exact curvature; quadratic near convergence |
| `soscf_threshold` | SOSCF (L-BFGS quasi-Newton) | no extra Fock builds inside the step |
| `trah_threshold` | Trust-region augmented Hessian | robust far from convergence |

## Turning it on

Each is a *threshold*: it stays off (`0.0`) until the DIIS error drops below
the value you set, then the SCF switches from DIIS to the second-order step.
A large threshold (`1.0`) switches almost immediately; a small one lets DIIS
get close first, then polishes:

```python
import vibeqc as vq

mol = vq.Molecule([vq.Atom(8, [0, 0, 0]),
                   vq.Atom(1, [0,  1.43, -0.98]),
                   vq.Atom(1, [0, -1.43, -0.98])])
basis = vq.BasisSet(mol, "6-31g*")

opts = vq.RHFOptions()
opts.trah_threshold = 1.0            # switch to TRAH once the DIIS error < 1.0
result = vq.run_rhf(mol, basis, opts)
print(result.energy, result.n_iter)
```

## What it buys you

H2O / 6-31G*, RHF, every method reaching the same energy (a vibe-qc 0.17.10
development build; the counts include the DIIS iterations before the switch):

| Method | Iterations | E (Ha) |
|---|---|---|
| plain DIIS | 9 | -76.00667789 |
| EDIIS+DIIS (default) | 9 | -76.00667789 |
| Newton, `newton_threshold = 1.0` | 10 | -76.00667789 |
| Newton, `newton_threshold = 1e-2` | 6 | -76.00667789 |
| TRAH, `trah_threshold = 1.0` | 10 | -76.00667789 |
| SOSCF, `soscf_threshold = 1.0` | 16 | -76.00667789 |

The first thing to read off this table is that the default accelerator
already converges this molecule in nine iterations, so a second-order
method has little to win. Switching to Newton or TRAH immediately
(threshold `1.0`) costs one iteration; letting DIIS get close first and
polishing with Newton below `1e-2` saves three. For PBE on the same
molecule the counts are 8 (default), 9 (Newton or TRAH at `1.0`) and 15
(SOSCF at `1.0`). All variants
land on the *same* energy: a second-order method changes *how* the SCF gets
there, not *where*. Their value is as a finalizer for cases where DIIS
reaches the asymptotic regime and then stalls, not as a repair for a poor
start; the caution below shows a case where they do not rescue a bad
starting density.

```{admonition} A poor start is not repaired by a second-order switch
:class: warning

Stretched triplet O₂ (R = 4.0 bohr, cc-pVDZ, UHF, core-Hamiltonian
initial guess) is a hard start. Plain DIIS and the default EDIIS+DIIS both
converge in 49 iterations to -149.54504115 Ha. SOSCF converges to the same
energy in 107 iterations. Newton, switched in at any of the thresholds
1.0, 0.1, 1e-2 or 1e-3, ended on a different state after the iteration cap
(E = -149.263 Ha, reported <S²> = 0.000, `converged = False`), and TRAH
at `trah_threshold` of 1.0, 1e-2 and 1e-3 ran to its iteration cap (200 or
100) without reporting convergence, although its energy matched the DIIS one
to 1e-5 Ha or better. On ordinary
open-shell cases the same options work (the OH radical in 6-31G\*
converges in 12 iterations by default and in 8 with Newton or TRAH at
`1e-2`; O₂ triplet at R = 2.3 bohr in 10 and 9). These iteration counts were measured on the 2026-09-29 development build; a
correction to the noncanonical UHF/UKS TRAH Hessian action (#55) landed on main
afterwards, so the stretched-O₂ TRAH behaviour above should be re-checked on a
current build. So use a second-order
finalizer to finish an SCF that DIIS has brought close, check
`result.converged`, and compare the energy and <S²> with a DIIS run
before trusting a hard case.
```

## Choosing among them

- **TRAH** is the safe choice for a hard case: the trust region keeps the
  step sane even far from convergence, so it rarely diverges.
- **Newton** uses the exact orbital Hessian and converges quadratically once
  close, at the cost of building that Hessian.
- **SOSCF** trades exactness for cheaper steps; good when the Hessian is
  expensive and you only need to get unstuck.

They are mutually exclusive: if you set more than one threshold, Newton
takes precedence, then TRAH, then SOSCF. All four of RHF, UHF, RKS, and UKS
support them, through the matching `UHFOptions` / `RKSOptions` /
`UKSOptions` fields.

## Theory

**Orbital rotations.** The SCF energy is stationary with respect to
mixing occupied and virtual orbitals. Writing the orbitals as a unitary
rotation of the current ones,

$$
\mathbf C(\boldsymbol\kappa) = \mathbf C_0\,e^{\boldsymbol\kappa},
\qquad
\boldsymbol\kappa^{\mathsf T} = -\boldsymbol\kappa ,
$$

only the occupied-virtual elements $\kappa_{ai}$ ($i$ occupied, $a$ virtual)
change the energy, so the SCF becomes an unconstrained minimisation over
$n_\text{occ}\,n_\text{virt}$ parameters ({vibeqc-cite}`bacskay_qcscf_1981`
is the classic quadratically convergent formulation). Expanding to second
order,

$$
E(\boldsymbol\kappa) \approx E_0 + \mathbf g^{\mathsf T}\boldsymbol\kappa
  + \tfrac12\,\boldsymbol\kappa^{\mathsf T}\mathbf H\,\boldsymbol\kappa ,
\qquad
g_{ai} = 4\,F^\text{MO}_{ai},
\qquad
H_{ai,bj} = 4\bigl[\delta_{ij}\delta_{ab}(\varepsilon_a - \varepsilon_i)
  + 4(ai|bj) - (ab|ij) - (aj|bi)\bigr],
$$

where $F^\text{MO}_{ai}$ is the Fock matrix in the current MO basis, $\varepsilon$
the orbital energies, and the closed-shell Hartree-Fock form is shown (Kohn-Sham
adds the exchange-correlation kernel and scales the exchange terms by the
hybrid fraction). The electronic gradient $\mathbf g$ is the Brillouin
condition: it vanishes at self-consistency, and its norm is the same
quantity the commutator error tracks.

**Newton.** The stationary point of the quadratic model is
$\mathbf H\boldsymbol\kappa = -\mathbf g$. The Hessian is never formed: each
Hessian-vector product costs about one Fock-type build (Coulomb, exchange and
the XC kernel for DFT), and vibe-qc solves the Newton equation with
preconditioned conjugate gradients (`newton_opts.cg_tol`,
`newton_opts.cg_max_iter`, typically 5 to 15 iterations per Newton step),
capping the step length at `newton_opts.trust_radius` (0.3). Convergence is
quadratic near a minimum, but the model is only trustworthy there: far from
it $\mathbf H$ can have negative eigenvalues and the Newton step can be far
too long.

**SOSCF.** The approximate second-order method
({vibeqc-cite}`neese_soscf_2000`) avoids the Hessian-vector products. It
starts from the diagonal $4(\varepsilon_a - \varepsilon_i)$ as an inverse
Hessian, which alone has no curvature information and, for Kohn-Sham DFT,
oscillates because the diagonal ignores the large XC response; vibe-qc then
builds the missing two-electron and XC curvature from a short L-BFGS history
of $(\Delta\boldsymbol\kappa, \Delta\mathbf g)$ pairs. The step needs no Fock
build beyond the one the SCF iteration already performs, at the price of
linear rather than quadratic convergence.

**TRAH.** The trust-region augmented-Hessian method
({vibeqc-cite}`helmich_paris_trah_2021`) keeps the exact Hessian but
minimises the model inside a ball of radius $\Delta$,

$$
\min_{\boldsymbol\kappa}\;\mathbf g^{\mathsf T}\boldsymbol\kappa
  + \tfrac12\boldsymbol\kappa^{\mathsf T}\mathbf H\boldsymbol\kappa
\quad\text{subject to}\quad \lVert\boldsymbol\kappa\rVert \le \Delta ,
$$

which is solved with a level shift $\mu \ge 0$ (the augmented-Hessian
step) such that $(\mathbf H + \mu\mathbf 1)\boldsymbol\kappa = -\mathbf g$
has $\lVert\boldsymbol\kappa\rVert = \Delta$ whenever the plain Newton step
would leave the ball or $\mathbf H$ is indefinite. After each step the ratio
$\rho$ of the actual to the predicted energy change adjusts the radius
(`trah_opts`: halved when $\rho < 0.25$, doubled when $\rho > 0.75$ and the
step reached the boundary, between `min_trust_radius` and
`max_trust_radius`). Because a bad step is rejected and shortened rather than
taken, TRAH is the second-order method least likely to diverge.

**Choosing.** The thresholds are DIIS-error values below which the SCF hands
over from DIIS to the second-order step, and only one may be active
(Newton takes precedence over TRAH, TRAH over SOSCF). Use Newton for a fast,
quadratic finish when DIIS has reached the asymptotic regime and stalls;
TRAH when the surroundings of the solution are uncertain (near-degenerate
occupations, stretched bonds) and a bad step must be caught; SOSCF when a
Hessian-vector product is too expensive relative to a Fock build (large
hybrid-DFT systems). Coverage limits apply: meta-GGA second-order SCF still
raises, and Newton and TRAH are disabled for range-separated hybrids because
their orbital-Hessian exchange response is not range-separation aware.

## See also

- [Molecular SCF with OpenTrustRegion](opentrustregion.md) and
  [READ restarts and orbital stability](opentrustregion_stability.md), for the
  separately selected optional orbital optimizer and its worked examples.
- [Stiff molecular SCF: EDIIS+DIIS](ediis_diis_hybrid.md), the first thing to
  reach for before second order.
- [Initial guesses](initial_guess_walkthrough.md), since a better starting
  density often fixes convergence without any of this.
- [scf_convergence](../user_guide/scf_convergence.md) and the
  [keyword index](../user_guide/keyword_index.md) for every convergence knob
  and its default.
