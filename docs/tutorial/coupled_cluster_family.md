---
myst:
  html_meta:
    "description": "Hands-on vibe-qc tutorial for the coupled-cluster and coupled-pair ladder on one molecule: MP2, CC2, CEPA(1), CCD, LCCD, CCSD, QCISD, CEPA(0), CCSD(T), QCISD(T), BCCD and BCCD(T) on water in 6-31G through run_job, with CC3 and CCSDT in STO-3G and a frozen-core CISD for the size-extensivity comparison. Tabulates the correlation energies and derives which terms of the CCSD equations each approximation drops."
    "og:title": "vibe-qc tutorial - the coupled-cluster family from MP2 to CCSDT"
---

# The coupled-cluster family: from MP2 to CCSDT

Between second-order perturbation theory and full configuration
interaction lies a ladder of approximations that all descend from the
same coupled-cluster (CC) equations: keep the doubles but not the
singles (CCD), drop the nonlinear terms (LCCD, CEPA(0)), add pair-specific
shifts (CEPA(1)), keep only the quadratic terms that restore size
extensivity (QCISD), approximate the doubles to first order (CC2), add
triples perturbatively (CCSD(T), QCISD(T)), iteratively but approximately
(CC3), or exactly (CCSDT), or rotate the orbitals until the singles
vanish (BCCD). Every one of them is a statement about which terms of the
CCSD amplitude equations matter, and the cleanest way to learn the
family is to run all of them on one small molecule, in one basis, with
one set of integrals, and look at the numbers side by side.

vibe-qc exposes the whole ladder through `run_job(method=...)` on a
closed-shell RHF reference; the coupled-pair and QCI variants share the
canonical CCSD kernel (`CCSDOptions.cc_variant`), CC3 and CCSDT run in an
exact determinant-space representation, and all of them are production
within the envelopes stated in the [CCSD user guide](../user_guide/ccsd.md)
(see [Methods: theory and status](../methods.md#how-to-read-the-status-labels)).
This page starts with a three-method pilot on water in 6-31G, provides a
twelve-method extension, and offers CC3/CCSDT in a smaller STO-3G space.
A frozen-core CISD comparison illustrates size extensivity. Timings below
are historical measurements, not resource guarantees.

## Setup

Run this page's blocks in order with the installed engine's Python, outside
the source checkout. Later sections reuse `h2o`, `vq` and `time`. Start with
the three-method pilot below; the complete ladder and determinant-space
variations are optional. Keep each method's output family for its own
convergence, resolved core count and runtime record.

The 6-31G basis has no registered RI auxiliary basis, so the run passes
`density_fit=False` and every method works on the same exact four-index
integrals. The separate STO-3G variation also uses conventional integrals,
but changes the orbital basis: compare its CC3/CCSDT values with its own
STO-3G CCSD(T) row, not with the 6-31G table. `run_job` freezes the oxygen
1s core by default for all of these methods.

```python
import time
import vibeqc as vq

ANG = 1.8897259886
h2o = vq.Molecule([
    vq.Atom(8, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.0,  0.7572 * ANG, 0.5865 * ANG]),
    vq.Atom(1, [0.0, -0.7572 * ANG, 0.5865 * ANG]),
], 0, 1)

full_ladder = ["mp2", "cc2", "cepa(1)", "ccd", "lccd", "ccsd", "qcisd", "cepa(0)",
          "ccsd(t)", "qcisd(t)", "bccd", "bccd(t)"]
ladder = ["mp2", "ccsd", "ccsd(t)"]  # use full_ladder after the pilot succeeds
rows = []
for method in ladder:
    t0 = time.perf_counter()
    res = vq.run_job(h2o, basis="6-31g", method=method, density_fit=False,
                     output="h2o_" + method.replace("(", "_").replace(")", ""), verbose=0)
    wall = time.perf_counter() - t0
    if not res.converged:
        raise RuntimeError(f"SCF reference did not converge for {method}")
    post = res.mp2 if method == "mp2" else res.ccsd      # MP2Result or CCSDResult
    if method != "mp2" and not post.converged:
        raise RuntimeError(f"Correlated solver did not converge for {method}")
    rows.append((method, post.e_hf, post.e_total - post.e_hf, post.e_total, wall))

print(f"{'method':9s} {'E_ref / Ha':>12s} {'E_corr / Ha':>12s} {'E_total / Ha':>14s} {'wall / s':>9s}")
for method, e_ref, e_corr, e_tot, wall in rows:
    print(f"{method:9s} {e_ref:12.6f} {e_corr:12.6f} {e_tot:14.6f} {wall:9.1f}")
```

The following table records the full ladder, not just the three pilot rows.
On a vibe-qc 0.17.10 development build (Apple M5 Max) each method takes
two to five seconds, RHF included:

```
method      E_ref / Ha  E_corr / Ha   E_total / Ha  wall / s
mp2         -75.983974    -0.127814     -76.111788       2.7
cc2         -75.983974    -0.128585     -76.112559       2.0
cepa(1)     -75.983974    -0.133233     -76.117208       2.0
ccd         -75.983974    -0.133791     -76.117765       2.0
lccd        -75.983974    -0.133959     -76.117934       1.7
ccsd        -75.983974    -0.134471     -76.118446       1.7
qcisd       -75.983974    -0.134612     -76.118586       3.0
cepa(0)     -75.983974    -0.134814     -76.118789       2.2
ccsd(t)     -75.983974    -0.135456     -76.119431       1.7
qcisd(t)    -75.983974    -0.135502     -76.119476       2.7
bccd        -75.983157    -0.135170     -76.118328       4.8
bccd(t)     -75.983157    -0.136265     -76.119423       4.3
```

The wall times were measured with other jobs running and vary by a
factor of two between runs; the energies do not.

## What just happened

The methods start from the same RHF calculation and 12 active orbitals
(13 minus the frozen core). The BCCD rows subsequently rotate the orbitals,
so their reported reference energy differs, as discussed below. The post-SCF result
is attached as `res.mp2` (an `MP2Result`: `e_hf`, `e_correlation`,
`e_total`) or `res.ccsd` (a `CCSDResult`: `e_hf`, `e_ccsd_correlation`,
`e_t`, `e_total`, `n_iter`, `cc_trace`), and `res.energy_total` is the
correlated total energy in both cases while `res.energy` stays the SCF
energy. Read the table from the top.

**MP2 to CC2 to CCSD.** MP2 recovers $-127.8$ mHa, CC2 $-128.6$ and CCSD
$-134.5$. CC2 is MP2's iterative cousin: its doubles are correct only to
first order in the fluctuation potential, so at a single geometry it
sits within a millihartree of MP2; its value is that the singles are
treated exactly, which makes it a proper model for response properties
and excitation energies where MP2 has none. The 6 mHa between them and
CCSD is the effect of the infinite-order doubles.

**CCD and LCCD versus CCSD.** Freezing the singles (CCD, $-133.8$) costs
0.7 mHa relative to CCSD: singles are small for a molecule at its
equilibrium geometry with a good RHF reference, and their role is to
relax the orbitals in the presence of correlation. Linearising the
doubles (LCCD, $-134.0$) *overshoots* CCD slightly. That is the general
rule: the quadratic $T_2^2$ terms that LCCD drops are net repulsive
(they contain the exclusion-principle-violating corrections), so every
linearised method recovers too much correlation energy.

**The coupled-pair family.** CEPA(0), which is LCCSD, overshoots
further ($-134.8$, beyond CCSD) for the same reason, while CEPA(1)
($-133.2$) applies Meyer's pair-specific energy shifts that mimic the
missing quadratic terms and lands *above* CCSD. CCSD is bracketed
between the two, which is the historical reason coupled-pair methods
were considered good approximations to it before CCSD was affordable.
The CEPA numbers depend on the choice of occupied orbitals (vibe-qc uses
canonical orbitals; see the user guide); the CEPA(0) value does not.

**QCISD and CCSD.** QCISD ($-134.6$) sits 0.14 mHa below CCSD. QCISD is
CISD plus the two quadratic terms needed for size extensivity; it lacks
the remaining nonlinear terms of CCSD, and it is known to overestimate
correlation slightly relative to CCSD for exactly the reason LCCD does.

**Triples.** The (T) increment is $-0.985$ mHa on CCSD and $-0.890$ mHa
on QCISD (`res.ccsd.e_t`); note that QCISD(T) uses the original
Pople-Head-Gordon-Raghavachari convention $E[T] + 2E_{ST}$ on QCISD
amplitudes, whereas CCSD(T) uses $E[T] + E_{ST}$, so the two triples
increments are defined differently and the totals should not be
compared term by term. Both land within 0.05 mHa of each other here,
which is typical for a well-behaved closed-shell molecule.

**Brueckner orbitals.** BCCD changes the reference: its $E_\text{ref}$
($-75.983157$ Ha) is the energy of the Brueckner determinant, 0.8 mHa
above RHF, because the orbitals have been rotated until the singles
amplitudes vanish rather than to minimise the determinant's energy. Its
correlation energy ($-135.2$ mHa) is correspondingly larger, and the
total BCCD energy is within 0.12 mHa of CCSD; add the triples and
BCCD(T) is within 0.01 mHa of CCSD(T). This is the expected behaviour
when singles are small; Brueckner orbitals earn their cost when the RHF
reference is poor (spin contamination, symmetry breaking, large $T_1$),
where BCCD(T) is markedly more stable than CCSD(T).

## Variation: CC3 and CCSDT in a minimal basis

CC3 and CCSDT live in an exact determinant-space representation whose
size grows combinatorially with the number of triples, so vibe-qc labels
them energy-only and benchmark-scale. In 6-31G the CC3 job on water runs
for minutes; in STO-3G, with two virtual orbitals, the whole hierarchy
takes seconds and shows the ordering exactly.

```python
# doc-audit: skip - continues with h2o, vq and time from Setup
for method in ("ccsd", "ccsd(t)", "cc3", "ccsdt"):
    t0 = time.perf_counter()
    res = vq.run_job(h2o, basis="sto-3g", method=method, density_fit=False,
                     output="h2o_sto3g_" + method.replace("(", "_").replace(")", ""), verbose=0)
    if not res.converged:
        raise RuntimeError(f"Calculation did not converge for {method}")
    if hasattr(res, "ccsd"):
        if not res.ccsd.converged:
            raise RuntimeError(f"Coupled-cluster amplitudes did not converge for {method}")
        e_corr, e_tot, extra = res.ccsd.e_total - res.ccsd.e_hf, res.ccsd.e_total, ""
    else:                                        # CC3Result / CCSDTResult
        e_corr, e_tot = res.e_correlation, res.energy
        extra = f"  ({res.n_doubles} doubles, {res.n_triples} triples, {res.n_frozen_core} frozen)"
    print(f"{method:8s} E_corr = {e_corr:.6f}  E_total = {e_tot:.6f} Ha  {time.perf_counter() - t0:5.1f} s{extra}")
```

```
ccsd     E_corr = -0.049360  E_total = -75.012383 Ha    2.4 s
ccsd(t)  E_corr = -0.049428  E_total = -75.012451 Ha    1.8 s
cc3      E_corr = -0.049428  E_total = -75.012451 Ha   16.5 s  (76 doubles, 96 triples, 1 frozen)
ccsdt    E_corr = -0.049454  E_total = -75.012477 Ha   17.3 s  (76 doubles, 96 triples, 1 frozen)
```

In this tiny space the perturbative (T) and the iterative CC3 triples
agree to $10^{-7}$ Ha, and full CCSDT adds another 0.026 mHa. The
ordering $|E_\text{CCSD}| < |E_\text{CCSD(T)}| \approx |E_\text{CC3}|
< |E_\text{CCSDT}|$ is the usual one near equilibrium; CC3 separates
from CCSD(T) when the triples become large (stretched bonds, multiple
bonds, open shells), where the iterative treatment keeps its accuracy
and the perturbative one degrades. CC3 saves nothing over CCSDT at this
size because the dense representation regenerates its approximate
triples at every iteration; the asymptotic scalings in the theory
section are what matter for anything larger. Both return a `SolverResult`-like
object (`CC3Result`, `CCSDTResult`) with `e_reference`, `e_correlation`,
`energy` and the residual norms, and `res.energy` *is* the correlated
total energy for these two, unlike the wrapped results above.

## Variation: CISD and size extensivity

CISD is the one method on the page that is *not* size-extensive. Run it
in the same frozen-core space (`active_space=(12, 8)`: 12 orbitals, 8
electrons, which freezes the lowest orbital):

```python
# doc-audit: skip - continues with h2o, vq and time from Setup
cisd = vq.run_job(h2o, basis="6-31g", method="cisd", active_space=(12, 8),
                  output="h2o_cisd", verbose=0)
rhf = vq.run_job(h2o, basis="6-31g", method="rhf", output="h2o_rhf", verbose=0)
if not cisd.converged or not rhf.converged:
    raise RuntimeError("CISD or its RHF comparison did not converge")
print(f"{cisd.method}: E_corr = {cisd.energy - rhf.energy:.6f} Ha")
```

```
cisd(ndet=1425): E_corr = -0.129228 Ha
```

CISD recovers $-129.2$ mHa, less than LCCD ($-134.0$) although it
contains every singly and doubly excited determinant that CCSD does.
The difference is the unlinked terms: in CISD the doubles equation
carries a term $-E_\text{corr}\,c_{ij}^{ab}$ that grows with the size of
the system and damps the amplitudes, so the fraction of the correlation
energy CISD recovers *decreases* as molecules get bigger and the energy
of two non-interacting molecules is not twice that of one. Every
coupled-cluster and coupled-pair variant above removes that term (CEPA
methods approximately, QCISD and the CC methods exactly), which is why
their energies are additively separable and why CISD has left the
production toolbox. The determinant CISD here is a Python solver and
takes about a minute; it is on the page for the comparison, not for
routine use.

## Theory

Coupled-cluster theory ({vibeqc-cite}`cizek_cc_1966`) writes the
correlated wavefunction as an exponential of excitation operators acting
on the reference determinant,

$$
|\Psi\rangle = e^{\hat T}|\Phi_0\rangle,
\qquad
\hat T = \hat T_1 + \hat T_2 + \hat T_3 + \cdots,
\qquad
\hat T_2 = \tfrac14\sum_{ijab} t_{ij}^{ab}\,a_a^\dagger a_b^\dagger a_j a_i ,
$$

with $i,j$ occupied and $a,b$ virtual spin orbitals. Projecting the
similarity-transformed Schrodinger equation
$\bar H = e^{-\hat T}\hat H e^{\hat T}$ onto the reference and onto the
excited determinants gives the energy and the amplitude equations,

$$
E = \langle\Phi_0|\bar H|\Phi_0\rangle,
\qquad
0 = \langle\Phi_{ij}^{ab}|\bar H|\Phi_0\rangle,
\quad
0 = \langle\Phi_i^{a}|\bar H|\Phi_0\rangle, \;\ldots
$$

Because $\bar H$ terminates at fourth order in $\hat T$ (the
Baker-Campbell-Hausdorff expansion of a two-body Hamiltonian), the CCSD
doubles residual is a finite polynomial: linear terms
($\langle ab\|ij\rangle$, the Fock-operator terms, the particle-particle
and hole-hole ladders and the ring terms in $t_2$), quadratic terms in
$t_2$ (the "disconnected quadruples" $\tfrac12 T_2^2$ contribution and
the exclusion-principle-violating terms), and the terms in $t_1$ up to
$t_1^4$. The exponential guarantees that every term is *connected*, and
connectedness is what makes the energy size-extensive: the energy of two
non-interacting fragments is the sum of their energies at every
truncation level of $\hat T$.

Each member of the family is a rule for which of those terms to keep:

| Method | Approximation to the CCSD (or CCSDT) equations |
|---|---|
| **CCD** | $\hat T = \hat T_2$; the singles equation is dropped and $t_1 = 0$ ({vibeqc-cite}`cizek_cc_1966`). |
| **LCCD** | CCD with the doubles residual truncated to the terms linear in $t_2$; the equations become a linear system, equivalent to summing all double-excitation diagrams of many-body perturbation theory to infinite order. |
| **CEPA(0)** = LCCSD | Singles and doubles, all nonlinear terms dropped. |
| **CEPA(1), (2), (3)** | The linear residual plus a pair-specific shift $-\Delta_{ij}\,t_{ij}^{ab}$ that models the missing exclusion-principle-violating quadratic terms; $\Delta_{ij}$ is built from the pair correlation energies $e_{kl}$ (CEPA(1): $\tfrac12\sum_k(e_{ik}+e_{jk})$; CEPA(2): $e_{ij}$; CEPA(3): $-e_{ij}+\sum_k(e_{ik}+e_{jk})$), following Meyer ({vibeqc-cite}`meyer_cepa_1973`) in the closed-shell convention tabulated by Wennmohs and Neese ({vibeqc-cite}`wennmohs_neese_cepa_2008`). |
| **QCISD** | CISD made size-extensive by adding the quadratic $\tfrac12\hat T_2^2$ term to the doubles equation and $\hat T_1\hat T_2$ to the singles equation, with the CI-like energy expression (no $\hat T_1^2$ contribution); the other nonlinear terms of CCSD are absent ({vibeqc-cite}`pople_qcisd_1987`). |
| **CC2** | The singles equation is the full CCSD one; the doubles equation keeps only the terms of first order in the fluctuation potential, evaluated with the $\hat T_1$-dressed Hamiltonian, plus $[\hat F, \hat T_2]$ ({vibeqc-cite}`christiansen_cc2_1995`). Its energy is correct to second order, like MP2, but the singles are treated to all orders. |
| **CCSD** | All singles and doubles terms ({vibeqc-cite}`purvis_bartlett_ccsd_1982`). |
| **CCSD(T)** | CCSD plus the non-iterative triples correction $E_{[T]} + E_{ST}$: the fourth-order triples energy from converged $t_2$ and the fifth-order singles-triples coupling ({vibeqc-cite}`raghavachari_ccsdt_1989`). QCISD(T) uses $E_{[T]} + 2E_{ST}$ on QCISD amplitudes. |
| **CC3** | Triples included iteratively, with the triples equation approximated to leading order (the $\hat T_1$-dressed Hamiltonian acting on $\hat T_2$) and no triples-triples coupling beyond the Fock term ({vibeqc-cite}`koch_cc3_1997`). |
| **CCSDT** | The full singles, doubles and triples equations ({vibeqc-cite}`noga_bartlett_ccsdt_1987`). |
| **BCCD, BCCD(T)** | CCD in the basis of Brueckner orbitals, rotated iteratively until the CCSD singles amplitudes vanish, so that the reference determinant has maximum overlap with the correlated wavefunction; (T) is then evaluated on that reference ({vibeqc-cite}`handy_bccd_1989`). |

The **cost** follows the highest excitation level and the iteration
structure. With $o$ occupied and $v$ virtual orbitals, MP2 is a single
$O(o^2 v^3)$ transformation; CC2 iterates an $O(o^2 v^3)$ step; CCD,
LCCD, CEPA($n$), QCISD and CCSD iterate the $O(o^2 v^4)$ particle-particle
ladder; the (T) correction is one non-iterative $O(o^3 v^4)$ pass; CC3
iterates an $O(o^3 v^4)$ step, so it costs a CCSD(T) at every iteration;
and CCSDT iterates $O(o^3 v^5)$ with $o^3 v^3$ amplitudes to store. That
is why (T) is the production triples method and CC3 and CCSDT are
benchmarks: on water in 6-31G the ladder up to BCCD(T) runs in seconds,
CC3 in minutes, and a triple-zeta CCSDT on anything larger than a
diatomic is out of reach without a dedicated tensor engine.

Where the family helps is the accuracy hierarchy it gives you at a known
price: near-equilibrium closed-shell molecules with a good RHF
reference, where CCSD(T) is within a kilocalorie of the exact answer in
a large basis and the cheaper rungs tell you how far from convergence
you are. Where it fails is where the single reference fails: stretched
or multiple bonds, biradicals, transition-metal multiplets, where the
$T_1$ diagnostic grows, (T) can overshoot badly, and the iterative
triples of CC3 or, better, a multireference treatment
([CASSCF and CASPT2](casscf_multireference.md)) takes over. The
coupled-pair and QCI variants are not more accurate than CCSD; they are
on the page because they are cheaper rungs with a clear diagrammatic
meaning, because they remain the workhorses of local-correlation methods
(LPNO-CEPA), and because the sign of their deviation from CCSD tells you
which terms of the equations are doing the work.

## Resources

- [CCSD user guide](../user_guide/ccsd.md): the `method=` and `citype=`
  selectors, the `cc_variant` option, the frozen-core convention, the
  ORCA parity numbers for each variant, and the memory limits.
- [Canonical CCSD(T)](coupled_cluster_ccsd_t.md): the open-shell,
  ROHF and frozen-natural-orbital routes, and the density-fitting
  versus exact-integral choice.
- [RI-MP2](ri_mp2_aux_verification.md) and [MP2 and double hybrids](../user_guide/mp2_and_double_hybrids.md):
  the bottom rung and its spin-component-scaled variants.
- [DLPNO local correlation](dlpno_local_correlation.md): the
  reduced-scaling CCSD(T) for molecules too large for the canonical
  kernel.
- [Non-HF solvers](../user_guide/non_hf_solvers.md): the determinant
  CISD, FCI and CC3/CCSDT solvers.
- [Feature matrix](../features.md) and
  [Methods: theory and status](../methods.md#the-wider-coupled-cluster-and-coupled-pair-family-cc2-cc3-ccsdt-qcisdt-cepan).

## References

- **Coupled-cluster theory.** J. Cizek, "On the correlation problem in
  atomic and molecular systems," *J. Chem. Phys.* **45**, 4256 (1966),
  {vibeqc-cite}`cizek_cc_1966`. The exponential ansatz and CCD.
- **CCSD.** G. D. Purvis and R. J. Bartlett, "A full coupled-cluster
  singles and doubles model," *J. Chem. Phys.* **76**, 1910 (1982),
  {vibeqc-cite}`purvis_bartlett_ccsd_1982`.
- **CCSD(T).** K. Raghavachari, G. W. Trucks, J. A. Pople and
  M. Head-Gordon, "A fifth-order perturbation comparison of electron
  correlation theories," *Chem. Phys. Lett.* **157**, 479 (1989),
  {vibeqc-cite}`raghavachari_ccsdt_1989`.
- **CCSDT.** J. Noga and R. J. Bartlett, "The full CCSDT model for
  molecular electronic structure," *J. Chem. Phys.* **86**, 7041 (1987),
  {vibeqc-cite}`noga_bartlett_ccsdt_1987`.
- **CC2.** O. Christiansen, H. Koch and P. Jorgensen, "The second-order
  approximate coupled cluster singles and doubles model CC2," *Chem.
  Phys. Lett.* **243**, 409 (1995), {vibeqc-cite}`christiansen_cc2_1995`.
- **CC3.** H. Koch, O. Christiansen, P. Jorgensen, A. M. Sanchez de
  Meras and T. Helgaker, "The CC3 model: An iterative coupled cluster
  approach including connected triples," *J. Chem. Phys.* **106**, 1808
  (1997), {vibeqc-cite}`koch_cc3_1997`.
- **QCISD and QCISD(T).** J. A. Pople, M. Head-Gordon and
  K. Raghavachari, "Quadratic configuration interaction. A general
  technique for determining electron correlation energies," *J. Chem.
  Phys.* **87**, 5968 (1987), {vibeqc-cite}`pople_qcisd_1987`.
- **CEPA.** W. Meyer, "PNO-CI studies of electron correlation effects.
  I.," *J. Chem. Phys.* **58**, 1017 (1973), {vibeqc-cite}`meyer_cepa_1973`;
  F. Wennmohs and F. Neese, "A comparative study of single reference
  correlation methods of the coupled-pair type," *Chem. Phys.* **343**,
  217 (2008), {vibeqc-cite}`wennmohs_neese_cepa_2008`, the shift
  conventions and the benchmark that places CEPA(1) beside CCSD.
- **Brueckner CCD.** N. C. Handy, J. A. Pople, M. Head-Gordon,
  K. Raghavachari and G. W. Trucks, "Size-consistent Brueckner theory
  limited to double substitutions," *Chem. Phys. Lett.* **164**, 185
  (1989), {vibeqc-cite}`handy_bccd_1989`.
- **CISD and size consistency.** J. A. Pople, R. Seeger and R. Krishnan,
  "Variational configuration interaction methods and comparison with
  perturbation theory," *Int. J. Quantum Chem.* **12**, 149 (1977),
  {vibeqc-cite}`pople_cisd_1977`.

## Next

- [Canonical CCSD(T)](coupled_cluster_ccsd_t.md) takes the top of the
  production ladder to open shells, larger bases and frozen natural
  orbitals.
- [Basis convergence](basis_convergence.md): the correlation energies
  above are 6-31G numbers; the basis-set error dwarfs every difference
  in the table.
- [Multireference: CASSCF, CASPT2, NEVPT2](casscf_multireference.md)
  is where to go when the single-reference ladder stops converging.
