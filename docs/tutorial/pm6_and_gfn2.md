---
myst:
  html_meta:
    "description": "Two more semiempirical engines in vibe-qc: PM6 (NDDO, experimental) and GFN2-xTB (extended tight-binding, experimental). NDDO theory in brief (multipole two-centre integrals, core-core terms, the PM6 parametrisation, the heat-of-formation convention and why MOPAC parity is still open), how to run each with run_job, real H2O energies, and why semiempirical totals live on each method's own reference scale."
    "og:title": "vibe-qc tutorial - PM6 and GFN2-xTB semiempirical methods"
---

# More semiempirical methods: PM6 and GFN2-xTB

Beyond [DFTB](semiempirical_dftb.md) and [MSINDO](msindo.md), vibe-qc ships
two more semiempirical engines for fast energies and gradients on large
systems: **PM6**, an NDDO method, and **GFN2-xTB**, an extended
tight-binding method. Both are experimental routes in vibe-qc (the status
boxes below say exactly what that means for each), both are far cheaper
than a minimal-basis Hartree-Fock, and both are the right tool for
preoptimization, conformer screening, and systems too large for ab initio.
The numbers on this page come from a vibe-qc 0.17.x development build on
Apple M2.

## Theory: NDDO in brief

PM6 belongs to the NDDO family (neglect of diatomic differential overlap)
that Dewar and Thiel introduced with MNDO
({vibeqc-cite}`dewar_thiel_1977`). The approximation is stated on the
atomic-orbital products: a product $\chi_\mu\chi_\nu$ of two orbitals on
*different* atoms is set to zero wherever it appears in a two-electron
integral, so of the four-index integrals only the one-centre
$(\mu_A\nu_A|\lambda_A\sigma_A)$ and the two-centre
$(\mu_A\nu_A|\lambda_B\sigma_B)$ survive, and the overlap matrix is taken as
the unit matrix in the Roothaan equations. The valence basis is minimal
($s$, $p$ and, in PM6, $d$ shells on the elements that need them); the core
electrons are folded into an effective core of charge $Z_A$ equal to the
number of valence electrons.

**Two-centre integrals from multipoles.** The surviving two-centre
integrals are not computed from the orbitals. Each charge distribution
$\mu_A\nu_A$ is replaced by a small set of point charges arranged as the
multipole it represents ($ss$ a monopole, $sp$ a dipole, $pp$ a monopole
plus a linear or square quadrupole) and the integral becomes a sum of
damped point-charge interactions in the Dewar-Sabelli-Klopman form,

$$
(\mu_A\nu_A|\lambda_B\sigma_B) \approx \sum_{i\in A}\sum_{j\in B}
\frac{q_i\,q_j}{\sqrt{\,r_{ij}^2 + \bigl(\rho_{l_i}^A + \rho_{l_j}^B\bigr)^2}} ,
$$

where the $q_i$ are the multipole point charges, $r_{ij}$ their
separations, and the additive terms $\rho_l^A$ are chosen so that the
integral reproduces the one-centre Slater-Condon parameters (the
element's $G_{ss}$, $G_{sp}$, $G_{pp}$, $G_{p2}$ and $H_{sp}$) at zero
separation and the correct $1/R$ at long range. The one-centre integrals
themselves are parameters fitted to atomic spectra, which is the historical
sense of "semiempirical". The one-electron resonance integral is
$\beta_{\mu\nu} = \tfrac12(\beta_\mu^A + \beta_\nu^B)\,S_{\mu\nu}$ with per-shell
parameters $\beta$ and the true overlap $S_{\mu\nu}$ (the overlap is used
here even though it is neglected in the secular equation).

**The core-core term.** Because the electron-core attraction is built
from the same damped multipole kernel, the core-core repulsion is written
consistently with it, as the $ss$ two-centre integral times the core
charges, so that well-separated neutral atoms do not interact. The
short-range balance, which the bare model gets wrong, is then set with an
empirical function,

$$
E_{\text{core}}^{AB} = Z_A Z_B\,(s_As_A|s_Bs_B)\,
\bigl[1 + f_{AB}(R_{AB})\bigr] ,
$$

and it is this $f_{AB}$ that most distinguishes the parametrisations:
MNDO used a sum of exponentials, and AM1 and PM3 added Gaussian corrections.
PM6 ({vibeqc-cite}`stewart_pm6_2007`) uses a *diatomic* form
$x_{AB}\exp\bigl[-\alpha_{AB}(R_{AB} + 0.0003\,R_{AB}^6)\bigr]$ whose two
parameters are fitted per element pair, keeps Gaussian corrections of the
AM1 type, and adds special-case terms for a few pairs. PM6 also reintroduced
$d$ orbitals for the main-group elements beyond the second row and for the
transition metals, and it was fitted to several thousand reference heats of
formation, geometries, dipoles and ionisation energies covering 70 elements.
The total energy is then the SCF electronic
energy plus $\sum_{A<B} E_{\text{core}}^{AB}$; vibe-qc reports it in
hartree, which is what the example below prints.

**Heats of formation and why MOPAC parity is open.** The quantity PM6
was fitted to is not the total energy but the standard heat of formation,

$$
\Delta H_f^\circ = E_{\text{tot}} - \sum_A E_{\text{el}}^A + \sum_A \Delta H_f^{\circ,A}
\quad\text{(in MOPAC's convention, with } E_{\text{tot}} \text{ in eV and } \Delta H_f \text{ in kcal/mol)} ,
$$

where $E_{\text{el}}^A$ is the PM6 energy of the isolated atom computed
with the same Hamiltonian and $\Delta H_f^{\circ,A}$ the experimental
atomisation enthalpy of the element. Reproducing MOPAC's number therefore
requires reproducing its atomic reference energies, its exact core-core
special cases and its unit conventions, not only its SCF. vibe-qc's PM6
carries the published Hamiltonian and the bundled MOPAC parameter file
({vibeqc-cite}`mopac_moussa_stewart_2026`), but exact heat-of-formation
parity with MOPAC is still open: where the bundled diatomic data is
incomplete the code falls back to the spherical Klopman-Ohno gamma and
generic core-core forms, and the comparison script
`examples/semiempirical/10_pm6_vs_mopac.py` documents the resulting,
characterised disagreement rather than parity. That is what the
experimental label below records. NDDO methods are good at organic
geometries, conformational trends and heats of formation of molecules
resembling the training set, and poor for anything dominated by dispersion
(no D-correction is built in), for transition states with unusual bonding,
and for elements whose parameters were fitted on sparse data.

## PM6

```{admonition} Status: experimental
:class: warning

PM6 (`method="pm6"`) is an experimental route. The code does not currently
emit a warning for it; the [semiempirical status table](../user_guide/semiempirical.md#method-status)
and the `vibeqc.semiempirical.semiempirical_route_status("pm6")` registry
record its maturity, exact MOPAC heat-of-formation parity is open, the
gradient is a native finite difference, UPM6 is molecular only, and periodic
PM6 (Gamma only, full-k fails closed) is experimental as well. Its numbers are
research output, not validated results, and it is not part of the release gate.
The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

PM6 is selected with `method="pm6"` and needs no basis-set argument: it
carries its own minimal valence basis and parameter set.

```python
import vibeqc as vq

mol = vq.Molecule([vq.Atom(8, [0, 0, 0]),
                   vq.Atom(1, [0,  1.43, -0.98]),
                   vq.Atom(1, [0, -1.43, -0.98])])
result = vq.run_job(mol, method="pm6", output="h2o_pm6")
print(result.energy)
```

```text
SCF converged in 41 iterations;  E = -11.7252749834 Ha
```

PM6 provides energies and finite-difference gradients, so it plugs into
geometry optimization the same way the ab-initio methods do.

`load_pm6_params()` covers H, C, N, O and F with the 2007 Stewart parameters;
for any other element `load_pm6_params_auto(atomic_numbers)` (in
`vibeqc.semiempirical.methods.pm6_params`) selects the bundled MOPAC parameter
file (75 elements). Molecules with several heavy atoms run (ethylene
converges to -10.9712 Ha in 51 iterations), but exact MOPAC heat-of-formation
parity is still open, and spherical Klopman-Ohno gamma and core-core fallbacks
apply where the bundled diatomic data is incomplete, so treat the totals as
PM6-in-vibe-qc numbers rather than a quantitative MOPAC reference. Elements
whose PM6 record needs $d$ orbitals fail closed in the periodic and SECCM
adapters.

## GFN2-xTB

```{admonition} Status: experimental
:class: warning

GFN2-xTB (`method="gfn2_xtb"`) is an experimental route. It emits
`GFN2ExperimentalWarning` on every evaluation; it is closed-shell only at
every public boundary; faithful periodic multipole electrostatics and external
`xtb` parity remain production gates. Its numbers are research output, not
validated results, and it is not part of the release gate. See
[Methods: theory and status](../methods.md#how-to-read-the-status-labels) and
the [GFN2-xTB workshop](gfn2_xtb_workshop.md), whose Theory section derives
the Hamiltonian.
```

GFN2-xTB is Grimme's extended semiempirical tight-binding method: a
second-order tight-binding expansion like SCC-DFTB, with shell-resolved
charges, anisotropic (dipole and quadrupole) electrostatics, a third-order
on-site term and D4 dispersion, parametrised up to radon. The warning is
opted past explicitly:

```python
import warnings
import vibeqc as vq

with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # opt in to the experimental method
    result = vq.run_job(mol, method="gfn2_xtb", output="h2o_gfn2")
print(result.energy)
```

```text
SCF converged in 47 iterations;  E = -5.0650921073 Ha
```

## Reading the numbers

Semiempirical total energies live on **each method's own parametrised
reference scale**. The PM6 number (−11.73 Ha) and the GFN2-xTB number
(−5.07 Ha) are not comparable to each other, nor to an ab-initio total (the
same water at HF / 6-31G* is −76.01 Ha). What *is* meaningful is **relative**
energies within one method (conformers, reaction energies) and the
geometries and properties the method produces. The standard workflow is to
get a structure or a trend fast with a semiempirical method, then refine the
energetics with DFT or a correlated method.

## See also

- [Semiempirical DFTB](semiempirical_dftb.md) and [MSINDO](msindo.md): the
  other two molecular semiempirical engines.
- [Machine-learning interatomic potentials with MACE](mace_mlip.md): the
  other fast, non-ab-initio route.
- [The cyclic cluster model](cyclic_cluster_model.md): MSINDO carried to
  periodic systems.
