---
myst:
  html_meta:
    "description": "Every method vibe-qc offers, with the theory behind it, the keyword that selects it, and its status label: production, preview, experimental or gated. Mean field, correlated wavefunction, multireference, excited states, semiempirical, machine-learned potentials, periodic routes, workflows and analysis."
    "og:title": "vibe-qc methods: theory and status"
---

# Methods: theory and status

This page is the map of everything vibe-qc can compute. Each method family
gets a short account of the physics it rests on, the working equations that
the code solves, the keyword that selects it, and the **status label** the
project gives it. The [feature matrix](features.md) has the exhaustive option
tables; the [tutorials](tutorial/index.md) derive each method at length and
run it; the [user guide](user_guide/index.md) documents options, defaults and
limits. Use this page to decide *which* method, then follow the links to
learn *how*.

## How to read the status labels

vibe-qc labels every route with one of four words. The label is not a
judgement of the underlying theory; it says how far *this implementation*
has been validated and whether the release gate protects it.

| Label | What it means | How you can tell at runtime |
|---|---|---|
| **production** | Validated within its documented scope against an external reference (PySCF, ORCA, CRYSTAL, CP2K or a published number), covered by the T1 release gate, and stable across patch releases. | No warning. The `.system` manifest records `method_status = "production"` where the route reports one. |
| **preview** | Reachable and maintained, but with a narrower validation envelope than production: a route that ships for one reference type, one dimensionality or one regime, with the rest fail-closed. | The `.out` names the caveat; some previews warn. Options outside the envelope raise `NotImplementedError`. |
| **experimental** | Research output. It runs and it is cited, but its numbers are not validated for publication, it is not release-gated, and it may change without a deprecation cycle. | Emits `ExperimentalWarning` or a feature-specific subclass every time it runs; the manifest records `method_status = "experimental"`. |
| **gated** | Fails closed. The spelling is recognised, but the route refuses with a scientific message naming the missing term rather than returning a number it cannot stand behind. | `ValueError` or `NotImplementedError` before any SCF iteration. |

```{admonition} Experimental lines never pretend to be production
:class: warning

The ab initio cyclic cluster model (AICCM, both the Gamma-CCM and the chi-CCM
constructions), the semiempirical cyclic cluster model (SECCM), GAPW
augmentation, periodic RIJCOSX, GFN2-xTB, PM6, the OMx family, SKALA, COSMO-RS,
Green's-function surface embedding and the MPI grid overlay are experimental
today. Every one of them warns at runtime. Treat their results as research
output. The [roadmap](roadmap.md#experimental-lines-not-release-gated) is the
authoritative list; the sections below repeat the label beside each method.
```

Silencing a warning is deliberate and per class, for example
`warnings.filterwarnings("ignore", category=vibeqc.AICCM2026DevAExperimentalWarning)`,
so an experimental route never runs silently by accident.

(methods-electronic-structure-problem)=

## The problem every method solves

Under the Born-Oppenheimer approximation the nuclei are clamped and the
electrons obey the electronic Schroedinger equation with the Hamiltonian

$$
\hat H = -\tfrac12\sum_i \nabla_i^2
      - \sum_{i,A}\frac{Z_A}{|\mathbf r_i-\mathbf R_A|}
      + \sum_{i<j}\frac{1}{|\mathbf r_i-\mathbf r_j|} ,
$$

in atomic units. Every method below is a different way of approximating the
lowest eigenvalue $E_0$ of this operator, or an excited eigenvalue, or a
derivative of one of them with respect to the nuclear positions or an external
field. The [tutorial introduction](tutorial/introduction.md) develops this
starting point at greater length. vibe-qc expands the one-electron orbitals in
Gaussian basis functions, so the first decision after the molecule is the
[basis set](user_guide/basis_sets.md); the second is the level of theory, which
is what this page organises.

## Mean-field methods

### Hartree-Fock: RHF, UHF, ROHF

**Theory.** The wavefunction is a single Slater determinant of orthonormal
spin orbitals. Minimising the energy over the orbitals gives the Fock
equations $\hat f \phi_p = \varepsilon_p \phi_p$, where the Fock operator
contains the kinetic and nuclear-attraction terms plus the Coulomb ($\hat J$)
and exchange ($\hat K$) operators built from the occupied orbitals. In a
Gaussian basis they become the Roothaan-Hall generalised eigenproblem

$$
\mathbf F\,\mathbf C = \mathbf S\,\mathbf C\,\boldsymbol\varepsilon,
\qquad
F_{\mu\nu} = h_{\mu\nu} + \sum_{\lambda\sigma} P_{\lambda\sigma}
\left[(\mu\nu|\lambda\sigma) - \tfrac12(\mu\sigma|\lambda\nu)\right],
$$

solved self-consistently because $\mathbf F$ depends on the density
$\mathbf P$ it produces. Each electron feels the others only through their
average field, so HF misses the *correlation energy*, typically about one
percent of the total energy but most of a bond energy.

- **RHF** doubly occupies spatial orbitals; closed shells only.
- **UHF** gives alpha and beta electrons separate orbitals; it describes
  radicals and bond breaking but its determinant is not a spin eigenfunction
  (spin contamination, reported as $\langle \hat S^2\rangle$ in the output).
- **ROHF** keeps one spatial orbital set with integer 2/1/0 occupations and
  couples the shells through the Roothaan or Guest-Saunders coupling
  ({vibeqc-cite}`roothaan_rohf_1960`, {vibeqc-cite}`guest_saunders_rohf_1974`),
  so the state is spin-pure.

**Select it.** `run_job(mol, basis="...", method="rhf" | "uhf" | "rohf")`,
or the drivers `run_rhf`, `run_uhf`, `run_rohf`. Open-shell multiplicity comes
from `Molecule(..., multiplicity=)`.

**Status.** **Production.** Energies agree with PySCF to machine precision;
analytic gradients on all three, analytic Hessians on RHF and UHF, and
finite-difference Hessians everywhere. Read the
[molecular HF tutorial](tutorial/molecular_hf.md), the
[open-shell tutorial](tutorial/open_shell.md), the
[ROHF guide](user_guide/rohf.md) and
[`examples/molecular/input-h2o-rhf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-rhf.py).

### Kohn-Sham density-functional theory: RKS, UKS, ROKS

**Theory.** Hohenberg and Kohn showed that the ground-state energy is a
functional of the electron density alone. Kohn and Sham made it practical by
introducing a fictitious non-interacting system with the same density, whose
orbitals satisfy

$$
\left[-\tfrac12\nabla^2 + v_\text{ext}(\mathbf r) + v_H[\rho](\mathbf r)
      + v_{xc}[\rho](\mathbf r)\right]\phi_i = \varepsilon_i\phi_i,
\qquad
\rho = \sum_i^{occ} |\phi_i|^2 .
$$

Everything unknown is in the exchange-correlation functional $E_{xc}[\rho]$,
whose functional derivative is $v_{xc}$. The approximations form Jacob's
ladder: the **LDA** uses the uniform-electron-gas energy at the local density;
**GGAs** add the density gradient (PBE, BLYP); **meta-GGAs** add the kinetic
energy density $\tau$ (TPSS, SCAN, r²SCAN, M06-L); **hybrids** mix in a
fraction of exact HF exchange (B3LYP, PBE0, PW1PW); **range-separated
hybrids** split the Coulomb operator with an error function and use exact
exchange at long range (ωB97X, ωB97X-D) or short range (HSE06);
**VV10-paired** functionals add a nonlocal correlation kernel for dispersion
(ωB97X-V, ωB97M-V); and **double hybrids** add a scaled MP2-like correlation
term on the KS orbitals (B2PLYP, DSD-PBEP86, revDSD-PBEP86-D4). The
XC integrals are evaluated on an atom-centred Becke grid.

**Select it.** `run_job(mol, basis, method="rks" | "uks" | "roks", functional="pbe0")`.
More than 500 libxc functionals resolve by name; custom weighted sums such as
`"0.2*HF + 0.8*GGA_X_PW91, GGA_C_PW91"` and runtime-defined functionals are
accepted. Note the B3LYP convention: bare `b3lyp` is the VWN5 flavour that
ORCA, TURBOMOLE and CRYSTAL mean; `b3lyp/g` is the Gaussian flavour.

**Status.** **Production** for LDA, GGA, meta-GGA and global hybrids,
including analytic gradients. Range-separated hybrids and VV10 have
production **energies**, but their analytic gradient omits the long-range
exchange and nonlocal-correlation terms, so geometry optimisations on them
fall back to full-energy finite differences and say so in the `.out`
(GitLab #571). ROKS is spin-pure and gradient-by-finite-differences. Read the
[molecular DFT tutorial](tutorial/molecular_dft.md), the
[functionals guide](user_guide/functionals.md), the
[functional comparison](tutorial/functional_comparison.md),
[range-separated hybrids](tutorial/range_separated_wb97x.md),
[VV10 and modern functionals](tutorial/vv10_modern_functionals.md) and
[`examples/molecular/input-h2o-rks-pbe-d3-opt.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-rks-pbe-d3-opt.py).

### SKALA-1.1 neural exchange-correlation functional

**Theory.** SKALA replaces the analytic $E_{xc}$ form with a neural network
trained on high-level reference energies. It consumes the same grid
quantities as a meta-GGA plus nonlocal features of the whole density, and
vibe-qc calls it through an external full-grid provider rather than libxc.

**Select it.** `run_job(..., method="rks", functional="skala-1.1")` with the
`[skala]` extra installed (PyTorch; the MIT-licensed checkpoint is fetched on
first use).

**Status.** **Experimental.** Energies only at a fixed geometry; forces,
Hessians, TDDFT and response fail closed. See [SKALA](user_guide/skala.md) and
the [worked tutorial](tutorial/skala_neural_xc.md).

### Dispersion corrections: D3(BJ), D4, gCP and the 3c composites

**Theory.** Semilocal functionals miss London dispersion because it arises
from correlated fluctuations between non-overlapping densities. Grimme's
schemes add an atom-pairwise term

$$
E_\text{disp} = -\sum_{A<B}\sum_{n=6,8}
   s_n \frac{C_n^{AB}}{R_{AB}^n + f_\text{damp}(R_{AB})^n},
$$

with Becke-Johnson damping in **D3(BJ)** ({vibeqc-cite}`grimme_d3bj_2011`)
and charge-dependent, geometry-dependent $C_6$ coefficients in **D4**
({vibeqc-cite}`caldeweyher_d4_2019`). The **geometric counterpoise (gCP)**
correction removes basis-set superposition error in small bases, and the
**3c composite methods** (HF-3c, PBEh-3c, B97-3c, B3LYP-3c, r²SCAN-3c,
ωB97X-3c, HSE-3c) bundle a tuned small basis with D3 or D4, gCP and a
short-range basis correction behind one keyword.

**Select it.** `run_job(..., dispersion="d3bj" | "d4")`; composites through
`method="hf-3c" | "pbeh-3c" | "b97-3c" | "b3lyp-3c" | "r2scan-3c" | "wb97x-3c" | "hse-3c"`;
`compute_d3bj`, `compute_d4` and `vibeqc.gcp` directly.

**Status.** **Production** for D3(BJ) and for D4 through the `dftd4` package.
The native D4 backend is parity-validated for H, He and B to Ne only and
refuses other elements. Periodic runs support D3(BJ) only. The composites are production for
molecules; their periodic flavours need recalibration before being trusted.
Read the [dispersion tutorial](tutorial/dispersion.md) and the
[composites guide](user_guide/composites.md).

### DFT+U

**Theory.** Semilocal functionals over-delocalise localised d and f
electrons because of self-interaction error. DFT+U adds a Hubbard penalty on
the occupation matrix $n^{I\sigma}$ of a chosen atomic shell in the
rotationally invariant form of Dudarev ({vibeqc-cite}`dudarev_dft_plus_u_1998`),

$$
E_U = \frac{U_\text{eff}}{2}\sum_{I,\sigma}
      \operatorname{Tr}\!\left[n^{I\sigma}(1 - n^{I\sigma})\right],
$$

which pushes occupations toward integers and opens gaps in Mott insulators.

**Select it.** `dft_plus_u=[vibeqc.HubbardSite(atom_index=0, l=2, U_ev=4.0)]`
on `run_periodic_job(..., jk_method="gpw")` or the molecular drivers.

**Status.** **Production** on the GPW route for Gamma RHF/RKS/UHF/UKS and
multi-k RKS. See [DFT+U](user_guide/dft_plus_u.md) and its
[tutorial](tutorial/dft_plus_u.md).

### Density fitting: RI-J, RIJK and RIJCOSX

**Theory.** The four-index electron-repulsion integrals $(\mu\nu|\lambda\sigma)$
are the cost of every Fock build. The resolution of the identity expands each
orbital product in an auxiliary basis $\{P\}$,
$(\mu\nu|\lambda\sigma) \approx \sum_{PQ}(\mu\nu|P)(P|Q)^{-1}(Q|\lambda\sigma)$,
which turns the Coulomb build into three-index work ({vibeqc-cite}`eichkorn_rij_1995`).
**RIJK** fits exchange the same way; **RIJCOSX** keeps RI for Coulomb and
evaluates exchange seminumerically on a grid with the chain-of-spheres
algorithm ({vibeqc-cite}`neese_rijcosx_2009`), which scales better for large
molecules.

**Select it.** `run_job(..., density_fit=True, aux_basis="def2-universal-jkfit")`,
plus `cosx=True` for RIJCOSX. The `JKBuilder` abstraction dispatches to
`FourIndexJKBuilder`, `DFJKBuilder` or `COSXJKBuilder`.

**Status.** **Production.** RIJCOSX energies sit within 0.26 mHa of ORCA 6.1.1
on glycine/def2-TZVP; the RIJCOSX gradient matches finite differences of its
own energy up to the neglected grid-motion band. Read
[density fitting](user_guide/density_fitting.md) and the
[RIJCOSX glycine tutorial](tutorial/rijcosx_glycine.md).

**A second integral engine.** Independently of these routes, `run_job` and
`run_periodic_job` can select the experimental MANTA engine and choose the
Coulomb build (exact, RI, BUPO multipole far field, RI-BUPO) and the
exchange build (exact, RI) separately, with an explicit resource budget and
an executed-provider record in every artifact. Its exact build reproduces
libint to 1e-10 Ha on water; it is slower and refuses most workflows beyond
the SCF and its static properties. See
[integral engines](user_guide/integral_engines.md).

### Effective core potentials

**Theory.** Core electrons of heavy atoms are chemically inert but
expensive and, past the fourth row, relativistic. An ECP replaces them with a
semilocal operator $\hat U_\text{ECP} = U_L(r) + \sum_{l<L}\sum_m |lm\rangle
[U_l(r)-U_L(r)]\langle lm|$ fitted to all-electron relativistic atomic
calculations, so only valence electrons are treated explicitly and scalar
relativistic effects enter implicitly. Integrals come from libecpint.

**Select it.** A bundled basis with an `.ecp` sidecar (def2, cc-pVnZ-PP,
LANL2DZ, pob-TZVP-rev2 for lanthanides) attaches its ECP automatically;
explicit `ecp_centers` and `ecp_library="ecp28mdf"` pick a Stuttgart-Koeln
potential.

**Status.** **Production** for SCF energies and ECP-aware analytic gradients
on RHF, UHF, RKS and UKS. MP2, CC, DLPNO and the double hybrids consume an ECP
reference and OVGF runs on one (without a dedicated validation); CASCI/CASSCF, NEVPT2 and CASPT2 still refuse one. Analytic ECP
Hessians and CPCM-with-ECP forces fail closed. See [ECPs](user_guide/ecp.md)
and the [Pt cluster tutorial](tutorial/pt_cluster_lanl2dz.md).

### Implicit solvation: CPCM, COSMO and COSMO-RS

**Theory.** A continuum of dielectric constant $\varepsilon$ surrounds a
molecular cavity built from atom-centred spheres. The conductor-like models
solve for apparent surface charges $q$ on the cavity such that the potential
vanishes on a perfect conductor, then scale by $f(\varepsilon) =
(\varepsilon-1)/(\varepsilon+x)$ with $x=0$ (CPCM) or $x = 1/2$ (COSMO). The
charges enter the Fock matrix as an extra one-electron potential and the
SCF is macro-iterated to self-consistency between charges and density.
**COSMO-RS** goes further and turns the surface charge density ("sigma
profile") into a statistical-thermodynamic model of the liquid, giving
chemical potentials, activity coefficients and solvation free energies.

**Select it.** `run_job(..., solvent="water")` or a custom `SolventModel`;
`run_job(..., cosmors=vq.CosmoRSOptions(solvent="water"))` for a COSMO-RS
solvation free energy; `vibeqc.solvation.cosmors` on a `ConductorSurface`.

**Status.** CPCM/COSMO energies and analytic gradients are **production** for
RHF, UHF, RKS and UKS; MSINDO has its own COSMO driver. COSMO-RS
(`run_job(cosmors=...)`) and Direct COSMO-RS are **experimental** and emit
`CosmoRSExperimentalWarning`: the published `klamt1998` constants are
transferred, not refitted, to vibe-qc's cavity and methods, and reproduce
eight pure-liquid vapour pressures at BPW91/def2-TZVP with an rms of 1.8
kcal/mol (2.9 on the Direct COSMO-RS density). Solvent with correlated, CAS, semiempirical (except
MSINDO) or MLIP methods is **gated**. Read [solvation](user_guide/solvation.md)
and the [water tutorial](tutorial/solvation_water.md).

### Getting the SCF to converge

**Theory.** The Roothaan-Hall fixed point is reached by iteration, and plain
iteration oscillates or diverges whenever the HOMO-LUMO gap is small.
**DIIS** ({vibeqc-cite}`pulay_diis_1980`) extrapolates the next Fock matrix
from a history of previous ones by minimising the commutator error
$\mathbf e = \mathbf F\mathbf P\mathbf S - \mathbf S\mathbf P\mathbf F$;
**EDIIS** and **ADIIS** ({vibeqc-cite}`kudin_ediis_2002`,
{vibeqc-cite}`hu_yang_adiis_2010`) minimise an energy model instead and are
robust far from convergence, so the default is an EDIIS-then-DIIS hybrid.
**Damping** and **level shifting** ({vibeqc-cite}`saunders_hillier_levelshift_1973`)
slow the density update or push virtual orbitals up; **second-order** methods
(SOSCF, Newton, trust-region augmented Hessian, and the external
OpenTrustRegion optimiser) use the orbital Hessian near convergence;
**stability analysis** ({vibeqc-cite}`seeger_pople_stability_1977`) checks
that the converged determinant is a minimum, not a saddle. The
**initial guess** (SAD, PATOM, SAP, core Hamiltonian, or a READ restart)
decides which stationary state the iteration lands on.

**Select it.** The `*_options` objects (`RHFOptions`, `RKSOptions`, ...) carry
`scf_accelerator`, `damping`, `level_shift`, `soscf_threshold`, `newton_threshold`,
`initial_guess` and friends; the [keyword index](user_guide/keyword_index.md)
lists defaults.

**Status.** **Production** for the whole DIIS family, damping, level shifts
and the second-order finalisers on molecules; periodic coverage is route by
route as [SCF convergence](user_guide/scf_convergence.md) documents.
`run_job(solver=...)` accepts `dense`, `davidson` and `lobpcg`; the
Jacobi-Davidson and GPLHR eigensolvers, reachable through
`vibeqc.solvers.eigensolver.solve_eigenproblem(solver="jd" | "gplhr")`, are
**experimental**.
Read the [EDIIS/DIIS tutorial](tutorial/ediis_diis_hybrid.md),
[second-order SCF](tutorial/second_order_scf.md),
[OpenTrustRegion](tutorial/opentrustregion.md) and the
[initial guess walkthrough](tutorial/initial_guess_walkthrough.md).

## Single-reference correlation

Everything in this section starts from a converged Hartree-Fock (or, for the
double hybrids, Kohn-Sham) determinant and recovers the correlation energy
$E_\text{corr} = E_\text{exact} - E_\text{HF}$ that the mean field leaves out.
The methods form a hierarchy of cost and accuracy; the
[correlated-energies learning path](tutorial/index.md#choose-a-learning-path)
walks it in order.

### Moller-Plesset perturbation theory: MP2, RI-MP2, SCS/SOS-MP2, UMP2

**Theory.** Treat the difference between the true electron repulsion and
the HF mean field as a perturbation ({vibeqc-cite}`moller_plesset_1934`).
First order returns the HF energy; second order gives

$$
E^{(2)} = \sum_{ij}^{occ}\sum_{ab}^{virt}
  \frac{(ia|jb)\left[2(ia|jb) - (ib|ja)\right]}
       {\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b},
$$

which scales as $O(N^5)$ and captures most of the dispersion and pair
correlation HF misses. **RI-MP2** fits the $(ia|jb)$ integrals in an
auxiliary basis ({vibeqc-cite}`feyereisen_rimp2_1993`) with negligible error;
**SCS-** and **SOS-MP2** rescale the same-spin and opposite-spin pair
energies ({vibeqc-cite}`grimme_scs_mp2_2003`, {vibeqc-cite}`jung_sos_mp2_2004`);
**UMP2** and **ROHF-MP2** extend the formula to open shells.

**Select it.** `run_job(..., method="mp2" | "scs-mp2" | "sos-mp2")`; add
`density_fit=True` for RI-MP2. An open-shell molecule dispatches to UMP2 on
the UHF reference, and `run_rohf_mp2` gives the semicanonical ROHF-MP2
energy. The drivers `run_mp2` and `run_ump2` take a converged SCF result. A
frozen core follows the ORCA convention by default.

**Status.** **Production.** Validated to 1e-9 Ha against PySCF. The dense
routes are accuracy references at small-molecule size rather than
production-size solvers; the memory envelope is documented in the
[MP2 guide](user_guide/mp2_and_double_hybrids.md). Read the
[RI-MP2 auxiliary-basis tutorial](tutorial/ri_mp2_aux_verification.md),
[natural orbitals](tutorial/natural_orbitals.md) and
[`examples/molecular/input-h2o-scs-mp2.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-scs-mp2.py).

### Double-hybrid functionals: B2PLYP, DSD-PBEP86, revDSD-PBEP86-D4

**Theory.** A double hybrid ({vibeqc-cite}`grimme_b2plyp_2006`) runs a
hybrid Kohn-Sham SCF with a large exact-exchange fraction, then adds a scaled
MP2 correlation term computed on the KS orbitals,
$E = E_\text{KS}^\text{hybrid} + c_2 E^{(2)}_\text{KS}$. The DSD variants
scale same- and opposite-spin contributions separately and pair the result
with D3 or D4 dispersion.

**Select it.** `run_job(..., method="b2plyp" | "dsd-pbep86" | "revdsd-pbep86" | "pwpb95")`.
A double hybrid names a complete method, so it is a `method=`, not a
`functional=`; the spelling `method="rks", functional="b2plyp"` is refused
because it would run only the SCF half.

**Status.** **Production**, validated to 1e-7 Ha against PySCF. Read the
[B2PLYP tutorial](tutorial/double_hybrid_b2plyp.md).

### Coupled cluster: CCSD, CCSD(T) and the open-shell variants

**Theory.** Coupled-cluster theory ({vibeqc-cite}`cizek_cc_1966`) writes the
wavefunction as an exponential of excitation operators acting on the
reference, $|\Psi\rangle = e^{\hat T}|\Phi_0\rangle$ with
$\hat T = \hat T_1 + \hat T_2 + \dots$. Truncating at doubles gives **CCSD**
({vibeqc-cite}`purvis_bartlett_ccsd_1982`), whose amplitude equations are
solved iteratively at $O(N^6)$; the exponential form makes the energy size
extensive, unlike truncated CI. **CCSD(T)** adds the connected triples by
non-iterative perturbation theory ({vibeqc-cite}`raghavachari_ccsdt_1989`) at
$O(N^7)$ and is the accepted "gold standard" for single-reference molecules.
**A-CCSD(T)** uses the Lambda-based asymmetric triples correction, and
**frozen natural orbitals (FNO)** truncate the virtual space with MP2 natural
orbitals to make the triples cheaper. Open-shell references run through
**UCCSD(T)** on UHF orbitals and a spin-orbital kernel on ROHF orbitals.

**Select it.** `run_job(..., method="ccsd" | "ccsd(t)")`, or `run_ccsd` with
`CCSDOptions` (density fitting on by default; `density_fit=False` for the
conventional cross-code route; `triples="A-CCSD(T)"`; FNO thresholds).

**Status.** **Production** as a dense small-molecule accuracy reference,
validated to microhartree against PySCF and cross-checked against ORCA 6.1.
The dense CC iteration and the FNO setup remain production-size memory gates.
Read the [CCSD(T) tutorial](tutorial/coupled_cluster_ccsd_t.md) and the
[coupled-cluster guide](user_guide/ccsd.md).

### The wider coupled-cluster and coupled-pair family: CC2, CC3, CCSDT, QCISD(T), CEPA(n)

**Theory.** **CC2** ({vibeqc-cite}`christiansen_cc2_1995`) keeps the singles
equations exact and approximates the doubles to first order, an $O(N^5)$
model mainly used for excitation energies and as a step between MP2 and
CCSD. **CC3** ({vibeqc-cite}`koch_cc3_1997`) treats triples iteratively but
approximately, and full **CCSDT** ({vibeqc-cite}`noga_bartlett_ccsdt_1987`)
solves the triples exactly at $O(N^8)$. **QCISD** and **QCISD(T)**
({vibeqc-cite}`pople_qcisd_1987`) are size-consistent quadratic CI
approximations to CCSD; **CCD**, **LCCD**, **LCCSD** and the **CEPA(n)**
family ({vibeqc-cite}`meyer_cepa_1973`) drop or linearise terms of the CCSD
equations, with CEPA adding pair-specific shifts that approximate the
exclusion-principle-violating corrections.

**Select it.** `run_job(..., method="cc2" | "cc3" | "ccsdt" | "qcisd" | "qcisd(t)" | "ccd" | "lccd" | "lccsd" | "cepa(0)" ... "cepa(3)" | "bccd" | "bccd(t)")`;
the last two are Brueckner coupled cluster ({vibeqc-cite}`handy_bccd_1989`),
which rotates the orbitals until the singles amplitudes vanish. The same
variants are reachable through `method="ci", citype=...`.

**Status.** **Production** at benchmark scale, closed-shell energies only.
CC2 and QCISD(T) are pinned against ORCA; CEPA(n) to sub-microhartree against
ORCA RI-CEPA. CC3 and CCSDT use a dense determinant-space representation that
grows combinatorially, so they are references for small molecules, not
production solvers. All are documented in the
[coupled-cluster guide](user_guide/ccsd.md).

### Local correlation: DLPNO-MP2, DLPNO-CCSD(T), DLPNO-(T1) and the open-shell solvers

**Theory.** Canonical correlation methods scale steeply because canonical
orbitals are delocalised. The domain-based local pair natural orbital scheme
({vibeqc-cite}`riplinger_dlpno_2013`, {vibeqc-cite}`pinski_dlpno_mp2_2015`)
localises the occupied orbitals (Foster-Boys), assigns each occupied pair a
domain of projected atomic orbitals, and compresses each pair's virtual
space into **pair natural orbitals**, the eigenvectors of the pair's
approximate MP2 density that carry occupation above a threshold
($T_\text{CutPNO}$). Pairs whose estimated correlation energy falls below
$T_\text{CutPairs}$ are treated at a cheaper level. The result is
near-linear scaling with a controllable truncation error; at zero thresholds
the equations reduce exactly to the canonical ones, which is the test the
implementation pins. The **(T1)** correction restores the off-diagonal
Fock coupling in the triples ({vibeqc-cite}`guo_dlpno_t1_2018`), and the
open-shell solvers keep independent alpha and beta PNO spaces
({vibeqc-cite}`saitow_openshell_dlpno_2017`).

**Select it.** `run_job(..., method="dlpno-mp2" | "dlpno-ccsd" | "dlpno-ccsd(t)")`
with `dlpno_options=DLPNOMP2Options(...)` (from `vibeqc.dlpno`), `dlpno_ccsd_options=` or
`dlpno_thresholds=` for thresholds; open-shell molecules
route to the spin-orbital DLPNO-UMP2 and DLPNO-UCCSD(T) solvers
automatically.

**Status.** **Production** for the molecular closed- and open-shell
solvers. At the shipped defaults (chemical frozen core, NormalPNO,
cc-pVDZ) on 17 closed-shell molecules from water to C₁₀H₂₂ the mean absolute
error against canonical DF-CCSD(T) is 0.26 kcal/mol for CCSD and 0.38 for
CCSD(T), and TightPNO brings it to 0.14 and 0.19; the per-molecule table is
in the [DLPNO guide](user_guide/dlpno_mp2.md#accuracy). State the thresholds
when you publish. The dense $O(N^6)$ open-shell pilot is opt-in only.

**On crystals.** `run_periodic_job(method="dlpno-ccsd(t)")` correlates an
RHF or UHF reference on the GDF route at the Gamma point or on a
Gamma-centred mesh folded to the real BvK supercell, with the molecular
engine and the reference's own retained fit as the correlation factors. At
full domains it reproduces PySCF KRCCSD(T) to the fit tolerance; its
accuracy at the shipped thresholds on real crystals is not measured yet, so
it is **preview**. See
[local correlation on crystals](user_guide/periodic_local_correlation.md)
and the [crystal tutorial](tutorial/periodic_dlpno_ccsd_t.md).

Read the
[DLPNO tutorial](tutorial/dlpno_local_correlation.md), the
[DLPNO-MP2 guide](user_guide/dlpno_mp2.md) and
[`examples/mlip/07_dlpno_ccsdt_on_mace_geometry.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/mlip/07_dlpno_ccsdt_on_mace_geometry.py).

### Configuration interaction with singles and doubles: CISD

**Theory.** Diagonalise the Hamiltonian in the space of the reference plus
all single and double excitations ({vibeqc-cite}`pople_cisd_1977`). CISD is
variational but not size extensive: the missing quadruples make the recovered
correlation energy per molecule shrink as the system grows, which is why the
coupled-pair and coupled-cluster methods above are preferred for energies.
It remains a clean benchmark and the natural entry point to multi-root
excited states.

**Select it.** `run_job(..., method="cisd")` with `CISDOptions` for roots and
determinant-space guards.

**Status.** **Production** for small molecules; see the
[post-HF table in the feature matrix](features.md#post-hf-correlation-methods).

## Multireference and exact methods

When one determinant is qualitatively wrong, at a stretched bond, a
diradical, a transition-metal d shell or an avoided crossing, the methods
above break down no matter how many excitations they include. The remedy is
to treat a small set of strongly correlated orbitals exactly and correct the
rest perturbatively. The [CASSCF tutorial](tutorial/casscf_multireference.md)
derives this section in detail.

### CASCI and CASSCF

**Theory.** Choose an **active space** of $n$ electrons in $m$ orbitals,
written CAS($n$,$m$). **CASCI** performs a full CI within that window on
fixed (usually HF) orbitals; **CASSCF** additionally optimises the orbitals,
so the inactive, active and virtual spaces rotate to minimise the CI energy
({vibeqc-cite}`roos_taylor_siegbahn_casci_1980`). State averaging optimises
one orbital set for several roots at once, which is what excited-state and
conical-intersection work needs. The inactive core enters through the
frozen-core dressing of the one-electron integrals,
$\tilde h_{pq} = h_{pq} + \sum_c [2(pq|cc) - (pc|cq)]$.

**Select it.** `run_job(..., method="casci" | "casscf", active_space=(n_orbitals, n_electrons), casscf_options=CASSCFOptions(nroots=N))`.

**Status.** **Production.** The state-specific closed-shell CASSCF analytic
nuclear gradient is finite-difference tight to about 1e-7; state-averaged
and open-shell CASSCF gradients fall back to finite differences. ECP
references are refused.

### CASPT2, NEVPT2 and MRCI

**Theory.** Dynamic correlation outside the active space is added on top of
the CAS reference. **CASPT2** ({vibeqc-cite}`andersson_caspt2_1990`,
{vibeqc-cite}`andersson_malmqvist_roos_caspt2_1992`) is second-order
perturbation theory with a one-electron zeroth-order Hamiltonian, internally
contracted here, with the optional IPEA shift; **NEVPT2**
({vibeqc-cite}`angeli_nevpt2_2001`) uses the Dyall Hamiltonian, which makes it
intruder-free and size consistent without shifts. **MRCI**
({vibeqc-cite}`werner_knowles_mrci_1988`) diagonalises singles and doubles out
of every reference configuration and is the most accurate, and most
expensive, of the three; the Davidson correction estimates the missing
size-extensivity error.

**Select it.** `run_job(..., method="caspt2" | "nevpt2" | "mrci", active_space=...)`,
optionally with `casscf_options` to use CASSCF rather than CASCI orbitals as
the reference.

**Status.** **Production** energies; CASPT2 was un-gated at v0.12.0 with a
large-basis direct active-CI path. Analytic PT2 gradients exist only in a
narrow envelope (state-specific, closed-shell, unshifted, explicit engine)
and every other case uses full-energy finite differences, so treat
multireference geometry optimisation as **preview**. Read the
[CASSCF tutorial](tutorial/casscf_multireference.md) and the
[non-HF solver guide](user_guide/non_hf_solvers.md).

### Exact and near-exact determinant solvers: FCI, selected CI, DMRG, v2RDM, transcorrelated CI

**Theory.** **Full CI** diagonalises the Hamiltonian in the complete
determinant space of the active window and is exact for that basis; its
cost grows factorially, so it is a reference for small systems. **Selected
CI** in the CIPSI form ({vibeqc-cite}`huron_malrieu_cipsi_1973`) grows the
determinant space iteratively, keeping the determinants whose second-order
perturbative contribution is largest and estimating the remainder with a
PT2 correction. **DMRG** represents the wavefunction as a matrix product
state and sweeps through the orbitals optimising two sites at a time; it
excels for one-dimensional-like orbital orderings. The **variational 2-RDM**
method optimises the two-particle reduced density matrix directly under
N-representability conditions. **Transcorrelated CI** similarity-transforms
the Hamiltonian with a Jastrow or Gaussian correlator so the cusp is
handled analytically before a selected CI finishes the job.

**Select it.** `run_job(..., method="fci" | "selected_ci" | "dmrg" | "v2rdm" | "transcorrelated_ci", active_space=...)`,
or the `vibeqc.solvers` functions on an abstract MO-basis `Hamiltonian`.

**Status.** FCI, selected CI and v2RDM are **production** as
small-active-space references (FCI parity tests on He and OH; the v2RDM
solver has no Q/G semidefinite feedback yet). DMRG is **experimental**: the
source calls it a toy implementation, it is cross-checked against FCI, and it
refuses more than 6 spatial orbitals (12 spin orbitals). Transcorrelated CI is
a **preview** with a research-grade correlator. Geometry optimisation on any
of these needs `optimizer_backend="native"` or `"brent"` and uses numerical
gradients (#357).
Read the [non-HF solvers tutorial](tutorial/non_hf_solvers.md) and
[`examples/molecular/input-h2-fci.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2-fci.py).

## Excited states and response properties

### Linear-response TDDFT: Casida, Tamm-Dancoff, CIS

**Theory.** A weak oscillating field perturbs the ground-state density;
the poles of the linear density response are the excitation energies.
In the Casida formulation this becomes the non-Hermitian eigenproblem

$$
\begin{pmatrix} \mathbf A & \mathbf B \\ -\mathbf B & -\mathbf A \end{pmatrix}
\begin{pmatrix} \mathbf X \\ \mathbf Y \end{pmatrix}
= \omega \begin{pmatrix} \mathbf X \\ \mathbf Y \end{pmatrix},
\qquad
A_{ia,jb} = \delta_{ij}\delta_{ab}(\varepsilon_a-\varepsilon_i)
          + (ia|jb) + (ia|f_{xc}|jb) - c_x (ij|ab),
$$

where $f_{xc}$ is the adiabatic exchange-correlation kernel and $c_x$ the
exact-exchange fraction. Setting $\mathbf B = 0$ gives the **Tamm-Dancoff
approximation**, which on a HF reference is **CIS**. Oscillator strengths
follow from the transition dipoles, and **natural transition orbitals**
({vibeqc-cite}`martin_nto_2003`) compress each excitation into a dominant
hole-particle pair.

**Select it.** `run_job(..., tddft=True, tddft_type="tda" | "casida", tddft_n_states=N, nto=True, tddft_spectrum=True)`,
or `run_tddft_tda` / `run_tddft_casida`; the `vibeqc tddft` CLI.

**Status.** Excitation energies are **production** for RHF, RKS, UHF and UKS with
LDA, GGA and global hybrid kernels (they match PySCF to 1e-4 eV); ROHF
supports TDA only; range-separated kernels are **gated**. Two known defects
matter: the oscillator strengths are currently too large (a factor of about 2
on closed-shell and 4 on unrestricted references, from the transition-dipole
normalisation), and `run_job(tddft_gradient=True)` differentiates a geometry
stretched by the bohr-to-angstrom factor. Excited-state gradients are finite
differences on the closed-shell RHF/CIS surface only (**preview**). Periodic
Gamma-point TDDFT is reachable through `run_periodic_job(tddft=True)`; the
wrapper currently fails on GDF Gamma results until the leading k axis is
sliced off. See the [TDDFT guide](user_guide/tddft.md) for the details. Read the
[TDDFT tutorial](tutorial/excited_states_tddft.md).

### Green's-function ionisation energies: GF2 and OVGF

**Theory.** The one-particle Green's function has poles at the exact
ionisation potentials and electron affinities. Solving the diagonal
quasiparticle equation $\omega = \varepsilon_p + \Sigma_{pp}(\omega)$ with
the second-order self-energy gives the **GF2** ionisation potentials, Koopmans
energies corrected for relaxation and correlation; the **outer valence Green's
function** ({vibeqc-cite}`cederbaum_ovgf_1975`, {vibeqc-cite}`vonniessen_ovgf_1984`)
adds third-order terms with a renormalisation factor. In vibe-qc `result.ovgf`
holds the diagonal second-order solution and pole strength for each orbital;
the renormalised third-order value is printed for the HOMO when the system is
small enough (at most 60 spin orbitals), and the literal OVGF A/B/C variants
are not implemented.

**Select it.** `run_job(..., method="ovgf")` and read `result.ovgf`; MSINDO
has `msindo_ovgf`.

**Status.** **Production** for HF references including the open-shell
propagator; renormalised GF2 prints automatically for small systems.

### Static polarizability and analytic Hessians: coupled-perturbed SCF

**Theory.** Second derivatives of the energy need the first-order response
of the orbitals. Differentiating the SCF conditions with respect to a
perturbation gives the coupled-perturbed Hartree-Fock or Kohn-Sham equations
({vibeqc-cite}`pople_derivatives_1979`), a linear system in the
occupied-virtual rotation parameters. With an electric-field perturbation the
solution gives the **static dipole polarizability**
$\alpha_{ij} = -\partial^2 E/\partial F_i\partial F_j$; with nuclear
displacements it gives the **analytic Hessian**, hence harmonic frequencies
and IR intensities without finite differences.

**Select it.** `dipole_polarizability_rhf` and
`vibeqc.cphf.dynamic_polarizability_rhf`; the analytic Hessians are the
library functions
`compute_hessian_{rhf,uhf,rks,uks}_analytic`. Note that `run_job(..., hessian=True)`
uses `compute_hessian_fd`, finite differences of the analytic gradient, on
every route; `run_periodic_job(hessian=True)` computes the molecular
finite-difference Hessian of the isolated unit-cell molecule.

**Status.** **Production** for the RHF analytic Hessian, which matches PySCF
to 2.5e-9 Ha/bohr², and for the closed-shell polarizability. The UHF, RKS and
UKS analytic Hessians use the LDA kernel for the response, so GGA and hybrid
functionals differ from finite differences by up to about 12 cm⁻¹; treat them
as **preview** beyond LDA. `run_job(hessian=True)` ignores `solvent=` and
`dispersion=` without saying so. Read the [vibrational frequencies tutorial](tutorial/vibrational_frequencies.md)
and the [properties guide](user_guide/properties.md).

## Semiempirical methods

Semiempirical Hamiltonians keep the self-consistent-field structure but
replace most integrals with parametrised expressions fitted to experiment or
to higher-level theory. They are one to three orders of magnitude faster than
DFT and are meant for screening, preoptimisation, dynamics and very large
systems. Their accuracy depends entirely on the parametrisation, which is why
the status column below is more varied than anywhere else on this page. The
[semiempirical guide](user_guide/semiempirical.md) has the per-route status
table that the tests pin.

### MSINDO

**Theory.** MSINDO ({vibeqc-cite}`ahlswede_jug_msindo_1_1999`) is an INDO-type
method with an orthogonalisation correction: overlap between atoms is kept in
the one-electron terms, two-centre repulsion integrals are evaluated
analytically over Slater orbitals, and pseudopotential-like terms model the
core. It was built for solids as well as molecules, which is why the
semiempirical cyclic cluster model was developed inside it. vibe-qc's
implementation is independent, covers H to Xe, and adds an NDDO mode for the
main-group elements.

**Select it.** `run_job(..., method="msindo")`, with `solvent=` for its
own COSMO driver and the dynamics, metadynamics and MECI workflows below.

**Status.** **Production within the documented scope**: INDO energies for
RHF and UHF, closed-shell analytic gradients, NDDO for H, Li to F and Na to
Cl. Read the [MSINDO guide](user_guide/msindo.md) and
[tutorial](tutorial/msindo.md).

### Density-functional tight binding: DFTB0 and SCC-DFTB

**Theory.** Expand the DFT energy around a reference density of superposed
atoms ({vibeqc-cite}`porezag_dftb0_1995`). The zeroth-order term gives a
non-self-consistent tight-binding Hamiltonian with precomputed two-centre
matrix elements plus a repulsive pair potential (**DFTB0**). The second-order
term adds Mulliken-charge fluctuations interacting through a distance-dependent
gamma function (**SCC-DFTB**, {vibeqc-cite}`elstner_scc_dftb_1998`), solved
self-consistently and capturing charge transfer.

**Select it.** `run_job(..., method="dftb0" | "scc_dftb")`; periodic Gamma
and full-k routes through `run_periodic_job`.

**Status.** **Preview**, for screening and preoptimisation only. The
parameters are in-house, there is no DFTB+ (mio-1-1) parity, and absolute
energies are not claimable while the repulsive is a placeholder (issue #306);
fixed-geometry differences remain valid. Read the
[DFTB tutorial](tutorial/semiempirical_dftb.md).

### GFN2-xTB

**Theory.** The extended tight-binding family ({vibeqc-cite}`bannwarth_gfn2_2019`)
adds to SCC-DFTB an anisotropic electrostatic term (atomic multipoles),
shell-resolved charges, a third-order on-site term and a built-in D4
dispersion, parametrised across the periodic table for geometries,
frequencies and non-covalent interactions.

**Select it.** `run_job(..., method="gfn2_xtb")`; parameters are fetched on
demand.

**Status.** **Experimental.** Molecular energies, analytic gradients and
the Gamma-periodic derivatives are implemented and finite-difference
validated, but external `xtb` parity and faithful periodic multipole
electrostatics remain production gates; the route is closed-shell only. Read
the [GFN2-xTB workshop](tutorial/gfn2_xtb_workshop.md) and
[PM6 and GFN2](tutorial/pm6_and_gfn2.md).

### NDDO methods: PM6, OM1, OM2, OM3 and the PM7 gate

**Theory.** Neglect of diatomic differential overlap keeps only one- and
two-centre two-electron integrals, evaluated from multipole expansions with
element-pair parameters. **PM6** ({vibeqc-cite}`stewart_pm6_2007`) is the
widely used MOPAC parametrisation; the orthogonalisation-corrected **OMx**
methods ({vibeqc-cite}`weber_thiel_om2_2000`, {vibeqc-cite}`scholten_om3_2003`)
add explicit orthogonalisation and effective core potential terms that
improve conformational energies and excited states.

**Select it.** `run_job(..., method="pm6" | "om1" | "om2" | "om3")`. PM7 is
recognised and refuses.

**Status.** PM6 is **experimental** (heat-of-formation parity with MOPAC is
open; periodic PM6 is experimental). OM1, OM2 and OM3 are **experimental and a
retirement target**: still executable as molecular prescreening, but scheduled
to leave the public portfolio. **PM7 is gated**: it has never been
executable in any release because the feathered electrostatics are not
implemented.

### SECCM: the semiempirical cyclic cluster model

**Theory.** Instead of Bloch functions and a k-mesh, the cyclic cluster
model ({vibeqc-cite}`bredow_geudtner_jug_ccm_2001`) takes one supercell,
imposes cyclic (Born-von Karman) boundary conditions on it directly in real
space, and weights every interaction by whether it lies inside the
Wigner-Seitz cell of the cluster. The result is a finite, molecule-like
Hamiltonian that has the full translational symmetry of the crystal and no
k-points at all, which makes defects, surfaces and localised states natural.
The long-range Coulomb field is restored through an Ewald-summed Madelung
embedding.

**Select it.** `run_job(..., method="seccm")` (normalised to `"ccm"`) with
`ccm_options` carrying the supercell lattice vectors; the semiempirical
package also exposes per-family drivers such as
`vibeqc.semiempirical.seccm.run_pm6_seccm` for MSINDO, DFTB0, SCC-DFTB,
GFN2-xTB, PM6 and OMx.

**Status.** **Experimental.** The MSINDO route is the most mature (closed
shell, 1-D to 3-D, native C++ for H to Kr); the DFTB, GFN2 and PM6 routes are
gated experimental with energy-only or acknowledgement-gated envelopes. The
Madelung embedding is opt-in and 1-D/2-D only. Read the
[cyclic cluster model guide](user_guide/cyclic_cluster_model.md) and the
[CCM tutorial](tutorial/cyclic_cluster_model.md).

## Machine-learned interatomic potentials

### MACE

**Theory.** MACE ({vibeqc-cite}`batatia_mace_2022`) is an equivariant
message-passing neural network that predicts energies and forces from atomic
environments, trained on large DFT datasets (MACE-MP for materials,
MACE-OFF for organic molecules). It is not an electronic-structure method:
there are no orbitals, no density and no wavefunction, and its absolute energy
scale is that of its training functional.

**Select it.** `run_job(..., method="mace")` for molecules; `run_periodic_mace`,
`PeriodicMACEEvaluator`, `optimize_periodic_mace_positions` and
`optimize_periodic_mace_cell` for periodic systems. Needs the `[mace]` extra
on Python 3.13 or earlier.

**Status.** The interface is **production**; the model is an **external
pre-trained potential**, attributed as such in the citations, and its numbers
are the model's, not vibe-qc's. Use it for geometries, screening and
pre-relaxation, then hand the structure to an electronic-structure method.
Read the [MLIP guide](user_guide/mlip.md), the
[MACE tutorial](tutorial/mace_mlip.md) and
[from MACE geometry to electronic structure](tutorial/mace_geometry_to_electronic_structure.md).

## Periodic systems

### Bloch's theorem and k-point sampling

**Theory.** In a crystal the one-electron Hamiltonian commutes with lattice
translations, so its eigenfunctions can be chosen as Bloch functions
$\psi_{n\mathbf k}(\mathbf r + \mathbf T) = e^{i\mathbf k\cdot\mathbf T}
\psi_{n\mathbf k}(\mathbf r)$. Expanding them in Bloch sums of the Gaussian
atomic orbitals, $\phi_\mu^{\mathbf k} = \sum_{\mathbf T}
e^{i\mathbf k\cdot\mathbf T}\chi_\mu(\mathbf r - \mathbf T)$, turns the
crystal problem into an independent Roothaan-Hall problem at every crystal
momentum $\mathbf k$ in the first Brillouin zone,

$$
\mathbf F(\mathbf k)\,\mathbf C(\mathbf k) = \mathbf S(\mathbf k)\,\mathbf C(\mathbf k)\,\boldsymbol\varepsilon(\mathbf k),
$$

coupled only through the density, which is a Brillouin-zone integral
approximated by a Monkhorst-Pack mesh ({vibeqc-cite}`monkhorst_pack_1976`)
reduced by symmetry to the irreducible wedge. Insulators converge quickly
with $\mathbf k$; metals need dense meshes plus a smearing of the occupation
numbers (Fermi-Dirac, Methfessel-Paxton or Marzari-Vanderbilt) so that the
Fermi surface is sampled smoothly, and a density mixer (Anderson, Broyden,
Kerker) to damp charge sloshing.

**Select it.** `PeriodicSystem(lattice, atoms, dim=1|2|3)`, `KPoints.monkhorst_pack`,
`run_periodic_job(system, basis, method=..., kpoints=..., smearing_temperature=..., density_mixer=...)`.

**Status.** **Production.** Read [k-points, the Brillouin zone and Bloch functions](tutorial/kpoints_brillouin_bloch.md),
the [k-point guide](user_guide/k_points.md), [smearing](user_guide/smearing.md)
and the [metallic BZ integration guide](user_guide/metallic_bz_integration.md).

### The periodic Coulomb problem and Ewald summation

**Theory.** In an infinite lattice the bare Coulomb sums over nuclei,
electrons and their cross terms are each conditionally convergent; only the
total for a neutral cell is defined. Ewald's method
({vibeqc-cite}`ewald_lattice_sum_1921`) splits $1/r$ into a short-range part
summed in real space and a smooth long-range part summed in reciprocal space
with a Gaussian screening parameter $\alpha$ that cancels between the two.
vibe-qc uses one shared $\alpha$ gauge for the nuclear, one-electron and
two-electron pieces, which is what makes the total energy independent of the
gauge; the two-dimensional analogue ({vibeqc-cite}`parry_2d_ewald_1975`) treats
slabs without a vacuum gap. The Madelung constants of NaCl, CsCl and ZnS are
reproduced to 1e-8 as the regression check.

**Select it.** Automatic on every periodic route; `ewald_point_charge_energy`
exposes the classical sum.

**Status.** **Production.** Read the [Ewald guide](user_guide/ewald.md) and
the [Madelung tutorial](tutorial/madelung_with_ewald.md).

### GDF: Gaussian density fitting, with RSGDF and MDF

**Theory.** The periodic four-index integrals are fitted in an auxiliary
Gaussian basis exactly as in the molecular RI, but every fitting quantity is
Bloch-summed and the $G = 0$ divergence of the Coulomb kernel is handled
separately ({vibeqc-cite}`sun_berkelbach_gdf_2017`). **Range-separated GDF**
({vibeqc-cite}`ye_berkelbach_rsgdf_2021`) evaluates the short-range part of
the fit analytically and the long-range part on a plane-wave grid, which
removes the lattice-sum truncation; **mixed density fitting** adds plane
waves to the auxiliary set to close the all-electron accuracy floor in dense
crystals. Exact exchange at $\mathbf k$ meshes uses the Ewald probe-charge
correction for the $\mathbf q = 0$ singularity
({vibeqc-cite}`gygi_baldereschi_exx_1986`).

**Select it.** `run_periodic_job(..., jk_method="gdf", gdf_method="rsgdf" | "compcell" | "mdf", kpoints=...)`;
drivers `run_rhf_periodic_gamma_gdf`, `run_krhf_periodic_gdf`, `run_krks_periodic_gdf`,
`run_kuhf_periodic_gdf`, `run_kuks_periodic_gdf`, `run_krohf_periodic_gdf`.

**Status.** **Production** for Gamma and multi-k RHF, RKS, UHF, UKS in 3-D
and Gamma-only in 1-D (a 1-D cell with a k mesh currently fails closed for
want of a consistent Coulomb gauge), with microhartree parity against PySCF
on LiH and MgO, and for the
bounded closed-shell 2-D slab route with analytic gradients. Periodic ROHF on
GDF is **preview** (Gamma and full multi-k, no gradients or smearing). Read
[periodic methods](user_guide/periodic_methods.md), [multi-k SCF](user_guide/multi_k_scf.md),
[density fitting](user_guide/density_fitting.md) and the
[LiH multi-k tutorial](tutorial/lih_multi_k.md).

### GPW and GAPW: the plane-wave route

**Theory.** The Gaussian and plane-wave method ({vibeqc-cite}`vandevondele_ot_2003`
lineage, as in CP2K) keeps Gaussian orbitals but represents the density on a
regular grid, so the Hartree potential comes from one FFT-based Poisson solve
and the XC potential from grid quadrature; only the smooth valence density
fits on an affordable grid, which is why GPW pairs naturally with
pseudopotentials. **GAPW** ({vibeqc-cite}`blochl_paw_1994` lineage) restores
all-electron accuracy by augmenting the grid density with one-centre
atomic expansions near each nucleus, so the hard core density never touches
the plane-wave grid.

**Select it.** `run_periodic_job(..., jk_method="gpw" | "gapw")`; GAPW needs
`gapw_molecular_limit=True` for RHF/UHF.

**Status.** GPW is the serial **production** plane-wave route within its
shipped envelope (Gamma RHF, UHF, RKS, UKS; multi-k pure-DFT RKS, ROKS, UKS;
Gamma analytic gradients; DFT+U; D3-BJ). The `.out` route description still
carries an experimental tag while the surrounding GPW/GAPW family is
completed, and the grid and J-builder infrastructure emit
`GAPWExperimentalWarning` once per build even through `run_periodic_job`
(the runner silences only the driver-level warning); read it as a scope
reminder. GAPW itself is **experimental**: not the automatic default, with molecular-limit
Gamma HF and block-construction DFT shipped and compact-crystal augmentation
validation pending. GPW ROHF/ROKS are a maintained **preview**. The MPI grid
overlay is **experimental**. Read [GPW and GAPW](user_guide/gapw.md), the
[GPW tutorial](tutorial/periodic_gpw.md) and the [GAPW tutorial](tutorial/periodic_gapw.md).

### BIPOLE: the exact Ewald J split

**Theory.** BIPOLE ({vibeqc-cite}`saunders_bipole_1992`) is the CRYSTAL-style
treatment of the periodic Coulomb term: the Hartree interaction of each
atomic-orbital product with the infinite lattice is split into an exact
near-field sum over overlapping distributions and a far field represented
by multipoles and summed with Ewald's method. vibe-qc's implementation
evaluates the far field exactly rather than through the multipole
approximation, so the total is gauge-independent, and applies the same split
to nuclei and electrons.

**Select it.** `run_periodic_job(..., jk_method="bipole")`; drivers
`run_pbc_bipole_rhf`, `_uhf`, `_rks`, `_uks`.

**Status.** **Production** energies for Gamma and multi-k RHF, UHF, RKS and
UKS in 3-D, with finite-difference forces and fixed-cell relaxation. The
analytic gradients are a **preview**, and the quartet multipole far-field
prototype is **gated** (every driver refuses `use_multipole_far_field=True`).
Read the [BIPOLE guide](user_guide/bipole.md) and the
[physical-domain tutorial](tutorial/bipole_physical_domain.md).

### Periodic RIJCOSX and the 2-D slab route

**Theory.** Periodic RIJCOSX takes the RI-J Coulomb from the GDF loop and
builds exact exchange seminumerically on a periodic Becke grid, extending the
molecular chain-of-spheres idea to Bloch orbitals. The 2-D slab route uses
the rigorous vacuum-free two-dimensional Ewald sum ({vibeqc-cite}`parry_2d_ewald_1975`,
{vibeqc-cite}`de_leeuw_perram_2d_ewald_1979`), so the third lattice vector is
pure bookkeeping and the energy is independent of it.

**Select it.** `run_periodic_job(..., jk_method="rijcosx")`; slabs through
`vibeqc.slab_2d(a1, a2, atoms)` with `jk_method="auto"`.

**Status.** Periodic RIJCOSX is **experimental**. The 2-D slab Ewald route is
**production** for RHF, RKS and UKS (no smearing yet). Read
[slabs and adsorbates](user_guide/slabs_and_adsorbates.md) and the
[slab adsorbate tutorial](tutorial/slab_adsorbate_dft.md).

### Electronic structure of solids: bands, DOS, COOP/COHP, bond orders, QTAIM

**Theory.** Once the periodic SCF has converged, diagonalising
$\mathbf F(\mathbf k)$ along a path of high-symmetry $\mathbf k$ gives the
**band structure**; histogramming the eigenvalues over a dense mesh with
Gaussian broadening gives the **density of states**, and weighting each state
by its Mulliken population on an atom or orbital gives the **projected DOS**
and **fat bands**. The **crystal orbital overlap population**
({vibeqc-cite}`hughbanks_hoffmann_coop_1983`) and **Hamilton population**
({vibeqc-cite}`dronskowski_cohp_1993`) resolve the DOS into bonding and
antibonding contributions between atom pairs by weighting with
$S_{\mu\nu}$ or $H_{\mu\nu}$ respectively. **Periodic Mayer bond orders**
generalise the molecular index over the k mesh, and **QTAIM**
({vibeqc-cite}`bader_qtaim_1985`) partitions space by the zero-flux surfaces
of the density gradient, locating bond critical points and bond paths.

**Select it.** `band_structure`, `density_of_states`, `band_structure_projected`,
`compute_coop_cohp`, `periodic_mayer_bond_orders`, `vibeqc.qtaim.qtaim_analysis`;
the `vibeqc coop` CLI; all written to QVF for vibe-view.

**Status.** Bands, DOS, PDOS, fat bands, periodic Mayer bond orders, QTAIM
and COOP/COHP are **production** on the GDF routes. COHP is computed with the
same spin Hamiltonian that produced the analysed eigenpairs, and the kernel
refuses any other operator rather than labelling it COHP; on BIPOLE, AICCM and
external-XC routes the DOS and COOP/COHP sections are skipped with a stated
reason. Read the [band structure tutorial](tutorial/band_structure.md),
[PDOS](tutorial/pdos.md), [COOP/COHP](user_guide/coop_cohp.md) and
[QTAIM](user_guide/qtaim.md).

### Phonons, stress and variable cells

**Theory.** Displacing each atom by a small step and differencing the
analytic forces gives the force-constant matrix; mass-weighting and
diagonalising it at $\mathbf q = 0$ gives the **Gamma-point phonons**. The
**stress tensor** is the strain derivative of the energy per volume; vibe-qc
obtains it by central differences of the energy over strain with the basis
rebuilt on each strained cell, so Hellmann-Feynman and Pulay terms arrive
together. An **equation-of-state fit** (third-order Birch-Murnaghan) turns an
energy-volume scan into the equilibrium volume and bulk modulus.

**Select it.** `vibeqc.PhononCalculator`, `compute_stress_gpw`, `compute_stress_gapw`,
`vibeqc.eos.fit_birch_murnaghan`.

**Status.** Phonons are a **preview** (Gamma only, GAPW forces). Stress is
finite-difference only and `optimize_cell=True` fails closed on every ab
initio route (**gated**); variable-cell relaxation is reachable on the full-k
semiempirical routes. The EOS fit is **production**. See the
[periodic geometry optimisation tutorial](tutorial/periodic_geometry_optimization.md).

### Surface embedding

**Theory.** A semi-infinite surface is partitioned into a surface region
treated explicitly and a bulk region whose influence enters through an
energy-dependent self-energy built from the bulk Green's function
({vibeqc-cite}`inglesfield_embedding_1981`), so the surface sees the true
bulk continuum instead of a truncated slab.

**Select it.** The embedded-surface calculator described in
[surface embedding](user_guide/surface_embedding.md).

**Status.** **Experimental.** The 3-D Gaussian foundation is validated;
thick-slab GDF parity is pending, so absolute energies and densities are
qualitative. Read the [surface embedding tutorial](tutorial/surface_embedding.md).

### AICCM: the ab initio cyclic cluster model

**Theory.** AICCM carries the cyclic cluster idea of SECCM to first
principles ({vibeqc-cite}`peintinger_ccm_2014`). A finite Born-von Karman
torus of $N_1\times N_2\times N_3$ unit cells replaces the k-mesh: the
electronic structure of the crystal is represented by one real-space
supercell with cyclic boundary conditions, and every two-electron integral
is weighted by where the interacting charge distributions fall relative to
the Wigner-Seitz cell of the torus. Because the Hamiltonian is a finite,
molecule-like object, the whole molecular correlation hierarchy (MP2, CCSD,
DLPNO) can be applied to it directly, which is the motivation for the
approach. vibe-qc maintains two independent constructions on purpose:

- **Gamma-CCM** is the neutral finite-torus construction with two admissible
  producers, `variant="real-gamma"` (the real-Gamma supercell
  representation, which carries the exchange $\mathbf q = 0$ seam explicitly)
  and `variant="neutral-bloch"` (the same Hamiltonian in its Bloch
  representation). The same development line also keeps
  `variant="four-center"`, the 2014 union-and-weight / Wigner-Seitz lineage,
  which is a different construction rather than a third producer.
- **chi-CCM** (`variant="chi"`) builds the Hamiltonian from the characters of
  the finite translation group on a Gamma-centred character mesh, with
  four-centre, RI and RIJCOSX backends.

Agreement between the two under a matched Coulomb gauge is the evidence for
a common limit; divergence decides which construction is admissible. No
cross-approach delta is reportable yet.

**Select it.** `run_periodic_job(system, basis, method="aiccm", variant=..., aiccm_lattice_extension=(N1, N2, N3), correlation="mp2" | "ccsd" | "dlpno-mp2" | "dlpno-ccsd")`.
`variant` is mandatory.

**Status.** **Experimental**, every variant, every driver. All are 3-D only;
the 1-D and 2-D absolute-energy backends fail closed; results are research
output and not part of any release gate. Read the
[AICCM guide](user_guide/aiccm.md), the [quickstart](tutorial/aiccm_quickstart.md),
the [Gamma-CCM](tutorial/aiccm2026dev_a.md) and [chi-CCM](tutorial/aiccm2026dev_b.md)
tutorials, and the [chi-CCM reference](user_guide/aiccm2026dev_b.md).

## Workflows on the energy surface

### Geometry optimisation, relaxed scans and intrinsic reaction coordinates

**Theory.** A stationary point satisfies $\partial E/\partial\mathbf R = 0$.
Quasi-Newton optimisers (BFGS and its line-search variant through ASE, or
the native optimiser) update an approximate Hessian from successive
gradients and converge in tens of steps for molecules; forces come from the
Hellmann-Feynman term plus the Pulay correction for the basis functions that
move with the nuclei ({vibeqc-cite}`pulay_forces_1969`). A **relaxed scan**
freezes one internal coordinate at a series of values and relaxes the rest.
The **intrinsic reaction coordinate** follows the mass-weighted steepest
descent path from a transition state down to both minima
(Ishida-Morokuma-Komornicki damped integrator).

**Select it.** `run_job(..., optimize=True)`, `optimize_molecule`, the
`VibeQC` ASE calculator; `vibeqc.relaxed_scan(...)` for scans; `run_irc`;
periodic fixed-cell relaxation through `run_periodic_job(..., optimize=True)`.

**Status.** **Production** for molecules on every method with a gradient
(analytic where it exists, finite differences otherwise, always stated in
the `.out`). Two caveats matter: `optimize=True` with a post-SCF method
(MP2, CC, DLPNO, OVGF) relaxes the mean-field reference surface and tags the
output with that `energy_method`, and the correlated methods without a fast
gradient (FCI, NEVPT2, CASPT2, MRCI) require `optimizer_backend="native"` or
`"brent"` for a finite-difference optimisation on their own surface (#357).
Periodic relaxation is production on BIPOLE (finite-difference forces) and the
analytic-gradient GDF routes; periodic IRC fails closed. Read
[geometry optimisation](tutorial/geometry_optimization.md),
[relaxed PES scans](tutorial/relaxed_pes_scan.md) and
[periodic geometry optimisation](tutorial/periodic_geometry_optimization.md).

### Transition states: nudged elastic band, dimer and eigenvector following

**Theory.** A chain of images between reactant and product is relaxed under
the true force perpendicular to the path plus a spring force along it; the
climbing image is pushed uphill along the tangent until it sits on the
saddle point, and the barrier is the energy of that image. The **dimer
method** needs no second image: two close replicas are rotated to find the
lowest curvature mode and translated uphill along it and downhill in every
other direction. **Eigenvector following** (EF, P-RFO) does the same with an
approximate Hessian and Bofill updates.

**Select it.** `run_neb` (HF, DFT, ROHF/ROKS, MACE, semiempirical and SECCM
surfaces) or the ASE `NEB` on `VibeQC` calculators; `run_dimer`;
`run_job(..., geom_target="transition_state", geom_opt="dimer" | "ef" | "prfo")`.

**Status.** NEB, the dimer method and the IRC are **production** for molecules
(the dimer needs a sensible `initial_direction`); eigenvector following and
P-RFO through `run_job` are a **preview**, because they did not converge from
an approximate start on HCN/HNC and NH3 in testing and the `.out` does not
say when max iterations was reached. Periodic ROHF/ROKS, ECP and DFT+U
combinations are gated route by route. Read the
[NEB tutorial](tutorial/neb_reaction_path.md), the [NEB guide](user_guide/neb.md)
and [IRC and dimer](user_guide/irc_and_dimer.md).

### Vibrational frequencies and thermochemistry

**Theory.** Diagonalising the mass-weighted Hessian after projecting out
translations and rotations gives the harmonic normal modes and frequencies;
the dipole derivative along each mode gives its IR intensity. The
rigid-rotor harmonic-oscillator partition functions turn frequencies,
moments of inertia and mass into zero-point energy, enthalpy, entropy, heat
capacity and Gibbs free energy at a given temperature and pressure.

**Select it.** `run_job(..., hessian=True)` computes the Hessian, the
frequencies, the IR intensities and the RRHO thermochemistry in one pass;
`thermo_options` sets temperature and pressure. The pieces are also
callable on their own: `compute_hessian_fd`, `compute_hessian_rhf_analytic`,
`ir_intensities`, `compute_thermochemistry`.

**Status.** **Production**, validated against PySCF to 1e-7 Ha in the
thermochemistry and below 0.01 cm⁻¹ in analytic frequencies. Read the
[frequencies tutorial](tutorial/vibrational_frequencies.md) and
[thermodynamics](tutorial/thermodynamics.md).

### Molecular dynamics, metadynamics and conical intersections

**Theory.** Born-Oppenheimer molecular dynamics integrates Newton's equations
on the electronic energy surface with the velocity-Verlet scheme; energy
conservation in the NVE ensemble is the consistency test between forces and
energy. **Well-tempered metadynamics** adds a history-dependent bias of
Gaussian hills along chosen collective variables so the trajectory escapes
minima and the accumulated bias reconstructs the free-energy surface. A
**minimum-energy conical intersection** is located by minimising the
average of two state energies under a penalty on their gap.

**Select it.** `from vibeqc.md import run_md, run_nvt` for any method with
forces, or ASE `VelocityVerlet` on a `VibeQC` calculator;
`vibeqc.metadynamics.run_metadynamics` layers the well-tempered bias on the
same driver with distance and angle collective variables;
`vibeqc.conical.optimize_conical_intersection` implements the
Levine-Coe-Martinez penalty and takes any two tracked states with energies
and gradients, in practice the HF-CIS finite-difference surface or MSINDO
CIS. The top-level `vibeqc.run_md` / `run_nve` / `run_nvt` / `run_npt` names
are the GAPW periodic helpers and inherit that route's label.

**Status.** **Production** for molecular dynamics, metadynamics and the
MECI optimiser on any surface with forces; the GAPW periodic dynamics helpers
follow GAPW. Read the [MD tutorial](tutorial/molecular_dynamics.md) and the
[MD guide](user_guide/molecular_dynamics.md).

### Basis-set optimisation

**Theory.** The exponents and contraction coefficients of a Gaussian basis
are variational parameters. vibe-basis minimises a molecular or cohesive
energy objective over them with analytic energy gradients and a BDIIS
update, which is how the pob family of solid-state bases was constructed.

**Select it.** `optimize_molecular_basis`, the `vb` command-line tool.

**Status.** **Production**; see [basis optimisation](user_guide/basis_optimization.md)
and the [basis convergence tutorial](tutorial/basis_convergence.md).

## Analysis of the wavefunction

### Charges, bond orders and localised orbitals

**Theory.** **Mulliken** populations split each overlap population equally
between the two centres; **Loewdin** populations use the symmetrically
orthogonalised basis and are less basis-set sensitive; **Hirshfeld** charges
partition the real-space density in proportion to superposed free-atom
densities ({vibeqc-cite}`hirshfeld_1977`) and are the most stable of the
three. **Intrinsic atomic orbitals** ({vibeqc-cite}`knizia_ibo_2013`) build a
minimal, polarised atomic basis that exactly spans the occupied space, giving
basis-set-independent charges and, after localisation, intrinsic bond
orbitals. **Mayer** ({vibeqc-cite}`mayer_bond_order_1983`) and **Wiberg**
({vibeqc-cite}`wiberg_bond_index_1968`) bond orders and the
**delocalisation index** measure electron sharing between atom pairs from
the density matrix.

**Select it.** `mulliken_charges`, `loewdin_charges`, `hirshfeld_charges`,
`vibeqc.bond_analysis`, `mayer_bond_orders`; all emitted into the
`.population.txt` and `.population.json` sidecars by `run_job`.

**Status.** **Production.** NPA charges are **gated** (raise
`NotImplementedError` until natural atomic orbitals land) and the NBO search
is a provisional exploratory helper. Read [properties](user_guide/properties.md),
[bond analysis](user_guide/bond_analysis.md), [IAO populations](user_guide/iao_population.md)
and the [post-SCF properties tutorial](tutorial/post_scf_properties.md).

### Energy decomposition and orbital entanglement

**Theory.** **Energy decomposition analysis** splits an interaction energy
between two fragments into electrostatic, exchange, repulsion, polarisation
and dispersion terms, in the LMO-EDA ({vibeqc-cite}`su_li_eda_2009`) and
Morokuma ({vibeqc-cite}`morokuma_eda_1971`) schemes. **Orbital entanglement**
measures ({vibeqc-cite}`legeza_entanglement_2003`) use the one- and
two-orbital reduced density matrices to compute single-orbital entropies and
mutual information, which reveal static correlation and guide active-space
selection.

**Select it.** `vibeqc.eda.eda_lmo`, `vibeqc.eda.eda_morokuma`,
`vibeqc.entanglement.single_orbital_entropy`, `mutual_information`,
`entanglement_bond_order`.

**Status.** The entanglement measures on a correlated reduced density
matrix are **production**; the convenience wrapper
`entanglement_from_density` is an occupation-based approximation (its
overlap transform was corrected on main, #418). The EDA functions are
**preview**: they return the real LMO-EDA and Kitaura-Morokuma terms for two
closed-shell Hartree-Fock fragments computed in the dimer basis, and the terms
sum to the interaction energy. Kohn-Sham, ECP, open-shell and
other-basis inputs are refused with a `ValueError`; the Kohn-Sham form, open-shell
fragments, the Kitaura-Morokuma split of polarisation, charge transfer and
coupling, and a `run_job` keyword are not implemented. You assemble the three
SCF runs yourself, as the
[EDA and entanglement tutorial](tutorial/eda_and_entanglement.md) shows.

### Natural orbitals, natural transition orbitals and volumetric data

**Theory.** Diagonalising a correlated one-particle density matrix gives
natural orbitals whose occupation numbers measure correlation; the same
operation on a transition density gives natural transition orbitals. Any
orbital or density can be sampled on a grid and written as a Gaussian cube,
an XSF volume, or, for Fermi surfaces, a BXSF file.

**Select it.** `natural_orbitals`, `run_job(..., nto=True)`,
`write_cube_density`, `write_cube_mo`, `write_cube_mos` and their periodic
siblings, `write_bxsf`; every run writes a Molden file and a QVF archive.

**Status.** **Production.** Read [volumetric data](user_guide/volumetric_data.md),
[orbital visualisation](tutorial/orbital_visualization.md) and
[XSF and BXSF](tutorial/xsf_bxsf_visualization.md).

## Input, output and interchange

Every user-facing file goes through `vibeqc.output`: the human-readable
`.out`, the `.system` manifest with provenance and method status, the `.bibtex`
and `.references` citation files for exactly the methods you used, the
population sidecars, Molden orbitals, ASE trajectories and the **QVF**
archive that carries the whole calculation for [vibe-view](visualization.md).
Wavefunctions can also leave as **TREXIO** files for other correlated codes
and as **QCSchema** JSON for workflow engines, and structures come in from
XYZ, extended XYZ, POSCAR and CIF, or from public databases through
`vqfetch` (OPTIMADE, Materials Project, COD, NIST). All of these are
**production**; see [output files](user_guide/output_files.md),
[TREXIO](user_guide/trexio.md), [QCSchema](qcschema.md),
[external structures](user_guide/external_structures.md) and the
[ASE integration guide](user_guide/ase_integration.md).

## Not in vibe-qc today

So that "every method" is a checkable claim, these standard capabilities are
absent rather than experimental: equation-of-motion and STEOM coupled
cluster for excited states, ADC, GW and Bethe-Salpeter, MP3 and higher MP
orders, explicitly correlated F12 methods, symmetry-adapted perturbation
theory, NMR shieldings and spin-spin couplings, EPR parameters, analytic
Hessians beyond the four mean-field references, periodic D4, periodic IRC and
variable-cell relaxation on the ab initio routes. The citation database
carries route keys for several of them because the output layer is prepared
for them, not because they run. The [roadmap](roadmap.md) says which are
planned.

## Status at a glance

| Family | Production | Preview | Experimental | Gated |
|---|---|---|---|---|
| Mean field | RHF, UHF, ROHF, RKS, UKS, ROKS; LDA to global hybrids with gradients; D3(BJ), D4; RIJK, RIJCOSX; ECPs; CPCM/COSMO; composites (molecular) | Range-separated and VV10 gradients (finite differences); ROKS gradients | SKALA; native D4 backend; COSMO-RS; Jacobi-Davidson and harmonic-Davidson (GPLHR) solvers; MANTA integral engine with its RI, BUPO and RI-BUPO J/K selectors | Solvent with correlated, CAS, semiempirical (except MSINDO) or MLIP methods |
| Single-reference correlation | MP2 family, double hybrids, CCSD, CCSD(T), open-shell CC, CC2, CC3, CCSDT, QCISD(T), CEPA(n), CISD, DLPNO-MP2, DLPNO-CCSD(T), open-shell DLPNO | Dense production-size memory envelopes; DLPNO-CCSD(T) on 3-D crystals (RHF or UHF GDF reference, Gamma or BvK-folded mesh) | | |
| Multireference and exact | CASCI, CASSCF (state-specific gradient), IC-CASPT2, NEVPT2, MRCI, FCI, selected CI, v2RDM | PT2 and state-averaged gradients (finite differences); transcorrelated CI | DMRG (toy, 6 spatial orbitals); SC-CASPT2 behind `VIBEQC_EXPERIMENTAL_MULTIREF`; embedded-cluster CAS (`vibeqc.embed`) | ECP references in CAS methods |
| Excited states and response | TDDFT Casida and TDA, NTOs, spectra; OVGF; CPHF polarizability; analytic Hessians | Excited-state gradients (RHF/CIS, finite differences) | | Range-separated TDDFT kernels |
| Semiempirical and MLIP | MSINDO within scope; MACE interface | DFTB0, SCC-DFTB | GFN2-xTB, PM6, OM1/OM2/OM3, SECCM | PM7 |
| Periodic | GDF, GPW, BIPOLE energies; 2-D slab Ewald; smearing, mixers; DFT+U; bands, DOS, PDOS, fat bands; COOP/COHP; periodic Mayer; QTAIM; EOS fits | Periodic ROHF/ROKS; BIPOLE analytic gradients; Gamma phonons; finite-difference stress; symmetry-reduced exact exchange (`symmetry_reduce_k`) | GAPW; periodic RIJCOSX; surface embedding; AICCM (all variants); MPI overlay; MANTA finite-domain periodic HF and KS | Variable-cell relaxation on ab initio routes; BIPOLE quartet multipole |
| Workflows and analysis | Optimisation, scans, IRC, NEB, dimer, MD, metadynamics, MECI, frequencies, thermochemistry, basis optimisation; charges, bond orders, IAO, EDA, entanglement; all I/O routes | Post-SCF `optimize=True` relaxes the reference surface | | NPA; periodic IRC; variable-cell relaxation on ab initio routes |

When this table and a method's own page disagree, the method page wins and
this table has a bug: please [report it](https://github.com/vibe-qc/vibe-qc/issues).
