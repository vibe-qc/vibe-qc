# Basis Sets in Quantum Chemistry: A Practical Survey for vibe-qc

**Author:** prepared for M. F. Peintinger / vibe-qc basis set development discussion
**Date:** 2026-05-08
**Scope:** survey and ranking of small, medium, and large basis sets for molecular and periodic calculations, broken out by chemical and material category

> **Source.** This document is a literature review delivered into the
> `basissetdev` chat by the user on 2026-05-08. It is preserved here
> verbatim as the canonical reference for [ROADMAP_BASIS_LIBRARY.md](ROADMAP_BASIS_LIBRARY.md),
> which translates the review's recommendations into an actionable
> per-basis-set status / priority table for the vibe-qc bundled
> library.

---

## 1. Executive Summary

Modern quantum chemistry has converged on a small number of well-engineered basis set families. For molecular work the practical default is the Karlsruhe **def2** family (Weigend & Ahlrichs, 2005), which spans `def2-SVP` through `def2-QZVPP` and is paired with consistent ECPs from Rb onward. For benchmark-grade work the **Dunning correlation-consistent** family (`cc-pVnZ`, `aug-cc-pVnZ`) and the **Jensen polarization-consistent** family (`pc-n`, `pcseg-n`) are the references. STO-3G has essentially no place in production calculations any more, but it remains useful as a teaching tool, a Hückel-quality starting guess, and a worst-case stress test.

For periodic solids the situation is more fragmented. Plane-wave plus pseudopotential codes (VASP, Quantum ESPRESSO, ABINIT, CASTEP) dominate, with **VASP PAW** and the **SSSP** (Quantum ESPRESSO) and **PseudoDojo** libraries being the de facto standards. For Gaussian-orbital periodic codes (CRYSTAL, CP2K) the **pob-TZVP / pob-TZVP-rev2** and **MOLOPT** families are standard. For all-electron periodic work, **FHI-aims** numerical atomic orbitals (NAO, tier1/tier2/tight/really_tight) and (L)APW+lo (WIEN2k, exciting) are the references.

A new class of "**3c**" composite methods (HF-3c, PBEh-3c, B97-3c, r2SCAN-3c, ωB97X-3c) bundles a small custom basis with empirical corrections (D3/D4 dispersion, gCP) and now consistently outperforms naive `B3LYP/6-31G*` at similar cost. These should be considered first-line tools for routine work.

For vibe-qc, our recommended priority order for first-class support is:

1. **def2** family (SVP, TZVP, TZVPP, QZVP, QZVPP), with the matching ECPs and dhf/x2c relativistic counterparts
2. **Pople** family for legacy compatibility (3-21G, 6-31G(d), 6-31+G(d,p), 6-31++G**)
3. **Dunning cc-pVnZ / aug-cc-pVnZ** for benchmarking and post-HF
4. **Jensen pcseg-n / aug-pcseg-n** as DFT-optimal alternative
5. **vDZP** as the new low-cost workhorse (with ECPs)
6. **STO-3G, MINI, MIDI** for guess generation and pedagogy
7. **Composite "3c" basis sets** (def2-mSVP, def2-mTZVP, def2-mTZVPP) when wrapped with their corrections
8. For solids: **PAW (VASP-style)**, **ONCV (PseudoDojo, SG15)**, **pob-TZVP-rev2**, **MOLOPT**

---

## 2. Background and Terminology

A basis set is the finite set of analytic functions used to expand the molecular orbitals. Modern molecular codes use contracted Gaussian-type orbitals (cGTOs), with the radial part of each primitive being $G_\alpha(r) = N e^{-\alpha r^2}$. Solids more often use plane waves $e^{i\mathbf{k}\cdot \mathbf{r}}$, numerical atom-centered orbitals, or augmented plane waves.

Key descriptors:

- **Zeta level (n-zeta).** Number of basis functions used to describe each valence orbital. STO-3G is single-zeta. 6-31G, def2-SVP, cc-pVDZ are double-zeta. def2-TZVP, cc-pVTZ, pcseg-2 are triple-zeta. cc-pVQZ, def2-QZVP are quadruple-zeta.
- **Polarization functions.** Higher-angular-momentum functions (d on C, p on H, f on transition metals, etc.) that allow the electron density to deform on bond formation. Modern usage demands them: unpolarized basis sets like 6-31G or cc-pVDZ-without-d should never be used for production.
- **Diffuse functions.** Low-exponent functions needed for anions, weakly bound species, Rydberg states, and many response properties (polarizabilities, NMR shielding tensors, TDDFT excitations). Indicated by `+`, `++`, `aug-`, or `ma-` prefixes.
- **Contraction.** Segmented vs. general. Segmented (Pople, def2, pcseg) is faster in most integral codes. Generally contracted (cc, ANO) is more compact and accurate per primitive but needs a code that exploits it.
- **Effective core potential (ECP).** Replaces inner-shell electrons with a parameterized potential. Reduces cost, builds in scalar relativistic effects. The Stuttgart/Cologne ECPs paired with def2 valence sets are the de facto standard above Kr.
- **Polarization-consistent vs. correlation-consistent design.** `cc-pVnZ` was optimized to recover correlation energy in CCSD/CCSD(T) calculations on atoms. `pc-n` was optimized to converge HF and DFT energies. For DFT, pcseg-n is technically the better-targeted choice, but def2-TZVP achieves nearly the same accuracy at lower cost in practice.

A note on **basis set superposition error (BSSE)** and **basis set incompleteness error (BSIE)**: these are large at the double-zeta level (several kcal/mol on noncovalent interactions) and are the reason a triple-zeta basis is now the recommended floor for any quantitative molecular work. The geometrical counterpoise correction (gCP) of Kruse and Grimme is the cheapest practical fix and is built into the 3c composite methods.

---

## 3. Molecular Basis Sets

### 3.1 Ranking by Size

#### Small (single-zeta to minimal-DZ, fast guess / pedagogical)

| Rank | Basis | Zeta | Notes |
|------|-------|------|-------|
| 1 | **STO-3G** | SZ | Single-zeta minimal, no polarization. Useful only for initial guesses, very large systems where qualitative geometries suffice, or pedagogy. |
| 2 | **MINI / MINIs (Huzinaga)** | SZ | Slightly better than STO-3G atomic energies. Niche use. |
| 3 | **3-21G** | DZ (unpol) | Old Pople split-valence DZ. Faster than 6-31G but unpolarized. Avoid for production. |
| 4 | **pcseg-0 / pc-0** | DZ (unpol) | Jensen DZ without polarization. Better-balanced than 3-21G but still unpolarized. |
| 5 | **6-31G** | DZ (unpol) | Historical workhorse without polarization. The Pitman et al. (2024) benchmark showed unpolarized DZ basis sets perform poorly and should not be used. |

**Verdict.** For "small" we recommend supporting STO-3G (legacy and pedagogical) and 3-21G (for the rare contexts where speed dominates). All other small basis sets in the table should carry warnings in vibe-qc's UI that they are unsuitable for production.

#### Medium (polarized double-zeta, the "everyday" tier)

| Rank | Basis | Notes |
|------|-------|-------|
| 1 | **def2-SVP** | Karlsruhe segmented DZ with polarization on heavy atoms. The most widely used DZ basis in modern DFT codes (Turbomole, ORCA, Q-Chem, Gaussian, Psi4). Best speed/accuracy balance for routine geometry optimization. |
| 2 | **vDZP** (Grimme, 2023) | Custom polarized DZ used inside ωB97X-3c. Uses large-core ECPs and deep contraction. On the TorsionNet206 drug-like benchmark, vDZP-based methods give MAEs of 0.4 to 0.5 kcal/mol, comparable to triple-zeta hybrid methods. Major asset for large heavy-atom systems. |
| 3 | **6-31++G(d,p) / 6-31++G**\*\* | Pitman et al. found this to be the best-performing DZ basis on diet-GMTKN55. The diffuse functions matter. |
| 4 | **6-31G(d,p) / 6-31G**\*\* | The classic Pople workhorse. Still acceptable for organic structures, but inferior to def2-SVP at similar cost. |
| 5 | **pcseg-1** | Jensen segmented DZ-polarized. Slightly better than def2-SVP for HF/DFT energies but less widely supported. |
| 6 | **cc-pVDZ** | Dunning DZ. Designed for correlation, so for pure DFT it is less efficient than pcseg-1 or def2-SVP. Useful as a stepping stone in cc-pVnZ extrapolations. |
| 7 | **def2-mSVP** | Modified def2-SVP used inside PBEh-3c and B3LYP-3c composite methods. Not for standalone use. |
| 8 | **LANL2DZ** | Hay-Wadt ECP plus DZ valence. Historically common for transition metals but inferior to def2-SVP/TZVP plus def2-ECP. Should be deprecated. |

**Verdict.** def2-SVP is the default. vDZP is a strong contender for any system with heavy atoms.

#### Large (triple-zeta and above, "production accuracy")

| Rank | Basis | Notes |
|------|-------|-------|
| 1 | **def2-TZVP** | The de facto reference for hybrid-DFT calculations on molecules of any size where it is affordable. Bursch et al. (2022) and many other benchmarks endorse it. Within 0.5 kcal/mol of def2-QZVPD for main-group thermochemistry. |
| 2 | **def2-TZVPP** | Extra polarization on hydrogen. Recommended for transition metals, weak interactions, and converging properties beyond geometries. |
| 3 | **pcseg-2** | Pitman et al. (2024) found this to be the best-performing TZ basis for DFT thermochemistry on diet-GMTKN55. Slightly better than def2-TZVP but less widely available. |
| 4 | **cc-pVTZ / aug-cc-pVTZ** | Dunning TZ, the reference for post-HF (MP2, CCSD(T)) calculations. Use aug-variant for anions, noncovalent, and response. |
| 5 | **def2-QZVP / def2-QZVPP** | Quadruple-zeta. For benchmarking small molecules, basis-set-limit reference, double-hybrid functionals. |
| 6 | **cc-pVQZ / aug-cc-pVQZ** | Quadruple-zeta correlation-consistent. Standard for CCSD(T)/CBS extrapolation in `(T,Q)` or `(Q,5)` schemes. |
| 7 | **pcseg-3 / pcseg-4** | Larger pc-n family members. Excellent DFT convergence but rarely needed in practice. |
| 8 | **def2-TZVPD / ma-def2-TZVP** | Triple-zeta with diffuse functions. Use for anions, TDDFT, polarizabilities. The "ma-" (minimally augmented) versions of Truhlar and Zheng are a cheaper alternative to full aug-. |
| 9 | **cc-pV5Z, cc-pV6Z, aug-cc-pV5Z** | For very small systems and CBS extrapolation only. |
| 10 | **ANO-RCC-VTZP, ANO-RCC-VQZP** | Generally contracted relativistic ANOs. Standard for CASPT2/NEVPT2/MS-CASPT2 (OpenMolcas, BAGEL). Compact per primitive, ideal for multireference. |

**Verdict.** def2-TZVP is the production-quality default. cc-pVTZ family is the reference for post-HF. ANO-RCC is mandatory for multireference.

---

### 3.2 Ranking by Molecular Category

#### Organic molecules (C, H, N, O, S, halogens, P)

| Tier | Recommendation |
|------|----------------|
| Small (geometry, screening) | def2-SVP or B97-3c (def2-mTZVP plus D3 plus gCP) |
| Medium (production) | **def2-TZVP** with a hybrid functional (ωB97X-D, ωB97M-V, M06-2X, B3LYP-D3) |
| Large (benchmarking) | aug-cc-pVTZ to aug-cc-pVQZ extrapolation, or ωB97M-V/def2-QZVPP |
| Anions, Rydberg, response | aug- variants required (aug-cc-pVTZ, ma-def2-TZVP, aug-pcseg-2) |

Pople basis sets (6-31G(d), 6-311G(2df,p)) remain in widespread use due to inertia. The polarized 6-311G family is poorly parameterized (Pitman et al. 2024 explicitly recommends avoiding it). For new code, support 6-31G(d) and 6-31++G(d,p) for compatibility, but encourage def2.

#### Metal-organic / organometallics (transition metal complexes, MOFs)

| Tier | Recommendation |
|------|----------------|
| Small (screening) | def2-SVP (with def2-ECP for 4d/5d) plus dispersion correction |
| Medium (production) | **def2-TZVP / def2-TZVPP** on the metal, def2-TZVP on ligands. Use a TM-tested functional: TPSSh, B3LYP*-D3(BJ), or PWPB95-D3(BJ) for benchmarking. |
| Large (benchmark) | def2-QZVPP, x2c-TZVPPall for explicit relativity |
| Multireference | **ANO-RCC-VTZP / VQZP** with CASPT2, NEVPT2, or DMRG-CASPT2 |
| Spin states | Truhlar's group recommends def2-TZVP for spin-splitting energies; extra polarization does not help |

Avoid `LANL2DZ` for new work. The Hay-Wadt ECPs are older and less accurate than the Stuttgart/Cologne ECPs that ship with def2.

For 4d and 5d transition metals, scalar relativistic effects matter. Either use def2-ECP (built in to the def2 valence set) or, for explicit treatment, dhf-TZVP / x2c-TZVPall (Pollak & Weigend 2017) or DKH-recontracted def2 (def2-TZVP-DKH).

#### Inorganic (main-group clusters, oxides, halides, hypervalent species)

| Tier | Recommendation |
|------|----------------|
| Small | def2-SVP (with care for hypervalent S, P needing extra d) |
| Medium | **def2-TZVPPD** (the extra d and diffuse matter) or aug-cc-pV(T+d)Z for second-row hypervalent |
| Large | def2-QZVP, aug-cc-pwCVTZ-DK for explicit core-valence |
| Anions, halide complexes | aug- variants required |
| Heavy main group (Sn, Pb, Bi) | def2-TZVP plus ECP, or all-electron x2c-TZVPall |

For second-row hypervalent (SO₃²⁻, SF₆, PF₅, ClO₄⁻), the "tight d" variants (e.g., aug-cc-pV(T+d)Z, Dunning et al. 2001) are essential. The def2 family already includes appropriate d-functions for these atoms.

#### Biomolecules (peptides, nucleic acids, lipids, sugars)

| Tier | Recommendation |
|------|----------------|
| Screening (large fragments) | **r2SCAN-3c** (def2-mTZVPP plus D4 plus gCP) or B97-3c |
| Production (single-point on optimized fragments) | def2-TZVP with ωB97X-D or B3LYP-D3(BJ) |
| Noncovalent / hydrogen bonding | def2-TZVPPD or ma-def2-TZVP, with explicit dispersion (D4 preferred) |
| Vibrational, NMR | property-specific basis sets (pcS-n for shielding, pcJ-n for J couplings) |

The 3c composite methods are particularly well-suited here because gCP corrects for the BSSE that dominates large biomolecule binding-energy errors. Recent benchmarks (Behara et al. 2024, TorsionNet206) showed B3LYP-D3BJ/6-31G(d) is now clearly inferior to vDZP-based or 3c methods at comparable cost.

#### Pharma / drug-like molecules

| Tier | Recommendation |
|------|----------------|
| High-throughput conformer / torsion scans | **ωB97X-3c (vDZP)** or r2SCAN-3c |
| Reference single-points for force-field parameterization | ωB97M-V/def2-TZVPPD or DLPNO-CCSD(T)/def2-TZVPP |
| QM/MM cores | def2-SVP (QM region), with def2-TZVP single-point refinement |

vDZP deserves particular attention here. The Rowan/Wagen TorsionNet206 study showed vDZP-based methods give 0.4 to 0.5 kcal/mol MAE on torsion energies (versus CCSD(T)/def2-TZVP), comparable to standard hybrid functionals with triple-zeta basis sets. The use of ECPs makes it especially efficient for halogenated and metal-containing drug candidates.

---

## 4. Solid-State Basis Sets

### 4.1 The Three Families

Periodic codes split into three philosophical camps:

1. **Plane waves plus pseudopotentials.** VASP, Quantum ESPRESSO, ABINIT, CASTEP, GPAW. A single cutoff energy parameter controls the basis. No BSSE. Smooth convergence. Heavy reliance on the quality of the pseudopotential or PAW dataset.
2. **Gaussian basis sets.** CRYSTAL, CP2K (mixed Gaussian-plane wave), Turbomole-Riper, FHI-aims (in some modes). Inherits molecular-style basis sets, but they need re-optimization for solids because the diffuse functions in molecular bases lead to linear dependencies in periodic systems.
3. **Numerical atomic orbitals or augmented plane waves.** FHI-aims, SIESTA, OpenMX (NAO); WIEN2k, exciting, ELK (LAPW+lo). Highly accurate, tend to be slower but with smaller basis sizes.

### 4.2 Plane Wave Pseudopotential Libraries

The pseudopotential is the basis-set-equivalent choice for plane-wave codes. The cutoff is the convergence parameter.

| Library | Code | Type | Characteristics |
|---------|------|------|-----------------|
| **VASP PAW (PBE)** | VASP | PAW | The de facto industry standard for materials science. "_GW" and "_sv" / "_pv" variants for semicore states. Closed-source. Cutoff range 250 to 600 eV depending on element. |
| **VASP PAW (PBE.54, PBE.64)** | VASP | PAW | Updated VASP datasets. Use latest available, especially for d-block elements. |
| **SSSP precision (1.3.0)** | Quantum ESPRESSO | mixed PAW + USPP + ONCV | Curated by EPFL/Marzari group. The most accurate open-source PSP library: average Δ-factor below 0.4 meV/atom against all-electron references (Prandini et al. 2018). |
| **SSSP efficiency (1.3.0)** | Quantum ESPRESSO | mixed | Same protocol with lower cutoffs, faster, slightly less accurate. Standard for high-throughput. |
| **PseudoDojo (ONCV, NC SR/FR)** | abinit, QE, others | norm-conserving (ONCV) | Hamann-Schlüter-Chiang norm-conserving, scalar and fully relativistic versions. State of the art for systematic, hybrid-functional, and GW calculations. |
| **SG15** | many codes | ONCV | Schlipf-Gygi 2015 norm-conserving library. Solid alternative to PseudoDojo, well-tested. |
| **GBRV** | Quantum ESPRESSO, ABINIT | USPP, PAW | Garrity-Bennett-Rabe-Vanderbilt high-throughput library. |
| **JTH PAW** | ABINIT | PAW | Jollet-Torrent-Holzwarth. ABINIT-native PAW. |

For **vibe-qc**, the recommended primary library to support is **PseudoDojo** (open-source, ONCV, scalar and fully relativistic, well-converged). SSSP precision is the natural complement when high-throughput materials screening is the goal.

The 2016 multi-code Δ-factor study (Lejaeghere et al., *Science*) showed that all of these libraries now agree with all-electron LAPW codes (WIEN2k, exciting) to within roughly 1 meV per atom on equation-of-state data, which is the practical floor for DFT precision.

### 4.3 Gaussian Basis Sets for Periodic Systems

| Basis | Code | Notes |
|-------|------|-------|
| **pob-TZVP** | CRYSTAL | Peintinger-Oliveira-Bredow 2013. Triple-zeta polarized for solids, derived from def2-TZVP by re-optimization to remove diffuse linear dependencies. The CRYSTAL community standard. |
| **pob-DZVP** | CRYSTAL | Double-zeta companion. Laun, Vilela Oliveira, Bredow 2018 extended to fifth period with full-relativistic ECPs. |
| **pob-TZVP-rev2** | CRYSTAL | Vilela Oliveira et al. 2019. Revised version with improved performance for transition metals. **Use this over the original pob-TZVP for any new work.** |
| **MOLOPT (SZV/DZVP/TZVP/TZV2P)** | CP2K | Optimized by VandeVondele and Hutter for numerical stability in periodic systems with the GPW approach. The standard CP2K choice. |
| **dcm-TZVP** | CRYSTAL | Daga, Civalleri, Maschio 2020. System-specific re-optimizations for diamond, graphene, carbyne. Illustrative of the limits of all-purpose libraries. |
| **MOLOPT-aug (aug-MOLOPT-ae)** | CP2K | Augmented MOLOPT for excited states (TDDFT, BSE-GW). Recent. |

### 4.4 Numerical Atomic Orbitals and (L)APW+lo

| Basis / Tier | Code | Notes |
|------|------|-------|
| **FHI-aims tier 1** | FHI-aims | "Light" species defaults. Equivalent to a polarized DZ. Sub-meV precision for energy differences in many cases. |
| **FHI-aims tier 2** | FHI-aims | "Tight" species defaults. Roughly QZVP-quality but smaller. The FHI-aims production standard. |
| **FHI-aims tier 3 / 4** | FHI-aims | Reference quality. For benchmarking only. |
| **tier2_aug2** | FHI-aims | tier 2 plus two lowest-angular-momentum aug-cc functions. Required for excited-state and weakly-bound-anion calculations. |
| **SIESTA SZ / DZ / DZP / TZP** | SIESTA | Numerical pseudo-atomic orbitals with confinement. DZP is the standard SIESTA production basis. |
| **(L)APW+lo (WIEN2k, exciting)** | WIEN2k, exciting | Augmented plane waves. The all-electron gold standard for periodic DFT. Slower than PAW but no pseudopotential approximation. |

### 4.5 Ranking by Solid Type

#### Metals (elemental, alloys, intermetallics)

| Tier | Recommendation |
|------|----------------|
| Small (screening) | PAW with default cutoff (e.g. VASP `ENCUT` from POTCAR), dense k-mesh (Methfessel-Paxton smearing) |
| Medium (production) | PseudoDojo standard ONCV or VASP PAW with semicore states (`_sv`, `_pv`) for early TMs, cutoff 400 to 500 eV |
| Large (benchmark) | LAPW+lo (WIEN2k, exciting) with all-electron references |
| Magnetic systems | Always include semicore states. For 3d magnetism use `_pv` or `_sv` variants. For spin-orbit use fully-relativistic ONCV. |

Common pitfalls: forgetting semicore p-states in early transition metals (Sc through Cr), using too low a cutoff for d-block elements, and underconverged k-meshes. The Choudhary-Tavazza 2019 NIST study on 30,000+ materials provides good convergence heuristics by element.

#### Oxides (binary oxides, perovskites, transition-metal oxides)

| Tier | Recommendation |
|------|----------------|
| Small | PAW with O standard, screening-functional (PBEsol or SCAN) |
| Medium | **VASP PAW with O_h or O_s plus high cutoff (520 eV minimum, often 600 to 700 eV)**. The Materials Project standard is 520 eV with O standard. SSSP precision in QE. |
| Large | PAW plus DFT+U (Dudarev) or hybrid (HSE06) for correlated TM oxides. For benchmarking, all-electron LAPW. |
| Strongly correlated | DFT+U with carefully chosen U (e.g. via Materials Project tabulated values), or hybrid functional (HSE06, PBE0). |
| Defects, polarons | Hybrid functional (HSE06) and large supercell. |

Oxygen p-states are deep and the O 2s semicore needs a cutoff substantially above what alkali-halide systems require. The 520 eV Materials Project default is widely accepted.

#### Semiconductors (Si, Ge, III-V, II-VI, halide perovskites)

| Tier | Recommendation |
|------|----------------|
| Small | PAW with PBE, modest cutoff (300 to 400 eV). |
| Medium | **PAW or ONCV (PseudoDojo) with PBE for structure**, hybrid (HSE06) or G₀W₀ for band gaps. Include spin-orbit for heavy elements (GaAs, halide perovskites). |
| Large (band-gap accuracy) | **G₀W₀ on top of HSE06 or PBE**, with NAO tier 2 plus aug2 (FHI-aims) or LAPW plus high-energy local orbitals (HLO). |
| Excitonic absorption | BSE-GW with appropriate basis (aug-MOLOPT-ae in CP2K, tier2+aug2 in FHI-aims, LAPW+HLO+lo in exciting). |

For halide perovskites and other heavy-element semiconductors, scalar plus spin-orbit relativity is essential. Use fully-relativistic PseudoDojo or VASP PAW with `LSORBIT = .TRUE.`.

---

## 5. Composite "3c" Methods

These bundle a tailored basis with corrections (geometric counterpoise gCP, dispersion D3/D4, sometimes a modified short-range correction).

| Method | Functional | Basis | Year | Notes |
|--------|-----------|-------|------|-------|
| **HF-3c** | Hartree-Fock | MINIX | 2013 | Cheapest. Useful for very large systems where mean-field qualitative behavior is sufficient. |
| **PBEh-3c** | PBE0 (42% HF) | def2-mSVP | 2015 | Hybrid. Strong for noncovalent geometries. |
| **HSE-3c** | HSE | def2-mSVP | 2015 | Range-separated hybrid variant of PBEh-3c. |
| **B97-3c** | B97 GGA | def2-mTZVP | 2018 | Workhorse GGA. Especially recommended for transition-metal systems. |
| **r2SCAN-3c** | r²SCAN meta-GGA | def2-mTZVPP | 2021 | The current "Swiss army knife" recommendation. Excellent default for routine work, including geometries, thermochemistry, and noncovalent interactions. |
| **B3LYP-3c** | B3LYP | def2-mSVP | 2022 | Tailored for IR spectra (B3LYP frequencies are particularly accurate). |
| **ωB97X-3c** | ωB97X-V | **vDZP** | 2023 | Range-separated hybrid plus vDZP. Often the best 3c for thermochemistry and barrier heights. |

For vibe-qc, providing first-class support for the modified basis sets `def2-mSVP`, `def2-mTZVP`, `def2-mTZVPP`, and `vDZP`, plus the gCP and D4 corrections, is the path to making these composite methods turnkey. The published 3c benchmark MAEs on GMTKN55 are competitive with much more expensive calculations.

---

## 6. Property-Specific Basis Sets

These are critical to know about and to support, but they are not "general-purpose" so they do not appear in the main rankings:

| Property | Basis family | Reference |
|----------|--------------|-----------|
| NMR shielding | **pcS-n** (Jensen) | Jensen 2008, J. Chem. Theory Comput. 4, 719 |
| NMR J-coupling | **pcJ-n** (Jensen) | Jensen 2006, J. Chem. Theory Comput. 2, 1360 |
| NMR (relativistic) | **x2c-TZVPall-s** (Franzke-Weigend) | Franzke et al. 2019, Phys. Chem. Chem. Phys. 21, 16658 |
| EPR (g-tensor, hyperfine) | **EPR-II, EPR-III, x2c-TZVPall-s** | Barone 1996; Franzke et al. |
| Core-level X-ray (XPS, NEXAFS) | **cc-pCVnZ, aug-cc-pCVnZ, ccX-DK** | Peterson et al.; Hanson-Heine et al. |
| Polarizability, hyperpolarizability | **Sadlej, aug-cc-pVnZ, LPolX** | Sadlej 1988; Bauernschmitt-Ahlrichs 1996 |
| Excited states (TDDFT) | aug-cc-pVTZ, ma-def2-TZVPP, aug-pcseg-2 | Many |

---

## 7. Master Ranking Table

Compact view of the recommended sets at each tier and category. "Open" indicates the basis is freely available from the Basis Set Exchange (basissetexchange.org) or built in to common open-source codes.

| Domain | Small (fast) | Medium (production) | Large (benchmark) | Open |
|--------|--------------|---------------------|-------------------|------|
| Organic molecules | def2-SVP, B97-3c | **def2-TZVP** | def2-QZVPP, aug-cc-pVQZ | yes |
| Metal-organic | def2-SVP plus def2-ECP | **def2-TZVPP** plus def2-ECP | def2-QZVPP, ANO-RCC-VTZP (multiref) | yes |
| Inorganic main-group | def2-SVP | def2-TZVPPD | aug-cc-pV(Q+d)Z, def2-QZVPPD | yes |
| Biomolecules | r2SCAN-3c, B97-3c | def2-TZVP plus D4 | DLPNO-CCSD(T)/def2-TZVPP | yes |
| Pharma / drug-like | ωB97X-3c (vDZP) | def2-TZVPPD | ωB97M-V/def2-QZVPP | yes |
| Excited states | def2-SVPD | ma-def2-TZVPP, aug-pcseg-2 | aug-cc-pVQZ, tier2+aug2 | yes |
| Multireference | ANO-RCC-VDZP | ANO-RCC-VTZP | ANO-RCC-VQZP | yes |
| Heavy elements (relativistic) | x2c-SVPall, dhf-SVP | x2c-TZVPall, dhf-TZVPP | x2c-QZVPPall | yes |
| Solids: metals (PW) | PseudoDojo standard | **PseudoDojo stringent / SSSP precision** | LAPW+lo (WIEN2k) | yes (most) |
| Solids: oxides (PW) | VASP PAW (520 eV) | **VASP PAW plus HSE06**, SSSP precision | LAPW+lo, all-electron | mostly |
| Solids: semiconductors (PW) | PAW plus PBE | **PAW plus G₀W₀ on HSE06** | LAPW+HLO, BSE-GW | mostly |
| Solids: Gaussian periodic | pob-DZVP | **pob-TZVP-rev2**, MOLOPT TZVP | dcm-TZVP (system-specific) | yes |
| Solids: NAO all-electron | FHI-aims tier 1 (light) | **FHI-aims tier 2 (tight)** | FHI-aims tier 3, LAPW+lo | yes |

Bold entries are the "if you support exactly one thing in this category, support this" recommendations.

---

## 8. Implications for vibe-qc Basis Set Development

Concrete suggestions, in priority order:

1. **Adopt the Basis Set Exchange (BSE) format as the canonical input.** All major basis sets are available there in JSON, NWChem, Gaussian, and Turbomole formats. Avoid hand-coding basis sets.

2. **Native first-class support for def2.** SVP, TZVP, TZVPP, QZVP, QZVPP, plus ECPs (def2-ECP, dhf-ECP). This single family covers 80% of practical molecular DFT.

3. **Native support for cc-pVnZ and aug-cc-pVnZ** for n in {D, T, Q, 5}. Required for any post-HF user. Generally contracted, so the integral code needs to handle that efficiently.

4. **Native support for pcseg-n** for n in {0, 1, 2, 3, 4}. The pure-DFT optimal choice. Segmented, easy to implement.

5. **Pople family for legacy compatibility:** STO-3G, 3-21G, 6-31G, 6-31G(d), 6-31G(d,p), 6-31+G(d,p), 6-31++G(d,p), 6-311G(d,p), 6-311+G(2d,p), 6-311+G(2df,p). Surface a UI warning that 6-311G* is not a true triple-zeta and that pcseg-2 or def2-TZVP are preferred.

6. **Composite 3c methods.** Require the modified bases (def2-mSVP, def2-mTZVP, def2-mTZVPP, vDZP) plus the gCP and D3/D4 correction code paths. These are arguably the highest-value additions for routine users today.

7. **ANO-RCC and ANO-R** for multireference users. Generally contracted, requires CASSCF/CASPT2/NEVPT2 in the code base.

8. **Relativistic counterparts** of def2: x2c-SVPall, x2c-TZVPall, x2c-QZVPall, plus dhf-SVP, dhf-TZVP, dhf-QZVP. Needs a relativistic Hamiltonian (X2C, DKH2, ZORA) implementation.

9. **For solids:** start with **PseudoDojo ONCV** and **MOLOPT** for Gaussian-orbital periodic. PAW comes later because the PAW projector formalism is a substantial implementation effort. **pob-TZVP-rev2** is a low-effort win for any code that already does periodic Gaussians.

10. **STO-3G stays in the catalog.** For pedagogy, for SCF guesses, and as a sanity-check stress test of the code. Just label it appropriately in the UI.

The single highest-impact addition for a new code is robust support for the **def2 family plus its ECPs**, paired with **D4 dispersion** and **gCP**. This combination produces the 3c composite methods, gives users access to the modern DFT workflow, and unlocks transition-metal chemistry through the def2-ECP heavy-element treatment.

---

## 9. Key References

### Reviews and Best-Practice Guides

- Bursch, M., Mewes, J.-M., Hansen, A., Grimme, S. *Best-Practice DFT Protocols for Basic Molecular Computational Chemistry.* Angew. Chem. Int. Ed. **2022**, 61, e202205735. DOI: 10.1002/anie.202205735
- Karton, A. *Good Practices in Database Generation for Benchmarking Density Functional Theory.* WIREs Comput. Mol. Sci. **2025**, 15, e1737. DOI: 10.1002/wcms.1737
- Goerigk, L., Hansen, A., Bauer, C., Ehrlich, S., Najibi, A., Grimme, S. *A look at the density functional theory zoo with the advanced GMTKN55 database.* Phys. Chem. Chem. Phys. **2017**, 19, 32184. DOI: 10.1039/C7CP04913G
- Pitman, S. J., Evans, A. K., Ireland, R. T., Lempriere, F., McKemmish, L. K. *Benchmarking Basis Sets for DFT Thermochemistry: Why Unpolarized Basis Sets and the Polarized 6-311G Family Should Be Avoided.* J. Phys. Chem. A **2023**, 127, 10295. DOI: 10.1021/acs.jpca.3c05573 (and arXiv:2409.03964 follow-up)

### Pople Family

- Hehre, W. J., Stewart, R. F., Pople, J. A. *Self-Consistent Molecular-Orbital Methods. I. Use of Gaussian Expansions of Slater-Type Atomic Orbitals.* J. Chem. Phys. **1969**, 51, 2657 (STO-nG)
- Binkley, J. S., Pople, J. A., Hehre, W. J. *Self-Consistent Molecular Orbital Methods. 21. Small Split-Valence Basis Sets for First-Row Elements.* J. Am. Chem. Soc. **1980**, 102, 939 (3-21G)
- Hariharan, P. C., Pople, J. A. *The Influence of Polarization Functions on Molecular Orbital Hydrogenation Energies.* Theor. Chim. Acta **1973**, 28, 213 (6-31G(d,p))
- Krishnan, R., Binkley, J. S., Seeger, R., Pople, J. A. *Self-Consistent Molecular Orbital Methods. XX. A Basis Set for Correlated Wave Functions.* J. Chem. Phys. **1980**, 72, 650 (6-311G)

### Dunning Correlation-Consistent

- Dunning, T. H. Jr. *Gaussian Basis Sets for Use in Correlated Molecular Calculations. I.* J. Chem. Phys. **1989**, 90, 1007 (cc-pVnZ)
- Kendall, R. A., Dunning, T. H. Jr., Harrison, R. J. *Electron Affinities of the First-Row Atoms Revisited.* J. Chem. Phys. **1992**, 96, 6796 (aug-cc-pVnZ)
- Dunning, T. H. Jr., Peterson, K. A., Wilson, A. K. *Gaussian Basis Sets for Use in Correlated Molecular Calculations. X. The Atoms Aluminum through Argon Revisited.* J. Chem. Phys. **2001**, 114, 9244 (cc-pV(n+d)Z)

### Karlsruhe def2 Family

- Weigend, F., Ahlrichs, R. *Balanced Basis Sets of Split Valence, Triple Zeta Valence and Quadruple Zeta Valence Quality for H to Rn: Design and Assessment of Accuracy.* Phys. Chem. Chem. Phys. **2005**, 7, 3297 (def2 series)
- Weigend, F. *Accurate Coulomb-Fitting Basis Sets for H to Rn.* Phys. Chem. Chem. Phys. **2006**, 8, 1057 (def2/J auxiliary)
- Rappoport, D., Furche, F. *Property-Optimized Gaussian Basis Sets for Molecular Response Calculations.* J. Chem. Phys. **2010**, 133, 134105 (def2-XVPD diffuse-augmented)

### Jensen Polarization-Consistent

- Jensen, F. *Polarization Consistent Basis Sets: Principles.* J. Chem. Phys. **2001**, 115, 9113 (pc-n)
- Jensen, F. *Polarization Consistent Basis Sets. III. The Importance of Diffuse Functions.* J. Chem. Phys. **2002**, 117, 9234 (aug-pc-n)
- Jensen, F. *Unifying General and Segmented Contracted Basis Sets.* J. Chem. Theory Comput. **2014**, 10, 1074 (pcseg-n)

### Relativistic and ANO

- Pollak, P., Weigend, F. *Segmented Contracted Error-Consistent Basis Sets of Double- and Triple-ζ Valence Quality for One- and Two-Component Relativistic All-Electron Calculations.* J. Chem. Theory Comput. **2017**, 13, 3696 (x2c-XVPall)
- Franzke, Y. J., Spiske, L., Pollak, P., Weigend, F. *Segmented Contracted Error-Consistent Basis Sets of Quadruple-ζ Valence Quality for One- and Two-Component Relativistic All-Electron Calculations.* J. Chem. Theory Comput. **2020**, 16, 5658
- Roos, B. O., Lindh, R., Malmqvist, P.-Å., Veryazov, V., Widmark, P.-O. *New Relativistic ANO Basis Sets for Transition Metal Atoms.* J. Phys. Chem. A **2005**, 109, 6575 (ANO-RCC TM)
- Zobel, J. P., Widmark, P.-O., Veryazov, V. *The ANO-R Basis Set.* J. Chem. Theory Comput. **2020**, 16, 278

### Composite 3c Methods

- Sure, R., Grimme, S. *Corrected Small Basis Set Hartree-Fock Method for Large Systems.* J. Comput. Chem. **2013**, 34, 1672 (HF-3c)
- Grimme, S., Brandenburg, J. G., Bannwarth, C., Hansen, A. *Consistent Structures and Interactions by Density Functional Theory with Small Atomic Orbital Basis Sets.* J. Chem. Phys. **2015**, 143, 054107 (PBEh-3c)
- Brandenburg, J. G., Bannwarth, C., Hansen, A., Grimme, S. *B97-3c: A Revised Low-Cost Variant of the B97-D Density Functional Method.* J. Chem. Phys. **2018**, 148, 064104
- Grimme, S., Hansen, A., Ehlert, S., Mewes, J.-M. *r²SCAN-3c: A "Swiss Army Knife" Composite Electronic-Structure Method.* J. Chem. Phys. **2021**, 154, 064103
- Müller, M., Hansen, A., Grimme, S. *ωB97X-3c: A Composite Range-Separated Hybrid DFT Method with a Molecule-Optimized Polarized Valence Double-ζ Basis Set.* J. Chem. Phys. **2023**, 158, 014103 (vDZP)
- Wagen, C. C. *The vDZP Basis Set Is Effective for Many Density Functionals.* Rowan publication, **2024**. (TorsionNet206 benchmarks)

### Periodic Gaussian Basis Sets

- Peintinger, M. F., Vilela Oliveira, D., Bredow, T. *Consistent Gaussian Basis Sets of Triple-Zeta Valence with Polarization Quality for Solid-State Calculations.* J. Comput. Chem. **2013**, 34, 451 (pob-TZVP)
- Vilela Oliveira, D., Laun, J., Peintinger, M. F., Bredow, T. *BSSE-Correction Scheme for Consistent Gaussian Basis Sets of Double- and Triple-Zeta Valence with Polarization Quality for Solid-State Calculations.* J. Comput. Chem. **2019**, 40, 2364 (pob-TZVP-rev2)
- Laun, J., Vilela Oliveira, D., Bredow, T. *Consistent Gaussian Basis Sets of Double- and Triple-Zeta Valence with Polarization Quality of the Fifth Period for Solid-State Calculations.* J. Comput. Chem. **2018**, 39, 1285
- VandeVondele, J., Hutter, J. *Gaussian Basis Sets for Accurate Calculations on Molecular Systems in Gas and Condensed Phases.* J. Chem. Phys. **2007**, 127, 114105 (MOLOPT)
- Daga, L. E., Civalleri, B., Maschio, L. *Gaussian Basis Sets for Crystalline Solids: All-Purpose Basis Set Libraries vs System-Specific Optimizations.* J. Chem. Theory Comput. **2020**, 16, 2192 (dcm-TZVP)

### Plane Wave and PAW Pseudopotentials

- Blöchl, P. E. *Projector Augmented-Wave Method.* Phys. Rev. B **1994**, 50, 17953
- Kresse, G., Joubert, D. *From Ultrasoft Pseudopotentials to the Projector Augmented-Wave Method.* Phys. Rev. B **1999**, 59, 1758 (VASP PAW)
- Hamann, D. R. *Optimized Norm-Conserving Vanderbilt Pseudopotentials.* Phys. Rev. B **2013**, 88, 085117 (ONCV)
- van Setten, M. J., Giantomassi, M., Bousquet, E., Verstraete, M. J., Hamann, D. R., Gonze, X., Rignanese, G.-M. *The PseudoDojo: Training and Grading a 85 Element Optimized Norm-Conserving Pseudopotential Table.* Comput. Phys. Commun. **2018**, 226, 39
- Schlipf, M., Gygi, F. *Optimization Algorithm for the Generation of ONCV Pseudopotentials.* Comput. Phys. Commun. **2015**, 196, 36 (SG15)
- Prandini, G., Marrazzo, A., Castelli, I. E., Mounet, N., Marzari, N. *Precision and Efficiency in Solid-State Pseudopotential Calculations.* npj Comput. Mater. **2018**, 4, 72 (SSSP)
- Lejaeghere, K. et al. *Reproducibility in Density Functional Theory Calculations of Solids.* Science **2016**, 351, aad3000 (Δ-factor benchmark across codes and PAW libraries)
- Garrity, K. F., Bennett, J. W., Rabe, K. M., Vanderbilt, D. *Pseudopotentials for High-Throughput DFT Calculations.* Comput. Mater. Sci. **2014**, 81, 446 (GBRV)

### Numerical Atomic Orbitals

- Blum, V., Gehrke, R., Hanke, F., Havu, P., Havu, V., Ren, X., Reuter, K., Scheffler, M. *Ab Initio Molecular Simulations with Numeric Atom-Centered Orbitals.* Comput. Phys. Commun. **2009**, 180, 2175 (FHI-aims)
- Soler, J. M. et al. *The SIESTA Method for Ab Initio Order-N Materials Simulation.* J. Phys.: Condens. Matter **2002**, 14, 2745

### Database / Resource

- Pritchard, B. P., Altarawy, D., Didier, B., Gibson, T. D., Windus, T. L. *New Basis Set Exchange.* J. Chem. Inf. Model. **2019**, 59, 4814. https://www.basissetexchange.org

---

## 10. Open Items for Discussion

For the vibe-qc design conversation, suggestions on issues that benefit from human judgment rather than just literature consensus:

1. **Generally vs segmented contracted integrals.** Most modern segmented codes (Turbomole, ORCA) handle def2/pcseg natively. Generally contracted (cc, ANO) need a different integral path. Decide which is the foundation and which is bolted on.

2. **PAW vs ONCV vs USPP for the periodic backend.** ONCV (PseudoDojo) is the cleanest open-source choice and has excellent accuracy. PAW is faster and what users expect, but the projector formalism is a much larger implementation lift. Recommendation: start with ONCV, add PAW later.

3. **Auxiliary basis sets for RI / DF / fitting.** def2/J, def2/JK, def2-TZVP/C, cc-pVnZ-RI are all standard. Whether to require these or auto-generate them at runtime is a design choice.

4. **How to expose composite methods.** Either as keyword shortcuts (`r2SCAN-3c`) that pull in basis, dispersion, and gCP corrections automatically, or as explicit user-specified combinations. Grimme's group's preference is the former.

5. **Default basis recommendation per task.** Modern codes (ORCA, Turbomole) do not pick a default for the user. A code that does (with user override) could be a major usability win, especially if the recommendation is grounded in benchmark data.
