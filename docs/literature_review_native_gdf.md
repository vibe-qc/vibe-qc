---
orphan: true
---

# Literature review, periodic GDF for vibe-qc

Authoritative reference for the native-GDF feature work. Commissioned
2026-05-09 from a separate Claude chat; the transient commissioning
brief was retired in the 2026-05-17 archive sweep. Pre-split implementation
notes are preserved in [Archived `handovers/HANDOVER_GDF_OUTSTANDING.md`](https://vibe-qc.com/docs/).
The results below resolve the open algorithmic question that blocked
slice 3c at the end of the previous session.

## Headline findings (read this if nothing else)

1. **PySCF's default for pure-GDF runs is now RSGDF** (Ye & Berkelbach
   2021), not the CCDF algorithm of Sun et al. 2017. RSGDF is **also
   strictly easier to implement** than CCDF, no compcell, no
   AO-pair multipoles, no inclusion-exclusion bookkeeping. **vibe-qc
   should implement RSGDF as the canonical periodic-GDF path.**

2. **Switch the Lpq fit from Cholesky to eigendecomposition +
   threshold** (Sun 2017 §III.B). Cholesky fails on larger aux bases
   even with a correct metric; truncating tiny eigenvalues
   (`linear_dep_threshold = 1e-9`) is the principled handling.

3. **DOI corrections to apply across the codebase docs**:
   - Sun et al. 2017 (Gaussian and plane-wave mixed density fitting):
     `10.1063/1.4998644` (NOT 1.5005184).
   - Ye & Berkelbach 2021 (RSGDF): `10.1063/5.0046617` in *JCP* 154,
     131104 (NOT JCTC 17, 4189).

## RSGDF algorithm (the new implementation target)

Range-separate the Coulomb kernel itself, not the densities:

$$
\frac{1}{|r-r'|} = \underbrace{\frac{\mathrm{erfc}(\omega |r-r'|)}{|r-r'|}}_{\text{SR}}
                 + \underbrace{\frac{\mathrm{erf}(\omega |r-r'|)}{|r-r'|}}_{\text{LR}}
$$

- **SR part**: decays as $\exp(-\omega^2 r^2)$ at large $r$. The lattice
  sum $\sum_T (P_0\,|\,\mathrm{erfc}(\omega \cdot)/|\cdot|\,|\,Q_T)$
  converges absolutely **with no compensation needed**. libint exposes
  this as `Operator::erfc_coulomb`, already used by vibe-qc in
  `compute_nuclear_erfc_lattice` ([cpp/src/lattice_integrals.cpp:144](../cpp/src/lattice_integrals.cpp)).
- **LR part**: $\widehat{\mathrm{erf}(\omega r)/r}(G) = (4\pi/G^2) \exp(-G^2/(4\omega^2))$.
  Decays as $\exp(-G^2/(4\omega^2))$ in reciprocal space → small G-mesh
  suffices. Computed analytically via Gaussian Fourier transforms.
- **Parameter $\omega$**: typical 0.1-0.4 bohr⁻¹. Recommended choice
  $\omega \approx \min(0.4, 0.5/r_{\rm cut})$ where $r_{\rm cut}$ is
  smallest nearest-neighbour distance (Ye 2021).
- **G = 0 handling**: omitted (charge-neutral assumption); divergent
  monopole-monopole piece cancels against the nuclear background. For
  2D/1D systems the treatment changes, defer to PySCF's
  `pyscf/pbc/df/aft.py` for the truncated-Coulomb low-dim version.

### Why RSGDF over CCDF for vibe-qc

- One unified kernel split applies to both 2c metric AND 3c tensor.
  No separate compcell construction, no per-AO-pair multipole
  bookkeeping.
- libint operators already exposed (`erfc_coulomb` available).
- Analytical Gaussian FTs are simple (closed form, no iterative
  schemes).
- Default in PySCF means our parity reference uses the same algorithm.
- Reported speedup: ~10× over CCDF for typical 3D solids (Ye 2021,
  Table II).
- Trade-off: one new parameter $\omega$ to tune; documented heuristic
  works for typical solids.

CCDF remains a fallback if $\omega$ choice is awkward (very large
lattice constants, low-dimensional systems). vibe-qc should ship
RSGDF as the v0.7.4+ default and treat CCDF as a research project.

### Final fit

After SR + LR assembly:
$J(\mathbf k) = U \Lambda U^\dagger$, drop modes with $\lambda <$
`linear_dep_threshold` (default $10^{-9}$ per Sun 2017 Fig. 2; the
choice matters, $10^{-12}$ reintroduces noise, $10^{-7}$
over-truncates), then form

$$ L^P_{\mu\nu}(\mathbf k_1, \mathbf k_2) = \sum_Q [\Lambda^{-1/2}]_{PQ}\, U^\dagger_{PR}\, T_{R, \mu\nu}(\mathbf k_1, \mathbf k_2). $$

This is the CDERI tensor consumed by the SCF driver.

## Validation oracle plan

The full lit-sweep §6 lays out three published reference suites
beyond PySCF for cross-validation:

| System | Method / basis | Source | DOI |
|---|---|---|---|
| MgO, NaCl, LiH (rock-salt) | HF / Gaussian, CRYSTAL | Doll 2002 | 10.1016/S0010-4655(01)00172-2 |
| LiH, LiF, MgO | HF + MP2 / multiple bases, CRYSCOR | Usvyat 2007 | 10.1063/1.2768359 |
| Diamond, LiF, MgO, BN | HF + MP2 / cc-pVnZ-solids, PySCF RSGDF | Ye & Berkelbach 2022 | 10.1021/acs.jctc.1c01245 |
| Diamond | HF, MP2, CCSD / plane-wave, VASP | Booth 2013 | 10.1038/nature11770 |

**Concrete test plan**: MgO HF/sto-3g vs PySCF (already have);
MgO HF/pob-tzvp vs Doll 2002 + Usvyat 2007 (independent oracles, real
basis); LiF HF/cc-pVDZ-solids vs Ye & Berkelbach 2022. If all three
agree to $10^{-4}$ Ha/cell, implementation is validated.

## Reference list with corrected DOIs

### Primary, required for the implementation

- **Sun et al. 2017** (CCDF + MDF). *J. Chem. Phys.* **147**, 164119.
  DOI: [10.1063/1.4998644](https://doi.org/10.1063/1.4998644).
  §II.A-§II.C derive the SR/LR split, compensating-charge construction,
  recovery formulas. §III.B + Fig. 2: linear-dependence calibration.
- **Ye & Berkelbach 2021** (RSGDF). *J. Chem. Phys.* **154**, 131104.
  DOI: [10.1063/5.0046617](https://doi.org/10.1063/5.0046617).
  The actual implementation target. Read first.
- **Ye, Berkelbach et al. 2021** (screening estimators companion).
  *J. Chem. Phys.* **155**, 124106. DOI:
  [10.1063/5.0064151](https://doi.org/10.1063/5.0064151). Tight
  distance-dependent estimators for SR integrals, read second.
- **Ye, Berkelbach et al. 2023** (PySCF production defaults).
  arXiv:2302.11307. Documents `cell.precision`-based threshold
  selection and k-point screening choices that the journal papers
  don't cover.

### Foundational, cite in code comments / docs

- **Whitten 1973** (original Coulomb-metric DF). *J. Chem. Phys.*
  **58**, 4496. DOI: [10.1063/1.1679103](https://doi.org/10.1063/1.1679103).
- **Dunlap, Connolly, Sabin 1979** (variational DF formulation; quadratic
  error property). *Int. J. Quantum Chem.* **16**, 81. DOI:
  [10.1002/qua.560160202](https://doi.org/10.1002/qua.560160202).
- **Eichkorn, Treutler, Öhm, Häser, Ahlrichs 1995** (modern atom-centred
  RI-J). *Chem. Phys. Lett.* **240**, 283. DOI:
  [10.1016/0009-2614(95)00621-A](https://doi.org/10.1016/0009-2614(95)00621-A).
- **Weigend 2008** (def2-universal-jkfit; what vibe-qc bundles).
  *J. Comput. Chem.* **29**, 167. DOI:
  [10.1002/jcc.20702](https://doi.org/10.1002/jcc.20702).

### Foundational, periodic Coulomb structure

- **Pisani, Dovesi, Roetti 1988** (periodic bipolar quartet expansion,
  Chapter II.4c). *Hartree-Fock Ab Initio Treatment of Crystalline
  Systems*, Lecture Notes in Chemistry 48. DOI:
  [10.1007/978-3-642-93385-1](https://doi.org/10.1007/978-3-642-93385-1).
- **Saunders, Freyria-Fava, Dovesi, Salasco, Roetti 1992**
  (Gaussian-density electrostatic potential, penetration, and spheropole;
  not the quartet dispatcher). *Mol. Phys.* **77**, 629.
  DOI: [10.1080/00268979200102671](https://doi.org/10.1080/00268979200102671).
- **Harris 1975** (zero-monopole decomposition for Gaussians).
  Chapter in *Theoretical Chemistry: Advances and Perspectives* vol. 1
  (Academic Press). Hard to fetch online.
- **Ewald 1921** (the original lattice-sum split). *Ann. Phys.* **369**,
  253. DOI: [10.1002/andp.19213690304](https://doi.org/10.1002/andp.19213690304).
  In German; modern restatement in Allen & Tildesley App. F.

### Alternative algorithms (for context)

- **VandeVondele et al. 2005** (GPW; CP2K's Quickstep).
  *Comput. Phys. Commun.* **167**, 103. DOI:
  [10.1016/j.cpc.2004.12.014](https://doi.org/10.1016/j.cpc.2004.12.014).
  Plane-wave-grid Coulomb instead of Gaussian aux. Different paradigm.
- **Burow, Sierka, Mohamed 2009** (Turbomole CFMM periodic).
  *J. Chem. Phys.* **131**, 214101. DOI:
  [10.1063/1.3267858](https://doi.org/10.1063/1.3267858). Continuous
  fast-multipole on Gaussian aux. Reference for the third independent
  algorithm.

### Aux basis design (pob-aux follow-up)

- **Stoychev, Auer, Neese 2017** (AutoAux). *J. Chem. Theory Comput.*
  **13**, 554. DOI: [10.1021/acs.jctc.6b01041](https://doi.org/10.1021/acs.jctc.6b01041).
  Generates aux from any orbital basis automatically.
- **Lehtola 2021** (AutoAux contraction via SVD). *J. Chem. Theory
  Comput.* **17**, 6886. DOI:
  [10.1021/acs.jctc.1c00607](https://doi.org/10.1021/acs.jctc.1c00607).
  Reduces AutoAux output size while preserving accuracy.
- **Hellweg, Rappoport 2015** (def2 aux design philosophy).
  *Phys. Chem. Chem. Phys.* **17**, 1010. DOI:
  [10.1039/c4cp04286g](https://doi.org/10.1039/c4cp04286g).
- **Ye & Berkelbach 2022** (cc-pVnZ-style solids bases + matched aux,
  pseudopotential). *J. Chem. Theory Comput.* **18**, 1595. DOI:
  [10.1021/acs.jctc.1c01245](https://doi.org/10.1021/acs.jctc.1c01245).
- **Peintinger, Vilela Oliveira, Bredow 2013** (pob-tzvp orbital
  basis). *J. Comput. Chem.* **34**, 451. DOI:
  [10.1002/jcc.23153](https://doi.org/10.1002/jcc.23153).
  No matching pob-aux exists; AutoAux + linear-dependence pruning is
  the recommended path for v0.7.x+. Hand-tuning a pob-aux remains
  paper-worthy follow-up.

### Validation oracles (HF energies on solids)

See "Validation oracle plan" table above for DOIs. Doll 2002, Usvyat
2007, Ye & Berkelbach 2022, Booth 2013.

### Loose ends, where the literature defers to source code

- Exact `drop_eta` / `linear_dep_threshold` defaults: PySCF computes
  from `cell.precision` rather than the values in the paper. Read
  `pyscf/pbc/df/df.py:make_modrho_basis` for current defaults.
- 2D/1D treatment of the $G=0$ Coulomb pole: see Sundararaman & Arias
  PRB **87**, 165122 (2013), DOI:
  [10.1103/PhysRevB.87.165122](https://doi.org/10.1103/PhysRevB.87.165122).
- k-point-dependent screening cutoffs: not in the journal papers; see
  arXiv:2302.11307.
