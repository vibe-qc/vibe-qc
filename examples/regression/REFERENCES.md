# References — vibe-qc regression suite + examples

Bibliography of the test sets, basis sets, and benchmark papers the
regression suite (`examples/regression/`) and input examples
(`examples/molecules/`, `examples/solids/`) draw from. **This list is
the canonical citation source** — `MoleculeSpec.citation` and
`PeriodicSpec.citation` strings reference entries here by short tag.

The list is organised by use: **A** = test sets currently wired or
staged; **B** = test sets from the Section-8 survey not yet
incorporated, scoped for future work; **C** = basis-set families
either already in vibe-qc or planned; **D** = methodology
(functionals, composite methods).

When adding new geometries, cite the original paper *and* the
distribution channel (e.g. GMTKN55 GitHub repo, BEGDB, refdata) so
the data provenance is reproducible.

---

## A. Test sets currently in the suite

### S22 — small noncovalent dimers (5 of 22 currently wired)

* Jurečka, P.; Šponer, J.; Černý, J.; Hobza, P. *Phys. Chem. Chem.
  Phys.* **2006**, 8, 1985. DOI: [10.1039/B600027D](https://doi.org/10.1039/B600027D).
  — S22 origin, original CCSD(T) interaction-energy reference.
* Marshall, M. S.; Burns, L. A.; Sherrill, C. D. *J. Chem. Phys.*
  **2011**, 135, 194102. DOI: [10.1063/1.3659142](https://doi.org/10.1063/1.3659142).
  — S22A: revised CCSD(T)/CBS reference values; standard since 2011.
* Geometries from the **GMTKN55** distribution
  (https://github.com/grimme-lab/GMTKN55, subset `S22/`).

**All 22 S22 systems** ship as `MoleculeSpec` modules under
`examples/regression/systems/molecules/s22_*.py`. Generated from
the GMTKN55 distribution via
`examples/regression/scripts/generate_s22_specs.py` (re-run if
GMTKN55 updates upstream geometries; export
`VIBEQC_TESTSETS_DIR=$(pwd)/_testsets` and clone GMTKN55 there
first).

Active in `WAVE1_MOLECULE_CASES` (10 representative subset, one or
more per S22 family — HB / dispersion / mixed):

* HB: `s22_water_dimer` (#02), `s22_ammonia_dimer` (#01),
  `s22_formamide_dimer` (#04), `s22_formic_acid_dimer` (#03)
* Dispersion: `s22_methane_dimer` (#08), `s22_ethene_dimer` (#09),
  `s22_methane_benzene` (#10)
* Mixed: `s22_benzene_dimer_t` (#20), `s22_ethene_ethyne` (#16),
  `s22_benzene_water` (#17)

Data-only (the other 12 — too large for cheap CI but available as
input examples + future-wireable when the suite gets multi-component
machinery for interaction-energy benchmarking):

* HB: `s22_uracil_dimer_hb` (#05), `s22_2py_2ampy` (#06),
  `s22_adenine_thymine_wc` (#07)
* Dispersion: `s22_benzene_dimer_pd` (#11), `s22_pyrazine_dimer`
  (#12), `s22_uracil_dimer_stacked` (#13),
  `s22_indole_benzene_stack` (#14), `s22_adenine_thymine_stack`
  (#15)
* Mixed: `s22_benzene_ammonia` (#18), `s22_benzene_hcn` (#19),
  `s22_indole_benzene_t` (#21), `s22_phenol_dimer` (#22)

### X23 — molecular crystals (3 of 23 currently wired)

* Reilly, A. M.; Tkatchenko, A. *J. Chem. Phys.* **2013**, 139, 024705.
  DOI: [10.1063/1.4812819](https://doi.org/10.1063/1.4812819).
  — X23 origin (extended C21 to 23 systems).
* Otero-de-la-Roza, A.; Johnson, E. R. *J. Chem. Phys.* **2012**, 137,
  054103. DOI: [10.1063/1.4738961](https://doi.org/10.1063/1.4738961).
  — C21 precursor to X23.
* Dolgonos, G. A.; Hoja, J.; Boese, A. D. *Phys. Chem. Chem. Phys.*
  **2019**, 21, 24333. DOI: [10.1039/C9CP04488D](https://doi.org/10.1039/C9CP04488D).
  — X23 revised reference values with thermal expansion.

Wired specs: `x23_urea`, `x23_ice_ih`, `x23_benzene_crystal`.

Per-system structure references:
* **Urea**: Swaminathan, S.; Craven, B. M.; McMullan, R. K. *Acta
  Cryst. B* **1984**, 40, 300. CCDC URECRA08 / ICSD #29585.
* **Ice Ih lattice**: Röttger, K.; Endriss, A.; Ihringer, J.; Doyle,
  S.; Kuhs, W. F. *Acta Cryst. B* **1994**, 50, 644.
  Proton-ordered XI approximant: Hayward, J. A.; Reimers, J. R.
  *J. Chem. Phys.* **1997**, 106, 1518.
* **Benzene crystal**: Bacon, G. E.; Curry, N. A.; Wilson, S. A.
  *Proc. R. Soc. A* **1964**, 279, 98 (138 K neutron diffraction).

### Existing periodic systems (LiH/NaCl/MgO/Al2O3 + Ne FCC + pob-TZVP set)

* Peintinger, M. F.; Vilela Oliveira, D.; Bredow, T. *J. Comput.
  Chem.* **2013**, 34, 451. DOI: [10.1002/jcc.23153](https://doi.org/10.1002/jcc.23153).
  — pob-TZVP basis-set paper; SI Tables 1-2 give PW1PW + HF
  reference total energies for ~70 ionic / covalent crystals at the
  experimental geometries listed in Tables 4-9 of the main paper.
  **The authoritative reference for any pob-TZVP periodic test once
  vibe-qc gets multi-k + PW1PW**.
* Vilela Oliveira, D.; Laun, J.; Peintinger, M. F.; Bredow, T.
  *J. Comput. Chem.* **2019**, 40, 2364.
  DOI: [10.1002/jcc.26013](https://doi.org/10.1002/jcc.26013).
  — pob-TZVP-rev2 + pob-DZVP-rev2; updated geometries for the same
  systems with the BSSE-corrected basis. Tables 2-4.
* Laun, J.; Vilela Oliveira, D.; Bredow, T. *J. Comput. Chem.*
  **2018**, 39, 1285. — pob-DZVP and pob-TZVP for the fifth period
  (relativistic ECPs).

Wired specs (pob-TZVP test-set seed, geometries from Peintinger 2013
Tables 4 + 8, published `published_energy_ha` PW1PW + HF from SI
Tables 1 + 2):

* Rocksalt Fm-3m (existing): `lih_rocksalt`, `nacl_rocksalt`,
  `mgo_rocksalt` — `expected/*.json` populated with both PW1PW and
  HF published values as advisory references.
* Rocksalt Fm-3m (new, not in `WAVE1_PERIODIC_CASES` yet — data ready
  for activation when prerequisites land): `lif_rocksalt`,
  `naf_rocksalt`, `kf_rocksalt`, `cao_rocksalt`, `nah_rocksalt`,
  `kh_rocksalt`.
* Zincblende F-43m (new): `bn_zincblende`, `sic_zincblende_b` —
  PW1PW reference only (no HF entry in Table 2 for these).
* Diamond cubic Fd-3m (new): `diamond_c` (pure C), `silicon` (pure
  Si). Same skeleton as zincblende but single-element sublattices.
* Fluorite Fm-3m, A·B₂ (new, 2026-05-10): `caf2_fluorite`,
  `srf2_fluorite`, `baf2_fluorite`. First A·B₂-stoichiometry family
  in the suite (12-atom conventional cell vs the 8-atom AB rocksalt
  and zincblende cells). Geometries from Wyckoff *Crystal Structures*
  Vol. 1, 2nd ed. (Wiley 1963), p. 240 — room-temperature
  experimental lattice constants 5.463 / 5.800 / 6.200 Å. All three
  appear in the Peintinger 2013 SI compound list; published PW1PW /
  HF totals can be backfilled into `expected/*.json` once read off
  the SI. CaF₂ is STO-3G compatible; SrF₂ / BaF₂ require pob-TZVP-
  rev2 (Sr / Ba past STO-3G horizon). The Ca / Sr / Ba cation-only
  series is a clean basis-transferability stress test —
  isostructural, single-element variation.

Each new spec module's docstring quotes the published PW1PW total in
Ha (where available) and cites the Peintinger 2013 SI Table 1 (or HF
Table 2 where PW1PW is unavailable). Strict numerical match against
these values needs the four prerequisites in `docs/roadmap.md`
§"Test-set expansion (queued)" §"Prerequisites in priority order" —
most critically vibe-qc PW1PW (currently absent) and multi-k
periodic SCF. **CRYSTAL14 via vq** (`run-crystal.sh` wrapper on
compute-reference, 2026-05-10) now provides an executable path to produce the
reference numbers ourselves at the pob-TZVP / PW1PW level — the
PW1PW gap on the vibe-qc side is the last blocker for strict
parity.

### Existing molecular reference geometries

* Szabo, A.; Ostlund, N. S. *Modern Quantum Chemistry: Introduction
  to Advanced Electronic Structure Theory*, McGraw-Hill **1989**.
  — H2O standard reference geometry used in `h2o`, `h2o2_dimer`,
  and tests/conftest.py.
* H2 R_e from NIST diatomic spectroscopic constants, ground-state
  X¹Σ_g⁺.

---

## B. Test sets surveyed but not yet wired

These were identified in the basis-set / test-set report as
high-leverage targets but require additional infrastructure (multi-
component reaction-energy machinery, separate downloads, or
functional support not yet in vibe-qc) before they're useful in the
suite. Wave-2+ work.

### Aggregator repositories

* **GMTKN55** — Goerigk, L.; Hansen, A.; Bauer, C.; Ehrlich, S.;
  Najibi, A.; Grimme, S. *Phys. Chem. Chem. Phys.* **2017**, 19, 32184.
  DOI: [10.1039/C7CP04913G](https://doi.org/10.1039/C7CP04913G).
  Repo: https://github.com/grimme-lab/GMTKN55. 1505 reaction
  energies / 2462 single-points across 55 subsets.
* **diet-150-GMTKN55** — Gould, T. *J. Chem. Theory Comput.* **2018**,
  14, 5252. DOI: [10.1021/acs.jctc.8b00748](https://doi.org/10.1021/acs.jctc.8b00748).
  Genetic-algorithm-reduced GMTKN55 for fast benchmarks.
* **ACCDB** — Morgante, P.; Peverati, R. *J. Comput. Chem.* **2019**,
  40, 839. DOI: [10.1002/jcc.25761](https://doi.org/10.1002/jcc.25761).
  Repo: https://github.com/peverati/ACCDB. 44,931 reference points
  bundling MGCDB84 + GMTKN55 + Minnesota 2015B + DP284 + W4-17 +
  Metals&EE.
* **ASCDB** — Morgante, P.; Peverati, R. *Phys. Chem. Chem. Phys.*
  **2019**, 21, 19092. DOI: [10.1039/C9CP03211H](https://doi.org/10.1039/C9CP03211H).
  Statistically-reduced ACCDB (350 geometries, 200 datapoints).
  Repo: https://github.com/peverati/ASCDB.
* **GSCDB137** — Liang, J. et al. *J. Chem. Theory Comput.* (2024
  submission). Repo: https://github.com/JiashuLiang/GSCDB. 137
  datasets / 8377 entries; modern revision of MGCDB84 with refreshed
  references.
* **MGCDB84** — Mardirossian, N.; Head-Gordon, M. *Mol. Phys.*
  **2017**, 115, 2315. DOI: [10.1080/00268976.2017.1333644](https://doi.org/10.1080/00268976.2017.1333644).
* **refdata** (Otero-de-la-Roza) — https://github.com/aoterodelaroza/refdata.
  Curated noncovalent benchmarks in xyz + .din format.
* **Truhlar Minnesota Database 2.0** — https://comp.chem.umn.edu/db/.

### Atomization / total-energy benchmarks

* **W4-17** — Karton, A.; Sylvetsky, N.; Martin, J. M. L. *J. Comput.
  Chem.* **2017**, 38, 2063. DOI: [10.1002/jcc.24854](https://doi.org/10.1002/jcc.24854).
* **W4-11** — Karton, A.; Daon, S.; Martin, J. M. L. *Chem. Phys.
  Lett.* **2011**, 510, 165. DOI: [10.1016/j.cplett.2011.05.007](https://doi.org/10.1016/j.cplett.2011.05.007).
* **NIST CCCBDB** — https://cccbdb.nist.gov, DOI 10.18434/T47C7Z.
  Experimental thermochemistry for 2186 small molecules.

### Noncovalent (additional)

* **S66 / S66x8 / S66x10** — Řezáč, J.; Riley, K. E.; Hobza, P.
  *J. Chem. Theory Comput.* **2011**, 7, 2427. DOI: [10.1021/ct2002946](https://doi.org/10.1021/ct2002946).
  Revised: Kříž et al., *Phys. Chem. Chem. Phys.* **2022**, 24, 14794.
  DOI: [10.1039/D2CP03938A](https://doi.org/10.1039/D2CP03938A).
* **A24** — Řezáč, J.; Hobza, P. *J. Chem. Theory Comput.* **2013**,
  9, 2151. DOI: [10.1021/ct400057w](https://doi.org/10.1021/ct400057w).
* **NCI Atlas** — https://github.com/Honza-R/NCI-atlas.
* **3B69** (3-body) — Řezáč, J.; Huang, Y.; Hobza, P.; Beran, G. J. O.
  *J. Chem. Theory Comput.* **2015**, 11, 3065. DOI: [10.1021/acs.jctc.5b00281](https://doi.org/10.1021/acs.jctc.5b00281).
* **XB18 / XB51** (halogen bonds) — Kozuch, S.; Martin, J. M. L.
  *J. Chem. Theory Comput.* **2013**, 9, 1918. DOI: [10.1021/ct301064t](https://doi.org/10.1021/ct301064t).
* **BEGDB** — Řezáč, J. et al. *Collect. Czech. Chem. Commun.*
  **2008**, 73, 1261. http://www.begdb.org.

### Transition-metal benchmarks

* **MOR41** (closed-shell organometallics) — Dohm, S.; Hansen, A.;
  Steinmetz, M.; Grimme, S.; Checinski, M. P. *J. Chem. Theory
  Comput.* **2018**, 14, 2596. DOI: [10.1021/acs.jctc.7b01183](https://doi.org/10.1021/acs.jctc.7b01183).
  https://www.chemie.uni-bonn.de/grimme/de/software/mor41.
* **ROST61** (open-shell organometallics) — Maurer, L. R.; Bursch,
  M.; Grimme, S.; Hansen, A. *J. Chem. Theory Comput.* **2021**, 17,
  6134. DOI: [10.1021/acs.jctc.1c00659](https://doi.org/10.1021/acs.jctc.1c00659).
* **MOBH35 / revMOBH35 / MOBH28** — Iron, M. A.; Janes, T.
  *J. Phys. Chem. A* **2019**, 123, 3761. DOI: [10.1021/acs.jpca.9b01546](https://doi.org/10.1021/acs.jpca.9b01546).
  Revised: Semidalas, E.; Martin, J. M. L. *J. Chem. Theory Comput.*
  **2022**, 18, 883. DOI: [10.1021/acs.jctc.1c01126](https://doi.org/10.1021/acs.jctc.1c01126).
  MOBH28 subset: Grotjahn, R.; Kaupp, M. *Israel J. Chem.* **2023**,
  63, e202200021. DOI: [10.1002/ijch.202200021](https://doi.org/10.1002/ijch.202200021).
* **WCCR10** — Weymuth, T.; Couzijn, E. P. A.; Chen, P.; Reiher, M.
  *J. Chem. Theory Comput.* **2014**, 10, 3092. DOI: [10.1021/ct500248h](https://doi.org/10.1021/ct500248h).
* **TMC151 / TMC34 / TMC32** — Chan, B.; Gill, P. M. W.; Kimura, M.
  *J. Chem. Theory Comput.* **2019**, 15, 3610. DOI: [10.1021/acs.jctc.9b00239](https://doi.org/10.1021/acs.jctc.9b00239).
* **MME55** (metalloenzyme model reactions) — Reinhardt, C. R. et al.
  *J. Chem. Theory Comput.* **2023**. DOI: [10.1021/acs.jctc.3c00568](https://doi.org/10.1021/acs.jctc.3c00568).
* **SSE17** (TM spin-state) — Drosou, M.; Pantazis, D. A. *Chem. Sci.*
  **2024**, 15. DOI: [10.1039/D4SC05471G](https://doi.org/10.1039/D4SC05471G).
* **tmQM** — Balcells, D.; Skjelstad, B. B. *J. Chem. Inf. Model.*
  **2020**, 60, 6135. DOI: [10.1021/acs.jcim.0c01041](https://doi.org/10.1021/acs.jcim.0c01041).
  https://github.com/uiocompcat/tmQM.

### Geometry benchmarks

* **ROT34** (rotational constants) — Risthaus, T.; Steinmetz, M.;
  Grimme, S. *J. Comput. Chem.* **2014**, 35, 1509. DOI: [10.1002/jcc.23649](https://doi.org/10.1002/jcc.23649).
  Note: not in GMTKN55 distribution despite occasional report claims;
  geometries are paper-only.
* **LB12** (long-bond test) — Goerigk, L.; Grimme, S. *J. Chem.
  Theory Comput.* **2010**, 6, 107. DOI: [10.1021/ct900489g](https://doi.org/10.1021/ct900489g).
* **HMGB11** (heavy main-group bonds) — Grimme, S.; Bannwarth, C.;
  Shushkov, P. *J. Chem. Theory Comput.* **2017**, 13, 1989.
  DOI: [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118).
* **MGAE109 / AE6 / LB1AE12** — Zhao, Y.; Schultz, N. E.; Truhlar,
  D. G. *J. Chem. Theory Comput.* **2005**, 2, 364.
  https://comp.chem.umn.edu/db/dbs/mgae109.html.

### Drug-like / pharma

* **TorsionNet206** — Behara, P. K. et al. *J. Phys. Chem. B*
  **2024**, 128, 7888. DOI: [10.1021/acs.jpcb.4c02888](https://doi.org/10.1021/acs.jpcb.4c02888).
  https://github.com/openforcefield/openff-torsionnet.
* **SPICE** — Eastman, P. et al. *Sci. Data* **2023**, 10, 11.
  https://github.com/openmm/spice-dataset.
* **BBI / SSI** — Burns, L. A. et al. *J. Chem. Phys.* **2017**, 147,
  161727. DOI: [10.1063/1.5001028](https://doi.org/10.1063/1.5001028).

### Molecular crystals (additional, beyond X23 currently wired)

* **ICE13** — Brandenburg, J. G.; Maas, T.; Grimme, S. *J. Chem.
  Phys.* **2015**, 142, 124104. DOI: [10.1063/1.4916067](https://doi.org/10.1063/1.4916067).
* **DMC-ICE13** — Della Pia, F.; Zen, A.; Alfè, D.; Michaelides, A.
  *J. Chem. Phys.* **2024**, 161, 064708. DOI: [10.1063/5.0219341](https://doi.org/10.1063/5.0219341).
* **POLY59** — incorporated in Dolgonos-Hoja-Boese 2019; see X23
  revised entry.
* **OMC25** — Gharakhanyan, V. et al. *Sci. Data* **2026**.
  https://huggingface.co/datasets/fairchem/OMC25 — 27M molecular
  crystal structures.

### Large-scale ML datasets

* **QM9** — Ramakrishnan, R.; Dral, P. O.; Rupp, M.; von Lilienfeld,
  O. A. *Sci. Data* **2014**, 1, 140022. DOI: [10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22).
* **MultiXC-QM9** — Nandi, S.; Vegge, T.; Bhowmik, A. *Sci. Data*
  **2023**, 10, 783. DOI: [10.1038/s41597-023-02690-2](https://doi.org/10.1038/s41597-023-02690-2).
* **ANI-1 / ANI-1ccx** — Smith, J. S. et al. *Sci. Data* **2017**, 4,
  170193 / *Nat. Commun.* **2019**, 10, 2903.

### Excited states

* **GW100** — van Setten, M. J. et al. *J. Chem. Theory Comput.*
  **2015**, 11, 5665. https://github.com/setten-mvs/GW100.

### Programmatic infrastructure

* **QCArchive (MolSSI)** — Smith, D. G. A. et al. *WIREs Comput.
  Mol. Sci.* **2021**, 11, e1491. DOI: [10.1002/wcms.1491](https://doi.org/10.1002/wcms.1491).
  https://qcarchive.molssi.org.

---

## C. Basis-set families (citation when used)

### Pople

* **STO-3G**: Hehre, W. J.; Stewart, R. F.; Pople, J. A. *J. Chem.
  Phys.* **1969**, 51, 2657.
* **3-21G**: Binkley, J. S.; Pople, J. A.; Hehre, W. J. *J. Am. Chem.
  Soc.* **1980**, 102, 939.
* **6-31G(d,p)**: Hariharan, P. C.; Pople, J. A. *Theor. Chim. Acta*
  **1973**, 28, 213.
* **6-311G**: Krishnan, R.; Binkley, J. S.; Seeger, R.; Pople, J. A.
  *J. Chem. Phys.* **1980**, 72, 650.

### Karlsruhe def2

* **def2 series**: Weigend, F.; Ahlrichs, R. *Phys. Chem. Chem. Phys.*
  **2005**, 7, 3297. DOI: [10.1039/B508541A](https://doi.org/10.1039/B508541A).
* **def2/J auxiliary**: Weigend, F. *Phys. Chem. Chem. Phys.* **2006**,
  8, 1057. DOI: [10.1039/B515623H](https://doi.org/10.1039/B515623H).
* **def2-XVPD diffuse**: Rappoport, D.; Furche, F. *J. Chem. Phys.*
  **2010**, 133, 134105.

### Dunning correlation-consistent

* **cc-pVnZ**: Dunning, T. H. Jr. *J. Chem. Phys.* **1989**, 90, 1007.
* **aug-cc-pVnZ**: Kendall, R. A.; Dunning, T. H. Jr.; Harrison, R. J.
  *J. Chem. Phys.* **1992**, 96, 6796.
* **cc-pV(n+d)Z**: Dunning, T. H. Jr.; Peterson, K. A.; Wilson, A. K.
  *J. Chem. Phys.* **2001**, 114, 9244.

### Jensen polarization-consistent

* **pc-n**: Jensen, F. *J. Chem. Phys.* **2001**, 115, 9113.
* **pcseg-n**: Jensen, F. *J. Chem. Theory Comput.* **2014**, 10, 1074.

### Periodic-Gaussian (CRYSTAL family)

* **pob-TZVP**: Peintinger 2013 (cited above in Section A).
* **pob-TZVP-rev2 / pob-DZVP-rev2**: Vilela Oliveira 2019 (cited
  above).

### MOLOPT (CP2K)

* **MOLOPT**: VandeVondele, J.; Hutter, J. *J. Chem. Phys.* **2007**,
  127, 114105.

### Composite-method modified bases

* **vDZP** (in ωB97X-3c): Müller, M.; Hansen, A.; Grimme, S. *J. Chem.
  Phys.* **2023**, 158, 014103.

### Programmatic source

* **Basis Set Exchange**: Pritchard, B. P.; Altarawy, D.; Didier, B.;
  Gibson, T. D.; Windus, T. L. *J. Chem. Inf. Model.* **2019**, 59,
  4814. DOI: [10.1021/acs.jcim.9b00725](https://doi.org/10.1021/acs.jcim.9b00725).
  https://www.basissetexchange.org.

---

## D. Methodology — functionals + composite methods

### PW1PW (the pob-TZVP reference functional)

* **Bredow, T.; Gerson, A. R.** *Effect of exchange and correlation
  on bulk properties of MgO, NiO, and CoO.* *Phys. Rev. B* **2000**,
  61, 5194-5201. DOI: [10.1103/PhysRevB.61.5194]
  (https://doi.org/10.1103/PhysRevB.61.5194). **Canonical proposal
  paper**; the functional was originally called "HF+PWGGA" in
  §III.D ("Hybrid approach based on the PWGGA method", p. 5199-5200).
  The 20 % HF mixing was empirically optimized on MgO/NiO/CoO bulk
  properties and matches B3LYP's mixing parameter coincidentally.

The "PW1PW" name came into use later (Peintinger 2013 onward) for
the canonical 1-parameter form. See
[`reference_pw1pw_functional.md`](../../user-memory/reference_pw1pw_functional.md)
in user-memory for the full attribution + formula breakdown:
`E_x = 0.20 · E_x^HF + 0.80 · E_x^PW91`, `E_c = E_c^PW91`. Not yet
in vibe-qc; prerequisite for strict numerical comparison vs the
Peintinger / Vilela Oliveira reference tables.

### B3LYP convention split (vibe-qc ships the ORCA definition)

"B3LYP" is a fixed hybrid recipe; codes differ only in *which*
Vosko-Wilk-Nusair parametrisation fills the LSDA-correlation
slot (Hertwig & Koch, Chem. Phys. Lett. 268, 345 (1997)) —
VWN5 (Ceperley-Alder fit) vs VWN-RPA (Gaussian's "VWN(III)").

* **vibe-qc `b3lyp` (= `b3lyp5`) / ORCA bare `B3LYP` / TURBOMOLE /
  CRYSTAL14 `B3LYP` keyword / CP2K `B3LYP` preset / PySCF
  `b3lyp5`** = VWN5 (libxc id 475, `XC_HYB_GGA_XC_B3LYP5`).
  Empirically pinned on H2/STO-3G at 1.4 bohr: CRYSTAL14
  -1.1586001474 Ha == libxc B3LYP5 -1.1586001482 Ha; ORCA bare
  B3LYP -1.158669 (DefGrid2).
* **vibe-qc `b3lyp/g` (= `b3lypg`) / Gaussian `B3LYP` / PySCF
  `b3lyp` and `b3lypg` / Psi4 `b3lyp` / ORCA `B3LYP/G` / GPAW
  `B3LYP`** = VWN-RPA (libxc id 402, `XC_HYB_GGA_XC_B3LYP`),
  -1.1654009284 Ha on the same system (gap = 0.19*(E_c[VWN_RPA]
  - E_c[VWN5]) = -6.8 mHa).

The cross-code parity layer pairs same-flavor spellings on both
sides: vibe-qc `b3lyp` <-> ORCA bare `B3LYP` / PySCF `b3lyp5` /
Psi4 `b3lyp5` / GPAW `HYB_GGA_XC_B3LYP5` / CRYSTAL `B3LYP` / CP2K
`B3LYP`; vibe-qc `b3lyp/g` / `b3lypg` <-> ORCA `B3LYP/G` / PySCF
`b3lyp` / Psi4 `b3lyp` / GPAW `B3LYP`. (The Psi4 and GPAW rows
used to pair bare names — silently crossing the flavor gap; fixed
2026-06-11.) CRYSTAL14 and CP2K cannot express the VWN-RPA
flavor, so the Gaussian-flavor spellings are deliberately
unsupported in those two runners. See the CHANGELOG entry and
`tests/test_b3lyp_convention.py`.

### Dispersion corrections (not yet in vibe-qc)

* **D3(BJ)**: Grimme, S.; Antony, J.; Ehrlich, S.; Krieg, H.
  *J. Chem. Phys.* **2010**, 132, 154104. Becke-Johnson damping:
  Grimme, S.; Ehrlich, S.; Goerigk, L. *J. Comput. Chem.* **2011**,
  32, 1456.
* **D4**: Caldeweyher, E.; Bannwarth, C.; Grimme, S. *J. Chem. Phys.*
  **2017**, 147, 034112.
* **gCP** (geometric counterpoise): Kruse, H.; Grimme, S. *J. Chem.
  Phys.* **2012**, 136, 154101.

Required for any quantitative match against X23 / S22 reference
interaction or lattice energies; the wave-2 dispersion-correction
addition unlocks strict comparison.

### Composite "3c" methods

* **HF-3c**: Sure, R.; Grimme, S. *J. Comput. Chem.* **2013**, 34,
  1672.
* **PBEh-3c**: Grimme, S.; Brandenburg, J. G.; Bannwarth, C.;
  Hansen, A. *J. Chem. Phys.* **2015**, 143, 054107.
* **B97-3c**: Brandenburg, J. G.; Bannwarth, C.; Hansen, A.; Grimme,
  S. *J. Chem. Phys.* **2018**, 148, 064104.
* **r²SCAN-3c**: Grimme, S.; Hansen, A.; Ehlert, S.; Mewes, J.-M.
  *J. Chem. Phys.* **2021**, 154, 064103.
* **B3LYP-3c**: see Bursch et al. 2022.
* **ωB97X-3c**: Müller-Hansen-Grimme 2023 (cited above under vDZP).

### Best-practice / methodology reviews

* Bursch, M.; Mewes, J.-M.; Hansen, A.; Grimme, S. *Best-Practice
  DFT Protocols for Basic Molecular Computational Chemistry.*
  *Angew. Chem. Int. Ed.* **2022**, 61, e202205735.
  DOI: [10.1002/anie.202205735](https://doi.org/10.1002/anie.202205735).
* Karton, A. *Good Practices in Database Generation for Benchmarking
  DFT.* *WIREs Comput. Mol. Sci.* **2025**, 15, e1737.
  DOI: [10.1002/wcms.1737](https://doi.org/10.1002/wcms.1737).
* Pitman, S. J. et al. *J. Phys. Chem. A* **2023**, 127, 10295.
  DOI: [10.1021/acs.jpca.3c05573](https://doi.org/10.1021/acs.jpca.3c05573).
  Plus arXiv:2409.03964 follow-up — basis-set thermochemistry
  benchmarks; recommends avoiding unpolarized DZ and 6-311G family.

---

## How to extend

When a new system is added to `examples/regression/systems/`:
1. Cite both the **structure** source (where the geometry came from)
   and the **reference** source (where the published energy / property
   came from) in `MoleculeSpec.citation` / `PeriodicSpec.citation`.
2. Add a one-line entry under §A here pointing back at the spec(s) it
   informs.
3. If the citation is new to this file, add it under §B (or §C/D as
   appropriate) with full bibliographic detail.

Out-of-scope-for-now items (multi-component reaction-energy machinery,
QCArchive ingestion, BSE auto-import, dispersion corrections,
PW1PW functional) are tracked in the project memory + the v0.8
roadmap, not here. This file is *bibliography*, not *roadmap*.
