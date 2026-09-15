# GPAW PW-limit r2SCAN atomization energies — consolidated reference

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6³ · atom box 12.0 Å · energies kJ/mol per formula unit.

## Production results — 28 systems

| Compound | E_exp | GPAW | pob | VASP | Δ(G−pob) | Δ(G−VASP) | Outlier? |
|----------|------:|-----:|----:|-----:|---------:|----------:|----------|
| AgCl | 548.0 | 523.4 | 539.7 | 526.0 | −16.3 | −2.6 | pob |
| AlAs | 741.7 | 744.9 | 788.4 | 742.6 | **−43.5** | +2.3 | **pob** |
| AlN | 1143.6 | 1114.7 | 1104.6 | 1119.1 | +10.1 | −4.4 | pob |
| AlP | 776.2 | 815.8 | 816.4 | 820.2 | −0.6 | −4.4 | all agree |
| BN | 1321.1 | 1314.1 | 1300.0 | 1311.4 | +14.1 | +2.7 | pob |
| BaO | 992.3 | 989.3 | 1120.8 | 1001.5 | **−131.5** | −12.2 | no consensus |
| BaS | 935.4 | 941.1 | 1099.3 | 950.4 | **−158.2** | −9.3 | **pob** |
| C-dia | 733.4 | 723.8 | 716.6 | 726.2 | +7.2 | −2.4 | all agree |
| CaF₂ | 1588.0 | 1589.6 | 1609.9 | 1597.8 | −20.3 | −8.2 | pob |
| CaO | 1079.7 | 1089.2 | 1032.9 | 1096.4 | **+56.3** | −7.2 | **pob** |
| CdSe | 499.1 | 495.5 | 524.8 | 489.8 | −29.3 | +5.7 | pob |
| CsCl | 655.7 | 620.7 | 678.5 | 625.7 | **−57.8** | −5.0 | pob |
| GaAs | 637.4 | 640.3 | 697.9 | 650.0 | **−57.6** | −9.7 | pob |
| GaN | 922.2 | 854.7 | 876.6 | 873.0 | −21.9 | −18.3 | GPAW |
| GaP | 720.0 | 698.1 | 723.6 | 717.9 | −25.5 | −19.8 | GPAW |
| InAs | 595.9 | 587.0 | 623.3 | 579.9 | −36.3 | +7.1 | pob |
| KBr | 609.8 | 593.6 | 661.8 | 620.8 | **−68.2** | **−27.2** | no consensus |
| KCl | 662.3 | 637.9 | 690.1 | 640.0 | **−52.2** | −2.1 | **pob** |
| LiH | 494.5 | 464.3 | 459.1 | 468.3 | +5.2 | −4.0 | all agree |
| Li₂O | 1196.0 | 1136.4 | 1122.2 | 1149.7 | +14.2 | −13.3 | no consensus |
| LiCl | 705.1 | 675.4 | 688.1 | 696.7 | −12.7 | −21.3 | GPAW |
| LiF | 874.2 | 840.2 | 843.7 | 866.6 | −3.5 | **−26.4** | **VASP** |
| MgO | 1017.4 | 1009.5 | 1044.3 | 1012.0 | **−34.8** | −2.5 | **pob** |
| MgS | 786.5 | 772.1 | 780.0 | 775.0 | −7.9 | −2.9 | all agree |
| NaCl | 655.5 | 633.0 | 662.8 | 627.3 | **−29.8** | +5.7 | **pob** |
| NaF | 778.8 | 754.0 | 761.8 | 749.6 | −7.8 | +4.4 | pob |
| SiC | 1262.9 | 1234.6 | 1190.3 | 1242.5 | **+44.3** | −7.9 | **pob** |
| ZnS | 628.8 | 615.1 | 639.1 | 615.1 | −24.0 | 0.0 | pob |

## Statistics (n = 28)

| Metric | GPAW−pob | GPAW−VASP |
|--------|---------:|----------:|
| MD | **−24.6** | **−6.5** |
| MAD | **35.4** | **8.5** |

**Outlier tally:** pob = 17, VASP = 1, GPAW = 3, all agree = 4, no consensus = 3

## Key findings for the paper

### 1. GPAW tracks VASP, not pob

GPAW's r2SCAN PW-limit atomization energies agree with VASP@900eV to **MAD 8.5 kJ/mol** across 28 systems. GPAW disagrees with pob by MAD 35.4 kJ/mol — a factor of 4× larger. The independent PW/PAW code (GPAW) confirms **VASP is the more reliable reference**, and the paper's Δb = E_pob − E_VASP is dominated by pob's deviations, not VASP artifacts.

### 2. VASP is the high outlier only for LiF; KBr splits three ways

LiF remains the ONLY system where VASP is the clear outlier (+26.4 kJ/mol above GPAW while GPAW ≈ pob). KBr (added 2026-07-18) is a three-way split: GPAW 593.6 < VASP 620.8 < pob 661.8. KCl with the *same* K free atom agrees with VASP to −2.1 kJ/mol, isolating the spread to the Br free-atom reference.

**Br convention diagnostic (2026-07-18, GPAW-PWREF-002):** holding the production KBr solid and K atom fixed and varying only the Br free-atom convention (r2SCAN, 800 eV, all runs formally converged):

| Br free-atom convention | E_Br (Ha) | implied E_at (kJ/mol) | vs VASP 620.8 |
|---|---:|---:|---:|
| aspherical, spin-polarised, Hund (production) | −0.369329 | 593.6 | −27.2 |
| spherical spin-polarised (cubic box + 0.1 eV smear) | −0.369151 | 594.1 | −26.7 |
| spin-restricted (spin-paired, 0.1 eV smear) | −0.355249 | 630.6 | +9.8 |

Spin restriction moves the implied atomization by **+37.0 kJ/mol**; sphericalising the minority p-hole within the spin-polarised reference costs only **+0.5 kJ/mol**. VASP's 620.8 sits at ~73% of the spin-restriction shift — its Br reference behaves like a largely (not fully) spin-restricted free atom, exactly the fingerprint found for LiF/F in the 2026-06-17 diagnostic. The GPAW production number uses the physically correct spin-polarised aspherical reference. pob's −68.2 vs GPAW is unaffected by this (CRYSTAL's SYMMREMO atoms are aspherical spin-polarised too) and remains a genuine basis-set deviation.

### 3. pob has large, systematic errors for heavy-element compounds

pob deviates by >30 kJ/mol from GPAW for: BaS (−158.2), BaO (−131.5), KBr (−68.2), CsCl (−57.8), GaAs (−57.6), CaO (+56.3), KCl (−52.2), SiC (+44.3), AlAs (−43.5), InAs (−36.3), MgO (−34.8). Most involve 4th/5th-row elements (Ba, Cs, Cd, In, As, Ga, K, Br) — exactly where basis-set incompleteness is expected. Note the sign is not uniform: pob *underbinds* CaO and SiC while overbinding the rest.

### 4. Light main-group compounds show good agreement

For compounds with only 2nd/3rd-row elements (LiF, NaF, LiH, C-diamond, BN, MgS, AlN, AlP), all three codes agree within ~15 kJ/mol.

### 5. MgO: pob overbinds dramatically

GPAW=1009.5, VASP=1012.0, pob=1044.3. The pob basis overbinds MgO by ~35 kJ/mol relative to both PW codes. Both GPAW and VASP agree with the published SCAN cohesive energy (1011 kJ/mol, Mejía-Rodríguez & Trickey 2018).

## Method notes

- **Cutoffs:** 800 eV for MgO, MgS, LiH, CaO, CaF₂, KBr, KCl, SiC, AlP, GaP and the Al/Ga/Cd/In/Cs systems (Mg/Si/F/Se/As atoms fail at 1000 eV with meta-GGA direct-minimisation). 800→1000 eV sweep for the rest. Atomization is flat at 800→1000 eV (Δ < 0.5 kJ/mol for all converged systems), so 800 eV single-point is adequate.
- **Atoms:** Direct minimisation (etdm-fdpw, conditional Hund) for all elements except H, He, Al, Ga, K, Ca, Cs, P, Ti, Br which use the SCF eigensolver (Davidson + mixing + smearing). The SCF fallback uses the full iteration budget and an explicit 10⁻⁶ eV²/e eigenstates criterion (K/Ca plateau indefinitely on the 4·10⁻⁸ default while the energy is stable to 10⁻⁶ eV). K additionally uses a damped mixer (β = 0.02, nmaxold 8) against intermittent charge-sloshing excursions, with its density criterion at 1.5·10⁻³; two independent converged runs (default vs nbands −6 Davidson subspace) agree on E(K) to 0.04 kJ/mol.
- **Known limitations:** the K free atom is expensive (~15–20 min); Cs untested at >800 eV. Si, Mg, F, Se, As atoms fail at >1000 eV — single 800 eV cutoff used for their systems. KBr's −27.2 vs VASP is diagnosed (finding 2): it is a Br free-atom *spin-treatment* convention difference (VASP's reference behaves largely spin-restricted), not PW convergence or a solid-state effect.
- **Provenance:** Full JSON sidecars in `runs/pwref_*.json` (GPAW version, PAW setups/Nv, geometries, cutoffs, k-mesh).

## Data files (`runs/`)

- `pwref_merged.json` — all 28 systems merged
- `pwref_r2scan.json` — validation set (LiF, NaCl, NaF, LiCl, C-dia, BN)
- `pwref_r2scan_main.json` — AgCl, BaO, BaS, ZnS
- `pwref_gaal.json` — Al/Ga zincblendes (4 systems)
- `pwref_lih.json` — LiH
- `pwref_mgo_mgs.json` — MgO + MgS
- `pwref_tracka.json` — Li₂O, CdSe, InAs, CsCl
- `pwref_li2o_sic.json`, `pwref_sic_alp_gap.json` — Li₂O/SiC re-runs + SiC, AlP, GaP
- `pwref_k_ca.json` — CaO, CaF₂, KCl (2026-07-18, GPAW-PWREF-001)
- `pwref_kbr.json` — KBr (2026-07-18, GPAW-PWREF-001)
- `pwref_noncubic.json` — BeO, ZnO wurtzite (not in the r2scan.dat benchmark table)
- `pwref_al2o3_pbe.json` — Al₂O₃ corundum PBE probe (not r2SCAN; excluded from stats)
