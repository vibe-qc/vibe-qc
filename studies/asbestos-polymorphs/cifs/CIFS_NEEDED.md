# CIF files needed

Drop CIFs here, one per mineral, then `_minerals.py` will load them
via `ase.io.read('<mineral>.cif')`.

## Status (2026-05-09 OPTIMADE fetch attempt)

| Mineral | OPTIMADE result |
|---|---|
| tremolite | ✓ fetched from Materials Project (`mp-1196550`, C 2/m verified by spglib). See `tremolite.cif`. |
| lizardite-1T | ✗ MP has 2 H₄Mg₃O₉Si₂ entries (mp-23764 P31m, mp-24097 P6₃cm); neither is lizardite (P-3 1 m). Need AMCSD. |
| chrysotile-clino | ✗ Same MP entries — neither is Cc. Need AMCSD. |
| antigorite-m17 | ✗ Same MP entries — neither is Pm. Need AMCSD. |
| anthophyllite-Mg | ✗ No matches in MP for H₂Mg₇O₂₄Si₈. Need AMCSD. |
| riebeckite | ✗ No matches in MP for Fe₅H₂Na₂O₂₄Si₈. Need AMCSD. |
| grunerite | ✗ No matches in MP for Fe₇H₂O₂₄Si₈. Need AMCSD. |

COD's OPTIMADE endpoint at `crystallography.net` was unreachable
during the fetch attempt; the qiserver mirror returned HTTP 500.
Either of those would likely have these structures (COD is the
natural-mineral OPTIMADE provider) — re-run `fetch_optimade.py` after
COD is back up.

The proper vibe-qc OPTIMADE fetcher (v0.8 deliverable, separate chat)
will replace `fetch_optimade.py` with a robust multi-provider client
and add AMCSD bridging.

## Recommended sources (free / academic-open)

| Mineral | Filename | Recommended source |
|---|---|---|
| Lizardite-1T | `lizardite-1T.cif` | AMCSD #0011073 (Mellini & Zanazzi 1987) — Mg₃Si₂O₅(OH)₄, P3̄1m, a=5.323, c=7.272 Å |
| Chrysotile (clino) | `chrysotile-clino.cif` | AMCSD #0009829 (Whittaker 1956) — Mg₃Si₂O₅(OH)₄, Cc, a=5.34, b=9.241, c=14.689 Å, β=93.27° |
| Antigorite | `antigorite-m17.cif` | AMCSD #0006097 (Capitani & Mellini 2004; m=17 superstructure, ~290 atoms) — start with smaller m if 17 is too heavy |
| Tremolite | `tremolite.cif` | AMCSD #0005135 (Hawthorne & Grundy 1976) — Ca₂Mg₅Si₈O₂₂(OH)₂, C2/m, a=9.84, b=18.07, c=5.28 Å, β=104.7° |
| Anthophyllite | `anthophyllite-Mg.cif` | AMCSD (Walitzi 1965 / Sueno et al. 1972) — Mg₇Si₈O₂₂(OH)₂, Pnma, a=18.56, b=17.88, c=5.28 Å |
| Riebeckite | `riebeckite.cif` | AMCSD (Whittaker 1949) — Na₂Fe₃²⁺Fe₂³⁺Si₈O₂₂(OH)₂, C2/m, a=9.79, b=18.06, c=5.34 Å, β=103.6° |
| Grunerite | `grunerite.cif` | AMCSD (Hawthorne 1983) — Fe₇²⁺Si₈O₂₂(OH)₂, C2/m, a=9.57, b=18.39, c=5.34 Å, β=102.0° |

## Sources

- **AMCSD** — American Mineralogist Crystal Structure Database
  <https://rruff.geo.arizona.edu/AMS/amcsd.php>
  Free, browser search, CIF download.
- **Crystallography Open Database (COD)**
  <https://www.crystallography.net/cod/>
  Free, mirror of many published crystal structures.
- **ICSD** — Inorganic Crystal Structure Database (subscription).
- **Materials Project** — DFT-relaxed structures; useful as starting
  points but **not** the experimental reference. For a paper modeled
  on Vilela Oliveira 2014, use the experimental refs above (the
  paper's pob-DZVP/PW1PW totals are at experimental geometries, no
  pre-relaxation).

## Notes per mineral

- **Antigorite**: the m=17 superstructure has ~290 atoms — stress
  test for the periodic SCF infrastructure. Start with a smaller m
  (5 or 7) for the first runs; document which m was used.
- **Chrysotile**: bulk crystals are rare; what's "chrysotile" in
  experiments is most often the *cylindrical scroll* of lizardite-like
  layers. The bulk `clino-chrysotile` Cc structure is the right
  reference for a bulk DFT comparison.
- **Riebeckite / Grunerite**: Fe-bearing — they live in different
  spin-flavour ground states (riebeckite has Fe²⁺ + Fe³⁺ at distinct
  sites; grunerite is all Fe²⁺). Need UKS at minimum, ideally with
  broken-symmetry initial guesses for proper antiferromagnetic
  characterisation. Start with high-spin ferromagnetic as the
  simplest test.
