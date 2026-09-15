# Asbestos polymorphs — bulk DFT reference paper

**Status:** scaffolding (input infrastructure landed, geometries TODO, runs gated by roadmap milestones).

**Modeled on:** the 2014 Al₂O₃ paper (Vilela Oliveira et al.) — bulk-only,
PW1PW/pob-DZVP, periodic CRYSTAL-style. We're using the same setup
philosophy (no surfaces, no adsorption, no nanotubes).

**Method:** RKS-PBE0 by default (canonical solid-state hybrid; works
in vibe-qc today via `run_periodic_job(method='RKS', functional='pbe0',
jk_method='gdf')`). PW1PW variants ride along for direct comparison
to the inspirational paper once the PW1PW functional lands (see
`feature/uks-periodic-gdf` chip).

## Per-mineral status

| Mineral | Space group | Geometry source | Method (today / paper) | Roadmap blocker |
|---|---|---|---|---|
| Lizardite-1T | P3̄1m (162) | TODO — Mellini & Zanazzi 1987 (AMCSD) | RKS-PBE0 / RKS-PW1PW | none |
| Chrysotile (clino) | Cc (9) | TODO — Whittaker 1956 (AMCSD) | RKS-PBE0 / RKS-PW1PW | none |
| Antigorite | Pm (6) | TODO — Capitani & Mellini 2004 (m=17 superstructure; ~290 atoms) | RKS-PBE0 / RKS-PW1PW | size only — feasibility TBD |
| Tremolite | C2/m (12) | TODO — Hawthorne & Grundy 1976 | RKS-PBE0 / RKS-PW1PW | none (Mg endmember) |
| Anthophyllite | Pnma (62) | TODO — Walitzi 1965; Sueno et al. 1972 | RKS-PBE0 / RKS-PW1PW | none (Mg endmember) |
| Riebeckite | C2/m (12) | TODO — Whittaker 1949 | UKS-PBE0 / UKS-PW1PW | **UKS** (Fe-bearing; needs `feature/uks-periodic-gdf`) |
| Grunerite | C2/m (12) | TODO — Hawthorne 1983 | UKS-PBE0 / UKS-PW1PW | **UKS** (Fe-bearing) |

## Roadmap dependencies

> **Update:** this section listed `feature/*` branches that no longer
> exist (`git branch -a` finds none of them) — they either landed and
> were retired, or were rewritten on newer infrastructure
> (`docs/roadmap.md:2059` notes `feature/uhf-periodic-gdf` specifically
> was retired 2026-05-13 and rebuilt). All blockers below have since
> shipped on `main`; the per-mineral table above was not updated when
> they did and should be re-verified against current `run_periodic_job`
> UKS/multi-k support rather than assumed still-blocked.

- ✓ **GDF Γ-only RHF + RKS** (v0.7.1-spike, on main): closed-shell systems runnable today (lizardite, chrysotile, tremolite-Mg, anthophyllite-Mg).
- ✓ **Open-shell periodic GDF** (was `feature/uhf-periodic-gdf` / `feature/uks-periodic-gdf`): `run_kuhf_periodic_gdf` / `run_kuks_periodic_gdf` ship multi-k UHF/UKS GDF (per-spin V_xc), with native GDF ROHF dispatch landing 2026-08-01 and open-shell ROKS/BIPOLE wiring 2026-08-02. **Unblocks the Fe-bearing minerals (riebeckite, grunerite).**
- ✓ **PW1PW functional**: validated 2026-06-08, accepted as `functional="pw1pw"`. **Unblocks the PW1PW comparisons against the 2014 Al₂O₃ paper / pob-TZVP paper SI Table 1.**
- ✓ **Multi-k periodic SCF** (was `feature/multi-k-periodic-scf`): multi-k GDF energies + analytic gradients for KRHF/KRKS/KUHF/KUKS shipped 2026-07-27 through 2026-08-02. **Needed for quantitative comparison vs CRYSTAL-converged literature totals** — re-check whether the ~30-110 mHa/f.u. Γ-only-undersampling discrepancy on simple ionics is resolved now that k-mesh sampling is available.
- Native GDF (PySCF-import-free periodic SCF) and the Coulomb-method comparison items were informational/non-blocking and not independently re-checked here.

## Layout

```
studies/asbestos-polymorphs/
├── README.md                    (this file)
├── _minerals.py                 (geometry library — loads from cifs/)
├── cifs/                        (CIFs to be dropped in; CIFS_NEEDED.md
│                                 lists the recommended source per mineral)
└── <mineral>/                   (one subdir per mineral)
    ├── <mineral>-RKS-PBE0-pobdzvp.py        (today)
    └── <mineral>-RKS-PW1PW-pobdzvp.py       (gated on PW1PW landing)
```

Output files (`.out`, `.system`, `.molden`, `.xsf`) are written
alongside each input by `vibeqc.run_periodic_job` and follow the
filename-stem convention used in `examples/periodic/<system>/`.

## Running the inputs

### Sanity check (today, closed-shell only)

```bash
.venv/bin/python studies/asbestos-polymorphs/lizardite-1T/lizardite-1T-RKS-PBE0-pobdzvp.py
```

PW1PW inputs will fail until the `feature/uks-periodic-gdf` chip
lands the functional. They're committed now so the input set is
complete and one merge later they all run.

### Once UKS + PW1PW + multi-k all land

```bash
# Closed-shell:
.venv/bin/python studies/asbestos-polymorphs/run_all.py     # parallel orchestrator (TODO)

# Fe-bearing:
# additional .py files for riebeckite + grunerite UKS runs.
```

## Paper plan reminder

**Bulk-only**. Surfaces / adsorption / single-walled nanotubes are
explicitly out of scope (per `project_asbestos_paper.md`). The
deliverable is novel total-energy + structural DFT data on the 7
asbestos polymorphs at converged k-mesh, comparable to the 2014
Al₂O₃ paper in methodological rigor.
