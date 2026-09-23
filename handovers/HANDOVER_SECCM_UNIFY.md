# HANDOVER: SECCM unification lane (periodic semiempirical cyclic-cluster model)

**Written:** 2026-09-16. **Repository:** `vibe-qc` (GitLab project 34), branch `main`.
**Supersedes as the lane entry point:** the 2026-09-13 SECCM handover and the
post-split plan, which now live in the private operations record together with
the maintainer decisions D1-D7 and the adenine finding. This file carries the
public working state only. A defect belongs on the tracker, not here.

> Issue numbers are project-34 iids. Pre-split numbers name different issues;
> map them through `docs/issue_renumbering_2026_09.md` before citing.

## 1. Start here

1. This file.
2. `handovers/MAP_SECCM_IMPLEMENTATIONS.md`: the cross-referenced map of the five
   SECCM adapters, the legacy MSINDO arm and the periodic Gamma driver, with
   the duplication ledger, the disagreements, the single-owner specification
   and the ordered refactor steps. It is the refactor's specification.
3. The tracker: #293, #294 (filed by this lane), #140, #145, #137, #69, #138,
   #135, #3, #247, #246, #245, #249, #64.

First commands, from a clone of your own:

```sh
git pull --rebase origin main
VIBEQC_REQUIRE_VENDORED=ON .venv/bin/pip install -e . --no-build-isolation --no-deps
.venv/bin/python -c "import vibeqc._vibeqc_core as c, os, time; print(time.ctime(os.path.getmtime(c.__file__)))"
```

A fresh clone rebuilds libint because the #271 recipe changed the vendored
stamp; a sibling clone built after 2026-09-10 carries libint and libecpint
with the current recipe hashes (`third_party/.build-stamp`), and cloning those
two install trees plus the stamp into the new checkout lets the installer skip
straight to the extension.

## 2. State at handover

**Lane inventory on the pristine tree** (`9be848b`, extension built 2026-09-16 19:14, sha256 `c9144d7d...`, 47 lane files matching `semiemp|dftb|xtb|gfn|seccm|msindo|ccm_semiempirical` plus the five extras named in the previous handover's command, `-q -p no:randomly -rfs`): **2 failed, 2604 passed, 2 skipped, 4 xfailed** in 5898 s on a host at load 100-150. Skips: the two `PM6 FD stress too slow for CI`; xfails unchanged at 4. Failures, both re-run in isolation: `test_seccm_topology.py::test_seccm_topology_build_is_not_the_wall_clock_bottleneck` (a 25 s wall-clock bound; 67 s on the loaded host, 5.5 s in the M6 record; a load artefact to re-verify with `gate_verdict.py --reverify`) and `test_test_gate_lanes.py::test_periodic_ccm_test_files_carry_experimental_pytestmark`, which names `tests/test_kpoints.py` (outside this lane; red at `HEAD` before any change here). Re-measure rather than quote; the inventory's file list is the filter above restricted to `.py`.

Landed by this lane on `main`:

* `007f0df` (#293) one `build_seccm_madelung_state` for the SCC-DFTB energy and
  gradient, GFN2-SECCM and PM6-SECCM. The embedded 3-D gradient used the 2-D
  slab kernel; on a distorted six-atom H-Li torus it differed from central
  differences by 4.4e-5 Ha/bohr and now agrees to 3.8e-7.
* `63a2a28` (#294) the GFN2-SECCM primary SCC attempt keeps its budget and
  reaches the ladder from a budget-independent checkpoint, with the new
  attempt exit reason `stalled_checkpoint`. `max_iter` 2499, 2500 and 3600 now
  all give one 1784-iteration attempt on the polar HF chain.
* `7bb7e93` this handover and the implementation map.

Both fixes await independent verification (requests on #293 and #294); do not
close either yourself.

Verified by this lane as a second actor (verdicts are issue comments):

* **#145** periodic GFN2 translation invariance: PASS on every closure criterion
  on `main` `9be848b` (issue repro 8e-13 Ha; ten relabel/permute/rigid cases
  4e-9 Ha raw, 1e-12 closure; face crossing continuous; Si16 6e-14; water box
  refused). Recommend closing; not closed by this lane.
* **#140** GFN2-SECCM AES representative dependence: PASS for the
  `aes_faithful=True` model (invariant to 1e-14); the default `include_aes=True`
  path is still representative-dependent by 1.3..1.6e-4 Ha, by design, so the
  issue as filed is not closable. Maintainer decision: move the SECCM and
  molecular `aes_faithful` defaults together (M4b ties them), or re-scope #140.
* **#64** MSINDO CCM citation selector: the hand-landed fix is in code and its
  pinned tests pass (58 selected tests).

## 3. What the map found (short form)

* Only `validate_common_inputs`, `max_abs` and the `WSTopology` struct are used
  by all five adapters. PM6 and OMx use no charge mixer, no periodic gamma, no
  canonical screening and no shared gap helper.
* The same physics is implemented more than once in 26 places (the ledger in
  the map): the Ewald alpha convention four times, the 1-D wire quadrature
  twice with different truncations, the image-record conversion in three
  shapes, the Madelung dimension dispatch four times (one of them wrong, #293),
  the gap guard three times, the SCC-DFTB mixers as copies of the molecular
  loop, the GFN2 stabilisation ladder with different budgets per driver, the
  `n_occ`/GAM3 scaling, the ad-hoc AES loop, `flatten_topology_records`
  (dftb0.py inline copy), the gradient bookkeeping helpers.
* Disagreements that are not cosmetic: #293 (wrong answer); the SECCM ladder
  overrides an explicitly requested temperature where the molecular driver
  honours it (#3 class; provenance intact; maintainer question); PM6 and OMx
  acknowledgement gates differ; SCC-DFTB does not record `gamma_form`,
  `ewald_gamma` or the mixer and reports `electrostatics_family="none"` for
  `ewald_gamma` runs (so it cites no Ewald route); PM6 stamps
  `electrostatics_family="none"` even with `madelung=True`.
* SECCM `energy` is the Mermin free energy on both SCC routes while the
  periodic and molecular drivers report `energy = E` and `free_energy = E - TS`.

## 4. Measurements of record (all on `main` `9be848b`, extension built 2026-09-16 19:14)

* MgO 2-atom cell, periodic Gamma GFN2, Elstner, T = 0: the raw binding
  converges to two fixed points 1.22e-4 Ha apart depending on the coordinate
  representative (only the Eyert default converges at all); at T = 0.0005 and
  0.001 every representative reaches one state (1e-12). Branch selection
  (#137 class), not a Hamiltonian defect.
* 2x2 LiH sheet, SCC-DFTB-SECCM, embedded, T = 0.005: at geometries displaced
  by 1e-4 bohr the default Aitken mixer converges (574 iterations) to a
  symmetry-broken state 0.93 mHa below the symmetric one; DIIS stays symmetric.
  A finite-difference oracle with the default mixer is therefore contaminated
  there. This is the #69 state-hopping class on a six-second fixture.
* GFN2-SECCM polar HF chain, `charge_mixing = 0.01`: 1784 iterations in one
  attempt at `max_iter = 2499`, cut at 500 and restarted at 2500 (#294).
* SCC-DFTB 3-D embedded gradient, distorted (3,1,1) H-Li: 4.4e-5 Ha/bohr off
  central differences before #293's fix.

## 5. Next steps, in priority order

1. Verification of #293 and #294 by someone other than this lane; the
   requests are on the issues.
2. Maintainer decisions (also listed as questions in the map, section 6):
   the `aes_faithful` default versus re-scoping #140; whether the SECCM ladder
   honours an explicit temperature; `energy` semantics at T > 0 across routes;
   the PM6/OMx acknowledgement predicate; the canonical 1-D wire truncation
   when the two quadratures merge; whether MSINDO joins the unified topology.
3. The remaining unification steps in the map's ordered list (steps 1, 2, 4,
   7, 10, 11, 12, 16, 17, 18 are behaviour-preserving by construction; 5, 6, 8,
   9, 13, 14, 15 need the named pin tests). Steps 3 (Madelung state) is done.
4. #137/#69: reproduce the Ge disproportionation and odd-MgO with SCC-DFTB on
   current main, starting from the sheet fixture above.
5. Phase 4: PM6-SECCM 3-D (the 3-D refusal at `pm6.cpp:213-218` now sits in
   front of a builder that already covers 3-D; the thermodynamic-limit
   measurement is the work), OMx variational Madelung (no `madelung` option
   exists in the OMx adapter at all), the D5 Hubbard U review (the H, C, N, O
   values in the in-house table are the mio-1-1 set; F, P, S, Cl, Li and the
   rest are round estimates), and #245.
6. #245 scoping result: xtb's periodic GFN1 Coulomb matrix is the same
   Ewald-split Klopman-Ohno construction as the shared kernel with the harmonic
   average (`KlopmanOhnoAverage::InverseHardnessMean`) and `gExp = 2`; xtb's
   `xtbrestart` carries the shell charges; a component anchor is one probe
   script plus a small seam that accepts explicit per-shell hardness. Diamond
   runs in xtb (total -3.827308842052 Eh, electrostatic 0.000548218763 Eh);
   MgO needs a 64-atom cell (2-atom and 8-atom Gamma-only cells have an
   indefinite overlap in xtb).

## 6. Learnings

* A two-atom cell on one lattice cannot see an error in a pair-summed
  potential term: for a neutral pair `V_A + V_B` vanishes for every kernel.
  Distort the cell before trusting a finite-difference agreement.
* Before attributing a finite-difference mismatch to the analytic gradient,
  compare the converged states at the displaced geometries under a second
  mixer. Branch hopping under 1e-4 bohr looks exactly like a wrong derivative.
* `scc_mixer=Simple` with `mixer_damping = 0` selects the Eyert default on the
  periodic Gamma driver; simple mixing needs `mixer_damping > 0`.
* `pkill -f <pattern>` kills the shell that contains the pattern in its own
  command line. Quote the pattern as a regex that does not match itself.
* The lane inventory filter `semiemp|dftb|xtb|gfn|seccm|msindo` also matches
  `.cpp` test sources; restrict it to `.py`.
