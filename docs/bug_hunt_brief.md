---
orphan: true
---

# vibe-qc bug-hunt brief

This is the **binding operating manual** for any chat or contributor
running a bug hunt on vibe-qc. If you're a Claude Code chat that's been
told to bug-hunt vibe-qc, this is your policy document: read it
end-to-end before doing any work.

> **Status:** active for the current **`main`** line (the **v0.15**
> development cycle). Latest tagged release: **v0.15.84** (*Neese's Cheetah*).
> Last updated 2026-07-29.

**Canonical source is `main`.** This file lives on `main` and is the
authoritative copy. The `release` branch is fast-forward-only from a tag
and carries whatever version of the brief shipped with that tag. Always
read it from `main`:

```sh
git fetch origin main
git show origin/main:docs/bug_hunt_brief.md | less
```

Or consult the canonical copy at
``https://vibe-qc.com/docs/bug_hunt_brief.html`` once the docs site
auto-deploys.

---

## 1. Scope (priority order)

You hunt for and fix:

1. **Wrong answers:** energies, gradients, properties that disagree with
   PySCF / CRYSTAL / ORCA at validated tolerances. Highest priority.
2. **Crashes / segfaults:** anything that exits without a Python
   traceback.
3. **Ergonomic dead-ends:** confusing errors, missing input validation,
   footguns that lead users into invalid states.
4. **Documentation accuracy:** code blocks in tutorials and docstrings
   that don't actually run on a clean checkout.

Bias toward (1) before (2 to 4). If you find a (2 to 4) issue while
hunting (1), drop a one-liner in your reply rather than getting derailed.

## 2. Branch model: work on `main`, never touch `release`

```{admonition} Branch lock
:class: warning

You work on **`main`**. Land every fix there (rebase-then-push, § 4).
`main` is the shared development line and other chats push to it in
parallel, so always `git pull --rebase origin main` before you push.

You **never push to `release`** and **never open an MR against it**.
`release` is protected and fast-forward-only: it advances only when the
release chat fast-forwards it to a new tag. You also **never cut a tag**
(§ 5). GitLab protected-branch and protected-tag rules enforce both, so a
direct push or tag attempt is rejected outright. (CLAUDE.md § 2, § 13.)
```

## 3. Setup

```sh
git clone https://github.com/vibe-qc/vibe-qc.git
cd vibe-qc
git checkout main
./scripts/setup_native_deps.sh
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -c "from vibeqc.banner import print_banner; print_banner()"
# On main the banner prints the dev version, e.g.  0.15.x.dev0+<sha>
# (a release tag would print  Release vX.Y.Z).
```

**Establish the green baseline before any other work.** The CI
`build-test` job is a *build-break gate only* (canonical bootstrap,
`import vibeqc`, `pytest --collect-only`); it runs no tests. The
authoritative full-suite verdict comes from **`scripts/test_gate/`** on a
dev box (per-file isolation plus a `known_reds_baseline.json` of
pre-existing failures):

```sh
python scripts/test_gate/run_full_suite.py     # capture pass / skip / xfail
python scripts/test_gate/gate_verdict.py       # compare against the known-reds baseline
```

Record the green baseline. Any *new* red beyond the known-reds baseline
means stop and find out why before touching source. (CLAUDE.md § 2, § 14.)

## 4. Workflow per bug

1. **Reproduce minimally.** Reduce to the smallest failing case.

2. **Pin it as a regression test.** Add a test to
   `tests/test_bug_repros.py` (create it if missing). It must fail before
   your fix and pass after; the docstring gives a one-line description
   plus the reproducing scenario.

3. **Fix.** Edit the source. Keep the diff small (about 300 lines or
   fewer, ideally one subsystem). Bigger or cross-subsystem means **stop
   and escalate** (§ 6). No papering over periodic-SCF bugs (CLAUDE.md
   § 7): an oscillation or an impossible (Madelung-scale over-bound)
   energy is the bug, not a damping-tuning problem.

4. **Run the gate.** `scripts/test_gate/run_full_suite.py` must show your
   new test passing and no new reds versus the § 3 baseline.

5. **If you touched docs, build clean:**
   ```sh
   .venv/bin/sphinx-build -b html docs /tmp/sb
   ```

6. **If your work adds anything citable** (a method, functional, basis,
   ECP, linked library) add the entry plus its route to the citation
   database in the *same commit* (CLAUDE.md § 8).

7. **Commit.** Use this format:

   ```
   fix(<area>): <one-line summary>

   <2-4 line problem description: what was wrong, what symptom>
   <2-4 line fix description: what changed, why it is the right fix>

   Found via: <e.g. "PySCF cross-check on ZnO / B3LYP / pob-TZVP">
   Reproducer: tests/test_bug_repros.py::test_<name>
   Patch-candidate: v0.15.x        # only if it should ride the next patch; omit otherwise

   Co-Authored-By: <your model identifier> <noreply@anthropic.com>
   ```

   The `Patch-candidate:` trailer is the *only* channel for asking that a
   fix ride a patch release; the release chat scans `main` for it
   (CLAUDE.md § 13). For an already-pushed commit, attach it via
   `git notes` rather than amending (never amend pushed history).

8. **Land on `main`** (rebase-then-push):
   ```sh
   git fetch origin --quiet
   git pull --rebase origin main
   git push origin HEAD:main
   ```
   After every rebase, re-read what landed
   (`git log --oneline <last-sync>..origin/main`) and re-run the affected
   tests: parallel chats move `main` (CLAUDE.md § 14).

## 5. Tagging: you do NOT cut tags, ever

You never run `git tag`, never push `release`, never open an MR against
`release`. Tagging `vX.Y.Z` and advancing `release` are exclusively the
release chat's job (CLAUDE.md § 13). To request a fix in the next patch,
use the `Patch-candidate:` trailer (§ 4.7). After landing, surface a
status line:

> Fix landed at `<sha>` on `main`, trailered `Patch-candidate: v0.12.x`.

## 6. Hard escalation gates: STOP and ask before any of these

- Push or MR to `release` (ever). Cut a git tag (ever).
- A change over 300 lines, or touching more than 5 files unrelated to the
  bug under investigation.
- A new dependency (Python or system): needs a roadmap conversation
  (CLAUDE.md § 9).
- Bumping the **major** or **minor** version in `pyproject.toml`.
- Force-push, history rewrite, branch deletion, protected-ref changes, or
  any hook bypass (`--no-verify`) (CLAUDE.md § 2).
- Bundling, vendoring, or fetching new data without checking its
  redistribution terms (CLAUDE.md § 1).
- Any operation you are not sure is reversible.

When in doubt, escalate. A stop-and-ask is far cheaper than an
unauthorized destructive action.

## 7. Hand-off protocol

Findings that need to leave the bug-hunt chat route through the
multi-chat drop-box convention (CLAUDE.md § 3) plus a short note under
`bug_repros/notes/<short-name>.md`:

- **Needs feature-level work** (new C++ kernel, public-API restructure,
  anything bigger than a fix): note it, surface tagged
  `engineering-chat-needed`.
- **Bug is in tutorial / docs / website prose** (not code logic): note
  it, surface tagged `docs-chat-needed`.
- **Touches periodic-SCF correctness:** reproduce against PySCF.pbc.GDF
  on the same system, file a regression test capturing the symptom, and
  bring it to the periodic-SCF chat. Don't fix-and-forget (CLAUDE.md § 7).

`bug_repros/notes/` is free-form scratch space that ships with the repo.

## 8. Reference values for spot-checks

These canaries should always pass. If any regress, the bug is upstream of
whatever you were hunting: stop, investigate, and report.

| System | Spec | Expected | Pinned in |
|---|---|---|---|
| H₂ | STO-3G / RHF | −1.11675931 Ha | `tests/test_periodic_rhf_ewald.py` |
| Zn²⁺ | LANL2DZ ECP / 6-31G / UHF | matches PySCF to µHa | `tests/test_ecp_validation.py` |
| NaCl | Madelung constant | −1.7475645946… | `tests/test_madelung.py` |
| H atom | any DFT functional / UKS | ⟨S²⟩ = 0.75 | `tests/test_periodic_uks_ewald.py` |
| Multi-k Ewald [1,1,1] | molecular-limit cell | matches Γ-only to 1e-10 Ha | `tests/test_periodic_*_multi_k_ewald.py` |

> The former H₂O / 6-31G\* RHF canary (`tests/test_h2o.py`, −76.0107 Ha)
> was retired in the test-suite reorganisation. Re-pin it to the current
> molecular-RHF reference test before relying on it.

## 9. Stop criteria for a session

End and report when you have either (a) made a full pass through the § 10
targets, or (b) accumulated **5 to 10 fixes** on `main`, whichever comes
first. In your reply, surface:

- Bugs found / fixed / escalated.
- SHAs landed on `main` (one-line summary each) plus their
  `Patch-candidate:` trailers.
- Paths of any `bug_repros/notes/*.md` hand-offs.
- Recommendation: is the v0.12.x line patch-worthy now, or keep hunting?

Quality over breadth.

---

## 10. Active hunt: current `main` (v0.15 cycle)

The high-value hunt areas for the current line. These are the newest,
least-validated surfaces; don't invent your own targets until you've
explored them. Each traces to a CHANGELOG `[Unreleased]` or v0.15.x
entry: read that entry's caveats first.

1. **BIPOLE multi-k corrected-gauge exchange.** The v0.13 flagship: the
   q≠0 long-range exchange channels (option (b), Phase 3). Cross-check a
   real crystal at a genuine k-mesh against the matched-k CRYSTAL
   reference (the BIPOLE-to-CRYSTAL gate): energies, band gaps, and gauge
   consistency.

2. **Megacell periodic MP2 + DLPNO-CCSD(T)** (local-correlation Stage 6).
   New periodic correlation via the finite-supercell to molecular route.
   Check supercell-size convergence of the correlation energy and the
   molecular limit against canonical MP2 / DLPNO-CCSD(T).

3. **Periodic broken-symmetry: READ / ATOMSPIN / SPINLOCK** plus per-spin
   Fermi-Dirac smearing for multi-k GDF. Drive an antiferromagnetic or
   ferrimagnetic crystal (NiO, MnO, …) to a broken-symmetry ground state
   from an `atomic_spins` seed; confirm the spin density, ⟨S²⟩, and that
   SPINLOCK releases cleanly after `spinlock_iterations`.

4. **Experimental / gated routes warn and stay gated.** Native D4
   (defective reference dataset, `D4NativeExperimentalWarning`), GPW /
   GAPW (`GAPWExperimentalWarning`), `caspt2(engine="cases")`, and the
   MSINDO CIS analytic gradient (`_experimental=True`). Confirm each
   emits its warning, is genuinely opt-in, and that enabling it never
   silently changes a default-path result.

5. **EWALD_3D retirement and fail-closed behaviour (§ 7 discipline).**
   EWALD_3D is retired as a user route, and Γ-only drivers must **fail
   closed** on image-overlapping (tight) cells rather than return a
   Madelung-scale over-bound energy. Probe tight cells and the
   `auto_optimize_truncation=False` cutoff floor: you must get a clear
   error, never a wrong number.

6. **Changed default: Γ closed-shell RHF/GDF `exxdiv='ewald'`.** The new
   default targets PySCF parity. Verify it on ionic crystals and confirm
   no regression versus pre-change reference energies.

7. **Periodic analytic gradients.** The corrected-gauge Γ analytic
   gradients (all four drivers) and the multi-k RKS/UKS GGA-σ Pulay fix.
   Validate analytic forces against finite differences on a GGA system at
   Γ and at multi-k.

8. **Docs and new public APIs run on a clean checkout.** Every tutorial
   (now descriptive-named, no number prefixes, under `docs/tutorial/`)
   must execute end-to-end. So must the new first-class surfaces:
   canonical CCSD(T), CCSD `triples=`, AutoCI `citype=`, periodic
   Pipek-Mezey / Foster-Boys localization, K6 generalized k-grids, and
   the CIF / Extended-XYZ readers.

---

## 11. Closed hunts

Prior hunt lines (v0.4.x through v0.12.x) are closed. Their targets and
known limitations live in each tag's `CHANGELOG.md` § Limitations; this
brief tracks only the current `main` line.
