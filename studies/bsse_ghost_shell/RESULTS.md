# ATOMBSSE ghost-shell convergence, pob-TZVP-REV2 / PBE

Which `(nstar, rmax)` is enough for a counterpoise free atom, per system and
per atom site. Ladder `(5,4.0) (10,5.0) (15,6.0) (20,7.5) (30,10.0)` throughout.

**Converged** = cheapest ladder point within **0.5 kJ/mol** of the densest one
that succeeded, **and at least 3 points succeeded**. That last clause is
load-bearing: with one surviving point the densest *is* the chosen one and the
difference is trivially zero, which reads identically to a real plateau.

Two sweeps, at **different protocols**, and the difference matters enough that
every row records which one it came from:

| sweep | job | emitter | protocol | scope |
|---|---|---|---|---|
| **1** | `618e0d8e1aa6` (compute-study) | vibe-basis 0.7.0 | `HUGEGRID`, **no** `TOLINTEG` | 24 systems / 45 sites / 225 runs |
| **2** | `8e74455587c4` (compute-reference) | vibe-basis 0.9.0 | `TOLINTEG 9 9 9 18 54`, **no** `HUGEGRID` | 12 sites re-run / 60 runs |

Sweep 2 re-ran the 9 sites sweep 1 could not settle, plus 3 already-converged
controls (`LiF|1`, `MgO|2`, `LiCl|1`) to test whether the protocol change moves
the answer. Regenerate the merged table with:

```sh
python examples/basisset_dev/bsse_sweep/merge_results.py \
    sweep1=<sweep1 results.json> sweep2=<sweep2 results.json>
```

**A site's verdict rests on one sweep only** -- the latest that produced any
point for it. Merging rung-by-rung looks right and is not: while sweep 2's
densest rung was still running, sweep 1's survived at that rung, so the plateau
would have been asserted across two protocols with the *densest* point, the one
every delta is measured against, coming from the protocol being replaced.

## Read this before using the sweep-1 table

**142 of 225 runs produced a usable energy.** The other 83 failed
overwhelmingly at the dense end -- 37 at `rmax=10.0`, 21 at `7.5`.

So for many sites the densest *successful* rung is 15/6.0 or 20/7.5, not
30/10.0, and "converged" means converged **within the range that computed**.
Where a site's chosen setting equals its densest successful rung, the plateau
is asserted over a shorter interval than the ladder was designed to probe.

| outcome | sites |
|---|---|
| converged (>=3 points) | 36 |
| INSUFFICIENT (1-2 points) | 6 |
| FAILED (0 points) | 3 |

The three total failures are **C-diamond, Si and Ge** -- every group-14
elemental solid in the set, and nothing else.

### The 83 failures are three different things, not one (revised 2026-08-01)

An earlier revision of this page said the failures were "resource failures,
not deck errors: a malformed deck fails in under a second." **That was right
about the timeouts and wrong about half the rest.** Reading back every failing
`out.txt` from the job's workdir splits them cleanly:

| count | status | what actually happened |
|---|---|---|
| 32 | `timeout` | genuinely out of wall clock at 5400 s |
| 36 | `no_energy` | **the SCF energy went `NaN`**, 30 of them then SIGSEGV |
| 15 | `ERROR **** MULTIP` | the non-symmorphic deck defect below |

The `no_energy` group is the one that was misread. Those runs are not short of
time or memory: they print `CYC n ETOT(AU) NaN DETOT NaN`, DIIS reports
`DIIS TEST: NaN`, and CRYSTAL usually dies inside Fortran with
`forrtl: severe (174): SIGSEGV`. Because the crash leaves no `ERROR ****`
line, the runner had nothing to name them by, and "no energy" read as
"presumably ran out of something".

They are also **not size-driven**, which is what makes the resource reading
untenable. `KH|2` NaN'd at 15/6.0 with 436 AOs after 22 s while `KH|1`
completed 30/10.0 with far more; `GaAs|1` NaN'd at 15/6.0 (1435 AOs, 109 s)
having succeeded at 10/5.0. The NaN arrives at cycle 1 in 26 of the 36 cases,
i.e. on the first density built from the ghost cluster. A near-singular
overlap in a dense counterpoise cluster is the obvious suspect, and CRYSTAL
has no automatic canonical-orthogonalisation drop to survive one.

Two things follow, and neither is a licence to tune:

* **Do not treat a NaN'd rung as "needs a longer timeout".** More wall clock
  cannot fix arithmetic that broke on cycle 1.
* **Do not paper over it with damping, level shifts or a looser tolerance**
  (CLAUDE.md section 7). What the ghost shell can be extended to before the
  SCF stops being computable is a real property of this protocol and belongs
  in the result, not hidden under a convergence aid.

Relevant, and worth knowing before assuming the route is meant to carry these
atoms at all: the CRYSTAL23 manual heads the keyword *"ATOMBSSE - counterpoise
for **closed shell** atoms and ions"* and adds *"It is suggested to compute the
atomic wave function using a program properly handling the electronic
configuration of open shell atoms."* The reference set nevertheless drives it
open-shell with `SPINLOCK`, which is why this study does too, but the manual is
explicit that this is outside the keyword's design centre.

The runner now records `scf_nan` as its own status, so a future sweep does not
have to be reverse-engineered from its outputs.

**Diagnosed (2026-07-28).** All three are space group **227** (Fd-3m, diamond),
the only SG-227 systems here, and all three fail identically at every rung
including the cheapest:

```
ERROR **** MULTIP **** SYMMOPS DO NOT FORM A GROUP
```

Isolated by running one system three ways: the plain bulk deck works;
counterpoise fails **with or without** `SYMMREMO`. So `ATOMBSSE` is the
trigger, not the spin treatment.

The reason is that Fd-3m is **non-symmorphic**. `ATOMBSSE` carves a 0D cluster
out of the crystal, and a symmetry operator carrying a translation cannot map a
finite cluster onto itself, so the surviving operators do not close into a
group. Rocksalt (225) and zincblende (216) are symmorphic, which is exactly why
everything else in this set was fine.

The reference set solves it the documented way, and its own diamond decks show
the fix: `ORIGIN` then `TRASREMO` immediately before `ATOMBSSE`. Per the
CRYSTAL23 manual, `ORIGIN` moves the origin "to minimize the number of symmetry
operators with finite translation components" and `TRASREMO` removes what
remains. Neither appears in the reference set's rocksalt decks, because neither
is needed there.

**Fixed in vibe-basis 0.9.0**, which emits `ORIGIN` + `TRASREMO` for every
counterpoise deck. Measured on CRYSTAL23 v1.0.1, PBE/pob-TZVP-REV2,
nstar=5 rmax=4.0:

| system | with | without |
|---|---|---|
| Si (SG 227, non-symmorphic) | **-289.21198632 Ha** | `MULTIP` error |
| LiCl (SG 225, symmorphic) | -459.94485757623 Ha | -459.94485757623 Ha |

Bit-identical on the symmorphic system, so it is emitted unconditionally rather
than keyed to a space-group table: where there are no translational operators,
`ORIGIN` has nothing to minimise and `TRASREMO` nothing to remove.

These three sites therefore need a **re-run, not a longer timeout**, and the
re-run should now work.

**Confirmed by sweep 2 (2026-08-01).** All three now produce energies where
every rung previously died at input parsing. Cross-checked between compute-reference
(x86) and a local arm64 CRYSTAL23 v1.0.1 on `Ge|1` at 5/4.0: **-2076.60117882
Ha on both**, agreeing to 0.00000 kJ/mol.

| site | 5/4.0 | 10/5.0 | 15/6.0 | 20/7.5 |
|---|---|---|---|---|
| `Si\|1` | -289.21198632 (178 s) | -289.21293026 (732 s) | -289.21318964 (2135 s) | -289.21340573 (9550 s) |
| `Ge\|1` | -2076.60117882 (507 s) | -2076.60226365 (1905 s) | -2076.60230230 (2832 s) | timeout |
| `C-diamond\|1` | -37.79517549 (5407 s) | timeout | timeout | timeout |

`Si|1` and `Ge|1` are converged. **`C-diamond|1` is not**, and cannot be from
this data: it produced exactly one point, because its 10/5.0 rung did not
finish inside a **four-hour** budget. That is a cost wall, not the old defect.

## Sweep 2 -- the re-run (2026-08-01)

**Job `8e74455587c4` on compute-reference**, 16 workers, `SWEEP_TIMEOUT_S=14400`
(4 h, up from sweep 1's 5400 s), vibe-basis 0.9.0 at current protocol defaults.

**The `NaN` class is gone entirely.** Of 48 recorded runs, **35 ok and 13
timeout -- zero `scf_nan`, zero `MULTIP`**. Rungs that NaN'd in sweep 1 now
converge: `KF|2` at 10/5.0 (NaN after 35 s, now 510 s ok), `KH|2` at 15/6.0
(NaN after 22 s, now 236 s ok), `GaAs|1` at 15/6.0 and 20/7.5, `NaH|2` at
15/6.0. The tightened `TOLINTEG` is the plausible cause -- a near-singular
overlap in a dense ghost cluster is sensitive to how accurately the integrals
that build it are screened -- but this sweep changed `TOLINTEG` and `HUGEGRID`
together, so **which of the two did it is not separated here.**

**The remaining constraint is wall clock, and the protocol made it worse.**
Every sweep-2 failure is a timeout, and the same rung costs **2.5x to 5.5x**
more than at sweep-1 settings (see the control table below). Three sites are
now bounded by cost rather than by any defect:

| site | outcome | why |
|---|---|---|
| `BN-beta\|1` | **INSUFFICIENT**, 1 point | 10/5.0 exceeded 4 h |
| `BN-beta\|2` | **INSUFFICIENT**, 1 point | 10/5.0 exceeded 4 h |
| `C-diamond\|1` | **INSUFFICIENT**, 1 point | 10/5.0 exceeded 4 h |
| `MgO\|2` | **INSUFFICIENT**, 2 points | 15/6.0 succeeded in sweep 1 (3030 s), exceeded 4 h here |

`MgO|2` is the honest cost of the protocol change stated plainly: a site that
was converged on 3 points at sweep-1 settings has only 2 at sweep-2 settings,
because a rung that used to fit no longer does. Its chosen setting is unchanged
(5/4.0); what it lost is the evidence for calling that a plateau.

**Still outstanding.** The 30/10.0 rung for all 12 sweep-2 sites was still
running when this was written (12 runs, each with a 4 h ceiling). Nothing below
rests on it, but a denser point can move a chosen rung, so the sweep-2 numbers
carry the *same* "converged within the range that computed" caveat as sweep 1.
Re-fetch `results.json` from job `8e74455587c4` and re-run `merge_results.py`
to fold them in.

### Protocol control -- does the 0.8.0 change move the answer? No.

Three already-converged sites re-run at the new protocol. Same geometry, same
basis, same functional; only `TOLINTEG`/`HUGEGRID` differ.

| site / rung | sweep 1 (0.7.0) | sweep 2 (0.9.0) | dE | t1 | t2 | cost |
|---|---|---|---|---|---|---|
| `LiF\|1` 5/4.0 | -7.45987820 | -7.45987859 | -0.001 | 102 s | 263 s | 2.6x |
| `LiF\|1` 10/5.0 | -7.46172385 | -7.46172396 | -0.000 | 1062 s | 5513 s | 5.2x |
| `LiF\|1` 15/6.0 | -7.46174235 | -7.46174224 | +0.000 | 1437 s | 6284 s | 4.4x |
| `MgO\|2` 5/4.0 | -75.00854010 | -75.00853576 | +0.011 | 245 s | 771 s | 3.1x |
| `MgO\|2` 10/5.0 | -75.00862064 | -75.00861628 | +0.011 | 1280 s | 5204 s | 4.1x |
| `LiCl\|1` 5/4.0 | -7.45756084 | -7.45756007 | +0.002 | 40 s | 99 s | 2.5x |
| `LiCl\|1` 10/5.0 | -7.45798473 | -7.45798392 | +0.002 | 97 s | 291 s | 3.0x |
| `LiCl\|1` 15/6.0 | -7.45869235 | -7.45869173 | +0.002 | 459 s | 1765 s | 3.8x |
| `LiCl\|1` 20/7.5 | -7.45877422 | -7.45877327 | +0.002 | 891 s | 4944 s | 5.5x |

dE in kJ/mol. **The largest disagreement is 0.011 kJ/mol, 45x inside the
0.5 kJ/mol convergence tolerance**, and the chosen `(nstar, rmax)` is
unchanged for all three controls (`LiF|1` 10/5.0, `LiCl|1` 15/6.0, `MgO|2`
5/4.0).

So **sweep 1's converged settings stand**, which is the reassurance the control
was run for: converged ghost-shell extents really are a basis-overlap property
and only weakly grid- and tolerance-dependent, as was assumed when converging
at PBE. What the protocol change buys is the removal of the `NaN` failures;
what it costs is 2.5-5.5x wall clock, and that cost is why sweep 2 reaches
fewer dense rungs than sweep 1 did.

## Merged verdicts

Sweep-2 rows supersede sweep-1 rows for the 12 re-run sites; every other row is
sweep 1 unchanged.

| outcome | sweep 1 | after sweep 2 |
|---|---|---|
| converged (>=3 points) | 36 | **41** |
| INSUFFICIENT (1-2 points) | 6 | 4 |
| FAILED (0 points) | 3 | **0** |

Net: **6 of the 9 unsettled sites are now converged** (`Si|1`, `Ge|1`,
`GaAs|1`, `KF|2`, `KH|2`, `NaH|2`), 3 are not (`C-diamond|1`, `BN-beta|1`,
`BN-beta|2`), and one previously-converged control (`MgO|2`) was downgraded by
the protocol's cost. Nothing fails outright any more.

| system | site | Z | verdict | at | dE vs densest | densest | pts | from |
|---|---|---|---|---|---|---|---|---|
| BN-beta | 1 | 5 | INSUFFICIENT | 5/4.0 | +0.000 kJ/mol | 5/4.0 | 1 | sweep2 |
| BN-beta | 2 | 7 | INSUFFICIENT | 5/4.0 | +0.000 kJ/mol | 5/4.0 | 1 | sweep2 |
| C-diamond | 1 | 6 | INSUFFICIENT | 5/4.0 | +0.000 kJ/mol | 5/4.0 | 1 | sweep2 |
| GaAs | 1 | 31 | converged | 20/7.5 | +0.000 kJ/mol | 20/7.5 | 4 | sweep2 |
| Ge | 1 | 32 | converged | 10/5.0 | +0.101 kJ/mol | 15/6.0 | 3 | sweep2 |
| KF | 2 | 9 | converged | 10/5.0 | +0.240 kJ/mol | 20/7.5 | 4 | sweep2 |
| KH | 2 | 1 | converged | 5/4.0 | +0.144 kJ/mol | 20/7.5 | 4 | sweep2 |
| LiCl | 1 | 3 | converged | 15/6.0 | +0.214 kJ/mol | 20/7.5 | 4 | sweep2 |
| LiF | 1 | 3 | converged | 10/5.0 | +0.048 kJ/mol | 15/6.0 | 3 | sweep2 |
| MgO | 2 | 8 | INSUFFICIENT | 5/4.0 | +0.211 kJ/mol | 10/5.0 | 2 | sweep2 |
| NaH | 2 | 1 | converged | 5/4.0 | +0.158 kJ/mol | 20/7.5 | 4 | sweep2 |
| Si | 1 | 14 | converged | 20/7.5 | +0.000 kJ/mol | 20/7.5 | 4 | sweep2 |

### What the three unconvergeable sites would need

Not a longer timeout by a small factor -- a different budget. `BN-beta|1`'s
5/4.0 rung alone took **5443 s**, and its 10/5.0, 15/6.0 and 20/7.5 rungs each
consumed the full 14400 s without finishing an SCF. `C-diamond|1` is the same
shape (5407 s at 5/4.0, then three timeouts). Before spending days of shared
compute on them, decide whether these sites need a converged ghost shell at
all, or whether a stated per-site cap with the uncertainty carried through is
enough. **That is a maintainer call, not a chat's.**

## Sweep 1 table (2026-07-28, vibe-basis 0.7.0 protocol)

Kept as run. The 12 sites re-run in sweep 2 are superseded by the merged table
above; the other 33 rows are the current answer for their sites.

| system | site | Z | verdict | at | dE vs densest | densest | pts |
|---|---|---|---|---|---|---|---|
| AlN | 1 | 13 | converged | 15/6.0 | +0.000 kJ/mol | 15/6.0 | 3 |
| AlN | 2 | 7 | converged | 10/5.0 | +0.082 kJ/mol | 15/6.0 | 3 |
| AlP | 1 | 13 | converged | 15/6.0 | +0.492 kJ/mol | 20/7.5 | 4 |
| AlP | 2 | 15 | converged | 15/6.0 | +0.155 kJ/mol | 30/10.0 | 5 |
| BN-beta | 1 | 5 | INSUFFICIENT | 5/4.0 | +0.000 kJ/mol | 5/4.0 | 1 |
| BN-beta | 2 | 7 | INSUFFICIENT | 5/4.0 | +0.000 kJ/mol | 5/4.0 | 1 |
| C-diamond | 1 | 6 | FAILED | - | - | - | 0 |
| CaF2 | 1 | 20 | converged | 20/7.5 | +0.000 kJ/mol | 20/7.5 | 4 |
| CaF2 | 2 | 9 | converged | 10/5.0 | +0.269 kJ/mol | 15/6.0 | 3 |
| CaO | 1 | 20 | converged | 15/6.0 | +0.254 kJ/mol | 20/7.5 | 4 |
| CaO | 2 | 8 | converged | 10/5.0 | +0.232 kJ/mol | 20/7.5 | 4 |
| GaAs | 1 | 31 | INSUFFICIENT | 10/5.0 | +0.000 kJ/mol | 10/5.0 | 2 |
| GaAs | 2 | 33 | converged | 10/5.0 | +0.161 kJ/mol | 15/6.0 | 3 |
| GaP | 1 | 31 | converged | 15/6.0 | +0.380 kJ/mol | 20/7.5 | 4 |
| GaP | 2 | 15 | converged | 10/5.0 | +0.318 kJ/mol | 15/6.0 | 3 |
| Ge | 1 | 32 | FAILED | - | - | - | 0 |
| K2O | 1 | 8 | converged | 5/4.0 | +0.309 kJ/mol | 20/7.5 | 4 |
| K2O | 2 | 19 | converged | 20/7.5 | +0.000 kJ/mol | 20/7.5 | 4 |
| K2S | 1 | 16 | converged | 5/4.0 | +0.206 kJ/mol | 20/7.5 | 4 |
| K2S | 2 | 19 | converged | 30/10.0 | +0.000 kJ/mol | 30/10.0 | 5 |
| KBr | 1 | 19 | converged | 20/7.5 | +0.000 kJ/mol | 20/7.5 | 4 |
| KBr | 2 | 35 | converged | 15/6.0 | +0.436 kJ/mol | 30/10.0 | 5 |
| KF | 1 | 19 | converged | 20/7.5 | +0.000 kJ/mol | 20/7.5 | 4 |
| KF | 2 | 9 | INSUFFICIENT | 5/4.0 | +0.000 kJ/mol | 5/4.0 | 1 |
| KH | 1 | 19 | converged | 30/10.0 | +0.000 kJ/mol | 30/10.0 | 5 |
| KH | 2 | 1 | INSUFFICIENT | 5/4.0 | +0.000 kJ/mol | 5/4.0 | 1 |
| LiCl | 1 | 3 | converged | 15/6.0 | +0.241 kJ/mol | 30/10.0 | 5 |
| LiCl | 2 | 17 | converged | 10/5.0 | +0.412 kJ/mol | 20/7.5 | 4 |
| LiF | 1 | 3 | converged | 10/5.0 | +0.049 kJ/mol | 15/6.0 | 3 |
| LiF | 2 | 9 | converged | 10/5.0 | +0.057 kJ/mol | 15/6.0 | 3 |
| LiH | 1 | 3 | converged | 15/6.0 | +0.135 kJ/mol | 20/7.5 | 4 |
| LiH | 2 | 1 | converged | 5/4.0 | +0.038 kJ/mol | 15/6.0 | 3 |
| MgO | 1 | 12 | converged | 15/6.0 | +0.000 kJ/mol | 15/6.0 | 3 |
| MgO | 2 | 8 | converged | 5/4.0 | +0.260 kJ/mol | 15/6.0 | 3 |
| Na2Se | 1 | 34 | converged | 10/5.0 | +0.447 kJ/mol | 20/7.5 | 4 |
| Na2Se | 2 | 11 | converged | 20/7.5 | +0.000 kJ/mol | 20/7.5 | 4 |
| NaCl | 1 | 11 | converged | 20/7.5 | +0.000 kJ/mol | 20/7.5 | 4 |
| NaCl | 2 | 17 | converged | 5/4.0 | +0.386 kJ/mol | 15/6.0 | 3 |
| NaF | 1 | 11 | converged | 15/6.0 | +0.000 kJ/mol | 15/6.0 | 3 |
| NaF | 2 | 9 | converged | 10/5.0 | +0.478 kJ/mol | 15/6.0 | 3 |
| NaH | 1 | 11 | converged | 15/6.0 | +0.376 kJ/mol | 20/7.5 | 4 |
| NaH | 2 | 1 | INSUFFICIENT | 5/4.0 | +0.138 kJ/mol | 10/5.0 | 2 |
| Si | 1 | 14 | FAILED | - | - | - | 0 |
| SiC-beta | 1 | 14 | converged | 15/6.0 | +0.000 kJ/mol | 15/6.0 | 3 |
| SiC-beta | 2 | 6 | converged | 10/5.0 | +0.407 kJ/mol | 15/6.0 | 3 |

## Provenance

* Both sweeps emitted by
  `vibe_basis.backends.crystal_atom.emit_input_atom_counterpoise`, run through
  CRYSTAL23, at **PBE / pob-TZVP-REV2**.
  * **Sweep 1** at vibe-basis 0.7.0, before the 0.8.0 protocol change: those
    decks carried `HUGEGRID` and no `TOLINTEG`. The assumption at the time was
    that converged shell extents are a basis-overlap property and only weakly
    grid-dependent, which is why converging at PBE was the plan. **Sweep 2
    tested that assumption and it held** -- see the protocol control above;
    the largest disagreement over nine comparable rungs is 0.011 kJ/mol.
  * **Sweep 2** at vibe-basis 0.9.0 with current defaults: `TOLINTEG
    9 9 9 18 54`, no `HUGEGRID`, plus the `ORIGIN`/`TRASREMO` prefix.
* Reference-set lattice constants, not the experimental ones in
  `vibe_basis.io.structures`.
* Nine systems were excluded before sweep 1 and remain excluded: six AFM, three
  needing transition-metal free-atom ground states the emitter refuses to
  guess.
* **The emitter's defaults are unchanged** (`nstar=30, rmax=10.0`, the
  reference set's LiF values). This study measures what each site needs; it
  does not itself select a per-site default, and nothing in the tree consumes
  `converged.json` yet.
* Sweep 1's workdir on compute-study is **109 GB**, essentially all CRYSTAL `fort.*`
  scratch. Do not `vq fetch --workdir` it; submit a 1-cpu probe job that reads
  it in place. `run_sweep.py` deletes the scratch per run as of 2026-08-01.
