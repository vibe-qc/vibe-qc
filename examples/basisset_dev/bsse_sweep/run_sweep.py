"""Converge the ATOMBSSE ghost shell per system, on a vq host.

Self-contained payload. Ships its own ``vibe_basis`` source and
pre-generated inline basis blocks, so it needs neither an installed
vibe-basis nor vibe-qc on the host: only a CRYSTAL binary and stdlib
Python.

For each system, each atom site, and each ``(nstar, rmax)`` on the
ladder, it emits a counterpoise free-atom deck, runs CRYSTAL, and
records the energy. The converged setting is the cheapest point whose
energy is within ``TOL_KJ`` of the densest one.

If the payload carries ``sites.json`` the sweep runs only the
``"System|site"`` entries it lists -- that is how a re-run targets the
sites a previous sweep could not settle without recomputing the rest.

Everything is written under ``$VQ_WORKDIR`` (CLAUDE.md section 15), and
the run is resumable: completed points are read back from
``results.json`` and skipped.

Environment
-----------
``SWEEP_FUNCTIONAL``   CRYSTAL method keyword (default ``pbe``).
``SWEEP_WORKERS``      concurrent CRYSTAL processes (default 8). Keep at or
                       under the host's per-submitter CPU quota, not its
                       core count.
``SWEEP_TIMEOUT_S``    per-run wall clock (default 5400).
``SWEEP_KEEP_SCRATCH`` keep CRYSTAL's ``fort.*`` scratch. Off by default:
                       the first full sweep left a 109 GB workdir, nearly
                       all of it scratch rather than output.
``SWEEP_SEED_RESULTS`` colon-separated ``results.json`` paths from earlier
                       jobs to resume from. A resubmission gets a new
                       workdir, so without this it recomputes points an
                       earlier job already finished. Seed only across runs
                       at the **same protocol**.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from vibe_basis.backends.crystal_atom import emit_input_atom_counterpoise  # noqa: E402
from vibe_basis.io.structures import STRUCTURES  # noqa: E402

HA_KJ = 2625.4996394798254
E_RE = re.compile(r"TOTAL ENERGY\([A-Z]+\)\(AU\)\(\s*\d+\)\s+(-?\d+\.\d+E[+-]\d+)")
SCF_OK = re.compile(r"SCF ENDED - CONVERGENCE ON (ENERGY|DENSITY MATRIX)")
#: The SCF energy going NaN. Distinct from every other failure: the run
#: is neither out of time nor rejected at input, the arithmetic broke.
#: In the first sweep this always went on to a SIGSEGV inside CRYSTAL,
#: so the output carries no ``ERROR ****`` line and the run would
#: otherwise be filed as an unexplained ``no_energy``.
SCF_NAN = re.compile(r"ETOT\(AU\)\s+NaN")

#: Ghost-shell ladder. The reference set spans 10/5.0 to 60/10.0.
#: Override with ``SWEEP_LADDER="5:4.0,10:5.0"`` to probe a shorter or a
#: denser range; the keys in ``results.json`` carry the rung, so a run at
#: one ladder resumes cleanly alongside a run at another.
_DEFAULT_LADDER = "5:4.0,10:5.0,15:6.0,20:7.5,30:10.0"
LADDER = [
    (int(n), float(r))
    for n, _, r in (
        rung.partition(":")
        for rung in os.environ.get("SWEEP_LADDER", _DEFAULT_LADDER).split(",")
    )
]

#: A point counts as converged when it sits this close to the densest.
TOL_KJ = 0.5

#: Per-run wall clock. A symmetry-free ghost cluster is not cheap, and a
#: single pathological atom must not hold the whole sweep. Raise it for a
#: re-run of the dense rungs, where the first sweep's 5400 s was itself
#: the binding constraint on 32 of 225 runs.
TIMEOUT_S = int(os.environ.get("SWEEP_TIMEOUT_S", "5400"))

FUNCTIONAL = os.environ.get("SWEEP_FUNCTIONAL", "pbe")
WORKERS = int(os.environ.get("SWEEP_WORKERS", "8"))
KEEP_SCRATCH = bool(os.environ.get("SWEEP_KEEP_SCRATCH"))

WORK = Path(os.environ.get("VQ_WORKDIR") or (HERE / "work"))
RESULTS = WORK / "results.json"


def crystal_binary() -> str:
    """Locate CRYSTAL without hardcoding anyone's home directory."""
    import shutil

    for name in ("crystal23", "crystal"):
        found = shutil.which(name)
        if found:
            return found
        candidate = Path.home() / "bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise SystemExit("no CRYSTAL binary found on PATH or in ~/bin")


CRYSTAL = crystal_binary()


def run_one(task):
    system, idx, Z, nstar, rmax, deck = task
    key = f"{system}|{idx}|{nstar}|{rmax}"
    d = WORK / "runs" / f"{system}_site{idx}_{nstar}_{rmax}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "INPUT").write_text(deck)
    t0 = time.perf_counter()
    try:
        with (d / "INPUT").open() as fh, (d / "out.txt").open("w") as out:
            subprocess.run([CRYSTAL], stdin=fh, stdout=out,
                           stderr=subprocess.STDOUT, timeout=TIMEOUT_S, cwd=d)
    except subprocess.TimeoutExpired:
        _tidy(d, keep_out=True)
        return key, dict(system=system, site=idx, Z=Z, nstar=nstar, rmax=rmax,
                         energy=None, status="timeout",
                         seconds=time.perf_counter() - t0)
    text = (d / "out.txt").read_text(errors="ignore")
    hits = E_RE.findall(text)
    if hits and SCF_OK.search(text):
        status = "ok"
    elif SCF_NAN.search(text):
        # Checked before "did we see any energy at all": a NaN run has
        # usually printed a good cycle-0 energy first, so it would
        # otherwise be misfiled as not_converged and its number believed.
        status = "scf_nan"
    elif hits:
        status = "not_converged"
    else:
        status = "no_energy"
        for line in text.splitlines():
            if "ERROR ****" in line:
                status = line.strip()[:80]
                break
    # Large outputs are the bulk of the footprint; keep only failures.
    _tidy(d, keep_out=status != "ok")
    return key, dict(system=system, site=idx, Z=Z, nstar=nstar, rmax=rmax,
                     energy=None if status != "ok" else float(hits[-1]),
                     status=status, seconds=time.perf_counter() - t0)


def _tidy(d: Path, *, keep_out: bool) -> None:
    """Drop CRYSTAL's scratch, and the output too once it is not needed.

    The first full sweep left a **109 GB** workdir across 225 runs, almost
    none of it the ``out.txt`` the previous version was careful to delete:
    it is the ``fort.*`` scratch a symmetry-free ghost cluster writes. On a
    shared host that is the part worth cleaning up.
    """
    if not keep_out:
        (d / "out.txt").unlink(missing_ok=True)
    if KEEP_SCRATCH:
        return
    for f in d.iterdir():
        if f.is_file() and f.name not in ("INPUT", "out.txt"):
            f.unlink(missing_ok=True)


def build_tasks(results):
    """Every site's cheapest rung first, then everyone's next rung up.

    Ladder-major, not system-major. The runs on this ladder differ in cost
    by two orders of magnitude, so a system-major order spends the first
    several hours on one system's densest rungs and leaves every other site
    with nothing at all. Ladder-major means that at any moment the sweep
    holds a **uniform-depth** picture across all sites -- every site
    converged over rungs 1..k -- which is a partial result somebody can
    read, and which degrades gracefully if the job is killed.

    It costs a little makespan against longest-job-first, since the densest
    rungs start last. That is the trade being made deliberately.
    """
    tasks = []
    for system in sorted(SYSTEMS):
        struct = STRUCTURES[system]
        basis = (HERE / "bases" / f"{system}.txt").read_text()
        for idx, atom in enumerate(struct.crystal_asymm_unit, start=1):
            if SITES is not None and f"{system}|{idx}" not in SITES:
                continue
            for nstar, rmax in LADDER:
                key = f"{system}|{idx}|{nstar}|{rmax}"
                if key in results:
                    continue
                deck = emit_input_atom_counterpoise(
                    struct, idx, basis, method=FUNCTIONAL,
                    nstar=nstar, rmax=rmax)
                if deck is None:
                    results[key] = dict(system=system, site=idx, Z=atom.Z,
                                        nstar=nstar, rmax=rmax, energy=None,
                                        status="emit_failed", seconds=0.0)
                    continue
                tasks.append((system, idx, atom.Z, nstar, rmax, deck))
    rung = {r: i for i, r in enumerate(LADDER)}
    tasks.sort(key=lambda t: (rung[(t[3], t[4])], t[0], t[1]))
    return tasks


def summarise(results):
    """Cheapest ladder point within TOL_KJ of the densest, per site."""
    by_site = {}
    for r in results.values():
        by_site.setdefault((r["system"], r["site"], r["Z"]), []).append(r)
    # MIN_POINTS: below this, "within TOL of the densest" is not a
    # convergence claim -- with one surviving point the densest IS the
    # chosen one and the difference is trivially zero. The dense rungs
    # are exactly the ones that fail, so this is common, not hypothetical.
    MIN_POINTS = 3
    lines = ["| system | site | Z | verdict | at | dE vs densest | densest | pts |",
             "|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for (system, site, Z), rows in sorted(by_site.items()):
        good = sorted((r for r in rows if r["status"] == "ok" and r["energy"]),
                      key=lambda r: (r["nstar"], r["rmax"]))
        if not good:
            lines.append(f"| {system} | {site} | {Z} | FAILED | | | | 0 |")
            verdict[f"{system}|{site}"] = dict(Z=Z, verdict="failed", n_points=0)
            continue
        ref = good[-1]
        chosen = next(
            (r for r in good
             if abs(r["energy"] - ref["energy"]) * HA_KJ <= TOL_KJ), ref)
        d = (chosen["energy"] - ref["energy"]) * HA_KJ
        v = "converged" if len(good) >= MIN_POINTS else "INSUFFICIENT"
        lines.append(f"| {system} | {site} | {Z} | {v} "
                     f"| {chosen['nstar']}/{chosen['rmax']} "
                     f"| {d:+.3f} kJ/mol | {ref['nstar']}/{ref['rmax']} "
                     f"| {len(good)} |")
        verdict[f"{system}|{site}"] = dict(Z=Z, verdict=v.lower(),
                                           nstar=chosen["nstar"],
                                           rmax=chosen["rmax"],
                                           delta_kj=d, n_points=len(good))
    return "\n".join(lines), verdict


def seed_results():
    """Points carried in from earlier jobs, via ``SWEEP_SEED_RESULTS``.

    A resubmission gets a **new** ``$VQ_WORKDIR``, so the in-workdir
    ``results.json`` that makes a single job resumable does nothing across
    job boundaries: a job killed to fix its worker count or its task order
    would recompute everything it had already finished. Colon-separated
    paths to previous ``results.json`` files are merged in first, later
    paths winning, and the sweep then skips those keys.

    Seed only from runs whose **protocol matches** -- the key carries the
    system, site and rung but says nothing about the functional, the
    integral tolerances or the emitter version. Seeding across a protocol
    change would silently blend two studies.
    """
    seeded = {}
    for path in filter(None, os.environ.get("SWEEP_SEED_RESULTS", "").split(":")):
        p = Path(path)
        if not p.is_file():
            print(f"seed: {p} not found, skipping", flush=True)
            continue
        loaded = json.loads(p.read_text())
        seeded.update(loaded)
        print(f"seed: {len(loaded)} points from {p}", flush=True)
    return seeded


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    results = seed_results()
    if RESULTS.is_file():
        results.update(json.loads(RESULTS.read_text()))
    print(f"CRYSTAL: {CRYSTAL}", flush=True)
    print(f"functional={FUNCTIONAL} workers={WORKERS} tol={TOL_KJ} kJ/mol "
          f"timeout={TIMEOUT_S}s", flush=True)
    print(f"ladder={LADDER}", flush=True)
    print(f"systems={len(SYSTEMS)} "
          f"sites={'all' if SITES is None else len(SITES)} "
          f"resumed={len(results)} workdir={WORK}", flush=True)

    tasks = build_tasks(results)
    print(f"{len(tasks)} runs to do", flush=True)
    # as_completed, not map: map yields strictly in submission order, so a
    # single slow early task withholds every result behind it. That is not
    # just cosmetic here -- results.json is written from this loop, so an
    # in-order yield means a killed job resumes from the slowest point
    # rather than from what actually finished. Cheapest ladder points
    # complete in minutes and the densest take hours, so the gap is large.
    done = 0
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        for fut in cf.as_completed(futures):
            key, rec = fut.result()
            results[key] = rec
            done += 1
            RESULTS.write_text(json.dumps(results, indent=1, sort_keys=True))
            print(f"[{done}/{len(tasks)}] {key} {rec['status']} "
                  f"{rec['seconds']:.0f}s", flush=True)

    table, verdict = summarise(results)
    tally = {}
    for r in results.values():
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    breakdown = "\n".join(f"* `{s}` -- {n}" for s, n in sorted(tally.items()))
    (WORK / "converged.json").write_text(json.dumps(verdict, indent=1, sort_keys=True))
    (WORK / "SUMMARY.md").write_text(
        f"# ATOMBSSE ghost-shell convergence ({FUNCTIONAL}, pob-TZVP-REV2)\n\n"
        f"Converged = cheapest ladder point within {TOL_KJ} kJ/mol of the "
        f"densest, over at least 3 successful points.\n\n"
        f"Ladder `{LADDER}`, timeout {TIMEOUT_S} s, "
        f"{'all sites' if SITES is None else f'{len(SITES)} selected sites'}.\n\n"
        f"## Run outcomes\n\n{breakdown}\n\n## Table\n\n"
        + table + "\n")
    print("\n" + breakdown + "\n\n" + table, flush=True)


SYSTEMS = json.loads((HERE / "systems.json").read_text())

#: Optional ``["System|site", ...]`` restriction written by
#: ``build_payload.py --sites``. ``None`` means every site of every system.
_SITES_FILE = HERE / "sites.json"
SITES = (set(json.loads(_SITES_FILE.read_text()))
         if _SITES_FILE.is_file() else None)

if __name__ == "__main__":
    main()
