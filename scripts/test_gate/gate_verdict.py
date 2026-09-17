#!/usr/bin/env python3
"""Gate verdict: compare a triage.jsonl run against a tracked-known-reds
baseline and decide green/red.

The gate is GREEN iff every non-PASS file is accounted for in the baseline
(a tracked, owned bug) — any *new* non-PASS file (a regression that slipped
onto main) turns it RED. This is what makes `main` reds unable to hide again.

baseline.json schema:
  {
    "known_reds": [
       {"file": "tests/x.py", "status": "FAIL", "tests": ["test_a"],
        "owner": "gdf-chat", "ref": "audit-2026-05-31/R3-R5",
        "reason": "multi-k GDF non-physical"}
    ],
    # files whose only "problem" is being slow (already @slow-marked, run on
    # the heavy shard) — informational, never gates:
    "known_slow": [{"file": "tests/test_direct_scf_smoke.py", "reason": "..."}]
  }

Usage: gate_verdict.py triage.jsonl baseline.json [--reverify --wt W --py P]
Exit 0 = green, 1 = red (new failures), 2 = usage error.

Contention re-verify (--reverify)
---------------------------------
The fast lane runs many files concurrently (`--jobs N`). On a shared box under
heavy load, a perfectly healthy file can be starved into a FAIL / ABORT /
TIMEOUT / OOM_KILLED / SIGKILLED that it does NOT reproduce when run alone — a *contention
artifact*, not a code regression. A v0.13.0-cut gate run produced exactly this:
~8 slow/memory-heavy files red-flagged under `--jobs 6` at box load ~80, every
one of which PASSED when re-run in isolation at `--jobs 1`.

`--reverify` makes the gate robust to that: before a NEW (unlisted) non-PASS
file is allowed to gate RED, it is re-run ONCE on its own (`--jobs 1`, a
generous timeout). It only stays red if it fails the second time too. This is
cheap by construction — only the (usually zero) NEW-red candidates are re-run,
never the whole suite, never the already-tracked known-red / known-slow files.
A file that recovers is reported as a contention artifact and does not gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

FAILING = {"FAIL", "SEGFAULT", "ABORT", "OOM_KILLED", "SIGKILLED", "TIMEOUT", "COLLECT_ERR"}

# Statuses that carry no information about the code, because no test ran. They
# are not reds and must never be counted as passes either: an artifact holding
# one cannot produce a verdict at all (#285).
VOID = {"STALE_CORE"}

# A re-verified single-file run is a contention artifact (does NOT gate) only
# if it now cleanly passes or collects nothing; anything else is a real red.
RECOVERED = {"PASS", "NOTESTS"}
BLOCKING_TIERS = {"T0", "T1"}


def load_jsonl(p):
    out = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def validate_blocking_artifact(runs, path):
    """Reject advisory, unclassified or void data before computing a verdict."""
    bad = sorted(
        {
            record.get("tier", "UNCLASSIFIED")
            for record in runs
            if record.get("tier") not in BLOCKING_TIERS
        }
    )
    if bad:
        labels = ", ".join(bad)
        raise ValueError(
            f"{path}: blocking verdict accepts only T0/T1 records; found {labels}"
        )

    # A stale compiled core is a condition of the host, not of any one file:
    # pytest refused before collecting, so every lane on that host ran nothing.
    # Reporting the run as GREEN would certify the previous commit's C++ under
    # this commit's name, and reporting it RED would charge the code for a
    # build problem. Neither is a verdict, so refuse to compute one (#285).
    voided = sorted({r.get("file", "?") for r in runs if r.get("status") in VOID})
    if voided:
        shown = ", ".join(voided[:3]) + (" ..." if len(voided) > 3 else "")
        raise ValueError(
            f"{path}: {len(voided)} target(s) exited STALE_CORE, so no test ran "
            f"on this host ({shown}). Rebuild the core and re-run the lane; "
            "this artifact cannot produce a verdict either way"
        )


def is_failing(status: str) -> bool:
    """A status that, on an unlisted file, would gate the build RED.

    ``VOID`` statuses are included so that no path can quietly count them as
    healthy. A blocking run never reaches here with one — it is rejected in
    :func:`validate_blocking_artifact` — but an advisory lane still has to show
    it rather than pass over it (#285).
    """
    return (
        status in FAILING
        or status in VOID
        or status.startswith(("SIGNAL_", "RC_"))
    )


def reverify_new_reds(candidates, runner):
    """Re-run each candidate NEW-red file once in isolation to rule out a
    contention artifact.

    A FAIL / ABORT / TIMEOUT / OOM_KILLED / SIGKILLED produced under high concurrency on a
    shared box is not evidence of a code regression if the file passes when run
    alone. This re-runs each candidate a single time (the caller supplies an
    isolated `--jobs 1`, generous-timeout runner) and partitions the result.

    Parameters
    ----------
    candidates : list[dict]
        Run-records (each a dict with at least ``"file"`` and ``"status"``)
        that would otherwise gate RED — i.e. the NEW, unlisted non-PASS files.
    runner : Callable[[str], dict | None]
        Given a file path (the record's ``"file"``), perform ONE isolated
        re-run and return its run-record (with a ``"status"`` key), or ``None``
        if the re-run itself could not be carried out.

    Returns
    -------
    (still_red, recovered) : tuple[list[dict], list[tuple[dict, dict | None]]]
        ``still_red`` is the subset of ``candidates`` that failed again (real
        reds — these gate). ``recovered`` is a list of
        ``(original_record, reverify_record)`` for files that PASSED (or
        collected NOTESTS) on the isolated re-run — contention artifacts that
        must NOT gate. A runner that returns ``None`` (or any non-passing
        status) is treated conservatively as still-red: an inconclusive re-run
        never clears a red.
    """
    still_red, recovered = [], []
    for rec in candidates:
        res = runner(rec["file"])
        status = (res or {}).get("status")
        if status in RECOVERED:
            recovered.append((rec, res))
        else:
            still_red.append(rec)
    return still_red, recovered


def _build_isolated_runner(args):
    """Wire `reverify_new_reds`'s `runner` to the per-file isolation runner
    from run_full_suite.py, configured for a single-file isolated re-run
    (`--jobs 1` is implicit — one file, one process — with a generous
    timeout). Imported lazily so the module imports cleanly without a built
    vibe-qc (the pure logic above is unit-testable on its own)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pathlib import Path

    from run_full_suite import run_one  # noqa: E402 (lazy by design)

    wt = Path(args.wt).resolve()

    def runner(rel: str):
        print(
            f"  [reverify] {rel}: re-running alone "
            f"(file_timeout={args.reverify_file_timeout:.0f}s, "
            f"test_timeout={args.reverify_test_timeout}s, "
            f"markexpr={args.markexpr!r}) ...",
            flush=True,
        )
        res = run_one(
            rel,
            wt,
            args.py,
            args.reverify_file_timeout,
            args.reverify_test_timeout,
            args.markexpr,
            args.heavy_env,
        )
        print(
            f"  [reverify] {rel}: -> {res['status']} "
            f"({res['elapsed_s']}s, {res['peak_rss_mb']}MB)",
            flush=True,
        )
        return res

    return runner


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("triage", help="triage.jsonl from run_full_suite.py")
    ap.add_argument("baseline", help="known_reds_baseline.json")
    ap.add_argument("--reverify", action="store_true",
                    help="before reding a NEW (unlisted) non-PASS file, re-run "
                         "it ONCE in isolation; only red it if it fails again "
                         "(contention-robust). Requires --wt and --py.")
    ap.add_argument("--wt", help="worktree under test (required with --reverify)")
    ap.add_argument("--py", help="venv python (required with --reverify)")
    ap.add_argument("--reverify-file-timeout", type=float, default=1800.0,
                    help="hard per-file wall-clock kill for the isolated re-run "
                         "(default: 1800s — generous, so the re-run itself "
                         "can't false-TIMEOUT)")
    ap.add_argument("--reverify-test-timeout", type=int, default=1500,
                    help="per-test timeout for the isolated re-run (default: 1500s)")
    ap.add_argument("--markexpr", default="",
                    help="pytest -m expression for the re-run; set it to match "
                         "the lane being verdicted (e.g. 'not slow') so the "
                         "re-run exercises the same tests")
    ap.add_argument("--heavy-env", action="store_true",
                    help="set VIBEQC_RUN_HEAVY_TESTS=1 on the re-run (mirror the "
                         "slow/heavy lane)")
    args = ap.parse_args(argv)

    if args.reverify and not (args.wt and args.py):
        ap.error("--reverify requires --wt and --py")

    runs = load_jsonl(args.triage)
    try:
        validate_blocking_artifact(runs, args.triage)
    except ValueError as exc:
        print(f"gate input rejected: {exc}", file=sys.stderr)
        return 2
    with open(args.baseline) as fh:
        baseline = json.load(fh)
    known = {r["file"]: r for r in baseline.get("known_reds", [])}
    # known_slow: files that exceed the fast-lane timeout — slow, not broken.
    # They do NOT gate; they run on the slow/nightly lane (generous timeout) and
    # carry a perf follow-up (E2/E3 driver-efficiency work).
    known_slow = {r["file"]: r for r in baseline.get("known_slow", [])}

    new_red, known_red, known_slow_hit, signal_anom = [], [], [], []
    n_pass = n_notests = 0
    for r in runs:
        st = r["status"]
        f = r["file"]
        if st == "PASS":
            n_pass += 1
            continue
        if st == "NOTESTS":
            n_notests += 1
            continue
        if st.startswith("SIGNAL_") or st.startswith("RC_"):
            signal_anom.append(r)
        if f in known:
            known_red.append(r)
        elif f in known_slow:
            known_slow_hit.append(r)
        elif is_failing(st):
            new_red.append(r)

    # A baseline entry that is now PASSing should be cleaned up (xpass / fixed).
    run_status = {r["file"]: r["status"] for r in runs}
    resolved = [f for f in {**known, **known_slow} if run_status.get(f) == "PASS"]

    # Contention re-verify: a NEW red on a shared box may be a starvation
    # artifact. Re-run each candidate alone before letting it gate. Cheap —
    # only the (usually zero) NEW-red candidates are re-run.
    recovered = []
    if args.reverify and new_red:
        print(f"\n-- re-verifying {len(new_red)} NEW non-PASS file(s) in "
              f"isolation before reding --", flush=True)
        runner = _build_isolated_runner(args)
        new_red, recovered = reverify_new_reds(new_red, runner)

    print("\n===== GATE VERDICT =====")
    print(f"pass={n_pass}  notests={n_notests}  known-red={len(known_red)}  "
          f"known-slow={len(known_slow_hit)}  NEW-red={len(new_red)}  "
          f"recovered={len(recovered)}  resolved={len(resolved)}")
    if known_red:
        print("\n-- tracked-known reds (do not gate) --")
        for r in sorted(known_red, key=lambda x: x["file"]):
            k = known[r["file"]]
            print(f"  {r['status']:11s} {r['file']}  [{k.get('owner','?')}] {k.get('reason','')}")
    if known_slow_hit:
        print(f"\n-- known-slow (do not gate; slow/nightly lane + perf follow-up): {len(known_slow_hit)} --")
    if recovered:
        print("\n-- contention artifacts (non-PASS under load, PASS alone — do NOT gate) --")
        for orig, res in sorted(recovered, key=lambda x: x[0]["file"]):
            rstat = (res or {}).get("status", "?")
            print(f"  {orig['status']:11s}->{rstat:8s} {orig['file']}")
    if resolved:
        print("\n-- baseline entries now PASSING (remove from baseline) --")
        for f in resolved:
            print(f"  {f}")
    if new_red:
        print("\n-- !!! NEW reds (gate RED) !!! --")
        for r in sorted(new_red, key=lambda x: x["file"]):
            print(f"  {r['status']:11s} {r['file']}  rc={r['rc']} {r['elapsed_s']}s "
                  f"{r['peak_rss_mb']}MB counts={r.get('counts')}")
    print("========================")
    return 1 if new_red else 0


if __name__ == "__main__":
    sys.exit(main())
