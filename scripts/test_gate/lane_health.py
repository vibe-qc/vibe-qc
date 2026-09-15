#!/usr/bin/env python3
"""Lane health: a REPORT-ONLY view of which test files are red on `main`.

This is deliberately not a gate. There is no per-commit CI (CLAUDE.md par. 2)
and the once-per-release pipeline runs the blocking T0+T1 profile only, so a
commit that reddens a T2/T3 lane produces no signal at all — not at push time,
not at release time — until someone happens to run that lane for an unrelated
reason. IID 450 collected six such lanes found by accident in two days, one of
them a default-path SCF convergence regression that shipped in three releases.

The information needed to spot them already exists: `run_full_suite.py` can run
the whole inventory with per-file process isolation, and `suite_manifest.json`
knows every file's tier and owner. What was missing is something that runs
periodically and *publishes the list*. That is this script.

What it answers, per red file:

  * which file, at what status, in which tier, owned by whom;
  * is this a tracked known red (`known_reds_baseline.json`) or a new one;
  * **when did it last pass, and what landed since** — the commit range and the
    commits in it that touched the file.

And, just as importantly, what it did NOT observe: a file in the inventory with
no record in the supplied artifacts is reported as `not_observed`, because "the
base rate is unknown" is the actual complaint IID 450 makes. A health report
that quietly omits what it never ran would reproduce the very blind spot it
exists to remove.

Not a gate, by construction
---------------------------
`gate_verdict.py` refuses any artifact carrying T2/T3 records, so advisory data
can never redden a cut. This script is the mirror image: it accepts every tier
and **always exits 0** on a successful report, however red the tree is. Exit 2
is a usage error. Nothing here may be wired into a blocking path; use
`gate_verdict.py` for verdicts.

Usage
-----
    $PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
        --tier T0,T1,T2,T3 --out health.jsonl
    $PY scripts/test_gate/lane_health.py health.jsonl \
        --wt "$WT" --out LANE_HEALTH.md --json-out lane_health.json

Feeding the previous run's `lane_health.json` back in with `--previous`
narrows "last passed" for files the manifest has never recorded a verified SHA
for: a file observed green by the last report passed at that report's SHA.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1

# Same failing-status vocabulary the blocking verdict uses. Kept as an import
# so the two can never drift into disagreeing about what "red" means.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_verdict import is_failing  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "test_gate" / "suite_manifest.json"
DEFAULT_BASELINE = REPO_ROOT / "scripts" / "test_gate" / "known_reds_baseline.json"


def load_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def latest_records(runs: list[dict]) -> dict[str, dict]:
    """Collapse run-records to one per file, last write wins.

    `run_full_suite.py` writes a *resumable* artifact: a re-run appends fresh
    records for the targets it repeated, so the same file can appear more than
    once and only the last one describes the tree as it now stands. Node-id
    targets (``tests/x.py::test_y``) are folded onto their file, and a red node
    keeps the file red even if a later record for a different node passed.
    """
    latest: dict[str, dict] = {}
    for rec in runs:
        target = rec.get("file", "")
        if not target:
            continue
        path = target.split("::", 1)[0]
        prior = latest.get(path)
        if prior is None:
            latest[path] = dict(rec, file=path)
            continue
        # A per-node artifact reports one node at a time; a red node must not be
        # erased by a green sibling recorded after it.
        if "::" in target and is_failing(rec.get("status", "")):
            latest[path] = dict(rec, file=path)
        elif "::" not in target:
            latest[path] = dict(rec, file=path)
    return latest


# Mirrors the pattern and allowlist the pre-commit hook enforces
# (.githooks/pre-commit), so a report can be committed without tripping it.
_FOREIGN_HOME = re.compile(r"/(?:Users|home)/(?!runner/|root/|user/|USER/)[a-z][A-Za-z0-9_.-]*")


def sanitize(text: str, worktree: Path | None) -> str:
    """Strip absolute local paths out of text destined for a published report.

    CLAUDE.md par. 12 forbids author home paths in committed content, and this
    report is meant to be published. pytest tails carry them routinely
    (rootdir lines, tracebacks), so redaction happens here rather than being
    left to whoever commits the artifact.

    The generic sweep at the end matters because a tail can carry a home that
    is neither this process's nor the worktree's -- a path baked into a fixture
    by whoever produced the artifact, on another box.
    """
    if worktree is not None:
        text = text.replace(str(worktree), "<repo>")
    home = str(Path.home())
    if home and home != "/":
        text = text.replace(home, "~")
    return _FOREIGN_HOME.sub("/home/USER", text)


def tail_excerpt(record: dict, worktree: Path | None, lines: int = 5) -> str:
    raw = record.get("tail") or ""
    kept = [ln for ln in raw.splitlines() if ln.strip()][-lines:]
    return sanitize("\n".join(kept), worktree)


def load_manifest_rows(path: str | Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["file"]: row for row in data["tests"]}


def load_baseline_index(path: str | Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {entry["file"]: entry for entry in data.get("known_reds", [])}


def resolve_last_pass(
    file: str,
    manifest_row: dict | None,
    previous_last_pass: dict[str, str],
) -> tuple[str | None, str]:
    """Where the file was last seen green, and on whose authority.

    A previous health report is preferred over the manifest: it is a direct
    observation at a known SHA, whereas `last_verified_sha` is whatever the
    last measurement import happened to stamp.
    """
    prior = previous_last_pass.get(file)
    if prior:
        return prior, "previous lane-health report"
    verified = (manifest_row or {}).get("last_verified_sha")
    if verified:
        return verified, "suite_manifest last_verified_sha"
    return None, "unknown — no recorded green"


def _git(worktree: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(worktree), *args],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def make_git_probe(worktree: Path | None):
    """Return `probe(since_sha, run_sha, file) -> (n_commits, suspects)`.

    Injected rather than called inline so the report logic is unit-testable
    without a git worktree. `suspects` are the commits in the range that
    touched the test file itself — the cheap, high-signal subset of the range.
    """
    if worktree is None:
        return lambda since, run_sha, file: (None, [])

    def probe(since: str, run_sha: str, file: str):
        if not since or not run_sha:
            return None, []
        rng = f"{since}..{run_sha}"
        count_out = _git(worktree, "rev-list", "--count", rng)
        try:
            count = int(count_out)
        except ValueError:
            # An unreachable SHA (shallow clone, rewritten history) is reported
            # as an unknown range rather than silently as zero commits.
            return None, []
        log_out = _git(worktree, "log", "--oneline", "--no-decorate", rng, "--", file)
        suspects = [ln for ln in log_out.splitlines() if ln.strip()]
        return count, suspects

    return probe


def build_report(
    records: dict[str, dict],
    manifest_rows: dict[str, dict],
    baseline: dict[str, dict],
    run_sha: str,
    *,
    previous: dict | None = None,
    git_probe=None,
    worktree: Path | None = None,
    max_suspects: int = 10,
    sources: list[str] | None = None,
) -> dict:
    """Assemble the health report. Pure apart from the injected `git_probe`."""
    probe = git_probe or (lambda since, run, file: (None, []))
    previous_last_pass = dict((previous or {}).get("last_pass", {}))

    red, passing, recovered = [], [], []
    for file in sorted(records):
        rec = records[file]
        status = rec.get("status", "")
        row = manifest_rows.get(file)
        if not is_failing(status):
            passing.append(file)
            if file in baseline:
                # gate_verdict says the same thing for blocking tiers; saying it
                # here too is what keeps the baseline from silently rotting on
                # the ~90% of the inventory the gate never runs.
                recovered.append({
                    "file": file,
                    "status": status,
                    "baseline_owner": baseline[file].get("owner"),
                    "baseline_reason": baseline[file].get("reason"),
                })
            continue
        since, since_source = resolve_last_pass(file, row, previous_last_pass)
        n_commits, suspects = probe(since, run_sha, file) if since else (None, [])
        entry = {
            "file": file,
            "status": status,
            "tier": rec.get("tier") or (row or {}).get("tier"),
            "maturity": (row or {}).get("maturity"),
            "owner": (row or {}).get("owner"),
            "in_inventory": row is not None,
            "tracked_known_red": file in baseline,
            "baseline_owner": baseline.get(file, {}).get("owner"),
            "baseline_reason": baseline.get(file, {}).get("reason"),
            "counts": rec.get("counts"),
            "last_pass_sha": since,
            "last_pass_source": since_source,
            "commit_range": f"{since}..{run_sha}" if since else None,
            "commits_since_last_pass": n_commits,
            "suspect_commits": suspects[:max_suspects],
            "suspect_commits_truncated": len(suspects) > max_suspects,
            "tail_excerpt": tail_excerpt(rec, worktree),
        }
        red.append(entry)

    observed = set(records)
    not_observed = sorted(set(manifest_rows) - observed)

    # Carry the last-known-green SHA forward so successive reports converge on a
    # tight range even for files the manifest never stamped.
    last_pass = dict(previous_last_pass)
    for file in passing:
        last_pass[file] = run_sha

    by_tier: dict[str, dict[str, int]] = {}
    for file, row in manifest_rows.items():
        tier = row.get("tier", "?")
        bucket = by_tier.setdefault(tier, {"inventory": 0, "observed": 0, "red": 0})
        bucket["inventory"] += 1
        if file in observed:
            bucket["observed"] += 1
    for entry in red:
        tier = entry.get("tier") or "?"
        by_tier.setdefault(tier, {"inventory": 0, "observed": 0, "red": 0})["red"] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "role": "report-only; never gates a push, a cut, or a tag",
        "run_sha": run_sha,
        "sources": sources or [],
        "coverage": {
            "inventory": len(manifest_rows),
            "observed": len(observed & set(manifest_rows)),
            "not_observed": len(not_observed),
            "observed_outside_inventory": sorted(observed - set(manifest_rows)),
            "by_tier": by_tier,
        },
        "red": red,
        "recovered": sorted(recovered, key=lambda e: e["file"]),
        "not_observed": [
            {"file": f, "tier": manifest_rows[f].get("tier"),
             "owner": manifest_rows[f].get("owner")}
            for f in not_observed
        ],
        "passing_count": len(passing),
        "last_pass": last_pass,
    }


def render_markdown(report: dict) -> str:
    cov = report["coverage"]
    out = [
        "# Lane health",
        "",
        f"Run SHA: `{report['run_sha']}`  ",
        f"Role: {report['role']}",
        "",
        f"Inventory {cov['inventory']} files — observed {cov['observed']}, "
        f"not observed {cov['not_observed']}, red {len(report['red'])}, "
        f"passing {report['passing_count']}.",
        "",
    ]
    if cov["not_observed"]:
        n = cov["not_observed"]
        plural = "file" if n == 1 else "files"
        out += [
            "> This report describes only what was run. "
            f"{n} inventory {plural} produced no record and "
            f"{'is' if n == 1 else 'are'} listed under *Not observed*; "
            "their health is unknown, not green.",
            "",
        ]
    out += ["## Red lanes", ""]
    if not report["red"]:
        out += ["None in the observed set.", ""]
    else:
        out += [
            "| file | status | tier | owner | tracked | last passed | commits since |",
            "|---|---|---|---|---|---|---|",
        ]
        for e in report["red"]:
            since = e["last_pass_sha"][:12] if e["last_pass_sha"] else "unknown"
            n = e["commits_since_last_pass"]
            out.append(
                f"| `{e['file']}` | {e['status']} | {e['tier'] or '?'} | "
                f"{e['owner'] or '?'} | {'yes' if e['tracked_known_red'] else '**NEW**'} | "
                f"`{since}` | {n if n is not None else '?'} |"
            )
        out.append("")
        for e in report["red"]:
            out += [f"### `{e['file']}` — {e['status']}", ""]
            if e["tracked_known_red"]:
                out.append(
                    f"Tracked known red (owner: {e['baseline_owner'] or '?'}): "
                    f"{e['baseline_reason'] or ''}".rstrip()
                )
            else:
                out.append("**Not in the known-reds baseline** — untracked red.")
            out.append("")
            out.append(f"Last passed: `{e['last_pass_sha'] or 'unknown'}` "
                       f"({e['last_pass_source']}).")
            if e["commit_range"]:
                n = e["commits_since_last_pass"]
                span = (
                    f"{n} commit{'' if n == 1 else 's'}" if n is not None
                    else "commit count unavailable — the anchor is not "
                         "reachable from this checkout"
                )
                out.append(f"Range since: `{e['commit_range']}` ({span}).")
            if e["suspect_commits"]:
                out += ["", "Commits in that range touching this file:", ""]
                out += [f"* `{c}`" for c in e["suspect_commits"]]
                if e["suspect_commits_truncated"]:
                    out.append("* …")
            elif e["commits_since_last_pass"]:
                out += ["", "No commit in that range touched this file, so the "
                            "break came from code it exercises, not from the "
                            "test."]
            if e["tail_excerpt"]:
                out += ["", "```", e["tail_excerpt"], "```"]
            out.append("")
    if report["recovered"]:
        out += ["## Baseline entries now passing (remove them)", ""]
        out += [f"* `{e['file']}` — owner {e['baseline_owner'] or '?'}"
                for e in report["recovered"]]
        out.append("")
    if report["not_observed"]:
        out += ["## Not observed", "",
                "Inventory files with no record in the supplied artifacts.", ""]
        out += [f"* `{e['file']}` ({e['tier'] or '?'}, {e['owner'] or '?'})"
                for e in report["not_observed"]]
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("triage", nargs="+",
                    help="one or more triage.jsonl artifacts from run_full_suite.py")
    ap.add_argument("--suite-manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--wt", default=None,
                    help="worktree the artifacts were produced in; used for the "
                         "commit-range lookups and for path redaction")
    ap.add_argument("--run-sha", default=None,
                    help="commit the artifacts were produced at "
                         "(default: HEAD of --wt)")
    ap.add_argument("--previous", default=None,
                    help="a previous lane_health.json, to sharpen 'last passed'")
    ap.add_argument("--out", default=None, help="Markdown report path (default: stdout)")
    ap.add_argument("--json-out", default=None, help="machine-readable report path")
    ap.add_argument("--max-suspects", type=int, default=10)
    args = ap.parse_args(argv)

    worktree = Path(args.wt).resolve() if args.wt else None
    run_sha = args.run_sha
    if not run_sha and worktree is not None:
        run_sha = _git(worktree, "rev-parse", "HEAD")
    if not run_sha:
        run_sha = "unknown"

    runs: list[dict] = []
    for path in args.triage:
        runs.extend(load_jsonl(path))

    previous = None
    if args.previous:
        previous = json.loads(Path(args.previous).read_text(encoding="utf-8"))

    report = build_report(
        latest_records(runs),
        load_manifest_rows(args.suite_manifest),
        load_baseline_index(args.baseline),
        run_sha,
        previous=previous,
        git_probe=make_git_probe(worktree),
        worktree=worktree,
        max_suspects=args.max_suspects,
        sources=[os.path.basename(p) for p in args.triage],
    )

    markdown = render_markdown(report)
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"lane-health markdown -> {args.out}")
    else:
        print(markdown)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"lane-health json -> {args.json_out}")

    # Report-only: a red tree is the thing being reported, not a failure of the
    # report. Never return non-zero here — see the module docstring.
    return 0


if __name__ == "__main__":
    sys.exit(main())
