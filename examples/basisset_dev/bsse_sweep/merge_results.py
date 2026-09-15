"""Merge ATOMBSSE sweeps into one table, keeping every sweep attributable.

    .venv/bin/python examples/basisset_dev/bsse_sweep/merge_results.py \
        sweep1=results-1.json sweep2=results-2.json

Later sweeps win per ``system|site|nstar|rmax`` key, and every row records
which sweep supplied the point it rests on. That attribution is the whole
point: the sweeps in this study did **not** run the same protocol -- the
first predates vibe-basis 0.8.0's ``TOLINTEG 9 9 9 18 54`` / no-``HUGEGRID``
switch -- so a merged table that forgot which run a number came from would
silently mix two protocols under one verdict.

A site is reported as converged only when at least ``MIN_POINTS`` ladder
points succeeded, and the sweeps that produced them are named. One
surviving point makes the densest point the chosen one and the difference
trivially zero, which is indistinguishable from a real plateau.

Reads ``results.json`` files as written by ``run_sweep.py``; writes the
markdown table to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HA_KJ = 2625.4996394798254
TOL_KJ = 0.5
MIN_POINTS = 3


def load(specs: list[str]) -> tuple[dict, dict, list, dict]:
    """``["name=path", ...]`` -> merged results, key -> sweep, order, raw."""
    merged: dict[str, dict] = {}
    origin: dict[str, str] = {}
    order: list[str] = []
    raw: dict[str, dict] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(name).stem, name
        loaded = json.loads(Path(path).read_text())
        raw[name] = loaded
        order.append(name)
        for key, rec in loaded.items():
            merged[key] = rec
            origin[key] = name
    return merged, origin, order, raw


def summarise(results: dict, origin: dict, order: list,
              blend_rungs: bool = False) -> tuple[str, dict]:
    by_site: dict[tuple, list] = {}
    for key, r in results.items():
        by_site.setdefault((r["system"], r["site"], r["Z"]), []).append((key, r))

    # One site, one protocol. Rung-level "later wins" looks right and is
    # not: when a re-run has finished rungs 1-4 and its rung 5 is still in
    # flight, the older sweep's rung 5 survives and the plateau is asserted
    # across two protocols -- with the *densest* point, the one every delta
    # is measured against, coming from the protocol being replaced. So a
    # site's verdict rests only on the latest sweep that produced any point
    # for it. That discards good older points when a re-run is partial;
    # that is the conservative direction, and the `from` column names which
    # sweep the surviving points came from. `--blend-rungs` opts back in.
    if not blend_rungs:
        rank = {name: i for i, name in enumerate(order)}
        for site, rows in by_site.items():
            latest = max(rank[origin[k]] for k, _ in rows)
            by_site[site] = [(k, r) for k, r in rows
                             if rank[origin[k]] == latest]

    lines = ["| system | site | Z | verdict | at | dE vs densest | densest "
             "| pts | from |",
             "|---|---|---|---|---|---|---|---|---|"]
    verdict: dict[str, dict] = {}
    for (system, site, Z), rows in sorted(by_site.items()):
        good = sorted(((k, r) for k, r in rows
                       if r["status"] == "ok" and r["energy"]),
                      key=lambda kr: (kr[1]["nstar"], kr[1]["rmax"]))
        sweeps = sorted({origin[k] for k, _ in rows})
        if not good:
            why = sorted({r["status"][:24] for _, r in rows})
            lines.append(f"| {system} | {site} | {Z} | FAILED | - | - | - | 0 "
                         f"| {'+'.join(sweeps)} |")
            verdict[f"{system}|{site}"] = dict(
                Z=Z, verdict="failed", n_points=0, statuses=why)
            continue
        ref_key, ref = good[-1]
        chosen_key, chosen = next(
            ((k, r) for k, r in good
             if abs(r["energy"] - ref["energy"]) * HA_KJ <= TOL_KJ),
            (ref_key, ref))
        d = (chosen["energy"] - ref["energy"]) * HA_KJ
        v = "converged" if len(good) >= MIN_POINTS else "INSUFFICIENT"
        from_ = "+".join(sorted({origin[k] for k, _ in good}))
        lines.append(
            f"| {system} | {site} | {Z} | {v} "
            f"| {chosen['nstar']}/{chosen['rmax']} | {d:+.3f} kJ/mol "
            f"| {ref['nstar']}/{ref['rmax']} | {len(good)} | {from_} |")
        verdict[f"{system}|{site}"] = dict(
            Z=Z, verdict=v.lower(), nstar=chosen["nstar"], rmax=chosen["rmax"],
            delta_kj=d, n_points=len(good), energy=chosen["energy"],
            from_sweep=origin[chosen_key])
    return "\n".join(lines), verdict


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--blend-rungs"]
    if not args:
        raise SystemExit(__doc__)
    results, origin, order, raw = load(args)
    table, verdict = summarise(results, origin, order,
                               blend_rungs="--blend-rungs" in sys.argv)

    # Tally each sweep as it actually ran, not as it survives the merge:
    # a sweep's own failure profile is what says whether a protocol change
    # helped, and that is lost if overridden keys are dropped from it.
    print("## Run outcomes, per sweep as run\n")
    for name in order:
        tally: dict[str, int] = {}
        for r in raw[name].values():
            tally[r["status"][:24]] = tally.get(r["status"][:24], 0) + 1
        total = sum(tally.values())
        print(f"* `{name}` ({total} runs): "
              + ", ".join(f"`{s}` {n}" for s, n in sorted(tally.items())))

    counts: dict[str, int] = {}
    for v in verdict.values():
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    print("\n## Verdicts\n")
    for k, n in sorted(counts.items()):
        print(f"* {k} -- {n}")
    print(f"* total sites -- {len(verdict)}")
    print("\n## Table\n")
    print(table)


if __name__ == "__main__":
    main()
