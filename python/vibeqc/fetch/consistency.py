"""Cross-provider consistency check (VFETCH-X6, v0.13.x).

``vqfetch consistency --formula MgO`` fetches the same formula from
every OPTIMADE provider that returns it, compares lattice constants
and space groups, and emits a Markdown report flagging disagreements.
Useful for choosing the canonical entry when the structural
fingerprint dedup says "different but related".
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from vibeqc.fetch import FetchCache, FetchCacheMiss, fetch_optimade


def _consistency_report(
    formula: str,
    provider: Optional[str] = None,
    *,
    use_cache: bool = True,
    cache_only: bool = False,
) -> str:
    """Return a Markdown report comparing lattice constants and space
    groups across all providers that return results for ``formula``.
    """
    cache = FetchCache()
    specs = fetch_optimade(
        formula=formula,
        provider=provider,
        cache=cache,
        use_cache=use_cache,
        cache_only=cache_only,
        max_results=50,
        dedup=False,  # we want ALL entries for the consistency check
    )

    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"# Cross-provider consistency report for `{formula}`",
        f"",
        f"Generated: {now}",
        f"Entries: {len(specs)}",
        f"",
        f"| Provider | ID | SG | a (Å) | b (Å) | c (Å) | a (°) | b (°) | g (°) | Atoms |",
        f"|----------|----|----|-------|-------|-------|--------|--------|--------|-------|",
    ]

    for s in specs:
        p = s.provenance
        provider_str = (p.source_db or "?").removeprefix("OPTIMADE/") if p else "?"
        sid = (p.source_id or "?") if p else "?"
        sg = s.space_group or "?"
        lat = s.lattice_ang

        # Compute lattice lengths and angles
        import math

        a = math.sqrt(sum(v * v for v in lat[0]))
        b = math.sqrt(sum(v * v for v in lat[1]))
        c = math.sqrt(sum(v * v for v in lat[2]))

        def _angle(v1, v2):
            dot = sum(a * b for a, b in zip(v1, v2, strict=True))
            return math.degrees(math.acos(max(-1.0, min(1.0, dot / (a * b)))))

        alpha = _angle(lat[1], lat[2])
        beta = _angle(lat[0], lat[2])
        gamma = _angle(lat[0], lat[1])
        n = len(s.atoms)

        lines.append(
            f"| {provider_str} | {sid} | {sg} | {a:.4f} | {b:.4f} | {c:.4f} | "
            f"{alpha:.2f} | {beta:.2f} | {gamma:.2f} | {n} |"
        )

    # ---- Summary: flag disagreements ------------------------------------
    sgs = {s.space_group for s in specs if s.space_group}
    if len(sgs) > 1:
        lines.append("")
        lines.append("## ⚠️ Space-group disagreement")
        lines.append("")
        lines.append(
            f"Found {len(sgs)} distinct space groups: "
            f"{', '.join(sorted(sgs))}. "
            f"Verify which polymorph is correct for your application."
        )

    a_vals = [s.lattice_ang[0][0] for s in specs if s.lattice_ang]
    if a_vals and max(a_vals) - min(a_vals) > 0.1:
        lines.append("")
        lines.append("## ⚠️ Lattice-constant spread")
        lines.append("")
        lines.append(
            f"`a` ranges from {min(a_vals):.4f} to {max(a_vals):.4f} Å "
            f"(Δ = {max(a_vals) - min(a_vals):.4f} Å). "
            f"This exceeds the typical 0.1 Å sanity-check window; "
            f"the entries may represent different phases or "
            f"DFT-vs-experimental spreads."
        )

    return "\n".join(lines) + "\n"


def _cmd_consistency(args: argparse.Namespace) -> int:
    """``vqfetch consistency`` handler."""
    try:
        report = _consistency_report(
            formula=args.formula,
            provider=args.provider,
            use_cache=not args.no_cache,
            cache_only=args.cache_only,
        )
    except FetchCacheMiss as exc:
        print(
            f"vqfetch: cache miss -- refusing to fetch live (--cache-only): {exc}",
            file=__import__("sys").stderr,
        )
        return 2

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.write_text(report, encoding="utf-8")
        print(out_path)
    else:
        print(report)
    return 0
