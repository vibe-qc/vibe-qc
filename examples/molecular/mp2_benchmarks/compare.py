"""Compare vibe-qc vs ORCA results across one of this directory's
benchmarks (S22 or open_shell) and emit a Markdown report.

Walks::

    <bench>/inputs/vibeqc/<system>/(<body>__)?<variant>.result
    <bench>/inputs/orca/<system>/(<body>__)?<variant>.out

Matches by (system, body, variant) — body is "" for the open-shell
complement, "dimer" / "monoA" / "monoB" for S22.

Output (written to ``<bench>/REPORT.md``):

  * Per-system table: ΔE_corr (vibe-qc − ORCA), ΔE_hf, ΔE_total,
    SCF iter both sides, parity verdict.
  * Aggregate summary: max / mean / median ΔE_corr per variant.
  * For S22: counterpoise-style interaction energy comparison
    (E_dimer − E_monoA − E_monoB) on both sides, ΔE_int side-by-
    side, plus the S22B reference if available.

Usage::

    .venv/bin/python examples/molecular/mp2_benchmarks/compare.py s22
    .venv/bin/python examples/molecular/mp2_benchmarks/compare.py open_shell
"""

from __future__ import annotations

import ast
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from parse_orca_mp2 import OrcaMP2ParseError, parse_orca_mp2

HA_TO_KCAL = 627.5094740631


def _parse_vibeqc_result(path: Path) -> dict[str, Any]:
    raw = path.read_text().strip()
    return ast.literal_eval(raw)


_NAME_RE = re.compile(r"^(?:([A-Za-z]+)__)?([A-Za-z0-9]+)$")


def _key(input_path: Path) -> tuple[str, str, str]:
    """Return (system_dir_name, body, variant) from an input filename
    stem like 'dimer__rimp2' or 'rimp2' (open-shell, no body)."""
    stem = input_path.stem
    m = _NAME_RE.match(stem)
    if not m:
        raise ValueError(f"unparseable stem: {stem!r}")
    body, variant = m.group(1) or "", m.group(2)
    system = input_path.parent.name
    return system, body, variant


def collect_results(bench_dir: Path) -> dict[tuple[str, str, str],
                                              dict[str, Any]]:
    """Return {(system, body, variant): {'vibeqc': {...}, 'orca': {...}}}."""
    out: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)

    for p in (bench_dir / "inputs" / "vibeqc").glob("*/*.result"):
        try:
            key = _key(p)
        except ValueError:
            continue
        try:
            out[key]["vibeqc"] = _parse_vibeqc_result(p)
        except Exception as exc:
            out[key]["vibeqc"] = {"_error": f"parse failed: {exc!r}"}

    for p in (bench_dir / "inputs" / "orca").glob("*/*.out"):
        try:
            key = _key(p)
        except ValueError:
            continue
        try:
            out[key]["orca"] = parse_orca_mp2(p)
        except (OrcaMP2ParseError, Exception) as exc:
            out[key]["orca"] = {"_error": f"parse failed: {exc!r}"}

    return out


# ---------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------

VARIANT_ORDER_CLOSED = ("mp2", "rimp2", "scsmp2", "sosmp2", "b2plyp")
VARIANT_ORDER_OPEN   = ("ump2", "riump2", "scsump2", "sosump2")


def _md_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def _fmt_e(x: float | None, width: int = 14, prec: int = 8) -> str:
    if x is None:
        return " " * width
    return f"{x:>{width}.{prec}f}"


def _fmt_d(x: int | None, width: int = 4) -> str:
    if x is None:
        return " " * width
    return f"{x:>{width}d}"


def _per_system_table(results: dict[tuple[str, str, str], dict[str, Any]],
                      systems: list[str], bodies: list[str],
                      variants: list[str]) -> list[str]:
    lines = []
    lines.append("")
    lines.append("| System | Body | Variant | E_corr vq | E_corr ORCA | ΔE_corr | ΔE_corr (kcal/mol) | iter vq | iter ORCA |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for sys_name in systems:
        for body in bodies:
            for variant in variants:
                pair = results.get((sys_name, body, variant))
                if not pair:
                    continue
                vq = pair.get("vibeqc", {})
                orca = pair.get("orca", {})
                vq_corr = vq.get("e_correlation")
                orca_corr = orca.get("e_correlation")
                delta = (vq_corr - orca_corr) if (vq_corr is not None and orca_corr is not None) else None
                row = [
                    sys_name.replace("s22-", ""),
                    body or "—",
                    variant,
                    _fmt_e(vq_corr),
                    _fmt_e(orca_corr),
                    _fmt_e(delta, prec=8) if delta is not None else "PENDING",
                    (f"{delta * HA_TO_KCAL:+8.4f}" if delta is not None else ""),
                    _fmt_d(vq.get("n_scf_iter")),
                    _fmt_d(orca.get("n_scf_iter")),
                ]
                lines.append("| " + " | ".join(row) + " |")
    return lines


def _interaction_energy_table(results: dict[tuple[str, str, str], dict[str, Any]],
                              systems: list[str],
                              variants: list[str],
                              s22b_ref: dict[int, float] | None = None) -> list[str]:
    """E_int = E_dimer − E_monoA − E_monoB. Only defined when all three
    bodies are present and converged for the given (system, variant)."""
    lines = []
    lines.append("")
    lines.append("### S22 interaction energies (no counterpoise correction)")
    lines.append("")
    lines.append("`E_int = E_total[dimer] − E_total[monoA] − E_total[monoB]` (kcal/mol).")
    lines.append("Without ghost-atom counterpoise correction these will be over-bound (BSSE-affected).")
    lines.append("")
    lines.append("| System | Variant | E_int vq (kcal/mol) | E_int ORCA (kcal/mol) | ΔE_int | S22B ref (CCSD(T)/CBS) |")
    lines.append("|---|---|---|---|---|---|")
    for sys_name in systems:
        idx = int(sys_name.split("-")[1])
        for variant in variants:
            triples = {body: results.get((sys_name, body, variant), {})
                       for body in ("dimer", "monoA", "monoB")}
            if any(not t for t in triples.values()):
                continue

            def _e_total(side: str, body: str) -> float | None:
                d = triples[body].get(side, {})
                return d.get("e_total")

            vq_e = [_e_total("vibeqc", b) for b in ("dimer", "monoA", "monoB")]
            orca_e = [_e_total("orca", b) for b in ("dimer", "monoA", "monoB")]
            if all(v is not None for v in vq_e):
                vq_int = (vq_e[0] - vq_e[1] - vq_e[2]) * HA_TO_KCAL
                vq_int_s = f"{vq_int:+9.4f}"
            else:
                vq_int = None
                vq_int_s = ""
            if all(v is not None for v in orca_e):
                orca_int = (orca_e[0] - orca_e[1] - orca_e[2]) * HA_TO_KCAL
                orca_int_s = f"{orca_int:+9.4f}"
            else:
                orca_int = None
                orca_int_s = ""

            delta = (f"{(vq_int - orca_int):+9.4f}"
                     if (vq_int is not None and orca_int is not None) else "")
            ref = (f"{s22b_ref[idx]:+9.4f}" if s22b_ref and idx in s22b_ref else "")
            lines.append(f"| {sys_name.replace('s22-', '')} | {variant} | "
                         f"{vq_int_s} | {orca_int_s} | {delta} | {ref} |")
    return lines


def _aggregate_summary(results: dict[tuple[str, str, str], dict[str, Any]],
                       variants: list[str]) -> list[str]:
    lines = ["", "### Aggregate ΔE_corr (vibe-qc − ORCA) per variant", ""]
    lines.append("| Variant | N pairs | max |Δ| Eh | mean |Δ| Eh | median |Δ| Eh |")
    lines.append("|---|---|---|---|---|")
    for variant in variants:
        deltas = []
        for (_, _, v), pair in results.items():
            if v != variant:
                continue
            vq_corr = pair.get("vibeqc", {}).get("e_correlation")
            orca_corr = pair.get("orca", {}).get("e_correlation")
            if vq_corr is not None and orca_corr is not None:
                deltas.append(abs(vq_corr - orca_corr))
        if not deltas:
            lines.append(f"| {variant} | 0 | — | — | — |")
            continue
        lines.append(f"| {variant} | {len(deltas)} | "
                     f"{max(deltas):.2e} | "
                     f"{statistics.mean(deltas):.2e} | "
                     f"{statistics.median(deltas):.2e} |")
    return lines


def render_report(bench: str) -> str:
    bench_dir = HERE / bench
    if not bench_dir.is_dir():
        raise SystemExit(f"no benchmark directory at {bench_dir}")

    results = collect_results(bench_dir)
    systems = sorted({k[0] for k in results})

    if bench == "s22":
        bodies = ["dimer", "monoA", "monoB"]
        variants = list(VARIANT_ORDER_CLOSED)
    else:
        bodies = [""]
        variants = list(VARIANT_ORDER_OPEN)

    s22b_ref = None
    if bench == "s22":
        try:
            sys.path.insert(0, str(bench_dir))
            import reference_energies as refmod  # noqa: E402
            s22b_ref = refmod.S22B
        except ImportError:
            pass

    lines = [
        f"# {bench.upper()} MP2 comparison: vibe-qc ↔ ORCA",
        "",
        "Generated by ``compare.py``. The two codes were driven with the same",
        "converger profile (plain DIIS, conv_tol_energy=1e-8, max_iter=200,",
        "DIIS subspace 8) — see ``s22/converger.py``.",
        "",
    ]
    lines += _per_system_table(results, systems, bodies, variants)
    if bench == "s22":
        lines += _interaction_energy_table(results, systems,
                                           ["mp2", "rimp2", "scsmp2",
                                            "sosmp2", "b2plyp"],
                                           s22b_ref=s22b_ref)
    lines += _aggregate_summary(results, variants)
    lines.append("")

    text = "\n".join(lines)
    out_path = bench_dir / "REPORT.md"
    out_path.write_text(text)
    print(f"wrote {out_path}")
    return text


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "s22"
    render_report(target)
