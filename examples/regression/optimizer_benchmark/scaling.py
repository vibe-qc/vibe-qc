"""Scaling analysis and advanced reporting for the optimizer benchmark.

Adds size-scaling plots, per-family (PES-type) breakdowns, and
comparison with external program results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .benchmark_driver import BenchmarkResult


def scaling_table(
    result: BenchmarkResult,
    scaling_systems: Optional[List[str]] = None,
    fmt: str = "plain",
) -> str:
    """Produce a scaling table — steps + wall time vs system size."""
    from .extra_systems import get_extra_system
    from .test_systems import get_system

    sys_sizes: dict[str, int] = {}
    for r in result.runs:
        if r.system_name in sys_sizes:
            continue
        try:
            s = get_system(r.system_name)
            sys_sizes[r.system_name] = s.n_atoms
        except ValueError:
            try:
                s = get_extra_system(r.system_name)
                sys_sizes[r.system_name] = s.n_atoms
            except ValueError:
                sys_sizes[r.system_name] = 0

    if scaling_systems is None:
        scaling_systems = sorted(sys_sizes, key=lambda n: sys_sizes.get(n, 0))
    else:
        scaling_systems = sorted(scaling_systems, key=lambda n: sys_sizes.get(n, 0))

    opt_names = sorted(set(r.optimizer_name for r in result.runs))
    header = ["System", "n_atoms"] + opt_names
    rows = []
    for sys_name in scaling_systems:
        if sys_name not in sys_sizes:
            continue
        sys_runs = {
            r.optimizer_name: r for r in result.runs if r.system_name == sys_name
        }
        row = [sys_name, str(sys_sizes[sys_name])]
        for opt_name in opt_names:
            run = sys_runs.get(opt_name)
            if run is None:
                row.append("—")
            elif run.converged:
                row.append(f"{run.n_steps} / {run.wall_time_s:.1f}s")
            elif run.error:
                row.append("ERR")
            else:
                row.append(f"{run.n_steps} (nc)")
        rows.append(row)

    if fmt == "markdown":
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    col_widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    lines = [sep]
    lines.append(
        "| " + " | ".join(h.ljust(w) for h, w in zip(header, col_widths)) + " |"
    )
    lines.append(sep)
    for row in rows:
        lines.append(
            "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        )
    lines.append(sep)
    return "\n".join(lines)


def plot_scaling(
    result: BenchmarkResult,
    output_path: Optional[str | Path] = None,
) -> "Any":
    """Plot steps and wall time vs system size across optimizers.

    Produces a two-panel figure: left = steps vs n_atoms, right = wall
    time vs n_atoms, with one curve per optimizer.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping scaling plot.", file=sys.stderr)
        return None

    from .extra_systems import get_extra_system
    from .test_systems import get_system

    opt_data: dict[str, list[tuple[int, int, float]]] = {}
    for r in result.runs:
        if not r.converged:
            continue
        try:
            s = get_system(r.system_name)
            n = s.n_atoms
        except ValueError:
            try:
                s = get_extra_system(r.system_name)
                n = s.n_atoms
            except ValueError:
                continue
        opt_data.setdefault(r.optimizer_name, []).append((n, r.n_steps, r.wall_time_s))

    if not opt_data:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for i, (opt_name, data) in enumerate(sorted(opt_data.items())):
        data.sort()
        ns = [d[0] for d in data]
        steps = [d[1] for d in data]
        times = [d[2] for d in data]
        color = colors[i % len(colors)]
        ax1.plot(ns, steps, "o-", label=opt_name, color=color, markersize=6)
        ax2.plot(ns, times, "s-", label=opt_name, color=color, markersize=6)

    ax1.set_xlabel("Number of atoms")
    ax1.set_ylabel("Steps to convergence")
    ax1.set_title("Steps vs system size")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Number of atoms")
    ax2.set_ylabel("Wall time (s)")
    ax2.set_title("Wall time vs system size")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def per_family_table(result: BenchmarkResult, fmt: str = "plain") -> str:
    """Mean steps per optimizer, broken down by PES category."""
    from .extra_systems import get_extra_system
    from .test_systems import get_system

    sys_categories: dict[str, str] = {}
    for r in result.runs:
        if r.system_name in sys_categories:
            continue
        try:
            s = get_system(r.system_name)
            sys_categories[r.system_name] = s.category
        except ValueError:
            try:
                s = get_extra_system(r.system_name)
                sys_categories[r.system_name] = s.category
            except ValueError:
                sys_categories[r.system_name] = "?"

    agg: dict[tuple[str, str], list[int]] = {}
    for r in result.runs:
        if not r.converged:
            continue
        cat = sys_categories.get(r.system_name, "?")
        key = (cat, r.optimizer_name)
        agg.setdefault(key, []).append(r.n_steps)

    categories = sorted(set(c for c, _ in agg))
    opt_names = sorted(set(o for _, o in agg))

    header = ["Category"] + opt_names
    rows = []
    for cat in categories:
        row = [cat]
        for opt_name in opt_names:
            steps = agg.get((cat, opt_name), [])
            if steps:
                mean = sum(steps) / len(steps)
                row.append(f"{mean:.1f}")
            else:
                row.append("—")
        rows.append(row)

    if fmt == "markdown":
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    col_widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    lines = [sep]
    lines.append(
        "| " + " | ".join(h.ljust(w) for h, w in zip(header, col_widths)) + " |"
    )
    lines.append(sep)
    for row in rows:
        lines.append(
            "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        )
    lines.append(sep)
    return "\n".join(lines)


def external_comparison_table(
    vibe_qc_results: BenchmarkResult,
    external_results: List[dict],
    fmt: str = "plain",
) -> str:
    """Compare vibe-qc optimizers against external program results.

    Parameters
    ----------
    vibe_qc_results : BenchmarkResult
        vibe-qc benchmark results.
    external_results : list of dict
        Results from :func:`external.run_orca_optimization` or
        :func:`external.run_crystal_optimization`.
    fmt : str
        "plain" or "markdown".

    Returns
    -------
    str
    """
    # Build a lookup: system_name -> {optimizer_name: n_steps}
    vq_data: dict[str, dict[str, int]] = {}
    for r in vibe_qc_results.runs:
        if r.converged:
            vq_data.setdefault(r.system_name, {})[r.optimizer_name] = r.n_steps

    # External data: system_name -> {program: n_steps}
    ext_data: dict[str, dict[str, tuple[int, float]]] = {}
    for r in external_results:
        name = r.get("system_name", "?")
        prog = r.get("program", "?")
        steps = r.get("n_steps", 0)
        time_s = r.get("wall_time_s", 0.0)
        converged = r.get("converged", False)
        if converged:
            ext_data.setdefault(name, {})[prog] = (steps, time_s)

    all_systems = sorted(set(vq_data) | set(ext_data))
    vq_opt_names = sorted(set(o for d in vq_data.values() for o in d))
    ext_progs = sorted(set(p for d in ext_data.values() for p in d))

    header = ["System"] + vq_opt_names + ext_progs
    rows = []
    for sys_name in all_systems:
        row = [sys_name]
        for opt in vq_opt_names:
            steps = vq_data.get(sys_name, {}).get(opt)
            row.append(str(steps) if steps is not None else "—")
        for prog in ext_progs:
            entry = ext_data.get(sys_name, {}).get(prog)
            if entry:
                row.append(f"{entry[0]} steps / {entry[1]:.1f}s")
            else:
                row.append("—")
        rows.append(row)

    if fmt == "markdown":
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    col_widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    lines = [sep]
    lines.append(
        "| " + " | ".join(h.ljust(w) for h, w in zip(header, col_widths)) + " |"
    )
    lines.append(sep)
    for row in rows:
        lines.append(
            "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        )
    lines.append(sep)
    return "\n".join(lines)
