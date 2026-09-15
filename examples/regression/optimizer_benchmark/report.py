"""Results analysis and reporting for the optimizer benchmark.

Produces comparison tables, convergence trajectory plots, and
summary statistics.  Designed to work post-hoc — load a saved
``BenchmarkResult`` JSON and generate reports without re-running
calculations.

Output formats:
  - Terminal tables (rich or plain text)
  - Markdown tables (for paper drafts)
  - Convergence plots (matplotlib)
  - Per-family summary stats
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .benchmark_driver import BenchmarkResult, OptimizerRun

# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def _fmt(val: Any, fmt_spec: str = ".4f") -> str:
    """Format a numeric value, handling NaN/inf."""
    if val is None:
        return "—"
    try:
        f = float(val)
        if f != f:
            return "—"
        if abs(f) == float("inf"):
            return "∞"
        return f"{f:{fmt_spec}}"
    except (TypeError, ValueError):
        return str(val)


# ---------------------------------------------------------------------------
# Summary table: one row per optimizer, aggregated across systems
# ---------------------------------------------------------------------------


def optimizer_summary_table(
    result: BenchmarkResult,
    fmt: str = "plain",
) -> str:
    """Produce a summary table comparing optimizers across all systems.

    Parameters
    ----------
    result : BenchmarkResult
    fmt : str
        ``"plain"``, ``"markdown"``, or ``"grid"``.

    Returns
    -------
    str
        Formatted table.
    """
    # Aggregate per optimizer
    agg: Dict[str, List[OptimizerRun]] = defaultdict(list)
    for run in result.runs:
        agg[run.optimizer_name].append(run)

    header = [
        "Optimizer",
        "Family",
        "Hessian?",
        "N converged",
        "Mean steps",
        "Mean wall (s)",
        "Mean ΔE (mHa)",
        "Best for",
    ]

    rows = []
    for opt_name, runs in sorted(agg.items()):
        info = runs[0].optimizer_info if runs else {}
        family = info.get("family", "?")
        uses_hess = "✓" if info.get("uses_hessian_approx") else "—"
        n_conv = sum(1 for r in runs if r.converged)
        n_total = len(runs)
        steps_conv = [r.n_steps for r in runs if r.converged]
        times_conv = [r.wall_time_s for r in runs if r.converged]
        de_conv = [
            (r.energy_initial_ha - r.energy_final_ha) * 1000.0
            for r in runs
            if r.converged
        ]

        mean_steps = sum(steps_conv) / len(steps_conv) if steps_conv else float("nan")
        mean_time = sum(times_conv) / len(times_conv) if times_conv else float("nan")
        mean_de = sum(de_conv) / len(de_conv) if de_conv else float("nan")

        # Best-for heuristic
        if family == "quasi_newton":
            best_for = "Most systems (default)"
        elif family == "inertial":
            best_for = "Far-from-minimum, rough PES"
        elif family == "cg":
            best_for = "Large systems, low memory"
        elif family == "bayesian":
            best_for = "Expensive evaluations"
        else:
            best_for = "—"

        rows.append(
            [
                opt_name,
                family,
                uses_hess,
                f"{n_conv}/{n_total}",
                _fmt(mean_steps, ".1f"),
                _fmt(mean_time, ".1f"),
                _fmt(mean_de, ".2f"),
                best_for,
            ]
        )

    if fmt == "markdown":
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    # Plain text table
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


# ---------------------------------------------------------------------------
# Per-system detail table
# ---------------------------------------------------------------------------


def per_system_table(
    result: BenchmarkResult,
    fmt: str = "plain",
) -> str:
    """Produce a table showing every optimizer×system pair.

    Parameters
    ----------
    result : BenchmarkResult
    fmt : str
        ``"plain"`` or ``"markdown"``.

    Returns
    -------
    str
    """
    # Collect system names and optimizer names in order.
    sys_names = sorted(set(r.system_name for r in result.runs))
    opt_names = sorted(set(r.optimizer_name for r in result.runs))

    # Sort systems by category then name.
    sys_categories = {}
    for r in result.runs:
        if r.system_name not in sys_categories:
            from .test_systems import get_system

            try:
                s = get_system(r.system_name)
                sys_categories[r.system_name] = s.category
            except Exception:
                sys_categories[r.system_name] = "?"
    sys_names.sort(key=lambda n: (sys_categories.get(n, "z"), n))

    header = ["System", "n_atoms", "Category"] + opt_names
    rows = []

    for sys_name in sys_names:
        sys_runs = {
            r.optimizer_name: r for r in result.runs if r.system_name == sys_name
        }
        n_atoms = ""
        category = sys_categories.get(sys_name, "?")
        # Get n_atoms from first run
        for r in sys_runs.values():
            from .test_systems import get_system

            try:
                s = get_system(sys_name)
                n_atoms = str(s.n_atoms)
            except Exception:
                pass
            break

        row = [sys_name, n_atoms, category]
        for opt_name in opt_names:
            run = sys_runs.get(opt_name)
            if run is None:
                row.append("—")
            elif run.converged:
                row.append(f"{run.n_steps} steps")
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


# ---------------------------------------------------------------------------
# Convergence trajectory data for plotting
# ---------------------------------------------------------------------------


def convergence_data(
    result: BenchmarkResult,
    system_name: str,
) -> Dict[str, List[Tuple[float, float]]]:
    """Extract (step, energy) pairs for all optimizers on one system.

    Returns
    -------
    dict
        ``{optimizer_name: [(step, energy_ha), ...]}``
    """
    data: Dict[str, List[Tuple[float, float]]] = {}
    for run in result.runs:
        if run.system_name != system_name:
            continue
        data[run.optimizer_name] = [(s.step, s.energy_ha) for s in run.steps]
    return data


# ---------------------------------------------------------------------------
# Convergence plot (matplotlib)
# ---------------------------------------------------------------------------


def plot_convergence(
    result: BenchmarkResult,
    system_name: str,
    output_path: Optional[str | Path] = None,
    relative_energy: bool = True,
) -> "Any":
    """Plot convergence trajectories for all optimizers on one system.

    Parameters
    ----------
    result : BenchmarkResult
    system_name : str
        Which system to plot.
    output_path : Path or None
        If provided, save the figure to this file.
    relative_energy : bool
        If True, plot ΔE from the best energy (relative to the lowest
        energy found by any optimizer). If False, plot absolute.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.", file=sys.stderr)
        return None

    data = convergence_data(result, system_name)
    if not data:
        print(f"No data for system {system_name!r}", file=sys.stderr)
        return None

    # Find best energy for relative scale
    best_energy = float("inf")
    for pairs in data.values():
        for _, e in pairs:
            if e == e and e < best_energy:
                best_energy = e

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    for i, (opt_name, pairs) in enumerate(sorted(data.items())):
        if not pairs:
            continue
        steps = [p[0] for p in pairs]
        energies = [p[1] for p in pairs]
        if relative_energy and best_energy != float("inf"):
            energies = [(e - best_energy) * 1000.0 for e in energies]  # mHa
            ylabel = "ΔE (mHa)"
        else:
            ylabel = "Energy (Ha)"

        color = colors[i % len(colors)]
        ax.plot(
            steps,
            energies,
            marker=".",
            label=opt_name,
            color=color,
            markersize=4,
            linewidth=1.5,
        )

    ax.set_xlabel("Optimization step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Convergence — {system_name}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def print_report(
    result: BenchmarkResult,
    output_dir: Optional[str | Path] = None,
) -> None:
    """Print a full benchmark report to stdout, optionally saving plots.

    Parameters
    ----------
    result : BenchmarkResult
    output_dir : Path or None
        If provided, save convergence plots and a JSON copy here.
    """
    print("=" * 72)
    print("  Optimizer Benchmark Report")
    print("=" * 72)
    cfg = result.config
    print(
        f"\n  Method: {cfg['method']}"
        + (f" / {cfg['functional']}" if cfg.get("functional") else "")
        + f"  Basis: {cfg.get('basis', 'N/A')}"
    )
    print(f"  fmax = {cfg['fmax_eva']} eV/Å  max_steps = {cfg['max_steps']}")
    print(f"  Timestamp: {result.timestamp}")
    print(f"  Total runs: {len(result.runs)}")
    print()

    print("  Systems:")
    sys_names = sorted(set(r.system_name for r in result.runs))
    from .test_systems import get_system

    for name in sys_names:
        try:
            s = get_system(name)
            print(f"    {name:20s}  {s.description}")
        except Exception:
            print(f"    {name:20s}")
    print()

    # Optimizer summary
    print("  Optimizer Summary")
    print("  " + "-" * 40)
    print(optimizer_summary_table(result))
    print()

    # Per-system detail
    print("  Per-System Detail (steps to convergence)")
    print("  " + "-" * 40)
    print(per_system_table(result))
    print()

    # Convergence failures
    failures = [r for r in result.runs if not r.converged]
    if failures:
        print(f"  Convergence failures ({len(failures)}/{len(result.runs)}):")
        for r in failures:
            reason = r.error or f"g={r.grad_final_ha_per_bohr:.2e}"
            print(f"    {r.system_name:20s}  {r.optimizer_name:20s}  {reason}")
        print()

    # Save plots if requested
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for sys_name in sys_names:
            plot_convergence(
                result,
                sys_name,
                output_path=out / f"convergence_{sys_name}.png",
            )
        print(f"  Convergence plots saved to {out}/")
        print()
