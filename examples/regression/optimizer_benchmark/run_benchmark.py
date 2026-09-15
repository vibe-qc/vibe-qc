#!/usr/bin/env python3
"""CLI entry point for the optimizer benchmark suite.

Run a sweep of optimizers across test systems and produce a report.

Usage::

    # Quick test
    python examples/regression/optimizer_benchmark/run_benchmark.py \
        --systems h2o,ch4,c2h4 \
        --optimizers BFGSLineSearch,BFGS,native \
        --output results/quick_test.json --report

    # Full benchmark
    python examples/regression/optimizer_benchmark/run_benchmark.py \
        --systems h2o,ch4,c2h4,benzene,h2o_dimer,glycine,aspirin \
        --extra-systems naphthalene,anthracene,tetracene \
        --optimizers BFGSLineSearch,BFGS,LBFGS,FIRE,MDMin,native \
        --method rks --functional PBE --basis def2-svp \
        --output results/full_benchmark.json --report

    # External comparison (ORCA must be on PATH)
    python examples/regression/optimizer_benchmark/run_benchmark.py \
        --systems h2o,ch4 --external orca \
        --method rks --functional PBE --basis def2-SVP \
        --output results/orca_comparison.json

    # Post-hoc report with plots
    python examples/regression/optimizer_benchmark/run_benchmark.py \
        --load results/full_benchmark.json --report --plot-dir results/plots/
"""

from __future__ import annotations

import argparse
import json as _json
import sys
import tempfile
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _parse_comma_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimizer benchmark — compare geometry optimization methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--systems", type=_parse_comma_list, default=None)
    parser.add_argument("--extra-systems", type=_parse_comma_list, default=None)
    parser.add_argument("--optimizers", type=_parse_comma_list, default=None)
    parser.add_argument("--method", default="rhf", choices=["rhf", "uhf", "rks", "uks"])
    parser.add_argument("--functional", default=None)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--list-systems", action="store_true")
    parser.add_argument("--list-opts", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--external", choices=["orca", "crystal"], default=None)
    parser.add_argument("--external-exe", default=None)
    parser.add_argument("--external-timeout", type=float, default=3600.0)
    parser.add_argument("--external-dir", type=Path, default=None)

    args = parser.parse_args()

    from examples.regression.optimizer_benchmark.benchmark_driver import (
        ASE_OPTIMIZERS,
        BenchmarkResult,
        _register_ase_optimizers,
        run_benchmark,
    )
    from examples.regression.optimizer_benchmark.external import (
        _write_xyz,
        run_crystal_optimization,
        run_orca_optimization,
    )
    from examples.regression.optimizer_benchmark.extra_systems import (
        _register_extra,
        get_extra_system,
        list_extra_systems,
    )
    from examples.regression.optimizer_benchmark.report import (
        optimizer_summary_table,
        per_system_table,
        print_report,
    )
    from examples.regression.optimizer_benchmark.test_systems import (
        get_system,
        list_systems,
    )

    # --list-systems
    if args.list_systems:
        print("Core test systems:")
        for name in list_systems():
            s = get_system(name)
            print(
                f"  {name:20s}  {s.n_atoms:3d} atoms  [{s.category}]  {s.description}"
            )
        print("\nExtra systems (scaling family):")
        for name in list_extra_systems():
            s = get_extra_system(name)
            print(
                f"  {name:20s}  {s.n_atoms:3d} atoms  [{s.category}]  {s.description}"
            )
        return

    # --list-opts
    if args.list_opts:
        _register_ase_optimizers()
        print("Available optimizers:")
        for name in sorted(ASE_OPTIMIZERS):
            info = ASE_OPTIMIZERS[name]
            print(f"  {name:25s}  [{info['family']:15s}]  {info['description']}")
        print(f"  {'native':25s}  [quasi_newton    ]  scipy L-BFGS-B (vibe-qc native)")
        return

    # ---- Load previous results ----
    result = None
    if args.load:
        print(f"Loading results from {args.load} ...")
        result = BenchmarkResult.load(args.load)

    # ---- External comparison ----
    elif args.external:
        all_systems = (args.systems or []) + (args.extra_systems or [])
        if not all_systems:
            parser.error("--external requires --systems or --extra-systems.")
        exe = args.external_exe or args.external
        print(f"External comparison: {args.external} ({exe})")

        tmpdir = Path(tempfile.mkdtemp(prefix="vq_ext_"))
        ext_results = []
        for sys_name in all_systems:
            try:
                s = get_system(sys_name)
            except ValueError:
                try:
                    s = get_extra_system(sys_name)
                except ValueError:
                    print(f"  Unknown system: {sys_name} - skipping")
                    continue
            xyz_path = tmpdir / f"{sys_name}.xyz"
            _write_xyz(xyz_path, s, comment=sys_name)
            print(f"  {sys_name:20s}  ...", end=" ", flush=True)
            if args.external == "orca":
                ext = run_orca_optimization(
                    xyz_path,
                    method=args.method.upper(),
                    functional=args.functional,
                    basis=args.basis,
                    fmax_eva=args.fmax,
                    max_steps=args.max_steps,
                    orca_exe=exe,
                    timeout_s=args.external_timeout,
                    work_dir=args.external_dir,
                )
            else:
                ext = run_crystal_optimization(
                    xyz_path,
                    method=args.method.upper(),
                    functional=args.functional,
                    basis=args.basis,
                    fmax_eva=args.fmax,
                    max_steps=args.max_steps,
                    crystal_exe=exe,
                    timeout_s=args.external_timeout,
                    work_dir=args.external_dir,
                )
            status = "OK" if ext.converged else "FAIL"
            detail = f"{ext.n_steps} steps" if ext.converged else (ext.error or "nc")
            print(f"{status}  [{detail}]")
            ext_results.append(ext)

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w") as f:
                _json.dump([r.to_dict() for r in ext_results], f, indent=2, default=str)
            print(f"\nExternal results saved to {out_p}")
        return

    # ---- vibe-qc benchmark ----
    elif args.systems or args.extra_systems:
        all_systems = (args.systems or []) + (args.extra_systems or [])
        if not args.optimizers:
            parser.error("--optimizers is required when running benchmarks.")
        result = run_benchmark(
            systems=all_systems,
            optimizers=args.optimizers,
            method=args.method,
            functional=args.functional,
            basis=args.basis,
            fmax=args.fmax,
            max_steps=args.max_steps,
            verbose=not args.quiet,
            system_getter=get_extra_system,
        )
        if args.output:
            result.save(args.output)
            print(f"\nResults saved to {args.output}")
    else:
        parser.error(
            "One of --systems+--optimizers, --load, or --external is required."
        )

    if result is None:
        return

    # ---- Report ----
    if args.report:
        if args.markdown:
            print(optimizer_summary_table(result, fmt="markdown"))
            print()
            print(per_system_table(result, fmt="markdown"))
        else:
            print_report(result, output_dir=args.plot_dir)
    elif args.plot_dir:
        print_report(result, output_dir=args.plot_dir)


if __name__ == "__main__":
    main()
