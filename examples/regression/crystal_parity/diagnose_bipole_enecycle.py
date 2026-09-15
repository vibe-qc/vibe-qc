#!/usr/bin/env python3
"""Compare CRYSTAL ``ENECYCLE`` components to vibe-qc BIPOLE components.

This is an out-of-process diagnostic. CRYSTAL is executed as a separate
program via ``examples.regression.core.runner_crystal.run_crystal_local``;
no CRYSTAL code is imported or linked into vibe-qc.

Cycle mapping:

* CRYSTAL ``CYC 0`` is the energy at the initial SAD density after the
  first Fock build.
* vibe-qc ``iter 1`` is the same structural point.

So the table compares ``CRYSTAL CYC n`` against ``vibe iter n + 1``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vibeqc as vq
from vibeqc import InitialGuess, monkhorst_pack
from vibeqc._vibeqc_core import PeriodicRHFOptions
from vibeqc.pbc_bipole import run_pbc_bipole_rhf

from examples.regression.core.runner_crystal import (
    add_crystal_enecycle_keyword,
    parse_crystal_enecycle,
    run_crystal_local,
)
from examples.regression.crystal_parity.crystal_demos import builders


_DEMO_DIR = Path(__file__).resolve().parent / "crystal_demos"


_SYSTEMS: dict[
    str,
    tuple[Callable[[], tuple[vq.PeriodicSystem, vq.BasisSet]], Path],
] = {
    "mgo": (builders.build_mgo_sto3g, _DEMO_DIR / "mgo_sto3g.d12"),
    "diamond": (builders.build_diamond_sto3g, _DEMO_DIR / "diamond_sto3g.d12"),
    "si": (builders.build_sibulk_sto3g, _DEMO_DIR / "sibulk_sto3g.d12"),
}


def _replace_keyword_value(deck: str, keyword: str, value_lines: list[str]) -> str:
    """Replace a simple CRYSTAL keyword block or insert before final END."""
    lines = deck.splitlines()
    key = keyword.upper()
    for idx, line in enumerate(lines):
        if line.strip().upper() == key:
            end = idx + 1 + len(value_lines)
            lines[idx + 1:end] = value_lines
            return "\n".join(lines) + "\n"

    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip().upper() == "END":
            lines[idx:idx] = [keyword, *value_lines]
            return "\n".join(lines) + "\n"

    return "\n".join([*lines, keyword, *value_lines, "END"]) + "\n"


def _prepare_crystal_deck(deck: str, *, shrink: int, maxcycle: int) -> str:
    deck = _replace_keyword_value(deck, "SHRINK", [f"{shrink} {shrink}"])
    deck = _replace_keyword_value(deck, "MAXCYCLE", [str(maxcycle)])
    return add_crystal_enecycle_keyword(deck)


def _write_prepared_deck(
    source: Path, *, workdir: Path, shrink: int, maxcycle: int,
) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    deck = _prepare_crystal_deck(
        source.read_text(), shrink=shrink, maxcycle=maxcycle,
    )
    out = workdir / f"{source.stem}-shrink{shrink}-enecycle.d12"
    out.write_text(deck)
    return out


def _fmt(value: float | None) -> str:
    return "        n/a" if value is None else f"{value:+14.8f}"


def _delta(vibe: float | None, crystal: float | None) -> str:
    if vibe is None or crystal is None:
        return "        n/a"
    return f"{(vibe - crystal) * 1000.0:+12.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=sorted(_SYSTEMS))
    parser.add_argument("--shrink", type=int, default=1)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--maxcycle", type=int, default=1)
    parser.add_argument("--ewald-precision", type=float, default=1e-6)
    parser.add_argument("--crystal-exe", default=None)
    parser.add_argument(
        "--fock-mode",
        choices=("ewald-j-split", "direct"),
        default="ewald-j-split",
        help=(
            "vibe-qc two-electron build to compare. 'direct' is the "
            "unfinished CRYSTAL BIPOLE scaffold without EXT cancellation; "
            "'ewald-j-split' is the current gauge-consistent approximation."
        ),
    )
    parser.add_argument(
        "--cycles",
        default=None,
        help="comma-separated CRYSTAL cycles to print, e.g. '0' or '0,1'",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/private/tmp/vibeqc-bipole-enecycle"),
    )
    parser.add_argument(
        "--use-symmetry",
        action="store_true",
        help="use symmetry-reduced kmesh in vibe-qc (default: full mesh)",
    )
    parser.add_argument(
        "--k-shift",
        choices=("gamma", "mp"),
        default="gamma",
        help=(
            "vibe-qc k-grid convention. 'gamma' matches CRYSTAL SHRINK's "
            "integer coordinates in units of IS; 'mp' applies the classical "
            "half-step shift on all three axes."
        ),
    )
    args = parser.parse_args(argv)

    build_fn, d12_path = _SYSTEMS[args.system]
    crystal_input = _write_prepared_deck(
        d12_path,
        workdir=args.workdir,
        shrink=args.shrink,
        maxcycle=args.maxcycle,
    )
    crystal_out = crystal_input.with_suffix(".out")
    run_crystal_local(
        crystal_input,
        output_path=crystal_out,
        executable=args.crystal_exe,
        enecycle=False,
    )
    crystal_records = parse_crystal_enecycle(crystal_out.read_text())
    if not crystal_records:
        raise RuntimeError(f"no CRYSTAL ENECYCLE records parsed from {crystal_out}")
    wanted = None
    if args.cycles:
        wanted = {int(x.strip()) for x in args.cycles.split(",") if x.strip()}
        crystal_records = [r for r in crystal_records if r.cycle in wanted]
        if not crystal_records:
            raise RuntimeError(
                f"no requested CRYSTAL cycles {sorted(wanted)} parsed "
                f"from {crystal_out}"
            )

    system, basis = build_fn()
    shift = [0, 0, 0] if args.k_shift == "gamma" else [1, 1, 1]
    kmesh = monkhorst_pack(
        system,
        [args.shrink, args.shrink, args.shrink],
        is_shift=shift,
        use_symmetry=bool(args.use_symmetry),
    )
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = float(args.cutoff)
    opts.lattice_opts.nuclear_cutoff_bohr = float(args.cutoff)
    opts.initial_guess = InitialGuess.SAD
    opts.max_iter = (
        max(wanted) + 1 if wanted is not None else int(args.maxcycle) + 1
    )
    opts.use_diis = False
    opts.damping = 0.0

    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=(args.fock_mode == "ewald-j-split"),
        ewald_precision=float(args.ewald_precision),
        progress=False,
    )

    vibe_records = {comp.iter - 1: comp for comp in result.energy_components}

    print(f"CRYSTAL output: {crystal_out}")
    print(
        f"vibe-qc: system={args.system} shrink={args.shrink} "
        f"cutoff={args.cutoff} fock_mode={args.fock_mode} "
        f"k_shift={args.k_shift}"
    )
    print()
    print(
        "cycle component               CRYSTAL           vibe-qc"
        "      delta(mHa)"
    )
    print("-" * 76)
    component_rows = [
        ("E_total", "e_total"),
        ("E_kin", "e_kinetic"),
        ("E_ne", "e_nuclear_attraction"),
        ("E_2e", "e_two_electron"),
        ("E_nn", "e_nuclear_repulsion"),
        ("BIELET zone", "e_bielet_zone_ee"),
        ("EXT EL-POLE", "e_ext_el_pole"),
        ("EXT SPHERO", "e_ext_el_spheropole"),
        ("J_SR", "e_j_short_range"),
        ("J_LR", "e_j_long_range"),
        ("K exch", "e_exchange"),
    ]
    for crystal in crystal_records:
        vibe = vibe_records.get(crystal.cycle)
        for label, attr in component_rows:
            c_val = getattr(crystal, attr, None)
            v_val = getattr(vibe, attr) if vibe is not None else None
            print(
                f"{crystal.cycle:5d} {label:<18s}"
                f"{_fmt(c_val)} {_fmt(v_val)} {_delta(v_val, c_val)}"
            )
        ext_sum = (
            None
            if (
                crystal.e_bielet_zone_ee is None
                or crystal.e_ext_el_pole is None
                or crystal.e_ext_el_spheropole is None
            )
            else (
                crystal.e_bielet_zone_ee
                + crystal.e_ext_el_pole
                + crystal.e_ext_el_spheropole
            )
        )
        print(
            f"{crystal.cycle:5d} {'BIELET+EXT':<18s}"
            f"{_fmt(ext_sum)} {'':>14s} {'':>12s}"
        )
        print("-" * 76)

    compared = [
        (crystal, vibe_records[crystal.cycle])
        for crystal in crystal_records
        if crystal.cycle in vibe_records
    ]
    if compared:
        last_c, last_v = compared[-1]
        if last_c.e_total is not None:
            print(
                f"last compared cycle {last_c.cycle} total delta = "
                f"{(last_v.e_total - last_c.e_total) * 1000.0:+.3f} mHa"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
