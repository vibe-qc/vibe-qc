#!/usr/bin/env python3
"""vibe-qc basis-set toolkit -- CLI.

Convert, validate, and inspect basis sets from the command line.

Usage::

    # Validate a QVF-Basis file
    python -m vibeqc.basis_toolkit.cli validate sto-3g.qvf.json

    # Convert BSE JSON to QVF-Basis (text mode)
    python -m vibeqc.basis_toolkit.cli convert sto-3g.bse.json -o sto-3g.qvf.json

    # Convert BSE JSON to legacy format
    python -m vibeqc.basis_toolkit.cli convert sto-3g.bse.json -o sto-3g.g94 -f g94
    python -m vibeqc.basis_toolkit.cli convert sto-3g.bse.json -o sto-3g.orca -f orca
    python -m vibeqc.basis_toolkit.cli convert sto-3g.bse.json -o sto-3g.nwchem -f nwchem

    # Show loss report for a conversion
    python -m vibeqc.basis_toolkit.cli loss -f g94 sto-3g.bse.json

    # Inspect a basis set (summary)
    python -m vibeqc.basis_toolkit.cli inspect sto-3g.qvf.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _load_input(path: Path) -> "BasisSetData":
    """Load a basis set from any supported format, auto-detecting type."""
    from .importer_bse import from_bse_file
    from .importer_g94 import from_g94_file
    from .importer_nwchem import from_nwchem_file
    from .importer_orca import from_orca_file
    from .qvf_basis import load_any

    suffix = path.suffix.lower()
    name_lower = path.name.lower()
    if name_lower.endswith(".qvf.json"):
        return load_any(path)
    elif suffix == ".g94":
        return from_g94_file(path)
    elif suffix == ".orca":
        return from_orca_file(path)
    elif suffix == ".nwchem":
        return from_nwchem_file(path)
    elif suffix in (".qvf",):
        return load_any(path)
    elif _looks_like_bse(path):
        return from_bse_file(path)
    else:
        # Try each format in order
        for loader, label in [
            (lambda: from_bse_file(path), "BSE"),
            (lambda: from_g94_file(path), "G94"),
            (lambda: from_orca_file(path), "ORCA"),
            (lambda: from_nwchem_file(path), "NWChem"),
            (lambda: load_any(path), "QVF"),
        ]:
            try:
                return loader()
            except Exception:
                continue
        raise ValueError(f"Could not parse {path} as any known basis-set format")


def _looks_like_bse(path: Path) -> bool:
    """Heuristic: does this JSON file look like BSE format?"""
    try:
        data = json.loads(path.read_text())
        return (
            "elements" in data
            and "name" in data
            and not data.get("schema_version", "").startswith("1.")
        )
    except Exception:
        return False


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate one or more basis-set files (any supported format)."""
    from .validator import validate_basis_dict

    exit_code = 0
    errors_list: list[str] = []
    file_count = 0

    for path in args.files:
        p = Path(path)
        if p.is_dir():
            # Recurse into directory: validate all recognized files.
            for f in sorted(p.rglob("*")):
                if f.is_file() and (
                    f.suffix in (".g94", ".orca", ".nwchem", ".qvf")
                    or f.name.endswith(".qvf.json")
                    or (f.suffix == ".json" and _looks_like_bse(f))
                ):
                    exit_code |= _validate_one(f, errors_list)
                    file_count += 1
        else:
            if not p.exists():
                print(f"ERROR: file not found: {p}", file=sys.stderr)
                exit_code = 1
                continue
            exit_code |= _validate_one(p, errors_list)
            file_count += 1

    if errors_list and not args.quiet:
        for err in errors_list:
            print(err)
    if file_count == 0:
        print("No basis files found.", file=sys.stderr)
        return 1
    if exit_code == 0:
        print(f"OK  {file_count} file(s)")
    else:
        print(
            f"FAIL  {len(errors_list)} error(s) in {file_count} file(s)",
            file=sys.stderr,
        )
    return exit_code


def _validate_one(path: Path, errors_out: list[str]) -> int:
    """Validate a single file, appending errors to errors_out.  Returns 1 on failure."""
    from .validator import validate_basis_dict

    try:
        data = _load_input(path)
    except Exception as e:
        errors_out.append(f"{path}: parse error: {e}")
        return 1
    errs = validate_basis_dict(data.to_dict())
    if errs:
        for e in errs:
            errors_out.append(f"{path}: {e}")
        return 1
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert between basis-set formats."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        return 1

    data = _load_input(input_path)

    output_path = Path(args.output)
    fmt = args.format

    if fmt is None:
        # Auto-detect from output extension
        ext = output_path.suffix.lower()
        if ext == ".g94":
            fmt = "g94"
        elif ext == ".nwchem" or ext == ".nw":
            fmt = "nwchem"
        elif ext == ".orca":
            fmt = "orca"
        elif ext == ".qvf":
            fmt = "qvf"
        elif ext == ".json":
            fmt = "qvf_json"
        else:
            print(
                f"ERROR: cannot detect format from extension {ext!r}; use -f",
                file=sys.stderr,
            )
            return 1

    if fmt == "g94":
        from .exporter_g94 import to_g94_file

        loss = to_g94_file(data, output_path)
    elif fmt == "orca":
        from .exporter_orca import to_orca_file

        loss = to_orca_file(data, output_path)
    elif fmt == "nwchem":
        from .exporter_nwchem import to_nwchem_file

        loss = to_nwchem_file(data, output_path)
    elif fmt == "qvf":
        from .qvf_basis import save_qvf

        save_qvf(data, output_path)
        print(f"Wrote {output_path} ({len(data.elements)} elements)")
        return 0
    elif fmt == "qvf_json":
        from .qvf_basis import save_qvf_json

        save_qvf_json(data, output_path)
        print(f"Wrote {output_path} ({len(data.elements)} elements)")
        return 0
    else:
        print(f"ERROR: unknown format {fmt!r}", file=sys.stderr)
        return 1

    print(f"Wrote {output_path} ({len(data.elements)} elements)")
    if not loss.is_lossless:
        print(
            f"Warning: conversion was lossy ({len(loss.fields_dropped)} fields dropped)"
        )
        if args.verbose:
            print(loss.summary())
    return 0


def cmd_loss(args: argparse.Namespace) -> int:
    """Show the loss report for a given format."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        return 1

    data = _load_input(input_path)
    fmt = args.format

    if fmt == "g94":
        from .exporter_g94 import loss_report_g94

        report = loss_report_g94(data)
    elif fmt == "orca":
        from .exporter_orca import loss_report_orca

        report = loss_report_orca(data)
    elif fmt == "nwchem":
        from .exporter_nwchem import loss_report_nwchem

        report = loss_report_nwchem(data)
    else:
        print(f"ERROR: unknown format {fmt!r}", file=sys.stderr)
        return 1

    print(f"Format: {fmt}")
    print(f"Lossless: {report.is_lossless}")
    print(f"Fields dropped ({len(report.fields_dropped)}):")
    for w in report.warnings:
        print(f"  - {w}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print a human-readable summary of a basis set."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        return 1

    data = _load_input(input_path)

    print(f"Name:        {data.name}")
    print(f"Role:        {data.role.value}")
    print(f"Family:      {data.basis_family or '--'}")
    print(f"Description: {data.description or '--'}")
    print(f"Schema:      {data.schema_version}")
    print(f"Elements:    {len(data.elements)} ({', '.join(data.elems_sorted())})")
    print()

    total_shells = 0
    total_primitives = 0
    for sym in data.elems_sorted():
        eb = data.elements[sym]
        n_shells = len(eb.shells)
        n_prim = sum(len(s.primitives) for s in eb.shells)
        total_shells += n_shells
        total_primitives += n_prim

        shell_labels = []
        for s in eb.shells:
            am = s.angular_momentum
            if am == [0]:
                shell_labels.append("S")
            elif am == [1]:
                shell_labels.append("P")
            elif am == [2]:
                shell_labels.append("D")
            elif am == [3]:
                shell_labels.append("F")
            elif am == [0, 1]:
                shell_labels.append("SP")
            else:
                shell_labels.append(str(am))
        print(
            f"  {sym:>3s}  {n_shells:2d} shell(s), {n_prim:3d} primitive(s): {' '.join(shell_labels)}"
        )

    print(
        f"\nTotal: {total_shells} shells, {total_primitives} primitives across {len(data.elements)} elements"
    )

    if data.ecps:
        print(f"\nECPs: {len(data.ecps)} element(s)")
        for sym, ecp in data.ecps.items():
            print(
                f"  {sym}: {ecp.n_core_electrons} core electrons, L_max={ecp.angular_momentum_max}"
            )

    if data.references:
        print(f"\nReferences: {len(data.references)}")
        for ref in data.references:
            doi_str = f" (DOI: {ref.doi})" if ref.doi else ""
            print(f"  [{ref.key}] {ref.description}{doi_str}")

    if data.provenance:
        p = data.provenance
        print(f"\nProvenance:")
        print(f"  Origin:   {p.origin}")
        if p.source_version:
            print(f"  Version:  {p.source_version}")
        if p.source_doi:
            print(f"  DOI:      {p.source_doi}")
        if p.retrieval_date:
            print(f"  Date:     {p.retrieval_date}")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vibe-qc-basis",
        description="vibe-qc basis-set toolkit -- convert, validate, inspect",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- validate ---
    p_val = sub.add_parser("validate", help="Validate basis-set files")
    p_val.add_argument(
        "files",
        nargs="+",
        help=".qvf.json, .qvf, .g94, .orca, .nwchem files or directories to validate",
    )
    p_val.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress per-file output"
    )
    p_val.set_defaults(func=cmd_validate)

    # --- convert ---
    p_conv = sub.add_parser("convert", help="Convert between basis-set formats")
    p_conv.add_argument("input", help="Input file (.bse.json, .qvf.json, .qvf)")
    p_conv.add_argument("-o", "--output", required=True, help="Output file path")
    p_conv.add_argument(
        "-f",
        "--format",
        choices=["g94", "orca", "nwchem", "qvf", "qvf_json"],
        help="Output format",
    )
    p_conv.add_argument(
        "-v", "--verbose", action="store_true", help="Show loss report details"
    )
    p_conv.set_defaults(func=cmd_convert)

    # --- loss ---
    p_loss = sub.add_parser("loss", help="Show loss report for a format conversion")
    p_loss.add_argument("input", help="Input basis-set file")
    p_loss.add_argument(
        "-f",
        "--format",
        required=True,
        choices=["g94", "orca", "nwchem"],
        help="Target format for loss analysis",
    )
    p_loss.set_defaults(func=cmd_loss)

    # --- inspect ---
    p_insp = sub.add_parser(
        "inspect", help="Print human-readable summary of a basis set"
    )
    p_insp.add_argument("input", help="Input basis-set file")
    p_insp.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
