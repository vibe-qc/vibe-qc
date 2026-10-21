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

    # Render the bundled library's records into libint's .g94 / .ecp text,
    # or check that the committed rendering is in step with the records
    python -m vibeqc.basis_toolkit.cli render --target build/basis_library/basis
    python -m vibeqc.basis_toolkit.cli check

    # Put your own basis set into the user store, where BasisSet(mol, name)
    # resolves it like a bundled one; list, resolve and remove store records
    vibe-qc-basis import my-basis.g94 --kind hand-written --cite some_key
    vibe-qc-basis import run-42/mgo-opt.g94        # vibe-basis export: sidecar supplies kind and run id
    vibe-qc-basis list
    vibe-qc-basis resolve my-basis def2-svp
    vibe-qc-basis remove my-basis
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
    if args.ecp:
        from .importer_g94 import from_g94_file

        sidecar = from_g94_file(args.ecp, name=data.name, role="ecp")
        merged = dict(data.ecps or {})
        merged.update(sidecar.ecps or {})
        data.ecps = merged or None
        for sym, element in data.elements.items():
            element.ecp_id = sym if sym in merged else None
    _apply_provenance_flags(data, args)

    output_path = Path(args.output)
    fmt = args.format
    if fmt is None and output_path.name.lower().endswith(".qvf.json"):
        fmt = "qvf_json"

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


def _apply_provenance_flags(data, args: argparse.Namespace) -> None:
    """Fill name, description and the 1.1.0 provenance from ``convert`` flags."""
    from .model import SCHEMA_VERSION, CitationRoute, Provenance, ProvenanceKind, Upstream

    if getattr(args, "name", None):
        data.name = args.name
    if getattr(args, "description", None) is not None:
        data.description = args.description
    kind = getattr(args, "kind", None)
    upstream = getattr(args, "upstream", None) or []
    cites = getattr(args, "cite", None) or []
    date = getattr(args, "retrieval_date", None)
    run_id = getattr(args, "run_id", None)
    if not (kind or upstream or cites or date or run_id):
        return
    prov = data.provenance or Provenance(origin="file")
    if kind:
        prov.kind = ProvenanceKind(kind)
    for spec in upstream:
        name, _, version = spec.partition("@")
        prov.upstream.append(Upstream(name=name, version=version or None))
    if cites:
        prov.citations.append(CitationRoute(keys=list(cites)))
    if date:
        prov.retrieval_date = date
    if run_id:
        prov.run_id = run_id
    data.schema_version = SCHEMA_VERSION
    prov.content_sha256 = data.content_sha256()
    data.provenance = prov


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
        if p.kind is not None:
            print(f"  Kind:     {p.kind.value}")
        for up in p.upstream:
            detail = f" ({up.detail})" if up.detail else ""
            version = f" {up.version}" if up.version else ""
            print(f"  Upstream: {up.name}{version}{detail}")
        if p.source_version:
            print(f"  Version:  {p.source_version}")
        if p.source_doi:
            print(f"  DOI:      {p.source_doi}")
        if p.retrieval_date:
            print(f"  Date:     {p.retrieval_date}")
        if p.content_sha256:
            print(f"  Content:  sha256 {p.content_sha256}")
        for route in p.citations:
            cover = ", ".join(route.elements) if route.elements is not None else "all elements"
            print(f"  Cites:    {', '.join(route.keys)}  [{cover}]")
        if p.run_id:
            print(f"  Run:      {p.run_id}")

    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Render every bundled record into a ``basis/`` directory."""
    from . import library

    target = Path(args.target) if args.target else library.renderings_dir()
    report = library.render_library(target)
    print(
        f"Rendered {len(library.stems())} records into {target}: "
        f"{len(report.written)} written, {len(report.unchanged)} unchanged, "
        f"{len(report.removed)} stale file(s) removed."
    )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Compare a rendering with the records; exit 1 on any difference."""
    from . import library

    target = Path(args.target) if args.target else library.renderings_dir()
    problems = library.check_library(target)
    for name, why in problems.items():
        print(f"{name}: {why}")
    if problems:
        print(
            f"FAIL  {len(problems)} rendering(s) out of step with the records; "
            "re-render with: vibe-qc-basis render",
            file=sys.stderr,
        )
        return 1
    print(f"OK  {len(library.stems())} records, renderings in {target} match")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """Import a basis into the user store (#410)."""
    from .. import basis_store
    from .model import Upstream

    upstream = []
    for spec in args.upstream or []:
        name, _, version = spec.partition("@")
        upstream.append(Upstream(name=name, version=version or None))
    try:
        stored = basis_store.import_record(
            args.source, name=args.name, kind=args.kind,
            replace_bundled=args.replace_bundled, replace=args.replace,
            description=args.description, upstream=upstream,
            citation_keys=args.cite or (), run_id=args.run_id, notes=args.notes,
            root=args.store,
        )
    except (basis_store.BasisStoreError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Imported {stored.name!r} ({stored.kind}) into {stored.path.parent.parent}")
    print(f"  content sha256 {stored.content_sha256}")
    print(f"  rendering      {stored.g94_path}" + (f", {stored.ecp_path}" if stored.ecp_path else ""))
    if stored.replaces_bundled:
        print("  replaces the bundled record of the same name for this user")
    print("BasisSet(mol, %r) resolves it in every new vibe-qc process." % stored.name)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List the user store's records."""
    from .. import basis_store

    records = basis_store.list_records(args.store)
    root = basis_store.user_store_root(args.store)
    if not records:
        print(f"No records in the user store {root}")
        return 0
    print(f"User store {root}: {len(records)} record(s)")
    for record in records:
        flag = "  (replaces bundled)" if record.replaces_bundled else ""
        run = f"  run {record.extra.get('run_id')}" if record.extra.get("run_id") else ""
        print(f"  {record.name:<28s} {record.kind:<13s} {record.content_sha256[:16]}  {record.recorded_at[:10]}{flag}{run}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    from .. import basis_store

    if basis_store.remove_record(args.name, root=args.store):
        print(f"Removed {args.name!r} from the user store")
        return 0
    print(f"{args.name!r} is not in the user store", file=sys.stderr)
    return 1


def cmd_resolve(args: argparse.Namespace) -> int:
    """Say which layer answers each name, without fetching."""
    import os

    from .. import basis_store
    from ..basis_registry import basis_file_path
    from ..basis_toolkit import library

    store = {r.name: r for r in basis_store.list_records(args.store)}
    status = 0
    for name in args.names:
        stem = name.strip().lower()
        if stem in store and (store[stem].replaces_bundled or stem not in library.stems()):
            where = f"user store ({store[stem].kind}, sha256 {store[stem].content_sha256[:16]})"
        elif stem in library.stems():
            where = "bundled library"
        elif basis_file_path(name) is not None:
            where = f"LIBINT_DATA_PATH {os.environ.get('LIBINT_DATA_PATH', '')}"
        else:
            hit = None
            try:
                hit = basis_store.cache_get(name)
            except basis_store.BasisStoreIntegrityError as exc:
                where = f"fetch cache, FAILS verification: {exc}"
                status = 1
            else:
                where = (
                    f"fetch cache ({hit.extra.get('bse_name', '?')} from BSE {hit.extra.get('bse_version', '?')})"
                    if hit is not None else "unresolved (not bundled, not in the store or cache)"
                )
                if hit is None:
                    status = 1
        print(f"{name:<28s} -> {where}")
    return status


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
    p_conv.add_argument("--ecp", help="Gaussian94 .ecp sidecar to merge into the record")
    p_conv.add_argument("--name", help="Record name (the identifier users pass to BasisSet)")
    p_conv.add_argument("--description", help="Free-text description stored in the record")
    p_conv.add_argument(
        "--kind", choices=["vendor", "bse", "hand-written", "vibe-basis"],
        help="provenance.kind of the record",
    )
    p_conv.add_argument(
        "--upstream", action="append", metavar="NAME[@VERSION]",
        help="An upstream source, e.g. 'basis-set-exchange@0.12' (repeatable)",
    )
    p_conv.add_argument(
        "--cite", action="append", metavar="KEY",
        help="Citation key in vibeqc's database.toml routed over every element (repeatable)",
    )
    p_conv.add_argument("--retrieval-date", dest="retrieval_date", help="ISO date the data was obtained")
    p_conv.add_argument("--run-id", dest="run_id", help="vibe-basis run identity")
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

    # --- render / check ---
    p_render = sub.add_parser(
        "render", help="Render the bundled records into libint's .g94 / .ecp files"
    )
    p_render.add_argument(
        "--target", help="basis/ directory to write (default: the committed rendering)"
    )
    p_render.set_defaults(func=cmd_render)

    p_check = sub.add_parser(
        "check", help="Verify a rendering is byte for byte what the records render to"
    )
    p_check.add_argument(
        "--target", help="basis/ directory to compare (default: the committed rendering)"
    )
    p_check.set_defaults(func=cmd_check)

    # --- the user store ---
    p_imp = sub.add_parser(
        "import", help="Put a basis set into the user store so BasisSet resolves it by name"
    )
    p_imp.add_argument("source", help=".qvf.json, .g94 (with an .ecp beside it), BSE .json, CRYSTAL file or directory")
    p_imp.add_argument("--name", help="Record name (default: the source's name)")
    p_imp.add_argument("--kind", choices=["hand-written", "vibe-basis"],
                       help="What the record is (a provenance sidecar may supply it)")
    p_imp.add_argument("--replace-bundled", action="store_true", dest="replace_bundled",
                       help="Let this record win over a bundled set of the same name")
    p_imp.add_argument("--replace", action="store_true", help="Overwrite a store record of the same name")
    p_imp.add_argument("--description")
    p_imp.add_argument("--upstream", action="append", metavar="NAME[@VERSION]")
    p_imp.add_argument("--cite", action="append", metavar="KEY", help="database.toml citation key (repeatable)")
    p_imp.add_argument("--run-id", dest="run_id", help="vibe-basis run identity")
    p_imp.add_argument("--notes")
    p_imp.add_argument("--store", help="Store root (default: $VIBEQC_BASIS_STORE or the XDG data dir)")
    p_imp.set_defaults(func=cmd_import)

    p_list = sub.add_parser("list", help="List the user store's records")
    p_list.add_argument("--store")
    p_list.set_defaults(func=cmd_list)

    p_rm = sub.add_parser("remove", help="Remove a record from the user store")
    p_rm.add_argument("name")
    p_rm.add_argument("--store")
    p_rm.set_defaults(func=cmd_remove)

    p_res = sub.add_parser("resolve", help="Say which layer answers each basis name")
    p_res.add_argument("names", nargs="+")
    p_res.add_argument("--store")
    p_res.set_defaults(func=cmd_resolve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
