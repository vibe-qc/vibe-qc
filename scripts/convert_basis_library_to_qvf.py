#!/usr/bin/env python3
"""Convert vibe-qc's bundled G94 basis library to QVF-Basis (.qvf.json).

Produces a dual-stored library: the existing .g94 files stay for libint;
new .qvf.json sidecars become the structured source of truth.

Usage::

    .venv/bin/python scripts/convert_basis_library_to_qvf.py           # dry run
    .venv/bin/python scripts/convert_basis_library_to_qvf.py --write   # write files
    .venv/bin/python scripts/convert_basis_library_to_qvf.py --validate # validate all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Locate the package root relative to this script.
REPO = Path(__file__).resolve().parent.parent
LIBRARY = REPO / "python" / "vibeqc" / "basis_library"
BASIS_DIR = LIBRARY / "basis"
CUSTOM_DIR = LIBRARY / "custom"
QVF_DIR = LIBRARY / "qvf"  # new directory for QVF sidecars


def merge_ecp_sidecar(data, ecp_path: Path):
    """Fold the ``.ecp`` sidecar next to a ``.g94`` into its QVF record.

    libint reads the orbital file; libecpint reads the sidecar. The QVF
    sidecar is the structured record of *both*, so every ECP block joins
    ``data.ecps`` and every element that has one is linked through
    ``ecp_id``. Without this the shipped dhf-TZVP record carried 18 of its
    35 ECPs and linked none of them.
    """
    from vibeqc.basis_toolkit import from_g94

    ecp_data = from_g94(
        ecp_path.read_text(encoding="utf-8"), name=data.name, role="ecp"
    )
    if not ecp_data.ecps:
        return data
    merged = dict(data.ecps or {})
    merged.update(ecp_data.ecps)
    data.ecps = merged
    for symbol, element in data.elements.items():
        if symbol in merged:
            element.ecp_id = symbol
    return data


def convert_g94_to_qvf(g94_path: Path, qvf_path: Path) -> int:
    """Convert a single .g94 file (plus its .ecp sidecar, when one exists)
    to .qvf.json.  Returns 0 on success, 1 on error."""
    from vibeqc.basis_toolkit import from_g94_file, save_qvf_json

    try:
        data = from_g94_file(g94_path)
        ecp_path = g94_path.with_suffix(".ecp")
        if ecp_path.is_file():
            data = merge_ecp_sidecar(data, ecp_path)
        save_qvf_json(data, qvf_path)
        return 0
    except Exception as e:
        print(f"  ERROR: {g94_path.name}: {e}", file=sys.stderr)
        return 1


def validate_qvf(qvf_path: Path) -> int:
    """Validate a .qvf.json file against the schema.  Returns 0 on success."""
    from vibeqc.basis_toolkit import validate_basis_file

    errors = validate_basis_file(qvf_path)
    if errors:
        # Filter out the "jsonschema not available" warning for CI.
        real_errors = [e for e in errors if "jsonschema not available" not in e]
        if real_errors:
            for e in real_errors:
                print(f"  {qvf_path.name}: {e}", file=sys.stderr)
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert vibe-qc G94 basis library to QVF-Basis"
    )
    parser.add_argument(
        "--write", action="store_true", help="Write .qvf.json files (default: dry run)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing .qvf.json files against the schema",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="With --write: rewrite sidecars that already exist",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="NAME",
        help="Restrict to these basis stems (repeatable)",
    )
    args = parser.parse_args(argv)

    if not LIBRARY.exists():
        print(f"Library not found at {LIBRARY}", file=sys.stderr)
        return 1

    g94_files = sorted(CUSTOM_DIR.glob("*.g94")) + sorted(BASIS_DIR.glob("*.g94"))
    if not g94_files:
        print("No .g94 files found in basis library.")
        return 0

    # Deduplicate: custom/ files take priority over basis/ files with the same name.
    seen: set[str] = set()
    unique_files: list[Path] = []
    for f in g94_files:
        if f.name not in seen:
            seen.add(f.name)
            unique_files.append(f)
    g94_files = unique_files
    if args.only:
        wanted = {name.lower() for name in args.only}
        g94_files = [f for f in g94_files if f.stem.lower() in wanted]
    if not g94_files:
        print("No .g94 files found in basis library.")
        return 0

    exit_code = 0

    if args.validate:
        qvf_files = sorted(QVF_DIR.glob("*.qvf.json"))
        if not qvf_files:
            print("No .qvf.json files found to validate.")
            return 1
        print(f"Validating {len(qvf_files)} QVF-Basis files...")
        for qvf_path in qvf_files:
            if validate_qvf(qvf_path) != 0:
                exit_code = 1
        if exit_code == 0:
            print(f"All {len(qvf_files)} files valid.")
        return exit_code

    print(f"Found {len(g94_files)} G94 files across basis/ + custom/")
    to_convert = []
    for g94_path in g94_files:
        qvf_path = QVF_DIR / g94_path.with_suffix(".qvf.json").name
        if qvf_path.exists() and not args.refresh:
            to_convert.append((g94_path, qvf_path, True))
        else:
            to_convert.append((g94_path, qvf_path, False))

    existing = sum(1 for _, _, e in to_convert if e)
    new_count = len(to_convert) - existing

    if args.write:
        QVF_DIR.mkdir(exist_ok=True)
        print(
            f"Writing {new_count} new QVF-Basis files (skipping {existing} existing)..."
        )
        for g94_path, qvf_path, exists in to_convert:
            if exists:
                continue
            status = convert_g94_to_qvf(g94_path, qvf_path)
            if status != 0:
                exit_code = 1
            else:
                print(f"  {qvf_path.name}")
        if exit_code == 0:
            print(f"Done.  {new_count} files written to {QVF_DIR}")
            print(f"Run with --validate to verify, or:")
            print(f"  python -m vibeqc.basis_toolkit.cli validate {QVF_DIR}")
    else:
        print(
            f"Dry run: {new_count} new files would be written (skipping {existing} existing)"
        )
        print("Run with --write to convert.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
