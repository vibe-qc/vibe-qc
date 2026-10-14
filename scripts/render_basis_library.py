#!/usr/bin/env python3
"""Render the bundled basis library from its QVF-Basis records.

``python/vibeqc/basis_library/qvf/*.qvf.json`` are the records; this writes
the ``.g94`` orbital files libint reads and the ``.ecp`` sidecars libecpint
reads from them (#406). It is what ``scripts/setup_basis_library.sh`` runs
to materialise the runtime overlay, and what a developer runs after
changing a record to refresh the committed ``basis/`` rendering.

Standard library only, and it loads ``vibeqc.basis_toolkit`` by path
without importing ``vibeqc`` itself, so it works on a checkout whose
native core is not built yet (the install scripts run it before ``pip
install``).

Usage::

    python scripts/render_basis_library.py --check
    python scripts/render_basis_library.py --target build/basis_library/basis
    python scripts/render_basis_library.py --target python/vibeqc/basis_library/basis

``--check`` compares the committed rendering with the records and exits 1
on any difference, naming the files. Python 3.11 or newer.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "python" / "vibeqc" / "basis_toolkit"


def load_toolkit():
    """Import ``basis_toolkit`` as a top-level package, by path.

    Relative imports inside the package resolve against this name, and
    nothing in the modules used here imports ``vibeqc``.
    """
    name = "vibeqc_basis_toolkit"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target", type=Path, default=None,
        help="basis/ directory to write (default: the committed rendering)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="compare instead of writing; exit 1 when a rendering differs",
    )
    parser.add_argument(
        "--records", type=Path, default=None,
        help="library root holding qvf/ (default: python/vibeqc/basis_library)",
    )
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        print("render_basis_library.py needs Python 3.11 or newer", file=sys.stderr)
        return 2

    toolkit = load_toolkit()
    library = toolkit.library
    root = args.records
    target = args.target if args.target is not None else library.renderings_dir(root)

    if args.check:
        problems = library.check_library(target, root=root)
        count = len(library.stems(root))
        if problems:
            for name, why in problems.items():
                print(f"{name}: {why}")
            print(
                f"{len(problems)} rendering(s) out of step with the {count} records "
                f"under {library.records_dir(root)}. Re-render with: "
                f"python scripts/render_basis_library.py --target {target}",
                file=sys.stderr,
            )
            return 1
        print(f"Basis library in step: {count} records, renderings in {target} match.")
        return 0

    report = library.render_library(target, root=root)
    print(
        f"Rendered {len(library.stems(root))} records into {target}: "
        f"{len(report.written)} written, {len(report.unchanged)} unchanged, "
        f"{len(report.removed)} stale file(s) removed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
