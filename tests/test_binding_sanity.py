"""Symbol-table guard against stale ``_vibeqc_core.so`` builds.

Every ``from ._vibeqc_core import X`` (or ``import X as Y``) anywhere in
the ``python/vibeqc/`` source tree is a contract: the symbol ``X`` has to
exist in the compiled C++ extension at runtime. When upstream lands a
new binding (e.g. ``39bf8629`` added ``compute_nuclear_with_charges`` to
both the ``__init__`` import list and ``cpp/src/bindings.cpp``), every
downstream dev chat with a pre-existing local ``.so`` silently breaks
until they rebuild — ``import vibeqc`` then fails partway through the
module body with ``ImportError: cannot import name 'X'``.

This test catches the entire class in seconds: walk the package, parse
each module's AST for ``ImportFrom`` nodes against ``_vibeqc_core``, and
assert every requested name is present on the loaded extension module.
If a chat's ``.so`` is stale, the test fails on the first missing symbol
with a clear ``rebuild required`` message naming what to do.

Per CLAUDE.md §9 (every commit ships green) and the integrals-chat
notes now retained in ``handovers/HANDOVER_GDF_V0_11_2026_05_29.md``
(Item 2b).
"""

from __future__ import annotations

import ast
import pathlib

import vibeqc
from vibeqc import _vibeqc_core as core


_PACKAGE_ROOT = pathlib.Path(vibeqc.__file__).resolve().parent


def _collect_requested_native_symbols() -> dict[str, list[pathlib.Path]]:
    """Walk ``python/vibeqc/`` and return ``{symbol: [files requesting it]}``.

    Considers both ``from ._vibeqc_core import X`` (relative) and
    ``from vibeqc._vibeqc_core import X`` (absolute) forms. The symbol
    we record is the *original* name in the C++ extension — i.e. the
    left-hand side of ``import X as Y``, not the alias ``Y``.
    """
    requested: dict[str, list[pathlib.Path]] = {}

    for py_path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in py_path.parts:
            continue
        try:
            source = py_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(source, filename=str(py_path))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            # Match `from ._vibeqc_core import ...` (relative, level=1)
            # and `from vibeqc._vibeqc_core import ...` (absolute).
            is_relative = node.level == 1 and node.module == "_vibeqc_core"
            is_absolute = node.level == 0 and node.module == "vibeqc._vibeqc_core"
            if not (is_relative or is_absolute):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue  # not used in vibe-qc; would defeat the point
                requested.setdefault(alias.name, []).append(py_path)

    return requested


def test_every_imported_native_symbol_resolves() -> None:
    """Every name imported from ``_vibeqc_core`` must exist on the .so.

    Catches the canonical "stale local .so after upstream binding change"
    failure mode. If this fails, rebuild with::

        pip install -e . --no-build-isolation

    or equivalently ``./scripts/install.sh`` after pulling.
    """
    requested = _collect_requested_native_symbols()
    assert requested, (
        f"binding-sanity walked {_PACKAGE_ROOT} but found no "
        f"`from ._vibeqc_core import ...` statements. The collector "
        f"is broken — investigate before trusting this test."
    )

    missing: dict[str, list[pathlib.Path]] = {}
    for symbol, sites in requested.items():
        if not hasattr(core, symbol):
            missing[symbol] = sites

    if missing:
        lines = [
            "_vibeqc_core.so is missing symbols that python/vibeqc/ "
            "imports. Your local .so is stale — rebuild with:",
            "",
            "    pip install -e . --no-build-isolation",
            "",
            "Missing symbols (and the files that import them):",
            "",
        ]
        for symbol, sites in sorted(missing.items()):
            lines.append(f"  - {symbol}")
            for site in sites[:3]:
                rel = site.relative_to(_PACKAGE_ROOT.parent.parent)
                lines.append(f"      {rel}")
            if len(sites) > 3:
                lines.append(f"      ... and {len(sites) - 3} more")
        raise AssertionError("\n".join(lines))


def test_init_reexports_match_native_module() -> None:
    """``vibeqc.X`` and ``vibeqc._vibeqc_core.X`` must be the same object.

    Sister check to the symbol-existence test above: even with every
    name present, a partially-loaded module (e.g. an aborted reload
    leaving stale bindings on the package namespace) can have
    ``vibeqc.RHFOptions is not core.RHFOptions``. Pin a handful of
    high-traffic public names so a regression is caught quickly.
    """
    # Only pure pass-through reexports — names like ``run_rhf`` are
    # redefined as Python wrappers in ``__init__.py`` and would fail an
    # identity check by design.
    pinned = [
        "BasisSet",
        "Molecule",
        "PeriodicSystem",
        "RHFOptions",
        "RKSOptions",
        "PeriodicRHFOptions",
        "compute_overlap",
        "compute_eri",
        "compute_3c_eri_lattice",
        "compute_nuclear_with_charges",
        "direct_lattice_cells",
    ]
    mismatches: list[str] = []
    for name in pinned:
        if not hasattr(core, name):
            mismatches.append(f"{name} missing from _vibeqc_core")
            continue
        if not hasattr(vibeqc, name):
            mismatches.append(f"{name} missing from vibeqc package namespace")
            continue
        if getattr(vibeqc, name) is not getattr(core, name):
            mismatches.append(
                f"{name} differs: vibeqc.{name} is not "
                f"_vibeqc_core.{name} — possible stale reload"
            )
    assert mismatches == [], "\n".join(mismatches)
