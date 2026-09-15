"""Guard: vibe-basis must never import vibe-qc.

The dependency arrow between the two packages runs **one way only**:

    vibe-qc  ──[basisopt] extra──►  vibe-basis      OK
    vibe-basis  ──────────────►  vibe-qc            FORBIDDEN

vibe-basis is a *driver*: it emits decks for external SCF programs,
submits them, and parses what comes back.  It computes nothing, needs
no native build, and carries no transitive GPL (vibe-qc dynamic-links
FFTW3, which is GPL v2).  Those three properties are what make
``pip install vibe-basis`` a ten-second install for a collaborator who
has their own CRYSTAL and no interest in building vibe-qc.

A single ``import vibeqc`` anywhere under ``src/`` destroys all three
silently -- the import would resolve fine on a developer machine where
vibe-qc happens to be installed, and fail only for the collaborator.
Hence a mechanical guard rather than a convention.

The rule and its rationale are recorded in
``handovers/HANDOVER_BASISOPT_COHESIVE_ROADMAP.md`` section 3.6.

If you need a vibe-qc capability from driver code, invert the call:
declare a protocol here and implement it in
``python/vibeqc/basis_optimization/`` (tier 2), which *may* import
vibe-basis.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

# Anything whose top-level package is one of these is a vibe-qc import.
# ``vibeqc`` is the module path; ``vibe_qc`` is not currently used but is
# the spelling a well-meaning contributor would try next.
FORBIDDEN_ROOTS = frozenset({"vibeqc", "vibe_qc"})


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _forbidden_imports(tree: ast.AST) -> list[tuple[int, str]]:
    """Return ``(lineno, module)`` for every vibe-qc import in *tree*.

    Catches both ``import vibeqc[...]`` and ``from vibeqc[...] import x``,
    at any nesting depth -- a function-level import is exactly as fatal
    as a module-level one for the collaborator install, it just fails
    later.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_ROOTS:
                    hits.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            # ``node.module`` is None for relative imports (``from . import x``),
            # which are always fine -- they cannot reach outside the package.
            if node.module and node.module.split(".")[0] in FORBIDDEN_ROOTS:
                hits.append((node.lineno, node.module))
    return hits


def test_src_tree_is_non_empty() -> None:
    """Self-check: a guard that scans nothing passes vacuously."""
    files = _python_files()
    assert len(files) >= 10, f"expected the vibe_basis source tree, found {files}"


def test_no_module_imports_vibeqc() -> None:
    """No module under ``src/`` may import vibe-qc, at any nesting depth."""
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, module in _forbidden_imports(tree):
            offenders.append(f"{path.relative_to(SRC)}:{lineno}: imports {module!r}")

    assert not offenders, (
        "vibe-basis must not import vibe-qc (see this module's docstring "
        "and HANDOVER_BASISOPT_COHESIVE_ROADMAP.md section 3.6):\n  "
        + "\n  ".join(offenders)
    )


def test_guard_detects_a_planted_import() -> None:
    """The guard must actually fire -- a green run should mean something."""
    planted = ast.parse(
        "import numpy\n"
        "from vibe_basis.optimize import OptResult\n"
        "def f():\n"
        "    import vibeqc as vq\n"
        "    return vq\n"
    )
    assert _forbidden_imports(planted) == [(4, "vibeqc")]

    planted_from = ast.parse("from vibeqc.basis_optimization import OptResult\n")
    assert _forbidden_imports(planted_from) == [(1, "vibeqc.basis_optimization")]
