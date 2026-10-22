"""External-reference boundary tests.

PySCF/CRYSTAL/ORCA are validation references, not vibe-qc backends. The
PySCF regression runner may launch an external Python process, but the
parent process must not import PySCF directly.

The broader scan (``test_no_forbidden_qc_imports_in_runtime_tree``)
enforces CLAUDE.md § 10 / AGENTS.md project rule 2 across the entire
``python/vibeqc/`` + ``cpp/`` runtime tree: no source file inside
those directories may ``import`` (Python) or ``#include`` (C++) any
external quantum-chemistry program. Libraries that vibe-qc links
against (libint, libxc, spglib, libecpint, Eigen, FFTW3, pybind11,
ASE, numpy, scipy) are explicitly allowed; full QC programs are not.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from examples.regression.core import runner_pyscf


# ---------------------------------------------------------------------------
# Runner sanity
# ---------------------------------------------------------------------------


def test_pyscf_regression_parent_has_no_direct_pyscf_imports() -> None:
    source = Path(runner_pyscf.__file__).read_text(encoding="utf-8")

    assert ("\nimport " + "pyscf") not in source
    assert ("\nfrom " + "pyscf") not in source


def test_pyscf_external_result_marker_parser() -> None:
    parsed = runner_pyscf._parse_external_result(
        "setup log\n"
        "VIBEQC-PYSCF-RESULT:{\"status\": \"ok\", \"energy_ha\": -1.0}\n",
    )

    assert parsed is not None
    assert parsed["status"] == "ok"
    assert parsed["energy_ha"] == -1.0


# ---------------------------------------------------------------------------
# Repo-wide guard — no forbidden QC-program imports under the runtime tree
# ---------------------------------------------------------------------------

# Programs that "run on their own as a quantum-chemistry program" per
# AGENTS.md § "External programs vs vendored libraries". Each entry is
# a Python top-level module / C++ public-header prefix that we must
# never pull into the vibe-qc runtime.
_FORBIDDEN_QC_PROGRAMS: tuple[str, ...] = (
    "pyscf",
    "psi4",
    "orca",
    "turbomole",
    "crystal",
    "nwchem",
    "gamess",
    "qchem",
    "molpro",
    "adf",
    "cp2k",
    "vasp",
    "quantum_espresso",
    "xtb",
    "gaussian",
)

# Exact-name allowlist: identifiers that match a forbidden prefix as a
# substring but are NOT one of those QC programs. Each entry is the
# full module/header name as it would appear after ``import`` /
# ``from`` / ``#include``. Keep this list minimal and explicit so a
# real regression cannot hide behind a fuzzy substring exemption.
_ALLOWED_EXACT_IMPORTS: frozenset[str] = frozenset(
    {
        # ASE's wrappers around external binaries — ASE itself is a
        # library and these wrappers invoke the external program as a
        # subprocess. Allowed per AGENTS.md § "External programs vs
        # vendored libraries" (ASE explicitly listed as a library).
        "ase.calculators.orca",
        "ase.calculators.nwchem",
        "ase.calculators.gaussian",
        "ase.calculators.crystal",
        "ase.calculators.psi4",
    }
)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_runtime_files() -> list[Path]:
    return sorted((_REPO_ROOT / "python" / "vibeqc").rglob("*.py"))


def _cpp_runtime_files() -> list[Path]:
    base = _REPO_ROOT / "cpp"
    files: list[Path] = []
    for pattern in ("*.cpp", "*.cc", "*.hpp", "*.h", "*.hh", "*.cxx"):
        files.extend(base.rglob(pattern))
    return sorted(files)


def _python_import_hits(text: str, filename: str = "<source>") -> list[tuple[int, str]]:
    """(line_number, module) for every absolute import statement in ``text``.

    Parsed, not pattern-matched, so a docstring or comment line that starts
    with "from PySCF ..." is prose, not an import (#542), while lazy imports
    inside functions are still found. Relative imports are vibe-qc's own.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(text, filename=filename)):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


_CPP_INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s+[<"]([^>"]+)[>"]',
    re.MULTILINE,
)


def _hits(text: str, regex: re.Pattern[str]) -> list[tuple[int, str]]:
    """Return a list of (line_number, matched_identifier)."""
    found: list[tuple[int, str]] = []
    for m in regex.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        found.append((line_no, m.group(1)))
    return found


def _is_forbidden(identifier: str) -> bool:
    if identifier in _ALLOWED_EXACT_IMPORTS:
        return False
    head = identifier.split(".", 1)[0].lower()
    return head in _FORBIDDEN_QC_PROGRAMS


def test_no_forbidden_qc_imports_in_python_runtime() -> None:
    """python/vibeqc/ may not ``import`` any external QC program.

    See CLAUDE.md § 10 and AGENTS.md project rule 2. Parity testing
    against external programs is done out-of-process via subprocess
    runners under ``examples/regression/`` — those files live outside
    ``python/vibeqc/`` and are not scanned here.
    """
    offenders: list[str] = []
    for path in _python_runtime_files():
        text = path.read_text(encoding="utf-8")
        for line_no, ident in _python_import_hits(text, str(path)):
            if _is_forbidden(ident):
                rel = path.relative_to(_REPO_ROOT)
                offenders.append(f"{rel}:{line_no}: import {ident}")
    assert not offenders, (
        "Forbidden external-QC imports detected under python/vibeqc/. "
        "External QC programs (PySCF, Psi4, ORCA, CRYSTAL, ...) must "
        "be driven out-of-process via subprocess runners under "
        "examples/regression/core/, not imported in-process. See "
        "CLAUDE.md § 10 / AGENTS.md project rule 2.\n\n"
        + "\n".join(offenders)
    )


def test_python_import_scan_ignores_prose_and_finds_real_imports() -> None:
    """The scan behind the runtime guard (#542): prose that begins with
    "from PySCF" or "import pyscf" is not an import; real import
    statements are found wherever they sit."""
    source = '''"""Agrees to 3.4 uHa
from PySCF MDF at the default mesh.
import pyscf is what a reference process does.
"""
# from pyscf import gto, in a comment
import numpy
import pyscf.pbc.gto as cell_gto


def reference():
    from pyscf.pbc import scf
    import ase.calculators.orca
    from . import sibling
    return scf
'''
    hits = _python_import_hits(source)
    forbidden = [(line, ident) for line, ident in hits if _is_forbidden(ident)]
    assert forbidden == [(7, "pyscf.pbc.gto"), (11, "pyscf.pbc")]
    assert (12, "ase.calculators.orca") in hits
    assert all(ident != "PySCF" for _, ident in hits)


def test_no_forbidden_qc_includes_in_cpp_runtime() -> None:
    """cpp/ may not ``#include`` any external QC program's headers.

    The C++ core links against numerical libraries only (libint,
    libxc, spglib, libecpint, Eigen, FFTW3, pybind11). It must never
    pull headers from another full QC program's source tree.
    """
    offenders: list[str] = []
    for path in _cpp_runtime_files():
        text = path.read_text(encoding="utf-8")
        for line_no, ident in _hits(text, _CPP_INCLUDE_RE):
            head = ident.split("/", 1)[0].lower()
            if head in _FORBIDDEN_QC_PROGRAMS:
                rel = path.relative_to(_REPO_ROOT)
                offenders.append(f"{rel}:{line_no}: #include <{ident}>")
    assert not offenders, (
        "Forbidden external-QC #include detected under cpp/. The C++ "
        "core may only include numerical libraries (libint, libxc, "
        "spglib, libecpint, Eigen, FFTW3, pybind11). See CLAUDE.md "
        "§ 10 / AGENTS.md project rule 2.\n\n" + "\n".join(offenders)
    )
