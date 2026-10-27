#!/usr/bin/env python3
"""Print a portable static eigensystem call-site inventory as Markdown.

This is a candidate inventory, not a semantic migration certificate. Python
uses AST calls; C++ lists lexical references, including declarations. Dynamic
dispatch and aliases require the manual family map in the design document.
Run from any directory; output contains repository-relative source paths only.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "eigh", "eigvalsh", "eig", "eigvals", "eigsh", "eigs",
    "solve_eigenproblem", "solve_operator", "solve_scf_step",
    "davidson_solve", "davidson_solve_matvec", "davidson_solve_hermitian",
    "davidson_solve_hermitian_matvec", "diagonalize_bloch",
    "davidson_solve_matmat", "davidson_solve_hermitian_matmat", "FockDiagonalizer",
    "_diag_in_orth_basis", "_eigh_safe", "solve_dense_fock", "_pnos",
}


def family(path):
    if path in {"cpp/src/manta_ri.cpp", "cpp/src/manta_ri_metric.cpp"}:
        return "A1"  # Complete auxiliary Coulomb-metric factorization, not a Fock solve.
    if path in {"cpp/src/rhf.cpp", "cpp/src/rks.cpp", "cpp/src/uhf.cpp", "cpp/src/uks.cpp"}:
        return "M1"
    if path in {"cpp/src/bloch.cpp", "cpp/src/periodic_scf.cpp", "python/vibeqc/periodic_rhf_multi_k_ewald.py"}:
        return "P1"
    if "eigensolver" in path or path.endswith(("/operators.py", "/_scf.py")) or any(path.endswith(f"/{n}.{ext}") for n in ("davidson", "lobpcg", "jd", "gplhr", "hermitian_operator") for ext in ("cpp", "hpp")):
        return "S1"
    if path == "cpp/src/semiempirical/dftb0.cpp":
        return "SE1"
    if "semiempirical" in path:
        return "SE2"
    if path == "python/vibeqc/tddft.py":
        return "R1"
    if "periodic/ccm" in path or "periodic/chi" in path:
        return "P3"
    if any(n in path for n in ("casci", "selected_ci", "_dmrg", "_cisd")):
        return "C1"
    if any(n in path for n in ("dlpno", "_mrpt", "_ms_caspt2", "/cc.py", "/ccsd.cpp")):
        return "C2"
    if "/gradient/" in path or "hessian" in path or "phonon" in path:
        return "A1"
    if any(n in path for n in ("guess", "rohf", "roks", "newton", "soscf", "trah", "orbital_objective", "scf_restart", "degenerate_frontier")):
        return "M2"
    if any(n in path for n in ("aux_basis", "orthogonalisation", "linear_dependence", "periodic_gdf_gradient", "/df.cpp", "/grid.cpp", "_v2rdm", "properties", "bond_analysis", "kerker", "relocalize", "qtaim", "geomopt", "entanglement", "output/", "nbo", "iao", "periodic_k_density", "periodic_mean_field_state", "_hamiltonian.py", "runner.py", "eda.py", "basis_optimization")):
        return "A1"
    if "periodic" in path or "pbc_" in path:
        return "P2"
    return "R2"


def inventory():
    for base, suffix in (("python/vibeqc", ".py"), ("cpp/src", ".cpp"), ("cpp/include", ".hpp"), ("cpp/src", ".h")):
        for path in sorted((ROOT / base).rglob(f"*{suffix}")):
            source = path.read_text()
            hits = []
            if suffix == ".py":
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.Call):
                        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                        if name in NAMES:
                            hits.append((node.lineno, name))
            else:
                for lineno, line in enumerate(source.splitlines(), 1):
                    if line.lstrip().startswith("//"):
                        continue
                    match = re.search(r"\b(SelfAdjointEigenSolver|GeneralizedSelfAdjointEigenSolver|GeneralizedEigenSolver|EigenSolver|davidson_solve\w*|lobpcg_solve\w*|jd_solve\w*|gplhr_solve\w*|solve_fock_matrix|solve_hermitian_matrix|solve_one_electron_metric|diagonalize_bloch|solve_bounded_periodic_rhf)\b", line)
                    if match:
                        hits.append((lineno, match.group(1)))
            if hits:
                relative = path.relative_to(ROOT).as_posix()
                yield relative, family(relative), sorted(hits)


if __name__ == "__main__":
    print("| Source | Family/status key | Audited static sites (line:symbol) |")
    print("|---|---|---|")
    for path, key, hits in sorted(inventory()):
        sites = ", ".join(f"{line}:{name}" for line, name in hits)
        print(f"| `{path}` | {key} | {sites} |")
