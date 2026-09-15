"""Guard: the canonical ``.out`` writers emit through ``vibeqc.output``, never ``print()``.

CLAUDE.md Sec. 16 requires every user-facing byte to flow through the central
output surface -- ``write()`` / :class:`~vibeqc.output.OutputChannel` / the
document layer / ``ProgressLogger`` -- not a bare ``print()``. This test pins
that invariant for the files that actually format and emit ``.out`` content:
the molecular and periodic runners, the double-hybrid writer in the package
``__init__``, and the output module's own writers. A future edit that reaches
for ``print()`` in one of these trips here instead of silently re-splitting the
output surface the way ``runner.py`` and ``periodic_runner.py`` had drifted
before the logger workstream unified them.

Deliberately **scoped**, not a repo-wide ``print()`` ban: console entry points
(``_cli.py``), live-progress (``ProgressLogger``), benchmark / data-generation
harnesses, and ``__main__`` demos legitimately print, and telling those apart
from ``.out`` content that escaped needs more than a grep (CLAUDE.md Sec. 16,
"Current conformance"). The curated list below is the subset that is both
proven clean and central enough that any ``print()`` in it is a regression.
The sibling ``tests/test_semiempirical_output_contract.py`` pins the same
invariant for the semiempirical subtree; ``tests/test_cpp_emits_no_user_output.py``
pins it for the C++ core.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]

# The files that format / emit ``.out`` content. Each is verified ``print()``-
# free today; a ``print()`` appearing in one means user-facing output escaped
# the central channel. Keep this list curated by role, not by directory -- the
# output ``formats/`` command-line validators (currently QVF) keep their
# legitimate console prints and are intentionally absent. Non-CLI artefact
# writers belong here even when their only former print was an I/O warning.
_GUARDED_WRITERS = (
    "python/vibeqc/runner.py",
    "python/vibeqc/periodic_runner.py",
    "python/vibeqc/bipole_optimize.py",
    "python/vibeqc/geomopt/optimizers.py",
    "python/vibeqc/geomopt/periodic_providers.py",
    "python/vibeqc/geomopt/ts.py",
    "python/vibeqc/__init__.py",
    "python/vibeqc/output/__init__.py",
    "python/vibeqc/output/channel.py",
    "python/vibeqc/output/document.py",
    "python/vibeqc/output/writer.py",
    "python/vibeqc/output/formats/crash_dump.py",
    "python/vibeqc/output/formats/perf.py",
    "python/vibeqc/output/formats/scf_log.py",
    "python/vibeqc/output/formats/structured_log.py",
    "python/vibeqc/output/citations/plain.py",
    "python/vibeqc/output/citations/registry.py",
    "python/vibeqc/solvers/_casscf.py",
    "python/vibeqc/solvers/_selected_ci.py",
    "python/vibeqc/solvers/_dmrg.py",
    "python/vibeqc/solvers/_v2rdm.py",
    "python/vibeqc/symmetry_scf.py",
    "python/vibeqc/periodic_gapw_ot.py",
    "python/vibeqc/molecular_optimize.py",
    "python/vibeqc/scan.py",
    "python/vibeqc/dimer.py",
    "python/vibeqc/conical.py",
    "python/vibeqc/basis_optimization/phonons.py",
)


def _print_call_lines(path: Path) -> list[int]:
    """Line numbers of every real ``print(...)`` call in ``path`` (AST, so
    docstring examples and the substring ``run_fingerprint(`` don't count)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def test_out_writers_do_not_print():
    offenders: dict[str, list[int]] = {}
    for rel in _GUARDED_WRITERS:
        path = _REPO_ROOT / rel
        # A silent rename would neuter the guard, so treat a missing file as a
        # failure rather than a skip -- update the list in the same commit.
        assert path.exists(), (
            f"guarded .out writer moved or renamed: {rel}. Update "
            f"_GUARDED_WRITERS in {Path(__file__).name} to match."
        )
        lines = _print_call_lines(path)
        if lines:
            offenders[rel] = lines
    assert not offenders, (
        "these .out-writing modules must emit through vibeqc.output.write(), "
        "the document layer, or ProgressLogger -- never print() (CLAUDE.md "
        "Sec. 16):\n"
        + "\n".join(f"  {rel}: line(s) {lines}" for rel, lines in offenders.items())
    )


def test_registry_toml_formatter_is_owned_by_output():
    owned = _REPO_ROOT / "python/vibeqc/output/formats/toml.py"
    bypass = _REPO_ROOT / "python/vibeqc/_toml_writer.py"
    assert owned.exists(), "registry TOML formatter moved outside vibeqc.output"
    assert not bypass.exists(), "do not recreate a top-level output writer"


def test_solver_verbose_text_uses_the_ambient_channel(capsys):
    """Verbose correlated solvers retain their diagnostics in ``.out``.

    The tiny diagonal Hamiltonians keep this an output-routing test rather than
    a numerical solver gate; the solver suites own the chemistry assertions.
    """
    import numpy as np

    from vibeqc.output import OutputChannel
    from vibeqc.solvers._casscf import casscf
    from vibeqc.solvers._common import Hamiltonian
    from vibeqc.solvers._dmrg import DMRGOptions, DMRGSolver
    from vibeqc.solvers._selected_ci import (
        SelectedCIOptions,
        solve_selected_ci,
    )
    from vibeqc.solvers._v2rdm import V2RDMOptions, V2RDMSolver

    h1 = np.asarray([[-1.0]])
    h2 = np.zeros((1, 1, 1, 1))
    hamiltonian = Hamiltonian(h1e=h1, h2e=h2, norb=1, nelec=2, ms2=0)
    stream = io.StringIO()

    with OutputChannel.to_stream(stream):
        solve_selected_ci(
            hamiltonian,
            SelectedCIOptions(
                target_size=1,
                max_iter=1,
                do_pt2_correction=False,
                verbose=1,
            ),
        )
        DMRGSolver(
            DMRGOptions(bond_dim_schedule=[2], n_sweeps=1, verbose=1)
        ).solve(hamiltonian)
        V2RDMSolver(V2RDMOptions(verbose=1)).solve(hamiltonian)
        casscf(
            np.diag([-1.0, 0.0]),
            np.zeros((2, 2, 2, 2)),
            n_active_elec=2,
            n_active_orb=1,
            max_macro=1,
            verbose=1,
        )

    assert capsys.readouterr().out == ""
    rendered = stream.getvalue()
    assert "Selected-CI iter" in rendered
    assert "DMRG exact fixed-N solve" in rendered
    assert "v2RDM exact N-representable solve" in rendered
    assert "CASSCF iter" in rendered


def test_symmetry_diagnostics_use_live_progress(capsys):
    """Opt-in standalone symmetry diagnostics retain their stdout contract."""
    import numpy as np

    from vibeqc.symmetry_scf import symmetrize_density, symmetrize_fock

    matrix = np.asarray([[1.0, 2.0], [0.0, 1.0]])
    permutation = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    kwargs = {
        "operations": [object()],
        "P_cache": [permutation],
        "report_asymmetry": True,
    }

    symmetrize_density(matrix, None, None, **kwargs)
    symmetrize_fock(matrix, None, None, **kwargs)

    rendered = capsys.readouterr().out
    assert "[symmetrize_density] ||D - D_sym||" in rendered
    assert "[symmetrize_fock] ||F - F_sym||" in rendered
