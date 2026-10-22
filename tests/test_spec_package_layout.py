"""Install layout of the specification model behind ``vqfetch``.

The dataclasses shared by the regression suite and the external structure
fetcher live in the top-level ``vibeqc_spec`` package, which the wheel ships.
They used to live only in ``examples/regression/core/spec.py``, and the wheel
installed that one file under ``examples/regression/core``. An editable
install's finder then mapped the package ``examples.regression.core`` onto
that one-file directory in site-packages, ahead of the repository root on
``sys.path``, so every regression runner import such as
``examples.regression.core.runner_pyscf`` raised ModuleNotFoundError.

These checks pin both halves: the source package resolves to the
repository, and the fetch clients and the SPEC modules they emit need no
``examples`` tree, as in an installed wheel.
"""
from __future__ import annotations

import ast
import importlib.metadata
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = REPO_ROOT / "examples" / "regression" / "core"

# Raised for any ``examples`` import, so the child process behaves as an
# installed wheel without the repository's examples tree.
_BLOCK_EXAMPLES = """\
import sys

class _NoExamples:
    def find_spec(self, name, path=None, target=None):
        if name == "examples" or name.startswith("examples."):
            raise ImportError(f"examples tree unavailable: {name}")
        return None

sys.meta_path.insert(0, _NoExamples())
"""


def _run_isolated(code: str, cwd: Path) -> str:
    """Run ``code`` in ``python -I`` from ``cwd``; return its stdout."""
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_regression_core_resolves_to_source_tree(tmp_path):
    """``examples.regression.core`` and its runners come from the checkout."""
    out = _run_isolated(textwrap.dedent(f"""\
        import importlib.util, sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        core = importlib.util.find_spec("examples.regression.core")
        print(list(core.submodule_search_locations))
        import examples.regression.core.runner_pyscf as runner
        print(runner.__file__)
        """), cwd=tmp_path)
    locations, runner_file = out.strip().splitlines()
    assert locations == repr([str(CORE_DIR)])
    assert Path(runner_file) == CORE_DIR / "runner_pyscf.py"


def test_distribution_installs_nothing_under_examples():
    """The wheel's RECORD has no ``examples/`` path to shadow the source."""
    try:
        files = importlib.metadata.files("vibe-qc")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("vibe-qc is not installed as a distribution")
    assert files is not None
    shipped = [str(f) for f in files if f.parts and f.parts[0] == "examples"]
    assert shipped == []


def test_wheel_ships_spec_package():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    packages = pyproject["tool"]["scikit-build"]["wheel"]["packages"]
    assert "python/vibeqc_spec" in packages


def test_examples_spec_reexports_the_same_classes():
    import vibeqc_spec
    from examples.regression.core import spec as examples_spec

    assert examples_spec.__all__ == vibeqc_spec.__all__
    for name in vibeqc_spec.__all__:
        assert getattr(examples_spec, name) is getattr(vibeqc_spec, name)


def test_runtime_tree_does_not_import_examples():
    """Nothing the wheel ships imports from the ``examples`` package."""
    offenders = []
    for root in ("python/vibeqc", "python/vibeqc_spec"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if name == "examples" or name.startswith("examples."):
                        rel = path.relative_to(REPO_ROOT)
                        offenders.append(f"{rel}:{node.lineno}: {name}")
    assert offenders == []


def test_fetch_clients_and_emitted_specs_need_no_examples_tree(tmp_path):
    """The fetcher finds its spec model as it would in an installed wheel."""
    out = _run_isolated(_BLOCK_EXAMPLES + textwrap.dedent(f"""\
        import importlib.util
        from pathlib import Path

        import vibeqc.fetch
        from vibeqc.fetch import cli, emit_input
        from vibeqc.fetch.client_cod import PeriodicSpec
        from vibeqc.fetch.emit_spec import emit_spec_module
        from vibeqc_spec import AtomCart, AtomFrac, MoleculeSpec, Provenance

        assert PeriodicSpec.__module__ == "vibeqc_spec", PeriodicSpec.__module__
        assert cli.MoleculeSpec is MoleculeSpec
        assert emit_input.PeriodicSpec is PeriodicSpec
        assert not any(m == "examples" or m.startswith("examples.")
                       for m in sys.modules)

        prov = Provenance(
            source_db="manual", source_id="layout-test", source_url="",
            original_reference="", license="CC0",
            fetched_at="2026-10-20T00:00:00+00:00", fetcher_version="0.1.0",
        )
        specs = [
            PeriodicSpec(
                id="layout_ne_sc", family="simple-cubic",
                lattice_ang=((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0)),
                space_group="Pm-3m",
                atoms=(AtomFrac(symbol="Ne", z=10, frac=(0.0, 0.0, 0.0)),),
                provenance=prov,
            ),
            MoleculeSpec(
                id="layout_h2",
                atoms=(AtomCart(symbol="H", z=1, xyz_ang=(0.0, 0.0, 0.0)),
                       AtomCart(symbol="H", z=1, xyz_ang=(0.0, 0.0, 0.74))),
                provenance=prov,
            ),
        ]
        out_dir = Path({str(tmp_path)!r})
        for spec in specs:
            path = emit_spec_module(spec, out_dir)
            loader_spec = importlib.util.spec_from_file_location(spec.id, path)
            module = importlib.util.module_from_spec(loader_spec)
            loader_spec.loader.exec_module(module)
            assert module.SPEC == spec, (module.SPEC, spec)
            print(type(module.SPEC).__module__, type(module.SPEC).__name__)
        """), cwd=tmp_path)
    assert out.split() == [
        "vibeqc_spec", "PeriodicSpec", "vibeqc_spec", "MoleculeSpec",
    ]
