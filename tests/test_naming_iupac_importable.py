"""Regression guard: ``vibeqc.naming.iupac_name`` must import and parse.

Background — the failure this pins against
------------------------------------------
Commit ``99e50629`` ("iupac naming: standalone vibeqc_naming package +
runner integration") landed a botched merge into
``python/vibeqc/naming/iupac_name.py``: two copies of the element->``-ide``
suffix map were concatenated (leaving a ``{`` unclosed), and
``_compositional_name_from_formula`` / ``_build_compositional_name`` were
left indented as nested functions instead of module-level symbols.

CI did not catch it: ``pytest --collect-only`` succeeds because no test
imported this module at collection time, and ``import vibeqc`` succeeds
because the user-facing naming path uses the standalone ``vibeqc_naming``
package (via ``naming/report.py``), not this in-tree module. So the module
and its importers (``name_systems.py``) were silently broken.

These tests exercise the two properties that were broken:

* the module actually parses and imports (``SyntaxError`` guard), and
  imports cleanly through the ``iupac_name`` <-> ``organic`` circular
  dependency (``ImportError`` guard);
* the helpers ``name_systems`` reaches into are at *module scope*, not
  nested inside another ``def`` (over-indentation guard).

They deliberately do **not** pin the *values* returned by the
compositional namer — that logic is validated (and, where wrong, fixed)
elsewhere; here we only assert the structural contract so this class of
merge damage cannot regress uncaught.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path


def test_iupac_name_module_parses():
    """The source file parses — a direct guard against the SyntaxError."""
    import vibeqc.naming.iupac_name as m

    src = Path(inspect.getfile(m)).read_text()
    ast.parse(src)  # raises SyntaxError if the merge damage returns


def test_iupac_name_and_name_systems_import():
    """Both the module and its importer load through the circular import."""
    import vibeqc.naming.iupac_name  # noqa: F401
    import vibeqc.naming.name_systems  # noqa: F401


def test_compositional_helpers_are_module_level():
    """The symbols ``name_systems`` imports live at module scope.

    ``name_systems.py`` does ``from vibeqc.naming.iupac_name import
    _compositional_name_from_formula``; if that (or its siblings) is
    nested inside another function again, this fails.
    """
    import vibeqc.naming.iupac_name as m

    for name in (
        "_compositional_name_from_formula",
        "_build_compositional_name",
        "_inorganic_name",
        "_ide_suffix",
        "ide_map_symbol",
    ):
        obj = getattr(m, name)
        assert callable(obj), f"{name} is not a module-level callable"
        # Module scope: defined directly in this module, not a closure.
        assert obj.__qualname__ == name, (
            f"{name} is nested (qualname={obj.__qualname__!r}); it must be "
            "defined at module scope so importers can reach it"
        )

    assert isinstance(m._IDE_SUFFIXES, dict) and m._IDE_SUFFIXES


def test_compositional_namer_runs():
    """The reconstructed namer executes end-to-end without raising.

    Value correctness is asserted elsewhere; here we only require that the
    de-tangled function bodies run.
    """
    import vibeqc.naming.iupac_name as m

    for formula in ("H2O", "CO2", "NaCl", "Fe2O3"):
        out = m._compositional_name_from_formula(formula)
        assert isinstance(out, str) and out


if __name__ == "__main__":  # allow running as a plain script for local checks
    test_iupac_name_module_parses()
    test_iupac_name_and_name_systems_import()
    test_compositional_helpers_are_module_level()
    test_compositional_namer_runs()
    print("all naming-import regression checks passed")
