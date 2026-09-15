"""Guard: no test module may leave a second copy of a vibeqc module behind.

Several test files load pure-Python vibeqc submodules straight from source, or
shim a namespace package under the name ``vibeqc``, so they keep working in a
checkout with no built C++ extension. Doing either under the *real* module
names is a session-wide hazard, because a module then has two live copies
reachable by two different routes:

* ``import vibeqc.basis_crystal as bc`` binds the ``vibeqc`` package
  **attribute**;
* ``from .basis_crystal import fetch_bredow_basis_sets`` inside ``vibeqc``
  resolves through **sys.modules**.

Those are the same object in a healthy interpreter. When they diverge, a test
that monkeypatches one is silently calling the other. That is not hypothetical:
``tests/basisset_dev/test_ld_penalty_inmemory.py`` registered its copies under
the real names, and for every run where it was collected first, the fail-closed
ECP-provenance guard in ``tests/test_periodic_ecp.py`` (CLAUDE.md § 1) called
the live network fetcher instead of the monkeypatched offline one -- one of its
two tests failed with "DID NOT RAISE", and the other passed vacuously. Fixed in
267cf9522; the sister files that shimmed unconditionally in c8ac0ff7c.

This guard imports every test file that manipulates ``sys.modules`` into a
**fresh interpreter** and checks the real package is intact afterwards. A
subprocess rather than this session, for two reasons:

* ``scripts/test_gate/run_full_suite.py`` executes each test file in its own
  pytest process, so a check of the ambient session would observe nothing and
  pass vacuously in exactly the run that gates a release;
* a file that does leak would poison the session doing the checking.

``test_probe_detects_a_planted_duplicate_module`` and its sibling plant the
historical bug in throwaway files and assert the probe reports them, so this
guard cannot quietly decay into one that always passes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
BASIS_CRYSTAL_SRC = REPO_ROOT / "python" / "vibeqc" / "basis_crystal.py"

# Runs in a fresh interpreter: import the given files, then report every vibeqc
# submodule whose sys.modules entry is not the object its parent package holds.
_PROBE = textwrap.dedent(
    '''
    import importlib.util
    import json
    import sys
    import types


    def duplicated_modules():
        bad = []
        for name, mod in sorted(sys.modules.items()):
            if not name.startswith("vibeqc.") or mod is None:
                continue
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            attr = getattr(parent, child, None) if parent is not None else None
            # A re-exported function or class shadowing its own submodule name
            # (``vibeqc.banner`` is the banner() function, not the module) is
            # not a duplicate -- only a second *module* object is.
            if not isinstance(attr, types.ModuleType):
                continue
            if attr is not mod:
                bad.append(name)
        return bad


    def copied_objects(module):
        """Names in *module* that claim a vibeqc home but are not what it exports.

        Catches the variant that restores sys.modules afterwards and so leaves
        no duplicate registered, yet still ran its own imports against a
        throwaway copy: the class the test then asserts on is not the class the
        rest of the process uses, and an isinstance check against the genuine
        one silently fails.
        """
        bad = []
        for attr_name, value in sorted(vars(module).items()):
            if attr_name.startswith("__"):
                continue
            home = getattr(value, "__module__", None)
            own_name = getattr(value, "__name__", None)
            # __name__ excludes instances, whose __module__ resolves through
            # their class; only classes and functions are checked.
            if not isinstance(home, str) or not isinstance(own_name, str):
                continue
            if home != "vibeqc" and not home.startswith("vibeqc."):
                continue
            real = sys.modules.get(home)
            if real is None:
                continue
            genuine = getattr(real, own_name, None)
            if genuine is not None and genuine is not value:
                bad.append(f"{attr_name} (a copy of {home}.{own_name})")
        return bad


    paths = json.loads(sys.argv[1])

    # Import the real package first, the way any ordinary pytest session has it
    # by the time these files are collected. basis_crystal explicitly: the
    # attribute has to exist for a duplicate registered under its name to be
    # detectable, and it is the module the historical leak actually hit.
    import vibeqc
    import vibeqc.basis_crystal

    imported = []
    import_errors = []
    copies = []
    for index, path in enumerate(paths):
        spec = importlib.util.spec_from_file_location(f"_leakprobe_{index}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException as exc:  # noqa: BLE001
            # A module-level ``pytest.skip(allow_module_level=True)`` raises
            # outside collection, and pytest outcomes derive from BaseException.
            # Keep going: a file that mutates sys.modules and *then* raises is
            # precisely the case whose cleanup never runs, so the check below
            # still has to see the resulting state.
            import_errors.append(
                {"path": path, "error": f"{type(exc).__name__}: {exc}"[:200]}
            )
        else:
            imported.append(path)
            for finding in copied_objects(module):
                copies.append({"path": path, "object": finding})

    print(json.dumps({
        "duplicates": duplicated_modules(),
        "vibeqc_file": getattr(sys.modules.get("vibeqc"), "__file__", None),
        "imported": imported,
        "import_errors": import_errors,
        "copies": copies,
    }))
    '''
)


def _candidate_files() -> list[Path]:
    """Test files that touch ``sys.modules`` and mention vibeqc.

    Content-based rather than a hardcoded list, so a new file that adopts the
    pattern anywhere under tests/ is covered the day it lands.
    """
    found = []
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "sys.modules" in text and "vibeqc" in text:
            found.append(path)
    return found


def _run_probe(paths: list[Path]) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps([str(p) for p in paths])],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"probe interpreter failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    # Imported modules may print or warn; the report is the last stdout line.
    report = json.loads(proc.stdout.strip().splitlines()[-1])

    # Repo-relative paths keep the diagnostics readable (and keep absolute home
    # paths out of CI logs).
    def shorten(path: str) -> str:
        prefix = f"{REPO_ROOT}/"
        return path[len(prefix):] if path.startswith(prefix) else path

    report["imported"] = [shorten(p) for p in report["imported"]]
    for entry in report["import_errors"] + report["copies"]:
        entry["path"] = shorten(entry["path"])
    return report


def test_no_test_module_registers_a_duplicate_vibeqc_module():
    """Importing every sys.modules-touching test file leaves vibeqc intact."""
    candidates = _candidate_files()
    # A discovery bug that returned nothing would make this pass while checking
    # nothing at all. There were 15 such files when this guard was written.
    assert len(candidates) >= 10, (
        "candidate discovery found only "
        f"{[str(p.relative_to(REPO_ROOT)) for p in candidates]} -- expected the "
        "test files that shim or reload vibeqc modules; fix the discovery "
        "rather than lowering this floor"
    )

    report = _run_probe(candidates)

    # Same reason as the floor above: if these files stopped importing, the
    # check below would pass while exercising nothing. One module-level skip
    # (test_basis_library_load.py, gated on VIBEQC_RUN_HEAVY_TESTS) is normal.
    assert len(report["imported"]) >= 10, (
        f"only {len(report['imported'])} of {len(candidates)} candidate files "
        f"imported, so this guard checked almost nothing: {report['import_errors']}"
    )

    assert report["duplicates"] == [], (
        "these vibeqc submodules have a second copy registered in sys.modules "
        f"after importing the test files above: {report['duplicates']}. A test "
        "that monkeypatches one copy will silently be calling the other. Load "
        "private copies under a private root package name (see "
        "tests/basisset_dev/test_ld_penalty_inmemory.py), or prefer the real "
        "package and shim only when `import vibeqc` fails (see "
        "tests/basisset_dev/test_crystal_ecp_parser.py)"
    )
    assert report["vibeqc_file"] is not None, (
        "a bare namespace shim was left registered under `vibeqc` itself, so "
        "every later-collected module would import the shim instead of the "
        "real package"
    )
    assert report["copies"] == [], (
        "these test modules hold copies of real vibeqc objects rather than the "
        f"objects vibeqc exports: {report['copies']}. Assertions against a copy "
        "do not constrain the shipped class, and isinstance against the genuine "
        "one fails. Prefer the real package and shim only when `import vibeqc` "
        "fails (see tests/basisset_dev/test_crystal_ecp_parser.py)"
    )


def test_probe_detects_a_planted_duplicate_module(tmp_path):
    """Positive control: the historical bug, planted, must be reported."""
    planted = tmp_path / "test_plants_a_duplicate.py"
    planted.write_text(
        textwrap.dedent(
            f'''
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location(
                "vibeqc.basis_crystal", {str(BASIS_CRYSTAL_SRC)!r}
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["vibeqc.basis_crystal"] = module
            spec.loader.exec_module(module)
            '''
        )
    )

    report = _run_probe([planted])

    assert report["duplicates"] == ["vibeqc.basis_crystal"], (
        "the probe did not notice a second copy of vibeqc.basis_crystal "
        f"registered under the real name; it reported {report['duplicates']}"
    )


def test_probe_detects_a_planted_object_copy(tmp_path):
    """Positive control: a file holding a copy of a real class must be reported.

    This is the shape that restores sys.modules afterwards, so the duplicate
    check above sees nothing wrong; only the object identity gives it away.
    """
    planted = tmp_path / "test_plants_an_object_copy.py"
    planted.write_text(
        textwrap.dedent(
            f'''
            import importlib.util
            import sys

            _real = sys.modules.get("vibeqc.basis_crystal")
            spec = importlib.util.spec_from_file_location(
                "vibeqc.basis_crystal", {str(BASIS_CRYSTAL_SRC)!r}
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["vibeqc.basis_crystal"] = module
            spec.loader.exec_module(module)
            CrystalECP = module.CrystalECP          # the copy, kept as a global
            sys.modules["vibeqc.basis_crystal"] = _real   # tidy restore
            '''
        )
    )

    report = _run_probe([planted])

    assert report["duplicates"] == [], (
        "this plant restores sys.modules, so the duplicate check should be "
        f"clean; it reported {report['duplicates']}"
    )
    assert [finding["object"] for finding in report["copies"]] == [
        "CrystalECP (a copy of vibeqc.basis_crystal.CrystalECP)"
    ], f"the probe did not notice the copied class; it reported {report['copies']}"


def test_probe_detects_a_planted_namespace_shim(tmp_path):
    """Positive control: a bare shim left under `vibeqc` must be reported."""
    planted = tmp_path / "test_plants_a_shim.py"
    planted.write_text(
        textwrap.dedent(
            '''
            import sys
            import types

            sys.modules["vibeqc"] = types.ModuleType("vibeqc")
            '''
        )
    )

    report = _run_probe([planted])

    assert report["vibeqc_file"] is None, (
        "the probe did not notice a bare namespace shim left registered under "
        "`vibeqc`"
    )
