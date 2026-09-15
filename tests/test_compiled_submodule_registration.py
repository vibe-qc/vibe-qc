"""Regression: dotted imports of the compiled pybind11 submodules
survive a ``sys.modules`` purge-and-reimport cycle.

pybind11's ``def_submodule`` registers ``vibeqc._vibeqc_core.semiempirical``
(plus ``.nddo`` / ``.xtb`` / ``.indo``) in ``sys.modules`` only while the
extension's one-time ``PyInit`` runs. CPython caches single-phase-init
extension modules, so when a test purges ``vibeqc*`` from ``sys.modules``
(the ``tests/basisset_dev/`` isolation pattern), a plain re-import used to
leave the dotted form -- ``from vibeqc._vibeqc_core.semiempirical import
nddo`` -- permanently broken for the rest of the process: ~70 semiempirical
tests failed with ModuleNotFoundError in single-process full-suite runs
while passing in per-file isolation (2026-06-11, see
handovers/HANDOVER_TEST_HEALTH.md). ``vibeqc/__init__.py`` now re-registers the
compiled submodule tree on every package import.
"""

from __future__ import annotations

import importlib
import sys

DOTTED_COMPILED_SUBMODULES = (
    "vibeqc._vibeqc_core.semiempirical",
    "vibeqc._vibeqc_core.semiempirical.nddo",
    "vibeqc._vibeqc_core.semiempirical.xtb",
    "vibeqc._vibeqc_core.semiempirical.indo",
)


def test_fresh_import_registers_dotted_submodules():
    import vibeqc  # noqa: F401

    for name in DOTTED_COMPILED_SUBMODULES:
        assert name in sys.modules, f"{name} not registered in sys.modules"


def test_package_attribute_survives_core_preserving_purge():
    """Re-import must re-bind the ``vibeqc._vibeqc_core`` ATTRIBUTE.

    The ``tests/basisset_dev/`` isolation headers purge ``vibeqc*`` from
    ``sys.modules`` but deliberately KEEP the compiled core's entries
    (they cannot be meaningfully re-imported). On the subsequent
    re-import the cached extension submodule skips the loader step that
    normally setattrs it onto the new package object; from-imports still
    work via the interpreter's ``sys.modules`` fallback, but
    attribute-chain access (``vibeqc._vibeqc_core.compute_eri``, used by
    several tests) raised AttributeError until ``vibeqc/__init__.py``
    started binding the name explicitly (2026-06-11 follow-up fix).
    """
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "vibeqc" or name.startswith("vibeqc.")
    }
    try:
        for name in saved:
            if name == "vibeqc._vibeqc_core" or name.startswith(
                "vibeqc._vibeqc_core."
            ):
                continue
            del sys.modules[name]
        import vibeqc

        core = getattr(vibeqc, "_vibeqc_core", None)
        assert core is not None, (
            "vibeqc._vibeqc_core attribute missing after re-import with "
            "preserved extension sys.modules entries"
        )
        assert hasattr(core, "compute_eri")
        for name in DOTTED_COMPILED_SUBMODULES:
            assert importlib.import_module(name) is not None
    finally:
        for name in list(sys.modules):
            if name == "vibeqc" or name.startswith("vibeqc."):
                del sys.modules[name]
        sys.modules.update(saved)


def test_skala_callback_is_adopted_after_core_preserving_purge():
    """Re-import adopts the immutable callback retained by the native core.

    The external-XC registry is intentionally process-lifetime state, while
    ``vibeqc.skala._REGISTERED_CALLBACKS`` belongs to a purgeable Python
    module.  Re-registering the three immutable names used to raise a
    duplicate-name error during the isolation pattern exercised here.
    """
    import vibeqc

    callback = vibeqc._skala_callback
    core = vibeqc._vibeqc_core
    names = ("skala-1.1", "skala-1.1-rev1", "skala")
    assert all(
        core._external_functional_registration(name)["callback"] is callback
        for name in names
    )

    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "vibeqc" or name.startswith("vibeqc.")
    }
    try:
        for name in saved:
            if name == "vibeqc._vibeqc_core" or name.startswith(
                "vibeqc._vibeqc_core."
            ):
                continue
            del sys.modules[name]

        reimported = importlib.import_module("vibeqc")

        assert reimported._skala_callback is callback
        assert all(
            core._external_functional_registration(name)["callback"]
            is callback
            for name in names
        )
    finally:
        for name in list(sys.modules):
            if name == "vibeqc" or name.startswith("vibeqc."):
                del sys.modules[name]
        sys.modules.update(saved)


def test_dotted_imports_survive_sys_modules_purge():
    """Purge every vibeqc* entry, then re-import via the dotted form.

    Pre-fix this raised ``ModuleNotFoundError: No module named
    'vibeqc._vibeqc_core.semiempirical'; 'vibeqc._vibeqc_core' is not a
    package`` because nothing re-registered the def_submodule names after
    the purge.
    """
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "vibeqc" or name.startswith("vibeqc.")
    }
    try:
        for name in saved:
            del sys.modules[name]
        for name in DOTTED_COMPILED_SUBMODULES:
            assert importlib.import_module(name) is not None
        # The exact import forms used by runner.py and
        # vibeqc/semiempirical/methods/*.py:
        from vibeqc._vibeqc_core.semiempirical import nddo  # noqa: F401
        from vibeqc._vibeqc_core.semiempirical.nddo import (  # noqa: F401
            PM6ParameterSet,
        )
    finally:
        # Discard whatever the re-import created, then restore the exact
        # pre-test module set so this test is itself pollution-free.
        for name in list(sys.modules):
            if name == "vibeqc" or name.startswith("vibeqc."):
                del sys.modules[name]
        sys.modules.update(saved)
