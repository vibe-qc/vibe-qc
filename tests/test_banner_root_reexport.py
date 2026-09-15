"""Package-root access to the provenance API in :mod:`vibeqc.banner`.

The package root re-exports a ``banner()`` FUNCTION, which shadows the
``vibeqc.banner`` SUBMODULE on ``vibeqc``. So ``from vibeqc import
banner`` binds the function, and a caller doing
``getattr(banner, "build_info", None)`` gets ``None`` instead of the
provenance helper -- a silent miss, not an exception. A release-
validation runtime-identity probe fell through that hole and crashed on
a bogus fallback path.

The fix is additive: every public name of ``vibeqc.banner`` is also
re-exported on the package root, so no caller needs the submodule (nor
``importlib.import_module("vibeqc.banner")``, which was the only
reliable access before). ``vibeqc.banner`` must stay the callable
function -- renaming it would break long-standing public API.

Pin the contract:

  1. The provenance API is reachable directly on the package root.
  2. Root-level ``build_info()`` is the same object as the submodule's,
     and returns an equal mapping.
  3. ``vibeqc.banner`` is still the function, still callable, and still
     produces the same string as the submodule's ``banner()``.
  4. ``__all__`` advertises the re-exported names.
"""

from __future__ import annotations

import importlib
import types

import vibeqc

# The module object, reached the one way that was reliable before the fix.
banner_module = importlib.import_module("vibeqc.banner")


# ---------------------------------------------------------------------------
# 1. + 2. The provenance API is reachable from the package root
# ---------------------------------------------------------------------------


def test_build_info_is_reachable_from_package_root():
    """This is the regression: ``vibeqc.build_info`` did not exist."""
    assert hasattr(vibeqc, "build_info")
    assert callable(vibeqc.build_info)


def test_root_build_info_is_the_submodule_function():
    assert vibeqc.build_info is banner_module.build_info


def test_root_build_info_returns_same_mapping_as_submodule():
    root = vibeqc.build_info()
    submodule = banner_module.build_info()
    assert dict(root) == dict(submodule)


def test_sibling_provenance_helpers_are_reachable_from_package_root():
    """Every public name of ``vibeqc.banner`` is on the package root."""
    for name in banner_module.__all__:
        assert hasattr(vibeqc, name), f"vibeqc.{name} is not re-exported"
        assert getattr(vibeqc, name) is getattr(banner_module, name)


def test_getattr_probe_on_root_finds_build_info():
    """The exact shape of the probe that silently failed before the fix."""
    probe = getattr(vibeqc, "build_info", None)
    assert probe is not None
    assert callable(probe)


# ---------------------------------------------------------------------------
# 3. ``vibeqc.banner`` is unchanged -- still the function, not the module
# ---------------------------------------------------------------------------


def test_root_banner_is_still_the_function_not_the_module():
    assert isinstance(vibeqc.banner, types.FunctionType)
    assert not isinstance(vibeqc.banner, types.ModuleType)
    assert vibeqc.banner is banner_module.banner


def test_root_banner_is_still_callable_and_matches_submodule():
    text = vibeqc.banner()
    assert isinstance(text, str)
    assert text == banner_module.banner()
    assert vibeqc.VIBEQC_VERSION in text


def test_root_banner_still_honours_the_width_keyword():
    narrow = vibeqc.banner(width=60)
    assert narrow == banner_module.banner(width=60)


# ---------------------------------------------------------------------------
# 4. ``__all__`` advertises what the root exports
# ---------------------------------------------------------------------------


def test_all_advertises_the_reexported_provenance_names():
    for name in banner_module.__all__:
        assert name in vibeqc.__all__, f"'{name}' missing from vibeqc.__all__"
