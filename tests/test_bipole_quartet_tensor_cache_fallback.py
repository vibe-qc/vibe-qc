"""Pure-Python fallback coverage for the quartet tensor cache.

``build_quartet_tensor_cache`` probes ``vibeqc._vibeqc_core`` for native
interaction-tensor kernels and drops to the pure-Python implementations in
:mod:`vibeqc.bipole_multipole` when they are absent.  That fallback branch
had no test of any kind, and it shipped unable to execute.

``7e94c7f18`` placed a function-local ``from .bipole_quartet_far_field
import quartet_effective_screening_parameter`` *inside* the
``if _use_cpp_tensor:`` branch.  Python binds a name imported anywhere in a
function body as function-local for the **entire** body, so that import
shadowed the working module-level binding of the same name.  Whenever the
probe failed and the ``else:`` branch ran, the call raised::

    UnboundLocalError: cannot access local variable
    'quartet_effective_screening_parameter' where it is not associated
    with a value

on every ``ewald_omega > 0`` build.  It was reachable in practice: any
``_vibeqc_core`` built before 1a26e6954 lacks the native symbols, so the
fallback is the live path there.  The sole call site
(``bipole_far_field_infrastructure``) wraps the builder in
``except Exception: pass`` -- "cache is an optimisation" -- so the error was
swallowed and affected runs silently lost the cache and re-assembled every
quartet tensor per SCF iteration.

The crash itself was fixed incidentally by ``3f33caee9``, which hoisted that
import above its use while fixing an unrelated ``mu_eff`` cache-key
collision.  Nothing pinned the behaviour, so these tests exist to keep it
fixed, and the AST guard below pins the *invariant* that was violated rather
than just the one symbol that violated it.

Also pinned: the native probe must cover every kernel the C++ branch calls.
Probing one while calling two would satisfy the probe and then raise
``ImportError`` mid-loop.  Both bindings landed together in 1a26e6954, so no
shipped build had that asymmetry -- this guards the rule, not a historical
build.

Provenance
----------
Pisani-Dovesi-Roetti (1988), Ch. II.4c, is the periodic quartet-
expansion source. The cache and screening composition tested here are
implementation prototypes.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

import vibeqc._vibeqc_core as _core
from vibeqc.bipole_multipole import (
    multipole_interaction_tensor,
    sr_multipole_interaction_tensor,
)
from vibeqc.bipole_quartet_far_field import (
    QuartetBipolarDispatch,
    quartet_effective_screening_parameter,
)
from vibeqc.bipole_quartet_tensor_cache import build_quartet_tensor_cache

_CPP_TENSOR_SYMBOLS = (
    "multipole_interaction_tensor",
    "multipole_erfc_interaction_tensor",
)

# Two product-distribution centres, 3 bohr apart along x, chosen so the
# prototype tensor is comfortably non-singular. The differing widths make
# the implementation-specific mu_eff composition non-trivial.
_CENTRE_BRA = np.array([0.0, 0.0, 0.0])
_CENTRE_KET = np.array([3.0, 0.0, 0.0])
_R_SEP = _CENTRE_KET - _CENTRE_BRA

_GAMMA_BRA = 0.5
_GAMMA_KET = 0.7
_L_ORDER = 2
_EWALD_OMEGA = 0.3


def _mu_eff(ewald_omega: float) -> float:
    """Screening parameter the cache keys on, or 0.0 for bare Coulomb."""
    if ewald_omega <= 0:
        return 0.0
    return quartet_effective_screening_parameter(
        _GAMMA_BRA, _GAMMA_KET, ewald_omega,
    )


class _StubMomentBuffer:
    """Minimal ``get_center`` provider for two shell pairs."""

    def get_center(self, s1: int, s2: int, cell: int):
        if (s1, s2, cell) == (0, 0, 0):
            return _CENTRE_BRA
        if (s1, s2, cell) == (1, 1, 0):
            return _CENTRE_KET
        return None


def _make_dispatch() -> QuartetBipolarDispatch:
    return QuartetBipolarDispatch(
        bra_pairs=[(0, 0, 0)],
        ket_pairs=[(1, 1, 0)],
        truncation_orders=[_L_ORDER],
        bra_widths=[_GAMMA_BRA],
        ket_widths=[_GAMMA_KET],
    )


def _force_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every native tensor symbol so the probe selects the fallback."""
    for name in _CPP_TENSOR_SYMBOLS:
        monkeypatch.delattr(_core, name, raising=False)


def _build(ewald_omega: float) -> np.ndarray:
    """Build the cache and return its single tensor, via the public lookup."""
    cache = build_quartet_tensor_cache(
        _StubMomentBuffer(), _make_dispatch(), ewald_omega=ewald_omega,
    )
    assert len(cache) == 1, f"expected one cached tensor, got {len(cache)}"
    tensor = cache.get(_R_SEP, _L_ORDER, _mu_eff(ewald_omega))
    assert tensor is not None, (
        "cache.get missed the entry build_quartet_tensor_cache just stored -- "
        "the store key and the lookup key disagree"
    )
    return tensor


def test_fallback_bare_coulomb_matches_pure_python_reference(monkeypatch):
    """omega == 0 fallback returns the bare pure-Python Coulomb tensor."""
    _force_python_fallback(monkeypatch)

    expected = multipole_interaction_tensor(_L_ORDER, _L_ORDER, _R_SEP)
    np.testing.assert_allclose(_build(0.0), expected, rtol=0, atol=0)


def test_fallback_ewald_screened_matches_pure_python_reference(monkeypatch):
    """omega > 0 fallback returns the erfc-screened pure-Python tensor.

    Direct regression guard for the ``UnboundLocalError``: before
    ``3f33caee9`` this build raised instead of returning a tensor.
    """
    _force_python_fallback(monkeypatch)

    expected = sr_multipole_interaction_tensor(
        _L_ORDER, _L_ORDER, _R_SEP, _mu_eff(_EWALD_OMEGA),
    )
    np.testing.assert_allclose(_build(_EWALD_OMEGA), expected, rtol=0, atol=0)


def test_fallback_screening_differs_from_bare_tensor(monkeypatch):
    """The two fallback sub-branches are genuinely distinct code paths.

    Guards against a future 'fix' that repairs a crash by quietly routing the
    screened path through the bare tensor.
    """
    _force_python_fallback(monkeypatch)

    assert not np.allclose(_build(0.0), _build(_EWALD_OMEGA)), (
        "erfc-screened tensor must not equal the bare Coulomb tensor"
    )


@pytest.mark.parametrize("present", _CPP_TENSOR_SYMBOLS)
def test_partial_native_symbols_fall_back_cleanly(monkeypatch, present):
    """Half a native kernel set must select the fallback, not crash.

    The probe must cover every kernel the C++ path calls, not just the first.
    Parametrised over both symbols so the invariant holds whichever one goes
    missing.  No shipped ``_vibeqc_core`` had this asymmetry (both bindings
    landed in 1a26e6954); this pins the rule for the next kernel added.
    """
    def _poisoned(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError(
            "native kernel called despite an incomplete symbol set"
        )

    for name in _CPP_TENSOR_SYMBOLS:
        if name == present:
            monkeypatch.setattr(_core, name, _poisoned, raising=False)
        else:
            monkeypatch.delattr(_core, name, raising=False)

    expected = sr_multipole_interaction_tensor(
        _L_ORDER, _L_ORDER, _R_SEP, _mu_eff(_EWALD_OMEGA),
    )
    np.testing.assert_allclose(_build(_EWALD_OMEGA), expected, rtol=0, atol=0)


@pytest.mark.parametrize("ewald_omega", [0.0, _EWALD_OMEGA])
def test_native_kernels_are_dispatched_with_the_right_arguments(
    monkeypatch, ewald_omega,
):
    """Stub both native kernels and pin which one is called, with what.

    Runs everywhere, including on an extension predating the kernels, so the
    C++ dispatch stays covered on machines where the numerical parity check
    below can only skip.  ``mu_eff`` is derived once above the C++/Python
    branch, so this pins that the screened kernel still receives it.
    """
    calls: dict[str, tuple] = {}
    n_sph = (_L_ORDER + 1) ** 2
    stub_result = np.zeros((n_sph, n_sph))

    def _record(name):
        def _stub(*args):
            calls[name] = args
            return stub_result
        return _stub

    for name in _CPP_TENSOR_SYMBOLS:
        monkeypatch.setattr(_core, name, _record(name), raising=False)

    build_quartet_tensor_cache(
        _StubMomentBuffer(), _make_dispatch(), ewald_omega=ewald_omega,
    )

    if ewald_omega > 0:
        assert set(calls) == {"multipole_erfc_interaction_tensor"}
        assert calls["multipole_erfc_interaction_tensor"] == pytest.approx(
            (_L_ORDER, _L_ORDER, *_R_SEP, _mu_eff(ewald_omega))
        )
    else:
        assert set(calls) == {"multipole_interaction_tensor"}
        assert calls["multipole_interaction_tensor"] == pytest.approx(
            (_L_ORDER, _L_ORDER, *_R_SEP)
        )


@pytest.mark.parametrize("ewald_omega", [0.0, _EWALD_OMEGA])
def test_native_path_agrees_with_python_fallback(monkeypatch, ewald_omega):
    """When both kernels exist, C++ and pure-Python agree numerically."""
    missing = [n for n in _CPP_TENSOR_SYMBOLS if not hasattr(_core, n)]
    if missing:
        pytest.skip(f"_vibeqc_core lacks native tensor kernels: {missing}")

    native = _build(ewald_omega)
    _force_python_fallback(monkeypatch)
    fallback = _build(ewald_omega)

    np.testing.assert_allclose(native, fallback, rtol=1e-10, atol=1e-12)


def test_no_function_local_import_shadows_a_module_level_name():
    """Static guard on the invariant the original defect violated.

    Any name imported inside ``build_quartet_tensor_cache`` becomes
    function-local for the whole body.  If such a name also exists at module
    level, every branch that does not execute the local import raises
    ``UnboundLocalError`` -- which is exactly how the fallback shipped broken.
    The native probe therefore binds its symbols under private ``_cpp_*``
    aliases.

    This is the guard that would have caught 7e94c7f18 directly, rather than
    only through the one screened-path symptom it happened to produce.
    """
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "python" / "vibeqc" / "bipole_quartet_tensor_cache.py"
    )
    tree = ast.parse(path.read_text())

    module_level = {
        alias.asname or alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    func = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_quartet_tensor_cache"
    )
    shadowed = {
        alias.asname or alias.name.split(".")[0]
        for node in ast.walk(func)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } & module_level

    assert not shadowed, (
        "function-local imports in build_quartet_tensor_cache shadow "
        f"module-level name(s) {sorted(shadowed)}; bind them under a "
        "private alias or hoist them to module level"
    )
