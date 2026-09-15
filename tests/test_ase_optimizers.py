"""ASE optimizer availability registry + SciPy-compat shim.

Regression coverage for the glycine optimizer SI matrix failures
(rp167-glyopt-*, 2026-07-01):

* ASE 3.22.1 + SciPy >= 1.14: ``from ase.optimize import
  BFGSLineSearch`` died with ``ImportError: cannot import name
  'cumtrapz' from 'scipy.integrate'`` -- after queue dispatch.
* ``ase.optimize.fire2`` / ``ase.optimize.rfo`` do not exist on ASE
  3.22.x; runners generated from current-ASE docs failed with a bare
  ModuleNotFoundError mid-queue instead of at generation time.
"""

from __future__ import annotations

import scipy.integrate

import pytest

ase = pytest.importorskip("ase")

from vibeqc.ase_optimizers import (  # noqa: E402
    ASE_OPTIMIZERS,
    ase_optimizer_availability,
    available_ase_optimizers,
    ensure_ase_scipy_compat,
    format_ase_optimizer_report,
    normalize_ase_optimizer_name,
    resolve_ase_optimizer,
)


# ---------------------------------------------------------------------------
# ensure_ase_scipy_compat
# ---------------------------------------------------------------------------


def test_scipy_compat_restores_cumtrapz(monkeypatch: pytest.MonkeyPatch) -> None:
    """The removed scipy aliases come back as their modern equivalents."""
    monkeypatch.delattr(scipy.integrate, "cumtrapz", raising=False)
    monkeypatch.delattr(scipy.integrate, "trapz", raising=False)
    monkeypatch.delattr(scipy.integrate, "simps", raising=False)

    ensure_ase_scipy_compat()

    assert scipy.integrate.cumtrapz is scipy.integrate.cumulative_trapezoid
    assert scipy.integrate.trapz is scipy.integrate.trapezoid
    assert scipy.integrate.simps is scipy.integrate.simpson


def test_scipy_compat_is_idempotent_and_preserves_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scipy that still ships the old names is left untouched."""
    sentinel = object()
    monkeypatch.setattr(scipy.integrate, "cumtrapz", sentinel, raising=False)

    ensure_ase_scipy_compat()
    ensure_ase_scipy_compat()

    assert scipy.integrate.cumtrapz is sentinel


def test_bfgs_linesearch_imports_after_compat() -> None:
    """The exact import that failed on compute-cluster works once the shim ran.

    On new ASE the import works regardless; on old ASE it works only
    because of the shim. Either way this is the end-to-end contract.
    """
    ensure_ase_scipy_compat()
    from ase.optimize import BFGSLineSearch  # noqa: F401


# ---------------------------------------------------------------------------
# name normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("bfgs", "bfgs"),
        ("ase-bfgs", "bfgs"),
        ("ASE-LBFGS", "lbfgs"),
        ("BFGSLineSearch", "bfgs-linesearch"),
        ("ase-bfgs-linesearch", "bfgs-linesearch"),
        ("bfgs_linesearch", "bfgs-linesearch"),
        ("QuasiNewton", "bfgs-linesearch"),
        ("oldqn", "goodoldquasinewton"),
        ("ase-goodoldquasinewton", "goodoldquasinewton"),
        ("fire2", "fire2"),
        ("ode12r", "ode12r"),
    ],
)
def test_normalize_names(raw: str, key: str) -> None:
    assert normalize_ase_optimizer_name(raw) == key


def test_unknown_name_lists_known_keys() -> None:
    with pytest.raises(ValueError, match="Unknown ASE optimizer"):
        normalize_ase_optimizer_name("simplex")
    with pytest.raises(ValueError, match="bfgs-linesearch"):
        normalize_ase_optimizer_name("simplex")


# ---------------------------------------------------------------------------
# availability probing + resolution
# ---------------------------------------------------------------------------


def test_availability_covers_whole_registry() -> None:
    report = ase_optimizer_availability()
    assert set(report) == set(ASE_OPTIMIZERS)
    for avail in report.values():
        assert avail.available == (avail.location is not None)
        assert avail.available == (avail.reason is None)


def test_core_optimizers_resolve_on_any_supported_ase() -> None:
    """The optimizers vibe-qc itself relies on exist on every ASE >= 3.22."""
    for name in ("bfgs", "lbfgs", "fire", "bfgs-linesearch", "gpmin", "mdmin"):
        cls = resolve_ase_optimizer(name)
        assert isinstance(cls, type), name
        assert name.replace("-linesearch", "") in cls.__name__.lower()


def test_resolve_unavailable_raises_with_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry entry the installed ASE lacks fails early and loudly."""
    import importlib

    real_import_module = importlib.import_module

    def fake_import_module(name: str, *args, **kwargs):
        if name in ("ase.optimize.rfo",):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        "vibeqc.ase_optimizers.importlib.import_module", fake_import_module
    )

    with pytest.raises(ImportError) as excinfo:
        resolve_ase_optimizer("ase-rfo")
    message = str(excinfo.value)
    assert "not available in the installed ASE" in message
    # The failure carries the availability report so batch generators
    # can see what IS usable on this deployment.
    assert "optimizer availability" in message
    assert "bfgs" in message


def test_available_ase_optimizers_is_sorted_subset() -> None:
    names = available_ase_optimizers()
    assert names == sorted(names)
    assert set(names) <= set(ASE_OPTIMIZERS)
    assert "bfgs" in names


def test_format_report_mentions_ase_version() -> None:
    report = format_ase_optimizer_report()
    assert ase.__version__ in report
    assert "bfgs-linesearch" in report
