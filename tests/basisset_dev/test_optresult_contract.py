"""Pin the ``OptResult`` data contract shared by vibe-qc and vibe-basis.

Two ``OptResult`` dataclasses exist, and **that is deliberate**:

* :class:`vibeqc.basis_optimization.drivers.OptResult` -- returned by
  vibe-qc's own optimisers (``optimize_scipy``, ``optimize_minuit``,
  and ``optimize_bdiis``, the analytic-gradient BDIIS driver).
* :class:`vibe_basis.optimize.OptResult` -- returned by the driver
  package's optimisers (adds NLopt BOBYQA).

The obvious cleanup -- delete one, import the other -- is wrong in
**both** directions:

* vibe-basis importing vibe-qc is forbidden outright; it would drag a
  native build and FFTW3's GPL onto a pure-Python driver
  (``vibe-basis/tests/test_no_vibeqc_dependency.py``).
* vibe-qc importing vibe-basis *here* would make the external-program
  driver package a hard requirement for ``import
  vibeqc.basis_optimization`` -- which today needs only vibe-qc itself.
  ``basis_optimization/__init__.py`` imports no third-party package,
  and ``bdiis.py`` (pure vibe-qc, native analytic gradients, no external
  program anywhere near it) returns an ``OptResult`` at module level.
  vibe-basis is declared in vibe-qc's ``[basisopt]`` **extra**, not its
  core dependencies, so that import would break a plain
  ``pip install vibe-qc``.

So the duplication stays, and this test is what makes it safe: the two
classes are a shared *structural* contract between independently
installable packages, and recipe code (``recipes/production.py`` uses
the vibe-basis one, ``bdiis.py`` the vibe-qc one) must be able to hand
either to the same consumer. If someone adds a field to one, this test
fails and forces the decision to be made on purpose.

Rationale + the alternative (promote vibe-basis to a core vibe-qc
dependency, collapsing the two) are in
``handovers/HANDOVER_BASISOPT_COHESIVE_ROADMAP.md`` section 3.6.
"""

from __future__ import annotations

import dataclasses

import pytest

from vibeqc.basis_optimization.drivers import OptResult as VibeQcOptResult

vibe_basis_optimize = pytest.importorskip(
    "vibe_basis.optimize",
    reason="vibe-basis is an optional [basisopt] extra; the contract is only "
    "checkable when it is installed",
)
VibeBasisOptResult = vibe_basis_optimize.OptResult


def _field_map(cls: type) -> dict[str, str]:
    """Field name -> type annotation, as written.

    Compared as source text rather than resolved types: the two packages
    annotate independently, and what matters for the contract is that a
    consumer reading ``result.x`` gets the same kind of thing.
    """
    return {f.name: str(f.type) for f in dataclasses.fields(cls)}


def test_both_are_dataclasses() -> None:
    assert dataclasses.is_dataclass(VibeQcOptResult)
    assert dataclasses.is_dataclass(VibeBasisOptResult)


def test_field_names_and_order_match() -> None:
    """Same fields, same order -- positional construction stays compatible."""
    vq = [f.name for f in dataclasses.fields(VibeQcOptResult)]
    vb = [f.name for f in dataclasses.fields(VibeBasisOptResult)]
    assert vq == vb, (
        "OptResult field sets have diverged between vibe-qc and vibe-basis.\n"
        f"  vibeqc.basis_optimization.drivers: {vq}\n"
        f"  vibe_basis.optimize:               {vb}\n"
        "Add the field to both, or read this module's docstring and decide "
        "deliberately which package owns the contract."
    )


def test_field_annotations_match() -> None:
    vq, vb = _field_map(VibeQcOptResult), _field_map(VibeBasisOptResult)
    mismatched = {k: (vq[k], vb[k]) for k in vq if vq[k] != vb[k]}
    assert not mismatched, (
        "OptResult field types have diverged (name: (vibeqc, vibe_basis)):\n"
        f"  {mismatched}"
    )


def test_the_documented_field_set() -> None:
    """Pin the contract itself, so *both* drifting together still fails.

    Without this, a field added to both packages in the same commit would
    slip past the comparison tests above.
    """
    expected = [
        "x",             # best-found parameter vector (optimizer space)
        "fun",           # objective value at x
        "success",       # driver convergence flag
        "message",       # driver-specific status string
        "n_evaluations",  # objective evaluations performed
        "wall_seconds",  # wall-clock time of the optimisation
        "driver",        # "nlopt" | "scipy" | "minuit" | "bdiis"
        "history",       # (x_k, f_k) pairs in evaluation order
        "raw",           # the driver's native result object
    ]
    assert [f.name for f in dataclasses.fields(VibeQcOptResult)] == expected


def test_instances_are_structurally_interchangeable() -> None:
    """A consumer reading the common fields must not care which it got."""
    import numpy as np

    kwargs = dict(
        x=np.array([1.0, 2.0]),
        fun=-1.5,
        success=True,
        message="converged",
        n_evaluations=7,
        wall_seconds=0.25,
        driver="scipy",
    )
    a, b = VibeQcOptResult(**kwargs), VibeBasisOptResult(**kwargs)

    for name in (f.name for f in dataclasses.fields(VibeQcOptResult)):
        got_a, got_b = getattr(a, name), getattr(b, name)
        if isinstance(got_a, np.ndarray):
            assert np.array_equal(got_a, got_b)
        else:
            assert got_a == got_b, f"field {name!r} differs: {got_a!r} vs {got_b!r}"
