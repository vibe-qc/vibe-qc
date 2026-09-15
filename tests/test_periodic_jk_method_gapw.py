"""GPW + GAPW enum-value sanity tests (gapw chat, M1b).

The v0.10.x periodic-acceleration program introduces two new
:class:`vibeqc.periodic_jk_method.PeriodicJKMethod` values — ``GPW``
and ``GAPW`` — as the entry points for the Gaussian + plane-wave and
Gaussian-augmented-plane-wave J builders (Lippert & Hutter 1999, CP2K
GAPW). The driver implementation lands at M2 / M3; this milestone
(M1b) just wires the enum + dispatch surface so:

* user input ``jk_method="gpw"`` / ``"gapw"`` is recognised by the
  string-coercing branch in :func:`pick_jk_method` (it returns the
  concrete enum value unchanged rather than raising "unknown method"),
* :func:`validate_jk_method` raises ``NotImplementedError`` pointing
  the user at the v0.10.x roadmap entry rather than silently dropping
  through into a half-wired code path (CLAUDE.md § 7),
* the ``AUTO`` heuristic is **unchanged** — GPW / GAPW stay opt-in
  through M4 per the design doc § 9 decision 4,
* :func:`describe_jk_method` returns a non-empty label so banner /
  output logs identify the route.

See [docs/design_periodic_gapw.md](../docs/design_periodic_gapw.md)
for the full design + decisions log.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc.periodic_jk_method import (
    PeriodicJKMethod,
    describe_jk_method,
    pick_jk_method,
    validate_jk_method,
)

# ---------- enum membership ----------------------------------------------


def test_gpw_and_gapw_are_distinct_enum_members():
    """Design doc § 9 decision 3: GPW and GAPW are two separate enum
    members, not one ``GAPW(augmentation=bool)``."""
    assert PeriodicJKMethod.GPW is not PeriodicJKMethod.GAPW
    assert PeriodicJKMethod.GPW.value == "gpw"
    assert PeriodicJKMethod.GAPW.value == "gapw"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("gpw", PeriodicJKMethod.GPW),
        ("GPW", PeriodicJKMethod.GPW),
        ("gapw", PeriodicJKMethod.GAPW),
        ("GAPW", PeriodicJKMethod.GAPW),
    ],
)
def test_pick_jk_method_recognises_gpw_gapw_strings(name, expected):
    """User input ``jk_method="gpw"`` / ``"gapw"`` (any case) is
    coerced to the right enum value rather than raising
    ``"Unknown periodic JK method"``."""
    resolved = pick_jk_method(
        name,
        lattice=np.eye(3) * 12.0,
        basis_name="sto-3g",
        n_atoms=2,
    )
    assert resolved is expected


# ---------- not-yet-implemented validation -------------------------------


def test_validate_gpw_accepts_after_m3b_dispatch():
    """As of the M3b runner-dispatch wiring,
    ``validate_jk_method(GPW)`` succeeds silently — the runner
    routes to ``run_periodic_rhf_gpw`` via the new adapter."""
    validate_jk_method(
        PeriodicJKMethod.GPW,
        lattice=np.eye(3) * 12.0,
        basis_name="sto-3g",
    )


def test_validate_gapw_accepts_now_implemented():
    """As of the M3c GAPW augmentation, validate_jk_method(GAPW)
    succeeds silently — the augmentation correction is wired and
    the route is end-to-end functional."""
    validate_jk_method(
        PeriodicJKMethod.GAPW,
        lattice=np.eye(3) * 12.0,
        basis_name="sto-3g",
    )


# ---------- AUTO heuristic unchanged --------------------------------------


@pytest.mark.parametrize(
    "scf_method, expected",
    [
        ("RHF", PeriodicJKMethod.GDF),
        ("RKS", PeriodicJKMethod.GDF),
        ("UHF", PeriodicJKMethod.BIPOLE),
        ("UKS", PeriodicJKMethod.BIPOLE),
    ],
)
def test_auto_does_not_pick_gpw_or_gapw(scf_method, expected):
    """Design doc § 9 decision 4: AUTO continues to pick GDF
    (closed-shell) / BIPOLE (open-shell). GPW / GAPW are opt-in
    through M4."""
    resolved = pick_jk_method(
        "auto",
        lattice=np.eye(3) * 12.0,
        basis_name="sto-3g",
        n_atoms=2,
        scf_method=scf_method,
    )
    assert resolved is expected
    assert resolved is not PeriodicJKMethod.GPW
    assert resolved is not PeriodicJKMethod.GAPW


# ---------- describe_jk_method has labels ---------------------------------


@pytest.mark.parametrize(
    "method",
    [
        PeriodicJKMethod.GPW,
        PeriodicJKMethod.GAPW,
    ],
)
def test_describe_jk_method_has_non_empty_label(method):
    """Banner / output logs must identify the resolved JK route, so
    every concrete enum member needs a description string."""
    label = describe_jk_method(method)
    assert isinstance(label, str)
    assert label
    # The label should mention the method name in upper case so a
    # user scanning the log can match it back to their `jk_method=`
    # input.
    assert method.name in label.upper()
