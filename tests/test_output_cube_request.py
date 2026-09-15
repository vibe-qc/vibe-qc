"""Cube-request parsing + plan integration for Phase O6.

Pins the contract for the ``write_cube=`` kwarg surface:

  1. ``parse_write_cube_kwarg`` accepts ``None`` / ``False`` /
     ``True`` / ``"density"`` / ``"homo"`` / ``"lumo"`` / int /
     list-of-mixed. Unknown shapes raise TypeError.
  2. ``requested_mo_indices`` resolves ``"homo"`` / ``"lumo"`` /
     ``"homo-N"`` / ``"lumo+N"`` against the result's
     ``mo_occupations``. Out-of-range / missing-data cases raise
     ValueError so the caller's wrapper can warn cleanly.
  3. ``OutputPlan.from_run_job_kwargs(write_cube_density=True,
     cube_mo_labels=("homo", "lumo"))`` declares the matching cube
     siblings in ``[[plan.files]]``.

The actual grid evaluation (in vibeqc.cube) is covered by the
existing tests/test_cube.py; this file only pins the
output-module routing layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vibeqc.output import OutputPlan
from vibeqc.output.formats.cube import (
    CubeRequest,
    parse_write_cube_kwarg,
    requested_mo_indices,
)


# ---------------------------------------------------------------------- #
# parse_write_cube_kwarg
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [None, False, ""])
def test_falsy_kwargs_yield_empty_request(value) -> None:
    req = parse_write_cube_kwarg(value)
    assert not req
    assert req.density is False
    assert req.mo_labels == ()


def test_true_yields_density_only() -> None:
    req = parse_write_cube_kwarg(True)
    assert req.density is True
    assert req.mo_labels == ()


def test_density_string_yields_density_only() -> None:
    req = parse_write_cube_kwarg("density")
    assert req.density is True
    assert req.mo_labels == ()


def test_homo_string_yields_mo_label() -> None:
    req = parse_write_cube_kwarg("homo")
    assert req.density is False
    assert req.mo_labels == ("homo",)


def test_int_yields_explicit_index() -> None:
    req = parse_write_cube_kwarg(5)
    assert req.mo_labels == (5,)


def test_mixed_list_collects_all() -> None:
    req = parse_write_cube_kwarg(
        ["density", "homo", "lumo", 7],
    )
    assert req.density is True
    assert req.mo_labels == ("homo", "lumo", 7)


def test_unknown_type_raises() -> None:
    with pytest.raises(TypeError):
        parse_write_cube_kwarg(3.14)
    with pytest.raises(TypeError):
        parse_write_cube_kwarg([1.5])


# ---------------------------------------------------------------------- #
# requested_mo_indices
# ---------------------------------------------------------------------- #

@dataclass
class _StubResult:
    mo_occupations: np.ndarray


def _result_homo_3() -> _StubResult:
    """5-MO result with HOMO=2 (3 occupied), 2 virtuals."""
    return _StubResult(np.array([2.0, 2.0, 2.0, 0.0, 0.0]))


def test_homo_resolves_to_last_occupied() -> None:
    out = requested_mo_indices(["homo"], _result_homo_3())
    assert out == [(2, "homo")]


def test_lumo_resolves_to_first_virtual() -> None:
    out = requested_mo_indices(["lumo"], _result_homo_3())
    assert out == [(3, "lumo")]


def test_homo_minus_one() -> None:
    out = requested_mo_indices(["homo-1"], _result_homo_3())
    assert out == [(1, "homo-1")]


def test_lumo_plus_one() -> None:
    out = requested_mo_indices(["lumo+1"], _result_homo_3())
    assert out == [(4, "lumo+1")]


def test_int_index_passes_through() -> None:
    out = requested_mo_indices([0, 4], _result_homo_3())
    assert out == [(0, "mo_0"), (4, "mo_4")]


def test_out_of_range_int_raises() -> None:
    with pytest.raises(ValueError):
        requested_mo_indices([99], _result_homo_3())


def test_out_of_range_homo_minus_raises() -> None:
    with pytest.raises(ValueError):
        requested_mo_indices(["homo-99"], _result_homo_3())


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError):
        requested_mo_indices(["nope"], _result_homo_3())


def test_no_virtual_lumo_raises() -> None:
    """An all-occupied MO set (e.g. minimal basis, full shell) has no
    LUMO — the helper should raise rather than silently return HOMO."""
    res = _StubResult(np.array([2.0, 2.0, 2.0]))  # all occupied
    with pytest.raises(ValueError):
        requested_mo_indices(["lumo"], res)


# ---------------------------------------------------------------------- #
# OutputPlan integration
# ---------------------------------------------------------------------- #

def test_plan_declares_density_cube(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf", basis="sto-3g", functional=None,
        write_cube_density=True,
    )
    paths = {str(f.path) for f in plan.files}
    assert str(tmp_path / "h2o.density.cube") in paths


def test_plan_declares_mo_cubes(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf", basis="sto-3g", functional=None,
        cube_mo_labels=("homo", "lumo"),
    )
    paths = {str(f.path) for f in plan.files}
    assert str(tmp_path / "h2o.homo.cube") in paths
    assert str(tmp_path / "h2o.lumo.cube") in paths


def test_plan_no_cubes_when_kwargs_off(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf", basis="sto-3g", functional=None,
    )
    cube_files = [f for f in plan.files if f.format == "cube"]
    assert cube_files == []
