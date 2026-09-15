"""Record-serialization regression tests for scripts/dual_engine_optimize.py.

IID 257: the dual-engine optimization script computed the BIPOLE SCF verdict
inside ``run_bipole`` and then discarded it. The ``SystemResult.bipole_converged``
field was never assigned, so every serialized record read ``false``, including
for runs that actually converged, and ``delta_mha`` was produced regardless of
the verdict. These tests pin that the computed verdict reaches the record and
that ``delta_mha`` is gated on it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "dual_engine_optimize.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "dual_engine_optimize", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve their namespace through sys.modules, so the
    # module must be registered before exec_module constructs them.
    sys.modules["dual_engine_optimize"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def script(monkeypatch):
    """Load the script with both engines and the basis fetch stubbed out."""
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "SYSTEMS",
        {
            "MgO": {
                "a_ang": 4.189,
                "z1": 12,
                "sym1": "Mg",
                "z2": 8,
                "sym2": "O",
            }
        },
    )
    monkeypatch.setattr(mod, "get_pob_tzvp_inline_crystal", lambda: "dummy\n")
    return mod


def _invoke_main(
    mod,
    monkeypatch,
    tmp_path: Path,
    *,
    bipole_energy,
    converged: bool,
    n_iter: int,
    crystal_energy,
):
    monkeypatch.setattr(
        mod, "run_crystal14", lambda name, workdir: crystal_energy
    )
    monkeypatch.setattr(
        mod,
        "run_bipole",
        lambda name, a_ang, z1, z2: mod.BipoleOutcome(
            energy=bipole_energy, converged=converged, n_iter=n_iter
        ),
    )
    monkeypatch.chdir(tmp_path)
    mod.main()
    return json.loads((tmp_path / "dual_engine_results.json").read_text())


def test_converged_run_serializes_true_and_delta(script, monkeypatch, tmp_path):
    record = _invoke_main(
        script,
        monkeypatch,
        tmp_path,
        bipole_energy=-100.0,
        converged=True,
        n_iter=9,
        crystal_energy=-101.0,
    )
    mg = record["MgO"]
    assert mg["bipole_converged"] is True
    assert mg["bipole_n_iter"] == 9
    assert mg["delta_mha"] == pytest.approx((-100.0 - (-101.0)) * 1000.0)


def test_nonconverged_run_serializes_false_and_withholds_delta(
    script, monkeypatch, tmp_path
):
    record = _invoke_main(
        script,
        monkeypatch,
        tmp_path,
        bipole_energy=-100.0,
        converged=False,
        n_iter=30,
        crystal_energy=-101.0,
    )
    mg = record["MgO"]
    assert mg["bipole_converged"] is False
    assert mg["bipole_n_iter"] == 30
    assert mg["delta_mha"] is None
    assert any("withheld" in note for note in mg["notes"])


def test_failed_run_serializes_false_and_no_delta(script, monkeypatch, tmp_path):
    record = _invoke_main(
        script,
        monkeypatch,
        tmp_path,
        bipole_energy=None,
        converged=False,
        n_iter=0,
        crystal_energy=-101.0,
    )
    mg = record["MgO"]
    assert mg["bipole_converged"] is False
    assert mg["bipole_energy_ha"] is None
    assert mg["delta_mha"] is None
