"""Native-free checks of final force serialization (#641), no calculations."""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

def helper():
    tree = ast.parse((ROOT / "python/vibeqc/runner.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_optimization_force_record")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "runner-force-record", "exec"), namespace)
    return namespace[function.name]

@pytest.mark.parametrize("units", ["eV/angstrom", "Eh/bohr"])
def test_exact_forces_units_and_convergence_are_preserved(units):
    f = np.array([[.001, -.002, .003], [.004, .005, -.006]])
    record = helper()(f, SimpleNamespace(atoms=[1, 2]), units=units, converged=False)
    assert record == {"converged": False, "final_forces": {"units": units, "values": f.tolist()}}
    f[:] = 0
    assert record["final_forces"]["values"][0][0] == .001

@pytest.mark.parametrize("forces", [None, [], [1, 2], [[float("nan"), 0, 0]], [[float("inf"), 0, 0]], [["bad", 0, 0]]])
def test_missing_or_invalid_forces_never_become_zero_evidence(forces):
    record = helper()(forces, SimpleNamespace(atoms=[1]), units="Eh/bohr", converged=True)
    assert record["converged"] is True
    assert "final_forces" not in record
    assert record["forces_unavailable"]

def test_cartesian_gradient_conversion_preserves_atom_order():
    gradient = np.array([.1, .2, .3, -.4, -.5, -.6])
    record = helper()(-gradient, SimpleNamespace(atoms=[1, 2]), units="Eh/bohr", converged=True)
    assert record["final_forces"]["values"] == [[-.1, -.2, -.3], [.4, .5, .6]]


def test_qvf_provenance_retains_force_record_without_changing_run_verdict(monkeypatch):
    import sys
    from types import ModuleType
    from typing import Any

    package = "force_evidence_test"
    parent = ModuleType(package)
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, package, parent)
    for name, attribute, function in (
        ("mlip_record", "mlip_record_fields", lambda result: {}),
        ("system_info", "_hostname_redacted", lambda: True),
    ):
        module = ModuleType(package + "." + name)
        setattr(module, attribute, function)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    tree = ast.parse((ROOT / "python/vibeqc/output/formats/qvf.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_provenance")
    namespace = {"Any": Any, "__package__": package}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "qvf-provenance", "exec"), namespace)
    optimization = helper()([[.1, .2, .3]], SimpleNamespace(atoms=[1]), units="Eh/bohr", converged=False)
    result = namespace["_build_provenance"]({"record_hostname": False, "optimization": optimization,
        "run_status": "failed", "run_verdict": {"usable": False}})
    assert result["optimization"] == optimization
    assert result["run_status"] == "failed" and result["verdict"]["usable"] is False
    assert "optimization" not in namespace["_build_provenance"]({"record_hostname": False})
