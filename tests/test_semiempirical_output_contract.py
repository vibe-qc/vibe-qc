from __future__ import annotations

import ast
from pathlib import Path


_SEMIEMPIRICAL_ROOT = (
    Path(__file__).resolve().parents[1] / "python" / "vibeqc" / "semiempirical"
)


def _print_call_sites() -> list[tuple[str, int]]:
    sites: list[tuple[str, int]] = []
    for path in sorted(_SEMIEMPIRICAL_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_SEMIEMPIRICAL_ROOT).as_posix()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                sites.append((rel, node.lineno))
    return sites


def test_semiempirical_modules_do_not_print_user_output():
    offenders = _print_call_sites()
    assert offenders == [], (
        "semiempirical modules must emit user-facing text through "
        "vibeqc.output.write(), not print(): "
        f"{offenders}"
    )


def _source_definition(relative_path: str, name: str, namespace: dict):
    """Exercise Python routing without importing an unrelated native build."""
    path = _SEMIEMPIRICAL_ROOT / relative_path
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body if getattr(node, "name", None) == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), definition],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def test_gfn2_model_selects_faithful_by_default_and_preserves_false():
    from types import SimpleNamespace

    class ModelBase:
        def __init__(self, molecule, **kwargs):
            pass

    namespace = {
        "SemiempiricalModel": ModelBase,
        "SemiempiricalRoutePlan": SimpleNamespace(from_request=lambda *a, **k: None),
        "_GFN2_SCC_MIXERS": {"simple": "simple"},
        "_xtb": SimpleNamespace(XTBSccOptions=lambda: SimpleNamespace(aes_faithful=False)),
    }
    model = _source_definition("methods/gfn2.py", "GFN2Model", namespace)
    molecule = SimpleNamespace(charge=0, multiplicity=1)
    assert model(molecule, object())._make_scc_options().aes_faithful is True
    assert model(molecule, object(), aes_faithful=False)._make_scc_options().aes_faithful is False


def test_gfn2_aes_provenance_uses_executed_identity():
    from types import SimpleNamespace

    helper = _source_definition("runner.py", "_gfn2_parameter_provenance", {})
    tree = ast.parse((_SEMIEMPIRICAL_ROOT / "runner.py").read_text())
    definition = next(n for n in tree.body if getattr(n, "name", None) == helper.__name__)
    keys = {
        node.slice.value for node in ast.walk(definition)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        and node.value.id == "snapshot" and isinstance(node.slice, ast.Constant)
    }
    for executed, requested in [("faithful", False), ("shell", True), ("unknown", True)]:
        snapshot = dict.fromkeys(keys, 0)
        snapshot.update(parameter_identity="captured", parameter_sha256="a" * 64,
                        aes_faithful=requested)
        native = SimpleNamespace(parameter_identity="captured", parameter_sha256="a" * 64,
                                 smearing_temperature=0, aes_functional=executed)
        provenance = helper(SimpleNamespace(_last_parameter_snapshot=snapshot, _last_result=native))
        assert provenance["aes_model"] == executed
        assert provenance["gfn2_aes_faithful"] is requested


def test_gfn2_native_and_parity_defaults_are_not_overridden():
    root = _SEMIEMPIRICAL_ROOT.parents[2]
    header = (root / "cpp/include/vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp").read_text()
    assert "bool aes_faithful = true;" in header
    parity = ast.parse((root / "tests/test_gfn2_xtb_parity_matrix.py").read_text())
    run = next(n for n in parity.body if getattr(n, "name", None) == "_run")
    assert not any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "aes_faithful"
                                          for t in n.targets)
        for n in ast.walk(run)
    )


def test_gfn2_runner_and_geometry_provider_inherit_model_default():
    root = _SEMIEMPIRICAL_ROOT.parents[2]
    for path in [
        _SEMIEMPIRICAL_ROOT / "runner.py",
        root / "python/vibeqc/geomopt/semiempirical_provider.py",
    ]:
        calls = [n for n in ast.walk(ast.parse(path.read_text()))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "GFN2Model"]
        assert calls, path
        assert all(not any(k.arg == "aes_faithful" for k in call.keywords) for call in calls)
