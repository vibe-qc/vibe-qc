"""Role-driven dispatch infrastructure — pre-v1.0 D1.

Pins the contract for :mod:`vibeqc.output.dispatch`:

  1. ``Dispatcher.register(role, format, fn)`` records a writer
     adapter; duplicate registrations raise ValueError unless
     ``overwrite=True``.
  2. ``Dispatcher.dispatch_planned_file(pf, stem=..., **ctx)``
     invokes the registered writer for the file's
     ``(role, format)`` pair; missing registration returns None;
     adapter exceptions are caught + logged at WARNING + return
     None (best-effort, never tank the SCF).
  3. ``Dispatcher.dispatch_plan(plan, only_role=..., only_always=
     True, **ctx)`` walks the plan, returns
     ``[(plan_file, written_path), ...]``.
  4. ``default_dispatcher()`` is pre-registered with every built-
     in writer adapter covering Phases O1–O6.
  5. ``OutputWriter.dispatch_role(role, **ctx)`` is the wired-up
     wrapper: walks the plan for that role, dispatches each
     ``always=True`` PlannedFile, records every successful write
     in the manifest's ``[[outputs.files]]``.
"""

from __future__ import annotations

import ast
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest
from vibeqc.output import (
    Dispatcher,
    OutputPlan,
    OutputWriter,
    PlannedFile,
    default_dispatcher,
)


def test_periodic_runner_has_no_direct_artifact_writer_calls() -> None:
    path = Path(__file__).parents[1] / "python/vibeqc/periodic_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {
        "write_molden",
        "write_population",
        "write_population_summary",
        "write_extended_xyz",
        "_write_poscar",
        "_write_cif",
        "write_bibtex",
        "write_references",
        "write_xsf_structure",
        "write_xsf_volume",
        "_write_qvf",
    }
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        if name in forbidden:
            calls.append(f"{name}:{node.lineno}")
    assert calls == []


# Duck-typed molecule / system so the test suite doesn't need the
# C++ core importable (same pattern as test_output_xyz.py).
@dataclass
class _Atom:
    Z: int
    xyz: tuple


@dataclass
class _Mol:
    atoms: list


# ---------------------------------------------------------------------- #
# Dispatcher.register / get_writer
# ---------------------------------------------------------------------- #


def test_register_records_writer() -> None:
    d = Dispatcher()
    fn = lambda *, stem, plan_file, **_: plan_file.path  # noqa: E731
    d.register("log", "text", fn)
    assert d.get_writer("log", "text") is fn


def test_duplicate_register_raises_without_overwrite() -> None:
    d = Dispatcher()
    fn = lambda *, stem, plan_file, **_: None  # noqa: E731
    d.register("log", "text", fn)
    with pytest.raises(ValueError, match="already registered"):
        d.register("log", "text", fn)


def test_duplicate_register_with_overwrite_replaces() -> None:
    d = Dispatcher()
    fn_a = lambda *, stem, plan_file, **_: Path("a")  # noqa: E731
    fn_b = lambda *, stem, plan_file, **_: Path("b")  # noqa: E731
    d.register("log", "text", fn_a)
    d.register("log", "text", fn_b, overwrite=True)
    assert d.get_writer("log", "text") is fn_b


def test_unregister_silent_when_missing() -> None:
    d = Dispatcher()
    d.unregister("log", "text")  # must not raise


def test_get_writer_returns_none_when_missing() -> None:
    d = Dispatcher()
    assert d.get_writer("log", "text") is None


def test_registered_keys_returns_insertion_order() -> None:
    d = Dispatcher()
    fn = lambda *, stem, plan_file, **_: None  # noqa: E731
    d.register("log", "text", fn)
    d.register("orbitals", "molden", fn)
    keys = d.registered_keys()
    assert keys == (("log", "text"), ("orbitals", "molden"))


# ---------------------------------------------------------------------- #
# dispatch_planned_file
# ---------------------------------------------------------------------- #


def _pf(role: str, fmt: str, path: Path) -> PlannedFile:
    return PlannedFile(
        role=role,
        path=path,
        format=fmt,  # type: ignore[arg-type]
        always=True,
        description="x",
    )


def test_dispatch_returns_path_when_writer_present(
    tmp_path: Path,
) -> None:
    d = Dispatcher()
    calls = []

    def writer(*, stem, plan_file, **ctx):
        calls.append((stem, plan_file, dict(ctx)))
        return plan_file.path

    d.register("orbitals", "molden", writer)
    pf = _pf("orbitals", "molden", tmp_path / "x.molden")
    out = d.dispatch_planned_file(
        pf,
        stem=tmp_path / "x",
        extra="hello",
    )
    assert out == pf.path
    assert len(calls) == 1
    assert calls[0][2]["extra"] == "hello"


def test_dispatch_returns_none_when_writer_missing(
    tmp_path: Path,
) -> None:
    d = Dispatcher()
    pf = _pf("orbitals", "molden", tmp_path / "x.molden")
    assert d.dispatch_planned_file(pf, stem=tmp_path / "x") is None


def test_dispatch_swallows_writer_exception(tmp_path: Path) -> None:
    d = Dispatcher()

    def boom(*, stem, plan_file, **_):
        raise RuntimeError("synthetic failure")

    d.register("orbitals", "molden", boom)
    pf = _pf("orbitals", "molden", tmp_path / "x.molden")
    # Must NOT raise — best-effort contract.
    assert d.dispatch_planned_file(pf, stem=tmp_path / "x") is None


def test_dispatch_strict_mode_raises_for_missing_writer(tmp_path: Path) -> None:
    d = Dispatcher()
    pf = _pf("orbitals", "molden", tmp_path / "x.molden")
    with pytest.raises(LookupError, match="no writer registered"):
        d.dispatch_planned_file(
            pf,
            stem=tmp_path / "x",
            raise_on_error=True,
        )


def test_dispatch_strict_mode_propagates_writer_exception(
    tmp_path: Path,
) -> None:
    d = Dispatcher()

    def boom(*, stem, plan_file, **_):
        raise RuntimeError("synthetic failure")

    d.register("orbitals", "molden", boom)
    pf = _pf("orbitals", "molden", tmp_path / "x.molden")
    with pytest.raises(RuntimeError, match="synthetic failure"):
        d.dispatch_planned_file(
            pf,
            stem=tmp_path / "x",
            raise_on_error=True,
        )


# ---------------------------------------------------------------------- #
# dispatch_plan
# ---------------------------------------------------------------------- #


def test_dispatch_plan_walks_every_file(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    d = Dispatcher()
    seen: list[tuple[str, str]] = []

    def record_writer(*, stem, plan_file, **_):
        seen.append((plan_file.role, plan_file.format))
        return plan_file.path

    # Register a "log all roles" writer for every key in the plan.
    for pf in plan.files:
        if (pf.role, pf.format) not in d.registered_keys():
            d.register(pf.role, pf.format, record_writer)

    results = d.dispatch_plan(plan)
    # Every always=True row dispatched exactly once.
    expected_always = sum(1 for pf in plan.files if pf.always)
    assert len(results) == expected_always


def test_dispatch_plan_filters_by_role(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    d = Dispatcher()
    fn = lambda *, stem, plan_file, **_: plan_file.path  # noqa: E731
    for pf in plan.files:
        if (pf.role, pf.format) not in d.registered_keys():
            d.register(pf.role, pf.format, fn)

    results = d.dispatch_plan(plan, only_role="citations")
    # Plan has exactly 2 citations rows (.bibtex + .references).
    assert len(results) == 2
    assert all(pf.role == "citations" for pf, _ in results)


def test_dispatch_plan_skips_conditional_by_default(
    tmp_path: Path,
) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
        crash_dump=True,  # declares an always=False crash row
    )
    d = Dispatcher()
    fn = lambda *, stem, plan_file, **_: plan_file.path  # noqa: E731
    for pf in plan.files:
        if (pf.role, pf.format) not in d.registered_keys():
            d.register(pf.role, pf.format, fn)
    results = d.dispatch_plan(plan)
    # Crash row was conditional; should NOT be in results.
    assert not any(pf.role == "crash" for pf, _ in results)


def test_dispatch_plan_includes_conditional_when_asked(
    tmp_path: Path,
) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
        crash_dump=True,
    )
    d = Dispatcher()
    fn = lambda *, stem, plan_file, **_: plan_file.path  # noqa: E731
    for pf in plan.files:
        if (pf.role, pf.format) not in d.registered_keys():
            d.register(pf.role, pf.format, fn)
    results = d.dispatch_plan(plan, only_always=False)
    assert any(pf.role == "crash" for pf, _ in results)


# ---------------------------------------------------------------------- #
# default_dispatcher — pre-registered keys
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "key",
    [
        ("geometry", "xyz"),
        ("geometry", "extended-xyz"),
        ("geometry", "poscar"),
        ("geometry", "cif"),
        ("geometry", "xsf"),
        ("orbitals", "molden"),
        ("orbitals", "nto-molden"),
        ("orbitals", "trexio"),
        ("population", "text"),
        ("population", "json"),
        ("population", "bipole-text"),
        ("population", "bipole-json"),
        ("population", "aiccm2026dev-b-text"),
        ("population", "aiccm2026dev-b-json"),
        ("density", "cube"),
        ("density", "xsf"),
        ("orbital_vol", "cube"),
        ("citations", "bibtex"),
        ("citations", "text"),
        ("qvf", "qvf"),
        ("container", "qvf"),
    ],
)
def test_default_dispatcher_has_builtin_writer(
    key: tuple[str, str],
) -> None:
    d = default_dispatcher()
    fn = d.get_writer(*key)
    assert fn is not None, f"missing default writer for {key!r}"
    assert callable(fn)


def test_default_dispatcher_dispatch_xyz(tmp_path: Path) -> None:
    """End-to-end: the default dispatcher correctly drives the xyz
    writer for a molecular geometry plan row."""
    mol = _Mol(
        [
            _Atom(8, (0.0, 0.0, 0.0)),
            _Atom(1, (0.0, 1.4, 0.0)),
            _Atom(1, (0.0, -1.4, 0.0)),
        ]
    )
    d = default_dispatcher()
    pf = _pf("geometry", "xyz", tmp_path / "h2o.xyz")
    out = d.dispatch_planned_file(
        pf,
        stem=tmp_path / "h2o",
        molecule=mol,
        energy_ha=-76.0,
    )
    assert out == tmp_path / "h2o.xyz"
    assert out.is_file()
    body = out.read_text()
    assert body.splitlines()[0] == "3"
    assert "energy=-76.0000000000" in body


# ---------------------------------------------------------------------- #
# OutputWriter.dispatch_role — wired-up wrapper
# ---------------------------------------------------------------------- #


def test_writer_dispatch_role_records_outputs(tmp_path: Path) -> None:
    """OutputWriter.dispatch_role walks the plan for the given role,
    invokes the registered writer for each, records the result in
    the manifest's [[outputs.files]]."""
    mol = _Mol([_Atom(8, (0.0, 0.0, 0.0))])
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    w = OutputWriter(plan)
    written = w.dispatch_role(
        "geometry",
        molecule=mol,
        energy_ha=-76.0,
    )
    assert len(written) == 1
    assert written[0] == (tmp_path / "h2o").with_suffix(".xyz")

    # Manifest [[outputs.files]] should have written=true for the
    # geometry row.
    body = tomllib.loads(
        w.manifest_path.read_text(encoding="utf-8"),
    )
    geom_rows = [r for r in body["outputs"]["files"] if r["path"].endswith(".xyz")]
    assert len(geom_rows) == 1
    assert geom_rows[0]["written"] is True


def test_writer_dispatch_role_custom_dispatcher(tmp_path: Path) -> None:
    """A caller can pass a custom dispatcher to dispatch_role —
    useful for tests and for downstream packages that add their own
    output formats."""
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    w = OutputWriter(plan)

    custom_d = Dispatcher()
    calls: list[Path] = []

    def my_xyz_writer(*, stem, plan_file, **_):
        # Write a sentinel-content file we can recognise.
        plan_file.path.write_text("CUSTOM XYZ")
        calls.append(plan_file.path)
        return plan_file.path

    custom_d.register("geometry", "xyz", my_xyz_writer)
    out = w.dispatch_role("geometry", dispatcher=custom_d)
    assert len(out) == 1
    assert out[0].read_text() == "CUSTOM XYZ"


def test_writer_dispatch_role_missing_adapter_skips(
    tmp_path: Path,
) -> None:
    """No writer registered for the role → dispatch returns empty
    list, no manifest entries flipped to written=True."""
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    w = OutputWriter(plan)
    empty = Dispatcher()  # nothing registered
    written = w.dispatch_role("geometry", dispatcher=empty)
    assert written == []


def test_writer_dispatch_role_filters_format(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    w = OutputWriter(plan)
    d = Dispatcher()
    calls: list[Path] = []

    def writer(*, stem, plan_file, **_):
        plan_file.path.write_text("citation", encoding="utf-8")
        calls.append(plan_file.path)
        return plan_file.path

    d.register("citations", "bibtex", writer)
    d.register("citations", "text", writer)
    written = w.dispatch_role(
        "citations",
        dispatcher=d,
        only_format="bibtex",
        raise_on_error=True,
    )
    assert written == [plan.stem.with_suffix(".bibtex")]
    assert calls == written


def test_writer_dispatch_role_runtime_artifact(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    w = OutputWriter(plan)
    d = Dispatcher()
    runtime_path = tmp_path / "h2o.nto_S1_hole.molden"

    def writer(*, stem, plan_file, **_):
        plan_file.path.write_text("nto", encoding="utf-8")
        return plan_file.path

    d.register("orbitals", "nto-molden", writer)
    written = w.dispatch_role(
        "orbitals",
        dispatcher=d,
        runtime_path=runtime_path,
        runtime_format="nto-molden",
        raise_on_error=True,
    )
    assert written == [runtime_path]
    outcome = next(o for o in w.outcomes() if o.path == runtime_path)
    assert outcome.written is True


def test_population_dispatch_computes_once_for_two_plan_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibeqc.output.formats import population as population_module

    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    calls = 0

    def fake_write_population(stem, result, basis, molecule):
        nonlocal calls
        calls += 1
        txt = Path(stem).parent / (Path(stem).name + ".population.txt")
        jsn = Path(stem).parent / (Path(stem).name + ".population.json")
        txt.write_text("txt", encoding="utf-8")
        jsn.write_text("json", encoding="utf-8")
        return txt, jsn

    monkeypatch.setattr(
        population_module,
        "write_population",
        fake_write_population,
    )
    w = OutputWriter(plan)
    written = w.dispatch_role(
        "population",
        result=object(),
        basis=object(),
        molecule=object(),
        raise_on_error=True,
    )
    assert calls == 1
    assert written == [
        tmp_path / "h2o.population.txt",
        tmp_path / "h2o.population.json",
    ]
    assert all(
        next(o for o in w.outcomes() if o.path == path).written
        for path in written
    )


def test_nto_molden_dispatch_preserves_default_title(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibeqc.output.formats import molden as molden_module

    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    seen: dict[str, object] = {}

    def fake_write(path, molecule, basis, result, *, title=""):
        seen.update(path=path, title=title)
        Path(path).write_text("nto", encoding="utf-8")

    monkeypatch.setattr(molden_module, "write_molden", fake_write)
    target = tmp_path / "h2o.nto_S1_hole.molden"
    written = OutputWriter(plan).dispatch_role(
        "orbitals",
        runtime_path=target,
        runtime_format="nto-molden",
        molecule=object(),
        basis=object(),
        result=object(),
        raise_on_error=True,
    )

    assert written == [target]
    assert seen == {"path": target, "title": ""}
