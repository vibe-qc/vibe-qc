"""Tests for ``vibeqc.print_settings`` / ``vibeqc.format_settings``.

The settings-introspection helper (added in v0.6.0) walks every
user-tunable option struct, marks user overrides with ``*``, and
includes the runtime env-vars vibe-qc reads. These tests pin the
contract: no-arg call works, every struct on the spec is supported,
modifications surface as ``*``, and the env-var section reflects
the current shell state.
"""

from __future__ import annotations

import pytest
import vibeqc
from vibeqc import (
    CPHFOptions,
    D3BJParams,
    DavidsonOptions,
    EwaldOptions,
    GradientOptions,
    GridOptions,
    HessianFDOptions,
    LatticeSumOptions,
    LOBPCGOptions,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    PeriodicSCFOptions,
    RHFOptions,
    RKSOptions,
    ThermoOptions,
    UHFOptions,
    UKSOptions,
    format_settings,
    print_settings,
)

# ---------------------------------------------------------------------------
# Public-API smoke
# ---------------------------------------------------------------------------


def test_module_exports_both_helpers():
    assert callable(vibeqc.print_settings)
    assert callable(vibeqc.format_settings)
    assert "print_settings" in vibeqc.__all__
    assert "format_settings" in vibeqc.__all__


def test_no_arg_call_works(capsys):
    """``print_settings()`` with no args must not raise and must produce
    output covering every named struct + the env-var section."""
    print_settings()
    captured = capsys.readouterr()
    assert captured.out, "print_settings() produced no output"

    # Every option struct must appear as a section header.
    for cls_name in [
        "RHFOptions",
        "UHFOptions",
        "RKSOptions",
        "UKSOptions",
        "PeriodicSCFOptions",
        "PeriodicRHFOptions",
        "PeriodicKSOptions",
        "LatticeSumOptions",
        "EwaldOptions",
        "GridOptions",
        "GradientOptions",
        "HessianFDOptions",
        "CPHFOptions",
        "ThermoOptions",
        "D3BJParams",
        "DavidsonOptions",
        "LOBPCGOptions",
    ]:
        assert cls_name in captured.out, f"missing section: {cls_name}"
    assert "Environment variables" in captured.out


def test_format_settings_returns_same_text_as_print(capsys):
    text = format_settings()
    print_settings()
    captured = capsys.readouterr()
    assert text == captured.out


# ---------------------------------------------------------------------------
# Per-struct: passing each instance should narrow output to that struct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        RHFOptions,
        UHFOptions,
        RKSOptions,
        UKSOptions,
        PeriodicSCFOptions,
        PeriodicRHFOptions,
        PeriodicKSOptions,
        LatticeSumOptions,
        EwaldOptions,
        GridOptions,
        GradientOptions,
        HessianFDOptions,
        CPHFOptions,
        ThermoOptions,
        D3BJParams,
        DavidsonOptions,
        LOBPCGOptions,
    ],
    ids=lambda f: f.__name__,
)
def test_each_struct_accepted(factory):
    """Every option struct must format cleanly when passed as the
    ``opts`` argument."""
    text = format_settings(factory())
    assert factory.__name__ in text
    # No struct other than this one should have its own section header
    # (the ``=====`` bar appears only above the rendered struct + the
    # leading/trailing rules — count of struct names in output ≤ 1).
    other_names = [
        "RHFOptions",
        "UHFOptions",
        "RKSOptions",
        "UKSOptions",
        "PeriodicSCFOptions",
        "PeriodicRHFOptions",
        "PeriodicKSOptions",
        "LatticeSumOptions",
        "EwaldOptions",
        "GridOptions",
        "GradientOptions",
        "HessianFDOptions",
        "CPHFOptions",
        "ThermoOptions",
        "D3BJParams",
        "DavidsonOptions",
        "LOBPCGOptions",
    ]
    other_names = [n for n in other_names if n != factory.__name__]
    for name in other_names:
        # Allowed in attribute values (e.g. "GridOptions" never appears
        # as a value, but be tolerant). Strict header check: the line
        # equal to ``name`` (no padding) is the section title.
        assert f"\n{name}\n" not in text, (
            f"section {name!r} leaked into single-struct dump for {factory.__name__}"
        )


def test_no_args_dumps_no_modifications():
    """No instance → no ``*`` markers (everything matches its default)."""
    text = format_settings()
    # The marker character is ``*`` placed in column 3 of the data rows.
    # Defaults-only: no ``*`` should appear in any struct row. The
    # env-var section is allowed to carry ``*`` for whatever the
    # process happens to have set, so check only the struct portion.
    structs_text = text.split("Environment variables")[0]
    assert "  * " not in structs_text, (
        "no-arg dump must not flag any struct attribute as modified"
    )


# ---------------------------------------------------------------------------
# Modifications get the ``*`` marker
# ---------------------------------------------------------------------------


def test_rhf_modification_marked():
    opts = RHFOptions()
    opts.max_iter = 250
    opts.damping = 0.7
    opts.conv_tol_energy = 1e-12
    text = format_settings(opts)

    # ``  * max_iter`` rows: one per modified attribute.
    assert "  * max_iter" in text
    assert "  * damping" in text
    assert "  * conv_tol_energy" in text
    # Untouched attributes do not get the marker.
    assert "  * use_diis" not in text
    assert "  * diis_subspace_size" not in text


def test_python_dataclass_modification_marked():
    """Cover the dataclass-defined options (CPHFOptions, HessianFDOptions,
    ThermoOptions) — the registry handles them via ``dataclasses.fields``,
    a different code path from the C++-bound structs."""
    cphf = CPHFOptions()
    cphf.max_iter = 500
    cphf.tol = 1e-10
    text = format_settings(cphf)
    assert "  * max_iter" in text
    assert "  * tol" in text
    assert "  * use_preconditioner" not in text


def test_nested_option_modification_marked():
    """``RKSOptions.grid`` is a nested ``GridOptions``. Modifications to
    nested attributes must surface as ``*`` against the dotted key
    (``grid.n_radial``), not just on the top-level ``grid`` field."""
    opts = RKSOptions()
    opts.functional = "PBE"
    opts.grid.n_radial = 99
    text = format_settings(opts)
    assert "  * functional" in text
    assert "  * grid.n_radial" in text
    # Other grid components stay default-marked.
    assert "    grid.n_phi" in text  # two leading spaces + space marker
    assert "    grid.becke_k" in text


def test_periodic_lattice_opts_nested():
    """PeriodicRHFOptions has a ``lattice_opts`` nested struct
    (LatticeSumOptions); make sure that path also flattens."""
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 30.0
    text = format_settings(opts)
    assert "  * lattice_opts.cutoff_bohr" in text
    assert "    lattice_opts.nuclear_cutoff_bohr" in text


def test_unknown_type_raises():
    with pytest.raises(TypeError, match="opts must be one of"):
        format_settings("not an options struct")
    with pytest.raises(TypeError):
        format_settings(42)


# ---------------------------------------------------------------------------
# Env-var section reflects the shell at format time
# ---------------------------------------------------------------------------


def test_env_section_marks_set_vars(monkeypatch):
    """A set env var (any of the documented ones) gets ``*`` and
    its current value rendered next to the variable name."""
    monkeypatch.setenv("VIBEQC_PERFLOG", "section")
    text = format_settings()
    # Find the env-var portion.
    env_text = text.split("Environment variables", 1)[1]
    # The line for our just-set var should carry ``*``.
    perflog_lines = [ln for ln in env_text.splitlines() if "VIBEQC_PERFLOG" in ln]
    assert perflog_lines, "VIBEQC_PERFLOG row missing from env-var section"
    assert any("*" in ln for ln in perflog_lines), (
        "VIBEQC_PERFLOG should be flagged when set in os.environ"
    )
    assert any("'section'" in ln for ln in perflog_lines), (
        "VIBEQC_PERFLOG row should display its current value"
    )


def test_env_section_marks_unset_vars(monkeypatch):
    """A var that's not in the environment renders as ``<unset>`` and
    carries no ``*`` marker."""
    monkeypatch.delenv("VIBEQC_NO_HOSTNAME", raising=False)
    monkeypatch.delenv("VIBEQC_BUILD_TAG", raising=False)
    text = format_settings()
    env_text = text.split("Environment variables", 1)[1]
    for varname in ("VIBEQC_NO_HOSTNAME", "VIBEQC_BUILD_TAG"):
        rows = [ln for ln in env_text.splitlines() if varname in ln]
        assert rows, f"{varname} row missing"
        assert all("<unset>" in ln for ln in rows), (
            f"{varname} should render as <unset>"
        )
        # No ``*`` marker in column 3 for unset vars.
        for ln in rows:
            assert not ln.lstrip().startswith("*"), (
                f"unset {varname} should not carry the modified marker"
            )


def test_env_section_lists_all_documented_vars():
    """Every env var vibe-qc reads at runtime must appear in the
    env-var section, regardless of which are currently set."""
    text = format_settings()
    env_text = text.split("Environment variables", 1)[1]
    for var in (
        "OMP_NUM_THREADS",
        "LIBINT_DATA_PATH",
        "VIBEQC_ECP_SHARE_DIR",
        "VIBEQC_FILTERED_BASIS_DIR",
        "VIBEQC_LIVE_LOGGING",
        "VIBEQC_NO_HOSTNAME",
        "VIBEQC_NO_CRASH_DUMP",
        "VIBEQC_PERFLOG",
        "VIBEQC_PERIODIC_XC_MAX_ESTIMATED_GB",
        "VIBEQC_STRUCTURED_LOG",
        "VIBEQC_VERBOSE",
        "VIBEQC_BUILD_BRANCH",
        "VIBEQC_BUILD_SHA",
        "VIBEQC_BUILD_TAG",
    ):
        assert var in env_text, f"missing env var in dump: {var}"


# ---------------------------------------------------------------------------
# Solver keyword
# ---------------------------------------------------------------------------


def test_solver_default():
    """With no solver= keyword, the Solver section shows 'dense' and no
    ``*`` marker."""
    text = format_settings()
    assert "Solver" in text
    solver_lines = [ln for ln in text.splitlines() if "solver" in ln.lower()]
    assert any("dense" in ln for ln in solver_lines), "default solver must be 'dense'"
    for ln in solver_lines:
        if "dense" in ln and "solver" in ln:
            assert not ln.lstrip().startswith("*"), (
                "default solver must not carry * marker"
            )


def test_solver_davidson_marked():
    """``solver='davidson'`` adds a Solver section with ``*`` marker."""
    text = format_settings(solver="davidson")
    assert "Solver" in text
    solver_lines = [ln for ln in text.splitlines() if "davidson" in ln]
    assert solver_lines, "davidson solver line missing"
    assert any(ln.lstrip().startswith("*") for ln in solver_lines), (
        "davidson solver must carry * marker"
    )


def test_solver_lobpcg_marked():
    """``solver='lobpcg'`` adds a Solver section with ``*`` marker."""
    text = format_settings(solver="lobpcg")
    assert "Solver" in text
    solver_lines = [ln for ln in text.splitlines() if "lobpcg" in ln]
    assert solver_lines, "lobpcg solver line missing"
    assert any(ln.lstrip().startswith("*") for ln in solver_lines), (
        "lobpcg solver must carry * marker"
    )
