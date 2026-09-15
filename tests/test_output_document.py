"""Tests for the semantic document primitives in ``vibeqc.output.document``.

These pin the properties the primitives exist to guarantee: that a rule is
as wide as its table, that precision and units are policy-driven rather
than baked into the emitting module, and that a quantity's unit survives
conversion round-trips.
"""

from __future__ import annotations

import math

import pytest

from vibeqc.output.document import (
    DEFAULT_POLICY,
    Column,
    CriteriaTable,
    Criterion,
    FormatPolicy,
    FormatSpec,
    HeaderlessBlock,
    HeaderlessColumn,
    OutputDocument,
    Quantity,
    Table,
    canonical_unit,
    dimension_of,
    known_units,
)


# --------------------------------------------------------------------
# Quantity
# --------------------------------------------------------------------


def test_quantity_defaults_to_canonical_unit():
    assert Quantity(-76.0, "energy").unit == "Ha"
    assert Quantity(1.0, "gradient").unit == "Ha/bohr"
    assert canonical_unit("length") == "bohr"


def test_quantity_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown quantity kind"):
        Quantity(1.0, "flux_capacitance")


def test_quantity_rejects_unit_from_another_kind():
    with pytest.raises(ValueError, match="not valid for kind"):
        Quantity(1.0, "energy", "bohr")


def test_quantity_converts_hartree_to_ev():
    # CODATA 2018: 1 Ha = 27.211386245988 eV
    q = Quantity(1.0, "energy").to("eV")
    assert q.unit == "eV"
    assert q.value == pytest.approx(27.211386245988)


def test_quantity_conversion_round_trips():
    original = Quantity(-76.02614, "energy")
    round_tripped = original.to("kcal/mol").to("eV").to("Ha")
    assert round_tripped.value == pytest.approx(original.value, rel=1e-12)


def test_quantity_to_same_unit_is_identity():
    q = Quantity(1.5, "energy")
    assert q.to("Ha") is q


def test_gradient_conversion_uses_both_factors():
    # 1 Ha/bohr = 27.211386245988 / 0.529177210903 eV/Angstrom
    q = Quantity(1.0, "gradient").to("eV/Angstrom")
    assert q.value == pytest.approx(27.211386245988 / 0.529177210903)


def test_known_units_lists_renderable_units():
    assert set(known_units("energy")) == {"Ha", "eV", "kcal/mol", "kJ/mol"}


def test_energy_delta_shares_the_energy_dimension_and_its_units():
    assert dimension_of("energy_delta") == "energy"
    assert known_units("energy_delta") == known_units("energy")


# --------------------------------------------------------------------
# FormatSpec / FormatPolicy
# --------------------------------------------------------------------


def test_default_energy_spec_matches_the_incumbent_scf_log_format():
    # output/formats/scf_log.py rendered energies as "%18.10f"; the default
    # policy reproduces it so adopting the policy is not also a restyle.
    rendered = DEFAULT_POLICY.render(Quantity(-76.02614083, "energy"))
    assert rendered == f"{-76.02614083:18.10f}"


def test_default_energy_delta_is_signed_scientific():
    rendered = DEFAULT_POLICY.render(Quantity(-1.234e-5, "energy_delta"))
    assert rendered == f"{-1.234e-5:+11.3e}"
    positive = DEFAULT_POLICY.render(Quantity(1.234e-5, "energy_delta"))
    assert positive.strip().startswith("+")


def test_precision_is_configurable_without_touching_the_caller():
    policy = DEFAULT_POLICY.with_spec("energy", precision=4, width=0)
    assert policy.render(Quantity(-76.02614083, "energy")) == "-76.0261"


def test_unit_is_configurable_and_conversion_happens_at_render():
    policy = DEFAULT_POLICY.with_unit("energy", "eV").with_spec(
        "energy", precision=4, width=0
    )
    # The caller still emits Hartree; only the render changes.
    assert policy.render(Quantity(1.0, "energy")) == "27.2114"
    assert policy.unit_of("energy") == "eV"


def test_switching_the_energy_unit_moves_energy_delta_with_it():
    # Regression: units are per-dimension, not per-kind. When these were
    # per-kind, `with_unit("energy", "eV")` left the dE column in Hartree
    # and a single SCF table rendered two different units side by side.
    policy = DEFAULT_POLICY.with_unit("energy", "eV")
    assert policy.unit_of("energy") == "eV"
    assert policy.unit_of("energy_delta") == "eV"
    # energy_delta renders at 3 significant digits, so compare loosely.
    rendered = policy.render(Quantity(1.0, "energy_delta"))
    assert float(rendered) == pytest.approx(27.211386245988, rel=1e-3)


def test_precision_stays_per_kind_even_though_units_are_per_dimension():
    policy = DEFAULT_POLICY.with_spec("energy", precision=2)
    assert policy.spec("energy").precision == 2
    assert policy.spec("energy_delta") == DEFAULT_POLICY.spec("energy_delta")


def test_with_spec_does_not_mutate_the_original_policy():
    derived = DEFAULT_POLICY.with_spec("energy", precision=2)
    assert derived.spec("energy").precision == 2
    assert DEFAULT_POLICY.spec("energy").precision == 10


def test_with_unit_does_not_mutate_the_original_policy():
    derived = DEFAULT_POLICY.with_unit("energy", "eV")
    assert derived.unit_of("energy") == "eV"
    assert DEFAULT_POLICY.unit_of("energy") == "Ha"


def test_policy_rejects_a_unit_from_the_wrong_dimension():
    with pytest.raises(ValueError, match="not valid for dimension"):
        DEFAULT_POLICY.with_unit("energy", "bohr")


def test_policy_rejects_an_unknown_dimension():
    with pytest.raises(ValueError, match="unknown dimension"):
        DEFAULT_POLICY.with_unit("charm", "Ha")


def test_policy_copies_its_mappings_so_default_policy_cannot_be_corrupted():
    # A frozen dataclass still hands out a mutable dict. Without the
    # defensive copy, mutating one policy's specs would restyle every
    # policy derived from the module defaults, process-wide.
    derived = DEFAULT_POLICY.with_spec("energy", precision=1)
    derived.specs["energy"] = FormatSpec(precision=99)
    derived.units["energy"] = "kJ/mol"
    assert DEFAULT_POLICY.spec("energy").precision == 10
    assert DEFAULT_POLICY.unit_of("energy") == "Ha"


def test_policy_rejects_a_kind_it_has_no_spec_for():
    sparse = FormatPolicy(specs={"dimensionless": FormatSpec(precision=2)})
    with pytest.raises(ValueError, match="no FormatSpec registered"):
        sparse.render(Quantity(1.0, "energy"))


def test_format_spec_validates_notation():
    with pytest.raises(ValueError, match="notation must be"):
        FormatSpec(precision=3, notation="q")


def test_format_spec_validates_precision_and_width():
    with pytest.raises(ValueError, match="precision must be"):
        FormatSpec(precision=-1)
    with pytest.raises(ValueError, match="width must be"):
        FormatSpec(precision=3, width=-2)


def test_policy_validates_unit_placement():
    with pytest.raises(ValueError, match="unit_placement must be"):
        FormatPolicy(specs=dict(DEFAULT_POLICY.specs), unit_placement="sideways")


# --------------------------------------------------------------------
# Table -- the primitive that retires the hardcoded rule widths
# --------------------------------------------------------------------


def test_table_rule_is_exactly_as_wide_as_the_table():
    table = Table(["iter", "energy"])
    table.add_row(1, Quantity(-76.02614083, "energy"))
    lines = table.render().splitlines()
    rule, row = lines[1], lines[2]
    # This is the invariant the "-" * 52 / * 54 / * 56 / * 70 constants
    # in the old writers each got wrong in a different way.
    assert len(rule) == len(row)
    assert set(rule.strip()) == {"-"}


def test_table_widens_columns_to_fit_the_widest_cell():
    table = Table(["n"])
    table.add_row("a-very-wide-cell")
    lines = table.render().splitlines()
    assert "a-very-wide-cell" in lines[2]
    assert len(lines[1]) == len(lines[2])


def test_table_footer_is_set_apart_by_a_rule_and_stays_aligned():
    table = Table(["elem", "charge"])
    table.add_row("O", "-0.68")
    table.add_row("H", "0.34")
    table.footer("sum", "0.00")
    lines = table.render().splitlines()
    # header, rule, O row, H row, rule, footer -> 6 lines; the footer is
    # preceded by a separator rule identical to the header rule.
    assert len(lines) == 6
    assert lines[1] == lines[4]                      # same rule above + below body
    assert set(lines[4].strip()) == {"-"}
    assert "sum" in lines[5] and "0.00" in lines[5]
    # all body/footer lines share the same width (footer counts toward it)
    assert len({len(l) for l in (lines[1], lines[2], lines[5])}) == 1


def test_table_footer_cell_widens_the_column():
    table = Table(["label", "v"])
    table.add_row("a", "1")
    table.footer("a-wide-footer-label", "2")
    lines = table.render().splitlines()
    assert "a-wide-footer-label" in lines[-1]
    assert len(lines[1]) == len(lines[-1])           # rule widened to fit footer


def test_table_widens_columns_to_fit_the_header():
    table = Table(["a-very-wide-header"])
    table.add_row("x")
    lines = table.render().splitlines()
    assert len(lines[1]) == len("  ") + len("a-very-wide-header")


def test_table_renders_quantities_through_the_policy():
    table = Table(["energy"])
    table.add_row(Quantity(1.0, "energy"))
    ev = DEFAULT_POLICY.with_unit("energy", "eV").with_spec(
        "energy", precision=3, width=0
    )
    assert "27.211" in table.render(ev)
    assert "1.0000000000" in table.render(DEFAULT_POLICY)


def test_table_never_mixes_units_across_energy_columns():
    table = Table(["energy", "dE"])
    table.add_row(Quantity(1.0, "energy"), Quantity(1.0, "energy_delta"))
    ev = DEFAULT_POLICY.with_unit("energy", "eV")
    cells = table.render(ev).splitlines()[2].split()
    # Both columns carry the same physical value, so under one policy they
    # must render to the same number. The dE column keeps its own 3-digit
    # scientific notation, hence the loose tolerance: what is pinned here
    # is that both converted to eV, not that they share a precision.
    assert float(cells[0]) == pytest.approx(float(cells[1]), rel=1e-3)


def test_table_renders_bools_as_yes_no_not_one_zero():
    # bool is an int subclass; a naive str() would print "True"/"1".
    table = Table(["converged"])
    table.add_row(True)
    table.add_row(False)
    body = table.render().splitlines()[2:]
    assert body[0].strip() == "yes"
    assert body[1].strip() == "no"


def test_table_rejects_wrong_cell_count():
    table = Table(["a", "b"])
    with pytest.raises(ValueError, match="row has 1 cells"):
        table.add_row(1)


def test_table_rejects_zero_columns():
    with pytest.raises(ValueError, match="at least one column"):
        Table([])


def test_table_with_no_rows_still_renders_header_and_rule():
    lines = Table(["iter", "energy"]).render().splitlines()
    assert len(lines) == 2
    assert set(lines[1].strip()) == {"-"}


def test_table_alignment_is_per_column():
    table = Table([Column("left", align="<"), Column("right", align=">")])
    table.add_row("a", "b")
    row = table.render().splitlines()[2]
    assert row.startswith("  a")
    assert row.rstrip().endswith("b")


def test_table_none_cell_renders_empty():
    table = Table(["dE"])
    table.add_row(None)
    assert table.render().splitlines()[2].strip() == ""


# --------------------------------------------------------------------
# CriteriaTable -- semantic value/threshold/status diagnostics
# --------------------------------------------------------------------


def test_criteria_table_renders_full_and_inline_forms():
    criteria = CriteriaTable().add(
        "gmax",
        Quantity(1.0e-3, "gradient"),
        Quantity(4.5e-4, "gradient"),
        False,
    )
    assert criteria.render_inline() == (
        "gmax=1.000e-03 <= 4.500e-04 Ha/bohr [fail]"
    )

    lines = criteria.render().splitlines()
    assert lines[0].split() == ["criterion", "value", "threshold", "unit", "status"]
    assert lines[2].split() == [
        "gmax",
        "1.000e-03",
        "4.500e-04",
        "Ha/bohr",
        "fail",
    ]
    assert len(lines[1]) == len(lines[0])


def test_criteria_table_converts_both_sides_with_the_policy():
    criteria = CriteriaTable().add(
        "gmax",
        Quantity(1.0e-3, "gradient"),
        Quantity(4.5e-4, "gradient"),
        True,
    )
    ev_per_angstrom = DEFAULT_POLICY.with_unit("gradient", "eV/Angstrom")
    assert criteria.render_inline(ev_per_angstrom) == (
        "gmax=5.142e-02 <= 2.314e-02 eV/Angstrom [pass]"
    )


def test_criteria_table_supports_missing_threshold_and_no_units():
    criteria = CriteriaTable(indent=0).add(
        "custom", Quantity(2.0), None, True
    )
    policy = DEFAULT_POLICY.with_unit_placement("none")
    assert criteria.render_inline(policy) == "custom=2.000000 [pass]"
    assert criteria.render(policy).splitlines()[0].split() == [
        "criterion",
        "value",
        "threshold",
        "status",
    ]


def test_criteria_table_validates_semantic_shape():
    with pytest.raises(ValueError, match="same quantity kind"):
        Criterion(
            "mixed",
            Quantity(1.0, "energy"),
            Quantity(1.0, "gradient"),
            False,
        )
    with pytest.raises(TypeError, match="Criterion objects"):
        CriteriaTable().extend(["not-a-criterion"])


# --------------------------------------------------------------------
# HeaderlessBlock -- titled label/value and matrix-shaped blocks
# --------------------------------------------------------------------


def test_headerless_rule_sizes_to_content_not_annotation():
    block = HeaderlessBlock(
        "Orbitals",
        [
            HeaderlessColumn(">", min_width=4),
            HeaderlessColumn(">", min_width=8),
        ],
        gutter=2,
    )
    block.add_row(1, "-0.5000", annotation="<-- a deliberately long HOMO marker")
    lines = block.render().splitlines()
    # 4 + gutter 2 + column 8; the right-hand marker is deliberately ragged
    # and must not extend the title rule.
    assert lines[1] == "  " + "-" * 14
    assert len(lines[2]) > len(lines[1])


def test_headerless_divider_and_footer_share_the_content_rule():
    block = HeaderlessBlock(
        "Matrix",
        [
            HeaderlessColumn("<", min_width=2),
            HeaderlessColumn(">", min_width=6),
            HeaderlessColumn(">", min_width=6),
        ],
        body_indent=2,
        gutter=1,
    )
    block.add_row("a1", "1.0", "0.0").divider()
    block.add_row("a2", "0.0", "1.0")
    block.footer("determinant = 1.0")
    lines = block.render().splitlines()
    assert lines[1] == lines[3]
    assert lines[-1].startswith("    determinant")
    assert len(lines[1]) >= len(lines[-1])


def test_headerless_footer_can_widen_the_rule():
    block = HeaderlessBlock("T", [HeaderlessColumn("<")])
    block.add_row("x").footer("a footer wider than the body")
    lines = block.render().splitlines()
    assert len(lines[1]) == len(lines[-1])


def test_headerless_quantities_follow_the_render_policy():
    block = HeaderlessBlock(
        "Energy components",
        [HeaderlessColumn("<"), HeaderlessColumn(">")],
        unit_for="energy",
    ).add_row("total", Quantity(1.0, "energy"))
    ev = DEFAULT_POLICY.with_unit("energy", "eV").with_spec(
        "energy", precision=3, width=0
    )
    rendered = block.render(ev)
    assert "Energy components (eV)" in rendered
    assert "27.211" in rendered


def test_headerless_validates_shapes_and_widths():
    with pytest.raises(ValueError, match="at least one column"):
        HeaderlessBlock("bad", [])
    with pytest.raises(ValueError, match="min_width"):
        HeaderlessColumn(min_width=-1)
    block = HeaderlessBlock("bad", [HeaderlessColumn(), HeaderlessColumn()])
    with pytest.raises(ValueError, match="row has 1 cells"):
        block.add_row("one")
    with pytest.raises(ValueError, match="footer has 3 cells"):
        block.footer("one", "two", "three")


# --------------------------------------------------------------------
# OutputDocument
# --------------------------------------------------------------------


def test_section_rule_matches_the_title_width():
    doc = OutputDocument().section("Energy components")
    lines = doc.render().splitlines()
    assert lines[0] == "  Energy components"
    assert lines[1] == "  " + "-" * len("Energy components")


def test_section_unit_label_follows_the_policy():
    doc = OutputDocument().section("Energy components", unit_for="energy")
    assert "Energy components (Ha)" in doc.render()
    ev = DEFAULT_POLICY.with_unit("energy", "eV")
    assert "Energy components (eV)" in doc.render(ev)


def test_section_unit_label_absent_when_units_go_on_rows():
    doc = OutputDocument().section("Energy components", unit_for="energy")
    row_policy = DEFAULT_POLICY.with_unit_placement("row")
    assert "(Ha)" not in doc.render(row_policy)


def test_scalar_unit_placement_row_appends_the_unit():
    doc = OutputDocument().scalar("Total energy", Quantity(-76.0, "energy"))
    row_policy = DEFAULT_POLICY.with_unit_placement("row")
    assert doc.render(row_policy).rstrip().endswith("Ha")
    # ...and the default header placement leaves the row bare.
    assert not doc.render(DEFAULT_POLICY).rstrip().endswith("Ha")


def test_scalar_labels_are_left_aligned_into_a_fixed_column():
    doc = OutputDocument()
    doc.scalar("Nuclear repulsion", Quantity(9.1671, "energy"))
    doc.scalar("Total energy", Quantity(-76.0261, "energy"))
    first, second = doc.render().splitlines()
    # The numbers line up because the label column is fixed-width.
    assert first.index(".") == second.index(".")


def test_document_render_is_policy_pure_and_repeatable():
    doc = OutputDocument().scalar("E", Quantity(1.0, "energy"))
    assert doc.render() == doc.render()
    ev = DEFAULT_POLICY.with_unit("energy", "eV").with_spec(
        "energy", precision=3, width=0
    )
    assert doc.render(ev) != doc.render(DEFAULT_POLICY)
    # Rendering under one policy must not mutate the document.
    assert doc.render(DEFAULT_POLICY) == doc.render()


def test_document_drops_the_leading_blank_from_an_opening_section():
    assert not OutputDocument().section("Title").render().startswith("\n")


def test_document_composes_sections_tables_and_scalars_in_order():
    table = Table(["iter", "energy"]).add_row(1, Quantity(-1.0, "energy"))
    doc = (
        OutputDocument()
        .section("SCF iterations")
        .table(table)
        .section("Energy components", unit_for="energy")
        .scalar("Total energy", Quantity(-1.0, "energy"))
    )
    out = doc.render()
    assert out.index("SCF iterations") < out.index("iter")
    assert out.index("iter") < out.index("Energy components (Ha)")
    assert out.index("Energy components (Ha)") < out.index("Total energy")


def test_text_block_is_spliced_verbatim():
    doc = OutputDocument().text("  raw   spacing  preserved")
    assert doc.render() == "  raw   spacing  preserved"


def test_no_line_has_trailing_whitespace():
    table = Table([Column("a", align="<"), Column("b", align="<")])
    table.add_row("x", "y")
    doc = OutputDocument().section("S").table(table).scalar("label", "v")
    for line in doc.render().splitlines():
        assert line == line.rstrip(), repr(line)


def test_nan_and_inf_render_without_raising():
    # A diverging SCF can hand the writer a non-finite energy; the writer
    # must not be the thing that crashes the run.
    doc = OutputDocument()
    doc.scalar("E", Quantity(math.nan, "energy"))
    doc.scalar("dE", Quantity(math.inf, "energy_delta"))
    out = doc.render()
    assert "nan" in out
    assert "inf" in out


# --------------------------------------------------------------------
# active_policy / output_units -- the process-wide render policy
# --------------------------------------------------------------------


@pytest.fixture
def _restore_units():
    from vibeqc.output import output_units
    saved = output_units()
    try:
        yield
    finally:
        output_units(saved)


def test_active_policy_defaults_to_hartree():
    from vibeqc.output import active_policy, output_units
    assert output_units() == "Ha"
    assert active_policy().unit_of("energy") == "Ha"


def test_output_units_switches_energy_unit(_restore_units):
    from vibeqc.output import active_policy, output_units, Quantity
    assert output_units("eV") == "eV"
    # 1 Ha renders as ~27.211 eV through the active policy.
    rendered = active_policy().render(Quantity(1.0, "energy"))
    assert float(rendered) == pytest.approx(27.211386245988, rel=1e-6)


def test_output_units_is_byte_identical_to_default_at_hartree():
    # The default must render exactly like the historical f"{e:18.10f}".
    from vibeqc.output import active_policy, Quantity
    assert active_policy().render(Quantity(-76.02614083, "energy")) == \
        f"{-76.02614083:18.10f}"


def test_set_active_policy_replaces_the_whole_policy(_restore_units):
    from vibeqc.output import DEFAULT_POLICY, active_policy, set_active_policy
    p = DEFAULT_POLICY.with_unit("energy", "kcal/mol").with_spec("energy", precision=2)
    set_active_policy(p)
    assert active_policy().unit_of("energy") == "kcal/mol"
    set_active_policy(DEFAULT_POLICY)


def test_env_var_sets_units_at_import():
    import subprocess, sys, os, textwrap
    code = textwrap.dedent(
        """
        from vibeqc.output import output_units
        assert output_units() == "eV", output_units()
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={**os.environ, "VIBEQC_OUTPUT_UNITS": "eV"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_render_energy_is_byte_identical_at_default_for_any_width():
    from vibeqc.output import render_energy
    e = -76.02614083
    for w, p in [(18, 10), (16, 10), (14, 8), (20, 10)]:
        assert render_energy(e, width=w, precision=p) == f"{e:{w}.{p}f}"
    assert render_energy(e, width=0, precision=10) == f"{e:.10f}"  # no padding


def test_render_energy_converts_in_ev(_restore_units):
    from vibeqc.output import render_energy, output_units
    output_units("eV")
    assert float(render_energy(1.0, width=0, precision=6)) == pytest.approx(
        27.211386245988, rel=1e-6
    )


def test_render_energy_explicit_unit_supports_dual_unit_columns(_restore_units):
    from vibeqc.output import render_energy, output_units

    output_units("kJ/mol")
    assert render_energy(1.0, width=0, precision=3, unit="Ha") == "1.000"
    assert float(render_energy(1.0, width=0, precision=6, unit="eV")) == (
        pytest.approx(27.211386245988, rel=1e-6)
    )


def test_render_energy_labeled_is_byte_identical_at_default():
    from vibeqc.output import render_energy_labeled
    e = -76.02614083
    for w, p in [(18, 10), (16, 10), (14, 8), (20, 10)]:
        assert render_energy_labeled(e, width=w, precision=p) == f"{e:{w}.{p}f} Ha"


def test_render_energy_labeled_moves_value_and_unit_together(_restore_units):
    from vibeqc.output import render_energy_labeled, output_units
    output_units("eV")
    out = render_energy_labeled(1.0, width=0, precision=6)
    number, unit = out.rsplit(" ", 1)
    assert unit == "eV"                                    # label followed the value
    assert float(number) == pytest.approx(27.211386245988, rel=1e-6)


def test_render_duration_is_byte_identical_at_default():
    from vibeqc.output import render_duration
    for w, p in [(12, 3), (0, 2), (8, 1)]:
        assert render_duration(3.14159, width=w, precision=p) == f"{3.14159:{w}.{p}f}"
    # NaN (unmeasured avg-per-iter) renders the same as the bare f-string.
    assert render_duration(float("nan")) == f"{float('nan'):12.3f}"


def test_render_frequency_is_byte_identical_at_default():
    from vibeqc.output import render_frequency
    for f in (1729.03, 0.04, 3712.5):
        assert render_frequency(f) == f"{f:.1f}"
        assert render_frequency(f, width=10, precision=2) == f"{f:10.2f}"


def test_render_frequency_converts_to_thz(_restore_units):
    from vibeqc.output import render_frequency, set_active_policy, active_policy, DEFAULT_POLICY
    saved = active_policy()
    try:
        set_active_policy(DEFAULT_POLICY.with_unit("wavenumber", "THz"))
        # 1000 cm-1 = c[cm/ps] * 1000 = 29.9792458 THz (render at enough
        # precision that the string rounding doesn't dominate the tolerance)
        assert float(render_frequency(1000.0, precision=6)) == pytest.approx(29.9792458)
    finally:
        set_active_policy(saved)


def test_render_temperature_is_byte_identical():
    from vibeqc.output import render_temperature
    assert render_temperature(298.15, precision=2) == f"{298.15:.2f}"
    assert render_temperature(1000.0, width=8, precision=1) == f"{1000.0:8.1f}"


def test_render_duration_converts_to_milliseconds():
    from vibeqc.output import (
        render_duration, set_active_policy, active_policy, DEFAULT_POLICY,
    )
    saved = active_policy()
    try:
        set_active_policy(DEFAULT_POLICY.with_unit("time", "ms"))
        assert float(render_duration(1.5, width=0, precision=1)) == pytest.approx(1500.0)
    finally:
        set_active_policy(saved)  # full restore -- _restore_units only resets energy
