"""Semantic document primitives for vibe-qc's human-readable output.

Every user-facing byte vibe-qc emits is formatted in Python: the C++ core
populates result structs and hands numbers across pybind11, it never
prints. This module is therefore the single place where a number becomes
text, and the single place a table decides how wide its rule is.

The problem it solves
---------------------

Before this module, the same physical quantity was formatted at five
different widths (``16.10f``, ``18.10f``, ``20.10f``, ``14.8f``, ``.12f``)
under four different labels (``Total energy`` / ``E_total`` / ``E(total)``
/ ``E_tot``), and horizontal rules were hardcoded as ``"-" * 52`` in one
module and ``"-" * 56`` in its neighbour. The two SCF-trace renderers
(the ``.out`` string builder and the live progress logger) were kept in
sync by a docstring promising they matched. They did not.

The fix is to stop passing pre-formatted strings around. A module emits a
:class:`Quantity` (a number, its kind, its unit); a :class:`FormatPolicy`
decides how that kind renders; a :class:`Table` computes its own widths.
Formatting is therefore defined exactly once, and is configurable without
touching the module that produced the number.

Units
-----

Values are carried in the unit they were produced in and converted only
at render time. Atomic units are canonical for every kind, matching what
the C++ core returns. A :class:`FormatPolicy` may display any unit
registered for the kind; nothing downstream has to know that happened.

Typical usage::

    doc = OutputDocument()
    doc.section("Energy components")
    doc.scalar("Nuclear repulsion", Quantity(9.1671, "energy"))
    doc.scalar("Total energy", Quantity(-76.0263, "energy"))
    print(doc.render())

To render the same document in eV at 6 decimals, pass a policy rather
than editing the emitting module::

    policy = (DEFAULT_POLICY
              .with_unit("energy", "eV")
              .with_spec("energy", precision=6))
    doc.render(policy)

Units are chosen per *dimension* and precision per *kind*, so
``with_unit("energy", "eV")`` moves total energies and SCF energy changes
together. A table cannot end up with a Hartree dE column beside an eV
energy column.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence, Union


__all__ = [
    "Quantity",
    "FormatSpec",
    "FormatPolicy",
    "DEFAULT_POLICY",
    "Column",
    "HeaderlessColumn",
    "HeaderlessBlock",
    "Table",
    "Criterion",
    "CriteriaTable",
    "OutputDocument",
    "active_policy",
    "output_units",
    "set_active_policy",
    "render_duration",
    "render_energy",
    "render_energy_labeled",
    "render_frequency",
    "render_temperature",
    "canonical_unit",
    "dimension_of",
    "known_units",
]


# CODATA 2018. Part of the field's shared background; no per-value
# citation is required (CLAUDE.md Sec. 8).
_HARTREE_TO_EV = 27.211386245988
_HARTREE_TO_KCAL_MOL = 627.5094740631
_HARTREE_TO_KJ_MOL = 2625.4996394799
_BOHR_TO_ANGSTROM = 0.529177210903
# Wavenumber (cm-1) conversions: nu[THz] = c[cm/ps]*nu~ ; E[meV] = hc*nu~.
_CM1_TO_THZ = 0.0299792458
_CM1_TO_MEV = 0.1239841984

# A *dimension* is a physical quantity (energy, length); a *kind* is a
# role a dimension plays in the output (a total energy and an SCF energy
# change are both energies, but one renders fixed-point and the other
# signed-scientific). Units are chosen per dimension and precision per
# kind: that split is what makes it structurally impossible to render a
# total energy in eV while its dE column stays in Hartree.
_DIMENSION_OF: dict[str, str] = {
    "energy": "energy",
    "energy_delta": "energy",
    "gradient": "gradient",
    "length": "length",
    "temperature": "temperature",
    "duration": "time",
    "frequency": "wavenumber",
    "dimensionless": "dimensionless",
}

# Per dimension: every renderable unit, expressed as
# ``value_in_unit = value_in_canonical * factor``. The first entry is
# canonical, and is what the C++ core returns.
_UNITS: dict[str, dict[str, float]] = {
    "energy": {
        "Ha": 1.0,
        "eV": _HARTREE_TO_EV,
        "kcal/mol": _HARTREE_TO_KCAL_MOL,
        "kJ/mol": _HARTREE_TO_KJ_MOL,
    },
    "gradient": {
        "Ha/bohr": 1.0,
        "eV/Angstrom": _HARTREE_TO_EV / _BOHR_TO_ANGSTROM,
    },
    "length": {
        "bohr": 1.0,
        "Angstrom": _BOHR_TO_ANGSTROM,
    },
    "temperature": {"K": 1.0},
    # Wall-clock durations: seconds canonical (what perf_counter deltas
    # carry); ms / min available for VIBEQC_OUTPUT_UNITS=time:ms etc.
    "time": {"s": 1.0, "ms": 1000.0, "min": 1.0 / 60.0},
    # Vibrational wavenumbers: cm-1 canonical (what the Hessian returns);
    # THz / meV available for VIBEQC_OUTPUT_UNITS=wavenumber:THz etc.
    "wavenumber": {"cm-1": 1.0, "THz": _CM1_TO_THZ, "meV": _CM1_TO_MEV},
    "dimensionless": {"": 1.0},
}


def dimension_of(kind: str) -> str:
    """Return the physical dimension ``kind`` measures."""
    try:
        return _DIMENSION_OF[kind]
    except KeyError:
        raise ValueError(
            f"unknown quantity kind {kind!r}; known kinds: "
            f"{', '.join(sorted(_DIMENSION_OF))}"
        ) from None


def canonical_unit(kind: str) -> str:
    """Return the internal unit ``kind`` is carried in."""
    return next(iter(_UNITS[dimension_of(kind)]))


def known_units(kind: str) -> tuple[str, ...]:
    """Return every unit ``kind`` can be rendered in."""
    return tuple(_UNITS[dimension_of(kind)])


@dataclass(frozen=True)
class Quantity:
    """A number that knows what it is and what unit it is in.

    ``kind`` selects the formatting rules and the legal units;
    ``unit`` defaults to the canonical unit for that kind, which is what
    the C++ core returns for every quantity it computes.

    ``energy_delta`` is a distinct kind from ``energy`` even though both
    are Hartree: an SCF energy change is universally rendered in signed
    scientific notation, a total energy in fixed-point. Keeping them
    apart is what lets a policy restyle one without disturbing the other.
    """

    value: float
    kind: str = "dimensionless"
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        canonical = canonical_unit(self.kind)
        if self.unit is None:
            object.__setattr__(self, "unit", canonical)
        elif self.unit not in known_units(self.kind):
            raise ValueError(
                f"unit {self.unit!r} is not valid for kind {self.kind!r}; "
                f"valid units: {', '.join(known_units(self.kind))}"
            )
        object.__setattr__(self, "value", float(self.value))

    def to(self, unit: str) -> "Quantity":
        """Return an equal quantity expressed in ``unit``."""
        if unit not in known_units(self.kind):
            raise ValueError(
                f"unit {unit!r} is not valid for kind {self.kind!r}; "
                f"valid units: {', '.join(known_units(self.kind))}"
            )
        if unit == self.unit:
            return self
        factors = _UNITS[dimension_of(self.kind)]
        canonical_value = self.value / factors[self.unit]
        return Quantity(canonical_value * factors[unit], self.kind, unit)


@dataclass(frozen=True)
class FormatSpec:
    """How one kind of quantity renders: precision, width, notation, sign.

    ``width`` of 0 means "as wide as the number needs"; a positive width
    pads into a fixed column, which is what keeps a table's numbers
    aligned under their header.

    Note there is no ``unit`` field. Units belong to the dimension, not
    the kind, and live on :class:`FormatPolicy`.
    """

    precision: int
    width: int = 0
    notation: str = "f"
    sign: bool = False

    def __post_init__(self) -> None:
        if self.notation not in ("f", "e", "g"):
            raise ValueError(
                f"notation must be one of 'f', 'e', 'g'; got {self.notation!r}"
            )
        if self.precision < 0:
            raise ValueError(f"precision must be >= 0; got {self.precision}")
        if self.width < 0:
            raise ValueError(f"width must be >= 0; got {self.width}")

    def render_value(self, value: float) -> str:
        """Render a bare number already expressed in the display unit."""
        sign = "+" if self.sign else ""
        width = f"{self.width}" if self.width else ""
        return f"{value:{sign}{width}.{self.precision}{self.notation}}"


# The defaults reproduce the format the tree already converged on most
# often, so that adopting the policy is not also a gratuitous restyle:
# energy at 18.10f (output/formats/scf_log.py), dE signed at .3e.
_DEFAULT_SPECS: dict[str, FormatSpec] = {
    "energy": FormatSpec(precision=10, width=18),
    "energy_delta": FormatSpec(precision=3, width=11, notation="e", sign=True),
    "gradient": FormatSpec(precision=3, width=10, notation="e"),
    "length": FormatSpec(precision=6, width=12),
    "temperature": FormatSpec(precision=1, width=8),
    "duration": FormatSpec(precision=3, width=12),
    "frequency": FormatSpec(precision=1, width=0),
    "dimensionless": FormatSpec(precision=6, width=0),
}

#: Display unit per dimension. Canonical (atomic units) by default.
_DEFAULT_UNITS: dict[str, str] = {
    dim: next(iter(units)) for dim, units in _UNITS.items()
}


@dataclass(frozen=True)
class FormatPolicy:
    """The precision / width / unit choice for every kind of quantity.

    A policy is immutable; :meth:`with_spec` returns a modified copy. This
    is what makes the rendered precision and units configurable without
    the module that produced a number knowing anything about it.
    """

    #: Precision / width / notation, per kind.
    specs: Mapping[str, FormatSpec]
    #: Display unit, per *dimension*. Setting ``energy`` to ``eV`` moves
    #: total energies and SCF energy changes together, so a table can
    #: never mix Hartree and eV across its columns.
    units: Mapping[str, str] = None  # type: ignore[assignment]
    #: Where a unit label is written: on each row, once in the section
    #: header, or nowhere. The molecular and periodic writers historically
    #: disagreed on this (``Energy components (Ha)`` with bare rows, versus
    #: a bare header with ``E_total (Ha)`` rows); making it a policy
    #: setting turns that into a decision rather than an accident.
    unit_placement: str = "header"

    def __post_init__(self) -> None:
        # Copy both mappings: a frozen dataclass still exposes a mutable
        # dict, and DEFAULT_POLICY would otherwise alias the module-level
        # defaults, so one caller's `policy.specs[k] = ...` would silently
        # restyle every subsequent run in the process.
        object.__setattr__(
            self, "units", dict(_DEFAULT_UNITS if self.units is None else self.units)
        )
        object.__setattr__(self, "specs", dict(self.specs))
        for dim, unit in self.units.items():
            if dim not in _UNITS:
                raise ValueError(f"unknown dimension {dim!r}")
            if unit not in _UNITS[dim]:
                raise ValueError(
                    f"unit {unit!r} is not valid for dimension {dim!r}; "
                    f"valid units: {', '.join(_UNITS[dim])}"
                )
        if self.unit_placement not in ("header", "row", "none"):
            raise ValueError(
                "unit_placement must be one of 'header', 'row', 'none'; "
                f"got {self.unit_placement!r}"
            )

    def spec(self, kind: str) -> FormatSpec:
        try:
            return self.specs[kind]
        except KeyError:
            raise ValueError(
                f"no FormatSpec registered for kind {kind!r}; known kinds: "
                f"{', '.join(sorted(self.specs))}"
            ) from None

    def unit_of(self, kind: str) -> str:
        """The unit label ``kind`` renders under."""
        return self.units.get(dimension_of(kind), canonical_unit(kind))

    def render(self, quantity: Quantity) -> str:
        """Render ``quantity`` as a bare number, converted per this policy."""
        spec = self.spec(quantity.kind)
        converted = quantity.to(self.unit_of(quantity.kind))
        return spec.render_value(converted.value)

    def with_spec(self, kind: str, **overrides) -> "FormatPolicy":
        """Return a copy with ``kind``'s precision / width / notation replaced.

        ``DEFAULT_POLICY.with_spec("energy", precision=6, width=0)``
        """
        merged = dict(self.specs)
        merged[kind] = replace(self.spec(kind), **overrides)
        return replace(self, specs=merged)

    def with_unit(self, dimension: str, unit: str) -> "FormatPolicy":
        """Return a copy rendering ``dimension`` in ``unit``.

        ``DEFAULT_POLICY.with_unit("energy", "eV")`` moves every energy
        kind at once.
        """
        merged = dict(self.units)
        merged[dimension] = unit
        return replace(self, units=merged)

    def with_unit_placement(self, placement: str) -> "FormatPolicy":
        return replace(self, unit_placement=placement)


DEFAULT_POLICY = FormatPolicy(specs=_DEFAULT_SPECS, units=_DEFAULT_UNITS)


# ---------------------------------------------------------------------------
# The active policy -- one process-wide FormatPolicy the runners render
# through, so display units and precision are configurable in one place
# instead of baked into every f-string. Defaults render byte-identically to
# DEFAULT_POLICY, so the .out is unchanged until a user moves a knob.
# ---------------------------------------------------------------------------

import os as _os  # local alias; document.py is otherwise dependency-free


def _policy_from_env() -> "FormatPolicy":
    """Build the active policy from ``VIBEQC_OUTPUT_UNITS`` (per dimension).

    ``VIBEQC_OUTPUT_UNITS=eV`` displays energies in eV;
    ``VIBEQC_OUTPUT_UNITS=energy:eV,length:Angstrom`` sets several. Unset or
    unrecognised leaves the canonical (atomic) units, so the ``.out`` is
    unchanged by default.
    """
    raw = _os.environ.get("VIBEQC_OUTPUT_UNITS", "").strip()
    if not raw:
        return DEFAULT_POLICY
    policy = DEFAULT_POLICY
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        dim, _, unit = token.partition(":")
        if not unit:  # bare unit -> apply to energy (the common case)
            dim, unit = "energy", dim
        try:
            policy = policy.with_unit(dim.strip(), unit.strip())
        except ValueError:
            continue  # ignore an unknown dimension/unit rather than crash a run
    return policy


_ACTIVE_POLICY = _policy_from_env()


def active_policy() -> "FormatPolicy":
    """The process-wide :class:`FormatPolicy` the output layer renders
    through. Defaults to :data:`DEFAULT_POLICY` (or the
    ``VIBEQC_OUTPUT_UNITS`` override)."""
    return _ACTIVE_POLICY


def output_units(unit: Optional[str] = None, *, dimension: str = "energy") -> str:
    """Get or set the active display unit for ``dimension`` (returns it).

    ``output_units("eV")`` renders energies in eV everywhere the runners
    route through :func:`active_policy`; ``output_units()`` reads the
    current energy unit.
    """
    global _ACTIVE_POLICY
    if unit is not None:
        _ACTIVE_POLICY = _ACTIVE_POLICY.with_unit(dimension, unit)
    return _ACTIVE_POLICY.unit_of(dimension)


def set_active_policy(policy: "FormatPolicy") -> None:
    """Replace the active policy wholesale (precision + units)."""
    global _ACTIVE_POLICY
    _ACTIVE_POLICY = policy


def render_energy(
    value: float,
    *,
    width: int = 18,
    precision: int = 10,
    sign: bool = False,
    unit: Optional[str] = None,
) -> str:
    """Render an energy at a caller-chosen ``width`` / ``precision``, in the
    active display unit (converting from Hartree).

    ``unit`` overrides the active energy unit for deliberate dual-unit
    displays such as orbital energies in parallel Hartree and eV columns.

    The per-site converter for the ~80 scattered ``f"{e:16.10f} Ha"``-style
    energy strings: it keeps each site's own field shape, so at the default
    (Hartree) unit it is **byte-identical** to ``f"{value:{width}.{precision}f}"``
    -- the ``.out`` is unchanged -- while ``VIBEQC_OUTPUT_UNITS=eV`` converts
    the value in place. Pair it with :func:`output_units` (or
    ``active_policy().unit_of("energy")``) for the unit label. ``width=0``
    emits no field padding (matches a bare ``f"{e:.10f}"``).
    """
    policy = active_policy().with_spec(
        "energy", width=width, precision=precision, sign=sign
    )
    if unit is not None:
        policy = policy.with_unit("energy", unit)
    return policy.render(Quantity(value, "energy"))


def render_energy_labeled(
    value: float, *, width: int = 18, precision: int = 10, sign: bool = False
) -> str:
    """:func:`render_energy` with the active energy unit appended.

    Returns ``"<number> <unit>"`` -- byte-identical to
    ``f"{value:{width}.{precision}f} Ha"`` at the default (Hartree), and in
    ``VIBEQC_OUTPUT_UNITS=eV`` the value **and** the label move together
    (``"... eV"``, not the ``"... Ha"`` mismatch that converting only the
    number would leave). This is the one-call form for the ``f"{e:16.10f}
    Ha"`` summary strings scattered through the runners: keep the site's own
    field shape, drop the hardcoded unit. The single space between number
    and unit matches the historical literal.
    """
    number = render_energy(value, width=width, precision=precision, sign=sign)
    return f"{number} {active_policy().unit_of('energy')}"


def render_duration(
    value: float, *, width: int = 12, precision: int = 3, sign: bool = False
) -> str:
    """Render a wall-clock ``value`` (seconds) at a caller-chosen field shape,
    in the active time unit.

    The timings-block counterpart of :func:`render_energy`: it keeps each
    site's own ``width`` / ``precision``, so at the default (seconds) it is
    **byte-identical** to ``f"{value:{width}.{precision}f}"`` -- the ``.out``
    timings are unchanged -- while ``VIBEQC_OUTPUT_UNITS=time:ms`` converts in
    place. The unit label (``seconds``) lives in the block header, so callers
    render bare numbers through this.
    """
    spec = active_policy().with_spec(
        "duration", width=width, precision=precision, sign=sign
    )
    return spec.render(Quantity(value, "duration"))


def render_frequency(
    value: float, *, width: int = 0, precision: int = 1, sign: bool = False
) -> str:
    """Render a vibrational wavenumber ``value`` (cm-1) at a caller-chosen
    field shape, in the active wavenumber unit.

    Byte-identical to ``f"{value:{width}.{precision}f}"`` at the default
    (cm-1) -- so the vibrational block is unchanged -- while
    ``VIBEQC_OUTPUT_UNITS=wavenumber:THz`` (or ``:meV``) converts in place.
    The caller keeps the imaginary-mode ``i`` suffix and any string padding;
    this renders the bare magnitude.
    """
    spec = active_policy().with_spec(
        "frequency", width=width, precision=precision, sign=sign
    )
    return spec.render(Quantity(value, "frequency"))


def render_temperature(
    value: float, *, width: int = 0, precision: int = 1, sign: bool = False
) -> str:
    """Render a ``value`` in kelvin at a caller-chosen field shape.

    Byte-identical to ``f"{value:{width}.{precision}f}"`` (kelvin is the only
    registered unit), routed through the policy so the thermochemistry block's
    temperature shares the single-source precision control rather than a bare
    f-string.
    """
    spec = active_policy().with_spec(
        "temperature", width=width, precision=precision, sign=sign
    )
    return spec.render(Quantity(value, "temperature"))


Cell = Union[str, int, float, Quantity, None]


@dataclass(frozen=True)
class Column:
    """One table column. Width is computed from the content, never fixed.

    ``align`` follows format-spec syntax: ``<`` left, ``>`` right, ``^``
    centre. Numbers default to right so their decimal points line up.
    """

    header: str
    align: str = ">"

    def __post_init__(self) -> None:
        if self.align not in ("<", ">", "^"):
            raise ValueError(f"align must be '<', '>' or '^'; got {self.align!r}")


@dataclass(frozen=True)
class HeaderlessColumn:
    """One fixed-shape column in a :class:`HeaderlessBlock`.

    Unlike :class:`Column`, it has no header. ``min_width`` preserves a
    domain-specific field shape while still widening for longer content;
    ``gutter_after`` overrides the block's default gutter after this column.
    """

    align: str = ">"
    min_width: int = 0
    gutter_after: Optional[int] = None

    def __post_init__(self) -> None:
        if self.align not in ("<", ">", "^"):
            raise ValueError(f"align must be '<', '>' or '^'; got {self.align!r}")
        if self.min_width < 0:
            raise ValueError(f"min_width must be >= 0; got {self.min_width}")
        if self.gutter_after is not None and self.gutter_after < 0:
            raise ValueError(
                f"gutter_after must be >= 0; got {self.gutter_after}"
            )


class HeaderlessBlock:
    """A titled, content-sized block without semantic column headers.

    Rows may represent label/value pairs, matrix entries, or an explicitly
    added display-heading row. Rules size to the title, regular cells, and
    optional footer. A right-hand annotation is appended to a row but excluded
    from rule sizing, which keeps ragged frontier-orbital markers outside the
    tabular body.
    """

    def __init__(
        self,
        title: str,
        columns: Sequence[Union[HeaderlessColumn, str]],
        *,
        indent: int = 2,
        body_indent: int = 0,
        gutter: int = 2,
        annotation_gutter: int = 2,
        rule_char: str = "-",
        unit_for: Optional[str] = None,
    ) -> None:
        if not columns:
            raise ValueError("a headerless block needs at least one column")
        if indent < 0 or body_indent < 0 or gutter < 0 or annotation_gutter < 0:
            raise ValueError("indent and gutter widths must be >= 0")
        if len(rule_char) != 1:
            raise ValueError("rule_char must be exactly one character")
        self.title = str(title)
        self.columns: list[HeaderlessColumn] = [
            c if isinstance(c, HeaderlessColumn) else HeaderlessColumn(c)
            for c in columns
        ]
        self.indent = indent
        self.body_indent = body_indent
        self.gutter = gutter
        self.annotation_gutter = annotation_gutter
        self.rule_char = rule_char
        self.unit_for = unit_for
        self._items: list[tuple[str, object]] = []
        self._footer: Optional[list[Cell]] = None

    def add_row(
        self,
        *cells: Cell,
        annotation: Optional[str] = None,
    ) -> "HeaderlessBlock":
        if len(cells) != len(self.columns):
            raise ValueError(
                f"row has {len(cells)} cells but the block has "
                f"{len(self.columns)} columns"
            )
        self._items.append(("row", (list(cells), annotation)))
        return self

    def extend(self, rows: Iterable[Sequence[Cell]]) -> "HeaderlessBlock":
        for row in rows:
            self.add_row(*row)
        return self

    def divider(self) -> "HeaderlessBlock":
        """Append a content-width rule inside or after the body."""
        self._items.append(("divider", None))
        return self

    def footer(self, *cells: Cell) -> "HeaderlessBlock":
        """Set an optional footer.

        One cell spans the body (useful for a gap or summary sentence).
        Otherwise the footer must have the same shape as a regular row.
        """
        if len(cells) not in (1, len(self.columns)):
            raise ValueError(
                f"footer has {len(cells)} cells but expected one spanning "
                f"cell or {len(self.columns)} columns"
            )
        self._footer = list(cells)
        return self

    @staticmethod
    def _render_cell(cell: Cell, policy: FormatPolicy) -> str:
        if cell is None:
            return ""
        if isinstance(cell, Quantity):
            return policy.render(cell)
        if isinstance(cell, bool):
            return "yes" if cell else "no"
        if isinstance(cell, float):
            return policy.render(Quantity(cell, "dimensionless"))
        return str(cell)

    def _title_for(self, policy: FormatPolicy) -> str:
        title = self.title
        if self.unit_for is not None and policy.unit_placement == "header":
            unit = policy.unit_of(self.unit_for)
            if unit:
                title = f"{title} ({unit})"
        return title

    def _gutters(self) -> list[int]:
        return [
            self.gutter if col.gutter_after is None else col.gutter_after
            for col in self.columns[:-1]
        ]

    def render(self, policy: FormatPolicy = DEFAULT_POLICY) -> str:
        rendered_items: list[tuple[str, object]] = []
        rows_for_sizing: list[list[str]] = []
        for kind, payload in self._items:
            if kind == "row":
                cells, annotation = payload
                rendered = [self._render_cell(cell, policy) for cell in cells]
                rows_for_sizing.append(rendered)
                rendered_items.append((kind, (rendered, annotation)))
            else:
                rendered_items.append((kind, payload))

        footer: Optional[list[str]] = None
        if self._footer is not None:
            footer = [self._render_cell(cell, policy) for cell in self._footer]
            if len(footer) == len(self.columns):
                rows_for_sizing.append(footer)

        widths = [
            max(col.min_width, *(len(row[i]) for row in rows_for_sizing))
            if rows_for_sizing
            else col.min_width
            for i, col in enumerate(self.columns)
        ]
        gutters = self._gutters()
        tabular_width = sum(widths) + sum(gutters)
        title = self._title_for(policy)
        content_width = max(len(title), self.body_indent + tabular_width)
        if footer is not None and len(footer) == 1:
            content_width = max(
                content_width,
                self.body_indent + len(footer[0]),
            )

        pad = " " * self.indent
        body_pad = pad + " " * self.body_indent
        rule = pad + self.rule_char * content_width

        def row_text(cells: list[str], annotation: Optional[str]) -> str:
            parts: list[str] = []
            for i, cell in enumerate(cells):
                parts.append(f"{cell:{self.columns[i].align}{widths[i]}}")
                if i < len(gutters):
                    parts.append(" " * gutters[i])
            base = (body_pad + "".join(parts)).rstrip()
            if annotation:
                base += " " * self.annotation_gutter + str(annotation)
            return base

        lines = [pad + title, rule]
        for kind, payload in rendered_items:
            if kind == "divider":
                lines.append(rule)
            else:
                cells, annotation = payload
                lines.append(row_text(cells, annotation))
        if footer is not None:
            if len(footer) == 1:
                lines.append((body_pad + footer[0]).rstrip())
            else:
                lines.append(row_text(footer, None))
        return "\n".join(lines)


class Table:
    """A text table that computes its own column widths and rule length.

    This is the primitive that retires the hardcoded ``"-" * 52`` /
    ``"-" * 54`` / ``"-" * 56`` / ``"-" * 70`` rules scattered across the
    writers, each of which was a different guess at the same table's width.
    """

    def __init__(
        self,
        columns: Sequence[Union[Column, str]],
        *,
        indent: int = 2,
        gutter: int = 2,
        rule_char: str = "-",
    ) -> None:
        if not columns:
            raise ValueError("a table needs at least one column")
        self.columns: list[Column] = [
            c if isinstance(c, Column) else Column(c) for c in columns
        ]
        self.indent = indent
        self.gutter = gutter
        self.rule_char = rule_char
        self._rows: list[list[Cell]] = []
        self._footer: Optional[list[Cell]] = None

    def add_row(self, *cells: Cell) -> "Table":
        """Append one row. Returns self so calls can chain."""
        if len(cells) != len(self.columns):
            raise ValueError(
                f"row has {len(cells)} cells but the table has "
                f"{len(self.columns)} columns"
            )
        self._rows.append(list(cells))
        return self

    def footer(self, *cells: Cell) -> "Table":
        """Set a footer row -- a total/sum line rendered after a separator
        rule, below the body. At most one; a second call replaces it. Its
        cells count toward the column widths, so the footer stays aligned.
        Use for a ``sum`` / total row (atomic-charge sums, an energy total)
        that a reader should see set apart from the per-item rows."""
        if len(cells) != len(self.columns):
            raise ValueError(
                f"footer has {len(cells)} cells but the table has "
                f"{len(self.columns)} columns"
            )
        self._footer = list(cells)
        return self

    def extend(self, rows: Iterable[Sequence[Cell]]) -> "Table":
        for row in rows:
            self.add_row(*row)
        return self

    def __len__(self) -> int:
        return len(self._rows)

    def _render_cell(self, cell: Cell, policy: FormatPolicy) -> str:
        if cell is None:
            return ""
        if isinstance(cell, Quantity):
            return policy.render(cell)
        if isinstance(cell, bool):
            # Before str(): bool is an int subclass, and "True" reads
            # better than "1" in a converged column.
            return "yes" if cell else "no"
        if isinstance(cell, float):
            return policy.render(Quantity(cell, "dimensionless"))
        return str(cell)

    def render(self, policy: FormatPolicy = DEFAULT_POLICY) -> str:
        """Render to text. Column widths fit the widest cell or header
        (including the footer row, when set)."""
        body = [[self._render_cell(c, policy) for c in row] for row in self._rows]
        foot = (
            [self._render_cell(c, policy) for c in self._footer]
            if self._footer is not None
            else None
        )
        # The footer's cells participate in width-fitting so it stays aligned.
        sizing = body + ([foot] if foot is not None else [])
        widths = [
            max(len(col.header), *(len(row[i]) for row in sizing)) if sizing
            else len(col.header)
            for i, col in enumerate(self.columns)
        ]

        pad = " " * self.indent
        sep = " " * self.gutter

        def _row_text(cells: list[str]) -> str:
            return pad + sep.join(
                f"{cell:{self.columns[i].align}{widths[i]}}"
                for i, cell in enumerate(cells)
            ).rstrip()

        header = pad + sep.join(
            f"{col.header:{col.align}{widths[i]}}"
            for i, col in enumerate(self.columns)
        ).rstrip()
        rule = pad + self.rule_char * (
            sum(widths) + self.gutter * (len(self.columns) - 1)
        )

        lines = [header, rule]
        lines.extend(_row_text(row) for row in body)
        if foot is not None:
            lines.append(rule)
            lines.append(_row_text(foot))
        return "\n".join(lines)


@dataclass(frozen=True)
class Criterion:
    """One measured value tested against an optional upper threshold.

    Both sides stay semantic :class:`Quantity` objects until render time, so
    a display-unit policy converts the measurement and threshold together.
    ``threshold=None`` supports reports whose producer knows only the verdict.
    """

    label: str
    value: Quantity
    threshold: Optional[Quantity]
    passed: bool

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("a criterion label must not be empty")
        if self.threshold is not None and self.value.kind != self.threshold.kind:
            raise ValueError(
                "criterion value and threshold must use the same quantity kind; "
                f"got {self.value.kind!r} and {self.threshold.kind!r}"
            )


class CriteriaTable:
    """Render value-versus-threshold verdicts as a table or compact cell.

    :meth:`render` produces a normal content-sized table for standalone
    diagnostics. :meth:`render_inline` owns the equivalent single-line form
    used inside a wider progress table. Status is deliberately the ASCII
    words ``pass`` / ``fail`` rather than caller-selected glyphs.
    """

    def __init__(self, *, indent: int = 2, gutter: int = 2) -> None:
        if indent < 0 or gutter < 0:
            raise ValueError("indent and gutter widths must be >= 0")
        self.indent = indent
        self.gutter = gutter
        self._criteria: list[Criterion] = []

    def add(
        self,
        label: str,
        value: Quantity,
        threshold: Optional[Quantity],
        passed: bool,
    ) -> "CriteriaTable":
        self._criteria.append(
            Criterion(str(label), value, threshold, bool(passed))
        )
        return self

    def extend(self, criteria: Iterable[Criterion]) -> "CriteriaTable":
        for criterion in criteria:
            if not isinstance(criterion, Criterion):
                raise TypeError("criteria must contain Criterion objects")
            self._criteria.append(criterion)
        return self

    def __len__(self) -> int:
        return len(self._criteria)

    @staticmethod
    def _number(quantity: Quantity, policy: FormatPolicy) -> str:
        # Criteria are magnitudes. Keep the active notation and precision but
        # let the enclosing table own width and suppress an energy-delta plus.
        compact = policy.with_spec(quantity.kind, width=0, sign=False)
        return compact.render(quantity)

    @staticmethod
    def _unit(criterion: Criterion, policy: FormatPolicy) -> str:
        if policy.unit_placement == "none":
            return ""
        return policy.unit_of(criterion.value.kind)

    def render(self, policy: FormatPolicy = DEFAULT_POLICY) -> str:
        """Render a headered, content-sized criteria table."""
        show_units = any(self._unit(c, policy) for c in self._criteria)
        columns: list[Column] = [
            Column("criterion", "<"),
            Column("value"),
            Column("threshold"),
        ]
        if show_units:
            columns.append(Column("unit", "<"))
        columns.append(Column("status", "<"))
        table = Table(columns, indent=self.indent, gutter=self.gutter)
        for criterion in self._criteria:
            threshold = (
                "--"
                if criterion.threshold is None
                else self._number(criterion.threshold, policy)
            )
            row: list[Cell] = [
                criterion.label,
                self._number(criterion.value, policy),
                threshold,
            ]
            if show_units:
                row.append(self._unit(criterion, policy))
            row.append("pass" if criterion.passed else "fail")
            table.add_row(*row)
        return table.render(policy)

    def render_inline(self, policy: FormatPolicy = DEFAULT_POLICY) -> str:
        """Render criteria compactly for one cell of a streaming table."""
        parts: list[str] = []
        for criterion in self._criteria:
            value = self._number(criterion.value, policy)
            comparison = f"{criterion.label}={value}"
            if criterion.threshold is not None:
                threshold = self._number(criterion.threshold, policy)
                comparison += f" <= {threshold}"
            unit = self._unit(criterion, policy)
            if unit:
                comparison += f" {unit}"
            status = "pass" if criterion.passed else "fail"
            parts.append(f"{comparison} [{status}]")
        return "; ".join(parts)


class OutputDocument:
    """An ordered sequence of blocks that renders to the ``.out`` text.

    Blocks are accumulated as semantic records, not strings, so the same
    document can be rendered under a different :class:`FormatPolicy` (say,
    eV at 6 decimals) without the emitting module being aware.
    """

    def __init__(self, *, indent: int = 2, label_width: int = 32) -> None:
        self.indent = indent
        self.label_width = label_width
        self._blocks: list[tuple[str, object]] = []

    def section(
        self, title: str, *, unit_for: Optional[str] = None
    ) -> "OutputDocument":
        """A titled block: blank line, title, rule sized to the title.

        ``unit_for`` names a quantity kind whose unit label is appended to
        the title when the policy places units in the header. Passing the
        kind rather than a literal ``"(Ha)"`` is what lets a policy switch
        to eV and have the header follow instead of going stale.
        """
        self._blocks.append(("section", (title, unit_for)))
        return self

    def scalar(
        self, label: str, value: Union[Quantity, str, float]
    ) -> "OutputDocument":
        """One ``label   value [unit]`` line."""
        self._blocks.append(("scalar", (label, value)))
        return self

    def table(self, table: Table) -> "OutputDocument":
        self._blocks.append(("table", table))
        return self

    def text(self, body: str) -> "OutputDocument":
        """A pre-formatted block, spliced verbatim. Escape hatch only."""
        self._blocks.append(("text", body))
        return self

    def blank(self) -> "OutputDocument":
        self._blocks.append(("blank", None))
        return self

    def _render_section(
        self, title: str, unit_for: Optional[str], policy: FormatPolicy
    ) -> list[str]:
        pad = " " * self.indent
        if unit_for is not None and policy.unit_placement == "header":
            unit = policy.unit_of(unit_for)
            if unit:
                title = f"{title} ({unit})"
        return ["", pad + title, pad + "-" * len(title)]

    def _render_scalar(
        self, label: str, value: Union[Quantity, str, float], policy: FormatPolicy
    ) -> str:
        pad = " " * self.indent
        if isinstance(value, Quantity):
            rendered = policy.render(value)
            if policy.unit_placement == "row":
                rendered = f"{rendered} {policy.unit_of(value.kind)}".rstrip()
        elif isinstance(value, float):
            rendered = policy.render(Quantity(value, "dimensionless"))
        else:
            rendered = str(value)
        return f"{pad}{label:<{self.label_width}s} {rendered}".rstrip()

    def render(self, policy: FormatPolicy = DEFAULT_POLICY) -> str:
        """Render every block to a single string."""
        lines: list[str] = []
        for kind, payload in self._blocks:
            if kind == "section":
                title, unit_for = payload
                lines.extend(self._render_section(title, unit_for, policy))
            elif kind == "scalar":
                label, value = payload
                lines.append(self._render_scalar(label, value, policy))
            elif kind == "table":
                lines.append(payload.render(policy))
            elif kind == "text":
                lines.append(payload)
            elif kind == "blank":
                lines.append("")
        # A leading blank from an opening section() is an artefact of the
        # block model, not something a caller asked for.
        while lines and not lines[0]:
            lines.pop(0)
        return "\n".join(lines)
