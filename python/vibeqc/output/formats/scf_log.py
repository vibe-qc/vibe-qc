"""Formatters and log emitters for SCF trace output.

The native RHF driver populates ``RHFResult.scf_trace`` (a list of
``SCFIteration`` records) on every run. This module turns that trace into
either:

- a formatted multi-line string (``format_scf_trace``), or
- a stream of structured log records via the standard ``logging`` module
  (``log_scf_trace``).

By convention, log records go to the ``vibeqc.scf`` logger. A library-level
``NullHandler`` is attached in ``vibeqc.__init__`` so nothing is emitted
unless the user explicitly configures logging -- consistent with how
PySCF, numpy, and every other well-behaved Python library behave.

Typical usage::

    import logging
    logging.basicConfig(level=logging.INFO)   # summary lines only
    # or:
    logging.getLogger("vibeqc.scf").setLevel(logging.DEBUG)  # per-iter too

    result = run_rhf(mol, basis)
    log_scf_trace(result)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ...banner import banner
from ...correlation_conventions import effective_electron_count
from ..channel import Level
from ..channel import write as _channel_write
from ..document import (
    Column,
    HeaderlessBlock,
    HeaderlessColumn,
    Quantity,
    Table,
    active_policy,
    render_energy,
)

try:
    # Imported lazily because properties.py pulls in the compiled core;
    # when vibeqc.scf_log is imported during test-harness probing of
    # vibeqc.banner in isolation, avoid the cascade. In normal use this
    # always succeeds.
    from ...properties import (
        dipole_moment,
        hirshfeld_charges,
        loewdin_charges,
        mayer_bond_orders,
        mulliken_charges,
    )
    from ...properties import prominent_bonds as _prominent_bonds

    _PROPERTIES_AVAILABLE = True
except Exception:
    _PROPERTIES_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - import guard
    from ..._vibeqc_core import RHFResult  # noqa: F401

_DEFAULT_LOGGER_NAME = "vibeqc.scf"

# Column layout:
#   iter(>=3)  energy(18.10f)  dE(+.3e)  ||grad||(.3e)  diis(2d)
_HEADER = "  iter     energy (Ha)            dE          ||[F,DS]||   DIIS"
_SEPARATOR = "  " + "-" * (len(_HEADER) - 2)

def format_scf_trace(
    result,
    *,
    molecule=None,
    basis=None,
    include_banner: bool = True,
    include_properties: bool = True,
    iterations_already_streamed: bool = False,
    n_virtual: int = 5,
    property_status: Optional[dict] = None,
    energy_label: str = "Total energy",
    nuclear_charges=None,
) -> str:
    """Return a multi-line SCF report with a convergence summary.

    When the producer supplies per-iteration records, the summary is preceded
    by the shared iteration table.  Result adapters that know only the final
    iteration count may expose an empty ``scf_trace``; those reports omit the
    otherwise-empty table rather than implying that DIIS/residual telemetry
    was captured.  Safe to print, log, or write to a file.

    Parameters
    ----------
    result
        Any SCF result struct with ``scf_trace``, ``converged``, ``n_iter``,
        and ``energy`` fields (RHFResult, UHFResult, RKSResult, UKSResult,
        PeriodicRHFResult, PeriodicKSResult, ...). ``scf_trace`` may be empty
        when the underlying engine exposes only the final iteration count.
    molecule
        Optional :class:`Molecule`. When supplied, the trace is extended
        with an energy-component breakdown (DFT only), an orbital-energy
        table, and HOMO/LUMO markers. Without it we can't infer occupation
        counts from the result alone, so those sections are skipped.
    basis
        Optional :class:`BasisSet`. Required alongside ``molecule`` to
        compute the post-SCF properties block (Mulliken / Löwdin charges,
        Mayer bond orders, dipole moment). When absent, the properties
        block is skipped.
    include_banner
        Prepend the vibe-qc version banner. Default ``True``; set ``False``
        to produce a trace without provenance (e.g. for unit-test
        comparisons where the version string would drift).
    include_properties
        Emit the Mulliken / Löwdin / Mayer / dipole block when molecule
        and basis are both supplied. Default ``True``.
    n_virtual
        Number of virtual orbitals above HOMO to print in the table.
        Default 5.
    property_status
        Optional mutable dict; when supplied, the post-SCF properties
        block records per-analysis success into it (currently
        ``property_status["hirshfeld"]``). Lets a caller gate a citation
        on whether a best-effort analysis actually computed, without
        recomputing it. Default ``None`` (no status reported).
    energy_label
        Label for the scalar energy in the component breakdown. ``run_job``
        uses ``"SCF energy"`` when post-SCF add-ons such as dispersion are
        requested, so parsers do not mistake the SCF trace for the final
        user-facing total.
    """
    segments = _scf_trace_segments(
        result,
        molecule=molecule,
        basis=basis,
        include_banner=include_banner,
        include_properties=include_properties,
        iterations_already_streamed=iterations_already_streamed,
        n_virtual=n_virtual,
        property_status=property_status,
        energy_label=energy_label,
        nuclear_charges=nuclear_charges,
    )
    # Each segment carries the separator that precedes it (default one blank
    # line, "\n\n"); the convergence verdict rejoins its table with a single
    # "\n", so concatenating reproduces the historical string exactly.
    if not segments:
        return ""
    parts = [_seg_parts(segments[0])[0]]
    for seg in segments[1:]:
        text, _level, sep = _seg_parts(seg)
        parts.append(sep + text)
    return "".join(parts)


# Segment = one logical block of the SCF trace, the verbosity band it
# belongs to, and (optionally) the separator that precedes it. A 2-tuple
# ``(text, level)`` takes the default ``"\n\n"`` separator; a 3-tuple
# ``(text, level, sep)`` overrides it (the convergence verdict uses ``"\n"``
# to stay welded to its table byte-for-byte while carrying its own level).
# The structure lets a block move to QUIET / VERBOSE / DEBUG in exactly one
# place (its entry here) and ``write_scf_trace`` will then filter it.
Segment = Union[Tuple[str, Level], Tuple[str, Level, str]]


def _seg_parts(seg: Segment) -> Tuple[str, Level, str]:
    """Unpack a segment into ``(text, level, sep)``, defaulting ``sep`` to
    the historical one-blank-line ``"\\n\\n"`` when a 2-tuple is given."""
    text, level = seg[0], seg[1]
    sep = seg[2] if len(seg) > 2 else "\n\n"
    return text, level, sep


def _scf_trace_segments(
    result,
    *,
    molecule=None,
    basis=None,
    include_banner: bool = True,
    include_properties: bool = True,
    iterations_already_streamed: bool = False,
    n_virtual: int = 5,
    property_status: Optional[dict] = None,
    energy_label: str = "Total energy",
    nuclear_charges=None,
) -> List[Segment]:
    """Build the SCF trace as ``(block_text, level)`` segments.

    Concatenating the block texts with ``"\\n\\n"`` reproduces
    :func:`format_scf_trace`'s string byte-for-byte;
    :func:`write_scf_trace` instead emits each block at its own level so a
    block can be gated to ``VERBOSE`` / ``DEBUG`` without touching the
    others. Assign a block's band by changing the ``Level`` on its
    ``segments.append(...)`` below.
    """
    segments: List[Segment] = []
    if include_banner:
        segments.append((banner(), Level.STANDARD))

    trace = list(getattr(result, "scf_trace", None) or ())
    if trace and iterations_already_streamed:
        # The runner streamed the header and each row into the .out as the
        # native SCF produced them (GitLab #482), so re-emitting the table
        # here would duplicate it.  Close the table it left open instead:
        # the same trailing rule, in the same place, so a completed run's
        # bytes are unchanged.
        segments.append((_SEPARATOR, Level.STANDARD, ""))
    elif trace:
        trace_lines = [_HEADER, _SEPARATOR]
        for step in trace:
            trace_lines.append(_format_iter_line(step))
        trace_lines.append(_SEPARATOR)
        segments.append(("\n".join(trace_lines), Level.STANDARD))
    # The convergence verdict + final energy is *the* essential result, so
    # it is tagged QUIET -- it survives VIBEQC_OUTPUT_LEVEL=quiet even when
    # the iteration table above it (STANDARD) is filtered out. It rejoins
    # that table with a single "\n" separator, so format_scf_trace stays
    # byte-identical to when the two were one segment.
    status = "converged" if result.converged else "NOT converged"
    conv_line = (
        f"  {status} in {result.n_iter} iterations; "
        f"E = {render_energy(result.energy, width=0)} {active_policy().unit_of('energy')}"
    )
    # With a table, the verdict remains welded to it by one newline.  Without
    # a table, use the ordinary block separator after an optional banner.
    conv_sep = "\n" if trace else "\n\n"
    segments.append((conv_line, Level.QUIET, conv_sep))

    # Spin-squared expectation value (UHF / UKS / ROHF).  A spin-contamination
    # diagnostic: <S^2>_ideal = S(S+1) for the requested multiplicity;
    # deviation signals spin contamination.  Printed at QUIET so it survives
    # output-level filtering alongside the convergence verdict.
    _s2 = getattr(result, "s_squared", None)
    _s2_ideal = getattr(result, "s_squared_ideal", None)
    if _s2 is not None and _s2_ideal is not None and _s2_ideal > 0.0:
        _s2_line = (
            f"  <S^2> = {_s2:.6f}  (ideal {_s2_ideal:.4f}, "
            f"delta = {_s2 - _s2_ideal:+.4f})"
        )
        segments.append((_s2_line, Level.QUIET, "\n"))

    # Internal-stability verdict (molecular UHF/UKS stability_check).
    # A vanishing gradient certifies only a stationary point; the character
    # of the extremum is the sign of the lowest orbital-rotation Hessian
    # eigenvalue (Lehtola, Molecules 25, 1218 (2020), Sec. 10; Seeger &
    # Pople, J. Chem. Phys. 66, 3045 (1977)). The verdict itself is decided
    # in cpp/src/uhf.cpp or uks.cpp against the corresponding stability_tol;
    # this block only reports it. The eigenvalue keeps the trace table's diagnostic
    # e-format (a curvature diagnostic, not a policy-formatted energy).
    if getattr(result, "stability_checked", False):
        _lam = float(getattr(result, "stability_eigenvalue", 0.0))
        _n_restarts = int(getattr(result, "n_stability_restarts", 0))
        if not getattr(result, "stability_analysis_converged", False):
            _stab_seg = (
                "  internal stability: eigensolver did not converge; "
                "solution character UNVERIFIED",
                Level.QUIET,
                "\n",
            )
        elif getattr(result, "internal_instability", False):
            # Loud: the returned energy is an excited SCF solution.
            _stab_seg = (
                "  WARNING: converged onto an internally UNSTABLE "
                f"stationary point (lowest Hessian eigenvalue {_lam:.3e}); "
                "the reported energy is an EXCITED SCF solution",
                Level.QUIET,
                "\n",
            )
        elif _n_restarts > 0:
            # Numbers changed relative to a naive single SCF: say so.
            _stab_seg = (
                f"  internal stability: stable after {_n_restarts} "
                "corrective restart(s) -- the first SCF converged onto an "
                f"excited solution (lowest Hessian eigenvalue {_lam:.3e})",
                Level.QUIET,
                "\n",
            )
        else:
            _stab_seg = (
                "  internal stability: stable "
                f"(lowest Hessian eigenvalue {_lam:.3e})",
                Level.STANDARD,
                "\n",
            )
        segments.append(_stab_seg)

    # State fingerprint (BUG 88).  A deterministic hash of the converged
    # electronic state: same-state runs produce identical fingerprints.
    _has_spin = getattr(result, "s_squared", None) is not None
    if _has_spin and molecule is not None:
        from ...state_fingerprint import compute_state_fingerprint
        _n_a = int(getattr(result, "n_alpha", 0))
        _n_b = int(getattr(result, "n_beta", 0))
        _s2 = float(getattr(result, "s_squared", 0.0))
        _fp = compute_state_fingerprint(
            energy=float(result.energy),
            n_alpha=_n_a,
            n_beta=_n_b,
            s_squared=_s2,
            molecule_symbols=[a.symbol for a in molecule.atoms],
        )
        segments.append((f"  state fingerprint: {_fp}", Level.QUIET, "\n"))
    if molecule is None:
        components = _format_energy_components(result, energy_label=energy_label)
        if components:
            segments.append((components, Level.STANDARD))
    if molecule is not None:
        orb = _format_orbital_sections(
            result, molecule, n_virtual=n_virtual, energy_label=energy_label
        )
        if orb:
            segments.append((orb, Level.STANDARD))
    if (
        include_properties
        and molecule is not None
        and basis is not None
        and _PROPERTIES_AVAILABLE
    ):
        props = _format_properties_block(
            result,
            molecule,
            basis,
            status=property_status,
            nuclear_charges=nuclear_charges,
        )
        if props:
            segments.append((props, Level.STANDARD))
    return segments


def write_scf_trace(
    result,
    *,
    write: Optional[Callable[..., object]] = None,
    trailing: str = "",
    molecule=None,
    basis=None,
    include_banner: bool = True,
    include_properties: bool = True,
    n_virtual: int = 5,
    property_status: Optional[dict] = None,
    energy_label: str = "Total energy",
    nuclear_charges=None,
    iterations_already_streamed: bool = False,
) -> None:
    """Emit the SCF trace block-by-block, each block at its own level.

    The level-aware counterpart to :func:`format_scf_trace`: instead of
    returning one string, it writes each segment through ``write`` (the
    ambient :func:`vibeqc.output.write` by default) tagged with the
    segment's :class:`Level`. With every segment at ``STANDARD`` -- the
    state today -- the emitted bytes are identical to
    ``write(format_scf_trace(...) + trailing)``; moving a segment to
    ``VERBOSE`` / ``DEBUG`` makes that block disappear on a lower-threshold
    channel while the rest stay.

    ``trailing`` is written verbatim after the last block (the runners pass
    the ``"\\n"`` / ``"\\n\\n"`` they used to append to the string).

    Each block's leading separator (one blank line by default; ``"\\n"`` for
    the convergence verdict welded to its table) is written at *that
    block's* level, so a suppressed block takes its own leading gap with it
    rather than leaving a dangling one. Because the convergence verdict is
    tagged ``QUIET`` while the iteration table above it is ``STANDARD``, a
    ``QUIET`` channel keeps the final energy + converged/not verdict and
    drops the per-iteration table.
    """
    emit = _channel_write if write is None else write
    segments = _scf_trace_segments(
        result,
        molecule=molecule,
        basis=basis,
        include_banner=include_banner,
        iterations_already_streamed=iterations_already_streamed,
        include_properties=include_properties,
        n_virtual=n_virtual,
        property_status=property_status,
        energy_label=energy_label,
        nuclear_charges=nuclear_charges,
    )
    for i, seg in enumerate(segments):
        text, level, sep = _seg_parts(seg)
        if i > 0:
            emit(sep, level)
        emit(text, level)
    if trailing:
        emit(trailing, Level.STANDARD)


def log_scf_trace(
    result,
    logger: Optional[logging.Logger] = None,
    *,
    level_header: int = logging.INFO,
    level_iter: int = logging.DEBUG,
    level_summary: int = logging.INFO,
) -> None:
    """Emit the SCF trace through the ``logging`` module.

    Log levels are split so users can get summary-only (default INFO) or
    full per-iteration output (DEBUG) by setting the logger level.
    """
    lg = logger if logger is not None else logging.getLogger(_DEFAULT_LOGGER_NAME)

    trace = list(getattr(result, "scf_trace", None) or ())
    if trace and lg.isEnabledFor(level_iter):
        lg.log(level_iter, _HEADER)
        lg.log(level_iter, _SEPARATOR)
        for step in trace:
            lg.log(level_iter, _format_iter_line(step))
        lg.log(level_iter, _SEPARATOR)

    status = "converged" if result.converged else "NOT CONVERGED"
    lg.log(
        level_summary,
        "SCF %s in %d iterations; E = %.10f Ha (= %.4f eV)",
        status,
        result.n_iter,
        result.energy,
        # 1 Hartree = 27.211386245988 eV (CODATA 2018)
        result.energy * 27.211386245988,
    )


def _step_get(step, name: str, default=0):
    """Read a trace field from either an ``SCFIteration`` object or a dict.

    The C++ drivers (RHF/UHF/RKS, GDF, BIPOLE) return ``SCFIteration``
    structs; the GAPW multi-k Python driver appends plain dicts
    (``periodic_gapw_open_shell.py``) with the same field names but no
    ``diis_subspace`` key. Reading through this accessor lets the one
    formatter render both, so periodic results reach ``format_scf_trace``
    without the runner needing a parallel hand-rolled table.
    """
    if isinstance(step, dict):
        return step.get(name, default)
    return getattr(step, name, default)


def _format_iter_line(step) -> str:
    it = _step_get(step, "iter")
    de = f"{_step_get(step, 'delta_e'):+.3e}" if it > 1 else "      --   "
    diis_n = _step_get(step, "diis_subspace")
    diis = f"{diis_n:3d}" if diis_n > 0 else "  -"
    return (
        f"  {it:4d}  {_step_get(step, 'energy'):18.10f}  {de}  "
        f"{_step_get(step, 'grad_norm'):.3e}  {diis}"
    )


def _format_orbital_sections(
    result,
    molecule,
    *,
    n_virtual: int,
    energy_label: str = "Total energy",
) -> str:
    """Energy-component breakdown (DFT only) + orbital-energy table.

    Returns an empty string if the result has neither ``e_xc`` nor
    ``mo_energies``/``mo_energies_alpha`` -- e.g. an MP2Result.
    """
    sections: list[str] = []

    components = _format_energy_components(result, energy_label=energy_label)
    if components:
        sections.append(components)

    n_electrons = effective_electron_count(molecule, result)
    multiplicity = molecule.multiplicity
    n_alpha = (n_electrons + multiplicity - 1) // 2
    n_beta = n_electrons - n_alpha

    # ROHF/ROKS keep one spin-restricted orbital set (the alpha/beta
    # columns would be identical), so print a single table with the
    # closed/open/virtual occupations (2/1/0) instead of the UHF
    # two-column layout.
    _method = str(getattr(result, "method", "")).lower()
    if _method in ("rohf", "roks") and hasattr(result, "mo_occupations"):
        sections.append(
            _format_orbital_table_rohf(
                result, n_alpha=n_alpha, n_beta=n_beta, n_virtual=n_virtual
            )
        )
    elif hasattr(result, "mo_energies_alpha"):
        sections.append(
            _format_orbital_table_uhf(
                result,
                n_alpha=n_alpha,
                n_beta=n_beta,
                n_virtual=n_virtual,
            )
        )
    elif hasattr(result, "mo_energies"):
        sections.append(
            _format_orbital_table_rhf(
                result, n_occ=n_electrons // 2, n_virtual=n_virtual
            )
        )

    return "\n\n".join(s for s in sections if s)


def _format_energy_components(result, *, energy_label: str = "Total energy") -> str:
    """Energy-component breakdown block.

    Supports two result shapes:

    * molecular / periodic KS results with scalar ``e_xc``-style fields;
    * BIPOLE periodic RHF results with a per-iteration
      ``energy_components`` history in CRYSTAL ``ENECYCLE`` terms.
    """
    history = getattr(result, "energy_components", None)
    if history:
        comp = history[-1]
        rows: list[tuple[str, float]] = [
            ("Nuclear repulsion", comp.e_nuclear_repulsion),
            ("Electronic", comp.e_electronic),
            ("Kinetic", comp.e_kinetic),
            ("Nuclear attraction", comp.e_nuclear_attraction),
        ]
        for label, attr in (
            ("BIELET zone E-E", "e_bielet_zone_ee"),
            ("EXT EL-POLE", "e_ext_el_pole"),
            ("EXT EL-SPHEROPOLE", "e_ext_el_spheropole"),
            ("J short-range", "e_j_short_range"),
            ("J long-range", "e_j_long_range"),
            ("HF exchange", "e_exchange"),
            # The q+G=0 / Madelung finite-size share of "HF exchange"
            # above (already counted in it, and in the total). Shown so a
            # cross-code comparison can see the gauge: a code that
            # applies no exchange-divergence correction differs from
            # vibe-qc by roughly this much at the same k-mesh (#82:
            # 0.64 Ha on MgO/STO-3G at (2,2,2)).
            ("  of which q=0 finite-size", "e_exchange_finite_size"),
        ):
            value = getattr(comp, attr, None)
            if value is not None:
                rows.append((label, value))
        # Dudarev DFT+U contribution. BIPOLE results carry e_dft_plus_u=0.0
        # even without +U sites, so presence alone cannot gate the row; only
        # shown when non-zero, matching the scalar branch below.
        _u = getattr(comp, "e_dft_plus_u", None)
        if _u is not None and float(_u) != 0.0:
            rows.append(("DFT+U (Dudarev)", float(_u)))
        rows.extend(
            [
                ("Two-electron (J+K)", comp.e_two_electron),
                (energy_label, comp.e_total),
            ]
        )
        block = HeaderlessBlock(
            "Energy components",
            [
                HeaderlessColumn("<", min_width=32),
                HeaderlessColumn(">", min_width=18),
            ],
            gutter=1,
            unit_for="energy",
        )
        for label, value in rows:
            block.add_row(label, Quantity(value, "energy"))
        block.divider()
        return block.render(active_policy())

    if not hasattr(result, "e_xc"):
        return ""

    rows: list[tuple[str, float]] = []
    for label, attr in (
        ("Nuclear repulsion", "e_nuclear"),
        ("Electronic", "e_electronic"),
        ("Coulomb (J)", "e_coulomb"),
        ("HF exchange (K)", "e_hf_exchange"),
        ("Exchange-correlation (XC)", "e_xc"),
        ("EXT EL-SPHEROPOLE", "e_ext_el_spheropole"),
    ):
        value = getattr(result, attr, None)
        if value is not None:
            rows.append((label, value))
    # Dudarev DFT+U contribution. Only shown when non-zero, so a job with
    # no +U sites (every molecular job, most periodic jobs) is unchanged.
    _u = getattr(result, "e_dft_plus_u", None)
    if _u is not None and float(_u) != 0.0:
        rows.append(("DFT+U (Dudarev)", float(_u)))
    rows.append((energy_label, result.energy))

    block = HeaderlessBlock(
        "Energy components",
        [
            HeaderlessColumn("<", min_width=32),
            HeaderlessColumn(">", min_width=18),
        ],
        gutter=1,
        unit_for="energy",
    )
    for label, value in rows:
        block.add_row(label, Quantity(value, "energy"))
    block.divider()
    return block.render(active_policy())


def _orbital_window(
    energies: Sequence[float], n_occ: int, n_virtual: int
) -> tuple[int, int]:
    """Slice bounds [start, end) to print: all occupied + up to n_virtual."""
    start = 0
    end = min(len(energies), n_occ + max(0, n_virtual))
    return start, end


def _orbital_table_lines(
    title: str,
    energies: Sequence[float],
    start: int,
    end: int,
    occ_of,
    marker_of,
    gap_line: Optional[str] = None,
) -> str:
    """The single definition of the ``#/occ/eps(Ha)/eps(eV)`` orbital table
    shared by the RHF / ROHF / UHF renderers.

    ``occ_of(i)`` and ``marker_of(i)`` supply each row's occupation and its
    HOMO / LUMO / SOMO / frontier marker; ``gap_line`` (when given) is
    appended after the rows. Keeping the header, the rule width, and the row
    format string in one place is why the orbital table has one format rather
    than four near-duplicates that can drift (the deliberate dual Ha/eV
    columns are intentional -- an orbital energy is universally quoted in
    both. Explicit unit overrides on :func:`render_energy` preserve that
    deliberate dual-unit presentation under the active format policy)."""
    block = HeaderlessBlock(
        title,
        [
            HeaderlessColumn(">", min_width=4),
            HeaderlessColumn(">", min_width=4),
            HeaderlessColumn(">", min_width=16),
            HeaderlessColumn(">", min_width=14),
        ],
        gutter=2,
        annotation_gutter=2,
    )
    block.add_row("#", "occ", "eps (Ha)", "eps (eV)").divider()
    for i in range(start, end):
        block.add_row(
            i + 1,
            f"{occ_of(i):.1f}",
            render_energy(energies[i], width=16, precision=8, unit="Ha"),
            render_energy(energies[i], width=14, precision=6, unit="eV"),
            annotation=marker_of(i).strip() or None,
        )
    if gap_line is not None:
        block.footer(gap_line.removeprefix("  "))
    return block.render(active_policy())


def _format_orbital_table_rhf(result, *, n_occ: int, n_virtual: int) -> str:
    energies = list(result.mo_energies)
    start, end = _orbital_window(energies, n_occ, n_virtual)

    def _marker(i: int) -> str:
        if i == n_occ - 1:
            return "  <-- HOMO"
        if i == n_occ:
            return "  <-- LUMO"
        return ""

    gap_line = None
    if 0 < n_occ < len(energies):
        homo = energies[n_occ - 1]
        lumo = energies[n_occ]
        gap = lumo - homo
        gap_line = (
            "  HOMO: "
            f"{render_energy(homo, width=0, precision=6, unit='Ha')} Ha "
            f"({render_energy(homo, width=0, precision=4, unit='eV')} eV); "
            "LUMO: "
            f"{render_energy(lumo, width=0, precision=6, unit='Ha')} Ha "
            f"({render_energy(lumo, width=0, precision=4, unit='eV')} eV); "
            "gap: "
            f"{render_energy(gap, width=0, precision=6, unit='Ha')} Ha "
            f"({render_energy(gap, width=0, precision=4, unit='eV')} eV)"
        )
    return _orbital_table_lines(
        "Molecular orbitals (Ha / eV)", energies, start, end,
        lambda i: 2.0 if i < n_occ else 0.0, _marker, gap_line,
    )


def _format_orbital_table_rohf(
    result, *, n_alpha: int, n_beta: int, n_virtual: int
) -> str:
    """Single-column orbital table for a restricted-open-shell result.

    Occupations are 2 (closed / doubly occupied), 1 (open / singly
    occupied = SOMO), fractional for degenerate open shells such as
    OH(2Pi), and 0 (virtual); the spatial orbitals are shared by both
    spins. ``n_alpha`` is the count of alpha-occupied orbitals,
    ``n_beta`` the beta electron count used for the ordinary integer
    fallback.
    """
    energies = list(result.mo_energies)
    # Prefer the result's own occupations when present; fall back to the
    # aufbau partition from n_alpha / n_beta.
    occ_vec = getattr(result, "mo_occupations", None)
    occ_arr = np.asarray(occ_vec, dtype=float) if occ_vec is not None else None
    start, end = _orbital_window(energies, n_alpha, n_virtual)

    first_virtual = n_alpha
    if occ_arr is not None:
        occupied = np.where(occ_arr > 1.0e-8)[0]
        if occupied.size:
            first_virtual = int(occupied[-1]) + 1

    def _occ(i: int) -> float:
        if occ_arr is not None:
            return float(occ_arr[i])
        return 2.0 if i < n_beta else (1.0 if i < n_alpha else 0.0)

    def _marker(i: int) -> str:
        if occ_arr is not None:
            occ = float(occ_arr[i])
            if 1.0e-8 < occ < 2.0 - 1.0e-8 and abs(occ - 1.0) > 1.0e-8:
                return "  <-- open"
            if abs(occ - 1.0) <= 1.0e-8:
                return "  <-- SOMO"
        if n_beta <= i < n_alpha:
            return "  <-- SOMO"
        if i == n_alpha - 1 and n_alpha == n_beta:
            return "  <-- HOMO"
        if i == first_virtual:
            return "  <-- LUMO"
        return ""

    gap_line = None
    if 0 < first_virtual < len(energies):
        homo = energies[first_virtual - 1]
        lumo = energies[first_virtual]
        gap = lumo - homo
        gap_line = (
            "  HOMO: "
            f"{render_energy(homo, width=0, precision=6, unit='Ha')} Ha "
            f"({render_energy(homo, width=0, precision=4, unit='eV')} eV); "
            "LUMO: "
            f"{render_energy(lumo, width=0, precision=6, unit='Ha')} Ha "
            f"({render_energy(lumo, width=0, precision=4, unit='eV')} eV); "
            "gap: "
            f"{render_energy(gap, width=0, precision=6, unit='Ha')} Ha "
            f"({render_energy(gap, width=0, precision=4, unit='eV')} eV)"
        )
    return _orbital_table_lines(
        "Molecular orbitals (Ha / eV)  [restricted open-shell]",
        energies, start, end, _occ, _marker, gap_line,
    )


def _format_orbital_table_uhf(
    result, *, n_alpha: int, n_beta: int, n_virtual: int
) -> str:
    blocks: list[str] = []
    for label, energies, n_occ in (
        ("Alpha", list(result.mo_energies_alpha), n_alpha),
        ("Beta", list(result.mo_energies_beta), n_beta),
    ):
        start, end = _orbital_window(energies, n_occ, n_virtual)
        tag = label[0]  # 'A' / 'B'

        def _marker(i: int, n_occ: int = n_occ, tag: str = tag) -> str:
            if i == n_occ - 1:
                return f"  <-- HO{tag}MO"
            if i == n_occ:
                return f"  <-- LU{tag}MO"
            return ""

        gap_line = None
        if 0 < n_occ < len(energies):
            homo = energies[n_occ - 1]
            lumo = energies[n_occ]
            gap = lumo - homo
            gap_line = (
                f"  HO{tag}MO: "
                f"{render_energy(homo, width=0, precision=6, unit='Ha')} Ha "
                f"({render_energy(homo, width=0, precision=4, unit='eV')} eV); "
                f"LU{tag}MO: "
                f"{render_energy(lumo, width=0, precision=6, unit='Ha')} Ha "
                f"({render_energy(lumo, width=0, precision=4, unit='eV')} eV); "
                f"gap: "
                f"{render_energy(gap, width=0, precision=6, unit='Ha')} Ha "
                f"({render_energy(gap, width=0, precision=4, unit='eV')} eV)"
            )
        blocks.append(
            _orbital_table_lines(
                f"Molecular orbitals - {label}", energies, start, end,
                lambda i, n_occ=n_occ: 1.0 if i < n_occ else 0.0,
                _marker, gap_line,
            )
        )
    return "\n\n".join(blocks)


_ELEMENT_SYMBOLS = (
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
)


def _element(z: int) -> str:
    if 0 < z < len(_ELEMENT_SYMBOLS):
        return _ELEMENT_SYMBOLS[z]
    return f"Z{z}"


def _fixed_unsigned_zero(value: float, precision: int) -> str:
    """``f"{value:.{precision}f}"``, minus the sign on a rounds-to-zero value.

    A component that is zero at the displayed precision (a symmetry-zero
    dipole component, a neutral molecule's charge sum) carries a noise sign:
    the same run renders ``-0.0000`` on one machine and ``0.0000`` on the
    next as the last bit flips. Tables here size their columns from content,
    so that stray minus also widens the whole column -- a machine-dependent
    *format* change the .out snapshot goldens would trip over. Genuinely
    nonzero negatives keep their sign.
    """
    text = f"{value:.{precision}f}"
    if text.startswith("-") and float(text) == 0.0:
        return text[1:]
    return text


def _format_properties_block(
    result, molecule, basis, *, status=None, nuclear_charges=None
) -> str:
    """Mulliken / Löwdin / Hirshfeld charges, Mayer bond orders,
    dipole moment.

    Emits nothing if the property computation raises (e.g. when the basis
    has shells vibe-qc's Python helpers can't parse); post-SCF properties
    are best-effort -- a missing block shouldn't crash the log writer.

    The Hirshfeld column is computed separately and degrades on its own:
    it needs a Becke-Lebedev grid build + SAD promolecule, either of
    which can fail independently of the (cheap, basis-space) Mulliken /
    Löwdin analyses. When Hirshfeld fails the table falls back to the
    two-column Mulliken / Löwdin layout.

    ``status``
        Optional mutable dict. When supplied, ``status["hirshfeld"]`` is
        set to whether the Hirshfeld column actually computed for this
        call (``True`` only when the charges reached the table). This is
        the single observable the citation assembler gates on, so the
        Hirshfeld 1977 reference is emitted iff the user can see the
        Hirshfeld charges -- never on a grid/promolecule build failure
        (CLAUDE.md Sec. 8). Left untouched when the required Mulliken /
        Löwdin step itself fails (the whole block is absent then)."""
    sections: list[str] = []

    try:
        mul = mulliken_charges(result, basis, molecule, nuclear_charges=nuclear_charges)
        low = loewdin_charges(result, basis, molecule, nuclear_charges=nuclear_charges)
    except Exception:
        return ""

    # Hirshfeld is best-effort on top of the required Mulliken/Löwdin --
    # a grid-build or SAD-promolecule failure must not drop the block.
    try:
        hirsh = hirshfeld_charges(
            result, basis, molecule, nuclear_charges=nuclear_charges
        ).charges
    except Exception:
        hirsh = None
    # Record the success exactly as it lands in the table, so the runner's
    # citation block can cite Hirshfeld 1977 only when it really computed.
    if status is not None:
        status["hirshfeld"] = hirsh is not None

    atoms = list(molecule.atoms)
    # Atomic charges through the Table primitive: columns + rule size to
    # content (retiring the hand-picked 66/52 widths), and the per-scheme
    # totals go in a footer -- a total row set apart by a rule, replacing the
    # hand-formatted "sum" line.
    _charge_cols = [Column("#", ">"), Column("elem", ">"),
                    Column("Mulliken", ">"), Column("Löwdin", ">")]
    if hirsh is not None:
        _charge_cols.append(Column("Hirshfeld", ">"))
    _charge_tbl = Table(_charge_cols)
    for i, atom in enumerate(atoms, start=1):
        _row = [i, _element(int(atom.Z)),
                _fixed_unsigned_zero(mul[i - 1], 6),
                _fixed_unsigned_zero(low[i - 1], 6)]
        if hirsh is not None:
            _row.append(_fixed_unsigned_zero(hirsh[i - 1], 6))
        _charge_tbl.add_row(*_row)
    _foot = ["", "sum",
             _fixed_unsigned_zero(float(np.sum(mul)), 6),
             _fixed_unsigned_zero(float(np.sum(low)), 6)]
    if hirsh is not None:
        _foot.append(_fixed_unsigned_zero(float(np.sum(hirsh)), 6))
    _charge_tbl.footer(*_foot)
    sections.append("  Atomic charges\n" + _charge_tbl.render())

    try:
        B = mayer_bond_orders(result, basis, molecule)
        bonds = _prominent_bonds(B, molecule, threshold=0.10)
    except Exception:
        bonds = []
    if bonds:
        _bond_tbl = Table([Column("atoms", ">"), Column("bond order", ">")])
        for i, j, bo in bonds:
            elem_i = _element(int(atoms[i].Z))
            elem_j = _element(int(atoms[j].Z))
            _bond_tbl.add_row(f"{elem_i}{i + 1} -- {elem_j}{j + 1}", f"{bo:.4f}")
        sections.append("  Bond orders (Mayer)\n" + _bond_tbl.render())

    try:
        dip = dipole_moment(result, basis, molecule, nuclear_charges=nuclear_charges)
    except Exception:
        dip = None
    if dip is not None:
        dx_d, dy_d, dz_d = dip.components_debye()
        _dip_tbl = Table([
            Column("component", ">"),
            Column("a.u. (e.bohr)", ">"),
            Column("Debye", ">"),
        ])
        _dip_tbl.add_row("x", _fixed_unsigned_zero(dip.x, 8),
                         _fixed_unsigned_zero(dx_d, 4))
        _dip_tbl.add_row("y", _fixed_unsigned_zero(dip.y, 8),
                         _fixed_unsigned_zero(dy_d, 4))
        _dip_tbl.add_row("z", _fixed_unsigned_zero(dip.z, 8),
                         _fixed_unsigned_zero(dz_d, 4))
        _dip_tbl.add_row("|mu|", _fixed_unsigned_zero(dip.total, 8),
                         _fixed_unsigned_zero(dip.total_debye, 4))
        sections.append(
            "  Dipole moment (origin: center of mass)\n" + _dip_tbl.render()
        )

    return "\n\n".join(sections)


def format_basis_summary(basis) -> str:
    """Return a multi-line plain-text summary of a :class:`BasisSet`:
    its name, total basis-function / shell count, and a per-shell
    table of exponents and contraction coefficients.

    Used by the periodic SCF live progress logger at verbose level
    >= 5 so users can confirm exactly which exponents were loaded
    when sweeping basis-set choices on a long-running bulk run.
    The output is plain ASCII and indented to match the rest of
    :class:`vibeqc.ProgressLogger` emit.

    The shell-by-shell table is truncated only by the natural shell
    count; very large basis sets (pob-tzvp on a 50-atom cell, say)
    will dump several hundred lines. That's the price of asking for
    ``verbose=5`` on a live run -- same convention PySCF follows
    when its own basis log is enabled.
    """
    name = getattr(basis, "name", "<unknown>")
    nbf = int(getattr(basis, "nbasis", 0))
    nshells = int(getattr(basis, "nshells", 0))
    lines = [
        "  Basis-set details",
        "  " + "-" * 32,
        f"  Name: {name}",
        f"  Functions: {nbf} in {nshells} shell{'s' if nshells != 1 else ''}",
        "",
        f"  {'shell':>5s}  {'l':>1s}  {'atom':>4s}  "
        f"{'n_prim':>6s}   exponent           coefficient",
        "  " + "-" * 60,
    ]
    L_LABEL = ["s", "p", "d", "f", "g", "h", "i"]
    try:
        shells = basis.shells()
    except Exception:
        return "\n".join(lines)
    for sh_idx, sh in enumerate(shells):
        l = int(sh.l)
        l_label = L_LABEL[l] if 0 <= l < len(L_LABEL) else str(l)
        atom_idx = int(sh.atom_index)
        exponents = list(sh.exponents)
        coeffs = list(sh.coefficients)
        n_prim = len(exponents)
        # First primitive on the header row, rest indented underneath.
        lines.append(
            f"  {sh_idx:5d}  {l_label:>1s}  {atom_idx:4d}  "
            f"{n_prim:6d}   {exponents[0]:14.6e}   {coeffs[0]:+14.6e}"
        )
        for e, c in zip(exponents[1:], coeffs[1:]):
            lines.append(
                f"  {'':5s}  {'':1s}  {'':4s}  {'':6s}   {e:14.6e}   {c:+14.6e}"
            )
    return "\n".join(lines)


__all__ = [
    "format_scf_trace",
    "write_scf_trace",
    "format_basis_summary",
    "log_scf_trace",
]
