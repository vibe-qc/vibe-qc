"""Matplotlib plotters for periodic-system observables.

This module is a *thin* presentation layer on top of
:mod:`vibeqc.bands`. It imports matplotlib lazily so that vibe-qc itself
does not impose a hard matplotlib dependency -- callers only pay the
import cost when they actually draw something.

All functions accept an optional ``ax`` (or ``axes``) so they compose
into existing figures, and return the matplotlib ``Figure`` so the
caller can save it.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .bands import BandStructure, DensityOfStates, ProjectedDensityOfStates
from .coop_cohp import COOPCOHPResult

__all__ = [
    "band_structure_figure",
    "bands_coop_figure",
    "bands_cohp_figure",
    "coop_figure",
    "cohp_figure",
    "dos_figure",
    "pdos_figure",
    "bands_dos_figure",
    "bands_pdos_figure",
]


_HARTREE_TO_EV = 27.211386245988


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:  # pragma: no cover - import error
        raise ImportError(
            "vibeqc.plot requires matplotlib. Install with `pip install matplotlib`."
        ) from e
    import matplotlib.pyplot as plt

    return plt


def band_structure_figure(
    bs: BandStructure,
    *,
    ax=None,
    units: str = "eV",
    shift_to_fermi: bool = True,
    color: str = "tab:blue",
    linewidth: float = 1.0,
    title: Optional[str] = None,
):
    """Draw a band-structure plot.

    Parameters
    ----------
    bs
        :class:`vibeqc.BandStructure` from :func:`vibeqc.band_structure`.
    ax
        Existing matplotlib axes; a new figure is made if ``None``.
    units
        ``"eV"`` (default) or ``"Hartree"``.
    shift_to_fermi
        If True and ``bs.e_fermi`` is set, shift the y-axis so the
        Fermi level sits at zero.
    """
    plt = _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    if units == "eV":
        scale = _HARTREE_TO_EV
        ylabel = "Energy (eV)"
    elif units in ("Hartree", "Ha", "hartree"):
        scale = 1.0
        ylabel = "Energy (Hartree)"
    else:
        raise ValueError(f"Unknown units: {units!r}")

    e0 = bs.e_fermi if (shift_to_fermi and bs.e_fermi is not None) else 0.0
    energies = (bs.energies - e0) * scale
    x = bs.kpath.distances

    for n in range(bs.n_bands):
        ax.plot(x, energies[:, n], color=color, lw=linewidth)

    if shift_to_fermi and bs.e_fermi is not None:
        ax.axhline(0.0, color="0.4", lw=0.6, ls="--")
        ylabel = ylabel.replace("Energy", "E - E_F")

    # High-symmetry tick marks.
    ax.set_xticks([d for d, _ in bs.kpath.labels])
    ax.set_xticklabels([lbl for _, lbl in bs.kpath.labels])
    for d, _ in bs.kpath.labels:
        ax.axvline(d, color="0.7", lw=0.5)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel(ylabel)
    ax.set_xlabel("k-path")
    if title:
        ax.set_title(title)
    return fig


def dos_figure(
    dos: DensityOfStates,
    *,
    ax=None,
    units: str = "eV",
    shift_to_fermi: bool = True,
    orientation: str = "vertical",
    color: str = "tab:orange",
    fill: bool = True,
    title: Optional[str] = None,
):
    """Draw a density-of-states plot.

    ``orientation="horizontal"`` puts energy on the y-axis (so the
    figure can sit beside a band-structure panel).
    """
    plt = _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    if units == "eV":
        scale = _HARTREE_TO_EV
        e_label = "Energy (eV)"
    elif units in ("Hartree", "Ha", "hartree"):
        scale = 1.0
        e_label = "Energy (Hartree)"
    else:
        raise ValueError(f"Unknown units: {units!r}")

    e0 = dos.e_fermi if (shift_to_fermi and dos.e_fermi is not None) else 0.0
    e = (dos.energies - e0) * scale
    d = dos.dos / scale  # density per unit energy -> rescale with units

    if orientation == "vertical":
        ax.plot(e, d, color=color, lw=1.0)
        if fill:
            ax.fill_between(e, 0.0, d, color=color, alpha=0.25)
        ax.set_xlabel(
            ("E - E_F " if (shift_to_fermi and dos.e_fermi is not None) else "")
            + e_label
        )
        ax.set_ylabel("DOS (states / energy / cell)")
        if shift_to_fermi and dos.e_fermi is not None:
            ax.axvline(0.0, color="0.4", lw=0.6, ls="--")
    elif orientation == "horizontal":
        ax.plot(d, e, color=color, lw=1.0)
        if fill:
            ax.fill_betweenx(e, 0.0, d, color=color, alpha=0.25)
        ax.set_ylabel(
            ("E - E_F " if (shift_to_fermi and dos.e_fermi is not None) else "")
            + e_label
        )
        ax.set_xlabel("DOS")
        if shift_to_fermi and dos.e_fermi is not None:
            ax.axhline(0.0, color="0.4", lw=0.6, ls="--")
    else:
        raise ValueError(
            f"orientation must be 'vertical' or 'horizontal', got {orientation!r}"
        )

    if title:
        ax.set_title(title)
    return fig


def bands_dos_figure(
    bs: BandStructure,
    dos: DensityOfStates,
    *,
    units: str = "eV",
    shift_to_fermi: bool = True,
    width_ratio: Tuple[float, float] = (3.0, 1.0),
    title: Optional[str] = None,
):
    """Combined band structure + DOS panel -- the standard solid-state
    layout (bands on the left with shared y-axis to a horizontal DOS on
    the right).
    """
    plt = _require_matplotlib()
    fig, (ax_b, ax_d) = plt.subplots(
        1,
        2,
        sharey=True,
        gridspec_kw={"width_ratios": list(width_ratio)},
        figsize=(7, 4),
    )
    band_structure_figure(
        bs,
        ax=ax_b,
        units=units,
        shift_to_fermi=shift_to_fermi,
    )
    dos_figure(
        dos,
        ax=ax_d,
        units=units,
        shift_to_fermi=shift_to_fermi,
        orientation="horizontal",
    )
    ax_d.set_ylabel("")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def pdos_figure(
    pdos: ProjectedDensityOfStates,
    *,
    ax=None,
    units: str = "eV",
    shift_to_fermi: bool = True,
    orientation: str = "vertical",
    show_total: bool = True,
    stack: bool = False,
    colors: Optional[Mapping[str, str]] = None,
    only: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
):
    """Draw a projected DOS plot.

    Parameters
    ----------
    pdos
        :class:`ProjectedDensityOfStates` from
        :func:`vibeqc.density_of_states_projected`.
    orientation
        ``"vertical"`` (energy on x-axis) or ``"horizontal"``
        (energy on y-axis, for stacking next to a band-structure panel).
    show_total
        If ``True``, draws the unprojected total in black on top of the
        per-group lines (gives an "is the projection exhaustive?" visual
        check -- the line should coincide with the sum of the colored
        contributions).
    stack
        If ``True``, ``fill_between`` the contributions cumulatively
        (the standard "stacked PDOS" look). If ``False``, plots one
        line per group with optional fill.
    colors
        Optional ``{label: matplotlib_color}`` map; otherwise matplotlib
        cycles through its default colors.
    only
        If given, restricts the plot to these labels (in order).
    """
    plt = _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    if units == "eV":
        scale = _HARTREE_TO_EV
        e_label = "Energy (eV)"
    elif units in ("Hartree", "Ha", "hartree"):
        scale = 1.0
        e_label = "Energy (Hartree)"
    else:
        raise ValueError(f"Unknown units: {units!r}")

    e0 = pdos.e_fermi if (shift_to_fermi and pdos.e_fermi is not None) else 0.0
    e = (pdos.energies - e0) * scale

    labels = list(only) if only is not None else list(pdos.contributions.keys())
    missing = [l for l in labels if l not in pdos.contributions]
    if missing:
        raise KeyError(f"pdos_figure: requested labels not in PDOS: {missing!r}")

    contribs = [pdos.contributions[l] / scale for l in labels]
    total = pdos.total / scale

    color_for = lambda l: (colors or {}).get(l)

    if orientation == "vertical":
        if stack:
            cum = np.zeros_like(e)
            for l, c in zip(labels, contribs):
                ax.fill_between(
                    e, cum, cum + c, label=l, color=color_for(l), alpha=0.6, lw=0.0
                )
                cum = cum + c
        else:
            for l, c in zip(labels, contribs):
                (line,) = ax.plot(e, c, label=l, lw=1.0, color=color_for(l))
                ax.fill_between(e, 0.0, c, color=line.get_color(), alpha=0.18)
        if show_total:
            ax.plot(e, total, color="0.15", lw=1.0, ls="--", label="total")
        ax.set_xlabel(("E - E_F " if e0 != 0.0 else "") + e_label)
        ax.set_ylabel("PDOS (states / energy / cell)")
        if e0 != 0.0:
            ax.axvline(0.0, color="0.4", lw=0.6, ls="--")
    elif orientation == "horizontal":
        if stack:
            cum = np.zeros_like(e)
            for l, c in zip(labels, contribs):
                ax.fill_betweenx(
                    e, cum, cum + c, label=l, color=color_for(l), alpha=0.6, lw=0.0
                )
                cum = cum + c
        else:
            for l, c in zip(labels, contribs):
                (line,) = ax.plot(c, e, label=l, lw=1.0, color=color_for(l))
                ax.fill_betweenx(e, 0.0, c, color=line.get_color(), alpha=0.18)
        if show_total:
            ax.plot(total, e, color="0.15", lw=1.0, ls="--", label="total")
        ax.set_ylabel(("E - E_F " if e0 != 0.0 else "") + e_label)
        ax.set_xlabel("PDOS")
        if e0 != 0.0:
            ax.axhline(0.0, color="0.4", lw=0.6, ls="--")
    else:
        raise ValueError(
            f"orientation must be 'vertical' or 'horizontal', got {orientation!r}"
        )

    ax.legend(loc="best", fontsize=8, frameon=False)
    if title:
        ax.set_title(title)
    return fig


def bands_pdos_figure(
    bs: BandStructure,
    pdos: ProjectedDensityOfStates,
    *,
    units: str = "eV",
    shift_to_fermi: bool = True,
    width_ratio: Tuple[float, float] = (3.0, 1.5),
    stack: bool = False,
    show_total: bool = True,
    colors: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
):
    """Combined band structure + projected DOS panel. Mirrors
    :func:`bands_dos_figure` but draws per-group PDOS contributions on
    the right pane (slightly wider by default to fit the legend)."""
    plt = _require_matplotlib()
    fig, (ax_b, ax_d) = plt.subplots(
        1,
        2,
        sharey=True,
        gridspec_kw={"width_ratios": list(width_ratio)},
        figsize=(8, 4),
    )
    band_structure_figure(
        bs,
        ax=ax_b,
        units=units,
        shift_to_fermi=shift_to_fermi,
    )
    pdos_figure(
        pdos,
        ax=ax_d,
        units=units,
        shift_to_fermi=shift_to_fermi,
        orientation="horizontal",
        stack=stack,
        show_total=show_total,
        colors=colors,
    )
    ax_d.set_ylabel("")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def bands_coop_figure(
    bs: BandStructure,
    result: COOPCOHPResult,
    *,
    units: str = "eV",
    shift_to_fermi: bool = True,
    width_ratio: Tuple[float, float] = (3.0, 1.8),
    max_pairs: int = 8,
    title: Optional[str] = None,
):
    """Combined band structure + COOP panel.

    Left pane: band structure. Right pane: COOP(E) per atom pair
    (horizontal orientation, energy axis shared with bands).
    """
    plt = _require_matplotlib()
    fig, (ax_b, ax_c) = plt.subplots(
        1,
        2,
        sharey=True,
        gridspec_kw={"width_ratios": list(width_ratio)},
        figsize=(8, 4),
    )
    band_structure_figure(
        bs,
        ax=ax_b,
        units=units,
        shift_to_fermi=shift_to_fermi,
    )
    coop_figure(
        result,
        ax=ax_c,
        units=units,
        shift_to_fermi=shift_to_fermi,
        max_pairs=max_pairs,
    )
    ax_c.set_ylabel("")
    ax_c.set_xlabel("COOP")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def bands_cohp_figure(
    bs: BandStructure,
    result: COOPCOHPResult,
    *,
    units: str = "eV",
    shift_to_fermi: bool = True,
    width_ratio: Tuple[float, float] = (3.0, 1.8),
    max_pairs: int = 8,
    title: Optional[str] = None,
):
    """Combined band structure + -COHP panel.

    Left pane: band structure. Right pane: -COHP(E) per atom pair
    (horizontal orientation, energy axis shared with bands).
    """
    plt = _require_matplotlib()
    fig, (ax_b, ax_c) = plt.subplots(
        1,
        2,
        sharey=True,
        gridspec_kw={"width_ratios": list(width_ratio)},
        figsize=(8, 4),
    )
    band_structure_figure(
        bs,
        ax=ax_b,
        units=units,
        shift_to_fermi=shift_to_fermi,
    )
    cohp_figure(
        result,
        ax=ax_c,
        units=units,
        shift_to_fermi=shift_to_fermi,
        max_pairs=max_pairs,
    )
    ax_c.set_ylabel("")
    ax_c.set_xlabel("-COHP")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# COOP / COHP plotters
# ---------------------------------------------------------------------------


def _pair_label(pair: dict) -> str:
    """Human-readable pair label like ``'Si0-Si1 (2.35 Å)'``."""
    sym_i = pair.get("symbol_i", "?")
    sym_j = pair.get("symbol_j", "?")
    i = pair.get("i", 0)
    j = pair.get("j", 0)
    d = pair.get("distance_ang", None)
    label = f"{sym_i}{i}-{sym_j}{j}"
    if d is not None:
        label += f" ({d:.2f} Å)"
    return label


def _coop_cohp_figure_impl(
    result: COOPCOHPResult,
    data: np.ndarray,
    integrated: np.ndarray,
    ylabel: str,
    *,
    ax=None,
    units: str = "eV",
    shift_to_fermi: bool = True,
    max_pairs: int = 16,
    colors: Optional[Mapping[int, str]] = None,
    title: Optional[str] = None,
):
    """Shared implementation for coop_figure / cohp_figure."""
    plt = _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if units == "eV":
        scale = _HARTREE_TO_EV
        e_label = "Energy (eV)"
    elif units in ("Hartree", "Ha", "hartree"):
        scale = 1.0
        e_label = "Energy (Hartree)"
    else:
        raise ValueError(f"Unknown units: {units!r}")

    e0 = result.fermi_energy if shift_to_fermi else 0.0
    e = (result.energies - e0) * scale

    n_pairs = data.shape[0]
    n_show = min(n_pairs, max_pairs)

    default_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#1a55FF",
        "#FF4444",
        "#44AA44",
        "#AA44AA",
        "#AAAA44",
        "#44AAAA",
    ]

    for p in range(n_show):
        color = (colors or {}).get(p, default_colors[p % len(default_colors)])
        pair = result.pairs[p] if p < len(result.pairs) else {}
        label = _pair_label(pair)
        ival = integrated[p] if p < len(integrated) else 0.0
        label += f" (I={ival:+.3f})"
        ax.plot(e, data[p, :] * scale, lw=1.0, color=color, label=label)

    if shift_to_fermi:
        ax.axvline(0.0, color="0.4", lw=0.6, ls="--")
        ax.set_xlabel(f"E - E_F ({e_label.split()[1].strip('()')})")
    else:
        ax.set_xlabel(e_label)
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color="0.3", lw=0.4, ls=":")

    if n_pairs <= 12:
        ax.legend(loc="best", fontsize=7, frameon=False, ncol=1)
    elif n_pairs <= 16:
        ax.legend(loc="best", fontsize=6, frameon=False, ncol=2)

    if title:
        ax.set_title(title)
    return fig


def coop_figure(
    result: COOPCOHPResult,
    *,
    ax=None,
    units: str = "eV",
    shift_to_fermi: bool = True,
    max_pairs: int = 16,
    colors: Optional[Mapping[int, str]] = None,
    title: Optional[str] = None,
):
    """Draw a COOP plot — one line per atom pair.

    Positive COOP = bonding interaction, negative = antibonding.
    The integrated COOP (ICOOP) up to E_F is printed in the legend.

    Parameters
    ----------
    result : COOPCOHPResult
        From :func:`vibeqc.compute_coop_cohp`.
    max_pairs : int
        Maximum number of pairs to show (default 16).
    colors : dict
        Optional ``{pair_index: matplotlib_color}`` map.
    """
    return _coop_cohp_figure_impl(
        result,
        result.coop,
        result.integrated_coop,
        "COOP",
        ax=ax,
        units=units,
        shift_to_fermi=shift_to_fermi,
        max_pairs=max_pairs,
        colors=colors,
        title=title,
    )


def cohp_figure(
    result: COOPCOHPResult,
    *,
    ax=None,
    units: str = "eV",
    shift_to_fermi: bool = True,
    max_pairs: int = 16,
    colors: Optional[Mapping[int, str]] = None,
    title: Optional[str] = None,
):
    """Draw a -COHP plot — one line per atom pair.

    Positive -COHP = bonding interaction (Lobster convention).
    The integrated -COHP (-ICOHP) up to E_F is printed in the legend.

    Parameters
    ----------
    result : COOPCOHPResult
        From :func:`vibeqc.compute_coop_cohp`.  ``result.cohp`` must
        not be ``None`` (pass ``include_cohp=True`` to ``compute_coop_cohp``).
    max_pairs : int
        Maximum number of pairs to show (default 16).
    colors : dict
        Optional ``{pair_index: matplotlib_color}`` map.
    """
    if result.cohp is None:
        raise ValueError(
            "cohp_figure: result.cohp is None. "
            "Pass include_cohp=True to compute_coop_cohp() to compute COHP."
        )
    return _coop_cohp_figure_impl(
        result,
        result.cohp,
        result.integrated_cohp,
        "-COHP (bonding > 0)",
        ax=ax,
        units=units,
        shift_to_fermi=shift_to_fermi,
        max_pairs=max_pairs,
        colors=colors,
        title=title,
    )
