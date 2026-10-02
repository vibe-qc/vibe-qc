"""Wording for a periodic result's band edges: ``.out`` block and manifest.

The numbers come from :mod:`vibeqc.band_edges`, the one classifier every
surface shares; this module only says them. Until GitLab #176 the ``.out``
block classified states itself and printed a band-overlap metal on a
zero-temperature global fill as an insulator, because under one chemical
potential the highest occupied sampled level always lies below the lowest
empty one: the two-k reproducer read ``gap (indirect) = 0.300000 Ha
(8.163 eV)`` for a metal whose band overlap is 0.3 Ha, while the public
:func:`vibeqc.frontier_bands` accessor said ``-0.3``.

* :func:`band_extrema_block` -- the ``Band extrema (multi-k)`` ``.out`` block.
* :func:`band_edge_manifest_fields` -- the same facts as flat scalars for the
  ``.system`` ``[run]`` section, under the field names the semiempirical
  k-route already writes (issue #426).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

from ..document import (
    HeaderlessBlock,
    HeaderlessColumn,
    active_policy,
    render_energy,
)

__all__ = [
    "band_edge_manifest_fields",
    "band_extrema_block",
]


def _energy_pair(value: float, *, width: int, precision: int) -> str:
    return (
        f"{render_energy(value, width=width, precision=precision, unit='Ha')}"
        " Ha  ("
        f"{render_energy(value, width=8, precision=3, unit='eV')} eV)"
    )


def _where(k_idx: int, kpoints: Optional[Sequence]) -> str:
    if kpoints is not None and 0 <= k_idx < len(kpoints):
        coords = ", ".join(f"{float(x):+.4f}" for x in kpoints[k_idx])
        return f"at k = [{coords}]"
    if kpoints is not None:
        return f"at k_idx = {k_idx}"
    return f"(k_idx = {k_idx})"


def band_extrema_block(
    edges,
    *,
    converged: bool = True,
    kpoints: Optional[Sequence] = None,
) -> str:
    """The ``Band extrema (multi-k)`` block for a :class:`BandEdges` record.

    Empty for ``None`` (the result carries no occupations to classify with)
    and for a single k-point, where the molecular-style frontier summary
    already says everything. A fractionally occupied frontier is reported as
    an occupation class rather than as edges, and a non-converged SCF's
    fractional frontier as an artefact of its last Fock (#106).
    """
    if edges is None or int(edges.n_kpoints) <= 1:
        return ""
    block = HeaderlessBlock(
        "Band extrema (multi-k)",
        [HeaderlessColumn("<")],
        body_indent=2,
        annotation_gutter=2,
    )
    if edges.fractional_fermi_group:
        # "fractionally occupied / smeared" is a statement about the
        # material. A non-converged SCF returns last-iteration eigenvalues,
        # and the fill attached to them is an artefact of an unconverged
        # Fock, not an occupation class -- reporting it as one is how the 2D
        # slab V_ne(k) defect was read as "h-BN is band-overlapping" (GitLab
        # #106, #85). Say what actually happened instead.
        if converged:
            block.add_row(
                "occupation class = fractionally occupied / smeared",
                annotation=(
                    "zero-temperature band edges and metallicity "
                    "not classified"
                ),
            )
        else:
            block.add_row(
                "SCF not converged - band edges not classified",
                annotation=(
                    "fractional frontier occupations here are an "
                    "artefact of the unconverged Fock"
                ),
            )
        return block.render(active_policy()) + "\n"

    vbm_k = int(edges.valence_band_max_k)
    cbm_k = int(edges.conduction_band_min_k)
    if vbm_k < 0 or cbm_k < 0:
        return ""
    vbm = float(edges.valence_band_max)
    cbm = float(edges.conduction_band_min)
    gap = float(edges.indirect_gap)
    block.add_row(
        f"VBM     = {_energy_pair(vbm, width=14, precision=10)}",
        annotation=_where(vbm_k, kpoints),
    )
    block.add_row(
        f"CBM     = {_energy_pair(cbm, width=14, precision=10)}",
        annotation=_where(cbm_k, kpoints),
    )
    gap_type = "direct" if vbm_k == cbm_k else "indirect"
    if edges.fermi_level_crosses_band:
        # The edges are the band pair at the Fermi filling, so the gap line
        # is the measured band overlap, <= 0 by construction.
        gap_note = "metallic: the Fermi level crosses a band (band overlap)"
    elif gap < 0.0:
        # Reported unclamped. When the CBM lies *below* the VBM without the
        # fill crossing a band, the band ordering across the mesh is
        # inverted (a per-point fill or a held non-Aufbau selection), and
        # the truthful report is the negative number: clamping it papers
        # over exactly what the reader needs to see.
        gap_note = "negative: band ordering inverted across the mesh"
    else:
        gap_note = None
    block.add_row(
        f"gap ({gap_type}) = {_energy_pair(gap, width=10, precision=6)}",
        annotation=gap_note,
    )
    direct_k = int(edges.direct_gap_k)
    if direct_k >= 0 and math.isfinite(float(edges.direct_gap)):
        block.add_row(
            "direct gap (min) = "
            f"{_energy_pair(float(edges.direct_gap), width=10, precision=6)}",
            annotation=f"(k_idx = {direct_k})",
        )
    else:
        # No k point carries both a valence and a conduction state, so the
        # direct gap is undefined everywhere on the mesh. Say that rather
        # than rendering a sentinel as a number (#85).
        block.add_row(
            "direct gap (min) = n/a",
            annotation="no k-point has both an occupied and a virtual state",
        )
    if edges.fermi_level_crosses_band:
        block.add_row(
            "occupation class = metallic",
            annotation="a band is occupied at some k and empty at others",
        )
    return block.render(active_policy()) + "\n"


def band_edge_manifest_fields(edges) -> Dict[str, Any]:
    """Flat ``.system`` ``[run]`` fields for a :class:`BandEdges` record.

    The names and the key set match the semiempirical k-route fields
    (#426), so a harvester reads every periodic route the same way. NaN is not a
    valid manifest float: a non-finite value means the edge does not exist in
    the model space and is recorded as the string ``"not-measured"``, never
    silently dropped. ``is_metallic`` travels with the numbers it qualifies,
    so a screen can read "metallic" without inferring it from a sign.
    """
    if edges is None:
        return {}
    fields: Dict[str, Any] = {}
    for name in (
        "indirect_gap",
        "direct_gap",
        "valence_band_max",
        "conduction_band_min",
        "gap_above_fermi_manifold",
    ):
        value = float(getattr(edges, name))
        fields[name] = value if math.isfinite(value) else "not-measured"
    fields["is_metallic"] = bool(edges.is_metallic)
    return fields
