"""Materials Project property-field client.

VFETCH-X3 (v0.13.x): when ``$MP_API_KEY`` is set, augment the
OPTIMADE-fetched ``PeriodicSpec`` with Materials-Project REST-API
property fields: band gap, formation energy per atom, energy above
hull, is_magnetic, total_magnetization. The magnetic data refines
the open-shell heuristic.

Without ``$MP_API_KEY``, the OPTIMADE-only path continues to work
unchanged -- structure + provenance only, no property fields.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import replace as _replace
from typing import Optional

from examples.regression.core.spec import PeriodicSpec

from .client_optimade import fetch_optimade
from .heuristics import open_shell_default

_MP_SUMMARY_URL = "https://api.materialsproject.org/materials/summary/"


def _fetch_mp_properties(
    mp_id: str,
    api_key: str,
) -> Optional[dict]:
    """Fetch the Materials Project summary record for one material id.

    Returns ``None`` on any transport / decode / missing-key error
    (the caller falls back to the OPTIMADE-only spec with no property
    augmentation -- the fetch succeeds, it's just property-poor).
    """
    url = f"{_MP_SUMMARY_URL}?material_ids={mp_id}"
    req = urllib.request.Request(url, headers={"X-API-KEY": api_key})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
    except Exception:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    records = data.get("data", [])
    if not isinstance(records, list) or len(records) == 0:
        return None
    return records[0]


def _augment_spec_with_mp_properties(
    spec: PeriodicSpec,
    mp_props: dict,
) -> PeriodicSpec:
    """Return a new ``PeriodicSpec`` with MP property fields folded in.

    * ``is_open_shell`` and ``magnetic_moments`` updated from MP magnetic
      data (gated on ``total_magnetization > 0.1 mu_B/cell``).
    * ``band_gap``, ``formation_energy_per_atom``, ``energy_above_hull``
      appended to ``Provenance.notes``.
    """
    notes_extra_parts = []

    band_gap = mp_props.get("band_gap")
    if band_gap is not None:
        notes_extra_parts.append(f"MP band_gap={band_gap:.3f} eV")

    form_e = mp_props.get("formation_energy_per_atom")
    if form_e is not None:
        notes_extra_parts.append(f"MP formation_energy_per_atom={form_e:.3f} eV")

    e_hull = mp_props.get("energy_above_hull")
    if e_hull is not None:
        notes_extra_parts.append(f"MP energy_above_hull={e_hull:.3f} eV/atom")

    is_magnetic = mp_props.get("is_magnetic")
    total_mag = mp_props.get("total_magnetization")

    is_open, mag_moments = open_shell_default(
        is_periodic=True,
        mp_is_magnetic=is_magnetic,
        mp_total_magnetization=total_mag,
    )

    # Merge notes.
    if notes_extra_parts:
        mp_note = "MP: " + "; ".join(notes_extra_parts)
        if spec.provenance is not None:
            merged_notes = (
                (spec.provenance.notes + "; " + mp_note)
                if spec.provenance.notes
                else mp_note
            )
            spec = _replace(
                spec,
                provenance=_replace(spec.provenance, notes=merged_notes),
            )

    spec = _replace(
        spec,
        is_open_shell=is_open,
        magnetic_moments=mag_moments,
    )
    return spec


def fetch_mp(
    *,
    mp_id: str,
    api_key: Optional[str] = None,
    quick: bool = False,
    slug_override: Optional[str] = None,
) -> PeriodicSpec:
    """Fetch a Materials Project entry by id.

    v1 path: route through the MP OPTIMADE endpoint
    (https://optimade.materialsproject.org). When ``api_key`` (or
    ``$MP_API_KEY``) is set, also fetch the MP summary record and
    augment the spec with magnetic / electronic / thermodynamic
    property fields (VFETCH-X3).
    """
    api_key = api_key or os.environ.get("MP_API_KEY", "")
    if not mp_id.startswith("mp-"):
        raise ValueError(f"Materials Project id should be 'mp-<n>'; got {mp_id!r}")
    spec = fetch_optimade(
        optimade_id=f"mp/{mp_id}",
        provider="mp",
        quick=quick,
        slug_override=slug_override,
    )[0]

    if api_key:
        mp_props = _fetch_mp_properties(mp_id, api_key)
        if mp_props is not None:
            spec = _augment_spec_with_mp_properties(spec, mp_props)

    return spec
