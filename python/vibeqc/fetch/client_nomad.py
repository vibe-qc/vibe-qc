"""NOMAD raw-archive hook.

VFETCH-X5 (v0.13.x): given a NOMAD entry id, resolve the structure
through the NOMAD OPTIMADE endpoint, then probe the raw input archive
URL and stash it in ``Provenance.notes`` so the user can pull the
original VASP / CRYSTAL / Quantum ESPRESSO / FHI-aims input by hand.
We deliberately do *not* parse VASP INCARs into vibe-qc options --
cross-code input translation is its own can of worms.

Useful when the user wants to reproduce a published DFT setup
(functional, k-mesh, plane-wave cutoff) from a specific paper.
"""

from __future__ import annotations

from dataclasses import replace as _replace
from typing import Optional
from urllib.request import Request, urlopen

from examples.regression.core.spec import PeriodicSpec

from .client_optimade import fetch_optimade

NOMAD_OPTIMADE_BASE = "https://nomad-lab.eu/prod/v1/optimade"
NOMAD_RAW_ARCHIVE_TEMPLATE = (
    "https://nomad-lab.eu/prod/v1/api/v1/entries/{entry_id}/archive"
)


def _probe_raw_archive_url(entry_id: str) -> Optional[str]:
    """Return the raw-archive URL if accessible (HEAD -> 200), else None.

    Does a lightweight HEAD request (no body transfer). Errors
    (timeout, 404, 403, network unreachable) return None silently
    -- the structure fetch succeeds regardless.
    """
    url = NOMAD_RAW_ARCHIVE_TEMPLATE.format(entry_id=entry_id)
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return url
    except Exception:
        pass
    return None


def fetch_nomad(
    *,
    entry_id: str,
    quick: bool = False,
    slug_override: Optional[str] = None,
    **kwargs,
) -> PeriodicSpec:
    """Resolve a NOMAD entry through its OPTIMADE endpoint.

    If the NOMAD raw archive is reachable, its URL is appended to
    ``Provenance.notes`` (prefixed with ``"raw archive: "``) so
    the caller can pull the original code-native input for
    reproducibility.
    """
    spec = fetch_optimade(
        optimade_id=f"nomad/{entry_id}",
        base_urls=[NOMAD_OPTIMADE_BASE],
        provider="nomad",
        quick=quick,
        slug_override=slug_override,
        **kwargs,
    )[0]

    archive_url = _probe_raw_archive_url(entry_id)
    if archive_url and spec.provenance is not None:
        raw_note = f"raw archive: {archive_url}"
        merged = (
            (spec.provenance.notes + "; " + raw_note)
            if spec.provenance.notes
            else raw_note
        )
        spec = _replace(
            spec,
            provenance=_replace(spec.provenance, notes=merged),
        )

    return spec
