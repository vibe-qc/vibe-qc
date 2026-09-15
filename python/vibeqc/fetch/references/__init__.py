"""Reference-data fetcher (NIST CCCBDB primary, NIST WebBook secondary).

Phase 2 of the vibe-qc external-data fetcher build-out. Pulls
experimental and computed reference data -- atomization energies,
enthalpies of formation, vibrational frequencies, dipoles,
ionization energies -- for QC method validation. See
``docs/tutorial/external_data_fetcher.md``.

The phase-1 fetch infrastructure (``FetchCache``, ``Provenance``,
``vqfetch`` CLI scaffolding) carries over directly.

**Architecture deviation from the handover-doc draft.** The doc
proposed many table-code-specific parsers
(``ea2x.asp``, ``enthalpyx.asp``, ``vibs2x.asp``, ...). Live probing
revealed that:

  * ``ea2x.asp``      -> 500 Internal Server Error
  * ``enthalpyx.asp`` -> 404 Not Found
  * ``ea1x.asp``      -> form-only landing page (the ``?casno=`` is
                          ignored by these "summary" pages)
  * **``exp2x.asp``** -> the *one-stop per-molecule* page that
                          contains every property we want for v1
                          (thermochem, vibrational, geometry, IE,
                          dipole, polarizability) in 23 well-
                          structured tables.

So v1 ships with one URL, one parser, one HTTP request per molecule.
Atomization energy isn't directly tabulated; we derive it from
``Hfg(0K)`` of the molecule + a small CODATA atomic-enthalpy table
(``atomic_enthalpies.py``) -- cleaner than scraping a separate
``atomize1x.asp``-style page that turned out to be form-only too.
"""
from __future__ import annotations

# Lazy-load the heavy client (which pulls in requests / the
# rate-limited HTTP layer) so unit tests that only exercise the
# parser don't trip on optional deps. ``vibeqc.fetch.references.parsers``
# uses only ``bs4``.
def __getattr__(name):
    if name == "fetch_cccbdb":
        from .client_cccbdb import fetch_cccbdb as f
        return f
    raise AttributeError(name)


from .geometry_bridge import experimental_geometry_to_molecule_spec
from .parsers import parse_exp2x_html

__all__ = [
    "fetch_cccbdb",
    "parse_exp2x_html",
    "experimental_geometry_to_molecule_spec",
]
