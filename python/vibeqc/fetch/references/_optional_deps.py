"""Lazy access to the optional HTML-parsing stack (BeautifulSoup4 + lxml).

The reference-data clients in this package (NIST CCCBDB, NIST WebBook,
ATcT) scrape HTML and parse it with :class:`bs4.BeautifulSoup` using the
``"lxml"`` tree builder. Both ``beautifulsoup4`` and ``lxml`` are
*optional* dependencies (they ship only with the ``[fetch]`` extra), so
``import vibeqc.fetch.references`` (and ``pytest --collect-only`` over the
whole test suite) must not hard-require them.

Each parser calls :func:`require_bs4` at the point it actually parses
HTML rather than importing ``bs4`` at module top. Absent the dependency
the user gets a clear ``pip install 'vibe-qc[fetch]'`` pointer instead of
a bare ``ModuleNotFoundError: No module named 'bs4'`` raised eagerly at
import time. Mirrors the lazy-import discipline already used for the
OPTIMADE client (``optimade`` imported inside ``fetch_optimade``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bs4 import BeautifulSoup as _BeautifulSoup


def require_bs4() -> "type[_BeautifulSoup]":
    """Return the :class:`bs4.BeautifulSoup` class, importing it lazily.

    Raises :class:`ModuleNotFoundError` with an actionable install
    pointer when the ``[fetch]`` extra is not installed. The reference
    parsers build with the ``"lxml"`` tree builder, which ships in the
    same extra; if ``lxml`` is missing while ``bs4`` is present, that
    surfaces as BeautifulSoup's own ``FeatureNotFound`` at parse time.
    """
    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "HTML reference-data parsing needs BeautifulSoup4 + lxml, "
            "which ship with the optional 'fetch' extra. Install with:\n"
            "    pip install 'vibe-qc[fetch]'"
        ) from exc
    return BeautifulSoup
