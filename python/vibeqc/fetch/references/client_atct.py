"""ATcT (Active Thermochemical Tables) client -- VFETCH-X8 (v0.13.x).

Fetches and parses the ATcT species list from
``https://atct.anl.gov/Thermochemical%20Data/version%201.124/``,
indexes by CAS number and chemical formula, and returns
:class:`ExperimentalReference` records with ``kind="evaluated"``.

The ATcT provides network-corrected thermochemistry with explicit
error bars -- often more accurate than individual literature values
for anchor molecules. The schema already supports this:
``ReferenceKind = "evaluated"`` is defined on ``ExperimentalReference``
with a pointer to this module.

Cache: 30-day TTL via :class:`FetchCache`, same pattern as CCCBDB.

Citations (required for published work):
  Ruscic et al., *J. Phys. Chem. A* **108**, 9979 (2004),
  doi:10.1021/jp047912y
  Ruscic et al., *J. Phys. Chem. A* **119**, 7810 (2015),
  doi:10.1021/acs.jpca.5b01346
  Ruscic & Bross, ATcT ver. 1.124, doi:10.17038/CSE/1885923
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.request import Request, urlopen

from examples.regression.core.spec import ExperimentalReference, Provenance

from .. import __version__ as FETCHER_VERSION
from ..cache import FetchCache, FetchCacheMiss
from ._optional_deps import require_bs4

log = logging.getLogger(__name__)

ATCT_VERSION = "1.124"
ATCT_BASE = "https://atct.anl.gov"
ATCT_SPECIES_URL = f"{ATCT_BASE}/Thermochemical%20Data/version%20{ATCT_VERSION}/"
ATCT_DOI = "doi:10.17038/CSE/1885923"
ATCT_LICENSE = "CC-BY-4.0 (U.S. Government work, ANL)"
ATCT_USER_AGENT = (
    f"vibe-qc-fetcher/{FETCHER_VERSION} "
    "(+https://vibe-qc.com/; mailto:mpei@vibe-qc.com)"
)

_throttle_lock = threading.Lock()
_last_request_at: float = 0.0


class MoleculeNotFound(Exception):
    """ATcT has no entry for this CAS / formula."""


def _http_get_atct(url: str) -> str:
    """Fetch ATcT page with declared UA + rate limiting."""
    global _last_request_at
    with _throttle_lock:
        elapsed = time.time() - _last_request_at
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
        _last_request_at = time.time()

    req = Request(url, headers={"User-Agent": ATCT_USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_ROW_RE = re.compile(
    r"\|\s*"  # leading pipe
    r"(?P<name>[^|]+?)\s*\|"  # species name
    r"\s*(?P<formula>[^|]+?)\s*\|"  # formula + phase
    r"\s*\|"  # image column (empty)
    r"\s*(?P<dh0>[^\|]*?)\s*\|"  # ΔfH°(0 K)
    r"\s*(?P<dh298>[^\|]*?)\s*\|"  # ΔfH°(298.15 K)
    r"\s*(?P<uncertainty>[^\|]*?)\s*\|"  # uncertainty
    r"\s*\|"  # units column
    r"\s*\|"  # molecular mass column
    r"\s*(?P<atct_id>[^\|]*?)\s*\|"  # ATcT ID
)

_CAS_FROM_ATCT = re.compile(r"^(\d+-\d+-\d+)\*")


def _parse_atct_html(html: str) -> dict[str, dict]:
    """Parse the ATcT species list into {cas: {fields}, ...} and
    {formula: {fields}, ...} lookup dicts.

    Returns a dict with keys ``"by_cas"`` and ``"by_formula"``, each
    mapping to ``{key: {field: value}}`` where ``field`` matches
    ``ExperimentalReference`` member names.
    """
    BeautifulSoup = require_bs4()
    soup = BeautifulSoup(html, "lxml")
    by_cas: dict[str, dict] = {}
    by_formula: dict[str, dict] = {}

    # The species table is the main <table> or list of <tr> elements.
    # The ATcT page uses <tr> rows with <td> cells in a specific order.
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 10:
            continue

        # Extract text from cells
        name = cells[0].get_text(strip=True)
        formula_raw = cells[1].get_text(strip=True)
        dh0_text = cells[3].get_text(strip=True)
        dh298_text = cells[4].get_text(strip=True)
        unc_text = cells[5].get_text(strip=True)
        atct_id = cells[9].get_text(strip=True) if len(cells) > 9 else ""

        # Skip header / element-only rows (ΔfH° = 0 / exact)
        if dh0_text in ("", "0") and dh298_text in ("", "0"):
            if "exact" in unc_text.lower():
                continue

        # Parse CAS from ATcT ID (e.g. "7732-18-5*500" -> "7732-18-5")
        cas = ""
        m = _CAS_FROM_ATCT.match(atct_id)
        if m:
            cas = m.group(1)

        # Parse formula (strip phase annotation like "(g)", "(cr,l)")
        formula = re.sub(r"\s*\([^)]*\)\s*$", "", formula_raw).strip()

        # Parse numeric values
        dh0 = _parse_float(dh0_text)
        dh298 = _parse_float(dh298_text)
        uncertainty = _parse_uncertainty(unc_text)

        entry = {
            "name": name,
            "formula": formula,
            "enthalpy_of_formation_0_kj_per_mol": dh0,
            "enthalpy_of_formation_0_uncertainty_kj_per_mol": uncertainty,
            "enthalpy_of_formation_298_kj_per_mol": dh298,
            "enthalpy_of_formation_298_uncertainty_kj_per_mol": uncertainty,
        }

        if cas:
            by_cas[cas] = entry
        if formula:
            by_formula[formula] = entry

    return {"by_cas": by_cas, "by_formula": by_formula}


def _parse_float(text: str) -> Optional[float]:
    """Parse a numeric string, returning None on failure."""
    if not text or text in ("", " ", " "):
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _parse_uncertainty(text: str) -> Optional[float]:
    """Parse ATcT uncertainty -- ``"± 0.025"`` or ``"exact"``."""
    if not text or "exact" in text.lower():
        return 0.0
    m = re.search(r"[\d.]+", text)
    if m:
        return float(m.group())
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_atct_index(
    cache: FetchCache,
    *,
    use_cache: bool = True,
    cache_only: bool = False,
) -> dict:
    """Load the ATcT species index, caching the parsed result for 30 days.

    Returns ``{"by_cas": {...}, "by_formula": {...}}``.
    """
    source_db = "ATcT"
    cache_id = f"species_index_v{ATCT_VERSION}"
    if use_cache:
        cached = cache.get(source_db, cache_id)
        if cached is not None:
            raw = cached.get("raw", {})
            result = raw.get("index")
            if isinstance(result, dict):
                return result

    if cache_only:
        raise FetchCacheMiss(f"{source_db}/{cache_id}")

    html = _http_get_atct(ATCT_SPECIES_URL)
    index = _parse_atct_html(html)
    cache.put(source_db, cache_id, raw={"index": index}, test_system=None)
    return index


def fetch_atct(
    *,
    cas: Optional[str] = None,
    formula: Optional[str] = None,
    cache: Optional[FetchCache] = None,
    use_cache: bool = True,
    cache_only: bool = False,
) -> ExperimentalReference:
    """Fetch ATcT thermochemistry for a molecule by CAS or formula.

    Returns a sparse ``ExperimentalReference`` with ``kind="evaluated"``.
    The ATcT provides ΔHf°(0 K) and ΔHf°(298 K) with uncertainties
    for 2791 species (v1.124). No atomization energies, vibrational
    data, dipoles, or geometries.

    One of ``cas`` or ``formula`` must be provided. CAS lookup is
    preferred (unambiguous); formula lookup returns the ground-state
    gas-phase entry when multiple phases exist for the same formula.
    """
    if (cas is None) == (formula is None):
        raise ValueError("exactly one of `cas` or `formula` must be set")

    if cache is None:
        cache = FetchCache()

    index = _load_atct_index(cache, use_cache=use_cache, cache_only=cache_only)

    entry: Optional[dict] = None
    if cas:
        entry = index["by_cas"].get(cas)
    elif formula:
        entry = index["by_formula"].get(formula)

    if entry is None:
        lookup = cas or formula
        raise MoleculeNotFound(f"ATcT v{ATCT_VERSION} has no entry for {lookup!r}")

    prov = Provenance(
        source_db=f"ATcT v{ATCT_VERSION}",
        source_id=cas or formula or "",
        source_url=ATCT_SPECIES_URL,
        original_reference=ATCT_DOI,
        license=ATCT_LICENSE,
        fetched_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        fetcher_version=FETCHER_VERSION,
        notes=(
            f"Active Thermochemical Tables v{ATCT_VERSION} (Argonne National "
            "Laboratory) -- network-corrected thermochemistry. Provides "
            "ΔHf°(0 K) and ΔHf°(298 K) with 95% confidence uncertainties. "
            "No atomization energies, vibrational data, or geometries. "
            "Cite: Ruscic et al., J. Phys. Chem. A 108, 9979 (2004) and "
            "Ruscic & Bross, ATcT v1.124, doi:10.17038/CSE/1885923."
        ),
    )

    return ExperimentalReference(
        cas=cas or "",
        formula=entry["formula"],
        name=entry["name"],
        kind="evaluated",
        enthalpy_of_formation_0_kj_per_mol=entry["enthalpy_of_formation_0_kj_per_mol"],
        enthalpy_of_formation_0_uncertainty_kj_per_mol=entry[
            "enthalpy_of_formation_0_uncertainty_kj_per_mol"
        ],
        enthalpy_of_formation_298_kj_per_mol=entry[
            "enthalpy_of_formation_298_kj_per_mol"
        ],
        enthalpy_of_formation_298_uncertainty_kj_per_mol=entry[
            "enthalpy_of_formation_298_uncertainty_kj_per_mol"
        ],
        provenance=prov,
    )
