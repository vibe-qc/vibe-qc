"""HTTP client for NIST Chemistry WebBook.

VFETCH-X4 (v0.13.x): fallback for molecules CCCBDB doesn't cover
(typically larger systems with experimental ΔHf° but outside CCCBDB's
<30-atom cutoff). Same cache-30-days / rate-limit / polite-UA pattern
as :mod:`client_cccbdb`.

Scrapes ``https://webbook.nist.gov/cgi/cbook.cgi?ID=<CAS>&Units=SI``
with a `?cTG=on&cTC=on` query string to request gas + condensed
phase thermochemistry tables. Parses via BeautifulSoup (already a
[fetch] dependency) and returns an :class:`ExperimentalReference`
with :class:`Provenance` (NIST SRD 69, license, fetched timestamp).

The returned record is necessarily sparse -- the WebBook has ΔHf° /
entropy / heat capacity but no atomization energies, vibrational
fundamentals, dipoles, polarizabilities, or Cartesian geometries.
Consumers should check ``ref.atomization_energy_kcal_per_mol is None``
before comparing.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from examples.regression.core.spec import (
    ExperimentalReference,
    Provenance,
    ReferenceKind,
)

from .. import __version__ as FETCHER_VERSION
from ..cache import SOURCE_DB_WEBBOOK, FetchCache, FetchCacheMiss
from ._optional_deps import require_bs4

if TYPE_CHECKING:
    # Annotation-only (strings under ``from __future__ import
    # annotations``); bs4 is lazy-imported via ``require_bs4()`` at the
    # parse callsite so importing this module needs no optional extra.
    from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

WEBBOOK_BASE = "https://webbook.nist.gov"
WEBBOOK_DOI = "doi:10.18434/T4D303"
WEBBOOK_LICENSE = "NIST SRD 69"
WEBBOOK_USER_AGENT = (
    f"vibe-qc-fetcher/{FETCHER_VERSION} "
    "(+https://vibe-qc.com/docs/ mailto:mpei@vibe-qc.com)"
)
WEBBOOK_RATE_LIMIT_S: float = 1.5

_throttle_lock = threading.Lock()
_last_request_at: float = 0.0


# ---------------------------------------------------------------------------
# Rate-limited HTTP
# ---------------------------------------------------------------------------


def _http_get_webbook(
    url: str,
    *,
    max_attempts: int = 3,
    initial_backoff_s: float = 5.0,
    max_backoff_s: float = 60.0,
) -> str:
    """Fetch ``url`` with declared UA + rate limiting + Retry-After."""
    global _last_request_at
    backoff = initial_backoff_s
    for attempt in range(1, max_attempts + 1):
        with _throttle_lock:
            elapsed = time.time() - _last_request_at
            if elapsed < WEBBOOK_RATE_LIMIT_S:
                time.sleep(WEBBOOK_RATE_LIMIT_S - elapsed)
            _last_request_at = time.time()

        req = Request(url, headers={"User-Agent": WEBBOOK_USER_AGENT})
        try:
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code == 404:
                raise MoleculeNotFound(
                    f"NIST WebBook returned 404 for CAS {url.rsplit('=', 1)[-1]}"
                )
            if exc.code in (429, 503):
                retry_after = exc.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else backoff
                )
                wait = min(wait, max_backoff_s)
                log.warning(
                    "WebBook %s on attempt %d; sleeping %.1fs", exc.code, attempt, wait
                )
                time.sleep(wait)
                backoff = min(backoff * 2, max_backoff_s)
                continue
            raise
        except URLError:
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff_s)
                continue
            raise
    raise RuntimeError(f"WebBook fetch failed after {max_attempts} attempts: {url}")


class MoleculeNotFound(Exception):
    """WebBook has no entry for this CAS."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"[-+]?\d+\.?\d*")


def _parse_webbook_html(html: str, cas: str) -> dict:
    """Extract thermochemistry from a WebBook HTML page.

    Looks for the "Gas phase thermochemistry data" and "Condensed phase
    thermochemistry data" tables. Returns a dict with keys matching
    ``ExperimentalReference`` field names (as strings). All values are
    ``None`` when not found.

    The WebBook layout is structurally stable (NIST templates, not
    ad-hoc per-molecule), but semi-structured -- we use BeautifulSoup
    for tokenisation and regex for the values.
    """
    BeautifulSoup = require_bs4()
    soup = BeautifulSoup(html, "lxml")
    result: dict = {
        "enthalpy_of_formation_298_kj_per_mol": None,
        "enthalpy_of_formation_298_uncertainty_kj_per_mol": None,
        "enthalpy_of_formation_0_kj_per_mol": None,
        "enthalpy_of_formation_0_uncertainty_kj_per_mol": None,
        "entropy_298_j_per_mol_per_k": None,
        "heat_capacity_298_j_per_mol_per_k": None,
        "ionization_energy_ev": None,
        "ionization_energy_uncertainty_ev": None,
        "electron_affinity_ev": None,
        "electron_affinity_uncertainty_ev": None,
        "name": "",
        "formula": "",
    }

    # ---- Molecule name + formula ----------------------------------------
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
        result["name"] = title

    # Look for "Formula:" in the page -- typically in a <strong> or <b> tag.
    for tag in soup.find_all(["strong", "b"]):
        text = tag.get_text(strip=True)
        if text.startswith("Formula:"):
            result["formula"] = text.removeprefix("Formula:").strip()
            break

    # ---- Gas-phase thermochemistry table --------------------------------
    gas_data = _extract_thermo_block(soup, "Gas phase thermochemistry data")
    if gas_data:
        result.update(gas_data)

    # Fall back to condensed phase if no gas data.
    if result["enthalpy_of_formation_298_kj_per_mol"] is None:
        cond_data = _extract_thermo_block(soup, "Condensed phase thermochemistry data")
        if cond_data:
            result.update(cond_data)

    # ---- Ion energetics (IE, EA) ----------------------------------------
    ie_data = _extract_ion_energetics(soup)
    if ie_data:
        result.update(ie_data)

    return result


def _extract_thermo_block(soup: BeautifulSoup, heading: str) -> dict:
    """Find a thermochemistry table by its ``<h2>`` or ``<h3>`` heading.

    The WebBook uses ``<h2>`` for major sections and lists data in
    ``<li>`` elements or an ``<table>``.
    """
    result: dict = {}
    header = soup.find(
        lambda tag: (
            tag.name in ("h2", "h3")
            and heading.lower() in tag.get_text("", strip=True).lower()
        )
    )
    if header is None:
        return result

    # Walk following siblings until next heading or end of container.
    parent = header.parent
    if parent is None:
        return result

    # Collect all text from following elements.
    container_text = ""
    for sibling in header.find_all_next(string=True):
        # Stop at next h2/h3
        if getattr(sibling, "name", None) in ("h2", "h3"):
            break
        if hasattr(sibling, "get_text"):
            container_text += " " + sibling.get_text(" ", strip=True)
        elif isinstance(sibling, str):
            container_text += " " + sibling.strip()

    # Parse ΔfH° -- the pattern is "ΔfH°gas" or "ΔfH°liquid" or "ΔfH°solid"
    # followed by a number ± uncertainty in kJ/mol.
    _parse_enthalpy(container_text, result)
    _parse_entropy(container_text, result)
    _parse_heat_capacity(container_text, result)

    return result


def _parse_enthalpy(text: str, result: dict) -> None:
    """Extract ΔfH° from text like 'ΔfH°gas  -241.8 ± 0.04  kJ/mol'."""
    # Look for patterns like: ΔfH°gas ... -241.8 ... ± 0.04 ... kJ/mol
    # or: ΔfH°gas = -241.8 kJ/mol
    m = re.search(
        r"ΔfH°(?:gas|liquid|solid)\s*[:=]?\s*"
        r"(" + _NUMBER_RE.pattern + r")\s*"
        r"(?:±\s*(" + _NUMBER_RE.pattern + r"))?"
        r"\s*kJ/mol",
        text,
        re.IGNORECASE,
    )
    if m:
        result["enthalpy_of_formation_298_kj_per_mol"] = float(m.group(1))
        if m.group(2):
            result["enthalpy_of_formation_298_uncertainty_kj_per_mol"] = float(
                m.group(2)
            )


def _parse_entropy(text: str, result: dict) -> None:
    """Extract S° from text like 'S°gas  188.8  J/mol*K'."""
    m = re.search(
        r"S°(?:gas|liquid|solid)\s*[:=]?\s*"
        r"(" + _NUMBER_RE.pattern + r")\s*"
        r"J/mol\*?K",
        text,
        re.IGNORECASE,
    )
    if m:
        result["entropy_298_j_per_mol_per_k"] = float(m.group(1))


def _parse_heat_capacity(text: str, result: dict) -> None:
    """Extract C_p from text like 'C_p,gas  33.6  J/mol*K'."""
    m = re.search(
        r"C[Pp](?:,|°)?\s*(?:gas|liquid|solid)\s*[:=]?\s*"
        r"(" + _NUMBER_RE.pattern + r")\s*"
        r"J/mol\*?K",
        text,
        re.IGNORECASE,
    )
    if m:
        result["heat_capacity_298_j_per_mol_per_k"] = float(m.group(1))


def _extract_ion_energetics(soup: BeautifulSoup) -> dict:
    """Extract IE and EA from the 'Gas phase ion energetics data' section."""
    result: dict = {}
    header = soup.find(
        lambda tag: (
            tag.name in ("h2", "h3")
            and "ion energetics" in tag.get_text("", strip=True).lower()
        )
    )
    if header is None:
        return result

    text = ""
    for sibling in header.find_all_next(string=True):
        if getattr(sibling, "name", None) in ("h2", "h3"):
            break
        if hasattr(sibling, "get_text"):
            text += " " + sibling.get_text(" ", strip=True)
        elif isinstance(sibling, str):
            text += " " + sibling.strip()

    # IE: "IE (eV)" or "Ionization energy" followed by a number
    m = re.search(
        r"(?:IE|Ionization energy)\s*(?:\(eV\))?\s*[:=]?\s*"
        r"(" + _NUMBER_RE.pattern + r")\s*"
        r"(?:±\s*(" + _NUMBER_RE.pattern + r"))?"
        r"\s*eV",
        text,
        re.IGNORECASE,
    )
    if m:
        result["ionization_energy_ev"] = float(m.group(1))
        if m.group(2):
            result["ionization_energy_uncertainty_ev"] = float(m.group(2))

    # EA: "EA (eV)" or "Electron affinity"
    m2 = re.search(
        r"(?:EA|Electron affinity)\s*(?:\(eV\))?\s*[:=]?\s*"
        r"(" + _NUMBER_RE.pattern + r")\s*"
        r"(?:±\s*(" + _NUMBER_RE.pattern + r"))?"
        r"\s*eV",
        text,
        re.IGNORECASE,
    )
    if m2:
        result["electron_affinity_ev"] = float(m2.group(1))
        if m2.group(2):
            result["electron_affinity_uncertainty_ev"] = float(m2.group(2))

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_webbook(
    cas: str,
    *,
    cache: Optional[FetchCache] = None,
    use_cache: bool = True,
    cache_only: bool = False,
) -> ExperimentalReference:
    """Fetch thermochemistry data from the NIST Chemistry WebBook.

    Returns a sparse ``ExperimentalReference`` -- the WebBook covers
    ΔHf°, entropy, heat capacity, IE, and EA for molecules of any size,
    but has no atomization energies, vibrational fundamentals, dipoles,
    polarizabilities, or Cartesian geometries.

    For molecules CCCBDB covers (< ~30 atoms, H/C/N/O/F), prefer
    :func:`vibeqc.fetch.references.client_cccbdb.fetch_cccbdb` -- it
    returns richer records. This function is the fallback for larger
    species and organics.
    """
    if cache is None:
        cache = FetchCache()

    source_db = "NIST WebBook"
    if use_cache:
        cached = cache.get(source_db, cas)
        if cached is not None:
            # Replay: cached is a JSON-serialised ExperimentalReference.
            from .parsers import _replay_cccbdb_from_cache

            return _replay_webbook_from_cache(cached)

    if cache_only:
        raise FetchCacheMiss(f"{source_db}/{cas}")

    url = f"{WEBBOOK_BASE}/cgi/cbook.cgi?ID={cas}&Units=SI&cTG=on&cTC=on"
    html = _http_get_webbook(url)

    # Quick 404-like check -- WebBook returns 200 with "no data" text
    # for unknown CAS numbers.
    if "no data available" in html.lower() or "species not found" in html.lower():
        raise MoleculeNotFound(f"NIST WebBook has no entry for CAS {cas}")

    parsed = _parse_webbook_html(html, cas)

    prov = Provenance(
        source_db=source_db,
        source_id=cas,
        source_url=url,
        original_reference=WEBBOOK_DOI,
        license=WEBBOOK_LICENSE,
        fetched_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        fetcher_version=FETCHER_VERSION,
        notes=(
            "NIST Chemistry WebBook (SRD 69) -- thermochemistry only; "
            "no atomization energies, vibrational data, or geometry. "
            "Prefer CCCBDB (SRD 101) for small molecules."
        ),
    )

    ref = ExperimentalReference(
        cas=cas,
        formula=parsed["formula"],
        name=parsed["name"],
        kind="experimental",
        enthalpy_of_formation_298_kj_per_mol=parsed[
            "enthalpy_of_formation_298_kj_per_mol"
        ],
        enthalpy_of_formation_298_uncertainty_kj_per_mol=parsed[
            "enthalpy_of_formation_298_uncertainty_kj_per_mol"
        ],
        enthalpy_of_formation_0_kj_per_mol=parsed["enthalpy_of_formation_0_kj_per_mol"],
        enthalpy_of_formation_0_uncertainty_kj_per_mol=parsed[
            "enthalpy_of_formation_0_uncertainty_kj_per_mol"
        ],
        entropy_298_j_per_mol_per_k=parsed["entropy_298_j_per_mol_per_k"],
        heat_capacity_298_j_per_mol_per_k=parsed["heat_capacity_298_j_per_mol_per_k"],
        ionization_energy_ev=parsed["ionization_energy_ev"],
        ionization_energy_uncertainty_ev=parsed["ionization_energy_uncertainty_ev"],
        electron_affinity_ev=parsed["electron_affinity_ev"],
        electron_affinity_uncertainty_ev=parsed["electron_affinity_uncertainty_ev"],
        provenance=prov,
    )

    cache.put(source_db, cas, raw={"html": html}, test_system=None)
    return ref


def _replay_webbook_from_cache(cached: dict) -> ExperimentalReference:
    """Reconstruct an ExperimentalReference from cached data."""
    raw = cached.get("raw", {})
    html = raw.get("html", "")
    cas = cached.get("source_id", "")
    parsed = _parse_webbook_html(html, cas) if html else {}

    prov = Provenance(
        source_db="NIST WebBook",
        source_id=cas,
        source_url="",
        original_reference=WEBBOOK_DOI,
        license=WEBBOOK_LICENSE,
        fetched_at=cached.get("fetched_at", ""),
        fetcher_version=cached.get("fetcher_version", FETCHER_VERSION),
        notes=(
            "NIST Chemistry WebBook (SRD 69) -- thermochemistry only; "
            "replayed from cache."
        ),
    )

    return ExperimentalReference(
        cas=cas,
        formula=parsed.get("formula", ""),
        name=parsed.get("name", ""),
        kind="experimental",
        enthalpy_of_formation_298_kj_per_mol=parsed.get(
            "enthalpy_of_formation_298_kj_per_mol"
        ),
        enthalpy_of_formation_298_uncertainty_kj_per_mol=parsed.get(
            "enthalpy_of_formation_298_uncertainty_kj_per_mol"
        ),
        enthalpy_of_formation_0_kj_per_mol=parsed.get(
            "enthalpy_of_formation_0_kj_per_mol"
        ),
        enthalpy_of_formation_0_uncertainty_kj_per_mol=parsed.get(
            "enthalpy_of_formation_0_uncertainty_kj_per_mol"
        ),
        entropy_298_j_per_mol_per_k=parsed.get("entropy_298_j_per_mol_per_k"),
        heat_capacity_298_j_per_mol_per_k=parsed.get(
            "heat_capacity_298_j_per_mol_per_k"
        ),
        ionization_energy_ev=parsed.get("ionization_energy_ev"),
        ionization_energy_uncertainty_ev=parsed.get("ionization_energy_uncertainty_ev"),
        electron_affinity_ev=parsed.get("electron_affinity_ev"),
        electron_affinity_uncertainty_ev=parsed.get("electron_affinity_uncertainty_ev"),
        provenance=prov,
    )
