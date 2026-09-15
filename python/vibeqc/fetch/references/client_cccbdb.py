"""HTTP client for NIST CCCBDB ``exp2x`` per-molecule pages.

Hits ``https://cccbdb.nist.gov/exp2x.asp?casno=<7-digit-CAS>``,
parses via :func:`vibeqc.fetch.references.parsers.parse_exp2x_html`,
and returns an :class:`ExperimentalReference` with full
:class:`Provenance` (NIST DOI, license, fetched timestamp).

Rate limiting (per the handover Sec. 7.2):
  * Sleep >= 1 s between consecutive requests in the same process.
  * Declared User-Agent identifying vibe-qc + version + repo URL.
  * Honour ``Retry-After`` on 429 / 503 with exponential backoff
    starting at 5 s, capped at 60 s, max 3 retries.
  * Aggressive 30-day cache via :class:`FetchCache` --
    every cached hit is one HTTP request NIST doesn't have to serve.

Architecture deviation from the handover-doc draft (logged in
``references/__init__.py``): we use a single URL + single parser
per molecule, not the multi-table-code orchestration the doc
proposed. Live probing showed that ``ea2x.asp`` returns 500,
``enthalpyx.asp`` returns 404, and the real per-molecule data is
all in ``exp2x.asp``.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from examples.regression.core.spec import ExperimentalReference, Provenance

from .. import __version__ as FETCHER_VERSION
from ..cache import FetchCache, FetchCacheMiss, SOURCE_DB_CCCBDB
from .atomic_enthalpies import derive_atomization_kj_per_mol
from .canonical_molecules import cas_to_url, composition_from_formula
from .parsers import CCCBDBNoData, parse_exp2x_html

log = logging.getLogger(__name__)

# CODATA 2018 -- same values as expected_bridge.py.
HARTREE_TO_KCAL_PER_MOL: float = 627.5094740631
KJ_TO_KCAL: float = 1.0 / 4.184

CCCBDB_DOI = "doi:10.18434/T47C7Z"
CCCBDB_LICENSE = "NIST SRD"
CCCBDB_USER_AGENT = (
    f"vibe-qc-fetcher/{FETCHER_VERSION} "
    "(+https://vibe-qc.com/docs/ mailto:mpei@vibe-qc.com)"
)
CCCBDB_RATE_LIMIT_S: float = 1.0


# Per-process throttle. CCCBDB is a US-government public resource;
# polite rate limiting is not optional.
_throttle_lock = threading.Lock()
_last_request_at: float = 0.0


def _http_get_with_rate_limit(
    url: str,
    *,
    max_attempts: int = 3,
    initial_backoff_s: float = 5.0,
    max_backoff_s: float = 60.0,
) -> str:
    """Fetch ``url`` with declared UA + >=1 s spacing + Retry-After honour.

    Returns the response body decoded as ISO-8859-1 (CCCBDB declares
    ``charset=iso-8859-1`` in its HTML meta tag).
    """
    global _last_request_at
    backoff = initial_backoff_s
    for attempt in range(1, max_attempts + 1):
        with _throttle_lock:
            elapsed = time.time() - _last_request_at
            if elapsed < CCCBDB_RATE_LIMIT_S:
                time.sleep(CCCBDB_RATE_LIMIT_S - elapsed)
            _last_request_at = time.time()

        req = Request(url, headers={"User-Agent": CCCBDB_USER_AGENT})
        try:
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode("iso-8859-1", errors="replace")
        except HTTPError as exc:
            if exc.code in (429, 503):
                # Honour Retry-After, fall back to exp backoff.
                retry_after = exc.headers.get("Retry-After")
                wait = (
                    float(retry_after) if retry_after and retry_after.isdigit()
                    else backoff
                )
                wait = min(wait, max_backoff_s)
                log.warning(
                    "CCCBDB %s on attempt %d/%d for %s; sleeping %.1fs",
                    exc.code, attempt, max_attempts, url, wait,
                )
                time.sleep(wait)
                backoff = min(backoff * 2, max_backoff_s)
                if attempt == max_attempts:
                    raise
                continue
            raise   # 4xx other than 429 -> don't retry, surface to caller
        except URLError as exc:
            log.warning(
                "CCCBDB transport error on attempt %d/%d for %s: %s",
                attempt, max_attempts, url, exc,
            )
            if attempt == max_attempts:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff_s)
    raise RuntimeError(f"CCCBDB fetch unreachable code path for {url!r}")


def _build_provenance(*, cas: str, source_url: str, notes: str = "") -> Provenance:
    return Provenance(
        source_db=SOURCE_DB_CCCBDB,
        source_id=cas,
        source_url=source_url,
        original_reference=CCCBDB_DOI,
        license=CCCBDB_LICENSE,
        fetched_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        fetcher_version=FETCHER_VERSION,
        notes=notes,
    )


def _build_reference(
    *,
    cas: str,
    source_url: str,
    parsed: dict,
) -> ExperimentalReference:
    """Compose an :class:`ExperimentalReference` from parsed property dict
    + CCCBDB provenance + derived atomization energy."""
    formula = parsed.get("formula", "") or ""
    name = parsed.get("name", "") or ""

    # Derive atomization energy from Hf°(0K) when possible.
    atomization_kcal: Optional[float] = None
    notes_extras = []
    h_f_0k = parsed.get("enthalpy_of_formation_0_kj_per_mol")
    if h_f_0k is not None and formula:
        try:
            comp = composition_from_formula(formula)
            ae_kj = derive_atomization_kj_per_mol(
                composition=comp,
                h_f_0k_molecule_kj_per_mol=h_f_0k,
            )
            atomization_kcal = ae_kj * KJ_TO_KCAL
            notes_extras.append(
                "atomization energy derived from Hf°(0K) + CODATA atomic refs "
                "(see references/atomic_enthalpies.py)"
            )
        except KeyError as exc:
            # An atomic Hf°(0K) is missing from our table; skip cleanly.
            notes_extras.append(
                f"atomization energy not derived: {exc}"
            )

    # Explicitly note the D_0 vs D_e distinction. CCCBDB-derived AE
    # is D_0 (thermodynamic, ZPE-included) -- the value most QC
    # textbooks show is D_e (electronic, ZPE-removed). Compute D_e =
    # D_0 + ZPE if your method comparison is electronic-only; the
    # vibrational fundamentals on this same record give you ZPE via
    # 0.5 * sum(fundamentals).
    notes_extras.insert(
        0,
        "atomization_energy_kcal_per_mol is D_0 (0 K, ZPE-included); "
        "for D_e add ZPE = 0.5 * sum(vibrational_fundamentals_cm_inv) "
        "converted to kcal/mol",
    )
    notes = "fetched from CCCBDB exp2x; " + "; ".join(notes_extras)
    provenance = _build_provenance(cas=cas, source_url=source_url, notes=notes)

    return ExperimentalReference(
        cas=cas,
        formula=formula,
        name=name,
        kind="experimental",
        atomization_energy_kcal_per_mol=atomization_kcal,
        # CCCBDB doesn't publish an atomization-energy uncertainty
        # directly; the dominant term is Hfg(0K) uncertainty propagated
        # through derive_atomization_kj_per_mol, plus atomic-ref
        # uncertainties (typically much smaller). For v1 we leave
        # uncertainty=None; v1.1 can sum the variances.
        atomization_energy_uncertainty_kcal_per_mol=None,
        enthalpy_of_formation_298_kj_per_mol=parsed.get(
            "enthalpy_of_formation_298_kj_per_mol",
        ),
        enthalpy_of_formation_298_uncertainty_kj_per_mol=parsed.get(
            "enthalpy_of_formation_298_uncertainty_kj_per_mol",
        ),
        enthalpy_of_formation_0_kj_per_mol=h_f_0k,
        enthalpy_of_formation_0_uncertainty_kj_per_mol=parsed.get(
            "enthalpy_of_formation_0_uncertainty_kj_per_mol",
        ),
        ionization_energy_ev=parsed.get("ionization_energy_ev"),
        ionization_energy_uncertainty_ev=parsed.get(
            "ionization_energy_uncertainty_ev",
        ),
        proton_affinity_kj_per_mol=parsed.get("proton_affinity_kj_per_mol"),
        entropy_298_j_per_mol_per_k=parsed.get("entropy_298_j_per_mol_per_k"),
        heat_capacity_298_j_per_mol_per_k=parsed.get(
            "heat_capacity_298_j_per_mol_per_k",
        ),
        vibrational_fundamentals_cm_inv=parsed.get(
            "vibrational_fundamentals_cm_inv", (),
        ),
        vibrational_harmonics_cm_inv=parsed.get(
            "vibrational_harmonics_cm_inv", (),
        ),
        ir_intensities_km_per_mol=parsed.get(
            "ir_intensities_km_per_mol", (),
        ),
        dipole_moment_debye=parsed.get("dipole_moment_debye"),
        polarizability_au=parsed.get("polarizability_au"),
        bond_lengths_ang=parsed.get("bond_lengths_ang", ()),
        bond_angles_deg=parsed.get("bond_angles_deg", ()),
        cartesian_geometry_ang=parsed.get("cartesian_geometry_ang", ()),
        provenance=provenance,
    )


# ---- Public API ------------------------------------------------------------

class MoleculeNotFound(LookupError):
    """Raised when CCCBDB has no per-molecule entry for the given CAS."""


def fetch_cccbdb(
    *,
    cas: str,
    cache: Optional[FetchCache] = None,
    use_cache: bool = True,
    cache_only: bool = False,
    html_path: Optional[Path] = None,
) -> ExperimentalReference:
    """Fetch a CCCBDB ``exp2x`` page by CAS, parse, return reference.

    Parameters
    ----------
    cas:
        Hyphenated CAS registry number, e.g. ``"7732-18-5"`` for water.
    cache:
        :class:`FetchCache` to read / write through. Default new
        instance.
    use_cache:
        If ``False``, bypass cache reads (still writes through after a
        live fetch).
    cache_only:
        If ``True``, only read from cache -- error on miss. Useful for
        offline / reproducibility runs.
    html_path:
        Escape hatch for tests -- when set, read the HTML from disk
        instead of hitting ``cccbdb.nist.gov``.
    """
    if cache is None:
        cache = FetchCache()

    if use_cache:
        cached = cache.get(SOURCE_DB_CCCBDB, cas)
        if cached is not None:
            html = cached.get("raw", {}).get("html")
            if html:
                parsed = parse_exp2x_html(html)
                return _build_reference(
                    cas=cas,
                    source_url=cas_to_url(cas),
                    parsed=parsed,
                )
    if cache_only:
        raise FetchCacheMiss(f"{SOURCE_DB_CCCBDB}/{cas}")

    if html_path is not None:
        html = Path(html_path).read_text(encoding="iso-8859-1")
    else:
        html = _http_get_with_rate_limit(cas_to_url(cas))

    try:
        parsed = parse_exp2x_html(html)
    except CCCBDBNoData as exc:
        raise MoleculeNotFound(
            f"CCCBDB has no exp2x data for CAS {cas!r}: {exc}"
        ) from exc

    spec = _build_reference(
        cas=cas, source_url=cas_to_url(cas), parsed=parsed,
    )

    cache.put(
        SOURCE_DB_CCCBDB, cas,
        raw={"html": html, "url": cas_to_url(cas)},
        test_system=None,                              # not a TestSystem
    )
    return spec
