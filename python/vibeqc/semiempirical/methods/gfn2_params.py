"""GFN2-xTB parameter loading -- fetch + parse + load into GFN2ParameterSet."""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import tomllib
import weakref
from pathlib import Path
from typing import Any

from vibeqc._vibeqc_core.semiempirical import xtb as _xtb

_XTB_PARAM_URL = (
    "https://raw.githubusercontent.com/grimme-lab/xtb/main/param_gfn2-xtb.txt"
)


def _default_cache_dir() -> Path:
    """Resolve the GFN2 parameter cache root (issue #127).

    Same precedence every other vibe-qc cache already uses --
    ``vibeqc.fetch.cache.cache_root``, ``basis_filter``, ``mlip.mace``:
    an explicit override first, then ``$XDG_CACHE_HOME``, then
    ``~/.cache``. GFN2 was the one cache in the tree hardwired to
    ``Path.home()``, which is precisely what made it per-host and
    un-seedable: ``docs/user_guide/reference_data.md`` documents the
    offline-cluster remedy as "rsync ``$XDG_CACHE_HOME/vibeqc/`` to the
    cluster", and GFN2 was the one thing that remedy could not reach. A
    compute node with no outbound network can now be pointed at a
    pre-populated shared cache instead of failing closed with
    ``GFN2ParameterUnavailableError``.

    Resolved at import, like the constants it feeds, so that a job's
    environment (set before the process starts) decides the path and the
    existing test monkeypatches on ``_CACHE_DIR`` / ``_CACHE_FILE`` keep
    working unchanged.

    This does NOT make the parameters available where no cache has been
    seeded -- the upstream file is LGPL-3.0 and bundling it is a
    maintainer licensing decision (CLAUDE.md § 1, § 11), not this
    function's to make.
    """
    override = os.environ.get("VIBEQC_GFN2_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "vibeqc"
    return Path("~/.cache/vibeqc").expanduser()


_CACHE_DIR = _default_cache_dir()
_CACHE_FILE = _CACHE_DIR / "gfn2_xtb_params.toml"
_PARAM_TOML_CACHE: (
    tuple[Path, int, int, dict[str, Any], str, str] | None
) = None
_PARAM_OBJECT_LINEAGE: weakref.WeakKeyDictionary[object, dict[str, str]] = (
    weakref.WeakKeyDictionary()
)


class GFN2ParameterUnavailableError(RuntimeError):
    """A usable GFN2-xTB parameter cache is unavailable."""


class _GFN2ParameterRefreshError(RuntimeError):
    """A parameter refresh failed before installing a usable cache."""


class _GFN2ParameterSourceRefreshError(_GFN2ParameterRefreshError):
    """The upstream download or source parsing failed."""


class _GFN2ParameterCacheWriteError(_GFN2ParameterRefreshError):
    """A refreshed parameter cache could not be written locally."""


class _GFN2ParameterCacheMalformedError(ValueError):
    """A parsed cache does not contain structurally usable records."""


class _GFN2ParameterSourceMalformedError(ValueError):
    """An upstream parameter record cannot be mapped onto a shell."""


_UPSTREAM_REFRESH_FAILED = "An upstream refresh was attempted and failed."
_CACHE_WRITE_FAILED = (
    "The upstream parameters were downloaded, but the local cache update "
    "could not be written; any existing cache file was left unchanged."
)
_CACHE_WRITE_REMEDY = (
    "check that the cache directory is writable and has free space, then "
    "retry the refresh"
)


def _unavailable_message(
    detail: str,
    *,
    remedy: str,
    secondary: str | None = None,
) -> str:
    parts = [f"GFN2ParameterUnavailable: {detail}."]
    if secondary is not None:
        parts.append(secondary)
    parts.extend(
        [
            "Native GFN2-xTB requires the published LGPL-3.0 parameter file.",
            f"Remedy: {remedy}.",
        ]
    )
    return " ".join(parts)


def _missing_cache_message(
    *,
    secondary: str | None = None,
    remedy: str | None = None,
) -> str:
    return _unavailable_message(
        "GFN2-xTB parameters are not present in the local vibe-qc cache at "
        f"{_CACHE_FILE}",
        remedy=remedy
        or (
            "run once on an internet-connected host to populate this cache, or "
            "copy a current cache file to that path"
        ),
        secondary=secondary,
    )


def _malformed_cache_message(
    *,
    secondary: str | None = None,
    remedy: str | None = None,
) -> str:
    return _unavailable_message(
        f"the local GFN2-xTB parameter cache at {_CACHE_FILE} is present but "
        "unreadable or malformed",
        remedy=remedy
        or (
            "remove or repair that file, then run once on an "
            "internet-connected host to create a current cache"
        ),
        secondary=secondary,
    )


def _stale_cache_message(
    reason: str,
    *,
    secondary: str | None = None,
    remedy: str | None = None,
) -> str:
    return _unavailable_message(
        f"the local GFN2-xTB parameter cache at {_CACHE_FILE} is present but "
        f"out of date; the parameter cache is stale: {reason}",
        remedy=remedy
        or (
            "refresh that file on an internet-connected host with "
            "load_gfn2_params(force_refetch=True), or replace it with a "
            "current cache"
        ),
        secondary=secondary,
    )


def _cache_problem_message(
    kind: str,
    reason: str | None,
    *,
    secondary: str | None = None,
    remedy: str | None = None,
) -> str:
    if kind == "missing":
        return _missing_cache_message(secondary=secondary, remedy=remedy)
    if kind == "malformed":
        return _malformed_cache_message(secondary=secondary, remedy=remedy)
    if kind == "stale" and reason is not None:
        return _stale_cache_message(
            reason,
            secondary=secondary,
            remedy=remedy,
        )
    raise ValueError(f"invalid GFN2 cache problem kind: {kind!r}")


def _refresh_failure_message() -> str:
    return _unavailable_message(
        "an upstream refresh was explicitly requested for the GFN2-xTB "
        f"parameter cache at {_CACHE_FILE} but failed",
        remedy=(
            "retry on an internet-connected host, or copy a current cache "
            "file to that path"
        ),
    )


def _refresh_write_failure_message() -> str:
    return _unavailable_message(
        "an upstream refresh was explicitly requested for the GFN2-xTB "
        f"parameter cache at {_CACHE_FILE}, but the local cache update could "
        "not be written",
        remedy=_CACHE_WRITE_REMEDY,
    )


def gfn2_parameter_cache_path() -> Path:
    """Return the local cache path used for the fetched GFN2-xTB parameters."""
    return _CACHE_FILE


def _source_sha256_from_cache_bytes(cache_bytes: bytes) -> str:
    """Extract a valid upstream-source hash from the exact parsed bytes."""
    for raw_line in cache_bytes.splitlines():
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("# source_sha256"):
            continue
        value = line.partition("=")[2].strip().strip('"').lower()
        if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
            return value
        break
    return "unavailable"


def _read_cached_toml(*, force_reload: bool = False) -> dict[str, Any]:
    global _PARAM_TOML_CACHE
    stat = _CACHE_FILE.stat()
    key = (_CACHE_FILE.resolve(), int(stat.st_mtime_ns), int(stat.st_size))
    if not force_reload and _PARAM_TOML_CACHE is not None:
        (
            cached_path,
            cached_mtime_ns,
            cached_size,
            cached_data,
            _cached_sha256,
            _cached_source_sha256,
        ) = _PARAM_TOML_CACHE
        if (cached_path, cached_mtime_ns, cached_size) == key:
            return cached_data
    cache_bytes = _CACHE_FILE.read_bytes()
    parse_stream = io.BytesIO(cache_bytes)
    parse_stream.name = str(_CACHE_FILE)
    data = tomllib.load(parse_stream)
    _PARAM_TOML_CACHE = (
        *key,
        data,
        hashlib.sha256(cache_bytes).hexdigest(),
        _source_sha256_from_cache_bytes(cache_bytes),
    )
    return data


def _lineage_for_cached_data(data: dict[str, Any]) -> dict[str, str] | None:
    """Return lineage only when ``data`` is the exact cached parse object."""
    if _PARAM_TOML_CACHE is None:
        return None
    cache_path, _, _, cached_data, cache_sha256, source_sha256 = (
        _PARAM_TOML_CACHE
    )
    if cached_data is not data:
        return None
    return {
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256,
        "cache_source_sha256": source_sha256,
        "cache_source_url": _XTB_PARAM_URL,
    }


def gfn2_parameter_lineage(
    params: _xtb.GFN2ParameterSet,
) -> dict[str, str] | None:
    """Return cache lineage bound to this exact, still-matching builder.

    Custom builders and builders mutated after loading deliberately return
    ``None``. This prevents a result from claiming whatever artifact happens
    to occupy the process-global cache path when its parameters came from
    different content.
    """
    lineage = _PARAM_OBJECT_LINEAGE.get(params)
    if lineage is None:
        return None
    try:
        current_sha256 = str(params.content_sha256())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if current_sha256 != lineage["parameter_sha256"]:
        return None
    return dict(lineage)


def _cache_staleness_reason(data: dict[str, Any]) -> str | None:
    """Return why a cache is stale, raising for structurally malformed data."""
    elems = data.get("element")
    if not isinstance(elems, list) or not elems:
        raise _GFN2ParameterCacheMalformedError

    for elem_blob in elems:
        if not isinstance(elem_blob, dict):
            raise _GFN2ParameterCacheMalformedError
        if "Z" not in elem_blob:
            raise _GFN2ParameterCacheMalformedError
        shells = elem_blob.get("shells")
        if not isinstance(shells, list) or not shells:
            raise _GFN2ParameterCacheMalformedError
        if any(not isinstance(shell, dict) for shell in shells):
            raise _GFN2ParameterCacheMalformedError

    try:
        for elem_blob in elems:
            int(elem_blob["Z"])
            for key, default in (
                ("gam", 0.5),
                ("gam3", 0.0),
                ("alpha", 1.0),
                ("dpol", 0.0),
                ("qpol", 0.0),
                ("mp_rad", 0.0),
                ("mp_vcn", 0.0),
                ("repa", 0.0),
                ("repb", 0.0),
            ):
                if not math.isfinite(float(elem_blob.get(key, default))):
                    raise ValueError
            for shell in elem_blob["shells"]:
                int(shell.get("l", 0))
                if "n" in shell:
                    int(shell["n"])
                for key, default in (
                    ("en", 0.0),
                    ("zeta", 1.0),
                    ("k_en", 1.0),
                    ("kcn", 0.0),
                    ("poly", 0.0),
                ):
                    if not math.isfinite(float(shell.get(key, default))):
                        raise ValueError

        has_shell_n = any(
            "n" in sh
            for elem_blob in elems
            for sh in elem_blob.get("shells", [])
        )
        if not has_shell_n:
            return "the cache predates the shell principal-quantum-number field"

        if data.get("projection") != "per-l":
            return (
                "the cache predates the per-angular-momentum kcn/poly "
                "projection (issue #43): d-first transition-metal shells "
                "carry rotated coordination and polynomial parameters"
            )

        has_shell_hubbard = any(
            abs(float(sh.get("k_en", 1.0)) - 1.0) > 1e-12
            for elem_blob in elems
            for sh in elem_blob.get("shells", [])
            if int(sh.get("l", 0)) > 0
        )
        bad_shell_hubbard = any(
            (
                float(sh.get("k_en", 1.0)) <= 0.0
                or float(sh.get("k_en", 1.0)) > 5.0
            )
            for elem_blob in elems
            for sh in elem_blob.get("shells", [])
            if int(sh.get("l", 0)) > 0
        )
        bad_gam3_scale = any(
            abs(float(elem_blob.get("gam3", 0.0))) > 0.5
            for elem_blob in elems
        )
        bad_mp_vcn = any(
            Z in _MP_VCN
            and abs(float(elem_blob.get("mp_vcn", -1.0)) - _MP_VCN[Z]) > 1e-12
            for elem_blob in elems
            for Z in [int(elem_blob.get("Z", 0))]
        )
    except (TypeError, ValueError, OverflowError):
        raise _GFN2ParameterCacheMalformedError from None

    if "dpol" not in elems[0]:
        return "the cache predates the faithful-AES multipole fields"
    if not has_shell_hubbard:
        return "the cache predates shell-Hubbard multipliers"
    if bad_shell_hubbard:
        return "the cache contains invalid shell-Hubbard multipliers"
    if bad_gam3_scale:
        return "the cache uses obsolete GAM3 scaling"
    if bad_mp_vcn:
        return "the cache contains obsolete multipole valence-CN data"
    return None


def _cache_state(
    *,
    force_reload: bool = False,
) -> tuple[dict[str, Any] | None, str, str | None]:
    """Return ``(data, kind, reason)`` for the cache without opening network."""
    try:
        data = _read_cached_toml(force_reload=force_reload)
    except FileNotFoundError:
        return None, "missing", None
    except (OSError, tomllib.TOMLDecodeError):
        return None, "malformed", None

    try:
        reason = _cache_staleness_reason(data)
    except _GFN2ParameterCacheMalformedError:
        return data, "malformed", None
    if reason is not None:
        return data, "stale", reason
    return data, "current", None


def gfn2_parameter_cache_available() -> bool:
    """Return ``True`` only when a current GFN2-xTB cache is locally present.

    This is a cache-only preflight probe for offline batch planning. It never
    opens the network and does not construct the native parameter objects.
    """
    _data, kind, _reason = _cache_state()
    return kind == "current"


def require_gfn2_parameter_cache() -> None:
    """Raise if a current GFN2-xTB cache is not locally available.

    Batch generators on offline hosts can call this before submitting GFN2 rows
    and mark the route unavailable once, instead of discovering the same cache
    miss inside every calculation script.
    """
    _data, kind, reason = _cache_state()
    if kind != "current":
        raise GFN2ParameterUnavailableError(
            _cache_problem_message(kind, reason)
        ) from None


def _read_cached_source_sha256() -> str | None:
    """Return the upstream source SHA-256 stored in the TOML cache header.

    The SHA is recorded as a ``# source_sha256 = "..."`` comment by
    :func:`_fetch_and_parse`.  Returns ``None`` if the cache predates
    this feature.
    """
    try:
        raw = _CACHE_FILE.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("# source_sha256"):
            # Extract the quoted hex string from '# source_sha256 = "abc..."'
            if '"' in stripped:
                return stripped.split('"')[1]
            return None
    return None


def gfn2_upstream_source_sha256() -> str | None:
    """Return the SHA-256 of the current upstream parameter file, or ``None``.

    Makes a network request to the Grimme group's GitHub repository.
    Returns ``None`` on any error (offline, timeout, upstream unavailable) --
    callers should treat ``None`` as "cannot determine" rather than "unchanged".
    """
    import hashlib
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(_XTB_PARAM_URL, timeout=15) as r:
            body = r.read()
    except (OSError, TimeoutError, urllib.error.URLError):
        return None
    return hashlib.sha256(body).hexdigest()


def gfn2_parameter_cache_matches_upstream() -> bool | None:
    """Return whether the cached parameters match the current upstream source.

    Returns:
        ``True`` -- cached source SHA matches upstream.
        ``False`` -- upstream source has changed since the cache was populated.
        ``None`` -- cannot determine (offline, cache predates SHA tracking, etc.).
    """
    cached_sha = _read_cached_source_sha256()
    if cached_sha is None:
        return None
    upstream_sha = gfn2_upstream_source_sha256()
    if upstream_sha is None:
        return None
    return cached_sha == upstream_sha


_SYMBOL_TO_Z = {
    "h": 1,
    "he": 2,
    "li": 3,
    "be": 4,
    "b": 5,
    "c": 6,
    "n": 7,
    "o": 8,
    "f": 9,
    "ne": 10,
    "na": 11,
    "mg": 12,
    "al": 13,
    "si": 14,
    "p": 15,
    "s": 16,
    "cl": 17,
    "ar": 18,
    "k": 19,
    "ca": 20,
    "sc": 21,
    "ti": 22,
    "v": 23,
    "cr": 24,
    "mn": 25,
    "fe": 26,
    "co": 27,
    "ni": 28,
    "cu": 29,
    "zn": 30,
    "ga": 31,
    "ge": 32,
    "as": 33,
    "se": 34,
    "br": 35,
    "kr": 36,
    "rb": 37,
    "sr": 38,
    "y": 39,
    "zr": 40,
    "nb": 41,
    "mo": 42,
    "tc": 43,
    "ru": 44,
    "rh": 45,
    "pd": 46,
    "ag": 47,
    "cd": 48,
    "in": 49,
    "sn": 50,
    "sb": 51,
    "te": 52,
    "i": 53,
    "xe": 54,
    "cs": 55,
    "ba": 56,
    "la": 57,
    "hf": 72,
    "ta": 73,
    "w": 74,
    "re": 75,
    "os": 76,
    "ir": 77,
    "pt": 78,
    "au": 79,
    "hg": 80,
    "tl": 81,
    "pb": 82,
    "bi": 83,
    "rn": 86,
}
_L_TO_NUM = {"s": 0, "p": 1, "d": 2, "f": 3}

# --- Faithful-AES per-element multipole radius / valence CN ------------------
# tblite GFN2 p_rad (bohr) and p_vcn, hardcoded in tblite src/tblite/xtb/gfn2.f90
# (Bannwarth, Ehlert & Grimme, JCTC 2019, 15, 1652, doi:10.1021/acs.jctc.8b01176).
# Only the values verified against the tblite source are listed; elements not
# present fall back in the C++ faithful-AES path.  NOT fabricated (CLAUDE.md §8).
_MP_RAD_BOHR = {
    1: 1.4, 2: 3.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 3.0, 7: 1.9, 8: 1.8, 9: 2.4, 10: 5.0,
}
_MP_VCN = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 2.0,
    5: 3.0,
    6: 3.0,
    7: 3.0,
    8: 2.0,
    9: 1.0,
    10: 1.0,
}


def _parse_ao_list(ao_str: str) -> list[tuple[int, int]]:
    result = []
    for m in re.finditer(r"(\d)([spdf])", ao_str):
        lv = _L_TO_NUM.get(m.group(2))
        if lv is not None:
            # The digit is the shell's principal quantum number (xtb
            # slaterToGauss selects the STO-NG row by n).
            result.append((int(m.group(1)), lv))
    return result


def _parse_z(header: str) -> int:
    """Parse '$Z=10 Tue...' or '$Z= 1 Wed...' -> atomic number."""
    parts = header.split()
    first = parts[0]
    if "=" in first:
        token = first.split("=", 1)[1]
        if not token and len(parts) > 1:
            token = parts[1]
    elif len(parts) > 1:
        token = parts[1]
    else:
        return 0
    if token.isdigit():
        return int(token)
    return _SYMBOL_TO_Z.get(token.lower(), 0)


def _fetch_and_parse() -> None:
    """Refresh the cache without exposing transport or parser internals."""
    try:
        _fetch_and_parse_impl()
    except _GFN2ParameterRefreshError:
        raise
    except (UnicodeError, ValueError, TypeError, IndexError):
        raise _GFN2ParameterSourceRefreshError from None


# The upstream file names its per-angular-momentum records by SHELL
# LETTER -- KCNS/KCNP/KCND and POLYS/POLYP/POLYD -- while every consumer
# indexes the derived arrays by the shell's angular momentum l. The
# letter IS the mapping; the line position only coincides with it. The
# published convention is per-l: Bannwarth, Ehlert & Grimme, J. Chem.
# Theory Comput. 15, 1652-1671 (2019), doi:10.1021/acs.jctc.8b01176,
# eq 17 (H^l_CN) and eq 19 (k^poly_{A,l}), tabulated per shell label in
# SI Table S52.
_SHELL_RECORD_L = {"S": 0, "P": 1, "D": 2}
_SHELL_RECORD_FAMILIES = ("KCN", "POLY")


def _parse_shell_record(line: str) -> tuple[str, str, int, float] | None:
    """Map one ``KCN<X>=``/``POLY<X>=`` record to its angular momentum.

    Returns ``(family, keyword, l, value)``, or ``None`` when ``line`` is
    not one of those records.

    Keying on the shell letter rather than on the record's position in
    the block is what makes the l assignment explicit. Collecting the
    records positionally is correct only while every block happens to
    list them in s, p, d order -- true of every one of the 86 blocks in
    the upstream revision current at 2026-08-28, but not enforced by
    anything. A revision that emitted KCND before KCNS would re-create
    the exact 3-cycle of issue #43 (re-attested under issue #446): the
    rotated cache is still structurally valid, so nothing fails; the
    loaded content digest merely moves off ``GFN2_PUBLISHED_SHA256``,
    which reads as tampering rather than as a mis-read source. Issue
    #467.

    A keyword in one of those families whose shell letter is not s/p/d
    (including a bare ``KCN=``/``POLY=``), or a payload that is not a
    single number, raises instead: both mean the upstream layout changed
    in a way this converter cannot map onto an l, and guessing would
    rotate the parameters silently.
    """
    keyword, separator, payload = line.partition("=")
    keyword = keyword.strip()
    if not separator:
        return None
    for family in _SHELL_RECORD_FAMILIES:
        if keyword.startswith(family):
            break
    else:
        return None
    letter = keyword[len(family) :]
    if letter not in _SHELL_RECORD_L:
        raise _GFN2ParameterSourceMalformedError(
            f"upstream parameter record {keyword!r} names shell "
            f"{letter!r}, which this converter cannot map to an angular "
            f"momentum (known: {', '.join(sorted(_SHELL_RECORD_L))})"
        )
    fields = payload.strip().split()
    if len(fields) != 1:
        raise _GFN2ParameterSourceMalformedError(
            f"upstream parameter record {keyword!r} carries "
            f"{len(fields)} values; exactly one per shell letter is "
            f"expected"
        )
    return family, keyword, _SHELL_RECORD_L[letter], float(fields[0])


def _render_parameter_cache(text: str) -> str:
    """Convert upstream ``param_gfn2-xtb.txt`` text into the cached TOML.

    Split out of :func:`_fetch_and_parse_impl` so the converter can be
    exercised on a synthetic parameter block without opening the network
    (issue #467). Structural validation of the rendered document stays
    with the caller.
    """
    source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    lines = text.split("\n")
    toml_lines = [
        "# GFN2-xTB parameters (Grimme group, 2019)",
        "# DOI: 10.1021/acs.jctc.8b01176",
        "# License: LGPL-3.0 (fetched at runtime, not bundled)",
        f'# source_sha256 = "{source_sha256}"',
        "",
        # Converter-generation marker: per-angular-momentum kcn/poly
        # projection (issue #43). Caches without it were baked with the
        # positional rotation and are treated as stale.
        'projection = "per-l"',
        "",
    ]

    # Elements
    current_z = 0
    ao_list: list[int] = []
    lev_values: list[float] = []
    exp_values: list[float] = []
    kcn_by_l: dict[int, float] = {}
    poly_by_l: dict[int, float] = {}
    gam_value = 0.5
    gam3_value = 0.0
    repa_value = 0.0
    repb_value = 0.0
    dpol_value = 0.0
    qpol_value = 0.0
    lparp_value = 1.0
    lpard_value = 1.0

    def flush():
        nonlocal \
            current_z, \
            ao_list, \
            lev_values, \
            exp_values, \
            kcn_by_l, \
            poly_by_l, \
            gam_value, \
            gam3_value, \
            repa_value, \
            repb_value, \
            dpol_value, \
            qpol_value, \
            lparp_value, \
            lpard_value
        if current_z == 0:
            return
        toml_lines.append("[[element]]")
        toml_lines.append(f"Z = {current_z}")
        toml_lines.append("shells = [")
        for i, (n_qn, lv) in enumerate(ao_list):
            # lev/exp lines carry one value per shell IN SHELL ORDER, so
            # they index by shell position i. KCNS/KCNP/KCND and
            # POLYS/POLYP/POLYD are PER-ANGULAR-MOMENTUM named lines, so
            # they must index by the shell's l. The two orders differ
            # exactly for the d-first transition-metal blocks
            # (ao=3d4s4p: Z 21-29, 39-47, 57-79); positional indexing
            # rotated kcn/poly across shells there (issue #43).
            #
            # The per-l keying is the published convention, not a local
            # choice: Bannwarth, Ehlert & Grimme, J. Chem. Theory
            # Comput. 15, 1652-1671 (2019), doi:10.1021/acs.jctc.8b01176
            # eq 17 reads H_kk = H^l_A - H^l_CN * CN'_A for
            # (kappa in l in A), and eq 19 builds the distance
            # polynomial from k^poly_{A,l}. Both parameters are indexed
            # by angular momentum l, and the paper's Supporting
            # Information Table S52 tabulates them per shell label.
            # Published check value (SI Table S52, Cu): k^poly is
            # 0.177983 / 0.149778 / -0.265089 for 4s / 4p / 3d, which
            # this converter must place on l = 0 / 1 / 2 respectively.
            # Re-attested under issue #446; see that issue's record at
            # tests/test_gfn2_parameter_identity.py.
            #
            # The l of each kcn/poly value is fixed at parse time by the
            # record's own shell letter (_parse_shell_record), not by the
            # order the records appear in, so this lookup cannot be
            # rotated by an upstream re-ordering (issue #467).
            en = lev_values[i] if i < len(lev_values) else 0.0
            zeta = exp_values[i] if i < len(exp_values) else 1.0
            kcn = kcn_by_l.get(lv, 0.0)
            poly = poly_by_l.get(lv, 0.0)
            en_ha = en / 27.2114  # eV -> Ha
            kcn_ha = kcn * 0.1 / 27.2114  # eV -> Ha (xtb scales kcn by 0.1)
            poly_scaled = poly * 0.01  # POLY x 0.01
            # GFN2's shell-resolved second-order Coulomb kernel scales the
            # element Hubbard parameter by l-specific multipliers.  The official
            # xtb parameter file stores p/d multiplier deltas as LPARP/LPARD in
            # tenths; s shells are the unit reference.
            k_en = 1.0
            if lv == 1:
                k_en = 1.0 + 0.1 * lparp_value
            elif lv == 2:
                k_en = 1.0 + 0.1 * lpard_value
            toml_lines.append(
                f"  {{ n = {n_qn}, l = {lv}, en = {en_ha:.6f}, zeta = {zeta:.6f}, "
                f"k_en = {k_en:.7f}, kcn = {kcn_ha:.6f}, "
                f"poly = {poly_scaled:.6f}}},"
            )
        toml_lines.append("]")
        toml_lines.append(f"gam = {gam_value:.6f}")
        toml_lines.append(f"gam3 = {gam3_value:.6f}")
        toml_lines.append(f"repa = {repa_value:.6f}")
        toml_lines.append(f"repb = {repb_value:.6f}")
        # Faithful-AES per-element data (DPOL/QPOL × 0.01; p_rad/p_vcn tables).
        toml_lines.append(f"dpol = {dpol_value * 0.01:.8f}")
        toml_lines.append(f"qpol = {qpol_value * 0.01:.8f}")
        toml_lines.append(f"mp_rad = {_MP_RAD_BOHR.get(current_z, 0.0):.6f}")
        toml_lines.append(f"mp_vcn = {_MP_VCN.get(current_z, 0.0):.6f}")
        toml_lines.append("")
        current_z = 0
        ao_list.clear()
        lev_values.clear()
        exp_values.clear()
        kcn_by_l.clear()
        poly_by_l.clear()
        gam_value = 0.5
        gam3_value = 0.0
        repa_value = 0.0
        repb_value = 0.0
        dpol_value = 0.0
        qpol_value = 0.0
        lparp_value = 1.0
        lpard_value = 1.0

    for line in lines:
        s = line.strip()
        if s.startswith("$Z="):
            flush()
            current_z = _parse_z(s)
        elif s == "$end" and current_z:
            flush()
        elif current_z:
            if s.startswith("ao="):
                ao_list = _parse_ao_list(s.split("=")[1].strip())
            elif s.startswith("lev="):
                lev_values = [float(x) for x in s.split("=")[1].strip().split()]
            elif s.startswith("exp="):
                exp_values = [float(x) for x in s.split("=")[1].strip().split()]
            elif (record := _parse_shell_record(s)) is not None:
                family, keyword, record_l, value = record
                store = kcn_by_l if family == "KCN" else poly_by_l
                if record_l in store:
                    raise _GFN2ParameterSourceMalformedError(
                        f"element block Z={current_z} repeats the "
                        f"{keyword!r} record; one value per shell letter "
                        f"is expected"
                    )
                store[record_l] = value
            elif s.startswith("GAM=") and "GAM3" not in s:
                try:
                    gam_value = float(s.split("=")[1].strip())
                except ValueError:
                    pass
            elif s.startswith("GAM3="):
                try:
                    # tblite's GFN2 source stores the third-order Hubbard
                    # derivatives as 0.1 * GAM3.  The xTB text parameter file
                    # carries the unscaled values, so apply the same factor
                    # when building vibe-qc's cached TOML.
                    gam3_value = 0.1 * float(s.split("=")[1].strip())
                except ValueError:
                    pass
            elif s.startswith("REPA="):
                try:
                    repa_value = float(s.split("=")[1].strip())
                except ValueError:
                    pass
            elif s.startswith("REPB="):
                try:
                    repb_value = float(s.split("=")[1].strip())
                except ValueError:
                    pass
            elif s.startswith("DPOL="):
                try:
                    dpol_value = float(s.split("=")[1].strip())
                except ValueError:
                    pass
            elif s.startswith("QPOL="):
                try:
                    qpol_value = float(s.split("=")[1].strip())
                except ValueError:
                    pass
            elif s.startswith("LPARP="):
                try:
                    lparp_value = float(s.split("=")[1].strip())
                except ValueError:
                    pass
            elif s.startswith("LPARD="):
                try:
                    lpard_value = float(s.split("=")[1].strip())
                except ValueError:
                    pass
    flush()

    return "\n".join(toml_lines)


def _fetch_and_parse_impl() -> None:
    """Download param_gfn2-xtb.txt, parse to TOML, cache locally."""
    import tempfile
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(_XTB_PARAM_URL, timeout=30) as r:
            text = r.read().decode("utf-8")
    except (OSError, TimeoutError, urllib.error.URLError):
        raise _GFN2ParameterSourceRefreshError from None

    rendered_cache = _render_parameter_cache(text)
    try:
        rendered_data = tomllib.loads(rendered_cache)
        rendered_staleness = _cache_staleness_reason(rendered_data)
        rendered_atomic_numbers = sorted(
            int(elem_blob["Z"])
            for elem_blob in rendered_data["element"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        tomllib.TOMLDecodeError,
        _GFN2ParameterCacheMalformedError,
    ):
        raise _GFN2ParameterSourceRefreshError from None
    if rendered_staleness is not None or rendered_atomic_numbers != list(
        range(1, 87)
    ):
        raise _GFN2ParameterSourceRefreshError

    temporary_path: Path | None = None
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=_CACHE_DIR,
            prefix=f".{_CACHE_FILE.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(rendered_cache)
        temporary_path.replace(_CACHE_FILE)
    except OSError:
        raise _GFN2ParameterCacheWriteError from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _refresh_or_raise(
    kind: str | None = None,
    reason: str | None = None,
    *,
    forced: bool = False,
) -> None:
    try:
        _fetch_and_parse()
    except _GFN2ParameterCacheWriteError:
        message = (
            _refresh_write_failure_message()
            if forced
            else _cache_problem_message(
                kind or "missing",
                reason,
                secondary=_CACHE_WRITE_FAILED,
                remedy=_CACHE_WRITE_REMEDY,
            )
        )
        raise GFN2ParameterUnavailableError(message) from None
    except _GFN2ParameterRefreshError:
        message = (
            _refresh_failure_message()
            if forced
            else _cache_problem_message(
                kind or "missing",
                reason,
                secondary=_UPSTREAM_REFRESH_FAILED,
            )
        )
        raise GFN2ParameterUnavailableError(message) from None


def load_gfn2_params(
    force_refetch: bool = False,
    *,
    allow_fetch: bool = True,
) -> _xtb.GFN2ParameterSet:
    """Load GFN2-xTB parameters, fetching from Grimme group repo if needed.

    Parameters are cached at ~/.cache/vibeqc/gfn2_xtb_params.toml.
    On first call, downloads and parses the published GFN2-xTB
    parameter set (LGPL-3.0 licensed, fetched at runtime).
    Set ``allow_fetch=False`` for offline preflight/cache-only loading; in
    that mode missing, malformed, or stale caches raise
    ``GFN2ParameterUnavailableError`` without opening the network.

    Returns a GFN2ParameterSet with 86 elements (H-Rn).
    """
    if force_refetch and not allow_fetch:
        raise ValueError(
            "load_gfn2_params: force_refetch=True requires allow_fetch=True"
        )

    if force_refetch:
        _refresh_or_raise(forced=True)
        data, kind, reason = _cache_state(force_reload=True)
        if kind != "current":
            raise GFN2ParameterUnavailableError(
                _cache_problem_message(
                    kind,
                    reason,
                    secondary=(
                        "The explicitly requested upstream refresh did not "
                        "produce a usable cache."
                    ),
                )
            ) from None
    else:
        data, kind, reason = _cache_state()
        if kind != "current":
            if not allow_fetch:
                raise GFN2ParameterUnavailableError(
                    _cache_problem_message(kind, reason)
                ) from None
            original_kind, original_reason = kind, reason
            _refresh_or_raise(kind, reason)
            data, kind, reason = _cache_state(force_reload=True)
            if kind != "current":
                raise GFN2ParameterUnavailableError(
                    _cache_problem_message(
                        original_kind,
                        original_reason,
                        secondary=(
                            "The attempted upstream refresh did not produce a "
                            "usable cache."
                        ),
                    )
                ) from None

    assert data is not None

    params = _xtb.GFN2ParameterSet()
    rep_params: dict[int, tuple[float, float]] = {}
    for elem_blob in data.get("element", []):
        Z = int(elem_blob["Z"])
        gam = float(elem_blob.get("gam", 0.5))
        shells_data = elem_blob.get("shells", [])

        ed = _xtb.GFN2ElementData()
        ed.Z = Z
        ed.gam = gam
        ed.gam3 = float(elem_blob.get("gam3", 0.0))
        ed.alpha = float(elem_blob.get("alpha", 1.0))
        # Faithful-AES per-element data (consumed only when aes_faithful=True).
        ed.dpol = float(elem_blob.get("dpol", 0.0))
        ed.qpol = float(elem_blob.get("qpol", 0.0))
        ed.mp_rad = float(elem_blob.get("mp_rad", 0.0))
        ed.mp_vcn = float(elem_blob.get("mp_vcn", 0.0))
        repa = float(elem_blob.get("repa", 0.0))
        repb = float(elem_blob.get("repb", 0.0))
        for sh in shells_data:
            ed.add_shell(
                int(sh.get("l", 0)),
                float(sh.get("en", 0.0)),
                float(sh.get("zeta", 1.0)),
                float(sh.get("k_en", 1.0)),
                float(sh.get("kcn", 0.0)),
                float(sh.get("poly", 0.0)),
                int(sh.get("n", 0)),
            )
        params.add_element(ed)
        # REPA = a repulsion exponent, REPB = effective nuclear charge Z_eff;
        # both feed the pairwise repulsion built below (they are not stored on
        # the C++ element record).
        rep_params[Z] = (repa, repb)

    # Compute pairwise repulsion from elemental REPA (a) / REPB (Z_eff).
    _compute_pairwise_repulsive(params, rep_params)

    cache_lineage = _lineage_for_cached_data(data)
    if cache_lineage is not None:
        cache_lineage["parameter_sha256"] = str(params.content_sha256())
        _PARAM_OBJECT_LINEAGE[params] = cache_lineage

    # If the network is available, check whether the cached parameter set
    # matches the current upstream source.  A mismatch means the Grimme group
    # has updated param_gfn2-xtb.txt on GitHub since the cache was populated;
    # the energies from this run may differ from those of a freshly-fetched
    # installation.  This is a non-fatal warning; the calculation proceeds
    # with the cached parameters so the result is reproducible.
    if allow_fetch:
        match = gfn2_parameter_cache_matches_upstream()
        if match is False:
            import warnings

            warnings.warn(
                f"GFN2-xTB: the cached parameter file ({_CACHE_FILE}) "
                f"was fetched from an older version of the upstream "
                f"source than what is currently published at "
                f"{_XTB_PARAM_URL}. The cached parameters will be used "
                f"(results are reproducible), but they may not match "
                f"the latest published parameter set. To update, call "
                f"load_gfn2_params(force_refetch=True) or delete the "
                f"cache file.",
                stacklevel=2,
            )

    return params


def _compute_pairwise_repulsive(
    params: _xtb.GFN2ParameterSet, rep_params: dict[int, tuple[float, float]]
) -> None:
    """Build GFN2-xTB pairwise repulsion from per-element REPA (a) / REPB (Z_eff).

    The combined parameters consumed by the C++ ``repulsive_energy`` are
    ``alpha = √(aᴬ.aᴮ)`` and ``k_ab = Zᴬ_eff.Zᴮ_eff`` for the form
    ``(k_ab/R).exp(-alpha.R^{3/2})`` (Bannwarth, Ehlert & Grimme, JCTC 2019,
    Eq. 6). Previously this used ``hubbard_u`` (the chemical hardness, not the
    repulsion parameters) with an arbitrary ``0.1`` scaling and the wrong
    functional form.
    """
    import math

    for Z1, (alpha1, zeff1) in rep_params.items():
        for Z2, (alpha2, zeff2) in rep_params.items():
            if Z2 < Z1:
                continue
            rp = _xtb.GFN2RepulsivePair()
            rp.alpha = math.sqrt(abs(alpha1 * alpha2))
            rp.k_ab = zeff1 * zeff2
            params.set_repulsive_pair(Z1, Z2, rp)
