"""COD (Crystallography Open Database) direct CIF client.

When the user already knows a COD ID (typical for organic molecular
crystals or experimental refinements with no MP entry), going via
OPTIMADE is overkill -- COD ships every entry as a CIF at a stable URL:

    https://www.crystallography.net/cod/<cod_id>.cif

ASE's ``ase.io.read(..., format="cif")`` parses these directly. We
convert the resulting :class:`ase.Atoms` to a :class:`PeriodicSpec`
and stamp full :class:`Provenance`.

COD CIFs are effectively immutable once published, so the cache uses
infinite TTL (``COD_TTL_SECONDS = None``) per Sec. 7.

See ``docs/tutorial/external_data_fetcher.md`` Sec. 3.5, Sec. 11.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from examples.regression.core.spec import (
    AtomFrac,
    PeriodicSpec,
    Provenance,
)

from . import __version__ as FETCHER_VERSION
from .cache import COD_TTL_SECONDS, FetchCache, FetchCacheMiss
from .client_optimade import (
    OPTIMADE_USER_AGENT,
    _atomic_number,
    _slugify,
)
from .heuristics import (
    open_shell_default,
    pick_damping,
    pick_initial_guess,
    pick_kmesh,
    pick_recommended_basis,
)


COD_CIF_URL = "https://www.crystallography.net/cod/{id}.cif"
COD_PERMALINK = "https://www.crystallography.net/cod/result.php?CODSEARCH={id}"
COD_LICENSE = "CC0"


def _download_cif(cod_id: str) -> str:
    url = COD_CIF_URL.format(id=cod_id)
    req = Request(url, headers={"User-Agent": OPTIMADE_USER_AGENT})
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(
            f"COD download failed for id {cod_id!r} ({url}): {exc}"
        ) from exc


def _ase_to_periodic_spec(
    *,
    atoms,
    cod_id: str,
    cif_text: str,
    quick: bool,
    slug_override: Optional[str],
) -> PeriodicSpec:
    import numpy as np

    cell = atoms.get_cell().array  # 3x3, Å
    if cell.shape != (3, 3):
        raise ValueError(
            f"COD CIF parser returned non-3x3 cell (shape {cell.shape})"
        )
    diag = np.diag(cell)
    off = np.abs(cell - np.diag(diag))
    on_scale = float(np.linalg.norm(diag))
    if (off.max() if off.size else 0.0) > 1e-3 * on_scale:
        raise NotImplementedError(
            f"COD entry {cod_id} has a non-orthorhombic cell; vibe-qc "
            "EWALD_3D currently requires an orthorhombic conventional "
            "cell. Pick a different polymorph or wait for triclinic "
            "support (docs/roadmap.md)."
        )

    fracs = atoms.get_scaled_positions(wrap=True)
    syms = list(atoms.get_chemical_symbols())
    atoms_frac = []
    for sym, f in zip(syms, fracs):
        atoms_frac.append(
            AtomFrac(
                symbol=sym, z=_atomic_number(sym),
                frac=tuple(float(v) for v in f),
            )
        )

    formula = atoms.get_chemical_formula(empirical=True) or syms[0]
    n_atoms = len(atoms_frac)
    initial_guess = pick_initial_guess(syms)
    damping = pick_damping(syms)
    kmesh = pick_kmesh(n_atoms)
    recommended_basis = pick_recommended_basis(
        symbols=syms, is_periodic=True, n_atoms=n_atoms, quick=quick,
    )
    is_open, mag = open_shell_default(is_periodic=True)

    base_id = slug_override or f"cod_{_slugify(cod_id)}"
    family = "rocksalt" if formula in ("MgO", "NaCl", "LiH", "KCl") else (
        "diamond" if formula in ("C", "Si") else "fetched"
    )

    # Try to extract space group + DOI from the CIF text.
    space_group = ""
    doi = ""
    for line in cif_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("_symmetry_space_group_name_H-M") or \
           s.startswith("_space_group_name_H-M_alt") or \
           s.startswith("_space_group.IT_number"):
            parts = s.split(None, 1)
            if len(parts) == 2:
                space_group = parts[1].strip().strip("'\"")
        if s.startswith("_journal_paper_doi"):
            parts = s.split(None, 1)
            if len(parts) == 2:
                doi = parts[1].strip().strip("'\"")

    prov = Provenance(
        source_db="COD",
        source_id=str(cod_id),
        source_url=COD_PERMALINK.format(id=cod_id),
        original_reference=doi,
        license=COD_LICENSE,
        fetched_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        fetcher_version=FETCHER_VERSION,
        notes="",
    )

    return PeriodicSpec(
        id=base_id,
        family=family,
        lattice_ang=tuple(tuple(float(v) for v in row) for row in cell),
        space_group=space_group,
        atoms=tuple(atoms_frac),
        default_kmesh=kmesh,
        default_initial_guess=initial_guess,
        default_damping=damping,
        notes=(
            f"Fetched from COD ({cod_id}). "
            f"# TODO: k-mesh convergence -- see input-k-mesh-convergence.py."
        ),
        citation=doi or COD_PERMALINK.format(id=cod_id),
        recommended_basis=recommended_basis,
        magnetic_moments=mag,
        is_open_shell=is_open,
        provenance=prov,
    )


def fetch_cod(
    *,
    cod_id: str,
    cache: Optional[FetchCache] = None,
    use_cache: bool = True,
    cache_only: bool = False,
    quick: bool = False,
    slug_override: Optional[str] = None,
    cif_path: Optional[Path] = None,
) -> PeriodicSpec:
    """Fetch a single COD entry by id and return a ``PeriodicSpec``.

    ``cif_path`` is an escape hatch for tests: when set, read the CIF
    from disk instead of hitting the live HTTP endpoint.
    """
    if cache is None:
        cache = FetchCache()
    source_db = "COD"

    if use_cache:
        cached = cache.get(source_db, cod_id, ttl_seconds=COD_TTL_SECONDS)
        if cached is not None:
            cif_text = cached.get("raw", {}).get("cif")
            if cif_text:
                from ase.io import read as ase_read

                atoms = ase_read(StringIO(cif_text), format="cif")
                return _ase_to_periodic_spec(
                    atoms=atoms, cod_id=cod_id, cif_text=cif_text,
                    quick=quick, slug_override=slug_override,
                )

    if cache_only:
        raise FetchCacheMiss(f"{source_db}/{cod_id}")

    if cif_path is not None:
        cif_text = Path(cif_path).read_text(encoding="utf-8")
    else:
        cif_text = _download_cif(cod_id)

    from ase.io import read as ase_read

    atoms = ase_read(StringIO(cif_text), format="cif")
    spec = _ase_to_periodic_spec(
        atoms=atoms, cod_id=cod_id, cif_text=cif_text,
        quick=quick, slug_override=slug_override,
    )
    cache.put(source_db, cod_id, raw={"cif": cif_text}, test_system=spec)
    return spec
