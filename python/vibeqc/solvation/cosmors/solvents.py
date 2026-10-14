"""Named solvents for COSMO-RS: a geometry library and a per-user surface cache.

``run_job(..., cosmors=CosmoRSOptions(solvent="water"))`` needs the solvent's
conductor surface at the solute's own protocol (design note Q3: the two sigma
distributions must come from the same method and basis). That surface is one
conductor COSMO run of the solvent molecule, 30 to 60 s for a small solvent at
BPW91/def2-TZVP and more for hexane, and it is the same for every solute run
at that protocol, so it is computed once and cached under the user's cache
directory (``$XDG_CACHE_HOME/vibeqc/cosmors``, by default
``~/.cache/vibeqc/cosmors``; never inside a checkout).

What is cached is the :class:`~vibeqc.solvation.surface.ConductorSurface`
record, not a sigma potential: the record is temperature-independent, so one
cached surface serves a potential at any temperature, and the file carries
the surface's full provenance. The key covers everything the surface depends
on: the solvent's geometry, the method and basis, the parameterization's
identity and radii, the fine-cavity settings, the record's schema version and
the vibe-qc version that built it. A change to any of them is a different
file; nothing is ever read back under a key it was not written with.

The geometries live in :mod:`vibeqc.solvation.cosmors.solvent_geometries`,
written by ``scripts/cosmors_solvent_library.py``, which states the protocol
each one was optimized at. They are vibe-qc's own results, and the library is
limited to the elements the shipped parameterization covers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..fine_cavity import (
    GRID_SPACING_ANG as FINE_GRID_SPACING_ANG,
    NSPA_FINE,
    NSPH_FINE,
)
from ..presets import canonicalise_solvent_name
from ..surface import SURFACE_SCHEMA_VERSION, ConductorSurface, build_conductor_surface
from .parameters import KLAMT_1998, T_ROOM_K, Parameterization
from .potential import sigma_potential_segments
from .sigma import segment_descriptors

__all__ = [
    "FINE_CAVITY_KIND",
    "available_solvents",
    "cosmors_cache_dir",
    "solvent_molecule",
    "solvent_surface",
    "solvent_sigma_potential_from_surface",
    "surface_cache_key",
]

#: The cavity kind a fine-cavity conductor surface is recorded as.
FINE_CAVITY_KIND = "fine-cfc"

_HF_TOKENS = frozenset({"hf", "rhf", "uhf", "rohf"})


def available_solvents() -> list[str]:
    """The solvents the library has a geometry for, canonical names."""
    from .solvent_geometries import SOLVENT_GEOMETRIES

    return sorted(SOLVENT_GEOMETRIES)


def _resolve_solvent_name(name: str) -> str:
    try:
        key = canonicalise_solvent_name(str(name))
    except KeyError:
        key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    from .solvent_geometries import SOLVENT_GEOMETRIES

    if key not in SOLVENT_GEOMETRIES:
        raise ValueError(
            f"COSMO-RS: no solvent geometry for {name!r}. The library covers "
            f"{', '.join(available_solvents())}. For another solvent compute "
            "its potential yourself (solvent_sigma_potential) and pass it as "
            "CosmoRSOptions(sigma_potential=...)."
        )
    return key


def solvent_molecule(name: str):
    """The library geometry of ``name`` as a :class:`vibeqc.Molecule`."""
    from vibeqc import Atom, Molecule

    from .solvent_geometries import SOLVENT_GEOMETRIES

    entry = SOLVENT_GEOMETRIES[_resolve_solvent_name(name)]
    return Molecule([Atom(int(z), tuple(float(c) for c in xyz)) for z, xyz in entry.atoms], 0, 1)


def cosmors_cache_dir(cache_dir: Optional[Path | str] = None) -> Path:
    """Root of the COSMO-RS solvent cache.

    ``cache_dir`` overrides; otherwise ``$XDG_CACHE_HOME/vibeqc/cosmors``, then
    ``~/.cache/vibeqc/cosmors``, mirroring :func:`vibeqc.basis_fetch.bse_cache_dir`.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg).expanduser() / "vibeqc" / "cosmors"
    return Path.home() / ".cache" / "vibeqc" / "cosmors"


def _method_kind(method: str) -> tuple[str, Optional[str]]:
    """``("rhf", None)`` for a Hartree-Fock token, ``("rks", functional)`` otherwise."""
    token = str(method).strip().lower()
    if not token:
        raise ValueError("COSMO-RS solvent: the method token is empty.")
    if token in _HF_TOKENS:
        return "rhf", None
    if token in {"rks", "uks", "roks", "dft", "ks"}:
        raise ValueError(
            f"COSMO-RS solvent: method {token!r} names no functional. Pass the "
            "functional's name (e.g. 'bpw91'), which is what the protocol records."
        )
    return "rks", token


def surface_cache_key(
    name: str,
    molecule,
    *,
    method: str,
    basis: str,
    params: Parameterization,
    fine_grid_spacing_ang: float,
    fine_segments_h: int,
    fine_segments_other: int,
    fine_frame: str,
) -> dict[str, Any]:
    """Everything a cached solvent surface depends on, as a plain dict."""
    from vibeqc import __version__

    atoms = list(molecule.atoms)
    return {
        "solvent": name,
        "geometry": [
            [int(a.Z)] + [round(float(c), 10) for c in a.xyz] for a in atoms
        ],
        "charge": int(molecule.charge),
        "multiplicity": int(molecule.multiplicity),
        "method": str(method).strip().lower(),
        "basis": str(basis).strip().lower(),
        "parameterization": {
            "name": params.name,
            "version": params.version,
            "cavity_radii": {str(k): float(v) for k, v in sorted(params.cavity_radii.items())},
        },
        "cavity": {
            "kind": FINE_CAVITY_KIND,
            "grid_spacing_ang": float(fine_grid_spacing_ang),
            "segments_h": int(fine_segments_h),
            "segments_other": int(fine_segments_other),
            "frame": str(fine_frame),
        },
        "surface_schema": int(SURFACE_SCHEMA_VERSION),
        "vibeqc": str(__version__),
    }


def _key_digest(key: dict[str, Any]) -> str:
    text = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(token: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(token).strip().lower())


def solvent_surface(
    name: str,
    *,
    method: str,
    basis: str,
    params: Parameterization = KLAMT_1998,
    fine_grid_spacing_ang: float = FINE_GRID_SPACING_ANG,
    fine_segments_h: int = NSPH_FINE,
    fine_segments_other: int = NSPA_FINE,
    fine_frame: str = "molecular",
    cache_dir: Optional[Path | str] = None,
    recompute: bool = False,
    tol_e_solv: float = 1e-10,
    max_macro_iter: int = 80,
) -> tuple[ConductorSurface, str]:
    """The conductor surface of a library solvent at ``method``/``basis``.

    Returns ``(surface, source)`` with ``source`` one of ``"cache"`` and
    ``"computed"``. The surface is a conductor run (``epsilon = inf``,
    ``variant="cosmo"``) on the fine cavity at the parameterization's own
    radii, unscaled, without a probe: exactly the model
    :func:`~vibeqc.solvation.cosmors.job.conductor_solvent_model` builds for
    the solute, so solute and solvent surfaces are comparable by
    construction. ``method`` is the protocol token: a Hartree-Fock name runs
    RHF, a functional's name runs RKS with it, on run_job's default KS grid.
    """
    key_name = _resolve_solvent_name(name)
    molecule = solvent_molecule(key_name)
    elements = sorted({int(a.Z) for a in molecule.atoms})
    missing = [z for z in elements if z not in params.cavity_radii]
    if missing:
        raise ValueError(
            f"COSMO-RS: {params.name} has no cavity radius for element(s) "
            f"{missing} of solvent {key_name!r}; its surface cannot be built at "
            "the radii the parameterization was fitted with."
        )
    key = surface_cache_key(
        key_name, molecule, method=method, basis=basis, params=params,
        fine_grid_spacing_ang=fine_grid_spacing_ang,
        fine_segments_h=fine_segments_h, fine_segments_other=fine_segments_other,
        fine_frame=fine_frame,
    )
    digest = _key_digest(key)
    root = cosmors_cache_dir(cache_dir)
    path = root / f"{_slug(key_name)}__{_slug(key['method'])}__{_slug(key['basis'])}__{digest[:16]}.json"

    if not recompute and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("key") == key:
            return ConductorSurface.from_dict(payload["surface"]), "cache"
        # A digest collision or a hand-edited file: compute rather than trust.

    surface = _compute_solvent_surface(
        key_name, molecule, key, params,
        fine_grid_spacing_ang=fine_grid_spacing_ang,
        fine_segments_h=fine_segments_h, fine_segments_other=fine_segments_other,
        fine_frame=fine_frame, tol_e_solv=tol_e_solv, max_macro_iter=max_macro_iter,
    )
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(
        json.dumps({"key": key, "surface": surface.to_dict()}, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return surface, "computed"


def _compute_solvent_surface(
    name, molecule, key, params, *, fine_grid_spacing_ang, fine_segments_h,
    fine_segments_other, fine_frame, tol_e_solv, max_macro_iter,
) -> ConductorSurface:
    import math

    from vibeqc import BasisSet, RHFOptions, RKSOptions

    from ..driver import SolventModel, run_cpcm_scf
    from .solvent_geometries import SOLVENT_GEOMETRIES

    kind, functional = _method_kind(key["method"])
    if kind == "rhf":
        options = RHFOptions()
    else:
        from vibeqc.runner import apply_ks_grid_default

        options = RKSOptions()
        options.functional = functional
        apply_ks_grid_default(options)
    model = SolventModel(
        epsilon=math.inf,
        name=f"COSMO-RS conductor ({params.name})",
        variant="cosmo",
        cavity="fine",
        radii=dict(params.cavity_radii),
        radii_scale=1.0,
        solvent_probe_radius_ang=0.0,
        fine_grid_spacing_ang=fine_grid_spacing_ang,
        fine_segments_h=fine_segments_h,
        fine_segments_other=fine_segments_other,
        fine_frame=fine_frame,
        tol_e_solv=tol_e_solv,
        max_macro_iter=max_macro_iter,
    )
    result = run_cpcm_scf(
        molecule, BasisSet(molecule, key["basis"]), method=kind,
        options=options, solvent=model,
    )
    if not result.converged:
        raise RuntimeError(
            f"COSMO-RS: the conductor run of solvent {name!r} at "
            f"{key['method']}/{key['basis']} did not converge in "
            f"{max_macro_iter} macro-iterations; nothing was cached."
        )
    entry = SOLVENT_GEOMETRIES[name]
    return build_conductor_surface(
        result,
        method=key["method"],
        basis=key["basis"],
        cavity_kind=FINE_CAVITY_KIND,
        energy_conductor=float(result.energy),
        notes=f"vibe-qc solvent library {name!r}; geometry: {entry.protocol}",
    )


def solvent_sigma_potential_from_surface(
    surface: ConductorSurface,
    params: Parameterization,
    *,
    temperature_k: float = T_ROOM_K,
    **potential_kwargs,
):
    """A pure solvent's sigma potential from its conductor surface record.

    The record-to-potential half of
    :func:`~vibeqc.solvation.cosmors.solvent.solvent_sigma_potential`, for a
    surface that already exists (from the cache, or a file). The potential
    records the surface's protocol and the solvent's area as that function
    does.
    """
    descriptors = segment_descriptors(surface, params.r_av)
    potential = sigma_potential_segments(
        [descriptors], np.array([1.0]), params,
        temperature_k=temperature_k, **potential_kwargs,
    )
    potential.protocol = surface.provenance.descriptor()
    potential.solvent_area = float(descriptors.total_area)
    return potential
