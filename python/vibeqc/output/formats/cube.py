"""Cube-file emission for ``run_job(..., write_cube=...)``.

Thin wrapper around :mod:`vibeqc.cube` (the existing low-level
density / MO grid writers) that exposes a single high-level call
matching the rest of the output module's writer convention:

    write_cube_density_for_run_job(stem, result, basis, molecule)
        -> {stem}.density.cube

Higher-level convenience for picking out HOMO / LUMO / specific MO
indices comes via :func:`requested_mo_indices` and the matching
:func:`write_cube_mo_for_run_job` helper. These together cover the
common operator workflows:

* ``write_cube=True`` or ``"density"`` -> just the total density.
* ``write_cube="homo"`` / ``"lumo"`` / ``["homo", "lumo"]`` -> those
  MOs as separate cube files.
* ``write_cube=[5, 7, "homo"]`` -> mixed index/label list.

Failures (basis-set shapes vibe-qc's Python side can't grid-evaluate,
ENOMEM on the volumetric tensor) are *non-fatal* -- the calling SCF
result is the load-bearing artefact, cubes are observability. The
caller wraps each emit in its own try/except.

For UHF / UKS, the total density is ``D_alpha + D_beta``; the
high-level wrapper handles that transparently.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ...correlation_conventions import effective_electron_count


__all__ = [
    "CubeRequest",
    "parse_write_cube_kwarg",
    "requested_mo_indices",
    "write_cube_density_for_run_job",
    "write_cube_mo_for_run_job",
]


# ---------------------------------------------------------------------- #
# Request parsing                                                        #
# ---------------------------------------------------------------------- #

class CubeRequest:
    """What cubes should be written for this run.

    Built by :func:`parse_write_cube_kwarg` from the user's
    ``write_cube`` kwarg; consumed by ``run_job`` to dispatch the
    actual writes.
    """

    __slots__ = ("density", "mo_labels")

    def __init__(
        self,
        *,
        density: bool = False,
        mo_labels: tuple[str | int, ...] = (),
    ) -> None:
        self.density: bool = bool(density)
        self.mo_labels: tuple[str | int, ...] = tuple(mo_labels)

    def __bool__(self) -> bool:
        return self.density or bool(self.mo_labels)


def parse_write_cube_kwarg(
    write_cube: Any,
) -> CubeRequest:
    """Translate the ``write_cube=`` kwarg into a :class:`CubeRequest`.

    Accepted forms:

    * ``None`` / ``False`` / ``""`` -- no cubes.
    * ``True`` / ``"density"`` -- density only.
    * ``"homo"`` / ``"lumo"`` -- that MO only.
    * Iterable of strings ``"homo"``/``"lumo"`` and / or ints --
      whichever mix. ``"density"`` in the list also triggers the
      density cube.
    """
    if write_cube is None or write_cube is False or write_cube == "":
        return CubeRequest()
    if write_cube is True or write_cube == "density":
        return CubeRequest(density=True)
    if isinstance(write_cube, str):
        return CubeRequest(mo_labels=(write_cube,))
    if isinstance(write_cube, int):
        return CubeRequest(mo_labels=(int(write_cube),))
    if isinstance(write_cube, Iterable):
        labels: list[str | int] = []
        density = False
        for item in write_cube:
            if item == "density" or item is True:
                density = True
            elif isinstance(item, str):
                labels.append(item)
            elif isinstance(item, int):
                labels.append(int(item))
            else:
                raise TypeError(
                    f"write_cube list entry {item!r} must be a string "
                    f"label ('density'/'homo'/'lumo'/...) or an int "
                    f"MO index"
                )
        return CubeRequest(density=density, mo_labels=tuple(labels))
    raise TypeError(
        f"write_cube must be None / bool / str / int / iterable, "
        f"got {type(write_cube).__name__}"
    )


def requested_mo_indices(
    labels: Sequence[str | int],
    result: Any,
    molecule: Any = None,
) -> list[tuple[int, str]]:
    """Resolve a list of MO labels into 0-based indices + display names.

    String labels handled: ``"homo"``, ``"lumo"``, ``"homo-1"``,
    ``"lumo+1"``, ``"homo-N"`` / ``"lumo+N"`` for integer N. Ints
    are taken as 0-based MO indices. Returns
    ``[(index, display_name), ...]``.

    When ``result`` exposes neither ``mo_occupations`` nor ``mo_occ``
    (the current RHF / RKS result types only carry ``mo_coeffs`` +
    ``mo_energies``), a closed-shell occupation pattern is derived
    from ``molecule.n_electrons()`` and the MO count read off
    ``result.mo_coeffs.shape[1]``. The ``molecule`` arg is required
    for that fallback and may be omitted when ``result`` already
    carries occupations.

    Raises :class:`ValueError` on an unknown label, or when neither
    an occupations attribute nor ``molecule`` is available to derive
    them from, so the caller's surrounding try/except can surface
    the error to the user.
    """
    # HOMO index = largest occupied MO. For restricted closed-shell,
    # occupation 2.0 => occupied; for unrestricted vibe-qc currently
    # exposes only the alpha block through ``result.mo_energies`` --
    # full UHF/UKS cube support is a follow-up.
    # Plain ``a or b`` short-circuit doesn't work for numpy arrays
    # (they raise on truthiness check), so test for None explicitly.
    occ_attr = getattr(result, "mo_occupations", None)
    if occ_attr is None:
        occ_attr = getattr(result, "mo_occ", None)
    if occ_attr is None:
        # Derive closed-shell occupations (2, ..., 2, 0, ..., 0) from the
        # result's effective-electron provenance and its MO count. For an ECP
        # reference this subtracts the physical core electrons absent from the
        # variational orbital space.
        if molecule is None:
            raise ValueError(
                "requested_mo_indices: result has no mo_occupations / "
                "mo_occ and no molecule was supplied to derive "
                "closed-shell occupations from n_electrons"
            )
        coeffs = getattr(result, "mo_coeffs", None)
        if coeffs is None:
            coeffs = getattr(result, "mo_coefficients", None)
        if coeffs is None:
            raise ValueError(
                "requested_mo_indices: result has no mo_coeffs and no "
                "occupations, cannot resolve HOMO/LUMO labels"
            )
        n_mo_derived = int(np.asarray(coeffs).shape[1])
        n_occ = effective_electron_count(molecule, result) // 2
        occ_arr = np.zeros(n_mo_derived, dtype=float)
        occ_arr[:n_occ] = 2.0
        occ_attr = occ_arr
    occ = np.asarray(occ_attr, dtype=float)
    if occ.size == 0:
        raise ValueError(
            "requested_mo_indices: empty mo_occupations -- "
            "HOMO/LUMO labels can't be resolved"
        )
    homo_idx = int(np.argmax(np.where(occ > 1e-8,
                                       np.arange(occ.size), -1)))
    n_mo = int(occ.size)
    # Sentinel: a non-existent LUMO encodes as n_mo so range-checks
    # below trigger reliably (the previous `lumo_idx = homo_idx`
    # fallback silently returned HOMO when no virtual MO existed --
    # confusing for a user who asked for the LUMO explicitly).
    lumo_idx = homo_idx + 1 if homo_idx + 1 < n_mo else n_mo

    resolved: list[tuple[int, str]] = []
    for lbl in labels:
        if isinstance(lbl, int):
            if not (0 <= lbl < n_mo):
                raise ValueError(
                    f"MO index {lbl} out of range [0, {n_mo})"
                )
            resolved.append((int(lbl), f"mo_{lbl}"))
            continue
        s = str(lbl).strip().lower()
        if s == "homo":
            resolved.append((homo_idx, "homo"))
        elif s == "lumo":
            if lumo_idx >= n_mo:
                raise ValueError(
                    "MO label 'lumo' requested but no virtual MO "
                    "exists in this calculation"
                )
            resolved.append((lumo_idx, "lumo"))
        elif s.startswith("homo-"):
            n = int(s.split("-", 1)[1])
            idx = homo_idx - n
            if idx < 0:
                raise ValueError(
                    f"MO label {lbl!r} would resolve to negative index"
                )
            resolved.append((idx, s))
        elif s.startswith("lumo+"):
            n = int(s.split("+", 1)[1])
            idx = lumo_idx + n
            if idx >= n_mo:
                raise ValueError(
                    f"MO label {lbl!r} would resolve to index {idx} "
                    f">= n_mo={n_mo}"
                )
            resolved.append((idx, s))
        else:
            raise ValueError(f"unknown MO label {lbl!r}")
    return resolved


# ---------------------------------------------------------------------- #
# High-level writers                                                     #
# ---------------------------------------------------------------------- #

def write_cube_density_for_run_job(
    stem: os.PathLike | str,
    result: Any,
    basis: Any,
    molecule: Any,
    *,
    spacing: float = 0.2,
    padding: float = 4.0,
) -> Path:
    """Write ``{stem}.density.cube`` -- total electron density.

    Handles RHF/RKS (``result.density`` is the total D) and UHF/UKS
    (``D_alpha + D_beta``) transparently. Returns the written path.
    """
    from ...cube import write_cube_density
    D_total = _total_density(result)
    s = Path(os.fspath(stem))
    target = s.parent / (s.name + ".density.cube")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_cube_density(
        target, D_total, basis, molecule,
        spacing=spacing, padding=padding,
        title=f"vibe-qc density / {s.name}",
    )
    return target


def write_cube_mo_for_run_job(
    stem: os.PathLike | str,
    result: Any,
    basis: Any,
    molecule: Any,
    mo_index: int,
    display_name: str,
    *,
    spacing: float = 0.2,
    padding: float = 4.0,
) -> Path:
    """Write ``{stem}.{display_name}.cube`` for a single MO.

    ``display_name`` is appended verbatim to the stem (no
    sanitisation), so callers should pre-validate it (the
    ``requested_mo_indices`` helper already returns lowercase ASCII
    names).
    """
    from ...cube import write_cube_mo
    C = np.asarray(_mo_coeffs(result), dtype=float)
    s = Path(os.fspath(stem))
    target = s.parent / (s.name + f".{display_name}.cube")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_cube_mo(
        target, C, mo_index, basis, molecule,
        spacing=spacing, padding=padding,
        title=f"vibe-qc MO {display_name} / {s.name}",
    )
    return target


# ---------------------------------------------------------------------- #
# Result-shape helpers                                                   #
# ---------------------------------------------------------------------- #

def _total_density(result: Any) -> np.ndarray:
    """RHF/RKS => ``result.density``. UHF/UKS =>
    ``density_alpha + density_beta``. Falls back to ``result.density``
    when neither alpha/beta is present (works for the unified
    result type)."""
    d_a = getattr(result, "density_alpha", None)
    d_b = getattr(result, "density_beta", None)
    if d_a is not None and d_b is not None:
        return np.asarray(d_a, dtype=float) + np.asarray(d_b, dtype=float)
    return np.asarray(result.density, dtype=float)


def _mo_coeffs(result: Any) -> np.ndarray:
    """RHF/RKS => ``result.mo_coeffs``. UHF/UKS => ``mo_coeffs_alpha``
    (a follow-up will surface the beta cubes too -- for now alpha-only
    matches the molden writer's restricted-display convention for
    UHF in a few visualizers)."""
    if hasattr(result, "mo_coeffs_alpha"):
        return result.mo_coeffs_alpha
    return result.mo_coeffs
