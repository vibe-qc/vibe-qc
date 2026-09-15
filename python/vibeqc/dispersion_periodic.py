"""Periodic D3-BJ dispersion correction.

Companion to :mod:`vibeqc.dispersion` (which handles molecules). The
public entry point :func:`compute_d3bj_periodic` returns the dispersion
energy per unit cell and, optionally, the per-atom gradient.

Two backends, mirroring the molecular one:

* **``"dftd3"``** -- Grimme's reference Fortran library, called through
  ``dftd3.interface.DispersionModel`` with the canonical ``lattice=``
  and ``periodic=`` arguments. This is the bit-exact periodic D3-BJ
  implementation. Optional dependency: install with
  ``pip install -e '.[dispersion]'``.

* **``"builtin"``** -- uses vibe-qc's native molecular
  :func:`vibeqc.compute_d3bj` on a supercell expansion of the unit
  cell, then divides by the number of replicated cells. Approximate
  (CN values near the supercell boundary are wrong; the per-cell
  energy is slightly underestimated in magnitude until the supercell
  is large enough). Suitable for qualitative use; pin
  ``backend="dftd3"`` for production. Refines with ``supercell=``.

* **``"auto"``** -- prefers ``"dftd3"`` if available, falls back to
  ``"builtin"``.

Surface-catalysis note: for a 2D-vacuum slab (``dim=3`` with vacuum
along one lattice vector), the dispersion sum naturally truncates in
the vacuum direction because no image atoms sit within cutoff. Pass
the slab's ``PeriodicSystem`` unchanged; the dftd3 backend's
``periodic=[True, True, True]`` flag handles the vacuum correctly via
the cutoff.

References
----------
* Grimme, Antony, Ehrlich, Krieg 2010 -- D3 (DOI:10.1063/1.3382344)
* Grimme, Ehrlich, Goerigk 2011 -- Becke-Johnson damping
  (DOI:10.1002/jcc.21759)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ._vibeqc_core import (
    Atom,
    D3BJParams,
    Molecule,
    PeriodicSystem,
)
from .dispersion import (
    _resolve_backend,
    compute_d3bj as _compute_d3bj_molecular,
    d3bj_params_for,
    dftd3_available,
)


__all__ = [
    "compute_d3bj_periodic",
    "PeriodicDispersionResult",
]


@dataclass
class PeriodicDispersionResult:
    """Per-unit-cell dispersion energy plus optional gradient.

    Attributes
    ----------
    energy : float
        E_disp per unit cell in Hartree.
    gradient : np.ndarray
        ``(n_atoms, 3)`` array of dE_disp / dr_A in Hartree / bohr,
        or shape ``(0, 3)`` when ``with_gradient=False``.
    stress : np.ndarray | None
        ``(3, 3)`` cell stress tensor in Hartree / bohr^3, or ``None``
        when not provided by the backend / not requested.
    backend : str
        Which backend produced the result (``"dftd3"`` / ``"builtin"``).
    supercell : tuple[int, int, int]
        Supercell expansion used by the builtin backend. ``(1, 1, 1)``
        for the dftd3 backend (it handles the lattice sum internally).
    """

    energy: float
    gradient: np.ndarray
    stress: Optional[np.ndarray]
    backend: str
    supercell: tuple


def _periodic_flags_from_dim(system: PeriodicSystem) -> tuple[bool, bool, bool]:
    """Map a vibe-qc ``dim`` into a length-3 boolean periodicity mask.

    The slab/wire conventions: ``dim=3`` -> fully periodic;
    ``dim=2`` -> periodic in the first two lattice vectors;
    ``dim=1`` -> periodic in the first only.
    """
    dim = int(system.dim)
    if dim < 1 or dim > 3:
        raise ValueError(
            f"compute_d3bj_periodic: PeriodicSystem.dim must be 1, 2, "
            f"or 3, got {dim}"
        )
    return (dim >= 1, dim >= 2, dim >= 3)


def _compute_d3bj_periodic_dftd3(
    system: PeriodicSystem,
    params: D3BJParams,
    *,
    with_gradient: bool,
    with_stress: bool,
) -> PeriodicDispersionResult:
    """Bit-exact periodic D3-BJ via Grimme's ``dftd3`` package.

    Inputs are converted to bohr (vibe-qc's native unit, also dftd3's).
    """
    try:
        from dftd3.interface import DispersionModel, RationalDampingParam
    except ImportError as e:
        raise ImportError(
            "compute_d3bj_periodic(backend='dftd3') requires the "
            "optional dftd3 package. Install with `pip install dftd3` "
            "into your vibe-qc venv, or `pip install -e '.[dispersion]'` "
            "from the repo checkout."
        ) from e

    atoms = list(system.unit_cell)
    if not atoms:
        return PeriodicDispersionResult(
            energy=0.0,
            gradient=np.zeros((0, 3)),
            stress=None,
            backend="dftd3",
            supercell=(1, 1, 1),
        )

    numbers = np.array([a.Z for a in atoms], dtype=np.int32)
    positions = np.array([list(a.xyz) for a in atoms], dtype=float)
    # PeriodicSystem.lattice columns are a, b, c. dftd3 wants rows.
    lattice = np.asarray(system.lattice, dtype=float).T.copy()
    periodic = np.array(_periodic_flags_from_dim(system), dtype=bool)

    model = DispersionModel(numbers, positions, lattice=lattice, periodic=periodic)
    # s9 explicit: dftd3's RationalDampingParam defaults s9=1.0, which would
    # silently turn the ATM three-body term on here and nowhere else.
    bj = RationalDampingParam(
        s6=params.s6, s8=params.s8, a1=params.a1, a2=params.a2,
        s9=params.s9,
    )
    res = model.get_dispersion(bj, grad=(with_gradient or with_stress))

    grad = (
        np.asarray(res["gradient"], dtype=float)
        if with_gradient
        else np.zeros((0, 3))
    )
    # dftd3 returns the virial (3, 3) tensor when grad=True for periodic
    # systems. The stress (Ha/bohr^3) is virial / cell_volume; we leave
    # the divide for callers that need it because the convention varies.
    stress = None
    if with_stress and "virial" in res:
        vol = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
        stress = (
            np.asarray(res["virial"], dtype=float) / vol
            if vol > 0
            else np.asarray(res["virial"], dtype=float)
        )

    return PeriodicDispersionResult(
        energy=float(res["energy"]),
        gradient=grad,
        stress=stress,
        backend="dftd3",
        supercell=(1, 1, 1),
    )


def _auto_supercell(
    system: PeriodicSystem,
    cutoff_bohr: float,
) -> tuple[int, int, int]:
    """Pick the smallest (n1, n2, n3) supercell whose half-extent in
    each periodic direction reaches the cutoff.

    Non-periodic directions (dim < 3) get n_i = 1. A direction with a
    very long lattice vector (e.g. a slab's vacuum axis) is unlikely
    to need replication and gets n_i = 1 automatically because the
    cutoff fits inside one cell.
    """
    lat = np.asarray(system.lattice, dtype=float)
    flags = _periodic_flags_from_dim(system)
    ns: list[int] = []
    for i, on in enumerate(flags):
        if not on:
            ns.append(1)
            continue
        L_i = float(np.linalg.norm(lat[:, i]))
        # Need n_i x L_i / 2 >= cutoff -> n_i >= 2 x cutoff / L_i (round up,
        # and make it odd so we get a centred supercell).
        n_required = int(np.ceil(2.0 * cutoff_bohr / max(L_i, 1e-12)))
        n = max(1, n_required)
        if n % 2 == 0:
            n += 1
        ns.append(n)
    return (ns[0], ns[1], ns[2])


def _compute_d3bj_periodic_builtin(
    system: PeriodicSystem,
    params: D3BJParams,
    *,
    cutoff_bohr: float,
    supercell: Optional[tuple],
    with_gradient: bool,
) -> PeriodicDispersionResult:
    """Supercell approximation: replicate the unit cell into an
    ``(n1, n2, n3)`` supercell, call molecular ``compute_d3bj`` once,
    divide the resulting energy (and gradient block on the central
    cell) by ``n1 * n2 * n3``.

    Caveat: atoms near the supercell boundary see fewer neighbours
    than they would in a true infinite crystal, so their CN values
    (and hence c6 coefficients) are slightly off. The per-cell energy
    converges with increasing supercell size. For production use,
    prefer the ``"dftd3"`` backend.
    """
    if supercell is None:
        supercell = _auto_supercell(system, cutoff_bohr)
    n1, n2, n3 = (int(x) for x in supercell)
    if min(n1, n2, n3) < 1:
        raise ValueError(
            f"compute_d3bj_periodic(builtin): supercell entries must "
            f"be >= 1, got {(n1, n2, n3)}"
        )

    lat = np.asarray(system.lattice, dtype=float)
    central = list(system.unit_cell)
    n_central = len(central)
    if n_central == 0:
        return PeriodicDispersionResult(
            energy=0.0,
            gradient=np.zeros((0, 3)),
            stress=None,
            backend="builtin",
            supercell=(n1, n2, n3),
        )

    # Build the supercell atom list. Index 0..(n_central - 1) is the
    # (0, 0, 0) image -- the central cell -- which lets us slice the
    # gradient cleanly afterwards.
    flags = _periodic_flags_from_dim(system)
    supercell_atoms: list[Atom] = []
    # First copy the central cell at the origin.
    for a in central:
        supercell_atoms.append(Atom(int(a.Z), list(a.xyz)))
    # Then add image cells.
    ranges = [
        range(-(n // 2), n // 2 + 1) if on else range(1)
        for n, on in zip((n1, n2, n3), flags)
    ]
    for ix in ranges[0]:
        for iy in ranges[1]:
            for iz in ranges[2]:
                if ix == 0 and iy == 0 and iz == 0:
                    continue
                shift = ix * lat[:, 0] + iy * lat[:, 1] + iz * lat[:, 2]
                for a in central:
                    pos = np.asarray(a.xyz, dtype=float) + shift
                    supercell_atoms.append(Atom(int(a.Z), list(pos)))

    super_mol = Molecule(supercell_atoms, 0, 1)
    # Use the explicit "builtin" backend to stay self-contained.
    disp = _compute_d3bj_molecular(
        super_mol, params, with_gradient=with_gradient, backend="builtin"
    )

    n_cells = n1 * n2 * n3  # 1 for non-periodic axes (range(1) above)
    energy_per_cell = float(disp.energy) / n_cells

    grad = np.zeros((0, 3))
    if with_gradient:
        super_grad = np.asarray(disp.gradient, dtype=float)
        # By translational symmetry of the lattice sum, the central-cell
        # gradient is the average over all replicas -- equivalently, just
        # take the first n_central rows. We multiply by 1 (no division)
        # because gradient is per-atom and atoms are not double-counted.
        grad = super_grad[:n_central, :].copy()

    return PeriodicDispersionResult(
        energy=energy_per_cell,
        gradient=grad,
        stress=None,
        backend="builtin",
        supercell=(n1, n2, n3),
    )


def compute_d3bj_periodic(
    system: PeriodicSystem,
    params: Union[D3BJParams, str],
    *,
    cutoff_bohr: float = 50.0,
    with_gradient: bool = False,
    with_stress: bool = False,
    backend: str = "auto",
    supercell: Optional[tuple] = None,
    functional: Optional[str] = None,
) -> PeriodicDispersionResult:
    """Periodic D3(BJ) dispersion energy per unit cell.

    Parameters
    ----------
    system : PeriodicSystem
        Crystal (``dim=3``) or slab / wire (``dim=2`` / ``dim=1``).
    params : D3BJParams | str
        Either explicit damping parameters or a functional name
        (resolved via :func:`vibeqc.d3bj_params_for`). Same calling
        convention as the molecular :func:`vibeqc.compute_d3bj`.
    cutoff_bohr : float, default 50.0
        Real-space cutoff for the lattice sum. Used by the dftd3
        backend's internal sum (Grimme's library default already
        respects this) and to size the supercell for the builtin
        fallback. 50 bohr ≈ 26 Å is usually overkill for r⁻⁶ damping
        but matches Grimme's recommended default.
    with_gradient : bool
        If True, populate ``.gradient`` with dE_disp / dr_A on the
        central-cell atoms (Hartree / bohr).
    with_stress : bool
        If True (and backend is dftd3), populate ``.stress`` with the
        cell stress tensor in Hartree / bohr^3. Not available from the
        builtin backend yet.
    backend : str
        ``"dftd3"``, ``"builtin"``, or ``"auto"``. Auto prefers dftd3
        when installed.
    supercell : (int, int, int), optional
        Override the builtin backend's supercell size. ``None`` ->
        auto-chosen so each periodic lattice direction is replicated
        until its half-extent reaches ``cutoff_bohr``.
    functional : str, optional
        Only used if ``params`` is the literal string ``"d3bj"`` or
        ``True``: the functional whose damping parameters to look up.

    Returns
    -------
    PeriodicDispersionResult
    """
    # Resolve params (mirror runner._resolve_dispersion semantics).
    if isinstance(params, str):
        key = params.strip().lower()
        if key in ("", "none", "false"):
            raise ValueError(
                "compute_d3bj_periodic: params is empty / 'none' -- "
                "call with explicit D3BJParams or a functional name."
            )
        if key in ("d3bj", "true", "yes", "on"):
            if not functional:
                raise ValueError(
                    "compute_d3bj_periodic: params='d3bj' requires "
                    "functional=... to look up damping parameters."
                )
            lookup = functional
        else:
            lookup = params
        p = d3bj_params_for(lookup)
        if p is None:
            raise ValueError(
                f"compute_d3bj_periodic: no D3-BJ parameters for "
                f"functional {lookup!r}"
            )
        params = p

    # Periodic auto retains the reference lattice-sum backend when present.
    # The native periodic route is a supercell approximation and is separate
    # from the now-quantitative molecular H-Ar kernel.
    chosen = (
        "dftd3" if dftd3_available() else "builtin"
    ) if backend == "auto" else _resolve_backend(backend)
    if chosen == "dftd3":
        return _compute_d3bj_periodic_dftd3(
            system, params,
            with_gradient=with_gradient,
            with_stress=with_stress,
        )
    return _compute_d3bj_periodic_builtin(
        system, params,
        cutoff_bohr=cutoff_bohr,
        supercell=supercell,
        with_gradient=with_gradient,
    )
