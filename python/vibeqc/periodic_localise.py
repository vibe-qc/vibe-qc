"""Periodic orbital localisation at the Γ-point -- Wannier increments 2a + 2b.

Localise the occupied Bloch orbitals of a converged Γ-point periodic SCF into
spatially compact, Wannier-like orbitals
([`handovers/HANDOVER_PERIODIC_LOCAL_CORRELATION.md`], Stage 2).

Two localisation criteria are available, both reusing the molecular localiser
(:mod:`vibeqc.localise`):

* **Pipek-Mezey** (``method="pipek-mezey"``, increment 2a) -- objective built from
  Mulliken atomic populations (overlap ``S`` + atom->AO map only). No position
  operator, so it is PBC-safe in every regime, including dense crystals whose
  orbitals wrap the cell boundary.
* **Foster-Boys** (``method="boys"``, increment 2b) -- maximises ``S_i <i|r|i>^2``
  using the home-cell dipole integrals. This *does* use the ordinary position
  operator, so it (and the centroids/spreads below) are faithful only when the
  occupied orbitals do **not** wrap the periodic boundary -- i.e. the
  molecule-in-a-box and dilute / large-cell regimes. Dense crystals need the
  Resta periodic position operator ``<mu|e^{-iG.r}|ν>`` (still a follow-up; no
  such integral exists yet), at which point ``method="boys"`` should be gated.

Wannier quantities reported
---------------------------
* ``centers`` -- Mulliken-charge-weighted centre ``S_A Q_i^A R_A`` (PBC-robust,
  approximate; increment 2a).
* ``centroids`` -- true position centroid ``<i|r>`` from the dipole integrals
  (exact when the orbital does not wrap; increment 2b).
* ``spreads`` -- ``<i|r^2> - <i|r>^2`` (bohr^2, the Marzari-Vanderbilt per-orbital
  spread) from the home-cell dipole + quadrupole integrals (same validity as
  ``centroids``; increment 2b).

Remaining for a later 2b: the Resta operator (boundary-wrapping crystals) and
the multi-k ``U(k)`` from overlap matrices ``M(k,b)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .localise import (
    boys_objective,
    foster_boys_localise,
    pipek_mezey_localise,
    pipek_mezey_objective,
)

__all__ = ["PeriodicWannierResult", "localise_periodic_gamma"]

_SUPPORTED_METHODS = ("pipek-mezey", "boys")


@dataclass
class PeriodicWannierResult:
    """Localised occupied Γ-point orbitals and their Wannier descriptors.

    Attributes
    ----------
    method : str
        Localisation criterion used (``"pipek-mezey"`` or ``"boys"``).
    C_loc : ndarray, shape (nbf, n_occ)
        Localised occupied MO coefficients at Γ, ``C_loc = C_occ @ U``.
    U : ndarray, shape (n_occ, n_occ)
        Unitary mixing matrix, ``U = C_occ.T @ S @ C_loc``.
    centers : ndarray, shape (n_occ, 3)
        Mulliken-charge-weighted Wannier centres (bohr) -- PBC-robust, approximate.
    centroids : ndarray, shape (n_occ, 3)
        True position centroids ``<i|r>`` (bohr). Faithful only for orbitals that
        do not wrap the cell boundary (see module docstring).
    spreads : ndarray, shape (n_occ,)
        Per-orbital spread ``<i|r^2> - <i|r>^2`` (bohr^2). Same validity as
        ``centroids``.
    charges : ndarray, shape (n_occ, natom)
        Mulliken population ``Q_i^A`` of localised orbital ``i`` on atom ``A``.
        Each row sums to 1.
    localization : ndarray, shape (n_occ,)
        Per-orbital Mulliken localisation metric ``S_A (Q_i^A)^2``.
    objective_initial, objective_final : float
        Localisation objective before / after (PM: ``S_i S_A (Q_i^A)^2``;
        Boys: ``S_i <i|r|i>^2``). ``final >= initial`` for a genuine localisation.
    n_occ : int
        Number of localised (doubly-occupied) orbitals.
    """

    method: str
    C_loc: np.ndarray
    U: np.ndarray
    centers: np.ndarray
    centroids: np.ndarray
    spreads: np.ndarray
    charges: np.ndarray
    localization: np.ndarray
    objective_initial: float
    objective_final: float
    n_occ: int


def _atom_maps(basis) -> tuple[np.ndarray, np.ndarray]:
    """Build the boolean atom->AO map and per-atom positions from a basis.

    Returns
    -------
    atom_basis_map : ndarray, shape (nbf, natom)
        ``atom_basis_map[mu, A] = 1`` iff AO ``mu`` is centred on atom ``A``.
        Spherical-harmonic shells (``2ℓ+1`` functions each), matching the
        convention used throughout vibe-qc (cf. ``dlpno/pao.py`` and
        ``tests/test_localise.py``).
    atom_pos : ndarray, shape (natom, 3)
        Cartesian position (bohr) of each atom, taken from its shell origins.
    """
    shells = list(basis.shells())
    if not shells:
        raise ValueError("basis has no shells")
    natom = max(int(sh.atom_index) for sh in shells) + 1
    nbf = int(basis.nbasis)

    amap = np.zeros((nbf, natom))
    atom_pos = np.zeros((natom, 3))
    seen = np.zeros(natom, dtype=bool)
    bf = 0
    for sh in shells:
        a = int(sh.atom_index)
        n_func = 2 * int(sh.l) + 1
        amap[bf : bf + n_func, a] = 1.0
        if not seen[a]:
            atom_pos[a] = np.asarray(sh.origin, dtype=float)
            seen[a] = True
        bf += n_func
    return amap, atom_pos


def _home_cell_moments(basis, system) -> tuple[np.ndarray, np.ndarray]:
    """Home-cell (g=0) dipole and diagonal-quadrupole AO integrals.

    Returns ``(dipoles, r2_diag)`` where ``dipoles[mu,ν,c] = <mu|r_c|ν>`` and
    ``r2_diag[mu,ν,c] = <mu|r_c^2|ν>`` for ``c = x, y, z``, expansion origin
    ``(0,0,0)``.

    Computed via libint's emultipole2 (``compute_multipole_moments_lattice``)
    with a 1-bohr lattice cutoff so only g=0 is in range -- ``blocks[0]`` is then
    the home cell (same construction as :mod:`vibeqc.multipole_jk_builder`).
    Component order (libint emultipole2): 0 overlap; 1-3 dipole x,y,z;
    4-9 quadrupole xx,xy,xz,yy,yz,zz.
    """
    from ._vibeqc_core import (
        LatticeSumOptions,
        compute_multipole_moments_lattice,
    )

    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 1.0  # home cell only (any lattice vector > 1 bohr)
    mp = compute_multipole_moments_lattice(basis, system, lat_opts, L_max=2)
    home = mp.blocks[0]

    nbf = int(basis.nbasis)
    dipoles = np.zeros((nbf, nbf, 3))
    r2_diag = np.zeros((nbf, nbf, 3))
    for c in range(3):
        dipoles[:, :, c] = np.asarray(home[1 + c], dtype=float)
    # diagonal quadrupole: xx=4, yy=7, zz=9
    for c, comp in enumerate((4, 7, 9)):
        r2_diag[:, :, c] = np.asarray(home[comp], dtype=float)
    return dipoles, r2_diag


def localise_periodic_gamma(
    result,
    basis,
    system,
    *,
    method: str = "pipek-mezey",
    n_occ: int | None = None,
    max_iter: int = 200,
    conv_tol: float = 1e-8,
) -> PeriodicWannierResult:
    """Localise the occupied Γ-point orbitals of a periodic SCF.

    Parameters
    ----------
    result
        A converged Γ-point periodic RHF result. Duck-typed: must expose
        ``mo_coeffs`` (nbf x norb, real at Γ) and ``overlap`` (nbf x nbf, the
        AO overlap S at Γ). ``PeriodicRHFEwaldResult`` and ``PeriodicRHFGDFResult``
        both qualify.
    basis
        The ``BasisSet`` used for the SCF (home unit cell).
    system
        The ``PeriodicSystem``. Provides the electron count (to derive ``n_occ``
        for closed-shell RHF when not given) and is passed to the home-cell
        moment-integral build.
    method
        ``"pipek-mezey"`` (2a, PBC-safe in every regime) or ``"boys"`` (2b,
        faithful only for non-wrapping orbitals -- see module docstring).
    n_occ
        Number of doubly-occupied orbitals to localise. Defaults to
        ``system.unit_cell_molecule().n_electrons() // 2``.
    max_iter, conv_tol
        Forwarded to the Jacobi localiser.

    Returns
    -------
    PeriodicWannierResult
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"localise_periodic_gamma: method={method!r} is not supported. "
            f"Choose one of {_SUPPORTED_METHODS}. The Resta periodic position "
            "operator (for crystals whose occupied orbitals wrap the cell "
            "boundary) and the multi-k U(k) localisation are later 2b work -- "
            "see handovers/HANDOVER_PERIODIC_LOCAL_CORRELATION.md Stage 2b."
        )

    if n_occ is None:
        n_occ = int(system.unit_cell_molecule().n_electrons()) // 2
    if n_occ < 1:
        raise ValueError(f"n_occ must be >= 1, got {n_occ}")

    C_all = np.asarray(result.mo_coeffs)
    if np.iscomplexobj(C_all):
        # Γ-point coefficients are real up to phase; guard anyway.
        C_all = np.real_if_close(C_all, tol=1000).real
    C_occ = np.array(C_all[:, :n_occ], dtype=float)
    S = np.asarray(result.overlap, dtype=float)

    amap, atom_pos = _atom_maps(basis)
    dipoles, r2_diag = _home_cell_moments(basis, system)

    if method == "pipek-mezey":
        obj0 = pipek_mezey_objective(C_occ, S, amap)
        C_loc = pipek_mezey_localise(
            C_occ, S, amap, max_iter=max_iter, conv_tol=conv_tol
        )
        obj1 = pipek_mezey_objective(C_loc, S, amap)
    else:  # "boys"
        obj0 = boys_objective(C_occ, dipoles)
        C_loc = foster_boys_localise(
            C_occ, dipoles, max_iter=max_iter, conv_tol=conv_tol
        )
        obj1 = boys_objective(C_loc, dipoles)

    # Unitary mixing: C_occ is S-orthonormal (C_occ^T S C_occ = I), so the
    # transform that produced C_loc is U = C_occ^T S C_loc.
    U = C_occ.T @ S @ C_loc

    # Mulliken population of each localised orbital on each atom.
    SC = S @ C_loc  # (nbf, n_occ)
    pop_ao = C_loc * SC  # (nbf, n_occ): pop_ao[mu,i] = C_loc[mu,i] (S C_loc)[mu,i]
    charges = (amap.T @ pop_ao).T  # (n_occ, natom); rows sum to 1
    localization = np.sum(charges**2, axis=1)  # (n_occ,)
    centers = charges @ atom_pos  # (n_occ, 3), bohr -- Mulliken charge-weighted

    # True position-operator descriptors (faithful for non-wrapping orbitals).
    centroids = np.einsum("mi,mnc,ni->ic", C_loc, dipoles, C_loc)  # (n_occ, 3)
    r2 = np.einsum("mi,mnc,ni->ic", C_loc, r2_diag, C_loc).sum(axis=1)  # (n_occ,)
    spreads = r2 - np.sum(centroids**2, axis=1)  # (n_occ,) = <r^2> - <r>^2

    return PeriodicWannierResult(
        method=method,
        C_loc=C_loc,
        U=U,
        centers=centers,
        centroids=centroids,
        spreads=spreads,
        charges=charges,
        localization=localization,
        objective_initial=float(obj0),
        objective_final=float(obj1),
        n_occ=n_occ,
    )
