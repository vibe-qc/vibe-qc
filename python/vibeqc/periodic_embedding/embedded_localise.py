"""Localised orbitals from embedded density matrices.

Constructs spatially-localized Wannier-like orbitals from the embedded
region-I density matrix (which already includes k-averaging via contour
integration).  Uses the existing Pipek-Mezey and Foster-Boys localisers
from :mod:`vibeqc.localise`, operating on the density's eigenvectors
(natural orbitals of the embedded system).

This gives a compact, localized basis for the ΔV support region in Layer B
without requiring a full multi-k SCF -- the contour integration already
folds in all surface k-points through S_emb(k,z).

Method validity under PBC
-------------------------
* ``"pipek-mezey"`` builds its objective from Mulliken atomic populations
  (region-I overlap + atom->AO map only).  No position operator, so it is
  PBC-safe in every regime.
* ``"boys"`` maximises ``S_i <i|r|i>^2`` using the home-cell (g=0) dipole
  integrals restricted to the region-I AOs.  That is the ordinary position
  operator, so the localisation and the reported ``centroids`` are faithful
  only when the region-I orbitals do **not** wrap the periodic boundary --
  i.e. the molecule-in-a-box, adsorbate, and dilute / large-cell regimes
  this embedding targets.  Boundary-wrapping orbitals need the Resta
  periodic position operator, which vibe-qc does not yet have; the same
  caveat is documented for the Γ-point localiser in
  :mod:`vibeqc.periodic_localise`.

``spreads`` would need the region-I quadrupole integrals ``<mu|r_c^2|ν>``.
The only route to those is ``compute_multipole_moments_lattice``, which
needs the ``PeriodicSystem`` this function is not given, so ``spreads`` is
reported as zero rather than guessed at.

References
----------
* Pipek, Mezey, J. Chem. Phys. 90, 4916 (1989)
* Foster, Boys, Rev. Mod. Phys. 32, 300 (1960)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .._vibeqc_core import BasisSet
from ..localise import foster_boys_localise, pipek_mezey_localise


@dataclass
class EmbeddedLocalisedResult:
    """Localised orbitals from an embedded density matrix.

    Attributes
    ----------
    C_loc
        Localised orbital coefficients, shape ``(n_i, n_occ)``,
        in the region-I AO basis.
    U
        Unitary mixing matrix from natural orbitals -> localised,
        shape ``(n_occ, n_occ)``.  ``C_loc = C_nat @ U``.
    centers
        Mulliken-charge-weighted centers (bohr, PBC-robust), shape
        ``(n_occ, 3)``.
    centroids
        Position centroids ``<i|r|i>`` (bohr), shape ``(n_occ, 3)``.
        Foster-Boys only -- zero for Pipek-Mezey, which never forms the
        position operator.  Faithful only for non-wrapping orbitals
        (see the module docstring).
    spreads
        Per-orbital spreads ``<r^2> - <r>^2`` (bohr^2).  Always zero: the
        region-I quadrupole integrals needed for ``<r^2>`` are not
        available here (see the module docstring).
    method
        Localisation method used.
    """

    C_loc: NDArray[np.float64]
    U: NDArray[np.float64]
    centers: NDArray[np.float64]
    centroids: NDArray[np.float64]
    spreads: NDArray[np.float64]
    method: str


def _ao_atom_map(basis: BasisSet, i_ao: NDArray[np.int64]):
    """Build atom->AO boolean map for region-I AOs only."""
    shells = basis.shells()
    n_i = len(i_ao)
    i_set = set(int(x) for x in i_ao)

    # Map region-I AO positions to atoms.
    ao_atom = {}
    bf = 0
    for sh in shells:
        n_func = 2 * int(sh.l) + 1
        for f in range(n_func):
            if bf in i_set:
                ao_atom[bf] = int(sh.atom_index)
            bf += 1

    # Find unique atoms in region I.
    atoms = sorted(set(ao_atom.values()))
    atom_idx = {a: i for i, a in enumerate(atoms)}
    n_atom = len(atoms)

    amap = np.zeros((n_i, n_atom))
    for pos, ao_idx in enumerate(i_ao):
        a = ao_atom[int(ao_idx)]
        amap[pos, atom_idx[a]] = 1.0

    atom_pos = np.zeros((n_atom, 3))
    seen = set()
    bf = 0
    for sh in shells:
        n_func = 2 * int(sh.l) + 1
        if bf in i_set and int(sh.atom_index) not in seen:
            a_idx = atom_idx[int(sh.atom_index)]
            atom_pos[a_idx] = np.asarray(sh.origin, dtype=float)
            seen.add(int(sh.atom_index))
        bf += n_func

    return amap, atom_pos


def _region_dipoles(basis: BasisSet, i_ao: NDArray[np.int64]) -> NDArray[np.float64]:
    """Home-cell Cartesian dipole integrals restricted to the region-I AOs.

    Returns ``dipoles[p, q, c] = <mu_p | r_c | ν_q>`` (bohr, expansion origin
    ``(0, 0, 0)``) for ``c = x, y, z``, where ``p``, ``q`` index the region-I
    AOs in the order given by ``i_ao``.  This is the layout
    :func:`vibeqc.localise.foster_boys_localise` expects, shape
    ``(n_i, n_i, 3)``.

    Only the g=0 (home-cell) block is formed -- the same choice as
    ``vibeqc.periodic_localise._home_cell_moments``, which reaches it by
    calling ``compute_multipole_moments_lattice`` with a 1-bohr lattice
    cutoff.  ``compute_dipole`` is that block directly, and needs no
    ``PeriodicSystem``.
    """
    from .._vibeqc_core import compute_dipole

    dip = compute_dipole(basis)
    nbf = int(basis.nbasis)
    full = np.zeros((nbf, nbf, 3))
    full[:, :, 0] = np.asarray(dip.x)
    full[:, :, 1] = np.asarray(dip.y)
    full[:, :, 2] = np.asarray(dip.z)

    block = np.ix_(i_ao, i_ao)
    return np.stack([full[:, :, c][block] for c in range(3)], axis=-1)


def embedded_localise(
    d_local: NDArray[np.float64],
    s_ii: NDArray[np.float64],
    basis: BasisSet,
    i_ao: NDArray[np.int64],
    *,
    method: str = "pipek-mezey",
    n_occ: Optional[int] = None,
) -> EmbeddedLocalisedResult:
    """Localise natural orbitals of the embedded density matrix.

    Parameters
    ----------
    d_local
        Embedded region-I density matrix, shape ``(n_i, n_i)``.
    s_ii
        Overlap matrix for region I AOs, shape ``(n_i, n_i)``.
    basis
        Full orbital basis set (for atom->AO map).
    i_ao
        Region-I AO indices into the full basis.
    method
        ``"pipek-mezey"`` (PBC-safe in every regime) or ``"boys"``
        (home-cell position operator; faithful only for orbitals that do
        not wrap the cell boundary -- see the module docstring).
    n_occ
        Number of occupied orbitals to localise.  If None, uses all
        orbitals with natural occupation > 1e-6.

    Returns
    -------
    EmbeddedLocalisedResult
        Localised orbital coefficients, mixing matrix, centers, spreads.
    """
    n_i = d_local.shape[0]

    # Diagonalize D.S to get natural orbitals.
    # D.S.C = C.n  ->  S^{1/2}.D.S^{1/2} . (S^{-1/2}.C) = (S^{-1/2}.C).n
    s_eig = np.linalg.eigh(s_ii)
    s_inv_sqrt = (
        s_eig.eigenvectors
        @ np.diag(1.0 / np.sqrt(np.maximum(s_eig.eigenvalues, 1e-12)))
        @ s_eig.eigenvectors.T
    )
    s_sqrt = (
        s_eig.eigenvectors
        @ np.diag(np.sqrt(np.maximum(s_eig.eigenvalues, 1e-12)))
        @ s_eig.eigenvectors.T
    )

    d_orth = s_sqrt @ d_local @ s_sqrt
    nat_occ, nat_vecs = np.linalg.eigh(d_orth)
    # Sort descending.
    order = np.argsort(-nat_occ)
    nat_occ = nat_occ[order]
    nat_vecs = nat_vecs[:, order]

    if n_occ is None:
        n_occ = int(np.sum(nat_occ > 1e-6))
    n_occ = min(n_occ, n_i)

    C_nat = s_inv_sqrt @ nat_vecs[:, :n_occ]

    # Build atom->AO map for region I.
    amap, atom_pos = _ao_atom_map(basis, np.asarray(i_ao, dtype=np.int64))

    # Both localisers return the localised coefficients ALONE (``C @ U``);
    # neither hands back the mixing matrix.  Recover it below.
    if method == "pipek-mezey":
        C_loc = pipek_mezey_localise(
            C_nat,
            s_ii,
            amap,
            max_iter=200,
            conv_tol=1e-8,
        )
        # PM never forms the position operator, so there is no centroid.
        centroids = np.zeros((n_occ, 3))
    elif method == "boys":
        # Boys needs the Cartesian dipole integrals <mu|r|ν>, shape
        # (n_i, n_i, 3) -- NOT the overlap matrix.
        dipoles = _region_dipoles(basis, np.asarray(i_ao, dtype=np.int64))
        C_loc = foster_boys_localise(
            C_nat,
            dipoles,
            max_iter=200,
            conv_tol=1e-8,
        )
        # True position centroids <i|r|i>; faithful for non-wrapping
        # orbitals only (see module docstring).
        centroids = np.einsum("mi,mnc,ni->ic", C_loc, dipoles, C_loc)
    else:
        raise ValueError(f"Unknown method: {method}")

    # C_nat is S-orthonormal by construction (C_nat = S^{-1/2} V with V
    # orthonormal, so C_nat^T S C_nat = I), hence the transform that
    # produced C_loc is U = C_nat^T S C_loc.  Same recovery as
    # ``vibeqc.periodic_localise.localise_periodic_gamma``.
    U = C_nat.T @ s_ii @ C_loc

    # Mulliken-charge-weighted centers, PBC-robust for both methods.
    charges = (C_loc * (s_ii @ C_loc)).T @ amap
    centers = charges @ atom_pos

    # <r^2> is unavailable here -- see the module docstring.
    spreads = np.zeros(n_occ)

    return EmbeddedLocalisedResult(
        C_loc=C_loc,
        U=U,
        centers=centers,
        centroids=centroids,
        spreads=spreads,
        method=method,
    )
