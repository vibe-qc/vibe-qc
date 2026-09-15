"""Wigner-Seitz supercell (WSSC) geometry and integral weights for the CCM.

This is the *geometry engine* of the Ab Initio Cyclic Cluster Model
(AICCM). It is deliberately **method- and integral-agnostic**: it knows
nothing about HF, DFT, MP2 or any electronic-structure method. It only
answers the geometric question that the CCM weighting scheme reduces to --

    "given a reference atom A and another atom B, which translational
     image(s) of B fall inside A's Wigner-Seitz supercell, and with what
     fractional weight?"

Every CCM integral (overlap, kinetic, nuclear attraction, ERIs) is then
built by folding the real-space lattice blocks against the weights this
module produces. Because the weighting lives here -- at the integral
layer -- and not inside any SCF routine, *all* downstream methods that
consume the weighted integrals (RHF/UHF/ROHF/RKS/UKS and the post-HF
correlation kernels) inherit CCM behaviour for free. That is the design
property required for the long-term goal of a method-general CCM.

Efficiency
----------
The minimum image of a pair is found in **O(1)** by reducing the
displacement to fractional coordinates of the cluster lattice and
correcting over a small fixed neighbour shell -- there is *no* loop over a
global cell list. The CCM caller (:class:`CCMSystem`) further exploits
the supercell's translational symmetry: the weight of a pair depends only
on the class ``(beta_A, beta_B, delta_cell)``, so the geometry is built
over ``O(n_beta * n_atoms)`` unique displacement classes (linear in the
cluster size) rather than ``O(n_atoms**2)``. This matters because the
core CCM workflow is a cluster-size convergence sweep over growing N.

Theory reference
----------------
M. F. Peintinger & T. Bredow, *J. Comput. Chem.* **35**, 839 (2014),
doi:10.1002/jcc.23550 -- "The Cyclic Cluster Model at Hartree-Fock
Level". The two-center weighting scheme is eqs (4)-(7):

    # Eq. 4:  omega_{mu nu'} = 1 / n_{nu'}
    #   n_{nu'} = number of translationally equivalent images of nu' that
    #   occur (at equal, minimal distance) within mu's WSSC.
    # Eq. 5:  S^CCM_{mu nu} = sum_{nu'} omega_{mu nu'} <mu | nu'>
    # Eq. 6:  T^CCM_{mu nu} = sum_{nu'} omega_{mu nu'} <mu | -1/2 nabla^2 | nu'>

An interior pair has a unique nearest image (``n = 1`` -> weight 1, the
ordinary molecular integral). A pair whose nearest image lies exactly on
a WSSC boundary is shared: ``n = 2`` on a face, ``4`` on an edge, ``8`` on
a corner in 3D (cf. Fig. 2 of the paper, ``S_{mu rho} = 1/2<mu|rho> +
1/2<mu|rho^->``).
"""

from __future__ import annotations

from itertools import product

import numpy as np

__all__ = [
    "minimum_image",
    "min_image_multiplicity",
    "first_shell_vectors",
    "wigner_seitz_weights_reference",
    "interplanar_spacings",
    "shortest_lattice_vector_length",
    "wsc_inscribed_radius",
    "nrep_for_interaction_range",
    "kspacing_for_interaction_range",
]


def first_shell_vectors(cluster_lattice: np.ndarray) -> np.ndarray:
    """The 27 translations ``i*a1 + j*a2 + k*a3`` for ``i,j,k in {-1,0,1}``.

    Parameters
    ----------
    cluster_lattice : (3, 3) array
        Cluster lattice with **rows** equal to the lattice vectors
        ``a1, a2, a3`` (the vibe-qc ``PeriodicSystem.lattice`` convention).

    Returns
    -------
    (27, 3) array of Cartesian translation vectors.
    """
    L = np.asarray(cluster_lattice, dtype=float)
    rng = (-1, 0, 1)
    out = [i * L[0] + j * L[1] + k * L[2] for i in rng for j in rng for k in rng]
    return np.asarray(out, dtype=float)


def _minkowski_reduce(lattice: np.ndarray):
    """Minkowski-reduce a 3x3 lattice (rows = vectors).

    Returns ``(rcell, op)`` with ``rcell = op @ lattice`` and ``op`` an
    integer unimodular matrix, so reduced-basis integer cells map back to
    the original basis by ``m = m_reduced @ op``. After reduction a small
    fixed neighbour search is provably sufficient for the closest-image
    problem in dimension <= 3 -- this is what makes the minimum image exact
    for *any* crystal lattice (triclinic, hexagonal, rhombohedral, fcc/bcc
    primitive, ...), not just orthogonal ones.

    Falls back to the identity transform if reduction is unavailable.
    """
    L = np.asarray(lattice, dtype=float)
    try:
        from ase.geometry import minkowski_reduce  # ASE is a core dependency
    except Exception:  # pragma: no cover - ASE always present in practice
        return L, np.eye(3, dtype=int)
    rcell, op = minkowski_reduce(L)
    return np.asarray(rcell, dtype=float), np.asarray(op, dtype=int)


# ---------------------------------------------------------------------------
# Real-space interaction range -> cluster size (the real-space dual of k-mesh
# sampling). See docs/aiccm2026dev_a_followon.md Sec. "Interaction-range
# parameterization". This is a lattice/finite-group geometry statement, not a
# Γ-CCM/χ-CCM construction-equivalence statement.
# ---------------------------------------------------------------------------


def interplanar_spacings(lattice_vectors: np.ndarray) -> np.ndarray:
    """Interplanar spacings ``d_i`` along each lattice direction (bohr).

    For a full-rank 3-D lattice with **rows** ``a_1, a_2, a_3`` the spacing
    between successive lattice planes spanned by ``(a_j, a_k)`` and stacked
    along ``a_i`` is

        d_i = V / |a_j x a_k| ,    V = |det(a_1, a_2, a_3)| ,  {j,k} = {1,2,3}\\{i}

    i.e. the component of ``a_i`` perpendicular to the other two. Equivalently
    ``d_i = 1/|b_i|`` with the (2pi-free) reciprocal vector ``b_i = (a_jxa_k)/V``
    (so ``a_i . b_j = d_ij`` and ``a_i`` projects onto the unit reciprocal
    direction ``b̂_i`` with length ``d_i``). For an orthogonal cell
    ``d_i = |a_i|``; for a vacuum-padded direction ``d_i`` is the (large)
    padding length.

    These spacings are the real-space dual of the reciprocal-vector lengths
    ``|b_i| = 1/d_i`` that set k-point density, and they fix the cutoff ->
    cluster-size map :func:`nrep_for_interaction_range`.
    """
    L = np.asarray(lattice_vectors, dtype=float)
    if L.shape != (3, 3):
        raise ValueError(
            f"lattice_vectors must be (3,3) with rows a_i, got {L.shape}"
        )
    V = abs(float(np.linalg.det(L)))
    if V < 1e-30:
        raise ValueError(
            "lattice is singular (zero cell volume); a full-rank 3x3 lattice is "
            "required -- pad non-periodic directions with a vacuum vector."
        )
    d = np.empty(3, dtype=float)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        d[i] = V / float(np.linalg.norm(np.cross(L[j], L[k])))
    return d


def shortest_lattice_vector_length(lattice_vectors: np.ndarray) -> float:
    """Length ``l₁`` of the shortest nonzero lattice vector (bohr).

    The basis is Minkowski-reduced first, so in dimension <= 3 a shortest
    nonzero vector is one of the reduced basis vectors and is therefore found
    among the 26 nonzero ``±1`` first-shell combinations of the reduced basis.
    Exact for any crystal lattice (triclinic, hexagonal, fcc/bcc primitive, ...).
    """
    L = np.asarray(lattice_vectors, dtype=float)
    reduced, _ = _minkowski_reduce(L)
    shell = first_shell_vectors(reduced)               # 27 combos incl. 0
    norms = np.linalg.norm(shell, axis=1)
    nz = norms[norms > 1e-12]
    if nz.size == 0:
        raise ValueError("degenerate lattice: no nonzero lattice vector found")
    return float(nz.min())


def wsc_inscribed_radius(lattice_vectors: np.ndarray) -> float:
    """Inscribed-sphere radius of the lattice's Wigner-Seitz cell (bohr).

    Equals **half the shortest lattice vector**, ``r_in = l₁/2``. Proof: a point
    ``x`` lies in the WS (Voronoi) cell iff ``x.R <= |R|^2/2`` for every lattice
    vector ``R``; for ``|x| = r`` the binding constraint is along ``R̂``, giving
    ``r <= |R|/2``, minimised by the shortest ``R`` -- so the largest centred
    sphere has radius ``l₁/2`` (the nearest Voronoi face perpendicularly bisects
    the shortest lattice vector). This is the radius out to which every
    neighbour is captured at full WSSC weight.
    """
    return 0.5 * shortest_lattice_vector_length(lattice_vectors)


def nrep_for_interaction_range(
    unit_lattice_vectors: np.ndarray,
    radius_bohr: float,
    *,
    dim: int = 3,
) -> tuple:
    """Minimal cluster size ``nrep`` whose supercell WS cell encloses a sphere
    of radius ``radius_bohr`` around every atom.

    The **real-space dual of k-point sampling**: instead of a k-mesh density it
    takes a real-space interaction cutoff ``R_c`` and returns the cyclic cluster
    (Born-von-Kármán torus) that reaches that far. Its reciprocal character
    mesh is the geometric dual for a fixed finite translation group; this does
    not identify different CCM construction maps.

    Derivation
    ----------
    The inscribed-sphere radius of a lattice's WS cell is ``r_in = l₁/2`` with
    ``l₁`` the shortest lattice vector (:func:`wsc_inscribed_radius`). For the
    cluster lattice ``L_c = {N_i a_i}`` we require ``r_in >= R_c`` i.e.
    ``l₁(L_c) >= 2 R_c``.

    A *per-direction* sufficient condition follows from a projection bound. Write
    any nonzero cluster vector ``R = S_i m_i N_i a_i`` and project onto the unit
    reciprocal direction ``b̂_i`` of the unit cell. Because ``a_i . b̂_i = d_i``
    (the interplanar spacing) and ``a_j . b̂_i = 0`` for ``j != i``, the cluster
    interplanar spacing is ``D_i = N_i d_i`` and ``|R| >= |R.b̂_i| = |m_i| D_i``.
    Hence ``l₁(L_c) >= min_i D_i``, so

        N_i = ⌈ 2 R_c / d_i ⌉   ⟹   D_i >= 2 R_c ∀ i   ⟹   l₁ >= 2 R_c   ⟹   r_in >= R_c.

    The bound is tight (equality) for orthogonal cells and conservative -- it
    never under-shoots -- for oblique cells (where ``l₁`` can exceed ``min_i D_i``,
    so the achieved ``r_in`` may be slightly larger than ``R_c``). The dual
    statement: a cyclic cluster of WS "diameter" ``2 R_c`` is a BvK torus whose
    allowed k-points are spaced by ``Δk = 2pi/(2R_c) = pi/R_c`` -- larger ``R_c`` ⇔
    denser k-mesh (:func:`kspacing_for_interaction_range`).

    Parameters
    ----------
    unit_lattice_vectors : (3, 3) array
        Unit-cell lattice, **rows** = ``a_1, a_2, a_3`` (bohr).
    radius_bohr : float
        Interaction range / WSC radius ``R_c`` (bohr), > 0.
    dim : int
        Number of periodic directions (1/2/3). Directions ``i >= dim`` are held
        at ``N_i = 1``; vacuum-padded directions self-limit to 1 via their large
        ``d_i`` regardless.

    Returns
    -------
    (N1, N2, N3) : tuple of int, each >= 1.
    """
    if radius_bohr <= 0.0:
        raise ValueError(f"radius_bohr must be > 0, got {radius_bohr}")
    d = interplanar_spacings(unit_lattice_vectors)
    nrep = [1, 1, 1]
    for i in range(3):
        if i >= int(dim):
            continue
        # ⌈2R_c/d_i⌉ with a tiny floor to avoid a spurious +1 when the ratio is
        # an exact integer perturbed upward by floating-point round-off.
        nrep[i] = max(1, int(np.ceil(2.0 * radius_bohr / d[i] - 1e-9)))
    return (nrep[0], nrep[1], nrep[2])


def kspacing_for_interaction_range(radius_bohr: float) -> float:
    """Equivalent reciprocal-space k-spacing ``Δk = pi/R_c`` (bohr⁻¹).

    The dual of :func:`nrep_for_interaction_range`: a real-space WSC interaction
    radius ``R_c`` corresponds to a uniform k-mesh of spacing ``Δk = pi/R_c``
    (2pi reciprocal convention), because the cyclic cluster of WS diameter
    ``2 R_c`` is a Born-von-Kármán torus sampling k on that spacing. Provided so
    a CCM radius scan can be reported alongside the k-mesh a Bloch calculation
    would use.
    """
    if radius_bohr <= 0.0:
        raise ValueError(f"radius_bohr must be > 0, got {radius_bohr}")
    return float(np.pi / radius_bohr)


def minimum_image(
    disps: np.ndarray,
    cluster_lattice: np.ndarray,
    *,
    tol_bohr: float = 1e-6,
    search: int = 2,
    reduced_basis=None,
):
    """Minimum-image cluster cell(s) and WSSC weights for displacements.

    For each Cartesian displacement ``d = r_A - r_B`` this returns the
    integer cluster-lattice cell(s) ``g`` (in the **original** basis) that
    minimise ``|d - g @ L_c|`` -- i.e. the cell(s) bringing B's image inside
    A's Wigner-Seitz supercell -- with the weight ``1/n`` for an ``n``-fold
    boundary tie.

    Correct for **all crystal lattices**: the basis is Minkowski-reduced
    first (:func:`_minkowski_reduce`), so the displacement reduced to
    fractional coordinates of the reduced cell has its closest image within
    a small fixed neighbour shell. The search is therefore **O(1)** per
    displacement (``(2*search+1)**3`` candidates, vectorised over the input)
    -- no global cell list is enumerated. Reduced-basis cells are mapped back
    to the original cluster basis via the unimodular transform.

    Parameters
    ----------
    disps : (M, 3) array
        Cartesian displacement vectors (bohr).
    cluster_lattice : (3, 3) array
        Cluster lattice, rows = ``a1, a2, a3``.
    tol_bohr : float
        Distance tolerance for declaring images equidistant (boundary tie).
    search : int
        Half-width of the reduced-basis neighbour shell (``2`` -> 5x5x5).
        ``1`` is provably sufficient after reduction; ``2`` adds margin so
        legitimate boundary ties never sit on the search rim.
    reduced_basis : (rcell, op), optional
        Precomputed Minkowski reduction (to amortise it across many calls
        on the same lattice). Computed from ``cluster_lattice`` if omitted.

    Returns
    -------
    cell_lists : list of (k_m, 3) int arrays
        The tied minimum-image cells (original basis) for each displacement.
    weight_lists : list of (k_m,) float arrays
        Equal weights ``1/k_m`` (so each pair's total weight is 1).
    """
    D = np.atleast_2d(np.asarray(disps, dtype=float))
    rcell, op = reduced_basis if reduced_basis is not None else _minkowski_reduce(
        np.asarray(cluster_lattice, dtype=float)
    )

    f = D @ np.linalg.inv(rcell)                              # (M, 3) fractional
    base = np.rint(f).astype(int)                             # (M, 3) nearest cell
    res = f - base                                            # (M, 3) in [-0.5, 0.5]

    rng = np.arange(-search, search + 1)
    shifts = np.array(list(product(rng, rng, rng)), dtype=int)        # (S, 3)
    # Cartesian residual of reduced-basis cell (base + shift): (res - shift) @ rcell.
    cart = (res[:, None, :] - shifts[None, :, :]) @ rcell             # (M, S, 3)
    dist = np.linalg.norm(cart, axis=2)                              # (M, S)
    dmin = dist.min(axis=1)                                          # (M,)
    tie = dist <= (dmin[:, None] + tol_bohr)                         # (M, S)

    # Rim guard: the chosen image must not sit on the outer shell, else a
    # closer image could exist just beyond the searched neighbourhood.
    smax = np.abs(shifts).max(axis=1)                               # (S,)
    if np.any(tie & (smax[None, :] == search) & (dmin[:, None] > tol_bohr)):
        raise ValueError(
            "minimum_image: a nearest image fell on the reduced-basis search "
            f"rim (search={search}) -- unexpected after Minkowski reduction; "
            "increase `search`."
        )

    cell_lists, weight_lists = [], []
    for m in range(D.shape[0]):
        sel = np.flatnonzero(tie[m])
        cells_reduced = base[m] + shifts[sel]                        # (k, 3)
        cell_lists.append(cells_reduced @ op)                        # -> original basis
        weight_lists.append(np.full(sel.size, 1.0 / sel.size))
    return cell_lists, weight_lists


def min_image_multiplicity(
    r_a: np.ndarray,
    r_b: np.ndarray,
    cluster_lattice: np.ndarray,
    tol_bohr: float = 1e-6,
    search: int = 2,
) -> int:
    """Boundary multiplicity ``n`` of atom B in atom A's WSSC (eq. 4's ``n``).

    1 for an interior pair; 2/4/8 for a pair on a WSSC face/edge/corner.
    """
    cells, _ = minimum_image(
        np.asarray(r_a, float) - np.asarray(r_b, float),
        cluster_lattice,
        tol_bohr=tol_bohr,
        search=search,
    )
    return int(len(cells[0]))


def wigner_seitz_weights_reference(
    positions: np.ndarray,
    cell_vectors: np.ndarray,
    tol_bohr: float = 1e-6,
) -> np.ndarray:
    """Reference (brute-force) WSSC weights ``w[g, A, B]`` -- test oracle only.

    Simple, obviously-correct O(n_cells * n_atoms**2) implementation kept
    as the validation oracle for the fast, symmetry-exploiting path in
    :class:`CCMSystem`. Not used in production.

    For each ordered pair (A, B), the images of B are ``r_B + R_g`` over
    the supplied cell translations; the nearest image(s) receive weight
    ``1/n``. ``cell_vectors`` must contain the true minimum-image cell of
    every pair (and the zero/home cell).
    """
    P = np.asarray(positions, dtype=float)
    R = np.asarray(cell_vectors, dtype=float)
    n_atoms = P.shape[0]
    n_cells = R.shape[0]
    weights = np.zeros((n_cells, n_atoms, n_atoms), dtype=float)
    for a in range(n_atoms):
        diff = P[a][None, None, :] - (P[None, :, :] + R[:, None, :])
        dist = np.linalg.norm(diff, axis=2)               # (n_cells, n_atoms)
        dmin = dist.min(axis=0)
        tie = dist <= (dmin[None, :] + tol_bohr)
        mult = tie.sum(axis=0)
        weights[:, a, :] = np.where(tie, 1.0 / mult[None, :], 0.0)
    return weights
