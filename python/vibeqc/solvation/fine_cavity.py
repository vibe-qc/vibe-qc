"""The COSMO FINE Cavity (CFC) -- Klamt & Diedenhofen 2018.

*A Refined Cavity Construction Algorithm for the Conductor-like Screening
Model*, J. Comput. Chem. **39**, 1648 (2018), doi:10.1002/jcc.25342.

Why a different cavity
----------------------
vibe-qc's other cavity (:mod:`vibeqc.solvation.cavity`) paves each atomic
sphere with a Lebedev grid and switches segments off where spheres overlap.
That describes the *convex* part of the solute surface well and leaves the
concave regions -- crevices between atoms -- unpaved, which is precisely the
COC construction's known deficiency (2018, "The original COSMO cavity
construction"). Klamt's own standard cavity (CSC) patches those regions with
analytic ring and triple-point segments, and that closure "increases the area
of the cavity by about 60%".

For COSMO-RS that matters more than for a solvation energy: the screening
charge *density* is the model's central descriptor, so polarization charge
piling up at the edges of paved regions distorts a sigma profile directly.

The CFC replaces the geometric patchwork with a single smooth iso-surface. A
scalar pseudo-density ``PD(r)`` is built from the atom positions, and the
cavity is its ``PD = 1`` iso-surface, triangulated by marching tetrahedra.
Concave regions are paved automatically because the iso-surface has no seams,
and because ``PD`` is an analytic function of the atom positions the segment
positions and areas are differentiable -- the requirement that made the older
constructions awkward for gradients.

What this module does and does not claim
----------------------------------------
It implements the construction of 2018 eq. 8 and its stated workflow, with the
paper's own parameters. It does **not** claim to reproduce COSMOtherm's CFC
segment-for-segment: the paper specifies the algorithm but not every tie-break,
and no reference segment set is available to check against. What is checked is
that the surface is closed, that its area and volume are converged in the grid
spacing, and that it behaves correctly in the limits where an exact answer
exists (a single atom is a sphere of the right radius and area).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .atom_partition import atom_partition
from .cavity import ANG_TO_BOHR

# ---------------------------------------------------------------------
# Pseudo-density (2018 eq. 8)
# ---------------------------------------------------------------------

# "The parameters were adjusted in a way that the resulting iso-surfaces
# achieve optimal agreement with the positions of the segment centers of the
# CSC cavity, and that the volumes of the resulting cavities are close to the
# volumes of the CSC cavities. This resulted in were chosen: a1 = -15,
# a2 = -9, c = 5 and m = 4." (2018, after eq. 8)
PD_A1 = -15.0
PD_A2 = -9.0
PD_C = 5.0
PD_M = 4

# "For numerical efficiency, in the evaluation of the PD function only the 5
# nearest atoms of a position r are taken into account." Note this makes PD
# non-smooth where the identity of the five nearest atoms changes, so it
# trades the differentiability the construction was designed for against
# speed; ``n_nearest=None`` keeps the exact smooth sum.
PD_N_NEAREST = 5

# "A grid size delta (0.3 A) is chosen for the marching tetrahedron
# algorithm" (2018 workflow step 2).
GRID_SPACING_ANG = 0.3

# "with NSPH(=92) and NSPA(=162) for hydrogen and other atoms, respectively"
# (2018 workflow step 6) -- note these are finer than the COC/CSC values of
# 32 and 92.
NSPH_FINE = 92
NSPA_FINE = 162

# "Basis grid points with an area smaller than a threshold area (1*10^-3
# bohr^2) are joined with the nearest neighbor basis grid point."
MIN_BASIS_AREA_BOHR2 = 1e-3

# Width of the band over which step 6's outward projection is eased off, as a
# fraction of the atom's own sphere radius (#770).
#
# The paper's rule is a hard one -- "segment centers with atom center distances
# smaller than the COSMO radius of the atom are moved outward to the sphere" --
# and a hard maximum is C0 but not C1: the slope of the segment position jumps
# where a centre crosses the sphere. Measured on an asymmetric solute, an
# area-weighted functional's slope jumped 0.07 percent across one crossing, and
# crossings arrive about every 0.025 bohr of displacement per coordinate. The
# energy is untouched -- it is the *gradient* that steps, so what this costs is
# finite-difference second derivatives.
#
# No parameter-free C1 smooth maximum exists, so the band is a choice, and it is
# bounded from both sides by things that were measured rather than assumed:
#
#  * It must be small against the projection depth it eases, or the correction
#    itself is blurred. That depth is the sagitta of a segment on its sphere:
#    median 0.019 bohr, max 0.033, against radii of 2.7 to 3.9 bohr. At
#    1e-03 of the radius the band is 0.0034 bohr, about six times smaller, and
#    only 51 of 401 segments move by more than 1e-04 bohr against the hard rule.
#  * It must be large enough for a finite difference to resolve, or it removes
#    the kink for the derivative without helping the consumer that cares. A
#    typical FD step is 1e-03 to 2e-03 bohr, so the band spans it.
#
# The form is a softplus, ``r_eff = rho + w ln(1 + exp((d - rho)/w))``, chosen
# over a spliced polynomial for one reason beyond smoothness: it is never below
# ``rho``, so no segment is ever placed *inside* the sphere it screens, which is
# the physical error the projection exists to prevent. It recovers the hard rule
# as ``w -> 0``, and measurably: on water the solvated energy moves 6.6e-07 Ha
# from the hard rule at this width, 2.1e-09 at a tenth of it, and nothing at all
# at a hundredth.
PROJECTION_SOFTNESS = 1.0e-3

# Segments whose smoothly-partitioned area falls below this are dropped. With a
# hard assignment most of an atom's directions received no basis point at all
# and were dropped exactly; with Becke weights every direction receives *some*
# weight, so a floor is needed. It is not a smoothing parameter: a segment of
# area ``a`` carries charge of order ``sqrt(a)`` (the Gaussian self-energy
# ``A_ii`` scales as ``a^{-1/2}``), so its contribution to the energy vanishes
# with the threshold and dropping it is continuous to that order. Set far below
# any area that matters, and its effect is measured rather than assumed.
MIN_SEGMENT_AREA_BOHR2 = 1e-4


@dataclass(frozen=True)
class PseudoDensityParams:
    """Parameters of 2018 eq. 8. Defaults are the published values."""

    a1: float = PD_A1
    a2: float = PD_A2
    c: float = PD_C
    m: int = PD_M
    n_nearest: Optional[int] = PD_N_NEAREST


def pseudo_density(
    points: np.ndarray,
    atom_positions: np.ndarray,
    atom_radii: np.ndarray,
    params: PseudoDensityParams | None = None,
) -> np.ndarray:
    """2018 eq. 8, evaluated at many points at once.

        PD(r) = sum_a exp{a1 (tau_a - 1)}
              + sum_a sum_{b>a} c (1 - tau_a . tau_b)^m
                                  exp{a2 (tau_a + tau_b - 2)}

    ``tau_a`` is the *relative* distance ``|r - R_a| / R_a`` and the vector
    ``tau_a`` is ``(r - R_a) / R_a``, so the dot product in the pair term is
    ``tau_a tau_b cos(angle)``.

    The first sum alone puts the iso-surface exactly on the van der Waals
    sphere of an isolated atom: ``tau_a = 1`` gives ``exp(0) = 1``. The pair
    term is largest *between* two atoms, where the relative distance vectors
    are antiparallel and the cosine is ``-1``, so it inflates the surface
    across bonds and fills the crevice that a union of spheres leaves open.
    """
    p = params or PseudoDensityParams()
    r = np.atleast_2d(np.asarray(points, dtype=np.float64))
    R = np.asarray(atom_positions, dtype=np.float64)
    rad = np.asarray(atom_radii, dtype=np.float64)
    if rad.shape != (R.shape[0],):
        raise ValueError(
            f"pseudo_density: {rad.size} radii for {R.shape[0]} atoms."
        )
    if np.any(rad <= 0.0):
        raise ValueError("pseudo_density: atom radii must be positive.")

    # tau vectors: (n_pts, n_atoms, 3); tau scalars: (n_pts, n_atoms)
    tau_vec = (r[:, None, :] - R[None, :, :]) / rad[None, :, None]
    tau = np.linalg.norm(tau_vec, axis=2)

    keep = None
    if p.n_nearest is not None and R.shape[0] > p.n_nearest:
        # Smallest relative distance, per the paper's "5 nearest atoms".
        keep = np.argsort(tau, axis=1)[:, : p.n_nearest]
        rows = np.arange(r.shape[0])[:, None]
        tau = tau[rows, keep]
        tau_vec = tau_vec[rows, keep]

    atomic = np.sum(np.exp(p.a1 * (tau - 1.0)), axis=1)

    n_a = tau.shape[1]
    if n_a < 2:
        return atomic
    ia, ib = np.triu_indices(n_a, k=1)
    dot = np.einsum("pak,pak->pa", tau_vec[:, ia, :], tau_vec[:, ib, :])
    pair = np.sum(
        p.c
        * np.power(1.0 - dot, p.m)
        * np.exp(p.a2 * (tau[:, ia] + tau[:, ib] - 2.0)),
        axis=1,
    )
    return atomic + pair


def log_pseudo_density(
    points: np.ndarray,
    atom_positions: np.ndarray,
    atom_radii: np.ndarray,
    params: PseudoDensityParams | None = None,
) -> np.ndarray:
    """``PD* = ln(PD)``, the field the marching tetrahedra actually cut.

    The paper evaluates ``PD*`` rather than ``PD`` because ``PD`` spans many
    orders of magnitude between the molecular interior and the vacuum, while
    ``ln PD`` is close to linear across the iso-surface -- which is what makes
    the interpolation along a cut edge accurate. The iso-value becomes zero.
    """
    pd = pseudo_density(points, atom_positions, atom_radii, params)
    with np.errstate(divide="ignore"):
        return np.log(np.where(pd > 0.0, pd, np.finfo(np.float64).tiny))


# ---------------------------------------------------------------------
# The molecule-fixed frame (2018 workflow step 1)
# ---------------------------------------------------------------------

# An orthogonal component smaller than this, relative to the largest
# atom-to-centre distance, counts as absent: the solute is linear (or a single
# atom) and the paper's construction determines no y-axis. It is not a
# smoothing parameter -- it only selects which branch reports "undetermined",
# and both branches return an orthonormal right-handed frame.
_FRAME_DEGENERATE_REL = 1e-12

# Two atoms count as equally far from the centre when their distances agree to
# this, relative to the largest of them, and the tie then goes to the lower atom
# index.
#
# Rotating a molecule does not change any distance, but it does change the
# rounding of the arithmetic that computes one, so a bare ``argmax`` over
# distances is *not* rotation-invariant when two of them tie: on water, whose
# hydrogens are exactly equidistant from the centroid, the selected x-axis
# flipped between the two of them from one orientation to the next. The atom
# index is the one key rotation cannot touch, so near-ties are resolved with it.
#
# The window has to sit far above the rounding it absorbs (a few ulp, ~1e-16
# relative) and far below any distance difference that means something. It does
# not add a discontinuity under nuclear motion: a bare ``argmax`` already jumps
# where the tie breaks, and this only moves that jump from a gap of zero to a
# gap of 1e-12.
_FRAME_TIE_REL = 1e-12

# eps_ijk, for the two cross products the frame Jacobian differentiates.
_LEVI_CIVITA = np.zeros((3, 3, 3))
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _LEVI_CIVITA[_i, _j, _k] = 1.0
    _LEVI_CIVITA[_i, _k, _j] = -1.0


def _leading_set(values: np.ndarray, scale: float) -> np.ndarray:
    """Every entry within the tie window of the largest, lowest index first."""
    top = float(np.max(values))
    return np.flatnonzero(values >= top - _FRAME_TIE_REL * scale)


def _leading_index(values: np.ndarray, scale: float) -> int:
    """The largest entry of ``values``, ties going to the lower index."""
    return int(_leading_set(values, scale)[0])


def _frame_completion(e1: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to ``e1``, for a solute that fixes none.

    A linear solute has no atom with an orthogonal component, so step 1 stops
    after the x-axis and the azimuth is free. **No rotation-equivariant choice
    exists**: the rotations that fix ``e1`` are exactly the rotations about it,
    and an equivariant ``e2(e1)`` would have to be invariant under all of them
    while being perpendicular to ``e1``. So this returns a deterministic
    lab-derived completion and the azimuthal dependence remains.

    What remains is small and bounded, which is why the frame is still worth
    having for a linear solute: the exact iso-surface of a linear molecule is
    axially symmetric, so all that the azimuth can change is how the segment
    grid paves a surface that looks the same from every azimuth -- discretization
    anisotropy, not a different surface. The full lab-frame defect of #769 is
    larger because there the *marching grid* also reorients, which changes the
    surface itself.
    """
    k = int(np.argmin(np.abs(e1)))
    u = np.zeros(3)
    u[k] = 1.0
    v = np.cross(e1, u)
    return v / np.linalg.norm(v)


def molecular_frame(atom_positions_bohr) -> np.ndarray:
    """The molecule-fixed axes of 2018 workflow step 1, as rows in lab coordinates.

    "For a given geometry of a molecule X, i.e. a set of atom positions {r_a},
    first a coordinate system is constructed by using the center of the atom
    positions as origin, and by defining the direction to the atom with the
    largest distance from the center as x-axis. Then the atom with the largest
    orthogonal component to the x-axis is determined, and its normalized
    orthogonal component is used as y-axis. The z-axis is defined as vector
    product of the x- and y-axes." (2018 workflow step 1)

    Returns ``R`` with ``R[i]`` the i-th axis, so molecular coordinates of a lab
    position are ``(x - centre) @ R.T`` and the inverse map is ``x @ R + centre``.
    ``det R = +1``: a proper rotation, so the geodesic direction grids and the
    marching lattice are carried over without being mirrored.

    Why the frame exists (#769). Every lab-frame choice in the construction --
    the axis-aligned marching box of step 2 and the geodesic segment directions
    of step 6 -- makes the *discretized* cavity a function of how the solute
    happens to be oriented, though the surface it discretizes is not. Measured
    on water at 0.4 A spacing, rotating the solute moved the cavity area by
    5.3e-04 relative and the enclosed volume by 6.6e-04, and changed the segment
    count from 240 to 234; at 0.2 A spacing the area figure is 5.9e-05, so it is
    a convergence-limited artefact rather than a bug in one stage. Building in
    this frame removes it exactly: the construction sees the same coordinates
    whatever the lab orientation.

    Equivariance, not smoothness. Distances to the centre do not change under
    rotation, so the two selections pick the same atoms and the axes rotate
    rigidly with the solute -- which is all that rotational invariance needs. It
    holds even when two atoms tie exactly, as water's hydrogens do, but only
    because the tie is resolved by atom index (``_FRAME_TIE_REL``): a bare
    ``argmax`` reads the *rounding* of two equal distances and flips between
    them from one orientation to the next.

    Under a *nuclear displacement* the same selection is discontinuous where the
    tie breaks, which is the #757 class of defect and the reason this frame is
    opt-in and the analytic derivative refuses it: see #769 and
    :class:`~vibeqc.solvation.cavity_derivative.FineCavityDerivative`.

    One case recovers equivariance only partly. A *nearly* linear solute has a
    largest orthogonal component of some small size ``eps`` relative to the
    molecule, and forming it cancels, so its direction is known to about
    ``1e-16 / eps``: the y-axis, and with it the cavity, is then equivariant to
    that rather than to machine precision. Still far below the 6e-04 the
    laboratory frame costs, and an exactly linear solute takes the
    ``_frame_completion`` branch instead.
    """
    pos = np.asarray(atom_positions_bohr, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(
            f"molecular_frame: positions must be (n, 3), got {pos.shape}"
        )
    if pos.shape[0] == 0:
        raise ValueError("molecular_frame: no atoms.")
    d = pos - pos.mean(axis=0)
    r = np.linalg.norm(d, axis=1)
    far = _leading_index(r, float(np.max(r)))
    if r[far] <= 0.0:
        # One atom, or all atoms coincident. The construction determines
        # nothing and nothing depends on it: the field of a single centre is
        # isotropic, so every frame gives the same cavity.
        return np.eye(3)
    e1 = d[far] / r[far]
    orth = d - np.outer(d @ e1, e1)
    o = np.linalg.norm(orth, axis=1)
    wide = _leading_index(o, r[far])
    if o[wide] <= _FRAME_DEGENERATE_REL * r[far]:
        e2 = _frame_completion(e1)
    else:
        e2 = orth[wide] / o[wide]
        # Re-orthogonalize: ``orth`` is perpendicular to ``e1`` only to
        # rounding, and the frame is applied as an exact rotation below.
        e2 = e2 - (e2 @ e1) * e1
        e2 = e2 / np.linalg.norm(e2)
    return np.stack([e1, e2, np.cross(e1, e2)])


def molecular_frame_jacobian(atom_positions_bohr) -> np.ndarray:
    """``d(e_i)_j / d(R_A)_c`` for the step-1 axes. Shape ``(3, 3, n_atoms, 3)``.

    The frame of :func:`molecular_frame` is a function of the geometry, so a
    cavity built in it has a lattice and a set of segment directions that turn
    as the atoms move. This is the derivative that
    :class:`~vibeqc.solvation.cavity_derivative.FrameCavityDerivative` composes
    with the construction's own chain rule; without it the assembled gradient is
    wrong by exactly the frame's contribution.

    The two selections are **piecewise constant**, so they contribute nothing:
    the derivative is taken at fixed choice of the far atom and the wide atom,
    which is the same convention every other stage of the CFC derivative uses
    for its discrete decisions (the segment topology, the atom assignment). It
    is also where the frame's own discontinuity lives -- see
    :func:`molecular_frame`.

    Writing ``P = I - e1 e1^T`` and ``a*``, ``b*`` for the two selected atoms::

        d(d_a)/d(R_A)   = (delta_aA - 1/n) I
        d(e1)/d(R_A)    = (delta_{a*A} - 1/n) P / r
        u               = P d_b*
        d(u)/d(R_A)     = (delta_{b*A} - 1/n) P
                          - (e1 . d_b*) d(e1)/d(R_A)
                          - e1 (d_b* . d(e1)/d(R_A))
        d(e2)/d(R_A)    = (I - e2 e2^T) d(u)/d(R_A) / |u|
        d(e3)/d(R_A)    = d(e1)/d(R_A) x e2 + e1 x d(e2)/d(R_A)

    Two exact invariants follow, and the tests use both instead of finite
    differences: the frame does not move under a rigid translation, so
    ``sum_A d(e_i)/d(R_A) = 0``; and it turns rigidly under a rigid rotation, so
    contracting with the field ``omega x R_A`` gives ``omega x e_i``. Both hold
    to machine precision on ordinary solutes. Finite differences agree to
    1.3e-10 at ``h = 1e-6``, which is the reference's own floor.

    A *nearly* linear solute is stiff rather than wrong: the largest orthogonal
    component is small, so ``1/|u|`` is large and the y-axis derivative with it
    (measured 1.0e+03 for a 1e-03 bohr off-axis atom). The invariants still hold
    there; finite differences do not, which is the usual sign that the exact
    invariants are the better oracle. An *exactly* linear solute is refused --
    see the branch below.
    """
    pos = np.asarray(atom_positions_bohr, dtype=np.float64)
    frame = molecular_frame(pos)
    n_at = pos.shape[0]
    out = np.zeros((3, 3, n_at, 3))

    d = pos - pos.mean(axis=0)
    r = np.linalg.norm(d, axis=1)
    far = _leading_index(r, float(np.max(r)))
    if r[far] <= 0.0:
        if n_at == 1:
            # The identity, for every geometry a single atom can have, so the
            # derivative really is zero rather than merely unavailable.
            return out
        raise NotImplementedError(
            "molecular_frame_jacobian: every atom sits at the centre, so the "
            "frame is the identity only until they separate and it jumps. That "
            "geometry has no cavity either; fix the geometry rather than "
            "differentiating this."
        )

    e1, e2 = frame[0], frame[1]

    # ``share[A]`` is ``delta_{sel,A} - 1/n``: the selected atom moves the
    # centre with it, so its own displacement is reduced by the centroid's.
    #
    # Averaging this over a *tied* selection was tried and is wrong, which is
    # worth recording because it is the obvious idea. At a tie the frame does
    # not vary smoothly with two one-sided derivatives to average -- it *jumps*
    # between two finite directions -- so an averaged Jacobian is not the
    # derivative of any frame, and it breaks the exact rotation identity that
    # the real one satisfies to machine precision. What a tie actually costs is
    # measured by ``test_the_gradient_is_ill_conditioned_near_a_frame_tie``.
    def share(sel: int) -> np.ndarray:
        s = np.full(n_at, -1.0 / n_at)
        s[sel] += 1.0
        return s

    proj1 = np.eye(3) - np.outer(e1, e1)
    # d(e1)_j / d(R_A)_c
    de1 = np.einsum("A,jc->jAc", share(far) / r[far], proj1)

    orth = d - np.outer(d @ e1, e1)
    o = np.linalg.norm(orth, axis=1)
    wide = _leading_index(o, r[far])
    if o[wide] <= _FRAME_DEGENERATE_REL * r[far]:
        # A linear solute takes ``_frame_completion``, and that branch has no
        # derivative to report: the frame is *discontinuous* there. Displace any
        # atom off the axis and step 1 leaves this branch for the geometric one,
        # where the y-axis is the azimuth of that displacement -- an arbitrary
        # angle away from the lab-derived completion, so the cavity is re-paved
        # and the energy jumps by the azimuthal paving anisotropy.
        #
        # The within-branch derivative is easy to write and would be the "smooth,
        # plausible, wrong" answer of #546: finite differences contradict it by
        # 1e+06, and the rotation identity below fails outright (measured 2.0e-01
        # against a scale of 0.70) because no equivariant completion exists.
        raise NotImplementedError(
            "molecular_frame_jacobian: this solute is linear (or a single "
            "point), so workflow step 1 fixes only the x-axis and the frame's "
            "azimuth is a discontinuous lab-derived choice -- it has no nuclear "
            "derivative. Use frame='lab' for an analytic gradient on a linear "
            "solute. Note cpcm_gradient_fd cannot rescue this one either: its "
            "displacements cross the same jump."
        )
    else:
        db = d[wide]
        du = (
            np.einsum("A,jc->jAc", share(wide), proj1)
            - (e1 @ db) * de1
            - np.einsum("j,Ac->jAc", e1, np.einsum("k,kAc->Ac", db, de1))
        )
        de2 = np.einsum("ij,jAc->iAc", np.eye(3) - np.outer(e2, e2),
                        du) / o[wide]

    de3 = (
        np.einsum("ijk,jAc,k->iAc", _LEVI_CIVITA, de1, e2)
        + np.einsum("ijk,j,kAc->iAc", _LEVI_CIVITA, e1, de2)
    )
    out[0], out[1], out[2] = de1, de2, de3
    return out


# ---------------------------------------------------------------------
# Geodesic spheres: the "magic numbers" of the COC segment grids
# ---------------------------------------------------------------------

_ICOSAHEDRON_PHI = (1.0 + 5.0 ** 0.5) / 2.0


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Unit-sphere icosahedron: 12 vertices, 20 faces."""
    phi = _ICOSAHEDRON_PHI
    verts = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=int,
    )
    return verts, faces


def geodesic_sphere(frequency: int) -> np.ndarray:
    """Unit-sphere points from icosahedral subdivision, ``10 n^2 + 2`` of them.

    These are the COC "magic numbers": 2018 states that only
    ``10 i^2 3^j + 2`` point counts are allowed, "12, 32, 42, 92, 122, 162,
    252, ...". The ``j = 0`` family is plain edge subdivision of the
    icosahedron, which covers the two counts the CFC needs -- ``n = 3`` gives
    NSPH = 92 and ``n = 4`` gives NSPA = 162.

    A count that is not of this form is rejected rather than silently rounded:
    an off-lattice grid would not tile the sphere evenly, and the segment areas
    the cavity reports would inherit the unevenness.
    """
    n = int(frequency)
    if n < 1:
        raise ValueError(f"geodesic_sphere: frequency must be >= 1 (got {n}).")
    verts, faces = _icosahedron()
    out: dict[tuple[int, int, int], np.ndarray] = {}
    for f in faces:
        a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                # Barycentric lattice point, deduplicated on the integer key
                # so shared edges and corners are not counted twice.
                key = _face_key(f, i, j, k)
                if key in out:
                    continue
                p = (i * a + j * b + k * c) / n
                out[key] = p / np.linalg.norm(p)
    pts = np.array(list(out.values()), dtype=np.float64)
    expected = 10 * n * n + 2
    if pts.shape[0] != expected:
        raise RuntimeError(
            f"geodesic_sphere: built {pts.shape[0]} points, expected "
            f"{expected} for frequency {n}."
        )
    return pts


def _face_key(face: np.ndarray, i: int, j: int, k: int) -> tuple:
    """A subdivision-lattice key shared between adjacent faces.

    Keyed on the corner indices carrying *nonzero* barycentric weight, sorted.
    Dropping the zero-weight entries is what makes the key face-independent: a
    subdivision point on a shared edge has weight only on the two corners of
    that edge, so both adjacent faces produce the same key and the point is
    counted once. Keeping the zeros would tag it with the third, differing
    corner of each face and every shared point would be duplicated.
    """
    return tuple(
        sorted(
            (int(v), int(w))
            for v, w in ((face[0], i), (face[1], j), (face[2], k))
            if w != 0
        )
    )


def sphere_points_for_count(count: int) -> np.ndarray:
    """The geodesic grid with exactly ``count`` points (``10 n^2 + 2``)."""
    n2, rem = divmod(count - 2, 10)
    n = int(round(n2 ** 0.5))
    if rem != 0 or n * n != n2:
        raise ValueError(
            f"sphere_points_for_count: {count} is not of the form 10 n^2 + 2 "
            f"(nearest are {10 * n * n + 2} and {10 * (n + 1) ** 2 + 2})."
        )
    return geodesic_sphere(n)


__all__ = [
    "GRID_SPACING_ANG",
    "MIN_BASIS_AREA_BOHR2",
    "NSPA_FINE",
    "NSPH_FINE",
    "PD_A1",
    "PD_A2",
    "PD_C",
    "PD_M",
    "PD_N_NEAREST",
    "PseudoDensityParams",
    "geodesic_sphere",
    "log_pseudo_density",
    "molecular_frame",
    "molecular_frame_jacobian",
    "pseudo_density",
    "sphere_points_for_count",
]


# ---------------------------------------------------------------------
# Marching tetrahedra (2018 workflow step 3)
# ---------------------------------------------------------------------

# A cube split into five tetrahedra: one inscribed regular tetrahedron plus
# the four corner pyramids that complete the volume. Corner index is the bit
# pattern (x + 2y + 4z).
#
# "Please note that the two mirror images of the inscribed tetrahedron have to
# be used in an alternating manner, so that two neighboring cubes always share
# the same side diagonal as tetrahedron edge." Without the alternation the
# face diagonals of adjacent cubes disagree and the triangulation develops
# cracks -- which for a cavity means a surface that is not closed, and a
# volume that is meaningless.
_TETS_EVEN = np.array(
    [[1, 2, 4, 7], [0, 1, 2, 4], [3, 1, 2, 7], [5, 1, 4, 7], [6, 2, 4, 7]],
    dtype=int,
)
_TETS_ODD = np.array(
    [[0, 3, 5, 6], [1, 0, 3, 5], [2, 0, 3, 6], [4, 0, 5, 6], [7, 3, 5, 6]],
    dtype=int,
)

_CUBE_OFFSETS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
     [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
    dtype=int,
)

_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass(frozen=True)
class IsoSurface:
    """A triangulated iso-surface."""

    vertices: np.ndarray          # (n_vert, 3) bohr
    triangles: np.ndarray         # (n_tri, 3) int
    grid_spacing: float           # bohr
    # The fixed grid edge each vertex was located on, as its two endpoints.
    # Retained because it is what makes the analytic derivative possible: the
    # marching grid does not move with the atoms, so a vertex has exactly one
    # degree of freedom -- its position along this edge. See
    # vibeqc.solvation.fine_cavity_gradient.
    edge_p0: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    edge_p1: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # How the grid itself moves when an atom moves, as per-axis weights:
    # ``d(grid point)[i] / d R_A[c] = delta_ic * grid_anchor[A, c]``.
    #
    # The grid is only a device for locating the level set, so it is tempting
    # to call it fixed. It is not: :func:`build_fine_cavity` places the box
    # relative to the molecule, so the box moves when the molecule does, and a
    # derivative that ignores that differentiates a different function. The
    # cost of ignoring it is not small and not random -- it breaks
    # translational invariance, leaving a net force on the molecule. Measured
    # on water at 0.4 A spacing: the total-area gradient failed to sum to zero
    # by 0.34 bohr against a largest component of 14.7.
    #
    # Zero rows mean a genuinely fixed grid, which is what the unit tests
    # construct when they want to isolate the field derivative from the box.
    grid_anchor: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # The molecule-fixed frame the lattice was laid out in, as rows (#769), or
    # empty for a lab-frame lattice.
    #
    # Empty is the statement the derivative needs: a lab-frame lattice depends
    # on the atoms only through the box translation that ``grid_anchor``
    # records, so ``d(edge)/dR_A = 0`` and a vertex keeps its single degree of
    # freedom along a fixed direction. A molecule-fixed lattice *rotates* with
    # the solute, so both endpoints and the edge direction carry derivatives
    # that ``grid_anchor`` cannot express; ``iso_point_jacobians`` refuses such
    # a surface rather than differentiating a lattice it assumes is not
    # turning.
    frame: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # Which tetragon each triangle came from, or -1 for a triangle that stands
    # alone (#779).
    #
    # A tetrahedron with two corners inside the surface and two outside cuts
    # four edges, and the paper splits the resulting tetragon on its shorter
    # diagonal. That comparison flips where the two diagonals are equal, and
    # while the quad's *total* area survives the flip, step 4's per-corner
    # assignment does not. Recording the grouping is what lets step 4 divide
    # the quad's area over its four corners instead, which no diagonal can
    # change. Forty percent of a typical triangulation comes from such quads.
    quad_id: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    def triangle_areas(self) -> np.ndarray:
        v = self.vertices[self.triangles]
        return 0.5 * np.linalg.norm(
            np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1
        )

    @property
    def area(self) -> float:
        return float(np.sum(self.triangle_areas()))

    def enclosed_volume(self) -> float:
        """Volume by the divergence theorem, ``|sum (a x b) . c| / 6``.

        Requires a closed mesh *and* consistent winding. An open surface gives
        a plausible-looking number that means nothing, which is what
        :meth:`is_closed` guards; mixed winding cancels to near zero, which is
        what the orientation pass in :func:`marching_tetrahedra` guards.
        """
        v = self.vertices[self.triangles]
        return abs(
            float(np.sum(np.einsum("ij,ij->i", np.cross(v[:, 0], v[:, 1]), v[:, 2])))
            / 6.0
        )

    def is_closed(self) -> bool:
        """True when every triangle edge is shared by exactly two triangles.

        The decisive check on the triangulation. A cavity with a hole silently
        breaks both the enclosed volume and, once charges are placed on it, the
        screening: the field leaks through the gap.
        """
        counts: dict[tuple[int, int], int] = {}
        for tri in self.triangles:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (int(min(a, b)), int(max(a, b)))
                counts[key] = counts.get(key, 0) + 1
        return bool(counts) and all(c == 2 for c in counts.values())


def _edge_root(
    p0: np.ndarray,
    p1: np.ndarray,
    f0: np.ndarray,
    f1: np.ndarray,
    field,
    quadratic: bool,
) -> np.ndarray:
    """Where ``field = 0`` on the segments ``p0 -> p1``.

    Linear interpolation is the usual marching-tetrahedra choice; the paper
    says "the position of the PD*=0 point on the edge is approximated by
    quadratic interpolation", so the default samples the edge midpoint as well
    and solves the quadratic through the three values. On ``ln PD`` the field
    is nearly linear across the surface, so the correction is small -- but it
    is the paper's stated procedure and it costs one extra field evaluation.
    """
    denom = f0 - f1
    t = np.where(np.abs(denom) > 0.0, f0 / np.where(denom != 0.0, denom, 1.0), 0.5)
    t = np.clip(t, 0.0, 1.0)
    if not quadratic:
        return p0 + t[:, None] * (p1 - p0)

    mid = 0.5 * (p0 + p1)
    fm = field(mid)
    # Quadratic through (0, f0), (0.5, fm), (1, f1): f(t) = a t^2 + b t + f0
    a = 2.0 * (f0 - 2.0 * fm + f1)
    b = -3.0 * f0 + 4.0 * fm - f1
    disc = b * b - 4.0 * a * f0
    ok = (np.abs(a) > 1e-30) & (disc >= 0.0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    for cand in ((-b + sq) / (2.0 * np.where(ok, a, 1.0)),
                 (-b - sq) / (2.0 * np.where(ok, a, 1.0))):
        good = ok & (cand >= 0.0) & (cand <= 1.0)
        t = np.where(good, cand, t)
        ok = ok & ~good
    return p0 + np.clip(t, 0.0, 1.0)[:, None] * (p1 - p0)


def marching_tetrahedra(
    field,
    lower: np.ndarray,
    upper: np.ndarray,
    spacing: float,
    *,
    quadratic: bool = True,
    refine: bool = True,
) -> IsoSurface:
    """Triangulate the ``field = 0`` iso-surface over an axis-aligned box.

    ``field`` maps an ``(n, 3)`` array of positions to ``n`` values; the
    surface is where it changes sign. Follows 2018 workflow step 3: five
    tetrahedra per cube with the inscribed tetrahedron's two mirror images
    alternating by cube parity, one triangle when a tetrahedron has one corner
    on one side, and two when it has two on each -- the resulting tetragon
    split on its shorter diagonal.
    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    h = float(spacing)
    dims = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)

    # Sample the field on the grid in one pass.
    axes = [lo[d] + h * np.arange(dims[d]) for d in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    values = np.asarray(field(grid_pts), dtype=np.float64).reshape(dims)

    def corner_index(cells: np.ndarray, off: np.ndarray) -> tuple:
        return (cells[:, 0] + off[0], cells[:, 1] + off[1], cells[:, 2] + off[2])

    cells = np.stack(
        np.meshgrid(*[np.arange(dims[d] - 1) for d in range(3)], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    parity = (cells.sum(axis=1) % 2).astype(bool)

    # Corner values and positions, (n_cells, 8)
    cv = np.empty((cells.shape[0], 8), dtype=np.float64)
    cp = np.empty((cells.shape[0], 8, 3), dtype=np.float64)
    for c, off in enumerate(_CUBE_OFFSETS):
        idx = corner_index(cells, off)
        cv[:, c] = values[idx]
        cp[:, c, :] = lo + h * (cells + off)

    verts: list[np.ndarray] = []
    e0s: list[np.ndarray] = []
    e1s: list[np.ndarray] = []
    tris: list[tuple[int, int, int]] = []
    quads: list[int] = []

    def emit(points: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        base = sum(v.shape[0] for v in verts)
        verts.append(points)
        e0s.append(p0)
        e1s.append(p1)
        return np.arange(base, base + points.shape[0])

    for odd, tet_set in ((False, _TETS_EVEN), (True, _TETS_ODD)):
        sel = parity == odd
        if not np.any(sel):
            continue
        sub_v, sub_p = cv[sel], cp[sel]
        for tet in tet_set:
            tv = sub_v[:, tet]                     # (n, 4)
            tp = sub_p[:, tet, :]                  # (n, 4, 3)
            neg = tv < 0.0
            n_neg = neg.sum(axis=1)
            for case in (1, 2, 3):
                mask = n_neg == case
                if not np.any(mask):
                    continue
                _emit_case(
                    tv[mask], tp[mask], neg[mask], case, field, quadratic,
                    emit, tris, quads,
                )

    if not verts:
        return IsoSurface(
            np.zeros((0, 3)), np.zeros((0, 3), dtype=int), h,
            np.zeros((0, 3)), np.zeros((0, 3)),
        )
    vertices = np.concatenate(verts, axis=0)
    triangles = np.array(tris, dtype=int) if tris else np.zeros((0, 3), dtype=int)
    surface = IsoSurface(
        vertices, triangles, h,
        np.concatenate(e0s, axis=0), np.concatenate(e1s, axis=0),
        quad_id=np.array(quads, dtype=int) if quads else np.zeros(0, dtype=int),
    )
    surface = _weld(surface)
    if refine:
        surface = refine_iso_vertices(surface, field)
    return _orient_outward(surface, field)


def _emit_case(tv, tp, neg, case, field, quadratic, emit, tris, quads) -> None:
    """One iso-surface case inside a batch of tetrahedra."""
    n = tv.shape[0]
    # Which of the six edges cross the surface: endpoints of opposite sign.
    cut = []
    for a, b in _TET_EDGES:
        cut.append(neg[:, a] != neg[:, b])
    cut = np.stack(cut, axis=1)                    # (n, 6)
    n_cut = cut.sum(axis=1)

    if case in (1, 3):
        # Three cut edges -> one triangle. (case 3 is case 1 with signs
        # flipped, so the same three-edge geometry applies.)
        assert np.all(n_cut == 3), "one-corner case must cut three edges"
        idx = np.argsort(~cut, axis=1, kind="stable")[:, :3]
    else:
        assert np.all(n_cut == 4), "two-corner case must cut four edges"
        idx = np.argsort(~cut, axis=1, kind="stable")[:, :4]

    pts = np.empty((n, idx.shape[1], 3), dtype=np.float64)
    ep0 = np.empty_like(pts)
    ep1 = np.empty_like(pts)
    rows = np.arange(n)
    for slot in range(idx.shape[1]):
        e = idx[:, slot]
        ea = np.array([_TET_EDGES[k][0] for k in e])
        eb = np.array([_TET_EDGES[k][1] for k in e])
        ep0[:, slot, :] = tp[rows, ea]
        ep1[:, slot, :] = tp[rows, eb]
        pts[:, slot, :] = _edge_root(
            tp[rows, ea], tp[rows, eb], tv[rows, ea], tv[rows, eb],
            field, quadratic,
        )

    if idx.shape[1] == 3:
        ids = emit(
            pts.reshape(-1, 3), ep0.reshape(-1, 3), ep1.reshape(-1, 3)
        ).reshape(n, 3)
        tris.extend(map(tuple, ids))
        quads.extend([-1] * n)
        return

    # Tetragon: order the four points into a simple quadrilateral, then split
    # on the shorter diagonal as the paper prescribes.
    quad, order = _order_quad(pts)
    q0 = np.take_along_axis(ep0, order[:, :, None], axis=1)
    q1 = np.take_along_axis(ep1, order[:, :, None], axis=1)
    ids = emit(
        quad.reshape(-1, 3), q0.reshape(-1, 3), q1.reshape(-1, 3)
    ).reshape(n, 4)
    d02 = np.linalg.norm(quad[:, 0] - quad[:, 2], axis=1)
    d13 = np.linalg.norm(quad[:, 1] - quad[:, 3], axis=1)
    use02 = d02 <= d13
    for i in range(n):
        a, b, c, d = ids[i]
        # The four cut points are freshly emitted for this tetrahedron, so the
        # first of them identifies the quad uniquely, and it survives welding,
        # which renumbers vertices but keeps triangles in order.
        qid = int(a)
        if use02[i]:
            tris.append((a, b, c))
            tris.append((a, c, d))
        else:
            tris.append((a, b, d))
            tris.append((b, c, d))
        quads.extend([qid, qid])


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Sort four coplanar-ish points into a non-self-intersecting ring.

    The four cut points of a two-versus-two tetrahedron come out in edge
    order, which is not necessarily ring order; triangulating them as they
    arrive produces bow-tie pairs whose areas partly cancel. Sorting by angle
    in the plane of their own centroid fixes it.
    """
    centre = pts.mean(axis=1, keepdims=True)
    rel = pts - centre
    normal = np.cross(rel[:, 0], rel[:, 1])
    nrm = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.where(nrm > 1e-300, normal / np.maximum(nrm, 1e-300), 0.0)
    e1 = rel[:, 0]
    e1 = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-300)
    e2 = np.cross(normal, e1)
    ang = np.arctan2(
        np.einsum("nij,nj->ni", rel, e2), np.einsum("nij,nj->ni", rel, e1)
    )
    order = np.argsort(ang, axis=1)
    return np.take_along_axis(pts, order[:, :, None], axis=1), order


def refine_iso_vertices(
    surface: IsoSurface,
    field,
    grad_field=None,
    *,
    max_iter: int = 8,
    tol: float = 1e-13,
) -> IsoSurface:
    """Newton-polish every vertex onto the exact ``field = 0`` iso-surface.

    The paper places a vertex by quadratic interpolation along its cut edge,
    which is cheap but *approximate*: measured on water at ``delta = 0.45 A``
    the quadratic placement leaves ``|PD - 1|`` up to 6.3e-2 (mean 6.9e-3),
    and plain linear interpolation up to 0.48. The surface is therefore not
    quite the iso-surface it is supposed to be.

    That is tolerable for an area, and fatal for an analytic derivative. The
    derivative of an iso-surface point is obtained by differentiating the
    constraint ``field = 0``; if the point does not satisfy the constraint,
    the derivative is of a different point than the one the mesh holds, and
    the resulting gradient is wrong by a percent or so -- smoothly, plausibly,
    and in exactly the way this package has learned to distrust (#546).

    Each vertex has one degree of freedom along its own fixed edge, so a
    scalar Newton iteration converges in two or three steps:

        s <- s - field(p0 + s e) / (grad field . e)

    Vertices are clamped to their own edge, so refinement can never move one
    into a neighbouring cell and break the mesh topology.
    """
    v = surface.vertices
    if v.shape[0] == 0 or surface.edge_p0.size == 0:
        return surface
    p0, p1 = surface.edge_p0, surface.edge_p1
    e = p1 - p0
    e2 = np.einsum("vi,vi->v", e, e)
    ok = e2 > 0.0
    s_par = np.where(
        ok, np.einsum("vi,vi->v", v - p0, e) / np.where(ok, e2, 1.0), 0.0
    )

    for _ in range(int(max_iter)):
        pts = p0 + s_par[:, None] * e
        f_val = np.asarray(field(pts), dtype=np.float64)
        if np.max(np.abs(f_val)) < tol:
            break
        if grad_field is None:
            # Directional derivative along the edge by central difference on
            # the edge parameter; the edge is short, so this is well behaved.
            eps = 1e-6
            fp = np.asarray(field(p0 + (s_par + eps)[:, None] * e), dtype=np.float64)
            fm = np.asarray(field(p0 + (s_par - eps)[:, None] * e), dtype=np.float64)
            slope = (fp - fm) / (2.0 * eps)
        else:
            slope = np.einsum("vi,vi->v", np.asarray(grad_field(pts)), e)
        good = np.abs(slope) > 1e-300
        step = np.where(good, f_val / np.where(good, slope, 1.0), 0.0)
        s_par = np.clip(s_par - step, 0.0, 1.0)

    return IsoSurface(
        p0 + s_par[:, None] * e, surface.triangles, surface.grid_spacing, p0, p1,
        quad_id=surface.quad_id,
    )


def _orient_outward(surface: IsoSurface, field, probe: float = 1e-4) -> IsoSurface:
    """Give every triangle an outward normal.

    Marching tetrahedra emits each triangle in whatever corner order the case
    analysis produced, so windings are mixed. That is invisible in the area
    (a cross-product norm) but fatal for anything signed: the divergence
    theorem volume of a mixed-winding mesh cancels to zero, and per-segment
    normals would point half in and half out.

    The field increases outward across the iso-surface by construction, so a
    triangle is correctly wound when stepping along its normal raises the
    field. One comparison per triangle, and it works for any field rather
    than assuming a star-shaped cavity, which a molecular surface is not.
    """
    if surface.n_triangles == 0:
        return surface
    v = surface.vertices[surface.triangles]
    centre = v.mean(axis=1)
    normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    nrm = np.linalg.norm(normal, axis=1, keepdims=True)
    unit = normal / np.maximum(nrm, 1e-300)
    step = probe * surface.grid_spacing
    outward = np.asarray(field(centre + step * unit), dtype=np.float64) - np.asarray(
        field(centre - step * unit), dtype=np.float64
    )
    flip = outward < 0.0
    tri = surface.triangles.copy()
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return IsoSurface(
        surface.vertices, tri, surface.grid_spacing,
        surface.edge_p0, surface.edge_p1, quad_id=surface.quad_id,
    )


def _weld(surface: IsoSurface, tol: float = 1e-9) -> IsoSurface:
    """Merge coincident vertices so edge sharing becomes detectable.

    Each tetrahedron emits its own copies of the cut points, so before welding
    every edge appears once per triangle and :meth:`IsoSurface.is_closed` can
    never be true. Welding is what turns a triangle soup into a mesh.

    ``np.unique`` already returns exactly the mapping needed: ``first`` picks
    one representative per group in the same order that ``inverse`` indexes
    into, so the triangles remap by a single lookup.
    """
    v = surface.vertices
    if v.shape[0] == 0:
        return surface
    keys = np.round(v / tol).astype(np.int64)
    _, first, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    inverse = np.asarray(inverse).reshape(-1)
    welded = v[first]
    tri = inverse[surface.triangles.ravel()].reshape(-1, 3)
    # Two tetrahedra that share a cut edge produce the same point on the same
    # edge, so keeping one representative's provenance is exact rather than a
    # choice.
    e0 = surface.edge_p0[first] if surface.edge_p0.size else surface.edge_p0
    e1 = surface.edge_p1[first] if surface.edge_p1.size else surface.edge_p1
    # Welding can collapse a sliver triangle onto a line; drop those rather
    # than carry zero-area segments that would divide by zero in sigma = q/s.
    keep = (
        (tri[:, 0] != tri[:, 1])
        & (tri[:, 1] != tri[:, 2])
        & (tri[:, 0] != tri[:, 2])
    )
    quad_id = (
        surface.quad_id[keep]
        if surface.quad_id.size == surface.triangles.shape[0]
        else surface.quad_id
    )
    return IsoSurface(
        welded, tri[keep], surface.grid_spacing, e0, e1, quad_id=quad_id
    )


__all__ += [
    "IsoSurface",
    "marching_tetrahedra",
    "quad_groups",
    "refine_iso_vertices",
]


# ---------------------------------------------------------------------
# Segment assembly (2018 workflow steps 4-6)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AtomCoarsening:
    """Everything one atom's smooth coarsening did, kept for the derivative.

    Stored rather than recomputed. The derivative has to differentiate the
    partition that was actually built, and a second code path that recomputes
    it is the two-copies-of-one-quantity hazard behind #546 -- here it would be
    invisible, because a recomputed partition would agree to the last digit
    right up until someone changed one of the two.

    Attributes
    ----------
    basis_idx : ndarray (n_b,)
        Rows of the global basis arrays this atom owns.
    seg_slice : slice
        Columns of the global segment arrays this atom owns.
    c1 : ndarray (n_dir, 3)
        Pass-1 centres, ``R_a + rho_a u_S``.
    w1 : ndarray (n_b, n_dir)
        Pass-1 Becke weights.
    centres : ndarray (n_dir, 3)
        Pass-2 centres: the pass-1 area-weighted means, with ``c1`` blended in
        at weight ``eps``.
    denom : ndarray (n_dir,)
        ``tot1 + eps``, the denominator those means were formed with.
    w2 : ndarray (n_b, n_dir)
        Pass-2 Becke weights, over *all* directions.
    keep : ndarray (n_dir,) bool
        Which directions survived the area floor.
    eps : float
    """

    basis_idx: np.ndarray
    seg_slice: slice
    c1: np.ndarray
    w1: np.ndarray
    centres: np.ndarray
    denom: np.ndarray
    w2: np.ndarray
    keep: np.ndarray
    eps: float


@dataclass(frozen=True)
class FineCavity:
    """A CFC tessellation, shaped like :class:`~vibeqc.solvation.cavity.CavityTessellation`.

    Field names match the older cavity deliberately, so the reaction-field
    engine, the gradient assembler and the conductor-surface record consume a
    CFC without knowing which construction produced it -- the cavity is a
    method-independent input to all three.

    Attributes
    ----------
    points : ndarray (n_seg, 3), bohr
        Segment representation centres.
    weights : ndarray (n_seg,), bohr^2
        Segment areas. Sum equals the triangulated surface area up to the
        segment-merging threshold.
    point_atom : ndarray (n_seg,), int
        Owning atom, by smallest *relative* distance -- the paper's criterion,
        which is not the same as smallest absolute distance when radii differ.
    normals : ndarray (n_seg, 3)
        Outward unit normals, area-weighted from the triangles.
    atom_positions, atom_radii
        The geometry the cavity was built from.
    surface : IsoSurface
        The underlying triangulation, retained so area, volume and closure
        remain checkable after coarsening.
    """

    points: np.ndarray
    weights: np.ndarray
    point_atom: np.ndarray
    normals: np.ndarray
    atom_positions: np.ndarray
    atom_radii: np.ndarray
    atom_numbers: np.ndarray
    surface: IsoSurface
    grid_spacing: float
    # Which segment each surviving basis point was assigned to, and which
    # surface vertex it came from. Retained because the analytic derivative
    # needs the assignment: a segment's position and area are functions of its
    # basis points, and that mapping is discrete, so it has to be held fixed
    # rather than re-derived. See vibeqc.solvation.fine_cavity_gradient.
    basis_segment: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    # Becke partition weights of basis points over segments, block-diagonal by
    # atom, rows summing to one (#757). This *is* the assignment now:
    # ``basis_segment`` is kept only as the dominant-column diagnostic, because
    # a hard assignment is what made the energy discontinuous.
    basis_weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # Per-basis-point area after the sub-threshold merge, i.e. the a_b that the
    # partition weights multiply.
    basis_area: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Per-atom coarsening records; see :class:`AtomCoarsening`.
    coarsening: tuple = ()
    basis_vertex: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    # Exact area accounting: (surface vertex, segment) pairs. A segment's area
    # is the sum of ``triangle_area/3`` over every vertex paired with it, which
    # includes vertices whose sub-threshold areas were merged into one of its
    # basis points. Positions use ``basis_vertex``; areas use these pairs.
    area_vertex: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    area_segment: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    # The basis point (index into ``basis_vertex``) each grouped vertex's area
    # went to. Segment-level grouping is enough for an area but not for a
    # position, whose weights are per-basis-point.
    area_basis: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    # Step 5 is a partition, not an assignment (#771), so one post-merge basis
    # point supplies a row in more than one atom's block. These three say which:
    # ``basis_source`` is the point a row came from, ``basis_share`` is the
    # step-5 weight that row was scaled by, and ``atom_share`` is the partition
    # itself, kept because the derivative has to differentiate the weights that
    # were actually used rather than a recomputed copy -- the two-copies hazard
    # behind #546.
    #
    # ``atom_share`` holds the coordinates of the frame the cavity was *built*
    # in, not the lab coordinates the arrays above are mapped back to. That is
    # deliberate and is what the derivative needs: ``smooth_coarsening_vjp``
    # runs on the construction-frame record (``to_construction_frame``), so its
    # adjoints and the partition's must be in one frame. The weights themselves
    # are invariants of the rotation, so only the stored points differ.
    basis_source: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    basis_share: np.ndarray = field(default_factory=lambda: np.zeros(0))
    atom_share: object = None
    # Post-merge area per basis *point*, before the step-5 share scales it.
    # ``basis_area`` holds the scaled area per row; this is what the share
    # multiplies, and the derivative needs both.
    merged_area: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # The surface vertex each post-merge basis point sits on. Rows have
    # ``basis_vertex``; the partition is over *points*, so its adjoint comes
    # back per point and needs this to reach the surface.
    merged_vertex: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    # Whether step 6's outward projection found this segment's area-weighted
    # mean inside its atom sphere. Since #770 the projection is eased over a
    # band rather than switched there, so this is a *diagnostic* -- the
    # derivative applies to every segment now, exponentially small for the ones
    # far outside -- and no longer selects a branch.
    projected: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # The molecule-fixed frame this cavity was constructed in, as rows (#769),
    # or empty when it was built in the laboratory frame. Every array above is
    # in *lab* coordinates either way -- the frame is an internal device, not a
    # coordinate system consumers have to know about -- so this is recorded for
    # one reason: a non-empty frame is geometry-dependent, and the nuclear
    # derivative of the construction is not yet derived through it.
    frame: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def total_area(self) -> float:
        return float(np.sum(self.weights))

    @property
    def switching(self) -> np.ndarray:
        """Unity everywhere -- a CFC has no switching function.

        The older cavity fades segments out where spheres overlap; the CFC has
        no overlaps to fade because it is one closed iso-surface. Present so
        consumers written against the Lebedev cavity keep working, and set to
        exactly 1 rather than omitted so nothing silently reads zeros.
        """
        return np.ones(self.n_points, dtype=np.float64)

    @property
    def charge_representation(self) -> str:
        """Gaussian charges. Not a preference -- a requirement (#744).

        Step 6 assigns basis points to each atom's segment grid independently,
        so segments owned by different atoms can land ~0.08 bohr apart near a
        sphere intersection. With the point-charge kernel ``1/r_ij`` the
        off-diagonal then exceeds the bounded diagonal and ``A`` stops being
        positive definite -- measured on water at 10 of 29 grid spacings
        between 0.24 and 0.52 A. That is not slow convergence: Lange & Herbert
        eq. 2.25 make positive definiteness of ``A`` the condition for the
        polarization energy to be a *minimum*, so an indefinite ``A`` means the
        reaction field can raise the energy and the apparent charges solve
        nothing. ``run_cpcm_scf`` reported convergence and a negative
        ``e_solv`` throughout.

        The Lebedev cavity escapes this only through its switching function,
        which the CFC does not have and cannot acquire -- it is one closed
        surface with nothing to switch. The York-Karplus Gaussian kernel
        removes the singularity instead of avoiding it, making ``A`` a Gram
        matrix and its definiteness structural. Measured over the same 29
        spacings: 0 indefinite.
        """
        from .cpcm import GAUSSIAN_CHARGE

        return GAUSSIAN_CHARGE


def projection_radius(dist: np.ndarray, radius: float | np.ndarray):
    """Step 6's eased outward projection: ``(r_eff, d r_eff / d dist)``.

    ``r_eff = rho + w ln(1 + exp((d - rho)/w))`` with ``w`` a fixed fraction of
    the sphere radius (:data:`PROJECTION_SOFTNESS`), so the derivative is the
    logistic of the same argument. Never *less* than ``rho``, so a segment is
    never placed inside the sphere it screens -- deep inside, the softplus
    underflows and the result is ``rho`` exactly, which is where the hard rule
    puts it too. Equal to ``d`` far outside, and the paper's hard rule in the
    limit of a vanishing band.

    One function for both directions on purpose. The forward pass needs
    ``r_eff`` and the reverse pass needs both it and its slope, and writing the
    softplus twice is the two-copies-of-one-quantity hazard behind #546 -- here
    it would be invisible, since the two would agree until someone changed one.

    Evaluated in the numerically stable form ``max(x, 0) + w log1p(exp(-|x|/w))``
    so a segment far outside its sphere cannot overflow the exponential.
    """
    d = np.asarray(dist, dtype=np.float64)
    rho = np.asarray(radius, dtype=np.float64)
    soft = PROJECTION_SOFTNESS * rho
    excess = d - rho
    r_eff = rho + np.maximum(excess, 0.0) + soft * np.log1p(
        np.exp(-np.abs(excess) / soft)
    )
    return r_eff, 0.5 * (1.0 + np.tanh(0.5 * excess / soft))


def _relative_distance(points: np.ndarray, positions: np.ndarray,
                       radii: np.ndarray) -> np.ndarray:
    """``tau`` for every (point, atom) pair -- the paper's assignment metric."""
    d = np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2)
    return d / radii[None, :]


def quad_groups(surface: "IsoSurface"):
    """Split a triangulation into lone triangles and tetragon pairs (#779).

    Returns ``(lone, pairs, corners)``: a boolean mask over triangles, an
    ``(n_quads, 2)`` array of the triangle rows forming each tetragon, and an
    ``(n_quads, 4)`` array of that tetragon's distinct corner vertices.

    Step 4 needs this because the diagonal a tetragon is split on is a hard
    ``argmin`` over its two diagonals, so it flips where they are equal. The
    quad's total area survives that flip -- it is the same four points either
    way -- but "a third of each triangle to each of its corners" does not, and
    each corner's basis area jumps by about 0.023 bohr^2 where it happens.
    Grouping the pair lets the area be divided over the four corners instead,
    which no diagonal can change.

    A tetragon whose two triangles do not share exactly one diagonal is
    returned as two lone triangles rather than silently mis-grouped. That
    happens when welding collapses one half onto a line and drops it, which is
    a real degeneracy and not something to paper over: one surviving triangle
    is genuinely all the area there is.
    """
    n_tri = surface.triangles.shape[0]
    empty2 = np.zeros((0, 2), dtype=int)
    empty4 = np.zeros((0, 4), dtype=int)
    qid = np.asarray(surface.quad_id, dtype=int)
    if qid.size != n_tri or n_tri == 0:
        return np.ones(n_tri, dtype=bool), empty2, empty4

    lone = qid < 0
    paired = np.flatnonzero(~lone)
    if paired.size == 0:
        return lone, empty2, empty4

    order = paired[np.argsort(qid[paired], kind="stable")]
    ids = qid[order]
    starts = np.flatnonzero(np.concatenate(([True], ids[1:] != ids[:-1])))
    sizes = np.diff(np.concatenate((starts, [ids.size])))
    full = sizes == 2
    lone = lone.copy()
    for s, n in zip(starts[~full], sizes[~full]):
        lone[order[s:s + n]] = True
    if not np.any(full):
        return lone, empty2, empty4

    pairs = np.stack([order[starts[full]], order[starts[full] + 1]], axis=1)
    six = np.sort(surface.triangles[pairs].reshape(-1, 6), axis=1)
    fresh = np.ones(six.shape, dtype=bool)
    fresh[:, 1:] = six[:, 1:] != six[:, :-1]
    good = fresh.sum(axis=1) == 4
    if not np.all(good):
        for row in pairs[~good]:
            lone[row] = True
        pairs = pairs[good]
        six, fresh = six[good], fresh[good]
    if pairs.shape[0] == 0:
        return lone, empty2, empty4
    return lone, pairs, six[fresh].reshape(-1, 4)


def _merge_tiny(points, areas, normals, source, min_area):
    """Join sub-threshold basis points into their nearest neighbour.

    2018 step 5: "Basis grid points with an area smaller than a threshold area
    (1*10^-3 bohr^2) are joined with the nearest neighbor basis grid point."
    Area is conserved by construction -- these points carry real surface, so
    discarding them would shrink the cavity instead of coarsening it.
    """
    keep = areas >= min_area
    n = areas.size
    if np.all(keep) or not np.any(keep):
        ident = np.arange(n)
        return points, areas, normals, source, source, ident
    tiny = ~keep
    tree_pts = points[keep]
    d = np.linalg.norm(points[tiny][:, None, :] - tree_pts[None, :, :], axis=2)
    host = np.argmin(d, axis=1)
    new_areas = areas[keep].copy()
    np.add.at(new_areas, host, areas[tiny])
    # Area accounting for the derivative: a kept point's area is its own plus
    # every absorbed point's, so its *derivative* is the same sum. Record the
    # grouping as (source vertex, kept index) pairs rather than assuming one
    # vertex per basis point -- assuming that made segment areas reconstruct
    # to only 5e-6 relative, which is a real inconsistency between the energy
    # and its gradient, not a rounding error.
    kept_idx = np.flatnonzero(keep)
    group_vertex = np.concatenate([source[keep], source[tiny]])
    group_basis = np.concatenate([np.arange(kept_idx.size), host])
    return (
        tree_pts, new_areas, normals[keep], source[keep],
        group_vertex, group_basis,
    )


def build_fine_cavity(
    atom_positions_bohr: np.ndarray,
    atom_numbers,
    atom_radii_bohr: np.ndarray,
    *,
    grid_spacing_ang: float = GRID_SPACING_ANG,
    n_segments_h: int = NSPH_FINE,
    n_segments_other: int = NSPA_FINE,
    pd_params: PseudoDensityParams | None = None,
    quadratic_edges: bool = True,
    min_basis_area_bohr2: float = MIN_BASIS_AREA_BOHR2,
    frame: str = "molecular",
) -> FineCavity:
    """Build the COSMO FINE Cavity of Klamt & Diedenhofen 2018.

    Follows the paper's workflow: triangulate the ``PD = 1`` iso-surface by
    marching tetrahedra (steps 2-3), turn triangles into area-carrying basis
    points (step 4), assign them to atoms by smallest relative distance and
    merge sub-threshold points (step 5), then coarsen onto a per-atom geodesic
    segment grid with a second assignment pass (step 6).

    Radii are supplied by the caller rather than looked up. A COSMO-RS
    parameterization is fitted against a *specific* radius set, so silently
    defaulting them here would be the cross-protocol transfer that
    :mod:`vibeqc.solvation.cosmors.parameters` exists to make visible.

    ``frame`` selects the coordinate system the workflow is carried out in.
    ``"molecular"`` (the default) performs step 1 -- see
    :func:`molecular_frame` -- so the cavity rotates rigidly with the solute.
    ``"lab"`` skips step 1 and lays the marching lattice and the segment
    directions out along the laboratory axes, which makes the discretized
    cavity depend on the solute's orientation (#769). The returned record is in
    lab coordinates either way.

    ``"lab"`` remains available and is the right choice for one case: a
    **linear** solute, whose frame determines no y-axis and so has no nuclear
    derivative at all, which
    :func:`molecular_frame_jacobian` refuses rather than approximates.
    """
    pos = np.asarray(atom_positions_bohr, dtype=np.float64)
    z = np.asarray(list(atom_numbers), dtype=int)
    rad = np.asarray(atom_radii_bohr, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"build_fine_cavity: positions must be (n, 3), got {pos.shape}")
    if rad.shape != (pos.shape[0],) or z.shape != (pos.shape[0],):
        raise ValueError(
            "build_fine_cavity: atom_numbers and atom_radii_bohr must have one "
            "entry per atom."
        )

    kind = str(frame).strip().lower()
    if kind not in {"lab", "molecular"}:
        raise ValueError(
            f"build_fine_cavity: frame must be 'lab' or 'molecular', got {frame!r}."
        )
    # Step 1. Everything below is written against ``pos``, so putting the
    # solute into its own frame here is enough to carry the whole workflow into
    # it, and the record is rotated back at the end. Doing it this way rather
    # than by turning the lattice and the direction grids keeps one
    # construction instead of two.
    rot = np.zeros((0, 0))
    origin = np.zeros(3)
    if kind == "molecular":
        rot = molecular_frame(pos)
        origin = pos.mean(axis=0)
        pos = (pos - origin) @ rot.T

    h = float(grid_spacing_ang) * ANG_TO_BOHR

    def field(p: np.ndarray) -> np.ndarray:
        return log_pseudo_density(p, pos, rad, pd_params)

    # Step 2: the box must completely include all spheres of radius R_alpha.
    # The bond term pushes the iso-surface outside them, so pad generously;
    # a clipped surface would not be closed and the volume would be wrong.
    pad = 1.8 * float(np.max(rad)) + 2.0 * h
    # Anchor the box to the centroid rather than to ``pos.min(axis=0)``.
    # Anchoring to an extremum makes the grid -- and so the discretized
    # surface -- follow whichever atom happens to be outermost on each axis,
    # which is non-differentiable wherever two atoms tie (water has all three
    # tied at x = 0) and loads the whole grid-drag onto that one atom: 2.3% of
    # its area gradient at 0.4 A spacing. The centroid is smooth and shares the
    # drag equally, so ``d(grid)/dR_A = I / n_atoms`` for every atom, and a
    # uniform translation of the molecule translates the surface rigidly.
    #
    # Rounding the half-extent up to a whole number of cells keeps every grid
    # point at a fixed offset from the centroid, so a change in the cell count
    # only grows the box into empty space outside the surface instead of
    # shifting the points that carry the surface.
    n_at = pos.shape[0]
    centroid = pos.mean(axis=0)
    half = h * np.ceil((np.abs(pos - centroid).max(axis=0) + pad) / h)
    surface = replace(
        marching_tetrahedra(
            field, centroid - half, centroid + half, h,
            quadratic=quadratic_edges,
        ),
        grid_anchor=np.full((n_at, 3), 1.0 / n_at),
        frame=rot,
    )
    if surface.n_triangles == 0:
        raise RuntimeError(
            "build_fine_cavity: the pseudo-density iso-surface is empty; the "
            "radii or grid spacing are inconsistent with the geometry."
        )

    # Step 4: "the area is calculated and equally assigned to the corner
    # points". A lone triangle gives a third to each of its three; a tetragon
    # gives a quarter to each of its four (#779).
    #
    # The paper says triangles, but its own step 3 produces the tetragons, and
    # which two triangles a tetragon becomes is decided by a hard comparison of
    # its diagonals. Dividing per triangle therefore makes each corner's area
    # jump where that comparison flips, by about 0.023 bohr^2, even though the
    # quad's total area is unchanged and the surface area moves by 1.8e-07.
    # Dividing over the tetragon is the same principle applied to the figure
    # the construction actually produced, and no diagonal can change it.
    tri = surface.triangles
    tri_area = surface.triangle_areas()
    v = surface.vertices
    tri_normal = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    basis_area = np.zeros(v.shape[0], dtype=np.float64)
    basis_normal = np.zeros_like(v)

    lone, quad_pairs, quad_corners = quad_groups(surface)
    if np.any(lone):
        np.add.at(basis_area, tri[lone].ravel(), np.repeat(tri_area[lone] / 3.0, 3))
        np.add.at(
            basis_normal, tri[lone].ravel(), np.repeat(tri_normal[lone], 3, axis=0)
        )
    if quad_pairs.shape[0]:
        np.add.at(
            basis_area, quad_corners.ravel(),
            np.repeat(tri_area[quad_pairs].sum(axis=1) / 4.0, 4),
        )
        np.add.at(
            basis_normal, quad_corners.ravel(),
            np.repeat(tri_normal[quad_pairs].sum(axis=1), 4, axis=0),
        )

    used = np.flatnonzero(basis_area > 0.0)
    b_pts, b_area, b_nrm = v[used], basis_area[used], basis_normal[used]
    b_pts, b_area, b_nrm, b_src, grp_vert, grp_basis = _merge_tiny(
        b_pts, b_area, b_nrm, used, min_basis_area_bohr2
    )

    # Step 5: smooth cells over atoms in place of the paper's argmin (#771).
    #
    # A hard assignment by smallest relative distance moves a basis point's
    # whole area to another atom at an infinitesimal displacement: 0.19 bohr^2
    # at once, a 6.9 percent error in the local slope, a 2.9e-08 Ha step in the
    # energy. The cells here are the same rule smoothed over the marching
    # spacing -- see vibeqc.solvation.atom_partition for why that coordinate and
    # not Becke's -- so a point near a boundary feeds both atoms' segment grids
    # in proportion, and the total area is conserved exactly because the rows
    # sum to one.
    share = atom_partition(b_pts, pos, rad, h)

    # Step 6: coarsen onto a per-atom geodesic grid, with a second assignment
    # pass from the resulting segment centres.
    seg_pts: list[np.ndarray] = []
    seg_area: list[np.ndarray] = []
    seg_nrm: list[np.ndarray] = []
    seg_atom: list[np.ndarray] = []
    bas_seg: list[np.ndarray] = []
    bas_vert: list[np.ndarray] = []
    n_atoms = pos.shape[0]
    kept_global = np.full((b_pts.shape[0], n_atoms), -1, dtype=int)
    basis_global = np.full((b_pts.shape[0], n_atoms), -1, dtype=int)
    seg_proj: list[np.ndarray] = []
    bas_w: list[tuple[np.ndarray, np.ndarray]] = []
    bas_src_global: list[np.ndarray] = []
    bas_share: list[np.ndarray] = []
    seg_base = 0
    basis_base = 0
    for ia in range(n_atoms):
        # Compactly supported cells, so this really is a subset: 1.8 atoms per
        # basis point at 0.40 A, 1.37 at 0.20 A, against 1.0 for the old hard
        # assignment. The rows overlap now, which is why the bookkeeping below
        # is indexed by (basis point, atom) rather than by basis point.
        sel = share.weights[:, ia] > 0.0
        if not np.any(sel):
            continue
        n_seg = n_segments_h if z[ia] == 1 else n_segments_other
        directions = sphere_points_for_count(n_seg)
        # NB: not ``owner`` -- that name holds the atom assignment this loop
        # iterates over, and shadowing it silently reuses the previous atom's
        # segment map on the next pass.
        p, a, nv, seg_of_basis, src, was_in, detail = _coarsen_atom(
            b_pts[sel], b_area[sel] * share.weights[sel, ia], b_nrm[sel],
            pos[ia], rad[ia], directions, source=b_src[sel],
        )
        if p.shape[0] == 0:
            continue
        seg_pts.append(p)
        seg_area.append(a)
        seg_nrm.append(nv)
        seg_atom.append(np.full(p.shape[0], ia, dtype=int))
        bas_seg.append(seg_of_basis + seg_base)
        bas_vert.append(src)
        seg_proj.append(was_in)
        # kept-basis (global) -> segment, for the area grouping below.
        sel_idx = np.flatnonzero(sel)
        bas_w.append((sel_idx, detail))
        bas_src_global.append(sel_idx)
        bas_share.append(share.weights[sel_idx, ia])
        kept_global[sel_idx, ia] = seg_of_basis + seg_base
        basis_global[sel_idx, ia] = np.arange(sel_idx.size) + basis_base
        seg_base += p.shape[0]
        basis_base += sel_idx.size

    if not seg_pts:
        raise RuntimeError("build_fine_cavity: no segments survived coarsening.")

    # Refuse duplicate segments rather than returning a singular A matrix.
    #
    # Two segments land on the same point when both of their Becke cells are
    # dominated by the *same* basis point, which happens when the basis grid
    # does not resolve the segment grid -- at 0.45 A on water some cells hold
    # a single basis point. Coincident segments make A rank-deficient (it is
    # the Gram matrix of their Gaussians, and two identical Gaussians are one
    # vector), so the apparent-charge solve has a null direction and the
    # screening is undefined along it. Measured: minimum eigenvalue -1.0e-14
    # against a matrix norm of 6.6e+03, i.e. exactly zero.
    #
    # Merging them is not the fix. A Gaussian of area ``a`` is wider than two
    # of area ``a/2``, so its self-energy differs by sqrt(2); merging as
    # segments approach coincidence would trade this for a discontinuity of
    # the kind #757 removed. The real requirement is resolution, so say so.
    all_pts = np.concatenate(seg_pts, axis=0)
    if all_pts.shape[0] > 1:
        gap = np.linalg.norm(all_pts[:, None, :] - all_pts[None, :, :], axis=2)
        np.fill_diagonal(gap, np.inf)
        worst = float(gap.min())
        if worst < 1e-8:
            raise RuntimeError(
                f"build_fine_cavity: two segments coincide to {worst:.2e} bohr, "
                f"so the A matrix would be singular. The marching grid "
                f"({grid_spacing_ang} A) is too coarse to resolve the "
                f"{n_segments_other}/{n_segments_h}-direction segment grid: some "
                f"segments are supported by a single basis point. Use a finer "
                f"grid_spacing_ang (the 2018 paper's value is 0.3 A) or fewer "
                f"segments per atom."
            )

    # Global (kept basis point) x (segment) weight matrix, block-diagonal by
    # atom: a basis point is partitioned only over its own atom's segments,
    # since the atom assignment upstream is what decides the sphere a segment
    # is projected onto.
    n_basis_tot = int(sum(d.w2.shape[0] for _, d in bas_w))
    n_seg_tot = int(sum(pp.shape[0] for pp in seg_pts))
    basis_w = np.zeros((n_basis_tot, n_seg_tot))
    row0 = 0
    col0 = 0
    records: list[AtomCoarsening] = []
    for (idx, detail), pp in zip(bas_w, seg_pts):
        kept = detail.w2[:, detail.keep]
        basis_w[row0:row0 + kept.shape[0], col0:col0 + pp.shape[0]] = kept
        records.append(AtomCoarsening(
            basis_idx=np.arange(row0, row0 + kept.shape[0]),
            seg_slice=slice(col0, col0 + pp.shape[0]),
            c1=detail.c1, w1=detail.w1, centres=detail.centres,
            denom=detail.denom, w2=detail.w2, keep=detail.keep,
            eps=detail.eps,
        ))
        row0 += kept.shape[0]
        col0 += pp.shape[0]
    # ``b_area`` carries the post-merge per-basis-point area; each row of the
    # weight blocks holds that area times the atom's step-5 share, which is what
    # the coarsening above consumed and so what its chain rule must
    # differentiate. The share and the row's source point are kept beside it,
    # because a basis point now appears in more than one block.
    basis_a = np.concatenate([
        b_area[idx] * w for (idx, _), w in zip(bas_w, bas_share)
    ])
    basis_src = np.concatenate(bas_src_global) if bas_src_global else np.zeros(
        0, dtype=int
    )
    basis_sh = np.concatenate(bas_share) if bas_share else np.zeros(0)

    # Exact area accounting, per (vertex, atom): a vertex whose sub-threshold
    # area was merged into basis point ``b`` contributes to every block ``b``
    # appears in, scaled by that block's share.
    area_v: list[np.ndarray] = []
    area_s: list[np.ndarray] = []
    area_b: list[np.ndarray] = []
    for ia in range(n_atoms):
        seg_of = kept_global[grp_basis, ia]
        live = seg_of >= 0
        if not np.any(live):
            continue
        area_v.append(grp_vert[live])
        area_s.append(seg_of[live])
        area_b.append(basis_global[grp_basis, ia][live])
    empty = np.zeros(0, dtype=int)

    cavity = FineCavity(
        points=np.concatenate(seg_pts, axis=0),
        weights=np.concatenate(seg_area, axis=0),
        point_atom=np.concatenate(seg_atom, axis=0),
        normals=np.concatenate(seg_nrm, axis=0),
        atom_positions=pos,
        atom_radii=rad,
        atom_numbers=z,
        surface=surface,
        grid_spacing=h,
        basis_segment=np.concatenate(bas_seg) if bas_seg else np.zeros(0, dtype=int),
        basis_weights=basis_w,
        basis_area=basis_a,
        coarsening=tuple(records),
        basis_vertex=np.concatenate(bas_vert) if bas_vert else np.zeros(0, dtype=int),
        basis_source=basis_src,
        basis_share=basis_sh,
        atom_share=share,
        merged_area=b_area,
        merged_vertex=b_src,
        area_vertex=np.concatenate(area_v) if area_v else empty,
        area_segment=np.concatenate(area_s) if area_s else empty,
        area_basis=np.concatenate(area_b) if area_b else empty,
        projected=(
            np.concatenate(seg_proj) if seg_proj else np.zeros(0, dtype=bool)
        ),
        frame=rot,
    )
    return cavity if rot.size == 0 else _to_lab_frame(cavity, rot, origin)


def _to_lab_frame(
    cavity: FineCavity, rot: np.ndarray, origin: np.ndarray
) -> FineCavity:
    """Map a cavity built in the molecule-fixed frame back to lab coordinates.

    The frame is a device for laying out the lattice and the segment
    directions, not a coordinate system the rest of the code should have to
    know about: the A matrix, the electrostatic potential, the conductor-surface
    record and COSMO-RS all work in the same coordinates as the wavefunction.
    So the whole record comes back, and ``FineCavity.frame`` is the only trace
    left.

    Positions map as ``x @ rot + origin`` and directions as ``x @ rot``; areas,
    partition weights, indices and the projection flags are invariants of a
    rotation and are carried over untouched.
    """
    def as_point(x: np.ndarray) -> np.ndarray:
        return np.asarray(x) @ rot + origin

    def as_direction(x: np.ndarray) -> np.ndarray:
        return np.asarray(x) @ rot

    surface = replace(
        cavity.surface,
        vertices=as_point(cavity.surface.vertices),
        edge_p0=as_point(cavity.surface.edge_p0),
        edge_p1=as_point(cavity.surface.edge_p1),
    )
    return replace(
        cavity,
        points=as_point(cavity.points),
        normals=as_direction(cavity.normals),
        atom_positions=as_point(cavity.atom_positions),
        surface=surface,
        coarsening=tuple(
            replace(rec, c1=as_point(rec.c1), centres=as_point(rec.centres))
            for rec in cavity.coarsening
        ),
    )


def to_construction_frame(cavity: FineCavity) -> FineCavity:
    """The same cavity in the frame it was *built* in, with ``frame`` cleared.

    The exact inverse of the map :func:`build_fine_cavity` applies on its way
    out, so this recovers the record the construction actually produced -- to
    rounding, which a test pins.

    It exists for the derivative. Every stage of
    :mod:`vibeqc.solvation.fine_cavity_gradient` is valid *in the construction
    frame* and only there: that is where the marching lattice is fixed, where a
    vertex has one degree of freedom along an edge that does not turn, and where
    the pass-1 segment centres ride rigidly on their atom. Reversing into this
    frame is therefore not a convenience -- it is what makes the existing chain
    rule applicable, and it is why
    :class:`~vibeqc.solvation.cavity_derivative.FrameCavityDerivative` needs no
    second copy of that chain.

    ``frame`` is cleared because the returned record describes a lab-frame
    construction as far as any consumer can tell, and leaving it set would trip
    the refusals that exist to stop exactly this chain from being applied in the
    wrong coordinates.

    A cavity built with ``frame="lab"`` is returned unchanged.
    """
    rot = np.asarray(cavity.frame, dtype=np.float64)
    if rot.size == 0:
        return cavity
    origin = cavity.atom_positions.mean(axis=0)

    def as_point(x: np.ndarray) -> np.ndarray:
        return (np.asarray(x) - origin) @ rot.T

    def as_direction(x: np.ndarray) -> np.ndarray:
        return np.asarray(x) @ rot.T

    empty = np.zeros((0, 0))
    surface = replace(
        cavity.surface,
        vertices=as_point(cavity.surface.vertices),
        edge_p0=as_point(cavity.surface.edge_p0),
        edge_p1=as_point(cavity.surface.edge_p1),
        frame=empty,
    )
    return replace(
        cavity,
        points=as_point(cavity.points),
        normals=as_direction(cavity.normals),
        atom_positions=as_point(cavity.atom_positions),
        surface=surface,
        coarsening=tuple(
            replace(rec, c1=as_point(rec.c1), centres=as_point(rec.centres))
            for rec in cavity.coarsening
        ),
        frame=empty,
    )


@dataclass(frozen=True)
class _CoarsenDetail:
    c1: np.ndarray
    w1: np.ndarray
    centres: np.ndarray
    denom: np.ndarray
    w2: np.ndarray
    keep: np.ndarray
    eps: float


def _coarsen_atom(pts, areas, normals, centre, radius, directions, source=None):
    """Group one atom's basis points onto its geodesic segment directions.

    Two assignment passes, as the paper specifies: basis points are first
    binned by nearest geodesic direction, the segment centre is recomputed as
    the area-weighted mean of its members, and the points are reassigned from
    those centres. The second pass matters because the first bins against
    idealised sphere directions while the actual basis points sit on the
    iso-surface, which is not that sphere.
    """
    from .segment_partition import smooth_partition

    n_dir = directions.shape[0]
    # Pass 1: Becke weights over the idealised sphere directions. These centres
    # ride rigidly on the atom, so this pass carries no geometry of its own.
    #
    # Both passes use *Euclidean* distance to the segment centres, which is the
    # paper's own criterion ("associated with their closest COSMO grid point",
    # step 3 of the COC construction). An earlier revision binned pass 1 by
    # angle instead; unifying the metric means one partition primitive and one
    # derivative rather than two of each.
    c1 = centre[None, :] + radius * directions
    W1 = smooth_partition(pts, c1).weights                     # (n_pts, n_dir)
    aw1 = W1 * areas[:, None]
    tot1 = aw1.sum(axis=0)

    # Pass-2 centres for EVERY direction, with the idealised sphere point
    # blended in at weight ``eps``.
    #
    # The obvious alternative -- keep only directions with ``tot1 > 0`` -- is
    # what a hard assignment could afford and a smooth one cannot. Becke
    # weights for a direction pointing into a bond underflow to exactly zero,
    # so ``tot1 > 0`` flickers as atoms move; and because these are the
    # *centres* of the second partition, one flicker changes every weight at
    # once. Measured with that pruning in place, the total area stayed smooth
    # (it is protected by ``sum_S W = 1``) while the per-segment distribution
    # swung by 0.3 in ``sum_S w_S^2`` and the energy slope by +-0.12 -- far
    # worse than the 6e-7 Ha jumps this was meant to remove.
    #
    # ``eps`` is a prior on the centre, not a model parameter: it pulls a
    # direction's centre toward its idealised sphere point with a fixed weight,
    # so a direction with real surface near it is essentially unaffected and a
    # direction with none sits on the sphere, which is where that segment
    # belongs. It cannot flicker because it is never compared against anything.
    #
    # Its *size* is set by the derivative, not by the value. The sensitivity
    # d(centre)/d(weight) goes as 1/(tot1 + eps), so a vanishing eps makes the
    # centres of surface-free directions infinitely sensitive to weights that
    # are identically zero -- mathematically consistent, numerically fatal. At
    # eps = 1e-12 rho^2 the reverse pass amplified partition rounding into an
    # adjoint of 10.0 against real contributions of 1.4e-05, and the assembled
    # gradient sat on a flat 2.3e-03 plateau against finite differences.
    #
    # A fraction of the mean segment area is the natural scale, and 1e-5 of it
    # puts both errors far below anything that matters: the centre of a live
    # direction moves by about 1e-5 of its distance to the sphere point (of
    # order 1e-5 bohr), while the amplification of the partition's own rounding
    # stays around 1e-8, three orders below the smallest real term. Scaled by
    # the radius, which is a constant, so eps carries no derivative of its own.
    CENTRE_PRIOR = 1e-5
    eps = CENTRE_PRIOR * 4.0 * np.pi * float(radius) ** 2 / directions.shape[0]
    live_idx = np.arange(directions.shape[0])
    centres = (aw1.T @ pts + eps * c1) / (tot1 + eps)[:, None]

    # Pass 2: Becke weights over the recomputed centres. The paper requires
    # this second pass because "the assembly of the basis grid points has
    # changed the centers of the segments"; it is not decoration -- dropping it
    # reassigns about 12 percent of basis points and moves segment areas by up
    # to 1.0 bohr^2.
    W2 = smooth_partition(pts, centres).weights                # (n_pts, n_live)
    aw2 = W2 * areas[:, None]

    out_area = np.zeros(n_dir)
    out_area[live_idx] = aw2.sum(axis=0)
    out_pos = np.zeros((n_dir, 3))
    out_pos[live_idx] = aw2.T @ pts
    out_nrm = np.zeros((n_dir, 3))
    out_nrm[live_idx] = W2.T @ normals

    keep = out_area > MIN_SEGMENT_AREA_BOHR2
    p = out_pos[keep] / out_area[keep, None]
    a = out_area[keep]
    nv = out_nrm[keep]
    # ``basis_segment`` is now only a diagnostic: the dominant column of the
    # partition, i.e. the segment a hard assignment *would* have chosen. The
    # assignment itself is the weight matrix. Kept because callers and the
    # conductor-surface record read it, and because "which segment does this
    # basis point mostly belong to" is still a useful question.
    kept_cols_local = np.flatnonzero(keep[live_idx])
    if kept_cols_local.size:
        local_owner = kept_cols_local[np.argmax(W2[:, kept_cols_local], axis=1)]
    else:
        local_owner = np.zeros(pts.shape[0], dtype=int)
    nv = nv / np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-300)

    # "Segment centers with atom center distances smaller than the COSMO
    # radius of the atom are moved outward to the sphere." An area-weighted
    # mean of points on a curved surface always falls inside it, so without
    # this every segment would sit slightly under the cavity -- and a charge
    # placed inside the surface it is meant to screen is a physical error, not
    # a rounding one.
    rel_p = p - centre
    dist = np.linalg.norm(rel_p, axis=1)
    inside = dist < radius
    # Eased over a band rather than switched at the sphere (#770): the hard
    # maximum is C0 but not C1, so the segment position's slope jumps where a
    # centre crosses. See PROJECTION_SOFTNESS for why the band is this wide and
    # why a softplus rather than a spliced polynomial.
    r_eff, _ = projection_radius(dist, float(radius))
    scale = np.where(dist > 1e-300, r_eff / np.maximum(dist, 1e-300), 1.0)
    p = centre + rel_p * scale[:, None]
    src = np.zeros(0, dtype=int) if source is None else np.asarray(source, dtype=int)
    # The partition weights of the surviving segments. Rows sum to slightly
    # under one -- the sub-threshold columns are dropped, not redistributed --
    # which is a measured 1.3e-10 of the total area on water and is the same
    # order as the merge threshold. Recorded rather than recomputed by the
    # derivative, which would be the two-copies-of-one-quantity hazard of #546.
    return p, a, nv, local_owner, src, inside, _CoarsenDetail(
        c1=c1, w1=W1, centres=centres, denom=tot1 + eps, w2=W2, keep=keep,
        eps=eps,
    )


def _weighted_centres(pts, areas, assign, n_dir, centre, radius):
    """Area-weighted mean position per bin; NaN rows for empty bins."""
    tot = np.zeros(n_dir)
    np.add.at(tot, assign, areas)
    acc = np.zeros((n_dir, 3))
    np.add.at(acc, assign, pts * areas[:, None])
    out = np.full((n_dir, 3), np.nan)
    live = tot > 0.0
    out[live] = acc[live] / tot[live, None]
    return out


__all__ += [
    "FineCavity",
    "build_fine_cavity",
    "projection_radius",
    "to_construction_frame",
]
