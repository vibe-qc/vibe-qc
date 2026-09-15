"""Analytic nuclear derivatives of the COSMO FINE cavity.

Klamt & Diedenhofen 2018 (doi:10.1002/jcc.25342) state the possibility but not
the algebra: "Since the pseudo-density PD is an analytical function of the atom
positions, the gradient of each of the intersection points with respect to
each atom position can be calculated, and thus analytic gradients of the
segment positions and areas, which are composed from the triangles with edge
points, are available." This module is that derivation.

The chain
---------
A CFC segment's position and area are not simple functions of one atom. They
are built up as::

    PD(r; R)                      the pseudo-density, eq. 8
      -> t_k                      iso-surface points, on fixed grid edges
      -> triangle areas
      -> basis-point areas        each triangle's area split over its corners
      -> segment position, area   area-weighted mean and sum

so the derivative has to be carried through every stage. Each stage below is
verified against finite differences independently, because a mistake in one
stage produces a smooth, plausible, wrong total -- the failure mode of #546.

The one genuinely non-obvious step
----------------------------------
An iso-surface point lies on a **fixed** grid edge: the marching grid does not
move with the atoms. So the point has only one degree of freedom, the edge
parameter ``s``, and differentiating the constraint ``F(p0 + s e; R) = 0``
gives

    ds/dR_A = - grad_{R_A} PD / (grad_t PD . e)

The factor ``PD`` cancels between ``grad F = grad PD / PD`` in numerator and
denominator, so this holds whether the field is ``PD`` or ``ln PD``. The
resulting Jacobian ``dt/dR_A = e (ds/dR_A)^T`` is **rank one**: when an atom
moves, an iso-surface point can only slide along its own edge. That is a
strong structural property and the tests assert it.

Validity
--------
Segment *assignment* is discrete and changes as atoms move, so these are
derivatives at fixed segment topology -- the same caveat the Lebedev cavity
carries through its ``drop_threshold``, and the reason the finite-difference
tests validate topology at every displacement.

The derivatives must be evaluated with the *same* ``PseudoDensityParams`` that
built the cavity, including the five-nearest-atom truncation. That truncation
makes ``PD`` only piecewise smooth, so a gradient taken with the full sum
against a surface built with the truncation would differentiate a different
function -- which is precisely how #546 happened one level up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fine_cavity import (
    FineCavity,
    IsoSurface,
    PseudoDensityParams,
    projection_radius,
    quad_groups,
)


@dataclass(frozen=True)
class PseudoDensityDerivatives:
    """Value and first derivatives of :func:`pseudo_density` at points.

    Attributes
    ----------
    value : ndarray (n_pts,)
    d_point : ndarray (n_pts, 3)
        ``grad_r PD`` -- derivative with respect to the field point.
    d_atom : ndarray (n_pts, n_atoms, 3)
        ``grad_{R_A} PD`` -- derivative with respect to each atom position.
    """

    value: np.ndarray
    d_point: np.ndarray
    d_atom: np.ndarray

    def translation_residual(self) -> float:
        """``max |sum_A grad_{R_A} PD + grad_r PD|``.

        Exactly zero for the true derivatives, and it needs no finite
        differences to check: ``PD`` depends on the atoms only through the
        differences ``r - R_alpha``, so translating the whole system while
        holding ``r`` fixed is the same as translating ``r`` the other way.
        Any term whose two derivative branches disagree shows up here, which
        makes it a sharper test than an FD comparison at the same cost.
        """
        return float(
            np.max(np.abs(np.sum(self.d_atom, axis=1) + self.d_point))
        )


def pseudo_density_derivatives(
    points: np.ndarray,
    atom_positions: np.ndarray,
    atom_radii: np.ndarray,
    params: PseudoDensityParams | None = None,
) -> PseudoDensityDerivatives:
    """Analytic first derivatives of eq. 8.

    With ``tau_a = (r - R_a)/rho_a`` (vector) and ``tau_a = |tau_a|``
    (scalar), writing ``A_a = exp{a1 (tau_a - 1)}`` for the atomic term and
    ``P_ab = c (1 - D)^m exp{a2 (tau_a + tau_b - 2)}``, ``D = tau_a . tau_b``,
    for the pair term::

        dA_a/dr     =  a1 A_a  tauhat_a / rho_a
        dA_a/dR_a   = -a1 A_a  tauhat_a / rho_a

        dD/dr       =  tau_b/rho_a + tau_a/rho_b
        dD/dR_a     = -tau_b/rho_a
        dP/dD       = -c m (1 - D)^(m-1) exp{a2 (tau_a + tau_b - 2)}
        dP_ab/dr    =  dP/dD (tau_b/rho_a + tau_a/rho_b)
                       + P a2 (tauhat_a/rho_a + tauhat_b/rho_b)
        dP_ab/dR_a  = -dP/dD  tau_b/rho_a - P a2 tauhat_a/rho_a

    Every atomic derivative is minus the corresponding piece of the point
    derivative, which is why :meth:`PseudoDensityDerivatives.translation_residual`
    vanishes identically rather than approximately.
    """
    p = params or PseudoDensityParams()
    r = np.atleast_2d(np.asarray(points, dtype=np.float64))
    R = np.asarray(atom_positions, dtype=np.float64)
    rho = np.asarray(atom_radii, dtype=np.float64)
    n_pts, n_at = r.shape[0], R.shape[0]
    if rho.shape != (n_at,):
        raise ValueError(
            f"pseudo_density_derivatives: {rho.size} radii for {n_at} atoms."
        )
    if np.any(rho <= 0.0):
        raise ValueError("pseudo_density_derivatives: radii must be positive.")

    tau_vec_all = (r[:, None, :] - R[None, :, :]) / rho[None, :, None]
    tau_all = np.linalg.norm(tau_vec_all, axis=2)

    # The truncation must match the PD that built the surface; see the module
    # docstring. ``sub`` maps a local (truncated) atom slot to a global index.
    if p.n_nearest is not None and n_at > p.n_nearest:
        sub = np.argsort(tau_all, axis=1)[:, : p.n_nearest]
    else:
        sub = np.broadcast_to(np.arange(n_at), (n_pts, n_at))
    rows = np.arange(n_pts)[:, None]
    tau_vec = tau_vec_all[rows, sub]                    # (n_pts, k, 3)
    tau = tau_all[rows, sub]                            # (n_pts, k)
    rho_l = rho[sub]                                    # (n_pts, k)
    k = tau.shape[1]

    safe_tau = np.maximum(tau, 1e-300)
    tau_hat = tau_vec / safe_tau[:, :, None]

    value = np.zeros(n_pts)
    d_point = np.zeros((n_pts, 3))
    d_local = np.zeros((n_pts, k, 3))

    # --- atomic terms -------------------------------------------------
    a_term = np.exp(p.a1 * (tau - 1.0))                 # (n_pts, k)
    value += np.sum(a_term, axis=1)
    grad_a = p.a1 * a_term[:, :, None] * tau_hat / rho_l[:, :, None]
    d_point += np.sum(grad_a, axis=1)
    d_local -= grad_a

    # --- pair terms ---------------------------------------------------
    if k >= 2:
        ia, ib = np.triu_indices(k, k=1)
        ta, tb = tau_vec[:, ia, :], tau_vec[:, ib, :]
        sa, sb = tau[:, ia], tau[:, ib]
        ha, hb = tau_hat[:, ia, :], tau_hat[:, ib, :]
        ra, rb = rho_l[:, ia], rho_l[:, ib]

        dot = np.einsum("pak,pak->pa", ta, tb)
        one_minus = 1.0 - dot
        env = np.exp(p.a2 * (sa + sb - 2.0))
        g = p.c * np.power(one_minus, p.m)
        pair = g * env
        value += np.sum(pair, axis=1)

        dP_dD = -p.c * p.m * np.power(one_minus, p.m - 1) * env

        # dD/dr and the a2 envelope's dr piece.
        dD_dr = tb / ra[:, :, None] + ta / rb[:, :, None]
        env_dr = p.a2 * (ha / ra[:, :, None] + hb / rb[:, :, None])
        d_point += np.sum(
            dP_dD[:, :, None] * dD_dr + pair[:, :, None] * env_dr, axis=1
        )

        # Per-atom pieces: minus the part of the r-derivative that came from
        # that atom's own dependence.
        contrib_a = -(
            dP_dD[:, :, None] * (tb / ra[:, :, None])
            + pair[:, :, None] * p.a2 * ha / ra[:, :, None]
        )
        contrib_b = -(
            dP_dD[:, :, None] * (ta / rb[:, :, None])
            + pair[:, :, None] * p.a2 * hb / rb[:, :, None]
        )
        for slot, contrib in ((ia, contrib_a), (ib, contrib_b)):
            np.add.at(
                d_local.transpose(1, 0, 2), slot, contrib.transpose(1, 0, 2)
            )

    # Scatter the truncated-slot derivatives back onto global atom indices.
    d_atom = np.zeros((n_pts, n_at, 3))
    np.add.at(
        d_atom.reshape(-1, 3),
        (rows * n_at + sub).ravel(),
        d_local.reshape(-1, 3),
    )
    return PseudoDensityDerivatives(value=value, d_point=d_point, d_atom=d_atom)


__all__ = ["PseudoDensityDerivatives", "pseudo_density_derivatives"]


@dataclass(frozen=True)
class IsoPointJacobians:
    """``dt_k/dR_A`` for every iso-surface vertex, in rank-one form.

    The Jacobian of an iso-surface point is **rank one**: the marching grid is
    fixed, so the point can only slide along its own edge. Storing the edge
    vector and the scalar gradient ``ds/dR_A`` separately is therefore not a
    compression trick -- it is the honest shape of the object, and it makes the
    downstream chain rule a scalar contraction instead of a matrix one.

    Attributes
    ----------
    edge : ndarray (n_vert, 3)
        ``e = p1 - p0`` for the grid edge the vertex sits on.
    ds_datom : ndarray (n_vert, n_atoms, 3)
        ``ds/dR_A``, the derivative of the edge parameter.
    grid_anchor : ndarray (n_atoms, 3)
        Per-axis weights of the box translation,
        ``d(grid point)[i]/dR_A[c] = delta_ic * grid_anchor[A, c]``. Zero for a
        grid pinned in space.
    """

    edge: np.ndarray
    ds_datom: np.ndarray
    grid_anchor: np.ndarray

    def jacobian(self, vertex: int, atom: int) -> np.ndarray:
        """The explicit ``3 x 3`` ``dt/dR_A`` for one (vertex, atom) pair."""
        j = np.outer(self.edge[vertex], self.ds_datom[vertex, atom])
        if self.grid_anchor.size:
            j = j + np.diag(self.grid_anchor[atom])
        return j

    def contract(self, adjoint: np.ndarray) -> np.ndarray:
        """Vector-Jacobian product ``sum_k adjoint[k] . dt_k/dR_A``.

        ``adjoint`` is ``(n_vert, 3)``; the result is ``(n_atoms, 3)``. Written
        as a contraction rather than by materializing the rank-one Jacobians,
        which is both the cheaper route and the one the reverse-mode interface
        in :mod:`vibeqc.solvation.gradient` consumes.
        """
        a = np.asarray(adjoint, dtype=np.float64)
        along = np.einsum("vi,vi->v", a, self.edge)
        out = np.einsum("v,vac->ac", along, self.ds_datom)
        if self.grid_anchor.size:
            out = out + self.grid_anchor * a.sum(axis=0)[None, :]
        return out


def iso_point_jacobians(
    surface: IsoSurface,
    atom_positions: np.ndarray,
    atom_radii: np.ndarray,
    params: PseudoDensityParams | None = None,
) -> IsoPointJacobians:
    """``ds/dR_A`` for every vertex of a triangulated iso-surface.

    The vertex satisfies ``PD(p0 + s e; R) = 1`` on a grid edge.
    Differentiating that constraint with respect to an atom position, with
    ``s`` the only free variable::

        (grad_t PD . e) ds/dR_A + grad_{R_A} PD = 0
        ds/dR_A = - grad_{R_A} PD / (grad_t PD . e)

    The same expression follows from the ``ln PD`` form the triangulation
    actually cuts, because the factor ``PD`` appears in numerator and
    denominator and cancels.

    The edge is only fixed if the *box* is. It is not:
    :func:`~vibeqc.solvation.fine_cavity.build_fine_cavity` places the box
    relative to the molecule, so both endpoints translate by
    ``diag(grid_anchor[A])`` when atom ``A`` moves. Their difference ``e`` is
    unchanged, so the vertex still has one degree of freedom, but the
    constraint picks up a term::

        ds/dR_A[c] = -( grad_{R_A} PD[c] + grad_t PD[c] . anchor[A, c] )
                     / (grad_t PD . e)
        dt/dR_A    = diag(anchor[A]) + e (ds/dR_A)^T

    Dropping it is not a rounding error. Summed over atoms it leaves the
    surface derivative equal to ``0`` instead of ``I``, i.e. rigidly
    translating the molecule appears to change the cavity -- a net force on an
    isolated molecule. Measured on water at 0.4 A spacing, the total-area
    gradient missed translational invariance by 0.34 bohr against a largest
    component of 14.7. With the term it is invariant to machine precision, and
    that check needs no finite differences.

    The denominator is the field's directional derivative along the edge.
    It cannot vanish for an edge the surface genuinely crosses -- the field has
    opposite signs at the endpoints -- but a nearly tangential crossing makes
    it small and the point's motion correspondingly large. That is a real
    property of the construction rather than a numerical artefact: such a
    vertex is genuinely ill-determined along its edge, and the tests report the
    conditioning instead of hiding it.
    """
    if np.asarray(getattr(surface, "frame", np.zeros((0, 0)))).size:
        raise NotImplementedError(
            "iso_point_jacobians: this surface's lattice was laid out in a "
            "molecule-fixed frame (#769), so its edges turn with the solute and "
            "'dt/dR_A = diag(anchor[A]) + e ds/dR_A' -- which assumes a fixed "
            "edge direction -- no longer holds. Build the cavity with "
            "frame='lab', or take the gradient by finite differences."
        )
    d = pseudo_density_derivatives(
        surface.vertices, atom_positions, atom_radii, params
    )
    edge = np.asarray(surface.edge_p1 - surface.edge_p0, dtype=np.float64)
    along = np.einsum("vi,vi->v", d.d_point, edge)          # (n_vert,)
    if np.any(along == 0.0):
        raise RuntimeError(
            "iso_point_jacobians: the pseudo-density is stationary along a "
            "cut edge, so the iso-surface point is not locally unique."
        )
    anchor = np.asarray(surface.grid_anchor, dtype=np.float64)
    if anchor.size == 0:
        anchor = np.zeros((atom_positions.shape[0], 3))
    if anchor.shape != (atom_positions.shape[0], 3):
        raise ValueError(
            f"iso_point_jacobians: grid_anchor must be (n_atoms, 3); got "
            f"{anchor.shape} for {atom_positions.shape[0]} atoms."
        )
    numer = d.d_atom + np.einsum("vc,ac->vac", d.d_point, anchor)
    ds = -numer / along[:, None, None]
    return IsoPointJacobians(edge=edge, ds_datom=ds, grid_anchor=anchor)


def triangle_area_gradients(
    surface: IsoSurface, jac: IsoPointJacobians
) -> np.ndarray:
    """``da_T/dR_A`` for every triangle. Shape ``(n_tri, n_atoms, 3)``.

    With ``u = t1 - t0``, ``v = t2 - t0``, ``n = u x v`` and ``a = |n|/2``,
    and using the rank-one vertex Jacobians ``dt_k = e_k ds_k``::

        da/dR = [ c0 ds0 + c1 ds1 + c2 ds2 ]
        c0 = -( n.(e0 x v) + n.(u x e0) ) / (2|n|)
        c1 =    n.(e1 x v)               / (2|n|)
        c2 =    n.(u x e2)               / (2|n|)

    so each triangle's area gradient is a *scalar* combination of its three
    vertices' edge-parameter gradients.

    The box translation ``diag(anchor[A])`` in ``dt_k/dR_A`` contributes
    nothing here, and unconditionally so: an area is invariant under a common
    translation of its vertices, hence ``sum_k da/dt_k = 0`` and the anchor
    term is that vector dotted into a per-axis weight. It does still reach the
    answer, through the ``ds_k`` it changes. Segment *positions* are a
    different matter -- see :func:`smooth_coarsening_vjp`.

    Degenerate triangles (``|n| = 0``) are
    given zero gradient rather than dividing by zero; they carry no area, so
    they contribute nothing to differentiate.
    """
    tri = surface.triangles
    t = surface.vertices[tri]                               # (n_tri, 3, 3)
    e = jac.edge[tri]                                       # (n_tri, 3, 3)
    u = t[:, 1] - t[:, 0]
    v = t[:, 2] - t[:, 0]
    n = np.cross(u, v)
    norm = np.linalg.norm(n, axis=1)
    inv = np.where(norm > 0.0, 1.0 / (2.0 * np.maximum(norm, 1e-300)), 0.0)

    c1 = np.einsum("ti,ti->t", n, np.cross(e[:, 1], v)) * inv
    c2 = np.einsum("ti,ti->t", n, np.cross(u, e[:, 2])) * inv
    c0 = -(
        np.einsum("ti,ti->t", n, np.cross(e[:, 0], v))
        + np.einsum("ti,ti->t", n, np.cross(u, e[:, 0]))
    ) * inv

    ds = jac.ds_datom[tri]                                  # (n_tri, 3, n_at, 3)
    return (
        c0[:, None, None] * ds[:, 0]
        + c1[:, None, None] * ds[:, 1]
        + c2[:, None, None] * ds[:, 2]
    )


def basis_area_gradients(
    surface: IsoSurface, jac: IsoPointJacobians
) -> np.ndarray:
    """``dw_b/dR_A`` per surface vertex. Shape ``(n_vert, n_atoms, 3)``.

    Step 4 gives a lone triangle's corner a third of its area and a tetragon's
    corner a quarter of the pair's (#779), so the derivative distributes the
    same way. It reads the grouping from the surface rather than recomputing
    it, so the two cannot drift apart.
    """
    d_tri = triangle_area_gradients(surface, jac)
    out = np.zeros((surface.vertices.shape[0],) + d_tri.shape[1:])
    lone, pairs, corners = quad_groups(surface)
    if np.any(lone):
        np.add.at(
            out, surface.triangles[lone].ravel(),
            np.repeat(d_tri[lone] / 3.0, 3, axis=0),
        )
    if pairs.shape[0]:
        np.add.at(
            out, corners.ravel(),
            np.repeat(d_tri[pairs].sum(axis=1) / 4.0, 4, axis=0),
        )
    return out


# The forward-mode ``SegmentJacobians`` / ``segment_jacobians`` pair that used
# to live here is gone. It summed over the hard basis-point-to-segment
# assignment, which #757 replaced with a Becke partition, so it no longer
# described the construction at all -- and a second, forward-mode derivative
# maintained alongside the reverse one is exactly the two-copies-of-one-quantity
# hazard of #546, made worse by the fact that the wrong copy would still have
# looked plausible. :func:`smooth_coarsening_vjp` plus
# :meth:`vibeqc.solvation.cavity_derivative.FineCavityDerivative.contract` are
# now the single implementation, and they are reverse-mode because that is what
# the consumer interface asks for and because the forward ``dW/dc`` tensor for
# one oxygen is 209 MB.


__all__ += [
    "IsoPointJacobians",
    "basis_area_gradients",
    "iso_point_jacobians",
    "triangle_area_gradients",
]


# =====================================================================
# The smooth two-pass coarsening, in reverse
# =====================================================================


def smooth_coarsening_vjp(
    cavity: FineCavity,
    adj_position: np.ndarray | None,
    adj_area: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reverse through step 6, returning basis-point and atom adjoints.

    Since #757 the coarsening is a Becke partition of basis points over
    segment centres, twice over, so a segment's area and position are smooth
    functions of *every* basis point rather than sums over a hard assignment.
    The chain, per atom::

        c1_S     = R_a + rho_a u_S                       (rigid)
        W1       = partition(t, c1)
        N_S      = sum_b W1_bS a_b t_b ;  tot1_S = sum_b W1_bS a_b
        centres_S= (N_S + eps c1_S) / (tot1_S + eps)
        W2       = partition(t, centres)
        w_S      = sum_b W2_bS a_b
        M_S      = sum_b W2_bS a_b t_b ;  p_S = M_S / w_S
        p'_S     = project(p_S)                          (step 6, on-sphere)

    Reverse mode rather than forward: the interface downstream is already a
    vector-Jacobian product, the partition adjoints are natural in reverse
    (the forward ``dW/dc`` tensor is 209 MB for one oxygen), and the two passes
    chain in one direction only.

    Returns ``(adj_vertex, adj_vertex_area, adj_atom)`` -- adjoints on the
    iso-surface vertex positions, on the per-vertex areas, and the part that
    lands directly on the atoms through the rigid pass-1 centres and the
    projection.
    """
    from .segment_partition import BECKE_K, SmoothPartition

    n_basis = cavity.basis_area.shape[0]
    n_seg = cavity.n_points
    n_at = cavity.atom_positions.shape[0]

    t_all = cavity.surface.vertices[cavity.basis_vertex]
    a_all = cavity.basis_area
    adj_t = np.zeros((n_basis, 3))
    adj_a = np.zeros(n_basis)
    adj_atom = np.zeros((n_at, 3))
    # Which atom's block each row belongs to, for the step-5 share adjoint
    # below. A row belongs to exactly one block even though a basis *point*
    # supplies rows in several (#771).
    row_atom = np.zeros(n_basis, dtype=int)

    adj_p = (
        np.zeros((n_seg, 3)) if adj_position is None
        else np.array(adj_position, dtype=np.float64)
    )
    adj_w = (
        np.zeros(n_seg) if adj_area is None
        else np.array(adj_area, dtype=np.float64)
    )

    # --- step 6's projection, in reverse ------------------------------
    # p' = R_a + (r_eff/r) g, with g the unprojected mean's offset from the
    # atom and r_eff the softplus of ``r`` against ``rho`` (#770). Writing
    # ``k = r_eff/r`` and ``sigma = d(r_eff)/dr = logistic((r - rho)/soft)``,
    #
    #   dp'/dg = k (I - u u^T) + sigma u u^T
    #
    # which is symmetric, so it is its own transpose, and
    # ``dp'/dx = delta I + dp'/dg (dp/dx - delta I)`` as before. The hard rule
    # is the limit ``sigma -> 0`` inside the sphere and ``k, sigma -> 1``
    # outside it, so this replaces a branch with one expression rather than
    # adding a second path beside it.
    if n_seg:
        owner = cavity.point_atom
        rho = cavity.atom_radii[owner]
        # The *unprojected* mean has to be reconstructed, since ``points``
        # holds the projected one.
        mean = _unprojected_means(cavity)
        g = mean - cavity.atom_positions[owner]
        r = np.maximum(np.linalg.norm(g, axis=1), 1e-300)
        u = g / r[:, None]
        r_eff, sigma = projection_radius(r, rho)
        k = r_eff / r
        uu = np.einsum("si,sj->sij", u, u)
        op = k[:, None, None] * (np.eye(3)[None, :, :] - uu) + (
            sigma[:, None, None] * uu
        )
        pulled = np.einsum("sij,sj->si", op, adj_p)
        np.add.at(adj_atom, owner, adj_p - pulled)
        adj_p = pulled
        mean_used = mean
    else:
        mean_used = cavity.points

    # --- p_S = M_S / w_S ---------------------------------------------
    w = cavity.weights
    adj_M = adj_p / w[:, None]
    adj_w = adj_w - np.einsum("si,si->s", adj_p, mean_used) / w

    # --- per-atom reverse pass ----------------------------------------
    for rec in cavity.coarsening:
        rows = rec.basis_idx
        cols = rec.seg_slice
        t = t_all[rows]
        a = a_all[rows]
        keep = rec.keep

        adj_M_a = np.zeros((rec.w2.shape[1], 3))
        adj_w_a = np.zeros(rec.w2.shape[1])
        adj_M_a[keep] = adj_M[cols]
        adj_w_a[keep] = adj_w[cols]

        # w_S and M_S, both linear in W2, a and t.
        adj_W2 = a[:, None] * (
            np.einsum("Si,bi->bS", adj_M_a, t) + adj_w_a[None, :]
        )
        adj_a[rows] += np.einsum(
            "bS,bS->b", rec.w2,
            np.einsum("Si,bi->bS", adj_M_a, t) + adj_w_a[None, :],
        )
        adj_t[rows] += a[:, None] * (rec.w2 @ adj_M_a)

        # W2 = partition(t, centres)
        part2 = SmoothPartition(
            weights=rec.w2, _points=t, _centres=rec.centres, _k=BECKE_K
        )
        d_t2, adj_centres = part2.vjp(adj_W2)
        adj_t[rows] += d_t2

        # centres_S = (N_S + eps c1_S) / denom_S
        adj_N = adj_centres / rec.denom[:, None]
        adj_denom = -np.einsum("Si,Si->S", adj_centres, rec.centres) / rec.denom
        adj_c1 = rec.eps * adj_centres / rec.denom[:, None]

        # N_S and tot1_S, both linear in W1, a and t.
        adj_W1 = a[:, None] * (
            np.einsum("Si,bi->bS", adj_N, t) + adj_denom[None, :]
        )
        adj_a[rows] += np.einsum(
            "bS,bS->b", rec.w1,
            np.einsum("Si,bi->bS", adj_N, t) + adj_denom[None, :],
        )
        adj_t[rows] += a[:, None] * (rec.w1 @ adj_N)

        # W1 = partition(t, c1)
        part1 = SmoothPartition(
            weights=rec.w1, _points=t, _centres=rec.c1, _k=BECKE_K
        )
        d_t1, adj_c1_part = part1.vjp(adj_W1)
        adj_t[rows] += d_t1
        adj_c1 = adj_c1 + adj_c1_part

        # c1_S = R_a + rho_a u_S: rigid, so the whole column lands on the atom.
        ia = int(cavity.point_atom[cols][0]) if np.any(keep) else 0
        adj_atom[ia] += adj_c1.sum(axis=0)
        row_atom[rows] = ia

    # --- basis points are surface vertices; areas are grouped ---------
    adj_vertex = np.zeros((cavity.surface.vertices.shape[0], 3))
    np.add.at(adj_vertex, cavity.basis_vertex, adj_t)

    # A row's area is the basis point's area times its step-5 share, so the
    # adjoint on the surface's own areas carries that share (#771). With the
    # old hard assignment every share was exactly one and this factor was
    # invisible, which is why it reads as new rather than as changed.
    share = np.asarray(cavity.basis_share, dtype=np.float64)
    adj_vertex_area = np.zeros(cavity.surface.vertices.shape[0])
    np.add.at(
        adj_vertex_area, cavity.area_vertex,
        adj_a[cavity.area_basis] * share[cavity.area_basis],
    )

    # The share is itself a function of the geometry: of where the basis point
    # is, and of where the atoms are. Omitting this term would leave a gradient
    # that is smooth, plausible and wrong -- and translation invariance would
    # not catch it, because a rigid translation moves the point and the atoms
    # together and the partition is built from relative distances. The tests use
    # finite differences and the partition's own exact identity instead.
    part = cavity.atom_share
    if part is not None and n_basis:
        merged_area = np.asarray(cavity.merged_area, dtype=np.float64)
        source = np.asarray(cavity.basis_source, dtype=int)
        adj_share = np.zeros(part.weights.shape)
        np.add.at(adj_share, (source, row_atom), adj_a * merged_area[source])
        d_points, d_atoms = part.vjp(adj_share)
        np.add.at(adj_vertex, np.asarray(cavity.merged_vertex, dtype=int), d_points)
        adj_atom = adj_atom + d_atoms
    return adj_vertex, adj_vertex_area, adj_atom


def _unprojected_means(cavity: FineCavity) -> np.ndarray:
    """The area-weighted means before step 6 pushed them onto the spheres.

    Recomputed from the stored weights rather than stored separately, because
    it is exactly ``sum_b W2 a t / w`` and storing a second copy of a derived
    quantity is what #546 was.
    """
    out = np.array(cavity.points, dtype=np.float64)
    t_all = cavity.surface.vertices[cavity.basis_vertex]
    a_all = cavity.basis_area
    for rec in cavity.coarsening:
        rows, cols, keep = rec.basis_idx, rec.seg_slice, rec.keep
        aw = rec.w2[:, keep] * a_all[rows][:, None]
        tot = aw.sum(axis=0)
        out[cols] = (aw.T @ t_all[rows]) / tot[:, None]
    return out


__all__ += ["smooth_coarsening_vjp"]
