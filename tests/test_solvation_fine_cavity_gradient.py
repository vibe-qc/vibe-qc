"""Analytic nuclear derivatives of the COSMO FINE cavity (issue #729).

Klamt & Diedenhofen 2018 assert that the CFC segment positions and areas are
differentiable because the pseudo-density is an analytic function of the atom
positions, but give no algebra. These tests verify the derivation stage by
stage against finite differences, because an error in any one stage produces a
smooth, plausible, wrong total -- the #546 failure mode.

Every cavity here is built with ``frame="lab"``. That is not the production
default since #769, and it is not an oversight: these tests are of the
construction's own chain rule, which is defined in the frame the cavity was
built in, and reaching it from a molecule-fixed cavity is
``FrameCavityDerivative``'s job -- composing this chain with the frame's
derivative, covered in ``test_solvation_fine_cavity.py``. Passing a
molecule-fixed cavity to the stages below is refused rather than silently
differentiating the wrong function.

The FD harness holds two things fixed that the production code does not:

* the **marching box**, so displacing an atom cannot shift the grid and change
  which edges are cut; and
* the **segment assignment**, which is discrete.

Vertices are matched between geometries by their **grid edge**, not by index:
welding sorts vertices by coordinate, so the ordering permutes as soon as an
atom moves. Matching on index instead silently compares unrelated vertices,
which during development looked exactly like a wrong derivative.
"""

from __future__ import annotations

import math

from dataclasses import replace

import numpy as np
import pytest
from vibeqc.solvation.cavity import ANG_TO_BOHR as A
from vibeqc.solvation.fine_cavity import (
    PseudoDensityParams,
    build_fine_cavity,
    log_pseudo_density,
    marching_tetrahedra,
    pseudo_density,
)
from vibeqc.solvation.fine_cavity_gradient import (
    basis_area_gradients,
    iso_point_jacobians,
    pseudo_density_derivatives,
    triangle_area_gradients,
)

FD_STEP = 1e-5
SPACING_ANG = 0.45


def _water():
    rho = np.array([1.72 * A, 1.30 * A, 1.30 * A])
    d = 0.9584 * A
    th = math.radians(104.45)
    pos = np.array(
        [[0.0, 0.0, 0.0], [d, 0.0, 0.0], [d * math.cos(th), d * math.sin(th), 0.0]]
    )
    return pos, [8, 1, 1], rho


def _box(pos, rho, h):
    """The box ``build_fine_cavity`` builds -- centroid-anchored, and so
    moving with the molecule.

    The FD harness must reproduce this rather than freeze it. The box is part
    of the function being differentiated: freezing it here while the shipped
    builder moves it would verify the derivative of something nobody computes,
    and would hide exactly the missing box-translation term this file exists
    to pin.
    """
    pad = 1.8 * float(np.max(rho)) + 2.0 * h
    centroid = pos.mean(axis=0)
    half = h * np.ceil((np.abs(pos - centroid).max(axis=0) + pad) / h)
    return centroid - half, centroid + half


def _surface_at(positions, rho, h, **kw):
    """A surface built exactly as ``build_fine_cavity`` builds one.

    Including the ``grid_anchor``: a surface without it describes a box pinned
    in space, and the derivative would then be correct for a function this
    harness is no longer evaluating.
    """
    lo, hi = _box(positions, rho, h)
    n = positions.shape[0]
    return replace(
        marching_tetrahedra(
            lambda p: log_pseudo_density(p, positions, rho), lo, hi, h, **kw
        ),
        grid_anchor=np.full((n, 3), 1.0 / n),
    )


def _edge_keys(surface, origin):
    """Identify a grid edge by its integer lattice index, not its coordinate.

    Once the box is anchored to the molecule, a displaced geometry puts the
    same edge at a different absolute position, so coordinate keys stop
    matching. Lattice indices are invariant under the rigid box translation,
    which is precisely the property that makes the vertex correspondence
    well defined at all.
    """
    h = surface.grid_spacing
    k = np.concatenate(
        [
            np.rint((surface.edge_p0 - origin) / h).astype(np.int64),
            np.rint((surface.edge_p1 - origin) / h).astype(np.int64),
        ],
        axis=1,
    )
    return [tuple(int(x) for x in r) for r in k]


def _reindex_to(reference, ref_origin, surface, surf_origin):
    """Permutation taking reference vertex order onto ``surface``'s.

    ``None`` when the topology changed, which is the signal to skip that
    displacement rather than compare different surfaces.
    """
    if surface.vertices.shape[0] != reference.vertices.shape[0]:
        return None
    ref = {k: i for i, k in enumerate(_edge_keys(reference, ref_origin))}
    perm = np.full(reference.vertices.shape[0], -1, dtype=int)
    for j, k in enumerate(_edge_keys(surface, surf_origin)):
        i = ref.get(k)
        if i is None:
            return None
        perm[i] = j
    return None if np.any(perm < 0) else perm


def _vertex_areas(surface):
    """Step 4's assignment, independently of the module under test.

    A lone triangle gives a third of its area to each of its three corners; a
    tetragon gives a quarter of the pair's to each of its four (#779). Written
    out here rather than imported, because this is the reference a finite
    difference is compared against and importing the implementation would make
    the comparison vacuous.
    """
    from vibeqc.solvation.fine_cavity import quad_groups

    area = surface.triangle_areas()
    out = np.zeros(surface.vertices.shape[0])
    lone, pairs, corners = quad_groups(surface)
    np.add.at(out, surface.triangles[lone].ravel(), np.repeat(area[lone] / 3.0, 3))
    if pairs.shape[0]:
        np.add.at(
            out, corners.ravel(), np.repeat(area[pairs].sum(axis=1) / 4.0, 4)
        )
    return out


# ---------------------------------------------------------------------
# Stage 1: the pseudo-density gradients
# ---------------------------------------------------------------------


def test_derivatives_reproduce_the_pseudo_density_value():
    pos, _, rho = _water()
    pts = pos.mean(axis=0) + np.random.default_rng(1).normal(size=(8, 3))
    d = pseudo_density_derivatives(pts, pos, rho)
    np.testing.assert_allclose(d.value, pseudo_density(pts, pos, rho), rtol=1e-14)


def test_translation_invariance_is_exact():
    """``sum_A grad_{R_A} PD = -grad_r PD``, with no finite differences.

    ``PD`` depends on the atoms only through ``r - R_alpha``, so translating
    every atom is the same as translating the field point the other way. Any
    term whose two derivative branches disagree shows up here, which makes it
    a sharper and cheaper test than an FD comparison -- and it caught nothing
    only because it was written first.
    """
    rng = np.random.default_rng(4)
    for n_at in (1, 2, 4, 7):
        R = rng.normal(size=(n_at, 3)) * 2.0
        rho = rng.uniform(2.0, 4.0, size=n_at)
        pts = rng.normal(size=(10, 3)) * 2.0
        d = pseudo_density_derivatives(pts, R, rho)
        assert d.translation_residual() < 1e-9


@pytest.mark.parametrize("n_nearest", [None, 5])
def test_pseudo_density_gradients_match_finite_differences(n_nearest):
    rng = np.random.default_rng(4)
    R = rng.normal(size=(4, 3)) * 2.0
    rho = np.array([3.2, 2.5, 2.5, 3.7])
    pts = rng.normal(size=(6, 3)) * 2.0
    params = PseudoDensityParams(n_nearest=n_nearest)
    d = pseudo_density_derivatives(pts, R, rho, params)

    h = FD_STEP
    scale = max(float(np.max(np.abs(d.d_point))), 1.0)
    for c in range(3):
        e = np.zeros(3)
        e[c] = h
        fd = (
            pseudo_density(pts + e, R, rho, params)
            - pseudo_density(pts - e, R, rho, params)
        ) / (2 * h)
        assert np.max(np.abs(fd - d.d_point[:, c])) < 1e-6 * scale

    scale_a = max(float(np.max(np.abs(d.d_atom))), 1.0)
    for a in range(R.shape[0]):
        for c in range(3):
            Rp, Rm = R.copy(), R.copy()
            Rp[a, c] += h
            Rm[a, c] -= h
            fd = (
                pseudo_density(pts, Rp, rho, params)
                - pseudo_density(pts, Rm, rho, params)
            ) / (2 * h)
            assert np.max(np.abs(fd - d.d_atom[:, a, c])) < 1e-6 * scale_a


# ---------------------------------------------------------------------
# Stage 2: iso-surface point Jacobians
# ---------------------------------------------------------------------


def test_iso_point_motion_is_confined_to_its_own_grid_edge():
    """The rank-one property, stated as a number.

    A vertex has exactly one degree of freedom *relative to its edge*, so once
    the box translation is subtracted the motion is parallel to the edge. If
    the recorded edge or the implicit differentiation were wrong, a component
    would survive that subtraction.

    The subtraction is not a fudge: the box moves rigidly, carrying both
    endpoints, so a vertex's freedom is the edge parameter and nothing else.
    Testing the raw motion instead would just be measuring the box.
    """
    pos, z, rho = _water()
    h = SPACING_ANG * A
    S = _surface_at(pos, rho, h)
    jac = iso_point_jacobians(S, pos, rho)

    rng = np.random.default_rng(0)
    sample = rng.choice(S.vertices.shape[0], size=12, replace=False)
    e = jac.edge
    worst = 0.0
    kept = 0
    for vi in sample:
        for a in range(3):
            for c in range(3):
                fd = _vertex_fd(S, vi, pos, rho, h, a, c)
                if fd is None:
                    continue
                # Remove the rigid box translation; what is left must lie on
                # the edge.
                fd = fd - np.diag(jac.grid_anchor[a])[:, c]
                nrm = float(np.linalg.norm(fd))
                # Skip vertices whose slide along the edge is genuinely zero
                # -- by symmetry, six of these 108 samples have an exactly
                # vanishing prediction, and the direction of a 4e-11 bisection
                # residual is noise. The floor sits three orders above that
                # noise and still keeps 102 samples, which the count below
                # pins so the test cannot pass by skipping everything.
                if nrm < 1e-7:
                    continue
                perp = fd - (fd @ e[vi]) / (e[vi] @ e[vi]) * e[vi]
                worst = max(worst, float(np.linalg.norm(perp)) / nrm)
                kept += 1
    assert kept >= 90, f"only {kept} resolvable samples"
    assert worst < 1e-6


def _locate_on_edge(p0, p1, positions, rho, tol=1e-14):
    """Bisect for ``PD = 1`` on the fixed edge ``p0 -> p1``."""
    def g(s):
        return float(
            pseudo_density((p0 + s * (p1 - p0))[None, :], positions, rho)[0] - 1.0
        )

    a, b = 0.0, 1.0
    ga, gb = g(a), g(b)
    if ga * gb > 0.0:
        return None
    for _ in range(200):
        m = 0.5 * (a + b)
        gm = g(m)
        if ga * gm <= 0.0:
            b, gb = m, gm
        else:
            a, ga = m, gm
        if b - a < tol:
            break
    return p0 + 0.5 * (a + b) * (p1 - p0)


def test_iso_point_jacobians_match_finite_differences():
    pos, z, rho = _water()
    h = SPACING_ANG * A
    S = _surface_at(pos, rho, h)
    jac = iso_point_jacobians(S, pos, rho)

    rng = np.random.default_rng(2)
    sample = rng.choice(S.vertices.shape[0], size=12, replace=False)
    scale = float(np.max(np.abs(jac.ds_datom))) * float(
        np.max(np.linalg.norm(jac.edge, axis=1))
    )
    worst = 0.0
    for vi in sample:
        for a in range(3):
            for c in range(3):
                fd = _vertex_fd(S, vi, pos, rho, h, a, c)
                if fd is None:
                    continue
                # The full Jacobian column: edge slide plus box translation.
                an = jac.jacobian(vi, a)[:, c]
                worst = max(worst, float(np.max(np.abs(fd - an))))
    assert worst < 1e-6 * max(scale, 1.0)


def _vertex_fd(S, vi, pos, rho, h, a, c):
    """``dt_vi/dR_a[c]`` by finite difference, tracking the moving box.

    The edge endpoints are grid points, and the grid is anchored to the
    molecule, so a displaced geometry puts *this* edge somewhere else. Holding
    the endpoints at their reference coordinates would bisect a different edge
    and measure the derivative of a different function -- which is how the
    missing box term stayed invisible.
    """
    o0 = _box(pos, rho, h)[0]
    out = []
    for sgn in (+1.0, -1.0):
        R = pos.copy()
        R[a, c] += sgn * FD_STEP
        shift = _box(R, rho, h)[0] - o0
        t = _locate_on_edge(S.edge_p0[vi] + shift, S.edge_p1[vi] + shift, R, rho)
        if t is None:
            return None
        out.append(t)
    return (out[0] - out[1]) / (2 * FD_STEP)


def test_vertices_must_lie_on_the_iso_surface_for_the_derivative_to_be_right():
    """Refinement is not cosmetic; the derivative depends on it.

    The paper places a vertex by quadratic interpolation, which leaves
    ``|PD - 1|`` up to ~6e-2. The derivative comes from differentiating the
    constraint ``PD = 1``, so an unrefined vertex yields the derivative of a
    *different* point. Measured on water, that shows up as a ~0.1-1 % error in
    the total-area gradient -- smooth, plausible and wrong.
    """
    pos, z, rho = _water()
    h = SPACING_ANG * A
    lo, hi = _box(pos, rho, h)
    coarse = marching_tetrahedra(
        lambda p: log_pseudo_density(p, pos, rho), lo, hi, h, refine=False
    )
    fine = _surface_at(pos, rho, h)
    dev_coarse = np.max(np.abs(pseudo_density(coarse.vertices, pos, rho) - 1.0))
    dev_fine = np.max(np.abs(pseudo_density(fine.vertices, pos, rho) - 1.0))
    assert dev_coarse > 1e-3
    assert dev_fine < 1e-10


# ---------------------------------------------------------------------
# Stages 3 and 4: triangle and basis-point areas
# ---------------------------------------------------------------------


def test_area_gradients_match_finite_differences():
    """Per-vertex areas, and hence the total, differentiate correctly."""
    pos, z, rho = _water()
    h = SPACING_ANG * A
    S = _surface_at(pos, rho, h)
    jac = iso_point_jacobians(S, pos, rho)
    dwv = basis_area_gradients(S, jac)
    assert triangle_area_gradients(S, jac).shape == (S.n_triangles, 3, 3)

    scale = max(float(np.max(np.abs(dwv))), 1.0)
    total_an = dwv.sum(axis=0)
    for a in range(3):
        for c in range(3):
            Rp, Rm = pos.copy(), pos.copy()
            Rp[a, c] += FD_STEP
            Rm[a, c] -= FD_STEP
            Sp = _surface_at(Rp, rho, h)
            Sm = _surface_at(Rm, rho, h)
            o0 = _box(pos, rho, h)[0]
            pp = _reindex_to(S, o0, Sp, _box(Rp, rho, h)[0])
            pm = _reindex_to(S, o0, Sm, _box(Rm, rho, h)[0])
            if pp is None or pm is None:
                continue
            fd_vert = (
                _vertex_areas(Sp)[pp] - _vertex_areas(Sm)[pm]
            ) / (2 * FD_STEP)
            assert np.max(np.abs(fd_vert - dwv[:, a, c])) < 1e-6 * scale
            # And the total, which is the physically meaningful aggregate.
            fd_tot = (Sp.area - Sm.area) / (2 * FD_STEP)
            assert abs(fd_tot - total_an[a, c]) < 1e-6 * max(abs(fd_tot), 1.0)


# ---------------------------------------------------------------------
# Stage 5: segment positions and areas
# ---------------------------------------------------------------------


def _segment_quantities(cav, surface, perm, positions):
    """Segment areas and positions at a displaced geometry, reference
    assignment held fixed, including step 6's outward projection."""
    wv = _vertex_areas(surface)[perm]
    verts = surface.vertices[perm]
    n_seg = cav.n_points
    area = np.zeros(n_seg)
    np.add.at(area, cav.area_segment, wv[cav.area_vertex])
    bw = np.zeros(cav.basis_vertex.size)
    np.add.at(bw, cav.area_basis, wv[cav.area_vertex])
    num = np.zeros((n_seg, 3))
    np.add.at(num, cav.basis_segment, verts[cav.basis_vertex] * bw[:, None])
    pos = num / np.maximum(area, 1e-300)[:, None]
    owner = cav.point_atom
    # Projection is onto the DISPLACED sphere; using the reference atom
    # position here is a harness bug that mimics a wrong derivative exactly.
    g = pos - positions[owner]
    r = np.linalg.norm(g, axis=1)
    pos = np.where(
        cav.projected[:, None],
        positions[owner] + g * (cav.atom_radii[owner] / np.maximum(r, 1e-300))[:, None],
        pos,
    )
    return area, pos


def test_segment_derivatives_match_finite_differences():
    """The assembled segment derivatives, areas and positions alike.

    Since #757 these come from one reverse-mode pass rather than an explicit
    Jacobian, so they are probed the way a reverse-mode derivative is: with
    random adjoints on the segment areas and positions, which tests every
    segment at once instead of one column at a time.

    The step is 1e-4 rather than smaller. The Becke partition is stiff --
    ``err/h^2`` for this functional is around 7e7, measured -- so finite
    differences here are truncation-limited well before they are
    roundoff-limited, and a tighter step does not help.
    """
    from vibeqc.solvation.cavity_derivative import FineCavityDerivative

    pos, z, rho = _water()
    cav = build_fine_cavity(pos, z, rho, grid_spacing_ang=SPACING_ANG, frame="lab")
    deriv = FineCavityDerivative(cav)

    assert cav.projected.size == cav.n_points
    assert 0 < int(cav.projected.sum()) < cav.n_points, "need both kinds"

    rng = np.random.default_rng(0)
    adj_area = rng.normal(size=cav.n_points)
    adj_pos = rng.normal(size=(cav.n_points, 3))
    an_a = deriv.contract(None, adj_area)
    an_p = deriv.contract(adj_pos, None)

    h = 1e-4
    for label, adj, analytic, functional in (
        ("area", adj_area, an_a, lambda c: float(adj_area @ c.weights)),
        ("position", adj_pos, an_p, lambda c: float(np.sum(adj_pos * c.points))),
    ):
        fd = np.zeros_like(analytic)
        for a in range(pos.shape[0]):
            for k in range(3):
                vals = []
                for sgn in (+1, -1):
                    p = pos.copy()
                    p[a, k] += sgn * h
                    c = build_fine_cavity(p, z, rho, frame="lab",
                                          grid_spacing_ang=SPACING_ANG)
                    assert c.n_points == cav.n_points, "topology moved"
                    vals.append(functional(c))
                fd[a, k] = (vals[0] - vals[1]) / (2 * h)
        scale = max(float(np.max(np.abs(fd))), 1.0)
        assert np.max(np.abs(analytic - fd)) < 2e-4 * scale, label


def test_translation_invariants_hold_without_finite_differences():
    """``sum_A dp_S/dR_A = I`` and ``sum_A dw_S/dR_A = 0``, exactly.

    The sharpest check available on the reverse pass, and the cheapest: it
    needs no displaced geometries and no tolerance argument. Both were how
    real defects announced themselves -- the missing box-translation term in
    #729, and the flickering pass-1 pruning that the #757 rework introduced
    and this caught.
    """
    from vibeqc.solvation.cavity_derivative import FineCavityDerivative

    pos, z, rho = _water()
    cav = build_fine_cavity(pos, z, rho, grid_spacing_ang=SPACING_ANG, frame="lab")
    pos_res, area_res = FineCavityDerivative(cav).translation_residuals()
    assert pos_res < 1e-8
    assert area_res < 1e-10


def test_total_area_gradient_matches_finite_differences():
    """The one cavity derivative with no partition dependence at all.

    Rows of the Becke weight matrix sum to one, so ``sum_S w_S = sum_b a_b``
    exactly and the *total* area gradient is independent of how the partition
    distributes area. That makes it a clean probe of the basis-area and
    iso-point stages on their own -- and it converges much better than the
    per-segment quantities, which inherit the partition's stiffness.
    """
    from vibeqc.solvation.cavity_derivative import FineCavityDerivative

    pos, z, rho = _water()
    cav = build_fine_cavity(pos, z, rho, grid_spacing_ang=SPACING_ANG, frame="lab")
    analytic = FineCavityDerivative(cav).contract(None, np.ones(cav.n_points))

    h = 1e-5
    fd = np.zeros_like(analytic)
    for a in range(pos.shape[0]):
        for k in range(3):
            vals = []
            for sgn in (+1, -1):
                p = pos.copy()
                p[a, k] += sgn * h
                vals.append(
                    build_fine_cavity(
                        p, z, rho, grid_spacing_ang=SPACING_ANG, frame="lab").total_area
                )
            fd[a, k] = (vals[0] - vals[1]) / (2 * h)
    scale = max(float(np.max(np.abs(fd))), 1.0)
    assert np.max(np.abs(analytic - fd)) < 1e-8 * scale
