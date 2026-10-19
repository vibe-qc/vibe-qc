"""Point-group operations of a nuclear framework, and gradient symmetrization.

Why this lives in the solvation package (#535). The COSMO FINE Cavity built in
its molecule-fixed frame (#769) is *not* symmetric at a symmetric solute: the
frame's step-1 selections pick one of several symmetry-equivalent atoms (water's
two hydrogens, ammonia's three), so the discretized cavity, and with it the
reported energy, belongs to one of several candidates related by the solute's
own symmetry operations. Where those candidates meet, the energy has a *cusp*:
a V of slope change 1.4e-3 Ha/bohr for Direct COSMO-RS on water, measured as a
4th second difference of 0.808 against 0.095 microhartree in a scan across the
tie. The analytic gradient then reports one branch's slope, a force along the
tie-breaking direction that the mirror symmetry requires to vanish, while a
central finite difference reports the average of the two branches.

The two agree once the analytic gradient is **projected onto the totally
symmetric subspace of the solute's point group**, which is also what the exact
(continuum) energy's gradient is at a symmetric geometry. Measured on
symmetric water, RHF/STO-3G, fine cavity at 0.40 A, against a central
difference at h = 1e-3 (Ha/bohr, max component)::

                      raw      symmetrized
    conductor       3.5e-05      4.1e-07
    Direct COSMO-RS 1.4e-03      1.0e-05

Projection is a linear average, ``G -> (1/|g|) sum_g P_g G``, so it preserves the
two exact invariants a gradient has: a zero net force stays zero, and a zero
net torque stays zero (the torque is covariant under the operations, and the
average of a zero vector is zero).

It is what other codes with grid-based energies do at a symmetric geometry, and
it is the same statement the optimizer needs: that a symmetric structure
can be a stationary point.

Tolerance, and what it costs. Operations are accepted when they map every atom
onto an atom of the same element to within ``tol_bohr``. At a near-symmetric
geometry the symmetric projection then also removes the true restoring force of
the symmetry-lowering modes, of order ``k * delta`` for a displacement
``delta <= tol``; at the default 1e-4 bohr that is about 4e-5 Ha/bohr for water's
antisymmetric stretch, below any optimizer threshold, while the cusp artifact it
removes is 1.4e-3. A tolerance of zero disables it.

If the accepted operations do not close under composition (a symmetry that
holds for some operations and misses the tolerance for others), no projection
is applied: the average of a set that is not a group is not a projector and
would be a plausible wrong answer, so the safe result is the raw gradient.

Linear solutes have an infinite group and no frame in this construction (the
molecule-fixed frame refuses them), so they are returned with the identity only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["symmetry_operations", "symmetrize_gradient", "DEFAULT_SYMMETRY_TOL_BOHR"]

#: Geometric tolerance for accepting a symmetry operation, bohr. See the module
#: docstring for what it trades.
DEFAULT_SYMMETRY_TOL_BOHR = 1.0e-4


def _orthonormal_frame(v1: np.ndarray, v2: np.ndarray) -> np.ndarray | None:
    """Columns ``(u1, u2, u3)`` of a right-handed frame from two vectors, or
    ``None`` when they are (numerically) parallel."""
    n1 = np.linalg.norm(v1)
    if n1 <= 0.0:
        return None
    u1 = v1 / n1
    w = v2 - (v2 @ u1) * u1
    n2 = np.linalg.norm(w)
    if n2 <= 1e-9 * max(np.linalg.norm(v2), 1.0):
        return None
    u2 = w / n2
    return np.stack([u1, u2, np.cross(u1, u2)], axis=1)


def symmetry_operations(
    atomic_numbers: Sequence[int],
    positions_bohr,
    tol_bohr: float = DEFAULT_SYMMETRY_TOL_BOHR,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Orthogonal operations that map the atoms onto themselves.

    Returns ``[(Q, perm), ...]`` with the identity first. ``Q`` is a ``3 x 3``
    orthogonal matrix (proper or improper) about the unweighted centroid, and
    ``perm[i]`` is the atom that atom ``i`` is carried to, so
    ``Q @ (x_i - c) + c`` lies within ``tol_bohr`` of atom ``perm[i]``, which has
    the same atomic number.

    An operation is fixed by where it sends two non-parallel position vectors,
    so candidates are generated from the images of one anchor pair, each in its
    proper and its improper completion, and kept when every atom lands on an
    atom. That finds every operation of any finite point group, including the
    ones that fix all atoms of a planar solute (the reflection in its plane).
    """
    Z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    X = np.asarray(positions_bohr, dtype=np.float64).reshape(-1, 3)
    n = Z.shape[0]
    identity = (np.eye(3), np.arange(n))
    tol = float(tol_bohr)
    if n < 2 or not tol > 0.0:
        return [identity]

    c = X.mean(axis=0)
    D = X - c
    r = np.linalg.norm(D, axis=1)

    # Anchor pair: non-parallel position vectors from the rarest classes, so the
    # candidate list is as short as the molecule allows.
    classes: dict[tuple[int, int], int] = {}
    keys = []
    for i in range(n):
        key = (int(Z[i]), int(round(r[i] / max(tol, 1e-12))))
        keys.append(key)
        classes[key] = classes.get(key, 0) + 1
    order = sorted(range(n), key=lambda i: (classes[keys[i]], i))
    anchor = None
    for i in order:
        if r[i] <= tol:
            continue
        for j in order:
            if j == i or r[j] <= tol:
                continue
            frame = _orthonormal_frame(D[i], D[j])
            if frame is not None:
                anchor = (i, j, frame)
                break
        if anchor is not None:
            break
    if anchor is None:
        return [identity]
    i0, j0, frame0 = anchor
    inner0 = float(D[i0] @ D[j0])

    same_z = Z[:, None] == Z[None, :]
    ops: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    scale = float(r[i0] + r[j0])
    for k in np.flatnonzero((Z == Z[i0]) & (np.abs(r - r[i0]) <= tol)):
        for l in np.flatnonzero((Z == Z[j0]) & (np.abs(r - r[j0]) <= tol)):
            if l == k or abs(float(D[k] @ D[l]) - inner0) > tol * scale:
                continue
            target = _orthonormal_frame(D[k], D[l])
            if target is None:
                continue
            for handed in (1.0, -1.0):
                t = target.copy()
                t[:, 2] *= handed
                Q = t @ frame0.T
                images = D @ Q.T
                dist = np.linalg.norm(images[:, None, :] - D[None, :, :], axis=2)
                dist = np.where(same_z, dist, np.inf)
                perm = np.argmin(dist, axis=1)
                if np.any(dist[np.arange(n), perm] > tol):
                    continue
                if np.unique(perm).size != n:
                    continue
                # Proper and improper operations can share an atom permutation
                # (the identity and the reflection in a planar solute's own
                # plane), so the handedness is part of the key.
                key = tuple(int(p) for p in perm) + (int(np.linalg.det(Q) > 0),)
                if key in seen:
                    continue
                seen.add(key)
                ops.append((Q, perm.astype(int)))
    # Identity first, and always present.
    ops = [op for op in ops if not (np.allclose(op[0], np.eye(3), atol=1e-9)
                                    and np.array_equal(op[1], np.arange(n)))]
    ops.insert(0, identity)
    return ops if _closed(ops) else [identity]


def _closed(ops, atol: float = 2e-3) -> bool:
    """Whether the operations are closed under composition.

    Compared on the matrices, to a tolerance that absorbs the tolerance the
    operations were accepted with (they are built from a near-symmetric
    geometry, so two copies of one operation differ by about ``tol / r``).
    """
    mats = [q for q, _ in ops]
    perms = [p for _, p in ops]
    for a in range(len(ops)):
        for b in range(len(ops)):
            prod = mats[a] @ mats[b]
            comp = perms[a][perms[b]]
            if not any(
                np.array_equal(comp, perms[c]) and np.allclose(prod, mats[c], atol=atol)
                for c in range(len(ops))
            ):
                return False
    return True


def symmetrize_gradient(
    gradient,
    atomic_numbers: Sequence[int],
    positions_bohr,
    tol_bohr: float = DEFAULT_SYMMETRY_TOL_BOHR,
) -> np.ndarray:
    """Project a nuclear gradient onto the totally symmetric subspace.

    ``G -> (1/|g|) sum_g P_g G`` over :func:`symmetry_operations`, where
    ``P_g`` carries the force on atom ``i`` to atom ``perm[i]`` rotated by ``Q``.
    A solute with no symmetry (the group is the identity alone) is returned
    unchanged, bit for bit.
    """
    G = np.asarray(gradient, dtype=np.float64)
    ops = symmetry_operations(atomic_numbers, positions_bohr, tol_bohr)
    if len(ops) == 1:
        return G.copy()
    out = np.zeros_like(G)
    for Q, perm in ops:
        out[perm] += G @ Q.T
    return out / len(ops)
