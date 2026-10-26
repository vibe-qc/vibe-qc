"""Nuclear derivatives of a cavity, as a vector-Jacobian product.

Every geometry dependence of the CPCM/COSMO energy that runs through the
cavity reaches it through exactly two per-segment quantities: the segment
**positions** ``p_S`` and the segment **areas** ``w_S``. The ``A`` matrix uses
both, the nuclear ESP uses positions, and the electronic ESP uses positions.
Nothing else in the solvated energy depends on the cavity at all (checked
against :mod:`vibeqc.solvation.driver`, where the only consumer is
``build_A_matrix(cavity.points, cavity.weights)``).

So the derivative of a cavity term factors cleanly::

    dG/dR_A = sum_S  (dG/dp_S) . (dp_S/dR_A)  +  (dG/dw_S) (dw_S/dR_A)

The left factors are *the term's* business and are the same for every cavity
construction. The right factors are *the cavity's* business and differ
completely between constructions. This module is that seam: a term computes
per-segment adjoints, and the cavity contracts them.

Writing it as a contraction rather than as an explicit Jacobian is not
housekeeping. The Lebedev cavity's position Jacobian is
``delta_{A,a(S)} I`` -- a scatter, never worth materializing -- while its area
Jacobian runs through the switching function and is dense. The FINE cavity's
is dense in both. A single interface that lets each one keep its own shape is
what stops a term from being written twice, which is how a correction gets
applied to one construction and silently not to the other (#546).

The adjoints are kept separate for left and right (``q_left``, ``q_right`` in
:func:`cavity_A_adjoints`) because Direct COSMO-RS differentiates
``u^T A v`` with distinct vectors, not only the energy's ``q^T A q``.

Both constructions must satisfy two exact invariants, and
:meth:`CavityDerivative.translation_residuals` checks them without any finite
differences: translating every atom together translates the cavity rigidly, so
``sum_A dp_S/dR_A = I`` and ``sum_A dw_S/dR_A = 0``. A violation is a net
force on an isolated molecule.
"""

from __future__ import annotations

import math
from typing import Optional, Protocol

import numpy as np

from .cpcm import CPCM_DIAG_ALPHA, GAUSSIAN_CHARGE, POINT_CHARGE

__all__ = [
    "CavityDerivative",
    "FineCavityDerivative",
    "FrameCavityDerivative",
    "LebedevCavityDerivative",
    "cavity_A_adjoints",
    "cavity_derivative",
]

_ERF = np.frompyfunc(math.erf, 1, 1)


class CavityDerivative(Protocol):
    """Contracts per-segment adjoints into a per-atom Cartesian gradient."""

    @property
    def n_atoms(self) -> int:
        ...

    def contract(
        self,
        adj_position: Optional[np.ndarray] = None,
        adj_area: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """``sum_S adj_position[S] . dp_S/dR_A + adj_area[S] dw_S/dR_A``.

        ``adj_position`` is ``(n_seg, 3)``, ``adj_area`` is ``(n_seg,)``, and
        either may be ``None``. Returns ``(n_atoms, 3)``.
        """
        ...

    def translation_residuals(self) -> tuple[float, float]:
        """``(max |sum_A dp_S/dR_A - I|, max |sum_A dw_S/dR_A|)``."""
        ...


# =====================================================================
# The cavity-independent half: adjoints of the A-matrix term
# =====================================================================


def cavity_A_adjoints(
    points: np.ndarray,
    weights: np.ndarray,
    q_left: np.ndarray,
    q_right: np.ndarray,
    scale: float,
    *,
    representation: str = POINT_CHARGE,
    switching: np.ndarray | None = None,
    zeta: float | None = None,
    n_points_per_sphere: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment adjoints of ``scale * u^T A v``.

    ``representation`` must be the one that *built* ``A``. Differentiating the
    point-charge kernel while the energy used the Gaussian one is #546's shape
    exactly, and it is not subtle enough to hide: on water/CFC it left the
    analytic-vs-FD residual at 5.5e-6 where the correct pairing gives 1.5e-6.

    With ``A_ij = 1/|s_i - s_j|`` off-diagonal and ``A_ii = C_S sqrt(4 pi/a_i)``
    (the point-charge form), differentiating with respect to segment geometry::

        dG/ds_i = scale * sum_{j != i} (u_i v_j + u_j v_i)
                          . ( -(s_i - s_j) / |s_i - s_j|^3 )
        dG/dw_i = scale * u_i v_i * ( -A_ii / (2 w_i) )

    The ``(u_i v_j + u_j v_i)`` factor is the honest general form: ``s_i``
    appears in row ``i`` *and* column ``i`` of ``A``, and for ``u = v = q``
    it collapses to the familiar ``2 q_i q_j``. Losing that factor of two was
    the 2026-05-31 audit's 3.5e-4 Ha/bohr residual; carrying it explicitly is
    cheaper than re-deriving it per call site.

    For the energy gradient the caller passes ``q_left = q_right = q`` and
    ``scale = 1/(2f)``, giving ``(1/(2f)) q^T (dA/dR) q``. ``q`` is stored
    scaled (``q = f q0``), so ``f`` is the factor that *solve* used and is not
    a free choice -- see :func:`vibeqc.solvation.gradient._screening_model`.
    """
    pts = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    u = np.asarray(q_left, dtype=np.float64).reshape(-1)
    v = np.asarray(q_right, dtype=np.float64).reshape(-1)

    diff = pts[:, None, :] - pts[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", diff, diff)
    r2[r2 == 0.0] = 1.0
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    np.fill_diagonal(inv_r3, 0.0)
    sym = u[:, None] * v[None, :] + u[None, :] * v[:, None]

    if representation == GAUSSIAN_CHARGE:
        return _gaussian_A_adjoints(
            pts, w, u, v, sym, diff, r2, inv_r3, scale,
            switching, zeta, n_points_per_sphere,
        )
    if representation != POINT_CHARGE:
        raise ValueError(
            f"cavity_A_adjoints: unknown representation {representation!r} "
            f"(use {POINT_CHARGE!r} or {GAUSSIAN_CHARGE!r})."
        )

    adj_position = scale * np.einsum("ij,ijk->ik", -sym * inv_r3, diff)

    A_diag = CPCM_DIAG_ALPHA / np.sqrt(w)
    adj_area = scale * (u * v) * (-A_diag / (2.0 * w))
    return adj_position, adj_area


def _gaussian_A_adjoints(
    pts, w, u, v, sym, diff, r2, inv_r3, scale,
    switching, zeta, n_points_per_sphere,
):
    """Adjoints of the York-Karplus Gaussian kernel.

    ``A_ij = erf(zeta_ij r_ij)/r_ij`` with
    ``zeta_ij = zeta_i zeta_j / sqrt(zeta_i^2 + zeta_j^2)`` and
    ``zeta_i = zeta sqrt(F_i / a_i)``.

    **Position.** Differentiating the kernel (Lange & Herbert eq. 3.9) gives
    the point-charge expression times one scalar factor::

        dA_ij/ds_i = -[ erf(x) - (2x/sqrt(pi)) e^{-x^2} ] (s_i - s_j)/r^3

    with ``x = zeta_ij r_ij``. The bracket tends to 1 as ``x`` grows, so
    distant pairs are numerically identical to the point-charge result and the
    whole correction lives where segments are close.

    **Area.** Here the two kernels genuinely differ in structure, and this is
    the term that is easy to miss: ``zeta_ij`` depends on the areas, so the
    *off-diagonal* elements have an area derivative that the point-charge
    kernel simply does not have. With ``d zeta_i/d a_i = -zeta_i/(2 a_i)`` and
    ``d zeta_ij/d zeta_i = (zeta_ij/zeta_i)^3``::

        dA_ij/da_i = (2/sqrt(pi)) e^{-x^2} (zeta_ij/zeta_i)^3
                     . ( -zeta_i / (2 a_i) )

    and the diagonal keeps the familiar form, since ``A_ii`` scales as
    ``a_i^{-1/2}`` in both representations::

        dA_ii/da_i = -A_ii / (2 a_i)

    Restricted to ``F == 1``. For a switched cavity ``a_i`` and ``F_i`` are not
    independent (``a_i = a_i^raw F_i``), so ``zeta_i`` is switching-invariant
    while ``A_ii`` carries ``1/F_i`` -- a different chain rule that is not
    derived here. It refuses rather than guessing.
    """
    from scipy.special import erf as _erf

    from .cpcm import gaussian_exponents, york_karplus_zeta

    if switching is not None and not np.allclose(switching, 1.0, atol=0.0,
                                                 rtol=1e-14):
        raise NotImplementedError(
            "cavity_A_adjoints: the Gaussian-kernel derivative is derived only "
            "for a cavity with no switching function (F == 1). For a switched "
            "cavity the area and the switching factor are not independent, so "
            "the chain rule differs; deriving and FD-verifying it is separate "
            "work. Use the point-charge representation, which is what every "
            "switched cavity in vibe-qc uses."
        )
    if zeta is None:
        zeta = york_karplus_zeta(n_points_per_sphere)
    z = gaussian_exponents(w, None, zeta=zeta)

    r = np.sqrt(r2)
    z_ij = np.outer(z, z) / np.sqrt(z[:, None] ** 2 + z[None, :] ** 2)
    x = z_ij * r
    gauss = np.exp(-x * x)
    erf_x = _erf(x)

    # --- position -----------------------------------------------------
    bracket = erf_x - (2.0 / np.sqrt(np.pi)) * x * gauss
    np.fill_diagonal(bracket, 0.0)
    adj_position = scale * np.einsum(
        "ij,ijk->ik", -sym * bracket * inv_r3, diff
    )

    # --- area ---------------------------------------------------------
    # Off-diagonal channel, absent for point charges.
    dA_dzij = (2.0 / np.sqrt(np.pi)) * gauss
    np.fill_diagonal(dA_dzij, 0.0)
    dzij_dzi = (z_ij / z[:, None]) ** 3
    dzi_dai = -z / (2.0 * w)
    adj_area = scale * dzi_dai * np.einsum(
        "ij,ij->i", sym * dA_dzij, dzij_dzi
    )
    # Diagonal channel.
    A_diag = z * np.sqrt(2.0 / np.pi)
    adj_area += scale * (u * v) * (-A_diag / (2.0 * w))
    return adj_position, adj_area


# =====================================================================
# The Lebedev cavity: rigid segments, geometry through the switch
# =====================================================================


class LebedevCavityDerivative:
    """Segments ride rigidly on their parent atom; areas follow the switch.

    A Lebedev segment sits at ``R_a + rho_a u_k`` for a fixed unit-sphere
    direction ``u_k``, so ``dp_S/dR_A = delta_{A,a(S)} I`` exactly. Its area is
    ``rho_a^2 w_k^{Lebedev} sigma_S`` with ``sigma_S = prod_{b != a} f_b(d_Sb)``
    the Scalmani-Frisch erf switch, so all of the area's geometry dependence
    runs through ``sigma``::

        dw_S/dR_b       = (w_S / f_b) df_b/dR_b                    (b != a)
        dw_S/dR_{a(S)}  = -sum_{b != a} (w_S / f_b) df_b/dR_b

    the second line by translation invariance: moving the parent moves the
    segment, which is the opposite of moving the neighbour.
    """

    def __init__(self, cavity, switching_sigma_bohr: float = 0.5) -> None:
        self._parent = np.asarray(cavity.point_atom)
        self._weights = np.asarray(cavity.weights, dtype=np.float64)
        self._switching = np.asarray(cavity.switching, dtype=np.float64)
        self._n_atoms = int(np.asarray(cavity.atom_positions).shape[0])
        self._chain = _switching_chain(
            np.asarray(cavity.points, dtype=np.float64),
            self._parent,
            np.asarray(cavity.atom_positions, dtype=np.float64),
            np.asarray(cavity.atom_radii, dtype=np.float64),
            self._weights,
            float(switching_sigma_bohr),
        )

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    def contract(
        self,
        adj_position: Optional[np.ndarray] = None,
        adj_area: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        out = np.zeros((self._n_atoms, 3), dtype=np.float64)
        if adj_position is not None:
            np.add.at(
                out, self._parent, np.asarray(adj_position, dtype=np.float64)
            )
        if adj_area is not None:
            # ``chain[i, b]`` *is* dw_i/dR_b, with the parent column already
            # zeroed, so the neighbour channel is one contraction and the
            # parent channel is minus its row sum.
            scaled = np.asarray(adj_area, dtype=np.float64)
            out += np.einsum("i,ibc->bc", scaled, self._chain)
            np.add.at(
                out, self._parent, -scaled[:, None] * self._chain.sum(axis=1)
            )
        return out

    def translation_residuals(self) -> tuple[float, float]:
        # Positions: exactly one parent per segment, so the sum over atoms is
        # the identity by construction. Areas: the parent channel is defined as
        # minus the sum of the neighbour channels, so they cancel identically.
        # Both are computed rather than asserted so a future edit that breaks
        # the construction is caught by the same check the FINE cavity uses.
        pos = np.zeros((len(self._parent), self._n_atoms, 3, 3))
        pos[np.arange(len(self._parent)), self._parent] = np.eye(3)
        area = np.zeros((len(self._parent), self._n_atoms, 3))
        area += self._chain
        np.add.at(
            area,
            (np.arange(len(self._parent)), self._parent),
            -self._chain.sum(axis=1),
        )
        return (
            float(np.max(np.abs(pos.sum(axis=1) - np.eye(3)[None, :, :]))),
            float(np.max(np.abs(area.sum(axis=1)))),
        )


def _switching_chain(
    points: np.ndarray,
    parent: np.ndarray,
    atom_positions: np.ndarray,
    atom_radii: np.ndarray,
    weights: np.ndarray,
    sigma_bohr: float,
) -> np.ndarray:
    """``dw_S/dR_b = (w_S / f_b) df_b/dR_b`` for every (segment, neighbour).

    Returned as ``(n_seg, n_atoms, 3)`` with the parent column zeroed, since a
    segment's own atom is excluded from the switch product; the parent's own
    derivative is minus the row sum and :meth:`LebedevCavityDerivative.contract`
    applies it there.

    ``w_S = rho^2 w^Lebedev sigma_S`` and only ``sigma_S`` carries geometry, so
    ``dw_S/dR = (w_S/sigma_S) dsigma_S/dR``. Composing that with
    ``dsigma_S/dR_b = (sigma_S/f_b) df_b/dR_b`` cancels the ``sigma_S``
    outright and leaves ``(w_S/f_b) df_b/dR_b`` -- the prefactor is ``w_S``,
    not ``w_S/sigma_S``. Returning ``dsigma/dR`` and leaving the caller to
    rescale is the same quantity in two conventions, which is the shape of
    #546; this returns the area derivative itself.
    """
    n_seg = points.shape[0]
    n_at = atom_positions.shape[0]
    chain = np.zeros((n_seg, n_at, 3), dtype=np.float64)

    diff = points[:, None, :] - atom_positions[None, :, :]      # (n_seg, n_at, 3)
    d = np.linalg.norm(diff, axis=2)
    d = np.where(d > 0.0, d, 1.0)
    arg = (d - atom_radii[None, :]) / sigma_bohr
    f_b = 0.5 * (1.0 + _ERF(arg).astype(np.float64))
    f_b = np.where(f_b > 1e-12, f_b, 1e-12)

    # df_b/dR_b = (1/2) erf'(arg) . d(arg)/dR_b, and d|p - R_b|/dR_b =
    # -(p - R_b)/|p - R_b|, so the sign is negative.
    erf_prime = (1.0 / math.sqrt(math.pi)) * np.exp(-arg * arg)   # = 0.5 * 2/sqrt(pi) e^-x^2
    coef = -erf_prime / (sigma_bohr * d)
    df = coef[:, :, None] * diff                                  # (n_seg, n_at, 3)

    chain = (weights[:, None] / f_b)[:, :, None] * df
    chain[np.arange(n_seg), parent] = 0.0
    return chain


# =====================================================================
# The FINE cavity: everything moves
# =====================================================================


class FineCavityDerivative:
    """Reverse-mode chain through the whole CFC construction.

    A CFC segment is an area-weighted mean of basis points taken from an
    iso-surface of the pseudo-density, optionally projected onto its atom's
    sphere, on a grid that is itself anchored to the molecule. Nothing about it
    is rigid, and every one of those stages contributes. The Jacobians are
    computed once and cached, since the assembled gradient contracts them three
    times (``A``, nuclear ESP, electronic ESP).
    """

    def __init__(self, cavity, params=None) -> None:
        frame = np.asarray(getattr(cavity, "frame", np.zeros((0, 0))))
        if frame.size:
            raise NotImplementedError(
                "FineCavityDerivative: this cavity was built in a "
                "molecule-fixed frame (#769), whose derivative is not part of "
                "the chain below. Every stage here differentiates a lattice and "
                "a set of segment directions that are constants; a "
                "molecule-fixed frame turns with the solute, so both acquire "
                "derivatives and the assembled gradient would be wrong by the "
                "frame's own contribution rather than obviously broken. Use "
                "frame='lab' for an analytic gradient, or cpcm_gradient_fd, "
                "which re-converges displaced geometries and so needs no "
                "cavity derivative at all."
            )
        self._cavity = cavity
        self._params = params
        self._n_atoms = int(np.asarray(cavity.atom_positions).shape[0])
        self._iso = None
        self._dw_vertex = None

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    def _stages(self):
        """Iso-point Jacobians and per-vertex area gradients, built once."""
        if self._iso is None:
            from .fine_cavity_gradient import (
                basis_area_gradients,
                iso_point_jacobians,
            )

            cav = self._cavity
            self._iso = iso_point_jacobians(
                cav.surface, cav.atom_positions, cav.atom_radii, self._params
            )
            self._dw_vertex = basis_area_gradients(cav.surface, self._iso)
        return self._iso, self._dw_vertex

    def contract(
        self,
        adj_position: Optional[np.ndarray] = None,
        adj_area: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reverse through the whole construction.

        Four links: step 6's smooth two-pass Becke coarsening (#757), then the
        per-vertex areas, then the iso-surface vertex positions, then the
        marching box. Each was verified against finite differences on its own,
        because an error in one produces a smooth, plausible, wrong total --
        the failure mode of #546.
        """
        from .fine_cavity_gradient import smooth_coarsening_vjp

        adj_vertex, adj_vertex_area, out = smooth_coarsening_vjp(
            self._cavity, adj_position, adj_area
        )
        iso, dw_vertex = self._stages()
        out = out + iso.contract(adj_vertex)
        out = out + np.einsum("v,vac->ac", adj_vertex_area, dw_vertex)
        return out

    def translation_residuals(self) -> tuple[float, float]:
        """The two exact invariants; see :func:`_translation_residuals`."""
        return _translation_residuals(self, self._cavity.n_points)


def _translation_residuals(deriv, n_seg: int) -> tuple[float, float]:
    """The two exact invariants of any cavity derivative, in reverse mode.

    ``sum_A dp_S/dR_A = I`` and ``sum_A dw_S/dR_A = 0`` per segment. Reading
    them off an explicit Jacobian would need one reverse pass per segment;
    contracting instead gives, for any adjoints,

        sum_A contract(P, w)[A]  =  sum_S P_S . I  +  0
                                 =  sum_S P_S

    so a single pass with *random* adjoints tests every segment at once -- a
    systematic violation in any one of them shows up, and no finite differences
    are involved. This is the check that found the missing box-translation term
    in #729 and the flickering pass-1 pruning in #757.
    """
    rng = np.random.default_rng(0)
    adj_p = rng.normal(size=(n_seg, 3))
    adj_w = rng.normal(size=n_seg)
    pos_res = float(
        np.max(np.abs(deriv.contract(adj_p, None).sum(axis=0) - adj_p.sum(axis=0)))
    )
    area_res = float(np.max(np.abs(deriv.contract(None, adj_w).sum(axis=0))))
    scale = max(float(np.max(np.abs(adj_p))), 1.0)
    return pos_res / scale, area_res


class FrameCavityDerivative:
    """The CFC chain rule composed with the derivative of its molecule-fixed frame.

    A cavity built with ``frame="molecular"`` (#769) is discretized in a
    coordinate system that is itself a function of the geometry, so an atom
    moving does two things: it changes the field, and it turns the frame -- and
    with the frame, the marching lattice and every segment direction.

    This class does **not** reimplement the chain for a turning lattice. It
    reverses the cavity into the frame it was built in, where the lattice is
    fixed and :class:`FineCavityDerivative` is exactly valid, and composes that
    with the frame's own derivative. Writing ``R`` for the frame (rows are its
    axes), ``c`` for the centre of the atom positions and ``p`` for a segment::

        pos_mol_B = R (R_B - c)          the construction's input
        p_lab_S   = R^T p_mol_S + c      its output, mapped back

    so with ``G[B] = dE/d(pos_mol_B)`` from the inner pass,

        dE/d(R_A) = sum_B G[B] . d(pos_mol_B)/d(R_A)
                    + sum_S (dE/dp_lab_S) . d(R^T p_mol_S + c)/d(R_A)

    and both remaining derivatives are the frame Jacobian plus ``dc/dR_A = I/n``.
    One chain rule for the construction, one for the frame, no third copy.

    The gain is not only correctness of the number. Because the energy is now a
    function of the internal geometry alone, the gradient satisfies an exact
    identity that no finite difference is needed to check: the torque vanishes.
    :meth:`rotation_residuals` is that check.
    """

    def __init__(self, cavity, params=None) -> None:
        from .fine_cavity import molecular_frame_jacobian, to_construction_frame

        self._rot = np.asarray(cavity.frame, dtype=np.float64)
        if self._rot.shape != (3, 3):
            raise ValueError(
                f"FrameCavityDerivative: expected a (3, 3) frame, got "
                f"{self._rot.shape}. Build the cavity with frame='molecular', or "
                f"use FineCavityDerivative for a lab-frame one."
            )
        self._pos = np.asarray(cavity.atom_positions, dtype=np.float64)
        self._n_atoms = int(self._pos.shape[0])
        inner_cavity = to_construction_frame(cavity)
        self._inner = FineCavityDerivative(inner_cavity, params)
        self._p_mol = np.asarray(inner_cavity.points, dtype=np.float64)
        self._p_lab = np.asarray(cavity.points, dtype=np.float64)
        self._n_seg = int(inner_cavity.n_points)
        # Raises for a linear solute, whose frame has no derivative at all --
        # here rather than at the first contraction, so the refusal arrives
        # before any number does.
        self._frame_jac = molecular_frame_jacobian(self._pos)

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    def contract(
        self,
        adj_position: Optional[np.ndarray] = None,
        adj_area: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reverse through the construction and then through the frame."""
        rot = self._rot
        jac = self._frame_jac
        n = self._n_atoms
        adj_p = (
            None if adj_position is None
            else np.asarray(adj_position, dtype=np.float64)
        )

        # The inner pass wants adjoints on the *construction*-frame positions:
        # p_lab = R^T p_mol, so dE/dp_mol = R dE/dp_lab.
        inner_adj_p = None if adj_p is None else adj_p @ rot.T
        g = self._inner.contract(inner_adj_p, adj_area)

        # d(pos_mol_B)_i/d(R_A)_c = sum_j dR_ij/d(R_A)_c (R_B - c)_j
        #                           + R_ic (delta_BA - 1/n)
        centred = self._pos - self._pos.mean(axis=0)
        out = np.einsum("Bi,ijAc,Bj->Ac", g, jac, centred)
        g_rot = g @ rot
        out = out + g_rot - g_rot.sum(axis=0)[None, :] / n

        if adj_p is not None:
            # The output map. Areas need no term here: a rotation does not
            # change them, which is why ``adj_area`` appears only above.
            out = out + np.einsum("Sj,ijAc,Si->Ac", adj_p, jac, self._p_mol)
            out = out + adj_p.sum(axis=0)[None, :] / n
        return out

    def translation_residuals(self) -> tuple[float, float]:
        """The two exact invariants; see :func:`_translation_residuals`."""
        return _translation_residuals(self, self._n_seg)

    def rotation_residuals(self) -> tuple[float, float]:
        """Zero torque, as a reverse-mode identity. The oracle for the frame term.

        A cavity built in a molecule-fixed frame turns rigidly with the solute,
        so along the rigid-rotation field ``dR_A = omega x R_A`` a segment moves
        by ``omega x p_S`` and its area does not move at all. Contracting the
        reverse pass against that field must therefore give

            sum_A (omega x R_A) . contract(P, 0)[A] = sum_S P_S . (omega x p_S)
            sum_A (omega x R_A) . contract(0, w)[A] = 0

        which is the rotational counterpart of
        :func:`_translation_residuals` and, like it, needs no displaced
        geometries and no tolerance. It is also the one check the frame term
        cannot pass by accident: drop it and the first residual is the size of
        the spurious torque the lab frame carries.

        Returned as ``(position, area)`` residuals, the first scaled by the
        adjoints and the geometry so it reads as a relative error.
        """
        rng = np.random.default_rng(1)
        adj_p = rng.normal(size=(self._n_seg, 3))
        adj_w = rng.normal(size=self._n_seg)
        omega = rng.normal(size=3)
        field = np.cross(omega[None, :], self._pos)

        got = float(np.einsum("Ac,Ac->", field, self.contract(adj_p, None)))
        want = float(
            np.einsum("Si,Si->", adj_p, np.cross(omega[None, :], self._p_lab))
        )
        area = float(np.einsum("Ac,Ac->", field, self.contract(None, adj_w)))
        scale = max(abs(want), 1.0)
        return abs(got - want) / scale, abs(area)


def cavity_derivative(cavity, *, switching_sigma_bohr: float = 0.5, params=None):
    """The :class:`CavityDerivative` for whichever construction built ``cavity``.

    Dispatches on type rather than on a flag so a cavity that grows a third
    construction cannot reach the assembler without an explicit derivative:
    the failure is an ``NotImplementedError`` naming the type, not a plausible
    wrong number.
    """
    from .fine_cavity import FineCavity

    if isinstance(cavity, FineCavity):
        if np.asarray(cavity.frame).size:
            return FrameCavityDerivative(cavity, params)
        return FineCavityDerivative(cavity, params)
    if hasattr(cavity, "point_atom") and hasattr(cavity, "switching"):
        return LebedevCavityDerivative(cavity, switching_sigma_bohr)
    raise NotImplementedError(
        f"cavity_derivative: no nuclear derivative is defined for "
        f"{type(cavity).__name__}. Add a CavityDerivative for it rather than "
        f"letting it fall through to another construction's chain rule."
    )
