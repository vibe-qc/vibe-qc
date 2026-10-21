"""Internal stability of a restricted real-Gamma torus solution (#519, RG-9).

A converged closed-shell SCF is a stationary point of the restricted energy.
It is a local minimum among real RHF determinants only when the real orbital
Hessian is positive definite. With canonical orbitals of the converged Fock
matrix and rotation parameters ``X`` (virtual by occupied), the two sectors of
Seeger and Pople, J. Chem. Phys. 66, 3045 (1977), doi:10.1063/1.434318, act as

    singlet (RHF -> RHF):  (A + B) X = de * X + C_v^T [2 J(dD) - K(dD)] C_o
    triplet (RHF -> UHF): 3(A + B) X = de * X - C_v^T K(dD) C_o

with ``de[a, i] = eps_a - eps_i`` and ``dD = C_v X C_o^T + C_o X^T C_v^T``,
the normalisation of the molecular drivers' ``stability_eigenvalue`` and of
the MSINDO CCM audit (Lehtola, Blockhuys and Van Alsenoy, Molecules 25, 1218
(2020), Sec. 10, doi:10.3390/molecules25051218). ``J`` and ``K`` are the
neutral torus operators the SCF ran on, ``K`` including the BvK Madelung seam
``xi S dD S``, so the Hessian is the second derivative of the energy the loop
minimised. ``dD`` is symmetric but indefinite; the exchange is applied through
two occupied-rank factors, ``dD = P - N`` with ``P = (Y + Z)(Y + Z)^T / 2`` and
``N = (Y - Z)(Y - Z)^T / 2`` for ``Y = C_v X`` and ``Z = C_o``.

The lowest eigenvalue of each sector comes from a Davidson solve
(E. R. Davidson, J. Comput. Phys. 17, 87 (1975),
doi:10.1016/0021-9991(75)90065-0) preconditioned by ``de``. The verdict
follows the molecular drivers: unstable when the lowest eigenvalue over both
sectors is below ``-tol``.

On a torus of ``N_c > 1`` cells a rotation can break the lattice translation
symmetry of the BvK torus. The translation-invariant rotations are those whose
``dD`` is block circulant. At a translation-invariant solution they form an
invariant subspace of the Hessian, the space in which the real-Gamma and
neutral-Bloch producers share their stationary points (Theorem 1). Every
sector reports the translation-invariant fraction of its lowest mode, and the
real-Gamma RHF driver follows the singlet sector inside that subspace only.
At ``N_c = 1`` every rotation is translation invariant.

Covered: the real-Gamma direct RHF driver. Kohn-Sham kernels and open-shell
references are not.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "STABILITY_TOL",
    "STABILITY_DESCENT_TOL",
    "STABILITY_MAX_MATVEC",
    "RESTART_ANGLES",
    "CCMStabilitySector",
    "CCMRHFStability",
    "analyse_rhf_stability",
    "ccm_rhf_stability",
    "circulant_projection",
    "descend_along_mode",
    "semicanonical_orbitals",
]

#: Instability threshold (Ha), the molecular RHF default (``RHFOptions``).
STABILITY_TOL = 1e-4
#: A scanned rotation must lower the energy by this much to count as descent
#: (the MSINDO CCM audit's resolution).
STABILITY_DESCENT_TOL = 1e-6
#: Rotation angles (radians along the unit mode) scanned for a restart seed,
#: as in the MSINDO CCM audit; both signs are tried.
RESTART_ANGLES = (0.1, 0.25, 0.5, 1.0, 2.0)
#: Matvec cap of one Davidson solve (the molecular default).
STABILITY_MAX_MATVEC = 60
#: Largest block-circulant deviation of a solution's density for which the
#: translation-invariant restriction is defined.
TRANSLATION_INVARIANCE_TOL = 1e-6

# vibeqc::StabilityStop codes (cpp/include/vibeqc/newton.hpp), as the
# molecular drivers expose them on ``stability_stop_reason``.
_STOP_CONVERGED, _STOP_MATVEC, _STOP_TRIVIAL, _STOP_STAGNATED = 0, 1, 4, 5


@dataclass(frozen=True)
class CCMStabilitySector:
    """The lowest eigenpair of one orbital-Hessian sector."""

    name: str                      # "singlet" | "triplet"
    lowest_eigenvalue: float       # Ha, normalisation of the module docstring
    converged: bool
    stop_reason: int               # vibeqc::StabilityStop code
    n_matvec: int
    residual_norm: float
    translation_invariant_fraction: float
    restricted_to_translation_invariant: bool
    mode: np.ndarray = field(repr=False, compare=False)   # (n_virt, n_occ), unit norm

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "lowest_eigenvalue": float(self.lowest_eigenvalue),
            "converged": bool(self.converged),
            "stop_reason": int(self.stop_reason),
            "n_matvec": int(self.n_matvec),
            "residual_norm": float(self.residual_norm),
            "translation_invariant_fraction":
                float(self.translation_invariant_fraction),
            "restricted_to_translation_invariant":
                bool(self.restricted_to_translation_invariant),
        }


@dataclass(frozen=True)
class CCMRHFStability:
    """Stability verdict of one converged real-Gamma RHF solution."""

    sectors: tuple
    tol: float
    energy: float                  # total torus energy of the analysed state
    frontier_gap: float            # LUMO - HOMO of the analysed state (Ha)
    wall_s: float
    operator: str                  # "dense" | "dense-from-block"
    #: Max |C_v^T F C_o| of the analysed state: its orbital gradient.
    orbital_gradient: float = 0.0
    #: ``(C_occ, C_virt)``: the semicanonical orbitals the modes refer to.
    orbitals: tuple = field(default=(), repr=False, compare=False)

    def sector(self, name):
        for s in self.sectors:
            if s.name == name:
                return s
        raise KeyError(name)

    @property
    def lowest_eigenvalue(self) -> float:
        return min(float(s.lowest_eigenvalue) for s in self.sectors)

    @property
    def converged(self) -> bool:
        return all(s.converged for s in self.sectors)

    @property
    def internal_instability(self) -> bool:
        """The molecular drivers' verdict: a converged analysis below ``-tol``."""
        return self.converged and self.lowest_eigenvalue < -self.tol

    @property
    def n_matvec(self) -> int:
        return sum(int(s.n_matvec) for s in self.sectors)

    def as_dict(self) -> dict:
        return {
            "method": ("real RHF orbital Hessian, Seeger-Pople singlet and "
                       "triplet sectors, Davidson"),
            "operator": self.operator,
            "tol": float(self.tol),
            "energy": float(self.energy),
            "frontier_gap": float(self.frontier_gap),
            "orbital_gradient": float(self.orbital_gradient),
            "lowest_eigenvalue": float(self.lowest_eigenvalue),
            "converged": bool(self.converged),
            "internal_instability": bool(self.internal_instability),
            "n_matvec": int(self.n_matvec),
            "wall_s": float(self.wall_s),
            "sectors": [s.as_dict() for s in self.sectors],
        }

    def stamp(self, result) -> None:
        """Set the molecular drivers' verdict fields on ``result``.

        :func:`vibeqc.output.verdict.verdict_from_result` reads them, so a
        real-Gamma result carries the stability vocabulary of a molecular
        one. ``stability_stop_reason`` is the first unconverged sector's stop
        code, else converged.
        """
        stops = [s.stop_reason for s in self.sectors if not s.converged]
        result.stability_checked = True
        result.stability_analysis_converged = bool(self.converged)
        result.stability_stop_reason = int(stops[0] if stops else _STOP_CONVERGED)
        result.stability_n_matvec = int(self.n_matvec)
        result.stability_wall_s = float(self.wall_s)
        result.stability_wall_budget_s = 0.0
        result.stability_eigenvalue = float(self.lowest_eigenvalue)
        result.internal_instability = bool(self.internal_instability)
        result.stability = self


def circulant_projection(nrep, M):
    """Translation average of a supercell matrix on the BvK torus.

    ``M`` is in the CCM cell-major AO order (cell ``(i, j, k)`` at linear
    index ``(i n_2 + j) n_3 + k``, ``n_mu`` AOs per cell). The result is the
    block-circulant matrix whose block at displacement ``g`` is the mean of
    the blocks of ``M`` at that displacement: the projection onto
    translation-invariant matrices, orthogonal in the Frobenius inner product
    and in ``Tr[S A S B]`` for a translation-invariant ``S``.
    """
    n1, n2, n3 = (int(n) for n in nrep)
    nc = n1 * n2 * n3
    M = np.asarray(M)
    if nc == 1:
        return M.copy()
    n = M.shape[0] // nc
    cells = np.array([(i, j, k) for i in range(n1) for j in range(n2)
                      for k in range(n3)])
    diff = cells[None, :, :] - cells[:, None, :]
    cls = (((diff[..., 0] % n1) * n2 + (diff[..., 1] % n2)) * n3
           + (diff[..., 2] % n3))
    blocks = M.reshape(nc, n, nc, n).transpose(0, 2, 1, 3)
    avg = np.zeros((nc, n, n), dtype=M.dtype)
    np.add.at(avg, cls.ravel(), blocks.reshape(nc * nc, n, n))
    avg /= nc
    return avg[cls].transpose(0, 2, 1, 3).reshape(nc * n, nc * n)


def _response(L, S, xi):
    """``(dD, J(dD), K(dD))`` of ``dD = Y Z^T + Z Y^T`` on the dense cderi."""
    from .ri import ccm_ri_j_neutral, ccm_ri_k_neutral_factored

    root = np.sqrt(0.5)

    def apply(Y, Z, *, coulomb=True):
        dD = Y @ Z.T
        dD = dD + dD.T
        K = (ccm_ri_k_neutral_factored(L, root * (Y + Z))
             - ccm_ri_k_neutral_factored(L, root * (Y - Z)))
        if xi:
            K = K + xi * (S @ dD @ S)
        J = ccm_ri_j_neutral(L, dD) if coulomb else None
        return dD, J, K

    return apply


def _davidson_lowest(matvec, diag, *, residual_tol, max_matvec, project=None,
                     n_start=4, max_space=40):
    """Lowest eigenpair of a symmetric operator by Davidson's method.

    ``diag`` is the diagonal preconditioner; ``project`` (optional) is the
    orthogonal projector onto an invariant subspace, applied to every trial
    vector and product. Returns ``(value, vector, converged, stop, n_matvec,
    residual_norm)``.
    """
    n = int(diag.size)
    if n == 0:
        return 0.0, np.zeros(0), True, _STOP_TRIVIAL, 0, 0.0
    proj = (lambda v: v) if project is None else project
    op = matvec if project is None else (lambda v: proj(matvec(v)))
    guesses = []
    for idx in np.argsort(diag, kind="stable")[: 8 * int(n_start)]:
        e = np.zeros(n)
        e[idx] = 1.0
        e = proj(e)
        if np.linalg.norm(e) > 1e-8:
            guesses.append(e)
        if len(guesses) == int(n_start):
            break
    if not guesses:
        guesses = [proj(np.random.default_rng(519).standard_normal(n))]
    V = np.zeros((n, 0))
    AV = np.zeros((n, 0))
    new = np.column_stack(guesses)
    n_mv = 0
    theta, x, rnorm = 0.0, np.zeros(n), np.inf
    while True:
        for _ in range(2):
            new = new - V @ (V.T @ new)
        q, r = np.linalg.qr(new)
        rd = np.abs(np.diag(r))
        q = q[:, rd > 1e-10 * max(1.0, float(rd.max(initial=0.0)))]
        if q.shape[1] == 0:
            return theta, x, False, _STOP_STAGNATED, n_mv, rnorm
        Aq = np.column_stack([op(q[:, j]) for j in range(q.shape[1])])
        n_mv += q.shape[1]
        V = np.hstack([V, q])
        AV = np.hstack([AV, Aq])
        H = V.T @ AV
        w, u = np.linalg.eigh(0.5 * (H + H.T))
        theta = float(w[0])
        x = V @ u[:, 0]
        ax = AV @ u[:, 0]
        r = ax - theta * x
        rnorm = float(np.linalg.norm(r))
        if rnorm < residual_tol:
            return theta, x, True, _STOP_CONVERGED, n_mv, rnorm
        if n_mv >= max_matvec:
            return theta, x, False, _STOP_MATVEC, n_mv, rnorm
        denom = diag - theta
        small = np.abs(denom) < 1e-8
        denom[small] = np.where(denom[small] < 0.0, -1e-8, 1e-8)
        t = proj(-r / denom)
        if V.shape[1] + 1 > max_space:
            # Thick restart on the current Ritz pair.
            V, AV = x[:, None], ax[:, None]
        new = t[:, None]


def semicanonical_orbitals(F, D, S, n_occ, *, lindep_tol=1e-7):
    """Orbitals of an evaluated closed-shell state ``(D, F[D])``.

    The occupied space is spanned by the ``n_occ`` natural orbitals of ``D``
    with the largest occupations in the metric ``S``, the virtual space by
    the rest of the linearly independent span; each block diagonalises the
    physical ``F``. The occupied space therefore reproduces ``D`` whatever
    produced it, and the eigenvalues are ``F[D]``'s. Returns ``(C_occ,
    eps_occ, C_virt, eps_virt)``.
    """
    from .scf import _evaluated_state_orbitals, _orthonormaliser

    eps, C = _evaluated_state_orbitals(
        np.asarray(F, dtype=float), np.asarray(D, dtype=float), S,
        _orthonormaliser(S, lindep_tol), n_occ)
    return C[:, :n_occ], eps[:n_occ], C[:, n_occ:], eps[n_occ:]


def analyse_rhf_stability(ccm, F, D, L, S, xi, *, energy,
                          tol=STABILITY_TOL, sectors=("singlet", "triplet"),
                          max_matvec=STABILITY_MAX_MATVEC, residual_tol=None,
                          translation_invariant=False, operator="dense",
                          lindep_tol=1e-7):
    """Lowest orbital-Hessian eigenpairs of the evaluated state ``(D, F)``.

    ``F`` is the physical Fock matrix of the closed-shell density ``D``,
    ``L`` the dense neutral torus cderi, ``S`` the torus overlap and ``xi``
    the Madelung seam coefficient the SCF used (0 without the seam). The
    orbitals are :func:`semicanonical_orbitals`; the Hessian of the module
    docstring assumes a stationary state, and ``orbital_gradient`` on the
    result says how far the state is from one.
    ``translation_invariant=True`` solves every sector inside the
    translation-invariant rotations; it needs a block-circulant solution.
    """
    t0 = time.perf_counter()
    F = np.asarray(F, dtype=float)
    D = np.asarray(D, dtype=float)
    n_occ = int(ccm.supercell.n_electrons()) // 2
    Co, eps_o, Cv, eps_v = semicanonical_orbitals(
        F, D, S, n_occ, lindep_tol=lindep_tol)
    nv, no = Cv.shape[1], Co.shape[1]
    SCo, SCv = S @ Co, S @ Cv
    de = (eps_v[:, None] - eps_o[None, :]).ravel()
    gap = (float(eps_v.min() - eps_o.max()) if nv and no else float("nan"))
    gradient = float(np.max(np.abs(Cv.T @ F @ Co), initial=0.0))
    nrep = tuple(int(n) for n in ccm.nrep)
    nc = int(np.prod(nrep))
    apply = _response(np.asarray(L), S, xi)
    if residual_tol is None:
        residual_tol = min(1e-5, 0.1 * float(tol))

    def dD_of(v):
        d = Cv @ v.reshape(nv, no) @ Co.T
        return d + d.T

    def s_norm2(M):
        # Tr[S M S M] for symmetric M and S.
        return float(np.sum((S @ M) * (M @ S)))

    def fraction(v):
        if nc == 1:
            return 1.0
        d = dD_of(v)
        total = s_norm2(d)
        return 1.0 if total == 0.0 else s_norm2(circulant_projection(nrep, d)) / total

    project = None
    if translation_invariant and nc > 1:
        deviation = float(np.max(np.abs(circulant_projection(nrep, D) - D)))
        if deviation > TRANSLATION_INVARIANCE_TOL:
            raise ValueError(
                "analyse_rhf_stability: the solution is not translation "
                f"invariant on the torus (density deviation {deviation:.3e}), "
                "so its translation-invariant rotations are not defined")

        def project(v):
            # X -> C_v^T S P(dD(X)) S C_o, the inverse of X -> dD(X) for
            # S-orthonormal C composed with the circulant projection.
            return (SCv.T @ circulant_projection(nrep, dD_of(v)) @ SCo).ravel()

    out = []
    for name in sectors:
        if name not in ("singlet", "triplet"):
            raise ValueError(f"analyse_rhf_stability: unknown sector {name!r}")
        singlet = name == "singlet"

        def matvec(v, singlet=singlet):
            _, J, K = apply(Cv @ v.reshape(nv, no), Co, coulomb=singlet)
            G = (2.0 * J - K) if singlet else -K
            return de * v + (Cv.T @ G @ Co).ravel()

        value, vector, converged, stop, n_mv, rnorm = _davidson_lowest(
            matvec, de, residual_tol=float(residual_tol),
            max_matvec=int(max_matvec), project=project)
        norm = float(np.linalg.norm(vector))
        mode = (vector / norm if norm > 0.0 else vector).reshape(nv, no)
        out.append(CCMStabilitySector(
            name=name, lowest_eigenvalue=float(value), converged=bool(converged),
            stop_reason=int(stop), n_matvec=int(n_mv), residual_norm=float(rnorm),
            translation_invariant_fraction=fraction(mode.ravel()),
            restricted_to_translation_invariant=project is not None,
            mode=mode))
    return CCMRHFStability(
        sectors=tuple(out), tol=float(tol), energy=float(energy),
        frontier_gap=gap, wall_s=time.perf_counter() - t0, operator=operator,
        orbital_gradient=gradient, orbitals=(Co, Cv))


def ccm_rhf_stability(result, ccm, *, tol=STABILITY_TOL,
                      sectors=("singlet", "triplet"),
                      max_matvec=STABILITY_MAX_MATVEC, residual_tol=None,
                      translation_invariant=False):
    """Stability of a converged real-Gamma RHF result on its torus.

    ``result`` is a :class:`~vibeqc.periodic.ccm.scf.CCMSCFResult` of
    :func:`~vibeqc.periodic.ccm.direct.run_ccm_rhf_direct` that still carries
    the cderi it converged on (``result.cderi``). Block storage is expanded
    to the dense array: the block operators act on the circulant projection
    of a density, which drops the response of a rotation that breaks
    translation symmetry. Returns a :class:`CCMRHFStability`;
    :meth:`CCMRHFStability.stamp` sets the molecular verdict fields.
    """
    from vibeqc.periodic.exchange_convention import BVK_EWALD

    from .block_cderi import is_block_cderi
    from .direct import ccm_exchange_q0_madelung

    L = getattr(result, "cderi", None)
    if L is None:
        raise ValueError(
            "ccm_rhf_stability: the result carries no cderi; the analysis "
            "needs the operator the SCF converged on (result.cderi)")
    if not bool(getattr(result, "converged", False)):
        raise ValueError(
            "ccm_rhf_stability: the SCF did not converge; a stability "
            "verdict needs a stationary point")
    operator = "dense"
    if is_block_cderi(L):
        L, operator = L.to_dense(), "dense-from-block"
    S = np.asarray(result.overlap, dtype=float)
    xi = (ccm_exchange_q0_madelung(ccm)
          if getattr(result, "exchange_q0", None) == BVK_EWALD else 0.0)
    return analyse_rhf_stability(
        ccm, result.fock, result.density, L, S, xi,
        energy=float(result.energy), tol=tol, sectors=sectors,
        max_matvec=max_matvec, residual_tol=residual_tol,
        translation_invariant=translation_invariant, operator=operator)


def descend_along_mode(orbitals, mode, energy_of, *, e0,
                       angles=RESTART_ANGLES,
                       descent_tol=STABILITY_DESCENT_TOL,
                       project_density=None):
    """Rotate the occupied space along ``mode`` and keep the lowest energy.

    ``orbitals`` is ``(C_occ, C_virt)``, the orbitals the mode refers to
    (:attr:`CCMRHFStability.orbitals`). The rotation ``exp(theta kappa)``
    with ``kappa_vo = mode`` is applied
    exactly through the SVD of the mode, as in the MSINDO CCM audit. Both
    signs of every angle in ``angles`` are evaluated with ``energy_of(D)``.
    Returns ``(best, scan)``: ``best`` is ``(energy, density, angle)`` of the
    lowest rotated density when it lies at least ``descent_tol`` below
    ``e0``, else ``None``; ``scan`` lists every ``(angle, energy)``.
    ``project_density`` (optional) maps each rotated density before its
    evaluation, e.g. the circulant projection.
    """
    Co, Cv = (np.asarray(c, dtype=float) for c in orbitals)
    u, s, vt = np.linalg.svd(np.asarray(mode, dtype=float), full_matrices=False)
    best, scan = None, []
    for sign in (1.0, -1.0):
        for theta in angles:
            a = sign * float(theta) * s
            occ = (Co + (Co @ vt.T * (np.cos(a) - 1.0)) @ vt
                   + (Cv @ u * np.sin(a)) @ vt)
            D = 2.0 * occ @ occ.T
            if project_density is not None:
                D = project_density(D)
            D = 0.5 * (D + D.T)
            E = float(energy_of(D))
            scan.append((sign * float(theta), E))
            if best is None or E < best[0]:
                best = (E, D, sign * float(theta))
    if best is None or best[0] > float(e0) - float(descent_tol):
        return None, tuple(scan)
    return best, tuple(scan)
