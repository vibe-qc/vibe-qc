"""Orbital Transformation (OT) SCF solver for the GAPW periodic route.

Avoids diagonalisation by directly minimising the total energy with respect
to the orbital coefficients under the orthonormality constraint, using an
exponential parametrisation:

    C_new = C . exp(κ)

where ``κ`` is an antisymmetric matrix (κ = -κ^T). The energy is minimised
via conjugate gradients (CG) over the independent parameters (the occ-virt
block of κ).

References
----------
VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003).
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

from . import _vibeqc_core as _core
from .periodic_screened_exchange import reject_unscreened_range_separated
from .progress import resolve_progress
from .periodic_gapw_grid import GAPWExperimentalWarning, PlaneWaveGrid
from .periodic_gapw_j import (
    GpwJBuilder,
    _build_v_ne,
    _eigh_safe,
    _evaluate_xc_on_grid,
    _kinetic_lattice_gamma,
    _overlap_lattice_gamma,
    _project_vtau_to_ao,
    _project_vxc_to_ao,
    _warn_if_non_neutral,
    collocate_density_on_grid,
    evaluate_gpw_energy,
    project_potential_to_ao,
)

__all__ = [
    "OrbitalTransformation",
    "run_ot_rhf_gapw",
    "run_ot_rks_gapw",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cayley_transform(kappa: np.ndarray) -> np.ndarray:
    """Cayley transform of an antisymmetric matrix.

    ``U = (I - κ/2)^{-1} (I + κ/2)`` is unitary when κ is real
    antisymmetric.  Cheaper than :func:`scipy.linalg.expm` and
    preserves orthonormality exactly.

    Parameters
    ----------
    kappa
        ``(n, n)`` real antisymmetric matrix.

    Returns
    -------
    U
        ``(n, n)`` unitary matrix.
    """
    n = kappa.shape[0]
    I = np.eye(n, dtype=kappa.dtype)
    half_k = 0.5 * kappa
    return np.linalg.solve(I - half_k, I + half_k)


def _expm(kappa: np.ndarray) -> np.ndarray:
    """Matrix exponential of an antisymmetric matrix.

    Falls back to :func:`scipy.linalg.expm` when available; otherwise
    uses the Cayley approximation.

    Parameters
    ----------
    kappa
        ``(n, n)`` real antisymmetric matrix.

    Returns
    -------
    U
        ``(n, n)`` unitary matrix.
    """
    try:
        from scipy.linalg import expm

        return expm(kappa)
    except ImportError:
        return _cayley_transform(kappa)


def _pack_kappa(kappa_ov: np.ndarray) -> np.ndarray:
    """Flatten occ-virt block of the antisymmetric κ matrix into a
    1-D parameter vector.

    Only the occ-virt block is independent (occ-occ and virt-virt
    rotations leave the density invariant for closed-shell systems;
    the virt-occ block is the transpose of occ-virt).

    Parameters
    ----------
    kappa_ov
        ``(n_occ, n_virt)`` occupied-virtual block of κ.

    Returns
    -------
    vec
        ``(n_occ * n_virt,)`` 1-D parameter vector.
    """
    return kappa_ov.ravel()


def _unpack_kappa(vec: np.ndarray, n_occ: int, n_virt: int) -> np.ndarray:
    """Reshape a 1-D parameter vector into the occ-virt block and
    embed it in a full ``(n_basis, n_basis)`` antisymmetric κ matrix.

    Parameters
    ----------
    vec
        ``(n_occ * n_virt,)`` parameter vector.
    n_occ
        Number of occupied orbitals.
    n_virt
        Number of virtual orbitals.

    Returns
    -------
    kappa
        ``(n_basis, n_basis)`` real antisymmetric matrix with zero
        occ-occ and virt-virt blocks.
    """
    n_basis = n_occ + n_virt
    kappa_ov = vec.reshape(n_occ, n_virt)
    kappa = np.zeros((n_basis, n_basis), dtype=float)
    kappa[:n_occ, n_occ:] = kappa_ov
    kappa[n_occ:, :n_occ] = -kappa_ov.T
    return kappa


# ---------------------------------------------------------------------------
# OrbitalTransformation class
# ---------------------------------------------------------------------------


class OrbitalTransformation:
    """Orbital transformation (OT) SCF solver.

    Minimises ``E(C)`` subject to ``C^T S C = I`` by parametrising
    orbital rotations via an antisymmetric matrix ``κ``:

        C_new(κ) = C . exp(κ)

    and optimising ``E(κ)`` with conjugate gradients.  No
    diagonalisation is performed -- the Fock matrix is never
    eigendecomposed.

    Parameters
    ----------
    basis
        :class:`vibeqc.BasisSet`.
    system
        :class:`vibeqc._vibeqc_core.PeriodicSystem`.
    grid
        :class:`PlaneWaveGrid` for the GPW Hartree-J build.
    functional
        Optional functional name (e.g. ``"lda"``, ``"pbe"``).
        ``None`` for pure HF.
    gapw_kwargs
        Extra keyword arguments forwarded to the JK / energy builders.
    method
        Optimisation method. ``"cg"`` (conjugate gradients) is the
        only currently implemented option.
    max_iter
        Maximum number of OT iterations.
    conv_tol
        Convergence threshold on the gradient norm
        ``||gradE||_F / sqrt(n_param)``.
    """

    def __init__(
        self,
        basis,
        system,
        grid: PlaneWaveGrid,
        functional: Optional[str] = None,
        gapw_kwargs: Optional[dict] = None,
        method: str = "cg",
        max_iter: int = 100,
        conv_tol: float = 1e-8,
    ) -> None:
        if method != "cg":
            raise ValueError(
                f"OrbitalTransformation: method={method!r} is not "
                f"implemented. Only 'cg' is supported."
            )

        kwargs = dict(gapw_kwargs or {})
        self._v_ne_convention = kwargs.pop("v_ne_convention", "ewald")
        self._smearing_alpha = kwargs.pop("smearing_alpha", None)
        if kwargs:
            raise ValueError(
                f"OrbitalTransformation: unrecognised gapw_kwargs: "
                f"{list(kwargs.keys())}"
            )

        self.basis = basis
        self.system = system
        self.grid = grid
        self.functional = functional
        self.method = method
        self.max_iter = int(max_iter)
        self.conv_tol = float(conv_tol)

        # Validate cell.
        if system.dim != 3:
            raise ValueError(
                f"OrbitalTransformation: only dim == 3 is supported "
                f"(got dim={system.dim})"
            )
        n_elec = sum(int(a.Z) for a in system.unit_cell)
        if n_elec % 2 != 0:
            raise ValueError(
                f"OrbitalTransformation: cell has {n_elec} electrons "
                f"(odd); closed-shell OT needs an even count."
            )
        self.n_elec = n_elec
        self.n_occ = n_elec // 2
        self.n_basis = basis.nbasis
        self.n_virt = self.n_basis - self.n_occ

        # Pre-build one-electron integrals (Γ-point).
        self._T = _kinetic_lattice_gamma(basis, system)
        self._V_ne = _build_v_ne(
            basis,
            system,
            self._v_ne_convention,
            self._smearing_alpha,
            grid,
        )
        self._S = _overlap_lattice_gamma(basis, system)
        self._Hcore = self._T + self._V_ne
        self._E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))

        # Pre-build orthogonaliser S^{-1/2} and S^{1/2}.
        self._build_orthogonalizer()

        # JK builders (J via GPW, K via Γ-only molecular-limit image sum).
        self._gpw = GpwJBuilder(basis, grid)
        self._lo = _core.LatticeSumOptions()
        self._lo.cutoff_bohr = 25.0

        # Functional setup.
        if functional is not None:
            self._func = _core.Functional(functional, 1)
            # The OT-SCF K build is full-range only.
            reject_unscreened_range_separated(
                self._func, where="OrbitalTransformation"
            )
            self._ex_frac = float(self._func.hf_exchange_fraction)
        else:
            self._func = None
            self._ex_frac = 1.0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_orthogonalizer(self) -> None:
        """Build ``S^{-1/2}`` and ``S^{1/2}`` from the cached overlap."""
        eigs, Ut = _eigh_safe(self._S)
        self._S_half_inv = Ut @ np.diag(1.0 / np.sqrt(eigs)) @ Ut.T
        self._S_half = Ut @ np.diag(np.sqrt(eigs)) @ Ut.T

    def _orthonormal_mo_set(self, C: np.ndarray) -> np.ndarray:
        """Return the MO coefficients in the canonical orthonormal basis:
        ``C̃ = S^{1/2} . C`` satisfies ``C̃^T C̃ = I``.
        """
        return self._S_half @ C

    def _ao_from_orthonormal_mo(self, C_tilde: np.ndarray) -> np.ndarray:
        """Transform orthonormal MOs back to the AO basis:
        ``C = S^{-1/2} @ C̃``.
        """
        return self._S_half_inv @ C_tilde

    def _density_from_orthonormal(self, C_tilde: np.ndarray) -> np.ndarray:
        """Closed-shell AO density ``D = 2 . (S^{-1/2} C̃_occ) @ (S^{-1/2} C̃_occ)^T``."""
        C_occ = self._S_half_inv @ C_tilde[:, : self.n_occ]
        return 2.0 * C_occ @ C_occ.T

    def _density_in_orthonormal_basis(self, D: np.ndarray) -> np.ndarray:
        """Transform AO density to the orthonormal basis: ``D̃ = S^{1/2} . D . S^{1/2}``."""
        return self._S_half @ D @ self._S_half

    # ------------------------------------------------------------------ #
    # Fock and energy evaluation
    # ------------------------------------------------------------------ #

    def _compute_fock_and_energy(
        self, D: np.ndarray
    ) -> Tuple[float, np.ndarray, float]:
        """Build the Fock matrix and total energy from an AO density.

        Parameters
        ----------
        D
            ``(n_basis, n_basis)`` closed-shell AO density matrix.

        Returns
        -------
        E
            Total energy (electronic + nuclear) in Hartree.
        F
            ``(n_basis, n_basis)`` Fock matrix in the AO basis.
        e_xc
            Exchange-correlation energy (0.0 for pure HF).
        """
        J = self._gpw.build_J(D)
        if self._ex_frac > 0.0:
            jk_mol = _core.build_jk_gamma_molecular_limit(
                self.basis,
                self.system,
                self._lo,
                D,
            )
            K = np.asarray(jk_mol.K)
        else:
            K = np.zeros_like(D)
        F = self._Hcore + J - 0.5 * self._ex_frac * K

        e_xc = 0.0
        if self._func is not None:
            rho_grid = collocate_density_on_grid(self.basis, D, self.grid)
            # basis + density_matrix let meta-GGA functionals build t(r);
            # LDA/GGA ignore them.
            e_xc, v_xc_grid, v_sigma_grid, grad_rho, v_tau_grid = _evaluate_xc_on_grid(
                rho_grid, self.grid, self._func,
                basis=self.basis, density_matrix=D,
            )
            V_xc = _project_vxc_to_ao(
                self.basis,
                v_xc_grid,
                self.grid,
                v_sigma_grid=v_sigma_grid,
                grad_rho=grad_rho,
            )
            if v_tau_grid is not None:
                # t Fock term: 1/2 ∫ v_tau gradchi.gradchi (see
                # periodic_gapw_j._project_vtau_to_ao).
                V_tau = _project_vtau_to_ao(self.basis, v_tau_grid, self.grid)
                V_xc = V_xc + V_tau
            F = F + V_xc

        # Total energy -- formula follows run_periodic_rhf_gpw.
        if self._func is None:
            # Pure HF: E_elec = 0.5 tr(D (Hcore + F)).
            E_elec = 0.5 * float(np.einsum("ij,ij->", D, self._Hcore + F))
            E = E_elec + self._E_nn
        else:
            # DFT: E = tr(D.Hcore) + 1/2 tr(D.J) - 1/4 ex.tr(D.K) + E_xc.
            E_elec = (
                float(np.einsum("ij,ij->", D, self._Hcore))
                + 0.5 * float(np.einsum("ij,ij->", D, J))
                - 0.25 * self._ex_frac * float(np.einsum("ij,ij->", D, K))
                + e_xc
            )
            E = E_elec + self._E_nn

        return E, F, e_xc

    # ------------------------------------------------------------------ #
    # Gradient
    # ------------------------------------------------------------------ #

    def _energy_and_grad(
        self, C_tilde: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """Evaluate the total energy and orbital-rotation gradient.

        Parameters
        ----------
        C_tilde
            ``(n_basis, n_basis)`` MO coefficients in the **orthonormal**
            basis (``C̃ = S^{1/2} . C``).

        Returns
        -------
        E
            Total energy in Hartree.
        grad_vec
            ``(n_param,)`` flattened gradient in parameter space.
        kappa
            ``(n_basis, n_basis)`` full antisymmetric gradient matrix
            ``G̃ = 4 . (F̃ . D̃ - D̃ . F̃)`` in the orthonormal basis.
        e_xc
            Exchange-correlation energy (0.0 for pure HF).
        """
        # AO density from current MOs.
        D = self._density_from_orthonormal(C_tilde)
        # Warn if the density is significantly non-neutral.
        _warn_if_non_neutral(D, self.basis, self.system, quiet=True)

        E, F_ao, e_xc = self._compute_fock_and_energy(D)

        # F̃ = S^{-1/2} . F . S^{-1/2}
        F_tilde = self._S_half_inv @ F_ao @ self._S_half_inv

        # D̃ = S^{1/2} . D . S^{1/2}  (density in orthonormal basis)
        D_tilde = self._density_in_orthonormal_basis(D)

        # Antisymmetric gradient in orthonormal basis.
        # G̃ = 4 . (F̃ . D̃ - D̃ . F̃)
        G_tilde = 4.0 * (F_tilde @ D_tilde - D_tilde @ F_tilde)

        # Extract independent parameters: the occ-virt block.
        kappa_ov = G_tilde[: self.n_occ, self.n_occ :]

        grad_vec = _pack_kappa(kappa_ov)
        return E, grad_vec, G_tilde, e_xc

    # ------------------------------------------------------------------ #
    # Line search
    # ------------------------------------------------------------------ #

    def _line_search(
        self,
        C_tilde: np.ndarray,
        direction: np.ndarray,
        E_current: float,
        grad_vec: np.ndarray,
        *,
        c1: float = 1e-4,
        max_backtracks: int = 20,
        alpha_init: float = 0.1,
    ) -> Tuple[float, np.ndarray]:
        """Simple backtracking line search along ``direction``.

        Finds ``a`` that satisfies the Armijo sufficient-decrease
        condition:

            E(a) <= E(0) + c1 . a . <g(0), d>

        Parameters
        ----------
        C_tilde
            Current orthonormal MO coefficients.
        direction
            ``(n_param,)`` search direction (CG or steepest-descent).
        E_current
            Energy at the current point ``E(0)``.
        grad_vec
            Gradient at the current point ``g(0)``.
        c1
            Armijo condition constant.
        max_backtracks
            Maximum backtracking steps before accepting.
        alpha_init
            Initial step size.

        Returns
        -------
        alpha
            Accepted step size.
        C_tilde_new
            Updated orthonormal MO coefficients.
        """
        n_occ, n_virt = self.n_occ, self.n_virt
        # Directional derivative <g, d>.
        dg = float(np.dot(grad_vec, direction))
        if dg >= 0.0:
            # Not a descent direction -- fall back to steepest descent.
            direction = -grad_vec
            dg = float(np.dot(grad_vec, direction))
            if dg >= 0.0:
                return 0.0, C_tilde

        alpha = alpha_init
        for _ in range(max_backtracks):
            # Build κ from the scaled direction.
            kappa = _unpack_kappa(alpha * direction, n_occ, n_virt)

            # Apply exponential (or Cayley) rotation.
            U = _expm(-kappa)
            C_tilde_trial = C_tilde @ U

            # Evaluate energy at trial point.
            D_trial = self._density_from_orthonormal(C_tilde_trial)
            E_trial, _F_trial, _ = self._compute_fock_and_energy(D_trial)

            # Armijo condition.
            if E_trial <= E_current + c1 * alpha * dg:
                return alpha, C_tilde_trial
            alpha *= 0.5

        # Fall back -- accept whatever we have after max backtracks.
        kappa = _unpack_kappa(alpha * direction, n_occ, n_virt)
        U = _expm(-kappa)
        C_tilde_new = C_tilde @ U
        return alpha, C_tilde_new

    # ------------------------------------------------------------------ #
    # CG step
    # ------------------------------------------------------------------ #

    def _step_cg(
        self,
        C_tilde: np.ndarray,
        grad_vec: np.ndarray,
        direction_prev: Optional[np.ndarray],
        grad_prev: Optional[np.ndarray],
        iteration: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One conjugate-gradient step.

        Parameters
        ----------
        C_tilde
            Current orthonormal MO coefficients.
        grad_vec
            Gradient at the current point.
        direction_prev
            Previous search direction (``None`` for first iteration).
        grad_prev
            Previous gradient (``None`` for first iteration).
        iteration
            0-based iteration index.

        Returns
        -------
        C_tilde_new
            Updated orthonormal MO coefficients.
        direction
            New search direction.
        direction_prev
            Updated previous direction (for the next call).
        grad_prev
            Updated previous gradient (for the next call).
        """
        if iteration == 0:
            # Steepest-descent first step.
            direction = -grad_vec
        else:
            # Polak-Ribière CG beta.
            diff = grad_vec - grad_prev
            gg = float(np.dot(grad_prev, grad_prev))
            beta_pr = float(np.dot(grad_vec, diff)) / gg if gg > 0.0 else 0.0
            beta = max(0.0, beta_pr)  # PR+ (non-negative).
            direction = -grad_vec + beta * direction_prev

        # Line search.
        _alpha, C_tilde_new = self._line_search(
            C_tilde, direction, self._E_prev if iteration > 0 else 1e10, grad_vec
        )

        return C_tilde_new, direction, direction_prev

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    def run(
        self,
        C_init: Optional[np.ndarray] = None,
        *,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run the OT SCF minimisation.

        Parameters
        ----------
        C_init
            Optional ``(n_basis, n_basis)`` initial MO coefficients in
            the **AO basis** (S-orthonormal: ``C^T S C = I``).  If
            ``None``, the Hcore guess is used.
        verbose
            If True, emit live progress per iteration.

        Returns
        -------
        C
            ``(n_basis, n_basis)`` converged MO coefficients in the
            AO basis.
        D
            ``(n_basis, n_basis)`` converged AO density matrix.
        mo_energies
            ``(n_basis,)`` MO energies (diagonal of ``F`` in the
            MO basis) -- only meaningful for post-hoc analysis since
            the OT method does not diagonalise the Fock matrix.
        info
            Convergence metadata dict with keys:
            ``"converged"``, ``"n_iter"``, ``"E_total"``,
            ``"grad_norm"``, ``"scf_trace"``.
        """
        # Initial MOs.
        if C_init is None:
            # Hcore guess: diagonalise Hcore in the orthonormal AO basis.
            e, C_orth = _eigh_safe(self._S_half_inv @ self._Hcore @ self._S_half_inv)
            C = self._S_half_inv @ C_orth
        else:
            C = np.asarray(C_init, dtype=float).copy()
            if C.shape != (self.n_basis, self.n_basis):
                raise ValueError(
                    f"C_init has shape {C.shape}; expected "
                    f"({self.n_basis}, {self.n_basis})"
                )

        # Transform to orthonormal basis.
        C_tilde = self._orthonormal_mo_set(C)

        # OT iteration.
        direction_prev: Optional[np.ndarray] = None
        grad_prev: Optional[np.ndarray] = None
        self._E_prev: float = 0.0
        converged = False
        n_iter = 0
        scf_trace: list[dict] = []
        plog = resolve_progress(bool(verbose), verbose=2)

        for it in range(self.max_iter):
            n_iter = it + 1
            E, grad_vec, G_tilde, e_xc = self._energy_and_grad(C_tilde)

            # Gradient norm (RMS of the independent parameters).
            n_param = grad_vec.size
            grad_norm = float(np.linalg.norm(grad_vec)) / max(
                1.0, np.sqrt(float(n_param))
            )

            plog.write_raw(
                f"OT iter {it + 1:3d}:  E = {E:+.12f}  "
                f"|g|_rms = {grad_norm:.2e}"
            )

            scf_trace.append(
                {
                    "iter": it + 1,
                    "energy": float(E),
                    "delta_e": float(E - self._E_prev) if it > 0 else 0.0,
                    "grad_norm": float(grad_norm),
                    "e_xc": float(e_xc),
                }
            )

            conv = grad_norm < self.conv_tol
            if it > 0 and conv:
                converged = True
                plog.write_raw(f"  -> Converged in {it + 1} iterations.")
                break

            # Build the CG direction and take a step.
            C_tilde, direction_prev, grad_prev = self._step_cg(
                C_tilde,
                grad_vec,
                direction_prev,
                grad_prev,
                it,
            )
            self._E_prev = E

            if it == self.max_iter - 1 and not converged:
                plog.write_raw(
                    f"  -> NOT converged after {self.max_iter} "
                    f"iterations. grad_norm = {grad_norm:.2e}"
                )

        # Transform back to AO basis.
        C_final = self._ao_from_orthonormal_mo(C_tilde)
        D_final = self._density_from_orthonormal(C_tilde)

        # Post-hoc MO energies: diagonal of F in the MO basis.
        _, F_ao, _ = self._compute_fock_and_energy(D_final)
        # C^T F C in the orthonormal basis -> eigenvalues are canonical
        # MO energies.
        F_tilde = self._S_half_inv @ F_ao @ self._S_half_inv
        # Approximate MO energies via Rayleigh quotient in the occupied
        # subspace (for occupied) and the virtual subspace (for virtual).
        # Since OT does not diagonalise, we compute the diagonal
        # elements of C^T F C for the post-hoc spectrum.
        mo_energies = np.array(
            [
                float(C_tilde[:, i] @ F_tilde @ C_tilde[:, i])
                for i in range(self.n_basis)
            ]
        )

        info = {
            "converged": converged,
            "n_iter": n_iter,
            "E_total": float(E) if converged else float(scf_trace[-1]["energy"]),
            "grad_norm": float(grad_norm),
            "scf_trace": scf_trace,
        }

        return C_final, D_final, mo_energies, info


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------


def _run_ot_gapw(
    system,
    basis,
    *,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    functional: Optional[str] = None,
    method: str = "cg",
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    gapw_kwargs: Optional[dict] = None,
    quiet: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Shared OT entry point used by both RHF and RKS convenience wrappers.

    Returns ``(C, D, mo_energies, info)`` -- see
    :meth:`OrbitalTransformation.run` for details.
    """
    if not quiet:
        warnings.warn(
            "periodic_gapw_ot: Orbital Transformation SCF is experimental. "
            "The OT solver avoids diagonalisation and directly minimises "
            "the energy -- use for systems where traditional SCF "
            "oscillates or fails to converge.",
            category=GAPWExperimentalWarning,
            stacklevel=3,
        )

    if system.dim != 3:
        raise ValueError(
            f"_run_ot_gapw: only dim == 3 is supported (got dim={system.dim})"
        )

    n_elec = sum(int(a.Z) for a in system.unit_cell)
    if n_elec % 2 != 0:
        raise ValueError(
            f"_run_ot_gapw: cell has {n_elec} electrons (odd); "
            f"closed-shell OT needs an even count."
        )

    # Build grid if not supplied.
    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid

        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(
            np.asarray(system.lattice, dtype=float),
            cutoff_ha=cutoff_ha,
        )

    solver = OrbitalTransformation(
        basis,
        system,
        grid,
        functional=functional,
        gapw_kwargs=gapw_kwargs,
        method=method,
        max_iter=max_iter,
        conv_tol=conv_tol,
    )
    return solver.run(verbose=not quiet)


def run_ot_rhf_gapw(
    system,
    basis,
    *,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    method: str = "cg",
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    gapw_kwargs: Optional[dict] = None,
    quiet: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """OT-driven closed-shell RHF on a periodic cell via the GAPW route.

    Parameters
    ----------
    system
        :class:`vibeqc._vibeqc_core.PeriodicSystem`.
    basis
        :class:`vibeqc.BasisSet`.
    grid
        Pre-built :class:`PlaneWaveGrid`, or ``None`` to build one
        from ``cutoff_ha``.
    cutoff_ha
        Plane-wave cutoff (Ha) when ``grid`` is ``None``.
    method
        Optimisation method -- ``"cg"`` (conjugate gradients).
    max_iter
        Maximum OT iterations.
    conv_tol
        Gradient RMS convergence threshold.
    gapw_kwargs
        Extra keyword arguments for the OT solver:
        ``v_ne_convention``, ``smearing_alpha``.
    quiet
        Suppress progress and warning output.

    Returns
    -------
    C
        ``(n_basis, n_basis)`` MO coefficients in the AO basis.
    D
        ``(n_basis, n_basis)`` converged AO density matrix.
    mo_energies
        ``(n_basis,)`` post-hoc MO energies (Rayleigh quotients).
    info
        Convergence metadata dict.
    """
    return _run_ot_gapw(
        system,
        basis,
        grid=grid,
        cutoff_ha=cutoff_ha,
        functional=None,
        method=method,
        max_iter=max_iter,
        conv_tol=conv_tol,
        gapw_kwargs=gapw_kwargs,
        quiet=quiet,
    )


def run_ot_rks_gapw(
    system,
    basis,
    *,
    functional: str,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    method: str = "cg",
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    gapw_kwargs: Optional[dict] = None,
    quiet: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """OT-driven closed-shell RKS on a periodic cell via the GAPW route.

    Parameters
    ----------
    system
        :class:`vibeqc._vibeqc_core.PeriodicSystem`.
    basis
        :class:`vibeqc.BasisSet`.
    functional
        XC functional name (e.g. ``"lda"``, ``"pbe"``, ``"blyp"``).
        **Required** -- for Hartree-Fock use :func:`run_ot_rhf_gapw`.
    grid
        Pre-built :class:`PlaneWaveGrid`, or ``None`` to build one
        from ``cutoff_ha``.
    cutoff_ha
        Plane-wave cutoff (Ha) when ``grid`` is ``None``.
    method
        Optimisation method -- ``"cg"`` (conjugate gradients).
    max_iter
        Maximum OT iterations.
    conv_tol
        Gradient RMS convergence threshold.
    gapw_kwargs
        Extra keyword arguments for the OT solver.
    quiet
        Suppress progress and warning output.

    Returns
    -------
    C
        ``(n_basis, n_basis)`` MO coefficients in the AO basis.
    D
        ``(n_basis, n_basis)`` converged AO density matrix.
    mo_energies
        ``(n_basis,)`` post-hoc MO energies (Rayleigh quotients).
    info
        Convergence metadata dict.
    """
    if not functional:
        raise ValueError(
            "run_ot_rks_gapw requires a functional= (e.g. 'lda', 'pbe', "
            "'b3lyp'); for Hartree-Fock use run_ot_rhf_gapw()."
        )
    return _run_ot_gapw(
        system,
        basis,
        grid=grid,
        cutoff_ha=cutoff_ha,
        functional=functional,
        method=method,
        max_iter=max_iter,
        conv_tol=conv_tol,
        gapw_kwargs=gapw_kwargs,
        quiet=quiet,
    )
