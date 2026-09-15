"""Variational two-electron reduced density matrix (v2RDM) solver.

The production route returns the exact ensemble-N-representable 1/2-RDM for
the supplied active-space Hamiltonian by diagonalising vibe-qc's own
determinant-space Hamiltonian and contracting the resulting CI vector.  This is
the finite-active-space optimum of the v2RDM energy functional,

    E[¹D, ^2D] = S_{ij} h_{ij} ¹D_{ij}
               + 1/2 S_{pqrs} (pq|rs) ^2D_{pqrs}
               + E_nuc,

within the exact N-representable set.  It replaces the former P-only
augmented-Lagrangian projector, which could converge to stationary but
incorrect energies while reporting success.

Q and G residual builders remain available for diagnostics, but requesting
``constraints`` containing ``"q"`` or ``"g"`` still raises until feedback from
those PSD cones is implemented as a genuine SDP route.

References
----------
* Mazziotti, *Phys. Rev. Lett.* 93, 213001 (2004).
* Mazziotti, *Acc. Chem. Res.* 39, 207 (2006).
* DePrince, *J. Chem. Phys.* 145, 164109 (2016).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import eigh

from vibeqc.output import write

from ._common import Hamiltonian, SolverOptions, SolverResult
from ._determinant import generate_determinants
from ._rdm import energy_from_rdms, make_rdm12
from ._slater_condon import build_hamiltonian_matrix_unrestricted


@dataclass
class V2RDMOptions(SolverOptions):
    """Options for the v2RDM solver.

    The production route accepts ``constraints="p"`` and returns the exact
    fixed-active-space N-representable 1/2-RDM from determinant
    diagonalisation.  Q and G constraint builders exist for diagnostics, but
    feedback is not yet implemented -- a NotImplementedError is raised if
    ``"q"`` or ``"g"`` appear in the constraints string.
    """

    constraints: str = "p"
    mu: float = 10.0
    mu_factor: float = 1.2
    mu_max: float = 1e8
    outer_max_iter: int = 500
    conv_tol_primal: float = 1e-6
    conv_tol_dual: float = 1e-6


class V2RDMSolver:
    """Variational 2-RDM solver over the exact determinant N-representable set."""

    def __init__(self, options: Optional[V2RDMOptions] = None):
        self.options = options or V2RDMOptions()

    def solve(
        self,
        hamiltonian: Hamiltonian,
        options: Optional[V2RDMOptions] = None,
    ) -> SolverResult:
        opts = options or self.options
        h1e = hamiltonian.h1e
        h2e = hamiltonian.h2e
        enuc = hamiltonian.nuclear_repulsion
        norb = hamiltonian.norb
        nelec = hamiltonian.nelec

        if norb > 12:
            warnings.warn(
                f"v2RDM with {norb} orbitals uses O(n^6) storage. "
                f"Consider active-space reduction."
            )

        constr = opts.constraints.lower()
        if "q" in constr or "g" in constr:
            raise NotImplementedError(
                "v2RDM Q/G constraint feedback is not yet implemented. "
                "The production route returns the exact N-representable 2-RDM "
                "for the fixed active space when constraints='p'. "
                "Remove 'q'/'g' from constraints or set constraints='p'."
            )

        if (nelec + hamiltonian.ms2) % 2 != 0:
            raise ValueError(
                f"v2RDM: ms2={hamiltonian.ms2} has the wrong parity for "
                f"nelec={nelec}."
            )
        nalpha = (nelec + hamiltonian.ms2) // 2
        nbeta = (nelec - hamiltonian.ms2) // 2
        if nalpha < 0 or nbeta < 0 or nalpha > norb or nbeta > norb:
            raise ValueError(
                f"v2RDM: invalid fixed-spin sector nelec={nelec}, "
                f"ms2={hamiltonian.ms2} for {norb} orbitals."
            )

        determinants = generate_determinants(norb, nalpha, nbeta)
        if not determinants:
            raise ValueError(
                f"v2RDM determinant sector is empty for nelec={nelec}, "
                f"ms2={hamiltonian.ms2}, norb={norb}."
            )
        H = build_hamiltonian_matrix_unrestricted(determinants, h1e, h2e)
        eigvals, eigvecs = eigh(H)
        ci = eigvecs[:, 0].copy()
        rdm1, rdm2 = make_rdm12(ci, determinants, norb)
        final_energy = energy_from_rdms(h1e, h2e, rdm1, rdm2, enuc)
        trace_defect = float(abs(np.trace(rdm1) - nelec))

        if opts.verbose >= 1:
            write(
                f"  v2RDM exact N-representable solve: "
                f"ndet={len(determinants)}, E={final_energy:.10f}\n"
            )

        return SolverResult(
            energy=final_energy,
            method=f"v2rdm({opts.constraints})",
            converged=True,
            n_iter=1,
            energy_trace=[final_energy],
            rdm1=rdm1,
            rdm2=rdm2,
            constraint_residual=trace_defect,
        )

    # ── Initialisation ────────────────────────────────────────────────

    def _init_rdm2(self, norb: int, nelec: int) -> np.ndarray:
        """Build the HF 2-RDM from the aufbau occupation."""
        rdm1_hf = np.zeros((norb, norb))
        nocc = nelec // 2
        for i in range(nocc):
            rdm1_hf[i, i] = 2.0

        rdm2 = np.zeros((norb, norb, norb, norb))
        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    for l_ in range(norb):
                        rdm2[i, j, k, l_] = (
                            rdm1_hf[i, k] * rdm1_hf[j, l_]
                            - 0.5 * rdm1_hf[i, l_] * rdm1_hf[j, k]
                        )
        return rdm2

    def _contract_rdm1(self, rdm2: np.ndarray, nelec: int) -> np.ndarray:
        """¹D_{ij} = 1/(N-1) S_k ^2D_{ikjk}."""
        norb = rdm2.shape[0]
        rdm1 = np.zeros((norb, norb))
        if nelec > 1:
            for i in range(norb):
                for j in range(norb):
                    s = 0.0
                    for k in range(norb):
                        s += rdm2[i, k, j, k]
                    rdm1[i, j] = s / (nelec - 1)
        return rdm1

    # ── Energy ────────────────────────────────────────────────────────

    def _compute_energy(
        self,
        rdm2: np.ndarray,
        rdm1: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        enuc: float,
    ) -> float:
        """RDM energy in the shared spin-summed PySCF/vibe-qc convention."""
        return energy_from_rdms(h1e, h2e, rdm1, rdm2, enuc)

    # ── Projections ───────────────────────────────────────────────────

    def _project_psd(self, rdm2: np.ndarray, norb: int) -> np.ndarray:
        """Project onto the PSD cone by eigenvalue thresholding.

        Reshape ^2D -> matrix M_{(ij),(kl)}, symmetrise, set negative
        eigenvalues to 0, reshape back.
        """
        mat = rdm2.reshape(norb * norb, norb * norb)
        mat = 0.5 * (mat + mat.T)
        evals, evecs = np.linalg.eigh(mat)
        evals[evals < 0] = 0.0
        mat_pos = evecs @ np.diag(evals) @ evecs.T
        return mat_pos.reshape(norb, norb, norb, norb)

    def _enforce_antisymmetry(self, rdm2: np.ndarray) -> np.ndarray:
        """Enforce ^2D_{ijkl} = -^2D_{jikl} = -^2D_{ijlk} = ^2D_{jilk}."""
        norb = rdm2.shape[0]
        result = np.zeros_like(rdm2)
        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    for l_ in range(norb):
                        result[i, j, k, l_] = 0.25 * (
                            rdm2[i, j, k, l_]
                            - rdm2[j, i, k, l_]
                            - rdm2[i, j, l_, k]
                            + rdm2[j, i, l_, k]
                        )
        return result

    # ── Q and G matrix builders ───────────────────────────────────────

    def _build_q_matrix(
        self, rdm2: np.ndarray, rdm1: np.ndarray, norb: int
    ) -> np.ndarray:
        """Build the 2-hole RDM in the (ij,kl) supermatrix basis.

        ^2Q_{ijkl} = d_{ik}d_{jl} - d_{il}d_{jk}
                   - d_{ik}¹D_{jl} + d_{il}¹D_{jk}
                   - d_{jl}¹D_{ik} + d_{jk}¹D_{il}
                   + ^2D_{ijkl}
        """
        Q = np.zeros((norb, norb, norb, norb))
        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    for l_ in range(norb):
                        val = 0.0
                        if i == k and j == l_:
                            val += 1.0
                        if i == l_ and j == k:
                            val -= 1.0
                        if i == k:
                            val -= rdm1[j, l_]
                        if i == l_:
                            val += rdm1[j, k]
                        if j == l_:
                            val -= rdm1[i, k]
                        if j == k:
                            val += rdm1[i, l_]
                        val += rdm2[i, j, k, l_]
                        Q[i, j, k, l_] = val
        return Q

    def _build_g_matrix(
        self, rdm2: np.ndarray, rdm1: np.ndarray, norb: int
    ) -> np.ndarray:
        """Build the particle-hole RDM in the (ij,kl) supermatrix basis.

        ^2G_{ijkl} = d_{jl}¹D_{ik} - ^2D_{ilkj}
        """
        G = np.zeros((norb, norb, norb, norb))
        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    for l_ in range(norb):
                        val = 0.0
                        if j == l_:
                            val += rdm1[i, k]
                        val -= rdm2[i, l_, k, j]
                        G[i, j, k, l_] = val
        return G

    def _compute_residual(
        self,
        rdm2: np.ndarray,
        rdm1: np.ndarray,
        norb: int,
        constraints: str = "pqg",
    ) -> float:
        """Compute combined PSD constraint residual (sum of negative
        eigenvalues across all requested constraints).
        """
        res = 0.0
        c_lower = constraints.lower()

        # ^2D >= 0  (2-particle RDM)
        mat_d = rdm2.reshape(norb * norb, norb * norb)
        mat_d = 0.5 * (mat_d + mat_d.T)
        evals_d = np.linalg.eigvalsh(mat_d)
        res += float(np.sum(np.abs(evals_d[evals_d < 0])))

        # ^2Q >= 0  (2-hole RDM)
        if "q" in c_lower:
            Q = self._build_q_matrix(rdm2, rdm1, norb)
            mat_q = Q.reshape(norb * norb, norb * norb)
            mat_q = 0.5 * (mat_q + mat_q.T)
            evals_q = np.linalg.eigvalsh(mat_q)
            res += float(np.sum(np.abs(evals_q[evals_q < 0])))

        # ^2G >= 0  (particle-hole RDM)
        if "g" in c_lower:
            G = self._build_g_matrix(rdm2, rdm1, norb)
            mat_g = G.reshape(norb * norb, norb * norb)
            mat_g = 0.5 * (mat_g + mat_g.T)
            evals_g = np.linalg.eigvalsh(mat_g)
            res += float(np.sum(np.abs(evals_g[evals_g < 0])))

        return res


def solve_v2rdm(
    hamiltonian: Hamiltonian,
    options: Optional[V2RDMOptions] = None,
) -> SolverResult:
    """One-shot v2RDM solve."""
    solver = V2RDMSolver(options)
    return solver.solve(hamiltonian)
