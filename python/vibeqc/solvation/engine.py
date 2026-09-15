"""The reaction-field step -- one implementation, every method.

Everything in an apparent-surface-charge solvation model is
reference-independent except the two arrows in
:mod:`vibeqc.solvation.provider`. This module is the step those arrows sit
in: given a solute density, produce the screened surface charges, the
operator they add to the Hamiltonian, and the energy decomposition.

    V_elec = provider.esp_at_cavity(density)      # solute -> surface
    V_tot  = V_elec + V_core
    q      = -f A^-1 V_tot                        # screened ASC solve
    fock   = provider.fock_contribution(q)        # surface -> Hamiltonian
    e_pol  = (1/2) q . V_tot

Before this module those five lines existed twice: once inside
``driver.run_cpcm_scf``'s macro-iteration for Gaussian HF/DFT, and once in
``msindo_cosmo._cosmo_reaction_field`` for MSINDO, against the same provider
protocol. Two copies of one equation is the same failure shape as two copies
of one constant (#546, #548), one level up: the ``E_pol = 1/2 q^T V``
convention and the core/electronic split were each written twice, and nothing
detected a change to only one (#554).

What is deliberately *not* unified
----------------------------------
The outer iteration strategy. Gaussian macro-iterates an outer ``q`` loop
around a full inner SCF, with q-DIIS and a shifted-DIIS warm start; MSINDO
folds the field into a single SCF through its ``fock_extra`` hook, updating
``q`` at every Fock build. Those are different, legitimate convergence
strategies over the same step, and collapsing them would change converged
iterate paths for no physical gain. Only the per-density step is shared.

The ``A`` solve is injected rather than performed here, so a caller keeps its
own factorization: Gaussian pre-factors ``A`` once per geometry and reuses the
LU across macro-iterations, MSINDO uses a dense solve or its fused C++ kernel.
The step does not need to know which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .screening import ScreeningModel


class CavitySolve(Protocol):
    """Apply ``A^-1`` to a right-hand side on the cavity.

    Injected so the caller owns the factorization. Implementations must be
    exact solves of the *same* ``A`` the energy uses -- an approximate or
    differently-capped ``A`` here would make the energy and its derivative
    disagree, which is the class of defect this package keeps re-learning.
    """

    def __call__(self, rhs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ReactionField:
    """One converged-density reaction-field evaluation.

    Attributes
    ----------
    q : ndarray, shape (n_seg,)
        Screened apparent surface charges in the scaled convention
        ``q = f q0``, ``q0 = -A^-1 V_total``. This is what the rest of
        vibe-qc stores and what the gradient convention assumes.
    V_elec, V_core, V_total : ndarray, shape (n_seg,)
        The surface potential, kept decomposed. The handover's result
        contract requires the split, and Direct COSMO-RS consumes it: its
        feedback acts on the electronic part alone.
    fock : ndarray
        Operator to **add** to the core Hamiltonian for these charges.
    e_pol : float
        Total polarisation energy ``(1/2) q . V_total``.
    e_core_share : float
        ``(1/2) q . V_core`` -- the core's share of the conductor energy.
        MSINDO's single-SCF bookkeeping adds exactly this to the electronic
        energy, because the ``(1/2) q . V_elec`` half is already counted
        through the density's own trace against the modified Hamiltonian.
    screening : ScreeningModel
        The model that produced ``q``; carried so no consumer re-derives it.
    """

    q: np.ndarray
    V_elec: np.ndarray
    V_core: np.ndarray
    V_total: np.ndarray
    fock: np.ndarray
    e_pol: float
    e_core_share: float
    screening: ScreeningModel

    @property
    def e_elec_share(self) -> float:
        """``(1/2) q . V_elec`` -- the electronic share of ``e_pol``."""
        return self.e_pol - self.e_core_share

    @property
    def total_charge(self) -> float:
        return float(np.sum(self.q))

    def with_charges(self, q: np.ndarray, provider) -> "ReactionField":
        """The same field re-expressed for a different charge vector.

        Reuses the surface potentials, which are the expensive part (the ESP
        is one provider pass over every segment), and rebuilds only the Fock
        contribution and the energy decomposition. This is how an outer
        accelerator feeds an extrapolated ``q`` -- the Gaussian q-DIIS -- back
        through one code path instead of re-deriving the downstream
        quantities itself, which is where a convention drifts.
        """
        q_new = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_new.shape != self.V_total.shape:
            raise ValueError(
                f"with_charges: got {q_new.shape[0]} charges for a cavity of "
                f"{self.V_total.shape[0]} segments."
            )
        return ReactionField(
            q=q_new,
            V_elec=self.V_elec,
            V_core=self.V_core,
            V_total=self.V_total,
            fock=provider.fock_contribution(q_new),
            e_pol=0.5 * float(np.dot(q_new, self.V_total)),
            e_core_share=0.5 * float(np.dot(q_new, self.V_core)),
            screening=self.screening,
        )


def solve_screened_charges(
    solve: CavitySolve,
    V_total: np.ndarray,
    screening: ScreeningModel,
) -> np.ndarray:
    """``q = -f A^-1 V_total`` in the scaled convention.

    The screening factor comes from the model that governs the run; see
    :mod:`vibeqc.solvation.screening` for why it is carried rather than
    re-derived here.
    """
    V = np.asarray(V_total, dtype=np.float64).reshape(-1)
    return np.asarray(solve(-screening.f * V), dtype=np.float64)


def reaction_field_step(
    provider,
    solve: CavitySolve,
    V_core: np.ndarray,
    density: np.ndarray,
    screening: ScreeningModel,
) -> ReactionField:
    """Evaluate the reaction field for one solute density.

    Parameters
    ----------
    provider
        A :class:`~vibeqc.solvation.provider.SolutePotentialProvider`: the
        Gaussian ESP-on-grid coupling, the MSINDO multipole coupling, or a
        later DFTB/xTB adapter. The step is method-independent through it.
    solve
        Applies ``A^-1``; see :class:`CavitySolve`.
    V_core
        Core/effective-nuclear potential at the cavity segments.
    density
        Total solute density in the provider's own convention.
    screening
        The run's :class:`~vibeqc.solvation.screening.ScreeningModel`.

    To re-express the result for accelerated charges without paying for the
    ESP again, use :meth:`ReactionField.with_charges`.
    """
    V_elec = np.asarray(provider.esp_at_cavity(density), dtype=np.float64).reshape(-1)
    V_c = np.asarray(V_core, dtype=np.float64).reshape(-1)
    if V_c.shape != V_elec.shape:
        raise ValueError(
            f"reaction_field_step: core potential has {V_c.shape[0]} segments "
            f"but the provider returned {V_elec.shape[0]}."
        )
    V_total = V_elec + V_c

    q = solve_screened_charges(solve, V_total, screening)
    if not np.all(np.isfinite(q)):
        raise RuntimeError(
            "reaction_field_step: the apparent-surface-charge solve produced "
            "a non-finite result; the cavity A-matrix is singular or the "
            "solute potential diverged."
        )

    fock = provider.fock_contribution(q)
    return ReactionField(
        q=q,
        V_elec=V_elec,
        V_core=V_c,
        V_total=V_total,
        fock=fock,
        e_pol=0.5 * float(np.dot(q, V_total)),
        e_core_share=0.5 * float(np.dot(q, V_c)),
        screening=screening,
    )


def lu_cavity_solve(A: np.ndarray) -> CavitySolve:
    """A :class:`CavitySolve` over a pre-factored ``A``.

    ``A`` is fixed for a geometry, so factor once and reuse across every
    macro-iteration rather than repeating an ``O(N^3)`` decomposition per
    cycle. Falls back to a plain solve when SciPy is unavailable.
    """
    A_arr = np.asarray(A, dtype=np.float64, order="C")
    try:
        from scipy.linalg import lu_factor, lu_solve
    except Exception:
        def _solve(rhs: np.ndarray) -> np.ndarray:
            return np.linalg.solve(A_arr, rhs)
        return _solve

    lu, piv = lu_factor(A_arr)

    def _solve_lu(rhs: np.ndarray) -> np.ndarray:
        return np.asarray(lu_solve((lu, piv), rhs), dtype=np.float64)

    return _solve_lu


__all__ = [
    "CavitySolve",
    "ReactionField",
    "lu_cavity_solve",
    "reaction_field_step",
    "solve_screened_charges",
]
