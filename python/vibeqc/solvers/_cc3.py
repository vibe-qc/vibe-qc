"""Dense ground-state CC3 from the defining commutator equations.

The implementation follows Koch et al., J. Chem. Phys. 106, 1808 (1997):
T1 and T2 are iterated, while T3 is regenerated at every iteration from

    [F, T3] + [exp(-T1) U exp(T1), T2] = 0.

The triples feedback added to the CCSD singles and doubles residuals is
``[H, T3]`` and ``[exp(-T1) H exp(T1), T3]``, respectively.  Sparse
determinant states generate the commutators exactly, avoiding a separate
hand-transcribed diagram list.  This remains a benchmark-scale route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Callable, Optional

import numpy as np

from ._ccsdt import (
    _Excitation,
    _accumulate,
    _build_excitations,
    _cluster_action,
    _diis_extrapolate,
    _excitation_rank,
    _hamiltonian_action,
    _project_similarity_transform,
    _reference_fock_diagonal,
    _reference_mask,
)
from ._common import Hamiltonian, SolverResult
from ._mrpt import apply_1body


@dataclass(kw_only=True)
class CC3Options:
    """Controls for the dense ground-state CC3 iteration.

    ``n_frozen_core=None`` selects vibe-qc's chemical-core default in the
    high-level runner.  The standalone :func:`cc3` entry point treats it as
    zero because it has no molecular element list from which to infer cores.
    """

    max_iter: int = 80
    conv_tol_energy: float = 1.0e-10
    conv_tol_residual: float = 1.0e-8
    diis_subspace_size: int = 6
    n_frozen_core: Optional[int] = None


@dataclass(frozen=True)
class CC3Iteration:
    """One CC3 T1/T2 iteration."""

    iteration: int
    energy: float
    delta_energy: float
    residual_rms: float
    residual_max: float
    diis_subspace: int


@dataclass
class CC3Result(SolverResult):
    """Result of a dense ground-state CC3 calculation."""

    e_reference: float = 0.0
    e_correlation: float = 0.0
    residual_rms: float = 0.0
    residual_max: float = 0.0
    t1_norm: float = 0.0
    t2_norm: float = 0.0
    t3_norm: float = 0.0
    n_singles: int = 0
    n_doubles: int = 0
    n_triples: int = 0
    n_frozen_core: int = 0
    scf_trace: list[object] = field(default_factory=list)
    amplitudes: np.ndarray = field(default_factory=lambda: np.empty(0))
    excitation_ranks: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int8)
    )
    cc3_trace: list[CC3Iteration] = field(default_factory=list)

    @property
    def e_total(self) -> float:
        return self.energy

    @property
    def e_corr(self) -> float:
        return self.e_correlation


_State = dict[int, float]
_StateAction = Callable[[_State], _State]


def _exp_cluster_state(
    state: _State,
    amplitudes: np.ndarray,
    excitations: list[_Excitation],
    reference: int,
    max_rank: int,
) -> _State:
    """Apply ``exp(T)`` to an arbitrary sparse determinant state."""
    total = dict(state)
    term = dict(state)
    for order in range(1, max_rank + 1):
        term = _cluster_action(
            term, amplitudes, excitations, reference, max_rank
        )
        if not term:
            break
        inverse_order = 1.0 / order
        term = {
            mask: coefficient * inverse_order
            for mask, coefficient in term.items()
        }
        _accumulate(total, term)
    return total


def _similarity_action(
    state: _State,
    t1_amplitudes: np.ndarray,
    excitations: list[_Excitation],
    reference: int,
    max_rank: int,
    operator_action: _StateAction,
) -> _State:
    right = _exp_cluster_state(
        state, t1_amplitudes, excitations, reference, max_rank
    )
    operated = operator_action(right)
    return _exp_cluster_state(
        operated, -t1_amplitudes, excitations, reference, max_rank
    )


def _commutator_on_reference(
    operator_action: _StateAction,
    cluster_amplitudes: np.ndarray,
    excitations: list[_Excitation],
    reference: int,
    max_rank: int,
) -> _State:
    cluster_reference = _cluster_action(
        {reference: 1.0},
        cluster_amplitudes,
        excitations,
        reference,
        max_rank,
    )
    left = operator_action(cluster_reference)
    right = _cluster_action(
        operator_action({reference: 1.0}),
        cluster_amplitudes,
        excitations,
        reference,
        max_rank,
    )
    _accumulate(left, right, scale=-1.0)
    return left


def _project_state(
    state: _State,
    excitations: list[_Excitation],
) -> np.ndarray:
    return np.fromiter(
        (state.get(excitation.target, 0.0) for excitation in excitations),
        dtype=float,
        count=len(excitations),
    )


def cc3(
    hamiltonian: Hamiltonian,
    options: Optional[CC3Options] = None,
) -> CC3Result:
    """Solve ground-state CC3 for a canonical closed-shell reference."""
    opts = options or CC3Options()
    if opts.max_iter < 1:
        raise ValueError("CC3Options.max_iter must be >= 1")
    if opts.conv_tol_energy <= 0.0 or opts.conv_tol_residual <= 0.0:
        raise ValueError("CC3 convergence tolerances must be positive")
    if opts.diis_subspace_size < 1:
        raise ValueError("CC3Options.diis_subspace_size must be >= 1")
    if hamiltonian.ms2 != 0 or hamiltonian.nelec % 2:
        raise ValueError("CC3 currently requires a closed-shell reference")

    n_frozen = int(opts.n_frozen_core or 0)
    if n_frozen < 0 or 2 * n_frozen >= hamiltonian.nelec:
        if n_frozen != 0:
            raise ValueError(
                f"invalid CC3 n_frozen_core={n_frozen} for "
                f"{hamiltonian.nelec} electrons"
            )
    active = (
        hamiltonian.active_space(
            hamiltonian.norb - n_frozen,
            hamiltonian.nelec - 2 * n_frozen,
        )
        if n_frozen
        else hamiltonian
    )

    nelec = int(active.nelec)
    norb = int(active.norb)
    nalpha = nbeta = nelec // 2
    if nalpha >= norb:
        raise ValueError("CC3 requires at least one virtual spatial orbital")

    h1e = np.ascontiguousarray(active.h1e, dtype=float)
    h2e = np.ascontiguousarray(active.h2e, dtype=float)
    eri_chemist = np.ascontiguousarray(h2e.transpose(0, 2, 1, 3))
    reference = _reference_mask(norb, nalpha, nbeta)
    orbital_energies = _reference_fock_diagonal(
        h1e, h2e, nalpha, nbeta
    )
    excitations = _build_excitations(
        norb, nalpha, nbeta, reference, orbital_energies
    )
    if not excitations:
        raise ValueError("CC3 excitation space is empty")

    ranks = np.fromiter(
        (excitation.rank for excitation in excitations),
        dtype=np.int8,
        count=len(excitations),
    )
    denominators = np.fromiter(
        (excitation.denominator for excitation in excitations),
        dtype=float,
        count=len(excitations),
    )
    singles = ranks == 1
    doubles = ranks == 2
    triples = ranks == 3
    sd = ranks <= 2

    max_possible_rank = min(nelec, 2 * norb - nelec)
    # A two-body Hamiltonian can lower excitation rank by at most two.
    # Ranks through five therefore contain every contribution that can
    # reach the projected CC3 singles, doubles, or triples equations.
    max_rank = min(5, max_possible_rank)
    reference_action = _hamiltonian_action(
        {reference: 1.0}, h1e, eri_chemist, norb, reference, max_rank
    )
    e_reference = float(reference_action.get(reference, 0.0)) + float(
        active.nuclear_repulsion
    )
    amplitudes = np.zeros(len(excitations), dtype=float)
    reference_projection = _project_state(reference_action, excitations)
    amplitudes[sd] = reference_projection[sd] / denominators[sd]

    fock = np.diag(orbital_energies[:norb])

    def h_action(state: _State) -> _State:
        return _hamiltonian_action(
            state, h1e, eri_chemist, norb, reference, max_rank
        )

    def fluctuation_action(state: _State) -> _State:
        out = h_action(state)
        f_state = apply_1body(state, fock, norb)
        f_state = {
            mask: coefficient
            for mask, coefficient in f_state.items()
            if _excitation_rank(mask, reference) <= max_rank
        }
        _accumulate(out, f_state, scale=-1.0)
        return out

    def cc3_residual(
        current: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        t1 = np.zeros_like(current)
        t1[singles] = current[singles]
        t2 = np.zeros_like(current)
        t2[doubles] = current[doubles]
        ccsd_amplitudes = t1 + t2

        def h_hat(state: _State) -> _State:
            return _similarity_action(
                state,
                t1,
                excitations,
                reference,
                max_rank,
                h_action,
            )

        def u_hat(state: _State) -> _State:
            return _similarity_action(
                state,
                t1,
                excitations,
                reference,
                max_rank,
                fluctuation_action,
            )

        triples_generator = _project_state(
            _commutator_on_reference(
                u_hat, t2, excitations, reference, max_rank
            ),
            excitations,
        )
        t3 = np.zeros_like(current)
        t3[triples] = (
            triples_generator[triples] / denominators[triples]
        )

        energy_electronic, ccsd_residual = _project_similarity_transform(
            ccsd_amplitudes,
            excitations,
            h1e,
            eri_chemist,
            norb,
            reference,
            min(4, max_rank),
        )
        singles_feedback = _project_state(
            _commutator_on_reference(
                h_action, t3, excitations, reference, max_rank
            ),
            excitations,
        )
        doubles_feedback = _project_state(
            _commutator_on_reference(
                h_hat, t3, excitations, reference, max_rank
            ),
            excitations,
        )
        residual = np.zeros_like(current)
        residual[singles] = (
            ccsd_residual[singles] + singles_feedback[singles]
        )
        residual[doubles] = (
            ccsd_residual[doubles] + doubles_feedback[doubles]
        )
        energy = energy_electronic + float(active.nuclear_repulsion)
        return energy, residual, t3

    amplitude_history: list[np.ndarray] = []
    residual_history: list[np.ndarray] = []
    trace: list[CC3Iteration] = []
    energy_trace: list[float] = []
    previous_energy = e_reference
    residual = np.zeros_like(amplitudes)
    t3 = np.zeros_like(amplitudes)
    converged = False

    for iteration in range(1, opts.max_iter + 1):
        energy, residual, t3 = cc3_residual(amplitudes)
        delta_energy = energy - previous_energy
        sd_residual = residual[sd]
        residual_rms = float(
            np.linalg.norm(sd_residual) / sqrt(sd_residual.size)
        )
        residual_max = float(np.max(np.abs(sd_residual)))
        trace.append(
            CC3Iteration(
                iteration=iteration,
                energy=energy,
                delta_energy=delta_energy,
                residual_rms=residual_rms,
                residual_max=residual_max,
                diis_subspace=len(amplitude_history),
            )
        )
        energy_trace.append(energy)
        if (
            abs(delta_energy) < opts.conv_tol_energy
            and residual_rms < opts.conv_tol_residual
        ):
            converged = True
            break
        if iteration == opts.max_iter:
            break

        candidate = amplitudes[sd] + sd_residual / denominators[sd]
        amplitude_history.append(candidate.copy())
        residual_history.append(sd_residual.copy())
        if len(amplitude_history) > opts.diis_subspace_size:
            amplitude_history.pop(0)
            residual_history.pop(0)
        amplitudes[sd] = _diis_extrapolate(
            amplitude_history, residual_history
        )
        previous_energy = energy

    amplitudes[triples] = t3[triples]
    rank_norm = {
        rank: float(np.linalg.norm(amplitudes[ranks == rank]))
        for rank in (1, 2, 3)
    }
    rank_count = {
        rank: int(np.count_nonzero(ranks == rank)) for rank in (1, 2, 3)
    }
    final_sd_residual = residual[sd]
    final_residual_rms = float(
        np.linalg.norm(final_sd_residual) / sqrt(final_sd_residual.size)
    )
    final_residual_max = float(np.max(np.abs(final_sd_residual)))
    final_energy = energy_trace[-1]
    return CC3Result(
        energy=final_energy,
        method="cc3",
        converged=converged,
        n_iter=len(trace),
        energy_trace=energy_trace,
        e_reference=e_reference,
        e_correlation=final_energy - e_reference,
        residual_rms=final_residual_rms,
        residual_max=final_residual_max,
        t1_norm=rank_norm[1],
        t2_norm=rank_norm[2],
        t3_norm=rank_norm[3],
        n_singles=rank_count[1],
        n_doubles=rank_count[2],
        n_triples=rank_count[3],
        n_frozen_core=n_frozen,
        amplitudes=amplitudes,
        excitation_ranks=ranks,
        cc3_trace=trace,
    )


__all__ = ["CC3Iteration", "CC3Options", "CC3Result", "cc3"]
