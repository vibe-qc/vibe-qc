"""Dense full CCSDT by exact determinant-space similarity projection.

The solver evaluates the defining projected equations directly,

    <mu| exp(-T) H exp(T) |0> = 0,  mu in {S, D, T},

with ``T = T1 + T2 + T3``.  Cluster operators act on bitmask Slater
determinants, so fermionic phases and all disconnected products are generated
by second quantization instead of a hand-transcribed CCSDT diagram list.

Only excitation ranks through five are needed in ``exp(T)|0>``: a two-body
Hamiltonian can lower rank by at most two, and the projected residual stops at
triples.  This is still a dense benchmark-scale algorithm, but it is the full
iterative CCSDT model rather than a perturbative triples approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import sqrt
from typing import Optional

import numpy as np

from ._common import Hamiltonian, SolverResult
from ._mrpt import _ann, _cre, apply_1body, apply_2body


@dataclass(kw_only=True)
class CCSDTOptions:
    """Controls for the dense full-CCSDT amplitude iteration.

    ``n_frozen_core=None`` selects vibe-qc's chemical-core default in the
    high-level runner.  The standalone :func:`ccsdt` entry point treats it as
    zero because it has no molecular element list from which to infer cores.
    """

    max_iter: int = 80
    conv_tol_energy: float = 1.0e-10
    conv_tol_residual: float = 1.0e-8
    diis_subspace_size: int = 6
    n_frozen_core: Optional[int] = None


@dataclass(frozen=True)
class CCSDTIteration:
    """One full-CCSDT amplitude iteration."""

    iteration: int
    energy: float
    delta_energy: float
    residual_rms: float
    residual_max: float
    diis_subspace: int


@dataclass
class CCSDTResult(SolverResult):
    """Result of a full iterative CCSDT calculation."""

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
    ccsdt_trace: list[CCSDTIteration] = field(default_factory=list)

    @property
    def e_total(self) -> float:
        """Total CCSDT energy, including nuclear repulsion."""
        return self.energy

    @property
    def e_corr(self) -> float:
        """CCSDT correlation energy relative to the reference determinant."""
        return self.e_correlation


@dataclass(frozen=True)
class _Excitation:
    holes: tuple[int, ...]
    particles: tuple[int, ...]
    target: int
    phase: int
    rank: int
    denominator: float


def _apply_raw_excitation(
    mask: int,
    holes: tuple[int, ...],
    particles: tuple[int, ...],
) -> tuple[int, int]:
    """Apply the canonical excitation string and return ``(phase, mask)``."""
    phase = 1
    current = mask
    for hole in holes:
        sign, current = _ann(current, hole)
        if sign == 0:
            return 0, 0
        phase *= sign
    for particle in reversed(particles):
        sign, current = _cre(current, particle)
        if sign == 0:
            return 0, 0
        phase *= sign
    return phase, current


def _reference_mask(norb: int, nalpha: int, nbeta: int) -> int:
    mask = 0
    for orbital in range(nalpha):
        mask |= 1 << orbital
    for orbital in range(nbeta):
        mask |= 1 << (norb + orbital)
    return mask


def _excitation_rank(mask: int, reference: int) -> int:
    return (reference & ~mask).bit_count()


def _reference_fock_diagonal(
    h1e: np.ndarray,
    h2e: np.ndarray,
    nalpha: int,
    nbeta: int,
) -> np.ndarray:
    """Spin-dependent diagonal Fock elements for the Aufbau reference."""
    f_alpha = np.asarray(h1e, dtype=float).copy()
    f_beta = np.asarray(h1e, dtype=float).copy()
    for occupied in range(nalpha):
        coulomb = h2e[:, occupied, :, occupied]
        exchange = h2e[:, occupied, occupied, :]
        f_alpha += coulomb - exchange
        f_beta += coulomb
    for occupied in range(nbeta):
        coulomb = h2e[:, occupied, :, occupied]
        exchange = h2e[:, occupied, occupied, :]
        f_beta += coulomb - exchange
        f_alpha += coulomb
    return np.concatenate((np.diag(f_alpha), np.diag(f_beta)))


def _build_excitations(
    norb: int,
    nalpha: int,
    nbeta: int,
    reference: int,
    orbital_energies: np.ndarray,
) -> list[_Excitation]:
    occupied_alpha = tuple(range(nalpha))
    occupied_beta = tuple(norb + i for i in range(nbeta))
    virtual_alpha = tuple(range(nalpha, norb))
    virtual_beta = tuple(norb + i for i in range(nbeta, norb))
    excitations: list[_Excitation] = []

    for rank in (1, 2, 3):
        for n_alpha_excited in range(rank + 1):
            n_beta_excited = rank - n_alpha_excited
            if (
                n_alpha_excited > len(occupied_alpha)
                or n_alpha_excited > len(virtual_alpha)
                or n_beta_excited > len(occupied_beta)
                or n_beta_excited > len(virtual_beta)
            ):
                continue
            for holes_alpha in combinations(occupied_alpha, n_alpha_excited):
                for holes_beta in combinations(occupied_beta, n_beta_excited):
                    holes = tuple(sorted(holes_alpha + holes_beta))
                    for particles_alpha in combinations(
                        virtual_alpha, n_alpha_excited
                    ):
                        for particles_beta in combinations(
                            virtual_beta, n_beta_excited
                        ):
                            particles = tuple(
                                sorted(particles_alpha + particles_beta)
                            )
                            raw_phase, target = _apply_raw_excitation(
                                reference, holes, particles
                            )
                            if raw_phase == 0:
                                raise RuntimeError(
                                    "CCSDT excitation generator produced an "
                                    "operator that annihilates the reference"
                                )
                            denominator = float(
                                sum(orbital_energies[i] for i in holes)
                                - sum(orbital_energies[a] for a in particles)
                            )
                            if abs(denominator) < 1.0e-12:
                                raise ValueError(
                                    "CCSDT encountered a zero orbital-energy "
                                    "denominator; the reference is degenerate "
                                    "and cannot be Jacobi-preconditioned"
                                )
                            excitations.append(
                                _Excitation(
                                    holes=holes,
                                    particles=particles,
                                    target=target,
                                    phase=raw_phase,
                                    rank=rank,
                                    denominator=denominator,
                                )
                            )
    return excitations


def _accumulate(
    destination: dict[int, float],
    source: dict[int, float],
    *,
    scale: float = 1.0,
) -> None:
    for mask, coefficient in source.items():
        value = destination.get(mask, 0.0) + scale * coefficient
        if value == 0.0:
            destination.pop(mask, None)
        else:
            destination[mask] = value


def _cluster_action(
    state: dict[int, float],
    amplitudes: np.ndarray,
    excitations: list[_Excitation],
    reference: int,
    max_rank: int,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for index, excitation in enumerate(excitations):
        amplitude = float(amplitudes[index])
        if amplitude == 0.0:
            continue
        operator_phase = excitation.phase
        for mask, coefficient in state.items():
            raw_phase, target = _apply_raw_excitation(
                mask, excitation.holes, excitation.particles
            )
            if raw_phase == 0:
                continue
            if _excitation_rank(target, reference) > max_rank:
                continue
            out[target] = out.get(target, 0.0) + (
                amplitude * operator_phase * raw_phase * coefficient
            )
    return out


def _exp_t_reference(
    amplitudes: np.ndarray,
    excitations: list[_Excitation],
    reference: int,
    max_rank: int,
) -> dict[int, float]:
    total = {reference: 1.0}
    term = {reference: 1.0}
    for order in range(1, max_rank + 1):
        term = _cluster_action(
            term,
            amplitudes,
            excitations,
            reference,
            max_rank,
        )
        if not term:
            break
        inverse_order = 1.0 / order
        term = {mask: coefficient * inverse_order for mask, coefficient in term.items()}
        _accumulate(total, term)
    return total


def _hamiltonian_action(
    state: dict[int, float],
    h1e: np.ndarray,
    eri_chemist: np.ndarray,
    norb: int,
    reference: int,
    max_rank: int,
) -> dict[int, float]:
    one_body = apply_1body(state, h1e, norb)
    two_body = apply_2body(state, eri_chemist, norb)
    out: dict[int, float] = {}
    for contribution in (one_body, two_body):
        for mask, coefficient in contribution.items():
            if _excitation_rank(mask, reference) <= max_rank:
                out[mask] = out.get(mask, 0.0) + coefficient
    return out


def _project_similarity_transform(
    amplitudes: np.ndarray,
    excitations: list[_Excitation],
    h1e: np.ndarray,
    eri_chemist: np.ndarray,
    norb: int,
    reference: int,
    ket_rank: int,
) -> tuple[float, np.ndarray]:
    right = _exp_t_reference(
        amplitudes,
        excitations,
        reference,
        ket_rank,
    )
    h_right = _hamiltonian_action(
        right,
        h1e,
        eri_chemist,
        norb,
        reference,
        3,
    )

    transformed = dict(h_right)
    term = h_right
    for order in range(1, 4):
        term = _cluster_action(
            term,
            amplitudes,
            excitations,
            reference,
            3,
        )
        if not term:
            break
        factor = -1.0 / order
        term = {mask: factor * coefficient for mask, coefficient in term.items()}
        _accumulate(transformed, term)

    energy_electronic = float(transformed.get(reference, 0.0))
    residual = np.fromiter(
        (transformed.get(excitation.target, 0.0) for excitation in excitations),
        dtype=float,
        count=len(excitations),
    )
    return energy_electronic, residual


def _diis_extrapolate(
    amplitude_history: list[np.ndarray],
    residual_history: list[np.ndarray],
) -> np.ndarray:
    size = len(amplitude_history)
    if size < 2:
        return amplitude_history[-1]
    matrix = np.empty((size + 1, size + 1), dtype=float)
    matrix[-1, :] = -1.0
    matrix[:, -1] = -1.0
    matrix[-1, -1] = 0.0
    for row in range(size):
        for column in range(size):
            matrix[row, column] = float(
                np.dot(residual_history[row], residual_history[column])
            )
    rhs = np.zeros(size + 1, dtype=float)
    rhs[-1] = -1.0
    try:
        coefficients = np.linalg.solve(matrix, rhs)[:size]
    except np.linalg.LinAlgError:
        return amplitude_history[-1]
    return sum(
        coefficients[index] * amplitude_history[index]
        for index in range(size)
    )


def ccsdt(
    hamiltonian: Hamiltonian,
    options: Optional[CCSDTOptions] = None,
) -> CCSDTResult:
    """Solve the full ground-state CCSDT equations for one determinant.

    The Hamiltonian must be in an orthonormal spatial-orbital basis ordered
    occupied before virtual.  Closed-shell canonical RHF orbitals are the
    normal production path; a common-orbital open-shell reference is also
    supported through ``Hamiltonian.ms2``.
    """
    opts = options or CCSDTOptions()
    if opts.max_iter < 1:
        raise ValueError("CCSDTOptions.max_iter must be >= 1")
    if opts.conv_tol_energy <= 0.0 or opts.conv_tol_residual <= 0.0:
        raise ValueError("CCSDT convergence tolerances must be positive")
    if opts.diis_subspace_size < 1:
        raise ValueError("CCSDTOptions.diis_subspace_size must be >= 1")

    n_frozen = int(opts.n_frozen_core or 0)
    if n_frozen < 0 or 2 * n_frozen >= hamiltonian.nelec:
        if n_frozen != 0:
            raise ValueError(
                f"invalid CCSDT n_frozen_core={n_frozen} for "
                f"{hamiltonian.nelec} electrons"
            )
    if n_frozen:
        active = hamiltonian.active_space(
            hamiltonian.norb - n_frozen,
            hamiltonian.nelec - 2 * n_frozen,
        )
    else:
        active = hamiltonian

    nelec = int(active.nelec)
    ms2 = int(active.ms2)
    if abs(ms2) > nelec or (nelec + ms2) % 2:
        raise ValueError(f"CCSDT electron count {nelec} is incompatible with ms2={ms2}")
    nalpha = (nelec + ms2) // 2
    nbeta = nelec - nalpha
    norb = int(active.norb)
    if nalpha > norb or nbeta > norb:
        raise ValueError(
            f"CCSDT cannot place nalpha={nalpha}, nbeta={nbeta} in {norb} orbitals"
        )
    if nalpha == norb and nbeta == norb:
        raise ValueError("CCSDT requires at least one virtual spin orbital")

    h1e = np.ascontiguousarray(active.h1e, dtype=float)
    h2e = np.ascontiguousarray(active.h2e, dtype=float)
    eri_chemist = np.ascontiguousarray(h2e.transpose(0, 2, 1, 3))
    reference = _reference_mask(norb, nalpha, nbeta)
    orbital_energies = _reference_fock_diagonal(h1e, h2e, nalpha, nbeta)
    excitations = _build_excitations(
        norb, nalpha, nbeta, reference, orbital_energies
    )
    if not excitations:
        raise ValueError("CCSDT excitation space is empty")

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
    reference_action = _hamiltonian_action(
        {reference: 1.0}, h1e, eri_chemist, norb, reference, 3
    )
    e_reference = float(reference_action.get(reference, 0.0)) + float(
        active.nuclear_repulsion
    )
    amplitudes = np.fromiter(
        (
            reference_action.get(excitation.target, 0.0)
            / excitation.denominator
            for excitation in excitations
        ),
        dtype=float,
        count=len(excitations),
    )

    max_possible_rank = min(nelec, 2 * norb - nelec)
    ket_rank = min(5, max_possible_rank)
    amplitude_history: list[np.ndarray] = []
    residual_history: list[np.ndarray] = []
    trace: list[CCSDTIteration] = []
    energy_trace: list[float] = []
    previous_energy = e_reference
    residual = np.zeros_like(amplitudes)
    converged = False

    for iteration in range(1, opts.max_iter + 1):
        energy_electronic, residual = _project_similarity_transform(
            amplitudes,
            excitations,
            h1e,
            eri_chemist,
            norb,
            reference,
            ket_rank,
        )
        energy = energy_electronic + float(active.nuclear_repulsion)
        delta_energy = energy - previous_energy
        residual_rms = float(np.linalg.norm(residual) / sqrt(residual.size))
        residual_max = float(np.max(np.abs(residual)))
        trace.append(
            CCSDTIteration(
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

        candidate = amplitudes + residual / denominators
        amplitude_history.append(candidate.copy())
        residual_history.append(residual.copy())
        if len(amplitude_history) > opts.diis_subspace_size:
            amplitude_history.pop(0)
            residual_history.pop(0)
        amplitudes = _diis_extrapolate(amplitude_history, residual_history)
        previous_energy = energy

    final_energy = energy_trace[-1]
    rank_norm = {
        rank: float(np.linalg.norm(amplitudes[ranks == rank]))
        for rank in (1, 2, 3)
    }
    rank_count = {rank: int(np.count_nonzero(ranks == rank)) for rank in (1, 2, 3)}
    final_residual_rms = float(np.linalg.norm(residual) / sqrt(residual.size))
    final_residual_max = float(np.max(np.abs(residual)))
    return CCSDTResult(
        energy=final_energy,
        method="ccsdt",
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
        ccsdt_trace=trace,
    )


__all__ = [
    "CCSDTIteration",
    "CCSDTOptions",
    "CCSDTResult",
    "ccsdt",
]
