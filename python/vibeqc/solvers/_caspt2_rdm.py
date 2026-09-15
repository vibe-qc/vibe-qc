from __future__ import annotations

"""Private helpers for the RDM-contracted internally-contracted CASPT2 path.

The public, production CASPT2 implementation remains the explicit contracted
determinant engine in :mod:`vibeqc.solvers._mrpt`.  This module contains the
validated phase-exact active/external factorization kernel used to build the
large-CAS route incrementally.  It is intentionally not exported or dispatched
until the complete RDM-contracted solver is validated against the explicit
engine and OpenMolcas.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from scipy.linalg import eigh

FactoredState: TypeAlias = dict[tuple[int, int], float]
ExcitationLabel: TypeAlias = tuple[tuple[int, int], ...]


def _multiset_l1(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Return the L1 distance between two small orbital-index multisets."""
    counts = Counter(left)
    counts.subtract(Counter(right))
    return sum(abs(value) for value in counts.values())


def _external_signatures_one_body_reachable(
    left: tuple[tuple[int, ...], tuple[int, ...]],
    right: tuple[tuple[int, ...], tuple[int, ...]],
) -> bool:
    """Whether one spin-free one-body move can connect two signatures."""
    hole_delta = _multiset_l1(left[0], right[0])
    particle_delta = _multiset_l1(left[1], right[1])
    return hole_delta + particle_delta <= 2


@dataclass(frozen=True)
class ContractedCandidate:
    """Metadata for one raw IC-CASPT2 contracted excitation."""

    label: ExcitationLabel
    order: int
    external_class: tuple[int, int]
    openmolcas_cases: tuple[str, ...]
    external_signature: tuple[tuple[int, ...], tuple[int, ...]]
    inactive_holes: tuple[int, ...]
    secondary_particles: tuple[int, ...]
    active_creations: tuple[int, ...]
    active_annihilations: tuple[int, ...]


@dataclass(frozen=True)
class FactoredICSystem:
    """Dense IC matrices plus candidate metadata for validation/scaffolding."""

    candidates: tuple[ContractedCandidate, ...]
    overlap: np.ndarray
    h0: np.ndarray
    coupling: np.ndarray
    e0: float


@dataclass(frozen=True)
class OrthonormalICSystem:
    """Metric-orthonormal first-order system with block provenance."""

    block_keys: tuple[object, ...]
    block_ranges: tuple[tuple[object, int, int], ...]
    shifted_h0: np.ndarray
    coupling: np.ndarray


@dataclass(frozen=True)
class IterativeICResult:
    """Result from a block-preconditioned orthonormal IC linear solve."""

    energy: float
    n_basis: int
    n_iter: int
    residual_norm: float
    converged: bool


@dataclass(frozen=True)
class SignatureBlockCoupling:
    """Nonzero shifted-H0 coupling between two signature-orthonormal blocks."""

    left_block: int
    right_block: int
    frobenius_norm: float


@dataclass(frozen=True)
class SignatureBlockMatrixCoupling:
    """Matrix-valued shifted-H0 coupling between two signature blocks."""

    left_block: int
    right_block: int
    matrix: np.ndarray


@dataclass(frozen=True)
class SignatureBlockOperator:
    """Block representation of a signature-orthonormal shifted-H0 matrix."""

    block_ranges: tuple[tuple[object, int, int], ...]
    diagonal_blocks: tuple[np.ndarray, ...]
    couplings: tuple[SignatureBlockMatrixCoupling, ...]
    shape: tuple[int, int]


@dataclass(frozen=True)
class SignatureBlockLinearSystem:
    """Block operator plus RHS for a signature-orthonormal IC solve."""

    block_keys: tuple[object, ...]
    operator: SignatureBlockOperator
    coupling: np.ndarray


@dataclass(frozen=True)
class SignatureRawBlock:
    """Raw S/H0/V arrays for one external signature block."""

    key: object
    candidate_indices: np.ndarray
    overlap: np.ndarray
    h0: np.ndarray
    coupling: np.ndarray


@dataclass(frozen=True)
class SignatureRawBlockCoupling:
    """Raw H0 coupling between two external signature blocks."""

    left_block: int
    right_block: int
    h0: np.ndarray


@dataclass(frozen=True)
class SignatureRawBlockSystem:
    """Raw signature-block IC matrices before metric orthonormalization."""

    blocks: tuple[SignatureRawBlock, ...]
    couplings: tuple[SignatureRawBlockCoupling, ...]
    e0: float


@dataclass(frozen=True)
class _MetricBlock:
    """Local metric-orthonormalization data for one raw signature block."""

    key: object
    candidate_indices: np.ndarray
    transform: np.ndarray


_OPENMOLCAS_CASES_BY_EXTERNAL_CLASS = {
    (0, 0): ("CAS",),
    (1, 0): ("A",),
    (2, 0): ("B+", "B-"),
    (0, 1): ("C",),
    (1, 1): ("D",),
    (2, 1): ("E+", "E-"),
    (0, 2): ("F+", "F-"),
    (1, 2): ("G+", "G-"),
    (2, 2): ("H+", "H-"),
}


def openmolcas_case_labels(external_class: tuple[int, int]) -> tuple[str, ...]:
    """Return the OpenMolcas/Celani-Werner case labels for an external class.

    The private RDM path currently uses a raw spin-summed excitation basis, so
    spin-adapted ``+/-`` OpenMolcas cases are represented as a case family here.
    The actual matrix elements remain validated against vibe-qc's explicit IC
    oracle; this metadata only gives the block solver stable CASPT2 names.
    """
    try:
        return _OPENMOLCAS_CASES_BY_EXTERNAL_CLASS[external_class]
    except KeyError as exc:
        message = f"unsupported CASPT2 external class {external_class}"
        raise ValueError(message) from exc


class ActiveExternalFactorization:
    """Tensor-product determinant algebra for active-CI times external masks.

    A state is stored as ``{(external_mask, active_mask): coeff}``.  The
    external mask uses the full global spin-orbital bit positions for inactive
    and virtual orbitals; the active mask uses local active spin-orbitals.  Each
    spin-orbital operation computes its fermionic phase from the full global
    occupation ordering, so mixed active/external operator matrix elements carry
    the same signs as the explicit determinant oracle without materializing the
    full active-by-external determinant space.
    """

    def __init__(self, ci, determinants, n_core: int, n_active: int, n_orb: int):
        self.n_core = n_core
        self.n_active = n_active
        self.n_orb = n_orb
        self.active_start = n_core
        self.active_stop = n_core + n_active
        self.ref = self._build_reference(ci, determinants)

    @classmethod
    def from_full_reference(
        cls, reference: dict[int, float], n_core: int, n_active: int, n_orb: int
    ) -> "ActiveExternalFactorization":
        """Build the factorization by splitting a full determinant reference."""
        obj = cls.__new__(cls)
        obj.n_core = n_core
        obj.n_active = n_active
        obj.n_orb = n_orb
        obj.active_start = n_core
        obj.active_stop = n_core + n_active
        obj.ref = obj._split_full_state(reference)
        return obj

    def _so(self, orbital: int, spin: int) -> int:
        return orbital + spin * self.n_orb

    def _is_active(self, orbital: int) -> bool:
        return self.active_start <= orbital < self.active_stop

    def _active_local(self, orbital: int, spin: int) -> int:
        return (orbital - self.active_start) + spin * self.n_active

    def _active_global(self, local: int) -> int:
        if local < self.n_active:
            return self.active_start + local
        return self.active_start + (local - self.n_active) + self.n_orb

    def _split_full_state(self, state: dict[int, float]) -> FactoredState:
        out: FactoredState = {}
        for full_mask, coeff in state.items():
            external_mask = full_mask
            active_mask = 0
            for orbital in range(self.active_start, self.active_stop):
                for spin in (0, 1):
                    global_spinorb = self._so(orbital, spin)
                    if (full_mask >> global_spinorb) & 1:
                        local = self._active_local(orbital, spin)
                        active_mask |= 1 << local
                        external_mask &= ~(1 << global_spinorb)
            self._add_to(out, (external_mask, active_mask), coeff)
        return out

    def _build_reference(self, ci, determinants) -> FactoredState:
        external_ref = 0
        for i in range(self.n_core):
            external_ref |= (1 << self._so(i, 0)) | (1 << self._so(i, 1))

        ref: FactoredState = {}
        for idx, (alpha_occ, beta_occ) in enumerate(determinants):
            coeff = float(ci[idx])
            if abs(coeff) < 1e-14:
                continue
            active_mask = 0
            for orb in alpha_occ:
                active_mask |= 1 << orb
            for orb in beta_occ:
                active_mask |= 1 << (orb + self.n_active)
            self._add_to(ref, (external_ref, active_mask), coeff)
        return ref

    def _active_mask_to_global(self, active_mask: int) -> int:
        out = 0
        mask = active_mask
        while mask:
            bit = mask & -mask
            local = bit.bit_length() - 1
            out |= 1 << self._active_global(local)
            mask ^= bit
        return out

    def _active_count_before(self, active_mask: int, global_spinorb: int) -> int:
        count = 0
        mask = active_mask
        while mask:
            bit = mask & -mask
            local = bit.bit_length() - 1
            if self._active_global(local) < global_spinorb:
                count += 1
            mask ^= bit
        return count

    def _count_before(
        self, external_mask: int, active_mask: int, global_spinorb: int
    ) -> int:
        return (
            (external_mask & ((1 << global_spinorb) - 1)).bit_count()
            + self._active_count_before(active_mask, global_spinorb)
        )

    @staticmethod
    def _add_to(state: FactoredState, key: tuple[int, int], value: float) -> None:
        if abs(value) > 1e-15:
            state[key] = state.get(key, 0.0) + value

    @classmethod
    def add(cls, left: FactoredState, right: FactoredState) -> FactoredState:
        out = dict(left)
        for key, value in right.items():
            cls._add_to(out, key, value)
        return out

    def _ann_one(
        self, external_mask: int, active_mask: int, orbital: int, spin: int
    ) -> tuple[int, int, int]:
        global_spinorb = self._so(orbital, spin)
        sign = (
            -1
            if self._count_before(external_mask, active_mask, global_spinorb) & 1
            else 1
        )
        if self._is_active(orbital):
            local = self._active_local(orbital, spin)
            if not ((active_mask >> local) & 1):
                return 0, external_mask, active_mask
            return sign, external_mask, active_mask & ~(1 << local)
        if not ((external_mask >> global_spinorb) & 1):
            return 0, external_mask, active_mask
        return sign, external_mask & ~(1 << global_spinorb), active_mask

    def _cre_one(
        self, external_mask: int, active_mask: int, orbital: int, spin: int
    ) -> tuple[int, int, int]:
        global_spinorb = self._so(orbital, spin)
        sign = (
            -1
            if self._count_before(external_mask, active_mask, global_spinorb) & 1
            else 1
        )
        if self._is_active(orbital):
            local = self._active_local(orbital, spin)
            if (active_mask >> local) & 1:
                return 0, external_mask, active_mask
            return sign, external_mask, active_mask | (1 << local)
        if (external_mask >> global_spinorb) & 1:
            return 0, external_mask, active_mask
        return sign, external_mask | (1 << global_spinorb), active_mask

    def ann(self, state: FactoredState, orbital: int, spin: int) -> FactoredState:
        out: FactoredState = {}
        for (external_mask, active_mask), coeff in state.items():
            sign, new_external, new_active = self._ann_one(
                external_mask, active_mask, orbital, spin
            )
            if sign:
                self._add_to(out, (new_external, new_active), sign * coeff)
        return out

    def cre(self, state: FactoredState, orbital: int, spin: int) -> FactoredState:
        out: FactoredState = {}
        for (external_mask, active_mask), coeff in state.items():
            sign, new_external, new_active = self._cre_one(
                external_mask, active_mask, orbital, spin
            )
            if sign:
                self._add_to(out, (new_external, new_active), sign * coeff)
        return out

    def apply_E(self, state: FactoredState, p: int, q: int) -> FactoredState:
        """Apply spin-summed ``E_pq = sum_s cre(p, s) ann(q, s)``."""
        out: FactoredState = {}
        for spin in (0, 1):
            out = self.add(out, self.cre(self.ann(state, q, spin), p, spin))
        return out

    def apply_E_sequence(
        self, state: FactoredState, excitations: Iterable[tuple[int, int]]
    ) -> FactoredState:
        out = state
        for p, q in excitations:
            out = self.apply_E(out, p, q)
        return out

    def apply_chemist_2body_unit(
        self, state: FactoredState, p: int, q: int, r: int, s: int
    ) -> FactoredState:
        """Apply one unit term from the chemist-integral two-body operator.

        The operator matches the term produced by ``_mrpt.apply_2body`` for a
        single nonzero
        ``eri[p, q, r, s]`` entry.
        """
        out: FactoredState = {}
        for spin_q in (0, 1):
            for spin_s in (0, 1):
                tmp = self.ann(state, q, spin_q)
                tmp = self.ann(tmp, s, spin_s)
                tmp = self.cre(tmp, r, spin_s)
                tmp = self.cre(tmp, p, spin_q)
                for key, value in tmp.items():
                    self._add_to(out, key, 0.5 * value)
        return out

    @staticmethod
    def dot(left: FactoredState, right: FactoredState) -> float:
        if len(right) < len(left):
            left, right = right, left
        return sum(coeff * right.get(key, 0.0) for key, coeff in left.items())

    def to_full_determinants(self, state: FactoredState) -> dict[int, float]:
        """Materialize a factored state for oracle tests and diagnostics."""
        out: dict[int, float] = {}
        for (external_mask, active_mask), coeff in state.items():
            full_mask = external_mask | self._active_mask_to_global(active_mask)
            out[full_mask] = out.get(full_mask, 0.0) + coeff
        return out

    def is_cas_external_mask(self, external_mask: int, n_frozen: int = 0) -> bool:
        """Return whether the external sector is CAS-like.

        CAS projection for the IC first-order space removes determinants with
        correlated inactive orbitals still doubly occupied and secondary
        orbitals still empty; the active mask is unrestricted.
        """
        for i in range(n_frozen, self.n_core):
            if not ((external_mask >> self._so(i, 0)) & 1):
                return False
            if not ((external_mask >> self._so(i, 1)) & 1):
                return False
        for a in range(self.active_stop, self.n_orb):
            if (external_mask >> self._so(a, 0)) & 1:
                return False
            if (external_mask >> self._so(a, 1)) & 1:
                return False
        return True

    def project_out_cas(
        self, state: FactoredState, n_frozen: int = 0
    ) -> FactoredState:
        """Drop the CAS-sector component, matching the explicit IC oracle."""
        return {
            key: value
            for key, value in state.items()
            if not self.is_cas_external_mask(key[0], n_frozen=n_frozen)
        }

    def apply_1body_matrix(self, state: FactoredState, h1: np.ndarray) -> FactoredState:
        """Apply a full spin-summed one-body operator in the factored algebra."""
        out: FactoredState = {}
        for q in range(self.n_orb):
            for p in range(self.n_orb):
                hpq = h1[p, q]
                if abs(hpq) < 1e-15:
                    continue
                term = self.apply_E(state, p, q)
                for key, value in term.items():
                    self._add_to(out, key, hpq * value)
        return out

    def apply_2body_matrix(
        self, state: FactoredState, eri_chem: np.ndarray
    ) -> FactoredState:
        """Apply the chemist-integral two-body operator in the factored algebra."""
        out: FactoredState = {}
        for p in range(self.n_orb):
            for q in range(self.n_orb):
                for r in range(self.n_orb):
                    for s in range(self.n_orb):
                        value = eri_chem[p, q, r, s]
                        if abs(value) < 1e-15:
                            continue
                        term = self.apply_chemist_2body_unit(state, p, q, r, s)
                        for key, coeff in term.items():
                            self._add_to(out, key, value * coeff)
        return out


def contracted_excitation_labels(
    n_core: int, n_active: int, n_orb: int, n_frozen: int = 0
) -> list[ExcitationLabel]:
    """Raw IC-CASPT2 excitation labels, ordered like ``_mrpt._ic_caspt2_corr``."""
    active = list(range(n_core, n_core + n_active))
    inactive = list(range(n_frozen, n_core))
    secondary = list(range(n_core + n_active, n_orb))
    occ = inactive + active
    virt = active + secondary
    exc = [(p, q) for p in virt for q in occ if p != q]
    labels: list[ExcitationLabel] = [((p, q),) for p, q in exc]
    labels.extend(((r, s), (p, q)) for p, q in exc for r, s in exc)
    return labels


def contracted_candidates(
    n_core: int, n_active: int, n_orb: int, n_frozen: int = 0
) -> list[ContractedCandidate]:
    """Return raw IC candidates with stable class/grouping metadata."""
    active_start = n_core
    active_stop = n_core + n_active
    candidates = []
    for label in contracted_excitation_labels(
        n_core, n_active, n_orb, n_frozen=n_frozen
    ):
        inactive_holes = tuple(q for _, q in label if n_frozen <= q < n_core)
        secondary_particles = tuple(p for p, _ in label if p >= active_stop)
        active_creations = tuple(p for p, _ in label if active_start <= p < active_stop)
        active_annihilations = tuple(q for _, q in label if active_start <= q < active_stop)
        candidates.append(
            ContractedCandidate(
                label=label,
                order=len(label),
                external_class=(len(inactive_holes), len(secondary_particles)),
                openmolcas_cases=openmolcas_case_labels(
                    (len(inactive_holes), len(secondary_particles))
                ),
                external_signature=(
                    tuple(sorted(inactive_holes)),
                    tuple(sorted(secondary_particles)),
                ),
                inactive_holes=inactive_holes,
                secondary_particles=secondary_particles,
                active_creations=active_creations,
                active_annihilations=active_annihilations,
            )
        )
    return candidates


def build_factored_ic_system_dense(
    h1e_mo,
    h2e_mo,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    n_frozen: int = 0,
) -> FactoredICSystem:
    """Build dense factored IC matrices for small validation cases.

    This mirrors the explicit determinant-space IC engine but builds the raw
    contracted states and matrix elements in the phase-exact active/external
    factorization.  It is a validation scaffold for the large-CAS route, not
    the production scalable block solve.
    """
    from ._mrpt import _semicanonical_prep

    prepared = _semicanonical_prep(
        h1e_mo, h2e_mo, n_core, n_active, n_active_elec, ms2
    )
    n_orb = prepared["norb"]
    h1 = prepared["h1"]
    eri = prepared["eri"]
    fock = prepared["F"]
    fac = ActiveExternalFactorization.from_full_reference(
        prepared["ref"], n_core, n_active, n_orb
    )
    raw_candidates = contracted_candidates(n_core, n_active, n_orb, n_frozen=n_frozen)

    states: list[FactoredState] = []
    kept_candidates = []
    for candidate in raw_candidates:
        state = fac.project_out_cas(
            fac.apply_E_sequence(fac.ref, candidate.label), n_frozen=n_frozen
        )
        norm = np.sqrt(fac.dot(state, state))
        if norm < 1e-14:
            continue
        kept_candidates.append(candidate)
        states.append({key: value / norm for key, value in state.items()})

    n_cand = len(states)
    if n_cand == 0:
        return FactoredICSystem(
            candidates=(),
            overlap=np.empty((0, 0)),
            h0=np.empty((0, 0)),
            coupling=np.empty(0),
            e0=0.0,
        )

    h_ref = fac.add(
        fac.apply_1body_matrix(fac.ref, h1),
        fac.apply_2body_matrix(fac.ref, eri),
    )
    f_ref = fac.apply_1body_matrix(fac.ref, fock)
    e0 = fac.dot(fac.ref, f_ref)
    f_states = [fac.apply_1body_matrix(state, fock) for state in states]

    overlap = np.empty((n_cand, n_cand))
    h0 = np.empty((n_cand, n_cand))
    for i, bra in enumerate(states):
        for j, ket in enumerate(states):
            overlap[i, j] = fac.dot(bra, ket)
            h0[i, j] = fac.dot(bra, f_states[j])
    h0 = 0.5 * (h0 + h0.T)
    coupling = np.array([fac.dot(state, h_ref) for state in states])
    return FactoredICSystem(
        candidates=tuple(kept_candidates),
        overlap=overlap,
        h0=h0,
        coupling=coupling,
        e0=e0,
    )


def solve_factored_ic_system(system: FactoredICSystem, thresh: float = 1e-8):
    """Solve a dense factored IC system in the metric-orthonormal basis."""
    n_cand = len(system.candidates)
    if n_cand == 0:
        return 0.0, 0

    metric_eig, metric_vec = eigh(system.overlap)
    keep = metric_eig > thresh
    transform = metric_vec[:, keep] / np.sqrt(metric_eig[keep])
    h0_orth = transform.T @ system.h0 @ transform
    coupling_orth = transform.T @ system.coupling
    n_basis = int(keep.sum())

    denominators, eigvec = eigh(h0_orth - system.e0 * np.eye(n_basis))
    coupling_diag = eigvec.T @ coupling_orth
    e_corr = float(-np.sum(coupling_diag**2 / denominators))
    return e_corr, n_basis


def _metric_block_transform(
    system: FactoredICSystem,
    keys: Iterable[object],
    thresh: float = 1e-8,
):
    """Build a block-diagonal S^(-1/2) transform and per-column block keys."""
    n_cand = len(system.candidates)
    transforms = []
    block_keys = []
    for metric_block in _metric_blocks(system, keys, thresh=thresh):
        local = metric_block.transform
        embedded = np.zeros((n_cand, local.shape[1]))
        embedded[metric_block.candidate_indices, :] = local
        transforms.append(embedded)
        block_keys.extend([metric_block.key] * local.shape[1])
    if not transforms:
        return np.zeros((n_cand, 0)), []
    return np.hstack(transforms), block_keys


def _metric_blocks(
    system: FactoredICSystem,
    keys: Iterable[object],
    thresh: float = 1e-8,
) -> tuple[_MetricBlock, ...]:
    """Return local S^(-1/2) transforms for caller-provided raw blocks."""
    groups: dict[object, list[int]] = {}
    for idx, key in enumerate(keys):
        groups.setdefault(key, []).append(idx)

    metric_blocks = []
    for key, indices in groups.items():
        candidate_indices = np.asarray(indices, dtype=int)
        block = system.overlap[np.ix_(candidate_indices, candidate_indices)]
        metric_eig, metric_vec = eigh(block)
        keep = metric_eig > thresh
        if not np.any(keep):
            continue
        transform = metric_vec[:, keep] / np.sqrt(metric_eig[keep])
        metric_blocks.append(_MetricBlock(key, candidate_indices, transform))
    return tuple(metric_blocks)


def signature_raw_block_system_from_factored(
    system: FactoredICSystem, coupling_tol: float = 0.0
) -> SignatureRawBlockSystem:
    """Slice a dense validation system into raw signature-block matrices."""
    groups: dict[object, list[int]] = {}
    for idx, candidate in enumerate(system.candidates):
        groups.setdefault(candidate.external_signature, []).append(idx)

    blocks = []
    for key, indices in groups.items():
        idx = np.asarray(indices, dtype=int)
        blocks.append(
            SignatureRawBlock(
                key=key,
                candidate_indices=idx,
                overlap=system.overlap[np.ix_(idx, idx)],
                h0=system.h0[np.ix_(idx, idx)],
                coupling=system.coupling[idx],
            )
        )

    couplings = []
    for left_block, left in enumerate(blocks):
        for right_block in range(left_block + 1, len(blocks)):
            right = blocks[right_block]
            matrix = system.h0[
                np.ix_(left.candidate_indices, right.candidate_indices)
            ]
            if np.linalg.norm(matrix) > coupling_tol:
                couplings.append(
                    SignatureRawBlockCoupling(left_block, right_block, matrix)
                )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=system.e0,
    )


def no_inactive_external_rdm_overlap_coupling(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """RDM/active-CI S/V for no-inactive singly/doubly external candidates.

    The full singly-external ``(0, 1)`` sector is evaluated with the
    spin-resolved active/external factorization pinned by
    ``tests/test_caspt2_rdm_factorization.py``.  This covers both simple
    ``E_at`` candidates and active-rearranged ``E_at E_xy`` / ``E_xy E_at``
    candidates without
    materializing a full determinant IC space.  The doubly-external ``(0, 2)``
    sector keeps the pinned closed two-RDM overlap and ``(av|bw) Gamma``
    coupling formulas from ``references/caspt2_rdm_assemble_scratch.py``.
    Returned matrices are normalized to the same unit-diagonal candidate
    convention as :func:`build_factored_ic_system_dense`.
    """
    from ._casci import casci
    from ._rdm import make_rdm12

    norb = h1.shape[0]
    if n_active > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=0,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    _, dm2 = make_rdm12(cas.ci_coeffs, cas.determinants, n_active)
    # The simple doubly-external overlap uses the Hermitian two-RDM pair
    # symmetry Γ[p,q,r,s] = Γ[r,s,p,q].  Canonicalize that symmetry here so
    # RDM roundoff does not leak into exact half-integer overlaps and raw-block
    # parity against the determinant-space oracle.
    dm2 = 0.5 * (dm2 + dm2.transpose(2, 3, 0, 1))
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=0,
        n_active=n_active,
        n_orb=norb,
    )
    h_ref = fac.add(
        fac.apply_1body_matrix(fac.ref, h1),
        fac.apply_2body_matrix(fac.ref, eri_chem),
    )

    active_states = {}
    for idx, candidate in enumerate(candidates):
        if not _no_inactive_singly_external_candidate(candidate, n_active):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        active_states[idx] = {
            key: value / norm for key, value in state.items()
        }

    descriptors = {}
    for idx, candidate in enumerate(candidates):
        if idx in active_states:
            continue
        descriptor = _no_inactive_external_descriptor(candidate, n_active)
        if descriptor is None or descriptor[0] != "d":
            continue
        norm2 = _no_inactive_external_overlap(descriptor, descriptor, dm2)
        if norm2 <= thresh:
            continue
        descriptors[idx] = (descriptor, float(np.sqrt(norm2)))

    indices = tuple(sorted(set(active_states) | set(descriptors)))
    n_sel = len(indices)
    overlap = np.empty((n_sel, n_sel))
    coupling = np.empty(n_sel)
    for i, left_idx in enumerate(indices):
        left_state = active_states.get(left_idx)
        left_descriptor = descriptors.get(left_idx)
        if left_state is not None:
            coupling[i] = fac.dot(left_state, h_ref)
        else:
            left, left_norm = left_descriptor
            coupling[i] = (
                _no_inactive_external_coupling(left, None, eri_chem, dm2)
                / left_norm
            )
        for j, right_idx in enumerate(indices):
            right_state = active_states.get(right_idx)
            right_descriptor = descriptors.get(right_idx)
            if left_state is not None and right_state is not None:
                overlap[i, j] = fac.dot(left_state, right_state)
            elif left_descriptor is not None and right_descriptor is not None:
                left, left_norm = left_descriptor
                right, right_norm = right_descriptor
                overlap[i, j] = (
                    _no_inactive_external_overlap(left, right, dm2)
                    / (left_norm * right_norm)
                )
            else:
                overlap[i, j] = 0.0
    return indices, overlap, coupling


def raw_block_system_with_no_inactive_external_rdm_sv(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return a raw block system with no-inactive S/V from RDM intermediates."""
    indices, overlap, coupling = no_inactive_external_rdm_overlap_coupling(
        candidates,
        h1,
        eri_chem,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_coupling = block.coupling.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        block_coupling[local_idx] = coupling[formula_idx]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block_coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def no_inactive_external_active_ci_h0(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray, float]:
    """Block-local no-inactive H0 from spin-resolved active-CI intermediates."""
    from ._casci import casci
    from ._mrpt import _generalized_fock
    from ._rdm import make_rdm1

    norb = h1.shape[0]
    if n_active > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=0,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    dm1 = make_rdm1(cas.ci_coeffs, cas.determinants, n_active)
    fock = _generalized_fock(h1, eri_chem, dm1, n_core=0, n_act=n_active)
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=0,
        n_active=n_active,
        n_orb=norb,
    )
    f_ref = fac.apply_1body_matrix(fac.ref, fock)
    e0 = fac.dot(fac.ref, f_ref)

    indices = []
    states = []
    for idx, candidate in enumerate(candidates):
        if not _no_inactive_external_candidate(candidate, n_active):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        indices.append(idx)
        states.append({key: value / norm for key, value in state.items()})

    f_states = [fac.apply_1body_matrix(state, fock) for state in states]
    n_sel = len(states)
    h0 = np.empty((n_sel, n_sel))
    for i, bra in enumerate(states):
        for j, f_ket in enumerate(f_states):
            h0[i, j] = fac.dot(bra, f_ket)
    h0 = 0.5 * (h0 + h0.T)
    return tuple(indices), h0, e0


def raw_block_system_with_no_inactive_external_active_ci_h0(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return a raw block system with no-inactive local H0 from active CI."""
    indices, h0, _ = no_inactive_external_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_h0 = block.h0.copy()
        block_h0[np.ix_(local_idx, local_idx)] = h0[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block.overlap,
                h0=block_h0,
                coupling=block.coupling,
            )
        )

    couplings = []
    for coupling in system.couplings:
        left = system.blocks[coupling.left_block]
        right = system.blocks[coupling.right_block]
        left_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(left.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        right_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(right.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not left_formula or not right_formula:
            couplings.append(coupling)
            continue

        left_local = np.asarray([local for local, _ in left_formula], dtype=int)
        right_local = np.asarray([local for local, _ in right_formula], dtype=int)
        left_formula_idx = np.asarray([pos for _, pos in left_formula], dtype=int)
        right_formula_idx = np.asarray([pos for _, pos in right_formula], dtype=int)
        coupling_h0 = coupling.h0.copy()
        coupling_h0[np.ix_(left_local, right_local)] = h0[
            np.ix_(left_formula_idx, right_formula_idx)
        ]
        couplings.append(
            SignatureRawBlockCoupling(
                coupling.left_block,
                coupling.right_block,
                coupling_h0,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=system.e0,
    )


def raw_block_system_with_no_inactive_external_active_ci(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return a raw block system with no-inactive S/V/H0 from active CI."""
    with_sv = raw_block_system_with_no_inactive_external_rdm_sv(
        system,
        candidates,
        h1,
        eri_chem,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    return raw_block_system_with_no_inactive_external_active_ci_h0(
        with_sv,
        candidates,
        h1,
        eri_chem,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )


def build_no_inactive_external_active_ci_raw_system(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
    coupling_tol: float = 0.0,
) -> SignatureRawBlockSystem:
    """Build a no-inactive raw block system from active-CI intermediates."""
    sv_indices, overlap, coupling = no_inactive_external_rdm_overlap_coupling(
        candidates,
        h1,
        eri_chem,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    h0_indices, h0, e0 = no_inactive_external_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != h0_indices:
        message = "no-inactive S/V and H0 builders selected different candidates"
        raise ValueError(message)

    selected_pos = {idx: pos for pos, idx in enumerate(sv_indices)}
    groups: dict[object, list[int]] = {}
    for idx in sv_indices:
        groups.setdefault(candidates[idx].external_signature, []).append(idx)

    blocks = []
    block_formula_positions = []
    for key, indices in groups.items():
        candidate_indices = np.asarray(indices, dtype=int)
        formula_idx = np.asarray([selected_pos[idx] for idx in indices], dtype=int)
        block_formula_positions.append(formula_idx)
        blocks.append(
            SignatureRawBlock(
                key=key,
                candidate_indices=candidate_indices,
                overlap=overlap[np.ix_(formula_idx, formula_idx)],
                h0=h0[np.ix_(formula_idx, formula_idx)],
                coupling=coupling[formula_idx],
            )
        )

    couplings = []
    for left_block, left_idx in enumerate(block_formula_positions):
        for right_block in range(left_block + 1, len(block_formula_positions)):
            right_idx = block_formula_positions[right_block]
            matrix = h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > coupling_tol:
                couplings.append(
                    SignatureRawBlockCoupling(left_block, right_block, matrix)
                )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=e0,
    )


def inactive_to_active_rdm_overlap_coupling(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """RDM S/V for inactive-to-active singles ``E_ti|0>``."""
    from ._casci import casci
    from ._rdm import make_rdm12

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    dm1, dm2 = make_rdm12(cas.ci_coeffs, cas.determinants, n_active)
    f_inactive = h1.copy()
    for i in range(n_core):
        f_inactive += 2.0 * eri_chem[:, :, i, i] - eri_chem[:, i, i, :]

    descriptors = []
    for idx, candidate in enumerate(candidates):
        descriptor = _inactive_to_active_descriptor(candidate, n_core, n_active)
        if descriptor is None:
            continue
        norm2 = _inactive_to_active_overlap(descriptor, descriptor, dm1)
        if norm2 <= thresh:
            continue
        descriptors.append((idx, descriptor, float(np.sqrt(norm2))))

    n_sel = len(descriptors)
    overlap = np.empty((n_sel, n_sel))
    coupling = np.empty(n_sel)
    for row, (_, left, left_norm) in enumerate(descriptors):
        coupling[row] = (
            _inactive_to_active_coupling(left, f_inactive, eri_chem, dm1, dm2)
            / left_norm
        )
        for col, (_, right, right_norm) in enumerate(descriptors):
            overlap[row, col] = (
                _inactive_to_active_overlap(left, right, dm1)
                / (left_norm * right_norm)
            )

    return tuple(idx for idx, _, _ in descriptors), overlap, coupling


def raw_block_system_with_inactive_to_active_rdm_sv(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return a raw block system with inactive-to-active single S/V from RDMs."""
    indices, overlap, coupling = inactive_to_active_rdm_overlap_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_coupling = block.coupling.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        block_coupling[local_idx] = coupling[formula_idx]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block_coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def inactive_to_active_active_ci_h0(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray, float]:
    """Block-local inactive-to-active single H0 from active-CI intermediates."""
    from ._casci import casci
    from ._mrpt import _generalized_fock
    from ._rdm import make_rdm1

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    dm1 = make_rdm1(cas.ci_coeffs, cas.determinants, n_active)
    fock = _generalized_fock(h1, eri_chem, dm1, n_core=n_core, n_act=n_active)
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    f_ref = fac.apply_1body_matrix(fac.ref, fock)
    e0 = fac.dot(fac.ref, f_ref)

    indices = []
    states = []
    for idx, candidate in enumerate(candidates):
        descriptor = _inactive_to_active_descriptor(candidate, n_core, n_active)
        if descriptor is None:
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        indices.append(idx)
        states.append({key: value / norm for key, value in state.items()})

    f_states = [fac.apply_1body_matrix(state, fock) for state in states]
    n_sel = len(states)
    h0 = np.empty((n_sel, n_sel))
    for i, bra in enumerate(states):
        for j, f_ket in enumerate(f_states):
            h0[i, j] = fac.dot(bra, f_ket)
    h0 = 0.5 * (h0 + h0.T)
    return tuple(indices), h0, e0


def raw_block_system_with_inactive_to_active_active_ci_h0(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return a raw block system with inactive-to-active single H0 from CI."""
    indices, h0, _ = inactive_to_active_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_h0 = block.h0.copy()
        block_h0[np.ix_(local_idx, local_idx)] = h0[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block.overlap,
                h0=block_h0,
                coupling=block.coupling,
            )
        )

    couplings = []
    for coupling in system.couplings:
        left = system.blocks[coupling.left_block]
        right = system.blocks[coupling.right_block]
        left_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(left.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        right_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(right.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not left_formula or not right_formula:
            couplings.append(coupling)
            continue

        left_local = np.asarray([local for local, _ in left_formula], dtype=int)
        right_local = np.asarray([local for local, _ in right_formula], dtype=int)
        left_formula_idx = np.asarray([pos for _, pos in left_formula], dtype=int)
        right_formula_idx = np.asarray([pos for _, pos in right_formula], dtype=int)
        coupling_h0 = coupling.h0.copy()
        coupling_h0[np.ix_(left_local, right_local)] = h0[
            np.ix_(left_formula_idx, right_formula_idx)
        ]
        couplings.append(
            SignatureRawBlockCoupling(
                coupling.left_block,
                coupling.right_block,
                coupling_h0,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=system.e0,
    )


def raw_block_system_with_inactive_to_active_active_ci(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return a raw block system with inactive-to-active single S/V/H0."""
    with_sv = raw_block_system_with_inactive_to_active_rdm_sv(
        system,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    return raw_block_system_with_inactive_to_active_active_ci_h0(
        with_sv,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )


def build_inactive_to_active_active_ci_raw_system(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
    coupling_tol: float = 0.0,
) -> SignatureRawBlockSystem:
    """Build inactive-to-active single raw blocks from active-CI intermediates."""
    sv_indices, overlap, coupling = inactive_to_active_rdm_overlap_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    h0_indices, h0, e0 = inactive_to_active_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != h0_indices:
        message = "inactive-to-active S/V and H0 builders selected different candidates"
        raise ValueError(message)

    selected_pos = {idx: pos for pos, idx in enumerate(sv_indices)}
    groups: dict[object, list[int]] = {}
    for idx in sv_indices:
        groups.setdefault(candidates[idx].external_signature, []).append(idx)

    blocks = []
    block_formula_positions = []
    for key, indices in groups.items():
        candidate_indices = np.asarray(indices, dtype=int)
        formula_idx = np.asarray([selected_pos[idx] for idx in indices], dtype=int)
        block_formula_positions.append(formula_idx)
        blocks.append(
            SignatureRawBlock(
                key=key,
                candidate_indices=candidate_indices,
                overlap=overlap[np.ix_(formula_idx, formula_idx)],
                h0=h0[np.ix_(formula_idx, formula_idx)],
                coupling=coupling[formula_idx],
            )
        )

    couplings = []
    for left_block, left_idx in enumerate(block_formula_positions):
        for right_block in range(left_block + 1, len(block_formula_positions)):
            right_idx = block_formula_positions[right_block]
            matrix = h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > coupling_tol:
                couplings.append(
                    SignatureRawBlockCoupling(left_block, right_block, matrix)
                )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=e0,
    )


def inactive_pair_to_active_pair_distinct_overlap(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Spin-resolved active-CI overlap for distinct-hole ``(2,0)`` doubles."""
    from ._casci import casci

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    pair_cache: dict[tuple[int, int, int, int], float] = {}

    def pair_add_overlap(t: int, u: int, v: int, w: int) -> float:
        key = (t, u, v, w)
        if key in pair_cache:
            return pair_cache[key]
        value = 0.0
        for spin_t in (0, 1):
            for spin_u in (0, 1):
                bra = fac.cre(fac.cre(fac.ref, u, spin_u), t, spin_t)
                ket = fac.cre(fac.cre(fac.ref, w, spin_u), v, spin_t)
                value += fac.dot(bra, ket)
        pair_cache[key] = value
        return value

    descriptors = []
    for idx, candidate in enumerate(candidates):
        descriptor = _inactive_pair_to_active_pair_distinct_descriptor(
            candidate,
            n_core,
            n_active,
        )
        if descriptor is None:
            continue
        norm2 = _inactive_pair_to_active_pair_distinct_raw_overlap(
            descriptor,
            descriptor,
            pair_add_overlap,
        )
        if norm2 <= thresh:
            continue
        descriptors.append((idx, descriptor, float(np.sqrt(norm2))))

    n_sel = len(descriptors)
    overlap = np.empty((n_sel, n_sel))
    for row, (_, left, left_norm) in enumerate(descriptors):
        for col, (_, right, right_norm) in enumerate(descriptors):
            overlap[row, col] = (
                _inactive_pair_to_active_pair_distinct_raw_overlap(
                    left,
                    right,
                    pair_add_overlap,
                )
                / (left_norm * right_norm)
            )

    return tuple(idx for idx, _, _ in descriptors), overlap


def raw_block_system_with_inactive_pair_distinct_overlap(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return a raw block system with distinct-hole I2 overlap from active CI."""
    indices, overlap = inactive_pair_to_active_pair_distinct_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block.coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def inactive_pair_to_active_pair_same_hole_overlap(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Spin-resolved active-CI overlap for same-hole ``(2,0)`` doubles."""
    from ._casci import casci

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    pair_cache: dict[tuple[int, int, int, int], float] = {}

    def same_hole_pair_overlap(t: int, u: int, v: int, w: int) -> float:
        key = (t, u, v, w)
        if key in pair_cache:
            return pair_cache[key]
        value = 0.0
        for spin_t in (0, 1):
            for spin_u in (0, 1):
                if spin_t == spin_u:
                    continue
                bra = fac.cre(fac.cre(fac.ref, u, spin_u), t, spin_t)
                direct = fac.cre(fac.cre(fac.ref, w, spin_u), v, spin_t)
                exchange = fac.cre(fac.cre(fac.ref, w, spin_t), v, spin_u)
                value += fac.dot(bra, direct) - fac.dot(bra, exchange)
        pair_cache[key] = value
        return value

    descriptors = []
    for idx, candidate in enumerate(candidates):
        descriptor = _inactive_pair_to_active_pair_same_hole_descriptor(
            candidate,
            n_core,
            n_active,
        )
        if descriptor is None:
            continue
        norm2 = _inactive_pair_to_active_pair_same_hole_raw_overlap(
            descriptor,
            descriptor,
            same_hole_pair_overlap,
        )
        if norm2 <= thresh:
            continue
        descriptors.append((idx, descriptor, float(np.sqrt(norm2))))

    n_sel = len(descriptors)
    overlap = np.empty((n_sel, n_sel))
    for row, (_, left, left_norm) in enumerate(descriptors):
        for col, (_, right, right_norm) in enumerate(descriptors):
            overlap[row, col] = (
                _inactive_pair_to_active_pair_same_hole_raw_overlap(
                    left,
                    right,
                    same_hole_pair_overlap,
                )
                / (left_norm * right_norm)
            )

    return tuple(idx for idx, _, _ in descriptors), overlap


def inactive_pair_to_active_pair_overlap(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Combined simple-I2 overlap for distinct- and same-hole ``(2,0)`` doubles."""
    distinct_indices, distinct_overlap = (
        inactive_pair_to_active_pair_distinct_overlap(
            candidates,
            h1,
            eri_chem,
            n_core,
            n_active,
            n_active_elec,
            ms2,
            thresh=thresh,
        )
    )
    same_indices, same_overlap = inactive_pair_to_active_pair_same_hole_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )

    indices = tuple(sorted(set(distinct_indices) | set(same_indices)))
    n_sel = len(indices)
    overlap = np.zeros((n_sel, n_sel))
    selected_pos = {idx: pos for pos, idx in enumerate(indices)}
    for source_indices, source_overlap in (
        (distinct_indices, distinct_overlap),
        (same_indices, same_overlap),
    ):
        positions = np.asarray(
            [selected_pos[idx] for idx in source_indices],
            dtype=int,
        )
        overlap[np.ix_(positions, positions)] = source_overlap
    return indices, overlap


def inactive_pair_to_active_pair_active_ci_coupling(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Active-CI ``<Phi|H|0>`` for simple ``(2,0)`` I2 doubles."""
    from ._casci import casci

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    h_ref = fac.add(
        fac.apply_1body_matrix(fac.ref, h1),
        fac.apply_2body_matrix(fac.ref, eri_chem),
    )

    indices = []
    coupling = []
    for idx, candidate in enumerate(candidates):
        distinct = _inactive_pair_to_active_pair_distinct_descriptor(
            candidate,
            n_core,
            n_active,
        )
        same = _inactive_pair_to_active_pair_same_hole_descriptor(
            candidate,
            n_core,
            n_active,
        )
        if distinct is None and same is None:
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        normalized = {key: value / norm for key, value in state.items()}
        indices.append(idx)
        coupling.append(fac.dot(normalized, h_ref))

    return tuple(indices), np.asarray(coupling)


def inactive_pair_to_active_pair_active_ci_h0(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray, float]:
    """Active-CI H0 for simple ``(2,0)`` I2 doubles."""
    from ._casci import casci
    from ._mrpt import _generalized_fock
    from ._rdm import make_rdm1

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    dm1 = make_rdm1(cas.ci_coeffs, cas.determinants, n_active)
    fock = _generalized_fock(h1, eri_chem, dm1, n_core=n_core, n_act=n_active)
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    f_ref = fac.apply_1body_matrix(fac.ref, fock)
    e0 = fac.dot(fac.ref, f_ref)

    indices = []
    states = []
    for idx, candidate in enumerate(candidates):
        distinct = _inactive_pair_to_active_pair_distinct_descriptor(
            candidate,
            n_core,
            n_active,
        )
        same = _inactive_pair_to_active_pair_same_hole_descriptor(
            candidate,
            n_core,
            n_active,
        )
        if distinct is None and same is None:
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        indices.append(idx)
        states.append({key: value / norm for key, value in state.items()})

    f_states = [fac.apply_1body_matrix(state, fock) for state in states]
    n_sel = len(states)
    h0 = np.empty((n_sel, n_sel))
    for i, bra in enumerate(states):
        for j, f_ket in enumerate(f_states):
            h0[i, j] = fac.dot(bra, f_ket)
    h0 = 0.5 * (h0 + h0.T)
    return tuple(indices), h0, e0


def raw_block_system_with_inactive_pair_same_hole_overlap(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return a raw block system with same-hole I2 overlap from active CI."""
    indices, overlap = inactive_pair_to_active_pair_same_hole_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block.coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def raw_block_system_with_inactive_pair_overlap(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with simple I2 distinct- and same-hole overlaps."""
    with_distinct = raw_block_system_with_inactive_pair_distinct_overlap(
        system,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    return raw_block_system_with_inactive_pair_same_hole_overlap(
        with_distinct,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )


def raw_block_system_with_inactive_pair_sv(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with simple-I2 overlap and coupling from active CI."""
    sv_indices, overlap = inactive_pair_to_active_pair_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    coupling_indices, coupling = inactive_pair_to_active_pair_active_ci_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != coupling_indices:
        message = "simple-I2 S and V builders selected different candidates"
        raise ValueError(message)
    formula_pos = {idx: pos for pos, idx in enumerate(sv_indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_coupling = block.coupling.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        block_coupling[local_idx] = coupling[formula_idx]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block_coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def raw_block_system_with_inactive_pair_active_ci_h0(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with simple-I2 H0 from active-CI intermediates."""
    indices, h0, _ = inactive_pair_to_active_pair_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_h0 = block.h0.copy()
        block_h0[np.ix_(local_idx, local_idx)] = h0[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block.overlap,
                h0=block_h0,
                coupling=block.coupling,
            )
        )

    couplings = []
    for coupling in system.couplings:
        left = system.blocks[coupling.left_block]
        right = system.blocks[coupling.right_block]
        left_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(left.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        right_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(right.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not left_formula or not right_formula:
            couplings.append(coupling)
            continue

        left_local = np.asarray([local for local, _ in left_formula], dtype=int)
        right_local = np.asarray([local for local, _ in right_formula], dtype=int)
        left_formula_idx = np.asarray([pos for _, pos in left_formula], dtype=int)
        right_formula_idx = np.asarray([pos for _, pos in right_formula], dtype=int)
        coupling_h0 = coupling.h0.copy()
        coupling_h0[np.ix_(left_local, right_local)] = h0[
            np.ix_(left_formula_idx, right_formula_idx)
        ]
        couplings.append(
            SignatureRawBlockCoupling(
                coupling.left_block,
                coupling.right_block,
                coupling_h0,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=system.e0,
    )


def raw_block_system_with_inactive_pair_active_ci(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with simple-I2 S/V/H0 from active-CI intermediates."""
    with_sv = raw_block_system_with_inactive_pair_sv(
        system,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    return raw_block_system_with_inactive_pair_active_ci_h0(
        with_sv,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )


def build_inactive_pair_active_ci_raw_system(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
    coupling_tol: float = 0.0,
) -> SignatureRawBlockSystem:
    """Build simple-I2 raw blocks from active-CI intermediates."""
    sv_indices, overlap = inactive_pair_to_active_pair_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    coupling_indices, coupling = inactive_pair_to_active_pair_active_ci_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    h0_indices, h0, e0 = inactive_pair_to_active_pair_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != coupling_indices or sv_indices != h0_indices:
        message = "simple-I2 S/V/H0 builders selected different candidates"
        raise ValueError(message)

    selected_pos = {idx: pos for pos, idx in enumerate(sv_indices)}
    groups: dict[object, list[int]] = {}
    for idx in sv_indices:
        groups.setdefault(candidates[idx].external_signature, []).append(idx)

    blocks = []
    block_formula_positions = []
    for key, indices in groups.items():
        candidate_indices = np.asarray(indices, dtype=int)
        formula_idx = np.asarray([selected_pos[idx] for idx in indices], dtype=int)
        block_formula_positions.append(formula_idx)
        blocks.append(
            SignatureRawBlock(
                key=key,
                candidate_indices=candidate_indices,
                overlap=overlap[np.ix_(formula_idx, formula_idx)],
                h0=h0[np.ix_(formula_idx, formula_idx)],
                coupling=coupling[formula_idx],
            )
        )

    couplings = []
    for left_block, left_idx in enumerate(block_formula_positions):
        for right_block in range(left_block + 1, len(block_formula_positions)):
            right_idx = block_formula_positions[right_block]
            matrix = h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > coupling_tol:
                couplings.append(
                    SignatureRawBlockCoupling(left_block, right_block, matrix)
                )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=e0,
    )


def inactive_pair_to_active_virtual_active_ci_overlap(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Active-CI overlap for ``(2,1)`` inactive-pair to active+virtual doubles."""
    from ._casci import casci

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )

    indices = []
    states = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (2, 1):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        indices.append(idx)
        states.append({key: value / norm for key, value in state.items()})

    n_sel = len(states)
    overlap = np.empty((n_sel, n_sel))
    for row, left in enumerate(states):
        for col, right in enumerate(states):
            overlap[row, col] = fac.dot(left, right)

    return tuple(indices), overlap


def raw_block_system_with_inactive_pair_active_virtual_overlap(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(2,1)`` inactive-pair active+virtual overlap."""
    indices, overlap = inactive_pair_to_active_virtual_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block.coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def inactive_pair_to_active_virtual_active_ci_coupling(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Active-CI ``<Phi|H|0>`` for ``(2,1)`` active+virtual doubles."""
    from ._casci import casci

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    h_ref = fac.add(
        fac.apply_1body_matrix(fac.ref, h1),
        fac.apply_2body_matrix(fac.ref, eri_chem),
    )

    indices = []
    coupling = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (2, 1):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        normalized = {key: value / norm for key, value in state.items()}
        indices.append(idx)
        coupling.append(fac.dot(normalized, h_ref))

    return tuple(indices), np.asarray(coupling)


def raw_block_system_with_inactive_pair_active_virtual_sv(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(2,1)`` overlap and coupling from active CI."""
    sv_indices, overlap = inactive_pair_to_active_virtual_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    coupling_indices, coupling = inactive_pair_to_active_virtual_active_ci_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != coupling_indices:
        message = "(2,1) S and V builders selected different candidates"
        raise ValueError(message)
    formula_pos = {idx: pos for pos, idx in enumerate(sv_indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_coupling = block.coupling.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        block_coupling[local_idx] = coupling[formula_idx]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block_coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def inactive_pair_to_active_virtual_active_ci_h0(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray, float]:
    """Active-CI H0 for ``(2,1)`` active+virtual doubles."""
    from ._casci import casci
    from ._mrpt import _generalized_fock
    from ._rdm import make_rdm1

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    dm1 = make_rdm1(cas.ci_coeffs, cas.determinants, n_active)
    fock = _generalized_fock(h1, eri_chem, dm1, n_core=n_core, n_act=n_active)
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    f_ref = fac.apply_1body_matrix(fac.ref, fock)
    e0 = fac.dot(fac.ref, f_ref)

    indices = []
    states = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (2, 1):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        indices.append(idx)
        states.append({key: value / norm for key, value in state.items()})

    f_states = [fac.apply_1body_matrix(state, fock) for state in states]
    n_sel = len(states)
    h0 = np.empty((n_sel, n_sel))
    for i, bra in enumerate(states):
        for j, f_ket in enumerate(f_states):
            h0[i, j] = fac.dot(bra, f_ket)
    h0 = 0.5 * (h0 + h0.T)
    return tuple(indices), h0, e0


def raw_block_system_with_inactive_pair_active_virtual_h0(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(2,1)`` H0 from active-CI intermediates."""
    indices, h0, _ = inactive_pair_to_active_virtual_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_h0 = block.h0.copy()
        block_h0[np.ix_(local_idx, local_idx)] = h0[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block.overlap,
                h0=block_h0,
                coupling=block.coupling,
            )
        )

    couplings = []
    for coupling in system.couplings:
        left = system.blocks[coupling.left_block]
        right = system.blocks[coupling.right_block]
        left_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(left.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        right_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(right.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not left_formula or not right_formula:
            couplings.append(coupling)
            continue

        left_local = np.asarray([local for local, _ in left_formula], dtype=int)
        right_local = np.asarray([local for local, _ in right_formula], dtype=int)
        left_formula_idx = np.asarray([pos for _, pos in left_formula], dtype=int)
        right_formula_idx = np.asarray([pos for _, pos in right_formula], dtype=int)
        coupling_h0 = coupling.h0.copy()
        coupling_h0[np.ix_(left_local, right_local)] = h0[
            np.ix_(left_formula_idx, right_formula_idx)
        ]
        couplings.append(
            SignatureRawBlockCoupling(
                coupling.left_block,
                coupling.right_block,
                coupling_h0,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=system.e0,
    )


def raw_block_system_with_inactive_pair_active_virtual_active_ci(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(2,1)`` S/V/H0 from active-CI intermediates."""
    with_sv = raw_block_system_with_inactive_pair_active_virtual_sv(
        system,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    return raw_block_system_with_inactive_pair_active_virtual_h0(
        with_sv,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )


def build_inactive_pair_active_virtual_active_ci_raw_system(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
    coupling_tol: float = 0.0,
) -> SignatureRawBlockSystem:
    """Build ``(2,1)`` active+virtual raw blocks from active-CI intermediates."""
    sv_indices, overlap = inactive_pair_to_active_virtual_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    coupling_indices, coupling = inactive_pair_to_active_virtual_active_ci_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    h0_indices, h0, e0 = inactive_pair_to_active_virtual_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != coupling_indices or sv_indices != h0_indices:
        message = "(2,1) S/V/H0 builders selected different candidates"
        raise ValueError(message)

    selected_pos = {idx: pos for pos, idx in enumerate(sv_indices)}
    groups: dict[object, list[int]] = {}
    for idx in sv_indices:
        groups.setdefault(candidates[idx].external_signature, []).append(idx)

    blocks = []
    block_formula_positions = []
    for key, indices in groups.items():
        candidate_indices = np.asarray(indices, dtype=int)
        formula_idx = np.asarray([selected_pos[idx] for idx in indices], dtype=int)
        block_formula_positions.append(formula_idx)
        blocks.append(
            SignatureRawBlock(
                key=key,
                candidate_indices=candidate_indices,
                overlap=overlap[np.ix_(formula_idx, formula_idx)],
                h0=h0[np.ix_(formula_idx, formula_idx)],
                coupling=coupling[formula_idx],
            )
        )

    couplings = []
    for left_block, left_idx in enumerate(block_formula_positions):
        for right_block in range(left_block + 1, len(block_formula_positions)):
            right_idx = block_formula_positions[right_block]
            matrix = h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > coupling_tol:
                couplings.append(
                    SignatureRawBlockCoupling(left_block, right_block, matrix)
                )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=e0,
    )


def inactive_active_to_virtual_pair_active_ci_overlap(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Active-CI overlap for ``(1,2)`` inactive+active to virtual-pair doubles."""
    from ._casci import casci

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )

    indices = []
    states = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (1, 2):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        indices.append(idx)
        states.append({key: value / norm for key, value in state.items()})

    n_sel = len(states)
    overlap = np.empty((n_sel, n_sel))
    for row, left in enumerate(states):
        for col, right in enumerate(states):
            overlap[row, col] = fac.dot(left, right)

    return tuple(indices), overlap


def raw_block_system_with_inactive_active_to_virtual_pair_overlap(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(1,2)`` inactive+active to virtual-pair overlap."""
    indices, overlap = inactive_active_to_virtual_pair_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block.coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def inactive_active_to_virtual_pair_active_ci_coupling(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Active-CI ``<Phi|H|0>`` for ``(1,2)`` virtual-pair doubles."""
    from ._casci import casci

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    h_ref = fac.add(
        fac.apply_1body_matrix(fac.ref, h1),
        fac.apply_2body_matrix(fac.ref, eri_chem),
    )

    indices = []
    coupling = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (1, 2):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        normalized = {key: value / norm for key, value in state.items()}
        indices.append(idx)
        coupling.append(fac.dot(normalized, h_ref))

    return tuple(indices), np.asarray(coupling)


def raw_block_system_with_inactive_active_to_virtual_pair_sv(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(1,2)`` overlap and coupling from active CI."""
    sv_indices, overlap = inactive_active_to_virtual_pair_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    coupling_indices, coupling = inactive_active_to_virtual_pair_active_ci_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != coupling_indices:
        message = "(1,2) S and V builders selected different candidates"
        raise ValueError(message)
    formula_pos = {idx: pos for pos, idx in enumerate(sv_indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_coupling = block.coupling.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        block_coupling[local_idx] = coupling[formula_idx]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block_coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def inactive_active_to_virtual_pair_active_ci_h0(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray, float]:
    """Active-CI H0 for ``(1,2)`` virtual-pair doubles."""
    from ._casci import casci
    from ._mrpt import _generalized_fock
    from ._rdm import make_rdm1

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    dm1 = make_rdm1(cas.ci_coeffs, cas.determinants, n_active)
    fock = _generalized_fock(h1, eri_chem, dm1, n_core=n_core, n_act=n_active)
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    f_ref = fac.apply_1body_matrix(fac.ref, fock)
    e0 = fac.dot(fac.ref, f_ref)

    indices = []
    states = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (1, 2):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        indices.append(idx)
        states.append({key: value / norm for key, value in state.items()})

    f_states = [fac.apply_1body_matrix(state, fock) for state in states]
    n_sel = len(states)
    h0 = np.empty((n_sel, n_sel))
    for i, bra in enumerate(states):
        for j, f_ket in enumerate(f_states):
            h0[i, j] = fac.dot(bra, f_ket)
    h0 = 0.5 * (h0 + h0.T)
    return tuple(indices), h0, e0


def raw_block_system_with_inactive_active_to_virtual_pair_h0(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(1,2)`` H0 from active-CI intermediates."""
    indices, h0, _ = inactive_active_to_virtual_pair_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_h0 = block.h0.copy()
        block_h0[np.ix_(local_idx, local_idx)] = h0[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block.overlap,
                h0=block_h0,
                coupling=block.coupling,
            )
        )

    couplings = []
    for coupling in system.couplings:
        left = system.blocks[coupling.left_block]
        right = system.blocks[coupling.right_block]
        left_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(left.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        right_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(right.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not left_formula or not right_formula:
            couplings.append(coupling)
            continue

        left_local = np.asarray([local for local, _ in left_formula], dtype=int)
        right_local = np.asarray([local for local, _ in right_formula], dtype=int)
        left_formula_idx = np.asarray([pos for _, pos in left_formula], dtype=int)
        right_formula_idx = np.asarray([pos for _, pos in right_formula], dtype=int)
        coupling_h0 = coupling.h0.copy()
        coupling_h0[np.ix_(left_local, right_local)] = h0[
            np.ix_(left_formula_idx, right_formula_idx)
        ]
        couplings.append(
            SignatureRawBlockCoupling(
                coupling.left_block,
                coupling.right_block,
                coupling_h0,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=system.e0,
    )


def raw_block_system_with_inactive_active_to_virtual_pair_active_ci(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(1,2)`` S/V/H0 from active-CI intermediates."""
    with_sv = raw_block_system_with_inactive_active_to_virtual_pair_sv(
        system,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    return raw_block_system_with_inactive_active_to_virtual_pair_h0(
        with_sv,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )


def build_inactive_active_to_virtual_pair_active_ci_raw_system(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
    coupling_tol: float = 0.0,
) -> SignatureRawBlockSystem:
    """Build ``(1,2)`` virtual-pair raw blocks from active-CI intermediates."""
    sv_indices, overlap = inactive_active_to_virtual_pair_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    coupling_indices, coupling = inactive_active_to_virtual_pair_active_ci_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    h0_indices, h0, e0 = inactive_active_to_virtual_pair_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != coupling_indices or sv_indices != h0_indices:
        message = "(1,2) S/V/H0 builders selected different candidates"
        raise ValueError(message)

    selected_pos = {idx: pos for pos, idx in enumerate(sv_indices)}
    groups: dict[object, list[int]] = {}
    for idx in sv_indices:
        groups.setdefault(candidates[idx].external_signature, []).append(idx)

    blocks = []
    block_formula_positions = []
    for key, indices in groups.items():
        candidate_indices = np.asarray(indices, dtype=int)
        formula_idx = np.asarray([selected_pos[idx] for idx in indices], dtype=int)
        block_formula_positions.append(formula_idx)
        blocks.append(
            SignatureRawBlock(
                key=key,
                candidate_indices=candidate_indices,
                overlap=overlap[np.ix_(formula_idx, formula_idx)],
                h0=h0[np.ix_(formula_idx, formula_idx)],
                coupling=coupling[formula_idx],
            )
        )

    couplings = []
    for left_block, left_idx in enumerate(block_formula_positions):
        for right_block in range(left_block + 1, len(block_formula_positions)):
            right_idx = block_formula_positions[right_block]
            matrix = h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > coupling_tol:
                couplings.append(
                    SignatureRawBlockCoupling(left_block, right_block, matrix)
                )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=e0,
    )


def inactive_pair_to_virtual_pair_active_ci_overlap(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Active-CI overlap for ``(2,2)`` inactive-pair to virtual-pair doubles."""
    from ._casci import casci

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )

    indices = []
    states = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (2, 2):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        indices.append(idx)
        states.append({key: value / norm for key, value in state.items()})

    n_sel = len(states)
    overlap = np.empty((n_sel, n_sel))
    for row, left in enumerate(states):
        for col, right in enumerate(states):
            overlap[row, col] = fac.dot(left, right)

    return tuple(indices), overlap


def raw_block_system_with_inactive_pair_to_virtual_pair_overlap(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(2,2)`` inactive-pair to virtual-pair overlap."""
    indices, overlap = inactive_pair_to_virtual_pair_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block.coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def inactive_pair_to_virtual_pair_active_ci_coupling(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Active-CI ``<Phi|H|0>`` for ``(2,2)`` virtual-pair doubles."""
    from ._casci import casci

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    h_ref = fac.add(
        fac.apply_1body_matrix(fac.ref, h1),
        fac.apply_2body_matrix(fac.ref, eri_chem),
    )

    indices = []
    coupling = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (2, 2):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        normalized = {key: value / norm for key, value in state.items()}
        indices.append(idx)
        coupling.append(fac.dot(normalized, h_ref))

    return tuple(indices), np.asarray(coupling)


def raw_block_system_with_inactive_pair_to_virtual_pair_sv(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(2,2)`` overlap and coupling from active CI."""
    sv_indices, overlap = inactive_pair_to_virtual_pair_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    coupling_indices, coupling = inactive_pair_to_virtual_pair_active_ci_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != coupling_indices:
        message = "(2,2) S and V builders selected different candidates"
        raise ValueError(message)
    formula_pos = {idx: pos for pos, idx in enumerate(sv_indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_coupling = block.coupling.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        block_coupling[local_idx] = coupling[formula_idx]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block_coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def inactive_pair_to_virtual_pair_active_ci_h0(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray, float]:
    """Active-CI H0 for ``(2,2)`` virtual-pair doubles."""
    from ._casci import casci
    from ._mrpt import _generalized_fock
    from ._rdm import make_rdm1

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    dm1 = make_rdm1(cas.ci_coeffs, cas.determinants, n_active)
    fock = _generalized_fock(h1, eri_chem, dm1, n_core=n_core, n_act=n_active)
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    f_ref = fac.apply_1body_matrix(fac.ref, fock)
    e0 = fac.dot(fac.ref, f_ref)

    indices = []
    states = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (2, 2):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        indices.append(idx)
        states.append({key: value / norm for key, value in state.items()})

    f_states = [fac.apply_1body_matrix(state, fock) for state in states]
    n_sel = len(states)
    h0 = np.empty((n_sel, n_sel))
    for i, bra in enumerate(states):
        for j, f_ket in enumerate(f_states):
            h0[i, j] = fac.dot(bra, f_ket)
    h0 = 0.5 * (h0 + h0.T)
    return tuple(indices), h0, e0


def raw_block_system_with_inactive_pair_to_virtual_pair_h0(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(2,2)`` H0 from active-CI intermediates."""
    indices, h0, _ = inactive_pair_to_virtual_pair_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_h0 = block.h0.copy()
        block_h0[np.ix_(local_idx, local_idx)] = h0[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block.overlap,
                h0=block_h0,
                coupling=block.coupling,
            )
        )

    couplings = []
    for coupling in system.couplings:
        left = system.blocks[coupling.left_block]
        right = system.blocks[coupling.right_block]
        left_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(left.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        right_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(right.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not left_formula or not right_formula:
            couplings.append(coupling)
            continue

        left_local = np.asarray([local for local, _ in left_formula], dtype=int)
        right_local = np.asarray([local for local, _ in right_formula], dtype=int)
        left_formula_idx = np.asarray([pos for _, pos in left_formula], dtype=int)
        right_formula_idx = np.asarray([pos for _, pos in right_formula], dtype=int)
        coupling_h0 = coupling.h0.copy()
        coupling_h0[np.ix_(left_local, right_local)] = h0[
            np.ix_(left_formula_idx, right_formula_idx)
        ]
        couplings.append(
            SignatureRawBlockCoupling(
                coupling.left_block,
                coupling.right_block,
                coupling_h0,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=system.e0,
    )


def raw_block_system_with_inactive_pair_to_virtual_pair_active_ci(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(2,2)`` S/V/H0 from active-CI intermediates."""
    with_sv = raw_block_system_with_inactive_pair_to_virtual_pair_sv(
        system,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    return raw_block_system_with_inactive_pair_to_virtual_pair_h0(
        with_sv,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )


def build_inactive_pair_to_virtual_pair_active_ci_raw_system(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
    coupling_tol: float = 0.0,
) -> SignatureRawBlockSystem:
    """Build ``(2,2)`` virtual-pair raw blocks from active-CI intermediates."""
    sv_indices, overlap = inactive_pair_to_virtual_pair_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    coupling_indices, coupling = inactive_pair_to_virtual_pair_active_ci_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    h0_indices, h0, e0 = inactive_pair_to_virtual_pair_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != coupling_indices or sv_indices != h0_indices:
        message = "(2,2) S/V/H0 builders selected different candidates"
        raise ValueError(message)

    selected_pos = {idx: pos for pos, idx in enumerate(sv_indices)}
    groups: dict[object, list[int]] = {}
    for idx in sv_indices:
        groups.setdefault(candidates[idx].external_signature, []).append(idx)

    blocks = []
    block_formula_positions = []
    for key, indices in groups.items():
        candidate_indices = np.asarray(indices, dtype=int)
        formula_idx = np.asarray([selected_pos[idx] for idx in indices], dtype=int)
        block_formula_positions.append(formula_idx)
        blocks.append(
            SignatureRawBlock(
                key=key,
                candidate_indices=candidate_indices,
                overlap=overlap[np.ix_(formula_idx, formula_idx)],
                h0=h0[np.ix_(formula_idx, formula_idx)],
                coupling=coupling[formula_idx],
            )
        )

    couplings = []
    for left_block, left_idx in enumerate(block_formula_positions):
        for right_block in range(left_block + 1, len(block_formula_positions)):
            right_idx = block_formula_positions[right_block]
            matrix = h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > coupling_tol:
                couplings.append(
                    SignatureRawBlockCoupling(left_block, right_block, matrix)
                )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=e0,
    )


def inactive_to_virtual_active_ci_overlap(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Active-CI overlap for ``(1,1)`` inactive-to-virtual candidates."""
    from ._casci import casci

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )

    indices = []
    states = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (1, 1):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        indices.append(idx)
        states.append({key: value / norm for key, value in state.items()})

    n_sel = len(states)
    overlap = np.empty((n_sel, n_sel))
    for row, left in enumerate(states):
        for col, right in enumerate(states):
            overlap[row, col] = fac.dot(left, right)

    return tuple(indices), overlap


def raw_block_system_with_inactive_to_virtual_overlap(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(1,1)`` inactive-to-virtual overlap."""
    indices, overlap = inactive_to_virtual_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block.coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def inactive_to_virtual_active_ci_coupling(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Active-CI ``<Phi|H|0>`` for ``(1,1)`` inactive-to-virtual candidates."""
    from ._casci import casci

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    h_ref = fac.add(
        fac.apply_1body_matrix(fac.ref, h1),
        fac.apply_2body_matrix(fac.ref, eri_chem),
    )

    indices = []
    coupling = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (1, 1):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        normalized = {key: value / norm for key, value in state.items()}
        indices.append(idx)
        coupling.append(fac.dot(normalized, h_ref))

    return tuple(indices), np.asarray(coupling)


def raw_block_system_with_inactive_to_virtual_sv(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(1,1)`` overlap and coupling from active CI."""
    sv_indices, overlap = inactive_to_virtual_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    coupling_indices, coupling = inactive_to_virtual_active_ci_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != coupling_indices:
        message = "(1,1) S and V builders selected different candidates"
        raise ValueError(message)
    formula_pos = {idx: pos for pos, idx in enumerate(sv_indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_overlap = block.overlap.copy()
        block_coupling = block.coupling.copy()
        block_overlap[np.ix_(local_idx, local_idx)] = overlap[
            np.ix_(formula_idx, formula_idx)
        ]
        block_coupling[local_idx] = coupling[formula_idx]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block_overlap,
                h0=block.h0,
                coupling=block_coupling,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=system.couplings,
        e0=system.e0,
    )


def inactive_to_virtual_active_ci_h0(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> tuple[tuple[int, ...], np.ndarray, float]:
    """Active-CI H0 for ``(1,1)`` inactive-to-virtual singles."""
    from ._casci import casci
    from ._mrpt import _generalized_fock
    from ._rdm import make_rdm1

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    dm1 = make_rdm1(cas.ci_coeffs, cas.determinants, n_active)
    fock = _generalized_fock(h1, eri_chem, dm1, n_core=n_core, n_act=n_active)
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    f_ref = fac.apply_1body_matrix(fac.ref, fock)
    e0 = fac.dot(fac.ref, f_ref)

    indices = []
    states = []
    for idx, candidate in enumerate(candidates):
        if candidate.external_class != (1, 1):
            continue
        state = fac.project_out_cas(fac.apply_E_sequence(fac.ref, candidate.label))
        norm2 = fac.dot(state, state)
        if norm2 <= thresh:
            continue
        norm = float(np.sqrt(norm2))
        indices.append(idx)
        states.append({key: value / norm for key, value in state.items()})

    f_states = [fac.apply_1body_matrix(state, fock) for state in states]
    n_sel = len(states)
    h0 = np.empty((n_sel, n_sel))
    for i, bra in enumerate(states):
        for j, f_ket in enumerate(f_states):
            h0[i, j] = fac.dot(bra, f_ket)
    h0 = 0.5 * (h0 + h0.T)
    return tuple(indices), h0, e0


def raw_block_system_with_inactive_to_virtual_h0(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(1,1)`` H0 from active-CI intermediates."""
    indices, h0, _ = inactive_to_virtual_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    formula_pos = {idx: pos for pos, idx in enumerate(indices)}

    blocks = []
    for block in system.blocks:
        local_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(block.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not local_formula:
            blocks.append(block)
            continue

        local_idx = np.asarray([local for local, _ in local_formula], dtype=int)
        formula_idx = np.asarray([pos for _, pos in local_formula], dtype=int)
        block_h0 = block.h0.copy()
        block_h0[np.ix_(local_idx, local_idx)] = h0[
            np.ix_(formula_idx, formula_idx)
        ]
        blocks.append(
            SignatureRawBlock(
                key=block.key,
                candidate_indices=block.candidate_indices,
                overlap=block.overlap,
                h0=block_h0,
                coupling=block.coupling,
            )
        )

    couplings = []
    for coupling in system.couplings:
        left = system.blocks[coupling.left_block]
        right = system.blocks[coupling.right_block]
        left_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(left.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        right_formula = [
            (local, formula_pos[int(candidate_idx)])
            for local, candidate_idx in enumerate(right.candidate_indices)
            if int(candidate_idx) in formula_pos
        ]
        if not left_formula or not right_formula:
            couplings.append(coupling)
            continue

        left_local = np.asarray([local for local, _ in left_formula], dtype=int)
        right_local = np.asarray([local for local, _ in right_formula], dtype=int)
        left_formula_idx = np.asarray([pos for _, pos in left_formula], dtype=int)
        right_formula_idx = np.asarray([pos for _, pos in right_formula], dtype=int)
        coupling_h0 = coupling.h0.copy()
        coupling_h0[np.ix_(left_local, right_local)] = h0[
            np.ix_(left_formula_idx, right_formula_idx)
        ]
        couplings.append(
            SignatureRawBlockCoupling(
                coupling.left_block,
                coupling.right_block,
                coupling_h0,
            )
        )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=system.e0,
    )


def raw_block_system_with_inactive_to_virtual_active_ci(
    system: SignatureRawBlockSystem,
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
) -> SignatureRawBlockSystem:
    """Return raw blocks with ``(1,1)`` S/V/H0 from active-CI intermediates."""
    with_sv = raw_block_system_with_inactive_to_virtual_sv(
        system,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    return raw_block_system_with_inactive_to_virtual_h0(
        with_sv,
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )


def build_inactive_to_virtual_active_ci_raw_system(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-14,
    coupling_tol: float = 0.0,
) -> SignatureRawBlockSystem:
    """Build ``(1,1)`` inactive-to-virtual raw blocks from active-CI data."""
    sv_indices, overlap = inactive_to_virtual_active_ci_overlap(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    coupling_indices, coupling = inactive_to_virtual_active_ci_coupling(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    h0_indices, h0, e0 = inactive_to_virtual_active_ci_h0(
        candidates,
        h1,
        eri_chem,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=thresh,
    )
    if sv_indices != coupling_indices or sv_indices != h0_indices:
        message = "(1,1) S/V/H0 builders selected different candidates"
        raise ValueError(message)

    selected_pos = {idx: pos for pos, idx in enumerate(sv_indices)}
    groups: dict[object, list[int]] = {}
    for idx in sv_indices:
        groups.setdefault(candidates[idx].external_signature, []).append(idx)

    blocks = []
    block_formula_positions = []
    for key, indices in groups.items():
        candidate_indices = np.asarray(indices, dtype=int)
        formula_idx = np.asarray([selected_pos[idx] for idx in indices], dtype=int)
        block_formula_positions.append(formula_idx)
        blocks.append(
            SignatureRawBlock(
                key=key,
                candidate_indices=candidate_indices,
                overlap=overlap[np.ix_(formula_idx, formula_idx)],
                h0=h0[np.ix_(formula_idx, formula_idx)],
                coupling=coupling[formula_idx],
            )
        )

    couplings = []
    for left_block, left_idx in enumerate(block_formula_positions):
        for right_block in range(left_block + 1, len(block_formula_positions)):
            right_idx = block_formula_positions[right_block]
            matrix = h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > coupling_tol:
                couplings.append(
                    SignatureRawBlockCoupling(left_block, right_block, matrix)
                )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=e0,
    )


def build_active_ci_signature_raw_system(
    candidates: tuple[ContractedCandidate, ...],
    h1: np.ndarray,
    eri_chem: np.ndarray,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    n_frozen: int = 0,
    thresh: float = 1e-14,
    coupling_tol: float = 0.0,
) -> SignatureRawBlockSystem:
    """Build all signature raw blocks directly from active-CI intermediates."""
    from ._casci import casci
    from ._mrpt import _generalized_fock
    from ._rdm import make_rdm1

    norb = h1.shape[0]
    active_stop = n_core + n_active
    if active_stop > norb:
        message = "active space exceeds orbital dimension"
        raise ValueError(message)

    h2_phys = eri_chem.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=n_active_elec,
        n_active_orb=n_active,
        n_core=n_core,
        nuclear_repulsion=0.0,
        ms2=ms2,
    )
    dm1 = make_rdm1(cas.ci_coeffs, cas.determinants, n_active)
    fock = _generalized_fock(h1, eri_chem, dm1, n_core=n_core, n_act=n_active)
    fac = ActiveExternalFactorization(
        cas.ci_coeffs,
        cas.determinants,
        n_core=n_core,
        n_active=n_active,
        n_orb=norb,
    )
    h_ref = fac.add(
        fac.apply_1body_matrix(fac.ref, h1),
        fac.apply_2body_matrix(fac.ref, eri_chem),
    )
    f_ref = fac.apply_1body_matrix(fac.ref, fock)
    e0 = fac.dot(fac.ref, f_ref)

    selected_indices = []
    selected_candidates = []
    states = []
    for idx, candidate in enumerate(candidates):
        state = fac.project_out_cas(
            fac.apply_E_sequence(fac.ref, candidate.label),
            n_frozen=n_frozen,
        )
        norm2 = fac.dot(state, state)
        norm = float(np.sqrt(norm2))
        if norm <= thresh:
            continue
        selected_indices.append(idx)
        selected_candidates.append(candidate)
        states.append({key: value / norm for key, value in state.items()})

    if not states:
        return SignatureRawBlockSystem(blocks=(), couplings=(), e0=e0)

    f_states = [fac.apply_1body_matrix(state, fock) for state in states]
    selected_coupling = np.asarray([fac.dot(state, h_ref) for state in states])
    groups: dict[object, list[int]] = {}
    for pos, candidate in enumerate(selected_candidates):
        groups.setdefault(candidate.external_signature, []).append(pos)

    blocks = []
    block_positions = []
    for key, positions in groups.items():
        n_block = len(positions)
        overlap = np.empty((n_block, n_block))
        h0 = np.empty((n_block, n_block))
        for row, left_pos in enumerate(positions):
            left = states[left_pos]
            for col, right_pos in enumerate(positions):
                overlap[row, col] = fac.dot(left, states[right_pos])
                h0[row, col] = fac.dot(left, f_states[right_pos])
        h0 = 0.5 * (h0 + h0.T)
        candidate_indices = np.asarray(
            [selected_indices[pos] for pos in positions], dtype=int
        )
        blocks.append(
            SignatureRawBlock(
                key=key,
                candidate_indices=candidate_indices,
                overlap=overlap,
                h0=h0,
                coupling=selected_coupling[positions],
            )
        )
        block_positions.append(positions)

    couplings = []
    for left_block, left_positions in enumerate(block_positions):
        for right_block in range(left_block + 1, len(block_positions)):
            if not _external_signatures_one_body_reachable(
                blocks[left_block].key,
                blocks[right_block].key,
            ):
                continue
            right_positions = block_positions[right_block]
            matrix = np.empty((len(left_positions), len(right_positions)))
            for row, left_pos in enumerate(left_positions):
                left = states[left_pos]
                for col, right_pos in enumerate(right_positions):
                    matrix[row, col] = 0.5 * (
                        fac.dot(left, f_states[right_pos])
                        + fac.dot(states[right_pos], f_states[left_pos])
                    )
            if np.linalg.norm(matrix) > coupling_tol:
                couplings.append(
                    SignatureRawBlockCoupling(left_block, right_block, matrix)
                )

    return SignatureRawBlockSystem(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        e0=e0,
    )


def _inactive_to_active_descriptor(
    candidate: ContractedCandidate,
    n_core: int,
    n_active: int,
) -> tuple[int, int, int] | None:
    """Return ``(t_local, t_global, i)`` for an inactive-to-active single."""
    if candidate.order != 1 or candidate.external_class != (1, 0):
        return None
    (t, i), = candidate.label
    if n_core <= t < n_core + n_active and 0 <= i < n_core:
        return t - n_core, t, i
    return None


def _inactive_pair_to_active_pair_distinct_descriptor(
    candidate: ContractedCandidate,
    n_core: int,
    n_active: int,
) -> tuple[int, int, int, int] | None:
    """Return ``(t, u, i, j)`` for distinct-hole ``E_ti E_uj|0>`` doubles."""
    if candidate.order != 2 or candidate.external_class != (2, 0):
        return None
    (t, i), (u, j) = candidate.label
    active_stop = n_core + n_active
    if not (n_core <= t < active_stop and n_core <= u < active_stop):
        return None
    if not (0 <= i < n_core and 0 <= j < n_core):
        return None
    if i == j:
        return None
    return t, u, i, j


def _inactive_pair_to_active_pair_same_hole_descriptor(
    candidate: ContractedCandidate,
    n_core: int,
    n_active: int,
) -> tuple[int, int, int] | None:
    """Return ``(t, u, i)`` for same-hole ``E_ti E_ui|0>`` doubles."""
    if candidate.order != 2 or candidate.external_class != (2, 0):
        return None
    (t, i), (u, j) = candidate.label
    active_stop = n_core + n_active
    if not (n_core <= t < active_stop and n_core <= u < active_stop):
        return None
    if not (0 <= i < n_core and j == i):
        return None
    return t, u, i


def _inactive_pair_to_active_pair_distinct_raw_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    pair_add_overlap: Callable[[int, int, int, int], float],
) -> float:
    """Raw distinct-hole I2 overlap from spin-resolved active-CI pairs."""
    t, u, i, j = left
    v, w, k, l = right
    value = 0.0
    if i == k and j == l:
        value += pair_add_overlap(t, u, v, w)
    if i == l and j == k:
        value += pair_add_overlap(t, u, w, v)
    return float(value)


def _inactive_pair_to_active_pair_same_hole_raw_overlap(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    same_hole_pair_overlap: Callable[[int, int, int, int], float],
) -> float:
    """Raw same-hole I2 overlap from spin-resolved active-CI pairs."""
    t, u, i = left
    v, w, k = right
    if i != k:
        return 0.0
    return float(same_hole_pair_overlap(t, u, v, w))


def _inactive_to_active_overlap(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    dm1: np.ndarray,
) -> float:
    """Raw inactive-to-active single overlap formula."""
    t, _, i = left
    u, _, j = right
    if i != j:
        return 0.0
    return 2.0 * (1.0 if t == u else 0.0) - dm1[t, u]


def _inactive_to_active_coupling(
    descriptor: tuple[int, int, int],
    f_inactive: np.ndarray,
    eri_chem: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
) -> float:
    """Raw inactive-to-active ``<Phi|H|0>`` formula."""
    t_local, t, i = descriptor
    active_start = t - t_local
    active = range(active_start, active_start + dm1.shape[0])

    def local(p: int) -> int:
        return p - active_start

    value = 2.0 * f_inactive[t, i]
    value -= sum(f_inactive[p, i] * dm1[local(p), t_local] for p in active)
    value += 2.0 * sum(
        eri_chem[t, i, u, v] * dm1[local(u), local(v)]
        for u in active
        for v in active
    )
    value -= 0.5 * sum(
        eri_chem[t, u, i, v] * dm1[local(u), local(v)]
        for u in active
        for v in active
    )
    value -= 0.5 * sum(
        eri_chem[i, u, t, v] * dm1[local(u), local(v)]
        for u in active
        for v in active
    )
    value -= sum(
        eri_chem[i, u, v, w] * dm2[t_local, local(u), local(v), local(w)]
        for u in active
        for v in active
        for w in active
    )
    return float(value)


def _no_inactive_external_descriptor(
    candidate: ContractedCandidate, n_active: int
) -> tuple[str, int, int] | tuple[str, int, int, int, int] | None:
    """Return the simple no-inactive external descriptor for a raw candidate."""
    if candidate.inactive_holes or candidate.active_creations:
        return None
    if candidate.order == 1:
        (a, t), = candidate.label
        if a >= n_active and 0 <= t < n_active:
            return ("s", a, t)
        return None
    if candidate.order == 2:
        (b, u), (a, t) = candidate.label
        if (
            a >= n_active
            and b >= n_active
            and 0 <= t < n_active
            and 0 <= u < n_active
        ):
            return ("d", a, t, b, u)
    return None


def _no_inactive_singly_external_candidate(
    candidate: ContractedCandidate, n_active: int
) -> bool:
    """Return whether a candidate belongs to the no-inactive ``(0, 1)`` sector."""
    if candidate.inactive_holes or candidate.external_class != (0, 1):
        return False
    return all(0 <= q < n_active for _, q in candidate.label)


def _no_inactive_external_candidate(
    candidate: ContractedCandidate, n_active: int
) -> bool:
    """Return whether a candidate belongs to a no-inactive external sector."""
    if (
        candidate.inactive_holes
        or candidate.external_class not in {(0, 1), (0, 2)}
    ):
        return False
    return all(0 <= q < n_active for _, q in candidate.label)


def _no_inactive_external_overlap(left, right, dm2: np.ndarray):
    """Raw no-inactive simple-external overlap formula."""
    if left[0] == "d" and right[0] == "d":
        _, a, t, b, u = left
        _, c, v, d, w = right
        direct = (1.0 if a == c and b == d else 0.0) * dm2[t, v, u, w]
        exchange = (1.0 if a == d and b == c else 0.0) * dm2[t, w, u, v]
        return direct + exchange
    return 0.0


def _no_inactive_external_coupling(
    descriptor,
    fock_mc: np.ndarray | None,
    eri_chem: np.ndarray,
    dm2: np.ndarray,
):
    """Raw no-inactive simple-external ``<Phi|H|0>`` formula."""
    if descriptor[0] == "s":
        if fock_mc is None:
            message = "simple singly-external coupling requires a Fock matrix"
            raise ValueError(message)
        _, a, t = descriptor
        return fock_mc[t, a]
    _, a, t, b, u = descriptor
    n_active = dm2.shape[0]
    return float(
        np.einsum(
            "vw,vw->",
            eri_chem[a, :n_active, b, :n_active],
            dm2[t, :, u, :],
            optimize=True,
        )
    )


def _block_ranges_from_keys(
    block_keys: tuple[object, ...] | list[object],
) -> tuple[tuple[object, int, int], ...]:
    """Return contiguous ``(key, start, stop)`` ranges for repeated block keys."""
    ranges = []
    start = 0
    while start < len(block_keys):
        key = block_keys[start]
        stop = start + 1
        while stop < len(block_keys) and block_keys[stop] == key:
            stop += 1
        ranges.append((key, start, stop))
        start = stop
    return tuple(ranges)


def _solve_factored_ic_system_block_metric(
    system: FactoredICSystem,
    keys: Iterable[object],
    thresh: float = 1e-8,
):
    """Solve after orthonormalizing the overlap metric in caller-provided blocks."""
    n_cand = len(system.candidates)
    if n_cand == 0:
        return 0.0, 0

    transform, _ = _metric_block_transform(system, keys, thresh=thresh)
    if transform.shape[1] == 0:
        return 0.0, 0

    h0_orth = transform.T @ system.h0 @ transform
    coupling_orth = transform.T @ system.coupling
    n_basis = transform.shape[1]

    denominators, eigvec = eigh(h0_orth - system.e0 * np.eye(n_basis))
    coupling_diag = eigvec.T @ coupling_orth
    e_corr = float(-np.sum(coupling_diag**2 / denominators))
    return e_corr, n_basis


def subset_factored_ic_system(
    system: FactoredICSystem, indices: Iterable[int]
) -> FactoredICSystem:
    """Return a dense subsystem for a selected candidate index list."""
    idx = np.asarray(list(indices), dtype=int)
    return FactoredICSystem(
        candidates=tuple(system.candidates[int(i)] for i in idx),
        overlap=system.overlap[np.ix_(idx, idx)],
        h0=system.h0[np.ix_(idx, idx)],
        coupling=system.coupling[idx],
        e0=system.e0,
    )


def solve_factored_ic_system_independent_signatures(
    system: FactoredICSystem,
    external_class: tuple[int, int],
    thresh: float = 1e-8,
):
    """Solve one external class as independent external-signature blocks."""
    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for idx, candidate in enumerate(system.candidates):
        if candidate.external_class == external_class:
            groups.setdefault(candidate.external_signature, []).append(idx)

    e_corr = 0.0
    n_basis = 0
    for indices in groups.values():
        block = subset_factored_ic_system(system, indices)
        e_block, n_block = solve_factored_ic_system(block, thresh=thresh)
        e_corr += e_block
        n_basis += n_block
    return e_corr, n_basis


def solve_factored_ic_system_class_metric(
    system: FactoredICSystem, thresh: float = 1e-8
):
    """Solve after orthonormalizing the overlap metric per external class."""
    return _solve_factored_ic_system_block_metric(
        system,
        (candidate.external_class for candidate in system.candidates),
        thresh=thresh,
    )


def solve_factored_ic_system_signature_metric(
    system: FactoredICSystem, thresh: float = 1e-8
):
    """Solve after orthonormalizing the metric per external hole/particle block."""
    return solve_orthonormal_ic_system(
        orthonormalize_factored_ic_system_signature(system, thresh=thresh)
    )


def orthonormalize_factored_ic_system_signature(
    system: FactoredICSystem, thresh: float = 1e-8
) -> OrthonormalICSystem:
    """Return the signature-block metric-orthonormal IC linear system."""
    transform, block_keys = _metric_block_transform(
        system,
        (candidate.external_signature for candidate in system.candidates),
        thresh=thresh,
    )
    shifted_h0 = transform.T @ system.h0 @ transform
    shifted_h0 -= system.e0 * np.eye(transform.shape[1])
    coupling = transform.T @ system.coupling
    return OrthonormalICSystem(
        block_keys=tuple(block_keys),
        block_ranges=_block_ranges_from_keys(block_keys),
        shifted_h0=shifted_h0,
        coupling=coupling,
    )


def build_signature_block_linear_system(
    system: FactoredICSystem,
    thresh: float = 1e-8,
    coupling_tol: float = 0.0,
) -> SignatureBlockLinearSystem:
    """Assemble the signature-block operator without a global transform."""
    raw_system = signature_raw_block_system_from_factored(
        system, coupling_tol=coupling_tol
    )
    return build_signature_block_linear_system_from_raw(
        raw_system, thresh=thresh, coupling_tol=coupling_tol
    )


def build_signature_block_linear_system_from_raw(
    system: SignatureRawBlockSystem,
    thresh: float = 1e-8,
    coupling_tol: float = 0.0,
) -> SignatureBlockLinearSystem:
    """Metric-orthonormalize a raw signature-block system."""
    ranges = []
    block_keys = []
    diagonal_blocks = []
    coupling_blocks = []
    transforms = []
    operator_block_by_raw: dict[int, int] = {}
    start = 0

    for raw_block_index, raw_block in enumerate(system.blocks):
        metric_eig, metric_vec = eigh(raw_block.overlap)
        keep = metric_eig > thresh
        if not np.any(keep):
            transforms.append(np.zeros((raw_block.overlap.shape[0], 0)))
            continue
        transform = metric_vec[:, keep] / np.sqrt(metric_eig[keep])
        transforms.append(transform)
        n_local = transform.shape[1]
        stop = start + n_local
        ranges.append((raw_block.key, start, stop))
        block_keys.extend([raw_block.key] * n_local)
        operator_block_by_raw[raw_block_index] = len(ranges) - 1

        shifted = transform.T @ raw_block.h0 @ transform
        shifted -= system.e0 * np.eye(n_local)
        diagonal_blocks.append(shifted)
        coupling_blocks.append(transform.T @ raw_block.coupling)
        start = stop

    matrix_couplings = []
    for coupling in system.couplings:
        if coupling.left_block not in operator_block_by_raw:
            continue
        if coupling.right_block not in operator_block_by_raw:
            continue
        left_transform = transforms[coupling.left_block]
        right_transform = transforms[coupling.right_block]
        matrix = left_transform.T @ coupling.h0 @ right_transform
        if np.linalg.norm(matrix) > coupling_tol:
            matrix_couplings.append(
                SignatureBlockMatrixCoupling(
                    operator_block_by_raw[coupling.left_block],
                    operator_block_by_raw[coupling.right_block],
                    matrix,
                )
            )

    coupling = np.concatenate(coupling_blocks) if coupling_blocks else np.empty(0)
    operator = SignatureBlockOperator(
        block_ranges=tuple(ranges),
        diagonal_blocks=tuple(diagonal_blocks),
        couplings=tuple(matrix_couplings),
        shape=(start, start),
    )
    return SignatureBlockLinearSystem(
        block_keys=tuple(block_keys),
        operator=operator,
        coupling=coupling,
    )


def signature_block_linear_system_from_orthonormal(
    system: OrthonormalICSystem, coupling_tol: float = 0.0
) -> SignatureBlockLinearSystem:
    """Wrap an existing dense orthonormal system as a block linear system."""
    return SignatureBlockLinearSystem(
        block_keys=system.block_keys,
        operator=orthonormal_signature_block_operator(system, tol=coupling_tol),
        coupling=system.coupling,
    )


def solve_orthonormal_ic_system(system: OrthonormalICSystem):
    """Solve an already metric-orthonormal IC linear system."""
    n_basis = system.shifted_h0.shape[0]
    if n_basis == 0:
        return 0.0, 0
    denominators, eigvec = eigh(system.shifted_h0)
    coupling_diag = eigvec.T @ system.coupling
    e_corr = float(-np.sum(coupling_diag**2 / denominators))
    return e_corr, n_basis


def _solve_symmetric_response(
    block: np.ndarray,
    rhs: np.ndarray,
    rcond: float,
    singular_message: str,
) -> np.ndarray:
    """Solve a symmetric response block with a relative singularity cutoff."""
    eigval, eigvec = eigh(block)
    scale = max(1.0, float(np.max(np.abs(eigval)))) * rcond
    keep = np.abs(eigval) > scale
    if not np.any(keep):
        raise np.linalg.LinAlgError(singular_message)
    return eigvec[:, keep] @ ((eigvec[:, keep].T @ rhs) / eigval[keep])


def solve_orthonormal_ic_system_independent_blocks(
    system: OrthonormalICSystem,
    key_filter: Callable[[object], bool] | None = None,
    rcond: float = 1e-12,
):
    """Sum independent solves over selected signature-orthonormal blocks."""
    e_corr = 0.0
    n_basis = 0
    for key, start, stop in system.block_ranges:
        if key_filter is not None and not key_filter(key):
            continue
        block = system.shifted_h0[start:stop, start:stop]
        rhs = system.coupling[start:stop]
        response = _solve_symmetric_response(
            block,
            rhs,
            rcond,
            "singular CASPT2 independent signature block",
        )
        e_corr -= float(np.dot(rhs, response))
        n_basis += stop - start
    return e_corr, n_basis


def _signature_block_preconditioner(
    system: OrthonormalICSystem, rcond: float = 1e-12
):
    """Return a block-diagonal inverse operator over signature blocks."""
    linear_system = signature_block_linear_system_from_orthonormal(system)
    return _signature_block_operator_preconditioner(
        linear_system.operator, rcond=rcond
    )


def _signature_block_operator_preconditioner(
    operator: SignatureBlockOperator, rcond: float = 1e-12
):
    """Return a block-diagonal inverse operator for a block operator."""
    from scipy.sparse.linalg import LinearOperator

    n_basis = operator.shape[0]
    eigensystems = []
    for (_, start, stop), block in zip(
        operator.block_ranges, operator.diagonal_blocks, strict=True
    ):
        eigval, eigvec = eigh(block)
        scale = max(1.0, float(np.max(np.abs(eigval)))) * rcond
        keep = np.abs(eigval) > scale
        if not np.any(keep):
            message = "singular CASPT2 signature preconditioner block"
            raise np.linalg.LinAlgError(message)
        eigensystems.append((start, stop, eigval[keep], eigvec[:, keep]))

    def matvec(vector):
        out = np.zeros_like(vector)
        for start, stop, eigval, eigvec in eigensystems:
            local = vector[start:stop]
            out[start:stop] = eigvec @ ((eigvec.T @ local) / eigval)
        return out

    return LinearOperator((n_basis, n_basis), matvec=matvec, dtype=np.float64)


def orthonormal_signature_block_couplings(
    system: OrthonormalICSystem, tol: float = 1e-12
) -> tuple[SignatureBlockCoupling, ...]:
    """Return nonzero off-diagonal shifted-H0 couplings by signature block."""
    operator = orthonormal_signature_block_operator(system, tol=tol)
    return tuple(
        SignatureBlockCoupling(
            coupling.left_block,
            coupling.right_block,
            float(np.linalg.norm(coupling.matrix)),
        )
        for coupling in operator.couplings
    )


def orthonormal_signature_block_operator(
    system: OrthonormalICSystem, tol: float = 1e-12
) -> SignatureBlockOperator:
    """Build a reusable block operator from an orthonormal shifted-H0 matrix."""
    diagonal_blocks = []
    matrix_couplings = []
    for left_block, (_, left_start, left_stop) in enumerate(system.block_ranges):
        diagonal_blocks.append(
            system.shifted_h0[left_start:left_stop, left_start:left_stop]
        )
        for right_block in range(left_block + 1, len(system.block_ranges)):
            _, right_start, right_stop = system.block_ranges[right_block]
            block = system.shifted_h0[left_start:left_stop, right_start:right_stop]
            norm = float(np.linalg.norm(block))
            if norm > tol:
                matrix_couplings.append(
                    SignatureBlockMatrixCoupling(left_block, right_block, block)
                )
    return SignatureBlockOperator(
        block_ranges=system.block_ranges,
        diagonal_blocks=tuple(diagonal_blocks),
        couplings=tuple(matrix_couplings),
        shape=system.shifted_h0.shape,
    )


def orthonormal_signature_block_components(
    system: OrthonormalICSystem, tol: float = 1e-12
) -> tuple[tuple[int, ...], ...]:
    """Return connected components of the signature-block coupling graph."""
    operator = orthonormal_signature_block_operator(system, tol=tol)
    return signature_block_operator_components(operator, tol=0.0)


def signature_block_operator_components(
    operator: SignatureBlockOperator, tol: float = 1e-12
) -> tuple[tuple[int, ...], ...]:
    """Return connected components for a signature-block operator."""
    n_blocks = len(operator.block_ranges)
    adjacency = [set() for _ in range(n_blocks)]
    for coupling in operator.couplings:
        if np.linalg.norm(coupling.matrix) <= tol:
            continue
        adjacency[coupling.left_block].add(coupling.right_block)
        adjacency[coupling.right_block].add(coupling.left_block)

    components = []
    seen: set[int] = set()
    for root in range(n_blocks):
        if root in seen:
            continue
        stack = [root]
        seen.add(root)
        component = []
        while stack:
            block = stack.pop()
            component.append(block)
            for neighbor in adjacency[block]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def solve_orthonormal_ic_system_block_components(
    system: OrthonormalICSystem,
    tol: float = 1e-12,
    rcond: float = 1e-12,
):
    """Solve disconnected signature-block components independently."""
    linear_system = signature_block_linear_system_from_orthonormal(system)
    return solve_signature_block_linear_system_components(
        linear_system, tol=tol, rcond=rcond
    )


def solve_signature_block_linear_system_components(
    system: SignatureBlockLinearSystem,
    tol: float = 1e-12,
    rcond: float = 1e-12,
):
    """Solve disconnected components of a signature-block linear system."""
    e_corr = 0.0
    n_basis = 0
    components = signature_block_operator_components(system.operator, tol=tol)
    for component in components:
        local_ranges = {}
        local_start = 0
        for block in component:
            _, global_start, global_stop = system.operator.block_ranges[block]
            local_stop = local_start + (global_stop - global_start)
            local_ranges[block] = (
                local_start,
                local_stop,
                global_start,
                global_stop,
                )
            local_start = local_stop

        local_size = local_start
        shifted_h0 = np.zeros((local_size, local_size))
        coupling = np.zeros(local_size)
        for block in component:
            local_start, local_stop, global_start, global_stop = local_ranges[block]
            shifted_h0[local_start:local_stop, local_start:local_stop] = (
                system.operator.diagonal_blocks[block]
            )
            coupling[local_start:local_stop] = system.coupling[
                global_start:global_stop
            ]

        component_blocks = set(component)
        for block_coupling in system.operator.couplings:
            if np.linalg.norm(block_coupling.matrix) <= tol:
                continue
            if block_coupling.left_block not in component_blocks:
                continue
            if block_coupling.right_block not in component_blocks:
                continue
            left = local_ranges[block_coupling.left_block]
            right = local_ranges[block_coupling.right_block]
            left_start, left_stop = left[0], left[1]
            right_start, right_stop = right[0], right[1]
            shifted_h0[left_start:left_stop, right_start:right_stop] = (
                block_coupling.matrix
            )
            shifted_h0[right_start:right_stop, left_start:left_stop] = (
                block_coupling.matrix.T
            )

        response = _solve_symmetric_response(
            shifted_h0,
            coupling,
            rcond,
            "singular CASPT2 signature component",
        )
        e_corr -= float(np.dot(coupling, response))
        n_basis += local_size
    return e_corr, n_basis


def orthonormal_signature_block_matvec(
    system: OrthonormalICSystem, vector: np.ndarray, tol: float = 0.0
) -> np.ndarray:
    """Apply shifted-H0 by traversing signature diagonal/off-diagonal blocks."""
    operator = orthonormal_signature_block_operator(system, tol=tol)
    return signature_block_operator_matvec(operator, vector)


def signature_block_operator_matvec(
    operator: SignatureBlockOperator, vector: np.ndarray
) -> np.ndarray:
    """Apply a precomputed signature-block shifted-H0 operator."""
    if vector.shape != (operator.shape[1],):
        message = (
            f"expected vector shape {(operator.shape[1],)}, got {vector.shape}"
        )
        raise ValueError(message)
    out = np.zeros_like(vector)
    for (_, start, stop), block in zip(
        operator.block_ranges, operator.diagonal_blocks, strict=True
    ):
        out[start:stop] += block @ vector[start:stop]
    for coupling in operator.couplings:
        _, left_start, left_stop = operator.block_ranges[coupling.left_block]
        _, right_start, right_stop = operator.block_ranges[coupling.right_block]
        out[left_start:left_stop] += coupling.matrix @ vector[right_start:right_stop]
        out[right_start:right_stop] += coupling.matrix.T @ vector[left_start:left_stop]
    return out


def solve_orthonormal_ic_system_block_iterative(
    system: OrthonormalICSystem,
    tol: float = 1e-12,
    maxiter: int | None = None,
) -> IterativeICResult:
    """Solve the orthonormal IC system with a signature-block preconditioner."""
    linear_system = signature_block_linear_system_from_orthonormal(system)
    return solve_signature_block_linear_system_iterative(
        linear_system, tol=tol, maxiter=maxiter
    )


def solve_signature_block_linear_system_imaginary(
    system: SignatureBlockLinearSystem,
    imaginary: float,
    tol: float = 1e-12,
    maxiter: int | None = None,
) -> IterativeICResult:
    """Imaginary-shifted (Forsberg-Malmqvist) signature-block solve.

    Hylleraas level-shift-corrected second-order energy (Forsberg &
    Malmqvist, Chem. Phys. Lett. 274, 196 (1997), Eq 11 -- the value
    OpenMolcas reports; identical to the eigenbasis form used by
    ``_mrpt._ic_caspt2_corr``)::

        E2(s) = -S_i |V_i|^2 d_i (d_i^2 + 2s^2) / (d_i^2 + s^2)^2

    In operator form with the symmetric ``A = H̃0 - E0`` of this system and
    ``z = (A^2 + s^2)⁻¹ V``::

        E2(s) = -V.(A z) - s^2 z.(A z) = -(V + s^2 z).(A z)

    so one real SPD solve replaces the eigendecomposition.  ``(A^2 + s^2)``
    is solved with CG preconditioned by the per-block ``(B^2 + s^2)⁻¹`` of
    the diagonal signature blocks (each matvec applies the block operator
    twice).  ``imaginary=0`` falls back to the unshifted GMRES solve.
    """
    from scipy.sparse.linalg import LinearOperator, cg

    if imaginary == 0.0:
        return solve_signature_block_linear_system_iterative(
            system, tol=tol, maxiter=maxiter
        )
    n_basis = system.operator.shape[0]
    if n_basis == 0:
        return IterativeICResult(0.0, 0, 0, 0.0, True)
    if not np.any(np.abs(system.coupling) > 0.0):
        return IterativeICResult(0.0, n_basis, 0, 0.0, True)

    sigma2 = float(imaginary) ** 2

    def _matvec(vector):
        once = signature_block_operator_matvec(system.operator, vector)
        return (
            signature_block_operator_matvec(system.operator, once)
            + sigma2 * vector
        )

    matrix = LinearOperator((n_basis, n_basis), matvec=_matvec, dtype=np.float64)

    # Per-block (B^2 + s^2)⁻¹: SPD by construction, so plain inverses of the
    # small dense diagonal blocks are safe.
    inverses = []
    for (_, start, stop), block in zip(
        system.operator.block_ranges,
        system.operator.diagonal_blocks,
        strict=True,
    ):
        shifted = block @ block + sigma2 * np.eye(block.shape[0])
        inverses.append((start, stop, np.linalg.inv(shifted)))

    def _pre(vector):
        out = np.zeros_like(vector)
        for start, stop, inv in inverses:
            out[start:stop] = inv @ vector[start:stop]
        return out

    precond = LinearOperator((n_basis, n_basis), matvec=_pre, dtype=np.float64)

    n_iter = 0

    def _count(_xk):
        nonlocal n_iter
        n_iter += 1

    z, info = cg(
        matrix,
        system.coupling,
        M=precond,
        rtol=tol,
        atol=tol,
        maxiter=maxiter,
        callback=_count,
    )
    residual = _matvec(z) - system.coupling
    residual_norm = float(np.linalg.norm(residual))
    az = signature_block_operator_matvec(system.operator, z)
    energy = float(-np.dot(system.coupling + sigma2 * z, az))
    return IterativeICResult(
        energy=energy,
        n_basis=n_basis,
        n_iter=n_iter,
        residual_norm=residual_norm,
        converged=(info == 0 and residual_norm <= tol),
    )


def solve_signature_block_linear_system_iterative(
    system: SignatureBlockLinearSystem,
    tol: float = 1e-12,
    maxiter: int | None = None,
) -> IterativeICResult:
    """Solve a signature-block linear system with block GMRES."""
    from scipy.sparse.linalg import LinearOperator, gmres

    n_basis = system.operator.shape[0]
    if n_basis == 0:
        return IterativeICResult(0.0, 0, 0, 0.0, True)
    if not np.any(np.abs(system.coupling) > 0.0):
        return IterativeICResult(0.0, n_basis, 0, 0.0, True)

    n_iter = 0

    def _count(_residual):
        nonlocal n_iter
        n_iter += 1

    matrix = LinearOperator(
        (n_basis, n_basis),
        matvec=lambda vector: signature_block_operator_matvec(
            system.operator, vector
        ),
        dtype=np.float64,
    )
    precond = _signature_block_operator_preconditioner(system.operator)
    restart = min(n_basis, 80)
    x, info = gmres(
        matrix,
        system.coupling,
        M=precond,
        rtol=tol,
        atol=tol,
        restart=restart,
        maxiter=maxiter,
        callback=_count,
        callback_type="pr_norm",
    )
    residual = signature_block_operator_matvec(system.operator, x) - system.coupling
    residual_norm = float(np.linalg.norm(residual))
    energy = float(-np.dot(system.coupling, x))
    return IterativeICResult(
        energy=energy,
        n_basis=n_basis,
        n_iter=n_iter,
        residual_norm=residual_norm,
        converged=(info == 0 and residual_norm <= tol),
    )


def solve_factored_ic_system_signature_iterative(
    system: FactoredICSystem,
    thresh: float = 1e-8,
    tol: float = 1e-12,
    maxiter: int | None = None,
) -> IterativeICResult:
    """Metric-orthonormalize by signature, then solve with block GMRES."""
    block_system = build_signature_block_linear_system(system, thresh=thresh)
    return solve_signature_block_linear_system_iterative(
        block_system, tol=tol, maxiter=maxiter
    )


def solve_factored_ic_system_signature_components(
    system: FactoredICSystem,
    thresh: float = 1e-8,
    tol: float = 1e-12,
):
    """Metric-orthonormalize by signature, then solve graph components."""
    block_system = build_signature_block_linear_system(system, thresh=thresh)
    return solve_signature_block_linear_system_components(block_system, tol=tol)


def solve_factored_ic_system_orthonormal_independent_signatures(
    system: FactoredICSystem,
    external_class: tuple[int, int],
    thresh: float = 1e-8,
):
    """Solve one external class as independent orthonormal signature blocks."""
    orth = orthonormalize_factored_ic_system_signature(system, thresh=thresh)
    return solve_orthonormal_ic_system_independent_blocks(
        orth,
        key_filter=lambda key: (len(key[0]), len(key[1])) == external_class,
    )


def orthonormal_signature_coupling_edges(
    system: OrthonormalICSystem, tol: float = 1e-12
) -> set[tuple[object, object]]:
    """Return nonzero off-diagonal block couplings in an orthonormal system."""
    edges = set()
    for i, left in enumerate(system.block_keys):
        for j in range(i + 1, len(system.block_keys)):
            right = system.block_keys[j]
            if left == right:
                continue
            if abs(system.shifted_h0[i, j]) > tol:
                edges.add((left, right) if repr(left) <= repr(right) else (right, left))
    return edges


def factored_ic_caspt2_corr_dense(
    h1e_mo,
    h2e_mo,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    n_frozen: int = 0,
    thresh: float = 1e-8,
) -> tuple[float, int]:
    """Dense factored IC-CASPT2 oracle for small validation cases."""
    system = build_factored_ic_system_dense(
        h1e_mo,
        h2e_mo,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        n_frozen=n_frozen,
    )
    return solve_factored_ic_system(system, thresh=thresh)


def factored_ic_caspt2_corr_signature_iterative(
    h1e_mo,
    h2e_mo,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    n_frozen: int = 0,
    thresh: float = 1e-8,
    tol: float = 1e-12,
    maxiter: int | None = None,
) -> IterativeICResult:
    """Private end-to-end IC-CASPT2 solve through signature-block GMRES."""
    system = build_factored_ic_system_dense(
        h1e_mo,
        h2e_mo,
        n_core,
        n_active,
        n_active_elec,
        ms2,
        n_frozen=n_frozen,
    )
    return solve_factored_ic_system_signature_iterative(
        system,
        thresh=thresh,
        tol=tol,
        maxiter=maxiter,
    )


def active_ci_caspt2_corr_signature_iterative(
    h1e_mo,
    h2e_mo,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    n_frozen: int = 0,
    thresh: float = 1e-8,
    candidate_thresh: float = 1e-14,
    tol: float = 1e-12,
    maxiter: int | None = None,
    coupling_tol: float = 0.0,
    imaginary: float = 0.0,
) -> IterativeICResult:
    """Private direct active-CI IC-CASPT2 solve through signature blocks.

    ``imaginary`` s applies the Forsberg-Malmqvist shift via the
    level-shift-corrected SPD solve
    (:func:`solve_signature_block_linear_system_imaginary`).
    """
    from ._mrpt import _semicanonical_prep

    prepared = _semicanonical_prep(
        h1e_mo,
        h2e_mo,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_active_elec,
        ms2=ms2,
    )
    candidates = tuple(
        contracted_candidates(
            n_core=n_core,
            n_active=n_active,
            n_orb=prepared["norb"],
            n_frozen=n_frozen,
        )
    )
    raw_system = build_active_ci_signature_raw_system(
        candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_active_elec,
        ms2,
        n_frozen=n_frozen,
        thresh=candidate_thresh,
        coupling_tol=coupling_tol,
    )
    block_system = build_signature_block_linear_system_from_raw(
        raw_system,
        thresh=thresh,
        coupling_tol=coupling_tol,
    )
    return solve_signature_block_linear_system_imaginary(
        block_system,
        imaginary=imaginary,
        tol=tol,
        maxiter=maxiter,
    )


def no_inactive_active_ci_caspt2_corr_signature_iterative(
    h1e_mo,
    h2e_mo,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-8,
    candidate_thresh: float = 1e-14,
    tol: float = 1e-12,
    maxiter: int | None = None,
    coupling_tol: float = 0.0,
) -> IterativeICResult:
    """Private no-inactive IC-CASPT2 solve from active-CI raw blocks."""
    from ._mrpt import _semicanonical_prep

    prepared = _semicanonical_prep(
        h1e_mo,
        h2e_mo,
        n_core=0,
        n_act=n_active,
        n_act_elec=n_active_elec,
        ms2=ms2,
    )
    candidates = tuple(
        contracted_candidates(
            n_core=0,
            n_active=n_active,
            n_orb=prepared["norb"],
        )
    )
    raw_system = build_no_inactive_external_active_ci_raw_system(
        candidates,
        prepared["h1"],
        prepared["eri"],
        n_active,
        n_active_elec,
        ms2,
        thresh=candidate_thresh,
        coupling_tol=coupling_tol,
    )
    block_system = build_signature_block_linear_system_from_raw(
        raw_system,
        thresh=thresh,
        coupling_tol=coupling_tol,
    )
    return solve_signature_block_linear_system_iterative(
        block_system,
        tol=tol,
        maxiter=maxiter,
    )


def inactive_to_active_active_ci_caspt2_corr_signature_iterative(
    h1e_mo,
    h2e_mo,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-8,
    candidate_thresh: float = 1e-14,
    tol: float = 1e-12,
    maxiter: int | None = None,
    coupling_tol: float = 0.0,
) -> IterativeICResult:
    """Private inactive-to-active single CASPT2 solve from raw blocks."""
    from ._mrpt import _semicanonical_prep

    prepared = _semicanonical_prep(
        h1e_mo,
        h2e_mo,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_active_elec,
        ms2=ms2,
    )
    candidates = tuple(
        contracted_candidates(
            n_core=n_core,
            n_active=n_active,
            n_orb=prepared["norb"],
        )
    )
    raw_system = build_inactive_to_active_active_ci_raw_system(
        candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=candidate_thresh,
        coupling_tol=coupling_tol,
    )
    block_system = build_signature_block_linear_system_from_raw(
        raw_system,
        thresh=thresh,
    )
    return solve_signature_block_linear_system_iterative(
        block_system,
        tol=tol,
        maxiter=maxiter,
    )


def inactive_pair_active_ci_caspt2_corr_signature_iterative(
    h1e_mo,
    h2e_mo,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-8,
    candidate_thresh: float = 1e-14,
    tol: float = 1e-12,
    maxiter: int | None = None,
    coupling_tol: float = 0.0,
) -> IterativeICResult:
    """Private simple-I2 CASPT2 solve from active-CI raw blocks."""
    from ._mrpt import _semicanonical_prep

    prepared = _semicanonical_prep(
        h1e_mo,
        h2e_mo,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_active_elec,
        ms2=ms2,
    )
    candidates = tuple(
        contracted_candidates(
            n_core=n_core,
            n_active=n_active,
            n_orb=prepared["norb"],
        )
    )
    raw_system = build_inactive_pair_active_ci_raw_system(
        candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=candidate_thresh,
        coupling_tol=coupling_tol,
    )
    block_system = build_signature_block_linear_system_from_raw(
        raw_system,
        thresh=thresh,
    )
    return solve_signature_block_linear_system_iterative(
        block_system,
        tol=tol,
        maxiter=maxiter,
    )


def inactive_pair_active_virtual_active_ci_caspt2_corr_signature_iterative(
    h1e_mo,
    h2e_mo,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-8,
    candidate_thresh: float = 1e-14,
    tol: float = 1e-12,
    maxiter: int | None = None,
    coupling_tol: float = 0.0,
) -> IterativeICResult:
    """Private ``(2,1)`` active+virtual CASPT2 solve from raw blocks."""
    from ._mrpt import _semicanonical_prep

    prepared = _semicanonical_prep(
        h1e_mo,
        h2e_mo,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_active_elec,
        ms2=ms2,
    )
    candidates = tuple(
        contracted_candidates(
            n_core=n_core,
            n_active=n_active,
            n_orb=prepared["norb"],
        )
    )
    raw_system = build_inactive_pair_active_virtual_active_ci_raw_system(
        candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=candidate_thresh,
        coupling_tol=coupling_tol,
    )
    block_system = build_signature_block_linear_system_from_raw(
        raw_system,
        thresh=thresh,
    )
    return solve_signature_block_linear_system_iterative(
        block_system,
        tol=tol,
        maxiter=maxiter,
    )


def inactive_active_to_virtual_pair_active_ci_caspt2_corr_signature_iterative(
    h1e_mo,
    h2e_mo,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-8,
    candidate_thresh: float = 1e-14,
    tol: float = 1e-12,
    maxiter: int | None = None,
    coupling_tol: float = 0.0,
) -> IterativeICResult:
    """Private ``(1,2)`` virtual-pair CASPT2 solve from raw blocks."""
    from ._mrpt import _semicanonical_prep

    prepared = _semicanonical_prep(
        h1e_mo,
        h2e_mo,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_active_elec,
        ms2=ms2,
    )
    candidates = tuple(
        contracted_candidates(
            n_core=n_core,
            n_active=n_active,
            n_orb=prepared["norb"],
        )
    )
    raw_system = build_inactive_active_to_virtual_pair_active_ci_raw_system(
        candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=candidate_thresh,
        coupling_tol=coupling_tol,
    )
    block_system = build_signature_block_linear_system_from_raw(
        raw_system,
        thresh=thresh,
    )
    return solve_signature_block_linear_system_iterative(
        block_system,
        tol=tol,
        maxiter=maxiter,
    )


def inactive_pair_to_virtual_pair_active_ci_caspt2_corr_signature_iterative(
    h1e_mo,
    h2e_mo,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-8,
    candidate_thresh: float = 1e-14,
    tol: float = 1e-12,
    maxiter: int | None = None,
    coupling_tol: float = 0.0,
) -> IterativeICResult:
    """Private ``(2,2)`` virtual-pair CASPT2 solve from raw blocks."""
    from ._mrpt import _semicanonical_prep

    prepared = _semicanonical_prep(
        h1e_mo,
        h2e_mo,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_active_elec,
        ms2=ms2,
    )
    candidates = tuple(
        contracted_candidates(
            n_core=n_core,
            n_active=n_active,
            n_orb=prepared["norb"],
        )
    )
    raw_system = build_inactive_pair_to_virtual_pair_active_ci_raw_system(
        candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=candidate_thresh,
        coupling_tol=coupling_tol,
    )
    block_system = build_signature_block_linear_system_from_raw(
        raw_system,
        thresh=thresh,
    )
    return solve_signature_block_linear_system_iterative(
        block_system,
        tol=tol,
        maxiter=maxiter,
    )


def inactive_to_virtual_active_ci_caspt2_corr_signature_iterative(
    h1e_mo,
    h2e_mo,
    n_core: int,
    n_active: int,
    n_active_elec: int,
    ms2: int,
    thresh: float = 1e-8,
    candidate_thresh: float = 1e-14,
    tol: float = 1e-12,
    maxiter: int | None = None,
    coupling_tol: float = 0.0,
) -> IterativeICResult:
    """Private ``(1,1)`` inactive-to-virtual CASPT2 solve from raw blocks."""
    from ._mrpt import _semicanonical_prep

    prepared = _semicanonical_prep(
        h1e_mo,
        h2e_mo,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_active_elec,
        ms2=ms2,
    )
    candidates = tuple(
        contracted_candidates(
            n_core=n_core,
            n_active=n_active,
            n_orb=prepared["norb"],
        )
    )
    raw_system = build_inactive_to_virtual_active_ci_raw_system(
        candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_active_elec,
        ms2,
        thresh=candidate_thresh,
        coupling_tol=coupling_tol,
    )
    block_system = build_signature_block_linear_system_from_raw(
        raw_system,
        thresh=thresh,
    )
    return solve_signature_block_linear_system_iterative(
        block_system,
        tol=tol,
        maxiter=maxiter,
    )
