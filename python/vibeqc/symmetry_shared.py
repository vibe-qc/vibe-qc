"""Experimental shared symmetry actions and evidence, independent of a solver.

A mathematical group/action is not an operator certificate. All numerical
checks here are diagnostics; no result authorizes production pair skipping.
Native kernels admit logical payload/work before scanning or allocating.
Python orchestration uses explicit budgets too, not allocator/RSS guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from hashlib import sha256
import json
from math import gcd, lcm
from typing import Callable, Hashable, Protocol

import numpy as np

__all__ = [
    "Budget", "SpaceIdentity", "FiniteGroup", "SpaceAction", "BlockSpaceAction",
    "Orbit", "orbit_of", "scatter_orbit", "gather_orbit_adjoint",
    "EvidenceKind", "OperatorContract", "QualificationEvidence",
    "audit_equivariance", "audit_metric_panels", "SubspaceTransportEvidence",
    "audit_subspace_panels", "BlochCharacter", "GroupTransportEvidence",
    "audit_group_transport",
    "PeriodicStateSubspaceEvidence", "audit_periodic_state_subspace",
    "PeriodicStateGroupEvidence", "audit_periodic_state_group",
    "SelectedGroupTransportEvidence", "audit_selected_group_transport",
    "GroupOperatorEvidence", "audit_group_operators",
    "SelectedOperatorEvidence", "audit_selected_operators",
]


def _core():
    from . import _vibeqc_core
    return _vibeqc_core


def _integer(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < int(positive) or value > 2**63-1:
        raise ValueError(f"{name} outside admitted integer range")
    return value


def _tolerance(value: float, name: str = "tolerance") -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicitly named")


def _array(a: np.ndarray, dtype: str, ndim: int) -> np.ndarray:
    if not isinstance(a, np.ndarray) or a.dtype != np.dtype(dtype) or a.ndim != ndim:
        raise TypeError(f"expected a {ndim}-dimensional {dtype} array")
    if not a.flags.c_contiguous or not a.flags.aligned:
        raise ValueError("symmetry arrays must be aligned and C-contiguous")
    return a


def _freeze(a: np.ndarray) -> np.ndarray:
    # bytes backing prevents callers from restoring WRITEABLE on the snapshot.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, slots=True)
class Budget:
    maximum_bytes: int
    maximum_work: int

    def __post_init__(self) -> None:
        _integer(self.maximum_bytes, "maximum_bytes", positive=True)
        _integer(self.maximum_work, "maximum_work", positive=True)

    def admit(self, byte_count: int, work: int) -> None:
        if byte_count > self.maximum_bytes:
            raise MemoryError("shared symmetry byte budget exceeded")
        if work > self.maximum_work:
            raise ValueError("shared symmetry work budget exceeded")

    def native(self):
        result = _core()._SymmetryBudget()
        result.maximum_bytes = self.maximum_bytes
        result.maximum_work = self.maximum_work
        return result


@dataclass(frozen=True, slots=True)
class SpaceIdentity:
    """Caller-declared identities, not hashes independently authenticated here."""
    geometry: str
    basis: str
    space: str
    gauge: str

    def __post_init__(self) -> None:
        for name in ("geometry", "basis", "space", "gauge"):
            _text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class FiniteGroup:
    """An admitted multiplication table, optionally a lattice extension.

    The integer cocycle is preserved in full. A nonzero cocycle means the
    table alone is a quotient action, not the complete space-group action.
    """
    _native: object
    identity_label: str

    @classmethod
    def from_table(cls, products: np.ndarray, antiunitary: np.ndarray, *,
                   identity: int, identity_label: str, budget: Budget,
                   rotations: np.ndarray | None = None,
                   cocycle: np.ndarray | None = None) -> FiniteGroup:
        _text(identity_label, "group identity_label")
        _array(products, "int64", 2)
        _array(antiunitary, "uint8", 1)
        identity = _integer(identity, "identity")
        if rotations is not None:
            _array(rotations, "int64", 3)
        if cocycle is not None:
            _array(cocycle, "int64", 3)
        native = _core()._make_symmetry_group(
            products, antiunitary, identity, rotations, cocycle, budget.native(),
        )
        return cls(native, identity_label)

    @classmethod
    def from_integer_rotations(cls, rotations: np.ndarray, *,
                               identity_label: str, budget: Budget) -> FiniteGroup:
        """Admit the rotation quotient only, with exact integer products.

        No translation, atom, basis or electronic-state symmetry is inferred.
        Seitz actions with nonzero image cocycles must use from_table instead.
        """
        _array(rotations, "int64", 3)
        n = rotations.shape[0]
        if n < 1 or rotations.shape != (n, 3, 3):
            raise ValueError("integer rotation group shape mismatch")
        budget.admit(4096+512*n*n+1024*n, 512*n**3+256*n*n)
        keys = [tuple(int(x) for x in r.ravel()) for r in rotations]
        lookup = {key: i for i, key in enumerate(keys)}
        if len(lookup) != n:
            raise ValueError("duplicate rotation parts do not form a group")
        identity = lookup.get((1, 0, 0, 0, 1, 0, 0, 0, 1))
        if identity is None:
            raise ValueError("rotation group is missing identity")
        table = np.empty((n, n), dtype=np.int64)
        for g, left in enumerate(keys):
            for h, right in enumerate(keys):
                # Python integers prevent overflow before the checked native
                # lattice representation audit. No rounded floating inverse.
                key = tuple(sum(left[3*a+c]*right[3*c+b] for c in range(3))
                            for a in range(3) for b in range(3))
                if key not in lookup:
                    raise ValueError("rotation operations do not close as a group")
                table[g, h] = lookup[key]
        return cls.from_table(table, np.zeros(n, dtype=np.uint8), identity=identity,
            identity_label=identity_label, budget=budget, rotations=rotations,
            cocycle=np.zeros((n, n, 3), dtype=np.int64))

    def conjugacy_classes(self, *, budget: Budget) -> tuple[tuple[int, ...], ...]:
        """Conjugacy classes in original operation order, using admitted algebra."""
        n = self.order
        budget.admit(4096+256*n+self._native.memory.bytes, 8*n*n)
        seen: set[int] = set()
        classes = []
        for i in range(n):
            if i in seen:
                continue
            members = tuple(sorted({self.product(self.product(self.inverse(g), i), g)
                                    for g in range(n)}))
            seen.update(members)
            classes.append(members)
        return tuple(classes)

    def __post_init__(self) -> None:
        if not isinstance(self._native, _core()._SymmetryGroup):
            raise TypeError("FiniteGroup requires an admitted native group")
        _text(self.identity_label, "group identity_label")

    @property
    def citation_numerics(self) -> tuple[str, ...]:
        return ("shared_symmetry",)

    @property
    def order(self) -> int:
        return self._native.order

    @property
    def identity(self) -> int:
        return self._native.identity

    def product(self, g: int, h: int) -> int:
        return self._native.product(_integer(g, "g"), _integer(h, "h"))

    def inverse(self, g: int) -> int:
        return self._native.inverse(_integer(g, "g"))

    def antiunitary(self, g: int) -> bool:
        return self._native.antiunitary(_integer(g, "g"))

    def cocycle(self, g: int, h: int) -> tuple[int, int, int]:
        return tuple(self._native.cocycle(_integer(g, "g"), _integer(h, "h")))


class SpaceAction(Protocol):
    """Semilinear coefficient-panel map between explicitly named spaces."""
    source: SpaceIdentity
    target: SpaceIdentity
    dimension: int
    antiunitary: bool

    def apply(self, coefficients: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True, init=False, eq=False)
class BlockSpaceAction:
    """Compact U K^a transport; blocks can mix complete local subspaces.

    No Euclidean unitarity or invertibility is assumed. Use metric/group
    audits before treating this transport map as a representation.
    """
    source: SpaceIdentity
    target: SpaceIdentity
    dimension: int
    antiunitary: bool
    offsets: np.ndarray
    destinations: np.ndarray
    matrices: np.ndarray
    budget: Budget

    def __init__(self, source: SpaceIdentity, target: SpaceIdentity,
                 offsets: np.ndarray, destinations: np.ndarray, matrices: np.ndarray,
                 *, antiunitary: bool, budget: Budget):
        if not isinstance(source, SpaceIdentity) or not isinstance(target, SpaceIdentity):
            raise TypeError("space actions require named SpaceIdentity objects")
        if type(antiunitary) is not bool:
            raise TypeError("antiunitary must be bool")
        _array(offsets, "int64", 1)
        _array(destinations, "int64", 1)
        _array(matrices, "complex128", 1)
        if offsets.size != destinations.size+1 or offsets.size < 2:
            raise ValueError("block descriptor shape mismatch")
        dimension = _integer(offsets[-1], "dimension", positive=True)
        # Constructor snapshots coexist with caller inputs and zero-column
        # native validation. Reserve snapshots before allocating any of them.
        owned = offsets.nbytes+destinations.nbytes+matrices.nbytes
        budget.admit(4096+2*owned, 8*matrices.size+32*destinations.size)
        native_budget = Budget(budget.maximum_bytes-owned, budget.maximum_work)
        _core()._apply_symmetry_blocks(offsets, destinations, matrices,
            np.empty((dimension, 0), dtype=np.complex128), antiunitary, False,
            native_budget.native())
        for key, value in dict(source=source, target=target, dimension=dimension,
                               antiunitary=antiunitary, offsets=_freeze(offsets),
                               destinations=_freeze(destinations), matrices=_freeze(matrices),
                               budget=budget).items():
            object.__setattr__(self, key, value)

    @property
    def citation_numerics(self) -> tuple[str, ...]:
        return ("shared_symmetry",)

    def _apply(self, coefficients: np.ndarray, *, adjoint: bool, antiunitary: bool) -> np.ndarray:
        _array(coefficients, "complex128", 2)
        if coefficients.shape[0] != self.dimension:
            raise ValueError("coefficient panel dimension disagrees with named space")
        return _core()._apply_symmetry_blocks(
            self.offsets, self.destinations, self.matrices, coefficients,
            antiunitary, adjoint, self.budget.native(),
        )

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        return self._apply(coefficients, adjoint=False, antiunitary=self.antiunitary)

    def adjoint(self, coefficients: np.ndarray) -> np.ndarray:
        """Adjoint under Re tr(X^dagger Y), also valid for scalar antiunitary maps."""
        return self._apply(coefficients, adjoint=True, antiunitary=self.antiunitary)

    def _matrix_budget(self, matrix: np.ndarray) -> Budget:
        _array(matrix, "complex128", 2)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("matrix dimension disagrees with named space")
        # Input, first result, transposed/conjugated panels and final result.
        reserve = 6*matrix.nbytes
        self.budget.admit(reserve+4096+self.matrices.nbytes, 1)
        return Budget(self.budget.maximum_bytes-reserve, self.budget.maximum_work//2)

    def _linear_with_budget(self, panel: np.ndarray, adjoint: bool, budget: Budget) -> np.ndarray:
        return _core()._apply_symmetry_blocks(self.offsets, self.destinations, self.matrices,
            panel, False, adjoint, budget.native())

    def push_density(self, density: np.ndarray) -> np.ndarray:
        """Contravariant density: D_target = U conjugate_if_a(D_source) U^dagger."""
        budget = self._matrix_budget(density)
        d = density.conj() if self.antiunitary else density
        left = self._linear_with_budget(d, False, budget)
        return self._linear_with_budget(
            np.ascontiguousarray(left.conj().T), False, budget,
        ).conj().T.copy()

    def pull_operator(self, operator: np.ndarray) -> np.ndarray:
        """Covariant pullback: conjugate_if_a(U^dagger F_target U).

        This preserves the real density/operator pairing for nonorthogonal
        representations; it does not replace U^{-dagger} by U.
        """
        budget = self._matrix_budget(operator)
        left = self._linear_with_budget(operator, True, budget)
        result = self._linear_with_budget(
            np.ascontiguousarray(left.conj().T), True, budget,
        ).conj().T.copy()
        return result.conj() if self.antiunitary else result


@dataclass(frozen=True, slots=True, init=False)
class Orbit:
    """Factory-only index orbit of an admitted group and checked index action.

    Callback keys must retain stable equality/hash semantics. They remain
    caller-owned identifiers, not authenticated physical-space snapshots.
    """
    group: FiniteGroup
    representative: Hashable
    members: tuple[Hashable, ...]
    # All operations reaching each member, including representative stabilizers.
    transporters: tuple[tuple[int, ...], ...]
    stabilizer: tuple[int, ...]

    def __init__(self, *args, **kwargs):
        raise TypeError("construct an admitted Orbit with orbit_of")

    @property
    def weight(self) -> int:
        return len(self.members)


def orbit_of(group: FiniteGroup, seed: Hashable,
             action: Callable[[int, Hashable], Hashable], *, budget: Budget) -> Orbit:
    """Build only one orbit, with exact orbit-stabilizer and local action audits.

    Callback payload sizes/work are caller-owned. This budget covers the
    bounded group-index/control inventory and counts callback invocations.
    It does not enumerate every translated pair or authorize skipping.
    """
    if not isinstance(group, FiniteGroup):
        raise TypeError("index orbits require an admitted FiniteGroup")
    n = group.order
    budget.admit(4096+256*n*n+group._native.memory.bytes, n*n+n)
    images = tuple(action(g, seed) for g in range(n))
    if images[group.identity] != seed:
        raise ValueError("orbit action identity failed")
    for g in range(n):
        for h in range(n):
            if action(g, images[h]) != images[group.product(g, h)]:
                raise ValueError("orbit action does not represent the supplied group quotient")
    members = tuple(dict.fromkeys(images))
    transporters = tuple(tuple(g for g, image in enumerate(images) if image == member)
                         for member in members)
    stabilizer = transporters[members.index(seed)]
    if any(len(t) != len(stabilizer) for t in transporters) or len(members)*len(stabilizer) != n:
        raise ValueError("orbit-stabilizer law failed")
    result = object.__new__(Orbit)
    for name, value in dict(group=group, representative=seed, members=members,
                            transporters=transporters, stabilizer=stabilizer).items():
        object.__setattr__(result, name, value)
    return result


class EvidenceKind(str, Enum):
    ANALYTIC = "analytic_argument"
    NUMERICAL = "finite_numerical_test"
    UNESTABLISHED = "unestablished"


@dataclass(frozen=True, slots=True)
class OperatorContract:
    """Declared provenance of one numerical operator, independent of geometry."""
    operator: str
    source_revision: str
    support: str
    screening: str
    convention: str

    def __post_init__(self) -> None:
        for name in ("operator", "source_revision", "support", "screening", "convention"):
            _text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class QualificationEvidence:
    contract: OperatorContract
    source_space: SpaceIdentity
    target_space: SpaceIdentity
    relation: str
    kind: EvidenceKind
    evidence: str
    residual: float | None = None
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.contract, OperatorContract) or not isinstance(
            self.kind, EvidenceKind
        ):
            raise TypeError("qualification requires typed contract and evidence kind")
        if not isinstance(self.source_space, SpaceIdentity) or not isinstance(
            self.target_space, SpaceIdentity
        ):
            raise TypeError("qualification requires named source and target spaces")
        _text(self.relation, "relation")
        _text(self.evidence, "evidence")
        if self.kind is EvidenceKind.NUMERICAL:
            if self.residual is None or self.tolerance is None:
                raise ValueError("numerical evidence requires residual and tolerance")
            _tolerance(self.residual, "residual")
            _tolerance(self.tolerance)
        elif self.residual is not None or self.tolerance is not None:
            raise ValueError("only numerical evidence carries a residual")

    @property
    def passed_probe(self) -> bool:
        return self.kind is EvidenceKind.NUMERICAL and self.residual <= self.tolerance

    @property
    def production_reduction_authorized(self) -> bool:
        # Deliberately no generic certificate-to-switch conversion. Analytic
        # claims and successful finite probes still need method-owned gates.
        return False


def audit_equivariance(
    action: BlockSpaceAction,
    density: np.ndarray,
    builder: Callable[[np.ndarray], np.ndarray],
    *,
    contract: OperatorContract,
    tolerance: float,
    probe_identity: str,
    target_builder: Callable[[np.ndarray], np.ndarray] | None = None,
) -> QualificationEvidence:
    """One probe: pull(F_target[push(D)]) versus F_source[D], without repair.

    For an invertible action this is the Fock equivariance relation. Neither
    this test nor several passing densities establish a universal bound.
    Builder resource admission and screening policy remain backend-owned.
    Different named spaces require an explicit target-gauge implementation
    of the declared operator. Reusing a source formula across gauges cannot
    be inferred from compatible array dimensions.
    """
    tolerance = _tolerance(tolerance)
    _array(density, "complex128", 2)
    _text(probe_identity, "probe_identity")
    if target_builder is None:
        if action.source != action.target:
            raise ValueError("different named spaces require an explicit target builder")
        target_builder = builder
    if not callable(builder) or not callable(target_builder):
        raise TypeError("operator builders must be callable")
    # Include an owned reference snapshot: backends may reuse output buffers.
    action.budget.admit(13*density.nbytes+4096+action.matrices.nbytes, 1)
    moved = action.push_density(density)
    original = builder(density)
    _array(original, "complex128", 2)
    if original.shape != density.shape or not np.all(np.isfinite(original)):
        raise ValueError("builder returned invalid operator")
    original = original.copy()
    transformed = target_builder(moved)
    pulled = action.pull_operator(transformed)
    residual = float(np.max(np.abs(pulled-original), initial=0.))
    return QualificationEvidence(contract, action.source, action.target,
        "Fock equivariance on the named density probe", EvidenceKind.NUMERICAL,
        probe_identity, residual, float(tolerance))


def _orbit_transport_budget(orbit: Orbit, transports: tuple[BlockSpaceAction, ...],
                            panel: np.ndarray, budget: Budget) -> None:
    if not isinstance(orbit, Orbit):
        raise TypeError("orbit execution requires an admitted Orbit from orbit_of")
    _array(panel, "complex128", 2)
    n, rows = len(transports), len(orbit.transporters)
    if n != orbit.group.order:
        raise ValueError("orbit transport inventory disagrees with admitted group")
    # Refuse oversized caller-supplied descriptors before building flattened
    # or sorted operation inventories, including malformed duplicate rows.
    budget.admit(4096+256*(n+rows), n+rows)
    calls = sum(len(row) for row in orbit.transporters)
    if calls != n:
        raise ValueError("orbit must use each group transporter exactly once")
    descriptor_work = calls*(n.bit_length()+8)+rows
    budget.admit(4096+256*(n+rows+calls), descriptor_work)
    if not transports or len(orbit.members) != rows or any(not row for row in orbit.transporters):
        raise ValueError("orbit is missing a transporter")
    operations = [g for row in orbit.transporters for g in row]
    if sorted(operations) != list(range(len(transports))):
        raise ValueError("orbit must use each group transporter exactly once")
    if any(not isinstance(t, BlockSpaceAction) for t in transports):
        raise TypeError("orbit execution requires typed block space actions")
    if any(t.antiunitary != orbit.group.antiunitary(g) for g, t in enumerate(transports)):
        raise ValueError("orbit transport antiunitary grading disagrees with admitted group")
    if any(t.source != transports[0].source for t in transports):
        raise ValueError("orbit transport source space identity mismatch")
    for row in orbit.transporters:
        if any(transports[g].target != transports[row[0]].target for g in row):
            raise ValueError("orbit transport target space identity mismatch")
    if any(t.dimension != panel.shape[0] for t in transports):
        raise ValueError("orbit transport space dimension mismatch")
    retained = orbit.group._native.memory.bytes + sum(
        t.offsets.nbytes+t.destinations.nbytes+t.matrices.nbytes+4096 for t in transports
    )
    work = sum(32*t.matrices.size*panel.shape[1]+8*t.matrices.size+32*t.destinations.size
               for t in transports)
    budget.admit(retained+(len(orbit.members)+4)*panel.nbytes,
                 work+calls*panel.size+descriptor_work)


def scatter_orbit(orbit: Orbit, representative: np.ndarray,
                  transports: tuple[BlockSpaceAction, ...], *,
                  budget: Budget, tolerance: float) -> tuple[np.ndarray, ...]:
    """Transport one orbit; refuse inconsistent stabilizers instead of averaging.

    Call only for a method-qualified tensor action. Passing this consistency
    check is not operator admissibility. Orbit multiplicity is never used in
    place of a matrix-valued transformation (Casassa 2006, Eq.3).
    """
    tolerance = _tolerance(tolerance)
    _orbit_transport_budget(orbit, transports, representative, budget)
    output = []
    for operations in orbit.transporters:
        value = transports[operations[0]].apply(representative)
        for g in operations[1:]:
            other = transports[g].apply(representative)
            if np.max(np.abs(other-value), initial=0.) > tolerance:
                raise ValueError("representative violates orbit stabilizer transport")
        output.append(value)
    return tuple(output)


def gather_orbit_adjoint(orbit: Orbit, values: tuple[np.ndarray, ...],
                         transports: tuple[BlockSpaceAction, ...], *,
                         budget: Budget) -> np.ndarray:
    """Real-inner-product adjoint of the selected scatter maps, with unit weights.

    All members contribute exactly once. This is an adjoint accumulation,
    not inverse reconstruction or averaging by the orbit size.
    """
    if not isinstance(orbit, Orbit):
        raise TypeError("orbit execution requires an admitted Orbit from orbit_of")
    if len(values) != len(orbit.members) or not values:
        raise ValueError("gather requires exactly one panel per orbit member")
    _orbit_transport_budget(orbit, transports, values[0], budget)
    if any(v.shape != values[0].shape for v in values):
        raise ValueError("gather panel shape mismatch")
    result = np.zeros_like(values[0])
    for v, operations in zip(values, orbit.transporters):
        result += transports[operations[0]].adjoint(v)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite orbit adjoint accumulation")
    return result


def audit_metric_panels(action: SpaceAction, coefficients: np.ndarray,
                        source_metric: np.ndarray, target_metric: np.ndarray, *,
                        contract: OperatorContract, tolerance: float,
                        probe_identity: str, budget: Budget) -> QualificationEvidence:
    """Shared metric witness for any coefficient-panel backend adapter.

    Compares Gram matrices of the supplied panels only. It neither builds a
    dense AO action nor supplies orbital-subspace leakage or operator proof.
    """
    _array(coefficients, "complex128", 2)
    for metric in (source_metric, target_metric):
        _array(metric, "complex128", 2)
        if metric.shape != (action.dimension, action.dimension):
            raise ValueError("metric dimension disagrees with named space")
    if coefficients.shape[0] != action.dimension:
        raise ValueError("metric panel dimension mismatch")
    n, columns = coefficients.shape
    budget.admit(
        4096+source_metric.nbytes+target_metric.nbytes+5*coefficients.nbytes+64*columns**2,
        32*n*n*columns+32*n*columns**2,
    )
    tolerance = _tolerance(tolerance)
    if not np.all(np.isfinite(source_metric)) or not np.all(np.isfinite(target_metric)):
        raise ValueError("nonfinite metric payload")
    moved = action.apply(coefficients)
    _array(moved, "complex128", 2)
    if moved.shape != coefficients.shape:
        raise ValueError("transported metric panel shape mismatch")
    source_gram = coefficients.conj().T @ source_metric @ coefficients
    if action.antiunitary:
        source_gram = source_gram.conj()
    target_gram = moved.conj().T @ target_metric @ moved
    residual = float(np.max(np.abs(target_gram-source_gram), initial=0.))
    return QualificationEvidence(contract, action.source, action.target,
        "metric isometry on the named coefficient panel", EvidenceKind.NUMERICAL,
        probe_identity, residual, float(tolerance))


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SubspaceTransportEvidence:
    """Finite retained-space witness; mixing is not an execution certificate.

    For source coordinates X, the diagnosed map is
    C_source X -> C_target mixing conjugate_if_a(X). Column gauges may mix
    the complete retained space. Tensor index and approximation gates remain
    method-owned, even when both numerical relations pass.
    """
    metric: QualificationEvidence
    containment: QualificationEvidence
    mixing: np.ndarray
    antiunitary: bool

    def __init__(self, *args, **kwargs):
        raise TypeError("construct subspace evidence with audit_subspace_panels")

    @property
    def passed_probe(self) -> bool:
        return self.metric.passed_probe and self.containment.passed_probe

    @property
    def production_reduction_authorized(self) -> bool:
        return False


def audit_subspace_panels(
    action: SpaceAction, source_coefficients: np.ndarray,
    target_coefficients: np.ndarray, source_metric: np.ndarray,
    target_metric: np.ndarray, *, source_subspace: SpaceIdentity,
    target_subspace: SpaceIdentity, contract: OperatorContract,
    metric_tolerance: float, leakage_tolerance: float,
    probe_identity: str, budget: Budget,
) -> SubspaceTransportEvidence:
    """Diagnose transport between named, metric-orthonormal retained panels.

    Equal, nonzero ranks are required. Rank selection/orthonormalization of
    PAOs, PNOs, frozen or active columns belongs to the caller. The supplied
    metrics must be Hermitian and the panels orthonormal within
    metric_tolerance (< 1 in Frobenius norm). No global positivity of an AO
    metric, group law, operator equivariance or retained-space selection
    policy is inferred from these finite relations.

    With V = action(C_source), report Gram isometry and the relative
    coefficient Frobenius residual ||V-C_target M||_F / ||V||_F separately,
    where M solves (C_target^dagger S_target C_target) M =
    C_target^dagger S_target V. The small Gram solve retains the admitted
    finite orthonormality error instead of turning it into leakage.
    This containment residual cannot
    hide leakage in a null or indefinite direction of a supplied metric.
    It is invariant under unitary retained-column gauge changes, but its
    magnitude depends on the declared ambient AO gauge.

    Admission precedes payload scans/copies. The budget covers borrowed
    matrices/panels and conservative live NumPy intermediates, with work
    O(n^2 r + n r^2). No n-by-n action or projector is constructed. Action
    storage/execution and BLAS workspace remain backend-owned. Snapshots
    protect references when an action reuses a caller-visible workspace.
    """
    if not isinstance(contract, OperatorContract):
        raise TypeError("subspace audit requires an OperatorContract")
    _text(probe_identity, "probe_identity")
    metric_tolerance = _tolerance(metric_tolerance, "metric_tolerance")
    leakage_tolerance = _tolerance(leakage_tolerance, "leakage_tolerance")
    if metric_tolerance >= 1:
        raise ValueError("orthonormality metric_tolerance must be less than one")
    if type(action.antiunitary) is not bool:
        raise TypeError("subspace antiunitary flag must be bool")
    n = _integer(action.dimension, "dimension", positive=True)
    for retained, ambient in ((source_subspace, action.source),
                              (target_subspace, action.target)):
        if not isinstance(retained, SpaceIdentity) or not isinstance(ambient, SpaceIdentity):
            raise TypeError("subspace audit requires named SpaceIdentity objects")
        if (retained.geometry, retained.basis) != (ambient.geometry, ambient.basis):
            raise ValueError("retained subspace geometry/basis disagrees with ambient action")
    for c in (source_coefficients, target_coefficients):
        _array(c, "complex128", 2)
    rank = source_coefficients.shape[1]
    if (not 0 < rank <= n or source_coefficients.shape != (n, rank)
            or target_coefficients.shape != (n, rank)):
        raise ValueError("retained panels require equal, nonzero ranks within the ambient space")
    for metric in (source_metric, target_metric):
        _array(metric, "complex128", 2)
        if metric.shape != (n, n):
            raise ValueError("retained-space metric dimension mismatch")
    arrays = (source_coefficients, target_coefficients, source_metric, target_metric)
    budget.admit(4096+24*(source_coefficients.nbytes+target_coefficients.nbytes)
                 +8*(source_metric.nbytes+target_metric.nbytes)+512*rank*rank,
                 256*n*n*rank+512*n*rank*rank+64*n*n+128*n*rank)
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("nonfinite retained-space payload")
    source, target, sm, tm = (a.copy() for a in arrays)
    for metric in (sm, tm):
        if np.max(np.abs(metric-metric.conj().T), initial=0.) > metric_tolerance:
            raise ValueError("retained-space metric must be Hermitian")
    source_gram = source.conj().T @ sm @ source
    target_gram = target.conj().T @ tm @ target
    identity = np.eye(rank)
    for gram in (source_gram, target_gram):
        error = float(np.linalg.norm(gram-identity))
        if not np.isfinite(error) or error > metric_tolerance:
            raise ValueError("retained panels must be metric-orthonormal")
    source.setflags(write=False)
    moved = action.apply(source)
    _array(moved, "complex128", 2)
    if moved.shape != source.shape or not np.all(np.isfinite(moved)):
        raise ValueError("invalid transported retained panel")
    moved_gram = moved.conj().T @ tm @ moved
    expected_gram = source_gram.conj() if action.antiunitary else source_gram
    metric_residual = float(np.linalg.norm(moved_gram-expected_gram))
    mixing = np.ascontiguousarray(np.linalg.solve(target_gram, target.conj().T @ tm @ moved))
    difference = moved-target @ mixing
    scale = max(float(np.max(np.abs(moved))), float(np.max(np.abs(difference))))
    if not np.isfinite(scale) or not np.all(np.isfinite(mixing)):
        raise ValueError("nonfinite retained-space residual")
    # A common scale keeps a tiny nonzero leaking map from underflowing to
    # an apparently exact containment witness, and avoids squared overflow.
    moved_norm = float(np.linalg.norm(moved/scale)) if scale else 0.
    difference_norm = float(np.linalg.norm(difference/scale)) if scale else 0.
    if not np.isfinite(moved_norm) or not np.isfinite(difference_norm):
        raise ValueError("nonfinite retained-space residual")
    if not moved_norm and difference_norm:
        raise ValueError("unrepresentable relative retained-space residual")
    leakage = difference_norm/moved_norm if moved_norm else 0.
    metric_evidence = QualificationEvidence(contract, source_subspace, target_subspace,
            "metric isometry on the named retained panel", EvidenceKind.NUMERICAL,
            probe_identity, metric_residual, metric_tolerance)
    containment = QualificationEvidence(contract, source_subspace, target_subspace,
            "relative coefficient leakage outside the named retained subspace",
            EvidenceKind.NUMERICAL, probe_identity, leakage, leakage_tolerance)
    result = object.__new__(SubspaceTransportEvidence)
    for name, value in dict(metric=metric_evidence, containment=containment,
                            mixing=_freeze(mixing), antiunitary=action.antiunitary).items():
        object.__setattr__(result, name, value)
    return result


def _int64_triple(value: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("lattice coordinates require an integer triple")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
           for x in value):
        raise TypeError("lattice coordinates require an integer triple")
    result = tuple(int(x) for x in value)
    if any(x < -(2**63) or x > 2**63-1 for x in result):
        raise ValueError("lattice coordinate outside int64 range")
    return result


def _phase(exponent: Fraction) -> complex:
    # Preserve exact fourth roots, including identity and zone-boundary signs.
    if exponent.denominator in (1, 2, 4):
        return (1.+0j, 1j, -1.+0j, -1j)[int(4*exponent)]
    return complex(np.exp(2j*np.pi*float(exponent)))


@dataclass(frozen=True, slots=True)
class BlochCharacter:
    """Declared rational k: T(ell) -> exp(-2 pi i numerator.ell/denominator).

    Reciprocal-integer shifts are equivalent. Integer arithmetic precedes
    phase evaluation, so large lattice images are never rounded to floats.
    This declaration does not authenticate a native mesh or orbital gauge.
    """
    numerator: tuple[int, int, int]
    denominator: int

    def __post_init__(self) -> None:
        denominator = _integer(self.denominator, "denominator", positive=True)
        numerator = tuple(x % denominator for x in _int64_triple(self.numerator))
        common = gcd(denominator, *numerator)
        object.__setattr__(self, "numerator", tuple(x//common for x in numerator))
        object.__setattr__(self, "denominator", denominator//common)

    def _exponent(self, image: tuple[int, int, int]) -> Fraction:
        return Fraction(-sum(a*b for a, b in zip(self.numerator, image)), self.denominator) % 1

    def phase(self, image: tuple[int, int, int]) -> complex:
        return _phase(self._exponent(_int64_triple(image)))


@dataclass(frozen=True, slots=True, init=False, eq=False)
class GroupTransportEvidence:
    """Finite group-law witness across named retained spaces, never admission."""
    group: FiniteGroup
    spaces: tuple[SpaceIdentity, ...]
    destinations: np.ndarray
    transports: tuple[tuple[SubspaceTransportEvidence, ...], ...]
    characters: tuple[BlochCharacter, ...] | None
    contract: OperatorContract
    probe_identity: str
    identity_residual: float
    composition_residual: float
    worst_identity_space: int
    worst_composition: tuple[int, int, int]
    local_probes_passed: bool
    tolerance: float

    def __init__(self, *args, **kwargs):
        raise TypeError("construct group transport evidence with audit_group_transport")

    @property
    def passed_probe(self) -> bool:
        return (self.local_probes_passed and self.identity_residual <= self.tolerance
                and self.composition_residual <= self.tolerance)

    @property
    def production_reduction_authorized(self) -> bool:
        return False


def audit_group_transport(
    group: FiniteGroup, spaces: tuple[SpaceIdentity, ...], destinations: np.ndarray,
    transports: tuple[tuple[SubspaceTransportEvidence, ...], ...], *,
    contract: OperatorContract, tolerance: float, probe_identity: str, budget: Budget,
    characters: tuple[BlochCharacter, ...] | None = None,
) -> GroupTransportEvidence:
    """Check a supplied group action and its retained mixing across all spaces.

    destinations[g,s] and transports[g][s] use operation g at source s.
    Products act h first: M[g,h.s] conjugate_if_a(g)(M[h,s]) must equal
    omega[g,h,s] M[gh,s]. For gh = T(ell[g,h]) (gh), omega is the declared
    Bloch character of the FINAL target space evaluated on the full image.
    Nonzero lattice cocycles require explicit characters, including at Gamma.
    The resulting normalized, antiunitary-twisted scalar cocycle is checked
    exactly as rational exponents before evaluating any floating phases.

    Frobenius identity/composition residuals and every local subspace probe
    must pass. No mixing is averaged, repaired, rephased or used to infer the
    factors. Source, mesh and selected-space identities remain declarations;
    operator equivariance and approximation policies are not established.

    Budget admission covers the retained native group, mixing payloads,
    destination snapshots, rational factors and conservative control/work
    reservations. It does not cover caller-owned identity strings, allocator
    overhead or BLAS workspace. No ambient AO tensors are materialized.
    """
    if not isinstance(group, FiniteGroup) or not isinstance(contract, OperatorContract):
        raise TypeError("group transport requires an admitted group and operator contract")
    _text(probe_identity, "probe_identity")
    tolerance = _tolerance(tolerance)
    if not isinstance(spaces, tuple) or not spaces:
        raise TypeError("group transport requires a nonempty tuple of named spaces")
    n, count = group.order, len(spaces)
    _array(destinations, "int64", 2)
    if destinations.shape != (n, count):
        raise ValueError("group destination inventory shape mismatch")
    controls = 4096+group._native.memory.bytes+1024*n*n*count+2048*n*count
    base_work = 512*n**3*count
    budget.admit(controls+2*destinations.nbytes, base_work)
    if any(not isinstance(s, SpaceIdentity) for s in spaces) or len(set(spaces)) != count:
        raise ValueError("group spaces must have distinct named identities")
    if (not isinstance(transports, tuple) or len(transports) != n
            or any(not isinstance(row, tuple) or len(row) != count for row in transports)):
        raise ValueError("group transport inventory shape mismatch")
    if characters is not None and (not isinstance(characters, tuple) or len(characters) != count
            or any(not isinstance(c, BlochCharacter) for c in characters)):
        raise ValueError("group characters must correspond to every named space")
    retained = 0
    for row in transports:
        for t in row:
            if not isinstance(t, SubspaceTransportEvidence):
                raise TypeError("group transport requires audited subspace witnesses")
            retained += t.mixing.nbytes
    ranks = [transports[group.identity][s].mixing.shape[0] for s in range(count)]
    workspace = 256*max(ranks)**2
    work = base_work+128*n*n*sum(r**3+r*r for r in ranks)
    budget.admit(controls+retained+workspace+2*destinations.nbytes, work)
    dest = _freeze(destinations)
    if np.any(dest < 0) or np.any(dest >= count):
        raise ValueError("group destination outside named spaces")
    for s in range(count):
        if dest[group.identity, s] != s:
            raise ValueError("group destination identity law failed")
    for g in range(n):
        for h in range(n):
            gh = group.product(g, h)
            for s in range(count):
                if dest[g, dest[h, s]] != dest[gh, s]:
                    raise ValueError("group destination composition law failed")
    local_pass = True
    for g, row in enumerate(transports):
        for s, t in enumerate(row):
            target = int(dest[g, s])
            if t.antiunitary != group.antiunitary(g):
                raise ValueError("retained transport antiunitary grading disagrees with group")
            for evidence in (t.metric, t.containment):
                if (evidence.contract != contract or evidence.source_space != spaces[s]
                        or evidence.target_space != spaces[target]):
                    raise ValueError("retained transport contract or space identity mismatch")
            if t.mixing.shape != (ranks[target], ranks[s]) or ranks[target] != ranks[s]:
                raise ValueError("retained transport rank mismatch within a group orbit")
            if not np.all(np.isfinite(t.mixing)):
                raise ValueError("nonfinite group mixing payload")
            local_pass = local_pass and t.passed_probe
    images = [[group.cocycle(g,h) for h in range(n)] for g in range(n)]
    if characters is None and any(any(image) for row in images for image in row):
        raise ValueError("nonzero lattice cocycle requires explicit Bloch characters")
    zero = Fraction(0)
    exponents = [[[characters[int(dest[group.product(g,h),s])]._exponent(images[g][h])
                   if characters is not None else zero for s in range(count)]
                  for h in range(n)] for g in range(n)]
    # omega(g,h,j.s) omega(gh,j,s) =
    # conjugate_if_a(g)(omega(h,j,s)) omega(g,hj,s).
    for g in range(n):
        sign = -1 if group.antiunitary(g) else 1
        for h in range(n):
            for j in range(n):
                for s in range(count):
                    difference = (exponents[g][h][int(dest[j,s])]
                        + exponents[group.product(g,h)][j][s]
                        - sign*exponents[h][j][s] - exponents[g][group.product(h,j)][s])
                    if difference % 1:
                        raise ValueError("Bloch characters violate the scalar cocycle law")
    identity_error, composition_error = 0., 0.
    worst_identity, worst_composition = 0, (group.identity, group.identity, 0)
    for s in range(count):
        error = float(np.linalg.norm(transports[group.identity][s].mixing-np.eye(ranks[s])))
        if not np.isfinite(error):
            raise ValueError("nonfinite group identity residual")
        if error > identity_error:
            identity_error, worst_identity = error, s
    for g in range(n):
        for h in range(n):
            for s in range(count):
                left = transports[g][int(dest[h,s])].mixing
                right = transports[h][s].mixing
                if group.antiunitary(g):
                    right = right.conj()
                expected = _phase(exponents[g][h][s])*transports[group.product(g,h)][s].mixing
                error = float(np.linalg.norm(left @ right-expected))
                if not np.isfinite(error):
                    raise ValueError("nonfinite group composition residual")
                if error > composition_error:
                    composition_error, worst_composition = error, (g,h,s)
    _tolerance(identity_error, "identity residual")
    result = object.__new__(GroupTransportEvidence)
    for name, value in dict(group=group, spaces=spaces, destinations=dest,
            transports=transports, characters=characters, contract=contract,
            probe_identity=probe_identity, identity_residual=identity_error,
            composition_residual=composition_error, worst_identity_space=worst_identity,
            worst_composition=worst_composition, local_probes_passed=local_pass,
            tolerance=tolerance).items():
        object.__setattr__(result, name, value)
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PeriodicStateSubspaceEvidence:
    """Shared panel witness bound to a native numerical state and its masks.

    Binding the actual numbers, mesh and supplied geometry/basis does not
    authenticate their physical source or the calculation declaration.
    """
    sewing: object
    transport: SubspaceTransportEvidence
    source_index: int
    target_index: int
    subspace: str
    source_bands: tuple[int, ...]
    target_bands: tuple[int, ...]
    source_character: BlochCharacter
    target_character: BlochCharacter
    operation_rotation: np.ndarray
    operation_translation: np.ndarray

    def __init__(self, *args, **kwargs):
        raise TypeError("use audit_periodic_state_subspace")

    @property
    def state(self):
        return self.sewing.state

    @property
    def passed_probe(self) -> bool:
        return self.transport.passed_probe

    @property
    def production_reduction_authorized(self) -> bool:
        return False


def _periodic_context_identities(basis, system) -> tuple[str, str]:
    # Called only after native count/work admission. Include full radial
    # profiles and shell order, not a display name or basis dimension alone.
    def digest(payload):
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("ascii")).hexdigest()
    geometry = digest(["shared-periodic-geometry/v1", int(system.dim),
        int(system.charge), int(system.multiplicity),
        np.asarray(system.lattice).tolist(),
        [[int(a.Z), np.asarray(a.xyz).tolist()] for a in system.unit_cell]])
    radial = digest(["shared-periodic-basis/v1", int(basis.nbasis),
        [[int(s.atom_index), int(s.l), bool(s.pure), list(s.exponents),
          list(s.coefficients), list(s.origin)] for s in basis.shells()]])
    return geometry, radial


def audit_periodic_state_subspace(
    state, basis, system, operation, source_index: int, antiunitary: bool, *,
    subspace: str, sewing_options, sewing_inventory, sewing_caps,
    metric_tolerance: float, leakage_tolerance: float,
    probe_identity: str, budget: Budget,
) -> PeriodicStateSubspaceEvidence:
    """Bridge immutable native frozen/occupied/virtual panels to shared audits.

    subspace is 'frozen_core', 'correlated_occupied' or 'virtual'. Selection
    comes exclusively from the state's masks; an empty selection is refused.
    Native orbital sewing first checks all masks, transported stationarity,
    energy intertwining and the requested scope using the supplied controls.
    A second native AO panel transport feeds the shared metric/leakage audit.
    Neither result is repaired. Native diagnostic shape/cap limits still apply.

    Names bind the actual numerical state, k index, selected bands and full
    supplied geometry/basis descriptors. Exact Bloch characters are derived
    from the state's native mesh addresses, including shifts. The operation
    is snapshotted; geometry and basis are borrowed and must not be modified
    concurrently. Observable context changes during the audit are refused.
    Returned evidence retains the immutable state, not a mutable SCF result.

    The aggregate budget includes native sewing/AO inventories, retained state,
    descriptor snapshots and shared panel work. It reserves native planning,
    two execution passes and numerical/hash storage before payload extraction. Python
    allocator overhead and unreported external library workspace are excluded.
    This binds supplied numbers, not their provenance: no physical-source,
    whole-group, approximation-policy or production certificate is inferred.
    """
    core = _core()
    if not isinstance(state, core._PeriodicRestrictedMeanFieldState):
        raise TypeError("periodic subspace audit requires an immutable native state")
    if not isinstance(budget, Budget):
        raise TypeError("periodic subspace audit requires a Budget")
    if not isinstance(basis, core.BasisSet) or not isinstance(system, core.PeriodicSystem):
        raise TypeError("periodic subspace audit requires native basis and system")
    if not isinstance(operation, core.SymmetryOp) or type(antiunitary) is not bool:
        raise TypeError("periodic subspace audit requires a native operation and bool grading")
    _text(probe_identity, "probe_identity")
    metric_tolerance = _tolerance(metric_tolerance, "metric_tolerance")
    leakage_tolerance = _tolerance(leakage_tolerance, "leakage_tolerance")
    if metric_tolerance >= 1:
        raise ValueError("orthonormality metric_tolerance must be less than one")
    source = _integer(source_index, "source_index")
    if source >= state.n_kpoints:
        raise ValueError("source index outside state mesh")
    kinds = ("frozen_core", "correlated_occupied", "virtual")
    if not isinstance(subspace, str) or subspace not in kinds:
        raise ValueError("subspace must name an explicit native state mask")
    slot = kinds.index(subspace)
    native_kind = (core._PeriodicOrbitalSubspace.FROZEN_CORE,
                   core._PeriodicOrbitalSubspace.CORRELATED_OCCUPIED,
                   core._PeriodicOrbitalSubspace.VIRTUAL)[slot]
    n, r = int(state.n_basis), int(state.n_effective_orbitals)
    base = 65536+state.resident_bytes+1024*n*n+1024*n*r+2048*r*r
    panel_work = 256*n*n*r+512*n*r*r+64*n*n+128*n*r
    budget.admit(base, panel_work)
    rotation = np.array(operation.rotation, dtype=np.int32, order="C", copy=True)
    translation = np.array(operation.translation, dtype=np.float64, order="C", copy=True)
    op = core._make_periodic_ao_bloch_operation(rotation, translation)
    plan = core._plan_periodic_orbital_sewing(state,basis,system,op,source,antiunitary,
                                             sewing_options,sewing_inventory,sewing_caps)
    if not plan.subspace_ranks[slot]:
        raise ValueError("selected native state subspace is empty")
    descriptor_bytes = core._symmetry_basis_snapshot_bytes(basis,budget.native())
    budget.admit(base+2*plan.required_node_inventoried_bytes+64*descriptor_bytes,
                 panel_work+4*plan.work_units_upper_bound+256*descriptor_bytes)
    context = _periodic_context_identities(basis,system)
    sewing = core._make_periodic_orbital_sewing(state,basis,system,op,source,antiunitary,
                                               sewing_options,sewing_inventory,sewing_caps)
    target = int(sewing.memory.target_index)
    rank = int(sewing.memory.subspace_ranks[slot])
    source_bands = tuple(sewing.source_band(native_kind,i) for i in range(rank))
    target_bands = tuple(sewing.target_band(native_kind,i) for i in range(rank))
    state_id = state.state_identity_sha256
    def space(k, bands):
        return SpaceIdentity(context[0], context[1], f"{subspace}:{bands}", f"{state_id}:k={k}")
    source_space, target_space = space(source,source_bands), space(target,target_bands)
    contract = OperatorContract("recorded restricted Fock/overlap", state_id,
        f"numerical-payload:{state.numerical_payload_sha256}", "state-defined",
        "native periodic restricted state; physical source unqualified")
    mesh = core._RegularKMesh(state.mesh,state.is_shift)
    modulus = tuple(int(x) for x in mesh.doubled_modulus)
    denominator = lcm(*modulus)
    def character(k):
        address = mesh.address(k)
        return BlochCharacter(tuple(int(a)*(denominator//m) for a,m in zip(address,modulus)),denominator)
    inventory = core._PeriodicAOBlochTransportInventory()
    for name in ("numerical_replicas", "external_node_bytes", "other_live_numerical_bytes_per_replica",
                 "other_live_control_bytes_per_replica", "backend_margin_bytes_per_replica"):
        setattr(inventory,name,getattr(sewing_inventory,name))
    class StateAction:
        dimension = n
        source = SpaceIdentity(context[0],context[1],"AO",source_space.gauge)
        target = SpaceIdentity(context[0],context[1],"AO",target_space.gauge)
        def apply(self, coefficients):
            result = core._apply_periodic_ao_bloch_operation(basis,system,op,mesh,
                source,antiunitary,coefficients,sewing_options.ao_transport,inventory,sewing_caps.ao_transport)
            if result.memory.target_index != target:
                raise ValueError("native transport target disagrees with state sewing")
            return result.coefficients_copy()
    action = StateAction()
    action.antiunitary = antiunitary
    cs = np.ascontiguousarray(state.coefficients(source)[:,source_bands],dtype=np.complex128)
    ct = np.ascontiguousarray(state.coefficients(target)[:,target_bands],dtype=np.complex128)
    sm = np.ascontiguousarray(state.overlap(source),dtype=np.complex128)
    tm = np.ascontiguousarray(state.overlap(target),dtype=np.complex128)
    transport = audit_subspace_panels(action,cs,ct,sm,tm,
        source_subspace=source_space,target_subspace=target_space,contract=contract,
        metric_tolerance=metric_tolerance,leakage_tolerance=leakage_tolerance,
        probe_identity=probe_identity,budget=budget)
    if context != _periodic_context_identities(basis,system):
        raise ValueError("periodic geometry/basis context changed during audit")
    result = object.__new__(PeriodicStateSubspaceEvidence)
    for name,value in dict(sewing=sewing,transport=transport,source_index=source,target_index=target,
        subspace=subspace,source_bands=source_bands,target_bands=target_bands,
        source_character=character(source),target_character=character(target),
        operation_rotation=_freeze(rotation),operation_translation=_freeze(translation)).items():
        object.__setattr__(result,name,value)
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PeriodicStateGroupEvidence:
    """Whole-mesh state witness with a checked, distinct Seitz inventory."""
    declared_group: FiniteGroup
    group_transport: GroupTransportEvidence
    bridges: tuple[tuple[PeriodicStateSubspaceEvidence, ...], ...]
    maximum_fractional_seitz_residual: float
    worst_seitz_product: tuple[int, int]
    seitz_tolerance: float
    admitted_bytes: int
    admitted_work: int

    def __init__(self, *args, **kwargs):
        raise TypeError("use audit_periodic_state_group")

    @property
    def state(self):
        return self.bridges[0][0].state

    @property
    def passed_probe(self) -> bool:
        return self.group_transport.passed_probe

    @property
    def production_reduction_authorized(self) -> bool:
        return False


def audit_periodic_state_group(
    group: FiniteGroup, bridges: tuple[tuple[PeriodicStateSubspaceEvidence, ...], ...], *,
    seitz_tolerance: float, composition_tolerance: float,
    probe_identity: str, budget: Budget,
) -> PeriodicStateGroupEvidence:
    """Bind a supplied table/cocycle to recorded operations and one full state.

    bridges[g][k] must cover every native mesh point in index order, using
    one immutable state owner, context and mask selection. The operation
    descriptors must agree across each row; scalar grading comes from the
    admitted group. Equal spatial cosets modulo integer translations are
    refused within each grading, so a nonfaithful Gamma panel cannot disguise
    a duplicated operation. Completeness relative to a crystal is not inferred.

    Integer rotation products are checked exactly. For g h = T(ell) (gh),
    W_g tau_h + tau_g - tau_gh - ell is bounded in fractional-coordinate
    Euclidean norm by seitz_tolerance (0..1e-6). Exact rational arithmetic on
    the supplied binary64 translations precedes this comparison, preserving
    full int64 images and avoiding cancellation through float conversion.
    The native group is re-admitted with the actual recorded rotations to
    check their exact action on the integer cocycle. No operation is fitted.

    Native bridge records supply all space names, destinations, characters
    and local mixing to the shared group audit. A failed local or matrix
    group probe remains a failure. No native sewing is rerun. Budget admission
    includes retained sewing/mixing/descriptors, the common state once, both
    group owners, rational/control storage and composition work. Caller-owned
    strings, Python allocator overhead and BLAS workspace are excluded.
    The numerical state/source/production qualifications remain unchanged.
    """
    if not isinstance(group, FiniteGroup) or not isinstance(budget, Budget):
        raise TypeError("periodic group audit requires an admitted group and Budget")
    seitz_tolerance = _tolerance(seitz_tolerance, "seitz_tolerance")
    if seitz_tolerance > 1e-6:
        raise ValueError("seitz_tolerance must not exceed 1e-6 fractional coordinates")
    composition_tolerance = _tolerance(composition_tolerance, "composition_tolerance")
    _text(probe_identity, "probe_identity")
    n = group.order
    if (not isinstance(bridges, tuple) or len(bridges) != n or not bridges
            or not isinstance(bridges[0], tuple) or not bridges[0]
            or not isinstance(bridges[0][0], PeriodicStateSubspaceEvidence)):
        raise TypeError("periodic group requires a tuple inventory of native state bridges")
    first = bridges[0][0]
    state, count = first.state, int(first.state.n_kpoints)
    controls = 65536+8192*n*n+4096*n*count+state.resident_bytes+group._native.memory.bytes
    inventory_work = 8192*n**3*count
    budget.admit(controls,inventory_work)
    if any(not isinstance(row,tuple) or len(row) != count for row in bridges):
        raise ValueError("periodic group rows must cover the complete native mesh")
    if any(not isinstance(item,PeriodicStateSubspaceEvidence) for row in bridges for item in row):
        raise TypeError("periodic group requires audited native state bridges")
    retained = mixing = 0
    for row in bridges:
        for item in row:
            p = item.sewing.memory
            # Control storage is conservatively retained at its native peak.
            retained += (p.retained_sewing_bytes+p.control_storage_bytes+item.transport.mixing.nbytes
                         +item.operation_rotation.nbytes+item.operation_translation.nbytes)
            mixing += item.transport.mixing.nbytes
    native_plan = _core()._plan_symmetry_group(n,True,budget.native())
    r = int(state.n_effective_orbitals)
    shared_bytes = 4096+native_plan.bytes+1024*n*n*count+2048*n*count+32*n*count+mixing+256*r*r
    shared_work = 512*n**3*count+128*n*n*count*(r**3+r*r)
    admitted_bytes = controls+retained+native_plan.bytes+shared_bytes
    admitted_work = inventory_work+native_plan.work+shared_work
    budget.admit(admitted_bytes,admitted_work)
    context = (first.transport.metric.source_space.geometry,first.transport.metric.source_space.basis)
    for g,row in enumerate(bridges):
        for k,item in enumerate(row):
            if item.state is not state:
                raise ValueError("periodic group requires one immutable state owner")
            if item.source_index != k or item.subspace != first.subspace:
                raise ValueError("periodic group source order or selected mask mismatch")
            if item.transport.antiunitary != group.antiunitary(g):
                raise ValueError("periodic group scalar grading mismatch")
            for space in (item.transport.metric.source_space,item.transport.metric.target_space):
                if (space.geometry,space.basis) != context:
                    raise ValueError("periodic group geometry/basis context mismatch")
            if (not np.array_equal(item.operation_rotation,row[0].operation_rotation)
                    or not np.array_equal(item.operation_translation,row[0].operation_translation)):
                raise ValueError("periodic group operation changes across its k row")
    rotations = [tuple(int(x) for x in row[0].operation_rotation.flat) for row in bridges]
    translations = [tuple(Fraction.from_float(float(x)) for x in row[0].operation_translation)
                    for row in bridges]
    if rotations[group.identity] != (1,0,0,0,1,0,0,0,1):
        raise ValueError("periodic group identity rotation mismatch")
    tolerance_squared = Fraction.from_float(seitz_tolerance)**2
    def norm_squared(delta):
        return sum(x*x for x in delta)
    for g in range(n):
        for h in range(g):
            if rotations[g] != rotations[h] or group.antiunitary(g) != group.antiunitary(h):
                continue
            difference = tuple(a-b for a,b in zip(translations[g],translations[h]))
            reduced = tuple(x-round(x) for x in difference)
            if norm_squared(reduced) <= tolerance_squared:
                raise ValueError("duplicate periodic Seitz coset within scalar grading")
    table = np.empty((n,n),dtype=np.int64)
    images = np.empty((n,n,3),dtype=np.int64)
    worst = (group.identity,group.identity)
    maximum_squared, maximum_delta = Fraction(0), (Fraction(0),)*3
    for g in range(n):
        for h in range(n):
            gh = group.product(g,h)
            product = tuple(sum(rotations[g][3*a+c]*rotations[h][3*c+b] for c in range(3))
                            for a in range(3) for b in range(3))
            if product != rotations[gh]:
                raise ValueError("periodic group rotation product disagrees with table")
            ell = group.cocycle(g,h)
            delta = tuple(sum(rotations[g][3*a+c]*translations[h][c] for c in range(3))
                          +translations[g][a]-translations[gh][a]-int(ell[a]) for a in range(3))
            squared = norm_squared(delta)
            if squared > tolerance_squared:
                raise ValueError("periodic Seitz product disagrees with full lattice cocycle")
            if squared > maximum_squared:
                maximum_squared,maximum_delta,worst = squared,delta,(g,h)
            table[g,h],images[g,h] = gh,ell
    bound_group = FiniteGroup.from_table(table,np.array([group.antiunitary(g) for g in range(n)],dtype=np.uint8),
        identity=group.identity,identity_label=group.identity_label,
        rotations=np.array(rotations,dtype=np.int64).reshape(n,3,3),cocycle=images,budget=budget)
    identity_row = bridges[group.identity]
    spaces = tuple(item.transport.metric.source_space for item in identity_row)
    destinations = np.array([[item.target_index for item in row] for row in bridges],dtype=np.int64)
    records = tuple(tuple(item.transport for item in row) for row in bridges)
    evidence = audit_group_transport(bound_group,spaces,destinations,records,
        characters=tuple(item.source_character for item in identity_row),
        contract=first.transport.metric.contract,tolerance=composition_tolerance,
        probe_identity=probe_identity,budget=budget)
    result = object.__new__(PeriodicStateGroupEvidence)
    for name,value in dict(declared_group=group,group_transport=evidence,bridges=bridges,
        maximum_fractional_seitz_residual=float(np.hypot.reduce([float(x) for x in maximum_delta])),
        worst_seitz_product=worst,seitz_tolerance=seitz_tolerance,
        admitted_bytes=admitted_bytes,admitted_work=admitted_work).items():
        object.__setattr__(result,name,value)
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SelectedGroupTransportEvidence:
    """Selection covariance in retained coordinates, preserving parent evidence."""
    parent: GroupTransportEvidence | PeriodicStateGroupEvidence | SelectedGroupTransportEvidence
    selection_identity: str
    selections: tuple[np.ndarray, ...]
    group_transport: GroupTransportEvidence
    admitted_bytes: int
    admitted_work: int

    def __init__(self, *args, **kwargs):
        raise TypeError("use audit_selected_group_transport")

    @property
    def passed_probe(self) -> bool:
        return self.parent.passed_probe and self.group_transport.passed_probe

    @property
    def production_reduction_authorized(self) -> bool:
        return False


def audit_selected_group_transport(
    parent: GroupTransportEvidence | PeriodicStateGroupEvidence | SelectedGroupTransportEvidence,
    selections: tuple[np.ndarray, ...], *, selection_identity: str,
    metric_tolerance: float, leakage_tolerance: float, composition_tolerance: float,
    probe_identity: str, budget: Budget,
) -> SelectedGroupTransportEvidence:
    """Audit method-selected columns in each parent's retained coordinate space.

    selections[s] has shape (parent_rank[s], selected_rank[s]) and must be
    Euclidean-orthonormal. Different orbits may select different nonzero
    ranks; every destination in one orbit must select the same rank. Dense
    column mixing and scalar antiunitary conjugation are retained. No rank,
    threshold, eigenvector or compatible domain is selected or repaired here.

    Each local audit uses M[g,s] conjugate_if_a(g)(selections[s]) and checks
    leakage outside selections[g.s]. The resulting mixing is checked against
    the parent's exact table, destinations and full Bloch characters. Names
    bind the supplied policy label, parent space and actual column snapshot.
    The label declares policy; it does not authenticate its implementation.

    The result retains its complete parent, including native periodic state
    evidence or earlier selections, and cannot pass if any parent failed.
    Pass the wrapper itself for subsequent selections to retain that chain.
    Residuals concern retained coordinates, not a fresh AO/source audit or an
    accumulated error bound relative to the original AO metric. No PAO/PNO/TNO
    approximation or production execution is qualified by this diagnostic.

    Budget admission precedes scans/copies and covers the parent inventory,
    selection snapshots, induced mixing and conservative local/group work.
    Parent admission peaks are conservatively retained for wrapper inputs.
    Identity strings, their serialization, allocator overhead and BLAS
    workspace are excluded.
    """
    if not isinstance(parent, (GroupTransportEvidence, PeriodicStateGroupEvidence,
                               SelectedGroupTransportEvidence)) or not isinstance(budget, Budget):
        raise TypeError("selected group audit requires parent group evidence and Budget")
    _text(selection_identity, "selection_identity")
    _text(probe_identity, "probe_identity")
    metric_tolerance = _tolerance(metric_tolerance, "metric_tolerance")
    leakage_tolerance = _tolerance(leakage_tolerance, "leakage_tolerance")
    composition_tolerance = _tolerance(composition_tolerance, "composition_tolerance")
    if metric_tolerance >= 1:
        raise ValueError("orthonormality metric_tolerance must be less than one")
    ambient = parent if isinstance(parent, GroupTransportEvidence) else parent.group_transport
    group, count = ambient.group, len(ambient.spaces)
    n = group.order
    controls = 4096+2048*n*count
    budget.admit(controls, n*count)
    if not isinstance(selections, tuple) or len(selections) != count:
        raise TypeError("selections must cover each parent space in a tuple")
    ranks, selected = [], []
    for s, panel in enumerate(selections):
        _array(panel, "complex128", 2)
        rank = ambient.transports[group.identity][s].mixing.shape[0]
        if panel.shape[0] != rank or not 0 < panel.shape[1] <= rank:
            raise ValueError("selected panel rank or parent dimension mismatch")
        ranks.append(rank)
        selected.append(panel.shape[1])
    if isinstance(parent, GroupTransportEvidence):
        parent_bytes = (4096+group._native.memory.bytes+ambient.destinations.nbytes
                        +2048*n*count+sum(t.mixing.nbytes for row in ambient.transports for t in row))
    else:
        parent_bytes = parent.admitted_bytes
    payload = 2*sum(panel.nbytes for panel in selections)+16*n*sum(k*k for k in selected)
    local_bytes = max(4096+768*r*k+256*r*r+512*k*k for r,k in zip(ranks,selected))
    local_work = n*sum(288*r*r*k+512*r*k*k+64*r*r+128*r*k for r,k in zip(ranks,selected))
    group_bytes = (4096+group._native.memory.bytes+1024*n*n*count+2048*n*count
                   +32*n*count+16*n*sum(k*k for k in selected)+256*max(selected)**2)
    group_work = 512*n**3*count+128*n*n*sum(k**3+k*k for k in selected)
    admitted_bytes = controls+parent_bytes+payload+local_bytes+group_bytes
    admitted_work = n*count+local_work+group_work
    budget.admit(admitted_bytes, admitted_work)
    for g in range(n):
        for s in range(count):
            if selected[s] != selected[int(ambient.destinations[g,s])]:
                raise ValueError("selected rank mismatch within a group orbit")
    if any(not np.all(np.isfinite(panel)) for panel in selections):
        raise ValueError("nonfinite selected panel")
    snapshots = tuple(_freeze(panel) for panel in selections)
    spaces = []
    for space,panel in zip(ambient.spaces,snapshots):
        descriptor = json.dumps([selection_identity,space.geometry,space.basis,
            space.space,space.gauge,panel.shape],ensure_ascii=True,separators=(",", ":"))
        digest = sha256(descriptor.encode()+b"\0"+panel.tobytes()).hexdigest()
        spaces.append(SpaceIdentity(space.geometry,space.basis,
            f"selected:{selection_identity}:{digest}",f"parent-coordinate columns:{digest}"))
    spaces = tuple(spaces)

    class RetainedAction:
        def __init__(self, g, s, target):
            self.source, self.target = ambient.spaces[s], ambient.spaces[target]
            self.dimension = ranks[s]
            self.antiunitary = group.antiunitary(g)
            self.mixing = ambient.transports[g][s].mixing

        def apply(self, coefficients):
            return np.ascontiguousarray(self.mixing @ (coefficients.conj()
                if self.antiunitary else coefficients))

    rows = []
    for g in range(n):
        row = []
        for s in range(count):
            target = int(ambient.destinations[g,s])
            metric = np.eye(ranks[s],dtype=np.complex128)
            row.append(audit_subspace_panels(RetainedAction(g,s,target),
                snapshots[s],snapshots[target],metric,metric,
                source_subspace=spaces[s],target_subspace=spaces[target],contract=ambient.contract,
                metric_tolerance=metric_tolerance,leakage_tolerance=leakage_tolerance,
                probe_identity=probe_identity,budget=budget))
        rows.append(tuple(row))
    evidence = audit_group_transport(group,spaces,ambient.destinations,tuple(rows),
        contract=ambient.contract,characters=ambient.characters,tolerance=composition_tolerance,
        probe_identity=probe_identity,budget=budget)
    result = object.__new__(SelectedGroupTransportEvidence)
    for name,value in dict(parent=parent,selection_identity=selection_identity,
        selections=snapshots,group_transport=evidence,admitted_bytes=admitted_bytes,
        admitted_work=admitted_work).items():
        object.__setattr__(result,name,value)
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class GroupOperatorEvidence:
    """Static retained-operator covariance with the complete parent evidence."""
    parent: GroupTransportEvidence | PeriodicStateGroupEvidence | SelectedGroupTransportEvidence
    operators: tuple[np.ndarray, ...]
    probes: tuple[tuple[QualificationEvidence, ...], ...]
    admitted_bytes: int
    admitted_work: int

    def __init__(self, *args, **kwargs):
        raise TypeError("use audit_group_operators")

    @property
    def passed_probe(self) -> bool:
        return self.parent.passed_probe and all(p.passed_probe for row in self.probes for p in row)

    @property
    def production_reduction_authorized(self) -> bool:
        return False


def audit_group_operators(
    parent: GroupTransportEvidence | PeriodicStateGroupEvidence | SelectedGroupTransportEvidence,
    operators: tuple[np.ndarray, ...], *, contract: OperatorContract,
    tolerance: float, probe_identity: str, budget: Budget,
) -> GroupOperatorEvidence:
    """Check static linear operators in the parent's orthonormal coordinates.

    operators[s] is a square complex128 matrix at parent space s. The
    Frobenius residual for each operation/source is
    A[g.s] M[g,s] - M[g,s] conjugate_if_antiunitary(g)(A[s]).
    Tolerance is absolute, in the supplied operator's units. General complex
    linear operators are accepted; Hermiticity is not inferred or required.

    The separate operator contract declares this matrix family's provenance;
    it need not equal the parent's transport contract. Immutable snapshots
    retain the actual matrices, and every parent failure remains attached.
    Full Bloch characters remain checked in the parent group evidence; no
    extra character is multiplied into this one-operation intertwining law.

    These are projected, static matrices, not a density-dependent builder
    equivariance test, source authentication, or a test of operator leakage
    outside the retained space. No approximation or production reduction is
    authorized. Admission precedes payload scans/copies and reserves parent
    inventory, snapshots, probes and matrix workspace. Wrapper parents are
    charged at their recorded peak; identity strings, allocator overhead
    and BLAS workspace are excluded.
    """
    if not isinstance(parent, (GroupTransportEvidence, PeriodicStateGroupEvidence,
                               SelectedGroupTransportEvidence)) or not isinstance(budget, Budget):
        raise TypeError("group operator audit requires parent group evidence and Budget")
    if not isinstance(contract, OperatorContract):
        raise TypeError("group operator audit requires an operator contract")
    tolerance = _tolerance(tolerance)
    _text(probe_identity, "probe_identity")
    ambient = parent if isinstance(parent, GroupTransportEvidence) else parent.group_transport
    group, count = ambient.group, len(ambient.spaces)
    n = group.order
    controls = 4096+2048*n*count
    budget.admit(controls, n*count)
    if not isinstance(operators, tuple) or len(operators) != count:
        raise TypeError("operators must cover each parent space in a tuple")
    ranks = []
    for s, operator in enumerate(operators):
        _array(operator, "complex128", 2)
        rank = ambient.transports[group.identity][s].mixing.shape[0]
        if operator.shape != (rank, rank):
            raise ValueError("operator shape must match its parent retained rank")
        ranks.append(rank)
    if isinstance(parent, GroupTransportEvidence):
        parent_bytes = (4096+group._native.memory.bytes+ambient.destinations.nbytes
                        +2048*n*count+sum(t.mixing.nbytes for row in ambient.transports for t in row))
    else:
        parent_bytes = parent.admitted_bytes
    admitted_bytes = controls+parent_bytes+2*sum(a.nbytes for a in operators)+256*max(ranks)**2
    admitted_work = n*count+128*n*sum(r**3+r*r for r in ranks)
    budget.admit(admitted_bytes, admitted_work)
    if any(not np.all(np.isfinite(a)) for a in operators):
        raise ValueError("nonfinite retained operator")
    snapshots = tuple(_freeze(a) for a in operators)
    rows = []
    for g in range(n):
        row = []
        for s in range(count):
            target = int(ambient.destinations[g,s])
            mixing = ambient.transports[g][s].mixing
            source = snapshots[s].conj() if group.antiunitary(g) else snapshots[s]
            with np.errstate(over="ignore", invalid="ignore"):
                difference = snapshots[target] @ mixing-mixing @ source
                # Scale before squaring: finite tiny defects must not become
                # passing zero, nor finite large residuals become infinity.
                scale = float(np.max(np.abs(difference),initial=0.))
                residual = float(np.linalg.norm(difference/scale))*scale if scale else 0.
            if not np.isfinite(residual):
                raise ValueError("nonfinite operator covariance residual")
            row.append(QualificationEvidence(contract, ambient.spaces[s], ambient.spaces[target],
                "static retained-operator covariance", EvidenceKind.NUMERICAL,
                probe_identity, residual, tolerance))
        rows.append(tuple(row))
    result = object.__new__(GroupOperatorEvidence)
    for name,value in dict(parent=parent,operators=snapshots,probes=tuple(rows),
        admitted_bytes=admitted_bytes,admitted_work=admitted_work).items():
        object.__setattr__(result,name,value)
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SelectedOperatorEvidence:
    """Operator and adjoint leakage through a method-owned column selection."""
    parent: GroupOperatorEvidence | SelectedOperatorEvidence
    selection: SelectedGroupTransportEvidence
    operator_transport: GroupOperatorEvidence
    containment: tuple[QualificationEvidence, ...]
    adjoint_containment: tuple[QualificationEvidence, ...]
    admitted_bytes: int
    admitted_work: int

    def __init__(self, *args, **kwargs):
        raise TypeError("use audit_selected_operators")

    @property
    def passed_probe(self) -> bool:
        return (self.parent.passed_probe and self.operator_transport.passed_probe
                and all(p.passed_probe for p in self.containment+self.adjoint_containment))

    @property
    def production_reduction_authorized(self) -> bool:
        return False


def audit_selected_operators(
    parent: GroupOperatorEvidence | SelectedOperatorEvidence,
    selection: SelectedGroupTransportEvidence, *, covariance_tolerance: float,
    leakage_tolerance: float, probe_identity: str, budget: Budget,
) -> SelectedOperatorEvidence:
    """Check a selected reducing subspace of supplied retained operators.

    The selection must reference the exact group-evidence parent of the
    supplied operator audit. At each space, solve G B = Q^dagger A Q with
    G = Q^dagger Q, retaining the admitted finite orthonormality error.
    Measure ||A Q-Q B||_F / ||A Q||_F and the analogous residual for A^dagger
    independently. The latter checks coupling into the selection and is
    essential for general non-Hermitian operators. Zero action has zero
    leakage. Scaled norms preserve tiny nonzero leakage without squaring
    very large or small entries. Leakage tolerance is dimensionless.

    The compressed B matrices undergo the shared static covariance audit
    with absolute covariance_tolerance and the original operator contract.
    The complete operator and selection parents remain attached. Pass this
    wrapper itself for further selections so earlier leakage cannot vanish
    from the evidence chain when a later cut discards the offending block.

    This tests leakage within the parent's retained coordinates, not beyond
    the original AO/retained boundary, and does not qualify a producer or
    approximation policy. No rank or operator is repaired. Admission before
    payload access covers parent peaks, compression and nested covariance
    work; identity strings, allocator overhead and BLAS workspace are excluded.
    """
    if not isinstance(parent, (GroupOperatorEvidence, SelectedOperatorEvidence)):
        raise TypeError("selected operator audit requires parent operator evidence")
    if not isinstance(selection, SelectedGroupTransportEvidence) or not isinstance(budget, Budget):
        raise TypeError("selected operator audit requires a group selection and Budget")
    covariance_tolerance = _tolerance(covariance_tolerance, "covariance_tolerance")
    leakage_tolerance = _tolerance(leakage_tolerance, "leakage_tolerance")
    _text(probe_identity, "probe_identity")
    ambient = parent if isinstance(parent, GroupOperatorEvidence) else parent.operator_transport
    if selection.parent is not ambient.parent:
        raise ValueError("selection and operator evidence must share the exact parent")
    group_transport = selection.group_transport
    n, count = group_transport.group.order, len(selection.selections)
    controls = 4096+4096*n*count
    budget.admit(controls, n*count)
    ranks = tuple(q.shape for q in selection.selections)
    child_bytes = (4096+2048*n*count+selection.admitted_bytes
                   +32*sum(k*k for r,k in ranks)+256*max(k for r,k in ranks)**2)
    child_work = n*count+128*n*sum(k**3+k*k for r,k in ranks)
    admitted_bytes = (controls+parent.admitted_bytes+selection.admitted_bytes+child_bytes
                      +16*sum(k*k for r,k in ranks)
                      +1024*max(r*r+r*k+k*k for r,k in ranks))
    admitted_work = n*count+child_work+512*sum(r*r*k+r*k*k+k**3 for r,k in ranks)
    budget.admit(admitted_bytes, admitted_work)
    contract = ambient.probes[0][0].contract

    def relative_leakage(moved, difference):
        if not np.all(np.isfinite(moved)) or not np.all(np.isfinite(difference)):
            raise ValueError("nonfinite selected operator action or residual")
        scale = max(float(np.max(np.abs(moved))),float(np.max(np.abs(difference))))
        if not np.isfinite(scale):
            raise ValueError("nonfinite selected operator residual scale")
        if not scale:
            return 0.
        numerator = float(np.linalg.norm(difference/scale))
        denominator = float(np.linalg.norm(moved/scale))
        if not denominator:
            raise ValueError("unrepresentable relative selected operator residual")
        residual = numerator/denominator
        if not np.isfinite(residual):
            raise ValueError("nonfinite selected operator leakage")
        return residual

    compressed, forward, adjoint = [], [], []
    for s,(a,q) in enumerate(zip(ambient.operators,selection.selections)):
        gram = q.conj().T @ q
        for is_adjoint,records in ((False,forward),(True,adjoint)):
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                moved = (a.conj().T if is_adjoint else a) @ q
                b = np.ascontiguousarray(np.linalg.solve(gram,q.conj().T @ moved))
                leakage = relative_leakage(moved,moved-q @ b)
            if not is_adjoint:
                compressed.append(b)
            source = ambient.probes[0][s].source_space
            target = group_transport.spaces[s]
            relation = ("relative adjoint leakage" if is_adjoint else "relative operator leakage")
            records.append(QualificationEvidence(contract,source,target,
                relation+" outside the selected subspace",EvidenceKind.NUMERICAL,
                probe_identity,leakage,leakage_tolerance))
    transport = audit_group_operators(selection,tuple(compressed),contract=contract,
        tolerance=covariance_tolerance,probe_identity=probe_identity,budget=budget)
    result = object.__new__(SelectedOperatorEvidence)
    for name,value in dict(parent=parent,selection=selection,operator_transport=transport,
        containment=tuple(forward),adjoint_containment=tuple(adjoint),
        admitted_bytes=admitted_bytes,admitted_work=admitted_work).items():
        object.__setattr__(result,name,value)
    return result
