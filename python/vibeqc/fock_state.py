"""Evaluated method states built from the existing fixed Fock operators.

This layer supplies no HF/KS energy formula, occupation rule, source selection
or SCF convergence decision. Dense producers opt into owned snapshots. Native
bounded kernels retain their own admitted ownership and execution contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np

from .solvers.operators import FockOperator, OperatorFrame, OperatorMetadata, _hermitian, _snapshot


@dataclass(frozen=True)
class SourceProvenance:
    """A portable frozen provider record, separate from the orbital frame.

    A digest identifies the declared numerical source; it does not certify
    symmetry, integral accuracy, finite-support convergence or immutability of
    a builder's captures. Builders must own or check those captures themselves.
    """
    provider: str
    details_json: str

    def __post_init__(self):
        if not self.provider:
            raise ValueError("Source provider identity must be nonempty")
        details = json.loads(self.details_json)
        if not isinstance(details, dict):
            raise ValueError("Source details must be a mapping")
        object.__setattr__(self, "details_json", json.dumps(
            details, sort_keys=True, separators=(",", ":"), allow_nan=False))

    @classmethod
    def capture(cls, provider, **details):
        return cls(provider, json.dumps(details, allow_nan=False))

    @property
    def identity(self):
        return hashlib.sha256((self.provider + "\n" + self.details_json).encode()).hexdigest()

    def as_dict(self):
        return dict(provider=self.provider, identity=self.identity,
                    details=json.loads(self.details_json))


def _energy_scalar(value):
    if np.ndim(value) != 0:
        raise ValueError("Energy values must be scalars")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError("Energy values must be finite")
    return scalar


def _scalars(values):
    items = tuple((str(key), _energy_scalar(value)) for key, value in values)
    if len({key for key, _ in items}) != len(items):
        raise ValueError("Scalar component names must be unique")
    if any(not key for key, _ in items):
        raise ValueError("Scalar component names must be nonempty")
    return items


@dataclass(frozen=True, kw_only=True)
class EnergyComponents:
    """Method-evaluated electronic energy and its additive decomposition.

    XC energy is an explicit method component, never inferred by a half trace
    of the KS potential. Nuclear and extra corrections remain separate.
    """
    electronic: float
    nuclear: float
    components: tuple[tuple[str, float], ...]
    corrections: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        electronic, nuclear = _energy_scalar(self.electronic), _energy_scalar(self.nuclear)
        components, corrections = _scalars(self.components), _scalars(self.corrections)
        try:
            component_sum = math.fsum(v for _, v in components)
            total = electronic + nuclear + math.fsum(v for _, v in corrections)
        except OverflowError as exc:
            raise ValueError("Energy component sums and total must be finite") from exc
        if not math.isfinite(total):
            raise ValueError("Total energy must be finite")
        if not math.isclose(component_sum, electronic,
                            rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError("Electronic components do not sum to the evaluated energy")
        object.__setattr__(self, "electronic", electronic)
        object.__setattr__(self, "nuclear", nuclear)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "corrections", corrections)

    @property
    def total(self):
        return self.electronic + self.nuclear + math.fsum(v for _, v in self.corrections)


@dataclass(frozen=True)
class FockChannel:
    """One physical spin/k Fock and the density evaluated by its builder."""
    operator: FockOperator
    density: np.ndarray
    weight: float

    def __post_init__(self):
        d = _snapshot(self.density)
        _hermitian(d, "Evaluated density")
        if d.shape != (self.operator.n, self.operator.n):
            raise ValueError("Density and Fock dimensions differ")
        if self.operator.metadata.role != "physical" or self.operator.metadata.kind not in {"fock", "ks"}:
            raise ValueError("An evaluated channel requires a physical Fock/KS operator")
        if self.operator.frame is None:
            raise ValueError("An evaluated channel requires an explicit frame")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Brillouin-zone weights must be finite and nonnegative")
        object.__setattr__(self, "density", d)


@dataclass(frozen=True, kw_only=True)
class FockBuild:
    """One evaluated density epoch, including all coupled spin/k channels.

    No callback into mutable method/provider state survives construction.
    ``from_dense`` is the first producer adapter; it owns D and F, shares the
    validated frame, and retains neither J/K workspaces nor a history of builds.
    This dense adapter does not advertise a hard memory bound.
    """
    method: str
    generation: int
    source: SourceProvenance
    energy: EnergyComponents
    channels: tuple[FockChannel, ...]
    density_id: str
    identity: str

    def __post_init__(self):
        if not self.method or not self.channels or self.generation < 0:
            raise ValueError("An evaluated build requires a method, channels and generation")
        object.__setattr__(self, "channels", tuple(self.channels))
        labels = [(c.operator.metadata.spin, c.operator.metadata.kpoint,
                   c.operator.metadata.symmetry_block) for c in self.channels]
        if len(set(labels)) != len(labels):
            raise ValueError("Duplicate spin/k/symmetry channels")
        for channel in self.channels:
            meta = channel.operator.metadata
            if (meta.build_id, meta.density_id, meta.generation) != (
                    self.identity, self.density_id, self.generation):
                raise ValueError("Operator and evaluated build identities differ")

    @classmethod
    def from_dense(cls, *, method, generation, source, energy, densities, focks,
                   frames, spins, kpoints, weights, kind="fock"):
        """Freeze a complete coupled build; weights never scale its operators."""
        inputs = tuple(map(tuple, (densities, focks, frames, spins, kpoints, weights)))
        if len({len(items) for items in inputs}) != 1 or not inputs[0]:
            raise ValueError("All channel inputs must have the same nonzero length")
        if not isinstance(generation, (int, np.integer)) or generation < 0:
            raise ValueError("Build generation must be a nonnegative integer")
        digest = hashlib.sha256()
        for d, frame, spin, kpoint, weight in zip(inputs[0], inputs[2], inputs[3], inputs[4], inputs[5]):
            if not isinstance(frame, OperatorFrame):
                raise TypeError("Build channels require OperatorFrame instances")
            a = np.asarray(d)
            digest.update(json.dumps([frame.identity, spin, kpoint, float(weight)], allow_nan=False).encode())
            digest.update(str((a.shape, a.dtype.str)).encode())
            digest.update(a.tobytes())
        density_id = digest.hexdigest()
        # The full state includes method-evaluated energy, not just its Fock.
        # Keep density/source/frame identities independent of this record.
        digest.update(json.dumps([
            "fock-build-v1", method, int(generation), source.identity, kind,
            energy.electronic, energy.nuclear, energy.components, energy.corrections,
        ], allow_nan=False, separators=(",", ":")).encode())
        for f in inputs[1]:
            a = np.asarray(f)
            digest.update(str((a.shape, a.dtype.str)).encode())
            digest.update(a.tobytes())
        identity = digest.hexdigest()
        channels = tuple(FockChannel(
            FockOperator(matrix=f, frame=frame, metadata=OperatorMetadata(
                kind=kind, spin=spin, kpoint=kpoint, generation=int(generation),
                build_id=identity, density_id=density_id)), d, float(weight)
        ) for d, f, frame, spin, kpoint, weight in zip(*inputs))
        return cls(method=method, generation=int(generation), source=source,
                   energy=energy, channels=channels, density_id=density_id, identity=identity)

    def solver_operator(self, channel, matrix, *, modification):
        """Associate a shifted/extrapolated solve with its physical parent."""
        physical = self.channels[channel].operator
        if not modification:
            raise ValueError("Describe the solver Fock modification explicitly")
        return FockOperator(matrix=matrix, frame=physical.frame, metadata=replace(
            physical.metadata, role="solver", modification=modification))

    @property
    def metric_electron_counts(self):
        # Per-channel counts; caller applies the BZ weights in its own sums.
        return tuple(float((np.trace(c.density) if c.operator.overlap is None else
                            np.einsum("ij,ji->", c.density, c.operator.overlap)).real)
                     for c in self.channels)

    @property
    def commutator_norm(self):
        total = 0.0
        for c in self.channels:
            ds = c.density if c.operator.overlap is None else c.density @ c.operator.overlap
            fds = c.operator.apply(ds)
            total += c.weight * float(np.linalg.norm(fds - fds.conj().T) ** 2)
        return math.sqrt(total)

    @property
    def storage_nbytes(self):
        """Retained dense array bytes, including each shared frame only once.

        Excludes Python objects, builders, integrals, temporaries and eigensolver
        work. This is a storage observation, not a peak-memory admission claim.
        """
        frames = {id(c.operator.frame): c.operator.frame for c in self.channels}
        return sum(c.density.nbytes + c.operator.matrix.nbytes + c.operator.diagonal.nbytes
                   for c in self.channels) + sum(f.storage_nbytes for f in frames.values())

    def output_fields(self):
        """Scalar identities usable by the existing output provenance route."""
        return dict(fock_build_id=self.identity, evaluated_density_id=self.density_id,
                    fock_build_generation=self.generation, fock_source_id=self.source.identity,
                    fock_source_provider=self.source.provider, fock_operator_role="physical",
                    fock_frame_ids=[c.operator.frame.identity for c in self.channels],
                    evaluated_metric_electrons=list(self.metric_electron_counts),
                    fock_energy_components=dict(self.energy.components),
                    fock_energy_corrections=dict(self.energy.corrections))
