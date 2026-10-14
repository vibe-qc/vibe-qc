"""Fixed whole-orbit screening of an audited finite molecular source.

Complete audits measure omitted finite-tensor norms. Optional native Q/V
envelopes bound omitted orbits before reads; neither path uses a Schwarz/BUPO
estimate. Thompson and Ochsenfeld, doi:10.1063/1.4994190, Eqs. (6)-(12),
distinguish upper bounds from tight estimates; neither is inferred here
from geometry, a representative norm or a libint null tile.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import math

from .symmetry_shared import Budget, _integer, _tolerance
from .symmetry_quartets import ShellQuartetPlan, _MantaSource


def _immutable_source(source, *, session=True):
    # A mutable callback could change an omitted tile without another read.
    # Do not treat its self-declared identity as an immutable producer seal.
    if type(source) is not _MantaSource:
        raise TypeError('screening requires the immutable MANTA P0 source')
    from . import _vibeqc_core as core
    if (type(source._provider) is not core._manta.ReferenceJKBuilder
            or (session and type(source._session) is not core._manta.TileSession)):
        raise TypeError('screening requires an immutable native producer and session')


def quartet_screening_requirements(plan, *, budget: Budget):
    """Reserve the live stored plan, sort workspace and immutable mask."""
    if type(plan) is not ShellQuartetPlan or plan.mode != 'spatial':
        raise TypeError('screening requires a stored spatial quartet audit')
    _immutable_source(plan.source)
    count = len(plan.orbits)
    size = plan.required_bytes + 4096 + 256*count
    work = plan.required_work + 64*count*(1+count.bit_length())
    budget.admit(size, work)
    return Budget(size, work)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class QuartetScreening:
    """Immutable mask bound to one source qualification and omission policy.

    ``maximum_omitted_norm`` is an explicit AO tensor error budget, not a
    changed scientific acceptance tolerance. Norms refer to the original
    ordered finite source in its declared AO coordinates. They are not
    Euclidean-invariant norms of nonunitary Cartesian representatives.
    Contraction bounds exclude floating-point arithmetic roundoff.
    """
    _plan: ShellQuartetPlan = field(repr=False)
    keep: 'tuple[bool, ...] | _PackedKeep'
    audit_identity: str
    producer_identity: str
    source_identity: str
    policy_identity: str
    maximum_omitted_norm: float
    omitted_norm: float
    retained_residual_norm: float
    total_residual_norm: float
    omitted_representatives: int
    omitted_ordered_tiles: int
    omitted_integral_values: int
    required_bytes: int
    required_work: int
    retained_digest: str | None
    policy: str

    def __init__(self, plan, *, maximum_omitted_norm, budget: Budget):
        memory = quartet_screening_requirements(plan, budget=budget)
        cap = _tolerance(maximum_omitted_norm, 'maximum omitted norm')
        if plan.source.source_identity != plan.source_identity:
            raise ValueError('quartet source identity changed; requalify before screening')
        keep, omitted = [True]*len(plan.orbits), 0.
        # Deterministic, density-independent policy: smaller complete-orbit
        # norms first, canonical representative as the tie breaker.
        for i in sorted(range(len(keep)), key=lambda i:
                        (plan.orbits[i].source_norm, plan.orbits[i].representative)):
            candidate = math.hypot(omitted, plan.orbits[i].source_norm)
            if candidate > cap or not math.isfinite(candidate):
                break
            keep[i], omitted = False, candidate
        retained = 0.
        omitted_reps = omitted_tiles = omitted_values = 0
        for enabled, orbit in zip(keep, plan.orbits):
            if enabled:
                retained = math.hypot(retained, orbit.residual_norm)
            else:
                omitted_reps += 1
                omitted_tiles += len(orbit.members)
                omitted_values += sum(math.prod(int(plan.offsets[s+1]-plan.offsets[s])
                    for s in member.shells) for member in orbit.members)
        total = math.hypot(omitted, retained)
        if not math.isfinite(total):
            raise ValueError('nonrepresentable screened-source residual bound')
        # The actual mask changes the numerical source. A looser policy that
        # chooses the same mask changes only the policy receipt identity.
        digest = sha256(b'vibeqc-fixed-orbit-source-v1\0')
        digest.update(plan.audit_identity.encode())
        digest.update(bytes(keep))
        identity = digest.hexdigest()
        policy = sha256((identity+'\0smallest-orbit-first-v1\0'+cap.hex()).encode()).hexdigest()
        for key, value in dict(_plan=plan, keep=tuple(keep),
                audit_identity=plan.audit_identity, producer_identity=plan.source_identity,
                source_identity=identity, policy_identity=policy,
                maximum_omitted_norm=cap, omitted_norm=omitted,
                retained_residual_norm=retained, total_residual_norm=total,
                omitted_representatives=omitted_reps, omitted_ordered_tiles=omitted_tiles,
                omitted_integral_values=omitted_values,
                retained_digest=None, policy='smallest-orbit-first-v1',
                required_bytes=memory.maximum_bytes, required_work=memory.maximum_work).items():
            object.__setattr__(self, key, value)

    def _validate(self, plan):
        if self._plan is not plan:
            raise ValueError('screening certificate belongs to a different source audit')
        if plan.source.source_identity != self.producer_identity:
            raise ValueError('quartet source identity changed; requalify before screening')

    @property
    def receipt(self):
        """Fresh JSON-compatible provenance, without densities or plan payload."""
        fields = ('audit_identity', 'producer_identity', 'source_identity',
                  'policy_identity', 'maximum_omitted_norm', 'omitted_norm',
                  'retained_residual_norm', 'total_residual_norm',
                  'omitted_representatives', 'omitted_ordered_tiles',
                  'omitted_integral_values', 'required_bytes', 'required_work')
        bounds = getattr(self._plan, 'bounds', None)
        return dict(schema='vibeqc-fixed-orbit-screen-v1',
                    bound_kind=('audited-finite-AO-tensor-Frobenius' if bounds is None else
                                'producer-bounds-plus-retained-audit-Frobenius'),
                    omitted_norm_kind='measured' if bounds is None else 'upper-bound',
                    qualification_scope='complete-source' if bounds is None else 'retained-orbits',
                    producer_bounds=None if bounds is None else bounds.receipt,
                    bound_queries=getattr(self._plan, 'bound_queries', 0),
                    bound_lookup_probes=getattr(self._plan, 'bound_lookup_probes', 0),
                    bound_lookup_cache_hits=getattr(self._plan, 'bound_lookup_cache_hits', 0),
                    bound_lookup_cache_peak_entries=getattr(self._plan, 'bound_lookup_cache_peak_entries', 0),
                    hierarchy_closure_refusals=getattr(self._plan, 'hierarchy_closure_refusals', 0),
                    density_independent=True, arithmetic_roundoff_included=False,
                    primitive_precision=0., source_tolerance=self._plan.source_tolerance,
                    policy=self.policy,
                    mask_payload_bytes=(len(self.keep.data) if isinstance(self.keep, _PackedKeep)
                                        else len(self.keep)),
                    **{key: getattr(self, key) for key in fields})


def _screening_overhead(plan, screening):
    if getattr(plan, 'bounds', None) is not None and screening is not plan.screening:
        raise ValueError('partially audited source requires its fixed mask; use the unreduced provider or full audit')
    if screening is None:
        return 0, 0
    if type(screening) is not QuartetScreening:
        raise TypeError('an immutable QuartetScreening certificate is required')
    screening._validate(plan)
    return (screening.required_bytes-plan.required_bytes,
            screening.required_work-plan.required_work)


@dataclass(frozen=True, slots=True)
class _PackedKeep:
    """One immutable bit per streamed orbit; no quartet/member inventory."""
    data: bytes
    count: int

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        index = _integer(index, 'mask index')
        if index >= self.count:
            raise IndexError(index)
        return bool(self.data[index // 8] & (1 << (index % 8)))


def _compact_screening_overhead(count):
    # Mutable admission buffer and immutable export overlap at sealing.
    return 8192 + 2*((count+7)//8), 128*count


class _StreamingScreenAudit:
    """Transient one-pass admission state, consumed by the complete audit.

    Canonical schedule order greedily spends the explicit tensor-norm cap.
    This avoids a global sort and retains no per-orbit norms or digests.
    It is a different policy from the stored smallest-first certificate.
    """
    policy = 'schedule-order-first-fit-v1'

    def __init__(self, source, count, cap):
        _immutable_source(source, session=False)
        self.cap = _tolerance(cap, 'maximum omitted norm')
        self.mask = bytearray((count+7)//8)
        self.count = count
        self.omitted = self.retained = 0.
        self.reps = self.tiles = self.values = self.visited = 0
        self.digest = sha256()

    def observe(self, orbit, original, norm, residual, offsets):
        from .symmetry_quartets_streaming import _digest_tile
        if not math.isfinite(norm) or not math.isfinite(residual):
            raise ValueError('nonrepresentable finite-source orbit norm')
        index = self.visited
        self.visited += 1
        candidate = math.hypot(self.omitted, norm)
        if candidate <= self.cap:
            self.omitted = candidate
            self.reps += 1
            self.tiles += len(orbit.members)
            self.values += sum(math.prod(int(offsets[s+1]-offsets[s])
                for s in member.shells) for member in orbit.members)
        else:
            self.mask[index//8] |= 1 << (index%8)
            self.retained = math.hypot(self.retained, residual)
            _digest_tile(self.digest, orbit.representative, original)

    def seal(self, plan):
        _immutable_source(plan.source)
        if self.visited != self.count:
            raise ValueError('streamed screening census failed')
        total = math.hypot(self.omitted, self.retained)
        if not math.isfinite(total):
            raise ValueError('nonrepresentable screened-source residual bound')
        data = bytes(self.mask)
        digest = sha256(b'vibeqc-streamed-orbit-source-v1\0')
        digest.update(plan.audit_identity.encode())
        digest.update(self.count.to_bytes(8, 'big'))
        digest.update(data)
        identity = digest.hexdigest()
        policy = self.policy
        policy_id = sha256((identity+'\0'+policy+'\0'+self.cap.hex()).encode()).hexdigest()
        certificate = object.__new__(QuartetScreening)
        for key, value in dict(_plan=plan, keep=_PackedKeep(data, self.count),
                audit_identity=plan.audit_identity, producer_identity=plan.source_identity,
                source_identity=identity, policy_identity=policy_id, policy=policy,
                maximum_omitted_norm=self.cap, omitted_norm=self.omitted,
                retained_residual_norm=self.retained, total_residual_norm=total,
                omitted_representatives=self.reps, omitted_ordered_tiles=self.tiles,
                omitted_integral_values=self.values, retained_digest=self.digest.hexdigest(),
                required_bytes=plan.required_bytes,
                required_work=plan.required_work).items():
            object.__setattr__(certificate, key, value)
        return certificate
