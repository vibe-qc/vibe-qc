"""Opt-in molecular HF consumer of a qualified, fixed quartet mask.

The Dupuis-King shell transport and finite-source certificate live in the
shared quartet layer. This method adapter supplies real native JKBuilder
calls and HF energy conventions, without selecting an electronic irrep,
projecting a density, changing SCF convergence, or admitting derivatives.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
import math
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from . import _vibeqc_core as core
from .fock_state import EnergyComponents, FockBuild, SourceProvenance
from .solvers.operators import OperatorFrame, _hermitian, _snapshot
from .symmetry_shared import Budget, _integer
from .symmetry_quartets import QuartetWork, _norm
from .symmetry_quartets_batch import quartet_batch_requirements
from .symmetry_quartets_screening import QuartetScreening
from .symmetry_quartets_ordered import (
    MolecularSourceQualification, qualified_molecular_source,
    ordered_quartet_requirements, ordered_quartet_jk,
)

if TYPE_CHECKING:
    from .symmetry_ri import MolecularRIRouteReceipt

_ROUTE_SIGNATURES = (('j',), ('k',), ('jk',), ('k', 'k'), ('j', 'k', 'k'))
_ROUTE_RECEIPT_BYTES = 131072
_ROUTE_REPETITIONS = 2


def _calibration_cannot_pay(expected_builds, repetitions=_ROUTE_REPETITIONS):
    # Each pair pays for one baseline build. Even a free candidate cannot
    # save more than min(baseline_times) per subsequent call. Thus the measured
    # policy below cannot repay its own probes when expected <= repetitions.
    return expected_builds <= repetitions


def _bound_product(a, b):
    value = a*b
    if not math.isfinite(value) or (value == 0. and a > 0. and b > 0.):
        raise ValueError('nonrepresentable HF error bound')
    return value


@dataclass(frozen=True)
class MolecularJKReceipt:
    """One successful build; no matrices or mutable provider captures."""
    source_identity: str
    targets: tuple[str, ...]
    density_error_bounds: tuple[float, ...]
    work: QuartetWork
    scalar_products: int
    block_contractions: int
    transport_scalar_products: int
    seconds: float
    logical_bytes: int
    route: 'MolecularRouteReceipt | MolecularRIRouteReceipt | None' = None
    executor: str = 'native'
    reused_from_builds: tuple[int, ...] = ()


@dataclass(frozen=True)
class _EvaluatedJK:
    """One owned evaluation on exact input bytes, scoped to one method."""
    densities: tuple[np.ndarray, ...]
    j: np.ndarray
    k: np.ndarray
    receipt: MolecularJKReceipt
    build_numbers: tuple[int, ...]

    @property
    def nbytes(self):
        return sum(d.nbytes for d in self.densities)+self.j.nbytes+self.k.nbytes


@dataclass(frozen=True)
class MolecularRouteReceipt:
    qualification: MolecularSourceQualification
    selected: str
    reason: str
    expected_builds: int = 0
    calibration_seconds: float = 0.
    maximum_difference: float = 0.
    samples: tuple[MolecularJKReceipt, ...] = ()
    logical_bytes: int = 0
    modeled_work: int = 0
    targets: tuple[str, ...] = ()


def _select_route(baseline_seconds, candidate_seconds, calibration_seconds, expected_builds,
        *, baseline='ordered', candidate='native'):
    # A conservative measured interval, not theoretical orbit speedup. Setup
    # already paid for source qualification is sunk; calibration is explicit.
    saving = min(baseline_seconds)-max(candidate_seconds)
    if saving > 0 and saving*expected_builds > calibration_seconds:
        return candidate, 'measured saving repays calibration within expected builds'
    return baseline, 'measured saving does not repay calibration within expected builds'


@dataclass(frozen=True)
class MolecularJKTotals:
    builds: int = 0
    work: QuartetWork = QuartetWork()
    scalar_products: int = 0
    block_contractions: int = 0
    transport_scalar_products: int = 0
    seconds: float = 0.
    maximum_logical_bytes: int = 0


class _MolecularJKCapture:
    """Shared method-local real AO snapshots, never source qualification.

    Both fitted and quartet bridges supply their qualified source identity,
    resource admission and density validation. No integral/provider state or
    prepared-source lifetime is owned here; only exact evaluated J/K bytes.
    """
    def _initialize_capture(self, capture_terminal):
        if type(capture_terminal) is not bool:
            raise TypeError('capture_terminal must be a boolean')
        self._capture_terminal = capture_terminal
        self._terminal = self._pending_j = None
        self._capture_targets = None
        self._capture_count = self._capture_values = self._comparison_values = 0
        self._capture_seconds = 0.

    def _capture_requirements(self):
        # Eight live real AO matrices and overlapping copies/comparisons.
        return ((4096+128*self.n**2, 128*self.n**2)
                if self._capture_terminal else (0, 0))

    def _begin_capture(self, targets):
        capture = self._capture_terminal and targets in (('jk',), ('j', 'k', 'k'))
        if capture:
            # A failed replacement cannot leave a prior full build eligible.
            self._terminal = self._pending_j = None
        return capture

    @property
    def terminal_capture(self):
        return dict(enabled=self._capture_terminal, captures=self._capture_count,
            copied_values=self._capture_values, comparison_values=self._comparison_values,
            seconds=self._capture_seconds,
            snapshot_bytes=0 if self._terminal is None else self._terminal.nbytes,
            pending_bytes=0 if self._pending_j is None else self._pending_j.nbytes,
            reservation_bytes=4096+128*self.n**2 if self._capture_terminal else 0)

    def _capture(self, densities, j, k, receipt, full):
        started = perf_counter()
        snapshot = None
        if full:
            self._capture_targets = receipt.targets
            snapshot = _EvaluatedJK(tuple(_snapshot(d) for d in densities),
                _snapshot(j), _snapshot(k), receipt, (self._totals.builds,))
            copied = snapshot.nbytes//8
        elif self._capture_targets != ('j', 'k', 'k'):
            # Native converged RHF builds J(D) and K(D) separately. Join only
            # their exact input bytes and immutable source; never mix spins
            # or infer that two arbitrary response channels form a state.
            if receipt.targets == ('j',):
                self._pending_j = _EvaluatedJK((_snapshot(densities[0]),),
                    _snapshot(j), _snapshot(k), receipt, (self._totals.builds,))
                self._capture_values += self._pending_j.nbytes//8
            elif receipt.targets == ('k',) and self._pending_j is not None:
                pending = self._pending_j
                self._comparison_values += densities[0].size
                if (pending.receipt.source_identity == receipt.source_identity
                        and np.array_equal(densities[0].view(np.uint64),
                                           pending.densities[0].view(np.uint64))):
                    a = pending.receipt
                    combined = replace(receipt, targets=('jk',),
                        density_error_bounds=(max(*a.density_error_bounds,
                                                  *receipt.density_error_bounds),),
                        work=QuartetWork(**{name: getattr(a.work, name)+getattr(receipt.work, name)
                            for name in QuartetWork.__dataclass_fields__}),
                        scalar_products=a.scalar_products+receipt.scalar_products,
                        block_contractions=a.block_contractions+receipt.block_contractions,
                        transport_scalar_products=a.transport_scalar_products+receipt.transport_scalar_products,
                        seconds=a.seconds+receipt.seconds, route=None, executor='joined-J/K')
                    snapshot = _EvaluatedJK(pending.densities, pending.j, _snapshot(k),
                        combined, (*pending.build_numbers, self._totals.builds))
                    copied = k.size
                    self._pending_j = None
                    self._capture_targets = ('jk',)
        if snapshot is not None:
            self._terminal = snapshot
            self._capture_count += 1
            self._capture_values += copied
        self._capture_seconds += perf_counter()-started

    def reuse_evaluated_channels(self, densities, *, targets):
        """Return a zero-new-work receipt only for an exact owned evaluation.

        No spatial action, tolerance, spin permutation or density digest alone
        can match an input. Standalone response calls do not overwrite this
        physical full-channel slot. Mismatches leave the cold path available.
        """
        reservation = self.requirements(targets)
        source_identity = self._qualified_source_identity()
        if not isinstance(densities, tuple) or len(densities) != len(targets):
            raise TypeError('densities must be a tuple matching the target channels')
        for d in densities:
            self._density(d)
        saved = self._terminal
        if saved is None or saved.receipt.targets != targets:
            return None
        if saved.receipt.source_identity != source_identity:
            return None
        started = perf_counter()
        for d, previous in zip(densities, saved.densities):
            self._comparison_values += d.size
            # Equal size bit views preserve signed zero and sub-ULP input
            # differences. Noncontiguous native inputs are valid too.
            if not np.array_equal(d.view(np.uint64), previous.view(np.uint64)):
                return None
        receipt = replace(saved.receipt, work=QuartetWork(), scalar_products=0,
            block_contractions=0, transport_scalar_products=0,
            seconds=perf_counter()-started, logical_bytes=reservation.maximum_bytes,
            reused_from_builds=saved.build_numbers)
        return saved.j, saved.k, receipt

    def reset_state(self):
        # Preserve work accounting, discard any optional physical capture.
        self._terminal = self._pending_j = None


class ScreenedMolecularJK(_MolecularJKCapture, core.JKBuilder):
    """Real AO bridge to native RHF/UHF for one qualified molecular source.

    Stored/streamed audited masks and ordered producer-bound masks share
    this boundary, with distinct execution permissions.

    Pass explicitly to ``run_rhf_scf_with_jk`` or ``run_uhf_scf_with_jk``.
    RHF fuses J and K in one representative traversal. Native UHF uses
    ``build_jk_uhf`` to share that traversal across total-density J and both
    spin K channels. Optional terminal capture retains one owned full-channel
    evaluation, not iteration history or an incremental-density approximation.

    The reservation covers the live quartet plan and this contraction's
    logical buffers, not the surrounding SCF driver or FockBuild snapshots.
    Dynamic Schwarz thresholds and incremental state are deliberately
    inactive: the admitted density-independent mask remains fixed.
    """

    def __init__(self, screening, *, budget: Budget, executor='native', capture_terminal=False):
        super().__init__()
        if type(screening) is not QuartetScreening:
            raise TypeError('an immutable QuartetScreening certificate is required')
        screening._validate(screening._plan)
        self._qualification = qualified_molecular_source(screening)
        self._screening = screening
        self._budget = budget
        self._executor = executor
        self._n = int(screening._plan.offsets[-1])
        self._initialize_capture(capture_terminal)
        self._routes = {}
        self._route_request = None
        self.requirements(('jk',))
        self._totals = MolecularJKTotals()
        self._last_build = None

    @property
    def screening(self):
        return self._screening

    @property
    def n(self):
        return self._n

    @property
    def totals(self):
        return self._totals

    @property
    def last_build(self):
        return self._last_build

    @property
    def route_calibrations(self):
        return tuple(self._routes.items())

    def fork(self, *, capture_terminal=False):
        """Fresh method counters with the same source and measured routes.

        Only immutable scalar calibration receipts are shared. No densities,
        Fock matrices, physical build receipts or cumulative work are copied.
        Callers must serialize access to the source's native tile session.
        """
        result = ScreenedMolecularJK(self._screening, budget=self._budget,
            executor=self._executor, capture_terminal=capture_terminal)
        result._routes = dict(self._routes)
        return result

    def requirements(self, targets):
        """Count-only reservation, before density inspection or tile reads."""
        if self._executor == 'auto':
            executors = ('ordered', 'native') if self._qualification.equivariance_qualified else ('ordered',)
            choices = [self._requirements_for(targets, e) for e in executors]
            return Budget(max(b.maximum_bytes for b in choices),
                          max(b.maximum_work for b in choices))
        return self._requirements_for(targets, self._executor)

    def _requirements_for(self, targets, executor):
        if executor == 'ordered':
            base = ordered_quartet_requirements(self._screening, len(targets),
                targets=targets, budget=self._budget)[0]
        else:
            base = quartet_batch_requirements(self._screening._plan, len(targets),
                targets=targets, budget=self._budget, executor=executor,
                screening=self._screening)
        # Real inputs, complex packing, real exports and RHF fusion scratch.
        # Kernel inputs/exports are also reserved by base; double counting
        # here intentionally bounds overlapping bridge lifetimes.
        count = len(targets)
        outputs = sum(len(t) for t in targets)
        size = base.maximum_bytes + 65536 + 32*(count+outputs+3)*self.n**2
        if self._executor == 'auto':
            # At most five HF signatures, ten scalar samples each, each
            # containing at most three density bounds; no matrix history.
            size += _ROUTE_RECEIPT_BYTES
        capture_bytes, capture_work = self._capture_requirements()
        size += capture_bytes
        work = base.maximum_work + 32*(count+outputs+3)*self.n**2 + capture_work
        self._budget.admit(size, work)
        return Budget(size, work)

    def _qualified_source_identity(self):
        qualified_molecular_source(self._screening)
        return self._screening.source_identity

    def _density(self, value):
        # Never let NumPy or Eigen discard an imaginary part implicitly.
        if not isinstance(value, np.ndarray) or value.dtype != np.float64:
            raise TypeError('native molecular HF requires real float64 AO densities')
        if value.shape != (self.n, self.n) or not np.isfinite(value).all():
            raise ValueError('invalid molecular AO density shape or values')
        return value

    def calibrate_routes(self, *, targets, expected_builds, repetitions=_ROUTE_REPETITIONS):
        """Opt-in measured routing for this exact mask and channel signature.

        Uses fixed nonsymmetric probes, alternating execution order. Probe
        work is separately recorded and never counted as physical SCF builds.
        Uncalibrated signatures use ordered execution; no online retuning of
        the mask, symmetry/state projection or provider substitution occurs.
        """
        if self._executor != 'auto':
            raise ValueError('route calibration requires executor="auto"')
        if targets not in _ROUTE_SIGNATURES:
            raise ValueError('calibration requires a native HF channel signature')
        expected_builds = _integer(expected_builds, 'expected builds', positive=True)
        repetitions = _integer(repetitions, 'repetitions', positive=True)
        if repetitions > 5:
            raise ValueError('at most five paired route probes are admitted')
        reservation = self.requirements(targets)
        if (_calibration_cannot_pay(expected_builds, repetitions)
                or not self._qualification.equivariance_qualified):
            reason = ('probe cost cannot repay within expected builds'
                if _calibration_cannot_pay(expected_builds, repetitions) else
                'ordered source has no angular-equivariance qualification')
            route = MolecularRouteReceipt(self._qualification, 'ordered', reason,
                expected_builds, logical_bytes=reservation.maximum_bytes, targets=targets)
            self._routes[targets] = route
            return route
        # Two real result stacks, trial matrices, comparison temporaries and
        # receipts coexist with one executing route. Bound all probe work.
        size = reservation.maximum_bytes + 65536 + 64*(len(targets)+sum(map(len, targets)))*self.n**2
        work = 2*repetitions*reservation.maximum_work
        self._budget.admit(size, work)
        self._routes.pop(targets, None)
        start = perf_counter()
        rng = np.random.default_rng(64129)
        densities = tuple(rng.uniform(-1., 1., (self.n, self.n)) for _ in targets)
        samples, times, difference = [], {'ordered': [], 'native': []}, 0.
        for repeat in range(repetitions):
            order = ('ordered', 'native') if repeat % 2 == 0 else ('native', 'ordered')
            results = {}
            for executor in order:
                j, k, receipt = self._execute(densities, targets, executor)
                results[executor] = (j, k)
                samples.append(receipt)
                times[executor].append(receipt.seconds)
            # Retained-source residual, not the omission budget: both routes
            # have exactly the same omitted tiles. Roundoff is still tested
            # at the existing source tolerance, never loosened by routing.
            for family, flag in enumerate(('j', 'k')):
                channels = [i for i, t in enumerate(targets) if flag in t]
                for out, channel in enumerate(channels):
                    delta = _norm(results['ordered'][family][out]-results['native'][family][out])
                    difference = max(difference, delta)
                    limit = (self.screening.retained_residual_norm*_norm(densities[channel])
                             + self.screening._plan.source_tolerance)
                    if not math.isfinite(delta) or delta > limit:
                        raise ValueError('route probe disagrees with qualified retained source')
            del results
        elapsed = perf_counter()-start
        selected, reason = _select_route(times['ordered'], times['native'], elapsed, expected_builds)
        route = MolecularRouteReceipt(self._qualification, selected, reason,
            expected_builds, elapsed, difference, tuple(samples), size, work, targets)
        # Publish only a complete, passing calibration. No partial winner.
        # The admitted finite set bounds receipt storage independent of run
        # length. Other signatures cannot borrow these measured timings.
        self._routes[targets] = route
        return route

    def build_channels(self, densities, *, targets):
        """Selected J/K channels for arbitrary real, even nonsymmetric D.

        Returns owned real J and K stacks plus a scalar receipt. Input
        indices and output ordering follow quartet_jk_batch. The error bound
        for each input applies to each requested J/K and excludes roundoff.
        """
        capture = self._begin_capture(targets)
        self.requirements(targets)
        if self._executor == 'auto':
            if self._route_request is not None:
                self._route_request(self, densities, targets)
            route = self._routes.get(targets, MolecularRouteReceipt(self._qualification,
                'ordered', 'no measured calibration for these channels'))
        else:
            route = MolecularRouteReceipt(self._qualification, self._executor, 'explicit executor')
        j, k, receipt = self._execute(densities, targets, route.selected)
        receipt = replace(receipt, route=route)
        old = self._totals
        work = QuartetWork(**{name: getattr(old.work, name)+getattr(receipt.work, name)
                              for name in QuartetWork.__dataclass_fields__})
        self._totals = MolecularJKTotals(old.builds+1, work,
            old.scalar_products+receipt.scalar_products,
            old.block_contractions+receipt.block_contractions,
            old.transport_scalar_products+receipt.transport_scalar_products,
            old.seconds+receipt.seconds,
            max(old.maximum_logical_bytes, receipt.logical_bytes))
        self._last_build = receipt
        if self._capture_terminal:
            self._capture(densities, j, k, receipt, capture)
        return j, k, receipt

    def _execute(self, densities, targets, executor):
        if not isinstance(densities, tuple) or len(densities) != len(targets):
            raise TypeError('densities must be a tuple matching the target channels')
        reservation = self._requirements_for(targets, executor)
        for density in densities:
            self._density(density)
        start = perf_counter()
        packed = np.empty((len(densities), self.n, self.n), np.complex128)
        for i, density in enumerate(densities):
            packed[i] = density
        if executor == 'ordered':
            result = ordered_quartet_jk(self._screening, packed,
                budget=self._budget, targets=targets)
        else:
            result = self._screening._plan.jk_batch(packed, budget=self._budget,
                executor=executor, targets=targets, screening=self._screening)
        if np.any(result.j.imag) or np.any(result.k.imag):
            raise ValueError('real molecular source returned a complex J/K')
        j, k = result.j.real.copy(), result.k.real.copy()
        receipt = MolecularJKReceipt(result.source_identity, targets,
            tuple(map(float, result.error_bounds)), result.work,
            result.scalar_products, result.block_contractions,
            result.transport_scalar_products, perf_counter()-start,
            reservation.maximum_bytes, executor=executor)
        return j, k, receipt

    def build_J(self, density):
        return self.build_channels((density,), targets=('j',))[0][0]

    def build_K(self, density):
        return self.build_channels((density,), targets=('k',))[1][0]

    def build_K_pair(self, alpha, beta):
        k = self.build_channels((alpha, beta), targets=('k', 'k'))[1]
        return k[0], k[1]

    def build_jk_uhf(self, alpha, beta):
        # Reserve all channels before even allocating the total density.
        # Spin channels can break the admitted spatial symmetry independently.
        targets = ('j', 'k', 'k')
        self.requirements(targets)
        self._density(alpha)
        self._density(beta)
        j, k, _ = self.build_channels((alpha+beta, alpha, beta), targets=targets)
        return j[0], k[0], k[1]

    def build_g_rhf(self, density, alpha_hf=1.):
        alpha_hf = float(alpha_hf)
        if not math.isfinite(alpha_hf):
            raise ValueError('nonfinite exact exchange fraction')
        if alpha_hf == 0.:
            return self.build_J(density)
        j, k, _ = self.build_channels((density,), targets=('jk',))
        return j[0]-.5*alpha_hf*k[0]

    def uses_runtime_schwarz_threshold(self):
        return False

    def set_schwarz_threshold(self, threshold):
        # Native hosts may call this even for stateless builders. The fixed
        # source is unchanged; uses_runtime_schwarz_threshold reports False.
        pass

    def set_incremental(self, enabled):
        pass

    def incremental_active(self):
        return False


@dataclass(frozen=True)
class MolecularHFEvaluation:
    build: FockBuild
    receipt: MolecularJKReceipt
    fock_error_bounds: tuple[float, ...]
    electronic_error_bound: float


class ScreenedMolecularHF:
    """Method-owned RHF/UHF formulas at one physical density epoch.

    Hcore and the supplied native-AO metric/retained frame are owned snapshots
    and bound to provenance. The caller supplies their physical meaning and
    ordering. This object neither certifies their geometry nor reduces their
    retained space. Dense FockBuild snapshots have no hard memory bound;
    their storage is reported independently of J/K execution admission.
    """

    def __init__(self, jk, *, frame, hcore, nuclear_energy):
        if type(jk) is ScreenedMolecularJK:
            provider, identity = 'screened-molecular-P0-HF-v2', jk.screening.source_identity
            if qualified_molecular_source(jk.screening).evidence == 'native-MD-constructor-and-full-numerical-audit':
                provider = 'screened-molecular-MD-HF-v1'
        else:
            from .symmetry_ri import MolecularRIJK
            if type(jk) is not MolecularRIJK:
                raise TypeError('an admitted ScreenedMolecularJK or MolecularRIJK is required')
            provider, identity = 'symmetry-molecular-RI-HF-v1', jk.source.source_identity
        if not isinstance(frame, OperatorFrame) or frame.n != jk.n or frame.overlap is None:
            raise ValueError('an explicit matching molecular AO frame is required')
        h = _snapshot(jk._density(hcore))
        _hermitian(h, 'Molecular core Hamiltonian')
        nuclear = float(nuclear_energy)
        if not math.isfinite(nuclear):
            raise ValueError('nonfinite nuclear energy')
        self._jk, self._frame, self._hcore, self._nuclear = jk, frame, h, nuclear
        # Qualification evidence and resource/cost policy live in receipts;
        # neither changes the Hamiltonian represented by this fixed mask.
        self._source = SourceProvenance.capture(provider,
            integral_source_identity=identity, frame=frame.identity,
            hcore_sha256=sha256(h.tobytes()).hexdigest(), nuclear_energy=nuclear,
            ao_ordering='caller-supplied native molecular AO',
            density_conventions={'RHF': 'total', 'UHF': 'per-spin'})

    @property
    def source(self):
        return self._source

    def evaluate(self, densities, *, method, generation, reuse_evaluated=False):
        """Evaluate full physical Focks; no symmetry projection or DIIS."""
        if type(reuse_evaluated) is not bool:
            raise TypeError('reuse_evaluated must be a boolean')
        generation = _integer(generation, 'generation')
        if method not in ('RHF', 'UHF'):
            raise ValueError('only molecular RHF and UHF are admitted')
        count = 1 if method == 'RHF' else 2
        if not isinstance(densities, tuple) or len(densities) != count:
            raise ValueError('RHF requires total D; UHF requires alpha and beta D')
        targets = ('jk',) if count == 1 else ('j', 'k', 'k')
        self._jk.requirements(targets)
        d = tuple(_snapshot(self._jk._density(a)) for a in densities)
        for a in d:
            _hermitian(a, 'HF density')
        total = d[0] if count == 1 else d[0]+d[1]
        batch = d if count == 1 else (total, *d)
        previous = (self._jk.reuse_evaluated_channels(batch, targets=targets)
                    if reuse_evaluated else None)
        j, k, receipt = (self._jk.build_channels(batch, targets=targets)
                         if previous is None else previous)
        exchange = .5 if count == 1 else 1.
        focks = tuple(self._hcore+j[0]-exchange*ks for ks in k)
        trace = lambda a, b: float(np.einsum('pq,qp->', a, b))
        components = (('one_electron', trace(total, self._hcore)),
                      ('hartree', .5*trace(total, j[0])),
                      ('exchange', -.5*exchange*sum(trace(a, b) for a, b in zip(d, k))))
        energy = EnergyComponents(electronic=math.fsum(v for _, v in components),
                                  nuclear=self._nuclear, components=components)
        build = FockBuild.from_dense(method=method, generation=generation,
            source=self._source, energy=energy, densities=d, focks=focks,
            frames=[self._frame]*count, spins=[None] if count == 1 else ['alpha', 'beta'],
            kpoints=[None]*count, weights=[1.]*count)
        bounds = receipt.density_error_bounds
        fock_bounds = (1.5*bounds[0],) if count == 1 else tuple(bounds[0]+b for b in bounds[1:])
        energy_terms = ([_bound_product(_norm(d[0]), fock_bounds[0])] if count == 1 else
                        [_bound_product(_norm(total), bounds[0])]+
                        [_bound_product(_norm(a), b) for a, b in zip(d, bounds[1:])])
        energy_bound = _bound_product(.5, sum(energy_terms))
        if not all(math.isfinite(b) for b in (*fock_bounds, energy_bound)):
            raise ValueError('nonrepresentable HF error bound')
        return MolecularHFEvaluation(build, receipt, fock_bounds, energy_bound)
