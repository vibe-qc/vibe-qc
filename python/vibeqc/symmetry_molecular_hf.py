"""Opt-in molecular HF consumer of an audited, fixed quartet mask.

The Dupuis-King shell transport and finite-source certificate live in the
shared quartet layer. This method adapter supplies real native JKBuilder
calls and HF energy conventions, without selecting an electronic irrep,
projecting a density, changing SCF convergence, or admitting derivatives.
"""
from dataclasses import dataclass
from hashlib import sha256
import math
from time import perf_counter

import numpy as np

from . import _vibeqc_core as core
from .fock_state import EnergyComponents, FockBuild, SourceProvenance
from .solvers.operators import OperatorFrame, _hermitian, _snapshot
from .symmetry_shared import Budget, _integer
from .symmetry_quartets import QuartetWork, _norm
from .symmetry_quartets_batch import quartet_batch_requirements
from .symmetry_quartets_screening import QuartetScreening


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


@dataclass(frozen=True)
class MolecularJKTotals:
    builds: int = 0
    work: QuartetWork = QuartetWork()
    scalar_products: int = 0
    block_contractions: int = 0
    transport_scalar_products: int = 0
    seconds: float = 0.
    maximum_logical_bytes: int = 0


class ScreenedMolecularJK(core.JKBuilder):
    """Real AO bridge to native RHF/UHF for one immutable P0 certificate.

    Stored and compact streamed complete-audit masks share this boundary.

    Pass explicitly to ``run_rhf_scf_with_jk`` or ``run_uhf_scf_with_jk``.
    RHF fuses J and K in one representative traversal. The current native
    UHF driver calls J, K-alpha and K-beta separately; no three-channel
    savings are claimed for that driver. ``build_channels`` can fuse them
    for a method consumer. No history of density or output matrices is kept.

    The reservation covers the live quartet plan and this contraction's
    logical buffers, not the surrounding SCF driver or FockBuild snapshots.
    Dynamic Schwarz thresholds and incremental state are deliberately
    inactive: the admitted density-independent mask remains fixed.
    """

    def __init__(self, screening, *, budget: Budget, executor='native'):
        super().__init__()
        if type(screening) is not QuartetScreening:
            raise TypeError('an immutable QuartetScreening certificate is required')
        screening._validate(screening._plan)
        self._screening = screening
        self._budget = budget
        self._executor = executor
        self._n = int(screening._plan.offsets[-1])
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

    def requirements(self, targets):
        """Count-only reservation, before density inspection or tile reads."""
        base = quartet_batch_requirements(self._screening._plan, len(targets),
            targets=targets, budget=self._budget, executor=self._executor,
            screening=self._screening)
        # Real inputs, complex packing, real exports and RHF fusion scratch.
        # Kernel inputs/exports are also reserved by base; double counting
        # here intentionally bounds overlapping bridge lifetimes.
        count = len(targets)
        outputs = sum(len(t) for t in targets)
        size = base.maximum_bytes + 65536 + 32*(count+outputs+3)*self.n**2
        work = base.maximum_work + 32*(count+outputs+3)*self.n**2
        self._budget.admit(size, work)
        return Budget(size, work)

    def _density(self, value):
        # Never let NumPy or Eigen discard an imaginary part implicitly.
        if not isinstance(value, np.ndarray) or value.dtype != np.float64:
            raise TypeError('native molecular HF requires real float64 AO densities')
        if value.shape != (self.n, self.n) or not np.isfinite(value).all():
            raise ValueError('invalid molecular AO density shape or values')
        return value

    def build_channels(self, densities, *, targets):
        """Selected J/K channels for arbitrary real, even nonsymmetric D.

        Returns owned real J and K stacks plus a scalar receipt. Input
        indices and output ordering follow quartet_jk_batch. The error bound
        for each input applies to each requested J/K and excludes roundoff.
        """
        if not isinstance(densities, tuple) or len(densities) != len(targets):
            raise TypeError('densities must be a tuple matching the target channels')
        reservation = self.requirements(targets)
        for density in densities:
            self._density(density)
        start = perf_counter()
        packed = np.empty((len(densities), self.n, self.n), np.complex128)
        for i, density in enumerate(densities):
            packed[i] = density
        result = self._screening._plan.jk_batch(packed, budget=self._budget,
            executor=self._executor, targets=targets, screening=self._screening)
        if np.any(result.j.imag) or np.any(result.k.imag):
            raise ValueError('real molecular source returned a complex J/K')
        j, k = result.j.real.copy(), result.k.real.copy()
        receipt = MolecularJKReceipt(result.source_identity, targets,
            tuple(map(float, result.error_bounds)), result.work,
            result.scalar_products, result.block_contractions,
            result.transport_scalar_products, perf_counter()-start,
            reservation.maximum_bytes)
        old = self._totals
        work = QuartetWork(**{name: getattr(old.work, name)+getattr(result.work, name)
                              for name in QuartetWork.__dataclass_fields__})
        self._totals = MolecularJKTotals(old.builds+1, work,
            old.scalar_products+receipt.scalar_products,
            old.block_contractions+receipt.block_contractions,
            old.transport_scalar_products+receipt.transport_scalar_products,
            old.seconds+receipt.seconds,
            max(old.maximum_logical_bytes, receipt.logical_bytes))
        self._last_build = receipt
        return j, k, receipt

    def build_J(self, density):
        return self.build_channels((density,), targets=('j',))[0][0]

    def build_K(self, density):
        return self.build_channels((density,), targets=('k',))[1][0]

    def build_K_pair(self, alpha, beta):
        k = self.build_channels((alpha, beta), targets=('k', 'k'))[1]
        return k[0], k[1]

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

    def reset_state(self):
        # There is no incremental numerical state. Preserve work accounting.
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
        if type(jk) is not ScreenedMolecularJK:
            raise TypeError('an admitted ScreenedMolecularJK is required')
        if not isinstance(frame, OperatorFrame) or frame.n != jk.n or frame.overlap is None:
            raise ValueError('an explicit matching molecular AO frame is required')
        h = _snapshot(jk._density(hcore))
        _hermitian(h, 'Molecular core Hamiltonian')
        nuclear = float(nuclear_energy)
        if not math.isfinite(nuclear):
            raise ValueError('nonfinite nuclear energy')
        self._jk, self._frame, self._hcore, self._nuclear = jk, frame, h, nuclear
        self._source = SourceProvenance.capture('screened-molecular-P0-HF-v1',
            screening=jk.screening.receipt, frame=frame.identity,
            hcore_sha256=sha256(h.tobytes()).hexdigest(), nuclear_energy=nuclear,
            ao_ordering='caller-supplied native molecular AO',
            density_conventions={'RHF': 'total', 'UHF': 'per-spin'})

    @property
    def source(self):
        return self._source

    def evaluate(self, densities, *, method, generation):
        """Evaluate full physical Focks; no symmetry projection or DIIS."""
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
        j, k, receipt = self._jk.build_channels(batch, targets=targets)
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
