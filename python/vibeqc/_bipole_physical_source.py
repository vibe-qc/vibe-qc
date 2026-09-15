"""Private basis-bound physical domains over one finite BIPOLE declaration.

This is bounded geometry/source orchestration, not a production HF driver.
An empty native source authenticates the common basis, lattice, mesh and
kernel; every selected AO quartet obtains its centers from that same basis.
Domains use the exact physical-support builders and native product sources.
No integral contraction or numerical symmetry reduction is done in Python.

The owner stores O(Nao) geometry, with no pair/quartet cache. Descriptor walks
have explicit work admission. Domain budgets inventory the owner plus ONE
materialized domain and its construction peak. Retaining multiple returned
domains, numerical panels or concurrent workers needs additional caller
inventory. Logical bytes/work exclude Python allocator overhead and RSS.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np

from . import _vibeqc_core as core
from ._bipole_physical_support import (
    _positive_integer, _rational, _inverse,
    _POLICY, _PRODUCT_POLICY,
    plan_product_support, build_product_support,
    plan_quartet_support, build_quartet_support,
)
from .symmetry_shared import Budget

_FIXED = 262144
_CAP_FIELDS = ('maximum_kpoints', 'maximum_images', 'maximum_cells',
               'maximum_context_storage_bytes', 'maximum_borrowed_numerical_bytes',
               'maximum_work_units')
_POLICY_ID = b'vibeqc.bipole.physical-source/basis-AO-lexicographic-quartets-v1'


def _record(cls, **fields):
    result = object.__new__(cls)
    for name, value in fields.items():
        object.__setattr__(result, name, value)
    return result


def _freeze(array):
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _budget(budget):
    if not isinstance(budget, Budget):
        raise TypeError('physical source requires an explicit shared Budget')
    return budget


def _index(index, limit, name, *, endpoint=False):
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise TypeError(f'{name} must be an integer')
    index = int(index)
    if index < 0 or index >= limit + int(endpoint):
        raise ValueError(f'{name} outside declared range')
    return index


def _copy_control(value, expected):
    """Snapshot exact native control records, including their nested caps."""
    if type(value) is not expected:
        raise TypeError(f'physical J/K requires {expected.__name__}')
    result = expected()
    for name, descriptor in expected.__dict__.items():
        if isinstance(descriptor, property) and descriptor.fset is not None:
            template = getattr(result, name)
            item = getattr(value, name)
            setattr(result, name, item if type(template) is int
                    else _copy_control(item, type(template)))
    return result


def _action_controls(inventory, panel_caps, stream_caps):
    inventory = _copy_control(inventory, core._BipoleFinitePanelInventory)
    panel_caps = _copy_control(panel_caps, core._BipoleFinitePanelCaps)
    stream_caps = _copy_control(stream_caps, core._BipoleProductJKStreamCaps)
    # Stay within the native stream binding's incoming-panel hard limits.
    # Narrow the producer itself so rejection occurs before integral work,
    # not only when the completed panel reaches stream.consume().
    if (stream_caps.maximum_panel_inventoried_bytes > 512 << 20
            or stream_caps.maximum_panel_work_units > 10**9):
        raise ValueError('physical J/K panel reservation exceeds native diagnostic bounds')
    panel_caps.maximum_node_bytes = min(panel_caps.maximum_node_bytes,
                                        stream_caps.maximum_panel_inventoried_bytes)
    panel_caps.maximum_work_units = min(panel_caps.maximum_work_units,
                                        stream_caps.maximum_panel_work_units)
    return inventory, panel_caps, stream_caps


def _selection(owner, target, quartet, k, exchange):
    s = core._BipoleEwaldGramSelection()
    s.q_index = owner.declaration.mesh.transfer_index(k, target) if exchange else 0
    s.left_k_index, s.right_k_index = (k if exchange else target), k
    a, b, c, d = quartet
    s.left_pair_begin, s.right_pair_begin = a*owner.memory.n_basis+b, c*owner.memory.n_basis+d
    s.left_pair_count = s.right_pair_count = 1
    return s


def _panel_args(owner, domain, selection, inventory, caps):
    return (domain.native_source, owner.basis, domain.images, domain.left_cells,
            domain.right_cells, selection, inventory, caps)


@dataclass(frozen=True, slots=True)
class PhysicalSourcePlan:
    n_basis: int
    quartet_count: int
    resident_bytes: int
    inventoried_bytes: int
    work_units: int


@dataclass(frozen=True, slots=True)
class PhysicalDomainPlan:
    quartet: tuple[int, int, int, int]
    image_count: int
    left_cell_count: int
    right_cell_count: int
    resident_bytes: int
    inventoried_bytes: int
    work_units: int


@dataclass(frozen=True, slots=True)
class PhysicalJKPlan:
    """Complete two-pass walk envelope; logical bytes/work, not Python RSS.

    One target k and the complete density grid. Geometry is counted twice
    for the admission walk and execution replay. The native stream reserves
    the numerical child envelopes and its density/state storage separately.
    No per-quartet receipts or numerical values are retained by this plan.
    """
    target_k_index: int
    quartet_count: int
    panel_calls: int
    inventoried_bytes: int
    work_units: int
    maximum_domain_bytes: int
    geometry_work_units: int
    domain_walk_sha256: str


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PhysicalDomain:
    """One complete AO quartet, retaining its common owner and exact labels.

    ``native_source`` binds the numerical labels and selected AO rows.
    ``domain_identity_sha256`` additionally binds their common physical policy.
    Numerical calls still pass the labels to the native verifier. Use
    owner.basis; changing any supplied payload is rejected by the native source.
    """
    owner: PhysicalSource
    memory: PhysicalDomainPlan
    images: np.ndarray
    left_cells: np.ndarray
    right_cells: np.ndarray
    native_source: object
    domain_identity_sha256: str

    def __init__(self, *args, **kwargs):
        raise TypeError('use PhysicalSource.build_domain')

    @property
    def physical_hamiltonian_certified(self):
        return False

    @property
    def symmetry_certified(self):
        return False


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PhysicalSource:
    """One immutable finite electron-electron policy, with on-demand domains.

    Only the native diagnostic size limits are supported. This authenticates
    basis-center ownership and uniform cutoffs, not completeness of the HF
    one-electron/nuclear operator, convergence of cutoffs or a space group.
    ``quartet_at`` uses full ordered AO indices, including crossed exchange
    quartets. There is no extra output-pair mask or symmetry skipping.
    """
    basis: object
    declaration: object
    ao_centers: np.ndarray
    lattice: np.ndarray
    pair_radius: float
    midpoint_radius: float
    maximum_candidates: int
    maximum_cells: int
    maximum_images: int
    memory: PhysicalSourcePlan
    source_identity_sha256: str
    _source_caps: tuple[int, ...]

    def __init__(self, *args, **kwargs):
        raise TypeError('use make_physical_source')

    @property
    def physical_hamiltonian_certified(self):
        return False

    @property
    def symmetry_certified(self):
        return False

    def quartet_at(self, index):
        index = _index(index, self.memory.quartet_count, 'quartet index')
        n = self.memory.n_basis
        d, index = index % n, index // n
        c, index = index % n, index // n
        return index // n, index % n, c, d

    def iter_quartets(self, *, budget, start=0, stop=None):
        """Admit a restartable descriptor range without allocating an Nao**4 list.

        Admission covers descriptor work only, not constructing/evaluating
        domains. Each build_domain call needs its own explicit budget.
        """
        _budget(budget)
        stop = self.memory.quartet_count if stop is None else stop
        start = _index(start, self.memory.quartet_count, 'start', endpoint=True)
        stop = _index(stop, self.memory.quartet_count, 'stop', endpoint=True)
        if start > stop:
            raise ValueError('quartet descriptor range is reversed')
        budget.admit(self.memory.resident_bytes + 4096, 65536 + 256*(stop-start))
        return (self.quartet_at(index) for index in range(start, stop))

    def plan_jk_stream(self, density, target_k_index, *, inventory, caps, budget):
        """Reserve a complete native panel-reduction envelope plus this owner.

        Numerical panel work/storage is reserved by the stream's caps. Domain
        generation and any additional live geometry still require admission
        through plan_domain/build_domain; this is not a production HF driver.
        """
        _budget(budget)
        plan = core._plan_bipole_product_jk_stream(
            self.declaration, density, target_k_index, inventory, caps)
        budget.admit(plan.required_node_inventoried_bytes
                     + inventory.numerical_replicas*self.memory.resident_bytes,
                     plan.work_units_upper_bound)
        return plan

    def start_jk_stream(self, density, target_k_index, *, inventory, caps, budget):
        """Start native density contraction with this owner's declared policy.

        The stream owns a density snapshot, requests each singleton panel in
        native schedule order, and cannot expose an incomplete result. Native
        validation binds common numerical inputs and repeated product cells;
        the physical generating-policy hash remains a declared qualification.
        """
        self.plan_jk_stream(density, target_k_index, inventory=inventory, caps=caps, budget=budget)
        return core._make_bipole_product_jk_stream(
            self.declaration, density, target_k_index, self.source_identity_sha256,
            inventory, caps)

    def plan_jk(self, density, target_k_index, *, inventory, panel_caps, stream_caps, budget):
        """Admit every physical domain and native panel before any integrals.

        The first pass materializes one domain at a time for native preflight.
        Density shape/storage is checked first, but its values are not scanned
        or copied here. A late-domain failure cannot leave a partial J/K action.
        """
        _budget(budget).admit(_FIXED, 1)
        controls = _action_controls(inventory, panel_caps, stream_caps)
        return self._plan_jk(density, target_k_index, *controls, budget)

    def _plan_jk(self, density, target, inventory, panel_caps, stream_caps, budget):
        target = _index(target, len(self.declaration.mesh), 'target k index')
        native = core._plan_bipole_product_jk_stream(
            self.declaration, density, target, inventory, stream_caps)
        replicas = inventory.numerical_replicas
        base = native.required_node_inventoried_bytes + replicas*_FIXED
        work = native.work_units_upper_bound + 4096*native.panel_calls + 1024*native.quartet_count
        peak, geometry = self.memory.resident_bytes, 0
        budget.admit(base + replicas*peak, work + 1)
        digest = hashlib.sha256(bytes.fromhex(self.source_identity_sha256))
        for ordinal in range(native.quartet_count):
            # Reserve both construction passes cumulatively. Budget is a
            # ceiling, not a mutable debit counter; each child gets only the
            # still-uncommitted allowance before it can allocate/traverse.
            domain_budget = Budget((budget.maximum_bytes-base)//replicas,
                                   (budget.maximum_work-work)//2)
            domain = self.build_domain(self.quartet_at(ordinal), budget=domain_budget)
            peak = max(peak, domain.memory.inventoried_bytes)
            geometry += domain.memory.work_units
            work += 2*domain.memory.work_units
            budget.admit(base + replicas*peak, work)
            digest.update(bytes.fromhex(domain.domain_identity_sha256))
            for k in range(native.n_kpoints):
                for exchange in (False, True):
                    selection = _selection(self, target, domain.memory.quartet, k, exchange)
                    panel = core._plan_bipole_finite_product_panel(
                        *_panel_args(self, domain, selection, inventory, panel_caps))
                    if (max(panel.required_node_inventoried_bytes, panel.per_replica_inventoried_bytes)
                            > stream_caps.maximum_panel_inventoried_bytes
                            or panel.work_units_upper_bound > stream_caps.maximum_panel_work_units):
                        raise ValueError('physical J/K panel exceeds the stream reservation')
                    del panel
            # Release before constructing the next domain (assignment alone
            # would keep the old domain alive throughout the next RHS call).
            del domain
        return PhysicalJKPlan(target, native.quartet_count, native.panel_calls,
                              base + replicas*peak, work, peak, geometry, digest.hexdigest())

    def contract_jk(self, density, target_k_index, *, inventory, panel_caps, stream_caps, budget):
        """Produce and contract one complete finite physical J/K action.

        All domains and numerical panels are preflighted before the native
        density snapshot or the first integral. Execution uses one domain and
        one singleton panel, with numerical contractions exclusively in C++.
        Only a finalized native result escapes. This private finite operator
        still has no production Hamiltonian or symmetry certificate.
        """
        _budget(budget).admit(_FIXED, 1)
        inventory, panel_caps, stream_caps = _action_controls(inventory, panel_caps, stream_caps)
        plan = self._plan_jk(density, target_k_index, inventory, panel_caps, stream_caps, budget)
        stream = core._make_bipole_product_jk_stream(
            self.declaration, density, plan.target_k_index, self.source_identity_sha256,
            inventory, stream_caps)
        digest = hashlib.sha256(bytes.fromhex(self.source_identity_sha256))
        geometry = 0
        for ordinal in range(plan.quartet_count):
            domain = self.build_domain(self.quartet_at(ordinal), budget=Budget(
                plan.maximum_domain_bytes, plan.geometry_work_units-geometry))
            geometry += domain.memory.work_units
            digest.update(bytes.fromhex(domain.domain_identity_sha256))
            for _ in range(2*stream.memory.n_kpoints):
                panel = core._make_bipole_finite_product_panel(*_panel_args(
                    self, domain, stream.next_selection(), inventory, panel_caps))
                stream.consume(panel)
                del panel
            del domain
        if geometry != plan.geometry_work_units or digest.hexdigest() != plan.domain_walk_sha256:
            raise RuntimeError('physical J/K domain replay differs from admitted walk')
        stream.finalize()
        return stream

    def plan_domain(self, quartet, *, budget):
        """Exact geometric census plus native source preflight, with no integrals.

        Small labels are materialized after geometric admission so native
        planning sees their real shapes. They are discarded on return.
        """
        return self._domain(quartet, budget, plan_only=True)

    def build_domain(self, quartet, *, budget):
        return self._domain(quartet, budget, plan_only=False)

    def _domain(self, quartet, budget, *, plan_only):
        _budget(budget)
        if not isinstance(quartet, tuple) or len(quartet) != 4:
            raise TypeError('quartet must be a tuple of four AO indices')
        quartet = tuple(_index(x, self.memory.n_basis, 'AO index') for x in quartet)
        base = self.memory.resident_bytes + 4096
        budget.admit(base + 1, 1)
        available = Budget(budget.maximum_bytes-base, budget.maximum_work)
        xyz = np.ascontiguousarray(self.ao_centers[list(quartet)])
        product_kw = dict(pair_radius=self.pair_radius, maximum_candidates=self.maximum_candidates,
                          maximum_cells=self.maximum_cells)
        quartet_kw = dict(pair_radius=self.pair_radius, midpoint_radius=self.midpoint_radius,
                         maximum_candidates=self.maximum_candidates, maximum_images=self.maximum_images)
        # Each child's conservative two-pass envelope is counted twice for
        # the count pass followed by build/replay. Admit cumulative work
        # before starting the next child; do not treat Budget as a debit counter.
        geometries = []
        peak, work = base, 0
        for fn, centers, kw in ((plan_product_support, xyz[:2], product_kw),
                                (plan_product_support, xyz[2:], product_kw),
                                (plan_quartet_support, xyz, quartet_kw)):
            budget.admit(peak + 1, work + 1)
            remaining = Budget(budget.maximum_bytes-peak, budget.maximum_work-work)
            p = fn(self.lattice, centers, budget=remaining, **kw)
            peak += p.inventoried_bytes
            work += 2*p.work_units
            budget.admit(peak, work)
            geometries.append(p)
        left = build_product_support(self.lattice, xyz[:2], budget=available, **product_kw)
        right = build_product_support(self.lattice, xyz[2:], budget=available, **product_kw)
        sr = build_quartet_support(self.lattice, xyz, budget=available, **quartet_kw)
        if [left.memory, right.memory, sr.memory] != geometries:
            raise RuntimeError('physical domain replay disagrees with admitted geometry')
        caps = core._BipoleFiniteSourceCaps()
        for name, value in zip(_CAP_FIELDS, self._source_caps):
            setattr(caps, name, value)
        domain = core._BipoleFiniteProductDomain()
        domain.left_pair_begin = quartet[0]*self.memory.n_basis + quartet[1]
        domain.right_pair_begin = quartet[2]*self.memory.n_basis + quartet[3]
        domain.left_pair_count = domain.right_pair_count = 1
        args = (self.declaration.mesh, sr.images, left.cells, right.cells,
                domain, self.declaration.options, caps)
        native = core._plan_bipole_finite_product_source(self.basis, *args)
        # Include source validation and repeated native wrapper preflights.
        peak += native.context_storage_bytes + native.borrowed_numerical_bytes
        work += 4*native.work_units_upper_bound
        budget.admit(peak, work)
        resident = (base + sr.images.nbytes + left.cells.nbytes + right.cells.nbytes
                    + native.context_storage_bytes)
        plan = PhysicalDomainPlan(quartet, len(sr.images), len(left.cells), len(right.cells),
                                  resident, peak, work)
        if plan_only:
            return plan
        system = core.PeriodicSystem()
        system.lattice = self.lattice
        source = core._make_bipole_finite_product_source(self.basis, system, *args)
        digest = hashlib.sha256(_POLICY_ID + b'/domain')
        digest.update(bytes.fromhex(self.source_identity_sha256))
        digest.update(bytes.fromhex(source.source_identity_sha256))
        digest.update(struct.pack('>4Q', *quartet))
        return _record(PhysicalDomain, owner=self, memory=plan, images=sr.images,
                       left_cells=left.cells, right_cells=right.cells, native_source=source,
                       domain_identity_sha256=digest.hexdigest())


def make_physical_source(basis, system, mesh, options, source_caps, *, pair_radius,
                         midpoint_radius, maximum_candidates, maximum_cells,
                         maximum_images, maximum_quartets, budget):
    """Bind one finite policy to the actual native basis before any domain walk.

    An empty native declaration authenticates basis/lattice/mesh/kernel data.
    ShellInfo copies and AO-center expansion are admitted from its native
    basis census first. Native BasisSet has no mutating Python interface;
    system/options/caps are snapshotted and not retained from the caller.
    No atom ordering or caller-supplied center array is used as an AO map.
    """
    _budget(budget).admit(_FIXED, 65536)
    for radius in (pair_radius, midpoint_radius):
        if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, float, np.integer, np.floating)):
            raise TypeError('physical source radii must be real numbers')
    rp, rm = _rational(pair_radius), _rational(midpoint_radius)
    if rp < 0 or rm < 0:
        raise ValueError('physical source radii must be nonnegative')
    maximum_candidates = _positive_integer(maximum_candidates, 'maximum_candidates')
    maximum_cells = _positive_integer(maximum_cells, 'maximum_cells')
    maximum_images = _positive_integer(maximum_images, 'maximum_images')
    maximum_quartets = _positive_integer(maximum_quartets, 'maximum_quartets')
    if not isinstance(basis, core.BasisSet):
        raise TypeError('physical source requires a native BasisSet')
    n = int(basis.nbasis)
    if n == 0 or n**4 > maximum_quartets:
        raise ValueError('physical source complete quartet count exceeds cap or is empty')
    if options.require_image_permutation_closure or options.require_cell_inversion_closure:
        raise ValueError('physical pair domains require cross-domain closure, not common-list closure flags')
    # Existing private native bindings admit at most 64 images/cells per call.
    if maximum_cells > min(64, source_caps.maximum_cells) or maximum_images > min(64, source_caps.maximum_images):
        raise ValueError('physical support caps exceed native source diagnostic caps')
    images, cells = np.empty((0,3,3), np.int64), np.empty((0,3), np.int64)
    native = core._plan_bipole_finite_source(basis, mesh, images, cells, options, source_caps)
    resident = _FIXED + native.context_storage_bytes + native.borrowed_numerical_bytes + 24*n + 72
    # Flattened ShellInfo can repeat primitive arrays for each contraction.
    # There are at most Nao contractions; count copies conservatively before
    # calling basis.shells(). Geometry freezing temporarily doubles its arrays.
    peak = resident + 2*n*native.borrowed_basis_numeric_bytes + 1024*n + 48*n + 144
    work = 4*native.work_units_upper_bound + 65536*(n+1)
    budget.admit(peak, work)
    declaration = core._make_bipole_finite_source(basis, system, mesh, images, cells, options, source_caps)
    centers = np.empty((n,3), float)
    offset = 0
    for shell in basis.shells():
        width = 2*shell.l+1 if shell.pure else (shell.l+1)*(shell.l+2)//2
        if width <= 0 or offset+width > n:
            raise ValueError('native shell AO enumeration disagrees with basis size')
        center = tuple(_rational(x) for x in shell.origin)
        centers[offset:offset+width] = tuple(float(x) for x in center)
        offset += width
    if offset != n:
        raise ValueError('native shell AO enumeration does not cover the basis')
    lattice = np.ascontiguousarray(declaration.direct_lattice)
    _inverse(tuple(tuple(_rational(x) for x in row) for row in lattice))
    digest = hashlib.sha256(_POLICY_ID + _POLICY + _PRODUCT_POLICY)
    digest.update(bytes.fromhex(declaration.source_identity_sha256))
    digest.update(struct.pack('>2dQ', float(rp), float(rm), n))
    return _record(PhysicalSource, basis=basis, declaration=declaration,
                   ao_centers=_freeze(centers), lattice=_freeze(lattice),
                   pair_radius=float(rp), midpoint_radius=float(rm),
                   maximum_candidates=maximum_candidates, maximum_cells=maximum_cells,
                   maximum_images=maximum_images, memory=PhysicalSourcePlan(n,n**4,resident,peak,work),
                   source_identity_sha256=digest.hexdigest(),
                   _source_caps=tuple(getattr(source_caps, name) for name in _CAP_FIELDS))
