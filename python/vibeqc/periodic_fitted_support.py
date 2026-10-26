"""Finite fitted operators for experimental chi, below any SCF/correlation claim.

Classic reciprocal fitting and SR/LR fitting are distinct finite approximations.
All radii below are executed finite settings, not certified energy error bounds.
"""
from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
import json
import math
import numbers

import numpy as np


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class ChiCoulombControls:
    one_electron_cutoff_bohr: float
    pair_cutoff_bohr: float
    reciprocal_cutoff_ha: float
    nuclear_real_cutoff_bohr: float
    nuclear_reciprocal_cutoff_bohr_inv: float
    nuclear_alpha_bohr_inv: float
    auxiliary_cutoff_bohr: float | None = None
    omega_bohr_inv: float | None = None
    tail_reciprocal_cutoff_ha: float | None = None
    metric_threshold: float = 1e-9
    integral_screen_error: float = 0.0
    fit_screen_threshold: float = 0.0
    support_policy: str = 'physical_pairs'
    memory_byte_cap: int = 512 * 1024**2
    native_workspace_byte_cap: int = 64 * 1024**2
    image_candidate_cap: int = 2_000_000
    reciprocal_candidate_cap: int = 2_000_000
    support_record_byte_cap: int = 16 * 1024**2

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == 'support_policy':
                if value not in ('physical_pairs', 'historical_index_ball'):
                    raise ValueError('invalid fitted support policy')
            elif value is None:
                if key not in ('auxiliary_cutoff_bohr','omega_bohr_inv','tail_reciprocal_cutoff_ha'):
                    raise ValueError(f'{key} must not be None')
            else:
                if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
                    raise ValueError(f'{key} must be finite real')
                if key.endswith('_cap'):
                    if value != int(value) or value <= 0:
                        raise ValueError(f'{key} must be a positive integer')
                    object.__setattr__(self, key, int(value))
                elif key in ('metric_threshold', 'integral_screen_error', 'fit_screen_threshold'):
                    if value < 0:
                        raise ValueError(f'{key} must be nonnegative')
                elif value <= 0:
                    raise ValueError(f'{key} must be positive')

    def require_source(self, source):
        if source not in ('classic', 'sr'):
            raise ValueError('explicit Coulomb controls require fit_source classic or sr; auto is ambiguous')
        if source == 'sr':
            if self.support_policy != 'physical_pairs':
                raise NotImplementedError('SR/LR has physical capsule support; historical_index_ball is classic-only')
            if self.auxiliary_cutoff_bohr is None or self.omega_bohr_inv is None:
                raise ValueError('SR requires explicit auxiliary_cutoff_bohr and omega_bohr_inv')
            if self.tail_reciprocal_cutoff_ha is not None or self.fit_screen_threshold != 0:
                raise ValueError('SR uses its explicit reciprocal extent and integral_screen_error, not classic tail/fit screening')
        elif (self.auxiliary_cutoff_bohr is not None or self.omega_bohr_inv is not None
              or self.integral_screen_error != 0):
            raise ValueError('classic reciprocal fitting has no auxiliary real cutoff, split omega or SR screening')
        if self.tail_reciprocal_cutoff_ha is not None and self.tail_reciprocal_cutoff_ha < self.reciprocal_cutoff_ha:
            raise ValueError('tail cutoff must not be below the base reciprocal cutoff')
        return self

    def ladder(self, field, values):
        """Explicit single-axis refinements; no monotonic-energy/error claim."""
        allowed = {'one_electron_cutoff_bohr', 'pair_cutoff_bohr',
                   'reciprocal_cutoff_ha', 'tail_reciprocal_cutoff_ha',
                   'auxiliary_cutoff_bohr', 'metric_threshold',
                   'integral_screen_error', 'fit_screen_threshold',
                   'nuclear_real_cutoff_bohr', 'nuclear_reciprocal_cutoff_bohr_inv'}
        if field not in allowed or not values:
            raise ValueError('a nonempty single-axis numerical refinement is required')
        return tuple(replace(self, **{field: value}) for value in values)


class ChiFittedOperator(Mapping):
    """Immutable borrowed finite factors, with explicit reuse identity checks.

    J and K here omit p=0 and all exchange constants. SCF restores full/LR
    Madelung and the finite physical erfc/erf constant using its separately
    executed overlap. The numerical source split is never a physical range.
    """
    __slots__ = ('_factors', '_descriptor', '_weights', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('fitted operator identity and factors are immutable')
        object.__setattr__(self, name, value)

    def __init__(self, factors, descriptor, weights):
        frozen = {}
        for pair, value in factors.items():
            a = np.ascontiguousarray(value, dtype=np.complex128)
            if not np.isfinite(a).all():
                raise ValueError('nonfinite fitted factor')
            frozen[pair] = np.frombuffer(a.tobytes(), dtype=np.complex128).reshape(a.shape)
        self._factors = MappingProxyType(frozen)
        self._descriptor = descriptor
        self._weights = tuple(float(w) for w in weights)
        self._sealed = True

    @property
    def factors(self): return self._factors
    @property
    def descriptor(self): return self._descriptor
    @property
    def identity(self): return self.descriptor.identity
    @property
    def kernel(self):
        return dict(self.descriptor.numerical_support).get('physical_kernel', 'coulomb')
    @property
    def omega_screen(self):
        return dict(self.descriptor.numerical_support).get('omega_screen', 0.0)
    def __iter__(self): return iter(self._factors)
    def __len__(self): return len(self._factors)
    def __getitem__(self, pair): return self._factors[pair]

    def require_identity(self, identity):
        if identity != self.identity:
            raise ValueError('fitted cache belongs to a different source/support/frame')
        return self

    def as_dict(self):
        domains = json.loads(self.descriptor.executed_domains_json)
        return dict(source=self.descriptor.as_dict(), source_identity=self.identity,
                    executed_domain_identity=hashlib.sha256(self.descriptor.executed_domains_json.encode()).hexdigest(),
                    kernel=('periodic Coulomb 4pi/(V p^2)' if self.kernel == 'coulomb'
                            else domains['kernel']),
                    physical_kernel=self.kernel, omega_screen=self.omega_screen,
                    q0_convention=('p=0 omitted; K excludes separately applied BvK Ewald Madelung'
                                   if self.kernel == 'coulomb' else domains['q0']),
                    exchange_zero_mode=domains.get('exchange_zero_mode', dict(full_madelung_coefficient=1.)),
                    normalization='primitive_cell_bz_weights_sum_one',
                    weights=self._weights)

    def _density(self, densities):
        n = self[(0, 0)].shape[-1]
        d = np.asarray(densities)
        if d.shape != (len(self._weights), n, n) or not np.isfinite(d).all():
            raise ValueError('density must be finite (nk,nao,nao) on this source')
        if np.max(abs(d-d.conj().transpose(0, 2, 1))) > 1e-10:
            raise ValueError('fitted density must be Hermitian')
        return d

    def coulomb(self, densities):
        from .aux_basis import _build_coulomb_from_diagonal_factors
        d = self._density(densities)
        return np.asarray(_build_coulomb_from_diagonal_factors(
            [self[(i,i)] for i in range(len(d))], d, self._weights))

    def exchange(self, densities):
        from .periodic_k_gdf import _build_k_from_lpq_cache
        d = self._density(densities)
        return np.asarray(_build_k_from_lpq_cache(self, d, self._weights, nbasis=d.shape[-1]))


def build_chi_fitted_operator(system, basis, auxiliary, kpoints_cart, weights, *,
                              source, controls, kernel='coulomb', omega_screen=0.0,
                              _retained_bytes=0):
    """Build the selected finite source, with no automatic support enlargement.

    The budget includes factors/immutable copies plus one q build reservation;
    it is not a process RSS promise. A refusal never switches source or cutoff.
    ``kernel`` and ``omega_screen`` define the physical fitted interaction;
    ``controls.omega_bohr_inv`` is only the SR source's numerical split.
    Every factor omits p=0. Required finite erfc/erf and full/LR exchange
    constants are recorded for the caller's exact-overlap restoration.
    """
    from .aux_basis import (_canonical_reciprocal_transfer, _basis_fingerprint,
        _range_separated_gdf_source_signature, build_lpq_bloch_native_fft_shared_q,
        _build_lpq_range_separated_shared_q, _physical_fit_kernel)
    from .periodic_fit_source import RSGDFFitSource, factor_digest
    from ._vibeqc_core import LatticeSumOptions
    if not isinstance(controls, ChiCoulombControls):
        raise TypeError('controls must be ChiCoulombControls')
    controls.require_source(source)
    kernel, omega_screen = _physical_fit_kernel(kernel, omega_screen)
    kernel_options = {} if kernel == 'coulomb' else dict(kernel=kernel, omega_screen=omega_screen)
    from ._aopair_ft import _aopair_ft_backend
    if source == 'classic' and _aopair_ft_backend() not in ('auto', 'cxx'):
        raise NotImplementedError('chi fitted operator requires native pair Fourier transforms')
    if int(system.dim) != 3:
        raise NotImplementedError('fitted physical support requires 3D')
    k = np.asarray(kpoints_cart)
    w = np.asarray(weights)
    if (k.ndim != 2 or k.shape[1] != 3 or len(k) == 0 or np.iscomplexobj(k)
            or not np.isfinite(k).all() or w.shape != (len(k),) or np.iscomplexobj(w)
            or not np.isfinite(w).all() or np.any(w < 0) or abs(float(w.sum())-1) > 1e-12):
        raise ValueError('finite real kpoints and nonnegative normalized weights required')
    n, na, nk = int(basis.nbasis), int(auxiliary.nbasis), len(k)
    if min(n, na) < 1:
        raise ValueError('nonempty orbital and auxiliary bases required')
    reserve = 32*nk*nk*na*n*n + controls.support_record_byte_cap
    if isinstance(_retained_bytes, bool) or int(_retained_bytes) != _retained_bytes or _retained_bytes < 0:
        raise ValueError('retained source byte reservation must be a nonnegative integer')
    build_cap = int(controls.memory_byte_cap)-reserve-int(_retained_bytes)
    if build_cap <= controls.native_workspace_byte_cap:
        raise MemoryError('retained fitted factors/provenance exceed complete source budget')
    groups = {}
    for i in range(nk):
        for j in range(nk):
            q = _canonical_reciprocal_transfer(system, k[j]-k[i])
            groups.setdefault(tuple(np.round(q,14)), (q, []))[1].append((i,j))
    factors, records = {}, []
    lat = LatticeSumOptions(); lat.cutoff_bohr = controls.pair_cutoff_bohr
    lat.pair_complete_1e = controls.support_policy == 'physical_pairs'
    for key in sorted(groups):
        q, pairs = groups[key]
        bras = np.asarray([k[i] for i,j in pairs])
        if source == 'classic':
            # Exact finite q sphere may be split into base and complementary
            # tail; the real AO-product support stays fixed in both sweeps.
            record = []
            values = build_lpq_bloch_native_fft_shared_q(system,basis,auxiliary,bras,q,
                ke_cutoff=controls.reciprocal_cutoff_ha,
                tail_ke_cutoff=controls.tail_reciprocal_cutoff_ha,
                lat_opts=lat, linear_dep_thr=controls.metric_threshold,
                fit_screen_threshold=controls.fit_screen_threshold,
                canonical_auxiliary_basis=True, support_policy=controls.support_policy,
                _support_recorder=record.append,
                _support_record_byte_cap=controls.support_record_byte_cap,
                memory_byte_cap=build_cap,reciprocal_candidate_cap=controls.reciprocal_candidate_cap,
                image_candidate_cap=controls.image_candidate_cap,tail_chunk_g=256,
                **kernel_options)
            if len(record) != 1:
                raise RuntimeError('classic source did not report its executed domain')
            records.append(record[0])
        else:
            record = []
            batch = _build_lpq_range_separated_shared_q(system,basis,auxiliary,bras,q,
                omega=controls.omega_bohr_inv,pair_cutoff=controls.pair_cutoff_bohr,
                auxiliary_cutoff=controls.auxiliary_cutoff_bohr,
                ke_cutoff=controls.reciprocal_cutoff_ha,linear_dep_thr=controls.metric_threshold,
                memory_byte_cap=build_cap,native_workspace_byte_cap=controls.native_workspace_byte_cap,
                image_candidate_cap=controls.image_candidate_cap,
                reciprocal_candidate_cap=controls.reciprocal_candidate_cap,
                integral_screen_error=controls.integral_screen_error,canonical_auxiliary_basis=True,
                _support_record_byte_cap=controls.support_record_byte_cap,
                _support_recorder=record.append, **kernel_options)
            values=batch.factors;records.append(batch.executed_support)
        if not records[-1]:
            raise RuntimeError('missing executed fitted support record')
        if sum(len(r.encode()) for r in records) > controls.support_record_byte_cap:
            raise MemoryError('complete fitted support record exceeds its budget')
        factors.update(zip(pairs,values))
    # Preserve the historical full-Coulomb descriptor and arrays. Physical
    # kernels add a distinct immutable identity even at numerically equal fits.
    kernel_record = {} if kernel == 'coulomb' else dict(
        physical_kernel=kernel, omega_screen=omega_screen,
        exchange_zero_mode=dict(
            full_madelung_coefficient=1. if kernel == 'erf' else 0.,
            finite_constant_sign=-1. if kernel == 'erf' else 1.,
            finite_constant='pi/(V*omega_screen^2) times the equal-k weight',
            uniform_mesh_constant='pi/(V_BvK*omega_screen^2)',
            contraction='K_k += constant_k * S_k @ D_k @ S_k',
            included_in_factors=False),
        numerical_split_omega_bohr_inv=controls.omega_bohr_inv)
    domains=_json(dict(schema='vibeqc.chi.fitted-domains/v1', controls=asdict(controls),
        q_records=[json.loads(r) for r in records], weights=w.tolist(),
        kernel='periodic Coulomb' if kernel == 'coulomb' else 'periodic '+kernel,
        q0=('p=0 omitted; BvK exchange correction external' if kernel == 'coulomb' else
            'p=0 omitted in matched metric/three-center fit; physical finite constant and full/LR Madelung external'),
        normalization='primitive cell, normalized Brillouin-zone weights',
        maturity='experimental',qualification='D93 v1 fitted/post-HF not-qualified',
        **kernel_record))
    if len(domains.encode('utf-8')) > controls.support_record_byte_cap:
        raise MemoryError('complete fitted support wrapper exceeds its record budget')
    descriptor=RSGDFFitSource(
        engine='native-chi-'+source+'-finite-contract',engine_version=1,source=source,
        source_signature=_range_separated_gdf_source_signature(system,basis,auxiliary),
        orbital_basis=str(basis.name),orbital_basis_sha256=_basis_fingerprint(basis),
        auxiliary_basis=str(auxiliary.name),auxiliary_basis_sha256=_basis_fingerprint(auxiliary),
        numerical_support=tuple(sorted(dict(asdict(controls), **({} if kernel == 'coulomb' else
            dict(physical_kernel=kernel, omega_screen=omega_screen))).items())),metric_threshold=controls.metric_threshold,
        auxiliary_frame='canonical',q_grouping=tuple((tuple(q),tuple(pairs)) for q,pairs in groups.values()),
        kpoints_cart=tuple(map(tuple,k)),base_reciprocal_cutoff_ha=controls.reciprocal_cutoff_ha,
        executed_tail_reciprocal_cutoff_ha=controls.tail_reciprocal_cutoff_ha,
        factor_shapes=tuple((pair,a.shape) for pair,a in sorted(factors.items())),
        factor_sha256=factor_digest(factors),executed_domains_json=domains)
    return ChiFittedOperator(factors,descriptor,w)


def fitted_one_electron_options(base, controls):
    from .periodic_k_gdf import _clone_lattice_options
    out = _clone_lattice_options(base)
    out.cutoff_bohr = controls.one_electron_cutoff_bohr
    out.nuclear_cutoff_bohr = controls.nuclear_real_cutoff_bohr
    out.pair_complete_1e = controls.support_policy == 'physical_pairs'
    return out


def preflight_fitted_one_electron(system, basis, options, controls, *, n_kpoints=1, ecp=False, return_workspace=False):
    """Refuse complete enclosures/FT storage before enumeration or integration.

    Counts are conservative Cartesian boxes, never substituted physical domains.
    The reservation covers numerical arrays; library state and allocator overhead
    remain outside the explicitly documented process-memory scope.
    """
    a = np.asarray(system.lattice, dtype=float)
    origins = np.asarray([shell.origin for shell in basis.shells()], dtype=float)
    if (int(system.dim) != 3 or a.shape != (3, 3) or not np.isfinite(a).all()
            or origins.ndim != 2 or origins.shape[1] != 3 or not np.isfinite(origins).all()):
        raise ValueError('finite 3D lattice and nonempty normalized basis required')
    inverse = np.linalg.inv(a)
    def ball_bound(radius, inverse_lattice):
        return math.prod(2*math.ceil(radius*np.linalg.norm(row))+3 for row in inverse_lattice)
    span = float(np.linalg.norm(np.ptp(origins, axis=0))) if options.pair_complete_1e else 0.
    oneel_cells = ball_bound(float(options.cutoff_bohr)+span, inverse)
    pair_cells = ball_bound(controls.pair_cutoff_bohr+span, inverse)
    nuclear_cells = ball_bound(controls.nuclear_real_cutoff_bohr, inverse)
    real_count = len(system.unit_cell)**2 * nuclear_cells
    reciprocal_count = ball_bound(controls.nuclear_reciprocal_cutoff_bohr_inv,a.T/(2*np.pi))
    if max(oneel_cells,pair_cells,nuclear_cells,real_count) > controls.image_candidate_cap:
        raise MemoryError('complete fitted one-electron/nuclear candidate enclosure exceeds image cap')
    if reciprocal_count > controls.reciprocal_candidate_cap:
        raise MemoryError('complete fitted nuclear reciprocal enclosure exceeds candidate cap')
    # Records include three operator-cell lists plus V_ne and NN vectors.
    record_bound = 256*(real_count+2*reciprocal_count+(4+2*int(ecp))*oneel_cells+len(origins)+len(system.unit_cell))
    if record_bound > controls.support_record_byte_cap:
        raise MemoryError('complete nuclear/electronic trace exceeds support record budget before integration')
    no = int(basis.nbasis)
    # Complete lattice outputs, native/Python block snapshots, Bloch matrices
    # and lossless support records remain resident. No AO-pair FT cache.
    retained_bytes = 96*oneel_cells*no**2 + 128*int(n_kpoints)*no**2 + controls.support_record_byte_cap
    if ecp:
        # Separate local/projector lattice blocks and four retained Bloch
        # components; split native calls are sequential, not replicated jobs.
        # Result attachment copies the four components while their assembly
        # arrays remain live, so reserve both copies at the peak.
        retained_bytes += 16*oneel_cells*no**2 + 128*int(n_kpoints)*no**2
    peak_bytes = retained_bytes
    peak_bytes += 128*(real_count+reciprocal_count+pair_cells+nuclear_cells)
    from ._vibeqc_core import get_num_threads
    # Each worker owns the shared selector's <=8 MiB cache (<=8 entries)
    # and one scratch source list; reserve the actual enclosure upper bound.
    source_list_bytes = 128*len(system.unit_cell)*nuclear_cells
    peak_bytes += max(1,int(get_num_threads()))*(min(8*1024**2,8*source_list_bytes)+source_list_bytes)
    peak_bytes += controls.native_workspace_byte_cap
    # The weighted native kernel visits shell pairs internally without a
    # shell-pair x G output. Its exact Hermite scratch scales with G tile width.
    # No optional radial/power cache or reciprocal-tail truncation is enabled.
    shells = basis.shells()
    lmax = max(int(getattr(sh, 'l', 0)) for sh in shells)
    if lmax > 6 or any(not getattr(sh, 'pure', True) and getattr(sh, 'l', 0) for sh in shells):
        raise NotImplementedError('bounded chi V_ne requires native pure shells through L=6')
    workers = max(1, int(get_num_threads()))
    powers = 2*lmax+1
    cart = (lmax+1)*(lmax+2)//2
    # E recurrences, real Cartesian accumulators, Hermite moments/flags,
    # copied shell primitives/offsets and cache descriptors (caches disabled).
    native_fixed = 65536 + 256*len(shells) + 64*sum(len(getattr(sh, 'exponents', ())) for sh in shells)
    native_fixed += workers*(65536 + 48*(lmax+1)**2*powers
                             + 16*cart**2 + 18*powers**2*(powers+1))
    native_per_g = 8 + 48*powers + 32*workers
    # One-cell accumulator, returned native block, correction/mask, assembly
    # temporaries; tiled nuclear phases and weight/kernel conversion copies.
    fixed = 128*no**2 + 128*no + 1024
    per_g = 128 + 64*len(system.unit_cell)
    available = controls.memory_byte_cap - peak_bytes - fixed
    native_available = controls.native_workspace_byte_cap - native_fixed
    tile = min(256, reciprocal_count, available//per_g, native_available//native_per_g)
    if tile < 1:
        raise MemoryError('bounded fitted V_ne Fourier workspace minimum exceeds numerical/native budget before integration')
    plan = dict(cell_tile=1, reciprocal_tile=int(tile), shell_policy='native disjoint shell-pair tasks',
        retained_bytes=int(retained_bytes), peak_accounted_bytes=int(peak_bytes+fixed+tile*per_g),
        minimum_required_bytes=int(peak_bytes+fixed+per_g),
        cell_enclosure=int(oneel_cells), reciprocal_enclosure=int(reciprocal_count),
        tile_payload_bytes=int(fixed+tile*per_g),
        native_workspace_bytes=int(native_fixed+tile*native_per_g),
        native_workspace_reserved_bytes=int(controls.native_workspace_byte_cap),
        removed_full_fourier_bytes=int(32*(oneel_cells+1)*no**2*reciprocal_count),
        cache_bytes=0, scope='numerical payload including complete outputs and block copies; excludes allocator, Python metadata, library state and downstream arrays')
    return plan if return_workspace else retained_bytes


def fitted_nuclear_operators(system, basis, options, overlap, controls):
    """Use one explicit Ewald convention for V_ne and actual E_nn traversal."""
    from . import _vibeqc_core as core
    from .pbc_bipole_common import _compute_nuclear_lattice_ewald_reciprocal_ft
    workspace = preflight_fitted_one_electron(system, basis, options, controls, return_workspace=True)
    eo = core.EwaldOptions()
    eo.alpha = controls.nuclear_alpha_bohr_inv
    eo.real_cutoff_bohr = controls.nuclear_real_cutoff_bohr
    eo.recip_cutoff_bohr_inv = controls.nuclear_reciprocal_cutoff_bohr_inv
    potential, reciprocal = _compute_nuclear_lattice_ewald_reciprocal_ft(
        basis, system, options, eo, overlap, precision=1e-12,
        K_max=controls.nuclear_reciprocal_cutoff_bohr_inv,
        resolved_erfc_options=options, bounded_workspace=workspace)
    from .output import write
    write('  chi V_ne workspace: ' + _json(reciprocal.workspace) + '\n')
    nuclear = core.ewald_nuclear_repulsion_with_support(system, eo)
    return potential, reciprocal, nuclear


def fitted_one_electron_record(system, basis, options, operators, reciprocal, nuclear, controls):
    def vectors(a): return np.asarray(a, dtype=float).reshape(-1, 3).tolist()
    record = dict(
        policy=controls.support_policy,
        lattice_bohr=vectors(system.lattice),
        shell_origins_bohr=vectors([s.origin for s in basis.shells()]),
        nuclei=[dict(Z=int(a.Z), xyz=list(a.xyz)) for a in system.unit_cell],
        operator_cells={key:[dict(index=list(map(int,c.index)),cart_bohr=list(c.r_cart))
                            for c in value.cells] for key,value in operators.items()},
        source_membership='lossless native selector specification before screening, not integral-call trace',
        nuclear_sources=dict(selector=('PairNuclearImageSelector AO midpoint' if options.pair_complete_1e
                                     else 'origin-centered direct_lattice_cells'),
                             cutoff_bohr=options.nuclear_cutoff_bohr),
        pair_cutoff_bohr=options.cutoff_bohr,
        v_ne_reciprocal_vectors=vectors(reciprocal.K_vectors),
        v_ne_pair_enclosure_cart=vectors(reciprocal.cells_r_cart),
        nuclear_repulsion=dict(energy_ha=float(nuclear.energy),alpha=float(nuclear.alpha),
            real_cutoff_bohr=float(nuclear.real_cutoff_bohr),
            reciprocal_cutoff_bohr_inv=float(nuclear.reciprocal_cutoff_bohr_inv),
            real_pair_displacements=[vectors(r) for r in nuclear.real_pair_displacements],
            reciprocal_vectors=vectors(nuclear.reciprocal_vectors),
            membership='actual centered Ewald traversal; ordered atom pairs, self omitted'),
        normalization='energies per primitive cell',
        screening=dict(schwarz_threshold=options.schwarz_threshold,
                       overlap_threshold=options.screening_overlap_threshold))
    payload = _json(record)
    if len(payload.encode()) > controls.support_record_byte_cap:
        raise MemoryError('executed nuclear/electronic support record exceeds budget')
    return payload
