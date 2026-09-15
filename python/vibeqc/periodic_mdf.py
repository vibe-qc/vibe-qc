"""Private common-source MDF implementation, pending numerical acceptance.

This module is absent from public SCF selectors and package exports. A typed
private SCF injection supports future common-source acceptance calculations.
The auxiliary basis is supplied explicitly in native normalization; no hidden
compensation transform, metric threshold floor, or zero-mode repair is applied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class _MdfFitState:
    three_center: np.ndarray
    eigenvectors: np.ndarray
    keep: np.ndarray
    lr_vectors: np.ndarray
    pw_vectors: np.ndarray
    pw_factors: np.ndarray
    ket_kpoints: np.ndarray
    source_key: tuple
    image_candidate_cap: int


@dataclass(frozen=True)
class _MdfBatch:
    factors: np.ndarray
    transfer: np.ndarray
    metric_eigenvalues: np.ndarray
    gaussian_rank: int
    plane_wave_count: int
    reciprocal_vector_count: int
    reserved_peak_bytes: int
    fit_state: _MdfFitState | None = None


@dataclass(frozen=True)
class _MdfCache(Mapping):
    factors: Mapping
    source_signature: str
    source_parameters: tuple
    kpoints_cart: tuple
    q_batch_sizes: tuple
    retained_factor_bytes: int
    reserved_peak_bytes: int
    bra_rows: tuple = ()
    need_k_pairs: bool = False
    aux_basis: object = None
    memory_byte_cap: int = 0
    native_workspace_byte_cap: int = 0

    def __getitem__(self, pair):
        return self.factors[pair]

    def __iter__(self):
        return iter(self.factors)

    def __len__(self):
        return len(self.factors)

    @property
    def retained_cache_bytes(self):
        return self.retained_factor_bytes + 4096 + 512*len(self) + 256*len(self.kpoints_cart)

    @property
    def k_cart_list(self):
        return self.kpoints_cart

    @property
    def lr_ke_cutoff(self):
        return float(dict(self.source_parameters[2])['ke_cutoff'])


@dataclass(frozen=True)
class _MdfScfSource:
    """Explicit diagnostic injection through the drivers' private cache hook.

    The normal SR/LR policy resolves finite source domains and SCF memory
    headroom. This object changes only the fitted source to Gaussian plus
    the selected PW span. It is not a public gdf_method or symmetry adapter.
    """
    plane_wave_cutoff: float

    def __post_init__(self):
        if (not np.isscalar(self.plane_wave_cutoff) or np.iscomplexobj(self.plane_wave_cutoff)
                or not np.isfinite(self.plane_wave_cutoff) or self.plane_wave_cutoff < 0):
            raise ValueError('private MDF SCF requires a finite nonnegative PW cutoff')

    def require_driver(self, system, *, gdf_method, k_exchange, ibz_native):
        if int(system.dim) != 3 or gdf_method != 'rsgdf' or k_exchange != 'gdf' or ibz_native:
            raise NotImplementedError(
                'private MDF SCF injection requires the 3D full-mesh rsgdf driver with GDF exchange')


@dataclass(frozen=True)
class _MdfJK:
    coulomb: tuple
    exchange: tuple
    bra_rows: tuple
    coulomb_energy: float
    exchange_energy: float | None
    reserved_peak_bytes: int


@dataclass(frozen=True)
class _MdfMeanField:
    cache: _MdfCache
    jk: _MdfJK
    focks: tuple
    one_electron_energy: float
    exchange_energy: float
    exxdiv_energy: float
    xc_energy: float
    nuclear_energy: float
    electronic_energy: float
    total_energy: float
    reserved_peak_bytes: int


def _finite(array, label):
    # Scan borrowed contiguous storage without a full-sized boolean mask.
    for row in np.asarray(array):
        flat = row.ravel(order='K')
        for first in range(0, len(flat), 4096):
            if not np.isfinite(flat[first:first+4096]).all():
                raise ValueError(f'MDF {label} contains nonfinite values')


def _momenta(values, label):
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[1] != 3 or not len(values)
            or np.iscomplexobj(values) or not np.isfinite(values).all()):
        raise ValueError(f'MDF {label} must have finite real shape (nk,3)')
    return np.asarray(values, dtype=np.float64)


def _vectors(system, q, cutoff, cap, candidates, *, omit_zero=False):
    from .aux_basis import _rsgdf_bounded_reciprocal_sphere

    if cutoff == 0:
        return np.empty((0, 3), dtype=np.float64)
    vectors = _rsgdf_bounded_reciprocal_sphere(
        system, q, cutoff, output_byte_cap=max(1, cap),
        workspace_byte_cap=4096+128*512, candidate_cap=candidates,
        allow_empty=omit_zero,
    )
    if omit_zero:
        # Same singular-mode convention as the native SR/LR kernel.
        vectors = vectors[np.einsum('ij,ij->i', vectors, vectors) > 1e-24]
    return vectors


def _mdf_gradient_workspace_bytes(n_aux, n_orb, n_pairs, n_k, n_pw, n_lr,
                                   n_atoms, native_workspace_byte_cap):
    """Incremental weighted-derivative reservation, excluding borrowed source.

    Match the Gaussian response helper's dense spectral/source temporaries,
    direct PW weights/panels, binding vector copies, native scratch and two
    atom-gradient outputs. Source construction uses this same reservation
    once reciprocal counts are known, before evaluating raw integrals.
    """
    gaussian = (256*n_aux*n_aux + 96*n_pairs*n_aux*n_orb*n_orb
                + 96*n_k*n_orb*n_orb + 4096 + 128*(n_aux+n_k))
    direct_pw = (16*n_pairs*n_pw*n_orb*n_orb + 128*n_pw
                 + 96*max(1, min(n_pw, 16))*n_orb*n_orb + 4096)
    binding = 24*(max(n_lr, n_pw)+n_pairs)+4096
    return gaussian+direct_pw+binding+int(native_workspace_byte_cap)+48*n_atoms


def _build_mdf_shared_q(
    system, orbital, auxiliary, bra_kpoints, q_transfer, *,
    omega, pair_cutoff, auxiliary_cutoff, ke_cutoff, plane_wave_cutoff,
    linear_dep_thr, memory_byte_cap, native_workspace_byte_cap,
    image_candidate_cap, reciprocal_candidate_cap, integral_screen_error=0.0,
    _metric_state=None, retain_fit_state=False,
    gradient_response_byte_cap=None, gradient_nkpoints=None,
):
    """MDF factors for one admitted shared-q batch on explicit finite domains.

    M and T come from the same double-image SR/LR source as RSGDF. Subtract
    the selected bare-Coulomb PW Gram blocks, whiten the residual Gaussian
    metric, and append the PW factors. M and T already omit the singular
    mode: adding a second Vbar correction would change the Hamiltonian.

    The numerical reservation includes raw/projection outputs, the final
    joined factors, LAPACK work, a retained q whitener and vector copies.
    Borrowed bases and library/allocator overhead remain separate. No full
    AO-pair-by-G temporary or real-space image list is built in Python.
    retain_fit_state preserves raw residual T, the full eigensystem and
    reciprocal lists for differentiation, sharing their admitted storage.
    The optional gradient budget admits the subsequent response separately
    from this source cap. It requires retained state and the full density
    mesh size; both vector-list copies are accounted for before integrals.
    """
    from scipy.linalg import get_lapack_funcs
    from . import _vibeqc_core as core
    from .aux_basis import (
        _RangeSeparatedGdfAdmissionError, _canonical_reciprocal_transfer,
        _range_separated_gdf_source_signature, _metric_keep_mask,
    )

    bras = _momenta(bra_kpoints, 'bra kpoints')
    q_input = np.asarray(q_transfer)
    if q_input.shape != (3,) or np.iscomplexobj(q_input) or not np.isfinite(q_input).all():
        raise ValueError('MDF q must be a finite real three-vector')
    q = _canonical_reciprocal_transfer(system, q_input)
    kets = np.ascontiguousarray(bras + q)
    n, a, k = int(orbital.nbasis), int(auxiliary.nbasis), len(bras)
    if min(n, a) <= 0:
        raise ValueError('MDF requires nonempty bases')
    caps = (memory_byte_cap, native_workspace_byte_cap,
            image_candidate_cap, reciprocal_candidate_cap)
    if any(int(v) != v or v <= 0 for v in caps):
        raise ValueError('MDF memory and candidate caps must be positive integers')
    if gradient_response_byte_cap is not None or gradient_nkpoints is not None:
        if (not retain_fit_state or gradient_response_byte_cap is None or gradient_nkpoints is None
                or any(int(v) != v or v <= 0 for v in (gradient_response_byte_cap, gradient_nkpoints))):
            raise ValueError('MDF gradient admission requires retained state, a positive cap and full mesh size')
    if any(not np.isfinite(v) or v <= 0 for v in
           (omega, pair_cutoff, auxiliary_cutoff, ke_cutoff)):
        raise ValueError('MDF omega and SR/LR cutoffs must be finite and positive')
    if any(not np.isfinite(v) or v < 0 for v in
           (plane_wave_cutoff, linear_dep_thr, integral_screen_error)):
        raise ValueError('MDF PW cutoff, metric threshold and screen must be finite and nonnegative')
    source_key = (
        _range_separated_gdf_source_signature(system, orbital, auxiliary),
        tuple(q), omega, pair_cutoff, auxiliary_cutoff, ke_cutoff,
        plane_wave_cutoff, linear_dep_thr, integral_screen_error,
    )
    hit = bool(_metric_state)
    if hit and _metric_state.get('source_key') != source_key:
        raise ValueError('MDF metric state belongs to a different source or PW span')
    metric_bytes, tensor_bytes = 16*a*a, 16*k*a*n*n
    heevd, query = get_lapack_funcs(('heevd', 'heevd_lwork'), dtype=np.complex128)
    work, integer_work, real_work, info = query(a, compute_v=1, lower=1)
    if info:
        raise RuntimeError(f'MDF LAPACK workspace query failed ({info})')
    lw, li, lr = int(np.ceil(work.real)), int(integer_work), int(np.ceil(real_work))
    eigen_bytes = 16*lw + heevd.int_dtype.itemsize*li + 8*lr
    fixed = (4096 + 128*(a+k) + 6*metric_bytes + 3*tensor_bytes + eigen_bytes
             + max(int(native_workspace_byte_cap), 4096+128*512))
    remaining = int(memory_byte_cap) - fixed
    if remaining <= 0:
        raise _RangeSeparatedGdfAdmissionError('MDF source/whitening reservation exceeds memory cap')
    # Each admitted PW point needs two factor copies at the joining peak.
    # Reserve at least one such point for every 24 vector bytes enumerated.
    per_pw = 64 + 32*k*n*n
    pw = _vectors(system, q, plane_wave_cutoff,
                  24*(remaining//(2*per_pw)), reciprocal_candidate_cap, omit_zero=True)
    lr_vectors = _vectors(system, q, ke_cutoff,
                          max(1, remaining//8), reciprocal_candidate_cap)
    count = len(pw)
    reserved = fixed + per_pw*count + 64*len(lr_vectors)
    if reserved > int(memory_byte_cap):
        raise _RangeSeparatedGdfAdmissionError('MDF reciprocal/factor reservation exceeds memory cap')
    if gradient_response_byte_cap is not None:
        response_bytes = _mdf_gradient_workspace_bytes(
            a, n, k, int(gradient_nkpoints), count, len(lr_vectors),
            len(system.unit_cell), native_workspace_byte_cap,
        )
        if response_bytes > gradient_response_byte_cap:
            raise _RangeSeparatedGdfAdmissionError('MDF response weights and reciprocal copies exceed memory cap')
    m, t, ng = core.compute_gdf_range_separated_integrals(
        orbital, auxiliary, system, q, kets, lr_vectors, omega,
        pair_cutoff, auxiliary_cutoff, metric_bytes+tensor_bytes,
        int(native_workspace_byte_cap), int(image_candidate_cap),
        integral_screen_error, not hit,
    )
    if not retain_fit_state:
        del lr_vectors
    if count:
        mpw, tpw, fpw, actual_count = core._compute_gdf_plane_wave_projection(
            orbital, auxiliary, system, q, kets, pw, pair_cutoff,
            metric_bytes+tensor_bytes+16*k*count*n*n,
            int(native_workspace_byte_cap), int(image_candidate_cap), not hit,
        )
        if actual_count != count:
            raise ValueError('MDF native projection changed the admitted PW span')
        t -= tpw
        if not hit:
            m -= mpw
        del mpw, tpw
    else:
        fpw = np.empty((k, 0, n, n), dtype=np.complex128)
    if hit:
        eigenvalues, whitener = _metric_state['eigenvalues'], _metric_state['whitener']
        vectors, keep = _metric_state['eigenvectors'], _metric_state['keep']
    else:
        _finite(m, 'residual metric')
        scale, error = 1., 0.
        for column in range(a):
            scale = max(scale, float(np.max(np.abs(m[:, column]))))
            error = max(error, float(np.max(np.abs(m[:, column] - m[column, :].conj()))))
        if error > 64*np.finfo(float).eps*a*scale:
            raise ValueError('MDF residual metric is not Hermitian; converge the source domains')
        eigenvalues, vectors, info = heevd(m, compute_v=1, lower=1, overwrite_a=0,
                                          lwork=lw, liwork=li, lrwork=lr)
        if info or not np.isfinite(eigenvalues).all():
            raise ValueError(f'MDF residual eigensystem failed ({info})')
        tolerance = max(linear_dep_thr, 64*np.finfo(float).eps*a*
                        max(1., float(np.max(np.abs(eigenvalues)))))
        if eigenvalues[0] < -tolerance:
            raise ValueError('MDF residual metric has negative modes; converge the source domains')
        keep = _metric_keep_mask(eigenvalues, linear_dep_thr)
        kept = np.flatnonzero(keep)
        if not len(kept) and not count:
            raise ValueError('MDF has no retained Gaussian or plane-wave modes')
        whitener = np.empty((len(kept), a), dtype=np.complex128)
        for row, index in enumerate(kept):
            whitener[row] = vectors[:, index].conj() / np.sqrt(eigenvalues[index])
    del m
    rank = len(whitener)
    factors = np.empty((k, rank+count, n, n), dtype=np.complex128)
    for i in range(k):
        _finite(t[i], 'residual tensor')
        if rank:
            np.matmul(whitener, t[i].reshape(a, -1), out=factors[i, :rank].reshape(rank, -1))
        factors[i, rank:] = fpw[i]
        _finite(factors[i], 'factors')
    if _metric_state is not None and not hit:
        eigenvalues.setflags(write=False)
        whitener.setflags(write=False)
        vectors.setflags(write=False)
        keep.setflags(write=False)
        _metric_state.update(source_key=source_key, eigenvalues=eigenvalues, whitener=whitener,
                             eigenvectors=vectors, keep=keep)
    fit_state = None
    if retain_fit_state:
        for array in (t, vectors, keep, lr_vectors, pw, kets, factors, eigenvalues, q):
            array.setflags(write=False)
        fit_state = _MdfFitState(t, vectors, keep, lr_vectors, pw, factors[:, rank:],
                                 kets, source_key, int(image_candidate_cap))
    return _MdfBatch(factors, q, eigenvalues, rank, count, int(ng), reserved, fit_state)


def _build_mdf_cache(system, orbital, auxiliary, kpoints_cart, need_k_pairs, *,
                     memory_byte_cap, plane_wave_cutoff, reciprocal_candidate_cap,
                     bra_rows=None, **source_options):
    """Private shared-q cache with PW-aware retention and admitted subdivision.

    All numerical domains are explicit in source_options. The cache carries
    a distinct MDF identity so a GDF-only derivative consumer cannot silently
    treat it as an undressed Gaussian fit. Public SCF selection and symmetry
    transport remain separate from the private injection and derivative path.
    """
    from .aux_basis import (
        _RangeSeparatedGdfAdmissionError, _canonical_reciprocal_transfer,
        _range_separated_gdf_source_signature,
    )

    points = _momenta(kpoints_cart, 'cache kpoints')
    nk, n, a = len(points), int(orbital.nbasis), int(auxiliary.nbasis)
    if min(n, a) <= 0 or int(memory_byte_cap) != memory_byte_cap or memory_byte_cap <= 0:
        raise ValueError('MDF cache requires nonempty bases and a positive integer memory cap')
    if not np.isfinite(plane_wave_cutoff) or plane_wave_cutoff < 0:
        raise ValueError('MDF PW cutoff must be finite and nonnegative')
    rows = tuple(range(nk)) if bra_rows is None else tuple(bra_rows)
    if (not rows or any(int(i) != i or not 0 <= i < nk for i in rows)
            or len(set(rows)) != len(rows)):
        raise ValueError('MDF bra rows must be distinct indices in the k mesh')
    rows = tuple(map(int, rows))
    pair_count = len(rows)*nk+nk-len(rows) if need_k_pairs else nk
    metadata = 4096+512*pair_count+256*nk
    reservation = metadata+16*pair_count*a*n*n
    census_workspace = 4096+128*512 if plane_wave_cutoff else 0
    if reservation+census_workspace >= memory_byte_cap:
        raise _RangeSeparatedGdfAdmissionError('MDF retained Gaussian cache and PW census exceed memory cap')
    pairs = ([(i, j) for i in rows for j in range(nk)] +
             [(i, i) for i in range(nk) if i not in set(rows)]
             if need_k_pairs else [(i, i) for i in range(nk)])
    groups = {}
    for i, j in pairs:
        q = _canonical_reciprocal_transfer(system, points[j]-points[i])
        key = tuple(float(x) for x in np.round(q, 14))
        groups.setdefault(key, (q, []))[1].append((i, j))
    counts = {}
    for key, (q, members) in groups.items():
        # Count before native sources or retained factors. Enumeration is
        # bounded and discarded one q at a time; never hold all PW meshes.
        remaining = int(memory_byte_cap)-reservation-census_workspace
        per_vector = 64+16*len(members)*n*n
        pw = _vectors(system, q, plane_wave_cutoff,
                      24*(remaining//per_vector), reciprocal_candidate_cap, omit_zero=True)
        counts[key] = len(pw)
        reservation += 16*len(members)*len(pw)*n*n
        del pw
    build_cap = int(memory_byte_cap)-reservation
    cache, sizes, retained, peak = {}, [], 0, reservation
    for key in sorted(groups):
        q, members = groups[key]
        state, widths = {}, []
        first, width = 0, len(members)
        while first < len(members):
            selected = members[first:first+width]
            try:
                batch = _build_mdf_shared_q(
                    system, orbital, auxiliary, points[[i for i, _ in selected]], q,
                    plane_wave_cutoff=plane_wave_cutoff, memory_byte_cap=build_cap,
                    reciprocal_candidate_cap=reciprocal_candidate_cap,
                    _metric_state=state, **source_options,
                )
            except _RangeSeparatedGdfAdmissionError:
                if len(selected) == 1:
                    raise
                width = max(1, len(selected)//2)
                continue
            if batch.plane_wave_count != counts[key]:
                raise ValueError('MDF batch changed the admitted PW span')
            batch.factors.setflags(write=False)
            for pair, factors in zip(selected, batch.factors):
                cache[pair] = factors
            retained += batch.factors.nbytes
            peak = max(peak, reservation+batch.reserved_peak_bytes)
            widths.append(len(selected))
            first += len(selected)
            del batch
        sizes.append(tuple(widths))
        del state
    return _MdfCache(
        MappingProxyType(cache), _range_separated_gdf_source_signature(system, orbital, auxiliary),
        ('mdf-sr-lr-pw-v1', plane_wave_cutoff,
         tuple(sorted(dict(source_options, reciprocal_candidate_cap=reciprocal_candidate_cap).items()))),
        tuple(map(tuple, points)), tuple(sizes), retained, peak, rows, bool(need_k_pairs),
        auxiliary, int(memory_byte_cap), int(source_options['native_workspace_byte_cap']),
    )


def _build_mdf_jk(
    cache, system, orbital, auxiliary, kpoints_cart, densities, weights, *,
    memory_byte_cap, workspace_byte_cap=32*1024**2, spin_resolved=False,
    with_exchange=True, bra_rows=None,
):
    """Apply the private MDF operator to an explicit fixed density.

    Borrow a real/complex NumPy density with shape (nk,nao,nao), or
    (2,nk,nao,nao) for separate alpha/beta channels. Restricted densities
    include both spins. J uses the total density and K uses each supplied
    channel with no spin prefactor. All ket weights contribute to every
    requested exchange bra; J is returned on the complete input mesh.

    E_J = 1/2 sum_k w_k Tr(D_total J). E_K uses -1/4 for a spin-summed
    restricted density and -1/2 for each separate spin density. E_K is
    absent for partial bra rows or when exchange is disabled. Neither the
    potentials nor the energies contain Madelung, nuclear or XC terms.

    This is an integration seam for numerical acceptance, not SCF dispatch.
    Geometry/bases and the ordered mesh must match the retained cache.
    The cap covers the cache, borrowed density, returned matrices and a
    serial contraction workspace; runtime/allocator overhead is separate.
    """
    from .aux_basis import (
        _RangeSeparatedGdfAdmissionError, _range_separated_gdf_source_signature,
        _build_coulomb_from_diagonal_factors, _accumulate_exchange_from_factors,
    )

    if not isinstance(cache, _MdfCache):
        raise TypeError('MDF J/K requires a common-source MDF cache')
    if cache.source_signature != _range_separated_gdf_source_signature(system, orbital, auxiliary):
        raise ValueError('MDF J/K geometry or bases differ from the retained source')
    points = _momenta(kpoints_cart, 'J/K kpoints')
    if not np.array_equal(points, np.asarray(cache.kpoints_cart)):
        raise ValueError('MDF J/K requires the same ordered k mesh as the cache')
    nk, n = len(points), int(orbital.nbasis)
    ns = 2 if spin_resolved else 1
    expected = (2, nk, n, n) if spin_resolved else (nk, n, n)
    if (not isinstance(densities, np.ndarray) or densities.shape != expected
            or densities.dtype not in (np.dtype('float32'), np.dtype('float64'),
                                       np.dtype('complex64'), np.dtype('complex128'))):
        raise ValueError(f'MDF J/K requires a floating-point density array of shape {expected}')
    channels = densities if spin_resolved else densities[None]
    k_weights = np.asarray(weights)
    if (k_weights.shape != (nk,) or np.iscomplexobj(k_weights)
            or not np.isfinite(k_weights).all() or np.any(k_weights < 0)
            or not np.isclose(k_weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError('MDF J/K requires normalized nonnegative real k weights')
    rows = (cache.bra_rows if bra_rows is None else tuple(bra_rows)) if with_exchange else ()
    if with_exchange and (
        not cache.need_k_pairs or not rows or len(set(rows)) != len(rows)
        or any(int(i) != i or not 0 <= i < nk for i in rows)
    ):
        raise ValueError('MDF exchange requires distinct admitted bra rows and a pair cache')
    rows = tuple(map(int, rows))
    required = {(i, i) for i in range(nk)} | {(i, j) for i in rows for j in range(nk)}
    if not required.issubset(cache):
        raise ValueError('MDF cache is missing a required diagonal or exchange pair')
    diagonal = [cache[(i, i)] for i in range(nk)]
    shape = np.shape(diagonal[0])
    if (len(shape) != 3 or shape[0] <= 0 or shape[1:] != (n, n)
            or any(np.shape(f) != shape for f in diagonal)
            or any(np.ndim(cache[p]) != 3 or np.shape(cache[p])[1:] != (n, n)
                   or np.shape(cache[p])[0] <= 0 for p in required)):
        raise ValueError('MDF J/K factors have inconsistent AO or diagonal fitting spaces')
    if any(int(v) != v or v <= 0 for v in (memory_byte_cap, workspace_byte_cap)):
        raise ValueError('MDF J/K memory caps must be positive integers')
    if with_exchange and workspace_byte_cap < 96*n*n:
        raise _RangeSeparatedGdfAdmissionError('MDF exchange workspace cannot hold one auxiliary panel')
    # Shared J uses at most 1 MiB of factors (or one AO matrix), panel
    # vectors and AO temporaries. Its final Hermitian output coexists with
    # the original matrices; charge both sets. Spin totals need one copy.
    panel = max(1, min(shape[0], 2**20 // max(1, 16*n*n)))
    j_scratch = 32*panel*n*n + 64*panel + 64*n*n
    persistent = (cache.retained_cache_bytes + densities.nbytes + 4096 + 256*nk
                  + 16*n*n*((nk if spin_resolved else 0) + 2*nk + ns*len(rows)))
    reserved = persistent + max(j_scratch, int(workspace_byte_cap) if with_exchange else 0)
    if reserved > memory_byte_cap:
        raise _RangeSeparatedGdfAdmissionError('MDF retained state and J/K workspace exceed memory cap')
    for channel in channels:
        for density in channel:
            _finite(density, 'density')
            scale = max(1., float(np.max(np.abs(density))))
            if np.max(np.abs(density-density.conj().T)) > 1e-12*scale:
                raise ValueError('MDF J/K requires Hermitian densities')
    total = channels[0]+channels[1] if spin_resolved else channels[0]
    coulomb = tuple(_build_coulomb_from_diagonal_factors(diagonal, total, k_weights))
    ej = .5*sum(float(w)*float(np.einsum('ij,ji->', d, j).real)
                for w, d, j in zip(k_weights, total, coulomb))
    exchange = []
    for channel in channels:
        matrices = []
        for i in rows:
            matrix = np.zeros((n, n), dtype=np.complex128)
            for j, (density, weight) in enumerate(zip(channel, k_weights)):
                if weight:
                    _accumulate_exchange_from_factors(
                        matrix, cache[(i, j)], density, float(weight), int(workspace_byte_cap),
                    )
            matrix = .5*(matrix+matrix.conj().T)
            _finite(matrix, 'exchange')
            matrices.append(matrix)
        exchange.append(tuple(matrices))
    ek = None
    if with_exchange and set(rows) == set(range(nk)):
        ek = (-.5 if spin_resolved else -.25)*sum(
            float(k_weights[i])*float(np.einsum('ij,ji->', channel[i], matrix).real)
            for channel, matrices in zip(channels, exchange)
            for i, matrix in zip(rows, matrices)
        )
    for matrix in coulomb:
        _finite(matrix, 'Coulomb')
    if not np.isfinite(ej) or (ek is not None and not np.isfinite(ek)):
        raise ValueError('MDF J/K energy is nonfinite')
    return _MdfJK(coulomb, tuple(exchange), rows, ej, ek, reserved)


def _evaluate_mdf_mean_field(
    cache, system, orbital, kpoints_cart, densities, weights, hcore, overlap, *,
    nuclear_energy, memory_byte_cap, workspace_byte_cap=32*1024**2,
    spin_resolved=False, alpha_hf=1., madelung=0., xc_potential=None, xc_energy=None,
):
    """Evaluate physical Focks and energy on the private MDF energy cache.

    Hcore and S have shape (nk,n,n); D and optional Vxc follow _build_mdf_jk's
    restricted or separate-spin convention. Hcore must already contain the
    caller's resolved nuclear/ECP operator. Vxc and Exc must be supplied
    together, evaluated at this D; nonunit exchange requires both. No XC
    functional, electrostatic gauge, nuclear energy or Madelung is inferred.

    F_s = H + J - c_s alpha (K_s + xi S D_s S) + Vxc_s, with c_s=1/2
    restricted or 1 separate-spin. Exc enters the energy once, without a
    Tr(D Vxc) double-counting term. This evaluation does not establish SCF
    convergence or construct the overlap Lagrangian needed for forces.

    The cap includes the cache, all borrowed input matrices, J/K and Fock
    outputs, and assembly/contraction scratch. External integral/XC builders,
    other SCF arrays and allocator overhead are outside this phase's budget.
    """
    from .aux_basis import _RangeSeparatedGdfAdmissionError

    if not isinstance(cache, _MdfCache) or cache.aux_basis is None:
        raise ValueError('MDF mean-field evaluation requires its cache and auxiliary basis')
    scalars = (alpha_hf, madelung, nuclear_energy)
    if any(not np.isscalar(v) or np.iscomplexobj(v) or not np.isfinite(v) for v in scalars):
        raise ValueError('MDF mean-field coefficients and nuclear energy must be finite and real')
    if not 0 <= alpha_hf <= 1:
        raise ValueError('MDF mean-field exchange fraction must be in [0,1]')
    if (xc_potential is None) != (xc_energy is None) or (alpha_hf != 1 and xc_potential is None):
        raise ValueError('MDF mean-field XC potential and energy must be supplied together for nonunit exchange')
    exc = 0. if xc_energy is None else xc_energy
    if not np.isscalar(exc) or np.iscomplexobj(exc) or not np.isfinite(exc):
        raise ValueError('MDF XC energy must be finite and real')
    if any(int(v) != v or v <= 0 for v in (memory_byte_cap, workspace_byte_cap)):
        raise ValueError('MDF mean-field memory caps must be positive integers')
    nk, n, ns = len(cache.kpoints_cart), int(orbital.nbasis), 2 if spin_resolved else 1
    density_shape = (ns, nk, n, n) if spin_resolved else (nk, n, n)
    matrices = [(densities, density_shape, 'density'), (hcore, (nk, n, n), 'Hcore'),
                (overlap, (nk, n, n), 'overlap')]
    if xc_potential is not None:
        matrices.append((xc_potential, density_shape, 'XC potential'))
    for value, shape, label in matrices:
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or value.dtype not in (np.dtype('float64'), np.dtype('complex128'))):
            raise ValueError(f'MDF {label} requires float64/complex128 shape {shape}')
    # J/K already admits D and the cache. Reserve other borrowed arrays,
    # all returned Focks and serial validation/SDS/Fock temporaries here.
    extra = (sum(value.nbytes for value, _, _ in matrices[1:])
             + 16*ns*nk*n*n + 128*n*n + 4096 + 256*ns*nk)
    jk_cap = int(memory_byte_cap)-extra
    if jk_cap <= cache.retained_cache_bytes+densities.nbytes:
        raise _RangeSeparatedGdfAdmissionError('MDF mean-field inputs and Focks exceed memory cap')
    for value, _, label in matrices[1:]:
        channels = value if value.ndim == 4 else value[None]
        for channel in channels:
            for matrix in channel:
                _finite(matrix, label)
                scale = max(1., float(np.max(np.abs(matrix))))
                if np.max(np.abs(matrix-matrix.conj().T)) > 1e-12*scale:
                    raise ValueError(f'MDF {label} must be Hermitian')
    jk = _build_mdf_jk(
        cache, system, orbital, cache.aux_basis, kpoints_cart, densities, weights,
        memory_byte_cap=jk_cap, workspace_byte_cap=workspace_byte_cap,
        spin_resolved=spin_resolved, with_exchange=bool(alpha_hf), bra_rows=tuple(range(nk)),
    )
    density_channels = densities if spin_resolved else densities[None]
    xc_channels = (xc_potential if spin_resolved else xc_potential[None]) if xc_potential is not None else None
    coefficient = float(alpha_hf)*(1. if spin_resolved else .5)
    one_electron, exxdiv, focks = 0., 0., []
    for spin, channel in enumerate(density_channels):
        fock_channel = []
        for k, density in enumerate(channel):
            weight = float(weights[k])
            one_electron += weight*float(np.einsum('ij,ji->', density, hcore[k]).real)
            fock = np.array(hcore[k], dtype=np.complex128, copy=True)
            fock += jk.coulomb[k]
            if coefficient:
                fock -= coefficient*jk.exchange[spin][k]
                if madelung:
                    shift = overlap[k]@density@overlap[k]
                    exxdiv -= .5*coefficient*float(madelung)*weight*float(np.einsum('ij,ji->', density, shift).real)
                    fock -= coefficient*float(madelung)*shift
                    del shift
            if xc_channels is not None:
                fock += xc_channels[spin][k]
            _finite(fock, 'physical Fock')
            fock.setflags(write=False)
            fock_channel.append(fock)
        focks.append(tuple(fock_channel))
    exchange = float(alpha_hf)*jk.exchange_energy if alpha_hf else 0.
    electronic = one_electron+jk.coulomb_energy+exchange+exxdiv+float(exc)
    total = electronic+float(nuclear_energy)
    if not np.isfinite(electronic) or not np.isfinite(total):
        raise ValueError('MDF mean-field energy is nonfinite')
    return _MdfMeanField(
        cache, jk, tuple(focks), one_electron, exchange, exxdiv, float(exc),
        float(nuclear_energy), electronic, total, jk.reserved_peak_bytes+extra,
    )


def _mdf_jk_response_weights(
    batch, pairs, densities, weights, *, workspace_byte_cap,
    coulomb_scale=1.0, exchange_scale=1.0, coulomb_sources=None,
    include_coulomb_metric=True,
):
    """Unconjugated residual-M/T and direct-PW weights for one q batch.

    Reuse the truncated-inverse response of the residual Gaussian metric,
    including its dropped modes. The independent PW Gram term responds
    directly through its factors. Restricted exchange has coefficient
    -1/4; a separate-spin caller uses exchange_scale=2 for each spin.

    Subdivided Hartree batches need both complete raw sources:
    sum_k w_k T_res(k):D(k).T and sum_k w_k F_PW(k):D(k).T. They are
    shared by every batch; include the Gaussian metric response once.
    There is no PW metric-response term because its metric is identity.
    Returned arrays and scratch are admitted here; source/density arrays
    and native derivative workspace are borrowed and charged by the caller.
    """
    from .periodic_gdf_gradient import _range_separated_jk_response_weights

    state = batch.fit_state
    if not isinstance(state, _MdfFitState):
        raise ValueError('MDF response requires its retained residual source')
    source, pw = state.three_center, state.pw_factors
    nk = len(densities)
    if (source.ndim != 4 or pw.ndim != 4 or pw.shape[0] != source.shape[0]
            or pw.shape[2:] != source.shape[2:] or len(state.source_key) != 9):
        raise ValueError('MDF response source dimensions or provenance are inconsistent')
    count, n = pw.shape[1], source.shape[-1]
    w = np.asarray(weights)
    if (w.shape != (nk,) or np.iscomplexobj(w) or not np.isfinite(w).all()
            or np.any(w < 0) or not np.isclose(w.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError('MDF response requires normalized nonnegative k weights')
    gaussian_source = pw_source = None
    if coulomb_sources is not None:
        if len(coulomb_sources) != 2:
            raise ValueError('MDF response needs both Gaussian and PW Hartree sources')
        gaussian_source, pw_source = map(np.asarray, coulomb_sources)
        if pw_source.shape != (count,) or not np.isfinite(pw_source).all():
            raise ValueError('MDF response PW Hartree source has incompatible dimensions or values')
    panel_rank = max(1, min(count, 16))
    direct_bytes = pw.size*16 + 128*count + 96*panel_rank*n*n + 4096
    gaussian_cap = int(workspace_byte_cap)-direct_bytes
    if gaussian_cap <= 0:
        raise MemoryError('MDF PW response exceeds workspace cap')
    metric_weight, tensor_weight = _range_separated_jk_response_weights(
        batch, pairs, densities, weights, linear_dep_thr=state.source_key[7],
        workspace_byte_cap=gaussian_cap, coulomb_scale=coulomb_scale,
        exchange_scale=exchange_scale, coulomb_source=gaussian_source,
        include_coulomb_metric=include_coulomb_metric,
    )
    # The Gaussian helper validates pairs, Hermitian densities, scales and
    # the fixed spectral threshold before either direct-PW allocation.
    density = [np.asarray(d) for d in densities]
    factor_weight = np.zeros(pw.shape, dtype=np.complex128)
    if coulomb_scale and count:
        if pw_source is None:
            if sorted(tuple(p) for p in pairs) != [(k, k) for k in range(nk)]:
                raise ValueError('MDF PW Hartree response requires every diagonal or its full source')
            pw_source = np.zeros(count, dtype=np.complex128)
            for position, (i, _) in enumerate(pairs):
                pw_source += w[i]*np.einsum('Pmn,nm->P', pw[position], density[i])
        for position, (i, _) in enumerate(pairs):
            for first in range(0, count, panel_rank):
                last = min(first+panel_rank, count)
                factor_weight[position, first:last] = (
                    coulomb_scale*w[i]*pw_source[first:last].conj()[:, None, None]*density[i].T
                )
    if exchange_scale and count:
        for position, (i, j) in enumerate(pairs):
            for first in range(0, count, panel_rank):
                last = min(first+panel_rank, count)
                factor_weight[position, first:last] -= (
                    .5*exchange_scale*w[i]*w[j]
                    * np.matmul(density[i], np.matmul(pw[position, first:last], density[j])).conj()
                )
    for value in (metric_weight, tensor_weight, factor_weight):
        _finite(value, 'response weights')
    return metric_weight, tensor_weight, factor_weight


def _compute_mdf_jk_gradient(
    system, orbital, auxiliary, batch, pairs, kpoints_cart, densities, weights, *,
    memory_byte_cap, native_workspace_byte_cap, coulomb_scale=1.0,
    exchange_scale=1.0, coulomb_sources=None, include_coulomb_metric=True,
):
    """Private fixed-density J/K atomic energy derivative of one MDF batch.

    dE = WM:dM_SRLR + WT:dT_SRLR - WM:dM_PW - WT:dT_PW + WF:dF_PW.
    The same retained residual eigensystem and native source domains serve
    every term. No eigensystem is reconstructed from whitened factors.

    The cap is incremental over borrowed source/factor/density storage;
    it admits response weights, binding copies, native scratch and output.
    Cell, momenta, reciprocal lists, image membership and density are fixed.
    This excludes Pulay, one-electron, nuclear, XC and Madelung derivatives
    and must not be exposed as a complete SCF force.
    """
    from .aux_basis import (
        _RangeSeparatedGdfAdmissionError, _canonical_reciprocal_transfer,
        _range_separated_gdf_source_signature,
    )
    from . import _vibeqc_core as core

    state = batch.fit_state
    if not isinstance(state, _MdfFitState) or len(state.source_key) != 9:
        raise ValueError('MDF gradient requires the retained energy source')
    if state.source_key[0] != _range_separated_gdf_source_signature(system, orbital, auxiliary):
        raise ValueError('MDF gradient geometry or bases differ from the retained source')
    points = _momenta(kpoints_cart, 'gradient kpoints')
    if (len(pairs) != len(state.ket_kpoints) or len(points) != len(densities)
            or any(len(p) != 2 or any(int(i) != i or not 0 <= i < len(points) for i in p)
                   for p in pairs)):
        raise ValueError('MDF gradient pairs do not match the retained batch')
    q = np.asarray(state.source_key[1])
    if not np.array_equal(batch.transfer, q):
        raise ValueError('MDF gradient transfer differs from the retained source')
    for position, (i, j) in enumerate(pairs):
        transfer = _canonical_reciprocal_transfer(system, points[j]-points[i])
        if (not np.allclose(transfer, q, atol=1e-12, rtol=0)
                or not np.allclose(points[i]+q, state.ket_kpoints[position], atol=1e-12, rtol=0)):
            raise ValueError('MDF gradient ordered k pairs differ from the retained source')
    if any(int(v) != v or v <= 0 for v in (memory_byte_cap, native_workspace_byte_cap)):
        raise ValueError('MDF gradient memory caps must be positive integers')
    _, a, n, _ = state.three_center.shape
    required = _mdf_gradient_workspace_bytes(
        a, n, len(pairs), len(points), len(state.pw_vectors), len(state.lr_vectors),
        len(system.unit_cell), native_workspace_byte_cap,
    )
    if required > memory_byte_cap:
        raise _RangeSeparatedGdfAdmissionError('MDF response weights and reciprocal copies exceed memory cap')
    output_bytes = 24*len(system.unit_cell)
    binding_bytes = max(state.lr_vectors.nbytes, state.pw_vectors.nbytes)+state.ket_kpoints.nbytes+4096
    response_cap = int(memory_byte_cap)-int(native_workspace_byte_cap)-2*output_bytes-binding_bytes
    if response_cap <= 0:
        raise MemoryError('MDF gradient response and native workspace exceed memory cap')
    wm, wt, wf = _mdf_jk_response_weights(
        batch, pairs, densities, weights, workspace_byte_cap=response_cap,
        coulomb_scale=coulomb_scale, exchange_scale=exchange_scale,
        coulomb_sources=coulomb_sources, include_coulomb_metric=include_coulomb_metric,
    )
    _, _, omega, pair_cutoff, auxiliary_cutoff, _, _, _, screen = state.source_key
    result = core.compute_gdf_range_separated_gradient_weighted(
        orbital, auxiliary, system, q, state.ket_kpoints, state.lr_vectors,
        omega, pair_cutoff, auxiliary_cutoff, wm, wt, output_bytes,
        int(native_workspace_byte_cap), state.image_candidate_cap, screen,
    )
    if len(state.pw_vectors):
        np.negative(wm, out=wm)
        np.negative(wt, out=wt)
        result += core._compute_gdf_plane_wave_projection_gradient_weighted(
            orbital, auxiliary, system, q, state.ket_kpoints, state.pw_vectors,
            pair_cutoff, wm, wt, wf, output_bytes, int(native_workspace_byte_cap),
            state.image_candidate_cap,
        )
    _finite(result, 'J/K gradient')
    return result


def _compute_mdf_cache_gradient(
    cache, system, orbital, auxiliary, kpoints_cart, densities, weights, *,
    memory_byte_cap, native_workspace_byte_cap, spin_resolved=False,
    coulomb_scale=1.0, exchange_scale=1.0,
):
    """Private fixed-density atomic derivative of a complete MDF J/K cache.

    Rebuild one shared-q batch at a time from the energy cache's domains.
    Two passes over the q=0 diagonals collect full Hartree sources, then
    differentiate each batch; partial-batch Hartree sources are incorrect.
    Spin-resolved inputs use J of the total density and K of each spin.
    Zero-transfer exchange pairs between reciprocal-equivalent mesh entries
    have a separate response group; they are not Hartree diagonal pairs.

    Admit the retained cache/density, source rebuild, response/native scratch
    and returned atom gradient together. Only explicit preallocation rejection
    permits subdivision. This is the fixed-density fitted contribution;
    complete SCF gradient composition is owned by the caller.
    """
    from .aux_basis import (
        _RangeSeparatedGdfAdmissionError, _canonical_reciprocal_transfer,
        _range_separated_gdf_source_signature,
    )

    if (not isinstance(cache, _MdfCache) or len(cache.source_parameters) != 3
            or cache.source_parameters[0] != 'mdf-sr-lr-pw-v1'
            or cache.source_signature != _range_separated_gdf_source_signature(system, orbital, auxiliary)):
        raise ValueError('MDF cache gradient requires the retained energy source')
    points = _momenta(kpoints_cart, 'cache gradient kpoints')
    if not np.array_equal(points, np.asarray(cache.kpoints_cart)):
        raise ValueError('MDF cache gradient requires the same ordered k mesh')
    nk, n, a = len(points), int(orbital.nbasis), int(auxiliary.nbasis)
    expected = (2, nk, n, n) if spin_resolved else (nk, n, n)
    if (not isinstance(densities, np.ndarray) or densities.shape != expected
            or densities.dtype not in (np.dtype('float64'), np.dtype('complex128'))):
        raise ValueError(f'MDF cache gradient requires float64/complex128 density shape {expected}')
    w = np.asarray(weights)
    if (w.shape != (nk,) or np.iscomplexobj(w) or not np.isfinite(w).all()
            or np.any(w < 0) or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0)
            or not np.isfinite([coulomb_scale, exchange_scale]).all()):
        raise ValueError('MDF cache gradient requires normalized k weights and finite scales')
    if any(int(v) != v or v <= 0 for v in (memory_byte_cap, native_workspace_byte_cap)):
        raise ValueError('MDF cache gradient memory caps must be positive integers')
    if any((i, i) not in cache for i in range(nk)) or (
        exchange_scale and any((i, j) not in cache for i in range(nk) for j in range(nk))
    ):
        raise ValueError('MDF cache gradient requires every energy pair; partial exchange rows are insufficient')
    options = dict(cache.source_parameters[2])
    required_options = {'omega', 'pair_cutoff', 'auxiliary_cutoff', 'ke_cutoff',
                        'linear_dep_thr', 'image_candidate_cap', 'reciprocal_candidate_cap'}
    if not required_options.issubset(options):
        raise ValueError('MDF cache gradient requires complete retained source controls')
    # Execution budgets may change, but the physical source controls must
    # remain the ones used by the energy cache. Never accept a new cutoff.
    options.pop('native_workspace_byte_cap', None)
    options.pop('retain_fit_state', None)
    options.pop('gradient_response_byte_cap', None)
    options.pop('gradient_nkpoints', None)
    options.update(plane_wave_cutoff=cache.source_parameters[1],
                   native_workspace_byte_cap=int(native_workspace_byte_cap),
                   retain_fit_state=True)
    diagonal_rank = int(cache[(0, 0)].shape[0])
    fixed = (cache.retained_cache_bytes+densities.nbytes+8192+512*len(cache)
             + 16*nk*n*n*(1 if spin_resolved else 0)
             + 64*(a+diagonal_rank) + 72*len(system.unit_cell))
    remaining = int(memory_byte_cap)-fixed
    source_cap, response_cap = remaining//2, remaining-remaining//2
    if min(source_cap, response_cap) <= native_workspace_byte_cap:
        raise _RangeSeparatedGdfAdmissionError('MDF retained cache, source and gradient exceed memory cap')
    channels = densities if spin_resolved else densities[None]
    for channel in channels:
        for d in channel:
            _finite(d, 'gradient density')
            tolerance = 128*np.finfo(float).eps*n*max(1., float(np.max(np.abs(d))))
            if np.max(np.abs(d-d.conj().T)) > tolerance:
                raise ValueError('MDF cache gradient requires Hermitian densities')
    total = channels[0]+channels[1] if spin_resolved else channels[0]
    groups = {}
    for i, j in cache:
        if not exchange_scale and i != j:
            continue
        q = _canonical_reciprocal_transfer(system, points[j]-points[i])
        # Reciprocal-equivalent mesh entries can have q=0 with i != j.
        # Hartree is diagonal in mesh indices, whereas exchange includes
        # these off-diagonal pairs too. Keep separate response groups even
        # when they share the same physical transfer and residual metric.
        key = (tuple(float(x) for x in np.round(q, 14)), i == j)
        groups.setdefault(key, (q, []))[1].append((i, j))
    result = np.zeros((len(system.unit_cell), 3))
    for key in sorted(groups):
        q, pairs = groups[key]
        metric_state = {}

        def batches():
            first, width = 0, len(pairs)
            while first < len(pairs):
                selected = pairs[first:first+width]
                # Preflight the response before rebuilding raw integrals.
                # Cache factors reveal each pair's PW rank upper bound.
                rank = max(len(cache[p]) for p in selected)
                bound = _mdf_gradient_workspace_bytes(
                    a, n, len(selected), nk, rank, 0, len(system.unit_cell), native_workspace_byte_cap)
                if bound > response_cap:
                    if len(selected) == 1:
                        raise _RangeSeparatedGdfAdmissionError('MDF single-pair gradient response exceeds memory cap')
                    width = max(1, len(selected)//2)
                    continue
                try:
                    batch = _build_mdf_shared_q(
                        system, orbital, auxiliary, points[[i for i, _ in selected]], q,
                        memory_byte_cap=source_cap, _metric_state=metric_state,
                        gradient_response_byte_cap=response_cap, gradient_nkpoints=nk, **options,
                    )
                except _RangeSeparatedGdfAdmissionError:
                    if len(selected) == 1:
                        raise
                    width = max(1, len(selected)//2)
                    continue
                for pair, factor in zip(selected, batch.factors):
                    cached = cache[pair]
                    if cached.shape != factor.shape:
                        raise ValueError('MDF gradient rebuild changed the retained fitting rank')
                    for expected_row, row in zip(cached, factor):
                        scale = max(1., float(np.max(np.abs(expected_row))))
                        if np.max(np.abs(expected_row-row)) > 256*np.finfo(float).eps*max(a, 1)*scale:
                            raise ValueError('MDF gradient rebuild differs from the energy factors')
                # Loop views would otherwise keep this entire factor batch
                # alive while constructing the next one after the yield.
                del factor, cached, expected_row, row
                yield selected, batch
                del batch
                first += len(selected)

        has_j = bool(coulomb_scale) and any(i == j for i, j in pairs)
        sources = None
        if has_j:
            gaussian_source, pw_source = np.zeros(a, complex), None
            for selected, batch in batches():
                state = batch.fit_state
                if pw_source is None:
                    pw_source = np.zeros(batch.plane_wave_count, complex)
                if len(pw_source) != batch.plane_wave_count or any(i != j for i, j in selected):
                    raise ValueError('MDF Hartree source requires one shared diagonal PW span')
                for position, (i, _) in enumerate(selected):
                    gaussian_source += w[i]*np.einsum('Pmn,nm->P', state.three_center[position], total[i])
                    pw_source += w[i]*np.einsum('Pmn,nm->P', state.pw_factors[position], total[i])
                del state, batch
            sources = gaussian_source, pw_source
        first_metric = True
        for selected, batch in batches():
            if has_j:
                result += _compute_mdf_jk_gradient(
                    system, orbital, auxiliary, batch, selected, points, total, w,
                    memory_byte_cap=response_cap, native_workspace_byte_cap=native_workspace_byte_cap,
                    coulomb_scale=coulomb_scale, exchange_scale=0., coulomb_sources=sources,
                    include_coulomb_metric=first_metric,
                )
            if exchange_scale:
                for channel in channels:
                    result += _compute_mdf_jk_gradient(
                        system, orbital, auxiliary, batch, selected, points, channel, w,
                        memory_byte_cap=response_cap, native_workspace_byte_cap=native_workspace_byte_cap,
                        coulomb_scale=0., exchange_scale=exchange_scale*(2 if spin_resolved else 1),
                    )
            first_metric = False
            del batch
    _finite(result, 'cache J/K gradient')
    return result


def _build_mdf_scf_energy_weighted_density(
    cache, density_channels, fock_channels, overlaps, coefficient_channels,
    energy_channels, weights, *, stationarity_tolerance, validation_tolerance=1e-8,
):
    """Build the private MDF overlap Lagrangian from the accepted SCF state.

    In an S-orthonormal retained orbital space, P=C^H S D S C and G=C^H F C.
    W=C (P G + G P)/2 C^H uses the accepted D and physical F, including spin
    and fractional occupations already in D. At stationarity this equals
    C diag(f*epsilon) C^H without substituting a fresh orbital refill for D.
    Sum W over spin channels, with no additional spin or k-weight factor.

    Check retained-space support and the reported physical eigenpairs, and
    apply the driver's weighted AO FDS commutator tolerance before returning.
    These are state consistency checks, not force or domain acceptance.
    The phase cap includes the cache, borrowed arrays, W and serial scratch.
    No occupation cutoff, inverse overlap, or new density is introduced.
    """
    from .aux_basis import _RangeSeparatedGdfAdmissionError

    if not isinstance(cache, _MdfCache):
        raise TypeError('MDF overlap Lagrangian requires its energy cache')
    if any(not np.isfinite(t) or t <= 0 for t in (stationarity_tolerance, validation_tolerance)):
        raise ValueError('MDF state tolerances must be finite and positive')
    nk, ns = len(cache.kpoints_cart), len(density_channels)
    families = (density_channels, fock_channels, coefficient_channels, energy_channels)
    if (ns not in (1, 2) or len(overlaps) != nk or not nk
            or any(len(family) != ns or any(len(channel) != nk for channel in family)
                   for family in families)):
        raise ValueError('MDF overlap Lagrangian requires one or two complete state channels')
    first = overlaps[0]
    if not isinstance(first, np.ndarray) or first.ndim != 2 or first.shape[0] <= 0:
        raise ValueError('MDF state requires nonempty AO overlap matrices')
    n = first.shape[0]
    if np.shape(cache[(0, 0)])[1:] != (n, n):
        raise ValueError('MDF SCF state and energy cache have different AO dimensions')
    arrays = list(overlaps)
    for family in families:
        arrays.extend(matrix for channel in family for matrix in channel)
    if any(not isinstance(a, np.ndarray) or a.dtype not in (np.dtype('float64'), np.dtype('complex128'))
           for a in arrays):
        raise ValueError('MDF state arrays must use float64/complex128 storage')
    for spin in range(ns):
        for k in range(nk):
            c, e = coefficient_channels[spin][k], energy_channels[spin][k]
            if (any(a.shape != (n, n) for a in (overlaps[k], density_channels[spin][k], fock_channels[spin][k]))
                    or c.ndim != 2 or c.shape[0] != n or not 0 < c.shape[1] <= n
                    or e.shape != (c.shape[1],) or np.iscomplexobj(e)):
                raise ValueError('MDF state has incompatible AO/orbital dimensions or complex energies')
    w = np.asarray(weights)
    if (w.shape != (nk,) or np.iscomplexobj(w) or not np.isfinite(w).all()
            or np.any(w < 0) or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError('MDF state requires normalized nonnegative k weights')
    reserved = (cache.retained_cache_bytes + sum(a.nbytes for a in arrays)
                + 16*nk*n*n + 256*n*n + 4096 + 512*ns*nk)
    if reserved > cache.memory_byte_cap:
        raise _RangeSeparatedGdfAdmissionError('MDF state validation and overlap Lagrangian exceed memory cap')
    for a in arrays:
        _finite(a.reshape(1, -1) if a.ndim == 1 else a, 'SCF state')
    result, residual_squared = [], 0.
    for k, overlap in enumerate(overlaps):
        wk = np.zeros((n, n), dtype=np.complex128)
        for spin in range(ns):
            d, f = density_channels[spin][k], fock_channels[spin][k]
            c, e = coefficient_channels[spin][k], energy_channels[spin][k]
            for a in (overlap, d, f):
                if np.max(np.abs(a-a.conj().T)) > validation_tolerance*max(1., float(np.max(np.abs(a)))):
                    raise ValueError('MDF SCF overlap, density and physical Fock must be Hermitian')
            sc = overlap@c
            if np.max(np.abs(c.conj().T@sc-np.eye(c.shape[1]))) > validation_tolerance:
                raise ValueError('MDF SCF orbitals are not S-orthonormal')
            p = sc.conj().T@d@sc
            if np.max(np.abs(c@p@c.conj().T-d)) > validation_tolerance*max(1., float(np.max(np.abs(d)))):
                raise ValueError('MDF accepted density lies outside the retained orbital space')
            g = c.conj().T@f@c
            if np.max(np.abs(g-np.diag(e))) > validation_tolerance*max(1., float(np.max(np.abs(g)))):
                raise ValueError('MDF reported eigenpairs differ from the physical Fock')
            fds = f@d@overlap
            residual_squared += float(w[k])*float(np.linalg.norm(fds-fds.conj().T)**2)
            pg = p@g
            wk += c@(.5*(pg+pg.conj().T))@c.conj().T
            del sc, p, g, fds, pg
        _finite(wk, 'overlap Lagrangian')
        wk.setflags(write=False)
        result.append(wk)
    if not np.isfinite(residual_squared) or np.sqrt(residual_squared) > stationarity_tolerance:
        raise ValueError('MDF accepted density and physical Fock are not stationary at the requested tolerance')
    return result


def _compute_mdf_mean_field_fit_gradient(
    cache, system, orbital, density_channels, weights, kpoints_cart, *, alpha_hf,
):
    """Private adapter for the existing RHF/UHF and KS gradient assemblers.

    One channel contains the restricted spin-summed density; two contain
    separate spin densities. Admit packing of the drivers' matrix lists
    before allocation, then apply the energy cache's exact MDF derivative.
    The cap covers the fitted-derivative phase and its density packing.
    Existing one-electron/XC workspaces and other SCF arrays stay caller-owned.
    """
    from .aux_basis import _RangeSeparatedGdfAdmissionError, _range_separated_gdf_source_signature

    if (not isinstance(cache, _MdfCache) or cache.aux_basis is None
            or cache.source_signature != _range_separated_gdf_source_signature(
                system, orbital, cache.aux_basis)):
        raise ValueError('MDF mean-field derivative requires its energy cache and auxiliary basis')
    if not np.isfinite(alpha_hf) or not 0 <= alpha_hf <= 1:
        raise ValueError('MDF mean-field derivative requires an exchange fraction in [0,1]')
    points = _momenta(kpoints_cart, 'mean-field gradient kpoints')
    if not np.array_equal(points, np.asarray(cache.kpoints_cart)):
        raise ValueError('MDF mean-field derivative requires the energy cache k mesh')
    nk, n = len(points), int(orbital.nbasis)
    if len(density_channels) not in (1, 2) or any(len(c) != nk for c in density_channels):
        raise ValueError('MDF mean-field derivative requires one or two full density channels')
    if any(not isinstance(d, np.ndarray) or d.shape != (n, n)
           or d.dtype not in (np.dtype('float64'), np.dtype('complex128'))
           for channel in density_channels for d in channel):
        raise ValueError('MDF mean-field derivative requires float64/complex128 AO density matrices')
    if cache.memory_byte_cap <= 0 or cache.native_workspace_byte_cap <= 0:
        raise ValueError('MDF mean-field derivative requires retained execution budgets')
    borrowed_bytes = sum(d.nbytes for channel in density_channels for d in channel)
    packed_bytes = 16*len(density_channels)*nk*n*n
    remaining = cache.memory_byte_cap-borrowed_bytes-4096-128*nk
    minimum = cache.retained_cache_bytes+packed_bytes+2*cache.native_workspace_byte_cap
    if remaining <= minimum:
        raise _RangeSeparatedGdfAdmissionError('MDF mean-field density packing and derivative exceed memory cap')
    packed = np.empty((len(density_channels), nk, n, n), dtype=np.complex128)
    for spin, channel in enumerate(density_channels):
        for k, density in enumerate(channel):
            packed[spin, k] = density
    return _compute_mdf_cache_gradient(
        cache, system, orbital, cache.aux_basis, points,
        packed if len(density_channels) == 2 else packed[0], weights,
        memory_byte_cap=remaining, native_workspace_byte_cap=cache.native_workspace_byte_cap,
        spin_resolved=len(density_channels) == 2, coulomb_scale=1., exchange_scale=float(alpha_hf),
    )
