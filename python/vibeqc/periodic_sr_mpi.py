"""Opt-in MPI ownership of admitted short-range GDF q batches.

Only the independent q builds are distributed. Every process retains the
complete ordered factor cache and executes its usual SCF afterwards.
"""
from __future__ import annotations

from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np


def _collective_call(label, call):
    """Finish the error collective before entering a dependent collective."""
    from .mpi import mpi_allgather

    value, error = None, None
    try:
        value = call()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    failures = mpi_allgather(error)
    if any(item is not None for item in failures):
        detail = "; ".join(f"rank {rank}: {item}" for rank, item in enumerate(failures)
                           if item is not None)
        raise RuntimeError(f"SR q MPI {label} refused collectively: {detail}")
    return value


def collective_sr_q_preflight(function):
    """Guard the earlier SCF factor-cache gate before the SR builder."""
    @wraps(function)
    def wrapped(*args, _sr_q=None, **kwargs):
        request = os.environ.get('VIBEQC_SR_Q_MPI', '0')
        if _sr_q is None and request == '0':
            return function(*args, **kwargs)
        from .mpi import mpi_allgather
        requests = mpi_allgather(request)
        if len(set(requests)) != 1 or request not in ('0', '1'):
            raise RuntimeError('SR q MPI ranks must agree on VIBEQC_SR_Q_MPI=0 or 1')
        if request == '0':
            return function(*args, **kwargs)
        sources = mpi_allgather(bool(_sr_q))
        if len(set(sources)) != 1:
            raise RuntimeError('SR q MPI ranks selected different SCF fit sources')
        if not _sr_q:
            return function(*args, **kwargs)
        return _collective_call('SCF source-memory preflight', lambda: function(*args, **kwargs))
    return wrapped


def collective_sr_q_cache(function):
    """Make preparation collective without altering the default serial body."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if (kwargs.get('_q_executor') is not None
                or os.environ.get('VIBEQC_SR_Q_MPI', '0') == '0'):
            return function(*args, **kwargs)
        from .mpi import mpi_world, mpi_allgather

        world = mpi_world()
        request = os.environ.get('VIBEQC_SR_Q_MPI', '0')
        requests = mpi_allgather(request)
        if len(set(requests)) != 1 or request not in ('0', '1'):
            raise RuntimeError('SR q MPI ranks must agree on VIBEQC_SR_Q_MPI=0 or 1')
        if request == '0':
            return function(*args, **kwargs)
        return _execute(lambda executor: function(*args, **kwargs, _q_executor=executor), world)
    return wrapped


def _stamp():
    try:
        import resource
    except ImportError:
        rss = 'not measured'
    else:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)
    return dict(perf_counter=time.perf_counter(), realtime=time.time(),
                load=os.getloadavg() if hasattr(os, 'getloadavg') else 'not measured', maxrss_bytes=rss)


def _elapsed(start):
    end = _stamp()
    return dict(perf_counter_elapsed=end['perf_counter']-start['perf_counter'],
                realtime_elapsed=end['realtime']-start['realtime'],
                load_start=start['load'], load_end=end['load'],
                maxrss_bytes=end['maxrss_bytes'])


def _domain(plan):
    """The fit identity includes actual transfers and their bra ordering."""
    return dict(schema='vibeqc.sr-q-domain/v1', source_signature=plan['source_signature'],
                source_parameters=plan['source_parameters'],
                requested_controls=plan['requested_controls'],
                kpoints=np.asarray(plan['kpoints']).tolist(),
                groups=[dict(key=key, transfer=np.asarray(plan['groups'][key][0]).tolist(),
                             pairs=plan['groups'][key][1]) for key in sorted(plan['groups'])],
                image_candidate_cap=plan['image_candidate_cap'],
                reciprocal_candidate_cap=plan['reciprocal_candidate_cap'],
                factor_frame='eigenmode')


def _validate_plan(plan):
    omega, pair, auxiliary, reciprocal, threshold, images, candidates, screen = plan['source_parameters']
    if any(not np.isfinite(value) or value <= 0 for value in (omega, pair, auxiliary, reciprocal)):
        raise ValueError('SR q MPI requires finite positive omega and resolved cutoffs')
    if any(not np.isfinite(value) or value < 0 for value in (threshold, screen)):
        raise ValueError('SR q MPI requires finite nonnegative metric and screening thresholds')
    if min(images, candidates, plan['memory_byte_cap'], plan['native_workspace_byte_cap']) <= 0:
        raise ValueError('SR q MPI requires positive memory and candidate caps')
    if min(int(plan['basis'].nbasis), int(plan['auxiliary'].nbasis)) < 1:
        raise ValueError('SR q MPI requires nonempty orbital and auxiliary bases')


def _thread_admission():
    from ._vibeqc_core import get_num_threads, blas_info

    requested = int(os.environ['OMP_NUM_THREADS'])
    budget = int(os.environ.get('VIBEQC_SR_Q_TOTAL_THREADS', '8'))
    actual = int(get_num_threads())
    if requested < 1 or actual != requested or budget < 1:
        raise ValueError('set an explicit positive OMP_NUM_THREADS matching the loaded runtime')
    names = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
    if any(os.environ.get(name) != '1' for name in names):
        raise ValueError('SR q MPI requires one BLAS thread per rank')
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        pools = 'not measured'
    else:
        pools = threadpool_info()
        if any(pool.get('user_api') == 'blas' and pool.get('num_threads', 1) != 1
               for pool in pools):
            raise ValueError('a loaded BLAS pool exceeds one thread')
    return dict(omp_threads=actual, total_thread_budget=budget, blas=blas_info(),
                loaded_thread_pools=pools, blas_environment={name:os.environ.get(name) for name in names},
                omp_environment={name:os.environ.get(name, 'not measured') for name in
                                 ('OMP_MAX_ACTIVE_LEVELS', 'OMP_WAIT_POLICY', 'KMP_BLOCKTIME')})


def _profile(fingerprint, n_tasks):
    path = os.environ.get('VIBEQC_SR_Q_COST_PROFILE')
    if not path:
        return None, None
    raw = Path(path).read_bytes()
    profile = json.loads(raw)
    if profile.get('domain_sha256') != fingerprint:
        raise ValueError('measured q-cost profile belongs to a different source/domain')
    costs = tuple(float(value) for value in profile['q_build_seconds'])
    if len(costs) != n_tasks or any(not np.isfinite(value) or value < 0 for value in costs):
        raise ValueError('q-cost profile must contain one finite nonnegative measured cost per ordered q')
    return costs, dict(sha256=sha256(raw).hexdigest(),
                       process_provenance=profile.get('process_provenance', 'not measured'))


def _profile_costs(fingerprint, n_tasks):
    return _profile(fingerprint, n_tasks)[0]


def _build_group(plan, key, build_cap, workspace):
    from .aux_basis import _build_lpq_range_separated_shared_q, _RangeSeparatedGdfAdmissionError

    q, pairs = plan['groups'][key]
    metric_state, factors, sizes = {}, [], []
    first, batch_size, count, peak = 0, len(pairs), None, 0
    omega, pair_cutoff, auxiliary_cutoff, ke_cutoff, threshold, _, _, screen = plan['source_parameters']
    start = _stamp()
    while first < len(pairs):
        selected = pairs[first:first+batch_size]
        try:
            batch = _build_lpq_range_separated_shared_q(
                plan['system'], plan['basis'], plan['auxiliary'],
                np.asarray([plan['kpoints'][i] for i, _ in selected]), q,
                omega=omega, pair_cutoff=pair_cutoff, auxiliary_cutoff=auxiliary_cutoff,
                ke_cutoff=ke_cutoff, linear_dep_thr=threshold, integral_screen_error=screen,
                memory_byte_cap=build_cap, native_workspace_byte_cap=workspace,
                image_candidate_cap=plan['image_candidate_cap'],
                reciprocal_candidate_cap=plan['reciprocal_candidate_cap'], _metric_state=metric_state)
        except _RangeSeparatedGdfAdmissionError as exc:
            if not exc.retry_with_fewer_kpoints or len(selected) == 1:
                raise
            batch_size = max(1, len(selected)//2)
            continue
        if count is not None and count != batch.reciprocal_vector_count:
            raise RuntimeError('range-separated GDF q batches changed reciprocal domain')
        count = int(batch.reciprocal_vector_count)
        factors.extend(batch.factors)
        peak = max(peak, int(batch.reserved_peak_bytes))
        sizes.append(len(selected))
        first += len(selected)
        del batch
    return dict(factors=factors, reciprocal_vector_count=count, batch_sizes=tuple(sizes),
                reserved_peak_bytes=peak, timing=_elapsed(start))


def _execute(prepare, world):
    from .mpi import KPointPartition, mpi_allgather
    from .periodic_k_gdf import _RangeSeparatedGdfCache
    from .banner import loaded_process_provenance

    def attest_process():
        value = loaded_process_provenance()
        if any(not isinstance(value.get(key), str) or not value[key] for key in
               ('release', 'source_revision', 'native_core_sha256')):
            raise ValueError('known loaded release, source revision and native core identities are required')
        return value
    provenance = _collective_call('loaded process provenance', attest_process)
    provenances = mpi_allgather(provenance)
    if any(item != provenance for item in provenances):
        raise RuntimeError('SR q MPI ranks loaded different releases, source revisions or native cores')

    # The callback returns a plan instead of entering the serial native loop.
    # The SCF wrapper reaches it only after its retained-state admission.
    def capture(**plan):
        return plan
    capture.world_size = world.size
    plan = _collective_call('source/domain preparation', lambda: prepare(capture))
    _collective_call('resolved numerical domain', lambda: _validate_plan(plan))
    domain = _collective_call('domain encoding', lambda: json.dumps(_domain(plan), sort_keys=True,
                                                                  separators=(',', ':'), allow_nan=False))
    fingerprints = mpi_allgather(sha256(domain.encode()).hexdigest())
    if len(set(fingerprints)) != 1:
        raise RuntimeError('SR q MPI source/domain differs between ranks')
    fingerprint = fingerprints[0]
    runtime = _collective_call('thread admission', _thread_admission)
    runtimes = mpi_allgather(runtime)
    budgets = {item['total_thread_budget'] for item in runtimes}
    if len(budgets) != 1 or sum(item['omp_threads'] for item in runtimes) > min(budgets):
        raise RuntimeError('SR q MPI rank teams exceed or disagree on the total thread budget')
    keys = sorted(plan['groups'])
    costs, profile = _collective_call('cost profile', lambda: _profile(fingerprint, len(keys)))
    all_costs = mpi_allgather((costs, profile))
    if any(item != (costs, profile) for item in all_costs):
        raise RuntimeError('SR q MPI ranks disagree on measured q costs')
    local_caps = _collective_call('memory caps', lambda: (int(plan['memory_byte_cap']),
                                                         int(plan['native_workspace_byte_cap'])))
    caps = mpi_allgather(local_caps)
    cap, workspace = min(item[0] for item in caps), min(item[1] for item in caps)
    # Object allgather retains local arrays while serializing its send parcel,
    # receiving all parcels and constructing NumPy arrays. Reserve six full
    # cache bounds for this overlap, including duplicated gathered storage.
    # During builds only one q's native workspace is live on each rank.
    gathered_cache_reservation = 6 * int(plan['cache_reservation'])
    communication_metadata = (64*1024*world.size + 1024*len(keys)
                              + 8*(len(domain.encode()) + len(json.dumps(runtimes).encode())))
    replicated_reservation = gathered_cache_reservation + communication_metadata
    build_cap = cap - replicated_reservation
    def admit():
        if workspace <= 0 or build_cap <= workspace:
            raise MemoryError('replicated gather/cache plus one native workspace exceeds memory cap')
    _collective_call('replicated-memory admission', admit)
    partition = _collective_call('q partition', lambda: KPointPartition.create(
        len(keys), strategy='weighted' if costs is not None else 'block', costs=costs))
    def build_local():
        start = _stamp()
        values = [_build_group(plan, keys[index], build_cap, workspace)
                  for index in partition.local_indices]
        return values, _elapsed(start)
    local, local_build = _collective_call('q build', build_local)
    # Serialization can itself fail locally. Check it before anyone enters
    # ordered gather, so an exception cannot leave its peers waiting there.
    def validate_parcel():
        if len(local) != partition.n_local:
            raise ValueError('q result count does not match ownership')
        for index, value in zip(partition.local_indices, local):
            expected = len(plan['groups'][keys[index]][1])
            if len(value['factors']) != expected:
                raise ValueError('q factor count does not match the ordered bra pairs')
            shapes = {factor.shape for factor in value['factors']}
            if len(shapes) != 1:
                raise ValueError('q factors do not share one retained metric frame')
            shape = next(iter(shapes))
            if (len(shape) != 3 or not 1 <= shape[0] <= int(plan['auxiliary'].nbasis)
                    or shape[1:] != (int(plan['basis'].nbasis),)*2
                    or any(factor.dtype != np.complex128 for factor in value['factors'])):
                raise ValueError('q factors do not match the admitted auxiliary/AO dimensions')
            if any(not np.isfinite(factor).all() for factor in value['factors']):
                raise ValueError('q build returned nonfinite factors')
        pickle.dumps(local, protocol=pickle.HIGHEST_PROTOCOL)
    _collective_call('gather parcel admission', validate_parcel)
    def gather_ordered():
        start = _stamp()
        values = partition.allgather_ordered(local)
        return values, _elapsed(start)
    ordered, gather = _collective_call('ordered gather', gather_ordered)
    local_retained = sum(factor.nbytes for value in local for factor in value['factors'])
    local_native_peak = max((value['reserved_peak_bytes'] for value in local), default=0)
    del local
    def assemble():
        factors = {}
        for key, value in zip(keys, ordered):
            for pair, factor in zip(plan['groups'][key][1], value['factors']):
                factors[pair] = factor
        return factors
    factors = _collective_call('ordered cache assembly', assemble)
    retained = sum(factor.nbytes for factor in factors.values())
    local_telemetry = dict(rank=world.rank, process_provenance=provenance,
                           owned_q_indices=list(partition.local_indices),
                           build=local_build, ordered_gather=gather, runtime=runtime,
                           memory=dict(cap_bytes=cap, native_workspace_cap_bytes=workspace,
                                       local_source_cap_bytes=local_caps[0],
                                       local_native_workspace_cap_bytes=local_caps[1],
                                       scf_admission=plan.get('scf_memory_admission', 'not measured'),
                                       source_cache_bound_bytes=int(plan['cache_reservation']),
                                       replicated_cache_gather_reservation_bytes=gathered_cache_reservation,
                                       communication_metadata_reservation_bytes=communication_metadata,
                                       local_retained_factor_bytes=local_retained,
                                       gathered_factor_bytes=retained,
                                       local_native_reserved_peak_bytes=local_native_peak,
                                       reserved_peak_bytes=replicated_reservation + local_native_peak))
    telemetry = dict(schema='vibeqc.sr-q-mpi/v1', world_size=world.size, domain_sha256=fingerprint,
                     domain=json.loads(domain), strategy=partition.strategy, measured_q_costs=costs,
                     cost_profile=profile,
                     task_counts=[sum(partition.owner_of(i) == rank for i in range(len(keys)))
                                  for rank in range(world.size)],
                     q_ownership=[partition.owner_of(i) for i in range(len(keys))],
                     q_build=[value['timing'] for value in ordered], ranks=mpi_allgather(local_telemetry),
                     aggregate_gathered_factor_bytes=world.size*retained,
                     clocks={name:vars(time.get_clock_info(name)) for name in ('perf_counter', 'time')})
    return _RangeSeparatedGdfCache(
        factors, replicated_reservation+max((v['reserved_peak_bytes'] for v in ordered), default=0),
        retained, tuple(v['reciprocal_vector_count'] for v in ordered),
        tuple(v['batch_sizes'] for v in ordered), plan['source_parameters'], plan['source_signature'],
        tuple(tuple(float(v) for v in row) for row in plan['kpoints']), plan['auxiliary'], cap, workspace,
        execution=telemetry)
