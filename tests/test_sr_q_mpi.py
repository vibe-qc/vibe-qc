"""Experimental SR q ownership, collective refusal and failure protocol."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_measured_cost_partition_preserves_global_indices(monkeypatch):
    from vibeqc import mpi
    costs = [8., 7., 6., 5., 4., 3., 2., 1.]
    partitions = []
    for rank in range(2):
        monkeypatch.setattr(mpi, 'mpi_world', lambda rank=rank: SimpleNamespace(rank=rank, size=2))
        partitions.append(mpi.KPointPartition.create(8, strategy='weighted', costs=costs))
    assert partitions[0].owners == partitions[1].owners
    assert [sum(costs[i] for i in part.local_indices) for part in partitions] == [18., 18.]
    assert sorted(i for part in partitions for i in part.local_indices) == list(range(8))
    for part in partitions:
        assert tuple(sorted(part.local_indices)) == part.local_indices
        assert all(part.owner_of(i) == part.rank for i in part.local_indices)


@pytest.mark.parametrize('costs', [None, [1.], [1., -1.], [np.nan, 1.], [np.inf, 1.]])
def test_weighted_partition_refuses_invalid_costs(costs):
    from vibeqc.mpi import KPointPartition
    with pytest.raises(ValueError, match='weighted costs'):
        KPointPartition.create(2, strategy='weighted', costs=costs)


def test_measured_cost_profile_binds_numerical_domain(tmp_path, monkeypatch):
    from vibeqc.periodic_sr_mpi import _profile_costs
    profile = tmp_path/'costs.json'
    profile.write_text(json.dumps(dict(domain_sha256='domain-a', q_build_seconds=[3., 1.])))
    monkeypatch.setenv('VIBEQC_SR_Q_COST_PROFILE', str(profile))
    assert _profile_costs('domain-a', 2) == (3., 1.)
    with pytest.raises(ValueError, match='different source/domain'):
        _profile_costs('domain-b', 2)
    with pytest.raises(ValueError, match='one finite nonnegative'):
        _profile_costs('domain-a', 3)


_WORKER = r'''
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
import vibeqc as vq
from vibeqc import mpi, periodic_k_gdf as gdf, periodic_sr_mpi as parallel
from vibeqc.aux_basis import make_aux_basis_set
world = mpi.mpi_world()
mode = sys.argv[1]
system = vq.PeriodicSystem(3, np.eye(3)*6., [vq.Atom(1,[0.,0.,0.]),vq.Atom(1,[1.4,0.,0.])])
basis = vq.BasisSet(system.unit_cell_molecule(),'sto-3g')
auxiliary = make_aux_basis_set(system.unit_cell_molecule(), aux_name='def2-svp-jk')
points = np.asarray([[0.,0.,0.],[np.pi/6.,0.,0.]])
if mode == 'success_serial_root':
    os.environ['VIBEQC_SR_Q_MPI'] = '0'
if mode == 'domain' and world.rank == 1:
    points[1,0] *= 1.01
if mode == 'source' and world.rank == 1:
    system = vq.PeriodicSystem(3, np.eye(3)*6.01, [vq.Atom(1,[0.,0.,0.]),vq.Atom(1,[1.4,0.,0.])])
if (mode == 'provenance' and world.rank == 1) or mode == 'missing_provenance':
    from importlib import import_module
    banner = import_module('vibeqc.banner')
    original_provenance = banner.loaded_process_provenance
    def different_provenance():
        return dict(original_provenance(), source_revision=None if mode=='missing_provenance' else 'different-source')
    banner.loaded_process_provenance = different_provenance
if mode == 'threads' and world.rank == 1:
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
if mode == 'flag' and world.rank == 1:
    os.environ['VIBEQC_SR_Q_MPI'] = '0'
options = dict(omega=.7, pair_cutoff=2., auxiliary_cutoff=2.,ke_cutoff=1.,
               linear_dep_thr=1e-9, memory_byte_cap=20*1024**2,
               native_workspace_byte_cap=64*1024, image_candidate_cap=10000,
               reciprocal_candidate_cap=10000)
if mode == 'success_bras':
    options['bra_rows'] = [1,0]
if mode == 'memory' and world.rank == 1:
    options['memory_byte_cap'] = 1
if mode == 'gather_memory' and world.rank == 1:
    options['memory_byte_cap'] = 3*(16*4*auxiliary.nbasis*basis.nbasis**2+4096+512*4+256*2)
if mode == 'adapter':
    options['pair_cache_builder'] = lambda *args: None
def build(plan,key,cap,workspace):
    if mode == 'build' and world.rank == 0:
        raise ArithmeticError('injected rank-local native failure')
    pairs = plan['groups'][key][1]
    result = dict(factors=[np.full((auxiliary.nbasis,2,2),i*2+j+1,dtype=complex) for i,j in pairs],
                reciprocal_vector_count=7,batch_sizes=(len(pairs),),reserved_peak_bytes=100000,
                timing=dict(perf_counter_elapsed=0.,realtime_elapsed=0.,load_start=[],load_end=[]))
    if mode == 'serialize' and world.rank == 1:
        result['unserializable'] = lambda: None
    if mode == 'shape' and world.rank == 1:
        result['factors'][0] = result['factors'][0][:,:,:1]
    return result
parallel._build_group = build
if mode == 'success_weighted':
    plan = gdf._build_range_separated_lpq_cache.__wrapped__(
        system,basis,auxiliary,points,True,**options,_q_executor=lambda **plan:plan)
    domain = json.dumps(parallel._domain(plan),sort_keys=True,separators=(',',':'),allow_nan=False)
    profile = Path(sys.argv[0]).parent/f'costs-{world.rank}.json'
    profile.write_text(json.dumps(dict(domain_sha256=hashlib.sha256(domain.encode()).hexdigest(),
                                      q_build_seconds=[1.,5.])))
    os.environ['VIBEQC_SR_Q_COST_PROFILE'] = str(profile)
try:
    if mode in ('preflight_memory', 'preflight_source', 'flag'):
        from types import SimpleNamespace
        from vibeqc import memory
        def check(*args, **kwargs):
            if mode == 'preflight_memory' and world.rank == 1:
                raise MemoryError('injected rank-local source preflight refusal')
        memory.check_periodic_gdf_memory = check
        gdf._preflight_gdf_lpq_memory(
            SimpleNamespace(info=lambda *args:None), n_basis=2, n_aux=auxiliary.nbasis,
            n_kpoints=2, need_k_pairs=True, open_shell=False, route_label='test',
            _sr_q=not (mode == 'preflight_source' and world.rank == 1))
        raise AssertionError('preflight failure was not collective')
    if mode == 'success_serial_root':
        result = None
        if world.rank == 0:
            from importlib import import_module
            from types import SimpleNamespace
            def native_leaf(system,basis,auxiliary,bras,q,**kwargs):
                factors=[]
                for bra in bras:
                    i=int(np.argmin(np.linalg.norm(points-np.asarray(bra),axis=1)))
                    j=i if np.linalg.norm(q)<1e-12 else 1-i
                    factors.append(np.full((auxiliary.nbasis,2,2),i*2+j+1,dtype=complex))
                return SimpleNamespace(factors=np.asarray(factors), reciprocal_vector_count=7,
                                       reserved_peak_bytes=100000)
            import_module('vibeqc.aux_basis')._build_lpq_range_separated_shared_q = native_leaf
            original_raw = gdf._build_range_separated_lpq_cache
            result = original_raw(system,basis,auxiliary,points,True,**options)
            gdf._build_range_separated_lpq_cache = lambda *args,**kwargs:result
            opts=vq.PeriodicRHFOptions()
            scf_cache=gdf._build_scf_range_separated_lpq_cache(
                system,basis,auxiliary,points,True,omega=.7,raw_integral_error=1e-7,
                ke_cutoff=1.,linear_dep_thr=1e-9,lat_opts=opts.lattice_opts,
                fit_screen_threshold=0.,options=opts,open_shell=False,progress=None)
            assert scf_cache is result
            gdf._preflight_gdf_lpq_memory(
                SimpleNamespace(info=lambda *args:None),n_basis=2,n_aux=auxiliary.nbasis,
                n_kpoints=2,need_k_pairs=True,open_shell=False,route_label='test')
    else:
        result = gdf._build_range_separated_lpq_cache(system,basis,auxiliary,points,True,**options)
    assert mode.startswith('success')
    if result is None:
        observation = dict(state='idle')
    else:
        for (i,j),factor in result.items():
            np.testing.assert_array_equal(factor,np.full_like(factor,i*2+j+1))
        observation = dict(state='ok', order=list(result), execution=result.execution)
except RuntimeError as exc:
    observation = dict(state='unexpected-error' if mode.startswith('success') else 'refused',error=str(exc))
except BaseException as exc:
    # A fixture error after entering the test must not strand an idle rank
    # at the observation gather; the parent still fails on this state.
    observation = dict(state='unexpected-error',error=f'{type(exc).__name__}: {exc}')
observations = mpi.mpi_allgather(observation)
if world.rank == 0:
    print('MPI_RESULT='+json.dumps(observations))
'''


@pytest.mark.parametrize('mode,ranks', [('success_serial_root',2), ('success',2), ('success',4), ('success_bras',2), ('success_weighted',2), ('domain',2),
                                       ('source',2), ('provenance',2), ('missing_provenance',2), ('threads',2),
                                       ('preflight_memory',2), ('preflight_source',2),
                                       ('memory',2), ('gather_memory',2), ('build',2),
                                       ('serialize',2), ('shape',2), ('adapter',2), ('flag',2)])
def test_real_mpi_collective_protocol(tmp_path, mode, ranks):
    launcher = shutil.which('mpiexec')
    if launcher is None or importlib.util.find_spec('mpi4py') is None:
        pytest.skip('optional mpiexec and mpi4py are required')
    worker = tmp_path/'worker.py'
    worker.write_text(textwrap.dedent(_WORKER))
    env = dict(os.environ, VIBEQC_SR_Q_MPI='1', VIBEQC_SR_Q_TOTAL_THREADS='8', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               BLIS_NUM_THREADS='1', OMPI_MCA_rmaps_base_oversubscribe='1',
               OMPI_MCA_hwloc_base_binding_policy='none')
    env.pop('VIBEQC_SR_Q_COST_PROFILE', None)
    command = [launcher, '-n', str(ranks), sys.executable, str(worker), mode]
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        # This test created the launcher session. Ask the launcher to tear
        # down its ranks before escalating; never match unrelated processes.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        (tmp_path/'stdout.log').write_text(stdout)
        (tmp_path/'stderr.log').write_text(stderr)
        pytest.fail(f'MPI collective timed out: {stdout}\n{stderr}')
    (tmp_path/'stdout.log').write_text(stdout)
    (tmp_path/'stderr.log').write_text(stderr)
    assert process.returncode == 0, stdout+'\n'+stderr
    row = next(line.removeprefix('MPI_RESULT=') for line in stdout.splitlines()
               if line.startswith('MPI_RESULT='))
    results = json.loads(row)
    assert len(results) == ranks
    if mode == 'success_serial_root':
        assert results[0]['state'] == 'ok' and results[1]['state'] == 'idle'
        assert results[0]['execution'] == {}
        assert results[0]['order'] == [[0,1],[1,0],[0,0],[1,1]]
        return
    if mode.startswith('success'):
        assert all(item['state'] == 'ok' for item in results)
        assert all(item['order'] == results[0]['order'] for item in results)
        expected = [[1,0],[0,1],[1,1],[0,0]] if mode=='success_bras' else [[0,1],[1,0],[0,0],[1,1]]
        assert results[0]['order'] == expected
        telemetry = results[0]['execution']
        if mode == 'success_weighted':
            assert telemetry['q_ownership'] == [1,0]
        assert sum(telemetry['task_counts']) == len(telemetry['q_ownership'])
        if ranks == 4:
            assert 0 in telemetry['task_counts']
        assert telemetry['aggregate_gathered_factor_bytes'] == ranks*telemetry['ranks'][0]['memory']['gathered_factor_bytes']
    else:
        assert all(item['state'] == 'refused' for item in results)
        assert len({item['error'] for item in results}) == 1
