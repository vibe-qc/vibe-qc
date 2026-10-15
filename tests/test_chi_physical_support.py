"""Small support contracts and independent point-charge/domain references.

Full electronic components and fixed-density J/K acceptance run separately on
an installed main runtime; these tests do not qualify a chemical model.
"""
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('chi_support_contract', ROOT / 'python/vibeqc/periodic/chi/support.py')
support = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = support
_spec.loader.exec_module(support)


@pytest.mark.parametrize('policy', support.POLICIES)
def test_explicit_policy(policy):
    assert support.resolve_support_policy(policy, backend='four_center', dimension=3) == policy


@pytest.mark.parametrize('policy', [True, False, 0, 'physical', [], {}])
def test_invalid_policy(policy):
    with pytest.raises(ValueError):
        support.resolve_support_policy(policy, backend='four_center', dimension=3)


@pytest.mark.parametrize('backend,dim,ecp', [('ri',3,False),('rijcosx',3,False),('four_center',2,False),('four_center',3,True)])
def test_unsupported_policy(backend,dim,ecp):
    with pytest.raises(NotImplementedError):
        support.resolve_support_policy('physical_pairs',backend=backend,dimension=dim,ecp=ecp)


def test_immutable_serialization():
    record = support.ChiExecutedSupport('{"policy":"physical_pairs","cells":[[0,0,0]]}')
    old = record.identity
    record.as_dict()['cells'].append([1,0,0])
    assert record.identity == old and record.as_dict()['cells'] == [[0,0,0]]


def test_cells_preserve_order_and_refuse_duplicates():
    cells = [SimpleNamespace(index=[1,0,0],r_cart=[5,0,0]),SimpleNamespace(index=[0,0,0],r_cart=[0,0,0])]
    assert [c['index'] for c in support._cells(cells)] == [(1,0,0),(0,0,0)]
    with pytest.raises(ValueError,match='duplicate'):
        support._cells(cells+cells)
    with pytest.raises(ValueError,match='integral'):
        support._cells([SimpleNamespace(index=[.5,0,0],r_cart=[0,0,0])])


@pytest.fixture
def core():
    core = pytest.importorskip('vibeqc._vibeqc_core')
    if not hasattr(core, 'ewald_nuclear_repulsion_with_support'):
        pytest.skip('matching native support recorder is required; unmeasured')
    return core


def system(core, lattice=None, positions=None):
    lat = np.array([[4.8,.3,.1],[.2,5.4,.4],[.1,.2,6.1]]) if lattice is None else lattice
    xyz = np.array([[.2,.3,.4],[1.5,.6,.8]]) if positions is None else positions
    return core.PeriodicSystem(3,lat,[core.Atom(1,row) for row in xyz])


def reference_ewald(lat, xyz, alpha, rcut, gcut):
    # Independent oversized integer cubes, no shared bounds or centering.
    grid = np.array(list(itertools.product(range(-7,8),repeat=3)))
    R = grid @ lat.T
    real=0.; terms=[]
    for a in xyz:
        for b in xyz:
            d = a-b+R; norms=np.linalg.norm(d,axis=1)
            keep=(norms>1e-14)&(norms<=rcut)
            terms.append(d[keep])
            real += sum(math.erfc(alpha*r)/r for r in norms[keep])*.5
    B=2*np.pi*np.linalg.inv(lat).T; G=grid@B.T; norm2=np.sum(G*G,axis=1)
    keep=(norm2>0)&(norm2<=gcut*gcut); G=G[keep]; norm2=norm2[keep]
    factors=np.exp(1j*(G@xyz.T)).sum(axis=1)
    volume=abs(np.linalg.det(lat))
    recip=2*np.pi/volume*np.sum(abs(factors)**2*np.exp(-norm2/(4*alpha*alpha))/norm2)
    return real+recip-alpha/math.sqrt(np.pi)*len(xyz)-np.pi*len(xyz)**2/(2*alpha*alpha*volume),terms,G


@pytest.mark.parametrize('change',['base','image','origin','permutation'])
@pytest.mark.parametrize('r,g',[(4.1,3.0),(7.3,4.5)])
def test_actual_ewald_trace_and_energy(core,change,r,g):
    lat=np.array([[4.8,.3,.1],[.2,5.4,.4],[.1,.2,6.1]])
    xyz=np.array([[.2,.3,.4],[1.5,.6,.8]])
    if change=='image': xyz[1]+=lat@np.array([2,-1,1])
    if change=='origin': xyz+=np.array([1.4,-.3,.7])
    if change=='permutation': xyz=xyz[::-1]
    opts=core.EwaldOptions();opts.alpha=.6;opts.real_cutoff_bohr=r;opts.recip_cutoff_bohr_inv=g
    observed=core.ewald_nuclear_repulsion_with_support(system(core,lat,xyz),opts)
    reference,terms,G=reference_ewald(lat,xyz,.6,r,g)
    assert observed.energy == pytest.approx(reference,abs=2e-13)
    assert observed.energy == pytest.approx(core.ewald_nuclear_repulsion(system(core,lat,xyz),opts),abs=2e-14)
    for actual,expected in zip(observed.real_pair_displacements,terms):
        # Compare physical vectors irrespective of image naming/order.
        assert sorted(map(tuple,np.round(actual,11))) == sorted(map(tuple,np.round(expected,11)))
    assert sorted(map(tuple,np.round(observed.reciprocal_vectors,11))) == sorted(map(tuple,np.round(G,11)))


@pytest.mark.parametrize('shift',[[0,0,0],[2,-1,1]])
@pytest.mark.parametrize('radius',[1.7,4.9,7.3])
def test_nuclear_source_labels_independent_cube(core,shift,radius):
    sys=system(core); lat=np.asarray(sys.lattice); xyz=np.array([a.xyz for a in sys.unit_cell])
    xyz[1]+=lat@shift; sys=system(core,lat,xyz)
    centers=np.array([[.5,.7,.9],[2.1,-.2,.1]])
    rows=core.physical_nuclear_source_indices(sys,centers,radius)
    for center,actual in zip(centers,rows):
        expected=[]
        for atom,position in enumerate(xyz):
            for n in itertools.product(range(-6,7),repeat=3):
                if np.linalg.norm(position+lat@np.array(n)-center)<=radius+1e-12*(1+radius):
                    expected.append((atom,*n))
        assert sorted(map(tuple,actual))==sorted(expected)


def test_native_midpoint_admission(core):
    for bad in (np.zeros((1,2)),np.array([[np.nan,0,0]])):
        with pytest.raises((ValueError,RuntimeError)):
            core.physical_nuclear_source_indices(system(core),bad,4.)


@pytest.mark.parametrize("gamma_only", [False, True])
@pytest.mark.parametrize("exact_exchange,full_range_exchange", [(False,False),(True,False),(True,True)])
def test_record_contains_executed_cells_zero_q_and_distinct_nuclear_domains(monkeypatch,gamma_only,exact_exchange,full_range_exchange):
    import types
    # Execute the recorder itself with tiny explicit producer values. This is
    # serialization coverage; the native tests above supply numerical oracles.
    package=types.ModuleType('vibeqc');package.__path__=[]
    core=SimpleNamespace(physical_nuclear_source_indices=lambda *args: [[[1,2,0,0]]])
    package._vibeqc_core=core
    auxiliary=types.ModuleType('vibeqc.aux_basis');auxiliary._basis_fingerprint=lambda basis:'basis-digest'
    monkeypatch.setitem(sys.modules,'vibeqc',package)
    monkeypatch.setitem(sys.modules,'vibeqc.aux_basis',auxiliary)
    monkeypatch.setattr(support,'__package__','vibeqc.periodic.chi')
    one=SimpleNamespace(values=(('cutoff_bohr',2.),))
    nuclear=SimpleNamespace(values=(('nuclear_cutoff_bohr',3.),))
    domain=SimpleNamespace(policy='physical_pairs',use_ewald_j_split=True,
        exchange_ewald_split=True,exchange_exxdiv='ewald',exact_zone_bohr=None,
        lattice_1e=one,lattice_2e=one,nuclear_erfc=nuclear,as_dict=lambda:{'policy':'physical_pairs'})
    cell=SimpleNamespace(index=[0,0,0],r_cart=[0,0,0]); block=SimpleNamespace(cells=[cell])
    sys_=SimpleNamespace(lattice=np.eye(3)*5,unit_cell=[SimpleNamespace(Z=1,xyz=[0,0,0])])
    basis=SimpleNamespace(shells=lambda:[SimpleNamespace(origin=[0,0,0])])
    j=SimpleNamespace(K_vectors=[[1,0,0]],cells_r_cart=[[0,0,0]])
    vn=SimpleNamespace(K_vectors=[[2,0,0]],cells_r_cart=[[0,0,0]])
    key=np.array([.5,0,0]).tobytes()
    x=SimpleNamespace(q_channels={key:SimpleNamespace(K_vectors=[[.5,0,0]])},
        _canonical_q=lambda q:(np.asarray(q,dtype=float).tobytes(),q))
    if gamma_only or not full_range_exchange: x=None
    kpoints = [[0,0,0]] if gamma_only else [[0,0,0],[.5,0,0]]
    nn=SimpleNamespace(energy=.1,alpha=.5,real_cutoff_bohr=4.,reciprocal_cutoff_bohr_inv=3.,
        real_pair_displacements=[[[5,0,0]]],reciprocal_vectors=[[3,0,0]])
    result=support.capture_direct_support(policy='physical_pairs',system=sys_,basis=basis,
        domain=domain,one_electron={'S':block,'T':block,'V_ne':block},density=block,fock=block,
        j_cache=j,x_cache=x,nuclear_support=nn,kpoints=kpoints,weights=[1/len(kpoints)]*len(kpoints),
        v_ne_cache=vn,exact_exchange=exact_exchange,full_range_exchange=full_range_exchange).as_dict()
    assert result['basis_sha256']=='basis-digest'
    if full_range_exchange:
        assert len(result['exchange_channels'])==(1 if gamma_only else 2)
        assert result['exchange_channels'][0]['q_fractional']==[0,0,0]
    else:
        assert result['exchange_channels']==[]
        assert result['exchange_transfer_fractional']==[]
        assert result['exact_exchange_applicability']==('active' if exact_exchange else 'inactive')
        assert result['full_range_exchange_applicability']=='inactive'
    assert result['nuclear_attraction']['source_atom_image_indices']==[[[1,2,0,0]]]
    assert result['nuclear_attraction']['reciprocal_vectors_bohr_inv']==[[2,0,0]]
    assert result['nuclear_repulsion']['reciprocal_vectors_bohr_inv']==[[3,0,0]]
    assert result['operator_cells']['S'][0]['index']==[0,0,0]
    domain.policy='historical_index_ball'
    with pytest.raises(RuntimeError,match='policies differ'):
        support.capture_direct_support(policy='physical_pairs',system=sys_,basis=basis,
            domain=domain,one_electron={},density=block,fock=block,j_cache=j,x_cache=x,
            nuclear_support=nn,kpoints=[],weights=[],v_ne_cache=vn)


def test_public_default_membership_and_exact_capacity(monkeypatch):
    import types
    # Recorded H2 public defaults. Independent oversized cubes discriminate the
    # old enclosure product from the entries actually retained in the record.
    lattice = np.array([[8.,.3,.1],[0.,8.5,.2],[0.,0.,9.]])
    xyz = np.array([[1.,1.,1.],[1.,1.,2.4]])
    indices = np.array(list(itertools.product(range(-6,7), repeat=3)))
    translations = indices @ lattice.T
    keep = np.linalg.norm(translations, axis=1) <= 16.4
    cells = [SimpleNamespace(index=n, r_cart=r) for n,r in zip(indices[keep],translations[keep])]
    expected_masks, expected_midpoints, ids = [], [], {}
    for g,cell in enumerate(cells):
        for p,a in enumerate(xyz):
            for q,b in enumerate(xyz):
                if np.linalg.norm(a-b-cell.r_cart) > 15.: continue
                # Keep v1's floating-point grouping for exact JSON identity.
                midpoint = tuple((a+(b+cell.r_cart))/2)
                if midpoint not in ids:
                    ids[midpoint] = len(expected_midpoints)
                    expected_midpoints.append(midpoint)
                expected_masks.append((p,q,g,ids[midpoint]))
    panels = []
    def selector(system_, centers, radius):
        panels.append(len(centers))
        rows = []
        for center in centers:
            rows.append([(atom,*map(int,n)) for atom,position in enumerate(xyz)
                         for n,r in zip(indices,translations)
                         if np.linalg.norm(position+r-center) <= radius+1e-12*(1+radius)])
        return rows
    package = types.ModuleType('vibeqc'); package.__path__ = []
    package._vibeqc_core = SimpleNamespace(
        pair_complete_lattice_cells=lambda *args:cells,
        physical_nuclear_source_indices=selector)
    monkeypatch.setitem(sys.modules,'vibeqc',package)
    monkeypatch.setattr(support,'__package__','vibeqc.periodic.chi')
    system_ = SimpleNamespace(lattice=lattice, unit_cell=[SimpleNamespace(xyz=r) for r in xyz])
    basis = SimpleNamespace(shells=lambda:[SimpleNamespace(origin=r) for r in xyz])
    pair = SimpleNamespace(cutoff_bohr=15.); nuclear = SimpleNamespace(nuclear_cutoff_bohr=25.)
    bound = lambda radius:math.prod(2*math.ceil(radius*np.linalg.norm(row))+3 for row in np.linalg.inv(lattice))
    assert 4*bound(16.4)*2*bound(25.) > support.MAX_SUPPORT_TERMS
    expected_sources = selector(system_, np.asarray(expected_midpoints), 25.)
    panels.clear()
    actual = support._nuclear_membership('physical_pairs',system_,xyz,cells,15.,25.)
    assert actual[:3] == (expected_masks,expected_midpoints,expected_sources)
    assert max(panels) <= 64 and sum(panels) == len(expected_midpoints)
    terms = len(expected_masks)+len(expected_midpoints)+sum(map(len,expected_sources))
    assert terms < support.MAX_SUPPORT_TERMS
    monkeypatch.setattr(support,'MAX_SUPPORT_TERMS',terms)
    support.DirectSupportRecorder('physical_pairs').preflight(
        system=system_,basis=basis,lattice=pair,nuclear=nuclear)
    monkeypatch.setattr(support,'MAX_SUPPORT_TERMS',terms-1)
    with pytest.raises(MemoryError,match='no physical domain was reduced'):
        support.DirectSupportRecorder('physical_pairs').preflight(
            system=system_,basis=basis,lattice=pair,nuclear=nuclear)
    assert pair.cutoff_bohr == 15. and nuclear.nuclear_cutoff_bohr == 25.


def test_native_capacity_counts_selected_labels(core):
    # Each row selects only one image. The former enclosing-box estimate
    # rejected these 80,000 rows, despite storing only 160,000 entries.
    sys_ = core.PeriodicSystem(3,np.eye(3),[core.Atom(1,[0,0,0])])
    centers = np.zeros((80_000,3))
    rows = core.physical_nuclear_source_indices(sys_,centers,0.)
    assert len(rows) == len(centers)
    assert all(list(map(tuple,row)) == [(0,0,0,0)] for row in rows)


@pytest.mark.parametrize('policy,historical',[('physical_pairs',False),('historical_index_ball',True)])
def test_library_wrapper_routes_policy_before_numerics(monkeypatch,policy,historical):
    import vibeqc as vq
    from vibeqc.periodic.chi import scf
    from vibeqc import _vibeqc_core as native
    monkeypatch.setattr(native,'physical_nuclear_source_indices',object(),raising=False)
    monkeypatch.setattr(native,'ewald_nuclear_repulsion_with_support',object(),raising=False)
    class Dispatched(Exception): pass
    seen={}
    def dispatch(*args,**kwargs):
        seen.update(kwargs)
        raise Dispatched
    monkeypatch.setattr(scf,'run_pbc_bipole_rhf',dispatch)
    sys_=vq.PeriodicSystem(3,np.eye(3)*6,[vq.Atom(1,[0,0,0]),vq.Atom(1,[1.4,0,0])])
    opts=vq.PeriodicRHFOptions(); original=opts.lattice_opts.pair_complete_1e
    with pytest.raises(Dispatched):
        scf.run_aiccm2026dev_b_rhf(sys_,SimpleNamespace(shells=lambda: []),(2,1,1),opts,support_policy=policy)
    assert seen['historical_domain'] is historical
    assert seen['_support_recorder'].policy==policy
    assert seen['use_fock_symmetry'] is False and seen['use_fock_symmetry_reduce'] is False
    assert opts.lattice_opts.pair_complete_1e==original


def test_library_wrapper_rejects_fitted_and_nondiagonal(monkeypatch):
    import vibeqc as vq
    from vibeqc.periodic.chi import scf
    def forbidden(*args,**kwargs): raise AssertionError('numerical dispatch reached')
    monkeypatch.setattr(scf,'run_krhf_periodic_gdf',forbidden)
    monkeypatch.setattr(scf,'run_pbc_bipole_rhf',forbidden)
    sys_=vq.PeriodicSystem(3,np.eye(3)*6,[vq.Atom(1,[0,0,0]),vq.Atom(1,[1.4,0,0])])
    with pytest.raises(NotImplementedError,match='four_center HF/KS'):
        scf.run_aiccm2026dev_b_rhf(sys_,object(),(2,1,1),backend='ri',support_policy='physical_pairs')
    with pytest.raises((ValueError,TypeError,NotImplementedError)):
        scf.run_aiccm2026dev_b_rhf(sys_,object(),[[2,1,0],[0,1,0],[0,0,1]],support_policy='physical_pairs')
