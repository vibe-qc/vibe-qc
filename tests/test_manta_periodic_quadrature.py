"""Bounded atomic quadrature with independent periodic integration witnesses."""
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta


def case(radial=8,order=5,stratmann=False):
    lattice=np.array([[6.,.3,.1],[.4,7.,.2],[.1,.2,8.]])
    system=vq.PeriodicSystem(3,lattice,[vq.Atom(1,[2.,3.,4.]),vq.Atom(2,[3.4,3.,4.])])
    o=manta.PeriodicQuadratureOptions();o.radial_points=radial;o.lebedev_order=order
    o.partition_radius_bohr=7.;o.stratmann=stratmann
    b=manta.ResourceBudget();b.memory_bytes=2**30;b.scalar_terms=10**18;b.kernel_terms=2**64-1
    return system,o,b


def independent_order5(system,o):
    # Degree-five spherical rule: six axial nodes of weight 4pi/15 and
    # eight cube vertices of weight 3pi/10. No native grid data used.
    angular=np.vstack([np.eye(3),-np.eye(3),np.array(list(itertools.product([-1,1],repeat=3)))/np.sqrt(3)])
    aw=4*np.pi*np.array([1/15]*6+[3/40]*8)
    lattice=system.lattice;origins=np.array([a.xyz for a in system.unit_cell])
    origins=(origins@np.linalg.inv(lattice).T%1)@lattice.T
    labels=np.array(list(itertools.product(range(-6,7),repeat=3)))
    centers=(origins[None]+(labels@lattice.T)[:,None,:]).reshape(-1,3)
    home=len(origins)*np.flatnonzero(np.all(labels==0,axis=1))[0]
    points=[];weights=[]
    for owner,atom in enumerate(system.unit_cell):
        xi={1:.8,2:.9}[atom.Z]
        theta=np.arange(1,o.radial_points+1)*np.pi/(o.radial_points+1);x=np.cos(theta)
        radial=xi/np.log(2)*(1+x)**.6*np.log(2/(1-x))
        derivative=xi/np.log(2)*(.6*(1+x)**(-.4)*np.log(2/(1-x))+(1+x)**.6/(1-x))
        rw=np.pi/(o.radial_points+1)*np.sin(theta)*radial**2*derivative
        for r,w in zip(radial,rw):
            for direction,weight in zip(angular,aw):
                point=origins[owner]+r*direction;dist=np.linalg.norm(centers-point,axis=1)
                radius=max(o.partition_radius_bohr,2*dist.min());partition=0.
                if r<=radius:
                    keep=np.flatnonzero(dist<=radius);positions=centers[keep];d=dist[keep]
                    separation=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)
                    np.fill_diagonal(separation,1.)
                    mu=(d[:,None]-d[None,:])/separation
                    if o.stratmann:
                        z=np.clip(mu/.64,-1,1);switch=.5*(1-z/16*(35-z*z*(35-z*z*(21-5*z*z))))
                    else:
                        for _ in range(3):mu=(3*mu-mu**3)/2
                        switch=(1-mu)/2
                    np.fill_diagonal(switch,1.);products=np.prod(switch,axis=1)
                    partition=products[np.flatnonzero(keep==home+owner)[0]]/products.sum()
                points.append((np.linalg.solve(lattice,point)%1)@lattice.T);weights.append(w*weight*partition)
    return np.array(points),np.array(weights)


def sorted_grid(points,weights):
    # Native Lebedev ordering is intentionally not prescribed by this oracle.
    order=np.lexsort(np.round(points,10).T[::-1]);return points[order],weights[order]


@pytest.mark.parametrize('stratmann',[False,True])
def test_independent_radial_angular_partition_and_fold(stratmann):
    s,o,b=case(stratmann=stratmann);r=manta.build_periodic_quadrature(s,o,b)
    p,w=independent_order5(s,o);got,gw=sorted_grid(r.points,r.weights);p,w=sorted_grid(p,w)
    np.testing.assert_allclose(got,p,atol=3e-13,rtol=0)
    np.testing.assert_allclose(gw,w,atol=2e-12,rtol=2e-13)
    assert r.plan.grid_points==len(p)==2*8*14
    assert r.receipt.weight_sum==pytest.approx(w.sum(),abs=3e-12)
    assert r.receipt.nonzero_weights==np.count_nonzero(w)
    assert r.receipt.cell_volume==pytest.approx(abs(np.linalg.det(s.lattice)))


def test_shared_producer_ownership_and_protocol_identity():
    s,o,b=case(radial=12,order=29);r=manta.build_periodic_quadrature(s,o,b)
    go=vq.GridOptions();go.angular='lebedev';go.n_radial=o.radial_points;go.lebedev_order=o.lebedev_order
    raw=vq.build_periodic_becke_grid(s,grid_options=go,image_radius_bohr=o.partition_radius_bohr)
    folded=(np.asarray(raw.points)@np.linalg.inv(s.lattice).T%1)@s.lattice.T
    np.testing.assert_allclose(r.points,folded,atol=5e-14,rtol=0)
    np.testing.assert_allclose(r.weights,raw.weights,atol=3e-13,rtol=0)
    points=r.points;weights=r.weights;points[:]=np.nan;weights[:]=np.nan
    assert np.isfinite(r.points).all() and np.isfinite(r.weights).all()
    identity=r.receipt.source_identity
    assert manta.build_periodic_quadrature(s,o,b).receipt.source_identity==identity
    o.stratmann=True
    assert manta.build_periodic_quadrature(s,o,b).receipt.source_identity!=identity
    # Charge and multiplicity do not define a geometry quadrature.
    s.charge=-2147483648;s.multiplicity=2147483647
    assert np.isfinite(manta.build_periodic_quadrature(s,o,b).weights).all()


@pytest.mark.parametrize('field',['memory_bytes','scalar_terms','kernel_terms'])
def test_exact_and_one_short_resource_caps(field):
    s,o,b=case();p=manta.plan_periodic_quadrature(s,o,b)
    limits=dict(memory_bytes=p.modeled_peak_bytes,scalar_terms=p.scalar_terms,kernel_terms=p.kernel_terms)
    for name,value in limits.items():setattr(b,name,value)
    assert manta.build_periodic_quadrature(s,o,b).plan.grid_points==p.grid_points
    setattr(b,field,limits[field]-1)
    for fn in (manta.plan_periodic_quadrature,manta.build_periodic_quadrature):
        with pytest.raises((ValueError,RuntimeError),match='budget'):fn(s,o,b)


def test_external_live_bytes_and_checked_overflow():
    s,o,b=case();p=manta.plan_periodic_quadrature(s,o,b);b.other_live_bytes=12345
    assert manta.plan_periodic_quadrature(s,o,b).modeled_peak_bytes==p.modeled_peak_bytes+12345
    b.other_live_bytes=2**64-1
    with pytest.raises(OverflowError,match='overflow'):manta.build_periodic_quadrature(s,o,b)


@pytest.mark.parametrize('name,value',[('radial_points',0),('radial_points',10001),('lebedev_order',6),
    ('partition_radius_bohr',0.),('partition_radius_bohr',-1.),('partition_radius_bohr',float('nan')),
    ('partition_radius_bohr',float('inf'))])
def test_unsupported_options(name,value):
    s,o,b=case();setattr(o,name,value)
    with pytest.raises(ValueError):manta.build_periodic_quadrature(s,o,b)


@pytest.mark.parametrize('kind',['dim','singular','ill-conditioned','nan','empty','coincident','coordinates'])
def test_unsupported_geometry(kind):
    s,o,b=case();a=s.lattice.copy();atoms=s.unit_cell
    if kind=='dim':s=vq.PeriodicSystem(2,a,atoms)
    if kind=='singular':a[:,2]=a[:,1];s=vq.PeriodicSystem(3,a,atoms)
    if kind=='ill-conditioned':a[:,2]*=1e-6;s=vq.PeriodicSystem(3,a,atoms)
    if kind=='nan':a[0,0]=np.nan;s=vq.PeriodicSystem(3,a,atoms)
    if kind=='empty':s=vq.PeriodicSystem(3,a,[])
    if kind=='coincident':s=vq.PeriodicSystem(3,a,[atoms[0],atoms[0]])
    if kind=='coordinates':s=vq.PeriodicSystem(3,a,[vq.Atom(1,[1e10,0.,0.])])
    with pytest.raises(ValueError):manta.build_periodic_quadrature(s,o,b)


def test_relabelled_nuclei_and_left_handed_lattice():
    s,o,b=case(radial=16,order=29);r=manta.build_periodic_quadrature(s,o,b)
    shifted=vq.PeriodicSystem(3,s.lattice,[vq.Atom(a.Z,np.array(a.xyz)+s.lattice@np.array([2,-1,3])) for a in s.unit_cell])
    q=manta.build_periodic_quadrature(shifted,o,b)
    np.testing.assert_allclose(q.points,r.points,atol=3e-13,rtol=0)
    np.testing.assert_allclose(q.weights,r.weights,atol=2e-12,rtol=0)
    left=s.lattice.copy();left[:,[0,1]]=left[:,[1,0]]
    q=manta.build_periodic_quadrature(vq.PeriodicSystem(3,left,s.unit_cell),o,b)
    np.testing.assert_allclose(q.points,r.points,atol=5e-13,rtol=0)
    np.testing.assert_allclose(q.weights,r.weights,atol=2e-12,rtol=0)


def xc_case(name='pbe',spins=2):
    from vibeqc import _vibeqc_core as core
    lattice=np.eye(3)*6.;atom=vq.Atom(2,[3.,3.,3.]);s=vq.PeriodicSystem(3,lattice,[atom])
    shells=[]
    for exponent in (.8,.25):
        shell=core.ShellInfo();shell.atom_index=0;shell.origin=atom.xyz;shell.l=0;shell.pure=True
        shell.exponents=[exponent];shell.coefficients=[1.];shells.append(shell)
    basis=vq.BasisSet(s.unit_cell_molecule(),shells,'quadrature-validation',False)
    k=np.array([[.13,0.,0.],[-.13,0.,0.]]);weights=np.array([.4,.6])
    d=np.array([[[1.,.12j],[-.12j,.4]],[[.7,-.08j],[.08j,.3]]],complex)
    d=np.repeat(d[:,None],2,axis=1)
    if spins==1:d=d.sum(axis=0,keepdims=True)
    b=manta.ResourceBudget();b.memory_bytes=2**30;b.scalar_terms=10**18;b.kernel_terms=2**64-1
    c=dict(basis=basis,functional=vq.Functional(name,spin=spins),lattice=lattice,
        images=np.array(list(itertools.product(range(-1,2),repeat=3)),float)@lattice.T,
        momenta=k,weights=weights,densities=d,grid_batch=128,budget=b)
    return s,c


def atomic_xc(radial,order,name='pbe',spins=2):
    s,c=xc_case(name,spins);o=manta.PeriodicQuadratureOptions();o.radial_points=radial;o.lebedev_order=order
    o.partition_radius_bohr=8.;grid=manta.build_periodic_quadrature(s,o,c['budget'])
    c.update(points=grid.points,quadrature_weights=grid.weights)
    return c,grid,manta.compute_periodic_semilocal_xc(**c)


def tensor_reference(c,order):
    from tests.test_manta_periodic_semilocal_xc import reference
    x,w=np.polynomial.legendre.leggauss(order);idx=np.array(list(itertools.product(range(order),repeat=3)))
    points=(x[idx]+1)/2@c['lattice'].T;weights=np.prod(w[idx]/2,axis=1)*abs(np.linalg.det(c['lattice']))
    energy=0.;potential=np.zeros_like(c['densities']);pop=np.zeros(len(c['densities']))
    # Independent polynomial Gaussian and libxc contractions, bounded Python
    # chunks for the witness only. No native AO evaluator or grid producer.
    for start in range(0,len(points),2048):
        q=dict(c,points=points[start:start+2048],quadrature_weights=weights[start:start+2048])
        e,v,n=reference(q,c['functional']);energy+=e;potential+=v;pop+=n
    return energy,potential,pop


def test_xc_energy_and_potential_refined_against_independent_cell_quadrature():
    c,_,coarse=atomic_xc(20,29)
    reference64=tensor_reference(c,64);reference80=tensor_reference(c,80)
    assert abs(reference64[0]-reference80[0])<1e-10
    np.testing.assert_allclose(reference64[1],reference80[1],atol=2e-10,rtol=0)
    c,_,fine=atomic_xc(80,95)
    e=abs(fine.receipt.energy-reference80[0]);v=np.max(abs(fine.values-reference80[1]))
    assert e<2e-6 and v<3e-6
    assert e<abs(coarse.receipt.energy-reference80[0])/10
    assert v<np.max(abs(coarse.values-reference80[1]))/10
    # The grid is a finite integration source, not an automatic tight profile.
    assert e>1e-8 and v>1e-8
    from tests.test_manta_periodic_semilocal_xc import reference
    same_e,same_v,same_n=0.,np.zeros_like(c['densities']),np.zeros(2)
    for start in range(0,len(c['points']),2048):
        q=dict(c,points=c['points'][start:start+2048],quadrature_weights=c['quadrature_weights'][start:start+2048])
        ei,vi,ni=reference(q,c['functional']);same_e+=ei;same_v+=vi;same_n+=ni
    assert fine.receipt.energy==pytest.approx(same_e,abs=2e-12)
    np.testing.assert_allclose(fine.values,same_v,atol=2e-12,rtol=0)
    np.testing.assert_allclose(fine.receipt.electrons_grid,same_n,atol=2e-12,rtol=0)


def test_grid_citation_routes(tmp_path):
    from vibeqc.output.citations import assemble
    from tests.test_citations import _plan
    got=assemble(_plan(tmp_path,basis='sto-3g'),numerics=['manta_periodic_quadrature','stratmann_grid'])
    keys={e.key for e in got.citations}
    assert {'becke_grid_1988','treutler_ahlrichs_1995','lebedev_laikov_1999','stratmann_grid_1996'}<=keys


def test_thread_count_accounting_and_identical_source(tmp_path):
    import json
    import os
    import subprocess
    import sys
    script='''import hashlib,json
from tests.test_manta_periodic_quadrature import case
from vibeqc import manta
s,o,b=case(radial=20,order=29)
r=manta.build_periodic_quadrature(s,o,b)
print(json.dumps(dict(threads=r.plan.threads,workspace=r.plan.workspace_bytes,
 identity=r.receipt.source_identity,bytes=hashlib.sha256(r.points.tobytes()+r.weights.tobytes()).hexdigest())))'''
    rows=[]
    for threads in (1,3):
        env=dict(os.environ,OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        r=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True,check=True)
        rows.append(json.loads(r.stdout))
    assert rows[0]['identity']==rows[1]['identity'] and rows[0]['bytes']==rows[1]['bytes']
    assert rows[1]['threads']>=rows[0]['threads']
    if rows[1]['threads']>rows[0]['threads']:assert rows[1]['workspace']>rows[0]['workspace']


def test_large_grid_refuses_before_payload_allocation():
    s,o,b=case();o.radial_points=10000;o.lebedev_order=95;b.memory_bytes=2**20
    with pytest.raises((ValueError,RuntimeError),match='memory budget'):manta.build_periodic_quadrature(s,o,b)
    s,o,b=case();b.shell_calls=0;b.primitive_quartets=0
    assert manta.build_periodic_quadrature(s,o,b).plan.grid_points==224


def test_analytic_periodized_gaussian_and_fourier_moment():
    errors=[]
    for radial,angular in ((20,29),(80,95)):
        c,g,_=atomic_xc(radial,angular);points=g.points;weights=g.weights
        labels=np.array(list(itertools.product(range(-2,3),repeat=3)))
        centers=3+labels@c['lattice'].T;integrals=np.zeros(2)
        for start in range(0,len(points),2048):
            p=points[start:start+2048];w=weights[start:start+2048]
            f=np.exp(-.8*np.sum((p[:,None]-centers[None])**2,axis=-1)).sum(axis=1)
            integrals += [w@f,w@(f*np.cos(2*np.pi*p[:,0]/6))]
        expected=(np.pi/.8)**1.5*np.array([1,-np.exp(-(2*np.pi/6)**2/3.2)])
        errors.append(abs(integrals-expected))
    assert np.max(errors[1])<3e-5
    assert np.all(errors[1]<errors[0]/10)
