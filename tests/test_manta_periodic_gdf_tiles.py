"""Immutable, bounded shell source for periodic fitted consumers."""
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_gdf import case, independent, legacy
from tests.test_manta_periodic_fitting import CAPS
from tests.test_aux_eri_sr_lattice_enumeration import _weighted_gradient_fixture


def plan(a):
    args=dict(a);args.pop('q');args['vectors']=len(args['vectors']);args['momenta']=len(args.pop('ket_kpoints'))
    return manta.plan_periodic_gdf_tiles(**args)


def assemble(source):
    p=source.plan;out=np.zeros((p.momenta,p.n_aux,p.n_orb,p.n_orb),complex)
    counts=np.zeros(2,dtype=np.uint64)
    for indices in itertools.product(*(range(n) for n in source.shell_extents)):
        t=source.evaluate(*indices);s=t.shape;o=t.offsets
        assert t.values.size<=p.maximum_tile_elements
        out[:,o[1]:o[1]+s[1],o[2]:o[2]+s[2],o[3]:o[3]+s[3]]=t.values
        counts+=np.array([t.receipt.shell_calls,t.receipt.primitive_products],dtype=np.uint64)
        assert t.receipt.three_center_source_identity==source.three_center_source_identity
    assert counts[0]<=p.coupling_shell_calls and counts[1]<=p.coupling_primitive_products
    return out


@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('screen',[0.,1e-4])
def test_tiles_reconstruct_same_dense_source_and_independent_gaussian_blocks(gamma,screen):
    a=case(gamma);a['options'].integral_screen_error=screen
    s=manta.PeriodicGDFTiles(**a);r=manta.compute_periodic_gdf(**a)
    metric=s.metric();tensor=assemble(s);m,t,ng=legacy(a);im,it=independent(a)
    np.testing.assert_array_equal(metric.values,r.metric)
    np.testing.assert_array_equal(tensor,r.three_center)
    np.testing.assert_allclose(metric.values,m,atol=3e-12,rtol=2e-13)
    np.testing.assert_allclose(tensor,t,atol=3e-12,rtol=2e-13)
    np.testing.assert_allclose(metric.values,im,atol=screen+3e-12,rtol=2e-13)
    np.testing.assert_allclose(tensor,it,atol=screen+3e-12,rtol=2e-13)
    assert metric.receipt.reciprocal_vectors==ng and metric.receipt.finite_zero_removed==gamma
    assert metric.receipt.shell_calls<=s.plan.metric_shell_calls
    assert metric.receipt.primitive_products<=s.plan.metric_primitive_products
    for name in ('metric_source_identity','three_center_source_identity','source_identity'):
        assert getattr(s,name)==getattr(r.receipt,name)
    for name in ('orbital_basis_identity','auxiliary_basis_identity'):
        assert getattr(s,name)==getattr(r.receipt.short_range,name)


@pytest.mark.parametrize('angular,pure',[(1,True),(2,False),(3,True)])
def test_angular_signed_contraction_tiles_are_the_identical_finite_source(angular,pure):
    a=case();system,b,aux=_weighted_gradient_fixture(angular,orbital_angular=1)
    def change(basis):
        shells=basis.shells()
        for sh in shells:sh.pure=pure
        return vq.BasisSet(system.unit_cell_molecule(),shells,'gdf-tile-angular',True)
    indices=np.array(list(itertools.product(range(-1,2),repeat=3)))
    a.update(system=system,orbital=change(b),auxiliary=change(aux),vectors=indices@system.reciprocal_lattice().T+a['q'])
    a['options'].pair_cutoff_bohr=4.3;a['options'].auxiliary_cutoff_bohr=5.1
    source=manta.PeriodicGDFTiles(**a);dense=manta.compute_periodic_gdf(**a)
    np.testing.assert_array_equal(source.metric().values,dense.metric)
    np.testing.assert_array_equal(assemble(source),dense.three_center)


def test_source_snapshots_and_exports_are_owned_and_immutable():
    a=case(True);s=manta.PeriodicGDFTiles(**a);m=s.metric().values;t=s.evaluate(0,1,0).values;identity=s.source_identity
    for key in ('q','ket_kpoints','vectors'):a[key][:]=np.nan
    a['options'].omega=0.;a['options'].compute_metric=False
    a['system'].lattice=np.eye(3)*99.
    s.metric().values[:]=np.nan;s.evaluate(0,1,0).values[:]=np.nan
    np.testing.assert_array_equal(s.metric().values,m);np.testing.assert_array_equal(s.evaluate(0,1,0).values,t)
    assert s.source_identity==identity
    exported=s.evaluate(0,1,0);exported.shape[0]=999;exported.offsets[0]=999
    assert exported.shape[0]==2 and exported.offsets[0]==0


@pytest.mark.parametrize('field,quantity',CAPS)
@pytest.mark.parametrize('metric',[False,True])
def test_joint_construction_and_one_sweep_caps_before_numerical_copies(field,quantity,metric):
    a=case(True);a['options'].compute_metric=metric;a['budget'].other_live_bytes=23456;p=plan(a)
    for cap,value in CAPS:setattr(a['budget'],cap,getattr(p,value))
    source=manta.PeriodicGDFTiles(**a)
    if metric:source.metric()
    source.evaluate(0,0,0)
    setattr(a['budget'],field,getattr(p,quantity)-1)
    for key in ('q','ket_kpoints','vectors'):a[key][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):plan(a)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicGDFTiles(**a)


def test_tile_admission_does_not_require_the_full_auxiliary_ao_pair_tensor():
    a=case(True);system=vq.PeriodicSystem(3,8*np.eye(3),[vq.Atom(2,[0.,0.,0.])])
    def basis(count):
        shells=[core.ShellInfo(0,0,True,[.3+.02*i],[1.],[0.,0.,0.]) for i in range(count)]
        return vq.BasisSet(system.unit_cell_molecule(),shells,'tile-memory-census',True)
    a.update(system=system,orbital=basis(24),auxiliary=basis(80),vectors=np.zeros((1,3)),ket_kpoints=np.zeros((4,3)))
    for field,_ in CAPS:setattr(a['budget'],field,2**64-1)
    p=plan(a);a['budget'].memory_bytes=p.modeled_peak_bytes
    source=manta.PeriodicGDFTiles(**a)
    assert source.plan.maximum_tile_elements==4
    assert 16*4*80*24**2>10**6
    # Only construction is needed to prove the dense allocation is absent;
    # no large numerical traversal is started for this admission witness.
    with pytest.raises((ValueError,RuntimeError),match='memory budget'):
        manta.compute_periodic_gdf(**a)


def test_layout_caps_metric_reuse_and_optional_metric_keep_source_identities():
    a=case();s=manta.PeriodicGDFTiles(**a);t=s.evaluate(0,0,0).values
    for name in ('q','ket_kpoints','vectors'):
        same=manta.PeriodicGDFTiles(**dict(a,**{name:np.asfortranarray(a[name])}))
        assert same.source_identity==s.source_identity
        np.testing.assert_array_equal(same.evaluate(0,0,0).values,t)
    p=plan(a);a['budget'].other_live_bytes+=1234
    assert plan(a).modeled_peak_bytes==p.modeled_peak_bytes+1234
    a['options'].image_candidate_cap*=2
    assert manta.PeriodicGDFTiles(**a).source_identity==s.source_identity
    a['options'].compute_metric=False;only=manta.PeriodicGDFTiles(**a)
    assert only.metric_source_identity=='' and only.three_center_source_identity==s.three_center_source_identity
    with pytest.raises(ValueError,match='metric'):only.metric()
    np.testing.assert_array_equal(only.evaluate(0,0,0).values,t)
    a['options'].compute_metric=True;a['options'].pair_cutoff_bohr=4.9
    a['ket_kpoints']=a['ket_kpoints'][:1].copy();changed=manta.PeriodicGDFTiles(**a)
    assert changed.metric_source_identity==s.metric_source_identity and changed.three_center_source_identity!=s.three_center_source_identity
    np.testing.assert_array_equal(changed.metric().values,s.metric().values)


@pytest.mark.parametrize('fault',['domain','nonfinite','dtype','strides','empty','shape','index'])
def test_source_and_tile_refusals(fault):
    a=case()
    if fault=='domain':a['vectors'][0,0]+=.01
    elif fault=='nonfinite':a['q'][0]=np.nan
    elif fault=='dtype':a['vectors']=a['vectors'].astype(np.float32)
    elif fault=='strides':a['ket_kpoints']=a['ket_kpoints'][::-1]
    elif fault=='empty':a['vectors']=np.empty((0,3))
    elif fault=='shape':a['vectors']=np.zeros((2,4))
    if fault=='index':
        with pytest.raises(IndexError):manta.PeriodicGDFTiles(**a).evaluate(999,0,0)
    else:
        with pytest.raises((ValueError,RuntimeError,TypeError)):manta.PeriodicGDFTiles(**a)


@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('kernel',['coulomb','erf','erfc'])
def test_metric_assembly_audit_preserves_stored_source(gamma,kernel):
    a=case(gamma)
    physical=manta.PeriodicGDFKernelOptions();physical.kernel=kernel;physical.omega=0. if kernel=='coulomb' else .4
    a['physical_kernel']=physical
    control=manta.PeriodicGDFTiles(**a);raw=control.metric()
    a['options'].metric_assembly_rank=True
    source=manta.PeriodicGDFTiles(**a);audited=source.metric()
    np.testing.assert_array_equal(audited.values,raw.values)
    assert source.source_identity==control.source_identity
    bounds=np.array(audited.metric_assembly_error_bounds).reshape(raw.values.shape)
    assert np.all(np.isfinite(bounds)) and np.all(bounds>=0) and np.max(bounds)>0
    assert audited.receipt.metric_assembly_evaluated
    assert audited.receipt.maximum_metric_assembly_error==bounds.max()
    assert audited.receipt.metric_assembly_certificate_identity
    assert source.plan.metric_peak_bytes>control.plan.metric_peak_bytes
    assert source.plan.metric_scalar_terms>control.plan.metric_scalar_terms
    with pytest.raises(ValueError,match='tile source'):manta.compute_periodic_gdf(**a)


@pytest.mark.parametrize('scale',[1e-30,1.,1e30])
def test_metric_assembly_decimal_stored_operand_enclosure(scale):
    # A single origin image and Gamma zero-mode subtraction let Decimal
    # reconstruct the exact assembly with the conditional reciprocal weight.
    # This deliberately makes no claim about primitive/Fourier accuracy.
    from decimal import Decimal,localcontext
    from tests.test_manta_range import _request
    a=case(True);system=vq.PeriodicSystem(3,8*np.eye(3),[vq.Atom(2,[0.,0.,0.])])
    shells=[core.ShellInfo(0,0,True,[.3+.2*i,1.2],[scale,-.27*scale],[0.,0.,0.]) for i in range(2)]
    aux=vq.BasisSet(system.unit_cell_molecule(),shells,'assembly-reference',True)
    a.update(system=system,auxiliary=aux,vectors=np.zeros((1,3)))
    a['options'].auxiliary_cutoff_bohr=.1;a['options'].metric_assembly_rank=True
    source=manta.PeriodicGDFTiles(**a);actual=source.metric()
    sr=manta.FittingTiles([aux,aux],request=_request('erfc',a['options'].omega,2))
    ft=manta.compute_auxiliary_fourier_panel(aux,np.zeros((1,3))).values.reshape(2)
    weight=_decimal_gdf_weight(np.zeros(3),a['options'].omega,4*np.pi/512.,'coulomb',0.,True)
    bounds=np.array(actual.metric_assembly_error_bounds).reshape(2,2)
    with localcontext() as ctx:
        ctx.prec=250
        for i in range(2):
            for j in range(2):
                integral=float(sr.evaluate((i,j))['values'].item())
                exact=Decimal(integral)+Decimal(weight)*Decimal(float(ft[i].real))*Decimal(float(ft[j].real))
                error=abs(Decimal(float(actual.values[i,j].real))-exact)
                assert error<=Decimal(float(bounds[i,j]))
                assert actual.values[i,j].imag==0


@pytest.mark.parametrize('field,quantity',CAPS)
def test_metric_assembly_exact_caps_and_one_short_refusal(field,quantity):
    a=case(True);a['options'].metric_assembly_rank=True;p=plan(a)
    for cap,value in CAPS:setattr(a['budget'],cap,getattr(p,value))
    assert manta.PeriodicGDFTiles(**a).metric().receipt.metric_assembly_evaluated
    setattr(a['budget'],field,getattr(p,quantity)-1)
    a['vectors'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicGDFTiles(**a)


@pytest.mark.parametrize('scale',[1e-30,1.,1e30])
@pytest.mark.parametrize('momentum',[[.19,-.13,.27],[1e6+.19,.2,-1e6+.27]])
def test_metric_bloch_phase_decimal_contracted_sum(scale,momentum):
    from decimal import Decimal as D,localcontext
    from tests.test_manta_fourier import _decimal_phase
    from tests.test_manta_range import _request
    a=case(False);system=vq.PeriodicSystem(3,6*np.eye(3),[vq.Atom(2,[0.,0.,0.]),vq.Atom(2,[.2,-.1,.3])])
    shells=[core.ShellInfo(i,0,True,[.3+.2*i,1.2],[scale,-.8*scale],[.2*i,-.1*i,.3*i]) for i in range(2)]
    aux=vq.BasisSet(system.unit_cell_molecule(),shells,'phase-reference',True)
    a.update(system=system,auxiliary=aux,q=np.asarray(momentum),vectors=np.asarray([momentum]))
    physical=manta.PeriodicGDFKernelOptions();physical.kernel='erfc';physical.omega=a['options'].omega
    a['physical_kernel']=physical
    a['options'].auxiliary_cutoff_bohr=6.5;a['options'].metric_assembly_rank=True
    source=manta.PeriodicGDFTiles(**a);actual=source.metric();bounds=np.asarray(actual.metric_assembly_error_bounds).reshape(2,2)
    assert actual.receipt.metric_phase_included and actual.receipt.maximum_metric_phase_error>0
    translations=np.asarray(list(itertools.product(range(-2,3),repeat=3)),float)*6
    with localcontext() as ctx:
        ctx.prec=800
        for i,bra in enumerate(aux.shells()):
            for j,ket in enumerate(aux.shells()):
                real,imag=D(0),D(0)
                for t in translations:
                    if np.linalg.norm(np.asarray(bra.origin)-ket.origin-t)>6.5:continue
                    moved=aux.shells()[j];moved.origin=(np.asarray(moved.origin)+t).tolist()
                    left_shell=aux.shells()[i];left_shell.atom_index=0;moved.atom_index=0
                    left=vq.BasisSet(vq.Molecule([vq.Atom(2,left_shell.origin)]),[left_shell],'phase-bra',True)
                    right=vq.BasisSet(vq.Molecule([vq.Atom(2,moved.origin)]),[moved],'phase-ket',True)
                    value=manta.FittingTiles([left,right],request=_request('erfc',a['options'].omega,2)).evaluate((0,0))['values'].item()
                    re,im=_decimal_phase(a['q'],t);real+=D(float(value))*re;imag+=D(float(value))*im
                error=abs(D(float(actual.values[i,j].real))-real)+abs(D(float(actual.values[i,j].imag))-imag)
                assert error<=D(float(bounds[i,j]))
    a['options'].metric_assembly_rank=False
    np.testing.assert_array_equal(actual.values,manta.PeriodicGDFTiles(**a).metric().values)


def _decimal_gdf_weight(momentum,omega,prefactor,kernel,physical,zero):
    from decimal import Decimal as D,localcontext
    with localcontext() as ctx:
        ctx.prec=250
        p2=sum(D(float(x))**2 for x in momentum);c=D(float(prefactor));eta=D(float(omega));w=D(float(physical))
        if zero:
            return D(0) if kernel=='erf' else (-c/(4*eta*eta) if kernel=='coulomb' else c/4*(1/(w*w)-1/(eta*eta)))
        if p2<=D(1e-24):return D(0)
        if kernel=='coulomb':return c/p2*(-p2/(4*eta*eta)).exp()
        if kernel=='erf':return c/p2*(-p2/(4*w*w)).exp()
        return c/p2*((-p2/(4*eta*eta)).exp()-(-p2/(4*w*w)).exp())


@pytest.mark.parametrize('kernel',['coulomb','erf','erfc'])
@pytest.mark.parametrize('momentum,omega,physical,zero',[
    ([0.,0.,0.],.65,.4,True),([0.,0.,0.],.65,.65,True),
    ([.3,-.7,1.1],.65,.4,False),([.3,-.7,1.1],.65,.65,False),
    ([.3,-.7,1.1],.65,np.nextafter(.65,1.),False),
    ([.3,-.7,1.1],.65,np.nextafter(.65,0.),False),
    ([1e-13,0.,0.],.65,.4,False),([1e-11,0.,0.],.65,.4,False),
    ([40.,-.7,1.1],.65,.4,False),([1e100,0.,0.],.65,.4,False),
    ([1e-11,0.,0.],1e100,2e100,False),
])
def test_metric_weight_independent_decimal(momentum,omega,physical,zero,kernel):
    from decimal import Decimal as D,localcontext
    q=np.array(momentum);p=manta.PeriodicGDFKernelOptions();p.kernel=kernel;p.omega=physical
    prefactor=.123;exact=_decimal_gdf_weight(q,omega,prefactor,kernel,physical,zero)
    # A deliberately shifted witness tests the enclosure independently of a
    # presumed correct exp/expm1 implementation.
    evaluated=float(exact)+.01;c=manta.certify_periodic_gdf_weight(q,omega,prefactor,p,zero,evaluated)
    assert D(c.lower)<=exact<=D(c.upper)
    with localcontext() as ctx:
        ctx.prec=250;assert abs(D(evaluated)-exact)<=D(c.error_bound)
    assert np.isfinite(c.error_bound)
    with pytest.raises(AttributeError):c.lower=0.


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_metric_weight_admission_and_refusals(field,quantity):
    p=manta.PeriodicGDFKernelOptions();b=manta.ResourceBudget();plan=manta.plan_periodic_gdf_weight(b)
    setattr(b,field,getattr(plan,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.certify_periodic_gdf_weight(np.full(3,np.nan),.65,1.,p,False,0.,b)
    setattr(b,field,getattr(plan,quantity));assert manta.certify_periodic_gdf_weight(np.zeros(3),.65,1.,p,False,0.,b).error_bound==0
    for args in [(np.full(3,np.nan),.65,1.,False,0.),(np.ones(3),0.,1.,False,0.),(np.ones(3),.65,-1.,False,0.),(np.ones(3),.65,1.,False,np.inf),(np.ones(3),.65,1.,True,0.)]:
        q,w,c,z,v=args
        with pytest.raises(ValueError):manta.certify_periodic_gdf_weight(q,w,c,p,z,v,b)
    with pytest.raises(RuntimeError,match='singular boundary'):manta.certify_periodic_gdf_weight(np.array([1e-12,0.,0.]),.65,1.,p,False,0.,b)
    with pytest.raises((RuntimeError,OverflowError),match='positive'):manta.certify_periodic_gdf_weight(np.ones(3),1e-200,1.,p,False,0.,b)
    with pytest.raises((RuntimeError,OverflowError),match='overflow'):manta.certify_periodic_gdf_weight(np.full(3,1e200),.65,1.,p,False,0.,b)


@pytest.mark.parametrize('scale',[1e-30,1.,1e30])
@pytest.mark.parametrize('physical',[.4,1e-3])
def test_metric_weight_decimal_fourier_gram(scale,physical):
    from decimal import Decimal as D,localcontext
    a=case(False);system=vq.PeriodicSystem(3,8*np.eye(3),[vq.Atom(2,[.3,-.2,.1])])
    shells=[core.ShellInfo(0,0,True,[.3+.2*i,1.2],[scale,-.27*scale],[.3,-.2,.1]) for i in range(2)]
    aux=vq.BasisSet(system.unit_cell_molecule(),shells,'weight-reference',True)
    a.update(system=system,auxiliary=aux,vectors=np.array([a['q']]))
    p=manta.PeriodicGDFKernelOptions();p.kernel='erf';p.omega=physical;a['physical_kernel']=p
    a['options'].metric_assembly_rank=True;source=manta.PeriodicGDFTiles(**a);actual=source.metric()
    ft=manta.compute_auxiliary_fourier_panel(aux,a['vectors']).values.reshape(2)
    # Stored reciprocal prefactor is the explicitly declared reference operand.
    weight=_decimal_gdf_weight(a['q'],a['options'].omega,4*np.pi/512.,'erf',physical,False)
    bounds=np.asarray(actual.metric_assembly_error_bounds).reshape(2,2)
    assert actual.receipt.metric_weight_included
    with localcontext() as ctx:
        ctx.prec=250
        for i in range(2):
            for j in range(2):
                ai,bi=D(float(ft[i].real)),D(float(ft[i].imag));aj,bj=D(float(ft[j].real)),D(float(ft[j].imag))
                re=weight*(ai*aj+bi*bj);im=weight*(ai*bj-bi*aj)
                error=abs(D(float(actual.values[i,j].real))-re)+abs(D(float(actual.values[i,j].imag))-im)
                assert error<=D(float(bounds[i,j]))
    a['options'].metric_assembly_rank=False
    np.testing.assert_array_equal(actual.values,manta.PeriodicGDFTiles(**a).metric().values)
