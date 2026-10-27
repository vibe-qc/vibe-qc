"""Finite projected inverse errors: independent Decimal and fixed-family checks."""
from decimal import Decimal,localcontext
import dataclasses
import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta
from tests.test_manta_hermitian_inverse import arrays
from tests.test_manta_hermitian_spectrum import options


def action(a,cut=.1,perturb=0.):
    a,w,q,f=arrays(a,cut)
    if f.size:f[0,-1]*=complex(1+perturb,-perturb/2)
    return manta.HermitianInverseAction(a,w,q,f,options(cut)),f


def dc(z):return Decimal.from_float(float(z.real)),Decimal.from_float(float(z.imag))
def add(a,b):return a[0]+b[0],a[1]+b[1]
def mul(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def conj(a):return a[0],-a[1]
def sub(a,b):return a[0]-b[0],a[1]-b[1]
def dot(a,b):
    z=(Decimal(0),Decimal(0))
    for x,y in zip(a,b):z=add(z,mul(x,y))
    return z


def decimal_error(a,f,t,d,cut):
    with localcontext() as ctx:
        ctx.prec=100;D=Decimal.from_float
        aa,bb=D(float(a[0,0].real)),D(float(a[1,1].real));zr,zi=dc(a[0,1])
        gap=((aa-bb)**2+4*(zr*zr+zi*zi)).sqrt();lo,hi=(aa+bb-gap)/2,(aa+bb+gap)/2
        dl=1/lo if lo>D(cut) else Decimal(0);dh=1/hi if hi>D(cut) else Decimal(0)
        slope=(dh-dl)/gap
        c=[[(dl+slope*(aa-lo),Decimal(0)),(slope*zr,slope*zi)],
           [(slope*zr,-slope*zi),(dl+slope*(bb-lo),Decimal(0))]]
        w=[[dot([dc(x) for x in f[i]],[conj(dc(x)) for x in f[j]]) for j in range(2)] for i in range(2)]
        delta=[[sub(w[i][j],c[i][j]) for j in range(2)] for i in range(2)]
        k=[]
        for p in range(t.shape[1]):
            row=[]
            for q in range(t.shape[1]):
                v=[dot(delta[i],[dc(x) for x in t[:,q]]) for i in range(2)]
                row.append(dot([conj(dc(x)) for x in t[:,p]],v))
            k.append(row)
        j=[dot(row,[dc(x) for x in d]) for row in k]
        e=mul((Decimal('.5'),Decimal(0)),dot([conj(dc(x)) for x in d],j))
        return [(x*x+y*y).sqrt() for x,y in j],(e[0]*e[0]+e[1]*e[1]).sqrt()


@pytest.mark.parametrize('cut',[0.,1.,5.])
@pytest.mark.parametrize('perturb',[0.,.001])
@pytest.mark.parametrize('component',[False,True])
def test_decimal_projected_j_energy_and_arbitrary_factor(cut,perturb,component):
    a=np.array([[.4,.13+.29j],[.13-.29j,2.7]],complex);owner,f=action(a,cut,perturb)
    t=np.array([[.13+.3j,2.5,1-.1j],[1.1,-.3+.6j,.4]],complex)
    columns=[owner.column(np.ascontiguousarray(t[:,j])) for j in range(3)]
    h=[.2,.4,.6] if component else [];bound=owner.family(columns,1.5,h)
    d=np.array([.08+.1j,-.12+.2j,.2-.3j]);j,e=decimal_error(a,f,t,d,cut)
    assert bound.qualified
    assert all(x<=Decimal.from_float(y) for x,y in zip(j,bound.j_error_bounds))
    assert e<=Decimal.from_float(bound.energy_error_bound)
    if cut==5.:assert bound.j_error_bound==bound.energy_error_bound==0.
    doubled=owner.family(columns,3.,[2*x for x in h])
    assert doubled.j_error_bound==pytest.approx(2*bound.j_error_bound,rel=2e-14)
    assert doubled.energy_error_bound==pytest.approx(4*bound.energy_error_bound,rel=2e-14)
    zero=owner.family(columns,1.5,[0.]*3);assert zero.j_error_bound==zero.energy_error_bound==0.


def test_projected_bound_avoids_worst_inverse_direction():
    q=np.array([[1.,1.],[-1.,1.]])/np.sqrt(2);a=(q@np.diag([1e-8,2.])@q.T).astype(complex)
    owner,f=action(a,0.);t=np.ascontiguousarray(q[:,1],dtype=complex)
    col=owner.column(t);b=owner.family([col],1.)
    fallback=owner.certificate.inverse_error_bound*col.coupling_norm**2
    assert b.qualified and b.j_error_bound < fallback/1000


@pytest.mark.parametrize('name,field',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_all_public_resource_boundaries(name,field):
    a,w,q,f=arrays(np.diag([.2,2.]));o=options();budget=manta.ResourceBudget()
    p=manta.plan_hermitian_inverse_action(2,o,budget);setattr(budget,name,getattr(p,field)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.HermitianInverseAction(np.full_like(a,np.nan),w,q,f,o,budget)
    setattr(budget,name,getattr(p,field));owner=manta.HermitianInverseAction(a,w,q,f,o,budget)
    budget=manta.ResourceBudget();p=manta.plan_hermitian_inverse_column(2,budget);setattr(budget,name,getattr(p,field)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):owner.column(np.array([np.nan,0],complex),budget)
    setattr(budget,name,getattr(p,field));col=owner.column(np.ones(2,complex),budget)
    budget=manta.ResourceBudget();p=manta.plan_hermitian_inverse_family(2,1,budget);setattr(budget,name,getattr(p,field)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):owner.family([col],1.,[],budget)
    setattr(budget,name,getattr(p,field));assert owner.family([col],1.,[],budget).qualified


def test_rank_crossing_foreign_columns_and_input_guards():
    owner,_=action(np.diag([.1,2.]));col=owner.column(np.ones(2,complex));assert not owner.family([col],1.).qualified
    other,_=action(np.diag([.2,2.]))
    with pytest.raises(ValueError,match='different owner'):other.family([col],1.)
    for t in (np.ones(3,complex),np.ones(2),np.ones(4,complex)[::2]):
        with pytest.raises(ValueError,match='contiguous'):other.column(t)
    with pytest.raises(ValueError,match='finite'):other.column(np.full(2,np.nan,complex))
    col=other.column(np.ones(2,complex))
    with pytest.raises(ValueError,match='entry'):other.family([col],1.,[-1.])


def raw_case(rank=6,component=True):
    from tests.test_manta_periodic_ri_fitting_bounds import case
    a,old,ref,o,w=case(rank=rank,component=component)
    a['options'].coupling_multipole_error=0.;a['options'].coupling_multipole_pair_tree=False
    src=manta.PeriodicGDFTiles(**a);ri=manta.PeriodicRIQBuilder(src,2,old.options if hasattr(old,'options') else metric_options(old),a['budget'])
    o.metric_action_j_error=o.metric_action_energy_error=1e20
    return a,src,ri,ref,o,w


def metric_options(ri):
    o=manta.RIOptions();o.relative_cutoff=0.;o.absolute_cutoff=ri.metric_info.cutoff;return o


@pytest.mark.parametrize('rank',[0,2,6])
@pytest.mark.parametrize('component',[False,True])
def test_raw_periodic_bounds_controls_and_strict_targets(rank,component):
    from tests.test_manta_periodic_gdf_tiles import assemble
    from tests.test_manta_periodic_ri_q import densities
    a,src,ri,ref,o,w=raw_case(rank,component);bound=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']);c=bound.receipt
    assert c.metric_action_qualified and c.metric_action_columns==8
    t=assemble(src).transpose(1,0,2,3).reshape(ri.plan.n_auxiliary,-1)
    eig,u=np.linalg.eigh(ri.metric);keep=eig>ri.metric_info.cutoff;ideal=(u[:,keep]/eig[keep])@u[:,keep].conj().T
    delta=ri.factor@ri.factor.conj().T-ideal;d=densities()*.01;total=np.sum(d,axis=0)*w[:,None,None]
    error=(t.T@delta.conj()@t.conj()@total.ravel()).reshape(total.shape)
    assert np.all(abs(error).ravel()<=np.asarray(bound.metric_action_j_bounds)+2e-12)
    actual=bound.build(d,w);control=ri.build(d,w,make_k=False);np.testing.assert_array_equal(actual.J,control.J)
    if c.metric_action_j_bound:
        o.metric_action_j_error=c.metric_action_j_bound;o.metric_action_energy_error=c.metric_action_energy_bound
        assert manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']).receipt.metric_action_qualified
        o.metric_action_j_error=np.nextafter(c.metric_action_j_bound,0.)
        with pytest.raises(RuntimeError,match='metric action targets unmet'):manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])


def action_job(directory,method='rhf',kind='fixed'):
    from tests.test_manta_periodic_rank_admission import rank_job
    job,args,policy=rank_job(directory,method,kind);policy.metric_rank_perturbation=None
    policy.metric_action_j_error=policy.metric_action_energy_error=1e8
    fitting=args['options'].fitting
    fitting.coupling_multipole_error=0.;fitting.coupling_multipole_pair_tree=False
    fitting.coupling_multipole_compress=False;fitting.coupling_multipole_global_actions=False
    args['options'].fitting=fitting
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy,
        accuracy=None,coupling_multipole_error=0.,coupling_multipole_pair_tree=False,
        coupling_multipole_compress=False,coupling_multipole_global_actions=False)
    return job,args,policy


@pytest.mark.parametrize('method,kind',[('rhf','fixed'),('uhf','fixed'),('rhf','reference'),('rhf','domain'),('rhf','auxiliary')])
def test_metric_action_qualification_drives_bounded_search(tmp_path,method,kind):
    from tests.test_manta_periodic_rank_admission import rank_control
    job,args,policy=action_job(tmp_path,method,kind);result=vq.run_periodic_job(**job)
    full=result.integral_provenance['fitting']['full_coulomb_accuracy'];assert result.converged and full['metric_action']['qualified']
    key={'fixed':'metric_selection','reference':'metric_selection','domain':'domain_selection','auxiliary':'auxiliary_selection'}[kind]
    trials=full[key]['trials'];first=trials[0] if kind in ('fixed','reference') else trials[0]['bound']
    assert len(trials)>=2 and not first['metric_action']['qualified']
    control=vq.run_periodic_job(**rank_control(job,args,policy,result,tmp_path/'control',kind))
    assert control.converged and control.energy==result.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    np.testing.assert_array_equal(result.native.fock,control.native.fock)


@pytest.mark.parametrize('kind',['fixed','reference','domain','auxiliary'])
def test_unattainable_action_target_refuses_before_scf(tmp_path,kind):
    job,args,policy=action_job(tmp_path,kind=kind);policy.metric_action_j_error=policy.metric_action_energy_error=1e-100
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)
    assert vq.run_periodic_job(**dict(job,dry_run=True)) is None
    with pytest.raises(RuntimeError,match='metric action targets unmet'):vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


@pytest.mark.parametrize('j,e',[(-1.,1.),(float('nan'),1.),(1.,float('inf')),(0.,1.)])
def test_invalid_action_controls_before_work(tmp_path,j,e):
    from tests.test_manta_periodic_ri_fitting_accuracy import accuracy_job
    job,args,_,policy=accuracy_job(tmp_path);policy.metric_action_j_error=j;policy.metric_action_energy_error=e
    with pytest.raises(ValueError,match='metric action targets'):dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)
    args['options'].fitting_accuracy=policy
    with pytest.raises(ValueError,match='metric action targets'):manta.plan_periodic_ri_rhf_from_cutoffs(**args)


@pytest.mark.parametrize('rank',[0,2,6])
@pytest.mark.parametrize('component',[False,True])
def test_pair_tree_metric_action_bounds_preserve_physical_build(rank,component):
    from tests.test_manta_periodic_ri_fitting_bounds import case
    from tests.test_manta_periodic_ri_q import densities
    a,ri,ref,o,w=case(rank=rank,component=component)
    assert not ri.plan.explicit_coupling_action and ri.plan.coupling_norm_supported
    o.metric_action_j_error=o.metric_action_energy_error=1e30
    bound=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    assert bound.receipt.metric_action_qualified
    d=densities()*.01
    audited=bound.build(d,w);control=ri.build(d,w,make_k=False)
    np.testing.assert_array_equal(audited.J,control.J)
    assert audited.receipt.metric_roundoff_evaluated and not control.receipt.metric_roundoff_evaluated
    assert audited.receipt.metric_j_roundoff_evaluated and not control.receipt.metric_j_roundoff_evaluated
    assert audited.receipt.paired_roundoff_evaluated and not control.receipt.paired_roundoff_evaluated
    norm=bound.receipt.metric_action_adjoint_norm_bound
    assert norm==max(manta.PeriodicGDFTiles(**a).coupling_norm_bounds(a['budget']).metric_column_norm_bounds)
    from fractions import Fraction as Q
    assert Q(audited.receipt.metric_j_roundoff_bound)>=Q(norm)*Q(audited.receipt.metric_coefficient_roundoff_bound)
    assert audited.receipt.coupling_passes==control.receipt.coupling_passes
    assert audited.receipt.shell_calls==control.receipt.shell_calls
    assert audited.receipt.primitive_products==control.receipt.primitive_products
    assert audited.receipt.coulomb_energy==control.receipt.coulomb_energy
    if rank==0:
        assert audited.receipt.paired_j_roundoff_bound==audited.receipt.paired_energy_roundoff_bound==0.
        assert bound.receipt.metric_action_j_bound==bound.receipt.metric_action_energy_bound==0.
        assert audited.receipt.metric_j_roundoff_bound==audited.receipt.metric_coefficient_roundoff_bound==audited.receipt.metric_energy_roundoff_bound==0.
    else:
        o.metric_action_j_error=bound.receipt.metric_action_j_bound
        o.metric_action_energy_error=bound.receipt.metric_action_energy_bound
        assert manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']).receipt.metric_action_qualified
        o.metric_action_j_error=np.nextafter(o.metric_action_j_error,0.)
        with pytest.raises(RuntimeError,match='metric action targets unmet'):
            manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])


def test_explicit_nonzero_multipole_source_keeps_separate_metric_bound():
    from tests.test_manta_periodic_ri_fitting_bounds import case
    from tests.test_manta_periodic_ri_q import densities
    a,old,ref,o,w=case();a['options'].coupling_multipole_pair_tree=False
    src=manta.PeriodicGDFTiles(**a);ri=manta.PeriodicRIQBuilder(src,2,metric_options(old),a['budget'])
    assert ri.plan.explicit_coupling_action and src.receipt.coupling_error_bound>0.
    o.metric_action_j_error=o.metric_action_energy_error=1e20
    bound=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    assert bound.receipt.metric_action_qualified
    actual=bound.build(densities()*.01,w);control=ri.build(densities()*.01,w,make_k=False)
    assert actual.receipt.coupling_error_bound>0.
    np.testing.assert_array_equal(actual.J,control.J)


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_joint_metric_action_source_admits_all_caps(field,quantity):
    a,src,ri,ref,o,w=raw_case();plan=manta.plan_periodic_ri_fitting_bound(ri,ref,o,a['budget'])
    value=getattr(plan,quantity);setattr(a['budget'],field,value-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    setattr(a['budget'],field,value)
    assert manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']).receipt.metric_action_qualified



def decimal_support(values,limit,entries):
    """Independent primal allocation, not the implementation's dual formula."""
    with localcontext() as ctx:
        ctx.prec=100;D=Decimal.from_float;remaining=D(limit);value=Decimal(0)
        for v,h in sorted(zip(values,entries),reverse=True):
            take=min(remaining,D(h));value+=D(v)*take;remaining-=take
        return value


@pytest.mark.parametrize('limit,entries',[
    (1.,[.0001]+[1.]*16), (1.,[0.]+[1.]*16),
    (1.,[.25]*4+[0.]*13), (10.,[.01]*17),
    (np.nextafter(.3,0.),[.1,.2]+[0.]*15),
    (np.nextafter(.3,1.),[.1,.2]+[0.]*15),
    (1.,[0.]*17), (1e-100,[1e-102]+[1e-100]*16),
    (1e100,[1e96]+[1e100]*16),
])
def test_joint_family_matches_independent_decimal_primal(limit,entries):
    owner,f=action(np.diag([.4,2.7]).astype(complex),0.,.001)
    t=np.array([[1000.]+[1.]*16,[0.]*17],complex)
    cols=[owner.column(np.ascontiguousarray(t[:,j])) for j in range(17)]
    actual=owner.family(cols,limit,entries)
    with localcontext() as ctx:
        ctx.prec=100;D=Decimal.from_float
        support={name:decimal_support([getattr(c,name) for c in cols],limit,entries)
                 for name in ('factor_score','residual_norm','coupling_norm')}
        j=[min(D(c.factor_score)*support['factor_score']+
               D(owner.certificate.inverse_norm_bound)*D(c.residual_norm)*support['residual_norm'],
               D(owner.certificate.inverse_error_bound)*D(c.coupling_norm)*support['coupling_norm']) for c in cols]
        assert all(D(bound)>=exact for bound,exact in zip(actual.j_error_bounds,j))
        np.testing.assert_allclose(actual.j_error_bounds,[float(v) for v in j],rtol=3e-14,atol=0.)
        energy=decimal_support(actual.j_error_bounds,limit,entries)/2
        assert D(actual.energy_error_bound)>=energy
        assert actual.energy_error_bound==pytest.approx(float(energy),rel=3e-14,abs=0.)
        if entries[0]==.0001:
            old={name:min(D(limit)*max(D(getattr(c,name)) for c in cols),
                          sum(D(h)*D(getattr(c,name)) for c,h in zip(cols,entries))) for name in support}
            old_j=max(min(D(c.factor_score)*old['factor_score']+
                          D(owner.certificate.inverse_norm_bound)*D(c.residual_norm)*old['residual_norm'],
                          D(owner.certificate.inverse_error_bound)*D(c.coupling_norm)*old['coupling_norm']) for c in cols)
            assert D(actual.j_error_bound)*10<old_j
    # Check the resulting certificate against actual complex factor error,
    # at a feasible extremal density supported by the two largest columns.
    d=np.zeros(17,complex);d[0]=min(entries[0],limit)
    d[1]=min(entries[1],max(0.,limit-d[0]))/2
    observed_j,observed_e=decimal_error(np.diag([.4,2.7]).astype(complex),f,t,d,0.)
    assert all(x<=Decimal.from_float(y) for x,y in zip(observed_j,actual.j_error_bounds))
    assert observed_e<=Decimal.from_float(actual.energy_error_bound)


def test_joint_family_permutation_and_public_admission():
    owner,_=action(np.diag([.4,2.7]).astype(complex),0.,.001)
    values=[1000.,1.,1.,2.,3.,4.,1.];h=[.0001,1.,.3,.1,.01,0.,.2]
    cols=[owner.column(np.array([v,1j],complex)) for v in values]
    b=owner.family(cols,1.,h);order=[6,2,1,0,4,3,5]
    perm=owner.family([cols[i] for i in order],1.,[h[i] for i in order])
    np.testing.assert_allclose(perm.j_error_bounds,np.array(b.j_error_bounds)[order],rtol=2e-14,atol=0.)
    assert perm.energy_error_bound==pytest.approx(b.energy_error_bound,rel=2e-14)
    for name,field in [('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:
        budget=manta.ResourceBudget();p=manta.plan_hermitian_inverse_family(2,len(cols),budget)
        setattr(budget,name,getattr(p,field)-1)
        with pytest.raises((ValueError,RuntimeError),match='budget'):owner.family(cols,1.,h,budget)
        setattr(budget,name,getattr(p,field));assert owner.family(cols,1.,h,budget).qualified



@pytest.mark.parametrize('cut',[0.,1.,5.])
def test_norm_family_bounds_actual_complex_metric_action(cut):
    a=np.array([[.4,.13+.29j],[.13-.29j,2.7]],complex);owner,f=action(a,cut,.001)
    t=np.array([[.13+.3j,2.5,1-.1j],[1.1,-.3+.6j,.4]],complex)
    norms=[float(sum(abs(z.real)+abs(z.imag) for z in t[:,j])) for j in range(3)]
    result=owner.norm_family(norms,1.5,[.2,.4,.6]);d=np.array([.08+.1j,-.12+.2j,.2-.3j])
    actual_j,actual_e=decimal_error(a,f,t,d,cut)
    assert result.qualified
    assert all(x<=Decimal.from_float(y) for x,y in zip(actual_j,result.j_error_bounds))
    assert actual_e<=Decimal.from_float(result.energy_error_bound)
    if cut==5.:assert result.j_error_bound==result.energy_error_bound==0.
    for bad in ([-1.]*3,[float('nan')]*3):
        with pytest.raises(ValueError,match='nonnegative'):owner.norm_family(bad,1.)
    for name,field in [('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:
        budget=manta.ResourceBudget();p=manta.plan_hermitian_inverse_family(2,3,budget)
        setattr(budget,name,getattr(p,field)-1)
        with pytest.raises((ValueError,RuntimeError),match='budget'):owner.norm_family([float('nan')]*3,1.,[],budget)
        setattr(budget,name,getattr(p,field));assert owner.norm_family(norms,1.5,[.2,.4,.6],budget).qualified


def compressed_metric_job(directory,method='rhf',global_actions=False,pair_tree=False):
    from tests.test_manta_periodic_ri_fitting_accuracy import accuracy_job
    job,args,reference,policy=accuracy_job(directory,method)
    policy.metric_action_j_error=policy.metric_action_energy_error=1e30
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],accuracy=None,full_coulomb_accuracy=policy,
        coupling_multipole_error=1.,coupling_multipole_order=0,coupling_multipole_hierarchy=True,
        coupling_multipole_reuse=True,coupling_multipole_pair_tree=pair_tree,coupling_multipole_compress=True,
        coupling_multipole_global_actions=global_actions)
    return job,policy


@pytest.mark.parametrize('global_actions,pair_tree',[(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('method',['rhf','uhf','uks'])
def test_complete_compressed_metric_admission_and_fixed_source(tmp_path,method,global_actions,pair_tree):
    job,policy=compressed_metric_job(tmp_path,method,global_actions,pair_tree)
    result=vq.run_periodic_job(**job);assert result.converged
    bound=result.integral_provenance['fitting']['full_coulomb_accuracy']['metric_action']
    assert bound['qualified'] and bound['j_bound']<=policy.metric_action_j_error
    assert bound['energy_bound']<=policy.metric_action_energy_error and 'compressed-moment' in bound['coupling']
    receipt=result.native.receipt.fitted_fock
    assert receipt.coupling_multipole_compressed_nodes>0
    assert receipt.metric_roundoff_evaluated and receipt.metric_j_roundoff_evaluated and receipt.coupling_roundoff_evaluated
    assert receipt.forward_metric_roundoff_evaluated and receipt.paired_roundoff_evaluated
    assert receipt.weighted_input_roundoff_evaluated and receipt.coupling_forward_input_error_included and receipt.paired_input_roundoff_included
    assert_assembly(result)
    evidence=result.integral_provenance.get('fitted_fock',result.integral_provenance.get('coupling_approximation'))
    for name in ('metric_projection_roundoff_bound','metric_coefficient_roundoff_bound','metric_energy_roundoff_bound','metric_j_roundoff_bound','coupling_forward_roundoff_bound','coupling_adjoint_roundoff_bound','forward_metric_coefficient_roundoff_bound','forward_metric_energy_roundoff_bound','paired_j_roundoff_bound','paired_energy_roundoff_bound','weighted_input_roundoff_bound'):
        assert 0.<getattr(receipt,name)<1e-8
        assert evidence[name]==getattr(receipt,name)
    if global_actions and not pair_tree:assert receipt.coupling_multipole_global_pair_reuses>0
    policy.metric_action_j_error=policy.metric_action_energy_error=0.
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control'/'job',
        periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)))
    assert not control.native.receipt.assembly_roundoff.evaluated
    assert control.converged and result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','overlap','hcore'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    policy.metric_action_j_error=policy.metric_action_energy_error=1e-100
    with pytest.raises(RuntimeError,match='metric action targets unmet'):
        vq.run_periodic_job(**dict(job,output=tmp_path/'refused'/'job',periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)))
    assert not (tmp_path/'refused'/'job.qvf').exists()


@pytest.mark.parametrize('global_actions,pair_tree',[(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_compressed_metric_parent_admits_all_caps(global_actions,pair_tree,field,quantity):
    from tests.test_manta_periodic_ri_fitting_bounds import case
    a,old,ref,o,w=case();a['options'].coupling_multipole_pair_tree=pair_tree
    a['options'].coupling_multipole_hierarchy=True;a['options'].coupling_multipole_compress=True
    a['options'].coupling_multipole_global_actions=global_actions
    src=manta.PeriodicGDFTiles(**a);ri=manta.PeriodicRIQBuilder(src,2,metric_options(old),a['budget'])
    o.metric_action_j_error=o.metric_action_energy_error=1e30
    plan=manta.plan_periodic_ri_fitting_bound(ri,ref,o,a['budget']);value=getattr(plan,quantity)
    setattr(a['budget'],field,value-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    setattr(a['budget'],field,value)
    assert manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']).receipt.metric_action_qualified


def exact_paired_action(f,x):
    """Exact rational complex arithmetic, independent of interval replay."""
    from fractions import Fraction as Q
    z=lambda v:v if isinstance(v,tuple) else (Q(float(v.real)),Q(float(v.imag)))
    def add(a,b):return (a[0]+b[0],a[1]+b[1])
    def mul(a,b):return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
    def total(values):
        out=(Q(0),Q(0))
        for v in values:out=add(out,v)
        return out
    rho=[total(mul(z(f[i,j]),z(x[i])) for i in range(len(x))) for j in range(f.shape[1])]
    c=[total(mul(z(f[i,j].conjugate()),rho[j]) for j in range(f.shape[1])) for i in range(len(x))]
    energy=sum((re*re+im*im for re,im in rho),Q(0))/2
    return rho,c,energy


@pytest.mark.parametrize('rank',[0,1,3,8])
@pytest.mark.parametrize('kind',['complex','cancellation','subnormal'])
def test_executed_paired_metric_roundoff_against_exact_rationals(rank,kind):
    from fractions import Fraction as Q
    rng=np.random.default_rng(745);f=np.ascontiguousarray(rng.normal(size=(8,rank))+1j*rng.normal(size=(8,rank)))
    x=np.ascontiguousarray(rng.normal(size=8)+1j*rng.normal(size=8))
    if kind=='cancellation':
        f[1::2]=f[::2];x[1::2]=-x[::2];x[-1]+=2**-48
    elif kind=='subnormal':
        x*=np.finfo(float).tiny;f*=.25
    rho=f.T@x;c=f.conj()@rho;energy=float(.5*np.vdot(rho,rho).real)
    bound=manta.certify_paired_metric_roundoff(f,x,rho,c,energy)
    er,ec,ee=exact_paired_action(f,x);assert bound.qualified
    for observed,expected,bounds in ((rho,er,bound.projection_bounds),(c,ec,bound.coefficient_bounds)):
        for value,(re,im),enclosure in zip(observed,expected,bounds):
            error=abs(Q(float(value.real))-re)+abs(Q(float(value.imag))-im)
            assert error<=Q(enclosure)
    assert abs(Q(energy)-ee)<=Q(bound.energy_bound)
    assert bound.coefficient_bound==max(bound.coefficient_bounds,default=0.)
    assert bound.projection_bound==max(bound.projection_bounds,default=0.)
    if rank==0:assert bound.coefficient_bound==bound.energy_bound==0.
    # The observed numerical witness matters, not merely a model of flops.
    wrong=c.copy();wrong[0]+=.25+.5j
    changed=manta.certify_paired_metric_roundoff(f,x,rho,wrong,energy+.125)
    assert changed.coefficient_bound>=.74999999999 and changed.energy_bound>=.12499999999


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_executed_metric_audit_caps_precede_payload(field,quantity):
    f=np.eye(3,dtype=complex);x=np.ones(3,complex);b=manta.ResourceBudget()
    p=manta.plan_paired_metric_roundoff(3,3,b);value=getattr(p,quantity)
    setattr(b,field,value-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        manta.certify_paired_metric_roundoff(f*np.nan,x,x,x,1.5,b)
    setattr(b,field,value)
    assert manta.certify_paired_metric_roundoff(f,x,x,x,1.5,b).qualified


@pytest.mark.parametrize('bad',['shape','dtype','stride','nan','overflow'])
def test_executed_metric_audit_refuses_invalid_or_unbounded_inputs(bad):
    f=np.eye(3,dtype=complex);x=np.ones(3,complex);rho=x.copy();c=x.copy();energy=1.5
    if bad=='shape':rho=rho[:2]
    elif bad=='dtype':x=x.real.copy()
    elif bad=='stride':x=np.ones(6,complex)[::2]
    elif bad=='nan':c[0]=np.nan
    else:f[0,0]=1e308;x[0]=1e308
    with pytest.raises((ValueError,RuntimeError,OverflowError)):
        manta.certify_paired_metric_roundoff(f,x,rho,c,energy)


def test_retained_mode_does_not_claim_paired_arithmetic():
    from tests.test_manta_periodic_ri_q import densities
    a,src,ri,ref,o,w=raw_case()
    r=ri.build(densities()*.01,w,make_k=False,audit_metric_arithmetic=True)
    assert r.receipt.coulomb_contraction=='retained-mode'
    assert not r.receipt.metric_roundoff_evaluated and not r.receipt.metric_j_roundoff_evaluated


@pytest.mark.parametrize('spin_scale',[0.,.01,-.03])
def test_explicit_paired_metric_j_contribution_exact_complex_adjoint(spin_scale):
    from fractions import Fraction as Q
    from tests.test_manta_periodic_gdf_tiles import assemble
    from tests.test_manta_periodic_ri_q import densities
    a,src,old,ref,o,w=raw_case()
    opts=metric_options(old);opts.mode_block=1
    ri=manta.PeriodicRIQBuilder(src,2,opts,a['budget'])
    bound=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    result=bound.build(densities()*spin_scale,w);c=result.receipt
    assert c.coulomb_contraction=='paired-raw' and c.metric_j_roundoff_evaluated
    t=assemble(src).transpose(1,0,2,3).reshape(ri.plan.n_auxiliary,-1)
    # Exact rational witness on the stored explicit coupling, with signed
    # coefficient perturbations bounded by the actual metric certificate.
    # No floating-point adjoint is used as the reference for this contribution.
    epsilon=Q(c.metric_coefficient_roundoff_bound);norm=Q(bound.receipt.metric_action_adjoint_norm_bound)
    for column in t.T:
        re=sum((Q(float(z.real))*epsilon*(-1 if i%2 else 1) for i,z in enumerate(column)),Q(0))
        im=sum((Q(float(z.imag))*epsilon*(-1 if i%2 else 1) for i,z in enumerate(column)),Q(0))
        assert abs(re)+abs(im)<=Q(c.metric_j_roundoff_bound)
        assert sum((abs(Q(float(z.real)))+abs(Q(float(z.imag))) for z in column),Q(0))<=norm
    control=ri.build(densities()*spin_scale,w,make_k=False,audit_metric_arithmetic=True)
    assert control.receipt.metric_roundoff_evaluated and not control.receipt.metric_j_roundoff_evaluated
    np.testing.assert_array_equal(result.J,control.J)
    assert result.receipt.snapshot_identity==control.receipt.snapshot_identity
    assert result.receipt.shell_calls==control.receipt.shell_calls
    if spin_scale==0.:assert c.metric_j_roundoff_bound==0.


@pytest.mark.parametrize('rank',[0,1,3,8])
@pytest.mark.parametrize('kind',['complex','cancellation','subnormal','zero'])
def test_paired_action_input_errors_against_exact_rationals(rank,kind):
    from fractions import Fraction as Q
    rng=np.random.default_rng(123)
    f=np.ascontiguousarray(rng.normal(size=(8,rank))+1j*rng.normal(size=(8,rank)))
    x=np.ascontiguousarray(rng.normal(size=8)+1j*rng.normal(size=8))
    eta=np.full(8,2**-24)
    if kind=='cancellation':f[1::2]=f[::2];x[1::2]=-x[::2]
    elif kind=='subnormal':x*=np.finfo(float).tiny;f*=.25;eta[:]=np.nextafter(0.,1.)
    elif kind=='zero':eta[:]=0.
    rho=f.T@x;c=f.conj()@rho;energy=float(.5*np.vdot(rho,rho).real)
    b=manta.certify_paired_metric_action(f,x,rho,c,energy,eta)
    assert b.qualified and b.input_error_included
    truth=[(Q(float(z.real))+(-1 if i%2 else 1)*Q(float(e))/2,
            Q(float(z.imag))+Q(float(e))/4) for i,(z,e) in enumerate(zip(x,eta))]
    er,ec,ee=exact_paired_action(f,truth)
    for values,expected,bounds in ((rho,er,b.projection_bounds),(c,ec,b.coefficient_bounds)):
        for z,(re,im),bound in zip(values,expected,bounds):
            assert abs(Q(float(z.real))-re)+abs(Q(float(z.imag))-im)<=Q(bound)
    assert abs(Q(energy)-ee)<=Q(b.energy_bound)
    if rank==0:assert b.coefficient_bound==b.energy_bound==0.
    if kind=='zero':
        old=manta.certify_paired_metric_roundoff(f,x,rho,c,energy)
        assert not old.input_error_included
        assert b.coefficient_bounds==old.coefficient_bounds
        assert b.projection_bounds==old.projection_bounds and b.energy_bound==old.energy_bound
    wrong=c.copy();wrong[0]+=.25+.5j
    changed=manta.certify_paired_metric_action(f,x,rho,wrong,energy+.125,eta)
    assert changed.coefficient_bound>=.74999 and changed.energy_bound>=.12499


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_paired_action_caps_precede_input_payload(field,quantity):
    f=np.eye(3,dtype=complex);x=np.ones(3,complex);eta=np.full(3,.01);b=manta.ResourceBudget()
    p=manta.plan_paired_metric_action(3,3,b);value=getattr(p,quantity)
    setattr(b,field,value-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        manta.certify_paired_metric_action(f,x,x,x,1.5,eta*np.nan,b)
    setattr(b,field,value)
    assert manta.certify_paired_metric_action(f,x,x,x,1.5,eta,b).qualified


@pytest.mark.parametrize('bad',['shape','dtype','stride','negative','nan','infinity','overflow'])
def test_paired_action_refuses_invalid_input_errors(bad):
    f=np.eye(3,dtype=complex);x=np.ones(3,complex);eta=np.full(3,.01)
    if bad=='shape':eta=eta[:2]
    elif bad=='dtype':eta=eta.astype(np.float32)
    elif bad=='stride':eta=np.zeros(6)[::2]
    elif bad=='negative':eta[0]=-.01
    elif bad=='nan':eta[0]=np.nan
    elif bad=='infinity':eta[0]=np.inf
    else:eta[0]=1e308;x[0]=1e308
    with pytest.raises((ValueError,RuntimeError,OverflowError)):
        manta.certify_paired_metric_action(f,x,np.ones(3,complex),np.ones(3,complex),1.5,eta)


@pytest.mark.parametrize('spin_cancellation',[False,True])
@pytest.mark.parametrize('weights',[(.5,.5),(.27,.73),(0.,1.)])
@pytest.mark.parametrize('scale',[0.,.01,-.03])
def test_complete_explicit_paired_j_energy_against_exact_rationals(scale,weights,spin_cancellation):
    from fractions import Fraction as Q
    from tests.test_manta_periodic_ri_fitting_bounds import case
    from tests.test_manta_periodic_gdf_tiles import assemble
    from tests.test_manta_periodic_ri_q import densities
    a,old,ref,o,w=case(weights=weights)
    a['options'].coupling_multipole_error=0.;a['options'].coupling_multipole_pair_tree=False
    src=manta.PeriodicGDFTiles(**a);opts=metric_options(old);opts.mode_block=1
    ri=manta.PeriodicRIQBuilder(src,2,opts,a['budget'])
    o.metric_action_j_error=o.metric_action_energy_error=1e30
    owner=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    d=densities()*scale
    if spin_cancellation:
        d[1]=-d[0];d[1,0,0,0]+=abs(scale)*2**-40
    result=owner.build(d,w);receipt=result.receipt
    assert receipt.coulomb_contraction=='paired-raw' and receipt.paired_roundoff_evaluated
    t=assemble(src).transpose(1,0,2,3).reshape(ri.plan.n_auxiliary,-1)
    z=lambda v:(Q(float(v.real)),Q(float(v.imag)))
    def qdot(a,b):
        parts=[mul(x,y) for x,y in zip(a,b)]
        return (sum((v[0] for v in parts),Q(0)),sum((v[1] for v in parts),Q(0)))
    # Exact rational spin sum and k weighting, before all coupled actions.
    # Non-power-of-two stored weights and two independent spin densities exercise
    # the newly included input construction; no reference tolerance is added.
    assert receipt.weighted_input_roundoff_evaluated and receipt.coupling_forward_input_error_included
    assert receipt.paired_input_roundoff_included
    weighted=[]
    for k in range(len(w)):
        for i in range(d.shape[-1]):
            for j in range(d.shape[-1]):
                re=sum((z(d[spin,k,i,j])[0] for spin in range(2)),Q(0))*Q(float(w[k]))
                im=sum((z(d[spin,k,i,j])[1] for spin in range(2)),Q(0))*Q(float(w[k]))
                weighted.append((re,im))
                stored=(d[0,k,i,j]+d[1,k,i,j])*w[k]
                assert abs(z(stored)[0]-re)+abs(z(stored)[1]-im)<=Q(receipt.weighted_input_roundoff_bound)
    x=[qdot([conj(z(v)) for v in row],weighted) for row in t]
    _,coefficients,energy=exact_paired_action(ri.factor,x)
    exact_j=[qdot([z(v) for v in column],coefficients) for column in t.T]
    for value,(re,im) in zip(result.J.ravel(),exact_j):
        assert abs(Q(float(value.real))-re)+abs(Q(float(value.imag))-im)<=Q(receipt.paired_j_roundoff_bound)
    assert abs(Q(receipt.coulomb_energy)-energy)<=Q(receipt.paired_energy_roundoff_bound)
    assert receipt.paired_energy_roundoff_bound==receipt.forward_metric_energy_roundoff_bound
    control=ri.build(d,w,make_k=False,audit_metric_arithmetic=True)
    assert control.receipt.forward_metric_roundoff_evaluated and not control.receipt.paired_roundoff_evaluated
    np.testing.assert_array_equal(result.J,control.J)
    assert result.receipt.snapshot_identity==control.receipt.snapshot_identity
    assert not owner.receipt.roundoff_included


# Assembly tests use exact Fractions independently of the native interval graph.
def assembly_data(spins=2,kind='ordinary',scale=.3):
    rng=np.random.default_rng(713)
    def hermitian(shape):
        a=(rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(complex)
        return np.ascontiguousarray(a+a.swapaxes(-1,-2).conj())
    h,j=hermitian((2,2,2)),hermitian((2,2,2))
    d,k,xc,c=[hermitian((spins,2,2,2)) for _ in range(4)]
    if kind=='cancellation':
        h*=2.**48;j=-h+j
    elif kind=='subnormal':
        for a in (h,j,d,k,xc,c):a*=2.**-530
    w=np.array([.27,.73])
    # Exercise the actual reconstruction/projection sequence, and independent
    # numerical trace ordering. The bound compares witnesses, not a presumed
    # implementation order. A deliberate skew proves projection is enclosed.
    xc[0,0,0,1]+=3e-12j
    f=h+j-scale*k+xc+c;f=(f-h)+h;f=np.ascontiguousarray((f+f.swapaxes(-1,-2).conj())*.5)
    ej,ex,en=.137,-.43,.83
    e=float(np.einsum('k,skij,kji->',w,d,h).real+ej+ex+en
            -.5*scale*np.einsum('k,skij,skji->',w,d,k).real+.5*np.einsum('k,skij,skji->',w,d,c).real)
    return dict(hcore=h,density=d,weights=w,coulomb=j,exchange=k,xc=xc,correction=c,
                exchange_scale=scale,coulomb_energy=ej,xc_energy=ex,nuclear_energy=en,fock=f,energy=e)


def exact_assembly_errors(a,exact_correction=None):
    from fractions import Fraction as Q
    def z(x):return Q(float(x.real)),Q(float(x.imag))
    def plus(x,y):return x[0]+y[0],x[1]+y[1]
    def prod(x,y):return x[0]*y[0]-x[1]*y[1],x[0]*y[1]+x[1]*y[0]
    def scaled(x,y):return x[0]*y,x[1]*y
    def correction(q):return z(a['correction'][q]) if exact_correction is None else exact_correction[q]
    energy=sum(Q(a[key]) for key in ('coulomb_energy','xc_energy','nuclear_energy'))
    maximum=Q(0);scale=Q(a['exchange_scale'])
    for s,k,i,j in np.ndindex(a['density'].shape):
        q=s,k,i,j;t=s,k,j,i
        expected=plus(plus(z(a['hcore'][k,i,j]),z(a['coulomb'][k,i,j])),z(a['xc'][q]))
        expected=plus(plus(expected,scaled(z(a['exchange'][q]),-scale)),correction(q))
        value=z(a['fock'][q]);maximum=max(maximum,abs(value[0]-expected[0])+abs(value[1]-expected[1]))
        density=scaled(z(a['density'][q]),Q(a['weights'][k]))
        energy+=prod(density,z(a['hcore'][k,j,i]))[0]
        energy-=Q(1,2)*scale*prod(density,z(a['exchange'][t]))[0]
        energy+=Q(1,2)*prod(density,correction(t))[0]
    return maximum,abs(Q(a['energy'])-energy)


@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('kind',['ordinary','cancellation','subnormal'])
@pytest.mark.parametrize('scale',[-.3,0.,.25,1.])
def test_evaluated_fock_assembly_exact_rational_witnesses(spins,kind,scale):
    from fractions import Fraction as Q
    a=assembly_data(spins,kind,scale);r=manta.certify_periodic_assembly_roundoff(**a)
    f,e=exact_assembly_errors(a)
    assert r.evaluated and not r.paired_coulomb_included
    assert f<=Q(r.fock_bound) and e<=Q(r.energy_bound)
    assert r.combined_fock_bound==r.fock_bound and r.combined_energy_bound==r.energy_bound
    # Deliberately different public witnesses must not reuse a producer-only
    # rounding estimate that misses terminal reconstruction/projection.
    a['fock'][0,0,0,0]+=1.;a['energy']+=1.
    r=manta.certify_periodic_assembly_roundoff(**a);f,e=exact_assembly_errors(a)
    assert f<=Q(r.fock_bound) and e<=Q(r.energy_bound)
    assert r.fock_bound>=.9 and r.energy_bound>=.9


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_assembly_caps_precede_input_and_match_exact_boundary(field,quantity):
    a=assembly_data();b=manta.ResourceBudget();p=manta.plan_periodic_assembly_roundoff(2,2,2,b)
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        manta.certify_periodic_assembly_roundoff(**dict(a,hcore=np.full_like(a['hcore'],np.nan)),budget=b)
    setattr(b,field,getattr(p,quantity));assert manta.certify_periodic_assembly_roundoff(**a,budget=b).evaluated


@pytest.mark.parametrize('key',['hcore','density','weights','coulomb','exchange','xc','correction','fock','energy','exchange_scale'])
def test_assembly_nonfinite_components_refused(key):
    a=assembly_data()
    if isinstance(a[key],np.ndarray):a[key].flat[0]=np.nan
    else:a[key]=np.nan
    with pytest.raises(ValueError,match='finite'):manta.certify_periodic_assembly_roundoff(**a)


def assert_assembly(result):
    from fractions import Fraction as Q
    r=result.native.receipt.assembly_roundoff;p=result.native.receipt.fitted_fock
    assert r.evaluated and r.paired_coulomb_included
    assert 0.<r.fock_bound<1e-8 and 0.<r.energy_bound<1e-8
    assert Q(r.combined_fock_bound)>=Q(r.fock_bound)+Q(p.paired_j_roundoff_bound)
    assert Q(r.combined_energy_bound)>=Q(r.energy_bound)+Q(p.paired_energy_roundoff_bound)
    for name in ('evaluated','paired_coulomb_included','fock_bound','energy_bound','combined_fock_bound','combined_energy_bound'):
        assert result.integral_provenance['assembly_roundoff'][name]==getattr(r,name)
    assert not result.integral_provenance['assembly_roundoff']['source_construction_included']
    assert 'manta_periodic_assembly_roundoff' in result.integral_provenance['citation_numerics']
    def plus_up(a,b):return np.nextafter(a+b,np.inf) if b else a
    assert r.combined_fock_bound==plus_up(plus_up(plus_up(r.fock_bound,p.paired_j_roundoff_bound),r.direct_exchange_fock_bound),r.exchange_correction_fock_bound)
    assert r.combined_energy_bound==plus_up(plus_up(plus_up(r.energy_bound,p.paired_energy_roundoff_bound),r.direct_exchange_energy_bound),r.exchange_correction_energy_bound)
    native=result.native.receipt
    assert r.exchange_correction_roundoff_included==(native.exchange_q0=='BvK-ewald')
    if r.exchange_correction_roundoff_included:
        assert 0.<r.exchange_correction_fock_bound<1e-8 and 0.<r.exchange_correction_energy_bound<1e-8
        assert 'manta_periodic_correction_roundoff' in result.integral_provenance['citation_numerics']
    else:assert r.exchange_correction_fock_bound==r.exchange_correction_energy_bound==0.
    for name in ('exchange_correction_roundoff_included','exchange_correction_fock_bound','exchange_correction_energy_bound'):
        assert result.integral_provenance['assembly_roundoff'][name]==getattr(r,name)
    components=[x for x in (getattr(native,'exact_exchange',None),getattr(native,'exchange',None)) if x is not None]
    components += [c.exchange for c in getattr(native,'exchange_components',[]) if c.fitted_fock is None]
    assert r.direct_exchange_roundoff_included==bool(components)
    for component in components:
        assert component.roundoff_evaluated and component.roundoff_bound>0.
        assert all(x>0. for x in component.energy_roundoff_bounds)
    if components:
        assert 0.<r.direct_exchange_fock_bound<1e-8 and 0.<r.direct_exchange_energy_bound<1e-8
        assert 'manta_periodic_exchange_roundoff' in result.integral_provenance['citation_numerics']
    else:assert r.direct_exchange_fock_bound==r.direct_exchange_energy_bound==0.
    for name in ('direct_exchange_roundoff_included','direct_exchange_fock_bound','direct_exchange_energy_bound'):
        assert result.integral_provenance['assembly_roundoff'][name]==getattr(r,name)


@pytest.mark.parametrize('method',['rhf','uhf','rks','uks','pbe0','hse06'])
@pytest.mark.parametrize('capped',[False,True])
def test_assembly_complete_methods_and_stopped_states(tmp_path,method,capped):
    job,policy=compressed_metric_job(tmp_path,method,True,True)
    if capped:job['max_iter']=1
    result=vq.run_periodic_job(**job)
    assert result.converged!=capped
    assert_assembly(result)
    policy.metric_action_j_error=policy.metric_action_energy_error=0.
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control'/'job',
        periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)))
    assert not control.native.receipt.assembly_roundoff.evaluated
    assert result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))


@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('scale',[-.65,0.,.25,1.])
@pytest.mark.parametrize('fitted',[False,True])
def test_assembly_exchange_input_error_exact_rational_composition(spins,scale,fitted):
    from fractions import Fraction as Q
    a=assembly_data(spins,scale=scale);a["fitted_exchange"]=fitted
    exact=dict(a,exchange=a['exchange']*(1.+1e-8))
    delta=exact['exchange']-a['exchange']
    radii=np.nextafter(np.abs(delta.real)+np.abs(delta.imag),np.inf)
    result=manta.certify_periodic_assembly_roundoff(**a,exchange_errors=radii)
    zero=manta.certify_periodic_assembly_roundoff(**a,exchange_errors=np.zeros_like(radii))
    plain=manta.certify_periodic_assembly_roundoff(**a)
    kind="fitted_exchange" if fitted else "direct_exchange"
    assert not getattr(result,("direct_exchange" if fitted else "fitted_exchange")+"_roundoff_included")
    assert getattr(zero,kind+"_roundoff_included") and not getattr(plain,kind+"_roundoff_included")
    assert zero.combined_fock_bound==plain.combined_fock_bound
    assert zero.combined_energy_bound==plain.combined_energy_bound
    f,e=exact_assembly_errors(exact)
    assert getattr(result,kind+"_roundoff_included") and not result.paired_coulomb_included
    assert f<=Q(result.combined_fock_bound) and e<=Q(result.combined_energy_bound)
    if scale==0.:
        assert getattr(result,kind+"_fock_bound")==getattr(result,kind+"_energy_bound")==0.
    else:assert getattr(result,kind+"_fock_bound")>0. and getattr(result,kind+"_energy_bound")>0.
    for bad in (np.full_like(radii,-1.),np.full_like(radii,np.nan)):
        with pytest.raises(ValueError,match='nonnegative'):manta.certify_periodic_assembly_roundoff(**a,exchange_errors=bad)


@pytest.mark.parametrize('method',['rks','uks'])
@pytest.mark.parametrize('capped',[False,True])
def test_direct_exchange_signed_range_composition(tmp_path,method,capped):
    job,policy=compressed_metric_job(tmp_path,method,True,True)
    job.update(functional='cam-b3lyp',exchange=manta.ExchangeBuildOptions(algorithm='exact'),exchange_exxdiv='ewald')
    if capped:job['max_iter']=1
    result=vq.run_periodic_job(**job);assert result.converged!=capped;assert_assembly(result)
    components=result.native.receipt.exchange_components
    assert len(components)==2 and components[0].coefficient>0. and components[1].coefficient<0.
    assert all(c.exchange.roundoff_evaluated for c in components)
    policy.metric_action_j_error=policy.metric_action_energy_error=0.
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control'/'job',
        periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)))
    assert result.energy==control.energy and result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))


# Independently expand all four AO indices with exact binary-rational inputs.
# Non-Hermitian S detects any unjustified S^dagger substitution in the diagnostic.
def exact_correction(s,d,madelung,scale):
    from fractions import Fraction as Q
    def z(x):return Q(float(x.real)),Q(float(x.imag))
    def prod(x,y):return x[0]*y[0]-x[1]*y[1],x[0]*y[1]+x[1]*y[0]
    factor=-Q(madelung)*Q(scale);out={}
    for spin,k,i,j in np.ndindex(d.shape):
        real=imag=Q(0)
        for a in range(s.shape[-1]):
            for b in range(s.shape[-1]):
                x,y=prod(prod(z(s[k,i,a]),z(d[spin,k,a,b])),z(s[k,b,j]))
                real+=x;imag+=y
        out[spin,k,i,j]=factor*real,factor*imag
    return out


def correction_data(spins=2,kind='ordinary',scale=.3):
    a=assembly_data(spins)
    s=a['hcore'].copy();s[0,0,1]+=.37j
    d=a['density'].copy();madelung=-.237
    if kind=='cancellation':
        s[:]=np.array([[1.,1.],[1.,-1.]])
        d[:]=np.array([[2.**48,.1j],[-.1j,-2.**48]])
    elif kind=='subnormal':s*=2.**-530
    c=np.ascontiguousarray(-scale*madelung*((s@d)@s))
    return dict(overlap=s,density=d,madelung=madelung,exchange_scale=scale,correction=c)


@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('kind',['ordinary','cancellation','subnormal'])
@pytest.mark.parametrize('scale',[-.65,0.,.3,1.])
def test_correction_construction_exact_rational_witnesses(spins,kind,scale):
    from fractions import Fraction as Q
    a=correction_data(spins,kind,scale)
    exact=exact_correction(a['overlap'],a['density'],a['madelung'],scale)
    for perturb in (False,True):
        if perturb:a['correction'][0,0,0,1]+=1.+.5j
        bounds=manta.certify_periodic_correction_roundoff(**a)
        assert bounds.shape==a['density'].shape and np.all(np.isfinite(bounds))
        for q in np.ndindex(bounds.shape):
            actual=a['correction'][q];x,y=exact[q]
            assert abs(Q(float(actual.real))-x)+abs(Q(float(actual.imag))-y)<=Q(bounds[q])
        if perturb:assert bounds[0,0,0,1]>=1.
        elif scale==0.:assert np.all(bounds==0.)


@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('scale',[-.65,0.,.3,1.])
def test_correction_construction_composes_into_fock_and_energy(spins,scale):
    from fractions import Fraction as Q
    c=correction_data(spins,scale=scale);a=assembly_data(spins)
    # Deliberately retain independent final witnesses; the enclosure must not
    # depend on the diagnostic repeating the physical summation order.
    a['correction']=c['correction'];a['density']=c['density']
    radii=manta.certify_periodic_correction_roundoff(**c)
    result=manta.certify_periodic_assembly_roundoff(**a,correction_errors=radii)
    f,e=exact_assembly_errors(a,exact_correction(c['overlap'],c['density'],c['madelung'],scale))
    assert f<=Q(result.combined_fock_bound) and e<=Q(result.combined_energy_bound)
    assert result.exchange_correction_roundoff_included
    assert result.exchange_correction_fock_bound==np.max(radii)
    zero=manta.certify_periodic_assembly_roundoff(**a,correction_errors=np.zeros_like(radii))
    plain=manta.certify_periodic_assembly_roundoff(**a)
    assert zero.exchange_correction_roundoff_included and not plain.exchange_correction_roundoff_included
    assert zero.combined_fock_bound==plain.combined_fock_bound
    assert zero.combined_energy_bound==plain.combined_energy_bound
    for bad in (np.full_like(radii,-1.),np.full_like(radii,np.nan)):
        with pytest.raises(ValueError,match='nonnegative'):
            manta.certify_periodic_assembly_roundoff(**a,correction_errors=bad)


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_correction_caps_precede_payload_and_match_boundary(field,quantity):
    a=correction_data();b=manta.ResourceBudget();p=manta.plan_periodic_correction_roundoff(2,2,2,b)
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        manta.certify_periodic_correction_roundoff(**dict(a,overlap=np.full_like(a['overlap'],np.nan)),budget=b)
    setattr(b,field,getattr(p,quantity))
    assert np.all(np.isfinite(manta.certify_periodic_correction_roundoff(**a,budget=b)))


@pytest.mark.parametrize('key',['overlap','density','madelung','exchange_scale','correction'])
def test_correction_nonfinite_inputs_refused(key):
    a=correction_data()
    if isinstance(a[key],np.ndarray):a[key].flat[0]=np.nan
    else:a[key]=np.nan
    with pytest.raises(ValueError,match='finite'):manta.certify_periodic_correction_roundoff(**a)


@pytest.mark.parametrize('method',['rhf','uhf','pbe0'])
def test_correction_absent_for_strict_zero_mode(tmp_path,method):
    job,_=compressed_metric_job(tmp_path,method,True,True);job['exchange_exxdiv']='none'
    result=vq.run_periodic_job(**job)
    assert result.converged and result.native.receipt.exchange_q0=='strict-zero-mode'
    assert_assembly(result)


@pytest.mark.parametrize('key',['overlap','density','correction'])
@pytest.mark.parametrize('kind',['real','shape','strided'])
def test_correction_raw_array_contract(key,kind):
    a=correction_data();x=a[key]
    if kind=='real':a[key]=x.real.copy()
    elif kind=='shape':a[key]=x[...,:1].copy()
    else:a[key]=x[...,::-1]
    with pytest.raises(ValueError):manta.certify_periodic_correction_roundoff(**a)


def retained_metric_job(directory,method='rhf'):
    from tests.test_manta_periodic_ri_fitting_accuracy import accuracy_job
    job,_,_,policy=accuracy_job(directory,method)
    policy.metric_action_j_error=policy.metric_action_energy_error=1e30
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],accuracy=None,coupling_multipole_error=0.,coupling_screen_error=0.,coupling_multipole_pair_tree=False,coupling_multipole_global_actions=False,coupling_multipole_compress=False,full_coulomb_accuracy=policy)
    job['coulomb'].metric.mode_block=64
    return job,policy


def assert_retained_assembly(result):
    a=result.native.receipt.assembly_roundoff;r=result.native.receipt.fitted_fock
    assert a.evaluated and a.retained_coulomb_included and not a.paired_coulomb_included
    assert r.retained_j_roundoff_evaluated and not r.paired_roundoff_evaluated
    assert 0.<r.retained_j_roundoff_bound<1e-8 and 0.<r.retained_energy_roundoff_bound<1e-8
    def up(x,y):return np.nextafter(x+y,np.inf) if x and y else x+y
    assert a.combined_fock_bound==up(up(up(a.fock_bound,r.retained_j_roundoff_bound),a.direct_exchange_fock_bound),a.exchange_correction_fock_bound)
    assert a.combined_energy_bound==up(up(up(a.energy_bound,r.retained_energy_roundoff_bound),a.direct_exchange_energy_bound),a.exchange_correction_energy_bound)
    assert result.integral_provenance['assembly_roundoff']['retained_coulomb_included']
    assert 'manta_periodic_retained_roundoff' in result.integral_provenance['citation_numerics']


@pytest.mark.parametrize('method',['rhf','uhf','rks','uks','pbe0','hse06'])
@pytest.mark.parametrize('capped',[False,True])
def test_retained_complete_methods_and_stopped_states(tmp_path,method,capped):
    job,policy=retained_metric_job(tmp_path,method)
    if capped:job['max_iter']=1
    result=vq.run_periodic_job(**job);assert result.converged!=capped;assert_retained_assembly(result)
    policy.metric_action_j_error=policy.metric_action_energy_error=0.
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control'/'job',periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)))
    assert not control.native.receipt.assembly_roundoff.evaluated
    assert result.energy==control.energy and result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))


def fitted_exchange_job(directory,profile='rhf'):
    method=profile if profile in ('rhf','uhf') else profile.split('-')[-1]
    job,policy=retained_metric_job(directory,method)
    if method in ('rks','uks'):job['functional']=profile.rsplit('-',1)[0]
    job.update(exchange=manta.ExchangeBuildOptions(algorithm='ri'),exchange_exxdiv='ewald')
    return job,policy


def assert_fitted_exchange_assembly(result):
    a=result.native.receipt.assembly_roundoff;r=result.native.receipt.fitted_fock
    assert a.evaluated and a.retained_coulomb_included and not a.paired_coulomb_included
    assert a.fitted_exchange_roundoff_included and not a.direct_exchange_roundoff_included
    assert 0.<a.fitted_exchange_fock_bound<1e-8 and 0.<a.fitted_exchange_energy_bound<1e-8
    def up(x,y):return np.nextafter(x+y,np.inf) if x and y else x+y
    assert a.combined_fock_bound==up(up(up(a.fock_bound,r.retained_j_roundoff_bound),a.fitted_exchange_fock_bound),a.exchange_correction_fock_bound)
    assert a.combined_energy_bound==up(up(up(a.energy_bound,r.retained_energy_roundoff_bound),a.fitted_exchange_energy_bound),a.exchange_correction_energy_bound)
    parts=getattr(result.native.receipt,'exchange_components',[])
    if parts:
        assert all(c.fitted_fock.fitted_exchange_roundoff_evaluated for c in parts)
    else:assert r.fitted_exchange_roundoff_evaluated
    assert result.integral_provenance['assembly_roundoff']['fitted_exchange_roundoff_included']
    assert 'manta_periodic_fitted_exchange_roundoff' in result.integral_provenance['citation_numerics']


@pytest.mark.parametrize('profile',['rhf','uhf','pbe0-rks','pbe0-uks','hse06-rks','hse06-uks','cam-b3lyp-rks','cam-b3lyp-uks'])
@pytest.mark.parametrize('capped',[False,True])
def test_fitted_exchange_complete_methods_and_stopped_states(tmp_path,profile,capped):
    job,policy=fitted_exchange_job(tmp_path,profile)
    if capped:job['max_iter']=1
    result=vq.run_periodic_job(**job);assert result.converged!=capped;assert_fitted_exchange_assembly(result)
    if profile.startswith('cam'):
        parts=result.native.receipt.exchange_components
        assert len(parts)==2 and parts[0].coefficient>0. and parts[1].coefficient<0.
    policy.metric_action_j_error=policy.metric_action_energy_error=0.
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control'/'job',periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)))
    assert not control.native.receipt.assembly_roundoff.evaluated
    assert result.energy==control.energy and result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))



def decimal_source_action(a,b,t,d,cut):
    from tests.test_manta_hermitian_inverse import _decimal_spectral_function
    with localcontext() as ctx:
        ctx.prec=180
        ca,cb=(_decimal_spectral_function(x,cut) for x in (a,b))
        aa,bb,rr,ii=[y-x for x,y in zip(ca,cb)]
        matrix=[[(aa,Decimal(0)),(rr,ii)],[(rr,-ii),(bb,Decimal(0))]]
        kernel=[]
        for i in range(t.shape[1]):
            row=[]
            for j in range(t.shape[1]):
                v=[dot(matrix[k],[dc(x) for x in t[:,j]]) for k in range(2)]
                row.append(dot([conj(dc(x)) for x in t[:,i]],v))
            kernel.append(row)
        j=[dot(row,[dc(x) for x in d]) for row in kernel]
        e=mul((Decimal('.5'),Decimal(0)),dot([conj(dc(x)) for x in d],j))
        return [(x*x+y*y).sqrt() for x,y in j],(e[0]*e[0]+e[1]*e[1]).sqrt()


@pytest.mark.parametrize('cut',[0.,1.,5.])
@pytest.mark.parametrize('component',[False,True])
def test_source_action_decimal_projected_coulomb_and_energy(cut,component):
    a=np.array([[.4,.13+.29j],[.13-.29j,2.7]],complex)
    owner,f=action(a,cut);response=owner.certificate.bound_perturbation(.01)
    t=np.array([[.13+.3j,2.5,1-.1j],[1.1,-.3+.6j,.4]],complex)
    norms=(1.01*np.sum(abs(t),axis=0)).tolist();h=[.2,.4,.6] if component else []
    bound=response.norm_family(norms,1.5,h);assert bound.qualified
    b=a+np.array([[.001,.002-.003j],[.002+.003j,-.002]])
    d=np.array([.08+.1j,-.12+.2j,.2-.3j]);j,e=decimal_source_action(a,b,t,d,cut)
    assert all(x<=Decimal.from_float(y) for x,y in zip(j,bound.j_error_bounds))
    assert e<=Decimal.from_float(bound.energy_error_bound)
    zero=response.norm_family(norms,1.5,[0.]*3)
    assert zero.j_error_bound==zero.energy_error_bound==0.
    doubled=response.norm_family(norms,3.,[2*x for x in h])
    assert doubled.j_error_bound==pytest.approx(2*bound.j_error_bound,rel=2e-14)
    assert doubled.energy_error_bound==pytest.approx(4*bound.energy_error_bound,rel=2e-14)
    if cut==5.:assert bound.j_error_bound==bound.energy_error_bound==0.


def test_source_action_norm_family_strict_admission_and_invalid_inputs():
    owner,_=action(np.diag([.2,2.]));c=owner.certificate.bound_perturbation(.001)
    for name,field in [('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:
        b=manta.ResourceBudget();p=manta.plan_hermitian_inverse_family(2,2,b);setattr(b,name,getattr(p,field)-1)
        with pytest.raises((ValueError,RuntimeError),match='budget'):c.norm_family([np.nan]*2,1.,[],b)
        setattr(b,name,getattr(p,field));assert c.norm_family([1.,2.],1.,[],b).qualified
    for norms,limit,entries in [([-1.,2.],1.,[]),([1.,np.inf],1.,[]),([1.,2.],0.,[]),([1.,2.],1.,[1.]),([1.,2.],1.,[-1.,1.])]:
        with pytest.raises(ValueError):c.norm_family(norms,limit,entries)
    unavailable=owner.certificate.bound_perturbation(owner.certificate.spectrum.rank_stability_radius)
    assert not unavailable.norm_family([1.,2.],1.).qualified


def source_action_case(rank=6,compressed=False):
    from tests.test_manta_periodic_ri_fitting_bounds import case
    a,old,ref,o,w=case(rank=rank,component=True)
    if not compressed:
        a['options'].coupling_multipole_error=0.;a['options'].coupling_multipole_pair_tree=False
    a['options'].metric_assembly_rank=True
    src=manta.PeriodicGDFTiles(**a);ri=manta.PeriodicRIQBuilder(src,2,metric_options(old),a['budget'])
    o.metric_action_j_error=o.metric_action_energy_error=1e30
    return a,src,ri,ref,o,w


@pytest.mark.parametrize('rank',[0,2,6])
@pytest.mark.parametrize('compressed',[False,True])
def test_source_action_periodic_components_and_strict_targets(rank,compressed):
    from tests.test_manta_periodic_ri_q import densities
    a,src,ri,ref,o,w=source_action_case(rank,compressed)
    base=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']);o.metric_action_include_assembly=True
    bound=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']);c=bound.receipt
    assert c.metric_action_qualified and c.metric_action_assembly_requested and c.metric_action_assembly_qualified
    assert c.metric_action_assembly_identity and c.certificate_identity==base.receipt.certificate_identity
    assert c.metric_action_stored_j_bound==base.receipt.metric_action_j_bound
    assert c.metric_action_stored_energy_bound==base.receipt.metric_action_energy_bound
    expect=np.asarray(base.metric_action_j_bounds)+bound.metric_action_assembly_j_bounds
    assert np.all(np.asarray(bound.metric_action_j_bounds)>=expect)
    assert c.metric_action_energy_bound>=c.metric_action_stored_energy_bound+c.metric_action_assembly_energy_bound
    d=densities()*.01
    np.testing.assert_array_equal(bound.build(d,w).J,base.build(d,w).J)
    if rank:
        assert c.metric_action_assembly_j_bound>0 and c.metric_action_assembly_energy_bound>0
        o.metric_action_j_error=c.metric_action_j_bound;o.metric_action_energy_error=c.metric_action_energy_bound
        assert manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']).receipt.metric_action_qualified
        o.metric_action_j_error=np.nextafter(c.metric_action_j_bound,0.)
        with pytest.raises(RuntimeError,match='metric action targets unmet'):manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    else:assert c.metric_action_assembly_j_bound==c.metric_action_assembly_energy_bound==0


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_source_action_periodic_caps(field,quantity):
    a,src,ri,ref,o,w=source_action_case();o.metric_action_include_assembly=True
    p=manta.plan_periodic_ri_fitting_bound(ri,ref,o,a['budget']);setattr(a['budget'],field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    setattr(a['budget'],field,getattr(p,quantity));assert manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']).receipt.metric_action_assembly_qualified


def test_source_action_missing_proof_and_targets_refuse(tmp_path):
    a,src,ri,ref,o,w=raw_case();o.metric_action_include_assembly=True
    with pytest.raises(ValueError,match='metric_assembly_rank'):manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    o.metric_action_j_error=o.metric_action_energy_error=0.
    with pytest.raises(ValueError,match='positive metric action targets'):manta.plan_periodic_ri_fitting_bound(ri,ref,o,a['budget'])
    job,policy=retained_metric_job(tmp_path);policy.metric_action_include_assembly=True
    with pytest.raises(ValueError,match='metric_assembly_rank'):dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)


def source_action_job(directory,method='rhf',compressed=None):
    if compressed is None:job,policy=retained_metric_job(directory,method)
    else:job,policy=compressed_metric_job(directory,method,*compressed)
    policy.metric_action_include_assembly=True
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],metric_assembly_rank=True,full_coulomb_accuracy=policy)
    return job,policy
