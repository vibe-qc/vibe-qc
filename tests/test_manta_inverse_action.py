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


def test_nonexplicit_coupling_refuses_only_requested_action_policy():
    from tests.test_manta_periodic_ri_fitting_bounds import case
    a,ri,ref,o,w=case();assert manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    o.metric_action_j_error=o.metric_action_energy_error=1.
    with pytest.raises(ValueError,match='explicit tile'):manta.plan_periodic_ri_fitting_bound(ri,ref,o,a['budget'])


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


def compressed_metric_job(directory,method='rhf',global_actions=False):
    from tests.test_manta_periodic_ri_fitting_accuracy import accuracy_job
    job,args,reference,policy=accuracy_job(directory,method)
    policy.metric_action_j_error=policy.metric_action_energy_error=1e30
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],accuracy=None,full_coulomb_accuracy=policy,
        coupling_multipole_error=1.,coupling_multipole_order=0,coupling_multipole_hierarchy=True,
        coupling_multipole_reuse=True,coupling_multipole_pair_tree=False,coupling_multipole_compress=True,
        coupling_multipole_global_actions=global_actions)
    return job,policy


@pytest.mark.parametrize('global_actions',[False,True])
@pytest.mark.parametrize('method',['rhf','uhf','uks'])
def test_complete_compressed_metric_admission_and_fixed_source(tmp_path,method,global_actions):
    job,policy=compressed_metric_job(tmp_path,method,global_actions)
    result=vq.run_periodic_job(**job);assert result.converged
    bound=result.integral_provenance['fitting']['full_coulomb_accuracy']['metric_action']
    assert bound['qualified'] and bound['j_bound']<=policy.metric_action_j_error
    assert bound['energy_bound']<=policy.metric_action_energy_error and 'compressed-moment' in bound['coupling']
    receipt=result.native.receipt.fitted_fock
    assert receipt.coupling_multipole_compressed_nodes>0
    if global_actions:assert receipt.coupling_multipole_global_pair_reuses>0
    policy.metric_action_j_error=policy.metric_action_energy_error=0.
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control'/'job',
        periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)))
    assert control.converged and result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','overlap','hcore'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    policy.metric_action_j_error=policy.metric_action_energy_error=1e-100
    with pytest.raises(RuntimeError,match='metric action targets unmet'):
        vq.run_periodic_job(**dict(job,output=tmp_path/'refused'/'job',periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)))
    assert not (tmp_path/'refused'/'job.qvf').exists()


@pytest.mark.parametrize('global_actions',[False,True])
@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_compressed_metric_parent_admits_all_caps(global_actions,field,quantity):
    from tests.test_manta_periodic_ri_fitting_bounds import case
    a,old,ref,o,w=case();a['options'].coupling_multipole_pair_tree=False
    a['options'].coupling_multipole_hierarchy=True;a['options'].coupling_multipole_compress=True
    a['options'].coupling_multipole_global_actions=global_actions
    src=manta.PeriodicGDFTiles(**a);ri=manta.PeriodicRIQBuilder(src,2,metric_options(old),a['budget'])
    o.metric_action_j_error=o.metric_action_energy_error=1e30
    plan=manta.plan_periodic_ri_fitting_bound(ri,ref,o,a['budget']);value=getattr(plan,quantity)
    setattr(a['budget'],field,value-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    setattr(a['budget'],field,value)
    assert manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']).receipt.metric_action_qualified
