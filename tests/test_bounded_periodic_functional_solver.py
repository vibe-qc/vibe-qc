"""Nonquadratic same-density energies for bounded periodic RKS/UKS numerics."""
import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_periodic_rhf_solver import _case, _options, _orthogonalizers


def case(opened=True, k=3, n=3, alpha=2, beta=1):
    c = _case(k, n)
    c.update(opened=opened, alpha=alpha if opened else 1, beta=beta if opened else 1,
             quadratic=.004, cubic=.001)
    if opened:
        c['operator'] = np.ascontiguousarray([c['operator'], .7*c['operator'][:, ::-1, ::-1]])
    return c


def plan(c, o=None, rank=None):
    k, n, _ = c['s'].shape
    return core._plan_bounded_periodic_functional_model(k,n,c['alpha'],c['beta'],c['opened'],
        n if rank is None else rank,_options() if o is None else o)


def caps(c, o=None, **changes):
    p = plan(c,o)
    out = core._BoundedPeriodicRHFCaps()
    out.maximum_owned_numerical_bytes = p.peak_owned_numerical_bytes
    out.maximum_total_numerical_bytes = p.total_numerical_bytes
    out.maximum_work_units = p.maximum_work_units
    out.maximum_fock_calls = 128
    for key, value in changes.items(): setattr(out,key,value)
    return out


def run(c, o=None, limits=None, **kwargs):
    o = _options() if o is None else o
    return core._solve_bounded_periodic_functional_model(c['s'],c['h'],c['operator'],
        c['alpha'],c['beta'],c['opened'],c['quadratic'],c['cubic'],c['nuclear'],
        o,caps(c,o) if limits is None else limits,**kwargs)


def functional(c,d):
    rho = np.sum(d*c['operator'].swapaxes(-1,-2)).real/len(c['s'])
    energy = c['quadratic']*rho**2/2+c['cubic']*rho**3/3
    potential = (c['quadratic']*rho+c['cubic']*rho**2)*c['operator']
    return energy,potential


def energy(c,d):
    return c['nuclear']+np.sum(d*c['h'].swapaxes(-1,-2)).real/len(c['s'])+functional(c,d)[0]


def occupy(c,f,o):
    x = _orthogonalizers(c,o)
    values,vectors = np.linalg.eigh(x.conj().swapaxes(-1,-2)@f@x)
    coefficients=x@vectors
    if c['opened']:
        d=np.array([coefficients[s,:,:,:count]@coefficients[s,:,:,:count].conj().swapaxes(-1,-2)
                    for s,count in enumerate((c['alpha'],c['beta']))])
    else:
        occupied=coefficients[:,:,:c['alpha']]
        d=2*occupied@occupied.conj().swapaxes(-1,-2)
    return d,values,coefficients


def initial(c,o):
    return occupy(c,np.stack([c['h']]*2) if c['opened'] else c['h'],o)[0]


@pytest.mark.parametrize('opened,k,alpha,beta', [(False,1,1,1),(False,3,1,1),
    (True,1,2,1),(True,3,2,1),(True,3,1,0),(True,2,0,1),(True,8,1,2)])
def test_nonquadratic_fixed_point_and_variational_energy(opened,k,alpha,beta):
    c=case(opened,k,alpha=alpha,beta=beta);o=_options()
    expected=initial(c,o)
    for _ in range(150):
        updated,_,_=occupy(c,c['h']+functional(c,expected)[1],o)
        if np.max(abs(updated-expected))<1e-13:
            expected=updated;break
        expected=updated
    else: pytest.fail('Independent nonlinear fixed point did not converge')
    result=run(c,o)
    assert result.converged
    d,f=result.density_copy(),result.fock_copy()
    np.testing.assert_allclose(d,expected,atol=4e-9,rtol=0)
    np.testing.assert_allclose(f,c['h']+functional(c,d)[1],atol=3e-12,rtol=0)
    assert result.final_snapshot.energy_per_cell == pytest.approx(energy(c,d),abs=4e-12)
    wrong=c['nuclear']+.5*np.sum(d*(c['h']+f).swapaxes(-1,-2)).real/k
    rho=np.sum(d*c['operator'].swapaxes(-1,-2)).real/k
    assert wrong-energy(c,d) == pytest.approx(c['cubic']*rho**3/6,abs=4e-12)
    assert abs(wrong-energy(c,d))>1e-7
    rng=np.random.default_rng(33)
    direction=rng.normal(size=d.shape)+1j*rng.normal(size=d.shape)
    direction=(direction+direction.conj().swapaxes(-1,-2))/2
    step=1e-5
    slope=(energy(c,d+step*direction)-energy(c,d-step*direction))/(2*step)
    assert slope == pytest.approx(np.sum(direction*f.swapaxes(-1,-2)).real/k,abs=2e-8)


@pytest.mark.parametrize('opened',[False,True])
@pytest.mark.parametrize('iterations',[1,2,3,4])
def test_capped_snapshots_and_observer_use_same_density_energy(opened,iterations):
    c=case(opened);o=_options(maximum_iterations=iterations)
    expected=initial(c,o);energies=[]
    for _ in range(iterations):
        energies.append(energy(c,expected))
        previous=expected
        expected,_,_=occupy(c,c['h']+functional(c,expected)[1],o)
    rows=[]
    result=run(c,o,callback=lambda row: rows.append(row))
    np.testing.assert_allclose(result.density_copy(),previous,atol=4e-12,rtol=0)
    np.testing.assert_allclose([r.energy_per_cell for r in rows],energies,atol=4e-12,rtol=0)
    assert result.final_snapshot.energy_per_cell == pytest.approx(energies[-1],abs=4e-12)
    assert len(rows)==iterations


@pytest.mark.parametrize('opened',[False,True])
@pytest.mark.parametrize('call',[1,2])
def test_missing_energy_never_reuses_previous_epoch(opened,call):
    c=case(opened);rows=[]
    with pytest.raises(ValueError,match='finite interaction energy'):
        run(c,callback=lambda row: rows.append(row),omit_energy_on_call=call)
    assert len(rows)==call-1


@pytest.mark.parametrize('opened',[False,True])
def test_missing_potential_and_nonfinite_energy_refuse(opened):
    c=case(opened)
    with pytest.raises((ValueError,RuntimeError,OverflowError),match='finite'):
        run(c,omit_potential=True)
    c['cubic']=np.nan
    with pytest.raises(ValueError,match='finite interaction energy'):run(c)


@pytest.mark.parametrize('opened',[False,True])
@pytest.mark.parametrize('field',['maximum_owned_numerical_bytes','maximum_total_numerical_bytes','maximum_work_units'])
def test_exact_resources_and_one_below_initial_admission(opened,field):
    c=case(opened);o=_options(maximum_iterations=1)
    exact=caps(c,o)
    assert run(c,o,limits=exact).final_snapshot.evaluated_iteration==1
    setattr(exact,field,getattr(exact,field)-1)
    with pytest.raises((ValueError,RuntimeError),match='cap|budget|reservation'):
        run(c,o,limits=exact,omit_energy_on_call=1)


@pytest.mark.parametrize('opened',[False,True])
def test_cancellation_retains_evaluated_nonlinear_state(opened):
    c=case(opened)
    result=run(c,callback=lambda row: row.evaluated_iteration<2)
    assert result.final_snapshot.status.name=='CANCELLED'
    assert result.final_snapshot.evaluated_iteration==2 and not result.converged
    assert result.final_snapshot.energy_per_cell == pytest.approx(energy(c,result.density_copy()),abs=4e-12)


@pytest.mark.parametrize('opened',[False,True])
def test_quadratic_limit_matches_existing_hf_solver(opened):
    if opened:
        from tests.test_bounded_periodic_uhf_solver import case as hf_case, run as hf_run
        reference=hf_case(interaction=.004)
    else:
        from tests.test_bounded_periodic_rhf_solver import _run as hf_run
        reference=_case(interaction=.004)
    c=case(opened);c['cubic']=0.
    functional_result=run(c)
    hf_result=hf_run(reference)
    assert functional_result.converged and hf_result.converged
    np.testing.assert_allclose(functional_result.density_copy(),hf_result.density_copy(),atol=3e-12,rtol=0)
    np.testing.assert_allclose(functional_result.fock_copy(),hf_result.fock_copy(),atol=3e-12,rtol=0)
    assert functional_result.final_snapshot.energy_per_cell == pytest.approx(hf_result.final_snapshot.energy_per_cell,abs=4e-12)


def test_restricted_limit_and_spin_interchange():
    restricted=case(False)
    unrestricted=case(True,alpha=1,beta=1)
    unrestricted['operator']=np.ascontiguousarray([restricted['operator']]*2)
    a,b=run(restricted),run(unrestricted)
    np.testing.assert_allclose(a.density_copy(),b.density_copy().sum(axis=0),atol=4e-10,rtol=0)
    assert a.final_snapshot.energy_per_cell == pytest.approx(b.final_snapshot.energy_per_cell,abs=4e-12)
    c=case(True);original=run(c)
    c['alpha'],c['beta']=c['beta'],c['alpha']
    c['operator']=c['operator'][::-1].copy()
    swapped=run(c)
    np.testing.assert_allclose(swapped.density_copy(),original.density_copy()[::-1],atol=4e-10,rtol=0)
    assert swapped.final_snapshot.energy_per_cell == pytest.approx(original.final_snapshot.energy_per_cell,abs=4e-12)


@pytest.mark.parametrize('opened',[False,True])
def test_gauge_transport_and_truncated_metric(opened):
    c=case(opened,k=2,n=4);c['s'][:,-1,:]=0;c['s'][:,:,-1]=0;c['s'][:,-1,-1]=1e-14
    original=run(c)
    assert original.converged and original.memory.retained_rank==3
    assert original.final_snapshot.energy_per_cell == pytest.approx(energy(c,original.density_copy()),abs=4e-12)
    phases=np.exp(1j*np.array([[.1,.7,-.3,.2],[.8,-.1,.2,.5]]))
    action=phases[:,:,None]*phases[:,None,:].conj()
    for name in ('s','h','operator'):c[name]=np.ascontiguousarray(c[name]*action)
    moved=run(c)
    np.testing.assert_allclose(moved.density_copy(),original.density_copy()*action,atol=4e-10,rtol=0)
    assert moved.final_snapshot.energy_per_cell == pytest.approx(original.final_snapshot.energy_per_cell,abs=4e-12)


@pytest.mark.parametrize('opened,alpha,beta', [
    (False,0,0),(False,3,3),(False,2**63+1,2**63+1),
    (True,0,0),(True,3,1),(True,1,3),(True,2**63+1,1),(True,1,2**63+1),
])
def test_population_refusal_precedes_restricted_doubling(opened,alpha,beta):
    c=case(opened=opened)
    limits=caps(c)
    c.update(alpha=alpha,beta=beta)
    with pytest.raises(ValueError,match='occupied and virtual'):
        plan(c)
    with pytest.raises(ValueError,match='occupied and virtual'):
        run(c,limits=limits)
