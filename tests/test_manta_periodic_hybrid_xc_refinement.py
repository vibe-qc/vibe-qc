"""Explicit hybrid XC components, independent grid defects and physical Fock deltas."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_xc_refinement import case
from tests.test_manta_periodic_semilocal_xc import reference
from tests.test_manta_periodic_range_separated_ks import TERMS, EMPTY_RANGE, MIXED_RANGE

CAPS = [('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
        ('primitive_quartets', 'primitive_points'), ('scalar_terms', 'scalar_terms'),
        ('kernel_terms', 'kernel_terms')]


def plan(args):
    a = dict(args); a.pop('densities'); a.pop('weights')
    a.update(images=len(a['images']), momenta=len(a['momenta']))
    return manta.plan_periodic_hybrid_xc_refinement(**a)


def independent(args, c):
    rows = []
    for options in args['grid_options']:
        grid = manta.build_periodic_quadrature(args['system'], options, args['budget'])
        rows.append(reference(dict(c, points=grid.points, quadrature_weights=grid.weights), args['functional']))
    return rows


@pytest.mark.parametrize('name,spins,empty', [
    ('pbe0',1,False),('pbe0',2,False),('b3lyp',2,False),
    ('hse06',1,False),('hse06',2,False),('cam-b3lyp',1,False),('cam-b3lyp',2,False),
    ('lc-wpbe',1,False),('wb97x',2,False),(MIXED_RANGE,2,False),(EMPTY_RANGE,2,True)])
def test_independent_hybrid_components_and_every_spin_k_defect(name,spins,empty):
    args,c = case(name,spins)
    if empty: args['densities'][1] = 0.
    expected = independent(args,c); r = manta.assess_periodic_hybrid_xc_refinement(**args)
    assert r.plan.semilocal_component_only and plan(args).semilocal_component_only
    assert 'semilocal-component-only' in r.comparison and 'not-an-error-bound' in r.comparison
    assert not hasattr(r,'converged') and not hasattr(r,'accuracy_bound')
    target = expected[args['reference_index']]
    np.testing.assert_allclose(r.reference_potential,target[1],atol=3e-11,rtol=0)
    for step,(e,v,pop) in zip(r.steps,expected):
        assert step.xc.semilocal_component_only
        assert step.xc.energy == pytest.approx(e,abs=3e-11)
        assert step.energy_difference == pytest.approx(e-target[0],abs=3e-11)
        delta = v-target[1]
        np.testing.assert_allclose(np.array(step.potential_max_abs).reshape(spins,2),np.max(abs(delta),axis=(-1,-2)),atol=3e-11,rtol=0)
        np.testing.assert_allclose(np.array(step.potential_frobenius).reshape(spins,2),np.linalg.norm(delta,axis=(-1,-2)),atol=3e-11,rtol=0)
        np.testing.assert_allclose(step.population_differences,pop-target[2],atol=3e-12,rtol=0)
        if name in TERMS:
            assert step.xc.required_exact_exchange_fraction == 0.
            terms = step.xc.required_exchange_components
            assert [a[0] for a in terms] == [a[0] for a in TERMS[name]]
            np.testing.assert_allclose([a[1:] for a in terms],[a[1:] for a in TERMS[name]],atol=1e-15,rtol=0)
        else:
            assert step.xc.required_exact_exchange_fraction == pytest.approx(.25 if name=='pbe0' else .2)
            assert not step.xc.required_exchange_components
    assert r.steps[2].energy_difference == 0 and max(r.steps[2].potential_max_abs) == 0
    assert r.shell_calls == sum(s.xc.shell_calls for s in r.steps)
    assert r.primitive_points == sum(s.xc.primitive_points for s in r.steps)
    r.reference_potential[:] = np.nan
    assert np.isfinite(r.reference_potential).all()
    saved = r.steps
    saved[0].xc.required_exchange_components.clear()
    saved[0].options.radial_points = 1
    assert r.steps[0].options.radial_points == 12
    with pytest.raises(ValueError): manta.assess_periodic_xc_refinement(**args)


@pytest.mark.parametrize('family,species,spins', [
    ('global_hybrid','H2',1),('global_hybrid','HeH',2),
    ('range_separated','H2',1),('range_separated','HeH',2),('range_separated','H',2)])
def test_semilocal_defect_is_full_fock_defect_at_identical_density(family,species,spins):
    if family == 'global_hybrid':
        from tests.test_manta_periodic_global_hybrid_domains import arguments, explicit
    else:
        from tests.test_manta_periodic_range_separated_domains import arguments, explicit
    args,c,k,old = arguments(species,spins,grid=True)
    args['options'].ks.scf.maximum_iterations = 1
    explicit(args,c,old)
    grid1 = args['grid_options']; grid2 = manta.PeriodicQuadratureOptions()
    grid2.radial_points=16; grid2.lebedev_order=29; grid2.partition_radius_bohr=12.; grid2.stratmann=True
    run = getattr(manta,f'run_periodic_{family}_ks_with_quadrature')
    a = run(**args); b = run(**dict(args,grid_options=grid2))
    # Source digests preserve the native reciprocal lattice's exact bytes;
    # the independent NumPy inverse in the oracle differs at roundoff level.
    k = np.array([c['system'].reciprocal_lattice()@c['mesh'].fractional(c['mesh'].address(i)) for i in range(len(k))])
    # HCORE gives the same first evaluated density, independently of XC grid.
    np.testing.assert_array_equal(a.density,b.density)
    controls = dict(basis=c['basis'],system=c['system'],functional=c['functional'],
        images=c['xc_images'],momenta=k,weights=np.full(len(k),1/len(k)),densities=a.density,
        grid_options=[grid1,grid2],reference_index=1,grid_batch=64,budget=c['budget'])
    r = manta.assess_periodic_hybrid_xc_refinement(**controls)
    delta = a.fock-b.fock
    step = r.steps[0]
    assert step.energy_difference == pytest.approx(a.receipt.energy-b.receipt.energy,abs=2e-12)
    assert step.xc.source_identity == a.receipt.xc.source_identity
    assert r.steps[1].xc.source_identity == b.receipt.xc.source_identity
    np.testing.assert_allclose(np.array(step.potential_max_abs).reshape(spins,len(k)),np.max(abs(delta),axis=(-1,-2)),atol=2e-12,rtol=0)
    np.testing.assert_allclose(np.array(step.potential_frobenius).reshape(spins,len(k)),np.linalg.norm(delta,axis=(-1,-2)),atol=2e-12,rtol=0)
    for field in ('exchange_energy','exchange_correction_energy','one_electron_energy','coulomb_energy','nuclear_energy'):
        assert getattr(a.receipt,field) == getattr(b.receipt,field)


@pytest.mark.parametrize('field,quantity',CAPS)
@pytest.mark.parametrize('name',['pbe0',MIXED_RANGE])
def test_joint_exact_caps_and_one_short_before_poisoned_inputs(field,quantity,name):
    args,_ = case(name); p = plan(args); b = args['budget']; b.other_live_bytes=12345
    assert plan(args).modeled_peak_bytes == p.modeled_peak_bytes+12345
    p = plan(args)
    for field0,quantity0 in CAPS: setattr(b,field0,getattr(p,quantity0))
    r = manta.assess_periodic_hybrid_xc_refinement(**args)
    assert r.plan.modeled_peak_bytes == p.modeled_peak_bytes
    assert r.shell_calls <= p.shell_calls and r.primitive_points <= p.primitive_points
    setattr(b,field,getattr(p,quantity)-1); args['densities'][:] = np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'): plan(args)
    with pytest.raises((ValueError,RuntimeError),match='budget'): manta.assess_periodic_hybrid_xc_refinement(**args)


@pytest.mark.parametrize('name',['pbe','scan','tpssh','b2plyp','wb97x-v'])
def test_other_method_components_refuse_before_density_payload(name):
    args,_ = case(name); args['densities'][:] = np.nan
    with pytest.raises(ValueError,match='(complete|global hybrid)'): plan(args)
    with pytest.raises(ValueError,match='(complete|global hybrid)'): manta.assess_periodic_hybrid_xc_refinement(**args)


def test_yukawa_descriptor_is_not_treated_as_erfc():
    name='manta-refinement-yukawa-test'
    vq.define_functional(name,[('HYB_GGA_XC_CAMY_B3LYP',1.)])
    args,_=case(name);args['densities'][:]=np.nan
    with pytest.raises(ValueError,match='Coulomb/erf/erfc'):manta.assess_periodic_hybrid_xc_refinement(**args)


@pytest.mark.parametrize('fraction',[0.,.2])
def test_cancelled_cam_metadata_retains_resolved_partial_contract(fraction):
    name='manta-refinement-cancelled-'+str(fraction)
    vq.define_functional(name,[('HYB_GGA_XC_HSE06',1.),('HYB_GGA_XC_HSE06',-1.),
        ('GGA_X_B88',.8),('GGA_C_LYP',1.)],hf_exchange_fraction=fraction)
    args,c=case(name);r=manta.assess_periodic_hybrid_xc_refinement(**args)
    expected=independent(args,c)
    np.testing.assert_allclose(r.reference_potential,expected[2][1],atol=3e-11,rtol=0)
    assert r.plan.semilocal_component_only
    assert r.steps[0].xc.required_exchange_components==([('coulomb',fraction,0.)] if fraction else [])


@pytest.mark.parametrize('name',['pbe0',MIXED_RANGE])
def test_ownership_batches_layouts_reference_order_and_shell_image_gauge(name):
    args,_ = case(name); args['grid_options'] = args['grid_options'][:2]; args['reference_index']=0
    original=manta.assess_periodic_hybrid_xc_refinement(**args)
    for field in ('images','momenta','weights','densities'):
        r=manta.assess_periodic_hybrid_xc_refinement(**dict(args,**{field:np.asfortranarray(args[field])}))
        assert r.source_identity==original.source_identity and r.density_identity==original.density_identity
        np.testing.assert_array_equal(r.reference_potential,original.reference_potential)
    batch=manta.assess_periodic_hybrid_xc_refinement(**dict(args,grid_batch=1))
    assert batch.source_identity==original.source_identity
    np.testing.assert_allclose(batch.reference_potential,original.reference_potential,atol=3e-12,rtol=0)
    duplicate=manta.assess_periodic_hybrid_xc_refinement(**dict(args,grid_options=[args['grid_options'][0]]*2))
    assert duplicate.steps[1].energy_difference==0 and max(duplicate.steps[1].potential_max_abs)==0
    shells=args['basis'].shells(); lattice=args['system'].lattice
    shells[0].origin=np.array(shells[0].origin)+lattice[:,0]
    args['basis']=vq.BasisSet(vq.Molecule([vq.Atom(2,s.origin) for s in shells],0,1),shells,'hybrid-grid-image',False)
    phase=np.ones((len(args['momenta']),2),complex);phase[:,0]=np.exp(1j*args['momenta']@lattice[:,0])
    action=phase[:,:,None]*phase.conj()[:,None,:];args['densities']*=action
    moved=manta.assess_periodic_hybrid_xc_refinement(**args)
    assert moved.source_identity!=original.source_identity and moved.density_identity!=original.density_identity
    np.testing.assert_allclose(moved.reference_potential,original.reference_potential*action,atol=3e-11,rtol=0)
    for a,b in zip(original.steps,moved.steps):
        assert b.energy_difference==pytest.approx(a.energy_difference,abs=3e-12)
        np.testing.assert_allclose(b.potential_max_abs,a.potential_max_abs,atol=3e-11,rtol=0)
        np.testing.assert_allclose(b.potential_frobenius,a.potential_frobenius,atol=3e-11,rtol=0)


@pytest.mark.parametrize('name',['pbe0','cam-b3lyp'])
def test_grid_energy_defect_derivative_is_weighted_potential_defect(name):
    args,_=case(name);args['grid_options']=args['grid_options'][:2];args['reference_index']=1
    base=args['densities'].copy();rng=np.random.default_rng(3261)
    trial=rng.normal(size=base.shape)+1j*rng.normal(size=base.shape)
    delta=.02*(trial+trial.conj().swapaxes(-1,-2))
    r=manta.assess_periodic_hybrid_xc_refinement(**args)
    reverse=manta.assess_periodic_hybrid_xc_refinement(**dict(args,reference_index=0))
    potential=reverse.reference_potential-r.reference_potential
    analytic=np.einsum('k,skij,skji->',args['weights'],delta,potential).real
    h=1e-3;energy={}
    for multiple in (-2,-1,1,2):
        shifted=manta.assess_periodic_hybrid_xc_refinement(**dict(args,densities=base+multiple*h*delta))
        energy[multiple]=shifted.steps[0].energy_difference
    numeric=(energy[-2]-8*energy[-1]+8*energy[1]-energy[2])/(12*h)
    assert numeric==pytest.approx(analytic,abs=3e-10)


def test_external_functional_refuses_without_calling_provider():
    def provider(_): pytest.fail('External full-grid provider must not run')
    name='manta-hybrid-refinement-external-test'
    vq.define_external_functional(name,provider)
    args,_=case(name);args['densities'][:]=np.nan
    with pytest.raises(ValueError,match='complete native'):plan(args)
    with pytest.raises(ValueError,match='complete native'):manta.assess_periodic_hybrid_xc_refinement(**args)
