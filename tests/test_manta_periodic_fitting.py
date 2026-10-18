"""Same-source periodic erfc fitting: analytic sums, gauges and joint admission."""
import itertools

import numpy as np
import pytest
from scipy.special import hyp1f1

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_aux_eri_sr_lattice_enumeration import _small_sr_basis, _sr_three_center_oracle, _weighted_gradient_fixture

CAPS=[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
      ('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]


def case(shift=False):
    system,orbital,auxiliary=_small_sr_basis(shift)
    o=manta.PeriodicGDFShortRangeOptions();o.omega=.65;o.pair_cutoff_bohr=5.;o.auxiliary_cutoff_bohr=5.
    b=manta.ResourceBudget();b.memory_bytes=2**30;b.shell_calls=10**10;b.primitive_quartets=10**12
    b.scalar_terms=10**17;b.kernel_terms=2**64-1
    return dict(orbital=orbital,auxiliary=auxiliary,system=system,
        q=np.array([.17,-.08,.11]),ket_kpoints=np.array([[.23,.09,-.13],[-.15,.2,.31]]),options=o,budget=b)


def plan(args):
    a=dict(args);a.pop('q');a['momenta']=len(a.pop('ket_kpoints'))
    return manta.plan_periodic_gdf_short_range(**a)


def libint_reference(args):
    s,o,a=args['system'],args['orbital'],args['auxiliary'];options=args['options'];caps=(2**28,options.image_candidate_cap,2**28)
    metric=core.compute_gdf_sr_metric(a,s,args['q'],options.omega,options.auxiliary_cutoff_bohr,*caps,
        integral_screen_error=options.integral_screen_error)
    tensor=core.compute_gdf_sr_three_center(o,a,s,args['q'],args['ket_kpoints'],options.omega,
        options.pair_cutoff_bohr,options.auxiliary_cutoff_bohr,*caps,integral_screen_error=options.integral_screen_error)
    return metric,tensor


def analytic_metric(args):
    a=args['auxiliary'];o=args['options'];lattice=args['system'].lattice
    translations=np.array(list(itertools.product(range(-5,6),repeat=3)))@lattice.T
    out=np.zeros((a.nbasis,a.nbasis),complex)
    for i,si in enumerate(a.shells()):
        alpha=si.exponents[0]
        for j,sj in enumerate(a.shells()):
            beta=sj.exponents[0];r=np.array(si.origin)-sj.origin-translations
            r2=np.sum(r*r,axis=1);keep=r2<=o.auxiliary_cutoff_bohr**2
            rho=alpha*beta/(alpha+beta);eta=o.omega/np.sqrt(o.omega**2+rho);u=rho*r2[keep]
            raw=2*np.pi**2.5/(alpha*beta*np.sqrt(alpha+beta))*(hyp1f1(.5,1.5,-u)-eta*hyp1f1(.5,1.5,-eta**2*u))
            out[i,j]=np.exp(1j*translations[keep]@args['q'])@raw
    return out


@pytest.mark.parametrize('gamma',[True,False])
@pytest.mark.parametrize('pair,auxiliary',[(5.,5.),(9.,2.),(2.,6.)])
def test_independent_boys_double_image_source_and_counts(gamma,pair,auxiliary):
    a=case();o=a['options'];o.pair_cutoff_bohr=pair;o.auxiliary_cutoff_bohr=auxiliary
    if gamma:a['q'][:]=0.;a['ket_kpoints']=np.zeros((1,3))
    r=manta.compute_periodic_gdf_short_range(**a)
    expected_m=analytic_metric(a)
    expected_t=np.array([_sr_three_center_oracle(a['system'],a['orbital'],a['auxiliary'],a['q'],k,o.omega,pair,auxiliary) for k in a['ket_kpoints']])
    np.testing.assert_allclose(r.metric,expected_m,atol=5e-12,rtol=2e-13)
    np.testing.assert_allclose(r.three_center,expected_t,atol=5e-12,rtol=2e-13)
    legacy_m,legacy_t=libint_reference(a)
    np.testing.assert_allclose(r.metric,legacy_m,atol=5e-12,rtol=2e-13)
    np.testing.assert_allclose(r.three_center,legacy_t,atol=5e-12,rtol=2e-13)
    np.testing.assert_allclose(r.metric,r.metric.conj().T,atol=5e-12,rtol=0)
    assert r.receipt.provider=='manta' and r.receipt.threads==1
    assert r.receipt.convention=='raw-erfc;finite-zero-included;no-jellium-subtraction;no-exchange-correction'
    assert r.receipt.metric_shell_calls<=r.plan.metric_shell_calls
    assert r.receipt.three_center_shell_calls<=r.plan.three_center_shell_calls
    assert r.receipt.primitive_products<=r.plan.primitive_products
    assert r.plan.output_bytes==r.metric.nbytes+r.three_center.nbytes
    assert r.receipt.phase_products>r.receipt.metric_shell_calls+r.receipt.three_center_shell_calls
    assert r.receipt.phase_accumulations>=r.receipt.phase_products


@pytest.mark.parametrize('angular',[0,1,2,3])
@pytest.mark.parametrize('pure',[True,False])
def test_signed_long_contractions_angular_shells_and_complex_fit_actions(angular,pure):
    a=case();s,b,aux=_weighted_gradient_fixture(angular,orbital_angular=min(angular,2))
    def representation(basis):
        shells=basis.shells()
        for sh in shells:sh.pure=pure
        return vq.BasisSet(s.unit_cell_molecule(),shells,'periodic-fitting-angular',True)
    a.update(system=s,orbital=representation(b),auxiliary=representation(aux))
    a['options'].pair_cutoff_bohr=4.3;a['options'].auxiliary_cutoff_bohr=5.1
    r=manta.compute_periodic_gdf_short_range(**a);m,t=libint_reference(a)
    np.testing.assert_allclose(r.metric,m,atol=2e-11,rtol=1e-11)
    np.testing.assert_allclose(r.three_center,t,atol=2e-11,rtol=1e-11)
    # General complex pair trial, independent of a particular SCF or symmetry.
    # This tests algebra on this finite SR metric, not a full RSGDF fit.
    rng=np.random.default_rng(1753);flat=r.three_center.reshape(len(t),len(m),-1)
    trial=rng.normal(size=flat.shape[2])+1j*rng.normal(size=flat.shape[2])
    for actual,expected in zip(flat,t.reshape(flat.shape)):
        action=actual.conj().T@np.linalg.solve(r.metric,actual@trial)
        reference=expected.conj().T@np.linalg.solve(m,expected@trial)
        np.testing.assert_allclose(action,reference,atol=2e-9,rtol=2e-11)
        assert np.vdot(trial,action).real==pytest.approx(np.vdot(trial,reference).real,abs=2e-8)


def test_cell_image_gauges_for_metric_and_each_k_coupling():
    a=case();b=case(True);original=manta.compute_periodic_gdf_short_range(**a);moved=manta.compute_periodic_gdf_short_range(**b)
    L=a['system'].lattice[:,0];q=a['q'];aux_phase=np.exp(1j*np.array([0,q@L]))
    np.testing.assert_allclose(moved.metric,original.metric*aux_phase[:,None]*aux_phase.conj()[None],atol=6e-12,rtol=0)
    for ik,k in enumerate(a['ket_kpoints']):
        bra=np.exp(1j*np.array([0,(k-q)@L]));ket=np.exp(-1j*np.array([0,k@L]))
        np.testing.assert_allclose(moved.three_center[ik],original.three_center[ik]*aux_phase[:,None,None]*bra[None,:,None]*ket[None,None,:],atol=6e-12,rtol=0)
    for name in ('orbital_basis_identity','auxiliary_basis_identity','metric_source_identity','three_center_source_identity','source_identity'):
        assert getattr(original.receipt,name)!=getattr(moved.receipt,name)


def test_transfer_reciprocal_relabelling_conjugacy_and_shared_q_batches():
    a=case();r=manta.compute_periodic_gdf_short_range(**a)
    reciprocal=a['system'].reciprocal_lattice()@np.array([1,-1,2])
    relabel=manta.compute_periodic_gdf_short_range(**dict(a,q=a['q']+reciprocal,ket_kpoints=a['ket_kpoints']+reciprocal))
    np.testing.assert_allclose(relabel.metric,r.metric,atol=1e-11,rtol=0)
    np.testing.assert_allclose(relabel.three_center,r.three_center,atol=1e-11,rtol=0)
    conjugate=manta.compute_periodic_gdf_short_range(**dict(a,q=-a['q'],ket_kpoints=-a['ket_kpoints']))
    np.testing.assert_array_equal(conjugate.metric,r.metric.conj())
    np.testing.assert_array_equal(conjugate.three_center,r.three_center.conj())
    for i,k in enumerate(a['ket_kpoints']):
        single=manta.compute_periodic_gdf_short_range(**dict(a,ket_kpoints=k[None]))
        np.testing.assert_array_equal(single.three_center[0],r.three_center[i])
        assert single.receipt.three_center_shell_calls==r.receipt.three_center_shell_calls
        assert single.receipt.primitive_products==r.receipt.primitive_products
        assert single.receipt.metric_source_identity==r.receipt.metric_source_identity


@pytest.mark.parametrize('error',[1e-10,1e-5,1e-2])
def test_shared_screening_controls_retained_source_and_per_entry_error(error):
    a=case();a['system'],a['orbital'],a['auxiliary']=_weighted_gradient_fixture(1)
    a['options'].pair_cutoff_bohr=8.;a['options'].auxiliary_cutoff_bohr=20.
    exact=manta.compute_periodic_gdf_short_range(**a)
    a['options'].integral_screen_error=error;r=manta.compute_periodic_gdf_short_range(**a);m,t=libint_reference(a)
    np.testing.assert_allclose(r.metric,m,atol=2e-11,rtol=2e-12)
    np.testing.assert_allclose(r.three_center,t,atol=2e-11,rtol=2e-12)
    assert np.max(abs(r.metric-exact.metric))<=error+2e-11
    assert np.max(abs(r.three_center-exact.three_center))<=error+2e-11
    assert r.receipt.three_center_shell_calls<exact.receipt.three_center_shell_calls
    assert r.receipt.source_identity!=exact.receipt.source_identity


@pytest.mark.parametrize('field,quantity',CAPS)
@pytest.mark.parametrize('metric',[True,False])
def test_joint_resources_exact_then_one_short_before_numerical_copies(field,quantity,metric):
    a=case();a['options'].compute_metric=metric;a['budget'].other_live_bytes=12345;p=plan(a)
    for cap,value in CAPS:setattr(a['budget'],cap,getattr(p,value))
    r=manta.compute_periodic_gdf_short_range(**a)
    assert r.plan.modeled_peak_bytes==p.modeled_peak_bytes
    setattr(a['budget'],field,getattr(p,quantity)-1);a['q'][:]=np.nan;a['ket_kpoints'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):plan(a)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.compute_periodic_gdf_short_range(**a)


def test_optional_metric_exact_owned_exports_layout_and_component_identity():
    a=case();r=manta.compute_periodic_gdf_short_range(**a);m,t=r.metric.copy(),r.three_center.copy()
    r.metric[:]=np.nan;r.three_center[:]=np.nan
    np.testing.assert_array_equal(r.metric,m);np.testing.assert_array_equal(r.three_center,t)
    for name in ('q','ket_kpoints'):
        other=manta.compute_periodic_gdf_short_range(**dict(a,**{name:np.asfortranarray(a[name])}))
        assert other.receipt.source_identity==r.receipt.source_identity
        np.testing.assert_array_equal(other.three_center,t)
    p=plan(a);a['budget'].other_live_bytes+=777
    assert plan(a).modeled_peak_bytes==p.modeled_peak_bytes+777
    a['options'].image_candidate_cap*=2
    assert manta.compute_periodic_gdf_short_range(**a).receipt.source_identity==r.receipt.source_identity
    joint=plan(a)
    a['options'].compute_metric=False;only=manta.compute_periodic_gdf_short_range(**a)
    assert only.metric.shape==(0,0) and not only.receipt.metric_source_identity
    assert only.receipt.metric_shell_calls==only.plan.metric_shell_calls==0
    np.testing.assert_array_equal(only.three_center,t)
    assert only.receipt.three_center_source_identity==r.receipt.three_center_source_identity
    assert only.receipt.source_identity!=r.receipt.source_identity
    assert only.plan.modeled_peak_bytes<joint.modeled_peak_bytes


@pytest.mark.parametrize('fault',['dim','omega','pair','auxiliary','screen','candidates','momentum','dtype','strides','empty'])
def test_source_and_layout_refusals(fault):
    a=case();o=a['options']
    if fault=='dim':a['system']=vq.PeriodicSystem(2,a['system'].lattice,a['system'].unit_cell)
    elif fault=='omega':o.omega=0.
    elif fault=='pair':o.pair_cutoff_bohr=float('nan')
    elif fault=='auxiliary':o.auxiliary_cutoff_bohr=-1.
    elif fault=='screen':o.integral_screen_error=-1.
    elif fault=='candidates':o.image_candidate_cap=1
    elif fault=='momentum':a['q'][0]=np.inf
    elif fault=='dtype':a['ket_kpoints']=a['ket_kpoints'].astype(np.float32)
    elif fault=='strides':a['ket_kpoints']=a['ket_kpoints'][::-1]
    else:a['ket_kpoints']=np.empty((0,3))
    with pytest.raises((ValueError,RuntimeError,TypeError)):manta.compute_periodic_gdf_short_range(**a)


def test_count_overflow_before_allocating_input_or_output():
    a=case()
    with pytest.raises((ValueError,RuntimeError,OverflowError)):
        manta.plan_periodic_gdf_short_range(a['orbital'],a['auxiliary'],a['system'],2**63,a['options'],a['budget'])


@pytest.mark.parametrize('angular',[4,5,6])
def test_high_angular_origin_image_against_independent_radial_integral(angular):
    a=case();system=vq.PeriodicSystem(3,np.eye(3)*20.,[vq.Atom(2,[0.,0.,0.])])
    def basis(l,name):
        shell=core.ShellInfo(0,l,True,[.8],[1.],[0.,0.,0.])
        return vq.BasisSet(system.unit_cell_molecule(),[shell],name,False)
    a.update(system=system,orbital=basis(angular,'high-orbital'),auxiliary=basis(0,'s-auxiliary'))
    a['options'].pair_cutoff_bohr=a['options'].auxiliary_cutoff_bohr=1.
    r=manta.compute_periodic_gdf_short_range(**a)
    from scipy.integrate import quad
    from scipy.special import erf
    alpha=a['auxiliary'].shells()[0].exponents[0];beta=a['orbital'].shells()[0].exponents[0]
    ca=a['auxiliary'].shells()[0].coefficients[0];cb=a['orbital'].shells()[0].coefficients[0]
    gamma=np.sqrt(alpha)*a['options'].omega/np.sqrt(alpha+a['options'].omega**2)
    # Central erfc potential of an s Gaussian times two solid harmonics.
    # Angular orthogonality gives 4pi/(2l+1); only a 1D radial integral remains.
    integral,_=quad(lambda x:x**(2*angular+1)*np.exp(-2*beta*x*x)*(erf(np.sqrt(alpha)*x)-erf(gamma*x)),0,np.inf,epsabs=1e-13,epsrel=1e-13)
    value=ca*(np.pi/alpha)**1.5*cb**2*4*np.pi/(2*angular+1)*integral
    expected=np.repeat((value*np.eye(2*angular+1))[None,None],len(a['ket_kpoints']),axis=0)
    np.testing.assert_allclose(r.three_center,expected,atol=2e-12,rtol=1e-11)
    if angular<6:
        m,t=libint_reference(a)
        np.testing.assert_allclose(r.metric,m,atol=2e-12,rtol=1e-11)
        np.testing.assert_allclose(r.three_center,t,atol=2e-12,rtol=1e-11)
    assert r.receipt.three_center_shell_calls==1


def test_auxiliary_l7_admission_and_l8_or_orbital_l7_refusal():
    a=case();system=vq.PeriodicSystem(3,np.eye(3)*20.,[vq.Atom(2,[0.,0.,0.])])
    def basis(l):
        return vq.BasisSet(system.unit_cell_molecule(),[core.ShellInfo(0,l,True,[.8],[1.],[0.,0.,0.])],'envelope',False)
    a.update(system=system,orbital=basis(0),auxiliary=basis(7))
    a['options'].pair_cutoff_bohr=a['options'].auxiliary_cutoff_bohr=1.
    r=manta.compute_periodic_gdf_short_range(**a)
    from scipy.special import gamma
    # Fourier transform of r^l C_lm exp(-alpha*r^2); integrate the radial
    # erfc multiplier analytically and use solid-harmonic orthogonality.
    sh=a['auxiliary'].shells()[0];l=sh.l;alpha=sh.exponents[0];coefficient=sh.coefficients[0]
    width=1/(2*alpha);screened=width+1/(4*a['options'].omega**2)
    radial=.5*gamma(l+.5)*(width**(-l-.5)-screened**(-l-.5))
    value=(4*np.pi/(2*np.pi)**3)*(4*np.pi/(2*l+1))*coefficient**2*(np.pi/alpha)**3/(2*alpha)**(2*l)*radial
    np.testing.assert_allclose(r.metric,value*np.eye(2*l+1),atol=2e-12,rtol=1e-11)
    np.testing.assert_allclose(r.three_center,0.,atol=2e-12,rtol=0)
    for extra in ({'orbital':basis(7)},{'auxiliary':basis(8)}):
        with pytest.raises(ValueError,match='angular'):manta.compute_periodic_gdf_short_range(**dict(a,**extra))


def test_citation_route_for_primitive_and_periodic_source(tmp_path):
    from vibeqc.output.citations import assemble
    from tests.test_citations import _plan
    got=assemble(_plan(tmp_path,basis='sto-3g'),numerics=['manta_periodic_gdf_short_range'])
    keys={citation.bibtex_key for citation in got.citations}
    assert {'mcmurchie_davidson_1978','ye_berkelbach_rsgdf_2021'}<=keys


def test_metric_reuse_is_separate_from_orbital_source_and_pair_cutoff():
    a=case();r=manta.compute_periodic_gdf_short_range(**a)
    shells=a['orbital'].shells();shells[0].exponents=[.77]
    a['orbital']=vq.BasisSet(a['system'].unit_cell_molecule(),shells,'changed-orbital',True)
    a['options'].pair_cutoff_bohr=4.9
    b=manta.compute_periodic_gdf_short_range(**a)
    np.testing.assert_array_equal(b.metric,r.metric)
    assert b.receipt.metric_source_identity==r.receipt.metric_source_identity
    assert b.receipt.three_center_source_identity!=r.receipt.three_center_source_identity
    assert np.max(abs(b.three_center-r.three_center))>1e-5


def test_left_handed_lattice_basis_relabel_keeps_physical_image_set():
    a=case();r=manta.compute_periodic_gdf_short_range(**a)
    lattice=a['system'].lattice.copy();lattice[:,[0,1]]=lattice[:,[1,0]]
    a['system']=vq.PeriodicSystem(3,lattice,a['system'].unit_cell)
    b=manta.compute_periodic_gdf_short_range(**a)
    np.testing.assert_allclose(b.metric,r.metric,atol=5e-12,rtol=0)
    np.testing.assert_allclose(b.three_center,r.three_center,atol=5e-12,rtol=0)
    assert b.receipt.metric_shell_calls==r.receipt.metric_shell_calls
    assert b.receipt.three_center_shell_calls==r.receipt.three_center_shell_calls
