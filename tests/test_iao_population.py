"""Determinant IAO charges/spins and spin-resolved IAO-Wiberg analysis.

Analytical states and an optional, matched-basis PySCF construction provide
independent oracles; molecular charges retain the Knizia tests in test_iao_ibo.
"""
from __future__ import annotations

import itertools
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, Molecule, RHFOptions, UHFOptions, compute_overlap, run_rhf, run_uhf
from vibeqc.bond_analysis import iao_wiberg_bond_orders, wiberg_bond_orders
from vibeqc.iao import build_iaos, iao_reference
from vibeqc.iao_population import IAOAnalysis, analyse_iao, analyse_iao_occupied
from vibeqc.output.formats.population import PopulationSummary, format_iao_analysis, format_population_json


def _two(ca, cb=None):
    return analyse_iao_occupied(np.asarray(ca), np.eye(2), np.eye(2), np.eye(2),
                               np.ones(2), np.arange(2), occupied_beta=cb)


@pytest.mark.parametrize("ca,cb,charge,bond,spin", [
    ([[1.], [0.]], None, [-1., 1.], 0., None),
    (np.ones((2, 1))/np.sqrt(2), None, [0., 0.], 1., None),
    (np.ones((2, 1))/np.sqrt(2), np.empty((2, 0)), [.5, .5], .5, [.5, .5]),
    (np.array([[1., 1.], [1., -1.]])/np.sqrt(2), None, [-1., -1.], 0., None),
    ([[np.sqrt(.8)], [np.sqrt(.2)]], None, [-.6, .6], .64, None),
    ([[np.sqrt(.8)], [1j*np.sqrt(.2)]], np.empty((2, 0)), [.2, .8], .32, [.8, .2]),
])
def test_analytical_states(ca, cb, charge, bond, spin):
    a = _two(ca, cb)
    assert a.charges == pytest.approx(charge, abs=1e-12)
    assert a.bond_orders[0, 1] == pytest.approx(bond, abs=1e-12)
    assert a.bond_orders == pytest.approx(a.bond_orders.T, abs=1e-12)
    assert np.diag(a.bond_orders) == pytest.approx(0., abs=1e-12)
    assert np.all(a.bond_orders >= 0)
    assert np.isfinite(a.bond_orders).all()
    if spin is None:
        assert a.spin_populations is None
    else:
        assert a.spin_populations == pytest.approx(spin, abs=1e-12)
    assert max(abs(v) for k, v in a.diagnostics.items() if 'residual' in k) < 1e-12


def test_spin_resolved_formula_is_not_square_of_spin_sum():
    ca = np.array([[1.], [1j]]) / np.sqrt(2)
    cb = np.array([[1.], [-1j]]) / np.sqrt(2)
    a = _two(ca, cb)
    assert a.bond_orders[0, 1] == pytest.approx(1.)
    spin_summed = ca @ ca.conj().T + cb @ cb.conj().T
    assert abs(spin_summed[0, 1])**2 == pytest.approx(0., abs=1e-14)


def _fock_probabilities(c):
    """Independent Slater determinant probabilities in the occupation basis."""
    return [(rows, abs(np.linalg.det(c[list(rows), :]))**2)
            for rows in itertools.combinations(range(c.shape[0]), c.shape[1])]


def test_bond_index_is_minus_twice_determinant_number_covariance():
    rng = np.random.default_rng(17)
    ca = np.linalg.qr(rng.normal(size=(4, 2)) + 1j*rng.normal(size=(4, 2)))[0]
    cb = np.linalg.qr(rng.normal(size=(4, 1)) + 1j*rng.normal(size=(4, 1)))[0]
    means = np.zeros(2)
    nab = 0.
    for (a, wa), (b, wb) in itertools.product(_fock_probabilities(ca), _fock_probabilities(cb)):
        counts = np.bincount(np.array(a + b)//2, minlength=2)
        means += wa*wb*counts
        nab += wa*wb*counts[0]*counts[1]
    bo = iao_wiberg_bond_orders(ca@ca.conj().T, cb@cb.conj().T, np.array([0,0,1,1]), 2)
    assert bo[0, 1] == pytest.approx(-2*(nab - means.prod()), abs=1e-12)


def test_orbital_phases_rotations_atom_permutation_and_local_iao_rotations():
    rng = np.random.default_rng(23)
    c = np.linalg.qr(rng.normal(size=(6, 2)) + 1j*rng.normal(size=(6, 2)))[0]
    u = np.linalg.qr(rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)))[0]
    # A complete orthonormal reference makes A=I, independently of occupied MOs.
    labels = np.array([0, 0, 1, 1, 2, 2])
    def run(c, labs=labels):
        return analyse_iao_occupied(c, np.eye(6), np.eye(6), np.eye(6), np.array([2,2,2]), labs)
    a, b = run(c), run(c@u)
    assert b.charges == pytest.approx(a.charges, abs=1e-12)
    assert b.bond_orders == pytest.approx(a.bond_orders, abs=1e-12)
    perm = np.array([2,0,1])
    reordered = run(c, perm[labels])
    assert reordered.charges[perm] == pytest.approx(a.charges, abs=1e-12)
    assert reordered.bond_orders[np.ix_(perm,perm)] == pytest.approx(a.bond_orders, abs=1e-12)
    local = np.zeros((6,6), complex)
    for i in range(0,6,2):
        local[i:i+2,i:i+2] = u
    d = c@c.conj().T
    rotated = local.conj().T@d@local
    assert iao_wiberg_bond_orders(rotated, rotated, labels, 3) == pytest.approx(a.bond_orders, abs=1e-12)


def test_complex_nonorthogonal_ao_factorization():
    t = np.array([[1.1, .2j], [.3, 1.4]])
    s = t.conj().T @ t
    c = np.linalg.solve(t, np.array([[1.], [1j]])/np.sqrt(2))
    ref = np.linalg.inv(t)
    a = analyse_iao_occupied(c, s, np.eye(2), s@ref, np.ones(2), np.arange(2))
    p = 2*c@c.conj().T
    d = a.iaos_alpha.conj().T@s@p@s@a.iaos_alpha
    assert a.populations == pytest.approx(d.diagonal().real, abs=1e-12)
    assert a.bond_orders[0,1] == pytest.approx(abs(d[0,1])**2, abs=1e-12)
    assert a.bond_orders[0,1] == pytest.approx(1.)


@pytest.mark.parametrize("which", ["ao", "reference", "projected", "nonhermitian", "nan", "labels", "occupied"])
def test_invalid_or_rank_deficient_spaces_fail_explicitly(which):
    s, s2, s12 = np.eye(2), np.eye(2), np.eye(2)
    labels = np.arange(2)
    c = np.array([[1.], [0.]])
    if which == 'ao': s[1,1] = 1e-12
    if which == 'reference': s2[1,1] = 1e-12
    if which == 'projected': s12[1,1] = 1e-12
    if which == 'nonhermitian': s[0,1] = .2
    if which == 'nan': s12[0,1] = np.nan
    if which == 'labels': labels = np.array([0,0])
    if which == 'occupied': c *= 2
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        analyse_iao_occupied(c,s,s2,s12,np.ones(2),labels)


@pytest.fixture(scope='module')
def h2():
    m = Molecule([Atom(1,[0,0,0]), Atom(1,[0,0,1.4])], charge=0, multiplicity=1)
    b = BasisSet(m, 'def2-svp')
    r = run_rhf(m,b,RHFOptions())
    assert r.converged
    return m,b,r


def test_molecular_h2_and_unrestricted_closed_shell_limit(h2):
    m,b,r = h2
    a = analyse_iao(r,b,m)
    assert a.available, a.unavailable_reason
    assert a.charges == pytest.approx([0.,0.], abs=1e-9)
    assert a.bond_orders[0,1] == pytest.approx(1., abs=1e-9)
    u = run_uhf(m,b,UHFOptions())
    au = analyse_iao(u,b,m)
    assert au.available, au.unavailable_reason
    assert au.bond_orders == pytest.approx(a.bond_orders, abs=1e-8)
    assert au.spin_populations == pytest.approx([0.,0.], abs=1e-8)


def test_empty_spin_molecular_h2_ion():
    m = Molecule([Atom(1,[0,0,0]), Atom(1,[0,0,2.])], charge=1, multiplicity=2)
    b = BasisSet(m,'def2-svp')
    r = run_uhf(m,b,UHFOptions())
    a = analyse_iao(r,b,m)
    assert a.available, a.unavailable_reason
    assert a.charges == pytest.approx([.5,.5], abs=1e-8)
    assert a.spin_populations.sum() == pytest.approx(1.)
    assert a.bond_orders[0,1] == pytest.approx(.5, abs=1e-8)


@pytest.mark.parametrize('case', ['H2O','CH4','HCN','benzene','OH'])
def test_small_molecules(case):
    from tests.test_iao_ibo import CH4_ATOMS, HCN_ATOMS, _benzene_atoms
    atoms = {'H2O': [(8,[0,0,0]),(1,[0,1.43,1.11]),(1,[0,-1.43,1.11])],
             'CH4': CH4_ATOMS, 'HCN': HCN_ATOMS, 'benzene': _benzene_atoms(),
             'OH': [(8,[0,0,0]),(1,[0,0,1.83])]}[case]
    m = Molecule([Atom(z,p) for z,p in atoms],charge=0,multiplicity=2 if case=='OH' else 1)
    b = BasisSet(m,'def2-svp')
    r = run_uhf(m,b,UHFOptions()) if case=='OH' else run_rhf(m,b,RHFOptions())
    a = analyse_iao(r,b,m)
    assert a.available, a.unavailable_reason
    assert a.charges.sum() == pytest.approx(0., abs=1e-7)
    assert a.populations.sum() == pytest.approx(m.n_electrons(), abs=1e-7)
    assert a.bond_orders == pytest.approx(a.bond_orders.T, abs=1e-12)
    if case=='OH':
        assert a.spin_populations.sum() == pytest.approx(1.,abs=1e-7)
        assert .5 < a.bond_orders[0,1] < 1.5
    if case=='HCN': assert 2.3 < a.bond_orders[1,2] < 3.2
    if case=='benzene':
        adjacent = [a.bond_orders[2*k,2*((k+1)%6)] for k in range(6)]
        assert adjacent == pytest.approx([adjacent[0]]*6,abs=1e-6)
        assert 1.1 < adjacent[0] < 1.8


@pytest.mark.parametrize('change,reason', [
    ({'density_beta':None},'Incomplete'),
    ({'occupations_alpha':np.array([.5,.5])},'Fractional'),
    ({'density_alpha':np.eye(2)},'density'),
    ({'converged':False},'converged'),
    ({'n_alpha':2},'n_alpha'),
])
def test_inconsistent_adapter_inputs(monkeypatch,change,reason):
    import vibeqc._vibeqc_core as core
    monkeypatch.setattr(core,'compute_overlap',lambda b:np.eye(2))
    m = Molecule([Atom(1,[0,0,0]),Atom(1,[0,0,1.4])],charge=0,multiplicity=1)
    c = np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
    p = c[:,:1]@c[:,:1].T
    fields = dict(method='uhf',converged=True,mo_coeffs_alpha=c,mo_coeffs_beta=c,
                  density_alpha=p,density_beta=p)
    fields.update(change)
    a = analyse_iao(SimpleNamespace(**fields),object(),m)
    assert not a.available
    assert reason.lower() in a.unavailable_reason.lower()
    assert a.charges is None and a.bond_orders is None


@pytest.mark.parametrize('kwargs,reason', [({'uses_ecp':True},'ECP'),({'periodic':True},'Periodic'),({'method':'mp2'},'determinants')])
def test_unsupported_analysis(h2,kwargs,reason):
    m,b,r = h2
    a = analyse_iao(r,b,m,**kwargs)
    assert not a.available
    assert reason in a.unavailable_reason
    payload = json.loads(format_population_json(PopulationSummary(iao_analysis=a)))['iao']
    assert payload['available'] is False and payload['charges'] is None
    assert 'unavailable' in format_iao_analysis(a)


def test_unsupported_ghost_and_reference_elements():
    from vibeqc.iao import iao_unsupported_reason
    for z in (0,87):
        m = SimpleNamespace(atoms=[SimpleNamespace(Z=z)])
        assert iao_unsupported_reason(m)


def test_display_threshold_never_changes_dense_result(h2):
    m,b,r = h2
    a = analyse_iao(r,b,m)
    assert 'No atom pairs' in format_iao_analysis(a,bond_threshold=2.)
    assert '0\t1\t1.00000000' in format_iao_analysis(a,bond_threshold=.05)
    assert a.to_dict()['bond_orders'][0][1] == pytest.approx(1.)


def test_matched_reference_pyscf_oracle(h2, monkeypatch):
    """PySCF's independent IAO algebra with exactly the native overlap inputs.

    This isolates AO ordering/integral normalization: B1, Huzinaga MINI B2,
    geometry, occupied MOs and reference atom labels are identical. PySCF's
    raw output is then symmetrically metric-orthonormalized before comparison.
    """
    pyscf = pytest.importorskip('pyscf')
    from pyscf.lo import iao, orth
    m,b,r = h2
    ref = iao_reference(m,b)
    s = np.asarray(compute_overlap(b))
    fake = SimpleNamespace(has_ecp=lambda:False, intor_symmetric=lambda key:s, verbose=0)
    fake_ref = SimpleNamespace(intor_symmetric=lambda key:ref.overlap)
    monkeypatch.setattr(iao,'reference_mol',lambda mol,minao:fake_ref)
    monkeypatch.setattr(iao.gto.mole,'intor_cross',lambda key,mol,pmol:ref.cross_overlap)
    c = np.asarray(r.mo_coeffs)[:,:1]
    raw = iao.iao(fake,c,minao='mini')
    oracle = raw@orth.lowdin(raw.conj().T@s@raw)
    a = analyse_iao(r,b,m)
    assert a.iaos_alpha == pytest.approx(oracle, abs=1e-9)
    d = 2*(oracle.conj().T@s@c)@(oracle.conj().T@s@c).conj().T
    assert a.charges == pytest.approx(1-d.diagonal().real,abs=1e-9)
    assert a.bond_orders[0,1] == pytest.approx(abs(d[0,1])**2,abs=1e-9)


@pytest.mark.parametrize("method", ["rhf", "rks", "uhf", "uks"])
def test_public_runner_without_localization_or_qvf(tmp_path, monkeypatch, method):
    import vibeqc.iao as iao
    from vibeqc import run_job
    def forbidden(*args,**kwargs):
        raise AssertionError('Population analysis requested localization or dipole integrals')
    monkeypatch.setattr(iao,'analyse_localization',forbidden)
    monkeypatch.setattr(iao,'_dipole_tensor',forbidden)
    m = Molecule([Atom(1,[0,0,0]),Atom(1,[0,0,1.4])],
                 charge=1 if method in ('uhf','uks') else 0,
                 multiplicity=2 if method in ('uhf','uks') else 1)
    stem=tmp_path/'h2'
    r = run_job(m,basis='def2-svp',method=method,
                functional='pbe' if method in ('rks','uks') else None,
                output=stem,name_molecule=False,
                iao_analysis=True,localize=False,output_qvf=False,write_molden_file=False,
                structured_log=True)
    assert r.iao_analysis.available, r.iao_analysis.unavailable_reason
    assert r.energy < 0.
    assert not stem.with_suffix('.qvf').exists()
    assert 'IAO-Wiberg' in stem.with_suffix('.out').read_text()
    payload=json.loads((tmp_path/'h2.population.json').read_text())
    assert payload['iao']['bond_orders'][0][1] == pytest.approx(
        .5 if method in ('uhf','uks') else 1., abs=1e-8)
    if method in ('uhf','uks'):
        assert sum(payload['iao']['spin_populations']) == pytest.approx(1., abs=1e-8)
    events=[json.loads(line) for line in (tmp_path/'h2.scf.jsonl').read_text().splitlines()]
    assert any('iao' in event for event in events)
    bib=stem.with_suffix('.bibtex').read_text()
    assert 'knizia_ibo_2013' in bib and 'wiberg_bond_index_1968' in bib


def test_public_runner_qvf_reuses_iao_build(h2,tmp_path,monkeypatch):
    import vibeqc.iao as iao
    import vibeqc.iao_population as populations
    from vibeqc import run_job
    calls=[]
    original=iao.build_iaos
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(iao,'build_iaos',counted)
    monkeypatch.setattr(populations,'build_iaos',counted)
    m,_,_=h2
    stem=tmp_path/'h2'
    r=run_job(m,basis='def2-svp',method='rhf',output=stem,name_molecule=False,
              iao_analysis=True,localize=['ibo','boys','pipek-mezey'],output_qvf=True,
              write_population_file=False,write_molden_file=False)
    assert r.iao_analysis.available
    assert len(calls)==1
    with zipfile.ZipFile(stem.with_suffix('.qvf')) as z:
        assert np.frombuffer(z.read('atom_properties/iao_charge.bin'),dtype='<f8') == pytest.approx(r.iao_analysis.charges)
        manifest=json.loads(z.read('manifest.json'))
        assert any(s['kind']=='x_vibeqc.iao_analysis' for s in manifest['sections'])
        assert json.loads(z.read('analysis/iao.json'))['method']=='iao-wiberg'


def test_periodic_request_gated_before_calculation():
    from vibeqc import run_periodic_job
    with pytest.raises(ValueError,match='Periodic IAO'):
        run_periodic_job(None,None,iao_analysis=True)


def test_diffuse_basis_and_wrong_geometry():
    m = Molecule([Atom(1,[0,0,0]),Atom(1,[0,0,1.4])],charge=0,multiplicity=1)
    b = BasisSet(m,'aug-cc-pvdz')
    r = run_rhf(m,b,RHFOptions())
    a = analyse_iao(r,b,m)
    assert a.available, a.unavailable_reason
    assert a.charges.sum() == pytest.approx(0.,abs=1e-8)
    displaced = Molecule([Atom(1,[0,0,0]),Atom(1,[0,0,1.7])],charge=0,multiplicity=1)
    invalid = analyse_iao(r,b,displaced)
    assert not invalid.available
    assert 'origins' in invalid.unavailable_reason


def test_public_runner_records_unavailable_request(h2,tmp_path,monkeypatch):
    import vibeqc.iao_population as populations
    from vibeqc import run_job
    reason='IAO reference space is rank deficient'
    monkeypatch.setattr(populations,'analyse_iao',lambda *a,**k:IAOAnalysis(unavailable_reason=reason))
    m,_,_=h2
    stem=tmp_path/'unavailable'
    result=run_job(m,basis='def2-svp',method='rhf',output=stem,name_molecule=False,
                   iao_analysis=True,localize=False,output_qvf=True,write_molden_file=False)
    assert not result.iao_analysis.available
    assert reason in stem.with_suffix('.out').read_text()
    payload=json.loads((tmp_path/'unavailable.population.json').read_text())
    assert payload['unavailable']['iao']==reason
    assert payload['iao']['charges'] is None
    with zipfile.ZipFile(stem.with_suffix('.qvf')) as z:
        assert json.loads(z.read('analysis/iao.json'))['unavailable_reason']==reason
        assert 'atom_properties/iao_charge.bin' not in z.namelist()


def test_pyscf_water_matched_mini_oracle(monkeypatch):
    pytest.importorskip('pyscf')
    from pyscf.lo import iao, orth
    m=Molecule([Atom(8,[0,0,0]),Atom(1,[0,1.43,1.11]),Atom(1,[0,-1.43,1.11])],charge=0,multiplicity=1)
    b=BasisSet(m,'def2-svp')
    r=run_rhf(m,b,RHFOptions())
    s=np.asarray(compute_overlap(b))
    ref=iao_reference(m,b)
    fake=SimpleNamespace(has_ecp=lambda:False,intor_symmetric=lambda key:s,verbose=0)
    monkeypatch.setattr(iao,'reference_mol',lambda mol,minao:SimpleNamespace(intor_symmetric=lambda key:ref.overlap))
    monkeypatch.setattr(iao.gto.mole,'intor_cross',lambda key,mol,pmol:ref.cross_overlap)
    c=np.asarray(r.mo_coeffs)[:,:5]
    raw=iao.iao(fake,c,minao='mini')
    oracle=raw@orth.lowdin(raw.conj().T@s@raw)
    a=analyse_iao(r,b,m)
    assert a.available, a.unavailable_reason
    assert a.iaos_alpha==pytest.approx(oracle,abs=1e-9)
    x=oracle.conj().T@s@c
    d=2*x@x.conj().T
    pops=np.bincount(ref.atom_indices,weights=d.diagonal().real)
    assert a.populations==pytest.approx(pops,abs=1e-9)
    for i,j in itertools.combinations(range(3),2):
        block=d[np.ix_(ref.atom_indices==i,ref.atom_indices==j)]
        assert a.bond_orders[i,j]==pytest.approx(np.sum(abs(block)**2),abs=1e-9)


def test_separate_spin_spaces_can_exceed_reference_in_union():
    ca=np.eye(3)[:,:2]
    cb=np.eye(3)[:,2:]
    cross=np.array([[1/np.sqrt(2),0.],[0.,1.],[1/np.sqrt(2),0.]])
    a=analyse_iao_occupied(ca,np.eye(3),np.eye(2),cross,np.ones(2),np.arange(2),occupied_beta=cb)
    assert a.populations.sum()==pytest.approx(3.)
    assert a.spin_populations.sum()==pytest.approx(1.)
    assert max(a.diagnostics[k] for k in a.diagnostics if 'span_residual' in k)<1e-12


def test_correlated_runner_does_not_relabel_its_reference(tmp_path):
    from vibeqc import run_job
    m=Molecule([Atom(1,[0,0,0]),Atom(1,[0,0,1.4])],charge=0,multiplicity=1)
    r=run_job(m,basis='sto-3g',method='mp2',output=tmp_path/'mp2',name_molecule=False,
              iao_analysis=True,localize=False,output_qvf=False,write_molden_file=False)
    assert not r.iao_analysis.available
    assert 'determinants' in r.iao_analysis.unavailable_reason


@pytest.mark.parametrize('density', [np.diag([.5,.5]), np.diag([1e200,0.]), np.array([[1.,1j],[1j,0.]])])
def test_bond_contraction_rejects_invalid_spin_densities(density):
    with pytest.raises(ValueError):
        iao_wiberg_bond_orders(density,np.zeros((2,2)),np.arange(2),2)


def test_molecular_adapter_rejects_periodic_geometry_before_integrals():
    a=analyse_iao(SimpleNamespace(),None,SimpleNamespace(lattice=np.eye(3)))
    assert not a.available
    assert 'Periodic IAO' in a.unavailable_reason


def test_unavailable_mp2_analysis_preserves_reference_localization(tmp_path):
    """Population support must not change separately requested SCF localization."""
    from vibeqc import run_job

    m = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])],
                 charge=0, multiplicity=1)
    reference_charges = []
    for requested in (False, True):
        stem = tmp_path / ("analysis-on" if requested else "analysis-off")
        result = run_job(
            m, basis="sto-3g", method="mp2", output=stem, name_molecule=False,
            iao_analysis=requested, localize="ibo", output_qvf=True,
            write_population_file=False, write_molden_file=False,
        )
        assert result.converged
        with zipfile.ZipFile(stem.with_suffix(".qvf")) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            sections = {section["id"]: section for section in manifest["sections"]}
            assert sections["wf_localized_ibo"]["kind"] == "wavefunction.gto"
            reference_charges.append(np.frombuffer(
                archive.read("atom_properties/iao_charge.bin"), dtype="<f8"))
            if requested:
                assert not result.iao_analysis.available
                analysis = json.loads(archive.read("analysis/iao.json"))
                assert not analysis["available"]
                assert analysis["charges"] is None
                assert "determinants" in analysis["unavailable_reason"]
    assert reference_charges[1] == pytest.approx(reference_charges[0], abs=1e-12)
