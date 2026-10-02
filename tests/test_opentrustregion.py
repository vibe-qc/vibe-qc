"""Pinned native OTR adapter, manifold derivatives, and molecular public routes.

Derivative tests use actual C exp(K) coordinates away from stationarity. They
run without the optional solver; only real-library tests are capability gated.
Greiner et al., JCTC 22, 881 (2026), doi:10.1021/acs.jctc.5c01576.
"""
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import pytest
from scipy.linalg import expm
import vibeqc as vq
from vibeqc import _vibeqc_core as core

requires_otr = pytest.mark.skipif(not vq.has_opentrustregion(), reason="optional OpenTrustRegion build")


def water():
    return vq.Molecule([vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 1.43, 1.1]), vq.Atom(1, [0, -1.43, 1.1])])


def options(method, otr=False):
    o = getattr(vq, method.upper() + "Options")()
    o.initial_guess = vq.InitialGuess.HCORE
    o.stability_check = False
    o.max_iter = 100
    o.conv_tol_energy = 1e-9
    o.conv_tol_grad = 1e-7
    if method.endswith("ks"):
        o.functional = "PBE"
        o.grid.n_radial = 25
        o.grid.n_theta = 12
        o.grid.n_phi = 24
    if otr:
        o.orbital_optimizer = "opentrustregion"
    return o


@pytest.fixture(scope="module")
def molecule_data():
    mol = water()
    basis = vq.BasisSet(mol, "sto-3g")
    s = np.asarray(vq.compute_overlap(basis))
    ref = vq.run_rhf(mol, basis, options("rhf"))
    jk = core.make_four_index_jk_builder(basis)
    d = np.asarray(ref.density)
    h = np.asarray(ref.fock) - jk.build_J(d) + 0.5 * jk.build_K(d)
    go = vq.GridOptions()
    go.n_radial = 25
    go.n_theta = 12
    go.n_phi = 24
    grid = vq.build_grid(mol, go)
    return mol, basis, s, ref, jk, h, grid


def rotation(c, v, n):
    k = np.zeros((c.shape[1], c.shape[1]))
    k[n:, :n] = np.asarray(v).reshape((c.shape[1] - n, n), order="F")
    k[:n, n:] = -k[n:, :n].T
    return c @ expm(k)


def probe(data, ca, cb, restricted=True, functional="", na=5, nb=5):
    mol, basis, s, ref, jk, h, grid = data
    exchange = vq.Functional(functional).hf_exchange_fraction if functional else 1.0
    return core._MolecularOrbitalObjective(s, h, 0.0, jk, ca, cb, na, nb,
                                           restricted, exchange, basis, grid, functional)


@pytest.mark.parametrize("restricted", [True, False])
@pytest.mark.parametrize("functional", ["", "LDA", "PBE", "PBE0"])
def test_off_stationary_derivatives_and_trial_isolation(molecule_data, restricted, functional):
    rng = np.random.default_rng(29)
    c = np.asarray(molecule_data[3].mo_coeffs)
    # Noncanonical occupied/virtual gauges plus occupied-virtual displacement.
    q = rng.normal(size=(c.shape[1], c.shape[1]))
    ca = c @ expm(0.15 * (q - q.T))
    cb = c @ expm(0.12 * (q.T - q))
    p = probe(molecule_data, ca, cb, restricted, functional)
    n = p.size()
    zero = np.zeros(n)
    m = p.update(zero)
    u, v = rng.normal(size=(2, n))
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    hv = np.asarray(p.hessian(v))
    assert u @ hv == pytest.approx(v @ p.hessian(u), abs=2e-9)
    eps = 2e-5
    de = (p.trial(eps*v) - p.trial(-eps*v))/(2*eps)
    assert de == pytest.approx(np.asarray(m.gradient) @ v, abs=3e-7)
    # Repeated trials in arbitrary order leave energy/model and response fixed.
    eplus = p.trial(eps*v)
    p.trial(-0.2*u)
    assert p.trial(eps*v) == pytest.approx(eplus, abs=1e-12)
    np.testing.assert_allclose(p.hessian(v), hv, atol=2e-11)
    assert p.trial(zero) == pytest.approx(m.energy, abs=1e-12)
    # Local gradients on displaced references differentiate to the local
    # Hessian at zero. OO/VV gauge commutators have zero energy gradient.
    off = 5*(c.shape[1]-5)
    grads = []
    for sign in (1, -1):
        pa = rotation(ca, sign*eps*v[:off], 5)
        pb = cb if restricted else rotation(cb, sign*eps*v[off:], 5)
        displaced = probe(molecule_data, pa, pb, restricted, functional)
        grads.append(np.asarray(displaced.update(zero).gradient))
    np.testing.assert_allclose((grads[0]-grads[1])/(2*eps), hv, rtol=2e-5, atol=2e-6)


def test_closed_shell_spin_limit_and_cross_response(molecule_data):
    c = np.asarray(molecule_data[3].mo_coeffs)
    rng = np.random.default_rng(17)
    c = rotation(c, rng.normal(scale=0.08, size=10), 5)
    for functional in ("", "LDA", "PBE"):
        r = probe(molecule_data, c, c, True, functional)
        u = probe(molecule_data, c, c, False, functional)
        rm, um = r.update(np.zeros(10)), u.update(np.zeros(20))
        assert rm.energy == pytest.approx(um.energy, abs=1e-9)
        np.testing.assert_allclose(rm.gradient, 2*np.asarray(um.gradient)[:10], atol=1e-9)
        v = rng.normal(size=10)
        np.testing.assert_allclose(r.hessian(v), 2*np.asarray(u.hessian(np.r_[v,v]))[:10], atol=2e-8)
        assert np.linalg.norm(np.asarray(u.hessian(np.r_[v,np.zeros(10)]))[10:]) > 1e-3


def quadratic(n=4, fail=None, delay=False):
    state = {"x": np.full(n, 0.2), "trials": [], "updates": []}
    diag = np.linspace(1, 2, n)
    def update(k):
        if fail == "update":
            raise RuntimeError("deliberate update failure")
        if delay:
            time.sleep(0.005)
        state["x"] = state["x"] + k
        state["updates"].append(state["x"].copy())
        x = state["x"]
        return 0.5*np.dot(x,diag*x), diag*x, diag
    def trial(k):
        if fail == "trial":
            raise RuntimeError("deliberate trial failure")
        state["trials"].append(k.copy())
        x = state["x"] + k
        return float("nan") if fail == "nonfinite" else 0.5*np.dot(x,diag*x)
    def hessian(v):
        if fail == "hessian":
            raise RuntimeError("deliberate hessian failure")
        return diag*v
    return state, update, trial, hessian


@requires_otr
@pytest.mark.parametrize("subsystem", ["davidson", "jacobi-davidson", "tcg"])
def test_quadratic_real_library(subsystem):
    state, *callbacks = quadratic()
    o = vq.OpenTrustRegionOptions()
    o.subsystem_solver = subsystem
    o.line_search = True
    report = core._otr_minimize(4, *callbacks, o, 60)
    assert report.error_code == 0, report.log
    assert report.stability_checked and report.stability_converged and report.stable
    assert np.linalg.norm(state["x"]) < 1e-7
    assert report.response_evaluations > 0 and report.trial_evaluations > 0
    assert report.micro_iterations == -1 and report.macro_iterations == -1


@requires_otr
def test_saddle_escape_and_tight_stability_threshold():
    # E=(x^2-1)^2+y^2, stationary indefinite origin. Upstream must escape.
    x = np.zeros(2)
    def update(k):
        nonlocal x
        x = x+k
        return (x[0]**2-1)**2+x[1]**2, np.array([4*x[0]*(x[0]**2-1),2*x[1]]), np.array([12*x[0]**2-4,2])
    def trial(k):
        y=x+k
        return (y[0]**2-1)**2+y[1]**2
    def hessian(v):
        return np.array([12*x[0]**2-4,2])*v
    o=vq.OpenTrustRegionOptions()
    o.stability="follow"
    r=core._otr_minimize(2,update,trial,hessian,o,100)
    assert r.error_code == 0, r.log
    assert abs(abs(x[0])-1) < 1e-6
    assert r.stable
    # Curvature -0.005 passes upstream's fixed -0.01 but must fail our
    # requested -0.0001 final verdict without automatically changing state.
    state=np.zeros(2)
    r=core._otr_minimize(2,lambda k:(0.,state,np.array([-.005,1.])),lambda k:0.,
                         lambda v:np.array([-.005,1.])*v,vq.OpenTrustRegionOptions(),4)
    assert r.error_code == 0
    assert r.stability_converged and not r.stable


@requires_otr
def test_inconclusive_embedded_stability_is_not_stable():
    rng = np.random.default_rng(41)
    q, _ = np.linalg.qr(rng.normal(size=(50, 50)))
    h = (q * np.linspace(0.1, 5.0, 50)) @ q.T
    o = vq.OpenTrustRegionOptions()
    o.stability_max_iterations = 1
    r = core._otr_minimize(50, lambda k: (0.0, np.zeros(50), np.diag(h)),
                          lambda k: 0.0, lambda v: h @ v, o, 4)
    assert r.error_code == 202 and r.termination == "stability_inconclusive"
    assert r.stability_checked and not r.stability_converged and not r.stable


@requires_otr
@pytest.mark.parametrize("fail", ["update", "trial", "hessian", "nonfinite"])
def test_callback_errors_preserve_last_accepted_state(fail):
    state,*callbacks=quadratic(fail=fail)
    r=core._otr_minimize(4,*callbacks,vq.OpenTrustRegionOptions(),30)
    assert r.termination == "callback_failure"
    assert r.callback_error
    assert np.isfinite(state["x"]).all()
    if state["updates"]:
        np.testing.assert_array_equal(state["x"],state["updates"][-1])
    # Failure leaves Fortran globals reusable.
    state,*callbacks=quadratic()
    assert core._otr_minimize(4,*callbacks,vq.OpenTrustRegionOptions(),30).error_code == 0


@requires_otr
def test_zero_dimension_exhaustion_nested_and_concurrent():
    state,*callbacks=quadratic(0)
    r=core._otr_minimize(0,*callbacks,vq.OpenTrustRegionOptions(),3)
    assert r.termination == "no_rotation_parameters" and r.stable
    state,*callbacks=quadratic()
    r=core._otr_minimize(4,*callbacks,vq.OpenTrustRegionOptions(),1)
    assert r.error_code == 102 and r.termination == "iteration_limit"
    def nested(k):
        _,*inner=quadratic()
        core._otr_minimize(4,*inner,vq.OpenTrustRegionOptions(),10)
    _,_,trial,hessian=quadratic()
    r=core._otr_minimize(4,nested,trial,hessian,vq.OpenTrustRegionOptions(),10)
    assert r.termination == "callback_failure" and "nested entry" in r.callback_error
    def solve(_):
        state,*cb=quadratic(delay=True)
        r=core._otr_minimize(4,*cb,vq.OpenTrustRegionOptions(),40)
        return r.error_code,np.linalg.norm(state["x"])
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(solve,range(6)))
    assert all(code == 0 and norm < 1e-7 for code,norm in results)


@requires_otr
@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
def test_small_molecules_native_parity_and_restart(method):
    mol = water()
    if method in ("uhf", "uks"):
        # Bent water cation avoids the freely orientable pi-hole density of
        # linear OH, so density comparisons select a nondegenerate solution.
        mol = vq.Molecule(water().atoms, charge=1, multiplicity=2)
    basis=vq.BasisSet(mol,"sto-3g")
    native_opts,otr_opts=options(method),options(method,True)
    native=getattr(vq,"run_"+method)(mol,basis,native_opts)
    otr=getattr(vq,"run_"+method)(mol,basis,otr_opts)
    assert native.converged and otr.converged, (otr.opentrustregion.termination,otr.opentrustregion.log)
    assert otr.energy == pytest.approx(native.energy,abs=2e-8)
    assert otr.opentrustregion.final_residual < otr_opts.conv_tol_grad
    s=np.asarray(vq.compute_overlap(basis))
    attrs=[("mo_coeffs","density")] if method in ("rhf","rks") else [("mo_coeffs_alpha","density_alpha"),("mo_coeffs_beta","density_beta")]
    for coeff,dens in attrs:
        c=np.asarray(getattr(otr,coeff));d=np.asarray(getattr(otr,dens))
        np.testing.assert_allclose(c.T@s@c,np.eye(c.shape[1]),atol=2e-10)
        np.testing.assert_allclose(d,getattr(native,dens),atol=2e-6)
    otr_opts.initial_guess=vq.InitialGuess.READ
    if method in ("rhf","rks"):
        otr_opts.read_density=otr.density
    else:
        otr_opts.read_density_alpha=otr.density_alpha
        otr_opts.read_density_beta=otr.density_beta
    restart=getattr(vq,"run_"+method)(mol,basis,otr_opts)
    assert restart.converged and restart.energy == pytest.approx(otr.energy,abs=2e-10)
    assert restart.n_iter <= 2


def test_capability_and_invalid_options():
    mol=water();basis=vq.BasisSet(mol,"sto-3g");o=options("rhf",True)
    if not vq.has_opentrustregion():
        with pytest.raises(RuntimeError,match="backend unavailable"):
            vq.run_rhf(mol,basis,o)
    else:
        o.opentrustregion.initial_trust_radius=float("nan")
        with pytest.raises(ValueError,match="finite and positive"):
            vq.run_rhf(mol,basis,o)
    o.orbital_optimizer="mystery"
    with pytest.raises(ValueError,match="orbital_optimizer"):
        vq.run_rhf(mol,basis,o)


@requires_otr
@pytest.mark.parametrize("mode", ["conflict", "cosx", "rsh", "mgga", "spinlock"])
def test_unsupported_combinations(mode):
    mol=water();basis=vq.BasisSet(mol,"sto-3g")
    method="uks" if mode=="spinlock" else "rks"
    o=options(method,True)
    if mode=="conflict":o.trah_threshold=1
    if mode=="cosx":o.cosx=True
    if mode=="rsh":o.functional="CAM-B3LYP"
    if mode=="mgga":o.functional="TPSS"
    if mode=="spinlock":o.atomic_spins=[0,0,0]
    with pytest.raises((ValueError,RuntimeError),match="OpenTrustRegion"):
        getattr(vq,"run_"+method)(mol,basis,o)


@requires_otr
@pytest.mark.parametrize("method", ["uhf", "uks"])
def test_native_multiguess_is_not_silently_reused(method):
    o = options(method, True)
    o.multi_guess_seeds = [11, 22]
    with pytest.raises(ValueError, match="OpenTrustRegion.*multi_guess_seeds"):
        getattr(vq, "run_" + method)(water(), "sto-3g", o)


@requires_otr
def test_public_job_output_citation_and_unsupported_modes(tmp_path):
    path=tmp_path/"otr"
    result=vq.run_job(water(),basis="sto-3g",method="rhf",orbital_optimizer="opentrustregion",
                      rhf_options=options("rhf"),output=path,name_molecule=False,
                      write_molden_file=False,write_population_file=False)
    assert result.converged
    out=path.with_suffix(".out").read_text()
    assert "Orbital optimizer: OpenTrustRegion" in out
    assert "10.1021/acs.jctc.5c01576" in out
    assert "Macro/micro iteration totals: unavailable" in out
    with pytest.raises(ValueError,match="integer occupations"):
        vq.run_job(water(),basis="sto-3g",orbital_optimizer="opentrustregion",smearing_temperature=0.01)
    with pytest.raises(ValueError,match="molecular RHF/UHF/RKS/UKS"):
        vq.run_job(water(),basis="sto-3g",method="rohf",orbital_optimizer="opentrustregion")
    with pytest.raises(ValueError,match="molecular-only"):
        vq.run_periodic_job(None,None,orbital_optimizer="opentrustregion")


@requires_otr
@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("backend", ["native", "opentrustregion"])
def test_public_citation_surfaces_and_reprint(tmp_path, capsys, method, backend):
    """Pin real run output, full publication metadata, and actual-use routing."""
    import re
    import tomllib
    from vibeqc.output.citations.cli import main as cite_main

    molecule = water()
    if method in ("uhf", "uks"):
        molecule = vq.Molecule(molecule.atoms, charge=1, multiplicity=2)
    stem = tmp_path / f"{method}-{backend}"
    result = vq.run_job(
        molecule, basis="sto-3g", method=method,
        functional="PBE" if method.endswith("ks") else None,
        orbital_optimizer=backend, output=stem, name_molecule=False,
        num_threads=1, write_molden_file=False, write_population_file=False,
        **{method + "_options": options(method)},
    )
    assert result.converged
    key = "greiner_opentrustregion_2026"
    doi = "10.1021/acs.jctc.5c01576"
    title = "A Reusable Library for Second-Order Orbital Optimization Using the Trust Region Method"
    expected = backend == "opentrustregion"
    for suffix in (".out", ".references", ".bibtex"):
        text = stem.with_suffix(suffix).read_text(encoding="utf-8")
        assert text.count(doi) == int(expected), suffix
        if expected:
            assert title in " ".join(text.split()), suffix
            assert "Høyvik, Ida-Marie" in text
            assert "pulay_diis_1980" not in text
            assert "10.1016/0009-2614(80)80396-4" not in text
            assert "10.1002/jcc.540030413" not in text
            if suffix != ".bibtex":
                assert "22(2), 881-895" in " ".join(text.split())
    bibtex = stem.with_suffix(".bibtex").read_text(encoding="utf-8")
    if expected:
        entry = bibtex.split("@article{" + key + ",", 1)[1].split("\n}", 1)[0]
        assert "Greiner, Jonas and Høyvik, Ida-Marie and Lehtola, Susi and Eriksen, Janus J." in entry
        for field, value in (("volume", 22), ("number", 2), ("year", 2026)):
            assert re.search(rf"{field}\s*=\s*{value}\b", entry)
        assert "{881-895}" in entry
    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    keys = [row["key"] for row in manifest["citations"]["entries"]]
    assert keys.count(key) == int(expected)
    capsys.readouterr()
    assert cite_main([str(stem)]) == 0
    assert capsys.readouterr().out.count(doi) == int(expected)
    assert cite_main([str(stem), "--bibtex-only"]) == 0
    assert capsys.readouterr().out.count(doi) == int(expected)
    original_references = stem.with_suffix(".references").read_text()
    assert cite_main([str(stem), "--write"]) == 0
    assert stem.with_suffix(".references").read_text() == original_references
    assert stem.with_suffix(".bibtex").read_text() == bibtex


def test_explicit_selection_blocks_automatic_native_retries():
    from types import SimpleNamespace
    from vibeqc import runner
    from vibeqc.memory import _dft_xc_requires_dense_ao_tables
    mol=water()
    for method,check in (("rhf",runner._rhf_tail_sad_retry_supported),
                         ("rks",runner._rks_tail_trah_retry_supported)):
        o=options(method,True)
        kwargs=dict(resolved_method=method,molecule=mol,basis=SimpleNamespace(nbasis=200),
                    result=SimpleNamespace(n_iter=100,density=np.eye(200)))
        kwargs[method+"_options"]=o
        if method=="rks":kwargs["functional"]="PBE"
        assert not check(**kwargs)
    assert _dft_xc_requires_dense_ao_tables({"scf_options":options("rks",True)})
    radical=vq.Molecule([vq.Atom(1,[0,0,0])],multiplicity=2)
    o=options("uhf",True)
    _,_,changed=runner._auto_open_shell_optimizer_trah("uhf",radical,optimize=True,uhf_options=o,uks_options=None)
    assert not changed and o.trah_threshold == 0


@requires_otr
def test_no_virtual_space_and_molecular_exhaustion():
    mol=vq.Molecule([vq.Atom(2,[0,0,0])])
    basis=vq.BasisSet(mol,"sto-3g")
    r=vq.run_rhf(mol,basis,options("rhf",True))
    assert r.converged and r.opentrustregion.termination == "no_rotation_parameters"
    mol=water();basis=vq.BasisSet(mol,"sto-3g")
    o=options("rhf",True);o.max_iter=1
    r=vq.run_rhf(mol,basis,o)
    assert not r.converged and r.opentrustregion.termination == "iteration_limit"
    s=np.asarray(vq.compute_overlap(basis));c=np.asarray(r.mo_coeffs)
    np.testing.assert_allclose(r.density,2*c[:,:5]@c[:,:5].T,atol=1e-12)
    np.testing.assert_allclose(c.T@s@c,np.eye(c.shape[1]),atol=2e-10)


@requires_otr
def test_stretched_h2_unrestricted_saddle():
    # At four bohr the restricted singlet determinant is unstable against
    # real spin-breaking rotations. OTR must retain Nalpha=Nbeta=1.
    mol=vq.Molecule([vq.Atom(1,[0,0,0]),vq.Atom(1,[0,0,4.0])])
    basis=vq.BasisSet(mol,"sto-3g")
    ro=options("rhf");rhf=vq.run_rhf(mol,basis,ro)
    o=options("uhf",True);o.initial_guess=vq.InitialGuess.READ
    o.read_density_alpha=np.asarray(rhf.density)/2
    o.read_density_beta=np.asarray(rhf.density)/2
    o.opentrustregion.stability="follow"
    r=vq.run_uhf(mol,basis,o)
    assert r.converged, (r.opentrustregion.termination,r.opentrustregion.log)
    assert r.opentrustregion.stable and r.energy < rhf.energy-0.05
    s=np.asarray(vq.compute_overlap(basis))
    for d in (r.density_alpha,r.density_beta):
        assert np.trace(d@s) == pytest.approx(1.,abs=1e-10)
    native=options("uhf");native.initial_guess=vq.InitialGuess.READ
    native.read_density_alpha=r.density_alpha;native.read_density_beta=r.density_beta
    reference=vq.run_uhf(mol,basis,native)
    assert reference.converged and reference.energy == pytest.approx(r.energy,abs=1e-9)


@requires_otr
def test_open_shell_o2_from_hcore():
    # A less forgiving cold start: quintet O2/cc-pVDZ from HCORE. The linear
    # molecule permits equivalent densities related by axial rotations, so
    # compare energy, spin and density invariants here. The benchmark script
    # separately diagnoses that degeneracy by rotating both spin densities.
    mol = vq.Molecule([vq.Atom(8, [0, 0, 0]), vq.Atom(8, [0, 0, 2.28])], multiplicity=5)
    basis = vq.BasisSet(mol, "cc-pvdz")
    native_options = options("uhf")
    native_options.stability_check = True
    native = vq.run_uhf(mol, basis, native_options)
    result = vq.run_uhf(mol, basis, options("uhf", True))
    assert native.converged and result.converged
    assert result.energy == pytest.approx(native.energy, abs=1e-8)
    assert result.s_squared == pytest.approx(native.s_squared, abs=1e-6)
    assert result.opentrustregion.final_residual < 1e-7
    assert result.opentrustregion.stability_converged and result.opentrustregion.stable
    s = np.asarray(vq.compute_overlap(basis))
    for spin, n in (("alpha", 10), ("beta", 6)):
        d = np.asarray(getattr(result, "density_" + spin))
        c = np.asarray(getattr(result, "mo_coeffs_" + spin))
        assert np.trace(d @ s) == pytest.approx(n, abs=1e-9)
        np.testing.assert_allclose(d @ s @ d, d, atol=1e-9)
        np.testing.assert_allclose(c.T @ s @ c, np.eye(c.shape[1]), atol=1e-9)


@requires_otr
def test_direct_incremental_cache_isolation(molecule_data):
    mol,basis,s,ref,_,h,grid=molecule_data
    jk=core.make_direct_jk_builder(basis,1e-13,True,8)
    c=np.asarray(ref.mo_coeffs)
    p=core._MolecularOrbitalObjective(s,h,0.0,jk,c,c,5,5,True,1.0,basis,grid,"")
    m=p.update(np.zeros(10))
    rng=np.random.default_rng(99);v=rng.normal(size=10)
    hv=p.hessian(v)
    for step in (.4,-.3,.01,.4):
        p.trial(step*v)
    np.testing.assert_allclose(p.hessian(v),hv,atol=1e-12)
    assert p.trial(np.zeros(10)) == pytest.approx(m.energy,abs=1e-12)
    o=options("rhf",True);o.scf_mode=vq.SCFMode.DIRECT;o.incremental_fock=True
    result=vq.run_rhf(mol,basis,o)
    assert result.converged
    assert result.energy == pytest.approx(ref.energy,abs=2e-9)


@requires_otr
def test_molecular_callback_failure_keeps_consistent_state(molecule_data):
    mol, basis, s, _, delegate, h, _ = molecule_data

    class FailingJK(core.JKBuilder):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def build_J(self, d):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("deliberate molecular response failure")
            return delegate.build_J(d)

        def build_K(self, d):
            return delegate.build_K(d)

    r = core.run_rhf_scf_with_jk(basis, 10, s, h, 0.0, FailingJK(),
                                options("rhf", True), molecule=mol)
    assert not r.converged and r.opentrustregion.termination == "callback_failure"
    assert "deliberate molecular response failure" in r.opentrustregion.callback_error
    c, d = np.asarray(r.mo_coeffs), np.asarray(r.density)
    np.testing.assert_allclose(d, 2*c[:, :5] @ c[:, :5].T, atol=1e-12)
    np.testing.assert_allclose(c.T @ s @ c, np.eye(c.shape[1]), atol=2e-10)
    f = h + delegate.build_J(d) - 0.5*delegate.build_K(d)
    assert r.energy == pytest.approx(0.5*np.sum(d*(h+f)), abs=1e-11)


@requires_otr
def test_density_fitted_reference_matches_native():
    mol=water();basis=vq.BasisSet(mol,"def2-svp")
    native=options("rhf");native.density_fit=True;native.aux_basis="def2-svp-jk"
    otr=options("rhf",True);otr.density_fit=True;otr.aux_basis=native.aux_basis
    nr=vq.run_rhf(mol,basis,native);tr=vq.run_rhf(mol,basis,otr)
    assert nr.converged and tr.converged, tr.opentrustregion.log
    assert nr.energy == pytest.approx(tr.energy,abs=3e-8)
