"""Complete all-electron RI/exact HF Lagrangian compositions and relaxed forces."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core


_SELECTIONS = [(True, True), (True, False), (False, True)]
_POSITIONS = np.array([[.1, -.2, .3], [.7, .4, -.8], [-.4, 1.1, .6]])


def _fixture(positions=_POSITIONS, pure=True):
    mol = vq.Molecule([vq.Atom(2, p.tolist()) for p in np.asarray(positions)], 0, 1)
    bases = []
    for angular in ([0, 1, 0], [1, 0, 2]):
        shells = []
        for i, l in enumerate(angular):
            s = core.ShellInfo()
            s.atom_index, s.origin, s.pure, s.l = i, positions[i].tolist(), pure, l
            s.exponents, s.coefficients = [.6+.15*i, 1.5+.2*i], [.8, -.12]
            shells.append(s)
        bases.append(vq.BasisSet(mol, shells, "manta-ri-hf-gradient", False))
    return mol, *bases


def _matrices(n):
    rng = np.random.default_rng(403)
    values = [rng.normal(size=(n, n))*.12 for _ in range(3)]
    return [np.ascontiguousarray((v+v.T)/2) for v in values]


def _policy(basis, auxiliary, truncate):
    o = manta.RIOptions(); o.mode_block = 2
    if truncate:
        lam = np.linalg.eigvalsh(vq.compute_2c_eri(auxiliary))
        gaps = np.flatnonzero(np.diff(lam)>1e-5*lam[-1])
        i = min(gaps, key=lambda i: abs(i-(len(lam)/2-1)))
        o.absolute_cutoff = (lam[i]+lam[i+1])/2
    return o


def _energies(mol, basis, auxiliary, matrices, options, fit_j, fit_k):
    # Independent small libint arrays. This oracle uses no MANTA kernels.
    a, b, w = matrices; total = a+b
    v, raw = vq.compute_2c_eri(auxiliary), vq.compute_3c_eri(basis, auxiliary)
    lam, u = np.linalg.eigh(v)
    keep = lam>max(options.absolute_cutoff, options.relative_cutoff*max(0., lam[-1]))
    g = (u[:, keep]/lam[keep])@u[:, keep].T
    if not fit_j or not fit_k:
        eri = np.asarray(vq.compute_eri(basis))
    j = np.einsum("pij,pq,qkl,kl->ij", raw, g, raw, total) if fit_j else np.einsum("ijkl,kl->ij", eri, total)
    exchange = 0.
    for d in (a, b):
        k = np.einsum("pik,pq,qjl,kl->ij", raw, g, raw, d) if fit_k else np.einsum("ikjl,kl->ij", eri, d)
        exchange -= .5*np.sum(d*k)
    return np.array([mol.nuclear_repulsion(), np.sum(total*(vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol))),
                     .5*np.sum(total*j), exchange, -np.sum(w*vq.compute_overlap(basis))])


@pytest.mark.parametrize("fit_j,fit_k", _SELECTIONS)
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("truncate", [False, True])
def test_complete_fixed_density_lagrangian_components_against_independent_differences(fit_j, fit_k, pure, truncate):
    mol, basis, auxiliary = _fixture(pure=pure)
    matrices = _matrices(basis.nbasis); o = _policy(basis, auxiliary, truncate)
    p = manta.plan_hf_ri_lagrangian_gradient(mol, basis, auxiliary, o, fit_j, fit_k)
    result = manta.compute_hf_ri_lagrangian_gradient(mol, basis, auxiliary, *matrices, o, fit_j, fit_k)
    components = [result.nuclear, result.one_electron, result.coulomb, result.exchange, result.pulay]
    np.testing.assert_allclose(result.two_electron, result.coulomb+result.exchange, atol=0., rtol=0.)
    np.testing.assert_allclose(result.values, sum(components), atol=2e-14)
    direction = np.array([[.2, -.3, .1], [-.1, .4, -.2], [.3, .1, -.4]])
    h = 1e-4
    samples = [_energies(*_fixture(_POSITIONS+k*h*direction, pure), matrices, o, fit_j, fit_k) for k in (-2, -1, 1, 2)]
    expected = (samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*h)
    np.testing.assert_allclose([np.sum(x*direction) for x in components], expected, atol=8e-9, rtol=3e-7)
    for component in components:
        np.testing.assert_allclose(component.sum(axis=0), 0., atol=2e-12)
    assert (result.coulomb_ri is not None) == (result.coulomb_metric is not None) == fit_j
    assert (result.exchange_ri is not None) == (result.exchange_metric is not None) == fit_k
    if fit_j:
        assert result.coulomb_ri.coulomb_energy == pytest.approx(_energies(mol, basis, auxiliary, matrices, o, fit_j, fit_k)[2], abs=1e-12)
    if fit_k:
        assert result.exchange_ri.exchange_energy == pytest.approx(_energies(mol, basis, auxiliary, matrices, o, fit_j, fit_k)[3], abs=1e-12)
    assert result.receipt.shell_calls == p.shell_calls
    assert result.receipt.primitive_integrals_evaluated <= p.primitive_evaluations_upper_bound
    assert result.receipt.modeled_peak_bytes == p.modeled_peak_bytes
    assert result.receipt.algorithm == "composed-ri-hf-lagrangian"
    assert len(result.receipt.source_identity) == len(result.receipt.density_identity) == 64


def _physical(unrestricted, positions=None, fit_j=True, fit_k=True, *, metric=None, budget=None, custom=False):
    if positions is None:
        positions = [[0., 0., 0.], [.2, -.1, 1.8]]+([[1.3, .4, .7]] if unrestricted else [])
    mol = vq.Molecule([vq.Atom(1, np.asarray(p).tolist()) for p in positions], 0, 2 if unrestricted else 1)
    basis = vq.BasisSet(mol, "sto-3g"); auxiliary = vq.BasisSet(mol, "sto-3g")
    if custom:
        shells = auxiliary.shells()
        for shell in shells:
            shell.exponents = [1.37*x for x in shell.exponents]
        # Same display name, different content: a name-based reconstruction
        # would silently differentiate another fitting source.
        auxiliary = vq.BasisSet(mol, shells, "sto-3g", False)
    if metric is None:
        metric = manta.RIOptions()
    if budget is None:
        budget = manta.ResourceBudget()
    options = vq.UHFOptions() if unrestricted else vq.RHFOptions()
    options.initial_guess = vq.InitialGuess.HCORE
    options.conv_tol_energy, options.conv_tol_grad, options.max_iter = 1e-13, 1e-11, 150
    options.stability_check = False
    driver = vq.run_uhf if unrestricted else vq.run_rhf
    result = driver(mol, basis, options, integral_engine=vq.IntegralEngineOptions(backend="manta", budget=budget),
                    coulomb=vq.CoulombBuildOptions("ri" if fit_j else "exact", auxiliary, metric),
                    exchange=vq.ExchangeBuildOptions("ri" if fit_k else "exact"))
    assert result.converged and result.engine_provenance.stationarity_confirmed
    if unrestricted:
        a, b = result.density_alpha, result.density_beta
        ca, cb = result.mo_coeffs_alpha[:, :2], result.mo_coeffs_beta[:, :1]
        w = (ca*result.mo_energies_alpha[:2])@ca.T+(cb*result.mo_energies_beta[:1])@cb.T
    else:
        a = b = .5*result.density
        c = result.mo_coeffs[:, :1]; w = 2*(c*result.mo_energies[:1])@c.T
    return mol, basis, auxiliary, result, [np.ascontiguousarray(m) for m in (a, b, w)]


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("fit_j,fit_k", _SELECTIONS)
def test_stationary_hf_complete_gradient_against_relaxed_energies(unrestricted, fit_j, fit_k):
    mol, basis, auxiliary, result, matrices = _physical(unrestricted, fit_j=fit_j, fit_k=fit_k)
    gradient = manta.compute_hf_ri_lagrangian_gradient(mol, basis, auxiliary, *matrices, fit_j=fit_j, fit_k=fit_k)
    selected_gradient = (vq.compute_gradient_uhf if unrestricted else vq.compute_gradient)(mol, basis, result)
    native_gradient = (core.compute_gradient_uhf if unrestricted else core.compute_gradient)(mol, basis, result)
    np.testing.assert_allclose(selected_gradient, gradient.values, atol=3e-15, rtol=3e-15)
    np.testing.assert_array_equal(native_gradient, selected_gradient)
    positions = np.array([a.xyz for a in mol.atoms])
    direction = np.array([[.1, -.2, .3], [-.3, .2, .1]]+([[.2, -.1, -.2]] if unrestricted else []))
    for h in (2e-4, 1e-4):
        energies = [_physical(unrestricted, positions+k*h*direction, fit_j, fit_k)[3].energy for k in (-1, 1)]
        assert np.sum(gradient.values*direction) == pytest.approx((energies[1]-energies[0])/(2*h), abs=4e-8)
    np.testing.assert_allclose(gradient.values.sum(axis=0), 0., atol=3e-12)
    np.testing.assert_allclose(np.cross(positions, gradient.values).sum(axis=0), 0., atol=1e-10)
    assert np.linalg.norm(gradient.pulay) > 1e-3
    if fit_j:
        assert gradient.coulomb_ri.ri_source_identity == result.engine_provenance.coulomb_source_identity
    if fit_k:
        assert gradient.exchange_ri.ri_source_identity == result.engine_provenance.exchange_source_identity


@pytest.mark.parametrize("unrestricted", [False, True])
def test_full_ri_stationary_forces_against_independent_pyscf(unrestricted):
    gto = pytest.importorskip("pyscf.gto"); scf = pytest.importorskip("pyscf.scf")
    mol, basis, auxiliary, result, matrices = _physical(unrestricted)
    reference_mol = gto.M(atom=[("H", a.xyz) for a in mol.atoms], basis="sto-3g", unit="Bohr", spin=int(unrestricted), verbose=0)
    mf = (scf.UHF(reference_mol) if unrestricted else scf.RHF(reference_mol)).density_fit(auxbasis="sto-3g")
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-11
    mf.kernel(dm0=np.array(matrices[:2]) if unrestricted else matrices[0]+matrices[1])
    assert mf.converged
    assert result.energy == pytest.approx(mf.e_tot, abs=3e-10)
    native = manta.compute_hf_ri_lagrangian_gradient(mol, basis, auxiliary, *matrices)
    reference_gradient = mf.nuc_grad_method()
    reference_gradient.auxbasis_response = True
    np.testing.assert_allclose(native.values, reference_gradient.kernel(), atol=3e-8, rtol=3e-8)


def test_all_ri_has_no_exact_four_center_derivative_pass_and_source_is_component_specific():
    mol, basis, auxiliary = _fixture(); matrices = _matrices(basis.nbasis)
    results = [manta.compute_hf_ri_lagrangian_gradient(mol, basis, auxiliary, *matrices, fit_j=j, fit_k=k) for j, k in _SELECTIONS]
    all_ri, mixed_j, mixed_k = results
    one_calls = (18+3*len(mol.atoms))*basis.nshells**2
    def calls(r):
        return r.metric_shell_calls+r.coupling_shell_calls+r.metric_derivative_shell_calls+r.coupling_derivative_shell_calls
    assert all_ri.receipt.shell_calls == one_calls+calls(all_ri.coulomb_ri)+calls(all_ri.exchange_ri)
    assert mixed_j.receipt.shell_calls == one_calls+12*basis.nshells**4+calls(mixed_j.coulomb_ri)
    assert mixed_k.receipt.shell_calls == one_calls+12*basis.nshells**4+calls(mixed_k.exchange_ri)
    assert len({r.receipt.source_identity for r in results}) == 3
    assert len({r.receipt.density_identity for r in results}) == 1


@pytest.mark.parametrize("fit_j,fit_k", _SELECTIONS)
@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"),
    ("primitive_quartets", "primitive_evaluations_upper_bound"), ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_composed_resource_boundaries_before_input_copy(fit_j, fit_k, field, plan_field):
    mol, basis, auxiliary = _fixture(); matrices = _matrices(basis.nbasis)
    b = manta.ResourceBudget(); b.other_live_bytes = 19283
    p = manta.plan_hf_ri_lagrangian_gradient(mol, basis, auxiliary, fit_j=fit_j, fit_k=fit_k, budget=b)
    setattr(b, field, getattr(p, plan_field))
    r = manta.compute_hf_ri_lagrangian_gradient(mol, basis, auxiliary, *matrices, fit_j=fit_j, fit_k=fit_k, budget=b)
    assert r.receipt.modeled_peak_bytes == p.modeled_peak_bytes
    setattr(b, field, getattr(p, plan_field)-1)
    matrices[0][:] = np.nan
    with pytest.raises(ValueError, match="budget"):
        manta.compute_hf_ri_lagrangian_gradient(mol, basis, auxiliary, *matrices, fit_j=fit_j, fit_k=fit_k, budget=b)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("kind", ["float32", "complex", "strided", "shape", "nan", "asymmetric"])
def test_composition_input_contracts(index, kind):
    mol, basis, auxiliary = _fixture(); n = basis.nbasis; matrices = _matrices(n)
    matrices[index] = {"float32": np.eye(n, dtype=np.float32), "complex": np.eye(n, dtype=complex),
                       "strided": np.eye(2*n)[::2, ::2], "shape": np.eye(n+1), "nan": np.eye(n)*np.nan,
                       "asymmetric": np.triu(np.ones((n, n)))}[kind]
    with pytest.raises(ValueError):
        manta.compute_hf_ri_lagrangian_gradient(mol, basis, auxiliary, *matrices)


def test_composition_refuses_unselected_ri_crossing_and_bad_geometry():
    mol, basis, auxiliary = _fixture(); matrices = _matrices(basis.nbasis)
    with pytest.raises(ValueError, match="at least one fitted"):
        manta.compute_hf_ri_lagrangian_gradient(mol, basis, auxiliary, *matrices, fit_j=False, fit_k=False)
    o = manta.RIOptions(); o.absolute_cutoff = np.linalg.eigvalsh(vq.compute_2c_eri(auxiliary))[4]
    with pytest.raises(RuntimeError, match="rank cutoff"):
        manta.compute_hf_ri_lagrangian_gradient(mol, basis, auxiliary, *matrices, o)
    moved, _, _ = _fixture(_POSITIONS+.1)
    with pytest.raises(ValueError, match="centers/atom map"):
        manta.compute_hf_ri_lagrangian_gradient(moved, basis, auxiliary, *matrices)


def test_complete_ri_hf_gradient_citation_route():
    from vibeqc.output.citations.registry import load_default_database
    refs = load_default_database().assemble(method="RHF", extra_methods=["manta_ri_hf_gradient"])
    assert {"higham_matrix_functions_2008", "neese_shark_2023", "mcmurchie_davidson_1978", "pople_derivatives_1979"} <= {c.key for c in refs.citations}


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("fit_j,fit_k", _SELECTIONS)
def test_selected_gradient_owns_custom_auxiliary_and_mutable_metric_options(unrestricted, fit_j, fit_k):
    _, _, auxiliary, _, _ = _physical(unrestricted, custom=True)
    spectrum = np.linalg.eigvalsh(vq.compute_2c_eri(auxiliary))
    policy = manta.RIOptions(); policy.mode_block = 1
    policy.absolute_cutoff = (spectrum[0]+spectrum[1])/2
    mol, basis, auxiliary, result, matrices = _physical(
        unrestricted, fit_j=fit_j, fit_k=fit_k, metric=policy, custom=True)
    expected = manta.compute_hf_ri_lagrangian_gradient(mol, basis, auxiliary, *matrices, policy, fit_j, fit_k)
    policy.absolute_cutoff = 1e20; policy.mode_block = 0
    # Shell exports are copies. Mutating caller data must not alter the source.
    shells = auxiliary.shells(); shells[0].exponents = [5.]*len(shells[0].exponents)
    actual = (vq.compute_gradient_uhf if unrestricted else vq.compute_gradient)(mol, basis, result)
    np.testing.assert_allclose(actual, expected.values, atol=3e-15, rtol=3e-15)
    info = expected.coulomb_metric if fit_j else expected.exchange_metric
    assert 0 < info.rank < auxiliary.nbasis


@pytest.mark.parametrize("unrestricted", [False, True])
def test_selected_gradient_retains_derivative_budget_before_caller_mutation(unrestricted):
    mol, basis, auxiliary, _, _ = _physical(unrestricted)
    plan = manta.plan_hf_ri_lagrangian_gradient(mol, basis, auxiliary)
    budget = manta.ResourceBudget(); budget.shell_calls = plan.shell_calls-1
    mol, basis, _, result, _ = _physical(unrestricted, budget=budget)
    budget.shell_calls = 10000000
    with pytest.raises(ValueError, match="budget"):
        (vq.compute_gradient_uhf if unrestricted else vq.compute_gradient)(mol, basis, result)


def test_selected_gradient_refuses_unresolved_metric_rank_boundary():
    _, _, auxiliary, _, _ = _physical(False)
    policy = manta.RIOptions()
    policy.absolute_cutoff = np.linalg.eigvalsh(vq.compute_2c_eri(auxiliary))[0]
    mol, basis, _, result, _ = _physical(False, metric=policy)
    with pytest.raises(RuntimeError, match="rank cutoff"):
        vq.compute_gradient(mol, basis, result)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_selected_gradient_emits_verified_source_only_after_success(unrestricted, monkeypatch):
    from types import SimpleNamespace
    from vibeqc import output
    mol, basis, _, result, _ = _physical(unrestricted)
    events = []
    monkeypatch.setattr(output, "record", lambda *args, **kwargs: events.append((args, kwargs)))
    gradient = vq.compute_gradient_uhf if unrestricted else vq.compute_gradient
    # Wrappers must report the native executed receipt, not an outer claim.
    wrapper = SimpleNamespace(_vibeqc_native_result=result, engine_provenance=object(),
                              ecp_provenance_verified=True, ecp_operator_applied=False, ecp_total_ncore=0)
    gradient(mol, basis, wrapper)
    args, data = events.pop()
    assert args[0] == "molecular_hf_gradient"
    assert "HF gradient: manta; J=ri; K=ri" in args[1]
    provenance = data["gradient_provenance"]
    assert provenance["citation_methods"] == ["manta_ri_hf_gradient"]
    for name in ("molecule_identity", "coulomb_source_identity", "exchange_source_identity",
                 "terminal_density_alpha_identity", "terminal_density_beta_identity"):
        assert provenance[name] == getattr(result.engine_provenance, name)
    wrong = vq.BasisSet(mol, "6-31g")
    with pytest.raises(ValueError, match="differs from the SCF source"):
        gradient(mol, wrong, result)
    options = vq.GradientOptions(); options.cosx = True
    with pytest.raises(ValueError, match="conflicts with derivative options"):
        gradient(mol, basis, result, options)
    assert not events
