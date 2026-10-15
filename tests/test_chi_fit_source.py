"""T3 contracts for selected chi fitted sources; no construction comparison."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest
import vibeqc as vq

from vibeqc.periodic import chi
from vibeqc.periodic.chi import scf, posthf
from vibeqc.periodic_fit_source import resolve_fit_source

pytestmark = pytest.mark.experimental


def cell():
    system = vq.PeriodicSystem(3, np.eye(3) * 6,
        [vq.Atom(1, [3.,3.,2.3]), vq.Atom(1, [3.,3.,3.7])])
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


@pytest.mark.parametrize("source", [None, "", "fast", "SR", True, 3])
def test_invalid_source_fails_before_shared_scf(source):
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf, run_kuhf_periodic_gdf
    system = SimpleNamespace(dim=3)
    for driver in (run_krhf_periodic_gdf, run_kuhf_periodic_gdf):
        with pytest.raises(ValueError, match="fit_source"):
            driver(system, None, fit_source=source)


@pytest.mark.parametrize("source", ["classic", "sr"])
@pytest.mark.parametrize("method,dim", [("mdf",3), ("compcell",3), ("rsgdf",2), ("rsgdf",1)])
def test_source_backend_admission(source, method, dim):
    with pytest.raises(NotImplementedError, match="3D"):
        resolve_fit_source(source, gdf_method=method, dimension=dim)


def test_historical_default_is_preserved():
    assert resolve_fit_source("auto", gdf_method="rsgdf", dimension=3) == "sr"
    assert resolve_fit_source("auto", gdf_method="rsgdf", dimension=1) == "classic"
    assert resolve_fit_source("auto", gdf_method="mdf", dimension=3) == "not-applicable"
    assert posthf._FiniteTorusConventionMixin().quantitative_status == "not-qualified"


class Dispatched(Exception):
    pass


@pytest.mark.parametrize("method,symbol", [("rhf","run_krhf_periodic_gdf"),
    ("rks","run_krks_periodic_gdf"), ("uhf","run_kuhf_periodic_gdf"),
    ("uks","run_kuks_periodic_gdf")])
@pytest.mark.parametrize("source", ["auto", "classic", "sr"])
def test_chi_reference_transport(monkeypatch, method, symbol, source):
    system, basis = cell()
    def backend(*args, **kwargs):
        assert kwargs["fit_source"] == source
        assert kwargs["gdf_linear_dep_threshold"] == 2e-9
        assert kwargs["rsgdf_tail_ke_cutoff"] == 300.
        raise Dispatched
    monkeypatch.setattr(scf, symbol, backend)
    args = (system,basis,"pbe",(2,1,1)) if method.endswith("ks") else (system,basis,(2,1,1))
    with pytest.raises(Dispatched):
        getattr(scf, "run_aiccm2026dev_b_"+method)(*args, backend="ri", fit_source=source,
            gdf_linear_dep_threshold=2e-9, rsgdf_tail_ke_cutoff=300., progress=False)


@pytest.mark.parametrize("source", ["auto", "classic", "sr"])
def test_gamma_admission_uses_selected_engine(monkeypatch, source):
    system,basis=cell()
    monkeypatch.setattr(scf, "_auto_rsgdf_tail_ke_cutoff", lambda *a,**k: pytest.fail("inactive Gamma policy executed"))
    def backend(*a,**kw):
        assert kw["rsgdf_tail_ke_cutoff"] is None
        raise Dispatched
    monkeypatch.setattr(scf,"run_krhf_periodic_gdf",backend)
    with pytest.raises(Dispatched):
        scf.run_aiccm2026dev_b_rhf(system,basis,(1,1,1),backend="ri",fit_source=source,progress=False)


@pytest.mark.parametrize("name", [n for n in posthf.__all__ if n.startswith("run_")])
@pytest.mark.parametrize("source,exc", [(None,ValueError), ("bad",ValueError), ("sr",NotImplementedError)])
def test_posthf_refuses_unsupported_source_before_scf(name, source, exc, monkeypatch):
    system,basis=cell()
    monkeypatch.setattr(posthf,"run_aiccm2026dev_b_rhf",lambda *a,**k:pytest.fail("SCF entered"))
    monkeypatch.setattr(posthf,"run_aiccm2026dev_b_uhf",lambda *a,**k:pytest.fail("SCF entered"))
    with pytest.raises(exc,match="source|SR"):
        getattr(posthf,name)(system,basis,(2,1,1),fit_source=source,progress=False)


def test_missing_retained_source_is_refused():
    system,basis=cell()
    with pytest.raises(ValueError,match="executed source"):
        posthf._selected_lpq_cache(system,basis,np.zeros((1,3)),"aux",reference=SimpleNamespace(),
            fit_source="classic",rsgdf_tail_ke_cutoff=None,lattice_cutoff_bohr=15.,
            rsgdf_ke_cutoff=200.,gdf_linear_dep_threshold=1e-9)


def test_toml_mesh_arrays_and_nested_execution():
    from vibeqc.output.manifest import _toml_value
    value={"mesh":(2,2,1), "matrix":((1.,0.),(0.,1.)), "support": {"cutoff":200.}}
    parsed=tomllib.loads("value = "+_toml_value(value))["value"]
    assert parsed == {"mesh":[2,2,1],"matrix":[[1.,0.],[0.,1.]],"support":{"cutoff":200.}}


def test_native_provenance_matches_loaded_core():
    import hashlib
    from pathlib import Path
    from vibeqc import _vibeqc_core
    from vibeqc.banner import loaded_process_provenance
    from vibeqc.output.formats.system_info import system_info
    p=loaded_process_provenance()
    assert p["native_core_sha256"] == hashlib.sha256(Path(_vibeqc_core.__file__).read_bytes()).hexdigest()
    assert system_info()["vibeqc"]["native_core_sha256"] == p["native_core_sha256"]
    assert len(p["source_revision"]) == 40


@pytest.mark.parametrize("source", [None, "invalid", "classic", "sr"])
def test_frontdoor_refuses_incompatible_source(tmp_path, source):
    system,basis=cell()
    with pytest.raises((ValueError,NotImplementedError),match="fit_source"):
        vq.run_periodic_job(system,basis,method="aiccm",variant="chi",
            aiccm_backend="four_center",fit_source=source,output=tmp_path/'refused',progress=False)


def test_frontdoor_transports_classic_and_orbital_tolerance(tmp_path, monkeypatch):
    import vibeqc.periodic_runner as runner
    system,basis=cell()
    def capture(*args,**kwargs):
        assert kwargs["fit_source"] == "classic"
        assert args[3].conv_tol_energy == 1e-11
        assert args[3].conv_tol_grad == 1e-9
        raise Dispatched
    monkeypatch.setattr(runner,"run_aiccm2026dev_b_rhf",capture)
    with pytest.raises(Dispatched):
        vq.run_periodic_job(system,basis,method="aiccm",variant="chi",
            aiccm_backend="ri",fit_source="classic",conv_tol_energy=1e-11,
            conv_tol_grad=1e-9,convergence="off",aiccm_lattice_extension=(2,1,1),
            output=tmp_path/'classic',progress=False,verbose=0)


def test_source_descriptor_and_retained_factors_are_bound():
    from vibeqc.aux_basis import default_aux_for, make_aux_basis_set
    from vibeqc.periodic_fit_source import RSGDFFitRequest, attach_fit_source
    system,basis=cell()
    name=default_aux_for(basis.name)
    auxiliary=make_aux_basis_set(system.unit_cell_molecule(),aux_name=name)
    opts=vq.LatticeSumOptions()
    cache={(0,0):np.zeros((auxiliary.nbasis,basis.nbasis,basis.nbasis),dtype=complex)}
    result=SimpleNamespace()
    attach_fit_source(result,system=system,basis=basis,auxiliary=auxiliary,auxiliary_name=name,
        cache=cache,request=RSGDFFitRequest("classic",200.,None),selected="classic",
        lat_opts=opts,oneel_lat_opts=opts,metric_threshold=1e-9,fit_screen_threshold=0.,
        kpoints_cart=np.zeros((1,3)),executed_tail=None,retain_canonical=True)
    with pytest.raises(FrozenInstanceError):result.executed_fit_source.engine="different"
    with pytest.raises(ValueError):result._rsgdf_canonical_factors[(0,0)][0,0,0]=1.
    kwargs=dict(reference=result,fit_source="classic",rsgdf_tail_ke_cutoff=None,
        lattice_cutoff_bohr=opts.cutoff_bohr,rsgdf_ke_cutoff=200.,gdf_linear_dep_threshold=1e-9)
    borrowed,_=posthf._selected_lpq_cache(system,basis,np.zeros((1,3)),name,**kwargs)
    assert borrowed[(0,0)] is cache[(0,0)]
    result._rsgdf_canonical_factors={(0,0):cache[(0,0)]+1.}
    with pytest.raises(ValueError,match="does not match"):
        posthf._selected_lpq_cache(system,basis,np.zeros((1,3)),name,**kwargs)
    result._rsgdf_canonical_factors=cache
    result.executed_fit_source=replace(result.executed_fit_source,engine="unidentified")
    with pytest.raises(ValueError,match="does not match"):
        posthf._selected_lpq_cache(system,basis,np.zeros((1,3)),name,**kwargs)


@pytest.mark.parametrize("projected", [False, True])
def test_auto_posthf_keeps_independent_executed_source(monkeypatch, projected):
    from vibeqc.aux_basis import default_aux_for, make_aux_basis_set
    from vibeqc.periodic_fit_source import RSGDFFitRequest, factor_digest
    system, basis = cell()
    name = default_aux_for(basis.name)
    aux = make_aux_basis_set(system.unit_cell_molecule(), aux_name=name)
    kcart = np.zeros((1, 3))
    shape = (aux.nbasis, 1, 1) if projected else (aux.nbasis, basis.nbasis, basis.nbasis)
    factors = {(0, 0): np.zeros(shape, dtype=complex)}
    prior = object()
    reference = SimpleNamespace(executed_fit_source=prior,
        rsgdf_fit_request=RSGDFFitRequest("auto", 200., None))
    controls = dict(reference=reference, fit_source="auto", rsgdf_tail_ke_cutoff=None,
        lattice_cutoff_bohr=15., rsgdf_ke_cutoff=200., gdf_linear_dep_threshold=1e-9)
    if projected:
        monkeypatch.setattr(posthf, "_build_canonical_lov_cache", lambda *a, **k: (factors, aux))
        borrowed, _ = posthf._selected_lov_cache(system, basis, kcart, [np.eye(2)], 1, name, **controls)
    else:
        monkeypatch.setattr(posthf, "_build_canonical_lpq_cache", lambda *a, **k: (factors, aux))
        borrowed, _ = posthf._selected_lpq_cache(system, basis, kcart, name, **controls)
    assert borrowed is factors
    result = posthf._FiniteTorusConventionMixin()
    result.hf_result = reference
    source = result.executed_fit_source
    assert source is not prior and result.reference_fit_source is prior
    assert source.engine == "native-classic-fft-shared-q"
    assert source.source == "classic" and source.auxiliary_frame == "canonical-modrho"
    assert source.factor_space == ("occupied-virtual" if projected else "ao-pair")
    assert source.factor_sha256 == factor_digest(factors)
    assert "one_electron_lattice" not in dict(source.numerical_support)
    assert result.quantitative_status == "not-qualified"


@pytest.mark.slow
def test_classic_source_matches_independent_operator_and_real_mp2():
    from vibeqc.aux_basis import default_aux_for, _build_coulomb_from_diagonal_factors
    from vibeqc.periodic_k_gdf import _build_k_from_lpq_cache, _madelung_for_kmesh
    from vibeqc.madelung import apply_exxdiv_ewald_to_K
    system, basis = cell()
    mesh = (2, 1, 1)
    opts = vq.PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-9
    opts.damping = 0.
    opts.dynamic_damping = False
    hf = scf.run_aiccm2026dev_b_rhf(system, basis, mesh, opts,
        backend="ri", fit_source="classic", progress=False, verbose=0)
    assert hf.converged and hf.quantitative_status == "not-qualified"
    kp = scf.cyclic_gamma_mesh(system, mesh)
    kcart = np.asarray(kp.kpoints_cart)
    weights = np.asarray(kp.weights)
    auxname = default_aux_for(basis.name)
    controls = dict(lattice_cutoff_bohr=opts.lattice_opts.cutoff_bohr,
        rsgdf_ke_cutoff=200., gdf_linear_dep_threshold=1e-9)
    oracle, _ = posthf._build_canonical_lpq_cache_pairwise(system, basis, kcart, auxname, **controls)
    selected, _ = posthf._selected_lpq_cache(system, basis, kcart, auxname,
        reference=hf, fit_source="classic", rsgdf_tail_ke_cutoff=None, **controls)
    assert all(selected[key] is hf._rsgdf_canonical_factors[key] for key in selected)
    C = np.asarray(hf.mo_coeffs)
    S = np.asarray(hf.overlap)
    H = np.asarray(hf.hcore)
    eps = np.asarray(hf.mo_energies)
    D = np.asarray([2 * c[:, :1] @ c[:, :1].conj().T for c in C])
    def two_electron(cache, density):
        J = np.asarray(_build_coulomb_from_diagonal_factors(
            [cache[(i, i)] for i in range(2)], density, weights))
        K = np.asarray(_build_k_from_lpq_cache(cache, density, weights, nbasis=basis.nbasis))
        return J - .5 * K
    rng = np.random.default_rng(20260917)
    probe = rng.normal(size=D.shape) + 1j * rng.normal(size=D.shape)
    probe = .5 * (probe + probe.conj().transpose(0, 2, 1))
    np.testing.assert_allclose(two_electron(selected, probe), two_electron(oracle, probe), atol=1e-10, rtol=0)
    J = np.asarray(_build_coulomb_from_diagonal_factors([oracle[(i, i)] for i in range(2)], D, weights))
    K = np.asarray(apply_exxdiv_ewald_to_K(
        _build_k_from_lpq_cache(oracle, D, weights, nbasis=basis.nbasis), S, D,
        _madelung_for_kmesh(system, mesh)))
    F = H + J - .5 * K
    np.testing.assert_allclose(F, hf.fock, atol=1e-10, rtol=0)
    assert np.max(np.abs(F @ D @ S - S @ D @ F)) <= 1e-9
    assert np.max(np.abs(F @ C - (S @ C) * eps[:, None, :])) <= 1e-9
    energy = .5 * sum(w * np.trace(d @ (h + f)).real for w, d, h, f in zip(weights, D, H, F)) + hf.e_nuclear
    assert abs(energy - hf.energy) <= 1e-10
    lov, _ = posthf._selected_lov_cache(system, basis, kcart, C, 1, auxname,
        reference=hf, fit_source="classic", rsgdf_tail_ke_cutoff=None, **controls)
    ss, os, _ = vq._vibeqc_core.aiccm2026dev_b_mp2_energy_from_lov(
        lov, np.ascontiguousarray(eps), mesh, (0, 0, 0), 1e-12)
    real = posthf._build_b_real_reference(system, basis, mesh, hf, dict(selected), auxname)
    assert abs(posthf._complete_space_mp2_audit(real) / 2 - ss - os) <= 1e-10


def test_chi_keeps_its_classic_scf_on_the_historical_support():
    """#466: the GDF drivers default the classic fit to physical pairs; chi
    pins the support its classic results were qualified with."""
    system = vq.PeriodicSystem(3, np.eye(3) * 6.0, [vq.Atom(1, [0.0, 0.0, 0.0]),
                                                     vq.Atom(1, [1.4, 0.0, 0.0])])
    assert scf._chi_classic_support_kwargs(system, "classic", "rsgdf") == dict(
        classic_support_policy="historical_index_ball")
    assert scf._chi_classic_support_kwargs(system, "auto", "rsgdf") == {}
    assert scf._chi_classic_support_kwargs(system, "sr", "rsgdf") == {}


def _translation_attested_gamma_model(n_occ=1):
    """A stationary supplied Hamiltonian with an authenticated factor cache."""
    from scipy.linalg import eigh
    from vibeqc.aux_basis import default_aux_for, make_aux_basis_set
    from vibeqc.periodic_fit_source import RSGDFFitRequest, attach_fit_source
    from vibeqc.banner import loaded_process_provenance

    system, basis = cell()
    if n_occ == 2:
        system = vq.PeriodicSystem(3, np.eye(3)*6,
            [vq.Atom(1, [2., 2., 2.3]), vq.Atom(1, [2., 2., 3.7]),
             vq.Atom(1, [4., 4., 2.3]), vq.Atom(1, [4., 4., 3.7])])
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = (1, 1, 1)
    kpoints = scf.cyclic_gamma_mesh(system, mesh)
    nbf = basis.nbasis
    S = np.eye(nbf)+.05*np.ones((nbf, nbf))
    F = np.diag(np.arange(nbf)-n_occ+.4)+.03*np.ones((nbf, nbf))
    eps, C = eigh(F, S)
    convention = scf._finite_torus_convention(
        system, mesh, full_range_exchange_coefficient=1.)
    exchange = scf.AICCM2026DevBExactExchangeAssembly(1., 0., 0.)
    diagnostics = scf.AICCM2026DevBDiagnostics(
        mesh=mesh, n_cyclic_cells=1, n_kpoints=1, wigner_seitz_partition_error=0.,
        density_idempotency_error=0., electron_count_error=0., inverse_bloch_imaginary_residual=0.,
        backend="ri", electronic_method="RHF", wall_time_seconds=0., exact_exchange_assembly=exchange,
        final_scf_delta_e_ha=0., finite_torus_convention=convention)
    hf = SimpleNamespace(
        converged=True, energy=-1., reference_kind="rhf", smearing_temperature=0.,
        overlap=[S], fock=[F], mo_coeffs=[C], mo_energies=[eps],
        density=(2*C[:, :n_occ] @ C[:, :n_occ].T)[None],
        occupations=np.array([[2.]*n_occ+[0.]*(nbf-n_occ)]),
        kpoints_frac=np.asarray(kpoints.kpoints_frac),
        kpoints_cart=np.asarray(kpoints.kpoints_cart),
        kpoint_weights=np.asarray(kpoints.weights),
        process_provenance=loaded_process_provenance(),
        exact_exchange_assembly=exchange, aiccm2026dev_b=diagnostics,
    )
    name = default_aux_for(basis.name)
    auxiliary = make_aux_basis_set(system.unit_cell_molecule(), aux_name=name)
    values = np.arange(auxiliary.nbasis*nbf**2).reshape(auxiliary.nbasis, nbf, nbf)*1e-4
    cache = {(0, 0): np.asarray(values+values.transpose(0, 2, 1), dtype=complex)}
    options = vq.LatticeSumOptions()
    attach_fit_source(hf, system=system, basis=basis, auxiliary=auxiliary,
        auxiliary_name=name, cache=cache, request=RSGDFFitRequest("classic", 200., None),
        selected="classic", lat_opts=options, oneel_lat_opts=options,
        metric_threshold=1e-9, fit_screen_threshold=0., kpoints_cart=hf.kpoints_cart,
        executed_tail=None, retain_canonical=True)
    return system, basis, hf, cache, auxiliary


@pytest.mark.parametrize("fit_source,tampered", [("classic", False), ("auto", False),
                                                 ("classic", True)])
def test_translation_source_uses_authenticated_selected_factors(monkeypatch, fit_source, tampered):
    from vibeqc.periodic.chi.translation import prepare_chi_translation_source
    from vibeqc.periodic.chi.reference import _digest
    from vibeqc.dlpno.translation_factors import TranslationHomeFactors

    system, basis, hf, cache, auxiliary = _translation_attested_gamma_model()
    executed = hf.executed_fit_source
    original = cache[(0, 0)].copy()
    options = vq.PeriodicRHFOptions()
    options.conv_tol_energy, options.conv_tol_grad = 1e-11, 1e-9
    options_before = _digest(options)
    monkeypatch.setattr(posthf, "run_aiccm2026dev_b_rhf", lambda *a, **k: hf)
    refits = []

    def refit(*args, **kwargs):
        refits.append(True)
        # Equal numerical factors do not turn a separate fit into reuse.
        return {(0, 0): original.copy()}, auxiliary

    monkeypatch.setattr(posthf, "_build_canonical_lpq_cache", refit)
    if tampered:
        hf._rsgdf_canonical_factors = {(0, 0): original+.01}
        with pytest.raises(ValueError, match="does not match"):
            prepare_chi_translation_source(system, basis, options=options,
                fit_source=fit_source, progress=False, verbose=0)
    else:
        source = prepare_chi_translation_source(system, basis, options=options,
            fit_source=fit_source, progress=False, verbose=0)
        assert type(source.factors) is TranslationHomeFactors
        np.testing.assert_allclose(source.factors.home_factors, original.real, atol=1e-15)
        assert source.conventions.same_source is (fit_source == "classic")
        adapter = source.provenance["adapter"]
        assert adapter["scf_fit"] == executed.identity
        assert adapter["correlation_fit_status"] == ("reused" if fit_source == "classic" else "built")
        correlation = executed if fit_source == "classic" else hf._chi_correlation_fit_source
        assert adapter["correlation_fit"] == correlation.identity
        assert correlation.factor_sha256 == executed.factor_sha256
    assert refits == ([True] if fit_source == "auto" else [])
    assert hf.executed_fit_source is executed
    assert _digest(options) == options_before
    np.testing.assert_array_equal(cache[(0, 0)], original)
    assert not cache[(0, 0)].flags.writeable


def test_translation_source_rejects_mesh_before_correlation_refit(monkeypatch):
    from vibeqc.periodic.chi.translation import prepare_chi_translation_source

    system, basis, hf, _, _ = _translation_attested_gamma_model()
    hf.kpoints_frac = np.array([[.25, 0., 0.]])
    monkeypatch.setattr(posthf, "run_aiccm2026dev_b_rhf", lambda *a, **k: hf)

    def forbidden(*args, **kwargs):
        raise AssertionError("inconsistent Bloch state reached the correlation fit")

    monkeypatch.setattr(posthf, "_selected_lpq_cache", forbidden)
    with pytest.raises(ValueError, match="character|mesh|cyclic|Gamma"):
        prepare_chi_translation_source(system, basis, fit_source="auto", progress=False)


@pytest.mark.slow
def test_translation_source_public_h2_matches_character_and_dense_mp2(monkeypatch, record_property):
    """One fresh SCF, three representations of that exact finite Hamiltonian."""
    from vibeqc.aux_basis import default_aux_for
    from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2
    from vibeqc.dlpno.translation_operators import CyclicAOOperator
    from vibeqc.periodic.chi.reference import _digest
    from vibeqc.periodic.chi.translation import prepare_chi_translation_source

    system, basis = cell()
    mesh = (2, 1, 1)
    options = vq.PeriodicRHFOptions()
    options.conv_tol_energy, options.conv_tol_grad = 1e-11, 1e-9
    options.damping, options.dynamic_damping = 0., False
    before = _digest(options)
    captured = []
    original_scf = posthf.run_aiccm2026dev_b_rhf

    def owned_scf(*args, **kwargs):
        result = original_scf(*args, **kwargs)
        captured.append(result)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("block preparation entered the dense finite-reference path")

    with monkeypatch.context() as preparation:
        preparation.setattr(posthf, "run_aiccm2026dev_b_rhf", owned_scf)
        preparation.setattr(posthf, "_build_b_real_reference", forbidden)
        preparation.setattr(posthf, "_periodic_matrix_to_real_supercell", forbidden)
        preparation.setattr(posthf, "_lpq_cache_to_real_supercell", forbidden)
        source = prepare_chi_translation_source(system, basis, mesh, options,
            fit_source="classic", progress=False, verbose=0)
    assert len(captured) == 1
    hf = captured[0]
    assert type(source.overlap) is type(source.reference.fock) is CyclicAOOperator
    assert source.conventions.same_source and source.conventions.exchange_q0 == "bvk-ewald"
    result = run_translation_dlpno_mp2(source, mesh,
        conv_tol_energy=1e-13, conv_tol_residual=1e-12)
    assert result.solution.converged and result.solution.max_residual < 1e-12
    assert result.pair_build.max_translation_error is None
    assert result.solution.provenance["transport_preparation"]["metric_backend"] == "cyclic-blocks"
    auxname = default_aux_for(basis.name)
    kcart = np.asarray(scf.cyclic_gamma_mesh(system, mesh).kpoints_cart)
    controls = dict(reference=hf, fit_source="classic", rsgdf_tail_ke_cutoff=None,
        lattice_cutoff_bohr=options.lattice_opts.cutoff_bohr,
        rsgdf_ke_cutoff=200., gdf_linear_dep_threshold=1e-9)
    selected, _ = posthf._selected_lpq_cache(system, basis, kcart, auxname, **controls)
    lov, _ = posthf._selected_lov_cache(system, basis, kcart,
        np.asarray(hf.mo_coeffs), 1, auxname, **controls)
    ss, os, _ = vq._vibeqc_core.aiccm2026dev_b_mp2_energy_from_lov(
        lov, np.ascontiguousarray(hf.mo_energies), mesh, (0, 0, 0), 1e-12)
    dense = posthf._build_b_real_reference(system, basis, mesh, hf, selected, auxname)
    indices = np.arange(source.basis.nbasis)
    np.testing.assert_allclose(source.overlap.block(indices, indices), dense.hf.overlap, atol=1e-12)
    np.testing.assert_allclose(source.reference.fock.block(indices, indices), dense.hf.fock, atol=1e-12)
    np.testing.assert_allclose(source.factors.home_factors, dense.df.home_factors, atol=1e-12)
    expected = posthf._complete_space_mp2_audit(dense)/np.prod(mesh)
    record_property("e_corr_per_cell", result.solution.e_corr_per_cell)
    record_property("character_e_corr_per_cell", ss+os)
    record_property("unreduced_e_corr_per_cell", expected)
    record_property("absolute_energy_error", abs(result.solution.e_corr_per_cell-expected))
    record_property("character_energy_error", abs(result.solution.e_corr_per_cell-ss-os))
    record_property("maximum_raw_residual", result.solution.max_residual)
    record_property("hf_energy_per_cell", hf.energy)
    assert result.solution.e_corr_per_cell == pytest.approx(ss+os, abs=1e-10)
    assert result.solution.e_corr_per_cell == pytest.approx(expected, abs=1e-10)
    assert result.solution.e_hf_per_cell == pytest.approx(hf.energy, abs=1e-12)
    assert result.solution.e_tot_per_cell == pytest.approx(hf.energy+expected, abs=1e-10)
    assert _digest(options) == before


def _accepted_translation_model(monkeypatch, *, fit_source="classic", representation="cyclic-blocks",
                                n_frozen=0, n_occ=1, degenerate_core=False):
    from vibeqc.periodic.chi.reference import prepare_chi_reference, _digest

    system, basis, hf, cache, auxiliary = _translation_attested_gamma_model(n_occ)
    if degenerate_core:
        hf.mo_energies[0][:n_occ] = -1.
        SC = hf.overlap[0] @ hf.mo_coeffs[0]
        hf.fock[0] = (SC*hf.mo_energies[0]) @ SC.T
    options = vq.PeriodicRHFOptions()
    options.conv_tol_energy, options.conv_tol_grad = 1e-11, 1e-9
    before = _digest(options)
    calls = []

    def owned_scf(actual_system, actual_basis, mesh, owned_options, **kwargs):
        assert owned_options is not options and _digest(owned_options) == before
        owned_options.max_iter = 7
        calls.append("scf")
        return hf

    def refit(*args, **kwargs):
        calls.append("refit")
        return {key: value.copy() for key, value in cache.items()}, auxiliary

    monkeypatch.setattr(posthf, "run_aiccm2026dev_b_rhf", owned_scf)
    monkeypatch.setattr(posthf, "_build_canonical_lpq_cache", refit)
    accepted = prepare_chi_reference(system, basis, options=options, fit_source=fit_source,
        matrix_representation=representation, n_frozen=n_frozen, progress=False, verbose=0)
    assert calls == (["scf", "refit"] if fit_source == "auto" else ["scf"])
    assert _digest(options) == before == accepted.options_identity
    return system, basis, options, accepted


@pytest.mark.parametrize("fit_source", ["classic", "auto"])
@pytest.mark.parametrize("representation", ["dense", "cyclic-blocks"])
def test_translation_accepted_reuse_detaches_state_without_refitting(
    monkeypatch, fit_source, representation,
):
    from vibeqc.dlpno import translation_bloch
    from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2
    from vibeqc.periodic.chi.reference import _digest
    from vibeqc.periodic.chi.translation import prepare_chi_translation_source

    system, basis, options, accepted = _accepted_translation_model(
        monkeypatch, fit_source=fit_source, representation=representation)
    before = _digest(options)
    C = accepted.real_reference.hf.mo_coeffs.copy()
    eps = accepted.real_reference.hf.mo_energies.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("accepted translation source repeated SCF, fitting or home transform")

    for name in ("run_aiccm2026dev_b_rhf", "_selected_lpq_cache", "_build_canonical_lpq_cache",
                 "_lpq_cache_to_real_home_auxiliary", "_lpq_cache_to_real_supercell",
                 "_periodic_matrix_to_real_supercell", "_build_b_real_reference"):
        monkeypatch.setattr(posthf, name, forbidden)
    if representation == "cyclic-blocks":
        monkeypatch.setattr(translation_bloch, "bloch_translation_reference", forbidden)
    kwargs = dict(reference=accepted, fit_source=fit_source, progress=False)
    first = prepare_chi_translation_source(system, basis, options=options, **kwargs)
    first_record = first.provenance["adapter"]["accepted_reference"]
    assert first_record["matrix_representation"] == representation
    assert first_record["bloch_transform"] == (
        "reused" if representation == "cyclic-blocks" else "built-from-sealed-bloch")
    assert first.provenance["adapter"]["correlation_fit_status"] == "reused"
    assert first.conventions.same_source is (fit_source == "classic")
    assert first.factors is accepted.real_reference.df
    assert not np.shares_memory(first.reference.mo_coeffs, accepted.real_reference.hf.mo_coeffs)
    assert not np.shares_memory(first.reference.mo_energies, accepted.real_reference.hf.mo_energies)
    if representation == "cyclic-blocks":
        assert first.overlap is accepted.real_reference.hf.overlap
    # Caller replacement of the returned reference and provenance cannot
    # modify the sealed handle or a later source derived from it.
    first.reference.mo_coeffs = np.zeros_like(first.reference.mo_coeffs)
    first.reference.mo_energies = np.zeros_like(first.reference.mo_energies)
    first.provenance["adapter"]["bloch_transform"]["n_frozen"] = 999
    second = prepare_chi_translation_source(system, basis, **kwargs)
    np.testing.assert_array_equal(accepted.real_reference.hf.mo_coeffs, C)
    np.testing.assert_array_equal(accepted.real_reference.hf.mo_energies, eps)
    assert second.reference is not first.reference
    assert second.provenance["adapter"]["bloch_transform"]["n_frozen"] == 0
    result = run_translation_dlpno_mp2(second, (1, 1, 1),
        conv_tol_energy=1e-13, conv_tol_residual=1e-12)
    assert result.solution.converged
    canonical = accepted.hf_result
    ov = np.einsum("Pmn,mi,na->Pia", accepted.factors[(0, 0)].real,
                   canonical.mo_coeffs[0][:, :1], canonical.mo_coeffs[0][:, 1:])
    gap = 2*(canonical.mo_energies[0][0]-canonical.mo_energies[0][1])
    expected = np.sum(ov*ov)**2/gap
    assert result.solution.e_corr_per_cell == pytest.approx(expected, abs=1e-12)
    assert _digest(options) == before


@pytest.mark.parametrize("fault", ["owner", "runtime", "controls", "mesh", "kfrac", "coefficients",
    "factor", "home_coefficients", "home_fock", "frame", "frozen", "record"])
def test_translation_accepted_reuse_refuses_changed_seals(monkeypatch, fault):
    from vibeqc.dlpno.translation_operators import CyclicAOOperator
    from vibeqc.periodic.chi import reference as reference_module
    from vibeqc.periodic.chi.translation import prepare_chi_translation_source

    system, basis, options, accepted = _accepted_translation_model(monkeypatch, n_occ=2)
    kwargs = dict(reference=accepted, fit_source="classic", progress=False)
    mesh = (1, 1, 1)
    if fault == "owner":
        kwargs["reference"] = replace(accepted, owner_pid=accepted.owner_pid+1)
    elif fault == "runtime":
        monkeypatch.setattr(reference_module, "loaded_process_provenance", lambda: {"source": "changed"})
    elif fault == "controls":
        kwargs["rsgdf_ke_cutoff"] = 201.
    elif fault == "mesh":
        mesh = (2, 1, 1)
    elif fault == "kfrac":
        accepted.hf_result.kpoints_frac = np.array([[.25, 0., 0.]])
    elif fault == "coefficients":
        accepted.hf_result.mo_coeffs[0] = accepted.hf_result.mo_coeffs[0][:, :-1]
    elif fault == "factor":
        changed = dict(accepted.factors)
        changed[(0, 0)] = changed[(0, 0)]+.01
        kwargs["reference"] = replace(accepted, factors=changed)
    elif fault == "home_coefficients":
        accepted.real_reference.hf.mo_coeffs = accepted.real_reference.hf.mo_coeffs[:, :-1]
    elif fault == "home_fock":
        F = accepted.real_reference.hf.fock
        accepted.real_reference.hf.fock = CyclicAOOperator(F.blocks*1.01, F.mesh)
    elif fault == "frame":
        kwargs["reference"] = replace(accepted,
            correlation_fit_source=replace(accepted.correlation_fit_source, auxiliary_frame="changed"))
    elif fault == "frozen":
        kwargs["n_frozen"] = 1
    else:
        accepted.translation_record["overlap"]["identity"] = "changed"

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid accepted source rebuilt work")

    monkeypatch.setattr(posthf, "run_aiccm2026dev_b_rhf", forbidden)
    monkeypatch.setattr(posthf, "_selected_lpq_cache", forbidden)
    with pytest.raises(ValueError, match="changed|mismatch|different process"):
        prepare_chi_translation_source(system, basis, mesh, options, **kwargs)


def test_translation_accepted_block_partition_and_legacy_refusal(monkeypatch):
    from vibeqc.periodic.chi.translation import prepare_chi_translation_source

    system, basis, options, accepted = _accepted_translation_model(monkeypatch, n_occ=2, n_frozen=1)
    source = prepare_chi_translation_source(system, basis, options=options,
        reference=accepted, fit_source="classic", n_frozen=1)
    assert source.provenance["adapter"]["n_frozen"] == accepted.n_frozen == 1
    with pytest.raises(ValueError, match="frozen partition mismatch"):
        prepare_chi_translation_source(system, basis, reference=accepted, fit_source="classic")
    for consumer in (posthf.run_aiccm2026dev_b_mp2, posthf.run_aiccm2026dev_b_dlpno_mp2):
        with pytest.raises(ValueError, match="dense accepted reference"):
            consumer(system, basis, options=options, reference=accepted, fit_source="classic")


@pytest.mark.parametrize("degenerate", [False, True])
def test_translation_dense_accepted_migration_audits_requested_frozen_gap(monkeypatch, degenerate):
    from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2
    from vibeqc.periodic.chi.translation import prepare_chi_translation_source

    system, basis, options, accepted = _accepted_translation_model(monkeypatch,
        n_occ=2, representation="dense", degenerate_core=degenerate)
    kwargs = dict(reference=accepted, fit_source="classic", n_frozen=1)
    if degenerate:
        with pytest.raises(ValueError, match="frozen|gap"):
            prepare_chi_translation_source(system, basis, options=options, **kwargs)
    else:
        source = prepare_chi_translation_source(system, basis, options=options, **kwargs)
        assert source.provenance["adapter"]["n_frozen"] == 1
        result = run_translation_dlpno_mp2(source, (1, 1, 1), n_frozen=1,
            conv_tol_energy=1e-13, conv_tol_residual=1e-12)
        expected = posthf._complete_space_mp2_audit(accepted.real_reference, n_frozen=1)
        assert result.solution.converged
        assert result.solution.e_corr_per_cell == pytest.approx(expected, abs=1e-12)


@pytest.mark.slow
def test_translation_accepted_block_public_h2_reuses_one_scf(monkeypatch, record_property):
    from vibeqc.dlpno import translation_bloch
    from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2
    from vibeqc.periodic.chi.reference import prepare_chi_reference
    from vibeqc.periodic.chi.translation import prepare_chi_translation_source

    system, basis = cell()
    mesh = (2, 1, 1)
    options = vq.PeriodicRHFOptions()
    options.conv_tol_energy, options.conv_tol_grad = 1e-11, 1e-9
    options.damping, options.dynamic_damping = 0., False
    with monkeypatch.context() as preparation:
        def forbidden_dense(*args, **kwargs):
            raise AssertionError("block handle preparation expanded a dense AO operator")
        preparation.setattr(posthf, "_build_b_real_reference", forbidden_dense)
        preparation.setattr(posthf, "_periodic_matrix_to_real_supercell", forbidden_dense)
        accepted = prepare_chi_reference(system, basis, mesh, options, fit_source="classic",
            matrix_representation="cyclic-blocks", progress=False, verbose=0)
    dense = posthf._build_b_real_reference(system, basis, mesh, accepted.hf_result,
        dict(accepted.factors), dict(accepted.controls)["auxiliary_name"])
    expected = posthf._complete_space_mp2_audit(dense)/np.prod(mesh)

    def forbidden(*args, **kwargs):
        raise AssertionError("accepted block source repeated construction")

    for name in ("run_aiccm2026dev_b_rhf", "_selected_lpq_cache", "_build_canonical_lpq_cache",
                 "_lpq_cache_to_real_home_auxiliary", "_build_b_real_reference"):
        monkeypatch.setattr(posthf, name, forbidden)
    monkeypatch.setattr(translation_bloch, "bloch_translation_reference", forbidden)
    sources = [prepare_chi_translation_source(system, basis, mesh, options,
        reference=accepted, fit_source="classic") for _ in range(2)]
    for source in sources:
        result = run_translation_dlpno_mp2(source, mesh,
            conv_tol_energy=1e-13, conv_tol_residual=1e-12)
        assert result.solution.converged
        assert result.solution.e_corr_per_cell == pytest.approx(expected, abs=1e-10)
        assert result.solution.e_hf_per_cell == pytest.approx(accepted.hf_result.energy, abs=1e-12)
        assert source.conventions.same_source
    assert sources[0].identity == sources[1].identity
    record_property("e_corr_per_cell", result.solution.e_corr_per_cell)
    record_property("unreduced_e_corr_per_cell", expected)
    record_property("absolute_energy_error", abs(result.solution.e_corr_per_cell-expected))
    record_property("maximum_raw_residual", result.solution.max_residual)
