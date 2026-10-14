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
