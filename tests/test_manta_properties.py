"""Static molecular properties follow the executed integral source."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, properties, bond_analysis
from vibeqc.output.formats.population import compute_population_summary
from tests.test_manta_bupo_scf import _system, _options, _bupo
from tests.test_manta_ri_bupo_scf import _coupling


def _run(method="rhf", algorithm="exact", backend="manta"):
    mol, basis = _system(method)
    result = getattr(vq, "run_"+method)(mol, basis, _options(method),
        integral_engine=vq.IntegralEngineOptions(backend=backend),
        coulomb=vq.CoulombBuildOptions(algorithm,
            basis if algorithm in ("ri", "ri_bupo") else None,
            bupo=_bupo(), ri_bupo=_coupling()))
    assert result.converged
    return mol, basis, result


def _density_only(result):
    fields = ("density_alpha", "density_beta") if hasattr(result, "density_alpha") else ("density",)
    return SimpleNamespace(**{field: getattr(result, field) for field in fields})


def _properties(result, basis, mol):
    dipole = properties.dipole_moment(result, basis, mol, origin=[.1, -.3, .7])
    return [properties.mulliken_charges(result, basis, mol),
            properties.loewdin_charges(result, basis, mol),
            properties.mayer_bond_orders(result, basis, mol),
            bond_analysis.wiberg_bond_orders(result, basis, mol),
            bond_analysis.delocalization_index(result, basis, mol),
            properties.natural_orbitals(result, basis).occupations,
            np.array([dipole.x, dipole.y, dipole.z])]


def _unexpected(*args, **kwargs):
    raise AssertionError("The selected property must not call the other provider")


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("algorithm", ["exact", "ri", "bupo", "ri_bupo"])
def test_selected_properties_match_independent_libint_with_no_hidden_calls(monkeypatch, method, algorithm):
    mol, basis, result = _run(method, algorithm)
    reference = _density_only(result)
    expected = _properties(reference, basis, mol)
    expected_summary = compute_population_summary(reference, basis, mol)
    scf_before = manta.molecular_engine_provenance(result)
    events = []
    monkeypatch.setattr(vq.output, "record", lambda *args, **kwargs: events.append((args, kwargs)))
    monkeypatch.setattr(properties, "compute_overlap", _unexpected)
    monkeypatch.setattr(properties, "compute_dipole", _unexpected)
    monkeypatch.setattr(properties, "sad_density", _unexpected)
    monkeypatch.setattr(bond_analysis, "compute_overlap", _unexpected)
    monkeypatch.setattr(vq._vibeqc_core, "compute_overlap", _unexpected)
    monkeypatch.setattr(vq._vibeqc_core, "compute_dipole", _unexpected)
    for actual, ref in zip(_properties(result, basis, mol), expected, strict=True):
        np.testing.assert_allclose(actual, ref, atol=2e-11, rtol=2e-11)
    summary = compute_population_summary(result, basis, mol)
    assert not summary.errors and not expected_summary.errors
    assert "Hirshfeld atomic SAD" in summary.unavailable["hirshfeld"]
    assert not summary.hirshfeld_atoms
    np.testing.assert_allclose([row[-1] for row in summary.mulliken_spin_atoms],
        [row[-1] for row in expected_summary.mulliken_spin_atoms], atol=2e-12)
    assert manta.molecular_engine_provenance(result) == scf_before
    evidence = [kwargs["property_integral_provenance"] for args, kwargs in events]
    assert len(evidence) == (17 if method in ("uhf", "uks") else 16)
    for receipt in evidence:
        assert receipt["provider"] == "manta"
        assert receipt["orbital_basis_identity"] == scf_before["orbital_basis_identity"]
        assert receipt["molecule_identity"] == scf_before["molecule_identity"]
        assert receipt["shell_calls"] == basis.nshells**2
        assert receipt["primitive_integrals_evaluated"] > 0
        assert len(receipt["source_identity"]) == 64
        assert receipt["budget_scope"] == "property-integral-build"
    json.dumps(evidence)


@pytest.mark.parametrize("selected", [False, True])
def test_legacy_and_explicit_libint_still_use_libint(monkeypatch, selected):
    mol, basis, result = _run(backend="libint")
    expected = _properties(result, basis, mol)
    monkeypatch.setattr(manta, "compute_one_electron", _unexpected)
    if not selected:
        result = _density_only(result)
    for actual, ref in zip(_properties(result, basis, mol), expected, strict=True):
        np.testing.assert_array_equal(actual, ref)


@pytest.mark.parametrize("backend", ["manta", "libint"])
@pytest.mark.parametrize("mismatch", ["basis", "geometry", "charge", "multiplicity"])
def test_mismatched_sources_refuse_before_evaluating_integrals(monkeypatch, backend, mismatch):
    mol, basis, result = _run(backend=backend)
    if mismatch == "basis":
        basis = vq.BasisSet(mol, "6-31g")
    elif mismatch == "geometry":
        mol = vq.Molecule([vq.Atom(1, [0., 0., .1]), vq.Atom(1, [.2, -.1, 1.8])], 0, 1)
    else:
        mol = vq.Molecule(mol.atoms, 1 if mismatch == "charge" else 0,
                          2 if mismatch == "charge" else 3)
    monkeypatch.setattr(properties, "compute_overlap", _unexpected)
    monkeypatch.setattr(properties, "compute_dipole", _unexpected)
    monkeypatch.setattr(manta, "compute_one_electron", _unexpected)
    for operation in [properties.mulliken_charges, properties.loewdin_charges,
                      properties.mayer_bond_orders, properties.dipole_moment,
                      properties.hirshfeld_charges, bond_analysis.wiberg_bond_orders,
                      bond_analysis.delocalization_index]:
        with pytest.raises(ValueError, match="does not match"):
            operation(result, basis, mol)
    if mismatch == "basis":
        with pytest.raises(ValueError, match="does not match"):
            properties.natural_orbitals(result, basis)


def test_dipole_origin_shift_and_retained_axis_budget(monkeypatch):
    mol = vq.Molecule([vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [.2, -.1, 1.8])], 1, 2)
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_uhf(mol, basis, _options("uhf"),
                        integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert result.converged
    calls = []
    original = manta.compute_one_electron
    def observed(bra, ket, request, budget):
        calls.append((list(request.origin), budget.other_live_bytes))
        return original(bra, ket, request, budget)
    monkeypatch.setattr(manta, "compute_one_electron", observed)
    zero = properties.dipole_moment(result, basis, mol, origin=[0., 0., 0.])
    shift = np.array([.7, -.2, 1.3])
    moved = properties.dipole_moment(result, basis, mol, origin=shift)
    np.testing.assert_allclose(np.array([moved.x, moved.y, moved.z]),
        np.array([zero.x, zero.y, zero.z])-mol.charge*shift, atol=2e-12)
    assert [call[1] for call in calls] == [0, 8*basis.nbasis**2, 16*basis.nbasis**2]*2


def test_same_content_basis_name_and_unconfirmed_density_are_valid():
    mol, basis = _system("rhf")
    options = _options("rhf"); options.max_iter = 0
    result = vq.run_rhf(mol, basis, options,
                        integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert not result.converged
    # Property evaluation does not manufacture convergence or require it.
    before = manta.molecular_engine_provenance(result)
    copy = vq.BasisSet(mol, "sto-3g")
    actual = _properties(result, copy, mol)
    expected = _properties(_density_only(result), basis, mol)
    for value, ref in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, ref, atol=2e-12)
    assert manta.molecular_engine_provenance(result) == before


def test_forged_receipt_is_not_used_as_provider_authority():
    mol, basis = _system("rhf")
    result = SimpleNamespace(density=np.eye(basis.nbasis),
                             engine_provenance=SimpleNamespace(one_electron_provider="manta"))
    with pytest.raises(TypeError, match="native engine provenance"):
        properties.mulliken_charges(result, basis, mol)


def test_unqualified_hirshfeld_refuses_before_grid_or_atomic_scf(monkeypatch):
    mol, basis, result = _run()
    monkeypatch.setattr(properties, "build_grid", _unexpected)
    monkeypatch.setattr(properties, "sad_density", _unexpected)
    with pytest.raises(NotImplementedError, match="SAD promolecules"):
        properties.hirshfeld_charges(result, basis, mol)


class _LegacyDensityView:
    engine_provenance = None

    def __init__(self, result):
        self.result = result

    def __getattr__(self, name):
        return getattr(self.result, name)


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("algorithm", ["exact", "ri", "bupo", "ri_bupo"])
def test_iao_population_uses_selected_overlaps_for_each_spin_and_coulomb_source(monkeypatch, method, algorithm):
    from vibeqc import iao
    from vibeqc.iao_population import analyse_iao
    mol, basis, result = _run(method, algorithm)
    expected = analyse_iao(_LegacyDensityView(result), basis, mol, method=method)
    assert expected.available, expected.unavailable_reason
    before = manta.molecular_engine_provenance(result)
    for name in ("compute_overlap", "compute_overlap_two_basis", "compute_dipole"):
        monkeypatch.setattr(iao, name, _unexpected)
    monkeypatch.setattr(vq._vibeqc_core, "compute_overlap", _unexpected)
    monkeypatch.setattr(properties, "compute_overlap", _unexpected)
    actual = analyse_iao(result, basis, mol, method=method)
    assert actual.available, actual.unavailable_reason
    for field in ("charges", "bond_orders", "populations", "iaos_alpha"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=3e-11)
    if method in ("uhf", "uks"):
        np.testing.assert_allclose(actual.spin_populations, expected.spin_populations, atol=3e-11)
    assert manta.molecular_engine_provenance(result) == before


@pytest.fixture(scope="module")
def _water_localization_source():
    mol = vq.Molecule([vq.Atom(8, [0., 0., 0.]), vq.Atom(1, [0., 1.4, -.9]),
                      vq.Atom(1, [.1, -1.5, -1.])], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_rhf(mol, basis, _options("rhf"),
                        integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert result.converged
    return mol, basis, result


@pytest.mark.parametrize("method", ["ibo", "boys", "pipek-mezey"])
@pytest.mark.parametrize("reuse", [False, True])
def test_localization_uses_manta_overlap_and_moments_without_legacy_calls(monkeypatch, method, reuse, _water_localization_source):
    from vibeqc import iao
    mol, basis, result = _water_localization_source
    occupied = np.asarray(result.mo_coeffs)[:, :mol.n_electrons()//2]
    expected = iao.analyse_localization(mol, basis, occupied, method=method)
    for name in ("compute_overlap", "compute_overlap_two_basis", "compute_dipole"):
        monkeypatch.setattr(iao, name, _unexpected)
    monkeypatch.setattr(properties, "compute_overlap", _unexpected)
    monkeypatch.setattr(properties, "compute_dipole", _unexpected)
    precomputed = {}
    if reuse:
        reference = iao.iao_reference(mol, basis, source_result=result)
        overlap = properties._property_overlap(result, basis, mol)
        iaos = iao.build_iaos(occupied, overlap, reference.overlap, reference.cross_overlap)
        precomputed = dict(reference=reference, iaos=iaos)
    actual = iao.analyse_localization(mol, basis, occupied, method=method,
                                     source_result=result, **precomputed)
    for field in ("charges", "centroids", "atom_populations"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=3e-10)
    assert actual.objective_final == pytest.approx(expected.objective_final, abs=3e-10)
    # Localization must preserve the occupied density, not just descriptors.
    np.testing.assert_allclose(actual.coefficients @ actual.coefficients.T,
                               occupied @ occupied.T, atol=3e-11)


def test_cross_basis_iao_receipts_and_reference_storage(monkeypatch):
    from vibeqc import iao
    mol, _ = _system("rhf")
    basis = vq.BasisSet(mol, "6-31g")
    result = vq.run_rhf(mol, basis, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    expected = iao.iao_reference(mol, basis)
    events, calls = [], []
    original = manta.compute_one_electron
    def observed(bra, ket, request, budget):
        calls.append((bra.nbasis, ket.nbasis, budget.other_live_bytes))
        return original(bra, ket, request, budget)
    monkeypatch.setattr(manta, "compute_one_electron", observed)
    monkeypatch.setattr(vq.output, "record", lambda *args, **kwargs: events.append(kwargs))
    actual = iao.iao_reference(mol, basis, source_result=result)
    np.testing.assert_allclose(actual.overlap, expected.overlap, atol=3e-12)
    np.testing.assert_allclose(actual.cross_overlap, expected.cross_overlap, atol=3e-12)
    assert actual.cross_overlap.shape == (4, 2)
    assert calls == [(2, 2, 0), (4, 2, actual.overlap.nbytes)]
    receipts = [event["property_integral_provenance"] for event in events]
    assert [receipt["integral_elements"] for receipt in receipts] == [4, 8]
    assert receipts[0]["source_identity"] != receipts[1]["source_identity"]
    assert receipts[0]["bra_basis_identity"] == receipts[0]["ket_basis_identity"]
    assert receipts[1]["bra_basis_identity"] != receipts[1]["ket_basis_identity"]
    assert receipts[1]["bra_basis_identity"] == result.engine_provenance.orbital_basis_identity


def test_localization_and_iao_reject_changed_source_before_integrals(monkeypatch):
    from vibeqc import iao
    from vibeqc.iao_population import analyse_iao
    mol, basis, result = _run()
    other = vq.BasisSet(mol, "6-31g")
    monkeypatch.setattr(manta, "compute_one_electron", _unexpected)
    with pytest.raises(ValueError, match="does not match"):
        iao.iao_reference(mol, other, source_result=result)
    with pytest.raises(ValueError, match="does not match"):
        iao.analyse_localization(mol, other, np.zeros((other.nbasis, 1)), source_result=result)
    unavailable = analyse_iao(result, other, mol, method="rhf")
    assert not unavailable.available and "does not match" in unavailable.unavailable_reason


def test_libint_localization_does_not_execute_manta(monkeypatch):
    from vibeqc import iao
    from vibeqc.iao_population import analyse_iao
    mol, basis, result = _run(backend="libint")
    occupied = np.asarray(result.mo_coeffs)[:, :1]
    expected = iao.analyse_localization(mol, basis, occupied)
    monkeypatch.setattr(manta, "compute_one_electron", _unexpected)
    actual = iao.analyse_localization(mol, basis, occupied, source_result=result)
    np.testing.assert_allclose(actual.charges, expected.charges, atol=2e-12)
    assert analyse_iao(result, basis, mol, method="rhf").available
