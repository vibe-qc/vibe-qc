"""Executable QVF option contracts, without electronic-structure work."""
from __future__ import annotations

import copy

import pytest
from vibeqc._qvf_options import _OPTION_TYPES, hydrate_option


@pytest.mark.parametrize("key", list(_OPTION_TYPES))
def test_all_declared_option_records_construct_and_reject_unknown(key):
    result = hydrate_option(key, {})
    assert type(result).__name__ == _OPTION_TYPES[key][1]
    with pytest.raises(ValueError, match=rf"options\.{key}\.not_an_option"):
        hydrate_option(key, {"not_an_option": 1})
    assert hydrate_option(key, None) is None
    with pytest.raises(ValueError, match=rf"options\.{key}.*JSON object"):
        hydrate_option(key, 7)


def test_rhf_requested_controls_and_defaults_are_preserved():
    request = {"use_diis": True, "damping": 0.0,
               "max_iter": 120, "conv_tol_energy": 1e-8}
    original = copy.deepcopy(request)
    result = hydrate_option("rhf_options", request)
    assert {k: getattr(result, k) for k in request} == request
    assert request == original
    default = type(result)()
    assert result.initial_guess == default.initial_guess
    assert result.dynamic_damping == default.dynamic_damping


@pytest.mark.parametrize("key,value", [
    ("max_iter", True), ("max_iter", 12.5), ("max_iter", "120"),
    ("use_diis", 1), ("use_diis", "false"), ("damping", False),
    ("damping", "0"), ("conv_tol_energy", float("nan")),
    ("conv_tol_energy", float("inf")), ("damping", 10**1000),
])
def test_native_wrong_types_refused_at_field_path(key, value):
    with pytest.raises(ValueError, match=rf"options\.rhf_options\.{key}"):
        hydrate_option("rhf_options", {key: value})


@pytest.mark.parametrize("key", ["rhf_options", "uhf_options", "rks_options", "uks_options"])
def test_native_enum_and_nested_record(key):
    result = hydrate_option(key, {
        "initial_guess": "sad", "scf_accelerator": "DIIS",
        "davidson": {"max_iter": 19}, "level_shift_schedule": [0.2, 0],
    })
    assert result.initial_guess.name == "SAD"
    assert result.scf_accelerator.name == "DIIS"
    assert result.davidson.max_iter == 19
    assert result.level_shift_schedule == [0.2, 0.0]
    with pytest.raises(ValueError, match=rf"options\.{key}\.davidson\.max_iter"):
        hydrate_option(key, {"davidson": {"max_iter": True}})
    with pytest.raises(ValueError, match=rf"options\.{key}\.initial_guess"):
        hydrate_option(key, {"initial_guess": 2})
    with pytest.raises(ValueError, match=rf"options\.{key}\.initial_guess"):
        hydrate_option(key, {"initial_guess": "not-a-guess"})


def test_nested_grid_and_vector_element_types():
    result = hydrate_option("rks_options", {"grid": {"n_radial": 37}})
    assert result.grid.n_radial == 37
    with pytest.raises(ValueError, match=r"options\.rks_options\.grid\.not_a_grid_field"):
        hydrate_option("rks_options", {"grid": {"not_a_grid_field": 1}})
    with pytest.raises(ValueError, match=r"options\.uhf_options\.atomic_spins\[1\]"):
        hydrate_option("uhf_options", {"atomic_spins": [1, False]})


def test_frozen_smearing_dataclass_uses_constructor_validation():
    result = hydrate_option("smearing", {"temperature": 0.01, "flavor": "fd"})
    assert result.temperature == 0.01
    assert result.flavor == "fermi-dirac"
    with pytest.raises(ValueError, match="options.smearing"):
        hydrate_option("smearing", {"temperature": -1})
    with pytest.raises(ValueError, match="options.smearing.mp_order"):
        hydrate_option("smearing", {"mp_order": True})


def test_python_dataclasses_nested_lists_optional_and_literals():
    result = hydrate_option("casscf_options", {
        "weights": [0.5, 0.5], "orbital_step": "nr",
        "selected_ci_options": {"max_det_per_iter": 71},
    })
    assert result.weights == [0.5, 0.5]
    assert result.selected_ci_options.max_det_per_iter == 71
    assert hydrate_option("thermo_options", {"electronic_degeneracy": 3}).electronic_degeneracy == 3
    assert hydrate_option("caspt2_options", {"nac_pair": [0, 1]}).nac_pair == (0, 1)
    with pytest.raises(ValueError, match=r"options\.dmrg_options\.bond_dim_schedule\[1\]"):
        hydrate_option("dmrg_options", {"bond_dim_schedule": [4, "8"]})
    with pytest.raises(ValueError, match="options.casscf_options.orbital_step"):
        hydrate_option("casscf_options", {"orbital_step": "unknown"})


def test_ccsd_constructor_preserves_explicitness_and_aliases():
    default = hydrate_option("ccsd_options", {})
    assert default._n_frozen_core_explicit is False
    explicit = hydrate_option("ccsd_options", {"n_frozen_core": 0, "triples": False})
    assert explicit._n_frozen_core_explicit is True
    assert explicit.n_frozen_core == 0
    assert explicit.compute_triples is False
    with pytest.raises(ValueError, match="options.ccsd_options.n_frozen_core"):
        hydrate_option("ccsd_options", {"n_frozen_core": True})


@pytest.mark.parametrize("n_frozen_core", [0, 2])
def test_ccsd_frozen_core_and_residual_tolerance_survive_archive(tmp_path, n_frozen_core):
    from vibeqc import Atom, Molecule
    from vibeqc.qvf_job import _runner_kwargs, load_job_container, write_pending_qvf

    request = {"n_frozen_core": n_frozen_core, "conv_tol_residual": 2e-9,
               "compute_triples": False}
    molecule = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    path = write_pending_qvf(
        molecule, tmp_path / "ccsd-controls", method="ccsd",
        options={"ccsd_options": request},
    )

    def runner(molecule, ccsd_options=None):
        raise AssertionError("option mapping must not execute CCSD")

    loaded = load_job_container(path)
    result = _runner_kwargs(loaded, runner)["ccsd_options"]
    assert loaded.spec["options"]["ccsd_options"] == request
    assert result._n_frozen_core_explicit is True
    assert {key: getattr(result, key) for key in request} == request


@pytest.mark.parametrize("value", [True, "1e-8", float("nan"), float("inf")])
def test_ccsd_residual_tolerance_refuses_invalid_types(value):
    with pytest.raises(ValueError, match=r"options\.ccsd_options\.conv_tol_residual"):
        hydrate_option("ccsd_options", {"conv_tol_residual": value})


@pytest.mark.parametrize("key", ["mp2_options", "ump2_options"])
def test_nullable_native_frozen_core_is_not_guessed_from_none(key):
    assert hydrate_option(key, {"n_frozen_core": None}).n_frozen_core is None
    assert hydrate_option(key, {"n_frozen_core": 0}).n_frozen_core == 0
    with pytest.raises(ValueError, match=f"options.{key}.n_frozen_core"):
        hydrate_option(key, {"n_frozen_core": False})


@pytest.mark.parametrize("key,value,path", [
    ("rhf_options", {"ecp_centers": [{}]}, "ecp_centers"),
    ("rhf_options", {"read_density": [[1.0]]}, "read_density"),
    ("casscf_options", {"pt2": {}}, "pt2"),
    ("dlpno_options", {"pair_distance_fn": {}}, "pair_distance_fn"),
    ("rhf_options", {"__class__": "dangerous"}, "rhf_options"),
])
def test_opaque_fields_fail_before_a_runner_is_called(key, value, path):
    with pytest.raises(ValueError, match=path):
        hydrate_option(key, value)


@pytest.mark.parametrize("key,value", [
    ("geom_opt_options", {"trust_radius": 0.1}),
    ("solvent", {"epsilon": 80.0}), ("fragments", [[0], [1]]),
    ("fmax", 0.01), ("write_cube", [0, 1]), ("dispersion", "d3bj"),
    ("dft_plus_u", []), ("geom_freeze", [0]), ("active_space", [2, 2]),
    ("dos_kmesh", [2, 2, 2]), ("localize", ["boys"]),
])
def test_existing_mapping_and_scalar_contracts_are_not_rewritten(key, value):
    assert hydrate_option(key, value) is value


@pytest.mark.parametrize("key", ["band_structure", "dlpno_ccsd_options", "dispersion"])
def test_unsupported_top_level_records_are_named(key):
    with pytest.raises(ValueError, match=f"options.{key}"):
        hydrate_option(key, {"unknown": 1})


@pytest.mark.parametrize("value", [[{"atoms": [0]}], [[True]], {"atoms": [0]}])
def test_fragment_runtime_records_are_not_atom_index_lists(value):
    with pytest.raises(ValueError, match="options.fragments"):
        hydrate_option("fragments", value)


@pytest.mark.parametrize("value", [[], "", False, 0, None])
def test_falsey_nonobject_options_are_not_silently_discarded(value):
    from pathlib import Path

    from vibeqc.qvf_job import QvfJobContainer, _runner_kwargs

    container = QvfJobContainer(
        path=Path("input.qvf"), manifest={}, structure={}, run_status="pending",
        spec={"job_type": "molecular", "options": value},
    )

    def runner(molecule, **kwargs):
        raise AssertionError("adapter validation must not call a runner")

    with pytest.raises(ValueError, match="options must be an object"):
        _runner_kwargs(container, runner)
