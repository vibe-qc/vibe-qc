"""Focused regressions for GFN2 parameter-content identity (issue #283)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.dispersion_d4_parameters import D4Params
from vibeqc.semiempirical.methods import gfn2 as _gfn2_method
from vibeqc.semiempirical.methods.gfn2 import GFN2Model
from vibeqc.semiempirical.methods import gfn2_params as _gfn2_params
from vibeqc.semiempirical.methods.gfn2_params import (
    GFN2ParameterUnavailableError,
    load_gfn2_params,
)
from vibeqc.semiempirical.runner import _gfn2_parameter_provenance


# Canonical content digest of the LOADED published GFN2-xTB parameter
# object. vibe-qc does not transcribe the parameter values into this
# repository: they are fetched at runtime from the grimme-lab/xtb
# reference file param_gfn2-xtb.txt (LGPL-3.0) and parsed into a
# GFN2ParameterSet. This digest therefore attests the parsed object, not
# the upstream file bytes; the latter are tracked separately as the
# cache's `source_sha256`.
#
# 2026-08-27 RE-ATTESTATION (issue #446, following issue #43's
# af133afd2). The previous pin,
#   71b83ab40d8f09081094ca6fedc18dc95e6ff3b325b23c3b70a10679e285b69e
# hashed a MIS-TRANSCRIBED load, and this is what was wrong with it:
# the upstream file lists each element's shells in occupation order,
# which is d-FIRST for the 41 transition-metal elements (ao=3d4s4p,
# 4d5s5p, 5d6s6p: Z 21-29, 39-47, 57-79), whereas KCNS/KCNP/KCND and
# POLYS/POLYP/POLYD are per-ANGULAR-MOMENTUM named lines. The TOML
# converter indexed those named lines by shell POSITION, so on exactly
# those 41 elements the s parameter landed on the d shell, the p
# parameter on the s shell, and the d parameter on the p shell (a
# 3-cycle). It manifested as an H0/band-channel error that grew with
# distance: Cu2 deviated from xtb by +35.8/+56.8/+139.5 mHa at
# d = 2.2/2.5/3.0 A, free Cu8/Cu16 never converged, and fcc Cu SECCM
# cells fell into a spurious charge-density-wave basin (issue #130).
#
# Nothing else in the hashed content moved. Rotating ONLY kcn/poly back
# to positional order on those 41 elements reproduces the old digest
# exactly, which is how a reader can confirm that this re-attestation
# changed a mapping and not a value. (The sibling l-projection of the
# derived on_site/zeta core arrays in gfn2_parameters.cpp, and the d
# shell STO-3G primitive-count fix in basis.cpp, do not feed this
# digest.)
#
# PUBLISHED SOURCE the re-attested values were checked against:
# Bannwarth, Ehlert & Grimme, J. Chem. Theory Comput. 15, 1652-1671
# (2019), doi:10.1021/acs.jctc.8b01176, Supporting Information
# Table S52, "Element-specific shell parameters employed in GFN2-xTB",
# which tabulates k^poly_{A,l}, the CN-dependent level enhancement,
# H^l_A and zeta_l PER SHELL LABEL. For Cu that table lists
# k^poly = 0.177983 (4s) / 0.149778 (4p) / -0.265089 (3d) and
# H^l_A = -6.922958 (4s) / -2.267723 (4p) / -9.506548 (3d) eV; the
# loaded set carries exactly those values on l = 0 / 1 / 2. The per-l
# keying is the paper's own: eq 17 defines
# H_kk = H^l_A - H^l_CN * CN'_A for (kappa in l in A), and eq 19
# defines the polynomial in terms of k^poly_{A,l}.
#
# NOTE ON WHAT WAS NOT CHECKED AGAINST THE PAPER: issue #43 located the
# defect by an element-by-element H0 bisection against tblite
# (h0.f90 / gfn2.f90). tblite is an independent implementation, not the
# publication, so that bisection is corroborating evidence only; the
# published-identity claim above rests on SI Table S52.
#
# To re-run the check instead of trusting this constant, see
# tests/test_gfn2_xtb.py::TestGFN2DFirstParameterProjection and the
# "Published-parameter identity" section of
# docs/user_guide/semiempirical.md.
GFN2_PUBLISHED_SHA256 = (
    "0b3c70a5a7a8dec49953a59e72e8709a9b6e9a58ec55fb063f7d08a3067cc8ef"
)
GFN2_PUBLISHED_IDENTITY = "published:gfn2-xtb-2019-v1"


@pytest.fixture
def published_params() -> _xtb.GFN2ParameterSet:
    """Return a fresh offline parameter builder or skip without fetching."""
    try:
        return load_gfn2_params(allow_fetch=False)
    except GFN2ParameterUnavailableError as exc:
        pytest.skip(f"current local GFN2 parameter cache unavailable: {exc}")


def _h2() -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
    )


def _element(
    Z: int,
    shells: tuple[tuple[int, int, float, float, float, float, float], ...],
    *,
    gam: float,
) -> _xtb.GFN2ElementData:
    element = _xtb.GFN2ElementData()
    element.Z = Z
    element.gam = gam
    for n, l, en, zeta, k_en, kcn, poly in shells:
        element.add_shell(l, en, zeta, k_en, kcn, poly, n)
    return element


def _pair(alpha: float, k_ab: float) -> _xtb.GFN2RepulsivePair:
    pair = _xtb.GFN2RepulsivePair()
    pair.alpha = alpha
    pair.k_ab = k_ab
    return pair


def _compact_parameters(
    *,
    reverse: bool = False,
    carbon_gam: float = 0.4,
    carbon_p_en: float = -0.2,
    cross_k_ab: float = 6.0,
) -> _xtb.GFN2ParameterSet:
    params = _xtb.GFN2ParameterSet()
    hydrogen = _element(
        1,
        ((1, 0, -0.5, 1.2, 1.0, 0.0, 0.0),),
        gam=0.5,
    )
    # Repeated n across distinct l is part of the published shell convention.
    carbon = _element(
        6,
        (
            (2, 0, -0.4, 1.6, 1.0, 0.01, 0.02),
            (2, 1, carbon_p_en, 1.7, 1.1, -0.01, -0.03),
        ),
        gam=carbon_gam,
    )
    elements = ((1, hydrogen), (6, carbon))
    pairs = (
        (1, 1, _pair(0.7, 1.0)),
        (1, 6, _pair(0.8, cross_k_ab)),
        (6, 6, _pair(0.9, 36.0)),
    )
    if reverse:
        elements = tuple(reversed(elements))
        pairs = tuple(
            (Z2, Z1, pair) for Z1, Z2, pair in reversed(pairs)
        )

    for _Z, element in elements:
        params.add_element(element)
    for Z1, Z2, pair in pairs:
        params.set_repulsive_pair(Z1, Z2, pair)
    return params


def _assert_custom(params: _xtb.GFN2ParameterSet) -> str:
    digest = params.content_sha256()
    assert params.parameter_identity() == f"custom:{digest}"
    assert params.metadata().origin == f"custom:{digest}"
    return digest


def _fake_native_result(params: _xtb.GFN2ParameterSet) -> SimpleNamespace:
    return SimpleNamespace(
        converged=True,
        energy=-1.0,
        n_iter=3,
        parameter_identity=params.parameter_identity(),
        parameter_sha256=params.content_sha256(),
    )


def test_loader_has_exact_published_identity(
    published_params: _xtb.GFN2ParameterSet,
) -> None:
    metadata = published_params.metadata()

    assert published_params.content_sha256() == GFN2_PUBLISHED_SHA256
    assert published_params.parameter_identity() == GFN2_PUBLISHED_IDENTITY
    assert metadata.parameter_hash == GFN2_PUBLISHED_SHA256
    assert metadata.method_name == "gfn2-xtb"
    assert metadata.version == "gfn2-xtb-2019"
    assert metadata.origin == "published"
    assert metadata.license == "LGPL-3.0 (original parameter set)"
    assert metadata.doi_or_url == "10.1021/acs.jctc.8b01176"
    assert metadata.n_elements == 86
    assert list(metadata.element_list) == list(range(1, 87))


def test_loader_cache_lineage_is_bound_to_the_exact_parameter_object(
    published_params: _xtb.GFN2ParameterSet,
) -> None:
    lineage = _gfn2_params.gfn2_parameter_lineage(published_params)

    assert lineage is not None
    assert lineage["parameter_sha256"] == GFN2_PUBLISHED_SHA256
    cache_bytes = Path(lineage["cache_path"]).read_bytes()
    assert lineage["cache_sha256"] == hashlib.sha256(cache_bytes).hexdigest()
    assert _gfn2_params.gfn2_parameter_lineage(_compact_parameters()) is None

    _mutate_hydrogen_gam(published_params)
    assert _gfn2_params.gfn2_parameter_lineage(published_params) is None


def _mutate_hydrogen_gam(params: _xtb.GFN2ParameterSet) -> None:
    hydrogen = params.element_data(1)
    hydrogen.gam = np.nextafter(float(hydrogen.gam), np.inf)
    params.add_element(hydrogen)


def test_informational_d4_globals_do_not_change_native_identity(
    published_params: _xtb.GFN2ParameterSet,
) -> None:
    baseline = published_params.content_sha256()
    published_params.d4_s8 = np.nextafter(float(published_params.d4_s8), np.inf)

    assert published_params.content_sha256() == baseline
    assert published_params.parameter_identity() == GFN2_PUBLISHED_IDENTITY


def test_element_scalar_and_shell_field_mutations_change_identity() -> None:
    baseline = _compact_parameters()
    scalar_mutation = _compact_parameters(
        carbon_gam=np.nextafter(0.4, np.inf)
    )
    shell_mutation = _compact_parameters(
        carbon_p_en=np.nextafter(-0.2, np.inf)
    )

    digests = {
        _assert_custom(baseline),
        _assert_custom(scalar_mutation),
        _assert_custom(shell_mutation),
    }
    assert len(digests) == 3


def test_pair_field_mutation_changes_identity() -> None:
    baseline = _compact_parameters()
    pair_mutation = _compact_parameters(
        cross_k_ab=np.nextafter(6.0, np.inf)
    )

    assert _assert_custom(baseline) != _assert_custom(pair_mutation)


def test_element_and_pair_insertion_order_is_canonical() -> None:
    forward = _compact_parameters(reverse=False)
    reverse = _compact_parameters(reverse=True)

    assert forward.content_sha256() == reverse.content_sha256()
    assert forward.parameter_identity() == reverse.parameter_identity()


def test_nonfinite_parameters_are_rejected() -> None:
    bad_element = _element(
        1,
        ((1, 0, -0.5, 1.2, 1.0, 0.0, 0.0),),
        gam=np.nan,
    )
    with pytest.raises(ValueError, match="must be finite"):
        _xtb.GFN2ParameterSet().add_element(bad_element)

    bad_pair = _pair(0.7, np.inf)
    with pytest.raises(ValueError, match="must be finite"):
        _xtb.GFN2ParameterSet().set_repulsive_pair(1, 1, bad_pair)


def test_molecular_result_is_bound_to_executed_parameter_snapshot(
    published_params: _xtb.GFN2ParameterSet,
) -> None:
    result = _xtb.run_gfn2_xtb(_h2(), published_params)

    assert result.converged
    assert result.parameter_sha256 == GFN2_PUBLISHED_SHA256
    assert result.parameter_identity == GFN2_PUBLISHED_IDENTITY

    _mutate_hydrogen_gam(published_params)
    assert result.parameter_sha256 == GFN2_PUBLISHED_SHA256
    assert result.parameter_identity == GFN2_PUBLISHED_IDENTITY


def test_model_cache_invalidates_after_parameter_mutation(
    published_params: _xtb.GFN2ParameterSet,
) -> None:
    model = GFN2Model(_h2(), published_params, warn=False)
    model.energy()
    original_result = model._last_result
    assert original_result.parameter_sha256 == GFN2_PUBLISHED_SHA256
    assert model.parameter_sha256 == GFN2_PUBLISHED_SHA256
    assert model.parameter_identity == GFN2_PUBLISHED_IDENTITY

    _mutate_hydrogen_gam(published_params)
    mutated_hash = published_params.content_sha256()
    model.energy()

    assert model._last_result is not original_result
    assert model._last_result.parameter_sha256 == mutated_hash
    assert model._last_result.parameter_identity == f"custom:{mutated_hash}"


def test_gradient_rejects_result_parameter_mismatch(
    published_params: _xtb.GFN2ParameterSet,
) -> None:
    molecule = _h2()
    result = _xtb.run_gfn2_xtb(molecule, published_params)
    gradient = np.asarray(
        _se.compute_gfn2_gradient(molecule, result, published_params)
    )
    assert gradient.shape == (2, 3)
    assert np.all(np.isfinite(gradient))

    _mutate_hydrogen_gam(published_params)
    with pytest.raises(ValueError, match="same immutable parameter snapshot"):
        _se.compute_gfn2_gradient(molecule, result, published_params)


def test_execution_provenance_freezes_every_scc_control_and_d4_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    params = _compact_parameters()
    molecule = _h2()
    model = GFN2Model(molecule, params, warn=False)

    requested = _xtb.XTBSccOptions()
    requested.max_iter = 17
    requested.conv_tol_charge = 2.5e-8
    requested.charge_mixing = 0.23
    requested.scc_mixer = _se.SCCMixer.Broyden
    requested.mixer_memory = 4
    requested.mixer_damping = 0.17
    requested.electronic_temperature = 0.012
    requested.auto_stabilize = False
    requested.aes_faithful = True
    requested.aes_damping = 0.43
    monkeypatch.setattr(model, "_make_scc_options", lambda: requested)

    d4_parameters = D4Params(
        s6=0.91,
        s8=2.71,
        a1=0.53,
        a2=5.01,
        s9=4.99,
        doi="test:d4-row",
    )
    d4_execution = _gfn2_method._GFN2D4Execution(
        reference_data=object(),
        functional="gfn2xtb-test",
        parameters=d4_parameters,
        refdata_path="<frozen-d4-refdata>",
        refdata_sha256="a" * 64,
        refdata_source_sha256="b" * 64,
    )
    monkeypatch.setattr(
        _gfn2_method,
        "_gfn2_d4_execution_snapshot",
        lambda: d4_execution,
    )

    source_sha256 = "c" * 64
    original_cache = (
        f'# source_sha256 = "{source_sha256}"\noriginal-cache\n'.encode()
    )
    cache_path = tmp_path / "gfn2.toml"
    cache_path.write_bytes(original_cache)
    monkeypatch.setattr(
        _gfn2_params,
        "gfn2_parameter_lineage",
        lambda actual_params: {
            "cache_path": str(cache_path),
            "cache_sha256": hashlib.sha256(original_cache).hexdigest(),
            "cache_source_sha256": source_sha256,
            "cache_source_url": _gfn2_params._XTB_PARAM_URL,
            "parameter_sha256": actual_params.content_sha256(),
        },
    )

    seen: dict[str, object] = {}

    def fake_run(mol, executed_params, executed_options):
        seen["molecule"] = mol
        seen["params"] = executed_params
        seen["options"] = executed_options
        seen["options_are_private"] = executed_options is not requested
        # Any result-time inspection would now observe a different artifact.
        cache_path.write_bytes(b"replaced-during-native-execution")
        return _fake_native_result(params)

    monkeypatch.setattr(_gfn2_method._xtb, "run_gfn2_xtb", fake_run)
    model._run_scc_at(molecule)

    def fail_post_execution_read(_path: Path) -> bytes:
        raise AssertionError("provenance performed a post-execution file read")

    monkeypatch.setattr(Path, "read_bytes", fail_post_execution_read)
    provenance = _gfn2_parameter_provenance(model)
    assert provenance is not None

    executed = seen["options"]
    assert seen["molecule"] is molecule
    assert seen["params"] is params
    assert seen["options_are_private"] is True
    expected_controls = {
        "gfn2_max_iter": 17,
        "gfn2_conv_tol_charge": 2.5e-8,
        "gfn2_charge_mixing": 0.23,
        "gfn2_scc_mixer": "broyden",
        "gfn2_mixer_memory": 4,
        "gfn2_mixer_damping": 0.17,
        "gfn2_electronic_temperature": 0.012,
        "gfn2_electronic_temperature_explicit": True,
        "gfn2_auto_stabilize": False,
        "gfn2_aes_faithful": True,
        "gfn2_aes_damping": 0.43,
    }
    assert {key: provenance[key] for key in expected_controls} == expected_controls
    assert int(executed.max_iter) == expected_controls["gfn2_max_iter"]
    assert float(executed.conv_tol_charge) == expected_controls[
        "gfn2_conv_tol_charge"
    ]

    assert provenance["gfn2_d4_functional"] == "gfn2xtb-test"
    assert provenance["gfn2_d4_s6"] == pytest.approx(d4_parameters.s6)
    assert provenance["gfn2_d4_s8"] == pytest.approx(d4_parameters.s8)
    assert provenance["gfn2_d4_s9"] == pytest.approx(d4_parameters.s9)
    assert provenance["gfn2_d4_a1"] == pytest.approx(d4_parameters.a1)
    assert provenance["gfn2_d4_a2"] == pytest.approx(d4_parameters.a2)
    assert provenance["gfn2_d4_doi"] == d4_parameters.doi
    assert provenance["gfn2_d4_refdata_path"] == "<frozen-d4-refdata>"
    assert provenance["gfn2_d4_refdata_sha256"] == "a" * 64
    assert provenance["gfn2_d4_refdata_source_sha256"] == "b" * 64
    assert not any(key.startswith("gfn2_parameter_d4_") for key in provenance)

    assert provenance["gfn2_cache_path"] == "<gfn2-parameter-cache>/gfn2.toml"
    assert provenance["gfn2_cache_sha256"] == hashlib.sha256(
        original_cache
    ).hexdigest()
    assert provenance["gfn2_cache_source_sha256"] == source_sha256


def test_d4_source_hash_is_bound_to_the_exact_parsed_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_bytes = _gfn2_method._gfn2_d4_refdata_bytes()
    source_path = tmp_path / "d4_reference_data.json"
    source_path.write_bytes(original_bytes)
    monkeypatch.setattr(
        _gfn2_method,
        "_gfn2_d4_refdata_path",
        lambda: source_path,
    )
    _gfn2_method._load_gfn2_d4_reference_data.cache_clear()
    try:
        first = _gfn2_method._gfn2_d4_execution_snapshot()
        source_path.write_bytes(b"replaced-after-parse")
        second = _gfn2_method._gfn2_d4_execution_snapshot()
    finally:
        _gfn2_method._load_gfn2_d4_reference_data.cache_clear()

    expected_source_sha256 = hashlib.sha256(original_bytes).hexdigest()
    assert first.refdata_source_sha256 == expected_source_sha256
    assert second.refdata_source_sha256 == expected_source_sha256
    assert first.refdata_sha256 == second.refdata_sha256


def test_cached_energy_reuses_the_frozen_d4_execution_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _compact_parameters()
    molecule = _h2()
    model = GFN2Model(molecule, params, warn=False)
    d4_execution = _gfn2_method._gfn2_d4_execution_snapshot()
    monkeypatch.setattr(
        _gfn2_method,
        "_gfn2_d4_execution_snapshot",
        lambda: d4_execution,
    )
    monkeypatch.setattr(
        _gfn2_method._xtb,
        "run_gfn2_xtb",
        lambda *_args: _fake_native_result(params),
    )

    seen: list[object] = []

    def fake_d4(_mol, _params, *, execution=None):
        seen.append(execution)
        return -0.01

    monkeypatch.setattr(_gfn2_method, "_compute_gfn2_d4", fake_d4)

    assert model.energy() == pytest.approx(-1.01)
    assert model.energy() == pytest.approx(-1.01)
    assert seen == [d4_execution, d4_execution]


def test_d4_execution_snapshot_detaches_mutable_cached_reference_data() -> None:
    source = _gfn2_method._load_gfn2_d4_reference_data()
    execution = _gfn2_method._gfn2_d4_execution_snapshot()
    atomic_number = execution.reference_data.supported_z[0]
    baseline = execution.reference_data.get_cns(atomic_number)[0]
    source_value = source.elements[atomic_number].cns[0]

    try:
        source.elements[atomic_number].cns[0] = float(source_value) + 1.0
        assert execution.reference_data.get_cns(atomic_number)[0] == baseline
        assert (
            _gfn2_method._gfn2_d4_reference_content_sha256(
                execution.reference_data
            )
            == execution.refdata_sha256
        )
    finally:
        source.elements[atomic_number].cns[0] = source_value

    with pytest.raises(TypeError):
        execution.reference_data._cns[atomic_number] = (baseline + 1.0,)


def test_captured_d4_row_determines_energy_after_global_row_mutation(
    published_params: _xtb.GFN2ParameterSet,
) -> None:
    from vibeqc import dispersion_d4_parameters as _d4_parameters

    molecule = _h2()
    execution = _gfn2_method._gfn2_d4_execution_snapshot()
    baseline = _gfn2_method._compute_gfn2_d4(
        molecule,
        published_params,
        execution=execution,
    )
    original = _d4_parameters._PARAMS["gfn2xtb"]
    mutated = original._replace(s8=float(original.s8) + 50.0)
    try:
        _d4_parameters._PARAMS["gfn2xtb"] = mutated
        frozen_energy = _gfn2_method._compute_gfn2_d4(
            molecule,
            published_params,
            execution=execution,
        )
        mutated_execution = _gfn2_method._gfn2_d4_execution_snapshot()
        mutated_energy = _gfn2_method._compute_gfn2_d4(
            molecule,
            published_params,
            execution=mutated_execution,
        )
    finally:
        _d4_parameters._PARAMS["gfn2xtb"] = original

    assert frozen_energy == pytest.approx(baseline, abs=0.0)
    assert mutated_energy != pytest.approx(baseline, rel=1e-8, abs=1e-12)
