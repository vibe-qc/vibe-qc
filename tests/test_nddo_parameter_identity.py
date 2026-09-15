"""Canonical parameter identity regressions for PM6, PM7 and OM1/OM2/OM3."""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest
from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import PeriodicSystem
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical.methods import omx_params as _omx_params
from vibeqc.semiempirical.methods import pm6_params as _pm6_params
from vibeqc.semiempirical.methods import pm7_params as _pm7_params
from vibeqc.semiempirical.methods.omx import OMxModel
from vibeqc.semiempirical.methods.omx_params import (
    load_om1_params,
    load_om2_params,
    load_om3_params,
)
from vibeqc.semiempirical.methods.periodic_pm6 import PeriodicPM6Model
from vibeqc.semiempirical.methods.pm6 import PM6Model, UPM6Model
from vibeqc.semiempirical.methods.pm6_params import (
    load_pm6_mopac_params,
    load_pm6_params,
)
from vibeqc.semiempirical.methods.pm7_params import (
    load_pm7_mopac_params,
    load_pm7_params,
)
from vibeqc.semiempirical.runner import run_semiempirical
from vibeqc.semiempirical.seccm import (
    bind_finite_group,
    build_seccm_topology,
)
from vibeqc.semiempirical.seccm.omx import run_omx_seccm
from vibeqc.semiempirical.seccm.pm6 import run_pm6_seccm

PM6_INLINE_SHA256 = (
    "b29a0554f7c311eea66dfa5b2368ccf75a20a7ae78bed548ef685a1ec43d4273"
)
PM6_FULL_SHA256 = (
    "105bb194eff3f4902758ea642839b3f296839ea87a9ccb8217b2ffe53c1e2d57"
)
# #440 (maintainer decision 2026-09-06): the inline PM7 pair table shipped
# between d333f5adc and that date carried 28 alpb/xfac values of unrecorded
# origin, matching neither MOPAC nor the bundled cache. They were replaced by
# MOPAC's parameters_for_PM7_C.F90 values, so the inline digest below is NOT
# the pre-fix loader's; the full-cache digest is unchanged. Both PM7 sets are
# MOPAC-derived and Apache-2.0, like PM6, and the DOI names the 2013 method
# paper. The PM7 Hamiltonian itself stays gated; these are provenance pins.
PM7_INLINE_SHA256 = (
    "7e51f94bdc4ff7a0bd5b086ff390b14b97c1bf97cdc54ec6ab8e3d36f2525191"
)
PM7_FULL_SHA256 = (
    "ee028a756f7bc26a559796c97dad0f2b2f090aceb25de90403989e3d090b1359"
)
OM1_SHA256 = "be0bb9541df6628dd399236d27c0591417da1e4b5469c32aa7347b89845436e1"
OM2_SHA256 = "9c73ec5520b0736b6f6215e8debbbd845a4c4edce1ee56eeb0e25f712a2096f7"
OM3_SHA256 = "26a937d0ada13634eb05487279a958ad8f2ddc7f5e91306bc2cc0a07e787575d"

# #440: the inline five-element set is MOPAC-derived, not a transcription of
# Stewart's published tables -- all 118 of its values (88 element + 30
# diatomic) reproduce MOPAC's Apache-2.0 parameters_for_PM6_C.F90 exactly at
# its six published decimals, and none of them appears in Stewart 2007. The
# identity and the reported license both have to say so; the content hash is
# unchanged, because no number moved.
PM6_INLINE_IDENTITY = "published:pm6-mopac-inline-v1"
PM6_FULL_IDENTITY = "published:pm6-mopac-full-v1"
PM7_INLINE_IDENTITY = "published:pm7-mopac-inline-v1"
PM7_FULL_IDENTITY = "published:pm7-mopac-full-v1"
OM1_IDENTITY = "published:om1-dral-2016-v1"
OM2_IDENTITY = "published:om2-dral-2016-v1"
OM3_IDENTITY = "published:om3-dral-2016-v1"

_PM6_DOI = "10.1007/s00894-007-0233-4"
_PM7_DOI = "10.1007/s00894-012-1667-x"
_OMX_DOI = "10.1021/acs.jctc.5b01046"


@pytest.mark.parametrize(
    (
        "loader",
        "sha256",
        "identity",
        "n_elements",
        "license_name",
        "doi",
    ),
    [
        (
            load_pm6_params,
            PM6_INLINE_SHA256,
            PM6_INLINE_IDENTITY,
            5,
            "Apache-2.0",
            _PM6_DOI,
        ),
        (
            load_pm6_mopac_params,
            PM6_FULL_SHA256,
            PM6_FULL_IDENTITY,
            75,
            "Apache-2.0",
            _PM6_DOI,
        ),
        (
            load_pm7_params,
            PM7_INLINE_SHA256,
            PM7_INLINE_IDENTITY,
            5,
            "Apache-2.0",
            _PM7_DOI,
        ),
        (
            load_pm7_mopac_params,
            PM7_FULL_SHA256,
            PM7_FULL_IDENTITY,
            75,
            "Apache-2.0",
            _PM7_DOI,
        ),
        (
            load_om1_params,
            OM1_SHA256,
            OM1_IDENTITY,
            5,
            "published parameter table",
            _OMX_DOI,
        ),
        (
            load_om2_params,
            OM2_SHA256,
            OM2_IDENTITY,
            5,
            "published parameter table",
            _OMX_DOI,
        ),
        (
            load_om3_params,
            OM3_SHA256,
            OM3_IDENTITY,
            5,
            "published parameter table",
            _OMX_DOI,
        ),
    ],
    ids=["pm6-inline", "pm6-full", "pm7-inline", "pm7-full", "om1", "om2", "om3"],
)
def test_published_nddo_loaders_have_exact_content_identity(
    loader,
    sha256: str,
    identity: str,
    n_elements: int,
    license_name: str,
    doi: str,
) -> None:
    params = loader()
    metadata = params.metadata()

    assert params.content_sha256() == sha256
    assert params.parameter_identity() == identity
    assert metadata.parameter_hash == sha256
    assert metadata.origin == "published"
    assert metadata.license == license_name
    assert metadata.doi_or_url == doi
    assert metadata.n_elements == n_elements
    assert list(metadata.element_list) == sorted(metadata.element_list)
    assert len(metadata.element_list) == n_elements


def _assert_custom(params) -> str:
    sha256 = params.content_sha256()
    identity = params.parameter_identity()
    metadata = params.metadata()
    assert identity == f"custom:{sha256}"
    assert metadata.parameter_hash == sha256
    assert metadata.origin == identity
    assert metadata.license == ""
    assert metadata.doi_or_url == ""
    return sha256


def test_pm6_element_pair_and_term_mutations_are_custom_and_restore() -> None:
    params = load_pm6_params()
    original_carbon = params.element_data(6)

    changed_carbon = params.element_data(6)
    changed_carbon.uss = math.nextafter(changed_carbon.uss, math.inf)
    params.add_element(changed_carbon)
    assert _assert_custom(params) != PM6_INLINE_SHA256
    params.add_element(original_carbon)
    assert params.parameter_identity() == PM6_INLINE_IDENTITY

    d1, d2 = _pm6_params._PM6_MOPAC_DIATOMIC[(1, 6)]
    params.add_diatomic(1, 6, d1, math.nextafter(d2, math.inf))
    assert _assert_custom(params) != PM6_INLINE_SHA256
    params.add_diatomic(6, 1, d1, d2)
    assert params.parameter_identity() == PM6_INLINE_IDENTITY

    changed_carbon = params.element_data(6)
    changed_carbon.add_gaussian_term(0.125, 2.5, 1.25)
    params.add_element(changed_carbon)
    assert _assert_custom(params) != PM6_INLINE_SHA256
    params.add_element(original_carbon)
    assert params.content_sha256() == PM6_INLINE_SHA256
    assert params.parameter_identity() == PM6_INLINE_IDENTITY


def test_om2_scalar_mutation_is_custom_and_restored_content_is_published() -> None:
    raw = _omx_params._make_om2_params()
    original_f1 = raw[6]["F1"]
    raw[6]["F1"] = math.nextafter(original_f1, math.inf)
    changed = _omx_params._build_params(raw, _se.nddo.OMxVariant.OM2)
    assert _assert_custom(changed) != OM2_SHA256

    raw[6]["F1"] = original_f1
    restored = _omx_params._build_params(raw, _se.nddo.OMxVariant.OM2)
    assert restored.content_sha256() == OM2_SHA256
    assert restored.parameter_identity() == OM2_IDENTITY


def test_omx_element_getter_is_detached_and_add_resynchronizes_base_view() -> None:
    params = load_om2_params()
    baseline_sha256 = params.content_sha256()
    baseline_uss = float(params.element_data(6).uss)
    carbon = params.omx_element_data(6)
    carbon.uss = math.nextafter(carbon.uss, math.inf)

    assert params.content_sha256() == baseline_sha256
    assert float(params.element_data(6).uss) == baseline_uss

    params.add_omx_element(carbon)
    assert params.content_sha256() != baseline_sha256
    assert float(params.element_data(6).uss) == float(carbon.uss)
    _assert_custom(params)


def _pm7_cache():
    import tomllib
    from pathlib import Path

    path = Path(_pm7_params.__file__).with_name("pm7_mopac_params.toml")
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def test_pm7_inline_tables_equal_the_bundled_mopac_cache() -> None:
    """Every inline PM7 number is the bundled cache's number (#440).

    The cache is parsed from MOPAC's parameters_for_PM7_C.F90 and carries
    that provenance in its header; the inline five-element set claims the
    same source. Pinning the two tables against each other (values, not
    prose) is what keeps the claim true: the 28 pair values of unrecorded
    origin that shipped before 2026-09-06 would have failed this test.
    """
    cache = _pm7_cache()
    elements = {int(e["Z"]): e for e in cache["element"]}
    for z, inline in _pm7_params._PM7_MOPAC_CORE.items():
        cached = elements[z]
        for key, value in inline.items():
            assert cached[key] == value, (z, key, value, cached[key])

    pairs = {
        (int(d["Z1"]), int(d["Z2"])): (float(d["alpb"]), float(d["xfac"]))
        for d in cache["diatomic"]
    }
    assert len(_pm7_params._PM7_MOPAC_DIATOMIC) == 15
    for key, value in _pm7_params._PM7_MOPAC_DIATOMIC.items():
        assert pairs[key] == value, (key, value, pairs[key])


def test_pm7_pair_mutation_is_custom_and_restore() -> None:
    params = load_pm7_params()
    assert params.parameter_identity() == PM7_INLINE_IDENTITY

    d1, d2 = _pm7_params._PM7_MOPAC_DIATOMIC[(1, 6)]
    params.add_diatomic(1, 6, d1, math.nextafter(d2, math.inf))
    assert _assert_custom(params) != PM7_INLINE_SHA256
    params.add_diatomic(6, 1, d1, d2)
    assert params.content_sha256() == PM7_INLINE_SHA256
    assert params.parameter_identity() == PM7_INLINE_IDENTITY
    assert params.metadata().license == "Apache-2.0"


def _ordered_pm6(*, reverse: bool):
    params = _se.nddo.PM6ParameterSet()
    elements = list(_pm6_params._PM6_MOPAC_CORE.items())
    pairs = list(_pm6_params._PM6_MOPAC_DIATOMIC.items())
    if reverse:
        elements.reverse()
        pairs.reverse()

    for atomic_number, values in elements:
        element = _se.nddo.NDDOElementData()
        element.Z = atomic_number
        _pm6_params._populate_element(element, values)
        params.add_element(element)
    for (z1, z2), (d1, d2) in pairs:
        if reverse:
            z1, z2 = z2, z1
        params.add_diatomic(z1, z2, d1, d2)
    return params


def test_pm6_element_pair_and_pair_orientation_order_are_canonical() -> None:
    forward = _ordered_pm6(reverse=False)
    reverse = _ordered_pm6(reverse=True)

    assert forward.content_sha256() == reverse.content_sha256()
    assert forward.content_sha256() == PM6_INLINE_SHA256
    assert forward.parameter_identity() == reverse.parameter_identity()
    assert forward.parameter_identity() == PM6_INLINE_IDENTITY


@pytest.mark.parametrize(
    ("maker", "variant", "sha256", "identity"),
    [
        (_omx_params._make_om1_params, _se.nddo.OMxVariant.OM1, OM1_SHA256, OM1_IDENTITY),
        (_omx_params._make_om2_params, _se.nddo.OMxVariant.OM2, OM2_SHA256, OM2_IDENTITY),
        (_omx_params._make_om3_params, _se.nddo.OMxVariant.OM3, OM3_SHA256, OM3_IDENTITY),
    ],
    ids=["om1", "om2", "om3"],
)
def test_omx_element_construction_order_is_canonical(
    maker,
    variant,
    sha256: str,
    identity: str,
) -> None:
    raw = maker()
    forward = _omx_params._build_params(raw, variant)
    reverse = _omx_params._build_params(
        dict(reversed(tuple(raw.items()))), variant
    )

    assert forward.content_sha256() == reverse.content_sha256() == sha256
    assert forward.parameter_identity() == reverse.parameter_identity() == identity


def _h2() -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])],
        charge=0,
        multiplicity=1,
    )


def _triplet_h2() -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])],
        charge=0,
        multiplicity=3,
    )


def _nitrogen_quartet() -> Molecule:
    return Molecule(
        [Atom(7, [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=4,
    )


def _periodic_h2_chain() -> PeriodicSystem:
    return PeriodicSystem(
        1,
        np.diag([4.0, 20.0, 20.0]),
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])],
        0,
        1,
    )


def test_pm6_native_result_binds_snapshot_and_gradient_rejects_mismatch() -> None:
    molecule = _h2()
    params = load_pm6_params()
    result = _se.nddo.run_pm6(molecule, params)

    assert result.parameter_sha256 == PM6_INLINE_SHA256
    assert result.parameter_identity == PM6_INLINE_IDENTITY
    gradient = np.asarray(
        _se.nddo.compute_pm6_gradient_fd_from_result(molecule, params, result)
    )
    assert gradient.shape == (2, 3)
    assert np.all(np.isfinite(gradient))

    changed_hydrogen = params.element_data(1)
    changed_hydrogen.uss = math.nextafter(changed_hydrogen.uss, math.inf)
    params.add_element(changed_hydrogen)
    _assert_custom(params)
    with pytest.raises(ValueError, match="same immutable parameter snapshot"):
        _se.nddo.compute_pm6_gradient_fd_from_result(molecule, params, result)


def test_om2_concurrent_mutation_yields_only_complete_parameter_snapshots() -> None:
    params = load_om2_params()
    original = params.omx_element_data(6)
    changed = params.omx_element_data(6)
    changed.F1 = math.nextafter(changed.F1, math.inf)
    published_sha256 = params.content_sha256()
    params.add_omx_element(changed)
    custom_sha256 = params.content_sha256()
    params.add_omx_element(original)
    expected_identities = {
        published_sha256: OM2_IDENTITY,
        custom_sha256: f"custom:{custom_sha256}",
    }

    started = threading.Event()
    stop = threading.Event()
    mutation_count = 0

    def mutate() -> None:
        nonlocal mutation_count
        started.set()
        while not stop.is_set():
            params.add_omx_element(changed)
            params.add_omx_element(original)
            mutation_count += 2

    worker = threading.Thread(target=mutate)
    worker.start()
    assert started.wait(timeout=1.0)
    results = []
    try:
        results = [_se.nddo.run_omx_v2(_h2(), params) for _ in range(6)]
    finally:
        stop.set()
        worker.join(timeout=5.0)

    assert not worker.is_alive()
    assert mutation_count > 0
    for result in results:
        assert result.parameter_sha256 in expected_identities
        assert (
            result.parameter_identity
            == expected_identities[result.parameter_sha256]
        )


@pytest.mark.parametrize(
    ("model_factory", "mutate"),
    [
        (
            lambda params: PM6Model(_h2(), params=params),
            lambda params: _mutate_pm6_hydrogen(params),
        ),
        (
            lambda params: UPM6Model(_triplet_h2(), params=params),
            lambda params: _mutate_pm6_hydrogen(params),
        ),
    ],
    ids=["pm6", "upm6"],
)
def test_pm6_model_gradient_rejects_post_energy_parameter_mutation(
    model_factory,
    mutate,
) -> None:
    params = load_pm6_params()
    model = model_factory(params)
    assert math.isfinite(model.energy())
    gradient = np.asarray(model.gradient())
    assert gradient.shape == (2, 3)
    assert np.all(np.isfinite(gradient))

    mutate(params)
    with pytest.raises(ValueError, match="same immutable parameter snapshot"):
        model.gradient()


@pytest.mark.parametrize(
    "molecule",
    [_h2(), _nitrogen_quartet()],
    ids=["om2", "uom2"],
)
def test_om2_model_gradient_rejects_post_energy_parameter_mutation(
    molecule: Molecule,
) -> None:
    params = load_om2_params()
    model = OMxModel(molecule, variant="om2", params=params)
    assert math.isfinite(model.energy())
    gradient = np.asarray(model.gradient())
    assert gradient.shape == (len(molecule.atoms), 3)
    assert np.all(np.isfinite(gradient))

    _mutate_om2_carbon(params)
    _assert_custom(params)
    with pytest.raises(ValueError, match="same immutable parameter snapshot"):
        model.gradient()


def _mutate_pm6_hydrogen(params) -> None:
    hydrogen = params.element_data(1)
    hydrogen.uss = math.nextafter(hydrogen.uss, math.inf)
    params.add_element(hydrogen)


def _mutate_om2_carbon(params) -> None:
    carbon = params.omx_element_data(6)
    carbon.F1 = math.nextafter(carbon.F1, math.inf)
    params.add_omx_element(carbon)


def test_periodic_pm6_model_retains_identity_and_rejects_mutated_derivative() -> None:
    params = load_pm6_params()
    model = PeriodicPM6Model(
        _periodic_h2_chain(),
        params=params,
        cutoff_bohr=5.0,
    )
    assert model.parameter_identity is None
    assert model.parameter_sha256 is None
    assert math.isfinite(model.energy())
    assert model.parameter_identity == PM6_INLINE_IDENTITY
    assert model.parameter_sha256 == PM6_INLINE_SHA256

    _mutate_pm6_hydrogen(params)
    with pytest.raises(RuntimeError, match="same immutable parameter snapshot"):
        model.gradient()


def test_generic_upm6_lazy_gradient_rejects_parameter_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = load_pm6_params()
    monkeypatch.setattr(
        _pm6_params,
        "load_pm6_params_auto",
        lambda _atomic_numbers: params,
    )
    result = run_semiempirical("pm6", _triplet_h2())
    _mutate_pm6_hydrogen(params)

    with pytest.raises(ValueError, match="same immutable parameter snapshot"):
        result.gradient()


@pytest.mark.parametrize(
    "molecule",
    [_h2(), _nitrogen_quartet()],
    ids=["om2", "uom2"],
)
def test_generic_om2_lazy_gradient_rejects_parameter_mutation(
    monkeypatch: pytest.MonkeyPatch,
    molecule: Molecule,
) -> None:
    params = load_om2_params()
    monkeypatch.setattr(_omx_params, "load_om2_params", lambda: params)
    result = run_semiempirical("om2", molecule)
    _mutate_om2_carbon(params)

    with pytest.raises(ValueError, match="same immutable parameter snapshot"):
        result.gradient()


@pytest.mark.parametrize(
    ("method", "sha256", "identity"),
    [
        ("pm6", PM6_INLINE_SHA256, PM6_INLINE_IDENTITY),
        ("om2", OM2_SHA256, OM2_IDENTITY),
    ],
)
def test_molecular_runner_projects_exact_native_parameter_identity(
    method: str,
    sha256: str,
    identity: str,
) -> None:
    result = run_semiempirical(method, _h2())

    assert result.parameter_sha256 == sha256
    assert result.parameter_identity == identity
    assert result.parameter_provenance == {
        "parameter_identity": identity,
        "parameter_sha256": sha256,
    }


@pytest.mark.parametrize(
    ("model_factory", "sha256", "identity"),
    [
        (
            lambda: PM6Model(_h2()),
            PM6_INLINE_SHA256,
            PM6_INLINE_IDENTITY,
        ),
        (
            lambda: UPM6Model(_triplet_h2()),
            PM6_INLINE_SHA256,
            PM6_INLINE_IDENTITY,
        ),
        (
            lambda: OMxModel(_h2(), variant="om2"),
            OM2_SHA256,
            OM2_IDENTITY,
        ),
        (
            lambda: OMxModel(_nitrogen_quartet(), variant="om2"),
            OM2_SHA256,
            OM2_IDENTITY,
        ),
    ],
    ids=["pm6", "upm6", "om2", "uom2"],
)
def test_nddo_models_retain_exact_native_parameter_identity(
    model_factory,
    sha256: str,
    identity: str,
) -> None:
    model = model_factory()

    assert model.parameter_sha256 is None
    assert model.parameter_identity is None
    assert math.isfinite(model.energy())
    assert model._last_result is not None
    assert model.parameter_sha256 == sha256
    assert model.parameter_identity == identity
    assert model.parameter_sha256 == str(model._last_result.parameter_sha256)
    assert model.parameter_identity == str(model._last_result.parameter_identity)


@pytest.mark.parametrize(
    ("params_factory", "model_factory"),
    [
        (
            lambda: _changed_pm6_params(),
            lambda params: PM6Model(_h2(), params=params),
        ),
        (
            lambda: _changed_om2_params(),
            lambda params: OMxModel(_h2(), variant="om2", params=params),
        ),
    ],
    ids=["pm6", "om2"],
)
def test_nddo_models_project_custom_native_parameter_identity(
    params_factory,
    model_factory,
) -> None:
    params = params_factory()
    sha256 = _assert_custom(params)
    model = model_factory(params)

    assert math.isfinite(model.energy())
    assert model.parameter_sha256 == sha256
    assert model.parameter_identity == f"custom:{sha256}"


def _changed_pm6_params():
    params = load_pm6_params()
    hydrogen = params.element_data(1)
    hydrogen.uss = math.nextafter(hydrogen.uss, math.inf)
    params.add_element(hydrogen)
    return params


def _changed_om2_params():
    raw = _omx_params._make_om2_params()
    raw[6]["F1"] = math.nextafter(raw[6]["F1"], math.inf)
    return _omx_params._build_params(raw, _se.nddo.OMxVariant.OM2)


def _molecular_limit_topology() -> tuple[Molecule, object]:
    coordinates = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    translation = np.array([40.0, 0.0, 0.0])
    topology = build_seccm_topology(
        coordinates,
        [translation],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[translation],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return _h2(), topology


def test_pm6_and_omx_seccm_project_exact_native_parameter_identity() -> None:
    molecule, topology = _molecular_limit_topology()

    pm6 = run_pm6_seccm(molecule, topology)
    assert pm6.parameter_sha256 == PM6_INLINE_SHA256
    assert pm6.parameter_identity == PM6_INLINE_IDENTITY

    for variant, sha256, identity in (
        ("om2", OM2_SHA256, OM2_IDENTITY),
        ("om3", OM3_SHA256, OM3_IDENTITY),
    ):
        result = run_omx_seccm(molecule, topology, variant=variant)
        assert result.parameter_sha256 == sha256
        assert result.parameter_identity == identity


def test_pm6_seccm_gradient_rejects_loader_snapshot_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    molecule, topology = _molecular_limit_topology()
    published = load_pm6_params()
    changed = _changed_pm6_params()
    calls = 0

    def alternating_loader(_atomic_numbers):
        nonlocal calls
        calls += 1
        return published if calls == 1 else changed

    monkeypatch.setattr(
        _pm6_params,
        "load_pm6_params_auto",
        alternating_loader,
    )
    with pytest.raises(
        RuntimeError,
        match="different immutable parameter snapshots",
    ):
        run_pm6_seccm(
            molecule,
            topology,
            compute_gradient=True,
        )


def test_bundled_mopac_caches_cite_their_own_originating_publication():
    """IID 271: each generated cache carries ITS OWN paper, not a sibling's.

    The defect was that the PM6 cache cited the PM7 DOI. It was repaired by
    deleting the wrong reference rather than correcting it, which left the
    PM6 cache with no originating publication at all while the PM7 cache
    kept one -- so a user auditing where the PM6 numbers came from got
    nothing. CLAUDE.md sec. 8 requires the per-publication reference in the
    file header, and this pins both caches so neither can regress into the
    other's citation or into silence.
    """
    import tomllib
    from pathlib import Path

    methods = Path(__file__).parents[1] / "python" / "vibeqc" / "semiempirical" / "methods"
    # Match on the publication year rather than the DOI: the two headers use
    # different citation formats (PM6 carries a DOI, PM7 a bare citation
    # string), and the year is what actually separates Stewart's PM6 paper
    # (2007) from his PM7 one (2013).
    expected = {
        "pm6_mopac_params.toml": ("2007", "2013"),
        "pm7_mopac_params.toml": ("2013", "2007"),
    }
    for name, (own_year, sibling_year) in expected.items():
        path = methods / name
        assert path.exists(), f"{name} is not present"
        meta = tomllib.loads(path.read_text())["metadata"]
        reference = meta.get("reference", "")
        assert reference, f"{name} carries no originating publication"
        assert own_year in reference, (
            f"{name} reference does not cite its own paper: {reference!r}"
        )
        assert sibling_year not in reference, (
            f"{name} cites the sibling method's paper: {reference!r}"
        )


@pytest.mark.parametrize(
    "name",
    [
        "pm6_params.py",
        "pm6_mopac_params.toml",
        "pm7_params.py",
        "pm7_mopac_params.toml",
    ],
)
def test_every_mopac_parameter_carrier_retains_full_notice(name: str) -> None:
    """Issue #522: Apache-2.0 attribution travels with every value file."""
    from pathlib import Path

    methods = (
        Path(__file__).parents[1] / "python" / "vibeqc" / "semiempirical" / "methods"
    )
    text = (methods / name).read_text(encoding="utf-8")
    notice = (
        "Molecular Orbital PACkage (MOPAC)",
        "Copyright 2021 Virginia Polytechnic Institute and State University",
        "Licensed under the Apache License, Version 2.0",
        "https://github.com/openmopac/mopac",
    )

    for notice_line in notice:
        assert notice_line in text, f"{name} is missing {notice_line!r}"
