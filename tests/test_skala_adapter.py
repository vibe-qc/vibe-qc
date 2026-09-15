from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.skala as skala
from vibeqc import _vibeqc_core as core


def _payload() -> dict[str, Any]:
    return {
        "functional": "SKALA",
        "grid_profile": "pyscf-level3",
        "points": np.array(
            [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [1.0, 1.1, 1.2]],
            dtype=np.float64,
        ),
        "grid_weights": np.array([0.2, 0.3, 0.5]),
        "atomic_grid_weights": np.array([0.4, 0.6, 1.0]),
        "atom_coords": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        "atomic_grid_sizes": [2, 1],
        "rho_alpha": np.array([0.1, 0.2, 0.3]),
        "rho_beta": np.array([0.05, 0.1, 0.15]),
        "grad_alpha": np.arange(9, dtype=np.float64).reshape(3, 3) / 10,
        "grad_beta": np.arange(9, 18, dtype=np.float64).reshape(3, 3) / 10,
        "tau_alpha": np.array([0.7, 0.8, 0.9]),
        "tau_beta": np.array([0.4, 0.5, 0.6]),
    }


class _FakeModel:
    def __init__(self) -> None:
        self.eval_called = False

    def get_exc(self, features: Any) -> Any:  # pragma: no cover - loader only
        raise AssertionError("loader test must not evaluate the model")

    def eval(self) -> None:
        self.eval_called = True


class _FakeJit:
    def __init__(self, *, extras: dict[str, bytes] | None = None) -> None:
        self.calls = 0
        self.loaded_bytes: bytes | None = None
        self.map_location: str | None = None
        self.model = _FakeModel()
        self.extras = extras or {
            "metadata": json.dumps({"name": "SKALA-1.1"}).encode(),
            "features": json.dumps(list(skala.MODEL_FEATURES)).encode(),
            "expected_d3_settings": json.dumps("b3lyp5").encode(),
            "protocol_version": json.dumps(skala.MODEL_PROTOCOL_VERSION).encode(),
        }

    def load(
        self,
        stream: Any,
        *,
        _extra_files: dict[str, bytes],
        map_location: str,
    ) -> _FakeModel:
        self.calls += 1
        self.loaded_bytes = stream.read()
        self.map_location = map_location
        _extra_files.update(self.extras)
        return self.model


class _FakeTorch:
    def __init__(self, *, extras: dict[str, bytes] | None = None) -> None:
        self.jit = _FakeJit(extras=extras)


def test_published_model_identity_is_fully_pinned() -> None:
    assert skala.CANONICAL_FUNCTIONAL == "skala-1.1"
    assert "skala" in skala.FUNCTIONAL_ALIASES
    assert skala.MODEL_REVISION == "99b5ed87e5f69d9216e1f9e30148b922eaea1241"
    assert skala.MODEL_FILENAME == "skala-1.1-rev1.fun"
    assert skala.MODEL_SIZE_BYTES == 2_388_988
    assert skala.MODEL_SHA256 == (
        "7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd"
    )
    assert skala.SOURCE_REVISION == "4f3f072b820183d4c77b47bcea6df9736655d12a"
    assert skala.MODEL_URL.endswith(
        f"/{skala.MODEL_REVISION}/{skala.MODEL_FILENAME}"
    )
    assert skala.MODEL_FEATURES == (
        "density",
        "kin",
        "grad",
        "grid_coords",
        "grid_weights",
        "atomic_grid_weights",
        "atomic_grid_sizes",
        "coarse_0_atomic_coords",
        "atomic_grid_size_bound_shape",
    )
    assert skala.MODEL_EXPECTED_D3_SETTINGS == "b3lyp5"
    assert skala.MODEL_FIRST_DERIVATIVE_BYTES_PER_POINT == 6680 * 8
    assert skala.DEFAULT_MAX_MODEL_POINTS_PER_CHUNK == 8192


def test_cache_path_honors_xdg_and_keeps_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    expected_root = tmp_path / "vibeqc" / "skala"
    assert skala.skala_cache_root() == expected_root
    assert skala.model_cache_path() == (
        expected_root
        / "microsoft"
        / "skala-1.1"
        / skala.MODEL_REVISION
        / skala.MODEL_FILENAME
    )


def test_download_is_verified_cached_and_records_scoped_source_notice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = b"small fake torchscript archive"
    digest = hashlib.sha256(artifact).hexdigest()
    monkeypatch.setattr(skala, "MODEL_SHA256", digest)
    monkeypatch.setattr(skala, "MODEL_SIZE_BYTES", len(artifact))
    calls: list[str] = []

    def download(url: str) -> bytes:
        calls.append(url)
        return artifact

    path = skala.ensure_skala_model(tmp_path, downloader=download)
    assert path == skala.model_cache_path(tmp_path)
    assert path.read_bytes() == artifact
    assert calls == [skala.MODEL_URL]

    provenance = json.loads((path.parent / "provenance.json").read_text())
    assert provenance["revision"] == skala.MODEL_REVISION
    assert provenance["sha256"] == digest
    assert provenance["url"] == skala.MODEL_URL
    assert provenance["expected_size_bytes"] == len(artifact)
    assert provenance["model_card_license_metadata"] == "MIT"
    assert provenance["source_revision"] == skala.SOURCE_REVISION
    assert "not checkpoint redistribution clearance" in provenance[
        "source_notice_scope"
    ]
    source_notice = path.parent / skala.SOURCE_NOTICE_FILENAME
    assert source_notice.read_text() == skala.SOURCE_LICENSE_NOTICE_TEXT
    assert "Copyright (c) Microsoft Corporation." in (
        skala.SOURCE_LICENSE_NOTICE_TEXT
    )

    def unexpected_download(url: str) -> bytes:
        raise AssertionError(f"unexpected second download from {url}")

    assert skala.ensure_skala_model(tmp_path, downloader=unexpected_download) == path


def test_network_download_is_size_bounded_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(skala, "MODEL_SIZE_BYTES", 4)

    class Response:
        def __init__(self) -> None:
            self.data = b"12345"
            self.offset = 0
            self.read_sizes: list[int] = []

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            self.read_sizes.append(size)
            chunk = self.data[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    response = Response()
    monkeypatch.setattr(
        skala.urllib.request,
        "urlopen",
        lambda request, timeout: response,
    )

    with pytest.raises(
        skala.SkalaModelIntegrityError,
        match="expected 4 bytes, got 5.*not hashed or cached",
    ):
        skala._download_bytes("https://example.invalid/model.fun")

    assert response.read_sizes == [5]


def test_custom_downloader_wrong_size_is_rejected_before_hashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(skala, "MODEL_SIZE_BYTES", 4)

    def unexpected_hash(data: bytes) -> str:
        raise AssertionError("wrong-size bytes must not be hashed")

    monkeypatch.setattr(skala, "_sha256", unexpected_hash)

    with pytest.raises(
        skala.SkalaModelIntegrityError,
        match="expected 4 bytes, got 5.*cache was not updated",
    ):
        skala.ensure_skala_model(tmp_path, downloader=lambda url: b"12345")


def test_bad_download_never_replaces_corrupt_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wanted = b"wanted model"
    monkeypatch.setattr(skala, "MODEL_SHA256", hashlib.sha256(wanted).hexdigest())
    monkeypatch.setattr(skala, "MODEL_SIZE_BYTES", len(wanted))
    path = skala.model_cache_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"previous corrupt bytes")

    with pytest.raises(skala.SkalaModelIntegrityError, match="cache was not updated"):
        skala.ensure_skala_model(tmp_path, downloader=lambda url: b"wrong download")

    assert path.read_bytes() == b"previous corrupt bytes"
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_load_hashes_exact_bytes_before_torchscript_deserialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = b"verified fake model"
    monkeypatch.setattr(skala, "MODEL_SHA256", hashlib.sha256(artifact).hexdigest())
    monkeypatch.setattr(skala, "MODEL_SIZE_BYTES", len(artifact))
    path = tmp_path / "model.fun"
    path.write_bytes(artifact)
    fake_torch = _FakeTorch()

    loaded = skala.load_skala_model(model_path=path, torch_module=fake_torch)

    assert fake_torch.jit.calls == 1
    assert fake_torch.jit.loaded_bytes == artifact
    assert fake_torch.jit.map_location == "cpu"
    assert fake_torch.jit.model.eval_called
    assert loaded.model is fake_torch.jit.model
    assert loaded.features == skala.MODEL_FEATURES
    assert loaded.expected_d3_settings == "b3lyp5"
    assert loaded.metadata == {"name": "SKALA-1.1"}

    path.write_bytes(b"changed after first load")
    with pytest.raises(skala.SkalaModelIntegrityError, match="not deserialized"):
        skala.load_skala_model(model_path=path, torch_module=fake_torch)
    assert fake_torch.jit.calls == 1


@pytest.mark.parametrize(
    ("extra_name", "value", "message"),
    [
        (
            "features",
            json.dumps(["density", "grid_weights"]).encode(),
            "feature contract mismatch",
        ),
        ("protocol_version", b"3", "Unsupported SKALA protocol 3"),
        ("expected_d3_settings", b'"b3lyp"', "D3 contract mismatch"),
    ],
)
def test_load_rejects_wrong_protocol_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_name: str,
    value: bytes,
    message: str,
) -> None:
    artifact = b"verified fake model"
    monkeypatch.setattr(skala, "MODEL_SHA256", hashlib.sha256(artifact).hexdigest())
    monkeypatch.setattr(skala, "MODEL_SIZE_BYTES", len(artifact))
    path = tmp_path / "model.fun"
    path.write_bytes(artifact)
    fake_torch = _FakeTorch()
    fake_torch.jit.extras[extra_name] = value

    with pytest.raises(skala.SkalaModelFormatError, match=message):
        skala.load_skala_model(model_path=path, torch_module=fake_torch)


def test_validation_normalizes_alias_and_shapes_without_torch() -> None:
    normalized = skala.validate_skala_inputs(_payload())

    assert normalized["functional"] == "skala"
    assert normalized["points"].shape == (3, 3)
    assert normalized["points"].dtype == np.float64
    assert normalized["atomic_grid_sizes"].dtype == np.int64
    assert normalized["atomic_grid_sizes"].tolist() == [2, 1]
    assert normalized["grid_profile"] == skala.REQUIRED_GRID_PROFILE
    for name in ("grad_alpha", "grad_beta"):
        assert normalized[name].shape == (3, 3)


def test_validation_accepts_and_normalizes_optional_core_metadata() -> None:
    payload = _payload()
    payload.update(
        {
            "atomic_numbers": [1, 8],
            "periodic": np.bool_(True),
            "periodic_dimension": np.int64(3),
            "lattice": np.diag([8.0, 9.0, 10.0]),
        }
    )

    normalized = skala.validate_skala_inputs(payload)

    assert normalized["atomic_numbers"].dtype == np.int64
    assert normalized["atomic_numbers"].tolist() == [1, 8]
    assert normalized["periodic"] is True
    assert normalized["periodic_dimension"] == 3
    np.testing.assert_array_equal(normalized["lattice"], payload["lattice"])
    assert set(skala.CALLBACK_OPTIONAL_INPUT_KEYS).issubset(skala.CALLBACK_INPUT_KEYS)


def test_atom_grid_chunk_plan_stably_sorts_homogeneous_complete_atoms() -> None:
    chunks = skala.plan_atom_grid_chunks([3, 5, 2, 9, 1], max_model_points=7)

    assert chunks == (
        skala.AtomGridChunk((4,), (19,), (1,)),
        skala.AtomGridChunk((2,), (8,), (2,)),
        skala.AtomGridChunk((0,), (0,), (3,)),
        skala.AtomGridChunk((1,), (3,), (5,)),
        skala.AtomGridChunk((3,), (10,), (9,)),
    )
    assert chunks[-1].num_points == 9
    assert chunks[-1].num_points > 7  # one oversized atom stays intact
    assert all(chunk.is_homogeneous for chunk in chunks)
    assert [index for chunk in chunks for index in chunk.atom_indices] == [
        4,
        2,
        0,
        1,
        3,
    ]


def test_atom_grid_chunk_plan_packs_equal_sizes_stably() -> None:
    chunks = skala.plan_atom_grid_chunks([3, 1, 2, 1, 2, 3], 4)

    assert chunks == (
        skala.AtomGridChunk((1, 3), (3, 6), (1, 1)),
        skala.AtomGridChunk((2, 4), (4, 7), (2, 2)),
        skala.AtomGridChunk((0,), (0,), (3,)),
        skala.AtomGridChunk((5,), (9,), (3,)),
    )
    assert [chunk.num_points for chunk in chunks] == [2, 4, 3, 3]


def test_atom_grid_chunk_slice_keeps_corresponding_atom_and_point_fields() -> None:
    payload = _payload()
    payload.update(
        {
            "atomic_numbers": [1, 8],
            "periodic": True,
            "periodic_dimension": 3,
            "lattice": np.diag([8.0, 9.0, 10.0]),
        }
    )
    chunks = skala.plan_atom_grid_chunks(payload["atomic_grid_sizes"], 2)
    first = skala.slice_skala_atom_chunk(payload, chunks[0])

    assert first["atomic_grid_sizes"].tolist() == [1]
    assert first["atomic_numbers"].tolist() == [8]
    np.testing.assert_array_equal(first["atom_coords"], payload["atom_coords"][1:])
    for name in (
        "points",
        "grid_weights",
        "atomic_grid_weights",
        "rho_alpha",
        "rho_beta",
        "grad_alpha",
        "grad_beta",
        "tau_alpha",
        "tau_beta",
    ):
        np.testing.assert_array_equal(first[name], payload[name][2:])
    assert first["periodic"] is True
    assert first["periodic_dimension"] == 3
    assert first["grid_profile"] == skala.REQUIRED_GRID_PROFILE
    np.testing.assert_array_equal(first["lattice"], payload["lattice"])


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_atom_grid_chunk_plan_rejects_invalid_limit(limit: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="max_model_points"):
        skala.plan_atom_grid_chunks([2, 1], limit)


@pytest.mark.parametrize(
    ("key", "bad_value", "message"),
    [
        ("grad_alpha", np.zeros((3, 2)), r"shape \(3, 3\)"),
        ("rho_beta", np.zeros(2), r"shape \(3,\)"),
        ("atom_coords", np.zeros((2, 2)), r"shape \(A, 3\)"),
        ("atomic_grid_sizes", [1, 1], "must sum"),
        ("atomic_grid_sizes", [2.0, 1.0], "must contain integers"),
        ("grid_weights", np.array([0.2, np.nan, 0.5]), "non-finite"),
    ],
)
def test_validation_rejects_malformed_callback_arrays(
    key: str, bad_value: Any, message: str
) -> None:
    payload = _payload()
    payload[key] = bad_value
    with pytest.raises((TypeError, ValueError), match=message):
        skala.validate_skala_inputs(payload)


def test_invalid_payload_fails_before_lazy_model_download(tmp_path: Path) -> None:
    downloads: list[str] = []

    def download(url: str) -> bytes:
        downloads.append(url)
        raise AssertionError("invalid input must not fetch a model")

    callback = skala.make_skala_callback(tmp_path, downloader=download)
    payload = _payload()
    payload["tau_alpha"] = np.zeros(2)

    with pytest.raises(ValueError, match="tau_alpha"):
        callback(payload)

    assert not callback.is_loaded
    assert downloads == []


def test_validation_rejects_wrong_grid_profile_before_model_use() -> None:
    payload = _payload()
    payload["grid_profile"] = "generic"
    with pytest.raises(ValueError, match="requires grid_profile='pyscf-level3'"):
        skala.validate_skala_inputs(payload)


def test_register_with_core_retains_one_lazy_callback() -> None:
    class Core:
        def __init__(self) -> None:
            self.registrations: list[tuple[str, Any, float, int, str]] = []

        def _external_functional_registration(self, name: str) -> None:
            return None

        def _define_external_functional_family(
            self,
            names: tuple[str, ...],
            callback: Any,
            *,
            hf_exchange_fraction: float,
            capability_version: int,
            required_grid_profile: str,
        ) -> None:
            self.registrations.extend(
                (
                    (
                        name,
                        callback,
                        hf_exchange_fraction,
                        capability_version,
                        required_grid_profile,
                    )
                    for name in names
                )
            )

    core = Core()
    callback = skala.register_with_core(core)

    assert [row[0] for row in core.registrations] == [
        skala.CANONICAL_FUNCTIONAL,
        *skala.FUNCTIONAL_ALIASES,
    ]
    assert all(row[1] is callback for row in core.registrations)
    assert all(row[2] == 0.0 for row in core.registrations)
    assert all(
        row[3] == skala.EXTERNAL_XC_CAPABILITY_VERSION
        for row in core.registrations
    )
    assert all(row[4] == skala.REQUIRED_GRID_PROFILE for row in core.registrations)
    assert not callback.is_loaded
    assert callback.provenance["revision"] == skala.MODEL_REVISION
    assert (
        callback.max_model_points_per_chunk
        == skala.DEFAULT_MAX_MODEL_POINTS_PER_CHUNK
    )

    assert skala.register_with_core(core) is callback
    assert len(core.registrations) == 3


def test_register_with_core_adopts_callback_after_module_state_loss() -> None:
    class Core:
        def __init__(self) -> None:
            self.registrations: dict[str, dict[str, Any]] = {}
            self.define_calls = 0

        def _external_functional_registration(
            self, name: str
        ) -> dict[str, Any] | None:
            return self.registrations.get(name)

        def _define_external_functional_family(
            self,
            names: tuple[str, ...],
            callback: Any,
            *,
            hf_exchange_fraction: float,
            capability_version: int,
            required_grid_profile: str,
        ) -> None:
            self.define_calls += 1
            if any(name in self.registrations for name in names):
                raise ValueError("one family name is already registered")
            self.registrations.update(
                {
                    name: {
                        "callback": callback,
                        "hf_exchange_fraction": hf_exchange_fraction,
                        "capability_version": capability_version,
                        "required_grid_profile": required_grid_profile,
                    }
                    for name in names
                }
            )

    core = Core()
    callback = skala.register_with_core(core)
    assert core.define_calls == 1

    # A fresh vibeqc.skala module has no module-local retention map, while
    # the native registry and its callback intentionally survive.
    skala._REGISTERED_CALLBACKS.pop(id(core))
    adopted = skala.register_with_core(core)

    assert adopted is callback
    assert core.define_calls == 1
    assert skala._REGISTERED_CALLBACKS.pop(id(core))[1] is callback


def test_register_with_core_does_not_adopt_foreign_collision() -> None:
    foreign_callback = lambda values: values

    class Core:
        def _external_functional_registration(
            self, name: str
        ) -> dict[str, Any]:
            return {
                "callback": foreign_callback,
                "hf_exchange_fraction": 0.0,
                "capability_version": skala.EXTERNAL_XC_CAPABILITY_VERSION,
                "required_grid_profile": skala.REQUIRED_GRID_PROFILE,
            }

        def _define_external_functional_family(
            self, *args: Any, **kwargs: Any
        ) -> None:
            raise AssertionError("a foreign registration must not be replaced")

    with pytest.raises(RuntimeError, match="was not registered by this"):
        skala.register_with_core(Core())


def test_register_with_core_does_not_repair_partial_alias_collision() -> None:
    callback = skala.make_skala_callback()

    class Core:
        def _external_functional_registration(
            self, name: str
        ) -> dict[str, Any] | None:
            if name != skala.CANONICAL_FUNCTIONAL:
                return None
            return {
                "callback": callback,
                "hf_exchange_fraction": 0.0,
                "capability_version": skala.EXTERNAL_XC_CAPABILITY_VERSION,
                "required_grid_profile": skala.REQUIRED_GRID_PROFILE,
            }

        def _define_external_functional_family(
            self, *args: Any, **kwargs: Any
        ) -> None:
            raise AssertionError("partial native state must fail closed")

    with pytest.raises(RuntimeError, match="only part of the immutable alias"):
        skala.register_with_core(Core())


def test_shared_alias_callback_serializes_model_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Core:
        def __init__(self) -> None:
            self.callbacks: list[Any] = []

        def _external_functional_registration(self, name: str) -> None:
            return None

        def _define_external_functional_family(
            self,
            names: tuple[str, ...],
            callback: Any,
            *,
            hf_exchange_fraction: float,
            capability_version: int,
            required_grid_profile: str,
        ) -> None:
            self.callbacks.extend(callback for _ in names)

    load_count = 0
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    def fake_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal load_count
        load_count += 1
        return object()

    def fake_evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.02)  # releases the GIL so an unlocked alias can overlap
            return {"energy": 0.0}
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(skala, "load_skala_model", fake_load)
    monkeypatch.setattr(skala, "evaluate_skala_model", fake_evaluate)
    core = Core()
    shared = skala.register_with_core(core)
    assert core.callbacks == [shared, shared, shared]

    start = threading.Barrier(len(core.callbacks))

    def invoke(callback: Any) -> dict[str, Any]:
        start.wait()
        return callback(_payload())

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(invoke, core.callbacks))

    assert results == [{"energy": 0.0}] * 3
    assert load_count == 1
    assert maximum_active == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch")
def test_chunked_energy_and_adjoints_equal_unchunked_analytic_model() -> None:
    torch = pytest.importorskip("torch")

    class AnalyticModel:
        def __init__(self) -> None:
            self.call_shapes: list[tuple[int, int]] = []

        def get_exc(self, features: dict[str, Any]) -> Any:
            weights = features["grid_weights"]
            density = features["density"]
            gradient = features["grad"]
            kinetic = features["kin"]
            atomic_sizes = features["atomic_grid_sizes"]
            coarse_per_point = torch.repeat_interleave(
                features["coarse_0_atomic_coords"].sum(dim=1), atomic_sizes
            )
            self.call_shapes.append((int(atomic_sizes.numel()), int(weights.numel())))
            integrand = (
                2.0 * density[0]
                + 3.0 * density[1]
                + 11.0 * gradient[0].sum(dim=0)
                + 13.0 * gradient[1].sum(dim=0)
                + 5.0 * kinetic[0]
                + 7.0 * kinetic[1]
                + 17.0 * features["atomic_grid_weights"]
                + 19.0 * features["grid_coords"].sum(dim=1)
                + 23.0 * coarse_per_point
            )
            return (weights * integrand).sum()

    model = AnalyticModel()
    loaded = skala.LoadedSkalaModel(
        model=model,
        path=Path("synthetic.fun"),
        metadata={},
        features=skala.MODEL_FEATURES,
        expected_d3_settings=None,
    )
    atomic_grid_sizes = [3, 1, 2, 1]
    num_points = sum(atomic_grid_sizes)
    payload = {
        "functional": "skala-1.1",
        "grid_profile": "pyscf-level3",
        "points": np.arange(num_points * 3, dtype=np.float64).reshape(-1, 3) / 9,
        "grid_weights": np.linspace(0.1, 0.7, num_points),
        "atomic_grid_weights": np.linspace(0.2, 0.8, num_points),
        "atom_coords": np.arange(12, dtype=np.float64).reshape(-1, 3) / 5,
        "atomic_grid_sizes": atomic_grid_sizes,
        "rho_alpha": np.linspace(0.1, 0.4, num_points),
        "rho_beta": np.linspace(0.05, 0.2, num_points),
        "grad_alpha": np.arange(num_points * 3, dtype=np.float64).reshape(-1, 3)
        / 10,
        "grad_beta": np.arange(num_points * 3, dtype=np.float64).reshape(-1, 3)
        / 11,
        "tau_alpha": np.linspace(0.7, 1.0, num_points),
        "tau_beta": np.linspace(0.4, 0.7, num_points),
    }
    unchunked = skala.evaluate_skala_model(
        loaded,
        payload,
        torch_module=torch,
        max_model_points_per_chunk=None,
    )
    result = skala.evaluate_skala_model(
        loaded,
        payload,
        torch_module=torch,
        max_model_points_per_chunk=3,
    )
    weights = payload["grid_weights"]

    # The finite plan stably gathers both non-contiguous one-point atoms,
    # follows with the two- and three-point groups, then scatters every
    # adjoint block back to the original atom-major point order.
    assert model.call_shapes == [(4, 7), (2, 2), (1, 2), (1, 3)]
    assert set(result) == set(skala.CALLBACK_OUTPUT_KEYS)
    assert result["energy"] == pytest.approx(unchunked["energy"], abs=1e-14)
    for name in skala.CALLBACK_OUTPUT_KEYS[1:]:
        np.testing.assert_allclose(result[name], unchunked[name], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(result["v_rho_alpha"], 2.0 * weights)
    np.testing.assert_allclose(result["v_rho_beta"], 3.0 * weights)
    np.testing.assert_allclose(result["v_tau_alpha"], 5.0 * weights)
    np.testing.assert_allclose(result["v_tau_beta"], 7.0 * weights)
    np.testing.assert_allclose(
        result["v_grad_alpha"],
        np.repeat(11.0 * weights[:, np.newaxis], 3, axis=1),
    )
    np.testing.assert_allclose(
        result["v_grad_beta"],
        np.repeat(13.0 * weights[:, np.newaxis], 3, axis=1),
    )


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch")
def test_real_checkpoint_native_he_rks_uks_oracle_offline() -> None:
    """Match Microsoft's official He/def2-SVP XC energy and AO potential.

    This fixture was generated with a fresh Microsoft SKALA 1.1.1
    ``SkalaNumInt`` under PySCF 2.14.0, using ``SkalaGrids(level=3)`` and the
    official SKALA-1.1 checkpoint at revision
    ``99b5ed87e5f69d9216e1f9e30148b922eaea1241``. The checkpoint SHA-256 is
    ``7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd``.
    The fixed positive-semidefinite density was normalized to Tr(D S) = 2;
    UKS uses D_alpha = 0.7 D and D_beta = 0.3 D. Expected matrices below are
    in vibe-qc/libint pure-p order (s, s, py, pz, px).

    The test is strictly offline: it skips unless the already-cached artifact
    has the pinned digest, then loads that explicit path through the adapter's
    no-download API. Neither PySCF nor the upstream ``skala`` package is
    imported.
    """

    torch_probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if torch_probe.returncode != 0:
        pytest.skip("optional torch module is present but cannot be imported")

    model_path = skala.model_cache_path()
    if not model_path.is_file():
        pytest.skip(f"pinned SKALA checkpoint is not cached at {model_path}")
    try:
        cached_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    except OSError as exc:
        pytest.skip(f"cached SKALA checkpoint cannot be read: {exc}")
    if cached_digest != skala.MODEL_SHA256:
        pytest.skip("cached SKALA checkpoint does not have the pinned SHA-256")

    # Keep Torch in a child process. On macOS, the Torch wheel and the native
    # core can otherwise initialize distinct libomp copies and abort before
    # Python can report a test failure. The child imports this checkout's
    # adapter module directly, not the upstream SKALA package. Passing
    # ``model_path`` to load_skala_model is the no-download code path.
    worker_script = r"""
import importlib.util
import sys
import numpy as np

adapter_path, model_path, input_path, output_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location(
    "vibeqc_skala_checkpoint_regression", adapter_path
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
with np.load(input_path, allow_pickle=False) as archive:
    values = {name: archive[name] for name in archive.files}
values["functional"] = str(values["functional"].item())
values["grid_profile"] = str(values["grid_profile"].item())
import torch
loaded = module.load_skala_model(model_path=model_path, torch_module=torch)
result = module.evaluate_skala_model(loaded, values, torch_module=torch)
np.savez(output_path, **result)
"""

    def isolated_checkpoint_callback(
        values: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "functional": np.asarray(skala.CANONICAL_FUNCTIONAL),
            "grid_profile": np.asarray(skala.REQUIRED_GRID_PROFILE),
        }
        for name in (
            "points",
            "grid_weights",
            "atomic_grid_weights",
            "atom_coords",
            "atomic_grid_sizes",
            "rho_alpha",
            "rho_beta",
            "grad_alpha",
            "grad_beta",
            "tau_alpha",
            "tau_beta",
        ):
            payload[name] = np.asarray(values[name])
        with tempfile.TemporaryDirectory(
            prefix="vibeqc-skala-checkpoint-test."
        ) as directory:
            input_path = Path(directory) / "input.npz"
            output_path = Path(directory) / "output.npz"
            np.savez(input_path, **payload)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    worker_script,
                    str(Path(skala.__file__).resolve()),
                    str(model_path),
                    str(input_path),
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if completed.returncode != 0:
                raise AssertionError(
                    "isolated SKALA checkpoint evaluation failed:\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            with np.load(output_path, allow_pickle=False) as result:
                return {
                    "energy": float(result["energy"]),
                    "v_rho_alpha": result["v_rho_alpha"].copy(),
                    "v_rho_beta": result["v_rho_beta"].copy(),
                    "v_grad_alpha": result["v_grad_alpha"].copy(),
                    "v_grad_beta": result["v_grad_beta"].copy(),
                    "v_tau_alpha": result["v_tau_alpha"].copy(),
                    "v_tau_beta": result["v_tau_beta"].copy(),
                }

    functional_name = (
        f"test-real-skala-checkpoint-{id(isolated_checkpoint_callback)}"
    )
    vq.define_external_functional(
        functional_name,
        isolated_checkpoint_callback,
        capability_version=skala.EXTERNAL_XC_CAPABILITY_VERSION,
        required_grid_profile=skala.REQUIRED_GRID_PROFILE,
    )
    molecule = vq.Molecule([vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(molecule, "def2-svp")
    grid_options = vq.GridOptions()
    grid_options.atomic_grid_profile = skala.REQUIRED_GRID_PROFILE
    grid = vq.build_grid(molecule, grid_options)
    assert len(grid.weights) == 7936
    functional = vq.Functional(functional_name, 2)

    density = np.array(
        [
            [
                0.3516134212905759,
                0.2633856993431251,
                0.25819583334621626,
                0.2984172948222599,
                0.27506289783617005,
            ],
            [
                0.2633856993431251,
                0.40740448075734625,
                0.25300596734930736,
                0.3542083542890303,
                0.23743636935858076,
            ],
            [
                0.25819583334621626,
                0.25300596734930736,
                0.29971476132148717,
                0.22316423786708137,
                0.20499970687790034,
            ],
            [
                0.2984172948222599,
                0.3542083542890303,
                0.22316423786708137,
                0.3503159547913487,
                0.2672780988408067,
            ],
            [
                0.27506289783617005,
                0.23743636935858076,
                0.20499970687790034,
                0.2672780988408067,
                0.256898366846989,
            ],
        ],
        dtype=np.float64,
    )
    expected_rks = np.array(
        [
            [-0.06121944070622484, -0.03364365223995563, -0.03340050825175153,
             -0.03565696354040812, -0.03372524811360278],
            [-0.03364365223995563, -0.02242865536291031, -0.02168584968537017,
             -0.02547654717573037, -0.02236662527560608],
            [-0.03340050825175153, -0.02168584968537017, -0.02680973181285822,
             -0.03033233419079788, -0.02164879677319363],
            [-0.03565696354040812, -0.02547654717573037, -0.03033233419079788,
             -0.06614109973503922, -0.02977756218958244],
            [-0.03372524811360278, -0.02236662527560608, -0.02164879677319363,
             -0.02977756218958244, -0.02734031462469267],
        ],
        dtype=np.float64,
    )
    expected_uks_alpha = np.array(
        [
            [-0.06161791379196799, -0.03837893315015099, -0.03078327803543561,
             -0.02863861824045125, -0.02818702857267887],
            [-0.03837893315015099, -0.02695789098777414, -0.02276437830760721,
             -0.02251113332425105, -0.01946345758068987],
            [-0.03078327803543561, -0.02276437830760721, -0.03572618197938491,
             -0.03786727294364883, -0.0151241013803245],
            [-0.02863861824045125, -0.02251113332425105, -0.03786727294364883,
             -0.10072674704813424, -0.01813096680210411],
            [-0.02818702857267887, -0.01946345758068987, -0.0151241013803245,
             -0.01813096680210411, -0.02197095497374679],
        ],
        dtype=np.float64,
    )
    expected_uks_beta = np.array(
        [
            [-0.04659995080893369, -0.02901752074333205, -0.02320518619419921,
             -0.02158325711173697, -0.02114981659552017],
            [-0.02901752074333205, -0.02039999790652265, -0.01714934996367922,
             -0.01693709072449387, -0.01465512715593532],
            [-0.02320518619419921, -0.01714934996367922, -0.02705996555706323,
             -0.02848293417796076, -0.01131080005837847],
            [-0.02158325711173697, -0.01693709072449387, -0.02848293417796076,
             -0.07603585604802596, -0.01361516990123478],
            [-0.02114981659552017, -0.01465512715593532, -0.01131080005837847,
             -0.01361516990123478, -0.01661589532015552],
        ],
        dtype=np.float64,
    )

    rks_alpha, rks_beta, rks_energy = core.evaluate_uks_xc_potential(
        functional,
        basis,
        grid,
        0.5 * density,
        0.5 * density,
    )
    assert rks_energy == pytest.approx(-0.166253712994328, abs=2e-10, rel=0.0)
    np.testing.assert_allclose(rks_alpha, expected_rks, atol=5e-10, rtol=1e-8)
    np.testing.assert_allclose(rks_beta, expected_rks, atol=5e-10, rtol=1e-8)

    uks_alpha, uks_beta, uks_energy = core.evaluate_uks_xc_potential(
        functional,
        basis,
        grid,
        0.7 * density,
        0.3 * density,
    )
    assert uks_energy == pytest.approx(
        -0.15455660678784058,
        abs=2e-10,
        rel=0.0,
    )
    np.testing.assert_allclose(
        uks_alpha,
        expected_uks_alpha,
        atol=5e-10,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        uks_beta,
        expected_uks_beta,
        atol=5e-10,
        rtol=1e-8,
    )


def test_missing_torch_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        assert name == "torch"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        skala,
        "sys",
        SimpleNamespace(version_info=(3, 13, 0), platform="linux"),
    )
    monkeypatch.setattr(skala.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"torch>=2\.12,<2\.14") as exc_info:
        skala._require_torch()
    assert "PySCF are intentionally not required" in str(exc_info.value)


@pytest.mark.parametrize(
    "torch_version",
    [
        "2.12.0",
        "2.13.1+cpu",
        "2.13.0a0+gitdeadbeef",
        "2.13.0.dev20260830+cpu.cxx11.abi",
    ],
)
def test_supported_torch_versions_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
    torch_version: str,
) -> None:
    fake_torch = SimpleNamespace(
        __version__=torch_version,
        jit=SimpleNamespace(load=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        skala,
        "sys",
        SimpleNamespace(version_info=(3, 13, 0), platform="linux"),
    )
    monkeypatch.setattr(
        skala.importlib,
        "import_module",
        lambda name: fake_torch,
    )

    assert skala._require_torch() is fake_torch


@pytest.mark.parametrize("torch_version", ["2.11.9", "2.14.0", "3.0.0+cpu"])
def test_unsupported_torch_versions_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    torch_version: str,
) -> None:
    fake_torch = SimpleNamespace(
        __version__=torch_version,
        jit=SimpleNamespace(load=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        skala,
        "sys",
        SimpleNamespace(version_info=(3, 13, 0), platform="linux"),
    )
    monkeypatch.setattr(
        skala.importlib,
        "import_module",
        lambda name: fake_torch,
    )

    with pytest.raises(ImportError, match=r"requires torch>=2\.12,<2\.14"):
        skala._require_torch()


@pytest.mark.parametrize("torch_version", [None, "", "release-2.13", "2.x.0"])
def test_unparseable_torch_versions_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    torch_version: object,
) -> None:
    fake_torch = SimpleNamespace(
        __version__=torch_version,
        jit=SimpleNamespace(load=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        skala,
        "sys",
        SimpleNamespace(version_info=(3, 13, 0), platform="linux"),
    )
    monkeypatch.setattr(
        skala.importlib,
        "import_module",
        lambda name: fake_torch,
    )

    with pytest.raises(ImportError, match=r"parseable torch\.__version__"):
        skala._require_torch()


def test_python_314_is_rejected_before_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        skala,
        "sys",
        SimpleNamespace(version_info=(3, 14, 0), platform="linux"),
    )

    def unexpected_import(name: str) -> Any:
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(skala.importlib, "import_module", unexpected_import)
    with pytest.raises(ImportError, match="Python 3.13 or earlier"):
        skala._require_torch()


def test_macos_is_rejected_before_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        skala,
        "sys",
        SimpleNamespace(version_info=(3, 13, 0), platform="darwin"),
    )

    def unexpected_import(name: str) -> Any:
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(skala.importlib, "import_module", unexpected_import)
    with pytest.raises(RuntimeError, match="not supported on macOS") as exc_info:
        skala._require_torch()
    assert "Linux CPU environment" in str(exc_info.value)


def test_unvalidated_non_linux_platform_is_rejected_before_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        skala,
        "sys",
        SimpleNamespace(version_info=(3, 13, 0), platform="win32"),
    )

    def unexpected_import(name: str) -> Any:
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(skala.importlib, "import_module", unexpected_import)
    with pytest.raises(RuntimeError, match="validated only on Linux"):
        skala._require_torch()


@pytest.mark.parametrize(
    ("version_info", "platform_name", "error_type", "message"),
    [
        ((3, 14, 0), "linux", ImportError, "Python 3.13 or earlier"),
        ((3, 13, 0), "darwin", RuntimeError, "not supported on macOS"),
        ((3, 13, 0), "win32", RuntimeError, "validated only on Linux"),
    ],
)
def test_public_loader_rejects_unsupported_runtime_before_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    version_info: tuple[int, int, int],
    platform_name: str,
    error_type: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(
        skala,
        "sys",
        SimpleNamespace(version_info=version_info, platform=platform_name),
    )

    def unexpected_download(url: str) -> bytes:
        raise AssertionError(f"unexpected download from {url}")

    with pytest.raises(error_type, match=message):
        skala.load_skala_model(tmp_path, downloader=unexpected_download)

    assert not skala.model_cache_path(tmp_path).exists()


def test_import_survives_sphinx_autodoc_mocked_core() -> None:
    """``import vibeqc`` succeeds under the documentation build's mocked core.

    ``scripts/build_site.sh`` runs Sphinx with
    ``autodoc_mock_imports=vibeqc._vibeqc_core``: a meta-path finder serves
    the core and every dotted name under it as a mock module whose every
    attribute is a callable stand-in and whose every call returns another
    stand-in.  The stand-in below mirrors ``sphinx.ext.autodoc.mock``
    (``MockFinder`` / ``MockLoader`` / ``_MockModule`` / ``_MockObject``) in
    the parts the package import touches: ``__sphinx_mock__``,
    ``__getattr__``, ``__call__``, ``__getitem__``, ``__mro_entries__``,
    empty iteration, and dotted submodule resolution.  Before #559 the
    import-time SKALA registration read the stand-in returned by
    ``_external_functional_registration`` as a collision and the whole
    package failed to import; the v0.15.158 docs-build gate was red for
    exactly that.  ``bash scripts/build_site.sh`` remains the real gate;
    this is the fast guard against the next import-time call into the core.
    """
    code = textwrap.dedent(
        """
        import importlib.abc
        import importlib.machinery
        import sys
        from types import ModuleType

        MOCKED = "vibeqc._vibeqc_core"

        class _MockObject:
            __sphinx_mock__ = True
            __display_name__ = "_MockObject"
            __name__ = ""

            def __init__(self, *args, **kwargs):
                self.__qualname__ = self.__name__

            def __len__(self):
                return 0

            def __contains__(self, key):
                return False

            def __iter__(self):
                return iter([])

            def __mro_entries__(self, bases):
                return (self.__class__,)

            def __getitem__(self, key):
                return _MockObject()

            def __getattr__(self, key):
                return _MockObject()

            def __call__(self, *args, **kwargs):
                return _MockObject()

            def __repr__(self):
                return self.__display_name__

        class _MockModule(ModuleType):
            __file__ = "/dev/null"
            __sphinx_mock__ = True

            def __init__(self, name):
                super().__init__(name)
                self.__all__ = []
                self.__path__ = []

            def __getattr__(self, name):
                return _MockObject()

            def __repr__(self):
                return self.__name__

        class _MockLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return _MockModule(spec.name)

            def exec_module(self, module):
                pass

        class _MockFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == MOCKED or fullname.startswith(MOCKED + "."):
                    return importlib.machinery.ModuleSpec(fullname, _MockLoader())
                return None

        sys.meta_path.insert(0, _MockFinder())
        import vibeqc

        assert getattr(vibeqc._vibeqc_core, "__sphinx_mock__", False)
        assert vibeqc._skala_callback is None, vibeqc._skala_callback
        print("mocked-core import ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "mocked-core import ok"
