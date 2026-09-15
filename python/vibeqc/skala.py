# SPDX-License-Identifier: MPL-2.0
#
# The checkpoint protocol and atom-grid chunk-planning contract adapt the
# MIT-licensed Microsoft SKALA implementation pinned in ``model_provenance``.
# Its copyright and permission notice ship in LICENSES/MICROSOFT-SKALA-MIT.

"""Lazy Microsoft SKALA-1.1 exchange-correlation model adapter.

The published model is fetched from an immutable Hugging Face revision on
first use and verified before TorchScript is allowed to deserialize it.  This
module deliberately does not depend on the upstream ``skala`` Python package:
that package imports PySCF, while vibe-qc's runtime must remain independent of
other quantum-chemistry programs.

Only PyTorch is required for inference, and even that import is delayed until
the first valid callback invocation.  The returned derivatives are adjoints of
the *integrated* XC energy, so the quadrature weights are already included.
They can therefore be projected directly into molecular or periodic XC matrix
builders without assuming that the functional is semilocal.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import re
import sys
import tempfile
import threading
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

CANONICAL_FUNCTIONAL = "skala-1.1"
"""Canonical vibe-qc name for the recommended Microsoft SKALA model."""

FUNCTIONAL_ALIASES = ("skala-1.1-rev1", "skala")
"""Names which resolve to the same immutable SKALA-1.1 revision."""

MODEL_REPOSITORY = "microsoft/skala-1.1"
MODEL_REVISION = "99b5ed87e5f69d9216e1f9e30148b922eaea1241"
MODEL_FILENAME = "skala-1.1-rev1.fun"
MODEL_SIZE_BYTES = 2_388_988
MODEL_SHA256 = "7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd"
MODEL_CARD_URL = (
    f"https://huggingface.co/{MODEL_REPOSITORY}/blob/{MODEL_REVISION}/README.md"
)
SOURCE_REPOSITORY = "microsoft/skala"
SOURCE_REVISION = "4f3f072b820183d4c77b47bcea6df9736655d12a"
SOURCE_NOTICE_FILENAME = "MICROSOFT-SKALA-SOURCE-MIT.txt"
MODEL_PROTOCOL_VERSION = 2
EXTERNAL_XC_CAPABILITY_VERSION = 1
REQUIRED_GRID_PROFILE = "pyscf-level3"
MODEL_EXPECTED_D3_SETTINGS = "b3lyp5"
MODEL_FIRST_DERIVATIVE_BYTES_PER_POINT = 6680 * 8
DEFAULT_MAX_MODEL_POINTS_PER_CHUNK = 8192
_SUPPORTED_TORCH_RELEASE_MIN = (2, 12)
_SUPPORTED_TORCH_RELEASE_MAX = (2, 14)
_TORCH_VERSION_PATTERN = re.compile(
    r"^\s*(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)"
    r"(?:\.(?:0|[1-9]\d*))?"
    r"(?:(?:a|b|rc)\d+)?"
    r"(?:\.?(?:post|dev)\d+)?"
    r"(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?\s*$",
    re.IGNORECASE,
)
MODEL_URL = (
    f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/"
    f"{MODEL_REVISION}/{MODEL_FILENAME}"
)

# Serialized by the official SKALA-1.1 implementation in this exact order.
# Luise et al., "Accurate and Scalable Exchange-Correlation with Deep
# Learning" (2025), Sec. 3 and Appendix A, arXiv:2506.14665, describe the
# semilocal density/gradient/kinetic features and atom-packed nonlocal grid.
MODEL_FEATURES = (
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

SOURCE_LICENSE_NOTICE_TEXT = """MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

CALLBACK_REQUIRED_INPUT_KEYS = (
    "functional",
    "grid_profile",
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
)

CALLBACK_OPTIONAL_INPUT_KEYS = (
    "atomic_numbers",
    "periodic",
    "periodic_dimension",
    "lattice",
)

CALLBACK_INPUT_KEYS = CALLBACK_REQUIRED_INPUT_KEYS + CALLBACK_OPTIONAL_INPUT_KEYS
"""All core callback fields understood by the adapter."""

CALLBACK_OUTPUT_KEYS = (
    "energy",
    "v_rho_alpha",
    "v_rho_beta",
    "v_grad_alpha",
    "v_grad_beta",
    "v_tau_alpha",
    "v_tau_beta",
)

_REGISTERED_CALLBACKS: dict[int, tuple[Any, "SkalaFunctionalCallback"]] = {}
_CALLBACK_REGISTRATION_ID = (
    "vibeqc.skala.external-xc.v1:"
    f"{MODEL_REVISION}:{MODEL_SHA256}"
)


class SkalaModelIntegrityError(RuntimeError):
    """The cached or downloaded model does not have its published digest."""


class SkalaModelFormatError(RuntimeError):
    """The verified TorchScript file violates the SKALA protocol contract."""


@dataclass(frozen=True)
class LoadedSkalaModel:
    """A verified SKALA TorchScript model and its embedded metadata."""

    model: Any
    path: Path
    metadata: Mapping[str, Any]
    features: tuple[str, ...]
    expected_d3_settings: str | None
    sha256: str = MODEL_SHA256


@dataclass(frozen=True)
class AtomGridChunk:
    """One complete-atom model chunk in its model-evaluation order.

    The indices refer to the original atom-major callback payload. Finite
    chunk plans contain only equal-sized atomic grids, so the model's padded
    point count equals the real point count. ``max_model_points=None`` is the
    explicit unchunked escape hatch and may produce one heterogeneous chunk.
    """

    atom_indices: tuple[int, ...]
    point_starts: tuple[int, ...]
    atomic_grid_sizes: tuple[int, ...]

    @property
    def num_atoms(self) -> int:
        return len(self.atom_indices)

    @property
    def num_points(self) -> int:
        return sum(self.atomic_grid_sizes)

    @property
    def is_homogeneous(self) -> bool:
        """Whether every atom in the chunk has the same grid size."""

        return len(set(self.atomic_grid_sizes)) <= 1


def skala_cache_root() -> Path:
    """Return the XDG cache root used for SKALA artifacts."""

    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "vibeqc" / "skala"


def model_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the immutable repository/revision directory for SKALA-1.1.

    ``cache_dir`` overrides the normal ``$XDG_CACHE_HOME/vibeqc/skala`` root;
    it is primarily useful for isolated applications and tests.
    """

    root = Path(cache_dir).expanduser() if cache_dir is not None else skala_cache_root()
    return root / MODEL_REPOSITORY / MODEL_REVISION


def model_cache_path(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the expected local path of the pinned TorchScript artifact."""

    return model_cache_dir(cache_dir) / MODEL_FILENAME


def model_provenance() -> dict[str, str | int]:
    """Return the immutable source and integrity record stored with the model."""

    return {
        "artifact": "Microsoft SKALA-1.1 TorchScript XC functional",
        "functional": CANONICAL_FUNCTIONAL,
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "filename": MODEL_FILENAME,
        "expected_size_bytes": MODEL_SIZE_BYTES,
        "url": MODEL_URL,
        "sha256": MODEL_SHA256,
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "expected_d3_settings": MODEL_EXPECTED_D3_SETTINGS,
        "model_card_license_metadata": "MIT",
        "model_card_url": MODEL_CARD_URL,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_notice_file": SOURCE_NOTICE_FILENAME,
        "source_notice_scope": (
            "Microsoft SKALA source repository only; not checkpoint "
            "redistribution clearance"
        ),
    }


def _run_provenance_fields() -> dict[str, str | int]:
    """Return flat scalar fields for a job's ``[run]`` manifest table.

    This record is available before PyTorch or the model is loaded.  It binds
    every SKALA energy to the immutable checkpoint identity and the execution
    contract rather than relying only on the user's mutable cache directory.
    """

    return {
        "xc_backend": "external-full-grid",
        "skala_model_repository": MODEL_REPOSITORY,
        "skala_model_revision": MODEL_REVISION,
        "skala_model_filename": MODEL_FILENAME,
        "skala_model_expected_size_bytes": MODEL_SIZE_BYTES,
        "skala_model_sha256": MODEL_SHA256,
        "skala_model_protocol_version": MODEL_PROTOCOL_VERSION,
        "skala_model_card_license_metadata": "MIT",
        "skala_source_repository": SOURCE_REPOSITORY,
        "skala_source_revision": SOURCE_REVISION,
        "skala_source_notice_file": SOURCE_NOTICE_FILENAME,
        "skala_source_notice_scope": "source repository only",
        "skala_model_device": "cpu",
        "skala_feature_adjoint_dtype": "float64",
        "skala_model_parameter_dtype": "float32",
        "skala_energy_accumulation_dtype": "float64",
        "skala_model_chunk_target_points": (
            DEFAULT_MAX_MODEL_POINTS_PER_CHUNK
        ),
        "skala_expected_d3_settings": MODEL_EXPECTED_D3_SETTINGS,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` with ``data`` in the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_if_changed(path: Path, data: bytes) -> None:
    try:
        if path.read_bytes() == data:
            return
    except OSError:
        pass
    _atomic_write(path, data)


def _write_model_records(directory: Path) -> None:
    provenance = json.dumps(
        model_provenance(), indent=2, sort_keys=True, ensure_ascii=True
    ).encode("utf-8") + b"\n"
    _write_if_changed(directory / "provenance.json", provenance)
    _write_if_changed(
        directory / SOURCE_NOTICE_FILENAME,
        SOURCE_LICENSE_NOTICE_TEXT.encode("utf-8"),
    )


def _require_model_size(data: bytes, *, source: str, consequence: str) -> None:
    """Reject an artifact with the wrong pinned byte length before hashing."""

    actual = len(data)
    if actual != MODEL_SIZE_BYTES:
        raise SkalaModelIntegrityError(
            f"{source} size mismatch: expected {MODEL_SIZE_BYTES} bytes, got "
            f"{actual}. {consequence}"
        )


def _download_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "vibe-qc-skala/1.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Read at most one byte beyond the immutable artifact size. This
            # bounds memory before SHA verification even if the endpoint or a
            # proxy serves an unexpectedly large response.
            remaining = MODEL_SIZE_BYTES + 1
            chunks: list[bytes] = []
            while remaining:
                chunk = response.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
                if remaining <= 0:
                    break
            data = b"".join(chunks)
            _require_model_size(
                data,
                source="Downloaded Microsoft SKALA-1.1 model",
                consequence="The response was not hashed or cached.",
            )
            return data
    except OSError as exc:
        raise RuntimeError(
            "Unable to download the pinned Microsoft SKALA-1.1 model from "
            f"{url}. Check network access or populate {model_cache_path()} "
            "with the published artifact."
        ) from exc


def _read_verified_model(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            data = handle.read(MODEL_SIZE_BYTES + 1)
    except OSError as exc:
        raise FileNotFoundError(
            f"SKALA model artifact is unavailable at {path}"
        ) from exc
    _require_model_size(
        data,
        source=f"Microsoft SKALA-1.1 model at {path}",
        consequence="TorchScript was not deserialized.",
    )
    actual = _sha256(data)
    if actual != MODEL_SHA256:
        raise SkalaModelIntegrityError(
            "Microsoft SKALA-1.1 model SHA-256 mismatch at "
            f"{path}: expected {MODEL_SHA256}, got {actual}. TorchScript was "
            "not deserialized."
        )
    return data


def ensure_skala_model(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    downloader: Callable[[str], bytes] | None = None,
) -> Path:
    """Return a verified local model, downloading it atomically if needed.

    A corrupt cache entry is never loaded.  It is replaced only after a newly
    downloaded artifact has passed the published SHA-256 check, so a failed
    repair leaves the previous bytes in place for diagnosis.
    """

    path = model_cache_path(cache_dir)
    try:
        _read_verified_model(path)
    except (FileNotFoundError, SkalaModelIntegrityError):
        fetch = downloader or _download_bytes
        data = fetch(MODEL_URL)
        if not isinstance(data, bytes):
            raise TypeError("SKALA model downloader must return bytes")
        _require_model_size(
            data,
            source="Downloaded Microsoft SKALA-1.1 model",
            consequence="The cache was not updated.",
        )
        actual = _sha256(data)
        if actual != MODEL_SHA256:
            raise SkalaModelIntegrityError(
                "Downloaded Microsoft SKALA-1.1 model SHA-256 mismatch: "
                f"expected {MODEL_SHA256}, got {actual}. The cache was not updated."
            )
        _atomic_write(path, data)
        _read_verified_model(path)

    _write_model_records(path.parent)
    return path


def _torch_release(version: object) -> tuple[int, int] | None:
    """Return the major/minor release from a normal PyTorch version string."""

    if not isinstance(version, str):
        return None
    match = _TORCH_VERSION_PATTERN.fullmatch(version)
    if match is None:
        return None
    return int(match.group("major")), int(match.group("minor"))


def _require_torch() -> Any:
    if sys.version_info >= (3, 14):
        raise ImportError(
            "Microsoft SKALA-1.1 requires Python 3.13 or earlier because "
            "this adapter has not yet validated the native-core/PyTorch "
            f"combination on Python {sys.version_info[0]}."
            f"{sys.version_info[1]}. "
            "Create a Python 3.11 through 3.13 environment and install "
            "vibe-qc's SKALA extra there. A published PyTorch wheel alone "
            "does not establish that the combined OpenMP runtime is safe."
        )
    if sys.platform == "darwin":
        raise RuntimeError(
            "In-process Microsoft SKALA-1.1 evaluation is not supported on "
            "macOS yet: the vibe-qc native core and current PyTorch wheels "
            "can initialize conflicting OpenMP runtimes. Use a compatible "
            "Linux CPU environment. Provenance inspection, model prefetch, "
            "and dry_run=True remain safe on macOS."
        )
    if sys.platform != "linux":
        raise RuntimeError(
            "In-process Microsoft SKALA-1.1 evaluation is currently "
            "validated only on Linux CPU environments; got platform "
            f"{sys.platform!r}. Provenance inspection, model prefetch, and "
            "dry_run=True remain available."
        )
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "Microsoft SKALA-1.1 requires the optional PyTorch runtime. "
            "Install vibe-qc's SKALA extra or a compatible CPU build with "
            "`python -m pip install 'torch>=2.12,<2.14'`. The upstream skala "
            "package and PySCF are intentionally not required."
        ) from exc
    torch_version = getattr(torch, "__version__", None)
    torch_release = _torch_release(torch_version)
    if torch_release is None:
        raise ImportError(
            "The installed PyTorch runtime does not expose a parseable "
            f"torch.__version__ value (got {torch_version!r}). Microsoft "
            "SKALA-1.1 in vibe-qc requires torch>=2.12,<2.14."
        )
    if not (
        _SUPPORTED_TORCH_RELEASE_MIN
        <= torch_release
        < _SUPPORTED_TORCH_RELEASE_MAX
    ):
        raise ImportError(
            f"Unsupported PyTorch version {torch_version!r}. Microsoft "
            "SKALA-1.1 in vibe-qc requires torch>=2.12,<2.14."
        )
    if not hasattr(torch, "jit") or not hasattr(torch.jit, "load"):
        raise ImportError(
            "The installed PyTorch build does not provide torch.jit.load, which "
            "is required by the published SKALA-1.1 artifact."
        )
    return torch


def _decode_json_extra(extra_files: Mapping[str, Any], name: str) -> Any:
    raw = extra_files.get(name)
    if not isinstance(raw, (bytes, bytearray)):
        raise SkalaModelFormatError(
            f"SKALA TorchScript extra file {name!r} is missing or is not bytes"
        )
    try:
        return json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkalaModelFormatError(
            f"SKALA TorchScript extra file {name!r} is not valid UTF-8 JSON"
        ) from exc


def load_skala_model(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    downloader: Callable[[str], bytes] | None = None,
    model_path: str | os.PathLike[str] | None = None,
    torch_module: Any | None = None,
) -> LoadedSkalaModel:
    """Verify and load the official CPU TorchScript model.

    The artifact is read into memory and hashed immediately before every
    ``torch.jit.load`` call.  Deserialization receives that verified byte
    buffer rather than reopening the path, which closes a check/load race.
    ``model_path`` is accepted for offline deployments but must contain the
    exact published bytes.
    """

    # Reject unsupported runtimes before cache lookup or network access. The
    # explicit ``ensure_skala_model`` API remains available for safe prefetch.
    torch = torch_module if torch_module is not None else _require_torch()
    path = (
        Path(model_path).expanduser()
        if model_path is not None
        else ensure_skala_model(cache_dir, downloader=downloader)
    )
    verified_bytes = _read_verified_model(path)

    extra_files: dict[str, bytes] = {
        "metadata": b"",
        "features": b"",
        "expected_d3_settings": b"",
        "protocol_version": b"",
    }
    try:
        model = torch.jit.load(
            io.BytesIO(verified_bytes),
            _extra_files=extra_files,
            map_location="cpu",
        )
    except Exception as exc:
        raise SkalaModelFormatError(
            "The verified SKALA-1.1 artifact could not be deserialized by "
            "torch.jit.load"
        ) from exc

    metadata = _decode_json_extra(extra_files, "metadata")
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) for key in metadata
    ):
        raise SkalaModelFormatError(
            "SKALA metadata must be a JSON object with string keys"
        )

    features = _decode_json_extra(extra_files, "features")
    if not isinstance(features, list) or not all(
        isinstance(feature, str) for feature in features
    ):
        raise SkalaModelFormatError("SKALA features must be a JSON list of strings")
    if tuple(features) != MODEL_FEATURES:
        raise SkalaModelFormatError(
            "SKALA model feature contract mismatch: expected "
            f"{MODEL_FEATURES!r}, got {tuple(features)!r}"
        )

    protocol = _decode_json_extra(extra_files, "protocol_version")
    if isinstance(protocol, bool) or not isinstance(protocol, int):
        raise SkalaModelFormatError("SKALA protocol_version must be an integer")
    if protocol != MODEL_PROTOCOL_VERSION:
        raise SkalaModelFormatError(
            f"Unsupported SKALA protocol {protocol}; vibe-qc supports "
            f"protocol {MODEL_PROTOCOL_VERSION}"
        )

    expected_d3 = _decode_json_extra(extra_files, "expected_d3_settings")
    if expected_d3 is not None and not isinstance(expected_d3, str):
        raise SkalaModelFormatError(
            "SKALA expected_d3_settings must be a JSON string or null"
        )
    if expected_d3 != MODEL_EXPECTED_D3_SETTINGS:
        raise SkalaModelFormatError(
            "SKALA model D3 contract mismatch: expected "
            f"{MODEL_EXPECTED_D3_SETTINGS!r}, got {expected_d3!r}"
        )
    if not callable(getattr(model, "get_exc", None)):
        raise SkalaModelFormatError(
            "SKALA TorchScript model has no callable get_exc method"
        )
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()

    return LoadedSkalaModel(
        model=model,
        path=path,
        metadata=dict(metadata),
        features=tuple(features),
        expected_d3_settings=expected_d3,
        sha256=MODEL_SHA256,
    )


def _as_float_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError(f"SKALA input {name!r} must be real-valued")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"SKALA input {name!r} must be a numeric array") from exc
    if array.shape != shape:
        raise ValueError(
            f"SKALA input {name!r} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"SKALA input {name!r} contains non-finite values")
    return np.ascontiguousarray(array)


def _as_atomic_grid_sizes(value: Any, num_atoms: int, num_points: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (num_atoms,):
        raise ValueError(
            "SKALA input 'atomic_grid_sizes' must have shape "
            f"({num_atoms},), got {raw.shape}"
        )
    values = raw.tolist()
    if not all(
        isinstance(size, Integral) and not isinstance(size, bool) for size in values
    ):
        raise TypeError("SKALA input 'atomic_grid_sizes' must contain integers")
    integer_values = [int(size) for size in values]
    if any(size <= 0 for size in integer_values):
        raise ValueError("SKALA atomic grid sizes must all be positive")
    if sum(integer_values) != num_points:
        raise ValueError(
            "SKALA atomic grid sizes must sum to the number of grid points: "
            f"got {sum(integer_values)} and {num_points}"
        )
    return np.ascontiguousarray(np.asarray(integer_values, dtype=np.int64))


def validate_skala_inputs(values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one external-functional callback payload.

    This function has no PyTorch or network dependency, so callers can fail
    fast on malformed molecular or periodic grid data.
    """

    if not isinstance(values, Mapping):
        raise TypeError("SKALA callback input must be a mapping")
    missing = [key for key in CALLBACK_REQUIRED_INPUT_KEYS if key not in values]
    if missing:
        raise KeyError("SKALA callback input is missing: " + ", ".join(missing))

    functional_value = values["functional"]
    if not isinstance(functional_value, str):
        raise TypeError("SKALA input 'functional' must be a string")
    functional = functional_value.strip().lower()
    supported = frozenset((CANONICAL_FUNCTIONAL, *FUNCTIONAL_ALIASES))
    if functional not in supported:
        raise ValueError(
            f"Unsupported SKALA functional {functional_value!r}; expected one of "
            f"{sorted(supported)!r}"
        )

    grid_profile_value = values["grid_profile"]
    if not isinstance(grid_profile_value, str):
        raise TypeError("SKALA input 'grid_profile' must be a string")
    grid_profile = grid_profile_value.strip().lower().replace("_", "-")
    if grid_profile != REQUIRED_GRID_PROFILE:
        raise ValueError(
            "vibe-qc's accepted SKALA profile requires "
            "grid_profile='pyscf-level3'; got "
            f"{grid_profile_value!r}"
        )

    points_raw = np.asarray(values["points"])
    if points_raw.ndim != 2 or points_raw.shape[1:] != (3,):
        raise ValueError(
            "SKALA input 'points' must have shape (N, 3), got "
            f"{points_raw.shape}"
        )
    num_points = int(points_raw.shape[0])
    if num_points == 0:
        raise ValueError("SKALA requires at least one grid point")

    atom_coords_raw = np.asarray(values["atom_coords"])
    if atom_coords_raw.ndim != 2 or atom_coords_raw.shape[1:] != (3,):
        raise ValueError(
            "SKALA input 'atom_coords' must have shape (A, 3), got "
            f"{atom_coords_raw.shape}"
        )
    num_atoms = int(atom_coords_raw.shape[0])
    if num_atoms == 0:
        raise ValueError("SKALA requires at least one atom")

    normalized: dict[str, Any] = {
        "functional": functional,
        "grid_profile": grid_profile,
        "points": _as_float_array("points", values["points"], (num_points, 3)),
        "atom_coords": _as_float_array(
            "atom_coords", values["atom_coords"], (num_atoms, 3)
        ),
        "atomic_grid_sizes": _as_atomic_grid_sizes(
            values["atomic_grid_sizes"], num_atoms, num_points
        ),
    }
    for name in (
        "grid_weights",
        "atomic_grid_weights",
        "rho_alpha",
        "rho_beta",
        "tau_alpha",
        "tau_beta",
    ):
        normalized[name] = _as_float_array(name, values[name], (num_points,))
    for name in ("grad_alpha", "grad_beta"):
        normalized[name] = _as_float_array(name, values[name], (num_points, 3))

    if "atomic_numbers" in values:
        raw_numbers = np.asarray(values["atomic_numbers"])
        if raw_numbers.shape != (num_atoms,):
            raise ValueError(
                "SKALA input 'atomic_numbers' must have shape "
                f"({num_atoms},), got {raw_numbers.shape}"
            )
        numbers = raw_numbers.tolist()
        if not all(
            isinstance(number, Integral) and not isinstance(number, bool)
            for number in numbers
        ):
            raise TypeError("SKALA input 'atomic_numbers' must contain integers")
        integer_numbers = [int(number) for number in numbers]
        if any(number < 1 or number > 118 for number in integer_numbers):
            raise ValueError("SKALA atomic numbers must be between 1 and 118")
        normalized["atomic_numbers"] = np.ascontiguousarray(
            np.asarray(integer_numbers, dtype=np.int64)
        )

    if "periodic" in values:
        periodic = values["periodic"]
        if not isinstance(periodic, (bool, np.bool_)):
            raise TypeError("SKALA input 'periodic' must be boolean")
        normalized["periodic"] = bool(periodic)

    if "periodic_dimension" in values:
        dimension = values["periodic_dimension"]
        if not isinstance(dimension, Integral) or isinstance(dimension, bool):
            raise TypeError("SKALA input 'periodic_dimension' must be an integer")
        dimension = int(dimension)
        if dimension < 0 or dimension > 3:
            raise ValueError("SKALA periodic dimension must be between 0 and 3")
        normalized["periodic_dimension"] = dimension

    if "lattice" in values:
        normalized["lattice"] = _as_float_array(
            "lattice", values["lattice"], (3, 3)
        )
    return normalized


def _validated_chunk_limit(max_model_points: int | None) -> int | None:
    if max_model_points is None:
        return None
    if not isinstance(max_model_points, Integral) or isinstance(
        max_model_points, bool
    ):
        raise TypeError("max_model_points must be a positive integer or None")
    limit = int(max_model_points)
    if limit <= 0:
        raise ValueError("max_model_points must be positive")
    return limit


def plan_atom_grid_chunks(
    atomic_grid_sizes: Sequence[int] | np.ndarray,
    max_model_points: int | None = DEFAULT_MAX_MODEL_POINTS_PER_CHUNK,
) -> tuple[AtomGridChunk, ...]:
    """Build stable size-sorted, homogeneous complete-atom model chunks.

    SKALA pads every model call to ``max(atomic_grid_sizes) * num_atoms``.
    Grouping equal-sized atoms makes that padded size equal the real point
    count, matching the official adapter and the published first-derivative
    memory calibration. Equal-sized atoms retain their input order. The limit
    is a target rather than permission to split an atom: one atom larger than
    ``max_model_points`` forms one oversized chunk by itself. ``None``
    requests one unpermuted, potentially heterogeneous model call.
    """

    raw_sizes = np.asarray(atomic_grid_sizes)
    if raw_sizes.ndim != 1 or raw_sizes.size == 0:
        raise ValueError("atomic_grid_sizes must be a non-empty one-dimensional array")
    size_values = raw_sizes.tolist()
    if not all(
        isinstance(size, Integral) and not isinstance(size, bool)
        for size in size_values
    ):
        raise TypeError("atomic_grid_sizes must contain integers")
    sizes = [int(size) for size in size_values]
    if any(size <= 0 for size in sizes):
        raise ValueError("atomic_grid_sizes must all be positive")

    limit = _validated_chunk_limit(max_model_points)
    point_starts: list[int] = []
    points_seen = 0
    for size in sizes:
        point_starts.append(points_seen)
        points_seen += size

    if limit is None:
        return (
            AtomGridChunk(
                atom_indices=tuple(range(len(sizes))),
                point_starts=tuple(point_starts),
                atomic_grid_sizes=tuple(sizes),
            ),
        )

    chunks: list[AtomGridChunk] = []
    atom_order = sorted(range(len(sizes)), key=sizes.__getitem__)
    group_start = 0
    while group_start < len(atom_order):
        atom_grid_size = sizes[atom_order[group_start]]
        group_stop = group_start + 1
        while (
            group_stop < len(atom_order)
            and sizes[atom_order[group_stop]] == atom_grid_size
        ):
            group_stop += 1

        atoms_per_chunk = max(1, limit // atom_grid_size)
        for chunk_start in range(group_start, group_stop, atoms_per_chunk):
            indices = tuple(
                atom_order[
                    chunk_start : min(chunk_start + atoms_per_chunk, group_stop)
                ]
            )
            chunks.append(
                AtomGridChunk(
                    atom_indices=indices,
                    point_starts=tuple(point_starts[index] for index in indices),
                    atomic_grid_sizes=(atom_grid_size,) * len(indices),
                )
            )
        group_start = group_stop
    return tuple(chunks)


def _chunk_point_indices(chunk: AtomGridChunk) -> np.ndarray:
    """Expand a compact complete-atom chunk into original point indices."""

    if chunk.num_atoms == 0:
        return np.empty(0, dtype=np.int64)
    blocks = [
        np.arange(start, start + size, dtype=np.int64)
        for start, size in zip(
            chunk.point_starts, chunk.atomic_grid_sizes, strict=True
        )
    ]
    return np.concatenate(blocks)


def _slice_validated_atom_chunk(
    normalized: Mapping[str, Any], chunk: AtomGridChunk
) -> dict[str, Any]:
    sizes = np.asarray(normalized["atomic_grid_sizes"], dtype=np.int64)
    num_atoms = int(sizes.shape[0])
    if (
        chunk.num_atoms == 0
        or len(chunk.point_starts) != chunk.num_atoms
        or len(chunk.atomic_grid_sizes) != chunk.num_atoms
        or len(set(chunk.atom_indices)) != chunk.num_atoms
        or any(index < 0 or index >= num_atoms for index in chunk.atom_indices)
    ):
        raise ValueError(f"invalid SKALA atom chunk layout: {chunk!r}")

    original_starts = np.cumsum(
        np.concatenate((np.zeros(1, dtype=np.int64), sizes[:-1]))
    )
    expected_starts = tuple(int(original_starts[index]) for index in chunk.atom_indices)
    expected_sizes = tuple(int(sizes[index]) for index in chunk.atom_indices)
    if (
        chunk.point_starts != expected_starts
        or chunk.atomic_grid_sizes != expected_sizes
    ):
        raise ValueError(
            "SKALA chunk layout does not match the complete atom blocks: "
            f"{chunk!r}"
        )

    atom_indices = np.asarray(chunk.atom_indices, dtype=np.int64)
    point_indices = _chunk_point_indices(chunk)
    sliced: dict[str, Any] = {
        "functional": normalized["functional"],
        "atom_coords": np.ascontiguousarray(normalized["atom_coords"][atom_indices]),
        "atomic_grid_sizes": np.ascontiguousarray(
            normalized["atomic_grid_sizes"][atom_indices]
        ),
    }
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
        sliced[name] = np.ascontiguousarray(normalized[name][point_indices])

    if "atomic_numbers" in normalized:
        sliced["atomic_numbers"] = np.ascontiguousarray(
            normalized["atomic_numbers"][atom_indices]
        )
    for name in ("grid_profile", "periodic", "periodic_dimension", "lattice"):
        if name in normalized:
            sliced[name] = normalized[name]
    return sliced


def slice_skala_atom_chunk(
    values: Mapping[str, Any], chunk: AtomGridChunk
) -> dict[str, Any]:
    """Validate and extract one complete-atom chunk without importing torch."""

    return _slice_validated_atom_chunk(validate_skala_inputs(values), chunk)


def _differentiable_tensor(torch: Any, array: np.ndarray) -> Any:
    tensor = torch.as_tensor(array, dtype=torch.float64, device="cpu")
    return tensor.detach().clone().requires_grad_(True)


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64).copy()


def _evaluate_skala_chunk(
    loaded: LoadedSkalaModel,
    normalized: Mapping[str, Any],
    *,
    torch: Any,
) -> dict[str, Any]:
    """Evaluate one already-validated complete-atom chunk."""

    with torch.enable_grad():
        rho_alpha = _differentiable_tensor(torch, normalized["rho_alpha"])
        rho_beta = _differentiable_tensor(torch, normalized["rho_beta"])
        grad_alpha = _differentiable_tensor(torch, normalized["grad_alpha"])
        grad_beta = _differentiable_tensor(torch, normalized["grad_beta"])
        tau_alpha = _differentiable_tensor(torch, normalized["tau_alpha"])
        tau_beta = _differentiable_tensor(torch, normalized["tau_beta"])

        atomic_sizes = torch.as_tensor(
            normalized["atomic_grid_sizes"], dtype=torch.int64, device="cpu"
        )
        max_atomic_size = int(normalized["atomic_grid_sizes"].max())
        features = {
            "density": torch.stack((rho_alpha, rho_beta), dim=0),
            "kin": torch.stack((tau_alpha, tau_beta), dim=0),
            "grad": torch.stack((grad_alpha.T, grad_beta.T), dim=0),
            "grid_coords": torch.as_tensor(
                normalized["points"], dtype=torch.float64, device="cpu"
            ),
            "grid_weights": torch.as_tensor(
                normalized["grid_weights"], dtype=torch.float64, device="cpu"
            ),
            "atomic_grid_weights": torch.as_tensor(
                normalized["atomic_grid_weights"],
                dtype=torch.float64,
                device="cpu",
            ),
            "atomic_grid_sizes": atomic_sizes,
            "coarse_0_atomic_coords": torch.as_tensor(
                normalized["atom_coords"], dtype=torch.float64, device="cpu"
            ),
            "atomic_grid_size_bound_shape": torch.empty(
                (max_atomic_size, 0), dtype=torch.int64, device="cpu"
            ),
        }

        # The protocol's get_exc method returns the integrated XC energy, not
        # an energy density: the official SkalaFunctional.get_exc pads the
        # atom-major rows, multiplies its enhancement density by grid_weights,
        # and sums.  Differentiating this scalar therefore yields the full
        # quadrature-weighted adjoints required by the AO projector.
        energy = loaded.model.get_exc(features)
        if not hasattr(energy, "numel") or int(energy.numel()) != 1:
            raise SkalaModelFormatError("SKALA get_exc must return one scalar energy")
        energy = energy.reshape(())
        variables = (
            rho_alpha,
            rho_beta,
            grad_alpha,
            grad_beta,
            tau_alpha,
            tau_beta,
        )
        try:
            adjoints = torch.autograd.grad(
                energy,
                variables,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "SKALA-1.1 could not differentiate its integrated XC energy "
                "with respect to density, gradient, and kinetic-energy features"
            ) from exc

    if len(adjoints) != len(variables) or any(value is None for value in adjoints):
        raise RuntimeError("SKALA-1.1 returned an incomplete feature adjoint set")
    return {
        "energy": float(energy.detach().cpu().item()),
        "v_rho_alpha": _tensor_to_numpy(adjoints[0]),
        "v_rho_beta": _tensor_to_numpy(adjoints[1]),
        "v_grad_alpha": _tensor_to_numpy(adjoints[2]),
        "v_grad_beta": _tensor_to_numpy(adjoints[3]),
        "v_tau_alpha": _tensor_to_numpy(adjoints[4]),
        "v_tau_beta": _tensor_to_numpy(adjoints[5]),
    }


def evaluate_skala_model(
    loaded: LoadedSkalaModel,
    values: Mapping[str, Any],
    *,
    torch_module: Any | None = None,
    max_model_points_per_chunk: int | None = DEFAULT_MAX_MODEL_POINTS_PER_CHUNK,
) -> dict[str, Any]:
    """Evaluate integrated XC energy and adjoints in complete-atom chunks.

    SKALA's atom-packed architecture has no cross-atom model edges: each atom
    owns a contiguous fine-grid block and its corresponding coarse coordinate.
    Stable size sorting and chunking only between complete, equal-sized atoms
    therefore preserves the functional while eliminating ragged-padding
    overhead and bounding the large first-derivative autograd graph. The
    default target is 8,192 real and padded points (roughly 417.5 MiB at
    53,440 bytes per point); one larger atom is evaluated intact. Pass
    ``None`` only when an explicitly unchunked evaluation is safe.
    """

    normalized = validate_skala_inputs(values)
    chunks = plan_atom_grid_chunks(
        normalized["atomic_grid_sizes"], max_model_points_per_chunk
    )
    torch = torch_module if torch_module is not None else _require_torch()

    point_outputs = {
        "v_rho_alpha": np.empty_like(normalized["rho_alpha"]),
        "v_rho_beta": np.empty_like(normalized["rho_beta"]),
        "v_grad_alpha": np.empty_like(normalized["grad_alpha"]),
        "v_grad_beta": np.empty_like(normalized["grad_beta"]),
        "v_tau_alpha": np.empty_like(normalized["tau_alpha"]),
        "v_tau_beta": np.empty_like(normalized["tau_beta"]),
    }
    energy_terms: list[float] = []
    for chunk in chunks:
        chunk_values = _slice_validated_atom_chunk(normalized, chunk)
        chunk_result = _evaluate_skala_chunk(loaded, chunk_values, torch=torch)
        energy_terms.append(float(chunk_result["energy"]))
        point_indices = _chunk_point_indices(chunk)
        for name, output in point_outputs.items():
            output[point_indices] = chunk_result[name]

    total_energy = (
        energy_terms[0] if len(energy_terms) == 1 else math.fsum(energy_terms)
    )
    return {"energy": total_energy, **point_outputs}


class SkalaFunctionalCallback:
    """Lazy callable implementing vibe-qc's external XC callback contract."""

    # A module purge creates a new class object, so ``isinstance`` cannot
    # identify the process-lifetime callback retained by the native registry.
    # This immutable model+adapter fingerprint lets a fresh module adopt only
    # the callback that the shipped SKALA adapter originally registered.
    _vibeqc_external_registration_id = _CALLBACK_REGISTRATION_ID

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        downloader: Callable[[str], bytes] | None = None,
        torch_module: Any | None = None,
        max_model_points_per_chunk: int | None = (
            DEFAULT_MAX_MODEL_POINTS_PER_CHUNK
        ),
    ) -> None:
        self._max_model_points_per_chunk = _validated_chunk_limit(
            max_model_points_per_chunk
        )
        self._cache_dir = cache_dir
        self._downloader = downloader
        self._torch_module = torch_module
        self._loaded: LoadedSkalaModel | None = None
        self._load_lock = threading.Lock()
        # The immutable alias family shares one native provider wrapper and
        # its mutex. Keep a Python-side lock as well so direct callback calls
        # (which bypass that wrapper) obey the same serialized model/autograd
        # evaluation contract.
        self._evaluation_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Whether this callback has deserialized its TorchScript model."""

        return self._loaded is not None

    @property
    def provenance(self) -> dict[str, str | int]:
        """Immutable model provenance, available without loading PyTorch."""

        return model_provenance()

    @property
    def max_model_points_per_chunk(self) -> int | None:
        """Target model points per complete-atom autograd graph."""

        return self._max_model_points_per_chunk

    def _get_loaded_model(self) -> LoadedSkalaModel:
        if self._loaded is None:
            with self._load_lock:
                if self._loaded is None:
                    self._loaded = load_skala_model(
                        self._cache_dir,
                        downloader=self._downloader,
                        torch_module=self._torch_module,
                    )
        return self._loaded

    def __call__(self, values: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_skala_inputs(values)
        with self._evaluation_lock:
            loaded = self._get_loaded_model()
            return evaluate_skala_model(
                loaded,
                normalized,
                torch_module=self._torch_module,
                max_model_points_per_chunk=self._max_model_points_per_chunk,
            )


def make_skala_callback(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    downloader: Callable[[str], bytes] | None = None,
    torch_module: Any | None = None,
    max_model_points_per_chunk: int | None = DEFAULT_MAX_MODEL_POINTS_PER_CHUNK,
) -> SkalaFunctionalCallback:
    """Construct a lazy CPU SKALA callback without importing PyTorch."""

    return SkalaFunctionalCallback(
        cache_dir,
        downloader=downloader,
        torch_module=torch_module,
        max_model_points_per_chunk=max_model_points_per_chunk,
    )


def _adopt_registered_callback(core_module: Any) -> Any | None:
    """Return this adapter's process-lifetime native callback, if present.

    The native external-XC registry intentionally outlives Python module
    objects.  A ``sys.modules`` purge therefore removes
    ``_REGISTERED_CALLBACKS`` without unregistering the provider.  Adoption
    is deliberately strict: every shipped name must resolve to the same
    Python callback and carry the exact SKALA fingerprint, capability record,
    HF fraction, and immutable checkpoint provenance.  A partial or foreign
    registration remains a hard collision.
    """

    lookup = getattr(core_module, "_external_functional_registration", None)
    if not callable(lookup):
        raise TypeError(
            "core module does not provide _external_functional_registration; "
            "rebuild/install the matching vibe-qc native extension"
        )

    names = (CANONICAL_FUNCTIONAL, *FUNCTIONAL_ALIASES)
    records = [lookup(name) for name in names]
    present = [record is not None for record in records]
    if not any(present):
        return None
    if not all(present):
        occupied = [name for name, found in zip(names, present) if found]
        missing = [name for name, found in zip(names, present) if not found]
        raise RuntimeError(
            "SKALA external-functional registration collision: only part of "
            f"the immutable alias set is registered (occupied={occupied}, "
            f"missing={missing})"
        )

    callbacks: list[Any] = []
    for name, raw_record in zip(names, records):
        if not isinstance(raw_record, Mapping):
            raise RuntimeError(
                "SKALA external-functional registration collision: native "
                f"registration query for {name!r} returned an invalid record"
            )
        record = raw_record
        expected_fields = {
            "hf_exchange_fraction": 0.0,
            "capability_version": EXTERNAL_XC_CAPABILITY_VERSION,
            "required_grid_profile": REQUIRED_GRID_PROFILE,
        }
        for field, expected in expected_fields.items():
            if record.get(field) != expected:
                raise RuntimeError(
                    "SKALA external-functional registration collision: "
                    f"{name!r} has {field}={record.get(field)!r}, expected "
                    f"{expected!r}"
                )
        callback = record.get("callback")
        if not callable(callback):
            raise RuntimeError(
                "SKALA external-functional registration collision: "
                f"{name!r} is not backed by an adoptable Python callback"
            )
        callbacks.append(callback)

    callback = callbacks[0]
    if any(candidate is not callback for candidate in callbacks[1:]):
        raise RuntimeError(
            "SKALA external-functional registration collision: canonical "
            "name and aliases do not share one callback"
        )
    if (
        getattr(callback, "_vibeqc_external_registration_id", None)
        != _CALLBACK_REGISTRATION_ID
    ):
        raise RuntimeError(
            "SKALA external-functional registration collision: the existing "
            "callback was not registered by this immutable SKALA adapter"
        )

    try:
        provenance = getattr(callback, "provenance", None)
    except Exception as exc:
        raise RuntimeError(
            "SKALA external-functional registration collision: the existing "
            "callback provenance could not be inspected"
        ) from exc
    expected_provenance = {
        "revision": MODEL_REVISION,
        "sha256": MODEL_SHA256,
        "filename": MODEL_FILENAME,
        "protocol_version": MODEL_PROTOCOL_VERSION,
    }
    if not isinstance(provenance, Mapping) or any(
        provenance.get(field) != expected
        for field, expected in expected_provenance.items()
    ):
        raise RuntimeError(
            "SKALA external-functional registration collision: the existing "
            "callback has different checkpoint provenance"
        )
    return callback


def register_with_core(
    core_module: Any,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    downloader: Callable[[str], bytes] | None = None,
    torch_module: Any | None = None,
    max_model_points_per_chunk: int | None = DEFAULT_MAX_MODEL_POINTS_PER_CHUNK,
) -> SkalaFunctionalCallback:
    """Register one retained lazy callback with a compatible native core."""

    define_family = getattr(
        core_module, "_define_external_functional_family", None
    )
    if not callable(define_family):
        raise TypeError(
            "core module does not provide "
            "_define_external_functional_family; rebuild/install the "
            "matching vibe-qc native extension"
        )

    key = id(core_module)
    retained = _REGISTERED_CALLBACKS.get(key)
    if retained is not None and retained[0] is core_module:
        return retained[1]

    adopted = _adopt_registered_callback(core_module)
    if adopted is not None:
        _REGISTERED_CALLBACKS[key] = (core_module, adopted)
        return adopted

    callback = make_skala_callback(
        cache_dir,
        downloader=downloader,
        torch_module=torch_module,
        max_model_points_per_chunk=max_model_points_per_chunk,
    )
    define_family(
        (CANONICAL_FUNCTIONAL, *FUNCTIONAL_ALIASES),
        callback,
        hf_exchange_fraction=0.0,
        capability_version=EXTERNAL_XC_CAPABILITY_VERSION,
        required_grid_profile=REQUIRED_GRID_PROFILE,
    )
    _REGISTERED_CALLBACKS[key] = (core_module, callback)
    return callback


__all__ = [
    "AtomGridChunk",
    "CALLBACK_INPUT_KEYS",
    "CALLBACK_OPTIONAL_INPUT_KEYS",
    "CALLBACK_OUTPUT_KEYS",
    "CALLBACK_REQUIRED_INPUT_KEYS",
    "CANONICAL_FUNCTIONAL",
    "DEFAULT_MAX_MODEL_POINTS_PER_CHUNK",
    "EXTERNAL_XC_CAPABILITY_VERSION",
    "FUNCTIONAL_ALIASES",
    "LoadedSkalaModel",
    "MODEL_FEATURES",
    "MODEL_EXPECTED_D3_SETTINGS",
    "MODEL_FIRST_DERIVATIVE_BYTES_PER_POINT",
    "MODEL_FILENAME",
    "MODEL_CARD_URL",
    "MODEL_PROTOCOL_VERSION",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "MODEL_SHA256",
    "MODEL_SIZE_BYTES",
    "REQUIRED_GRID_PROFILE",
    "MODEL_URL",
    "SOURCE_LICENSE_NOTICE_TEXT",
    "SOURCE_NOTICE_FILENAME",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "SkalaFunctionalCallback",
    "SkalaModelFormatError",
    "SkalaModelIntegrityError",
    "ensure_skala_model",
    "evaluate_skala_model",
    "load_skala_model",
    "make_skala_callback",
    "model_cache_dir",
    "model_cache_path",
    "model_provenance",
    "plan_atom_grid_chunks",
    "register_with_core",
    "slice_skala_atom_chunk",
    "skala_cache_root",
    "validate_skala_inputs",
]
