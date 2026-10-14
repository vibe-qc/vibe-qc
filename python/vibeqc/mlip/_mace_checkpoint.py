"""Load registered MACE weights from the same bytes their digest describes.

Checkpoint URLs follow ACEsuit's MP-0, MPA-0 and OFF23 model releases.
Weights remain external, fetched on demand after the wrapper's license gate.
The digest identifies the loaded checkpoint, not an endorsement of its
scientific accuracy or a hash of a subsequently modified PyTorch model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import tempfile
import urllib.request

from ._mace_models import resolve_model


_MP_RELEASE = "https://github.com/ACEsuit/mace-mp/releases/download/"
_OFF_RELEASE = "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/"
_CHECKPOINT_URLS = {
    "medium-mpa-0": _MP_RELEASE + "mace_mpa_0/mace-mpa-0-medium.model",
    "small": _MP_RELEASE + "mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
    "medium": _MP_RELEASE + "mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
    "large": _MP_RELEASE + "mace_mp_0/MACE_MPtrj_2022.9.model",
    "off23-small": _OFF_RELEASE + "MACE-OFF23_small.model",
    "off23-medium": _OFF_RELEASE + "MACE-OFF23_medium.model",
    "off23-large": _OFF_RELEASE + "MACE-OFF23_large.model",
}


@dataclass(frozen=True)
class CheckpointIdentity:
    sha256: str
    size_bytes: int
    registry_url: str
    loader: str = "vibeqc.mlip._mace_checkpoint"


def _download(url: str, target: Path) -> None:
    """Publish only a complete download; each caller owns its temporary file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
            if content_type.strip().lower() == "text/html":
                raise RuntimeError("MACE checkpoint download returned an HTML page")
            expected = response.headers.get("Content-Length")
            expected = int(expected) if expected is not None else None
            size = 0
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".mace-", delete=False) as output:
                temporary = Path(output.name)
                while block := response.read(256 * 1024):
                    output.write(block)
                    size += len(block)
            if size == 0 or (expected is not None and size != expected):
                raise RuntimeError("MACE checkpoint download is empty or incomplete")
        # A concurrent completed download may already have populated the
        # target. Linking without replacement preserves that caller's file.
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _checkpoint_path(info) -> Path:
    from .mace import mace_cache_root

    if info.key == "medium-mpa-0":
        # Upstream distributions may include this registered checkpoint.
        # Preserve their existing preference without copying it into a cache.
        import mace.calculators.foundations_models as foundations

        bundled = Path(foundations.local_model_path)
        if bundled.is_file():
            return bundled
    url = _CHECKPOINT_URLS[info.key]
    filename = url.rsplit("/", 1)[-1]
    if info.loader == "mace_mp":
        filename = "".join(c for c in filename if c.isalnum() or c == "_")
    target = mace_cache_root() / filename
    if not target.is_file():
        _download(url, target)
    return target


def _load(info, *, device, default_dtype, return_raw_model=False,
          expected_sha256=None, accept_academic_license=False):
    if info.is_academic_only and not accept_academic_license:
        raise PermissionError("MACE OFF23 requires acknowledgment of the Academic Software License")
    path = _checkpoint_path(info)
    # Read once. File replacement or mutation after this point cannot change
    # the bytes that torch consumes or the identity attached to the result.
    payload = path.read_bytes()
    identity = CheckpointIdentity(hashlib.sha256(payload).hexdigest(), len(payload),
                                  _CHECKPOINT_URLS[info.key])
    if not payload:
        raise ValueError("MACE checkpoint is empty")
    if expected_sha256 is not None and identity.sha256 != expected_sha256.lower():
        raise ValueError("MACE checkpoint SHA256 does not match the requested weight_sha256")
    import torch
    from mace.calculators import MACECalculator

    # Registered foundation checkpoints are serialized modules, not plain
    # state dictionaries. This matches the upstream loader's pickle format.
    model = torch.load(io.BytesIO(payload), map_location=device, weights_only=False)
    model._vibeqc_checkpoint_identity = identity
    if return_raw_model:
        return model
    calculator = MACECalculator(models=model, device=device, default_dtype=default_dtype)
    calculator._vibeqc_checkpoint_identity = identity
    return calculator


def load_mp(*, model, **kwargs):
    info = resolve_model(model)
    if info.loader != "mace_mp":
        raise ValueError("Expected a registered MACE materials checkpoint")
    return _load(info, **kwargs)


def load_off(*, model, **kwargs):
    return _load(resolve_model("off23-" + model), **kwargs)
