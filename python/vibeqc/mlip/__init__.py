"""Machine-learning interatomic potential (MLIP) interfaces.

This subpackage exposes *external, pre-trained* ML interatomic
potentials as first-class energy/force engines. Unlike
:mod:`vibeqc.semiempirical` -- which is vibe-qc's own implementation --
these wrap a third-party pre-trained model: vibe-qc marshals geometry
in and reads energy / forces / stress out, attributing the model as
such. This is a maintainer-approved extension of ``CLAUDE.md`` Sec.10 to
admit pre-trained-model engines (vibe-qc computes nothing here; it does
not claim the energy as its own).

Backends
--------
- :mod:`vibeqc.mlip.mace` -- ACEsuit MACE (MIT code; foundation-model
  weights fetched on demand, MIT-ungated / ASL-gated).

The heavy ML stack (PyTorch, e3nn) is the optional ``[mace]`` extra and
each backend is import-gated, so ``import vibeqc.mlip`` stays cheap and
dependency-free; the cost -- and any missing-dependency error -- is
deferred to the point of use. The model registry + options below are
pure-Python (no torch) and always importable.
"""

from __future__ import annotations

from ._mace_models import (
    MACE_MODELS,
    MaceModelInfo,
    mace_model_registry,
    resolve_model,
    save_mace_model_registry,
)
from .options import MLIPOptions

__all__ = [
    "MLIPOptions",
    "MACE_MODELS",
    "MaceModelInfo",
    "mace_model_registry",
    "resolve_model",
    "save_mace_model_registry",
]
