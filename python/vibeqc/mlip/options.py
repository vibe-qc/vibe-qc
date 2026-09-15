"""Options for machine-learning interatomic potential (MLIP) runs."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class MLIPOptions:
    """Configuration for a ``method="mace"`` (MLIP) job.

    Parameters
    ----------
    model
        Foundation-model key (see :data:`vibeqc.mlip.MACE_MODELS`) or an
        alias (``"mpa-0"``, ``"off23"``, ...). Default: the MIT-licensed
        MACE-MPA-0. ASL (academic, non-commercial) models require
        ``accept_academic_license`` (or the ``VIBEQC_ACCEPT_ASL`` env var).
    device
        Torch device -- ``"cpu"`` (default), ``"mps"``, or ``"cuda"``.
    dtype
        ``"float64"`` (default; recommended for energies / geometry) or
        ``"float32"`` (faster, for MD).
    accept_academic_license
        Set ``True`` to acknowledge the Academic Software License (ASL)
        when selecting an ASL model -- i.e. that your use is academic and
        non-commercial. Equivalent to setting ``VIBEQC_ACCEPT_ASL=1``.
        MIT-licensed models ignore this flag.
    """

    model: str = "medium-mpa-0"
    device: str = "cpu"
    dtype: str = "float64"
    accept_academic_license: bool = False

    def academic_license_acknowledged(self) -> bool:
        """True if the ASL is acknowledged via this flag or the
        ``VIBEQC_ACCEPT_ASL`` environment variable."""
        if self.accept_academic_license:
            return True
        env = os.environ.get("VIBEQC_ACCEPT_ASL", "")
        return env.strip().lower() not in ("", "0", "false", "no", "off")
