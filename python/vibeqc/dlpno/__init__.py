"""DLPNO -- Domain-based Local Pair Natural Orbital methods.

Local correlation in the Neese-group DLPNO framework
(Riplinger & Neese, J. Chem. Phys. 138, 034106, 2013;
Pinski, Riplinger, Valeev, Neese, J. Chem. Phys. 143, 034108, 2015).

Pipeline
--------
1. Occupied orbital localisation (Foster-Boys) -- in `vibeqc.localise`
2. Pair classification (strong / weak / distant) -- `pairs.py`
3. PAO construction, domains, semicanonical virtuals -- `pao.py`
4. **DLPNO-MP2** -- `mp2.py` (`run_dlpno_mp2`): PNO construction +
   coupled LMP2, validated against canonical DF-MP2 (exactness limit
   <=1 µHa; `tests/test_dlpno_mp2.py`). Open-shell
   (UHF-reference) `ump2.py` (`run_dlpno_ump2`): aa/bb/ab spin-channel
   UMP2, exact vs canonical UMP2 (M1 canonical-occupied PNOs; M1b
   Boys-localised coupled residual; M1c Boys + per-pair PNO truncation
   via cross-pair projections; `tests/test_dlpno_ump2.py`)
5. **DLPNO-CCSD pilot** -- `ccsd.py` (`run_dlpno_ccsd_pilot`):
   subspace-projected CCSD through the FCI-anchored spin-orbital
   engine `_ccsd_ref.py` (`tests/test_dlpno_ccsd.py`). Open-shell
   (UHF-reference) `uccsd.py` (`run_dlpno_uccsd_pilot`): spin-orbital
   subspace-projected UCCSD(T), exact vs the spin-orbital `run_ref_uccsd`
   anchor at full domains (`tests/test_dlpno_uccsd.py`)
6. **Local open-shell UCCSD engine** -- `uccsd_local_solver.py`
   (`run_local_dlpno_uccsd`): per-pair PNO/local-occupied assembly around
   the native residual kernel, exact vs the dense pilot at full domains;
   NormalPNO per-spin PAO pair domains and weak-pair screening,
   singles-specific natural orbitals, distinct-triple TNO `(T0)`, and opt-in
   spin-block-semicanonical `(T1)` occupied coupling. Open-shell
   `dlpno-ccsd` and `dlpno-ccsd(t)` jobs dispatch here; callers can select the
   dense pilot explicitly as a correctness oracle.
"""

from __future__ import annotations

from .mp2 import DLPNOMP2Options, DLPNOMP2Result, run_dlpno_mp2
from .pno_density import PNO_NORMS, pair_density, resolve_pno_norm
from .thresholds import (
    DLPNO_DEFAULTS,
    DLPNO_LOOSE,
    DLPNO_TIGHT,
    DLPNOThresholdProvenance,
    DLPNOThresholds,
    apply_dlpno_thresholds,
    describe_dlpno_thresholds,
    options_from_dlpno_thresholds,
    resolve_dlpno_thresholds,
)
from .uccsd import (
    DLPNOUCCSDPilotOptions,
    DLPNOUCCSDPilotResult,
    run_dlpno_uccsd_pilot,
)
from .uccsd_local_solver import (
    LocalUCCSDOptions,
    LocalUCCSDResult,
    run_local_dlpno_uccsd,
)
from .ump2 import DLPNOUMP2Options, DLPNOUMP2Result, run_dlpno_ump2

__all__ = [
    "DLPNOThresholds",
    "DLPNOThresholdProvenance",
    "DLPNO_DEFAULTS",
    "DLPNO_TIGHT",
    "DLPNO_LOOSE",
    "resolve_dlpno_thresholds",
    "apply_dlpno_thresholds",
    "describe_dlpno_thresholds",
    "options_from_dlpno_thresholds",
    "DLPNOMP2Options",
    "DLPNOMP2Result",
    "run_dlpno_mp2",
    "PNO_NORMS",
    "pair_density",
    "resolve_pno_norm",
    "DLPNOUMP2Options",
    "DLPNOUMP2Result",
    "run_dlpno_ump2",
    "DLPNOUCCSDPilotOptions",
    "DLPNOUCCSDPilotResult",
    "run_dlpno_uccsd_pilot",
    "LocalUCCSDOptions",
    "LocalUCCSDResult",
    "run_local_dlpno_uccsd",
]
