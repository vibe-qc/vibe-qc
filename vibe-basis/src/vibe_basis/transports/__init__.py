"""External-SCF job transports.

A transport ships a backend-emitted input file (e.g. CRYSTAL .d12)
to a CPU and waits for completion. Implementations:

* ``local`` — direct ``subprocess.run`` to a locally-installed
  external program. Synchronous; useful for laptop smoke tests
  and developer workflows.
* ``vq`` — submit through the vibe-queue cross-machine job queue,
  targeting a remote daemon (compute-host for this lab). Used in
  production Stage 0+ runs where each evaluation fans out to N
  per-compound jobs.
* ``ledger`` — wraps either and journals its submissions, so a
  restarted campaign driver adopts the jobs its predecessor left on
  the queue instead of submitting them again.

The transport interface is intentionally minimal: ``submit``,
``poll``, ``wait`` (concrete poll-loop in the base class),
``fetch``, plus a ``run`` convenience that chains all three.
Backends sit above transports; they don't know whether the SCF
ran locally or on a remote daemon.
"""

from .base import (
    SUCCESS_STATES,
    TERMINAL_STATES,
    JobHandle,
    JobResult,
    JobSubmitError,
    JobTimeoutError,
    Transport,
    TransportError,
)
from .ledger import LedgerTransport
from .local import LocalTransport
from .vq import VqTransport

__all__ = [
    "Transport",
    "TransportError",
    "JobSubmitError",
    "JobTimeoutError",
    "JobHandle",
    "JobResult",
    "TERMINAL_STATES",
    "SUCCESS_STATES",
    "LedgerTransport",
    "LocalTransport",
    "VqTransport",
]
