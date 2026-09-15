"""Embedded-cluster CASSCF/CASPT2 for periodic multireference (MR6-MR10).

.. warning::

   **EXPERIMENTAL / RESEARCH PROVENANCE. NOT VALIDATED FOR PRODUCTION.**

   This package landed on ``main`` 2026-08-12 to preserve the work when its
   branch was retired, not because it passed review. It is deliberately NOT
   re-exported at the ``vibeqc`` top level: there is no ``vq.embed_cluster``.
   Import it explicitly (``from vibeqc import embed``) and treat every number
   it produces as unvalidated.

   The ``handovers/HANDOVER_PERIODIC_MULTIREF.md`` orphan-branch audit of
   2026-07-15 found concrete defects, none of which have been fixed here:

   * **MR6** uses formal charges rather than the periodic-SCF-density route,
     and **mis-enumerates translated/skew cells**.
   * **MR7** omits the AIMP ``PROJOP`` projector / orthogonality terms, so the
     embedding potential is incomplete.
   * **MR8** omits NEVPT2 and hand-formats its report outside
     ``vibeqc.output``, which CLAUDE.md § 16 does not permit for new code.
   * **MR9** is a printing example, not a regression test.
   * **MR10** exercises only RHF XField, behind an OpenMolcas skip.
   * The ``[routes.embed]`` citation table is not consumed by the current
     registry, so its citations do not yet fire (CLAUDE.md § 8).

   That audit's verdict stands: this is "research provenance, not code to
   merge or cherry-pick wholesale", and each production increment should be
   **re-derived against current main** using this only as a reference.
   Anything built on top of it must close the defects above first.

The pipeline it sketches: carve a finite QM cluster from a periodic crystal,
embed it in a Madelung point-charge array, and hand it to the molecular CAS
solvers.

See ``handovers/HANDOVER_PERIODIC_MULTIREF.md`` for the roadmap, the audit,
and milestone status, and ``studies/embedded-cluster-cas/GOTCHAS.md`` for the
workstream's recorded gotchas (OpenMolcas parity basis, AIMP licensing).

Public API
----------
``carve_cluster``
    Carve a finite cluster from a ``PeriodicSystem``.
``evjen_array``
    Evjen (1932) charge-neutral boundary-corrected Madelung array.
``fitted_array``
    Derenzo-Klintenberg-Weber (2000) least-squares fitted array.
``embed_cluster``
    Carve + embed + CAS in one call (the MR8 driver).
``EmbeddedClusterResult``
    Result container for ``embed_cluster``.
``parse_aimp_file``
    Parse an OpenMolcas AIMP library file (MR7).
``resolve_aimp_library``
    Locate the EMB-AIMP file on the user's system.
``find_aimp``
    Filter AIMP entries by element / crystal.
"""

from ._aimp import AIMPEntry, find_aimp, parse_aimp_file, resolve_aimp_library
from ._aimp_ecp import build_aimp_ecp_xml, write_aimp_ecp_library
from ._carve import carve_cluster
from ._driver import EmbeddedClusterResult, embed_cluster
from ._madelung import (
    embedding_target,
    evjen_array,
    fitted_array,
    offsite_probe_ball,
)

__all__ = [
    "carve_cluster",
    "embed_cluster",
    "EmbeddedClusterResult",
    "embedding_target",
    "evjen_array",
    "fitted_array",
    "offsite_probe_ball",
    "AIMPEntry",
    "find_aimp",
    "parse_aimp_file",
    "resolve_aimp_library",
    "build_aimp_ecp_xml",
    "write_aimp_ecp_library",
]
