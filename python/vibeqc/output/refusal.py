"""The receipt a pre-compute capability refusal leaves under its stem (#351).

A ``NotImplementedError`` raised before a job's output writer exists used to
leave nothing on disk: from the manifest layer alone such a run was
indistinguishable from one never submitted. ``run_periodic_job`` writes a
minimal ``.system`` for that case since 5ee40dd; :func:`
write_preflight_refusal_receipt` is the same record for the molecular
``run_job`` (and any other runner): ``run.refusal_stage = "preflight"``,
``outputs.status = "crashed"``, ``outputs.result_status = "refused"``, the
reason in ``result_error``, and a plan that declares only the manifest.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from ._stem_paths import stem_sibling
from .plan import OutputPlan, PlannedFile
from .writer import OutputWriter

__all__ = ["write_preflight_refusal_receipt"]


def write_preflight_refusal_receipt(
    output: str | os.PathLike,
    *,
    error: BaseException,
    job_kind: str = "molecular_scf",
    method: object = "auto",
    basis: object = None,
    functional: object = None,
    record_hostname: bool = True,
    extra_run_fields: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``{stem}.system`` for a refusal raised before any output plan.

    Returns the manifest path. The caller re-raises the refusal; this
    function only records it, through the ordinary output layer so the
    receipt has the same runtime-environment block as a full manifest.
    """
    stem = Path(os.fspath(output))
    path = stem_sibling(stem, ".system")
    basis_name = (
        "<basis-free>" if basis is None else str(getattr(basis, "name", basis))
    )
    plan = OutputPlan(
        stem=stem,
        job_kind=job_kind,  # type: ignore[arg-type]
        method=str(method),
        basis=basis_name,
        functional=None if functional is None else str(functional),
        files=(
            PlannedFile(
                role="manifest",
                path=path,
                format="toml",
                always=True,
                description="Pre-compute capability refusal record.",
            ),
        ),
    )
    fields: dict[str, Any] = dict(extra_run_fields or {})
    fields["refusal_stage"] = "preflight"
    writer = OutputWriter(
        plan, record_hostname=record_hostname, extra_run_fields=fields,
    )
    writer.set_result_status(
        "refused", error=f"{type(error).__name__}: {error}"
    )
    writer.crash()
    return path
