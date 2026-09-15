"""``vibeqc.output`` -- unified output, logging, and citation surface.

The thin-layer phase (v0.8.x -> pre-v1.0) introduces the public surface
that the eventual full coordinator rewrite will preserve verbatim:

* :class:`OutputPlan` + :class:`PlannedFile` -- declarative pre-flight
  contract for every artefact a job will write.
* :class:`OutputWriter` -- per-job coordinator. Constructs the
  :class:`~vibeqc.output.manifest.ManifestUpdater` for ``{stem}.system``
  and tracks artefact completion via ``record(...)``.
* :mod:`vibeqc.output.citations` -- TOML-backed citation database +
  ``.bibtex`` / ``.references`` writers (see the submodule's docstring).

The thin layer leaves existing writer modules in place
(:mod:`vibeqc.scf_log`, :mod:`vibeqc.io.molden`,
:mod:`vibeqc.system_info`, :mod:`vibeqc.structured_log`,
:mod:`vibeqc.perf`, :mod:`vibeqc.crash_dump`,
:mod:`vibeqc.io.trajectory`). They each remain responsible for the
bytes of their respective files; the new ``OutputWriter`` is the
coordinator that knows the *set* of files and the manifest status.

The pre-v1.0 refactor will move those modules under
``vibeqc.output.formats`` and replace ``run_job``'s ad-hoc dispatch
with a generic ``OutputWriter.dispatch_all(...)`` driven by the plan's
roles. Public API surface (the names re-exported from this module +
the file-format identities + the ``.system`` schema) is being designed
to survive that move unchanged. See ``docs/design_output_module.md``.
"""

from __future__ import annotations

from .channel import (
    Level,
    OutputChannel,
    active_channel,
    flush,
    note,
    output_level,
    record,
    section,
    section_header,
    strict_output,
    warn,
    write,
)
from ._cpp_diagnostics import (
    install_diagnostics_bridge,
    install_progress_handler,
)
from .dispatch import (
    Dispatcher,
    WriterAdapter,
    default_dispatcher,
)
from .document import (
    DEFAULT_POLICY,
    Column,
    CriteriaTable,
    Criterion,
    HeaderlessBlock,
    HeaderlessColumn,
    FormatPolicy,
    FormatSpec,
    OutputDocument,
    Quantity,
    Table,
    active_policy,
    output_units,
    set_active_policy,
    render_duration,
    render_energy,
    render_energy_labeled,
    render_frequency,
    render_temperature,
)
from .dry_run import (
    dry_run_manifest,
    is_dry_run_requested,
    is_dry_run_estimate_requested,
    print_dry_run_summary,
)
from .formats.structured_log import (
    StructuredLog,
    active_structured_log,
    emit,
    structured_log,
)
from .formats import (
    CubeRequest,
    parse_write_cube_kwarg,
    qvf_bytes,
    read_trexio,
    requested_mo_indices,
    validate_qvf,
    write_cif,
    write_cube_density_for_run_job,
    write_cube_mo_for_run_job,
    write_extended_xyz,
    write_population,
    write_poscar,
    write_qvf,
    write_trexio,
    write_xyz,
)
from .manifest import (
    FileOutcome,
    IncompleteOutputError,
    ManifestStatus,
    ManifestUpdater,
    write_initial_manifest,
)
from .outputs_cli import main as outputs_cli_main
from .plan import (
    OutputFormat,
    OutputPlan,
    OutputRole,
    PlannedFile,
)
from .writer import OutputWriter

__all__ = [
    # channel.py
    "Level",
    "OutputChannel",
    "active_channel",
    "flush",
    "output_level",
    "note",
    "record",
    "section",
    "section_header",
    "strict_output",
    "warn",
    "write",
    # _cpp_diagnostics.py
    "install_diagnostics_bridge",
    "install_progress_handler",
    # formats/structured_log.py -- re-exported here as that module's
    # docstring has always promised ("the top-level vibeqc.output
    # re-exports"), and as CLAUDE.md Sec. 16 tells other chats to import.
    "StructuredLog",
    "active_structured_log",
    "emit",
    "structured_log",
    # document.py
    "DEFAULT_POLICY",
    "Column",
    "CriteriaTable",
    "Criterion",
    "HeaderlessBlock",
    "HeaderlessColumn",
    "FormatPolicy",
    "FormatSpec",
    "OutputDocument",
    "Quantity",
    "Table",
    "active_policy",
    "output_units",
    "set_active_policy",
    "render_duration",
    "render_energy",
    "render_energy_labeled",
    "render_frequency",
    "render_temperature",
    # plan.py
    "OutputFormat",
    "OutputPlan",
    "OutputRole",
    "PlannedFile",
    # manifest.py
    "FileOutcome",
    "IncompleteOutputError",
    "ManifestStatus",
    "ManifestUpdater",
    "write_initial_manifest",
    # writer.py
    "OutputWriter",
    # dispatch.py (pre-v1.0 D1)
    "Dispatcher",
    "WriterAdapter",
    "default_dispatcher",
    # formats/
    "write_xyz",
    "write_trexio",
    "read_trexio",
    "write_extended_xyz",
    "write_poscar",
    "write_cif",
    "write_population",
    "write_qvf",
    "qvf_bytes",
    "validate_qvf",
    "CubeRequest",
    "parse_write_cube_kwarg",
    "requested_mo_indices",
    "write_cube_density_for_run_job",
    "write_cube_mo_for_run_job",
    # dry_run.py
    "dry_run_manifest",
    "is_dry_run_requested",
    "is_dry_run_estimate_requested",
    "print_dry_run_summary",
    # outputs_cli.py
    "outputs_cli_main",
]
