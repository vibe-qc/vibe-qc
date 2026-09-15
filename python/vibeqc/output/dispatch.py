"""Role-driven writer dispatch for plan-declared artefacts.

The molecular runner routes its single-shot artefact files through
:meth:`OutputWriter.dispatch_role`; the periodic runner follows in the next
D7b milestone.  The coordinator walks the :class:`OutputPlan` and invokes the
writer registered for each declared :class:`PlannedFile`.

This module is the dispatch table: a :class:`Dispatcher` keyed by
``(role, format)`` pairs that maps each plan-declared artefact to
its writer function. The default instance comes pre-registered
with adapters for the built-in single-shot formats (molden / TREXIO / xyz /
extended-xyz / POSCAR / CIF / population / cube / XSF / QVF /
.bibtex / .references). New writers register themselves via
:meth:`Dispatcher.register`.

Adapter contract
----------------

Each registered writer is a callable with the signature::

    def writer(*, stem: Path, plan_file: PlannedFile, **context) -> Path:
        ...

* ``stem`` -- the job's path stem (e.g. ``Path("output-h2o")``).
  Comes from :attr:`OutputWriter.stem`.
* ``plan_file`` -- the :class:`PlannedFile` row driving this
  emission. Carries the target ``path`` and any per-artefact
  metadata.
* ``**context`` -- kwargs the caller passes through
  :meth:`OutputWriter.dispatch_role`. Each adapter pulls the
  fields it needs (e.g. ``result`` / ``basis`` / ``molecule`` /
  ``energy_ha``) and ignores the rest via ``**_``.
* **Returns** the on-disk path of the written file (typically
  ``plan_file.path``, but the writer may rewrite it -- the
  dispatcher uses the returned path for ``record()``).

Adapters are fail-soft by default: a writer that raises is logged and the
failure surfaces as ``None``.  Runner call sites opt into strict propagation
inside their established per-writer warning handlers, preserving both the
user-visible warning and manifest failure outcome.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .plan import OutputPlan, PlannedFile

__all__ = [
    "Dispatcher",
    "WriterAdapter",
    "default_dispatcher",
]

log = logging.getLogger("vibeqc.output.dispatch")


@runtime_checkable
class WriterAdapter(Protocol):
    """Callable shape for a dispatch-registered writer."""

    def __call__(
        self,
        *,
        stem: Path,
        plan_file: PlannedFile,
        **context: Any,
    ) -> Optional[Path]: ...


# ---------------------------------------------------------------------- #
# Dispatcher                                                             #
# ---------------------------------------------------------------------- #


class Dispatcher:
    """Routes (role, format) pairs to writer adapters.

    Construct via :func:`default_dispatcher` to get the bundled
    registrations, or instantiate empty and call :meth:`register`
    to build a custom table (useful for tests and for downstream
    packages that add their own output formats).
    """

    def __init__(self) -> None:
        self._writers: dict[tuple[str, str], WriterAdapter] = {}

    # -- registration ------------------------------------------------- #

    def register(
        self,
        role: str,
        format: str,
        fn: WriterAdapter,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a writer for the given ``(role, format)`` pair.

        Raises :class:`ValueError` if the pair is already
        registered, unless ``overwrite=True``.
        """
        key = (role, format)
        if key in self._writers and not overwrite:
            raise ValueError(
                f"dispatcher: writer for {key!r} already "
                f"registered; pass overwrite=True to replace it"
            )
        self._writers[key] = fn

    def unregister(self, role: str, format: str) -> None:
        """Remove the writer for ``(role, format)``. Silent no-op
        when the key isn't registered."""
        self._writers.pop((role, format), None)

    def get_writer(
        self,
        role: str,
        format: str,
    ) -> Optional[WriterAdapter]:
        """Return the writer registered for ``(role, format)``, or
        ``None`` when no writer is registered."""
        return self._writers.get((role, format))

    def registered_keys(self) -> tuple[tuple[str, str], ...]:
        """Snapshot of the currently-registered ``(role, format)``
        pairs. Order is insertion order."""
        return tuple(self._writers.keys())

    # -- dispatch ----------------------------------------------------- #

    def dispatch_planned_file(
        self,
        plan_file: PlannedFile,
        *,
        stem: Path,
        raise_on_error: bool = False,
        **context: Any,
    ) -> Optional[Path]:
        """Invoke the writer for one :class:`PlannedFile`.

        Returns the path the writer reported. Returns ``None`` if no
        writer is registered for the file's ``(role, format)`` pair,
        or if the writer raised an exception (the exception is
        logged at WARNING; the caller decides on the fail-loud /
        fail-soft policy). Pass ``raise_on_error=True`` to propagate
        missing registrations and writer exceptions into a runner's
        existing output-failure handler. The matching :class:`OutputWriter`
        instance is the right place to call ``record()`` on the
        returned path so the manifest's ``[[outputs.files]]`` row
        gets updated -- :meth:`OutputWriter.dispatch_role` does this
        automatically.
        """
        fn = self.get_writer(plan_file.role, plan_file.format)
        if fn is None:
            if raise_on_error:
                raise LookupError(
                    "dispatcher: no writer registered for "
                    f"({plan_file.role!r}, {plan_file.format!r}) "
                    f"(target: {plan_file.path})"
                )
            log.debug(
                "dispatcher: no writer registered for (%r, %r) -- skipping %s",
                plan_file.role,
                plan_file.format,
                plan_file.path,
            )
            return None
        try:
            return fn(stem=stem, plan_file=plan_file, **context)
        except Exception as exc:  # noqa: BLE001
            if raise_on_error:
                raise
            log.warning(
                "dispatcher: writer for (%r, %r) raised %s: %s (target: %s)",
                plan_file.role,
                plan_file.format,
                type(exc).__name__,
                exc,
                plan_file.path,
            )
            return None

    def dispatch_plan(
        self,
        plan: OutputPlan,
        *,
        only_role: Optional[str] = None,
        only_always: bool = True,
        raise_on_error: bool = False,
        **context: Any,
    ) -> list[tuple[PlannedFile, Optional[Path]]]:
        """Dispatch every :class:`PlannedFile` in a plan.

        Returns a list of ``(plan_file, written_path_or_None)``
        pairs so callers can post-process (e.g. call
        :meth:`OutputWriter.record` on the successes, warn on the
        failures).

        Parameters
        ----------
        plan
            The :class:`OutputPlan` whose declarations drive the
            dispatch.
        only_role
            If set, restrict to plan files with this role.
        only_always
            If ``True`` (default), skip conditional artefacts
            (``always=False``) -- those are typically triggered by
            external signals (e.g. crash-dump only fires on SCF
            failure).
        **context
            Forwarded verbatim to every adapter.
        """
        out: list[tuple[PlannedFile, Optional[Path]]] = []
        for pf in plan.files:
            if only_role is not None and pf.role != only_role:
                continue
            if only_always and not pf.always:
                continue
            written = self.dispatch_planned_file(
                pf,
                stem=plan.stem,
                raise_on_error=raise_on_error,
                **context,
            )
            out.append((pf, written))
        return out


# ---------------------------------------------------------------------- #
# Built-in writer adapters                                               #
# ---------------------------------------------------------------------- #
#
# Each adapter has the documented `(stem, plan_file, **context)`
# signature. They wrap the existing writer functions in their
# canonical `output/formats/` locations (post-R1...R7 relocation).
# Failures bubble up to `Dispatcher.dispatch_planned_file` which
# logs them and returns None.


def _adapter_xyz(
    *,
    stem: Path,
    plan_file: PlannedFile,
    molecule: Any,
    energy_ha: Optional[float] = None,
    **_: Any,
) -> Path:
    from .formats.xyz import write_xyz

    return write_xyz(plan_file.path, molecule, energy_ha=energy_ha)


def _adapter_extended_xyz(
    *,
    stem: Path,
    plan_file: PlannedFile,
    system: Any,
    energy_ha: Optional[float] = None,
    comment: Optional[str] = None,
    **_: Any,
) -> Path:
    from .formats.extended_xyz import write_extended_xyz

    return write_extended_xyz(
        plan_file.path,
        system,
        energy_ha=energy_ha,
        comment=comment,
    )


def _adapter_poscar(
    *,
    stem: Path,
    plan_file: PlannedFile,
    system: Any,
    comment: Optional[str] = None,
    **_: Any,
) -> Path:
    from .formats.poscar import write_poscar

    return write_poscar(plan_file.path, system, comment=comment)


def _adapter_cif(
    *,
    stem: Path,
    plan_file: PlannedFile,
    system: Any,
    comment: Optional[str] = None,
    **_: Any,
) -> Path:
    from .formats.cif import write_cif

    return write_cif(plan_file.path, system, comment=comment)


def _adapter_population(
    *,
    stem: Path,
    plan_file: PlannedFile,
    result: Any = None,
    basis: Any = None,
    molecule: Any = None,
    population_summary: Any = None,
    _dispatch_cache: Optional[dict[Any, Any]] = None,
    **_: Any,
) -> Path:
    """Write both population siblings once per role dispatch.

    The plan carries one row for ``.txt`` and one for ``.json``.  Both rows
    share the per-call cache installed by :meth:`OutputWriter.dispatch_role`,
    so population analysis and serialization run once while each planned row
    still gets its own manifest outcome.
    """
    from .formats.population import write_population, write_population_summary

    cache = _dispatch_cache if _dispatch_cache is not None else {}
    cache_key = (
        "population",
        str(stem),
        id(population_summary) if population_summary is not None else id(result),
    )
    paths = cache.get(cache_key)
    if paths is None:
        if population_summary is not None:
            paths = write_population_summary(stem, population_summary)
        else:
            paths = write_population(stem, result, basis, molecule)
        cache[cache_key] = paths
    txt_path, json_path = paths
    name = plan_file.path.name
    return json_path if name.endswith(".json") else txt_path


def _adapter_molden(
    *,
    stem: Path,
    plan_file: PlannedFile,
    result: Any,
    basis: Any,
    molecule: Any,
    title: Optional[str] = None,
    **_: Any,
) -> Path:
    from .formats.molden import write_molden

    out_path = plan_file.path
    write_molden(
        out_path,
        molecule,
        basis,
        result,
        title=title or stem.name,
    )
    return out_path


def _adapter_trexio(
    *,
    stem: Path,
    plan_file: PlannedFile,
    result: Any,
    basis: Any,
    molecule: Any,
    trexio_backend: str = "hdf5",
    trexio_description: str = "",
    uses_ecp: bool = False,
    **_: Any,
) -> Path:
    from .formats.trexio import write_trexio

    return write_trexio(
        plan_file.path,
        molecule,
        basis,
        result,
        backend=trexio_backend,
        description=trexio_description or stem.name,
        uses_ecp=uses_ecp,
    )


def _adapter_cube_density(
    *,
    stem: Path,
    plan_file: PlannedFile,
    result: Any,
    basis: Any,
    molecule: Any,
    cube_spacing: float = 0.2,
    cube_padding: float = 4.0,
    **_: Any,
) -> Path:
    from .formats.cube import write_cube_density_for_run_job

    return write_cube_density_for_run_job(
        stem,
        result,
        basis,
        molecule,
        spacing=cube_spacing,
        padding=cube_padding,
    )


def _adapter_cube_mo(
    *,
    stem: Path,
    plan_file: PlannedFile,
    result: Any,
    basis: Any,
    molecule: Any,
    mo_index: int,
    display_name: str,
    cube_spacing: float = 0.2,
    cube_padding: float = 4.0,
    **_: Any,
) -> Path:
    from .formats.cube import write_cube_mo_for_run_job

    return write_cube_mo_for_run_job(
        stem,
        result,
        basis,
        molecule,
        mo_index,
        display_name,
        spacing=cube_spacing,
        padding=cube_padding,
    )


def _adapter_xsf_structure(
    *,
    stem: Path,
    plan_file: PlannedFile,
    system: Any,
    **_: Any,
) -> Path:
    from ..xsf import write_xsf_structure

    return write_xsf_structure(plan_file.path, system)


def _adapter_xsf_volume(
    *,
    stem: Path,
    plan_file: PlannedFile,
    system: Any,
    data: Any,
    name: str = "density",
    origin: Any = None,
    span: Any = None,
    **_: Any,
) -> Path:
    from ..xsf import write_xsf_volume

    kwargs: dict[str, Any] = {"data": data, "name": name}
    if origin is not None:
        kwargs["origin"] = origin
    if span is not None:
        kwargs["span"] = span
    return write_xsf_volume(plan_file.path, system, **kwargs)


def _adapter_nto_molden(
    *,
    stem: Path,
    plan_file: PlannedFile,
    result: Any,
    basis: Any,
    molecule: Any,
    title: Optional[str] = None,
    **_: Any,
) -> Path:
    from .formats.molden import write_molden

    write_molden(
        plan_file.path,
        molecule,
        basis,
        result,
        # The direct NTO call historically omitted ``title``. Preserve the
        # Molden writer's stable fallback unless a caller explicitly supplies
        # one.
        title=title or "",
    )
    return plan_file.path


def _adapter_citations_bibtex(
    *,
    stem: Path,
    plan_file: PlannedFile,
    citations: Any,
    **_: Any,
) -> Path:
    """``citations`` is an :class:`AssembledCitations` or any
    iterable of :class:`Citation` entries (the bibtex writer accepts
    both)."""
    from .citations.bibtex import write_bibtex

    return write_bibtex(stem, citations)


def _adapter_citations_plain(
    *,
    stem: Path,
    plan_file: PlannedFile,
    citations: Any,
    **_: Any,
) -> Path:
    from .citations.plain import write_references

    return write_references(stem, citations)


def _adapter_qvf(
    *,
    stem: Path,
    plan_file: PlannedFile,
    plan: OutputPlan,
    _dispatch_cache: Optional[dict[Any, Any]] = None,
    **context: Any,
) -> Path:
    """QVF container writer -- produces a single .qvf file with
    all sections (structure, volumes, bands, spectra, trajectory,
    vibrations, citations, provenance)."""
    from .formats.qvf import write_qvf

    return write_qvf(stem, plan, **context)


# ---------------------------------------------------------------------- #
# Default-dispatcher factory                                             #
# ---------------------------------------------------------------------- #


def default_dispatcher() -> Dispatcher:
    """Return a :class:`Dispatcher` pre-registered with the
    built-in adapters.

    Covered ``(role, format)`` pairs:

    * ``("geometry", "xyz")`` -> ``write_xyz`` (plain XYZ; molecular)
    * ``("geometry", "extended-xyz")`` -> ``write_extended_xyz``
      (lattice in comment; periodic)
    * ``("geometry", "poscar")`` -> ``write_poscar``
    * ``("geometry", "cif")`` -> ``write_cif``
    * ``("geometry", "xsf")`` -> ``write_xsf_structure``
    * ``("orbitals", "molden")`` -> ``write_molden``
    * ``("orbitals", "nto-molden")`` -> ``write_molden``
    * ``("orbitals", "trexio")`` -> ``write_trexio``
    * ``("population", "text")`` / ``("population", "json")`` ->
      ``write_population`` (returns the matching sibling)
    * ``("density", "cube")`` -> ``write_cube_density_for_run_job``
    * ``("density", "xsf")`` -> ``write_xsf_volume``
    * ``("orbital_vol", "cube")`` -> ``write_cube_mo_for_run_job``
    * ``("citations", "bibtex")`` -> ``write_bibtex``
    * ``("citations", "text")`` -> ``write_references``
    * ``("qvf", "qvf")`` -> ``write_qvf``

    Roles deliberately *not* covered today, with the rationale:

    * ``("log", "text")`` (the ``.out`` file) -- assembled from many
      partial writes (banner + iter rows + properties + references);
      a single-call adapter doesn't fit its streaming shape. It is
      instead served by :mod:`vibeqc.output.channel`: the runner
      installs an :class:`~vibeqc.output.channel.OutputChannel` and any
      module emits into it via ``vibeqc.output.write``. A dispatch
      adapter would still not fit, because the ``.out`` is a stream,
      not a file produced by one call.
    * ``("manifest", "toml")`` (the ``.system`` file) -- written by
      the :class:`OutputWriter` itself via :class:`ManifestUpdater`,
      not by an adapter call.
    * ``("structured", "ndjson")`` / ``("perf", "text")`` --
      stream-style writers that need a live handle for the duration
      of the run, not a single-shot write. Their integration is the
      next dispatcher step (D2).
    * ``("crash", "toml")`` -- conditional (only on failure); the
      caller fires it explicitly from the exception path, not via
      the always-on dispatch loop.
    """
    d = Dispatcher()
    d.register("geometry", "xyz", _adapter_xyz)
    d.register("geometry", "extended-xyz", _adapter_extended_xyz)
    d.register("geometry", "poscar", _adapter_poscar)
    d.register("geometry", "cif", _adapter_cif)
    d.register("geometry", "xsf", _adapter_xsf_structure)
    d.register("orbitals", "molden", _adapter_molden)
    d.register("orbitals", "nto-molden", _adapter_nto_molden)
    d.register("orbitals", "trexio", _adapter_trexio)
    d.register("population", "text", _adapter_population)
    d.register("population", "json", _adapter_population)
    d.register("population", "bipole-text", _adapter_population)
    d.register("population", "bipole-json", _adapter_population)
    d.register("population", "aiccm2026dev-b-text", _adapter_population)
    d.register("population", "aiccm2026dev-b-json", _adapter_population)
    d.register("density", "cube", _adapter_cube_density)
    d.register("density", "xsf", _adapter_xsf_volume)
    d.register("orbital_vol", "cube", _adapter_cube_mo)
    d.register("citations", "bibtex", _adapter_citations_bibtex)
    d.register("citations", "text", _adapter_citations_plain)
    d.register("qvf", "qvf", _adapter_qvf)
    # Compatibility alias for the pre-D7b prototype.  OutputPlan has always
    # declared role="qvf"; downstream custom plans may still carry the old
    # provisional role while migrating.
    d.register("container", "qvf", _adapter_qvf)
    return d
