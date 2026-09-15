"""Declarative output plan -- "what files will this job produce?".

A :class:`OutputPlan` is the contract that lets the ``vq`` queue, the
user (via ``--vibeqc-dry-run``), and CI know in advance which sibling
files a vibe-qc job is going to emit alongside ``{stem}.out``. The plan
is built at job start, *before* any compute, and is serialised into the
``[plan]`` section of ``{stem}.system`` so external tools can read it
off the workspace.

Each declared file is a :class:`PlannedFile` with a stable ``role``
(``"log"``, ``"orbitals"``, ``"citations"``, ...), a target ``path``, a
``format`` identifier, and an ``always`` flag distinguishing guaranteed
outputs from conditional ones (a ``.dump`` only appears on failure;
a ``.traj`` only when ``optimize=True``).

This module is part of the v0.8.x -> v1.0 thin-layer transitional shape
for ``vibeqc.output``. The public API documented here (``OutputPlan``,
``PlannedFile``, ``OutputRole``, ``OutputFormat``, the factory
``OutputPlan.from_run_job_kwargs``) is designed to survive the pre-v1.0
coordinator rewrite -- internals under ``vibeqc.output._adapters`` are
explicitly transitional. See ``docs/design_output_module.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
from ._stem_paths import stem_sibling

__all__ = [
    "OutputRole",
    "OutputFormat",
    "PlannedFile",
    "OutputPlan",
]


def _resolve_sidecar_request(
    requested: bool | None,
    *,
    supported: bool,
    option: str,
    method: str,
    caller: str,
    unavailable_reason: str,
) -> bool:
    """Resolve a tri-state runner sidecar request against route capability.

    ``None`` is the public runner default and means "emit when this route can
    produce the format truthfully".  Explicit ``True`` is a guarantee: an
    unsupported route must fail before calculation rather than declare an
    artefact that can never be written.  ``False`` remains an unconditional
    opt-out.

    This helper is intentionally private to the runner/plan seam.  The stable
    public contract is the tri-state runner keyword, not a second planning API.
    """
    if requested is None:
        return bool(supported)
    if not isinstance(requested, bool):
        raise TypeError(
            f"{caller}: {option} must be True, False, or None (auto); "
            f"got {type(requested).__name__}."
        )
    if requested and not supported:
        raise NotImplementedError(
            f"{caller}: {option}=True is not supported for "
            f"method={method!r}: {unavailable_reason} Pass None (the "
            "default, capability-aware auto mode) or False."
        )
    return requested


OutputRole = Literal[
    "log",  # .out
    "manifest",  # .system
    "orbitals",  # .molden, .fchk
    "geometry",  # .xyz, .cif, POSCAR, .xsf
    "density",  # .cube (density)
    "orbital_vol",  # .cube (per-orbital)
    "population",  # .population.{txt,json}
    "citations",  # .bibtex, .references
    "trajectory",  # .traj
    "perf",  # .perf
    "structured",  # .scf.jsonl
    "crash",  # .dump + .dump.*.npy
    "checkpoint",  # .h5 (P4)
    "bands",  # .bands.dat, .bands.gnuplot
    "qvf",  # .qvf (QVF visualisation archive)
]

OutputFormat = Literal[
    "text",
    "toml",
    "json",
    "ndjson",
    "bipole-text",
    "bipole-json",
    "aiccm2026dev-b-text",
    "aiccm2026dev-b-json",
    "molden",
    "nto-molden",
    "trexio",
    "xyz",
    "extended-xyz",
    "cif",
    "poscar",
    "xsf",
    "cube",
    "ase-traj",
    "bibtex",
    "fchk",
    "npy",
    "hdf5",
    "gnuplot",
    "qvf",
]


@dataclass(frozen=True)
class PlannedFile:
    """One declared output artefact.

    Frozen so a plan emitted at job start cannot be quietly mutated.
    Runtime status (``written`` / ``bytes`` / ``sha256`` / ``wall_time_s``)
    lives in the ``[outputs]`` section of the manifest, not here -- this
    object is the pre-flight *declaration*. The matching post-hoc result
    is :class:`vibeqc.output.manifest.FileOutcome`.
    """

    role: OutputRole
    path: Path
    format: OutputFormat
    always: bool
    description: str

    def to_toml_table(self) -> dict[str, Any]:
        """Return a plain-Python mapping suitable for the manifest's
        ``[[plan.files]]`` array-of-tables emitter. Path is stringified
        (TOML has no native Path type)."""
        return {
            "role": str(self.role),
            "path": str(self.path),
            "format": str(self.format),
            "always": bool(self.always),
            "description": str(self.description),
        }

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping for ``--vibeqc-dry-run``
        and other tools that prefer JSON over TOML."""
        return self.to_toml_table()


@dataclass(frozen=True)
class OutputPlan:
    """The full pre-flight declaration of what a job will write.

    Built by :meth:`from_run_job_kwargs` from the same kwargs that
    ``vibeqc.runner.run_job`` consumes, so a dry-run pass that
    short-circuits before compute can construct the plan with no
    additional information.
    """

    stem: Path
    job_kind: Literal[
        "molecular_scf",
        "periodic_scf",
        "opt",
        "hessian",
        "post_scf",
        "neb",
    ]
    method: str
    basis: str
    functional: str | None
    files: tuple[PlannedFile, ...]
    options_digest: str = ""

    # ------------------------------------------------------------------ #
    # Factories                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_run_job_kwargs(
        cls,
        *,
        output: str | os.PathLike,
        method: str,
        basis: str,
        functional: str | None,
        optimize: bool = False,
        write_molden_file: bool = True,
        perf_log: bool | str | os.PathLike | None = None,
        structured_log: bool | str | os.PathLike | None = False,
        crash_dump: bool | str | os.PathLike | None = True,
        citations: bool = True,
        write_xyz: bool = True,
        # Phase O6 kwargs: population dump + volumetric cubes.
        write_population: bool = True,
        population_variant: str = "standard",
        write_cube_density: bool = False,
        cube_mo_labels: tuple[str | int, ...] = (),
        # Periodic-mode kwargs (Phase O5). Ignored when
        # job_kind="molecular_scf"; honoured when job_kind="periodic_scf".
        write_poscar: bool = False,
        write_xsf_structure: bool = False,
        write_density_xsf: bool = False,
        write_cif: bool = False,
        options_digest: str = "",
        job_kind: str = "molecular_scf",
        # QVF visualisation archive (v1).
        output_qvf: bool = True,
        # TREXIO wavefunction container (#573): the resolved artefact path,
        # or None when not requested. Opt-in, so no default-on row.
        write_trexio_file: str | os.PathLike | None = None,
    ) -> "OutputPlan":
        """Build the canonical plan from ``run_job`` kwargs.

        Mirrors the kwarg surface of :func:`vibeqc.runner.run_job` so the
        same call can be used pre-flight (dry-run) and at job start. The
        list of declared files matches the actual files ``run_job``
        will emit, in the order they appear in the manifest.

        Parameters mirror ``run_job`` for the artefact-producing kwargs
        only. Verbosity / progress kwargs (``verbose``, ``progress``,
        ``use_logging``, ...) do not affect the declared file set and so
        are not consumed here.
        """
        stem = Path(os.fspath(output))
        files: list[PlannedFile] = []

        # Always-on artefacts.
        files.append(
            PlannedFile(
                role="log",
                path=stem_sibling(stem, ".out"),
                format="text",
                always=True,
                description="Human-readable SCF log (banner, iter table, "
                "orbital block, references).",
            )
        )
        files.append(
            PlannedFile(
                role="manifest",
                path=stem_sibling(stem, ".system"),
                format="toml",
                always=True,
                description="Runtime manifest -- hardware, libs, plan, outputs status.",
            )
        )

        # Default-on artefacts (opt-out kwargs).
        if write_molden_file:
            files.append(
                PlannedFile(
                    role="orbitals",
                    path=stem_sibling(stem, ".molden"),
                    format="molden",
                    always=True,
                    description="Molecular orbitals -- coefficients, "
                    "energies, occupations.",
                )
            )
        if write_trexio_file is not None:
            files.append(
                PlannedFile(
                    role="orbitals",
                    path=Path(os.fspath(write_trexio_file)),
                    format="trexio",
                    always=True,
                    description="TREXIO wavefunction container -- nuclei, "
                    "Gaussian basis, AO conventions, MOs, one-electron "
                    "integrals (HDF5 file or text directory).",
                )
            )
        if write_xyz:
            # Periodic jobs emit Extended XYZ (lattice in the comment
            # line via the ASE convention); molecular jobs emit plain
            # XYZ. Both share the same ``.xyz`` path -- the format
            # field disambiguates which writer the dispatcher routes
            # to ((geometry, xyz) -> write_xyz vs (geometry,
            # extended-xyz) -> write_extended_xyz). Without this
            # branch, the dispatcher's geometry sweep on a periodic
            # plan would try to invoke the molecular writer with
            # ``system=`` instead of ``molecule=`` and silently drop
            # the file on the floor with a TypeError-swallowed
            # warning.
            xyz_fmt = "extended-xyz" if job_kind == "periodic_scf" else "xyz"
            xyz_desc = (
                "Final geometry -- Extended XYZ with lattice in "
                "the comment line (ASE convention)."
                if job_kind == "periodic_scf"
                else "Final geometry in Angstrom."
            )
            files.append(
                PlannedFile(
                    role="geometry",
                    path=stem_sibling(stem, ".xyz"),
                    format=xyz_fmt,  # type: ignore[arg-type]
                    always=True,
                    description=xyz_desc,
                )
            )
        if citations:
            files.append(
                PlannedFile(
                    role="citations",
                    path=stem_sibling(stem, ".bibtex"),
                    format="bibtex",
                    always=True,
                    description="BibTeX entries for every method, basis, "
                    "and library cited.",
                )
            )
            files.append(
                PlannedFile(
                    role="citations",
                    path=stem_sibling(stem, ".references"),
                    format="text",
                    always=True,
                    description="Plain-text reference list (Chicago-ish formatting).",
                )
            )

        # Conditional artefacts.
        if optimize:
            files.append(
                PlannedFile(
                    role="trajectory",
                    path=stem_sibling(stem, ".traj"),
                    format="ase-traj",
                    # Only the ASE optimizer writes this sibling. Native,
                    # Brent, and geomopt routes retain their histories for QVF
                    # but do not promise an ASE .traj file.
                    always=False,
                    description=(
                        "ASE trajectory -- emitted when the selected "
                        "optimizer provides one frame per step."
                    ),
                )
            )

        # Phase O6 artefacts. Population dump is default-on for
        # molecular jobs (cheap, always useful); volumetric cubes are
        # opt-in (the grid evaluation can be expensive).
        if write_population:
            if population_variant == "standard":
                population_text_format = "text"
                population_json_format = "json"
            elif population_variant in ("bipole", "aiccm2026dev-b"):
                population_text_format = f"{population_variant}-text"
                population_json_format = f"{population_variant}-json"
            else:
                raise ValueError(
                    "population_variant must be 'standard', 'bipole', or "
                    f"'aiccm2026dev-b'; got {population_variant!r}"
                )
            files.append(
                PlannedFile(
                    role="population",
                    path=stem.parent / (stem.name + ".population.txt"),
                    format=population_text_format,  # type: ignore[arg-type]
                    always=True,
                    description=(
                        "Available atomic populations / bond properties / "
                        "dipole -- tab-separated; unavailable analyses carry "
                        "explicit N/A markers."
                    ),
                )
            )
            files.append(
                PlannedFile(
                    role="population",
                    path=stem.parent / (stem.name + ".population.json"),
                    format=population_json_format,  # type: ignore[arg-type]
                    always=True,
                    description=(
                        "Available population / properties dump -- JSON form "
                        "with structured errors for unavailable analyses."
                    ),
                )
            )
        if write_cube_density:
            files.append(
                PlannedFile(
                    role="density",
                    path=stem.parent / (stem.name + ".density.cube"),
                    format="cube",
                    always=True,
                    description="Total electron density on a uniform grid "
                    "(Gaussian cube).",
                )
            )
        for label in cube_mo_labels:
            display_label = f"mo_{label}" if isinstance(label, int) else str(label)
            files.append(
                PlannedFile(
                    role="orbital_vol",
                    path=stem.parent / (stem.name + f".{display_label}.cube"),
                    format="cube",
                    always=True,
                    description=(
                        f"MO {display_label} on a uniform grid (Gaussian cube)."
                    ),
                )
            )

        # Periodic-mode artefacts (Phase O5). Only relevant when the
        # caller is the periodic-runner wrapper; molecular runs leave
        # these kwargs at their False defaults.
        if write_poscar:
            files.append(
                PlannedFile(
                    role="geometry",
                    path=stem_sibling(stem, ".POSCAR"),
                    format="poscar",
                    always=True,
                    description="VASP-5 POSCAR -- lattice + fractional "
                    "coords for VASP / pymatgen / ASE.",
                )
            )
        if write_xsf_structure:
            # The density writer owns ``{stem}.xsf``.  When both XSF
            # artefacts are requested, keep the structure in the existing
            # ``{stem}.structure.xsf`` sibling so neither writer overwrites
            # the other.
            xsf_structure_path = (
                stem_sibling(stem, ".structure.xsf")
                if write_density_xsf
                else stem_sibling(stem, ".xsf")
            )
            files.append(
                PlannedFile(
                    role="geometry",
                    path=xsf_structure_path,
                    format="xsf",
                    always=True,
                    description="XSF crystal structure -- lattice + atoms, "
                    "opens in VESTA / XCrySDen.",
                )
            )
        if write_cif:
            files.append(
                PlannedFile(
                    role="geometry",
                    path=stem_sibling(stem, ".cif"),
                    format="cif",
                    always=True,
                    description="Crystallographic Information File "
                    "(IUCr standard, P 1 cell) -- opens in "
                    "pymatgen / ASE / VESTA / "
                    "Materials Project.",
                )
            )
        if write_density_xsf:
            files.append(
                PlannedFile(
                    role="density",
                    path=stem_sibling(stem, ".xsf"),
                    format="xsf",
                    always=True,
                    description="SCF electron density on a primitive-cell "
                    "grid (XSF DATAGRID_3D).",
                )
            )

        if perf_log:
            # The perf writer renders the same tracker as either prose
            # or JSON, selected by the target's suffix. Ask it which,
            # rather than restating "text" here -- a manifest that
            # declares the wrong format is what a harvest reads before
            # it opens the file.
            # Function-local import, matching dispatch.py's convention
            # for reaching into .formats from a module the formats
            # subpackage may itself be imported alongside.
            from .formats.perf import perf_artefact_format

            perf_path = _resolve_optional_path(perf_log, stem, ".perf")
            perf_fmt = perf_artefact_format(perf_path)
            files.append(
                PlannedFile(
                    role="perf",
                    path=perf_path,
                    format=perf_fmt,  # type: ignore[arg-type]
                    always=True,
                    description="Post-mortem performance breakdown -- "
                    "phase wall + CPU times, memory snapshots.",
                )
            )

        if structured_log:
            files.append(
                PlannedFile(
                    role="structured",
                    path=_resolve_optional_path(structured_log, stem, ".scf.jsonl"),
                    format="ndjson",
                    always=True,
                    description="One JSON record per SCF transition "
                    "(banner, iter, converged, properties, "
                    "job_end).",
                )
            )

        # Crash dump is conditional on failure -- always declared so vq
        # can advise the user *"may produce: x.dump (only on SCF
        # failure)"*, but not guaranteed.
        if crash_dump:
            files.append(
                PlannedFile(
                    role="crash",
                    path=stem_sibling(stem, ".dump"),
                    format="toml",
                    always=False,
                    description="Post-mortem snapshot -- only written on "
                    "SCF failure or max-iter without "
                    "convergence.",
                )
            )

        # QVF visualisation archive -- always-on when requested.
        if output_qvf:
            files.append(
                PlannedFile(
                    role="qvf",
                    path=stem_sibling(stem, ".qvf"),
                    format="qvf",
                    always=True,
                    description="QVF visualisation archive -- zip with "
                    "manifest.json + typed binary/text payloads.",
                )
            )

        return cls(
            stem=stem,
            job_kind=job_kind,  # type: ignore[arg-type]
            method=method.upper(),
            basis=basis,
            functional=functional,
            files=tuple(files),
            options_digest=options_digest
            or _default_options_digest(
                method=method,
                basis=basis,
                functional=functional,
            ),
        )

    # ------------------------------------------------------------------ #
    # Serialisation                                                      #
    # ------------------------------------------------------------------ #

    def to_toml_section(self) -> dict[str, Any]:
        """Return a plain-Python mapping suitable for the manifest's
        ``[plan]`` section emitter. The returned dict carries:

        * ``stem``, ``job_kind``, ``method``, ``basis``, ``functional``,
          ``options_digest`` as scalar keys.
        * ``files`` as a list of dicts for the ``[[plan.files]]``
          array-of-tables.
        """
        return {
            "stem": str(self.stem),
            "job_kind": str(self.job_kind),
            "method": str(self.method),
            "basis": str(self.basis),
            "functional": "" if self.functional is None else str(self.functional),
            "options_digest": str(self.options_digest),
            "files": [f.to_toml_table() for f in self.files],
        }

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping mirroring the TOML
        section. Used by ``--vibeqc-dry-run`` for tooling that prefers
        JSON over TOML."""
        return {
            "stem": str(self.stem),
            "job_kind": str(self.job_kind),
            "method": str(self.method),
            "basis": str(self.basis),
            "functional": self.functional,
            "options_digest": str(self.options_digest),
            "files": [f.to_jsonable() for f in self.files],
        }

    def to_json(self) -> str:
        """Render the plan as a JSON string (sorted keys, indented)."""
        return json.dumps(self.to_jsonable(), indent=2, sort_keys=True)

    # ------------------------------------------------------------------ #
    # Inspection helpers                                                 #
    # ------------------------------------------------------------------ #

    def files_by_role(self, role: OutputRole) -> tuple[PlannedFile, ...]:
        return tuple(f for f in self.files if f.role == role)

    def has_role(self, role: OutputRole) -> bool:
        return any(f.role == role for f in self.files)

    def guaranteed_files(self) -> tuple[PlannedFile, ...]:
        """All declared files with ``always=True`` -- the set vq should
        unconditionally fetch and the user should expect on disk after
        a successful run."""
        return tuple(f for f in self.files if f.always)


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def _resolve_optional_path(
    value: bool | str | os.PathLike,
    stem: Path,
    default_suffix: str,
) -> Path:
    """Translate a ``True | str | PathLike`` opt-in kwarg into a target
    path. Mirrors the convention used by ``run_job(perf_log=...)`` and
    ``run_job(structured_log=...)`` -- ``True`` => sibling next to the
    stem, an explicit path => use it verbatim."""
    if value is True:
        return stem_sibling(stem, default_suffix)
    if isinstance(value, (str, os.PathLike)):
        return Path(os.fspath(value))
    # Any falsy value should have been filtered before reaching here;
    # fall back defensively to the sibling.
    return stem_sibling(stem, default_suffix)


def _default_options_digest(
    *,
    method: str,
    basis: str,
    functional: str | None,
) -> str:
    """Short (12-char) hex digest of the user-facing job identity.

    Used by vq's "did this run change?" check. The digest is *not* a
    cryptographic guarantee -- it identifies the job by its declared
    method+basis+functional, which is enough for the queue's purposes.
    """
    payload = f"{method.lower()}|{basis.lower()}|{(functional or '').lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
