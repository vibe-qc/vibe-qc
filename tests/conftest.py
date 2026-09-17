"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import datetime as dt
import json
import os
import warnings
from pathlib import Path

import pytest
from vibeqc import Atom, BasisSet, Molecule, RHFOptions, run_rhf
from vibeqc.output import OutputPlan

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
SUITE_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "test_gate"
    / "suite_manifest.json"
)

#: Process exit status for a session refused because the compiled core is older
#: than ``cpp/``. Deliberately outside pytest's own range (0-5), so neither a
#: reader nor ``scripts/test_gate/run_full_suite.py::classify`` can mistake the
#: refusal for a test failure. It extends the vocabulary
#: ``scripts/head_stable_run.sh`` established for the same hazard: 98 a voided
#: measurement, 99 a setup error (#285).
STALE_CORE_EXIT_STATUS = 97

#: Set truthy to downgrade the refusal to a warning. For the case where the
#: rebuild is impossible and the session wants the Python-only tests anyway; no
#: number it prints can be trusted.
STALE_CORE_OVERRIDE_ENV = "VIBEQC_ALLOW_STALE_CORE"

_CPP_SOURCE_GLOBS = ("*.cpp", "*.hpp", "*.h", "*.cc", "*.cu")
_REBUILD_COMMAND = (
    "pip install -e . --no-build-isolation "
    "--config-settings=build-dir=build-local"
)


def _planned_job_artifact_suffixes() -> tuple[str, ...]:
    """Derive the guard inventory from the authoritative output planner.

    Cube orbital labels are caller-defined, so their planned paths collapse to
    the shared ``.cube`` tail. Crash-dump NumPy attachments are runtime-derived
    and handled separately in :func:`job_artifacts_in`.
    """
    stem = Path("__vq_pytest_artifact_guard__")
    common = {
        "output": stem,
        "method": "rhf",
        "basis": "sto-3g",
        "functional": None,
        "optimize": True,
        "structured_log": True,
        "crash_dump": True,
        "write_population": True,
        "write_cube_density": True,
        "cube_mo_labels": (0, "homo"),
        "write_poscar": True,
        "write_xsf_structure": True,
        "write_density_xsf": True,
        "write_cif": True,
        "job_kind": "periodic_scf",
        "output_qvf": True,
    }
    plans = (
        OutputPlan.from_run_job_kwargs(**common, perf_log=True),
        OutputPlan.from_run_job_kwargs(
            **common,
            perf_log=Path(f"{stem}.perf.json"),
        ),
    )
    suffixes = set()
    for plan in plans:
        for planned in plan.files:
            if planned.format == "cube":
                suffixes.add(".cube")
                continue
            name = planned.path.name
            if name.startswith(stem.name):
                suffixes.add(name[len(stem.name) :])
    return tuple(sorted(suffixes))


# A file with one of these planner-derived suffixes appearing in the directory
# pytest was invoked from means a test passed a relative output stem and wrote
# its artifacts into the caller's working tree.
JOB_ARTIFACT_SUFFIXES = _planned_job_artifact_suffixes()


def job_artifacts_in(directory: Path) -> set[str]:
    """Names of files in ``directory`` that look like vibe-qc job artifacts.

    Non-recursive and non-throwing: a job writes its artifacts as siblings of
    its output stem, and an unreadable directory is not this guard's problem.
    """
    try:
        entries = list(directory.iterdir())
    except OSError:
        return set()
    return {
        entry.name
        for entry in entries
        if entry.is_file()
        and (
            entry.name.endswith(JOB_ARTIFACT_SUFFIXES)
            or (".dump." in entry.name and entry.name.endswith(".npy"))
        )
    }


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Apply the release-tier and maturity markers from the suite manifest.

    Keeping file classification in one machine-readable manifest avoids marker
    drift across hundreds of modules while retaining normal ``pytest -m``
    selection.  A missing classification is a collection error, not an
    implicit advisory demotion.
    """
    if not SUITE_MANIFEST.is_file():
        raise pytest.UsageError(f"missing test-suite manifest: {SUITE_MANIFEST}")
    records = json.loads(SUITE_MANIFEST.read_text(encoding="utf-8"))["tests"]
    by_file = {record["file"]: record for record in records}
    root = Path(str(config.rootpath)).resolve()
    missing = set()
    for item in items:
        try:
            rel = item.path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        record = by_file.get(rel)
        if record is None:
            missing.add(rel)
            continue
        item.add_marker(getattr(pytest.mark, f"tier{record['tier'][1:]}"))
        maturity = record["maturity"].replace("-", "_")
        item.add_marker(getattr(pytest.mark, f"maturity_{maturity}"))
    if missing:
        paths = ", ".join(sorted(missing))
        raise pytest.UsageError(f"unclassified test files: {paths}")


def newest_cpp_source(cpp: Path) -> tuple[float, Path] | None:
    """Return ``(mtime, path)`` of the most recently modified C++ source.

    ``None`` when ``cpp`` holds no source at all.

    Compares working-tree **file mtimes**, never the newest ``cpp/`` commit
    timestamp. A commit's committer time is stamped on whatever machine
    authored it, so a commit dated 13:42 can land in this tree at 13:56, after
    a 13:50 build; the mtime ``git checkout`` writes when it updates the file
    is the only local record of when these sources last changed here.
    """
    newest, newest_path = 0.0, None
    for glob in _CPP_SOURCE_GLOBS:
        for src in cpp.rglob(glob):
            try:
                mtime = src.stat().st_mtime
            except OSError:
                continue
            if mtime > newest:
                newest, newest_path = mtime, src
    if newest_path is None:
        return None
    return newest, newest_path


def stale_core_report() -> str | None:
    """Describe a compiled core older than ``cpp/``, or ``None`` if it is current.

    Silent ``None`` when the core will not import — that is not this function's
    business — and when ``cpp/`` is absent, which is the normal shape of an
    sdist, a CI image or an installed wheel.
    """
    try:
        from vibeqc import _vibeqc_core
    except Exception:  # noqa: BLE001 -- an import failure is not this hook's business
        return None

    core = Path(getattr(_vibeqc_core, "__file__", "") or "")
    cpp = Path(__file__).resolve().parent.parent / "cpp"
    if not core.is_file() or not cpp.is_dir():
        return None

    found = newest_cpp_source(cpp)
    if found is None:
        return None
    newest, newest_path = found

    built = core.stat().st_mtime
    if built >= newest:
        return None

    fmt = "%Y-%m-%d %H:%M"
    return (
        f"{core.name} was built {dt.datetime.fromtimestamp(built):{fmt}} but "
        f"{newest_path.relative_to(cpp.parent)} changed "
        f"{dt.datetime.fromtimestamp(newest):{fmt}}."
    )


def stale_core_refusal(report: str) -> str:
    """The message printed in place of running the session."""
    return (
        f"STALE COMPILED CORE — refusing to run this session (#285)\n"
        f"  {report}\n"
        "  Auto-rebuild is off, so this checkout's Python would run against an\n"
        "  older checkout's C++. That pairing does not reliably fail to import.\n"
        "  It computes WRONG NUMBERS, in both directions:\n"
        "    a red test reads as a physics regression that does not exist, and\n"
        "    a green test certifies the previous commit under this one's name.\n"
        "  Rebuild:\n"
        f"      {_REBUILD_COMMAND}\n"
        "  To run anyway, accepting that no number this session prints can be\n"
        "  trusted:\n"
        f"      {STALE_CORE_OVERRIDE_ENV}=1 pytest ...\n"
        f"  Exit status {STALE_CORE_EXIT_STATUS} is reserved for this refusal, so it "
        "is never counted as a test failure."
    )


def stale_core_warning(report: str) -> str:
    """The message printed when the refusal is overridden."""
    return (
        f"STALE COMPILED CORE: {report} Auto-rebuild is off, so this session may "
        "compute WRONG NUMBERS rather than fail to import, and "
        f"{STALE_CORE_OVERRIDE_ENV} is set, so it was allowed to start anyway. No "
        "result below is evidence about this commit — neither a failure nor a "
        "pass. Rebuild before trusting either:\n"
        f"    {_REBUILD_COMMAND}"
    )


def pytest_sessionstart(session):
    """Snapshot the invocation directory, then refuse a stale compiled core.

    The snapshot is half of the #508 guard: it records which job artifacts were
    already lying in the directory pytest was started from, so
    ``pytest_sessionfinish`` can tell which ones *this* session created. See
    that hook for why it matters.

    The rest of this hook refuses to run when the compiled core predates the
    newest ``cpp/`` source.

    The editable install has scikit-build auto-rebuild OFF, so a ``git pull``
    that carries C++ commits leaves a stale ``_vibeqc_core*.so`` behind. A stale
    core does not reliably fail with ``ImportError``: it silently produces
    **wrong numbers**, so the symptom looks like a physics regression rather
    than a build problem.

    Concretely (2026-07-10): two ``tests/test_ccm_direct.py`` LiH gates failed
    with a 2.04e-2 Ha/cell direct-vs-GDF gap against a ``.so`` built the
    previous afternoon, predating ``f8c213e8`` (``cpp/src/aopair_ft.cpp``,
    "disable the Gamma pair-FT mirror on momentum-shifted meshes"). Rebuilding
    fixed it with no source change, after a bisect of a regression that did not
    exist. Again on 2026-09-13: a 900 s timeout in
    ``tests/test_basis_filter.py`` was charged to the branch under test, and
    the cause was a core predating ``312c34b`` (``cpp/src/schwarz.cpp``). A
    python-only ``git bisect`` is likewise meaningless whenever ``cpp/`` moved
    in the range.

    **Why this refuses rather than warns (#285).** Both sessions above *were*
    warned. The warning is printed before the first test and has scrolled past
    the failure by the time anyone reads it; ``-p no:warnings`` drops it, and
    under ``pytest-xdist`` each worker emits it into a stream nobody is
    watching. It also does nothing at all about the worse direction, a pass on
    code that is not running.

    The refusal is not a red suite, which is what the warning was protecting:
    it happens before collection and exits ``STALE_CORE_EXIT_STATUS``, a status
    no test failure produces. ``scripts/update.sh:496`` and
    ``scripts/head_stable_run.sh`` already answer this hazard with a stop
    rather than a warning; this hook was the outlier.

    Silent no-op when the sources are absent (sdist, CI image, installed
    wheel), and overridable through ``STALE_CORE_OVERRIDE_ENV`` — which then
    keeps the warning *and* repeats it under the results, where it cannot
    scroll away.
    """
    invocation_dir = Path(str(session.config.invocation_params.dir))
    session.config._vibeqc_invocation_dir = invocation_dir
    session.config._vibeqc_artifacts_at_start = job_artifacts_in(invocation_dir)

    report = stale_core_report()
    if report is None:
        return

    override = os.environ.get(STALE_CORE_OVERRIDE_ENV, "")
    if override.strip().lower() in {"1", "true", "yes", "on"}:
        session.config._vibeqc_stale_core = report
        warnings.warn(stale_core_warning(report), UserWarning, stacklevel=1)
        return

    pytest.exit(
        stale_core_refusal(report),
        returncode=STALE_CORE_EXIT_STATUS,
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    """Repeat the stale-core notice *below* the results, not above them.

    Only reached when the refusal was overridden. The session-start warning is
    the first thing printed, so by the last screen of a long run it is gone;
    this is the copy the reader actually sees. It is printed whatever the
    outcome, because a pass on a stale core is the more misleading of the two.
    """
    report = getattr(config, "_vibeqc_stale_core", None)
    if report is None:
        return
    terminalreporter.write_sep("=", "results computed on a STALE COMPILED CORE", red=True)
    terminalreporter.write_line(stale_core_warning(report))


def pytest_sessionfinish(session, exitstatus):
    """Fail the session if it left job artifacts in the invocation directory.

    ``run_job(output=...)`` takes a path *stem*, so a bare relative name writes
    the job's ``.out`` / ``.xyz`` / ``.system`` / ``.bibtex`` / ``.references``
    / ``.scf.jsonl`` siblings into whatever directory pytest was started from
    — for this suite, a git checkout. No test should write into the repository
    it is testing.

    This is a gate rather than a warning because the failure is otherwise
    silent for days. #508: a full-suite sweep left six ``h2_cas_test.*`` files
    in compute-medium's ``vq admin update``-managed release checkout, and the pollution
    only surfaced when the *next* fleet roll refused to deploy over a dirty
    tree — a guard in a different system, tripping on a symptom whose cause was
    nowhere near it. The same six files also sat untracked in a coordinator
    clone for a whole session without anyone noticing.

    Only files that appeared *during* this session count; anything already
    present at ``pytest_sessionstart`` is somebody else's mess and is not
    reported here. Under ``pytest-xdist`` the check runs in the controller
    only, since the workers share its working directory.
    """
    if hasattr(session.config, "workerinput"):
        return
    start = getattr(session.config, "_vibeqc_artifacts_at_start", None)
    directory = getattr(session.config, "_vibeqc_invocation_dir", None)
    if start is None or directory is None:
        return

    leaked = sorted(job_artifacts_in(directory) - start)
    if not leaked:
        return

    listing = "\n".join(f"      {name}" for name in leaked)
    message = (
        "TESTS WROTE JOB ARTIFACTS INTO THE WORKING DIRECTORY (#508)\n"
        f"  {directory}\n"
        f"{listing}\n"
        "  A test passed a bare relative stem to output=, so the job wrote its\n"
        "  artifacts next to the caller instead of into a temporary directory.\n"
        "  Pass output=str(tmp_path / \"stem\") — or tmp_path_factory.mktemp(...)\n"
        "  from a module/session-scoped fixture — and delete the files above."
    )
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", "working directory polluted", red=True)
        reporter.write_line(message)
    if exitstatus == 0:
        session.exitstatus = 1


@pytest.fixture
def tight_rhf_opts() -> RHFOptions:
    """RHF options tightened to machine precision for reference comparisons."""
    opts = RHFOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.damping = 0.0
    return opts


def make_molecule(atoms_bohr, charge: int = 0, multiplicity: int = 1) -> Molecule:
    """Build a vibe-qc Molecule from [(Z, [x, y, z])] with positions in bohr."""
    return Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in atoms_bohr],
        charge,
        multiplicity,
    )


def run_vibeqc_rhf(atoms_bohr, basis_name: str, opts: RHFOptions):
    mol = make_molecule(atoms_bohr)
    basis = BasisSet(mol, basis_name)
    return run_rhf(mol, basis, opts)


def run_pyscf_rhf(atoms_bohr, basis_name: str):
    """Reference RHF via PySCF. Returns (e_tot, mo_energy ndarray)."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    # PySCF's default conv_tol_grad = sqrt(conv_tol), so with conv_tol = 1e-12
    # the gradient threshold is only 1e-6 — loose enough that MO energies
    # still carry ~1e-7 residual. Match our own gradient tolerance (1e-10)
    # so orbital energies settle to machine precision on the reference side.
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    assert mf.converged, f"PySCF reference did not converge on {basis_name}"
    return mf.e_tot, mf.mo_energy


# Standard molecular geometries, positions in bohr.
GEOMETRIES = {
    "H2": [
        (1, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 1.4]),  # R = 1.4 bohr exactly (Szabo/Ostlund reference)
    ],
    "H2O": [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
        (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    ],
    "CH4": [  # Tetrahedral, C-H = 1.087 Angstrom
        (6, [0.0, 0.0, 0.0]),
        (1, [+0.626 * ANGSTROM_TO_BOHR, +0.626 * ANGSTROM_TO_BOHR, +0.626 * ANGSTROM_TO_BOHR]),
        (1, [-0.626 * ANGSTROM_TO_BOHR, -0.626 * ANGSTROM_TO_BOHR, +0.626 * ANGSTROM_TO_BOHR]),
        (1, [-0.626 * ANGSTROM_TO_BOHR, +0.626 * ANGSTROM_TO_BOHR, -0.626 * ANGSTROM_TO_BOHR]),
        (1, [+0.626 * ANGSTROM_TO_BOHR, -0.626 * ANGSTROM_TO_BOHR, -0.626 * ANGSTROM_TO_BOHR]),
    ],
}


# ---------------------------------------------------------------------------
# macOS OpenMP safety: prevent nested-threading crashes on Apple Silicon.
# The vendored libomp on macOS does not support nested parallelism;
# when numpy/BLAS threads are active inside an OpenMP parallel region,
# the runtime crashes.  Setting OMP_NUM_THREADS=1 prevents this.
# ---------------------------------------------------------------------------

def pytest_configure(config):  # noqa: ARG001
    """Set OMP_NUM_THREADS=1 on macOS if not already configured."""
    import os
    import sys

    if sys.platform == "darwin":
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "1"
        if "OPENBLAS_NUM_THREADS" not in os.environ:
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
        if "MKL_NUM_THREADS" not in os.environ:
            os.environ["MKL_NUM_THREADS"] = "1"
