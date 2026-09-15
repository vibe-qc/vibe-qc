"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import json
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


def pytest_sessionstart(session):
    """Snapshot the invocation directory, then warn on a stale compiled core.

    The snapshot is half of the #508 guard: it records which job artifacts were
    already lying in the directory pytest was started from, so
    ``pytest_sessionfinish`` can tell which ones *this* session created. See
    that hook for why it matters.

    The rest of this hook warns when the compiled core predates the newest
    ``cpp/`` commit.

    The editable install has scikit-build auto-rebuild OFF, so a ``git pull`` that
    carries C++ commits leaves a stale ``_vibeqc_core*.so`` behind. A stale core does
    not reliably fail with ``ImportError``: it can silently produce **wrong numbers**,
    so the symptom looks like a physics regression rather than a build problem.

    Concretely (2026-07-10): two ``tests/test_ccm_direct.py`` LiH gates failed with a
    2.04e-2 Ha/cell direct-vs-GDF gap against a ``.so`` built the previous afternoon,
    predating ``f8c213e8`` (``cpp/src/aopair_ft.cpp``, "disable the Gamma pair-FT
    mirror on momentum-shifted meshes"). Rebuilding fixed it with no source change,
    after a bisect of a regression that did not exist. A python-only ``git bisect`` is
    likewise meaningless whenever ``cpp/`` moved in the range.

    Compares against working-tree **file mtimes**, not the newest ``cpp/`` commit
    timestamp. A commit's committer time is stamped on whatever machine authored it,
    so a commit dated 13:42 can land in this tree at 13:56, after a 13:50 build; the
    mtime that ``git checkout`` writes when it updates the file is the only local
    record of when these sources last changed.

    Warning only, never an error: a stale core is a local-environment condition and
    must not red anyone's suite. Silent no-op when the sources are absent (sdist, CI
    image, installed wheel).
    """
    import datetime as _dt
    import pathlib
    import warnings

    invocation_dir = Path(str(session.config.invocation_params.dir))
    session.config._vibeqc_invocation_dir = invocation_dir
    session.config._vibeqc_artifacts_at_start = job_artifacts_in(invocation_dir)

    try:
        from vibeqc import _vibeqc_core
    except Exception:  # noqa: BLE001 -- an import failure is not this hook's business
        return

    so = pathlib.Path(getattr(_vibeqc_core, "__file__", "") or "")
    cpp = pathlib.Path(__file__).resolve().parent.parent / "cpp"
    if not so.is_file() or not cpp.is_dir():
        return

    newest, newest_path = 0.0, None
    for suffix in ("*.cpp", "*.hpp", "*.h", "*.cc", "*.cu"):
        for src in cpp.rglob(suffix):
            try:
                m = src.stat().st_mtime
            except OSError:
                continue
            if m > newest:
                newest, newest_path = m, src
    if newest_path is None:
        return

    built = so.stat().st_mtime
    if built >= newest:
        return

    fmt = "%Y-%m-%d %H:%M"
    warnings.warn(
        "STALE COMPILED CORE: "
        f"{so.name} was built {_dt.datetime.fromtimestamp(built):{fmt}} but "
        f"{newest_path.relative_to(cpp.parent)} changed "
        f"{_dt.datetime.fromtimestamp(newest):{fmt}}. Auto-rebuild is off, so this "
        "session may compute WRONG NUMBERS rather than fail to import. Rebuild before "
        "trusting any numerical failure:\n"
        "    pip install -e . --no-build-isolation "
        "--config-settings=build-dir=build-local",
        UserWarning,
        stacklevel=1,
    )


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
