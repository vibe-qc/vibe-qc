"""Regression / parity test-suite runner.

Invocation::

    .venv/bin/python -m examples.regression.run_suite [--target dev|release|both] \
        [--systems all|nacl_rocksalt,...] [--bases all|sto-3g,...] \
        [--methods all|rks-lda,...]

Current scope:
  * one target (dev — the editable install)
  * molecular cross-code rows: vibe-qc + PySCF + ORCA when available
  * periodic cross-code rows: vibe-qc + PySCF.pbc, with optional
    CRYSTAL / CP2K references behind CLI flags
  * writes a self-contained run directory under ~/vibeqc-runs by default

Each new (system, basis, method) cell requires:
  * a `systems/<periodic|molecules>/<id>.py` exporting `SPEC`
  * (optionally) `expected/<system_id>__<basis>__<method_id>.json`
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .core.case import (
    CaseRecord,
    collect_env,
    write_csv,
    write_env_json,
)
from .core.compare import annotate_against_reference
from .core.output_paths import (
    DEFAULT_OUTPUT_ROOT,
    RUNS_DIR_ENV,
    make_run_id,
    resolve_output_root,
)
from .core.report import render_summary
from .core.spec import ExpectedRef, MethodSpec, MoleculeSpec, PeriodicSpec
from .methods.catalog import METHODS

REPO_ROOT = Path(__file__).resolve().parents[2]
REGRESSION_ROOT = Path(__file__).resolve().parent
EXPECTED_DIR = REGRESSION_ROOT / "expected"


# Wave-1 enabled cases. Adding a (system, basis, method) tuple here is
# the one place to extend coverage as new systems / bases come online.
#
# Order matters: cheap / easy cases first, so a sanity-check run fails
# fast on the easy stuff before burning wall on the harder ones.

# Molecules — three-way compare (vibe-qc, PySCF, ORCA). DF cases keep
# all three rows when the local ORCA capability supports the requested
# setup; otherwise ORCA self-gates as 'unavailable'.
WAVE1_MOLECULE_CASES: Tuple[Tuple[str, str, str], ...] = (
    # Diatomic / atomic — smoke set.
    ("h2", "sto-3g", "rhf"),  # ~1 s
    ("ne_atom", "sto-3g", "rhf"),
    ("hf", "sto-3g", "rhf"),
    # Small polyatomics — direct, both methods.
    ("ch4", "sto-3g", "rhf"),
    ("nh3", "sto-3g", "rhf"),
    ("h2o", "sto-3g", "rhf"),
    ("h2o", "sto-3g", "rks-lda"),
    ("h2co", "sto-3g", "rhf"),
    ("h2co", "sto-3g", "rks-lda"),
    # Method-coverage cases (cycle through GGA + hybrid + post-HF on the
    # workhorse small molecules so every catalog entry sees at least one
    # case under cross-code parity).
    ("h2", "sto-3g", "mp2"),
    ("h2o", "sto-3g", "rks-pbe"),
    ("h2o", "sto-3g", "rks-blyp"),
    ("h2o", "sto-3g", "rks-b3lyp"),
    ("h2o", "sto-3g", "mp2"),
    ("ch4", "sto-3g", "rks-b3lyp"),
    ("ch4", "sto-3g", "mp2"),
    ("nh3", "sto-3g", "rks-b3lyp"),
    # Aromatic showcase.
    ("benzene", "sto-3g", "rhf"),
    ("benzene", "sto-3g", "rks-b3lyp"),
    # Open-shell stress tests.
    ("o2", "sto-3g", "uhf"),  # ³Σg⁻ diradical
    ("o2", "sto-3g", "uks-lda"),
    ("o2", "sto-3g", "uks-b3lyp"),
    ("o2", "sto-3g", "ump2"),
    ("o3", "sto-3g", "rhf"),  # singlet biradical-character
    ("o3", "sto-3g", "rks-lda"),
    ("o3", "sto-3g", "rks-b3lyp"),
    # Density-fitting cross-checks.
    ("h2o", "def2-svp", "rhf-df"),
    ("benzene", "def2-svp", "rhf"),  # direct baseline for DF comparison
    ("benzene", "def2-svp", "rhf-df"),
    ("benzene", "def2-svp", "rks-pbe-df"),
    # S22 noncovalent dimers — cross-code parity on dimer SCF (interaction-
    # energy benchmarking is wave-2; this exercises the molecular-SCF + ERI
    # stack on noncovalent-relevant geometries). The 5 originally wired
    # cases plus 5 more picked from the remaining 17 to span all three S22
    # families (HB / dispersion / mixed) without ballooning wall budget;
    # the other 12 ship as data-only spec modules in
    # `examples/regression/systems/molecules/s22_*.py`.
    ("s22_water_dimer", "sto-3g", "rhf"),
    ("s22_ammonia_dimer", "sto-3g", "rhf"),
    ("s22_formamide_dimer", "sto-3g", "rhf"),
    ("s22_methane_dimer", "sto-3g", "rhf"),
    ("s22_benzene_dimer_t", "sto-3g", "rhf"),
    ("s22_formic_acid_dimer", "sto-3g", "rhf"),  # HB: doubly-bonded carboxylic
    ("s22_ethene_dimer", "sto-3g", "rhf"),  # dispersion: π-π
    ("s22_ethene_ethyne", "sto-3g", "rhf"),  # mixed: π/HB
    ("s22_benzene_water", "sto-3g", "rhf"),  # mixed: OH⋯π
    ("s22_methane_benzene", "sto-3g", "rhf"),  # dispersion: CH⋯π
)

# Periodic — vibe-qc + PySCF.pbc only (ORCA periodic is out of scope).
WAVE1_PERIODIC_CASES: Tuple[Tuple[str, str, str], ...] = (
    # Easy reference (wide-gap, closed-shell, converges in seconds).
    ("ne_fcc", "sto-3g", "rks-lda"),
    # Ionic rocksalts — defaults set SAD guess + damping≥0.7 to dodge the
    # HCORE-divergence diagnostic the v0.7 SCF driver raises.
    ("lih_rocksalt", "sto-3g", "rks-lda"),
    ("nacl_rocksalt", "sto-3g", "rks-lda"),
    ("mgo_rocksalt", "sto-3g", "rks-lda"),
    # pob-dzvp-rev2 needs cross-code basis-name mapping (PySCF doesn't
    # ship it); deferred to wave 1.5.
    # ("nacl_rocksalt", "pob-dzvp-rev2",  "rks-lda"),
    # ("mgo_rocksalt",  "pob-dzvp-rev2",  "rks-lda"),
    # ("lih_rocksalt",  "pob-dzvp-rev2",  "rks-lda"),
    # Al₂O₃ corundum — 10-atom unit cell with O octahedral coordination;
    # slow at sanity-check budgets. Deferred to wave 1.5.
    # ("al2o3_corundum","sto-3g",         "rks-lda"),
    # ("al2o3_corundum","pob-dzvp-rev2",  "rks-lda"),
    # X23 molecular crystals — hand-crafted from textbook structures.
    # Specs are present (`x23_urea`, `x23_ice_ih`, `x23_benzene_crystal`)
    # but commented out of WAVE1 because the bare RKS-LDA path on
    # 16-48 atom unit cells without dispersion correction will produce
    # unphysical lattice energies; useful as inputs / examples but not
    # as active CI gates until the dispersion-correction work lands.
    # ("x23_urea",            "sto-3g", "rks-lda"),
    # ("x23_ice_ih",          "sto-3g", "rks-lda"),
    # ("x23_benzene_crystal", "sto-3g", "rks-lda"),
)


SMOKE_MOLECULE_CASES: Tuple[Tuple[str, str, str], ...] = (
    ("h2", "sto-3g", "rhf"),
)
SMOKE_PERIODIC_CASES: Tuple[Tuple[str, str, str], ...] = ()


def _selected_triples(
    *,
    systems: str,
    bases: str,
    methods: str,
    smoke: bool = False,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    sys_keep = systems.split(",")
    bas_keep = bases.split(",")
    met_keep = methods.split(",")

    def _keep(triple: Tuple[str, str, str]) -> bool:
        s, b, m = triple
        return (
            (s in sys_keep or "all" in sys_keep)
            and (b in bas_keep or "all" in bas_keep)
            and (m in met_keep or "all" in met_keep)
        )

    molecule_cases = SMOKE_MOLECULE_CASES if smoke else WAVE1_MOLECULE_CASES
    periodic_cases = SMOKE_PERIODIC_CASES if smoke else WAVE1_PERIODIC_CASES
    return (
        [t for t in molecule_cases if _keep(t)],
        [t for t in periodic_cases if _keep(t)],
    )


def _case_id(
    system_id: str,
    basis: str,
    method_id: str,
    kmesh: Tuple[int, int, int],
) -> str:
    km = "mol" if kmesh == (0, 0, 0) else "x".join(str(k) for k in kmesh)
    return f"{system_id}__{basis}__{method_id}__{km}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_case_notes(case_dir: Path, case: CaseRecord) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {case.label}",
        "",
        f"- system: `{case.system_id}`",
        f"- family: `{case.family}`",
        f"- basis: `{case.basis}`",
        f"- method: `{case.method_id}`",
        f"- kmesh: `{case.kmesh}`",
        f"- verbose log: `../../verbose/{case.label}.log`",
        "",
        "## Rows",
        "",
    ]
    for row in case.rows:
        lines.append(
            f"- `{row.code}`: status=`{row.status}`, "
            f"energy={row.energy_ha!r}, note={row.note or 'n/a'}"
        )
    (case_dir / "NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_row_artifact(artifact_dir: Path, row) -> None:
    _write_json(artifact_dir / "parsed.json", asdict(row))
    for name in ("stdout.log", "stderr.log"):
        path = artifact_dir / name
        if not path.exists():
            path.write_text("", encoding="utf-8")


def _write_manifest(
    *,
    run_dir: Path,
    output_root: Path,
    run_id: str,
    env: dict,
    molecule_triples: List[Tuple[str, str, str]],
    periodic_triples: List[Tuple[str, str, str]],
    cases: Optional[List[CaseRecord]] = None,
    status: str = "running",
    dry_run: bool = False,
) -> None:
    planned = [
        {
            "case_id": _case_id(s, b, m, (0, 0, 0)),
            "kind": "molecule",
            "system_id": s,
            "basis": b,
            "method_id": m,
            "kmesh": "mol",
        }
        for s, b, m in molecule_triples
    ]
    for s, b, m in periodic_triples:
        spec = _load_periodic_spec(s)
        planned.append(
            {
                "case_id": _case_id(s, b, m, spec.default_kmesh),
                "kind": "periodic",
                "system_id": s,
                "basis": b,
                "method_id": m,
                "kmesh": "x".join(str(k) for k in spec.default_kmesh),
            }
        )

    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "dry_run": bool(dry_run),
        "created_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "git_sha": env.get("git_sha", "unknown"),
        "git_branch": env.get("git_branch", "unknown"),
        "artifacts": {
            "env": "env.json",
            "results": "results.csv",
            "summary": "summary.md",
            "cases": "cases/",
            "verbose": "verbose/",
        },
        "planned_cases": planned,
        "completed_cases": [
            {
                "case_id": case.label,
                "system_id": case.system_id,
                "basis": case.basis,
                "method_id": case.method_id,
                "kmesh": "mol" if case.kmesh == (0, 0, 0)
                         else "x".join(str(k) for k in case.kmesh),
                "rows": [asdict(row) for row in case.rows],
            }
            for case in (cases or [])
        ],
    }
    _write_json(run_dir / "manifest.json", payload)


def _load_periodic_spec(system_id: str) -> PeriodicSpec:
    mod = importlib.import_module(
        f".systems.periodic.{system_id}",
        package="examples.regression",
    )
    spec = getattr(mod, "SPEC", None)
    if not isinstance(spec, PeriodicSpec):
        raise RuntimeError(
            f"systems.periodic.{system_id}: module must export SPEC: PeriodicSpec"
        )
    return spec


def _load_molecule_spec(system_id: str) -> MoleculeSpec:
    mod = importlib.import_module(
        f".systems.molecules.{system_id}",
        package="examples.regression",
    )
    spec = getattr(mod, "SPEC", None)
    if not isinstance(spec, MoleculeSpec):
        raise RuntimeError(
            f"systems.molecules.{system_id}: module must export SPEC: MoleculeSpec"
        )
    return spec


def _load_expected(
    system_id: str,
    basis: str,
    method_id: str,
    kmesh: Tuple[int, int, int],
) -> ExpectedRef:
    fname = f"{system_id}__{basis}__{method_id}.json"
    path = EXPECTED_DIR / fname
    if not path.is_file():
        # Sensible default: pyscf-at-runtime, generous tolerance.
        return ExpectedRef(
            system_id=system_id,
            basis=basis,
            method_id=method_id,
            kmesh=kmesh,
            primary_source="pyscf_at_runtime",
            tolerance_ha=5e-3,
            tolerance_rationale="default: 5 mHa (no expected.json on disk)",
        )
    with open(path) as fh:
        d = json.load(fh)
    return ExpectedRef(
        system_id=d.get("system_id", system_id),
        basis=d.get("basis", basis),
        method_id=d.get("method_id", method_id),
        kmesh=tuple(d.get("kmesh", kmesh)),  # type: ignore[arg-type]
        primary_source=d.get("primary_source", "pyscf_at_runtime"),
        energy_ha=d.get("energy_ha"),
        tolerance_ha=float(d.get("tolerance_ha", 5e-3)),
        tolerance_rationale=d.get("tolerance_rationale", ""),
        published_energy_ha=d.get("published_energy_ha"),
        published_source=d.get("published_source", ""),
    )


def _filter(items: Iterable[str], keep: Optional[List[str]]) -> List[str]:
    items = list(items)
    if keep is None or "all" in keep:
        return items
    return [i for i in items if i in keep]


def _resolve_method(method_id: str) -> MethodSpec:
    if method_id not in METHODS:
        raise SystemExit(
            f"unknown method_id {method_id!r}; known: {sorted(METHODS.keys())}"
        )
    return METHODS[method_id]


def _vibeqc_code_version(env: dict) -> str:
    return f"{env.get('vibeqc_version', '?')}@{env.get('git_sha', '?')}"


_RESULT_MARKER = "ISOLATED-RUNNER-RESULT:"


def _apply_static_metadata(row: "CodeRow", spec) -> None:
    atoms = tuple(getattr(spec, "atoms", ()) or ())
    if row.n_atoms is None and atoms:
        row.n_atoms = len(atoms)
    if row.n_electrons is None and atoms:
        charge = int(getattr(spec, "charge", 0) or 0)
        row.n_electrons = int(sum(getattr(at, "z", 0) for at in atoms) - charge)


def _row_size_kwargs(spec) -> dict[str, Optional[int]]:
    from .core.case import CodeRow  # noqa: WPS433

    row = CodeRow(
        run_id="",
        target="",
        system_id="",
        family="",
        basis="",
        method_id="",
        kmesh="",
        code="",
        code_version="",
    )
    _apply_static_metadata(row, spec)
    return {
        "n_atoms": row.n_atoms,
        "n_electrons": row.n_electrons,
    }


def _isolated_runner_call(
    *,
    code: str,
    spec_kind: str,
    spec,
    basis: str,
    method: MethodSpec,
    log_path: Path,
    workdir: Optional[Path],
    artifact_dir: Optional[Path],
    run_id: str,
    target: str,
    code_version: str,
    timeout_s: float = 7200.0,
    rsgdf_tail_ke_cutoff: Optional[float] = None,
    write_qvf_artifact: bool = False,
) -> "CodeRow":
    """Run one (case, code) pair in a subprocess via _isolated_runner.

    Insulates the dispatcher from interpreter-level crashes (segfaults,
    OOM, signals). On non-zero subprocess returncode the dispatcher
    synthesises an `error` CodeRow so the suite continues to the next
    case instead of bricking — see `examples.regression.core.
    _isolated_runner` for the rationale.
    """
    from .core.case import CodeRow  # noqa: WPS433

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    kmesh_str = (
        "mol"
        if spec_kind == "molecule"
        else "x".join(str(k) for k in spec.default_kmesh)
    )
    family = getattr(spec, "family", "")
    cmd = [
        sys.executable,
        "-m",
        "examples.regression.core._isolated_runner",
        code,
        spec_kind,
        spec.id,
        basis,
        method.id,
        str(log_path),
        str(workdir) if workdir is not None else "-",
        run_id,
        target,
        code_version,
    ]
    if artifact_dir is not None or write_qvf_artifact:
        cmd.append(str(artifact_dir) if artifact_dir is not None else "-")
        cmd.append("1" if write_qvf_artifact else "0")
    child_env = None
    if (
        rsgdf_tail_ke_cutoff is not None
        and spec_kind == "periodic"
        and code == "vibeqc"
    ):
        child_env = os.environ.copy()
        child_env["VIBEQC_REGRESSION_RSGDF_TAIL_KE_CUTOFF"] = str(
            float(rsgdf_tail_ke_cutoff)
        )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        row = CodeRow(
            run_id=run_id,
            target=target,
            system_id=spec.id,
            family=family,
            basis=basis,
            method_id=method.id,
            kmesh=kmesh_str,
            code=code,
            code_version="unknown",
            status="error",
            note=f"isolated subprocess timeout (>{timeout_s:.0f}s)",
            **_row_size_kwargs(spec),
        )
        if artifact_dir is not None:
            _write_row_artifact(artifact_dir, row)
        return row

    if artifact_dir is not None:
        (artifact_dir / "isolated_stdout.log").write_text(
            result.stdout or "",
            encoding="utf-8",
        )
        (artifact_dir / "isolated_stderr.log").write_text(
            result.stderr or "",
            encoding="utf-8",
        )
        stdout_path = artifact_dir / "stdout.log"
        stderr_path = artifact_dir / "stderr.log"
        if not stdout_path.exists():
            stdout_path.write_text(result.stdout or "", encoding="utf-8")
        if not stderr_path.exists():
            stderr_path.write_text(result.stderr or "", encoding="utf-8")

    if result.returncode != 0:
        # Either the runner itself raised (no result line on stdout),
        # or the interpreter died below the Python boundary (segfault,
        # SIGKILL, …). Either way, surface as an `error` row with the
        # diagnostic information we have.
        signal_hint = ""
        if result.returncode < 0:
            signal_hint = f"signal {-result.returncode}"
        else:
            signal_hint = f"returncode {result.returncode}"
        stderr_tail = (result.stderr or "")[-200:].strip()
        note = f"isolated subprocess died ({signal_hint})"
        if stderr_tail:
            note += f"; stderr tail: {stderr_tail}"
        row = CodeRow(
            run_id=run_id,
            target=target,
            system_id=spec.id,
            family=family,
            basis=basis,
            method_id=method.id,
            kmesh=kmesh_str,
            code=code,
            code_version="unknown",
            status="error",
            note=note,
            **_row_size_kwargs(spec),
        )
        if artifact_dir is not None:
            _write_row_artifact(artifact_dir, row)
        return row

    for line in reversed((result.stdout or "").splitlines()):
        if line.startswith(_RESULT_MARKER):
            payload = line[len(_RESULT_MARKER) :].strip()
            try:
                d = json.loads(payload)
            except json.JSONDecodeError as exc:
                row = CodeRow(
                    run_id=run_id,
                    target=target,
                    system_id=spec.id,
                    family=family,
                    basis=basis,
                    method_id=method.id,
                    kmesh=kmesh_str,
                    code=code,
                    code_version="unknown",
                    status="error",
                    note=f"isolated subprocess JSON decode failed: {exc}",
                    **_row_size_kwargs(spec),
                )
                if artifact_dir is not None:
                    _write_row_artifact(artifact_dir, row)
                return row
            row = CodeRow(**d)
            _apply_static_metadata(row, spec)
            if artifact_dir is not None:
                _write_row_artifact(artifact_dir, row)
            return row

    row = CodeRow(
        run_id=run_id,
        target=target,
        system_id=spec.id,
        family=family,
        basis=basis,
        method_id=method.id,
        kmesh=kmesh_str,
        code=code,
        code_version="unknown",
        status="error",
        note="isolated subprocess returned 0 but emitted no result line",
        **_row_size_kwargs(spec),
    )
    if artifact_dir is not None:
        _write_row_artifact(artifact_dir, row)
    return row


def _build_reference_attacher(source: Optional[str]):
    """Return a function ``system_id -> ExperimentalReference`` or
    ``None`` (when ``--include-experimental-reference`` was unset).

    Looks up each molecular system's CAS via
    ``vibeqc.fetch.references.canonical_molecules`` and pulls a
    cached / live record from CCCBDB. Cache misses fall through to
    a live fetch; failures (CAS not in canonical set,
    ``MoleculeNotFound``, transport error) are logged + the case
    runs without a reference.
    """
    if source is None:
        return None
    if source != "cccbdb":
        raise SystemExit(
            f"--include-experimental-reference: unsupported source {source!r}"
        )
    try:
        from vibeqc.fetch.references.canonical_molecules import find as _find_mol
        from vibeqc.fetch.references.client_cccbdb import (
            MoleculeNotFound,
            fetch_cccbdb,
        )
    except ImportError as exc:
        raise SystemExit(
            f"--include-experimental-reference cccbdb requires the fetcher "
            f"extra: pip install -e '.[fetch]'. Import failed: {exc}"
        )

    def _attach(system_id: str):
        try:
            mol = _find_mol(system_id)
        except KeyError:
            print(
                f"[regression]   no CCCBDB canonical entry for {system_id!r}; "
                "ref omitted",
            )
            return None
        try:
            return fetch_cccbdb(cas=mol.cas)
        except MoleculeNotFound as exc:
            print(f"[regression]   CCCBDB miss for {system_id} ({mol.cas}): {exc}")
            return None
        except Exception as exc:  # transport / parse / network
            print(
                f"[regression]   CCCBDB transport error for {system_id} "
                f"({mol.cas}): {type(exc).__name__}: {exc} — ref omitted",
            )
            return None

    return _attach


def run_one_periodic_case_dev(
    *,
    run_id: str,
    system_id: str,
    basis: str,
    method_id: str,
    target: str,
    env: dict,
    run_dir: Path,
    case_dir: Optional[Path] = None,
    include_crystal: bool = False,
    include_cp2k: bool = False,
    rsgdf_tail_ke_cutoff: Optional[float] = None,
    output_qvf: bool = False,
) -> CaseRecord:
    """Run one periodic (system, basis, method) case against the dev target.

    Two-way (vibe-qc + PySCF.pbc) by default; three-way when
    ``include_crystal=True`` and the CRYSTAL14-via-vq chain is
    available on the host, or ``include_cp2k=True`` with CP2K
    installed locally (M3c GAPW parity oracle). Both runners
    self-gate on missing executables / unsupported method+basis,
    so the flag flips on without further configuration are safe —
    affected cases just land as ``status='unavailable'`` rather
    than crashing.

    All in-process runner calls go through :func:`_isolated_runner_call`
    so a C-level crash in any runner (PySCF.pbc segfault, vibe-qc abort,
    …) only kills its own subprocess; the dispatcher continues to the
    next case. The CRYSTAL14 runner is already subprocess-driven via
    ``vq submit``, so it's invoked directly without an additional
    isolation layer.
    """
    spec = _load_periodic_spec(system_id)
    method = _resolve_method(method_id)
    kmesh = spec.default_kmesh
    expected = _load_expected(system_id, basis, method_id, kmesh)

    case = CaseRecord(
        system_id=system_id,
        family=spec.family,
        basis=basis,
        method_id=method_id,
        kmesh=kmesh,
    )
    log_path = run_dir / "verbose" / f"{case.label}.log"
    case.verbose_log_path = log_path
    if case_dir is None:
        case_dir = run_dir / "cases" / case.label
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "vibeqc").mkdir(parents=True, exist_ok=True)
    (case_dir / "reference").mkdir(parents=True, exist_ok=True)

    case.rows.append(
        _isolated_runner_call(
            code="vibeqc",
            spec_kind="periodic",
            spec=spec,
            basis=basis,
            method=method,
            log_path=log_path,
            workdir=None,
            artifact_dir=case_dir / "vibeqc",
            run_id=run_id,
            target=target,
            code_version=_vibeqc_code_version(env),
            rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
            write_qvf_artifact=output_qvf,
        )
    )
    case.rows.append(
        _isolated_runner_call(
            code="pyscf",
            spec_kind="periodic",
            spec=spec,
            basis=basis,
            method=method,
            log_path=log_path,
            workdir=None,
            artifact_dir=case_dir / "reference" / "pyscf",
            run_id=run_id,
            target=target,
            code_version="unknown",
        )
    )

    if include_crystal:
        from .core import runner_crystal

        crystal_row = runner_crystal.run_periodic_case(
            run_id=run_id,
            target=target,
            spec=spec,
            basis_name=basis,
            method=method,
            kmesh=kmesh,
            log_path=log_path,
            workdir=case_dir / "reference" / "crystal",
        )
        case.rows.append(crystal_row)
        _write_row_artifact(case_dir / "reference" / "crystal", crystal_row)

    if include_cp2k:
        from .core import runner_cp2k

        cp2k_row = runner_cp2k.run_periodic_case(
            run_id=run_id,
            target=target,
            spec=spec,
            basis_name=basis,
            method=method,
            kmesh=kmesh,
            log_path=log_path,
            workdir=case_dir / "reference" / "cp2k",
        )
        case.rows.append(cp2k_row)
        _write_row_artifact(case_dir / "reference" / "cp2k", cp2k_row)

    annotate_against_reference(case.rows, ref_code="pyscf", expected=expected)
    for row in case.rows:
        target_dir = (
            case_dir / "vibeqc"
            if row.code == "vibeqc"
            else case_dir / "reference" / row.code
        )
        _write_row_artifact(target_dir, row)
    _write_case_notes(case_dir, case)
    return case


def run_one_molecule_case_dev(
    *,
    run_id: str,
    system_id: str,
    basis: str,
    method_id: str,
    target: str,
    env: dict,
    run_dir: Path,
    case_dir: Optional[Path] = None,
    output_qvf: bool = False,
) -> CaseRecord:
    """Run one molecular case against the dev target. Three-way compare:
    vibe-qc, PySCF, ORCA. ORCA is the primary reference when available;
    PySCF if not.

    All runner calls go through :func:`_isolated_runner_call` for fault
    isolation — see the periodic dispatcher's docstring.
    """
    spec = _load_molecule_spec(system_id)
    method = _resolve_method(method_id)
    kmesh = (0, 0, 0)  # sentinel for molecules
    expected = _load_expected(system_id, basis, method_id, kmesh)

    case = CaseRecord(
        system_id=system_id,
        family=spec.family,
        basis=basis,
        method_id=method_id,
        kmesh=kmesh,
    )
    log_path = run_dir / "verbose" / f"{case.label}.log"
    case.verbose_log_path = log_path
    if case_dir is None:
        case_dir = run_dir / "cases" / case.label
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "vibeqc").mkdir(parents=True, exist_ok=True)
    (case_dir / "reference").mkdir(parents=True, exist_ok=True)
    orca_workdir = case_dir / "reference" / "orca"

    case.rows.append(
        _isolated_runner_call(
            code="vibeqc",
            spec_kind="molecule",
            spec=spec,
            basis=basis,
            method=method,
            log_path=log_path,
            workdir=None,
            artifact_dir=case_dir / "vibeqc",
            run_id=run_id,
            target=target,
            code_version=_vibeqc_code_version(env),
            write_qvf_artifact=output_qvf,
        )
    )
    pyscf_row = _isolated_runner_call(
        code="pyscf",
        spec_kind="molecule",
        spec=spec,
        basis=basis,
        method=method,
        log_path=log_path,
        workdir=None,
        artifact_dir=case_dir / "reference" / "pyscf",
        run_id=run_id,
        target=target,
        code_version="unknown",
    )
    case.rows.append(pyscf_row)
    orca_row = _isolated_runner_call(
        code="orca",
        spec_kind="molecule",
        spec=spec,
        basis=basis,
        method=method,
        log_path=log_path,
        workdir=orca_workdir,
        artifact_dir=case_dir / "reference" / "orca",
        run_id=run_id,
        target=target,
        code_version="unknown",
    )
    case.rows.append(orca_row)

    # Reference selection: ORCA when available, else PySCF.
    if orca_row.energy_ha is not None and orca_row.status != "unavailable":
        ref_code = "orca"
    else:
        ref_code = "pyscf"
    annotate_against_reference(case.rows, ref_code=ref_code, expected=expected)
    for row in case.rows:
        target_dir = (
            case_dir / "vibeqc"
            if row.code == "vibeqc"
            else case_dir / "reference" / row.code
        )
        _write_row_artifact(target_dir, row)
    _write_case_notes(case_dir, case)
    return case


def _write_dry_run_summary(
    *,
    out_path: Path,
    run_id: str,
    run_dir: Path,
    mol_triples: List[Tuple[str, str, str]],
    per_triples: List[Tuple[str, str, str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# vibe-qc regression dry run `{run_id}`",
        "",
        f"- output: `{run_dir}`",
        f"- planned cases: {len(mol_triples) + len(per_triples)}",
        "",
        "## Planned cases",
        "",
    ]
    for s, b, m in mol_triples:
        lines.append(f"- molecule `{s}` / `{b}` / `{m}`")
    for s, b, m in per_triples:
        lines.append(f"- periodic `{s}` / `{b}` / `{m}`")
    lines.extend(["", "## Action items", "", "(dry run only; no cases executed)", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _print_case_list(
    mol_triples: List[Tuple[str, str, str]],
    per_triples: List[Tuple[str, str, str]],
) -> None:
    print("kind,system_id,basis,method_id")
    for s, b, m in mol_triples:
        print(f"molecule,{s},{b},{m}")
    for s, b, m in per_triples:
        print(f"periodic,{s},{b},{m}")


def _error_case_record(
    *,
    run_id: str,
    target: str,
    system_id: str,
    basis: str,
    method_id: str,
    family: str,
    kmesh: Tuple[int, int, int],
    note: str,
) -> CaseRecord:
    from .core.case import CodeRow  # noqa: WPS433

    case = CaseRecord(
        system_id=system_id,
        family=family,
        basis=basis,
        method_id=method_id,
        kmesh=kmesh,
    )
    row = CodeRow(
        run_id=run_id,
        target=target,
        system_id=system_id,
        family=family,
        basis=basis,
        method_id=method_id,
        kmesh="mol" if kmesh == (0, 0, 0) else "x".join(str(k) for k in kmesh),
        code="suite",
        code_version="n/a",
        status="error",
        note=note,
    )
    case.rows.append(row)
    return case


def _print_row_progress(row) -> None:
    tag = row.code.ljust(7)
    e_str = f"{row.energy_ha:>14.8f}" if row.energy_ha is not None else " " * 14
    d_str = (
        f"{row.delta_ha_vs_ref:>+11.3e}"
        if row.delta_ha_vs_ref is not None
        else " " * 11
    )
    mha_str = (
        f"{row.delta_mha_vs_ref:+.3f} mHa"
        if row.delta_mha_vs_ref is not None
        else "n/a"
    )
    per_atom = (
        f"{row.abs_delta_mha_per_atom_vs_ref:.3f} mHa/atom"
        if row.abs_delta_mha_per_atom_vs_ref is not None
        else "n/a"
    )
    print(
        f"[regression]     {tag} E={e_str.strip()} "
        f"Δ={d_str.strip()} ({mha_str}; {per_atom}) "
        f"{row.status} {row.note}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="vibe-qc regression / parity test-suite runner",
    )
    p.add_argument(
        "--target",
        choices=["dev", "release", "both"],
        default="dev",
        help="Which install to test. Wave 1 only supports 'dev'.",
    )
    p.add_argument(
        "--systems",
        default="all",
        help="Comma-separated system ids, or 'all'.",
    )
    p.add_argument(
        "--bases",
        default="all",
        help="Comma-separated basis names, or 'all'.",
    )
    p.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method ids, or 'all'.",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Override the auto-generated run id (timestamp-uuid).",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help=(
            "Directory that owns regression runs. Precedence: this CLI "
            f"option, then ${RUNS_DIR_ENV}, then ~/vibeqc-runs."
        ),
    )
    p.add_argument(
        "--list-cases",
        action="store_true",
        help="List selected cases after filters and exit without creating a run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Create run metadata for the selected cases but do not execute them.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run the cheapest end-to-end case set (currently H2/STO-3G/RHF).",
    )
    p.add_argument(
        "--output-qvf",
        action="store_true",
        help=(
            "Write a validated case-local vibeqc.qvf archive for every "
            "vibe-qc row. Use this for paper/SI candidate runs; it is "
            "off by default to keep routine parity runs small."
        ),
    )
    p.add_argument(
        "--include-crystal",
        action="store_true",
        help="Add a third comparison column for periodic cases using "
        "CRYSTAL14 dispatched via vq submit (out-of-process per "
        "CLAUDE.md § 10). Requires `vq` on PATH and a configured "
        "default_host or VIBEQC_CRYSTAL_VQ_HOST. Unsupported "
        "method+basis combos (or missing vq) emit "
        "status='unavailable' rather than crashing. See "
        "examples.regression.core.runner_crystal for the env-var "
        "knobs (CPU count, wall-time cap, wrapper path, "
        "poll interval).",
    )
    p.add_argument(
        "--include-cp2k",
        action="store_true",
        help="Add a comparison column for periodic cases using a local "
        "CP2K subprocess (M3c GAPW parity oracle, per the design "
        "doc decision on the CP2K reference implementation). "
        "Requires `cp2k.psmp` / `cp2k.ssmp` / `cp2k` on PATH. "
        "Unsupported method+basis combos (or missing CP2K) emit "
        "status='unavailable' rather than crashing. See "
        "examples.regression.core.runner_cp2k for the env-var "
        "knobs.",
    )
    p.add_argument(
        "--include-experimental-reference",
        default=None,
        choices=("cccbdb",),
        help="Attach NIST CCCBDB experimental references to molecular "
        "cases (atomization energy, vibrational fundamentals, "
        "dipole, IE, etc.) and render them in the summary report. "
        "Cached after first fetch — see vibeqc.fetch.references.",
    )
    p.add_argument(
        "--rsgdf-tail-ke-cutoff",
        type=float,
        default=None,
        help=(
            "Pass an explicit high-G RSGDF tail kinetic-energy cutoff to "
            "vibe-qc periodic GDF rows. Dense-core Gamma RSGDF auto-sizes "
            "this by default; explicit undersized values are diagnostic."
        ),
    )
    args = p.parse_args(argv)

    if args.target != "dev":
        raise SystemExit(
            "--target release / both not yet implemented (wave 3); "
            "use --target dev for wave 1."
        )

    mol_triples, per_triples = _selected_triples(
        systems=args.systems,
        bases=args.bases,
        methods=args.methods,
        smoke=bool(args.smoke),
    )
    if not mol_triples and not per_triples:
        raise SystemExit(
            "no cases selected after filtering — check --systems / --bases / --methods."
        )
    if args.list_cases:
        _print_case_list(mol_triples, per_triples)
        return 0

    run_id = args.run_id or make_run_id()
    output_root = resolve_output_root(args.output_root, create=True)
    run_dir = output_root / run_id
    (run_dir / "verbose").mkdir(parents=True, exist_ok=True)
    (run_dir / "cases").mkdir(parents=True, exist_ok=True)

    env = collect_env(REPO_ROOT)
    write_env_json(run_dir / "env.json", env)
    _write_manifest(
        run_dir=run_dir,
        output_root=output_root,
        run_id=run_id,
        env=env,
        molecule_triples=mol_triples,
        periodic_triples=per_triples,
        cases=[],
        status="dry_run" if args.dry_run else "running",
        dry_run=bool(args.dry_run),
    )

    print(f"[regression] run_id={run_id}")
    print(
        f"[regression] cases  = {len(mol_triples) + len(per_triples)} "
        f"({len(mol_triples)} molecule + {len(per_triples)} periodic)"
    )
    print(f"[regression] output = {run_dir}")
    if args.output_qvf:
        print("[regression] qvf    = enabled (cases/*/vibeqc/vibeqc.qvf)")

    if args.dry_run:
        write_csv(run_dir / "results.csv", [])
        _write_dry_run_summary(
            out_path=run_dir / "summary.md",
            run_id=run_id,
            run_dir=run_dir,
            mol_triples=mol_triples,
            per_triples=per_triples,
        )
        _write_manifest(
            run_dir=run_dir,
            output_root=output_root,
            run_id=run_id,
            env=env,
            molecule_triples=mol_triples,
            periodic_triples=per_triples,
            cases=[],
            status="dry_run",
            dry_run=True,
        )
        print("[regression] dry run complete; no cases executed")
        return 0

    cases: List[CaseRecord] = []
    t_total0 = time.perf_counter()

    # CCCBDB ref-attach helper — set up once outside the loop. Looks
    # up each molecular system's CAS in the canonical-molecule table
    # and pulls the cached / live CCCBDB reference. Returns None for
    # systems not in the canonical set (caller treats as "no reference
    # available", report omits that row).
    _attach_ref = _build_reference_attacher(args.include_experimental_reference)

    # Molecules first — small and fast; periodic systems can take orders
    # of magnitude longer, so failing-fast on molecule plumbing is cheap.
    for s, b, m in mol_triples:
        print(f"[regression]   running molecule {s} / {b} / {m} ...")
        try:
            case = run_one_molecule_case_dev(
                run_id=run_id,
                system_id=s,
                basis=b,
                method_id=m,
                target="dev",
                env=env,
                run_dir=run_dir,
                case_dir=run_dir / "cases" / _case_id(s, b, m, (0, 0, 0)),
                output_qvf=args.output_qvf,
            )
        except Exception as exc:
            print(f"[regression]   {s}/{b}/{m}: ABORT ({type(exc).__name__}: {exc})")
            case = _error_case_record(
                run_id=run_id,
                target="dev",
                system_id=s,
                basis=b,
                method_id=m,
                family="molecule",
                kmesh=(0, 0, 0),
                note=f"{type(exc).__name__}: {exc}",
            )
            case_dir = run_dir / "cases" / case.label
            _write_case_notes(case_dir, case)
        if _attach_ref is not None:
            case.experimental_reference = _attach_ref(s)
        cases.append(case)
        for r in case.rows:
            _print_row_progress(r)

    for s, b, m in per_triples:
        print(f"[regression]   running periodic {s} / {b} / {m} ...")
        try:
            spec_for_dir = _load_periodic_spec(s)
            case = run_one_periodic_case_dev(
                run_id=run_id,
                system_id=s,
                basis=b,
                method_id=m,
                target="dev",
                env=env,
                run_dir=run_dir,
                case_dir=run_dir / "cases" / _case_id(
                    s, b, m, spec_for_dir.default_kmesh,
                ),
                include_crystal=args.include_crystal,
                include_cp2k=args.include_cp2k,
                rsgdf_tail_ke_cutoff=args.rsgdf_tail_ke_cutoff,
                output_qvf=args.output_qvf,
            )
        except Exception as exc:
            print(f"[regression]   {s}/{b}/{m}: ABORT ({type(exc).__name__}: {exc})")
            case = _error_case_record(
                run_id=run_id,
                target="dev",
                system_id=s,
                basis=b,
                method_id=m,
                family="periodic",
                kmesh=(1, 1, 1),
                note=f"{type(exc).__name__}: {exc}",
            )
            case_dir = run_dir / "cases" / case.label
            _write_case_notes(case_dir, case)
        cases.append(case)
        for r in case.rows:
            _print_row_progress(r)

    csv_path = run_dir / "results.csv"
    write_csv(csv_path, [r for c in cases for r in c.rows])

    summary_path = run_dir / "summary.md"
    render_summary(
        run_id=run_id,
        env=env,
        cases=cases,
        out_path=summary_path,
        csv_path=csv_path,
    )
    _write_manifest(
        run_dir=run_dir,
        output_root=output_root,
        run_id=run_id,
        env=env,
        molecule_triples=mol_triples,
        periodic_triples=per_triples,
        cases=cases,
        status="complete",
        dry_run=False,
    )

    t_total = time.perf_counter() - t_total0
    print(f"[regression] done in {t_total:.1f}s")
    print(f"[regression] CSV     = {csv_path}")
    print(f"[regression] summary = {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
