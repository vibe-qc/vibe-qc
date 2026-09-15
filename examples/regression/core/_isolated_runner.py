"""Run a single (case, code) pair in subprocess isolation.

Invoked from `run_suite.py` via :func:`subprocess.run`. The whole point
of this layer is **fault isolation**: when the underlying runner
crashes the Python interpreter at C level (e.g. PySCF 2.13.0 on
compute-reference segfaulting inside ``mf.kernel()`` for some DFT cases — no
Python traceback because the crash is below the Python boundary), the
crash kills only this subprocess. The parent dispatcher sees a
non-zero returncode and synthesises a :class:`CodeRow` with
``status='error'`` and a "subprocess died with signal N" note, then
moves on to the next case.

Usage::

    python -m examples.regression.core._isolated_runner \
        <code> <spec_kind> <system_id> <basis> <method_id> \
        <log_path> <workdir_or_-> <run_id> <target> <code_version> \
        [artifact_dir] [write_qvf_artifact]

Writes the resulting CodeRow as JSON on a single ``ISOLATED-RUNNER-
RESULT:`` line on stdout (everything else on stdout is the runner's
own info / debug output and is captured by the parent for the
verbose log if needed).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


_RESULT_MARKER = "ISOLATED-RUNNER-RESULT:"


def _run_molecule(code: str, spec, basis: str, method, log_path: Path,
                  workdir: "Path | None", run_id: str, target: str,
                  code_version: str, artifact_dir: "Path | None" = None,
                  write_qvf_artifact: bool = False):
    if code == "vibeqc":
        from examples.regression.core import runner_vibeqc
        return runner_vibeqc.run_molecule_case(
            run_id=run_id, target=target, code_version=code_version,
            spec=spec, basis_name=basis, method=method,
            log_path=log_path, artifact_dir=artifact_dir,
            write_qvf_artifact=write_qvf_artifact,
        )
    if code == "pyscf":
        from examples.regression.core import runner_pyscf
        return runner_pyscf.run_molecule_case(
            run_id=run_id, target=target,
            spec=spec, basis_name=basis, method=method,
            log_path=log_path, artifact_dir=artifact_dir,
        )
    if code == "orca":
        from examples.regression.core import runner_orca
        return runner_orca.run_molecule_case(
            run_id=run_id, target=target,
            spec=spec, basis_name=basis, method=method,
            log_path=log_path, workdir=workdir,
            artifact_dir=artifact_dir,
        )
    raise ValueError(f"_isolated_runner: unknown code {code!r}")


def _run_periodic(code: str, spec, basis: str, method, log_path: Path,
                  run_id: str, target: str, code_version: str,
                  artifact_dir: "Path | None" = None,
                  write_qvf_artifact: bool = False):
    if code == "vibeqc":
        from examples.regression.core import runner_vibeqc
        tail_raw = os.environ.get("VIBEQC_REGRESSION_RSGDF_TAIL_KE_CUTOFF")
        tail_cutoff = float(tail_raw) if tail_raw else None
        return runner_vibeqc.run_periodic_case(
            run_id=run_id, target=target, code_version=code_version,
            spec=spec, basis_name=basis, method=method,
            kmesh=spec.default_kmesh, log_path=log_path,
            artifact_dir=artifact_dir,
            rsgdf_tail_ke_cutoff=tail_cutoff,
            write_qvf_artifact=write_qvf_artifact,
        )
    if code == "pyscf":
        from examples.regression.core import runner_pyscf
        return runner_pyscf.run_periodic_case(
            run_id=run_id, target=target,
            spec=spec, basis_name=basis, method=method,
            kmesh=spec.default_kmesh, log_path=log_path,
            artifact_dir=artifact_dir,
        )
    raise ValueError(
        f"_isolated_runner: code {code!r} is not wired for periodic cases"
    )


def main(argv: list[str]) -> int:
    if len(argv) < 11:
        print(
            "usage: _isolated_runner <code> <spec_kind> <system_id> "
            "<basis> <method_id> <log_path> <workdir_or_-> <run_id> "
            "<target> <code_version> [artifact_dir] [write_qvf_artifact]",
            file=sys.stderr,
        )
        return 2

    code, spec_kind, system_id, basis, method_id = argv[1:6]
    log_path = Path(argv[6])
    workdir = Path(argv[7]) if argv[7] != "-" else None
    run_id, target, code_version = argv[8], argv[9], argv[10]
    artifact_dir = Path(argv[11]) if len(argv) > 11 and argv[11] != "-" else None
    write_qvf_artifact = (
        len(argv) > 12 and argv[12].strip().lower() in {"1", "true", "yes", "on"}
    )

    if spec_kind == "molecule":
        mod = importlib.import_module(
            f"examples.regression.systems.molecules.{system_id}",
        )
    elif spec_kind == "periodic":
        mod = importlib.import_module(
            f"examples.regression.systems.periodic.{system_id}",
        )
    else:
        raise ValueError(f"_isolated_runner: unknown spec_kind {spec_kind!r}")
    spec = mod.SPEC

    from examples.regression.methods.catalog import METHODS
    method = METHODS[method_id]

    if spec_kind == "molecule":
        row = _run_molecule(
            code, spec, basis, method, log_path, workdir,
            run_id, target, code_version, artifact_dir, write_qvf_artifact,
        )
    else:
        row = _run_periodic(
            code, spec, basis, method, log_path,
            run_id, target, code_version, artifact_dir, write_qvf_artifact,
        )

    sys.stdout.write(_RESULT_MARKER + json.dumps(asdict(row)) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
