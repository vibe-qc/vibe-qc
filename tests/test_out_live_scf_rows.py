"""The .out keeps the SCF rows a killed job had already completed (#482).

Before this, the iteration table was rendered from ``result.scf_trace`` only
*after* the native SCF returned, so a process killed inside the SCF never
reached the write: 372 of 373 walltime-killed production cases had a .out that
stopped at the memory-estimate banner. The structured sidecar kept its rows
(#25) but is not written unless ``structured_log`` is enabled.

The contract has two halves and both matter:

* a killed run's .out retains the rows that completed, and
* a completed run's .out is unchanged -- the rows are streamed into the same
  place the table occupied, and ``write_scf_trace`` closes that table instead
  of emitting a second one.

#482 hypothesised that the .out was "assembled at job end" and that an
incremental-flush hook would be an ask to the IO chat. That is not the
mechanism: ``write()`` + ``flush()`` reaches disk mid-run and survives SIGTERM.
The runner simply never wrote there.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import pytest

_GEOM = (
    "examples/molecular/mp2_benchmarks/s22/geometries/"
    "s22-05-uracil-dimer-hb-monoA.xyz"
)


def _iteration_rows(text: str) -> list[str]:
    """Rows of the .out SCF table: leading index then a negative energy."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                float(parts[1])
            except ValueError:
                continue
            if parts[1].startswith("-"):
                rows.append(line)
    return rows


def test_completed_run_out_still_carries_exactly_one_iteration_table(
    tmp_path, monkeypatch
):
    """Streaming must not duplicate the table on a normal run.

    The failure this guards is subtle: the rows are written twice, once live
    and once by ``write_scf_trace``, and every energy is still correct -- so
    only a count catches it.
    """
    import vibeqc as vq

    monkeypatch.chdir(tmp_path)
    geom = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _GEOM
    )
    mol = vq.Molecule.from_xyz(geom)
    vq.run_job(mol, basis="sto-3g", method="rhf", output="done")

    out = (tmp_path / "done.out").read_text()
    header_count = out.count("iter     energy (Ha)")
    assert header_count == 1, (
        f"{header_count} iteration-table headers in the .out; "
        "the live rows and write_scf_trace are both emitting one"
    )
    assert _iteration_rows(out), "a completed run lost its iteration table"


@pytest.mark.slow
def test_sigtermed_scf_retains_its_completed_rows_in_the_out(tmp_path):
    """A killed SCF's .out keeps the rows it had already produced.

    Spawned as a subprocess because the point is to kill it: an in-process
    SIGTERM would take pytest with it. Uracil dimer / cc-pVDZ at a tolerance it
    cannot reach keeps iterating until the signal arrives.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = textwrap.dedent(
        f"""
        import os, signal, threading, vibeqc as vq
        def kill_soon():
            import time; time.sleep(12.0)
            os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=kill_soon, daemon=True).start()
        o = vq.RHFOptions(); o.conv_tol_energy = 1e-12; o.max_iter = 200
        vq.run_job(
            vq.Molecule.from_xyz({os.path.join(repo, _GEOM)!r}),
            basis="cc-pvdz", method="rhf", rhf_options=o, output="killed",
        )
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path,
        capture_output=True, timeout=180,
    )
    assert proc.returncode == -signal.SIGTERM or proc.returncode == 143, (
        f"expected a SIGTERM exit, got {proc.returncode}; the job may have "
        "finished before the signal -- lengthen the sleep or tighten conv_tol"
    )

    out_path = tmp_path / "killed.out"
    assert out_path.exists(), "the killed job left no .out at all"
    rows = _iteration_rows(out_path.read_text())
    assert rows, (
        "the killed job's .out has no iteration rows -- it stops at the "
        "memory-estimate banner, which is the #482 symptom"
    )
