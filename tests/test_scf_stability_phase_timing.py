"""The post-convergence internal-stability phase is measured and reported.

GitLab issue #205 (UKS-STABILITY-ANALYSIS-COST-INVISIBLE). Since the
internal stability analysis became default-on, every converged UHF / UKS
job pays for a post-convergence Davidson solve that measured **0.8-3.0x
the wall of the entire SCF loop it certifies** on the BH9 03_8TS UKS /
PBE / def2-QZVPP repro (1000 bf; 4025 s of SCF iterations followed by
11373 s of stability analysis, compute-cluster-amd 64 threads, v0.15.136).

Nothing recorded that phase. ``perf.json`` had no entry for it, the
``.out`` had no row for it, and -- the part that actively misled -- the
runner divided the *whole* driver wall by ``n_iter``, so "SCF avg. per
iteration" silently absorbed it. Any wall sized from that average
under-estimated by 2-4x, which is how 59 of 60 UKS members of a wave
sized at 3x their measured runtime still died on the walltime limit.

The driver now charges the phase to ``stability_wall_s`` /
``stability_cpu_s`` on the result, the runner reports it on its own
``.out`` row and as its own ``perf.json`` phase, and the per-iteration
average is computed from the loop wall alone.

Scope note: this file pins *visibility*. Making the phase cheaper
(issue #205 closure criterion 3, "stability wall <= 1.0x SCF-loop wall")
is separate performance work and is not asserted here.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from vibeqc import (
    Atom,
    HubbardSite,
    InitialGuess,
    Molecule,
    RHFOptions,
    UHFOptions,
    UKSOptions,
    run_job,
    run_uhf,
    run_uks,
)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _oh_doublet() -> Molecule:
    """OH radical, R(O-H) = 0.970 A -- small, converges fast, and its
    converged solution is internally stable, so the analysis runs to a
    verdict without taking a corrective-restart branch."""
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 0.970 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=2,
    )


# ---------------------------------------------------------------------------
# 1. The driver charges the phase.
# ---------------------------------------------------------------------------

def test_uhf_stability_phase_is_charged_on_the_result() -> None:
    mol = _oh_doublet()
    opts = UHFOptions()
    opts.max_iter = 200
    res = run_uhf(mol, "sto-3g", opts)

    assert res.converged
    assert res.stability_checked
    # The Davidson solve cannot be free.
    assert res.stability_wall_s > 0.0
    assert math.isfinite(res.stability_wall_s)
    # CPU is summed over OpenMP threads, so it is >= 0 and may exceed wall.
    assert res.stability_cpu_s >= 0.0
    assert math.isfinite(res.stability_cpu_s)


def test_uks_stability_phase_is_charged_on_the_result() -> None:
    mol = _oh_doublet()
    opts = UKSOptions()
    opts.max_iter = 200
    opts.functional = "pbe"
    res = run_uks(mol, "sto-3g", opts)

    assert res.converged
    assert res.stability_checked
    assert res.stability_wall_s > 0.0
    assert res.stability_cpu_s >= 0.0


def test_stability_opt_out_charges_no_phase() -> None:
    """``stability_check=False`` does no analysis, so it owes no wall.

    Exact zero distinguishes "did not run" from a measured phase that merely
    happened to be short."""
    mol = _oh_doublet()
    opts = UHFOptions()
    opts.max_iter = 200
    opts.stability_check = False
    res = run_uhf(mol, "sto-3g", opts)

    assert res.converged
    assert not res.stability_checked
    assert res.stability_wall_s == 0.0
    assert res.stability_cpu_s == 0.0


def test_uhf_other_skipped_stability_routes_charge_exactly_zero() -> None:
    """Nonconvergence and unsupported +U response do no stability work."""
    mol = _oh_doublet()

    nonconverged_options = UHFOptions()
    nonconverged_options.max_iter = 1
    nonconverged_options.conv_tol_energy = 1e-14
    nonconverged_options.conv_tol_grad = 1e-14
    nonconverged = run_uhf(mol, "sto-3g", nonconverged_options)
    assert not nonconverged.converged
    assert not nonconverged.stability_checked
    assert nonconverged.stability_wall_s == 0.0
    assert nonconverged.stability_cpu_s == 0.0

    plus_u = run_uhf(
        mol,
        "sto-3g",
        dft_plus_u=[HubbardSite(0, 1, U_ev=1.0)],
    )
    assert plus_u.converged
    assert not plus_u.stability_checked
    assert plus_u.stability_wall_s == 0.0
    assert plus_u.stability_cpu_s == 0.0


def test_uks_skipped_stability_charges_exactly_zero_phase() -> None:
    """Opt-out and implicit unsupported routes did not run the phase."""
    mol = _oh_doublet()

    opted_out = UKSOptions()
    opted_out.functional = "PBE"
    opted_out.stability_check = False
    result = run_uks(mol, "sto-3g", opted_out)
    assert result.converged
    assert not result.stability_checked
    assert result.stability_wall_s == 0.0
    assert result.stability_cpu_s == 0.0

    unsupported = UKSOptions()
    unsupported.functional = "TPSS"
    result = run_uks(mol, "sto-3g", unsupported)
    assert result.converged
    assert not result.stability_checked
    assert result.stability_wall_s == 0.0
    assert result.stability_cpu_s == 0.0


# ---------------------------------------------------------------------------
# 2. The .out reports it, and stops folding it into the iteration average.
# ---------------------------------------------------------------------------

_TIMING_ROW = re.compile(
    r"^\s{2,}(?P<label>\S.*?)\s{2,}(?P<value>-?\d+\.\d+)(?:\s+\((?P<ann>[^)]*)\))?\s*$"
)
_TIMING_RENDER_RESOLUTION_S = 1e-3


def _rendered_reconstruction_slop(n_iter: int) -> float:
    """Worst-case roundoff when reconstructing a sum from timing rows.

    Each row is rendered independently to milliseconds.  The iteration
    average can round upward by half a millisecond ``n_iter`` times, the
    stability row can round upward once, and the total can round downward
    once.  One floating-point ulp keeps the inclusive decimal bound inclusive
    after parsing the rendered strings back to binary floats.
    """
    decimal_bound = 0.5 * _TIMING_RENDER_RESOLUTION_S * (n_iter + 2)
    return math.nextafter(decimal_bound, math.inf)


def _timing_rows(out_text: str) -> dict[str, float]:
    """Parse the runner's ``Timings (wall clock, ...)`` block."""
    rows: dict[str, float] = {}
    in_block = False
    for line in out_text.splitlines():
        if line.strip().startswith("Timings (wall clock"):
            in_block = True
            continue
        if in_block:
            match = _TIMING_ROW.match(line)
            if match is None:
                if rows:
                    break
                continue
            rows[match.group("label").strip()] = float(match.group("value"))
    return rows


def test_out_reports_the_stability_phase_on_its_own_row(
    tmp_path: Path,
) -> None:
    stem = tmp_path / "oh_uhf"
    # cc-pVDZ rather than STO-3G: the timing block renders milliseconds,
    # and an STO-3G stability solve can finish inside one rendered digit,
    # which would make the assertion below flaky rather than wrong.
    result = run_job(
        _oh_doublet(),
        basis="cc-pvdz",
        method="uhf",
        output=stem,
        progress=False,
        write_molden_file=False,
        write_xyz_file=False,
    )
    assert result.converged

    text = stem.with_suffix(".out").read_text()
    assert "SCF stability analysis" in text

    rows = _timing_rows(text)
    assert "SCF total" in rows
    assert "SCF stability analysis" in rows
    assert "SCF avg. per iteration" in rows
    assert rows["SCF stability analysis"] > 0.0

    # The average is now the LOOP wall per iteration: reconstructing the
    # loop from it must leave room for the stability phase inside the SCF
    # total. Before the fix this identity read
    # ``avg * n_iter == SCF total`` and the phase was invisible.
    n_iter = int(result.n_iter)
    assert n_iter > 0
    reconstructed = rows["SCF avg. per iteration"] * n_iter
    render_slop = _rendered_reconstruction_slop(n_iter)
    assert reconstructed <= rows["SCF total"] + render_slop
    assert (
        reconstructed + rows["SCF stability analysis"]
        <= rows["SCF total"] + render_slop
    )


def test_uhf_opt_out_emits_no_stability_row_or_perf_phase(
    tmp_path: Path,
) -> None:
    """A skipped branch is not a zero-duration measured phase."""
    stem = tmp_path / "oh_uhf_opt_out"
    target = Path(str(stem) + ".perf.json")
    opts = UHFOptions()
    opts.stability_check = False
    result = run_job(
        _oh_doublet(),
        basis="sto-3g",
        method="uhf",
        uhf_options=opts,
        output=stem,
        perf_log=str(target),
        progress=False,
        write_molden_file=False,
        write_xyz_file=False,
    )

    assert result.converged
    assert not result.stability_checked
    assert result.stability_wall_s == 0.0
    assert result.stability_cpu_s == 0.0
    out_text = stem.with_suffix(".out").read_text()
    assert "SCF stability analysis" not in out_text
    payload = json.loads(target.read_text())
    names = [phase["name"] for phase in payload["phases"]]
    assert "scf.uhf.stability_analysis" not in names


def test_uks_out_and_perf_report_the_stability_phase(tmp_path: Path) -> None:
    """UKS gets the same end-to-end visibility contract as UHF."""
    stem = tmp_path / "oh_uks_perf"
    target = Path(str(stem) + ".perf.json")
    opts = UKSOptions()
    opts.functional = "PBE"
    result = run_job(
        _oh_doublet(),
        basis="sto-3g",
        method="uks",
        uks_options=opts,
        output=stem,
        perf_log=str(target),
        progress=False,
        write_molden_file=False,
        write_xyz_file=False,
    )

    assert result.converged
    assert result.stability_checked
    assert result.stability_wall_s > 0.0
    assert "SCF stability analysis" in stem.with_suffix(".out").read_text()
    payload = json.loads(target.read_text())
    phases = {phase["name"]: phase for phase in payload["phases"]}
    stability = phases["scf.uks.stability_analysis"]
    assert stability["wall_s"] == pytest.approx(result.stability_wall_s)
    assert stability["cpu_s"] == pytest.approx(result.stability_cpu_s)


def test_uks_restart_timing_uses_initial_loop_iteration_divisor(
    tmp_path: Path,
) -> None:
    """A corrective SCF belongs to the measured stability phase."""
    stem = tmp_path / "h2_uks_restart_timing"
    mol = Molecule(
        [Atom(1, [0.0, 0.0, -2.0]), Atom(1, [0.0, 0.0, 2.0])]
    )
    opts = UKSOptions()
    opts.functional = "PBE"
    opts.initial_guess = InitialGuess.SAD
    opts.max_iter = 120
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-7
    result = run_job(
        mol,
        basis="sto-3g",
        method="uks",
        uks_options=opts,
        output=stem,
        progress=False,
        write_molden_file=False,
        write_xyz_file=False,
    )

    assert result.n_stability_restarts >= 1
    assert 0 < result.n_iter_before_stability < result.n_iter
    text = stem.with_suffix(".out").read_text()
    avg_match = next(
        match
        for line in text.splitlines()
        if (match := _TIMING_ROW.match(line)) is not None
        and match.group("label").strip() == "SCF avg. per iteration"
    )
    assert avg_match.group("ann") == (
        f"{result.n_iter_before_stability} iters"
    )
    assert avg_match.group("ann") != f"{result.n_iter} iters"

    rows = _timing_rows(text)
    reconstructed = (
        rows["SCF avg. per iteration"] * result.n_iter_before_stability
    )
    assert (
        reconstructed + rows["SCF stability analysis"]
        <= rows["SCF total"]
        + _rendered_reconstruction_slop(result.n_iter_before_stability)
    )


def test_rhf_out_has_no_stability_row(tmp_path: Path) -> None:
    """An explicit RHF opt-out does no stability work or timing."""
    stem = tmp_path / "h2o_rhf"
    opts = RHFOptions()
    opts.stability_check = False
    run_job(
        Molecule(
            [
                Atom(8, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 1.43, 1.11]),
                Atom(1, [0.0, -1.43, 1.11]),
            ]
        ),
        basis="sto-3g",
        method="rhf",
        rhf_options=opts,
        output=stem,
        progress=False,
        write_molden_file=False,
        write_xyz_file=False,
    )
    text = stem.with_suffix(".out").read_text()
    assert "SCF total" in text
    assert "SCF stability analysis" not in text


# ---------------------------------------------------------------------------
# 3. perf.json carries the phase, so a wall can be planned from it.
# ---------------------------------------------------------------------------

def test_perf_json_carries_the_stability_phase(tmp_path: Path) -> None:
    stem = tmp_path / "oh_uhf_perf"
    target = Path(str(stem) + ".perf.json")
    run_job(
        _oh_doublet(),
        basis="sto-3g",
        method="uhf",
        output=stem,
        perf_log=str(target),
        progress=False,
        write_molden_file=False,
        write_xyz_file=False,
    )
    assert target.is_file()

    payload = json.loads(target.read_text())
    phases = {phase["name"]: phase for phase in payload["phases"]}
    assert "scf.uhf" in phases
    assert "scf.uhf.stability_analysis" in phases

    stability = phases["scf.uhf.stability_analysis"]
    assert stability["wall_s"] > 0.0
    # The phase is a sub-phase of the driver call, never larger than it.
    assert stability["wall_s"] <= phases["scf.uhf"]["wall_s"] + 1e-6
    # CPU is real, not a fabricated zero that would render as 0.00x
    # parallelism and mis-flag the phase as a serial hot spot.
    assert stability["cpu_s"] > 0.0


def test_perf_json_has_no_stability_phase_for_rhf(tmp_path: Path) -> None:
    stem = tmp_path / "h2o_rhf_perf"
    target = Path(str(stem) + ".perf.json")
    opts = RHFOptions()
    opts.stability_check = False
    run_job(
        Molecule(
            [
                Atom(8, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 1.43, 1.11]),
                Atom(1, [0.0, -1.43, 1.11]),
            ]
        ),
        basis="sto-3g",
        method="rhf",
        rhf_options=opts,
        output=stem,
        perf_log=str(target),
        progress=False,
        write_molden_file=False,
        write_xyz_file=False,
    )
    payload = json.loads(target.read_text())
    names = [phase["name"] for phase in payload["phases"]]
    assert "scf.rhf" in names
    assert not any(name.endswith(".stability_analysis") for name in names)
