"""Tests for vibe_basis.backends.crystal.

Synthetic fixtures only — no CRYSTAL14 install needed. The shape
of each fixture matches the actual CRYSTAL14 14.1.0 output format
(spot-checked against MgO PBE/POB-TZVP runs on compute-host-d during the
2026-05-10 vq + run-crystal.sh validation).

A real-output validation against a fresh CRYSTAL14 run lives in
the recipe drivers, not here — to keep this suite laptop-fast and
CRYSTAL-free.
"""

from __future__ import annotations

import textwrap

import pytest

from vibe_basis.backends.crystal import (
    CrystalEnergyResult,
    parse_output,
    parse_output_file,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

SUCCESSFUL_HF = textwrap.dedent("""\
                    * MGO ROCKSALT — RHF/POB-TZVP                         *
                    *  ------------------------------                     *

     CYCLE   1 TOTAL ENERGY(HF)(AU)(   1)        -2.7400000000E+02 DE-1.0E+02
     CYCLE   2 TOTAL ENERGY(HF)(AU)(   2)        -2.7460000000E+02 DE-6.0E-01
     CYCLE   3 TOTAL ENERGY(HF)(AU)(   3)        -2.7467890123E+02 DE-7.9E-02
     CYCLE   4 TOTAL ENERGY(HF)(AU)(   4)        -2.7468175399E+02 DE-2.9E-03
     CYCLE   5 TOTAL ENERGY(HF)(AU)(   5)        -2.7468175400E+02 DE-9.5E-09

     == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -2.7468175400E+02

     EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
""")

SUCCESSFUL_DFT = textwrap.dedent("""\
     CYCLE   1 TOTAL ENERGY(DFT)(AU)(   1)        -2.7500000000E+02 DE-1.0E+02
     CYCLE  12 TOTAL ENERGY(DFT)(AU)(  12)        -2.7547759489E+02 DE-3.2E-11

     == SCF ENDED - CONVERGENCE ON DENSITY MATRIX      E(AU) -2.7547759489E+02

     EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
""")

SUCCESSFUL_HF_CRYSTAL23 = textwrap.dedent("""\
                    * MGO ROCKSALT — RHF/POB-TZVP (CRYSTAL23)             *
                    *  ------------------------------                     *

     CYCLE   1 TOTAL ENERGY(HF)(AU)(   1)        -2.7400000000E+02 DE-1.0E+02
     CYCLE   5 TOTAL ENERGY(HF)(AU)(   5)        -2.7468175400E+02 DE-9.5E-09

     == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -2.7468175400E+02

     EEEEEEEEEE TERMINATION  DATE 18 05 2026 TIME 15:14:00.2
""")  # NB: CRYSTAL23 terminator carries a date/time stamp, not a bare E-line

NON_CONVERGED = textwrap.dedent("""\
     CYCLE 100 TOTAL ENERGY(HF)(AU)( 100)        -2.7468175000E+02 DE-1.0E-04
     CYCLE 200 TOTAL ENERGY(HF)(AU)( 200)        -2.7468175000E+02 DE-1.0E-05

     == SCF ENDED - TOO MANY CYCLES      E(AU) -2.7468175000E+02

     EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
""")

EXPLICIT_NOT_CONVERGED = textwrap.dedent("""\
     CYCLE  50 TOTAL ENERGY(HF)(AU)(  50)        -2.7468175000E+02 DE-1.0E-04
     SCF NOT CONVERGED — basis-set linear dependence detected

     EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
""")

TRUNCATED_BEFORE_END = textwrap.dedent("""\
     CYCLE   3 TOTAL ENERGY(HF)(AU)(   3)        -2.7467890123E+02 DE-7.9E-02
     CYCLE   4 TOTAL ENERGY(HF)(AU)(   4)        -2.7468175399E+02 DE-2.9E-03

     == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -2.7468175399E+02
""")  # NB: no EEEE terminator — simulates wall-time-kill or vq STARVED

NO_ENERGY_AT_ALL = textwrap.dedent("""\
     This output contains everything except the actual SCF result.
     CRYSTAL14 might have crashed before reaching the SCF loop, e.g.
     basis-file parse error.

     EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
""")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_successful_hf_returns_ok_with_last_cycle_energy():
    result = parse_output(SUCCESSFUL_HF)
    assert isinstance(result, CrystalEnergyResult)
    assert result.ok
    assert result.energy == pytest.approx(-274.68175400, rel=1e-12)
    assert result.method == "HF"
    assert result.last_cycle == 5
    assert result.converged
    assert not result.truncated
    assert result.failure_mode is None


def test_successful_crystal23_termination_banner_is_recognized():
    # Regression for b5d4f20: CRYSTAL23 ends a converged run with
    # "EEEEEEEEEE TERMINATION  DATE ..." rather than a bare line of
    # E's. The old `^\\s*E+\\s*$` terminator regex missed it and
    # mis-classified every good CRYSTAL23 output as truncated.
    result = parse_output(SUCCESSFUL_HF_CRYSTAL23)
    assert result.ok
    assert not result.truncated
    assert result.failure_mode is None
    assert result.energy == pytest.approx(-274.68175400, rel=1e-12)
    assert result.method == "HF"


def test_successful_dft_classifies_method_label():
    result = parse_output(SUCCESSFUL_DFT)
    assert result.ok
    assert result.method == "DFT"
    assert result.last_cycle == 12
    assert result.energy == pytest.approx(-275.47759489, rel=1e-12)


def test_non_converged_via_too_many_cycles_is_not_ok():
    result = parse_output(NON_CONVERGED)
    assert not result.ok
    assert result.failure_mode == "non_converged"
    assert result.energy == pytest.approx(-274.68175000, rel=1e-12)
    assert not result.converged
    assert result.last_cycle == 200


def test_explicit_scf_not_converged_marker_is_caught():
    result = parse_output(EXPLICIT_NOT_CONVERGED)
    assert not result.ok
    assert result.failure_mode == "scf_not_converged"
    assert result.energy == pytest.approx(-274.68175000, rel=1e-12)


def test_truncated_output_without_eeee_terminator():
    result = parse_output(TRUNCATED_BEFORE_END)
    assert not result.ok
    assert result.failure_mode == "truncated"
    assert result.truncated
    assert result.energy == pytest.approx(-274.68175399, rel=1e-12)


def test_output_with_no_energy_line_returns_none_energy():
    result = parse_output(NO_ENERGY_AT_ALL)
    assert not result.ok
    assert result.energy is None
    assert result.failure_mode == "no_energy_line"
    assert result.last_cycle is None


def test_result_post_init_enforces_ok_invariant():
    with pytest.raises(AssertionError):
        CrystalEnergyResult(
            ok=True, energy=None,
            method="HF", last_cycle=5, converged=True, truncated=False,
            failure_mode=None, n_lines_scanned=10,
        )
    with pytest.raises(AssertionError):
        CrystalEnergyResult(
            ok=False, energy=-274.68,
            method="HF", last_cycle=5, converged=True, truncated=False,
            failure_mode=None,
            n_lines_scanned=10,
        )


def test_file_wrapper_reads_from_disk(tmp_path):
    out_path = tmp_path / "mgo_test.out"
    out_path.write_text(SUCCESSFUL_HF)
    result = parse_output_file(out_path)
    assert result.ok
    assert result.energy == pytest.approx(-274.68175400, rel=1e-12)


def test_file_wrapper_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_output_file(tmp_path / "does-not-exist.out")


def test_empty_file_behaves_like_no_energy():
    result = parse_output("")
    assert not result.ok
    assert result.energy is None
    assert result.failure_mode == "no_energy_line"
    assert result.n_lines_scanned == 0
