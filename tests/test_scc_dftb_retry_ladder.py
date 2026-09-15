"""The SCC-DFTB Mermin retry ladder, on the systems that forced it wider.

A zero-temperature SCC-DFTB failure on a pi-conjugated N/O system is a
non-smooth charge-response fixed point, not a mixer defect; Fermi-Dirac
occupation smearing at low charge mixing is the standard DFTB+/tblite remedy
and is what ``run_job`` retries with (see
``handovers/HANDOVER_SCC_DFTB_CONVERGENCE.md`` and
``_RUNNER_SCC_DFTB_RETRY_SETTINGS``).

Issue #78 left four archived single-point rows failing closed after that
ladder shipped. Two of the four have physically impossible geometries and are
input defects, not convergence defects (thymine carries a C-C contact at
0.629 A and an H at 0.430 A from a ring carbon; the archived "S22
benzene-water" places the water in the benzene plane with an H-H contact at
0.600 A). The two whose geometries are sound are pinned here, because each
exposed a different missing rung:

* the S22 formic-acid dimer converges on the existing T = 0.0012 Ha rung but
  needs 573 iterations, so the historical 500-iteration retry budget turned a
  converging rung into a failure;
* adenine does not converge on any rung up to T = 0.0015 Ha even at 1500
  iterations, and needs the warmer T = 0.002 Ha (~632 K) rung added for it.

Both energies below are Mermin free energies at the rung that converged,
measured 2026-08-21. They are regression pins on vibe-qc's own SCC-DFTB, not
reference values: the geometries are the hand-built approximations from
qc-input-library's ``scripts/_geometries.py``, not published benchmark
structures.
"""

from __future__ import annotations

import pytest

from vibeqc import Atom, Molecule

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

# qc-input-library scripts/_geometries.py :: adenine()  (Angstrom)
ADENINE = [
    (7, 0.000, 1.280, 0.000),
    (6, 1.110, 0.552, 0.000),
    (6, -1.110, 0.552, 0.000),
    (7, 0.690, -0.806, 0.000),
    (6, -0.690, -0.806, 0.000),
    (7, 2.272, 1.250, 0.000),
    (7, -2.180, 1.400, 0.000),
    (6, -1.490, -1.710, 0.000),
    (7, -2.820, -1.120, 0.000),
    (6, 1.490, -1.710, 0.000),
    (1, 3.170, 0.792, 0.000),
    (1, 2.169, 2.252, 0.000),
    (1, -3.170, 0.900, 0.000),
    (1, -2.100, 2.390, 0.000),
    (1, -1.450, -2.790, 0.000),
]

# qc-input-library scripts/_geometries.py :: s22_formic_acid_dimer()  (Angstrom)
FORMIC_ACID_DIMER = [
    (6, -1.888896, -0.179692, 0.0),
    (8, -1.493280, 1.073689, 0.0),
    (8, -1.170435, -1.166590, 0.0),
    (1, -2.705488, -0.460805, 0.0),
    (1, -0.528749, 1.135085, 0.0),
    (6, 1.888896, 0.179692, 0.0),
    (8, 1.493280, -1.073689, 0.0),
    (8, 1.170435, 1.166590, 0.0),
    (1, 2.705488, 0.460805, 0.0),
    (1, 0.528749, -1.135085, 0.0),
]

# Mermin free energies at the converging rung, measured 2026-08-21.
PINNED_FREE_ENERGY_HA = {
    "adenine": -19.63356440,          # T = 0.002 Ha rung, 232 iterations
    "formic_acid_dimer": -16.43989628,  # T = 0.0012 Ha rung, 573 iterations
}
TOL_HA = 1e-5


def _molecule(geometry):
    return Molecule(
        [
            Atom(
                z,
                [
                    x * BOHR_PER_ANGSTROM,
                    y * BOHR_PER_ANGSTROM,
                    zz * BOHR_PER_ANGSTROM,
                ],
            )
            for (z, x, y, zz) in geometry
        ],
        charge=0,
        multiplicity=1,
    )


@pytest.mark.parametrize(
    "name,geometry",
    [
        ("adenine", ADENINE),
        ("formic_acid_dimer", FORMIC_ACID_DIMER),
    ],
)
def test_retry_ladder_converges_the_issue_78_rows(name, geometry, tmp_path):
    """``run_job`` reaches a converged, physical SCC-DFTB solution."""
    from vibeqc.runner import run_job

    result = run_job(
        _molecule(geometry),
        method="scc-dftb",
        output=tmp_path / name,
        optimize=False,
        hessian=False,
        write_xyz_file=False,
        output_qvf=False,
        citations=False,
        record_hostname=False,
    )
    energy = float(result.energy)
    # A bound neutral closed-shell molecule cannot have a positive SCC-DFTB
    # total energy.  The archived thymine row of #78 "converges" at +475 Ha
    # because its geometry buries an H 0.430 A inside a ring carbon, which is
    # the shape of failure this assertion exists to catch.
    assert energy < 0.0
    assert energy == pytest.approx(PINNED_FREE_ENERGY_HA[name], abs=TOL_HA)


def test_ladder_is_ordered_and_reaches_the_adenine_rung():
    """The rungs stay monotone in temperature and still include T = 0.002 Ha.

    Adenine stalls at -18.7886 / -18.9226 / -19.1639 Ha on the 0.001 / 0.0012 /
    0.0015 Ha rungs even at 1500 iterations, so removing the warmest rung
    silently reopens #78.
    """
    from vibeqc.semiempirical.runner import (
        _RUNNER_SCC_DFTB_RETRY_MAX_ITER,
        _RUNNER_SCC_DFTB_RETRY_SETTINGS,
    )

    temperatures = [
        rung["electronic_temperature"] for rung in _RUNNER_SCC_DFTB_RETRY_SETTINGS
    ]
    assert temperatures == sorted(temperatures)
    assert temperatures[-1] >= 0.002
    # The formic-acid dimer needs 573 iterations on its rung.
    assert _RUNNER_SCC_DFTB_RETRY_MAX_ITER >= 600
    assert all(
        rung["charge_mixing"] == 0.05 and not rung["use_diis"]
        for rung in _RUNNER_SCC_DFTB_RETRY_SETTINGS
    )
