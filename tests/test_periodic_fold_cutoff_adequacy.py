"""Pins how far the AO lattice sums actually reach at the default cutoff.

Background (BUG 124, 2026-08-06): every one of ten `vibeqc-lif-prim-*`
bundles failed with "NOT converged in 150 iterations" -- across GAPW and GPW,
across RHF / RKS-PBE / r2SCAN, with DFT+U, with D3BJ, and in the FD-Hessian
and stress outer loops. A failure that is invariant under the functional, the
J route, and the add-on stack is not a functional bug; it is upstream of all
of them. It is the shared real-space AO lattice sum.

The BIPOLE drivers already measure this: ``s_fold_truncation_drift`` compares
the Bloch-folded overlap against the same sum at 1.5x the cutoff, and the
drivers refuse to enter SCF above 1e-2 because an under-converged metric
admits spurious SCF states (`pbc_bipole.py`, c8 diagnosis 2026-06-10:
converged 0.70 Ha below the PySCF reference with the electron count off by
0.43). The documented reliability target is a drift below 1e-4.

What this file pins is *which cells the default cutoff is adequate for*. The
answer is basis- and element-dependent, and the standard periodic fixture
(MgO) is not representative:

    cell        drift @ 15 bohr (default)   @ 20      @ 25      @ 30
    LiF/STO-3G  8.4e-02                     2.2e-03   2.2e-05   3.6e-08
    LiH/STO-3G  7.1e-02                     1.7e-03   3.4e-05   6.6e-08
    MgO/STO-3G  2.3e-06                     ...

MgO clears the target at the default by two orders of magnitude, which is why
this went unnoticed; Li-bearing compact ionic cells miss it by ~840x. Li's
STO-3G valence shell is diffuse enough that cross-cell overlap survives well
past 15 bohr.

These are measurements of AO tail extent, not of a policy anyone chose. If the
default cutoff is ever raised (or made basis-adaptive),
``test_default_cutoff_is_inadequate_for_li_bearing_cells`` is the test that
should be updated deliberately, and its failure is the intended signal.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as _core
from vibeqc.pbc_bipole_common import s_fold_truncation_drift

ANG = 1.8897261246257702

#: The drift below which the BIPOLE drivers call the lattice sums converged.
RELIABLE_DRIFT = 1e-4
#: The drift above which they refuse to enter SCF at all.
UNRELIABLE_DRIFT = 1e-2


def _rocksalt(z1: int, z2: int, a_ang: float):
    a = a_ang * ANG
    lat = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return vq.PeriodicSystem(
        3, lat, [vq.Atom(z1, [0.0, 0.0, 0.0]), vq.Atom(z2, 0.5 * a * np.ones(3))]
    )


def _drift(system, basis_name: str, cutoff_bohr: float) -> float:
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    opts = _core.LatticeSumOptions()
    opts.cutoff_bohr = cutoff_bohr
    return float(s_fold_truncation_drift(basis, system, opts, k_points=None))


def test_default_lattice_cutoff_is_15_bohr():
    """The number the rest of this file is measured against."""
    assert _core.LatticeSumOptions().cutoff_bohr == pytest.approx(15.0)


def test_mgo_clears_the_target_at_the_default_cutoff():
    """Why the inadequacy went unnoticed: the standard fixture is fine."""
    assert _drift(_rocksalt(12, 8, 4.212), "sto-3g", 15.0) < RELIABLE_DRIFT


@pytest.mark.parametrize(
    "name,z1,z2,a_ang", [("LiF", 3, 9, 4.03), ("LiH", 3, 1, 4.084)]
)
def test_default_cutoff_is_inadequate_for_li_bearing_cells(name, z1, z2, a_ang):
    """BUG 124's root cause, stated as a measurement.

    Not merely above the 1e-4 reliability target -- above the 1e-2 threshold
    at which the BIPOLE drivers refuse to run, i.e. squarely in the regime
    documented to admit spurious SCF states.
    """
    drift = _drift(_rocksalt(z1, z2, a_ang), "sto-3g", 15.0)
    assert drift > UNRELIABLE_DRIFT, f"{name} drift {drift:.1e} at the default"


@pytest.mark.parametrize(
    "name,z1,z2,a_ang", [("LiF", 3, 9, 4.03), ("LiH", 3, 1, 4.084)]
)
def test_li_bearing_cells_are_converged_by_25_bohr(name, z1, z2, a_ang):
    """The cutoff those cells actually need -- the number to hand a user."""
    drift = _drift(_rocksalt(z1, z2, a_ang), "sto-3g", 25.0)
    assert drift < RELIABLE_DRIFT, f"{name} drift {drift:.1e} at 25 bohr"


@pytest.mark.parametrize("name,z1,z2,a_ang", [("LiF", 3, 9, 4.03)])
def test_drift_decreases_monotonically_with_cutoff(name, z1, z2, a_ang):
    """Convergence, not coincidence: a truncation artefact must shrink.

    Pins that the failure is under-converged support rather than a defect in
    the fold itself -- CLAUDE.md section 7's distinction.
    """
    system = _rocksalt(z1, z2, a_ang)
    drifts = [_drift(system, "sto-3g", c) for c in (12.0, 15.0, 20.0, 25.0, 30.0)]
    assert all(b < a for a, b in zip(drifts, drifts[1:])), drifts
    assert drifts[-1] < 1e-6


def test_bipole_refuses_li_cells_at_the_default_rather_than_running_them(tmp_path):
    """The fail-closed half of BUG 124.

    BIPOLE turns the under-converged metric into an actionable error before
    SCF. The GPW/GAPW route has no equivalent preflight and instead spends
    150 iterations not converging -- that asymmetry is the open half of the
    bug, tracked in handovers/HANDOVER_OPEN_BUGS_V015.md.
    """
    from vibeqc.pbc_bipole_common import BipoleFoldUnreliableError

    system = _rocksalt(3, 9, 4.03)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(BipoleFoldUnreliableError) as exc:
        vq.run_periodic_job(
            system, basis, method="RHF", jk_method="bipole", kpoints=(1, 1, 1),
            max_iter=2, output_qvf=False, citations=False, progress=False,
            # The preflight refuses after the .out and manifest are open, so a
            # tmp_path stem is what keeps the refusal out of the tree (#508).
            output=str(tmp_path / "bipole_fold_refusal"),
        )
    assert "fold truncation" in str(exc.value)
