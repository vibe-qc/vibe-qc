"""#316: the BIPOLE SCF energy of a dense ionic cell is relabelling invariant.

Every vacuum-box fixture is structurally blind to this class (the historical
index ball moved no atom into another index shell there), so this test uses
KI/STO-3G rocksalt and describes the same crystal twice: iodine at its
fractional (1/2, 1/2, 1/2) site, and iodine moved by the lattice vector
a1 + a2. On the shipped physical pair-support default the two descriptions
must give the same total energy to floating-point noise; the historical ball
differed by 2.8e-1 in S(Gamma) at this cutoff and by -1.08e-2 Ha in the
converged energy (#316 note 41317).

The converged pair at cutoff 15 agrees to 1.8e-12 Ha (8 iterations, about
7 minutes each at 3 threads; evidence on #316). This lane version stops
after the first iteration, where the SAD density is per-atom and therefore
exactly covariant, and compares the cold two-electron, nuclear and
one-electron operators through the iteration-1 energy and the exact
final-density evaluation at the cap. It is an operator-level invariance
check, not a convergence or reference comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import PeriodicRHFOptions

ANG2BOHR = 1.0 / 0.529177210903
CUTOFF_BOHR = 12.0


def _ki(shifted: bool):
    a = 7.066 * ANG2BOHR
    lattice = (a / 2.0) * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    iodine = np.array([a / 2.0] * 3)
    if shifted:
        iodine = iodine + lattice[:, 0] + lattice[:, 1]
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(19, [0.0, 0.0, 0.0]), vq.Atom(53, iodine)])
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _first_iteration(shifted: bool):
    system, basis = _ki(shifted)
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = CUTOFF_BOHR
    opts.lattice_opts.nuclear_cutoff_bohr = CUTOFF_BOHR
    opts.max_iter = 1
    result = vq.run_pbc_bipole_rhf(
        system, basis, vq.monkhorst_pack(system, [1, 1, 1]), opts, progress=False,
    )
    assert not result.converged
    assert result.resolved_domain.policy == "physical_pairs"
    return result


@pytest.mark.slow
def test_ki_sto3g_gamma_energy_is_invariant_under_lattice_relabelling():
    home = _first_iteration(False)
    moved = _first_iteration(True)
    # Both the loop's first-iteration energy (SAD density, cold build) and
    # the exact final-density evaluation at the cap must agree.
    assert home.scf_trace[0].energy == pytest.approx(moved.scf_trace[0].energy, abs=1e-9)
    assert float(home.energy) == pytest.approx(float(moved.energy), abs=1e-9)
    assert home.sr_image_extent_bohr == pytest.approx(moved.sr_image_extent_bohr, abs=1e-12)
    # The two descriptions are not the same computation: the AO lattice
    # enclosure differs, only the physical pair support does not.
    assert home.resolved_domain.settings_sha256 == moved.resolved_domain.settings_sha256
    # Sanity on the ionic cell: a 72-electron closed shell at a bound energy.
    assert -7445.0 < float(home.energy) < -7440.0
