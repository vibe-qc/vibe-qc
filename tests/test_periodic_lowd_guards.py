"""Fast, actionable error on the BIPOLE periodic reference route for 1-D/2-D.

A stream-A benchmark blocker: ``jk_method="bipole"`` on a vacuum-padded 1-D/2-D
cell. The bipolar-expansion Fock build uses a 3-D Ewald/Madelung lattice sum
undefined for a low-dimensional cell; previously this surfaced only deep in the
SCF setup as a terse ``ValueError`` (``probe_charge_madelung_supercell requires
dim=3``). ``run_periodic_job`` now fails fast at the API level with an actionable
``NotImplementedError`` naming the low-D alternatives, before any setup.

(The companion GDF rsgdf low-D mesh blowup is handled separately by the
dimension-aware ``rsgdf_dense_g_mesh``, which meshes only the periodic
directions — covered by the GDF test suite, not here.)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, PeriodicSystem, run_periodic_job


def _h_chain_1d(a=7.9376582, vac=40.0):
    """A 1-D H₂ chain in a vacuum box (dim=1) — the stream-A h-chain shape."""
    return PeriodicSystem(
        1, np.diag([a, vac, vac]),
        [Atom(1, [0, 0, 0]), Atom(1, [0.0933333 * a, 0, 0])],
        charge=0, multiplicity=1)


def test_bipole_rejects_low_dim_at_api():
    """run_periodic_job(jk_method='bipole') on a 1-D cell fails fast and clearly."""
    sysp = _h_chain_1d()
    b = BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    t0 = time.time()
    with pytest.raises(NotImplementedError) as exc:
        run_periodic_job(
            sysp, b, method="RHF", jk_method="bipole", kpoints=(2, 1, 1),
            max_iter=5, write_molden_file=False, write_density=False,
            write_cif_file=False, write_xsf_structure_file=False,
            write_poscar_file=False, write_xyz_file=False,
            write_population_file=False)
    assert time.time() - t0 < 5.0                      # before any SCF setup
    msg = str(exc.value)
    assert "dim=1" in msg and "bipole" in msg
    assert "gdf" in msg                                # points at the low-D route
