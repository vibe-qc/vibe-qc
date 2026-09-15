"""ASE→PeriodicSystem lattice-vector convention regression.

ASE stores lattice vectors as the **rows** of ``atoms.cell`` (``atoms.cell[i]
== a_i``); vibe-qc's :class:`PeriodicSystem` stores them as the **columns** of
its lattice matrix (``cpp/include/vibeqc/periodic.hpp:32`` —
"Columns = Cartesian lattice vectors"; the real-space lattice sum computes
``r_cart = lattice @ index``, and only the column hypothesis
``L.T @ B / 2π == I`` reproduces :meth:`PeriodicSystem.reciprocal_lattice`).

The ASE→engine boundary helpers therefore **must transpose** the cell. Omitting
the transpose silently *transposes* every non-orthogonal cell, so the engine
integrates the wrong crystal — wrong k-paths, bands, forces and energies via the
ASE calculator path. Orthogonal/cubic cells (and, incidentally, the FCC/BCC
*primitive* cells used elsewhere in the suite, whose matrices are symmetric) are
transpose-invariant and hide the bug, which is why it survived undetected. This
module pins the convention at all three one-way ASE boundaries
(:func:`vibeqc.ase_periodic.atoms_to_periodic_system` and the GPW/GAPW
calculator converters) and checks — end to end — that the periodic SCF treats a
non-orthogonal cell as the *same physical crystal* whether it is built ASE-side
(rows) or engine-side (columns), and that the energy is invariant under a rigid
rotation of that crystal.

External validation (CLAUDE.md §7 / §10). On a dilute hexagonal He cell the
engine column lattice (columns = a_i) reproduces an out-of-process PySCF
KRHF/GDF reference (``cell.a = rows = a_i`` — PySCF's row convention) to
3.05 mHa on the *same* geometry in the converged/dilute limit, and the two codes
build identical ``lattice_vectors()``; that fixes the *direction* of the fix
(columns = a_i is physical, the transpose is the bug). PySCF is a build-time
parity reference, executed out-of-process and never imported by vibe-qc; the
reproduction lives in
``examples/regression/parity_hexagonal_lattice_vs_pyscf.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms  # noqa: E402
from ase.units import Bohr  # noqa: E402

import vibeqc as vq  # noqa: E402
from vibeqc.ase_periodic import atoms_to_periodic_system  # noqa: E402


# A genuinely non-orthogonal (hexagonal, γ = 120°) cell whose matrix is *not*
# symmetric, so the row↔column transpose actually changes the geometry. Two He
# at general positions make the periodic-image interaction — and hence the
# energy — sensitive to the lattice shape (the transpose moves this fixture's
# energy by ~10 mHa; see the parity script).
_A_ANG, _C_ANG = 3.2, 4.2


def _hex_atoms() -> Atoms:
    cell = np.array([[_A_ANG, 0.0, 0.0],
                     [-_A_ANG / 2.0, _A_ANG * np.sqrt(3.0) / 2.0, 0.0],
                     [0.0, 0.0, _C_ANG]])
    return Atoms("He2",
                 positions=[[0.0, 0.0, 0.0], [1.05, 0.55, 0.55]],
                 cell=cell, pbc=True)


def _gamma_rhf_energy(system: "vq.PeriodicSystem") -> float:
    """Γ-only periodic RHF energy (Ha) via the Ewald-3D driver — the blessed
    periodic-Coulomb path used by the regression runner."""
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 22.0
    opts.lattice_opts.nuclear_cutoff_bohr = 22.0
    opts.conv_tol_energy = 1e-10
    opts.max_iter = 300
    basis = vq.make_basis(system.unit_cell_molecule(), "sto-3g")
    result = vq.run_rhf_periodic_scf(
        system, basis, vq.KPoints.gamma(system), opts)
    assert result.converged, "fixture SCF did not converge"
    return float(result.energy)


def test_reciprocal_lattice_confirms_column_convention():
    """Guards the premise: PeriodicSystem stores lattice vectors as columns."""
    # Upper-triangular so L != L.T and the two hypotheses are distinguishable.
    L = np.array([[4.0, 1.0, 0.5],
                  [0.0, 3.0, 0.7],
                  [0.0, 0.0, 5.0]])
    B = np.asarray(vq.PeriodicSystem(3, L, []).reciprocal_lattice())
    # a_i · b_j = 2π δ_ij  with a_i = columns of L  ⟺  L.T @ B == 2π I.
    assert np.allclose(L.T @ B / (2.0 * np.pi), np.eye(3)), "columns hypothesis"
    assert not np.allclose(L @ B / (2.0 * np.pi), np.eye(3)), "rows hypothesis is wrong"


def test_atoms_to_periodic_system_puts_ase_rows_in_engine_columns():
    atoms = _hex_atoms()
    cell = np.asarray(atoms.cell.array, dtype=float)
    # Sanity: this fixture must be non-orthogonal, else the test is vacuous.
    assert not np.allclose(cell, cell.T), "fixture cell must be non-symmetric"

    L = np.asarray(atoms_to_periodic_system(atoms).lattice, dtype=float)
    for i in range(3):
        # ASE's i-th lattice vector (row i) must land in engine column i (bohr).
        assert np.allclose(L[:, i], cell[i] / Bohr), f"a_{i} not in column {i}"
    # And it must NOT be the untransposed matrix (the pre-fix bug) for this cell.
    assert not np.allclose(L, cell / Bohr), "cell was not transposed (bug)"
    assert np.allclose(L, (cell / Bohr).T), "engine lattice == ASE cell transposed"


def test_gpw_and_gapw_converters_use_column_convention():
    """The GPW and GAPW ASE calculators carry their own one-way converter with
    the identical convention; pin them too."""
    from vibeqc.ase_periodic_gapw import VibeqcGAPW
    from vibeqc.ase_periodic_gpw import VibeqcGPW

    atoms = _hex_atoms()
    cell = np.asarray(atoms.cell.array, dtype=float)
    for cls in (VibeqcGPW, VibeqcGAPW):
        L = np.asarray(cls._atoms_to_periodic_system(atoms).lattice, dtype=float)
        for i in range(3):
            assert np.allclose(L[:, i], cell[i] / Bohr), (
                f"{cls.__name__}: a_{i} not in column {i}")


def test_nonorthogonal_scf_matches_engine_native_geometry():
    """End-to-end: the SCF energy of the ASE-built non-orthogonal cell must
    equal that of the engine-native crystal built with columns = a_i. They are
    the same physical crystal, so the energies must agree to SCF tolerance; the
    pre-fix (transposed) boundary makes them disagree (~10 mHa for this cell)."""
    atoms = _hex_atoms()
    a_cols = np.asarray(atoms.cell.array, dtype=float).T / Bohr  # columns = a_i
    pos_bohr = atoms.positions / Bohr
    native = vq.PeriodicSystem(
        3, a_cols, [vq.Atom(int(z), list(p)) for z, p in zip(atoms.numbers, pos_bohr)])

    e_ase = _gamma_rhf_energy(atoms_to_periodic_system(atoms))
    e_native = _gamma_rhf_energy(native)
    assert e_ase == pytest.approx(e_native, abs=1e-7), (
        f"ASE-path E={e_ase:.8f} != engine-native E={e_native:.8f} Ha")


def test_nonorthogonal_energy_is_rotation_invariant():
    """A rigid rotation of the crystal cannot change its energy. Pre-fix, the
    transposed lattice breaks this (the rotated input maps to a different
    engine crystal); post-fix it holds to µHa."""
    atoms = _hex_atoms()
    e0 = _gamma_rhf_energy(atoms_to_periodic_system(atoms))

    theta = np.deg2rad(41.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rotated = Atoms(numbers=atoms.numbers,
                    positions=atoms.positions @ R.T,   # rotate each position vector
                    cell=atoms.cell.array @ R.T,        # rotate each lattice (row) vector
                    pbc=True)
    e_rot = _gamma_rhf_energy(atoms_to_periodic_system(rotated))
    assert e_rot == pytest.approx(e0, abs=1e-6), (
        f"energy not rotation-invariant: {e0:.8f} vs {e_rot:.8f} Ha")
