"""Megacell (finite-supercell) periodic MP2 — Stage 6, megacell route.

Validates :func:`vibeqc.periodic_megacell_mp2.megacell_mp2`:

1. the ``(1,1,1)`` megacell equals **molecular MP2 on the unit cell** under
   the same explicit all-electron recipe (the base-case oracle — a finite
   supercell is a molecular calculation, so no periodic exxdiv contaminates
   the MP2 denominator);
2. the per-cell correlation energy **converges** as the megacell grows (shrinking
   increments toward the thermodynamic limit);
3. bookkeeping (cells / atoms / total) is consistent.

System: a 1-D chain of one H₂ per cell (cell 6 bohr along z, 20-bohr vacuum in
x, y) — core-free, closed-shell, and weakly coupled cell-to-cell so the per-cell
energy converges quickly.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import MP2Options, RHFOptions, run_job, run_mp2, run_rhf
from vibeqc.periodic_megacell_mp2 import (
    build_supercell_molecule,
    megacell_mp2,
    megacell_mp2_tdl,
    megacell_run_job,
)


def _h2_chain_cell():
    """One H₂ per cell; periodic along z (6 bohr), vacuum in x, y (20 bohr)."""
    return vq.PeriodicSystem(
        3,
        np.diag([20.0, 20.0, 6.0]),
        [vq.Atom(1, [10.0, 10.0, 2.3]), vq.Atom(1, [10.0, 10.0, 3.7])],
    )


@pytest.fixture(scope="module")
def chain():
    return _h2_chain_cell()


@pytest.fixture(scope="module")
def per_cell(chain):
    """Per-cell correlation energy for growing megacells along the chain."""
    out = {}
    for n in (1, 2, 3):
        r = megacell_mp2(chain, "sto-3g", (1, 1, n))
        out[n] = r
    return out


def test_unit_cell_equals_molecular_mp2(chain):
    """(1,1,1) megacell == molecular MP2 on the unit cell, exactly."""
    mc = megacell_mp2(chain, "sto-3g", (1, 1, 1))

    mol = chain.unit_cell_molecule()
    basis = vq.BasisSet(mol, "sto-3g")
    rhf = run_rhf(mol, basis, RHFOptions())
    mp2_options = MP2Options()
    mp2_options.n_frozen_core = 0
    ref = run_mp2(mol, basis, rhf, mp2_options)

    assert mc.e_corr == pytest.approx(float(ref.e_correlation), abs=1e-10)
    assert mc.e_corr_per_cell == pytest.approx(float(ref.e_correlation), abs=1e-10)


def test_per_cell_correlation_converges(per_cell):
    """Per-cell correlation is negative and converges (shrinking increments)."""
    pc = {n: per_cell[n].e_corr_per_cell for n in (1, 2, 3)}
    assert all(pc[n] < 0.0 for n in (1, 2, 3))
    d21 = abs(pc[2] - pc[1])
    d32 = abs(pc[3] - pc[2])
    # Converging toward the TDL: the cell-to-cell increment shrinks.
    assert d32 < d21
    # And it has not blown up — per-cell stays within a sane window of cell 1.
    assert abs(pc[3] - pc[1]) < abs(pc[1])


def test_bookkeeping(per_cell):
    for n in (1, 2, 3):
        r = per_cell[n]
        assert r.n_cells == n
        assert r.n_atoms == 2 * n  # 2 H per cell
        assert r.e_total == pytest.approx(r.e_hf + r.e_corr, abs=1e-10)
        assert r.e_corr_per_cell == pytest.approx(r.e_corr / n, abs=1e-12)


def test_supercell_molecule_geometry(chain):
    """The 1-D replication places n H₂ units spaced by the 6-bohr cell."""
    mol = build_supercell_molecule(chain, (1, 1, 3))
    atoms = list(mol.atoms)
    assert len(atoms) == 6
    zs = np.sort([float(np.asarray(a.xyz)[2]) for a in atoms])
    # Three H₂ bonds at z ≈ {2.3,3.7}, {8.3,9.7}, {14.3,15.7} (cell length 6).
    np.testing.assert_allclose(
        zs, [2.3, 3.7, 8.3, 9.7, 14.3, 15.7], atol=1e-9
    )


def test_supercell_molecule_uses_lattice_columns_on_all_axes():
    """Skew-cell replicas follow PeriodicSystem lattice columns, not rows."""
    lattice = np.array(
        [
            [4.0, 0.7, -0.2],
            [0.3, 5.0, 0.8],
            [0.1, -0.4, 6.0],
        ]
    )
    origin = np.array([0.25, -0.5, 1.0])
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, origin)])

    mol = build_supercell_molecule(system, (2, 2, 2))
    positions = np.array([atom.xyz for atom in mol.atoms], dtype=float)
    expected = np.array(
        [
            origin
            + i * lattice[:, 0]
            + j * lattice[:, 1]
            + k * lattice[:, 2]
            for i in range(2)
            for j in range(2)
            for k in range(2)
        ]
    )

    np.testing.assert_allclose(positions, expected, atol=1e-12, rtol=0.0)


def test_megacell_dlpno_ccsd_t_matches_molecular(chain, tmp_path):
    """Periodic DLPNO-CCSD(T) via the megacell route (Nejad II): the (1,1,1)
    megacell equals molecular DLPNO-CCSD(T) under the same explicit historical
    recipe. def2-svp is used because its DLPNO RI auxiliary exists."""
    mc = megacell_run_job(
        chain, "def2-svp", (1, 1, 1), method="dlpno-ccsd(t)",
        output_dir=str(tmp_path),
    )
    assert mc.method == "dlpno-ccsd(t)"
    assert mc.n_cells == 1 and mc.n_atoms == 2

    mol = chain.unit_cell_molecule()
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

    legacy_options = LocalCCSDOptions(
        localise="boys",
        tcut_pno=1e-7,
        tcut_mkn=0.0,
        tcut_pairs=1e-4,
        residual_domain="pair",
    )
    ref = run_job(
        mol, basis="def2-svp", method="dlpno-ccsd(t)",
        output=str(tmp_path / "ref"), write_molden_file=False,
        write_xyz_file=False, write_population_file=False,
        citations=False, record_hostname=False,
        frozen_core=0, dlpno_ccsd_options=legacy_options,
    )
    assert mc.e_total == pytest.approx(float(ref.energy_total), abs=1e-10)
    assert mc.e_total_per_cell == pytest.approx(float(ref.energy_total), abs=1e-10)


def test_megacell_run_job_forwards_pre_sweep_correlation_contract(
    chain, tmp_path, monkeypatch
):
    """Every correlated facade call forwards its explicit periodic recipe."""
    from types import SimpleNamespace

    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions
    from vibeqc.dlpno.mp2 import DLPNOMP2Options
    from vibeqc.dlpno.uccsd_local_solver import LocalUCCSDOptions
    from vibeqc.dlpno.ump2 import DLPNOUMP2Options

    open_shell = vq.PeriodicSystem(
        3,
        np.eye(3) * 12.0,
        [vq.Atom(1, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    captured = []

    def fake_run_job(molecule, **kwargs):
        captured.append((int(molecule.multiplicity), dict(kwargs)))
        return SimpleNamespace(energy_total=-1.0)

    monkeypatch.setattr(vq, "run_job", fake_run_job)
    routes = (
        (chain, "mp2"),
        (chain, "ccsd(t)"),
        (chain, "dlpno-mp2"),
        (open_shell, "dlpno-mp2"),
        (chain, "dlpno-ccsd(t)"),
        (open_shell, "dlpno-ccsd"),
    )
    for index, (system, method) in enumerate(routes):
        megacell_run_job(
            system,
            "def2-svp",
            (1, 1, 1),
            method=method,
            output_dir=str(tmp_path / f"route-{index}"),
        )

    for _, kwargs in captured:
        assert kwargs["frozen_core"] == 0

    closed_mp2 = captured[2][1]["dlpno_options"]
    assert isinstance(closed_mp2, DLPNOMP2Options)
    assert closed_mp2.n_frozen is None
    assert (
        closed_mp2.localise,
        closed_mp2.tcut_pno,
        closed_mp2.tcut_pno_weak,
        closed_mp2.tcut_mkn,
        closed_mp2.tcut_pairs,
        closed_mp2.tcut_pairs_weak,
        closed_mp2.pno_norm,
    ) == ("boys", 1e-8, 1e-7, 1e-3, 1e-6, 1e-4, "legacy")

    open_mp2 = captured[3][1]["dlpno_options"]
    assert isinstance(open_mp2, DLPNOUMP2Options)
    assert (open_mp2.n_frozen, open_mp2.localise) == (None, "none")
    assert (open_mp2.tcut_pno, open_mp2.tcut_pairs) == (1e-8, 0.0)

    closed_cc = captured[4][1]["dlpno_ccsd_options"]
    assert isinstance(closed_cc, LocalCCSDOptions)
    assert closed_cc.n_frozen is None
    assert (
        closed_cc.localise,
        closed_cc.tcut_pno,
        closed_cc.tcut_mkn,
        closed_cc.tcut_pairs,
        closed_cc.residual_domain,
        closed_cc.pno_norm,
        closed_cc.pno_correction,
    ) == ("boys", 1e-7, 0.0, 1e-4, "pair", "legacy", False)

    open_cc = captured[5][1]["dlpno_ccsd_options"]
    assert isinstance(open_cc, LocalUCCSDOptions)
    assert open_cc.n_frozen is None
    assert (
        open_cc.localise,
        open_cc.tcut_pno,
        open_cc.tcut_mkn,
        open_cc.tcut_pairs,
    ) == ("boys", 1e-7, 0.0, 0.0)


def test_megacell_tdl_extrapolation(chain):
    """Bulk per-cell MP2 correlation = slope of E_corr(N) = b·N + a; the
    finite-N average is edge-contaminated and the surface term is positive."""
    tdl = megacell_mp2_tdl(chain, "sto-3g", (1, 2, 3, 4, 5))

    # E_corr(N) = b·N + a is an excellent model for the gapped chain
    # (measured max residual ~1.2e-6).
    assert tdl.fit_max_residual < 1e-5
    # The bulk per-cell (slope) equals the converged add-one-cell increment.
    assert abs(tdl.e_corr_per_cell_bulk - tdl.increments[-1]) < 5e-6
    # Add-one-cell increments converge (successive differences shrink).
    diffs = [
        abs(tdl.increments[i] - tdl.increments[i - 1])
        for i in range(1, len(tdl.increments))
    ]
    assert all(diffs[i] <= diffs[i - 1] + 1e-12 for i in range(1, len(diffs)))
    # Bulk is more correlated than the edge-contaminated finite-N average, and
    # the surface (edge) correction is positive (edge cells have fewer neighbours).
    assert tdl.e_corr_per_cell_bulk < tdl.e_corr_per_cell[-1]
    assert tdl.edge_correction > 0.0
    assert -0.02 < tdl.e_corr_per_cell_bulk < -0.005
