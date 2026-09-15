"""Integration smoke test: all wavefunction methods via run_job on H2/STO-3G."""

from __future__ import annotations

import pytest
from vibeqc import Atom, Molecule


@pytest.fixture
def h2():
    return Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])


@pytest.fixture
def lih():
    return Molecule([Atom(3, [0, 0, 0]), Atom(1, [0, 0, 3.015])])


@pytest.fixture
def water():
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.1173]),
            Atom(1, [0.0, 1.4315, -0.9314]),
            Atom(1, [0.0, -1.4315, -0.9314]),
        ],
        charge=0,
        multiplicity=1,
    )


class TestRunJobIntegration:
    def test_fci_via_run_job(self, h2, tmp_path):
        from vibeqc.runner import run_job

        r = run_job(
            h2,
            basis="sto-3g",
            method="fci",
            output=tmp_path / "fci",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )
        assert r.converged
        assert r.energy < 0
        assert len(r.ci_labels) == 4
        assert abs(r.energy - (-1.1372759436)) < 1e-8

    def test_selected_ci_via_run_job(self, h2, tmp_path):
        from vibeqc.runner import run_job

        r = run_job(
            h2,
            basis="sto-3g",
            method="selected_ci",
            output=tmp_path / "sci",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )
        assert r.converged
        assert r.energy < 0
        assert r.ci_labels is not None
        assert r.pt2_correction is not None
        assert abs(r.pt2_correction) < 1e-12
        out = (tmp_path / "sci.out").read_text()
        assert "E(variational):" in out
        assert "E(PT2 correction):" in out
        assert "0.0000000000 Ha" in out

    def test_dmrg_via_run_job(self, h2, tmp_path):
        from vibeqc.runner import run_job

        r = run_job(
            h2,
            basis="sto-3g",
            method="dmrg",
            output=tmp_path / "dmrg",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )
        assert r.converged
        assert r.energy < 0
        assert r.bond_dim is not None

    def test_v2rdm_via_run_job(self, h2, tmp_path):
        from vibeqc.runner import run_job

        r = run_job(
            h2,
            basis="sto-3g",
            method="v2rdm",
            output=tmp_path / "v2rdm",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )
        assert r.converged
        assert r.energy < 0
        assert r.rdm1 is not None

    def test_tc_via_run_job(self, h2, tmp_path):
        from vibeqc.runner import run_job

        r = run_job(
            h2,
            basis="sto-3g",
            method="transcorrelated_ci",
            output=tmp_path / "tc",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )
        assert r.converged
        assert r.energy < 0

    def test_auto_routing_fci(self, h2, tmp_path):
        """method='auto' with 2 electrons should route to FCI."""
        from vibeqc.runner import run_job

        r = run_job(
            h2,
            basis="sto-3g",
            method="auto",
            output=tmp_path / "auto",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )
        assert "fci" in r.method.lower()

    def test_out_file_has_solver_output(self, h2, tmp_path):
        """The .out file should contain solver diagnostics."""
        from vibeqc.runner import run_job

        run_job(
            h2,
            basis="sto-3g",
            method="fci",
            output=tmp_path / "outtest",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )
        out = (tmp_path / "outtest.out").read_text()
        assert "Non-mean-field solver result" in out
        assert "Total energy:" in out
        assert "Leading configurations:" in out
        assert "a|" in out  # SpinDet formatting

    def test_active_space_selected_ci(self, h2, tmp_path):
        """active_space=(2,2) with selected_ci should work (full space)."""
        from vibeqc.runner import run_job
        r = run_job(h2, basis="sto-3g", method="selected_ci",
                    active_space=(2, 2), output=tmp_path / "as_sci",
                    write_molden_file=False, write_xyz_file=False,
                    write_population_file=False, citations=False)
        assert r.converged
        assert r.energy < 0

    def test_active_space_fci(self, h2, tmp_path):
        """active_space=(2,2) with fci should work (full space)."""
        from vibeqc.runner import run_job
        r = run_job(h2, basis="sto-3g", method="fci",
                    active_space=(2, 2), output=tmp_path / "as_fci",
                    write_molden_file=False, write_xyz_file=False,
                    write_population_file=False, citations=False)
        assert r.converged
        assert len(r.ci_labels) >= 1

    def test_v2rdm_rejects_qg_constraints(self, h2, tmp_path):
        """v2RDM with constraints='pqg' must raise NotImplementedError."""
        from vibeqc.runner import run_job
        from vibeqc.solvers import V2RDMOptions
        with pytest.raises(NotImplementedError):
            run_job(h2, basis="sto-3g", method="v2rdm",
                    v2rdm_options=V2RDMOptions(constraints="pqg"),
                    output=tmp_path / "v2rdm_qg",
                    write_molden_file=False, write_xyz_file=False,
                    write_population_file=False, citations=False)

    def test_active_space_dmrg(self, h2, tmp_path):
        """active_space=(2,2) with dmrg should work."""
        from vibeqc.runner import run_job
        r = run_job(h2, basis="sto-3g", method="dmrg",
                    active_space=(2, 2), output=tmp_path / "as_dmrg",
                    write_molden_file=False, write_xyz_file=False,
                    write_population_file=False, citations=False)
        assert r.converged
        assert r.energy < 0

    def test_active_space_selected_ci_truncates_public_result(self, lih, tmp_path):
        """run_job(active_space=...) must expose an active-space determinant basis."""
        from vibeqc.runner import run_job
        from vibeqc.solvers import SelectedCIOptions

        r = run_job(
            lih,
            basis="sto-3g",
            method="selected_ci",
            active_space=(2, 2),
            selected_ci_options=SelectedCIOptions(
                target_size=10,
                max_iter=5,
                do_pt2_correction=False,
                verbose=0,
            ),
            output=tmp_path / "lih_as_sci",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )

        assert r.converged
        assert r.ci_labels is not None
        assert r.ci_labels
        # The default basis is unrestricted since #107, so a label is a
        # SpinDet ((alpha_occ), (beta_occ)); each spin string holds the one
        # active electron of that spin and every orbital index is active.
        for label in r.ci_labels:
            spin_strings = label if isinstance(label[0], tuple) else (label,)
            assert all(len(occ) == 1 for occ in spin_strings), label
            assert max(max(occ) for occ in spin_strings) < 2, label

    def test_active_space_dmrg_avoids_full_water_fock_space(self, water, tmp_path):
        """run_job(active_space=...) must apply before DMRG's spin-orbital guard."""
        import math

        from vibeqc.runner import run_job
        from vibeqc.solvers import DMRGOptions

        r = run_job(
            water,
            basis="sto-3g",
            method="dmrg",
            active_space=(2, 2),
            dmrg_options=DMRGOptions(
                bond_dim_schedule=[4],
                n_sweeps=2,
                conv_tol_energy=1e-5,
                verbose=0,
            ),
            output=tmp_path / "water_as_dmrg",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )

        assert r.bond_dim == 4
        assert r.n_iter >= 1
        assert math.isfinite(r.energy)

    def test_dmrg_rejects_large(self):
        """DMRG > 12 spin-orbitals must raise clear error."""
        from vibeqc.solvers._dmrg import DMRGSolver
        import numpy as np
        s = DMRGSolver()
        s._h1e = np.eye(14); s._h2e = np.zeros((14, 14, 14, 14))
        s._norb = 14; s._nelec = 2
        with pytest.raises(ValueError, match="Maximum is 12"):
            s._build_full_hamiltonian()
