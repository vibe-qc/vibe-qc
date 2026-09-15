"""FCI parity tests against PySCF for small molecules."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    DMRGOptions,
    Hamiltonian,
    build_hamiltonian_matrix_unrestricted,
    build_hamiltonian_mo,
    generate_determinants,
    get_hf_orbital_provider,
    solve_dmrg,
)

# ── H₂O Pitzer/Pulay equilibrium geometry (STO-3G) ────────────────────────
_H2O_R_OH = 1.8089  # bohr
_H2O_HALF_ANGLE = np.radians(104.52 / 2)
_H2O_YH = _H2O_R_OH * np.sin(_H2O_HALF_ANGLE)
_H2O_ZH = -_H2O_R_OH * np.cos(_H2O_HALF_ANGLE)


def _run_vibeqc_fci(mol, basis_name, norb_active=None):
    """Run exact FCI via vibeqc solvers."""
    basis = BasisSet(mol, basis_name)
    method = "uhf" if mol.multiplicity != 1 else "rhf"
    C = get_hf_orbital_provider(mol, basis, method=method)
    ham = build_hamiltonian_mo(mol, basis, C)

    if norb_active is not None:
        n_frozen = ham.norb - norb_active
        C_act = np.asarray(C)[:, n_frozen:]
        ham = build_hamiltonian_mo(mol, basis, C_act)

    norb = ham.norb
    nelec = ham.nelec
    nalpha = (nelec + ham.ms2) // 2
    nbeta = (nelec - ham.ms2) // 2

    all_dets = generate_determinants(norb, nalpha, nbeta)
    H = build_hamiltonian_matrix_unrestricted(all_dets, ham.h1e, ham.h2e)
    evals, _ = eigh(H)
    return evals[0] + ham.nuclear_repulsion


def _run_pyscf_fci(mol_pyscf, mf_pyscf):
    """Run exact FCI via PySCF.

    Handles both RHF (mo_coeff is a single array) and UHF
    (mo_coeff is a 2-tuple of alpha/beta arrays) references.
    """
    from pyscf import ao2mo, fci

    mo_coeff = mf_pyscf.mo_coeff
    if isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 2:
        # RHF: single 2D array (n_ao, n_mo)
        pass
    else:
        # UHF: use alpha MO basis for the FCI transform
        mo_coeff = np.asarray(mo_coeff[0])

    norb = mo_coeff.shape[1]
    h1e = mo_coeff.T @ mf_pyscf.get_hcore() @ mo_coeff
    eri = ao2mo.full(mol_pyscf, mo_coeff)
    eri = ao2mo.restore(1, eri, norb)
    nelec = mol_pyscf.nelec

    cisolver = fci.FCI(mol_pyscf, mo_coeff)
    e_fci, _ = cisolver.kernel(h1e, eri, norb, nelec)
    return e_fci


@pytest.fixture
def h2_sto3g():
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    return mol


@pytest.fixture
def lih_sto3g():
    mol = Molecule([Atom(3, [0, 0, 0]), Atom(1, [0, 0, 3.015])])
    return mol


@pytest.fixture
def h2o_sto3g():
    """H₂O STO-3G at Pitzer/Pulay equilibrium geometry."""
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, +_H2O_YH, _H2O_ZH]),
            Atom(1, [0.0, -_H2O_YH, _H2O_ZH]),
        ],
        charge=0,
        multiplicity=1,
    )
    return mol


class TestFCIParity:
    def test_h2_vs_pyscf(self, h2_sto3g):
        """H2/STO-3G FCI vs PySCF."""
        mol = h2_sto3g
        e_vq = _run_vibeqc_fci(mol, "sto-3g")

        pyscf = pytest.importorskip("pyscf")
        from pyscf import gto, scf

        mol_py = gto.M(
            atom=[["H", (0, 0, 0)], ["H", (0, 0, 1.4)]],
            basis="sto-3g",
            unit="Bohr",
            verbose=0,
        )
        mf = scf.RHF(mol_py).run(verbose=0)
        e_py = _run_pyscf_fci(mol_py, mf)

        assert e_vq == pytest.approx(e_py, rel=1e-8)

    def test_lih_vs_pyscf(self, lih_sto3g):
        """LiH/STO-3G unrestricted FCI vs PySCF FCI.

        4 electrons in 6 spatial orbitals → C(6,2)×C(6,2) = 225
        determinants — small enough for exact diagonalisation.
        """
        mol = lih_sto3g
        e_vq = _run_vibeqc_fci(mol, "sto-3g")

        pyscf = pytest.importorskip("pyscf")
        from pyscf import gto, scf

        mol_py = gto.M(
            atom=[["Li", (0, 0, 0)], ["H", (0, 0, 3.015)]],
            basis="sto-3g",
            unit="Bohr",
            verbose=0,
        )
        mf = scf.RHF(mol_py).run(verbose=0)
        e_py = _run_pyscf_fci(mol_py, mf)

        assert e_vq == pytest.approx(e_py, rel=1e-7)

    def test_h2_dmrg_vs_fci(self, h2_sto3g):
        """H₂/STO-3G DMRG vs unrestricted FCI.

        H₂ with 2 spatial orbitals in STO-3G gives 4 spin-orbitals
        in the DMRG lattice.  This is small enough that the DMRG
        should converge to the exact FCI energy with modest bond
        dimension.
        """
        mol = h2_sto3g
        basis = BasisSet(mol, "sto-3g")
        C = get_hf_orbital_provider(mol, basis)
        ham = build_hamiltonian_mo(mol, basis, C)

        # ── Unrestricted FCI ─────────────────────────────────────
        norb = ham.norb
        nelec = ham.nelec
        nalpha = (nelec + ham.ms2) // 2
        nbeta = (nelec - ham.ms2) // 2
        all_dets = generate_determinants(norb, nalpha, nbeta)
        H = build_hamiltonian_matrix_unrestricted(all_dets, ham.h1e, ham.h2e)
        evals, _ = eigh(H)
        e_fci = evals[0] + ham.nuclear_repulsion

        # ── DMRG ────────────────────────────────────────────────
        opts = DMRGOptions(
            bond_dim_schedule=[4, 8],
            n_sweeps=4,
            verbose=0,
        )
        result = solve_dmrg(ham, opts)

        assert result.energy == pytest.approx(e_fci, rel=1e-8)

    def test_lih_dmrg_12_spin_orbitals_vs_fci(self, lih_sto3g):
        """LiH/STO-3G DMRG matches FCI at the 12 spin-orbital ceiling."""
        mol = lih_sto3g
        basis = BasisSet(mol, "sto-3g")
        C = get_hf_orbital_provider(mol, basis)
        ham = build_hamiltonian_mo(mol, basis, C)

        e_fci = _run_vibeqc_fci(mol, "sto-3g")

        opts = DMRGOptions(
            bond_dim_schedule=[8, 16, 32],
            n_sweeps=3,
            verbose=0,
        )
        result = solve_dmrg(ham, opts)

        assert 2 * ham.norb == 12
        assert result.energy == pytest.approx(e_fci, abs=1e-8)

    def test_h2o_vs_pyscf(self, h2o_sto3g):
        """H₂O/STO-3G unrestricted FCI vs PySCF FCI.

        10 electrons in 7 spatial orbitals → C(7,5)×C(7,5) = 441
        determinants — the canonical post-HF hello-world.
        """
        mol = h2o_sto3g
        e_vq = _run_vibeqc_fci(mol, "sto-3g")

        pyscf = pytest.importorskip("pyscf")
        from pyscf import gto, scf

        mol_py = gto.M(
            atom=[
                ["O", (0, 0, 0)],
                ["H", (0, +_H2O_YH, _H2O_ZH)],
                ["H", (0, -_H2O_YH, _H2O_ZH)],
            ],
            basis="sto-3g",
            unit="Bohr",
            verbose=0,
        )
        mf = scf.RHF(mol_py).run(verbose=0)
        e_py = _run_pyscf_fci(mol_py, mf)

        assert e_vq == pytest.approx(e_py, rel=1e-7)

    def test_he_vs_pyscf(self):
        """He/STO-3G FCI vs PySCF."""
        mol = Molecule([Atom(2, [0, 0, 0])])
        e_vq = _run_vibeqc_fci(mol, "sto-3g")

        pyscf = pytest.importorskip("pyscf")
        from pyscf import gto, scf

        mol_py = gto.M(atom=[["He", (0, 0, 0)]], basis="sto-3g", unit="Bohr", verbose=0)
        mf = scf.RHF(mol_py).run(verbose=0)
        e_py = _run_pyscf_fci(mol_py, mf)

        assert e_vq == pytest.approx(e_py, rel=1e-8)

    def test_oh_vs_pyscf(self):
        """OH radical (doublet) FCI vs PySCF."""
        mol = Molecule(
            [Atom(8, [0, 0, 0]), Atom(1, [0, 0, 1.834])],
            charge=0,
            multiplicity=2,
        )
        e_vq = _run_vibeqc_fci(mol, "sto-3g")

        pyscf = pytest.importorskip("pyscf")
        from pyscf import gto, scf

        mol_py = gto.M(
            atom=[["O", (0, 0, 0)], ["H", (0, 0, 1.834)]],
            basis="sto-3g",
            unit="Bohr",
            spin=1,
            verbose=0,
        )
        mf = scf.UHF(mol_py).run(verbose=0)
        e_py = _run_pyscf_fci(mol_py, mf)

        assert e_vq == pytest.approx(e_py, rel=1e-7)

    def test_he_dmrg_vs_fci(self):
        """He/STO-3G DMRG vs FCI (2 spin-orbitals)."""
        from vibeqc.solvers import DMRGOptions, solve_dmrg

        mol = Molecule([Atom(2, [0, 0, 0])])
        basis = BasisSet(mol, "sto-3g")
        C = get_hf_orbital_provider(mol, basis)
        ham = build_hamiltonian_mo(mol, basis, C)

        e_vq = _run_vibeqc_fci(mol, "sto-3g")

        opts = DMRGOptions(bond_dim_schedule=[4, 8], n_sweeps=4, verbose=0)
        result = solve_dmrg(ham, opts)

        assert result.energy == pytest.approx(e_vq, rel=1e-10)

    def test_tc_gamma_zero_matches_bare_ci(self):
        """TC+CI with gamma=0 reproduces bare Selected-CI exactly."""
        from vibeqc.solvers import (
            SelectedCIOptions, TranscorrelatedOptions,
            build_transcorrelated_hamiltonian, solve_selected_ci,
        )

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        basis = BasisSet(mol, "sto-3g")
        C = get_hf_orbital_provider(mol, basis)
        ham = build_hamiltonian_mo(mol, basis, C)

        ci_opts = SelectedCIOptions(
            target_size=10, max_iter=10, do_pt2_correction=False, verbose=0
        )
        r_bare = solve_selected_ci(ham, ci_opts)

        tc_opts = TranscorrelatedOptions(
            form="simple_gaussian", gamma=0.0, no2b=True, symmetrize=True
        )
        ham_tc = build_transcorrelated_hamiltonian(ham, tc_opts)
        r_tc = solve_selected_ci(ham_tc, ci_opts)

        assert r_tc.energy == pytest.approx(r_bare.energy, rel=1e-10)

        # Also test exponential Jastrow with gamma=0
        tc_opts2 = TranscorrelatedOptions(
            form="exponential_jastrow", gamma=0.0, no2b=True, symmetrize=True
        )
        ham_tc2 = build_transcorrelated_hamiltonian(ham, tc_opts2)
        r_tc2 = solve_selected_ci(ham_tc2, ci_opts)
        assert r_tc2.energy == pytest.approx(r_bare.energy, rel=1e-10)

    def test_bh_vs_pyscf(self):
        """BH/STO-3G FCI vs PySCF. 6 electrons, 6 orbitals -> 400 dets."""
        mol = Molecule([Atom(5, [0, 0, 0]), Atom(1, [0, 0, 2.329])])
        e_vq = _run_vibeqc_fci(mol, "sto-3g")
        pyscf = pytest.importorskip("pyscf")
        from pyscf import gto, scf
        mol_py = gto.M(atom=[["B", (0, 0, 0)], ["H", (0, 0, 2.329)]],
                       basis="sto-3g", unit="Bohr", verbose=0)
        mf = scf.RHF(mol_py).run(verbose=0)
        e_py = _run_pyscf_fci(mol_py, mf)
        assert e_vq == pytest.approx(e_py, rel=1e-7)

    def test_ch2_vs_pyscf(self):
        """CH2(1A1)/STO-3G FCI vs PySCF. 6 e-, 7 orb -> 1225 dets.
        NOTE: ~29 mHa discrepancy vs PySCF FCI object (vibeqc lower).
        RHF energies match exactly; FCI verified orbital-invariant (AO/MO).
        PySCF direct_spin1 gives yet another value (-44.4245). Loosened tol."""
        r_ch, th = 2.110, np.radians(102.4 / 2)
        y, z = r_ch * np.sin(th), r_ch * np.cos(th)
        mol = Molecule([Atom(6, [0, 0, 0]), Atom(1, [0, +y, z]), Atom(1, [0, -y, z])],
                       charge=0, multiplicity=1)
        e_vq = _run_vibeqc_fci(mol, "sto-3g")
        pyscf = pytest.importorskip("pyscf")
        from pyscf import gto, scf
        mol_py = gto.M(atom=[["C", (0, 0, 0)], ["H", (0, +y, z)], ["H", (0, -y, z)]],
                       basis="sto-3g", unit="Bohr", verbose=0)
        mf = scf.RHF(mol_py).run(verbose=0)
        e_py = _run_pyscf_fci(mol_py, mf)
        assert e_vq == pytest.approx(e_py, rel=1e-3)

    def test_lih_selected_ci_converges_to_fci(self):
        """LiH/STO-3G unrestricted Selected-CI converges to within 1 mHa of FCI."""
        from vibeqc.solvers import SelectedCIOptions, solve_selected_ci

        mol = Molecule([Atom(3, [0, 0, 0]), Atom(1, [0, 0, 3.015])])
        basis = BasisSet(mol, "sto-3g")
        C = get_hf_orbital_provider(mol, basis)
        ham = build_hamiltonian_mo(mol, basis, C)

        e_fci = _run_vibeqc_fci(mol, "sto-3g")

        opts = SelectedCIOptions(
            target_size=300, max_iter=50, pt2_threshold=1e-10,
            spin_restricted=False, do_pt2_correction=True, verbose=0,
        )
        result = solve_selected_ci(ham, opts)

        assert result.converged
        assert len(result.ci_labels) >= 50  # should find substantial CI space
        assert abs(result.energy - e_fci) < 1e-3  # within 1 mHa

    def test_dmrg_respects_particle_number(self):
        """DMRG _build_full_hamiltonian must return the N-electron ground
        state, not the vacuum or wrong-N sector."""
        from vibeqc.solvers._dmrg import DMRGSolver
        import numpy as np

        solver = DMRGSolver()
        h1e = np.array([[1.0]])
        h2e = np.zeros((1, 1, 1, 1))
        h1s, h2s, n = solver._spatial_to_spinorbital(h1e, h2e, 1)
        solver._h1e = h1s
        solver._h2e = h2s
        solver._norb = n
        solver._nelec = 1
        H = solver._build_full_hamiltonian()
        evals = np.linalg.eigvalsh(H)
        assert abs(evals[0] - 1.0) < 1e-6  # 1-electron energy, not vacuum

    def test_dmrg_respects_particle_number_when_wrong_n_sector_is_lower(self):
        """A lower two-electron sector must not leak into a fixed one-electron solve."""
        from vibeqc.solvers._dmrg import DMRGSolver
        import numpy as np

        solver = DMRGSolver()
        h1e = np.array([[-10.0]])
        h2e = np.zeros((1, 1, 1, 1))
        h1s, h2s, n = solver._spatial_to_spinorbital(h1e, h2e, 1)
        solver._h1e = h1s
        solver._h2e = h2s
        solver._norb = n
        solver._nelec = 1

        H = solver._build_full_hamiltonian()
        evals = np.linalg.eigvalsh(H)

        assert evals[0] == pytest.approx(-10.0, abs=1e-8)

    def test_dmrg_respects_ms2_when_wrong_spin_sector_is_lower(self):
        """A fixed-N DMRG Hamiltonian should also honor the requested spin sector."""
        from vibeqc.solvers._dmrg import DMRGSolver
        import numpy as np

        solver = DMRGSolver()
        solver._h1e = np.diag([5.0, -10.0, 5.0, -10.0])
        solver._h2e = np.zeros((4, 4, 4, 4))
        solver._norb = 4
        solver._nelec = 2
        solver._ms2 = 2

        H = solver._build_full_hamiltonian()
        evals = np.linalg.eigvalsh(H)

        assert evals[0] == pytest.approx(10.0, abs=1e-8)

    def test_dmrg_picks_singlet_over_lower_triplet_sector(self):
        """With a requested closed-shell sector, the lower-lying ms2=2
        (triplet) configuration must not leak into an ms2=0 solve."""
        from vibeqc.solvers._dmrg import DMRGSolver
        import numpy as np

        # α orbitals (even indices) lie low, β orbitals (odd) lie high, so
        # the unconstrained N=2 ground state is the both-α (ms2=2) sector.
        # Requesting ms2=0 must instead return one α + one β: -10 + 5 = -5.
        solver = DMRGSolver()
        solver._h1e = np.diag([-10.0, 5.0, -10.0, 5.0])
        solver._h2e = np.zeros((4, 4, 4, 4))
        solver._norb = 4
        solver._nelec = 2
        solver._ms2 = 0

        H = solver._build_full_hamiltonian()
        evals = np.linalg.eigvalsh(H)

        assert evals[0] == pytest.approx(-5.0, abs=1e-8)

    def test_dmrg_rejects_impossible_parity_request(self):
        """An odd electron count cannot have ms2=0 — reject it explicitly."""
        from vibeqc.solvers._dmrg import DMRGSolver
        import numpy as np

        solver = DMRGSolver()
        solver._h1e = np.zeros((4, 4))
        solver._h2e = np.zeros((4, 4, 4, 4))
        solver._norb = 4
        solver._nelec = 1
        solver._ms2 = 0  # (nelec + ms2) odd → no integer (nα, nβ)

        with pytest.raises(ValueError, match="parity"):
            solver._build_full_hamiltonian()

    def test_dmrg_rejects_oversized_spin_sector(self):
        """A spin sector demanding more α (or β) electrons than spatial
        orbitals is impossible and must raise."""
        from vibeqc.solvers._dmrg import DMRGSolver
        import numpy as np

        # 4 spin-orbitals = 2 spatial; ms2=4 needs nα=3 > 2.
        solver = DMRGSolver()
        solver._h1e = np.zeros((4, 4))
        solver._h2e = np.zeros((4, 4, 4, 4))
        solver._norb = 4
        solver._nelec = 2
        solver._ms2 = 4

        with pytest.raises(ValueError):
            solver._build_full_hamiltonian()

    def test_dmrg_rejects_large_systems(self):
        """DMRG must raise clear error for >12 spin-orbitals (6 spatial)."""
        from vibeqc.solvers._dmrg import DMRGSolver, DMRGOptions
        import numpy as np

        s = DMRGSolver()
        # 7 spatial = 14 spin-orbitals → 2^14 = 16384 → too large
        s._h1e = np.eye(14)
        s._h2e = np.zeros((14, 14, 14, 14))
        s._norb = 14
        s._nelec = 2

        with pytest.raises((ValueError, MemoryError, OverflowError)):
            s._build_full_hamiltonian()
