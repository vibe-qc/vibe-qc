"""C++ DF-UCCSD / UCCSD(T) pinned to the in-repo spin-orbital anchor.

The C++ kernel (cpp/src/uccsd.cpp) evaluates the Stanton-Gauss-Watts-Bartlett
(1991) working equations directly over spin orbitals on a UHF reference. The
spin-orbital original lives in ``vibeqc.dlpno._ccsd_ref.run_ref_uccsd`` (the
unrestricted sibling of the FCI-anchored ``run_ref_ccsd``); it is itself
validated against PySCF cc.UCCSD/UCCSD(T) to < 1e-3 uHa on CH3/NH2/O2
(exact integrals; 2026-06-17). These tests pin the C++ correlation and (T)
energies to that anchor on small open-shell radicals, so the C++ port cannot
drift from the validated equation set.

Always-on gate: no external QC program is imported (CLAUDE.md section 10);
the anchor is vibe-qc's own spin-orbital kernel. The out-of-process PySCF
cross-check lives in tests/test_uccsd.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import BasisSet, UHFOptions, run_uhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.cc import CCSDOptions, run_uccsd
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno._ccsd_ref import run_ref_uccsd

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"
MICRO_HA = 1e-6

# Small open-shell radicals (atoms in Bohr; multiplicity).
OH_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.0, 0.969 * ANGSTROM_TO_BOHR]),
]
CH3_ATOMS = [
    (6, [0.0, 0.0, 0.0]),
    (1, [1.079 * ANGSTROM_TO_BOHR, 0.0, 0.0]),
    (1, [-0.5395 * ANGSTROM_TO_BOHR, 0.9344 * ANGSTROM_TO_BOHR, 0.0]),
    (1, [-0.5395 * ANGSTROM_TO_BOHR, -0.9344 * ANGSTROM_TO_BOHR, 0.0]),
]


def _uhf(atoms, basis_name, mult):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=mult)
    basis = BasisSet(mol, basis_name)
    opts = UHFOptions()
    opts.density_fit = True
    opts.aux_basis = AUX
    opts.max_iter = 300
    opts.level_shift = 0.3
    opts.conv_tol_energy = 1e-11
    uhf = run_uhf(mol, basis, opts)
    assert uhf.converged, f"UHF not converged: {basis_name}"
    return mol, basis, uhf


def _anchor(mol, basis, uhf, *, triples):
    """Spin-orbital reference correlation / (T) on the same UHF + DF."""
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    Ca = np.asarray(uhf.mo_coeffs_alpha)
    Cb = np.asarray(uhf.mo_coeffs_beta)
    Fa = np.asarray(uhf.fock_alpha)
    Fb = np.asarray(uhf.fock_beta)
    n_elec = mol.n_electrons()
    two_s = mol.multiplicity - 1
    n_a = (n_elec + two_s) // 2
    n_b = n_elec - n_a
    ref = run_ref_uccsd(
        Ca.T @ Fa @ Ca, Cb.T @ Fb @ Cb,
        df.mo_transform(Ca, Ca), df.mo_transform(Cb, Cb),
        n_a, n_b, e_hf=uhf.energy, compute_triples=triples,
    )
    assert ref.converged
    return ref


def _cpp(
    mol,
    basis,
    uhf,
    *,
    triples,
    n_frozen=0,
    memory_mode="auto",
    memory_bytes=0,
    max_threads=0,
):
    opts = CCSDOptions(aux_basis=AUX, compute_triples=triples,
                       n_frozen_core=n_frozen)
    opts.triples_memory_mode = memory_mode
    opts.requested_memory_bytes = memory_bytes
    opts.triples_max_threads = max_threads
    return run_uccsd(mol, basis, uhf, opts)


class TestCppAnchorCH3:
    @pytest.fixture(scope="class")
    @classmethod
    def ch3(cls):
        return _uhf(CH3_ATOMS, "sto-3g", 2)

    def test_uccsd_matches_anchor(self, ch3):
        mol, basis, uhf = ch3
        ref = _anchor(mol, basis, uhf, triples=False)
        res = _cpp(mol, basis, uhf, triples=False)
        assert res.converged
        assert res.n_iter < 60
        assert abs(res.e_ccsd_correlation - ref.e_corr) < 5.0 * MICRO_HA

    def test_triples_matches_anchor(self, ch3):
        mol, basis, uhf = ch3
        ref = _anchor(mol, basis, uhf, triples=True)
        res = _cpp(mol, basis, uhf, triples=True)
        assert res.converged
        assert abs(res.e_ccsd_correlation - ref.e_corr) < 5.0 * MICRO_HA
        assert abs(res.e_t - ref.e_t) < 1.0 * MICRO_HA
        # (T) lowers the correlation energy further (more negative)
        assert res.e_t < 0.0

    def test_result_identities(self, ch3):
        mol, basis, uhf = ch3
        res = _cpp(mol, basis, uhf, triples=True)
        assert abs(res.e_ccsd - (res.e_hf + res.e_ccsd_correlation)) < 1e-12
        assert abs(res.e_ccsd_t - (res.e_ccsd + res.e_t)) < 1e-12
        assert res.e_total == res.e_ccsd_t
        assert res.t1_norm > 0.0
        assert res.t2_norm > 0.0

    def test_direct_triples_honours_budget_and_matches_fast(self, ch3):
        mol, basis, uhf = ch3
        fast = _cpp(
            mol,
            basis,
            uhf,
            triples=True,
            memory_mode="fast",
            max_threads=2,
        )
        direct = _cpp(
            mol,
            basis,
            uhf,
            triples=True,
            memory_mode="direct",
            memory_bytes=64 * 1024**2,
            max_threads=2,
        )
        assert direct.triples_memory_mode_used == "direct"
        assert direct.triples_tile_size_used == 0
        # The option caps worker count but does not override a lower global
        # OpenMP limit established by an earlier runner call.
        assert 1 <= direct.triples_threads_used <= 2
        assert 0 < direct.triples_workspace_bytes <= 64 * 1024**2
        assert direct.triples_workspace_bytes < fast.triples_workspace_bytes
        assert direct.triples_disk_bytes == 0
        assert direct.e_t == pytest.approx(fast.e_t, abs=1e-12)

        auto = _cpp(
            mol,
            basis,
            uhf,
            triples=True,
            memory_mode="auto",
            memory_bytes=direct.triples_workspace_bytes,
            max_threads=2,
        )
        assert auto.triples_memory_mode_used == "direct"
        assert auto.triples_workspace_bytes <= direct.triples_workspace_bytes
        assert auto.e_t == pytest.approx(fast.e_t, abs=1e-12)

        blocked = _cpp(
            mol,
            basis,
            uhf,
            triples=True,
            memory_mode="blocked",
            memory_bytes=direct.triples_workspace_bytes,
            max_threads=2,
        )
        # The current spin-orbital kernel has no useful virtual tile between
        # its dense and scalar forms, so blocked truthfully realizes direct.
        assert blocked.triples_memory_mode_used == "direct"
        assert blocked.triples_workspace_bytes <= direct.triples_workspace_bytes
        assert blocked.e_t == pytest.approx(fast.e_t, abs=1e-12)

    def test_open_shell_budget_floor_and_disk_mode_fail_closed(self, ch3):
        mol, basis, uhf = ch3
        with pytest.raises(RuntimeError, match="cannot hold"):
            _cpp(
                mol,
                basis,
                uhf,
                triples=True,
                memory_mode="direct",
                memory_bytes=1,
                max_threads=1,
            )
        with pytest.raises(RuntimeError, match=r"dense spin-orbital n\^4"):
            _cpp(
                mol,
                basis,
                uhf,
                triples=True,
                memory_mode="disk",
                memory_bytes=64 * 1024**2,
                max_threads=1,
            )


class TestCppAnchorOH:
    @pytest.fixture(scope="class")
    @classmethod
    def oh(cls):
        return _uhf(OH_ATOMS, "sto-3g", 2)

    def test_uccsd_t_matches_anchor(self, oh):
        mol, basis, uhf = oh
        ref = _anchor(mol, basis, uhf, triples=True)
        res = _cpp(mol, basis, uhf, triples=True)
        assert res.converged
        assert abs(res.e_ccsd_correlation - ref.e_corr) < 5.0 * MICRO_HA
        assert abs(res.e_t - ref.e_t) < 1.0 * MICRO_HA


@pytest.mark.slow
class TestCppAnchorDZ:
    @pytest.fixture(scope="class")
    @classmethod
    def ch3_dz(cls):
        return _uhf(CH3_ATOMS, "def2-svp", 2)

    def test_uccsd_t_matches_anchor_dz(self, ch3_dz):
        mol, basis, uhf = ch3_dz
        ref = _anchor(mol, basis, uhf, triples=True)
        res = _cpp(mol, basis, uhf, triples=True)
        assert res.converged
        assert abs(res.e_ccsd_correlation - ref.e_corr) < 5.0 * MICRO_HA
        assert abs(res.e_t - ref.e_t) < 1.0 * MICRO_HA


class TestFrozenCore:
    def test_frozen_core_window_is_consistent(self):
        """Freezing the C 1s removes a little correlation, never adds."""
        mol, basis, uhf = _uhf(CH3_ATOMS, "sto-3g", 2)
        res_ae = _cpp(mol, basis, uhf, triples=False, n_frozen=0)
        res_fc = _cpp(mol, basis, uhf, triples=False, n_frozen=1)
        assert res_fc.converged
        assert res_fc.e_ccsd_correlation > res_ae.e_ccsd_correlation
        assert res_fc.e_ccsd_correlation < 0.95 * res_ae.e_ccsd_correlation


def test_uccsd_rejects_unconverged_uhf():
    """UCCSD requires a converged UHF reference."""
    mol = Molecule([Atom(z, p) for z, p in OH_ATOMS], charge=0, multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    opts = UHFOptions()
    opts.max_iter = 1
    opts.use_diis = False
    uhf = run_uhf(mol, basis, opts)
    assert not uhf.converged
    with pytest.raises(RuntimeError, match="not converged"):
        run_uccsd(mol, basis, uhf, CCSDOptions(aux_basis=AUX))
