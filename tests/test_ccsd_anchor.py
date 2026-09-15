"""C++ DF-CCSD / CCSD(T) pinned to the in-repo spin-orbital CCSD anchor.

The C++ kernel (cpp/src/ccsd.cpp) implements the closed-shell spin
integration of the Stanton-Gauss-Watts-Bartlett (1991) equations; the
spin-orbital original lives in ``vibeqc.dlpno._ccsd_ref`` and is itself
FCI-anchored (tests/test_dlpno_ccsd.py::TestAnchor). These tests pin the
C++ correlation energies to that anchor on H2O/STO-3G and H2O/def2-SVP,
and the (T) correction to the blockwise spin-integrated reference values
(cross-checked against the spin-orbital Raghavachari formulas at machine
precision, 2026-06-10).

History: before the 2026-06-10 rewrite the C++ kernel overshot the FCI
correlation energy by 41 mHa on H2O/STO-3G (unphysical; CCSD must
recover *less* correlation than FCI). These gates make that class of
defect unrepresentable.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import BasisSet, RHFOptions, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.cc import CCSDOptions, run_ccsd
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno._ccsd_ref import run_ref_ccsd

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"
MICRO_HA = 1e-6

H2O_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
]

# Full-space CASCI(7,10)/STO-3G correlation energy for the geometry above
# (vibe-qc direct CAS-CI engine, 2026-06-10); FCI for this basis.
E_FCI_CORR_H2O_STO3G = -0.055587908

# (T) references from the blockwise closed-shell spin integration of the
# Raghavachari 1989 formulas, evaluated on anchor-converged amplitudes and
# cross-checked against the brute-force spin-orbital evaluation to
# machine precision (2026-06-10 prototype harness, derivation documented
# in cpp/src/ccsd.cpp compute_triples).
E_T_H2O_STO3G = -0.000076577
E_T_H2O_DEF2SVP = -0.003222076


def _system(basis_name):
    mol = Molecule([Atom(z, p) for z, p in H2O_ATOMS], charge=0, multiplicity=1)
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    return mol, basis, rhf


def _anchor_corr(mol, basis, rhf):
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    C = np.asarray(rhf.mo_coeffs)
    F = np.asarray(rhf.fock)
    ref = run_ref_ccsd(
        C.T @ F @ C, df.mo_transform(C, C),
        n_occ=mol.n_electrons() // 2, e_hf=rhf.energy,
    )
    assert ref.converged
    return ref.e_corr


def _cpp_ccsd(mol, basis, rhf, *, triples):
    opts = CCSDOptions(
        aux_basis=AUX,
        compute_triples=triples,
        n_frozen_core=0,
    )
    return run_ccsd(mol, basis, rhf, opts)


def _cpp_ccsd_with_triples_plan(
    mol,
    basis,
    rhf,
    *,
    mode,
    tile=0,
    budget=64 * 1024**2,
    threads=2,
    scratch_directory="",
):
    opts = CCSDOptions(
        aux_basis=AUX,
        compute_triples=True,
        n_frozen_core=0,
    )
    opts.triples_memory_mode = mode
    opts.requested_memory_bytes = budget
    opts.triples_tile_size = tile
    opts.triples_max_threads = threads
    opts.triples_scratch_directory = scratch_directory
    return run_ccsd(mol, basis, rhf, opts)


class TestCppAnchorSTO3G:
    @pytest.fixture(scope="class")
    @classmethod
    def h2o(cls):
        return _system("sto-3g")

    def test_ccsd_matches_anchor_and_respects_fci_bound(self, h2o):
        mol, basis, rhf = h2o
        e_anchor = _anchor_corr(mol, basis, rhf)
        res = _cpp_ccsd(mol, basis, rhf, triples=True)
        assert res.converged
        assert res.n_iter < 50
        # pinned to the spin-orbital anchor
        assert abs(res.e_ccsd_correlation - e_anchor) < 5.0 * MICRO_HA
        # CCSD recovers less correlation than FCI, never more
        assert res.e_ccsd_correlation > E_FCI_CORR_H2O_STO3G

    def test_triples_matches_reference_and_stays_above_fci(self, h2o):
        mol, basis, rhf = h2o
        res = _cpp_ccsd(mol, basis, rhf, triples=True)
        assert res.converged
        assert abs(res.e_t - E_T_H2O_STO3G) < 0.5 * MICRO_HA
        # (T) closes part of the CCSD-to-FCI gap without crossing it
        e_ccsd_t_corr = res.e_ccsd_correlation + res.e_t
        assert e_ccsd_t_corr < res.e_ccsd_correlation
        assert e_ccsd_t_corr > E_FCI_CORR_H2O_STO3G

    def test_result_identities(self, h2o):
        mol, basis, rhf = h2o
        res = _cpp_ccsd(mol, basis, rhf, triples=True)
        assert abs(res.e_ccsd - (res.e_hf + res.e_ccsd_correlation)) < 1e-12
        assert abs(res.e_ccsd_t - (res.e_ccsd + res.e_t)) < 1e-12
        assert res.e_total == res.e_ccsd_t
        assert res.t1_norm > 0.0
        assert res.t2_norm > 0.0


@pytest.mark.slow
class TestCppAnchorDZ:
    @pytest.fixture(scope="class")
    @classmethod
    def h2o_dz(cls):
        return _system("def2-svp")

    def test_ccsd_matches_anchor_dz(self, h2o_dz):
        mol, basis, rhf = h2o_dz
        e_anchor = _anchor_corr(mol, basis, rhf)
        res = _cpp_ccsd(mol, basis, rhf, triples=True)
        assert res.converged
        assert abs(res.e_ccsd_correlation - e_anchor) < 5.0 * MICRO_HA

    def test_triples_matches_reference_dz(self, h2o_dz):
        mol, basis, rhf = h2o_dz
        res = _cpp_ccsd(mol, basis, rhf, triples=True)
        assert res.converged
        assert abs(res.e_t - E_T_H2O_DEF2SVP) < 1.0 * MICRO_HA


class TestFrozenCore:
    def test_frozen_core_window_is_consistent(self):
        """Freezing the O 1s removes a little correlation, never adds."""
        mol, basis, rhf = _system("sto-3g")
        res_ae = _cpp_ccsd(mol, basis, rhf, triples=False)
        opts = CCSDOptions(aux_basis=AUX, compute_triples=False,
                           n_frozen_core=1)
        res_fc = run_ccsd(mol, basis, rhf, opts)
        assert res_fc.converged
        # less correlation than all-electron, but the O 1s contribution
        # in a minimal basis is small (well under 10% of E_corr)
        assert res_fc.e_ccsd_correlation > res_ae.e_ccsd_correlation
        assert res_fc.e_ccsd_correlation < 0.9 * res_ae.e_ccsd_correlation
        assert res_fc.e_ccsd_correlation < -0.04


class TestTriplesMemoryPlans:
    @pytest.fixture(scope="class")
    @classmethod
    def h2o(cls):
        return _system("sto-3g")

    @pytest.fixture(scope="class")
    @classmethod
    def fast(cls, h2o):
        return _cpp_ccsd_with_triples_plan(*h2o, mode="fast")

    @pytest.mark.parametrize(
        ("mode", "tile", "expected_mode", "expected_tile"),
        [
            ("blocked", 1, "blocked", 1),
            ("blocked", 2, "blocked", 2),
            ("low", 1, "blocked", 1),
            ("direct", 0, "direct", 0),
            ("disk", 1, "disk", 1),
            ("disk", 2, "disk", 2),
        ],
    )
    def test_bounded_strategies_preserve_both_triples_pieces(
        self, h2o, fast, mode, tile, expected_mode, expected_tile
    ):
        """Factor-direct tiles and scalar direct reproduce E[T] and E_ST."""
        result = _cpp_ccsd_with_triples_plan(*h2o, mode=mode, tile=tile)

        assert result.converged
        assert result.triples_memory_mode_used == expected_mode
        assert result.triples_tile_size_used == expected_tile
        # ``triples_max_threads`` is an upper bound; an earlier runner test
        # may have reduced the process-wide OpenMP limit.
        assert 1 <= result.triples_threads_used <= 2
        assert 0 < result.triples_workspace_bytes <= 64 * 1024**2
        if expected_mode == "disk":
            assert result.triples_disk_bytes > 0
        else:
            assert result.triples_disk_bytes == 0
        assert result.e_t4 == pytest.approx(fast.e_t4, abs=2e-11)
        assert result.e_t5_st == pytest.approx(fast.e_t5_st, abs=2e-11)
        assert result.e_t == pytest.approx(fast.e_t, abs=2e-11)
        assert result.e_t == pytest.approx(
            result.e_t4 + result.e_t5_st, abs=1e-15
        )

    def test_blocked_workspace_is_aggregate_thread_budget(self, h2o):
        result = _cpp_ccsd_with_triples_plan(
            *h2o, mode="blocked", tile=1, threads=2
        )
        n_occ = h2o[0].n_electrons() // 2
        n_vir = h2o[1].nbasis - n_occ
        work_entries = n_occ * (n_occ + 1) * (n_occ + 2) // 6
        n_aux = BasisSet(h2o[0], AUX).nbasis
        retained_doubles = (
            n_occ * n_vir  # T1
            + n_occ * n_occ * n_vir * n_vir  # T2
            + n_occ * n_vir  # f_ov
            + n_occ + n_vir  # orbital energies
            + (n_occ * n_vir) ** 2  # (ia|jb)
            + n_occ**3 * n_vir  # (mi|ne)
            + n_aux * (n_occ * n_vir + n_vir * n_vir)  # B_ov + B_vv
        )
        expected = (
            retained_doubles * 8
            + work_entries * 3 * np.dtype(np.intp).itemsize
            + result.triples_threads_used * 2 * 1 * n_vir * n_vir * 8
        )
        assert result.triples_workspace_bytes == expected

    def test_too_small_budget_fails_instead_of_overcommitting(self, h2o):
        with pytest.raises(RuntimeError, match="cannot hold"):
            _cpp_ccsd_with_triples_plan(
                *h2o, mode="direct", budget=1, threads=1
            )

    def test_invalid_mode_fails_closed(self, h2o, tmp_path):
        with pytest.raises((ValueError, RuntimeError), match="unknown"):
            _cpp_ccsd_with_triples_plan(
                *h2o,
                mode="unknown",
                scratch_directory=str(tmp_path),
            )
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("tile", [1, 2])
    def test_disk_factor_spill_is_cleaned_up(self, h2o, fast, tmp_path, tile):
        result = _cpp_ccsd_with_triples_plan(
            *h2o,
            mode="disk",
            tile=tile,
            scratch_directory=str(tmp_path),
        )
        assert result.triples_memory_mode_used == "disk"
        assert result.triples_disk_bytes > 0
        assert result.e_t == pytest.approx(fast.e_t, abs=2e-11)
        assert list(tmp_path.iterdir()) == []

    def test_disk_read_failure_propagates_and_cleans_spill(
        self, h2o, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("VIBEQC_TEST_TRIPLES_DISK_READ_FAILURE", "1")

        with pytest.raises(RuntimeError, match="injected triples scratch read"):
            _cpp_ccsd_with_triples_plan(
                *h2o,
                mode="disk",
                tile=1,
                threads=2,
                scratch_directory=str(tmp_path),
            )

        assert list(tmp_path.iterdir()) == []
