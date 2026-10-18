"""UNO-CAS open-shell starting reference (Pulay & Hamilton 1988).

``get_hf_orbital_provider(method="uno")`` diagonalizes the total UHF
density in the orthogonalized AO basis; the descending-occupation
orbital set is the open-shell default reference of the determinant-
solver family (``run_job(cas_reference=...)`` overrides).

PySCF is the natural-orbital oracle via importorskip (CLAUDE.md §10).
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.runner import run_job
from vibeqc.solvers import build_hamiltonian_mo, casci, get_hf_orbital_provider
from vibeqc.solvers._casscf import casscf

O2 = Molecule([Atom(8, [0.0, 0.0, 0.0]), Atom(8, [0.0, 0.0, 2.28])], multiplicity=3)
O2_ATOM = "O 0 0 0; O 0 0 2.28"


def _assert_same_uhf_reference(mf, result, occupations):
    from pyscf import mcscf

    assert abs(mf.e_tot - result.energy) < 1e-8, (
        "different UHF stationary point: "
        f"PySCF={mf.e_tot:.14f}, reference={result.energy:.14f} Ha"
    )
    assert abs(mf.spin_square()[0] - result.s_squared) < 1e-6
    noons, _ = mcscf.addons.make_natural_orbitals(mf)
    assert np.max(np.abs(np.sort(occupations) - np.sort(noons))) < 1e-8


def _stable_pyscf_reference(mf, max_restarts=5):
    """Follow internal instabilities with bounded MOM-tracked UHF solves."""
    from pyscf import scf

    for restart in range(max_restarts + 1):
        assert mf.converged, (
            f"PySCF UHF root {restart} did not converge: E={mf.e_tot:.14f} Ha"
        )
        mo, _, stable, _ = mf.stability(
            internal=True, external=False, return_status=True, verbose=0
        )
        if stable:
            return mf
        if restart == max_restarts:
            break
        # Track the occupied subspace of the stability-rotated determinant.
        # Ordinary Aufbau reassigns occupations after the rotation and can
        # return to the unstable first root (or oscillate at one thread).
        dm0 = mf.make_rdm1(mo, mf.mo_occ)
        mf = scf.addons.mom_occ(scf.UHF(mf.mol), mo, mf.mo_occ)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 200
        mf.kernel(dm0=dm0)
    pytest.fail(
        f"PySCF UHF root remains internally unstable after {max_restarts} "
        f"restarts: E={mf.e_tot:.14f} Ha"
    )


@pytest.fixture(scope="module")
def o2_basis():
    return BasisSet(O2, "sto-3g")


class TestUNOProvider:
    def test_orthonormal_and_occupations_vs_pyscf(self, o2_basis):
        pytest.importorskip("pyscf")
        from pyscf import gto, scf
        from vibeqc._vibeqc_core import UHFOptions, compute_overlap, run_uhf

        C = get_hf_orbital_provider(O2, o2_basis, method="uno")
        S = np.asarray(compute_overlap(o2_basis))
        assert np.max(np.abs(C.T @ S @ C - np.eye(C.shape[1]))) < 1e-12

        opts = UHFOptions()
        opts.max_iter = 200
        opts.conv_tol_energy = 1e-12
        opts.conv_tol_grad = 1e-10
        r = run_uhf(O2, o2_basis, opts)
        P = np.asarray(r.density_alpha) + np.asarray(r.density_beta)
        occ = np.diag(C.T @ S @ P @ S @ C)
        assert abs(occ.sum() - 16.0) < 1e-9  # total electron count
        assert np.all(np.diff(occ) < 1e-9)  # descending order

        pyscf_mol = gto.M(atom=O2_ATOM, unit="Bohr", basis="sto-3g", spin=2, verbose=0)
        mf = scf.UHF(pyscf_mol).run(
            conv_tol=1e-12, conv_tol_grad=1e-10, max_cycle=200, verbose=0
        )
        # Compare the determinants only after following internal instabilities.
        mf = _stable_pyscf_reference(mf)
        _assert_same_uhf_reference(mf, r, occ)

    def test_unstable_first_root_is_rejected_as_a_different_determinant(self, o2_basis):
        pytest.importorskip("pyscf")
        from pyscf import gto, scf
        from vibeqc._vibeqc_core import UHFOptions, compute_overlap, run_uhf

        mf = scf.UHF(gto.M(
            atom=O2_ATOM, unit="Bohr", basis="sto-3g", spin=2, verbose=0
        )).run(conv_tol=1e-12, conv_tol_grad=1e-10, max_cycle=200, verbose=0)
        _, _, stable, _ = mf.stability(
            internal=True, external=False, return_status=True, verbose=0
        )
        assert mf.converged and not stable, "control must use the unstable first root"
        opts = UHFOptions()
        opts.max_iter = 200
        opts.conv_tol_energy = 1e-12
        opts.conv_tol_grad = 1e-10
        result = run_uhf(O2, o2_basis, opts)
        C = get_hf_orbital_provider(O2, o2_basis, method="uno")
        S = np.asarray(compute_overlap(o2_basis))
        P = np.asarray(result.density_alpha) + np.asarray(result.density_beta)
        occupations = np.diag(C.T @ S @ P @ S @ C)
        with pytest.raises(AssertionError, match="different UHF stationary point"):
            _assert_same_uhf_reference(mf, result, occupations)

    def test_uno_reference_improves_o2_casci_and_casscf_start(self, o2_basis):
        # The π* singly-occupied pair sits exactly in the CAS(2,2) window
        # with UNO ordering; the UHF-α reference misrepresents the β space.
        results = {}
        for ref in ("uhf", "uno"):
            C = get_hf_orbital_provider(O2, o2_basis, method=ref)
            H = build_hamiltonian_mo(O2, o2_basis, C)
            ci = casci(
                H.h1e, H.h2e, 2, 2, 7,
                nuclear_repulsion=H.nuclear_repulsion, ms2=2,
            )
            sc = casscf(
                H.h1e, H.h2e, 2, 2, 7,
                nuclear_repulsion=H.nuclear_repulsion, ms2=2,
            )
            results[ref] = (ci.e_total, sc.e_total, sc.n_iter, sc.converged)
        # Recorded 2026-06-10: UNO-CASCI -147.63211277 vs UHF-α-CASCI
        # -147.63001316 (2.1 mHa); CASSCF basin identical (-147.63211653),
        # 4 vs 27 macro-iterations.
        assert results["uno"][0] < results["uhf"][0] - 1e-3
        assert results["uno"][3] and results["uhf"][3]
        assert abs(results["uno"][1] - results["uhf"][1]) < 1e-7
        assert results["uno"][2] < results["uhf"][2]


class TestRunJobCASReference:
    def _run(self, tmp_path, name, **kw):
        return run_job(
            O2,
            basis="sto-3g",
            method="casci",
            active_space=(2, 2),
            output=tmp_path / name,
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
            **kw,
        )

    def test_open_shell_default_is_uno(self, tmp_path):
        e_default = self._run(tmp_path, "default").energy
        e_uno = self._run(tmp_path, "uno", cas_reference="uno").energy
        e_uhf = self._run(tmp_path, "uhf", cas_reference="uhf").energy
        assert abs(e_default - e_uno) < 1e-10
        assert e_uno < e_uhf - 1e-3  # the M7 improvement

    def test_invalid_reference_raises(self, tmp_path):
        with pytest.raises(ValueError, match="cas_reference"):
            self._run(tmp_path, "bad", cas_reference="bogus")

    def test_uno_citation_reaches_bibtex(self, tmp_path):
        run_job(
            O2,
            basis="sto-3g",
            method="casci",
            active_space=(2, 2),
            output=tmp_path / "cite",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=True,
        )
        bib = (tmp_path / "cite.bibtex").read_text()
        assert "pulay_hamilton_uno_1988" in bib
