"""Multi-root CASCI and state-averaged CASSCF tests.

Validated against PySCF mcscf.CASCI (multi-root) and mcscf.CASSCF (SA).
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    build_hamiltonian_mo,
    casci,
    casscf,
    get_hf_orbital_provider,
)

H2O = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.93]),
        Atom(1, [0.0, -1.43, -0.93]),
    ]
)
H2O_ATOM = "O 0 0 0; H 0 1.43 -0.93; H 0 -1.43 -0.93"
H2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])

TOL = 1e-9


def _ham(mol, basis_name):
    b = BasisSet(mol, basis_name)
    c = get_hf_orbital_provider(mol, b, method="rhf")
    return build_hamiltonian_mo(mol, b, c)


class TestMultiRootCASCI:
    """Multi-root CASCI (nroots > 1)."""

    def test_multi_root_energies_vs_pyscf(self):
        pytest.importorskip("pyscf")
        from pyscf import gto, mcscf, scf

        H = _ham(H2O, "sto-3g")
        res = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=4,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=3,
        )
        assert res.nroots == 3
        assert len(res.e_totals) == 3
        assert res.e_totals[0] == res.e_total
        assert res.e_totals[0] <= res.e_totals[1] <= res.e_totals[2]

        mf = scf.RHF(gto.M(atom=H2O_ATOM, unit="Bohr", basis="sto-3g", verbose=0)).run(
            verbose=0
        )
        mc = mcscf.CASCI(mf, 2, 2)
        mc.fcisolver.nroots = 3
        mc.run(verbose=0)
        for k in range(3):
            assert abs(res.e_totals[k] - mc.e_tot[k]) < TOL

    def test_multi_root_ci_coeffs_shape(self):
        H = _ham(H2O, "sto-3g")
        res = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=4,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=2,
        )
        n_det = len(res.determinants)
        assert res.ci_coeffs_all.shape == (n_det, 2)
        assert np.allclose(res.ci_coeffs, res.ci_coeffs_all[:, 0])

    def test_multi_root_backward_compat(self):
        H = _ham(H2, "6-31g")
        res = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        assert res.nroots == 1
        assert res.e_totals == [res.e_total]
        assert res.ci_coeffs_all is None
        assert res.ci_coeffs.ndim == 1


class TestSACASSCF:
    """State-averaged CASSCF (nroots > 1)."""

    def test_sa_basic_energy(self):
        H = _ham(H2O, "sto-3g")
        n_core, n_act, n_elec = 4, 2, 2
        fci = casci(
            H.h1e,
            H.h2e,
            H.nelec,
            H.norb,
            0,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total

        res = casscf(
            H.h1e,
            H.h2e,
            n_elec,
            n_act,
            n_core,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=2,
        )
        assert res.converged and res.grad_norm < 1e-6
        assert res.e_total >= fci - 1e-9
        # SA energy matches weighted average of converged per-root energies
        assert abs(res.e_total - np.mean(res.e_totals)) < 1e-9

    def test_sa_vs_pyscf(self):
        pytest.importorskip("pyscf")
        from pyscf import gto, mcscf, scf

        H = _ham(H2O, "sto-3g")
        n_core, n_act, n_elec = 4, 2, 2
        w = [0.6, 0.4]

        # spin_pure=False: PySCF's default FCI solver (direct_spin1)
        # state-averages the raw M_s = 0 sector roots, i.e. the
        # mixed-spin ensemble, the deliberate opt-out from the
        # spin-pure SA default (2026-06-11 flip).  This pins that the
        # opt-out reproduces PySCF's averaging exactly.
        res = casscf(
            H.h1e,
            H.h2e,
            n_elec,
            n_act,
            n_core,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=2,
            weights=w,
            spin_pure=False,
        )
        assert res.converged and res.grad_norm < 1e-6

        mf = scf.RHF(gto.M(atom=H2O_ATOM, unit="Bohr", basis="sto-3g", verbose=0)).run(
            verbose=0
        )
        mc = mcscf.CASSCF(mf, 2, 2)
        mc.fcisolver.nroots = 2
        mc.state_average_(w, 2)
        mc.run(verbose=0)
        assert abs(res.e_total - mc.e_tot) < 1e-6

    def test_sa_weight_validation(self):
        H = _ham(H2O, "sto-3g")
        with pytest.raises(ValueError, match="must sum to 1"):
            casscf(
                H.h1e,
                H.h2e,
                2,
                2,
                4,
                nuclear_repulsion=H.nuclear_repulsion,
                nroots=2,
                weights=[0.5, 0.3],
            )
        with pytest.raises(ValueError, match="len"):
            casscf(
                H.h1e,
                H.h2e,
                2,
                2,
                4,
                nuclear_repulsion=H.nuclear_repulsion,
                nroots=3,
                weights=[0.5, 0.5],
            )

    def test_sa_e_totals_populated(self):
        H = _ham(H2O, "sto-3g")
        res = casscf(
            H.h1e,
            H.h2e,
            2,
            2,
            4,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=2,
        )
        assert len(res.e_totals) == 2
        assert res.e_totals[0] <= res.e_totals[1]

    def test_sa_single_state_equivalent(self):
        H = _ham(H2, "6-31g")
        res_ss = casscf(
            H.h1e,
            H.h2e,
            2,
            2,
            0,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=1,
        )
        res_sa = casscf(
            H.h1e,
            H.h2e,
            2,
            2,
            0,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=1,
            weights=[1.0],
        )
        assert abs(res_ss.e_total - res_sa.e_total) < 1e-12
        assert res_ss.e_totals == res_sa.e_totals
