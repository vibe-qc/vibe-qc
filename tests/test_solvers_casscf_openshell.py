"""Open-shell and edge-case CASSCF tests.

Covers UHF-based CASSCF start, open-shell systems, and convergence
edge cases.
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

H2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
H2O = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.93]),
        Atom(1, [0.0, -1.43, -0.93]),
    ]
)
# Li atom (open-shell doublet, 3 electrons)
LI_ATOM = Molecule([Atom(3, [0.0, 0.0, 0.0])], multiplicity=2)
TOL = 1e-9


def _ham(mol, basis_name, uhf=False):
    b = BasisSet(mol, basis_name)
    method = "uhf" if uhf else "rhf"
    c = get_hf_orbital_provider(mol, b, method=method)
    return build_hamiltonian_mo(mol, b, c)


class TestOpenShellCASSCF:
    """Open-shell CASSCF (ms2 != 0 via UHF start)."""

    def test_li_atom_doublet(self):
        """Li atom (3e, 3o active) — CAS(3,3) with ms2=1 (doublet)."""
        H = _ham(LI_ATOM, "sto-3g", uhf=True)
        # CAS(3,3) — full active space, ms2=1 (doublet)
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=3,
            n_active_orb=3,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
            ms2=1,
        )
        assert res.converged and res.grad_norm < 1e-6
        # Must be variational: at or above FCI (same space)
        fci = casci(
            H.h1e,
            H.h2e,
            3,
            3,
            0,
            nuclear_repulsion=H.nuclear_repulsion,
            ms2=1,
        ).e_total
        assert res.e_total >= fci - 1e-9
        # Should match CASCI (full active space -> no orbital rotations)
        assert abs(res.e_total - fci) < TOL

    def test_h2_triplet(self):
        """H2 triplet (2e, 2o active) — ms2=2 (triplet)."""
        H = _ham(H2, "sto-3g", uhf=True)
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
            ms2=2,
        )
        assert res.converged and res.grad_norm < 1e-6

    def test_open_shell_vs_pyscf(self):
        """Open-shell CASSCF vs PySCF."""
        pytest.importorskip("pyscf")
        from pyscf import gto, mcscf, scf

        # Li atom, CAS(1,3) — one active electron in 3 orbitals
        H = _ham(LI_ATOM, "sto-3g", uhf=True)
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=1,
            n_active_orb=3,
            n_core=1,
            nuclear_repulsion=H.nuclear_repulsion,
            ms2=1,
        )
        assert res.converged and res.grad_norm < 1e-6

        mf = scf.UHF(
            gto.M(atom="Li 0 0 0", unit="Bohr", basis="sto-3g", verbose=0, spin=1)
        ).run(verbose=0)
        mc = mcscf.CASSCF(mf, 3, 1)
        mc.run(verbose=0)
        assert abs(res.e_total - mc.e_tot) < 1e-6


class TestCASSCFEdgeCases:
    """Edge-case and robustness CASSCF tests."""

    def test_h2o_cas44_superci_energy_descends(self):
        """H2O CAS(4,4) with Super-CI only: energy descends monotonically.

        Super-CI converges linearly; it may not reach 1e-6 gradient but the
        energy is guaranteed monotone.  'auto' mode (tested below) switches
        to NR for tight convergence.
        """
        H = _ham(H2O, "sto-3g")
        res = casscf(
            H.h1e, H.h2e, n_active_elec=4, n_active_orb=4, n_core=3,
            nuclear_repulsion=H.nuclear_repulsion,
            orbital_step="superci", max_macro=50,
        )
        # Energy must descend monotonically (the backtracking guarantees it)
        for i in range(1, len(res.energy_trace)):
            assert res.energy_trace[i] <= res.energy_trace[i - 1] + 1e-12

    def test_h2o_cas44_auto_converges(self):
        """H2O CAS(4,4) with auto (Super-CI then NR)."""
        H = _ham(H2O, "sto-3g")
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=4,
            n_active_orb=4,
            n_core=3,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        assert res.converged and res.grad_norm < 1e-6

    def test_h2o_cas44_nr_converges(self):
        """H2O CAS(4,4) with NR only."""
        H = _ham(H2O, "sto-3g")
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=4,
            n_active_orb=4,
            n_core=3,
            nuclear_repulsion=H.nuclear_repulsion,
            orbital_step="nr",
            max_macro=100,
        )
        assert res.converged and res.grad_norm < 1e-6

    def test_energy_monotone(self):
        """CASSCF energy is monotone non-increasing."""
        H = _ham(H2O, "sto-3g")
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=4,
            n_active_orb=4,
            n_core=3,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        for i in range(1, len(res.energy_trace)):
            assert res.energy_trace[i] <= res.energy_trace[i - 1] + 1e-12

    def test_invalid_orbital_step_raises(self):
        """Invalid orbital_step raises ValueError."""
        H = _ham(H2, "sto-3g")
        with pytest.raises(ValueError, match="orbital_step"):
            casscf(
                H.h1e,
                H.h2e,
                2,
                2,
                0,
                nuclear_repulsion=H.nuclear_repulsion,
                orbital_step="invalid",
            )

    def test_superci_same_energy_as_nr(self):
        """Super-CI and NR should converge to the same stationary point."""
        H = _ham(H2, "6-31g")
        res_nr = casscf(
            H.h1e,
            H.h2e,
            2,
            2,
            0,
            nuclear_repulsion=H.nuclear_repulsion,
            orbital_step="nr",
        )
        res_sci = casscf(
            H.h1e,
            H.h2e,
            2,
            2,
            0,
            nuclear_repulsion=H.nuclear_repulsion,
            orbital_step="superci",
        )
        assert abs(res_nr.e_total - res_sci.e_total) < 1e-7
