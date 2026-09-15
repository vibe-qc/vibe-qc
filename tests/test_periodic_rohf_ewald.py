"""Periodic Γ-point ROHF (EWALD_3D) tests.

Runs on a build box (needs the compiled periodic J/K). The Roothaan
coupling, DIIS and occupation are shared with the molecular ROHF driver
(unit-tested in ``tests/test_rohf.py``); these tests pin the *periodic*
integration:

* closed-shell limit equals periodic RHF (the Roothaan effective Fock
  collapses to the closed-shell Fock; same Ewald J / real-space K / e_nuc),
* exact ⟨S²⟩ = S(S+1) for an open shell,
* a one-electron doublet equals periodic UHF (no closed shell → identical),
* ROHF lies at or above UHF (variational; UHF has more freedom) and is
  spin-pure where UHF may be contaminated.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _options():
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.damping = 0.3
    o.max_iter = 60
    return o


def _h2_closed_shell(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _atom_doublet(Z: int, box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(Z, [c, c, c])])
    sysp.multiplicity = 2
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def test_periodic_rohf_closed_shell_matches_rhf():
    """Closed-shell H2 in a box: periodic ROHF == periodic RHF, ⟨S²⟩ = 0."""
    sysp, basis = _h2_closed_shell()
    opts = _options()
    r_rhf = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    r_rohf = vq.run_rohf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    assert r_rhf.converged and r_rohf.converged
    assert r_rohf.energy == pytest.approx(r_rhf.energy, abs=1e-8)
    assert abs(r_rohf.s_squared) < 1e-12


def test_periodic_rohf_h_atom_equals_uhf():
    """One-electron doublet: no closed shell, so ROHF == UHF exactly, and
    ⟨S²⟩ = 0.75."""
    sysp, basis = _atom_doublet(1)
    opts = _options()
    r_rohf = vq.run_rohf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    r_uhf = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    assert r_rohf.converged and r_uhf.converged
    assert r_rohf.s_squared == pytest.approx(0.75, abs=1e-12)
    assert r_rohf.energy == pytest.approx(r_uhf.energy, abs=1e-7)
    # ROHF occupations: one singly-occupied orbital, rest empty.
    assert float(np.sum(r_rohf.mo_occupations)) == pytest.approx(1.0, abs=1e-10)


def test_periodic_rohf_li_atom_spin_pure_and_above_uhf():
    """Li atom in a box (3 e⁻, doublet: 1 closed + 1 open): ROHF is
    spin-pure (⟨S²⟩ = 0.75 exactly) and lies at or above UHF."""
    sysp, basis = _atom_doublet(3)
    opts = _options()
    r_rohf = vq.run_rohf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    assert r_rohf.converged
    assert r_rohf.s_squared == pytest.approx(0.75, abs=1e-10)
    # occupations: 1 closed (2.0) + 1 open (1.0) = 3 electrons.
    assert float(np.sum(r_rohf.mo_occupations)) == pytest.approx(3.0, abs=1e-10)

    r_uhf = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    if r_uhf.converged:
        assert r_uhf.energy <= r_rohf.energy + 1e-7


def test_periodic_rohf_invalid_multiplicity_raises():
    box = 30.0
    c = box / 2
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(1, [c, c, c])])
    sysp.multiplicity = 1  # 1 electron + singlet is impossible
    # The impossible spin state is rejected fail-fast at molecule construction
    # (Molecule validates n_electrons vs multiplicity, cpp/src/molecule.cpp),
    # before the ROHF driver ever runs. Wrap the whole build+run so the guard
    # is asserted wherever it fires.
    with pytest.raises(ValueError, match="multiplicity"):
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        vq.run_rohf_periodic_gamma_ewald3d(sysp, basis, _options())
