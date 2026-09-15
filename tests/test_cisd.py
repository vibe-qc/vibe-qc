"""Tests for the general fixed-space CISD solver (vibeqc.solvers.cisd).

Reference-agnostic CI mechanics, validated against vibe-qc's own FCI (casci
full space) on libint Hamiltonians — no external/semiempirical dependence.
The MSINDO-specific CISD wiring is covered in ``tests/test_msindo.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import BasisSet, RHFOptions, run_rhf
from vibeqc.solvers import (
    build_hamiltonian_mo,
    casci,
    cisd,
    generate_cisd_determinants,
    get_hf_orbital_provider,
)


# --------------------------------------------------------------------------- #
# Determinant-space generator (pure, no integrals).                           #
# --------------------------------------------------------------------------- #

def test_generate_cisd_counts_closed_shell():
    """CISD / CIS determinant counts for a (norb, nα, nβ) closed shell."""
    # (6, 4, 4): nocc=4, nvir=2.  singles = 2·(4·2)=16; same-spin doubles =
    # 2·C(4,2)·C(2,2)=12; opposite-spin doubles = (4·2)·(4·2)=64 → +1 ref = 93.
    assert len(generate_cisd_determinants(6, 4, 4)) == 93
    # CIS = reference + singles only.
    assert len(generate_cisd_determinants(6, 4, 4, max_excitation=1)) == 17
    # Two electrons in two orbitals: CISD spans the whole FCI space (4 dets).
    assert len(generate_cisd_determinants(2, 1, 1)) == 4


def test_cis_space_is_subset_of_cisd_space():
    """``max_excitation=1`` (CIS) is structurally contained in CISD."""
    cis = set(generate_cisd_determinants(6, 4, 4, max_excitation=1))
    cisd_space = set(generate_cisd_determinants(6, 4, 4, max_excitation=2))
    assert cis < cisd_space


def test_opposite_spin_double_into_same_spatial_orbital_present():
    """The αβ double promoting both electrons into the SAME spatial virtual
    (the leading HOMO²→LUMO² correlating determinant) must be in the space.

    Regression guard: a generator that skips ``a == b`` αβ doubles silently
    drops the most important double and makes CISD ≠ FCI even for 2 electrons.
    """
    space = set(generate_cisd_determinants(2, 1, 1))
    assert ((1,), (1,)) in space            # 0α→1, 0β→1  (both into orbital 1)


def test_generate_cisd_rejects_bad_excitation():
    with pytest.raises(ValueError):
        generate_cisd_determinants(4, 2, 2, max_excitation=3)


# --------------------------------------------------------------------------- #
# CISD vs FCI on libint Hamiltonians.                                         #
# --------------------------------------------------------------------------- #

def _mo_hamiltonian(mol):
    basis = BasisSet(mol, "sto-3g")
    C = get_hf_orbital_provider(mol, basis)
    return build_hamiltonian_mo(mol, basis, C)


def test_cisd_equals_fci_for_two_electrons():
    """For a 2-electron system no triples/quadruples exist, so CISD == FCI."""
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])     # H2, bohr
    ham = _mo_hamiltonian(mol)
    cs = cisd(ham.h1e, ham.h2e, ham.nelec, ham.norb,
              nuclear_repulsion=ham.nuclear_repulsion)
    fci = casci(ham.h1e, ham.h2e, n_active_elec=ham.nelec,
                n_active_orb=ham.norb, nuclear_repulsion=ham.nuclear_repulsion)
    assert cs.e_total == pytest.approx(fci.e_total, abs=1e-10)
    assert cs.n_det == fci.n_det                                    # full space


def test_cisd_reference_energy_is_hf():
    """The CISD reference-determinant energy equals the RHF total energy
    (canonical HF MOs ⇒ Brillouin ⇒ reference decouples from singles)."""
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "sto-3g")
    opts = RHFOptions(); opts.conv_tol_energy = 1e-12
    e_hf = run_rhf(mol, basis, opts).energy
    ham = _mo_hamiltonian(mol)
    cs = cisd(ham.h1e, ham.h2e, ham.nelec, ham.norb,
              nuclear_repulsion=ham.nuclear_repulsion)
    assert cs.e_ref == pytest.approx(e_hf, abs=1e-8)


def test_cisd_brackets_fci_and_beats_cis():
    """E_FCI ≤ E_CISD ≤ E_HF, and CISD recovers more correlation than CIS."""
    mol = Molecule([Atom(3, [0, 0, 0]), Atom(1, [0, 0, 3.015])])   # LiH, bohr
    ham = _mo_hamiltonian(mol)
    cs = cisd(ham.h1e, ham.h2e, ham.nelec, ham.norb,
              nuclear_repulsion=ham.nuclear_repulsion)
    fci = casci(ham.h1e, ham.h2e, n_active_elec=ham.nelec,
                n_active_orb=ham.norb, nuclear_repulsion=ham.nuclear_repulsion)
    cis = cisd(ham.h1e, ham.h2e, ham.nelec, ham.norb,
               nuclear_repulsion=ham.nuclear_repulsion, max_excitation=1)
    assert fci.e_total <= cs.e_total + 1e-10           # FCI is the lower bound
    assert cs.e_total <= cs.e_ref + 1e-10              # CISD ≤ HF
    assert cs.e_total <= cis.e_total + 1e-10           # CISD ≤ CIS (more space)
    assert cs.e_corr < -1e-4                           # real correlation captured
    assert 0.9 < cs.reference_weight <= 1.0            # single-reference system


def test_cisd_dominant_configurations_and_corr():
    """The ground state is reference-dominated; ``dominant_configurations`` is
    sorted by weight with the reference first."""
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    ham = _mo_hamiltonian(mol)
    cs = cisd(ham.h1e, ham.h2e, ham.nelec, ham.norb,
              nuclear_repulsion=ham.nuclear_repulsion)
    configs = cs.dominant_configurations(3)
    assert configs[0][0] == "reference"
    weights = [w for _d, _c, w in configs]
    assert weights == sorted(weights, reverse=True)
    assert cs.e_corr <= 0.0                            # variational lowering
