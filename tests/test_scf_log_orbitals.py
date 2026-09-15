"""Orbital-energy table in format_scf_trace() output."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vibeqc import (
    Atom,
    BasisSet,
    InitialGuess,
    Molecule,
    RKSOptions,
    UHFOptions,
    format_scf_trace,
    run_rhf,
    run_rks,
    run_uhf,
)
from vibeqc.output.formats.scf_log import _format_orbital_sections
from vibeqc.pbc_bipole import PBCBipoleEnergyComponents


def _h2o() -> Molecule:
    return Molecule([
        Atom(8, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])


def test_orbital_table_absent_without_molecule():
    """Legacy callers without a molecule see only the iteration trace."""
    mol = _h2o()
    basis = BasisSet(mol, "sto-3g")
    result = run_rhf(mol, basis)
    text = format_scf_trace(result, include_banner=False)
    assert "HOMO" not in text
    assert "Molecular orbitals" not in text


def test_bipole_energy_components_print_without_molecule():
    result = SimpleNamespace(
        scf_trace=[
            SimpleNamespace(
                iter=1,
                energy=-10.0,
                delta_e=0.0,
                grad_norm=1.0e-4,
                diis_subspace=0,
            )
        ],
        converged=False,
        n_iter=1,
        energy=-10.0,
        energy_components=[
            PBCBipoleEnergyComponents(
                iter=1,
                e_total=-10.0,
                e_electronic=-11.0,
                e_kinetic=2.0,
                e_nuclear_attraction=-15.0,
                e_two_electron=2.0,
                e_nuclear_repulsion=1.0,
            )
        ],
    )

    text = format_scf_trace(result, include_banner=False)

    assert "Energy components" in text
    assert "Kinetic" in text
    assert "Nuclear attraction" in text
    assert "Two-electron (J+K)" in text


def test_orbital_table_rhf_has_homo_lumo_markers():
    mol = _h2o()
    basis = BasisSet(mol, "sto-3g")
    result = run_rhf(mol, basis)

    text = format_scf_trace(result, molecule=mol, include_banner=False)

    assert "Molecular orbitals" in text
    assert "<-- HOMO" in text
    assert "<-- LUMO" in text
    # BUG 81 (43f1e433b) replaced the "HOMO-LUMO gap:" footer with the
    # explicit HOMO / LUMO / gap triple.
    assert "HOMO:" in text
    assert "LUMO:" in text
    assert "gap:" in text
    # H2O at this geometry has 5 doubly occupied orbitals.
    assert text.count(" 2.0 ") >= 5


def test_orbital_table_uses_effective_ecp_electron_count():
    """ECP-removed core orbitals cannot be labelled occupied in output."""
    result = SimpleNamespace(
        mo_energies=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.1, 0.3],
        ecp_total_ncore=2,
    )

    text = _format_orbital_sections(result, _h2o(), n_virtual=3)
    homo_line = next(line for line in text.splitlines() if "<-- HOMO" in line)
    lumo_line = next(line for line in text.splitlines() if "<-- LUMO" in line)

    assert homo_line.split()[0] == "4"
    assert lumo_line.split()[0] == "5"
    assert text.count(" 2.0 ") == 4


def test_orbital_table_rks_includes_energy_components():
    mol = _h2o()
    basis = BasisSet(mol, "sto-3g")
    opts = RKSOptions()
    opts.functional = "PBE"
    result = run_rks(mol, basis, opts)

    text = format_scf_trace(result, molecule=mol, include_banner=False)

    assert "Energy components" in text
    assert "Exchange-correlation (XC)" in text
    assert "Coulomb (J)" in text
    assert "Nuclear repulsion" in text


def test_orbital_table_uhf_has_both_spin_blocks():
    # OH radical (same setup as the molden UHF test — known to converge).
    A2B = 1.0 / 0.529177210903
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.97 * A2B, 0.0, 0.0]),
        ],
        multiplicity=2,
    )
    basis = BasisSet(mol, "6-31g*")
    opts = UHFOptions()
    opts.initial_guess = InitialGuess.SAD
    result = run_uhf(mol, basis, opts)

    text = format_scf_trace(result, molecule=mol, include_banner=False)

    assert "Molecular orbitals - Alpha" in text
    assert "Molecular orbitals - Beta" in text
    assert "<-- HOAMO" in text
    assert "<-- HOBMO" in text
    # BUG 81 (43f1e433b): the per-spin gap footers now name the frontier
    # orbitals explicitly ("HOAMO: ...; LUAMO: ...; gap: ...").
    assert "HOAMO:" in text
    assert "LUAMO:" in text
    assert "HOBMO:" in text
    assert "LUBMO:" in text


def test_n_virtual_controls_virtual_rows():
    mol = _h2o()
    basis = BasisSet(mol, "sto-3g")
    result = run_rhf(mol, basis)

    short = format_scf_trace(result, molecule=mol, include_banner=False,
                             n_virtual=1)
    long = format_scf_trace(result, molecule=mol, include_banner=False,
                            n_virtual=10)
    # Count rows with occ 0.0 in each.
    short_virt = short.count(" 0.0 ")
    long_virt = long.count(" 0.0 ")
    assert long_virt > short_virt


def test_include_banner_false_strips_banner():
    mol = _h2o()
    basis = BasisSet(mol, "sto-3g")
    result = run_rhf(mol, basis)
    text = format_scf_trace(result, molecule=mol, include_banner=False)
    assert "vibe-qc" not in text.splitlines()[0]
