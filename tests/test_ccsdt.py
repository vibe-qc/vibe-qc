"""Full iterative CCSDT solver anchors."""

from __future__ import annotations

import json
from zipfile import ZipFile

import pytest

from vibeqc import Atom, BasisSet, Molecule, run_job
from vibeqc.output.formats.qvf import validate_qvf
from vibeqc.solvers import (
    CCSDTOptions,
    CCSDTResult,
    build_hamiltonian_mo,
    casci,
    ccsdt,
    get_hf_orbital_provider,
)


def _hamiltonian(molecule: Molecule, basis_name: str):
    basis = BasisSet(molecule, basis_name)
    orbitals = get_hf_orbital_provider(molecule, basis, method="rhf")
    return build_hamiltonian_mo(molecule, basis, orbitals)


def test_ccsdt_is_exact_for_two_electrons() -> None:
    molecule = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
    )
    hamiltonian = _hamiltonian(molecule, "sto-3g")
    result = ccsdt(
        hamiltonian,
        CCSDTOptions(
            max_iter=40,
            conv_tol_energy=1.0e-12,
            conv_tol_residual=1.0e-10,
        ),
    )
    fci = casci(
        hamiltonian.h1e,
        hamiltonian.h2e,
        n_active_elec=hamiltonian.nelec,
        n_active_orb=hamiltonian.norb,
        n_core=0,
        nuclear_repulsion=hamiltonian.nuclear_repulsion,
        ms2=0,
    )

    assert result.converged
    assert result.energy == pytest.approx(fci.e_total, abs=2.0e-12)
    assert result.t3_norm == 0.0
    assert result.residual_rms < 1.0e-10


def test_ccsdt_nonzero_triples_matches_orca_h4_anchor() -> None:
    molecule = Molecule(
        [
            Atom(1, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.6]),
            Atom(1, [0.0, 0.0, 3.2]),
            Atom(1, [0.0, 0.0, 4.8]),
        ]
    )
    result = ccsdt(
        _hamiltonian(molecule, "sto-3g"),
        CCSDTOptions(
            max_iter=80,
            conv_tol_energy=1.0e-11,
            conv_tol_residual=1.0e-9,
        ),
    )

    # ORCA 6.1.1, RHF AUTOCI-CCSDT/STO-3G: -2.177916325478 Ha.
    assert result.converged
    assert result.energy == pytest.approx(-2.177916325478, abs=2.0e-8)
    assert result.t3_norm > 1.0e-4
    assert result.n_triples > 0
    assert result.residual_rms < 1.0e-9


def test_public_ccsdt_is_exact_for_three_electron_open_shell(tmp_path) -> None:
    molecule = Molecule([Atom(3, [0.0, 0.0, 0.0])], multiplicity=2)
    basis = BasisSet(molecule, "sto-3g")
    orbitals = get_hf_orbital_provider(molecule, basis, method="rohf")
    hamiltonian = build_hamiltonian_mo(molecule, basis, orbitals)
    result = run_job(
        molecule,
        basis="sto-3g",
        method="ccsdt",
        ccsdt_options=CCSDTOptions(
            n_frozen_core=0,
            max_iter=40,
            conv_tol_energy=1.0e-11,
            conv_tol_residual=1.0e-9,
        ),
        output=tmp_path / "li-ccsdt",
        progress=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )
    fci = casci(
        hamiltonian.h1e,
        hamiltonian.h2e,
        n_active_elec=hamiltonian.nelec,
        n_active_orb=hamiltonian.norb,
        n_core=0,
        nuclear_repulsion=hamiltonian.nuclear_repulsion,
        ms2=hamiltonian.ms2,
    )

    assert result.converged
    assert result.energy == pytest.approx(fci.e_total, abs=2.0e-11)
    assert result.n_triples > 0


def test_run_job_ccsdt_writes_valid_qvf_and_manifest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBEQC_OUTPUT_LEVEL", "standard")
    molecule = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
    )
    stem = tmp_path / "h2-ccsdt"
    result = run_job(
        molecule,
        basis="sto-3g",
        method="ccsdt",
        ccsdt_options=CCSDTOptions(n_frozen_core=0),
        output=stem,
        progress=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )

    assert isinstance(result, CCSDTResult)
    assert result.converged
    assert validate_qvf(stem.with_suffix(".qvf"))["valid"] is True
    with ZipFile(stem.with_suffix(".qvf")) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert any(
        section["kind"] == "scf_history" for section in manifest["sections"]
    )
    assert 'method           = "CCSDT"' in stem.with_suffix(".system").read_text()
    out_text = stem.with_suffix(".out").read_text()
    assert "CCSDT (full iterative T1/T2/T3)" in out_text
    assert "RMS residual  Max residual" not in out_text
    assert "Noga" in out_text
