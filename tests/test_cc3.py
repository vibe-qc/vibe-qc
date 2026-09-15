"""Ground-state CC3 solver anchors."""

from __future__ import annotations

import json
from zipfile import ZipFile

import pytest

from vibeqc import Atom, BasisSet, CC3Result, Molecule, run_job
from vibeqc.output.formats.qvf import validate_qvf
from vibeqc.solvers import (
    CC3Options,
    build_hamiltonian_mo,
    casci,
    cc3,
    get_hf_orbital_provider,
)


def _hamiltonian(molecule: Molecule, basis_name: str):
    basis = BasisSet(molecule, basis_name)
    orbitals = get_hf_orbital_provider(molecule, basis, method="rhf")
    return build_hamiltonian_mo(molecule, basis, orbitals)


def test_cc3_is_exact_for_two_electrons() -> None:
    molecule = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
    )
    hamiltonian = _hamiltonian(molecule, "sto-3g")
    result = cc3(
        hamiltonian,
        CC3Options(
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


def test_cc3_nonzero_triples_matches_orca_h4_anchor() -> None:
    molecule = Molecule(
        [
            Atom(1, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.6]),
            Atom(1, [0.0, 0.0, 3.2]),
            Atom(1, [0.0, 0.0, 4.8]),
        ]
    )
    result = cc3(
        _hamiltonian(molecule, "sto-3g"),
        CC3Options(
            max_iter=80,
            conv_tol_energy=1.0e-11,
            conv_tol_residual=1.0e-9,
        ),
    )

    # ORCA 6.1.1, UHF AUTOCI-CC3/STO-3G: -2.177909751349 Ha.
    # ORCA exposes CC3 through its UHF spin-orbital module; the converged
    # closed-shell UHF determinant is identical to the RHF reference here.
    assert result.converged
    assert result.energy == pytest.approx(-2.177909751349, abs=2.0e-9)
    assert result.t3_norm > 1.0e-4
    assert result.n_triples > 0
    assert result.residual_rms < 1.0e-9


def test_cc3_frozen_core_reduces_to_exact_two_active_electrons() -> None:
    molecule = Molecule([Atom(4, [0.0, 0.0, 0.0])])
    hamiltonian = _hamiltonian(molecule, "sto-3g")
    result = cc3(
        hamiltonian,
        CC3Options(
            n_frozen_core=1,
            max_iter=40,
            conv_tol_energy=1.0e-12,
            conv_tol_residual=1.0e-10,
        ),
    )
    active = hamiltonian.active_space(
        hamiltonian.norb - 1, hamiltonian.nelec - 2
    )
    fci = casci(
        active.h1e,
        active.h2e,
        n_active_elec=active.nelec,
        n_active_orb=active.norb,
        n_core=0,
        nuclear_repulsion=active.nuclear_repulsion,
        ms2=0,
    )

    assert result.converged
    assert result.n_frozen_core == 1
    assert result.energy == pytest.approx(fci.e_total, abs=2.0e-11)
    assert result.t3_norm == 0.0


def test_cc3_rejects_open_shell_reference() -> None:
    molecule = Molecule([Atom(3, [0.0, 0.0, 0.0])], multiplicity=2)
    basis = BasisSet(molecule, "sto-3g")
    orbitals = get_hf_orbital_provider(molecule, basis, method="rohf")
    hamiltonian = build_hamiltonian_mo(molecule, basis, orbitals)

    with pytest.raises(ValueError, match="closed-shell"):
        cc3(hamiltonian)


def test_run_job_cc3_writes_valid_qvf_and_manifest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBEQC_OUTPUT_LEVEL", "standard")
    molecule = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
    )
    stem = tmp_path / "h2-cc3"
    result = run_job(
        molecule,
        basis="sto-3g",
        method="cc3",
        cc3_options=CC3Options(n_frozen_core=0),
        output=stem,
        progress=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )

    assert isinstance(result, CC3Result)
    assert result.converged
    assert validate_qvf(stem.with_suffix(".qvf"))["valid"] is True
    with ZipFile(stem.with_suffix(".qvf")) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert any(
        section["kind"] == "scf_history" for section in manifest["sections"]
    )
    assert 'method           = "CC3"' in stem.with_suffix(".system").read_text()
    out_text = stem.with_suffix(".out").read_text()
    assert "CC3 (iterative T1/T2, approximate T3)" in out_text
    assert "RMS residual  Max residual" not in out_text
    assert "Koch" in out_text


def test_run_job_cc3_rejects_open_shell_before_integral_build(tmp_path) -> None:
    molecule = Molecule([Atom(3, [0.0, 0.0, 0.0])], multiplicity=2)
    with pytest.raises(ValueError, match="closed-shell singlet RHF"):
        run_job(
            molecule,
            basis="sto-3g",
            method="cc3",
            output=tmp_path / "li-cc3",
            progress=False,
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
        )
