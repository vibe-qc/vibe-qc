"""CASSCF natural orbitals: exposure on the result, and the QVF section.

Before this, a correlated solver's orbitals never left the solver. `run_job(
method="casscf", output_qvf=True)` wrote structure, citations and
run.record and no wavefunction at all, because the QVF wavefunction payload
is built from mean-field `mo_coeffs` that a CASSCF result does not have.

What it does have is better suited to the archive anyway: natural orbitals,
whose occupation numbers are the multireference character the calculation
was run to measure.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

vq = pytest.importorskip("vibeqc")


# H2 at 2.4 A in 6-31G with CAS(2,2): far enough out that the two-
# configuration character is unmistakable, small enough to run in a second.
BOND = 2.4
BASIS = "6-31g"
ACTIVE = (2, 2)


@pytest.fixture(scope="module")
def casscf_output_stem(tmp_path_factory):
    """Where the shared CASSCF run below writes its artifacts.

    ``output=`` is a path *stem*, so a bare relative name would put the job's
    six artifacts wherever pytest happened to be invoked — which for this
    suite is a git checkout. That is what #508 was: a full-suite sweep left
    ``h2_cas_test.{out,xyz,system,bibtex,references,scf.jsonl}`` in compute-medium's
    ``vq admin update``-managed release checkout, and the next roll refused
    to deploy over a dirty tree. ``tmp_path_factory`` rather than ``tmp_path``
    because the consuming fixture is module-scoped and ``tmp_path`` is
    function-scoped.
    """
    return tmp_path_factory.mktemp("casscf-natural-orbitals") / "h2_cas_test"


@pytest.fixture(scope="module")
def h2_casscf(casscf_output_stem):
    """One CASSCF run shared by the assertions below."""
    mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, BOND])])
    result = vq.run_job(
        mol, basis=BASIS, method="casscf", active_space=ACTIVE,
        output=str(casscf_output_stem), output_qvf=False,
    )
    return mol, result


def test_casscf_fixture_artifacts_are_pytest_managed(
    h2_casscf, casscf_output_stem
):
    assert not casscf_output_stem.is_relative_to(Path.cwd())
    artifacts = tuple(casscf_output_stem.parent.glob(f"{casscf_output_stem.name}.*"))
    assert casscf_output_stem.with_suffix(".out") in artifacts
    assert all(artifact.is_file() for artifact in artifacts)
    assert not tuple(Path.cwd().glob(f"{casscf_output_stem.name}.*"))


def test_casscf_result_exposes_natural_orbitals(h2_casscf):
    mol, result = h2_casscf
    assert result.natural_orbitals is not None, "CASSCF must surface its orbitals"
    assert result.natural_occupations is not None

    C = np.asarray(result.natural_orbitals)
    occ = np.asarray(result.natural_occupations)
    assert C.ndim == 2
    assert C.shape[1] == occ.size, "one occupation per orbital"


def test_occupations_are_an_electron_count(h2_casscf):
    """They must sum to the electrons in the system.

    This is not bookkeeping: it is the property a consumer uses to tell an
    occupancy vector from a set of transition weights, and the QVF spec
    defines `occupations` as an electron count per MO.
    """
    mol, result = h2_casscf
    occ = np.asarray(result.natural_occupations)
    assert occ.sum() == pytest.approx(mol.n_electrons(), abs=1e-8)


def test_occupations_are_fractional_and_ordered(h2_casscf):
    """A stretched bond gives genuine fractional occupancies, high first.

    Integer occupations here would mean the active space collapsed to a
    single determinant and the test system stopped exercising the thing it
    exists for.
    """
    _mol, result = h2_casscf
    occ = np.asarray(result.natural_occupations)
    active = occ[(occ > 1e-8) & (occ < 2.0 - 1e-8)]
    assert active.size >= 2, f"expected fractional occupancies, got {occ}"
    # Descending, so "highest occupied" refers to something.
    assert np.all(np.diff(occ) <= 1e-12), f"not ordered by occupancy: {occ}"


def test_natural_orbitals_are_orthonormal(h2_casscf):
    """C^T S C = I. Orbitals that are not orthonormal are not orbitals."""
    mol, result = h2_casscf
    basis = vq.BasisSet(mol, BASIS)
    S = np.asarray(vq.compute_overlap(basis))
    C = np.asarray(result.natural_orbitals)
    gram = C.T @ S @ C
    assert np.abs(gram - np.eye(gram.shape[0])).max() < 1e-10


def test_density_from_natural_orbitals_recovers_the_electron_count(h2_casscf):
    """tr(P S) with P built from the orbitals and their occupations.

    Ties the two arrays together: they could each be individually plausible
    and still not describe the same wavefunction.
    """
    mol, result = h2_casscf
    basis = vq.BasisSet(mol, BASIS)
    S = np.asarray(vq.compute_overlap(basis))
    C = np.asarray(result.natural_orbitals)
    occ = np.asarray(result.natural_occupations)
    P = (C * occ) @ C.T
    assert np.trace(P @ S) == pytest.approx(mol.n_electrons(), abs=1e-8)


def test_casscf_qvf_carries_a_natural_wavefunction_section(tmp_path):
    """The archive gains a wavefunction.gto marked natural, with real occupations."""
    mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, BOND])])
    stem = tmp_path / "h2_cas"
    vq.run_job(
        mol, basis=BASIS, method="casscf", active_space=ACTIVE,
        output=str(stem), output_qvf=True,
    )
    path = stem.with_suffix(".qvf")
    assert path.exists()

    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        wf = [s for s in manifest["sections"] if s["kind"] == "wavefunction.gto"]
        assert wf, "CASSCF archive still has no wavefunction section"
        meta = json.loads(z.read(wf[0]["members"]["mo_metadata"]["path"]))

    assert meta["orbital_kind"] == "natural"
    occ = np.asarray(meta["occupations"], dtype=float)
    assert occ.sum() == pytest.approx(mol.n_electrons(), abs=1e-8)
    assert ((occ > 0.02) & (occ < 1.98)).sum() >= 2, "expected an active space"
    # No fabricated orbital spectrum: a natural orbital's eigenvalue is its
    # occupancy, not an energy.
    assert all(e == 0.0 for e in meta["energies"])


# ---------------------------------------------------------------------------
# GitLab #33: an ECP reference silently lost the whole wavefunction section.
#
# `_cas_natural_orbitals` sanity-checks its own core/active partition by
# summing the occupations and comparing them against the electron count. On an
# ECP reference the two counts are different things: the replaced core is not
# in the wavefunction, so the occupations sum to the VALENCE count while
# `molecule.n_electrons()` is the PHYSICAL one. The CASSCF branch passed the
# physical count, so the check compared 8 against 18 for H2S/LANL2DZ, decided
# the partition was wrong, and returned None.
#
# Nothing then said so. `natural_orbitals` came back None, the QVF gate that
# keys on it was skipped, and -- because a CASSCF has no mean-field MOs to
# fall back on -- the archive was written with no wavefunction section at all.
# The `.out` was unaffected, since its occupations come from the RDM directly,
# so the loss was visible only in the archive. These pin the valence count and
# the section that depends on it.
# ---------------------------------------------------------------------------

# H2S with LANL2DZ: 18 physical electrons, 10 replaced by the ECP, so the
# wavefunction holds 8. The two counts differ, which is the whole point.
ECP_PHYSICAL_ELECTRONS = 18
ECP_VALENCE_ELECTRONS = 8


def _h2s():
    return vq.Molecule(
        [
            vq.Atom(16, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.815, 1.425]),
            vq.Atom(1, [0.0, -1.815, 1.425]),
        ],
        0,
        1,
    )


@pytest.fixture(scope="module")
def h2s_ecp_casscf(tmp_path_factory):
    """One CASSCF(4e,4o) on an ECP reference, shared by the tests below."""
    stem = tmp_path_factory.mktemp("ecp_cas") / "h2s_lanl2dz"
    result = vq.run_job(
        _h2s(), basis="lanl2dz", method="casscf", active_space=(4, 4),
        output=str(stem), output_qvf=True, progress=False,
    )
    return result, stem


def test_ecp_casscf_publishes_natural_orbitals_at_the_valence_count(
    h2s_ecp_casscf,
):
    """The occupations must sum to the electrons the wavefunction holds.

    Pre-fix this returned None outright, because the sum (8) was checked
    against the physical count (18).
    """
    result, _ = h2s_ecp_casscf
    assert _h2s().n_electrons() == ECP_PHYSICAL_ELECTRONS  # guard the premise
    occ = result.natural_occupations
    assert occ is not None, (
        "an ECP reference lost its CASSCF natural orbitals entirely"
    )
    occ = np.asarray(occ, dtype=float)
    assert occ.sum() == pytest.approx(ECP_VALENCE_ELECTRONS, abs=1e-8)
    assert occ.sum() != pytest.approx(ECP_PHYSICAL_ELECTRONS, abs=1e-6)
    assert result.natural_orbitals is not None
    # Still a real active space, not a closed-shell determinant relabelled.
    assert ((occ > 0.002) & (occ < 1.998)).sum() >= 2


def test_ecp_casscf_archive_keeps_its_wavefunction_section(h2s_ecp_casscf):
    """The archive must not silently lose the wavefunction on an ECP.

    A CASSCF has no mean-field MOs, so when the natural-orbital payload is
    skipped there is no fallback: the .qvf came out with structure, citations
    and the run record and nothing else. The .out was unaffected, which is
    why this was invisible without opening the archive.
    """
    _, stem = h2s_ecp_casscf
    path = stem.with_suffix(".qvf")
    assert path.exists()
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        manifest = json.loads(z.read("manifest.json"))
        wf = [s for s in manifest["sections"] if s["kind"] == "wavefunction.gto"]
        assert wf, f"ECP CASSCF archive has no wavefunction section: {names}"
        meta = json.loads(z.read(wf[0]["members"]["mo_metadata"]["path"]))

    assert meta["orbital_kind"] == "natural"
    occ = np.asarray(meta["occupations"], dtype=float)
    # The archive states the valence count too, not the physical one.
    assert occ.sum() == pytest.approx(ECP_VALENCE_ELECTRONS, abs=1e-8)
