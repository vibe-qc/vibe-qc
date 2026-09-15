"""TREXIO writer + round-trip reader (GitLab #573, increment 1).

What is pinned here, and why each pin exists:

1. **The optional-extra boundary.** ``write_trexio`` / ``read_trexio`` import
   ``trexio`` lazily; without it they raise an ``ImportError`` that names the
   ``[trexio]`` extra. That test runs with the package *absent* (the module
   is masked), so it never depends on the extra being installed.
2. **The primitive-normalization convention.** ``basis.prim_factor`` is the
   axial-primitive norm ``N(alpha, l)``; the TREXIO specification's own
   worked H2 example (trex.org, "Basis set (basis group)", v2.6.1) prints
   the values for three primitives, and those published numbers are the
   targets here. A different convention (e.g. the 4-pi radial norm, which
   differs by ``sqrt(2l+1)`` for ``l >= 1``) fails this test at ``l = 1``.
3. **The AO ordering.** TREXIO ``m = 0, +1, -1, ...`` versus libint
   ``m = -l..+l`` with ``p = (py, pz, px)``; pinned as index lists.
4. **Exact round trips.** H2O/def2-SVP RHF and the OH radical UHF are
   written (both back ends), read back into vibe-qc objects, and every
   quantity agrees to 1e-12 -- including ``C^T S C = 1`` with the overlap
   rebuilt from the *file's* basis group, which is what proves the basis
   group reproduces the functions the MOs were expanded in.
5. **Refusals.** An ECP run (flag, and the density's electron count as the
   safety net) is refused rather than written with an incomplete
   Hamiltonian; a basis-free route refuses ``run_job(trexio=True)`` before
   the calculation.
6. **Runner wiring + citation.** ``run_job(trexio=True)`` produces the
   artefact, records it in the ``.system`` manifest, and fires
   ``posenitskiy2023trexio`` into ``.bibtex``; the negative control is the
   same route with the feature off.
7. **Out-of-process cross-check** (CLAUDE.md § 10): when
   ``VIBEQC_TREXIO_PYTHON`` names an interpreter with ``trexio`` + ``pyscf``,
   ``examples/regression/runner_trexio_pyscf.py`` rebuilds the SCF energy
   from the stored MOs with PySCF's own integrals and must agree with the
   file's ``state.energy``. This and the permutation check in section 1 are
   the only tests that can see a *consistent* AO convention error; the
   round trips in section 4 cancel it. Without that interpreter the check
   skips with a reason naming the gate that did not run, and fails instead
   when ``VIBEQC_REQUIRE_TREXIO_REFERENCE`` is set (#253, see
   ``tests/trexio_reference.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

import vibeqc
from tests.trexio_reference import reference_python
from vibeqc import Atom, BasisSet, Molecule, compute_kinetic, compute_nuclear, compute_overlap
from vibeqc._primitive_norm import libint_primitive_norm
from vibeqc.output.citations.registry import load_default_database
from vibeqc.output.formats import trexio as trexio_format
from vibeqc.output.formats.trexio import (
    _build_ao_permutation,
    _pure_trexio_to_libint,
    read_trexio,
    write_trexio,
)

_REPO = Path(__file__).resolve().parent.parent
_RUNNER = _REPO / "examples" / "regression" / "runner_trexio_pyscf.py"
_RESULT_MARKER = "VIBEQC-TREXIO-PYSCF-RESULT:"


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


def _h2o() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.98]),
            Atom(1, [0.0, -1.43, -0.98]),
        ]
    )


def _oh_radical() -> Molecule:
    return Molecule([Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.83])], multiplicity=2)


@pytest.fixture(scope="module")
def h2o_rhf():
    mol = _h2o()
    basis = BasisSet(mol, "def2-svp")
    result = vibeqc.run_rhf(mol, basis)
    assert result.converged
    return mol, basis, result


@pytest.fixture(scope="module")
def oh_uhf():
    mol = _oh_radical()
    basis = BasisSet(mol, "def2-svp")
    result = vibeqc.run_uhf(mol, basis)
    assert result.converged
    return mol, basis, result


# --------------------------------------------------------------------- #
# 1. Optional-extra boundary (runs WITHOUT trexio)                      #
# --------------------------------------------------------------------- #


def test_missing_trexio_raises_importerror_naming_the_extra(monkeypatch, tmp_path):
    # Masking the module makes ``import trexio`` raise ImportError whether or
    # not the package is installed, so this test is meaningful in both venvs.
    monkeypatch.setitem(sys.modules, "trexio", None)
    with pytest.raises(ImportError) as excinfo:
        trexio_format._require_trexio()
    message = str(excinfo.value)
    assert "vibe-qc[trexio]" in message
    assert "BSD" in message
    # Both public entry points go through the same gate.
    with pytest.raises(ImportError, match=r"vibe-qc\[trexio\]"):
        read_trexio(tmp_path / "absent.h5")
    with pytest.raises(ImportError, match=r"vibe-qc\[trexio\]"):
        write_trexio(tmp_path / "x.h5", None, None, None)


def test_trexio_is_not_a_core_dependency():
    # CLAUDE.md § 9: the package must stay importable and usable without the
    # extra. The writer module itself must not import trexio at module level.
    source = (
        _REPO / "python" / "vibeqc" / "output" / "formats" / "trexio.py"
    ).read_text(encoding="utf-8")
    module_level_imports = [
        line
        for line in source.splitlines()
        if line.startswith(("import trexio", "from trexio"))
    ]
    assert module_level_imports == []
    pyproject = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'trexio = [' in pyproject
    core = pyproject.split("[project.optional-dependencies]")[0]
    assert "trexio" not in core.split("dependencies = [")[-1].split("]")[0]


# --------------------------------------------------------------------- #
# 2. + 3. Conventions pinned against the specification                  #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "alpha, l, published",
    [
        # trex.org v2.6.1, "Basis set (basis group)", worked H2 example:
        # ``prim_factor`` for the first S primitive, the first P primitive and
        # the D primitive of the listed GAMESS-format basis.
        (33.87, 0, 1.0006253235944540e01),
        (1.407, 1, 2.1842769845268308e00),
        (1.057, 2, 1.8135965626177861e00),
    ],
)
def test_prim_factor_reproduces_the_specifications_worked_example(alpha, l, published):
    assert libint_primitive_norm(alpha, l) == pytest.approx(published, rel=1e-13)


def test_spherical_ao_permutation_matches_trexio_order():
    # TREXIO: m = 0, +1, -1, +2, -2, ...; p is (pz, px, py).
    # libint: m = -l..+l; pure p is (py, pz, px).
    assert _pure_trexio_to_libint(0) == [0]
    assert _pure_trexio_to_libint(1) == [1, 2, 0]
    assert _pure_trexio_to_libint(2) == [2, 3, 1, 4, 0]
    assert _pure_trexio_to_libint(3) == [3, 4, 2, 5, 1, 6, 0]
    perm = _build_ao_permutation([0, 1, 2])
    assert perm.tolist() == [0, 2, 3, 1, 6, 7, 5, 8, 4]
    assert sorted(perm.tolist()) == list(range(9))


# --------------------------------------------------------------------- #
# 4. Exact round trips (need the extra)                                 #
# --------------------------------------------------------------------- #

trexio = pytest.importorskip("trexio")


def _target(tmp_path: Path, backend: str, stem: str) -> Path:
    return tmp_path / (f"{stem}.trexio.h5" if backend == "hdf5" else f"{stem}.trexio")


@pytest.mark.parametrize("backend", ["hdf5", "text"])
def test_rhf_round_trip_is_exact(h2o_rhf, tmp_path, backend):
    mol, basis, result = h2o_rhf
    path = write_trexio(
        _target(tmp_path, backend, "h2o"), mol, basis, result, backend=backend,
        description="pytest",
    )
    assert path.exists()
    if backend == "text":
        assert path.is_dir()
        assert {"nucleus.txt", "basis.txt", "ao.txt", "mo.txt", "ao_1e_int.txt"} <= set(
            os.listdir(path)
        )

    data = read_trexio(path)

    # nuclei: charges, bohr coordinates, labels, repulsion
    assert data.labels == ["O", "H", "H"]
    np.testing.assert_allclose(data.charges, [8.0, 1.0, 1.0], atol=0.0)
    np.testing.assert_allclose(
        data.coords, [[a.xyz[0], a.xyz[1], a.xyz[2]] for a in mol.atoms], atol=1e-12
    )
    assert data.nuclear_repulsion == pytest.approx(mol.nuclear_repulsion(), abs=1e-12)
    assert (data.n_up, data.n_dn) == (5, 5)

    # basis: shell metadata and the a_ks / f_ks split reproduce c_eff exactly
    shells = list(basis.shells())
    assert data.shell_ang_mom == [s.l for s in shells]
    assert data.shell_atom == [s.atom_index for s in shells]
    np.testing.assert_allclose(data.shell_factor, 1.0, atol=0.0)
    k = 0
    for s, shell in enumerate(shells):
        for alpha, c_eff in zip(shell.exponents, shell.coefficients):
            assert data.prim_shell[k] == s
            assert data.exponents[k] == pytest.approx(alpha, rel=1e-14)
            assert data.prim_factor[k] == pytest.approx(
                libint_primitive_norm(alpha, shell.l), rel=1e-14
            )
            assert data.coefficients[k] * data.prim_factor[k] == pytest.approx(
                c_eff, rel=1e-12, abs=1e-14
            )
            k += 1
    assert k == len(data.exponents)

    # molecule + basis rebuilt from the FILE reproduce the overlap exactly
    mol2 = data.molecule()
    assert mol2.n_electrons() == 10 and mol2.multiplicity == 1 and mol2.charge == 0
    basis2 = data.basis_set(mol2)
    assert basis2.nbasis == basis.nbasis == 24
    S1 = np.asarray(compute_overlap(basis))
    S2 = np.asarray(compute_overlap(basis2))
    np.testing.assert_allclose(S2, S1, atol=1e-12)

    # MOs: coefficients, energies, occupations, orthonormality in S2
    (block,) = data.mo_blocks()
    assert block.spin == 0
    C = np.asarray(result.mo_coeffs)
    np.testing.assert_allclose(block.coefficients, C, atol=1e-12)
    np.testing.assert_allclose(block.energies, np.asarray(result.mo_energies), atol=1e-12)
    assert block.occupations.tolist() == [2.0] * 5 + [0.0] * 19
    assert block.classes[:5] == ["Inactive"] * 5 and block.classes[5] == "Virtual"
    gram = block.coefficients.T @ S2 @ block.coefficients
    assert np.abs(gram - np.eye(24)).max() < 1e-12
    assert data.mo_type == "RHF"

    # one-electron integrals stored in TREXIO AO order, mapped back exactly
    np.testing.assert_allclose(data.to_libint_order(data.overlap), S1, atol=1e-12)
    T = np.asarray(compute_kinetic(basis))
    V = np.asarray(compute_nuclear(basis, mol))
    np.testing.assert_allclose(data.to_libint_order(data.kinetic), T, atol=1e-12)
    np.testing.assert_allclose(data.to_libint_order(data.potential_n_e), V, atol=1e-12)
    np.testing.assert_allclose(data.to_libint_order(data.core_hamiltonian), T + V, atol=1e-12)
    # and the stored matrices are what the stored MOs are orthonormal in
    Cs = data.mo_coefficient  # (mo, ao) in TREXIO order
    assert np.abs(Cs @ data.overlap @ Cs.T - np.eye(24)).max() < 1e-12

    # energy + provenance
    assert data.energy == pytest.approx(result.energy, abs=1e-12)
    assert data.metadata["code"] == [f"vibe-qc {vibeqc.__version__}"]
    assert "pytest" in data.metadata["description"]
    assert "git" in data.metadata["description"]
    assert "author" not in data.metadata  # nothing identifying by default


def test_uhf_round_trip_writes_spin_orbital_blocks(oh_uhf, tmp_path):
    mol, basis, result = oh_uhf
    path = write_trexio(tmp_path / "oh.h5", mol, basis, result)
    data = read_trexio(path)
    assert (data.n_up, data.n_dn) == (5, 4)
    assert data.mo_type == "UHF"
    assert data.mo_coefficient.shape == (2 * basis.nbasis, basis.nbasis)
    assert sorted(set(data.mo_spin.tolist())) == [0, 1]

    mol2 = data.molecule()
    assert mol2.multiplicity == 2 and mol2.charge == 0
    S2 = np.asarray(compute_overlap(data.basis_set(mol2)))
    blocks = {b.spin: b for b in data.mo_blocks()}
    for spin, C_ref, e_ref, n_occ in (
        (0, result.mo_coeffs_alpha, result.mo_energies_alpha, 5),
        (1, result.mo_coeffs_beta, result.mo_energies_beta, 4),
    ):
        block = blocks[spin]
        np.testing.assert_allclose(block.coefficients, np.asarray(C_ref), atol=1e-12)
        np.testing.assert_allclose(block.energies, np.asarray(e_ref), atol=1e-12)
        assert block.occupations.sum() == pytest.approx(n_occ, abs=0.0)
        assert block.occupations[:n_occ].tolist() == [1.0] * n_occ
        assert np.abs(block.coefficients.T @ S2 @ block.coefficients - np.eye(basis.nbasis)).max() < 1e-12
    assert data.energy == pytest.approx(result.energy, abs=1e-12)


def test_rewrite_replaces_previous_artefact(h2o_rhf, tmp_path):
    mol, basis, result = h2o_rhf
    target = tmp_path / "again.h5"
    write_trexio(target, mol, basis, result)
    write_trexio(target, mol, basis, result)  # TREXIO refuses duplicates unless we clear
    assert read_trexio(target).energy == pytest.approx(result.energy, abs=1e-12)
    with pytest.raises(FileExistsError):
        write_trexio(target, mol, basis, result, overwrite=False)
    # A directory that is not a TREXIO text artefact is never removed.
    foreign = tmp_path / "keep"
    foreign.mkdir()
    (foreign / "precious.dat").write_text("do not delete")
    with pytest.raises(FileExistsError):
        write_trexio(foreign, mol, basis, result, backend="text")
    assert (foreign / "precious.dat").exists()


# --------------------------------------------------------------------- #
# 5. Refusals                                                            #
# --------------------------------------------------------------------- #


def test_ecp_flag_is_refused_with_a_clear_message(h2o_rhf, tmp_path):
    mol, basis, result = h2o_rhf
    with pytest.raises(ValueError, match="ecp"):
        write_trexio(tmp_path / "ecp.h5", mol, basis, result, uses_ecp=True)
    assert not (tmp_path / "ecp.h5").exists()


class _ValenceOnlyResult:
    """A converged-looking result whose density holds fewer electrons than
    the molecule -- the signature of an ECP run reaching the writer without
    the flag."""

    def __init__(self, result, fraction: float):
        self.energy = result.energy
        self.mo_coeffs = np.asarray(result.mo_coeffs)
        self.mo_energies = np.asarray(result.mo_energies)
        self.density = fraction * np.asarray(result.density)


def test_density_electron_count_mismatch_is_refused(h2o_rhf, tmp_path):
    mol, basis, result = h2o_rhf
    with pytest.raises(ValueError, match="core potential"):
        write_trexio(tmp_path / "valence.h5", mol, basis, _ValenceOnlyResult(result, 0.8))
    assert not (tmp_path / "valence.h5").exists()
    # Negative control: the same stub with the full density writes fine.
    write_trexio(tmp_path / "full.h5", mol, basis, _ValenceOnlyResult(result, 1.0))
    assert (tmp_path / "full.h5").exists()


def test_bad_backend_is_refused(h2o_rhf, tmp_path):
    mol, basis, result = h2o_rhf
    with pytest.raises(ValueError, match="backend"):
        write_trexio(tmp_path / "x.h5", mol, basis, result, backend="netcdf")


def test_run_job_refuses_trexio_on_a_basis_free_route(tmp_path):
    # Guarantee semantics: fail before the calculation, like write_molden_file=True.
    with pytest.raises(NotImplementedError, match="trexio"):
        vibeqc.run_job(
            _h2o(), method="dftb0", output=tmp_path / "dftb", trexio=True,
            verbose=False, progress=False,
        )
    assert not (tmp_path / "dftb.trexio.h5").exists()


# --------------------------------------------------------------------- #
# 6. Runner wiring + citation                                            #
# --------------------------------------------------------------------- #


def test_citation_route_fires_only_when_trexio_is_used():
    db = load_default_database()
    with_trexio = db.assemble(method="rhf", basis="def2-svp", extra_libraries=["trexio"])
    keys = {c.key for c in with_trexio.citations}
    assert "posenitskiy2023trexio" in keys
    assert "posenitskiy2023trexio" in {c.key for c in with_trexio.printable}
    # Negative control: the same route with the feature off (L125).
    without = db.assemble(method="rhf", basis="def2-svp")
    assert "posenitskiy2023trexio" not in {c.key for c in without.citations}


def test_citation_entry_is_the_verified_paper():
    entry = load_default_database().entries()["posenitskiy2023trexio"]
    assert entry.doi == "10.1063/5.0148161"
    assert int(entry.year) == 2023
    assert entry.authors[0].startswith("Posenitskiy")
    assert entry.authors[-1].startswith("Scemama")


@pytest.mark.parametrize("backend", ["hdf5", "text"])
def test_run_job_writes_records_and_cites_the_artefact(tmp_path, backend):
    stem = tmp_path / "h2o"
    result = vibeqc.run_job(
        _h2o(), basis="def2-svp", method="rhf", output=stem, trexio=True,
        trexio_backend=backend, verbose=False, progress=False,
    )
    artefact = _target(tmp_path, backend, "h2o")
    assert artefact.exists()
    data = read_trexio(artefact)
    assert data.energy == pytest.approx(result.energy, abs=1e-10)
    assert "RHF/def2-svp" in data.metadata["description"]

    manifest = tomllib.loads((tmp_path / "h2o.system").read_text(encoding="utf-8"))
    # Declared in the plan with its role / format ...
    plan_rows = [r for r in manifest["plan"]["files"] if r["path"].endswith(artefact.name)]
    assert len(plan_rows) == 1
    assert plan_rows[0]["format"] == "trexio"
    assert plan_rows[0]["role"] == "orbitals"
    # ... and recorded as written in the outputs table.
    rows = [r for r in manifest["outputs"]["files"] if r["path"].endswith(artefact.name)]
    assert len(rows) == 1
    assert rows[0]["written"] is True
    assert rows[0]["error"] == ""
    if backend == "hdf5":
        assert rows[0]["bytes"] == artefact.stat().st_size
        assert len(rows[0]["sha256"]) == 64
        assert rows[0]["checksum_status"] == "sha256"

    assert "posenitskiy" in (tmp_path / "h2o.bibtex").read_text(encoding="utf-8").lower()
    assert "TREXIO" in (tmp_path / "h2o.references").read_text(encoding="utf-8")
    assert "TREXIO wavefunction queued" in (tmp_path / "h2o.out").read_text(encoding="utf-8")
    # The Molden sibling is untouched by the new artefact.
    assert (tmp_path / "h2o.molden").exists()


def test_run_job_trexio_accepts_an_explicit_path(tmp_path):
    target = tmp_path / "custom" / "wavefunction.h5"
    vibeqc.run_job(
        _h2o(), basis="def2-svp", method="rhf", output=tmp_path / "h2o",
        trexio=target, verbose=False, progress=False,
    )
    assert target.exists()
    assert not (tmp_path / "h2o.trexio.h5").exists()


def test_run_job_default_writes_no_trexio_and_cites_nothing(tmp_path):
    # Negative control for the wiring: the same route with the feature off.
    vibeqc.run_job(
        _h2o(), basis="def2-svp", method="rhf", output=tmp_path / "h2o",
        verbose=False, progress=False,
    )
    assert not (tmp_path / "h2o.trexio.h5").exists()
    assert "posenitskiy" not in (tmp_path / "h2o.bibtex").read_text(encoding="utf-8").lower()
    manifest = tomllib.loads((tmp_path / "h2o.system").read_text(encoding="utf-8"))
    assert not any(r["format"] == "trexio" for r in manifest["plan"]["files"])
    assert not any("trexio" in Path(r["path"]).name for r in manifest["outputs"]["files"])


# --------------------------------------------------------------------- #
# 7. Out-of-process PySCF cross-check (CLAUDE.md § 10)                   #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("case", ["rhf", "uhf"])
def test_pyscf_rebuilds_the_scf_energy_from_the_stored_mos(
    h2o_rhf, oh_uhf, tmp_path, case
):
    # Skips with a reason naming the gate, or fails when
    # VIBEQC_REQUIRE_TREXIO_REFERENCE is set (#253).
    exe = reference_python()
    mol, basis, result = h2o_rhf if case == "rhf" else oh_uhf
    path = write_trexio(tmp_path / f"{case}.h5", mol, basis, result)
    verdict_path = tmp_path / f"{case}.verdict.json"
    proc = subprocess.run(
        [exe, str(_RUNNER), str(path), "--json", str(verdict_path)],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [l for l in proc.stdout.splitlines() if l.startswith(_RESULT_MARKER)]
    assert len(lines) == 1, proc.stdout
    verdict = json.loads(lines[0][len(_RESULT_MARKER):])
    assert verdict_path.exists()
    print(f"[trexio-pyscf {case}] verdict at {verdict_path}: {verdict}")
    assert verdict["verdict"] == "pass", verdict
    assert verdict["energy_diff"] < 1e-8
    assert verdict["max_orthonormality_error"] < 1e-8
    assert verdict["overlap_max_diff"] < 1e-8
