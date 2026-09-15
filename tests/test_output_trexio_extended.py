"""TREXIO ECP, density, periodic, general-data and READ contracts.

The numerical checks reconstruct the Hamiltonian/density in a fresh basis;
raw-data checks cover distinct TREXIO storage types on both real backends.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.output.formats.trexio import (
    TrexioSparse, read_trexio, read_trexio_fields, write_trexio, write_trexio_fields,
)

trexio = pytest.importorskip("trexio")


@pytest.fixture(params=["hdf5", "text"])
def backend(request):
    return request.param


@pytest.fixture(scope="module")
def water():
    molecule = Molecule([Atom(8, [0, 0, 0]), Atom(1, [0, 1.43, -.98]), Atom(1, [0, -1.43, -.98])])
    basis = BasisSet(molecule, "def2-svp")
    result = vibeqc.run_rhf(molecule, basis)
    assert result.converged
    return molecule, basis, result


@pytest.fixture(scope="module")
def nah():
    molecule = Molecule([Atom(11, [0, 0, 0]), Atom(1, [0, 0, 3.5])])
    basis = BasisSet(molecule, "lanl2dz")
    result = vibeqc.run_rhf(molecule, basis)
    assert result.converged
    assert result.ecp_total_ncore == 10
    return molecule, basis, result


def test_density_and_mo_hamiltonian(water, backend, tmp_path):
    mol, basis, result = water
    data = read_trexio(write_trexio(tmp_path / "wave", mol, basis, result,
                                  backend=backend, write_mo_integrals=True))
    C = data.mo_blocks()[0].coefficients
    np.testing.assert_allclose(data.density_matrices()[0], result.density, atol=2e-12)
    np.testing.assert_allclose(C @ data.fields["rdm_1e"] @ C.T, result.density, atol=2e-12)
    H = vibeqc.compute_kinetic(basis) + vibeqc.compute_nuclear(basis, mol)
    np.testing.assert_allclose(data.fields["mo_1e_int_core_hamiltonian"], C.T @ H @ C, atol=2e-12)
    assert np.trace(data.fields["rdm_1e"]) == pytest.approx(mol.n_electrons(), abs=1e-10)


def test_fractional_density_populations_are_not_replaced_by_aufbau(water, backend, tmp_path):
    mol, basis, result = water
    C = np.asarray(result.mo_coeffs)
    occ = np.zeros(C.shape[1])
    occ[:4], occ[4:6] = 2., [1.5, .5]
    density = (C * occ) @ C.T
    source = SimpleNamespace(mo_coeffs=C, mo_energies=result.mo_energies, density=density)
    data = read_trexio(write_trexio(tmp_path / "ensemble", mol, basis, source, backend=backend))
    np.testing.assert_allclose(data.mo_occupation, occ, atol=1e-12)
    np.testing.assert_allclose(data.density_matrices()[0], density, atol=2e-12)


def test_unrestricted_orbitals_without_optional_energies(water, backend, tmp_path):
    mol, basis, result = water
    source = SimpleNamespace(mo_coeffs_alpha=result.mo_coeffs, mo_coeffs_beta=result.mo_coeffs,
                             density_alpha=np.asarray(result.density) / 2,
                             density_beta=np.asarray(result.density) / 2)
    data = read_trexio(write_trexio(tmp_path / "no-energies", mol, basis, source, backend=backend))
    assert "mo_energy" not in data.fields
    for block, density in zip(data.mo_blocks(), data.density_matrices()):
        assert np.all(np.isnan(block.energies))
        np.testing.assert_allclose(density, source.density_alpha, atol=2e-12)


def test_ecp_hamiltonian_and_seeded_calculation(nah, backend, tmp_path):
    from vibeqc.ecp_metadata import one_electron_hamiltonian
    mol, basis, result = nah
    path = write_trexio(tmp_path / ("nah.h5" if backend == "hdf5" else "nah"), mol, basis, result, backend=backend)
    data = read_trexio(path)
    np.testing.assert_array_equal(data.atomic_numbers, [11, 1])
    np.testing.assert_array_equal(data.charges, [1, 1])
    np.testing.assert_array_equal(data.fields["ecp_z_core"], [10, 0])
    rebuilt_mol, rebuilt_basis = data.molecule(), data.basis_set()
    options = data.ecp_options()
    H, enuc, nelec, _ = one_electron_hamiltonian(rebuilt_mol, rebuilt_basis, options)
    np.testing.assert_allclose(H, data.to_libint_order(data.core_hamiltonian), atol=2e-11)
    assert enuc == pytest.approx(data.nuclear_repulsion, abs=1e-13)
    assert nelec == data.n_electrons == 2
    periodic_options = data.ecp_options(vibeqc.PeriodicRHFOptions())
    assert periodic_options.ecp_total_ncore == 10
    np.testing.assert_allclose(periodic_options.ecp_home_centers, options.ecp_primitive_centers)
    options.initial_guess = vibeqc.InitialGuess.READ
    restarted = vibeqc.run_rhf(rebuilt_mol, rebuilt_basis, options, read_from=path)
    assert restarted.converged
    assert restarted.energy == pytest.approx(result.energy, abs=2e-9)
    # Re-export the inline restart result: no dependency on the XML library.
    second = read_trexio(write_trexio(tmp_path / "inline", rebuilt_mol, rebuilt_basis, restarted, backend=backend))
    np.testing.assert_allclose(second.core_hamiltonian, data.core_hamiltonian, atol=2e-11)
    no_labels = dict(data.fields)
    del no_labels["nucleus_label"]
    restored = read_trexio(write_trexio_fields(tmp_path / "nolabel", no_labels, backend=backend))
    assert [a.Z for a in restored.molecule().atoms] == [11, 1]


def test_molecular_run_job_reads_trexio(water, backend, tmp_path):
    mol, basis, result = water
    path = write_trexio(tmp_path / "prior", mol, basis, result, backend=backend)
    # Directory detection also accepts a text target without a suffix.
    if backend == "hdf5":
        renamed = path.with_suffix(".h5")
        path.rename(renamed)
        path = renamed
    restarted = vibeqc.run_job(mol, basis="def2-svp", method="rhf", output=tmp_path / "restart",
                               initial_guess="read", read_from=path, verbose=False, progress=False)
    assert restarted.converged
    assert restarted.energy == pytest.approx(result.energy, abs=2e-9)


def test_general_groups_and_backend_conversion(backend, tmp_path):
    sparse = TrexioSparse(np.array([[0, 0, 0, 0], [1, 1, 1, 1]]), np.array([.6, .2]))
    fields = dict(
        mo_num=np.int64(2), electron_up_num=1, electron_dn_num=1,
        determinant_list=np.array([[1, 1], [2, 2]], dtype=np.int64),
        determinant_coefficient=np.array([.8, .6]),
        csf_num=1, csf_coefficient=np.array([1.]),
        csf_det_coefficient=TrexioSparse(np.array([[0, 0], [0, 1]]), np.array([.8, .6])),
        mo_2e_int_eri=sparse, rdm_2e=sparse, rdm_1e=np.diag([1.28, .72]),
        amplitude_single=TrexioSparse(np.array([[0, 1]]), np.array([.1])),
        grid_num=2, grid_coord=np.array([0., 1.]), grid_weight=np.ones(2),
        qmc_num=2, qmc_e_loc=np.array([-1., -.9]), qmc_psi=np.array([.8, .7]),
        jastrow_type="CHAMP", jastrow_ee_num=2, jastrow_ee=np.array([.5, .1]),
        state_num=2, state_id=1, state_energy=-1., state_label=["ground", "excited"],
    )
    path = write_trexio_fields(tmp_path / "raw", fields, backend=backend, chunk_size=1)
    data = read_trexio(path)
    assert data.energy == -1.
    # Partial files and non-wavefunction groups need no synthetic basis.
    with pytest.raises(ValueError, match="Gaussian"):
        data.basis_set()
    other = "text" if backend == "hdf5" else "hdf5"
    roundtrip = read_trexio_fields(data.write(tmp_path / "converted", backend=other), chunk_size=1)
    for key, value in fields.items():
        if isinstance(value, TrexioSparse):
            np.testing.assert_array_equal(roundtrip[key].indices, value.indices)
            np.testing.assert_array_equal(roundtrip[key].values, value.values)
        else:
            np.testing.assert_array_equal(roundtrip[key], value)
    assert roundtrip["determinant_num"] == 2
    assert set(read_trexio_fields(path, fields=["state_energy"])) == {"state_energy"}
    data.write(path, backend=backend)


def test_unicode_strings_and_long_hdf5_metadata(backend, tmp_path):
    fields = dict(metadata_description="Orbital αβγ", metadata_author_num=2,
                  metadata_author=["André", "Zoë"], mo_num=10000,
                  mo_class=["Virtual"] * 10000)
    path = write_trexio_fields(tmp_path / "unicode", fields, backend=backend)
    restored = read_trexio_fields(path)
    for name, expected in fields.items():
        assert restored[name] == expected
    fields["metadata_description"] *= 1000
    fields["metadata_author"][0] *= 1000
    long_path = write_trexio_fields(tmp_path / "long.h5", fields)
    restored = read_trexio_fields(long_path)
    for name, expected in fields.items():
        assert restored[name] == expected
    # TREXIO 2.6's text parser truncates lines at 1023 bytes. Refuse an
    # unrepresentable conversion and keep the previous artifact intact.
    with pytest.raises(ValueError, match="use HDF5"):
        write_trexio_fields(path, fields, backend="text")
    assert read_trexio_fields(path)["metadata_description"] == "Orbital αβγ"


def test_ci_bitfields_across_the_signed_64_bit_boundary(backend, tmp_path):
    from vibeqc.output.formats._trexio_ci import ci_wavefunction
    ci = SimpleNamespace(determinants=[((63,), (64,))], ci_coeffs=np.array([1.]),
                         n_active_orb=70, e_total=0.)
    result = ci_wavefunction(np.eye(70), ci)
    fields = dict(result.trexio_fields, mo_num=70, electron_up_num=1, electron_dn_num=1)
    path = write_trexio_fields(tmp_path / "bits", fields, backend=backend)
    bits = read_trexio_fields(path)["determinant_list"][0]
    assert bits[0] == np.iinfo(np.int64).min
    alpha, beta = trexio.to_orbital_list_up_dn(2, bits)
    np.testing.assert_array_equal(alpha, [63])
    np.testing.assert_array_equal(beta, [64])


def test_sparse_eri_physicist_order(backend, tmp_path):
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "sto-3g")
    result = vibeqc.run_rhf(mol, basis)
    data = read_trexio(write_trexio(tmp_path / "eri", mol, basis, result, backend=backend, write_eri=True))
    values = data.fields["ao_2e_int_eri"]
    g = np.zeros((2,) * 4)
    g[tuple(values.indices.T)] = values.values
    # Density contraction reconstructed from physicists' indices must yield
    # the electronic HF interaction energy, not merely a reshaped tensor.
    D = np.asarray(result.density)
    interaction = .5 * np.einsum('pq,rs,prqs', D, D, g) - .25 * np.einsum('pq,rs,prsq', D, D, g)
    onebody = np.einsum('pq,pq', D, data.core_hamiltonian)
    assert onebody + interaction + data.nuclear_repulsion == pytest.approx(result.energy, abs=1e-10)


def test_complex_periodic_blocks_and_restart(water, backend, tmp_path):
    from vibeqc.guess_read import resolve_periodic_read_density_k_closed
    mol, basis, result = water
    lattice = np.array([[13., 1., .5], [0., 14., 1.], [0., 0., 15.]])
    system = vibeqc.PeriodicSystem(dim=3, lattice=lattice, unit_cell=mol.atoms)
    fractional = np.array([[.125, .25, 0.], [-.125, -.25, 0.]])
    points = fractional @ (2 * np.pi * np.linalg.inv(lattice))
    coefficients = [np.asarray(result.mo_coeffs) * np.exp(1j * phase) for phase in [.2, -.2]]
    occupations = np.zeros(basis.nbasis)
    occupations[:5] = 2.
    source = SimpleNamespace(mo_coeffs=coefficients, mo_energies=[result.mo_energies] * 2,
                             occupations=[occupations] * 2, energy=result.energy,
                             kpoints_cart=points, kpoint_weights=np.array([.4, .6]))
    path = write_trexio(tmp_path / "bloch.h5" if backend == "hdf5" else tmp_path / "bloch",
                       mol, basis, source, system=system, backend=backend)
    data = read_trexio(path)
    np.testing.assert_allclose(data.fields["pbc_k_point"], fractional, atol=1e-15)
    np.testing.assert_allclose(data.kpoints, points, atol=1e-15)
    np.testing.assert_allclose(data.periodic_system().lattice, lattice)
    assert [b.k_point for b in data.mo_blocks()] == [0, 1]
    for block, expected in zip(data.mo_blocks(), coefficients):
        np.testing.assert_allclose(block.coefficients, expected, atol=1e-13)
    # Multi-k AO integrals have no matching TREXIO dimensions; never store a
    # home-cell molecular overlap under a Bloch-wavefunction label.
    assert data.overlap is None
    mesh = SimpleNamespace(kpoints=points[::-1], weights=np.array([.6, .4]))
    densities = resolve_periodic_read_density_k_closed(read_from=path, basis=basis, system=system, kmesh=mesh)
    for density, C in zip(densities, coefficients[::-1]):
        # This synthetic source carries MOs only. Its density is defined by
        # those MOs, not the native SCF's last pre-diagonalization density.
        np.testing.assert_allclose(density, (C * occupations) @ C.conj().T, atol=2e-12)


@pytest.mark.parametrize("nk", [1, 2])
def test_periodic_restricted_open_shell_preserves_degenerate_occupations(water, backend, nk, tmp_path):
    # The periodic ROHF/ROKS producers expose mo_occupations plus identical
    # alpha/beta MO aliases, without occupations_alpha/beta. Three degenerate
    # frontier orbitals share two alpha electrons in this triplet fixture.
    mol, basis, reference = water
    system = vibeqc.PeriodicSystem(3, np.eye(3) * 30., mol.atoms, multiplicity=3)
    occupations = np.zeros(basis.nbasis)
    occupations[:4], occupations[4:7] = 2., 2. / 3.
    def layout(array):
        return array if nk == 1 else [array] * nk
    C, energies = layout(reference.mo_coeffs), layout(reference.mo_energies)
    source = SimpleNamespace(mo_coeffs=C, mo_coeffs_alpha=C, mo_coeffs_beta=C,
                             mo_energies=energies, mo_energies_alpha=energies,
                             mo_energies_beta=energies, mo_occupations=layout(occupations))
    # Raw GPW Bloch results also carry Gamma matrices under *_k aliases.
    overlap = vibeqc.compute_overlap(basis)
    hcore = vibeqc.compute_kinetic(basis) + vibeqc.compute_nuclear(basis, mol)
    source.overlap_k, source.hcore_k = [overlap] * nk, [hcore] * nk
    mesh = vibeqc.monkhorst_pack(system, [nk, 1, 1])
    path = write_trexio(tmp_path / "open-shell", mol, basis, source, system=system,
                       kpoints=mesh.kpoints, weights=mesh.weights, backend=backend)
    data = read_trexio(path)
    assert (data.n_up, data.n_dn) == (6, 4)
    if nk == 1:
        np.testing.assert_allclose(data.to_libint_order(data.overlap), overlap, atol=1e-14)
        np.testing.assert_allclose(data.to_libint_order(data.core_hamiltonian), hcore, atol=1e-14)
    else:
        assert data.overlap is None and data.core_hamiltonian is None
    for block in data.mo_blocks():
        expected = np.minimum(occupations, 1.) if block.spin == 0 else np.maximum(occupations - 1., 0.)
        np.testing.assert_allclose(block.occupations, expected, atol=1e-15)
        assert block.occupations.sum() == pytest.approx(6 if block.spin == 0 else 4)


def test_interleaved_shells_and_ao_normalization(water, backend, tmp_path):
    mol, basis, result = water
    path = write_trexio(tmp_path / "base", mol, basis, result, backend=backend)
    fields = read_trexio_fields(path)
    norm = np.linspace(.8, 1.3, basis.nbasis)
    shuffle = np.arange(basis.nbasis)[::-1]
    # Reverse the order of whole shell chunks, retaining TREXIO m ordering.
    shuffle = np.concatenate([np.flatnonzero(np.asarray(fields["ao_shell"]) == shell)
                              for shell in reversed(range(fields["basis_shell_num"]))])
    fields["ao_normalization"] = norm[shuffle]
    fields["ao_shell"] = np.asarray(fields["ao_shell"])[shuffle]
    fields["mo_coefficient"] = (fields["mo_coefficient"] / norm)[:, shuffle]
    for key in list(fields):
        if key.startswith("ao_1e_int_"):
            fields[key] = (fields[key] * np.outer(norm, norm))[np.ix_(shuffle, shuffle)]
    data = read_trexio(write_trexio_fields(tmp_path / "scaled", fields, backend=backend))
    np.testing.assert_allclose(data.mo_blocks()[0].coefficients, result.mo_coeffs, atol=1e-13)
    np.testing.assert_allclose(data.to_libint_order(data.overlap), vibeqc.compute_overlap(data.basis_set()), atol=1e-12)


def test_periodic_open_shell_and_gpw_layout(backend, tmp_path):
    from vibeqc.guess_read import resolve_periodic_read_densities_k_open
    mol = Molecule([Atom(3, [0., 0., 0.])], multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    result = vibeqc.run_uhf(mol, basis)
    system = vibeqc.PeriodicSystem(3, np.eye(3) * 12., mol.atoms, multiplicity=2)
    points = np.array([[.1, 0., 0.], [-.1, 0., 0.]])
    values = dict(energy=result.energy, kpoints_cart=points, kpoint_weights=np.array([.5, .5]))
    for spin in ("alpha", "beta"):
        C = np.asarray(getattr(result, "mo_coeffs_" + spin))
        values["mo_coeffs_" + spin + "_k"] = [C * np.exp(.3j), C * np.exp(-.3j)]
        values["mo_energies_" + spin + "_k"] = [np.asarray(getattr(result, "mo_energies_" + spin), dtype=complex)] * 2
    path = write_trexio(tmp_path / "spin.h5" if backend == "hdf5" else tmp_path / "spin",
                       mol, basis, SimpleNamespace(**values), system=system, backend=backend)
    data = read_trexio(path)
    assert [(b.k_point, b.spin) for b in data.mo_blocks()] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    mesh = SimpleNamespace(kpoints=points, weights=np.array([.5, .5]))
    for densities, spin in zip(resolve_periodic_read_densities_k_open(
            read_from=path, basis=basis, system=system, kmesh=mesh), ("alpha", "beta")):
        for density in densities:
            np.testing.assert_allclose(density, getattr(result, "density_" + spin), atol=2e-11)
    # A general RDM may carry coherences that a spin/k-block SCF cannot use.
    # Preserve it as data, but never silently delete the off-block entries.
    rdm = np.diag(data.mo_occupation)
    rdm[0, basis.nbasis] = rdm[basis.nbasis, 0] = .01
    data.fields["rdm_1e"] = rdm
    with pytest.raises(ValueError, match="coherence"):
        data.restart_result()


def test_preserve_non_gaussian_and_complex_basis(backend, tmp_path):
    fields = dict(basis_type="Slater", basis_shell_num=1, basis_prim_num=1,
                  basis_r_power=np.array([1]), basis_coefficient_im=np.array([.2]),
                  basis_coefficient=np.array([1.]), basis_exponent=np.array([.5]))
    data = read_trexio(write_trexio_fields(tmp_path / "slater", fields, backend=backend))
    with pytest.raises(ValueError, match="Gaussian"):
        data.basis_set()
    for key, expected in fields.items():
        np.testing.assert_array_equal(data.fields[key], expected)


def test_failed_write_preserves_existing(water, backend, tmp_path, monkeypatch):
    mol, basis, result = water
    path = write_trexio(tmp_path / "safe", mol, basis, result, backend=backend)
    before = {p.name: p.read_bytes() for p in path.iterdir()} if path.is_dir() else path.read_bytes()
    original = trexio.write_mo_coefficient
    def fail(handle, values):
        original(handle, values)
        raise RuntimeError("injected after partial write")
    monkeypatch.setattr(trexio, "write_mo_coefficient", fail)
    with pytest.raises(ValueError, match="injected"):
        write_trexio(path, mol, basis, result, backend=backend)
    after = {p.name: p.read_bytes() for p in path.iterdir()} if path.is_dir() else path.read_bytes()
    assert before == after
    assert not list(tmp_path.glob(".safe.*"))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("valuable text")
    with pytest.raises(FileExistsError):
        write_trexio(path=foreign, molecule=mol, basis=basis, result=result, backend="text")
    assert (foreign / "notes.txt").read_text() == "valuable text"


@pytest.mark.parametrize("method", ["casci", "casscf", "fci"])
def test_correlated_runner_exports_exact_determinants(method, backend, tmp_path):
    from vibeqc.solvers import build_hamiltonian_matrix_unrestricted
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.8])])
    basis = BasisSet(mol, "sto-3g")
    result = vibeqc.run_job(mol, basis="sto-3g", method=method, output=tmp_path / "ci",
                            trexio=True, trexio_backend=backend, verbose=False, progress=False)
    path = tmp_path / ("ci.trexio.h5" if backend == "hdf5" else "ci.trexio")
    data = read_trexio(path)
    assert data.mo_type == "CI"
    assert "mo_energy" not in data.fields
    assert "determinant_list" in data.fields
    assert data.energy == pytest.approx(result.energy, abs=1e-10)
    C = data.mo_blocks()[0].coefficients
    rebuilt = data.basis_set()
    h = C.T @ (vibeqc.compute_kinetic(rebuilt) + vibeqc.compute_nuclear(rebuilt, data.molecule())) @ C
    eri = np.asarray(vibeqc.compute_eri(rebuilt)).reshape((2,) * 4)
    g = np.einsum('ap,bq,cr,ds,abcd->prqs', C, C, C, C, eri)
    determinants = []
    for bitfield in data.fields["determinant_list"]:
        determinants.append(tuple(tuple(i for i in range(2) if int(bits) & (1 << i)) for bits in bitfield))
    H = build_hamiltonian_matrix_unrestricted(determinants, h, g)
    vector = data.fields["determinant_coefficient"]
    assert vector @ H @ vector + data.nuclear_repulsion == pytest.approx(data.energy, abs=1e-10)
    spin_source = data.restart_result()
    assert np.trace(spin_source.density_alpha @ vibeqc.compute_overlap(rebuilt)) == pytest.approx(1., abs=1e-10)
    assert np.trace(spin_source.density_beta @ vibeqc.compute_overlap(rebuilt)) == pytest.approx(1., abs=1e-10)
    # Valid files may carry only the two spin RDMs, without total RDM or
    # MO occupations. They still define both restart densities exactly.
    fields = dict(data.fields)
    del fields["rdm_1e"], fields["mo_occupation"]
    spin_only = read_trexio(write_trexio_fields(tmp_path / "spin-only", fields, backend=backend))
    np.testing.assert_allclose(spin_only.restart_result().density_alpha,
                               spin_source.density_alpha, atol=1e-12)


def _pyscf_verdict(path, tmp_path, *, tolerance=1e-8):
    import json
    import subprocess

    from tests.trexio_reference import reference_python

    # Skips naming the gate that did not run, or fails when
    # VIBEQC_REQUIRE_TREXIO_REFERENCE is set (#253).
    executable = reference_python()
    script = Path(__file__).resolve().parents[1] / "examples/regression/runner_trexio_pyscf.py"
    target = tmp_path / "reference.json"
    proc = subprocess.run([executable, str(script), str(path), "--json", str(target),
                           "--tol-energy", str(tolerance)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    verdict = json.loads(target.read_text())
    assert verdict["verdict"] == "pass", verdict
    return verdict


def test_independent_ecp_reference(nah, backend, tmp_path):
    path = write_trexio(tmp_path / "nah", *nah, backend=backend)
    # libecpint and PySCF/libcint use different ECP quadratures. Existing
    # cross-library ECP checks use microhartree tolerances (test_ecp_correlated).
    # test_ecp_hamiltonian_and_seeded_calculation separately checks that
    # serialization preserves the native Hamiltonian to 2e-11 hartree.
    verdict = _pyscf_verdict(path, tmp_path, tolerance=1e-6)
    assert verdict["ecp_core_electrons"] == 10
    assert verdict["energy_diff"] < 1e-6
    assert verdict["hcore_max_diff"] < 1e-6


def test_independent_casscf_frozen_core_reference(backend, tmp_path):
    mol = Molecule([Atom(3, [0., 0., 0.]), Atom(1, [0., 0., 3.])])
    result = vibeqc.run_job(mol, basis="sto-3g", method="casscf", active_space=(2, 2),
                            output=tmp_path / "lih", trexio=True, trexio_backend=backend,
                            verbose=False, progress=False)
    path = tmp_path / ("lih.trexio.h5" if backend == "hdf5" else "lih.trexio")
    data = read_trexio(path)
    assert data.mo_class.count("Core") == 1
    assert data.mo_class.count("Active") == 2
    assert all(int(bits) & 1 for det in data.fields["determinant_list"] for bits in det)
    assert data.energy == pytest.approx(result.energy, abs=1e-10)
    verdict = _pyscf_verdict(path, tmp_path)
    assert verdict["wavefunction"] == "CI"
    assert verdict["energy_diff"] < 1e-8


def test_retained_ci_result_selects_the_requested_root(backend, tmp_path):
    from vibeqc.solvers import CASCIOptions
    mol = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.8])])
    basis = BasisSet(mol, "sto-3g")
    result = vibeqc.run_job(mol, basis="sto-3g", method="casci",
                            casci_options=CASCIOptions(nroots=2), output=tmp_path / "roots",
                            verbose=False, progress=False)
    assert abs(result.root_energies[1] - result.root_energies[0]) > .01
    path = write_trexio(tmp_path / "excited", mol, basis, result, root=1, backend=backend)
    data = read_trexio(path)
    assert data.fields["state_num"] == 2 and data.fields["state_id"] == 1
    assert data.energy == pytest.approx(result.root_energies[1], abs=1e-12)
    assert _pyscf_verdict(path, tmp_path)["energy_diff"] < 1e-10
    with pytest.raises(ValueError, match="outside"):
        write_trexio(path, mol, basis, result, root=2, backend=backend)


def test_xml_ecp_export(nah, tmp_path):
    from vibeqc.ecp_metadata import one_electron_hamiltonian
    mol, basis, _ = nah
    options = vibeqc.RHFOptions()
    options.ecp_centers = [vibeqc.ECPCenter(11, [0., 0., 0.])]
    options.ecp_library = "lanl2dz"
    result = vibeqc.run_rhf(mol, basis, options)
    assert result.ecp_xml_centers
    data = read_trexio(write_trexio(tmp_path / "xml.h5", mol, basis, result))
    H, _, _, _ = one_electron_hamiltonian(data.molecule(), data.basis_set(), data.ecp_options())
    np.testing.assert_allclose(H, data.to_libint_order(data.core_hamiltonian), atol=2e-11)
    assert min(data.fields["ecp_power"]) == -2


def test_zero_core_ecp_is_kept(tmp_path):
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "sto-3g")
    block = vibeqc.ECPPrimitiveBlock()
    block.n_primitive, block.exponents, block.coefficients, block.ams, block.ns = 1, [1.], [.05], [0], [2]
    options = vibeqc.RHFOptions()
    options.ecp_primitive_blocks = [block]
    options.ecp_primitive_centers = [[0., 0., 0.]]
    options.ecp_effective_charges = [1., 1.]
    options.ecp_total_ncore = 0
    result = vibeqc.run_rhf(mol, basis, options)
    data = read_trexio(write_trexio(tmp_path / "zero-core.h5", mol, basis, result))
    assert data.fields["ecp_num"] == 1
    np.testing.assert_array_equal(data.fields["ecp_z_core"], [0, 0])
    assert np.max(np.abs(data.fields["ao_1e_int_ecp"])) > 1e-4


def test_periodic_runner_trexio_manifest_and_citation(backend, tmp_path):
    import tomllib
    system = vibeqc.PeriodicSystem(3, np.eye(3) * 7., [Atom(2, [0., 0., 0.])])
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    result = vibeqc.run_periodic_job(
        system, basis, method="RHF", jk_method="gdf", kpoints=(2, 1, 1),
        aux_basis="def2-svp-jk", gdf_method="rsgdf", rsgdf_ke_cutoff=12.,
        rsgdf_tail_ke_cutoff=0., convergence="off",
        max_iter=60, conv_tol_energy=1e-8, output=tmp_path / "periodic",
        trexio=True, trexio_backend=backend, progress=False, verbose=0,
        write_molden_file=False, write_population_file=False,
    )
    path = tmp_path / ("periodic.trexio.h5" if backend == "hdf5" else "periodic.trexio")
    data = read_trexio(path)
    assert data.periodic
    assert len(data.mo_blocks()) == 2
    assert data.energy == pytest.approx(result.energy, abs=1e-11)
    for block, expected in zip(data.mo_blocks(), result.mo_coeffs):
        np.testing.assert_allclose(block.coefficients, expected, atol=1e-12)
    assert "TREXIO" in (tmp_path / "periodic.references").read_text()
    manifest = tomllib.loads((tmp_path / "periodic.system").read_text())
    assert any(row["format"] == "trexio" for row in manifest["plan"]["files"])
