"""Finite-torus occupied-localization invariants for AICCM2026DEV-B."""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic.chi.localization as localization

from vibeqc.periodic.chi.scf import AICCM2026DevBExperimentalWarning
from vibeqc.periodic.chi.localization import (
    AICCM2026DevBLocalizationWarning,
    _ao_metadata,
    _canonical_wannier_coefficients,
    _canonical_wannier_coefficients_from_k,
    _centers_spreads_aliasing,
    _localization_projector_audit,
    _localization_projector_audit_python,
    _real_space_matrix,
    _real_space_matrix_from_k,
    localize_aiccm2026dev_b_occupied_blocks,
)


def _synthetic_blocks(n_cells: int = 4):
    kfrac = np.zeros((n_cells, 3))
    kfrac[:, 0] = np.arange(n_cells) / n_cells
    # A deliberately rough Bloch gauge.  It produces extended canonical
    # Wannier columns without changing the one-dimensional occupied subspace
    # at any k.
    indices = np.arange(n_cells, dtype=float)
    phases = np.exp(1j * (0.37 * indices**2 + 0.11 * indices))
    coefficients = []
    overlaps = []
    focks = []
    energies = []
    for phase in phases:
        coefficients.append(np.asarray([[phase, 0.0], [0.0, 1.0]], dtype=complex))
        overlaps.append(np.eye(2))
        focks.append(np.diag([-1.0, 0.5]))
        energies.append(np.asarray([-1.0, 0.5]))
    ao_fractional = np.asarray(
        [[cell + 0.20, 0.0, 0.0] for cell in range(n_cells) for _ in range(2)]
    )
    ao_fractional[1::2, 0] += 0.35
    ao_atoms = np.arange(2 * n_cells, dtype=int)
    return {
        "coefficients_k": coefficients,
        "overlap_k": overlaps,
        "kpoints_fractional": kfrac,
        "mesh": (n_cells, 1, 1),
        "n_occ_per_cell": 1,
        "ao_fractional": ao_fractional,
        "ao_atom_indices": ao_atoms,
        "lattice_bohr": np.diag([2.0, 10.0, 10.0]),
        "dimension": 1,
        "fock_k": focks,
        "orbital_energies_k": energies,
    }


def test_native_localization_matrix_transform_matches_einsum_oracle() -> None:
    rng = np.random.default_rng(20260628)
    kfrac = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    translations = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    phases = np.exp(2j * np.pi * (kfrac @ translations.T))
    matrices = []
    for _ in range(2):
        raw = rng.normal(size=(3, 3))
        matrices.append(raw + raw.T)

    got = _real_space_matrix_from_k(
        matrices,
        kfrac,
        translations,
        label="test",
    )
    expected = _real_space_matrix(matrices, phases)

    np.testing.assert_allclose(got, expected.real, atol=1e-14)


def test_native_localization_matrix_transform_rejects_complex_residue() -> None:
    kfrac = np.asarray([[0.0, 0.0, 0.0]])
    translations = np.asarray([[0, 0, 0]], dtype=np.int64)
    matrix = np.asarray([[[1.0, 1.0j], [-1.0j, 2.0]]], dtype=complex)

    with pytest.raises(RuntimeError, match="imaginary residual"):
        _real_space_matrix_from_k(matrix, kfrac, translations, label="test")


def test_native_canonical_wannier_transform_matches_einsum_oracle() -> None:
    rng = np.random.default_rng(20260629)
    kfrac = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    translations = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    phases = np.exp(2j * np.pi * (kfrac @ translations.T))
    coefficients = rng.normal(size=(2, 3, 4)) + 1j * rng.normal(size=(2, 3, 4))

    got = _canonical_wannier_coefficients_from_k(
        coefficients,
        kfrac,
        translations,
        n_occ=2,
    )
    expected = _canonical_wannier_coefficients(coefficients, phases, n_occ=2)

    np.testing.assert_allclose(got, expected, atol=1e-14)


def test_native_localization_projector_audit_matches_dense_complex_oracle() -> None:
    rng = np.random.default_rng(20260801)
    mesh = (2, 1, 2)
    translations = np.asarray(
        [[x, y, z] for x in range(2) for y in range(1) for z in range(2)],
        dtype=np.int64,
    )
    nbf = 3
    n_ao = len(translations) * nbf
    n_occ = 5
    canonical = rng.normal(size=(n_ao, n_occ)) + 1j * rng.normal(
        size=(n_ao, n_occ)
    )
    raw_unitary = rng.normal(size=(n_occ, n_occ)) + 1j * rng.normal(
        size=(n_occ, n_occ)
    )
    unitary, _ = np.linalg.qr(raw_unitary)
    localized = canonical @ unitary
    localized[3, 2] += 0.031 - 0.017j
    raw_fock = rng.normal(size=(n_ao, n_ao)) + 1j * rng.normal(
        size=(n_ao, n_ao)
    )
    fock = raw_fock + raw_fock.conj().T

    expected = _localization_projector_audit_python(
        canonical,
        localized,
        fock,
        translations,
        mesh,
        nbf,
    )
    got = _localization_projector_audit(
        canonical,
        localized,
        fock,
        translations,
        mesh,
        nbf,
    )

    assert all(value > 1.0e-8 for value in expected)
    np.testing.assert_allclose(got, expected, rtol=2.0e-11, atol=2.0e-12)


def test_native_projector_audit_resolves_nonzero_cancellation_fallback() -> None:
    mesh = (2, 1, 1)
    translations = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    canonical = np.zeros((4, 2), dtype=complex)
    canonical[0, 0] = 1.0
    canonical[2, 1] = 1.0
    phase = np.exp(0.37j)
    unitary = np.asarray(
        [
            [np.cos(0.41), -phase.conjugate() * np.sin(0.41)],
            [phase * np.sin(0.41), np.cos(0.41)],
        ]
    )
    localized = canonical @ unitary
    localized[1, 0] += 1.0e-7 * (1.0 + 0.4j)

    expected = _localization_projector_audit_python(
        canonical,
        localized,
        None,
        translations,
        mesh,
        2,
    )
    got = _localization_projector_audit(
        canonical,
        localized,
        None,
        translations,
        mesh,
        2,
    )

    assert expected[0] > 1.0e-9
    assert expected[2] > 1.0e-9
    assert got[0] > 1.0e-9
    assert got[2] > 1.0e-9
    np.testing.assert_allclose(got, expected, rtol=1.0e-12, atol=1.0e-15)


def test_projector_audit_preserves_real_fock_dtype(monkeypatch) -> None:
    seen_dtypes: list[np.dtype] = []
    native = localization.aiccm2026dev_b_localization_projector_audit

    def dtype_spy(canonical, localized, fock, translations, mesh, nbf):
        seen_dtypes.append(np.asarray(fock).dtype)
        return native(canonical, localized, fock, translations, mesh, nbf)

    monkeypatch.setattr(
        localization,
        "aiccm2026dev_b_localization_projector_audit",
        dtype_spy,
    )
    canonical = np.asarray([[1.0], [0.0]], dtype=complex)
    localized = np.asarray([[1.0], [0.2j]], dtype=complex)
    fock = np.asarray([[1.2, 0.3], [0.3, -0.7]], dtype=np.float64)
    expected = _localization_projector_audit_python(
        canonical,
        localized,
        fock,
        np.asarray([[0, 0, 0]]),
        (1, 1, 1),
        2,
    )

    got = _localization_projector_audit(
        canonical,
        localized,
        fock,
        np.asarray([[0, 0, 0]]),
        (1, 1, 1),
        2,
    )

    assert seen_dtypes == [np.dtype(np.float64)]
    np.testing.assert_allclose(got, expected, rtol=1.0e-12, atol=1.0e-14)


def test_native_localization_projector_audit_validates_inputs() -> None:
    canonical = np.ones((4, 2), dtype=complex)
    localized = np.array(canonical, copy=True)
    translations = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)

    with pytest.raises(RuntimeError, match="same shape"):
        _localization_projector_audit(
            canonical,
            localized[:-1],
            None,
            translations,
            (2, 1, 1),
            2,
        )
    with pytest.raises(RuntimeError, match="Fock matrix has the wrong AO shape"):
        _localization_projector_audit(
            canonical,
            localized,
            np.eye(3),
            translations,
            (2, 1, 1),
            2,
        )
    with pytest.raises(RuntimeError, match="duplicate residue"):
        _localization_projector_audit(
            canonical,
            localized,
            None,
            np.asarray([[0, 0, 0], [0, 0, 0]]),
            (2, 1, 1),
            2,
        )
    nonfinite = np.array(canonical, copy=True)
    nonfinite[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="coefficients must be finite"):
        _localization_projector_audit(
            nonfinite,
            localized,
            None,
            translations,
            (2, 1, 1),
            2,
        )


def test_localization_production_uses_native_projector_audit(monkeypatch) -> None:
    calls = 0
    native = localization._localization_projector_audit

    def native_spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return native(*args, **kwargs)

    def dense_oracle_must_not_run(*_args, **_kwargs):
        raise AssertionError("production localization called the dense oracle")

    monkeypatch.setattr(localization, "_localization_projector_audit", native_spy)
    monkeypatch.setattr(
        localization,
        "_localization_projector_audit_python",
        dense_oracle_must_not_run,
    )

    result = localize_aiccm2026dev_b_occupied_blocks(**_synthetic_blocks())

    assert calls == 1
    assert result.density_invariance_error < 1.0e-12
    assert result.one_particle_energy_error < 1.0e-12
    assert result.translation_projector_error < 1.0e-12


def test_skew_lattice_localization_metadata_uses_column_vectors() -> None:
    lattice = np.asarray(
        [
            [7.0, 0.4, 0.2],
            [0.3, 8.0, 0.5],
            [0.1, 0.6, 9.0],
        ]
    )
    fractional = np.asarray([0.17, 0.23, 0.31])
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(2, (lattice @ fractional).tolist())],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    ao_fractional, atom_indices = _ao_metadata(
        basis,
        system,
        np.asarray([[0, 0, 0]], dtype=int),
    )

    np.testing.assert_allclose(ao_fractional, [fractional], atol=1.0e-14)
    np.testing.assert_array_equal(atom_indices, [0])

    centers, centers_bohr, spreads, alias, detected = _centers_spreads_aliasing(
        np.ones((1, 1)),
        np.ones((1, 1)),
        ao_fractional,
        lattice,
        (2, 3, 4),
        3,
        0.1,
    )
    np.testing.assert_allclose(centers, [fractional], atol=1.0e-14)
    np.testing.assert_allclose(centers_bohr, [(lattice @ fractional)], atol=1.0e-14)
    np.testing.assert_allclose(spreads, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(alias, 0.0, atol=1.0e-14)
    assert not detected[0]


def test_wannier_rotation_preserves_projector_and_one_particle_energy() -> None:
    result = localize_aiccm2026dev_b_occupied_blocks(**_synthetic_blocks())

    assert result.method == "wannier"
    assert result.coefficients.shape == (8, 4)
    assert result.objective_final >= result.objective_initial - 1e-12
    assert result.density_invariance_error < 1e-12
    assert result.one_particle_energy_error < 1e-12
    assert result.orthonormality_error < 1e-12
    assert result.unitary_error < 1e-12
    assert result.translation_projector_error < 1e-12
    assert result.translation_covariance_error < 1e-10
    assert result.real_gauge_residual < 1e-6
    assert result.band_gap_hartree == pytest.approx(1.5)
    np.testing.assert_allclose(result.atomic_populations.sum(axis=1), 1.0, atol=1e-12)


def test_wannier_centres_form_translated_family_without_wrap_alias() -> None:
    result = localize_aiccm2026dev_b_occupied_blocks(**_synthetic_blocks())

    centers = np.sort(result.centers_fractional[:, 0])
    # One Wannier orbital per cyclic translation, all on the first AO family.
    np.testing.assert_allclose(centers, [0.2, 1.2, 2.2, 3.2], atol=3e-2)
    assert np.max(result.spreads_bohr2) < 1e-2
    assert not np.any(result.aliasing_detected)


def test_iao_population_gauge_has_the_same_hard_invariants() -> None:
    data = _synthetic_blocks()
    # In this synthetic minimal-basis case the target and IAO reference spaces
    # coincide.  These are the exact finite-torus metric/cross-overlap blocks.
    identity = np.eye(8)
    data.update(
        method="iao",
        iao_target_reference_overlap=identity,
        iao_reference_overlap=identity,
        iao_reference_atom_indices=np.arange(8, dtype=int),
        iao_reference_basis="synthetic-minimal",
    )
    result = localize_aiccm2026dev_b_occupied_blocks(**data)

    assert result.method == "iao"
    assert result.iao_reference_basis == "synthetic-minimal"
    assert result.iao_orthonormality_error is not None
    assert result.iao_orthonormality_error < 1e-12
    assert result.objective_final >= result.objective_initial - 1e-12
    assert result.density_invariance_error < 1e-12
    assert result.one_particle_energy_error < 1e-12
    assert result.orthonormality_error < 1e-12


def test_rejects_metallic_or_entangled_occupied_manifold() -> None:
    data = _synthetic_blocks()
    data["orbital_energies_k"] = [np.asarray([-1.0, -1.0])] * 4
    with pytest.raises(ValueError, match="gapped occupied manifold"):
        localize_aiccm2026dev_b_occupied_blocks(**data)


def test_one_cell_cluster_flags_unresolvable_wraparound() -> None:
    data = _synthetic_blocks(n_cells=1)
    with pytest.warns(AICCM2026DevBLocalizationWarning, match="antipode"):
        result = localize_aiccm2026dev_b_occupied_blocks(**data)
    assert result.aliasing_detected[0]
    assert result.aliasing_fraction[0] == pytest.approx(1.0)


def test_incomplete_iao_reference_fails_closed() -> None:
    data = _synthetic_blocks()
    data.update(
        method="iao",
        iao_target_reference_overlap=np.ones((8, 1)),
        iao_reference_overlap=np.ones((1, 1)),
        iao_reference_atom_indices=np.zeros(1, dtype=int),
    )
    with pytest.raises(RuntimeError, match="does not span"):
        localize_aiccm2026dev_b_occupied_blocks(**data)


def test_public_wrapper_localizes_a_vacuum_padded_b_stream_reference() -> None:
    system = vq.PeriodicSystem(
        3,
        np.diag([5.0, 20.0, 20.0]),
        [vq.Atom(1, [1.8, 10.0, 10.0]), vq.Atom(1, [3.2, 10.0, 10.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.max_iter = 80
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        scf = vq.run_aiccm2026dev_b_rhf(
            system,
            basis,
            (4, 1, 1),
            options,
            backend="ri",
            progress=False,
        )
    with pytest.warns(AICCM2026DevBLocalizationWarning, match="antipode"):
        localized = vq.localize_aiccm2026dev_b_occupied(
            scf, system, basis, method="wannier"
        )

    assert localized.density_invariance_error < 1e-12
    assert localized.one_particle_energy_error < 1e-12
    assert localized.translation_covariance_error < 1e-10
    assert localized.real_gauge_residual < 1e-6
    np.testing.assert_allclose(
        np.sort(localized.centers_bohr[:, 0]),
        [2.5, 7.5, 12.5, 17.5],
        atol=1e-5,
    )


@pytest.mark.slow
def test_real_chain_spread_stabilizes_as_the_cyclic_cluster_grows() -> None:
    system = vq.PeriodicSystem(
        3,
        np.diag([5.0, 20.0, 20.0]),
        [vq.Atom(1, [1.8, 10.0, 10.0]), vq.Atom(1, [3.2, 10.0, 10.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.max_iter = 80
    spreads = {}
    for count in (2, 4, 6):
        with pytest.warns(AICCM2026DevBExperimentalWarning):
            scf = vq.run_aiccm2026dev_b_rhf(
                system,
                basis,
                (count, 1, 1),
                options,
                backend="ri",
                progress=False,
            )
        with pytest.warns(AICCM2026DevBLocalizationWarning, match="antipode"):
            localized = vq.localize_aiccm2026dev_b_occupied(scf, system, basis)
        spreads[count] = float(np.mean(localized.spreads_bohr2))

    # Real-run values on this pinned setup are approximately 0.5067, 0.5597,
    # and 0.5650 bohr^2.  The comparison checks stabilization without turning
    # those platform-sensitive descriptors into claimed reference constants.
    assert abs(spreads[4] - spreads[6]) < abs(spreads[2] - spreads[6])
    assert abs(spreads[4] - spreads[6]) < 0.01
