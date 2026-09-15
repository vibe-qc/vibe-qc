"""Post-HF invariants for the independent AICCM2026DEV-B stream."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq

from vibeqc.periodic.chi.scf import (
    AICCM2026DevBExperimentalWarning,
    AICCM2026DevBFiniteTorusConvention,
)
from vibeqc.periodic.chi.posthf import (
    AICCM2026DevBDLPNOCCSDResult,
    AICCM2026DevBDLPNOMP2Result,
    AICCM2026DevBMP2Result,
    AICCM2026DevBUCCSDResult,
    AICCM2026DevBUMP2Result,
    _build_supercell_system,
    _build_canonical_lpq_cache,
    _build_canonical_lov_cache,
    _BRealDFAdapter,
    _cyclic_translations,
    _lpq_cache_to_real_home_auxiliary,
    _lpq_cache_to_real_supercell,
    _reference_finite_torus_convention,
    _validate_posthf_numerical_support_cutoffs,
)


@pytest.mark.parametrize(
    "result_type",
    [
        AICCM2026DevBMP2Result,
        AICCM2026DevBUMP2Result,
        AICCM2026DevBUCCSDResult,
        AICCM2026DevBDLPNOMP2Result,
        AICCM2026DevBDLPNOCCSDResult,
    ],
)
def test_posthf_results_delegate_chi_ccm_construction_identity(
    result_type: type,
) -> None:
    convention = AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0",
        exchange_q0="bvk-ewald",
        exchange_q0_applicability="active",
        boundary_model="3d-periodic",
        periodic_dimension=3,
        character_mesh_shape=(1, 1, 1),
        bvk_madelung_supercell_repetitions=(1, 1, 1),
        bvk_madelung_supercell_lattice_bohr=(
            (8.0, 0.0, 0.0),
            (0.0, 8.0, 0.0),
            (0.0, 0.0, 8.0),
        ),
    )
    result = object.__new__(result_type)
    object.__setattr__(result, "finite_torus_convention", convention)

    assert result.ccm_approach == "chi-ccm"
    assert result.ccm_construction == "finite-translation-group-character"
    assert result.evaluation_representation == "gamma-centred-character-mesh"


def test_skew_real_supercell_uses_column_lattice_vectors() -> None:
    lattice = np.asarray(
        [
            [7.0, 0.4, 0.2],
            [0.3, 8.0, 0.5],
            [0.1, 0.6, 9.0],
        ]
    )
    origin = np.asarray([0.2, 0.3, 0.4])
    mesh = (2, 3, 1)
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(2, origin.tolist())],
    )

    supercell = _build_supercell_system(system, mesh)

    np.testing.assert_allclose(
        np.asarray(supercell.lattice),
        lattice @ np.diag(mesh),
    )
    expected_positions = np.asarray(
        [
            origin + lattice @ np.asarray([i, j, 0], dtype=float)
            for i in range(mesh[0])
            for j in range(mesh[1])
        ]
    )
    actual_positions = np.asarray([atom.xyz for atom in supercell.unit_cell])
    np.testing.assert_allclose(actual_positions, expected_positions)


def test_native_b_stream_posthf_kernels_match_numpy_reference() -> None:
    from vibeqc._vibeqc_core import (
        aiccm2026dev_b_3index_mo_transform,
        aiccm2026dev_b_3index_mo_transform_complex,
        aiccm2026dev_b_lpq_to_real_supercell,
        aiccm2026dev_b_matrix_to_real_supercell,
    )

    rng = np.random.default_rng(20260623)
    kfrac = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    translations = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    phases = np.exp(2j * np.pi * (kfrac @ translations.T))

    matrices = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
    got_matrix, got_imag = aiccm2026dev_b_matrix_to_real_supercell(
        np.ascontiguousarray(matrices),
        kfrac,
        translations,
    )
    ref_blocks = np.einsum(
        "kr,kmn,ks->rmsn",
        phases,
        matrices,
        phases.conj(),
        optimize=True,
    ) / 2
    ref_matrix_complex = ref_blocks.reshape(6, 6)
    ref_imag = float(np.max(np.abs(ref_matrix_complex.imag)))
    ref_matrix = 0.5 * (ref_matrix_complex + ref_matrix_complex.conj().T)
    np.testing.assert_allclose(got_matrix, ref_matrix.real, atol=2e-14)
    assert got_imag == pytest.approx(ref_imag, abs=2e-14)

    lpq_cache = {
        (ki, ka): np.ascontiguousarray(
            rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
        )
        for ki in range(2)
        for ka in range(2)
    }
    got_factors, got_cderi_imag, got_cderi_sym = aiccm2026dev_b_lpq_to_real_supercell(
        lpq_cache,
        kfrac,
        translations,
    )
    ref_factors = np.zeros((2, 2, 2, 3, 2, 3), dtype=complex)
    for t_index, translation in enumerate(translations):
        for ki in range(2):
            for ka in range(2):
                auxiliary_phase = np.exp(
                    2j * np.pi * ((kfrac[ka] - kfrac[ki]) @ translation)
                )
                cell_phase = (
                    phases[ki, :, None] * phases[ka, None, :].conj()
                    * auxiliary_phase
                )
                ref_factors[t_index] += 0.25 * np.einsum(
                    "rs,Pmn->Prmsn",
                    cell_phase,
                    lpq_cache[(ki, ka)],
                    optimize=True,
                )
    ref_factors = ref_factors.reshape(4, 6, 6)
    ref_cderi_imag = float(np.max(np.abs(ref_factors.imag)))
    ref_cderi_sym = float(np.max(np.abs(ref_factors - ref_factors.swapaxes(1, 2))))
    ref_factors = 0.5 * (ref_factors + ref_factors.swapaxes(1, 2))
    np.testing.assert_allclose(got_factors, ref_factors.real, atol=2e-14)
    assert got_cderi_imag == pytest.approx(ref_cderi_imag, abs=2e-14)
    assert got_cderi_sym == pytest.approx(ref_cderi_sym, abs=2e-14)

    factors = rng.normal(size=(4, 6, 6))
    c_left = rng.normal(size=(6, 2))
    c_right = rng.normal(size=(6, 3))
    np.testing.assert_allclose(
        aiccm2026dev_b_3index_mo_transform(factors, c_left, c_right),
        np.einsum("mi,Pmn,nj->Pij", c_left, factors, c_right, optimize=True),
        atol=2e-14,
    )

    z_factors = factors + 1j * rng.normal(size=(4, 6, 6))
    z_left = c_left + 1j * rng.normal(size=(6, 2))
    z_right = c_right + 1j * rng.normal(size=(6, 3))
    np.testing.assert_allclose(
        aiccm2026dev_b_3index_mo_transform_complex(z_factors, z_left, z_right),
        np.einsum(
            "mi,Pmn,nj->Pij",
            z_left.conj(),
            z_factors,
            z_right,
            optimize=True,
        ),
        atol=2e-14,
    )

def _dual_character_mesh(mesh: tuple[int, int, int]) -> np.ndarray:
    translations = _cyclic_translations(mesh)
    return translations / np.asarray(mesh, dtype=float)[None, :]


def _dense_factors_from_home(
    home_factors: np.ndarray,
    mesh: tuple[int, int, int],
) -> np.ndarray:
    """Expand the group-circulant factor as an independent test oracle."""

    translations = _cyclic_translations(mesh)
    lookup = {tuple(value): index for index, value in enumerate(translations)}
    n_cells = len(translations)
    n_aux, n_ao, _ = home_factors.shape
    nbf = n_ao // n_cells
    dense = np.empty((n_cells * n_aux, n_ao, n_ao), dtype=home_factors.dtype)
    moduli = np.asarray(mesh, dtype=int)
    for t_index, translation in enumerate(translations):
        for r_index, row_translation in enumerate(translations):
            home_row = lookup[tuple((row_translation - translation) % moduli)]
            row = slice(r_index * nbf, (r_index + 1) * nbf)
            source_row = slice(home_row * nbf, (home_row + 1) * nbf)
            for s_index, column_translation in enumerate(translations):
                home_column = lookup[
                    tuple((column_translation - translation) % moduli)
                ]
                column = slice(s_index * nbf, (s_index + 1) * nbf)
                source_column = slice(
                    home_column * nbf,
                    (home_column + 1) * nbf,
                )
                dense[
                    t_index * n_aux : (t_index + 1) * n_aux,
                    row,
                    column,
                ] = home_factors[:, source_row, source_column]
    return dense


@pytest.mark.parametrize("mesh", [(2, 1, 1), (3, 1, 1), (2, 3, 1)])
def test_home_auxiliary_inverse_transform_recovers_cyclic_factor(
    mesh: tuple[int, int, int],
) -> None:
    rng = np.random.default_rng(20260821 + int(np.prod(mesh)))
    translations = _cyclic_translations(mesh)
    kfrac = _dual_character_mesh(mesh)
    n_cells = len(translations)
    n_aux = 2
    nbf = 2
    n_ao = n_cells * nbf
    raw = rng.normal(size=(n_aux, n_ao, n_ao))
    home_reference = 0.5 * (raw + raw.swapaxes(1, 2))
    home_blocks = home_reference.reshape(n_aux, n_cells, nbf, n_cells, nbf)
    phases = np.exp(2j * np.pi * (kfrac @ translations.T))
    lpq_cache = {
        (ki, ka): np.ascontiguousarray(
            np.einsum(
                "r,s,Prmsn->Pmn",
                phases[ki].conj(),
                phases[ka],
                home_blocks,
                optimize=True,
            )
        )
        for ki in range(n_cells)
        for ka in range(n_cells)
    }

    home, imaginary, symmetry = _lpq_cache_to_real_home_auxiliary(
        lpq_cache,
        kfrac,
        mesh,
    )
    dense = _dense_factors_from_home(home, mesh)
    dense_native, _, _ = _lpq_cache_to_real_supercell(lpq_cache, kfrac, mesh)

    assert home.shape == (n_aux, n_ao, n_ao)
    assert home.dtype == np.float64
    assert home.flags.c_contiguous
    np.testing.assert_allclose(home, home_reference, rtol=0.0, atol=2e-13)
    np.testing.assert_allclose(dense, dense_native, rtol=0.0, atol=2e-13)
    assert imaginary < 2e-13
    assert symmetry < 2e-13
    assert home.nbytes * n_cells == dense.nbytes


@pytest.mark.parametrize("mesh", [(2, 1, 1), (3, 1, 1), (2, 3, 1)])
def test_home_auxiliary_circulant_mo_transform_matches_dense_oracle(
    mesh: tuple[int, int, int],
) -> None:
    from vibeqc._vibeqc_core import (
        aiccm2026dev_b_home_auxiliary_3index_mo_transform_complex,
    )

    rng = np.random.default_rng(20260831 + int(np.prod(mesh)))
    n_cells = int(np.prod(mesh))
    n_aux = 2
    nbf = 2
    n_ao = n_cells * nbf
    raw = rng.normal(size=(n_aux, n_ao, n_ao))
    home = 0.5 * (raw + raw.swapaxes(1, 2))
    dense = _dense_factors_from_home(home, mesh)
    left = rng.normal(size=(n_ao, 3)) + 1j * rng.normal(size=(n_ao, 3))
    right = rng.normal(size=(n_ao, 2)) + 1j * rng.normal(size=(n_ao, 2))
    reference = np.einsum(
        "mi,Pmn,nj->Pij",
        left.conj(),
        dense,
        right,
        optimize=True,
    )

    transformed = aiccm2026dev_b_home_auxiliary_3index_mo_transform_complex(
        np.ascontiguousarray(home, dtype=np.complex128),
        np.ascontiguousarray(left),
        np.ascontiguousarray(right),
        np.ascontiguousarray(np.asarray(mesh, dtype=np.int64)),
    )
    adapter = _BRealDFAdapter(
        home,
        SimpleNamespace(nbasis=n_cells * n_aux),
        mesh,
    )

    assert transformed.shape == (n_cells * n_aux, 3, 2)
    assert float(np.max(np.abs(transformed.imag))) > 1e-8
    np.testing.assert_allclose(transformed, reference, rtol=0.0, atol=2e-13)
    np.testing.assert_allclose(
        adapter.mo_transform(left, right),
        reference,
        rtol=0.0,
        atol=2e-13,
    )
    no_conjugation = np.einsum(
        "mi,Pmn,nj->Pij",
        left,
        dense,
        right,
        optimize=True,
    )
    assert float(np.max(np.abs(transformed - no_conjugation))) > 1e-6
    assert adapter.home_factors.nbytes * n_cells == dense.nbytes
    assert adapter.n_aux == n_cells * n_aux
    assert adapter.n_orb == n_ao
    assert not hasattr(adapter, "B")
    assert not hasattr(adapter, "three_center")
    assert not hasattr(adapter, "metric")

    real_left = rng.normal(size=(n_ao, 2))
    real_reference = np.einsum(
        "mi,Pmn,nj->Pij",
        real_left,
        dense,
        real_left,
        optimize=True,
    )
    np.testing.assert_allclose(
        adapter.mo_transform(real_left),
        real_reference,
        rtol=0.0,
        atol=2e-13,
    )


def test_home_auxiliary_inverse_transform_rejects_nondual_group() -> None:
    from vibeqc._vibeqc_core import aiccm2026dev_b_lpq_to_real_home_auxiliary

    mesh = np.asarray((2, 1, 1), dtype=np.int64)
    translations = _cyclic_translations((2, 1, 1)).astype(np.int64)
    cache = {
        (ki, ka): np.ones((1, 1, 1), dtype=np.complex128)
        for ki in range(2)
        for ka in range(2)
    }
    with pytest.raises(RuntimeError, match="dual cyclic character"):
        aiccm2026dev_b_lpq_to_real_home_auxiliary(
            cache,
            np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
            translations,
            mesh,
        )
    with pytest.raises(RuntimeError, match="lexicographic cyclic translation"):
        aiccm2026dev_b_lpq_to_real_home_auxiliary(
            cache,
            _dual_character_mesh((2, 1, 1)),
            translations[::-1].copy(),
            mesh,
        )
    with pytest.raises(RuntimeError, match="distinct dual cyclic characters"):
        aiccm2026dev_b_lpq_to_real_home_auxiliary(
            cache,
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            translations,
            mesh,
        )
    with pytest.raises(RuntimeError, match="one complete character"):
        aiccm2026dev_b_lpq_to_real_home_auxiliary(
            cache,
            _dual_character_mesh((2, 1, 1)),
            translations[:1],
            mesh,
        )
    with pytest.raises(RuntimeError, match="positive cyclic mesh"):
        aiccm2026dev_b_lpq_to_real_home_auxiliary(
            cache,
            _dual_character_mesh((2, 1, 1)),
            translations,
            np.asarray((0, 1, 1), dtype=np.int64),
        )


@pytest.mark.parametrize("invalid_character", [np.nan, np.inf, -np.inf, 1.0e300])
def test_home_auxiliary_inverse_rejects_invalid_character(
    invalid_character: float,
) -> None:
    from vibeqc._vibeqc_core import aiccm2026dev_b_lpq_to_real_home_auxiliary

    cache = {(0, 0): np.ones((1, 1, 1), dtype=np.complex128)}
    with pytest.raises(RuntimeError, match="finite dual cyclic characters"):
        aiccm2026dev_b_lpq_to_real_home_auxiliary(
            cache,
            np.asarray([[invalid_character, 0.0, 0.0]]),
            np.zeros((1, 3), dtype=np.int64),
            np.ones(3, dtype=np.int64),
        )


def test_home_auxiliary_kernels_reject_mesh_product_overflow() -> None:
    from vibeqc._vibeqc_core import (
        aiccm2026dev_b_home_auxiliary_3index_mo_transform,
        aiccm2026dev_b_lpq_to_real_home_auxiliary,
    )

    huge_mesh = np.asarray((2**62, 4, 1), dtype=np.int64)
    cache = {(0, 0): np.ones((1, 1, 1), dtype=np.complex128)}
    with pytest.raises(RuntimeError, match="mesh product"):
        aiccm2026dev_b_lpq_to_real_home_auxiliary(
            cache,
            np.zeros((1, 3)),
            np.zeros((1, 3), dtype=np.int64),
            huge_mesh,
        )
    with pytest.raises(RuntimeError, match="mesh product"):
        aiccm2026dev_b_home_auxiliary_3index_mo_transform(
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.ones((1, 1)),
            huge_mesh,
        )


def test_canonical_mp2_lov_cache_streams_lpq(monkeypatch) -> None:
    import vibeqc.periodic.chi.posthf as posthf

    rng = np.random.default_rng(20260626)
    n_aux = 2
    nbf = 3
    n_occ = 1
    kcart = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    coefficients = [
        np.ascontiguousarray(np.eye(nbf, dtype=complex)),
        np.ascontiguousarray(
            np.array(
                [
                    [0.9, 0.1, 0.0],
                    [-0.1, 0.9, 0.2],
                    [0.0, -0.2, 0.9],
                ],
                dtype=complex,
            )
        ),
    ]
    lpq_by_key = {}
    for ki, bra_k in enumerate(kcart):
        for ka, ket_k in enumerate(kcart):
            lpq_by_key[(tuple(bra_k), tuple(ket_k))] = (
                rng.normal(size=(n_aux, nbf, nbf))
                + 1j * rng.normal(size=(n_aux, nbf, nbf))
                + (10 * ki + ka)
            )

    auxiliary = SimpleNamespace(nbasis=n_aux)
    molecule = object()
    system = SimpleNamespace(unit_cell_molecule=lambda: molecule)
    built: list[tuple[tuple[float, ...], tuple[float, ...], float, float, bool]] = []

    monkeypatch.setattr(posthf, "make_aux_basis_set", lambda *_args, **_kw: auxiliary)
    monkeypatch.setattr(posthf, "make_modrho_aux_basis", lambda aux, _mol: aux)

    def fake_build_lpq(
        _system,
        _basis,
        _auxiliary,
        bra_k,
        ket_k,
        *,
        ke_cutoff,
        lat_opts,
        linear_dep_thr,
        canonical_auxiliary_basis,
    ):
        bra_key = tuple(float(x) for x in np.asarray(bra_k))
        ket_key = tuple(float(x) for x in np.asarray(ket_k))
        built.append(
            (
                bra_key,
                ket_key,
                float(ke_cutoff),
                float(lat_opts.cutoff_bohr),
                bool(canonical_auxiliary_basis),
            )
        )
        assert float(linear_dep_thr) == pytest.approx(1e-8)
        return np.array(lpq_by_key[(bra_key, ket_key)], copy=True)

    monkeypatch.setattr(posthf, "build_lpq_bloch_native_fft", fake_build_lpq)
    lov, got_auxiliary = _build_canonical_lov_cache(
        system,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        kcart,
        coefficients,
        n_occ,
        "aux",
        lattice_cutoff_bohr=23.0,
        rsgdf_ke_cutoff=321.0,
        gdf_linear_dep_threshold=1e-8,
    )
    assert got_auxiliary is auxiliary
    assert len(built) == 4
    assert all(item[2] == pytest.approx(321.0) for item in built)
    assert all(item[3] == pytest.approx(23.0) for item in built)
    assert all(item[4] for item in built)
    for ki in range(2):
        occupied = coefficients[ki][:, :n_occ]
        for ka in range(2):
            virtual = coefficients[ka][:, n_occ:]
            key = (tuple(kcart[ki]), tuple(kcart[ka]))
            expected = np.einsum(
                "mi,Pmn,nj->Pij",
                occupied.conj(),
                lpq_by_key[key],
                virtual,
                optimize=True,
            )
            np.testing.assert_allclose(lov[(ki, ka)], expected, atol=1e-13)

    with pytest.raises(ValueError, match="one coefficient block per character"):
        _build_canonical_lov_cache(
            system,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            kcart,
            coefficients[:1],
            n_occ,
            "aux",
            lattice_cutoff_bohr=23.0,
            rsgdf_ke_cutoff=321.0,
            gdf_linear_dep_threshold=1e-8,
        )


def test_posthf_reference_convention_distinguishes_exchange_q0_seams() -> None:
    convention = AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0",
        exchange_q0="bvk-ewald",
        exchange_q0_applicability="active",
        boundary_model="3d-periodic",
        periodic_dimension=3,
        character_mesh_shape=(2, 1, 1),
        bvk_madelung_supercell_repetitions=(2, 1, 1),
        bvk_madelung_supercell_lattice_bohr=(
            (16.0, 0.0, 0.0),
            (0.0, 8.0, 0.0),
            (0.0, 0.0, 8.0),
        ),
    )
    good_reference = SimpleNamespace(
        aiccm2026dev_b=SimpleNamespace(finite_torus_convention=convention)
    )
    assert _reference_finite_torus_convention(good_reference) == convention

    strict_zero_reference = SimpleNamespace(
        aiccm2026dev_b=SimpleNamespace(
            finite_torus_convention=replace(convention, exchange_q0="strict-zero")
        )
    )
    with pytest.raises(ValueError, match="exchange_q0"):
        _reference_finite_torus_convention(strict_zero_reference)

    inactive_reference = SimpleNamespace(
        aiccm2026dev_b=SimpleNamespace(
            finite_torus_convention=replace(
                convention,
                exchange_q0_applicability="inactive",
            )
        )
    )
    with pytest.raises(ValueError, match="exchange_q0_applicability"):
        _reference_finite_torus_convention(inactive_reference)

    ecp_reference = SimpleNamespace(
        aiccm2026dev_b=SimpleNamespace(
            finite_torus_convention=convention,
            ecp_total_ncore=2,
        )
    )
    with pytest.raises(NotImplementedError, match="periodic ECP references"):
        _reference_finite_torus_convention(ecp_reference)


def _h2_chain() -> tuple[vq.PeriodicSystem, vq.BasisSet]:
    system = vq.PeriodicSystem(
        3,
        np.diag([20.0, 20.0, 6.0]),
        [
            vq.Atom(1, [10.0, 10.0, 2.3]),
            vq.Atom(1, [10.0, 10.0, 3.7]),
        ],
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _options() -> vq.PeriodicRHFOptions:
    options = vq.PeriodicRHFOptions()
    options.max_iter = 60
    options.conv_tol_energy = 1e-9
    options.conv_tol_grad = 1e-7
    options.damping = 0.0
    return options


@pytest.mark.parametrize(
    "field",
    (
        "lattice_cutoff_bohr",
        "rsgdf_ke_cutoff",
        "gdf_linear_dep_threshold",
    ),
)
@pytest.mark.parametrize(
    "invalid",
    (
        True,
        False,
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        10**1000,
        1.0 + 0.0j,
        "1.0",
    ),
)
def test_posthf_numerical_support_cutoffs_require_positive_finite_reals(
    field: str,
    invalid: object,
) -> None:
    cutoffs: dict[str, object] = {
        "lattice_cutoff_bohr": 15.0,
        "rsgdf_ke_cutoff": 200.0,
        "gdf_linear_dep_threshold": 1.0e-9,
    }
    cutoffs[field] = invalid

    with pytest.raises(ValueError, match=field):
        _validate_posthf_numerical_support_cutoffs(**cutoffs)


@pytest.mark.parametrize(
    "cutoffs",
    (
        (15, 200, 1),
        (np.float32(15.0), np.float64(200.0), np.float64(1.0e-9)),
        (Fraction(15, 1), Fraction(200, 1), Fraction(1, 10**9)),
    ),
)
def test_posthf_numerical_support_cutoffs_preserve_valid_reals(
    cutoffs: tuple[object, object, object],
) -> None:
    _validate_posthf_numerical_support_cutoffs(
        lattice_cutoff_bohr=cutoffs[0],
        rsgdf_ke_cutoff=cutoffs[1],
        gdf_linear_dep_threshold=cutoffs[2],
    )


@pytest.mark.parametrize(
    "route",
    (
        "run_aiccm2026dev_b_mp2",
        "run_aiccm2026dev_b_dlpno_mp2",
        "run_aiccm2026dev_b_dlpno_ccsd",
        "run_aiccm2026dev_b_ump2",
        "run_aiccm2026dev_b_uccsd",
    ),
)
@pytest.mark.parametrize(
    "field",
    (
        "lattice_cutoff_bohr",
        "rsgdf_ke_cutoff",
        "gdf_linear_dep_threshold",
    ),
)
@pytest.mark.parametrize("invalid", (np.nan, np.inf))
def test_posthf_gateways_reject_nonfinite_cutoffs_before_electronic_work(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    field: str,
    invalid: float,
) -> None:
    import vibeqc.periodic.chi.posthf as posthf

    system, basis = _h2_chain()

    def unexpected_electronic_work(*args: object, **kwargs: object) -> None:
        pytest.fail("electronic reference work began before cutoff validation")

    monkeypatch.setattr(
        posthf,
        "run_aiccm2026dev_b_rhf",
        unexpected_electronic_work,
    )
    monkeypatch.setattr(
        posthf,
        "_prepare_b_real_reference",
        unexpected_electronic_work,
    )
    monkeypatch.setattr(
        posthf,
        "_prepare_b_real_ureference",
        unexpected_electronic_work,
    )

    with pytest.raises(ValueError, match=field):
        getattr(vq, route)(
            system,
            basis,
            (1, 1, 1),
            _options(),
            progress=False,
            **{field: invalid},
        )


def test_mp2_reference_inherits_quadratic_fallback_fail_close() -> None:
    system, basis = _h2_chain()
    options = _options()
    options.quadratic_fallback_iter = 3

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            NotImplementedError,
            match="quadratic SCF fallback is not implemented",
        ):
            vq.run_aiccm2026dev_b_mp2(
                system,
                basis,
                (2, 1, 1),
                options,
                progress=False,
            )


@pytest.mark.parametrize("mesh", ((2, 1, 1), (3, 1, 1)))
def test_character_factor_operator_matches_real_gamma_fold(mesh) -> None:
    """Character and real-Gamma RI factors frame one fitted torus operator.

    The even mesh guards the ``+/-1/2`` Nyquist time-reversal pair. The odd
    mesh contains raw pair differences ``+/-2/3`` that are equivalent to the
    finite-group transfers ``-/+1/3``. At finite cutoff, mishandling either
    case produces a real-torus imaginary residue and fitted-Gram mismatch.
    This is a representation control for the same neutral fitted Hamiltonian;
    it is not evidence that the sibling union-and-weight Γ-CCM construction
    equals the finite-character χ-CCM construction. Their declared common
    exchange-q=0 convention fixes the comparison, not that construction-level
    equality.
    """

    from vibeqc._vibeqc_core import LatticeSumOptions
    from vibeqc.periodic.ccm import CCMSystem
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_fold

    lattice = np.array(
        [[6.0, 0.3, 0.2], [0.0, 6.5, 0.4], [0.0, 0.0, 7.0]],
        dtype=float,
    ).T
    system = vq.PeriodicSystem(
        3,
        lattice,
        [
            vq.Atom(1, [0.2, 0.3, 0.4]),
            vq.Atom(1, [1.6, 0.3, 0.4]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kpoints = vq.cyclic_gamma_mesh(system, mesh)
    cache, _ = _build_canonical_lpq_cache(
        system,
        basis,
        np.asarray(kpoints.kpoints_cart),
        "def2-svp-jk",
        lattice_cutoff_bohr=15.0,
        rsgdf_ke_cutoff=20.0,
        gdf_linear_dep_threshold=1.0e-9,
    )
    character_factors, imaginary_residual, symmetry_residual = (
        _lpq_cache_to_real_supercell(
            cache,
            np.asarray(kpoints.kpoints_frac),
            mesh,
        )
    )

    lattice_options = LatticeSumOptions()
    lattice_options.cutoff_bohr = 15.0
    gamma_factors = ccm_neutral_cderi_fold(
        CCMSystem(system, mesh, "sto-3g"),
        ke_cutoff=20.0,
        aux_basis="def2-svp-jk",
        lat_opts=lattice_options,
    )
    gamma_flat = np.asarray(gamma_factors).reshape(gamma_factors.shape[0], -1)
    character_flat = np.asarray(character_factors).reshape(
        character_factors.shape[0], -1
    )
    gamma_gram = gamma_flat.T @ gamma_flat
    character_gram = character_flat.T @ character_flat
    scale = max(float(np.linalg.norm(gamma_gram)), 1.0)

    assert imaginary_residual < 2.0e-12
    assert symmetry_residual < 2.0e-12
    assert float(np.linalg.norm(character_gram - gamma_gram)) / scale < 2.0e-12


def _li_doublet() -> tuple[vq.PeriodicSystem, vq.BasisSet]:
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 15.0,
        [vq.Atom(3, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def test_restricted_local_defaults_pin_pre_sweep_periodic_convention(
    monkeypatch,
) -> None:
    """Molecular #140/#448 defaults cannot repin the periodic B stream."""
    import vibeqc.dlpno.ccsd_local_solver as local_cc
    import vibeqc.dlpno.mp2 as local_mp2
    import vibeqc.periodic.chi.posthf as posthf

    system, basis = _h2_chain()
    captured: dict[str, dict[str, object]] = {}

    class CapturingMP2Options(local_mp2.DLPNOMP2Options):
        def __init__(self, *args, **kwargs):
            captured["mp2"] = dict(kwargs)
            super().__init__(*args, **kwargs)

    class CapturingCCOptions(local_cc.LocalCCSDOptions):
        def __init__(self, *args, **kwargs):
            captured["ccsd_t"] = dict(kwargs)
            super().__init__(*args, **kwargs)

    class StopBeforeElectronicWork(Exception):
        pass

    def stop_before_electronic_work(*args, **kwargs):
        raise StopBeforeElectronicWork

    monkeypatch.setattr(local_mp2, "DLPNOMP2Options", CapturingMP2Options)
    monkeypatch.setattr(local_cc, "LocalCCSDOptions", CapturingCCOptions)
    monkeypatch.setattr(
        posthf,
        "_prepare_b_real_reference",
        stop_before_electronic_work,
    )

    with pytest.raises(StopBeforeElectronicWork):
        posthf.run_aiccm2026dev_b_dlpno_mp2(
            system,
            basis,
            (1, 1, 1),
            _options(),
            progress=False,
        )
    with pytest.raises(StopBeforeElectronicWork):
        posthf.run_aiccm2026dev_b_dlpno_ccsd_t(
            system,
            basis,
            (1, 1, 1),
            _options(),
            progress=False,
        )

    assert captured["mp2"] == {
        "localise": "pipek-mezey",
        "n_frozen": 0,
        "tcut_pno": 1e-8,
        "tcut_pno_weak": 1e-7,
        "tcut_mkn": 1e-3,
        "tcut_pairs": 0.0,
        "tcut_pairs_weak": 0.0,
        "pno_norm": "legacy",
    }
    assert captured["ccsd_t"] == {
        "localise": "pipek-mezey",
        "n_frozen": 0,
        "tcut_pno": 1e-7,
        "pno_norm": "legacy",
        "pno_correction": False,
        "tcut_mkn": 0.0,
        "tcut_pairs": 0.0,
        "coupling_radius": 0.0,
        "residual_domain": "pair",
        "compute_triples": True,
    }


def test_unrestricted_defaults_pin_pre_sweep_periodic_convention(
    monkeypatch,
) -> None:
    """Canonical and local unrestricted wrappers pin old periodic inputs."""
    import vibeqc.periodic.chi.posthf as posthf

    monkeypatch.setattr(
        posthf,
        "_run_aiccm2026dev_b_ump2",
        lambda *args, **kwargs: kwargs["ump2_options"],
    )
    ump2 = posthf.run_aiccm2026dev_b_ump2(None, None)
    local_ump2 = posthf.run_aiccm2026dev_b_dlpno_ump2(None, None)

    canonical_ucc = posthf._canonical_ucc_options(None, triples=True)
    monkeypatch.setattr(
        posthf,
        "_run_aiccm2026dev_b_uccsd",
        lambda *args, **kwargs: kwargs["cc_options"],
    )
    local_ucc = posthf._run_dlpno_ucc(
        None,
        None,
        None,
        None,
        cc_options=None,
        triples=False,
        aux_basis=None,
        lattice_cutoff_bohr=15.0,
        rsgdf_ke_cutoff=200.0,
        gdf_linear_dep_threshold=1e-9,
        progress=False,
        verbose=None,
    )

    assert (ump2.localise, ump2.n_frozen, ump2.tcut_pno, ump2.tcut_pairs) == (
        "none",
        0,
        0.0,
        0.0,
    )
    assert (
        local_ump2.localise,
        local_ump2.n_frozen,
        local_ump2.tcut_pno,
        local_ump2.tcut_pairs,
    ) == ("wannier", 0, 1e-8, 0.0)
    assert (
        canonical_ucc.localise,
        canonical_ucc.n_frozen,
        canonical_ucc.tcut_pno,
        canonical_ucc.compute_triples,
    ) == ("none", 0, 0.0, True)
    assert (
        local_ucc.localise,
        local_ucc.n_frozen,
        local_ucc.tcut_pno,
        local_ucc.compute_triples,
    ) == ("wannier", 0, 1e-7, False)


def test_dlpno_mp2_local_df_fails_closed_until_periodic_domains_exist() -> None:
    from vibeqc.dlpno.mp2 import DLPNOMP2Options

    system, basis = _h2_chain()
    local_options = DLPNOMP2Options(
        localise="none",
        local_df=True,
        fit_buffer=1.0e9,
        tcut_pairs=0.0,
        tcut_pairs_weak=0.0,
    )

    with pytest.raises(NotImplementedError, match="minimum-image auxiliary domains"):
        vq.run_aiccm2026dev_b_dlpno_mp2(
            system,
            basis,
            (1, 1, 2),
            _options(),
            dlpno_options=local_options,
            progress=False,
        )


@pytest.mark.slow
def test_restricted_canonical_ccsd_t_wrapper_selects_complete_domain() -> None:
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 15.0,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_aiccm2026dev_b_ccsd_t(
            system,
            basis,
            (1, 1, 1),
            _options(),
            progress=False,
        )
    assert result.converged
    assert result.backend == "aiccm2026dev-b-ccsd(t)"
    # The native reduction order differs at the sub-nHa level across the
    # supported toolchains; retain a tight numerical anchor without making
    # the experimental wrapper platform-bitwise.
    assert result.e_corr_per_cell == pytest.approx(-0.0164613902342, abs=5e-10)
    assert result.solver_result.n_pairs == 1


@pytest.mark.slow
def test_unrestricted_real_torus_ump2_and_properties() -> None:
    system, basis = _li_doublet()
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_aiccm2026dev_b_ump2(
            system,
            basis,
            (1, 1, 1),
            _options(),
            progress=False,
        )
    assert isinstance(result, AICCM2026DevBUMP2Result)
    assert result.converged
    assert result.e_corr_per_cell == pytest.approx(-0.0002432596727, abs=2e-10)
    assert result.e_corr_per_cell == pytest.approx(
        result.e_aa_per_cell + result.e_bb_per_cell + result.e_ab_per_cell,
        abs=1e-14,
    )
    assert result.cderi_imaginary_residual < 1e-10
    assert result.cderi_symmetry_residual < 1e-10

    from vibeqc.dlpno.ump2 import DLPNOUMP2Options

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        local = vq.run_aiccm2026dev_b_dlpno_ump2(
            system,
            basis,
            (1, 1, 1),
            _options(),
            ump2_options=DLPNOUMP2Options(
                localise="wannier",
                n_frozen=0,
                tcut_pno=0.0,
                tcut_pairs=0.0,
            ),
            progress=False,
        )
    assert local.converged
    assert local.localization == "wannier"
    assert local.e_total_per_cell == pytest.approx(result.e_total_per_cell, abs=2e-10)

    properties = vq.derive_aiccm2026dev_b_scf_properties(
        result.hf_result,
        system,
        basis,
    )
    assert properties.n_electrons == pytest.approx(3.0, abs=1e-10)
    assert properties.n_alpha == pytest.approx(2.0, abs=1e-10)
    assert properties.n_beta == pytest.approx(1.0, abs=1e-10)
    assert properties.mulliken_spin_populations.sum() == pytest.approx(1.0)
    assert properties.s_squared == pytest.approx(0.75, abs=1e-10)


@pytest.mark.slow
def test_unrestricted_full_domain_uccsd_t_and_local_pno_limit() -> None:
    system, basis = _li_doublet()
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        canonical = vq.run_aiccm2026dev_b_uccsd_t(
            system,
            basis,
            (1, 1, 1),
            _options(),
            progress=False,
        )
    assert isinstance(canonical, AICCM2026DevBUCCSDResult)
    assert canonical.converged
    assert canonical.e_corr_per_cell == pytest.approx(-0.0002578091059, abs=2e-10)
    assert canonical.e_total_per_cell == pytest.approx(
        canonical.e_hf_per_cell
        + canonical.e_corr_per_cell
        + canonical.e_t_per_cell,
        abs=1e-13,
    )

    from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions

    local_options = DLPNOUCCSDPilotOptions(
        localise="wannier",
        n_frozen=0,
        tcut_pno=0.0,
        compute_triples=True,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        local = vq.run_aiccm2026dev_b_dlpno_uccsd_t(
            system,
            basis,
            (1, 1, 1),
            _options(),
            cc_options=local_options,
            progress=False,
        )
    assert local.converged
    assert local.localization == "wannier"
    assert local.localization_result.density_invariance_error < 1e-10
    assert local.e_total_per_cell == pytest.approx(
        canonical.e_total_per_cell,
        abs=2e-10,
    )


@pytest.mark.slow
def test_b_stream_ri_mp2_is_momentum_conserving_and_size_intensive() -> None:
    system, basis = _h2_chain()
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_aiccm2026dev_b_mp2(
            system,
            basis,
            (1, 1, 2),
            _options(),
            progress=False,
        )

    assert isinstance(result, AICCM2026DevBMP2Result)
    assert result.mesh == (1, 1, 2)
    assert result.n_cyclic_cells == 2
    assert result.n_kpoints == 2
    assert result.e_corr_per_cell < 0.0
    assert result.e_corr_per_cell == pytest.approx(
        result.e_corr_ss_per_cell + result.e_corr_os_per_cell,
        abs=1e-14,
    )
    assert result.e_total_per_cell == pytest.approx(
        result.e_hf_per_cell + result.e_corr_per_cell,
        abs=1e-14,
    )
    assert result.momentum_conservation_error < 1e-12
    assert result.max_energy_imaginary_residual < 1e-12
    assert result.hf_diagnostics.density_idempotency_error < 1e-7
    assert result.finite_torus_convention == (
        result.hf_diagnostics.finite_torus_convention
    )
    assert result.exchange_q0 == "bvk-ewald"
    assert result.coulomb_kernel == "3d-periodic-g0"
    assert result.e_corr_per_cell == pytest.approx(-0.0130227001540, abs=5e-10)


@pytest.mark.slow
def test_b_stream_ri_mp2_matches_external_kmp2() -> None:
    from examples.regression.core.runner_pyscf import run_periodic_mp2

    system, basis = _h2_chain()
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_aiccm2026dev_b_mp2(
            system,
            basis,
            (1, 1, 2),
            _options(),
            progress=False,
        )

    bohr_to_angstrom = 0.52917721067
    reference = run_periodic_mp2(
        lattice_ang=np.diag([20.0, 20.0, 6.0]) * bohr_to_angstrom,
        atoms_frac=[
            ("H", (0.5, 0.5, 2.3 / 6.0)),
            ("H", (0.5, 0.5, 3.7 / 6.0)),
        ],
        basis="sto-3g",
        kmesh=(1, 1, 2),
        dimension=3,
    )
    if reference.status == "unavailable":
        pytest.skip("PySCF is unavailable to the out-of-process reference runner")
    assert reference.status == "ok", reference.note
    assert reference.converged
    assert result.e_hf_per_cell == pytest.approx(reference.e_hf, abs=2e-8)
    assert result.e_corr_per_cell == pytest.approx(reference.e_corr, abs=2e-8)


@pytest.mark.slow
@pytest.mark.parametrize("localization", ["none", "wannier"])
def test_real_torus_no_truncation_dlpno_mp2_matches_character_mp2(
    localization: str,
) -> None:
    from vibeqc.dlpno.mp2 import DLPNOMP2Options

    system, basis = _h2_chain()
    local_options = DLPNOMP2Options(
        localise=localization,
        n_frozen=0,
        tcut_pno=0.0,
        tcut_pno_weak=0.0,
        tcut_mkn=0.0,
        tcut_pairs=0.0,
        tcut_pairs_weak=0.0,
        conv_tol_energy=1e-11,
        conv_tol_residual=1e-9,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_aiccm2026dev_b_dlpno_mp2(
            system,
            basis,
            (1, 1, 2),
            _options(),
            dlpno_options=local_options,
            progress=False,
        )

    assert isinstance(result, AICCM2026DevBDLPNOMP2Result)
    assert result.converged
    assert result.localization == localization
    assert result.e_corr_per_cell == pytest.approx(-0.0130227001540, abs=5e-10)
    assert result.cderi_imaginary_residual < 1e-7
    assert result.cderi_symmetry_residual < 1e-7
    assert result.matrix_imaginary_residual < 1e-7
    if localization == "wannier":
        assert result.local_correlation_space is not None
        assert result.local_correlation_space.pao_rank == 2
        assert result.local_correlation_space.translation_orbits_validated
        assert result.local_correlation_space.n_translation_unique_pairs == 2
        assert result.e_corr_per_cell == pytest.approx(
            result.raw_local_e_corr_per_cell
            + result.complete_space_correction_per_cell,
            abs=1e-14,
        )
    else:
        assert result.local_correlation_space is None
        assert abs(result.complete_space_correction_per_cell) < 1e-12


@pytest.mark.slow
def test_public_dlpno_ccsd_t_energy_accounting_and_guards() -> None:
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

    system, basis = _h2_chain()
    local_options = LocalCCSDOptions(
        localise="none",
        n_frozen=0,
        tcut_pno=0.0,
        tcut_mkn=0.0,
        tcut_pairs=0.0,
        coupling_radius=0.0,
        residual_domain="full",
        compute_triples=True,
        tcut_tno=0.0,
        triples_mode="exact",
        conv_tol_energy=1e-11,
        conv_tol_residual=1e-8,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_aiccm2026dev_b_dlpno_ccsd_t(
            system,
            basis,
            (1, 1, 2),
            _options(),
            cc_options=local_options,
            progress=False,
        )

    assert isinstance(result, AICCM2026DevBDLPNOCCSDResult)
    assert result.converged
    assert result.with_triples
    assert result.e_corr_per_cell == pytest.approx(-0.0170866207889, abs=5e-10)
    assert result.e_total_per_cell == pytest.approx(
        result.e_hf_per_cell + result.e_corr_per_cell + result.e_t_per_cell,
        abs=1e-13,
    )

    bad = LocalCCSDOptions(localise="pipek-mezey", coupling_radius=8.0)
    with pytest.raises(NotImplementedError, match="minimum-image"):
        vq.run_aiccm2026dev_b_dlpno_ccsd_t(
            system,
            basis,
            (1, 1, 2),
            _options(),
            cc_options=bad,
            progress=False,
        )


@pytest.mark.slow
def test_no_truncation_dlpno_ccsd_t_has_nonzero_triples_oracle() -> None:
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

    system = vq.PeriodicSystem(
        3,
        np.diag([20.0, 20.0, 7.0]),
        [
            vq.Atom(3, [10.0, 10.0, 2.5]),
            vq.Atom(1, [10.0, 10.0, 4.1]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    local_options = LocalCCSDOptions(
        localise="none",
        n_frozen=0,
        tcut_pno=0.0,
        tcut_mkn=0.0,
        tcut_pairs=0.0,
        coupling_radius=0.0,
        residual_domain="full",
        compute_triples=True,
        triples_mode="exact",
        conv_tol_energy=1e-10,
        conv_tol_residual=1e-7,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_aiccm2026dev_b_dlpno_ccsd_t(
            system,
            basis,
            (1, 1, 2),
            _options(),
            cc_options=local_options,
            progress=False,
    )

    assert result.converged
    # As above, allow the sub-nHa native reduction-order envelope.
    assert result.e_corr_per_cell == pytest.approx(-0.0151237664685, abs=5e-10)
    assert result.e_t_per_cell == pytest.approx(-4.6701557478e-5, abs=2e-11)
    assert result.cderi_symmetry_residual < 1e-7
