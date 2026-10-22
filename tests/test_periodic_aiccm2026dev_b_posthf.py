"""Post-HF invariants for the independent AICCM2026DEV-B stream."""

from __future__ import annotations

from dataclasses import asdict, replace
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


@pytest.mark.parametrize("invalid_character", [np.nan, np.inf, -np.inf])
def test_home_auxiliary_inverse_rejects_invalid_character(
    invalid_character: float,
) -> None:
    from vibeqc._vibeqc_core import aiccm2026dev_b_lpq_to_real_home_auxiliary

    cache = {(0, 0): np.ones((1, 1, 1), dtype=np.complex128)}
    with pytest.raises(ValueError, match="kfrac must contain finite real values"):
        aiccm2026dev_b_lpq_to_real_home_auxiliary(
            cache,
            np.asarray([[invalid_character, 0.0, 0.0]]),
            np.zeros((1, 3), dtype=np.int64),
            np.ones(3, dtype=np.int64),
        )


@pytest.mark.parametrize("representative", [1.0e300, -1.0e300])
def test_home_auxiliary_inverse_accepts_large_trivial_character(
    representative: float,
) -> None:
    from vibeqc._vibeqc_core import aiccm2026dev_b_lpq_to_real_home_auxiliary

    # These finite binary floats are exact integers, hence the trivial
    # character modulo the primitive reciprocal lattice, regardless of size.
    factor = np.asarray([[[2.0, -0.5], [-0.5, 3.0]]])
    kfrac = np.asarray([[representative, -representative, 0.0]])
    direct = aiccm2026dev_b_lpq_to_real_home_auxiliary(
        {(0, 0): factor}, kfrac, np.zeros((1, 3), dtype=np.int64), (1, 1, 1)
    )
    wrapped = _lpq_cache_to_real_home_auxiliary({(0, 0): factor}, kfrac, (1, 1, 1))
    for actual, imaginary, symmetry in (direct, wrapped):
        np.testing.assert_array_equal(actual, factor)
        assert imaginary == 0.0
        assert symmetry == 0.0


@pytest.mark.parametrize("offset", [2.0**40, -(2.0**40)])
def test_home_auxiliary_inverse_preserves_shifted_nontrivial_characters(
    offset: float,
) -> None:
    from vibeqc._vibeqc_core import aiccm2026dev_b_lpq_to_real_home_auxiliary

    # Independent Z/2 forward transform: H F H. The large shift still
    # represents the half-integer exactly; the second axis is trivial.
    expected = np.asarray([[[2.0, -0.5], [-0.5, 3.0]]])
    characters = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    transformed = characters @ expected[0] @ characters.T
    cache = {(i, j): transformed[i, j].reshape(1, 1, 1)
             for i in range(2) for j in range(2)}
    kfrac = np.asarray([[offset, 1.0e300, 0.0], [offset + 0.5, -1.0e300, 0.0]])
    direct = aiccm2026dev_b_lpq_to_real_home_auxiliary(
        cache, kfrac, np.asarray([[0, 0, 0], [1, 0, 0]]), (2, 1, 1)
    )
    wrapped = _lpq_cache_to_real_home_auxiliary(cache, kfrac, (2, 1, 1))
    for actual, imaginary, symmetry in (direct, wrapped):
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0.0)
        assert imaginary < 2e-14
        assert symmetry < 2e-14

    # Canonicalisation must not turn a distinct off-mesh value or duplicate
    # character into a complete group. The fractions remain representable.
    for bad in (0.25, 1.0):
        invalid = kfrac.copy()
        invalid[1, 0] = offset + bad
        with pytest.raises(RuntimeError, match="dual cyclic character"):
            aiccm2026dev_b_lpq_to_real_home_auxiliary(
                cache, invalid, np.asarray([[0, 0, 0], [1, 0, 0]]), (2, 1, 1)
            )
        with pytest.raises(ValueError, match="character"):
            _lpq_cache_to_real_home_auxiliary(cache, invalid, (2, 1, 1))


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
    lov, got_auxiliary = posthf._build_canonical_lov_cache_pairwise(
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
        posthf._build_canonical_lov_cache_pairwise(
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


@pytest.mark.parametrize("fit_source", ["classic", "auto", "unknown"])
@pytest.mark.parametrize("with_triples", [False, True])
def test_chi_cc_carries_its_metric_and_compact_factor_source(
    monkeypatch, fit_source, with_triples,
):
    """The shared engine must not replace an interleaved molecule's overlap."""
    import vibeqc._vibeqc_core as core
    import vibeqc.dlpno.ccsd_local_solver as solver
    import vibeqc.dlpno.translation_factors as translation_factors
    import vibeqc.periodic.chi.posthf as posthf

    system, basis = _h2_chain()
    mesh = (1, 1, 2)
    super_system = _build_supercell_system(system, mesh)
    molecule = super_system.unit_cell_molecule()
    super_basis = vq.BasisSet(molecule, "sto-3g")
    S = np.diag([1., 1.1, 1.2, 1.3])
    hf = SimpleNamespace(
        energy=-2., converged=True, overlap=S, fock=np.diag([-1., -.9, .5, .6]),
        mo_coeffs=np.diag(1 / np.sqrt(S.diagonal())),
    )
    df = _BRealDFAdapter(np.ones((1, 4, 4)), object(), mesh)
    reference = posthf._BRealReference(
        super_system, molecule, super_basis, df, hf, 2, 0., 0., 0.,
    )
    convention = AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0", exchange_q0="bvk-ewald",
        exchange_q0_applicability="active", boundary_model="3d-periodic",
        periodic_dimension=3, character_mesh_shape=mesh,
        bvk_madelung_supercell_repetitions=mesh,
        bvk_madelung_supercell_lattice_bohr=tuple(map(tuple, super_system.lattice)),
    )
    hf_result = SimpleNamespace(
        aiccm2026dev_b=SimpleNamespace(finite_torus_convention=convention),
        executed_fit_source=SimpleNamespace(identity="scf-fit") if fit_source != "unknown" else None,
        rsgdf_fit_request=SimpleNamespace(source=fit_source),
        _chi_correlation_fit_source=SimpleNamespace(identity="correlation-refit"),
    )
    monkeypatch.setattr(posthf, "_prepare_b_real_reference", lambda *a, **k: (reference, hf_result))
    compute_overlap = core.compute_overlap
    molecular_overlap = np.asarray(compute_overlap(basis))

    def no_expansion(*args, **kwargs):
        raise AssertionError("source construction must not expand home-auxiliary factors")

    # TranslationHomeFactors is sealed, so its mo_transform cannot be replaced.
    # Guard the native transforms it calls, and check that the guard fires.
    for name in (
        "aiccm2026dev_b_home_auxiliary_3index_mo_transform",
        "aiccm2026dev_b_home_auxiliary_3index_mo_transform_complex",
    ):
        monkeypatch.setattr(translation_factors, name, no_expansion)
    for probe in (hf.mo_coeffs, hf.mo_coeffs.astype(complex)):
        with pytest.raises(AssertionError, match="must not expand home-auxiliary factors"):
            df.mo_transform(probe)

    def solve(mol, basis, reference, factors, options, *, source=None):
        assert core.compute_overlap is compute_overlap
        np.testing.assert_array_equal(core.compute_overlap(basis_original), molecular_overlap)
        np.testing.assert_array_equal(source.overlap_matrix(), S)
        assert source.reference is hf
        assert source.factors is df
        assert source.conventions.same_source == (fit_source == "classic")
        assert source.conventions.kmesh == mesh
        assert source.conventions.exchange_q0 == "bvk-ewald"
        assert source.provenance["factor_representation"] == "chi-home-auxiliary"
        assert source.provenance["factors"]["shape"] == [1, 4, 4]
        assert source.provenance["n_aux"] == 2
        assert options.compute_triples == with_triples
        return SimpleNamespace(
            e_hf=-2., e_corr=-.02, e_t=-.001 if with_triples else 0.,
            n_pairs=3, n_screened=0, n_iter=1, converged=True, t1_norm=.01,
            source_kind=source.kind, source_identity=source.identity,
        )

    basis_original = basis
    monkeypatch.setattr(solver, "run_local_dlpno_ccsd", solve)
    options = solver.LocalCCSDOptions(localise="none", n_frozen=0, coupling_radius=0,
                                      compute_triples=not with_triples)
    driver = (posthf.run_aiccm2026dev_b_dlpno_ccsd_t if with_triples
              else posthf.run_aiccm2026dev_b_dlpno_ccsd)
    result = driver(system, basis, mesh, cc_options=options, progress=False)
    assert result.source_kind == result.solver_result.source_kind == "chi-real-torus"
    assert result.source_identity == result.solver_result.source_identity
    assert len(result.source_identity) == 64
    assert result.same_source == (fit_source == "classic")
    assert result.exchange_q0 == "bvk-ewald"
    assert result.e_total_per_cell == pytest.approx((-2.02 - (.001 if with_triples else 0)) / 2)
    assert options.compute_triples == (not with_triples)
    assert core.compute_overlap is compute_overlap


@pytest.mark.parametrize("method", ["mp2", "ump2", "uccsd", "uccsd_t"])
@pytest.mark.parametrize("fit_source", ["classic", "auto", "unknown", "refit_same_identity"])
@pytest.mark.parametrize("fail", [False, True])
def test_chi_mp2_and_open_shell_sources_are_calculation_local(
    monkeypatch, method, fit_source, fail,
):
    """Nested drivers retain both metrics, compact fits and caller state."""
    import vibeqc._vibeqc_core as core
    import vibeqc.dlpno.mp2 as mp2
    import vibeqc.dlpno.ump2 as ump2
    import vibeqc.dlpno.uccsd as uccsd
    import vibeqc.dlpno.translation_factors as translation_factors
    import vibeqc.periodic.chi.posthf as posthf

    unrestricted = method != "mp2"
    system, basis = _li_doublet() if unrestricted else _h2_chain()
    mesh = (1, 1, 2)
    super_system = _build_supercell_system(
        system, mesh, multiplicity=(3 if unrestricted else 1),
    )
    molecule = super_system.unit_cell_molecule()
    super_basis = vq.BasisSet(molecule, "sto-3g")
    nbf = int(super_basis.nbasis)
    convention = AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0", exchange_q0="bvk-ewald",
        exchange_q0_applicability="active", boundary_model="3d-periodic",
        periodic_dimension=3, character_mesh_shape=mesh,
        bvk_madelung_supercell_repetitions=mesh,
        bvk_madelung_supercell_lattice_bohr=tuple(map(tuple, super_system.lattice)),
    )
    hf_result = SimpleNamespace(
        aiccm2026dev_b=SimpleNamespace(finite_torus_convention=convention),
        executed_fit_source=SimpleNamespace(identity="scf-fit") if fit_source != "unknown" else None,
        rsgdf_fit_request=SimpleNamespace(source=fit_source),
        _chi_correlation_fit_source=SimpleNamespace(
            identity="scf-fit" if fit_source == "refit_same_identity" else "correlation-refit",
        ),
    )
    references = []
    snapshots = []

    def no_expansion(*args, **kwargs):
        raise AssertionError("source construction expanded compact AO factors")

    # TranslationHomeFactors is sealed, so its mo_transform cannot be replaced.
    # Guard the native transforms it calls instead.
    for name in (
        "aiccm2026dev_b_home_auxiliary_3index_mo_transform",
        "aiccm2026dev_b_home_auxiliary_3index_mo_transform_complex",
    ):
        monkeypatch.setattr(translation_factors, name, no_expansion)

    for scale in (1., 2.):
        overlap = np.diag(scale + np.arange(nbf) / 10)
        coefficients = np.diag(1 / np.sqrt(overlap.diagonal()))
        energies = np.linspace(-1., 1., nbf)
        hf = SimpleNamespace(energy=-2. * scale, converged=True, overlap=overlap)
        if unrestricted:
            rotation = np.eye(nbf)
            angle = .2
            rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]]
            for spin, orbitals, eps in (
                ("alpha", coefficients, energies),
                ("beta", coefficients @ rotation, energies + .15),
            ):
                setattr(hf, "mo_coeffs_" + spin, orbitals)
                setattr(hf, "mo_energies_" + spin, eps)
                setattr(hf, "fock_" + spin, overlap @ orbitals @ np.diag(eps)
                        @ orbitals.T @ overlap)
        else:
            hf.mo_coeffs = coefficients
            hf.mo_energies = energies
            hf.fock = overlap @ coefficients @ np.diag(energies) @ coefficients.T @ overlap
        df = _BRealDFAdapter(np.ones((1, nbf, nbf)) * scale, object(), mesh)
        # The guard must catch both the real and the complex dispatch.
        for probe in (coefficients, coefficients.astype(complex)):
            with pytest.raises(AssertionError, match="expanded compact AO factors"):
                df.mo_transform(probe)
        reference_type = posthf._BRealUReference if unrestricted else posthf._BRealReference
        references.append(reference_type(
            super_system, molecule, super_basis, df, hf, 2, 0., 0., 0.,
        ))
        snapshots.append({key: value.copy() if isinstance(value, np.ndarray) else value
                          for key, value in vars(hf).items()})
    pending = iter(references)
    prepare = "_prepare_b_real_ureference" if unrestricted else "_prepare_b_real_reference"
    monkeypatch.setattr(posthf, prepare, lambda *a, **k: (next(pending), hf_result))
    module, engine, option_type, option_key = {
        "mp2": (mp2, "run_dlpno_mp2", mp2.DLPNOMP2Options, "dlpno_options"),
        "ump2": (ump2, "run_dlpno_ump2", ump2.DLPNOUMP2Options, "ump2_options"),
        "uccsd": (uccsd, "run_dlpno_uccsd_pilot", uccsd.DLPNOUCCSDPilotOptions, "cc_options"),
        "uccsd_t": (uccsd, "run_dlpno_uccsd_pilot", uccsd.DLPNOUCCSDPilotOptions, "cc_options"),
    }[method]
    options = option_type(localise="none", n_frozen=1)
    if hasattr(options, "tcut_pairs"):
        options.tcut_pairs = 0.
    if method == "mp2":
        options.tcut_pairs_weak = 0.
    if method.startswith("ucc"):
        options.compute_triples = method != "uccsd_t"
    before_options = asdict(options)
    driver = getattr(posthf, "run_aiccm2026dev_b_dlpno_" + method)
    compute_overlap = core.compute_overlap
    molecular_overlap = np.asarray(compute_overlap(basis))
    sources = []

    def solve(mol, ao_basis, hf, df, solver_options, *, source=None):
        index = len(sources)
        sources.append(source)
        own = references[index]
        assert core.compute_overlap is compute_overlap
        np.testing.assert_array_equal(core.compute_overlap(basis), molecular_overlap)
        assert source.reference is own.hf
        assert source.factors is own.df
        assert source.reference_kind == ("uhf" if unrestricted else "rhf")
        assert source.molecule.n_electrons() == molecule.n_electrons()
        assert source.molecule.multiplicity == molecule.multiplicity
        assert source.conventions.same_source == (fit_source == "classic")
        assert source.conventions.kmesh == mesh
        assert source.conventions.exchange_q0 == "bvk-ewald"
        assert source.provenance["factor_representation"] == "chi-home-auxiliary"
        assert source.provenance["factors"]["shape"] == [1, nbf, nbf]
        assert source.provenance["n_aux"] == 2
        assert solver_options.n_frozen == 1
        assert solver_options is not options
        if method.startswith("ucc"):
            assert solver_options.compute_triples == (method == "uccsd_t")
        if unrestricted:
            assert not np.array_equal(source.reference.mo_coeffs_alpha, source.reference.mo_coeffs_beta)
            assert not np.array_equal(source.reference.fock_alpha, source.reference.fock_beta)
        np.testing.assert_array_equal(source.overlap_matrix(), snapshots[index]["overlap"])
        if index == 0:
            driver(system, basis, mesh, **{option_key: options}, progress=False)
            np.testing.assert_array_equal(source.overlap_matrix(), snapshots[index]["overlap"])
        elif fail:
            raise RuntimeError("synthetic solver failure")
        return SimpleNamespace(
            e_hf=own.hf.energy, e_corr=-.02, e_total=own.hf.energy-.02,
            e_aa=-.004, e_bb=-.006, e_ab=-.01, e_t=-.001,
            n_pairs=3, n_pairs_screened=0, n_pairs_aa=1, n_pairs_bb=1, n_pairs_ab=1,
            n_iter=1, converged=True, t1_norm=.01, avg_pno=1.,
        )

    monkeypatch.setattr(module, engine, solve)
    if fail:
        with pytest.raises(RuntimeError, match="synthetic solver failure"):
            driver(system, basis, mesh, **{option_key: options}, progress=False)
    else:
        result = driver(system, basis, mesh, **{option_key: options}, progress=False)
        assert result.source_identity == sources[0].identity
        assert result.source_kind == "chi-real-torus"
        assert result.same_source == (fit_source == "classic")
        assert result.e_corr_per_cell == pytest.approx(-.01)
    assert len(sources) == 2
    assert sources[0].identity != sources[1].identity
    assert core.compute_overlap is compute_overlap
    assert asdict(options) == before_options
    for reference, snapshot in zip(references, snapshots):
        assert vars(reference.hf).keys() == snapshot.keys()
        for name, value in snapshot.items():
            np.testing.assert_array_equal(getattr(reference.hf, name), value)


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
        # Match the character cache's base-only reciprocal support. The fold
        # otherwise auto-adds a high-|G| tail even for H/STO-3G at this low
        # cutoff, which defines a different fitted operator.
        tail_ke_cutoff=0.0,
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
        "domain_criterion": "mulliken",
    }
    assert captured["ccsd_t"] == {
        "localise": "pipek-mezey",
        "n_frozen": 0,
        "tcut_pno": 1e-7,
        "pno_norm": "legacy",
        "pno_correction": False,
        "tcut_pno_diag_scale": 1.0,
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


@pytest.mark.parametrize("requested,n_frozen", [(0, 0), (1, 1), (None, 1)])
def test_complete_space_mp2_preserves_solver_frozen_core(
    monkeypatch, requested, n_frozen,
):
    """Public-driver wiring over a synthetic tensor, not a physical-cell oracle."""
    import vibeqc.dlpno.mp2 as mp2
    import vibeqc.periodic.chi.posthf as posthf

    system, basis = _h2_chain()
    mesh = (1, 1, 2)
    super_system = _build_supercell_system(system, mesh)
    convention = AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0", exchange_q0="bvk-ewald",
        exchange_q0_applicability="active", boundary_model="3d-periodic",
        periodic_dimension=3, character_mesh_shape=mesh,
        bvk_madelung_supercell_repetitions=mesh,
        bvk_madelung_supercell_lattice_bohr=tuple(map(tuple, super_system.lattice)),
    )
    # One auxiliary, two occupieds, one virtual. For one virtual the direct
    # and exchange integrals coincide, so E = sum_ij (L_i L_j)^2 / D_ij.
    eps = np.array([-2.0, -1.0, 1.0])
    factors = np.array([0.4, 0.2])
    expected = sum(
        (factors[i] * factors[j]) ** 2 / (eps[i] + eps[j] - 2 * eps[2])
        for i in range(n_frozen, 2) for j in range(n_frozen, 2)
    )
    ao_factor = np.zeros((1, 3, 3))
    ao_factor[0, :2, 2] = factors
    ao_factor[0, 2, :2] = factors

    def transform(occupied, virtual):
        # Frozen occupieds must never reappear as PAO virtuals.
        assert virtual.shape == (3, 1)
        np.testing.assert_allclose(virtual[:2], 0, atol=1e-14)
        return np.einsum("pi,Ppq,qa->Pia", occupied, ao_factor, virtual)

    reference = SimpleNamespace(
        super_system=super_system,
        molecule=SimpleNamespace(n_electrons=lambda: 4),
        basis=SimpleNamespace(nbasis=3, name="synthetic"),
        hf=SimpleNamespace(mo_coeffs=np.eye(3), overlap=np.eye(3),
                           fock=np.diag(eps), energy=-5.0, converged=True),
        df=SimpleNamespace(mo_transform=transform, home_factors=ao_factor, n_aux=1),
        n_cells=2, cderi_imaginary_residual=0.0,
        cderi_symmetry_residual=0.0, matrix_imaginary_residual=0.0,
    )
    solver_result = SimpleNamespace(
        n_frozen=n_frozen, e_corr=expected, e_hf=-5.0,
        e_total=-5.0 + expected, e_pno_correction=0.0,
        n_pairs=(2-n_frozen)*(3-n_frozen)//2,
        n_pairs_screened=0, n_iter=1, converged=True,
    )
    monkeypatch.setattr(posthf, "_prepare_b_real_reference",
                        lambda *a, **k: (reference, object()))
    monkeypatch.setattr(posthf, "_reference_finite_torus_convention",
                        lambda hf: convention)
    monkeypatch.setattr(mp2, "run_dlpno_mp2", lambda *a, **k: solver_result)
    options = mp2.DLPNOMP2Options(
        localise="none", n_frozen=requested, tcut_pno=0, tcut_pno_weak=0,
        tcut_mkn=0, tcut_pairs=0, tcut_pairs_weak=0,
    )
    result = posthf.run_aiccm2026dev_b_dlpno_mp2(
        system, basis, (1, 1, 2), dlpno_options=options, progress=False,
    )
    assert result.e_corr_per_cell == pytest.approx(expected / 2, abs=1e-12)
    assert result.complete_space_correction_per_cell == pytest.approx(0, abs=1e-12)
    assert result.solver_result.e_pno_correction == pytest.approx(0, abs=1e-12)
    assert result.e_total_per_cell == pytest.approx((-5 + expected) / 2, abs=1e-12)
    assert result.solver_result.n_frozen == n_frozen


@pytest.mark.parametrize("method,localise", [
    (method, localise)
    for method in ("mp2", "ccsd", "ccsd_t", "ump2", "uccsd", "uccsd_t")
    for localise in ("none", "wannier", "iao", "pipek-mezey")
    if not (method.startswith("u") and localise == "pipek-mezey")
])
@pytest.mark.parametrize("selector", [0, 1, None, "published", "all-electron"])
def test_local_frozen_core_guard_precedes_electronic_work(
    monkeypatch, method, localise, selector,
):
    import vibeqc.periodic.chi.posthf as posthf
    from vibeqc.dlpno.mp2 import DLPNOMP2Options
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions
    from vibeqc.dlpno.ump2 import DLPNOUMP2Options
    from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions

    # Ne has one published core orbital; the two-cell torus has two.
    system = vq.PeriodicSystem(3, np.eye(3) * 8, [vq.Atom(10, [0, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    option_type, key = {
        "mp2": (DLPNOMP2Options, "dlpno_options"),
        "ccsd": (LocalCCSDOptions, "cc_options"),
        "ccsd_t": (LocalCCSDOptions, "cc_options"),
        "ump2": (DLPNOUMP2Options, "ump2_options"),
        "uccsd": (DLPNOUCCSDPilotOptions, "cc_options"),
        "uccsd_t": (DLPNOUCCSDPilotOptions, "cc_options"),
    }[method]
    options = option_type(localise=localise, n_frozen=selector)
    if not method.startswith("ucc"):
        options.tcut_pairs = 0
    if method == "mp2":
        options.tcut_pairs_weak = 0
    if method in {"ccsd", "ccsd_t"}:
        options.coupling_radius = 0

    class ReachedReference(Exception):
        pass

    def stop(*args, **kwargs):
        raise ReachedReference

    monkeypatch.setattr(posthf, "_prepare_b_real_reference", stop)
    monkeypatch.setattr(posthf, "_prepare_b_real_ureference", stop)
    driver = getattr(posthf, "run_aiccm2026dev_b_dlpno_" + method)
    frozen = selector not in (0, "all-electron")
    if frozen and localise != "none":
        with pytest.raises(NotImplementedError, match="frozen.*localis"):
            driver(system, basis, (1, 1, 2), **{key: options}, progress=False)
    else:
        with pytest.raises(ReachedReference):
            driver(system, basis, (1, 1, 2), **{key: options}, progress=False)
    assert options.localise == localise
    assert options.n_frozen == selector


@pytest.mark.parametrize("selector", [None, True, False, "published", "all-electron", 0])
def test_local_frozen_core_guard_allows_zero_chemical_core(selector):
    from vibeqc.periodic.chi.posthf import _validate_localized_frozen_core

    system, _ = _h2_chain()
    _validate_localized_frozen_core(
        system, SimpleNamespace(localise="wannier", n_frozen=selector),
    )


@pytest.mark.parametrize("method", ["ump2", "uccsd_t"])
@pytest.mark.parametrize("n_frozen", [0, 1])
def test_chi_open_shell_wrappers_preserve_supplied_finite_hamiltonian(
    monkeypatch, method, n_frozen,
):
    """Numerical transport gate on a two-cell tensor model, not Li accuracy."""
    from vibeqc.dlpno._ccsd_ref import run_ref_uccsd
    from vibeqc.dlpno.ump2 import DLPNOUMP2Options
    from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions
    import vibeqc.periodic.chi.posthf as posthf

    system, basis = _li_doublet()
    mesh = (1, 1, 2)
    super_system = _build_supercell_system(system, mesh, multiplicity=3)
    molecule = super_system.unit_cell_molecule()
    # Three synthetic AOs per cell, four alpha and two beta electrons. The
    # metric and both Focks commute with the two-cell translation. This
    # supplied model exercises tensor transport without a fitted Li SCF.
    model_basis = SimpleNamespace(name="finite-torus-model", nbasis=6)
    overlap = np.kron([[1., .08], [.08, 1.]], np.diag([1.1, .9, 1.3]))
    values, vectors = np.linalg.eigh(overlap)
    orthogonalizer = (vectors / np.sqrt(values)) @ vectors.T
    characters = np.kron(np.array([[1., 1.], [1., -1.]]) / np.sqrt(2), np.eye(3))
    rng = np.random.default_rng(719)
    hf = SimpleNamespace(energy=-12., converged=True, overlap=overlap)
    for spin, energies in (
        ("alpha", np.array([-2., -.8, .5, -1.8, -.6, .9])),
        ("beta", np.array([-1.9, -.7, .7, -1.7, -.4, 1.1])),
    ):
        rotations = np.zeros((6, 6))
        rotations[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        rotations[3:, 3:] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        order = np.argsort(energies)
        coefficients = (orthogonalizer @ characters @ rotations)[:, order]
        energies = energies[order]
        fock = overlap @ coefficients @ np.diag(energies) @ coefficients.T @ overlap
        setattr(hf, "mo_coeffs_" + spin, coefficients)
        setattr(hf, "mo_energies_" + spin, energies)
        setattr(hf, "fock_" + spin, fock)
        np.testing.assert_allclose(coefficients.T @ overlap @ coefficients, np.eye(6), atol=2e-14)
    home = rng.normal(scale=.06, size=(2, 6, 6))
    home = (home + home.transpose(0, 2, 1)) / 2
    factors = _BRealDFAdapter(home, object(), mesh)
    reference = posthf._BRealUReference(
        super_system, molecule, model_basis, factors, hf, 2, 0., 0., 0.,
    )
    convention = AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0", exchange_q0="bvk-ewald",
        exchange_q0_applicability="active", boundary_model="3d-periodic",
        periodic_dimension=3, character_mesh_shape=mesh,
        bvk_madelung_supercell_repetitions=mesh,
        bvk_madelung_supercell_lattice_bohr=tuple(map(tuple, super_system.lattice)),
    )
    hf_result = SimpleNamespace(
        aiccm2026dev_b=SimpleNamespace(finite_torus_convention=convention),
        rsgdf_fit_request=SimpleNamespace(source="auto"),
        executed_fit_source=SimpleNamespace(identity="model-scf"),
        _chi_correlation_fit_source=SimpleNamespace(identity="model-correlation"),
    )
    monkeypatch.setattr(posthf, "_prepare_b_real_ureference", lambda *a, **k: (reference, hf_result))
    snapshots = {name: value.copy() for name, value in vars(hf).items()
                 if isinstance(value, np.ndarray)}
    home_before = factors.home_factors.copy()
    for name in snapshots:
        getattr(hf, name).flags.writeable = False
    factors.home_factors.flags.writeable = False
    assert not np.allclose(hf.mo_coeffs_alpha, hf.mo_coeffs_beta)
    assert not np.allclose(hf.fock_alpha, hf.fock_beta)

    if method == "ump2":
        options = DLPNOUMP2Options(localise="none", n_frozen=n_frozen, tcut_pno=0., tcut_pairs=0.)
        before_options = asdict(options)
        result = posthf.run_aiccm2026dev_b_ump2(
            system, basis, mesh, ump2_options=options, progress=False,
        )
        # Independent full-space RI-UMP2 spin sums on the supplied factors.
        channels = []
        for spin, n_occ in (("alpha", 4), ("beta", 2)):
            coefficients = getattr(hf, "mo_coeffs_" + spin)
            energies = getattr(hf, "mo_energies_" + spin)
            channels.append((factors.mo_transform(coefficients[:, n_frozen:n_occ], coefficients[:, n_occ:]),
                             energies[n_frozen:n_occ], energies[n_occ:]))
        expected = []
        for left, right, same_spin in ((channels[0], channels[0], True),
                                       (channels[1], channels[1], True),
                                       (channels[0], channels[1], False)):
            bleft, eleft, vleft = left
            bright, eright, vright = right
            integrals = np.einsum("Pia,Pjb->ijab", bleft, bright)
            denominator = (eleft[:, None, None, None] + eright[None, :, None, None]
                           - vleft[None, None, :, None] - vright[None, None, None, :])
            numerator = integrals - integrals.swapaxes(2, 3) if same_spin else integrals
            expected.append(float(np.sum(numerator**2 / denominator)) * (.25 if same_spin else 1.))
        np.testing.assert_allclose(
            [result.e_aa_per_cell, result.e_bb_per_cell, result.e_ab_per_cell],
            np.array(expected) / 2, atol=2e-14, rtol=0,
        )
        assert result.e_corr_per_cell == pytest.approx(sum(expected) / 2, abs=2e-14)
    else:
        # The full spin-orbital CC anchor has no PNO projection or wrapper.
        ca, cb = hf.mo_coeffs_alpha[:, n_frozen:], hf.mo_coeffs_beta[:, n_frozen:]
        expected = run_ref_uccsd(
            ca.T @ hf.fock_alpha @ ca, cb.T @ hf.fock_beta @ cb,
            factors.mo_transform(ca, ca), factors.mo_transform(cb, cb),
            4 - n_frozen, 2 - n_frozen, e_hf=hf.energy,
            conv_tol=1e-12, compute_triples=True,
        )
        assert expected.converged
        options = DLPNOUCCSDPilotOptions(
            localise="none", n_frozen=n_frozen, tcut_pno=0., compute_triples=False,
            conv_tol_energy=1e-12, conv_tol_residual=1e-11,
        )
        before_options = asdict(options)
        result = posthf.run_aiccm2026dev_b_uccsd_t(
            system, basis, mesh, cc_options=options, progress=False,
        )
        assert abs(expected.e_t) > 1e-15
        assert result.e_corr_per_cell == pytest.approx(expected.e_corr / 2, abs=2e-12)
        assert result.e_t_per_cell == pytest.approx(expected.e_t / 2, abs=2e-13)
        assert result.solver_result.triples_executed
    assert result.converged
    assert result.n_cyclic_cells == 2
    assert result.e_hf_per_cell == hf.energy / 2
    assert result.e_total_per_cell == pytest.approx(result.solver_result.e_total / 2, abs=2e-13)
    assert result.solver_result.n_frozen == n_frozen
    assert result.source_kind == "chi-real-torus"
    assert result.source_identity == result.solver_result.source_identity
    assert not result.same_source
    assert asdict(options) == before_options
    for name, value in snapshots.items():
        np.testing.assert_array_equal(getattr(hf, name), value)
    np.testing.assert_array_equal(factors.home_factors, home_before)
