"""χ-CCM / aiccm2026dev-b analytic-gradient guards."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic.chi.gradient as b_gradient

from vibeqc.periodic.chi.scf import AICCM2026DevBFiniteTorusConvention


def test_b_gradient_module_exports_match_package_root() -> None:
    for name in b_gradient.__all__:
        assert name in vq.__all__
        assert getattr(vq, name) is getattr(b_gradient, name)


def _convention(
    mesh: tuple[int, int, int] = (2, 1, 1),
    primitive_lattice: object | None = None,
) -> AICCM2026DevBFiniteTorusConvention:
    lattice = (
        10.0 * np.eye(3)
        if primitive_lattice is None
        else np.asarray(primitive_lattice, dtype=float)
    )
    bvk_lattice = lattice @ np.diag(np.asarray(mesh, dtype=float))
    return AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0",
        exchange_q0="bvk-ewald",
        exchange_q0_applicability="active",
        boundary_model="3d-periodic",
        periodic_dimension=3,
        character_mesh_shape=mesh,
        bvk_madelung_supercell_repetitions=mesh,
        bvk_madelung_supercell_lattice_bohr=tuple(
            tuple(float(value) for value in row) for row in bvk_lattice
        ),
    )


def test_b_gradient_status_preserves_scf_convention() -> None:
    convention = _convention()
    result = SimpleNamespace(
        backend="aiccm2026dev-b-ri",
        aiccm2026dev_b=SimpleNamespace(
            electronic_method="RHF",
            backend="ri",
            lattice_extension=(2, 1, 1),
            finite_torus_convention=convention,
        ),
    )

    status = vq.aiccm2026dev_b_gradient_status(result)

    assert isinstance(status, vq.AICCM2026DevBGradientStatus)
    assert status.finite_torus_convention == convention
    assert status.coulomb_kernel == "3d-periodic-g0"
    assert status.exchange_q0 == "bvk-ewald"
    assert status.exchange_q0_applicability == "active"
    assert status.boundary_model == "3d-periodic"
    assert status.electronic_method == "RHF"
    assert status.backend == "aiccm2026dev-b-ri"
    assert status.lattice_extension == (2, 1, 1)
    assert not status.analytic_gradient_implemented
    assert "3D Ewald nuclear-repulsion derivative" in status.implemented_terms[0]
    assert any(
        "overlap Lagrangian component" in term for term in status.implemented_terms
    )
    assert any(
        "overlap AO-centre derivative" in term
        for term in status.implemented_terms
    )
    assert any(
        "fixed-input restricted energy-weighted-density" in term
        for term in status.implemented_terms
    )
    assert any(
        "fixed-input unrestricted energy-weighted-density" in term
        for term in status.implemented_terms
    )
    assert any(
        "restricted active BvK exchange-q=0 seam energy" in term
        for term in status.implemented_terms
    )
    assert any(
        "restricted active BvK exchange-q=0 seam AO-centre" in term
        for term in status.implemented_terms
    )
    assert any(
        "unrestricted active BvK exchange-q=0 seam energy" in term
        for term in status.implemented_terms
    )
    assert any(
        "unrestricted active BvK exchange-q=0 seam AO-centre" in term
        for term in status.implemented_terms
    )
    assert any("kinetic energy component" in term for term in status.implemented_terms)
    assert any("kinetic AO-centre derivative" in term for term in status.implemented_terms)
    assert any("electron-nuclear derivative" in term for term in status.implemented_terms)
    assert any(
        "electrostatic component bundle" in term
        for term in status.implemented_terms
    )
    assert any("energy component bundle" in term for term in status.implemented_terms)
    assert any("density inverse-Bloch fold" in term for term in status.implemented_terms)
    assert any(
        "finite-torus mesh and BvK lattice binding" in term
        for term in status.implemented_terms
    )
    assert any("SCF-density 3D Ewald" in term for term in status.implemented_terms)
    assert any("exchange-q=0" in term for term in status.blocked_terms)
    assert any(
        "stationary SCF energy-weighted density binding" in term
        for term in status.blocked_terms
    )
    assert any("exchange-correlation" in term for term in status.blocked_terms)
    assert not any("Ewald electron-nuclear" in term for term in status.blocked_terms)


def test_b_gradient_status_preserves_posthf_convention() -> None:
    convention = _convention()
    result = SimpleNamespace(
        backend="aiccm2026dev-b-dlpno-ccsd(t)",
        finite_torus_convention=convention,
        mesh=(2, 1, 1),
    )

    status = vq.aiccm2026dev_b_gradient_status(result)

    assert status.finite_torus_convention == convention
    assert status.backend == "aiccm2026dev-b-dlpno-ccsd(t)"
    assert status.lattice_extension == (2, 1, 1)
    assert "post-HF relaxed-density" in status.blocked_terms[-1]


def test_b_gradient_fails_closed_with_declared_hamiltonian() -> None:
    convention = _convention()
    result = SimpleNamespace(
        backend="aiccm2026dev-b-ri",
        aiccm2026dev_b=SimpleNamespace(
            electronic_method="RKS",
            lattice_extension=(2, 1, 1),
            finite_torus_convention=convention,
        ),
    )

    with pytest.raises(NotImplementedError) as error:
        vq.compute_aiccm2026dev_b_gradient(result)

    message = str(error.value)
    assert "coulomb_kernel='3d-periodic-g0'" in message
    assert "exchange_q0='bvk-ewald'" in message
    assert "union-and-weight Γ-CCM gradient" in message
    assert "real-Gamma/GDF control gradient" in message
    assert "docs/user_guide/aiccm2026dev_b.md" in message
    assert "RI/RIJCOSX three-center derivative" in message


def test_b_gradient_rejects_results_without_b_convention() -> None:
    with pytest.raises(TypeError, match="finite-torus convention"):
        vq.aiccm2026dev_b_gradient_status(SimpleNamespace(backend="aiccm2026dev-a"))


def test_b_gradient_rejects_convention_without_result_mesh_alias() -> None:
    with pytest.raises(TypeError, match="explicit result mesh"):
        vq.aiccm2026dev_b_gradient_status(
            SimpleNamespace(
                backend="aiccm2026dev-b-ri",
                finite_torus_convention=_convention(),
            )
        )


def _h2_input() -> tuple[vq.PeriodicSystem, vq.BasisSet]:
    system = vq.PeriodicSystem(
        3,
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _b_result_with_convention(
    convention: AICCM2026DevBFiniteTorusConvention,
) -> SimpleNamespace:
    return SimpleNamespace(
        backend="aiccm2026dev-b-4c",
        finite_torus_convention=convention,
        aiccm2026dev_b=SimpleNamespace(
            electronic_method="RHF",
            backend="four_center",
            mesh=convention.character_mesh_shape,
            lattice_extension=convention.character_mesh_shape,
            finite_torus_convention=convention,
        ),
    )


def _fixed_restricted_w_inputs(
    mesh: tuple[int, int, int],
) -> tuple[
    vq.PeriodicSystem,
    vq.BasisSet,
    SimpleNamespace,
    vq.LatticeSumOptions,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build exact-support canonical fixed inputs for the D112 audit."""

    lattice = np.array(
        [[12.0, 0.3, 0.1], [0.0, 12.5, 0.4], [0.0, 0.0, 13.0]],
        dtype=float,
    ).T
    system = vq.PeriodicSystem(
        3,
        lattice,
        [
            vq.Atom(1, [0.1, 0.2, 0.3]),
            vq.Atom(1, [1.3, 0.2, 0.4]),
            vq.Atom(1, [0.4, 1.5, 0.2]),
            vq.Atom(1, [1.6, 1.4, 0.5]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kpoints = vq.cyclic_gamma_mesh(system, mesh)
    result = _b_result_with_convention(
        _convention(mesh, primitive_lattice=lattice)
    )
    result.kpoints_frac = np.asarray(kpoints.kpoints_frac, dtype=float)
    result.kpoint_weights = np.asarray(kpoints.weights, dtype=float)
    lattice_options = vq.LatticeSumOptions()
    lattice_options.cutoff_bohr = 15.0
    overlap_lattice = vq.compute_overlap_lattice(
        basis,
        system,
        lattice_options,
    )
    translations = np.asarray(
        [tuple(int(value) for value in cell.index) for cell in overlap_lattice.cells],
        dtype=int,
    )
    phases = np.exp(2j * np.pi * (translations @ result.kpoints_frac.T))
    overlap_k = np.einsum(
        "gk,gij->kij",
        phases,
        np.asarray(overlap_lattice.blocks, dtype=float),
        optimize=True,
    )

    nbf = int(basis.nbasis)
    n_occ = 2
    mesh_array = np.asarray(mesh, dtype=int)
    labels = np.rint(result.kpoints_frac * mesh_array[None, :]).astype(int)
    residues = [
        tuple(int(value) for value in np.mod(label, mesh_array))
        for label in labels
    ]
    residue_to_index = {residue: index for index, residue in enumerate(residues)}
    occupied = np.empty((len(residues), nbf, n_occ), dtype=np.complex128)
    fock = np.empty((len(residues), nbf, nbf), dtype=np.complex128)
    expected_w = np.empty_like(fock)
    handled: set[int] = set()
    rng = np.random.default_rng(20260821)
    trial = rng.normal(size=(nbf, nbf)) + 1j * rng.normal(size=(nbf, nbf))
    complex_unitary, _ = np.linalg.qr(trial)
    real_trial = rng.normal(size=(nbf, nbf))
    real_unitary, _ = np.linalg.qr(real_trial)
    energies = np.linspace(-0.8, 0.7, nbf)
    for index, residue in enumerate(residues):
        if index in handled:
            continue
        opposite = tuple(
            int(value) for value in np.mod(-np.asarray(residue), mesh_array)
        )
        opposite_index = residue_to_index[opposite]
        eigenvalues, eigenvectors = np.linalg.eigh(overlap_k[index])
        assert np.min(eigenvalues) > 0.0
        inverse_sqrt = (
            eigenvectors
            @ np.diag(eigenvalues**-0.5)
            @ eigenvectors.conj().T
        )
        unitary = real_unitary if opposite_index == index else complex_unitary
        full_coefficients = inverse_sqrt @ unitary
        fock_block = (
            overlap_k[index]
            @ full_coefficients
            @ np.diag(energies)
            @ full_coefficients.conj().T
            @ overlap_k[index]
        )
        occupied[index] = full_coefficients[:, :n_occ]
        fock[index] = fock_block
        expected_w[index] = (
            2.0
            * occupied[index]
            @ np.diag(energies[:n_occ])
            @ occupied[index].conj().T
        )
        if opposite_index != index:
            occupied[opposite_index] = occupied[index].conj()
            fock[opposite_index] = fock[index].conj()
            expected_w[opposite_index] = expected_w[index].conj()
        handled.update((index, opposite_index))
    return (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        expected_w,
    )


def _fixed_unrestricted_w_inputs(
    mesh: tuple[int, int, int],
    *,
    alpha_rank: int = 2,
    beta_rank: int = 1,
) -> tuple[
    vq.PeriodicSystem,
    vq.BasisSet,
    SimpleNamespace,
    vq.LatticeSumOptions,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build explicit spin-channel inputs for the D113 algebra audit."""

    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        restricted_fock,
        _,
    ) = _fixed_restricted_w_inputs(mesh)
    result.aiccm2026dev_b.electronic_method = "UHF"
    occupied_alpha = occupied[:, :, :alpha_rank].copy()
    occupied_beta = occupied[:, :, :beta_rank].copy()
    fock_alpha = restricted_fock.copy()
    fock_beta = 0.73 * restricted_fock + 0.11 * overlap_k
    expected_w = np.empty_like(fock_alpha)
    for index in range(len(overlap_k)):
        density_alpha = (
            occupied_alpha[index] @ occupied_alpha[index].conj().T
        )
        density_beta = occupied_beta[index] @ occupied_beta[index].conj().T
        expected_w[index] = (
            density_alpha @ fock_alpha[index] @ density_alpha
            + density_beta @ fock_beta[index] @ density_beta
        )
    return (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied_alpha,
        occupied_beta,
        fock_alpha,
        fock_beta,
        expected_w,
    )


def test_b_component_guard_accepts_column_scaled_skew_bvk_lattice() -> None:
    mesh = (2, 3, 1)
    lattice = np.array(
        [
            [7.0, 0.4, 0.2],
            [0.3, 8.0, 0.5],
            [0.1, 0.6, 9.0],
        ]
    )
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.2, 0.1, 1.4])],
    )
    result = _b_result_with_convention(
        _convention(mesh, primitive_lattice=lattice)
    )
    options = vq.EwaldOptions()
    options.alpha = 0.45
    options.real_cutoff_bohr = 14.0
    options.recip_cutoff_bohr_inv = 8.0

    gradient = vq.compute_aiccm2026dev_b_ewald_nuclear_gradient(
        system,
        result,
        ewald_options=options,
    )

    assert gradient.shape == (2, 3)


def test_b_component_guard_rejects_row_scaled_skew_bvk_lattice() -> None:
    mesh = (2, 3, 1)
    lattice = np.array(
        [
            [7.0, 0.4, 0.2],
            [0.3, 8.0, 0.5],
            [0.1, 0.6, 9.0],
        ]
    )
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.2, 0.1, 1.4])],
    )
    convention = _convention(mesh, primitive_lattice=lattice)
    wrong_lattice = np.diag(np.asarray(mesh, dtype=float)) @ lattice
    assert not np.allclose(wrong_lattice, lattice @ np.diag(mesh))
    convention = replace(
        convention,
        bvk_madelung_supercell_lattice_bohr=tuple(
            tuple(float(value) for value in row) for row in wrong_lattice
        ),
    )

    with pytest.raises(ValueError, match="recorded BvK supercell lattice"):
        vq.compute_aiccm2026dev_b_ewald_nuclear_gradient(
            system,
            _b_result_with_convention(convention),
        )


def test_b_gradient_status_rejects_inconsistent_mesh_provenance() -> None:
    convention = _convention()
    result = _b_result_with_convention(convention)
    result.aiccm2026dev_b.lattice_extension = (3, 1, 1)

    with pytest.raises(ValueError, match="result mesh alias"):
        vq.aiccm2026dev_b_gradient_status(result)

    inconsistent_repetitions = replace(
        convention,
        bvk_madelung_supercell_repetitions=(3, 1, 1),
    )
    with pytest.raises(ValueError, match="character mesh and BvK repetitions"):
        vq.aiccm2026dev_b_gradient_status(
            _b_result_with_convention(inconsistent_repetitions)
        )


def test_b_gradient_status_rejects_disagreeing_convention_descriptors() -> None:
    convention = _convention()
    result = _b_result_with_convention(convention)
    result.finite_torus_convention = replace(convention, exchange_q0="strict-zero")

    with pytest.raises(ValueError, match="top-level finite-torus conventions"):
        vq.aiccm2026dev_b_gradient_status(result)


def test_b_gradient_status_rejects_nonpositive_character_mesh() -> None:
    convention = replace(
        _convention(),
        character_mesh_shape=(0, 1, 1),
        bvk_madelung_supercell_repetitions=(0, 1, 1),
    )

    with pytest.raises(ValueError, match="entries must be positive"):
        vq.aiccm2026dev_b_gradient_status(
            _b_result_with_convention(convention)
        )


def test_b_component_guard_rejects_boundary_and_nonfinite_bvk_lattice() -> None:
    system, _ = _h2_input()
    convention = _convention()

    with pytest.raises(NotImplementedError, match="boundary_model='3d-periodic'"):
        vq.compute_aiccm2026dev_b_ewald_nuclear_gradient(
            system,
            _b_result_with_convention(
                replace(convention, boundary_model="3d-periodic-vacuum")
            ),
        )

    bad_lattice = np.asarray(
        convention.bvk_madelung_supercell_lattice_bohr,
        dtype=float,
    )
    bad_lattice[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite recorded 3x3 BvK lattice"):
        vq.compute_aiccm2026dev_b_ewald_nuclear_gradient(
            system,
            _b_result_with_convention(
                replace(
                    convention,
                    bvk_madelung_supercell_lattice_bohr=tuple(
                        tuple(float(value) for value in row) for row in bad_lattice
                    ),
                )
            ),
        )


@pytest.mark.parametrize(
    ("kpoints_frac", "weights", "n_density_blocks", "message"),
    [
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [0.5, 0.5],
            2,
            "residue exactly once",
        ),
        (
            [[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
            [0.5, 0.5],
            2,
            "unshifted Gamma-centred",
        ),
        (
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [0.6, 0.4],
            2,
            "uniform weights",
        ),
        (
            [[0.0, 0.0, 0.0]],
            [1.0],
            1,
            "complete unreduced",
        ),
    ],
)
def test_b_scf_density_lattice_rejects_unattested_character_net(
    kpoints_frac: list[list[float]],
    weights: list[float],
    n_density_blocks: int,
    message: str,
) -> None:
    system, basis = _h2_input()
    result = _b_result_with_convention(_convention())
    block = np.eye(basis.nbasis)
    result.density = [block.copy() for _ in range(n_density_blocks)]
    result.effective_n_electrons = 2
    result.kpoints_frac = np.asarray(kpoints_frac, dtype=float)
    result.kpoint_weights = np.asarray(weights, dtype=float)

    with pytest.raises(ValueError, match=message):
        vq.compute_aiccm2026dev_b_scf_density_lattice(
            system,
            basis,
            result,
            lattice_options=vq.LatticeSumOptions(),
        )


def test_b_scf_density_lattice_accepts_equivalent_character_labels() -> None:
    system, basis = _h2_input()
    result = _b_result_with_convention(_convention((3, 1, 1)))
    result.density = [
        np.eye(basis.nbasis),
        0.8 * np.eye(basis.nbasis),
        0.8 * np.eye(basis.nbasis),
    ]
    result.effective_n_electrons = 2
    result.kpoints_frac = np.array(
        [[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0], [-1.0 / 3.0, 0.0, 0.0]]
    )
    result.kpoint_weights = np.full(3, 1.0 / 3.0)

    density = vq.compute_aiccm2026dev_b_scf_density_lattice(
        system,
        basis,
        result,
        lattice_options=vq.LatticeSumOptions(),
    )

    assert len(density.blocks) == len(density.cells)


def test_b_scf_density_lattice_rejects_complex_character_metadata() -> None:
    system, basis = _h2_input()
    result = _b_result_with_convention(_convention())
    result.density = [np.eye(basis.nbasis), np.eye(basis.nbasis)]
    result.effective_n_electrons = 2
    result.kpoints_frac = np.array(
        [[0.0, 0.0, 0.0], [0.5 + 1.0j, 0.0, 0.0]],
        dtype=np.complex128,
    )
    result.kpoint_weights = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="real numeric k-points"):
        vq.compute_aiccm2026dev_b_scf_density_lattice(
            system,
            basis,
            result,
            lattice_options=vq.LatticeSumOptions(),
        )


def test_b_scf_density_lattice_rejects_result_from_other_lattice() -> None:
    system, basis = _h2_input()
    result = _b_result_with_convention(
        _convention(primitive_lattice=8.0 * np.eye(3))
    )
    result.density = [np.eye(basis.nbasis), np.eye(basis.nbasis)]
    result.effective_n_electrons = 2
    result.kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result.kpoint_weights = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="recorded BvK supercell lattice"):
        vq.compute_aiccm2026dev_b_scf_density_lattice(
            system,
            basis,
            result,
            lattice_options=vq.LatticeSumOptions(),
        )


def test_b_scf_density_lattice_folds_closed_shell_density() -> None:
    system, basis = _h2_input()
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0
    d_gamma = np.array([[1.2, 0.1], [0.1, 0.8]])
    d_edge = np.array([[0.6, -0.2], [-0.2, 0.4]])
    result = _b_result_with_convention(_convention())
    result.density = [d_gamma, d_edge]
    result.effective_n_electrons = 2
    result.kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result.kpoint_weights = np.array([0.5, 0.5])

    density = vq.compute_aiccm2026dev_b_scf_density_lattice(
        system,
        basis,
        result,
        lattice_options=lat_opts,
    )
    translations = [tuple(int(x) for x in cell.index) for cell in density.cells]
    expected = vq.inverse_bloch_transform(
        [d_gamma, d_edge],
        result.kpoints_frac,
        translations,
        result.kpoint_weights,
    )

    assert len(density.blocks) == len(expected)
    for got, want in zip(density.blocks, expected):
        assert np.max(np.abs(np.asarray(got) - want.real)) < 1.0e-12


def test_b_scf_density_lattice_rejects_inconsistent_closed_shell_metadata() -> None:
    system, basis = _h2_input()
    result = _b_result_with_convention(_convention())
    result.density = [
        np.zeros((basis.nbasis, basis.nbasis)),
        np.zeros((basis.nbasis, basis.nbasis)),
    ]
    result.effective_n_electrons = 1
    result.kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result.kpoint_weights = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="closed-shell records"):
        vq.compute_aiccm2026dev_b_scf_density_lattice(
            system,
            basis,
            result,
            lattice_options=vq.LatticeSumOptions(),
        )


def test_b_scf_density_lattice_folds_unrestricted_total_density() -> None:
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 10.0,
        [vq.Atom(1, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0
    alpha_gamma = np.array([[0.8]])
    beta_gamma = np.array([[0.0]])
    alpha_edge = np.array([[0.5]])
    beta_edge = np.array([[0.0]])
    result = _b_result_with_convention(_convention())
    result.aiccm2026dev_b.electronic_method = "UHF"
    result.density_alpha = [alpha_gamma, alpha_edge]
    result.density_beta = [beta_gamma, beta_edge]
    result.effective_n_electrons = 1
    result.kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result.kpoint_weights = np.array([0.5, 0.5])

    density = vq.compute_aiccm2026dev_b_scf_density_lattice(
        system,
        basis,
        result,
        lattice_options=lat_opts,
    )
    translations = [tuple(int(x) for x in cell.index) for cell in density.cells]
    expected = vq.inverse_bloch_transform(
        [alpha_gamma + beta_gamma, alpha_edge + beta_edge],
        result.kpoints_frac,
        translations,
        result.kpoint_weights,
    )

    for got, want in zip(density.blocks, expected):
        assert np.max(np.abs(np.asarray(got) - want.real)) < 1.0e-12


def test_b_scf_density_lattice_rejects_inconsistent_spin_metadata() -> None:
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 10.0,
        [vq.Atom(1, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    result = _b_result_with_convention(_convention())
    result.aiccm2026dev_b.electronic_method = "UHF"
    result.density_alpha = [np.array([[0.8]]), np.array([[0.5]])]
    result.density_beta = [np.array([[0.0]]), np.array([[0.0]])]
    result.effective_n_electrons = 2
    result.kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result.kpoint_weights = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="integer alpha/beta occupations"):
        vq.compute_aiccm2026dev_b_scf_density_lattice(
            system,
            basis,
            result,
            lattice_options=vq.LatticeSumOptions(),
        )


def test_b_scf_density_lattice_rejects_wrong_k_density_ao_shape() -> None:
    system, basis = _h2_input()
    result = _b_result_with_convention(_convention())
    result.density = [np.zeros((1, 1)), np.zeros((1, 1))]
    result.effective_n_electrons = 2
    result.kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result.kpoint_weights = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="k-density block"):
        vq.compute_aiccm2026dev_b_scf_density_lattice(
            system,
            basis,
            result,
            lattice_options=vq.LatticeSumOptions(),
        )


def test_b_scf_density_lattice_rejects_complex_residue() -> None:
    system, basis = _h2_input()
    result = _b_result_with_convention(_convention())
    result.density = [
        np.zeros((basis.nbasis, basis.nbasis)),
        1j * np.eye(basis.nbasis),
    ]
    result.effective_n_electrons = 2
    result.kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result.kpoint_weights = np.array([0.5, 0.5])

    with pytest.raises(NotImplementedError, match="non-real"):
        vq.compute_aiccm2026dev_b_scf_density_lattice(
            system,
            basis,
            result,
            lattice_options=vq.LatticeSumOptions(),
            imaginary_tolerance=0.0,
        )


def test_b_scf_ewald_electrostatic_components_match_explicit_density() -> None:
    system, basis = _h2_input()
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    d_gamma = np.array([[1.2, 0.1], [0.1, 0.8]])
    d_edge = np.array([[0.6, -0.2], [-0.2, 0.4]])
    result = _b_result_with_convention(_convention())
    result.density = [d_gamma, d_edge]
    result.effective_n_electrons = 2
    result.kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result.kpoint_weights = np.array([0.5, 0.5])
    result.ewald_alpha_bohr_inv = 0.45

    explicit_density = vq.compute_aiccm2026dev_b_scf_density_lattice(
        system,
        basis,
        result,
        lattice_options=lat_opts,
    )
    explicit = vq.compute_aiccm2026dev_b_ewald_electrostatic_gradient_components(
        system,
        basis,
        result,
        explicit_density,
        lattice_options=lat_opts,
    )
    folded = vq.compute_aiccm2026dev_b_scf_ewald_electrostatic_gradient_components(
        system,
        basis,
        result,
        lattice_options=lat_opts,
    )

    assert isinstance(folded, vq.AICCM2026DevBEwaldElectrostaticGradientComponents)
    np.testing.assert_allclose(folded.nuclear, explicit.nuclear)
    np.testing.assert_allclose(folded.electron_nuclear, explicit.electron_nuclear)
    np.testing.assert_allclose(
        folded.total_fixed_density,
        explicit.total_fixed_density,
    )
    assert folded.ewald_alpha_bohr_inv == pytest.approx(0.45)
    assert folded.finite_torus_convention == _convention()


def test_b_scf_ewald_electrostatic_energy_components_match_explicit_density() -> None:
    system, basis = _h2_input()
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    d_gamma = np.array([[1.2, 0.1], [0.1, 0.8]])
    d_edge = np.array([[0.6, -0.2], [-0.2, 0.4]])
    result = _b_result_with_convention(_convention())
    result.density = [d_gamma, d_edge]
    result.effective_n_electrons = 2
    result.kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result.kpoint_weights = np.array([0.5, 0.5])
    result.ewald_alpha_bohr_inv = 0.45

    explicit_density = vq.compute_aiccm2026dev_b_scf_density_lattice(
        system,
        basis,
        result,
        lattice_options=lat_opts,
    )
    explicit = vq.compute_aiccm2026dev_b_ewald_electrostatic_energy_components(
        system,
        basis,
        result,
        explicit_density,
        lattice_options=lat_opts,
    )
    folded = vq.compute_aiccm2026dev_b_scf_ewald_electrostatic_energy_components(
        system,
        basis,
        result,
        lattice_options=lat_opts,
    )

    assert isinstance(folded, vq.AICCM2026DevBEwaldElectrostaticEnergyComponents)
    assert folded.nuclear == pytest.approx(explicit.nuclear)
    assert folded.electron_nuclear == pytest.approx(explicit.electron_nuclear)
    assert folded.total_fixed_density == pytest.approx(explicit.total_fixed_density)
    assert folded.total_fixed_density == pytest.approx(
        folded.nuclear + folded.electron_nuclear
    )
    assert folded.ewald_alpha_bohr_inv == pytest.approx(0.45)
    assert folded.finite_torus_convention == _convention()


def test_b_scf_ewald_electrostatic_components_keep_density_guard() -> None:
    system, basis = _h2_input()
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    result = _b_result_with_convention(_convention())
    result.density = [
        np.zeros((basis.nbasis, basis.nbasis)),
        1j * np.eye(basis.nbasis),
    ]
    result.effective_n_electrons = 2
    result.kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result.kpoint_weights = np.array([0.5, 0.5])
    result.ewald_alpha_bohr_inv = 0.45

    with pytest.raises(NotImplementedError, match="non-real"):
        vq.compute_aiccm2026dev_b_scf_ewald_electrostatic_gradient_components(
            system,
            basis,
            result,
            lattice_options=lat_opts,
            imaginary_tolerance=0.0,
        )
    with pytest.raises(NotImplementedError, match="non-real"):
        vq.compute_aiccm2026dev_b_scf_ewald_electrostatic_energy_components(
            system,
            basis,
            result,
            lattice_options=lat_opts,
            imaginary_tolerance=0.0,
        )


def test_b_ewald_nuclear_gradient_component_matches_fd() -> None:
    system, _ = _h2_input()
    result = _b_result_with_convention(_convention())
    opts = vq.EwaldOptions()
    opts.alpha = 0.45
    opts.real_cutoff_bohr = 18.0
    opts.recip_cutoff_bohr_inv = 9.0

    analytic = vq.compute_aiccm2026dev_b_ewald_nuclear_gradient(
        system,
        result,
        ewald_options=opts,
    )

    atoms = list(system.unit_cell)
    lattice = np.asarray(system.lattice, dtype=float)
    step = 1.0e-5
    fd = np.zeros_like(analytic)
    for atom_index in range(len(atoms)):
        for axis in range(3):

            def displaced(sign: float) -> float:
                shifted = [vq.Atom(atom.Z, list(atom.xyz)) for atom in atoms]
                xyz = list(shifted[atom_index].xyz)
                xyz[axis] += sign * step
                shifted[atom_index] = vq.Atom(shifted[atom_index].Z, xyz)
                shifted_system = vq.PeriodicSystem(3, lattice, shifted)
                return float(vq.ewald_nuclear_repulsion(shifted_system, opts))

            fd[atom_index, axis] = (displaced(+1.0) - displaced(-1.0)) / (
                2.0 * step
            )

    assert analytic.shape == (2, 3)
    assert np.max(np.abs(analytic - fd)) < 1.0e-6
    assert np.max(np.abs(analytic.sum(axis=0))) < 1.0e-8


def test_b_ewald_electron_nuclear_gradient_component_matches_fd() -> None:
    from vibeqc.bipole_ext_el_pole import (
        crystal_default_ewald_alpha,
        crystal_ewald_reciprocal_cutoff,
    )
    from vibeqc.pbc_bipole import (
        _compute_nuclear_lattice_ewald_reciprocal_ft,
        _crystal_ewald_options,
    )

    lattice = 8.0 * np.eye(3)
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    alpha = crystal_default_ewald_alpha(float(abs(np.linalg.det(lattice))))
    k_max = crystal_ewald_reciprocal_cutoff(float(abs(np.linalg.det(lattice))))

    density = vq.compute_overlap_lattice(basis, system, lat_opts)
    rng = np.random.default_rng(11)
    fixed_blocks = []
    for cell_index in range(len(density.cells)):
        block = rng.standard_normal((basis.nbasis, basis.nbasis)) * 0.05
        density.set_block(cell_index, block)
        fixed_blocks.append(block.copy())

    result = _b_result_with_convention(_convention(primitive_lattice=lattice))
    result.ewald_alpha_bohr_inv = alpha

    analytic = vq.compute_aiccm2026dev_b_ewald_electron_nuclear_gradient(
        system,
        basis,
        result,
        density,
        lattice_options=lat_opts,
    )

    def energy_at(perturbed_atoms: list[vq.Atom]) -> float:
        shifted_system = vq.PeriodicSystem(3, lattice, perturbed_atoms)
        shifted_basis = vq.BasisSet(shifted_system.unit_cell_molecule(), "sto-3g")
        overlap = vq.compute_overlap_lattice(shifted_basis, shifted_system, lat_opts)
        ewald_options = _crystal_ewald_options(
            lat_opts,
            alpha_bohr_inv=alpha,
            tolerance=1.0e-8,
            recip_cutoff_bohr_inv=k_max,
        )
        v_ne, _ = _compute_nuclear_lattice_ewald_reciprocal_ft(
            shifted_basis,
            shifted_system,
            lat_opts,
            ewald_options,
            overlap,
            precision=1.0e-8,
            K_max=k_max,
        )
        return sum(
            float(np.sum(fixed_blocks[index] * np.asarray(v_ne.blocks[index])))
            for index in range(len(fixed_blocks))
        )

    step = 1.0e-5
    fd = np.zeros_like(analytic)
    for atom_index in range(len(atoms)):
        for axis in range(3):

            def displaced(sign: float) -> float:
                shifted = [vq.Atom(atom.Z, list(atom.xyz)) for atom in atoms]
                xyz = list(shifted[atom_index].xyz)
                xyz[axis] += sign * step
                shifted[atom_index] = vq.Atom(shifted[atom_index].Z, xyz)
                return energy_at(shifted)

            fd[atom_index, axis] = (displaced(+1.0) - displaced(-1.0)) / (
                2.0 * step
            )

    assert analytic.shape == (2, 3)
    assert np.max(np.abs(analytic - fd)) < 1.0e-6


def test_b_ewald_electrostatic_component_bundle_matches_fd() -> None:
    from vibeqc.bipole_ext_el_pole import (
        crystal_default_ewald_alpha,
        crystal_ewald_reciprocal_cutoff,
    )

    lattice = 8.0 * np.eye(3)
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    alpha = crystal_default_ewald_alpha(float(abs(np.linalg.det(lattice))))
    k_max = crystal_ewald_reciprocal_cutoff(float(abs(np.linalg.det(lattice))))

    density = vq.compute_overlap_lattice(basis, system, lat_opts)
    rng = np.random.default_rng(13)
    for cell_index in range(len(density.cells)):
        block = rng.standard_normal((basis.nbasis, basis.nbasis)) * 0.05
        density.set_block(cell_index, block)

    result = _b_result_with_convention(_convention(primitive_lattice=lattice))
    result.ewald_alpha_bohr_inv = alpha

    energy_components = (
        vq.compute_aiccm2026dev_b_ewald_electrostatic_energy_components(
            system,
            basis,
            result,
            density,
            lattice_options=lat_opts,
        )
    )
    components = vq.compute_aiccm2026dev_b_ewald_electrostatic_gradient_components(
        system,
        basis,
        result,
        density,
        lattice_options=lat_opts,
    )

    def energy_at(perturbed_atoms: list[vq.Atom]) -> float:
        shifted_system = vq.PeriodicSystem(3, lattice, perturbed_atoms)
        shifted_basis = vq.BasisSet(shifted_system.unit_cell_molecule(), "sto-3g")
        shifted_result = _b_result_with_convention(
            _convention(primitive_lattice=lattice)
        )
        shifted_result.ewald_alpha_bohr_inv = alpha
        return (
            vq.compute_aiccm2026dev_b_ewald_electrostatic_energy_components(
                shifted_system,
                shifted_basis,
                shifted_result,
                density,
                lattice_options=lat_opts,
            ).total_fixed_density
        )

    assert isinstance(
        energy_components,
        vq.AICCM2026DevBEwaldElectrostaticEnergyComponents,
    )
    assert energy_components.total_fixed_density == pytest.approx(
        energy_components.nuclear + energy_components.electron_nuclear
    )
    assert energy_components.ewald_alpha_bohr_inv == pytest.approx(alpha)
    assert energy_components.nuclear_cutoff_bohr == pytest.approx(8.0)
    assert energy_components.reciprocal_cutoff_bohr_inv == pytest.approx(k_max)
    assert energy_components.finite_torus_convention == _convention(
        primitive_lattice=lattice
    )

    step = 1.0e-5
    fd = np.zeros_like(components.total_fixed_density)
    for atom_index in range(len(atoms)):
        for axis in range(3):

            def displaced(sign: float) -> float:
                shifted = [vq.Atom(atom.Z, list(atom.xyz)) for atom in atoms]
                xyz = list(shifted[atom_index].xyz)
                xyz[axis] += sign * step
                shifted[atom_index] = vq.Atom(shifted[atom_index].Z, xyz)
                return energy_at(shifted)

            fd[atom_index, axis] = (displaced(+1.0) - displaced(-1.0)) / (
                2.0 * step
            )

    assert isinstance(
        components,
        vq.AICCM2026DevBEwaldElectrostaticGradientComponents,
    )
    assert components.nuclear.shape == (2, 3)
    assert components.electron_nuclear.shape == (2, 3)
    assert np.allclose(
        components.total_fixed_density,
        components.nuclear + components.electron_nuclear,
    )
    assert components.ewald_alpha_bohr_inv == pytest.approx(alpha)
    assert components.nuclear_cutoff_bohr == pytest.approx(8.0)
    assert components.reciprocal_cutoff_bohr_inv == pytest.approx(k_max)
    assert components.finite_torus_convention == _convention(
        primitive_lattice=lattice
    )
    assert np.max(np.abs(components.total_fixed_density - fd)) < 1.0e-6


def test_b_ewald_electron_nuclear_gradient_component_rejects_non_ewald_lattice() -> None:
    system, basis = _h2_input()
    density = vq.compute_overlap_lattice(basis, system, vq.LatticeSumOptions())
    result = _b_result_with_convention(_convention())
    result.ewald_alpha_bohr_inv = 0.45

    with pytest.raises(NotImplementedError, match="CoulombMethod.EWALD_3D"):
        vq.compute_aiccm2026dev_b_ewald_electron_nuclear_gradient(
            system,
            basis,
            result,
            density,
            lattice_options=vq.LatticeSumOptions(),
        )


def test_b_ewald_electrostatic_component_bundle_rejects_non_ewald_lattice() -> None:
    system, basis = _h2_input()
    density = vq.compute_overlap_lattice(basis, system, vq.LatticeSumOptions())
    result = _b_result_with_convention(_convention())
    result.ewald_alpha_bohr_inv = 0.45

    with pytest.raises(NotImplementedError, match="CoulombMethod.EWALD_3D"):
        vq.compute_aiccm2026dev_b_ewald_electrostatic_energy_components(
            system,
            basis,
            result,
            density,
            lattice_options=vq.LatticeSumOptions(),
        )
    with pytest.raises(NotImplementedError, match="CoulombMethod.EWALD_3D"):
        vq.compute_aiccm2026dev_b_ewald_electrostatic_gradient_components(
            system,
            basis,
            result,
            density,
            lattice_options=vq.LatticeSumOptions(),
        )


def test_b_ewald_electrostatic_component_bundle_rejects_cell_mismatch() -> None:
    system, basis = _h2_input()
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    wide_opts = vq.LatticeSumOptions()
    wide_opts.cutoff_bohr = 16.0
    wide_opts.nuclear_cutoff_bohr = 16.0
    wide_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    density = vq.compute_overlap_lattice(basis, system, wide_opts)
    narrow = vq.compute_overlap_lattice(basis, system, lat_opts)
    assert len(density.cells) > len(narrow.cells)
    result = _b_result_with_convention(_convention())
    result.ewald_alpha_bohr_inv = 0.45

    with pytest.raises(ValueError, match="fixed-density cell list"):
        vq.compute_aiccm2026dev_b_ewald_electron_nuclear_gradient(
            system,
            basis,
            result,
            density,
            lattice_options=lat_opts,
        )
    with pytest.raises(ValueError, match="fixed-density cell list"):
        vq.compute_aiccm2026dev_b_ewald_electrostatic_energy_components(
            system,
            basis,
            result,
            density,
            lattice_options=lat_opts,
        )
    with pytest.raises(ValueError, match="fixed-density cell list"):
        vq.compute_aiccm2026dev_b_ewald_electrostatic_gradient_components(
            system,
            basis,
            result,
            density,
            lattice_options=lat_opts,
        )


def test_b_ewald_electrostatic_component_bundle_rejects_malformed_density() -> None:
    from vibeqc import _vibeqc_core as core

    system, basis = _h2_input()
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    template = vq.compute_overlap_lattice(basis, system, lat_opts)
    result = _b_result_with_convention(_convention())
    result.ewald_alpha_bohr_inv = 0.45

    wrong_nbf = core.make_lattice_matrix_set(
        basis.nbasis + 1,
        list(template.cells),
        [np.zeros((basis.nbasis, basis.nbasis)) for _ in template.cells],
    )
    with pytest.raises(ValueError, match="density AO dimension"):
        vq.compute_aiccm2026dev_b_ewald_electrostatic_gradient_components(
            system,
            basis,
            result,
            wrong_nbf,
            lattice_options=lat_opts,
        )

    missing_blocks = core.make_lattice_matrix_set(
        basis.nbasis,
        list(template.cells),
        [],
    )
    with pytest.raises(ValueError, match="blocks to align"):
        vq.compute_aiccm2026dev_b_ewald_electron_nuclear_gradient(
            system,
            basis,
            result,
            missing_blocks,
            lattice_options=lat_opts,
        )

    wrong_shape = core.make_lattice_matrix_set(
        basis.nbasis,
        list(template.cells),
        [np.zeros((1, 1)) for _ in template.cells],
    )
    with pytest.raises(ValueError, match="density block shape"):
        vq.compute_aiccm2026dev_b_ewald_electrostatic_energy_components(
            system,
            basis,
            result,
            wrong_shape,
            lattice_options=lat_opts,
        )


def test_b_fixed_density_kinetic_gradient_matches_fd() -> None:
    lattice = 8.0 * np.eye(3)
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0

    density = vq.compute_overlap_lattice(basis, system, lat_opts)
    translations = [
        tuple(int(value) for value in cell.index) for cell in density.cells
    ]
    density_k = [
        np.array([[1.2, 0.1], [0.1, 0.8]]),
        np.array([[0.6, -0.2], [-0.2, 0.4]]),
    ]
    density_blocks = vq.inverse_bloch_transform(
        density_k,
        np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        translations,
        np.array([0.5, 0.5]),
    )
    assert np.max(np.abs(density_blocks.imag)) < 1.0e-14
    for cell_index, block in enumerate(density_blocks.real):
        density.set_block(cell_index, np.ascontiguousarray(block))

    result = _b_result_with_convention(_convention(primitive_lattice=lattice))
    analytic = vq.compute_aiccm2026dev_b_fixed_density_kinetic_gradient(
        system,
        basis,
        result,
        density,
        lattice_options=lat_opts,
    )

    def energy_at(perturbed_atoms: list[vq.Atom]) -> float:
        shifted_system = vq.PeriodicSystem(3, lattice, perturbed_atoms)
        shifted_basis = vq.BasisSet(
            shifted_system.unit_cell_molecule(),
            "sto-3g",
        )
        return float(
            vq.compute_aiccm2026dev_b_fixed_density_kinetic_energy(
                shifted_system,
                shifted_basis,
                result,
                density,
                lattice_options=lat_opts,
            )
        )

    step = 1.0e-5
    finite_difference = np.zeros_like(analytic)
    for atom_index in range(len(atoms)):
        for axis in range(3):

            def displaced(sign: float) -> float:
                shifted = [vq.Atom(atom.Z, list(atom.xyz)) for atom in atoms]
                xyz = list(shifted[atom_index].xyz)
                xyz[axis] += sign * step
                shifted[atom_index] = vq.Atom(shifted[atom_index].Z, xyz)
                return energy_at(shifted)

            finite_difference[atom_index, axis] = (
                displaced(+1.0) - displaced(-1.0)
            ) / (2.0 * step)

    assert analytic.shape == (2, 3)
    assert np.max(np.abs(analytic - finite_difference)) < 1.0e-7
    assert np.max(np.abs(analytic.sum(axis=0))) < 1.0e-10


@pytest.mark.parametrize("mesh", ((2, 1, 1), (3, 1, 1), (2, 3, 1)))
def test_b_fixed_restricted_w_matches_character_oracle(
    mesh: tuple[int, int, int],
) -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        expected_w_k,
    ) = _fixed_restricted_w_inputs(mesh)

    actual = (
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
        )
    )
    translations = np.asarray(
        [tuple(int(value) for value in cell.index) for cell in actual.cells],
        dtype=int,
    )
    phases_minus = np.exp(
        -2j * np.pi * (translations @ result.kpoints_frac.T)
    )
    expected = np.einsum(
        "gq,q,qij->gij",
        phases_minus,
        result.kpoint_weights,
        expected_w_k,
        optimize=True,
    )

    assert float(np.max(np.abs(expected.imag))) < 2.0e-13
    np.testing.assert_allclose(
        np.asarray(actual.blocks),
        expected.real,
        rtol=0.0,
        atol=2.0e-12,
    )
    for occupied_q, fock_q, expected_q in zip(occupied, fock, expected_w_k):
        density_q = 2.0 * occupied_q @ occupied_q.conj().T
        np.testing.assert_allclose(
            0.5 * density_q @ fock_q @ density_q,
            expected_q,
            rtol=0.0,
            atol=2.0e-12,
        )
    if mesh == (3, 1, 1):
        phases_plus = np.exp(
            2j * np.pi * (translations @ result.kpoints_frac.T)
        )
        wrong_sign = np.einsum(
            "gq,q,qij->gij",
            phases_plus,
            result.kpoint_weights,
            expected_w_k,
            optimize=True,
        )
        assert float(np.max(np.abs(expected.real - wrong_sign.real))) > 1.0e-3
        assert any(
            float(np.max(np.abs(block - block.T))) > 1.0e-8
            for block in np.asarray(actual.blocks)
        )


def test_b_fixed_restricted_w_is_occupied_unitary_invariant() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        _,
    ) = _fixed_restricted_w_inputs((3, 1, 1))
    reference = (
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
        )
    )
    theta = 0.37
    phase = 0.43
    complex_rotation = np.array(
        [
            [np.cos(theta), np.sin(theta) * np.exp(1j * phase)],
            [-np.sin(theta) * np.exp(-1j * phase), np.cos(theta)],
        ]
    )
    real_rotation = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    rotated = occupied.copy()
    rotated[0] = occupied[0] @ real_rotation
    rotated[1] = occupied[1] @ complex_rotation
    rotated[2] = occupied[2] @ complex_rotation.conj()
    actual = (
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            rotated,
            fock,
            lattice_options=lattice_options,
        )
    )

    np.testing.assert_allclose(
        np.asarray(actual.blocks),
        np.asarray(reference.blocks),
        rtol=0.0,
        atol=2.0e-12,
    )


def test_b_fixed_restricted_w_composes_with_d80_gradient() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        _,
    ) = _fixed_restricted_w_inputs((3, 1, 1))
    energy_weighted_density = (
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
        )
    )
    analytic = vq.compute_aiccm2026dev_b_fixed_energy_weighted_overlap_gradient(
        system,
        basis,
        result,
        energy_weighted_density,
        lattice_options=lattice_options,
    )
    step = 1.0e-4
    values: list[float] = []
    for displacement in (step, -step):
        atoms = [vq.Atom(atom.Z, list(atom.xyz)) for atom in system.unit_cell]
        shifted = np.asarray(atoms[0].xyz, dtype=float)
        shifted[0] += displacement
        atoms[0] = vq.Atom(atoms[0].Z, shifted)
        shifted_system = vq.PeriodicSystem(3, np.asarray(system.lattice), atoms)
        shifted_basis = vq.BasisSet(
            shifted_system.unit_cell_molecule(),
            "sto-3g",
        )
        values.append(
            vq.compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian(
                shifted_system,
                shifted_basis,
                result,
                energy_weighted_density,
                lattice_options=lattice_options,
            )
        )
    finite_difference = (values[0] - values[1]) / (2.0 * step)

    assert analytic[0, 0] == pytest.approx(finite_difference, abs=2.0e-7)
    np.testing.assert_allclose(analytic.sum(axis=0), 0.0, rtol=0.0, atol=2.0e-9)


def test_b_fixed_restricted_w_rejects_unattested_inputs() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        _,
    ) = _fixed_restricted_w_inputs((3, 1, 1))
    helper = (
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice
    )

    bad_overlap = overlap_k.copy()
    bad_overlap[0, 0, 0] += 1.0e-3
    with pytest.raises(ValueError, match="overlap blocks do not match"):
        helper(
            system,
            basis,
            result,
            bad_overlap,
            occupied,
            fock,
            lattice_options=lattice_options,
        )

    nonhermitian_fock = fock.copy()
    nonhermitian_fock[0, 0, 1] += 1.0e-3j
    with pytest.raises(ValueError, match="Hermitian candidate variational Fock"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            nonhermitian_fock,
            lattice_options=lattice_options,
        )

    nonorthonormal = occupied.copy()
    nonorthonormal[0] *= 1.01
    with pytest.raises(ValueError, match="occupied S-orthonormality"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            nonorthonormal,
            fock,
            lattice_options=lattice_options,
        )

    nonfinite = occupied.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite occupied coefficient"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            nonfinite,
            fock,
            lattice_options=lattice_options,
        )

    broken_time_reversal = fock.copy()
    broken_time_reversal[1, 0, 0] += 1.0e-3
    with pytest.raises(NotImplementedError, match="time-reversal-consistent"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            broken_time_reversal,
            lattice_options=lattice_options,
        )

    duplicate_result = SimpleNamespace(**vars(result))
    duplicate_result.kpoints_frac = result.kpoints_frac.copy()
    duplicate_result.kpoints_frac[-1] = duplicate_result.kpoints_frac[0]
    with pytest.raises(ValueError, match="every finite-character residue"):
        helper(
            system,
            basis,
            duplicate_result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
        )

    incomplete_result = SimpleNamespace(**vars(result))
    incomplete_result.kpoints_frac = result.kpoints_frac[:-1]
    incomplete_result.kpoint_weights = result.kpoint_weights[:-1]
    with pytest.raises(ValueError, match="complete unreduced"):
        helper(
            system,
            basis,
            incomplete_result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
        )

    wrong_shape = [block.copy() for block in fock]
    wrong_shape[0] = wrong_shape[0][:-1, :-1]
    with pytest.raises(ValueError, match="candidate variational Fock block 0"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            wrong_shape,
            lattice_options=lattice_options,
        )

    unrestricted_result = _b_result_with_convention(
        _convention((3, 1, 1), primitive_lattice=np.asarray(system.lattice))
    )
    unrestricted_result.kpoints_frac = result.kpoints_frac
    unrestricted_result.kpoint_weights = result.kpoint_weights
    unrestricted_result.aiccm2026dev_b.electronic_method = "UHF"
    with pytest.raises(NotImplementedError, match="restricted RHF/RKS"):
        helper(
            system,
            basis,
            unrestricted_result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
        )

    with pytest.raises(ValueError, match="consistency_tolerance"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
            consistency_tolerance=0.0,
        )

    with pytest.raises(ValueError, match="imaginary_tolerance"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
            imaginary_tolerance=0.0,
        )


@pytest.mark.parametrize(
    "malformed_method",
    ("UHF", "RKS", "RKS/", "RKS/ ", "RKS//", "RKS/PBE\n0", "RKS/PBE\x00"),
)
def test_b_fixed_restricted_w_rejects_non_rks_method_tokens(
    malformed_method: str,
) -> None:
    inputs = list(_fixed_restricted_w_inputs((1, 1, 1)))
    inputs[2].aiccm2026dev_b.electronic_method = malformed_method

    with pytest.raises(NotImplementedError, match="restricted RHF/RKS"):
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[4],
            inputs[5],
            inputs[6],
            lattice_options=inputs[3],
        )


@pytest.mark.parametrize(
    "method",
    ("RKS/PBE0", "RKS/B3LYP/G", "RKS/402, 101"),
)
def test_b_fixed_restricted_w_accepts_nonempty_rks_method(method: str) -> None:
    inputs = list(_fixed_restricted_w_inputs((1, 1, 1)))
    inputs[2].aiccm2026dev_b.electronic_method = method

    actual = (
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[4],
            inputs[5],
            inputs[6],
            lattice_options=inputs[3],
        )
    )

    assert np.all(np.isfinite(np.asarray(actual.blocks)))


@pytest.mark.parametrize(
    ("attribute", "expected_label"),
    (("kpoints_frac", "real numeric k-points"),
     ("kpoint_weights", "real numeric character weights")),
)
def test_b_fixed_restricted_w_rejects_complex_character_metadata(
    attribute: str,
    expected_label: str,
) -> None:
    inputs = list(_fixed_restricted_w_inputs((1, 1, 1)))
    metadata = np.asarray(getattr(inputs[2], attribute), dtype=np.complex128)
    metadata.flat[0] += 1.0j
    setattr(inputs[2], attribute, metadata)

    with pytest.raises(ValueError, match=expected_label):
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[4],
            inputs[5],
            inputs[6],
            lattice_options=inputs[3],
        )


def test_b_fixed_restricted_w_rejects_nondual_character_coordinate() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        _,
    ) = _fixed_restricted_w_inputs((3, 1, 1))
    result.kpoints_frac = result.kpoints_frac.copy()
    result.kpoints_frac[1, 0] = 0.2

    with pytest.raises(ValueError, match="unshifted Gamma-centred"):
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
        )


def test_b_fixed_restricted_w_rejects_unbounded_character_coordinate() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        _,
    ) = _fixed_restricted_w_inputs((1, 1, 1))
    result.kpoints_frac = np.array([[1.0e308, 0.0, 0.0]])

    with pytest.raises(ValueError, match="supported exact-integer range"):
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
        )


def test_b_fixed_restricted_w_rejects_nonfinite_derived_algebra() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        _,
    ) = _fixed_restricted_w_inputs((1, 1, 1))
    fock[0] = np.eye(int(basis.nbasis)) * 1.0e308

    with pytest.raises(ValueError, match="produced non-finite"):
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
        )


def test_b_fixed_restricted_w_rejects_projector_time_reversal_break() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        _,
    ) = _fixed_restricted_w_inputs((3, 1, 1))
    broken = occupied.copy()
    occupied_q = broken[1]
    overlap_q = overlap_k[1]
    trial = np.array(
        [1.0 + 0.2j, -0.4 + 0.7j, 0.3 - 0.8j, 0.6 + 0.1j],
        dtype=np.complex128,
    )
    trial -= occupied_q @ (occupied_q.conj().T @ overlap_q @ trial)
    trial /= np.sqrt(np.real(trial.conj() @ overlap_q @ trial))
    broken[1, :, 0] = trial

    with pytest.raises(NotImplementedError, match="occupied projector"):
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            broken,
            fock,
            lattice_options=lattice_options,
        )


def test_b_fixed_restricted_w_rejects_inverse_transform_imaginary_residue() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        _,
    ) = _fixed_restricted_w_inputs((3, 1, 1))
    perturbed_fock = fock.copy()
    perturbed_fock[1, 0, 0] += 1.0e-6

    with pytest.raises(NotImplementedError, match="non-real inverse-character"):
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            perturbed_fock,
            lattice_options=lattice_options,
            consistency_tolerance=1.0e-4,
            imaginary_tolerance=1.0e-12,
        )


def test_b_fixed_restricted_w_keeps_low_dimension_fail_closed() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        _,
    ) = _fixed_restricted_w_inputs((1, 1, 1))
    slab = vq.PeriodicSystem(2, np.asarray(system.lattice), system.unit_cell)

    with pytest.raises(NotImplementedError, match="only for 3D periodic"):
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            slab,
            basis,
            result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
        )


@pytest.mark.parametrize("mesh", ((2, 1, 1), (3, 1, 1), (2, 3, 1)))
def test_b_fixed_unrestricted_w_matches_spin_channel_oracle(
    mesh: tuple[int, int, int],
) -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied_alpha,
        occupied_beta,
        fock_alpha,
        fock_beta,
        expected_w_k,
    ) = _fixed_unrestricted_w_inputs(mesh)

    actual = vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(
        system,
        basis,
        result,
        overlap_k,
        occupied_alpha,
        occupied_beta,
        fock_alpha,
        fock_beta,
        lattice_options=lattice_options,
    )
    translations = np.asarray(
        [tuple(int(value) for value in cell.index) for cell in actual.cells],
        dtype=int,
    )
    phases_minus = np.exp(
        -2j * np.pi * (translations @ result.kpoints_frac.T)
    )
    expected = np.einsum(
        "gq,q,qij->gij",
        phases_minus,
        result.kpoint_weights,
        expected_w_k,
        optimize=True,
    )

    assert float(np.max(np.abs(expected.imag))) < 2.0e-13
    np.testing.assert_allclose(
        np.asarray(actual.blocks),
        expected.real,
        rtol=0.0,
        atol=2.0e-12,
    )
    for index in range(len(overlap_k)):
        density_alpha = (
            occupied_alpha[index] @ occupied_alpha[index].conj().T
        )
        density_beta = occupied_beta[index] @ occupied_beta[index].conj().T
        np.testing.assert_allclose(
            expected_w_k[index],
            density_alpha @ fock_alpha[index] @ density_alpha
            + density_beta @ fock_beta[index] @ density_beta,
            rtol=0.0,
            atol=2.0e-12,
        )
    if mesh == (3, 1, 1):
        density_alpha = occupied_alpha[1] @ occupied_alpha[1].conj().T
        density_beta = occupied_beta[1] @ occupied_beta[1].conj().T
        spurious_cross_spin = (
            (density_alpha + density_beta)
            @ (0.5 * (fock_alpha[1] + fock_beta[1]))
            @ (density_alpha + density_beta)
        )
        assert (
            float(np.max(np.abs(expected_w_k[1] - spurious_cross_spin)))
            > 1.0e-3
        )
        phases_plus = np.exp(
            2j * np.pi * (translations @ result.kpoints_frac.T)
        )
        wrong_sign = np.einsum(
            "gq,q,qij->gij",
            phases_plus,
            result.kpoint_weights,
            expected_w_k,
            optimize=True,
        )
        assert float(np.max(np.abs(expected.real - wrong_sign.real))) > 1.0e-3


def test_b_fixed_unrestricted_w_reduces_exactly_to_restricted_w() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        _,
    ) = _fixed_restricted_w_inputs((3, 1, 1))
    restricted = (
        vq.compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            fock,
            lattice_options=lattice_options,
        )
    )
    result.aiccm2026dev_b.electronic_method = "UHF"
    unrestricted = (
        vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            occupied,
            fock,
            fock,
            lattice_options=lattice_options,
        )
    )

    np.testing.assert_allclose(
        np.asarray(unrestricted.blocks),
        np.asarray(restricted.blocks),
        rtol=0.0,
        atol=2.0e-12,
    )


def test_b_fixed_unrestricted_w_is_independently_spin_unitary_invariant() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied_alpha,
        occupied_beta,
        fock_alpha,
        fock_beta,
        _,
    ) = _fixed_unrestricted_w_inputs((3, 1, 1), beta_rank=2)
    helper = (
        vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice
    )
    reference = helper(
        system,
        basis,
        result,
        overlap_k,
        occupied_alpha,
        occupied_beta,
        fock_alpha,
        fock_beta,
        lattice_options=lattice_options,
    )

    def rotation(theta: float, phase: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(theta), np.sin(theta) * np.exp(1j * phase)],
                [-np.sin(theta) * np.exp(-1j * phase), np.cos(theta)],
            ]
        )

    rotated_alpha = occupied_alpha.copy()
    rotated_beta = occupied_beta.copy()
    alpha_complex = rotation(0.37, 0.43)
    beta_complex = rotation(-0.28, 0.61)
    alpha_real = rotation(0.37, 0.0).real
    beta_real = rotation(-0.28, 0.0).real
    rotated_alpha[0] = occupied_alpha[0] @ alpha_real
    rotated_alpha[1] = occupied_alpha[1] @ alpha_complex
    rotated_alpha[2] = occupied_alpha[2] @ alpha_complex.conj()
    rotated_beta[0] = occupied_beta[0] @ beta_real
    rotated_beta[1] = occupied_beta[1] @ beta_complex
    rotated_beta[2] = occupied_beta[2] @ beta_complex.conj()
    actual = helper(
        system,
        basis,
        result,
        overlap_k,
        rotated_alpha,
        rotated_beta,
        fock_alpha,
        fock_beta,
        lattice_options=lattice_options,
    )

    np.testing.assert_allclose(
        np.asarray(actual.blocks),
        np.asarray(reference.blocks),
        rtol=0.0,
        atol=2.0e-12,
    )


@pytest.mark.parametrize("zero_spin", ("alpha", "beta"))
def test_b_fixed_unrestricted_w_accepts_one_zero_rank_spin(
    zero_spin: str,
) -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied_alpha,
        occupied_beta,
        fock_alpha,
        fock_beta,
        _,
    ) = _fixed_unrestricted_w_inputs((3, 1, 1))
    zero = np.empty((len(overlap_k), int(basis.nbasis), 0), dtype=np.complex128)
    if zero_spin == "alpha":
        occupied_alpha = zero
    else:
        occupied_beta = zero
    expected_w_k = np.empty_like(fock_alpha)
    for index in range(len(overlap_k)):
        density_alpha = (
            occupied_alpha[index] @ occupied_alpha[index].conj().T
        )
        density_beta = occupied_beta[index] @ occupied_beta[index].conj().T
        expected_w_k[index] = (
            density_alpha @ fock_alpha[index] @ density_alpha
            + density_beta @ fock_beta[index] @ density_beta
        )

    actual = vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(
        system,
        basis,
        result,
        overlap_k,
        occupied_alpha,
        occupied_beta,
        fock_alpha,
        fock_beta,
        lattice_options=lattice_options,
    )
    translations = np.asarray(
        [tuple(int(value) for value in cell.index) for cell in actual.cells],
        dtype=int,
    )
    expected = np.einsum(
        "gq,q,qij->gij",
        np.exp(-2j * np.pi * (translations @ result.kpoints_frac.T)),
        result.kpoint_weights,
        expected_w_k,
        optimize=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual.blocks),
        expected.real,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_b_fixed_unrestricted_w_validates_zero_rank_spin_fock_sewing() -> None:
    inputs = list(_fixed_unrestricted_w_inputs((3, 1, 1)))
    zero = np.empty(
        (len(inputs[4]), int(inputs[1].nbasis), 0),
        dtype=np.complex128,
    )
    inputs[6] = zero
    inputs[8] = inputs[8].copy()
    inputs[8][1, 0, 0] += 1.0e-3

    with pytest.raises(NotImplementedError, match="beta.*time-reversal"):
        vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7],
            inputs[8],
            lattice_options=inputs[3],
        )


def test_b_fixed_unrestricted_w_rejects_spin_residue_cancellation() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied,
        fock,
        _,
    ) = _fixed_restricted_w_inputs((3, 1, 1))
    result.aiccm2026dev_b.electronic_method = "UHF"
    fock_alpha = fock.copy()
    fock_beta = fock.copy()
    fock_alpha[1, 0, 0] += 1.0e-6
    fock_beta[1, 0, 0] -= 1.0e-6

    with pytest.raises(
        NotImplementedError,
        match="alpha channel.*non-real inverse-character",
    ):
        vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied,
            occupied,
            fock_alpha,
            fock_beta,
            lattice_options=lattice_options,
            consistency_tolerance=1.0e-4,
            imaginary_tolerance=1.0e-12,
        )


def test_b_fixed_unrestricted_w_is_character_alias_and_order_invariant() -> None:
    inputs = list(_fixed_unrestricted_w_inputs((3, 1, 1)))
    helper = (
        vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice
    )
    reference = helper(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        lattice_options=inputs[3],
    )
    permutation = np.array([2, 0, 1])
    inputs[2].kpoints_frac = inputs[2].kpoints_frac.copy()
    inputs[2].kpoints_frac[1, 0] += 1.0
    inputs[2].kpoints_frac = inputs[2].kpoints_frac[permutation]
    inputs[2].kpoint_weights = inputs[2].kpoint_weights[permutation]
    for index in range(4, 9):
        inputs[index] = inputs[index][permutation]

    actual = helper(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        lattice_options=inputs[3],
    )

    np.testing.assert_allclose(
        np.asarray(actual.blocks),
        np.asarray(reference.blocks),
        rtol=0.0,
        atol=2.0e-12,
    )


def test_b_fixed_unrestricted_w_composes_with_d80_gradient() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied_alpha,
        occupied_beta,
        fock_alpha,
        fock_beta,
        _,
    ) = _fixed_unrestricted_w_inputs((3, 1, 1))
    energy_weighted_density = (
        vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied_alpha,
            occupied_beta,
            fock_alpha,
            fock_beta,
            lattice_options=lattice_options,
        )
    )
    analytic = vq.compute_aiccm2026dev_b_fixed_energy_weighted_overlap_gradient(
        system,
        basis,
        result,
        energy_weighted_density,
        lattice_options=lattice_options,
    )
    step = 1.0e-4
    values: list[float] = []
    for displacement in (step, -step):
        atoms = [vq.Atom(atom.Z, list(atom.xyz)) for atom in system.unit_cell]
        shifted = np.asarray(atoms[0].xyz, dtype=float)
        shifted[0] += displacement
        atoms[0] = vq.Atom(atoms[0].Z, shifted)
        shifted_system = vq.PeriodicSystem(3, np.asarray(system.lattice), atoms)
        shifted_basis = vq.BasisSet(
            shifted_system.unit_cell_molecule(),
            "sto-3g",
        )
        values.append(
            vq.compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian(
                shifted_system,
                shifted_basis,
                result,
                energy_weighted_density,
                lattice_options=lattice_options,
            )
        )
    finite_difference = (values[0] - values[1]) / (2.0 * step)

    assert analytic[0, 0] == pytest.approx(finite_difference, abs=2.0e-7)
    np.testing.assert_allclose(analytic.sum(axis=0), 0.0, rtol=0.0, atol=2.0e-9)


def test_b_fixed_unrestricted_w_rejects_unattested_spin_inputs() -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied_alpha,
        occupied_beta,
        fock_alpha,
        fock_beta,
        _,
    ) = _fixed_unrestricted_w_inputs((3, 1, 1))
    helper = (
        vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice
    )

    broken_fock_alpha = fock_alpha.copy()
    broken_fock_alpha[1, 0, 0] += 1.0e-3
    with pytest.raises(NotImplementedError, match="alpha.*time-reversal"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            occupied_alpha,
            occupied_beta,
            broken_fock_alpha,
            fock_beta,
            lattice_options=lattice_options,
        )

    broken_projector_beta = occupied_beta.copy()
    occupied_q = broken_projector_beta[1]
    overlap_q = overlap_k[1]
    trial = np.array(
        [1.0 - 0.3j, 0.2 + 0.8j, -0.7 + 0.1j, 0.5 + 0.4j],
        dtype=np.complex128,
    )
    trial -= occupied_q @ (occupied_q.conj().T @ overlap_q @ trial)
    trial /= np.sqrt(np.real(trial.conj() @ overlap_q @ trial))
    broken_projector_beta[1, :, 0] = trial
    with pytest.raises(NotImplementedError, match="beta occupied projector"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            occupied_alpha,
            broken_projector_beta,
            fock_alpha,
            fock_beta,
            lattice_options=lattice_options,
        )

    nonorthonormal_alpha = occupied_alpha.copy()
    nonorthonormal_alpha[0] *= 1.01
    with pytest.raises(ValueError, match="alpha occupied S-orthonormality"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            nonorthonormal_alpha,
            occupied_beta,
            fock_alpha,
            fock_beta,
            lattice_options=lattice_options,
        )

    nonfinite_beta = occupied_beta.copy()
    nonfinite_beta[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite beta occupied coefficient"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            occupied_alpha,
            nonfinite_beta,
            fock_alpha,
            fock_beta,
            lattice_options=lattice_options,
        )

    rank_mismatch_beta = [block.copy() for block in occupied_beta]
    rank_mismatch_beta[0] = np.column_stack(
        (rank_mismatch_beta[0], np.zeros(int(basis.nbasis)))
    )
    with pytest.raises(ValueError, match="common beta occupied rank"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            occupied_alpha,
            rank_mismatch_beta,
            fock_alpha,
            fock_beta,
            lattice_options=lattice_options,
        )

    wrong_shape_beta_fock = [block.copy() for block in fock_beta]
    wrong_shape_beta_fock[0] = wrong_shape_beta_fock[0][:-1, :-1]
    with pytest.raises(ValueError, match="beta candidate variational Fock block 0"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            occupied_alpha,
            occupied_beta,
            fock_alpha,
            wrong_shape_beta_fock,
            lattice_options=lattice_options,
        )

    zero = np.empty((len(overlap_k), int(basis.nbasis), 0), dtype=np.complex128)
    with pytest.raises(ValueError, match="at least one occupied spin orbital"):
        helper(
            system,
            basis,
            result,
            overlap_k,
            zero,
            zero,
            fock_alpha,
            fock_beta,
            lattice_options=lattice_options,
        )


@pytest.mark.parametrize(
    "malformed_method",
    (
        "RHF",
        "RKS/PBE",
        "UKS",
        "UKS/",
        "UKS/ ",
        "UKS//",
        "UKS/PBE\n0",
        "UKS/PBE\x00",
    ),
)
def test_b_fixed_unrestricted_w_rejects_non_uks_method_tokens(
    malformed_method: str,
) -> None:
    (
        system,
        basis,
        result,
        lattice_options,
        overlap_k,
        occupied_alpha,
        occupied_beta,
        fock_alpha,
        fock_beta,
        _,
    ) = _fixed_unrestricted_w_inputs((1, 1, 1))
    result.aiccm2026dev_b.electronic_method = malformed_method

    with pytest.raises(NotImplementedError, match="UHF or UKS/<functional>"):
        vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(
            system,
            basis,
            result,
            overlap_k,
            occupied_alpha,
            occupied_beta,
            fock_alpha,
            fock_beta,
            lattice_options=lattice_options,
        )


@pytest.mark.parametrize(
    "method",
    ("UKS/PBE0", "UKS/B3LYP/G", "UKS/402, 101"),
)
def test_b_fixed_unrestricted_w_accepts_nonempty_uks_method(method: str) -> None:
    inputs = list(_fixed_unrestricted_w_inputs((1, 1, 1)))
    inputs[2].aiccm2026dev_b.electronic_method = method

    actual = vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        lattice_options=inputs[3],
    )

    assert np.all(np.isfinite(np.asarray(actual.blocks)))


def test_b_fixed_unrestricted_w_keeps_low_dimension_fail_closed() -> None:
    inputs = list(_fixed_unrestricted_w_inputs((1, 1, 1)))
    slab = vq.PeriodicSystem(
        2,
        np.asarray(inputs[0].lattice),
        inputs[0].unit_cell,
    )

    with pytest.raises(NotImplementedError, match="only for 3D periodic"):
        vq.compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(
            slab,
            inputs[1],
            inputs[2],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7],
            inputs[8],
            lattice_options=inputs[3],
        )


def test_b_fixed_energy_weighted_overlap_gradient_matches_fd() -> None:
    lattice = 8.0 * np.eye(3)
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.35, 0.2, 1.3]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0

    energy_weighted_density = vq.compute_overlap_lattice(
        basis,
        system,
        lat_opts,
    )
    translations = [
        tuple(int(value) for value in cell.index)
        for cell in energy_weighted_density.cells
    ]
    weighted_positive = np.array(
        [[0.7, 0.11 + 0.23j], [0.11 - 0.23j, 0.4]],
    )
    weighted_k = [
        np.array([[1.2, 0.2], [0.2, 0.8]]),
        weighted_positive,
        weighted_positive.conj(),
    ]
    kpoints_frac = np.array(
        [[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0], [2.0 / 3.0, 0.0, 0.0]],
    )
    weights = np.full(3, 1.0 / 3.0)
    weighted_blocks = vq.inverse_bloch_transform(
        weighted_k,
        kpoints_frac,
        translations,
        weights,
    )
    assert np.max(np.abs(weighted_blocks.imag)) < 1.0e-14
    assert max(
        np.max(np.abs(block - block.T)) for block in weighted_blocks.real
    ) > 0.1
    for cell_index, block in enumerate(weighted_blocks.real):
        energy_weighted_density.set_block(
            cell_index,
            np.ascontiguousarray(block),
        )

    result = _b_result_with_convention(
        _convention((3, 1, 1), primitive_lattice=lattice)
    )
    scalar = (
        vq.compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian(
            system,
            basis,
            result,
            energy_weighted_density,
            lattice_options=lat_opts,
        )
    )
    overlap = vq.compute_overlap_lattice(basis, system, lat_opts)
    explicit_scalar = -sum(
        float(np.sum(weighted * overlap_block))
        for weighted, overlap_block in zip(
            energy_weighted_density.blocks,
            overlap.blocks,
        )
    )
    assert scalar == pytest.approx(explicit_scalar, abs=1.0e-14)
    wrong_orientation = -sum(
        float(np.sum(np.asarray(weighted).T * overlap_block))
        for weighted, overlap_block in zip(
            energy_weighted_density.blocks,
            overlap.blocks,
        )
    )
    assert abs(scalar - wrong_orientation) > 1.0e-5

    def k_space_scalar(overlap_blocks: object) -> float:
        phases = np.exp(
            2j
            * np.pi
            * (np.asarray(translations, dtype=int) @ kpoints_frac.T)
        )
        overlap_k = np.einsum(
            "gk,gij->kij",
            phases,
            np.asarray(overlap_blocks),
            optimize=True,
        )
        return -sum(
            float(weight * np.trace(weighted @ overlap_block).real)
            for weight, weighted, overlap_block in zip(
                weights,
                weighted_k,
                overlap_k,
            )
        )

    assert scalar == pytest.approx(k_space_scalar(overlap.blocks), abs=1.0e-14)

    analytic = vq.compute_aiccm2026dev_b_fixed_energy_weighted_overlap_gradient(
        system,
        basis,
        result,
        energy_weighted_density,
        lattice_options=lat_opts,
    )

    def scalar_at(perturbed_atoms: list[vq.Atom]) -> float:
        shifted_system = vq.PeriodicSystem(3, lattice, perturbed_atoms)
        shifted_basis = vq.BasisSet(
            shifted_system.unit_cell_molecule(),
            "sto-3g",
        )
        public_scalar = float(
            vq.compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian(
                shifted_system,
                shifted_basis,
                result,
                energy_weighted_density,
                lattice_options=lat_opts,
            )
        )
        shifted_overlap = vq.compute_overlap_lattice(
            shifted_basis,
            shifted_system,
            lat_opts,
        )
        oracle_scalar = k_space_scalar(shifted_overlap.blocks)
        assert public_scalar == pytest.approx(oracle_scalar, abs=1.0e-12)
        return oracle_scalar

    step = 1.0e-5
    finite_difference = np.zeros_like(analytic)
    for atom_index in range(len(atoms)):
        for axis in range(3):

            def displaced(sign: float) -> float:
                shifted = [vq.Atom(atom.Z, list(atom.xyz)) for atom in atoms]
                xyz = list(shifted[atom_index].xyz)
                xyz[axis] += sign * step
                shifted[atom_index] = vq.Atom(shifted[atom_index].Z, xyz)
                return scalar_at(shifted)

            finite_difference[atom_index, axis] = (
                displaced(+1.0) - displaced(-1.0)
            ) / (2.0 * step)

    assert analytic.shape == (2, 3)
    assert np.max(np.abs(analytic - finite_difference)) < 1.0e-7
    assert np.max(np.abs(analytic.sum(axis=0))) < 1.0e-10


def _restricted_seam_fixture() -> tuple[
    np.ndarray,
    list[vq.Atom],
    vq.PeriodicSystem,
    vq.BasisSet,
    vq.LatticeSumOptions,
    SimpleNamespace,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
]:
    lattice = np.array(
        [
            [8.0, 0.3, 0.1],
            [0.0, 8.5, 0.4],
            [0.0, 0.0, 9.0],
        ]
    )
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.35, 0.2, 1.3]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lattice_options = vq.LatticeSumOptions()
    lattice_options.cutoff_bohr = 8.0
    lattice_options.nuclear_cutoff_bohr = 8.0
    kpoints_frac = np.array(
        [[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0], [2.0 / 3.0, 0.0, 0.0]],
    )
    weights = np.full(3, 1.0 / 3.0)
    convention = _convention((3, 1, 1), primitive_lattice=lattice)
    result = _b_result_with_convention(convention)
    result.kpoints_frac = kpoints_frac.copy()
    result.kpoint_weights = weights.copy()

    overlap = vq.compute_overlap_lattice(basis, system, lattice_options)
    translations = np.asarray([cell.index for cell in overlap.cells], dtype=int)
    phases = np.exp(2j * np.pi * (translations @ kpoints_frac.T))
    overlap_k = np.einsum(
        "gk,gij->kij",
        phases,
        np.asarray(overlap.blocks),
        optimize=True,
    )

    def normalized_density(raw: object, overlap_block: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(raw, dtype=np.complex128)
        coefficients /= np.sqrt(
            np.vdot(coefficients, overlap_block @ coefficients).real
        )
        return 2.0 * np.outer(coefficients, coefficients.conj())

    density_gamma = normalized_density([1.0, 0.4], overlap_k[0])
    density_positive = normalized_density([1.0, 0.2 + 0.35j], overlap_k[1])
    density_k = [density_gamma, density_positive, density_positive.conj()]
    for density, overlap_block in zip(density_k, overlap_k):
        assert np.trace(density @ overlap_block).real == pytest.approx(2.0)
        assert np.max(np.abs(density @ overlap_block @ density - 2.0 * density)) < (
            1.0e-12
        )
    return (
        lattice,
        atoms,
        system,
        basis,
        lattice_options,
        result,
        kpoints_frac,
        weights,
        density_k,
    )


def test_b_fixed_density_restricted_bvk_seam_energy_and_gradient_match_fd() -> None:
    from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
    from vibeqc.madelung import exxdiv_ewald_energy_shift

    (
        lattice,
        atoms,
        system,
        basis,
        lattice_options,
        result,
        kpoints_frac,
        weights,
        density_k,
    ) = _restricted_seam_fixture()
    eta = float(probe_charge_madelung_supercell(system, (3, 1, 1)))

    def overlap_character_blocks(
        current_system: vq.PeriodicSystem,
        current_basis: vq.BasisSet,
    ) -> tuple[object, np.ndarray, np.ndarray]:
        overlap = vq.compute_overlap_lattice(
            current_basis,
            current_system,
            lattice_options,
        )
        translations = np.asarray([cell.index for cell in overlap.cells], dtype=int)
        phases = np.exp(2j * np.pi * (translations @ kpoints_frac.T))
        overlap_k = np.einsum(
            "gk,gij->kij",
            phases,
            np.asarray(overlap.blocks),
            optimize=True,
        )
        return overlap, translations, overlap_k

    overlap, translations, overlap_k = overlap_character_blocks(system, basis)
    oracle_energy = -0.25 * eta * sum(
        float(weight * np.trace(density @ s @ density @ s).real)
        for weight, density, s in zip(weights, density_k, overlap_k)
    )
    energy = (
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy(
            system,
            basis,
            result,
            density_k,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    assert energy == pytest.approx(oracle_energy, abs=1.0e-14)
    assert energy == pytest.approx(-eta, abs=1.0e-14)
    assert energy == pytest.approx(
        exxdiv_ewald_energy_shift(
            density_k,
            overlap_k,
            eta,
            hf_exchange_fraction=1.0,
            weights=weights,
        ),
        abs=1.0e-14,
    )

    seam_response_k = np.asarray(
        [density @ s @ density for density, s in zip(density_k, overlap_k)]
    )
    seam_response_g = vq.inverse_bloch_transform(
        seam_response_k,
        kpoints_frac,
        translations,
        weights,
    )
    assert np.max(np.abs(seam_response_g.imag)) < 1.0e-14
    assert max(
        np.max(np.abs(block - block.T)) for block in seam_response_g.real
    ) > 0.1
    fixed_response_lagrangian = -0.5 * eta * sum(
        float(np.sum(response * overlap_block))
        for response, overlap_block in zip(seam_response_g.real, overlap.blocks)
    )
    assert fixed_response_lagrangian == pytest.approx(2.0 * energy, abs=1.0e-13)

    analytic = (
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_gradient(
            system,
            basis,
            result,
            density_k,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )

    def energy_at(perturbed_atoms: list[vq.Atom]) -> float:
        shifted_system = vq.PeriodicSystem(3, lattice, perturbed_atoms)
        shifted_basis = vq.BasisSet(
            shifted_system.unit_cell_molecule(),
            "sto-3g",
        )
        _, _, shifted_overlap_k = overlap_character_blocks(
            shifted_system,
            shifted_basis,
        )
        oracle = -0.25 * eta * sum(
            float(weight * np.trace(density @ s @ density @ s).real)
            for weight, density, s in zip(weights, density_k, shifted_overlap_k)
        )
        public = (
            vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy(
                shifted_system,
                shifted_basis,
                result,
                density_k,
                operator_coefficient_eta=eta,
                lattice_options=lattice_options,
            )
        )
        assert public == pytest.approx(oracle, abs=1.0e-12)
        return oracle

    step = 1.0e-5
    finite_difference = np.zeros_like(analytic)
    for atom_index in range(len(atoms)):
        for axis in range(3):

            def displaced(sign: float) -> float:
                shifted = [vq.Atom(atom.Z, list(atom.xyz)) for atom in atoms]
                xyz = list(shifted[atom_index].xyz)
                xyz[axis] += sign * step
                shifted[atom_index] = vq.Atom(shifted[atom_index].Z, xyz)
                return energy_at(shifted)

            finite_difference[atom_index, axis] = (
                displaced(+1.0) - displaced(-1.0)
            ) / (2.0 * step)

    assert np.max(np.abs(analytic - finite_difference)) < 1.0e-7
    assert np.max(np.abs(analytic.sum(axis=0))) < 1.0e-10
    scaled_eta = 1.7 * eta
    scaled_energy = (
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy(
            system,
            basis,
            result,
            density_k,
            operator_coefficient_eta=scaled_eta,
            lattice_options=lattice_options,
        )
    )
    scaled_gradient = (
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_gradient(
            system,
            basis,
            result,
            density_k,
            operator_coefficient_eta=scaled_eta,
            lattice_options=lattice_options,
        )
    )
    assert scaled_energy == pytest.approx(1.7 * energy, abs=1.0e-14)
    assert np.max(np.abs(scaled_gradient - 1.7 * analytic)) < 1.0e-13
    hybrid_result = _b_result_with_convention(result.finite_torus_convention)
    hybrid_result.aiccm2026dev_b.electronic_method = "RKS/PBE0"
    hybrid_result.kpoints_frac = result.kpoints_frac
    hybrid_result.kpoint_weights = result.kpoint_weights
    hybrid_energy = (
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy(
            system,
            basis,
            hybrid_result,
            density_k,
            operator_coefficient_eta=0.25 * eta,
            lattice_options=lattice_options,
        )
    )
    assert hybrid_energy == pytest.approx(0.25 * energy, abs=1.0e-14)


@pytest.mark.parametrize("eta", [0.0, -0.1, np.nan, np.inf, True])
def test_b_fixed_density_restricted_bvk_seam_rejects_invalid_eta(
    eta: object,
) -> None:
    (
        _,
        _,
        system,
        basis,
        lattice_options,
        result,
        _,
        _,
        density_k,
    ) = _restricted_seam_fixture()

    with pytest.raises(ValueError, match="operator coefficient eta"):
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy(
            system,
            basis,
            result,
            density_k,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )


def test_b_fixed_density_restricted_bvk_seam_rejects_inactive_or_unrestricted() -> None:
    (
        _,
        _,
        system,
        basis,
        lattice_options,
        result,
        _,
        _,
        density_k,
    ) = _restricted_seam_fixture()
    inactive = replace(
        result.finite_torus_convention,
        exchange_q0_applicability="inactive",
    )
    inactive_result = _b_result_with_convention(inactive)
    inactive_result.aiccm2026dev_b.electronic_method = "RKS/PBE"
    inactive_result.kpoints_frac = result.kpoints_frac
    inactive_result.kpoint_weights = result.kpoint_weights
    with pytest.raises(NotImplementedError, match="applicability='active'"):
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy(
            system,
            basis,
            inactive_result,
            density_k,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )

    unrestricted = _b_result_with_convention(result.finite_torus_convention)
    unrestricted.aiccm2026dev_b.electronic_method = "UHF"
    unrestricted.kpoints_frac = result.kpoints_frac
    unrestricted.kpoint_weights = result.kpoint_weights
    with pytest.raises(NotImplementedError, match="restricted RHF/RKS"):
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_gradient(
            system,
            basis,
            unrestricted,
            density_k,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )


def test_b_fixed_density_restricted_bvk_seam_rejects_bad_character_contract() -> None:
    (
        _,
        _,
        system,
        basis,
        lattice_options,
        result,
        _,
        _,
        density_k,
    ) = _restricted_seam_fixture()
    helper = (
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy
    )

    incomplete = _b_result_with_convention(result.finite_torus_convention)
    incomplete.kpoints_frac = result.kpoints_frac[:2]
    incomplete.kpoint_weights = np.full(2, 0.5)
    with pytest.raises(ValueError, match="complete unreduced"):
        helper(
            system,
            basis,
            incomplete,
            density_k[:2],
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )

    shifted = _b_result_with_convention(result.finite_torus_convention)
    shifted.kpoints_frac = result.kpoints_frac + np.array([1.0 / 6.0, 0.0, 0.0])
    shifted.kpoint_weights = result.kpoint_weights
    with pytest.raises(ValueError, match="unshifted Gamma-centred"):
        helper(
            system,
            basis,
            shifted,
            density_k,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )

    nonuniform = _b_result_with_convention(result.finite_torus_convention)
    nonuniform.kpoints_frac = result.kpoints_frac
    nonuniform.kpoint_weights = np.array([0.5, 0.25, 0.25])
    with pytest.raises(ValueError, match="uniform weights"):
        helper(
            system,
            basis,
            nonuniform,
            density_k,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )

    complex_weights = _b_result_with_convention(result.finite_torus_convention)
    complex_weights.kpoints_frac = result.kpoints_frac
    complex_weights.kpoint_weights = np.asarray(
        result.kpoint_weights,
        dtype=np.complex128,
    )
    complex_weights.kpoint_weights[0] += 1.0j
    with pytest.raises(ValueError, match="real numeric character weights"):
        helper(
            system,
            basis,
            complex_weights,
            density_k,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )


def test_b_fixed_density_restricted_bvk_seam_rejects_bad_density_contract() -> None:
    (
        _,
        _,
        system,
        basis,
        lattice_options,
        result,
        _,
        _,
        density_k,
    ) = _restricted_seam_fixture()
    helper = (
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_gradient
    )

    nonhermitian = [block.copy() for block in density_k]
    nonhermitian[0][0, 1] += 0.2j
    with pytest.raises(ValueError, match="Hermitian restricted density"):
        helper(
            system,
            basis,
            result,
            nonhermitian,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )

    time_reversal_broken = [block.copy() for block in density_k]
    time_reversal_broken[2] = time_reversal_broken[1].copy()
    with pytest.raises(NotImplementedError, match="time-reversal-consistent"):
        helper(
            system,
            basis,
            result,
            time_reversal_broken,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )

    nonfinite = [block.copy() for block in density_k]
    nonfinite[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite restricted density"):
        helper(
            system,
            basis,
            result,
            nonfinite,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )

    with pytest.raises(ValueError, match="AO shape"):
        helper(
            system,
            basis,
            result,
            [np.zeros((1, 1)) for _ in density_k],
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )


def test_b_fixed_density_restricted_bvk_seam_rejects_lowdim() -> None:
    convention = AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0",
        exchange_q0="bvk-ewald",
        exchange_q0_applicability="active",
        boundary_model="3d-periodic-vacuum",
        periodic_dimension=1,
        character_mesh_shape=(2, 1, 1),
        bvk_madelung_supercell_repetitions=(2, 1, 1),
        bvk_madelung_supercell_lattice_bohr=(
            (20.0, 0.0, 0.0),
            (0.0, 40.0, 0.0),
            (0.0, 0.0, 40.0),
        ),
    )
    system = vq.PeriodicSystem(
        1,
        [[10.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    result = _b_result_with_convention(convention)
    result.kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result.kpoint_weights = np.full(2, 0.5)
    lattice_options = vq.LatticeSumOptions()

    with pytest.raises(NotImplementedError, match="3D periodic"):
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy(
            system,
            basis,
            result,
            [np.eye(basis.nbasis), np.eye(basis.nbasis)],
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )

    result.aiccm2026dev_b.electronic_method = "UHF"
    result.density_alpha = [np.eye(basis.nbasis), np.eye(basis.nbasis)]
    result.density_beta = [np.zeros((basis.nbasis, basis.nbasis))] * 2
    with pytest.raises(NotImplementedError, match="3D periodic"):
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(
            system,
            basis,
            result,
            result.density_alpha,
            result.density_beta,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )


def _unrestricted_seam_fixture() -> tuple[
    np.ndarray,
    list[vq.Atom],
    vq.PeriodicSystem,
    vq.BasisSet,
    vq.LatticeSumOptions,
    SimpleNamespace,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
]:
    (
        lattice,
        atoms,
        system,
        basis,
        lattice_options,
        result,
        kpoints_frac,
        weights,
        restricted_density,
    ) = _restricted_seam_fixture()
    overlap = vq.compute_overlap_lattice(basis, system, lattice_options)
    translations = np.asarray([cell.index for cell in overlap.cells], dtype=int)
    phases = np.exp(2j * np.pi * (translations @ kpoints_frac.T))
    overlap_k = np.einsum(
        "gk,gij->kij",
        phases,
        np.asarray(overlap.blocks),
        optimize=True,
    )

    def normalized_spin_density(raw: object, overlap_block: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(raw, dtype=np.complex128)
        coefficients /= np.sqrt(
            np.vdot(coefficients, overlap_block @ coefficients).real
        )
        return np.outer(coefficients, coefficients.conj())

    density_alpha = [0.5 * block for block in restricted_density]
    beta_gamma = normalized_spin_density([0.35, 1.0], overlap_k[0])
    beta_positive = normalized_spin_density(
        [0.55 - 0.25j, 1.0],
        overlap_k[1],
    )
    density_beta = [beta_gamma, beta_positive, beta_positive.conj()]
    for density_blocks in (density_alpha, density_beta):
        for density, overlap_block in zip(density_blocks, overlap_k):
            assert np.trace(density @ overlap_block).real == pytest.approx(1.0)
            assert np.max(
                np.abs(density @ overlap_block @ density - density)
            ) < 1.0e-12
    assert max(
        np.max(np.abs(alpha - beta))
        for alpha, beta in zip(density_alpha, density_beta)
    ) > 0.1

    result.aiccm2026dev_b.electronic_method = "UHF"
    result.density_alpha = [block.copy() for block in density_alpha]
    result.density_beta = [block.copy() for block in density_beta]
    return (
        lattice,
        atoms,
        system,
        basis,
        lattice_options,
        result,
        kpoints_frac,
        weights,
        density_alpha,
        density_beta,
    )


def test_b_fixed_density_unrestricted_bvk_seam_matches_spin_oracles_and_fd(
) -> None:
    from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
    from vibeqc.madelung import apply_exxdiv_ewald_to_K

    (
        lattice,
        atoms,
        system,
        basis,
        lattice_options,
        result,
        kpoints_frac,
        weights,
        density_alpha,
        density_beta,
    ) = _unrestricted_seam_fixture()
    eta = float(probe_charge_madelung_supercell(system, (3, 1, 1)))

    def overlap_character_blocks(
        current_system: vq.PeriodicSystem,
        current_basis: vq.BasisSet,
    ) -> tuple[np.ndarray, np.ndarray]:
        overlap = vq.compute_overlap_lattice(
            current_basis,
            current_system,
            lattice_options,
        )
        translations = np.asarray([cell.index for cell in overlap.cells], dtype=int)
        phases = np.exp(2j * np.pi * (translations @ kpoints_frac.T))
        overlap_k = np.einsum(
            "gk,gij->kij",
            phases,
            np.asarray(overlap.blocks),
            optimize=True,
        )
        return translations, overlap_k

    translations, overlap_k = overlap_character_blocks(system, basis)
    oracle_energy = -0.5 * eta * sum(
        float(
            weight
            * (
                np.trace(alpha @ overlap @ alpha @ overlap)
                + np.trace(beta @ overlap @ beta @ overlap)
            ).real
        )
        for weight, alpha, beta, overlap in zip(
            weights,
            density_alpha,
            density_beta,
            overlap_k,
        )
    )
    total_density = [
        alpha + beta for alpha, beta in zip(density_alpha, density_beta)
    ]
    wrong_cross_spin_energy = -0.5 * eta * sum(
        float(weight * np.trace(total @ overlap @ total @ overlap).real)
        for weight, total, overlap in zip(weights, total_density, overlap_k)
    )
    assert abs(oracle_energy - wrong_cross_spin_energy) > 1.0e-3

    energy = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(
            system,
            basis,
            result,
            density_alpha,
            density_beta,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    assert energy == pytest.approx(oracle_energy, abs=1.0e-14)

    zero_k = [np.zeros_like(overlap) for overlap in overlap_k]
    shift_alpha = apply_exxdiv_ewald_to_K(
        zero_k,
        overlap_k,
        density_alpha,
        eta,
    )
    shift_beta = apply_exxdiv_ewald_to_K(
        zero_k,
        overlap_k,
        density_beta,
        eta,
    )
    fock_contraction = -0.5 * sum(
        float(
            weight
            * (np.trace(alpha @ ka) + np.trace(beta @ kb)).real
        )
        for weight, alpha, beta, ka, kb in zip(
            weights,
            density_alpha,
            density_beta,
            shift_alpha,
            shift_beta,
        )
    )
    assert energy == pytest.approx(fock_contraction, abs=1.0e-14)

    analytic = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_gradient(
            system,
            basis,
            result,
            density_alpha,
            density_beta,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    response_k = np.asarray(
        [
            eta * (alpha @ overlap @ alpha + beta @ overlap @ beta)
            for alpha, beta, overlap in zip(
                density_alpha,
                density_beta,
                overlap_k,
            )
        ]
    )
    response_g = vq.inverse_bloch_transform(
        response_k,
        kpoints_frac,
        translations,
        weights,
    )
    assert np.max(np.abs(response_g.imag)) < 1.0e-13
    assert max(
        np.max(np.abs(block - block.T)) for block in response_g.real
    ) > 0.01

    def energy_at(perturbed_atoms: list[vq.Atom]) -> float:
        shifted_system = vq.PeriodicSystem(3, lattice, perturbed_atoms)
        shifted_basis = vq.BasisSet(
            shifted_system.unit_cell_molecule(),
            "sto-3g",
        )
        _, shifted_overlap_k = overlap_character_blocks(
            shifted_system,
            shifted_basis,
        )
        oracle = -0.5 * eta * sum(
            float(
                weight
                * (
                    np.trace(alpha @ overlap @ alpha @ overlap)
                    + np.trace(beta @ overlap @ beta @ overlap)
                ).real
            )
            for weight, alpha, beta, overlap in zip(
                weights,
                density_alpha,
                density_beta,
                shifted_overlap_k,
            )
        )
        public = (
            vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(
                shifted_system,
                shifted_basis,
                result,
                density_alpha,
                density_beta,
                operator_coefficient_eta=eta,
                lattice_options=lattice_options,
            )
        )
        assert public == pytest.approx(oracle, abs=1.0e-12)
        return oracle

    step = 1.0e-5
    finite_difference = np.zeros_like(analytic)
    for atom_index in range(len(atoms)):
        for axis in range(3):

            def displaced(sign: float) -> float:
                shifted = [vq.Atom(atom.Z, list(atom.xyz)) for atom in atoms]
                xyz = list(shifted[atom_index].xyz)
                xyz[axis] += sign * step
                shifted[atom_index] = vq.Atom(shifted[atom_index].Z, xyz)
                return energy_at(shifted)

            finite_difference[atom_index, axis] = (
                displaced(+1.0) - displaced(-1.0)
            ) / (2.0 * step)
    assert np.max(np.abs(analytic - finite_difference)) < 1.0e-7
    assert np.max(np.abs(analytic.sum(axis=0))) < 1.0e-10

    scaled_gradient = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_gradient(
            system,
            basis,
            result,
            density_alpha,
            density_beta,
            operator_coefficient_eta=1.7 * eta,
            lattice_options=lattice_options,
        )
    )
    assert np.max(np.abs(scaled_gradient - 1.7 * analytic)) < 1.0e-13

    zero_density = [np.zeros_like(block) for block in density_alpha]
    energy_alpha = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(
            system,
            basis,
            result,
            density_alpha,
            zero_density,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    energy_beta = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(
            system,
            basis,
            result,
            zero_density,
            density_beta,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    swapped = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(
            system,
            basis,
            result,
            density_beta,
            density_alpha,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    assert energy == pytest.approx(energy_alpha + energy_beta, abs=1.0e-14)
    assert swapped == pytest.approx(energy, abs=1.0e-14)

    gradient_alpha = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_gradient(
            system,
            basis,
            result,
            density_alpha,
            zero_density,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    gradient_beta = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_gradient(
            system,
            basis,
            result,
            zero_density,
            density_beta,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    swapped_gradient = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_gradient(
            system,
            basis,
            result,
            density_beta,
            density_alpha,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    assert np.max(np.abs(analytic - gradient_alpha - gradient_beta)) < 1.0e-13
    assert np.max(np.abs(analytic - swapped_gradient)) < 1.0e-13

    delta_positive = np.array(
        [[0.12, 0.07 + 0.03j], [0.07 - 0.03j, -0.08]],
        dtype=np.complex128,
    )
    delta_alpha = [
        np.array([[0.2, -0.15], [-0.15, 0.05]], dtype=np.complex128),
        delta_positive,
        delta_positive.conj(),
    ]
    density_step = 1.0e-6
    energy_plus = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(
            system,
            basis,
            result,
            [d + density_step * delta for d, delta in zip(density_alpha, delta_alpha)],
            density_beta,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    energy_minus = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(
            system,
            basis,
            result,
            [d - density_step * delta for d, delta in zip(density_alpha, delta_alpha)],
            density_beta,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    density_fd = (energy_plus - energy_minus) / (2.0 * density_step)
    density_oracle = -eta * sum(
        float(weight * np.trace(delta @ overlap @ alpha @ overlap).real)
        for weight, delta, alpha, overlap in zip(
            weights,
            delta_alpha,
            density_alpha,
            overlap_k,
        )
    )
    density_from_shift = -sum(
        float(weight * np.trace(delta @ shift).real)
        for weight, delta, shift in zip(weights, delta_alpha, shift_alpha)
    )
    assert density_fd == pytest.approx(density_oracle, abs=1.0e-9)
    assert density_from_shift == pytest.approx(density_oracle, abs=1.0e-14)

    hybrid_result = _b_result_with_convention(result.finite_torus_convention)
    hybrid_result.aiccm2026dev_b.electronic_method = "UKS/PBE0"
    hybrid_result.kpoints_frac = result.kpoints_frac
    hybrid_result.kpoint_weights = result.kpoint_weights
    hybrid_result.density_alpha = result.density_alpha
    hybrid_result.density_beta = result.density_beta
    hybrid_energy = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(
            system,
            basis,
            hybrid_result,
            density_alpha,
            density_beta,
            operator_coefficient_eta=0.25 * eta,
            lattice_options=lattice_options,
        )
    )
    assert hybrid_energy == pytest.approx(0.25 * energy, abs=1.0e-14)


def test_b_fixed_density_unrestricted_bvk_seam_reduces_to_restricted() -> None:
    (
        _,
        _,
        system,
        basis,
        lattice_options,
        restricted_result,
        _,
        _,
        restricted_density,
    ) = _restricted_seam_fixture()
    eta = 0.173
    density_alpha = [0.5 * density for density in restricted_density]
    density_beta = [0.5 * density for density in restricted_density]
    unrestricted_result = _b_result_with_convention(
        restricted_result.finite_torus_convention
    )
    unrestricted_result.aiccm2026dev_b.electronic_method = "UHF"
    unrestricted_result.kpoints_frac = restricted_result.kpoints_frac
    unrestricted_result.kpoint_weights = restricted_result.kpoint_weights
    unrestricted_result.density_alpha = density_alpha
    unrestricted_result.density_beta = density_beta

    restricted_energy = (
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy(
            system,
            basis,
            restricted_result,
            restricted_density,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    unrestricted_energy = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(
            system,
            basis,
            unrestricted_result,
            density_alpha,
            density_beta,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    restricted_gradient = (
        vq.compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_gradient(
            system,
            basis,
            restricted_result,
            restricted_density,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    unrestricted_gradient = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_gradient(
            system,
            basis,
            unrestricted_result,
            density_alpha,
            density_beta,
            operator_coefficient_eta=eta,
            lattice_options=lattice_options,
        )
    )
    assert unrestricted_energy == pytest.approx(restricted_energy, abs=1.0e-14)
    assert np.max(np.abs(unrestricted_gradient - restricted_gradient)) < 1.0e-13


def test_b_fixed_density_unrestricted_bvk_seam_rejects_bad_route() -> None:
    (
        _,
        _,
        system,
        basis,
        lattice_options,
        result,
        _,
        _,
        density_alpha,
        density_beta,
    ) = _unrestricted_seam_fixture()
    helper = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy
    )

    inactive_convention = replace(
        result.finite_torus_convention,
        exchange_q0_applicability="inactive",
    )
    inactive = _b_result_with_convention(inactive_convention)
    inactive.aiccm2026dev_b.electronic_method = "UKS/HSE06"
    inactive.kpoints_frac = result.kpoints_frac
    inactive.kpoint_weights = result.kpoint_weights
    inactive.density_alpha = density_alpha
    inactive.density_beta = density_beta
    with pytest.raises(NotImplementedError, match="applicability='active'"):
        helper(
            system,
            basis,
            inactive,
            density_alpha,
            density_beta,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )

    restricted = _b_result_with_convention(result.finite_torus_convention)
    restricted.kpoints_frac = result.kpoints_frac
    restricted.kpoint_weights = result.kpoint_weights
    with pytest.raises(NotImplementedError, match="only unrestricted"):
        helper(
            system,
            basis,
            restricted,
            density_alpha,
            density_beta,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )

    for malformed_method in (
        "UKS",
        "UKS/",
        "UKS/ ",
        "UKS/\t",
        "UKS//",
        "UHF/PBE0",
        "ROHF",
        "ROKS/PBE0",
    ):
        malformed = _b_result_with_convention(result.finite_torus_convention)
        malformed.aiccm2026dev_b.electronic_method = malformed_method
        malformed.kpoints_frac = result.kpoints_frac
        malformed.kpoint_weights = result.kpoint_weights
        malformed.density_alpha = density_alpha
        malformed.density_beta = density_beta
        with pytest.raises(NotImplementedError, match="only unrestricted"):
            helper(
                system,
                basis,
                malformed,
                density_alpha,
                density_beta,
                operator_coefficient_eta=0.1,
                lattice_options=lattice_options,
            )

    missing_spin_state = _b_result_with_convention(result.finite_torus_convention)
    missing_spin_state.aiccm2026dev_b.electronic_method = "UHF"
    missing_spin_state.kpoints_frac = result.kpoints_frac
    missing_spin_state.kpoint_weights = result.kpoint_weights
    with pytest.raises(TypeError, match="density_alpha and density_beta"):
        helper(
            system,
            basis,
            missing_spin_state,
            density_alpha,
            density_beta,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )

    with pytest.raises(ValueError, match="operator coefficient eta"):
        helper(
            system,
            basis,
            result,
            density_alpha,
            density_beta,
            operator_coefficient_eta=10**1000,
            lattice_options=lattice_options,
        )
    with pytest.raises(ValueError, match="imaginary_tolerance"):
        helper(
            system,
            basis,
            result,
            density_alpha,
            density_beta,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
            imaginary_tolerance=10**1000,
        )


def test_b_fixed_density_unrestricted_bvk_seam_validates_spins_independently(
) -> None:
    (
        _,
        _,
        system,
        basis,
        lattice_options,
        result,
        _,
        _,
        density_alpha,
        density_beta,
    ) = _unrestricted_seam_fixture()
    helper = (
        vq.compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_gradient
    )

    for label in ("alpha", "beta"):
        bad_alpha = [block.copy() for block in density_alpha]
        bad_beta = [block.copy() for block in density_beta]
        target = bad_alpha if label == "alpha" else bad_beta
        target[0][0, 1] += 0.2j
        with pytest.raises(ValueError, match=rf"Hermitian {label} density"):
            helper(
                system,
                basis,
                result,
                bad_alpha,
                bad_beta,
                operator_coefficient_eta=0.1,
                lattice_options=lattice_options,
            )

        bad_alpha = [block.copy() for block in density_alpha]
        bad_beta = [block.copy() for block in density_beta]
        target = bad_alpha if label == "alpha" else bad_beta
        target[2] = target[1].copy()
        with pytest.raises(
            NotImplementedError,
            match=rf"time-reversal-consistent {label}",
        ):
            helper(
                system,
                basis,
                result,
                bad_alpha,
                bad_beta,
                operator_coefficient_eta=0.1,
                lattice_options=lattice_options,
            )

    with pytest.raises(ValueError, match="matching alpha density"):
        helper(
            system,
            basis,
            result,
            density_alpha[:2],
            density_beta,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )
    nonfinite_beta = [block.copy() for block in density_beta]
    nonfinite_beta[0][0, 0] = np.inf
    with pytest.raises(ValueError, match="finite beta density"):
        helper(
            system,
            basis,
            result,
            density_alpha,
            nonfinite_beta,
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )
    with pytest.raises(ValueError, match="beta k-density block.*AO shape"):
        helper(
            system,
            basis,
            result,
            density_alpha,
            [np.zeros((1, 1)) for _ in density_beta],
            operator_coefficient_eta=0.1,
            lattice_options=lattice_options,
        )


@pytest.mark.parametrize(
    "helper_name",
    [
        "compute_aiccm2026dev_b_fixed_density_kinetic_energy",
        "compute_aiccm2026dev_b_fixed_density_kinetic_gradient",
        "compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian",
        "compute_aiccm2026dev_b_fixed_energy_weighted_overlap_gradient",
    ],
)
def test_b_fixed_one_electron_component_rejects_malformed_support(
    helper_name: str,
) -> None:
    from vibeqc import _vibeqc_core as core

    system, basis = _h2_input()
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0
    template = vq.compute_overlap_lattice(basis, system, lat_opts)
    result = _b_result_with_convention(_convention())
    helper = getattr(vq, helper_name)
    zero_blocks = [
        np.zeros((basis.nbasis, basis.nbasis)) for _ in template.cells
    ]

    wrong_nbf = core.make_lattice_matrix_set(
        basis.nbasis + 1,
        list(template.cells),
        zero_blocks,
    )
    with pytest.raises(ValueError, match="density AO dimension"):
        helper(
            system,
            basis,
            result,
            wrong_nbf,
            lattice_options=lat_opts,
        )

    missing_blocks = core.make_lattice_matrix_set(
        basis.nbasis,
        list(template.cells),
        [],
    )
    with pytest.raises(ValueError, match="blocks to align"):
        helper(
            system,
            basis,
            result,
            missing_blocks,
            lattice_options=lat_opts,
        )

    wrong_shape = core.make_lattice_matrix_set(
        basis.nbasis,
        list(template.cells),
        [np.zeros((1, 1)) for _ in template.cells],
    )
    with pytest.raises(ValueError, match="density block shape"):
        helper(
            system,
            basis,
            result,
            wrong_shape,
            lattice_options=lat_opts,
        )

    wide_opts = vq.LatticeSumOptions()
    wide_opts.cutoff_bohr = 16.0
    wide_opts.nuclear_cutoff_bohr = 16.0
    wide_density = vq.compute_overlap_lattice(basis, system, wide_opts)
    assert len(wide_density.cells) > len(template.cells)
    with pytest.raises(ValueError, match="fixed-density cell list"):
        helper(
            system,
            basis,
            result,
            wide_density,
            lattice_options=lat_opts,
        )

    shifted_lattice = 0.99 * np.asarray(system.lattice, dtype=float)
    shifted_atoms = [vq.Atom(atom.Z, list(atom.xyz)) for atom in system.unit_cell]
    shifted_system = vq.PeriodicSystem(3, shifted_lattice, shifted_atoms)
    shifted_basis = vq.BasisSet(shifted_system.unit_cell_molecule(), "sto-3g")
    shifted_density = vq.compute_overlap_lattice(
        shifted_basis,
        shifted_system,
        wide_opts,
    )
    assert [tuple(cell.index) for cell in shifted_density.cells] == [
        tuple(cell.index) for cell in wide_density.cells
    ]
    with pytest.raises(ValueError, match="Cartesian translations"):
        helper(
            system,
            basis,
            result,
            shifted_density,
            lattice_options=wide_opts,
        )


@pytest.mark.parametrize(
    "helper_name",
    [
        "compute_aiccm2026dev_b_fixed_density_kinetic_energy",
        "compute_aiccm2026dev_b_fixed_density_kinetic_gradient",
        "compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian",
        "compute_aiccm2026dev_b_fixed_energy_weighted_overlap_gradient",
    ],
)
def test_b_fixed_one_electron_component_rejects_lowdim(helper_name: str) -> None:
    convention = AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0",
        exchange_q0="bvk-ewald",
        exchange_q0_applicability="active",
        boundary_model="3d-periodic-vacuum",
        periodic_dimension=1,
        character_mesh_shape=(2, 1, 1),
        bvk_madelung_supercell_repetitions=(2, 1, 1),
        bvk_madelung_supercell_lattice_bohr=(
            (20.0, 0.0, 0.0),
            (0.0, 40.0, 0.0),
            (0.0, 0.0, 40.0),
        ),
    )
    system = vq.PeriodicSystem(
        1,
        [[10.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    density = vq.compute_overlap_lattice(basis, system, lat_opts)

    with pytest.raises(NotImplementedError, match="3D periodic"):
        getattr(vq, helper_name)(
            system,
            basis,
            _b_result_with_convention(convention),
            density,
            lattice_options=lat_opts,
        )


def test_b_ewald_nuclear_gradient_component_rejects_lowdim() -> None:
    convention = AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0",
        exchange_q0="bvk-ewald",
        exchange_q0_applicability="active",
        boundary_model="3d-periodic-vacuum",
        periodic_dimension=1,
        character_mesh_shape=(2, 1, 1),
        bvk_madelung_supercell_repetitions=(2, 1, 1),
        bvk_madelung_supercell_lattice_bohr=(
            (16.0, 0.0, 0.0),
            (0.0, 8.0, 0.0),
            (0.0, 0.0, 8.0),
        ),
    )
    system = vq.PeriodicSystem(
        1,
        [[10.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])],
    )

    with pytest.raises(NotImplementedError, match="3D periodic"):
        vq.compute_aiccm2026dev_b_ewald_nuclear_gradient(
            system,
            _b_result_with_convention(convention),
        )


def test_periodic_runner_blocks_b_geometry_optimization(tmp_path) -> None:
    system, basis = _h2_input()

    with pytest.raises(NotImplementedError, match="geometry optimization"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="aiccm2026dev-b",
            output=tmp_path / "b-opt",
            kpoints=(1, 1, 1),
            optimize=True,
            progress=False,
        )


def test_periodic_runner_blocks_b_hessian(tmp_path) -> None:
    system, basis = _h2_input()

    with pytest.raises(NotImplementedError, match="force constants"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="aiccm2026dev-b",
            output=tmp_path / "b-hessian",
            kpoints=(1, 1, 1),
            hessian=True,
            progress=False,
        )
